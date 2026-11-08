"""The RLM's REPL and prompt assembly."""

import io
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import types
import warnings

import pytest

from codetwine_loom import config, dispatch, plan, progress, testcreate
from codetwine_loom import llm as llm_mod
from codetwine_loom import rlm as rlm_mod
from codetwine_loom.llm import PromptTooLong
from codetwine_loom.tools import explore
from codetwine_loom.rlm import (cut, CUT_NOTE, one_line, run_stage, CompleteCall, EMPTY_REPLY_NOTE,
                                   EMPTY_RETRY_NOTE, NO_CODE_MARK, RLM, Repl,
                                   CUT_REPLY_NOTE,
                                   CUT_RETRY_NOTE, build_prompt,
                                   build_prompt_block_list, extract_code,
                                   fence_of, fold_by_count,
                                   fold_history,
                                   late_answer, need_late_answer, no_answer_reason,
                                   prompt_of, prompt_text, tool_line,
                                   NO_OUTPUT_NOTE, TOOL_REFUSAL_NOTE)


def sample_tool(x: str) -> str:
    """One-line tool description."""
    return x


# == Repl ============================
def test_repl_returns_what_was_printed():
    repl = Repl({"project_data": {"files": [1, 2, 3]}}, {})
    out = repl.run("n = len(project_data['files'])\nprint('files', n)", 100)
    assert out == "files 3\n"


def test_repl_namespace_survives_between_calls():
    repl = Repl({"project_data": {"files": [1, 2, 3]}}, {})
    repl.run("n = len(project_data['files'])", 100)
    assert repl.run("print('n is', n)", 100) == "n is 3\n"


def test_repl_returns_both_a_tools_print_and_its_return_value():
    def noisy(x):
        print("tool says", x)
        return x * 2

    repl = Repl({}, {"noisy": noisy})
    out = repl.run("print(noisy(3))", 100)
    assert "tool says 3" in out and "6" in out


def test_repl_turns_an_exception_into_the_last_3_traceback_lines():
    out = Repl({}, {}).run("1/0", 500)
    assert out.strip().endswith("ZeroDivisionError: division by zero")
    assert len(out.splitlines()) == 3


def test_repl_returns_what_the_block_wrote_to_standard_error_in_order(capsys):
    code = ("import sys, traceback\n"
            "print('one')\n"
            "print('two', file=sys.stderr)\n"
            "try:\n"
            "    1/0\n"
            "except ZeroDivisionError:\n"
            "    traceback.print_exc()\n"
            "print('three')")
    out = Repl({}, {}).run(code, 4000)
    assert out.index("one") < out.index("two") < out.index("ZeroDivisionError") \
        < out.index("three")
    assert capsys.readouterr().err == ""


def test_repl_keeps_what_was_printed_before_an_exception():
    """What a tool printed before an exception is kept in front of the traceback."""
    out = Repl({}, {}).run("print('guidance')\n{}['code']", 500)
    assert "guidance" in out
    assert out.strip().endswith("KeyError: 'code'")


def test_repl_keeps_the_traceback_and_cuts_the_output_side():
    out = Repl({}, {}).run("print('x' * 500)\n1/0", 200)
    assert out.strip().endswith("ZeroDivisionError: division by zero")
    assert "chars omitted" in out


# The cut markers a turn's output may carry on top of max_char: one per part at most
_MARKER_ROOM = 200


def test_a_traceback_longer_than_the_limit_is_cut_line_by_line():
    out = Repl({}, {}).run("src = 'x' * 200000\nassert src.startswith('y'), src", 4000)
    assert len(out) <= 4000 + _MARKER_ROOM
    assert "AssertionError: xxx" in out


def test_the_refusal_notes_share_the_limit_with_the_output():
    def run_it(what):
        return {"ok": False, "returncode": 1, "output": "x" * 40000, "error": ""}
    code = ("res = [run_it(f'case{i}') for i in range(5)]\n"
            "print('passed:', sum(r['ok'] for r in res))\n")
    out = Repl({}, {"run_it": run_it}).run(code, 4000)
    assert len(out) <= 4000 + 6 * _MARKER_ROOM
    assert out.startswith("passed: 0\n")
    assert out.count("note: run_it() returned") == 5


def test_a_short_part_is_kept_whole_and_the_long_ones_share_the_rest():
    part_list = rlm_mod._share_budget(["ab", "x" * 10, "y" * 10], 12)
    assert part_list == ["ab", cut("x" * 10, 5), cut("y" * 10, 5)]


def test_repl_returns_guidance_when_nothing_was_printed():
    assert Repl({}, {}).run("y = 1", 100) == NO_OUTPUT_NOTE


def test_repl_cuts_output_over_the_limit():
    out = Repl({}, {}).run("print('x' * 20)", 10)
    assert out == "x" * 10 + "\n... (11 chars omitted)"


# == Repl: a tool's refusal the turn did not print =========
def refuse(path: str) -> dict:
    """A tool that refuses every path."""
    return {"ok": False, "error": f"no such path: {path}"}


def test_a_refusal_the_turn_did_not_print_is_noted_after_the_output():
    repl = Repl({}, {"refuse": refuse})
    out = repl.run("r = refuse('x')\nprint(r['ok'])", 500)
    assert out == "False\n" + TOOL_REFUSAL_NOTE.format(
        tool="refuse", result='{"error": "no such path: x"}')


def test_a_refusal_the_turn_printed_gets_no_note():
    out = Repl({}, {"refuse": refuse}).run("print(refuse('x')['error'])", 500)
    assert out == "no such path: x\n"


def test_a_refusal_is_noted_when_nothing_else_was_printed():
    out = Repl({}, {"refuse": refuse}).run("r = refuse('x')", 500)
    assert out.startswith("note: refuse() returned")
    assert NO_OUTPUT_NOTE not in out


def test_a_tool_that_returned_ok_gets_no_note():
    def accept(path):
        return {"ok": True, "error": ""}
    assert Repl({}, {"accept": accept}).run("r = accept('x')\nprint(1)", 500) == "1\n"


def test_a_tool_that_returns_no_dict_gets_no_note():
    repl = Repl({}, {"sample_tool": sample_tool})
    assert repl.run("r = sample_tool('x')\nprint(1)", 500) == "1\n"


def test_a_refusal_without_an_error_key_is_noted_whole():
    def refuse_bare(what):
        return {"ok": False, "recorded": 2}
    out = Repl({}, {"refuse_bare": refuse_bare}).run("r = refuse_bare('x')", 500)
    assert out.endswith('{"recorded": 2}')


def test_a_refusal_printed_whole_gets_no_note():
    def refuse_bare(what):
        return {"ok": False, "recorded": 2}
    out = Repl({}, {"refuse_bare": refuse_bare}).run("print(refuse_bare('x'))", 500)
    assert out == "{'ok': False, 'recorded': 2}\n"


def test_a_program_that_failed_and_whose_output_was_printed_gets_no_note():
    def run_it(what):
        return {"ok": False, "returncode": 1, "output": "boom\n", "error": ""}
    repl = Repl({}, {"run_it": run_it})
    assert repl.run("print(run_it('x')['output'])", 500) == "boom\n\n"
    assert "note:" in repl.run("print(run_it('x')['returncode'])", 500)


def test_a_return_value_printed_whole_counts_as_printed_however_it_reads():
    """print(dict) writes the output text with its newlines escaped."""
    def run_it(what):
        return {"ok": False, "returncode": 1, "output": "one\ntwo\n", "error": ""}
    assert "note:" not in Repl({}, {"run_it": run_it}).run("print(run_it('x'))", 500)


def test_a_refusal_is_noted_when_the_turn_then_raised():
    out = Repl({}, {"refuse": refuse}).run("r = refuse('x')\n1/0", 500)
    assert "ZeroDivisionError: division by zero" in out
    assert out.endswith(TOOL_REFUSAL_NOTE.format(tool="refuse",
                                                 result='{"error": "no such path: x"}'))


def test_a_refusal_is_noted_on_the_turn_that_called_complete():
    repl = Repl({}, {"refuse": refuse})
    with pytest.raises(CompleteCall) as call:
        repl.run("r = refuse('x')\nCOMPLETE('done')", 500)
    assert call.value.output.endswith('{"error": "no such path: x"}')


def test_a_refusal_is_forgotten_after_its_turn():
    repl = Repl({}, {"refuse": refuse})
    repl.run("r = refuse('x')", 500)
    assert repl.run("print(1)", 500) == "1\n"


def test_the_refusal_note_is_cut_to_the_limit():
    def refuse_long(path):
        return {"ok": False, "error": "e" * 300}
    out = Repl({}, {"refuse_long": refuse_long}).run("r = refuse_long('x')", 80)
    assert "chars omitted" in out and len(out) < 200


def test_the_wrapped_tool_keeps_its_line_in_the_prompt():
    repl = Repl({}, {"sample_tool": sample_tool})
    assert tool_line(repl.namespace_dict["sample_tool"]) == tool_line(sample_tool)


def test_the_repl_answers_to_exactly_the_calls_that_end_a_run():
    """The name of an ending call is decided in END_CALL_DICT and nowhere else.

    A name written straight into the namespace is one the instructions, the refusals and
    the harvest's parser would each have to be told about separately.
    """
    names = {name for name in Repl({}, {}).namespace_dict if name.isupper()}
    assert names == set(rlm_mod.END_CALL_DICT.values())


def test_complete_leaves_through_done():
    with pytest.raises(CompleteCall) as e:
        Repl({}, {}).run("COMPLETE('fixed it')", 100)
    assert e.value.answer == "fixed it"


def test_complete_is_not_swallowed_by_the_models_try_except_exception():
    """Making Done derive from Exception would keep the loop running to its limit."""
    with pytest.raises(CompleteCall):
        Repl({}, {}).run("try:\n    COMPLETE('x')\nexcept Exception:\n    pass", 100)


def _fence_list(reply: str) -> list[tuple[str, str]] | None:
    """Return the fenced blocks the code of a reply not cut off is chosen from.

    None when the reply's parts cannot be told apart (_run_part_list).
    """
    part_list = rlm_mod._run_part_list(reply, list(rlm_mod._block_iter(reply)), False)
    return None if part_list is None else [(p.label, p.text) for p in part_list if p.is_fence]


def test_code_block_takes_the_last_block_of_a_reply_with_prose():
    reply = ("First we do this.\n```python\nprint(0)\n```\n"
             "Then this.\n```py\nprint(1)\n```")
    assert _fence_list(reply) == [("python", "print(0)\n"), ("py", "print(1)\n")]
    assert extract_code(reply) == ("print(1)\n", False)


def test_code_block_returns_an_empty_list_without_a_code_block():
    assert _fence_list("I will not write any code.") == []


def test_a_fence_line_inside_a_string_the_block_writes_ends_the_block():
    """A document with a code block in it is written from one Python string."""
    reply = ('```python\ntext = """# Flow\n```mermaid\nflowchart TD\n    A --> B\n```\n"""\n'
             'print(create_file("docs/flow.md", text))\n```')
    assert extract_code(reply) == ('text = """# Flow\n```mermaid\nflowchart TD\n    A --> B\n',
                                   False)
    assert len(_fence_list(reply)) == 1
    # A longer fence around the block keeps the document whole
    longer = reply.replace("```python", "````python").removesuffix("```") + "````"
    assert extract_code(longer)[0].endswith('print(create_file("docs/flow.md", text))\n')


def test_a_fence_inside_a_comment_does_not_close_the_block():
    reply = '```python\n# the document keeps its ``` fences\nx = 1\nprint(x)\n```'
    assert extract_code(reply)[0] == "# the document keeps its ``` fences\nx = 1\nprint(x)\n"


def test_a_python_block_that_does_not_compile_ends_the_list():
    """A syntax error comes back as the code that holds it."""
    reply = "```python\nprint(1\n```\nsome prose\n```python\nprint(2)\n```"
    assert _fence_list(reply) == [("python", "print(1\n")]
    assert extract_code("```python\nprint(0)\n```\n" + reply, True) == ("print(1\n", False)


def test_a_block_with_no_label_that_does_not_compile_is_text():
    reply = "It reads:\n```\nsome output, not code\n```\n```python\nprint(2)\n```"
    assert _fence_list(reply) == [("", "some output, not code\n"), ("python", "print(2)\n")]
    reply = "```\ns = '''x\n```\n```python\nprint(2)\n```"
    assert _fence_list(reply) == [("", "s = '''x\n"), ("python", "print(2)\n")]
    reply = "It printed:\n```\nFAILED - SyntaxError\n```\nIt opens `\"\"\"` and never closes it.\n" \
            "```python\nprint(2)\n```"
    assert extract_code(reply) == ("print(2)\n", False)
    assert extract_code("It printed:\n```\nsome output, not code\n```") == (None, False)
    assert extract_code("```\nprint(1)\n```\n```\nsome output\n```") == ("print(1)\n", False)
    # An output that parses but calls nothing is no code either
    assert extract_code("```\nx = 1\n```") == (None, False)
    assert extract_code("It printed:\n```\n3\n```\n" + _CALL) == ("print(1)", True)


_DOC_EXAMPLE = "import shutil\nshutil.rmtree('build')\n"


@pytest.mark.parametrize("label", ["python", ""])
@pytest.mark.parametrize("document", [
    # The fence is named in the prose before the example
    f"# D\n\nWrap code in ``` fences.\n\n```python\n{_DOC_EXAMPLE}```\n\nThen ",
    # A block of another language comes before the example
    f"# D\n\n```sh\nmake\n```\n\n```python\n{_DOC_EXAMPLE}```\n\nThen ",
    # The example is nested in a fence of four backticks
    f"# D\n\n````markdown\n```python\n{_DOC_EXAMPLE}```\n````\n\nThen ",
    f"# D\n\n````\n```python\n{_DOC_EXAMPLE}```\n````\n\nThen ",
])
@pytest.mark.parametrize("head", ["", "print('Here's the plan')\n", "x = = 1\n",
                                  'doc.append("```sh")\n'])
def test_an_example_inside_a_document_a_block_writes_is_not_run(label, document, head):
    """Whether the reply is cut there or the block is written out to its end."""
    code = f"{head}create_file('d.md', '''{document}"
    for is_cut, reply in ((True, f"```{label}\n{code}"),
                          (False, f"```{label}\n{code}''')\n```\n"),
                          (False, f"Last turn I ran:\n```python\nprint(1)\n```\n```{label}\n"
                                  f"{code}''')\n```\n")):
        run = extract_code(reply, is_cut)[0]
        assert run is None or "shutil" not in run or not rlm_mod._parse_ok(run)


def test_the_code_in_front_of_a_comment_that_names_a_fence_is_not_run_alone():
    front = "print('front')  # the ```sh blocks below are examples\n"
    reply = f"```python\n{front}create_file('d.md', '''# D\n\n```sh\nmake\n```\n\nmore "
    assert extract_code(reply)[0] == f"{front}create_file('d.md', '''# D\n\n```sh\nmake\n"
    assert not rlm_mod._parse_ok(extract_code(reply, True)[0])
    # A fence after a comment closes nothing
    assert extract_code("```python\nprint(1)  # ```\nprint(2)\n")[0] is None


def test_a_heading_inside_a_block_does_not_end_the_reply():
    reply = ("```python\nprint('front')  # see the ``` fence in the rules\n"
             "create_file('RULES.md', '''# Rules\n\n## Turn 1\nDo one step.\n''')\n```\n")
    assert rlm_mod.reply_end(reply) is None
    assert rlm_mod.reply_end(reply + "### Turn 2 output\n") == len(reply) - 1
    assert rlm_mod.reply_end("```python\nprint(1)  # ```\nx = 2\n## Turn 1\n") is None
    assert rlm_mod.reply_end("```python\nx = 2\n```\n## Turn 1\n") == 19


def _parse_count(monkeypatch) -> list:
    """Count the texts handed to the parser."""
    seen_list = []
    parse_tree = rlm_mod.parse_tree
    monkeypatch.setattr(rlm_mod, "parse_tree",
                        lambda code: (seen_list.append(len(code)), parse_tree(code))[1])
    return seen_list


def test_each_python_block_is_parsed_once(monkeypatch):
    seen_list = _parse_count(monkeypatch)
    reply = "```python\nprint(1)\n```\nprose\n```\n1\n```\n" * 200
    assert len(_fence_list(reply)) == 400
    assert len(seen_list) == 200 and max(seen_list) < 20


def test_a_reply_is_split_into_blocks_and_headings_without_the_parser(monkeypatch):
    seen_list = _parse_count(monkeypatch)
    document = "## Turn 1\n```text\nmove a pawn\n```\n" * 200
    reply = f"```python\ncreate_file('doc/rules.md', '''# Rules\n{document}''')\n```\n"
    end = rlm_mod.reply_end(reply)
    assert end == reply.index("\n## Turn 1", reply.index("move a pawn"))
    other = "```python\n" + "x = 1  # ```\n" * 100 + "s = '''\n```\nt = \"\"\"\n" + "## Turn 1\n" * 100
    assert rlm_mod.reply_end(other) == other.index("\n## Turn 1")
    assert seen_list == []
    assert extract_code(reply[:end]) == extract_code(reply)


def test_a_fence_in_a_comment_after_code_that_parses_leaves_the_block_open():
    reply = "```python\nx = 1  # ```\ny = 2  # and ```\nprint(x + y)\n```\n"
    assert _fence_list(reply) == [
        ("python", "x = 1  # ```\ny = 2  # and ```\nprint(x + y)\n")]


@pytest.mark.parametrize("head", ["", "print('front')  # the ```sh blocks are examples\n"])
def test_fences_inside_one_line_strings_do_not_end_the_block(head):
    code = head + "".join('part_list.append("```")\n' for _ in range(200)) + "print(1)\n"
    reply = f"```python\n{code}```\n"
    assert _fence_list(reply) == [("python", code)]
    assert extract_code(reply, True) == (code, False)
    example = "create_file('u.md', '''# U\n\n```python\nimport shutil\n```\n''')\n"
    reply = f"```python\n{code}{example}```\n"
    assert extract_code(reply)[0] == code + "create_file('u.md', '''# U\n\n```python\nimport shutil\n"


@pytest.mark.parametrize("odd", ["\x00", "\ud83d", "(" * 100000, "\u2028", "\x0c"])
@pytest.mark.parametrize("reply", [
    "```python\nif x:\n    y = 1\n{odd}\n## Turn 1 output\n",
    "```python\nx = 1\n# note {odd}\nprint(x)\n```\n### Turn 2\n",
    "```python\ns = '''a\n{odd}\n## Turn 1\n```text\nb\n```\n",
    '<invoke name="python">\n<parameter name="code">s = """\n{odd}\n## Turn 1\n',
    "see {odd} here\n```python\nprint(1)\n```\n"])
def test_a_reply_no_parser_reads_is_read_as_one_that_does_not_compile(reply, odd):
    """Whatever a reply holds, taking its code out raises nothing."""
    reply = reply.replace("{odd}", odd)
    _fence_list(reply)
    extract_code(reply)
    extract_code(reply, True)
    rlm_mod.reply_end(reply)


def test_a_reply_with_an_invalid_escape_is_read_with_no_warning(capsys):
    code = "import re\nprint(re.findall('\\d+\\.\\d+', s))\n"
    reply = f"```python\n{code}```\n### Turn 2 output\n"
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        assert extract_code(reply) == (code, False)
        assert rlm_mod.reply_end(reply) == len(reply) - len("\n### Turn 2 output\n")
        assert rlm_mod.tool_call_text(code, ["print"]) == "print"
        assert rlm_mod._in_string_or_comment(code + "s = '''\\d\n")
    assert capsys.readouterr().err == ""


def test_a_text_with_a_nul_ends_inside_no_string():
    assert rlm_mod._in_string_or_comment('s = ""\n s\n\x00') is False
    assert rlm_mod.reply_end("```python\nif x:\n    y = 1\n\x00\n```\n## Turn 1 output\n") == 31


def test_a_block_opened_with_four_backticks_closes_with_four():
    reply = "````python\nprint('```')\n```\n````"
    assert _fence_list(reply) == [("python", "print('```')\n```\n")]


@pytest.mark.parametrize("reply, block_list", [
    # Up to three spaces before a fence, which the text loses as well
    ("1. Run:\n   ```python\n   print(1)\n   ```\n", [("python", "print(1)\n")]),
    ("  ```python\nprint(1)\n```", [("python", "print(1)\n")]),
    # Four spaces make no fence
    ("    ```python\n    print(1)\n    ```\n", []),
    # Tildes, and a closing fence longer than the opening one
    ("~~~python\nprint('```')\n~~~\n", [("python", "print('```')\n")]),
    ("```python\nprint(1)\n````\n", [("python", "print(1)\n")]),
    # The label is the first word of the info string
    ("```python title=a.py\nprint(1)\n```", [("python", "print(1)\n")]),
    # A fence in the middle of a line opens and closes nothing, and the block the next
    # fence opens is not closed
    ("Run ```python\nprint(1)\n```\n", None),
    ("```python\nprint(1)```\n", None),
    # A fence with an info string inside a block that is not Python leaves the reply unsplit
    ("```\nREADME:\n```sh\nmake\n```\n```python\nprint(2)\n```\n", None),
    ("```\nx\n````\n```python\nprint(2)\n```\n", None),
    ("```python\nprint(1)\n```\n```markdown\n```python\nx\n```\n```", None),
])
def test_blocks_are_split_by_the_fence_lines_alone(reply, block_list):
    assert _fence_list(reply) == block_list


def test_a_block_with_no_closing_fence_is_no_block():
    assert _fence_list("```python\nprint(1)\n") is None
    reply = "```python\nprint(1)\n"
    assert rlm_mod._run_part_list(reply, list(rlm_mod._block_iter(reply)), True) == []


def test_a_python_block_is_run_before_a_later_block_with_no_label():
    reply = "```python\nprint(1)\n```\nIt prints:\n```\n1\n```"
    assert extract_code(reply) == ("print(1)\n", False)


def test_a_block_in_another_language_is_skipped_whole():
    reply = '```json\n{"a": 1}\n```\nThen:\n```python\nprint(1)\n```'
    assert _fence_list(reply) == [("json", '{"a": 1}\n'), ("python", "print(1)\n")]
    assert extract_code(reply) == ("print(1)\n", False)
    assert extract_code('```json\n{"a": 1}\n```') == (None, False)


def test_a_block_with_no_label_runs_when_there_is_no_python_block():
    assert extract_code("```\nprint(1)\n```") == ("print(1)\n", False)


@pytest.mark.parametrize("label", ["python3", "ipython", "Python3"])
def test_a_block_labelled_as_another_name_of_python_runs(label):
    assert extract_code(f"```{label}\nprint(1)\n```") == ("print(1)\n", False)
    reply = f"```{label}\nprint(1)\n```\nIt prints:\n```\n1\n```"
    assert extract_code(reply) == ("print(1)\n", False)


def test_a_fence_that_ends_its_line_with_crlf_opens_a_block():
    assert extract_code("```python\r\nprint(1)\r\n```") == ("print(1)\r\n", False)


def test_tool_line_joins_the_signature_and_the_one_line_description():
    assert tool_line(sample_tool) == \
        "- sample_tool(x: str) -> str — One-line tool description."


# == build_prompt ====================================
def test_fence_of_is_one_longer_than_the_longest_run_inside():
    assert fence_of("print(1)") == "```"
    assert fence_of("a ``` b") == "````"
    assert fence_of("a ````` b ``` c") == "``````"


def test_build_prompt_order():
    """## Turn i/n goes after the history."""
    p = build_prompt("INSTR", {"sample_tool": sample_tool}, "VARS", "REQ",
                     [("code1", "out1")], 2, 20)
    order = [p.index(s) for s in ("INSTR", "## Variables", "## Tools",
                                  "## Request", "### Turn 1 code",
                                  "### Turn 1 output", "## Turn 2/20")]
    assert order == sorted(order)


def test_build_prompt_fences_code_and_output_longer_than_any_run_inside():
    p = build_prompt("I", {}, "V", "R", [("a ``` b", "c ```` d")], 2, 20)
    body = p.split("### Turn 1 code", 1)[1].split("## Turn 2/20", 1)[0]
    assert "\n````python\na ``` b\n````\n" in body
    assert "\n`````\nc ```` d\n`````\n" in body
    # A history without a backtick keeps the three-backtick fences
    p = build_prompt("I", {}, "V", "R", [("print(1)", "1")], 2, 20)
    assert "\n```python\nprint(1)\n```\n" in p


def test_build_prompt_includes_every_turn_of_the_history():
    p = build_prompt("I", {}, "V", "R",
                     [("c1", "o1"), ("c2", "o2"), ("c3", "o3")], 4, 20)
    assert all(f"### Turn {j} code" in p for j in (1, 2, 3))


def test_the_blocks_joined_are_the_prompt():
    kwargs = dict(instruction="INSTR", tool_dict={"sample_tool": sample_tool},
                  var_doc="VARS", request="REQ",
                  history_list=[("c1", "o1"), ("c2", "o2")], turn=3, max_turn=20,
                  turn_note="NOTE")
    assert prompt_text(build_prompt_block_list(**kwargs)) == build_prompt(**kwargs)
    kwargs["history_list"], kwargs["turn_note"] = [], ""
    assert prompt_text(build_prompt_block_list(**kwargs)) == build_prompt(**kwargs)


def test_a_cache_point_follows_the_head_and_the_newest_turn():
    block_list = build_prompt_block_list("I", {}, "V", "R", [("c1", "o1"), ("c2", "o2")],
                                         3, 20, turn_note="NOTE")
    assert [is_cache_point for _, is_cache_point in block_list] == [True, False, True, False]
    assert block_list[0][0].startswith("I\n") and "## Request\nR\n" in block_list[0][0]
    assert block_list[1][0].startswith("\n### Turn 1 code") and "### Turn 2" not in block_list[1][0]
    assert block_list[3][0] == "\nNOTE\n\n## Turn 3/20\nWrite the next ```python block."
    assert [x for _, x in build_prompt_block_list("I", {}, "V", "R", [], 1, 20)] == [True, False]


def test_what_the_llm_is_sent_follows_the_cache_setting(monkeypatch):
    block_list = build_prompt_block_list("I", {}, "V", "R", [("c1", "o1")], 2, 20)
    monkeypatch.setattr(config, "PROMPT_CACHE", False)
    assert prompt_of(block_list, "NOTE") == prompt_text(block_list) + "NOTE"
    monkeypatch.setattr(config, "PROMPT_CACHE", True)
    sent = prompt_of(block_list, "NOTE")
    assert sent[:-1] == block_list[:-1]
    assert sent[-1] == (block_list[-1][0] + "NOTE", False)
    assert prompt_of(block_list) is block_list


def test_run_sends_the_blocks_when_the_cache_is_on(monkeypatch):
    monkeypatch.setattr(config, "PROMPT_CACHE", True)
    llm = _ScriptedLLM(["```python\nprint('x')\n```", "```python\nCOMPLETE('d')\n```"])
    RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert isinstance(llm.prompts[1], list) and llm.prompts[1][0][1] is True
    assert "### Turn 1 code" in _sent_text(llm.prompts[1])


def test_the_turn_mark_matches_the_history_headings_and_no_other_heading():
    p = build_prompt("I", {}, "V", "R", [("c1", "o1")], 2, 20)
    assert [m.group().strip() for m in rlm_mod.TURN_MARK_RE.finditer(p)] == \
        ["### Turn 1", "### Turn 1", "## Turn 2"]
    assert not rlm_mod.TURN_MARK_RE.search("x\n## Turnaround time\n### Turn on the flag\n")


_CALL = '<invoke name="python">\n<parameter name="code">print(1)</parameter>\n</invoke>'
_FENCE_2 = "```python\nprint(2)\n```"

# Replies in every form extract_code() reads, each written on past its first code
_LONG_REPLY_DICT = {
    "calls": _CALL + '\n\n<invoke name="python">\n<parameter name="code">pass</parameter>\n'
             '</invoke>\n\n' * 3,
    "call then closing tags": _CALL + "\n\n</parameter>\n</invoke>\n\n</parameter>\n</invoke>\n",
    "call then text then call": _CALL + "\n(no output)\n" + _CALL,
    "call then fence": "  \n" + _CALL + "\n" + _FENCE_2,
    "call with no code then fence": '<invoke name="python">\n</invoke>\n' + _FENCE_2 + "\n",
    "call with empty code then fence":
        '<invoke name="python">\n<parameter name="code"></parameter>\n</invoke>\n\n' + _FENCE_2,
    "other call then fence then heading":
        '<invoke name="x">\n</invoke>\n' + _FENCE_2 + "\n### Turn 3 output\n```\n2\n```",
    "closing tag in the code":
        '<invoke name="python">\n<parameter name="code">s = "</parameter>"\nprint(s)'
        '</parameter>\n</invoke>\n\n' + _CALL,
    "parameter before the code":
        '<invoke name="python">\n<parameter name="timeout">5</parameter>\n'
        '<parameter name="code">print(1)</parameter>\n</invoke>\n' + _CALL,
    "fences then heading": "```python\nprint(1)\n```\n" + _FENCE_2 + "\n### Turn 2 output\n```\n2\n```\n",
    "fence then next turn": "```python\nprint(1)\n```\n\n## Turn 3/20\nWrite the next",
    "unfenced code then heading": "print(1)\n### Turn 2 output\n```\n1\n```",
    "text then calls then fence": "I will look first.\n" + _CALL + "\n" + _CALL + "\n" + _FENCE_2,
    "calls of a tool": '<invoke name="double">\n<parameter name="value">21</parameter>\n</invoke>\n' * 3,
    "parameter then call": '<parameter name="python">print(1)</parameter>\n' + _CALL,
    "heading first": "\n### Turn 2 code\n```python\nprint(1)\n```",
    "words": "No code here, only words.",
    "colon then calls": ":\n" + _CALL + "\n:\n" + _CALL + "\n:\n" + _CALL,
    "fence then calls": _FENCE_2 + "\n" + _CALL + "\n" + _CALL,
    "unfenced code holding a call in a string":
        'write_file("doc.md", """\nexample:\n' + _CALL + '\n""")\nprint("ok")',
    "unfenced code holding a call on one line":
        "text = '" + _CALL.replace("\n", "") + "'\nprint(text)",
    "fenced code holding a heading that is not the history's":
        '```python\nwrite_file("a.md", """x\n## Turnaround\n### Turn on\n""")\n```',
    "fenced code holding the history's heading in a string":
        '```python\nwrite_file("rules.md", """# Rules\n\n## Turn 1\nDraw a card.\n""")\n'
        'print("ok")\n```\n### Turn 2 output\n```\nok\n```',
    "call holding the history's heading in a string":
        '<invoke name="python">\n<parameter name="code">s = """x\n### Turn 3 notes\n"""\n'
        'print(s)</parameter>\n</invoke>\n\n' + _CALL,
    "calls whose code does not compile":
        '<invoke name="python">\n<parameter name="code">print(1\n</parameter>\n</invoke>\n\n'
        * 3,
    "call whose code holds the closing tags in a string":
        '<invoke name="python">\n<parameter name="code">s = """</parameter>\n</invoke>\n"""\n'
        'print(s)</parameter>\n</invoke>\n\n' + _CALL,
}


@pytest.mark.parametrize("name", _LONG_REPLY_DICT)
def test_a_reply_ended_as_it_arrives_gives_the_code_the_whole_reply_gives(name):
    reply = _LONG_REPLY_DICT[name]
    # A reply that writes a heading is compared up to the heading
    mark_match = rlm_mod._turn_mark(reply)
    whole = reply[:mark_match.start()] if mark_match else reply
    for size in range(1, len(reply) + 1):
        end = rlm_mod.reply_end(reply[:size])
        if end is not None:
            assert extract_code(reply[:end]) == extract_code(whole)
            break


@pytest.mark.parametrize("name, code", [
    ("text then calls then fence", "print(1)"),
    ("colon then calls", "print(1)"),
    ("fence then calls", "print(2)\n"),
    ("call with empty code then fence", "print(2)\n"),
    ("call with no code then fence", "print(2)\n"),
    ("unfenced code holding a call in a string",
     _LONG_REPLY_DICT["unfenced code holding a call in a string"]),
    ("unfenced code holding a call on one line",
     _LONG_REPLY_DICT["unfenced code holding a call on one line"]),
])
def test_a_call_written_before_any_code_is_the_code_and_an_empty_one_is_not(name, code):
    assert extract_code(_LONG_REPLY_DICT[name])[0] == code


@pytest.mark.parametrize("reply, kept", [
    (_CALL, _CALL),
    (_CALL + "\n\n", _CALL),
    (_LONG_REPLY_DICT["calls"], _CALL),
    (_LONG_REPLY_DICT["call then closing tags"], _CALL),
    (_LONG_REPLY_DICT["call then text then call"], _CALL),
    (_LONG_REPLY_DICT["fences then heading"], "```python\nprint(1)\n```\n" + _FENCE_2),
    (_LONG_REPLY_DICT["fence then next turn"], "```python\nprint(1)\n```\n"),
    (_LONG_REPLY_DICT["unfenced code then heading"], "print(1)"),
    (_LONG_REPLY_DICT["text then calls then fence"], "I will look first.\n" + _CALL),
    (_LONG_REPLY_DICT["heading first"], ""),
    (_LONG_REPLY_DICT["colon then calls"], ":\n" + _CALL),
    (_LONG_REPLY_DICT["fenced code holding the history's heading in a string"],
     _LONG_REPLY_DICT["fenced code holding the history's heading in a string"].split(
         "\n### Turn 2 output")[0]),
    (_LONG_REPLY_DICT["call holding the history's heading in a string"],
     _LONG_REPLY_DICT["call holding the history's heading in a string"][:-len(_CALL) - 2]),
    (_LONG_REPLY_DICT["calls whose code does not compile"],
     '<invoke name="python">\n<parameter name="code">print(1\n</parameter>\n</invoke>'),
    # A closing tag the call's closing tag follows ends the code, inside a string too
    (_LONG_REPLY_DICT["call whose code holds the closing tags in a string"],
     '<invoke name="python">\n<parameter name="code">s = """</parameter>\n</invoke>'),
    ('<invoke name="python">\n<parameter name="code">s = """</parameter>\n</invoke>\nmore',
     '<invoke name="python">\n<parameter name="code">s = """</parameter>\n</invoke>'),
    ('<invoke name="python">\n<parameter name="code">s = "</parameter></invoke>',
     '<invoke name="python">\n<parameter name="code">s = "</parameter></invoke>'),
    # A closing tag text follows is no closing tag
    ('<invoke name="python">\n<parameter name="code">x = 1  # the </parameter> tag\n'
     'print(x)</parameter>\n</invoke>\n' + _CALL,
     '<invoke name="python">\n<parameter name="code">x = 1  # the </parameter> tag\n'
     'print(x)</parameter>\n</invoke>'),
    ('<invoke name="python">\n<parameter name="code">print(1)</parameter>\n'
     '<parameter name="timeout">5</parameter>\n</invoke>',
     '<invoke name="python">\n<parameter name="code">print(1)</parameter>'),
])
def test_a_reply_ends_after_its_first_call_or_in_front_of_a_heading(reply, kept):
    assert reply[:rlm_mod.reply_end(reply)] == kept


@pytest.mark.parametrize("front", [
    "x = '\"\"\"'\n",                       # three quotes inside another string
    "y = \"'''\" + '\"\"\"'\n",
])
def test_a_call_after_three_quotes_inside_another_string_is_the_first_call(front):
    reply = front + _CALL + "\n\n" + _CALL
    assert reply[:rlm_mod.reply_end(reply)] == front + _CALL


def test_a_call_inside_a_multi_line_string_is_not_the_first_call():
    front = "doc = '''\nquote: \"\"\"\n"
    assert rlm_mod.reply_end(front + _CALL + "\n'''\n") is None


@pytest.mark.parametrize("reply", [
    _CALL[:-3],
    '<invoke name="python">\n<parameter name="code">print(1',
    '<invoke name="python">\n<parameter name="code">print(1)</parameter>',
    '<invoke name="python">\n<parameter name="code">print(1)</parameter>\n</inv',
    '<invoke name="python">\n<parameter name="code">x = 1  # the </parameter> tag\n',
    '```python\nwrite_file("rules.md", """# Rules\n\n## Turn 1\nDraw a card.\n',
    _LONG_REPLY_DICT["call with no code then fence"],
    _LONG_REPLY_DICT["call with empty code then fence"],
    _LONG_REPLY_DICT["calls of a tool"],
    _LONG_REPLY_DICT["fence then calls"],
    _LONG_REPLY_DICT["unfenced code holding a call in a string"],
    _LONG_REPLY_DICT["unfenced code holding a call on one line"],
    _LONG_REPLY_DICT["fenced code holding a heading that is not the history's"],
    "```python\nprint(1)\n```\n\n\n" + _FENCE_2,
    "### Turn 2 code",
    "\n### Turn ",
    "",
])
def test_a_reply_whose_code_may_still_change_does_not_end(reply):
    assert rlm_mod.reply_end(reply) is None


@pytest.mark.parametrize("code", [
    "x = '\"\"\"'\nprint(x)\n",                       # three quotes inside another string
    "print('''''' + 'a')\nprint(1)\n",
    'print("""a""" + \'b\')\n',
])
def test_a_heading_after_a_closed_block_ends_the_reply_whatever_quotes_it_holds(code):
    reply = "```python\n" + code + "```\n### Turn 2 output\n```\nx\n```"
    assert reply[:rlm_mod.reply_end(reply)] == "```python\n" + code + "```"


@pytest.mark.parametrize("code", [
    'write_file("a.md", f"""# {name}\n',
    "write_file('a.md', '''# Rules\n",
    'write_file("a.md", """```text\nx\n',            # a fence with a label inside the string
])
def test_a_heading_inside_a_string_of_the_code_does_not_end_the_reply(code):
    assert rlm_mod.reply_end("```python\n" + code + "\n## Turn 1\nDraw.\n") is None


@pytest.mark.parametrize("kept", [
    '```python\ns = """abc\n```',
    "```python\nw('a.md', '''```text\nx\n```",
    '<invoke name="python">\n<parameter name="code">s = """abc\n</parameter>\n</invoke>',
])
def test_a_heading_after_a_closed_block_whose_code_leaves_a_string_open_ends_the_reply(kept):
    reply = kept + "\n### Turn 2 output\n```\nx\n```\n" + _FENCE_2
    assert reply[:rlm_mod.reply_end(reply)] == kept
    assert extract_code(kept) == extract_code(kept + "\n")


_DOC_WITH_EXAMPLE = "create_file('d.md', '''# D\n\n```python\nprint('EXAMPLE')\n```\n\nThen "


@pytest.mark.parametrize("front", ['<invoke name="python">\n<parameter name="code">',
                                   '<parameter name="code">'])
@pytest.mark.parametrize("head", ["", "x = 1  # see the </parameter> tag\n"])
def test_the_fences_inside_the_code_of_a_markup_call_are_no_blocks(front, head):
    reply = front + head + _DOC_WITH_EXAMPLE
    for is_cut in (False, True):
        assert extract_code(reply, is_cut) == (None, False)
    whole = reply + "it.\n''')\nprint(1 2)\n</parameter>\n</invoke>\n"
    assert extract_code(whole) == (head + _DOC_WITH_EXAMPLE + "it.\n''')\nprint(1 2)\n", True)


def test_the_fences_inside_the_arguments_of_a_call_of_a_tool_are_no_blocks():
    content = "# D\n\n```python\nprint('EXAMPLE')\n```\n"
    reply = (f'<invoke name="create_file">\n<parameter name="path">d.md</parameter>\n'
             f'<parameter name="content">{content}</parameter>\n</invoke>')
    assert extract_code(reply) == (f"print(create_file(path='d.md', content={content!r}))", True)
    assert extract_code(reply[:-len("</parameter>\n</invoke>")]) == (None, False)


@pytest.mark.parametrize("code", ['create_file("README.md", "# Title\\n")',
                                  'print(read_file("docs/a.md#intro"))',
                                  'x = 1\nprint(f"#{x}")'])
def test_a_closing_tag_on_the_line_of_a_string_that_holds_a_hash_ends_the_code(code):
    reply = f'<invoke name="python">\n<parameter name="code">{code}</parameter>\n</invoke>\n'
    assert extract_code(reply) == (code, True)
    assert extract_code(reply, True) == (code, True)
    assert rlm_mod.reply_end(reply) == len(reply) - 1
    assert extract_code(f'<parameter name="code">{code}</parameter>') == (code, True)


def test_a_block_quoted_before_a_text_block_that_holds_an_open_string_is_not_run():
    quoted = "Last turn I ran:\n```python\nrevert_file('src/area.py')\n```\n"
    fix = "```python\nreplace_in_file('src/area.py', '3.14', 'math.pi')\n```\n"
    for middle in ('The file starts:\n```\ndef area(r):\n    """Return the area.\n```\nFix:\n',
                   'It printed:\n```\nFAILED\n```\nIt opens with `"""` and never closes. Fix:\n'):
        for is_cut in (False, True):
            assert extract_code(quoted + middle + fix, is_cut) == (
                "replace_in_file('src/area.py', '3.14', 'math.pi')\n", False)


_QUOTED_CALL = ('<invoke name="python">\n<parameter name="code">revert_file(\'src/a.py\')'
                '</parameter>\n</invoke>')
_QUOTED_TOOL = ('<invoke name="delete_file">\n<parameter name="path">src/a.py</parameter>\n'
                '</invoke>')


@pytest.mark.parametrize("reply", [
    f"I sent this by mistake:\n```\n{_QUOTED_CALL}\n```\nLet me look at the file first.",
    f"The markup looks like:\n```xml\n{_QUOTED_CALL}\n```\nI'll explain more.",
    f"Do not do this:\n```xml\n{_QUOTED_TOOL}\n```\nInstead I'll read it.",
    f"```markdown\n# Markup\n{_QUOTED_TOOL}\n```\n",
    f"```python\ncreate_file('docs/m.md', '''# Markup\n\n{_QUOTED_CALL}\n\nMore text",
    f"create_file('docs/m.md', '''# Markup\n\n{_QUOTED_CALL}\n\nMore",
    f"x = = 1\ncreate_file('docs/m.md', '''# Markup\n\n{_QUOTED_TOOL}\n''')\n",
    f"x = = 1\ndoc = '{_QUOTED_TOOL.replace(chr(10), '')}'\n",
])
def test_a_call_quoted_in_a_block_or_a_string_is_not_run(reply):
    for is_cut in (False, True):
        assert extract_code(reply, is_cut) == (None, False)


def test_a_call_written_outside_the_blocks_is_run():
    sample = "It printed:\n```\n3\n```\n"
    assert extract_code("I will delete it.\n" + _QUOTED_TOOL) == (
        "print(delete_file(path='src/a.py'))", True)
    assert extract_code(sample + _QUOTED_TOOL, True) == (
        "print(delete_file(path='src/a.py'))", True)
    assert extract_code(sample + _QUOTED_CALL) == ("revert_file('src/a.py')", True)
    # A call with no parameter is passed over
    assert extract_code('No <invoke name="read_file"> here.\n' + _QUOTED_TOOL) == (
        "print(delete_file(path='src/a.py'))", True)
    # The code parameter that starts a reply is read up to its closing tag
    code = 'doc = """Write <invoke name="python"> then </invoke>."""\nprint(doc)\n'
    assert extract_code(f'<parameter name="code">{code}</parameter>') == (code, True)


@pytest.mark.parametrize("front", ["I'll write the doc now.\n", "Let's write it.\n",
                                   "Here's the doc.\nIt isn't long.\n",
                                   "```text\nplan\n```\nNow I'll write it.\n"])
@pytest.mark.parametrize("call", [_QUOTED_CALL, _QUOTED_TOOL])
def test_a_call_quoted_in_unfenced_code_after_prose_is_not_run(front, call):
    reply = f"{front}create_file('docs/m.md', '''# Calls\n\n{call}\n\nmore\n''')\n"
    for is_cut in (False, True):
        assert extract_code(reply, is_cut) == (None, False)
    assert rlm_mod.reply_end(reply) is None


_REAL_BLOCK = "```python\nprint(read_file('src/a.py'))\n```\n"
_LINE_CALL = ('<invoke name="python"><parameter name="code">revert_file(\'src/a.py\')'
              '</parameter></invoke>')
_LINE_TOOL = ('<invoke name="delete_file"><parameter name="path">src/a.py</parameter>'
              '</invoke>')


@pytest.mark.parametrize("quote", [
    f"Last turn I sent `{_LINE_CALL}` by mistake. Now:\n",
    f"Last turn I sent ``{_LINE_CALL}`` by mistake. Now:\n",
    f"Last turn I sent `\n{_QUOTED_CALL}\n` by mistake. Now:\n",
    f"Last turn:\n> {_LINE_CALL}\nNow:\n",
    f"<!-- {_LINE_CALL} -->\n",
    f"<!--\n{_QUOTED_CALL}\n-->\n",
    f"Last turn:\n\n    {_LINE_CALL}\n\nNow:\n",
])
def test_a_call_quoted_in_markdown_is_not_run(quote):
    tool_quote = quote.replace(_LINE_CALL, _LINE_TOOL).replace(_QUOTED_CALL, _QUOTED_TOOL)
    for is_cut in (False, True):
        assert extract_code(quote + _REAL_BLOCK, is_cut) == (
            "print(read_file('src/a.py'))\n", False)
        assert extract_code(tool_quote, is_cut) == (None, False)
    assert rlm_mod.reply_end(quote + _REAL_BLOCK) is None


@pytest.mark.parametrize("front", [
    "I'll read the file first.\n", "I'll read it.\n\n", "I'll do:\nx = 1\n",
    "It's `x` that fails; I'll check `src/a.py`.\n", "> The task: fix it.\n\n",
    "<!-- plan --> I'll read it.\n", "Steps:\n    1. read\n", "Notes:\n\n    text\n\nNow:\n",
    "It printed:\n```\n3\n```\nI'll read it.\n",
])
def test_a_call_written_after_prose_is_run(front):
    assert extract_code(front + _QUOTED_TOOL) == ("print(delete_file(path='src/a.py'))", True)
    assert extract_code(front + _QUOTED_CALL) == ("revert_file('src/a.py')", True)


@pytest.mark.parametrize("front", [
    "Here's the doc: create_file('docs/m.md', '''# Calls\n\n",
    'I\'ll call create_file("docs/m.md", """# Calls\n\n',
    "It's in C:\\\ncreate_file('docs/m.md', '''# Calls\n\n",
    "Plan (draft\ncreate_file('docs/m.md', '''# Calls\n\n",
    "Close it with '''.\ncreate_file('docs/m.md', '''# Calls\n\n",
    "I'll write it.\ncreate_file(\n    'docs/m.md',\n    '''# Calls\n",
    "I'll write it.\ncreate_file('docs/m.md', f'''# Calls\n{\n",
    "I'll write it.\ncreate_file('docs/m.md', f'''# Calls }\n\n",
    "I'll write it.\nx = 'calls: \\\n",
    "Styles:\n- ''' in a.py\n\nHere is the doc:\ncreate_file('docs/m.md', '''# Calls\n\n",
])
def test_a_call_quoted_in_unfenced_code_on_a_line_of_prose_is_not_run(front):
    for is_cut in (False, True):
        assert extract_code(front + _QUOTED_TOOL + "\n''')\n", is_cut) == (None, False)
        assert extract_code(front + _QUOTED_CALL + "\n''')\n" + _REAL_BLOCK, is_cut) == (
            "print(read_file('src/a.py'))\n", False)
    assert rlm_mod.reply_end(front + _QUOTED_CALL + "\n''')\n" + _REAL_BLOCK) is None


@pytest.mark.skipif(sys.version_info < (3, 12), reason="needs f-strings that reuse their quote")
@pytest.mark.parametrize("front", [
    "I'll write it.\ncreate_file('docs/m.md', f\"# {title or \"Untitled doc\"}\" + '''# Calls\n\n",
    "I'll write it.\ncreate_file('docs/m.md', f'{row['first name']}' + '''# Calls\n\n",
    "I'll write it.\ncreate_file('docs/m.md', rf\"{d[\"k k\"]}\" + '''# Calls\n\n",
    "I'll write it.\ncreate_file(\n    'docs/m.md',\n    f\"{a or \"b c\"}\"\n    + '''# Calls\n",
    "I'll write it.\ncreate_file('docs/m.md', f\"{d['k k']}\" + '''# Calls\n\n",
    "I'll write it.\ncreate_file('docs/m.md', f\"{x + '}' + \"a b\"}\" + '''# Calls\n\n",
    "I'll write it.\ncreate_file('docs/m.md', f\"{'''a'}''' + \"b c\"}\" + '''# Calls\n\n",
    "I'll write it.\ndoc = f\"{'''a'}''' + \"b c\"}\" + '''# Calls\n\n",
    "I'll write it.\ncreate_file('docs/m.md', f\"{x if y else f'{'a}'}' + \"b c\"}\" + '''# Calls\n\n",
    "I'll write it.\ncreate_file('docs/m.md', f\"\"\"{x  # }\n+ \"\"\"a b\"\"\"}\"\"\" + '''# Calls\n\n",
    "I'll write it.\ncreate_file('docs/m.md', f\"\"\"{x  # (a}\n+ \"\"\"a b\"\"\"}\"\"\" + '''# Calls\n\n",
])
def test_a_call_quoted_after_an_f_string_that_holds_strings_of_words_is_not_run(front):
    for is_cut in (False, True):
        assert extract_code(front + _QUOTED_TOOL + "\n''')\n", is_cut) == (None, False)
        assert extract_code(front + _QUOTED_CALL + "\n''')\n" + _REAL_BLOCK, is_cut) == (
            "print(read_file('src/a.py'))\n", False)
    assert rlm_mod.reply_end(front + _QUOTED_CALL + "\n''')\n" + _REAL_BLOCK) is None


@pytest.mark.parametrize("code", [
    "create_file('docs/m.md', (\n    {name}\n    if {name}\n    else '''# Calls\n{call}\n'''))\n",
    "create_file('docs/m.md', (\n    {name} if\n    {name}\n    else '''# Calls\n{call}\n'''))\n",
    "create_file('docs/m.md', ({name} if {name}\n    else '''# Calls\n{call}\n'''))\n",
    "create_file('docs/m.md', ({name} if '''# Calls\n{call}\n''' else {name}))\n",
    "create_file('docs/m.md', ''.join([p for p, q\n    in {name}.items()]) + '''# Calls\n"
    "{call}\n''')\n",
    "x = (*(p for p in {name}\n      if p), '''# Calls\n{call}\n''')\n",
    "create_file(\n    'docs/m.md',\n    (\n        {name}\n        if (\n            {name}\n"
    "            and {name}\n        )\n        else '''# Calls\n{call}\n'''\n    ),\n)\n",
    "create_file(\n    'docs/m.md',\n    (\n        {name}\n        if isinstance(\n"
    "            {name},\n            str,\n        )\n        else '''# Calls\n{call}\n'''\n"
    "    ),\n)\n",
    "x = {name} if isinstance(\n    {name}, str\n) else '''# Calls\n{call}\n'''\n",
    "create_file('docs/m.md', [v if ok(\n    v) else '''# Calls\n{call}\n''' for v in {name}])\n",
    "create_file('docs/m.md', ({name} if ({name} if {name}\n    else {name})\n"
    "    else '''# Calls\n{call}\n'''))\n",
])
def test_a_call_quoted_in_unfenced_code_cut_inside_an_expression_is_not_run(code):
    for pad in range(24):
        front = "I will write the doc.\n\n" + code.format(name="v" + "x" * pad, call="")
        assert rlm_mod._in_string_or_comment(front[:front.index("# Calls")]) is True
    front = "I will write the doc.\n\n" + code.format(name="v" + "x" * 9, call="{call}")
    for is_cut in (False, True):
        assert extract_code(front.format(call=_QUOTED_TOOL), is_cut) == (None, False)
        assert extract_code(front.format(call=_QUOTED_CALL) + _REAL_BLOCK, is_cut) == (
            "print(read_file('src/a.py'))\n", False)
    assert rlm_mod.reply_end(front.format(call=_QUOTED_CALL) + _REAL_BLOCK) is None


@pytest.mark.parametrize("front", [
    "Now change the docstrings from `'''` to `\"\"\"`.\n\n",
    "Now change the docstrings from `'''` to `\"\"\"`.\n",
    'The module docstring opens with """ but never closes.\n\n',
    "The docstring opening `\"\"\"` on line 3 is never closed, so I will fix it.\n\n",
    "Replace the `'''` quotes.\n\n", "The tests use ''' strings.\n\n",
    "Fix issue #12 first.\n", "## Step 1\n\n", "Plan (draft, see below\n",
    "Use a.b and c.d then e.f: g(h).\n", "def f():\n    return 1\n", "x = 1  # it's set\n",
    "- > the task says to fix it\n\n", "Step #1: ", '""" is never closed.\n\n',
    "The doc is '''# A\nx\n'''\n", "'''\n",
    "The module mixes two styles:\n- ''' in a.py\n- \"\"\" in b.py\n\nI will read a.py.\n\n",
    "Found these:\n- \"\"\" opens the module docstring but never closes.\n\n",
    "Styles:\n+ ''' (single)\n\n", "Plan:\n1. Fix it\n   - ''' should become \"\"\"\n\n",
    "Styles:\n* ''' in a.py\n\n", "2) ''' in a.py\n\n", "Use *'''* in a.py.\n\n",
    "*'''* is the old style.\n\n", "Use **\"\"\"** there.\n\n", "~~'''~~ is gone.\n\n",
    "Fix parse(text) if text is \"\"\"-quoted.\n\n",
    "It returns f(x) if x is ''' and None otherwise.\n\n",
    "設定ファイル(settings.py if 存在)の'''を直します。\n\n",
    "I will check (a.py if present\nin a.py) has ''' here.\n\n",
    "I will check (a.py if present\n''' in a.py)\n\n", "Plan (f(x) if g(y)\n''' in a.py)\n\n",
    "Plan (f(x) if g(y,\nz)\n''' in a.py)\n\n", "See (a if b(\nc) ''' d)\n\n",
    "See (a if (b if c\nd) ''' e)\n\n",
])
def test_a_call_after_prose_that_holds_quotes_or_marks_of_code_is_run(front):
    for is_cut in (False, True):
        assert extract_code(front + _QUOTED_TOOL, is_cut) == (
            "print(delete_file(path='src/a.py'))", True)
        assert extract_code(f"{front}<function_calls>\n{_QUOTED_CALL}\n</function_calls>",
                            is_cut) == ("revert_file('src/a.py')", True)


@pytest.mark.parametrize("quote", [
    f"Last turn:\n- > {_LINE_CALL}\n\nNow:\n", f"Last turn:\n1. > {_LINE_CALL}\n\nNow:\n",
    f"Last turn:\n* > {_LINE_CALL}\n\nNow:\n", f"Last turn:\n  - 2) > {_LINE_CALL}\n\nNow:\n",
    f"I'll read it.\n# never send {_LINE_CALL}\nNow:\n",
    f"I'll read it.\nx = f(1,  # not {_LINE_CALL}\nNow:\n",
])
def test_a_call_quoted_in_a_list_item_or_a_comment_is_not_run(quote):
    tool_quote = quote.replace(_LINE_CALL, _LINE_TOOL)
    for is_cut in (False, True):
        assert extract_code(quote + _REAL_BLOCK, is_cut) == (
            "print(read_file('src/a.py'))\n", False)
        assert extract_code(tool_quote, is_cut) == (None, False)
    assert rlm_mod.reply_end(quote + _REAL_BLOCK) is None


def test_only_the_lines_that_read_as_python_are_read_for_strings_and_comments():
    in_string = rlm_mod._in_string_or_comment
    assert in_string("I'll write it.\nx = '''a\n") is True
    assert in_string("It's\nx = 1\n  y = 2\n z = '''\n") is True
    assert in_string("x = '''a'''\nI'll run it.\n") is False
    assert in_string("I'll write it.\nDon't.\n") is False
    assert in_string("Use `x = '''` there.\n") is False
    assert in_string("def f(\n    a,\n):\n    \"\"\"Doc.\n\n    More.\n    \"\"\"\n") is False
    assert in_string("x = (1 +\n     '''abc\n") is True
    assert in_string("I'll read it.\n# never send ") is True
    assert in_string("See issue #12 ") is False
    assert in_string("x = 1\r\ny = '''\r\n") is True
    assert in_string("f(# \r1, # f(#\nx for x in y) # '''") is True
    lead = "ab(c)(d)(e)(f)(g)('''j''' for #"
    assert in_string(f"I'll do it.\n{lead} {lead}\ni in y) # '''") is True
    assert in_string("'''# A\n") is False
    assert in_string("I'll write the doc:\n'''# A\n'''\n") is False
    assert in_string("'''\nx = '''a\n") is True
    assert in_string("Styles:\n- ''' in a.py\n") is False
    assert in_string("Steps:\n- x = '''a\n") is True
    assert in_string("Steps:\n  1. f('''a\n") is True
    assert in_string("Use *'''* in a.py.\n") is False
    assert in_string("x = 2 * '''a\n") is True
    assert in_string("Note: the file is ready if you run it, '''a\n") is False
    assert in_string("y = 1 if x is not None else '''a\n") is True
    assert in_string("from a import b as c; d = f'{c}' or '''a\n") is True
    assert in_string("Fix parse(text) if text is '''a\n") is False
    assert in_string("x = (a if '''a\n") is True
    assert in_string("x = (a\n     if b\n     else '''a\n") is True
    assert in_string("x = (a\n     if (\n         b\n     )\n     else '''a\n") is True
    assert in_string("x = a if f(\n    b\n) else '''a\n") is True
    assert in_string("x = (a.py if b\n'''a\n") is False
    assert in_string("if x == '''a\n") is True
    assert in_string("値 = f(x)  # 「a」\nx = '''a\n") is True
    assert in_string("नमस्ते = (1) if a else '''a\n") is True
    assert in_string("関数f(x)を直す。'''a\n") is False
    assert in_string("x = f\"{d[\"k k\"]}\" + '''a\n") is (sys.version_info >= (3, 12))
    assert in_string("x = f\"{d[\"「k」\"]}\" + '''a\n") is (sys.version_info >= (3, 12))


@pytest.mark.parametrize("text, is_in", [
    ("{a}", False), ("{{a}} b", False), ("a {b} {c!r:>{w}} d", False), ("}", False),
    ("{title or ", True), ("{x:{w}", True), ("{{a}} {b[", True), ("{d[{1: 2}[1]", True),
    ("it's {d['}']} b", False), ("{x + '}' + ", True), ("{x + '", True),
    ("{d['a\\'}'] + ", True), ("{d['a\\'}']} b", False),
    ("{'''a'}''' + ", True), ("{'''a'''} b", False), ("{d[''''k''']} b", False),
    ("{x:'>10} b", False), ("{x!r:'^{w}} b", False), ("{x:'>10}{y + ", True),
    ("{d[1:'{']} b", False), ("{x:{'a'}} b", False), ("{f'{d['a}']}'} b", False),
    ("{x if y else f'{'a}'}' + ", True), ("{f'{x}' + ", True), ("{f'a\\'{x}'} b", False),
    ("{x  # }\n+ ", True), ("{x  # }", True), ("{x  # {\n} b", False),
    ("{x  # 'a\n} b", False), ("{x  # (a\n} b", False), ("{x:#x} b", False),
    ("{x  # a\n+ '#'} b", False),
])
def test_the_end_of_an_f_string_as_read_is_told_in_a_field_or_not(text, is_in):
    assert rlm_mod._is_in_field("x = f'" + text + "'", 6, 6 + len(text)) is is_in


@pytest.mark.parametrize("line, tail", [("f(\n", "Hi there.\n"), ("Plan (draft\n", ""),
                                        ("The tests use ''' strings.\n", ""),
                                        ("I'll read `a.py` (and fix it.\n", ""),
                                        ("Use ''' in a.py. ", "\n")])
def test_the_text_in_front_of_a_call_is_parsed_in_pieces_no_longer_than_it(monkeypatch,
                                                                            line, tail):
    seen_list = []
    error_line = rlm_mod._python_error_line
    monkeypatch.setattr(rlm_mod, "_python_error_line", lambda code, close_text: (
        seen_list.append(len(code + (close_text or ""))), error_line(code, close_text))[1])
    front = line * 2000 + tail
    assert rlm_mod._in_string_or_comment(front) is False
    assert sum(seen_list) < 10 * len(front)


@pytest.mark.parametrize("head, part, tail", [
    ("I'll ", "The file is ready. ", "\n"), ("I'll ", "see a.py, ", "\n"),
    ("I'll ", "Call f(x) now. ", "\n"), ("I'll ", "It's a.py and b's. ", "\n"),
    ("I'll ", "Use `x.y` here. ", "\n"), ("I'll ", "a.py (see b). ", "\n"),
    ("I'll ", "The file is ready. ", "(see below:\n"), ("I'll ", "see f(x. ", "\n"),
    ("I will ", "Call f(x) now. ", ""), ("I will ", "The file is ready. ", "x = '''abc"),
    ("", "ファイルa.pyを読み、関数f(x)を直します。", "(下書き\n"),
    ("", "ファイルa.pyを読み、関数f(x)を直します。", ""),
    ("", "ファイルa.pyを読み,関数f(x)を直す.", "(draft\n"), ("", "関数f(xを見る。", "\n"),
    ("", "Call print(f\"{a}\") now. ", "(draft\n"), ("", "Set name = f\"{a}\" and go. ", "(draft\n"),
    ("", "see log(f\"{a}\" now. ", "\n"), ("", "print(f\"{a}\")を呼ぶ。", "(下書き\n"),
    ("", "Call print(f\"{x:'>10}\") now. ", "(draft\n"),
    ("", "Call print(f\"{d[''''k''']}\") now. ", "(draft\n"),
    ("", "a=#1 ", "\n"), ("", "Set color:#fff and border:#000 here. ", "\n"),
    ("", "Call print(f\"{x:#x}\") now. ", "(draft\n"),
    ("I'll ", "See the list[#1] and ", "\n"), ("I'll ", "We fix the [#1 ", "\n"),
    ("", "a[#1 ", "\n"), ("", "fixed by [#123](https://github.com/o/r/pull/123) and ", "\n"),
    ("", "We fix the z.{#1 ", "\n"),
])
def test_one_long_line_in_front_of_a_call_is_read_in_a_time_that_grows_with_it(monkeypatch,
                                                                               head, part,
                                                                               tail):
    scan_list = []
    mark_re = rlm_mod._CODE_MARK_PAIR_RE
    in_code_span = rlm_mod._in_code_span
    error_line = rlm_mod._python_error_line
    start_in_range_iter = rlm_mod._start_in_range_iter

    class ReadCountList(list):
        def __getitem__(self, key):
            scan_list.append(1)
            return list.__getitem__(self, key)

    def search(text, at):
        mark = mark_re.search(text, at)
        scan_list.append((mark.end() if mark else len(text)) - at)
        return mark

    monkeypatch.setattr(rlm_mod, "_CODE_MARK_PAIR_RE", types.SimpleNamespace(search=search))
    monkeypatch.setattr(rlm_mod, "_in_code_span", lambda front: (
        scan_list.append(len(front)), in_code_span(front))[1])
    monkeypatch.setattr(rlm_mod, "_python_error_line", lambda code, close_text: (
        scan_list.append(len(code + (close_text or ""))), error_line(code, close_text))[1])
    monkeypatch.setattr(rlm_mod, "_start_in_range_iter", lambda start_list, index, range_list: (
        start_in_range_iter(ReadCountList(start_list), index, range_list)))
    front = head + part * 2000 + tail
    is_in_string = tail.startswith("x = ")
    assert rlm_mod._in_string_or_comment(front) is is_in_string
    assert sum(scan_list) < 10 * len(front)
    assert rlm_mod._in_string_or_comment(head + part * 200 + "x = '''abc\n") is True
    assert extract_code(front + _QUOTED_TOOL) == (
        (None, False) if is_in_string else ("print(delete_file(path='src/a.py'))", True))


_QUOTED_BLOCK ="Last turn I ran:\n```python\nrevert_file('src/a.py')\n```\n"
_FIX_BLOCK = "Now the fix:\n```python\nreplace_in_file('src/a.py', '3.14', 'math.pi')\n```\n"


@pytest.mark.parametrize("middle", [
    "The README shows:\n```markdown\n# Usage\n```python\nimport lib\n```\n```\n",
    "Output:\n```\nok\n````\n",
    "The doc says:\n```text\nRun it with\n```bash\nmake\n```\n```\n",
    'I will not use <invoke name="python"><parameter name="code"> markup.\n',
    'I will not use <invoke name="read_file"><parameter name="path"> markup.\n',
])
def test_a_reply_whose_parts_cannot_be_told_apart_runs_nothing(middle):
    for is_cut in (False, True):
        assert extract_code(_QUOTED_BLOCK + middle + _FIX_BLOCK, is_cut) == (None, False)


def test_a_block_left_open_runs_nothing_unless_the_reply_was_cut():
    reply = _QUOTED_BLOCK + "Now the fix:\n```python\nreplace_in_file('src/a.py', 'a', 'b')\n"
    assert extract_code(reply) == (None, False)
    assert extract_code(reply, True) == ("revert_file('src/a.py')\n", False)


@pytest.mark.parametrize("tail", ["```", "```\n", "\n```\n", "```python is the label I used.",
                                  "Done.\n```python is the label I used.\n"])
def test_a_fence_line_with_nothing_after_it_does_not_hide_the_block_in_front(tail):
    for front in ("", _QUOTED_BLOCK + "Now:\n"):
        for is_cut in (False, True):
            assert extract_code(front + _REAL_BLOCK + tail, is_cut) == (
                "print(read_file('src/a.py'))\n", False)
    crlf = (_REAL_BLOCK + tail).replace("\n", "\r\n")
    assert extract_code(crlf) == ("print(read_file('src/a.py'))\r\n", False)


@pytest.mark.parametrize("reply", [
    "See:\n```\n# Usage\n```python\nimport shutil\n```\n```\n",
    "It printed:\n```\n3\n```\nFix:\n" + _REAL_BLOCK + "```\n",
    "```text\nplan\n```\n" + _REAL_BLOCK + "```",
    "```\ndef f():\n    '''Use ```python\n```\n'''\ncreate_file('a.md', '''# A\n```python\n"
    "import shutil\n```\n''')\n```\n",
    _REAL_BLOCK + "Done.\n```\n",
    _REAL_BLOCK + "```python\n",
    _REAL_BLOCK + "```\nDone.\n",
    "Done.\n```\n",
])
def test_a_fence_line_that_may_close_a_block_in_front_runs_nothing(reply):
    assert extract_code(reply) == (None, False)


def test_unfenced_code_that_holds_parts_no_one_can_tell_apart_runs_whole():
    code = "create_file('R.md', '''\n```markdown\n```python\nx\n```\n```\n''')\n"
    assert extract_code(code) == (code.strip(), True)
    assert extract_code(code, True) == (None, False)


def test_a_reply_that_quotes_a_call_runs_nothing_in_the_loop():
    ran_list = []

    def delete_file(path: str) -> dict:
        """Delete a file."""
        ran_list.append(path)
        return {"ok": True}

    quote = f"Do not do this:\n```xml\n{_QUOTED_TOOL}\n```\nInstead I'll read it."
    llm = _ScriptedLLM([quote, _QUOTED_BLOCK + "It shows:\n```\nok\n````\n" + _FIX_BLOCK,
                        "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [delete_file], "I", max_turn=5).run("REQ", {}, "V")
    assert ran_list == [] and r["answer"] == "x"
    assert [code.startswith(NO_CODE_MARK) for code, _ in r["history"]] == [True, True, False]
    assert r["history"][0][1] == rlm_mod.NO_CODE_NOTE_DICT["prose"][0]


@pytest.mark.parametrize("name", ["text", "new_text"])
def test_a_value_that_holds_a_closing_tag_is_read_whole_or_not_at_all(name):
    def call(value: str) -> str:
        return (f'<invoke name="create_file">\n<parameter name="path">docs/calls.md</parameter>\n'
                f'<parameter name="{name}">{value}</parameter>\n</invoke>')

    value = "Close each value with </parameter> and the call with </invoke>.\n"
    assert extract_code(call(value)) == (
        f"print(create_file(path='docs/calls.md', {name}={value!r}))", True)
    # A value whose text closes a call as a reply would leaves where it ends open
    value = ('A call is written as:\n\n<invoke name="read_file">\n'
             '<parameter name="path">src/a.py</parameter>\n</invoke>\n\nand it runs.\n')
    for front in ("", "I will write the doc.\n\n"):
        for is_cut in (False, True):
            assert extract_code(front + call(value), is_cut) == (None, False)
    # A parameter that starts a reply and that a fence follows has no closing tag
    assert extract_code('<parameter name="other"></parameter>\n```python\nprint(1)\n```') == (
        None, False)


# == RLM.run =========================================
class _ScriptedLLM:
    """A stand-in LLM returning prepared replies in order. It records the prompts."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "I have no idea."


def test_run_puts_the_completing_turn_in_the_history():
    llm = _ScriptedLLM(["```python\nCOMPLETE('done')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["answer"] == "done"
    assert r["iters"] == 1
    assert len(r["history"]) == 1
    assert r["history"][0][0] == "COMPLETE('done')\n"


def test_run_answers_none_when_it_hits_the_limit():
    llm = _ScriptedLLM(["```python\nprint(1)\n```"] * 3)
    r = RLM(llm, [], "I", max_turn=3).run("REQ", {}, "V")
    assert r["answer"] is None
    assert r["iters"] == 3
    assert len(r["history"]) == 3


def test_a_turn_without_a_code_block_still_counts_as_a_turn():
    llm = _ScriptedLLM(["Let me explain the plan.", "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["iters"] == 2


def test_every_line_of_a_reply_without_code_becomes_a_comment():
    """Prefixing a single # leaves every line after the first as bare prose."""
    llm = _ScriptedLLM(["First the plan.\nThen I will write the code.",
                        "```python\nCOMPLETE('x')\n```"])
    RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    # Take turn 1's code slot as it was rendered into turn 2's prompt
    block = llm.prompts[1].split("### Turn 1 code\n```python\n", 1)[1]
    block = block.split("\n```", 1)[0]
    assert all(line.startswith("#") for line in block.splitlines() if line)


def test_no_fence_gets_into_the_output_slot_of_a_turn_without_code():
    llm = _ScriptedLLM(["prose that happens to contain ```",
                        "```python\nCOMPLETE('x')\n```"])
    RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    out = llm.prompts[1].split("### Turn 1 output\n```\n", 1)[1]
    assert "```" not in out.split("\n```", 1)[0]


# == A call that did not happen =============================
class _FailingLLM(_ScriptedLLM):
    """A stand-in LLM whose call raises once the prepared replies have run out."""

    def __call__(self, prompt: str) -> str:
        if not self.replies:
            raise RuntimeError("the broker did not make the call")
        return super().__call__(prompt)


def test_a_call_that_could_not_be_made_ends_the_loop_instead_of_the_run():
    llm = _FailingLLM(["```python\nprint(1)\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["answer"] is None
    assert "RuntimeError" in r["error"]


def test_the_turns_before_a_call_that_could_not_be_made_are_kept():
    llm = _FailingLLM(["```python\nprint(1)\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["iters"] == 1
    assert len(r["history"]) == 1


def test_a_run_that_went_through_reports_no_error():
    llm = _ScriptedLLM(["```python\nCOMPLETE('done')\n```"])
    assert RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")["error"] == ""


def test_a_run_that_used_every_turn_reports_no_error():
    llm = _ScriptedLLM(["```python\nprint(1)\n```"] * 2)
    assert RLM(llm, [], "I", max_turn=2).run("REQ", {}, "V")["error"] == ""


# == Turns cut off at AGENT_MAX_TOKEN ================
class _CutLLM(_ScriptedLLM):
    """A stand-in LLM reporting a cut-off reply through last_finish_reason.

    make_llm passes the finish reason on the callable's attribute rather than changing
    the return type. This pins down that the RLM reads that attribute.
    """

    def __init__(self, replies: list[str], cut_turns: set[int]):
        super().__init__(replies)
        self.cut_turns = cut_turns
        self.last_finish_reason = None

    def __call__(self, prompt: str) -> str:
        reply = super().__call__(prompt)
        self.last_finish_reason = ("length" if len(self.prompts) in self.cut_turns
                                   else "stop")
        return reply


def test_a_cut_turn_gets_the_cut_note_in_its_history():
    """Without a closing fence no block is found and the no-code path is taken.

    Retries are off (truncated_retries=0) so the unrecovered path is what is observed.
    """
    llm = _CutLLM(["```python\nreplace_in_file('a.py', '--- a/a.py",
                   "```python\nCOMPLETE('x')\n```"], cut_turns={1})
    r = RLM(llm, [], "I", max_turn=5, cut_retry_count=0).run("REQ", {}, "V")
    assert CUT_REPLY_NOTE in r["history"][0][1]
    assert "No code block" not in r["history"][0][1]


def test_a_cut_turn_whose_block_does_not_parse_gets_the_cut_note():
    """The fences inside a document close a block the limit cut short."""
    llm = _CutLLM(['```python\ntext = """# Flow\n```mermaid\nA --> B\n```\n\nmore of the',
                   "```python\nCOMPLETE('x')\n```"], cut_turns={1})
    r = RLM(llm, [], "I", max_turn=5, cut_retry_count=0).run("REQ", {}, "V")
    assert CUT_REPLY_NOTE in r["history"][0][1]
    assert "SyntaxError" not in r["history"][0][1]


def test_a_call_whose_code_does_not_compile_comes_back_as_its_syntax_error():
    """The reply is ended after its first call, so it is not taken as cut."""
    bad = '<invoke name="python">\n<parameter name="code">print(1\n</parameter>\n</invoke>'
    whole = (bad + "\n\n") * 50
    end = rlm_mod.reply_end(whole)
    # A reply with nowhere to end runs on to the output limit
    llm = _CutLLM([whole[:end], "```python\nCOMPLETE('x')\n```"],
                  cut_turns=set() if end is not None else {1})
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert "SyntaxError" in r["history"][0][1] and r["truncated_replies"] == 0
    assert rlm_mod.FENCE_LINE_NOTE not in r["history"][0][1]


def test_a_block_a_fence_line_in_its_string_ends_comes_back_as_its_syntax_error_and_the_rule():
    reply = ("```python\ncreate_file('d.md', '''# D\n\n```python\nprint('EXAMPLE')\n```\n''')\n"
             "print('written')\n```\n")
    llm = _ScriptedLLM([reply, "```python\nprint('next')\n```", "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    code, output = r["history"][0]
    assert code == "create_file('d.md', '''# D\n\n```python\nprint('EXAMPLE')\n"
    assert "SyntaxError" in output and output.endswith(rlm_mod.FENCE_LINE_NOTE)
    assert "EXAMPLE" not in output
    assert r["history"][1] == ("print('next')\n", "next\n")


def test_a_cut_turn_of_unfenced_code_is_not_run_and_is_asked_again():
    """Cut at a line's end, the first half of unfenced code still parses."""
    ran_list = []

    def write(text):
        ran_list.append(text)
        return {"ok": True}

    llm = _CutLLM(["write('step 1')\n", "```python\nCOMPLETE('x')\n```"], cut_turns={1})
    r = RLM(llm, [write], "I", max_turn=5, cut_retry_count=1).run("REQ", {}, "V")
    assert ran_list == [] and r["answer"] == "x" and r["iters"] == 1
    assert llm.prompts[1].endswith(CUT_RETRY_NOTE)


def test_a_cut_turn_whose_block_parses_is_run(caplog):
    llm = _CutLLM(["```python\nprint('ran')\n```\nand then some prose that was",
                   "```python\nCOMPLETE('x')\n```"], cut_turns={1})
    with caplog.at_level(logging.INFO, logger="codetwine_loom.rlm"):
        r = RLM(llm, [], "I", max_turn=5, cut_retry_count=0).run("REQ", {}, "V")
    assert "ran" in r["history"][0][1]
    # What followed the block is not in the history, so the log keeps its end
    assert "and then some prose that was" in caplog.text


def test_the_cut_notes_name_no_stage_specific_work():
    """Every stage's loop shares the two notes, and most stages write no diff."""
    for note in (CUT_REPLY_NOTE, CUT_RETRY_NOTE, rlm_mod.BRIEF_RETRY_NOTE):
        assert "diff" not in note.lower() and "hunk" not in note.lower()


def test_a_stage_that_lost_a_turn_to_the_output_limit_says_so_at_warning(caplog):
    llm = _CutLLM(["```python\nreplace_in_file('a.py', '--- a/a.py",
                   "```python\nCOMPLETE('x')\n```"], cut_turns={1})
    with caplog.at_level("INFO", logger="codetwine_loom.rlm"):
        RLM(llm, [], "I", stage="edit", max_turn=5, cut_retry_count=0).run("REQ", {}, "V")
    record = [r for r in caplog.records if "output limit cut" in r.getMessage()][0]
    assert record.levelname == "WARNING"
    assert record.getMessage() == ("the edit stage had replies the output limit cut: "
                                   "truncated_replies=1, truncated_lost=1")
    assert caplog.records[-1].getMessage().startswith("the edit stage ends after")


def test_a_cut_that_the_retry_recovered_is_logged_at_info(caplog):
    llm = _CutLLM(["```python\nreplace_in_file('a.py', '--- a/a.py",
                   "```python\nprint(1)\n```", "```python\nCOMPLETE('x')\n```"],
                  cut_turns={1})
    with caplog.at_level("INFO", logger="codetwine_loom.rlm"):
        RLM(llm, [], "I", stage="edit", max_turn=5, cut_retry_count=1).run("REQ", {}, "V")
    record = [r for r in caplog.records if "output limit cut" in r.getMessage()][0]
    assert record.levelname == "INFO"
    assert record.getMessage().endswith("truncated_replies=1")


def test_a_clean_stage_logs_no_cut_line(caplog):
    llm = _ScriptedLLM(["```python\nCOMPLETE('x')\n```"])
    with caplog.at_level("INFO", logger="codetwine_loom.rlm"):
        RLM(llm, [], "I", stage="edit", max_turn=5).run("REQ", {}, "V")
    assert not [r for r in caplog.records if "output limit cut" in r.getMessage()]


def test_a_cut_turn_is_asked_again_within_the_same_turn():
    """A retry spends no turn. A recovered turn leaves no trace of the cut."""
    llm = _CutLLM(["```python\nreplace_in_file('a.py', '--- a/a.py",
                   "```python\nCOMPLETE('x')\n```"], cut_turns={1})
    r = RLM(llm, [], "I", max_turn=5, cut_retry_count=1).run("REQ", {}, "V")
    assert r["answer"] == "x"
    assert r["iters"] == 1              # a retry does not count as a turn
    assert len(llm.prompts) == 2        # the LLM was called twice
    assert r["truncated_replies"] == 1  # the count of cut calls is unchanged
    assert r["truncated_lost"] == 0
    assert CUT_REPLY_NOTE not in r["history"][0][1]
    assert NO_CODE_MARK not in r["history"][0][0]


def test_the_retry_note_is_added_to_that_call_only():
    """Stored in the history, "write it shorter" would keep shaving later diffs."""
    llm = _CutLLM(["```python\nreplace_in_file('a.py', '--- a/a.py",
                   "```python\nprint(1)\n```",
                   "```python\nCOMPLETE('x')\n```"], cut_turns={1})
    RLM(llm, [], "I", max_turn=5, cut_retry_count=1).run("REQ", {}, "V")
    assert not llm.prompts[0].endswith(CUT_RETRY_NOTE)
    assert llm.prompts[1].endswith(CUT_RETRY_NOTE)
    assert CUT_RETRY_NOTE not in llm.prompts[2]


def test_a_turn_still_cut_after_the_retry_counts_as_lost():
    llm = _CutLLM(["```python\nreplace_in_file('a.py', '--- a/a.py"] * 2
                  + ["```python\nCOMPLETE('x')\n```"], cut_turns={1, 2})
    r = RLM(llm, [], "I", max_turn=5, cut_retry_count=1).run("REQ", {}, "V")
    assert r["truncated_replies"] == 2   # 2 calls were cut
    assert r["truncated_lost"] == 1      # 1 turn was lost
    assert CUT_REPLY_NOTE in r["history"][0][1]
    assert r["iters"] == 2               # COMPLETE is reached on turn 2


def test_retries_of_0_drop_the_turn_as_before():
    llm = _CutLLM(["```python\nreplace_in_file('a.py', '--- a/a.py",
                   "```python\nCOMPLETE('x')\n```"], cut_turns={1})
    r = RLM(llm, [], "I", max_turn=5, cut_retry_count=0).run("REQ", {}, "V")
    assert len(llm.prompts) == 2   # no retry, so turn 1 makes a single call
    assert r["iters"] == 2
    assert r["truncated_lost"] == 1


def test_no_retry_when_a_block_was_taken_even_if_the_reply_was_cut():
    """A closed block runs as it is; a retry would throw that execution away."""
    llm = _CutLLM(["```python\nprint(1)\n```\nand next",
                   "```python\nCOMPLETE('x')\n```"], cut_turns={1})
    r = RLM(llm, [], "I", max_turn=5, cut_retry_count=1).run("REQ", {}, "V")
    assert r["history"][0][1].startswith("1\n")
    assert r["truncated_lost"] == 0
    assert CUT_RETRY_NOTE not in llm.prompts[1]


def test_a_turn_that_was_not_cut_keeps_the_usual_wording():
    llm = _CutLLM(["Let me explain the plan.", "```python\nCOMPLETE('x')\n```"],
                  cut_turns=set())
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert "No code block" in r["history"][0][1]
    assert CUT_REPLY_NOTE not in r["history"][0][1]


def test_the_note_is_added_even_when_a_block_was_taken_before_the_cut():
    """A reply can be cut after a closed block. It runs, but the warning stays."""
    llm = _CutLLM(["```python\nprint(1)\n```\nand next",
                   "```python\nCOMPLETE('x')\n```"], cut_turns={1})
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["history"][0][1].startswith("1\n")
    assert CUT_REPLY_NOTE in r["history"][0][1]


def test_the_number_of_cut_turns_is_in_the_return_value():
    llm = _CutLLM(["```python\nprint(1)\n```"] * 3, cut_turns={1, 3})
    r = RLM(llm, [], "I", max_turn=3).run("REQ", {}, "V")
    assert r["truncated_replies"] == 2


# == Unfenced replies ==============

def test_unfenced_code_runs_when_it_parses_as_python():
    """A reply of bare code with no ```python fence is run when it parses."""
    llm = _ScriptedLLM(['print("hi")', "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["history"][0][1] == "hi\n"
    assert r["bare_code_turns"] == 1


def test_prose_is_not_run_even_without_a_fence():
    llm = _ScriptedLLM(["Let me explain the plan.", "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert "No code block" in r["history"][0][1]
    assert r["bare_code_turns"] == 0


def test_a_reply_without_a_call_is_not_run():
    """A single word such as "OK" parses as an expression, hence the call requirement."""
    llm = _ScriptedLLM(["OK", "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert "No code block" in r["history"][0][1]
    assert r["bare_code_turns"] == 0


def test_only_the_fenced_part_runs_when_there_is_a_fence():
    """In a reply mixing prose and code, the prose must not be executed."""
    llm = _ScriptedLLM(["First I read it.\n```python\nprint(1)\n```",
                        "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["history"][0][1] == "1\n"
    assert r["bare_code_turns"] == 0


# == Replies that write the call in markup ==================

_MARKUP_REPLY = ('<invoke name="python">\n<parameter name="code">print(1)\n'
                 'print(2)</parameter>\n</invoke>')


def test_a_markup_call_named_python_gives_its_code():
    assert extract_code(_MARKUP_REPLY) == ("print(1)\nprint(2)", True)
    assert extract_code("I read it first.\n" + _MARKUP_REPLY) == ("print(1)\nprint(2)", True)


def test_only_the_first_markup_call_is_taken():
    reply = _MARKUP_REPLY + '\n\n<invoke name="python">\n<parameter name="code">print(3)' \
        '</parameter>\n</invoke>'
    assert extract_code(reply) == ("print(1)\nprint(2)", True)


def test_a_markup_call_keeps_the_fences_and_closing_tags_its_code_holds():
    code = 'text = """```python\nprint(0)\n```\n</parameter>"""\nprint(text)'
    reply = f'<invoke name="python">\n<parameter name="code">{code}</parameter>\n</invoke>'
    assert extract_code(reply) == (code, True)


def test_a_parameter_that_holds_code_needs_no_call_around_it():
    reply = '<parameter name="python">print(1)</parameter>\n</invoke>'
    assert extract_code(reply) == ("print(1)", True)


def test_a_markup_call_of_a_function_becomes_a_call_that_prints():
    reply = ('<invoke name="read">\n<parameter name="path">a/b.py</parameter>\n'
             '<parameter name="line_list">[1, 20]</parameter>\n'
             '<parameter name="is_whole">true</parameter>\n</invoke>')
    assert extract_code(reply) == (
        "print(read(path='a/b.py', line_list=[1, 20], is_whole=True))", True)


@pytest.mark.parametrize("reply", [
    '<invoke name="python">\n<parameter name="code">print(1)',
    '<invoke name="read">\n<parameter name="path">a.py</parameter>',
    '<invoke name="read">\n</invoke>',
    '<invoke name="not a name">\n<parameter name="path">a.py</parameter>\n</invoke>',
    '<invoke name="read">\n<parameter name="not a name">a.py</parameter>\n</invoke>',
    '<parameter name="path">a.py</parameter>',
])
def test_a_markup_call_left_open_or_misnamed_gives_no_code(reply):
    assert extract_code(reply) == (None, False)


@pytest.mark.parametrize("head", [
    '<invoke name="python">\n</invoke>\n',
    '<invoke name="other">\n</invoke>\n',
    '<invoke>\n<parameter name="other">1</parameter>\n</invoke>\n',
])
def test_a_fenced_block_after_markup_that_holds_no_code_is_the_one_run(head):
    assert extract_code(head + "```python\nprint(1)\n```") == ("print(1)\n", False)


def test_a_markup_reply_is_run_and_counted_as_unfenced():
    llm = _ScriptedLLM([_MARKUP_REPLY, "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["history"][0] == ("print(1)\nprint(2)", "1\n2\n")
    assert r["bare_code_turns"] == 1


def test_a_markup_call_of_a_tool_prints_what_the_tool_returns():
    def double(value):
        """Return value twice."""
        return value * 2

    llm = _ScriptedLLM(['<invoke name="double">\n<parameter name="value">21</parameter>'
                        '\n</invoke>', "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [double], "I", max_turn=5).run("REQ", {}, "V")
    assert r["history"][0] == ("print(double(value=21))", "42\n")


def test_complete_is_reached_from_an_unfenced_reply_too():
    llm = _ScriptedLLM(["COMPLETE('x')"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["answer"] == "x"


# == Turns whose reply had no characters =============

def test_an_empty_reply_is_asked_again_within_the_same_turn():
    """A stop hit at the start returns no content block. Same as a cut turn."""
    llm = _ScriptedLLM(["", "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5, cut_retry_count=1).run("REQ", {}, "V")
    assert r["answer"] == "x"
    assert r["iters"] == 1           # a retry does not count as a turn
    assert len(llm.prompts) == 2
    assert llm.prompts[1].endswith(EMPTY_RETRY_NOTE)
    assert r["empty_replies"] == 0   # a recovered turn is not counted as lost


def test_a_turn_still_empty_after_the_retry_counts_as_lost():
    llm = _ScriptedLLM(["", "", "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5, cut_retry_count=1).run("REQ", {}, "V")
    assert r["empty_replies"] == 1
    assert EMPTY_REPLY_NOTE in r["history"][0][1]
    assert r["iters"] == 2


def test_a_prose_only_turn_is_not_asked_again():
    """The same prose would come back, so the retry would only cost money."""
    llm = _ScriptedLLM(["Let me explain the plan.", "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5, cut_retry_count=1).run("REQ", {}, "V")
    assert len(llm.prompts) == 2     # one call on turn 1, one on turn 2
    assert r["empty_replies"] == 0
    assert "No code block" in r["history"][0][1]


def test_it_works_with_an_llm_that_has_no_such_attribute():
    """A stand-in LLM has no last_finish_reason; getattr falls back."""
    llm = _ScriptedLLM(["```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["truncated_replies"] == 0


def test_run_keeps_the_raw_output_in_the_returned_history():
    """The output keeps its backticks; the prompt fences it with a longer run."""
    llm = _ScriptedLLM(["```python\nprint(chr(96) * 3)\n```",
                        "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert "```" in r["history"][0][1]
    body = _sent_text(llm.prompts[1]).split("### Turn 1 output\n", 1)[1]
    assert body.startswith("````\n```\n") and "\n````\n" in body


def test_a_refusal_the_model_did_not_print_reaches_the_history_and_the_next_prompt():
    llm = _ScriptedLLM(["```python\nr = refuse('x')\n```",
                        "```python\nCOMPLETE('x')\n```"])
    r = RLM(llm, [refuse], "I", max_turn=5).run("REQ", {}, "V")
    assert 'refuse() returned "ok": False' in r["history"][0][1]
    assert "no such path: x" in llm.prompts[1]


def test_a_text_cut_to_a_limit_is_not_cut_again_to_that_limit():
    """The count of what was left out stays the one the first cut wrote."""
    once = cut("a" * 500, 50)
    assert once.endswith("\n... (450 chars omitted)") and cut(once, 50) == once
    tail_once = rlm_mod.cut_tail("a" * 500, 50)
    assert tail_once.startswith("... (450 chars omitted)\n")
    assert rlm_mod.cut_tail(tail_once, 50) == tail_once
    # To a lower limit, and with more behind or in front of it, the count is still the
    # number of characters of the first text that are left out
    assert cut(once, 40) == "a" * 40 + "\n... (460 chars omitted)"
    assert cut(once, 60) == once
    assert cut(once + "b", 50) == "a" * 50 + "\n... (451 chars omitted)"
    assert rlm_mod.cut_tail(tail_once, 40) == "... (460 chars omitted)\n" + "a" * 40
    assert rlm_mod.cut_tail("b" + tail_once, 50) == "... (451 chars omitted)\n" + "a" * 50


@pytest.mark.parametrize("change", [
    lambda text: text.replace("\n", " "), lambda text: " ".join(text.split())])
def test_a_cut_text_whose_marker_stands_at_its_end_is_a_cut_text(change):
    """One line and a commit's body each hold the marker behind another white space."""
    for body in ("a" * 500, "word " * 100):
        once = change(cut(body, 50))
        assert "(450 chars omitted)" in once
        assert cut(once, 50) == once


@pytest.mark.parametrize("change", [
    lambda text: text + "\n", lambda text: text.replace("\n... ", "  ... "),
    lambda text: text + " \n ", lambda text: text + " " * 200_000,
    lambda text: text.replace("\n... ", " " * 200_000 + "\n... ")])
def test_white_space_around_a_marker_is_cut_as_any_text_is(change):
    """Whatever stands behind the marker, and more than one white space in front of
    it, is counted with what is left out, so the text is never over the limit."""
    once = change(cut("a" * 500, 50))
    again = cut(once, 50)
    assert again.startswith("a" * 50 + "\n... (") and again.count("chars omitted") == 1
    assert len(again) <= 50 + len("\n... (200450 chars omitted)")
    count = int(re.search(r"\((\d+) chars", again)[1])
    assert 450 < count <= 450 + 200_000 and cut(again, 50) == again
    # Cut to a lower limit, the marker is read and not counted as text
    lower = cut(once, 20)
    assert lower.startswith("a" * 20 + "\n... (") and lower.count("chars omitted") == 1
    assert int(re.search(r"\((\d+) chars", lower)[1]) == count + 30


@pytest.mark.parametrize("pad", [" " * 200_000, "\n" * 200_000, "\u3000\n\t " * 50_000])
def test_a_marker_with_white_space_alone_around_it_does_not_pass_the_limit(pad):
    text = "x" * 4000 + "\n... (1 chars omitted)" + pad
    assert len(cut(text, 4000)) <= 4000 + len("\n... (200001 chars omitted)")
    tail_text = pad + "... (1 chars omitted)\n" + "x" * 700
    assert len(rlm_mod.cut_tail(tail_text, 700)) <= 700 + len("... (200001 chars omitted)\n")


@pytest.mark.parametrize("digit_count", [16, 4300, 4301, 100_000])
def test_a_count_of_too_many_digits_is_no_marker(digit_count):
    """A run of digits no count of characters has is text, and nothing is raised."""
    mark = "... (" + "9" * digit_count + " chars omitted)"
    text = "x" * 100 + "\n" + mark + "\n" + "y" * 100
    assert cut(text, 50) == "x" * 50 + f"\n... ({len(text) - 50} chars omitted)"
    assert rlm_mod.cut_tail(text, 50) == f"... ({len(text) - 50} chars omitted)\n" + "y" * 50
    assert cut("x" * 50 + "\n" + mark, 50).endswith(f"({len(mark) + 1} chars omitted)")
    # One of the most digits read is a marker
    most = "x" * 50 + "\n... (" + "9" * 15 + " chars omitted)"
    assert cut(most, 50) == most


def test_a_line_put_in_front_of_a_cut_tail_is_counted_with_what_was_left_out():
    tail_once = rlm_mod.cut_tail("a" * 500, 50)
    text = "note: one line\n" + tail_once
    assert rlm_mod.cut_tail(text, 50) == "... (465 chars omitted)\n" + "a" * 50
    # A marker shorter than the part that is dropped is not cut in two
    assert rlm_mod.cut_tail("... (7 chars omitted)\n" + "a" * 60, 70) == \
        "... (7 chars omitted)\n" + "a" * 60
    assert cut("a" * 60 + "\n... (7 chars omitted)", 70) == "a" * 60 + "\n... (7 chars omitted)"


def test_a_marker_in_the_middle_of_what_a_cut_leaves_out_adds_its_count():
    text = "a" * 10 + "\n... (100 chars omitted)\n" + "b" * 10
    assert cut(text, 5) == "a" * 5 + "\n... (116 chars omitted)"
    assert rlm_mod.cut_tail(text, 5) == "... (116 chars omitted)\n" + "b" * 5


def test_what_a_stage_cut_is_shown_to_the_next_stage_with_the_count_it_was_cut_by(
        monkeypatch):
    from codetwine_loom import answer, dispatch
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 50)
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 40)
    question = cut("Q" * 300, 40)
    block = answer.render_answer_list([{"n": 1, "request": question,
                                        "answer": cut("A" * 5000, 50), "error": ""}])
    assert "... (260 chars omitted)" in block and "... (4950 chars omitted)" in block
    assert block.count("chars omitted") == 2
    earlier = dispatch._earlier_run_text(
        {"asked": question, "did": "x", "verdict": "failed", "why": "", "gap": "",
         "gap_done": ""}, [])
    assert "... (260 chars omitted)" in earlier and earlier.count("chars omitted") == 1


def test_cut_tail_keeps_the_tail_and_marks_the_head():
    assert rlm_mod.cut_tail("abcdef", 2) == "... (4 chars omitted)\nef"
    assert rlm_mod.cut_tail("abcdef", 6) == "abcdef"
    assert rlm_mod.cut_tail("abcdef", 0) == "abcdef"


def test_limit_text_states_a_limit_and_leaves_out_one_that_is_off():
    assert rlm_mod.limit_text("under {limit} chars", 6000) == "under 6000 chars"
    assert rlm_mod.limit_text("under {limit} chars", 0) == ""
    assert rlm_mod.limit_text("under {limit} chars", -1) == ""


def test_the_output_limit_phrase_follows_the_setting(monkeypatch):
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 1234)
    assert rlm_mod.output_limit_note(rlm_mod.OUTPUT_LIMIT_NOTE) == (
        "; only the first 1234\ncharacters come back")
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 0)
    assert rlm_mod.output_limit_note(rlm_mod.OUTPUT_LIMIT_NOTE) == ""


# == Folding the history =====================================
def test_fold_history_leaves_a_history_that_fits_alone():
    history_list = [("c1", "o1"), ("c2", "o2")]
    assert fold_history(history_list, 1000, 1, 10) == (history_list, (0, 0))
    assert fold_history(history_list, 1000, 1, 10)[0] is history_list


def test_fold_history_is_off_at_0():
    history_list = [("c", "x" * 5000)] * 3
    assert fold_history(history_list, 0, 1, 10)[0] is history_list


def test_fold_history_folds_the_oldest_turn_first():
    history_list = [("c1", "a" * 500), ("c2", "b" * 500), ("c3", "c" * 500)]
    folded, count_tuple = fold_history(history_list, 900, 1, 20)
    assert folded[0][1].startswith("a" * 20) and "chars dropped" in folded[0][1]
    assert folded[2] == history_list[2]
    assert count_tuple[0] == 2


def test_fold_history_never_folds_the_kept_turns():
    history_list = [("c1", "a" * 500), ("c2", "b" * 500), ("c3", "c" * 500)]
    folded, _ = fold_history(history_list, 10, 2, 20)
    assert folded[1] == history_list[1] and folded[2] == history_list[2]


def test_fold_history_keeps_the_code_when_folding_the_outputs_is_enough():
    history_list = [("code one", "a" * 500), ("code two", "b" * 500)]
    folded, _ = fold_history(history_list, 200, 0, 20)
    assert [c for c, _ in folded] == ["code one", "code two"]
    assert all("chars dropped" in o for _, o in folded)


def test_fold_history_folds_old_turns_code_when_the_outputs_alone_do_not_fit():
    history_list = [("x" * 500, "a" * 500), ("y" * 500, "b" * 500)]
    folded, count_tuple = fold_history(history_list, 100, 0, 20)
    assert folded[0][0].startswith("x" * 20 + "\n# ... (this turn's code was folded")
    assert "480 chars dropped" in folded[0][0] and "480 chars dropped" in folded[1][0]
    assert sum(len(c) + len(o) for c, o in folded) < 500
    assert count_tuple == (2, 2)


def test_fold_history_folds_every_output_before_any_code():
    """Both old outputs are folded and that alone reaches the target, so no code is."""
    history_list = [("x" * 500, "a" * 2000), ("y" * 500, "b" * 2000), ("z" * 10, "c" * 10)]
    folded, _ = fold_history(history_list, 3000, 1, 20)
    assert "chars dropped" in folded[0][1] and "chars dropped" in folded[1][1]
    assert folded[0][0] == "x" * 500 and folded[1][0] == "y" * 500


def test_fold_history_never_folds_the_kept_turns_code():
    history_list = [("x" * 500, "a" * 500), ("y" * 500, "b" * 500), ("z" * 500, "c" * 500)]
    folded, _ = fold_history(history_list, 10, 1, 20)
    assert folded[2] == history_list[2]
    assert "code was folded" in folded[0][0] and "code was folded" in folded[1][0]


def test_fold_history_folds_down_to_half_the_budget_once_over_it():
    """Over 1400 with three turns of 500: the first two are folded to get under 700."""
    history_list = [("c1", "a" * 500), ("c2", "b" * 500), ("c3", "c" * 500)]
    folded, _ = fold_history(history_list, 1400, 1, 20)
    assert "chars dropped" in folded[0][1] and "chars dropped" in folded[1][1]
    assert folded[2] == history_list[2]


def test_fold_history_does_not_start_under_the_budget():
    history_list = [("c1", "a" * 500), ("c2", "b" * 500), ("c3", "c" * 500)]
    assert fold_history(history_list, 2000, 0, 20)[0] is history_list


def test_fold_history_keeps_the_turns_an_earlier_call_folded_and_folds_no_further():
    """Four turns of 500 are over 1900 unfolded and under it with the first one folded."""
    history_list = [("c1", "a" * 500), ("c2", "b" * 500), ("c3", "c" * 500), ("c4", "d" * 500)]
    folded, count_tuple = fold_history(history_list, 1900, 1, 20, (1, 0))
    assert count_tuple == (1, 0)
    assert "chars dropped" in folded[0][1]
    assert folded[1:] == history_list[1:]


def test_fold_history_goes_on_from_the_turns_an_earlier_call_folded():
    history_list = [("c1", "a" * 500), ("c2", "b" * 500), ("c3", "c" * 500), ("c4", "d" * 500)]
    folded, count_tuple = fold_history(history_list, 1400, 1, 20, (1, 0))
    assert count_tuple[0] == 3
    assert all("chars dropped" in output for _, output in folded[:3])
    assert folded[3] == history_list[3]


def test_fold_by_count_folds_what_the_counts_say():
    history_list = [("x" * 500, "a" * 500), ("y" * 500, "b" * 500)]
    assert fold_by_count(history_list, (0, 0), 20) is history_list
    folded = fold_by_count(history_list, (2, 1), 20)
    assert "code was folded" in folded[0][0] and folded[1][0] == "y" * 500
    assert all("chars dropped" in output for _, output in folded)


def _same_block_count(before_list: list, after_list: list) -> int:
    """Return how many turns at the front of two rendered histories are the same."""
    count = 0
    for before, after in zip(before_list, after_list):
        if before != after:
            break
        count += 1
    return count


def test_a_run_over_the_budget_sends_the_turns_of_the_prompt_before_unchanged(monkeypatch):
    """The turns of one prompt are the front of the next one's, except on the turns
    the history is folded on, which come once per half budget of new turns."""
    monkeypatch.setattr(config, "HISTORY_MAX_CHAR", 60000)
    monkeypatch.setattr(config, "HISTORY_KEEP_TURN", 3)
    monkeypatch.setattr(config, "HISTORY_FOLD_HEAD_CHAR", 200)
    stage = RLM(lambda prompt: "", [], "I", max_turn=60)
    history_list, before_list, change_turn_list = [], [], []
    for turn in range(1, 61):
        after_list = stage.fold(history_list)
        if _same_block_count(before_list, after_list) < len(before_list):
            change_turn_list.append(turn)
        assert sum(len(c) + len(o) for c, o in after_list) <= 60000
        before_list = after_list
        history_list.append(("x" * 50, "y" * 2000))
    assert len(change_turn_list) == 2
    assert all(later - earlier > 10 for earlier, later
               in zip(change_turn_list, change_turn_list[1:]))


def test_run_folds_the_history_it_puts_in_the_prompt(monkeypatch):
    """The prompt shrinks; what run() returns as the history does not."""
    monkeypatch.setattr(config, "HISTORY_MAX_CHAR", 200)
    monkeypatch.setattr(config, "HISTORY_KEEP_TURN", 1)
    monkeypatch.setattr(config, "HISTORY_FOLD_HEAD_CHAR", 10)
    llm = _ScriptedLLM(["```python\nprint('z' * 3000)\n```",
                        "```python\nprint('y' * 10)\n```",
                        "```python\nCOMPLETE('done')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert "chars dropped" in llm.prompts[2]
    assert "z" * 3000 not in llm.prompts[2]
    assert "z" * 3000 in r["history"][0][1]


def test_run_carries_every_turn():
    llm = _ScriptedLLM(["```python\nprint('one')\n```",
                        "```python\nCOMPLETE('done')\n```"])
    RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert "print('one')" in llm.prompts[1]


def _sent_text(prompt) -> str:
    """The text of what an LLM was sent, whichever form PROMPT_CACHE gave it."""
    return prompt if isinstance(prompt, str) else prompt_text(prompt)


class _RefusingLLM(_ScriptedLLM):
    """A stand-in LLM refusing the prompt for its length on the given calls."""

    def __init__(self, replies: list[str], refuse_calls: set[int]):
        super().__init__(replies)
        self.refuse_calls = refuse_calls

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) in self.refuse_calls:
            raise PromptTooLong("ContextWindowExceededError: prompt is too long")
        return self.replies.pop(0) if self.replies else "I have no idea."


def test_a_prompt_the_provider_refuses_for_length_is_folded_harder_and_asked_again(
        monkeypatch, caplog):
    """The second call of turn 2 carries a shorter history, and turn 2 counts once."""
    monkeypatch.setattr(config, "HISTORY_MAX_CHAR", 0)
    monkeypatch.setattr(config, "HISTORY_KEEP_TURN", 0)
    monkeypatch.setattr(config, "HISTORY_FOLD_HEAD_CHAR", 10)
    llm = _RefusingLLM(["```python\nprint('z' * 3000)\n```",
                        "```python\nCOMPLETE('done')\n```"], refuse_calls={2})
    with caplog.at_level("WARNING", logger="codetwine_loom.rlm"):
        r = RLM(llm, [], "I", stage="edit", max_turn=5).run("REQ", {}, "V")
    assert r["answer"] == "done" and r["iters"] == 2 and r["error"] == ""
    assert len(llm.prompts) == 3
    assert "z" * 3000 in _sent_text(llm.prompts[1])
    assert "z" * 3000 not in _sent_text(llm.prompts[2])
    assert "chars dropped" in _sent_text(llm.prompts[2])
    assert "refused the prompt of the edit stage for its length" in caplog.text


def test_a_prompt_refused_twice_in_one_turn_ends_the_stage_with_the_error():
    llm = _RefusingLLM(["```python\nprint(1)\n```"], refuse_calls={2, 3})
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["answer"] is None and r["iters"] == 1
    assert r["error"].startswith("ContextWindowExceededError")
    assert len(llm.prompts) == 3


def test_the_lowered_history_budget_holds_for_the_rest_of_the_run(monkeypatch):
    """One turn of 3000 fits 8000 and its half; the budget goes under what the refused
    prompt carried, so the prompt sent again is folded and so is every later one."""
    monkeypatch.setattr(config, "HISTORY_MAX_CHAR", 8000)
    monkeypatch.setattr(config, "HISTORY_KEEP_TURN", 0)
    monkeypatch.setattr(config, "HISTORY_FOLD_HEAD_CHAR", 10)
    llm = _RefusingLLM(["```python\nprint('z' * 3000)\n```",
                        "```python\nprint('y' * 3000)\n```",
                        "```python\nCOMPLETE('done')\n```"], refuse_calls={2})
    stage = RLM(llm, [], "I", max_turn=5)
    stage.run("REQ", {}, "V")
    assert stage.history_max_char < 3100
    assert config.HISTORY_MAX_CHAR == 8000
    assert "z" * 3000 in _sent_text(llm.prompts[1])
    assert "z" * 3000 not in _sent_text(llm.prompts[2])
    assert "chars dropped" in _sent_text(llm.prompts[-1])


def test_a_grown_prompt_refused_at_the_kept_output_limit_is_folded_not_halved(monkeypatch):
    """The first call keeps 8192; turn 2's longer prompt folds and goes at 8192 again."""
    monkeypatch.setattr(config, "HISTORY_MAX_CHAR", 80000)
    monkeypatch.setattr(config, "HISTORY_KEEP_TURN", 0)
    monkeypatch.setattr(config, "HISTORY_FOLD_HEAD_CHAR", 10)
    monkeypatch.setattr(llm_mod, "_fit_warn_pair_set", set())
    reply_list = ["```python\nprint('z' * 3000)\n```", "```python\nCOMPLETE('done')\n```"]
    seen = []

    def completion(**kwargs):
        text = json.dumps(kwargs["messages"])
        seen.append((kwargs["max_tokens"], "z" * 3000 in text))
        if kwargs["max_tokens"] > 8192 or "z" * 3000 in text:
            raise llm_mod.litellm.ContextWindowExceededError(
                "maximum context length is 12000 tokens", "m", "hosted_vllm")
        message = type("Message", (), {"content": reply_list.pop(0)})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice]})()

    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    stage = RLM(llm_mod.make_llm("m", 32768), [], "I", max_turn=5)
    assert stage.run("REQ", {}, "V")["answer"] == "done"
    assert seen == [(32768, False), (16384, False), (8192, False), (8192, True),
                    (8192, False)]


def test_a_refused_prompt_under_half_the_budget_is_folded_before_it_is_sent_again(monkeypatch):
    """Half of 80000 is far over the one turn of 3000 the refused prompt carried."""
    monkeypatch.setattr(config, "HISTORY_MAX_CHAR", 80000)
    monkeypatch.setattr(config, "HISTORY_KEEP_TURN", 0)
    monkeypatch.setattr(config, "HISTORY_FOLD_HEAD_CHAR", 10)
    llm = _RefusingLLM(["```python\nprint('z' * 3000)\n```",
                        "```python\nCOMPLETE('done')\n```"], refuse_calls={2})
    stage = RLM(llm, [], "I", max_turn=5)
    assert stage.run("REQ", {}, "V")["answer"] == "done"
    assert "z" * 3000 not in _sent_text(llm.prompts[2])


def test_a_run_starts_with_nothing_folded(monkeypatch):
    monkeypatch.setattr(config, "HISTORY_MAX_CHAR", 200)
    monkeypatch.setattr(config, "HISTORY_KEEP_TURN", 0)
    monkeypatch.setattr(config, "HISTORY_FOLD_HEAD_CHAR", 10)
    stage = RLM(_ScriptedLLM(["```python\nCOMPLETE('done')\n```"]), [], "I", max_turn=5)
    stage.fold([("c1", "z" * 3000), ("c2", "y" * 3000)])
    assert stage.fold_count_tuple != (0, 0)
    result_dict = stage.run("REQ", {}, "V")
    assert stage.fold_count_tuple == (0, 0) and result_dict["answer"] == "done"


def test_the_harvest_is_folded_harder_once_on_a_refused_prompt(monkeypatch):
    monkeypatch.setattr(config, "HISTORY_MAX_CHAR", 0)
    monkeypatch.setattr(config, "HISTORY_KEEP_TURN", 0)
    monkeypatch.setattr(config, "HISTORY_FOLD_HEAD_CHAR", 10)
    llm = _RefusingLLM(["I changed a.py"], refuse_calls={1})
    stage = RLM(llm, [], "I", max_turn=5)
    answer = late_answer(stage, "REQ", "V", [("c1", "z" * 3000)], "## Out of turns",
                         "Answer with the summary.")
    assert answer == "I changed a.py" and len(llm.prompts) == 2
    assert "z" * 3000 not in _sent_text(llm.prompts[1])
    llm = _RefusingLLM(["I changed a.py"], refuse_calls={1, 2})
    with pytest.raises(PromptTooLong):
        late_answer(RLM(llm, [], "I", max_turn=5), "REQ", "V", [("c1", "z" * 3000)],
                    "## Out of turns", "Answer with the summary.")


# == Calls that run no code: the harvest and the report =====
class _CutOnceLLM(_ScriptedLLM):
    """A stand-in LLM whose reply is cut off on the given calls."""

    def __init__(self, replies: list[str], cut_calls: set[int]):
        super().__init__(replies)
        self.cut_calls = cut_calls
        self.last_finish_reason = None

    def __call__(self, prompt) -> str:
        reply = super().__call__(prompt)
        self.last_finish_reason = "length" if len(self.prompts) in self.cut_calls else "stop"
        return reply


def test_ask_whole_takes_the_reply_whole_from_a_turns_llm():
    """A turn's llm ends a reply in front of the history's headings; an answer that
    walks through the turns under those headings is not a turn."""
    answer = "I changed two files.\n### Turn 3\nAdded the parser.\n"
    turn_list = []

    def llm(prompt):
        turn_list.append(prompt)
        return answer[:rlm_mod.reply_end(answer)]

    llm.whole = lambda prompt: answer
    stage = RLM(llm, [], "I", max_turn=5)
    assert late_answer(stage, "REQ", "V", [("c1", "o1")], "## Out of turns",
                       "Answer.") == answer
    assert turn_list == []


def test_ask_whole_makes_one_call_when_the_reply_is_whole():
    llm = _CutOnceLLM(["whole"], cut_calls=set())
    assert rlm_mod.ask_whole(llm, "PROMPT", 2) == ("whole", False)
    assert llm.prompts == ["PROMPT"]


def test_ask_whole_takes_a_callable_with_no_finish_reason_as_never_cut():
    llm = _ScriptedLLM(["whole"])
    assert rlm_mod.ask_whole(llm, "PROMPT", 2) == ("whole", False)


def test_ask_whole_asks_again_more_briefly_while_the_reply_is_cut():
    """The note goes on the retry's prompt only; the whole reply of call 2 is returned."""
    llm = _CutOnceLLM(["first half", "all of it"], cut_calls={1})
    assert rlm_mod.ask_whole(llm, "PROMPT", 2) == ("all of it", False)
    assert llm.prompts == ["PROMPT", "PROMPT" + rlm_mod.BRIEF_RETRY_NOTE]


def test_ask_whole_reports_a_reply_still_cut_after_the_retries(caplog):
    llm = _CutOnceLLM(["a", "b", "c"], cut_calls={1, 2, 3})
    with caplog.at_level("WARNING", logger="codetwine_loom.rlm"):
        assert rlm_mod.ask_whole(llm, "PROMPT", 2) == ("c", True)
    assert len(llm.prompts) == 3 and "still cut off" in caplog.text


def test_ask_whole_follows_the_configured_retry_count(monkeypatch):
    monkeypatch.setattr(config, "CUT_REPLY_MAX_RETRY", 0)
    llm = _CutOnceLLM(["a", "b"], cut_calls={1, 2})
    assert rlm_mod.ask_whole(llm, "PROMPT") == ("a", True)


def test_ask_whole_puts_the_note_on_the_last_block_of_a_block_prompt(monkeypatch):
    monkeypatch.setattr(config, "PROMPT_CACHE", True)
    llm = _CutOnceLLM(["a", "b"], cut_calls={1})
    block_list = [("head\n", True), ("tail", False)]
    rlm_mod.ask_whole(llm, block_list, 1)
    assert llm.prompts[0] is block_list
    assert llm.prompts[1][-1] == ("tail" + rlm_mod.BRIEF_RETRY_NOTE, False)


def test_the_harvest_asks_again_when_its_reply_was_cut():
    llm = _CutOnceLLM(["I changed a.py and", "I changed a.py"], cut_calls={1})
    stage = RLM(llm, [], "I", max_turn=5)
    assert late_answer(stage, "REQ", "V", [("c1", "o1")], "## Out of turns",
                       "Answer.") == "I changed a.py"
    assert len(llm.prompts) == 2
    assert _sent_text(llm.prompts[1]).endswith(rlm_mod.BRIEF_RETRY_NOTE)


def test_a_harvest_still_cut_after_the_retries_carries_the_cut_marker():
    llm = _CutOnceLLM(["I changed", "I changed a", "I changed a."], cut_calls={1, 2, 3})
    stage = RLM(llm, [], "I", max_turn=5, cut_retry_count=2)
    answer = late_answer(stage, "REQ", "V", [("c1", "o1")], "## Out of turns", "Answer.")
    assert answer == "I changed a." + rlm_mod.CUT_END_NOTE


def test_the_harvest_prompt_is_folded_too(monkeypatch):
    """Otherwise the run's most expensive prompt is the one folding was meant to stop."""
    monkeypatch.setattr(config, "HISTORY_MAX_CHAR", 200)
    monkeypatch.setattr(config, "HISTORY_KEEP_TURN", 1)
    monkeypatch.setattr(config, "HISTORY_FOLD_HEAD_CHAR", 10)
    llm = _ScriptedLLM(["I changed a.py"])
    stage = RLM(llm, [], "I", max_turn=5)
    late_answer(stage, "REQ", "V", [("c1", "z" * 3000), ("c2", "y" * 10)],
            "## Out of turns", "Answer with the summary.")
    assert "chars dropped" in llm.prompts[0]
    assert "z" * 3000 not in llm.prompts[0]


# == turn_note and on_complete (the two-stage hooks) ====
def test_turn_note_goes_between_the_history_and_the_turn_line():
    p = build_prompt("I", {}, "V", "R", [("c1", "o1")], 2, 20,
                     turn_note="## Plan status\n  a.py NOT TOUCHED")
    order = [p.index(s) for s in ("### Turn 1 output", "## Plan status", "## Turn 2/20")]
    assert order == sorted(order)


def test_run_rebuilds_the_turn_note_every_turn():
    llm = _ScriptedLLM(["```python\nprint(1)\n```", "```python\nCOMPLETE('x')\n```"])
    RLM(llm, [], "I", max_turn=5).run(
        "REQ", {}, "V", turn_note=lambda turn, max_turn: f"NOTE {turn}/{max_turn}")
    assert "NOTE 1/5" in llm.prompts[0] and "NOTE 2/5" in llm.prompts[1]


def test_on_complete_can_send_one_completion_back():
    llm = _ScriptedLLM(["```python\nCOMPLETE('too early')\n```",
                        "```python\nCOMPLETE('done')\n```"])
    refusals = ["b.py has not been touched", ""]
    r = RLM(llm, [], "I", max_turn=5).run(
        "REQ", {}, "V", on_complete=lambda answer: refusals.pop(0))
    assert r["answer"] == "done"
    assert r["refused_completes"] == 1
    # The refusal is the refused turn's output, so the model can read why
    assert r["history"][0][1] == "b.py has not been touched"


def test_a_refused_completion_still_spends_its_turn():
    """The refusal on the last turn adds the one turn it is answered in, and no more."""
    llm = _ScriptedLLM(["```python\nCOMPLETE('too early')\n```"] * 4)
    r = RLM(llm, [], "I", max_turn=3).run(
        "REQ", {}, "V", on_complete=lambda answer: "not yet")
    assert r["answer"] is None and r["iters"] == 4 and r["refused_completes"] == 4


def test_a_refusal_on_the_last_turn_is_answered_in_one_more_turn():
    llm = _ScriptedLLM(["```python\nprint(1)\n```", "```python\nCOMPLETE('early')\n```",
                        "```python\nCOMPLETE('with the reason')\n```"])
    refusal_list = []

    def once(answer):
        if refusal_list:
            return ""
        refusal_list.append(answer)
        return "not yet"

    r = RLM(llm, [], "I", max_turn=2).run("REQ", {}, "V", on_complete=once)
    assert r["answer"] == "with the reason" and r["outcome"] == rlm_mod.COMPLETE_KEY
    assert r["iters"] == 3 and r["refused_completes"] == 1
    assert "## Turn 3/3" in llm.prompts[-1]


def test_a_refusal_before_the_last_turn_adds_no_turn():
    llm = _ScriptedLLM(["```python\nCOMPLETE('early')\n```", "```python\nprint(1)\n```"])
    state = {"refused": False}

    def once(answer):
        if state["refused"]:
            return ""
        state["refused"] = True
        return "not yet"

    r = RLM(llm, [], "I", max_turn=2).run("REQ", {}, "V", on_complete=once)
    assert r["answer"] is None and r["iters"] == 2


def test_complete_is_accepted_when_there_is_no_guard():
    llm = _ScriptedLLM(["```python\nCOMPLETE('done')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["answer"] == "done" and r["refused_completes"] == 0


# == COMPLETE(): the one way a stage ends itself ============
def test_complete_ends_the_run_and_says_it_ended_itself():
    llm = _ScriptedLLM(["```python\nCOMPLETE('done')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["answer"] == "done"
    assert r["outcome"] == rlm_mod.COMPLETE_KEY and r["iters"] == 1


def test_a_run_that_never_ends_itself_has_no_outcome():
    llm = _ScriptedLLM(["```python\nprint(1)\n```"])
    r = RLM(llm, [], "I", max_turn=2).run("REQ", {}, "V")
    assert r["answer"] is None and r["outcome"] is None


def test_there_is_no_second_way_to_end_a_run():
    """Ending with nothing changed is read out of the repository, not out of a second
    call the model can pick."""
    assert list(rlm_mod.END_CALL_DICT.values()) == ["COMPLETE"]
    llm = _ScriptedLLM(["```python\nANSWER('nothing to change')\n```",
                        "```python\nCOMPLETE('done')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    # The first turn raises NameError inside the REPL and is a turn like any other
    assert r["answer"] == "done" and r["iters"] == 2
    assert "NameError" in r["history"][0][1]


def test_the_history_names_the_call_that_was_made():
    """A turn that ended the run is in the history as the call the model wrote."""
    llm = _ScriptedLLM(["```python\nCOMPLETE('already handled')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["history"][-1][1] == "COMPLETE() was called: already handled"


def test_the_guard_is_handed_what_the_run_ended_with():
    seen = []
    llm = _ScriptedLLM(["```python\nCOMPLETE('nothing to do')\n```",
                        "```python\nCOMPLETE('done')\n```"])

    def guard(answer):
        seen.append(answer)
        return "say what you read" if len(seen) == 1 else ""

    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V", on_complete=guard)
    assert seen == ["nothing to do", "done"]
    assert r["answer"] == "done" and r["refused_completes"] == 1


# == the harvest's reply is sometimes the call itself ==
def test_a_harvested_call_is_unwrapped_to_its_argument():
    """A harvest reply of COMPLETE("...") instead of the text is unwrapped to the text."""
    assert rlm_mod.unwrap_end_call('COMPLETE("it is read in config.py")') == \
        "it is read in config.py"
    assert rlm_mod.unwrap_end_call('COMPLETE("changed a.py")') == "changed a.py"


def test_a_fenced_call_is_unwrapped_too():
    assert rlm_mod.unwrap_end_call('```python\nCOMPLETE("in config.py")\n```') == \
        "in config.py"


def test_a_call_the_parser_cannot_read_is_unwrapped_too():
    """The real shape: several paragraphs inside one plain "...", spanning lines."""
    reply = 'COMPLETE("one line\n\nand another **one**")'
    assert rlm_mod.unwrap_end_call(reply) == "one line\n\nand another **one**"


def test_an_answer_that_quotes_a_fence_is_still_unwrapped():
    """extract_code() would take the quoted block and leave the wrapper standing."""
    reply = 'COMPLETE("look at:\n\n```\nEXCLUDE = []\n```\n\nin settings.py")'
    assert rlm_mod.unwrap_end_call(reply).startswith("look at:")
    assert "COMPLETE(" not in rlm_mod.unwrap_end_call(reply)


def test_prose_comes_back_exactly_as_it_was():
    prose = "it is read in config.py, by read_config()"
    assert rlm_mod.unwrap_end_call(prose) == prose


def test_anything_that_is_not_one_call_with_one_string_is_left_alone():
    for reply in ('COMPLETE(x)', 'print(1); COMPLETE("two")', 'COMPLETE("a", "b")',
                  'other_call("a")', 'it says COMPLETE("no") in the middle'):
        assert rlm_mod.unwrap_end_call(reply) == reply


def test_the_harvest_unwraps_what_it_gets():
    llm = _ScriptedLLM(['COMPLETE("it is read in config.py")'])
    stage = RLM(llm, [], "I", max_turn=5)
    assert late_answer(stage, "REQ", "V", [("c1", "o1")], "## Out of turns",
                   "Answer with the answer.") == "it is read in config.py"


# == whether the harvest is asked for at all ==================
def test_a_stage_that_used_every_turn_is_asked():
    assert need_late_answer({"answer": None, "outcome": None, "error": ""}) is True


def test_a_stage_that_handed_something_over_is_not_asked():
    assert need_late_answer({"answer": "I changed a.py", "outcome": rlm_mod.COMPLETE_KEY,
                             "error": ""}) is False


def test_a_stage_that_ended_itself_with_none_is_not_asked():
    assert need_late_answer({"answer": None, "outcome": rlm_mod.COMPLETE_KEY,
                             "error": ""}) is False


def test_a_run_that_stopped_early_is_not_asked():
    """The call that ended the run would not go through for the harvest either."""
    assert need_late_answer({"answer": None, "outcome": None,
                           "error": "RuntimeError: the broker did not answer"}) is False


def test_a_stage_whose_harvest_is_turned_off_is_not_asked():
    assert need_late_answer({"answer": None, "outcome": None, "error": ""},
                            on=False) is False


def test_a_result_with_no_error_key_counts_as_one_that_did_not_stop_early():
    assert need_late_answer({"answer": None, "outcome": None}) is True


def test_a_loop_ended_with_none_makes_no_extra_call():
    llm = _ScriptedLLM(["```python\nCOMPLETE(None)\n```", "a late summary"])
    out = run_stage(RLM(llm, [], "I", max_turn=3), "REQ", {}, "V", "## Out of turns",
                    "Answer.")
    assert (out["answer"], out["outcome"], out["harvested"]) == (
        None, rlm_mod.COMPLETE_KEY, False)
    assert len(llm.prompts) == 1


# == the guard on a JSON hand-over ===========================
def test_none_is_sent_back_once_by_the_json_guard():
    guard = rlm_mod.json_guard("the plan")
    refusal = guard(None)
    assert refusal == ("COMPLETE() was not accepted: COMPLETE() got None in place of "
                       "the plan. Hand over the object itself, COMPLETE({...}) or "
                       "COMPLETE([...]). This is asked once only.")
    assert guard(None) == ""


def test_an_unreadable_answer_does_not_use_up_the_one_refusal_of_the_shape():
    guard = rlm_mod.json_guard("the plan", lambda value: "it holds no entry")
    assert "could not be read as JSON" in guard("{not json")
    assert guard("[]") == ("COMPLETE() was not accepted: it holds no entry. "
                           "This is asked once only.")
    assert "could not be read as JSON" in guard("{not json")
    assert guard("[]") == ""


def test_the_json_guard_asks_the_shape_only_of_what_it_could_read():
    seen = []

    def shape_reason(value):
        seen.append(value)
        return "it holds no entry" if not value else ""

    guard = rlm_mod.json_guard("the plan", shape_reason)
    assert "could not be read as JSON" in guard("{not json")
    assert seen == []
    guard = rlm_mod.json_guard("the plan", shape_reason)
    assert guard("[]") == ("COMPLETE() was not accepted: it holds no entry. "
                           "This is asked once only.")
    assert guard("[]") == "" and seen == [[]]
    assert rlm_mod.json_guard("the plan", shape_reason)({"a": 1}) == ""


def _holds_itself() -> dict:
    """Return a dict one of whose values is the dict itself."""
    value_dict = {"files": []}
    value_dict["self"] = value_dict
    return value_dict


@pytest.mark.parametrize("make", [
    lambda: {"files": {"a.py", "b.py"}}, lambda: [{"file": b"a.py"}],
    lambda: {"files": [object()]}, lambda: {("a", 1): "x"}, _holds_itself])
def test_an_object_json_cannot_hold_is_sent_back_each_time_and_not_read(make):
    """A set, bytes, an object of a class, a key that is no string, and a dict that
    holds itself. Nothing raises, at the gate or behind it."""
    guard = rlm_mod.json_guard("the plan", lambda value: "the shape was asked")
    refusal = guard(make())
    assert refusal.startswith("COMPLETE() was not accepted: the plan holds a value JSON "
                              "cannot hold (")
    assert "dicts, lists, strings, numbers, True, False and None alone" in refusal
    assert guard(make()).startswith("COMPLETE() was not accepted: the plan holds")
    read_dict = rlm_mod.parse_json_answer(make(), "the plan")
    assert read_dict["ok"] is False and read_dict["error"].startswith("not JSON: ")
    assert isinstance(read_dict["raw"], str)
    assert plan.parse_plan(make())["ok"] is False
    assert dispatch.parse_dispatch(make())["ok"] is False
    assert testcreate.parse_case_list(make(), "", [])["ok"] is False
    out = explore.parse_explore_answer(make())
    assert out["ok"] is False and out["error"].startswith("not JSON: ")


@pytest.mark.parametrize("make", [
    lambda: 10 ** 5000, lambda: {"files": [{"path": "a.py", "line": 10 ** 5000}]},
    lambda: "[" + "1" * 4301 + "]", lambda: '{"files": ' + "9" * 5000 + "}",
    lambda: "[" * 100_000 + "]" * 100_000])
def test_a_number_or_a_nesting_json_does_not_read_is_no_answer_and_raises_nothing(make):
    """An integer of more digits than are written or read, as the object and inside
    the JSON text, and a text nested deeper than is read."""
    guard = rlm_mod.json_guard("the plan", lambda value: "the shape was asked")
    assert guard(make()).startswith("COMPLETE() was not accepted: the plan ")
    assert guard(make()).startswith("COMPLETE() was not accepted: the plan ")
    read_dict = rlm_mod.parse_json_answer(make(), "the plan")
    assert read_dict["ok"] is False and isinstance(read_dict["raw"], str)
    assert len(read_dict["raw"]) <= config.RAW_EXCERPT_CHAR
    assert plan.parse_plan(make())["ok"] is False
    assert dispatch.parse_dispatch(make())["ok"] is False
    assert testcreate.parse_case_list(make(), "", [])["ok"] is False
    assert explore.parse_explore_answer(make())["ok"] is False


def test_a_value_str_does_not_write_is_named_by_its_type():
    assert rlm_mod.text_of(10 ** 5000) == "<int>"
    assert rlm_mod.text_of([10 ** 5000]) == "<list>"
    assert rlm_mod.text_of({"a": 1}) == "{'a': 1}" and rlm_mod.text_of("x") == "x"
    out = rlm_mod.run_stage(_StageThatEnds(10 ** 5000), "request", {}, "", "", "",
                            text_answer=True)
    assert out["answer"] == "<int>"


_NO_TEXT_CODE = ("class NoText:\n    def __str__(self):\n        raise RuntimeError('no')\n"
                 "COMPLETE(NoText())")


@pytest.mark.parametrize("code, answer_text", [
    ("COMPLETE(10 ** 5000)", "<int>"),
    ("COMPLETE({'files': [], 'n': 10 ** 5000})", "<dict>"),
    ("node = []\nfor _ in range(100000):\n    node = [node]\nCOMPLETE(node)", "<list>"),
    (_NO_TEXT_CODE, "<NoText>"),
])
def test_a_value_that_cannot_be_written_out_ends_the_run_like_any_other(code, answer_text):
    """The turn that ended the run goes into the history with the value as text_of()
    writes it, and the value itself is handed over."""
    stage = RLM(llm=lambda prompt: f"```python\n{code}\n```", tool_list=[],
                instruction="i", stage="edit", max_turn=3)
    out = stage.run("r", {}, "")
    assert (out["outcome"], out["iters"], out["error"]) == ("complete", 1, "")
    assert out["history"][-1][1].endswith(f"COMPLETE() was called: {answer_text}")
    assert rlm_mod.text_of(out["answer"]) == answer_text


def test_a_json_stage_handed_a_value_that_cannot_be_written_reads_it_as_no_json():
    stage = RLM(llm=lambda prompt: "```python\nCOMPLETE(10 ** 5000)\n```", tool_list=[],
                instruction="i", stage="plan", max_turn=3)
    out = stage.run("r", {}, "", on_complete=rlm_mod.refuse_once(
        rlm_mod.json_guard("the plan")))
    assert (out["outcome"], out["refused_completes"]) == ("complete", 1)
    assert rlm_mod.parse_json_answer(out["answer"], "the plan")["ok"] is False


@pytest.mark.parametrize("code", ["raise SystemExit(10 ** 5000)",
                                  "import sys\nsys.exit([10 ** 5000])"])
def test_an_exit_with_a_value_that_cannot_be_written_out_is_a_turn_s_output(code):
    out = rlm_mod.Repl({}, {}).run(code, 4000)
    assert out.startswith("SystemExit(<") and "is not allowed here" in out


_ODD_SYNTAX_ERROR_CODE = "print('before')\nraise SyntaxError('m', ('f', 'x', None, 5, 'y', 'z'))"


@pytest.mark.parametrize("code, error", [
    ("import sys\nprint('before')\nsys.stdout.close()", "ValueError"),
    ("import sys\nout = sys.stdout\nout.write('before')\nout.close()\n1/0", "ValueError"),
    ("import sys\nsys.stderr.close()\nprint('after')", "ValueError"),
    ("import sys\nsys.stdout.getvalue = lambda: 5", "TypeError"),
    (_ODD_SYNTAX_ERROR_CODE, "AttributeError"),
])
def test_a_turn_whose_output_cannot_be_read_back_says_so_in_its_place(code, error):
    repl = Repl({}, {})
    assert repl.run(code, 500) == rlm_mod.OUTPUT_READ_ERROR_NOTE.format(error=error)
    assert repl.run("print('next')", 500) == "next\n"


@pytest.mark.parametrize("code, last_line", [
    ("print('before')\nraise BaseException('x')", "BaseException: x"),
    ("print('before')\nraise GeneratorExit()", "GeneratorExit"),
    ("import asyncio\nprint('before')\nraise asyncio.CancelledError()",
     "asyncio.exceptions.CancelledError"),
])
def test_what_a_block_raises_that_is_no_exception_is_a_turn_s_output(code, last_line):
    repl = Repl({}, {})
    out = repl.run(code, 500)
    assert out.startswith("before\n") and out.splitlines()[-1] == last_line
    assert repl.run("print('next')", 500) == "next\n"


def test_a_keyboard_interrupt_in_a_block_ends_the_run():
    with pytest.raises(KeyboardInterrupt):
        Repl({}, {}).run("raise KeyboardInterrupt()", 500)


def test_a_turn_that_closed_its_output_and_called_complete_keeps_the_answer():
    code = "import sys\nsys.stdout.close()\nCOMPLETE('done')"
    stage = RLM(llm=lambda prompt: f"```python\n{code}\n```", tool_list=[],
                instruction="i", stage="edit", max_turn=3)
    out = stage.run("r", {}, "")
    assert (out["outcome"], out["iters"], out["answer"]) == ("complete", 1, "done")
    assert out["history"][-1][1] == (
        rlm_mod.OUTPUT_READ_ERROR_NOTE.format(error="ValueError")
        + "\nCOMPLETE() was called: done")


@pytest.mark.parametrize("change", [
    "r['n'] = 10 ** 5000", "r[(1, 2)] = 'x'\nr['error'] = None", "r['all'] = [r]\nr.pop('error')",
    "r.clear()"])
def test_a_refusal_is_noted_as_the_tool_returned_it_whatever_the_turn_did_to_it(change):
    out = Repl({}, {"refuse": refuse}).run(f"r = refuse('x')\n{change}", 500)
    assert out == TOOL_REFUSAL_NOTE.format(tool="refuse",
                                           result='{"error": "no such path: x"}')


def test_a_refusal_holding_a_value_that_cannot_be_written_out_costs_the_turn_s_output():
    def refuse_with_list(path):
        return {"ok": False, "seen": []}
    repl = Repl({}, {"refuse_with_list": refuse_with_list})
    out = repl.run("r = refuse_with_list('x')\nr['seen'].append(10 ** 5000)", 500)
    assert out == rlm_mod.OUTPUT_READ_ERROR_NOTE.format(error="ValueError")
    assert repl.run("print('next')", 500) == "next\n"


_NO_END_CLASS = "class NoEnd{base}:\n    def __str__(self):\n        while True:\n            pass\n"


@pytest.mark.parametrize("code, want", [
    (_NO_END_CLASS.format(base="(Exception)") + "print('before')\nraise NoEnd()",
     "NoEnd: <exception str() failed>"),
    (_NO_END_CLASS.format(base="") + "raise SystemExit(NoEnd())",
     rlm_mod.OUTPUT_READ_ERROR_NOTE.format(error="TurnTimeout")),
])
def test_a_block_that_ended_with_a_value_whose_text_never_comes_is_a_turn_s_output(
        monkeypatch, code, want):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    repl = Repl({}, {})
    assert want in repl.run(code, 500)
    assert repl.run("print('next')", 500) == "next\n"


class _TextThatRaises:
    """A value whose str() raises what it was built with."""

    def __init__(self, error):
        self.error = error

    def __str__(self):
        raise self.error


class _TextSubclass(str):
    """A str that formats as another text."""

    def __format__(self, spec):
        return "another"


class _TextOfSubclass:
    """A value whose str() is a str of a subclass."""

    def __str__(self):
        return _TextSubclass("plain")


@pytest.mark.parametrize("error", [SystemExit(0), CompleteCall("again"), GeneratorExit(),
                                   rlm_mod.TurnTimeout("x"), RuntimeError("no")])
def test_a_value_whose_text_raises_anything_is_named_by_its_type(error):
    assert rlm_mod.text_of(_TextThatRaises(error)) == "<_TextThatRaises>"


def test_a_value_whose_text_raises_a_keyboard_interrupt_passes_it_on():
    with pytest.raises(KeyboardInterrupt):
        rlm_mod.text_of(_TextThatRaises(KeyboardInterrupt()))


def test_the_text_of_a_value_is_a_plain_str():
    for value in (_TextSubclass("plain"), _TextOfSubclass()):
        text = rlm_mod.text_of(value)
        assert type(text) is str and f"{text}" == "plain"


def test_a_value_whose_text_never_comes_is_named_by_its_type_at_the_turn_limit(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    code = _NO_END_CLASS.format(base="") + "COMPLETE(NoEnd())"
    stage = RLM(llm=lambda prompt: f"```python\n{code}\n```", tool_list=[],
                instruction="i", stage="edit", max_turn=3)
    out = rlm_mod.run_stage(stage, "r", {}, "", "", "", text_answer=True)
    assert (out["outcome"], out["iters"], out["answer"]) == ("complete", 1, "<NoEnd>")
    assert out["history"][-1][1] == "COMPLETE() was called: <NoEnd>"


def test_a_timer_another_block_set_is_not_taken_for_the_one_of_the_text(monkeypatch):
    """A text asked for inside a turn's own limit ends with that turn."""
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)

    def read(value):
        return rlm_mod.text_of(value)

    code = _NO_END_CLASS.format(base="") + "read(NoEnd())\nprint('the rest of the turn ran')"
    out = Repl({}, {"read": read}).run(code, 500)
    assert "TURN_TIMEOUT_SEC" in out and "the rest of the turn ran" not in out


_NOT_PYTHON_TEXT_LIST = [
    "I looked at it \ud83d and will go on.",
    "print('\ud83d')",
    "x = " + " + ".join(["1"] * 10000) + "\nprint(x)",
    "a" + ".b()" * 6000,
    "print(1)\x00",
    "print(",
]


@pytest.mark.parametrize("text", _NOT_PYTHON_TEXT_LIST)
def test_a_text_the_parser_does_not_read_is_not_python_wherever_it_is_asked(text):
    assert rlm_mod.parse_tree(text) is None
    assert rlm_mod._parse_ok(text) is False
    assert rlm_mod.bare_code(text) is None
    assert rlm_mod.tool_call_text(text, ["print"]) == ""
    assert rlm_mod._unwrap_text(text) is None
    assert rlm_mod.unwrap_end_call(text) == text
    assert extract_code(text) == (None, False)
    assert extract_code(f"```python\n{text}\n```") == (f"{text}\n", False)


def test_a_lone_surrogate_in_a_reply_stops_no_reading_of_where_it_ends():
    reply = ("```python\ns = '''x \ud83d\n```\n### Turn 2 output\n"
             '<invoke name="python">\n<parameter name="code">print(1)</parameter>\n</invoke>')
    assert rlm_mod._in_string_or_comment("s = '''x \ud83d\n") is True
    assert rlm_mod._in_string_or_comment("s = 'x \ud83d'\n") is False
    assert rlm_mod.reply_end(reply) == reply.index("\n### Turn 2")
    assert rlm_mod.reply_end("s = '''\ud83d\n" + reply[reply.index("<invoke"):]) is None


def test_a_reply_of_words_with_a_lone_surrogate_is_a_turn_with_no_code():
    reply_list = ["I looked at it \ud83d and will go on.", "```python\nCOMPLETE('done')\n```"]
    stage = RLM(llm=lambda prompt: reply_list.pop(0), tool_list=[], instruction="i",
                stage="edit", max_turn=3)
    out = stage.run("r", {}, "")
    assert (out["outcome"], out["iters"], out["answer"]) == ("complete", 2, "done")
    assert out["history"][0][1] == rlm_mod.NO_CODE_NOTE_DICT["prose"][0]


def test_an_answer_with_a_lone_surrogate_given_after_the_last_turn_is_kept():
    reply_list = ["```python\nprint(1)\n```"] * 2 + ["I changed a.py \ud83d as asked."]
    stage = RLM(llm=lambda prompt: reply_list.pop(0), tool_list=[], instruction="i",
                stage="edit", max_turn=2)
    out = run_stage(stage, "r", {}, "", "late", "close", text_answer=True)
    assert (out["answer"], out["error"]) == ("I changed a.py \ud83d as asked.", "")
    assert out["harvested"] is True


def test_a_markup_call_with_an_argument_nested_too_deep_to_read_is_passed_as_its_text():
    deep = "[" * 20000 + "]" * 20000
    reply = f'<invoke name="f">\n<parameter name="a">{deep}</parameter>\n</invoke>'
    code = rlm_mod.extract_code(reply)[0]
    assert code.startswith("print(f(a='[[[[") and code.endswith("]]]]'))")


def test_an_object_json_can_hold_is_read_as_the_json_written_of_it_reads_back():
    value = {"files": [{"path": "a.py", "n": 1.5, "ok": True, "none": None}], 3: "x"}
    assert rlm_mod.parse_json_answer(value, "the plan") == {"ok": True, "value": {
        "files": [{"path": "a.py", "n": 1.5, "ok": True, "none": None}], "3": "x"}}
    assert rlm_mod.json_read_reason(value, "the plan") == ""


class _DictSubclass(dict):
    """A dict of a subclass."""


class _IntSubclass(int):
    """An int of a subclass."""


def test_an_object_handed_over_is_read_as_lists_text_keys_and_plain_values():
    value = _DictSubclass(files=({"path": _TextSubclass("a.py"), "lines": (1, _IntSubclass(2))},),
                          order=("a.py",))
    plain_value = rlm_mod.parse_json_answer(value, "the plan")["value"]
    assert plain_value == {"files": [{"path": "a.py", "lines": [1, 2]}], "order": ["a.py"]}
    assert type(plain_value) is dict and type(plain_value["files"]) is list
    assert type(plain_value["files"][0]["path"]) is str
    assert type(plain_value["files"][0]["lines"][1]) is int
    assert rlm_mod.parse_json_answer((1, 2), "the findings")["value"] == [1, 2]
    text_dict = rlm_mod.parse_json_answer(_TextSubclass('{"a": [1]}'), "the plan")
    assert text_dict == {"ok": True, "value": {"a": [1]}}


def test_a_plan_a_split_the_findings_and_the_cases_read_a_tuple_as_the_list_it_is(
        monkeypatch):
    entry = {"path": "a.py", "change": "add parse()", "anchor": "new file"}
    plan_dict = {"files": (entry,), "order": ("a.py",), "interfaces": ("a.py: parse(text)",),
                 "facts": ("f1", "f2")}
    guard = rlm_mod.json_guard("the plan", plan._file_entry_error)
    assert guard({"files": ("no entry",)}).startswith("COMPLETE() was not accepted: none of")
    out = plan.parse_plan(plan_dict)["plan"]
    assert (out["files"], out["order"], out["interfaces"], out["facts"]) == (
        [entry], ["a.py"], ["a.py: parse(text)"], ["f1", "f2"])
    assert dispatch.parse_dispatch({"tasks": ("do this", "do that")}) == {
        "ok": True, "tasks": ["do this", "do that"]}
    monkeypatch.setattr(config, "EXPLORE_MAX_RESULT", 2)
    find_tuple = tuple({"file": f"f{n}.py", "lines": (1, 2), "why": "w"} for n in range(5))
    out = explore.parse_explore_answer(find_tuple)
    assert out["findings"] == [{"file": f"f{n}.py", "lines": [1, 2], "why": "w"}
                               for n in range(2)]
    target_list = [{"file": "y.py", "name": "h", "signature": "def h():", "tests": []}]
    read_dict = testcreate.parse_case_list({"cases": ({"name": "h", "expect": "e"},)}, "",
                                           target_list)
    assert [case["expect"] for case in read_dict["cases"]] == ["e"]


class _DictThatRaises(dict):
    """A dict whose items() raises what it was built with."""

    def __init__(self, error):
        super().__init__(files=[])
        self.error = error

    def items(self):
        raise self.error


@pytest.mark.parametrize("error", [KeyError("k"), SystemExit(0), CompleteCall("again")])
def test_an_object_whose_writing_raises_anything_is_no_json_and_raises_nothing(error):
    guard = rlm_mod.json_guard("the plan")
    assert guard(_DictThatRaises(error)).startswith(
        f"COMPLETE() was not accepted: the plan holds a value JSON cannot hold "
        f"({type(error).__name__}: ")
    read_dict = rlm_mod.parse_json_answer(_DictThatRaises(error), "the plan")
    assert read_dict["ok"] is False
    assert read_dict["error"].startswith(f"not JSON: {type(error).__name__}: ")
    assert plan.parse_plan(_DictThatRaises(error))["ok"] is False


def test_an_object_whose_writing_never_ends_is_no_json_at_the_turn_limit(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)

    class NoEnd(dict):
        def items(self):
            while True:
                pass

    read_dict = rlm_mod.parse_json_answer(NoEnd(files=[]), "the plan")
    assert read_dict["ok"] is False
    assert read_dict["error"].startswith("not JSON: TurnTimeout: ")


# == The settings the loop holds ============================

def test_the_loop_follows_a_turn_limit_rewritten_after_import(monkeypatch):
    """A default argument would have frozen it at import, while build_instruction()
    and cli.print_result() read the setting per call and would state another number.
    """
    monkeypatch.setattr(config, "RLM_MAX_TURN", 99)
    assert RLM(llm=lambda p: "", tool_list=[], instruction="x").max_turn == 99


def test_the_loop_follows_an_output_limit_rewritten_after_import(monkeypatch):
    """The instructions name this number, so the loop has to cut at the same one."""
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 1234)
    stage = RLM(llm=lambda p: "", tool_list=[], instruction="x")
    assert stage.max_output_char == 1234


def test_the_loop_follows_a_retry_count_rewritten_after_import(monkeypatch):
    monkeypatch.setattr(config, "CUT_REPLY_MAX_RETRY", 4)
    assert RLM(llm=lambda p: "", tool_list=[],
               instruction="x").cut_retry_count == 4


def test_a_setting_passed_in_wins_over_the_one_in_config(monkeypatch):
    """The stages with a limit of their own (DISPATCH_MAX_TURN, PLAN_MAX_TURN)
    pass it."""
    monkeypatch.setattr(config, "RLM_MAX_TURN", 99)
    assert RLM(llm=lambda p: "", tool_list=[], instruction="x",
               max_turn=7).max_turn == 7


def test_a_limit_of_zero_is_kept_rather_than_read_as_unset(monkeypatch):
    monkeypatch.setattr(config, "CUT_REPLY_MAX_RETRY", 4)
    assert RLM(llm=lambda p: "", tool_list=[], instruction="x",
               cut_retry_count=0).cut_retry_count == 0


# == What a stage that answers in its own words handed over ==

class _StageThatEnds:
    """A stand-in for RLM whose run ends itself with one answer."""

    stage = "stand-in"

    def __init__(self, answer, outcome=rlm_mod.COMPLETE_KEY):
        self.answer, self.outcome = answer, outcome

    def run(self, **kwargs):
        return {"answer": self.answer, "outcome": self.outcome, "history": [], "iters": 1,
                "refused_completes": 0, "error": "" if self.outcome else "the call failed"}


@pytest.mark.parametrize("answer, want", [
    ("done", "done"), ({"changed": ["a.py"]}, "{'changed': ['a.py']}"), (None, ""),
    ({"a.py"}, "{'a.py'}"), (5, "5")])
def test_a_stage_that_answers_in_words_hands_over_text_whatever_complete_was_given(
        answer, want):
    out = rlm_mod.run_stage(_StageThatEnds(answer), "request", {}, "", "", "",
                            text_answer=True)
    assert out["answer"] == want and out["harvested"] is False


def test_what_a_stage_handed_over_is_left_as_it_is_without_the_setting():
    """A stage that hands an object over, and one that did not end itself."""
    answer = {"files": []}
    assert rlm_mod.run_stage(_StageThatEnds(answer), "request", {}, "", "", "")[
        "answer"] is answer
    assert rlm_mod.run_stage(_StageThatEnds(None, outcome=None), "request", {}, "", "", "",
                             text_answer=True)["answer"] is None


# == Why a stage handed nothing over ========================

def test_a_stage_that_used_every_turn_says_so():
    assert no_answer_reason("", 20, "COMPLETE") == "no COMPLETE() within 20 turns"


def test_a_stage_that_ended_itself_with_nothing_says_that_and_not_the_turn_limit():
    assert no_answer_reason("", 20, "COMPLETE", "complete") == (
        "COMPLETE() was called with an empty answer")
    assert no_answer_reason("RuntimeError: x", 20, "COMPLETE", "complete") == (
        "the run stopped early: RuntimeError: x")


def test_every_stage_that_reads_json_tells_an_empty_hand_over_from_the_turn_limit(
        monkeypatch, capsys):
    """The planning stage, the splitting stage, the stage that decides the cases and
    the line of the command line."""
    from codetwine_loom import cli
    empty = "COMPLETE() was called with an empty answer"
    assert plan.parse_plan(None, "", "complete")["error"] == (
        f"the planning stage handed no plan over: {empty}")
    assert dispatch.parse_dispatch(None, "", outcome="complete")["error"] == (
        f"the splitting stage handed nothing over: {empty}")
    assert testcreate.parse_case_list(None, "", [], outcome="complete")["error"] == empty
    assert cli.no_answer_line({"error": "", "outcome": "complete"}, 20, "COMPLETE") == (
        f"answer: none ({empty})")
    # A stage that did not end itself used its turns, and one that stopped says why
    for max_name in ("PLAN_MAX_TURN", "DISPATCH_MAX_TURN", "TEST_CREATE_MAX_TURN"):
        monkeypatch.setattr(config, max_name, 20)
    turn = "no COMPLETE() within 20 turns"
    assert plan.parse_plan(None, "", None)["error"].endswith(turn)
    assert dispatch.parse_dispatch(None, "")["error"].endswith(turn)
    assert testcreate.parse_case_list(None, "", [])["error"] == turn
    assert cli.no_answer_line({"error": ""}, 20, "COMPLETE") == f"answer: none ({turn})"
    assert plan.parse_plan(None, "RuntimeError: x", "complete")["error"].endswith(
        "the run stopped early: RuntimeError: x")


def test_a_stage_whose_call_did_not_happen_says_that_instead():
    """Reported as a turn limit, the reader spends another call on the same failure."""
    assert no_answer_reason("RuntimeError: the broker did not answer", 20,
                            "COMPLETE") == ("the run stopped early: RuntimeError: "
                                            "the broker did not answer")


def test_an_output_within_the_head_is_not_folded():
    assert rlm_mod._fold("short", 10, rlm_mod.FOLD_NOTE) == "short"
    assert rlm_mod.FOLD_NOTE.split("{")[0] in rlm_mod._fold("x" * 11, 10, rlm_mod.FOLD_NOTE)


# == cut: a cap of 0 is no cap ===============================
def test_a_cap_of_zero_keeps_the_whole_text():
    assert cut("abc", 0) == "abc" and cut("abc", -1) == "abc"
    assert cut("abc", 2) == "ab" + CUT_NOTE.format(omitted=1)


def test_one_line_folds_every_run_of_whitespace():
    assert one_line("  a\n\n b\t c  ") == "a b c"


# == run_stage: the loop, then the one extra call ============
class _StageStub:
    """An RLM whose run() returns what it was built with and whose llm() is recorded."""

    instruction, tool_dict, stage, max_turn = "INSTR", {}, "plan", 3
    call = 1

    def __init__(self, answer, error=""):
        self.result = {"answer": answer, "outcome": None if answer is None else "complete",
                       "iters": 3, "history": [("c", "o")], "error": error}
        self.prompt_list = []

    def run(self, request, variable_dict, var_doc, **run_kw):
        self.kw = run_kw
        return dict(self.result)

    def fold(self, history_list):
        return history_list

    def llm(self, prompt):
        self.prompt_list.append(prompt)
        return "harvested answer"


def test_run_stage_takes_the_answer_the_loop_gave():
    stage = _StageStub("done")
    out = run_stage(stage, "REQ", {}, "V", "## Out of turns", "Answer.",
                    turn_note=None)
    assert out["answer"] == "done" and out["harvested"] is False
    assert stage.prompt_list == [] and stage.kw == {"turn_note": None}


def test_run_stage_asks_once_when_the_turns_ran_out():
    stage = _StageStub(None)
    out = run_stage(stage, "REQ", {}, "V", "## Out of turns", "Answer.")
    assert out["answer"] == "harvested answer" and out["harvested"] is True
    assert len(stage.prompt_list) == 1 and "## Out of turns" in stage.prompt_list[0]


def test_run_stage_puts_the_state_in_front_of_the_note_of_the_one_extra_call():
    """The state is asked for the turn after the last one the stage ran."""
    asked_list = []

    def state_note(turn, max_turn):
        asked_list.append((turn, max_turn))
        return "## State"

    stage = _StageStub(None)
    run_stage(stage, "REQ", {}, "V", "## Out of turns", "Answer.", late_state=state_note)
    assert asked_list == [(4, 3)] and stage.kw == {}
    assert "\n## State\n\n## Out of turns\n" in stage.prompt_list[0]
    stage = _StageStub(None)
    run_stage(stage, "REQ", {}, "V", "## Out of turns", "Answer.",
              late_state=lambda turn, max_turn: "")
    assert "\n\n## Out of turns\n" in stage.prompt_list[0]
    assert "## State" not in stage.prompt_list[0]
    done = _StageStub("done")
    run_stage(done, "REQ", {}, "V", "## Out of turns", "Answer.", late_state=state_note)
    assert asked_list == [(4, 3)]


def test_the_one_extra_call_is_headed_with_the_last_turn_the_stage_ran():
    """A COMPLETE() sent back on the last turn adds a turn, and the call made after it
    names that turn in the state and in the turn line alike."""
    llm = _ScriptedLLM(["```python\nprint(1)\n```", "```python\nCOMPLETE('early')\n```",
                        "```python\nprint(2)\n```", "what was done"])
    stage = RLM(llm, [], "I", max_turn=2)
    out = run_stage(stage, "REQ", {}, "V", "## Out of turns", "Answer.",
                    late_state=lambda turn, max_turn: f"## State {min(turn, max_turn)}/{max_turn}",
                    on_complete=rlm_mod.refuse_once(lambda answer: "not yet"))
    assert out["iters"] == 3 and out["harvested"] and out["answer"] == "what was done"
    assert _sent_text(llm.prompts[-1]).endswith(
        "\n## State 3/3\n\n## Out of turns\n\n## Turn 3/3\nAnswer.")


def test_run_stage_does_not_ask_a_stage_that_stopped_early_or_is_told_not_to():
    stopped = _StageStub(None, error="the broker did not answer")
    assert run_stage(stopped, "REQ", {}, "V", "n", "c")["answer"] is None
    assert stopped.prompt_list == []
    off = _StageStub(None)
    assert run_stage(off, "REQ", {}, "V", "n", "c", late_on=False)["harvested"] is False
    assert off.prompt_list == []


def test_run_stage_ends_without_an_answer_when_the_one_extra_call_fails(caplog):
    stage = _StageStub(None)

    def llm(prompt):
        raise RuntimeError("503 from the provider")

    stage.llm = llm
    with caplog.at_level(logging.WARNING):
        out = run_stage(stage, "REQ", {}, "V", "## Out of turns", "Answer.")
    assert (out["answer"], out["harvested"]) == (None, False)
    assert out["error"] == "RuntimeError: 503 from the provider"
    assert out["iters"] == 3 and out["history"] == [("c", "o")]
    assert "the call made after the last turn of the plan stage did not happen" in caplog.text


def test_run_stage_takes_a_prompt_refused_twice_on_the_one_extra_call(monkeypatch):
    monkeypatch.setattr(config, "HISTORY_MAX_CHAR", 0)
    llm = _RefusingLLM(["```python\nprint(1)\n```"], refuse_calls={2, 3})
    out = run_stage(RLM(llm, [], "I", max_turn=1), "REQ", {}, "V", "## Out of turns",
                    "Answer.")
    assert (out["answer"], out["harvested"], out["iters"]) == (None, False, 1)
    assert out["error"].startswith("PromptTooLong: ") and len(llm.prompts) == 3


def test_refuse_up_to_sends_the_call_back_a_bounded_number_of_times():
    seen = []
    guard = rlm_mod.refuse_up_to(lambda answer, count: seen.append(count) or "no", 2)
    assert [guard("a"), guard("a"), guard("a")] == ["no", "no", ""]
    assert seen == [0, 1]


def test_refuse_up_to_with_zero_never_asks_the_check():
    guard = rlm_mod.refuse_up_to(lambda answer, count: "no", 0)
    assert guard("a") == ""


# == what one turn's code may not do ========================================

def test_a_turn_that_does_not_finish_costs_that_turn_and_no_more(monkeypatch):
    """Nothing else in this stage bounds the block the model wrote."""
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    out = Repl({}, {}).run("print('before')\nwhile True:\n    pass", 1000)
    assert "before" in out and "TURN_TIMEOUT_SEC" in out


def _explore_like(reply_list: list[str]) -> dict:
    """Run a loop of its own inside the caller's turn, as explore() does."""
    stage = RLM(lambda prompt: reply_list.pop(0), [], "I", stage="explore", max_turn=3)
    return stage.run("REQ", {}, "V")


def test_a_loop_run_inside_a_turn_leaves_the_turns_limit_in_place(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    repl = Repl({"inner_reply_list": ["```python\nprint('inner')\n```",
                                      "```python\nCOMPLETE('[]')\n```"]},
                {"explore_like": _explore_like})
    out = repl.run("explore_like(inner_reply_list)\nimport time\ntime.sleep(3)\n"
                   "print('slept')", 1000)
    assert "TimeoutError: the turn did not finish inside TURN_TIMEOUT_SEC" in out
    assert "slept" not in out


def test_the_turns_limit_is_not_held_back_by_an_except_exception_on_its_way(monkeypatch):
    """A tool's loop turns every exception its model call raises into CallFailure."""
    import time
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)

    def slow_llm(prompt):
        time.sleep(3)
        return "```python\nCOMPLETE('[]')\n```"

    def explore_like() -> dict:
        return RLM(slow_llm, [], "I", stage="explore", max_turn=3).run("REQ", {}, "V")

    start = time.monotonic()
    out = Repl({}, {"explore_like": explore_like}).run(
        "print(explore_like())\nprint('the rest of the turn ran')", 1000)
    assert "TURN_TIMEOUT_SEC" in out and "the rest of the turn ran" not in out
    assert time.monotonic() - start < 2.5


def test_an_except_exception_in_the_models_code_does_not_hold_the_limit_back(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    out = Repl({}, {}).run("try:\n    while True:\n        pass\nexcept Exception:\n"
                           "    print('held')", 1000)
    assert "held" not in out and "TURN_TIMEOUT_SEC" in out


def test_a_turn_inside_the_limit_is_not_stopped(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 5)
    assert Repl({}, {}).run("print('done')", 1000).strip() == "done"


def test_the_limit_is_lifted_again_after_the_turn(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    Repl({}, {}).run("print('x')", 1000)
    import signal
    import time
    time.sleep(1.2)                       # the timer would ring here if it were still on
    assert signal.getsignal(signal.SIGALRM) in (signal.SIG_DFL, signal.SIG_IGN)


def _sleep_arg_list() -> list[str]:
    """Return the argument of every sleep process the system lists."""
    arg_list = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/cmdline", "rb") as cmd_file:
                    argv = cmd_file.read().split(b"\0")
            except OSError:
                continue
            if argv[0] == b"sleep":
                arg_list.append(argv[1].decode())
    return arg_list


def _limit_state() -> tuple:
    """Return what a block's limit sets in the process, as it is now."""
    wake_fd = signal.set_wakeup_fd(-1)
    signal.set_wakeup_fd(wake_fd)
    return (rlm_mod._deadline_depth, rlm_mod._tool_depth, wake_fd,
            signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL),
            sys.gettrace(), threading.active_count())


needs_process_list = pytest.mark.skipif(not os.path.isdir("/proc/self"),
                                        reason="the system lists no processes")


@needs_process_list
@pytest.mark.parametrize("command, arg_list", [
    ("sleep 30.7301", ["30.7301"]),
    ("(sleep 30.7302; echo late) & sleep 30.7303", ["30.7302", "30.7303"])])
def test_a_block_waiting_for_a_process_no_handler_can_end_is_stopped_after_the_grace(
        monkeypatch, command, arg_list):
    """os.system() comes back only when the process ends, so the processes the block
    started are killed, with the ones under them."""
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    monkeypatch.setattr(rlm_mod, "_TURN_GRACE_SEC", 0.5)
    state = _limit_state()
    start = time.monotonic()
    out = Repl({}, {}).run(f"import os\nos.system({command!r})\nprint('after')", 1000)
    assert time.monotonic() - start < 4
    assert "TimeoutError: the turn did not finish inside TURN_TIMEOUT_SEC (1s)" in out
    assert "after" not in out
    time.sleep(0.2)
    assert not set(arg_list) & set(_sleep_arg_list())
    assert _limit_state() == state


@pytest.mark.parametrize("code", [
    "import time\nwhile True:\n    try:\n        time.sleep(1)\n    except:\n        pass",
    "import time\nwhile True:\n    try: time.sleep(1)\n    except BaseException: pass",
    "import contextlib, time\nwhile True:\n"
    "    with contextlib.suppress(BaseException): time.sleep(1)",
    "import time\ndef work():\n    try:\n        time.sleep(1)\n    except:\n"
    "        return 1\nwhile True:\n    work()",
    "import time\ntry:\n    time.sleep(60)\nexcept:\n    print('held once')\n"
    "start = time.monotonic()\nwhile time.monotonic() - start < 30:\n    pass\n"
    "print('went on')",
])
def test_a_block_that_catches_the_limit_is_stopped_at_the_next_ring(monkeypatch, code):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    monkeypatch.setattr(rlm_mod, "_TURN_GRACE_SEC", 0.5)
    state = _limit_state()
    start = time.monotonic()
    out = Repl({}, {}).run(code, 1000)
    assert time.monotonic() - start < 3
    assert out.endswith(
        "TimeoutError: the turn did not finish inside TURN_TIMEOUT_SEC (1s)\n")
    assert "went on" not in out
    assert _limit_state() == state


@needs_process_list
def test_a_tool_going_on_at_a_later_ring_is_left_to_end_itself(monkeypatch):
    """The first ring ends the tool's wait; after it, a tool's own wait and the process
    it started are left alone."""
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    monkeypatch.setattr(rlm_mod, "_TURN_GRACE_SEC", 0.4)

    def wait_tool() -> int:
        return subprocess.run(["sleep", "2.7304"], check=False).returncode

    out = Repl({}, {"wait_tool": wait_tool}).run(
        "try:\n    wait_tool()\nexcept BaseException as e:\n    print(type(e).__name__)\n"
        "print('status', wait_tool())", 1000)
    assert out == "TurnTimeout\nstatus 0\n"
    assert rlm_mod._tool_depth == 0


@needs_process_list
def test_a_block_inside_a_tool_is_stopped_as_the_outer_block_is(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    monkeypatch.setattr(rlm_mod, "_TURN_GRACE_SEC", 0.5)
    repl = Repl({"inner_reply_list": [
        "```python\nimport os\nos.system('sleep 30.7305')\n```"]},
        {"explore_like": _explore_like})
    start = time.monotonic()
    out = repl.run("explore_like(inner_reply_list)\nprint('after')", 1000)
    assert time.monotonic() - start < 4
    assert "TURN_TIMEOUT_SEC" in out and "after" not in out
    time.sleep(0.2)
    assert "30.7305" not in _sleep_arg_list() and rlm_mod._tool_depth == 0


@needs_process_list
def test_a_signal_that_stops_the_run_ends_a_wait_for_a_process_after_the_grace(
        monkeypatch):
    def stop(signal_number, frame):
        raise KeyboardInterrupt("stopped")

    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 60)
    monkeypatch.setattr(rlm_mod, "_TURN_GRACE_SEC", 0.5)
    before = signal.signal(signal.SIGTERM, stop)
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM))
    try:
        timer.start()
        start = time.monotonic()
        with pytest.raises(KeyboardInterrupt, match="stopped"):
            Repl({}, {}).run("import os\nos.system('sleep 30.7306')", 1000)
        assert time.monotonic() - start < 4
    finally:
        timer.cancel()
        signal.signal(signal.SIGTERM, before)
    time.sleep(0.2)
    assert "30.7306" not in _sleep_arg_list()
    assert rlm_mod._deadline_depth == 0


@needs_process_list
def test_a_signal_sent_to_a_process_the_block_forked_is_not_taken_for_a_stop(monkeypatch):
    """The forked process holds the handlers of this one; its signals are its own."""
    def stop(signal_number, frame):
        raise KeyboardInterrupt("stopped")

    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 60)
    monkeypatch.setattr(rlm_mod, "_TURN_GRACE_SEC", 0.5)
    before = signal.signal(signal.SIGTERM, stop)
    try:
        out = Repl({}, {}).run(
            "import os, signal, subprocess, time\npid = os.fork()\n"
            "if pid == 0:\n    try:\n        time.sleep(30)\n    finally:\n"
            "        os._exit(0)\n"
            "time.sleep(0.2)\nos.kill(pid, signal.SIGTERM)\nos.waitpid(pid, 0)\n"
            "print('status', subprocess.run(['sleep', '1.7310']).returncode)", 1000)
    finally:
        signal.signal(signal.SIGTERM, before)
    assert out == "status 0\n"


def test_the_signals_the_command_line_stops_on_are_the_ones_the_limit_watches_for():
    from codetwine_loom import cli
    assert set(cli._STOP_SIGNAL_NAME_TUPLE) <= set(rlm_mod._STOP_SIGNAL_NAME_TUPLE)


@needs_process_list
def test_the_processes_started_before_a_block_are_not_the_ones_killed():
    keep = subprocess.Popen(["sleep", "30.7307"])
    try:
        keep_set = set(rlm_mod._child_dict()[os.getpid()])
        assert keep.pid in keep_set
        new = subprocess.Popen(["sh", "-c", "sleep 30.7308 & sleep 30.7309"])
        time.sleep(0.3)
        rlm_mod._kill_child(keep_set)
        assert new.wait(5) == -signal.SIGKILL
        time.sleep(0.2)
        assert keep.poll() is None
        assert not {"30.7308", "30.7309"} & set(_sleep_arg_list())
    finally:
        keep.kill()
        keep.wait()


def test_a_ring_raises_nothing_once_the_block_is_being_left(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 5)
    state = _limit_state()
    exit_frame = types.SimpleNamespace(
        f_code=rlm_mod._TurnDeadline.__exit__.__code__, f_back=sys._getframe())
    with rlm_mod.turn_deadline() as is_own:
        deadline = signal.getsignal(signal.SIGALRM).__self__
        assert is_own and deadline._on_alarm(signal.SIGALRM, exit_frame) is None
        with pytest.raises(rlm_mod.TurnTimeout):
            deadline._on_alarm(signal.SIGALRM, sys._getframe())
    assert deadline._on_alarm(signal.SIGALRM, sys._getframe()) is None
    assert _limit_state() == state


def test_a_ring_after_a_block_left_without_its_limit_lifted_lifts_it(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 5)
    state = _limit_state()

    def leave_open() -> object:
        deadline = rlm_mod.turn_deadline()
        deadline.__enter__()
        return deadline

    deadline = leave_open()
    assert rlm_mod._deadline_depth == 1
    assert deadline._on_alarm(signal.SIGALRM, sys._getframe()) is None
    assert _limit_state() == state


def test_no_limit_is_set_from_a_thread_other_than_the_main_one(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    state, seen_list = _limit_state(), []

    def run() -> None:
        with rlm_mod.turn_deadline() as is_own:
            seen_list.append((is_own, rlm_mod._deadline_depth,
                              signal.getitimer(signal.ITIMER_REAL)))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert seen_list == [(False, 0, (0.0, 0.0))] and _limit_state() == state


def _stop(signal_number, frame):
    """Stand in for the handler the command line stops a run with."""
    raise KeyboardInterrupt(f"stopped by {signal_number}")


@needs_process_list
@pytest.mark.parametrize("name, code", [
    ("SIGHUP", "import os\ndef tidy():\n    return 1\ntry:\n    os.system('sleep 1.9311')\n"
               "finally:\n    tidy()\nprint('after')"),
    ("SIGHUP", "import contextlib, os\nwith contextlib.ExitStack():\n"
               "    os.system('sleep 1.9312')\nprint('after')"),
    ("SIGHUP", "import os\nos.system('sleep 1.9313')\nprint('after')"),
    ("SIGTERM", "import os\nos.system('sleep 1.9314')\nprint('after')"),
    ("SIGTERM", "import os\ndef tidy():\n    return 1\ntry:\n    os.system('sleep 1.9315')\n"
                "finally:\n    tidy()\nprint('after')"),
])
def test_a_stop_that_waits_with_the_limit_for_a_block_ends_the_run_and_lifts_the_limit(
        monkeypatch, name, code):
    """The block is past its limit, waiting for a process, when the signal comes: the
    two are handed to the block one after the other once the process ends. The signal
    is what ends the block, and nothing of the limit is left once it is out."""
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    number = getattr(signal, name)
    before = signal.signal(number, _stop)
    state = _limit_state()
    timer = threading.Timer(1.4, os.kill, (os.getpid(), number))
    try:
        timer.start()
        with pytest.raises(KeyboardInterrupt, match=f"stopped by {int(number)}"):
            Repl({}, {}).run(code, 1000)
        assert _limit_state() == state and signal.getsignal(number) is _stop
        with rlm_mod.turn_deadline() as is_own:
            assert is_own
    finally:
        timer.cancel()
        signal.signal(number, before)


def test_a_stop_that_comes_while_the_limit_is_lifted_is_held_and_sent_again(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 5)
    seen_list = []

    def stop(signal_number, frame):
        seen_list.append((signal_number, rlm_mod._deadline_depth,
                          signal.getsignal(signal.SIGALRM)))
        raise KeyboardInterrupt("stopped")

    before = signal.signal(signal.SIGTERM, stop)
    state = _limit_state()
    try:
        with pytest.raises(KeyboardInterrupt, match="stopped"):
            with rlm_mod.turn_deadline():
                deadline = signal.getsignal(signal.SIGALRM).__self__
                assert signal.getsignal(signal.SIGTERM) == deadline._on_stop
                exit_frame = types.SimpleNamespace(
                    f_code=rlm_mod._TurnDeadline.__exit__.__code__,
                    f_locals={"self": deadline})
                assert deadline._on_stop(signal.SIGTERM, exit_frame) is None
                assert seen_list == []
        # The handler ran once the limit was lifted, as the signal's own handler
        assert seen_list == [(signal.SIGTERM, 0, state[3])]
        assert signal.getsignal(signal.SIGTERM) is stop and _limit_state() == state
    finally:
        signal.signal(signal.SIGTERM, before)


def test_a_ring_after_a_stop_raises_what_the_stop_raised(monkeypatch):
    """A block that held the stop back gets it again where it would get the limit."""
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 5)
    before = signal.signal(signal.SIGHUP, _stop)
    state = _limit_state()
    try:
        with rlm_mod.turn_deadline():
            deadline = signal.getsignal(signal.SIGALRM).__self__
            with pytest.raises(rlm_mod.TurnTimeout):
                deadline._on_alarm(signal.SIGALRM, sys._getframe())
            with pytest.raises(KeyboardInterrupt, match="stopped by"):
                deadline._on_stop(signal.SIGHUP, sys._getframe())
            for _ in range(2):
                with pytest.raises(KeyboardInterrupt, match="stopped by"):
                    deadline._on_alarm(signal.SIGALRM, sys._getframe())
        assert _limit_state() == state and signal.getsignal(signal.SIGHUP) is _stop
    finally:
        signal.signal(signal.SIGHUP, before)


def test_a_signal_with_no_handler_of_its_own_is_left_as_it_is_by_the_limit(monkeypatch):
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 5)
    before = signal.getsignal(signal.SIGHUP)
    assert before in (signal.SIG_DFL, signal.SIG_IGN)
    with rlm_mod.turn_deadline() as is_own:
        assert is_own and signal.getsignal(signal.SIGHUP) is before
    assert signal.getsignal(signal.SIGHUP) is before


# == what a block writes past sys.stdout ====================================

def _own_stream_dict(monkeypatch) -> dict:
    """Put the two streams behind TermStreams that do not write through descriptors
    1 and 2, as progress.set_up_output() leaves them, and return what they write to."""
    stream_dict = {name: io.StringIO() for name in ("stdout", "stderr")}
    for fd, (name, stream) in enumerate(stream_dict.items(), start=1):
        monkeypatch.setattr(sys, name, progress.TermStream(stream, apart_fd=fd))
    return stream_dict


@pytest.mark.parametrize("code, want", [
    ("import os\nos.system('echo from-system')", "from-system\n"),
    ("import subprocess\nsubprocess.run(['echo', 'from-run'])", "from-run\n"),
    ("import sys\nprint('to-first-stream', file=sys.__stdout__)", "to-first-stream\n"),
    ("import os\nos.write(1, b'to-fd-1\\n')\nos.write(2, b'to-fd-2\\n')",
     "to-fd-1\nto-fd-2\n"),
    ("import subprocess\nsubprocess.run(['sh', '-c', 'echo child-err >&2'])",
     "child-err\n"),
    ("import os\nprint('printed')\nos.system('echo from-child')\nprint('printed last')",
     "printed\nprinted last\nfrom-child\n"),
    ("import os\nos.system('printf \"\\\\033[2J\\\\377\"')", "\x1b[2J\ufffd"),
    ("import sys\nsys.stdout = sys.__stdout__\nprint('after the restore')",
     "after the restore\n"),
    ("import os\nos.system('echo a1; echo a2; echo b > /dev/stdout; echo c')",
     "a1\na2\nb\nc\n"),
    ("import os\nos.system('echo a1; echo warn > /dev/stderr; echo b >> /dev/stdout; "
     "echo c | tee /dev/stderr')", "a1\nwarn\nb\nc\nc\n"),
    ("import subprocess\nsubprocess.run(['sh', '-c', 'echo out1; echo err1 >&2; "
     "echo ERR > /dev/stderr; echo out2'])", "out1\nerr1\nERR\nout2\n"),
    ("x = 1", NO_OUTPUT_NOTE),
])
def test_what_a_block_writes_to_the_descriptors_is_the_turn_s_output(
        monkeypatch, capfd, code, want):
    """The output of a process the block started, and what the block wrote past
    sys.stdout, comes back to the model and does not reach the terminal."""
    own_stream = _own_stream_dict(monkeypatch)
    assert Repl({}, {}).run(code, 1000) == want
    os.write(1, b"descriptor 1 again\n")
    os.write(2, b"descriptor 2 again\n")
    assert capfd.readouterr() == ("descriptor 1 again\n", "descriptor 2 again\n")
    assert own_stream["stdout"].getvalue() == own_stream["stderr"].getvalue() == ""


@pytest.mark.parametrize("code, want", [
    ("import os\nos.system('echo before')\nraise ValueError('boom')", "ValueError: boom"),
    ("import os\nos.system('echo before')\nimport time\ntime.sleep(5)", "TimeoutError: "),
    ("import os\nos.system('echo before')\nimport sys\nsys.exit(3)", "SystemExit(3)"),
    ("import os\nos.system('echo before')\nos.close(1)\nos.close(2)", ""),
])
def test_the_descriptors_are_back_however_the_block_ended(monkeypatch, capfd, code,
                                                          want):
    _own_stream_dict(monkeypatch)
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    fd_count = len(os.listdir("/dev/fd"))
    out = Repl({}, {}).run(code, 1000)
    assert out.startswith("before\n") and want in out
    os.write(1, b"1 again\n")
    os.write(2, b"2 again\n")
    assert capfd.readouterr() == ("1 again\n", "2 again\n")
    assert len(os.listdir("/dev/fd")) == fd_count and rlm_mod._fd_output_depth == 0


def test_the_descriptors_are_back_after_the_call_that_ends_the_run(monkeypatch, capfd):
    _own_stream_dict(monkeypatch)
    with pytest.raises(CompleteCall) as call:
        Repl({}, {}).run("import os\nos.system('echo before')\nCOMPLETE('done')", 1000)
    assert call.value.output == "before\n"
    os.write(1, b"1 again\n")
    assert capfd.readouterr().out == "1 again\n"


def test_what_this_process_prints_while_a_block_runs_goes_where_it_went(monkeypatch,
                                                                        capfd):
    """The lines of this process are written through copies of the descriptors, so
    they are not taken in with what the block writes."""
    for name, fd in (("stdout", 1), ("stderr", 2)):
        monkeypatch.setattr(sys, name, open(fd, "w", encoding="utf-8", closefd=False))
    progress.set_up_output()
    own_stream = sys.stdout
    assert progress.apart_fd_list() == [1, 2] and own_stream.fileno() > 2

    def say_tool() -> None:
        own_stream.write("a line of this process\n")
        own_stream.flush()

    out = Repl({}, {"say_tool": say_tool}).run(
        "import os\nos.system('echo from-child')\nsay_tool()\nos.write(2, b'err\\n')",
        1000)
    assert out == "from-child\nerr\n"
    assert capfd.readouterr() == ("a line of this process\n", "")


def test_nothing_is_taken_in_while_this_process_writes_through_the_descriptors(capfd):
    assert progress.apart_fd_list() == []
    out = Repl({}, {}).run("import os\nos.system('echo to-the-terminal')", 1000)
    assert out == NO_OUTPUT_NOTE
    assert capfd.readouterr().out == "to-the-terminal\n"


def test_nothing_is_taken_in_on_a_thread_other_than_the_main_one(monkeypatch, capfd):
    _own_stream_dict(monkeypatch)
    out_list = []
    thread = threading.Thread(target=lambda: out_list.append(Repl({}, {}).run(
        "import os\nos.system('echo to-the-terminal')", 1000)))
    thread.start()
    thread.join()
    assert out_list == [NO_OUTPUT_NOTE]
    assert capfd.readouterr().out == "to-the-terminal\n"


def test_a_block_inside_a_tool_takes_in_what_it_wrote_itself(monkeypatch, capfd):
    _own_stream_dict(monkeypatch)

    def inner_tool() -> str:
        return Repl({}, {}).run("import os\nos.system('echo inner')", 1000)

    out = Repl({}, {"inner_tool": inner_tool}).run(
        "import os\nos.system('echo outer-1')\nprint(repr(inner_tool()))\n"
        "os.system('echo outer-2')", 1000)
    assert out == "'inner\\n'\nouter-1\nouter-2\n"
    assert capfd.readouterr() == ("", "") and rlm_mod._fd_output_depth == 0


def test_what_the_descriptors_took_in_is_read_back_up_to_a_limit(monkeypatch):
    _own_stream_dict(monkeypatch)
    monkeypatch.setattr(rlm_mod, "_FD_OUTPUT_MAX_BYTE", 10)
    out = Repl({}, {}).run("import os\nos.write(1, b'0123456789' * 5)", 1000)
    assert out == "0123456789" + CUT_NOTE.format(omitted=40)


def test_a_pipe_that_cannot_be_opened_takes_nothing_in(monkeypatch, capfd):
    def no_pipe(*args, **kwargs):
        raise OSError(24, "Too many open files")

    _own_stream_dict(monkeypatch)
    monkeypatch.setattr(progress, "PipeOutput", no_pipe)
    assert Repl({}, {}).run("import os\nos.system('echo to-the-terminal')\nprint('p')",
                            1000) == "p\n"
    assert capfd.readouterr().out == "to-the-terminal\n"


def test_only_a_descriptor_this_process_writes_past_is_pointed_elsewhere(monkeypatch,
                                                                        capfd):
    """Standard output was closed when the process started: what holds its number now
    is not standard output, and stays where it is while a block runs."""
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", progress.TermStream(io.StringIO(), apart_fd=2))
    out = Repl({}, {}).run("import os\nos.write(1, b'one\\n')\nos.write(2, b'two\\n')",
                           1000)
    assert out == "two\n"
    assert capfd.readouterr() == ("one\n", "")


def test_a_process_that_writes_more_than_a_pipe_holds_is_read_while_the_block_waits(
        monkeypatch):
    _own_stream_dict(monkeypatch)
    out = Repl({}, {}).run(
        "import os\nos.system(\"head -c 300000 /dev/zero | tr '\\\\0' x\")", 400000)
    assert out == "x" * 300000


def test_a_process_a_block_left_running_goes_on_writing_and_none_of_it_is_shown(
        monkeypatch, capfd):
    """The block is over before the process writes: the write works, and it reaches
    neither a later turn nor the terminal."""
    _own_stream_dict(monkeypatch)
    count = threading.active_count(), len(os.listdir("/dev/fd"))
    repl = Repl({}, {})
    start = time.monotonic()
    out = repl.run("import subprocess\nleft = subprocess.Popen(['sh', '-c', "
                   "'echo early; sleep 1; echo late && echo written >&2'])\n"
                   "import time\ntime.sleep(0.3)", 1000)
    assert out == "early\n" and time.monotonic() - start < 1
    assert repl.run("print(left.wait(10))", 1000) == "0\n"
    limit = time.monotonic() + 3
    while (threading.active_count(), len(os.listdir("/dev/fd"))) != count \
            and time.monotonic() < limit:
        time.sleep(0.05)
    assert (threading.active_count(), len(os.listdir("/dev/fd"))) == count
    assert capfd.readouterr() == ("", "")


def test_sys_exit_is_answered_rather_than_ending_the_run():
    """The only way to end a run is the call that ends it; exit() would leave no
    record, no verdict and an exit status of 0."""
    out = Repl({}, {}).run("print('before')\nimport sys\nsys.exit(0)", 1000)
    assert "before" in out and "SystemExit(0) is not allowed here" in out
    assert "COMPLETE()" in out


def test_a_bare_exit_is_answered_the_same_way():
    out = Repl({}, {}).run("exit()", 1000)
    assert "not allowed here" in out


def test_what_the_completing_turn_printed_is_kept():
    """A turn that wrote and ended in one block leaves the same record as any other."""
    llm = _ScriptedLLM(["```python\nprint('the edit landed')\nCOMPLETE('done')\n```"])
    r = RLM(llm, [], "I", max_turn=5).run("REQ", {}, "V")
    assert r["answer"] == "done"
    assert "the edit landed" in r["history"][0][1]
    assert "COMPLETE() was called: done" in r["history"][0][1]


def test_what_a_refused_completing_turn_printed_is_kept():
    """The turn after a refusal would otherwise start with everything it read gone."""
    llm = _ScriptedLLM(["```python\nprint('I read this')\nCOMPLETE('done')\n```",
                        "```python\nCOMPLETE('done again')\n```"])
    refusal_list = ["not yet", ""]
    r = RLM(llm, [], "I", max_turn=5).run(
        "REQ", {}, "V", on_complete=lambda answer: refusal_list.pop(0))
    assert "I read this" in r["history"][0][1] and "not yet" in r["history"][0][1]


# == What a stage logs while it goes ========================
def test_tool_call_text_counts_each_tool_in_the_order_it_is_first_called():
    code = "a = read_file(p)\nb = read_file(q)\nrun_test()\nprint(len(a))\nCOMPLETE('x')"
    assert rlm_mod.tool_call_text(code, ["run_test", "read_file", "COMPLETE"]) \
        == "read_file x2, run_test, COMPLETE"


def test_tool_call_text_leaves_out_a_name_that_is_not_a_tool():
    assert rlm_mod.tool_call_text("print(len('abc'))", ["read_file"]) == ""


def test_tool_call_text_is_empty_for_code_that_cannot_be_parsed():
    assert rlm_mod.tool_call_text("x = (", ["read_file"]) == ""


def test_a_stage_logs_its_start_each_turn_and_its_end(caplog):
    llm = _ScriptedLLM(["```python\nsample_tool('a')\nsample_tool('b')\n```",
                        "No code here.", "```python\nCOMPLETE('done')\n```"])
    with caplog.at_level("DEBUG", logger="codetwine_loom.rlm"):
        RLM(llm, [sample_tool], "I", stage="edit", max_turn=5).run("REQ", {}, "V")
    message_list = [r.getMessage() for r in caplog.records]
    assert message_list == [
        "the edit stage starts: at most 5 turns",
        "edit turn 1 of at most 5: sample_tool x2",
        "edit turn 2 of at most 5: no code (prose)",
        "edit turn 3 of at most 5: COMPLETE",
        "the edit stage ends after 3 of at most 5 turns: COMPLETE()"]


def test_the_turn_is_logged_before_its_code_runs(caplog):
    seen_list = []

    def look(x: str) -> str:
        """Record what the log holds when the tool is called."""
        seen_list.extend(r.getMessage() for r in caplog.records)
        return x

    llm = _ScriptedLLM(["```python\nlook('a')\n```"])
    with caplog.at_level("DEBUG", logger="codetwine_loom.rlm"):
        RLM(llm, [look], "I", stage="edit", max_turn=1).run("REQ", {}, "V")
    assert "edit turn 1 of at most 1: look" in seen_list


def test_the_turns_are_carried_only_when_every_line_is_asked_for(caplog):
    llm = _ScriptedLLM(["```python\nx = 1\n```"] * 2)
    with caplog.at_level("INFO", logger="codetwine_loom.rlm"):
        RLM(llm, [], "I", stage="plan", max_turn=2).run("REQ", {}, "V")
    assert [r.getMessage() for r in caplog.records] == [
        "the plan stage starts: at most 2 turns",
        "the plan stage ends after 2 of at most 2 turns: no turn left"]


def test_a_turn_that_calls_no_tool_says_so(caplog):
    llm = _ScriptedLLM(["```python\nx = 1\n```"])
    with caplog.at_level("DEBUG", logger="codetwine_loom.rlm"):
        RLM(llm, [], "I", stage="plan", max_turn=1).run("REQ", {}, "V")
    assert "plan turn 1 of at most 1: no tool call" in [
        r.getMessage() for r in caplog.records]


def test_a_stage_whose_call_did_not_happen_says_so(caplog):
    llm = _FailingLLM(["```python\nx = 1\n```"])
    with caplog.at_level("INFO", logger="codetwine_loom.rlm"):
        RLM(llm, [], "I", stage="edit", max_turn=5).run("REQ", {}, "V")
    assert caplog.records[-1].getMessage().startswith(
        "the edit stage ends after 1 of at most 5 turns: a call that did not happen (")


def test_a_json_answer_is_read_whole_however_long():
    text = '{"files": ["' + "a" * 100000 + '"]}'
    read_dict = rlm_mod.parse_json_answer(text, "the plan")
    assert read_dict["ok"] is True and len(read_dict["value"]["files"][0]) == 100000
    read_dict = rlm_mod.parse_json_answer("{no", "x")
    assert read_dict["ok"] is False and read_dict["error"].startswith("not valid JSON")


@pytest.mark.parametrize("text, drop_count", [
    ("a" * 500, 100), ("a" * 500, 1), ("a" * 500, 470), ("abc\n" * 200, 300),
    ("x" * 40 + "\n... (10 chars omitted)", 20), ("日本語の文。" * 100, 250)])
def test_a_text_cut_by_a_count_is_that_much_shorter_and_ends_with_the_marker(text, drop_count):
    from codetwine_loom import textcut
    out = textcut.cut_by(text, drop_count)
    assert len(json.dumps(out, ensure_ascii=False)) <= len(json.dumps(text, ensure_ascii=False)) \
        - drop_count
    assert textcut.CUT_NOTE_RE.search(out) and text.startswith(out.rpartition("\n")[0][:20])
    assert textcut.cut_by(text, 0) == text


def test_a_text_too_short_to_hold_the_marker_is_cut_to_nothing():
    from codetwine_loom import textcut
    assert textcut.cut_by("a" * 30, 25) == ""
    assert textcut.cut_by("a" * 40, 16) == "... (40 chars omitted)"
    entry_dict = {"path": "a.py", "change": "c" * 300, "anchor": "a" * 30}
    assert textcut.cut_entry_text(entry_dict, 200, ("path",)) == 0
    assert entry_dict["anchor"] == "a" * 30 and entry_dict["path"] == "a.py"
    assert len(entry_dict["change"]) <= 100 and entry_dict["change"].endswith("omitted)")


@pytest.mark.parametrize("reply, want", [
    ('<invoke_result_check = None\nprint(run_test())\nprint(work["tests"])',
     'print(run_test())\nprint(work["tests"])'),
    ("<invoke_tool>\nprint(run_test())", "print(run_test())"),
    ('<\n<parameter name="code">print(run_test())\nprint(work["left"])</parameter>\n</invoke>',
     'print(run_test())\nprint(work["left"])'),
    ('<invoke_result>\nr = replace_in_file("a.js", "x", "y")\n</invoke_result>',
     'r = replace_in_file("a.js", "x", "y")'),
    # What is left has to be code holding a call
    ("<invoke_result>\npass\n</invoke_result>", None),
    ('<invoke_result>\n<parameter name="check">Reviewing a.js now.</parameter>\n</invoke_result>',
     None),
    # A markup line inside the code is the code's own
    ('<tag>\nhtml = """\n<div>\n"""\nprint(len(html))', 'html = """\n<div>\n"""\nprint(len(html))')])
def test_an_unfenced_reply_runs_its_code_once_the_markup_lines_around_it_are_left_out(
        reply, want):
    assert extract_code(reply) == ((want, True) if want else (None, False))


@pytest.mark.parametrize("reply, want", [
    # Code starting on the line of a tag with a prefix, or with no name
    ('<ns:parameter name="python">r = replace_in_file("a.py", "x", "y")\nprint(run_test())'
     '</ns:parameter>\n</ns:invoke>',
     'r = replace_in_file("a.py", "x", "y")\nprint(run_test())'),
    ('<parameter="code">s0 = read_source_file("a.py")\nprint(s0)',
     's0 = read_source_file("a.py")\nprint(s0)'),
    ('<invoke><parameter name="code">print(f())</parameter></invoke>', "print(f())"),
    # A tag followed by a label alone is a line of markup
    ('<ns:parameter name="code">python\nprint(f())', "print(f())")])
def test_code_written_on_the_line_of_a_tag_is_kept(reply, want):
    assert extract_code(reply) == (want, True)


@pytest.mark.parametrize("reply, want", [
    # A markup call with no parameter, the code written straight inside it
    ('<invoke name="python">\nprint(f())\n</invoke>', "print(f())"),
    # A first line that is an element closed on the same line
    ("<parameter-length>0</parameter-length>\nprint(f())", "print(f())")])
def test_markup_with_no_parameter_around_code_is_taken_away(reply, want):
    assert extract_code(reply) == (want, True)


def test_a_closing_tag_that_ends_a_string_on_the_first_line_is_kept():
    reply = ('<ns:parameter name="python">new = """<label>Name</label>\n"""\n'
             'r = replace_in_file("a.html", "x", new)</ns:parameter>')
    assert extract_code(reply) == (
        'new = """<label>Name</label>\n"""\nr = replace_in_file("a.html", "x", new)', True)


def test_a_fence_line_left_at_the_end_of_markup_wrapped_code_is_taken_away():
    reply = '<invoke/\nprint(work["test_files"])\nprint(read_source_file("a.js"))\n```'
    assert extract_code(reply) == (
        'print(work["test_files"])\nprint(read_source_file("a.js"))', True)


@pytest.mark.parametrize("reply", ["python\nprint(f())\n```", "python\nprint(f())"])
def test_an_opening_fence_that_lost_its_backticks_leaves_its_label_out(reply):
    assert extract_code(reply) == ("print(f())", True)


@pytest.mark.parametrize("reply, want", [
    ('<\n<parameter name="code">print(work["targets"])\nprint(work["complete"])</parameter>\n'
     '</invoke>\n._', 'print(work["targets"])\nprint(work["complete"])'),
    ('<invoke_tool>\nprint(run_test())\n<invoke_result>\n(no output)\n</invoke_result>\nok',
     "print(run_test())")])
def test_what_follows_markup_wrapped_code_after_a_markup_line_is_left_out(reply, want):
    assert extract_code(reply) == (want, True)
