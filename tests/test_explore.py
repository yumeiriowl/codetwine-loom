"""The explorer.

explore() builds and runs a fresh read-only RLM, so the LLM is substituted for these
checks. What is substituted is make_llm, under the name tools/explore.py imported.
"""

import json
import re

import pytest

from codetwine_loom import config, rlm
from codetwine_loom.tools import explore

FINDINGS_JSON = ('[{"file": "cfg.py", "lines": [10, 42], '
                 '"signature": "def load_config(path):", '
                 '"why": "reads the config"}]')


def _scripted_llm(*replies: str):
    """Build an LLM returning the given replies in order, repeating the last one."""
    seq = list(replies)

    def call(prompt: str) -> str:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return call


@pytest.fixture
def complete_llm(monkeypatch):
    """Set up an explorer that COMPLETE()s a JSON array on its first turn."""
    llm = _scripted_llm(f"```python\nCOMPLETE({FINDINGS_JSON!r})\n```")
    monkeypatch.setattr(explore, "make_llm", lambda *a, **k: llm)
    # The request stays Japanese: a Japanese request is what production takes
    explore.set_context("設定の読み込みを直す", {"files": [], "defs": []})
    return llm


def test_findings_come_back_with_their_4_keys(complete_llm):
    out = explore.explore("config loading")
    assert out["ok"] is True
    assert out["findings"] == [{"file": "cfg.py", "lines": [10, 42],
                                "signature": "def load_config(path):",
                                "why": "reads the config"}]
    assert out["calls_left"] == config.EXPLORE_MAX_CALL - 1


def test_an_empty_array_is_still_ok_true(monkeypatch):
    """Finding nothing means COMPLETE('[]'), which is not an error."""
    monkeypatch.setattr(explore, "make_llm",
                        lambda *a, **k: _scripted_llm("```python\nCOMPLETE('[]')\n```"))
    explore.set_context("req", {})
    assert explore.explore("nothing here") == {"ok": True, "findings": [],
                                               "calls_left":
                                                   config.EXPLORE_MAX_CALL - 1}


def test_no_completion_is_ok_false(monkeypatch):
    """Even when the explorer never COMPLETEs, and answers the one extra call with
    another block, the main loop keeps going."""
    monkeypatch.setattr(config, "EXPLORE_MAX_TURN", 2)
    monkeypatch.setattr(explore, "make_llm",
                        lambda *a, **k: _scripted_llm("```python\nprint(1)\n```"))
    explore.set_context("req", {})
    out = explore.explore("config loading")
    assert out["ok"] is False
    assert out["error"].startswith("not valid JSON") and "print(1)" in out["raw"]
    assert out["calls_left"] == config.EXPLORE_MAX_CALL - 1


def test_an_explorer_that_used_every_turn_is_asked_once_for_what_it_found(monkeypatch):
    prompt_list = []
    reply_list = ["```python\nprint(1)\n```", "```python\nprint(2)\n```", FINDINGS_JSON]

    def llm(prompt):
        prompt_list.append(prompt)
        return reply_list[len(prompt_list) - 1]

    monkeypatch.setattr(config, "EXPLORE_MAX_TURN", 2)
    monkeypatch.setattr(config, "EXPLORE_MAX_RESULT", 4)
    monkeypatch.setattr(explore, "make_llm", lambda *a, **k: llm)
    explore.set_context("req", {})
    out = explore.explore("config loading")
    assert out["ok"] is True and out["findings"][0]["file"] == "cfg.py"
    assert len(prompt_list) == 3
    late = prompt_list[2]
    # The call runs nothing: the history, then the note, then the closing line
    assert late.index("### Turn 2 output") < late.index("## Out of turns")
    assert "single JSON array" in late and "At most 4 entries" in late
    assert late.rstrip().endswith("Answer with the JSON array and nothing else.")
    assert "## Out of turns" not in prompt_list[1]


def test_an_explorer_that_ended_itself_is_not_asked_again(monkeypatch):
    prompt_list = []

    def llm(prompt):
        prompt_list.append(prompt)
        return f"```python\nCOMPLETE({FINDINGS_JSON!r})\n```"

    monkeypatch.setattr(config, "EXPLORE_MAX_TURN", 1)
    monkeypatch.setattr(explore, "make_llm", lambda *a, **k: llm)
    explore.set_context("req", {})
    assert explore.explore("config loading")["ok"] is True
    assert len(prompt_list) == 1


def test_an_extra_call_that_did_not_happen_is_answered_as_a_stop(monkeypatch):
    """The main loop gets the refusal with its budget, not an exception."""
    call_list = []

    def llm(prompt):
        call_list.append(prompt)
        if len(call_list) > 1:
            raise RuntimeError("the broker did not answer")
        return "```python\nprint(1)\n```"

    monkeypatch.setattr(config, "EXPLORE_MAX_TURN", 1)
    monkeypatch.setattr(explore, "make_llm", lambda *a, **k: llm)
    explore.set_context("req", {})
    out = explore.explore("config loading")
    assert out == {"ok": False, "calls_left": config.EXPLORE_MAX_CALL - 1,
                   "error": "the explorer stopped early: RuntimeError: the broker did "
                            "not answer"}


def test_raw_is_attached_when_the_reply_is_not_json(monkeypatch):
    monkeypatch.setattr(
        explore, "make_llm",
        lambda *a, **k: _scripted_llm("```python\nCOMPLETE('not json')\n```"))
    explore.set_context("req", {})
    out = explore.explore("config loading")
    assert out["ok"] is False
    assert out["raw"] == "not json"


def test_a_call_past_max_calls_is_ok_false(monkeypatch, complete_llm):
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 3)
    for left in (2, 1, 0):
        assert explore.explore("aspect")["calls_left"] == left
    out = explore.explore("aspect")
    assert out["ok"] is False
    assert "call limit" in out["error"]


def test_reset_explore_calls_restores_the_budget(monkeypatch, complete_llm):
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 1)
    explore.explore("aspect")
    assert explore.explore("aspect")["ok"] is False
    explore.reset_explore_count()
    assert explore.explore("aspect")["ok"] is True


def test_the_explorer_gets_no_writing_tools():
    """Building the change and running the tests is the main loop's job alone."""
    names = {t.__name__ for t in explore.EXPLORE_TOOL_LIST}
    assert names.isdisjoint({"apply_patch", "run_test", "git_diff_stat",
                             "revert_file"})


def test_the_explorers_tools_are_the_6_read_only_ones():
    assert {t.__name__ for t in explore.EXPLORE_TOOL_LIST} == {
        "read_source_file", "read_definition", "grep_source",
        "graph_search", "get_use_of_file", "search_again"}


def test_a_search_budget_of_0_takes_search_again_and_its_lines_away(monkeypatch):
    assert explore.explore_tool_list() == explore.EXPLORE_TOOL_LIST
    assert "search_again(" in explore.build_explore_instruction("py")
    monkeypatch.setattr(config, "SEARCH_AGAIN_MAX_CALL", 0)
    assert [t.__name__ for t in explore.explore_tool_list()] == [
        "read_source_file", "read_definition", "grep_source", "graph_search",
        "get_use_of_file"]
    text = explore.build_explore_instruction("py")
    assert "search_again" not in text
    assert "before the last file.\n\nWhen you are done" in text


def test_the_explorer_is_handed_the_list_its_budgets_leave(monkeypatch):
    handed = []
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 1)
    monkeypatch.setattr(config, "SEARCH_AGAIN_MAX_CALL", 0)
    monkeypatch.setattr(explore, "make_llm", lambda *a, **kw: (lambda p: ""))
    monkeypatch.setattr(explore.RLM, "run", lambda self, **kw: handed.append(
        list(self.tool_dict)) or {
            "answer": "[]", "outcome": "complete", "iters": 1, "history": [],
            "error": ""})
    explore.reset_explore_count()
    explore.explore("error handling")
    assert handed == [["read_source_file", "read_definition", "grep_source",
                       "graph_search", "get_use_of_file"]]


def test_the_explorer_instructions_have_no_editing_or_test_sections():
    text = explore.build_explore_instruction("py")
    assert "apply_patch" not in text
    assert "run_test" not in text
    assert "Editing steps" not in text


def test_the_explorer_instructions_inject_the_language_and_the_settings(monkeypatch):
    monkeypatch.setattr(config, "EXPLORE_ANSWER_MAX_CHAR", 999)
    text = explore.build_explore_instruction("java")
    assert "a Java\nrepository" in text
    assert "under 999 characters" in text
    # The JSON example's braces are escaped and no placeholder is left
    assert "{{" not in text and "{explore" not in text


def test_the_explorer_does_not_return_its_history_to_the_main_loop(complete_llm):
    """All that reaches the main prompt is the findings."""
    assert "history" not in explore.explore("config loading")


def test_findings_over_the_char_budget_are_read_whole_then_trimmed_from_the_end(
        monkeypatch):
    """A list that is valid JSON is never lost to the budget; the tail of it is."""
    monkeypatch.setattr(config, "EXPLORE_MAX_RESULT", 10)
    many = "[" + ", ".join('{"file": "f%d.py", "lines": [1, 2], "why": "x"}' % i
                           for i in range(5)) + "]"
    monkeypatch.setattr(config, "EXPLORE_ANSWER_MAX_CHAR", len(many) - 1)
    out = explore.parse_explore_answer(many)
    assert out["ok"] is True
    assert [f["file"] for f in out["findings"]] == ["f0.py", "f1.py", "f2.py", "f3.py"]
    monkeypatch.setattr(config, "EXPLORE_ANSWER_MAX_CHAR", 0)
    assert len(explore.parse_explore_answer(many)["findings"]) == 5


def test_a_first_finding_that_alone_is_over_the_budget_is_cut_and_stays(monkeypatch):
    """Findings there were are never handed back as none found."""
    monkeypatch.setattr(config, "EXPLORE_ANSWER_MAX_CHAR", 300)
    out = explore.parse_explore_answer([
        {"file": "a.py", "lines": [1, 2], "signature": "s" * 5000, "why": "w" * 100},
        {"file": "b.py", "lines": [1, 2], "signature": "def g()", "why": "w"}])
    assert out["ok"] is True and [f["file"] for f in out["findings"]] == ["a.py"]
    assert 0 < len(out["findings"][0]["signature"]) < 300
    assert out["findings"][0]["signature"].startswith("ssss")
    # The text that was cut says so, with the marker every cut leaves
    assert re.search(r"\n\.\.\. \(\d+ chars omitted\)$", out["findings"][0]["signature"])
    assert out["findings"][0]["why"] == "w" * 100
    assert len(json.dumps(out["findings"], ensure_ascii=False)) <= 300
    # One no text of which can be cut far enough is a refusal, with the budget on it
    out = explore.parse_explore_answer([{"file": "a.py", "lines": list(range(400))}])
    assert out["ok"] is False and "calls_left" in out and out["error"] == (
        "the explorer's findings are over the 300 characters they may take, even with "
        "the first one alone and its texts cut")
    assert explore.parse_explore_answer([])["findings"] == []


@pytest.mark.parametrize("budget, signature, why", [
    (140, "def main(argv):", "main() builds the result \n... (38 chars omitted)"),
    (100, "def main(argv):", ""), (90, "", ""), (76, "", "")])
def test_the_path_of_a_finding_cut_to_fit_is_kept_whole(monkeypatch, budget, signature,
                                                        why):
    monkeypatch.setattr(config, "EXPLORE_ANSWER_MAX_CHAR", budget)
    out = explore.parse_explore_answer([
        {"file": "src/app/main.py", "lines": [10, 24], "signature": "def main(argv):",
         "why": "main() builds the result that the request wants written as JSON"}])
    assert out["findings"] == [{"file": "src/app/main.py", "lines": [10, 24],
                                "signature": signature, "why": why}]
    assert len(json.dumps(out["findings"], ensure_ascii=False)) <= budget


def test_a_finding_whose_path_alone_is_over_the_budget_is_refused(monkeypatch):
    monkeypatch.setattr(config, "EXPLORE_ANSWER_MAX_CHAR", 75)
    out = explore.parse_explore_answer([
        {"file": "src/app/main.py", "lines": [10, 24], "signature": "def main(argv):",
         "why": "the place"}])
    assert out["ok"] is False and out["error"] == (
        "the explorer's findings are over the 75 characters they may take, even with "
        "the first one alone and its texts cut")


def test_more_findings_than_the_cap_are_cut(monkeypatch):
    """The instructions state a cap; what reaches the main prompt is what enforces it."""
    monkeypatch.setattr(config, "EXPLORE_MAX_RESULT", 2)
    monkeypatch.setattr(config, "EXPLORE_ANSWER_MAX_CHAR", 10_000)
    many = "[" + ", ".join('{"file": "a.py", "lines": [1, 2], "why": "x"}'
                           for _ in range(5)) + "]"
    out = explore.parse_explore_answer(many)
    assert out["ok"] is True and len(out["findings"]) == 2


def test_the_stated_cap_is_the_one_that_is_enforced(monkeypatch):
    monkeypatch.setattr(config, "EXPLORE_MAX_RESULT", 4)
    assert "At most 4 entries" in explore.build_explore_instruction("py")


# == An explorer that handed nothing over ====================

def test_an_explorer_whose_call_did_not_happen_says_so_to_the_model():
    out = explore.parse_explore_answer(None, "RuntimeError: the broker did not answer")
    assert out["ok"] is False
    assert out["error"] == ("the explorer stopped early: RuntimeError: the broker did "
                            "not answer")
    assert "turn limit" not in out["error"]


def test_an_explorer_that_ended_itself_with_nothing_is_not_said_to_have_stopped_early():
    out = explore.parse_explore_answer(None, "", rlm.COMPLETE_KEY)
    assert out["ok"] is False and "calls_left" in out
    assert out["error"] == ("the explorer handed nothing over: COMPLETE() was called "
                            "with an empty answer")
    # Without a call that failed and without an end of its own, the turns ran out
    assert explore.parse_explore_answer(None)["error"] == (
        f"the explorer handed nothing over: no COMPLETE() within "
        f"{config.EXPLORE_MAX_TURN} turns")


def test_an_explorer_that_calls_complete_with_nothing_twice_gives_the_model_a_reason(
        monkeypatch):
    """The first COMPLETE(None) is sent back once and the second is taken."""
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 1)
    monkeypatch.setattr(explore, "make_llm",
                        lambda *a, **kw: (lambda prompt: "```python\nCOMPLETE(None)\n```"))
    explore.reset_explore_count()
    out = explore.explore("error handling")
    assert out == {"ok": False, "calls_left": 0,
                   "error": "the explorer handed nothing over: COMPLETE() was called "
                            "with an empty answer"}


def test_the_budget_is_on_the_answer_of_an_explorer_that_handed_nothing_over():
    assert "calls_left" in explore.parse_explore_answer(
        None, "RuntimeError: the broker did not answer")


def test_the_stages_error_reaches_the_reader(monkeypatch):
    """explore() hands parse_explore_answer() what the loop said, not just the answer."""
    seen = []
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 1)
    monkeypatch.setattr(explore, "parse_explore_answer",
                        lambda answer, error="", outcome=None: seen.append(error)
                        or {"ok": False})
    monkeypatch.setattr(explore, "make_llm", lambda *a, **kw: (lambda p: ""))
    monkeypatch.setattr(explore.RLM, "run", lambda self, **kw: {
        "answer": None, "outcome": None, "iters": 1, "history": [],
        "error": "RuntimeError: the broker did not answer"})
    explore.reset_explore_count()
    explore.explore("error handling")
    assert seen == ["RuntimeError: the broker did not answer"]


def test_what_the_explorer_does_to_scope_does_not_reach_the_search_result(monkeypatch):
    scope = {"files": ["a.py"], "defs": [{"key": "a.py::f"}], "confidence": "high"}
    turn = ('```python\nscope["files"].append("x.py")\nscope["defs"].clear()\nscope.pop("confidence")\nCOMPLETE("[]")\n```')
    monkeypatch.setattr(explore, "make_llm", lambda *a, **k: lambda prompt: turn)
    explore.set_context("req", scope)
    assert explore.explore("config loading")["ok"] is True
    assert scope == {"files": ["a.py"], "defs": [{"key": "a.py::f"}],
                     "confidence": "high"}


def test_unreadable_findings_are_sent_back_once_before_the_call_is_spent(monkeypatch):
    prompts = []
    replies = iter(["```python\nCOMPLETE('not json')\n```",
                    f"```python\nCOMPLETE({FINDINGS_JSON!r})\n```"])

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(explore, "make_llm", lambda *a, **k: llm)
    explore.set_context("req", {})
    out = explore.explore("config loading")
    assert out["ok"] is True and len(prompts) == 2
    assert "the findings could not be read as JSON" in prompts[1]
