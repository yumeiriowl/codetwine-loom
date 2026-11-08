"""The two answering stages: one agent answering one question, and the run that puts
the agents' answers together.

Both build and run a fresh read-only RLM, so the LLM is substituted for these checks,
the same way tests/test_dispatch.py substitutes the splitting stage's.
"""

import re

import pytest

from codetwine_loom import answer, config, dispatch, plan
from codetwine_loom.tools import run
from codetwine_loom.languages import LANG_DICT


def _scripted_llm(*replies: str):
    """Build an LLM returning the given replies in order, repeating the last one."""
    seq = list(replies)

    def call(prompt: str) -> str:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return call


def _complete(text: str) -> str:
    """Return a turn that hands the text to COMPLETE()."""
    return f"```python\nCOMPLETE({text!r})\n```"


_ANSWERED = [{"n": 1, "request": "where is it read?", "attempt": 1,
              "answer": "in read_config, config.py", "exit_status": 0, "error": ""},
             {"n": 2, "request": "what happens without it?", "attempt": 2,
              "answer": "", "exit_status": 1,
              "error": "the run that answers handed no answer back"}]


@pytest.fixture
def stub_search(monkeypatch):
    """Skip the search: these tests are about what happens after it."""
    scope = {"files": [], "defs": [], "file_summaries": {}, "callers": {},
             "confidence": "high", "spread": 0}
    monkeypatch.setattr(answer, "search_scope", lambda request: scope)
    return scope


# == the instructions =======================================
@pytest.mark.parametrize("build", [answer.build_answer_instruction,
                                   answer.build_combine_instruction])
@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_the_instructions_build_with_no_injection_left_unresolved(build, ext):
    text = build(ext)
    assert text
    assert not re.search(r"\{[a-z_]+\}", text)


@pytest.mark.parametrize("build", [answer.build_answer_instruction,
                                   answer.build_combine_instruction])
def test_the_instructions_say_nothing_is_edited_and_end_in_complete(build):
    text = build("py")
    assert "no editing tools" in text and 'COMPLETE("<the answer>")' in text
    assert "replace_in_file" not in text


@pytest.mark.parametrize("build", [answer.build_answer_instruction,
                                   answer.build_combine_instruction])
def test_the_answer_is_asked_for_in_the_language_of_the_question(build):
    assert "in the language it was asked in" in build("py")


def test_the_reading_rules_are_the_splitting_stages(monkeypatch):
    """One text for how to read, so a rule added for one stage reaches the others."""
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 1234)
    monkeypatch.setattr(config, "RUN_COMMAND", [])
    rules = dispatch.read_rule("py", plan.answer_tool_list())
    assert "1234" in rules
    assert rules in answer.build_answer_instruction("py")
    assert rules in dispatch.build_dispatch_instruction("py")


@pytest.mark.parametrize("build", [answer.build_answer_instruction,
                                   answer.build_combine_instruction])
def test_the_instructions_do_not_describe_run_project_the_stages_are_not_handed(
        build, monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    assert "run_project" not in build("py")
    assert "run_project" in dispatch.build_dispatch_instruction("py")


def test_the_web_lines_follow_the_tools_the_answering_stages_are_handed(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 3)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    text = answer.build_answer_instruction("py")
    assert "web_search(" in text and "web_fetch(" in text
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 0)
    text = answer.build_answer_instruction("py")
    assert "web_search(" in text and "web_fetch(" not in text


def test_only_the_combining_stage_is_told_about_the_agents_answers():
    heading = "## Putting the agents' answers together"
    assert heading in answer.build_combine_instruction("py")
    assert heading not in answer.build_answer_instruction("py")


def test_the_combining_stage_is_told_where_to_read_and_where_not_to():
    text = answer.build_combine_instruction("py")
    assert "not read everything again" in text
    assert "where two answers disagree" in text


def test_the_answer_limit_is_injected_from_the_settings(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 777)
    assert "under 777 characters" in answer.build_answer_instruction("py")


# == the tools ==============================================
def test_the_tools_are_the_reading_ones_without_the_program(stub_search,
                                                            monkeypatch):
    handed = []

    class _RLM:
        def __init__(self, llm, tool_list, instruction, stage, max_turn):
            handed.append(tool_list)
            self.llm, self.tool_dict, self.instruction = llm, tool_list, instruction
            self.stage, self.max_turn = stage, max_turn

        def run(self, **kw):
            return {"answer": "x", "error": "", "outcome": "complete", "iters": 1,
                    "refused_completes": 0, "history": []}

    monkeypatch.setattr(answer, "RLM", _RLM)
    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: None)
    answer.run_answer("q")
    assert [[tool.__name__ for tool in tool_list] for tool_list in handed] == [
        [tool.__name__ for tool in plan.answer_tool_list()]]
    # A question changes nothing, and on the host the program would run in the user's
    # own working tree
    assert run.run_project not in handed[0]


# == render_answer_list ========================================
def test_the_block_holds_every_question_and_marks_the_one_with_no_answer():
    block = answer.render_answer_list(_ANSWERED)
    assert block.startswith("## What the agents answered")
    assert "1. where is it read?" in block and "answered: in read_config" in block
    assert "2. what happens without it?" in block
    assert "no answer came back: the run that answers handed no answer back" in block


def test_the_block_cuts_a_question_and_an_answer_to_their_settings(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 10)
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 10)
    block = answer.render_answer_list([{"n": 1, "request": "q" * 100, "answer": "a" * 100,
                                     "error": ""}])
    assert "q" * 11 not in block and "a" * 11 not in block


# == run_answer =============================================
def test_what_the_stage_does_to_scope_does_not_reach_the_search_result(stub_search,
                                                                      monkeypatch):
    stub_search.update(files=["a.py"], defs=[{"key": "a.py::f"}])
    turn = ('```python\nscope["files"].append("x.py")\nscope["defs"].clear()\nscope.pop("confidence")\n'
            'COMPLETE("in config.py")\n```')
    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: _scripted_llm(turn))
    out = answer.run_answer("どこで読んでいる?")
    assert out["answer"] == "in config.py" and out["scope"] is stub_search
    assert stub_search == {"files": ["a.py"], "defs": [{"key": "a.py::f"}],
                           "file_summaries": {}, "callers": {}, "confidence": "high",
                           "spread": 0}


def test_the_answer_comes_back_as_a_string_with_the_search_result(stub_search,
                                                                   monkeypatch):
    monkeypatch.setattr(answer, "make_llm",
                        lambda *a, **k: _scripted_llm(_complete("in config.py")))
    out = answer.run_answer("どこで読んでいる?")
    assert out["ok"] is True and out["answer"] == "in config.py"
    assert out["scope"] is stub_search and out["error"] == ""
    assert out["harvested"] is False and out["iters"] == 1


def test_the_question_is_what_the_model_reads(stub_search, monkeypatch):
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return _complete("here")

    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: llm)
    answer.run_answer("どこで読んでいる?")
    assert "どこで読んでいる?" in prompts[0]


def test_the_agent_runs_on_the_agents_settings(stub_search, monkeypatch):
    monkeypatch.setattr(config, "ANSWER_MODEL", "provider/agent")
    monkeypatch.setattr(config, "INTEGRATE_MODEL", "provider/combiner")
    monkeypatch.setattr(config, "ANSWER_MAX_TURN", 3)
    monkeypatch.setattr(config, "INTEGRATE_MAX_TURN", 99)
    asked, prompts = [], []

    def make(model, max_tokens, setting_name="MEDIUM_MODEL", **kw):
        asked.append((model, setting_name))
        return lambda prompt: prompts.append(prompt) or _complete("here")

    monkeypatch.setattr(answer, "make_llm", make)
    answer.run_answer("q")
    assert asked == [("provider/agent", "ANSWER_MODEL")]
    assert "## Turn 1/3" in prompts[0]


def test_a_stage_that_runs_out_of_turns_is_asked_once_for_the_answer(stub_search,
                                                                     monkeypatch):
    monkeypatch.setattr(config, "RLM_MAX_TURN", 2)
    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: _scripted_llm(
        "```python\nprint('still reading')\n```",
        "```python\nprint('still reading')\n```",
        "it is read in config.py"))
    out = answer.run_answer("q")
    assert out["ok"] is True and out["harvested"] is True
    assert out["answer"] == "it is read in config.py"


def test_the_harvest_is_the_answering_stages_own_setting(stub_search, monkeypatch):
    monkeypatch.setattr(config, "ANSWER_MAX_TURN", 1)
    monkeypatch.setattr(config, "ANSWER_LATE_ANSWER", False)
    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: _scripted_llm(
        "```python\nprint('still reading')\n```"))
    out = answer.run_answer("q")
    assert out["ok"] is False and out["answer"] is None
    assert "no COMPLETE() within 1 turns" in out["error"]


def test_a_run_whose_call_never_went_through_says_so(stub_search, monkeypatch):
    def llm(prompt):
        raise RuntimeError("the broker did not answer")

    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: llm)
    out = answer.run_answer("q")
    assert out["ok"] is False and "the broker did not answer" in out["error"]


def test_an_empty_answer_is_no_answer(stub_search, monkeypatch):
    monkeypatch.setattr(answer, "make_llm",
                        lambda *a, **k: _scripted_llm(_complete("  ")))
    assert answer.run_answer("q")["ok"] is False


# == run_combine ============================================
def test_the_combining_run_reads_the_request_and_every_answer(stub_search,
                                                              monkeypatch):
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return _complete("both")

    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: llm)
    out = answer.run_combine("設定はどこで読んでいる?", _ANSWERED)
    assert out["answer"] == "both"
    assert "設定はどこで読んでいる?" in prompts[0]
    assert "## What the agents answered" in prompts[0]
    assert "in read_config, config.py" in prompts[0]
    assert "no answer came back" in prompts[0]


def test_the_combining_run_runs_on_the_merging_stages_settings(stub_search,
                                                              monkeypatch):
    monkeypatch.setattr(config, "ANSWER_MODEL", "provider/agent")
    monkeypatch.setattr(config, "INTEGRATE_MODEL", "provider/combiner")
    monkeypatch.setattr(config, "RLM_MAX_TURN", 99)
    monkeypatch.setattr(config, "INTEGRATE_MAX_TURN", 3)
    asked, prompts = [], []

    def make(model, max_tokens, setting_name="MEDIUM_MODEL", **kw):
        asked.append((model, setting_name))
        return lambda prompt: prompts.append(prompt) or _complete("here")

    monkeypatch.setattr(answer, "make_llm", make)
    answer.run_combine("q", _ANSWERED)
    assert asked == [("provider/combiner", "INTEGRATE_MODEL")]
    assert "## Turn 1/3" in prompts[0]


def test_the_search_is_made_on_the_request_alone(monkeypatch):
    searched = []
    scope = {"files": [], "defs": [], "file_summaries": {}, "callers": {},
             "confidence": "high", "spread": 0}
    monkeypatch.setattr(answer, "search_scope",
                        lambda request: searched.append(request) or scope)
    monkeypatch.setattr(answer, "make_llm",
                        lambda *a, **k: _scripted_llm(_complete("here")))
    answer.run_combine("設定はどこで読んでいる?", _ANSWERED)
    assert searched == ["設定はどこで読んでいる?"]


# == an answer over the limit is sent back once before it is cut ==
def test_an_answer_a_cut_left_is_not_sent_back(monkeypatch):
    """The gate measures as the stage cuts: an answer copied as a cut left it passes."""
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 10)
    left = answer.cut("x" * 30, 10)
    assert len(left) > 10 and answer._over_the_limit(left) == ""
    assert answer._over_the_limit("x" * 11) != ""


def test_an_answer_over_the_limit_is_sent_back_once(stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 50)
    prompts = []
    replies = iter([_complete("x" * 80), _complete("x" * 40)])

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: llm)
    out = answer.run_answer("q")
    assert out["answer"] == "x" * 40 and out["iters"] == 2
    assert "80 characters and the limit is 50" in prompts[1]


def test_an_answer_still_over_after_being_sent_back_is_cut(stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 50)
    monkeypatch.setattr(answer, "make_llm",
                        lambda *a, **k: _scripted_llm(_complete("x" * 80)))
    out = answer.run_answer("q")
    assert out["iters"] == 2 and len(out["answer"]) < 80
    assert "chars omitted" in out["answer"]


def test_an_answer_within_the_limit_is_not_sent_back(stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 50)
    monkeypatch.setattr(answer, "make_llm",
                        lambda *a, **k: _scripted_llm(_complete("x" * 50)))
    assert answer.run_answer("q")["iters"] == 1


def test_a_limit_that_is_off_sends_nothing_back(stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 0)
    monkeypatch.setattr(answer, "make_llm",
                        lambda *a, **k: _scripted_llm(_complete("x" * 5000)))
    out = answer.run_answer("q")
    assert out["iters"] == 1 and len(out["answer"]) == 5000


def test_a_limit_of_zero_leaves_the_block_whole(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 0)
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 0)
    block = answer.render_answer_list([{"n": 1, "request": "q" * 100, "answer": "a" * 100,
                                     "error": ""}])
    assert "q" * 100 in block and "a" * 100 in block and "omitted" not in block


# == the answering stage's own settings =====================

def test_the_turn_limit_is_the_answering_stages_own(stub_search, monkeypatch):
    """The stage may run on another model, and what the editing stage is given is not
    what a stage that only reads needs."""
    seen = {}

    class _RLM:
        def __init__(self, llm, tool_list, instruction, stage, max_turn):
            seen["max_turn"] = max_turn
            self.llm, self.tool_dict, self.instruction = llm, tool_list, instruction
            self.stage, self.max_turn = stage, max_turn

        def run(self, **kw):
            return {"answer": "x", "error": "", "outcome": "complete", "iters": 1,
                    "refused_completes": 0, "history": []}

    monkeypatch.setattr(config, "ANSWER_MAX_TURN", 7)
    monkeypatch.setattr(answer, "RLM", _RLM)
    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: None)
    answer.run_answer("q")
    assert seen["max_turn"] == 7


# == an answer that is empty ================================

def test_an_empty_answer_is_sent_back_once(stub_search, monkeypatch):
    """What leaves the session is COMPLETE()'s argument, so an empty one hands over
    nothing."""
    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: _scripted_llm(
        "```python\nCOMPLETE('')\n```", "```python\nCOMPLETE('it is in config.py')\n```"))
    out = answer.run_answer("q")
    assert out["ok"] is True and out["answer"] == "it is in config.py"
    assert out["refused_completes"] == 1


def test_an_answer_still_empty_is_not_reported_as_running_out_of_turns(stub_search,
                                                                       monkeypatch):
    monkeypatch.setattr(config, "ANSWER_LATE_ANSWER", False)
    monkeypatch.setattr(answer, "make_llm", lambda *a, **k: _scripted_llm(
        "```python\nCOMPLETE('')\n```", "```python\nCOMPLETE('   ')\n```"))
    out = answer.run_answer("q")
    assert out["ok"] is False
    assert "empty answer" in out["error"] and "within" not in out["error"]


# == the limit that is off ==================================

def test_no_length_is_asked_for_when_the_limit_is_off(monkeypatch):
    """0 is no limit, and the guard and the cut both read it that way."""
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 0)
    text = answer.build_answer_instruction("py")
    assert "under 0 characters" not in text and "Keep it under" not in text
