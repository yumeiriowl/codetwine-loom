"""Assembling the instructions, rendering the scope, and the tool list.

The injections are lang_name, primary_ext, path_example, test_command,
test_path, test_target_note, max_patch_round, max_output_char,
cut_note_shape, explore_max_call, read_limit_note and tool_result.
These tests check that none of them is hard-coded (which would make changing a setting
stop changing the behaviour).
"""

import inspect
import logging
import pathlib
import re
import subprocess
import sys

import pytest

from conftest import git_repo
from codetwine_loom import agent, config, history, integrate
from codetwine_loom import rlm as rlm_mod
from codetwine_loom.languages import LANG_DICT, PATH_EXAMPLE_DICT, SIGNATURE_EXAMPLE_DICT
from codetwine_loom.rlm import COMPLETE_KEY
from codetwine_loom.tools import edit, explore, source, web


def test_the_instructions_build_for_all_11_extensions():
    """A gap in LANG_NAME_DICT or PATH_EXAMPLE_DICT surfaces here as a KeyError."""
    assert set(agent.LANG_NAME_DICT) == set(LANG_DICT)
    assert set(PATH_EXAMPLE_DICT) == set(LANG_DICT)
    for ext in LANG_DICT:
        config.set_primary_ext(ext)
        assert agent.build_instruction(ext)
    config.set_primary_ext("py")


def test_no_injection_is_left_unresolved():
    """No leftover placeholder such as {max_output_char} reaches the output.

    The return-shape section (## What the tools return) contains braces of its own,
    such as {"ok", "path"}, so the check looks for the placeholder format rather than
    for braces.
    """
    text = agent.build_instruction("py")
    assert not re.search(r"\{[a-z_]+\}", text)


def test_the_instructions_describe_delete_file_and_its_return(monkeypatch):
    text = agent.build_instruction("py")
    assert "- delete_file(path) deletes a file" in text
    assert "keep its tests and\n     point them at the new code" in text
    assert re.search(r'- delete_file → \{"ok", "path", "stat".*"stat" is "deleted"', text)
    assert ("Tests that test only code the request asked you to delete go with it"
            in " ".join(text.split()))
    monkeypatch.setattr(config, "RUN_TEST", False)
    no_test_text = agent.build_instruction("py")
    assert "- delete_file(path) deletes a file" in no_test_text
    assert "Tests that test only code" not in no_test_text


def test_the_merge_stage_deletes_only_the_tests_of_code_a_side_was_asked_to_delete():
    assert ("Tests that test only code a side was asked to delete go with it"
            in " ".join(integrate._INTEGRATE_TEST_NOTE.split()))


def test_the_output_limit_is_injected_from_the_settings(monkeypatch):
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 1234)
    monkeypatch.setattr(config, "READ_MAX_CHAR", 1000)
    text = agent.build_instruction("py")
    assert "1234 characters" in text
    # the default of 4000 is not hard-coded anywhere
    assert "4000" not in text


def test_the_read_limit_is_injected_from_the_settings(monkeypatch):
    monkeypatch.setattr(config, "READ_MAX_CHAR", 2500)
    text = agent.build_instruction("py")
    assert "at most 2500 characters per call" in text


def test_read_max_chars_of_0_restores_the_whole_file_wording(monkeypatch):
    """With both caps off, the instructions must not still assume a cut."""
    monkeypatch.setattr(config, "READ_MAX_CHAR", 0)
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 0)
    text = agent.build_instruction("py")
    assert "returns the whole file as a Python string" in text
    assert "characters per call" not in text


@pytest.mark.parametrize("read_max_char", [0, 2500])
def test_no_turn_output_limit_leaves_no_sentence_about_the_limit(monkeypatch,
                                                                read_max_char):
    monkeypatch.setattr(config, "READ_MAX_CHAR", read_max_char)
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 0)
    text = agent.build_instruction("py")
    assert "the limit" not in text
    assert "- read_definition returns one definition instead\n  — prefer it." in text
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 6000)
    assert "one definition instead, which is far less than the limit" in \
        agent.build_instruction("py")


def test_the_read_limit_named_is_the_one_the_tool_applies(monkeypatch):
    """READ_MAX_CHAR of 0 under a turn limit still cuts at read_limit(), and the
    instructions say that number rather than promising whole files."""
    monkeypatch.setattr(config, "READ_MAX_CHAR", 0)
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 5000)
    text = agent.build_instruction("py")
    assert f"at most {source.read_limit()} characters per call" in text
    assert "returns the whole file" not in text
    monkeypatch.setattr(config, "READ_MAX_CHAR", 9000)
    assert f"at most {source.read_limit()} characters per call" in agent.build_instruction("py")


def test_the_patch_rounds_are_injected_from_the_settings(monkeypatch):
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 7)
    assert "7 edit-then-test rounds" in agent.build_instruction("py")


def test_the_summary_is_asked_for_in_the_language_of_the_request():
    """What COMPLETE() carries is committed, printed and reported as it is."""
    assert "in the language of the request" in agent.build_instruction("py")


def test_the_instructions_state_the_output_limit_and_how_to_continue():
    """The instructions tell the model the output limit.

    The last line of a return value cut at READ_MAX_CHAR says how to continue, and
    the instructions point at that.
    """
    text = agent.build_instruction("py")
    assert "Output limit" in text
    assert "call it again with that number" in text


def test_the_instructions_name_the_clip_marker_the_repl_actually_uses():
    """Naming a marker the REPL does not emit teaches the model the wrong shape."""
    from codetwine_loom.rlm import CUT_NOTE, CUT_NOTE_SHAPE
    assert CUT_NOTE_SHAPE in agent.build_instruction("py")
    assert CUT_NOTE.format(omitted=7).strip().startswith(
        CUT_NOTE_SHAPE.split("N")[0])


def test_the_language_name_and_path_example_change_per_language():
    config.set_primary_ext("java")
    text = agent.build_instruction("java")
    assert "a Java repository" in text
    assert "src/main/java/com/example/Config.java" in text
    config.set_primary_ext("py")


def test_dry_run_appends_its_section(monkeypatch):
    plain = agent.build_instruction("py")
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    dry = agent.build_instruction("py")
    assert dry == plain + agent.dry_run_note()


@pytest.mark.parametrize("targets, want", [
    ("path", "Called without arguments it runs everything under that directory"),
    ("separator", "forwarded after `--`"),
    ("none", "always runs the whole suite; call it without arguments"),
])
def test_all_3_test_targets_modes_reach_the_instructions(monkeypatch, targets, want):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", targets)
    assert want in agent.build_instruction("py")


# == Scope rules ===================

def test_the_instructions_no_longer_say_to_read_every_file():
    """The instructions do not tell the model to read every file in scope."""
    text = agent.build_instruction("py")
    assert 'Read every file in scope["files"]' not in text


def test_the_instructions_put_read_definition_and_summaries_first():
    text = agent.build_instruction("py")
    assert "read_definition(name)" in text
    assert 'scope["file_summaries"]' in text


def test_the_explore_budget_is_injected_from_the_settings(monkeypatch):
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 4)
    assert "You get\n  4 calls" in agent.build_instruction("py")


def test_search_again_does_not_appear_in_the_instructions():
    """Naming a tool the main loop does not have makes it try and lose a turn."""
    assert "search_again" not in agent.build_instruction("py")


# == The tool list =========================================================

def _tool_names(monkeypatch):
    """Take the names create_agent() handed to the RLM, without building an LLM."""
    captured = {}

    class _FakeRLM:
        def __init__(self, llm, tool_list, instruction, **kwargs):
            captured["names"] = [t.__name__ for t in tool_list]

    monkeypatch.setattr(agent, "RLM", _FakeRLM)
    monkeypatch.setattr(agent, "make_llm", lambda *a, **k: (lambda p: ""))
    agent.create_agent()
    return captured["names"]


def test_the_main_loop_gets_14_tools_and_15_with_an_endpoint(monkeypatch):
    """14 by default: the web question is handed over only with an endpoint to send
    it to."""
    names = _tool_names(monkeypatch)
    assert len(names) == 14
    assert names.index("delete_file") == names.index("create_file") + 1
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    assert len(_tool_names(monkeypatch)) == 15


def test_the_web_question_goes_with_the_budget(monkeypatch):
    """With an endpoint, a budget of 0 is what takes the tool away."""
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    assert "web_search" in _tool_names(monkeypatch)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    names = _tool_names(monkeypatch)
    assert "web_search" not in names
    assert len(names) == 14


def test_no_endpoint_takes_the_tool_away(monkeypatch):
    """A budget with no endpoint behind it would describe a tool every call of which
    can only be refused."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 3)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "")
    assert "web_search" not in _tool_names(monkeypatch)


def test_the_main_loop_gets_neither_search_again_nor_apply_patch(monkeypatch):
    """The main loop is handed neither search_again nor apply_patch."""
    names = _tool_names(monkeypatch)
    assert "search_again" not in names
    assert "apply_patch" not in names
    assert "read_definition" in names
    assert "explore" in names
    assert "replace_in_file" in names
    assert "replace_line_range" in names
    assert "create_file" in names
    assert "run_project" in names


def test_a_run_budget_of_0_takes_run_project_and_its_step_away(monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    assert "run_project" in agent.build_instruction("py")
    monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", 0)
    names = _tool_names(monkeypatch)
    assert "run_project" not in names and len(names) == 13
    assert names[-2:] == ["git_diff_stat", "revert_file"]
    assert agent.run_note() == ""
    assert "run_project" not in agent.build_instruction("py")


def test_the_check_step_names_run_project_only_when_it_is_handed_over(monkeypatch):
    monkeypatch.setattr(agent.smoke, "is_active", lambda: True)
    monkeypatch.setattr(agent.smoke, "command_line", lambda: "launcher --flag")
    assert agent.check_note().endswith(
        "   It only tells you the program still starts. Whether it now does what the "
        "request\n   asked for is still yours to check with run_project.\n")
    monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", 0)
    assert agent.check_note().endswith(
        "   It only tells you the program still starts.\n")
    assert "run_project" not in agent.check_note()


def test_the_check_step_says_complete_is_sent_back_only_while_the_guard_is_on(
        monkeypatch):
    monkeypatch.setattr(agent.smoke, "is_active", lambda: True)
    monkeypatch.setattr(agent.smoke, "command_line", lambda: "launcher --flag")
    monkeypatch.setattr(config, "SMOKE_COMPLETE_GUARD", True)
    assert ("do not call COMPLETE() while it\n   is failing - a COMPLETE() is sent back "
            "once for it.\n") in agent.check_note()
    monkeypatch.setattr(config, "SMOKE_COMPLETE_GUARD", False)
    assert "do not call COMPLETE() while it\n   is failing.\n" in agent.check_note()
    assert "sent back" not in agent.check_note()


def test_the_instructions_explain_when_to_use_each_editing_tool():
    """The line-number tool reaches the model in a form it can choose."""
    text = agent.build_instruction("py")
    assert "replace_line_range(path, start_line, end_line, new_text)" in text
    assert "replace_in_file(path, old_text, new_text)" in text
    assert "create_file(path, text)" in text
    assert "re-read the file" in text


def test_apply_patch_does_not_appear_in_the_instructions():
    assert "apply_patch" not in agent.build_instruction("py")


# == The shape of the return values =======================================

def test_every_tool_handed_over_has_its_return_shape_written_out(monkeypatch):
    """rlm.tool_line() omits the return shape, so it only reaches the instructions."""
    text = agent.build_instruction("py")
    for name in _tool_names(monkeypatch):
        assert f"- {name} → " in text


def test_the_3_keys_that_were_guessed_wrong_are_named(monkeypatch):
    """The instructions name hits' path, the absence of code on failure and insertions."""
    text = agent.render_tool_result()
    assert '"path", "line", "text"' in text
    assert 'NO "code" key' in text
    assert '"path", "insertions", "deletions"' in text


def test_the_return_shapes_are_addressable_by_tool_name():
    """They are held as data so the tests can compare them to the real values."""
    assert agent.TOOL_RESULT_KEY_DICT["run_test"][0] == (
        "status", "failed", "failed_excerpt", "targets", "output")
    assert agent.TOOL_RESULT_KEY_DICT["grep_source"][1]["hits"] == (
        "path", "line", "text", "before", "after")


# == render_scope ==========================================================

SCOPE = {
    "confidence": "high",
    "spread": 2,
    "files": ["cfg.py"],
    "defs": [{"key": "cfg.py::load_config", "lines": [10, 42],
              "signature": "def load_config(path):"}],
    "file_summaries": {"cfg.py": "This file reads and writes settings."},
    "callers": {"cfg.py::load_config": ["main.py::main"]},
}


def test_render_scope_prints_a_definition_as_range_plus_first_line():
    text = agent.render_scope(SCOPE)
    assert "  cfg.py::load_config  [10-42]" in text
    assert "    def load_config(path):" in text


def test_render_scope_prints_the_file_summaries():
    text = agent.render_scope(SCOPE)
    assert "File summaries:" in text
    assert "This file reads and writes settings." in text


def test_render_scope_omits_the_section_when_there_are_no_summaries():
    scope = dict(SCOPE, file_summaries={})
    assert "File summaries:" not in agent.render_scope(scope)


def test_render_scope_puts_spread_in_the_heading():
    assert "spread: 2" in agent.render_scope(SCOPE)


# == Harvesting the answer when the turns run out ==
class _EmptyRLM:
    """A stand-in editing stage that uses every turn without ever calling COMPLETE()."""

    instruction = "INSTR"
    tool_dict = {}
    stage = "edit"
    max_turn = 50
    call = 1

    def __init__(self, history_list=None):
        self.history = history_list if history_list is not None else [("code one", "output one")]
        self.prompt = ""

    def fold(self, history_list):
        return history_list

    def llm(self, prompt: str) -> str:
        self.prompt = prompt
        return "edited codetwine/pipeline.py: called the cache first"

    def run(self, request, variable_dict, var_doc, turn_note=None, on_complete=None,
            after_turn=None):
        # The same keys RLM.run() returns for a loop that used every turn, so a caller
        # reading one this stand-in leaves out fails here rather than in a real run
        return {"answer": None, "outcome": None, "iters": self.max_turn,
                "history": self.history, "error": "",
                "truncated_replies": 0, "truncated_lost": 0, "empty_replies": 0,
                "bare_code_turns": 0, "refused_completes": 0}


@pytest.fixture
def _no_search(monkeypatch):
    """Skip the search: these tests are about what happens after the loop."""
    scope = {"files": [], "defs": [], "file_summaries": {}, "callers": {},
             "confidence": "high", "spread": 0}
    monkeypatch.setattr(agent, "search_scope", lambda request: scope)
    monkeypatch.setattr(agent.explore, "set_context", lambda request, scope: None)
    monkeypatch.setattr(config, "TWO_STAGE", False)


def test_a_stage_that_runs_out_of_turns_is_asked_once_what_it_changed(_no_search):
    rlm = _EmptyRLM()
    result = agent.run_edit(rlm, "差分解析にして")
    assert result["harvested"] is True
    assert result["answer"] == "edited codetwine/pipeline.py: called the cache first"


def test_a_stage_that_stopped_early_is_not_asked_what_it_changed(_no_search,
                                                                 monkeypatch):
    """The call that would ask it is the one that did not go through."""
    rlm = _EmptyRLM()
    monkeypatch.setattr(rlm, "run", lambda *a, **k: {
        "answer": None, "outcome": None, "iters": 1, "history": [],
        "error": "RuntimeError: no call",
        "truncated_replies": 0, "truncated_lost": 0, "empty_replies": 0,
        "bare_code_turns": 0, "refused_completes": 0})
    result = agent.run_edit(rlm, "差分解析にして")
    assert result["harvested"] is False and result["answer"] is None
    assert rlm.prompt == ""


def test_the_run_carries_the_verdict_the_record_was_judged_with(_no_search):
    """The exit status follows it, so it travels with the result rather than being
    read back out of the history file."""
    result = agent.run_edit(_EmptyRLM(), "差分解析にして")
    assert result["verdict"] == history.VERDICT_FAIL
    assert "turns ran out" in result["why"] or "no edit landed" in result["why"]


def test_the_one_extra_call_of_the_editing_stage_carries_its_state(_no_search,
                                                                    monkeypatch):
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    monkeypatch.setattr(agent.plan, "change_path_set", set)

    class _ReadingRLM(_EmptyRLM):
        """A stand-in whose loop reads one file on its last turn."""

        def run(self, request, variable_dict, var_doc, turn_note=None, on_complete=None,
                after_turn=None):
            turn_note(self.max_turn, self.max_turn)
            integrate._STAGE_DICT["read_dict"]["a.py"] = integrate.stage_turn()
            return super().run(request, variable_dict, var_doc)

    rlm = _ReadingRLM()
    agent.run_edit(rlm, "差分解析にして")
    state = ("## State (turn 50/50)\n\n"
             "Files you have changed, as of this turn: none.\n\n"
             "Files you have read, as of this turn:\n"
             "  a.py  on turn 50\n\n"
             "## Out of turns")
    assert state in rlm.prompt
    assert rlm.prompt.index("output one") < rlm.prompt.index("## State")


def test_the_harvest_can_be_turned_off(_no_search, monkeypatch):
    monkeypatch.setattr(config, "EDIT_LATE_ANSWER", False)
    result = agent.run_edit(_EmptyRLM(), "差分解析にして")
    assert result["answer"] is None and result["harvested"] is False


def test_the_harvest_prompt_says_the_turns_are_gone_and_carries_the_history(_no_search):
    rlm = _EmptyRLM()
    agent.run_edit(rlm, "差分解析にして")
    assert "## Out of turns" in rlm.prompt
    assert "code one" in rlm.prompt and "output one" in rlm.prompt
    # It comes after the history, like every turn note, and it asks for no code block
    assert rlm.prompt.index("output one") < rlm.prompt.index("## Out of turns")
    assert rlm.prompt.rstrip().endswith("Answer with the summary and nothing else.")


def test_a_run_that_reached_complete_is_not_asked_again(_no_search, monkeypatch):
    rlm = _EmptyRLM()
    monkeypatch.setattr(rlm, "run", lambda *a, **k: {
        "answer": "done", "outcome": COMPLETE_KEY, "iters": 3, "history": [],
        "truncated_replies": 0,
        "truncated_lost": 0, "empty_replies": 0, "bare_code_turns": 0,
        "refused_completes": 0})
    result = agent.run_edit(rlm, "差分解析にして")
    assert result["answer"] == "done" and result["harvested"] is False
    assert rlm.prompt == ""


# == carrying the previous run over ============
class _CompletingRLM(_EmptyRLM):
    """A stand-in editing stage that records the request and reaches COMPLETE()."""

    def run(self, request, variable_dict, var_doc, turn_note=None, on_complete=None,
            after_turn=None):
        self.request = request
        return {"answer": "done", "outcome": COMPLETE_KEY, "iters": 2, "history": [],
                "truncated_replies": 0, "truncated_lost": 0, "empty_replies": 0,
                "bare_code_turns": 0, "refused_completes": 0}


@pytest.fixture
def _history_in(tmp_path, monkeypatch):
    """Put the history under tmp_path and report a repository with one changed file."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(agent.source, "project_name", "codetwine")
    monkeypatch.setattr(config, "EDIT_HISTORY", True)
    monkeypatch.setattr(history.edit, "change_by_kind_dict",
                        lambda: {"added": [], "modified": ["a.py"], "deleted": []})
    monkeypatch.setattr(history.edit, "write_path_set", lambda: {"a.py"})
    monkeypatch.setattr(history.edit, "patch_count_dict",
                        lambda: {"ok": 2, "ng": 0, "ng_after_ok": 0})
    monkeypatch.setattr(history.edit, "last_test_ok", lambda: None)


def test_a_first_run_leaves_the_prompt_exactly_as_it_was(_no_search, _history_in):
    """A first run's request reaches the model with no history block in front of it."""
    rlm = _CompletingRLM()
    agent.run_edit(rlm, "差分解析にして")
    assert rlm.request.startswith("差分解析にして")


def test_the_run_is_recorded(_no_search, _history_in):
    agent.run_edit(_CompletingRLM(), "差分解析にして")
    attempts = history.load()["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["did"] == "done" and attempts[0]["files"]["modified"] == ["a.py"]


def test_the_next_run_is_handed_the_unfinished_change(_no_search, _history_in):
    agent.run_edit(_CompletingRLM(), "差分解析にして")
    rlm = _CompletingRLM()
    agent.run_edit(rlm, "差分解析にして")
    assert rlm.request.startswith("## Unfinished change from the previous run")
    assert "差分解析にして" in rlm.request


def test_the_search_is_never_run_over_the_previous_summary(_no_search, _history_in,
                                                           monkeypatch):
    """Expanding the query over the block would search for the summary's words."""
    seen = []
    monkeypatch.setattr(agent, "search_scope",
                        lambda request: seen.append(request) or {
                            "files": [], "defs": [], "file_summaries": {},
                            "callers": {}, "confidence": "high", "spread": 0})
    agent.run_edit(_CompletingRLM(), "差分解析にして")
    agent.run_edit(_CompletingRLM(), "差分解析にして")
    assert seen == ["差分解析にして", "差分解析にして"]


def test_the_history_off_changes_neither_the_prompt_nor_the_file(_no_search,
                                                                 _history_in,
                                                                 monkeypatch):
    monkeypatch.setattr(config, "EDIT_HISTORY", False)
    rlm = _CompletingRLM()
    agent.run_edit(rlm, "差分解析にして")
    assert rlm.request.startswith("差分解析にして")
    assert history.load()["attempts"] == []


def test_a_run_that_ended_with_an_error_is_recorded_and_the_error_is_raised(
        _no_search, _history_in, monkeypatch):
    """The patch is written from what the repository holds however the run ended, so a
    run that left no record would leave a patch nothing accounts for."""
    rlm = _CompletingRLM()

    def boom(*args, **kwargs):
        raise RuntimeError("the broker did not make the call")

    monkeypatch.setattr(rlm, "run", boom)
    with pytest.raises(RuntimeError):
        agent.run_edit(rlm, "差分解析にして")
    attempts = history.load()["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["verdict"] == history.VERDICT_FAIL
    assert attempts[0]["signals"]["error"].startswith("RuntimeError")


def test_a_run_that_was_stopped_is_recorded_and_the_stop_goes_on(
        _no_search, _history_in, monkeypatch):
    rlm = _CompletingRLM()

    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(rlm, "run", stop)
    with pytest.raises(KeyboardInterrupt):
        agent.run_edit(rlm, "差分解析にして")
    attempts = history.load()["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["verdict"] == history.VERDICT_FAIL
    assert attempts[0]["signals"]["error"] == "the run was stopped"
    assert history.open_attempt()["n"] == 1
    carry_on = _CompletingRLM()
    agent.run_edit(carry_on, "差分解析にして")
    assert carry_on.request.startswith("## Unfinished change from the previous run")


def test_a_stop_that_says_why_is_recorded_with_what_it_says(_no_search, _history_in,
                                                            monkeypatch):
    rlm = _CompletingRLM()

    def stop(*args, **kwargs):
        raise KeyboardInterrupt("the run was stopped by a signal")

    monkeypatch.setattr(rlm, "run", stop)
    with pytest.raises(KeyboardInterrupt):
        agent.run_edit(rlm, "差分解析にして")
    assert history.load()["attempts"][0]["signals"]["error"] == \
        "the run was stopped by a signal"


def test_a_write_that_went_round_the_tools_is_said_even_with_the_history_off(
        _no_search, _history_in, monkeypatch, caplog):
    monkeypatch.setattr(config, "EDIT_HISTORY", False)
    monkeypatch.setattr(agent.edit, "no_tool_path_set", lambda: {"by_hand.py"})
    with caplog.at_level(logging.WARNING):
        agent.run_edit(_CompletingRLM(), "差分解析にして")
    assert "by_hand.py" in caplog.text


def test_a_first_run_reports_no_carry_over(_no_search, _history_in):
    rlm = _CompletingRLM()
    result = agent.run_edit(rlm, "差分解析にして")
    assert result["carried_over"] == 0


def test_a_later_run_reports_how_much_was_carried_over(_no_search, _history_in):
    """A later run reports a carried_over count above 0."""
    agent.run_edit(_CompletingRLM(), "差分解析にして")
    result = agent.run_edit(_CompletingRLM(), "差分解析にして")
    assert result["carried_over"] > 0


# The names of a particular target repository, which none of the examples the
# instructions carry may hold. "python" and "probe.py" stand for a Python script shown
# to a session of another language
_TARGET_SPECIFIC = ("codetwine", "process_all_files", "get_config_value",
                    "main.py", "--project-dir", "project_knowledge", "probe.py",
                    "qa_tools")

# This tool's own name, which holds "codetwine" and is taken out before the check
_OWN_NAME = re.compile(r"codetwine[-_]loom")


def without_own_name(text: str) -> str:
    """Return the text with this tool's own name taken out."""
    return _OWN_NAME.sub("", text)

# A line range with three or more digits in it was copied out of a real file: an example
# range is small enough to read as an example. "[145-249]" and "[412, 478]" are caught
# here, which a check of identifiers alone does not catch
_REAL_LINE_RANGE = re.compile(r"\[\s*\d{3,}\s*[-,]")


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_nothing_the_model_reads_names_a_particular_repository(ext):
    """An example out of one repository is read as an instruction about this one.

    It covers every text model_text_dict() collects, not the three instruction sets
    alone: the tools' errors and the notes the machine adds to a turn reach the model
    just as directly, and that is where the leftovers were.
    """
    for label, text in agent.model_text_dict(ext).items():
        for name in _TARGET_SPECIFIC:
            assert name not in without_own_name(text), \
                f"{name} reaches the model for {ext} in {label}"
        found = _REAL_LINE_RANGE.search(text)
        assert not found, f"{found.group(0)} reaches the model for {ext} in {label}"


def test_the_model_facing_texts_hold_the_four_instruction_sets():
    """The collection is what the check above runs on, so it must not come back thin."""
    texts = agent.model_text_dict("py")
    for label in ("agent.build_instruction", "plan.build_plan_instruction",
                  "dispatch.build_dispatch_instruction",
                  "explore.build_explore_instruction"):
        assert texts[label]
    # One entry per tool any stage is handed, from the same lists the stages use.
    # Not main_tool_list() alone: search_again is rendered into the planning and splitting
    # prompts and the editing loop never sees it, so reading only that list left its
    # line out of every check here
    assert sum(k.startswith("tool_line.") for k in texts) == len(agent.all_tool_list())
    assert "tool_line.search_again" in texts
    assert source.search_again not in agent.main_tool_list()


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_the_signature_example_belongs_to_the_language(ext):
    """A Python def shown to a Java session is an example of the wrong thing."""
    assert SIGNATURE_EXAMPLE_DICT[ext] in explore.build_explore_instruction(ext)


# == one ending, and the guards on it ==========
class _RecordingRLM:
    """A stand-in editing stage that ends the run the way the test asks for."""

    instruction = "INSTR"
    tool_dict = {}
    stage = "edit"
    max_turn = 20
    call = 1

    def __init__(self, answer="done", outcome=COMPLETE_KEY):
        self.answer = answer
        self.outcome = outcome
        self.on_complete = None

    def run(self, request, variable_dict, var_doc, turn_note=None, on_complete=None,
            after_turn=None):
        self.on_complete = on_complete
        return {"answer": self.answer, "outcome": self.outcome, "iters": 1,
                "history": [], **dict(rlm_mod.RUN_COUNT_DICT)}


def test_the_instructions_offer_no_second_ending():
    """The stage in front of the agents decides whether a request is a question, so
    this loop has one way out and it is COMPLETE()."""
    text = agent.build_instruction("py")
    assert "ANSWER(" not in text
    assert 'Call COMPLETE(' in text


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_no_language_is_offered_a_second_ending(ext):
    assert "ANSWER(" not in agent.build_instruction(ext)


def test_the_guards_are_chained_and_the_first_refusal_wins():
    seen = []

    def first(answer):
        seen.append("first")
        return ""

    def second(answer):
        seen.append("second")
        return "not yet"

    def third(answer):
        seen.append("third")
        return "nor this"

    assert agent.complete_guard_chain(first, second, third)("done") == "not yet"
    assert seen == ["first", "second"]


def test_a_guard_that_is_off_is_left_out():
    assert agent.complete_guard_chain(None, None) is None
    only = agent.complete_guard_chain(None, lambda answer: "no")
    assert only("done") == "no"


def test_the_editing_loop_is_handed_the_chained_guard(_no_search):
    rlm = _RecordingRLM()
    agent.run_edit(rlm, "この関数は何をしている?")
    # smoke's guard is off without SMOKE_ARG and there is no plan, so nothing is
    # chained and the loop is handed None rather than a guard that always accepts
    assert rlm.on_complete is None


def test_run_edit_reports_that_the_run_ended_itself(_no_search):
    rlm = _RecordingRLM(answer="it already does that")
    result = agent.run_edit(rlm, "空のときに落ちないようにして")
    assert result["outcome"] == COMPLETE_KEY
    assert result["answer"] == "it already does that"


def test_what_complete_was_handed_is_kept_as_text(_no_search, _history_in):
    value = {"changed": ["a.py"], "tests": "passed"}
    result = agent.run_edit(_RecordingRLM(answer=value), "差分解析にして")
    assert result["answer"] == str(value) and result["harvested"] is False
    assert history.load()["attempts"][0]["did"] == \
        "{'changed': ['a.py'], 'tests': 'passed'}"
    assert agent.run_edit(_RecordingRLM(answer=["a", 1]), "r")["answer"] == "['a', 1]"
    assert agent.run_edit(_RecordingRLM(answer=7), "r")["answer"] == "7"


def test_a_run_that_ended_itself_with_none_is_one_that_ended_itself(_no_search,
                                                                    _history_in):
    rlm = _RecordingRLM(answer=None)
    rlm.llm = lambda prompt: pytest.fail("the one extra call was made")
    result = agent.run_edit(rlm, "差分解析にして")
    assert result["answer"] == "" and result["harvested"] is False
    assert result["outcome"] == COMPLETE_KEY
    entry = history.load()["attempts"][0]
    assert entry["signals"]["completed"] is True and entry["did"] == ""


# == the deadline note of the editing stage ==================
class _NoteTakingRLM(_EmptyRLM):
    """A stand-in that keeps the turn_note it was handed."""

    def run(self, request, variable_dict, var_doc, turn_note=None, on_complete=None,
            after_turn=None):
        self.turn_note = turn_note
        return {"answer": "done", "outcome": COMPLETE_KEY, "iters": 1, "history": [],
                **dict(rlm_mod.RUN_COUNT_DICT)}


def test_the_editing_stage_is_told_to_hand_over_on_its_last_turns(_no_search,
                                                                   monkeypatch):
    """The other three stages count down; a run that used every turn is judged
    unfinished whatever it wrote, so this one does too."""
    monkeypatch.setattr(config, "EDIT_DEADLINE_TURN", 3)
    rlm = _NoteTakingRLM()
    agent.run_edit(rlm, "差分解析にして")
    assert rlm.turn_note(1, 20) == ""
    assert "3 turns left" in rlm.turn_note(18, 20)
    assert "COMPLETE" in rlm.turn_note(20, 20)


def test_the_deadline_note_can_be_turned_off(_no_search, monkeypatch):
    monkeypatch.setattr(config, "EDIT_DEADLINE_TURN", 0)
    rlm = _NoteTakingRLM()
    agent.run_edit(rlm, "差分解析にして")
    assert rlm.turn_note(20, 20) == ""


def test_the_editing_stage_reads_what_it_read_and_changed_every_turn(_no_search,
                                                                     monkeypatch):
    monkeypatch.setattr(config, "EDIT_DEADLINE_TURN", 0)
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    rlm = _NoteTakingRLM()
    agent.run_edit(rlm, "差分解析にして")
    integrate._STAGE_DICT["read_dict"]["a.py"] = 1
    assert rlm.turn_note(2, 20) == ("## State (turn 2/20)\n\n"
                                    "Files you have changed, as of this turn: none.\n\n"
                                    "Files you have read, as of this turn:\n"
                                    "  a.py  on turn 1")


def test_a_dry_run_leaves_the_changed_files_out_of_the_state(_no_search, monkeypatch):
    monkeypatch.setattr(config, "EDIT_DEADLINE_TURN", 0)
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    rlm = _NoteTakingRLM()
    agent.run_edit(rlm, "差分解析にして")
    integrate._STAGE_DICT["read_dict"]["a.py"] = 1
    assert rlm.turn_note(2, 20) == ("## State (turn 2/20)\n\n"
                                    "Files you have read, as of this turn:\n"
                                    "  a.py  on turn 1")


def test_the_editing_loop_keeps_what_its_reading_tools_read(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    tool_dict = agent.create_agent().tool_dict
    assert list(tool_dict) == [tool.__name__ for tool in agent.main_tool_list()]
    integrate.reset_read()
    tool_dict["read_source_file"]("no/such/file.py")
    tool_dict["read_source_file"]("a.py")
    assert [entry_dict["path"] for entry_dict in integrate.read_list()] == ["a.py"]
    integrate.reset_read()


def test_the_plan_status_and_the_deadline_are_shown_together():
    both = agent.turn_note_chain(lambda t, m: "## Plan status", lambda t, m: "## 1 turns left")
    assert both(20, 20) == "## Plan status\n\n## 1 turns left"
    assert agent.turn_note_chain(None, lambda t, m: "")(1, 20) == ""


# == what may be changed in a test ==========================
def test_the_line_a_change_made_stale_is_the_one_to_update(monkeypatch, tmp_path):
    """The stage already updates such a test; what it is told is where to stop."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    text = agent.build_instruction("py")
    assert "update that\n   assertion, nothing else in that file" in text
    assert "delete a test to make it pass" in text
    assert "recorded in the commit" in text


def test_a_repository_with_no_tests_is_told_none_of_that(monkeypatch, tmp_path):
    """It is part of the tests step, which is not there when there is nothing to run."""
    monkeypatch.setattr(config, "RUN_TEST", False)
    text = agent.build_instruction("py")
    assert "delete a test to make it pass" not in text and "assertion" not in text


# == The guards on how a run ends, chained =================
def test_a_chain_whose_guards_all_accept_accepts():
    on_complete = agent.complete_guard_chain(lambda answer: "", lambda answer: "")
    assert on_complete("done") == ""


def test_the_first_guard_that_refuses_wins():
    on_complete = agent.complete_guard_chain(lambda answer: "", lambda answer: "no",
                                             lambda answer: "later")
    assert on_complete("done") == "no"


def test_no_guard_in_force_is_none():
    assert agent.complete_guard_chain(None, None) is None


def test_a_single_guard_is_handed_back_as_it_is():
    guard = lambda answer: ""  # noqa: E731
    assert agent.complete_guard_chain(None, guard) is guard


def test_a_test_file_the_run_changed_is_said_in_the_log(monkeypatch, caplog):
    monkeypatch.setattr(agent.edit, "no_tool_path_set", lambda: set())
    monkeypatch.setattr(agent.edit, "test_change_set", lambda: {"tests/test_a.py"})
    with caplog.at_level(logging.WARNING):
        agent._warn_no_tool_path()
    assert "1 test file(s) were changed" in caplog.text and "tests/test_a.py" in caplog.text


def test_the_page_read_makes_16_tools_with_both_endpoints_and_none_without_search(
        monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    names = _tool_names(monkeypatch)
    assert len(names) == 16 and names.index("web_search") + 1 == names.index("web_fetch")
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "")
    names = _tool_names(monkeypatch)
    assert len(names) == 14 and "web_fetch" not in names


def test_the_page_read_goes_with_its_budget(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    assert "web_fetch" in _tool_names(monkeypatch)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 0)
    assert "web_fetch" not in _tool_names(monkeypatch)


def test_no_reading_endpoint_takes_the_page_read_away(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 3)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "")
    assert "web_fetch" not in _tool_names(monkeypatch)


# == the check at the end of a run on its own ==================================

_DONE = {"answer": "renamed it", "harvested": False}


@pytest.fixture
def _final(monkeypatch):
    """Stand in for what the final check reads, and return the calls it made.

    One file is written by the run, the analysis reads as {"at": its path} and the
    static check found nothing before the change.
    """
    seen = {"reason": [], "call": []}
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    monkeypatch.setattr(agent.integrate, "analyse", lambda path: {"at": path})
    monkeypatch.setattr(agent.edit, "run_check", lambda: "counter")
    monkeypatch.setattr(agent.edit, "write_path_set", lambda: {"a.py"})
    monkeypatch.setattr(agent.edit, "no_tool_path_set", lambda: set())

    def ref_reason_tuple(project_dir, output_dir, start_dict, exclude_list,
                         check_before=None):
        seen["call"].append((start_dict, exclude_list, check_before))
        return list(seen["reason"]), True

    monkeypatch.setattr(agent.testcreate, "ref_reason_tuple", ref_reason_tuple)
    return seen


def test_what_the_change_is_compared_against_is_read_before_the_run(_final):
    assert agent.check_start("/out/k.sqlite", ["vendor"]) == {
        "analysis": {"at": "/out/k.sqlite"}, "check": "counter", "exclude": ["vendor"]}


def test_the_check_is_left_out_of_a_dry_run(_final, monkeypatch):
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    assert agent.check_start("/out/k.sqlite", []) is None
    assert agent.final_check(None, _DONE) is None


def test_an_analysis_that_cannot_be_read_leaves_the_check_out(_final, monkeypatch,
                                                              caplog):
    def refuse(path):
        raise OSError("no such file")

    monkeypatch.setattr(agent.integrate, "analyse", refuse)
    assert agent.check_start("/out/k.sqlite", []) is None
    assert "not held to the references" in caplog.text


def test_a_change_is_held_to_what_was_read_before_it(_final):
    start_dict = agent.check_start("/out/k.sqlite", ["vendor"])
    assert agent.final_check(start_dict, _DONE) == []
    assert _final["call"] == [({"at": "/out/k.sqlite"}, ["vendor"], "counter")]
    _final["reason"] = ["- `ruff` finds what it did not find before you started: x"]
    assert agent.final_check(start_dict, _DONE) == _final["reason"]


@pytest.mark.parametrize("result", [{"answer": None, "harvested": False},
                                    {"answer": "edited a.py", "harvested": True}])
def test_a_run_that_did_not_end_itself_is_not_checked(_final, result):
    assert agent.final_check(agent.check_start("/out/k.sqlite", []), result) is None
    assert _final["call"] == []


def test_a_run_that_changed_nothing_is_not_checked(_final, monkeypatch):
    monkeypatch.setattr(agent.edit, "write_path_set", lambda: set())
    assert agent.final_check(agent.check_start("/out/k.sqlite", []), _DONE) is None
    monkeypatch.setattr(agent.edit, "no_tool_path_set", lambda: {"b.py"})
    assert agent.final_check(agent.check_start("/out/k.sqlite", []), _DONE) == []


def test_what_the_check_found_makes_the_run_failed(_no_search, _final, monkeypatch):
    _final["reason"] = ["- these calls resolved before you started and no longer do: f"]
    result = agent.run_edit(_RecordingRLM(answer="renamed it"), "名前を変えて",
                            agent.check_start("/out/k.sqlite", []))
    assert result["final_check"] == _final["reason"]
    assert result["verdict"] == history.VERDICT_FAIL
    assert "was not there before the change" in result["why"]


def test_a_run_given_nothing_to_compare_against_is_judged_as_before(_no_search, _final):
    result = agent.run_edit(_RecordingRLM(answer="renamed it"), "名前を変えて")
    assert result["final_check"] is None and _final["call"] == []
    assert result["verdict"] == history.VERDICT_NO_PROOF


# == COMPLETE() is sent back for what the check finds =========================

def test_a_run_not_held_to_the_check_has_no_guard_for_it():
    assert agent.make_check_guard(None) is None


def test_complete_is_sent_back_while_the_check_finds_something(_final, monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_MAX_REFUSAL", 2)
    guard = agent.make_check_guard(agent.check_start("/out/k.sqlite", ["vendor"]))
    _final["reason"] = ["- these calls resolved before you started and no longer do: f"]
    first = guard("done")
    assert first.startswith("COMPLETE() was not accepted. The change is not settled:\n"
                            "- these calls resolved before you started and no longer do: f")
    assert "It can be sent back 1 more time(s)." in first
    assert "This is the last time it is sent back." in guard("done")
    # The bound is reached: the call is accepted and the check is not made again
    made = len(_final["call"])
    assert guard("done") == "" and len(_final["call"]) == made


def test_complete_is_accepted_once_the_check_finds_nothing(_final):
    guard = agent.make_check_guard(agent.check_start("/out/k.sqlite", []))
    _final["reason"] = ["- x"]
    assert guard("done")
    _final["reason"] = []
    assert guard("done") == ""


def test_a_run_that_changed_nothing_is_not_sent_back(_final, monkeypatch):
    monkeypatch.setattr(agent.edit, "write_path_set", lambda: set())
    _final["reason"] = ["- x"]
    guard = agent.make_check_guard(agent.check_start("/out/k.sqlite", []))
    assert guard("it needs no change") == "" and _final["call"] == []


def test_the_editing_loop_is_handed_the_check_guard_of_a_run_on_its_own(_no_search,
                                                                        _final):
    rlm = _RecordingRLM(answer="renamed it")
    agent.run_edit(rlm, "名前を変えて")
    assert rlm.on_complete is None
    _final["reason"] = ["- x"]
    agent.run_edit(rlm, "名前を変えて", agent.check_start("/out/k.sqlite", []))
    assert rlm.on_complete("done").startswith("COMPLETE() was not accepted. The change")


# == no tool raises over a value of the wrong kind =============
# What a tool is handed in place of each of its arguments, one at a time
_ODD_VALUE_TUPLE = (
    None, True, 0, -1, 1.5, float("nan"), 10 ** 30, "", " ", "\0", "a\0b", "\ud800",
    "a\udc00b", ".", "./", "..", "/", "//x", "~", "*", "**", "**x", "[", "(", "a(b",
    "-x", "x" * 5000, "cfg.py\n", "a{99999999999}", "(?u)(?a)x", "(" * 5000 + ")" * 5000,
    "cfg.py::\ud800", b"cfg.py", [], ["cfg.py"], [None], [["cfg.py"]], [5],
    {}, {"a": 1}, {1: 2}, (), ("cfg.py",), set(), {"cfg.py"}, object(),
    pathlib.Path("cfg.py"), re.compile("cfg"), re.compile(b"cfg"), print)

# What each argument is given while another one is handed an odd value, by its name;
# an argument not named here takes its default
_PLAIN_VALUE_DICT = {
    "path": "cfg.py", "name": "load_config", "target_file": "cfg.py", "pattern": "cfg",
    "aspect": "error handling", "question": "what changed", "old_text": "cfg 1",
    "url": "https://example.invalid/guide", "new_text": "cfg one", "start_line": 1,
    "end_line": 1, "text": "x = 1\n", "arg_list": [], "query": "config",
    "branch": "", "what": "kept both", "why": "they do not overlap"}


@pytest.fixture
def _every_tool(tmp_path, monkeypatch, fake_index):
    """A repository with fake_index's two files and every tool switched on, none of
    them reaching a model or a network."""
    root = tmp_path / "p"
    git_repo(root)
    (root / "cfg.py").write_text("".join(f"cfg {i}\n" for i in range(1, 101)))
    (root / "util.py").write_text("".join(f"util {i}\n" for i in range(1, 11)))
    (root / "tests").mkdir()
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, capture_output=True)
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    monkeypatch.setattr(config, "QUERY_EXPAND", False)
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_COMMAND", [sys.executable, "-c", "print('1 passed')"])
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "RUN_COMMAND", [sys.executable, "-c", "pass"])
    monkeypatch.setattr(config, "SMOKE_ARG", [])
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 10 ** 6)
    for name in ("WEB_SEARCH", "WEB_FETCH"):
        monkeypatch.setattr(config, f"{name}_MAX_CALL", 10 ** 6)
        monkeypatch.setattr(config, f"{name}_URL", "https://endpoint.invalid/")
    monkeypatch.setattr(web, "_search", lambda question: [
        {"title": "Guide", "url": "https://example.invalid/guide", "content": "c"}])
    monkeypatch.setattr(web, "_fetch", lambda url: "the page")
    monkeypatch.setattr(explore, "run_stage",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    monkeypatch.setattr(edit, "refill_cache_on_dep_change", lambda: "")
    edit.reset_patch_round(10 ** 6)
    web.web_search("what changed")
    return root


def _raise_list(tool, odd_tuple) -> list[str]:
    """Call one tool with each of its arguments in turn replaced by each odd value.

    Args:
        tool: The tool.
        odd_tuple: The values to hand it.

    Returns:
        "<tool>(<argument>=<value>) raised <error>" for each call that raised anything
        but the ValueError web_search and web_fetch raise for a blank text.
    """
    name_list = list(inspect.signature(tool).parameters)
    raise_list = []
    for odd_name in name_list:
        for value in odd_tuple:
            kwarg_dict = {name: _PLAIN_VALUE_DICT[name] for name in name_list
                          if name in _PLAIN_VALUE_DICT}
            kwarg_dict[odd_name] = value
            try:
                tool(**kwarg_dict)
            except Exception as e:                         # noqa: BLE001
                is_blank = isinstance(value, str) and not value.strip()
                if not (is_blank and tool.__name__ in ("web_search", "web_fetch")
                        and isinstance(e, ValueError)):
                    raise_list.append(f"{tool.__name__}({odd_name}={value!r:.40}) "
                                      f"raised {type(e).__name__}: {str(e):.80}")
    return raise_list


def test_no_tool_raises_whatever_one_argument_is_handed(_every_tool):
    """Every tool any stage hands over, every argument of it, every odd value: the
    answer is the tool's own refusal or its result, and never an exception."""
    tool_list = agent.all_tool_list()
    assert {"grep_source", "run_test", "run_project", "web_search", "web_fetch",
            "explore", "read_branch_file", "record_gap", "search_again",
            "delete_file"} <= {tool.__name__ for tool in tool_list}
    odd_tuple = _ODD_VALUE_TUPLE + (f"{_every_tool}/a\0b", f"{_every_tool}/\udc80",
                                    f"{_every_tool}/cfg.py", str(_every_tool))
    raise_list = [line for tool in tool_list for line in _raise_list(tool, odd_tuple)]
    assert not raise_list, "\n".join(raise_list)


def test_a_refusal_for_the_kind_of_a_value_spends_none_of_a_tool_s_calls(_every_tool):
    web.reset_web_search_count()
    web.reset_web_fetch_count()
    explore.reset_explore_count()
    source.reset_search_count()
    for value in (None, 5, b"x", ["x"], {"x": 1}, object()):
        assert web.web_search(value)["ok"] is False
        assert web.web_fetch(value)["ok"] is False
        assert explore.explore(value)["ok"] is False
        assert "error" in source.search_again(value)
        assert "error" in source.grep_source(value)
    assert (web.web_search_count(), web.web_fetch_count()) == (0, 0)
    assert explore.call_remain_count() == config.EXPLORE_MAX_CALL
    assert source._search_again_count == 0
