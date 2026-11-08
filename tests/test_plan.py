"""The planning stage and the two-stage wiring.

run_plan() builds and runs a fresh read-only RLM, so the LLM is substituted for these
checks, the same way tests/test_explore.py substitutes the explorer's.
"""

import json
import logging
import re
import subprocess
import time

import pytest

from codetwine_loom import agent, config, integrate, plan
from codetwine_loom import rlm as rlm_mod
from codetwine_loom.rlm import COMPLETE_KEY
from codetwine_loom.tools import edit, web
from tests.conftest import git_repo

PLAN_JSON = json.dumps({
    "files": [{"path": "codetwine/analysis_cache.py",
               "change": "hold the cache index",
               "anchor": "new file"},
              {"path": "codetwine/pipeline.py",
               "change": "call the cache before step 2",
               "anchor": "process_all_files [145-249]"}],
    "order": ["codetwine/analysis_cache.py", "codetwine/pipeline.py"],
    "interfaces": ["plan_analysis(files) -> {analyzed, skipped, removed}"],
    "verify": "run_project on a 2-file sample twice and compare the stats file"})


def _text(prompt) -> str:
    """The text of what an LLM was sent, whichever form PROMPT_CACHE gave it."""
    return prompt if isinstance(prompt, str) else rlm_mod.prompt_text(prompt)


def _scripted_llm(*replies: str):
    """Build an LLM returning the given replies in order, repeating the last one."""
    seq = list(replies)

    def call(prompt: str) -> str:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return call


@pytest.fixture
def plan_llm(monkeypatch):
    """Set up a planning stage that COMPLETE()s a plan on its first turn."""
    llm = _scripted_llm(f"```python\nCOMPLETE({PLAN_JSON!r})\n```")
    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: llm)
    return llm


@pytest.fixture
def no_git(monkeypatch):
    """Make git_diff_stat report a repository where nothing has been touched."""
    def stat():
        return {"files": [], "untracked": [], "total_insertions": 0,
                "total_deletions": 0}

    monkeypatch.setattr(plan.edit, "git_diff_stat", stat)


# == run_plan and parse_plan ================================
def test_run_plan_returns_the_files_the_plan_lists(plan_llm):
    out = plan.run_plan("差分解析にして", {"files": [], "defs": []})
    assert out["ok"] is True
    assert plan.plan_path_list(out["plan"]) == ["codetwine/analysis_cache.py",
                                               "codetwine/pipeline.py"]


def test_run_plan_reports_the_turns_it_used(plan_llm):
    assert plan.run_plan("差分解析にして", {"files": [], "defs": []})["iters"] == 1


def test_a_plan_that_never_completes_is_not_ok(monkeypatch):
    monkeypatch.setattr(plan, "make_llm",
                        lambda *a, **k: _scripted_llm("```python\nprint(1)\n```"))
    monkeypatch.setattr(config, "PLAN_MAX_TURN", 2)
    out = plan.run_plan("差分解析にして", {"files": [], "defs": []})
    assert out["ok"] is False and out["iters"] == 2


def test_a_reply_that_is_not_json_comes_back_as_an_error():
    out = plan.parse_plan("I could not decide")
    assert out["ok"] is False and "not valid JSON" in out["error"]


def test_a_json_object_without_files_is_refused():
    assert plan.parse_plan('{"order": ["a.py"]}')["ok"] is False


def test_files_are_capped_at_plan_max_files(monkeypatch):
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 2)
    answer = json.dumps({"files": [{"path": f"f{i}.py"} for i in range(5)]})
    assert len(plan.parse_plan(answer)["plan"]["files"]) == 2


def test_an_entry_without_a_path_is_dropped():
    answer = json.dumps({"files": [{"change": "no path here"}, {"path": "a.py"}]})
    assert plan.plan_path_list(plan.parse_plan(answer)["plan"]) == ["a.py"]


def test_the_entries_left_out_are_counted_in_the_log(caplog):
    answer = json.dumps({"files": ["b.py", {"change": "no path here"}, {"path": "a.py"}]})
    with caplog.at_level(logging.WARNING):
        assert plan.parse_plan(answer)["ok"] is True
    assert '2 of the 3 entries of the plan\'s "files" name no path' in caplog.text


@pytest.mark.parametrize("file_list", [["a.py", "b.py"],
                                       [{"file": "a.py", "change": "x"}],
                                       [None, 3, {"path": ""}]])
def test_a_plan_whose_files_name_no_path_at_all_is_not_a_plan(file_list):
    """It is not read as the plan with an empty "files" list, which says that nothing
    has to change."""
    out = plan.parse_plan({"files": file_list, "order": ["a.py"], "facts": ["f"]})
    assert out["ok"] is False and "plan" not in out
    assert out["error"].startswith(
        f'none of the {len(file_list)} entries of "files" is an object with a "path"')


def test_a_plan_with_an_empty_files_list_is_still_a_plan():
    out = plan.parse_plan({"files": [], "facts": ["it already does that"]})
    assert out["ok"] is True and out["plan"]["files"] == []


# == facts: what the planning stage checked instead of assuming ==
def test_facts_survive_into_the_plan():
    answer = json.dumps({"files": [{"path": "a.py"}],
                         "facts": ["const_spec carries the name in an identifier"]})
    plan_out = plan.parse_plan(answer)["plan"]
    assert plan_out["facts"] == ["const_spec carries the name in an identifier"]


def test_a_plan_without_facts_still_parses():
    """A plan with no "facts" key parses, with facts as an empty list."""
    assert plan.parse_plan(PLAN_JSON)["plan"]["facts"] == []


def test_facts_are_capped_and_stripped(monkeypatch):
    monkeypatch.setattr(config, "PLAN_MAX_FACT", 2)
    answer = json.dumps({"files": [{"path": "a.py"}],
                         "facts": ["  one  ", "", "two", "three"]})
    assert plan.parse_plan(answer)["plan"]["facts"] == ["one", "two"]


def test_a_fact_that_is_not_a_line_is_dropped():
    """A nested object would be rendered as a Python repr nobody can read."""
    answer = json.dumps({"files": [{"path": "a.py"}],
                         "facts": [{"node": "const_spec"}, ["x"], "the readable one"]})
    assert plan.parse_plan(answer)["plan"]["facts"] == ["the readable one"]


# == render_plan ============================================
def test_the_rendered_plan_carries_the_anchors_and_the_hand_offs():
    text = plan.render_plan(json.loads(PLAN_JSON))
    assert "codetwine/pipeline.py  [process_all_files [145-249]]" in text
    assert "plan_analysis(files) -> {analyzed, skipped, removed}" in text
    assert text.startswith("## Plan")


def test_the_rendered_plan_says_complete_is_sent_back_only_while_the_guard_is_on(
        monkeypatch):
    sentence = ("one has been changed yet. While a file in the plan is still untouched, "
                "COMPLETE() is sent\nback once.\n\nFiles:")
    monkeypatch.setattr(config, "PLAN_COMPLETE_GUARD", True)
    assert sentence in plan.render_plan(json.loads(PLAN_JSON))
    monkeypatch.setattr(config, "PLAN_COMPLETE_GUARD", False)
    text = plan.render_plan(json.loads(PLAN_JSON))
    assert "one has been changed yet.\n\nFiles:" in text and "sent" not in text


def test_the_rendered_plan_carries_the_facts():
    plan_dict = json.loads(PLAN_JSON) | {"facts": ["tree-sitter-go: type_spec"]}
    text = plan.render_plan(plan_dict)
    assert "tree-sitter-go: type_spec" in text
    assert "not guesses" in text


def test_a_plan_with_no_facts_renders_no_section():
    assert "not guesses" not in plan.render_plan(json.loads(PLAN_JSON))


# == the planning stage may run the program ==
def test_the_planner_can_run_the_program_but_cannot_edit():
    names = [t.__name__ for t in plan.plan_tool_list()]
    assert "run_project" in names
    assert not {"replace_in_file", "replace_line_range", "create_file"} & set(names)


def test_a_run_budget_of_0_takes_run_project_away_from_the_planner(monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", 0)
    assert "run_project" not in [t.__name__ for t in plan.plan_tool_list()]
    assert plan.plan_run_note() == ""
    assert "run_project" not in plan.build_plan_instruction("py")


def test_a_search_budget_of_0_takes_search_again_away_from_the_reading_stages(
        monkeypatch):
    assert "search_again" in [t.__name__ for t in plan.plan_tool_list()]
    assert "search_again" in [t.__name__ for t in plan.answer_tool_list()]
    monkeypatch.setattr(config, "SEARCH_AGAIN_MAX_CALL", 0)
    assert "search_again" not in [t.__name__ for t in plan.plan_tool_list()]
    assert "search_again" not in [t.__name__ for t in plan.answer_tool_list()]
    assert "run_project" in [t.__name__ for t in plan.plan_tool_list()]


def test_the_web_question_reaches_the_planner_only_when_it_is_switched_on(monkeypatch):
    """What it learns crosses to the editing stage as one line of "facts"."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    assert "web_search" not in [t.__name__ for t in plan.plan_tool_list()]
    assert "web_search" not in plan.build_plan_instruction("py")
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    text = plan.build_plan_instruction("py")
    assert "web_search" in [t.__name__ for t in plan.plan_tool_list()]
    assert "You get 2 calls" in text
    assert 'through "facts"' in text


def _no_search_error(tool_list) -> str:
    """Return what a stage's web_search answers when the search does not run."""
    tool = {tool.__name__: tool for tool in tool_list}["web_search"]
    web.reset_web_search_count()
    return tool("what changed")["error"]


def test_web_search_names_run_project_only_to_a_stage_that_has_it(monkeypatch):
    def raising(question):
        raise RuntimeError("no route")

    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(web, "_search", raising)
    assert "run_project" in _no_search_error(plan.plan_tool_list())
    assert "run_project" in _no_search_error(agent.main_tool_list())
    assert "run_project" not in _no_search_error(plan.answer_tool_list())
    monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", 0)
    assert "run_project" not in _no_search_error(plan.plan_tool_list())
    assert "run_project" not in _no_search_error(agent.main_tool_list())


def test_the_run_note_is_left_out_without_a_launcher(monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", [])
    assert plan.plan_run_note() == ""
    assert "run_project" not in plan.build_plan_instruction("py")


def test_the_run_note_names_the_launcher(monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    text = plan.build_plan_instruction("py")
    assert "`uv run`" in text
    # The placeholder has to survive both format() passes to be usable as written
    assert '"{workspace}" path' in text


def test_the_instructions_carry_the_configured_limits(monkeypatch):
    monkeypatch.setattr(config, "PLAN_MAX_FACT", 7)
    assert 'At most 7 entries in "facts"' in plan.build_plan_instruction("py")


# == the per-turn status ====================================
def test_the_status_marks_an_untouched_file(no_git):
    note = plan.make_turn_note(json.loads(PLAN_JSON))(28, 40)
    assert note.startswith("## Plan status (turn 28/40)")
    assert note.count("NOT TOUCHED") == 2


def test_a_file_that_was_written_is_no_longer_not_touched(monkeypatch, no_git):
    monkeypatch.setattr(plan, "change_path_set",
                        lambda turn=0: {"codetwine/analysis_cache.py"})
    note = plan.make_turn_note(json.loads(PLAN_JSON))(3, 40)
    assert "codetwine/analysis_cache.py" in note and note.count("NOT TOUCHED") == 1
    assert "written" in note


def test_only_what_this_request_changed_counts_as_written(monkeypatch):
    """A run that starts on a tree with changes in it, or carries on from an earlier
    one, would otherwise find its planned files written before it began."""
    monkeypatch.setattr(plan.edit, "own_change_set", lambda: {"a.py"})
    monkeypatch.setattr(plan.edit, "write_path_set", lambda: {"new.py"})
    assert plan.change_path_set() == {"a.py", "new.py"}


# == the per-turn state =====================================
def read_source_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """Stand in for the reading tool of the same name."""
    return f"text of {path}"


def read_definition(name: str, file: str | None = None) -> dict:
    """Stand in for the reading tool of the same name."""
    return {"ok": name != "gone", "file": "src/b.py"}


def grep_source(pattern: str) -> dict:
    """Stand in for a tool that reads no one file."""
    return {"ok": True}


@pytest.fixture
def state_repo(tmp_path, monkeypatch):
    """Point PROJECT_DIR at a repository holding one a.py and start a request in it."""
    root = tmp_path / "proj"
    git_repo(root)
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    edit.reset_patch_round()
    return root


def test_the_state_is_empty_while_nothing_was_read_or_changed(state_repo):
    assert plan.make_state_note()(1, 40) == ""
    assert plan.make_state_note(with_change=True)(1, 40) == ""


def test_the_state_names_each_file_read_with_the_last_turn_it_was_read_on():
    read, definition, grep = integrate.read_record_tool_list(
        [read_source_file, read_definition, grep_source])
    assert grep is grep_source and read.__name__ == "read_source_file"
    note = plan.make_state_note()
    note(1, 40)
    assert read("src/a.py") == "text of src/a.py"
    note(2, 40)
    read("./src/a.py", 10)
    definition("total")
    definition("gone")
    grep("x")
    assert note(3, 40) == ("## State (turn 3/40)\n\n"
                           "Files you have read, as of this turn:\n"
                           "  src/a.py  on turn 2\n"
                           "  src/b.py  on turn 2")


def test_a_new_state_starts_from_an_empty_read_record():
    read = integrate.read_record_tool_list([read_source_file])[0]
    note = plan.make_state_note()
    note(1, 40)
    read("src/a.py")
    assert "src/a.py" in note(2, 40)
    assert plan.make_state_note()(1, 40) == ""


def test_the_state_of_a_stage_that_writes_names_what_the_request_changed(state_repo):
    (state_repo / "b.py").write_text("one\ntwo\n", encoding="utf-8")
    (state_repo / "c.py").write_text("gone soon\n", encoding="utf-8")
    for args in (("add", "-A"), ("commit", "--quiet", "-m", "more")):
        subprocess.run(["git", "-C", str(state_repo), *args], check=True,
                       capture_output=True)
    edit.reset_patch_round()
    note = plan.make_state_note(with_change=True)
    assert edit.replace_in_file("b.py", "two\n", "2\n3\n")["ok"]
    assert edit.create_file("new.py", "x = 1\n")["ok"]
    assert edit.delete_file("c.py")["ok"]
    assert note(2, 40) == ("## State (turn 2/40)\n\n"
                           "Files you have changed, as of this turn, with the lines "
                           "against the last commit:\n"
                           "  b.py  +2 -1\n"
                           "  c.py  deleted\n"
                           "  new.py  new file\n\n"
                           "Files you have read, as of this turn: none.")


def test_the_state_leaves_out_what_the_tree_held_before_the_request(state_repo):
    (state_repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    edit.reset_patch_round()
    note = plan.make_state_note(with_change=True)
    assert note(1, 40) == ""
    assert edit.create_file("new.py", "x = 1\n")["ok"]
    assert "a.py" not in note(2, 40).replace("new.py", "")


def test_the_state_of_a_stage_that_only_reads_names_no_change(state_repo):
    assert edit.create_file("new.py", "x = 1\n")["ok"]
    read = integrate.read_record_tool_list([read_source_file])[0]
    note = plan.make_state_note()
    note(1, 40)
    read("a.py")
    assert "changed" not in note(2, 40)


def test_the_state_says_what_the_whole_suite_last_said_and_whether_it_is_old(
        state_repo, monkeypatch):
    note = plan.make_state_note(with_change=True)
    note(1, 40)
    assert edit.create_file("new.py", "x = 1\n")["ok"]
    monkeypatch.setattr(edit, "_last_whole_run_dict", {
        "n": 1, "result": {"status": "failed", "failed": ["test_a", "test_b"]},
        "change": edit.stage_change_hash_dict()})
    assert note(2, 40).endswith(
        "The whole test suite, the last time you ran it (on turn 1): failed, 2 failing "
        "test(s) named.")
    assert edit.create_file("other.py", "y = 1\n")["ok"]
    # The run is the same one, so the turn it ran on stays
    assert note(3, 40).endswith(
        "The whole test suite, the last time you ran it (on turn 1): failed, 2 failing "
        "test(s) named.\nYou have changed a file since that run.")


def test_the_state_asked_after_the_last_turn_is_headed_with_that_turn():
    read = integrate.read_record_tool_list([read_source_file])[0]
    note = plan.make_state_note()
    note(40, 40)
    read("a.py")
    assert note(41, 40) == ("## State (turn 40/40)\n\n"
                            "Files you have read, as of this turn:\n"
                            "  a.py  on turn 40")


def test_a_list_of_the_state_over_the_item_limit_says_how_many_it_left_out(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    read = integrate.read_record_tool_list([read_source_file])[0]
    note = plan.make_state_note()
    note(1, 40)
    for name in ("a.py", "b.py", "c.py", "d.py"):
        read(name)
    assert note(2, 40).endswith("  a.py  on turn 1\n  b.py  on turn 1\n  ... and 2 more")


def test_the_planning_stage_reads_its_state_after_the_history(monkeypatch, state_repo):
    """The block is the last part of the prompt before the turn line, and the parts in
    front of it are the ones the turn before sent."""
    prompt_list = []

    def llm(prompt):
        prompt_list.append(prompt)
        return ("```python\nprint(read_source_file('a.py')[:5])\n```" if len(prompt_list) == 1
                else f"```python\nCOMPLETE({PLAN_JSON!r})\n```")

    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: llm)
    assert plan.run_plan("REQ", {"files": [], "defs": []})["ok"]
    first, second = prompt_list
    assert "## State" not in first
    assert "## State (turn 2/" in second and "  a.py  on turn 1" in second
    assert second.index("### Turn 1 output") < second.index("## State (turn 2/")
    assert second.startswith(first[:first.index("## Turn 1/")])


def test_the_guard_counts_a_file_the_same_block_wrote(monkeypatch):
    """The status is worked out before the turn's code runs and the guard after it."""
    write_set: set[str] = set()
    monkeypatch.setattr(plan.edit, "own_change_set", lambda: set(write_set))
    monkeypatch.setattr(plan.edit, "write_path_set", lambda: set(write_set))
    monkeypatch.setattr(config, "PLAN_COMPLETE_GUARD", True)

    def replace_in_file(path: str) -> dict:
        write_set.add(path)
        return {"ok": True}

    reply_list = ["```python\nreplace_in_file('a.py')\nCOMPLETE('changed a.py')\n```"]
    plan_dict = {"files": [{"path": "a.py"}]}
    stage = rlm_mod.RLM(llm=lambda prompt: reply_list.pop(0), tool_list=[replace_in_file],
                        instruction="I", stage="edit", max_turn=3)
    out_dict = stage.run("REQ", {}, "", turn_note=plan.make_turn_note(plan_dict),
                         on_complete=plan.make_complete_guard(plan_dict))
    assert out_dict["outcome"] == "complete" and out_dict["refused_completes"] == 0


def test_a_plan_path_is_matched_however_it_was_written(monkeypatch, tmp_path):
    """"./a.py" and an absolute path name the file git reports as "a.py"."""
    monkeypatch.setattr(plan.config, "PROJECT_DIR", str(tmp_path))
    plan_dict = {"files": [{"path": "./a.py"}, {"path": str(tmp_path / "b.py")},
                           {"path": "a.py"}]}
    assert plan.plan_path_list(plan_dict) == ["a.py", "b.py"]


# == the complete guard =======================================
def test_complete_is_refused_while_a_planned_file_is_untouched(no_git):
    guard = plan.make_complete_guard(json.loads(PLAN_JSON))
    refusal = guard("done")
    assert "codetwine/pipeline.py" in refusal


def test_complete_is_refused_once_only(no_git):
    guard = plan.make_complete_guard(json.loads(PLAN_JSON))
    guard("done")
    assert guard("done again") == ""


def test_complete_passes_when_every_planned_file_was_touched(monkeypatch, no_git):
    monkeypatch.setattr(plan, "change_path_set",
                        lambda: {"codetwine/analysis_cache.py",
                                 "codetwine/pipeline.py"})
    assert plan.make_complete_guard(json.loads(PLAN_JSON))("done") == ""


def test_the_guard_can_be_turned_off(monkeypatch, no_git):
    monkeypatch.setattr(config, "PLAN_COMPLETE_GUARD", False)
    assert plan.make_complete_guard(json.loads(PLAN_JSON))("done") == ""


# == agent.run_edit with the two stages =====================
@pytest.fixture
def stub_search(monkeypatch):
    """Make the search return a fixed scope without touching the index."""
    scope = {"files": ["codetwine/pipeline.py"],
             "defs": [{"key": "codetwine/pipeline.py::process_all_files",
                       "lines": [145, 249],
                       "signature": "def process_all_files(paths):"}],
             "file_summaries": {"codetwine/pipeline.py": "runs the analysis"},
             "callers": {}, "confidence": "high", "spread": 2}
    monkeypatch.setattr(agent, "search_scope", lambda request: scope)
    monkeypatch.setattr(agent.explore, "set_context", lambda request, scope: None)
    return scope


class _RecordingRLM:
    """A stand-in for the editing stage that records what run() was handed."""

    tool_dict = {}

    def __init__(self):
        self.request = ""
        self.turn_note = None
        self.on_complete = None
        self.after_turn = None

    def run(self, request, variable_dict, var_doc, turn_note=None, on_complete=None,
            after_turn=None):
        self.request = request
        self.turn_note = turn_note
        self.on_complete = on_complete
        self.after_turn = after_turn
        return {"answer": "done", "outcome": COMPLETE_KEY, "iters": 1, "history": [],
                "truncated_replies": 0, "truncated_lost": 0, "empty_replies": 0,
                "bare_code_turns": 0, "refused_completes": 0}


def test_a_single_stage_run_attaches_no_plan(stub_search, monkeypatch):
    monkeypatch.setattr(config, "TWO_STAGE", False)
    rlm = _RecordingRLM()
    result = agent.run_edit(rlm, "差分解析にして")
    assert result["plan"] is None
    assert "## Plan" not in rlm.request and rlm.turn_note(1, 40) == ""


def test_the_editing_stage_gets_the_plan_and_both_hooks(stub_search, plan_llm,
                                                        monkeypatch, no_git):
    monkeypatch.setattr(config, "TWO_STAGE", True)
    rlm = _RecordingRLM()
    result = agent.run_edit(rlm, "差分解析にして")
    assert result["plan"]["ok"] is True
    assert "## Plan" in rlm.request
    assert rlm.turn_note(1, 40).startswith("## Plan status")
    assert "codetwine/pipeline.py" in rlm.on_complete("done")


def test_the_editing_stage_drops_the_file_summaries_behind_a_plan(stub_search, plan_llm,
                                                                  monkeypatch):
    monkeypatch.setattr(config, "TWO_STAGE", True)
    rlm = _RecordingRLM()
    agent.run_edit(rlm, "差分解析にして")
    assert "File summaries:" not in rlm.request
    assert "Matched definitions:" in rlm.request


def test_the_planning_stages_program_runs_are_not_the_editing_stages(stub_search,
                                                                     plan_llm,
                                                                     monkeypatch,
                                                                     no_git):
    """run_calls() means "how often the editing model ran what it built", so the
    planner's own runs are taken and then cleared."""
    monkeypatch.setattr(agent.run, "run_count", lambda: 2)
    cleared = []
    monkeypatch.setattr(agent.run, "reset_run_state", lambda: cleared.append(True))
    monkeypatch.setattr(plan.run, "run_count", lambda: 2)
    monkeypatch.setattr(config, "TWO_STAGE", True)
    result = agent.run_edit(_RecordingRLM(), "差分解析にして")
    assert result["plan"]["run_calls"] == 2
    assert cleared == [True]


def test_an_unreadable_plan_leaves_the_editing_stage_as_it_was(stub_search, monkeypatch):
    monkeypatch.setattr(config, "TWO_STAGE", True)
    monkeypatch.setattr(plan, "make_llm",
                        lambda *a, **k: _scripted_llm("```python\nCOMPLETE('nope')\n```"))
    rlm = _RecordingRLM()
    result = agent.run_edit(rlm, "差分解析にして")
    assert result["plan"]["ok"] is False
    assert "## Plan" not in rlm.request and rlm.turn_note(1, 40) == ""
    # The full scope comes back when there is no plan to replace it
    assert "File summaries:" in rlm.request


# == The deadline note and a plan wrapped in prose ==========
def _plan_turn_note(turn, max_turn):
    return rlm_mod.make_deadline_note(config.PLAN_DEADLINE_TURN, plan._DEADLINE_WHAT)(
        turn, max_turn)


def test_no_deadline_note_while_there_is_time():
    assert _plan_turn_note(5, 15) == ""


def test_the_deadline_note_says_how_many_turns_are_left():
    note = _plan_turn_note(13, 15)
    assert note.startswith("## 3 turns left") and "COMPLETE()" in note


def test_no_deadline_note_for_a_turn_past_the_last_one():
    note = rlm_mod.make_deadline_note(3, "Hand over now.")
    assert note(20, 20).startswith("## 1 turns left")
    assert note(21, 20) == ""


def test_the_deadline_note_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(config, "PLAN_DEADLINE_TURN", 0)
    assert _plan_turn_note(15, 15) == ""


def test_a_plan_wrapped_in_a_fence_is_still_read():
    answer = '```json\n{"files": [{"path": "a.py"}]}\n```'
    assert plan.plan_path_list(plan.parse_plan(answer)["plan"]) == ["a.py"]


def test_a_plan_with_a_sentence_around_it_is_still_read():
    answer = 'Here is the plan: {"files": [{"path": "a.py"}]} — that is all.'
    assert plan.parse_plan(answer)["ok"] is True


def test_a_reply_with_no_object_at_all_is_still_an_error():
    assert plan.parse_plan("no json here")["ok"] is False


def test_complete_called_with_the_dict_itself_is_read():
    """The REPL is Python, so a plan handed over as a dict is the natural form."""
    out = plan.parse_plan(json.loads(PLAN_JSON))
    assert out["ok"] is True
    assert plan.plan_path_list(out["plan"])[0] == "codetwine/analysis_cache.py"


def test_a_dict_without_files_is_still_refused():
    assert plan.parse_plan({"order": ["a.py"]})["ok"] is False


def test_a_completion_of_some_other_type_is_an_error():
    assert plan.parse_plan(42)["ok"] is False


def test_a_planning_stage_that_raises_does_not_end_the_run(stub_search, monkeypatch):
    """One bad plan must not cost the whole editing run."""
    monkeypatch.setattr(config, "TWO_STAGE", True)

    def boom(request, scope, prior_text="", on_complete=None):
        raise RuntimeError("the planner blew up")

    monkeypatch.setattr(plan, "run_plan", boom)
    rlm = _RecordingRLM()
    result = agent.run_edit(rlm, "差分解析にして")
    assert result["plan"]["ok"] is False and "RuntimeError" in result["plan"]["error"]
    assert result["answer"] == "done" and rlm.turn_note(1, 40) == ""


# == The harvest call =========================
def test_a_stage_that_runs_out_of_turns_is_asked_once_for_the_plan(monkeypatch):
    """The reading is paid for; only the hand-over is missing."""
    replies = ["```python\nprint('still reading')\n```"] * 3 + [PLAN_JSON]
    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: _scripted_llm(*replies))
    monkeypatch.setattr(config, "PLAN_MAX_TURN", 3)
    out = plan.run_plan("差分解析にして", {"files": [], "defs": []})
    assert out["ok"] is True and out["harvested"] is True
    assert plan.plan_path_list(out["plan"])[0] == "codetwine/analysis_cache.py"


def test_a_run_whose_call_never_went_through_is_not_asked_again(monkeypatch):
    """The same call would not go through for the harvest either."""
    calls = []

    def dead(prompt):
        calls.append(prompt)
        raise RuntimeError("the broker did not answer")

    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: dead)
    out = plan.run_plan("差分解析にして", {"files": [], "defs": []})
    assert len(calls) == 1
    assert out["ok"] is False and out["harvested"] is False


def test_the_harvest_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(config, "PLAN_LATE_ANSWER", False)
    monkeypatch.setattr(plan, "make_llm",
                        lambda *a, **k: _scripted_llm("```python\nprint(1)\n```"))
    monkeypatch.setattr(config, "PLAN_MAX_TURN", 2)
    out = plan.run_plan("差分解析にして", {"files": [], "defs": []})
    assert out["ok"] is False and out["harvested"] is False


def test_the_harvest_prompt_says_the_turns_are_gone_and_carries_the_history():
    class _Stage:
        instruction = "INSTR"
        tool_dict = {}
        stage = "plan"
        max_turn = 15
        call = 1

        def __init__(self):
            self.prompt = ""

        def fold(self, history_list):
            return history_list

        def llm(self, prompt):
            self.prompt = prompt
            return "{}"

    stage = _Stage()
    rlm_mod.late_answer(stage, "REQ", "VARS", [("code one", "output one")],
                        plan._LATE_ANSWER_NOTE.format(plan_max_file=8, plan_max_fact=12),
                        plan._LATE_ANSWER_CLOSE)
    assert "## Out of turns" in stage.prompt
    assert "code one" in stage.prompt and "output one" in stage.prompt
    # It comes after the history, like every turn note
    assert stage.prompt.index("output one") < stage.prompt.index("## Out of turns")


def test_a_harvest_that_says_nothing_useful_is_not_ok(monkeypatch):
    monkeypatch.setattr(plan, "make_llm",
                        lambda *a, **k: _scripted_llm("```python\nprint(1)\n```",
                                                      "{}"))
    monkeypatch.setattr(config, "PLAN_MAX_TURN", 1)
    assert plan.run_plan("差分解析にして", {"files": [], "defs": []})["ok"] is False


# == the unfinished change reaches the planning stage too ===
def test_the_planning_stage_is_given_the_previous_run(monkeypatch):
    """The plan decides where to write, so it needs to know what is already written."""
    seen = {}

    def call(prompt: str) -> str:
        seen["prompt"] = prompt
        return f"```python\nCOMPLETE({PLAN_JSON!r})\n```"

    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: call)
    plan.run_plan("差分解析にして", {"files": [], "defs": []},
                  prior_text="## Unfinished change from the previous run\nIt did a thing")
    assert "It did a thing" in seen["prompt"]
    assert "差分解析にして" in seen["prompt"]


def test_no_previous_run_leaves_the_planning_prompt_as_it_was(monkeypatch):
    seen = {}

    def call(prompt: str) -> str:
        seen["prompt"] = prompt
        return f"```python\nCOMPLETE({PLAN_JSON!r})\n```"

    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: call)
    plan.run_plan("差分解析にして", {"files": [], "defs": []})
    assert "## Request\n差分解析にして" in seen["prompt"]


# == the deadline note and the ending ========================
def test_the_deadline_note_is_the_text_it_was_measured_with():
    """It moved to rlm.make_deadline_note(); the planner must read the same words."""
    assert _plan_turn_note(18, 20) == (
        "## 3 turns left\nStop reading and call COMPLETE() with the plan as the dict itself now. "
        "A plan that is only printed is lost.")
    assert _plan_turn_note(17, 20) == ""


def test_the_instructions_offer_no_second_ending():
    """A request that needs no change is one the stage in front of the agents already
    read as a question, so this stage has one way out and it is the plan."""
    text = plan.build_plan_instruction("py")
    assert "ANSWER(" not in text
    assert 'empty "files" list' in text


def test_a_plan_that_was_handed_over_reports_that_ending(plan_llm):
    assert plan.run_plan("差分解析にして", {"files": [], "defs": []})["outcome"] == \
        COMPLETE_KEY


# == A stage that handed no plan over ========================

def test_a_planner_that_used_every_turn_is_reported_as_that(monkeypatch):
    monkeypatch.setattr(config, "PLAN_MAX_TURN", 20)
    assert plan.parse_plan(None) == {
        "ok": False,
        "error": "the planning stage handed no plan over: no COMPLETE() within 20 turns"}


def test_a_planner_whose_call_did_not_happen_is_not_reported_as_out_of_turns():
    """The run stopped at turn 1; saying it read to its last one hides the failure."""
    out = plan.parse_plan(None, "RuntimeError: the broker did not answer")
    assert out["ok"] is False
    assert "the run stopped early" in out["error"]
    assert "the broker did not answer" in out["error"]
    assert "turns" not in out["error"]


def test_the_stages_error_reaches_the_reader(stub_search, monkeypatch):
    """run_plan() hands parse_plan() what the loop said, not just the answer."""
    seen = []

    def parse(answer, error="", outcome=None):
        seen.append((error, outcome))
        return {"ok": False, "error": "x"}

    monkeypatch.setattr(plan, "parse_plan", parse)
    monkeypatch.setattr(plan.RLM, "run", lambda self, **kw: {
        "answer": None, "outcome": None, "iters": 1, "history": [],
        "error": "RuntimeError: the broker did not answer"})
    plan.run_plan("直して", {"files": [], "defs": []})
    assert seen == [("RuntimeError: the broker did not answer", None)]


def test_an_unreadable_plan_is_sent_back_once_before_it_is_lost(stub_search, monkeypatch):
    """The escapes of a JSON string inside a Python string are gone by the time the
    plan is read, and a plan lost that way leaves the editing stage without one."""
    prompts = []
    replies = iter(["```python\nCOMPLETE('nope')\n```",
                    f"```python\nCOMPLETE({PLAN_JSON!r})\n```"])

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: llm)
    out = plan.run_plan("req", stub_search)
    assert out["ok"] is True and out["iters"] == 2
    assert "the plan could not be read as JSON" in prompts[1]


def test_what_the_planning_stage_does_to_scope_does_not_reach_the_editing_stage(
        stub_search, monkeypatch, no_git):
    before = json.loads(json.dumps(stub_search))
    turn = ('```python\nscope["files"].append("x.py")\nscope["defs"].clear()\nscope.pop("confidence")\n'
            f'COMPLETE({PLAN_JSON!r})\n```')
    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: lambda prompt: turn)
    monkeypatch.setattr(config, "TWO_STAGE", True)

    class _ScopeChangingRLM(_RecordingRLM):
        """A stand-in editing stage whose code changes "scope" in place too."""

        def run(self, request, variable_dict, var_doc, **kw):
            self.variable_dict = variable_dict
            variable_dict["scope"]["files"].clear()
            return super().run(request, variable_dict, var_doc, **kw)

    rlm = _ScopeChangingRLM()
    result = agent.run_edit(rlm, "差分解析にして")
    assert result["plan"]["ok"] is True and stub_search == before
    assert rlm.variable_dict["scope"] is not stub_search
    assert result["scope"] is stub_search
    assert "Files to read:\n  codetwine/pipeline.py" in rlm.request


def test_a_plan_whose_files_name_no_path_is_sent_back_once_before_it_is_lost(
        stub_search, monkeypatch):
    prompts = []
    replies = iter(['```python\nCOMPLETE({"files": ["a.py", "b.py"]})\n```',
                    f"```python\nCOMPLETE({PLAN_JSON!r})\n```"])

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: llm)
    out = plan.run_plan("req", stub_search)
    assert out["ok"] is True and out["iters"] == 2
    assert ('COMPLETE() was not accepted: none of the 2 entries of "files" is an object '
            'with a "path"; each entry is {"path": "...", "change": "...", "anchor": '
            '"..."}. This is asked once only.') in _text(prompts[1])


def test_a_plan_sent_back_for_its_json_is_still_sent_back_once_for_its_files(
        stub_search, monkeypatch):
    """An unreadable plan does not use up the one refusal of a plan whose files hold no
    usable entry; the second such plan is taken as it is."""
    replies = iter(["```python\nCOMPLETE('nope')\n```",
                    '```python\nCOMPLETE({"files": ["a.py"]})\n```',
                    '```python\nCOMPLETE({"files": ["a.py"]})\n```'])
    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: lambda prompt: next(replies))
    out = plan.run_plan("req", stub_search)
    assert out["ok"] is False and out["iters"] == 3
    assert out["error"].startswith('none of the 1 entries of "files"')


def test_a_plan_handed_over_as_none_is_sent_back_once(stub_search, monkeypatch):
    prompts = []
    replies = iter(["```python\nCOMPLETE(None)\n```",
                    f"```python\nCOMPLETE({PLAN_JSON!r})\n```"])

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(plan, "make_llm", lambda *a, **k: llm)
    out = plan.run_plan("req", stub_search)
    assert out["ok"] is True and out["iters"] == 2 and out["harvested"] is False
    assert "COMPLETE() got None in place of the plan" in _text(prompts[1])


# == Fitting the plan to the budget ==========================
def test_a_plan_handed_over_as_a_string_over_the_budget_is_read_whole_then_fitted(
        monkeypatch):
    """The guard reads the whole string, and so does the parse: what is over the budget
    is trimmed from the parsed plan rather than lost to a JSON cut mid-way."""
    import json
    monkeypatch.setattr(config, "PLAN_ANSWER_MAX_CHAR", 400)
    handed_over = json.dumps({
        "files": [{"path": f"src/f{i}.py", "change": "x" * 40} for i in range(8)],
        "order": [f"src/f{i}.py" for i in range(8)],
        "facts": ["z" * 40 for _ in range(12)],
        "verify": "w" * 200})
    assert len(handed_over) > 400
    out = plan.parse_plan(handed_over)
    assert out["ok"] is True and out["plan"]["files"]
    assert len(json.dumps(out["plan"], ensure_ascii=False)) <= 400


def test_a_budget_of_0_leaves_the_plan_as_it_is(monkeypatch):
    monkeypatch.setattr(config, "PLAN_ANSWER_MAX_CHAR", 0)
    plan_dict = {"files": [{"path": "a.py"}] * 5, "verify": "x" * 100}
    assert plan.fit_to_budget(plan_dict) is plan_dict and len(plan_dict["files"]) == 5


def test_a_text_that_is_most_of_the_plan_is_cut_before_any_entry_is_dropped(monkeypatch):
    monkeypatch.setattr(config, "PLAN_ANSWER_MAX_CHAR", 90)
    plan_dict = {"files": [{"path": "a.py"}], "order": ["a.py"], "verify": "v" * 200}
    plan.fit_to_budget(plan_dict)
    assert plan_dict["files"] == [{"path": "a.py"}]
    assert 0 < len(plan_dict["verify"]) < 200
    assert len(json.dumps(plan_dict, ensure_ascii=False)) <= 90
    # With room for the rest of the plan, the text alone is cut, and says so
    monkeypatch.setattr(config, "PLAN_ANSWER_MAX_CHAR", 120)
    plan_dict = {"files": [{"path": "a.py"}], "order": ["a.py"], "verify": "v" * 200}
    plan.fit_to_budget(plan_dict)
    assert plan_dict["files"] == [{"path": "a.py"}] and plan_dict["order"] == ["a.py"]
    assert 115 <= len(json.dumps(plan_dict, ensure_ascii=False)) <= 120
    assert plan_dict["verify"].startswith("vvvv") \
        and re.search(r"\n\.\.\. \((\d+) chars omitted\)$", plan_dict["verify"])
    # With no room for a text and its marker, the text goes whole
    monkeypatch.setattr(config, "PLAN_ANSWER_MAX_CHAR", 60)
    plan_dict = {"files": [{"path": "a.py"}], "verify": "v" * 200}
    plan.fit_to_budget(plan_dict)
    assert plan_dict == {"files": [{"path": "a.py"}], "verify": ""}


@pytest.mark.parametrize("key", ["change", "anchor", "verify"])
def test_a_text_of_a_plan_that_is_cut_to_fit_ends_with_the_marker_of_a_cut(monkeypatch, key):
    monkeypatch.setattr(config, "PLAN_ANSWER_MAX_CHAR", 2000)
    entry_list = [{"path": "a.py", "change": "Add parse(). ", "anchor": "top"},
                  {"path": "b.py", "change": "Call it.", "anchor": "main"}]
    plan_dict = {"files": entry_list, "verify": "run it"}
    holder = plan_dict if key == "verify" else entry_list[0]
    holder[key] = "Add parse(). " * 400
    out = plan.parse_plan(plan_dict)
    assert out["ok"] is True
    text = (out["plan"] if key == "verify" else out["plan"]["files"][0])[key]
    count = int(re.search(r"\n\.\.\. \((\d+) chars omitted\)$", text)[1])
    assert len(text.rpartition("\n")[0]) + count == 13 * 400
    assert len(json.dumps(out["plan"], ensure_ascii=False)) <= 2000
    assert "chars omitted)" in plan.render_plan(out["plan"])


def test_the_order_and_the_hand_offs_of_a_plan_name_no_file_the_plan_does_not_keep(
        monkeypatch, caplog):
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 8)
    entry_list = [{"path": f"src/m{number}.py", "change": "c", "anchor": "a"}
                  for number in range(12)]
    plan_dict = {"files": entry_list,
                 "order": [f"./src/m{number}.py" for number in reversed(range(12))]
                 + ["src/other.py"],
                 "interfaces": ["src/m1.py hands a dict to src/m2.py",
                                "src/m9.py reads what src/m0.py writes",
                                "src/m10.py: nothing", "src/m1.pyx is no file of the plan"]}
    with caplog.at_level("WARNING", logger="codetwine_loom.plan"):
        out = plan.parse_plan(plan_dict)["plan"]
    assert [entry["path"] for entry in out["files"]] == [f"src/m{n}.py" for n in range(8)]
    assert out["order"] == [f"./src/m{n}.py" for n in reversed(range(8))]
    assert out["interfaces"] == ["src/m1.py hands a dict to src/m2.py",
                                 "src/m1.pyx is no file of the plan"]
    assert "the plan lists 12 files and the first 8 are kept" in caplog.text
    render = plan.render_plan(out)
    assert not re.search(r"src/m(8|9|10|11)\.py", render)


def _entry_list(*path_tuple):
    return [{"path": path, "change": "c", "anchor": "a"} for path in path_tuple]


@pytest.mark.parametrize("order_list", [
    ["1. a.py", "a.py (first)", "b.py - then", "`b.py`", "a.py -> b.py"],
    ["first add parse() to a.py", "then call it from b.py"],
    ["a.py", "./b.py", "other.py"],
])
def test_the_order_of_a_plan_within_the_cap_is_kept_as_it_was_written(monkeypatch,
                                                                     order_list):
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 8)
    out = plan.parse_plan({"files": _entry_list("a.py", "b.py"), "order": order_list})
    assert out["plan"]["order"] == order_list
    assert "Order: " + " -> ".join(order_list) in plan.render_plan(out["plan"])


def test_only_the_lines_that_name_a_file_the_cap_left_out_leave_the_order(monkeypatch):
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 2)
    out = plan.parse_plan({
        "files": _entry_list("src/a.py", "src/b.py", "src/c.py", "src/d.py"),
        "order": ["1. src/a.py", "2. src/c.py", "then `src/b.py`", "./src/d.py",
                  "src/../src/c.py", "last: src/a.py -> src/d.py", "src/c.pyx"]})["plan"]
    assert out["order"] == ["1. src/a.py", "then `src/b.py`"]
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 3)
    out = plan.parse_plan({
        "files": _entry_list("src/a.py", "src/b.py", "src/c.py", "src/d.py"),
        "order": ["1. src/a.py", "2. src/c.py", "3. src/d.py", "src/c.pyx", "4. src/b.py"]})
    assert out["plan"]["order"] == ["1. src/a.py", "2. src/c.py", "src/c.pyx"]


def test_a_path_on_both_sides_of_the_cap_is_a_file_the_plan_keeps(monkeypatch):
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 8)
    path_list = ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "g.py", "h.py", "./h.py",
                 "i.py"]
    out = plan.parse_plan({
        "files": _entry_list(*path_list), "order": ["h.py", "i.py", "a.py"],
        "interfaces": ["h.py exports run() to a.py", "a.py calls b.py", "i.py is new"]})
    assert out["plan"]["order"] == ["h.py", "a.py"]
    assert out["plan"]["interfaces"] == ["h.py exports run() to a.py", "a.py calls b.py"]


def test_a_word_of_a_longer_name_or_of_a_call_does_not_name_a_file_left_out(monkeypatch):
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 2)
    line_list = ["a.py reads config.get('k') and passes it on to b.py",
                 "b.py: main() is the entry", "a.py uses config.py", "main.py"]
    out = plan.parse_plan({"files": _entry_list("a.py", "b.py", "config", "main"),
                           "order": line_list + ["main", "config."],
                           "interfaces": line_list + ["b.py starts main.", "see config"]})
    assert out["plan"]["order"] == line_list[:2]
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 4)
    out = plan.parse_plan({"files": _entry_list("a.py", "b.py", "x.py", "y.py", "config",
                                                "main"),
                           "order": line_list + ["main", "config."],
                           "interfaces": line_list + ["b.py starts main.", "see config"]})
    assert out["plan"]["order"] == line_list
    assert out["plan"]["interfaces"] == line_list


def test_a_file_left_out_is_named_where_text_that_is_not_ascii_stands_against_it(
        monkeypatch):
    """A sentence written without spaces puts its own letters right before and right
    behind the path."""
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 2)
    name_line_list = ["src/d.pyを更新する", "次にsrc/d.pyを直す",
                      "src/a.pyからsrc/d.pyのhelper()を呼ぶ", "docs/設計.mdを直す",
                      "Makefileにターゲットを足す", "「src/d.py」、main.py の順"]
    keep_line_list = ["src/a.pyを先に直す", "main.pyを最後に直す", "lib/src/d.py は別のファイル",
                      "src/d.pyc と Makefile.am と xMakefile は残す"]
    for key in ("order", "interfaces"):
        file_list = _entry_list("src/a.py", "main.py", "src/d.py", "docs/設計.md", "Makefile")
        for line in name_line_list:
            assert plan.parse_plan({"files": file_list, key: [line]})["plan"][key] == []
        for line in keep_line_list:
            assert plan.parse_plan({"files": file_list, key: [line]})["plan"][key] == [line]


def test_the_lines_of_a_long_order_are_read_in_time_that_grows_with_their_count(
        monkeypatch):
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 8)
    monkeypatch.setattr(config, "PLAN_ANSWER_MAX_CHAR", 0)
    path_list = [f"dir{k % 89}_{k}/part{k}/file{k}.py" for k in range(1500)]
    start = time.monotonic()
    out = plan.parse_plan({
        "files": _entry_list(*path_list),
        "order": [f"step {k}: update the callers of package {k % 89}"
                  for k in range(1500)] + [f"{k}. {path}" for k, path in enumerate(path_list)],
        "interfaces": [f"{path_list[k]} hands its result to {path_list[k + 8]}"
                       for k in range(8)]})
    assert time.monotonic() - start < 5
    assert out["plan"]["order"] == [f"step {k}: update the callers of package {k % 89}"
                                    for k in range(8)]
    assert out["plan"]["interfaces"] == []


def test_under_a_dry_run_a_file_a_tool_was_asked_to_write_counts_as_touched(
        tmp_path, monkeypatch):
    """The tools write nothing under --dry-run, so the plan is measured against what
    they were asked to write."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path.parent / "out"))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "m.py").write_text("old\n")
    (tmp_path / "gone.py").write_text("x\n")
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    plan.edit.reset_patch_round()
    plan_dict = {"files": [{"path": "m.py"}, {"path": "n.py"}, {"path": "gone.py"}]}
    guard = plan.make_complete_guard(plan_dict)
    assert plan.edit.replace_in_file("m.py", "old", "new")["ok"] is True
    assert plan.edit.create_file("n.py", "x = 1\n")["ok"] is True
    note = plan.make_turn_note(plan_dict)(2, 9)
    assert "NOT TOUCHED" in note.split("gone.py")[1].splitlines()[0]
    assert "NOT TOUCHED" not in note.split("gone.py")[0]
    assert plan.edit.delete_file("gone.py")["ok"] is True
    assert plan.change_path_set() == {"m.py", "n.py", "gone.py"}
    assert "NOT TOUCHED" not in plan.make_turn_note(plan_dict)(3, 9)
    assert guard("done") == ""
    assert (tmp_path / "m.py").read_text() == "old\n" and not (tmp_path / "n.py").exists()
    # The next request starts with nothing asked for
    plan.edit.reset_patch_round()
    assert plan.change_path_set() == set()


def _plan_file_list(count: int) -> list[dict]:
    """Return the entries of a plan's "files" for that many files."""
    return [{"path": f"src/m{number}.py", "change": f"add f{number}",
             "anchor": f"f{number} [1-9]"} for number in range(count)]


@pytest.mark.parametrize("make, file_count", [
    (lambda: {"files": _plan_file_list(4), "order": ["src/m0.py"], "facts": ["x is y"],
              "verify": "run it. " + "v" * 41_000}, 4),
    (lambda: {"files": _plan_file_list(4), "facts": ["x is y"], "verify": "run it",
              "notes": "n" * 41_000}, 4),
    (lambda: {"files": [{"path": "src/new.py", "change": "write:\n" + "c" * 41_000,
                         "anchor": "new file"}] + _plan_file_list(4),
              "facts": ["x is y"], "verify": "run it"}, 5),
    (lambda: {"files": [{"path": "src/new.py", "change": "c" * 30_000},
                        {"path": "src/more.py", "change": "d" * 30_000}],
              "verify": "v" * 30_000}, 1),
])
def test_a_plan_over_the_budget_is_never_cut_down_to_one_that_changes_nothing(
        make, file_count):
    """A long "verify", a key the plan does not name, and a first entry that alone is
    over the budget: the files stay, and so does what the planner found out."""
    assert rlm_json_size(make()) > config.PLAN_ANSWER_MAX_CHAR
    out = plan.parse_plan(make())
    assert out["ok"] is True and len(out["plan"]["files"]) == file_count
    assert out["plan"]["files"][0]["path"] == make()["files"][0]["path"]
    assert set(out["plan"]) == {"files", "order", "interfaces", "facts", "verify"}
    assert rlm_json_size(out["plan"]) <= config.PLAN_ANSWER_MAX_CHAR
    if file_count > 1:
        assert out["plan"]["facts"] == ["x is y"]


def test_a_plan_whose_first_path_alone_is_over_the_budget_is_no_plan(monkeypatch):
    monkeypatch.setattr(config, "PLAN_ANSWER_MAX_CHAR", 200)
    out = plan.parse_plan({"files": [{"path": "p" * 300, "change": "x"}], "verify": "v"})
    assert out == {"ok": False, "error": (
        "the plan is over the 200 characters it may take, even with one entry of "
        '"files" and its texts cut')}


def rlm_json_size(value: object) -> int:
    """Return the length of a value as the JSON the budget is counted in."""
    return len(json.dumps(value, ensure_ascii=False))


def test_the_page_read_reaches_the_planner_only_when_it_is_switched_on(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 0)
    assert "web_fetch" not in [t.__name__ for t in plan.plan_tool_list()]
    assert "web_fetch" not in plan.build_plan_instruction("py")
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 3)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    # It reads only what a search returned, so it waits for the search
    assert "web_fetch" not in plan.build_plan_instruction("py")
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    text = plan.build_plan_instruction("py")
    assert [t.__name__ for t in plan.plan_tool_list()][-1] == "web_fetch"
    assert "web_search" in text and "web_fetch" in text
    assert "get 3 calls" in " ".join(text.split())
    assert text.count("a guess with a URL attached") == 1
