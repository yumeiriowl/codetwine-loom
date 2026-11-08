"""The automatic check that starts the program after an edit.

The checks run a tiny Python program written into tmp_path as the "project": main.py
imports lib.py and calls it, so an edit that breaks lib.py makes main.py fail (a name
that is not defined, an argument the callee does not accept).
config.RUN_COMMAND points at the current interpreter, so no build tool is involved.
"""

import subprocess
import sys

import pytest

from codetwine_loom import agent, config, smoke
from codetwine_loom.tools import edit, run

# The stand-in for the target program. It calls into lib.py, which is the file the tests
# edit, and writes a file so a run can be told apart from a start that did nothing
MAIN = """import os, sys
from lib import value

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
# value() runs before the file is opened, so a start can tell whether the start before
# it left anything behind
written = value()
with open(os.path.join(out_dir, "result.txt"), "a", encoding="utf-8") as fp:
    fp.write(str(written) + "\\n")
print("ran")
"""

LIB = """def value():
    return 1
"""

ARGS = ["main.py", "{workspace}/out"]


def _git(root, *args) -> str:
    """Run git for the tests."""
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A git repository holding main.py and lib.py, with the check configured for it."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "main.py").write_text(MAIN)
    (root / "lib.py").write_text(LIB)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    monkeypatch.setattr(config, "RUN_COMMAND", [sys.executable])
    monkeypatch.setattr(config, "SMOKE_ARG", list(ARGS))
    monkeypatch.setattr(config, "SMOKE_FILE", {})
    monkeypatch.setattr(config, "SMOKE_ENV", {})
    monkeypatch.setattr(config, "SMOKE_REPEAT", 1)
    monkeypatch.setattr(config, "SMOKE_COMPLETE_GUARD", True)
    edit.reset_patch_round()
    run.reset_run_state()
    smoke.reset_smoke_state()
    yield root
    run.reset_run_state()


def _break_the_program(root) -> None:
    """Edit lib.py through the editing tool so it raises a NameError when called."""
    result = edit.replace_in_file("lib.py", "    return 1", "    return missing_name")
    assert result["ok"] is True


# == When the check is on at all ============================
def test_no_check_without_arguments_to_start_the_program(project, monkeypatch):
    monkeypatch.setattr(config, "SMOKE_ARG", [])
    assert smoke.is_on() is False
    assert smoke.make_after_turn() is None and smoke.make_complete_guard() is None


def test_no_check_under_dry_run(project, monkeypatch):
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    assert smoke.is_on() is False


def test_arguments_with_no_launcher_to_start_them_are_warned_about(project, monkeypatch,
                                                                   caplog):
    monkeypatch.setattr(config, "RUN_COMMAND", [])
    assert smoke.check_baseline() is False
    assert caplog.text.count("the automatic check is off: SMOKE_ARG is set, but "
                             "RUN_COMMAND is not") == 1
    caplog.clear()
    # Nothing is said under a dry run, and nothing without the arguments
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    assert smoke.check_baseline() is False
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    monkeypatch.setattr(config, "SMOKE_ARG", [])
    assert smoke.check_baseline() is False
    assert "the automatic check is off" not in caplog.text


def test_a_sample_the_check_reaches_through_the_environment_keeps_the_check_on(
        project, monkeypatch):
    (project / "by_env.py").write_text(
        "import os\nprint(open(os.path.join(os.environ['DATA_DIR'], 'a.txt')).read())\n")
    monkeypatch.setattr(config, "SMOKE_ARG", ["by_env.py"])
    monkeypatch.setattr(config, "SMOKE_FILE", {"sample/a.txt": "x\n"})
    monkeypatch.setattr(config, "SMOKE_ENV", {"DATA_DIR": "{workspace}/sample"})
    assert smoke.check_baseline() is True


def test_the_baseline_turns_the_check_on_when_the_program_runs(project):
    assert smoke.check_baseline() is True
    assert smoke.is_active() is True
    assert smoke.count_dict()["baseline_ok"] is True


def test_a_program_that_was_already_broken_is_not_checked(project, caplog):
    (project / "lib.py").write_text("def value():\n    return missing_name\n")
    assert smoke.check_baseline() is False
    assert smoke.is_active() is False
    assert "already failed before any edit" in caplog.text


def test_the_baseline_does_not_count_as_the_model_running_the_program(project):
    smoke.check_baseline()
    assert run.run_count() == 0


def test_the_check_keeps_its_own_workspace(project):
    made = run.run_project(["main.py", "{workspace}/out"])
    assert made["ok"] is True
    smoke.check_baseline()
    # The model's sample is still there: a check must not empty the workspace the model
    # built its sample in
    assert run.run_project(["main.py", "{workspace}/out"])["produced"] == []


# == The per-turn hook ======================================
def test_a_turn_that_edited_nothing_is_not_checked(project):
    smoke.check_baseline()
    after_turn = smoke.make_after_turn()
    assert after_turn(3, "printed something") == ""
    assert smoke.count_dict()["checks"] == 1        # the baseline only


def test_an_edit_that_keeps_the_program_running_gets_one_line_back(project):
    smoke.check_baseline()
    after_turn = smoke.make_after_turn()
    edit.replace_in_file("lib.py", "    return 1", "    return 2")
    note = after_turn(4, "ok")
    assert note.startswith("## Automatic check\n") and "exited 0" in note
    assert smoke.count_dict()["checks"] == 2 and smoke.count_dict()["failures"] == 0


def test_the_check_output_says_when_it_was_cut(project, monkeypatch):
    monkeypatch.setattr(config, "SMOKE_OUTPUT_TAIL", 3)
    smoke.check_baseline()
    after_turn = smoke.make_after_turn()
    _break_the_program(project)
    note = after_turn(4, "ok")
    assert "... (" in note and "chars omitted)" in note
    assert smoke._state_dict["last_output"].startswith("... (")
    assert len(smoke._state_dict["last_output"].split("\n", 1)[1]) == 3


def test_a_change_that_produces_nothing_new_is_said_so(project):
    """A check after a change that leaves what the program produces as it was says so,
    not only "exited 0"."""
    smoke.check_baseline()
    after_turn = smoke.make_after_turn()
    edit.replace_in_file("lib.py", "    return 1", "    return 2")
    note = after_turn(4, "ok")
    assert "nothing you have written has changed what it produces yet" in note


def test_a_change_that_writes_a_new_file_is_said_so(project):
    smoke.check_baseline()
    after_turn = smoke.make_after_turn()
    edit.replace_in_file(
        "main.py", 'print("ran")',
        'open(os.path.join(out_dir, "extra.txt"), "w").write("x")\nprint("ran")')
    note = after_turn(4, "ok")
    assert "What it writes has changed" in note and "extra.txt" in note


def test_nothing_is_compared_when_there_is_no_baseline(project):
    """A run whose baseline never happened has nothing to compare with."""
    assert smoke._output_line(["out/a.txt"]) == ""


def test_an_edit_that_breaks_the_program_comes_back_with_the_failure(project):
    smoke.check_baseline()
    after_turn = smoke.make_after_turn()
    _break_the_program(project)
    note = after_turn(5, "ok")
    assert "FAILED" in note and "NameError" in note
    assert smoke.count_dict()["failures"] == 1 and smoke.count_dict()["last_ok"] is False


def test_a_failure_that_printed_a_fence_is_fenced_longer(project, monkeypatch):
    monkeypatch.setitem(smoke._state_dict, "last_why", "exit status 1")
    monkeypatch.setitem(smoke._state_dict, "last_output", "```\nx\n```")
    assert "\n````\n```\nx\n```\n````" in smoke._render({"ok": False})


def test_the_check_does_not_run_twice_for_one_edit(project):
    smoke.check_baseline()
    after_turn = smoke.make_after_turn()
    edit.replace_in_file("lib.py", "    return 1", "    return 2")
    after_turn(4, "ok")
    assert after_turn(5, "ok") == ""
    assert smoke.count_dict()["checks"] == 2


def test_a_revert_that_puts_the_program_back_is_checked_again(project):
    """revert_file() lands no edit, and the last check's failure is what the guard and
    the record read."""
    smoke.check_baseline()
    after_turn = smoke.make_after_turn()
    guard = smoke.make_complete_guard()
    edit.replace_in_file("main.py", 'print("ran")', 'print("ran again")')
    _break_the_program(project)
    assert "FAILED" in after_turn(4, "ok")
    assert edit.revert_file("lib.py")["ok"] is True
    assert "exited 0" in after_turn(5, "ok")
    assert smoke.count_dict()["last_ok"] is True
    assert guard("reverted") == ""


def test_a_write_made_without_a_tool_is_checked(project):
    smoke.check_baseline()
    after_turn = smoke.make_after_turn()
    (project / "lib.py").write_text("def value():\n    return missing_name\n")
    assert "FAILED" in after_turn(4, "ok")
    assert after_turn(5, "ok") == ""
    assert smoke.count_dict()["checks"] == 2


def test_nothing_is_checked_while_the_baseline_is_failing(project):
    (project / "lib.py").write_text("def value():\n    return missing_name\n")
    smoke.check_baseline()
    after_turn = smoke.make_after_turn()
    edit.replace_in_file("lib.py", "missing_name", "3")
    assert after_turn(4, "ok") == ""


# == COMPLETE while the program does not run ==================
def test_complete_is_refused_once_while_the_program_does_not_run(project):
    smoke.check_baseline()
    guard = smoke.make_complete_guard()
    _break_the_program(project)
    smoke.make_after_turn()(5, "ok")
    refusal = guard("I changed lib.py")
    assert "COMPLETE() was not accepted" in refusal and "NameError" in refusal
    # It gives way the second time: the model may have a reason
    assert guard("I changed lib.py") == ""
    assert smoke.count_dict()["refusals"] == 1


def test_complete_checks_an_edit_that_landed_on_the_completing_turn_itself(project):
    # The edit lands on the same turn that calls COMPLETE()
    smoke.check_baseline()
    guard = smoke.make_complete_guard()
    _break_the_program(project)
    assert "COMPLETE() was not accepted" in guard("done")
    assert smoke.count_dict()["checks"] == 2


def test_complete_is_accepted_while_the_program_runs(project):
    smoke.check_baseline()
    guard = smoke.make_complete_guard()
    edit.replace_in_file("lib.py", "    return 1", "    return 2")
    assert guard("done") == ""


def test_the_guard_can_be_switched_off(project, monkeypatch):
    monkeypatch.setattr(config, "SMOKE_COMPLETE_GUARD", False)
    smoke.check_baseline()
    guard = smoke.make_complete_guard()
    _break_the_program(project)
    assert guard("done") == ""
    # The completing turn's edit is checked all the same, and the outcome is kept
    count_dict = smoke.count_dict()
    assert (count_dict["checks"], count_dict["last_ok"], count_dict["refusals"]) == (
        2, False, 0)


def test_an_edit_on_the_completing_turn_is_checked_after_the_one_refusal(project):
    smoke.check_baseline()
    guard = smoke.make_complete_guard()
    _break_the_program(project)
    assert "COMPLETE() was not accepted" in guard("done")
    # The fix and the second COMPLETE() come in one turn
    assert edit.replace_in_file("lib.py", "missing_name", "2")["ok"] is True
    assert guard("fixed") == ""
    count_dict = smoke.count_dict()
    assert (count_dict["checks"], count_dict["last_ok"], count_dict["refusals"]) == (
        3, True, 1)
    # And an edit that breaks it again on a later completing turn is recorded
    _break_the_program_again = edit.replace_in_file("lib.py", "return 2",
                                                    "return missing_name")
    assert _break_the_program_again["ok"] is True
    assert guard("done") == ""
    assert smoke.count_dict()["last_ok"] is False
    # Nothing changed since: no further start
    assert guard("done") == "" and smoke.count_dict()["checks"] == 4


# == Starting the program more than once ====================
def test_a_second_start_catches_what_the_first_one_cannot(project, monkeypatch):
    # main.py appends to result.txt, so the file is only there on the second start.
    # A feature that reads back what the run before it wrote (a cache) fails here
    monkeypatch.setattr(config, "SMOKE_REPEAT", 2)
    (project / "lib.py").write_text(
        "import os, sys\n\n\ndef value():\n"
        "    if os.path.exists(os.path.join(sys.argv[1], 'result.txt')):\n"
        "        raise RuntimeError('the second start blew up')\n"
        "    return 1\n")
    assert smoke.check_baseline() is False
    assert "on start 2 of 2" in smoke.count_dict()["last_why"]


def test_what_a_check_wrote_is_what_every_start_of_it_wrote(project, monkeypatch):
    """main.py writes out/result.txt, which the second start finds already there."""
    monkeypatch.setattr(config, "SMOKE_REPEAT", 2)
    assert smoke.check_baseline() is True
    assert smoke._state_dict["baseline_produced"] == ["out/result.txt"]
    edit.replace_in_file(
        "main.py", 'print("ran")',
        'open(os.path.join(sys.argv[1], "more.txt"), "w").close()\nprint("ran")')
    note = smoke.make_after_turn()(1, "")
    assert "1 new (out/more.txt)" in note and "no longer written" not in note


def test_one_start_does_not_see_the_second_start_failure(project, monkeypatch):
    monkeypatch.setattr(config, "SMOKE_REPEAT", 1)
    (project / "lib.py").write_text(
        "import os, sys\n\n\ndef value():\n"
        "    if os.path.exists(os.path.join(sys.argv[1], 'result.txt')):\n"
        "        raise RuntimeError('the second start blew up')\n"
        "    return 1\n")
    assert smoke.check_baseline() is True


# == What the model is told about it ========================
def test_the_instructions_describe_the_check_only_while_it_is_running(project):
    assert agent.check_note() == ""             # the baseline has not run yet
    smoke.check_baseline()
    note = agent.check_note()
    assert "3.6." in note and "started" in note
    assert note in agent.build_instruction("py")


def test_the_instructions_stay_silent_when_the_baseline_failed(project):
    (project / "lib.py").write_text("def value():\n    return missing_name\n")
    smoke.check_baseline()
    assert agent.check_note() == ""


# == Chaining the two guards ================================
def test_the_first_refusal_wins():
    combined = agent.complete_guard_chain(lambda a: "",
                                     lambda a: "no, the program is broken")
    assert combined("done") == "no, the program is broken"


def test_no_guard_at_all_stays_none():
    assert agent.complete_guard_chain(None, None) is None


def test_a_file_the_baseline_wrote_and_this_run_did_not_is_said(monkeypatch):
    monkeypatch.setitem(smoke._state_dict, "baseline_produced", ["a.txt", "b.txt"])
    line = smoke._output_line(["a.txt"])
    assert "1 no longer written (b.txt)" in line and "new" not in line


def test_a_file_this_run_wrote_and_the_baseline_did_not_is_said(monkeypatch):
    monkeypatch.setitem(smoke._state_dict, "baseline_produced", ["a.txt"])
    line = smoke._output_line(["a.txt", "c.txt"])
    assert "1 new (c.txt)" in line


def test_without_a_baseline_nothing_is_said(monkeypatch):
    monkeypatch.setitem(smoke._state_dict, "baseline_produced", None)
    assert smoke._output_line(["a.txt"]) == ""
