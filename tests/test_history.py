"""Carrying one change over to the next run.

The history file is written under tmp_path, so nothing here touches a real out/
directory. The one check that matters most is the leakage guard: what resolve() stores
must never reach render(), so the reason a caller that checked the result gives is
never handed to the model.
"""

import json
import logging
import os
import pathlib
import subprocess
import sys

import pytest

from codetwine_loom import config, history
from codetwine_loom.tools import edit, run, source


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    """Point the history at a fresh output directory of its own."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(source, "project_name", "codetwine")
    monkeypatch.setattr(config, "EDIT_HISTORY", True)
    return tmp_path / "out"


def _git_says(monkeypatch, added=(), modified=(), deleted=(), written=None,
              untooled=()):
    """Report what git sees, where the editing tools landed a write, and which paths
    changed without one."""
    monkeypatch.setattr(history.edit, "change_by_kind_dict",
                        lambda: {"added": list(added), "modified": list(modified),
                                 "deleted": list(deleted)})
    seen = set(added) | set(modified) | set(deleted) if written is None else set(written)
    monkeypatch.setattr(history.edit, "write_path_set", lambda: seen)
    monkeypatch.setattr(history.edit, "no_tool_path_set", lambda: set(untooled))


@pytest.fixture
def no_git(monkeypatch):
    """Report a repository where two files were changed and one was added."""
    _git_says(monkeypatch, added=["new.py"], modified=["a.py", "b.py"])


def _result(answer="wrote the go registry", last_ok=None, baseline_ok=True,
            **extra) -> dict:
    """Build a run_edit() return value with only the keys the history reads."""
    return {"answer": answer,
            "smoke": {"last_ok": last_ok, "baseline_ok": baseline_ok}, **extra}


def _landed(monkeypatch, ok=3, tests_ok=None):
    """Report that `ok` edits landed and how the model's last run_test ended."""
    monkeypatch.setattr(history.edit, "patch_count_dict",
                        lambda: {"ok": ok, "ng": 0, "ng_after_ok": 0})
    monkeypatch.setattr(history.edit, "last_test_ok", lambda: tests_ok)


# == where the file goes ====================================
def test_the_file_sits_beside_the_index(out_dir):
    assert history.path() == str(out_dir / "codetwine" / "edit_history.json")


def test_a_missing_file_reads_as_an_empty_history(out_dir):
    assert history.load()["attempts"] == []


def test_a_broken_file_does_not_stop_the_run(out_dir):
    (out_dir / "codetwine").mkdir(parents=True)
    (out_dir / "codetwine" / "edit_history.json").write_text("{ not json")
    assert history.load()["attempts"] == []


@pytest.mark.parametrize("leave", ["pipe", "directory", "deep"])
def test_a_file_that_is_no_regular_file_or_too_deep_does_not_stop_the_run(out_dir, leave):
    """A named pipe is not waited on."""
    (out_dir / "codetwine").mkdir(parents=True)
    path = out_dir / "codetwine" / "edit_history.json"
    if leave == "deep":
        path.write_text("[" * 200000)
    else:
        (os.mkfifo if leave == "pipe" else os.mkdir)(path)
    assert history.load()["attempts"] == []


def test_a_file_from_another_schema_is_ignored(out_dir):
    history.save({"schema": 99, "project": "x", "attempts": [{"n": 1}]})
    assert history.load()["attempts"] == []


@pytest.mark.parametrize("body", ['{"schema": 1, "attempts": null}',
                                  '{"schema": 1, "attempts": {"a": 1}}',
                                  '{"schema": 1, "attempts": "x"}'])
def test_attempts_that_are_not_a_list_read_as_an_empty_history(out_dir, no_git, body,
                                                             monkeypatch, caplog):
    _landed(monkeypatch)
    (out_dir / "codetwine").mkdir(parents=True)
    (out_dir / "codetwine" / "edit_history.json").write_text(body, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert history.load()["attempts"] == []
    assert "is not a list" in caplog.text
    assert history.open_attempt() is None and history.render() == ""
    assert history.last_attempt(history.path()) == {}
    history.resolve("closed")
    assert history.record("r", _result())["n"] == 1


def test_an_entry_that_is_not_an_attempt_with_a_number_is_left_out(out_dir, no_git,
                                                                   monkeypatch, caplog):
    _landed(monkeypatch)
    (out_dir / "codetwine").mkdir(parents=True)
    keep_dict = {"n": 3, "at": "2026-09-01T00:00:00", "verdict": "failed",
                 "request": "kept"}
    entry_list = [1, {}, {"n": "2"}, {"n": True}, None, keep_dict, [4]]
    (out_dir / "codetwine" / "edit_history.json").write_text(
        json.dumps({"schema": 1, "attempts": entry_list}), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert history.load()["attempts"] == [keep_dict]
    assert "6 of the 7 entries" in caplog.text
    assert history.open_attempt() is None and history.render() == ""
    history.resolve("closed")
    assert history.record("r", _result())["n"] == 4


# == judge: machine signals only ============================
def test_a_run_that_stopped_early_is_a_failure():
    verdict, why = history.judge({"edits_landed": 4, "completed": True,
                                  "tests_ok": True, "check_ok": True,
                                  "error": "RuntimeError: no call"})
    assert verdict == history.VERDICT_FAIL and "stopped before the end" in why


def test_the_error_is_asked_before_a_green_run_s_own_signals():
    """Everything else says the run finished, and it did not."""
    green = {"edits_landed": 4, "completed": True, "tests_ok": True, "check_ok": True}
    stopped = {**green, "error": "RuntimeError: no call"}
    assert history.judge(green)[0] == history.VERDICT_NO_PROOF
    assert history.judge(stopped)[0] == history.VERDICT_FAIL


def test_the_reason_a_run_stopped_early_stays_out_of_what_is_rendered():
    """why reaches the next run's prompt, so the raw message is not put in it."""
    _, why = history.judge({"edits_landed": 0, "error": "RuntimeError: no call"})
    assert "RuntimeError" not in why


def test_running_out_of_turns_is_a_failure():
    verdict, why = history.judge({"edits_landed": 4, "completed": False})
    assert verdict == history.VERDICT_FAIL and "turns ran out" in why


def test_a_harvested_answer_is_not_a_completion(out_dir, no_git, monkeypatch):
    """A run asked once what it changed used every turn - carry on from exactly that."""
    _landed(monkeypatch)
    entry = history.record("r", _result(harvested=True))
    assert entry["signals"]["completed"] is False
    assert entry["verdict"] == history.VERDICT_FAIL and "turns ran out" in entry["why"]


def test_reaching_complete_counts_as_a_completion(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    assert history.record("r", _result(harvested=False))["signals"]["completed"] is True


def test_a_failing_test_is_its_own_verdict_and_not_a_run_that_did_not_get_there():
    """A change meant to change behaviour leaves the tests asserting the old behaviour
    red however right it is, so a red test is a result to read rather than a run to
    hand out again. The merging stage's third gate is what settles it."""
    verdict, why = history.judge({"edits_landed": 4, "completed": True,
                                  "tests_ok": False})
    assert verdict == history.VERDICT_TEST_RED and verdict != history.VERDICT_FAIL
    assert "read" in why


def test_a_run_with_a_red_test_is_still_carried_over_to_the_next_one():
    """It is not finished work: whatever is red is still to be settled."""
    assert history.VERDICT_TEST_RED in history._OPEN_VERDICT_TUPLE


def test_a_program_that_does_not_run_is_a_failure():
    verdict, why = history.judge({"edits_landed": 4, "completed": True,
                                  "tests_ok": True, "check_ok": False})
    assert verdict == history.VERDICT_FAIL and "did not run" in why


def test_a_program_that_does_not_run_is_a_failure_with_a_red_test_too():
    signal_dict = {"edits_landed": 4, "completed": True, "tests_ok": False}
    verdict, why = history.judge({**signal_dict, "check_ok": False})
    assert verdict == history.VERDICT_FAIL and "did not run" in why
    assert history.judge({**signal_dict, "check_ok": True})[0] == history.VERDICT_TEST_RED
    assert history.judge({**signal_dict, "check_ok": None})[0] == history.VERDICT_TEST_RED


def test_everything_green_is_unverified_not_passed():
    """Nothing in this tool checks the request itself, so it cannot say "passed"."""
    verdict, _ = history.judge({"edits_landed": 4, "completed": True,
                                "tests_ok": True, "check_ok": True})
    assert verdict == history.VERDICT_NO_PROOF


def test_tests_that_never_ran_are_not_a_pass():
    """tests_ok None means the model never ran them, which is not evidence of a pass."""
    verdict, _ = history.judge({"edits_landed": 4, "completed": True, "tests_ok": None})
    assert verdict == history.VERDICT_NO_PROOF


# == record =================================================
def test_an_attempt_is_recorded_with_its_files(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    entry = history.record("Go を解析できるように", _result())
    assert entry["n"] == 1
    assert entry["verdict"] == history.VERDICT_NO_PROOF
    assert entry["files"] == {"added": ["new.py"], "modified": ["a.py", "b.py"],
                              "deleted": []}
    assert history.load()["attempts"][0]["did"] == "wrote the go registry"


def test_attempts_are_numbered_in_order(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    history.record("one", _result())
    assert history.record("two", _result())["n"] == 2


def test_only_the_last_entries_are_kept(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    monkeypatch.setattr(config, "EDIT_HISTORY_KEEP", 2)
    for i in range(4):
        history.record(f"r{i}", _result())
    kept = history.load()["attempts"]
    assert [a["n"] for a in kept] == [3, 4]


def test_the_request_and_the_answer_are_capped(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    monkeypatch.setattr(config, "EDIT_HISTORY_REQUEST_CHAR", 10)
    monkeypatch.setattr(config, "EDIT_HISTORY_DID_CHAR", 10)
    entry = history.record("x" * 500, _result(answer="y" * 500))
    assert entry["request"] == "x" * 10 + " ... (490 chars omitted)"
    assert entry["did"] == "y" * 10 + " ... (490 chars omitted)"


def test_newlines_are_flattened_so_one_entry_stays_one_line(out_dir, no_git,
                                                            monkeypatch):
    _landed(monkeypatch)
    entry = history.record("a\nb", _result(answer="c\n\nd"))
    assert entry["request"] == "a b" and entry["did"] == "c d"


def test_the_file_list_is_capped(out_dir, monkeypatch):
    _landed(monkeypatch)
    monkeypatch.setattr(config, "EDIT_HISTORY_MAX_FILE", 2)
    _git_says(monkeypatch, modified=[f"f{i}.py" for i in range(9)])
    assert len(history.record("r", _result())["files"]["modified"]) == 2


def test_nothing_is_recorded_when_the_history_is_off(out_dir, no_git, monkeypatch):
    """The run is still judged: the exit status follows the verdict whether or not
    the record is kept."""
    _landed(monkeypatch)
    monkeypatch.setattr(config, "EDIT_HISTORY", False)
    entry = history.record("r", _result())
    assert entry["verdict"] == history.VERDICT_NO_PROOF and entry["why"]
    assert history.load()["attempts"] == []
    assert not pathlib.Path(history.path()).exists()


def test_only_what_this_run_wrote_is_recorded(out_dir, monkeypatch):
    """A repository can start with files already changed or already untracked."""
    _landed(monkeypatch)
    _git_says(monkeypatch, added=["tests/left_over.py"], modified=["a.py", "b.py"],
              written=["a.py"])
    assert history.record("r", _result())["files"] == {"added": [], "modified": ["a.py"],
                                                       "deleted": []}


def test_a_repository_git_cannot_read_still_records(out_dir, monkeypatch):
    """The history is an aid; failing to write it must not end a run that has edited."""
    _landed(monkeypatch)

    def boom():
        raise FileNotFoundError(2, "No such file or directory", "")

    monkeypatch.setattr(history.edit, "change_by_kind_dict", boom)
    entry = history.record("r", _result())
    assert entry["files"] == {"added": [], "modified": [], "deleted": []}


# == a write that did not go through a tool ==================
def test_a_run_that_went_through_the_tools_records_no_untooled_paths(out_dir, no_git,
                                                                     monkeypatch):
    """An entry of such a run is what it always was."""
    _landed(monkeypatch)
    assert "untooled" not in history.record("r", _result())


def test_a_write_made_without_a_tool_is_recorded(out_dir, monkeypatch):
    """The patch is what git holds, so the record has to say the same."""
    _landed(monkeypatch)
    _git_says(monkeypatch, added=["by_hand.py"], modified=["a.py"], written=["a.py"],
              untooled=["by_hand.py"])
    assert history.record("r", _result())["untooled"] == ["by_hand.py"]


def test_a_file_written_without_a_tool_is_named_with_the_others(out_dir, monkeypatch):
    """The patch carries it, so the next run is told it is there."""
    _landed(monkeypatch)
    _git_says(monkeypatch, added=["by_hand.py"], modified=["a.py"], written=["a.py"],
              untooled=["by_hand.py"])
    history.record("r", _result())
    assert "It wrote: a.py, by_hand.py" in history.render()


def test_a_run_that_changed_files_only_without_a_tool_is_carried_over(out_dir,
                                                                      monkeypatch):
    """It is recorded as unverified, so it is open and the next run is asked about it."""
    _landed(monkeypatch, ok=0)
    _git_says(monkeypatch, modified=["a.py"], written=[], untooled=["a.py"])
    entry = history.record("r", _result())
    assert entry["verdict"] == history.VERDICT_NO_PROOF
    assert history.open_attempt()["n"] == entry["n"]
    assert "It wrote: a.py" in history.render("r")


def test_a_write_put_back_as_it_was_is_a_no_change_run(out_dir, tmp_path, monkeypatch):
    """Writing a file back to its starting contents leaves the repository as it was."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    for args in (["init", "-q", "-b", "main"], ["add", "."],
                 ["-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q",
                  "-m", "init"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    edit.reset_patch_round()
    assert edit.replace_in_file("a.py", "x = 1", "x = 2")["ok"]
    assert edit.replace_in_file("a.py", "x = 2", "x = 1")["ok"]
    entry = history.record("r", _result(answer="it was fine as it was"))
    assert entry["verdict"] == history.VERDICT_NO_CHANGE
    assert history.open_attempt() is None


# == what the next run sees =================================
def test_an_empty_history_renders_nothing(out_dir):
    """An empty history renders no block at all."""
    assert history.render() == ""


def test_the_history_off_renders_nothing(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    history.record("r", _result())
    monkeypatch.setattr(config, "EDIT_HISTORY", False)
    assert history.render() == ""


def test_an_unfinished_change_is_rendered(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    history.record("Go を解析できるように", _result(answer="registered go"))
    text = history.render()
    assert text.startswith("## Unfinished change from the previous run")
    assert "Go を解析できるように" in text and "registered go" in text
    assert "a.py" in text and "new.py" in text


def test_a_long_file_list_says_how_many_more(out_dir, monkeypatch):
    _landed(monkeypatch)
    _git_says(monkeypatch, modified=[f"f{i}.py" for i in range(7)])
    history.record("r", _result())
    assert "and 3 more" in history.render()


def test_a_resolved_attempt_is_no_longer_open(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    history.record("r", _result())
    history.resolve("passed")
    assert history.open_attempt() is None
    assert history.render() == ""


def test_with_the_history_off_nothing_is_open_and_nothing_is_resolved(out_dir, no_git,
                                                                     monkeypatch):
    _landed(monkeypatch)
    history.record("r", _result())
    monkeypatch.setattr(config, "EDIT_HISTORY", False)
    assert history.open_attempt() is None
    history.resolve("closed", "the user chose to start fresh")
    monkeypatch.setattr(config, "EDIT_HISTORY", True)
    assert history.open_attempt() is not None


def test_a_dry_run_resolves_nothing(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    history.record("r", _result())
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    history.resolve("closed", "the user chose to start fresh")
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    assert history.load()["attempts"][-1]["verified"] is None
    assert history.open_attempt() is not None and history.render() != ""


def test_a_dry_run_that_proposed_a_change_is_not_a_run_that_needed_none(out_dir,
                                                                       monkeypatch):
    _landed(monkeypatch)
    _git_says(monkeypatch)
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    monkeypatch.setattr(history.edit, "dry_change_list", lambda: ["--- a/a.py\n"])
    entry = history.record("r", _result())
    assert entry["verdict"] == history.VERDICT_NO_PROOF
    assert not (out_dir / "codetwine" / history.HISTORY_FILE).exists()
    monkeypatch.setattr(history.edit, "dry_change_list", lambda: [])
    assert history.record("r", _result())["verdict"] == history.VERDICT_NO_CHANGE


def test_a_resolution_of_failed_leaves_the_change_open(out_dir, no_git, monkeypatch):
    """Somebody confirming it is not done is the strongest reason to carry it over.

    A caller that resolves each attempt with its check's verdict still gets the block
    for every attempt after the first.
    """
    _landed(monkeypatch)
    history.record("r", _result())
    history.resolve("failed")
    assert history.open_attempt() is not None
    assert history.render() != ""


def test_the_user_declining_to_resume_closes_it(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    history.record("r", _result())
    history.resolve("closed", "the user chose to start fresh")
    assert history.open_attempt() is None


def test_resolving_an_empty_history_does_nothing(out_dir):
    history.resolve("passed")
    assert history.load()["attempts"] == []


def test_clear_removes_the_file(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    history.record("r", _result())
    history.clear()
    assert history.load()["attempts"] == []


# == the leakage guard (the point of resolve) ===============
def test_what_resolve_stores_never_reaches_the_model(out_dir, no_git, monkeypatch):
    """The reason given to resolve() never appears in what render() returns."""
    _landed(monkeypatch)
    history.record("Go を解析できるように", _result())
    secret = "service.go needs ['memory.go', 'types.go'] among its callees"
    history.resolve("failed", secret)
    # Reopen it the way a later attempt leaves it, so render() has something to show
    data = history.load()
    data["attempts"][-1]["verified"] = None
    history.save(data)

    text = history.render()
    assert text != ""
    assert secret not in text
    assert "memory.go" not in text


def test_render_reads_no_key_outside_the_whitelist(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    history.record("r", _result())
    data = history.load()
    data["attempts"][-1]["smuggled"] = "the answer is in usages.py"
    history.save(data)
    assert "usages.py" not in history.render()


_WHOLE_ENTRY = {"n": 1, "at": "2026-09-01T00:00:00", "verdict": "failed", "request": "r",
                "did": "d", "why": "w", "last_run": "", "verified": None,
                "files": {"added": [], "modified": ["a.py"], "deleted": []},
                "untooled": ["a.py"], "tests_changed": ["tests/test_a.py"]}


@pytest.mark.parametrize("change_dict", [
    {"at": None}, {"verdict": None}, {"at": 5}, {"verdict": ["failed"]},
    {"files": ["a.py"]}, {"files": None}, {"files": {"added": None}},
    {"files": {"modified": [1]}}, {"untooled": "a.py"}, {"tests_changed": [None]},
    {"request": 3}, {"did": {"a": 1}}, {"why": None}, {"last_run": 0}, {"verified": 1}])
def test_an_entry_holding_a_value_of_another_shape_is_left_out_whoever_reads_it(
        out_dir, no_git, monkeypatch, caplog, capsys, change_dict):
    """The start of an edit, the history subcommand and the reader of a container's
    record all read the file through read_at()."""
    from codetwine_loom import cli
    _landed(monkeypatch)
    (out_dir / "codetwine").mkdir(parents=True)
    wrong_dict = {**_WHOLE_ENTRY, "n": 2, **change_dict}
    for key, value in change_dict.items():
        if value is None and key in ("at", "verdict"):
            del wrong_dict[key]
    (out_dir / "codetwine" / "edit_history.json").write_text(
        json.dumps({"schema": 1, "attempts": [_WHOLE_ENTRY, wrong_dict]}),
        encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert history.load()["attempts"] == [_WHOLE_ENTRY]
    assert "1 of the 2 entries" in caplog.text
    assert history.open_attempt() == _WHOLE_ENTRY and history.render() != ""
    assert history.last_attempt(history.path()) == _WHOLE_ENTRY
    cli.print_history()
    assert "  1  2026-09-01T00:00:00  failed (confirmed: -)  add0 mod1 del0" in \
        capsys.readouterr().out


def test_a_history_file_of_more_bytes_than_one_is_read_in_is_an_empty_history(
        out_dir, caplog):
    (out_dir / "codetwine").mkdir(parents=True)
    path = out_dir / "codetwine" / "edit_history.json"
    record_text = json.dumps({"schema": 1, "attempts": [_WHOLE_ENTRY]})
    path.write_text(record_text + " " * (history.FILE_MAX_BYTE - len(record_text)),
                    encoding="utf-8")
    assert history.read_at(str(path))["attempts"] == [_WHOLE_ENTRY]
    path.write_text(record_text + " " * (history.FILE_MAX_BYTE - len(record_text) + 1),
                    encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert history.read_at(str(path))["attempts"] == []
    assert f"over the {history.FILE_MAX_BYTE} read of one file" in caplog.text
    assert history.last_attempt(str(path)) == {}


def test_an_entry_without_the_keys_that_may_be_missing_is_read(out_dir):
    (out_dir / "codetwine").mkdir(parents=True)
    entry_dict = {"n": 1, "at": "2026-09-01T00:00:00", "verdict": "failed"}
    (out_dir / "codetwine" / "edit_history.json").write_text(
        json.dumps({"schema": 1, "attempts": [entry_dict]}), encoding="utf-8")
    assert history.load()["attempts"] == [entry_dict]


def test_the_file_stays_small(out_dir, no_git, monkeypatch):
    """The whole point of the caps: every run reads this file."""
    _landed(monkeypatch)
    for i in range(config.EDIT_HISTORY_KEEP + 2):
        history.record("x" * 5000, _result(answer="y" * 5000))
    size = len(json.dumps(history.load()))
    assert size < 4096 * config.EDIT_HISTORY_KEEP


def test_a_failing_check_makes_the_attempt_a_failure(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    entry = history.record("r", _result(last_ok=False))
    assert entry["verdict"] == history.VERDICT_FAIL and "did not run" in entry["why"]


def test_a_check_that_was_already_red_before_the_run_is_not_read(out_dir, no_git,
                                                                 monkeypatch):
    """A baseline that failed turns the check off; it says nothing about this run."""
    _landed(monkeypatch)
    entry = history.record("r", _result(last_ok=False, baseline_ok=False))
    assert entry["verdict"] == history.VERDICT_NO_PROOF
    assert entry["signals"]["check_ok"] is None


# == the countermeasure to "this was already here" ==
def test_the_block_says_who_wrote_the_files(out_dir, no_git, monkeypatch):
    """The block says the files were written by an earlier run, not original code."""
    _landed(monkeypatch)
    history.record("Go を解析できるように", _result())
    text = history.render()
    assert "NOT" in text and "an earlier run of this same request wrote them" in text


def test_the_block_does_not_ask_for_the_work_to_be_read_back(out_dir, no_git,
                                                             monkeypatch):
    """The block says verifying the work again proves nothing, and does not ask for the
    work to be read back."""
    _landed(monkeypatch)
    history.record("r", _result())
    text = history.render()
    assert "read them before rewriting" not in text
    assert "verifying it again proves nothing" in text


def test_the_last_program_run_is_carried_over(out_dir, no_git, monkeypatch):
    """What the tool says the program printed, not what the turn around it printed."""
    _landed(monkeypatch)
    monkeypatch.setattr(run, "last_output",
                        lambda: 'ok\n{"callees": [], "callers": []}')
    entry = history.record("r", _result(history=[("print(1)", "1")]))
    assert '"callees": []' in entry["last_run"]
    assert '"callees": []' in history.render()


def test_a_turn_that_only_names_the_tool_is_not_a_run(out_dir, no_git, monkeypatch):
    """The code the model wrote is not what decides whether the program ran."""
    _landed(monkeypatch)
    run.reset_run_state()
    turns = [("print('run_project(...) comes later')",
              "run_project(...) comes later")]
    entry = history.record("r", _result(history=turns))
    assert entry["last_run"] == ""


def test_a_run_that_never_started_the_program_carries_nothing(out_dir, no_git,
                                                              monkeypatch):
    _landed(monkeypatch)
    run.reset_run_state()
    entry = history.record("r", _result(history=[("print(1)", "1")]))
    assert entry["last_run"] == ""
    assert "ran the program itself" not in history.render()


def test_the_carried_output_is_capped(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    monkeypatch.setattr(run, "last_output", lambda: "x" * 5000)
    last_run = history.record("r", _result())["last_run"]
    assert last_run == "... (4300 chars omitted)\n" + "x" * 700


# == what a follow-up request walks into ==
def test_a_finished_request_is_still_described_to_a_different_request(out_dir, no_git,
                                                                     monkeypatch):
    """A passed entry is still described to a follow-up request."""
    _landed(monkeypatch)
    history.record("Go を解析できるように", _result(answer="registered go"))
    history.resolve("passed")
    text = history.render("エイリアス import も解決できるように")
    assert text.startswith("## Earlier work already in this repository")
    assert "which is finished" in text
    assert "a.py" in text and "new.py" in text
    assert "It is not finished" not in text


def test_asking_the_finished_request_again_renders_nothing(out_dir, no_git, monkeypatch):
    """There is nothing left to carry on from when that very request is satisfied."""
    _landed(monkeypatch)
    history.record("Go を解析できるように", _result())
    history.resolve("passed")
    assert history.render("Go を解析できるように") == ""


def test_a_different_request_is_not_told_it_is_the_same_one(out_dir, no_git, monkeypatch):
    """The wording says "this same request" only to the same request."""
    _landed(monkeypatch)
    history.record("Go を解析できるように", _result())
    text = history.render("エイリアス import も解決できるように")
    assert "this same request" not in text
    assert text.startswith("## Unfinished work already in this repository")
    assert "the request below is\na new one" in text.lower()
    assert "Go を解析できるように" in text


def test_the_same_request_still_reads_as_a_continuation(out_dir, no_git, monkeypatch):
    """--attempts hands the identical text back, and that text must not change."""
    _landed(monkeypatch)
    history.record("Go を解析できるように", _result())
    assert history.render("Go を解析できるように") == history.render()


def test_a_request_that_was_cut_when_recorded_still_matches(out_dir, no_git, monkeypatch):
    """The entry keeps a flattened, capped copy, so the comparison has to use one too."""
    _landed(monkeypatch)
    monkeypatch.setattr(config, "EDIT_HISTORY_REQUEST_CHAR", 20)
    history.record("x" * 80, _result())
    assert "this same request" in history.render("x" * 80)


def test_every_attempt_s_files_are_named_not_just_the_last_one_s(out_dir, monkeypatch):
    """The block names the files of every attempt, not only those of the last one."""
    _landed(monkeypatch)
    _git_says(monkeypatch, modified=["imports.py", "usage_analysis.py"])
    history.record("r", _result())
    _git_says(monkeypatch, modified=["usages.py"])
    history.record("r", _result())
    text = history.render("r")
    assert "imports.py" in text and "usage_analysis.py" in text and "usages.py" in text


def test_an_attempt_that_wrote_nothing_is_not_what_the_block_reports(out_dir,
                                                                    monkeypatch):
    """An attempt that wrote nothing does not hide what the attempts before it wrote."""
    _landed(monkeypatch)
    _git_says(monkeypatch, modified=["imports.py"])
    history.record("r", _result(answer="wrote the go registry"))
    _git_says(monkeypatch, modified=[])
    _landed(monkeypatch, ok=0)
    history.record("r", _result(answer="I read the sources and stopped"))
    text = history.render("r")
    assert "It wrote: imports.py" in text
    assert "wrote the go registry" in text
    assert "none" not in text


def test_nothing_is_rendered_when_no_attempt_wrote_anything(out_dir, monkeypatch):
    _landed(monkeypatch, ok=0)
    _git_says(monkeypatch, modified=[])
    history.record("r", _result())
    assert history.render("r") == ""
    assert history.open_attempt() is None


def test_starting_fresh_drops_the_attempts_before_it_too(out_dir, no_git, monkeypatch):
    """"closed" is the user saying carry nothing over, not just "not this one"."""
    _landed(monkeypatch)
    history.record("one", _result())
    history.record("two", _result())
    history.resolve("closed", "the user chose to start fresh")
    assert history.render("three") == ""
    assert history.open_attempt() is None


def test_an_attempt_after_a_pass_is_carried_over_normally(out_dir, no_git, monkeypatch):
    """The pass is not a boundary: what came after it is this run's own unfinished work."""
    _landed(monkeypatch)
    history.record("one", _result())
    history.resolve("passed")
    history.record("two", _result(answer="still going"))
    text = history.render("two")
    assert text.startswith("## Unfinished change from the previous run")
    assert "still going" in text


def test_what_resolve_stores_stays_out_of_the_follow_up_block_too(out_dir, no_git,
                                                                 monkeypatch):
    """The leakage guard has to hold in every state, not only the continuation one."""
    _landed(monkeypatch)
    history.record("Go を解析できるように", _result())
    secret = "service.go needs ['memory.go', 'types.go'] among its callees"
    history.resolve("passed", secret)
    text = history.render("エイリアス import も解決できるように")
    assert text != ""
    assert secret not in text and "memory.go" not in text


def test_a_character_utf8_cannot_encode_is_written_as_its_escape(out_dir, no_git,
                                                               monkeypatch):
    _landed(monkeypatch)
    entry = history.record("name: \udce3\udc81", _result(answer="wrote \ud83d"))
    assert history.load()["attempts"] == [json.loads(json.dumps(entry))]
    with open(history.path(), encoding="utf-8") as fp:
        text = fp.read()
    assert "name: \\udce3\\udc81" in text and "wrote \\ud83d" in text


def test_a_failed_write_leaves_the_previous_history_readable(out_dir, monkeypatch):
    """A truncated file would read back as an empty history and drop the carry-over."""
    first_dict = {"n": 1, "at": "2026-09-01T00:00:00", "verdict": "failed",
                  "request": "first"}
    history.save({"schema": history.SCHEMA, "attempts": [first_dict]})
    before = history.load()
    assert before["attempts"] == [first_dict]

    def die(*a, **k):
        raise OSError("no space left on device")

    monkeypatch.setattr(history.json, "dump", die)
    history.save({"schema": history.SCHEMA,
                  "attempts": [{**first_dict, "n": 2, "request": "second"}]})
    assert history.load() == before
    # and the temporary file is not left behind
    directory = pathlib.Path(history.path()).parent
    assert [p.name for p in directory.iterdir() if p.name.endswith(".tmp")] == []


# == A run that ended itself with no change ====
def test_ending_with_no_change_is_not_a_failure():
    verdict, why = history.judge({"edits_landed": 0, "completed": True})
    assert verdict == history.VERDICT_NO_CHANGE and "needed no change" in why


def test_running_out_of_turns_with_no_change_is_still_a_failure():
    """A run that decided nothing was wrong is told apart from one that never got
    anywhere."""
    verdict, why = history.judge({"edits_landed": 0, "completed": False})
    assert verdict == history.VERDICT_FAIL and "no edit landed" in why


def test_an_ending_with_an_edit_behind_it_is_judged_as_the_change_it_is():
    """The loop reports that the run ended itself; what it means depends on what is in
    the repository."""
    verdict, _ = history.judge({"edits_landed": 2, "completed": True,
                                "tests_ok": True, "check_ok": True})
    assert verdict == history.VERDICT_NO_PROOF


def test_a_write_made_without_a_tool_is_not_a_run_that_changed_nothing(out_dir,
                                                                        monkeypatch):
    """A write the model made with its own open() is in the patch like any other, so
    the run left something behind and "the request needed no change" would be a
    commit saying the opposite of what it carries."""
    _git_says(monkeypatch, modified=["a.py"], written=[], untooled=["a.py"])
    _landed(monkeypatch, ok=0, tests_ok=True)
    entry = history.record("空のときに落ちないようにして", _result())
    assert entry["verdict"] == history.VERDICT_NO_PROOF
    assert entry["untooled"] == ["a.py"]


def test_a_no_change_run_is_recorded_but_carried_over_to_nobody(out_dir, monkeypatch):
    _git_says(monkeypatch)
    _landed(monkeypatch, ok=0)
    entry = history.record("空のときに落ちないようにして",
                           _result(answer="it already returns early"))
    assert entry["verdict"] == history.VERDICT_NO_CHANGE
    # Nothing was left in the repository, so the next run is told nothing and is not
    # asked whether to carry on from it
    assert history.render("次の依頼") == ""
    assert history.open_attempt() is None


def test_finished_work_is_not_relabelled_by_a_later_no_change_run(out_dir, no_git,
                                                                  monkeypatch):
    """The block describes the last attempt that WROTE something.

    An attempt that changed nothing coming after finished work does not make the next
    request read that work as unfinished.
    """
    _landed(monkeypatch)
    history.record("Go を解析できるように", _result(answer="registered go"))
    history.resolve(history.VERDICT_PASS)
    _git_says(monkeypatch)
    _landed(monkeypatch, ok=0)
    history.record("これは何をしている?", _result(answer="it registers go"))
    text = history.render("次の依頼")
    assert "Earlier work already in this repository" in text
    assert "not finished" not in text
    assert history.open_attempt() is None


def test_an_unfinished_change_still_carries_over_after_a_no_change_run(out_dir, no_git,
                                                                      monkeypatch):
    _landed(monkeypatch, ok=3, tests_ok=False)
    history.record("Go を解析できるように", _result(answer="half of it"))
    _git_says(monkeypatch)
    _landed(monkeypatch, ok=0)
    history.record("これは何をしている?", _result(answer="it registers go"))
    assert "It is not finished" in history.render("Go を解析できるように")
    assert history.open_attempt()["did"] == "half of it"


# == Reading a history file from outside the start-up =========

def test_the_newest_attempt_is_read_from_a_path_that_is_given(tmp_path):
    """sandbox.py reads the file without going through the start-up path() reads."""
    file_path = tmp_path / history.HISTORY_FILE
    file_path.write_text(json.dumps(
        {"schema": history.SCHEMA, "project": "codetwine",
         "attempts": [{"n": 1, "at": "2026-09-01T00:00:00", "verdict": "failed"},
                      {"n": 2, "at": "2026-09-01T00:01:00", "verdict": "unverified"}]}),
        encoding="utf-8")
    assert history.last_attempt(str(file_path))["n"] == 2


def test_a_history_that_is_not_there_has_no_newest_attempt(tmp_path):
    assert history.last_attempt(str(tmp_path / history.HISTORY_FILE)) == {}


def test_a_history_that_holds_no_attempt_has_no_newest_one(tmp_path):
    file_path = tmp_path / history.HISTORY_FILE
    file_path.write_text(json.dumps({"schema": history.SCHEMA, "project": "codetwine"}),
                         encoding="utf-8")
    assert history.last_attempt(str(file_path)) == {}


def test_the_file_this_project_writes_is_the_one_it_reads(out_dir, no_git, monkeypatch):
    _landed(monkeypatch, ok=1)
    history.record("Go を解析できるように", _result())
    newest = history.load()["attempts"][-1]
    assert history.last_attempt(history.path())["n"] == newest["n"]


# == Several runs writing one history =========================

_APPENDS = '''
import sys, time
sys.path.insert(0, {package!r})
from codetwine_loom import config, history
from codetwine_loom.tools import source

config.OUTPUT_DIR = {out!r}
config.EDIT_HISTORY_KEEP = 100
source.project_name = "codetwine"
for _ in range({each}):
    with history.open_to_update() as data:
        # What a caller does between reading and writing: another writer's entry
        # landing here is what the lock is for
        time.sleep(0.01)
        data["attempts"].append({{"n": len(data["attempts"]) + 1, "at": "",
                                 "verdict": "failed", "who": {tag!r}}})
'''


def _appending(tmp_path, out_dir, tag, each):
    """Start a process that appends to this project's history, and return it."""
    script = tmp_path / f"{tag}.py"
    script.write_text(_APPENDS.format(
        package=str(pathlib.Path(history.__file__).resolve().parent.parent),
        out=str(out_dir), tag=tag, each=each), encoding="utf-8")
    return subprocess.Popen([sys.executable, str(script)])


def test_two_runs_recording_at_once_lose_no_entry(tmp_path, out_dir):
    """Reading, adding and writing is one step under a lock: without it the later
    write is the whole file, and everything the other run wrote goes with it."""
    each = 8
    writers = [_appending(tmp_path, out_dir, tag, each) for tag in ("one", "two")]
    for writer in writers:
        assert writer.wait(timeout=60) == 0
    attempts = history.load()["attempts"]
    assert len(attempts) == 2 * each
    assert sorted(a["who"] for a in attempts) == sorted(["one"] * each + ["two"] * each)
    assert [a["n"] for a in attempts] == list(range(1, 2 * each + 1))


# == the tests one attempt changed ==========================
def test_an_attempt_that_changed_a_test_says_so(out_dir, no_git, monkeypatch):
    """Changing a test is how a run can be green without the code being right."""
    _landed(monkeypatch)
    monkeypatch.setattr(history.edit, "test_change_set",
                        lambda: {"tests/test_a.py"})
    entry = history.record("戻り値を変えて", _result())
    assert entry["tests_changed"] == ["tests/test_a.py"]


def test_an_attempt_that_left_the_tests_alone_carries_nothing_about_them(out_dir,
                                                                         no_git,
                                                                         monkeypatch):
    _landed(monkeypatch)
    monkeypatch.setattr(history.edit, "test_change_set", lambda: set())
    assert "tests_changed" not in history.record("r", _result())


# == end_text: how one run ended, said the same way everywhere ==
def test_how_a_run_ended_is_the_verdict_and_why():
    assert history.end_text({"verdict": "failed", "why": "no edit landed"}) == \
        "failed - no edit landed"


def test_a_verdict_with_no_reason_stands_alone():
    assert history.end_text({"verdict": "unverified", "why": ""}) == "unverified"


def test_nothing_recorded_is_said_as_such():
    assert history.end_text({"verdict": "", "why": "x"}) == history.NO_VERDICT_TEXT


def test_the_reason_is_cut_to_what_one_account_is_cut_to(monkeypatch):
    monkeypatch.setattr(config, "EDIT_HISTORY_DID_CHAR", 5)
    assert history.end_text({"verdict": "unsettled", "why": "y" * 20}).startswith(
        "unsettled - yyyyy\n... (15 chars omitted)")


# == A history whose directory cannot be made ===============
def test_a_history_directory_that_cannot_be_made_leaves_the_run_with_an_empty_one(
        out_dir, monkeypatch, caplog):
    def refuse(path, exist_ok=False):
        raise PermissionError("read-only")

    monkeypatch.setattr(history.os, "makedirs", refuse)
    with history.open_to_update() as data_dict:
        assert data_dict["attempts"] == []
    assert "could not be written" in caplog.text
    assert not (out_dir / "codetwine").exists()


# == what counts as an edit that landed =====================================

def test_a_run_that_reverted_everything_it_wrote_is_a_no_change_run(monkeypatch):
    """The calls returned ok, and nothing of them is in the repository."""
    monkeypatch.setattr(history.edit, "patch_count_dict",
                        lambda: {"ok": 3, "ng": 0, "ng_after_ok": 0})
    monkeypatch.setattr(history.edit, "write_path_set", set)
    monkeypatch.setattr(history.edit, "last_test_ok", lambda: None)
    signal_dict = history._signal_dict({"answer": "reverted it all"}, [])
    assert signal_dict["edits_landed"] is False
    assert history.judge(signal_dict)[0] == history.VERDICT_NO_CHANGE


def test_a_dry_run_is_a_no_change_run(monkeypatch):
    """Under --dry-run every call returns ok and nothing is written."""
    monkeypatch.setattr(history.edit, "patch_count_dict",
                        lambda: {"ok": 2, "ng": 0, "ng_after_ok": 0})
    monkeypatch.setattr(history.edit, "write_path_set", set)
    monkeypatch.setattr(history.edit, "last_test_ok", lambda: None)
    assert history._signal_dict({"answer": "done"}, [])["edits_landed"] is False


def test_a_change_written_outside_a_tool_still_counts(monkeypatch):
    monkeypatch.setattr(history.edit, "patch_count_dict",
                        lambda: {"ok": 0, "ng": 0, "ng_after_ok": 0})
    monkeypatch.setattr(history.edit, "write_path_set", set)
    monkeypatch.setattr(history.edit, "last_test_ok", lambda: None)
    assert history._signal_dict({"answer": "done"}, ["a.py"])["edits_landed"] is True


# == the check at the end of a run on its own ==================================

def test_what_the_final_check_found_is_a_run_that_failed():
    signal_dict = {"edits_landed": 1, "completed": True, "tests_ok": True}
    assert history.judge({**signal_dict, "final_ok": True})[0] == history.VERDICT_NO_PROOF
    assert history.judge({**signal_dict, "final_ok": None})[0] == history.VERDICT_NO_PROOF
    verdict, why = history.judge({**signal_dict, "final_ok": False})
    assert verdict == history.VERDICT_FAIL
    assert "was not there before the change" in why


def test_what_the_final_check_found_is_said_before_a_red_test():
    signal_dict = {"edits_landed": 1, "completed": True, "tests_ok": False}
    assert history.judge({**signal_dict, "final_ok": False})[0] == history.VERDICT_FAIL
    assert history.judge({**signal_dict, "final_ok": True})[0] == history.VERDICT_TEST_RED


def test_the_entry_keeps_what_the_final_check_found(out_dir, no_git, monkeypatch):
    _landed(monkeypatch)
    entry = history.record("r", {**_result(harvested=False), "final_check": ["- x"]})
    assert entry["final_check"] == ["- x"] and entry["signals"]["final_ok"] is False
    entry = history.record("r", {**_result(harvested=False), "final_check": []})
    assert "final_check" not in entry and entry["signals"]["final_ok"] is True
    entry = history.record("r", _result(harvested=False))
    assert entry["signals"]["final_ok"] is None


def test_what_the_record_cut_onto_one_line_is_not_cut_again_by_a_later_stage():
    """The count of what was left out stays the one the record wrote."""
    from codetwine_loom import textcut
    answer = "did this. " * 320
    once = history._cut(answer, 800)
    assert once.endswith("... (2399 chars omitted)") and "\n" not in once
    assert history._cut(once, 800) == once and textcut.cut(once, 800) == once
    assert textcut.cut(f"{once}\n", 800) == once.replace(" ... (2399", "\n... (2400")
