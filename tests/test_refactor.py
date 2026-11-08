"""Refactoring what a combined change touched, and the tests of that change."""

import collections
import logging
import os
import re
import stat
import subprocess
import sys

import pytest

from codetwine_loom import (agent, config, gitcmd, guide, integrate, refactor,
                             testcreate)
from codetwine_loom.languages import LANG_DICT
from codetwine_loom.tools import edit

_CHANGE = integrate.REFACTOR_STAGE
_TEST = integrate.TEST_REFACTOR_STAGE

_X = "p/x_py/x.py"
_Y = "p/y_py/y.py"
_RANGE_DICT = {_X: [(1, 4, "f"), (5, 8, "g")], _Y: [(1, 2, "h")]}
_START_DICT = {"no_resolve": {}, "name": {}, "range": _RANGE_DICT, "ref": [], "use": [],
               "signature": {}}


def _git(directory, *args):
    """Run one git command in a test repository and fail the test when it does not."""
    proc = subprocess.run(["git", "-C", str(directory), *args],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"git {args[0]} failed: {proc.stderr}"
    return proc.stdout


def _write(directory, name, text):
    """Write one file in a test repository, making its directory."""
    path = os.path.join(str(directory), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _read(directory, name):
    """Read one file of a test repository."""
    with open(os.path.join(str(directory), name), encoding="utf-8") as handle:
        return handle.read()


def _test_result(status="passed", failed=(), excerpt=""):
    """One run_test() return value."""
    return {"status": status, "failed": list(failed), "failed_excerpt": excerpt,
            "targets": [], "output": ""}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository whose HEAD changes f in x.py, adds y.py, changes one test file, adds
    one and deletes one.

    Returns:
        (path, the commit HEAD was written against).
    """
    directory = str(tmp_path / "p")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    _write(directory, "x.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n\n\n")
    _write(directory, "tests/test_x.py", "def test_f():\n    pass\n")
    _write(directory, "tests/test_old.py", "def test_old():\n    pass\n")
    _write(directory, "tests/test_gone.py", "def test_gone():\n    pass\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _write(directory, "x.py", "def f():\n    return 10\n\n\ndef g():\n    return 2\n\n\n")
    _write(directory, "y.py", "def h():\n    return 3\n")
    _write(directory, "tests/test_x.py", "def test_f_case_1():\n    assert True\n")
    _write(directory, "tests/test_y.py", "def test_h():  # case_2\n    assert True\n")
    os.remove(os.path.join(directory, "tests/test_gone.py"))
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "change")
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    monkeypatch.setattr(config, "EDIT_GUIDE", "")
    monkeypatch.setattr(config, "TEST_GUIDE", "")
    monkeypatch.setattr(config, "REFACTOR_GUIDE", "")
    monkeypatch.setattr(config, "TEST_REFACTOR_GUIDE", "")
    guide.reset()
    return directory, base


# == What a stage refactors and where it writes ==============


def test_the_change_is_refactored_in_the_files_whose_definitions_it_touched(repo):
    directory, base = repo
    assert refactor.target_dict_of(_CHANGE, directory, base, _RANGE_DICT, "p") == \
        {"x.py": ["f"], "y.py": ["h"]}


def test_a_stage_told_so_refactors_only_the_files_the_test_command_runs(repo,
                                                                      monkeypatch, caplog):
    directory, base = repo
    monkeypatch.setattr(refactor.edit, "runnable_path_list",
                        lambda path_list: [path for path in path_list if path == "y.py"])
    with caplog.at_level(logging.INFO, logger="codetwine_loom.testcreate"):
        assert refactor.target_dict_of(_CHANGE, directory, base, _RANGE_DICT, "p",
                                       runnable_only=True) == {"y.py": ["h"]}
    assert "no refactoring is done for x.py" in caplog.text
    monkeypatch.setattr(refactor.edit, "runnable_path_list",
                        lambda path_list: [path for path in path_list
                                           if path == "tests/test_y.py"])
    assert refactor.target_dict_of(_TEST, directory, base, _RANGE_DICT, "p",
                                   runnable_only=True) == {"tests/test_y.py": []}


@pytest.mark.parametrize("runnable_only", [True, False])
def test_the_run_hands_on_whether_to_keep_only_the_files_the_tests_run(
        repo, monkeypatch, runnable_only):
    directory, base = repo
    seen = []
    monkeypatch.setattr(integrate, "analyse", lambda _: _START_DICT)
    monkeypatch.setattr(refactor, "target_dict_of",
                        lambda *a: seen.append(a[-1]) or {})
    refactor.run_refactor(_CHANGE, base, "", "p", "out", runnable_only=runnable_only)
    assert seen == [runnable_only]


def test_the_tests_refactored_are_the_ones_the_change_added_or_changed(repo):
    """A test file the change deleted and one it left alone are not among them."""
    directory, base = repo
    assert refactor.target_dict_of(_TEST, directory, base, _RANGE_DICT, "p") == \
        {"tests/test_x.py": [], "tests/test_y.py": []}


def test_the_stage_on_the_change_writes_its_files_and_nothing_new(repo):
    may_write = refactor.may_write_of(_CHANGE, ["x.py", "y.py"],
                                      ["x.py", "y.py", "z.py", "tests/test_x.py"])
    assert [may_write(path) for path in ("x.py", "y.py")] == [True, True]
    assert [may_write(path) for path in ("z.py", "new.py", "tests/test_x.py",
                                         "tests/test_new.py")] == [False] * 4


def test_the_stage_on_the_tests_writes_its_files_and_new_files_under_the_tests(repo):
    may_write = refactor.may_write_of(_TEST, ["tests/test_x.py"],
                                      ["x.py", "tests/test_x.py", "tests/test_old.py"])
    assert [may_write(path) for path in ("tests/test_x.py", "tests/conftest.py")] == \
        [True, True]
    assert [may_write(path) for path in ("tests/test_old.py", "x.py", "new.py")] == [False] * 3


@pytest.mark.parametrize("stage, new_file_text", [
    (_CHANGE, "lists."), (_TEST, "lists and new files under the tests' directory.")])
def test_a_refused_write_says_what_this_run_writes(repo, stage, new_file_text):
    directory, _ = repo
    edit.reset_patch_round(round_limit=0)
    refuse_list = []
    tool_dict = {tool.__name__: tool for tool in refactor.stage_tool_list(
        stage, lambda path: False, refuse_list)}
    before = _read(directory, "x.py")
    answer = tool_dict["replace_in_file"]("x.py", "return 10", "return 11")
    assert answer["ok"] is False and _read(directory, "x.py") == before
    assert f'writes the files work["targets"] {new_file_text}' in answer["error"]
    assert "{" not in answer["error"] and refuse_list == ["x.py"]


# == When no model is called =================================


class _StageThatRuns:
    """A stand-in for RLM that records how it was built, does what act says and then
    calls COMPLETE() until the guard takes it."""

    built_list = []
    act = None

    def __init__(self, *args, **kwargs):
        self.tool_dict = {tool.__name__: tool for tool in kwargs["tool_list"]}
        _StageThatRuns.built_list.append(
            {"stage": kwargs.get("stage"), "instruction": kwargs["instruction"],
             "tools": sorted(self.tool_dict)})

    def run(self, **kwargs):
        _StageThatRuns.built_list[-1]["request"] = kwargs["request"]
        _StageThatRuns.built_list[-1]["work"] = kwargs["variable_dict"]["work"]
        if type(self).act:
            type(self).act(self.tool_dict)
        refuse_count = 0
        while kwargs["on_complete"]("nothing needed changing"):
            refuse_count += 1
        return {"answer": "nothing needed changing", "history": [], "iters": 1,
                "refused_completes": refuse_count, "error": "", "outcome": "complete"}


def _run_of(stage, directory, base, monkeypatch, test_list, act=None):
    """Run run_refactor() with the analysis stubbed and run_test() answering from a
    list, oldest first and the last one again after that."""
    monkeypatch.setattr(integrate, "analyse", lambda _: _START_DICT)
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    answer_list = list(test_list)
    monkeypatch.setattr(edit, "run_test", lambda *a, **k: answer_list.pop(0)
                        if len(answer_list) > 1 else answer_list[0])
    monkeypatch.setattr(refactor, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(refactor, "RLM", _StageThatRuns)
    _StageThatRuns.built_list.clear()
    monkeypatch.setattr(_StageThatRuns, "act", staticmethod(act) if act else None)
    return refactor.run_refactor(stage, base, "", "p", "out", request="change f")


def test_a_change_with_nothing_to_refactor_calls_no_model(repo, monkeypatch):
    directory, _ = repo
    head = _git(directory, "rev-parse", "HEAD").strip()
    for stage in refactor.STAGE_TUPLE:
        out_dict = _run_of(stage, directory, head, monkeypatch, [_test_result()])
        assert (out_dict["called"], out_dict["settled"]) == (False, True)
        assert out_dict["skipped"] == refactor._SKIP_NO_TARGET_DICT[stage]
    assert _StageThatRuns.built_list == []


def test_tests_whose_run_over_the_base_names_a_part_of_them_call_no_model(repo,
                                                                         monkeypatch):
    """The run over the base's tree ended at its first failing test; the stage that
    refactors the change does not read that run and goes on."""
    directory, base = repo
    monkeypatch.setattr(testcreate, "base_run_dict", lambda *args: {
        "status": testcreate.BASE_RED, "failed": ["tests/test_x.py::test_f"],
        "is_part": True, "left_out": [],
        "is_no_run": False})
    out_dict = _run_of(integrate.TEST_REFACTOR_STAGE, directory, base, monkeypatch,
                       [_test_result()])
    assert (out_dict["called"], out_dict["settled"]) == (False, True)
    assert out_dict["skipped"] == refactor._SKIP_BASE_PART
    assert _StageThatRuns.built_list == []
    out_dict = _run_of(integrate.REFACTOR_STAGE, directory, base, monkeypatch,
                       [_test_result()])
    assert out_dict["called"] is True and out_dict["skipped"] == ""
    # Every name the run could give: the stage runs
    monkeypatch.setattr(testcreate, "base_run_dict", lambda *args: {
        "status": testcreate.BASE_RED, "failed": ["tests/test_x.py::test_f"],
        "is_part": False, "left_out": [],
        "is_no_run": False})
    out_dict = _run_of(integrate.TEST_REFACTOR_STAGE, directory, base, monkeypatch,
                       [_test_result()])
    assert out_dict["called"] is True and out_dict["skipped"] == ""


def test_tests_that_reach_no_verdict_over_the_base_call_no_model(repo, monkeypatch):
    """The tests did not run over the base's tree, or its tree could not be put in
    place; the stage that refactors the change does not read that run and goes on."""
    directory, base = repo
    no_run_dict = {"status": testcreate.BASE_UNKNOWN, "failed": [], "is_part": False,
                   "left_out": [], "is_no_run": True}
    monkeypatch.setattr(testcreate, "base_run_dict", lambda *args: no_run_dict)
    out_dict = _run_of(integrate.TEST_REFACTOR_STAGE, directory, base, monkeypatch,
                       [_test_result()])
    assert (out_dict["called"], out_dict["settled"]) == (False, True)
    assert out_dict["skipped"] == refactor._SKIP_BASE_NO_RUN
    assert _StageThatRuns.built_list == []
    out_dict = _run_of(integrate.REFACTOR_STAGE, directory, base, monkeypatch,
                       [_test_result()])
    assert out_dict["called"] is True and out_dict["skipped"] == ""
    # A run that reached a verdict and names no test but the ones failing before
    monkeypatch.setattr(testcreate, "base_run_dict",
                        lambda *args: {**no_run_dict, "is_no_run": False})
    out_dict = _run_of(integrate.TEST_REFACTOR_STAGE, directory, base, monkeypatch,
                       [_test_result()])
    assert out_dict["called"] is True and out_dict["skipped"] == ""


_A_LINE = "Execution failed for task ':a:compileKotlin'."
_B_LINE = "Execution failed for task ':b:compileKotlin'."
_PASS_LINE_LIST = [refactor._BASE_PASS_REASON.format(tests="ATest > one(), ATest > two()")]


@pytest.mark.parametrize("start_list, end_list, line_list", [
    # Every test was run both times
    ([], [], _PASS_LINE_LIST),
    # The run made afterwards left out what the first one did, or less
    ([_A_LINE], [_A_LINE], _PASS_LINE_LIST),
    ([_A_LINE, _B_LINE], [_B_LINE, _A_LINE], _PASS_LINE_LIST),
    ([_A_LINE, _B_LINE], [_A_LINE], _PASS_LINE_LIST),
    ([_A_LINE], [], _PASS_LINE_LIST),
    # It left out something the first one did not: that is what is left, and no test
    # is said to fail no more
    ([], [_A_LINE], [refactor._BASE_LEFT_OUT_REASON.format(lines=_A_LINE)]),
    ([_A_LINE], [_B_LINE], [refactor._BASE_LEFT_OUT_REASON.format(lines=_B_LINE)]),
    ([_A_LINE], [_A_LINE, _B_LINE], [refactor._BASE_LEFT_OUT_REASON.format(lines=_B_LINE)]),
    ([_A_LINE], [_A_LINE, _A_LINE], [refactor._BASE_LEFT_OUT_REASON.format(lines=_A_LINE)]),
    ([], [_B_LINE, _A_LINE],
     [refactor._BASE_LEFT_OUT_REASON.format(lines=f"{_A_LINE}, {_B_LINE}")]),
])
def test_a_test_missing_from_a_run_over_the_base_that_left_tests_out_is_not_read_as_passing(
        repo, monkeypatch, start_list, end_list, line_list):
    """The run made when the stage started names the tests of two modules; the run
    made afterwards names those of one, each saying what it left out."""
    directory, base = repo
    start_dict = {"status": testcreate.BASE_RED, "left_out": start_list,
                  "failed": ["ATest > one()", "ATest > two()", "ZTest > one()"]}
    end_dict = {"status": testcreate.BASE_RED, "failed": ["ZTest > one()"],
                "left_out": end_list}
    def change():
        _write(directory, "tests/test_x.py", "def test_f_case_1():\n    pass\n")

    got_list, _, _ = _check_of(_TEST, directory, base, monkeypatch,
                               [_test_result(), _test_result()], change,
                               base_list=[start_dict, end_dict])
    assert got_list == line_list
    # A run that turned green is another status whichever way it ended
    got_list, _, _ = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_x.py", "def test_f_case_1():\n    return\n"),
        base_list=[start_dict, {"status": testcreate.BASE_GREEN, "failed": [],
                                "left_out": end_list}])
    assert got_list == [refactor._BASE_STATUS_REASON.format(
        before=testcreate.BASE_RED, after=testcreate.BASE_GREEN)]


def test_a_run_over_the_base_that_leaves_out_more_than_it_did_is_what_is_left(
        repo, monkeypatch):
    """The run made afterwards is red without naming a test, having left out what the
    run made when the stage started ran."""
    directory, base = repo
    start_dict = {"status": testcreate.BASE_RED, "left_out": [],
                  "failed": ["ATest > one()"]}
    got_list, _, _ = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_x.py", "def test_f_case_1():\n    pass\n"),
        base_list=[start_dict, {"status": testcreate.BASE_RED, "failed": [],
                                "left_out": [_A_LINE]}])
    assert got_list == [
        "- run over the code as it was before the change, the tests now leave out what "
        "they ran when you started; the run says so in these lines, which it did not "
        "print then: Execution failed for task ':a:compileKotlin'.. Change what you "
        "changed, or take it back, so that the run gets as far as it did when you started"]
    # The same run with nothing more left out is every test failing still where the
    # run made when the stage started named no test either
    got_list, _, _ = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_x.py", "def test_f_case_1():\n    return\n"),
        base_list=[{**start_dict, "failed": [], "left_out": [_A_LINE]},
                   {"status": testcreate.BASE_RED, "failed": [], "left_out": [_A_LINE]}])
    assert got_list == []


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
def test_tests_that_cannot_be_run_call_no_model(repo, monkeypatch, stage):
    directory, base = repo
    monkeypatch.setattr(integrate, "analyse", lambda _: _START_DICT)
    monkeypatch.setattr(edit, "has_test_command", lambda: False)
    monkeypatch.setattr(refactor, "RLM", _StageThatRuns)
    _StageThatRuns.built_list.clear()
    out_dict = refactor.run_refactor(stage, base, "", "p", "out")
    assert out_dict["skipped"] == testcreate.SKIP_NO_TEST_COMMAND
    assert out_dict["settled"] is True and _StageThatRuns.built_list == []


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
def test_a_test_command_with_no_test_directory_is_told_as_such(repo, monkeypatch, stage):
    directory, base = repo
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", ["pytest"])
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "no_such_tests")
    target = "x.py" if stage == _CHANGE else "tests/test_x.py"
    assert refactor.baseline_or_skip(stage, [target]) == (
        frozenset(), [], "the repository holds no test directory: no_such_tests")


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
@pytest.mark.parametrize("exit_code, output", [
    (5, "no tests ran in 0.01s"),
    (5, "\n" + "-" * 70 + "\nRan 0 tests in 0.000s\n\nNO TESTS RAN"),
    (0, "\n" + "-" * 70 + "\nRan 0 tests in 0.000s\n\nOK")])
def test_a_command_that_found_no_test_calls_no_model_whichever_runner_says_so(
        repo, monkeypatch, stage, exit_code, output):
    """pytest's line for a run over no tests and unittest's are read the same way."""
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", f"import sys; print({output!r}); sys.exit({exit_code})"])
    edit.reset_patch_round()
    target = "x.py" if stage == _CHANGE else "tests/test_x.py"
    assert refactor.baseline_or_skip(stage, [target]) == (
        frozenset(), [], testcreate.SKIP_NO_BASELINE)
    assert edit.no_test_run(edit.run_test()) is True


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
@pytest.mark.parametrize("output", [
    "..ss\n" + "-" * 70 + "\nRan 2 tests in 0.000s\n\nOK (skipped=2)",
    "test_a (t.T.test_a) ... ok\ntest_b (t.T.test_b) ... ok\n"
    "setUpClass (t.Db) ... skipped 'no db'\nsetUpModule (n) ... skipped 'no net'\n\n"
    + "-" * 70 + "\nRan 2 tests in 0.000s\n\nOK (skipped=2)"])
def test_tests_that_ran_beside_classes_skipped_in_their_fixture_are_a_baseline(
        repo, monkeypatch, stage, output):
    """unittest counts a class or module skipped in setUpClass or setUpModule among its
    skips and not among the tests it ran."""
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", f"import sys; sys.stderr.write({output!r})"])
    edit.reset_patch_round()
    target = "x.py" if stage == _CHANGE else "tests/test_x.py"
    baseline_set, target_list, why = refactor.baseline_or_skip(stage, [target])
    assert why == "" and edit._ran_dict(baseline_set) == {"": (2, None, None, False)}
    assert target_list == [target]


_UNRUN_OUTPUT = ("=== short test summary info ===\n"
                 "ERROR tests/sub - ModuleNotFoundError: No module named 'x'\n"
                 "ERROR tests/test_c.py - ImportError: no module\n"
                 "PASSED tests/test_p.py::test_p\n1 passed, 2 errors in 0.04s")


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
@pytest.mark.parametrize("target_list, kept_list", [
    (["tests/sub/test_s.py", "tests/test_p.py"], ["tests/test_p.py"]),
    (["tests/sub/test_s.py", "tests/test_c.py"], []),
    (["tests/subway/test_w.py"], ["tests/subway/test_w.py"])])
def test_the_tests_refactored_leave_out_the_files_whose_tests_did_not_run_at_the_start(
        repo, monkeypatch, stage, target_list, kept_list):
    """A test file the run named alone, or one under a directory it named alone, ran no
    test when the stage that refactors the tests started: the stage leaves it out, and
    calls no model where that leaves nothing. The stage that refactors the change keeps
    its files."""
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", f"import sys; print({_UNRUN_OUTPUT!r}); sys.exit(1)"])
    edit.reset_patch_round()
    baseline_set, start_list, why = refactor.baseline_or_skip(stage, target_list)
    if stage == _CHANGE:
        assert (start_list, why) == (target_list, "")
    elif kept_list:
        assert (start_list, why) == (kept_list, "") and "tests/sub" in baseline_set
    else:
        assert (baseline_set, start_list) == (frozenset(), [])
        assert why == refactor._SKIP_UNRUN.format(paths=", ".join(target_list))
    # A file of which a command's run ran a test, or the file itself as a part, is no
    # such file
    pytest_run = _ran_run(name_list=["tests/sub/test_s.py::test_s"])
    pytest_run.load_fail_path_list = ["tests/sub", "test/a.test.js"]
    whole_run = edit.RunResult({"status": "failed", "failed": [], "failed_excerpt": "",
                                "targets": [], "output": ""})
    whole_run.part_dict = {"pytest": pytest_run,
                           "jest": _ran_run(unit_dict={"test/a.test.js": None})}
    assert edit.unrun_path_list(["test/a.test.js", "tests/sub/test_s.py", "tests/sub/t.py"],
                                whole_run) == ["tests/sub/t.py"]
    assert edit.unrun_path_list(["tests/test_p.py"], {"status": "failed",
                                                      "failed": ["tests"]}) == []


def _ran_run(name_list=None, unit_dict=None):
    """Return a RunResult of a whole run that ran two tests, named or by part."""
    result = edit.RunResult({"status": "failed", "failed": [], "failed_excerpt": "",
                             "targets": [], "output": ""})
    result.ran_count, result.ran_name_list, result.ran_unit_dict = 2, name_list, unit_dict
    return result


_UNITTEST_UNRUN_OUTPUT = (
    "E.\n" + "=" * 70 + "\nERROR: tests.test_sub (unittest.loader._FailedTest.tests.test_sub)\n"
    + "-" * 70 + "\nImportError: Failed to import test module: tests.test_sub\n"
    "ModuleNotFoundError: No module named 'no_such_mod'\n\n\n" + "-" * 70
    + "\nRan 2 tests in 0.001s\n\nFAILED (errors=1)\n")


@pytest.mark.parametrize("target_list, kept_list", [
    (["tests/test_sub.py", "tests/test_more.py"], ["tests/test_more.py"]),
    (["tests/test_sub.py"], [])])
def test_the_tests_refactored_leave_out_a_module_unittest_could_not_import(
        repo, monkeypatch, target_list, kept_list):
    """unittest names a test module it could not import by its dotted name: the stage
    that refactors the tests leaves its file out."""
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c",
        f"import sys; sys.stderr.write({_UNITTEST_UNRUN_OUTPUT!r}); sys.exit(1)"])
    edit.reset_patch_round()
    _, start_list, why = refactor.baseline_or_skip(_TEST, target_list)
    if kept_list:
        assert (start_list, why) == (kept_list, "")
    else:
        assert (start_list, why) == ([], refactor._SKIP_UNRUN.format(paths=target_list[0]))


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
def test_tests_that_name_nothing_before_the_stage_call_no_model(repo, monkeypatch, stage):
    directory, base = repo
    out_dict = _run_of(stage, directory, base, monkeypatch, [_test_result("failed")])
    assert out_dict["skipped"] == testcreate.SKIP_NO_BASELINE
    assert _StageThatRuns.built_list == []


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
def test_tests_that_ended_before_every_test_was_run_call_no_model(repo, monkeypatch, stage):
    """A test file that cannot be collected: the run names it and runs no test."""
    directory, base = repo
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    early_dict = dict(_test_result("failed", ["tests/test_plot.py"]), output=(
        "ERROR tests/test_plot.py\n!!!!!!!! Interrupted: 1 error during collection "
        "!!!!!!!!\n1 error in 0.07s"))
    out_dict = _run_of(stage, directory, base, monkeypatch, [early_dict])
    assert out_dict["skipped"] == testcreate.SKIP_NO_BASELINE
    assert out_dict["settled"] is True and _StageThatRuns.built_list == []


# == One run =================================================


@pytest.mark.parametrize("stage, name, target_list", [
    (_CHANGE, "refactor", [{"file": "x.py", "names": ["f"]}, {"file": "y.py", "names": ["h"]}]),
    (_TEST, "test_refactor", [{"file": "tests/test_x.py", "names": []},
                              {"file": "tests/test_y.py", "names": []}])])
def test_a_run_that_changes_nothing_is_settled_and_leaves_nothing(repo, monkeypatch, stage,
                                                                 name, target_list):
    directory, base = repo
    monkeypatch.setattr(edit, "run_test_as_is", lambda path_list=None: _test_result())
    out_dict = _run_of(stage, directory, base, monkeypatch, [_test_result()])
    assert (out_dict["called"], out_dict["settled"], out_dict["left"]) == (True, True, [])
    assert out_dict["tests_changed"] == out_dict["non_test_changed"] == []
    assert _git(directory, "status", "--porcelain") == ""
    built_dict = _StageThatRuns.built_list[0]
    assert built_dict["stage"] == name and built_dict["work"]["targets"] == target_list
    assert "change f" in built_dict["request"]
    assert {"revert_file", "record_decision", "replace_in_file"} <= set(built_dict["tools"])


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
def test_a_run_that_stopped_on_a_call_is_left_open(repo, monkeypatch, stage):
    """The check finds nothing in a tree nobody changed, and the run is not settled."""
    directory, base = repo
    monkeypatch.setattr(_StageThatRuns, "run", lambda self, **kwargs: {
        "answer": None, "history": [], "iters": 0, "refused_completes": 0,
        "error": "RuntimeError: overloaded", "outcome": None})
    monkeypatch.setattr(edit, "run_test_as_is", lambda path_list=None: _test_result())
    out_dict = _run_of(stage, directory, base, monkeypatch, [_test_result()])
    assert (out_dict["called"], out_dict["settled"]) == (True, False)
    assert out_dict["left"] == [
        refactor._CALL_ERROR_REASON.format(why="RuntimeError: overloaded")]
    assert integrate.end_of(out_dict) == integrate.END_ERROR


def test_the_stage_on_the_change_is_left_open_by_a_test_it_wrote_to(repo, monkeypatch):
    """The tool refuses the write, so this is a write made without one."""
    directory, base = repo

    def act(tool_dict):
        assert tool_dict["replace_in_file"]("tests/test_x.py", "True", "1")["ok"] is False
        _write(directory, "tests/test_x.py", "def test_f_case_1():\n    assert 1\n")
        assert tool_dict["replace_in_file"]("x.py", "return 10", "return 5 + 5")["ok"] is True

    out_dict = _run_of(_CHANGE, directory, base, monkeypatch, [_test_result()], act)
    assert out_dict["settled"] is False
    assert out_dict["left"] == [testcreate.NOT_OWN_REASON.format(paths="tests/test_x.py")]
    assert out_dict["refused"] == ["tests/test_x.py"]
    assert out_dict["tests_changed"] == ["tests/test_x.py"]
    assert out_dict["non_test_changed"] == ["x.py"]
    assert out_dict["untooled"] == ["tests/test_x.py"]


def _run_with_a_report(stage, directory, base, monkeypatch, is_tracked, act):
    """Run run_refactor() with the analysis stubbed and a test command that passes and
    writes the time to report.txt on every run."""
    if is_tracked:
        _write(directory, "report.txt", "0\n")
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", "report")
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "import time; open('report.txt', 'w').write(str("
        "time.time_ns())); print('1 passed in 0.01s')"])
    monkeypatch.setattr(config, "REFACTOR_MAX_REFUSAL", 2)
    monkeypatch.setattr(integrate, "analyse", lambda _: _START_DICT)
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(refactor, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(refactor, "RLM", _StageThatRuns)
    _StageThatRuns.built_list.clear()
    monkeypatch.setattr(_StageThatRuns, "act", staticmethod(act))
    return refactor.run_refactor(stage, base, "", "p", "out", request="change f")


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
@pytest.mark.parametrize("is_tracked", [True, False])
def test_a_file_the_test_command_writes_does_not_leave_the_stage_open(
        repo, monkeypatch, stage, is_tracked):
    """The run before the stage, the stage's own run_test() and the check's runs all
    write it, and none of that is the stage writing outside its files."""
    directory, base = repo
    out_dict = _run_with_a_report(stage, directory, base, monkeypatch, is_tracked,
                                  lambda tool_dict: tool_dict["run_test"]())
    assert (out_dict["settled"], out_dict["left"]) == (True, [])
    assert out_dict["refused_completes"] == 0 and out_dict["refused"] == []


# A test command that passes, and leaves a named pipe at a test file whose last line is
# the line the check of the whole run adds (edit.in_whole_run)
_PIPE_ON_PROBE_ARGV = [sys.executable, "-c", (
    "import os, pathlib, sys\n"
    "for p in pathlib.Path('tests').rglob('*.py'):\n"
    "    if p.is_file() and p.read_text().splitlines()[-1:] == [')']:\n"
    "        os.remove(p); os.mkfifo(p); sys.exit(1)\n"
    "print('1 passed in 0.01s')\n")]

# A test command that passes, and leaves a named pipe at tests/test_x.py on every run
_PIPE_ALWAYS_ARGV = [sys.executable, "-c", (
    "import os\n"
    "if os.path.isfile('tests/test_x.py'):\n"
    "    os.remove('tests/test_x.py'); os.mkfifo('tests/test_x.py')\n"
    "print('1 passed in 0.01s')\n")]


@pytest.mark.parametrize("stage, argv", [(_TEST, _PIPE_ON_PROBE_ARGV),
                                         (_CHANGE, _PIPE_ALWAYS_ARGV)])
def test_a_test_file_the_check_cannot_put_back_leaves_the_stage_open_with_a_reason(
        repo, monkeypatch, stage, argv):
    """The stage ends with what is left, not with an error, and its other files are as
    it left them. The stage on the change checks no test file with a line added, and a
    test command that leaves a named pipe on every run does not stop it."""
    directory, base = repo
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", argv)
    monkeypatch.setattr(config, "REFACTOR_MAX_REFUSAL", 2)
    monkeypatch.setattr(integrate, "analyse", lambda _: _START_DICT)
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(refactor, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(refactor, "RLM", _StageThatRuns)
    _StageThatRuns.built_list.clear()
    reason_list = []
    check = testcreate.ref_and_test_reason_list

    def keep_reason(*args, **kwargs):
        line_list = check(*args, **kwargs)
        reason_list.extend(line_list)
        return line_list

    monkeypatch.setattr(testcreate, "ref_and_test_reason_list", keep_reason)
    path, old, new = ("tests/test_x.py", "assert True", "assert 1") if stage == _TEST \
        else ("x.py", "return 2", "return 1 + 1")
    monkeypatch.setattr(_StageThatRuns, "act", staticmethod(
        lambda tool_dict: tool_dict["replace_in_file"](path, old, new)))
    out_dict = refactor.run_refactor(stage, base, "", "p", "out", request="change f")
    assert stat.S_ISFIFO(os.lstat(os.path.join(directory, "tests/test_x.py")).st_mode)
    if stage == _TEST:
        # The check after the first names the pipe as a test file that is gone
        assert out_dict["settled"] is False
        assert "are gone: tests/test_x.py" in out_dict["left"][0]
        assert any(line.startswith("- tests/test_x.py no longer holds what you wrote")
                   and line.endswith("Put it back with revert_file, and the same for any "
                                     "file that run deleted or put in its place "
                                     "(in git_diff_stat, one you did not change that "
                                     "has deletions only or is untracked), then write "
                                     "it again")
                   for line in reason_list)
        assert _read(directory, "x.py") == "def f():\n    return 10\n\n\ndef g():\n" \
                                           "    return 2\n\n\n"
    else:
        assert _read(directory, "x.py") == "def f():\n    return 10\n\n\ndef g():\n" \
                                           "    return 1 + 1\n\n\n"
    assert _read(directory, "tests/test_y.py") == "def test_h():  # case_2\n    assert True\n"


class _StageThatPutsBack(_StageThatRuns):
    """_StageThatRuns, which does what a refusal that says a file no longer holds what
    was written asks with its own tools (revert_file, then act again) before calling
    COMPLETE() again."""

    def run(self, **kwargs):
        type(self).act(self.tool_dict)
        refuse_count = 0
        for _ in range(3):
            refusal = kwargs["on_complete"]("tidied the tests")
            if not refusal:
                break
            refuse_count += 1
            assert "no longer holds what you wrote" in str(refusal)
            assert self.tool_dict["revert_file"]("tests/test_x.py")["ok"]
            type(self).act(self.tool_dict)
        return {"answer": "tidied the tests", "history": [], "iters": 1,
                "refused_completes": refuse_count, "error": "", "outcome": "complete"}


@pytest.mark.parametrize("leave", ["named pipe", "directory", "empty directory"])
def test_a_test_file_the_check_cannot_put_back_is_put_back_by_the_stage_s_revert_file(
        repo, monkeypatch, tmp_path, leave):
    """The test command leaves something other than a file, once, where the test file
    the stage changed was: the stage's revert_file puts it back, and the stage settles
    once it changes it again."""
    directory, base = repo
    make = {"named pipe": "os.remove(p); os.mkfifo(p)",
            "directory": "os.remove(p); os.mkdir(p); open(p + '/keep', 'w').close()",
            "empty directory": "os.remove(p); os.mkdir(p)"}[leave]
    mark = tmp_path / "left"
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", [sys.executable, "-c", (
        "import os, pathlib, sys\n"
        f"mark = {str(mark)!r}\n"
        "for p in map(str, pathlib.Path('tests').rglob('*.py')):\n"
        "    if os.path.isfile(p) and open(p).read().splitlines()[-1:] == [')']:\n"
        "        if not os.path.exists(mark):\n"
        f"            {make}; open(mark, 'w').close()\n"
        "        sys.exit(1)\n"
        "print('1 passed in 0.01s')\n")])
    monkeypatch.setattr(config, "REFACTOR_MAX_REFUSAL", 2)
    monkeypatch.setattr(integrate, "analyse", lambda _: _START_DICT)
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(refactor, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(refactor, "RLM", _StageThatPutsBack)
    _StageThatRuns.built_list.clear()
    monkeypatch.setattr(_StageThatPutsBack, "act", staticmethod(
        lambda tool_dict: tool_dict["replace_in_file"]("tests/test_x.py", "assert True",
                                                       "assert 1")))
    out_dict = refactor.run_refactor(_TEST, base, "", "p", "out", request="change f")
    assert mark.exists()
    assert out_dict["settled"] is True, out_dict["left"]
    assert "assert 1" in _read(directory, "tests/test_x.py")


def test_a_file_the_stage_wrote_itself_is_its_own_whatever_the_tests_write_there(
        repo, monkeypatch):
    directory, base = repo
    out_dict = _run_with_a_report(
        _CHANGE, directory, base, monkeypatch, True,
        lambda tool_dict: _write(directory, "report.txt", "mine\n"))
    assert out_dict["left"] == [testcreate.NOT_OWN_REASON.format(paths="report.txt")]


def test_the_revert_the_refusal_names_puts_back_a_file_outside_the_stages_own(
        repo, monkeypatch):
    directory, base = repo

    def act(tool_dict):
        _write(directory, "report.txt", "mine\n")
        assert tool_dict["revert_file"]("report.txt")["ok"] is True

    out_dict = _run_with_a_report(_CHANGE, directory, base, monkeypatch, True, act)
    assert (out_dict["settled"], out_dict["refused"]) == (True, [])


def test_a_refactored_change_whose_tests_say_what_they_said_is_settled(repo, monkeypatch):
    directory, base = repo
    out_dict = _run_of(
        _CHANGE, directory, base, monkeypatch, [_test_result("failed", ["t::old"])],
        lambda tool_dict: tool_dict["replace_in_file"]("x.py", "return 10", "return 5 + 5"))
    assert out_dict["settled"] is True and out_dict["non_test_changed"] == ["x.py"]
    assert out_dict["work"]["already_failing"] == ["t::old"]


# == What the machine checks =================================


def _check_of(stage, directory, base, monkeypatch, test_list, change, broken=None,
              base_list=None):
    """Run reason_list() over what change() leaves, with the analysis stubbed, run_test()
    answering from a list and the run over the base's tree answering from another."""
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(integrate, "analyse", lambda _: dict(
        _START_DICT, no_resolve=broken or {}))
    call_list, base_call_list = [], []

    def run_test(path_list=None):
        call_list.append(path_list)
        return test_list.pop(0)

    def base_run_dict(project_dir, commit, path_list, baseline_set):
        base_call_list.append(path_list)
        return {"is_part": False, "left_out": [], "is_no_run": False,
                **base_list.pop(0)}

    monkeypatch.setattr(edit, "run_test", run_test)
    monkeypatch.setattr(testcreate, "base_run_dict", base_run_dict)
    base_list = list(base_list or [{"status": testcreate.BASE_RED, "failed": ["t::a"]}])
    edit.reset_patch_round(round_limit=0)
    target_list = sorted(refactor.target_dict_of(stage, directory, base, _RANGE_DICT, "p"))
    start_file_list = _git(directory, "ls-files").split()
    record_dict = refactor.start_record_dict(stage, directory, base, target_list,
                                             frozenset())
    change()
    line_list = refactor.reason_list(
        stage, directory, "out", base, _START_DICT, frozenset({"t::old"}), None,
        target_list, start_file_list,
        refactor.may_write_of(stage, target_list, start_file_list), record_dict)
    return line_list, call_list, base_call_list


@pytest.mark.parametrize("stage", [integrate.REFACTOR_STAGE, integrate.TEST_REFACTOR_STAGE])
def test_a_name_the_stage_left_undefined_is_a_reason(stage, repo, monkeypatch):
    directory, base = repo
    old = "x.py: F821 Undefined name `a`"
    new = "x.py: F821 Undefined name `helper`"
    answer_iter = iter([collections.Counter([old]), collections.Counter([old, new])])
    monkeypatch.setattr(edit, "run_check", lambda: next(answer_iter))
    line_list, _, _ = _check_of(
        stage, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: None)
    assert len(line_list) == 1 and new in line_list[0] and old not in line_list[0]


def test_a_refactored_change_is_checked_by_the_whole_suite_alone(repo, monkeypatch):
    directory, base = repo
    line_list, call_list, base_call_list = _check_of(
        _CHANGE, directory, base, monkeypatch, [_test_result()],
        lambda: _write(directory, "x.py", "def f():\n    return 5 + 5\n"))
    assert line_list == [] and call_list == [None] and base_call_list == []


@pytest.mark.parametrize("stage, test_list", [
    (_CHANGE, [_test_result()]), (_TEST, [_test_result(), _test_result()])])
def test_a_file_the_test_command_left_under_the_tests_is_not_a_file_written_to(
        repo, monkeypatch, stage, test_list):
    directory, base = repo
    line_list, _, _ = _check_of(
        stage, directory, base, monkeypatch, test_list,
        lambda: _write(directory, "tests/cache/last_run", "1\n"),
        base_list=[{"status": testcreate.BASE_RED, "failed": []}] * 2)
    assert line_list == []


@pytest.mark.parametrize("stage, path", [(_CHANGE, "tests/test_x.py"), (_CHANGE, "z.py"),
                                         (_TEST, "x.py"), (_TEST, "tests/test_old.py")])
def test_a_file_that_is_not_the_stage_s_to_write_and_was_written_to_is_a_reason(
        repo, monkeypatch, stage, path):
    """Whichever side of the tests the file is on, for both stages. The tools refuse the
    write, so this is a write made without one."""
    directory, base = repo
    _write(directory, "z.py", "z = 1\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "z")
    line_list, _, _ = _check_of(
        stage, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, path, "changed = True\n"))
    assert line_list == [testcreate.NOT_OWN_REASON.format(paths=path)]


def test_a_call_that_no_longer_resolves_and_a_test_that_fails_now_are_reasons(repo,
                                                                            monkeypatch):
    directory, base = repo
    line_list, _, _ = _check_of(
        _CHANGE, directory, base, monkeypatch,
        [_test_result("failed", ["t::old", "t::new"], "boom")],
        lambda: _write(directory, "x.py", "def f():\n    return 5 + 5\n"),
        broken={(_Y, _X, "f"): [1]})
    assert len(line_list) == 2
    assert "no longer do" in line_list[0] and "f in y.py" in line_list[0]
    assert "t::new" in line_list[1] and "t::old" not in line_list[1]


def test_refactored_tests_are_run_alone_then_whole_then_over_the_base(repo, monkeypatch):
    """The run over the base's tree takes the files the stage started with and the ones
    it added, and says what it said before the stage started."""
    directory, base = repo

    def change():
        _write(directory, "tests/test_x.py", "from helper import one\n\n\n"
               "def test_f_case_1():\n    assert one\n")
        _write(directory, "tests/helper.py", "one = True\n")

    same_dict = {"status": testcreate.BASE_RED, "failed": ["t::a"]}
    line_list, call_list, base_call_list = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()], change,
        base_list=[same_dict, dict(same_dict, failed=["t::a", "t::b"])])
    assert line_list == []
    assert call_list == [["tests/helper.py", "tests/test_x.py"], None]
    assert base_call_list == [["tests/test_x.py", "tests/test_y.py"],
                              ["tests/helper.py", "tests/test_x.py", "tests/test_y.py"]]


def test_tests_nobody_changed_are_not_run_over_the_base_again(repo, monkeypatch):
    directory, base = repo
    line_list, call_list, base_call_list = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result()], lambda: None)
    assert line_list == [] and call_list == [None] and len(base_call_list) == 1


@pytest.mark.parametrize("after_dict, reason", [
    ({"status": testcreate.BASE_GREEN, "failed": []},
     refactor._BASE_STATUS_REASON.format(before="red", after="green")),
    ({"status": testcreate.BASE_RED, "failed": ["t::b"]},
     refactor._BASE_PASS_REASON.format(tests="t::a"))])
def test_tests_that_stop_failing_over_the_base_are_a_reason(repo, monkeypatch, after_dict,
                                                            reason):
    directory, base = repo
    line_list, _, _ = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_x.py", "def test_f_case_1():\n    pass\n"),
        base_list=[{"status": testcreate.BASE_RED, "failed": ["t::a"]}, after_dict])
    assert line_list == [reason]


@pytest.mark.parametrize("before_list, after_list, pass_text", [
    # The import of what the change added moved to the top of the file, and back
    (["tests/test_x.py::test_f"], ["tests/test_x.py"], ""),
    (["tests/test_x.py"], ["tests/test_x.py::test_f"], ""),
    (["tests/test_x.py::test_f", "tests/test_x.py::test_g"], ["tests/test_x.py"], ""),
    # Another test of the same file failing is not the test failing
    (["tests/test_x.py::test_f", "tests/test_x.py::test_g"], ["tests/test_x.py::test_g"],
     "tests/test_x.py::test_f"),
    # Another file named alone is not this test's file
    (["tests/test_x.py::test_f"], ["tests/test_y.py"], "tests/test_x.py::test_f"),
    (["tests/test_x.py"], ["tests/test_y.py::test_h"], "tests/test_x.py"),
    (["tests/test_x.py"], ["tests/test_x.py2::t"], "tests/test_x.py"),
    # A name that starts with no test file is told by itself alone
    (["Suite > adds"], ["tests/test_x.py"], "Suite > adds")])
def test_a_test_file_named_alone_and_a_test_of_it_are_one_failure_over_the_base(
        repo, monkeypatch, before_list, after_list, pass_text):
    directory, base = repo
    line_list, _, _ = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_x.py", "def test_f_case_1():\n    pass\n"),
        base_list=[{"status": testcreate.BASE_RED, "failed": before_list},
                   {"status": testcreate.BASE_RED, "failed": after_list}])
    assert line_list == ([refactor._BASE_PASS_REASON.format(tests=pass_text)]
                         if pass_text else [])


@pytest.mark.parametrize("after_list, line_list", [
    # Two of the three subtests of one test no longer fail
    (["t::a", "t::b"], [refactor._BASE_FEWER_REASON.format(tests="t::a (3 times then, 1 now)")]),
    (["t::a", "t::a", "t::a", "t::b"], []),
    (["t::a", "t::a", "t::a", "t::a", "t::b"], []),
    (["t::a"], [refactor._BASE_PASS_REASON.format(tests="t::b"),
                refactor._BASE_FEWER_REASON.format(tests="t::a (3 times then, 1 now)")])])
def test_a_test_that_fails_fewer_times_over_the_base_is_a_reason(repo, monkeypatch,
                                                                after_list, line_list):
    """The names of a test's subtests are the test's own, so a weakened subtest leaves
    the name and takes one of its times away."""
    directory, base = repo
    result_list, _, _ = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_x.py", "def test_f_case_1():\n    pass\n"),
        base_list=[{"status": testcreate.BASE_RED, "failed": ["t::a", "t::a", "t::a", "t::b"]},
                   {"status": testcreate.BASE_RED, "failed": after_list}])
    assert result_list == line_list


def test_tests_red_over_the_base_without_a_name_are_tests_that_all_fail(repo, monkeypatch):
    """A run that names no test failed before one test ran, so every test of the run
    fails there: one now leaves every test failing still where the run made as the
    stage started named none either, and one made as the stage started leaves a test
    the run now does not name one that fails there no more. A test the run made as the
    stage started named, and the run now names none, fails there no more."""
    directory, base = repo
    line_list, _, _ = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_x.py", "def test_f_case_1():\n    pass\n"),
        base_list=[{"status": testcreate.BASE_RED, "failed": []},
                   {"status": testcreate.BASE_RED, "failed": []}])
    assert line_list == []
    line_list, _, _ = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_x.py", "def test_f_case_1():\n    return None\n"),
        base_list=[{"status": testcreate.BASE_RED, "failed": ["t::a"]},
                   {"status": testcreate.BASE_RED, "failed": []}])
    assert line_list == [refactor._BASE_PASS_REASON.format(tests="t::a")]
    line_list, _, _ = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_x.py", "def test_f_case_1():\n    return\n"),
        base_list=[{"status": testcreate.BASE_RED, "failed": []},
                   {"status": testcreate.BASE_RED, "failed": ["t::a"]}])
    assert line_list == [refactor._BASE_NAMED_REASON.format(tests="t::a")]


def test_tests_run_over_the_base_by_no_run_at_the_start_are_compared_with_nothing(
        repo, monkeypatch):
    """The stage started with no test file the test command runs, and added one."""
    directory, base = repo
    line_list, _, base_call_list = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_new.py", "def test_n():\n    pass\n"),
        base_list=[{"status": "", "failed": []}])
    assert line_list == []
    # The one run over the base's tree is the one made when the stage started
    assert len(base_call_list) == 1


def test_a_case_id_that_moves_to_another_test_file_is_kept(
        repo, monkeypatch):
    directory, base = repo

    def move():
        _write(directory, "tests/test_x.py", "def test_f():\n    assert True\n")
        _write(directory, "tests/test_y.py", "def test_h():  # case_2\n    assert True\n\n\n"
               "def test_f_case_1():\n    assert True\n")

    line_list, _, _ = _check_of(_TEST, directory, base, monkeypatch,
                                [_test_result(), _test_result()], move,
                                base_list=[{"status": "red", "failed": []}] * 2)
    assert line_list == []


def test_a_case_id_that_is_in_no_test_file_any_more_is_a_reason(repo, monkeypatch):
    """The tests are not run over the base's tree while something else is left."""
    directory, base = repo
    line_list, _, base_call_list = _check_of(
        _TEST, directory, base, monkeypatch, [_test_result(), _test_result()],
        lambda: _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n"),
        base_list=[{"status": "red", "failed": []}])
    assert line_list == [refactor._CASE_REASON.format(ids="case_2")]
    assert len(base_call_list) == 1


def test_a_test_file_that_is_gone_and_a_file_outside_the_tests_are_reasons(repo,
                                                                          monkeypatch):
    directory, base = repo

    def change():
        os.remove(os.path.join(directory, "tests/test_y.py"))
        _write(directory, "x.py", "def f():\n    return 11\n")

    line_list, _, _ = _check_of(_TEST, directory, base, monkeypatch, [_test_result()],
                                change)
    assert line_list == [testcreate.NOT_OWN_REASON.format(paths="x.py"),
                         refactor._DELETE_REASON.format(paths="tests/test_y.py"),
                         refactor._CASE_REASON.format(ids="case_2")]


# == The instructions and the state ==========================


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_the_instructions_are_filled_in_and_name_only_tools_the_stage_is_handed(
        ext, stage, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    text = refactor.build_refactor_instruction(ext, stage)
    assert not re.search(r"\{[a-z_]+\}", text)
    hand_set = {tool.__name__ for tool in refactor.stage_tool_list(stage)}
    for name in set(agent.TOOL_RESULT_KEY_DICT) - hand_set:
        assert name not in text


def test_the_instructions_name_the_start_block_for_the_tests_that_were_failing():
    text = " ".join(refactor.build_refactor_instruction("py", _TEST).split())
    assert 'and the tests "## At the start" lists as already failing.' in text
    for stage in refactor.STAGE_TUPLE:
        assert "the state lists" not in " ".join(
            refactor.build_refactor_instruction("py", stage).split())


def test_the_two_stages_open_with_words_the_other_does_not_hold():
    change_text = refactor.build_refactor_instruction("py", _CHANGE)
    test_text = refactor.build_refactor_instruction("py", _TEST)
    assert change_text.startswith("You are refactoring a change that was just")
    assert test_text.startswith("You are refactoring the tests of a change that was just")
    assert "You are refactoring the tests of" not in change_text
    assert "You are refactoring a change" not in test_text


@pytest.mark.parametrize("stage, own_text", [
    (_CHANGE, "Shared helpers go in util."), (_TEST, "Shared set-up goes in a fixture.")])
def test_each_stage_ends_with_the_one_file_written_for_it(repo, monkeypatch, stage,
                                                          own_text):
    directory, _ = repo
    bare = refactor.build_refactor_instruction("py", stage)
    assert "<repository_instructions>" not in bare
    text_dict = {"EDIT_GUIDE": "Log with the logger.", "TEST_GUIDE": "Fixtures go in conftest.",
                 "REFACTOR_GUIDE": "Shared helpers go in util.",
                 "TEST_REFACTOR_GUIDE": "Shared set-up goes in a fixture."}
    for setting, file_text in text_dict.items():
        monkeypatch.setattr(config, setting, f"{setting}.md")
        _write(directory, f"{setting}.md", file_text + "\n")
    guide.reset()
    text = refactor.build_refactor_instruction("py", stage)
    assert text.startswith(bare) and text.count("<repository_instructions>") == 1
    assert [file_text for file_text in text_dict.values() if file_text in text] == [own_text]


@pytest.mark.parametrize("stage", refactor.STAGE_TUPLE)
def test_the_test_directory_is_named_only_where_the_setting_names_one(repo, monkeypatch,
                                                                     stage):
    assert "live under `tests`" in refactor.build_refactor_instruction("py", stage)
    for test_path in (".", "", "spec"):
        monkeypatch.setattr(config, "TEST_PATH", test_path)
        assert "live under" not in refactor.build_refactor_instruction("py", stage)


def test_the_state_holds_the_test_names_of_a_baseline_of_several_commands(repo):
    directory, base = repo
    baseline = frozenset({"t::old", edit.baseline_mark("red before", "py"),
                          edit.baseline_mark("not compared", "js")})
    work_dict = refactor.build_work_dict(
        directory, base, "change f", {"x.py": ["f"]}, baseline, [])
    assert work_dict["already_failing"] == ["t::old"]
    assert "\x1f" not in repr(work_dict)


def test_the_union_is_the_commit_a_clone_is_checked_out_at(repo, tmp_path):
    directory, base = repo
    union = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "branch", "union")
    _git(directory, "reset", "-q", "--hard", base)
    clone = str(tmp_path / "clone")
    _git(str(tmp_path), "clone", "-q", directory, clone)
    _git(clone, "checkout", "-q", "--detach", "origin/union")
    work_dict = refactor.build_work_dict(clone, base, "change f", {"x.py": ["f"]},
                                         frozenset(), [])
    assert work_dict["union"] == union != base
    assert f"held by {gitcmd.short(union)}" in refactor.start_text(_CHANGE, work_dict)


def test_the_state_leaves_out_a_file_that_holds_what_the_test_command_left_there(
        repo, monkeypatch):
    directory, base = repo
    _write(directory, "report.txt", "0\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "report")
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "import time; open('report.txt', 'w').write(str("
        "time.time_ns())); print('1 passed in 0.01s')"])
    edit.reset_patch_round(round_limit=0)
    integrate.reset_stage()
    work_dict = refactor.build_work_dict(directory, base, "change f", {"x.py": ["f"]},
                                         frozenset(), [])
    turn_note, _ = refactor.make_state(_CHANGE, directory, work_dict, lambda: [],
                                       frozenset(), _RANGE_DICT, "p")
    assert edit.run_test()["status"] == "passed"
    assert edit.own_change_set() == {"report.txt"}
    assert "Files you have changed, as of this turn: none." in turn_note(1, 5)
    assert work_dict["changed"] == []
    # What the stage writes there itself is listed
    _write(directory, "report.txt", "mine\n")
    assert "  report.txt  +1 -1" in turn_note(2, 5)


def test_the_state_reads_a_whole_run_with_no_primary_test_as_the_check_does(repo,
                                                                           monkeypatch):
    directory, base = repo
    no_test_dict = dict(_test_result("did not run"), output="no tests ran in 0.01s")
    run = edit.RunResult(_test_result())
    run.part_dict = {"py": edit.RunResult(no_test_dict), "js": edit.RunResult(_test_result())}
    run.part_dict["py"].is_no_test = True
    run.primary_name, run.is_whole = "py", True
    baseline = testcreate.baseline_of(run, is_no_test_green=True)
    edit.reset_patch_round(round_limit=0)
    integrate.reset_stage()
    work_dict = refactor.build_work_dict(directory, base, "change f", {"x.py": ["f"]},
                                         baseline, [])
    turn_note, _ = refactor.make_state(_CHANGE, directory, work_dict, lambda: [], baseline,
                                       _RANGE_DICT, "p")
    monkeypatch.setattr(edit, "last_whole_run",
                        lambda: {"n": 1, "result": run, "change": {}})
    assert "said nothing it did not say before you started" in turn_note(2, 5)
    assert work_dict["tests"] == ""
    monkeypatch.setattr(edit, "last_whole_run",
                        lambda: {"n": 2, "result": run, "change": {"x.py": "hash"}})
    turn_note(3, 5)
    assert work_dict["tests"].startswith("`py` did not run")


def test_the_state_names_the_targets_and_what_the_run_changed(repo, monkeypatch):
    directory, base = repo
    monkeypatch.setattr(integrate, "analyse", lambda _: _START_DICT)
    edit.reset_patch_round(round_limit=0)
    integrate.reset_stage()
    work_dict = refactor.build_work_dict(
        directory, base, "change f", {"x.py": ["f"], "y.py": ["h"]}, frozenset({"t::old"}),
        [{"asked": "change f", "tests_changed": ["tests/test_x.py"]}])
    turn_note, check_and_keep = refactor.make_state(
        _CHANGE, directory, work_dict, lambda: ["- something"], frozenset({"t::old"}),
        _RANGE_DICT, "p")
    _write(directory, "x.py", "def f():\n    return 5 + 5\n\n\ndef g():\n    return 2\n\n\n")
    text = turn_note(1, 5)
    assert text.startswith("## State (turn 1/5)\n\nFiles you have changed, as of this turn:\n"
                           "  x.py  +1 -1 (f)\n")
    assert testcreate._NO_CHECK_YET in text
    # What was measured before the first turn is in the block after the request
    start = refactor.start_text(_CHANGE, work_dict)
    assert start.startswith("## At the start\n\nThe change, written against ")
    assert "  x.py — f" in start and "  y.py — h" in start
    assert "asked: change f" in start and "tests it changed: tests/test_x.py" in start
    assert start.endswith("they are not yours to change:\n  t::old")
    assert "asked: change f" not in text and "t::old" not in text and "y.py — h" not in text
    assert check_and_keep() == ["- something"]
    assert "  - something" in turn_note(2, 5)
    # The call made after the last turn reads the block under the last turn's heading
    late = turn_note(6, 5)
    assert late.startswith("## State (turn 5/5)") and "turns left" not in late
    assert integrate.stage_turn() == 5
