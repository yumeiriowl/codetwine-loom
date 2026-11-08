"""Writing tests for the definitions a combined change touched."""

import collections
import json
import logging
import os
import stat
import subprocess
import sys

import pytest
from codetwine.utils import file_utils

from codetwine_loom import agent, config, integrate, rlm, testcreate, tmpfile
from codetwine_loom.languages import LANG_DICT, dep_file_example
from codetwine_loom.paths import repo_to_knowledge
from codetwine_loom.tools import edit, source

_X = "p/x_py/x.py"
_Y = "p/y_py/y.py"
_TEST_X = "p/tests/test_x_py/test_x.py"


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


def _test_result(status="passed", failed=(), excerpt="", output=""):
    """One run_test() return value."""
    return {"status": status, "failed": list(failed), "failed_excerpt": excerpt,
            "targets": [], "output": output}


# What pytest prints over a directory that holds no test
_NO_TEST_OUTPUT = "no tests ran in 0.01s"

# What pytest prints when a test file cannot be collected: no test is run
_EARLY_STOP_OUTPUT = ("ERROR tests/test_plot.py\n"
                      "!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection "
                      "!!!!!!!!!!!!!!!!!!!!\n1 error in 0.07s")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository whose HEAD changes f in x.py, adds y.py and changes a test.

    x.py holds f (lines 1-4) and g (lines 5-8) at both commits; HEAD rewrites line 2.

    Yields:
        (path, the commit HEAD was written against).
    """
    directory = str(tmp_path / "p")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    _write(directory, "x.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n\n\n")
    _write(directory, "tests/test_x.py", "def test_f():\n    pass\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _write(directory, "x.py", "def f():\n    return 10\n\n\ndef g():\n    return 2\n\n\n")
    _write(directory, "y.py", "def h():\n    return 3\n")
    _write(directory, "tests/test_x.py", "def test_f():\n    assert True\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "change\n\nCodetwine-Loom-Did: changed f and added h")
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    return directory, base


_RANGE_DICT = {_X: [(1, 4, "f"), (5, 8, "g")], _Y: [(1, 2, "h")],
               _TEST_X: [(1, 2, "test_f")]}


# == What the change touched =================================


def test_the_definitions_are_the_ones_head_holds_tests_left_out(repo):
    directory, base = repo
    assert testcreate.change_def_dict(directory, base, _RANGE_DICT, "p") == \
        {"x.py": ["f"], "y.py": ["h"]}


def test_a_change_outside_every_definition_names_no_file(repo):
    directory, base = repo
    assert testcreate.change_def_dict(directory, base, {}, "p") == {}


@pytest.mark.parametrize("primary, keep_list", [
    ("py", ["x.py"]), ("ts", ["web/app.js", "web/view.ts"])])
def test_only_the_files_the_test_command_can_test_get_a_test(monkeypatch, primary,
                                                              keep_list):
    monkeypatch.setattr(config, "PRIMARY_EXT", primary)
    def_dict = {"x.py": ["f"], "web/app.js": ["g"], "web/view.ts": ["v"],
                "data/q.sql": ["q"]}
    assert sorted(testcreate.testable_def_dict(def_dict)) == keep_list


def test_a_definition_a_test_refers_to_is_told_from_one_none_does():
    find_dict = testcreate.target_dict(
        {"x.py": ["f"], "y.py": ["h"]},
        [(_TEST_X, _X, "f", [3, 1]), (_Y, _X, "f", [2])], _RANGE_DICT, "p")
    assert find_dict == {
        "referenced": [{"file": "x.py", "name": "f",
                        "tests": [{"path": "tests/test_x.py", "lines": [1, 3]}]}],
        "unreferenced": [{"file": "y.py", "name": "h"}]}


def test_a_reference_to_the_enclosing_definition_counts_for_the_one_inside(monkeypatch):
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    range_dict = {_X: [(1, 10, "C"), (2, 4, "m")]}
    find_dict = testcreate.target_dict({"x.py": ["m"]}, [(_TEST_X, _X, "C", [5])],
                                       range_dict, "p")
    assert [entry["name"] for entry in find_dict["referenced"]] == ["m"]
    assert find_dict["unreferenced"] == []


def test_one_of_two_methods_of_one_name_is_a_target_of_its_own(tmp_path, monkeypatch):
    """B.run changed and the test uses A: B.run is a target no test refers to, shown
    with its own first line."""
    from conftest import write_knowledge_db
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    path = str(tmp_path / "k.sqlite")
    write_knowledge_db({"files": [
        {"file": "p/calc_py/calc.py",
         "file_dependencies": {"definitions": [
             {"name": "A", "type": "class_definition", "start_line": 1, "end_line": 3,
              "context": "class A:"},
             {"name": "run", "type": "function_definition", "start_line": 2,
              "end_line": 3, "context": "    def run(self, a):"},
             {"name": "B", "type": "class_definition", "start_line": 6, "end_line": 8,
              "context": "class B:"},
             {"name": "run", "type": "function_definition", "start_line": 7,
              "end_line": 8, "context": "    def run(self, b, c):"}],
             "callee_usages": [], "caller_usages": []},
         "doc": {"sections": [], "summary": ""}},
        {"file": "p/tests/test_a_py/test_a.py",
         "file_dependencies": {"definitions": [], "callee_usages": [
             {"from": "p/calc_py/calc.py", "name": "A", "lines": [5],
              "target_name": "A", "target_start_line": 1}],
             "caller_usages": []},
         "doc": {"sections": [], "summary": ""}}]}, path, "p")
    start_dict = integrate.analyse(path)
    monkeypatch.setattr(integrate, "change_dict_of", lambda *a: {
        "add": [], "delete": [], "line": {"calc.py": [(8, 8)]}})
    def_dict = testcreate.change_def_dict("", "base", start_dict["range"], "p")
    assert def_dict == {"calc.py": ["B.run"]}
    find_dict = testcreate.target_dict(def_dict, start_dict["ref"], start_dict["range"], "p")
    assert find_dict == {"referenced": [], "unreferenced": [{"file": "calc.py",
                                                             "name": "B.run"}]}
    work_dict = testcreate.case_work_dict([], "", [], find_dict, start_dict["signature"], "p")
    assert work_dict["targets"][0]["signature"] == "    def run(self, b, c):"


@pytest.mark.parametrize("file, name, expect", [
    ("calc.py", "B.run", ("calc.py", "B.run")), ("calc.py", "A.run", ("calc.py", "A.run")),
    ("calc.py", "calc.B.run", ("calc.py", "B.run")),
    # Its own name alone matches when one definition has it, and not when two do
    ("x.py", "f", ("x.py", "C.f")), ("calc.py", "run", None)])
def test_a_case_names_a_definition_inside_another_by_its_whole_name_or_its_own(
        file, name, expect):
    target_list = [{"file": "calc.py", "name": "A.run"}, {"file": "calc.py", "name": "B.run"},
                   {"file": "x.py", "name": "C.f"}]
    assert testcreate._target_key({"file": file, "name": name}, target_list) == expect


# == When no model is called =================================


def test_no_test_command_no_test_file_and_no_definition_each_skip(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(edit, "has_test_command", lambda: False)
    assert testcreate.skip_reason(directory, {"x.py": ["f"]}) == \
        testcreate.SKIP_NO_TEST_COMMAND
    monkeypatch.setattr(edit, "has_test_command", lambda: True)
    assert testcreate.skip_reason(directory, {}) == testcreate._SKIP_NO_DEFINITION
    assert testcreate.skip_reason(directory, {"x.py": ["f"]}) == ""
    # No test file: the stage still runs where it can write the language's dependency
    # file, and calls no model where the language has none
    monkeypatch.setattr(config, "TEST_PATH", "spec")
    assert testcreate.skip_reason(directory, {"x.py": ["f"]}) == ""
    monkeypatch.setattr(config, "PRIMARY_EXT", "sql")
    assert testcreate.skip_reason(directory, {"x.py": ["f"]}) == \
        testcreate._SKIP_NO_TEST_FILE
    # The tests' directory being there or not is not what decides
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    assert testcreate.skip_reason(directory, {"x.py": ["f"]}) == ""


def test_the_baseline_is_the_failing_names_or_nothing_to_compare_by():
    assert testcreate.baseline_of(_test_result()) == frozenset()
    assert testcreate.baseline_of(_test_result("failed", ["t::a"])) == frozenset({"t::a"})
    assert testcreate.baseline_of(_test_result("failed")) is None
    assert testcreate.baseline_of(_test_result("did not run")) is None


def _command_run(**part_by_command):
    """A run of several test commands as edit.run_test() returns it: {command: its own
    result}, the first being the primary language's."""
    status_list = [part["status"] for part in part_by_command.values()]
    status = "failed" if "failed" in status_list else \
        "did not run" if "did not run" in status_list else "passed"
    result = edit.RunResult(_test_result(
        status, [name for part in part_by_command.values() for name in part["failed"]]))
    result.part_dict = {}
    for command, part in part_by_command.items():
        result.part_dict[command] = edit.RunResult(part)
        result.part_dict[command].is_no_test = part["output"] == _NO_TEST_OUTPUT
        result.part_dict[command].is_early_stop = part["output"] == _EARLY_STOP_OUTPUT
    result.primary_name = "py" if "py" in part_by_command else ""
    result.is_no_test = all(part.is_no_test for part in result.part_dict.values())
    result.is_whole = True
    return result


def _baseline(name_list=(), **word_by_command):
    """A baseline of several test commands: the names, and what each command is in it
    ("not compared", "green before" or "red before")."""
    return frozenset(name_list) | {edit.baseline_mark(word, command)
                                   for command, word in word_by_command.items()}


_NOT_RUN = _test_result("did not run", excerpt="jest: not found")
_NO_TEST = _test_result("did not run", output=_NO_TEST_OUTPUT)


def test_a_command_that_does_not_run_leaves_the_stage_held_to_the_others():
    before = _command_run(py=_test_result(), js=_NOT_RUN)
    baseline = testcreate.baseline_of(before)
    assert baseline == _baseline(py="green before", js="not compared")
    assert edit.baseline_name_list(baseline) == []
    assert testcreate.run_reason(before, baseline) == ""
    # The primary command's tests are held as with one command
    assert testcreate.run_reason(
        _command_run(py=_test_result("failed", ["t::new"]), js=_NOT_RUN),
        baseline).startswith("`py` fails t::new")
    assert testcreate.run_reason(_command_run(py=_test_result("failed"), js=_NOT_RUN),
                                 baseline).startswith("`py` fails (")
    assert testcreate.run_reason(_command_run(py=_NOT_RUN, js=_NOT_RUN),
                                 baseline).startswith("`py` did not run")
    # The set goes to the test tasks as a list and comes back as it was
    assert testcreate.run_reason(before, frozenset(sorted(baseline))) == ""


def test_a_command_that_stops_running_is_a_reason():
    baseline = testcreate.baseline_of(_command_run(py=_test_result(), js=_test_result()))
    assert baseline == _baseline(py="green before", js="green before")
    # A command that is not part of a later run at all is one that did not run
    assert testcreate.run_reason(_command_run(py=_test_result()), baseline).startswith(
        "`js` did not run (the command was not part of this run")
    assert testcreate.run_reason(_command_run(js=_test_result()), baseline).startswith(
        "`py` did not run")
    # A run over some files alone is held by the commands it ran
    own_run = _command_run(py=_test_result())
    own_run.is_whole = False
    assert testcreate.run_reason(own_run, baseline) == ""
    # and so is every command where nothing could be run
    reason = testcreate.run_reason(_test_result("did not run", excerpt="tests is gone"),
                                   baseline)
    assert reason == "`js` did not run (tests is gone); `py` did not run (tests is gone)"
    assert testcreate.run_reason(_command_run(py=_test_result(), js=_NOT_RUN),
                                 baseline) == "`js` did not run (jest: not found)"
    # Another language's command that found no test of its own is a reason where its
    # tests ran when the stage started, and says nothing where they found none then
    assert testcreate.run_reason(_command_run(py=_test_result(), js=_NO_TEST),
                                 baseline) == "`js` did not run ()"
    zero_baseline = testcreate.baseline_of(_command_run(py=_test_result(), js=_NO_TEST))
    assert zero_baseline == _baseline(py="green before", js="no test before")
    assert testcreate.run_reason(_command_run(py=_test_result(), js=_NO_TEST),
                                 zero_baseline) == ""
    assert testcreate.run_reason(_command_run(py=_test_result(),
                                              js=_test_result("failed", ["j::new"])),
                                 zero_baseline).startswith("`js` fails j::new")
    # The primary command finding none is a reason, as with one command
    assert testcreate.run_reason(_command_run(py=_NO_TEST, js=_test_result()),
                                 baseline).startswith("`py` did not run")
    assert testcreate.run_reason(_NO_TEST, frozenset()).endswith("did not run ()")


def test_a_primary_command_that_does_not_run_is_no_baseline(repo, monkeypatch):
    """The stage writes the primary language's tests, so it is not started."""
    directory, _ = repo
    assert testcreate.baseline_of(_command_run(py=_NOT_RUN, js=_test_result())) is None
    assert testcreate.baseline_of(_command_run(py=_test_result("failed"),
                                          js=_test_result())) is None
    monkeypatch.setattr(edit, "run_test", lambda: _command_run(py=_NOT_RUN, js=_test_result()))
    assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}) == \
        (frozenset(), testcreate.SKIP_NO_BASELINE)
    # A primary language with no test yet is a baseline, and the other command's
    # failing tests stay in it
    monkeypatch.setattr(edit, "run_test", lambda: _command_run(
        py=_NO_TEST, js=_test_result("failed", ["j::old"])))
    assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}) == \
        (_baseline(["j::old"], py="no test before", js="red before"), "")
    monkeypatch.setattr(edit, "run_test", lambda: _command_run(py=_NO_TEST, js=_NOT_RUN))
    assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}) == \
        (_baseline(py="no test before", js="not compared"), "")
    # A run the primary command is not part of is no baseline
    assert testcreate.baseline_of(_command_run(js=_test_result())) is None
    assert testcreate.baseline_of(_command_run(js=_test_result(), rs=_test_result())) is None


def test_a_run_that_ended_before_every_test_was_run_is_no_baseline(repo, monkeypatch):
    """The file the run names is not the one test failing: no test was run."""
    directory, _ = repo
    early = _test_result("failed", ["tests/test_plot.py"], output=_EARLY_STOP_OUTPUT)
    assert testcreate.baseline_of(early) is None
    assert testcreate.baseline_of(_command_run(py=early, js=_test_result())) is None
    monkeypatch.setattr(edit, "run_test", lambda: dict(early))
    assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}) == \
        (frozenset(), testcreate.SKIP_NO_BASELINE)
    # Another language's command that was red and ended early leaves no baseline either,
    # where the primary command found no test as well
    assert testcreate.baseline_of(_command_run(py=_test_result(), js=early)) is None
    for run in (_command_run(py=_test_result(), js=early), _command_run(py=_NO_TEST, js=early)):
        monkeypatch.setattr(edit, "run_test", lambda _run=run: _run)
        assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}) == \
            (frozenset(), testcreate.SKIP_NO_BASELINE)
    # A test named as an error by a run that went on to the end is a baseline
    named = _test_result("failed", ["t::a"], output="ERROR t::a - boom\n1 error in 0.1s")
    assert testcreate.baseline_of(named) == frozenset({"t::a"})


def test_a_run_that_says_nothing_of_the_tests_it_ran_is_no_baseline(repo, monkeypatch):
    """A command whose run prints no line counting its tests leaves no baseline, green,
    red or over no test, and the stage calls no model, where the primary command found
    no test as well."""
    directory, _ = repo
    for part in (_test_result(), _test_result("failed", ["t::a"]), _NOT_RUN):
        unread = edit.RunResult(part)
        unread.is_ran_unread = True
        assert testcreate.baseline_of(unread) is None
        assert (testcreate.baseline_of(_command_run(py=part, js=_test_result())) is None) \
            is (part is _NOT_RUN)
        run = _command_run(py=part, js=_test_result())
        run.part_dict["py"].is_ran_unread = True
        assert testcreate.baseline_of(run) is None
    for primary in (_test_result(), _NO_TEST):
        run = _command_run(py=primary, js=_test_result())
        run.part_dict["js"].is_ran_unread = True
        monkeypatch.setattr(edit, "run_test", lambda _run=run: _run)
        assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}) == \
            (frozenset(), testcreate.SKIP_NO_BASELINE)


def _record_output(name):
    """The output of one recorded run of a test command."""
    path = os.path.join(os.path.dirname(__file__), "data", "test_output", f"{name}.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["output"]


@pytest.mark.parametrize("ext, name_list, output, whole_output, stop_line", [
    ("py", ["tests/test_plot.py"], _EARLY_STOP_OUTPUT,
     "ERROR tests/test_plot.py - boom\n1 error in 0.1s",
     "!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!"),
    ("java", ["com.example.b.BTest.one", "BTest.one"], _record_output("java_B_multi"),
     _record_output("java_R"), "[ERROR] COMPILATION ERROR : "),
    ("kt", ["BTest > one()"], _record_output("kt_B_multi"), _record_output("kt_R_multi"),
     "Execution failed for task ':a:compileKotlin'.")])
def test_a_run_that_ended_before_every_test_was_run_is_a_reason_whatever_it_named(
        monkeypatch, ext, name_list, output, whole_output, stop_line):
    """Every failing test the run names failed before the stage started. The reason
    starts with the lines saying the run ended early."""
    monkeypatch.setattr(config, "PRIMARY_EXT", ext)
    baseline = frozenset(name_list)
    early = _test_result("failed", name_list, excerpt="the excerpt", output=output)
    reason = testcreate.run_reason(early, baseline)
    assert f"stopped before it had run every test ({stop_line}\n" in reason
    assert "\nthe excerpt). The failing tests it named up to there: " in reason
    assert reason.endswith(f"up to there: {', '.join(name_list)}")
    assert testcreate.whole_run_reason({"result": early, "change": {}}, baseline) == reason
    # A run that named no test at all
    assert testcreate.run_reason(dict(early, failed=[]), baseline).endswith("up to there: none")
    # A test the baseline does not hold is named as one that did not fail before
    assert "which did not fail before you started" in testcreate.run_reason(
        dict(early, failed=[*name_list, "t::new"]), baseline)
    # The same names from a run that went on to its end are no reason
    assert testcreate.run_reason(dict(early, output=whole_output), baseline) == ""


def test_a_run_over_no_test_is_a_baseline_with_nothing_failing(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(edit, "run_test",
                        lambda: _test_result("did not run", output=_NO_TEST_OUTPUT))
    assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}) == \
        (edit.NO_TEST_BASELINE, "")
    assert edit.baseline_name_list(edit.NO_TEST_BASELINE) == []
    # A run that did not run for another reason is still no baseline
    monkeypatch.setattr(edit, "run_test",
                        lambda: _test_result("did not run", output="Traceback"))
    assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}) == \
        (frozenset(), testcreate.SKIP_NO_BASELINE)


# What a test command that found no test prints and ends with, as pytest and as
# unittest (Python 3.12 and later, and 3.11) say it
_NO_TEST_RUN_LIST = [
    (5, "no tests ran in 0.01s"),
    (5, "\n" + "-" * 70 + "\nRan 0 tests in 0.000s\n\nNO TESTS RAN"),
    (0, "\n" + "-" * 70 + "\nRan 0 tests in 0.000s\n\nOK")]


@pytest.mark.parametrize("exit_code, output", _NO_TEST_RUN_LIST)
def test_a_command_that_found_no_test_is_a_baseline_whichever_runner_says_so(
        repo, monkeypatch, exit_code, output):
    directory, _ = repo
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", f"import sys; print({output!r}); sys.exit({exit_code})"])
    edit.reset_patch_round()
    assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}) == \
        (edit.NO_TEST_BASELINE, "")


def test_the_tests_directory_is_made_where_the_tests_are_reached_by_path(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(config, "TEST_PATH", "spec")
    monkeypatch.setattr(edit, "run_test",
                        lambda: _test_result("did not run", output=_NO_TEST_OUTPUT))
    assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}) == \
        (edit.NO_TEST_BASELINE, "")
    assert os.path.isdir(os.path.join(directory, "spec"))
    # A test task handed its baseline gets the directory too, and runs nothing
    monkeypatch.setattr(config, "TEST_PATH", "spec2")
    monkeypatch.setattr(edit, "run_test", _never_built)
    assert testcreate.baseline_or_skip(directory, {"x.py": ["f"]}, ["t::old"]) == \
        (frozenset({"t::old"}), "")
    assert os.path.isdir(os.path.join(directory, "spec2"))
    # Not for a build tool that finds the tests itself, and not where the stage skips
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_PATH", "spec3")
    testcreate.baseline_or_skip(directory, {"x.py": ["f"]}, [])
    assert not os.path.exists(os.path.join(directory, "spec3"))
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    testcreate.baseline_or_skip(directory, {}, [])
    assert not os.path.exists(os.path.join(directory, "spec3"))


_CASE_ANSWER = {"cases": [
    {"file": "y.py", "name": "h", "input": "h()", "expect": "returns 3",
     "from": "add h, which returns 3"},
    {"file": "z.py", "name": "nobody", "input": "nobody()", "expect": "returns 0",
     "from": "made up"}],
    "no_case": [{"file": "x.py", "name": "f", "why": "test_f already checks it"}]}


class _StageThatRuns:
    """A stand-in for RLM that records it was built and answers a settled run.

    The first stage hands _CASE_ANSWER over, or whatever case_answer holds.
    """

    built = []
    seen_dict = {}
    case_answer = _CASE_ANSWER

    def __init__(self, *args, **kwargs):
        self.stage = kwargs.get("stage")
        _StageThatRuns.built.append(self.stage)
        _StageThatRuns.seen_dict[self.stage] = {
            "tools": [tool.__name__ for tool in kwargs["tool_list"]]}

    def run(self, **kwargs):
        _StageThatRuns.seen_dict[self.stage]["work"] = kwargs["variable_dict"]["work"]
        answer = json.dumps(type(self).case_answer) if self.stage == "test_case" \
            else "wrote them"
        return {"answer": answer, "history": [], "iters": 1, "refused_completes": 0,
                "error": "", "outcome": "complete"}


def _run_of(directory, base, monkeypatch, test_list):
    """Run run_test_create() with the analysis stubbed and run_test() answering from a
    list, oldest first."""
    monkeypatch.setattr(integrate, "analyse", lambda _: {
        "no_resolve": {}, "name": {}, "range": _RANGE_DICT, "ref": [], "use": [],
        "signature": {_Y: {"h": "def h():"}}})
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    answer_list = list(test_list)
    monkeypatch.setattr(edit, "run_test", lambda *a, **k: answer_list.pop(0))
    monkeypatch.setattr(testcreate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(testcreate, "RLM", _StageThatRuns)
    _StageThatRuns.built.clear()
    _StageThatRuns.seen_dict.clear()
    return testcreate.run_test_create(base, "", "p", "out", request="change f")


@pytest.mark.parametrize("test_path", [".", ""])
def test_a_setting_under_which_no_path_is_a_test_calls_no_model(repo, monkeypatch,
                                                               test_path):
    """Python has a dependency file and no test file names: with no directory named,
    nothing the stage wrote would be a test."""
    directory, base = repo
    monkeypatch.setattr(config, "TEST_PATH", test_path)
    monkeypatch.setattr(edit, "has_test_command", lambda: True)
    assert testcreate.skip_reason(directory, {"x.py": ["f"]}) == \
        testcreate._SKIP_NO_TEST_RULE
    out_dict = _run_of(directory, base, monkeypatch, [])
    assert (out_dict["called"], out_dict["settled"]) == (False, True)
    assert out_dict["skipped"] == testcreate._SKIP_NO_TEST_RULE
    assert _StageThatRuns.built == []
    # A language whose test files are told by their names still writes them
    monkeypatch.setattr(config, "PRIMARY_EXT", "js")
    assert testcreate.skip_reason(directory, {"x.py": ["f"]}) == ""


def test_a_repository_without_tests_and_without_a_dependency_file_calls_no_model(
        repo, monkeypatch):
    directory, base = repo
    monkeypatch.setattr(config, "TEST_PATH", "spec")
    # SQL with a TEST_COMMAND of its own: a test command and no dependency file
    monkeypatch.setattr(config, "PRIMARY_EXT", "sql")
    monkeypatch.setattr(config, "TEST_COMMAND", ["sh", "tests/run.sh"])
    out_dict = _run_of(directory, base, monkeypatch, [])
    assert out_dict["called"] is False and out_dict["settled"] is True
    assert out_dict["skipped"] == testcreate._SKIP_NO_TEST_FILE
    assert _StageThatRuns.built == []


def test_a_repository_without_tests_gets_its_first_ones(repo, monkeypatch):
    directory, base = repo
    _write(directory, "pyproject.toml", "[project]\nname = 'p'\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "manifest")
    # With the tests under spec, tests/test_x.py is code the change touched
    monkeypatch.setattr(config, "TEST_PATH", "spec")
    monkeypatch.setattr(_StageThatRuns, "case_answer", dict(_CASE_ANSWER, no_case=[
        *_CASE_ANSWER["no_case"], {"file": "tests/test_x.py", "name": "test_f", "why": "-"}]))
    out_dict = _run_of(directory, base, monkeypatch,
                       [_test_result("did not run", output=_NO_TEST_OUTPUT),
                        _test_result(), _test_result(), _test_result()])
    assert out_dict["called"] is True and out_dict["skipped"] == ""
    assert _StageThatRuns.built == ["test_case", "test_create"]
    # The stage starts with an empty baseline, the tests' directory made, no test file
    # to read, and the dependency file among what it can write
    assert out_dict["work"]["already_failing"] == []
    assert os.path.isdir(os.path.join(directory, "spec"))
    assert _StageThatRuns.seen_dict["test_case"]["work"]["test_files"] == []
    assert out_dict["work"]["dep_files"] == ["pyproject.toml"]


def test_the_stage_writes_the_files_that_say_which_tests_are_built(repo, monkeypatch):
    """Those of edit.test_build_file_pattern_set(), a makefile among them for C."""
    directory, base = repo
    _write(directory, "pyproject.toml", "[project]\nname = 'p'\n")
    _write(directory, "Makefile", "test:\n\tpytest\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "manifest")
    monkeypatch.setattr(config, "TEST_PATH", "spec")
    monkeypatch.setattr(edit, "test_build_file_pattern_set",
                        lambda: frozenset({"pyproject.toml", "Makefile"}))
    monkeypatch.setattr(_StageThatRuns, "case_answer", dict(_CASE_ANSWER, no_case=[
        *_CASE_ANSWER["no_case"],
        {"file": "tests/test_x.py", "name": "test_f", "why": "-"}]))
    out_dict = _run_of(directory, base, monkeypatch,
                       [_test_result("did not run", output=_NO_TEST_OUTPUT),
                        _test_result(), _test_result(), _test_result()])
    assert out_dict["work"]["dep_files"] == ["Makefile", "pyproject.toml"]


def test_tests_that_name_nothing_before_the_stage_call_no_model(repo, monkeypatch):
    directory, base = repo
    out_dict = _run_of(directory, base, monkeypatch, [_test_result("failed")])
    assert out_dict["skipped"] == testcreate.SKIP_NO_BASELINE
    assert _StageThatRuns.built == []


def test_the_cases_are_decided_first_and_handed_to_the_stage_that_writes(repo,
                                                                          monkeypatch):
    directory, base = repo
    out_dict = _run_of(directory, base, monkeypatch, [_test_result(), _test_result()])
    assert _StageThatRuns.built == ["test_case", "test_create"]
    assert out_dict["called"] is True and out_dict["case_iters"] == 1
    # The case naming a definition the change did not touch is left out
    assert out_dict["cases"] == [{"id": "case_1", "file": "y.py", "name": "h",
                                  "input": "h()", "expect": "returns 3",
                                  "from": "add h, which returns 3"}]
    assert [case["id"] for case in out_dict["work"]["cases"]] == ["case_1"]
    assert out_dict["tests_changed"] == [] and out_dict["tests_base"] == ""


class _StageThatWrites(_StageThatRuns):
    """_StageThatRuns, whose second stage writes the test of case_1 with create_file and
    calls COMPLETE() until the guard takes it."""

    def run(self, **kwargs):
        if self.stage == "test_case":
            return super().run(**kwargs)
        edit.create_file("tests/test_y.py", "def test_h_case_1():\n    assert True\n")
        refuse_count = 0
        while kwargs["on_complete"]("wrote them"):
            refuse_count += 1
        return {"answer": "wrote them", "history": [], "iters": 1,
                "refused_completes": refuse_count, "error": "", "outcome": "complete"}


def test_a_case_file_the_check_cannot_put_back_leaves_the_stage_open_with_a_reason(
        repo, monkeypatch):
    """The test command leaves a named pipe at the file the check of the whole run added
    a line to: the stage ends with that as what is left, not with an error."""
    directory, base = repo
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", [sys.executable, "-c", (
        "import os, pathlib, sys\n"
        "for p in pathlib.Path('tests').rglob('*.py'):\n"
        "    if p.is_file() and p.read_text().splitlines()[-1:] == [')']:\n"
        "        os.remove(p); os.mkfifo(p); sys.exit(1)\n"
        "print('1 passed in 0.01s')\n")])
    monkeypatch.setattr(integrate, "analyse", lambda _: {
        "no_resolve": {}, "name": {}, "range": _RANGE_DICT, "ref": [], "use": [],
        "signature": {_Y: {"h": "def h():"}}})
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(testcreate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(testcreate, "RLM", _StageThatWrites)
    _StageThatRuns.built.clear()
    _StageThatRuns.seen_dict.clear()
    reason_list = []
    check = testcreate.ref_and_test_reason_list

    def keep_reason(*args, **kwargs):
        line_list = check(*args, **kwargs)
        reason_list.append(line_list)
        return line_list

    monkeypatch.setattr(testcreate, "ref_and_test_reason_list", keep_reason)
    out_dict = testcreate.run_test_create(base, "", "p", "out", request="change f")
    # The first check names the file it could not put back; the ones after it find the
    # case in no test file
    assert reason_list[0] == [
        f"- tests/test_y.py no longer holds what you wrote: the whole run tried with a "
        f"line added to it left it so that it could not be written back "
        f"({os.path.join(directory, 'tests/test_y.py')} is not a regular file). Put it "
        f"back with revert_file, and the same for any file that run deleted or put in "
        f"its place (in git_diff_stat, one you did not change that has deletions only "
        f"or is untracked), then write it again"]
    assert out_dict["settled"] is False and len(out_dict["left"]) == 1
    assert "case_1" in out_dict["left"][0]
    assert stat.S_ISFIFO(os.lstat(os.path.join(directory, "tests/test_y.py")).st_mode)
    assert _read(directory, "tests/test_x.py") == "def test_f():\n    assert True\n"


class _StageThatPutsBack(_StageThatRuns):
    """_StageThatRuns, whose second stage writes the test of case_1 into PUT_BACK_PATH
    with its own create_file or replace_in_file, and does what a refusal that says the
    file no longer holds it asks with its own tools before calling COMPLETE() again."""

    PUT_BACK_PATH = "tests/test_y.py"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_dict = {tool.__name__: tool for tool in kwargs["tool_list"]}

    def run(self, **kwargs):
        if self.stage == "test_case":
            return super().run(**kwargs)
        tool_dict = self.tool_dict
        path = self.PUT_BACK_PATH

        def write_case():
            if os.path.isfile(os.path.join(config.PROJECT_DIR, path)):
                return tool_dict["replace_in_file"](path, "assert True",
                                                    "assert True  # case_1")
            return tool_dict["create_file"](path, "def test_h_case_1():\n    assert True\n")

        first_dict = write_case()
        assert first_dict["ok"], first_dict
        refuse_count = 0
        for _ in range(3):
            refusal = kwargs["on_complete"]("wrote them")
            if not refusal:
                break
            refuse_count += 1
            assert "no longer holds what you wrote" in str(refusal)
            assert tool_dict["revert_file"](path)["ok"]
            assert write_case()["ok"]
        return {"answer": "wrote them", "history": [], "iters": 1,
                "refused_completes": refuse_count, "error": "", "outcome": "complete"}


@pytest.mark.parametrize("path", ["tests/test_y.py", "tests/test_x.py"])
@pytest.mark.parametrize("leave", ["named pipe", "directory", "empty directory"])
def test_a_case_file_the_check_cannot_put_back_is_put_back_by_the_stage_s_revert_file(
        repo, monkeypatch, tmp_path, path, leave):
    """The test command leaves something other than a file, once, where the file the
    stage wrote, or the one HEAD holds that it changed, was: the stage's revert_file puts
    it back and the stage settles once it writes the case again."""
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
    use_list = [(_TEST_X, _Y)] if path == "tests/test_x.py" else []
    monkeypatch.setattr(integrate, "analyse", lambda _: {
        "no_resolve": {}, "name": {}, "range": _RANGE_DICT, "ref": [], "use": use_list,
        "signature": {_Y: {"h": "def h():"}}})
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(testcreate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(_StageThatPutsBack, "PUT_BACK_PATH", path)
    monkeypatch.setattr(testcreate, "RLM", _StageThatPutsBack)
    out_dict = testcreate.run_test_create(base, "", "p", "out", request="change f")
    assert mark.exists()
    assert out_dict["settled"] is True, out_dict["left"]
    assert os.path.isfile(os.path.join(directory, path))
    assert "case_1" in _read(directory, path)


def test_the_first_stage_is_shown_no_code_and_neither_stage_what_a_run_said_it_did(
        repo, monkeypatch):
    directory, base = repo
    _run_of(directory, base, monkeypatch, [_test_result(), _test_result()])
    case_seen, write_seen = (_StageThatRuns.seen_dict[name]
                             for name in ("test_case", "test_create"))
    assert case_seen["tools"] == ["read_source_file"]
    assert set(case_seen["work"]) == {"request", "branches", "targets", "test_files",
                                      "complete"}
    assert {"file": "y.py", "name": "h", "signature": "def h():", "tests": []} \
        in case_seen["work"]["targets"]
    assert case_seen["work"]["test_files"] == ["tests/test_x.py"]
    for work_dict in (case_seen["work"], write_seen["work"]):
        assert work_dict["branches"] == [{"asked": "", "tests_changed": []}]
        assert "changed f and added h" not in json.dumps(work_dict)
    assert "replace_in_file" in write_seen["tools"]


def test_what_was_asked_is_in_the_block_after_the_first_stages_request():
    text = testcreate.case_start_text({
        "request": "add h", "branches": [{"asked": "add h to y.py", "tests_changed": []},
                                         {"asked": "", "tests_changed": []}],
        "targets": [{"file": "y.py", "name": "h", "signature": "def h():", "tests": []},
                    {"file": "x.py", "name": "f", "signature": "def f():",
                     "tests": [{"path": "tests/test_x.py", "lines": [1, 3]}]}],
        "test_files": ["tests/test_x.py"]})
    assert text.startswith("## What was asked\n\nThe request: add h\n")
    assert "  add h to y.py" in text
    assert "h in y.py — `def h():` — no test refers to it" in text
    assert "f in x.py — `def f():` — tests/test_x.py line(s) 1, 3" in text
    assert text.endswith("which read_source_file reads:\n  tests/test_x.py")
    assert "turn" not in text and "COMPLETE()" not in text


def test_the_first_stage_is_sent_what_was_asked_once_and_the_countdown_every_turn(
        monkeypatch):
    """The block is part of the request; what follows the turns is the countdown alone."""
    handed = {}

    def run_stage(stage, request, variable_dict, var_doc, late_note, late_close, **run_kw):
        handed.update(request=request, **run_kw)
        return {"answer": {"cases": []}, "error": "", "iters": 1, "outcome": "complete"}

    monkeypatch.setattr(config, "EDIT_DEADLINE_TURN", 1)
    monkeypatch.setattr(testcreate, "run_stage", run_stage)
    monkeypatch.setattr(testcreate, "_make_stage", lambda *args: None)
    work_dict = {"request": "add h", "branches": [{"asked": "add h to y.py",
                                                  "tests_changed": []}],
                 "targets": [{"file": "y.py", "name": "h", "signature": "def h():",
                              "tests": []}], "test_files": []}
    testcreate.run_test_case("add h", work_dict, 1)
    assert handed["request"].startswith(
        "Decide the test cases of the change made for this request: add h\n\n"
        "## What was asked\n\n")
    assert "h in y.py — `def h():`" in handed["request"]
    assert handed["turn_note"](1, 3) == ""
    assert handed["turn_note"](3, 3).startswith("## 1 turns left") and "late_state" not in handed


def test_the_first_stage_reads_the_test_files_and_nothing_else(repo, monkeypatch):
    directory, _ = repo
    read_list = []

    def read_source_file(path, *a, **k):
        read_list.append(path)
        return "text"

    monkeypatch.setattr(source, "read_source_file", read_source_file)
    (tool,) = testcreate.case_tool_list()
    assert tool.__name__ == "read_source_file"
    assert tool("tests/test_x.py") == "text"
    for path in ("x.py", "tests/../x.py", os.path.join(directory, "y.py")):
        assert tool(path).startswith("Error: ") and "not a test file" in tool(path)
    assert read_list == ["tests/test_x.py"]


def test_cases_that_cannot_be_read_end_the_run_unsettled_with_nothing_written(
        repo, monkeypatch):
    directory, base = repo
    monkeypatch.setattr(_StageThatRuns, "case_answer", ["not", "an", "object"])
    out_dict = _run_of(directory, base, monkeypatch, [_test_result()])
    assert _StageThatRuns.built == ["test_case"]
    assert out_dict["settled"] is False and out_dict["called"] is False
    assert "the test cases could not be read" in out_dict["left"][0]
    assert out_dict["skipped"] == ""


def test_a_reason_for_every_definition_writes_nothing_and_is_settled(repo, monkeypatch):
    directory, base = repo
    monkeypatch.setattr(_StageThatRuns, "case_answer", {"cases": [], "no_case": [
        {"file": "x.py", "name": "f", "why": "checked"},
        {"file": "y.py", "name": "h", "why": "checked"}]})
    out_dict = _run_of(directory, base, monkeypatch, [_test_result()])
    assert _StageThatRuns.built == ["test_case"]
    assert out_dict["settled"] is True and out_dict["skipped"] == testcreate._SKIP_NO_CASE
    assert [entry["name"] for entry in out_dict["no_case"]] == ["f", "h"]


def test_a_definition_with_no_case_and_no_reason_leaves_the_run_unsettled(repo,
                                                                         monkeypatch):
    directory, base = repo
    monkeypatch.setattr(_StageThatRuns, "case_answer", {"cases": [_CASE_ANSWER["cases"][0]]})
    out_dict = _run_of(directory, base, monkeypatch, [_test_result()])
    # The stage that writes is not started, whatever cases were decided
    assert _StageThatRuns.built == ["test_case"]
    assert out_dict["settled"] is False and out_dict["called"] is False
    assert out_dict["skipped"] == "" and [c["id"] for c in out_dict["cases"]] == ["case_1"]
    assert out_dict["uncovered"] == [{"file": "x.py", "name": "f"}]
    assert out_dict["left"] == ["- no case was decided for: f in x.py"]
    # A reason that says nothing is no reason
    monkeypatch.setattr(_StageThatRuns, "case_answer", {
        "cases": [_CASE_ANSWER["cases"][0]],
        "no_case": [{"file": "x.py", "name": "f", "why": "  "}]})
    out_dict = _run_of(directory, base, monkeypatch, [_test_result()])
    assert _StageThatRuns.built == ["test_case"]
    assert out_dict["uncovered"] == [{"file": "x.py", "name": "f"}]
    # With no case at all it is unsettled too
    monkeypatch.setattr(_StageThatRuns, "case_answer", {"cases": []})
    out_dict = _run_of(directory, base, monkeypatch, [_test_result()])
    assert _StageThatRuns.built == ["test_case"]
    assert out_dict["settled"] is False and out_dict["skipped"] == ""
    assert len(out_dict["uncovered"]) == 2


def test_the_case_ids_of_a_test_file_that_is_not_utf_8_are_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(file_utils, "SOURCE_ENCODING", ["cp932"])
    (tmp_path / "test_sj.py").write_bytes(
        "def test_f_case_11():  # 表示\n    assert True\n".encode("cp932"))
    assert testcreate.file_case_number_set(str(tmp_path), "test_sj.py") == {11}


def test_a_run_of_digits_no_case_is_numbered_with_is_no_case_id(tmp_path):
    (tmp_path / "test_x.py").write_text(
        "# case_" + "9" * 5000 + " and case_12 and case_" + "1" * 19 + "\n")
    assert testcreate.file_case_number_set(str(tmp_path), "test_x.py") == {12}
    assert testcreate.next_case_number(str(tmp_path), ["test_x.py"]) == 13


def test_a_test_file_that_is_a_named_pipe_or_too_large_holds_no_case_id(tmp_path,
                                                                         monkeypatch):
    os.mkfifo(tmp_path / "test_p.py")
    assert testcreate.file_case_number_set(str(tmp_path), "test_p.py") == set()
    (tmp_path / "test_x.py").write_text("# case_12\n")
    monkeypatch.setattr(tmpfile, "READ_MAX_BYTE", 9)
    assert testcreate.file_case_number_set(str(tmp_path), "test_x.py") == set()
    monkeypatch.setattr(tmpfile, "READ_MAX_BYTE", 10)
    assert testcreate.file_case_number_set(str(tmp_path), "test_x.py") == {12}


def test_the_cases_are_numbered_on_from_the_ids_the_test_files_hold(repo, monkeypatch):
    directory, base = repo
    _write(directory, "tests/test_x.py",
           "def test_f_case_7():  # case_07, testcase_30 and showcase_40\n    assert True\n")
    _write(directory, "tests/blob.bin", "case_99")
    with open(os.path.join(directory, "tests/blob.bin"), "wb") as handle:
        handle.write(b"\xff\xfe case_99")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "tests an earlier run wrote")
    assert testcreate.next_case_number(directory, ["tests/test_x.py", "tests/blob.bin",
                                                   "tests/gone.py"]) == 8
    assert testcreate.next_case_number(directory, []) == 1
    out_dict = _run_of(directory, base, monkeypatch, [_test_result(), _test_result()])
    assert [case["id"] for case in out_dict["cases"]] == ["case_8"]
    # A line added to the file that holds the earlier ids places none of this run's
    edit.reset_patch_round(round_limit=0)
    _write(directory, "tests/test_x.py", _read(directory, "tests/test_x.py") + "# more\n")
    assert testcreate.case_place_dict(directory, out_dict["cases"]) == {"case_8": []}
    _write(directory, "tests/test_x.py",
           _read(directory, "tests/test_x.py") + "def test_h_case_8():\n    pass\n")
    assert testcreate.case_place_dict(directory, out_dict["cases"]) == {
        "case_8": ["tests/test_x.py"]}
    # A data file the test command does not run places no case
    _write(directory, "tests/data/pairs.json", '{"case_8": [1, 2]}\n')
    assert testcreate.case_place_dict(directory, out_dict["cases"]) == {
        "case_8": ["tests/test_x.py"]}


_H_TARGET = [{"file": "y.py", "name": "h", "signature": "def h():", "tests": []}]


def test_the_cases_are_numbered_by_the_machine_and_none_is_left_out_for_their_count(
        monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    answer = {"cases": [{"name": "h", "expect": f"e{n}", "id": "mine"} for n in range(5)]
              + [{"name": "h"}, "text"]}
    read_dict = testcreate.parse_case_list(answer, "", _H_TARGET)
    assert [(case["id"], case["expect"]) for case in read_dict["cases"]] == \
        [(f"case_{n + 1}", f"e{n}") for n in range(5)]
    assert "cut" not in read_dict
    assert "At most" not in testcreate.build_test_case_instruction("py")
    # A long answer handed over as a string is read whole
    text = json.dumps({"cases": [{"name": "h", "expect": "e" * 800} for _ in range(100)]})
    assert len(text) > config.PLAN_ANSWER_MAX_CHAR
    assert len(testcreate.parse_case_list(text, "", _H_TARGET)["cases"]) == 100
    assert testcreate.parse_case_list(None, "", _H_TARGET)["ok"] is False


def test_a_case_names_a_cobol_definition_in_either_case():
    target_list = [{"file": "src/MAIN.cbl", "name": "MAIN-PARA", "signature": "", "tests": []},
                   {"file": "y.py", "name": "h", "signature": "def h():", "tests": []}]
    read_dict = testcreate.parse_case_list(
        {"cases": [{"name": "main-para", "expect": "1"}, {"name": "H", "expect": "2"},
                   {"name": None, "expect": "3"}]}, "", target_list)
    assert [(case["file"], case["name"]) for case in read_dict["cases"]] == \
        [("src/MAIN.cbl", "MAIN-PARA")]


def test_every_definition_is_shown_to_the_first_stage_and_one_with_no_case_is_uncovered(
        monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    find_dict = {"referenced": [], "unreferenced": [{"file": "y.py", "name": f"h{n}"}
                                                    for n in range(5)]}
    work_dict = testcreate.case_work_dict([], "do it", [], find_dict, {}, "repo")
    assert [target["name"] for target in work_dict["targets"]] == [f"h{n}" for n in range(5)]
    assert all(f"h{n} in y.py" in testcreate.case_start_text(work_dict) for n in range(5))
    answer = {"cases": [{"file": "y.py", "name": f"h{n}", "expect": "e"} for n in range(3)]}
    read_dict = testcreate.parse_case_list(answer, "", work_dict["targets"])
    assert [entry["name"] for entry in read_dict["uncovered"]] == ["h3", "h4"]


def test_a_case_that_expects_zero_or_false_is_kept_and_one_that_expects_nothing_is_not():
    answer = {"cases": [{"name": "h", "expect": 0}, {"name": "h", "expect": False},
                        {"name": "h", "expect": ""}, {"name": "h", "expect": None},
                        {"name": "h"}]}
    read_dict = testcreate.parse_case_list(answer, "", _H_TARGET)
    assert [case["expect"] for case in read_dict["cases"]] == ["0", "False"]


def test_a_name_two_files_define_is_told_apart_by_the_file(monkeypatch):
    monkeypatch.setattr(config, "PROJECT_DIR", "/nowhere/p")
    target_list = [{"file": f"{package}/price.py", "name": "cut_price", "signature": "",
                    "tests": []} for package in ("shop", "depot")]
    answer = {"cases": [{"file": "depot/price.py", "name": "cut_price", "expect": 1},
                        {"file": "elsewhere.py", "name": "cut_price", "expect": 2}],
              "no_case": [{"file": "nowhere.py", "name": "cut_price", "why": "?"}]}
    read_dict = testcreate.parse_case_list(answer, "", target_list)
    # A file that is none of the targets' is not guessed at when the name is not unique
    assert [case["file"] for case in read_dict["cases"]] == ["depot/price.py"]
    assert read_dict["no_case"] == []
    assert read_dict["uncovered"] == [{"file": "shop/price.py", "name": "cut_price"}]
    # One target of that name takes the entry whatever file it names
    read_dict = testcreate.parse_case_list(answer, "", target_list[:1])
    assert [case["file"] for case in read_dict["cases"]] == ["shop/price.py"] * 2
    assert read_dict["uncovered"] == []


def test_the_hand_over_is_sent_back_while_a_definition_is_in_neither_list(monkeypatch):
    monkeypatch.setattr(config, "TEST_CREATE_MAX_REFUSAL", 2)
    monkeypatch.setattr(config, "PROJECT_DIR", "/nowhere/p")
    target_list = _H_TARGET + [{"file": "x.py", "name": "f", "signature": "", "tests": []}]
    guard = testcreate.make_case_guard(target_list)
    part = {"cases": [{"file": "y.py", "name": "h", "expect": 3}]}
    refusal = guard(part)
    assert "f in x.py" in refusal and "h in y.py" not in refusal
    assert "not valid JSON" in guard("{not json") or guard("{not json")
    whole = dict(part, no_case=[{"file": "x.py", "name": "f", "why": "checked"}])
    assert guard(whole) == ""
    # Sent back twice and no more
    assert guard(part) != "" and guard(part) == ""


@pytest.mark.parametrize("answer", [
    [{"file": "y.py", "name": "h", "expect": 3}],
    {"case_list": [{"file": "y.py", "name": "h", "expect": 3}]},
    {"cases": {"file": "y.py", "name": "h", "expect": 3}}])
def test_a_hand_over_that_is_not_the_two_lists_is_sent_back_once_with_the_shape(
        monkeypatch, answer):
    monkeypatch.setattr(config, "TEST_CREATE_MAX_REFUSAL", 2)
    monkeypatch.setattr(config, "PROJECT_DIR", "/nowhere/p")
    guard = testcreate.make_case_guard(_H_TARGET)
    refusal = guard(answer)
    assert 'the answer has no "cases" list' in refusal
    assert 'COMPLETE({"cases": case_list, "no_case": no_case_list})' in refusal
    assert guard(answer) == ""
    assert testcreate.parse_case_list(answer, "", _H_TARGET) == {
        "ok": False, "error": 'the answer has no "cases" list'}


def test_the_note_over_the_last_turns_asks_for_the_shape_the_instructions_ask_for():
    shape = 'COMPLETE({"cases": case_list, "no_case": no_case_list})'
    assert shape in testcreate._CASE_DEADLINE_WHAT
    assert shape in testcreate.build_test_case_instruction("py")


def test_the_targets_are_listed_with_the_baseline_and_no_model(repo, monkeypatch):
    directory, base = repo
    monkeypatch.setattr(integrate, "analyse", lambda _: {
        "no_resolve": {}, "name": {}, "range": _RANGE_DICT, "ref": [], "use": [], "signature": {}})
    monkeypatch.setattr(edit, "run_test", lambda *a, **k: _test_result("failed", ["t::b",
                                                                                  "t::a"]))
    monkeypatch.setattr(testcreate, "RLM", _never_built)
    assert testcreate.list_target(base, "", "p") == {
        "targets": {"x.py": ["f"], "y.py": ["h"]}, "tests": {"x.py": [], "y.py": []},
        "baseline": ["t::a", "t::b"], "skipped": "", "first_case": 1}
    monkeypatch.setattr(config, "TEST_PATH", "spec")
    monkeypatch.setattr(config, "PRIMARY_EXT", "sql")
    monkeypatch.setattr(config, "TEST_COMMAND", ["sh", "tests/run.sh"])
    assert testcreate.list_target(base, "", "p")["skipped"] == testcreate._SKIP_NO_TEST_FILE


def _never_built(*args, **kwargs):
    raise AssertionError("a stage was built")


def test_a_test_task_takes_its_own_files_and_the_baseline_it_is_handed(repo, monkeypatch):
    directory, base = repo
    monkeypatch.setattr(_StageThatRuns, "case_answer", {"cases": [_CASE_ANSWER["cases"][0]]})
    monkeypatch.setattr(integrate, "analyse", lambda _: {
        "no_resolve": {}, "name": {}, "range": _RANGE_DICT, "ref": [], "use": [], "signature": {}})
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    run_list = []
    monkeypatch.setattr(edit, "run_test",
                        lambda *a, **k: run_list.append(a) or _test_result())
    monkeypatch.setattr(testcreate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(testcreate, "RLM", _StageThatRuns)
    _StageThatRuns.seen_dict.clear()
    out_dict = testcreate.run_test_create(
        base, "", "p", "out",
        task_dict={"paths": ["y.py"], "test_paths": ["tests/test_x.py"], "baseline": ["t::old"]})
    # x.py is another test task's, so f is neither a target nor left uncovered here
    assert [t["name"] for t in _StageThatRuns.seen_dict["test_case"]["work"]["targets"]] == ["h"]
    assert out_dict["settled"] is True and out_dict["uncovered"] == []
    assert out_dict["work"]["already_failing"] == ["t::old"]
    # The baseline was handed over, so the tests were not run to read it
    assert run_list == []
    assert out_dict["work"]["own_files"] == ["y.py"]
    assert out_dict["work"]["test_files_to_extend"] == ["tests/test_x.py"]


def test_a_test_task_numbers_its_cases_from_the_number_and_by_the_step_it_is_handed(
        repo, monkeypatch):
    directory, base = repo
    case = _CASE_ANSWER["cases"][0]
    monkeypatch.setattr(_StageThatRuns, "case_answer", {"cases": [case, case]})
    monkeypatch.setattr(integrate, "analyse", lambda _: {
        "no_resolve": {}, "name": {}, "range": _RANGE_DICT, "ref": [], "use": [], "signature": {}})
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "run_test", lambda *a, **k: _test_result())
    monkeypatch.setattr(testcreate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(testcreate, "RLM", _StageThatRuns)
    task_dict = {"paths": ["y.py"], "test_paths": [], "baseline": []}
    out_dict = testcreate.run_test_create(base, "", "p", "out",
                                          task_dict={**task_dict, "first_case": 4,
                                                     "case_step": 3})
    assert [entry["id"] for entry in out_dict["cases"]] == ["case_4", "case_7"]
    out_dict = testcreate.run_test_create(base, "", "p", "out", task_dict=task_dict)
    assert [entry["id"] for entry in out_dict["cases"]] == ["case_1", "case_2"]


def test_the_test_files_that_use_a_file_are_read_off_every_use_of_it(
        monkeypatch):
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    found_dict = testcreate.test_by_file_dict(
        ["x.py", "y.py"], [(_TEST_X, _X), (_Y, _X), ("p/tests/test_all_py/test_all.py", _X),
                           ("p/tests/test_all_py/test_all.py", "p/z_py/z.py")], "p")
    assert found_dict == {"x.py": ["tests/test_all.py", "tests/test_x.py"], "y.py": []}


def test_a_run_writes_its_own_files_the_tests_that_refer_to_them_and_new_tests(repo,
                                                                              monkeypatch):
    directory, _ = repo
    may_write = testcreate.may_write_of(["y.py"], ["tests/test_x.py"],
                                        ["x.py", "y.py", "tests/test_x.py", "tests/conftest.py"])
    assert [may_write(path) for path in ("y.py", "tests/test_x.py", "tests/test_new.py")] == \
        [True] * 3
    # Another run's file, a test file nobody handed over, and a new file outside the tests
    assert [may_write(path) for path in ("x.py", "tests/conftest.py", "z.py")] == [False] * 3
    assert may_write("pyproject.toml") is False
    # A dependency file, at any depth, where the run is given the language's names
    may_write = testcreate.may_write_of(["y.py"], [], ["x.py", "y.py"],
                                        frozenset({"pyproject.toml"}))
    assert [may_write(path) for path in ("pyproject.toml", "sub/pyproject.toml")] == [True] * 2
    assert [may_write(path) for path in ("x.py", "pyproject.txt")] == [False] * 2


def test_a_run_in_a_repository_with_no_test_directory_writes_beside_its_tests(monkeypatch):
    """A fixture under another directory the language keeps tests in is written, and a
    write refused elsewhere is the only one counted against the run."""
    monkeypatch.setattr(config, "PRIMARY_EXT", "ts")
    monkeypatch.setattr(config, "TEST_PATH", "test")
    monkeypatch.setattr(config, "TEST_PATH_ALT_TUPLE", ("tests",))
    may_write = testcreate.may_write_of(["src/a.ts"], [], ["src/a.ts", "src/b.ts"])
    assert [may_write(path) for path in ("tests/a.test.ts", "tests/fixtures/app/a.ts",
                                         "test/helper.ts", "src/b.ts", "src/c.ts")] == \
        [True, True, True, False, False]
    assert testcreate._other_refuse_list(["tests/fixtures/app/a.ts", "src/c.ts"]) == \
        ["src/c.ts"]


def test_a_c_run_writes_the_makefile_that_says_which_tests_are_built(monkeypatch):
    """A test file the makefile's test target does not build is never run."""
    monkeypatch.setattr(config, "PRIMARY_EXT", "c")
    monkeypatch.setattr(edit, "other_test_list", lambda: [])
    pattern_set = edit.test_build_file_pattern_set()
    assert {"CMakeLists.txt", "Makefile", "makefile", "GNUmakefile"} <= pattern_set
    # The package cache is refilled for a dependency file alone
    assert "Makefile" not in edit.dep_file_pattern_set()
    may_write = testcreate.may_write_of(["src/a.c"], [], ["src/a.c", "Makefile"],
                                        pattern_set)
    assert [may_write(path) for path in ("Makefile", "tests/Makefile", "src/b.c")] == \
        [True, True, False]
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    assert "Makefile" not in edit.test_build_file_pattern_set()


def test_a_run_writes_a_file_that_sets_how_the_tests_are_found_and_run(repo, monkeypatch):
    """Where it is one of the run's kinds of files: new under the tests, handed over, or
    of a dependency file's name."""
    may_write = testcreate.may_write_of(["y.py"], ["tests/conftest.py"], ["x.py", "y.py"],
                                        frozenset({"pyproject.toml"}))
    assert [may_write(path) for path in (
        "tests/conftest.py", "tests/sub/conftest.py", "tests/pytest.ini",
        "pyproject.toml")] == [True] * 4
    assert [may_write(path) for path in ("conftest.py", "pytest.ini")] == [False] * 2


def test_a_write_outside_the_runs_files_is_refused_and_nothing_is_written(repo, monkeypatch):
    directory, _ = repo
    edit.reset_patch_round(round_limit=0)
    refuse_list = []
    tool_dict = {tool.__name__: tool for tool in testcreate.stage_tool_list(
        testcreate.may_write_of(["y.py"], [], ["x.py", "y.py", "tests/test_x.py"]),
        refuse_list)}
    before = _read(directory, "x.py")
    for path in ("x.py", "tests/../x.py", os.path.join(directory, "x.py")):
        answer = tool_dict["replace_in_file"](path, "return 10", "return 11")
        assert answer["ok"] is False and "not one of this run's files" in answer["error"]
    assert tool_dict["create_file"]("z.py", "z = 1\n")["ok"] is False
    assert _read(directory, "x.py") == before
    assert not os.path.exists(os.path.join(directory, "z.py"))
    assert refuse_list == ["x.py", "z.py"]
    # Its own file and a new test file are written, under the tool's own name
    assert tool_dict["replace_in_file"]("y.py", "return 3", "return 4")["ok"] is True
    assert tool_dict["create_file"]("tests/test_y.py", "def test_h():\n    pass\n")["ok"] is True
    assert tool_dict["replace_in_file"].__name__ == "replace_in_file"
    assert refuse_list == ["x.py", "z.py"]


def test_a_revert_of_a_file_the_run_did_not_change_does_nothing_and_is_not_refused(
        repo, monkeypatch):
    directory, _ = repo
    edit.reset_patch_round(round_limit=0)
    refuse_list = []
    tool_dict = {tool.__name__: tool for tool in testcreate.stage_tool_list(
        testcreate.may_write_of(["y.py"], [], ["x.py", "y.py", "tests/test_x.py"]),
        refuse_list)}
    before = _read(directory, "x.py")
    assert tool_dict["revert_file"]("x.py") == {"ok": True, "path": "x.py", "error": ""}
    assert _read(directory, "x.py") == before and refuse_list == []
    # A file of the run's own that it changed is reverted
    assert tool_dict["replace_in_file"]("y.py", "return 3", "return 4")["ok"] is True
    assert tool_dict["revert_file"]("y.py")["ok"] is True
    assert "return 3" in _read(directory, "y.py")
    # A directory is not a file the run did not change
    assert tool_dict["revert_file"](".")["ok"] is False
    assert refuse_list == ["."]


def test_a_revert_of_a_file_the_run_changed_goes_through_whichever_file_it_is(
        repo, monkeypatch):
    """A file outside the run's own is not written by a tool, so this is a write made
    without one."""
    directory, _ = repo
    edit.reset_patch_round(round_limit=0)
    refuse_list = []
    tool_dict = {tool.__name__: tool for tool in testcreate.stage_tool_list(
        testcreate.may_write_of(["y.py"], [], ["x.py", "y.py", "tests/test_x.py"]),
        refuse_list)}
    before = _read(directory, "x.py")
    _write(directory, "x.py", "def f():\n    return 11\n")
    _write(directory, "z.py", "z = 1\n")
    assert tool_dict["revert_file"]("x.py") == {"ok": True, "path": "x.py", "error": ""}
    assert tool_dict["revert_file"]("z.py")["ok"] is True
    assert _read(directory, "x.py") == before
    assert not os.path.exists(os.path.join(directory, "z.py")) and refuse_list == []
    # The other writing tools still refuse those paths
    assert tool_dict["replace_in_file"]("x.py", "return 10", "return 11")["ok"] is False
    assert refuse_list == ["x.py"]


def test_a_knowledge_path_is_handed_to_the_tool_as_the_path_it_was_judged_as(
        repo, monkeypatch):
    directory, _ = repo
    edit.reset_patch_round(round_limit=0)
    name = os.path.basename(directory)
    tool_dict = {tool.__name__: tool for tool in testcreate.stage_tool_list(
        testcreate.may_write_of(["y.py"], [], ["x.py", "y.py", "tests/test_x.py"]), [])}
    knowledge = repo_to_knowledge("tests/test_new.py", name)
    assert tool_dict["create_file"](knowledge, "def test_n():\n    pass\n")["ok"] is True
    assert os.path.exists(os.path.join(directory, "tests", "test_new.py"))
    assert not os.path.exists(os.path.join(directory, name))
    read = testcreate.case_tool_list()[0]
    assert read(repo_to_knowledge("tests/test_x.py", name)) == _read(directory,
                                                                     "tests/test_x.py")


def test_a_symbolic_link_is_judged_as_the_file_it_points_at(repo, monkeypatch):
    """A write through a link lands on the link's target, and so does a read."""
    directory, _ = repo
    os.symlink(os.path.join("..", "x.py"), os.path.join(directory, "tests", "test_link.py"))
    edit.reset_patch_round(round_limit=0)
    refuse_list = []
    tool_dict = {tool.__name__: tool for tool in testcreate.stage_tool_list(
        testcreate.may_write_of(["y.py"], [], ["x.py", "y.py", "tests/test_x.py"]),
        refuse_list)}
    before = _read(directory, "x.py")
    answer = tool_dict["replace_in_file"]("tests/test_link.py", "return 10", "return 11")
    assert answer["ok"] is False and "not one of this run's files" in answer["error"]
    assert _read(directory, "x.py") == before and refuse_list == ["x.py"]
    monkeypatch.setattr(source, "read_source_file", lambda path, *a, **k: "text")
    (tool,) = testcreate.case_tool_list()
    assert "not a test file" in tool("tests/test_link.py")


def test_a_refused_write_outside_the_tests_leaves_the_run_unsettled(repo, monkeypatch):
    directory, base = repo

    class _StageThatReaches(_StageThatRuns):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tool_dict = {tool.__name__: tool for tool in kwargs["tool_list"]}

        def run(self, **kwargs):
            if self.stage == "test_create":
                # x.py is another test task's file, tests/conftest.py nobody's
                self.tool_dict["replace_in_file"](type(self).reach, "a", "b")
            return super().run(**kwargs)

    monkeypatch.setattr(integrate, "analyse", lambda _: {
        "no_resolve": {}, "name": {}, "range": _RANGE_DICT, "ref": [], "use": [], "signature": {}})
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "run_test", lambda *a, **k: _test_result())
    monkeypatch.setattr(testcreate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(testcreate, "RLM", _StageThatReaches)
    monkeypatch.setattr(_StageThatRuns, "case_answer", {"cases": [_CASE_ANSWER["cases"][0]]})
    task_dict = {"paths": ["y.py"], "test_paths": [], "baseline": []}
    _StageThatReaches.reach = "x.py"
    out_dict = testcreate.run_test_create(base, "", "p", "out", task_dict=task_dict)
    assert out_dict["settled"] is False and out_dict["refused"] == ["x.py"]
    assert out_dict["left"] == ["- a change outside this run's files was refused: x.py"]
    # The check found nothing and sent COMPLETE() back at no time
    assert out_dict["refused_completes"] == 0
    assert integrate.end_of(out_dict) == integrate.END_COMPLETE
    # A test file it was not handed is refused and recorded, and the run is not left open
    _write(directory, "tests/conftest.py", "a = 1\n")
    _StageThatReaches.reach = "tests/conftest.py"
    out_dict = testcreate.run_test_create(base, "", "p", "out", task_dict=task_dict)
    assert out_dict["settled"] is True and out_dict["refused"] == ["tests/conftest.py"]
    assert _read(directory, "tests/conftest.py") == "a = 1\n"
    # A dependency file is the run's to write, and the write lands
    _StageThatReaches.reach = "pyproject.toml"
    _write(directory, "pyproject.toml", "a\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "manifest")
    out_dict = testcreate.run_test_create(base, "", "p", "out", task_dict=task_dict)
    assert out_dict["settled"] is True and out_dict["refused"] == []
    assert _read(directory, "pyproject.toml") == "b\n"
    assert out_dict["non_test_changed"] == ["pyproject.toml"]


def test_a_test_and_a_source_file_the_stage_wrote_are_told_apart(repo, monkeypatch):
    directory, base = repo

    class _StageThatWrites(_StageThatRuns):
        def run(self, **kwargs):
            if self.stage == "test_create":
                _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
                _write(directory, "y.py", "def h():\n    return 4\n")
            return super().run(**kwargs)

    monkeypatch.setattr(integrate, "analyse", lambda _: {
        "no_resolve": {}, "name": {}, "range": _RANGE_DICT, "ref": [], "use": [], "signature": {}})
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    # The baseline, then the base's tree: the stand-in stage ends without the guard
    answer_list = [_test_result(), _test_result("failed", ["tests/test_y.py::test_h"])]
    monkeypatch.setattr(edit, "run_test", lambda *a, **k: answer_list.pop(0))
    monkeypatch.setattr(edit, "run_test_as_is", lambda *a, **k: answer_list.pop(0))
    monkeypatch.setattr(testcreate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(testcreate, "RLM", _StageThatWrites)
    out_dict = testcreate.run_test_create(base, "", "p", "out")
    assert out_dict["settled"] is True
    assert out_dict["tests_changed"] == ["tests/test_y.py"]
    assert out_dict["non_test_changed"] == ["y.py"]
    assert out_dict["tests_base"] == testcreate.BASE_RED
    assert out_dict["base_failed"] == ["tests/test_y.py::test_h"]
    # What the stage wrote is back in the working tree after the base's tree was read
    assert _read(directory, "y.py") == "def h():\n    return 4\n"
    assert os.path.isfile(os.path.join(directory, "tests/test_y.py"))


# == The check ===============================================


def _check_of(directory, monkeypatch, own_test_list, test_list, broken=None,
              case_list=None, check_before=None, in_whole_run=None,
              baseline=frozenset({"t::old"})):
    """Run reason_list() with the analysis stubbed, run_test() answering from a list and
    edit.in_whole_run() answering True or as in_whole_run does."""
    after_dict = {"no_resolve": broken or {}, "name": {}, "range": {}, "ref": []}
    monkeypatch.setattr(edit, "in_whole_run", in_whole_run or (lambda path, whole: True))
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(integrate, "analyse", lambda _: after_dict)
    monkeypatch.setattr(edit, "test_change_set", lambda: set(own_test_list))
    call_list = []

    def run_test(path_list=None):
        call_list.append(path_list)
        return test_list.pop(0)

    monkeypatch.setattr(edit, "run_test", run_test)
    start_dict = {"no_resolve": {}, "name": {}, "range": {}, "ref": []}
    return testcreate.reason_list(directory, "out", start_dict, baseline,
                                  None, case_list or [], check_before=check_before), call_list


def test_what_the_static_check_finds_that_it_did_not_before_is_a_reason(repo, monkeypatch):
    directory, _ = repo
    old = "x.py: F821 Undefined name `a`"
    new = "y.py: F821 Undefined name `g`"
    monkeypatch.setattr(edit, "run_check", lambda: collections.Counter([old, new]))
    # A red run returns early; the check is a reason on that path too
    line_list, _ = _check_of(directory, monkeypatch, ["tests/test_x.py"],
                             [_test_result("failed", ["t::new"], "boom")],
                             check_before=collections.Counter([old]))
    assert any(new in line and old not in line for line in line_list)
    line_list, _ = _check_of(directory, monkeypatch, ["tests/test_x.py"],
                             [_test_result(), _test_result()],
                             check_before=collections.Counter([old, new]))
    assert line_list == []


def test_nothing_is_left_when_both_runs_pass(repo, monkeypatch):
    directory, _ = repo
    line_list, call_list = _check_of(directory, monkeypatch, ["tests/test_x.py"],
                                     [_test_result(), _test_result()])
    assert line_list == [] and call_list == [["tests/test_x.py"], None]


def test_a_test_file_that_fails_without_naming_a_test_is_a_reason(repo, monkeypatch):
    """A file that cannot be collected names no test, and the whole suite's names are
    all in the baseline."""
    directory, _ = repo
    line_list, call_list = _check_of(
        directory, monkeypatch, ["tests/test_x.py"],
        [_test_result("failed", excerpt="ImportError"),
         _test_result("failed", ["t::old"])])
    assert len(line_list) == 1 and "tests/test_x.py alone" in line_list[0]
    assert "ImportError" in line_list[0]
    assert call_list == [["tests/test_x.py"]]


def test_a_test_that_did_not_fail_before_is_a_reason_and_an_old_one_is_not(repo,
                                                                          monkeypatch):
    directory, _ = repo
    line_list, _ = _check_of(directory, monkeypatch, [],
                             [_test_result("failed", ["t::old", "t::new"], "boom")])
    assert len(line_list) == 1 and "t::new" in line_list[0] and "t::old" not in line_list[0]
    line_list, _ = _check_of(directory, monkeypatch, [],
                             [_test_result("failed", ["t::old"])])
    assert line_list == []


def test_a_build_tool_that_takes_no_paths_runs_the_whole_suite_alone(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    _, call_list = _check_of(directory, monkeypatch, ["tests/test_x.py"], [_test_result()])
    assert call_list == [None]


def test_a_file_the_test_command_does_not_run_is_not_run_alone(repo, monkeypatch):
    """A data file under the tests is kept out of the run over the files the stage
    changed, and the test next to it is run."""
    directory, _ = repo
    _write(directory, "tests/data/pairs.json", "[[1, 2, 3]]\n")
    line_list, call_list = _check_of(directory, monkeypatch,
                                     ["tests/data/pairs.json", "tests/test_x.py"],
                                     [_test_result(), _test_result()])
    assert line_list == [] and call_list == [["tests/test_x.py"], None]
    # A data file alone is not run alone at all
    line_list, call_list = _check_of(directory, monkeypatch, ["tests/data/pairs.json"],
                                     [_test_result()])
    assert line_list == [] and call_list == [None]


def test_a_helper_that_holds_no_test_is_no_reason_and_a_file_with_a_case_is(repo,
                                                                            monkeypatch):
    directory, _ = repo
    _write(directory, "tests/conftest.py", "import os\n")
    no_test = _test_result("did not run", excerpt=_NO_TEST_OUTPUT, output=_NO_TEST_OUTPUT)
    line_list, call_list = _check_of(directory, monkeypatch, ["tests/conftest.py"],
                                     [dict(no_test), _test_result()])
    assert line_list == [] and call_list == [["tests/conftest.py"], None]
    _write(directory, "tests/check_x.py", "# case_1\n")
    line_list, _ = _check_of(directory, monkeypatch, ["tests/check_x.py"], [dict(no_test)])
    assert len(line_list) == 1 and "tests/check_x.py alone" in line_list[0]


def _ran_result(name_list, status="passed", failed=()):
    """One run_test() return value of one command that named the tests it ran."""
    result = edit.RunResult(_test_result(status, failed))
    result.ran_name_list, result.ran_count = sorted(name_list), len(name_list)
    return result


def test_a_test_that_ran_when_the_stage_started_and_runs_no_more_is_a_reason(repo,
                                                                            monkeypatch):
    """Whatever keeps it from running - a setting, a conftest.py, a skip - the whole
    suite's run is held to the tests the run before the stage ran."""
    directory, _ = repo
    monkeypatch.setattr(config, "TEST_COMMAND", ["t"])
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    _write(directory, "tests/check_b.py", "def test_b():\n    pass\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-q", "-m", "tests")
    baseline = testcreate.baseline_of(_ran_result(
        ["tests/test_x.py::test_f", "tests/check_b.py::test_b"]))
    _write(directory, "tests/pytest.ini", "[pytest]\nmarkers =\n    slow\n")
    line_list = _check_of(directory, monkeypatch, [], [_ran_result(
        ["tests/test_x.py::test_f", "tests/test_new.py::test_n"])], baseline=baseline)[0]
    assert line_list == ["- `t` no longer runs tests/check_b.py::test_b, which ran before "
                         "you started: they are not found, are left out or skip now"]
    # A new failing test is said first; the run that keeps every test is no reason
    assert "fails t::x" in testcreate.run_reason(
        _ran_result(["tests/test_x.py::test_f"], "failed", ["t::x"]), baseline)
    assert testcreate.run_reason(_ran_result(
        ["tests/test_x.py::test_f", "tests/check_b.py::test_b", "tests/test_new.py::t"]),
        baseline) == ""
    # A whole run that no longer says which tests it ran
    quiet = edit.RunResult(_test_result("passed", ()))
    quiet.is_ran_hold = True
    assert testcreate.run_reason(quiet, baseline) == (
        "`t` no longer prints which of its tests ran, as it did before you started, so "
        "they cannot be checked against the tests that ran then")


def test_a_file_that_is_not_the_stage_s_to_write_and_was_changed_is_a_reason(repo,
                                                                           monkeypatch):
    """However it was written: the stage's own code writes past its tools."""
    directory, _ = repo
    edit.reset_patch_round(round_limit=0)
    start_file_list = ["x.py", "tests/test_x.py"]
    may_write = testcreate.may_write_of(["x.py"], [], start_file_list)
    _write(directory, "tests/test_x.py", "import os\n")
    _write(directory, "x.py", "def f():\n    return 3\n")
    _write(directory, "tests/test_new.py", "def test_n():\n    pass\n")
    assert testcreate.not_own_change_list(start_file_list, may_write) == ["tests/test_x.py"]
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(integrate, "analyse", lambda _: {"no_resolve": {}, "name": {},
                                                         "range": {}, "ref": []})
    monkeypatch.setattr(edit, "run_test", lambda path_list=None: _test_result())
    start_dict = {"no_resolve": {}, "name": {}, "range": {}, "ref": []}
    line_list = testcreate.reason_list(directory, "out", start_dict, frozenset(), None, [],
                                       start_file_list=start_file_list, may_write=may_write)
    assert line_list[0] == testcreate.NOT_OWN_REASON.format(paths="tests/test_x.py")


def test_a_whole_run_with_no_test_is_no_reason_where_the_stage_left_nothing(repo,
                                                                           monkeypatch):
    """A repository with no test yet, every case of which the stage recorded and none
    of which it left a test for: the run is the one the stage started from."""
    directory, _ = repo
    no_test = _test_result("did not run", excerpt=_NO_TEST_OUTPUT, output=_NO_TEST_OUTPUT)
    monkeypatch.setattr(edit, "stage_change_hash_dict", lambda: {})
    line_list, call_list = _check_of(directory, monkeypatch, [], [dict(no_test)],
                                     baseline=edit.NO_TEST_BASELINE)
    assert line_list == [] and call_list == [None]
    # A run that did not run for another reason is a reason
    line_list, _ = _check_of(directory, monkeypatch, [], [_test_result(
        "did not run", excerpt="pytest: not found")], baseline=edit.NO_TEST_BASELINE)
    assert line_list == ["- `run them` did not run (pytest: not found)"]
    # and so is one with no test where the tests ran when the stage started
    line_list, _ = _check_of(directory, monkeypatch, [], [dict(no_test)])
    assert line_list == [f"- `run them` did not run ({_NO_TEST_OUTPUT})"]
    # and one with no test once the stage has left something in the repository
    monkeypatch.setattr(edit, "stage_change_hash_dict", lambda: {"pyproject.toml": "h"})
    line_list, _ = _check_of(directory, monkeypatch, [], [dict(no_test)],
                             baseline=edit.NO_TEST_BASELINE)
    assert line_list == [f"- `run them` did not run ({_NO_TEST_OUTPUT})"]


def test_the_state_reads_a_whole_run_with_no_primary_test_as_the_check_does(repo,
                                                                           monkeypatch):
    """The primary language holds no test yet and another language's tests pass."""
    directory, base = repo
    run = _command_run(py=_NO_TEST, js=_test_result())
    baseline = testcreate.baseline_of(run, is_no_test_green=True)
    work_dict = testcreate.build_work_dict(
        directory, base, "change f", {"referenced": [], "unreferenced": []}, baseline,
        [], [], ["tests/test_x.py"])
    integrate.reset_stage()
    edit.reset_patch_round(round_limit=0)
    turn_note, _ = testcreate.make_state(directory, work_dict, lambda: [], baseline,
                                         _RANGE_DICT, "p")
    # The stage has left nothing in the repository: the run is the one it started from
    _whole_run(monkeypatch, 1, run)
    assert "said nothing it did not say before you started" in turn_note(2, 9)
    assert work_dict["tests"] == "" and work_dict["tests_at"] == 1
    assert testcreate.whole_run_reason(edit.last_whole_run(), baseline) == ""
    # Once the stage has changed something, a primary command that runs no test says so
    _write(directory, "pyproject.toml", "[project]\n")
    _whole_run(monkeypatch, 2, run)
    assert "`py` did not run (" in turn_note(3, 9)
    assert work_dict["tests"].startswith("`py` did not run")


def test_a_test_that_fails_more_times_than_when_the_stage_started_is_a_reason():
    baseline = testcreate.baseline_of(_test_result("failed", ["t::a", "t::a"]))
    assert edit.baseline_name_list(baseline) == ["t::a"]
    assert testcreate.run_reason(_test_result("failed", ["t::a", "t::a"]), baseline) == ""
    assert testcreate.run_reason(_test_result("failed", ["t::a"]), baseline) == ""
    assert "` fails t::a, which did not fail before you started" in testcreate.run_reason(
        _test_result("failed", ["t::a", "t::a", "t::a"]), baseline)


def test_a_primary_command_that_found_no_test_is_read_as_the_start_reads_it():
    assert testcreate.run_reason(_NO_TEST, edit.NO_TEST_BASELINE,
                                 is_no_test_green=True) == ""
    assert testcreate.run_reason(_NOT_RUN, edit.NO_TEST_BASELINE,
                                 is_no_test_green=True).endswith(
        " did not run (jest: not found)")
    # Where tests ran when the stage started, finding none is a reason
    assert testcreate.run_reason(_NO_TEST, frozenset(), is_no_test_green=True).endswith(
        " did not run ()")
    baseline = testcreate.baseline_of(_command_run(py=_NO_TEST, js=_test_result()),
                                      is_no_test_green=True)
    assert testcreate.run_reason(_command_run(py=_NO_TEST, js=_test_result()), baseline,
                                 is_no_test_green=True) == ""
    assert testcreate.run_reason(
        _command_run(py=_NO_TEST, js=_test_result("failed", ["t::new"])), baseline,
        is_no_test_green=True).startswith("`js` fails t::new")


def test_a_file_with_a_case_the_whole_suite_never_builds_is_a_reason(repo, monkeypatch):
    """Where the build tool takes no paths, each file holding a case is checked to be
    one the whole suite builds, and a file holding no case is not."""
    directory, _ = repo
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "PRIMARY_EXT", "c")
    _write(directory, "tests/test_add.c", "/* case_1 */\n")
    _write(directory, "tests/test_in.c", "/* case_2 */\n")
    _write(directory, "tests/helper.c", "int h;\n")
    probe_list = []

    def in_whole_run(path, whole):
        probe_list.append(path)
        return path != "tests/test_add.c"

    line_list, call_list = _check_of(
        directory, monkeypatch, ["tests/helper.c", "tests/test_add.c", "tests/test_in.c"],
        [_test_result()], in_whole_run=in_whole_run)
    assert call_list == [None]
    assert probe_list == ["tests/test_add.c", "tests/test_in.c"]
    assert len(line_list) == 1 and "never runs tests/test_add.c" in line_list[0]
    # A red whole run is the reason, and nothing is probed
    probe_list.clear()
    line_list, _ = _check_of(directory, monkeypatch, ["tests/test_add.c"],
                             [_test_result("failed", ["t::new"], "boom")],
                             in_whole_run=in_whole_run)
    assert len(line_list) == 1 and "t::new" in line_list[0] and probe_list == []


def test_a_file_with_a_case_the_whole_suite_never_collects_is_a_reason_where_paths_are_taken(
        repo, monkeypatch):
    """pytest runs a file named on the command line whatever its name, and the whole run
    collects only the names its rules match."""
    directory, _ = repo
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    _write(directory, "tests/check_add.py", "def test_add():  # case_1\n    assert True\n")
    probe_list = []

    def in_whole_run(path, whole):
        probe_list.append(path)
        return False

    line_list, call_list = _check_of(directory, monkeypatch, ["tests/check_add.py"],
                                     [_test_result(), _test_result()],
                                     in_whole_run=in_whole_run)
    assert call_list == [["tests/check_add.py"], None]
    assert probe_list == ["tests/check_add.py"]
    assert len(line_list) == 1 and "never runs tests/check_add.py" in line_list[0]


def test_a_test_file_that_is_gone_and_a_call_that_no_longer_resolves_are_reasons(
        repo, monkeypatch):
    directory, _ = repo
    line_list, _ = _check_of(directory, monkeypatch, ["tests/test_gone.py"],
                             [_test_result()],
                             broken={(_TEST_X, _X, "f"): [1]})
    assert any("tests/test_gone.py" in line for line in line_list)
    assert any("f in tests/test_x.py" in line for line in line_list)


def test_a_case_in_no_test_and_no_decision_is_a_reason(repo, monkeypatch):
    directory, _ = repo
    case_list = [{"id": f"case_{n}"} for n in (1, 2, 10)]
    _write(directory, "tests/test_x.py", "def test_case_10_f():\n    assert True\n")
    integrate.reset_stage()
    line_list, _ = _check_of(directory, monkeypatch, ["tests/test_x.py"],
                             [_test_result(), _test_result()], case_list=case_list)
    assert len(line_list) == 1 and "case_1 and case_2" in line_list[0].replace(",", " and")
    assert "case_10" not in line_list[0]
    integrate.record_decision("case_1: the signature takes no argument", "left out")
    _write(directory, "tests/test_x.py", "# case_2\ndef test_case_10_f():\n    pass\n")
    line_list, _ = _check_of(directory, monkeypatch, ["tests/test_x.py"],
                             [_test_result(), _test_result()], case_list=case_list)
    assert line_list == []


# == The base's tree =========================================


def test_the_tests_are_run_over_the_base_and_the_working_tree_is_put_back(repo,
                                                                         monkeypatch):
    directory, base = repo
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    seen_dict = {}

    def run_test(path_list=None):
        seen_dict["x"] = _read(directory, "x.py")
        seen_dict["y"] = os.path.exists(os.path.join(directory, "y.py"))
        seen_dict["test"] = os.path.exists(os.path.join(directory, "tests/test_y.py"))
        return _test_result("failed", excerpt="no module y")

    monkeypatch.setattr(edit, "run_test_as_is", run_test)
    head = _git(directory, "rev-parse", "HEAD").strip()
    base_dict = testcreate.base_run_dict(directory, base, ["tests/test_y.py"], frozenset())
    assert base_dict == {"status": testcreate.BASE_RED, "failed": [], "is_part": False, "left_out": [],
        "is_no_run": False}
    assert seen_dict == {"x": "def f():\n    return 1\n\n\ndef g():\n    return 2\n\n\n",
                         "y": False, "test": True}
    assert "return 10" in _read(directory, "x.py")
    assert os.path.exists(os.path.join(directory, "y.py"))
    assert _git(directory, "rev-parse", "HEAD").strip() == head
    assert "tests/test_y.py" in _git(directory, "status", "--porcelain")


def test_the_run_over_the_base_leaves_the_index_as_it_was(repo, monkeypatch):
    """A file the stage made stays one git does not track, which revert_file() deletes,
    and what was staged stays staged."""
    directory, base = repo
    edit.reset_patch_round(round_limit=0)
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    _write(directory, "x.py", "def f():\n    return 11\n\n\ndef g():\n    return 2\n\n\n")
    _write(directory, "y.py", "def h():\n    return 4\n")
    _git(directory, "add", "y.py")
    before = _git(directory, "status", "--porcelain")
    monkeypatch.setattr(edit, "run_test_as_is", lambda path_list=None: _test_result())
    assert testcreate.base_run_dict(directory, base, ["tests/test_y.py"],
                                    frozenset())["status"] == testcreate.BASE_GREEN
    assert _git(directory, "status", "--porcelain") == before
    assert before.splitlines() == [" M x.py", "M  y.py", "?? tests/test_y.py"]
    assert not [name for name in os.listdir(os.path.join(directory, ".git"))
                if name.startswith("ctloom-index")]
    assert edit.revert_file("tests/test_y.py") == {
        "ok": True, "path": "tests/test_y.py", "error": ""}
    assert not os.path.exists(os.path.join(directory, "tests/test_y.py"))


def _fail_to_put_back(monkeypatch):
    """Have integrate.place() fail the second time it is called: the call that puts
    the working tree back after a run over the base."""
    real_place, call_list = integrate.place, []

    def place(project_dir, commit, env_dict=None):
        call_list.append(commit)
        if len(call_list) == 2:
            raise RuntimeError("could not put the merged tree in the working tree: locked")
        real_place(project_dir, commit, env_dict)

    monkeypatch.setattr(integrate, "place", place)


def test_a_working_tree_that_cannot_be_put_back_is_told_apart_from_a_run_not_made(
        repo, monkeypatch):
    directory, base = repo
    edit.reset_patch_round(round_limit=0)
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    monkeypatch.setattr(edit, "run_test_as_is", lambda path_list=None: _test_result())
    _fail_to_put_back(monkeypatch)
    with pytest.raises(testcreate.RestoreError, match="could not be put back.*locked"):
        testcreate.base_run_dict(directory, base, ["tests/test_y.py"], frozenset())
    assert issubclass(testcreate.RestoreError, RuntimeError)


def test_a_working_tree_that_cannot_be_put_back_leaves_the_run_open(repo, monkeypatch):
    directory, base = repo

    class _StageThatWrites(_StageThatRuns):
        def run(self, **kwargs):
            if self.stage == "test_create":
                _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
            return super().run(**kwargs)

    monkeypatch.setattr(integrate, "analyse", lambda _: {
        "no_resolve": {}, "name": {}, "range": _RANGE_DICT, "ref": [], "use": [], "signature": {}})
    monkeypatch.setattr(testcreate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    monkeypatch.setattr(edit, "run_test", lambda *a, **k: _test_result())
    monkeypatch.setattr(edit, "run_test_as_is", lambda *a, **k: _test_result())
    monkeypatch.setattr(testcreate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(testcreate, "RLM", _StageThatWrites)
    _fail_to_put_back(monkeypatch)
    out_dict = testcreate.run_test_create(base, "", "p", "out")
    assert out_dict["settled"] is False and "restore_error" not in out_dict
    assert len(out_dict["left"]) == 1 and out_dict["left"][0].startswith(
        "- after the tests were run over the code as it was before the change, the "
        "working tree could not be put back")
    assert out_dict["tests_base"] == testcreate.BASE_UNKNOWN
    assert out_dict["tests_changed"] == ["tests/test_y.py"]


def test_the_dependency_files_the_change_holds_are_laid_over_the_base_too(repo,
                                                                        monkeypatch):
    """One dependency file HEAD added against the base, and one the run has added since."""
    directory, base = repo
    _write(directory, "pyproject.toml", "[project]\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "the change adds a dependency file")
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    _write(directory, "requirements.txt", "attrs\n")
    _write(directory, "notes.txt", "no dependency file\n")
    edit.reset_patch_round(round_limit=0)
    assert testcreate._base_dep_path_list(directory, base) == ["pyproject.toml",
                                                               "requirements.txt"]
    seen_dict = {}

    def run_test(path_list=None):
        seen_dict["there"] = [name for name in ("pyproject.toml", "requirements.txt",
                                                "notes.txt", "y.py")
                              if os.path.exists(os.path.join(directory, name))]
        seen_dict["targets"] = list(path_list)
        return _test_result()

    monkeypatch.setattr(edit, "run_test_as_is", run_test)
    base_dict = testcreate.base_run_dict(directory, base, ["tests/test_y.py"], frozenset())
    assert base_dict["status"] == testcreate.BASE_GREEN
    assert seen_dict == {"there": ["pyproject.toml", "requirements.txt"],
                         "targets": ["tests/test_y.py"]}
    assert _read(directory, "requirements.txt") == "attrs\n"
    # A dependency file the change removed is laid over nothing
    os.remove(os.path.join(directory, "requirements.txt"))
    assert testcreate._base_dep_path_list(directory, base) == ["pyproject.toml"]
    monkeypatch.setattr(config, "PRIMARY_EXT", "cbl")
    assert testcreate._base_dep_path_list(directory, base) == []


def test_the_makefile_a_c_change_holds_is_laid_over_the_base_too(repo, monkeypatch):
    """It says which tests are built, so the tests written are run over the base."""
    directory, base = repo
    monkeypatch.setattr(config, "PRIMARY_EXT", "c")
    monkeypatch.setattr(edit, "other_test_list", lambda: [])
    _write(directory, "Makefile", "test:\n\t./t\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "the change registers a test")
    edit.reset_patch_round(round_limit=0)
    assert testcreate._base_dep_path_list(directory, base) == ["Makefile"]
    # One the run itself has changed, not committed
    _git(directory, "reset", "-q", "--hard", base)
    _write(directory, "Makefile", "all:\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "a makefile on the base")
    new_base = _git(directory, "rev-parse", "HEAD").strip()
    edit.reset_patch_round(round_limit=0)
    _write(directory, "Makefile", "test:\n\t./t\n")
    assert testcreate._base_dep_path_list(directory, new_base) == ["Makefile"]
    # A refill of the package cache is for a dependency file alone
    assert edit.change_dep_path_list() == []


def test_a_dependency_file_the_change_removed_is_taken_out_of_the_base(repo, monkeypatch):
    """The base holds three dependency files: HEAD moved one to another name, the run
    has removed one since, and one is as it was."""
    directory, _ = repo
    _write(directory, "pyproject.toml", "[tool.x]\n")
    _write(directory, "sub/requirements.txt", "attrs\n")
    _write(directory, "setup.py", "setup()\n")
    _write(directory, "notes.txt", "no dependency file\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "dependency files")
    base = _git(directory, "rev-parse", "HEAD").strip()
    os.remove(os.path.join(directory, "pyproject.toml"))
    os.remove(os.path.join(directory, "notes.txt"))
    _write(directory, "setup.cfg", "[x]\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "the change moves a dependency file")
    os.remove(os.path.join(directory, "sub/requirements.txt"))
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    edit.reset_patch_round(round_limit=0)
    assert testcreate._base_drop_path_list(directory, base) == ["pyproject.toml",
                                                                "sub/requirements.txt"]
    assert testcreate._base_dep_path_list(directory, base) == ["setup.cfg"]
    before = _git(directory, "status", "--porcelain")
    seen_list = []

    def run_test(path_list=None):
        seen_list.extend(name for name in ("pyproject.toml", "sub/requirements.txt",
                                           "setup.py", "setup.cfg", "notes.txt")
                         if os.path.exists(os.path.join(directory, name)))
        return _test_result()

    monkeypatch.setattr(edit, "run_test_as_is", run_test)
    base_dict = testcreate.base_run_dict(directory, base, ["tests/test_y.py"], frozenset())
    assert base_dict["status"] == testcreate.BASE_GREEN
    # A file of the base that is no dependency file stays, removed by the change or not
    assert seen_list == ["setup.py", "setup.cfg", "notes.txt"]
    assert _git(directory, "status", "--porcelain") == before
    assert before.splitlines() == [" D sub/requirements.txt", "?? tests/test_y.py"]
    assert not os.path.exists(os.path.join(directory, "pyproject.toml"))
    monkeypatch.setattr(config, "PRIMARY_EXT", "cbl")
    assert testcreate._base_drop_path_list(directory, base) == []


def test_a_run_over_the_base_that_ended_early_is_made_once_where_no_path_is_taken(
        repo, monkeypatch):
    """A test command that takes no path runs the same tests for every file."""
    directory, base = repo
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    call_list = []

    def run_test(path_list=None):
        call_list.append(list(path_list))
        return _test_result("failed", output=_EARLY_STOP_OUTPUT)

    monkeypatch.setattr(edit, "run_test_as_is", run_test)
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    path_list = ["tests/test_x.py", "tests/test_y.py"]
    assert testcreate.base_run_dict(directory, base, path_list, frozenset()) == {
        "status": testcreate.BASE_RED, "failed": [], "is_part": False,
        "left_out": ["!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection "
                     "!!!!!!!!!!!!!!!!!!!!"],
        "is_no_run": False}
    assert call_list == [path_list]


def test_a_run_over_the_base_that_ended_early_is_made_again_one_file_at_a_time(
        repo, monkeypatch):
    directory, base = repo
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    call_list = []
    answer_dict = {
        ("tests/test_x.py", "tests/test_y.py"): _test_result(
            "failed", ["tests/test_y.py"], output=_EARLY_STOP_OUTPUT),
        ("tests/test_x.py",): _test_result("failed", ["tests/test_x.py::test_f"]),
        ("tests/test_y.py",): _test_result("failed", ["tests/test_y.py"],
                                           output=_EARLY_STOP_OUTPUT)}

    def run_test(path_list=None):
        call_list.append(list(path_list))
        return answer_dict[tuple(path_list)]

    monkeypatch.setattr(edit, "run_test_as_is", run_test)
    path_list = ["tests/test_x.py", "tests/test_y.py"]
    assert testcreate.base_run_dict(directory, base, path_list, frozenset()) == {
        "status": testcreate.BASE_RED,
        "failed": ["tests/test_x.py::test_f", "tests/test_y.py"], "is_part": False, "left_out": [],
        "is_no_run": False}
    assert call_list == [path_list, ["tests/test_x.py"], ["tests/test_y.py"]]
    # One file alone is run once, however its run ended
    call_list.clear()
    assert testcreate.base_run_dict(directory, base, ["tests/test_y.py"], frozenset()) == {
        "status": testcreate.BASE_RED, "failed": ["tests/test_y.py"], "is_part": False, "left_out": [],
        "is_no_run": False}
    assert call_list == [["tests/test_y.py"]]
    # Every file passing alone, and one that does not run alone
    answer_dict[("tests/test_x.py",)] = _test_result()
    answer_dict[("tests/test_y.py",)] = _test_result()
    assert testcreate.base_run_dict(directory, base, path_list, frozenset())["status"] == \
        testcreate.BASE_GREEN
    answer_dict[("tests/test_y.py",)] = _test_result("did not run")
    assert testcreate.base_run_dict(directory, base, path_list, frozenset())["status"] == \
        testcreate.BASE_UNKNOWN


_MAX_FAIL_OUTPUT = ("FAILED tests/test_y.py::test_h - assert 0\n"
                    "!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!\n"
                    "1 failed in 0.07s")


def test_a_run_over_the_base_that_ended_at_a_failing_test_names_a_part_of_them(
        repo, monkeypatch):
    """The test command ends at the first failing test, so the tests after it are not
    named, alone and one file at a time."""
    directory, base = repo
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    answer_dict = {
        ("tests/test_x.py", "tests/test_y.py"): _test_result(
            "failed", ["tests/test_x.py::test_f"], output=_MAX_FAIL_OUTPUT),
        ("tests/test_x.py",): _test_result("failed", ["tests/test_x.py"],
                                           output=_EARLY_STOP_OUTPUT),
        ("tests/test_y.py",): _test_result("failed", ["tests/test_y.py::test_h"],
                                           output=_MAX_FAIL_OUTPUT)}
    monkeypatch.setattr(edit, "run_test_as_is",
                        lambda path_list=None: answer_dict[tuple(path_list)])
    assert testcreate.base_run_dict(directory, base, ["tests/test_y.py"], frozenset()) == {
        "status": testcreate.BASE_RED, "failed": ["tests/test_y.py::test_h"],
        "is_part": True, "left_out": [],
        "is_no_run": False}
    assert testcreate.base_run_dict(
        directory, base, ["tests/test_x.py", "tests/test_y.py"], frozenset()) == {
            "status": testcreate.BASE_RED,
            "failed": ["tests/test_x.py", "tests/test_y.py::test_h"], "is_part": True, "left_out": [],
        "is_no_run": False}
    # A file the run could not load is named alone: every test of it is failing
    assert testcreate.base_run_dict(directory, base, ["tests/test_x.py"], frozenset()) == {
        "status": testcreate.BASE_RED, "failed": ["tests/test_x.py"], "is_part": False, "left_out": [],
        "is_no_run": False}
    # The same names from a run that went on to its end
    answer_dict[("tests/test_y.py",)] = _test_result("failed", ["tests/test_y.py::test_h"])
    assert testcreate.base_run_dict(directory, base, ["tests/test_y.py"],
                                    frozenset())["is_part"] is False
    # Where no path is taken, the one run is read
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    assert testcreate.base_run_dict(
        directory, base, ["tests/test_x.py", "tests/test_y.py"], frozenset())["is_part"] \
        is True


def test_a_run_over_the_base_that_could_not_load_a_conftest_is_red_naming_its_test_files(
        repo, monkeypatch):
    """pytest could not import a conftest.py there: where every file it ran lies under
    the conftest's directory, each of them but a conftest.py is named alone, as a file it
    could not load is; where it ran a file outside that directory too, or conftest.py
    files alone, it did not run."""
    directory, base = repo
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    _write(directory, "tests/conftest.py", "from calc import mul\n")
    _write(directory, "tests/b/conftest.py", "from calc import mul\n")
    _write(directory, "tests/b/test_y.py", "def test_y():\n    pass\n")

    def fail_to_load(conftest_rel: str) -> None:
        output = (f"ImportError while loading conftest "
                  f"'{os.path.join(directory, conftest_rel)}'.\n"
                  f"{conftest_rel}:1: in <module>\n    from calc import mul\n"
                  "E   ImportError: cannot import name 'mul' from 'calc'\n")
        monkeypatch.setattr(edit, "run_test_as_is", lambda path_list=None: _test_result(
            "did not run", output=output))

    fail_to_load("tests/conftest.py")
    assert testcreate.base_run_dict(directory, base, ["tests/test_x.py"], frozenset()) == {
        "status": testcreate.BASE_RED, "failed": ["tests/test_x.py"], "is_part": False,
        "left_out": [], "is_no_run": False}
    assert testcreate.base_run_dict(
        directory, base, ["tests/b/conftest.py", "tests/b/test_y.py", "tests/conftest.py",
                          "tests/test_x.py"], frozenset())["failed"] == [
        "tests/b/test_y.py", "tests/test_x.py"]
    fail_to_load("tests/b/conftest.py")
    assert testcreate.base_run_dict(
        directory, base, ["tests/b/conftest.py", "tests/b/test_y.py"], frozenset()) == {
        "status": testcreate.BASE_RED, "failed": ["tests/b/test_y.py"], "is_part": False,
        "left_out": [], "is_no_run": False}
    assert testcreate.base_run_dict(
        directory, base, ["tests/b/conftest.py", "tests/b/test_y.py", "tests/test_x.py"],
        frozenset()) == {"status": testcreate.BASE_UNKNOWN, "failed": [], "is_part": False,
                         "left_out": [], "is_no_run": True}
    # Where it ran that conftest.py alone, it did not run
    assert testcreate.base_run_dict(directory, base, ["tests/b/conftest.py"], frozenset()) == {
        "status": testcreate.BASE_UNKNOWN, "failed": [], "is_part": False, "left_out": [],
        "is_no_run": True}
    # A run that did not run for another reason is not
    monkeypatch.setattr(edit, "run_test_as_is", lambda path_list=None: _test_result(
        "did not run", output="ModuleNotFoundError: No module named 'pytest'\n"))
    assert testcreate.base_run_dict(directory, base, ["tests/test_x.py"], frozenset()) == {
        "status": testcreate.BASE_UNKNOWN, "failed": [], "is_part": False, "left_out": [],
        "is_no_run": True}


def test_a_run_over_the_base_that_counts_failing_tests_it_names_none_of_is_not_compared(
        repo, monkeypatch):
    """A run that counts failing tests and names none of them (edit.RunResult.is_name_short)
    did not run, alone and one file at a time; another red run naming no test is red."""
    directory, base = repo
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    nameless = edit.RunResult(_test_result("failed"))
    nameless.is_name_short = True
    answer_dict = {
        ("tests/test_x.py", "tests/test_y.py"): _test_result(
            "failed", ["tests/test_y.py"], output=_EARLY_STOP_OUTPUT),
        ("tests/test_x.py",): _test_result("failed", ["tests/test_x.py::test_f"]),
        ("tests/test_y.py",): nameless}
    monkeypatch.setattr(edit, "run_test_as_is",
                        lambda path_list=None: answer_dict[tuple(path_list)])
    unknown_dict = {"status": testcreate.BASE_UNKNOWN, "failed": [], "is_part": False,
                    "left_out": [], "is_no_run": True}
    assert testcreate.base_run_dict(directory, base, ["tests/test_y.py"],
                                    frozenset()) == unknown_dict
    assert testcreate.base_run_dict(directory, base, ["tests/test_x.py", "tests/test_y.py"],
                                    frozenset()) == {**unknown_dict,
                                                     "failed": ["tests/test_x.py::test_f"]}
    nameless.is_name_short = False
    assert testcreate.base_run_dict(directory, base, ["tests/test_y.py"], frozenset()) == {
        **unknown_dict, "status": testcreate.BASE_RED, "is_no_run": False}


def test_a_run_over_the_base_that_stopped_early_naming_no_failing_test_is_not_compared(
        repo, monkeypatch):
    """A run that stopped at a count of failing tests (-x) and names none of those it
    counts did not run, alone and one file at a time."""
    directory, base = repo
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "print('!!!!! stopping after 1 failures !!!!!'); "
                              "print('1 failed in 0.01s'); exit(1)"])
    edit.reset_patch_round()
    unknown_dict = {"status": testcreate.BASE_UNKNOWN, "failed": [], "is_part": False,
                    "left_out": [], "is_no_run": True}
    for path_list in (["tests/test_y.py"], ["tests/test_x.py", "tests/test_y.py"]):
        assert testcreate.base_run_dict(directory, base, path_list, frozenset()) == unknown_dict


_SKIP_C_LINE ="[INFO] c .................................................. SKIPPED"


@pytest.mark.parametrize("ext, record, baseline_list, no_run_line_list", [
    # A module that failed its tests and one left out after it
    ("java", "java_R_multi", [], [_SKIP_C_LINE]),
    # A module that did not compile, beside one whose test fails
    ("java", "java_B_multi", [], ["[ERROR] COMPILATION ERROR :", _SKIP_C_LINE]),
    ("java", "java_B_multi", ["BTest.one", "com.example.b.BTest.one"],
     ["[ERROR] COMPILATION ERROR :", _SKIP_C_LINE]),
    ("kt", "kt_B_multi", [], ["Execution failed for task ':a:compileKotlin'."]),
    ("kt", "kt_B_multi", ["BTest > one()"], ["Execution failed for task ':a:compileKotlin'."]),
    ("kt", "kt_B_kapt", [], ["Execution failed for task ':a:kaptKotlin'."]),
])
def test_a_run_over_the_base_that_left_a_module_out_names_every_test_of_what_it_ran(
        repo, monkeypatch, ext, record, baseline_list, no_run_line_list):
    directory, base = repo
    monkeypatch.setattr(config, "PRIMARY_EXT", ext)
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    output = _record_output(record)
    result_dict = _test_result("failed", [edit._fail_name(line, ext) for line
                                          in output.split("\n") if edit._fail_name(line, ext)],
                               output=output)
    assert result_dict["failed"] and edit.early_stop_run(result_dict)
    assert edit.fail_stop_run(result_dict) is False
    monkeypatch.setattr(edit, "run_test_as_is", lambda path_list=None: result_dict)
    monkeypatch.setattr(edit, "runnable_path_list", lambda path_list: list(path_list))
    run_dict = testcreate.base_run_dict(directory, base, ["tests/test_x.py"],
                                        frozenset(baseline_list))
    assert run_dict["is_part"] is False
    assert (run_dict["left_out"], run_dict["is_no_run"]) == (no_run_line_list, False)
    # A test command that takes paths names every failing test of the files it ran
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    assert testcreate.base_run_dict(directory, base, ["tests/test_x.py"],
                                    frozenset(baseline_list))["left_out"] == []
    assert run_dict["failed"] == sorted(set(result_dict["failed"]) - set(baseline_list))


@pytest.mark.parametrize("status, output, expect_tuple", [
    ("did not run", "", (testcreate.BASE_UNKNOWN, [], True)),
    ("failed", "", (testcreate.BASE_RED, [], False)),
    ("passed", "", (testcreate.BASE_GREEN, [], False)),
])
def test_a_run_over_the_base_that_reached_no_verdict_is_told_from_one_that_did(
        repo, monkeypatch, status, output, expect_tuple):
    directory, base = repo
    result_dict = _test_result(status, [], output=output)
    monkeypatch.setattr(edit, "run_test_as_is", lambda path_list=None: result_dict)
    run_dict = testcreate.base_run_dict(directory, base, ["tests/test_x.py"], frozenset())
    assert (run_dict["status"], run_dict["left_out"], run_dict["is_no_run"]) == \
        expect_tuple


def test_a_run_that_stopped_at_tests_failing_before_the_stage_names_no_part_of_them(
        repo, monkeypatch):
    directory, base = repo
    result_dict = _test_result("failed", ["tests/test_old.py::test_h"],
                               output=_MAX_FAIL_OUTPUT)
    assert edit.fail_stop_run(result_dict) is True
    assert edit.fail_stop_run(_test_result("failed", ["tests/test_x.py"],
                                           output=_EARLY_STOP_OUTPUT)) is False
    monkeypatch.setattr(edit, "run_test_as_is", lambda path_list=None: result_dict)
    assert testcreate.base_run_dict(
        directory, base, ["tests/test_x.py"], frozenset(["tests/test_old.py::test_h"])) == {
            "status": testcreate.BASE_UNKNOWN, "failed": [], "is_part": False, "left_out": [],
        "is_no_run": False}
    assert testcreate.base_run_dict(directory, base, ["tests/test_x.py"],
                                    frozenset())["is_part"] is True


def test_a_file_the_base_tracks_and_an_ignore_rule_covers_now_is_as_it_was_afterwards(
        tmp_path, monkeypatch):
    """The change stopped tracking a file and ignores it; the file on disk and a link
    of the same kind are read as the base holds them while the tests run there."""
    directory = str(tmp_path / "p")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    _write(directory, "local.json", '{"base": 1}\n')
    _write(directory, "conf/own.ini", "base\n")
    _write(directory, "tests/test_x.py", "def test_f():\n    assert True\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "rm", "-q", "--cached", "local.json", "conf/own.ini")
    _write(directory, ".gitignore", "local.json\nconf/\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "the change stops tracking them")
    _write(directory, "local.json", '{"mine": 2}\n')
    os.remove(os.path.join(directory, "conf/own.ini"))
    os.symlink("../local.json", os.path.join(directory, "conf/own.ini"))
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    seen_dict = {}

    def run_test(path_list=None):
        seen_dict["local"] = _read(directory, "local.json")
        seen_dict["own"] = _read(directory, "conf/own.ini")
        return _test_result()

    monkeypatch.setattr(edit, "run_test_as_is", run_test)
    status = _git(directory, "status", "--porcelain", "--ignored")
    assert testcreate.base_run_dict(directory, base, ["tests/test_y.py"],
                                    frozenset())["status"] == testcreate.BASE_GREEN
    assert seen_dict == {"local": '{"base": 1}\n', "own": "base\n"}
    assert _read(directory, "local.json") == '{"mine": 2}\n'
    assert os.readlink(os.path.join(directory, "conf/own.ini")) == "../local.json"
    assert _git(directory, "status", "--porcelain", "--ignored") == status
    # The same when the run raises
    monkeypatch.setattr(edit, "run_test_as_is", _never_built)
    with pytest.raises(AssertionError):
        testcreate.base_run_dict(directory, base, ["tests/test_y.py"], frozenset())
    assert _read(directory, "local.json") == '{"mine": 2}\n'
    assert os.path.islink(os.path.join(directory, "conf/own.ini"))


def _ignored_now_repo(tmp_path, monkeypatch, base_file_dict, ignore_text):
    """A repository whose base tracks base_file_dict and whose change stops tracking
    those files, takes them off the disk and ignores them, set up for a run of pytest
    files.

    Returns:
        (path, the base commit).
    """
    directory = str(tmp_path / "p")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    for name, text in {**base_file_dict,
                       "tests/test_x.py": "def test_f():\n    assert True\n"}.items():
        _write(directory, name, text)
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "rm", "-q", "--", *base_file_dict)
    _write(directory, ".gitignore", ignore_text)
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "the change stops tracking them")
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    return directory, base


@pytest.mark.parametrize("kind", ["file", "link"])
def test_what_stands_where_the_base_holds_a_directory_is_as_it_was_afterwards(
        tmp_path, monkeypatch, kind):
    """The base tracks conf/default.json; now conf is a file, or a link to a directory
    outside the repository, that an ignore rule covers."""
    directory, base = _ignored_now_repo(tmp_path, monkeypatch,
                                        {"conf/default.json": '{"base": 1}\n'}, "/conf\n")
    outside = tmp_path / "shared"
    outside.mkdir()
    (outside / "mine.json").write_text("{}\n", encoding="utf-8")
    conf = os.path.join(directory, "conf")
    if kind == "file":
        _write(directory, "conf", "notes kept in a file\n")
    else:
        os.symlink(str(outside), conf)
    seen_dict = {}

    def run_test(path_list=None):
        seen_dict["default"] = _read(directory, "conf/default.json")
        seen_dict["is_link"] = os.path.islink(conf)
        # What the run leaves under the directory goes when the path is put back
        _write(directory, "conf/made_by_the_run.txt", "x\n")
        return _test_result()

    monkeypatch.setattr(edit, "run_test_as_is", run_test)
    status = _git(directory, "status", "--porcelain", "--ignored")
    assert testcreate.base_run_dict(directory, base, ["tests/test_x.py"],
                                    frozenset())["status"] == testcreate.BASE_GREEN
    assert seen_dict == {"default": '{"base": 1}\n', "is_link": False}
    if kind == "file":
        assert _read(directory, "conf") == "notes kept in a file\n"
    else:
        assert os.readlink(conf) == str(outside)
    assert sorted(os.listdir(outside)) == ["mine.json"]
    assert _git(directory, "status", "--porcelain", "--ignored") == status


@pytest.mark.parametrize("kind", ["fifo", "fifo above", "unreadable"])
def test_the_tests_are_not_run_over_the_base_where_what_is_in_its_way_cannot_be_kept(
        tmp_path, monkeypatch, kind):
    """What stands at a path the base tracks, or above one, is a pipe, or a file this
    process cannot read."""
    directory, base = _ignored_now_repo(
        tmp_path, monkeypatch,
        {"run.sock": "placeholder\n", "conf/default.json": "{}\n", "local.json": "{}\n"},
        "run.sock\n/conf\nlocal.json\n")
    path = os.path.join(directory, {"fifo": "run.sock", "fifo above": "conf",
                                    "unreadable": "local.json"}[kind])
    if kind == "unreadable":
        _write(directory, "local.json", '{"mine": 2}\n')
        os.chmod(path, 0)
        if os.access(path, os.R_OK):
            pytest.skip("this process reads a file whatever its mode says")
    else:
        os.mkfifo(path)
    monkeypatch.setattr(edit, "run_test_as_is", _never_built)
    status = _git(directory, "status", "--porcelain", "--ignored")
    mode = os.lstat(path).st_mode
    assert testcreate.base_run_dict(directory, base, ["tests/test_x.py"], frozenset()) == {
        "status": testcreate.BASE_UNKNOWN, "failed": [], "is_part": False, "left_out": [],
        "is_no_run": True}
    assert os.lstat(path).st_mode == mode
    assert not os.path.exists(os.path.join(directory, "conf", "default.json"))
    assert _git(directory, "status", "--porcelain", "--ignored") == status
    with pytest.raises(integrate.KeepError, match="could not be kept aside"):
        with integrate.keep_file(directory, base, "HEAD"):
            raise AssertionError("the block was entered")
    os.chmod(path, 0o600)


def test_a_test_file_whose_name_reads_as_a_pattern_lays_itself_over_the_base_alone(
        repo, monkeypatch):
    directory, base = repo
    base_text = _git(directory, "show", f"{base}:tests/test_x.py")
    _write(directory, "tests/test_x.py", base_text + "# changed by the stage\n")
    _write(directory, "tests/test_[x].py", "def test_h():\n    assert True\n")
    seen_dict = {}

    def run_test(path_list=None):
        seen_dict["test_x"] = _read(directory, "tests/test_x.py")
        return _test_result()

    monkeypatch.setattr(edit, "run_test_as_is", run_test)
    testcreate.base_run_dict(directory, base, ["tests/test_[x].py"], frozenset())
    assert seen_dict == {"test_x": base_text}
    assert _read(directory, "tests/test_x.py").endswith("# changed by the stage\n")


def test_tests_that_pass_over_the_base_are_green_and_no_test_is_no_run(repo, monkeypatch):
    directory, base = repo
    monkeypatch.setattr(edit, "run_test_as_is", lambda path_list=None: _test_result())
    assert testcreate.base_run_dict(directory, base, ["tests/test_x.py"],
                                    frozenset())["status"] == testcreate.BASE_GREEN
    assert testcreate.base_run_dict(directory, base, [], frozenset()) == \
        {"status": "", "failed": [], "is_part": False, "left_out": [],
        "is_no_run": False}


def test_a_run_over_the_base_that_cannot_be_made_is_unknown_and_not_an_error(repo,
                                                                            monkeypatch):
    directory, base = repo
    edit.reset_patch_round(round_limit=0)
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")

    def broken(*args, **kwargs):
        raise RuntimeError("could not write the working tree to a tree: boom")

    monkeypatch.setattr(testcreate, "base_run_dict", broken)
    change_dict = testcreate._own_change_dict(directory, base, frozenset())
    assert change_dict["tests_base"] == testcreate.BASE_UNKNOWN
    assert change_dict["restore_error"] == ""
    assert change_dict["base_failed"] == [] and change_dict["tests_changed"] == [
        "tests/test_y.py"]


def test_only_the_files_the_test_command_runs_are_run_over_the_base(repo, monkeypatch):
    directory, base = repo
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    _write(directory, "tests/data/pairs.json", "[]\n")
    seen_dict = {}

    def run_test(path_list=None):
        seen_dict["targets"] = list(path_list)
        seen_dict["data"] = os.path.exists(os.path.join(directory, "tests/data/pairs.json"))
        return _test_result()

    monkeypatch.setattr(edit, "run_test_as_is", run_test)
    base_dict = testcreate.base_run_dict(
        directory, base, ["tests/data/pairs.json", "tests/test_y.py"], frozenset())
    assert base_dict["status"] == testcreate.BASE_GREEN
    assert seen_dict == {"targets": ["tests/test_y.py"], "data": True}
    seen_dict.clear()
    assert testcreate.base_run_dict(directory, base, ["tests/data/pairs.json"],
                                    frozenset()) == {"status": "", "failed": [],
                                                     "is_part": False, "left_out": [],
        "is_no_run": False}
    assert seen_dict == {}


def test_the_run_over_the_base_asks_for_no_refill_and_leaves_the_stage_runs_alone(
        repo, monkeypatch):
    """The base's dependency file differs from HEAD's, which a run that asks for refills
    reads as a change."""
    directory, _ = repo
    _write(directory, "pyproject.toml", "[project]\ndependencies = []\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "manifest")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _write(directory, "pyproject.toml", "[project]\ndependencies = ['attrs']\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "dependency")
    _write(directory, "tests/test_y.py", "def test_h():\n    assert True\n")
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c",
        "print('=== short test summary info ==='); print('FAILED tests/test_y.py::test_h - x');"
        " print('1 failed in 0.01s'); exit(1)"])
    ask_list = []
    monkeypatch.setattr(edit.broker, "refill", lambda: ask_list.append(1) or {
        "ok": True, "skip": False, "refills_left": 0})
    edit.reset_patch_round()
    assert edit.change_dep_path_list() == []
    base_dict = testcreate.base_run_dict(directory, base, ["tests/test_y.py"], frozenset())
    assert base_dict["status"] == testcreate.BASE_RED
    assert ask_list == [] and edit.last_test_ok() is None and edit.patch_round_count() == 0


# == The instructions ========================================


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_the_instructions_name_only_tools_the_stage_is_handed(ext, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    text = testcreate.build_test_create_instruction(ext)
    hand_set = {tool.__name__ for tool in testcreate.stage_tool_list()}
    assert "run_test" in hand_set
    for name in set(agent.TOOL_RESULT_KEY_DICT) - hand_set:
        assert name not in text


def test_the_instructions_name_the_start_block_for_the_tests_that_were_failing(repo):
    text = " ".join(testcreate.build_test_create_instruction("py").split())
    assert 'leave the tests "## At the start" lists as already failing alone.' in text
    assert "the state lists" not in text


def test_the_test_directory_is_named_only_where_the_repository_has_one(repo, monkeypatch):
    directory, _ = repo
    assert "live under `tests`" in testcreate.build_test_create_instruction("py")
    monkeypatch.setattr(config, "TEST_PATH", "spec")
    assert "live under" not in testcreate.build_test_create_instruction("py")


@pytest.mark.parametrize("test_path", [".", ""])
def test_the_repository_root_is_not_named_as_where_the_tests_live(repo, monkeypatch,
                                                                 test_path):
    monkeypatch.setattr(config, "TEST_PATH", test_path)
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    assert edit.has_test_path() is True and edit.has_test_dir() is False
    assert "live under" not in testcreate.build_test_create_instruction("py")
    note = agent.test_note()
    assert "run_test() runs" in note and "live" not in note
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    assert edit.has_test_dir() is True and "under `tests`" in " ".join(
        agent.test_note().split())


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_the_dependency_file_is_named_only_where_the_language_has_one(ext, repo):
    text = testcreate.build_test_create_instruction(ext)
    dep_file_tuple = LANG_DICT[ext].dep_file_tuple
    if dep_file_tuple:
        example = " or ".join([dep_file_example(dep_file_tuple[0])]
                              + list(LANG_DICT[ext].test_build_file_tuple[-1:]))
        assert f"a dependency file (such as {example})" in text
        assert "*" not in example
    else:
        assert "dependency file" not in text
    # The refusal names the dependency files only where the run writes them
    assert 'work["dep_files"]' in testcreate.not_own_error(True)
    assert "dep_files" not in testcreate.not_own_error()
    assert "{path}" in testcreate.not_own_error(True)


@pytest.mark.parametrize("refusal_max, text", [
    (0, "is recorded, and the call is accepted as it is."), (1, "sends it back once."),
    (3, "sends it back, up to 3 times.")])
def test_the_sentence_on_the_check_ends_by_how_often_it_is_sent_back(refusal_max, text):
    assert testcreate.refusal_text(refusal_max) == text


def test_the_state_holds_the_test_names_of_a_baseline_of_several_commands(repo):
    directory, base = repo
    baseline = _baseline(["t::old"], py="red before", js="not compared")
    work_dict = testcreate.build_work_dict(
        directory, base, "change f", {"referenced": [], "unreferenced": []}, baseline,
        testcreate.branch_list_of(directory, base), [], [])
    assert work_dict["already_failing"] == ["t::old"]
    assert "\x1f" not in repr(work_dict)


def test_the_state_names_what_was_written_and_what_else_changed(repo, monkeypatch):
    directory, base = repo
    monkeypatch.setattr(config, "TEST_CREATE_MAX_TURN", 5)
    work_dict = testcreate.build_work_dict(
        directory, base, "change f",
        {"referenced": [], "unreferenced": [{"file": "y.py", "name": "h"}]},
        frozenset({"t::old"}), testcreate.branch_list_of(directory, base),
        [{"id": "case_1", "file": "y.py", "name": "h", "input": "h()",
          "expect": "returns 3", "from": "add h"},
         {"id": "case_2", "file": "y.py", "name": "h", "input": "h(1)",
          "expect": "raises TypeError", "from": "add h"}], ["tests/test_x.py"])
    integrate.reset_stage()
    edit.reset_patch_round(round_limit=0)
    _write(directory, "tests/test_y.py", "def test_h():  # case_1\n    assert True\n")
    _write(directory, "y.py", "def h():\n    return 4\n")
    turn_note, check_and_keep = testcreate.make_state(
        directory, work_dict, lambda: ["- something"], frozenset({"t::old"}),
        _RANGE_DICT, "p")
    text = turn_note(1, 5)
    assert [entry["path"] for entry in work_dict["written"]] == ["tests/test_y.py"]
    assert [entry["path"] for entry in work_dict["also_changed"]] == ["y.py"]
    # What each turn measures again is in the state, by the id of the case
    assert text.startswith(
        "## State (turn 1/5)\n\n"
        "Where each case is, among the test files you have changed, as of this turn:\n"
        "  case_1 h in y.py [tests/test_y.py]\n"
        "  case_2 h in y.py [in no test yet]\n")
    assert "returns 3" not in text and "t::old" not in text
    assert "tests/test_x.py" not in text
    assert work_dict["cases"][0]["in"] == ["tests/test_y.py"]
    assert work_dict["test_files"] == ["tests/test_x.py"]
    # What was measured before the first turn is in the block after the request
    work_dict.update({"own_files": ["y.py"], "dep_files": ["pyproject.toml"]})
    start = testcreate.start_text(work_dict)
    assert start.startswith("## At the start\n\nThe change, written against ")
    assert "  case_1 h in y.py — h() → returns 3\n" in start
    assert "  case_2 h in y.py — h(1) → raises TypeError\n" in start
    assert "in no test yet" not in start and "as of this turn" not in start
    assert "The repository's test files, as they were when you started:\n  tests/test_x.py" \
        in start
    assert "  y.py\n  pyproject.toml  (dependency file)\n" in start
    assert start.endswith("they are not yours to change:\n  t::old")
    assert check_and_keep() == ["- something"] and work_dict["checked_at"] == 1
    assert "- something" in turn_note(2, 5)
    # The start block opens with the head and the state closes, before the deadline
    # note, with the tail the stages that refactor render too
    text = turn_note(3, 5)
    head_list = testcreate.change_head_line_list(work_dict)
    tail_list = testcreate.run_tail_line_list(work_dict)
    assert start.startswith("\n".join(head_list)) and "\n".join(tail_list) in text
    assert head_list[0] == "## At the start" and head_list[-1].startswith("  asked: ")
    assert tail_list[-1] == testcreate._NO_DECISION and "  - something" in tail_list
    # The call made after the last turn reads the block under the last turn's heading
    late = turn_note(6, 5)
    assert late.startswith("## State (turn 5/5)") and "turns left" not in late
    assert integrate.stage_turn() == 5


def _state_of(directory, base, monkeypatch, case_list):
    """Build the second stage's state over the fixture, with one case list."""
    monkeypatch.setattr(config, "TEST_CREATE_MAX_TURN", 9)
    work_dict = testcreate.build_work_dict(
        directory, base, "change f", {"referenced": [], "unreferenced": []}, frozenset(),
        [], case_list, ["tests/test_x.py"])
    integrate.reset_stage()
    edit.reset_patch_round(round_limit=0)
    turn_note, _ = testcreate.make_state(directory, work_dict, lambda: [], frozenset(),
                                         _RANGE_DICT, "p")
    return turn_note


def _whole_run(monkeypatch, n, result_dict):
    """Have edit.last_whole_run() say the whole suite was run n times, ending this way,
    the last run over what the stage has changed now."""
    run_dict = {"n": n, "result": result_dict, "change": edit.stage_change_hash_dict()}
    monkeypatch.setattr(edit, "last_whole_run", lambda: run_dict)


def test_nothing_left_is_said_once_every_case_is_placed_and_the_suite_ran_after_the_change(
        repo, monkeypatch):
    directory, base = repo
    case = {"id": "case_1", "file": "y.py", "name": "h", "input": "h()",
            "expect": "returns 3", "from": "add h"}
    turn_note = _state_of(directory, base, monkeypatch, [case])
    # Nothing written and no whole run yet
    assert testcreate._NOTHING_LEFT not in turn_note(1, 9)
    # The test is written, and the suite has not been run since
    _write(directory, "tests/test_y.py", "def test_h():  # case_1\n    assert True\n")
    assert testcreate._NOTHING_LEFT not in turn_note(2, 9)
    # The whole suite passes over what is written
    _whole_run(monkeypatch, 1, _test_result())
    assert testcreate._NOTHING_LEFT in turn_note(3, 9)
    assert testcreate._NOTHING_LEFT in turn_note(4, 9)
    # The call made after the last turn is not told to end the stage
    assert testcreate._NOTHING_LEFT not in turn_note(10, 9)
    # A change after that run takes the line away until the suite is run again
    _write(directory, "tests/test_y.py", "def test_h():  # case_1\n    assert 1\n    pass\n")
    assert testcreate._NOTHING_LEFT not in turn_note(5, 9)
    _whole_run(monkeypatch, 2, _test_result("failed", ["t::new"], "boom"))
    assert testcreate._NOTHING_LEFT not in turn_note(6, 9)
    _whole_run(monkeypatch, 3, _test_result())
    assert testcreate._NOTHING_LEFT in turn_note(7, 9)


def test_nothing_left_is_not_said_after_a_change_made_in_the_turn_of_the_run(repo,
                                                                          monkeypatch):
    directory, base = repo
    case = {"id": "case_1", "file": "y.py", "name": "h", "input": "h()",
            "expect": "returns 3", "from": "add h"}
    turn_note = _state_of(directory, base, monkeypatch, [case])
    _write(directory, "tests/test_y.py", "def test_h():  # case_1\n    assert True\n")
    turn_note(1, 9)
    # The suite runs over that, and the same turn changes the test afterwards
    _whole_run(monkeypatch, 1, _test_result())
    _write(directory, "tests/test_y.py", "def test_h():  # case_1\n    assert 1\n")
    assert testcreate._NOTHING_LEFT not in turn_note(2, 9)
    assert testcreate._NOTHING_LEFT not in turn_note(3, 9)


def test_nothing_left_is_not_said_while_a_case_is_in_no_test_and_no_decision(repo,
                                                                            monkeypatch):
    directory, base = repo
    case_list = [{"id": f"case_{n}", "file": "y.py", "name": "h", "input": "h()",
                  "expect": "returns 3", "from": "add h"} for n in (1, 2)]
    turn_note = _state_of(directory, base, monkeypatch, case_list)
    _write(directory, "tests/test_y.py", "def test_h():  # case_1\n    assert True\n")
    turn_note(1, 9)
    _whole_run(monkeypatch, 1, _test_result())
    assert testcreate._NOTHING_LEFT not in turn_note(2, 9)
    integrate.record_decision("case_2", "h takes no argument")
    assert testcreate._NOTHING_LEFT in turn_note(3, 9)


def _state_and_check_of(directory, base, monkeypatch, check, refuse_list=None):
    """Build the second stage's state and check over the fixture, with one case whose
    test is written and a whole run over it that passed."""
    monkeypatch.setattr(config, "TEST_CREATE_MAX_TURN", 9)
    case = {"id": "case_1", "file": "y.py", "name": "h", "input": "h()",
            "expect": "returns 3", "from": "add h"}
    work_dict = testcreate.build_work_dict(
        directory, base, "change f", {"referenced": [], "unreferenced": []}, frozenset(),
        [], [case], ["tests/test_x.py"])
    integrate.reset_stage()
    edit.reset_patch_round(round_limit=0)
    turn_note, check_and_keep = testcreate.make_state(
        directory, work_dict, check, frozenset(), _RANGE_DICT, "p", refuse_list)
    _write(directory, "tests/test_y.py", "def test_h():  # case_1\n    assert True\n")
    turn_note(1, 9)
    _whole_run(monkeypatch, 1, _test_result())
    return turn_note, check_and_keep


def test_nothing_left_is_not_said_while_what_the_check_found_stands_unchanged(repo,
                                                                             monkeypatch):
    directory, base = repo
    reason = "- these calls resolved before you started and no longer do: h in y.py"
    turn_note, check_and_keep = _state_and_check_of(directory, base, monkeypatch,
                                                    lambda: [reason])
    assert testcreate._NOTHING_LEFT in turn_note(2, 9)
    assert check_and_keep() == [reason]
    note = turn_note(3, 9)
    assert reason in note and testcreate._NOTHING_LEFT not in note
    assert testcreate._NOTHING_LEFT not in turn_note(4, 9)
    # A change, and a whole run over it, is something the check has not seen
    _write(directory, "tests/test_y.py", "def test_h():  # case_1\n    assert 1\n")
    _whole_run(monkeypatch, 2, _test_result())
    assert testcreate._NOTHING_LEFT in turn_note(5, 9)


def test_nothing_left_is_said_after_a_check_that_found_nothing(repo, monkeypatch):
    directory, base = repo
    turn_note, check_and_keep = _state_and_check_of(directory, base, monkeypatch,
                                                    lambda: [])
    assert check_and_keep() == []
    assert testcreate._NOTHING_LEFT in turn_note(2, 9)


def test_nothing_left_is_not_said_once_a_write_outside_the_tests_was_refused(repo,
                                                                            monkeypatch):
    directory, base = repo
    refuse_list = []
    turn_note, _ = _state_and_check_of(directory, base, monkeypatch, lambda: [],
                                       refuse_list)
    refuse_list.append("tests/test_other.py")
    assert testcreate._NOTHING_LEFT in turn_note(2, 9)
    refuse_list.append("z.py")
    assert testcreate._NOTHING_LEFT not in turn_note(3, 9)


@pytest.mark.parametrize("build", [testcreate.build_test_case_instruction,
                                   testcreate.build_test_create_instruction])
def test_both_stages_are_told_that_the_repl_does_not_reach_the_repository(build, repo):
    assert f"- {rlm.REPL_REACH_NOTE}" in build("py")
    assert f"- {rlm.REPL_REACH_NOTE}" in agent.build_instruction("py")


# == Cases the change does not do =============================


def test_a_case_named_by_record_gap_is_placed_and_kept_apart_from_the_decisions(
        repo, monkeypatch):
    directory, base = repo
    case_list = [{"id": f"case_{n}", "file": "y.py", "name": "h", "input": "h()",
                  "expect": "returns 3", "from": "add h"} for n in (1, 2)]
    turn_note = _state_of(directory, base, monkeypatch, case_list)
    _write(directory, "tests/test_y.py", "def test_h():  # case_1\n    assert True\n")
    integrate.record_gap("case_2", "y.py never returns 3")
    _whole_run(monkeypatch, 1, _test_result())
    turn_note(1, 9)
    note = turn_note(2, 9)
    assert testcreate._CASE_IN_GAP in note
    assert "case_2 — y.py never returns 3" in note
    assert testcreate._NOTHING_LEFT in note
    assert integrate.decision_list() == []
    assert testcreate.decision_id_set(case_list) == {"case_2"}


def test_the_second_stage_is_handed_record_gap_and_told_when_to_call_it():
    assert integrate.record_gap in testcreate.test_create_tool_list()
    assert integrate.record_gap not in testcreate.stage_tool_list()
    assert 'record_gap("<the id>"' in testcreate.build_test_create_instruction("py")


def test_a_file_where_the_tools_write_is_no_target(repo, monkeypatch):
    directory, base = repo
    monkeypatch.setattr(testcreate.edit, "build_output_path_list",
                        lambda path_list: [path for path in path_list if path == "y.py"])
    assert testcreate.change_def_dict(directory, base, _RANGE_DICT, "p") == {"x.py": ["f"]}


# == the language the stages after the merge take ==============
_ROOT_TEST_LIST = [(ext, ["t"], "") for ext in ("c", "java", "js", "kt", "rs")]
_ALL_SET = {"c", "java", "js", "rs"}


@pytest.mark.parametrize("primary, path_list, root_list, new_set, expect", [
    ("py", [], _ROOT_TEST_LIST, _ALL_SET, ""),
    ("py", ["calc/basic.py", "src/lib.rs"], _ROOT_TEST_LIST, _ALL_SET, ""),
    ("py", ["src/lib.rs"], _ROOT_TEST_LIST, {"rs"}, "rs"),
    ("py", ["src/lib.rs"], _ROOT_TEST_LIST, set(), ""),
    ("py", ["static/app.js"], _ROOT_TEST_LIST, set(), ""),
    ("cbl", ["src/main/java/calc/CalcOps.java", "src/CALCOPS.cbl"], _ROOT_TEST_LIST,
     {"java"}, "java"),
    ("cbl", ["src/main/java/calc/CalcOps.java"], _ROOT_TEST_LIST, set(), ""),
    ("cbl", ["db/schema.sql"], _ROOT_TEST_LIST, set(), ""),
    ("py", ["b/Ops.java", "a/lib.rs"], _ROOT_TEST_LIST, _ALL_SET, "java"),
    ("py", ["b/Ops.java", "a/lib.rs"], _ROOT_TEST_LIST, {"rs"}, "rs"),
    ("py", ["a/x.rs", "b/y.rs", "c/Z.java"], _ROOT_TEST_LIST, _ALL_SET, "rs"),
    ("py", ["web/view.tsx", "web/app.mjs"], _ROOT_TEST_LIST, _ALL_SET, "js"),
    ("js", ["web/view.ts"], _ROOT_TEST_LIST, _ALL_SET, ""),
    ("py", ["src/a.c", "src/a.h", "src/b.h"], _ROOT_TEST_LIST, _ALL_SET, "c"),
    ("py", ["src/a.c"], [], _ALL_SET, ""),
    ("py", ["native/src/lib.rs"], [("rs", ["cargo", "test"], "native")], _ALL_SET, ""),
    ("py", ["web/view.tsx"], [("js", ["npm", "test"], "web")], _ALL_SET, ""),
    ("py", ["native/a.cpp"], [("c", ["t"], "native")], _ALL_SET, ""),
    ("py", ["build.gradle.kts", "src/main/java/A.java"], _ROOT_TEST_LIST, {"kt"}, ""),
    ("py", ["build.gradle.kts", "src/main/kotlin/A.kt"], _ROOT_TEST_LIST, {"kt"}, "kt"),
])
def test_the_change_s_language_is_taken_where_the_primary_one_tests_none_of_it(
        monkeypatch, primary, path_list, root_list, new_set, expect):
    for name in ("TEST_COMMAND", "TEST_PATH", "TEST_TARGET_STYLE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "PRIMARY_EXT", primary)
    monkeypatch.setattr(config, "TEST_COMMAND", config.test_command_of(primary))
    monkeypatch.setattr(testcreate.integrate, "analyse", lambda path: {"range": {}})
    monkeypatch.setattr(testcreate, "change_def_dict",
                        lambda project_dir, base, range_dict, project_name:
                        {path: ["f"] for path in path_list})
    monkeypatch.setattr(testcreate.edit, "other_test_list", lambda: root_list)
    monkeypatch.setattr(testcreate, "_root_dep_lang_set", lambda base: new_set)
    assert testcreate.change_lang_ext("b" * 40, "k.sqlite", "repo") == expect


def test_the_languages_of_the_dependency_files_head_adds_at_the_root(repo):
    directory, _ = repo
    head = _git(directory, "rev-parse", "HEAD").strip()
    _write(directory, "Cargo.toml", "[package]\n")
    _write(directory, "sub/pom.xml", "<project/>\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "add")
    lang_set = testcreate._root_dep_lang_set(head)
    assert "rs" in lang_set and "java" not in lang_set
    assert testcreate._root_dep_lang_set("HEAD") == set()
    head = _git(directory, "rev-parse", "HEAD").strip()
    _write(directory, "Cargo.lock", "version = 3\n")
    _write(directory, "App.sln", "\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "add more")
    lang_set = testcreate._root_dep_lang_set(head)
    assert "rs" not in lang_set and "cs" in lang_set


@pytest.mark.parametrize("path_list, why", [
    (["calc/basic.py", "src/lib.rs"], "the change touches files its test command runs"),
    (["db/schema.sql"], "no changed file is of one of them whose tests run from there")])
def test_a_dependency_file_added_without_a_change_of_language_says_so(
        monkeypatch, caplog, path_list, why):
    for name in ("TEST_COMMAND", "TEST_PATH", "TEST_TARGET_STYLE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_COMMAND", config.test_command_of("py"))
    monkeypatch.setattr(testcreate.integrate, "analyse", lambda path: {"range": {}})
    monkeypatch.setattr(testcreate, "change_def_dict",
                        lambda *a: {path: ["f"] for path in path_list})
    monkeypatch.setattr(testcreate.edit, "other_test_list", lambda: _ROOT_TEST_LIST)
    monkeypatch.setattr(testcreate, "_root_dep_lang_set", lambda base: {"rs"})
    with caplog.at_level(logging.INFO, logger="codetwine_loom.testcreate"):
        assert testcreate.change_lang_ext("b" * 40, "k.sqlite", "repo") == ""
    assert why in caplog.text


def test_a_dependency_file_of_the_primary_language_s_test_command_is_passed_over(
        monkeypatch, caplog):
    for name in ("TEST_COMMAND", "TEST_PATH", "TEST_TARGET_STYLE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_COMMAND", config.test_command_of("py"))
    monkeypatch.setattr(testcreate, "_root_dep_lang_set", lambda base: {"py"})
    monkeypatch.setattr(testcreate, "change_def_dict",
                        lambda *a: pytest.fail("the change was read"))
    with caplog.at_level(logging.INFO, logger="codetwine_loom.testcreate"):
        assert testcreate.change_lang_ext("b" * 40, "k.sqlite", "repo") == ""
    assert caplog.text == ""


@pytest.mark.parametrize("name, value", [
    ("TEST_COMMAND", "make test"), ("TEST_PATH", "test"), ("TEST_TARGET_STYLE", "none")])
def test_a_test_setting_set_by_hand_keeps_the_primary_language(monkeypatch, caplog, name,
                                                               value):
    for other in ("TEST_COMMAND", "TEST_PATH", "TEST_TARGET_STYLE"):
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(testcreate, "_root_dep_lang_set", lambda base: {"java"})
    monkeypatch.setattr(testcreate, "change_def_dict",
                        lambda *a: pytest.fail("the change was read"))
    with caplog.at_level(logging.INFO, logger="codetwine_loom.testcreate"):
        assert testcreate.change_lang_ext("b" * 40, "k.sqlite", "repo") == ""
    assert f"the language of the change is not looked at as {name} is set" in caplog.text
