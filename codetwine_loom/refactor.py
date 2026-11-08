"""Refactor what a combined change touched, and the tests of that change.

The run is over a repository whose HEAD is the combined change. There are two stages
(STAGE_TUPLE: integrate.REFACTOR_STAGE and integrate.TEST_REFACTOR_STAGE).

The first refactors the files the change touched outside the tests. It writes to those
files alone. The second refactors the test files the change added or changed. It writes
to those and to new test files (paths.under_test_path()). With runnable_only, either
takes only the files the test command runs.

What a stage leaves is checked without a model. For both: no file that was there when
the stage started and that the stage does not write holds anything else, what a command
the machine ran wrote there aside, the static check finds nothing it did not find when
the stage started, the references that resolved at HEAD still resolve, and no test fails
that did not fail before the stage started. For the second as well: no test file it
started with is gone, every case id the test files held is still in one of them, the
test files it changed that the test command runs pass when run alone where the build
tool takes paths, the whole suite runs each of them that holds a case, and those test
files say over the tree of the commit the change was written against what they said
there before the stage started.

A stage with nothing to refactor, a repository whose tests cannot be run and a test
suite that could not be read before the stage started call no model. Neither does the
second stage when none of its files' tests ran, when the names its tests fail with over
the tree of the commit the change was written against are of a part of them, or when,
with their status there not testcreate.BASE_RED, they reach no verdict there or count
failing tests there and name none of them. A stage that changes nothing is settled and
leaves nothing to commit. A stage whose run stopped on a call that did not happen is not
settled, whatever the check finds.

This module imports nothing from agent.py.
"""

import collections
import functools
import logging
import os
from collections.abc import Callable

from codetwine_loom import config, gitcmd, guide, integrate, testcreate
from codetwine_loom.languages import LANG_NAME_DICT, PATH_EXAMPLE_DICT, edit_paragraph
from codetwine_loom.llm import make_llm
from codetwine_loom.paths import list_text, under_test_path
from codetwine_loom.rlm import (COMPLETE_KEY, OUTPUT_LIMIT_NOTE, REPL_REACH_NOTE, RLM,
                                make_deadline_note, output_limit_note, run_stage)
from codetwine_loom.tools import edit

logger = logging.getLogger(__name__)

# The two stages, by name
STAGE_TUPLE = (integrate.REFACTOR_STAGE, integrate.TEST_REFACTOR_STAGE)

_REFACTOR_INSTRUCTION_TEMPLATE = """{opening}

Each turn you write ONE ```python block. It runs, you see what it printed, then you write
the next block. print() what you want to see{output_limit}. Names you define stay \
available in later turns. Stop at the closing
fence — anything you write after it is thrown away and costs you the turn.

- {repl_reach_note}

Your earlier turns stay in the prompt. After the request comes an "## At the start"
block: what the machine measured before the first turn, which does not change. Before
every turn you are also given a "## State" block that the machine rebuilds from the
repository. record_decision(what, why) puts a line in the state, where it is read again
after this run.

{rule}

Where nothing needs changing, change nothing and call COMPLETE() saying so. A run that
leaves every file as it found it has ended well.

This repository's own instructions below, where there are any, say more about what to
refactor here.

run_test() runs `{test_command}` from the repository root{test_path_note}.
{test_target_note} Run the tests after each change, and the whole suite before you
finish. revert_file(path) puts a file back to what it held when you started.

Every path you pass to a tool is relative to the repository root, e.g. "{path_example}".
{lang_note}
When you are done, call COMPLETE("<what you changed, or that nothing needed changing>")
and nothing else. Write it in the language work["request"] is written in. It is checked
before it is accepted. {check} {refusal_text}{guide_note}"""

# What differs between the two stages' instructions: how they open, what they are told
# to look for and to leave alone, and what the check on COMPLETE() finds
_STAGE_TEXT_DICT = {
    integrate.REFACTOR_STAGE: {
        "opening": """You are refactoring a change that was just made to a {lang_name}
repository. The change is already in the working tree and does what was asked. What is
left is to leave the code it wrote easier to read and to keep, without changing anything
that code does.""",
        "rule": """work["targets"] lists the files the change touched and, for each, the
definitions it touched there. Those files are all you can write to: a write to any other
path, a test and a new file among them, is refused. Look at those definitions and at
nothing else:

- **Code the change wrote that does not read like the code around it**: naming, layout,
  error handling, how the neighbouring code does the same thing. Read the neighbours
  before you decide.
- **The same thing written more than once by the change.** work["branches"] lists the
  tasks the request was split into. They were done at the same time, none seeing what
  the others wrote, so two of them can each have left their own copy of a helper, a
  constant or a check. Keep one and make the others use it.
- **What the change left that nothing uses**: imports, variables, helpers, code that is
  commented out.

What stays as it is:

- **What the code does.** Every input gives the value, the error and the effect it gave
  before you started.
- **Every name, signature and return shape another file or a test uses.** Read who uses
  a definition with get_use_of_file() or grep_source() before you rename or move it.
- **Every test.** A test that fails after a change of yours says the change did more
  than refactor: put the file back with revert_file() and leave that part alone.
- **Lines the change did not touch**, except where one has to change for a copy to go.
  A new feature, a bug fix and reformatting are not this run's work: where you notice
  one, record_decision("<where>", "<what you noticed>") and leave it.""",
        "check": ("A file that is not this run's to write and holds something else than "
                  "when you started, a call that no longer resolves, or a test that "
                  "fails and did not fail before you started"),
        "request": "Refactor the change made for this request: {request}",
        "no_request": "Refactor the change HEAD holds.",
        "target_title": ("The files you can write to, each with the definitions the change "
                         "touched there. Every other path is refused:"),
        "deadline": ("Make the tests pass, revert what does not, and call COMPLETE(). A "
                     "file left as it was is better than a file left half changed."),
    },
    integrate.TEST_REFACTOR_STAGE: {
        "opening": """You are refactoring the tests of a change that was just made to a
{lang_name} repository. The change and its tests are already in the working tree. What
is left is to leave those tests easier to read and to keep, without changing anything
they check.""",
        "rule": """work["targets"] lists the test files the change added or changed. Those
and new files under the tests' directory are all you can write to: a write to any other
path is refused. In a file that sets how the tests are found and run, leave which tests
it finds, which of them run and how their outcome is read as they are. Look at the tests
the change wrote in those files and at nothing else:

- **Tests written unlike this repository's tests**: the framework, the fixtures, the
  helpers, the naming, the layout. Read a test file the change did not touch before you
  decide.
- **Set-up the change's tests repeat.** The tests were written by runs working at the
  same time, none seeing what the others wrote, so several of them can build the same
  objects or carry their own copy of a helper. Move that to the fixture or the helper
  this repository's tests use for it.
- **What the change left that no test uses**: imports, helpers, fixtures.

What stays as it is:

- **What each test checks**: what it passes in, and every value and error it expects.
- **Every test's name, and every case id (case_<n>) exactly as it is spelled.** No test
  is removed, skipped, or folded into another, and no test file is deleted.
- **Everything outside the tests.** Where a test cannot be tidied without changing the
  code, record_decision("<the test>", "<what would have to change>") and leave it.
- **Tests the change did not write**, and the tests "## At the start" lists as already
  failing.""",
        "check": ("A file that is not this run's to write and holds something else than "
                  "when you started, a test file that is gone, a case id that is in no "
                  "test file any more, a call that no longer resolves, a test that fails "
                  "and did not fail before you started, a test file you changed that "
                  "does not pass when run alone, or tests that say something else than "
                  "they did about the code as it was before the change"),
        "request": "Refactor the tests of the change made for this request: {request}",
        "no_request": "Refactor the tests of the change HEAD holds.",
        "target_title": ("The test files you can write to, besides new files under the "
                         "tests' directory. Every other path is refused:"),
        "deadline": ("Make the tests pass, revert what does not, and call COMPLETE(). A "
                     "test left as it was is better than a test left half changed."),
    },
}

# The setting naming the instruction file each stage follows
_GUIDE_SETTING_DICT = {integrate.REFACTOR_STAGE: "REFACTOR_GUIDE",
                       integrate.TEST_REFACTOR_STAGE: "TEST_REFACTOR_GUIDE"}

# The variables a stage is handed, in the shape rlm.build_prompt() shows them
_REFACTOR_VAR_DOC = """- work (dict) — what the "## At the start" and "## State" blocks
  show, so you can read it in code. It has "request" (what the change was made for), "base" and
  "union" (the commit the change was written against, and the commit holding it, which
  is HEAD), "branches" (a list of {"asked", "tests_changed"}: what each run that made
  the change was asked, and the test files it changed itself), "targets" (a list of
  {"file", "names"} for every file you can write to, "names" being the definitions the
  change touched there and empty for a test file), "changed" (a list of {"path",
  "insertions", "deletions", "new", "defs"} for every file you have changed, measured
  again every turn), "read" (a list of {"path", "turn"} for every file you have read
  with read_source_file, read_definition or get_use_of_file), "tests" (what the whole
  test suite said, the last time it was run, that it did not say before you started,
  and "" when it said nothing new), "tests_at" (the turn of that run, 0 for before the
  first turn), "already_failing" (the tests that were failing before you started, which
  are not counted against you), "left" and "checked_at" (what the machine's last check
  found, and the turn it ran on), "decisions" (what you have recorded) and "complete"
  (False when a list was too long to show whole). Writing to it changes nothing: the
  machine rebuilds it before every turn."""

# The late-answer note and its closing line, as run_stage() takes them
_LATE_ANSWER_NOTE = """## Out of turns

You have no turns left, so nothing more can be changed. Say what you changed and what
you left, and NOTHING else - no ```python fence."""

_LATE_ANSWER_CLOSE = "Answer with what you changed and what you left, and nothing else."

# What the guard sends back with the reasons that are still open
_COMPLETE_REFUSAL = ("{call}() was not accepted. The refactoring is not settled:\n{reasons}\n"
                     "Settle these - revert_file() puts a file back - or call {call}() "
                     "again saying why they cannot be settled. {left_text}")

# What a writing tool returns for a path that is not this run's
_NOT_OWN_ERROR = ("{path} is not one of this run's files, so nothing was written. This run "
                  "writes the files work[\"targets\"] lists{new_file_text}. Say what would "
                  "have to change there with record_decision().")
_NEW_TEST_FILE_TEXT = " and new files under the tests' directory"

# What the check finds, one line each
_DELETE_REASON = "- these test files were there when you started and are gone: {paths}"
_CASE_REASON = ("- these case ids were in a test file when you started and are in none "
                "now: {ids}")
_BASE_STATUS_REASON = ("- run over the code as it was before the change, the tests you "
                       "changed were {before} when you started and are {after} now")
_BASE_PASS_REASON = ("- run over the code as it was before the change, these tests failed "
                     "when you started and do not now: {tests}")
_BASE_FEWER_REASON = ("- run over the code as it was before the change, these tests fail "
                      "fewer times now than when you started: {tests}")
_BASE_NAMED_REASON = ("- run over the code as it was before the change, the tests you "
                      "changed failed when you started without naming a test, as tests "
                      "that cannot be built or loaded there do, and now only these fail "
                      "there: {tests}. Any other of them that runs there now passes, "
                      "which none did then")
_FEWER_TEXT = "{name} ({before} times then, {after} now)"
_BASE_LEFT_OUT_REASON = ("- run over the code as it was before the change, the tests now "
                         "leave out what they ran when you started; the run says so in "
                         "these lines, which it did not print then: {lines}. Change what "
                         "you changed, or take it back, so that the run gets as far as it "
                         "did when you started")
_BASE_ERROR_REASON = "- the tests could not be run over the code as it was before the change: {why}"
_CALL_ERROR_REASON = "- the run stopped before it was done, on a call that did not happen: {why}"

# Why a stage called no model
_SKIP_NO_TARGET_DICT = {
    integrate.REFACTOR_STAGE: "the change touched no definition outside the tests",
    integrate.TEST_REFACTOR_STAGE: "the change added or changed no test file",
}
_SKIP_NO_TEST_PATH = "the repository holds no test directory: {path}"
_SKIP_UNRUN = ("the tests ran none of the tests of the test files the change added or "
               "changed when the stage started, the run having failed to load those files "
               "or a directory above them: {paths}")
_SKIP_BASE_PART = ("run over the code as it was before the change, the tests ended before "
                   "every test was run, so what they say there cannot be compared")
_SKIP_BASE_NO_RUN = ("run over the code as it was before the change, the tests reached no "
                     "verdict or failed naming no failing test, so there is nothing to "
                     "compare what they say there by")

# The start block's target lines and the state block's own section
_TARGET_LINE = "  {file}"
_TARGET_NAME_LINE = "  {file} — {names}"
_CHANGE_NOW = "Files you have changed, as of this turn:"
_NO_CHANGE_NOW = "Files you have changed, as of this turn: none."


def build_refactor_instruction(primary_ext: str, stage: str) -> str:
    """Assemble one stage's instructions for the primary language.

    Args:
        primary_ext: The primary language (config.PRIMARY_EXT).
        stage: One of STAGE_TUPLE.

    Returns:
        The assembled instructions, ending with the section of the repository's own
        instruction file for the stage when there is one.
    """
    text_dict = _STAGE_TEXT_DICT[stage]
    return _REFACTOR_INSTRUCTION_TEMPLATE.format(
        opening=text_dict["opening"].format(lang_name=LANG_NAME_DICT[primary_ext]),
        rule=text_dict["rule"],
        check=text_dict["check"],
        path_example=PATH_EXAMPLE_DICT[primary_ext],
        lang_note=edit_paragraph(primary_ext,
                                 {tool.__name__ for tool in stage_tool_list(stage)}),
        output_limit=output_limit_note(OUTPUT_LIMIT_NOTE),
        repl_reach_note=REPL_REACH_NOTE,
        test_command=edit.test_command_text(),
        test_path_note=testcreate.TEST_PATH_NOTE.format(test_path=config.TEST_PATH)
        if edit.has_test_dir() else "",
        test_target_note=edit.TARGET_NOTE_DICT[config.TEST_TARGET_STYLE],
        refusal_text=testcreate.refusal_text(config.REFACTOR_MAX_REFUSAL),
        guide_note=guide.section(_GUIDE_SETTING_DICT[stage]))


# == What a stage refactors and where it writes ==============


def target_dict_of(stage: str, project_dir: str, base: str, range_dict: dict,
                   project_name: str, runnable_only: bool = False) -> dict:
    """Return the files one stage refactors.

    Args:
        stage: One of STAGE_TUPLE.
        project_dir: The repository, whose HEAD holds the change.
        base: The commit the change was written against.
        range_dict: integrate.analyse()'s "range" for HEAD.
        project_name: The directory name of PROJECT_DIR, for the path conversion.
        runnable_only: Keep only the files the test command runs
            (testcreate.runnable_def_dict).

    Returns:
        {repository-relative path: [definition name, ...]}. For the stage that refactors
        the change, testcreate.change_def_dict()'s mapping. For the stage that refactors
        the tests, the test files (under_test_path()) that HEAD added or changed against
        the commit and that are there now, each with no name.
    """
    if stage == integrate.REFACTOR_STAGE:
        target_dict = testcreate.change_def_dict(project_dir, base, range_dict, project_name)
    else:
        status_dict = gitcmd.name_status(project_dir, base, "HEAD")
        target_dict = {path: [] for path in under_test_path(
            sorted(status_dict["added"] + status_dict["modified"]))
            if os.path.isfile(os.path.join(project_dir, path))}
    return testcreate.runnable_def_dict(
        target_dict, "no refactoring is done for %s: the test command does not run that "
                     "language") \
        if runnable_only else target_dict


def may_write_of(stage: str, target_list: list[str],
                 start_file_list: list[str]) -> Callable[[str], bool]:
    """Build the rule on which paths one stage writes.

    Args:
        stage: One of STAGE_TUPLE.
        target_list: The files the stage refactors.
        start_file_list: Every file of the repository when the run started.

    Returns:
        (repository-relative path) -> True for one of target_list and, for the stage
        that refactors the tests, a test file (under_test_path()) that was not there
        when the run started.
    """
    if stage == integrate.REFACTOR_STAGE:
        return set(target_list).__contains__
    return testcreate.may_write_of([], target_list, start_file_list)


def baseline_or_skip(stage: str, target_list: list[str]
                     ) -> tuple[frozenset[str], list[str], str]:
    """Return the baseline before a stage starts and the files it refactors.

    Where the stage calls no model, the reason is returned instead. The tests are run
    only where there is something to refactor. The stage that refactors the tests leaves
    out the files whose tests that run did not run (edit.unrun_path_list): a file it
    named as one it could not load, or one under a directory it named so.

    Args:
        stage: One of STAGE_TUPLE.
        target_list: The files the stage refactors.

    Returns:
        (testcreate.baseline_of()'s set, the files of target_list the stage refactors,
        "") or (an empty set, [], why).
    """
    if not target_list:
        return frozenset(), [], _SKIP_NO_TARGET_DICT[stage]
    if not edit.has_test_command():
        return frozenset(), [], testcreate.SKIP_NO_TEST_COMMAND
    if not edit.test_can_run():
        return frozenset(), [], _SKIP_NO_TEST_PATH.format(path=config.TEST_PATH)
    test_dict = edit.run_test()
    baseline_set = testcreate.baseline_of(test_dict)
    if baseline_set is None:
        return frozenset(), [], testcreate.SKIP_NO_BASELINE
    if stage != integrate.TEST_REFACTOR_STAGE:
        return baseline_set, target_list, ""
    unrun_list = edit.unrun_path_list(target_list, test_dict)
    if len(unrun_list) == len(target_list):
        return frozenset(), [], _SKIP_UNRUN.format(paths=list_text(unrun_list))
    return baseline_set, [path for path in target_list if path not in unrun_list], ""


# == What the machine checks before a run is allowed to end ==


def _case_number_set(project_dir: str) -> set[int]:
    """Return the numbers of the case ids the repository's test files hold now.

    Args:
        project_dir: The repository.

    Returns:
        The numbers, over every test file (under_test_path()).
    """
    path_list = testcreate.repo_test_file_list(project_dir)
    return set().union(*(testcreate.file_case_number_set(project_dir, path)
                         for path in path_list))


def start_record_dict(stage: str, project_dir: str, base: str, target_list: list[str],
                      baseline_set: frozenset[str]) -> dict:
    """Read what the check of a stage that refactors compares against.

    Args:
        stage: One of STAGE_TUPLE.
        project_dir: The repository, as it is when the run starts.
        base: The commit the change was written against.
        target_list: The files the stage refactors.
        baseline_set: baseline_or_skip()'s set for the run before the stage started.

    Returns:
        {"check": edit.run_check()'s value, and for the stage that refactors the tests
        also "case_numbers": _case_number_set()'s value and "base_run":
        testcreate.base_run_dict() over target_list}.

    Raises:
        RuntimeError: When testcreate.base_run_dict() raises it.
    """
    check_dict = {"check": edit.run_check()}
    if stage != integrate.TEST_REFACTOR_STAGE:
        return check_dict
    return {**check_dict, "case_numbers": _case_number_set(project_dir),
            "base_run": testcreate.base_run_dict(project_dir, base, target_list,
                                                 baseline_set)}


def _base_reason_list(project_dir: str, base: str, path_list: list[str],
                      baseline_set: frozenset[str], before_dict: dict) -> list[str]:
    """Return what the tests say over the base's tree now that they did not before.

    Args:
        project_dir: The repository, holding what the stage left.
        base: The commit the change was written against.
        path_list: The test files run there: the ones the stage started with and the
            ones it added.
        baseline_set: baseline_or_skip()'s set for the run before the stage started.
        before_dict: start_record_dict()'s "base_run".

    Returns:
        One line when the run's status there is another, else one line when the run says
        it left out something the run before the stage started did not ("left_out", the
        lines counted one by one), else one line when a test that failed there fails no
        more and one when a test fails there fewer times than it did (its subtests,
        say), and one line when the run could not be made. A run that is red there and
        names no test now gives no line where the run when the stage started named none
        either, and otherwise the third kind's line for every test that run named. One
        that named none when the stage started gives one line, naming the tests that
        fail there now, when the run now names some. A test file named alone, which is
        how a file the run could not load is named, and a test of that file are one
        failure (_is_red_still). No line, and no run, when nothing was run there before
        the stage started.
    """
    if not before_dict["status"]:
        return []
    try:
        after_dict = testcreate.base_run_dict(project_dir, base, path_list, baseline_set)
    except RuntimeError as e:
        return [_BASE_ERROR_REASON.format(why=e)]
    if after_dict["status"] != before_dict["status"]:
        return [_BASE_STATUS_REASON.format(before=before_dict["status"],
                                           after=after_dict["status"])]
    more_dict = collections.Counter(after_dict["left_out"]) - collections.Counter(
        before_dict["left_out"])
    if more_dict:
        return [_BASE_LEFT_OUT_REASON.format(lines=list_text(sorted(more_dict.elements())))]
    if after_dict["status"] == testcreate.BASE_RED and not after_dict["failed"] \
            and not before_dict["failed"]:
        return []
    if after_dict["status"] == testcreate.BASE_RED and not before_dict["failed"]:
        return [_BASE_NAMED_REASON.format(tests=list_text(after_dict["failed"]))]
    before_count_dict = collections.Counter(before_dict["failed"])
    after_count_dict = collections.Counter(after_dict["failed"])
    pass_list = sorted(name for name in before_count_dict
                       if not _is_red_still(name, set(after_count_dict), path_list))
    fewer_list = [_FEWER_TEXT.format(name=name, before=count, after=after_count_dict[name])
                  for name, count in sorted(before_count_dict.items())
                  if 0 < after_count_dict[name] < count]
    line_list = [_BASE_PASS_REASON.format(tests=list_text(pass_list))] if pass_list else []
    if fewer_list:
        line_list.append(_BASE_FEWER_REASON.format(tests=list_text(fewer_list)))
    return line_list


def _test_file_of(name: str, path_list: list[str]) -> str:
    """Return the test file a failing test's name starts with.

    Examples:
        ("tests/test_a.py::test_x", ["tests/test_a.py"]) -> "tests/test_a.py"
        ("tests/test_a.py", ["tests/test_a.py"])         -> ""
        ("Suite > adds", ["test/a.test.js"])             -> ""

    Args:
        name: The name of a failing test, as the run printed it.
        path_list: The test files that were run.

    Returns:
        The path the name starts with and goes on after with a character no path
        holds next, or "" when the name is no test of one of those files.
    """
    for path in path_list:
        rest = name[len(path):] if name.startswith(path) else ""
        if rest and not (rest[0].isalnum() or rest[0] in "/._-"):
            return path
    return ""


def _is_red_still(name: str, after_set: set[str], path_list: list[str]) -> bool:
    """Say whether a test that failed over the base's tree fails there now.

    Args:
        name: A name of the run made when the stage started.
        after_set: The names of the run made now.
        path_list: The test files that were run.

    Returns:
        True when the run names it now, when it is a test of a file the run names
        alone now, and when it is a file named alone and the run names a test of that
        file now.
    """
    if name in after_set:
        return True
    if name in path_list:
        return any(_test_file_of(other, path_list) == name for other in after_set)
    return _test_file_of(name, path_list) in after_set


def reason_list(stage: str, project_dir: str, output_dir: str, base: str, start_dict: dict,
                baseline_set: frozenset[str], exclude_list: list[str] | None,
                target_list: list[str], start_file_list: list[str],
                may_write: Callable[[str], bool], record_dict: dict) -> list[str]:
    """Return what is still wrong with what a stage left, in the model's words.

    The analysis is made again over the working tree. Nothing here calls a model. The
    tests are run over the base's tree only when nothing else was found and the stage
    changed a test.

    Args:
        stage: One of STAGE_TUPLE.
        project_dir: The repository.
        output_dir: Where the analysis is written.
        base: The commit the change was written against.
        start_dict: integrate.analyse()'s return value for HEAD.
        baseline_set: baseline_or_skip()'s set for the run before the stage started.
        exclude_list: The extra names kept out of the analysis, as the build takes them.
        target_list: The files the stage refactors.
        start_file_list: Every file of the repository when the run started.
        may_write: may_write_of()'s return value, the rule the stage's tools write by.
        record_dict: start_record_dict()'s return value.

    Returns:
        One line per reason, empty when there is nothing left.
    """
    line_list = []
    # A file that was there when the run started, is changed, and is not the run's to
    # write. A file the run started without is left out, and so is one that holds what a
    # command the machine ran left there
    touch_list = testcreate.not_own_change_list(start_file_list, may_write)
    if touch_list:
        line_list.append(testcreate.NOT_OWN_REASON.format(paths=list_text(touch_list)))
    if stage == integrate.REFACTOR_STAGE:
        return line_list + testcreate.ref_and_test_reason_list(
            project_dir, output_dir, start_dict, baseline_set, exclude_list, [],
            check_before=record_dict.get("check"))

    exist_list, delete_list = testcreate.own_test_tuple(project_dir)
    if delete_list:
        line_list.append(_DELETE_REASON.format(paths=list_text(delete_list)))
    miss_list = sorted(record_dict["case_numbers"] - _case_number_set(project_dir))
    if miss_list:
        line_list.append(_CASE_REASON.format(ids=list_text(
            [f"{testcreate.CASE_ID_PREFIX}{number}" for number in miss_list])))
    line_list += testcreate.ref_and_test_reason_list(
        project_dir, output_dir, start_dict, baseline_set, exclude_list, exist_list,
        check_before=record_dict.get("check"))
    if line_list or not exist_list:
        return line_list
    base_path_list = sorted(set(exist_list) | {
        path for path in target_list if os.path.isfile(os.path.join(project_dir, path))})
    return _base_reason_list(project_dir, base, base_path_list, baseline_set,
                             record_dict["base_run"])


# == The work list and the stage the model runs ==========


def _request_of(stage: str, request: str) -> str:
    """Return what a stage is asked.

    Args:
        stage: One of STAGE_TUPLE.
        request: The request the change was made for, or "".

    Returns:
        The stage's request, the change's own cut to config.DISPATCH_TASK_MAX_CHAR.
    """
    text_dict = _STAGE_TEXT_DICT[stage]
    return testcreate.ask_text(request, text_dict["request"], text_dict["no_request"])


def build_work_dict(project_dir: str, base: str, request: str, target_dict: dict,
                    baseline_set: frozenset[str], branch_list: list[dict]) -> dict:
    """Build one stage's work dict.

    Args:
        project_dir: The repository, whose HEAD holds the change.
        base: The commit the change was written against.
        request: The request the change was made for, or "".
        target_dict: target_dict_of()'s return value.
        baseline_set: baseline_or_skip()'s set for the run before the stage started.
        branch_list: testcreate.branch_list_of()'s return value.

    Returns:
        The keys _REFACTOR_VAR_DOC names, "branches" and "targets" cut by
        testcreate.cut_work_list().
    """
    target_list = [{"file": path, "names": name_list}
                   for path, name_list in sorted(target_dict.items())]
    return testcreate.cut_work_list(
        {"request": request.strip(), "base": base,
         "union": gitcmd.head_commit(project_dir),
         "changed": [], "read": [], "tests": "", "tests_at": 0,
         "already_failing": edit.baseline_name_list(baseline_set),
         "left": [], "checked_at": 0, "decisions": [], "complete": True},
        {"branches": branch_list, "targets": target_list})


def start_text(stage: str, work_dict: dict) -> str:
    """Render what was measured before the first turn: the block after the request.

    Args:
        stage: One of STAGE_TUPLE.
        work_dict: build_work_dict()'s return value.

    Returns:
        The block, its heading first: the change, the files the stage can write to and
        the tests that were failing.
    """
    line_list = testcreate.change_head_line_list(work_dict)

    line_list += ["", _STAGE_TEXT_DICT[stage]["target_title"]]
    line_list += [(_TARGET_NAME_LINE if entry_dict["names"] else _TARGET_LINE).format(
        file=entry_dict["file"], names=", ".join(entry_dict["names"]))
        for entry_dict in work_dict["targets"]]
    return "\n".join(line_list + testcreate.already_red_line_list(work_dict))


def _state_line_list(work_dict: dict, turn: int, max_turn: int) -> list[str]:
    """Render the state block for the turn about to run.

    Args:
        work_dict: build_work_dict()'s return value, as the turn before this one left it.
        turn: Which turn is about to run (1-based).
        max_turn: The stage's turn limit.

    Returns:
        The lines of the block, the heading first.
    """
    line_list = [testcreate.STATE_TITLE.format(turn=turn, max_turn=max_turn),
                 "", _CHANGE_NOW if work_dict["changed"] else _NO_CHANGE_NOW]
    line_list += testcreate.change_line_list(work_dict["changed"])
    return line_list + testcreate.run_tail_line_list(work_dict)


def make_state(stage: str, project_dir: str, work_dict: dict,
               check: Callable[[], list[str]], baseline_set: frozenset[str],
               range_dict: dict, project_name: str
               ) -> tuple[Callable[[int, int], str], Callable[[], list[str]]]:
    """Build the state block's turn_note() and the check that keeps the block current.

    What the stage changed and read is measured again on every turn, a file that holds
    what a command the machine ran left there (edit.command_write_set) left out of what
    it changed. What the tests say follows the last run of the whole suite, whichever
    side ran it.

    Args:
        stage: One of STAGE_TUPLE.
        project_dir: The repository.
        work_dict: build_work_dict()'s return value, which both return values write to.
        check: () -> [reason, ...], the machine's own reading of what is still wrong.
        baseline_set: baseline_or_skip()'s set for the run before the stage started.
        range_dict: integrate.analyse()'s "range" for HEAD.
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        (turn_note(turn, max_turn) -> str, a check that records what it found in the
        state before returning it). A turn past max_turn is headed and kept as max_turn.
    """
    run_count_dict = {"n": (edit.last_whole_run() or {"n": 0})["n"]}
    is_cut = not work_dict["complete"]
    deadline = make_deadline_note(config.EDIT_DEADLINE_TURN,
                                  _STAGE_TEXT_DICT[stage]["deadline"])

    def turn_note(turn: int, max_turn: int) -> str:
        """Measure the state again and render it for the turn about to run."""
        integrate.set_turn(min(turn, max_turn))
        work_dict["complete"] = not is_cut
        command_write_set = edit.command_write_set()
        testcreate.cut_work_list(work_dict, {
            "changed": [entry_dict for entry_dict in integrate.change_entry_list(
                project_dir, "HEAD", range_dict, project_name)
                if entry_dict["path"] not in command_write_set],
            "read": integrate.read_list()})
        work_dict["decisions"] = integrate.decision_list()
        run_dict = edit.last_whole_run()
        if run_dict and run_dict["n"] != run_count_dict["n"]:
            run_count_dict["n"] = run_dict["n"]
            work_dict["tests"] = testcreate.whole_run_reason(run_dict, baseline_set)
            work_dict["tests_at"] = turn - 1
        note = deadline(turn, max_turn)
        return "\n".join(_state_line_list(work_dict, min(turn, max_turn), max_turn)
                         + (["", note] if note else []))

    return turn_note, testcreate.make_check_and_keep(work_dict, check)


def stage_tool_list(stage: str, may_write: Callable[[str], bool] | None = None,
                    refuse_list: list[str] | None = None) -> list:
    """Return testcreate.stage_tool_list()'s tools (no record_gap) for one stage.

    Args:
        stage: One of STAGE_TUPLE.
        may_write: (repository-relative path) -> whether this run writes there. None
            hands the writing tools over as they are.
        refuse_list: Where each path a write was refused for is added.

    Returns:
        The tool callables, in testcreate.stage_tool_list()'s order.
    """
    return testcreate.stage_tool_list(
        may_write, refuse_list,
        _NOT_OWN_ERROR.replace("{new_file_text}", _NEW_TEST_FILE_TEXT
                               if stage == integrate.TEST_REFACTOR_STAGE else ""))


def _no_run_dict(skip: str) -> dict:
    """Log why a stage calls no model, and return run_refactor()'s value for it.

    Args:
        skip: Why the stage calls no model: baseline_or_skip()'s reason, _SKIP_BASE_PART
            or _SKIP_BASE_NO_RUN.

    Returns:
        The keys run_refactor() returns, the run settled and nothing changed.
    """
    logger.info("nothing is refactored: %s; no model was called", skip)
    return {"work": {"decisions": []}, "called": False, "left": [], "settled": True,
            "skipped": skip, "answer": "", "history": [], "iters": 0, "refused": [],
            "refused_completes": 0, "harvested": False, "error": "", "outcome": None,
            "tests_changed": [], "non_test_changed": [], "untooled": []}


def run_refactor(stage: str, base: str, knowledge_path: str, project_name: str,
                 output_dir: str, exclude_list: list[str] | None = None,
                 request: str = "", runnable_only: bool = False) -> dict:
    """Refactor what HEAD changed against a commit, and check what was left.

    Args:
        stage: One of STAGE_TUPLE: the change outside the tests, or its tests.
        base: The commit the change HEAD holds was written against.
        knowledge_path: The analysis of HEAD.
        project_name: The directory name of PROJECT_DIR.
        output_dir: Where the analysis is written when it is made again.
        exclude_list: The extra names kept out of the analysis, as the build takes them.
        request: The request the change was made for. "" leaves it out.
        runnable_only: As target_dict_of() takes it.

    Returns:
        The stage's RLM.run() return value with "work", "called" (whether a model was
        called), "skipped" (why the stage called no model: baseline_or_skip()'s reason,
        which says there is nothing to refactor, no test command or no test directory,
        that what the tests say could not be read before the stage started, or, for the
        stage that refactors the tests, that none of its files' tests ran; or why the
        stage that refactors the tests has nothing to compare them by: the names its
        tests fail with over the base's tree are of a part of them, or, with their
        status there not testcreate.BASE_RED, they reached no verdict there or counted
        failing tests there and named none of them (testcreate.base_run_dict); or ""),
        "refused" (the paths a write was refused for), "left" (what the machine still
        finds wrong, and the call that did not happen when the run stopped on one),
        "settled", "tests_changed" and "non_test_changed" (the test files and the other
        files the stage changed) and "untooled" (the files it changed without an editing
        tool) added.

    Raises:
        RuntimeError: When testcreate.base_run_dict() raises it before the stage that
            refactors the tests started.
    """
    project_dir = config.PROJECT_DIR
    integrate.reset_stage()
    edit.reset_patch_round(round_limit=0)
    guide.reset()

    # == Step 1: what there is to refactor, and how the tests stand ==
    start_dict = integrate.analyse(knowledge_path)
    target_dict = target_dict_of(stage, project_dir, base, start_dict["range"], project_name,
                                 runnable_only)
    baseline_set, target_list, reason = baseline_or_skip(stage, sorted(target_dict))
    if reason:
        return _no_run_dict(reason)
    target_dict = {path: target_dict[path] for path in target_list}
    start_file_list = gitcmd.work_tree_file_list(project_dir) or []
    record_dict = start_record_dict(stage, project_dir, base, target_list, baseline_set)
    base_run_dict = record_dict.get("base_run", {})
    if base_run_dict.get("is_part") or base_run_dict.get("is_no_run"):
        reason = _SKIP_BASE_PART if base_run_dict.get("is_part") else _SKIP_BASE_NO_RUN
        return _no_run_dict(reason)
    work_dict = build_work_dict(project_dir, base, request, target_dict, baseline_set,
                                testcreate.branch_list_of(project_dir, base))
    may_write = may_write_of(stage, target_list, start_file_list)
    refuse_list: list[str] = []

    # == Step 2: the stage ======================================
    check = functools.partial(reason_list, stage, project_dir, output_dir, base, start_dict,
                              baseline_set, exclude_list, target_list, start_file_list,
                              may_write, record_dict)
    rlm_stage = RLM(llm=make_llm(config.REFACTOR_MODEL, config.REFACTOR_MAX_TOKEN,
                                 "REFACTOR_MODEL", is_turn_reply=True,
                                 think_level=config.REFACTOR_THINK_LEVEL),
                    tool_list=stage_tool_list(stage, may_write, refuse_list),
                    instruction=build_refactor_instruction(config.PRIMARY_EXT, stage),
                    stage=integrate.stage_key(stage),
                    max_turn=config.REFACTOR_MAX_TURN)
    turn_note, check_and_keep = make_state(stage, project_dir, work_dict, check,
                                           baseline_set, start_dict["range"], project_name)
    out_dict = run_stage(rlm_stage,
                         "\n\n".join([_request_of(stage, request),
                                      start_text(stage, work_dict)]),
                         {"work": work_dict},
                         _REFACTOR_VAR_DOC, _LATE_ANSWER_NOTE, _LATE_ANSWER_CLOSE,
                         late_state=turn_note, turn_note=turn_note, text_answer=True,
                         on_complete=integrate.make_complete_guard(
                             check_and_keep, config.REFACTOR_MAX_REFUSAL,
                             _COMPLETE_REFUSAL))

    # == Step 3: the check, where the guard did not make it ====
    need_check = not (out_dict["outcome"] == COMPLETE_KEY
                      and out_dict["refused_completes"] < config.REFACTOR_MAX_REFUSAL)
    out_dict["left"] = (check_and_keep() if need_check else work_dict["left"]) \
        + ([_CALL_ERROR_REASON.format(why=out_dict["error"])] if out_dict["error"] else [])
    out_dict["settled"] = not out_dict["left"]
    work_dict["decisions"] = integrate.decision_list()
    test_change_set = edit.test_change_set()
    out_dict.update({"work": work_dict, "called": True, "skipped": "",
                     "refused": sorted(refuse_list),
                     "tests_changed": sorted(test_change_set),
                     "non_test_changed": sorted(edit.own_change_set() - test_change_set),
                     "untooled": sorted(edit.no_tool_path_set())})
    return out_dict
