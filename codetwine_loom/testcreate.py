"""Write tests for the definitions a combined change touched.

The run is over a repository whose HEAD is the combined change. It reads which
definitions the change touched off the diff against the commit the change was written
against, and reads which of the repository's tests refer to them off the analysis.

Two stages follow. The first decides the test cases from the request, the tasks the
request was split into and the first line of each definition; the one reading tool it is
handed reads the test files alone. The second is handed those cases with the tools that
read and write the whole repository, and writes the tests. Its instructions end with the
repository's own instruction file on its tests (guide.py); the first stage's do not. It
is started only when every definition has a case or a reason it gets none. The cases are
numbered from the first number a task names, or else on from the highest case id the
repository's test files hold, by the step a task names, or else one apart
(run_test_create's task_dict).

What the second stage leaves is checked without a model: no file that was there when the
run started and that the stage does not write holds anything else, what a command the
machine ran wrote there aside, every case is in a test file it wrote or in a decision it
recorded, no test file it changed is gone, the static check finds nothing it did not
find when the run started, the references that resolved at HEAD still resolve, no test
fails that did not fail before the run started, the test files it wrote that the test
command runs pass when they are run alone where the build tool takes paths, and the
whole suite runs each of them that holds a case. The test files it wrote are then run
over the tree of the commit the change was written against, and whether they fail there
is recorded. A working tree that cannot be put back after that run leaves the stage
unsettled.

A repository with no test command, one where no path can be a test file (TEST_PATH names
no directory and the language has no test file names), one with no test file and no
dependency file the stage could write, a test suite that could not be read before the
stage started, and a change that touched no definition outside the tests call no model.
A test suite that ran nothing is read as one with nothing failing: a repository that
holds no test yet gets its first ones here, in a dependency file where a package or a
test script is missing.

This module imports nothing from agent.py.
"""

import collections
import contextlib
import functools
import logging
import os
import re
import shlex
from collections.abc import Callable

from codetwine.utils.file_utils import read_source_text

from codetwine_loom import config, gitcmd, guide, integrate, tmpfile
from codetwine_loom.knowledge import ensure_knowledge
from codetwine_loom.languages import (ALIAS_DICT, LANG_DICT, LANG_NAME_DICT,
                                      PATH_EXAMPLE_DICT, LangConfig, dep_file_example,
                                      edit_paragraph, header_owner, is_dep_file, lang_ext,
                                      name_key, root_target_pattern_tuple)
from codetwine_loom.llm import make_llm
from codetwine_loom.paths import (has_test_rule, knowledge_to_repo, list_text, real_rel,
                                  repo_to_knowledge, to_repo_relative, under_test_path)
from codetwine_loom.rlm import (COMPLETE_KEY, END_CALL_DICT, OUTPUT_LIMIT_NOTE,
                                REPL_REACH_NOTE, RLM, cut, json_guard,
                                make_deadline_note, no_answer_reason, output_limit_note,
                                parse_json_answer, refuse_up_to, run_stage)
from codetwine_loom.tools import edit, source

logger = logging.getLogger(__name__)

# What the stage records about the tests it wrote, run over the tree of the commit the
# change was written against: they fail there, they pass there as well, or neither can
# be told. A stage that wrote no test records none
BASE_RED = "red"
BASE_GREEN = "green"
BASE_UNKNOWN = "unknown"

# What a case is named by: the prefix and a number
CASE_ID_PREFIX = "case_"

# A case id in a text, the number its one group. An id that follows a letter or a digit
# is part of another word
_CASE_NUMBER_RE = re.compile(rf"(?<![A-Za-z0-9]){CASE_ID_PREFIX}(\d{{1,18}})(?!\d)")

# How a stage's instructions end the sentence on the check, by the stage's refusal limit
_REFUSAL_TEXT_DICT = {0: "is recorded, and the call is accepted as it is.",
                      1: "sends it back once."}
_REFUSAL_TEXT_MANY = "sends it back, up to {count} times."

_TEST_CASE_INSTRUCTION_TEMPLATE = """You decide what the tests of a change to a
{lang_name} repository have to check. The change is made already, and you are not shown
how: what a test expects has to come from what was asked for, so that a change that does
the wrong thing is caught rather than copied. Somebody else writes the tests from your
list afterwards, with the code in front of them.

Each turn you write ONE ```python block. It runs, you see what it printed, then you write
the next block. print() what you want to see{output_limit}. Names you define stay \
available in later turns. Stop at the closing
fence — anything you write after it is thrown away and costs you the turn.

- {repl_reach_note}

After the request you are given a "## What was asked" block: the request, the tasks it
was split into, each as it was handed to the person who did it, the definitions the
change touched with their first lines and the tests that already refer to them, and the
repository's test files. The same is in work, to read in code. read_source_file(path)
reads a test file; it reads nothing else.

Write one case for each thing the request or a task says a definition does. Keep the
cases in a list you add to as you go, one definition at a time, and hand that list over
at the end: a case you print is a case you will have to write out again.

- **Take every case from a sentence of the request or of a task.** Put that sentence
  in "from". What nobody asked for is not a case: where the request names an error but
  not its message, the case expects the error and says nothing of the message.
- **Say what goes in and what comes out.** "input" is how the definition is called, in
  the terms of its first line; "expect" is what comes back, or what is raised.
- **Cover what was asked at its edges as well as in the middle**, where the request
  states an edge: a value it names, a limit it gives, a condition it puts on an error.
- **Leave out what a test already checks.** Read the tests the block names before you
  write a case for a definition they refer to.
- **Every definition the block lists gets at least one case, or a reason it gets none**:
  the tests already check what was asked of it, or nothing that was asked of it can be
  checked by a test.

A case is a dict, and so is a reason:

case_list.append({{"file": "<the definition's file>", "name": "<the definition>",
                  "input": "<how it is called>", "expect": "<what comes back or is raised>",
                  "from": "<the sentence it is taken from>"}})
no_case_list.append({{"file": "<the definition's file>", "name": "<the definition>",
                     "why": "<why it gets no case>"}})

When every definition is in one of the two lists, call
COMPLETE({{"cases": case_list, "no_case": no_case_list}}) and nothing else. It is checked
before it is accepted: a definition that is in neither list {refusal_text}"""

# The variables the first stage is handed, in the shape rlm.build_prompt() shows them
_TEST_CASE_VAR_DOC = """- work (dict) — "request" (what the change was made for),
  "branches" (a list of {"asked", "tests_changed"}: what each run that made the change
  was asked, and the test files it changed itself), "targets" (a list of {"file", "name",
  "signature", "tests"} for every definition the change touched, "signature" being its
  first line and "tests" a list of {"path", "lines"} for the tests that refer to it),
  "test_files" (the repository's test files) and "complete" (False when a list was too
  long to show whole)."""

# Shown over the first stage's last turns
_CASE_DEADLINE_WHAT = ('Stop reading and call COMPLETE({"cases": case_list, "no_case": '
                       'no_case_list}) now.')

# The first stage's late-answer note and its closing line, as run_stage() takes them
_CASE_LATE_ANSWER_NOTE = """## Out of turns

You have no turns left. Reply with the JSON object holding the cases you have, and
NOTHING else - no ```python fence."""

_CASE_LATE_ANSWER_CLOSE = "Answer with the JSON object and nothing else."

# The block placed after the first stage's request, which holds what work holds
_CASE_START_TITLE = "## What was asked"
_CASE_REQUEST_LINE = "The request: {request}"
_CASE_NO_REQUEST_LINE = "The request was not recorded; the tasks below are what was asked."
_CASE_TASK_TITLE = "The tasks it was split into, each as it was handed over:"
_CASE_TARGET_TITLE = ("The definitions the change touched, each with its first line and "
                      "the tests that already refer to it:")
_CASE_TARGET_LINE = "  {name} in {file} — `{signature}` — {tests}"
_CASE_NO_TEST = "no test refers to it"
_CASE_TEST_FILE_TITLE = "The repository's test files, which read_source_file reads:"

# What a writing tool of the second stage returns for a path that is not this run's.
# {dep_file_text} is _NOT_OWN_DEP_FILE_TEXT for a language with a dependency file
_NOT_OWN_ERROR = ("{path} is not one of this run's files, so nothing was written. This run "
                  "writes work[\"own_files\"], work[\"test_files_to_extend\"]{dep_file_text} "
                  "and new test files. Where the code there does not do a case, say so "
                  "with record_gap() and the test you would have written; where no test "
                  "file can reach a case, say so with record_decision().")
_NOT_OWN_DEP_FILE_TEXT = ", work[\"dep_files\"]"
_REFUSE_REASON = "- a change outside this run's files was refused: {paths}"

# What the first stage's guard sends back while a definition is in neither list
_CASE_REFUSAL = ("{call}() was not accepted. These definitions have no case and no reason "
                 "they get none:\n{names}\nAdd a case for each to case_list, or a reason "
                 "to no_case_list, and call {call}() again with both lists.")
_UNCOVER_REASON = "- no case was decided for: {names}"

# What is wrong with a first stage's answer that is not an object holding the cases: the
# error parse_case_list() returns, and the text _case_shape_reason() fills in
_NO_CASE_LIST_ERROR = 'the answer has no "cases" list'
_CASE_SHAPE_REASON = (_NO_CASE_LIST_ERROR + ': hand over one object holding both lists, '
                      '{call}({{"cases": case_list, "no_case": no_case_list}})')

# What the first stage's reading tool returns for a path that is not a test file
_NOT_A_TEST_ERROR = ("Error: {path} is not a test file. This stage reads the files "
                     "work[\"test_files\"] lists and nothing else.")

# What the first stage is asked, with and without the request the change was made for
_CASE_REQUEST_TEXT = "Decide the test cases of the change made for this request: {request}"
_CASE_NO_REQUEST_TEXT = "Decide the test cases of the change the tasks in work describe."

_TEST_CREATE_INSTRUCTION_TEMPLATE = """You are writing tests for a change that was just
made to a {lang_name} repository. The change is already in the working tree. What is
left is to leave tests behind that say what the change was asked to do.

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

Your one job is to write the tests work["cases"] lists. The cases were decided from the
request before anyone read the code, and each holds the sentence it was taken from:

- **Write one test for each case, and put its id in that test**, in a comment or in the
  test's name, exactly as work["cases"] spells it.
- **Take what a test expects from the case, not from the code.** Read the code to learn
  how the definition is called. A value copied out of what the code returns today proves
  nothing about the request.
- **Where a case cannot be written as it stands**, or asks for something its "from"
  sentence does not say, call record_decision("<the id>", "<why>") and write what the
  sentence does say, or leave the case out.
- **Write the tests the way this repository writes them.** Read a test file that is
  already there before you write one: the same framework, the same layout, the same
  helpers, and nothing the existing tests do not already import.
- **Add to the test file that already covers a definition** where
  work["test_files_to_extend"] names one. Create a new test file only where none does.
  Those, new test files{dep_file_text} and work["own_files"] are all you can write to: a
  write to any other path is refused, and another run writes the tests of the other
  files at the same time. In a file that sets how the tests are found and run, leave
  which tests it finds, which of them run and how their outcome is read as they are.
- **Do not weaken, skip or delete a test that is already there**, and leave the tests
  "## At the start" lists as already failing alone.
- **A test must not reach the network, read the clock, depend on the order the tests
  run in, or name a path outside the repository.**

work["referenced"] lists the definitions the change touched that a test already refers
to, with the test files and lines. work["unreferenced"] lists the ones no test refers
to. Both come from the references the analysis could follow, so a definition a test
reaches some other way can be in the second list: read before you write.

Where a test you wrote fails, read the test, the code and the case's "from" sentence
before you change anything. Where the test is built wrong, fix the test. Where the code
does not do what the sentence says and it is in one of work["own_files"], fix the code,
with the smallest change that makes it do that, and say so with
record_decision("<the id>", "<what the code did not do>"). Where it is in any other
file, or the change it needs is more than a small one, change nothing there:
record_gap("<the id>", "<which file, and what it does that the sentence does not say>",
"<a new test file's path>", "<that test file's whole text>") with the test you would
have written, and leave that case's test out of the suite. The machine runs that test
once and takes it back out; it runs it again once every run's tests are combined, and a
case another run has made the code do by then is no longer counted. Every file you
change that is not a test is recorded in the commit.

run_test() runs `{test_command}` from the repository root{test_path_note}.
{test_target_note} Run what
you wrote as you go, and the whole suite before you finish. revert_file(path) puts a
file back to what it held when you started.

Every path you pass to a tool is relative to the repository root, e.g. "{path_example}".
{lang_note}
When the tests are written and pass, call COMPLETE("<what you did>") and nothing else.
Write it in the language work["request"] is written in. It is checked before it is
accepted. A case that is in no test you wrote and in no decision, a call that no longer
resolves, a test that fails and did not fail before you started, or a test file you
wrote that does not pass when run alone {refusal_text}{guide_note}"""

# Where the tests are, left out when TEST_PATH names no directory in the repository
# (edit.has_test_dir())
TEST_PATH_NOTE = "; this repository's tests live under `{test_path}`"

# The dependency files the second stage writes to as well, left out for a language
# that has none. {example} is the first name of the language's table, and the last of
# its test_build_file_tuple after " or " when it has one
_DEP_FILE_TEXT = (", a dependency file (such as {example}) where a test needs a package "
                  "or a test script that is not there")


# The variables the stage is handed, in the shape rlm.build_prompt() shows them
_TEST_CREATE_VAR_DOC = """- work (dict) — what the "## At the start" and "## State" blocks
  show, so you can read it in code. It has "request" (what the change was made for), "base" and
  "union" (the commit the change was written against, and the commit holding it, which
  is HEAD), "branches" (a list of {"asked", "tests_changed"}: what each run that made
  the change was asked, and the test files it changed itself), "cases" (a list of
  {"id", "file", "name", "input", "expect", "from", "in"} for every test to write, "from"
  being the sentence of the request the case was taken from and "in" the test files you
  have changed that hold its id, measured again every turn),
  "referenced" (a list of {"file", "name", "tests"} for every definition the change
  touched that a test refers to, "tests" being a list of {"path", "lines"}),
  "unreferenced" (a list of {"file", "name"} for every one no test refers to),
  "test_files" (the repository's test files as they were when you started),
  "own_files" (the files whose definitions your tests are for, which with the
  dependency files are the only files outside the tests you can write to),
  "test_files_to_extend" (the test files that already refer to them, which with new
  test files are the only tests you can write to), "dep_files" (the repository's
  dependency files, which you can write to as well), "written"
  (a list of {"path", "insertions", "deletions", "new", "defs"} for every test file you
  have changed, measured again every turn), "also_changed" (the same for every file you
  have changed that is not a test), "read" (a list of {"path", "turn"} for every file
  you have read with read_source_file, read_definition or get_use_of_file), "tests"
  (what the whole test suite said, the last time it was run, that it did not say before
  you started, and "" when it said nothing new), "tests_at" (the turn of that run, 0 for
  before the first turn), "already_failing" (the tests that were failing before you
  started, which are not counted against you), "left" and "checked_at" (what the
  machine's last check found, and the turn it ran on), "decisions" and "gap" (what you
  have recorded with record_decision() and record_gap()) and "complete" (False when a list was too long to show whole). Writing to it
  changes nothing: the machine rebuilds it before every turn."""

# Shown over the stage's last turns
_DEADLINE_WHAT = ("Make the tests you have written pass, revert what does not, and call "
                  "COMPLETE(). A test left failing is worse than a test not written.")

# The late-answer note and its closing line, as run_stage() takes them
_LATE_ANSWER_NOTE = """## Out of turns

You have no turns left, so nothing more can be changed. Say which tests you wrote and
what you left, and NOTHING else - no ```python fence."""

_LATE_ANSWER_CLOSE = "Answer with which tests you wrote and what you left, and nothing else."

# What the guard sends back with the reasons that are still open
_COMPLETE_REFUSAL = ("{call}() was not accepted. The tests are not settled:\n{reasons}\n"
                     "Settle these, or call {call}() again saying why they cannot be "
                     "settled. {left_text}")

# What the stage is asked, with and without the request the change was made for
_REQUEST_TEXT = "Write tests for the change made for this request: {request}"
_NO_REQUEST_TEXT = "Write tests for the change HEAD holds."

_CASE_REASON = ("- these cases are in no test file you changed and no record_decision() "
                "or record_gap() names them: {ids}")
_BROKEN_REASON = "- these calls resolved before you started and no longer do: {calls}"
_ANALYSIS_REASON = "- the analysis could not be made again: {why}"
_DELETE_REASON = "- these test files were there before you started and are gone: {paths}"
NOT_OWN_REASON = ("- these files are not this run's to write and do not hold what they "
                  "held when you started: {paths}")
# What one test run says, without the bullet the check's lines start with
_TEST_BREAK_REASON = ("`{command}` fails {tests}, which did not fail before you "
                      "started ({why})")
_TEST_REASON = "`{command}` fails ({why})"
_TEST_EARLY_STOP_REASON = ("`{command}` stopped before it had run every test ({why}). The "
                           "failing tests it named up to there: {tests}")
_NO_TEST_NAME_TEXT = "none"
_TEST_NO_RUN_REASON = "`{command}` did not run ({why})"
_TEST_UNRUN_REASON = ("`{command}` no longer runs {tests}, which ran before you started: "
                      "they are not found, are left out or skip now")
_TEST_FEWER_RUN_REASON = ("`{command}` runs {now} of the {before} tests it ran before you "
                          "started: the others are not found, are left out or skip now")
_TEST_UNREAD_REASON = ("`{command}` no longer prints which of its tests ran, as it did "
                       "before you started, so they cannot be checked against the tests "
                       "that ran then")
_TEST_SHORT_REASON = ("the `PASSED` lines of `{command}` do not name each test it counts as "
                      "passed, now or before you started, so the tests it runs cannot be "
                      "checked against the tests that ran then")
_OWN_TEST_REASON = "- run over {paths} alone, {reason}"
_UNRUN_REASON = ("- `{command}` never runs {paths}: the whole run does not take them in, "
                 "so no test in them runs")

# Why a stage is held back when a file holding a case could not be put back after the
# whole run was tried with a line added to it (edit.in_whole_run)
_PUT_BACK_REASON = ("- {path} no longer holds what you wrote: the whole run tried with a "
                    "line added to it left it so that it could not be written back ({why}). "
                    "Put it back with revert_file, and the same for any file that run "
                    "deleted or put in its place (in git_diff_stat, one you did not "
                    "change that has deletions only or is untracked), then write it "
                    "again")

# Why the stage called no model
SKIP_NO_TEST_COMMAND = "there is no test command to run"
_SKIP_NO_TEST_RULE = ("TEST_PATH names no directory and this language tells no test file "
                      "by its name, so no path would be a test file. Set TEST_PATH to "
                      "the directory the tests are in")
_SKIP_NO_TEST_FILE = ("the repository holds no test file, and this language has no "
                      "dependency file the stage could write")
SKIP_NO_BASELINE = "what the tests say could not be read before the stage started"
_SKIP_NO_DEFINITION = "the change touched no definition outside the tests"
_SKIP_NO_CASE = ("no case was decided: the tests already check what was asked, or "
                 "nothing that was asked can be checked by a test")

# Why the run ended without the second stage
_NO_CASE_REASON = "- the test cases could not be read: {why}"

# Why a run whose check passed is not settled: the run over the base's tree left that
# tree in the working tree
_RESTORE_REASON = ("- after the tests were run over the code as it was before the change, "
                   "{why}")

# The block placed after the request: what was measured before the first turn and stays
# as it is for the whole run
_START_TITLE = "## At the start"
_CHANGE_TITLE = "The change, written against {base} and held by {union}:"
_CASE_TITLE = ("The cases to write a test for, decided from the request before the code was "
               "read. Where each id is, among the test files you have changed, is in the "
               "state block:")
_CASE_LINE = "  {id} {name} in {file} — {input} → {expect}"

# The state block, rebuilt from the repository before every turn
STATE_TITLE = "## State (turn {turn}/{max_turn})"
_CASE_NOW = ("Where each case is, among the test files you have changed, as of this "
             "turn:")
_CASE_NOW_LINE = "  {id} {name} in {file} [{where}]"
_CASE_NOWHERE = "in no test yet"
_CASE_IN_DECISION = "in a decision you recorded"
_CASE_IN_GAP = "in a gap you recorded"
_REF_TITLE = ("Definitions the change touched that a test refers to, as measured "
              "before the first turn:")
_NO_REF = "No test refers to a definition the change touched."
_REF_LINE = "  {name} in {file} — {tests}"
_NO_REF_TITLE = ("Definitions the change touched that no test refers to, as "
                 "measured before the first turn:")
_ALL_REF = "Every definition the change touched has a test referring to it."
_TEST_FILE_TITLE = "The repository's test files, as they were when you started:"
_OWN_FILE_TITLE = ("The files you can write to, besides new test files. Every other path "
                   "is refused:")
_DEP_FILE_LINE = "  {path}  (dependency file)"
_NOTHING_LEFT = ("Every case is in a test or a decision, and the whole suite, run after your "
                 "last change, said nothing it did not say before you started: the check "
                 "has nothing left to find. Call COMPLETE() now.")
_OWN_TEST_NOW = "Test files you have changed, as of this turn:"
_NO_OWN_TEST_NOW = "Test files you have changed, as of this turn: none."
_OTHER_CHANGE_NOW = ("Files you have changed that are not tests, as of this turn. Each "
                     "is recorded in the commit:")
_NO_OTHER_CHANGE_NOW = "Files you have changed that are not tests, as of this turn: none."
_READ_NOW = "Files you have read, as of this turn:"
_NO_READ_NOW = "Files you have read, as of this turn: none."
_TEST_BREAK = ("What the whole test suite said the last time it was run ({when}) that it "
               "did not say before you started:")
_NO_TEST_BREAK = ("The whole test suite, the last time it was run ({when}), said nothing "
                  "it did not say before you started.")
_TEST_BEFORE_FIRST_TURN = "before the first turn"
_TEST_ON_TURN = "on turn {turn}"
_ALREADY_RED = ("Tests that were failing before you started. They are not counted "
                "against you, and they are not yours to change:")
_CHECK_AT = "The machine last checked the repository on turn {turn}. It found:"
_CHECK_FIND_NOTHING = "  nothing left"
_NO_CHECK_YET = ("The machine has not checked the repository yet. It does that when you "
                 "call COMPLETE().")
_DECISION_TITLE = "What you have recorded with record_decision():"
_NO_DECISION = "You have recorded nothing with record_decision() yet."
_GAP_TITLE = "What you have recorded with record_gap():"


def refusal_text(refusal_max: int) -> str:
    """Return how a stage's instructions end the sentence on the check.

    Args:
        refusal_max: How many times the stage's COMPLETE() is sent back.

    Returns:
        The end of the sentence, for that count.
    """
    return _REFUSAL_TEXT_DICT.get(refusal_max, _REFUSAL_TEXT_MANY.format(count=refusal_max))


def build_test_create_instruction(primary_ext: str) -> str:
    """Assemble the stage's instructions for the primary language.

    Args:
        primary_ext: The primary language (config.PRIMARY_EXT).

    Returns:
        The assembled instructions, ending with the section of the repository's own
        instruction file on its tests when there is one.
    """
    dep_file_tuple = LANG_DICT[primary_ext].dep_file_tuple
    return _TEST_CREATE_INSTRUCTION_TEMPLATE.format(
        lang_name=LANG_NAME_DICT[primary_ext],
        path_example=PATH_EXAMPLE_DICT[primary_ext],
        lang_note=edit_paragraph(primary_ext,
                                 {tool.__name__ for tool in test_create_tool_list()}),
        output_limit=output_limit_note(OUTPUT_LIMIT_NOTE),
        repl_reach_note=REPL_REACH_NOTE,
        test_command=edit.test_command_text(),
        test_path_note=TEST_PATH_NOTE.format(test_path=config.TEST_PATH)
        if edit.has_test_dir() else "",
        dep_file_text=_DEP_FILE_TEXT.format(example=" or ".join(
            [dep_file_example(dep_file_tuple[0])]
            + list(LANG_DICT[primary_ext].test_build_file_tuple[-1:])))
        if dep_file_tuple else "",
        test_target_note=edit.TARGET_NOTE_DICT[config.TEST_TARGET_STYLE],
        refusal_text=refusal_text(config.TEST_CREATE_MAX_REFUSAL),
        guide_note=guide.section("TEST_GUIDE"))


# The tools of the second stage that write to a file, whose first argument is the path
_WRITE_TOOL_NAME_TUPLE = ("replace_in_file", "replace_line_range", "create_file",
                          "revert_file")


def _land_path(path: str) -> str:
    """Return the repository-relative path of the file a tool's path argument lands on.

    Examples:
        ("./tests/a.py")                        -> "tests/a.py"
        ("tests/link.py") (-> src/real.py)      -> "src/real.py"

    Args:
        path: The path as the tool was given it, in any notation to_repo_relative() reads.

    Returns:
        The path with its symbolic links resolved, or to_repo_relative()'s answer for a
        path resolve_in_project() refuses or cannot resolve (RecursionError).
    """
    file_rel = to_repo_relative(path)
    with contextlib.suppress(ValueError, RecursionError):
        return real_rel(file_rel)
    return file_rel


def _own_file_write(tool: Callable, may_write: Callable[[str], bool],
                    refuse_list: list[str], refuse_text: str) -> Callable:
    """Wrap a writing tool to refuse what may_write refuses, revert_file over a file aside.

    The wrapped tool takes the tool's name, docstring and signature (functools.wraps).
    The tool is handed the path as to_repo_relative() converts it. revert_file over a
    file this run has not changed does nothing and succeeds, whichever file it is, and
    over a file this run has changed it is passed on, whichever file it is. What counts
    as changed is edit.own_change_set()'s answer for the path as it is given, a symbolic
    link included, or for the file it lands on.

    Args:
        tool: The writing tool, whose first argument is the path.
        may_write: (repository-relative path) -> whether this run writes there.
        refuse_list: Where each path a write was refused for is added, once.
        refuse_text: What a refused write is answered with, with {path}.

    Returns:
        The wrapped tool, which hands back edit.refusal_dict() with refuse_text for a
        write it refuses.
    """
    @functools.wraps(tool)
    def write_tool(path: str, *args: object, **kwargs: object) -> dict:
        """Run the tool on a path this run writes, or refuse the write."""
        own_path = _land_path(str(path))
        is_dir = os.path.isdir(os.path.join(config.PROJECT_DIR, own_path))
        if tool.__name__ == "revert_file" and not is_dir:
            change_set = edit.own_change_set()
            if own_path not in change_set and to_repo_relative(str(path)) not in change_set:
                return {"ok": True, "path": str(path), "error": ""}
            return tool(to_repo_relative(str(path)), *args, **kwargs)
        if may_write(own_path):
            return tool(to_repo_relative(str(path)), *args, **kwargs)
        if own_path not in refuse_list:
            refuse_list.append(own_path)
        logger.info("a write to %s was refused: it is not one of this run's files", own_path)
        return edit.refusal_dict(str(path), refuse_text.format(path=own_path))

    return write_tool


def not_own_error(with_dep_file: bool = False) -> str:
    """Return what a writing tool of the second stage answers for a path not this run's.

    Args:
        with_dep_file: Whether the run writes the dependency files too.

    Returns:
        _NOT_OWN_ERROR with {path} left in, naming the dependency files when asked.
    """
    return _NOT_OWN_ERROR.replace("{dep_file_text}",
                                  _NOT_OWN_DEP_FILE_TEXT if with_dep_file else "")


def stage_tool_list(may_write: Callable[[str], bool] | None = None,
                    refuse_list: list[str] | None = None,
                    refuse_text: str | None = None) -> list:
    """Return the second stage's tools but record_gap.

    They are integrate.integrate_tool_list()'s tools, with git_diff_stat and
    revert_file, both of which read against HEAD.

    Args:
        may_write: (repository-relative path) -> whether this run writes there. None
            hands the writing tools over as they are.
        refuse_list: Where each path a write was refused for is added.
        refuse_text: What a refused write is answered with, with {path}. None answers
            with not_own_error().

    Returns:
        The tool callables.
    """
    tool_list = integrate.integrate_tool_list() + [edit.git_diff_stat, edit.revert_file]
    if may_write is None:
        return tool_list
    return [_own_file_write(tool, may_write, refuse_list if refuse_list is not None else [],
                            refuse_text if refuse_text is not None else not_own_error())
            if tool.__name__ in _WRITE_TOOL_NAME_TUPLE else tool for tool in tool_list]


def test_create_tool_list(may_write: Callable[[str], bool] | None = None,
                          refuse_list: list[str] | None = None,
                          refuse_text: str | None = None) -> list:
    """Return stage_tool_list()'s tools and record_gap.

    Args:
        may_write: As stage_tool_list() takes it.
        refuse_list: As stage_tool_list() takes it.
        refuse_text: As stage_tool_list() takes it.

    Returns:
        The tool callables.
    """
    return stage_tool_list(may_write, refuse_list, refuse_text) + [integrate.record_gap]


def may_write_of(own_file_list: list[str], test_path_list: list[str],
                 start_file_list: list[str],
                 dep_file_pattern_set: frozenset[str] = frozenset()) -> Callable[[str], bool]:
    """Build the rule on which paths one run of the second stage writes.

    Args:
        own_file_list: The files whose definitions this run writes tests for.
        test_path_list: The test files that refer to those files.
        start_file_list: Every file of the repository when the run started.
        dep_file_pattern_set: The base-name patterns of the dependency files the run
            writes too, at any depth (languages.is_dep_file()). Empty for none.

    Returns:
        (repository-relative path) -> True for one of own_file_list, one of
        test_path_list, a test file (under_test_path()) that was not there when the run
        started, and a path whose base name matches dep_file_pattern_set.
    """
    allow_set = set(own_file_list) | set(test_path_list)
    start_set = set(start_file_list)

    def may_write(path: str) -> bool:
        """Say whether this run writes the repository-relative path."""
        return path in allow_set or (path not in start_set and bool(under_test_path([path]))) \
            or is_dep_file(path, dep_file_pattern_set)

    return may_write


def build_test_case_instruction(primary_ext: str) -> str:
    """Assemble the first stage's instructions for the primary language.

    Args:
        primary_ext: The primary language (config.PRIMARY_EXT).

    Returns:
        The assembled instructions.
    """
    return _TEST_CASE_INSTRUCTION_TEMPLATE.format(
        lang_name=LANG_NAME_DICT[primary_ext],
        output_limit=output_limit_note(OUTPUT_LIMIT_NOTE),
        repl_reach_note=REPL_REACH_NOTE,
        refusal_text=refusal_text(config.TEST_CREATE_MAX_REFUSAL))


def _test_file_read(tool: Callable) -> Callable:
    """Wrap a reading tool; the wrapped tool reads test files (under_test_path()) alone.

    The wrapped tool takes the tool's name, docstring and signature (functools.wraps).
    The tool is handed the path as to_repo_relative() converts it.

    Args:
        tool: The reading tool, whose first argument is the path.

    Returns:
        The wrapped tool, which returns _NOT_A_TEST_ERROR for any other path.
    """
    @functools.wraps(tool)
    def read_tool(path: str, *args: object, **kwargs: object) -> str:
        """Run the tool on a test file, or answer that the path is not one."""
        if not under_test_path([_land_path(str(path))]):
            return _NOT_A_TEST_ERROR.format(path=path)
        return tool(to_repo_relative(str(path)), *args, **kwargs)

    return read_tool


def case_tool_list() -> list:
    """Return read_source_file, wrapped to read the test files alone.

    Returns:
        The tool callables.
    """
    return [_test_file_read(source.read_source_file)]


# == What the change touched and which tests refer to it =====


def change_def_dict(project_dir: str, base: str, range_dict: dict,
                    project_name: str) -> dict:
    """Return the definitions outside the tests that HEAD changed against a commit.

    The diff is read from HEAD to the commit: the lines are HEAD's and are mapped
    onto HEAD's definitions. A file the change added holds every name it defines.

    Args:
        project_dir: The repository, whose HEAD holds the change.
        base: The commit the change was written against.
        range_dict: integrate.analyse()'s "range" for HEAD.
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        {repository-relative path: [definition name, ...]}, the names sorted. A test
        file (under_test_path()), a file under a directory a build_output_tuple entry
        starting with "/" names, tracked or not (edit.build_output_path_list), and a file
        whose changed lines lie outside every definition are left out.
    """
    side_dict = integrate.side_dict_of(
        integrate.change_dict_of(project_dir, "HEAD", base), range_dict, project_name)
    def_dict = {knowledge_to_repo(file, project_name): sorted(name_set - {""})
                for file, name_set in side_dict.items()}
    drop_set = set(under_test_path(def_dict)) \
        | set(edit.build_output_path_list(list(def_dict)))
    return {path: name_list for path, name_list in sorted(def_dict.items())
            if name_list and path not in drop_set}


def runnable_def_dict(def_dict: dict, drop_text: str) -> dict:
    """Keep the entries of a {path: ...} mapping whose file the test command runs.

    Args:
        def_dict: {repository-relative path: anything}.
        drop_text: The line logged of the paths left out, with "%s" for them.

    Returns:
        The entries whose path edit.runnable_path_list() keeps: a file of the primary
        language, or of a language with the same test command.
    """
    keep_set = set(edit.runnable_path_list(list(def_dict)))
    drop_list = [path for path in def_dict if path not in keep_set]
    if drop_list:
        logger.info(drop_text, list_text(drop_list))
    return {path: value for path, value in def_dict.items() if path in keep_set}


def testable_def_dict(def_dict: dict) -> dict:
    """Keep the files of change_def_dict()'s mapping that the test command can test.

    Args:
        def_dict: change_def_dict()'s return value.

    Returns:
        runnable_def_dict()'s mapping of it.
    """
    return runnable_def_dict(def_dict, "no test is written for %s: the test command does "
                                       "not run that language")


def change_lang_ext(base: str, knowledge_path: str, project_name: str) -> str:
    """Return the language the stages after the merge are to take as the primary one.

    The files counted are those outside the tests whose definitions HEAD changed against
    base (change_def_dict), dependency files (_dep_pattern_tuple) left out. A language is
    taken when:

    - none of those files is one the primary language's test command runs
      (edit.runnable_path_list);
    - HEAD adds the first dependency file of the language at the repository root
      (_root_dep_lang_set), its test command is not the primary language's, and it runs
      from there as another language's (edit.other_test_list);
    - most of those files are in it among such languages, a header counted as the
      language header_owner() gives it, the first by name among as many.

    Args:
        base: The commit the change HEAD holds was written against.
        knowledge_path: The analysis of HEAD.
        project_name: The directory name of PROJECT_DIR.

    Returns:
        The language's LANG_DICT key, or "" to keep the primary language: always while
        TEST_COMMAND, TEST_PATH or TEST_TARGET_STYLE in the environment names a value.
        Keeping it is logged (_keep_primary) whenever HEAD adds the first dependency
        file at the root of a language whose test command is not the primary
        language's.
    """
    primary_tuple = LANG_DICT[config.PRIMARY_EXT].test_command_tuple
    new_set = {lang for lang in _root_dep_lang_set(base)
               if not primary_tuple or LANG_DICT[lang].test_command_tuple != primary_tuple}
    if not new_set:
        return ""
    hand_list = config.hand_env_list(config.TEST_SETTING_TUPLE)
    if hand_list:
        return _keep_primary(new_set, f"the language of the change is not looked at as "
                                      f"{list_text(hand_list)} is set")
    pattern_tuple = _dep_pattern_tuple()
    def_dict = {path: name_list for path, name_list in change_def_dict(
        config.PROJECT_DIR, base, integrate.analyse(knowledge_path)["range"],
        project_name).items() if not is_dep_file(path, pattern_tuple)}
    if not def_dict:
        return _keep_primary(new_set, "the primary language is kept as the change touches "
                                      "no definition outside the tests and the dependency "
                                      "files")
    if config.TEST_COMMAND and edit.runnable_path_list(list(def_dict)):
        return _keep_primary(new_set, "the primary language is kept as the change touches "
                                      "files its test command runs")
    root_command_set = {LANG_DICT[ext].test_command_tuple
                        for ext, _, directory in edit.other_test_list() if not directory}
    ext_count_dict = collections.Counter(map(lang_ext, def_dict))
    count_dict: collections.Counter = collections.Counter()
    for ext, count in ext_count_dict.items():
        own_ext = header_owner(ext, ext_count_dict)
        lang = ALIAS_DICT.get(own_ext, own_ext)
        if lang in new_set and LANG_DICT[lang].test_command_tuple in root_command_set:
            count_dict[lang] += count
    if not count_dict:
        return _keep_primary(new_set, "the primary language is kept as no changed file is "
                                      "of one of them whose tests run from there")
    return min(count_dict, key=lambda lang: (-count_dict[lang], lang))


def _keep_primary(new_set: set[str], why: str) -> str:
    """Log that the primary language is kept although HEAD adds a dependency file.

    Args:
        new_set: _root_dep_lang_set()'s languages.
        why: Why the primary language is kept.

    Returns:
        "", change_lang_ext()'s value for keeping it.
    """
    logger.info("HEAD adds a dependency file of %s at the repository root; %s",
                list_text(sorted({LANG_NAME_DICT[lang] for lang in new_set})), why)
    return ""


def _dep_pattern_tuple(lang_config: LangConfig | None = None) -> tuple[str, ...]:
    """Return the patterns of the dependency files of a language, or of every language.

    Args:
        lang_config: The language's settings. None takes every language's.

    Returns:
        The fnmatch patterns of its LangConfig dep_file_tuple and root_target_tuple.
    """
    config_list = [lang_config] if lang_config is not None else list(LANG_DICT.values())
    return tuple(pattern for one in config_list
                 for pattern in (*one.dep_file_tuple, *root_target_pattern_tuple(one)))


def _root_dep_lang_set(base: str) -> set[str]:
    """Return the languages HEAD adds a dependency file of at the repository root.

    A language's dependency files are those _dep_pattern_tuple() gives it.

    Args:
        base: The commit HEAD is compared against.

    Returns:
        The LANG_DICT keys with a dependency file at the root of PROJECT_DIR that HEAD
        holds and base does not, and none at the root of base.
    """
    add_list = [path for path in gitcmd.name_status(config.PROJECT_DIR, base, "HEAD")["added"]
                if "/" not in path]
    if not add_list:
        return set()
    base_list = gitcmd.run(config.PROJECT_DIR, "ls-tree", "--name-only", base).stdout.split("\n")
    lang_set = set()
    for lang, lang_config in LANG_DICT.items():
        pattern_tuple = _dep_pattern_tuple(lang_config)
        if any(is_dep_file(path, pattern_tuple) for path in add_list) \
                and not any(is_dep_file(path, pattern_tuple) for path in base_list):
            lang_set.add(lang)
    return lang_set


def _outer_name_list(range_list: list[tuple], name: str) -> list[str]:
    """Return the definitions of one file that enclose a definition of that file.

    Args:
        range_list: [(start line, end line, name), ...] for the file.
        name: The definition looked for.

    Returns:
        The names whose lines hold every definition called name, without name itself.
    """
    span_list = [(start, end) for start, end, entry_name in range_list if entry_name == name]
    return sorted({entry_name for start, end, entry_name in range_list if entry_name != name
                   and any(start <= inner_start and inner_end <= end
                           for inner_start, inner_end in span_list)})


def _test_line_dict(ref_by_def_dict: dict, range_dict: dict, file: str,
                    name: str) -> dict[str, set]:
    """Gather the test lines that refer to a definition or to one that encloses it.

    Args:
        ref_by_def_dict: {(file, name): {test path: {line, ...}}}, target_dict()'s
            references from the test files.
        range_dict: integrate.analyse()'s "range" for HEAD.
        file: The definition's file, as the analysis names it.
        name: The definition's name.

    Returns:
        {test path: {line, ...}}, empty when no test refers to it.
    """
    line_by_test_dict: dict = {}
    for look_name in [name] + _outer_name_list(range_dict.get(file, []), name):
        for test_path, line_set in ref_by_def_dict.get((file, look_name), {}).items():
            line_by_test_dict.setdefault(test_path, set()).update(line_set)
    return line_by_test_dict


def target_dict(def_dict: dict, ref_list: list[tuple], range_dict: dict,
                project_name: str) -> dict:
    """Split the definitions a change touched by whether a test refers to them.

    A reference to a definition that encloses one counts for the one it encloses.

    Args:
        def_dict: change_def_dict()'s return value.
        ref_list: integrate.analyse()'s "ref" for HEAD.
        range_dict: integrate.analyse()'s "range" for HEAD.
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        {"referenced": [{"file", "name", "tests": [{"path", "lines"}, ...]}, ...],
        "unreferenced": [{"file", "name"}, ...]}, every path relative to the repository
        root and every list sorted.
    """
    ref_by_def_dict: dict = {}
    for from_file, to_file, name, line_list in ref_list:
        from_path = knowledge_to_repo(from_file, project_name)
        if under_test_path([from_path]):
            ref_by_def_dict.setdefault((to_file, name), {}).setdefault(
                from_path, set()).update(line_list)
    split_dict: dict = {"referenced": [], "unreferenced": []}
    for path, name_list in sorted(def_dict.items()):
        file = repo_to_knowledge(path, project_name)
        for name in name_list:
            line_by_test_dict = _test_line_dict(ref_by_def_dict, range_dict, file, name)
            if line_by_test_dict:
                split_dict["referenced"].append({
                    "file": path, "name": name,
                    "tests": [{"path": test_path, "lines": sorted(line_set)}
                              for test_path, line_set in sorted(line_by_test_dict.items())]})
            else:
                split_dict["unreferenced"].append({"file": path, "name": name})
    return split_dict


def test_by_file_dict(path_list: list[str], use_list: list[tuple],
                      project_name: str) -> dict:
    """Return the test files that use each file.

    Args:
        path_list: The repository-relative paths to look the uses up for.
        use_list: integrate.analyse()'s "use".
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        {path: [the test files (under_test_path()) that use anything of it, under its
        own name or another, sorted]}, every path of path_list a key.
    """
    test_set_dict: dict = {path: set() for path in path_list}
    for from_file, to_file in use_list:
        to_path = knowledge_to_repo(to_file, project_name)
        from_path = knowledge_to_repo(from_file, project_name)
        if to_path in test_set_dict and under_test_path([from_path]):
            test_set_dict[to_path].add(from_path)
    return {path: sorted(test_set) for path, test_set in test_set_dict.items()}


def baseline_of(test_dict: dict, is_no_test_green: bool = False) -> frozenset[str] | None:
    """Return the baseline before the stage starts.

    A run of several test commands is read command by command (edit.part_dict_of) and
    the sets are put together (edit.join_baseline): a command whose tests did not run
    is left out of what the stage is held to, and the others are not. Another
    language's command that found no test to run is held by edit.NO_TEST_BASELINE. A
    command whose run was red and ended before it had run every test leaves no
    baseline at all (_is_red_stop_run), and so does one whose run says nothing of how
    many tests it ran (_is_unread_run).

    Args:
        test_dict: run_test()'s return value over HEAD.
        is_no_test_green: Whether a primary command that found no test to run is held
            by edit.NO_TEST_BASELINE.

    Returns:
        The names of the failing tests with the entry of the tests the run ran
        (_part_baseline_of()), for a run of several commands put together as
        edit.join_baseline() returns them, or None when the tests of the primary
        language's command did not run, failed without naming a test or ended before
        every test was run, when that command was not part of the run, when any
        command's run was red and ended before every test was run, and when any
        command's run says nothing of how many tests it ran.
    """
    if _is_unread_run(test_dict):
        return None
    part_dict = edit.part_dict_of(test_dict)
    if not part_dict:
        return _part_baseline_of(test_dict, is_no_test_green)
    primary_name = getattr(test_dict, "primary_name", "")
    by_command_dict = {
        command: _part_baseline_of(part, is_no_test_green or command != primary_name)
        for command, part in part_dict.items()}
    if by_command_dict.get(primary_name) is None or _is_red_stop_run(test_dict):
        return None
    return edit.join_baseline(by_command_dict, primary_name)


def _is_unread_run(test_dict: dict) -> bool:
    """Say whether a test command of a run says nothing of how many tests it ran.

    Such a command is one whose language's runs do say it
    (edit.RunResult.is_ran_unread).

    Args:
        test_dict: run_test()'s return value.

    Returns:
        True when one of the run's commands is such a command.
    """
    return any(getattr(part, "is_ran_unread", False)
               for part in edit.part_list_of(test_dict))


def _is_red_stop_run(test_dict: dict) -> bool:
    """Say whether a test command of a run was red and ended before it ran every test.

    A command ended early when edit.early_stop_run() says True of its result.

    Args:
        test_dict: run_test()'s return value.

    Returns:
        True when one of the run's commands failed and ended early.
    """
    return any(part["status"] == "failed" and edit.early_stop_run(part)
               for part in edit.part_list_of(test_dict))


def _part_baseline_of(test_dict: dict, is_no_test_green: bool) -> frozenset[str] | None:
    """Return baseline_of()'s set for the run of one test command.

    Args:
        test_dict: The command's own result.
        is_no_test_green: Whether a run that found no test is held by
            edit.NO_TEST_BASELINE.

    Returns:
        The failing names (edit.fail_entry_set), empty when every test passes, each
        with the entry of the tests the run ran (edit.ran_entry_set),
        edit.NO_TEST_BASELINE for a run that found no test where is_no_test_green is
        True, or None when the tests did not run, failed without naming a test, or
        the command ended before it had run every test (edit.early_stop_run).
    """
    if test_dict["status"] == "passed":
        return edit.ran_entry_set(test_dict)
    if is_no_test_green and edit.no_test_run(test_dict):
        return edit.NO_TEST_BASELINE
    if edit.early_stop_run(test_dict):
        logger.info("the tests ended before every test was run, so the failing ones "
                    "are no baseline: %s", list_text(test_dict["failed"]))
        return None
    if test_dict["status"] == "failed" and test_dict["failed"]:
        return edit.fail_entry_set(test_dict["failed"]) | edit.ran_entry_set(test_dict)
    return None


def baseline_or_skip(project_dir: str, def_dict: dict,
                     baseline_list: list[str] | None = None) -> tuple[frozenset[str], str]:
    """Return the baseline before the stage starts, or why the stage calls no model.

    The tests are run only where there is something to write a test for and no baseline
    was passed. The tests' directory is made where the tests are reached by path and it
    is not there. A run that found no test to run is a baseline with nothing failing.

    Args:
        project_dir: The repository.
        def_dict: change_def_dict()'s return value.
        baseline_list: The baseline before the stage, as list_target() returns it in
            "baseline". None runs the tests.

    Returns:
        (baseline_list as a set, or else baseline_of()'s set, "") or
        (an empty set, one of the SKIP_* and _SKIP_* texts).
    """
    reason = skip_reason(project_dir, def_dict)
    if reason:
        return frozenset(), reason
    edit.make_test_path()
    if baseline_list is not None:
        return frozenset(baseline_list), ""
    test_dict = edit.run_test()
    baseline_set = baseline_of(test_dict)
    if baseline_set is not None:
        return baseline_set, ""
    if edit.no_test_run(edit.primary_part(test_dict)) and not _is_red_stop_run(test_dict) \
            and not _is_unread_run(test_dict):
        return baseline_of(test_dict, is_no_test_green=True) or frozenset(), ""
    return frozenset(), SKIP_NO_BASELINE


def _setting_skip_reason() -> str:
    """Say why the stage runs no test under the settings as they are, or "".

    Returns:
        SKIP_NO_TEST_COMMAND when RUN_TEST is off or the primary language has no test
        command (edit.has_test_command), _SKIP_NO_TEST_RULE when no path can be a test
        (has_test_rule), or "".
    """
    if not edit.has_test_command():
        return SKIP_NO_TEST_COMMAND
    if not has_test_rule():
        return _SKIP_NO_TEST_RULE
    return ""


def repo_test_file_list(project_dir: str) -> list[str]:
    """Return the test files of a repository as its working tree holds them now.

    Args:
        project_dir: The repository.

    Returns:
        The files of gitcmd.work_tree_file_list() that are test files
        (under_test_path()), in its order. Empty for a directory that is not a
        repository.
    """
    return under_test_path(gitcmd.work_tree_file_list(project_dir) or [])


def _start_skip_reason(project_dir: str) -> str:
    """Say why the stage runs no test whatever a change touched, or "".

    Args:
        project_dir: The repository. "" reads the settings alone.

    Returns:
        _setting_skip_reason()'s text, _SKIP_NO_TEST_FILE when the repository holds no
        test file and the languages whose tests run have no dependency file the stage
        could write (edit.dep_file_pattern_set), or "".
    """
    reason = _setting_skip_reason()
    if reason or not project_dir:
        return reason
    if not edit.dep_file_pattern_set() and not repo_test_file_list(project_dir):
        return _SKIP_NO_TEST_FILE
    return ""


def is_stage_on() -> bool:
    """Say whether the stage that writes tests is on and can run the tests.

    Returns:
        True when TEST_CREATE is on and neither a setting nor what PROJECT_DIR holds
        keeps the stage from running tests (_start_skip_reason). What the change
        touched is not looked at.
    """
    return bool(config.TEST_CREATE) and not _start_skip_reason(config.PROJECT_DIR)


def skip_reason(project_dir: str, def_dict: dict) -> str:
    """Say why the stage runs no test and calls no model, or "" when it does both.

    Args:
        project_dir: The repository.
        def_dict: change_def_dict()'s return value.

    Returns:
        One of the SKIP_* and _SKIP_* texts, or "".
    """
    reason = _start_skip_reason(project_dir)
    if reason:
        return reason
    if not def_dict:
        return _SKIP_NO_DEFINITION
    return ""


# == What the machine checks before a run is allowed to end ==


def run_reason(result_dict: dict, baseline_set: frozenset[str],
               is_no_test_green: bool = False) -> str:
    """Return what one test run says that the baseline does not, or "" when nothing.

    Args:
        result_dict: run_test()'s return value.
        baseline_set: baseline_of()'s set for the run before the stage started.
        is_no_test_green: Whether a primary command that found no test to run, as it
            found none when the baseline was read, says nothing.

    Returns:
        The reasons of the test commands that have one (edit.compare_part_list), or
        "". A command that failed and named no test, and one that did not run, are
        both a reason.
    """
    return "; ".join(reason for reason in (
        _part_run_reason(part, baseline_set, command)
        for command, part, _ in edit.compare_part_list(
            result_dict, baseline_set, hold_primary_no_test=not is_no_test_green))
        if reason)


def whole_run_reason(run_dict: dict, baseline_set: frozenset[str]) -> str:
    """Return what the last run of the whole suite says that the baseline does not.

    The run is read as the check reads the whole run it makes
    (ref_and_test_reason_list): a primary command that found no test to run, as it
    found none when the stage started, is no reason where the stage had left the
    repository as it found it when the run ended.

    Args:
        run_dict: edit.last_whole_run()'s value.
        baseline_set: baseline_of()'s set for the run before the stage started.

    Returns:
        run_reason()'s text.
    """
    return run_reason(run_dict["result"], baseline_set,
                      is_no_test_green=not run_dict.get("change"))


def _part_run_reason(result_dict: dict, baseline_set: frozenset[str], command: str) -> str:
    """Return run_reason()'s reason for the run of one test command.

    Args:
        result_dict: The command's own result.
        baseline_set: baseline_of()'s set for the run before the stage started.
        command: The command, as edit.compare_part_list() names it.

    Returns:
        One reason, or "". A failing run that ended before every test was run
        (edit.early_stop_run) is a reason whatever tests it named. A run that passed,
        or failed with no test the baseline does not hold, is held to the tests that
        ran when the baseline was read (edit.unrun_reason).
    """
    shown = edit.command_text(command)
    if result_dict["status"] == "did not run":
        return _TEST_NO_RUN_REASON.format(command=shown, why=result_dict["failed_excerpt"])
    unrun = functools.partial(edit.unrun_reason, result_dict, baseline_set, command,
                              _TEST_UNRUN_REASON, _TEST_FEWER_RUN_REASON,
                              _TEST_UNREAD_REASON, _TEST_SHORT_REASON)
    if result_dict["status"] != "failed":
        return unrun()
    break_list = sorted(set(edit.fail_name_list(
        edit.fail_entry_set(result_dict["failed"]) - baseline_set)))
    if break_list:
        return _TEST_BREAK_REASON.format(command=shown, tests=list_text(break_list),
                                         why=result_dict["failed_excerpt"])
    if edit.early_stop_run(result_dict):
        return _TEST_EARLY_STOP_REASON.format(
            command=shown, why=edit.early_stop_why(result_dict),
            tests=list_text(result_dict["failed"]) or _NO_TEST_NAME_TEXT)
    if result_dict["failed"]:
        return unrun()
    return _TEST_REASON.format(command=shown, why=result_dict["failed_excerpt"])


def own_test_tuple(project_dir: str) -> tuple[list[str], list[str]]:
    """Return the test files the stage changed, split by whether they are still there.

    Args:
        project_dir: The repository.

    Returns:
        (the paths that are files now, the paths that are gone), both sorted. A path
        where something other than a file stands counts as gone when the index names it,
        and as neither otherwise.
    """
    path_list = sorted(edit.test_change_set())
    exist_list = [path for path in path_list
                  if os.path.isfile(os.path.join(project_dir, path))]
    other_list = [path for path in path_list if path not in exist_list
                  and os.path.lexists(os.path.join(project_dir, path))]
    keep_set = set(gitcmd.tracked_path_list(project_dir, other_list))
    return exist_list, [path for path in path_list if path not in exist_list
                        and (path not in other_list or path in keep_set)]


def _case_number_set(text: str) -> set[int]:
    """Return the numbers of the case ids one text holds.

    Examples:
        "def test_h_case_2():  # case_10" -> {2, 10}
        "testcase_3"                      -> set()

    Args:
        text: The text to read.

    Returns:
        The numbers.
    """
    return {int(number) for number in _CASE_NUMBER_RE.findall(text)}


def file_case_number_set(project_dir: str, path: str) -> set[int]:
    """Return the numbers of the case ids one file holds.

    Args:
        project_dir: The repository.
        path: The file, relative to the repository root.

    Returns:
        The numbers, empty for a file that cannot be read and for one that is not a
        regular file of at most tmpfile.READ_MAX_BYTE bytes (tmpfile.check_file).
    """
    file_abs = os.path.join(project_dir, path)
    try:
        tmpfile.check_file(file_abs)
        return _case_number_set(read_source_text(file_abs))
    except (OSError, ValueError):
        return set()


def _case_number(case_dict: dict) -> int:
    """Return the number in one case's id."""
    return int(case_dict["id"][len(CASE_ID_PREFIX):])


def next_case_number(project_dir: str, test_file_list: list[str]) -> int:
    """Return the number the first case of a run takes.

    Args:
        project_dir: The repository, as it is when the run starts.
        test_file_list: The repository's test files.

    Returns:
        One more than the highest number a case id in those files holds, 1 when none
        holds one.
    """
    number_set = set().union(*(file_case_number_set(project_dir, path)
                               for path in test_file_list))
    return max(number_set, default=0) + 1


def case_place_dict(project_dir: str, case_list: list[dict]) -> dict:
    """Return where each case's id is, among the test files the stage changed.

    Only the test files the test command runs (edit.runnable_path_list()) are read.

    Args:
        project_dir: The repository.
        case_list: parse_case_list()'s cases.

    Returns:
        {case id: [repository-relative path, ...]}, the paths sorted, empty for an id
        no such file holds. An id followed by a digit is another id.
    """
    number_dict = {path: file_case_number_set(project_dir, path)
                   for path in edit.runnable_path_list(own_test_tuple(project_dir)[0])}
    return {case_dict["id"]: sorted(path for path, number_set in number_dict.items()
                                    if _case_number(case_dict) in number_set)
            for case_dict in case_list}


def decision_id_set(case_list: list[dict], entry_list: list[dict] | None = None) -> set[str]:
    """Return the ids of the cases a recorded entry names in its "what".

    Args:
        case_list: parse_case_list()'s cases.
        entry_list: The entries to read. None reads what record_decision() and
            record_gap() have recorded.

    Returns:
        The ids.
    """
    if entry_list is None:
        entry_list = integrate.decision_list() + integrate.gap_list()
    number_set = _case_number_set("\n".join(entry_dict["what"] for entry_dict in entry_list))
    return {case_dict["id"] for case_dict in case_list
            if _case_number(case_dict) in number_set}


def not_own_change_list(start_file_list: list[str], may_write: Callable[[str], bool]
                        ) -> list[str]:
    """Return the files there at a stage's start that it changed and may not write.

    A file counts however it was written.

    Args:
        start_file_list: Every file of the repository when the run started.
        may_write: The rule the stage's tools write by.

    Returns:
        The paths, sorted: of edit.own_change_set() without what a command the machine
        ran left there (edit.command_write_set), those start_file_list holds and
        may_write refuses.
    """
    return sorted(path for path in (edit.own_change_set() - edit.command_write_set())
                  & set(start_file_list) if not may_write(path))


def reason_list(project_dir: str, output_dir: str, start_dict: dict,
                baseline_set: frozenset[str], exclude_list: list[str] | None,
                case_list: list[dict],
                check_before: collections.Counter | None = None,
                start_file_list: list[str] | None = None,
                may_write: Callable[[str], bool] | None = None) -> list[str]:
    """Return what is still wrong with what the stage left, in the model's words.

    The analysis is made again over the working tree. Nothing here calls a model.

    Args:
        project_dir: The repository.
        output_dir: Where the analysis is written.
        start_dict: integrate.analyse()'s return value for HEAD.
        baseline_set: baseline_of()'s set for the run before the stage started.
        exclude_list: The extra names kept out of the analysis, as the build takes them.
        case_list: parse_case_list()'s cases.
        check_before: edit.run_check() when the stage started. None leaves the static
            check out.
        start_file_list: Every file of the repository when the run started. None, or
            may_write None, leaves out the files changed that are not the stage's to
            write (not_own_change_list).
        may_write: may_write_of()'s return value, the rule the stage's tools write by.

    Returns:
        One line per reason, empty when there is nothing left.
    """
    line_list = []
    if start_file_list is not None and may_write is not None:
        not_own_list = not_own_change_list(start_file_list, may_write)
        if not_own_list:
            line_list.append(NOT_OWN_REASON.format(paths=list_text(not_own_list)))
    place_dict = case_place_dict(project_dir, case_list)
    open_list = [case_id for case_id, path_list in place_dict.items()
                 if not path_list and case_id not in decision_id_set(case_list)]
    if open_list:
        line_list.append(_CASE_REASON.format(ids=list_text(open_list)))
    exist_list, delete_list = own_test_tuple(project_dir)
    if delete_list:
        line_list.append(_DELETE_REASON.format(paths=list_text(delete_list)))
    return line_list + ref_and_test_reason_list(project_dir, output_dir, start_dict,
                                                baseline_set, exclude_list, exist_list,
                                                check_before=check_before)


def ref_reason_tuple(project_dir: str, output_dir: str, start_dict: dict,
                     exclude_list: list[str] | None,
                     check_before: collections.Counter | None = None
                     ) -> tuple[list[str], bool]:
    """Return what the static check and the references say is wrong with a change.

    The analysis is made again over the working tree. Nothing here calls a model.

    Args:
        project_dir: The repository.
        output_dir: Where the analysis is written.
        start_dict: integrate.analyse()'s return value for the state before the change.
        exclude_list: The extra names kept out of the analysis, as the build takes them.
        check_before: edit.run_check() before the change. None leaves the static check
            out.

    Returns:
        (one line per reason, in the model's words: what the static check finds that it
        did not find before, the analysis that could not be made again, and the calls
        that no longer resolve; whether the analysis was made again).
    """
    line_list = integrate.check_reason(check_before)
    try:
        path, _, project_name, _ = ensure_knowledge(project_dir, output_dir, rebuild=True,
                                                    with_doc=False,
                                                    exclude_list=exclude_list)
    except (ValueError, RuntimeError) as e:
        logger.warning("the analysis could not be made again (%s), so what the "
                       "references say now is not known", e)
        return line_list + [_ANALYSIS_REASON.format(why=e)], False
    break_ref_list = integrate.broken_list_of(start_dict, integrate.analyse(path), project_name)
    if break_ref_list:
        line_list.append(_BROKEN_REASON.format(
            calls=list_text([f"{entry_dict['name']} in {entry_dict['file']}"
                             for entry_dict in break_ref_list])))
    return line_list, True


def ref_and_test_reason_list(project_dir: str, output_dir: str, start_dict: dict,
                             baseline_set: frozenset[str], exclude_list: list[str] | None,
                             own_test_list: list[str],
                             check_before: collections.Counter | None = None) -> list[str]:
    """Return what the references, the static check and the tests say a stage left wrong.

    The analysis is made again over the working tree. Nothing here calls a model.

    Args:
        project_dir: The repository.
        output_dir: Where the analysis is written.
        start_dict: integrate.analyse()'s return value for HEAD.
        baseline_set: baseline_of()'s set for the run before the stage started.
        exclude_list: The extra names kept out of the analysis, as the build takes them.
        own_test_list: The test files the stage changed that are there now. Those the
            test command runs (edit.runnable_path_list()) are run alone before the
            whole suite is, where the build tool takes paths, and each of them holding
            a case is checked to be one the whole suite runs (edit.in_whole_run()).
        check_before: edit.run_check() when the stage started. None leaves the static
            check out.

    Returns:
        One line per reason, in the model's words: what the static check finds that it
        did not find before, the analysis that could not be made again, the calls that
        no longer resolve, what one test run says that the baseline does not (the tests
        of the whole suite that ran when the stage started and run no more included),
        and the files holding a case that the whole suite does not run, then the file
        holding a case that could not be put back after that check (_PUT_BACK_REASON),
        which ends the checks. Empty when there is nothing. A whole suite that found no test to run, as it found none when the
        stage started, is no reason where the stage left the repository as it found it
        (edit.stage_change_hash_dict).
    """
    line_list, has_analysis = ref_reason_tuple(project_dir, output_dir, start_dict,
                                               exclude_list, check_before)
    if not has_analysis:
        return line_list

    # The files the stage changed that the test command runs are run alone first, where
    # the build tool takes paths. A run of files that hold no case and found no test
    # to run says nothing
    run_list = edit.runnable_path_list(own_test_list)
    case_path_list = [path for path in run_list if file_case_number_set(project_dir, path)]
    if run_list and config.TEST_TARGET_STYLE != "none":
        own_dict = edit.run_test(run_list)
        reason = "" if edit.no_test_run(own_dict) and not case_path_list \
            else run_reason(own_dict, baseline_set)
        if reason:
            return line_list + [_OWN_TEST_REASON.format(paths=list_text(run_list),
                                                        reason=reason)]
    whole_dict = edit.run_test()
    reason = run_reason(whole_dict, baseline_set,
                        is_no_test_green=not edit.stage_change_hash_dict())
    if reason:
        return line_list + [f"- {reason}"]
    # Each file holding a case has to be one the whole suite runs. A file that could not
    # be put back after the run with a line added to it ends the checks
    unrun_list, put_back_line = [], ""
    for path in case_path_list:
        try:
            if edit.in_whole_run(path, whole_dict) is False:
                unrun_list.append(path)
        except OSError as e:
            put_back_line = _PUT_BACK_REASON.format(path=path, why=e)
            break
    if unrun_list:
        line_list.append(_UNRUN_REASON.format(command=shlex.join(edit.primary_test_command()),
                                              paths=list_text(unrun_list)))
    if put_back_line:
        line_list.append(put_back_line)
    return line_list


class RestoreError(RuntimeError):
    """The working tree could not be put back after the tests were run over another tree.

    The working tree holds that tree with the test files laid over it.
    """


def _base_dep_path_list(project_dir: str, base: str) -> list[str]:
    """Return the dependency files the working tree holds that a commit does not hold so.

    Args:
        project_dir: The repository.
        base: The commit the change was written against.

    Returns:
        The files matching edit.test_build_file_pattern_set() among the paths HEAD adds
        or changes against the commit and those among the paths the run has added or
        changed since (edit.change_dep_path_list), those the working tree holds now,
        sorted.
    """
    pattern_set = edit.test_build_file_pattern_set()
    if not pattern_set:
        return []
    status_dict = gitcmd.name_status(project_dir, base, "HEAD")
    path_set = {path for path in status_dict["added"] + status_dict["modified"]
                if is_dep_file(path, pattern_set)} \
        | set(edit.change_dep_path_list(pattern_set))
    return sorted(path for path in path_set
                  if os.path.isfile(os.path.join(project_dir, path)))


def _base_drop_path_list(project_dir: str, base: str) -> list[str]:
    """Return the dependency files a commit holds that the working tree does not hold.

    Args:
        project_dir: The repository.
        base: The commit the change was written against.

    Returns:
        The files of the commit matching edit.test_build_file_pattern_set() that the
        working tree holds nothing at now, sorted.

    Raises:
        RuntimeError: When the files of the commit could not be read.
    """
    pattern_set = edit.test_build_file_pattern_set()
    if not pattern_set:
        return []
    proc = gitcmd.run(project_dir, "ls-tree", "-r", "-z", "--full-tree", "--name-only",
                      base, text=False)
    if proc.returncode != 0:
        raise RuntimeError(f"could not read the files of {gitcmd.short(base)}: "
                           f"{gitcmd.shown_text(proc.stderr).strip()}")
    path_list = [gitcmd.shown_text(raw_path) for raw_path in proc.stdout.split(b"\0")
                 if raw_path]
    return sorted(path for path in path_list
                  if is_dep_file(path, pattern_set)
                  and not os.path.lexists(os.path.join(project_dir, path)))


def _each_file_run_dict(run_list: list[str], baseline_set: frozenset[str]) -> dict:
    """Run some test files one at a time and put what the runs said together.

    Args:
        run_list: The test files, relative to the repository root.
        baseline_set: baseline_of()'s set for the run before the stage started.

    Returns:
        {"status": "failed" when a run failed, "passed" when every run passed, "did
        not run" otherwise, "failed": the failing names of every run, "is_part":
        whether one of the runs is one _is_part_run() says True of, "left_out": [],
        "is_name_short": whether one of the runs is one _is_name_short_run() says True
        of}.
    """
    file_result_list = [edit.run_test_as_is([path]) for path in run_list]
    status_set = {result_dict["status"] for result_dict in file_result_list}
    if "failed" in status_set:
        status = "failed"
    else:
        status = "passed" if status_set == {"passed"} else "did not run"
    return {"status": status,
            "failed": [name for result_dict in file_result_list
                       for name in result_dict["failed"]],
            "is_part": any(_is_part_run(result_dict, [path], baseline_set)
                           for result_dict, path in zip(file_result_list, run_list)),
            "left_out": [],
            "is_name_short": any(_is_name_short_run(result_dict)
                                 for result_dict in file_result_list)}


def _is_name_short_run(result_dict: dict) -> bool:
    """Say whether a command of a run counts failing tests and names none of them.

    The command's result says so in edit.RunResult.is_name_short, for a run given paths.

    Args:
        result_dict: edit.run_test()'s return value for the run.

    Returns:
        True when one of the run's commands is such a command.
    """
    return any(getattr(part, "is_name_short", False)
               for part in edit.part_list_of(result_dict))


def _no_run_line_list(result_dict: dict) -> list[str]:
    """Return what a run of the whole suite says it left out.

    The tests of what it left out are not named, whether they fail or not.

    Args:
        result_dict: edit.run_test()'s return value for the run.

    Returns:
        edit.early_stop_line_list()'s lines where the test command takes no path
        (config.TEST_TARGET_STYLE is "none") for a run edit.early_stop_run() says True
        of and edit.fail_stop_run() does not, as one is that could not build a module or
        left one out. [] for any other.
    """
    if config.TEST_TARGET_STYLE != "none" or not edit.early_stop_run(result_dict) \
            or edit.fail_stop_run(result_dict):
        return []
    return edit.early_stop_line_list(result_dict)


def _is_part_run(result_dict: dict, run_list: list[str],
                 baseline_set: frozenset[str]) -> bool:
    """Say whether the failing names of a run over test files are of a part of their tests.

    Args:
        result_dict: edit.run_test()'s return value for the run.
        run_list: The test files that were run.
        baseline_set: baseline_of()'s set for the run before the stage started.

    Returns:
        True for a run that stopped at a count of failing tests (edit.fail_stop_run) and
        names a failing test that is none of the files and none of baseline_set. A file
        named alone is one the run could not load.
    """
    return edit.fail_stop_run(result_dict) \
        and any(name not in run_list and name not in baseline_set
                for name in result_dict["failed"])


def base_run_dict(project_dir: str, base: str, path_list: list[str],
                  baseline_set: frozenset[str]) -> dict:
    """Run the test files the stage wrote over the tree of another commit.

    What the working tree holds is written to a tree first, the commit's tree is put in
    the working tree with those test files and the dependency files the change holds
    (_base_dep_path_list) laid over it and the dependency files the change removed
    (_base_drop_path_list) taken out of it, the ones of the test files the test command
    runs (edit.runnable_path_list()) are run against the package cache as it is, and the
    tree written first is put back. A run of several files that ended before every test
    was run (edit.early_stop_run) is made again one file at a time where the test
    command takes paths (config.TEST_TARGET_STYLE is not "none"). All of it goes through
    an index file of this call's own (gitcmd.work_tree_index()). HEAD and the
    repository's index are left as they are, and so are what the stage's own test runs
    left behind and the files of the working tree the tree written first does not hold
    that the commit's tree is written over (integrate.keep_file).

    The flow:
    1. The test files to run, and the dependency files to lay over and take out
    2. The commit's tree in the working tree, the tests run, the tree put back
       (_lay_base, _base_test_dict, _put_back)
    3. What the run says there (_base_status_dict)

    Args:
        project_dir: The repository, holding what the stage wrote.
        base: The commit the change was written against.
        path_list: The test files the stage wrote, relative to the repository root.
        baseline_set: baseline_of()'s set for the run before the stage started.

    Returns:
        {"status": BASE_RED when the run fails there with a test the baseline does not
        hold or with no test named, or could not load a file the tests under its
        directory need there (edit.load_fail_run), every file it ran lies under that
        directory and one of them is not of that file's name, BASE_GREEN when it passes,
        BASE_UNKNOWN otherwise (a run that reached no verdict, one that failed naming
        only tests the baseline holds, and a run of a command that counts failing tests
        and names none of them, _is_name_short_run), "failed": the names of the tests
        failing there, each as many times as it failed beyond the times the baseline
        holds (edit.fail_entry_set), sorted; for a run that could not load such a file,
        each file it ran, named alone, but those of the name of the file it could not
        load, "is_part": whether those names are of a part of the tests alone
        (_is_part_run, over the last runs made), "left_out": the lines of the run that
        say what it left out, a test of which the run does not name (_no_run_line_list,
        over the last runs made), "is_no_run": whether the tests reached no verdict
        there, or counted failing tests and named none of them, and the status is not
        BASE_RED}. "", [], False, [] and False when path_list holds no file the test
        command runs. BASE_UNKNOWN, [] and "is_no_run" True, with the commit's tree not
        put in place and no test run, when a file of the working tree in its way cannot
        be kept (integrate.KeepError), which is logged.

    Raises:
        RestoreError: When the tree written first could not be put back.
        RuntimeError: When, before the tests were run, the files of the commit could not
            be read, a tree could not be written or put in place, or a file could not be
            laid over the commit's tree or taken out of it.
    """
    # == Step 1: the test files to run, and the dependency files to lay over and take out ==
    run_list = edit.runnable_path_list(path_list)
    if not run_list:
        return {"status": "", "failed": [], "is_part": False, "left_out": [],
                "is_no_run": False}
    dep_path_list = _base_dep_path_list(project_dir, base)
    drop_path_list = _base_drop_path_list(project_dir, base)

    # == Step 2: the commit's tree in the working tree, the tests run, the tree put back ==
    with contextlib.ExitStack() as stack:
        tree, env_dict = stack.enter_context(gitcmd.work_tree_index(project_dir))
        if not tree:
            raise RuntimeError("could not write the working tree to a tree")
        try:
            stack.enter_context(integrate.keep_file(project_dir, base, tree, env_dict))
        except integrate.KeepError as e:
            logger.warning("the tests are not run over %s: %s", gitcmd.short(base), e)
            return {"status": BASE_UNKNOWN, "failed": [], "is_part": False,
                    "left_out": [], "is_no_run": True}
        try:
            _lay_base(project_dir, base, tree, env_dict, path_list + dep_path_list,
                      drop_path_list)
            test_dict = _base_test_dict(run_list, baseline_set)
        finally:
            _put_back(project_dir, tree, env_dict)

    # == Step 3: what the run says there ========================
    return _base_status_dict(test_dict, run_list, baseline_set)


def _lay_base(project_dir: str, base: str, tree: str, env_dict: dict[str, str],
              lay_path_list: list[str], drop_path_list: list[str]) -> None:
    """Put a commit's tree in the working tree with some files of another tree over it.

    Args:
        project_dir: The repository.
        base: The commit whose tree is put in place.
        tree: The tree the laid files are taken from.
        env_dict: The settings naming the index file the run goes through.
        lay_path_list: The files laid over the commit's tree, relative to the
            repository root.
        drop_path_list: The files taken out of the commit's tree, relative to the
            repository root.

    Raises:
        RuntimeError: When the commit's tree could not be put in place, or a file could
            not be laid over it or taken out of it.
    """
    integrate.place(project_dir, base, env_dict)
    proc = gitcmd.run(project_dir, "checkout", tree, "--",
                      *[f":(literal){path}" for path in lay_path_list],
                      env_dict=env_dict)
    if proc.returncode != 0:
        raise RuntimeError(f"could not lay the test files over "
                           f"{gitcmd.short(base)}: "
                           f"{(proc.stderr or proc.stdout).strip()}")
    if drop_path_list:
        proc = gitcmd.run(project_dir, "rm", "-f", "-q", "--",
                          *[f":(literal){path}" for path in drop_path_list],
                          env_dict=env_dict)
        if proc.returncode != 0:
            raise RuntimeError(f"could not take the dependency files the change "
                               f"removed out of {gitcmd.short(base)}: "
                               f"{(proc.stderr or proc.stdout).strip()}")


def _base_test_dict(run_list: list[str], baseline_set: frozenset[str]) -> dict:
    """Run some test files over the tree in place, leaving the stage's test state alone.

    A run of several files that ended before every test was run (edit.early_stop_run) is
    made again one file at a time where the test command takes paths
    (config.TEST_TARGET_STYLE is not "none").

    Args:
        run_list: The test files the test command runs, relative to the repository root.
        baseline_set: baseline_of()'s set for the run before the stage started.

    Returns:
        The run's result with "is_part", "left_out" and "is_name_short", or
        _each_file_run_dict()'s value for a run made again one file at a time.
    """
    with edit.keep_test_state():
        test_dict = edit.run_test_as_is(run_list)
        if len(run_list) > 1 and config.TEST_TARGET_STYLE != "none" \
                and edit.early_stop_run(test_dict):
            test_dict = _each_file_run_dict(run_list, baseline_set)
        else:
            test_dict = {**test_dict,
                         "is_part": _is_part_run(test_dict, run_list, baseline_set),
                         "left_out": _no_run_line_list(test_dict),
                         "is_name_short": _is_name_short_run(test_dict)}
    return test_dict


def _base_status_dict(test_dict: dict, run_list: list[str],
                      baseline_set: frozenset[str]) -> dict:
    """Read what a run over the tree of another commit says, as base_run_dict() returns it.

    Args:
        test_dict: _base_test_dict()'s return value.
        run_list: The test files that were run.
        baseline_set: baseline_of()'s set for the run before the stage started.

    Returns:
        base_run_dict()'s {"status", "failed", "is_part", "left_out", "is_no_run"} for a
        run that was made.
    """
    load_fail_dir, _, load_fail_name = edit.load_fail_run(test_dict).rpartition("/")
    load_fail_list = [path for path in run_list if path.rpartition("/")[2] != load_fail_name]
    is_load_fail = bool(load_fail_name) and bool(load_fail_list) and all(
        not load_fail_dir or path.startswith(load_fail_dir + "/") for path in run_list)
    fail_list = edit.fail_name_list(edit.fail_entry_set(
        load_fail_list if is_load_fail else test_dict["failed"]) - baseline_set)
    is_name_short = test_dict["is_name_short"] and not is_load_fail
    if test_dict["status"] == "passed":
        status = BASE_GREEN
    elif (test_dict["status"] == "failed" and not is_name_short
          and (fail_list or not test_dict["failed"])) or is_load_fail:
        status = BASE_RED
    else:
        status = BASE_UNKNOWN
    return {"status": status, "failed": fail_list, "is_part": test_dict["is_part"],
            "left_out": test_dict["left_out"],
            "is_no_run": (test_dict["status"] not in ("passed", "failed") or is_name_short)
            and not is_load_fail}


def _put_back(project_dir: str, tree: str, env_dict: dict[str, str]) -> None:
    """Put the tree base_run_dict() wrote first back in the working tree.

    Args:
        project_dir: The repository.
        tree: The tree the working tree held before the run.
        env_dict: The settings naming the index file the run goes through.

    Raises:
        RestoreError: When the tree could not be put in place.
    """
    try:
        integrate.place(project_dir, tree, env_dict)
    except RuntimeError as e:
        raise RestoreError(f"the working tree could not be put back to what it held: "
                           f"{e}") from e


# == The work list and the stage the model runs ==========


def _request_of(request: str) -> str:
    """Return what the stage is asked.

    Args:
        request: The request the change was made for, or "".

    Returns:
        The stage's request, the change's own cut to config.DISPATCH_TASK_MAX_CHAR.
    """
    return ask_text(request, _REQUEST_TEXT, _NO_REQUEST_TEXT)


def ask_text(request: str, request_text: str, no_request_text: str) -> str:
    """Return what a stage is asked, with the request or without one.

    Args:
        request: The request the change was made for, or "".
        request_text: What is asked when there is a request, with {request}.
        no_request_text: What is asked when there is none.

    Returns:
        request_text holding the request cut to config.DISPATCH_TASK_MAX_CHAR, or
        no_request_text for a request that is empty or blank.
    """
    if not request.strip():
        return no_request_text
    return request_text.format(request=cut(request.strip(), config.DISPATCH_TASK_MAX_CHAR))


def _make_stage(tool_list: list, instruction: str, stage: str) -> RLM:
    """Build one of the two stages on the stage's model, output limit and turn limit.

    Args:
        tool_list: The tool callables the stage is handed.
        instruction: The stage's assembled instructions.
        stage: The stage's name in the turn log ("test_case" or "test_create").

    Returns:
        The stage, ready to run.
    """
    return RLM(llm=make_llm(config.TEST_CREATE_MODEL, config.TEST_CREATE_MAX_TOKEN,
                            "TEST_CREATE_MODEL", is_turn_reply=True,
                            think_level=config.TEST_CREATE_THINK_LEVEL),
               tool_list=tool_list, instruction=instruction, stage=stage,
               max_turn=config.TEST_CREATE_MAX_TURN)


def branch_list_of(project_dir: str, base: str) -> list[dict]:
    """Return what each run that made the change was asked, read off its commit.

    Args:
        project_dir: The repository, whose HEAD holds the change.
        base: The commit the change was written against.

    Returns:
        [{"asked", "tests_changed"}, ...], oldest first, "asked" cut to
        config.DISPATCH_TASK_MAX_CHAR. The commits whose stage is one of
        integrate.STAGE_TUPLE are left out, and so is what a run said it did.
    """
    record_list = integrate.commit_record_list(project_dir, base, "HEAD", own_only=True)
    return [{"asked": cut(entry_dict["asked"], config.DISPATCH_TASK_MAX_CHAR),
             "tests_changed": entry_dict["tests_changed"].split()}
            for entry_dict in record_list
            if entry_dict["stage"] not in integrate.STAGE_TUPLE]


def case_work_dict(test_file_list: list[str], request: str, branch_list: list[dict],
                   find_dict: dict, signature_dict: dict, project_name: str) -> dict:
    """Build the first stage's work dict.

    Args:
        test_file_list: The repository's test files.
        request: The request the change was made for, or "".
        branch_list: branch_list_of()'s return value.
        find_dict: target_dict()'s return value.
        signature_dict: integrate.analyse()'s "signature" for HEAD.
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        The keys _TEST_CASE_VAR_DOC names, "branches" and "test_files" cut by
        cut_work_list().
    """
    target_list = [{"file": entry_dict["file"], "name": entry_dict["name"],
                    "signature": signature_dict.get(
                        repo_to_knowledge(entry_dict["file"], project_name), {}
                    ).get(entry_dict["name"], ""),
                    "tests": entry_dict.get("tests", [])}
                   for entry_dict in find_dict["referenced"] + find_dict["unreferenced"]]
    return cut_work_list({"request": request.strip(), "targets": target_list,
                          "complete": True},
                         {"branches": branch_list, "test_files": test_file_list})


def cut_work_list(work_dict: dict, list_dict: dict[str, list]) -> dict:
    """Put lists in a stage's work dict, each cut to config.INTEGRATE_WORK_MAX_ITEM.

    Args:
        work_dict: The work dict. Its "complete" is set to False when a list is cut, and
            left as it is otherwise.
        list_dict: {key: the whole list}.

    Returns:
        work_dict.
    """
    limit = config.INTEGRATE_WORK_MAX_ITEM
    for key, item_list in list_dict.items():
        work_dict[key] = item_list[:limit]
        if len(item_list) > limit:
            work_dict["complete"] = False
    return work_dict


def case_start_text(work_dict: dict) -> str:
    """Render what the first stage is asked over, as the block placed after its request.

    Args:
        work_dict: case_work_dict()'s return value.

    Returns:
        The block, its heading first: the request, the tasks it was split into, the
        definitions the change touched and the repository's test files.
    """
    line_list = [_CASE_START_TITLE, "",
                 _CASE_REQUEST_LINE.format(request=work_dict["request"])
                 if work_dict["request"] else _CASE_NO_REQUEST_LINE, "", _CASE_TASK_TITLE]
    line_list += [f"  {entry_dict['asked']}" for entry_dict in work_dict["branches"]
                  if entry_dict["asked"]]
    line_list += ["", _CASE_TARGET_TITLE]
    line_list += [_CASE_TARGET_LINE.format(
        name=entry_dict["name"], file=entry_dict["file"], signature=entry_dict["signature"],
        tests=list_text([f"{test_dict['path']} line(s) "
                         f"{', '.join(str(line_no) for line_no in test_dict['lines'])}"
                         for test_dict in entry_dict["tests"]]) or _CASE_NO_TEST)
        for entry_dict in work_dict["targets"]]
    line_list += ["", _CASE_TEST_FILE_TITLE]
    line_list += [f"  {path}" for path in work_dict["test_files"]]
    return "\n".join(line_list)


def _target_key(entry: object, target_list: list[dict]) -> tuple[str, str] | None:
    """Return the definition one entry of the first stage's answer is about.

    Args:
        entry: One entry of "cases" or "no_case".
        target_list: case_work_dict()'s "targets".

    Returns:
        (file, name) of the target, or None for an entry about none of them. A target
        whose name is the entry's is taken before one whose name is the entry's with
        the names of the definitions around it in front ("B.run" for "run"). An entry
        whose file is not the target's is matched by name when one target has that
        name. A name of a language that ignores case matches in either case.
    """
    if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
        return None
    key_list = [(target["file"], target["name"]) for target in target_list
                if name_key(target["file"], target["name"])
                == name_key(target["file"], entry["name"])] \
        or [(target["file"], target["name"]) for target in target_list
            if integrate.same_name(name_key(target["file"], target["name"]),
                                   name_key(target["file"], entry["name"]))]
    same_file_list = [key for key in key_list
                      if key[0] == to_repo_relative(str(entry.get("file") or ""))]
    if same_file_list:
        return same_file_list[0] if len(same_file_list) == 1 else None
    return key_list[0] if len(key_list) == 1 else None


def _case_shape_reason(value: object) -> str:
    """Say that a first stage's answer is not an object with a "cases" list, or accept it.

    Args:
        value: The answer as read from JSON.

    Returns:
        _CASE_SHAPE_REASON filled in, or "" for a dict whose "cases" is a list.
    """
    if isinstance(value, dict) and isinstance(value.get("cases"), list):
        return ""
    return _CASE_SHAPE_REASON.format(call=END_CALL_DICT[COMPLETE_KEY])


def parse_case_list(answer: object, error: str, target_list: list[dict],
                    with_log: bool = True, first_number: int = 1, step: int = 1,
                    outcome: str | None = None) -> dict:
    """Read the first stage's COMPLETE() argument, as a dict or as JSON text.

    Args:
        answer: The "answer" of RLM.run(): a dict, the JSON as a string, or None when
            the stage handed nothing over.
        error: RLM.run()'s "error".
        target_list: case_work_dict()'s "targets". A case about any other definition,
            and one that expects nothing, is left out. So is a reason that is about
            any other definition, or says nothing.
        with_log: Whether what is left out is logged.
        first_number: The number the first case's id holds.
        step: How much each case's number is above the one before it.
        outcome: RLM.run()'s "outcome", which tells a stage that ended itself with
            nothing from one that did not end itself.

    Returns:
        {"ok": True, "cases": [{"id", "file", "name", "input", "expect", "from"}, ...],
        "no_case": [{"file", "name", "why"}, ...], "uncovered": [{"file", "name"}, ...]
        for the targets in neither list}, the ids CASE_ID_PREFIX and a number counted
        from first_number by step and every text cut to config.EDIT_HISTORY_DID_CHAR,
        or {"ok": False, "error": str}.
    """
    if answer is None:
        return {"ok": False, "error": no_answer_reason(
            error, config.TEST_CREATE_MAX_TURN, END_CALL_DICT[COMPLETE_KEY], outcome)}
    read_dict = parse_json_answer(answer, "the cases")
    if not read_dict["ok"]:
        return {"ok": False, "error": read_dict["error"]}
    value = read_dict["value"]
    if _case_shape_reason(value):
        return {"ok": False, "error": _NO_CASE_LIST_ERROR}
    case_list = []
    for entry in value["cases"]:
        key = _target_key(entry, target_list)
        # A case expects nothing only when "expect" is missing or empty; 0 and False count
        has_expect = key is not None and entry.get("expect") is not None \
            and str(entry["expect"]).strip() != ""
        if not has_expect:
            if with_log:
                logger.info("a case is left out: it names no definition the change "
                            "touched, or expects nothing: %s",
                            cut(str(entry), config.RAW_EXCERPT_CHAR))
            continue
        case_list.append({"id": f"{CASE_ID_PREFIX}{first_number + step * len(case_list)}",
                          "file": key[0], "name": key[1],
                          **{name: cut(str(entry.get(name, "")), config.EDIT_HISTORY_DID_CHAR)
                             for name in ("input", "expect", "from")}})
    no_case_list = []
    for entry in value.get("no_case") if isinstance(value.get("no_case"), list) else []:
        key = _target_key(entry, target_list)
        why = str(entry.get("why") or "").strip() if key is not None else ""
        if not why:
            if with_log:
                logger.info("a reason is left out: it names no definition the change "
                            "touched, or says nothing: %s",
                            cut(str(entry), config.RAW_EXCERPT_CHAR))
            continue
        no_case_list.append({"file": key[0], "name": key[1],
                             "why": cut(why, config.EDIT_HISTORY_DID_CHAR)})
    say_set = {(entry_dict["file"], entry_dict["name"])
               for entry_dict in case_list + no_case_list}
    return {"ok": True, "cases": case_list, "no_case": no_case_list,
            "uncovered": [{"file": target["file"], "name": target["name"]}
                          for target in target_list
                          if (target["file"], target["name"]) not in say_set]}


def make_case_guard(target_list: list[dict]) -> Callable[[object], str]:
    """Build the guard on how the first stage ends.

    An answer that cannot be read as JSON is sent back each time, and one that is not an
    object with a "cases" list once (rlm.json_guard). One that can be read is sent back,
    at most config.TEST_CREATE_MAX_REFUSAL times, while a target has neither a case nor
    a reason it gets none.

    Args:
        target_list: case_work_dict()'s "targets".

    Returns:
        on_complete(answer) -> str for RLM.run(): "" accepts, text refuses.
    """
    def check(answer: object, refuse_count: int) -> str:
        """Refuse an answer that leaves a target with neither a case nor a reason."""
        read_dict = parse_case_list(answer, "", target_list, with_log=False)
        if not read_dict["ok"] or not read_dict["uncovered"]:
            return ""
        logger.info("%s() refused (%d of %d): %d definition(s) with no case and no reason",
                    END_CALL_DICT[COMPLETE_KEY], refuse_count + 1,
                    config.TEST_CREATE_MAX_REFUSAL, len(read_dict["uncovered"]))
        return _CASE_REFUSAL.format(
            call=END_CALL_DICT[COMPLETE_KEY],
            names="\n".join(f"  {entry_dict['name']} in {entry_dict['file']}"
                            for entry_dict in read_dict["uncovered"]))

    read_guard = json_guard("the cases", _case_shape_reason)
    cover_guard = refuse_up_to(check, config.TEST_CREATE_MAX_REFUSAL)
    return lambda answer: read_guard(answer) or cover_guard(answer)


def run_test_case(request: str, work_dict: dict, first_number: int, step: int = 1) -> dict:
    """Run the first stage and return the cases it decided.

    Args:
        request: The request the change was made for, or "".
        work_dict: case_work_dict()'s return value.
        first_number: The number the first case's id holds.
        step: How much each case's number is above the one before it.

    Returns:
        parse_case_list()'s return value with "iters" added.
    """
    stage = _make_stage(case_tool_list(), build_test_case_instruction(config.PRIMARY_EXT),
                        "test_case")
    out_dict = run_stage(stage,
                         "\n\n".join([ask_text(request, _CASE_REQUEST_TEXT,
                                               _CASE_NO_REQUEST_TEXT),
                                      case_start_text(work_dict)]),
                         {"work": work_dict}, _TEST_CASE_VAR_DOC,
                         _CASE_LATE_ANSWER_NOTE, _CASE_LATE_ANSWER_CLOSE,
                         turn_note=make_deadline_note(config.EDIT_DEADLINE_TURN,
                                                      _CASE_DEADLINE_WHAT),
                         on_complete=make_case_guard(work_dict["targets"]))
    case_dict = parse_case_list(out_dict["answer"], out_dict["error"],
                                work_dict["targets"], first_number=first_number, step=step,
                                outcome=out_dict["outcome"])
    case_dict["iters"] = out_dict["iters"]
    if case_dict["ok"]:
        logger.info("%d test case(s) were decided after %d turns, and %d definition(s) "
                    "have neither a case nor a reason", len(case_dict["cases"]),
                    out_dict["iters"], len(case_dict["uncovered"]))
    else:
        logger.warning("the test cases could not be read: %s", case_dict["error"])
    return case_dict


def build_work_dict(project_dir: str, base: str, request: str, find_dict: dict,
                    baseline_set: frozenset[str], branch_list: list[dict],
                    case_list: list[dict], test_file_list: list[str]) -> dict:
    """Build the second stage's work dict.

    Args:
        project_dir: The repository, whose HEAD holds the change.
        base: The commit the change was written against.
        request: The request the change was made for, or "".
        find_dict: target_dict()'s return value.
        baseline_set: baseline_of()'s return value.
        branch_list: branch_list_of()'s return value.
        case_list: parse_case_list()'s cases.
        test_file_list: The repository's test files.

    Returns:
        The keys _TEST_CREATE_VAR_DOC names, "branches", "referenced", "unreferenced"
        and "test_files" cut by cut_work_list().
    """
    return cut_work_list(
        {"request": request.strip(), "base": base,
         "union": gitcmd.head_commit(project_dir),
         "cases": [dict(case_dict, **{"in": []}) for case_dict in case_list],
         "own_files": [], "test_files_to_extend": [], "dep_files": [],
         "written": [], "also_changed": [], "read": [],
         "tests": "", "tests_at": 0,
         "already_failing": edit.baseline_name_list(baseline_set),
         "left": [], "checked_at": 0, "decisions": [], "gap": [],
         "complete": True},
        {"branches": branch_list, "referenced": find_dict["referenced"],
         "unreferenced": find_dict["unreferenced"], "test_files": test_file_list})


def change_line_list(entry_list: list[dict]) -> list[str]:
    """Render the files a stage changed, one line each.

    Args:
        entry_list: integrate.change_entry_list()'s entries.

    Returns:
        The lines.
    """
    return [f"  {entry_dict['path']}  "
            + ("new file" if entry_dict["new"]
               else f"+{entry_dict['insertions']} -{entry_dict['deletions']}")
            + (f" ({', '.join(entry_dict['defs'])})" if entry_dict["defs"] else "")
            for entry_dict in entry_list]


def change_head_line_list(work_dict: dict) -> list[str]:
    """Render the head of the start block: the heading and the change the stage is over.

    Args:
        work_dict: The stage's work dict, with "base", "union" and "branches" as
            build_work_dict() sets them.

    Returns:
        The lines, the heading first.
    """
    line_list = [_START_TITLE, "",
                 _CHANGE_TITLE.format(base=gitcmd.short(work_dict["base"]),
                                      union=gitcmd.short(work_dict["union"]))]
    for entry_dict in work_dict["branches"]:
        line_list.append(f"  asked: {entry_dict['asked']}")
        if entry_dict["tests_changed"]:
            line_list.append(f"    tests it changed: {list_text(entry_dict['tests_changed'])}")
    return line_list


def already_red_line_list(work_dict: dict) -> list[str]:
    """Render the tests that were failing before the stage started.

    Args:
        work_dict: The stage's work dict, with "already_failing" as build_work_dict()
            sets it.

    Returns:
        The lines of the section, after an empty line, or [] when none was failing.
    """
    if not work_dict["already_failing"]:
        return []
    return ["", _ALREADY_RED, *[f"  {name}" for name in work_dict["already_failing"]]]


def run_tail_line_list(work_dict: dict) -> list[str]:
    """Render the tail of the state block.

    The tail is what the run read, what the tests said, what the check found and what
    was recorded.

    Args:
        work_dict: The stage's work dict, with "read", "tests", "tests_at",
            "checked_at", "left" and "decisions" as build_work_dict() sets them.

    Returns:
        The lines, an empty one first.
    """
    line_list = ["", _READ_NOW if work_dict["read"] else _NO_READ_NOW]
    line_list += [f"  {entry_dict['path']}  on turn {entry_dict['turn']}"
                  for entry_dict in work_dict["read"]]

    when = _TEST_ON_TURN.format(turn=work_dict["tests_at"]) if work_dict["tests_at"] \
        else _TEST_BEFORE_FIRST_TURN
    line_list += ["", (_TEST_BREAK if work_dict["tests"] else _NO_TEST_BREAK).format(when=when)]
    line_list += [f"  {work_dict['tests']}"] if work_dict["tests"] else []

    line_list += [""]
    if work_dict["checked_at"]:
        line_list.append(_CHECK_AT.format(turn=work_dict["checked_at"]))
        line_list += [f"  {reason}" for reason in work_dict["left"]] or [_CHECK_FIND_NOTHING]
    else:
        line_list.append(_NO_CHECK_YET)

    line_list += ["", _DECISION_TITLE if work_dict["decisions"] else _NO_DECISION]
    line_list += [f"  {entry_dict['what']} — {entry_dict['why']}"
                  for entry_dict in work_dict["decisions"]]
    if work_dict.get("gap"):
        line_list += ["", _GAP_TITLE]
        line_list += [f"  {entry_dict['what']} — {entry_dict['why']}"
                      for entry_dict in work_dict["gap"]]
    return line_list


def start_text(work_dict: dict) -> str:
    """Render what was measured before the first turn: the block after the request.

    Args:
        work_dict: build_work_dict()'s return value, with "own_files",
            "test_files_to_extend" and "dep_files" set.

    Returns:
        The block, its heading first: the change, the cases, the definitions a test
        does and does not refer to, the test files, the files the stage can write to and
        the tests that were failing.
    """
    line_list = change_head_line_list(work_dict)

    line_list += ["", _CASE_TITLE]
    line_list += [_CASE_LINE.format(**case_dict) for case_dict in work_dict["cases"]]

    line_list += ["", _REF_TITLE if work_dict["referenced"] else _NO_REF]
    line_list += [_REF_LINE.format(
        name=entry_dict["name"], file=entry_dict["file"],
        tests=list_text([test_dict["path"] for test_dict in entry_dict["tests"]]))
        for entry_dict in work_dict["referenced"]]
    line_list += ["", _NO_REF_TITLE if work_dict["unreferenced"] else _ALL_REF]
    line_list += [f"  {entry_dict['name']} in {entry_dict['file']}"
                  for entry_dict in work_dict["unreferenced"]]

    line_list += ["", _TEST_FILE_TITLE]
    line_list += [f"  {path}" for path in work_dict["test_files"]]
    line_list += ["", _OWN_FILE_TITLE]
    line_list += [f"  {path}" for path in work_dict["own_files"]
                  + work_dict["test_files_to_extend"]]
    line_list += [_DEP_FILE_LINE.format(path=path) for path in work_dict["dep_files"]]
    return "\n".join(line_list + already_red_line_list(work_dict))


def _state_line_list(work_dict: dict, turn: int, max_turn: int) -> list[str]:
    """Render the state block for the turn about to run.

    Args:
        work_dict: build_work_dict()'s return value, as the turn before this one left it.
        turn: Which turn is about to run (1-based).
        max_turn: The stage's turn limit.

    Returns:
        The lines of the block, the heading first.
    """
    gap_set = decision_id_set(work_dict["cases"], work_dict.get("gap", []))
    decision_set = decision_id_set(work_dict["cases"], work_dict["decisions"])
    line_list = [STATE_TITLE.format(turn=turn, max_turn=max_turn), "", _CASE_NOW]
    line_list += [_CASE_NOW_LINE.format(
        id=case_dict["id"], name=case_dict["name"], file=case_dict["file"],
        where=list_text(case_dict["in"])
        or (_CASE_IN_GAP if case_dict["id"] in gap_set
            else _CASE_IN_DECISION if case_dict["id"] in decision_set else _CASE_NOWHERE))
        for case_dict in work_dict["cases"]]

    line_list += ["", _OWN_TEST_NOW if work_dict["written"] else _NO_OWN_TEST_NOW]
    line_list += change_line_list(work_dict["written"])
    line_list += ["", _OTHER_CHANGE_NOW if work_dict["also_changed"] else _NO_OTHER_CHANGE_NOW]
    line_list += change_line_list(work_dict["also_changed"])
    return line_list + run_tail_line_list(work_dict)


def make_state(project_dir: str, work_dict: dict, check: Callable[[], list[str]],
               baseline_set: frozenset[str], range_dict: dict, project_name: str,
               refuse_list: list[str] | None = None
               ) -> tuple[Callable[[int, int], str], Callable[[], list[str]]]:
    """Build the state block's turn_note() and the check that keeps the block current.

    What the stage changed and read is measured again on every turn. What the tests say
    follows the last run of the whole suite, whichever side ran it. _NOTHING_LEFT closes
    the block when every case is placed and a whole run since the last change said
    nothing new, unless the check's last reasons stand over an unchanged tree or a
    write outside the tests was refused.

    Args:
        project_dir: The repository.
        work_dict: build_work_dict()'s return value, which both return values write to.
        check: () -> [reason, ...], the machine's own reading of what is still wrong.
        baseline_set: baseline_of()'s set for the run before the stage started.
        range_dict: integrate.analyse()'s "range" for HEAD.
        project_name: The directory name of PROJECT_DIR, for the path conversion.
        refuse_list: The paths a write was refused for, read again on every turn. None
            reads none.

    Returns:
        (turn_note(turn, max_turn) -> str, a check that records what it found in the
        state before returning it). A turn past max_turn is headed and kept as max_turn.
    """
    # "n" is the count of whole test runs the state last read, "change" the contents of
    # what the stage had changed when that run ended (edit.stage_change_hash_dict), and
    # "check_change" the same as the check last ended. "change" is None until turn_note
    # reads the first new whole run, "check_change" until the first check
    run_count_dict = {"n": (edit.last_whole_run() or {"n": 0})["n"], "change": None,
                      "check_change": None}
    is_cut = not work_dict["complete"]
    deadline = make_deadline_note(config.EDIT_DEADLINE_TURN, _DEADLINE_WHAT)

    def turn_note(turn: int, max_turn: int) -> str:
        """Measure the state again and render it for the turn about to run."""
        integrate.set_turn(min(turn, max_turn))
        change_list = integrate.change_entry_list(project_dir, "HEAD", range_dict,
                                                  project_name)
        test_path_set = set(under_test_path(entry_dict["path"] for entry_dict in change_list))
        work_dict["complete"] = not is_cut
        cut_work_list(work_dict, {
            "written": [entry_dict for entry_dict in change_list
                        if entry_dict["path"] in test_path_set],
            "also_changed": [entry_dict for entry_dict in change_list
                             if entry_dict["path"] not in test_path_set],
            "read": integrate.read_list()})
        work_dict["decisions"] = integrate.decision_list()
        work_dict["gap"] = integrate.gap_list()
        place_dict = case_place_dict(project_dir, work_dict["cases"])
        for case_dict in work_dict["cases"]:
            case_dict["in"] = place_dict[case_dict["id"]]
        change_hash_dict = edit.stage_change_hash_dict()
        run_dict = edit.last_whole_run()
        if run_dict and run_dict["n"] != run_count_dict["n"]:
            run_count_dict["n"] = run_dict["n"]
            run_count_dict["change"] = run_dict["change"]
            work_dict["tests"] = whole_run_reason(run_dict, baseline_set)
            work_dict["tests_at"] = turn - 1
        line_list = _state_line_list(work_dict, min(turn, max_turn), max_turn)
        # Every case placed, and a whole run since the last change that said nothing new
        decision_set = decision_id_set(work_dict["cases"])
        is_check_open = bool(work_dict["left"]) \
            and run_count_dict["check_change"] == change_hash_dict
        if turn <= max_turn and change_hash_dict \
                and run_count_dict["change"] == change_hash_dict \
                and not work_dict["tests"] \
                and not is_check_open and not _other_refuse_list(refuse_list or []) \
                and all(case_dict["in"] or case_dict["id"] in decision_set
                        for case_dict in work_dict["cases"]):
            line_list += ["", _NOTHING_LEFT]
        note = deadline(turn, max_turn)
        return "\n".join(line_list + (["", note] if note else []))

    check_and_keep = make_check_and_keep(work_dict, check)

    def check_and_keep_change() -> list[str]:
        """Run the check and keep what the stage had changed when it ended."""
        left_list = check_and_keep()
        run_count_dict["check_change"] = edit.stage_change_hash_dict()
        return left_list

    return turn_note, check_and_keep_change


def _other_refuse_list(refuse_list: list[str]) -> list[str]:
    """Return the paths a write was refused for that are not tests.

    Args:
        refuse_list: The paths the stage's tools refused.

    Returns:
        Those of them under_test_path() does not keep, sorted.
    """
    return sorted(set(refuse_list) - set(under_test_path(refuse_list)))


def make_check_and_keep(work_dict: dict, check: Callable[[], list[str]]
                        ) -> Callable[[], list[str]]:
    """Build the check that records what it found in the state before returning it.

    Args:
        work_dict: The stage's work dict, whose "left", "checked_at", "decisions" and "gap"
            the check writes.
        check: () -> [reason, ...], the machine's own reading of what is still wrong.

    Returns:
        () -> the reasons check() returned.
    """
    def check_and_keep() -> list[str]:
        """Run the check and write what it found in the work dict."""
        work_dict["left"] = check()
        work_dict["checked_at"] = integrate.stage_turn()
        work_dict["decisions"] = integrate.decision_list()
        work_dict["gap"] = integrate.gap_list()
        return work_dict["left"]

    return check_and_keep


def _no_write_dict(skip: str = "", error: str = "", case_dict: dict | None = None) -> dict:
    """Return run_test_create()'s value for a run whose second stage did not run.

    Args:
        skip: Why there was nothing to write, one of the SKIP_* and _SKIP_* texts. ""
            with error.
        error: Why the cases could not be read. "" with skip.
        case_dict: run_test_case()'s return value, or None when the first stage did not
            run.

    Returns:
        The keys run_test_create() returns. A run with nothing to write is settled
        unless a definition was left with neither a case nor a reason; one whose cases
        could not be read is not, and "left" holds the reasons.
    """
    case_dict = case_dict or {}
    left_list = ([_NO_CASE_REASON.format(why=error)] if error else []) \
        + _uncover_reason_list(case_dict.get("uncovered", []))
    return {"work": {"decisions": [], "gap": []}, "called": False, "left": left_list,
            "settled": not left_list, "skipped": "" if left_list else skip, "answer": "",
            "history": [], "iters": 0, "case_iters": case_dict.get("iters", 0),
            "cases": case_dict.get("cases", []), "no_case": case_dict.get("no_case", []),
            "uncovered": case_dict.get("uncovered", []),
            "refused": [], "refused_completes": 0, "harvested": False, "error": error,
            "outcome": None,
            "tests_changed": [], "non_test_changed": [], "untooled": [],
            "tests_base": "", "base_failed": []}


def _uncover_reason_list(uncover_list: list[dict]) -> list[str]:
    """Return the reason line for the definitions left with no case and no reason.

    Args:
        uncover_list: parse_case_list()'s "uncovered".

    Returns:
        One line, or [] when there is none.
    """
    if not uncover_list:
        return []
    return [_UNCOVER_REASON.format(names=list_text(
        [f"{entry_dict['name']} in {entry_dict['file']}" for entry_dict in uncover_list]))]


def list_target(base: str, knowledge_path: str, project_name: str) -> dict:
    """Say which definitions HEAD changed against a commit and how the tests stand.

    No model is called. The tests are run once, and only where there is something to
    write a test for.

    Args:
        base: The commit the change HEAD holds was written against.
        knowledge_path: The analysis of HEAD.
        project_name: The directory name of PROJECT_DIR.

    Returns:
        {"targets": change_def_dict()'s mapping, "tests": test_by_file_dict() over those
        files, "baseline": baseline_or_skip()'s set, sorted, "skipped": why there is
        nothing to write a test for, one of the SKIP_* and _SKIP_* texts, or "",
        "first_case": next_case_number() over the repository's test files}.
    """
    project_dir = config.PROJECT_DIR
    start_dict = integrate.analyse(knowledge_path)
    def_dict = testable_def_dict(change_def_dict(project_dir, base, start_dict["range"],
                                                 project_name))
    baseline_set, reason = baseline_or_skip(project_dir, def_dict)
    test_file_list = repo_test_file_list(project_dir)
    return {"targets": def_dict,
            "tests": test_by_file_dict(list(def_dict), start_dict["use"], project_name),
            "baseline": sorted(baseline_set), "skipped": reason,
            "first_case": next_case_number(project_dir, test_file_list)}


def run_test_create(base: str, knowledge_path: str, project_name: str, output_dir: str,
                    exclude_list: list[str] | None = None, request: str = "",
                    task_dict: dict | None = None) -> dict:
    """Write tests for what HEAD changed against a commit, and check what was written.

    The flow:
    1. What the change touched, and which tests refer to it
    2. The cases, decided without the code (run_test_case, _case_stop_dict)
    3. The tests, written from the cases
    4. The check, where the guard did not make it
    5. What was written, and what it says over the base (_add_own_change)

    Args:
        base: The commit the change HEAD holds was written against.
        knowledge_path: The analysis of HEAD.
        project_name: The directory name of PROJECT_DIR.
        output_dir: Where the analysis is written when it is made again.
        exclude_list: The extra names kept out of the analysis, as the build takes them.
        request: The request the change was made for. "" leaves it out.
        task_dict: {"paths": the files whose definitions this run writes tests for,
            "test_paths": the test files that refer to them, "baseline": the baseline
            before it, as list_target() returns it, and optionally "first_case": the
            number this run's first case takes and "case_step": how much each case's
            number is above the one before it}. None takes every file the change
            touched, reads the test files off the analysis and runs the tests to read
            the baseline. Without "first_case" the cases are numbered on from
            next_case_number(), and without "case_step" they are one apart.

    Returns:
        The second stage's RLM.run() return value with "work", "called" (whether the
        second stage ran), "skipped" (why there was nothing to write, or ""), "cases"
        (what the first stage decided) and "case_iters" (the turns the first stage
        took), "no_case" and "uncovered" (as parse_case_list() returns them), "refused"
        (the paths a write was refused for), "left" (what the machine still finds
        wrong), "settled", "tests_changed" and "non_test_changed" (the test files and
        the other files the stage changed), "untooled" (the files it changed without an
        editing tool), "tests_base" (BASE_RED, BASE_GREEN, BASE_UNKNOWN or "") and
        "base_failed" added, those two as _own_change_dict() returns them, or "" and []
        when the second stage did not run.
    """
    project_dir = config.PROJECT_DIR
    integrate.reset_stage()
    edit.reset_patch_round(round_limit=0)
    guide.reset()

    # == Step 1: what the change touched, and which tests refer to it ==
    start_dict = integrate.analyse(knowledge_path)
    def_dict = testable_def_dict(change_def_dict(project_dir, base, start_dict["range"],
                                                 project_name))
    if task_dict is not None:
        def_dict = {path: name_list for path, name_list in def_dict.items()
                    if path in set(task_dict["paths"])}
    baseline_set, reason = baseline_or_skip(
        project_dir, def_dict, task_dict["baseline"] if task_dict is not None else None)
    if reason:
        logger.info("no test is written: %s; no model was called", reason)
        return _no_write_dict(skip=reason)
    check_before = edit.run_check()
    find_dict = target_dict(def_dict, start_dict["ref"], start_dict["range"], project_name)
    branch_list = branch_list_of(project_dir, base)
    start_file_list = gitcmd.work_tree_file_list(project_dir) or []
    test_file_list = under_test_path(start_file_list)
    own_file_list = sorted(def_dict)
    dep_pattern_set = edit.test_build_file_pattern_set()
    dep_file_list = [path for path in start_file_list
                     if is_dep_file(path, dep_pattern_set)]
    extend_list = sorted(task_dict["test_paths"]) if task_dict is not None else sorted(
        {path for path_list in test_by_file_dict(
            own_file_list, start_dict["use"], project_name).values() for path in path_list})

    # == Step 2: the cases, decided without the code ===========
    number_dict = task_dict or {}
    case_dict = run_test_case(
        request, case_work_dict(test_file_list, request, branch_list, find_dict,
                                start_dict["signature"], project_name),
        number_dict.get("first_case") or next_case_number(project_dir, test_file_list),
        number_dict.get("case_step", 1))
    stop_dict = _case_stop_dict(case_dict)
    if stop_dict is not None:
        return stop_dict
    case_list = case_dict["cases"]
    work_dict = build_work_dict(project_dir, base, request, find_dict, baseline_set,
                                branch_list, case_list, test_file_list)
    cut_work_list(work_dict, {"own_files": own_file_list, "test_files_to_extend": extend_list,
                              "dep_files": dep_file_list})
    refuse_list: list[str] = []

    # == Step 3: the tests, written from the cases =============
    integrate.reset_stage()
    integrate.set_gap_check(functools.partial(integrate.gap_test_result,
                                              baseline_set=baseline_set))
    may_write = may_write_of(own_file_list, extend_list, start_file_list, dep_pattern_set)
    check = functools.partial(reason_list, project_dir, output_dir, start_dict,
                              baseline_set, exclude_list, case_list,
                              check_before=check_before, start_file_list=start_file_list,
                              may_write=may_write)
    stage = _make_stage(
        test_create_tool_list(may_write, refuse_list, not_own_error(bool(dep_pattern_set))),
        build_test_create_instruction(config.PRIMARY_EXT), "test_create")
    turn_note, check_and_keep = make_state(project_dir, work_dict, check, baseline_set,
                                           start_dict["range"], project_name, refuse_list)
    out_dict = run_stage(stage,
                         "\n\n".join([_request_of(request), start_text(work_dict)]),
                         {"work": work_dict},
                         _TEST_CREATE_VAR_DOC, _LATE_ANSWER_NOTE, _LATE_ANSWER_CLOSE,
                         late_state=turn_note, turn_note=turn_note, text_answer=True,
                         on_complete=integrate.make_complete_guard(
                             check_and_keep, config.TEST_CREATE_MAX_REFUSAL,
                             _COMPLETE_REFUSAL))

    # == Step 4: the check, where the guard did not make it ====
    need_check = not (out_dict["outcome"] == COMPLETE_KEY
                      and out_dict["refused_completes"] < config.TEST_CREATE_MAX_REFUSAL)
    other_refuse_list = _other_refuse_list(refuse_list)
    out_dict["left"] = (check_and_keep() if need_check else work_dict["left"]) \
        + ([_REFUSE_REASON.format(paths=list_text(other_refuse_list))]
           if other_refuse_list else [])
    out_dict["settled"] = not out_dict["left"]
    work_dict["decisions"] = integrate.decision_list()
    work_dict["gap"] = integrate.gap_list()
    out_dict.update({"work": work_dict, "called": True, "skipped": "",
                     "cases": case_list, "case_iters": case_dict["iters"],
                     "no_case": case_dict["no_case"], "uncovered": case_dict["uncovered"],
                     "refused": sorted(refuse_list)})

    # == Step 5: what was written, and what it says over the base ==
    return _add_own_change(out_dict, project_dir, base, baseline_set)


def _case_stop_dict(case_dict: dict) -> dict | None:
    """Return run_test_create()'s value where the cases keep the second stage from starting.

    Args:
        case_dict: run_test_case()'s return value.

    Returns:
        _no_write_dict()'s value for cases that could not be read, for a definition left
        with neither a case nor a reason and for no case at all (the last two logged),
        or None when the second stage starts.
    """
    if not case_dict["ok"]:
        return _no_write_dict(error=case_dict["error"], case_dict=case_dict)
    if case_dict["uncovered"]:
        logger.info("no test is written: %d definition(s) have neither a case nor a "
                    "reason", len(case_dict["uncovered"]))
        return _no_write_dict(case_dict=case_dict)
    if not case_dict["cases"]:
        logger.info("no test is written: %s", _SKIP_NO_CASE)
        return _no_write_dict(skip=_SKIP_NO_CASE, case_dict=case_dict)
    return None


def _add_own_change(out_dict: dict, project_dir: str, base: str,
                    baseline_set: frozenset[str]) -> dict:
    """Add what the second stage changed and what its tests say over the base to its value.

    Args:
        out_dict: run_test_create()'s value so far, with "left".
        project_dir: The repository, holding what the stage left.
        base: The commit the change was written against.
        baseline_set: baseline_of()'s set for the run before the stage started.

    Returns:
        out_dict with _own_change_dict()'s keys but "restore_error" added. A working
        tree that could not be put back adds a line to "left" and leaves the run
        unsettled.
    """
    change_dict = _own_change_dict(project_dir, base, baseline_set)
    restore_error = change_dict.pop("restore_error")
    out_dict.update(change_dict)
    if restore_error:
        out_dict["left"] = out_dict["left"] + [_RESTORE_REASON.format(why=restore_error)]
        out_dict["settled"] = False
    return out_dict


def _own_change_dict(project_dir: str, base: str, baseline_set: frozenset[str]) -> dict:
    """Say what the second stage changed, and run the test files it wrote over the base.

    Args:
        project_dir: The repository, holding what the stage left.
        base: The commit the change was written against.
        baseline_set: baseline_of()'s set for the run before the stage started.

    Returns:
        {"tests_changed": the test files the stage changed, "non_test_changed": the
        other files it changed, "untooled": the files it changed without an editing
        tool, "tests_base" and "base_failed": base_run_dict()'s "status" and "failed"
        over the test files that are there now, BASE_UNKNOWN and [] when it raises
        RuntimeError, "restore_error": why the working tree could not be put back after
        that run, or ""}, every list sorted.
    """
    exist_list, delete_list = own_test_tuple(project_dir)
    change_dict = {
        "tests_changed": sorted(exist_list + delete_list),
        "non_test_changed": sorted(edit.own_change_set() - edit.test_change_set()),
        "untooled": sorted(edit.no_tool_path_set())}
    restore_error = ""
    try:
        base_dict = base_run_dict(project_dir, base, exist_list, baseline_set)
    except RuntimeError as e:
        logger.warning("the tests written could not be run over %s: %s",
                       gitcmd.short(base), e)
        base_dict = {"status": BASE_UNKNOWN, "failed": []}
        restore_error = str(e) if isinstance(e, RestoreError) else ""
    return {**change_dict, "tests_base": base_dict["status"],
            "base_failed": base_dict["failed"], "restore_error": restore_error}
