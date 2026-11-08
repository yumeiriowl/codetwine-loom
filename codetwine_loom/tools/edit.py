"""The editing tools.

They work in PROJECT_DIR, and a write they make is checked against it.

Four of them write: replace_in_file, which pastes the old text, replace_line_range,
which works by line number and demands less of the format, create_file, which makes a
file that is not there yet, and delete_file, which deletes one.
"""

import ast
import collections
import contextlib
import difflib
import fnmatch
import functools
import hashlib
import logging
import json
import os
import posixpath
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import warnings
from collections.abc import Callable, Iterator

from codetwine_loom import (broker, cobol, config, gitcmd, progress, textfile,
                            tmpfile)
from codetwine_loom.languages import (ALIAS_DICT, LANG_DICT, PY_REQUIREMENT_FILE_TUPLE,
                                      is_dep_file, lang_of, name_only_dir_set,
                                      root_target_pattern_tuple, split_output_tuple)
from codetwine_loom.paths import (has_name_of, is_git_dir_path, is_test_path, list_text,
                                  path_text, real_rel, resolve_in_project, test_root,
                                  to_repo_relative, under_test_path)
from codetwine_loom.textcut import (CUT_NOTE, cut, cut_tail, line_list_of, one_line,
                                    term_text, text_error, whole_number_error)

logger = logging.getLogger(__name__)

# The languages tested by the same build tool as another language, which print the same
# lines: TypeScript's suite is run by npm as JavaScript's is, and C++'s by ctest as C's
# is. The pattern tables below are written out for the extension on the right only, and
# _with_alias_dict() fills in the one on the left
_SAME_REPORTER_DICT = {"ts": "js", "cpp": "c"}


def _with_alias_dict(table_dict: dict) -> dict:
    """Return one of the pattern tables with every other extension filled in.

    Args:
        table_dict: A pattern per extension, written out for one extension per build tool.

    Returns:
        The same table with each extension that shares a build tool
        (_SAME_REPORTER_DICT) and then each alias of a language (languages.ALIAS_DICT)
        mapped to the pattern already there. An extension whose source is not in the
        table is left out.
    """
    share_dict = table_dict | {ext: table_dict[base] for ext, base in _SAME_REPORTER_DICT.items()
                               if base in table_dict}
    return share_dict | {alias: share_dict[base] for alias, base in ALIAS_DICT.items()
                         if base in share_dict}


# The head of the last line of a unittest run that was not green, after "FAILED ":
# FAILED (failures=1) / FAILED (errors=2) / FAILED (failures=1, errors=1) /
# FAILED (skipped=1, unexpected successes=1)
_UNITTEST_COUNT_PATTERN = (r"\((?:failures|errors|skipped|expected failures"
                           r"|unexpected successes)=\d+[,)]")

# unittest's line for a failing test, with "<method> (<id>)" as the group:
# FAIL: test_x (tests.test_m.CaseTest.test_x) / ERROR: setUpClass (tests.test_m.CaseTest)
_UNITTEST_FAIL_PATTERN = r"^(?:FAIL|ERROR): (\S+ \([\w.]+\))(?= .|$)"

# A pytest test id at the end of a line of its short summary, or before the " - " that
# its message follows: tests/t.py::t_x, tests/t.py::t_x[a b] (a parameter id is kept
# whole, and an id whose "[" is not closed is not read), tests/sub dir/t.py::t_x and
# tests/a - b/t.py::t_x (the file's path before the first "::" may hold a space and
# " - " when it starts with no space, holds no ":" and ends in a name with a suffix;
# another path that holds a space is not read), tests/t.py (a file: a path with no space
# that ends in a suffix, one with no space that ends the line, or one that ends the line
# in a file's name with a suffix after a "/", starts with no space and holds no ":" -
# tests/a - b/t.py, tests/a b/t.py)
_PYTEST_ID_PATTERN = (
    r"((?=\S)(?:[^\n\[:]*?\.\w+::|(?=[^\s\[]*::))(?>[^\s\[]+)"
    r"(?:\[(?:(?!\] - |\]$).)*+\](?>\S*))?|[^\s\[:]*\.[A-Za-z]\w*|[^\s\[]+(?=$)"
    r"|(?=\S)[^\n\[:]*/[^\n\[:/]*\.[A-Za-z]\w*(?=$))"
    r"(?= - |$)")

# bun test's own lines: the line it starts its output with ("bun test v1.4.4
# (38afee6c8)"), the heading of each test file ("test/a.test.ts:", the path as its group),
# the line under a heading for an error outside the file's tests ("# Unhandled error
# between tests"), and the line its second listing of the failing tests starts with
# ("1 tests failed:")
_BUN_HEAD_PATTERN = r"^bun test v\d"
_BUN_FILE_HEAD_RE = re.compile(r"^(\S+\.[cm]?[jt]sx?):$")
_BUN_ERROR_MARK_RE = re.compile(r"^# Unhandled error between tests$")
_BUN_REPEAT_HEAD_PATTERN = r"^\d+ tests? failed:$"

# How the names of failing tests are taken: one regex per language, applied line by
# line, with the group that took part in the match as the name
_FAIL_RE_DICT = _with_alias_dict({
    # pytest: FAILED tests/t.py::t_x - ... / ERROR tests/t.py::t_x - ... (a fixture or
    # setup that failed) / ERROR tests/t.py (the file could not be collected) / ERROR
    # tests/sub - ModuleNotFoundError: ... (a directory whose conftest.py could not be
    # imported: a path with no space, "[" or ":" followed by " - ", where what follows
    # is not the rest of a file's path that ends in a suffix and then " - " or the
    # line's end) / FAILED tests/t.py::t_x[a b] - ... (a parameter id is kept whole).
    # unittest's last line (_UNITTEST_COUNT_PATTERN) names no test. A failing subtest:
    # SUBFAILED(i=1) tests/t.py::T::t_x - ... / SUBFAILED[msg] (i=1) tests/t.py::t_x -
    # ... (the name is the test's id that follows the subtest's msg and arguments: every
    # subtest of one test has the test's name)
    # unittest: FAIL: test_x (tests.test_m.CaseTest.test_x) / ERROR: test_x
    # (tests.test_m.CaseTest) / ERROR: setUpClass (tests.test_m.CaseTest) / FAIL: test_x
    # (tests.test_m.CaseTest.test_x) [msg] (i=1) (a subtest: the name ends after the
    # test's id, and every subtest of one test has the test's name). pytest's lines are
    # read from the head of its short summary on (_FAIL_HEAD_DICT)
    "py": re.compile(r"^(?:FAILED|ERROR) (?!" + _UNITTEST_COUNT_PATTERN + r")" + _PYTEST_ID_PATTERN
                     + r"|^ERROR ([^\s\[:]+)(?= - (?![^\n\[:]*\.[A-Za-z]\w*(?: - |$)))"
                     + r"|^SUBFAILED[\[(].*? (?=[^\s\[]*::)" + _PYTEST_ID_PATTERN
                     + r"|" + _UNITTEST_FAIL_PATTERN),
    # [ERROR] com.e.FooTest.bar -- Time elapsed: ... / [ERROR] com.e.FooTest$Inner.deep
    # -- Time elapsed: ... / [ERROR] com.e.FooTest.doubles(int)[2] -- Time elapsed: ...
    # (the name runs up to " -- Time elapsed") / [ERROR]   FooTest.bar:12 expected: ...
    # (a name followed by neither of them runs the match on to the end of the line)
    "java": re.compile(r"^\[ERROR\]\s+([\w.$]+\.[\w$]+(?:.*?(?= -- Time elapsed))?)"
                       r"(?:(?= -- Time elapsed|:\d)|.*$)"),
    # FooTest > bar() FAILED / FooTest > adds two numbers() FAILED /
    # FooTest > doubles(int) > [2] 2 FAILED / FooTest > Inner > deep() FAILED.
    # "> Task :test FAILED" has no " > " after its first character
    "kt": re.compile(r"^(?=.* FAILED$)(\S.*? > .+) FAILED$"),
    # jest: FAIL test/a.test.js (one for each file that failed, and for one that failed
    # to run; " FAIL " with a space in front where its badge was colored) and the
    # indented "● FooTest › bar" / "● Test suite failed to run" heading
    # of each failure. vitest: " FAIL  test/a.test.js > calc > bar" for each failing
    # test and " FAIL  test/a.test.js [ test/a.test.js ]" for a file that failed to run,
    # the bracket that repeats the name left out of it. node --test: "not ok 2 - bar"
    # (TAP, at any depth, not a TODO or SKIP) and "✖ bar (1.2ms)" (spec), for each
    # failing test, each suite holding one and each file that failed to load, but not
    # eslint's "✖ 1 problem (0 errors, 1 warning)". mocha: the indented "1) bar" of each
    # failing test in its list of the tests it runs. bun test: "(fail) calc > bar [0.13ms]"
    # for each failing test (a file that failed to load is named by _FAIL_UNDER_HEAD_DICT).
    # A trailing "(5.2 s)" or "[0.13ms]" is left out of the name
    "js": re.compile(r"^(?:FAIL\s+| FAIL\s+|\s+●\s+"
                     r"|\s*not ok \d+ - (?!.*\s#\s(?:TODO|SKIP)\b)"
                     r"|\s*✖\s+(?!failing tests:$)(?!\d+ problems? \()|\s+\d+\) |\(fail\) )"
                     r"(.+?)(?: \[ \1 \])?"
                     r"(?:\s+\(\d+(?:\.\d+)?\s*m?s\)|\s+\[\d+(?:\.\d+)?m?s\])?\s*$"),
    # \t 2 - FooTest.bar (Failed), and (Not Run) when the executable was never built.
    # Only the block under "The following tests FAILED:" has this shape
    "c": re.compile(r"^\s*\d+\s+-\s+(\S+)\s+\([\w ]+\)\s*$"),
    # test tests::subtracts ... FAILED / test src/lib.rs - add (line 5) ... FAILED /
    # error: test failed, to rerun pass `--test basic` / error: doctest failed, to rerun
    # pass `--doc` (a target that failed, the one line for a test binary that was killed
    # before its summary)
    "rs": re.compile(r"^(?:test (.+?) \.\.\. FAILED"
                     r"|error: (?:doc)?test failed, to rerun pass [`'](.+)[`'])$"),
    # VSTest:   Failed Calc.Tests.T.Adds(a: 1) [3 ms] / [< 1 ms]
    # Microsoft.Testing.Platform: failed Calc.Tests.T.Adds (15ms)
    "cs": re.compile(r"^\s*[Ff]ailed (.+) [\[(][<\d][^\])]*[\])]$"),
    # testthat's check reporter: ── Failure ('test-basic.R:6:3'): add adds ────
    # and ── Error ('test-basic.R:8:3'): add errors ────
    "r": re.compile(r"^── (?:Failure|Error) \('[^']*'\): (.+?) ─+$"),
})

# Where a run prints its failing tests a second time, one pair of regexes per language:
# (the line the second listing starts at, the line it ends before). The lines between
# them are not read for failing names. jest lists every failing test again under
# "Summary of all failing tests" when it ran more than 20 suites, before "Test Suites:",
# and bun test under "1 tests failed:" before its counts (" 25 pass").
# automake's make check with VERBOSE=yes prints test-suite.log, the logs of the tests that
# failed, from its heading ("   calc 1.0: ./test-suite.log") to its summary
_REPEAT_RE_DICT = _with_alias_dict({
    "js": (re.compile(r"^Summary of all failing tests$|^ +\d+ failing$|"
                      + _BUN_REPEAT_HEAD_PATTERN),
           re.compile(r"^Test Suites: |^ +\d+ pass$")),
    "c": (re.compile(r"^\s+.+: \S*test-suite\.log$"), re.compile(r"^Testsuite summary"))})

# Where a run lists the tests that failed and then passed when they were run again, one
# pair of regexes per language: (the listing, with its lines as the first group; a test's
# name in it). A failing name the listing holds is not a failing test of the run.
# surefire, after "Results:": "[WARNING] Flakes: ", then "[WARNING] com.e.FooTest.adds"
# for each test, followed by its "[ERROR]   Run 1: ..." / "[INFO]   Run 2: PASS" lines
_PASS_AGAIN_RE_DICT = _with_alias_dict({
    "java": (re.compile(r"^\[WARNING\] Flakes: ?\n((?:\[(?:INFO|WARNING|ERROR)\] +\S.*\n)+)",
                        re.M),
             re.compile(r"^\[WARNING\] (\S.*)$", re.M))})

# What a failing test's name holds that differs from one run of the same tests to the
# next, one regex per language, left out of the name: the identity hash the JVM writes
# after a class name (com.example.Obj@6d06d69c) and the address in the name of a lambda's
# class (FooTest$$Lambda/0x000071c4e8003a00), which JUnit 5 writes for an argument of a
# parameterized test with no toString() of its own
_RUN_PART_RE_DICT = _with_alias_dict({
    ext: re.compile(r"(?<=[\w$])(?:@[0-9a-f]{1,8}|/0x[0-9a-f]+)\b") for ext in ("java", "kt")})

# How many characters at the head of a line a failing test's name is looked for in, and
# the longest line the test file a command says it runs is looked for in (a longer one
# is not read). A line saying a run ended early is cut to this length
_MAX_FAIL_LINE_CHAR = 2000

# A control sequence a terminal acts on, which a test command told to color its output
# writes into it: a CSI sequence ("\x1b[31m", "\x1b[0m", "\x1b[2K") or an OSC one (a
# link, ended by BEL or by ESC and a backslash)
_TERM_SEQUENCE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def _plain_text(text: str) -> str:
    """Return a command's output without the control sequences a terminal acts on.

    Examples:
        "\\x1b[31mFAILED\\x1b[0m t.py::t" -> "FAILED t.py::t"

    Args:
        text: The output, or one line of it.

    Returns:
        The text with every _TERM_SEQUENCE_RE match taken out.
    """
    return _TERM_SEQUENCE_RE.sub("", text)


def _fail_name(line: str, ext: str, fail_re: re.Pattern | None = None) -> str | None:
    """Return the failing test one line of a test command's output names.

    Args:
        line: One line of the output.
        ext: The language the output is read as (_FAIL_RE_DICT).
        fail_re: The regex the line is read with; None for the language's _FAIL_RE_DICT
            one.

    Returns:
        The name without what _RUN_PART_RE_DICT finds in it, or None when the line names
        no failing test, or the language has no such line. A line longer than
        _MAX_FAIL_LINE_CHAR is read by its first _MAX_FAIL_LINE_CHAR characters, and is
        None where the match runs on to the end of them.
    """
    fail_re = fail_re or _FAIL_RE_DICT.get(ext)
    if fail_re is None:
        return None
    head = line[:_MAX_FAIL_LINE_CHAR]
    match = fail_re.match(head)
    if match is None or (len(head) < len(line) and match.end() == len(head)):
        return None
    run_part_re = _RUN_PART_RE_DICT.get(ext)
    name = match.group(match.lastindex)
    return run_part_re.sub("", name) if run_part_re is not None else name


def _fail_pair_list(out: str, ext: str) -> list[tuple[int, str]]:
    """Return the lines of a test command's output that name a failing test, with names.

    Args:
        out: The text read as a test command's output.
        ext: The language the output is read as.

    Returns:
        (the line's index in line_list_of(out), its _fail_name()) for each line in
        order, a line before the language's head of its failing tests read with the
        regex for such a line (_FAIL_HEAD_DICT), the lines of a second listing of the
        failing tests (_REPEAT_RE_DICT) and the names of the tests that passed when they
        were run again (_PASS_AGAIN_RE_DICT) left out. A line the language's regex names
        nothing in is read with _FAIL_IF_RE_DICT's where its summary is in out. A line
        saying something failed outside the tests (_FAIL_UNDER_HEAD_DICT) is paired with
        the name of the heading above it, where there is one.
    """
    start_re, end_re = _REPEAT_RE_DICT.get(ext, (None, None))
    summary_re, if_name_re = _FAIL_IF_RE_DICT.get(ext, (None, None))
    if summary_re is not None and not summary_re.search(out):
        if_name_re = None
    under_head_re, under_mark_re = _FAIL_UNDER_HEAD_DICT.get(ext, (None, None))
    head_name = None
    list_re, again_name_re = _PASS_AGAIN_RE_DICT.get(ext, (None, None))
    head_re, before_head_re = (None, None) if _CUT_LINE_RE.search(out) \
        else _FAIL_HEAD_DICT.get(ext, (None, None))
    run_part_re = _RUN_PART_RE_DICT.get(ext)
    again_set = set() if list_re is None else {
        run_part_re.sub("", name) if run_part_re is not None else name
        for match in list_re.finditer(out)
        for name in again_name_re.findall(match.group(1))}
    pair_list = []
    is_repeat = False
    is_head = head_re is None
    for index, line in enumerate(line_list_of(out)):
        if start_re is not None and start_re.match(line):
            is_repeat = True
        elif is_repeat and end_re.match(line):
            is_repeat = False
        is_head = is_head or bool(head_re.match(line))
        line_re = None if is_head else before_head_re
        name = _fail_name(line, ext, line_re)
        if name is None and if_name_re is not None:
            name = _fail_name(line, ext, if_name_re)
        if name is None and under_head_re is not None:
            head_match = under_head_re.match(line)
            if head_match is not None:
                head_name = head_match.group(1)
            elif under_mark_re.match(line):
                name = head_name
        if not is_repeat and name is not None and name not in again_set:
            pair_list.append((index, name))
    return pair_list


# unittest's name for a module or a package it could not import, with the module's
# dotted name as the group: tests.test_m (unittest.loader._FailedTest.tests.test_m) /
# tests.test_m (unittest.loader._FailedTest)
_UNITTEST_LOAD_FAIL_RE = re.compile(r"(\S+) \(unittest\.loader\._FailedTest(?:\.\1)?\)")

# A frame of a Python traceback in which a module is loaded, with the file it names as
# the group: the module's top level (in <module>), its load_tests (in load_tests), and
# the place of a SyntaxError (no function)
_PY_LOAD_FRAME_RE = re.compile(r'^\s*File "(.+)", line \d+(?:, in (?:<module>|load_tests))?\s*$')

# The lines unittest puts above and below the traceback of each failing test
_UNITTEST_SEPARATOR_RE = re.compile(r"^(?:=+|-+)$")


def _run_dir_rel(path: str, run_dir: str) -> str | None:
    """Return a path a traceback names relative to the directory a command ran in.

    Examples:
        ("/repo/tests/t.py", "/repo") -> "tests/t.py"
        ("tests/t.py", "/repo")       -> "tests/t.py"
        ("/usr/lib/x.py", "/repo")    -> None

    Args:
        path: The path, absolute or relative to run_dir.
        run_dir: The absolute directory the command ran in.

    Returns:
        The relative path; None for a path outside run_dir (its real path as well).
    """
    if not os.path.isabs(path):
        rel = posixpath.normpath(path)
        return None if rel == ".." or rel.startswith("../") else rel
    return next((os.path.relpath(path, base)
                 for base in (os.path.abspath(run_dir), os.path.realpath(run_dir))
                 if path.startswith(base.rstrip("/") + "/")), None)


def _unittest_frame_stem_list(line_list: list[str], index: int, dot_name: str,
                              run_dir: str) -> list[str]:
    """Return the module files the traceback under a unittest ERROR line names.

    The ERROR line is unittest's for a module it could not import.

    Examples:
        (an "ERROR: test_m (unittest.loader._FailedTest.test_m)" line whose traceback has
        the frame File "/repo/pkg/test_m.py", line 1, in <module>, "test_m", "/repo")
        -> ["pkg/test_m"]

    Args:
        line_list: The lines of a test command's output as _own_text() returns it
            (line_list_of).
        index: Where the ERROR line is in line_list; the lines under it up to unittest's
            next separator line are read.
        dot_name: The module's dotted name the line gives, from the directory unittest
            imports it under.
        run_dir: The absolute directory the command ran in.

    Returns:
        Each file under run_dir a frame of that traceback loading a module
        (_PY_LOAD_FRAME_RE) names whose path, without ".py" or "/__init__.py", is
        dot_name as a path or ends with "/" and it; that path, relative to run_dir, each
        once.
    """
    stem_tail = dot_name.replace(".", "/")
    stem_list: list[str] = []
    start = index + 1
    if start < len(line_list) and _UNITTEST_SEPARATOR_RE.match(line_list[start]):
        start += 1
    for position in range(start, len(line_list)):
        text = line_list[position]
        if _UNITTEST_SEPARATOR_RE.match(text):
            break
        match = _PY_LOAD_FRAME_RE.match(text)
        rel = None if match is None else _run_dir_rel(match.group(1), run_dir)
        if rel is None:
            continue
        if rel.endswith("/__init__.py"):
            stem = rel[:-len("/__init__.py")]
        elif rel.endswith(".py"):
            stem = rel[:-len(".py")]
        else:
            continue
        if (stem == stem_tail or stem.endswith("/" + stem_tail)) and stem not in stem_list:
            stem_list.append(stem)
    return stem_list


def _py_load_fail_path_list(line_list: list[str], index: int, name: str,
                            run_dir: str) -> list[str]:
    """Return the paths a failing name of a Python run stands for.

    The name is one of something the run could not load.

    Examples (the line at index, the name):
        ("ERROR tests/t.py - ImportError: x", "tests/t.py") -> ["tests/t.py"]
        ("ERROR tests/sub - ModuleNotFoundError: x", "tests/sub") -> ["tests/sub"]
        ("ERROR: tests.t (unittest.loader._FailedTest.tests.t)",
         "tests.t (unittest.loader._FailedTest.tests.t)") -> ["tests/t.py", "tests/t"]
        ("ERROR: t (unittest.loader._FailedTest.t)", "t (unittest.loader._FailedTest.t)")
            under which a frame names /repo/pkg/t.py, line 1, in <module>, run_dir
            "/repo" -> ["pkg/t.py", "pkg/t"]
        ("FAILED tests/t.py::t_x - x", "tests/t.py::t_x") -> []

    Args:
        line_list: The lines of a test command's output as _own_text() returns it
            (line_list_of).
        index: Where the line the name was read off is in line_list (_fail_pair_list).
        name: The failing name.
        run_dir: The absolute directory the command ran in.

    Returns:
        Paths relative to run_dir: the name of pytest's "ERROR <file>" and "ERROR
        <directory> - ..." lines (a file it could not collect, a directory whose
        conftest.py it could not import); the file and the directory of a module or a
        package unittest could not import, from the files its traceback names
        (_unittest_frame_stem_list), else from a dotted name with a package in it; []
        for any other name.
    """
    line = line_list[index]
    if line.startswith("ERROR ") and "::" not in name:
        return [name]
    match = _UNITTEST_LOAD_FAIL_RE.fullmatch(name)
    if not line.startswith("ERROR: ") or match is None:
        return []
    dot_name = match.group(1)
    stem_list = _unittest_frame_stem_list(line_list, index, dot_name, run_dir)
    if not stem_list and "." in dot_name:
        stem_list = [dot_name.replace(".", "/")]
    return [path for stem in stem_list for path in (f"{stem}.py", stem)]


# How a run names what it could not load, and so ran no test of, one reader per
# language: (the lines of the command's output as _own_text() returns it, where a line
# naming a failing test is in them, the name read off the line, the absolute directory
# the command ran in) -> the paths, relative to that directory, of the files and
# directories the name stands for, [] for a test that ran. An extension that is not here
# names nothing so
_LOAD_FAIL_PATH_DICT = _with_alias_dict({"py": _py_load_fail_path_list})


# What a run prints to say it stopped at a count of failing tests and ran none of the
# tests after them, one pattern per language. An extension that is not here has no such
# line
_FAIL_STOP_PATTERN_DICT = {
    # !!!! stopping after 1 failures !!!! (-x and --maxfail) /
    # !!!! xdist.dsession.Interrupted: stopping after 1 failures !!!! (the same, as
    # pytest-xdist writes it)
    "py": r"^!+ (?:\S*Interrupted: )?stopping after \d+ failures? !+$",
    # Bailed out after 1 failure (bun test --bail)
    "js": r"^Bailed out after \d+ failures?$",
}

# What a run prints to say it ended before it had run every test it was given, for
# another reason than a count of failing tests, one pattern per language
_OTHER_EARLY_STOP_PATTERN_DICT = {
    # !!!! Interrupted: 1 error during collection !!!! (a test file that cannot be
    # collected, after which pytest runs no test)
    "py": r"^!+ \S*Interrupted: .+ !+$",
    # [INFO] c ........ SKIPPED (the reactor's line for a module that was not built: one
    # after the first failing module, or with -fae one that depends on a failing
    # module. A name of 51 characters or more is followed by no dots) /
    # [ERROR] COMPILATION ERROR : (the module's tests were not run)
    "java": r"^\[INFO\] \S.* SKIPPED$|COMPILATION ERROR",
    # Execution failed for task ':a:compileKotlin'. / ':a:kaptKotlin'. / ':a:kspKotlin'.
    # (the tests of that project and of each project that depends on it were not run)
    "kt": r"^Execution failed for task ':(?:[^\s:']+:)*(?:compile|kapt|ksp)",
}

# The lines of _FAIL_STOP_PATTERN_DICT, one regex per language
_FAIL_STOP_RE_DICT = _with_alias_dict({
    ext: re.compile(pattern, re.M) for ext, pattern in _FAIL_STOP_PATTERN_DICT.items()})

# What a run prints to say it ended before it had run every test it was given, one
# regex per language: the language's line of _FAIL_STOP_PATTERN_DICT or its line of
# _OTHER_EARLY_STOP_PATTERN_DICT. An extension that is not here has no such line
_EARLY_STOP_RE_DICT = _with_alias_dict({
    ext: re.compile("|".join(
        pattern_dict[ext]
        for pattern_dict in (_FAIL_STOP_PATTERN_DICT, _OTHER_EARLY_STOP_PATTERN_DICT)
        if ext in pattern_dict), re.M)
    for ext in _FAIL_STOP_PATTERN_DICT | _OTHER_EARLY_STOP_PATTERN_DICT})

# How many lines after a line of _EARLY_STOP_RE_DICT are kept with it as what the run
# said of why it ended
_EARLY_STOP_AFTER_LINE = 3

# What run_test reports as "status", one word per verdict (_test_verdict)
TEST_STATUS_DICT = {True: "passed", False: "failed", None: "did not run"}

# What a pattern of bun test's own lines that is indented like another tool's output
# (" 1 fail") is put after, so that it matches only in an output holding _BUN_HEAD_PATTERN
_ON_BUN_PATTERN = r"\A(?=[\s\S]*?" + _BUN_HEAD_PATTERN + r")[\s\S]*?"

# What a run prints once it got far enough for the exit code to be about the code: the
# reporter's own summary line, or the build failing on the project's own sources. It is
# the second of the two things that make a non-zero exit code count as red, the first
# being a failing test name from _FAIL_RE_DICT, and it is what catches a run that was red
# without naming any test. A pattern does not match what the same command prints in a
# repository with no tests or in one that was never set up. An extension that is not
# here relies on _FAIL_RE_DICT alone
_REPORT_RE_DICT = _with_alias_dict({
    # 1 failed, 2 passed in 0.05s / 1 error in 0.09s / no tests ran in 0.00s and
    # 1 skipped in 0.00s (no match) /
    # unittest's FAILED (failures=1)
    "py": re.compile(r"\b\d+ (passed|failed|errors?|xfailed|xpassed)\b"
                     r"|(?m:^FAILED " + _UNITTEST_COUNT_PATTERN + r")"),
    # [ERROR] Tests run: 2, Failures: 1, ... / [ERROR] COMPILATION ERROR :
    "java": re.compile(r"Tests run: \d+, Failures: \d+|COMPILATION ERROR"),
    # 2 tests completed, 1 failed / Execution failed for task ':compileKotlin'. /
    # Execution failed for task ':a:compileKotlin'. (a project of a multi-project build)
    # / ':a:kaptKotlin'. / ':a:kspKotlin'. (the annotation processing of the sources)
    "kt": re.compile(r"^\d+ tests? completed"
                     r"|^Execution failed for task ':(?:[^\s:']+:)*(?:compile|kapt|ksp)",
                     re.M),
    # Test Suites: 1 failed, 1 total. Printed for a suite that failed to run at all,
    # and not printed when jest found no test files. vitest's " Test Files  1 failed (1)"
    # is the same, and node --test: "# tests 3" (TAP) / "ℹ tests 3" (spec), with at
    # least one test. mocha: "  2 passing (5ms)" / "  1 failing" (a run whose test files
    # could not be loaded, " Exception during run: ...", ran no test and prints neither).
    # bun test: "Ran 3 tests across 2 files." (not printed when it found no test file)
    "js": re.compile(r"^Test Suites: |^ ?Test Files {2}\d|^(?:# |ℹ )tests [1-9]"
                     r"|^ +\d+ passing \(|^ +\d+ failing$|^Ran \d+ tests? across \d+ files?\.",
                     re.M),
    # 50% tests passed, 1 tests failed out of 2. Printed for a test whose executable
    # was never built as well. FAILED: CMakeFiles/t.dir/tests/t.c.o is Ninja's line for
    # a build step that failed, after which ctloom-test-c runs no test. make: ***
    # [Makefile:8: test] Error 1 / Aborted (core dumped) / Segmentation fault is make's
    # line for a step of a makefile that failed or was killed (ctloom-test-c leaves
    # them out of a run whose only failure is a command that is not installed). make:
    # *** No rule to make target 'src/a.c', needed by 'src/a.o'. is a source the
    # makefile names that is gone, and Makefile:12: *** missing separator.  Stop. is
    # make's own error in a makefile
    "c": re.compile(r"^\d+% tests passed, \d+ tests failed out of \d+|^FAILED: "
                    r"|^make(?:\[\d+\])?: \*\*\* \[[^\]\n]+\] \S"
                    r"|^make(?:\[\d+\])?: \*\*\* No rule to make target '[^'\n]+', "
                    r"needed by |^[^:\s]+:\d+: \*\*\* ", re.M),
    # test result: FAILED. 1 passed; 1 failed; ... / error: could not compile `calc`
    "rs": re.compile(r"^test result: FAILED\.|^error: could not compile ", re.M),
    # Failed!  - Failed:     1, Passed: ... (VSTest) / Test run summary: Failed!
    # (Microsoft.Testing.Platform) / src/A.cs(14,25): error CS0117: ...
    "cs": re.compile(r"^Failed!\s+- Failed:|^Test run summary: Failed!|: error CS\d+: ",
                     re.M),
    # [ FAIL 1 | WARN 0 | SKIP 0 | PASS 2 ] / ! Failed to load 'R/basic.R' (a source of
    # the package that does not parse)
    "r": re.compile(r"^\[ FAIL [1-9]\d* \| |^! Failed to load '", re.M),
})

# A summary line of a run that counts a failing test, one regex per language. A run that
# prints one is red whatever its exit code. An extension that is not here has none
_FAIL_COUNT_RE_DICT = _with_alias_dict({
    # pytest: 1 failed, 2 passed in 0.05s / 1 error in 0.09s; unittest: FAILED
    # (failures=1)
    "py": re.compile(r"^(?:=+ )?(?:\d+ [a-z]+(?: [a-z]+)*, )*[1-9]\d* (?:failed|errors?)\b"
                     r"(?:, \d+ [a-z]+(?: [a-z]+)*)* in \d+(?:\.\d+)?s\b"
                     r"|^FAILED " + _UNITTEST_COUNT_PATTERN, re.M),
    # A module's count after "Results:", not a class's line (", Time elapsed"), which
    # counts the first try of a test surefire ran again and passed:
    # [ERROR] Tests run: 2, Failures: 1, Errors: 0, Skipped: 0 /
    # [ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 0, Flakes: 1
    "java": re.compile(r"^\[(?:INFO|ERROR|WARNING)\] Tests run: \d+, (?:Failures: [1-9]\d*, "
                       r"Errors: \d+|Failures: \d+, Errors: [1-9]\d*), Skipped: \d+"
                       r"(?:, Flakes: \d+)?$", re.M),
    # 2 tests completed, 1 failed
    "kt": re.compile(r"^\d+ tests? completed, [1-9]\d* failed", re.M),
    # Tests:       1 failed, 1 passed, 2 total / vitest's "      Tests  1 failed | 1 passed
    # (2)" / # fail 1 / ℹ fail 1 / mocha's "  1 failing" / bun test's " 1 fail"
    # (_ON_BUN_PATTERN) and "Bailed out after 1 failure" (--bail, which prints no count)
    "js": re.compile(r"^Tests: +(?:.*, )?[1-9]\d* failed\b|^ +Tests {2}[1-9]\d* failed\b"
                     r"|^(?:# |ℹ )fail [1-9]|^ +[1-9]\d* failing$"
                     r"|" + _ON_BUN_PATTERN + r"^ +[1-9]\d* fail$"
                     r"|^Bailed out after [1-9]\d* failures?$", re.M),
    # 50% tests passed, 1 tests failed out of 2
    "c": re.compile(r"^\d+% tests passed, [1-9]\d* tests? failed out of ", re.M),
    # test result: FAILED. 1 passed; 1 failed; ...
    "rs": re.compile(r"^test result: FAILED\. ", re.M),
    # Failed!  - Failed:     1, Passed: ... (VSTest) /   failed: 1
    # (Microsoft.Testing.Platform)
    "cs": re.compile(r"^Failed!\s+- Failed:\s+[1-9]|^ +failed: [1-9]", re.M),
    # [ FAIL 1 | WARN 0 | SKIP 0 | PASS 2 ]
    "r": re.compile(r"^\[ FAIL [1-9]\d* \| ", re.M),
})

# A line a run prints where its tool ends with a non-zero exit code whatever its tests
# did - a test file could not be loaded, the project's sources could not be built, or
# the build failed - one regex per language. A run that prints one is read as one that
# ended with a non-zero exit code whatever its exit code. An extension that is not here
# has none
_FAIL_EXIT_RE_DICT = _with_alias_dict({
    # [INFO] BUILD FAILURE (printed under --fail-never as well, which ends with 0) /
    # [ERROR] COMPILATION ERROR :
    "java": re.compile(r"^\[INFO\] BUILD FAILURE$|COMPILATION ERROR", re.M),
    # Execution failed for task ':a:compileKotlin'. / ':a:kaptKotlin'. / ':a:kspKotlin'.
    "kt": re.compile(r"^Execution failed for task ':(?:[^\s:']+:)*(?:compile|kapt|ksp)",
                     re.M),
    # Test Suites: 1 failed, 1 passed, 2 total / vitest's " Test Files  1 failed | 1
    # passed (2)" (a test file that failed to run, or one with a failing test) / bun
    # test's " 1 error" (_ON_BUN_PATTERN: an error outside the tests, a test file that
    # failed to load)
    "js": re.compile(r"^Test Suites: +(?:.*, )?[1-9]\d* failed\b"
                     r"|^ ?Test Files {2}[1-9]\d* failed\b"
                     r"|" + _ON_BUN_PATTERN + r"^ +[1-9]\d* errors?$", re.M),
    # FAILED: CMakeFiles/t.dir/tests/t.c.o
    "c": re.compile(r"^FAILED: ", re.M),
    # error: could not compile `calc` / error: test failed, to rerun pass `--test basic`
    "rs": re.compile(r"^error: (?:could not compile |(?:doc)?test failed, to rerun pass )",
                     re.M),
    # src/A.cs(14,25): error CS0117: ...
    "cs": re.compile(r": error CS\d+: ", re.M),
    # ! Failed to load 'R/basic.R'
    "r": re.compile(r"^! Failed to load '", re.M),
})

# What a run prints to say it found nothing to run, one pair of regexes per language:
# (a line saying nothing was run, a line saying a test was run or None). A run found
# nothing when it printed the first line and, where the pair has a second, none of that
# one. A run that found nothing is "did not run" with a zero exit code as well: ctest
# ends that way for a project that registers no test, and maven and gradle end that way
# when the test phase had no sources. An extension that is not here has no such line.
# The same lines are read off a run that did not run: pytest's "no tests ran", jest's
# "No tests found", npm's "Missing script" and the "no test specified" of the script npm
# init writes
_NO_TEST_RE_DICT = _with_alias_dict({
    # pytest's summary that counts no test that was run - "no tests ran in 0.00s",
    # "1 skipped in 0.00s", "2 deselected, 1 warning in 0.01s", with or without the
    # "=====" around it - or unittest's "Ran 0 tests in 0.000s", with no "Ran N tests"
    # of N above 0 and no count pytest prints of tests that were run
    "py": (re.compile(r"^(?:=+ )?(?:no tests ran|\d+ (?:skipped|deselected|warnings?)"
                      r"(?:, \d+ (?:skipped|deselected|warnings?))*) in "
                      r"|^Ran 0 tests in ", re.M),
           re.compile(r"^Ran [1-9]\d* tests? in "
                      r"|\b[1-9]\d* (?:passed|failed|errors?|xfailed|xpassed)\b", re.M)),
    # "No tests to run." with no "Tests run: N" of N above 0: maven prints one of them
    # for each module
    "java": (re.compile(r"^\[INFO\] No tests to run\.$", re.M),
             re.compile(r"Tests run: [1-9]")),
    # "> Task :test NO-SOURCE" or "> Task :a:test NO-SOURCE", with no test task in any
    # other state: gradle prints one line for each project's test task
    "kt": (re.compile(r"^> Task \S*:test NO-SOURCE$", re.M),
           re.compile(r"^> Task \S*:test(?= |$)(?! NO-SOURCE$)", re.M)),
    # bun test: "error: 0 test files matching ...", "The following filters did not match
    # any test files ..." (paths it was given) and "error: regex "x" matched 0 tests. ..."
    # (its -t)
    "js": (re.compile(r"^No tests found\b|^(?:Error: )?No test files found\b"
                      r"|Missing script: \"test\""
                      r"|Error: no test specified|^(?:# |ℹ )tests 0$"
                      r"|^error: 0 test files matching |^The following filters did not match "
                      r"any test files|^error: regex .* matched 0 tests\.", re.M), None),
    "c": (re.compile(r"^No tests were found!!!$", re.M), None),
    # "running 0 tests" with no "running N tests" of N above 0: cargo prints one of
    # them for each target
    "rs": (re.compile(r"^running 0 tests$", re.M), re.compile(r"^running [1-9]", re.M)),
    # "No test is available in <dll>" (VSTest) or "Test run summary: Zero tests ran"
    # (Microsoft.Testing.Platform), with no test project that passed or failed: dotnet
    # prints one of them for each test project
    "cs": (re.compile(r"^(?:No test is available in |Test run summary: Zero tests ran)",
                      re.M),
           re.compile(r"^(?:Passed!|Failed!|Test run summary: (?:Passed|Failed))", re.M)),
    # [ FAIL 0 | WARN 0 | SKIP 1 | PASS 0 ] with no line of PASS above 0: ctloom-test-r
    # prints one line for each test file or directory it was given. "! No test files
    # found" is testthat's line for a tests directory that holds no test file
    "r": (re.compile(r"^(?:\[ FAIL 0 \| WARN \d+ \| SKIP \d+ \| PASS 0 \]"
                     r"|! No test files found)$", re.M),
          re.compile(r"^\[ FAIL \d+ \| WARN \d+ \| SKIP \d+ \| PASS [1-9]", re.M)),
})

# What a shell, or make running a command itself, prints for a command it could not
# find: "sh: 1: bun: not found", "bash: line 1: bun: command not found",
# "make: valgrind: No such file or directory"
MISSING_COMMAND_RE = re.compile(
    r"\b(?:sh|bash|dash)(?:: line \d+|: \d+)?: ([^/:\s]+): (?:command )?not found\b"
    r"|^make(?:\[\d+\])?: ([^/:\s]+): No such file or directory$", re.M)

# The part of a reason saying why a test command did not run: "`npm test` did not run
# (<what it printed>)", up to where the next reason starts ("; " or " / ") or the end
_NO_RUN_PART_RE = re.compile(r"` did not run \((.*?)\)(?= / |; `|$)", re.S)


def missing_command_list(text: str) -> list[str]:
    """Return the commands a test command did not run for lack of, in a reason.

    Only the parts saying why a test command did not run (_NO_RUN_PART_RE) are read.

    Args:
        text: A reason, as a stage's "why" holds it.

    Returns:
        The commands MISSING_COMMAND_RE names there, each once, in their order.
    """
    return list(dict.fromkeys(
        command for part in _NO_RUN_PART_RE.findall(text)
        for pair in MISSING_COMMAND_RE.findall(part) for command in pair if command))


def _npm_test_without_script(argv: list[str], run_dir: str) -> str:
    """Say why npm test ran no test, when package.json names no test script.

    npm test --silent prints nothing then, as it does where there is no package.json.

    Args:
        argv: The test command as it was run.
        run_dir: Where it was run.

    Returns:
        A line saying the directory's package.json names no "test" script, when the
        command is npm's test and that package.json can be read and has no "test" among
        its "scripts"; "" otherwise.
    """
    if len(argv) < 2 or os.path.basename(argv[0]) != "npm" or argv[1] != "test":
        return ""
    try:
        with open(os.path.join(run_dir, "package.json"), encoding="utf-8-sig") as file:
            package_dict = json.load(file)
    except (OSError, ValueError):
        return ""
    if not isinstance(package_dict, dict):
        return ""
    script_dict = package_dict.get("scripts")
    if isinstance(script_dict, dict) and script_dict.get("test"):
        return ""
    return 'package.json has no "test" script, so npm test runs no test'


# What says from the repository's files that a run which printed nothing found no test
# to run, one function per language: (the command as run, where it was run) -> the line
# saying so, or "". An extension that is not here has none
_NO_TEST_FILE_DICT = _with_alias_dict({"js": _npm_test_without_script})

# What a test command prints to say it ended before it got to the tests because the
# project is not set up, one regex per language. An extension that is not here has no
# such line
_NO_PROJECT_RE_DICT = {
    # [ERROR] The goal you specified requires a project to execute but there is no POM
    # in this directory (...)
    "java": re.compile(r"^\[ERROR\] .*there is no POM in this directory", re.M),
    # Directory '...' does not contain a Gradle build.
    "kt": re.compile(r"^Directory '.*' does not contain a Gradle build\.$", re.M),
    # npm test --silent prints nothing in a directory with no package.json (and in one
    # whose package.json names no test script, which _NO_TEST_FILE_DICT reads first)
    "js": re.compile(r"\A\s*\Z"),
    # ctloom-test-c: there is no CMakeLists.txt in ... (each line the launcher prints under
    # its own name is one for a run that built and ran no test)
    "c": re.compile(r"^ctloom-test-c: ", re.M),
    # error: could not find `Cargo.toml` in `...` or any parent directory
    "rs": re.compile(r"^error: could not find `Cargo\.toml` in ", re.M),
    # MSBUILD : error MSB1003: Specify a project or solution file.
    "cs": re.compile(r"^MSBUILD : error MSB1003: ", re.M),
    # ctloom-test-r: there is no tests/testthat in ... (each line the launcher prints under
    # its own name is one for a run that ran no test)
    "r": re.compile(r"^ctloom-test-r: ", re.M),
}

# What a test command prints to say it ended before it got to the tests because what the
# tests need could not be fetched, one regex per language. An extension that is not here
# has no such line
_FETCH_FAIL_RE_DICT = {
    # error: Request failed after 3 retries in 26.9s /
    #   Caused by: Failed to fetch: `https://pypi.org/simple/pytest-asyncio/` /
    #   cause: Failed to fetch: `https://pypi.org/simple/pytest-asyncio/`
    "py": re.compile(r"^error: Request failed after \d+ retries"
                     r"|^(?:error|[ \t]+(?:Caused by|cause)): Failed to fetch: ", re.M),
    # [ERROR] Plugin ...:maven-resources-plugin:3.3.1 or one of its dependencies could
    # not be resolved: / [ERROR] 	The following artifacts could not be resolved: ...
    "java": re.compile(r"^\[ERROR\] .*could not be resolved", re.M),
    # Plugin [id: '...', version: '...'] was not found in any of the following sources:
    "kt": re.compile(r"^Plugin \[id: .*\] was not found in any of the following sources:$",
                     re.M),
    # error: no matching package named `itoa` found
    "rs": re.compile(r"^error: no matching package named ", re.M),
    # .../Calc.Tests.csproj : error NU1301: Failed to retrieve information about ... /
    #   Failed to restore .../Calc.Tests.csproj (in 5.84 sec).
    "cs": re.compile(r": error NU\d+: |^[ \t]*Failed to restore ", re.M),
    # ctloom-test-r: these packages are not installed and could not be installed: ...
    "r": re.compile(r"^ctloom-test-r: these packages are not installed and could not be "
                    r"installed: ", re.M),
}


def _either_re_dict(*table_tuple: dict) -> dict:
    """Return one regex per extension that matches where any of the tables' regexes does.

    Args:
        table_tuple: Pattern tables, one regex per extension.

    Returns:
        {extension: the regexes of that extension joined as alternatives, with re.M}.
    """
    pattern_dict: dict[str, list[str]] = {}
    for table_dict in table_tuple:
        for ext, pattern in table_dict.items():
            pattern_dict.setdefault(ext, []).append(f"(?:{pattern.pattern})")
    return {ext: re.compile("|".join(pattern_list), re.M)
            for ext, pattern_list in pattern_dict.items()}


# What a test command prints to say it ended before it got to the tests - the project
# is not set up, or what the tests need could not be fetched. An extension that is not
# here has no such line
_STOP_SHORT_RE_DICT = _with_alias_dict(_either_re_dict(_NO_PROJECT_RE_DICT,
                                                       _FETCH_FAIL_RE_DICT))
_FETCH_FAIL_RE_DICT = _with_alias_dict(_FETCH_FAIL_RE_DICT)

# What a test command prints to say it could not load a file the tests under its
# directory need, and ran none of its tests, one regex per language with the file's path
# as the group: pytest's line for a conftest.py it could not import. An extension that
# is not here has no such line
_LOAD_FAIL_RE_DICT = _with_alias_dict({
    "py": re.compile(r"^ImportError while loading conftest '(.+)'\.?$", re.M)})

# The line a language's test command prints for each test file it runs, whichever file
# it names (the language's run_line_pattern), one regex per language that has one
_RUN_LINE_RE_DICT = {
    ext: re.compile(lang_config.run_line_pattern.replace("{name}", r"\S+"))
    for ext, lang_config in LANG_DICT.items() if lang_config.run_line_pattern}

# What a test command is given in its environment that makes it name each test that
# passed, one (variable, words added after what the variable already holds) per
# language: pytest's verbosity is set to 0 (--verbosity=0), which undoes a -q or -v of
# the ini file's addopts and of what the variable already holds, and the command line's
# alone counts; its short summary then holds every kind of line but the output of the
# passed tests ("a"), and a "PASSED <id>" line for each test that passed ("p"), and a
# test file that cannot be collected, or a directory below the ones named on the command
# line whose conftest.py cannot be imported, is named by an "ERROR <path>" line while
# the other tests run (--continue-on-collection-errors)
_RAN_ENV_DICT = _with_alias_dict({"py": ("PYTEST_ADDOPTS",
                                         "--verbosity=0 -rap --continue-on-collection-errors")})

# What a test command is given in its environment, as fixed values, one dict per
# language: CI=true, under which vitest and react-scripts run the tests once instead of
# waiting for file changes
_TEST_ENV_DICT = _with_alias_dict({"js": {"CI": "true"}})

# The line naming a test that passed that _RAN_ENV_DICT makes a command print, one
# regex per language, with the line end after it: pytest's "PASSED <id>", the rest of
# the line being the id (tests/api.yaml::Get user by id)
_PASS_LINE_RE_DICT = _with_alias_dict({
    "py": re.compile(r"^PASSED (\S(?:[^\n]*\S)?)[ \t]*(?:\n|$)", re.M)})

# The lines naming a test that did not fail, left out of what run_test returns, one regex
# per language, with the line end after it: pytest's PASSED lines (_PASS_LINE_RE_DICT),
# and bun test's "(pass) calc > adds [0.02ms]", "(skip) ..." and "(todo) ..."
_OMIT_LINE_RE_DICT = _with_alias_dict({
    "py": _PASS_LINE_RE_DICT["py"],
    "js": re.compile(r"^\((?:pass|skip|todo)\) [^\n]*(?:\n|$)", re.M)})

# pytest's summary of a run: 2 failed, 4 passed, 1 skipped, 2 subtests passed in 0.02s
# / ==== 1 xfailed, 1 warning in 0.01s (0:00:01) ====, which counts at least one of
# pytest's own words (a plugin's word beside them is read as well). "200 rows in 0.01s"
# is none
_PYTEST_WORD_PATTERN = (r"(?:passed|failed|errors?|skipped|deselected|xfailed|xpassed"
                        r"|warnings?|subtests (?:passed|failed|skipped)|tests? collected)")
_PYTEST_SUMMARY_RE = re.compile(
    r"^(?:=+ )?(?=(?:\d+ [a-z]+(?: [a-z]+)*, )*\d+ " + _PYTEST_WORD_PATTERN + r"(?:, | in ))"
    r"(\d+ [a-z]+(?: [a-z]+)*(?:, \d+ [a-z]+(?: [a-z]+)*)*) in \d+(?:\.\d+)?s\b",
    re.M)

# The words of pytest's summary that count tests that ran
_PYTEST_RAN_WORD_TUPLE = ("passed", "failed", "error", "errors", "xfailed", "xpassed")

# A failing subtest of pytest 9, which its summary counts in "failed" beside its test
_SUBFAIL_LINE_RE = re.compile(r"^SUBFAILED[\[(]", re.M)

# unittest's "Ran 3 tests in 0.001s", and its last line when it skipped some:
# OK (skipped=2) / FAILED (failures=1, skipped=2)
_UNITTEST_RAN_RE = re.compile(r"^Ran (\d+) tests? in ", re.M)
_UNITTEST_SKIP_RE = re.compile(r"^(?:OK|FAILED) \((?:[^)\n]*, )?skipped=(\d+)", re.M)

# unittest -v's line for a skip, with its indent as the group: "test_a
# (tests.test_m.T.test_a) ... skipped 'why'" for a test skipped whole, the same indented
# for a skipped subtest, and "<the first line of the test's docstring> ... skipped 'why'"
# for a test with a docstring, after a line of its own that holds no result
# (_UNITTEST_DESCRIBE_RE), indented for a subtest
_UNITTEST_SKIP_LINE_RE = re.compile(r"^([ \t]*)\S[^\n]*? \.\.\. skipped\b", re.M)
_UNITTEST_DESCRIBE_RE = re.compile(r"^([ \t]*)\S+ \([\w.]+\)(?: \S.*)?$")

# unittest -v's line for a class or module skipped in its fixture, whose tests "Ran N"
# does not count: "setUpClass (tests.test_m.T) ... skipped 'why'" / "setUpModule
# (tests.test_m) ... skipped 'why'"
_UNITTEST_FIXTURE_SKIP_RE = re.compile(
    r"(?:setUpClass|setUpModule|tearDownClass|tearDownModule) \([\w.]+\) \.\.\. skipped\b")

# unittest's line of one character per test where it is not -v, followed by the line of
# "=" its failures start with or by the line of "-" before "Ran N tests": "." a test that
# passed, "x" an expected failure, "s" a skip (a test, a subtest, a class or a module),
# "F", "E" and "u" the rest
_UNITTEST_PROGRESS_RE = re.compile(r"^([.sxuFE]+)\n(?:={70}|-{70})$", re.M)

# maven's count of a module's tests, after "Results:":
# [ERROR] Tests run: 2, Failures: 1, Errors: 0, Skipped: 1 (a class's own line goes on
# with ", Time elapsed") / [WARNING] Tests run: 2, Failures: 0, Errors: 0, Skipped: 0,
# Flakes: 1 (a test that failed and passed when surefire ran it again counts once)
_MAVEN_RAN_RE = re.compile(r"^\[(?:INFO|ERROR|WARNING)\] Tests run: (\d+), Failures: \d+, "
                           r"Errors: \d+, Skipped: (\d+)(?:, Flakes: \d+)?$", re.M)

# jest's "Tests:       1 failed, 1 skipped, 3 passed, 5 total", and node --test's
# "# pass 3" / "# fail 1" (TAP) and "ℹ pass 3" / "ℹ fail 1" (spec)
_JEST_RAN_RE = re.compile(r"^Tests: +(.*\d+ total)$", re.M)
# vitest's: "      Tests  1 failed | 1 passed (2)"
_VITEST_RAN_RE = re.compile(r"^ +Tests {2}(.*\(\d+\))$", re.M)
_NODE_RAN_RE = re.compile(r"^(?:# |ℹ )(?:pass|fail) (\d+)$", re.M)
# mocha's: "  2 passing (5ms)" / "  1 failing"
_MOCHA_RAN_RE = re.compile(r"^ +(\d+) (?:passing \(.*\)|failing)$", re.M)
# bun test's first line (_BUN_HEAD_PATTERN), its " 3 pass" / " 1 fail", and each of its
# count lines (" 1 skip", " 1 todo", " 1 error")
_BUN_HEAD_RE = re.compile(_BUN_HEAD_PATTERN, re.M)
_BUN_RAN_RE = re.compile(r"^ +(\d+) (?:pass|fail)$", re.M)
_BUN_COUNT_LINE_RE = re.compile(r"^ +\d+ (?:pass|fail|skip|todo|errors?)$", re.M)

# ctest's line for the end of a run, its line for each test (1/2 Test #1: FooTest.one
# ......   Passed    0.00 sec), and that line for a test it ran: Passed, ***Failed,
# ***Exception, ***Timeout, ***Not Run, not ***Skipped and not ***Not Run (Disabled)
_CTEST_END_RE = re.compile(r"^\d+% tests passed, \d+ tests failed out of (\d+)$", re.M)
_CTEST_TEST_RE = re.compile(r"^ *\d+/\d+ Test +#\d+: ", re.M)
_CTEST_RAN_RE = re.compile(r"^ *\d+/\d+ Test +#\d+: .*?(?: +|\*\*\*)"
                           r"(?:Passed|Failed|Exception|Timeout|Not Run(?! \(Disabled\)))\b",
                           re.M)

# cargo's "test result: ok. 2 passed; 0 failed; 1 ignored; ..." for each target
_CARGO_RAN_RE = re.compile(r"^test result: \w+\. (\d+) passed; (\d+) failed;", re.M)

# dotnet's "Passed!  - Failed:     0, Passed:     4, ..." for each test project
# (VSTest) and "  failed: 0" / "  succeeded: 4" (Microsoft.Testing.Platform)
_DOTNET_RAN_RE = re.compile(r"^(?:Passed|Failed)!\s+- Failed:\s+(\d+), Passed:\s+(\d+),"
                            r"|^ +(?:failed|succeeded): (\d+)$", re.M)

# cargo's line for each target it runs the tests of, on standard error, in the order of
# the "running N tests" lines on standard output: "Running unittests src/lib.rs
# (target/debug/deps/calc-f3ab8b63)", "Running tests/basic.rs (target/debug/deps/
# basic-3791478)" and "Doc-tests calc"; its "running 2 tests"; and its line for a test
# that passed or failed, "test tests::adds ... ok" (not "... ignored")
_CARGO_TARGET_RE = re.compile(r"^\s*(?:Running (?:unittests )?(\S+) \(\S*?deps/([\w-]+?)-[0-9a-f]+"
                              r"(?:\.exe)?\)|Doc-tests (\S+))$", re.M)
_CARGO_BLOCK_RE = re.compile(r"^running \d+ tests?$", re.M)
_CARGO_TEST_RE = re.compile(r"^test (.+?) \.\.\. (?:ok|FAILED)$", re.M)

# surefire's line for each test class, before "Results:":
# [INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 1, Time elapsed: 0.04 s -- in
# com.example.FooTest (and "- in" of an older surefire)
_MAVEN_CLASS_RE = re.compile(r"^\[(?:INFO|ERROR|WARNING)\] Tests run: (\d+), Failures: \d+, "
                             r"Errors: \d+, Skipped: (\d+), Time elapsed: [^\n]*? -?- in (\S+)$",
                             re.M)

# jest's line for each test file it ran: "PASS test/a.test.js" / "FAIL test/a.test.js
# (5.2 s)", " PASS " where its badge was colored
_JEST_FILE_RE = re.compile(r"^ ?(?:PASS|FAIL) +(\S+)(?: \(\d+(?:\.\d+)? ?m?s\))?$", re.M)
# vitest's: " ✓ test/a.test.js  (2 tests) 2ms" / " ❯ test/a.test.js (2 tests | 1 failed)"
# (one space before the count from vitest 2), with a project's " |unit| " in front in a
# workspace (the project's label, then the file's path from the project's root)
_VITEST_FILE_RE = re.compile(r"^ [✓❯] (\|[^|\n]+\| )?(\S+) {1,2}\(\d+ tests?\b", re.M)

# The lines of a pytest run's output that give it its shape, each matched against one
# line: the line a session starts with (searched for, as it follows what a test printed
# with -s on the same line), the head of its short summary, the line it ends with (its
# summary, or "no tests ran"), the line of "!" it says it stopped early with right
# before that, a line written between "=" or "_" that a section or a test's report
# starts with ("==== FAILURES ====", "____ test_x ____"), the head of what it took of
# a test's own output ("---- Captured stdout call ----", "---- Captured log setup ----"),
# the head of a section that holds its tests' reports with what it took of their output
# ("==== FAILURES ====", "==== ERRORS ====", "==== PASSES ====", "==== XFAILURES ====",
# "==== XPASSES ====", with the order a run prints them in), the line a test's report
# there starts with and a line of a run's progress under -q ("..F      [ 60%]",
# "..F  [ 3/5]", "..F", with the letters of subtests "u", "y", "," and "-" and of a test
# run again "R", and one letter that is not "," or "-" at least)
_PYTEST_START_RE = re.compile(r"=+ test session starts =+$")
_PYTEST_HEAD_RE = re.compile(r"=+ short test summary info =+$")
_PYTEST_END_RE = re.compile(_PYTEST_SUMMARY_RE.pattern + r"|^(?:=+ )?no tests ran in \d")
_PYTEST_STOP_RE = re.compile(r"!+ .+ !+$")
_PYTEST_SECTION_RE = re.compile(r"([=_])\1* \S(?:.*\S)? \1+$")
_PYTEST_CAPTURE_RE = re.compile(r"-+ Captured \S+ (?:setup|call|teardown) -+$")
_PYTEST_REPORT_RE = re.compile(r"=+ (X?FAILURES|ERRORS|X?PASSES) =+$")
_PYTEST_REPORT_TUPLE = ("ERRORS", "FAILURES", "XFAILURES", "PASSES", "XPASSES")
_PYTEST_TEST_HEAD_RE = re.compile(r"_+ \S(?:.*\S)? _+$")
_PYTEST_PROGRESS_RE = re.compile(
    r"(?=[,-]*[.FEsxXuyR])[.FEsxXuyR,-]+(?: +\[ *\d+(?:%|/\d+)\])?$")

# Where a run lists its failing tests under a head of their own, one entry per language:
# (the head, the regex a line before the first such head is read with in place of the
# language's _FAIL_RE_DICT one). pytest's FAILED, ERROR and SUBFAILED lines are read from
# the head of its short summary on, and before it unittest's FAIL: and ERROR: lines
# alone. An output a part of which was left out (_CUT_LINE_RE) is read with
# _FAIL_RE_DICT alone
_FAIL_HEAD_DICT = _with_alias_dict({
    "py": (_PYTEST_HEAD_RE, re.compile(_UNITTEST_FAIL_PATTERN))})

# Where a run names failing tests in a second way only when the summary of the harness
# that writes them is in the output, one entry per language: (that summary, the regex of
# such a line, the name as its group). automake's make check: "# TOTAL: 7" and
# "FAIL: tests/calc_test" / "ERROR: tests/io_test" (its summary's "# FAIL:  1" is not
# one)
_FAIL_IF_RE_DICT = _with_alias_dict({
    "c": (re.compile(r"^# TOTAL: +\d+$", re.M), re.compile(r"^(?:FAIL|ERROR): (\S+)"))})

# Where a run names what failed outside its tests only by the heading it prints the
# failure under, one pair per language: (the heading, with the name as its group; the
# line saying something under it failed). bun test: "test/a.test.ts:" ahead of each
# file's lines, and "# Unhandled error between tests" for a file that failed to load or
# whose describe threw
_FAIL_UNDER_HEAD_DICT = _with_alias_dict({"js": (_BUN_FILE_HEAD_RE, _BUN_ERROR_MARK_RE)})


def _pytest_block_list(line_list: list[str],
                       session_list: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return where the summary of each pytest session is among the lines of an output.

    Args:
        line_list: The output's lines.
        session_list: Its sessions (_pytest_session_list).

    Returns:
        [(the index of its first line, the index of its end line)] in order, one for each
        line _PYTEST_END_RE matches. Its first line is the head of its short summary
        (_PYTEST_HEAD_RE) where one lies after the end line before it, else the first of
        the lines of "!" (_PYTEST_STOP_RE) right above the end line, else the end line.
        A session whose start line (_pytest_session_list) lies after the end line before
        it, or else whose head does, after the head of another short summary that no
        end line has followed yet is printed inside that short summary: its first line
        is that start line (the line after it where the start line follows what was
        printed on the same line), else its head, and the other head is left for the
        end lines after it. The lines of such a session are no part of another's.
    """
    session_start_dict = {end: start for start, end in session_list}
    block_list = []
    inner_line_set = set()
    last_end = -1
    for index, line in enumerate(line_list):
        if not _PYTEST_END_RE.match(line):
            continue
        start = next((at for at in range(index - 1, last_end, -1)
                      if at not in inner_line_set and _PYTEST_HEAD_RE.match(line_list[at])),
                     None)
        session_start = session_start_dict.get(index, -1)
        first = session_start if session_start > last_end else start
        if first is not None and any(_PYTEST_HEAD_RE.match(line_list[at])
                                     for at in range(last_end + 1, first)
                                     if at not in inner_line_set):
            if first == session_start \
                    and _PYTEST_START_RE.search(line_list[session_start]).start() > 0:
                first += 1
            block_list.append((first, index))
            inner_line_set.update(range(first, index + 1))
            continue
        if start is None:
            start = index
            while start - 1 > last_end and _PYTEST_STOP_RE.match(line_list[start - 1]):
                start -= 1
        block_list.append((start, index))
        last_end = index
    return block_list


def _pytest_session_list(line_list: list[str],
                         report_set: set[int]) -> list[tuple[int, int]]:
    """Return the pytest sessions whose start line an output holds outside report sections.

    A start line (_PYTEST_START_RE) is paired with the next end line written between "="
    that no later start line was paired with first: a session that prints its start line
    ends with such a line, and one started inside it ends first. A session started right
    inside one whose start line no end line is paired with, inside a start line that
    begins its line, is left out: its end line may be that start line's. The lines of
    report_set, the sections that hold a run's tests' reports, are passed over.

    Args:
        line_list: The output's lines.
        report_set: The lines inside those sections (_pytest_report_line_set).

    Returns:
        [(the index of the start line, the index of the end line)].
    """
    open_list, session_list = [], []
    parent_dict = {}
    line_open_count = 0
    for index, line in enumerate(line_list):
        if index in report_set:
            continue
        if _PYTEST_START_RE.search(line):
            parent_dict[index] = open_list[-1] if line_open_count else -1
            open_list.append(index)
            line_open_count += bool(_PYTEST_START_RE.match(line))
        elif open_list and line.startswith("=") and _PYTEST_END_RE.match(line):
            start = open_list.pop()
            line_open_count -= bool(_PYTEST_START_RE.match(line_list[start]))
            session_list.append((start, index))
    open_set = set(open_list)
    return [(start, end) for start, end in session_list if parent_dict[start] not in open_set]


def _pytest_is_head(line: str) -> bool:
    """Say whether a line is the head of a section of a pytest run.

    Such a line is written between "=" (_PYTEST_SECTION_RE), and is neither a start line
    nor an end line.
    """
    return line.startswith("=") and bool(_PYTEST_SECTION_RE.match(line)) \
        and not _PYTEST_START_RE.match(line) and not _PYTEST_END_RE.match(line)


def _pytest_capture_list(line_list: list[str]) -> list[bool]:
    """Return, for each line of an output, whether it lies in what pytest took of a test.

    What a pytest run took of a test's output goes from the line after the head of it
    (_PYTEST_CAPTURE_RE) to the line before the next such head, test's head
    (_PYTEST_TEST_HEAD_RE) or head of a section (_pytest_is_head).
    """
    capture_list = []
    is_capture = False
    for line in line_list:
        is_mark = bool(_PYTEST_CAPTURE_RE.match(line))
        if is_mark or _PYTEST_TEST_HEAD_RE.match(line) or _pytest_is_head(line):
            capture_list.append(False)
            is_capture = is_mark
        else:
            capture_list.append(is_capture)
    return capture_list


def _pytest_end_dict(line_list: list[str]) -> dict[int, int | None]:
    """Return the end line of the session each start line of an output begins.

    From a start line on, the session's end line is the first end line written between
    "=" (_PYTEST_END_RE) outside what the session took of a test's output: from the line
    after the head of it (_PYTEST_CAPTURE_RE) to the line before the next such head,
    test's head (_PYTEST_TEST_HEAD_RE) or head of a section (_pytest_is_head). A session
    whose start line lies in what was taken is passed over whole, and a start line that
    begins its line outside it says the session has no end line.

    Args:
        line_list: The output's lines.

    Returns:
        {the index of a start line: the index of its end line, or None where it has
        none}.
    """
    # The end line reached from a line on, read outside and inside what was taken
    free_end_list: list[int | None] = [None] * (len(line_list) + 1)
    capture_end_list: list[int | None] = [None] * (len(line_list) + 1)
    end_dict: dict[int, int | None] = {}
    for index in range(len(line_list) - 1, -1, -1):
        line = line_list[index]
        after = index + 1
        if _PYTEST_START_RE.search(line):
            end_dict[index] = free_end_list[after]
            free_end_list[index] = None if _PYTEST_START_RE.match(line) \
                else free_end_list[after]
            capture_end_list[index] = capture_end_list[after] if end_dict[index] is None \
                else capture_end_list[end_dict[index] + 1]
        elif line.startswith("=") and _PYTEST_END_RE.match(line):
            free_end_list[index] = index
            capture_end_list[index] = capture_end_list[after]
        elif _PYTEST_CAPTURE_RE.match(line):
            free_end_list[index] = capture_end_list[index] = capture_end_list[after]
        elif _PYTEST_TEST_HEAD_RE.match(line) or _pytest_is_head(line):
            free_end_list[index] = capture_end_list[index] = free_end_list[after]
        else:
            free_end_list[index] = free_end_list[after]
            capture_end_list[index] = capture_end_list[after]
    return end_dict


def _pytest_end_ahead_tuple(line_list: list[str], capture_list: list[bool],
                            end_dict: dict[int, int | None]
                            ) -> tuple[tuple[list[bool], list[bool]],
                                       tuple[list[bool], list[bool]]]:
    """Return, for each line of an output, whether its pytest run's end line is to come.

    From a line on, the lines are read up to a start line that begins its line outside
    what a run took of a test's output (capture_list), where none is to come, or an end
    line outside it, where one is; a session whose start line lies in what was taken is
    passed over whole (end_dict). From a line that follows a session a test printed in
    what was taken, the lines are first what the test printed after the session, up to
    the next head of what was taken (_PYTEST_CAPTURE_RE), test's head
    (_PYTEST_TEST_HEAD_RE) or head of a section (_pytest_is_head): a session printed
    there is passed over whole, and a start line that begins its line there and has no
    end line says none is to come.

    Args:
        line_list: The output's lines.
        capture_list: _pytest_capture_list()'s value.
        end_dict: _pytest_end_dict()'s value.

    Returns:
        (from any line, from a line that follows such a session), each (for an end line
        of any kind, for an end line written between "="), each a list one longer than
        line_list whose last item is False.
    """
    count = len(line_list)
    # Read from a head on, and right after a session; each (any kind, written between "=")
    head_pair = ([False] * (count + 1), [False] * (count + 1))
    after_pair = ([False] * (count + 1), [False] * (count + 1))
    for index in range(count - 1, -1, -1):
        line = line_list[index]
        end = end_dict.get(index)
        is_start = bool(_PYTEST_START_RE.search(line))
        for kind in (0, 1):
            if capture_list[index]:
                head_pair[kind][index] = head_pair[kind][index + 1 if end is None else end + 1]
            elif _PYTEST_START_RE.match(line):
                head_pair[kind][index] = False
            elif _PYTEST_END_RE.match(line):
                head_pair[kind][index] = not kind or line.startswith("=") \
                    or head_pair[kind][index + 1]
            else:
                head_pair[kind][index] = head_pair[kind][index + 1]
            if is_start and end is not None:
                after_pair[kind][index] = after_pair[kind][end + 1]
            elif is_start and _PYTEST_START_RE.match(line):
                after_pair[kind][index] = False
            elif _PYTEST_CAPTURE_RE.match(line) or _PYTEST_TEST_HEAD_RE.match(line) \
                    or _pytest_is_head(line):
                after_pair[kind][index] = head_pair[kind][index]
            else:
                after_pair[kind][index] = after_pair[kind][index + 1]
    return head_pair, after_pair


def _pytest_report_line_set(line_list: list[str]) -> set[int]:
    """Return the lines of an output inside the sections that hold pytest's test reports.

    The sections are headed by _PYTEST_REPORT_RE; none of their lines gives a run its
    shape. Such a section goes on from its head up to the next head of a section
    (_pytest_is_head), start line that begins its line, or end line outside what the run
    took of a test's output (_pytest_capture_list): the run's own, where it prints no
    short summary. A session that a test printed in what was taken (_pytest_end_dict) is
    part of the section as a whole where the run's own end line is still to come after
    it (_pytest_end_ahead_tuple): one written between "=" for a run that printed its
    start line. Any other start line there is a line the test printed, but one that
    begins its line after an end line printed in the same block of what was taken begins
    another run, as does a line of a run's progress under -q there (_PYTEST_PROGRESS_RE)
    and the head of such a section that a run prints before this one or as this one
    (_pytest_report_end): every end line printed in the section is then read as a run's
    own, as it is where the run's own end line is not still to come from where the
    section ends.

    Args:
        line_list: The output's lines.

    Returns:
        The indexes of those lines.
    """
    line_set: set[int] = set()
    if not any(_PYTEST_REPORT_RE.match(line) for line in line_list):
        return line_set
    capture_list = _pytest_capture_list(line_list)
    end_dict = _pytest_end_dict(line_list)
    ahead_tuple = _pytest_end_ahead_tuple(line_list, capture_list, end_dict)
    is_eq = False
    index = 0
    while index < len(line_list):
        line = line_list[index]
        report = _PYTEST_REPORT_RE.match(line)
        if report:
            index = _pytest_report_end(line_list, index + 1,
                                       _PYTEST_REPORT_TUPLE.index(report.group(1)), is_eq,
                                       end_dict, ahead_tuple, line_set)
            continue
        if _PYTEST_START_RE.match(line):
            is_eq = True
        elif _PYTEST_END_RE.match(line):
            is_eq = False
        index += 1
    return line_set


def _pytest_report_end(line_list: list[str], index: int, rank: int, is_eq: bool,
                       end_dict: dict[int, int | None],
                       ahead_tuple: tuple[tuple[list[bool], list[bool]],
                                          tuple[list[bool], list[bool]]],
                       line_set: set[int]) -> int:
    """Put the lines of one section of pytest's test reports in line_set; return its end.

    The section is read as _pytest_report_line_set() reads it. The section ends at
    another run's start where a start line or a line of a run's progress under -q ends
    it, or the head of such a section that the run prints before this one or as this one
    (_PYTEST_REPORT_TUPLE). Every end line printed in it is then read as a run's own, as
    it is where the run's own end line is not still to come from where the section ends.

    Args:
        line_list: The output's lines.
        index: The line after the section's head.
        rank: The index of the section's title in _PYTEST_REPORT_TUPLE.
        is_eq: Whether the run printed its start line.
        end_dict: _pytest_end_dict()'s value.
        ahead_tuple: _pytest_end_ahead_tuple()'s value.
        line_set: The lines found so far, added to.

    Returns:
        The index of the line the section ends before; len(line_list) where it goes on
        to the end of the output.
    """
    head_pair, after_pair = ahead_tuple
    # Whether an end line was printed in the block of what was taken being read
    is_capture = is_new = is_block_end = False
    print_end_list = []
    while index < len(line_list):
        line = line_list[index]
        end = end_dict.get(index)
        if is_capture and end is not None and after_pair[is_eq][end + 1]:
            line_set.update(range(index, end + 1))
            index = end + 1
            continue
        report = _PYTEST_REPORT_RE.match(line)
        is_start = bool(_PYTEST_START_RE.match(line))
        is_new = (is_start and not is_capture) \
            or (is_block_end and (is_start or bool(_PYTEST_PROGRESS_RE.match(line)))) \
            or (report is not None and _PYTEST_REPORT_TUPLE.index(report.group(1)) <= rank)
        if is_new or _pytest_is_head(line) or (not is_capture and _PYTEST_END_RE.match(line)):
            break
        if _PYTEST_CAPTURE_RE.match(line) or _PYTEST_TEST_HEAD_RE.match(line):
            is_capture = bool(_PYTEST_CAPTURE_RE.match(line))
            is_block_end = False
        elif _PYTEST_END_RE.match(line):
            is_block_end = True
            print_end_list.append(index)
        line_set.add(index)
        index += 1
    if is_new or not head_pair[is_eq][index]:
        line_set.difference_update(print_end_list)
    return index


def _pytest_inner_set(line_list: list[str], block_list: list[tuple[int, int]],
                      session_list: list[tuple[int, int]], report_set: set[int]) -> set[int]:
    """Return the summaries of an output that are those of a pytest session a test ran.

    Such a session's output is printed inside the run's: in what the run took of a
    test's output, among its progress where it took none (-s), or in the message of a
    failing test in its short summary. The last summary is the run's own. An earlier
    one is a test's when:
    1. it lies inside the summary of a later one (_pytest_block_list);
    2. it ends inside a session whose start line the output holds
       (_pytest_session_list), unless it starts with the head of a short summary and
       the next summary has none (the head may be the next one's);
    3. it ends a session whose start line follows what was printed on the same line;
    4. it ends inside a section that holds a run's tests' reports
       (_pytest_report_line_set).
    Any other is the summary of a run of its own: a command that runs pytest more than
    once.

    Args:
        line_list: The output's lines.
        block_list: Its summaries (_pytest_block_list).
        session_list: Its sessions (_pytest_session_list).
        report_set: The lines inside those sections.

    Returns:
        The indexes into block_list of the summaries that are a test's.
    """
    after_print_end_set = {end for start, end in session_list
                           if _PYTEST_START_RE.search(line_list[start]).start() > 0}
    inner_set = set()
    next_start = len(line_list)
    for index in range(len(block_list) - 2, -1, -1):
        start, end = block_list[index]
        next_start = min(next_start, block_list[index + 1][0])
        is_head_shared = bool(_PYTEST_HEAD_RE.match(line_list[start])) \
            and not _PYTEST_HEAD_RE.match(line_list[next_start])
        if next_start < start \
                or (not is_head_shared
                    and any(session_start < end < session_end
                            for session_start, session_end in session_list)) \
                or end in after_print_end_set or end in report_set:
            inner_set.add(index)
    return inner_set


def _py_own_text(out: str) -> str:
    """Return the lines of a Python test run's output that are the run's own report.

    A pytest run's own are the short summary and the summary line of each of its
    sessions (_pytest_block_list, _pytest_inner_set): from the head of its short
    summary, or where it has none from the end of the run's summary before it, to its
    summary line, without the summaries of the sessions its tests ran; a line of "!"
    (_PYTEST_STOP_RE) is one of them in its last summary alone. A unittest run -
    one whose "Ran N tests" line comes after every pytest summary line - is everything
    but those summaries.

    Args:
        out: The command's stdout followed by its stderr, as run_in_project() returns
            them, without the control sequences a terminal acts on (_plain_text).

    Returns:
        Those lines, joined; out itself when it holds no pytest summary line and when a
        part of it was left out (_CUT_LINE_RE).
    """
    if _CUT_LINE_RE.search(out):
        return out
    line_list = out.split("\n")
    report_set = _pytest_report_line_set(line_list)
    session_list = _pytest_session_list(line_list, report_set)
    block_list = _pytest_block_list(line_list, session_list)
    if not block_list:
        return out
    ran_index = max((index for index, line in enumerate(line_list)
                     if _UNITTEST_RAN_RE.match(line)), default=-1)
    inner_set = set(range(len(block_list))) if ran_index > block_list[-1][1] \
        else _pytest_inner_set(line_list, block_list, session_list, report_set)
    inner_line_set = {line_index for index in inner_set
                      for line_index in range(block_list[index][0], block_list[index][1] + 1)}
    if len(inner_set) == len(block_list):
        return "\n".join(line for index, line in enumerate(line_list)
                         if index not in inner_line_set)
    keep_list = []
    after = 0
    for index, (start, end) in enumerate(block_list):
        if index in inner_set:
            continue
        first = start if _PYTEST_HEAD_RE.match(line_list[start]) else after
        is_last = index == len(block_list) - 1
        keep_list += [line_list[line_index] for line_index in range(first, end + 1)
                      if line_index not in inner_line_set
                      and (is_last or not _PYTEST_STOP_RE.match(line_list[line_index]))]
        after = end + 1
    return "\n".join(keep_list)


# The lines of a run's output that are the run's own report, one reader per language:
# (the command's stdout followed by its stderr, as run_in_project() returns them,
# without the control sequences a terminal acts on) -> those lines. An extension that is
# not here reads the whole output
_OWN_TEXT_DICT = _with_alias_dict({"py": _py_own_text})


def _own_text(out: str, ext: str) -> str:
    """Return the lines of a test command's output that are its run's own report.

    Args:
        out: The command's stdout followed by its stderr, as run_in_project() returns
            them, without the control sequences a terminal acts on (_plain_text).
        ext: The language the output is read as (_OWN_TEXT_DICT).

    Returns:
        The reader's lines, or out for a language with no reader.
    """
    reader = _OWN_TEXT_DICT.get(ext)
    return reader(out) if reader is not None else out


def _pytest_count_dict(out: str) -> dict[str, int] | None:
    """Return what pytest's summary lines of a run count, word by word.

    Examples:
        "2 failed, 4 passed, 2 subtests passed in 0.02s"
            -> {"failed": 2, "passed": 4, "subtests passed": 2}

    Args:
        out: The text read as a test command's output.

    Returns:
        {word: count}, summed over every summary line; None when there is none.
    """
    count_dict: dict[str, int] = {}
    match_list = list(_PYTEST_SUMMARY_RE.finditer(out))
    if not match_list:
        return None
    for match in match_list:
        for pair in match.group(1).split(", "):
            count, word = pair.split(" ", 1)
            count_dict[word] = count_dict.get(word, 0) + int(count)
    return count_dict


def _py_ran_count(out: str) -> int | None:
    """Return how many tests a Python run ran: pytest's or unittest's count.

    Args:
        out: The text read as a test command's output.

    Returns:
        pytest's passed, failed, error(s), xfailed and xpassed less its SUBFAILED lines,
        else the larger of unittest's "Ran N" less the tests it skipped
        (_unittest_skip_tuple), at least 0, and the "." and "x" of its lines of one
        character per test (_UNITTEST_PROGRESS_RE), each summed over the lines; None
        when the output holds neither summary.
    """
    count_dict = _pytest_count_dict(out)
    if count_dict is not None:
        return sum(count_dict.get(word, 0) for word in _PYTEST_RAN_WORD_TUPLE) \
            - len(_SUBFAIL_LINE_RE.findall(out))
    ran_list = _UNITTEST_RAN_RE.findall(out)
    if not ran_list:
        return None
    progress_count = sum(line.count(".") + line.count("x")
                         for line in _UNITTEST_PROGRESS_RE.findall(out))
    return max(sum(map(int, ran_list)) - _unittest_skip_tuple(out)[0], progress_count, 0)


def _unittest_skip_tuple(out: str) -> tuple[int, bool]:
    """Return how many tests a unittest run skipped whole, and whether that is known.

    Its "skipped=K" counts a skipped subtest and a class or module skipped in its
    fixture as well as a test skipped whole; its -v lines tell them apart
    (_UNITTEST_SKIP_LINE_RE, _UNITTEST_FIXTURE_SKIP_RE). A docstring's line is read
    with the indent of the test's own line before it (_UNITTEST_DESCRIBE_RE).

    Args:
        out: The text read as a test command's output.

    Returns:
        (the tests skipped whole, True) where the -v lines are as many as the K of
        every "skipped=K"; else (the sum of the K, False).
    """
    skip_count = sum(map(int, _UNITTEST_SKIP_RE.findall(out)))
    whole_count = line_count = 0
    for match in _UNITTEST_SKIP_LINE_RE.finditer(out):
        line_count += 1
        if _UNITTEST_FIXTURE_SKIP_RE.match(out, match.start()):
            continue
        indent = match.group(1)
        end = match.start() - 1
        previous_line = out[out.rfind("\n", 0, end) + 1:end] if end >= 0 else ""
        describe = _UNITTEST_DESCRIBE_RE.match(previous_line)
        if not indent and describe is not None and " ... " not in previous_line:
            indent = describe.group(1)
        if not indent:
            whole_count += 1
    if line_count == skip_count:
        return whole_count, True
    return skip_count, False


def _java_ran_count(out: str) -> int | None:
    """Return how many tests a maven run ran: each module's run less skipped, summed."""
    pair_list = _MAVEN_RAN_RE.findall(out)
    return sum(int(run) - int(skip) for run, skip in pair_list) if pair_list else None


def _js_ran_count(out: str) -> int | None:
    """Return how many tests a jest, vitest, mocha, node --test or bun test run ran.

    The passed and the failed are counted.
    """
    line_list = _JEST_RAN_RE.findall(out) or _VITEST_RAN_RE.findall(out)
    if line_list:
        return sum(int(count) for line in line_list
                   for count, word in re.findall(r"(\d+) (\w+)", line)
                   if word in ("passed", "failed"))
    count_list = _MOCHA_RAN_RE.findall(out) or _NODE_RAN_RE.findall(out) \
        or (_BUN_RAN_RE.findall(out) if _BUN_HEAD_RE.search(out) else [])
    return sum(map(int, count_list)) if count_list else None


def _c_ran_count(out: str) -> int | None:
    """Return how many tests a ctest run ran.

    The count is that of its lines of a test that was run, or the count its end line
    says where it printed no line for a test.
    """
    total_list = _CTEST_END_RE.findall(out)
    if not total_list:
        return None
    if not _CTEST_TEST_RE.search(out):
        return sum(map(int, total_list))
    return len(_CTEST_RAN_RE.findall(out))


def _rs_ran_count(out: str) -> int | None:
    """Return how many tests a cargo run ran: the passed and the failed of each target."""
    pair_list = _CARGO_RAN_RE.findall(out)
    return sum(int(pass_count) + int(fail_count) for pass_count, fail_count in pair_list) \
        if pair_list else None


def _cs_ran_count(out: str) -> int | None:
    """Return how many tests a dotnet run ran: the failed and the passed of each project."""
    match_list = _DOTNET_RAN_RE.findall(out)
    return sum(int(count) for group_tuple in match_list for count in group_tuple if count) \
        if match_list else None


# How many tests a run ran - those that passed or failed, not those it skipped or left
# out - one reader per language: (the text read as the command's output) -> the count,
# or None when the output does not say. An extension that is not here has no count
_RAN_COUNT_DICT = _with_alias_dict({
    "py": _py_ran_count, "java": _java_ran_count, "js": _js_ran_count,
    "c": _c_ran_count, "rs": _rs_ran_count, "cs": _cs_ran_count})

# The lines that say which and how many tests a run ran, per language
_RAN_LINE_RE_DICT = _with_alias_dict({
    "py": (_PASS_LINE_RE_DICT["py"], _PYTEST_SUMMARY_RE, _SUBFAIL_LINE_RE,
           _UNITTEST_RAN_RE, _UNITTEST_SKIP_RE, _UNITTEST_SKIP_LINE_RE, _UNITTEST_DESCRIBE_RE,
           re.compile(r"^(?:[.sxuFE]+|={70}|-{70})$")),
    "java": (_MAVEN_RAN_RE, _MAVEN_CLASS_RE),
    "js": (_JEST_RAN_RE, _VITEST_RAN_RE, _MOCHA_RAN_RE, _NODE_RAN_RE, _BUN_HEAD_RE,
           _BUN_COUNT_LINE_RE, _JEST_FILE_RE, _VITEST_FILE_RE),
    "c": (_CTEST_END_RE, _CTEST_RAN_RE),
    "rs": (_CARGO_RAN_RE, _CARGO_TARGET_RE, _CARGO_BLOCK_RE, _CARGO_TEST_RE),
    "cs": (_DOTNET_RAN_RE,)})


def _ran_count(out: str, ext: str) -> int | None:
    """Return how many tests a run of one language's test command ran (_RAN_COUNT_DICT).

    Args:
        out: The text read as a test command's output.
        ext: The language the output is read as.

    Returns:
        The count, or None for an output that does not say and a language with no
        reader.
    """
    reader = _RAN_COUNT_DICT.get(ext)
    return reader(out) if reader is not None else None


def _pytest_ran_name_list(out: str, fail_list: list[str]) -> list[str] | None:
    """Return the tests a pytest run ran: its PASSED lines and its failing tests' ids.

    Returns:
        The names, or None where the PASSED lines are fewer or more than its summary
        counts as passed. A failing name that is no test of a file ("<file>::<name>")
        is left out.
    """
    count_dict = _pytest_count_dict(out)
    if count_dict is None:
        return None
    pass_list = [match.group(1) for match in _PASS_LINE_RE_DICT["py"].finditer(out)]
    if len(pass_list) != count_dict.get("passed", 0):
        return None
    return pass_list + [name for name in fail_list if "::" in name]


def _cargo_ran_name_list(out: str, fail_list: list[str]) -> list[str] | None:
    """Return the tests a cargo run ran, each "<the target's source>::<test> [<target>]".

    The tests are read block by block ("running N tests"), the n-th block being the
    test of the n-th target line. A doctest's source is the part of its name before
    " - ".

    Examples:
        "test tests::adds ... ok" under "Running unittests src/lib.rs (target/debug/deps/
        calc-f3ab)" -> "src/lib.rs::tests::adds [calc]"

    Returns:
        The names, or None where the blocks and the target lines are not as many, a
        block has no "test result:" line (its binary ended before its summary, after
        however many of its tests), or the names are fewer or more than the "test
        result:" lines count.
    """
    target_list = _CARGO_TARGET_RE.findall(out)
    block_text_list = _CARGO_BLOCK_RE.split(out)[1:]
    if len(target_list) != len(block_text_list) \
            or any("\ntest result:" not in block_text for block_text in block_text_list):
        return None
    name_list = []
    for (path, deps_name, doc_name), block_text in zip(target_list, block_text_list):
        for test in _CARGO_TEST_RE.findall(block_text.split("\ntest result:")[0]):
            source = path or test.split(" - ", 1)[0]
            name_list.append(f"{source}::{test} [{deps_name or doc_name}]")
    return name_list if len(name_list) == _rs_ran_count(out) else None


# The names of the tests a run ran, where its output names each, one reader per
# language: (the command's output as _own_text() returns it, its failing names) -> the
# names, or None
_RAN_NAME_DICT = _with_alias_dict({"py": _pytest_ran_name_list, "rs": _cargo_ran_name_list})


def _ran_name_list(out: str, ext: str, fail_list: list[str],
                   directory: str) -> list[str] | None:
    """Return the names of the tests a run ran, where the output names every one.

    Args:
        out: The command's output as _own_text() returns it.
        ext: The language the output is read as (_RAN_NAME_DICT).
        fail_list: The failing names read off the output (_FAIL_RE_DICT).
        directory: Where the command was started, relative to the repository root.

    Returns:
        The names, each once and sorted; a Python test's name, whose file is relative
        to directory, with directory in front of it (a cargo name's source is relative
        to its package). None for a language with no reader and where the reader says
        None.
    """
    reader = _RAN_NAME_DICT.get(ext)
    name_list = reader(out, fail_list) if reader is not None else None
    if name_list is None:
        return None
    return sorted({posixpath.join(directory, name) if directory and ext == "py" else name
                   for name in name_list})


def _pytest_is_name_short(out: str, fail_list: list[str]) -> bool:
    """Say whether a pytest run's output does not name each test its summary counts.

    That is so where the summary counts failing tests (failed or error) while no failing
    test's name was read, or the run prints PASSED lines that do not name as many tests
    as its summary counts as passed (a line whose id is not read included).
    """
    count_dict = _pytest_count_dict(out)
    if count_dict is None:
        return False
    if not fail_list and any(count_dict.get(word, 0) for word in ("failed", "error", "errors")):
        return True
    if not re.search(r"^PASSED ", out, re.M):
        return False
    return len(_PASS_LINE_RE_DICT["py"].findall(out)) != count_dict.get("passed", 0)


# Whether a run's output does not name each test it counts, one reader per language:
# (the command's output as _own_text() returns it, the failing names read off it) -> the
# answer. An extension that is not here is never so
_NAME_SHORT_DICT = _with_alias_dict({"py": _pytest_is_name_short})

# pytest's exit statuses of a run that ran its tests (every test passed, some failed),
# and of a run that collected no test
_PYTEST_RAN_EXIT_TUPLE = (0, 1)
_PYTEST_NO_TEST_EXIT = 5

# pytest's line for a run over no test: "no tests ran in 0.00s", with or without the
# "=====" around it
_PYTEST_NO_RAN_RE = re.compile(r"^(?:=+ )?no tests ran in \d", re.M)


def _pytest_is_mark_line(line: str) -> bool:
    """Say whether a line is one only pytest's own report holds.

    Such a line is the line starting the session, the head of the short summary, a line
    of the progress that ends in how far the run is ("..s   [100%]", "..s   [3/5]"), or
    the line saying the run stopped after a count of failing tests (-x, --maxfail:
    _FAIL_STOP_RE_DICT).
    """
    return bool(_PYTEST_START_RE.match(line) or _PYTEST_HEAD_RE.match(line)
                or (line.endswith("]") and _PYTEST_PROGRESS_RE.match(line))
                or _FAIL_STOP_RE_DICT["py"].match(line))


def _pytest_is_unread(out: str, returncode: int) -> bool:
    """Say whether a pytest run prints no line counting the tests it ran.

    That is so where there is no summary line, no "no tests ran" line and no unittest
    "Ran N tests" line, and the run either ended with one of _PYTEST_RAN_EXIT_TUPLE (a
    run whose process ended in the middle of its tests among them) and printed a line of
    pytest's own report (_pytest_is_mark_line), or collected no test
    (_PYTEST_NO_TEST_EXIT) and printed nothing.
    """
    if _PYTEST_SUMMARY_RE.search(out) or _PYTEST_NO_RAN_RE.search(out) \
            or _UNITTEST_RAN_RE.search(out):
        return False
    if returncode == _PYTEST_NO_TEST_EXIT:
        return not out.strip()
    return returncode in _PYTEST_RAN_EXIT_TUPLE \
        and any(_pytest_is_mark_line(line) for line in out.split("\n"))


# Whether a run's output says nothing of how many tests it ran while it is a run of the
# language's that says so, one reader per language: (the command's output as _own_text()
# returns it, its exit status) -> the answer. An extension that is not here is never so
_UNREAD_DICT = _with_alias_dict({"py": _pytest_is_unread})

# The programs and the module that start pytest on a command line, and the short flags
# of no value that stand together with -q and -v in one word (-qx)
_PYTEST_PROGRAM_TUPLE = ("pytest", "py.test")
_PYTEST_MODULE = "pytest"
_PYTEST_FLAG_CHAR_SET = frozenset("qvxslhV")

# The lowest verbosity at which pytest prints the line that counts its tests
_PYTEST_COUNT_VERBOSITY = -1


def _pytest_word_list(argv: list[str]) -> list[str] | None:
    """Return the words of a command line that pytest reads as its own options.

    Examples:
        ["uv", "run", "--with", "pytest", "pytest", "-q", "tests"] -> ["-q", "tests"]
        ["python", "-m", "pytest", "-qq", "--", "-q"]              -> ["-qq"]
        ["make", "test"]                                           -> None

    Args:
        argv: The command line.

    Returns:
        The words after the last word that starts pytest (a program named in
        _PYTEST_PROGRAM_TUPLE, or "-m pytest"), up to "--"; None where no word does.
    """
    word_list = None
    for index, word in enumerate(argv):
        if posixpath.basename(word) in _PYTEST_PROGRAM_TUPLE or word == f"-m{_PYTEST_MODULE}":
            word_list = argv[index + 1:]
        elif word == "-m" and argv[index + 1:index + 2] == [_PYTEST_MODULE]:
            word_list = argv[index + 2:]
    if word_list is None:
        return None
    return word_list[:word_list.index("--")] if "--" in word_list else word_list


def _pytest_verbosity(word_list: list[str]) -> int:
    """Return the verbosity pytest's own options on a command line set, from 0.

    Examples:
        ["-qq"]                    -> -2
        ["-q", "-x", "--quiet"]    -> -2
        ["--verbosity=-2", "-v"]   -> -1
        ["-rqq"]                   -> 0

    Args:
        word_list: The words _pytest_word_list() returns, in order.

    Returns:
        The value of the last --verbosity, plus one for each -v and --verbose after it
        and less one for each -q and --quiet after it. A word of short flags is read up
        to its first letter outside _PYTEST_FLAG_CHAR_SET, which takes the rest as its
        value.
    """
    level = 0
    for index, word in enumerate(word_list):
        if word == "--verbosity" or word.startswith("--verbosity="):
            value = word.partition("=")[2] if "=" in word else \
                "".join(word_list[index + 1:index + 2])
            try:
                level = int(value)
            except ValueError:
                continue
        elif word in ("--quiet", "--verbose"):
            level += 1 if word == "--verbose" else -1
        elif word.startswith("-") and not word.startswith("--"):
            for char in word[1:]:
                if char not in _PYTEST_FLAG_CHAR_SET:
                    break
                level += {"q": -1, "v": 1}.get(char, 0)
    return level


def quiet_test_error() -> str:
    """Return the line refusing a test command whose words stop pytest counting its tests.

    "" when no command does. The commands checked are TEST_COMMAND, read as
    config.PRIMARY_EXT's, and those of config.OTHER_TEST_LIST, each where its language's
    run is pytest's (_RAN_ENV_DICT) and its words start pytest (_pytest_word_list).

    Returns:
        The line naming the first such command whose words set pytest's verbosity below
        _PYTEST_COUNT_VERBOSITY (_pytest_verbosity), and what to change, or "".
    """
    for ext, argv in [(config.PRIMARY_EXT, config.TEST_COMMAND), *config.OTHER_TEST_LIST]:
        word_list = _pytest_word_list(argv) if ext in _RAN_ENV_DICT else None
        if word_list is None:
            continue
        level = _pytest_verbosity(word_list)
        if level < _PYTEST_COUNT_VERBOSITY:
            return (f"the test command `{' '.join(argv)}` sets pytest's verbosity to "
                    f"{level}, at which pytest prints no line counting the tests it ran, "
                    f"so what the tests say cannot be checked: give it one -q at most "
                    f"(TEST_COMMAND)")
    return ""


def _maven_ran_unit_dict(out: str) -> dict[str, int | None] | None:
    """Return how many tests each class of a maven run ran (run less skipped).

    None where the classes' counts do not add up to the modules' (_java_ran_count). A
    class is counted by its first line in each module: a later line of it in the same
    module is surefire running its failing tests again.
    """
    unit_dict: dict[str, int | None] = {}
    start = 0
    for end in [match.end() for match in _MAVEN_RAN_RE.finditer(out)] + [len(out)]:
        module_class_set = set()
        for run, skip, name in _MAVEN_CLASS_RE.findall(out, start, end):
            if name not in module_class_set:
                module_class_set.add(name)
                unit_dict[name] = (unit_dict.get(name) or 0) + int(run) - int(skip)
        start = end
    return unit_dict if sum(unit_dict.values()) == _java_ran_count(out) else None


def _jest_ran_unit_dict(out: str) -> dict[str, int | None] | None:
    """Return the test files a jest or vitest run ran, each with None (not counted apart).

    None where it ran tests and named no file (node --test, bun test), and where vitest
    named a file under a project's label, whose path is not from the repository root.
    """
    vitest_pair_list = _VITEST_FILE_RE.findall(out)
    if any(label for label, _ in vitest_pair_list):
        return None
    file_set = set(_JEST_FILE_RE.findall(out)) | {path for _, path in vitest_pair_list}
    if not file_set and _js_ran_count(out):
        return None
    return dict.fromkeys(file_set)


# The parts a run ran tests of, where its output names them - a test class, a test file
# - one reader per language: (the command's output as _own_text() returns it) ->
# {the part: how many tests of it ran, or None where its tests are not counted}, or None
_RAN_UNIT_DICT = _with_alias_dict({"java": _maven_ran_unit_dict, "js": _jest_ran_unit_dict})


def _ran_unit_dict(out: str, ext: str, directory: str) -> dict[str, int | None] | None:
    """Return the parts a run ran tests of (_RAN_UNIT_DICT).

    Args:
        out: The command's output as _own_text() returns it.
        ext: The language the output is read as.
        directory: Where the command was started, relative to the repository root.

    Returns:
        {part: count or None}, a file (not a Java class) with directory in front of
        it; None for a language with no reader and where the reader says None.
    """
    reader = _RAN_UNIT_DICT.get(ext)
    unit_dict = reader(out) if reader is not None else None
    if unit_dict is None:
        return None
    return {posixpath.join(directory, unit) if directory and ext != "java" else unit: count
            for unit, count in unit_dict.items()}


# The line run_in_project() puts where it left out a part of what a command printed
_CUT_LINE_RE = re.compile(
    "^" + re.escape(CUT_NOTE.strip("\n")).replace(re.escape("{omitted}"), r"\d+") + "$",
    re.M)

# The regexes of the tables above that find a line of a test command's output, beside
# the failing names (_FAIL_RE_DICT), the lines of _EARLY_STOP_RE_DICT and the lines of
# _RUN_LINE_RE_DICT, per language, with the lines a second listing of the failing tests
# starts and ends at (_REPEAT_RE_DICT)
_READ_LINE_RE_DICT = {
    ext: tuple(line_re for line_re in (
        _REPORT_RE_DICT.get(ext), _FAIL_COUNT_RE_DICT.get(ext), _FAIL_EXIT_RE_DICT.get(ext),
        _STOP_SHORT_RE_DICT.get(ext),
        *_NO_TEST_RE_DICT.get(ext, ()), *_RAN_LINE_RE_DICT.get(ext, ()),
        *_FAIL_IF_RE_DICT.get(ext, ()), *_FAIL_UNDER_HEAD_DICT.get(ext, ()),
        *_REPEAT_RE_DICT.get(ext, ()))
        if line_re is not None)
    for ext in _FAIL_RE_DICT}


def _read_line_count(line: str, ext: str) -> int:
    """Return how many lines of a test command's output are read from one line on.

    Args:
        line: One line of the output.
        ext: The language the output is read as.

    Returns:
        1 + _EARLY_STOP_AFTER_LINE for a line _EARLY_STOP_RE_DICT finds, 1 for a line
        that names a failing test (_fail_name), that is at most _MAX_FAIL_LINE_CHAR long
        and names a test file that was run (_RUN_LINE_RE_DICT), or that one of
        _READ_LINE_RE_DICT's regexes finds, 0 for a blank line and for any other.
    """
    if not line or line.isspace():
        return 0
    early_stop_re = _EARLY_STOP_RE_DICT.get(ext)
    if early_stop_re is not None and early_stop_re.search(line):
        return 1 + _EARLY_STOP_AFTER_LINE
    if _fail_name(line, ext) is not None:
        return 1
    run_line_re = _RUN_LINE_RE_DICT.get(ext)
    if run_line_re is not None and len(line) <= _MAX_FAIL_LINE_CHAR \
            and run_line_re.search(line):
        return 1
    for line_re in _READ_LINE_RE_DICT.get(ext, ()):
        if line_re.search(line):
            return 1
    return 0


def _is_no_test_text(out: str, ext: str) -> bool:
    """Whether a test command's output says it found no test of a language to run.

    Args:
        out: The text read as a test command's output.
        ext: The language whose lines are looked for (_NO_TEST_RE_DICT).

    Returns:
        True when the output holds the language's line for a run over no tests and
        none of its line for a test that was run, and when it counts no test that ran
        (_ran_count is 0: every test it found was skipped). False for a language with
        neither.
    """
    no_test_re, test_re = _NO_TEST_RE_DICT.get(ext, (None, None))
    return (no_test_re is not None and bool(no_test_re.search(out))
            and not (test_re is not None and test_re.search(out))) \
        or _ran_count(out, ext) == 0


# == Patch rounds ====================================================================
# What is counted is not how many editing calls were made but how many "edit then
# run_test" rounds happened. Patching 3 files and testing once is one round

# Rounds finished so far. keep_test_state() puts it back as it was before its block
_patch_round_count = 0

# How many rounds this request may use. None reads config.MAX_PATCH_ROUND; 0 enforces
# none
_round_limit: int | None = None

# Whether a patch landed since reset_patch_round() or since a test command was last
# started. keep_test_state() puts it back as it was before its block
_is_patch_since_test = False

# How the editing calls of this request ended

# Calls that landed (under --dry-run, ones that passed the checks)
_patch_ok = 0

# Calls that did not land, including those refused by a check (under --dry-run, ones
# that failed the checks)
_patch_ng = 0

# Calls that failed after an earlier one had landed (under --dry-run, after an earlier
# one had passed the checks)
_patch_ng_after_ok = 0

# How the last run_test() call ended. None means no run reached a verdict: there was
# none since reset_patch_round(), or the tests did not run. keep_test_state() puts it
# back as it was before its block
_last_test_ok = None

# The last run_test() call that ran the whole suite and reached a verdict, as
# {"n": how many such runs there have been, "result": what it returned, "change":
# stage_change_hash_dict() as the run ended}, or None. keep_test_state() puts it back as
# it was before its block
_last_whole_run_dict: dict | None = None

# How much of a file is read at a time when its contents are digested
_HASH_BLOCK_BYTE = 1 << 20

# The exit status of "git ls-files --error-unmatch" for a path the index does not name
_NO_MATCH_STATUS = 1

# What a write through a tool left at each path during this request, as
# {path: digest of the contents it wrote}
_write_hash_dict: dict[str, str] = {}

# What each path that already differed from HEAD held when the request started, in the
# same form. A path that still holds it is not a change of this request
_base_hash_dict: dict[str, str] = {}

# What the editing tools would have written under --dry-run, in the order they were
# asked: a unified diff per write and "deleted <path>" per deletion
_dry_change_list: list[str] = []

# The paths the editing tools were asked to write or delete under --dry-run
_dry_path_set: set[str] = set()

# What a path of _base_hash_dict held when the request started, kept just before the
# first tool write to it, as {path: bytes, or None for a path that was not there}. A
# path reverted is put back from here
_start_byte_dict: dict[str, bytes | None] = {}

# Where a path of _base_hash_dict that was a symbolic link when the request started
# pointed, kept just before a tool deletes it, as {path: the link's target}
_start_link_dict: dict[str, str] = {}

# The directories keep_new_dir() recorded during this request that remove_empty_dir()
# has not deleted, as repository-relative paths
_new_dir_set: set[str] = set()

# What the digest of a symbolic link is made from besides where it points
_LINK_HASH_HEAD = b"link\0"

# What a command the machine ran left at each path it wrote during this request, as
# {path: the digests of the contents such a command left there}
_command_write_hash_dict: dict[str, set[str]] = {}


# What run_test does with its paths under each of the 3 TEST_TARGET_STYLE modes, one
# line each
TARGET_NOTE_DICT = {
    "path": "Called without arguments it runs everything under that directory; you can "
            "also pass a list of test file paths.",
    "separator": "Called without arguments it runs the whole suite; you can also pass a "
                 "list of test file paths, which are forwarded after `--`.",
    "none": "It ignores its paths argument for this build tool and always runs the "
            "whole suite; call it without arguments.",
}


def reset_patch_round(round_limit: int | None = None) -> None:
    """Reset the round and outcome counters to 0.

    The counters are module level. What the working tree already holds against HEAD when
    this is called is read as the request's starting point: own_change_set() answers for
    this request alone.

    Args:
        round_limit: How many "edit then run_test" rounds this request may use before
            the editing tools refuse to write. None follows config.MAX_PATCH_ROUND; 0
            enforces none.
    """
    global _patch_round_count, _is_patch_since_test, _last_test_ok, _last_whole_run_dict
    global _patch_ok, _patch_ng, _patch_ng_after_ok, _round_limit
    _round_limit = round_limit
    _patch_round_count = 0
    _is_patch_since_test = False
    _patch_ok = 0
    _patch_ng = 0
    _patch_ng_after_ok = 0
    _last_test_ok = None
    _last_whole_run_dict = None
    _write_hash_dict.clear()
    _base_hash_dict.clear()
    _start_byte_dict.clear()
    _start_link_dict.clear()
    _new_dir_set.clear()
    _command_write_hash_dict.clear()
    _dry_change_list.clear()
    _dry_path_set.clear()
    # What is already changed is not this request's doing. A directory that is not a
    # repository leaves it empty, which is what it was before this call
    try:
        _base_hash_dict.update({path: _file_hash(path) for path in _git_change_set()})
    except OSError as e:
        logger.info("what was already changed could not be listed (%s)", e)


def patch_round_count() -> int:
    """Return how many "edit then run_test" rounds this request has finished.

    A round closed inside keep_test_state() is not counted once its block ends.
    """
    return _patch_round_count


def patch_count_dict() -> dict:
    """Return how the editing calls of this request ended.

    Returns:
        {"ok": calls that landed, "ng": calls that did not,
         "ng_after_ok": calls that failed after an earlier one had landed}. Under
        --dry-run, "ok" counts the calls that passed the checks, "ng" the ones that
        failed them, and "ng_after_ok" the ones that failed after an earlier one had
        passed them.
    """
    return {"ok": _patch_ok, "ng": _patch_ng, "ng_after_ok": _patch_ng_after_ok}


def write_path_set() -> set[str]:
    """Return the paths a write landed on and still stands on during this request.

    A path revert_file() undid is taken back out, and so is a path a write put back to
    what it held when the request started.

    Returns:
        Repository-relative paths, empty before the first write and under --dry-run.
    """
    return set(_write_hash_dict)


def dry_change_list() -> list[str]:
    """Return what the editing tools would have written during this request's --dry-run.

    Returns:
        A unified diff per write and "deleted <path>" per deletion, in the order they
        were asked. Empty when nothing was asked for, and outside a dry run.
    """
    return list(_dry_change_list)


def dry_path_set() -> set[str]:
    """Return the paths the editing tools were asked to write or delete under --dry-run.

    Returns:
        Repository-relative paths, as the file each lands on (real_rel()). Empty when
        nothing was asked for, and outside a dry run.
    """
    return set(_dry_path_set)


def _link_entry(path: str) -> str:
    """Return where the symbolic link a path names as its last part stands.

    Args:
        path: A path relative to PROJECT_DIR.

    Returns:
        The absolute path of the link, its directories resolved and its last part
        left as it is, or "" when the last part of the path is no symbolic link.

    Raises:
        ValueError: When the directory of the path is one resolve_in_project() refuses.
    """
    parent, name = os.path.split(os.path.normpath(path))
    entry = os.path.join(resolve_in_project(parent or "."), name)
    return entry if os.path.islink(entry) else ""


def _file_hash(path: str) -> str:
    """Return a digest of what one path holds now.

    A path whose last part is a symbolic link is digested as the link it is: by where
    it points, whatever is there.

    Args:
        path: A path relative to PROJECT_DIR.

    Returns:
        The hex digest of the file's bytes, or of _LINK_HASH_HEAD and the link's
        target for a symbolic link. "" for a path that is not there, is not a regular
        file (tmpfile.check_file) or cannot be read or looked at (_PATH_ERROR_TUPLE).
    """
    digest = hashlib.sha256()
    try:
        link = _link_entry(path)
        if link:
            digest.update(_LINK_HASH_HEAD + os.fsencode(os.readlink(link)))
            return digest.hexdigest()
        full_path = resolve_in_project(path)
        tmpfile.check_file(full_path, sys.maxsize)
        with open(full_path, "rb") as fp:
            for block in iter(lambda: fp.read(_HASH_BLOCK_BYTE), b""):
                digest.update(block)
    except (*_PATH_ERROR_TUPLE, ValueError):
        return ""
    return digest.hexdigest()


def _holds_no_file(path: str) -> bool:
    """Say whether something other than a regular file or a symbolic link stands at a path.

    Args:
        path: A path relative to PROJECT_DIR.

    Returns:
        True for a directory, a named pipe, a device and a socket. False for a regular
        file, a symbolic link, a path where nothing stands and a path
        resolve_in_project() refuses or that cannot be looked at (_PATH_ERROR_TUPLE).
    """
    try:
        if _link_entry(path):
            return False
        full_path = resolve_in_project(path)
    except (*_PATH_ERROR_TUPLE, ValueError):
        return False
    return os.path.lexists(full_path) and not os.path.isfile(full_path)


def own_change_set() -> set[str]:
    """Return the paths this request changed, however they were changed.

    Every path git says differs from HEAD, and every path that differed from HEAD when
    the request started, is compared by contents against what it held when the request
    started (reset_patch_round). One that still holds that is not this request's doing.
    A path that differed then and has since gone, or been put back to HEAD, is this
    request's doing. So is a path an editing tool wrote during this request
    (write_path_set) where something other than a regular file or a symbolic link now
    stands (_holds_no_file), which git does not list when it does not track the path.

    Returns:
        Repository-relative paths. Empty when git could not be run, which is logged.
    """
    try:
        change_set = _git_change_set()
    except OSError as e:
        logger.warning("the changed files could not be listed (%s), so what this run "
                       "changed cannot be told", e)
        return set()
    return {path for path in change_set | set(_base_hash_dict)
            if _file_hash(path) != _base_hash_dict.get(path)} \
        | {path for path in _write_hash_dict if _holds_no_file(path)}


def own_change_hash_dict() -> dict[str, str]:
    """Return what each path this request changed holds now.

    Returns:
        {repository-relative path: the digest of its contents}, over the paths
        own_change_set() names. "" for a path that is gone.
    """
    return {path: _file_hash(path) for path in own_change_set()}


def no_tool_path_set() -> set[str]:
    """Return the paths this request changed without going through an editing tool.

    Each path own_change_set() names is compared by contents against what a tool wrote
    there. A write made without calling a tool matches nothing, and so does a second
    write to a path a tool had already written.

    Returns:
        Repository-relative paths, empty when every change went through a tool. Empty
        as well when git could not be run, which is logged.
    """
    return {path for path in own_change_set()
            if _file_hash(path) != _write_hash_dict.get(path)}


def _keep_command_write(before_dict: dict[str, str]) -> None:
    """Record what a command the machine ran left at the paths it wrote.

    A path is recorded when it holds something else than before the command ran, and
    held before it what it held when the request started or what an earlier such
    command left there.

    Args:
        before_dict: own_change_hash_dict() as the command started.
    """
    for path, file_hash in own_change_hash_dict().items():
        if file_hash == before_dict.get(path):
            continue
        if path not in before_dict \
                or before_dict[path] in _command_write_hash_dict.get(path, ()):
            _command_write_hash_dict.setdefault(path, set()).add(file_hash)


@contextlib.contextmanager
def _command_write_record() -> Iterator[None]:
    """Run a command inside it, and record what it wrote however it ends.

    Yields:
        Nothing. What the paths hold when the block is left - at its end, or on any
        exception, an interrupt included - is recorded against what they held when it
        was entered (_keep_command_write).
    """
    before_dict = own_change_hash_dict()
    try:
        yield
    finally:
        _keep_command_write(before_dict)


def command_write_set() -> set[str]:
    """Return the paths that hold what a command run_in_project() started left there.

    Each is a path own_change_set() names without this request's tools or its code
    having written what it holds. A run_in_project() call with is_machine_run False
    records nothing here.

    Returns:
        Repository-relative paths.
    """
    return {path for path, hash_set in _command_write_hash_dict.items()
            if _file_hash(path) in hash_set}


def stage_change_hash_dict() -> dict[str, str]:
    """Return what each path this request's tools or its code changed holds now.

    Returns:
        own_change_hash_dict() without the paths that hold what a command left there
        (command_write_set).
    """
    return {path: file_hash for path, file_hash in own_change_hash_dict().items()
            if file_hash not in _command_write_hash_dict.get(path, ())}


def test_change_set() -> set[str]:
    """Return the test files this run changed, however they were changed.

    Which paths are tests follows paths.under_test_path().

    Returns:
        The test files (under_test_path()) among the ones own_change_set() names. Empty
        when none of them is a test, and when git could not be run.
    """
    return set(under_test_path(own_change_set()))


def _git_change_set() -> set[str]:
    """Return every path git says differs from HEAD, tracked and untracked alike.

    Returns:
        Repository-relative paths.

    Raises:
        OSError: When git could not be run at all.
    """
    by_kind_dict = change_by_kind_dict()
    return {path for path_list in by_kind_dict.values() for path in path_list}


def last_whole_run() -> dict | None:
    """Return the last run_test() call that ran the whole suite to a verdict.

    keep_test_state() puts back what a run inside it set.

    Returns:
        {"n": how many such runs there have been since reset_patch_round(), "result":
        run_test()'s return value, "change": stage_change_hash_dict() as the run ended},
        or None when there has been none. A run over some paths only, or one that did
        not run, is not counted.
    """
    return _last_whole_run_dict


@contextlib.contextmanager
def keep_test_state() -> Iterator[None]:
    """Run tests inside it and put the test state back when it ends.

    What the last run said, whether an edit came after it, the rounds and the last whole
    run are put back when it ends.

    Yields:
        Nothing. The test state is put back however the block ended.
    """
    global _patch_round_count, _is_patch_since_test, _last_test_ok, _last_whole_run_dict
    state_tuple = (_patch_round_count, _is_patch_since_test, _last_test_ok,
                   _last_whole_run_dict)
    try:
        yield
    finally:
        (_patch_round_count, _is_patch_since_test, _last_test_ok,
         _last_whole_run_dict) = state_tuple


def _keep_whole_run(result_dict: dict, is_whole: bool) -> None:
    """Store a run for last_whole_run() when it ran the whole suite to a verdict.

    A run counts when one of its test commands reached a verdict.

    Args:
        result_dict: run_test()'s return value.
        is_whole: Whether no paths were passed.
    """
    global _last_whole_run_dict
    part_list = list(part_dict_of(result_dict).values()) or [result_dict]
    if is_whole and any(part["status"] != "did not run" for part in part_list):
        _last_whole_run_dict = {"n": (_last_whole_run_dict or {"n": 0})["n"] + 1,
                                "result": result_dict,
                                "change": stage_change_hash_dict()}


def last_test_ok() -> bool | None:
    """Return how the last run_test() call ended.

    keep_test_state() puts back what a run inside it set.

    Returns:
        True when the last call's tests passed, False when they were red, None when it
        reached no verdict (status "did not run") or there was no call since
        reset_patch_round().
    """
    return _last_test_ok


def has_test_path() -> bool:
    """Whether TEST_PATH is a directory in the repository.

    Returns:
        True when it is there. An empty setting and "." are the repository root.
    """
    return os.path.isdir(os.path.join(config.PROJECT_DIR, config.TEST_PATH))


def has_test_dir() -> bool:
    """Whether TEST_PATH names a directory of its own that is in the repository.

    Returns:
        True when it is there and is not the repository root (paths.test_root()).
    """
    return bool(test_root()) and has_test_path()


def make_test_path() -> None:
    """Make TEST_PATH's directory when tests are reached by path and it is missing."""
    if config.TEST_TARGET_STYLE == "path" and config.TEST_PATH and not has_test_path():
        os.makedirs(os.path.join(config.PROJECT_DIR, config.TEST_PATH), exist_ok=True)


def write_fill_test() -> list[str]:
    """Write the fill tests of the languages whose test commands the run runs.

    For the primary language and each language of other_test_list() that has a
    LangConfig fill_test_tuple, when no file of the language is under the parent of the
    tests' directory of any of its module directories (_module_test_path_list), the fill
    test is written in each of those tests' directories.

    Returns:
        The files written, relative to PROJECT_DIR.
    """
    file_list = gitcmd.work_tree_file_list(config.PROJECT_DIR) or []
    ext_list = [config.PRIMARY_EXT, *(ext for ext, _, _ in other_test_list(file_list))]
    path_list = []
    for ext in dict.fromkeys(ext_list):
        lang_config = LANG_DICT.get(ext)
        if lang_config is None or not lang_config.fill_test_tuple:
            continue
        test_path_list = _module_test_path_list(ext, file_list)
        if any(_has_file_of(os.path.dirname(test_path), ext) for test_path in test_path_list):
            continue
        name, text = lang_config.fill_test_tuple
        for test_path in test_path_list:
            os.makedirs(os.path.join(config.PROJECT_DIR, test_path), exist_ok=True)
            with open(os.path.join(config.PROJECT_DIR, test_path, name), "w",
                      encoding="utf-8") as handle:
                handle.write(text)
            path_list.append(os.path.join(test_path, name))
    if path_list:
        logger.info("no test of the language is there, so %s is written", list_text(path_list))
    return path_list


def _module_test_path_list(ext: str, file_list: list[str]) -> list[str]:
    """Return the tests' directories of a language's module directories.

    Args:
        ext: The language's LANG_DICT key.
        file_list: The files of PROJECT_DIR, relative to it.

    Returns:
        config.test_dir_of() of each directory holding one of the language's LangConfig
        module_file_tuple files, joined to that directory, sorted.
    """
    module_tuple = LANG_DICT[ext].module_file_tuple
    return [os.path.join(directory, config.test_dir_of(
                ext, os.path.join(config.PROJECT_DIR, directory)))
            for directory in sorted({os.path.dirname(path) for path in file_list
                                     if os.path.basename(path) in module_tuple})]


def _has_file_of(directory: str, ext: str) -> bool:
    """Say whether a directory holds a file of a language at any depth.

    Args:
        directory: The directory, relative to PROJECT_DIR. "" is PROJECT_DIR itself.
        ext: The language's LANG_DICT key.

    Returns:
        True when lang_of() of one of its files is the language's LangConfig.
    """
    lang_config = LANG_DICT[ext]
    return any(lang_of(name) is lang_config for _, _, name_list
               in os.walk(os.path.join(config.PROJECT_DIR, directory)) for name in name_list)


def has_test_command() -> bool:
    """Whether there is a test command to run: RUN_TEST is on and TEST_COMMAND is set.

    Returns:
        True when RUN_TEST is on and TEST_COMMAND is not empty, False otherwise.
    """
    return bool(config.RUN_TEST and config.TEST_COMMAND)


class RunResult(dict):
    """run_test()'s return value: the 5-key dict, with what the run said as attributes.

    It also holds whether every test command of the run found no test of its language to
    run (no_test_run), whether one of them ended before it had run every test
    (early_stop_run) and the lines that say so (early_stop_why), whether one of them was
    started at all (is_start_run), whether one of them stopped short of its tests
    (stop_short_run), and what each command of the run returned on its own
    (part_dict_of). The first three are read off the run's own report in what a command
    printed (_own_text), not off the part of it "output" holds.
    """

    is_no_test: bool = False
    is_early_stop: bool = False
    # Whether a command stopped at a count of failing tests (fail_stop_run)
    is_fail_stop: bool = False
    # The lines saying the run ended early, with the lines after each
    early_stop_text: str = ""
    # The lines saying a command ended early for another reason than a count of
    # failing tests (early_stop_line_list)
    early_stop_line_list: "list[str]" = []
    is_start: bool = False
    # Whether the command ran past its time limit
    is_time_out: bool = False
    # Whether a command is known to have stopped short of its tests: it could not be
    # started, or it reached no verdict, is not read as a run over no tests
    # (_is_no_test_text, _NO_TEST_FILE_DICT) and printed its language's line for a run
    # ended before the tests (_STOP_SHORT_RE_DICT)
    is_stop_short: bool = False
    # Whether it stopped short of its tests because what they need could not be fetched:
    # it printed its language's line for that (_FETCH_FAIL_RE_DICT), or the dependencies
    # the tests load could not be installed
    is_fetch_fail: bool = False
    # {a command as the settings hold it, without a target or a path: its own result}
    part_dict: "dict[str, RunResult]" = {}
    # The primary language's command among them, "" when it did not take part
    primary_name: str = ""
    # Whether the run was given no paths
    is_whole: bool = False
    # How many tests one command's run ran and their names (_ran_count, _ran_name_list),
    # None where the output does not say, for a run given paths and for one that reached
    # no verdict or ended with a non-zero exit code before it had run every test
    ran_count: "int | None" = None
    ran_name_list: "list[str] | None" = None
    # The parts it ran tests of, as _ran_unit_dict() reads them, under the same conditions
    ran_unit_dict: "dict[str, int | None] | None" = None
    # Whether its output does not name each test it counts: it counts failing tests and
    # names none of them, or names some of the tests that passed but not every one
    # (_NAME_SHORT_DICT), for a run that reached a verdict, one that ended before it had
    # run every test included; for a run given paths, where it is red and names no
    # failing test alone
    is_name_short: bool = False
    # Whether its output says nothing of how many tests it ran while it is a run of the
    # language's that says so (_UNREAD_DICT), whether it reached a verdict or not
    is_ran_unread: bool = False
    # Whether ran_count, ran_name_list and ran_unit_dict were read off it: a run given no
    # paths that reached a verdict and did not end with a non-zero exit code before it
    # had run every test
    is_ran_hold: bool = False
    # The paths, relative to the repository root, of the files and directories one
    # command's red run named as ones it could not load and ran no test under
    # (_LOAD_FAIL_PATH_DICT)
    load_fail_path_list: "list[str]" = []


# What a baseline holds, beside the names of the failing tests, for a run of several
# test commands: one entry per command, saying whether its tests are left out of the
# comparison, were all passing, had failing ones, or found no test to run. An entry is
# _MARK_SEPARATOR, the word, _MARK_SEPARATOR and the command, which no test name is
_MARK_SEPARATOR = "\x1f"
_NOT_COMPARED_WORD = "not compared"
_GREEN_BEFORE_WORD = "green before"
_RED_BEFORE_WORD = "red before"
_NO_TEST_BEFORE_WORD = "no test before"
_MARK_WORD_TUPLE = (_NOT_COMPARED_WORD, _GREEN_BEFORE_WORD, _RED_BEFORE_WORD,
                    _NO_TEST_BEFORE_WORD)

# What a baseline holds of the tests a command's run ran (ran_entry_set): an entry of
# _MARK_SEPARATOR, the word, _MARK_SEPARATOR, the command ("" in the baseline of one
# command), _MARK_SEPARATOR and the count, followed by a line for each test's name
# under _RAN_NAME_WORD, a line "<part>\t<count or nothing>" for each part under
# _RAN_PART_WORD, and by nothing under _RAN_COUNT_WORD and _RAN_SHORT_WORD (a run that
# does not name each test it counts, or says nothing of how many it ran).
# _RAN_STOP_WORD's entry (stop_entry_set), with the count 0, stands for a red run that
# ended before it had run every test
_RAN_NAME_WORD = "ran"
_RAN_PART_WORD = "ran by part"
_RAN_COUNT_WORD = "ran count"
_RAN_SHORT_WORD = "ran short of names"
_RAN_STOP_WORD = "ran up to a stop"
_RAN_WORD_TUPLE = (_RAN_NAME_WORD, _RAN_PART_WORD, _RAN_COUNT_WORD, _RAN_SHORT_WORD,
                   _RAN_STOP_WORD)

# The most characters the names or the parts of one entry hold; an entry that would hold
# more holds the count alone
_MAX_RAN_NAME_CHAR = 1 << 20

# How a part whose tests run fewer than before is named, and one whose tests are not
# counted apart and run no more
_PART_FEWER_TEXT = "{lost} of the {before} tests of {part}"
_PART_ABSENT_TEXT = "the tests of {part}"

# The languages whose runs name a doctest by an id of its own (_is_doctest_name); such a
# doctest of a file that differs from HEAD is not held as a test that no longer runs
_DOCTEST_ONLY_EXT_TUPLE = ("py",)

# The languages whose runs name a test's file relative to its package; such a file is
# matched by the end of a repository-relative path
_PACKAGE_PATH_EXT_TUPLE = ("rs",)

# The baseline of one test command whose run found no test to run (no_test_run): one
# entry with no command, read as the primary command's
NO_TEST_BASELINE = frozenset({f"{_MARK_SEPARATOR}{_NO_TEST_BEFORE_WORD}{_MARK_SEPARATOR}"})

# What stands for the result of a command that was not part of a run
_NO_PART_TEXT = ("the command was not part of this run: its dependency file or its "
                 "tests' directory is gone, or its command is not here")
_NO_PART_DICT = {"status": "did not run", "failed": [], "failed_excerpt": _NO_PART_TEXT,
                 "targets": [], "output": _NO_PART_TEXT}


def part_dict_of(result_dict: dict) -> dict:
    """Return what each test command of a run returned on its own.

    Args:
        result_dict: run_test()'s return value.

    Returns:
        {the command as the settings hold it: its own 5-key result}, in the order the
        commands ran; {} for a dict that is not a RunResult.
    """
    return dict(getattr(result_dict, "part_dict", None) or {})


def part_list_of(result_dict: dict) -> list[dict]:
    """Return what each test command of a run returned on its own, as a list.

    Args:
        result_dict: run_test()'s return value.

    Returns:
        part_dict_of()'s results in the order the commands ran, or [result_dict] when
        what part_dict_of() returns is empty.
    """
    return list((part_dict_of(result_dict) or {"": result_dict}).values())


def part_of(result_dict: dict, name: str) -> dict:
    """Return what one test command returned in a run.

    Args:
        result_dict: run_test()'s return value.
        name: The command, as part_dict_of() names it.

    Returns:
        The command's own result; a result saying it did not run when it was not part
        of the run; result_dict itself for a dict that is not a RunResult.
    """
    part_dict = part_dict_of(result_dict)
    if not part_dict:
        return result_dict
    return part_dict.get(name, RunResult(_NO_PART_DICT))


def primary_part(result_dict: dict) -> dict:
    """Return what the primary language's test command returned in a run.

    Args:
        result_dict: run_test()'s return value.

    Returns:
        The primary command's own result; result_dict itself for a dict that is not
        a RunResult and for a run the primary command was not part of.
    """
    return part_dict_of(result_dict).get(getattr(result_dict, "primary_name", ""),
                                         result_dict)


def baseline_mark(word: str, command: str) -> str:
    """Return the entry a baseline holds for one test command.

    Args:
        word: One of _NOT_COMPARED_WORD, _GREEN_BEFORE_WORD, _RED_BEFORE_WORD and
            _NO_TEST_BEFORE_WORD.
        command: The command, as part_dict_of() names it.

    Returns:
        _MARK_SEPARATOR, the word, _MARK_SEPARATOR and the command, joined.
    """
    return f"{_MARK_SEPARATOR}{word}{_MARK_SEPARATOR}{command}"


def _mark_dict(baseline_set: frozenset[str]) -> dict[str, str]:
    """Read the entries join_baseline() added to a baseline.

    Args:
        baseline_set: A baseline.

    Returns:
        {a command: the word its entry holds}; {"": the word} for NO_TEST_BASELINE and
        {} for any other baseline of one command.
    """
    mark_dict = {}
    for name in sorted(baseline_set):
        part_list = name.split(_MARK_SEPARATOR, 2)
        if len(part_list) == 3 and not part_list[0] and part_list[1] in _MARK_WORD_TUPLE:
            mark_dict[part_list[2]] = part_list[1]
    return mark_dict


def _has_fail_entry(baseline_set: frozenset[str] | set[str]) -> bool:
    """Say whether a baseline holds a failing test (an entry that is no mark)."""
    return any(not name.startswith(_MARK_SEPARATOR) for name in baseline_set)


def ran_entry_set(result_dict: dict) -> frozenset[str]:
    """Return the entry a baseline holds of the tests one command's run ran.

    Args:
        result_dict: One command's own result (RunResult).

    Returns:
        The one entry, with no command: of _RAN_SHORT_WORD with the count 0 where the
        run says nothing of how many tests it ran while its language's runs say so
        (RunResult.is_ran_unread), of _RAN_SHORT_WORD where the run said how many tests
        it ran and does not name each test it counts (RunResult.is_name_short), else of
        _RAN_NAME_WORD where the run named the tests it ran, else of _RAN_PART_WORD
        where it named the parts it ran tests of, else of _RAN_COUNT_WORD where it said
        how many it ran; names and parts that do not fit in _MAX_RAN_NAME_CHAR
        characters are passed over. Empty where it said neither.
    """
    is_unread = getattr(result_dict, "is_ran_unread", False)
    count = 0 if is_unread else getattr(result_dict, "ran_count", None)
    if count is None:
        return frozenset()
    if is_unread or getattr(result_dict, "is_name_short", False):
        return frozenset({f"{_MARK_SEPARATOR}{_RAN_SHORT_WORD}{_MARK_SEPARATOR}"
                          f"{_MARK_SEPARATOR}{count}"})
    name_list = getattr(result_dict, "ran_name_list", None)
    unit_dict = getattr(result_dict, "ran_unit_dict", None)
    word, rest = _RAN_COUNT_WORD, ""
    for each_word, line_list in (
            (_RAN_NAME_WORD, name_list),
            (_RAN_PART_WORD, None if unit_dict is None else [
                f"{unit}\t{'' if unit_count is None else unit_count}"
                for unit, unit_count in sorted(unit_dict.items())])):
        each_rest = "".join(f"\n{line}" for line in line_list or ())
        if line_list is not None and len(each_rest) <= _MAX_RAN_NAME_CHAR:
            word, rest = each_word, each_rest
            break
    return frozenset({f"{_MARK_SEPARATOR}{word}{_MARK_SEPARATOR}{_MARK_SEPARATOR}{count}"
                      + rest})


def _ran_part_list(name: str) -> list[str] | None:
    """Return the parts of a baseline's entry of ran_entry_set().

    The parts are "", the word, the command and the rest; None for any other entry.
    """
    part_list = name.split(_MARK_SEPARATOR, 3)
    if len(part_list) == 4 and not part_list[0] and part_list[1] in _RAN_WORD_TUPLE:
        return part_list
    return None


def _ran_dict(baseline_set: frozenset[str]
              ) -> dict[str, tuple[int, set[str] | None, dict[str, int | None] | None, bool]]:
    """Read the entries ran_entry_set() put in a baseline.

    Returns:
        {a command, "" in the baseline of one command: (how many tests its run ran,
        their names or None, {part: count or None} or None, whether its run does not
        name each test it counts)}.
    """
    ran_dict = {}
    for name in baseline_set:
        part_list = _ran_part_list(name)
        if part_list is None or part_list[1] == _RAN_STOP_WORD:
            continue
        count_text, *line_list = part_list[3].split("\n")
        unit_dict = None
        if part_list[1] == _RAN_PART_WORD:
            unit_dict = {unit: int(unit_count) if unit_count else None
                         for unit, unit_count in (line.rsplit("\t", 1) for line in line_list)}
        ran_dict[part_list[2]] = (int(count_text), set(line_list)
                                  if part_list[1] == _RAN_NAME_WORD else None, unit_dict,
                                  part_list[1] == _RAN_SHORT_WORD)
    return ran_dict


def _ran_entry_of(name_set: frozenset[str] | set[str], command: str) -> set[str]:
    """Return one command's entries of ran_entry_set() and stop_entry_set(), named.

    The entries are written as a baseline of several commands holds them, which name the
    command.
    """
    return {_MARK_SEPARATOR.join(["", part_list[1], command, part_list[3]])
            for name in name_set if (part_list := _ran_part_list(name)) is not None}


def stop_entry_set() -> frozenset[str]:
    """Return the entry a baseline of one command holds for a run that stopped early.

    The run is the one the baseline was read from: it was red and ended before it had
    run every test.
    """
    return frozenset({_MARK_SEPARATOR.join(["", _RAN_STOP_WORD, "", "0"])})


def is_stop_before(baseline_set: frozenset[str], command: str) -> bool:
    """Say whether a baseline holds stop_entry_set()'s entry for a test command.

    Args:
        baseline_set: A baseline.
        command: The command, as compare_part_list() names it; the entry of the
            baseline of one command is the command's whatever its name.

    Returns:
        True when the baseline holds the entry for the command, or the entry of a
        baseline of one command.
    """
    command_set = {part_list[2] for name in baseline_set
                   if (part_list := _ran_part_list(name)) is not None
                   and part_list[1] == _RAN_STOP_WORD}
    return command in command_set or "" in command_set


def join_baseline(by_command_dict: dict, primary_name: str = "") -> frozenset[str] | None:
    """Put the baselines of several test commands together as one set.

    Args:
        by_command_dict: {a command, as part_dict_of() names it: the names of its
            tests that are not counted, with the entry of the tests its run ran
            (ran_entry_set), NO_TEST_BASELINE when its run found no test to run, or
            None when its tests are left out of the comparison}.
        primary_name: The primary language's command, "" when it is not one of them.

    Returns:
        None when every command is left out. The primary command's value alone where it
        is the one command. Otherwise every name and the entry of the tests each
        command's run ran, naming the command, with one entry per command saying which
        of the four it is (baseline_mark).
    """
    if all(name_set is None for name_set in by_command_dict.values()):
        return None
    if list(by_command_dict) == [primary_name]:
        return frozenset(by_command_dict[primary_name])
    join_set = set()
    for command, name_set in by_command_dict.items():
        if name_set is None or name_set == NO_TEST_BASELINE:
            word = _NOT_COMPARED_WORD if name_set is None else _NO_TEST_BEFORE_WORD
            join_set.add(baseline_mark(word, command))
            continue
        word = _RED_BEFORE_WORD if _has_fail_entry(name_set) else _GREEN_BEFORE_WORD
        join_set |= {name for name in name_set if not name.startswith(_MARK_SEPARATOR)} \
            | _ran_entry_of(name_set, command) | {baseline_mark(word, command)}
    return frozenset(join_set)


def not_compared_list(baseline_set: frozenset[str] | None) -> list[str]:
    """Return the test commands a baseline leaves out of the comparison.

    Args:
        baseline_set: A baseline, or None.

    Returns:
        The commands, sorted; [] for None and for the baseline of one command.
    """
    return [command for command, word in _mark_dict(baseline_set or frozenset()).items()
            if word == _NOT_COMPARED_WORD]


def baseline_name_list(baseline_set: frozenset[str] | None) -> list[str]:
    """Return the test names a baseline holds, without what join_baseline() added.

    Args:
        baseline_set: A baseline, or None.

    Returns:
        The names, sorted, each once; [] for None.
    """
    return sorted(name for name in baseline_set or () if _MARK_SEPARATOR not in name)


def unrun_path_list(path_list: list[str], test_dict: dict) -> list[str]:
    """Return the files whose tests a run did not run.

    Such a file is one a command's run named as a file it could not load, or one under a
    directory it named so (RunResult.load_fail_path_list), where no command's run named
    a test of the file or the file as a part it ran tests of (ran_name_list,
    ran_unit_dict).

    Examples:
        (["tests/sub/test_s.py", "tests/test_p.py"], a run whose "ERROR tests/sub - ..."
        line names tests/sub) -> ["tests/sub/test_s.py"]

    Args:
        path_list: Files, relative to the repository root.
        test_dict: run_test()'s return value.

    Returns:
        Those of path_list, in its order.
    """
    part_list = part_list_of(test_dict)
    load_fail_list = [path for part in part_list
                      for path in getattr(part, "load_fail_path_list", ())]
    ran_name_set = {name for part in part_list
                    for name in getattr(part, "ran_name_list", None) or ()}
    ran_unit_set = {unit for part in part_list
                    for unit in getattr(part, "ran_unit_dict", None) or ()}
    return [path for path in path_list
            if any(path == name or path.startswith(name.rstrip("/") + "/")
                   for name in load_fail_list)
            and path not in ran_unit_set
            and not any(name.startswith(path + "::") for name in ran_name_set)]


def fail_entry_set(name_list: list[str]) -> frozenset[str]:
    """Return the failing names of a run as a set that keeps how many times each failed.

    A name is one entry the first time it fails and one more for each later time:
    taking one set from another leaves the times a name fails over the times it failed
    in the other, and two sets meet at the fewer times.

    Examples:
        ["t::a", "t::b", "t::a"] -> {"t::a", "t::b", "t::a\\x1f2"}

    Args:
        name_list: The failing names of a run, a name as many times as it failed.

    Returns:
        The name itself for its first time, and the name, _MARK_SEPARATOR and the count
        for each later time.
    """
    count_dict = collections.Counter()
    entry_set = set()
    for name in name_list:
        count_dict[name] += 1
        entry_set.add(name if count_dict[name] == 1
                      else f"{name}{_MARK_SEPARATOR}{count_dict[name]}")
    return frozenset(entry_set)


def fail_name_list(entry_set: frozenset[str] | set[str]) -> list[str]:
    """Return the names a set of fail_entry_set() entries is of; the reverse of it.

    Examples:
        {"t::a", "t::b", "t::a\\x1f2"} -> ["t::a", "t::a", "t::b"]

    Args:
        entry_set: Entries of fail_entry_set(), or of a baseline.

    Returns:
        Each name as many times as it has entries, sorted. The entries a baseline holds
        for its commands (baseline_mark) are left out.
    """
    return sorted(entry.split(_MARK_SEPARATOR, 1)[0] for entry in entry_set
                  if not entry.startswith(_MARK_SEPARATOR))


def compare_part_list(result_dict: dict, baseline_set: frozenset[str],
                      hold_primary_no_test: bool = False) -> list[tuple[str, dict, bool]]:
    """Return the test commands a run is held against a baseline by.

    It pairs the run before, as the baseline holds it, with the run now. A command whose
    run before found no test to run and whose run now finds none either cannot be
    compared and is not held. A command whose run before found none is held as one whose
    tests all passed otherwise, and one whose tests ran before is held whatever its run
    now says, a run that finds no test included.

    Args:
        result_dict: run_test()'s return value.
        baseline_set: The baseline, as join_baseline() made it or as one command's
            names.
        hold_primary_no_test: Whether the primary command is held when neither run
            found a test to run.

    Returns:
        [(the command's line for a reason, its own result, whether the baseline holds
        failing tests of it), ...].

        A baseline with entries (join_baseline) names its commands: every one it
        does not leave out is returned. A baseline without entries, and
        NO_TEST_BASELINE, are the primary command's. A command the baseline does not
        know is not held against it.

        A command of the baseline that is not part of the run is returned with a
        result saying it did not run when the run was given no paths, and is passed
        over when it was given some. Where the run holds no command at all (a dict
        that is not a RunResult), result_dict itself is each command's result.
    """
    part_dict = part_dict_of(result_dict)
    primary_name = getattr(result_dict, "primary_name", "") \
        or " ".join(config.TEST_COMMAND) or test_command_text()
    one_name = primary_name if part_dict else test_command_text()
    mark_dict = _mark_dict(baseline_set)
    if not mark_dict:
        mark_dict = {one_name: _RED_BEFORE_WORD if _has_fail_entry(baseline_set)
                     else _GREEN_BEFORE_WORD}
    elif "" in mark_dict:
        mark_dict = {one_name: mark_dict[""]}
    primary_dict = primary_part(result_dict)
    entry_list = []
    for command in [*part_dict, *(mark_command for mark_command in mark_dict
                                  if mark_command not in part_dict)]:
        word = mark_dict.get(command, _NOT_COMPARED_WORD)
        if word == _NOT_COMPARED_WORD:
            continue
        part = part_dict.get(command)
        if part is None and part_dict:
            if not getattr(result_dict, "is_whole", False):
                continue
            part = RunResult(_NO_PART_DICT)
        part = result_dict if part is None else part
        if word == _NO_TEST_BEFORE_WORD and no_test_run(part) \
                and not (hold_primary_no_test and part is primary_dict):
            continue
        entry_list.append((command, part, word == _RED_BEFORE_WORD))
    return entry_list


def unrun_reason(result_dict: dict, baseline_set: frozenset[str], command: str,
                 name_reason: str, count_reason: str, unread_reason: str,
                 short_reason: str) -> str:
    """Return what one command's run says of the tests that ran before it and run no more.

    The run is held to the baseline's entry of the tests the command's run before ran
    (ran_entry_set). A run they were read off (RunResult.is_ran_hold) is a reason where
    it or the run before does not name each test it counts (RunResult.is_name_short),
    where the run before said nothing of how many tests it ran
    (RunResult.is_ran_unread), and where it says less of them than the run before - no
    count, no names where it named the tests, no parts where it named the parts. By
    names, where both runs named them: a test that ran before and does not now
    (_miss_name_list). Else by parts, where both runs named the parts they ran tests of:
    a part whose tests run fewer than before, or no more where they are not counted
    (_miss_part_list); a part whose tests are not counted leaves the counts compared as
    well. Else by count: fewer tests than before, unless a test file of the command's
    language that HEAD holds was modified or is gone (_command_test_file_change).

    Args:
        result_dict: The command's own result now.
        baseline_set: The baseline.
        command: The command, as compare_part_list() names it.
        name_reason: The reason for tests named, with {command} and {tests}.
        count_reason: The reason for a count, with {command}, {before} and {now}.
        unread_reason: The reason for a run that says less, with {command}.
        short_reason: The reason for runs that do not name each test they count, with
            {command}.

    Returns:
        One reason, or "" where nothing ran before that does not now, where the run
        before said neither how many tests it ran nor which, and where they were not
        read off the run now.
    """
    shown = command_text(command)
    ran_dict = _ran_dict(baseline_set)
    before_tuple = ran_dict.get(command, ran_dict.get(""))
    if before_tuple is None:
        return ""
    before_count, before_name_set, before_unit_dict, is_before_short = before_tuple
    now_count = getattr(result_dict, "ran_count", None)
    now_name_list = getattr(result_dict, "ran_name_list", None)
    now_unit_dict = getattr(result_dict, "ran_unit_dict", None)
    is_hold = getattr(result_dict, "is_ran_hold", False)
    if is_hold and (is_before_short or getattr(result_dict, "is_name_short", False)):
        return short_reason.format(command=shown)
    if is_hold and (
            now_count is None or (before_name_set is not None and now_name_list is None)
            or (before_unit_dict is not None and now_unit_dict is None)):
        return unread_reason.format(command=shown)
    if now_count is None:
        return ""
    ext = _command_ext(command)
    if before_name_set is not None and now_name_list is not None:
        miss_list = _miss_name_list(before_name_set, now_name_list, ext)
        return name_reason.format(command=shown, tests=list_text(miss_list)) \
            if miss_list else ""
    if before_unit_dict is not None and now_unit_dict is not None:
        miss_list = _miss_part_list(before_unit_dict, now_unit_dict, ext)
        if miss_list:
            return name_reason.format(command=shown, tests=list_text(miss_list))
        if None not in before_unit_dict.values():
            return ""
    if now_count < before_count and not _command_test_file_change(command):
        return count_reason.format(command=shown, before=before_count, now=now_count)
    return ""


def _command_ext(command: str) -> str:
    """Return the language whose test command a command is, as part_dict_of() names it.

    That is the language of config.OTHER_TEST_LIST that holds it, else the primary
    language.
    """
    for ext, argv in config.OTHER_TEST_LIST:
        if " ".join(argv) == command:
            return ext
    return config.PRIMARY_EXT


def _change_path_set_tuple() -> tuple[set[str], set[str]]:
    """Return the paths git says are deleted and every path that differs from HEAD.

    Returns:
        (the deleted paths, a moved path included; every path that differs from HEAD).
        Both empty when git could not be run, which is logged.
    """
    try:
        by_kind_dict = change_by_kind_dict()
    except OSError as e:
        logger.info("which files differ from HEAD could not be read (%s), so every test "
                    "that no longer runs is counted", e)
        return set(), set()
    return set(by_kind_dict["deleted"]), {path for path_list in by_kind_dict.values()
                                          for path in path_list}


def _part_path_match(part: str, path_set: set[str], ext: str) -> bool:
    """Say whether one of some paths is the file a part of a run is of.

    Examples:
        ("tests/test_a.py", {"tests/test_a.py"}, "py")                  -> True
        ("tests/test_a.py", {"pkg/tests/test_a.py"}, "py")              -> False
        ("src/lib.rs", {"crates/a/src/lib.rs"}, "rs")                   -> True
        ("com.e.FooTest$Inner", {"a/src/test/java/com/e/FooTest.java"}, "java") -> True

    Args:
        part: A test file, as a run names it: relative to the repository root, a
            cargo source relative to its package (_PACKAGE_PATH_EXT_TUPLE); or for
            Java a test class.
        path_set: Repository-relative paths.
        ext: The language of the command that ran it.

    Returns:
        True for a path that is the part; for a language of _PACKAGE_PATH_EXT_TUPLE,
        one that is the part or ends in "/<part>"; for Java, one whose path without
        its extension is, or ends in, the class's package and outer class as a path.
    """
    if ext == "java":
        tail = part.split("$", 1)[0].replace(".", "/")
        return any((stem := os.path.splitext(path)[0]) == tail or stem.endswith(f"/{tail}")
                   for path in path_set)
    if ext in _PACKAGE_PATH_EXT_TUPLE:
        return any(path == part or path.endswith(f"/{part}") for path in path_set)
    return part in path_set


def _name_part_tuple(name: str, ext: str) -> tuple[str, str, str]:
    """Return the parts of a test's name a run before and a run now are compared by.

    Examples:
        ("tests/t.py::T::test_q[pg-1]", "py")
            -> ("tests/t.py", "tests/t.py", "tests/t.py::T::test_q")
        ("src/lib.rs::tests::adds [calc]", "rs")
            -> ("src/lib.rs", "src/lib.rs [calc]", "src/lib.rs::tests::adds [calc]")

    Args:
        name: A test's name, as _ran_name_list() returns it.
        ext: The language of the command that ran it.

    Returns:
        (its file; its file with the cargo target its test is of, whose tests are
        counted together; the test without the parameters of a pytest id).
    """
    file_path, _, rest = name.partition("::")
    if ext == "py":
        return file_path, file_path, f"{file_path}::{rest.split('[', 1)[0]}"
    head, mark, target = name.rpartition(" [")
    return file_path, f"{file_path}{mark}{target}" if head else file_path, name


def _miss_name_list(name_set: set[str], now_name_list: list[str], ext: str) -> list[str]:
    """Return the tests that ran before and do not run now.

    Args:
        name_set: The names of the tests that ran before, each "<file>::<name>".
        now_name_list: The names of the tests that ran now.
        ext: The language of the command that ran them (_part_path_match).

    Returns:
        The names that do not run now, sorted, of which no test runs now without the
        parameters of its name and whose file (with its target) runs fewer tests than
        before (_name_part_tuple), and whose file git does not say is deleted (a moved
        file included) and does not differ from HEAD while it still runs a test now.
        For a language in _DOCTEST_ONLY_EXT_TUPLE, a doctest (_is_doctest_name) of a
        file that differs from HEAD is left out as well.
    """
    now_name_set = set(now_name_list)
    before_count_dict = collections.Counter(
        _name_part_tuple(name, ext)[1] for name in name_set)
    now_count_dict = collections.Counter(
        _name_part_tuple(name, ext)[1] for name in now_name_set)
    now_test_set = {_name_part_tuple(name, ext)[2] for name in now_name_set}
    absent_list = [name for name in sorted(name_set - now_name_set)
                   if (part_tuple := _name_part_tuple(name, ext))[2] not in now_test_set
                   and now_count_dict[part_tuple[1]] < before_count_dict[part_tuple[1]]]
    if not absent_list:
        return []
    delete_set, change_set = _change_path_set_tuple()
    miss_list = []
    for name in absent_list:
        file_path, group, _ = _name_part_tuple(name, ext)
        if _part_path_match(file_path, delete_set, ext):
            continue
        if _part_path_match(file_path, change_set, ext) and (
                now_count_dict[group] > 0 or (
                    ext in _DOCTEST_ONLY_EXT_TUPLE and _is_doctest_name(name))):
            continue
        miss_list.append(name)
    return miss_list


def _is_doctest_name(name: str) -> bool:
    """Say whether a pytest test id is that of a doctest, not of a test function.

    A doctest's id is "<file>::<the name of a text file>" (--doctest-glob) or
    "<file>::<module>[.<name>]" (--doctest-modules), the module's dotted name ending in
    the file's stem (in its directory's name for __init__.py). An id with no dot that
    starts with "test" may be a test function's and is not taken for a doctest.

    Examples:
        "b.py::b.fmt" -> True, "pkg/__init__.py::pkg" -> True,
        "docs/a.txt::a.txt" -> True, "test_main.py::test_m1" -> False,
        "test_main.py::test_main" -> False, "t.py::T::test_a" -> False

    Args:
        name: A test's name, as _ran_name_list() returns it.

    Returns:
        True when the id is a doctest's.
    """
    file_path, _, rest = name.partition("::")
    if not rest or "::" in rest or "[" in rest:
        return False
    base = posixpath.basename(file_path)
    if rest == base:
        return True
    stem = posixpath.splitext(base)[0]
    module = posixpath.basename(posixpath.dirname(file_path)) if stem == "__init__" else stem
    if "." in rest:
        return module in rest.split(".")
    return rest == module and not rest.startswith("test")


def _miss_part_list(before_dict: dict[str, int | None], now_dict: dict[str, int | None],
                    ext: str) -> list[str]:
    """Return the texts naming the parts whose tests run fewer than before.

    Args:
        before_dict: {part: how many of its tests ran, or None} of the run before.
        now_dict: The same of the run now.
        ext: The language of the command that ran them (_part_path_match).

    Returns:
        _PART_FEWER_TEXT for a counted part whose tests run fewer now, _PART_ABSENT_TEXT
        for an uncounted one that runs no test now, sorted by part. A part whose file
        git says is deleted is left out, and so is one whose file differs from HEAD and
        that still runs a test.
    """
    delete_set, change_set = _change_path_set_tuple()
    miss_list = []
    for part, before_count in sorted(before_dict.items()):
        now_count = now_dict.get(part)
        is_run = part in now_dict and (now_count is None or now_count > 0)
        if _part_path_match(part, delete_set, ext) \
                or (is_run and _part_path_match(part, change_set, ext)):
            continue
        if before_count is None:
            if part not in now_dict:
                miss_list.append(_PART_ABSENT_TEXT.format(part=part))
        elif (now_count or 0) < before_count:
            miss_list.append(_PART_FEWER_TEXT.format(
                lost=before_count - (now_count or 0), before=before_count, part=part))
    return miss_list


def _is_language_test_path(path: str, ext: str, directory: str) -> bool:
    """Say whether a path, relative to where a language's tests run from, is one of them.

    Such a path is under the language's test directory there (config.test_dir_of) at
    any depth, or has a part its test_name_tuple matches.
    """
    test_dir = config.test_dir_of(ext, os.path.join(config.PROJECT_DIR, directory))
    return is_test_path(path, test_dir, LANG_DICT[ext].test_name_tuple)


def _command_test_file_change(command: str) -> bool:
    """Say whether HEAD's test files of a command's language were modified or deleted.

    Args:
        command: The command, as part_dict_of() names it.

    Returns:
        True when a path git says is modified or deleted (a moved path included) is a
        test of the command's language (_test_file_of). False when git could not be run.
    """
    try:
        by_kind_dict = change_by_kind_dict()
    except OSError:
        return False
    other_list = other_test_list()
    return any(_test_file_of(path, other_list) == (command, True)
               for path in sorted(set(by_kind_dict["modified"]) | set(by_kind_dict["deleted"])))


def _test_file_of(path: str, other_list: list[tuple[str, list[str], str]]
                  ) -> tuple[str, bool] | None:
    """Return the test command that runs a path's language, and whether it is a test.

    Args:
        path: A repository-relative path.
        other_list: other_test_list()'s value.

    Returns:
        (the command, as part_dict_of() names it; True for a test of the primary
        language (runnable_path_list and under_test_path) or of another language whose
        command it is (_path_by_language and _is_language_test_path)), or None for a
        path of no language a test command runs.
    """
    _, by_other_dict = _path_by_language([(path, path)], other_list)
    if by_other_dict:
        ext, _, directory = other_list[next(iter(by_other_dict))]
        return (" ".join(dict(config.OTHER_TEST_LIST)[ext]),
                _is_language_test_path(posixpath.relpath(path, directory or "."), ext,
                                       directory))
    if runnable_path_list([path]):
        return " ".join(config.TEST_COMMAND), bool(under_test_path([path]))
    return None


def no_test_run(result_dict: dict) -> bool:
    """Whether one run_test() return value says the run found no test to run.

    Args:
        result_dict: run_test()'s return value.

    Returns:
        What a RunResult holds: True when no test command reached a verdict and
        _is_no_test_text() reads the run's own report in what each printed (_own_text)
        as a run over no tests of its language, or the command printed nothing and the
        repository's files say it had no test to run (_NO_TEST_FILE_DICT). For any other
        dict, True when the status is "did not run" and _is_no_test_text() reads its
        "output" as such a run of the primary language. False for a language whose
        output _is_no_test_text() never reads as such a run, and for a run that did not
        run for another reason.
    """
    if isinstance(result_dict, RunResult):
        return result_dict.is_no_test
    return _no_test_in(result_dict, config.PRIMARY_EXT)


def load_fail_run(result_dict: dict) -> str:
    """Return the file one run_test() return value says the primary command could not load.

    The file is one the tests under its directory need; the command ran none of the
    tests it was given.

    Examples:
        "did not run", "ImportError while loading conftest '<PROJECT_DIR>/tests/b/
        conftest.py'." -> "tests/b/conftest.py"

    Args:
        result_dict: run_test()'s return value, or a dict of its keys.

    Returns:
        The file's path relative to the repository root, read off the primary language's
        line for it (_LOAD_FAIL_RE_DICT) in "output" where the status is "did not run";
        "" for any other run and a language with no such line.
    """
    load_fail_re = _LOAD_FAIL_RE_DICT.get(config.PRIMARY_EXT)
    if result_dict["status"] != TEST_STATUS_DICT[None] or load_fail_re is None:
        return ""
    match = load_fail_re.search(result_dict.get("output", ""))
    if match is None:
        return ""
    path = os.path.realpath(os.path.join(config.PROJECT_DIR, match.group(1)))
    return os.path.relpath(path, os.path.realpath(config.PROJECT_DIR)).replace(os.sep, "/")


def early_stop_run(result_dict: dict) -> bool:
    """Whether one run_test() return value says a test command ended before every test.

    The command ended before it had run every test it was given. The failing names of
    such a run are the tests it reached: a test it does not name may not have run at
    all.

    Args:
        result_dict: run_test()'s return value, or one command's own result.

    Returns:
        What a RunResult holds: True when the run's own report in what one of its
        commands printed (_own_text) holds that language's line for a run ended early
        (_EARLY_STOP_RE_DICT). For any other dict, True when its "output" holds the
        primary language's line. False for a language with no such line.
    """
    if isinstance(result_dict, RunResult):
        return result_dict.is_early_stop
    return _early_stop_in(result_dict, config.PRIMARY_EXT)


def fail_stop_run(result_dict: dict) -> bool:
    """Whether one run_test() return value says a test command stopped at a fail count.

    The command stopped at a count of failing tests. The failing names of such a run are
    the first of them: a test after them was not run, in a file the run had started on
    as well.

    Args:
        result_dict: run_test()'s return value, or one command's own result.

    Returns:
        What a RunResult holds: True when the run's own report in what one of its
        commands printed (_own_text) holds that language's line for a run stopped at a
        count of failing tests (_FAIL_STOP_RE_DICT). For any other dict, True when its
        "output" holds the primary language's line. False for a language with no such
        line.
    """
    if isinstance(result_dict, RunResult):
        return result_dict.is_fail_stop
    return _fail_stop_in(result_dict, config.PRIMARY_EXT)


def early_stop_why(result_dict: dict) -> str:
    """Return what a run that ended before it had run every test says of why it failed.

    Args:
        result_dict: run_test()'s return value, or one command's own result.

    Returns:
        The lines saying the run ended early, each with the _EARLY_STOP_AFTER_LINE lines
        after it, cut to config.FAIL_EXCERPT_CHAR characters for each command, and
        "failed_excerpt" under them. A RunResult holds the lines of the run's own report
        in what its commands printed (_own_text); for any other dict they are read off
        its "output" as the primary language's. "failed_excerpt" alone when it holds
        those lines or there are none.
    """
    text = result_dict.early_stop_text if isinstance(result_dict, RunResult) \
        else _early_stop_text(result_dict["output"], config.PRIMARY_EXT)
    excerpt = result_dict["failed_excerpt"]
    if not text or text in excerpt:
        return excerpt
    return f"{text}\n{excerpt}" if excerpt else text


def early_stop_line_list(result_dict: dict) -> list[str]:
    """Return the lines of one run_test() return value that say a command ended early.

    The command ended before it had run every test, for another reason than a count of
    failing tests. The lines name what the run left out: a module that was not built, a
    task that failed before the tests.

    Args:
        result_dict: run_test()'s return value, or one command's own result.

    Returns:
        What a RunResult holds: every line of the run's own report in what its commands
        printed (_own_text) that the language's line for a run ended early finds
        (_EARLY_STOP_RE_DICT) and its line for a run stopped at a count of failing tests
        does not (_FAIL_STOP_RE_DICT), each with what a terminal acts on written out
        (textcut.term_text), cut to _MAX_FAIL_LINE_CHAR characters and folded into one
        line without the white space around it (textcut.one_line), in the order printed,
        a line printed twice twice. For any other dict, those lines of its "output" read
        as the primary language's. [] for a language with no such line.
    """
    if isinstance(result_dict, RunResult):
        return list(result_dict.early_stop_line_list)
    return _early_stop_line_list(result_dict["output"], config.PRIMARY_EXT)


def _early_stop_line_list(out: str, ext: str) -> list[str]:
    """Return the lines of a test command's output that say the run ended early.

    The run ended before every test was run, for another reason than a count of failing
    tests.

    Args:
        out: The text read as a test command's output.
        ext: The language whose lines are looked for.

    Returns:
        early_stop_line_list()'s lines for that output.
    """
    early_stop_re, fail_stop_re = _EARLY_STOP_RE_DICT.get(ext), _FAIL_STOP_RE_DICT.get(ext)
    if early_stop_re is None:
        return []
    return [one_line(cut(term_text(line), _MAX_FAIL_LINE_CHAR)) for line in line_list_of(out)
            if early_stop_re.search(line)
            and not (fail_stop_re is not None and fail_stop_re.search(line))]


def _early_stop_text(out: str, ext: str) -> str:
    """Return the lines of a test command's output that say the run ended before every test.

    Args:
        out: The text read as a test command's output.
        ext: The language whose line for a run ended early is looked for.

    Returns:
        Every line _EARLY_STOP_RE_DICT finds for the language and the
        _EARLY_STOP_AFTER_LINE lines after each, in the order they were printed and
        each once, cut to config.FAIL_EXCERPT_CHAR characters. "" when there is none.
    """
    early_stop_re = _EARLY_STOP_RE_DICT.get(ext)
    if early_stop_re is None:
        return ""
    line_list = line_list_of(out)
    keep_until = 0
    keep_list = []
    for index, line in enumerate(line_list):
        if early_stop_re.search(line):
            keep_until = index + _EARLY_STOP_AFTER_LINE + 1
        if index < keep_until:
            keep_list.append(line)
    return cut("\n".join(keep_list).strip(), config.FAIL_EXCERPT_CHAR)


def is_start_run(result_dict: dict) -> bool:
    """Whether one run_test() return value says a test command was started.

    Args:
        result_dict: run_test()'s return value.

    Returns:
        What a RunResult holds: True when one of its commands was started, whether it
        then ran to its end or past its time limit. False when none could be started,
        and for any other dict, which is what a call refused before a command was
        built returns.
    """
    return isinstance(result_dict, RunResult) and result_dict.is_start


def stop_short_run(result_dict: dict) -> bool:
    """Whether one run_test() return value says a test command stopped short of its tests.

    Args:
        result_dict: run_test()'s return value.

    Returns:
        What a RunResult holds: True when one of its commands could not be started, or
        ran to its end without a verdict and with its language's line for a run ended
        before it got to the tests (_STOP_SHORT_RE_DICT) in its run's own report
        (_own_text), which neither _is_no_test_text() nor _NO_TEST_FILE_DICT reads as a
        run over no tests.
        False otherwise. True for any other dict too, which is what a call refused
        before a command was built returns.
    """
    return not isinstance(result_dict, RunResult) or result_dict.is_stop_short


def fetch_fail_run(result_dict: dict) -> bool:
    """Whether one run_test() return value says what the tests need could not be fetched.

    Args:
        result_dict: run_test()'s return value.

    Returns:
        What a RunResult holds (RunResult.is_fetch_fail). False for any other dict.
    """
    return isinstance(result_dict, RunResult) and result_dict.is_fetch_fail


def _early_stop_in(result_dict: dict, ext: str) -> bool:
    """Whether one test command's result says the run ended before every test was run.

    Args:
        result_dict: The 5-key dict documented in run_test's Returns, or a dict whose
            "output" is the run's own report in what the command printed (_own_text).
        ext: The language whose line for a run ended early is looked for.

    Returns:
        True when the output holds that line (_EARLY_STOP_RE_DICT).
    """
    early_stop_re = _EARLY_STOP_RE_DICT.get(ext)
    return early_stop_re is not None and bool(early_stop_re.search(result_dict["output"]))


def _fail_stop_in(result_dict: dict, ext: str) -> bool:
    """Whether one test command's result says the run stopped at a count of failing tests.

    Args:
        result_dict: The 5-key dict documented in run_test's Returns, or a dict whose
            "output" is the run's own report in what the command printed (_own_text).
        ext: The language whose line for such a run is looked for.

    Returns:
        True when the output holds that line (_FAIL_STOP_RE_DICT).
    """
    fail_stop_re = _FAIL_STOP_RE_DICT.get(ext)
    return fail_stop_re is not None and bool(fail_stop_re.search(result_dict["output"]))


def _no_test_in(result_dict: dict, ext: str) -> bool:
    """Whether one test command's result says the run found no test of a language.

    Args:
        result_dict: The 5-key dict documented in run_test's Returns, or a dict of its
            "status" and, as "output", the run's own report in what the command printed
            (_own_text).
        ext: The language whose line for a run over no tests is looked for.

    Returns:
        True when the status is "did not run" and _is_no_test_text() reads the output as
        a run over no tests of the language.
    """
    return result_dict["status"] == TEST_STATUS_DICT[None] \
        and _is_no_test_text(result_dict["output"], ext)


def runnable_path_list(path_list: list[str]) -> list[str]:
    """Return the paths among some that the primary language's test command runs.

    Examples:
        (["tests/data/pairs.json", "tests/test_add.py"], primary "py")
            -> ["tests/test_add.py"]
        (["test/a.test.ts", "test/b.test.js"], primary "js") -> both

    Args:
        path_list: Repository-relative paths.

    Returns:
        The paths whose language is the primary language, or one with the same
        test_command_tuple, in the order given.
    """
    primary_config = LANG_DICT.get(config.PRIMARY_EXT)
    if primary_config is None:
        return []
    return [path for path in path_list
            if (lang_config := lang_of(path)) is primary_config
            or (lang_config is not None and lang_config.test_command_tuple
                and lang_config.test_command_tuple == primary_config.test_command_tuple)]


def _named_in_run(path: str, run_line_pattern: str, result_dict: dict) -> bool | None:
    """Say whether a run's output names a test file on its line for each test file run.

    The line is the one the file's command prints for each test file it runs.

    Args:
        path: The file, relative to the repository root.
        run_line_pattern: The file's language's run_line_pattern.
        result_dict: run_test()'s return value.

    Returns:
        True when a line names the file. False when none does and the output holds
        such lines for other files and is all the command printed: not cut to
        TEST_OUTPUT_TAIL, and with no part left out where it was received
        (_CUT_LINE_RE). None otherwise.
    """
    if not run_line_pattern:
        return None
    output = result_dict["output"]
    name = os.path.splitext(os.path.basename(path))[0]
    if re.search(run_line_pattern.replace("{name}", re.escape(name)), output, re.M):
        return True
    is_whole = (config.TEST_OUTPUT_TAIL <= 0 or len(output) < config.TEST_OUTPUT_TAIL) \
        and not _CUT_LINE_RE.search(output)
    if is_whole and re.search(run_line_pattern.replace("{name}", r"\S+"), output, re.M):
        return False
    return None


def _test_command_name(path: str) -> str:
    """Return the test command that runs a file, as part_dict_of() names it.

    Args:
        path: The file, relative to the repository root.

    Returns:
        The command a run_test() call naming the file hands it to (_path_by_language):
        another language's that can be run now, or the primary language's.
    """
    other_list = other_test_list()
    _, by_other_dict = _path_by_language([(path, path)], other_list)
    for position in by_other_dict:
        return " ".join(dict(config.OTHER_TEST_LIST)[other_list[position][0]])
    return " ".join(config.TEST_COMMAND)


def is_other_test_path(path: str) -> bool:
    """Say whether a run_test() call naming a file hands it to another language's command.

    The other command is one besides the primary language's.

    Args:
        path: The file, relative to the repository root.

    Returns:
        True when the command that runs the file is not TEST_COMMAND.
    """
    return _test_command_name(path) != " ".join(config.TEST_COMMAND)


def in_whole_run(path: str, whole_dict: dict) -> bool | None:
    """Say whether the whole suite runs a file.

    Where the file's language has a run_line_pattern, the output of the whole suite is
    read first. Where that tells nothing, the line the language has as
    compile_error_line is added at the end of the file, the whole suite is run against
    the package cache as it is, and the file is put back byte for byte. The test state
    is put back after that run (keep_test_state).

    Args:
        path: The file, relative to the repository root.
        whole_dict: run_test()'s return value for the whole suite with the file as it
            is.

    Returns:
        True when the output names the file, or the run with the line added says
        something else: another status, other failing tests, or that it ended before
        every test was run where the run without the line did not (early_stop_run).
        False when the output names other files and not this one, or the run with
        the line added says the same. None for a language with neither, for a
        whole_dict that did not run, and for a file that cannot be read or written
        (one that is not a regular file of at most tmpfile.READ_MAX_BYTE bytes is not
        read, tmpfile.read_file).

    Raises:
        OSError: When the file cannot be put back after the run: it cannot be written,
            or something other than a regular file stands at its path then
            (tmpfile.check_file). An exception other than OSError raised while the line
            is added or the suite runs, KeyboardInterrupt and SystemExit included, is
            raised as it is after the file is put back where it can be; a file that
            cannot be put back then is named in the log.
    """
    lang_config = lang_of(path)
    # What is read is the own run of the command that runs the file's language
    command = _test_command_name(path)
    whole_dict = part_of(whole_dict, command)
    if lang_config is None or whole_dict["status"] == TEST_STATUS_DICT[None]:
        return None
    named = _named_in_run(path, lang_config.run_line_pattern, whole_dict)
    if named is not None:
        return named
    if not lang_config.compile_error_line:
        return None
    try:
        full_path = resolve_in_project(path)
        body = tmpfile.read_file(full_path)
    except (*_PATH_ERROR_TUPLE, ValueError) as e:
        logger.info("whether the whole suite builds %s could not be told: %s", path, e)
        return None
    try:
        with open(full_path, "ab") as fp:
            fp.write(f"\n{lang_config.compile_error_line}\n".encode())
        with keep_test_state():
            probe_dict = part_of(run_test_as_is(), command)
    except OSError as e:
        logger.info("whether the whole suite builds %s could not be told: %s", path, e)
        probe_dict = None
    except BaseException:
        # An exception on its way out, an interrupt included, is raised as it is
        try:
            _put_back_byte(full_path, body)
        except OSError as e:
            logger.warning("%s could not be put back after the whole suite was run with "
                           "a line added to it: %s", path, e)
        raise
    _put_back_byte(full_path, body)
    if probe_dict is None:
        return None
    return (probe_dict["status"], sorted(probe_dict["failed"]), early_stop_run(probe_dict)) \
        != (whole_dict["status"], sorted(whole_dict["failed"]), early_stop_run(whole_dict))


def _put_back_byte(full_path: str, body: bytes) -> None:
    """Write a file's bytes back where nothing or a regular file stands.

    Args:
        full_path: The file, as an absolute path.
        body: The bytes it held.

    Raises:
        OSError: When the file cannot be written, or something other than a regular
            file stands at the path (tmpfile.check_file).
    """
    if os.path.exists(full_path):
        tmpfile.check_file(full_path, sys.maxsize)
    with open(full_path, "wb") as fp:
        fp.write(body)


def _no_run_reason(path_list: list[str] | None) -> str | None:
    """Return why run_test cannot run the primary language's tests, or None if it can.

    RUN_TEST turns it off whatever the repository holds. A primary language with no
    default test command (SQL, COBOL, BMS, JCL, CSD) runs nothing until TEST_COMMAND is
    set. Otherwise, a TEST_PATH that is not there is a reason under TEST_TARGET_STYLE
    "path" and under no other style.

    Args:
        path_list: The test paths a run_test call named. Explicit paths are run even
            when TEST_PATH itself is not in the repository.

    Returns:
        The line saying what to set, or None.
    """
    if not config.RUN_TEST:
        return "the tests are turned off for this run. Set RUN_TEST=true to run them"
    if not config.TEST_COMMAND:
        return ("TEST_COMMAND is not set and this language has no default test command. "
                "Set TEST_COMMAND to what runs this repository's tests")
    if not path_list and config.TEST_TARGET_STYLE == "path" and not has_test_path():
        return f"{config.TEST_PATH} does not exist. Point TEST_PATH at the target repository"
    return None


def test_can_run() -> bool:
    """Say whether run_test can run any test command.

    Returns:
        True when the primary language's tests can be run (_no_run_reason) or another
        language's can (_other_whole_list). False when RUN_TEST is off, and when neither
        can: the primary language has no test command and TEST_COMMAND is not set, or
        its tests are reached by path and that path is not in the repository.
    """
    return _no_run_reason(None) is None or bool(_other_whole_list())


def _count_patch(ok: bool) -> None:
    """Count the outcome of one edit. Counted under --dry-run too.

    Args:
        ok: Whether the edit landed (under --dry-run, whether it passed the checks).
    """
    global _patch_ok, _patch_ng, _patch_ng_after_ok
    if ok:
        _patch_ok += 1
        return
    _patch_ng += 1
    if _patch_ok:
        _patch_ng_after_ok += 1


def _close_patch_round() -> None:
    """Count a round when _is_patch_since_test is set, and clear it."""
    global _patch_round_count, _is_patch_since_test
    if _is_patch_since_test:
        _patch_round_count += 1
        _is_patch_since_test = False


def _patch_result_dict(ok: bool, path: str, stat: str = "", error: str = "",
                       diff: str = "") -> dict:
    """Build the return value of the editing tools (the ones that write).

    Each call counts one outcome of an edit (_count_patch()). dry_run and rounds carry
    the current setting and round count.

    Args:
        ok: Whether the edit landed (under --dry-run, whether it passed the checks).
        path: The target repository-relative path.
        stat: A summary of the change; empty on a failing path.
        error: Why it failed; empty on a successful path.
        diff: The diff that would have been applied, only under --dry-run.

    Returns:
        The 7-key dict documented in replace_in_file's Returns.
    """
    _count_patch(ok)
    return {"ok": ok, "path": path, "stat": stat, "error": error,
            "dry_run": config.EDIT_DRY_RUN, "diff": diff, "rounds": _patch_round_count}


def refusal_dict(path: str, error: str) -> dict:
    """Return what a writing tool hands back for a write it did not make.

    The write is counted as an edit that did not land (_count_patch()).

    Args:
        path: The path as the tool was given it.
        error: Why nothing was written.

    Returns:
        The 7-key dict documented in replace_in_file's Returns, "ok" False.
    """
    return _patch_result_dict(False, path, error=error)


def _over_round_limit(path: str) -> dict | None:
    """Return the refusal to hand back when the round limit is used up, else None.

    The limit is the one reset_patch_round() was given, and config.MAX_PATCH_ROUND when
    it was given none.

    Args:
        path: A path relative to PROJECT_DIR, echoed in the returned dict.

    Returns:
        The 7-key refusal dict, or None when there is budget left.
    """
    limit = config.MAX_PATCH_ROUND if _round_limit is None else _round_limit
    if 0 < limit <= _patch_round_count:
        return _patch_result_dict(
            False, path,
            error=f"the patch-and-test rounds hit their limit of "
                  f"{limit}. Nothing more will be written. "
                  f"Use revert_file to roll back, or COMPLETE() with what is still "
                  f"failing")
    return None


# What looking a path up raises when the system or git cannot take it: a name or a
# list of arguments that is too long, more directories than can be gone through
_PATH_ERROR_TUPLE = (OSError, RecursionError)

# What a writing tool answers when the checks of its path raised one of them
_PATH_ERROR_TEXT = "cannot check the path: {error}"


def _path_error_text(error: BaseException) -> str:
    """Return what a writing tool answers when the checks of its path raised.

    Args:
        error: One of _PATH_ERROR_TUPLE.

    Returns:
        _PATH_ERROR_TEXT with the system's reason, which names no path, or with the
        error's own text when it has none.
    """
    return _PATH_ERROR_TEXT.format(error=getattr(error, "strerror", None) or error)


def _target_path(path: str) -> tuple[str, str | None, dict | None]:
    """Resolve a path for writing, answering with a dict rather than raising.

    A path the checks refuse, or cannot be made of (_PATH_ERROR_TUPLE), comes back as
    the same 7-key dict as any other failure. The system is asked of the path once: one
    it cannot take, such as one that is too long, is refused here, and one that is not
    there or cannot be reached is not.

    Args:
        path: A path relative to PROJECT_DIR, as the tool was handed it.

    Returns:
        (the path as a string (path_text), the absolute path, None), or (the path as it
        was handed over, None, the 7-key failure dict).
    """
    try:
        text, target = path_text(path), resolve_in_project(path)
        with contextlib.suppress(FileNotFoundError, NotADirectoryError, PermissionError):
            os.lstat(target)
        return text, target, None
    except ValueError as e:
        return path, None, _patch_result_dict(False, path, error=str(e))
    except _PATH_ERROR_TUPLE as e:
        return path, None, _patch_result_dict(
            False, path, error=_path_error_text(e))


def _open_for_patch(path: str) -> tuple[str | None, textfile.TextForm, dict | None]:
    """Prepare for an edit: enforce the round limit and read the target file.

    The file has to be there already.

    How the file is stored - its encoding, its BOM and its line ending - is read off it
    and given back, and the text returned holds "\\n" alone as line ending.

    Args:
        path: A path relative to PROJECT_DIR.

    Returns:
        (the whole file, how it is stored, None), or
        (None, UTF8_FORM, the 7-key failure dict). A file that cannot be written back in
        the encoding it was read in is refused.
    """
    path, target, refuse_dict = _target_path(path)
    if target is None:
        return None, textfile.UTF8_FORM, refuse_dict
    refuse_dict = _over_round_limit(path)
    if refuse_dict:
        return None, textfile.UTF8_FORM, refuse_dict
    if not os.path.exists(target):
        return None, textfile.UTF8_FORM, _patch_result_dict(
            False, path,
            error=f"{path} does not exist. Use create_file to make a new file")
    try:
        text, form = textfile.read_for_edit(target)
    except OSError as e:
        return None, textfile.UTF8_FORM, _patch_result_dict(False, path,
                                                            error=f"cannot read: {e}")
    except ValueError as e:
        return None, textfile.UTF8_FORM, _patch_result_dict(
            False, path, error=f"cannot edit {path}: {e}")
    return text, form, None


def _no_carry_reason(path: str, real_path: str) -> str:
    """Say why git carries nothing of a path out of this run, or "" when it does.

    Args:
        path: The path as the tool was given it, which the reason names.
        real_path: The repository-relative path of the file it lands on (real_rel()).

    Returns:
        "git ignores <path>" for a path git ignores, "<path> is inside <directory>, a
        repository of its own" for one under a submodule or a nested repository
        (gitcmd.gitlink_of), "<path> is a repository of its own" for a path that is
        a submodule itself, "" for any other.
    """
    if gitcmd.is_ignored(config.PROJECT_DIR, real_path):
        return f"git ignores {path}"
    link = gitcmd.gitlink_of(config.PROJECT_DIR, real_path)
    if link == real_path:
        return f"{path} is a repository of its own"
    if link:
        return f"{path} is inside {link}, a repository of its own"
    return ""


def _write_patch(path: str, src: str, new_src: str, stat: str,
                 form: textfile.TextForm = textfile.UTF8_FORM,
                 is_new: bool = False) -> dict:
    """Write the edited contents. Under --dry-run it returns the diff without writing.

    A path git carries nothing of is refused (_no_carry_reason). So is a path where
    something other than a regular file or a directory stands (tmpfile.check_file), a
    text holding a character the file's encoding cannot write, both under --dry-run
    too, and a path the checks cannot be made of (_PATH_ERROR_TUPLE). A call that is
    refused leaves the directories as they are, and so does one whose new file could
    not be written or whose directories could not all be made: the directories made for
    it are deleted again (remove_empty_dir), and an empty directory the file was to
    take the place of is put back.

    Args:
        path: A path relative to PROJECT_DIR.
        src: The whole file before the edit (used to build the diff).
        new_src: The whole file after the edit.
        stat: The summary of the change for the return value.
        form: How the file is stored. The text is written with its encoding and its BOM,
            and the lines the edit did not touch keep their bytes. The file is given a
            modification time and size the path has not had (tmpfile.move_mtime_on).
        is_new: True makes room for a file that is not there, just before it is
            written (_make_room).

    Returns:
        The 7-key dict documented in replace_in_file's Returns.
    """
    global _is_patch_since_test

    path, target, refuse_dict = _target_path(path)
    if target is None:
        return refuse_dict
    # A write through a symbolic link is recorded and checked under the file it lands on
    try:
        real_path = real_rel(path)
        reason = _no_carry_reason(path, real_path)
    except _PATH_ERROR_TUPLE as e:
        return _patch_result_dict(False, path, error=_path_error_text(e))
    if reason:
        return _patch_result_dict(
            False, path,
            error=f"{reason}, so nothing written there is carried out of "
                  f"this run. Write what the change needs into a file git tracks")
    try:
        if os.path.exists(target) and not os.path.isdir(target):
            tmpfile.check_file(target, sys.maxsize)
    except OSError as e:
        return _patch_result_dict(False, path, error=f"cannot write: {e}")
    try:
        content = textfile.encode_text(src, new_src, form)
    except ValueError as e:
        return _patch_result_dict(False, path, error=f"cannot write: {e}")
    if config.EDIT_DRY_RUN:
        diff = "".join(difflib.unified_diff(line_list_of(src, keepends=True),
                                            line_list_of(new_src, keepends=True),
                                            fromfile=f"a/{path}", tofile=f"b/{path}"))
        _dry_change_list.append(diff)
        _dry_path_set.add(real_path)
        return _patch_result_dict(True, path, stat=stat, diff=diff)
    is_dir_remove = False
    if is_new:
        try:
            is_dir_remove = _make_room(target)
        except OSError as e:
            remove_empty_dir(real_path)
            return _patch_result_dict(False, path, error=f"cannot create directory: {e}")
    _keep_start(real_path)
    before_stamp = tmpfile.file_stamp(target)
    try:
        with open(target, "wb") as fp:
            fp.write(content)
    except OSError as e:
        if is_new:
            _take_new_file_back(target, real_path, is_dir_remove)
        return _patch_result_dict(False, path, error=f"cannot write: {e}")
    tmpfile.move_mtime_on(target, before_stamp)
    # The next test command that is started closes one round. keep_test_state() puts the
    # round count and _is_patch_since_test back as they were before its block
    _is_patch_since_test = True
    if _hold_start(real_path):
        _write_hash_dict.pop(real_path, None)
    else:
        _write_hash_dict[real_path] = _file_hash(real_path)
    return _patch_result_dict(True, path, stat=stat)


def keep_new_dir(rel_dir: str) -> None:
    """Record a directory as one made during this request.

    Args:
        rel_dir: The repository-relative path of a directory made during this request.
    """
    _new_dir_set.add(rel_dir)


def _make_parent_dir(target: str) -> None:
    """Make the directories above a file that are not there, and record each as made.

    Args:
        target: The absolute path of the file, as resolve_in_project() gives it.

    Raises:
        OSError: When a directory cannot be made.
    """
    root = os.path.realpath(config.PROJECT_DIR)
    parent = os.path.dirname(target)
    new_dir_list = []
    directory = parent
    while not os.path.lexists(directory):
        new_dir_list.append(directory)
        directory = os.path.dirname(directory)
    # The one nearest the repository root first, each recorded once it is made
    for directory in reversed(new_dir_list):
        os.mkdir(directory)
        keep_new_dir(os.path.relpath(directory, root))
    os.makedirs(parent, exist_ok=True)


def _is_empty_dir(target: str) -> bool:
    """Say whether a path is a directory with nothing in it.

    Args:
        target: The absolute path, as resolve_in_project() gives it.

    Returns:
        False for the repository root, and for a directory that cannot be listed.
    """
    try:
        return os.path.isdir(target) and not os.listdir(target) \
            and target != os.path.realpath(config.PROJECT_DIR)
    except OSError:
        return False


def _make_room(target: str) -> bool:
    """Make the place a new file is written at.

    A directory with nothing in it at the path is taken away, and the directories above
    the path are made (_make_parent_dir).

    Args:
        target: The absolute path of the file, as resolve_in_project() gives it.

    Returns:
        Whether a directory was taken away.

    Raises:
        OSError: When the directory cannot be taken away or one cannot be made.
    """
    is_dir_remove = _is_empty_dir(target)
    if is_dir_remove:
        os.rmdir(target)
    _make_parent_dir(target)
    return is_dir_remove


def _take_new_file_back(target: str, real_path: str, is_dir_remove: bool) -> None:
    """Leave the working tree as it was before a new file that could not be written.

    Args:
        target: The absolute path of the file.
        real_path: Its repository-relative path.
        is_dir_remove: Whether _make_room() took an empty directory away at the path,
            which is made again.
    """
    with contextlib.suppress(OSError):
        os.remove(target)
    remove_empty_dir(real_path)
    if is_dir_remove:
        with contextlib.suppress(OSError):
            os.mkdir(target)


def remove_empty_dir(rel_path: str) -> None:
    """Delete the directories made during this request that a removed file leaves empty.

    They are removed by walking up from the file that was deleted, or that could not be
    made. A directory above the file that is not there, one that could not be made
    either, is passed over. The walk ends at the first directory keep_new_dir() did not
    record: such a directory stays, empty or not.

    Args:
        rel_path: The repository-relative path of the file that was removed, or was
            not made.
    """
    root = os.path.realpath(config.PROJECT_DIR)
    rel_dir = os.path.dirname(rel_path)
    while rel_dir and rel_dir not in _new_dir_set \
            and not os.path.lexists(os.path.join(root, rel_dir)):
        rel_dir = os.path.dirname(rel_dir)
    while rel_dir in _new_dir_set:
        try:
            os.rmdir(os.path.join(root, rel_dir))
        except OSError:
            return          # not empty, or gone already
        _new_dir_set.discard(rel_dir)
        rel_dir = os.path.dirname(rel_dir)


def _keep_start(path: str) -> None:
    """Keep what a path held when the request started, before a tool first writes it.

    Only a path that already differed from HEAD when the request started is kept, and
    only while it still holds what it held then (_base_hash_dict). A path that is
    neither a symbolic link nor a regular file (tmpfile.check_file) is not read and is
    not kept.

    Args:
        path: A repository-relative path, as the file it lands on (real_rel()).
    """
    if path in _start_byte_dict or path in _start_link_dict \
            or path not in _base_hash_dict or _file_hash(path) != _base_hash_dict[path]:
        return
    try:
        link = _link_entry(path)
        if link:
            _start_link_dict[path] = os.readlink(link)
            return
        full_path = resolve_in_project(path)
        tmpfile.check_file(full_path, sys.maxsize)
        with open(full_path, "rb") as fp:
            _start_byte_dict[path] = fp.read()
    except FileNotFoundError:
        _start_byte_dict[path] = None
    except (*_PATH_ERROR_TUPLE, ValueError) as e:
        logger.info("what %s held when the request started could not be kept (%s)",
                    path, e)


def _hold_start(path: str) -> bool:
    """Say whether a path holds what it held when the request started.

    A path that already differed from HEAD then is compared with what it held
    (_base_hash_dict). Any other path is compared with HEAD: it holds what it started
    with when git tracks it and reports no difference from HEAD, and when git does not
    track it and nothing is there.

    Args:
        path: A repository-relative path, as the file it lands on (real_rel()).

    Returns:
        True when the path holds its starting contents. False when it does not, and
        when git could not answer.
    """
    if path in _base_hash_dict:
        return _file_hash(path) == _base_hash_dict[path]
    literal_path = f":(literal){path}"
    try:
        track_proc = gitcmd.run(config.PROJECT_DIR, "ls-files", "--error-unmatch", "--",
                                literal_path)
        if track_proc.returncode == _NO_MATCH_STATUS:
            return not os.path.lexists(
                os.path.join(os.path.realpath(config.PROJECT_DIR), path))
        if track_proc.returncode != 0:
            return False
        diff_proc = gitcmd.run(config.PROJECT_DIR, *gitcmd.PATH_DIFF_TUPLE, "--quiet",
                               "HEAD", "--", literal_path)
    except OSError:
        return False
    return diff_proc.returncode == 0


def _match_line_list(src: str, old_text: str) -> list[int]:
    """Return the 1-based line numbers of every place old_text appears.

    Overlapping matches are not counted, the same way str.find counts.

    Args:
        src: The whole file as a string.
        old_text: The string to look for.

    Returns:
        The starting line number of each match.
    """
    line_list = []
    at = src.find(old_text)
    while at != -1:
        line_list.append(src.count("\n", 0, at) + 1)
        at = src.find(old_text, at + len(old_text))
    return line_list


def _no_match_hint(src: str, old_text: str, path: str = "") -> str:
    """Return a one-line guess at why old_text was not found.

    It separates "only the indentation or trailing whitespace differs" from "the text
    itself differs from line 2 on". In a file whose lines can be in fixed format, the
    first says the columns differ, and lines whose code matches while their sequence or
    identification areas do not (cobol.line_key_list()) are said to differ there.

    Args:
        src: The whole file as a string.
        old_text: The string that was not found.
        path: The file's path, which says its language. "" is no language in particular.

    Returns:
        The guidance text, or an empty string when there is no clue.
    """
    def strip_each_line(text: str) -> str:
        """Strip leading and trailing whitespace per line, to compare shapes only."""
        return "\n".join(line.strip() for line in line_list_of(text))

    lang_config = lang_of(path) if path else None
    is_column = lang_config is not None and lang_config.column_layout
    if strip_each_line(old_text) in strip_each_line(src):
        if is_column:
            return (" (only the spaces differ, which puts the code in other columns. "
                    "Pass exactly the string read_source_file returned, with every "
                    "space)")
        return (" (whitespace is the only difference. Pass exactly the string "
                "read_source_file returned)")
    if is_column:
        src_key_list = cobol.line_key_list(line_list_of(src))
        old_key_list = cobol.line_key_list(line_list_of(old_text))
        key_count = len(old_key_list)
        if key_count and any(src_key_list[start:start + key_count] == old_key_list
                             for start in range(len(src_key_list) - key_count + 1)):
            return (" (the code matches, but columns 1-6 (the sequence numbers), the "
                    "columns from 73 on or the spaces differ. Pass the lines exactly as "
                    "read_source_file returned them, columns 1-6 included)")
    head = line_list_of(old_text.strip())[0].strip() if old_text.strip() else ""
    if head and head in src:
        return (f" (line 1 {head[:config.RAW_EXCERPT_CHAR]!r} is found; "
                f"the lines after it disagree)")
    return ""


def replace_in_file(path: str, old_text: str, new_text: str) -> dict:
    """Replace the single occurrence of old_text in a file with new_text.

    It takes no line numbers, no "@@" and no context lines, and handles multi-line
    replacements.

    old_text must match exactly once. With 0 or 2+ matches nothing is written and which
    it was goes into "error". Add surrounding lines until it is unique.

    Args:
        path: A path relative to PROJECT_DIR, e.g. "myapp/config/settings.py".
        old_text: The text before the change. Use exactly what read_source_file
            returned, with the original indentation and line breaks.
        new_text: The text after the change. An empty string deletes old_text.

    Returns:
        {"ok": bool,       # whether the replacement happened
         "path": str,      # the target path
         "stat": str,      # the line number replaced and the line delta; "" on failure
         "error": str,     # why it failed; "" on success
         "dry_run": bool,  # True when config.EDIT_DRY_RUN is True; nothing was written
         "diff": str,      # the unified diff of the replacement, only when dry_run
         "rounds": int}    # patch_round_count() as the call returns
        When "dry_run" is True, "ok" says whether the replacement passed the checks.

    """
    # The path is checked before old_text
    path, target, refuse_dict = _target_path(path)
    if target is None:
        return refuse_dict
    argument_error = text_error(old_text=old_text, new_text=new_text)
    if argument_error:
        return _patch_result_dict(False, path, error=argument_error)
    old_text, new_text = textfile.lf_text(old_text), textfile.lf_text(new_text)
    if not old_text:
        return _patch_result_dict(
            False, path, error="old_text is empty. Pass the text before the change")

    src, form, early_dict = _open_for_patch(path)
    if src is None:
        return early_dict

    # Exactly one match is required; 0 matches and 2+ matches get their own message
    at_line_list = _match_line_list(src, old_text)
    if not at_line_list:
        return _patch_result_dict(
            False, path,
            error=f"old_text not found{_no_match_hint(src, old_text, path)}")
    if len(at_line_list) > 1:
        return _patch_result_dict(
            False, path,
            error=f"old_text matches in {len(at_line_list)} places "
                  f"(lines {at_line_list[:config.MAX_NAME_ITEM]}). Add surrounding lines to "
                  f"make it unique")
    if old_text == new_text:
        return _patch_result_dict(False, path, error="old_text and new_text are identical")

    new_src = src.replace(old_text, new_text, 1)
    add_count = len(line_list_of(new_text))
    remove_count = len(line_list_of(old_text))
    stat = f"{path} line {at_line_list[0]} | +{add_count} -{remove_count} lines"
    return _write_patch(path, src, new_src, stat, form)


def _range_error(path: str, start_line: int, end_line: int,
                 total: int) -> str:
    """Return why replace_line_range's line range is invalid, or "" when it is fine.

    The file's line count goes into the message.

    Args:
        path: The target repository-relative path.
        start_line: The line to start replacing at (1-based).
        end_line: The line to stop replacing at.
        total: The total number of lines in that file.

    Returns:
        Why it failed, or an empty string when the range is valid.
    """
    # An empty file has no valid range; the message names the tool that can write it
    if total == 0:
        return (f"{path} is empty, so it has no lines to replace. "
                f"Use create_file to write its whole contents")
    if start_line < 1:
        return f"start_line must be 1 or more: {start_line}"
    if end_line < start_line:
        return (f"end_line must be start_line or more: "
                f"start_line={start_line} end_line={end_line}")
    if end_line > total:
        return (f"{path} has only {total} lines: end_line={end_line}. "
                f"Re-read it with read_source_file to get the line numbers again")
    return ""


def _line_range_stat(path: str, start_line: int, end_line: int,
                     old_line_list: list[str], new_text: str) -> str:
    """Build replace_line_range's "stat": the line delta and the range's first lines.

    Args:
        path: The target repository-relative path.
        start_line: The line the replacement started at.
        end_line: The line the replacement ended at.
        old_line_list: The lines of that range before the change (newlines included).
        new_text: The text after the change.

    Returns:
        One "<path> lines A-B | +N -M lines | was: <the first lines>" line, holding at
        most config.STAT_ECHO_LINE of them, each cut to config.STAT_ECHO_CHAR.
    """
    before_text = " / ".join(line.rstrip("\n")[:config.STAT_ECHO_CHAR]
                             for line in old_line_list[:config.STAT_ECHO_LINE])
    return (f"{path} lines {start_line}-{end_line} | "
            f"+{len(line_list_of(new_text))} -{len(old_line_list)} lines | "
            f"was: {before_text}")


def replace_line_range(path: str, start_line: int, end_line: int,
                       new_text: str) -> dict:
    """Replace lines start_line..end_line of a file with new_text (no text match needed).

    This tool takes the line numbers read_definition or read_source_file returned as
    they are. Unlike replace_in_file it never needs the old text pasted back.

    Line numbers drift with every edit. After one edit, re-read the file with
    read_source_file before building the next call.

    Args:
        path: A path relative to PROJECT_DIR, e.g. "myapp/config/settings.py".
        start_line: The line to start replacing at (1-based, inclusive).
        end_line: The line to stop replacing at (inclusive).
        new_text: The text after the change. An empty string deletes that range.

    Returns:
        The same 7-key dict as replace_in_file. "stat" holds the line delta and the
        first config.STAT_ECHO_LINE lines of the range before the change.

    """
    argument_error = (whole_number_error("start_line", start_line)
                      or whole_number_error("end_line", end_line)
                      or text_error(new_text=new_text))
    if argument_error:
        return _patch_result_dict(False, path, error=argument_error)
    new_text = textfile.lf_text(new_text)
    src, form, early_dict = _open_for_patch(path)
    if src is None:
        return early_dict

    line_list = line_list_of(src, keepends=True)
    error = _range_error(path, start_line, end_line, len(line_list))
    if error:
        return _patch_result_dict(False, path, error=error)

    old_line_list = line_list[start_line - 1:end_line]
    body = new_text
    # Whole lines are replaced: a trailing newline is added unless the range's last line
    # had no line ending either (the end of the file)
    if body and not body.endswith("\n") and old_line_list[-1].endswith(("\n", "\r")):
        body += "\n"
    new_src = "".join(line_list[:start_line - 1]) + body + "".join(line_list[end_line:])
    if new_src == src:
        return _patch_result_dict(
            False, path,
            error=f"lines {start_line}-{end_line} already match new_text")
    return _write_patch(
        path, src, new_src,
        _line_range_stat(path, start_line, end_line, old_line_list, new_text), form)


def _bom_only_form(target: str) -> textfile.TextForm | None:
    """Return how a file that holds bytes and no line is stored.

    It is the file replace_line_range refuses as empty (_range_error): one whose text
    read_for_edit() returns as "".

    Args:
        target: The file's absolute path.

    Returns:
        The file's TextForm when it holds nothing but a BOM. None for a file holding
        text and for one that cannot be read.
    """
    try:
        src, form = textfile.read_for_edit(target)
    except (OSError, ValueError):
        return None
    return form if src == "" else None


def create_file(path: str, text: str) -> dict:
    """Create a new file with text as its whole contents.

    The other two editing tools only rewrite a file that is already there. This one
    refuses a path whose file has contents; use replace_in_file or replace_line_range for
    that. A file that is there but holds no line, one of no bytes or of a BOM alone,
    is written, the BOM kept: neither of the other two can write it.

    Missing parent directories are created, and a directory with nothing in it at the
    path gives way to the file, both just before the file is written: a call that is
    refused leaves them as they are. A new file is untracked: git_diff_stat reports it
    under "untracked" rather than "files", and revert_file deletes it with the
    directories made for it.

    Args:
        path: A path relative to PROJECT_DIR that does not exist yet or is empty,
            e.g. "myapp/cache.py".
        text: The whole contents of the new file, read as textfile.lf_text() gives
            it. A trailing newline is added when it is missing.

    Returns:
        The same 7-key dict as replace_in_file. "stat" holds the line count.
    """
    path, target, refuse_dict = _target_path(path)
    if target is None:
        return refuse_dict
    refuse_dict = _over_round_limit(path)
    if refuse_dict:
        return refuse_dict
    argument_error = text_error(text=text)
    if argument_error:
        return _patch_result_dict(False, path, error=argument_error)
    # A path that holds contents is refused. A file with no line in it is written over,
    # and a directory with nothing in it gives way to the file
    form = None
    if not _is_empty_dir(target) and os.path.exists(target) \
            and (os.path.isdir(target) or os.path.getsize(target) > 0):
        form = _bom_only_form(target)
        if form is None:
            return _patch_result_dict(
                False, path,
                error=f"{path} already exists. Use replace_in_file or replace_line_range "
                      f"to change a file that is already there")
    text = textfile.lf_text(text)
    if not text:
        return _patch_result_dict(False, path, error="text is empty")

    body = text if text.endswith("\n") else text + "\n"
    return _write_patch(path, "", body, f"new file, {len(line_list_of(body))} lines",
                        form or textfile.new_file_form(config.PROJECT_DIR, path),
                        is_new=True)


def path_arg(path: str) -> str:
    """Return a path in the form a command reads as a path and not as an option.

    Examples:
        "-x.py"           -> "./-x.py"
        "-x.py::test_a"   -> "./-x.py::test_a"
        "tests/test_a.py" -> "tests/test_a.py"

    Args:
        path: A path relative to the directory the command is started in.

    Returns:
        path, with "./" in front of it when it starts with "-".
    """
    return f"./{path}" if path.startswith("-") else path


def _root_target(project_dir: str, ext: str) -> str:
    """Return the file in the repository root a language's build tool is pointed at.

    Examples:
        (a root holding Calc.slnx and Build.csproj, "cs") -> "Calc.slnx"
        (a root holding A.sln and B.sln, "cs")            -> ""
        (a root holding App.csproj only, "cs")            -> "App.csproj"
        (a root holding -App.sln only, "cs")              -> "./-App.sln"

    Args:
        project_dir: The directory the command is started in.
        ext: A LANG_DICT key.

    Returns:
        The one file name the first root_target_tuple group with a match matches, as
        path_arg() writes it, or "" when that group matches more than one, when no
        group matches and for a language with no root_target_tuple.
    """
    lang_config = LANG_DICT.get(ext)
    if lang_config is None or not lang_config.root_target_tuple:
        return ""
    try:
        name_list = sorted(entry.name for entry in os.scandir(project_dir)
                           if entry.is_file())
    except OSError:
        return ""
    for pattern_tuple in lang_config.root_target_tuple:
        match_list = [name for name in name_list
                      if any(fnmatch.fnmatchcase(name, pattern) for pattern in pattern_tuple)]
        if match_list:
            return path_arg(match_list[0]) if len(match_list) == 1 else ""
    return ""


def _pyproject_dict(root: str) -> dict | None:
    """Return a directory's pyproject.toml read as TOML.

    Args:
        root: The directory, as an absolute path.

    Returns:
        The table, {} when there is no pyproject.toml, or None when it cannot be read.
    """
    try:
        with open(os.path.join(root, "pyproject.toml"), "rb") as fp:
            return tomllib.load(fp)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError):
        return None


# The modules setuptools' build of a setup.py can import past the standard library:
# setuptools itself and what it puts on sys.path
_SETUPTOOLS_BUILD_MODULE_SET = frozenset({
    "setuptools", "distutils", "_distutils_hack", "packaging", "wheel", "tomli",
    "importlib_metadata", "zipp", "more_itertools", "platformdirs", "jaraco",
    "backports", "autocommand"})

# The build backends that run setup.py ("" when the build-system table names none)
_SETUPTOOLS_BACKEND_TUPLE = ("", "setuptools.build_meta", "setuptools.build_meta:__legacy__")

# The build-system requires that bring no module past _SETUPTOOLS_BUILD_MODULE_SET: wheel,
# and setuptools with no upper bound or pin before any marker
_SETUPTOOLS_REQUIRE_RE = re.compile(
    r"\s*(?:wheel(?![\w.-]).*|setuptools(?![\w.-])\s*(?:\[[^\]]*\])?(?:[^;<=~]|>=|!=)*"
    r"(?:;.*)?)$",
    re.I)

# A line of a requirement file that reads another one, with that file's path
_INCLUDE_REQUIREMENT_RE = re.compile(r"^\s*(?:-r|--requirement)[\s=]\s*(\S+)")

# A line of a requirement file that installs the directory it is in
_SELF_REQUIREMENT_RE = re.compile(
    r"^\s*(?:(?:-e|--editable)\s+)?(?:file:)?\.(?:/)?(?:\[[^\]]*\])?\s*(?:#.*)?$")

# The directories whose project was said to be left uninstalled (uv_project_word_list)
_unbuild_warn_root_set: set[str] = set()


def _changes_sys_path(tree: ast.Module) -> bool:
    """Say whether a module adds to or sets sys.path anywhere in it.

    Args:
        tree: The parsed module.

    Returns:
        True when it calls a method of sys.path or assigns to sys.path or a slice of it.
    """
    for node in ast.walk(tree):
        target_list = [node.func.value] if isinstance(node, ast.Call) \
            and isinstance(node.func, ast.Attribute) else []
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            target_list = node.targets if isinstance(node, ast.Assign) else [node.target]
        target_list = [target.value if isinstance(target, ast.Subscript) else target
                       for target in target_list]
        if any(isinstance(target, ast.Attribute) and target.attr == "path"
               and isinstance(target.value, ast.Name) and target.value.id == "sys"
               for target in target_list):
            return True
    return False


def _top_import_set(path: str) -> set[str] | None:
    """Return the top-level names a Python file imports directly in its module body.

    Args:
        path: The file, as an absolute path.

    Returns:
        The first part of each absolute import outside a try statement, or None when
        the file cannot be read or parsed, or changes sys.path (_changes_sys_path).
    """
    try:
        with open(path, "rb") as fp:
            source = fp.read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return None
    if _changes_sys_path(tree):
        return None
    name_set = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            name_set.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            name_set.add(node.module.split(".")[0])
    return name_set


def _setup_py_unbuild_list(root: str) -> list[str]:
    """Return the modules a directory's setup.py imports that setuptools' build cannot.

    Args:
        root: The directory holding setup.py, as an absolute path.

    Returns:
        The names _top_import_set() finds in setup.py, and in the file or package
        beside it that setup.py imports, that are neither in the standard library or
        _SETUPTOOLS_BUILD_MODULE_SET nor a file or package beside setup.py, sorted.
        Empty when setup.py gives None.
    """
    name_set = _top_import_set(os.path.join(root, "setup.py"))
    if name_set is None:
        return []

    def own_path(name: str) -> str:
        """Return a module's file beside setup.py, a directory with no __init__.py, or ""."""
        for path in (os.path.join(root, name + ".py"),
                     os.path.join(root, name, "__init__.py")):
            if os.path.isfile(path):
                return path
        dir_path = os.path.join(root, name)
        return dir_path if os.path.isdir(dir_path) else ""

    for name in sorted(name_set):
        path = own_path(name)
        if path.endswith(".py"):
            name_set |= _top_import_set(path) or set()
    return sorted(
        name for name in name_set
        if name not in sys.stdlib_module_names and name not in _SETUPTOOLS_BUILD_MODULE_SET
        and not own_path(name))


def _installs_own_dir(root: str, name: str, visit_set: set[str] | None = None) -> bool:
    """Say whether a requirement file installs the directory it is in.

    Args:
        root: The directory, as an absolute path.
        name: The requirement file's path from it.
        visit_set: The real paths of the files already read on the way here.

    Returns:
        True when one of its lines matches _SELF_REQUIREMENT_RE, or one reads
        (_INCLUDE_REQUIREMENT_RE) a file in the same directory that does.
    """
    path = os.path.realpath(os.path.join(root, name))
    visit_set = set() if visit_set is None else visit_set
    if path in visit_set or os.path.dirname(path) != os.path.realpath(root):
        return False
    visit_set.add(path)
    try:
        with open(path, encoding="utf-8", errors="replace") as fp:
            line_list = line_list_of(fp.read())
    except OSError:
        return False
    for line in line_list:
        if _SELF_REQUIREMENT_RE.match(line):
            return True
        include = _INCLUDE_REQUIREMENT_RE.match(line)
        if include and _installs_own_dir(root, include.group(1), visit_set):
            return True
    return False


def uv_project_word_list(root: str) -> list[str]:
    """Return what "uv run" is given in a directory that holds no project uv installs.

    Examples:
        (a directory with pyproject.toml holding [project])  -> []
        (a directory with setup.py and requirements-dev.txt)
            -> ["--isolated", "--with-editable", ".",
                "--with-requirements", "requirements-dev.txt"]
        (a directory whose pyproject.toml holds [tool.black] alone) -> ["--isolated"]
        (a directory with no pyproject.toml whose setup.py imports Cython) -> ["--isolated"]

    Args:
        root: The directory the command is started in, as an absolute path.

    Returns:
        [] where pyproject.toml has a project table or cannot be read. Otherwise
        "--isolated", then "--with-editable ." where there is a setup.py, or a
        pyproject.toml with a build-system table that is not Poetry's with package-mode
        false, then "--with-requirements <file>" for each file of
        PY_REQUIREMENT_FILE_TUPLE there. Where setup.py is built by setuptools alone (no
        build-system table, or one naming a setuptools backend or none and requiring
        setuptools and wheel only) and imports what that build cannot
        (_setup_py_unbuild_list), "--with-editable ." and the files that install the
        directory itself (_installs_own_dir) are left out, which is logged once per
        directory.
    """
    pyproject_dict = _pyproject_dict(root)
    if pyproject_dict is None or "project" in pyproject_dict:
        return []
    poetry_dict = pyproject_dict.get("tool", {}).get("poetry", {})
    build_dict = pyproject_dict.get("build-system", {})
    has_setup_py = os.path.isfile(os.path.join(root, "setup.py"))
    is_package = has_setup_py or (
        "build-system" in pyproject_dict and poetry_dict.get("package-mode", True))
    unbuild_list = []
    require_list = build_dict.get("requires", []) if isinstance(build_dict, dict) else None
    if has_setup_py and isinstance(require_list, list) \
            and build_dict.get("build-backend", "") in _SETUPTOOLS_BACKEND_TUPLE \
            and all(_SETUPTOOLS_REQUIRE_RE.match(str(item)) for item in require_list):
        unbuild_list = _setup_py_unbuild_list(root)
    word_list = ["--isolated"] + (["--with-editable", "."]
                                  if is_package and not unbuild_list else [])
    skip_list = []
    for name in PY_REQUIREMENT_FILE_TUPLE:
        if not os.path.isfile(os.path.join(root, name)):
            continue
        if unbuild_list and _installs_own_dir(root, name):
            skip_list.append(name)
        else:
            word_list += ["--with-requirements", name]
    if unbuild_list and root not in _unbuild_warn_root_set:
        _unbuild_warn_root_set.add(root)
        logger.warning(
            "%s imports %s, which the build of the project cannot load, so the commands "
            "uv runs there do not install the project%s; set the command to install it "
            "another way", os.path.join(root, "setup.py"), ", ".join(unbuild_list),
            f" and leave out {', '.join(skip_list)}, which install it" if skip_list else "")
    return word_list


def with_uv_project_word(argv: list[str], default_tuple: tuple[str, ...],
                         root: str) -> list[str]:
    """Return a command that is a "uv run" default with uv_project_word_list() in it.

    Args:
        argv: The command as the settings hold it.
        default_tuple: The language's default for it.
        root: The directory the command is started in, as an absolute path.

    Returns:
        argv with uv_project_word_list() of root after its "uv run" when default_tuple
        starts with "uv run" and argv starts with default_tuple; otherwise argv as it
        is, as a new list.
    """
    if default_tuple[:2] != ("uv", "run") \
            or argv[:len(default_tuple)] != list(default_tuple):
        return list(argv)
    return argv[:2] + uv_project_word_list(root) + argv[2:]


def _with_root_target(argv: list[str], default_tuple: tuple[str, ...],
                      ext: str = "") -> list[str]:
    """Return a command with a language's root target after it.

    Examples:
        (["dotnet", "test"], ("dotnet", "test")) -> ["dotnet", "test", "Calc.slnx"]
        (["dotnet", "test", "src/A.sln"], ("dotnet", "test"))
            -> ["dotnet", "test", "src/A.sln"]

    Args:
        argv: config.TEST_COMMAND or config.CHECK_COMMAND.
        default_tuple: The language's default for it (test_command_tuple or
            check_command_tuple).
        ext: The language's LANG_DICT key. Empty takes the primary language.

    Returns:
        argv followed by _root_target() of PROJECT_DIR when argv is the default, with or
        without the language's offline_arg_tuple, and there is a root target; otherwise
        argv unchanged.
    """
    ext = ext or config.PRIMARY_EXT
    lang_config = LANG_DICT.get(ext)
    if lang_config is None or not default_tuple or argv not in (
            list(default_tuple), list(default_tuple + lang_config.offline_arg_tuple)):
        return list(argv)
    target = _root_target(config.PROJECT_DIR, ext)
    return list(argv) + [target] if target else list(argv)


def _bare_test_command(ext: str) -> list[str]:
    """Return the bare test command a language's tests run with as the primary one, or [].

    Args:
        ext: The language's LANG_DICT key.

    Returns:
        The language's bare_test_command_tuple when it has one, config.TEST_COMMAND is
        its default test_command_tuple, PROJECT_DIR is set, the files of the repository
        can be listed and it holds no file of its dep_file_tuple outside the directories
        its build_output_tuple names by name alone; [] otherwise.
    """
    lang_config = LANG_DICT.get(ext)
    if lang_config is None or not lang_config.bare_test_command_tuple \
            or config.TEST_COMMAND != list(lang_config.test_command_tuple) \
            or not config.PROJECT_DIR:
        return []
    file_list = gitcmd.work_tree_file_list(config.PROJECT_DIR)
    if file_list is None:
        return []
    skip_set = name_only_dir_set(lang_config.build_output_tuple)
    if any(is_dep_file(path, lang_config.dep_file_tuple)
           and not has_name_of(path.split("/")[:-1], skip_set) for path in file_list):
        return []
    return list(lang_config.bare_test_command_tuple)


def _bare_test_file_list(ext: str, file_list: list[str]) -> list[str]:
    """Return the test files a bare test command runs when it is given no paths.

    Args:
        ext: The language's LANG_DICT key.
        file_list: gitcmd.work_tree_file_list() of PROJECT_DIR.

    Returns:
        The files of file_list in the working tree that are of the language and are
        tests (under_test_path()) or match its bare_test_name_tuple (is_test_path()),
        outside the directories its build_output_tuple names by name alone, each as
        path_arg() writes it.
    """
    lang_config = LANG_DICT[ext]
    skip_set = name_only_dir_set(lang_config.build_output_tuple)
    test_set = set(under_test_path(file_list))
    return [path_arg(path) for path in file_list
            if lang_of(path) is lang_config
            and (path in test_set or is_test_path(path, "", lang_config.bare_test_name_tuple))
            and not has_name_of(path.split("/")[:-1], skip_set)
            and os.path.isfile(os.path.join(config.PROJECT_DIR, path))]


def command_text(command: str) -> str:
    """Return a test command as compare_part_list() names it, written as it runs.

    Args:
        command: The command's line, as the settings hold it.

    Returns:
        primary_test_command() for the primary language's command, command otherwise.
    """
    if command == " ".join(config.TEST_COMMAND):
        return " ".join(primary_test_command())
    return command


def primary_test_command() -> list[str]:
    """Return the primary language's test command as run_test() starts it.

    Returns:
        _bare_test_command() of the primary language when it is not [], otherwise
        config.TEST_COMMAND.
    """
    return _bare_test_command(config.PRIMARY_EXT) or list(config.TEST_COMMAND)


def _build_argv(path_list: list[str] | None, style: str, test_path: str,
                ext: str, command: list[str] | None = None
                ) -> tuple[list[str], list[str], str]:
    """Build argv and the targets according to TEST_TARGET_STYLE.

    "path" appends the targets to the command, "separator" appends them after "--", and
    "none" drops them and runs the whole suite. A primary language's bare test command
    (_bare_test_command) takes the targets as paths, and given none, the test files
    (_bare_test_file_list).

    Args:
        path_list: The test file paths passed to run_test. May be None or empty.
        style: config.TEST_TARGET_STYLE: "path" / "separator" / "none".
        test_path: config.TEST_PATH.
        ext: The language's extension, used in the note and, where command is None, for
            the default test command config.TEST_COMMAND is compared with
            (_with_root_target).
        command: The test command to build on. None takes the primary language's bare
            test command, or else config.TEST_COMMAND with the primary language's root
            target and with_uv_project_word() of PROJECT_DIR.

    Returns:
        (argv, targets, note). Each target is written as path_arg() gives it. note
        holds one line only when "none" made the paths be thrown away.
    """
    if command is None:
        bare = _bare_test_command(ext)
        if bare:
            target_list = [path_arg(path) for path in path_list or []]
            return bare + (target_list or _bare_test_file_list(
                ext, gitcmd.work_tree_file_list(config.PROJECT_DIR) or [])), target_list, ""
        default_tuple = LANG_DICT[ext].test_command_tuple if ext in LANG_DICT else ()
        command = with_uv_project_word(
            _with_root_target(config.TEST_COMMAND, default_tuple), default_tuple,
            config.PROJECT_DIR)
    if style == "path":
        target_list = [path_arg(path) for path in path_list or [test_path]]
        return command + target_list, target_list, ""
    if style == "separator":
        target_list = [path_arg(path) for path in path_list or []]
        return command + (["--"] + target_list if target_list else []), target_list, ""
    note = (f"note: {ext} cannot take paths, so they were dropped and the whole "
            f"suite ran\n" if path_list else "")
    return command, [], note


def _keep_verdict(verdict: bool | None) -> None:
    """Store how the tests stood, for last_test_ok().

    Args:
        verdict: True when they passed, False when they were red, None when the run
            never reached a verdict.
    """
    global _last_test_ok
    _last_test_ok = verdict


def _test_verdict(returncode: int, out: str, ext: str,
                  fail_list: list[str]) -> bool | None:
    """Say how the tests stood, or None when the run never reached a verdict.

    A non-zero exit code is red only when the output shows the tests ran and something
    in them was red - a report line of _REPORT_RE_DICT, or a failing test the reporter
    named. With neither, the answer is None.

    A zero exit code is not a pass when the output is read as a run over no tests
    (_is_no_test_text), which is None, and when the run's summary counts a failing test
    (_FAIL_COUNT_RE_DICT), which is red. A run that prints a line of _FAIL_EXIT_RE_DICT
    is read as one with a non-zero exit code.

    Args:
        returncode: The test command's exit code.
        out: What the command printed.
        ext: The language the output is read as, used to look the report line up in
            _REPORT_RE_DICT, the failing count up in _FAIL_COUNT_RE_DICT, the line of a
            failing build up in _FAIL_EXIT_RE_DICT and to tell a run over no tests
            (_is_no_test_text).
        fail_list: The failing test names run_test scraped with _FAIL_RE_DICT.

    Returns:
        True when everything passed, False when the run reported red tests, None when
        nothing in the output says the tests ran.
    """
    fail_exit_re = _FAIL_EXIT_RE_DICT.get(ext)
    if returncode == 0 and not (fail_exit_re is not None and fail_exit_re.search(out)):
        fail_count_re = _FAIL_COUNT_RE_DICT.get(ext)
        if fail_count_re is not None and fail_count_re.search(out):
            return False
        return None if _is_no_test_text(out, ext) else True
    report_re = _REPORT_RE_DICT.get(ext)
    return False if (report_re is not None and report_re.search(out)) or fail_list else None


def _test_fail_dict(output: str, target_list: list[str] | None = None) -> dict:
    """Build run_test's return value for when the tests could not be run at all.

    Args:
        output: The single line saying what to fix and how.
        target_list: What was actually passed to the command, or None when the failure
            happened before argv was built.

    Returns:
        The 5-key dict documented in run_test's Returns, with status "did not run".
    """
    # A call that ran no test clears what the last run said
    _keep_verdict(None)
    return _no_run_dict(output, target_list)


def _no_run_dict(output: str, target_list: list[str] | None = None) -> dict:
    """Build the return value of one test command that could not be run at all.

    Args:
        output: The single line saying why.
        target_list: What was passed to the command, or None.

    Returns:
        The 5-key dict documented in run_test's Returns, with status "did not run".
    """
    return {"status": TEST_STATUS_DICT[None], "failed": [], "failed_excerpt": output,
            "targets": target_list or [], "output": output}


# cargo's line of what it is doing ("   Compiling calc v0.1.0 (...)", "    Finished
# `test` profile ...", "     Running tests/basic.rs (...)"); libtest's lines of a target
# that ran: "running 2 tests", "test tests::adds ... ok", "test x ... ignored", "test
# result: ok. ..." and the progress of its terse form with no failure in it ("....",
# ". 1/2"); and rustdoc's "all doctests ran in 0.14s; merged doctests compilation took
# 0.13s"
_CARGO_QUIET_LINE_RE = re.compile(
    r"^\s+(?:Compiling|Checking|Building|Finished|Running|Doc-tests|Downloading"
    r"|Downloaded|Locking|Adding|Updating|Blocking|Fresh|Removing)\b"
    r"|^running \d+ tests?$|^test .+ \.\.\. (?:ok|ignored)(?:, .*)?$|^test result: ok\. "
    r"|^\.+(?: \d+/\d+)?$|^all doctests ran in \d")

# libtest's head of what one failing test printed: "---- tests::adds stdout ----"
_CARGO_TEST_OUTPUT_RE = re.compile(r"---- .+ std(?:out|err) ----$")

# A line under a rustc warning: " --> src/lib.rs:3:9", "  |", "3 |     let x = 1;",
# "...  |" (the lines of a long snippet left out), "  = note: ...", "help: ..." and
# "note: ..."
_WARNING_BODY_RE = re.compile(r"\s|\d|\.\.\.|(?:help|note): ")


def _cargo_excerpt_line_list(line_list: list[str]) -> list[str]:
    """Return the lines of a cargo run that say something of what failed.

    Lines of _CARGO_QUIET_LINE_RE are left out, each warning ("warning: ..." and the
    lines after it that start with a blank, a digit, "...", "help:" or "note:", as
    rustc's lines under a warning do) and a blank line after another. What a test
    printed (from its "---- name stdout ----" line to the next "failures:" or "test
    result:" line) is kept whole.

    Args:
        line_list: The lines of the output.

    Returns:
        The lines kept, in order.
    """
    keep_list = []
    is_warning = is_test_output = False
    for line in line_list:
        if _CARGO_TEST_OUTPUT_RE.match(line):
            is_test_output = True
        elif line.startswith(("failures:", "test result:")):
            is_test_output = False
        if is_test_output:
            keep_list.append(line)
            continue
        if line.startswith("warning: "):
            is_warning = True
            continue
        if is_warning and line.strip() and _WARNING_BODY_RE.match(line):
            continue
        is_warning = False
        if _CARGO_QUIET_LINE_RE.search(line) or (
                not line.strip() and (not keep_list or not keep_list[-1].strip())):
            continue
        keep_list.append(line)
    return keep_list


# A line of automake's make check summary: its rules, its title, its counts and the
# lines after them ("See ./test-suite.log" and the three lines of "Some test(s) failed.
# Please report this to ...")
_AUTOMAKE_SUMMARY_LINE_RE = re.compile(
    r"^=+$|^Testsuite summary|^# (?:TOTAL|PASS|SKIP|XFAIL|FAIL|XPASS|ERROR): +\d+$"
    r"|^See \S*test-suite\.log|^Some test\(s\) failed\.|^Please report "
    r"|^together with the test-suite\.log file|^information\.  Thanks\.$")

# A line make prints when a step failed ("make[2]: *** [Makefile:9: calc.o] Error 1") or
# when it enters or leaves a directory
_MAKE_NOISE_LINE_RE = re.compile(r"^make(?:\[\d+\])?: (?:\*\*\* \[[^\]\n]+\] Error \d+$"
                                 r"|(?:Entering|Leaving) directory )")


def _c_excerpt_line_list(line_list: list[str]) -> list[str]:
    """Return the lines of a C / C++ test run's output its failure text is cut from.

    Args:
        line_list: The lines of the output.

    Returns:
        Where automake's summary is there (_FAIL_IF_RE_DICT), the lines without those of
        its summaries (_AUTOMAKE_SUMMARY_LINE_RE) and make's own (_MAKE_NOISE_LINE_RE).
        Otherwise the lines from the first that names a failing test (_fail_name) on, or
        every line when none does.
    """
    if not any(_FAIL_IF_RE_DICT["c"][0].match(line) for line in line_list):
        start = next((index for index, line in enumerate(line_list)
                      if _fail_name(line, "c") is not None), 0)
        return line_list[start:]
    return [line for line in line_list if not _AUTOMAKE_SUMMARY_LINE_RE.match(line)
            and not _MAKE_NOISE_LINE_RE.match(line)]


# The line bun test starts its second listing of the failing tests with
_BUN_REPEAT_HEAD_RE = re.compile(_BUN_REPEAT_HEAD_PATTERN)


def _bun_start(line_list: list[str], start: int) -> int:
    """Return where the report of the failing test bun test names at a line starts.

    bun prints a failing test's report (its source, the expected and received values, the
    stack) above its "(fail)" line.

    Args:
        line_list: The lines of bun test's output, without those of _OMIT_LINE_RE_DICT.
        start: The index of a line that names a failing test.

    Returns:
        For a "(fail)" line, the index after the "(fail)" line before it, or of the
        heading of its file (_BUN_FILE_HEAD_RE), which is kept, or after bun's first
        line; 0 when there is none of them. start for any other line.
    """
    if not line_list[start].startswith("(fail) "):
        return start
    for index in range(start - 1, -1, -1):
        line = line_list[index]
        if _BUN_FILE_HEAD_RE.match(line):
            return index
        if line.startswith("(fail) ") or _BUN_HEAD_RE.match(line):
            return index + 1
    return 0


def _js_excerpt_line_list(line_list: list[str]) -> list[str]:
    """Return the lines of a JavaScript or TypeScript test run its failure text is cut from.

    Args:
        line_list: The lines of the output, without those of _OMIT_LINE_RE_DICT.

    Returns:
        The lines from the first that names a failing test (_fail_pair_list) on, or every
        line when none does. In bun test's output (_BUN_HEAD_RE), from where that test's
        report starts (_bun_start) on, without a heading of a file that has no report
        under it (_is_bun_body_after).
    """
    pair_list = _fail_pair_list("\n".join(line_list), "js")
    if not pair_list:
        return line_list
    start = pair_list[0][0]
    if not any(_BUN_HEAD_RE.match(line) for line in line_list):
        return line_list[start:]
    keep_list = line_list[_bun_start(line_list, start):]
    return [line for index, line in enumerate(keep_list)
            if not _BUN_FILE_HEAD_RE.match(line) or _is_bun_body_after(keep_list, index)]


def _is_bun_body_after(line_list: list[str], index: int) -> bool:
    """Whether a heading of a file in bun test's output has a report of its own under it.

    Args:
        line_list: The lines kept of bun test's output.
        index: The index of the heading.

    Returns:
        True when the next line that is not blank is neither another heading
        (_BUN_FILE_HEAD_RE), nor the start of bun's second listing of the failing tests
        (_BUN_REPEAT_HEAD_RE), nor one of its count lines (_BUN_COUNT_LINE_RE).
    """
    next_line = next((line for line in line_list[index + 1:] if line.strip()), None)
    return next_line is not None and not _BUN_FILE_HEAD_RE.match(next_line) \
        and not _BUN_REPEAT_HEAD_RE.match(next_line) and not _BUN_COUNT_LINE_RE.match(next_line)


# What is left out of a test command's output before its failure text is cut to its end,
# one function per language: (the lines of the output) -> the lines kept.
# An extension that is not here keeps every line
_EXCERPT_LINE_DICT = _with_alias_dict({"rs": _cargo_excerpt_line_list,
                                       "c": _c_excerpt_line_list,
                                       "js": _js_excerpt_line_list})


def _fail_excerpt(out: str, ext: str) -> str:
    """Extract just the failure text, briefly, from the test output.

    Args:
        out: What the test command printed, without the lines of _OMIT_LINE_RE_DICT.
        ext: The language the output is read as, used to look the failure line pattern
            up in _FAIL_RE_DICT and the lines left out up in _EXCERPT_LINE_DICT.

    Returns:
        Everything from the first failure to the end, cut to its last FAIL_EXCERPT_CHAR
        characters with the "... (N chars omitted)" marker in front. When no failure
        line is found, the same length from the end of the output. For a language of
        _EXCERPT_LINE_DICT, the lines it keeps of the whole output, cut the same way.
    """
    line_list = line_list_of(out)
    keep_line_of = _EXCERPT_LINE_DICT.get(ext)
    if keep_line_of is not None:
        return cut_tail("\n".join(keep_line_of(line_list)).strip(),
                        config.FAIL_EXCERPT_CHAR)
    start_list = [index for index, line in enumerate(line_list)
                  if _fail_name(line, ext) is not None]
    body = ("\n".join(line_list[start_list[0]:]) if start_list else out).strip()
    return cut_tail(body, config.FAIL_EXCERPT_CHAR)


def dep_file_pattern_set() -> frozenset[str]:
    """Return the base-name patterns of the dependency files of the test languages.

    The languages are the primary language and the ones other_test_list() returns.

    Returns:
        The dep_file_tuple entries of those languages' tables; empty when none of them
        has such a file. An extension that is not in the table adds nothing.
    """
    ext_list = [config.PRIMARY_EXT] + [ext for ext, _, _ in other_test_list()]
    return frozenset(pattern for ext in ext_list if ext in LANG_DICT
                     for pattern in LANG_DICT[ext].dep_file_tuple)


def test_build_file_pattern_set() -> frozenset[str]:
    """Return the base-name patterns of the files the stage that writes tests may write.

    Returns:
        dep_file_pattern_set() and the primary language's test_build_file_tuple.
    """
    lang_config = LANG_DICT.get(config.PRIMARY_EXT)
    return dep_file_pattern_set() | frozenset(
        lang_config.test_build_file_tuple if lang_config is not None else ())


def change_dep_path_list(pattern_set: frozenset[str] | None = None) -> list[str]:
    """Return the paths of the dependency files this run has changed.

    The patterns are matched against a changed path's base name: a dependency file at
    any depth is found. A language with no such file adds none.

    Examples:
        a change to "backend/pyproject.toml" -> ["backend/pyproject.toml"]

    Args:
        pattern_set: The base-name patterns. None is dep_file_pattern_set().

    Returns:
        The dependency files among the changed and added paths, sorted.
    """
    if pattern_set is None:
        pattern_set = dep_file_pattern_set()
    if not pattern_set:
        return []
    by_kind_dict = change_by_kind_dict()
    return sorted(path for path in by_kind_dict["added"] + by_kind_dict["modified"]
                  if is_dep_file(path, pattern_set))


def refill_cache_on_dep_change() -> str:
    """Have the package cache filled again when a dependency file was changed.

    The broker is asked to fill the cache. Nothing happens when the call is being made
    in this process (config.LLM_BROKER empty). The broker is what tells whether the
    cache already holds what the dependency files fetch, and starts no filling then.

    A refill call that fails (broker.refill() returns "ok" False) is logged and said in
    the returned line.

    Returns:
        One line saying whether the cache was filled again and for which dependency
        files, or "" when nothing was asked or the broker started no filling.
    """
    if not config.LLM_BROKER:
        return ""
    path_list = change_dep_path_list()
    if not path_list:
        return ""
    name_text = ", ".join(path_list)
    logger.info("%s changed, so the package cache is asked for again", name_text)
    result_dict = broker.refill()
    if not result_dict["ok"]:
        logger.info("the tests run against the cache as it is: %s", result_dict["error"])
        return (f"note: the package cache was not filled again for {name_text} "
                f"({result_dict['error']}); the tests run against the cache as it is\n")
    if result_dict["skip"]:
        return ""
    return (f"note: the package cache was filled again for {name_text}, "
            f"{result_dict['refills_left']} refill(s) left this run\n")


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill every process in the group a command leads, then wait for the command.

    A platform without process groups kills the command alone.

    Args:
        proc: A command run_in_project() started, the leader of a process group of its
            own.
    """
    if hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    else:
        proc.kill()
    proc.wait()


class StartError(Exception):
    """A command run_in_project() could not start. Its text is the system's reason."""


# How many of the first bytes, and how many of the last, a command run_in_project()
# started wrote to standard output are returned; the same of standard error
_OUTPUT_HEAD_BYTE = 8 << 20
_OUTPUT_TAIL_BYTE = 8 << 20

# How many bytes of the lines between the two that run_in_project()'s keep_line keeps
# are returned of standard output; the same of standard error
_OUTPUT_LINE_BYTE = 8 << 20


# What a pom.xml says the Java version is: <maven.compiler.release>8</...>,
# <source>1.7</source>, <testRelease>21</testRelease>, <java.version>1.8</java.version>
# and the like, with the value as the group (a number or "${a.property}")
_POM_TARGET_RE = re.compile(
    r"<((?:maven\.compiler\.)?(?:release|source|target|testRelease|testSource|testTarget)"
    r"|java\.version)>\s*([^<\s]+)\s*</\1>")

# An element of a pom.xml holding one word, which may be a property: <name>value</name>
_POM_WORD_RE = re.compile(r"<([\w.-]+)>\s*([^<\s]+)\s*</\1>")

# A module of a pom.xml: <module>core</module> / <subproject>core</subproject>
_POM_MODULE_RE = re.compile(r"<(module|subproject)>\s*([^<]+?)\s*</\1>")

# A comment of a pom.xml
_POM_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)

# A Java version as a pom.xml writes it, its number as the group
_JAVA_VERSION_RE = re.compile(r"(?:1\.)?(\d+)")

# A <profile> of a pom.xml, and the JDK its <activation> names in <jdk>
_POM_PROFILE_RE = re.compile(r"<profile>(.*?)</profile>", re.S)
_POM_JDK_RE = re.compile(r"<activation>.*?<jdk>\s*([^<]*?)\s*</jdk>.*?</activation>", re.S)

# One range of a <jdk> value: "[17,)", "(,11]", "[1.8,11)"
_JDK_RANGE_RE = re.compile(r"([\[(])\s*([\d.]*)\s*,\s*([\d.]*)\s*([\])])")

# A reference to a property, its name as the group
_POM_PROPERTY_REF_RE = re.compile(r"\$\{([\w.-]+)\}")

# A reference to a property maven itself sets to a place or a name, never to a Java
# version (${project.build.directory}, ${basedir})
_POM_MAVEN_REF_RE = re.compile(r"\$\{(?:project|pom|user)\.|\$\{maven\.(?!compiler\.)"
                               r"|\$\{(?:basedir|java\.home)\}")

# A text that can come to a Java version once its properties are replaced: digits and
# dots around the references
_POM_VERSION_SHAPE_RE = re.compile(r"[\d.]*")

# The JDK the Java image holds beside its default one, its version (the newest Java
# version it compiles for), and the version at or below which a project is built with it
_OLD_JDK_DIR = "/opt/ctloom-jdk11"
_OLD_JDK_TOP_TARGET = 11
_OLD_JDK_USE_TARGET = 8

# What maven or a plugin prints when the JDK it runs on is older than it needs: a class
# built for a newer Java (as the JVM, javac and ASM say it), the enforcer's Java version
# rule, a tool that requires a newer Java, and javac asked for a release the JDK does not
# know
_OLD_JDK_REFUSE_RE = re.compile(
    r"UnsupportedClassVersionError|compiled by a more recent version of the Java Runtime"
    r"|Detected JDK [Vv]ersion:? \S+ .*is not in the allowed range"
    r"|class file has wrong version (?:5[6-9]|[6-9]\d)\.0"
    r"|Unsupported class file major version (?:5[6-9]|[6-9]\d)"
    r"|\brequires (?:Java|JDK) (?:version )?(?:1[2-9]|[2-9]\d)\b"
    r"|invalid target release: (?:1[2-9]|[2-9]\d)\b"
    r"|release version (?:1[2-9]|[2-9]\d) not supported")

# The directories, relative to the repository root, where a maven build refused the old
# JDK (_OLD_JDK_REFUSE_RE): maven runs there with the default JDK from then on
_old_jdk_refuse_dir_set: set[str] = set()


@functools.cache
def _default_jdk_version() -> int | None:
    """Return the Java version of the JDK the javac on PATH belongs to, or None.

    It is read off the JAVA_VERSION line of the release file in the JDK's directory
    (the one above the bin directory javac resolves to).
    """
    javac = shutil.which("javac")
    if javac is None:
        return None
    jdk_dir = os.path.dirname(os.path.dirname(os.path.realpath(javac)))
    try:
        with open(os.path.join(jdk_dir, "release"), encoding="utf-8",
                  errors="replace") as file:
            version = re.search(r'^JAVA_VERSION="(?:1\.)?(\d+)', file.read(), re.M)
    except OSError:
        return None
    return int(version.group(1)) if version else None


# How many pom.xml files of one build are read at most, and how many texts one value may
# come to with its properties replaced
_POM_MAX_FILE = 2000
_POM_MAX_EXPANSION = 64


def _reactor_pom_text_list(start_dir: str, jdk_version: int) -> list[str]:
    """Return the pom.xml files maven builds from a directory with a JDK of one version.

    Args:
        start_dir: The directory maven starts in.
        jdk_version: The JDK's version: the profiles it does not activate are left out
            (_without_off_profile).

    Returns:
        The text of its pom.xml and of those of the modules it names (<module>,
        <subproject>, in the profiles kept too), in turn, without their comments and the
        profiles left out, up to _POM_MAX_FILE files. A module naming a directory reads
        the pom.xml in it.
    """
    text_list, read_set = [], set()
    queue = collections.deque([os.path.join(start_dir, "pom.xml")])
    while queue and len(text_list) < _POM_MAX_FILE:
        path = os.path.realpath(queue.popleft())
        if path in read_set:
            continue
        read_set.add(path)
        try:
            with open(path, encoding="utf-8", errors="replace") as file:
                text = _without_off_profile(_POM_COMMENT_RE.sub("", file.read()),
                                            jdk_version)
        except OSError:
            continue
        text_list.append(text)
        for _, module in _POM_MODULE_RE.findall(text):
            module_path = os.path.join(os.path.dirname(path), module)
            queue.append(os.path.join(module_path, "pom.xml")
                         if os.path.isdir(module_path) else module_path)
    return text_list


def _pom_expand(value: str, word_dict: dict[str, set[str]],
                replace_set: frozenset[str] = frozenset()) -> set[str] | None:
    """Return what a pom.xml value comes to with every "${property}" in it replaced.

    Args:
        value: The value.
        word_dict: {a property: every value the pom.xml files of the build give it}.
        replace_set: The properties being replaced around this call.

    Returns:
        Each text the value comes to, one for each value of each property it names, up
        to _POM_MAX_EXPANSION of them, less those that cannot be a version (text besides
        digits and dots around the references, or a property maven sets to a place,
        _POM_MAVEN_REF_RE); None when a text that can be one names a property word_dict
        does not hold (one left to the environment, ${env.X}, among them), or one it is
        already inside.
    """
    if _POM_MAVEN_REF_RE.search(value) \
            or not _POM_VERSION_SHAPE_RE.fullmatch(_POM_PROPERTY_REF_RE.sub("", value)):
        return set()
    reference = _POM_PROPERTY_REF_RE.search(value)
    if reference is None:
        return {value}
    name = reference.group(1)
    if name not in word_dict or name in replace_set:
        return None
    text_set: set[str] = set()
    for property_value in word_dict[name]:
        inner_set = _pom_expand(property_value, word_dict, replace_set | {name})
        if inner_set is None:
            return None
        for inner in inner_set:
            rest_set = _pom_expand(
                value[:reference.start()] + inner + value[reference.end():], word_dict,
                replace_set)
            if rest_set is None:
                return None
            text_set |= rest_set
            if len(text_set) > _POM_MAX_EXPANSION:
                return None
    return text_set


def _jdk_version_of(text: str) -> int | None:
    """Return the Java version a <jdk> bound names ("1.8" -> 8, "17" -> 17), or None."""
    version = _JAVA_VERSION_RE.match(text)
    return int(version.group(1)) if version else None


def _jdk_activation_match(spec: str, version: int) -> bool | None:
    """Say whether a profile's <jdk> activation holds for a JDK of one version.

    Examples:
        ("[17,)", 11) -> False      ("1.8", 11) -> False      ("!1.8", 11) -> True
        ("(,11]", 11) -> True       ("[1.6,1.8),[11,)", 11) -> True

    Args:
        spec: The <jdk> value: a version the JDK's starts with, ranges, or either after
            "!".
        version: The JDK's version.

    Returns:
        Whether it holds, or None for a value read neither way.
    """
    if spec.startswith("!"):
        match = _jdk_activation_match(spec[1:].strip(), version)
        return None if match is None else not match
    range_list = _JDK_RANGE_RE.findall(spec)
    if not range_list:
        exact = _JAVA_VERSION_RE.fullmatch(spec)
        return None if exact is None else int(exact.group(1)) == version
    if _JDK_RANGE_RE.sub("", spec).replace(",", "").strip():
        return None
    for open_mark, low, high, close_mark in range_list:
        low_version, high_version = _jdk_version_of(low), _jdk_version_of(high)
        if (low and low_version is None) or (high and high_version is None):
            return None
        if (low_version is None or version > low_version
                or (open_mark == "[" and version == low_version)) \
                and (high_version is None or version < high_version
                     or (close_mark == "]" and version == high_version)):
            return True
    return False


def _without_off_profile(text: str, version: int) -> str:
    """Return a pom.xml's text less the profiles a JDK of one version does not activate.

    Args:
        text: The pom.xml's text.
        version: The JDK's version.

    Returns:
        The text with each profile whose <activation> names a <jdk> that does not hold
        for the version (_jdk_activation_match) taken out; one whose <jdk> cannot be read
        is kept.
    """
    def keep(match: re.Match) -> str:
        jdk = _POM_JDK_RE.search(match.group(1))
        return "" if jdk and _jdk_activation_match(jdk.group(1), version) is False \
            else match.group(0)
    return _POM_PROFILE_RE.sub(keep, text)


def _pom_java_version_set(text_list: list[str]) -> set[int] | None:
    """Return the Java versions the pom.xml files of a build say, or None for one unknown.

    Args:
        text_list: _reactor_pom_text_list()'s return value.

    Returns:
        The numbers of the versions they say (_POM_TARGET_RE), each "${property}" in a
        value replaced by every value those files give it (_POM_WORD_RE, _pom_expand). What
        cannot be a version (a directory another plugin names <source>, a property maven
        sets) is passed over; None when a value names a property that cannot be replaced.
    """
    word_dict = collections.defaultdict(set)
    for text in text_list:
        for name, value in _POM_WORD_RE.findall(text):
            word_dict[name].add(value)
    version_set: set[int] = set()
    for text in text_list:
        for _, value in _POM_TARGET_RE.findall(text):
            text_set = _pom_expand(value, word_dict)
            if text_set is None:
                return None
            for version_text in text_set:
                version = _JAVA_VERSION_RE.fullmatch(version_text)
                if version:
                    version_set.add(int(version.group(1)))
    return version_set


def _maven_env(argv: list[str], directory: str) -> dict[str, str]:
    """Return the JDK maven is run with when the project targets an old Java.

    Args:
        argv: The command about to be run.
        directory: Where it is run, relative to the repository root.

    Returns:
        {"JAVA_HOME": _OLD_JDK_DIR, "PATH": its bin before the PATH held} when the command
        is mvn or mvnw, _OLD_JDK_DIR is there, and the pom.xml files maven builds from the
        directory with that JDK (_reactor_pom_text_list) say Java versions, all of them
        known (_pom_java_version_set), of which the lowest is _OLD_JDK_USE_TARGET or older
        and the highest _OLD_JDK_TOP_TARGET or older; {} otherwise. {} as well for a
        directory where a build refused that JDK (_old_jdk_refuse_dir_set), and for a
        build whose lowest version is _OLD_JDK_USE_TARGET, which the default JDK builds
        too, when a profile the default JDK activates and that JDK does not names a
        <jdk>.
    """
    if not argv or os.path.basename(argv[0]) not in ("mvn", "mvnw") \
            or not os.path.isdir(_OLD_JDK_DIR) or directory in _old_jdk_refuse_dir_set:
        return {}
    start_dir = os.path.join(config.PROJECT_DIR, directory)
    version_set = _pom_java_version_set(_reactor_pom_text_list(start_dir,
                                                               _OLD_JDK_TOP_TARGET))
    if not version_set or min(version_set) > _OLD_JDK_USE_TARGET \
            or max(version_set) > _OLD_JDK_TOP_TARGET:
        return {}
    default_version = _default_jdk_version()
    if min(version_set) == _OLD_JDK_USE_TARGET and default_version is not None and any(
            _jdk_activation_match(spec, _OLD_JDK_TOP_TARGET) is False
            and _jdk_activation_match(spec, default_version)
            for text in _reactor_pom_text_list(start_dir, default_version)
            for profile in _POM_PROFILE_RE.findall(text)
            for spec in _POM_JDK_RE.findall(profile)):
        return {}
    env_path = config.project_env().get("PATH", "")
    jdk_bin = os.path.join(_OLD_JDK_DIR, "bin")
    return {"JAVA_HOME": _OLD_JDK_DIR,
            "PATH": jdk_bin + (f":{env_path}" if env_path else "")}


def run_in_project(argv: list[str], timeout_sec: int,
                   extra_env_dict: dict[str, str] | None = None,
                   directory: str = "", is_machine_run: bool = True,
                   keep_line: Callable[[str], int] | None = None
                   ) -> subprocess.CompletedProcess:
    """Run a command from the repository root within a time limit and capture it.

    A maven command is run with the JDK _maven_env() picks, and once more with the
    default JDK, within what is left of timeout_sec, when it fails and prints that the
    JDK is too old (_OLD_JDK_REFUSE_RE).

    What a command run with is_machine_run True wrote in the repository is recorded as
    the command's, however the run ends (_command_write_record). The command leads a
    process group of its own, and the wait is for the command itself: once it ends, the
    processes it started that are still in the group are killed, whether or not they
    hold its output open, and what was printed until then is returned. When it runs past
    the limit, or the wait for it ends with any other exception (an interrupt included),
    the whole group is killed before the exception goes on. A process that left the
    group for a session of its own is not killed.

    Standard output and standard error are each a pipe read while the command runs
    (progress.PipeOutput). Of each, the first _OUTPUT_HEAD_BYTE bytes and the last
    _OUTPUT_TAIL_BYTE are returned, with textcut.CUT_NOTE after the first when more
    was written. Of the lines between the two, the ones keep_line keeps are returned
    in their place, up to _OUTPUT_LINE_BYTE; the first bytes then end and the last
    start at a line end. A process that outlives the command with a pipe still open
    is read for progress.PIPE_END_WAIT_SEC more, and the pipe is closed then.

    Args:
        argv: The command and its arguments.
        timeout_sec: How many seconds the command may run.
        extra_env_dict: Values added to config.project_env(). None adds none.
        directory: The directory under the repository root the command is started from,
            relative to it. Empty starts it from the root.
        is_machine_run: False leaves what the command writes unrecorded as a command's:
            it counts as the request's own change.
        keep_line: Called with each line that lies between the first bytes returned and
            the last, without its line end. It returns how many lines are kept from that
            line on: 0 for none, 1 for the line alone. None keeps none.

    Returns:
        The finished process, its stdout and stderr read as text (config.CHILD_TEXT_DICT),
        every line end as "\\n".

    Raises:
        StartError: When the command cannot be started: argv[0] is not there, is not
            executable, is not a program the system can run, or an argument holds a
            character no argument can (NUL, half of a surrogate pair).
        subprocess.TimeoutExpired: When the command ran past timeout_sec.
    """
    old_jdk_dict = _maven_env(argv, directory)
    start_time = time.monotonic()
    proc = _run_with_env(argv, timeout_sec, {**old_jdk_dict, **(extra_env_dict or {})},
                         directory, is_machine_run, keep_line)
    if old_jdk_dict and proc.returncode != 0 \
            and _OLD_JDK_REFUSE_RE.search(proc.stdout + proc.stderr):
        logger.info("the build refused %s, so it runs again with the default JDK, as "
                    "maven does in %s from now on", old_jdk_dict["JAVA_HOME"],
                    directory or ".")
        _old_jdk_refuse_dir_set.add(directory)
        left_sec = timeout_sec - (time.monotonic() - start_time)
        if left_sec <= 0:
            raise subprocess.TimeoutExpired(argv, timeout_sec)
        proc = _run_with_env(argv, left_sec, extra_env_dict or {}, directory,
                             is_machine_run, keep_line)
    return proc


def _run_with_env(argv: list[str], timeout_sec: float, extra_env_dict: dict[str, str],
                  directory: str, is_machine_run: bool,
                  keep_line: Callable[[str], int] | None) -> subprocess.CompletedProcess:
    """Run a command as run_in_project() does, with values added to its environment.

    Args:
        argv: The command and its arguments.
        timeout_sec: How many seconds the command may run.
        extra_env_dict: Values added to config.project_env().
        directory: The directory under the repository root the command is started from.
        is_machine_run: Whether what the command writes is recorded as the command's.
        keep_line: What run_in_project() is given as keep_line.

    Returns:
        The finished process, as run_in_project() returns it.

    Raises:
        StartError: When the command cannot be started.
        subprocess.TimeoutExpired: When the command ran past timeout_sec.
    """
    record = _command_write_record() if is_machine_run else contextlib.nullcontext()
    encoding, errors = config.CHILD_TEXT_DICT["encoding"], config.CHILD_TEXT_DICT["errors"]
    keep_raw_line = None if keep_line is None else (
        lambda raw_line: keep_line(raw_line.decode(encoding, errors)))
    with progress.PipeOutput(2, _OUTPUT_HEAD_BYTE, _OUTPUT_TAIL_BYTE, keep_raw_line,
                             _OUTPUT_LINE_BYTE) as output, record:
        try:
            proc = subprocess.Popen(argv, cwd=os.path.join(config.PROJECT_DIR, directory),
                                    stdout=output.write_fd_list[0],
                                    stderr=output.write_fd_list[1],
                                    env=config.project_env(extra_env_dict),
                                    start_new_session=True)
        except (OSError, ValueError) as e:
            raise StartError(getattr(e, "strerror", None) or str(e)) from e
        try:
            proc.wait(timeout=timeout_sec)
        finally:
            _kill_group(proc)
    stdout, stderr = (
        textfile.lf_text(output.text(index, encoding, errors)) for index in (0, 1))
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def run_test(path_list: list[str] | None = None) -> dict:
    """Run the primary language's test command, then those of the other languages.

    The other languages are the ones other_test_list() returns; a call that names no
    paths runs those _other_whole_list() returns. A call that names paths gives the
    command of an entry of other_test_list() the paths under the entry's directory whose
    language has that test command, and the primary language's command every other path,
    as _path_by_language() splits them.

    When a patch landed just before, a call whose command actually ran closes one
    "edit then test" round (MAX_PATCH_ROUND). A command that could not be started at all
    closes none. That count is not in the return value; read "rounds" from
    replace_in_file or call patch_round_count() instead. keep_test_state() puts the
    round count and whether a patch landed back as they were before its block.

    Args:
        path_list: The test file paths to run, relative to PROJECT_DIR. What omitting them
            does depends on the primary language's TEST_TARGET_STYLE.
            "path" ... the whole TEST_PATH is the target (py)
            "separator" ... nothing is added; the build tool's default applies
            "none" ... paths cannot be passed; they are dropped and everything runs

    Returns:
        {"status": str,       # "passed", "failed" or "did not run"; see _test_verdict
         "failed": list[str], # names of the failing tests; empty unless status is
                              # "failed", and empty then too when they were unreadable
         "failed_excerpt": str,  # an excerpt of the failure, or of why the run reached
                              # no verdict; "" when everything passed
         "targets": list[str],# what was passed to the command
         "output": str}       # the last TEST_OUTPUT_TAIL characters of stdout+stderr
                              # without the lines naming each passing test, after any
                              # note line. For a command that could not be started or
                              # ran past TEST_TIMEOUT_SEC, and for a call refused before
                              # a command was built, the line saying why instead
    """
    return _run_test(path_list, refill=True)


def run_test_as_is(path_list: list[str] | None = None) -> dict:
    """Run the test command as run_test() does, against the package cache as it is.

    Args:
        path_list: As run_test() takes it.

    Returns:
        run_test()'s return value. The broker is not asked to fill the package cache
        again, whatever dependency files the working tree changes.
    """
    return _run_test(path_list, refill=False)


def other_test_list(file_list: list[str] | None = None
                    ) -> list[tuple[str, list[str], str]]:
    """Return the tests of the languages besides the primary one that can be run now.

    A language of config.OTHER_TEST_LIST is taken when RUN_TEST is on, its command is on
    PATH and the repository holds one of its dependency files (LangConfig
    dep_file_tuple) and a directory to run its tests from (_other_start_dir), which is
    where they are run.

    Args:
        file_list: gitcmd.work_tree_file_list() of PROJECT_DIR. None lists the files
            here.

    Returns:
        [(the language's extension, its test command with with_uv_project_word() and
        the root target of that directory, the directory relative to the repository root,
        "" for the root), ...].
    """
    if not config.RUN_TEST or not config.OTHER_TEST_LIST:
        return []
    if file_list is None:
        file_list = gitcmd.work_tree_file_list(config.PROJECT_DIR) or []
    run_list = []
    for ext, argv in config.OTHER_TEST_LIST:
        if shutil.which(argv[0]) is None:
            logger.debug("%s is not on PATH, so the tests of .%s files do not run",
                         argv[0], ext)
            continue
        directory = _other_start_dir(ext, file_list)
        if directory is None:
            continue
        start = os.path.join(config.PROJECT_DIR, directory)
        target = _root_target(start, ext)
        command = with_uv_project_word(argv, LANG_DICT[ext].test_command_tuple, start)
        run_list.append((ext, command + ([target] if target else []), directory))
    return run_list


def _file_dir_list(file_list: list[str], pattern_list: list[str], root: str) -> list[str]:
    """Return the directories holding a file whose name matches one of some patterns.

    Args:
        file_list: Files relative to root, "/" between their parts.
        pattern_list: fnmatch patterns a file's name is matched against.
        root: The directory the files are in, as an absolute path. A file that is
            not there any more is not counted.

    Returns:
        The directories relative to root, "" for root itself, each once, nearest the
        root first and by name among those as near. One whose name holds a line break
        is passed over.
    """
    directory_set = set()
    for path in file_list:
        name = posixpath.basename(path)
        if "\n" in path or "\r" in path or not any(
                fnmatch.fnmatchcase(name, pattern) for pattern in pattern_list):
            continue
        if os.path.isfile(os.path.join(root, path)):
            directory_set.add(posixpath.dirname(path))
    return sorted(directory_set, key=lambda directory: (
        directory.count("/") + bool(directory), directory))


def _project_dir_list(ext: str, file_list: list[str], root: str) -> list[str]:
    """Return the directories one language's tools write beside its project files in.

    Args:
        ext: The language's LANG_DICT key.
        file_list: Files relative to root.
        root: The directory the files are in, as an absolute path.

    Returns:
        The directories holding one of the language's dependency files and, for a
        language with a root target, one its root_target_tuple patterns match as well
        ("*.csproj" and not "NuGet.Config"), as _file_dir_list() returns them.
    """
    dep_pattern_list = list(LANG_DICT[ext].dep_file_tuple)
    target_pattern_list = list(root_target_pattern_tuple(LANG_DICT[ext]))
    if not target_pattern_list:
        return _file_dir_list(file_list, dep_pattern_list, root)
    keep_list = [path for path in file_list if any(
        fnmatch.fnmatchcase(posixpath.basename(path), pattern)
        for pattern in target_pattern_list)]
    return _file_dir_list(keep_list, dep_pattern_list, root)


def _source_dir_list(ext: str, directory_list: list[str],
                     file_list: list[str]) -> list[str]:
    """Return the directories that hold a source of a language's test command at any depth.

    Args:
        ext: The language's LANG_DICT key. Its sources are the files of every language
            whose test command is the same (C and C++, JavaScript and TypeScript).
        directory_list: Directories relative to the repository root, "" for the root.
        file_list: The repository's files, relative to its root.

    Returns:
        The directories of directory_list that do, in their order: "" while the
        repository holds such a source at all.
    """
    command_tuple = LANG_DICT[ext].test_command_tuple
    holder_set = set()
    for path in file_list:
        lang_config = lang_of(path)
        if lang_config is None or lang_config.test_command_tuple != command_tuple:
            continue
        directory = posixpath.dirname(path)
        while directory not in holder_set:
            holder_set.add(directory)
            if not directory:
                break
            directory = posixpath.dirname(directory)
    return [directory for directory in directory_list if directory in holder_set]


def _other_start_dir(ext: str, file_list: list[str],
                     project_dir: str | None = None) -> str | None:
    """Return the directory another language's tests are run from.

    Only a directory that holds a source of the language at any depth is taken
    (_source_dir_list): a dependency file among test data is passed over.

    Args:
        ext: The language's LANG_DICT key.
        file_list: The repository's files, relative to its root.
        project_dir: The repository, as an absolute path. None is config.PROJECT_DIR.

    Returns:
        The directory nearest the repository root that holds one of the language's
        dependency files, "" for the root. For a language with a root target it is the
        nearest directory holding a file its root_target_tuple patterns match in which
        the target is one file (_root_target). None when the repository holds no
        dependency file of the language, or no such directory.
    """
    root = config.PROJECT_DIR if project_dir is None else project_dir
    lang_config = LANG_DICT[ext]
    dep_dir_list = _source_dir_list(
        ext, _file_dir_list(file_list, list(lang_config.dep_file_tuple), root), file_list)
    if not dep_dir_list:
        return None
    if not lang_config.root_target_tuple:
        return dep_dir_list[0]
    for directory in _source_dir_list(
            ext, _file_dir_list(file_list, list(root_target_pattern_tuple(lang_config)),
                                root), file_list):
        if _root_target(os.path.join(root, directory), ext):
            return directory
    return None


def _has_other_test_path(ext: str, directory: str, project_dir: str | None = None) -> bool:
    """Say whether another language's tests are there for a call that names no paths.

    Args:
        ext: The language's LANG_DICT key.
        directory: Where its tests are run from, relative to the repository root.
        project_dir: The repository, as an absolute path. None is config.PROJECT_DIR.

    Returns:
        False for a language whose tests are reached by path (LangConfig test_target)
        when its test directory (config.test_dir_of) is not under directory, else True.
    """
    start = os.path.join(config.PROJECT_DIR if project_dir is None else project_dir,
                         directory)
    return LANG_DICT[ext].test_target != "path" or os.path.isdir(
        os.path.join(start, config.test_dir_of(ext, start)))


def other_test_ext_list(ext_list: list[str], file_list: list[str],
                        project_dir: str) -> list[str]:
    """Return the languages whose tests a repository holds what it takes to run.

    A language is kept when the repository holds one of its dependency files, a
    directory to run from (_other_start_dir), and its tests there where they are reached
    by path (_has_other_test_path). Whether its command is on PATH is not looked at: the
    answer is the same on a machine that does not have the language's tools.

    Args:
        ext_list: The languages to look at, as LANG_DICT keys: those of
            config.other_test_list_of().
        file_list: The repository's files, relative to its root.
        project_dir: The repository, as an absolute path.

    Returns:
        The languages among ext_list, in their order.
    """
    keep_list = []
    for ext in ext_list:
        directory = _other_start_dir(ext, file_list, project_dir)
        if directory is not None and _has_other_test_path(ext, directory, project_dir):
            keep_list.append(ext)
    return keep_list


def _other_whole_list() -> list[tuple[str, list[str], str]]:
    """Return the entries of other_test_list() a run_test() call naming no paths runs."""
    return [(ext, argv, directory) for ext, argv, directory in other_test_list()
            if _has_other_test_path(ext, directory)]


def test_command_text() -> str:
    """Return the command lines run_test() runs when it is given no paths, as one text.

    Examples:
        the primary language alone -> "uv run pytest -q"
        with another language      -> "uv run pytest -q / npm test --silent (in web)"

    Returns:
        primary_test_command() alone while _other_whole_list() is empty. Otherwise
        primary_test_command(), when the primary language's tests can be run
        (_no_run_reason), followed by the
        command of every entry of _other_whole_list() with the directory it is started
        from, joined by " / ".
    """
    other_list = _other_whole_list()
    primary = " ".join(primary_test_command())
    if not other_list:
        return primary
    text_list = [primary] if _no_run_reason(None) is None else []
    text_list += [" ".join(argv) + (f" (in {directory})" if directory else "")
                  for _, argv, directory in other_list]
    return " / ".join(text_list)


# The characters a .gitignore pattern reads as more than themselves
_PATTERN_SPECIAL_RE = re.compile(r"([\\*?\[])")


def _pattern_under(directory: str, pattern: str) -> str:
    """Return a .gitignore pattern that matches what another one does under a directory.

    Examples:
        ("web", "target/")    -> "/web/**/target/"   (at any depth under web)
        ("web", "uv.lock")    -> "/web/**/uv.lock"
        ("web", "/dist/")     -> "/web/dist/"        (in web itself)

    Args:
        directory: The directory, relative to the repository root, with "/" between
            its parts.
        pattern: A build_output_tuple entry.

    Returns:
        The pattern, read from the repository root.
    """
    head = "/" + _PATTERN_SPECIAL_RE.sub(r"\\\1", directory)
    if pattern.startswith("/"):
        return head + pattern
    return f"{head}/**/{pattern}"


def _top_dir_list(directory_list: list[str]) -> list[str]:
    """Return the directories among some that are under none of the others.

    Examples:
        ["src/app", "src/app/gen", "tests/app"] -> ["src/app", "tests/app"]
        ["", "src/app"]                         -> [""]

    Args:
        directory_list: Paths relative to the repository root, "/" between their
            parts, "" for the root itself.

    Returns:
        The directories, sorted, each once.
    """
    if "" in directory_list:
        return [""]
    top_list: list[str] = []
    for directory in sorted(set(directory_list)):
        if not any(directory.startswith(f"{top}/") for top in top_list):
            top_list.append(directory)
    return top_list


def output_place_list_of(ext_list: list[str], file_list: list[str],
                         root: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return where the tools of some languages write among the files of a directory.

    A language's tools write beside its project files (_project_dir_list).

    Args:
        ext_list: The languages' LANG_DICT keys.
        file_list: Files relative to root.
        root: The directory the files are in, as an absolute path.

    Returns:
        [(a directory relative to root, "" for root itself, entries of the language's
        build_output_tuple), ...], in the order of the languages: the entries not
        starting with "/" with each directory under no other one of the language, then
        those starting with "/" with each of the directories that is not inside one the
        language's entries name by name alone ("node_modules/"). A language with no
        build_output_tuple gives none.
    """
    place_list = []
    for ext in ext_list:
        output_tuple = LANG_DICT[ext].build_output_tuple
        dir_list = _project_dir_list(ext, file_list, root) if output_tuple else []
        near_tuple, place_tuple = split_output_tuple(output_tuple)
        if near_tuple:
            place_list += [(directory, near_tuple) for directory in _top_dir_list(dir_list)]
        if place_tuple:
            name_set = name_only_dir_set(near_tuple)
            place_list += [(directory, place_tuple) for directory in sorted(set(dir_list))
                           if not has_name_of(directory.split("/"), name_set)]
    return place_list


def output_pattern_list_of(place_list: list[tuple[str, tuple[str, ...]]]) -> list[str]:
    """Return the .gitignore patterns for what is written in some places.

    Args:
        place_list: output_place_list_of()'s return value.

    Returns:
        Each place's patterns under its directory (_pattern_under), and as they are,
        matching at any depth, where the directory is the root; each pattern once.
    """
    pattern_dict: dict[str, None] = {}
    for directory, output_tuple in place_list:
        pattern_dict.update(dict.fromkeys(
            _pattern_under(directory, pattern) if directory else pattern
            for pattern in output_tuple))
    return list(pattern_dict)


def other_output_place_list(file_list: list[str] | None = None
                            ) -> list[tuple[str, tuple[str, ...]]]:
    """Return where the tools of the languages besides the primary one write here.

    A language is taken when one of its commands runs here: its tests
    (other_test_list), or its static check (config.OTHER_CHECK_LIST) when the command
    is on PATH and the repository root holds one of its dependency files or its root
    target (_root_target).

    Args:
        file_list: gitcmd.work_tree_file_list() of PROJECT_DIR. None lists the files
            here.

    Returns:
        output_place_list_of() for those languages over the files of PROJECT_DIR;
        empty, without listing any file, when the index holds no other language.
    """
    if not config.OTHER_TEST_LIST and not config.OTHER_CHECK_LIST:
        return []
    root = config.PROJECT_DIR
    if file_list is None:
        file_list = gitcmd.work_tree_file_list(root) or []
    test_ext_set = {ext for ext, _, _ in other_test_list(file_list)}
    check_ext_set = {ext for ext, argv in config.OTHER_CHECK_LIST
                     if shutil.which(argv[0]) is not None}
    ext_list = []
    for ext in dict.fromkeys([ext for ext, _ in config.OTHER_TEST_LIST]
                             + [ext for ext, _ in config.OTHER_CHECK_LIST]):
        is_check_run = ext in check_ext_set and (
            bool(_root_target(root, ext)) or "" in _file_dir_list(
                file_list, list(LANG_DICT[ext].dep_file_tuple), root))
        if ext in test_ext_set or is_check_run:
            ext_list.append(ext)
    return output_place_list_of(ext_list, file_list, root)


def in_other_build_output(path: str,
                          place_list: list[tuple[str, tuple[str, ...]]]) -> bool:
    """Say whether a file lies in a directory another language's tools write.

    Args:
        path: The file, relative to the repository root, "/" between its parts.
        place_list: other_output_place_list()'s return value.

    Returns:
        True when one of the file's directories, under a directory place_list names,
        has the name a build_output_tuple entry gives by name alone, read as an fnmatch
        pattern ("obj/", "*.egg-info/"; "/dist/" and "uv.lock" are not asked).
    """
    part_list = path.split("/")[:-1]
    for directory, output_tuple in place_list:
        depth = len(directory.split("/")) if directory else 0
        if directory and part_list[:depth] != directory.split("/"):
            continue
        if has_name_of(part_list[depth:], name_only_dir_set(output_tuple)):
            return True
    return False


def _output_pattern_pair() -> tuple[list[str], list[str]]:
    """Return the patterns for what the tools of the languages here write.

    The work tree is listed once, when a pattern is to be placed.

    Returns:
        (the patterns: the primary language's build_output_tuple entries not starting
        with "/" as they are, matching at any depth, _place_pattern_list(), then
        output_pattern_list_of() of other_output_place_list()'s places of the entries not
        starting with "/", each pattern once; _place_pattern_list()).
    """
    near_tuple = split_output_tuple(LANG_DICT[config.PRIMARY_EXT].build_output_tuple)[0]
    file_list = _place_file_list()
    if file_list is None:
        return list(near_tuple), []
    place_pattern_list = _place_pattern_list(file_list)
    near_place_list = [(directory, output_tuple) for directory, output_tuple
                       in other_output_place_list(file_list)
                       if not output_tuple[0].startswith("/")]
    pattern_list = list(dict.fromkeys(
        [*near_tuple, *place_pattern_list, *output_pattern_list_of(near_place_list)]))
    return pattern_list, place_pattern_list


def _place_ext_list() -> list[str]:
    """Return the languages whose entries starting with "/" are placed here.

    Returns:
        The primary language, then each other language of config.OTHER_TEST_LIST and
        config.OTHER_CHECK_LIST, each once.
    """
    return list(dict.fromkeys([config.PRIMARY_EXT, *(ext for ext, _ in config.OTHER_TEST_LIST),
                               *(ext for ext, _ in config.OTHER_CHECK_LIST)]))


def _place_file_list() -> list[str] | None:
    """Return the files of PROJECT_DIR when a pattern of the languages here is placed.

    Returns:
        gitcmd.work_tree_file_list() of PROJECT_DIR when there is a PROJECT_DIR and an
        other language (_place_ext_list) or a primary language with an entry starting
        with "/"; None otherwise, without listing any file.
    """
    if not config.PROJECT_DIR or not (
            config.OTHER_TEST_LIST or config.OTHER_CHECK_LIST
            or split_output_tuple(LANG_DICT[config.PRIMARY_EXT].build_output_tuple)[1]):
        return None
    return gitcmd.work_tree_file_list(config.PROJECT_DIR) or []


def _place_pattern_list(file_list: list[str]) -> list[str]:
    """Return the patterns made from the entries starting with "/" of the languages here.

    Args:
        file_list: The files of PROJECT_DIR (_place_file_list).

    Returns:
        output_pattern_list_of() of the places output_place_list_of() gives those entries
        of _place_ext_list() over the files.
    """
    return output_pattern_list_of(
        [(directory, output_tuple) for directory, output_tuple
         in output_place_list_of(_place_ext_list(), file_list, config.PROJECT_DIR)
         if output_tuple[0].startswith("/")])


def build_output_path_list(path_list: list[str]) -> list[str]:
    """Return the files among some under the directories entries starting with "/" name.

    Args:
        path_list: Files relative to PROJECT_DIR, "/" between their parts.

    Returns:
        The files under a directory one of _place_pattern_list()'s patterns names
        (_pattern_dir), tracked or not, in the order given.
    """
    file_list = _place_file_list()
    if file_list is None:
        return []
    prefix_tuple = tuple(_pattern_dir(pattern) + "/" for pattern in _place_pattern_list(file_list))
    if not prefix_tuple:
        return []
    return [path for path in path_list if path.startswith(prefix_tuple)]


def exclude_build_output() -> list[str]:
    """Leave what the languages' tools write out of what this clone's git lists.

    The patterns (_output_pattern_pair) go into the exclude file of the clone's own git
    directory; what they match is not listed by git as new. A pattern made from an
    entry starting with "/" whose directory holds a file the index tracks is left out.

    Returns:
        The patterns that were added.
    """
    pattern_list, place_pattern_list = _output_pattern_pair()
    if place_pattern_list:
        track_dir_set = _track_dir_set()
        place_pattern_set = set(place_pattern_list)
        pattern_list = [pattern for pattern in pattern_list
                        if pattern not in place_pattern_set
                        or _pattern_dir(pattern) not in track_dir_set]
    if not pattern_list:
        return []
    try:
        add_list = gitcmd.add_exclude(config.PROJECT_DIR, pattern_list)
    except OSError as e:
        logger.warning("what the languages' tools write could not be left out of this "
                       "clone's git (%s), so it may reach the patch", e)
        return []
    if add_list:
        logger.info("what the languages' tools write is left out of this clone's git: %s",
                    ", ".join(add_list))
    return add_list


def _track_dir_set() -> set[str]:
    """Return the directories holding a file the index of PROJECT_DIR tracks.

    Returns:
        Every directory above a tracked file, relative to PROJECT_DIR, "/" between its
        parts.
    """
    dir_set = set()
    for path in gitcmd.run(config.PROJECT_DIR, "ls-files", "-z").stdout.split("\0"):
        part_list = path.split("/")[:-1]
        for depth in range(1, len(part_list) + 1):
            dir_set.add("/".join(part_list[:depth]))
    return dir_set


def _pattern_dir(pattern: str) -> str:
    """Return the directory a pattern made from an entry starting with "/" names.

    Examples:
        "/web/dist/" -> "web/dist"
        "/w\\[b]/dist/" -> "w[b]/dist"

    Args:
        pattern: The pattern, with the characters _pattern_under() escapes escaped.

    Returns:
        The directory, relative to PROJECT_DIR.
    """
    return re.sub(r"\\(.)", r"\1", pattern.strip("/"))


def _exclude_in_container() -> None:
    """Inside a container, leave what the languages' tools write out of git.

    The primary language is config.PRIMARY_EXT, and the places of the entries starting
    with "/" and those of the other languages are read off the repository as it stands
    now (exclude_build_output).
    """
    from codetwine_loom.sandbox import in_container
    if in_container():
        exclude_build_output()


def _test_target(path: object) -> tuple[str, str]:
    """Read one path run_test() was given as the file it names and what a command is given.

    The file is the part before the first "::". What follows it, a test's name inside
    the file, is kept as it was written. The file a command is given is the path
    without "." and ".." parts (to_repo_relative), its symbolic links left as they
    were written, and the file checked is the one that path leads to.

    Examples:
        "./tests/test_a.py::test_x[a//b]" -> ("tests/test_a.py",
                                              "tests/test_a.py::test_x[a//b]")
        "tests/link.py" (a link to ../shared/real.py) -> ("shared/real.py", "tests/link.py")
        "link/../t.py" (link a link to a/b)           -> ("t.py", "t.py")
        "../myapp/tests/test_a.py" (PROJECT_DIR myapp) -> ("tests/test_a.py",
                                                          "tests/test_a.py")

    Args:
        path: What was passed as one test path, in any notation to_repo_relative()
            reads.

    Returns:
        (the file's repository-relative path with its symbolic links resolved
        (real_rel), the target: the file's repository-relative path as a command is
        given it, followed by what came after it). A path that still starts above the
        repository root once it is normalised is given as the first of the two.

    Raises:
        ValueError: When the path is not one path_text() takes, or
            resolve_in_project() refuses the file the target leads to.
    """
    file_path, mark, name = path_text(path).partition("::")
    target = to_repo_relative(file_path)
    real_path = real_rel(target)
    if os.path.isabs(target) or target.split("/")[0] == "..":
        target = real_path
    return real_path, target + mark + name


def _path_by_language(pair_list: list[tuple[str, str]],
                      other_list: list[tuple[str, list[str], str]]
                      ) -> tuple[list[str], dict[int, list[str]]]:
    """Split the test paths of one run_test() call by the command that runs them.

    Args:
        pair_list: _test_target()'s return value for each path the call named.
        other_list: other_test_list()'s return value.

    Returns:
        (the targets the primary language's command is given, {the position in
        other_list: the targets that entry's command is given, the file of each
        relative to the entry's directory}). A target goes to an entry when its file's
        language has that entry's test command and the file lies under that entry's
        directory; every other target is the primary command's.
    """
    primary_list: list[str] = []
    by_other_dict: dict[int, list[str]] = {}
    for real_path, target in pair_list:
        lang_config = lang_of(real_path)
        for position, (ext, _, directory) in enumerate(other_list):
            is_under = not directory or real_path.startswith(f"{directory}/")
            if lang_config is not None and is_under and \
                    lang_config.test_command_tuple == LANG_DICT[ext].test_command_tuple:
                file_path, mark, name = target.partition("::")
                by_other_dict.setdefault(position, []).append(
                    posixpath.relpath(file_path, directory or ".") + mark + name)
                break
        else:
            primary_list.append(target)
    return primary_list, by_other_dict


def _one_test_run(argv: list[str], target_list: list[str], note: str, ext: str,
                  directory: str = "") -> tuple[bool | None, RunResult]:
    """Run one test command and read what it printed as that language's tests.

    Args:
        argv: The command with its targets.
        target_list: What was passed to the command as targets.
        note: What goes in front of the output, and of the excerpt when the tests were
            not green.
        ext: The language the output is read as.
        directory: Where the command is started, relative to the repository root.

    Returns:
        (True when the tests passed, False when they were red, None when the run reached
        no verdict; the 5-key dict documented in run_test's Returns, as a RunResult).
        In a container, the language's dependencies are installed first (_install_dep),
        and the command is not run when that fails. The command is given the
        language's _TEST_ENV_DICT and _RAN_ENV_DICT values. A command that could not be
        started, or ran past config.TEST_TIMEOUT_SEC, reached no verdict: the RunResult
        of one that could not be started has is_stop_short set, and that of one that ran
        past the limit has is_start and is_time_out set. What a command printed
        otherwise is read by _read_test_run().
    """
    install_result = _install_dep(ext, target_list, directory)
    if install_result is not None:
        return None, install_result
    env_pair = _RAN_ENV_DICT.get(ext)
    extra_env_dict = _TEST_ENV_DICT.get(ext, {}) | ({} if env_pair is None else {
        env_pair[0]: " ".join(word for word in (config.project_env().get(env_pair[0], ""),
                                                env_pair[1]) if word)})
    try:
        proc = run_in_project(argv, config.TEST_TIMEOUT_SEC, extra_env_dict=extra_env_dict,
                              directory=directory,
                              keep_line=lambda line: _read_line_count(_plain_text(line), ext))
    except StartError as e:
        result_dict = RunResult(_no_run_dict(
            f"{argv[0]} cannot be started ({e}). Set TEST_COMMAND to something runnable",
            target_list))
        result_dict.is_stop_short = True
        return None, result_dict
    except subprocess.TimeoutExpired:
        result_dict = RunResult(_no_run_dict(
            f"timeout: over {config.TEST_TIMEOUT_SEC} seconds", target_list))
        result_dict.is_start = True
        result_dict.is_time_out = True
        return None, result_dict
    return _read_test_run(proc, target_list, note, ext, directory)


def install_dep() -> None:
    """Inside a container, put the dependencies of the languages whose tests run in place.

    The primary language's are installed at the repository root, and every other
    language's at the directory its tests run from (other_test_list), each by
    _install_dep(). The static checks (run_check) and the programs run_project starts
    call it first. An install that fails is logged, and the tests that run next report
    it.
    """
    from codetwine_loom.sandbox import in_container
    if not in_container():
        return
    pair_list = [(config.PRIMARY_EXT, "")] + [(ext, directory)
                                               for ext, _, directory in other_test_list()]
    for ext, directory in pair_list:
        result_dict = _install_dep(ext, [], directory)
        if result_dict is not None:
            logger.info("the dependencies of the .%s files could not be installed: %s",
                        ext, one_line(result_dict["output"]))


def _install_dep(ext: str, target_list: list[str], directory: str) -> RunResult | None:
    """Inside a container, put the dependencies a language's tests load where they read them.

    The language's install_command_tuple is run from the directory its test command is
    started in. Outside a container this tool started, for a language with none, and
    when the command is not in the image, nothing is run.

    Args:
        ext: The language's LANG_DICT key.
        target_list: What the test command is to be passed as targets.
        directory: Where the test command is started, relative to the repository root.

    Returns:
        None when the tests are to run, or the RunResult of a run that stopped short of
        its tests (is_stop_short) with what the install printed, when it ended with a
        non-zero status or ran past config.TEST_TIMEOUT_SEC.
    """
    from codetwine_loom.sandbox import in_container
    lang_config = LANG_DICT.get(ext)
    if lang_config is None or not lang_config.install_command_tuple or not in_container():
        return None
    argv = list(lang_config.install_command_tuple)
    try:
        proc = run_in_project(argv, config.TEST_TIMEOUT_SEC, directory=directory)
    except StartError as e:
        logger.debug("%s cannot be started (%s), so the tests run without it", argv[0], e)
        return None
    except subprocess.TimeoutExpired:
        output = f"{argv[0]}: timeout: over {config.TEST_TIMEOUT_SEC} seconds"
    else:
        if proc.returncode == 0:
            return None
        output = cut_tail(_plain_text(proc.stdout + proc.stderr), config.TEST_OUTPUT_TAIL)
    result_dict = RunResult(_no_run_dict(
        f"the dependencies the tests load could not be installed, so no test ran:\n"
        f"{output}", target_list))
    result_dict.is_stop_short = result_dict.is_fetch_fail = True
    return result_dict


def _read_test_run(proc: subprocess.CompletedProcess, target_list: list[str], note: str,
                   ext: str, directory: str) -> tuple[bool | None, RunResult]:
    """Read what one test command printed as that language's tests.

    Args:
        proc: The finished command, as run_in_project() returns it.
        target_list: What was passed to the command as targets.
        note: What goes in front of the output, and of the excerpt when the tests were
            not green.
        ext: The language the output is read as.
        directory: Where the command was started, relative to the repository root.

    Returns:
        (True when the tests passed, False when they were red, None when the run reached
        no verdict; the 5-key dict documented in run_test's Returns, as a RunResult
        saying that the command was started, whether the run reached no verdict and
        _is_no_test_text() reads the run's own report in what it printed as a run over
        no tests (or it printed nothing and the repository's files say it had no test
        to run: _NO_TEST_FILE_DICT, whose line is then the "output"), whether that
        report holds the language's line for a run ended early, the lines saying it
        ended early, whether it is known to have stopped short of its tests, how many
        tests it ran and their names, where it reached a verdict
        without ending early with a non-zero exit code, whether it does not name each
        test it counts, where it reached a verdict, whether it says nothing of how many
        tests it ran (_UNREAD_DICT), and the paths it named as ones it could not load,
        with directory in front of them, where it was red). "output" is shown without
        the lines of _OMIT_LINE_RE_DICT.
    """
    # The failing names, the verdict and the lines of the tables are read off the run's
    # own report in what was returned (_own_text), without the control sequences of a
    # colored output; "output" holds the tail of the whole, and the failure text is cut
    # from the whole to no more than that tail, both without the lines naming each test
    # that did not fail
    whole = _plain_text(proc.stdout + proc.stderr)
    own = _own_text(whole, ext)
    omit_line_re = _OMIT_LINE_RE_DICT.get(ext)
    excerpt_text = whole if omit_line_re is None else omit_line_re.sub("", whole)
    out = cut_tail(excerpt_text, config.TEST_OUTPUT_TAIL)

    fail_pair_list = _fail_pair_list(own, ext)
    fail_list = [name for _, name in fail_pair_list]

    verdict = _test_verdict(proc.returncode, own, ext, fail_list)
    result_dict = RunResult({
        "status": TEST_STATUS_DICT[verdict],
        "failed": fail_list if verdict is False else [],
        "failed_excerpt": "" if verdict else note + cut_tail(
            _fail_excerpt(excerpt_text, ext), config.TEST_OUTPUT_TAIL),
        "targets": target_list, "output": note + out})
    result_dict.is_start = True
    load_fail = _LOAD_FAIL_PATH_DICT.get(ext)
    result_dict.load_fail_path_list = []
    if verdict is False and load_fail is not None:
        own_line_list = line_list_of(own)
        run_dir = os.path.join(config.PROJECT_DIR, directory)
        result_dict.load_fail_path_list = [
            posixpath.join(directory, path) for index, name in fail_pair_list
            for path in load_fail(own_line_list, index, name, run_dir)]
    own_dict = {"status": result_dict["status"], "output": own}
    no_test_file = _NO_TEST_FILE_DICT.get(ext)
    no_test_line = ""
    if verdict is None and not own.strip() and no_test_file is not None:
        no_test_line = no_test_file(list(proc.args),
                                    os.path.join(config.PROJECT_DIR, directory))
    if no_test_line:
        result_dict["output"] = result_dict["failed_excerpt"] = note + no_test_line
    result_dict.is_no_test = _no_test_in(own_dict, ext) or bool(no_test_line)
    result_dict.is_early_stop = _early_stop_in(own_dict, ext)
    result_dict.is_fail_stop = _fail_stop_in(own_dict, ext)
    result_dict.early_stop_text = _early_stop_text(own, ext)
    result_dict.early_stop_line_list = _early_stop_line_list(own, ext)
    stop_short_re = _STOP_SHORT_RE_DICT.get(ext)
    result_dict.is_stop_short = verdict is None and not result_dict.is_no_test \
        and stop_short_re is not None and bool(stop_short_re.search(own))
    fetch_fail_re = _FETCH_FAIL_RE_DICT.get(ext)
    result_dict.is_fetch_fail = result_dict.is_stop_short \
        and fetch_fail_re is not None and bool(fetch_fail_re.search(own))
    # A run with a zero exit code is held to the tests it ran whatever lines it prints
    if verdict is not None and not (result_dict.is_early_stop and proc.returncode != 0):
        result_dict.is_ran_hold = True
        result_dict.ran_count = _ran_count(own, ext)
        result_dict.ran_name_list = _ran_name_list(own, ext, result_dict["failed"],
                                                   directory)
        result_dict.ran_unit_dict = _ran_unit_dict(own, ext, directory)
    if verdict is not None:
        name_short = _NAME_SHORT_DICT.get(ext)
        result_dict.is_name_short = name_short is not None \
            and name_short(own, result_dict["failed"])
    unread = _UNREAD_DICT.get(ext)
    result_dict.is_ran_unread = unread is not None and unread(own, proc.returncode)
    return verdict, result_dict


def _merge_test_run(part_list: list[tuple[bool | None, RunResult, str]]
                    ) -> tuple[bool | None, dict]:
    """Put the runs of several test commands together as one run_test() return value.

    Args:
        part_list: [(the run's verdict, its own result as _one_test_run() returns it,
            the line naming the command), ...], in the order they ran. One entry is
            returned as it is.

    Returns:
        (False when any run was red; otherwise None when a run reached no verdict for
        another reason than finding no test of its language, True when one passed,
        and None when none reached a verdict; the 5-key dict: the failing names and
        the targets of every run, each run's excerpt and output under the line naming
        its command).
    """
    if len(part_list) == 1:
        return part_list[0][0], part_list[0][1]
    verdict_list = [verdict for verdict, _, _ in part_list]
    is_part_not_run = any(verdict is None and not result_dict.is_no_test
                          for verdict, result_dict, _ in part_list)
    if False in verdict_list:
        verdict = False
    elif is_part_not_run or True not in verdict_list:
        verdict = None
    else:
        verdict = True
    return verdict, {
        "status": TEST_STATUS_DICT[verdict],
        "failed": [name for _, result_dict, _ in part_list
                   for name in result_dict["failed"]],
        "failed_excerpt": "\n".join(
            f"{head}\n{result_dict['failed_excerpt']}"
            for _, result_dict, head in part_list if result_dict["failed_excerpt"]),
        "targets": [target for _, result_dict, _ in part_list
                    for target in result_dict["targets"]],
        "output": "\n".join(f"{head}\n{result_dict['output']}"
                            for _, result_dict, head in part_list)}


def _run_head(argv: list[str], directory: str) -> str:
    """Return the line one test command's output is put under when several ran.

    Examples:
        (["npm", "test", "--silent"], "web") -> "== npm test --silent (in web) =="

    Args:
        argv: The command with its targets.
        directory: Where it was started, relative to the repository root.

    Returns:
        "== <command> ==", with " (in <directory>)" after the command when directory is
        not empty; the command's first word without its directory.
    """
    command = _command_text(argv)
    return f"== {command}{f' (in {directory})' if directory else ''} =="


def _test_pair_list(path_list: list[str] | tuple[str, ...] | None
                    ) -> tuple[list[tuple[str, str]], dict | None]:
    """Read the paths one run_test() call was given as the files and targets they name.

    Each path is validated before reaching argv, and a bad one comes back as a result
    whose status is "did not run". The commands are told apart by the file each names
    and are given the target _test_target() writes. A target that starts with "-" is
    refused.

    Args:
        path_list: The paths, a string already made a list of one; None for none.

    Returns:
        ([_test_target()'s value for each path, in order], None), or
        ([], a result whose status is "did not run") for a value that is no list and a
        path that is refused.
    """
    if path_list is not None and not isinstance(path_list, (list, tuple)):
        return [], _test_fail_dict(f"path_list must be a list of test file paths, not "
                                   f"{type(path_list).__name__}")
    pair_list = []
    for path in path_list or ():
        try:
            real_path, target = _test_target(path)
        except ValueError as e:
            return [], _test_fail_dict(str(e))
        if target.startswith("-"):
            return [], _test_fail_dict(f"{path}: a test path cannot start with \"-\"")
        pair_list.append((real_path, target))
    return pair_list, None


def _clear_count(part_list: list[tuple[bool | None, RunResult, str]]) -> None:
    """Take what the runs of a call given paths say of the tests they ran out of them.

    A run given paths ran some of the tests only; one that is red and names no failing
    test still says it does not name each test it counts.

    Args:
        part_list: [(the run's verdict, its own result, the line naming the command),
            ...]; each result is changed in place.
    """
    for _, part, _ in part_list:
        part.ran_count, part.ran_name_list, part.ran_unit_dict = None, None, None
        part.is_name_short = part.is_name_short and part["status"] == "failed" \
            and not part["failed"]
        part.is_ran_hold = False


def _join_test_run(part_list: list[tuple[bool | None, RunResult, str]],
                   name_list: list[str], is_start: bool) -> tuple[bool | None, RunResult]:
    """Put the runs of the test commands of one call together as one RunResult.

    Args:
        part_list: [(the run's verdict, its own result, the line naming the command),
            ...], in the order they ran.
        name_list: The name each command's result is kept under, in the same order.
        is_start: Whether one of the commands was started.

    Returns:
        (the verdict _merge_test_run() reads; the RunResult holding the merged 5-key
        dict, whether a command was started, ran past its time limit or stopped short of
        its tests, whether every run found no test, whether one ended early or stopped
        at a count of failing tests, the lines saying so, and each command's own result
        by its name).
    """
    verdict, merge_dict = _merge_test_run(part_list)
    result_dict = RunResult(merge_dict)
    result_dict.is_start = is_start
    result_dict.is_time_out = any(part.is_time_out for _, part, _ in part_list)
    result_dict.is_stop_short = any(part.is_stop_short for _, part, _ in part_list)
    result_dict.is_fetch_fail = any(part.is_fetch_fail for _, part, _ in part_list)
    result_dict.is_no_test = verdict is None and all(
        part.is_no_test for _, part, _ in part_list)
    result_dict.is_early_stop = any(part.is_early_stop for _, part, _ in part_list)
    result_dict.is_fail_stop = any(part.is_fail_stop for _, part, _ in part_list)
    result_dict.early_stop_text = "\n".join(
        part.early_stop_text for _, part, _ in part_list if part.early_stop_text)
    result_dict.early_stop_line_list = [
        line for _, part, _ in part_list for line in part.early_stop_line_list]
    result_dict.part_dict = {name: part
                             for (_, part, _), name in zip(part_list, name_list)}
    return verdict, result_dict


def _run_test(path_list: list[str] | None, refill: bool) -> dict:
    """Run the tests, as run_test() says.

    The primary language's command runs first, then the commands of the other languages
    that can be run (other_test_list). A call that names no paths runs those
    _other_whole_list() returns. A call that names paths runs each command over the
    paths that are its own (_path_by_language) and leaves out a command none of them is
    for.

    The flow:
    1. Read and check the paths (_test_pair_list).
    2. Choose the commands to run, and return the refusal when the primary language's
       tests cannot run (_no_run_reason) and no other command is chosen; then, with
       refill, call refill_cache_on_dep_change().
    3. Run each command (_one_test_run).
    4. Close the round when a command was started, put the runs together as one
       RunResult (_join_test_run), and keep the verdict and the last whole run.

    Args:
        path_list: As run_test() takes it. One path alone may be a string, and each
            path is run as _test_target() writes it.
        refill: Whether the broker is asked to fill the package cache again when a
            dependency file was changed (refill_cache_on_dep_change()).

    Returns:
        run_test()'s return value.
    """
    # Read through config at each call: a value replaced at run time is followed
    test_path, ext, style = config.TEST_PATH, config.PRIMARY_EXT, config.TEST_TARGET_STYLE

    # == Step 1: read the paths ===========================================
    # One path alone may come as a string
    if isinstance(path_list, str):
        path_list = [path_list]
    pair_list, fail_dict = _test_pair_list(path_list)
    if fail_dict is not None:
        return fail_dict

    # == Step 2: choose the commands ======================================
    _exclude_in_container()
    other_list = other_test_list()
    primary_list, by_other_dict = _path_by_language(pair_list, other_list)
    # Without paths every command is chosen; with them, the ones some path is for
    other_run_list = [(position, entry) for position, entry in enumerate(other_list)
                      if not path_list or position in by_other_dict]

    is_primary_run = not path_list or bool(primary_list)
    no_run_reason = _no_run_reason(primary_list or None) if is_primary_run else None
    if no_run_reason is not None and not other_run_list:
        return _test_fail_dict(no_run_reason)

    # What came of the refill goes in front of the output, and of the excerpt when the
    # tests were not green
    note = refill_cache_on_dep_change() if refill else ""

    # == Step 3: run each command =========================================
    part_list: list[tuple[bool | None, RunResult, str]] = []
    # What each command is named where its result is kept on its own: its command
    # line as the settings hold it, without a target or a path
    name_list: list[str] = []
    base_command_dict = dict(config.OTHER_TEST_LIST)
    if is_primary_run and no_run_reason is None:
        argv, target_list, path_note = _build_argv(primary_list or None, style,
                                                   test_path, ext)
        verdict, part = _one_test_run(argv, target_list, note + path_note, ext)
        part_list.append((verdict, part, _run_head(argv, "")))
        name_list.append(" ".join(config.TEST_COMMAND))
        note = ""
    for position, (other_ext, command, directory) in other_run_list:
        lang_config = LANG_DICT[other_ext]
        other_path_list = by_other_dict.get(position)
        if not other_path_list and not _has_other_test_path(other_ext, directory):
            continue
        argv, target_list, path_note = _build_argv(
            other_path_list, lang_config.test_target,
            config.test_dir_of(other_ext, os.path.join(config.PROJECT_DIR, directory)),
            other_ext, command=command)
        verdict, part = _one_test_run(
            argv, target_list, note + path_note, other_ext, directory)
        part_list.append((verdict, part, _run_head(argv, directory)))
        name_list.append(" ".join(base_command_dict.get(other_ext, command)))
        note = ""
    if not part_list:
        return _test_fail_dict(no_run_reason or "there is no test to run")
    if path_list:
        _clear_count(part_list)

    # == Step 4: put the runs together ====================================
    # A round closes once a command was started; the gates above and a command that
    # could not be started close none
    is_start = any(part.is_start for _, part, _ in part_list)
    if is_start:
        _close_patch_round()
    # One verdict, read two ways: the word goes in "status" and the value itself is kept
    # (last_test_ok())
    verdict, result_dict = _join_test_run(part_list, name_list, is_start)
    if is_primary_run and no_run_reason is None:
        result_dict.primary_name = name_list[0]
    result_dict.is_whole = not path_list
    _keep_verdict(verdict)
    _keep_whole_run(result_dict, not path_list)
    return result_dict


# One finding of the static check: "<path>:<line>[:<column>]: <message>" or
# "<path>(<line>,<column>): <message>"
_CHECK_LINE_RE = re.compile(
    r"^(?P<path>[^:\s][^:]*?)(?::\d+(?::\d+)?|\(\d+,\d+\)):\s*(?P<message>.+)$")

# The exit status of a check that cannot examine the tree: the shell's for a command it
# cannot find, and the default checks' for a tool that is missing or fails to run
_CHECK_CANNOT_RUN = 127


def _check_pair_list() -> list[tuple[str, list[str]]]:
    """Return the static checks run_check() runs, each under the name it is kept by.

    Returns:
        [(the check's command line without a root target, its argument list with the
        root target), ...]: CHECK_COMMAND, when it is set or the primary language has a
        default, followed by the default check of every other language the index holds a
        file of (config.OTHER_CHECK_LIST) whose command is on PATH; CHECK_COMMAND alone
        while CHECK_COMMAND in the environment names a command.
    """
    lang_config = LANG_DICT.get(config.PRIMARY_EXT)
    pair_list = []
    if config.CHECK_COMMAND:
        pair_list.append((_command_text(config.CHECK_COMMAND), _with_root_target(
            config.CHECK_COMMAND,
            lang_config.check_command_tuple if lang_config is not None else ())))
    for ext, other_argv in config.OTHER_CHECK_LIST:
        if shutil.which(other_argv[0]) is None:
            logger.debug("%s is not on PATH, so the static check of .%s files does not "
                         "run", other_argv[0], ext)
            continue
        pair_list.append((_command_text(other_argv), _with_root_target(
            other_argv, LANG_DICT[ext].check_command_tuple, ext)))
    return pair_list


def _command_text(argv: list[str]) -> str:
    """Return a command line with its first word's directory left out.

    Examples:
        ["/opt/venv/bin/python", "-m", "ruff", "check"] -> "python -m ruff check"

    Args:
        argv: The command and its arguments, at least the command.

    Returns:
        The words joined by spaces.
    """
    return " ".join([os.path.basename(argv[0])] + list(argv[1:]))


def check_argv_list() -> list[list[str]]:
    """Return the static checks run_check() runs, each as an argument list.

    Returns:
        The argument lists of _check_pair_list(), each with its language's root
        target.
    """
    return [argv for _, argv in _check_pair_list()]


def check_command_text() -> str:
    """Return the command lines of the static checks run_check() runs, as one text.

    Examples:
        one check    -> "ruff check --select F821 ."
        two checks   -> "ruff check --select F821 . / ctloom-check-ts"

    Returns:
        Each command with its first word's directory left out, joined by " / "; "" when
        there is no check to run.
    """
    return " / ".join(_command_text(argv) for argv in check_argv_list())


class CheckCount(collections.Counter):
    """run_check()'s return value: what the checks that ran found between them.

    It also holds what each found on its own, by its command line without a root target
    (new_finding_list).
    """

    by_check_dict: dict[str, collections.Counter]


def new_finding_list(before: collections.Counter, after: collections.Counter) -> list[str]:
    """Return what the static checks find now that they did not find before.

    Args:
        before: run_check() before a change.
        after: run_check() after it.

    Returns:
        The findings, sorted, one entry per time each is found more often than before.
        Two CheckCount values are compared check by check, over the checks that ran
        both times; anything else is compared whole.
    """
    before_dict = getattr(before, "by_check_dict", None)
    after_dict = getattr(after, "by_check_dict", None)
    if before_dict is None or after_dict is None:
        return sorted((after - before).elements())
    return sorted(finding for command in after_dict if command in before_dict
                  for finding in (after_dict[command] - before_dict[command]).elements())


def run_check() -> collections.Counter | None:
    """Run every static check from the repository root and count what they find.

    The checks are check_argv_list()'s.

    Examples:
        "a.py:210:42: F821 Undefined name `f`"  -> "a.py: F821 Undefined name `f`"
        "a.ts(3,5): error TS2304: Cannot ..."   -> "a.ts: error TS2304: Cannot ..."
        "<PROJECT_DIR>/src/a.c:4:3: error: ..." -> "src/a.c: error: ..."
        "Found 1 error."                        -> (not a finding)

    Returns:
        A CheckCount: {"<path>: <message>": how many times it is found} over the checks
        that ran, the line and column left out (a finding that moves is the same
        finding) and a path under PROJECT_DIR made relative to it, or None when there is
        no check to run or none of them ran (_check_count).
    """
    _exclude_in_container()
    install_dep()
    by_check_dict = {}
    pair_list = _check_pair_list()
    for name, argv in pair_list:
        counter = _check_count(argv)
        if counter is not None:
            by_check_dict[name] = by_check_dict.get(name, collections.Counter()) + counter
    if not by_check_dict:
        return None
    check_count = CheckCount(sum(by_check_dict.values(), collections.Counter()))
    check_count.by_check_dict = by_check_dict
    return check_count


def _check_count(argv: list[str]) -> collections.Counter | None:
    """Run one static check from the repository root and count what it finds.

    Args:
        argv: The check, as check_argv_list() returns it.

    Returns:
        The findings as run_check() counts them, or None when the command cannot be
        started, exits 127, exits with another status than 0 without printing a line
        that reads as a finding, or runs out of TEST_TIMEOUT_SEC.
    """
    try:
        proc = run_in_project(
            argv, config.TEST_TIMEOUT_SEC,
            keep_line=lambda line: int(
                _CHECK_LINE_RE.match(_plain_text(line).strip()) is not None))
    except StartError as e:
        logger.warning("%s cannot be started (%s), so the static check does not run. "
                       "Set CHECK_COMMAND to something runnable", argv[0], e)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("the static check ran over %d seconds and is not counted",
                       config.TEST_TIMEOUT_SEC)
        return None
    if proc.returncode == _CHECK_CANNOT_RUN:
        logger.warning("%s cannot check this tree, so the static check does not run: %s",
                       argv[0], cut_tail(proc.stderr.strip(), config.TEST_OUTPUT_TAIL))
        return None
    root = os.path.join(os.path.abspath(config.PROJECT_DIR), "")
    finding_counter: collections.Counter = collections.Counter()
    for line in line_list_of(_plain_text(proc.stdout + proc.stderr)):
        match = _CHECK_LINE_RE.match(line.strip())
        if match:
            path = match["path"].removeprefix(root)
            finding_counter[f"{path}: {match['message']}"] += 1
    if proc.returncode != 0 and not finding_counter:
        logger.warning("%s exited %d without a finding that can be read, so the static "
                       "check is not counted: %s", argv[0], proc.returncode,
                       cut_tail((proc.stdout + proc.stderr).strip(),
                                config.TEST_OUTPUT_TAIL))
        return None
    logger.info("the static check (%s) found %d finding(s)", os.path.basename(argv[0]),
                sum(finding_counter.values()))
    return finding_counter


def run_format() -> dict | None:
    """Run FORMAT_COMMAND from the repository root over the files changed since HEAD.

    The files are the ones git reports as changed or new in PROJECT_DIR that are still
    there, and their paths follow the command, each as path_arg() writes it. What the
    command changes in them, or beside them, is part of the working tree from then on.

    Returns:
        {"files": the paths the command was given, relative to the repository,
        "exit_status": the command's, or None when it could not be started or ran past
        TEST_TIMEOUT_SEC, "output": the end of what it printed, or the line saying why
        when "exit_status" is None}, or None when FORMAT_COMMAND is empty or no file was
        changed.
    """
    if not config.FORMAT_COMMAND:
        return None
    root = config.PROJECT_DIR
    file_list = [path for path in gitcmd.status_path_list(root, disk=True)
                 if os.path.isfile(os.path.join(root, path))]
    if not file_list:
        return None
    argv = [*config.FORMAT_COMMAND, *(path_arg(path) for path in file_list)]
    result_dict = {"files": file_list, "exit_status": None, "output": ""}
    try:
        proc = run_in_project(argv, config.TEST_TIMEOUT_SEC)
    except StartError as e:
        result_dict["output"] = (f"{argv[0]} cannot be started ({e}). Set "
                                 f"FORMAT_COMMAND to something runnable")
    except subprocess.TimeoutExpired:
        result_dict["output"] = f"it ran over {config.TEST_TIMEOUT_SEC} seconds"
    else:
        result_dict.update(exit_status=proc.returncode, output=cut_tail(
            (proc.stdout + proc.stderr).strip(), config.TEST_OUTPUT_TAIL))
    if result_dict["exit_status"] != 0:
        logger.warning("the formatting command (%s) did not end cleanly over %d "
                       "file(s): %s", os.path.basename(argv[0]), len(file_list),
                       result_dict["output"] if result_dict["exit_status"] is None
                       else f"exit status {result_dict['exit_status']}")
    else:
        logger.info("the formatting command (%s) ran over %d file(s)",
                    os.path.basename(argv[0]), len(file_list))
    return result_dict


def git_diff_stat() -> dict:
    """Return the files that were changed and by how many lines.

    Returns:
        {"files": [{"path": str, "insertions": int, "deletions": int}],
         "untracked": list[str],   # paths of files git does not track
         "total_insertions": int,
         "total_deletions": int}
        A file git could not read has "?" (gitcmd.UNREAD_COUNT) in place of both of its
        numbers and is not in the totals.
    """
    file_list = gitcmd.numstat_list(config.PROJECT_DIR, "HEAD")
    new_file_list = gitcmd.new_file_list(config.PROJECT_DIR)
    count_list = [file_entry for file_entry in file_list
                  if file_entry["insertions"] != gitcmd.UNREAD_COUNT]

    return {"files": file_list, "untracked": new_file_list,
            "total_insertions": sum(file_entry["insertions"] for file_entry in count_list),
            "total_deletions": sum(file_entry["deletions"] for file_entry in count_list)}


def change_by_kind_dict() -> dict:
    """Return the changed paths split into added, modified and deleted.

    The diff itself is read by gitcmd.name_status(). The untracked files are added to
    "added" here.

    Returns:
        {"added": list[str], "modified": list[str], "deleted": list[str]}, each sorted.
        A file created with create_file is untracked and lands in "added".
    """
    by_kind_dict = gitcmd.name_status(config.PROJECT_DIR)
    by_kind_dict["added"] = sorted(set(by_kind_dict["added"])
                                   | set(gitcmd.new_file_list(config.PROJECT_DIR)))
    return by_kind_dict


def revert_file(path: str) -> dict:
    """Undo this request's changes to one file, putting back what it held at the start.

    A file that matched HEAD when the request started is restored from HEAD, and one
    git did not track then is deleted, which is what undoes a create_file; so is one
    where HEAD holds a directory, which is not taken out of HEAD. A file that
    already differed from HEAD is put back to the contents an editing tool found there
    before its first write (_keep_start). A file this request has not changed is left
    as it is and refused, and so is one that already differed from HEAD and was changed
    without an editing tool, whose starting contents were never kept, and one git
    carries nothing of (_no_carry_reason). At a path an editing tool wrote during this
    request (write_path_set) where a directory, a named pipe or another thing that is no
    regular file now stands, that thing is deleted, a directory with all it holds,
    before the file is put back. A --dry-run writes nothing and refuses. A
    file put back is given a modification time after the one it had
    (tmpfile.move_mtime_on).

    A path whose last part is a symbolic link is put back as the link's own path when
    that path is one this request changed (own_change_set): a file replaced by a link,
    a link pointed elsewhere, a link that was not there. Any other link is put back as
    the file it points at.

    A revert that went through takes the path back out of write_path_set(). A revert
    that failed leaves it there.

    Args:
        path: A path relative to PROJECT_DIR.

    Returns:
        {"ok": bool, "path": str, "error": str}
    """
    if config.EDIT_DRY_RUN:
        return _revert_dict(False, path,
                            "this is a dry run: nothing was written, so there is nothing to "
                            "revert")
    try:
        link = _link_entry(path_text(path))
        change_set = own_change_set()
        link_path = os.path.relpath(link, os.path.realpath(config.PROJECT_DIR)) \
            if link else ""
        if link_path in change_set:
            target, real_path = link, link_path
        else:
            target = resolve_in_project(path)
            real_path = real_rel(path)
    except ValueError as e:
        return _revert_dict(False, path, str(e))
    except _PATH_ERROR_TUPLE as e:
        return _revert_dict(False, path, _path_error_text(e))
    if not os.path.islink(target) and os.path.isdir(target) \
            and real_path not in _write_hash_dict:
        return _revert_dict(False, path, f"{path} is a directory, not a file")
    if real_path not in change_set:
        if not os.path.lexists(target):
            return _revert_dict(False, path, f"{path} does not exist")
        try:
            reason = _no_carry_reason(path, real_path)
        except _PATH_ERROR_TUPLE as e:
            return _revert_dict(False, path, _path_error_text(e))
        if reason:
            return _revert_dict(False, path,
                                f"{reason}, so what it held when this request started is "
                                f"not known and it cannot be reverted")
        return _revert_dict(False, path,
                            f"{path} holds what it held when this request started, so there "
                            f"is nothing of this request to revert")
    # What an editing tool wrote that has since become something other than a file is
    # taken away first, the directory with all it holds
    is_clear = real_path in _write_hash_dict and _holds_no_file(real_path)
    if is_clear:
        try:
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
        except OSError as e:
            return _revert_dict(False, path, f"cannot delete: {e}")
    if real_path in _base_hash_dict:
        return _put_back_start(path, real_path, target)
    # The path is read as itself, not as a pattern git matches files against
    literal_path = f":(literal){real_path}"
    # A path neither HEAD nor the index holds as a file, a link or a repository of its
    # own is deleted, a directory of HEAD's included. One HEAD holds so comes back from
    # HEAD, whatever the index says of it now
    head_proc = gitcmd.run(config.PROJECT_DIR, "ls-tree", "-z", "HEAD", "--",
                           literal_path, text=False)
    track_proc = gitcmd.run(config.PROJECT_DIR, "ls-files", "-z", "--", literal_path,
                            text=False)
    raw_path = os.fsencode(real_path)
    is_head = any(entry.partition(b"\t")[2] == raw_path
                  and entry.split(b" ", 2)[1:2] in ([b"blob"], [b"commit"])
                  for entry in head_proc.stdout.split(b"\0"))
    if not is_head and raw_path not in track_proc.stdout.split(b"\0"):
        if not os.path.lexists(target) and not is_clear:
            return _revert_dict(False, path, f"{path} does not exist")
        tmpfile.file_stamp(target)
        try:
            if os.path.lexists(target):
                os.remove(target)
        except OSError as e:
            return _revert_dict(False, path, f"cannot delete: {e}")
        remove_empty_dir(real_path)
        _write_hash_dict.pop(real_path, None)
        return _revert_dict(True, path, "")
    # The file comes back from HEAD's commit, not from the index
    before_stamp = tmpfile.file_stamp(target)
    proc = gitcmd.run(config.PROJECT_DIR, "checkout", "HEAD", "--", literal_path)
    if proc.returncode == 0:
        tmpfile.move_mtime_on(target, before_stamp)
        _write_hash_dict.pop(real_path, None)
    return _revert_dict(proc.returncode == 0, path, proc.stderr.strip())


def _revert_dict(ok: bool, path: str, error: str) -> dict:
    """Build revert_file()'s return value.

    Args:
        ok: Whether the file was put back.
        path: The path as the tool was given it.
        error: Why it was not; "" when it was, or what git wrote on standard error for a
            file put back from HEAD.

    Returns:
        {"ok": ok, "path": path, "error": error}
    """
    return {"ok": ok, "path": path, "error": error}


def _put_back_start(path: str, real_path: str, target: str) -> dict:
    """Put a path that already differed from HEAD back to what it held at the start.

    Bytes are not written where something other than a regular file or a symbolic link
    stands (tmpfile.check_file); the call answers that it cannot put it back.

    Args:
        path: The path as the tool was given it.
        real_path: The repository-relative path of the file it lands on.
        target: The absolute path of that file.

    Returns:
        revert_file()'s return value.
    """
    if real_path not in _start_byte_dict and real_path not in _start_link_dict:
        return _revert_dict(False, path,
                            f"{path} already held changes nobody had committed when this "
                            f"request started, and it was changed without an editing tool, "
                            f"so what it held then is not known. Restoring it from HEAD would "
                            f"throw those changes away: change it back with replace_in_file "
                            f"or replace_line_range instead")
    start_byte = _start_byte_dict.get(real_path)
    before_stamp = tmpfile.file_stamp(target)
    try:
        if real_path in _start_link_dict:
            if os.path.lexists(target):
                os.remove(target)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.symlink(_start_link_dict[real_path], target)
        elif start_byte is None:
            if os.path.lexists(target):
                os.remove(target)
                remove_empty_dir(real_path)
        else:
            if os.path.islink(target):
                os.remove(target)
            elif os.path.exists(target):
                tmpfile.check_file(target, sys.maxsize)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as fp:
                fp.write(start_byte)
    except OSError as e:
        return _revert_dict(False, path, f"cannot put it back: {e}")
    tmpfile.move_mtime_on(target, before_stamp)
    _write_hash_dict.pop(real_path, None)
    return _revert_dict(True, path, "")


def _link_path(path: str) -> tuple[str, dict | None]:
    """Return where a symbolic link a path names stands, for deleting the link itself.

    Args:
        path: A path relative to PROJECT_DIR.

    Returns:
        (the absolute path of the link, its directories resolved and its last part left
        as it is, None) when the last part of the path is a symbolic link, ("", None)
        when it is not, or ("", the 7-key failure dict) when the link's directory is one
        resolve_in_project() refuses or the link itself stands where git keeps the
        repository (paths.is_git_dir_path()).
    """
    try:
        entry = _link_entry(path)
    except ValueError as e:
        return "", _patch_result_dict(False, path, error=str(e))
    if entry and is_git_dir_path(entry):
        return "", _patch_result_dict(False, path,
                                      error=f"inside the git directory: {path}")
    return entry, None


def delete_file(path: str) -> dict:
    """Delete one file from the working tree.

    A path whose last part is a symbolic link deletes the link, wherever it points, and
    leaves what it points at; a link that stands where git keeps the repository, or in
    it, is refused. The directories made during this request that the deletion leaves
    empty are deleted with it (remove_empty_dir).

    Args:
        path: A path relative to PROJECT_DIR, of a file that is there,
            e.g. "myapp/cache.py".

    Returns:
        The same 7-key dict as replace_in_file. "stat" is "deleted", under --dry-run
        too.
    """
    global _is_patch_since_test

    try:
        path = path_text(path)
    except ValueError as e:
        return _patch_result_dict(False, path, error=str(e))
    try:
        target, refuse_dict = _link_path(path)
    except _PATH_ERROR_TUPLE as e:
        return _patch_result_dict(False, path, error=_path_error_text(e))
    if refuse_dict:
        return refuse_dict
    is_link = bool(target)
    if not is_link:
        path, target, refuse_dict = _target_path(path)
        if target is None:
            return refuse_dict
    refuse_dict = _over_round_limit(path)
    if refuse_dict:
        return refuse_dict
    if not is_link and os.path.isdir(target):
        return _patch_result_dict(False, path, error=f"{path} is a directory, not a file")
    if not is_link and not os.path.exists(target):
        return _patch_result_dict(False, path, error=f"{path} does not exist")
    real_path = os.path.relpath(target, os.path.realpath(config.PROJECT_DIR))
    try:
        reason = _no_carry_reason(path, real_path)
    except _PATH_ERROR_TUPLE as e:
        return _patch_result_dict(False, path, error=_path_error_text(e))
    if reason:
        return _patch_result_dict(
            False, path,
            error=f"{reason}, so deleting it is not carried out of this run")
    if config.EDIT_DRY_RUN:
        _dry_change_list.append(f"deleted {path}\n")
        _dry_path_set.add(real_path)
        return _patch_result_dict(True, path, stat="deleted")
    _keep_start(real_path)
    tmpfile.file_stamp(target)
    try:
        os.remove(target)
    except OSError as e:
        return _patch_result_dict(False, path, error=f"cannot delete: {e}")
    remove_empty_dir(real_path)
    _is_patch_since_test = True
    if _hold_start(real_path):
        _write_hash_dict.pop(real_path, None)
    else:
        _write_hash_dict[real_path] = _file_hash(real_path)
    return _patch_result_dict(True, path, stat="deleted")
