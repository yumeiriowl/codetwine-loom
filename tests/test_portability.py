"""Checks over the whole package for kinds of mistake that can appear anywhere in it.

- a name out of the repository the tool happened to be pointed at ends up in text the
  model reads
- the instructions state a cap the code never applies
- a setting's name is a name the operating system or a CI shell already uses
- a text tells the model to make a call the REPL does not answer to, which is what a
  renamed ending call leaves behind
- a caller still names something a rename took away

Each check below is written so that the next one of its kind fails here rather than in a
run: they read the tables and the source rather than a list somebody has to extend.
"""

import ast
import itertools
import json
import importlib
import os
import pathlib
import re

import pytest
from codetwine.config.settings import EXT_TO_DEFINITION_DICT

from codetwine_loom import (agent, config, dispatch, integrate, plan, refactor, rlm,
                             testcreate, websearch)
from codetwine_loom import answer as answer_mod
from codetwine_loom.languages import (ALIAS_DICT, DEF_EXAMPLE_DICT, LANG_DICT, LANG_NAME_DICT,
                                         PATH_EXAMPLE_DICT, RUN_EXAMPLE_DICT,
                                         SIGNATURE_EXAMPLE_DICT, edit_note, edit_paragraph)
from codetwine_loom.search import query
from codetwine_loom.search import scope as scope_mod
from codetwine_loom.tools import edit, explore, source, web

from tests.test_agent import _TARGET_SPECIFIC, without_own_name

_PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "codetwine_loom"

# The package's own name
_PKG = "codetwine_loom"

# One captured run of a real build tool per file; test_edit.py replays them
_CAPTURED_RUNS = pathlib.Path(__file__).resolve().parent / "data" / "test_output"

# The languages run_test has a command for. The tables read against a build tool's
# output cover these; a language with no default command (SQL) has no such output
_EXT_WITH_TEST_COMMAND_SET = {ext for ext, lang_config in LANG_DICT.items()
                              if lang_config.test_command_tuple}

# The two modules whose whole job is the tool this project is built on. Naming it there
# is the point, not a leftover: knowledge.py calls it, and cli.py's --help describes the
# subcommand that runs it
_ABOUT_THE_DEPENDENCY = {"knowledge.py", "cli.py"}

# What the source scan looks for. The dependency's own output file name only appears in
# the two modules the scan already leaves out, so nothing is excused here
_IN_THE_SOURCE = _TARGET_SPECIFIC

# Names an operating system, a shell profile or a CI runner sets for its own purposes.
# A setting that answers to one of these is read from an environment nobody meant to
# configure this tool with
_TAKEN_NAMES = {
    "CI", "DEBUG", "DRY_RUN", "EDITOR", "ENV", "HOME", "HOST", "HOSTNAME", "LANG",
    "LANGUAGE", "LC_ALL", "LOGLEVEL", "MODEL", "PAGER", "PATH", "PWD", "PYTHONPATH",
    "SHELL", "TEMP", "TERM", "TIMEOUT", "TMP", "TMPDIR", "TZ", "USER", "VERBOSE",
    "VERSION", "LOG_LEVEL", "TEST_TARGET",
}


def _string_literals(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return every string literal in one file that is not a docstring.

    Docstrings are left out on purpose: they do not reach the model (the loop shows one
    line of a tool's docstring, and test_agent covers those through
    model_text_dict()), and they may describe the dependency.

    Args:
        path: The file to read.

    Returns:
        (line number, the string) for each literal, in source order.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings]


def _source_files() -> list[pathlib.Path]:
    """Return the package's source files, apart from the two about the dependency."""
    return [p for p in sorted(_PACKAGE.rglob("*.py"))
            if p.name not in _ABOUT_THE_DEPENDENCY]


def _importing_files() -> list[pathlib.Path]:
    """Return every file of the package, each of which may name something of it."""
    return sorted(_PACKAGE.rglob("*.py"))


def test_the_scan_reaches_the_whole_package():
    """A scan that found no files would pass every check below without reading a line."""
    files = _source_files()
    assert len(files) > 10
    assert any(p.name == "agent.py" for p in files)


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_string_in_the_source_names_a_particular_repository(path):
    """Every new error message is scanned, not only the ones somebody registered.

    test_agent checks the assembled texts. This checks the source they are built from,
    so a message added inside a function is covered too.
    """
    for lineno, text in _string_literals(path):
        for name in _IN_THE_SOURCE:
            assert name not in without_own_name(text), f"{path.name}:{lineno} names {name}"


def test_no_setting_is_named_like_a_common_environment_variable():
    """A setting's name is its environment variable's name, so it has to be its own."""
    taken = sorted(_TAKEN_NAMES & set(_setting_names()))
    assert not taken, f"these settings answer to a name something else sets: {taken}"


def _setting_names() -> list[str]:
    """Return every environment variable config.py reads.

    config.py records each name as it reads it, so a setting added there is checked
    here without anyone registering it. test_sandbox keeps that record complete by
    checking that config.py reads the environment only through its own helpers.

    Returns:
        The variable names, sorted.
    """
    return list(config.env_name_tuple())


# What .env.example may name besides a setting config.py reads: codetwine's own, and
# the proxy variables and the file of certificate authorities the HTTP clients read
_READ_BY_THE_DEPENDENCY = {"DOC_TEMPLATE_PATH", *config.PROXY_NAME_TUPLE,
                           config.CA_FILE_ENV}


def _example_names() -> list[str]:
    """Return every name .env.example sets, commented out or not."""
    return [entry.name for entry in config.env_template_entry_list()]


def test_the_example_file_names_only_settings_that_are_read():
    unknown = sorted(set(_example_names()) - set(_setting_names()) - _READ_BY_THE_DEPENDENCY)
    assert not unknown, f"nothing reads these names in .env.example: {unknown}"


def test_the_example_file_holds_the_least_that_has_to_be_set():
    assert {"LLM_API_KEY", "PROJECT_DIR", "PRESET"} <= set(_example_names())


def test_the_setting_names_were_actually_found():
    """An empty list would make the check above pass over nothing."""
    names = _setting_names()
    assert "DEF_K" in names and "EDIT_GUIDE" in names and len(names) > 50


# The shape of a call an instruction tells the model to make: an all-capitals name
# followed by "(". The calls that end a run are put in the REPL from rlm.END_CALL_DICT, so a
# name typed into a text by hand is a name the REPL does not answer to
_CALL_IN_A_TEXT = re.compile(r"\b[A-Z][A-Z_]{2,}(?=\()")


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_every_call_the_texts_name_is_one_the_repl_answers_to(ext):
    """Renaming an ending call leaves the old name in the texts nobody thought of.

    The names are read from rlm.END_CALL_DICT, which is also what Repl puts in the
    namespace, so this reads the table rather than a list somebody has to extend.
    """
    for label, text in agent.model_text_dict(ext).items():
        for name in _CALL_IN_A_TEXT.findall(text):
            assert name in rlm.END_CALL_DICT.values(), \
                f"{label} tells the model to call {name}(), which the REPL has no name for"


def test_the_call_scan_finds_the_calls_that_are_there():
    """A scan that matched nothing would pass over every text."""
    found = {name for text in agent.model_text_dict("py").values()
             for name in _CALL_IN_A_TEXT.findall(text)}
    assert found == set(rlm.END_CALL_DICT.values())


@pytest.mark.parametrize("table, name", [
    (LANG_NAME_DICT, "LANG_NAME_DICT"), (PATH_EXAMPLE_DICT, "PATH_EXAMPLE_DICT"),
    (SIGNATURE_EXAMPLE_DICT, "SIGNATURE_EXAMPLE_DICT"), (RUN_EXAMPLE_DICT, "RUN_EXAMPLE_DICT"),
    (DEF_EXAMPLE_DICT, "DEF_EXAMPLE_DICT"),
    (edit._FAIL_RE_DICT, "_FAIL_RE_DICT"), (edit._REPORT_RE_DICT, "_REPORT_RE_DICT"),
])
def test_every_per_language_table_covers_every_language(table, name):
    """A language the table misses is one the model is shown another language's answer
    for - or, for _FAIL_RE_DICT, one whose failing tests come back with no failure names.
    The tables read against a test command's output cover the languages that have one.
    """
    expected = _EXT_WITH_TEST_COMMAND_SET if name.endswith("_RE_DICT") else set(LANG_DICT)
    assert set(table) == expected, f"{name} does not line up with LANG"


def test_lang_covers_every_extension_codetwine_parses():
    """An extension codetwine parses and LANG lacks is one whose repository is indexed
    but tested, run and described as FALLBACK_EXT's language.

    EXT_TO_DEFINITION_DICT is the set codetwine's file walk reads, so what it holds is what
    reaches the index. ext_of() lowers an extension before LANG is looked up.
    """
    assert set(LANG_DICT) == {ext.lower() for ext in EXT_TO_DEFINITION_DICT}, \
        "LANG does not line up with the extensions codetwine parses"


def test_every_language_has_a_captured_run_of_its_test_command():
    """A language with no capture is one whose reporter nobody has ever looked at.

    The patterns that decide whether a run was red are regular expressions written
    against what a build tool prints, and each is matched against a captured run of
    that tool. A wrong pattern either hides a red test or calls a repository without
    tests broken.
    """
    captured = {name.split("_")[0] for name in os.listdir(_CAPTURED_RUNS)}
    for ext in _EXT_WITH_TEST_COMMAND_SET:
        assert ALIAS_DICT.get(ext, ext) in captured, \
            f"no run of the {ext} test command was ever captured"


def test_the_word_count_the_prompt_asks_for_is_the_one_that_is_kept(monkeypatch):
    """The expansion states a range and this is where it is enforced."""
    monkeypatch.setattr(config, "QUERY_EXPAND_WORD_RANGE", "2-3")
    monkeypatch.setattr(config, "QUERY_EXPAND", True)
    asked = {}

    def llm(prompt):
        asked["prompt"] = prompt
        return "one two three four five"

    out, _ = query.expand_query("日本語の依頼", llm)
    assert "2-3 words" in asked["prompt"]
    assert out == "one two three"


def test_a_reply_within_the_word_count_is_left_alone(monkeypatch):
    monkeypatch.setattr(config, "QUERY_EXPAND_WORD_RANGE", "20-40")
    monkeypatch.setattr(config, "QUERY_EXPAND", True)
    assert query.expand_query("日本語の依頼", lambda _: "alpha beta") == ("alpha beta", True)


def test_the_echoed_line_count_and_width_both_come_from_the_settings(monkeypatch):
    """The instructions tell the model to read these lines back, so what comes back has
    to be what they describe.
    """
    monkeypatch.setattr(config, "STAT_ECHO_LINE", 1)
    monkeypatch.setattr(config, "STAT_ECHO_CHAR", 5)
    assert "first 1 lines" in agent.build_instruction("py")
    stat = edit._line_range_stat("a.py", 1, 2, ["abcdefghij\n", "second\n"], "new\n")
    assert "was: abcde" in stat and "second" not in stat


@pytest.mark.parametrize("build, run", [
    (answer_mod.build_answer_instruction,
     lambda: answer_mod.run_answer("where is it read?")),
    (answer_mod.build_combine_instruction,
     lambda: answer_mod.run_combine("where is it read?", [])),
])
def test_the_answer_length_an_answering_stage_states_is_the_one_it_returns(
        build, run, monkeypatch):
    """The instructions name a limit, and the answer the stage returns keeps it.

    The answer is followed out of the stage's run rather than through the cut on its
    own, so the check fails if the stage stops going through it. Both answering stages
    state it, and both are followed.
    """
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 30)
    assert "under 30 characters" in build("py")
    monkeypatch.setattr(answer_mod, "search_scope", lambda request: {
        "files": [], "defs": [], "file_summaries": {}, "callers": {},
        "confidence": "high", "spread": 0})
    reply = "x" * 500
    monkeypatch.setattr(answer_mod, "make_llm",
                        lambda *a, **k: lambda prompt: "```python\n"
                        f"COMPLETE({reply!r})\n```")
    assert len(run()["answer"]) < 500


def test_the_task_length_the_splitting_stage_states_is_the_one_it_hands_over(monkeypatch):
    """The same rule for the other half of what this stage can hand over."""
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 40)
    assert "40 characters each" in dispatch.build_dispatch_instruction("py")
    out = dispatch.parse_dispatch(json.dumps({"tasks": ["y" * 150]}))
    assert len(out["tasks"][0]) < 150


def test_the_task_count_the_splitting_stage_states_is_the_one_it_hands_over(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 2)
    assert "At most 2 entries" in dispatch.build_dispatch_instruction("py")
    out = dispatch.parse_dispatch(json.dumps({"tasks": ["a", "b", "c"]}))
    assert len(out["tasks"]) == 2


def test_the_open_point_count_the_splitting_stage_states_is_the_one_it_hands_over(
        monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_UNCLEAR", 2)
    assert "At most 2 entries" in dispatch.build_dispatch_instruction("py")
    out = dispatch.parse_dispatch(json.dumps({"unclear": ["a?", "b?", "c?"]}))
    assert len(out["unclear"]) == 2


def test_a_reply_wrapped_in_prose_is_read_by_both_stages_that_ask_for_json():
    """One reader, so a fence around the JSON is read the same way wherever it lands
    The explorer's answer is an array and the plan's an object.
    """
    fenced = "Here it is:\n```json\n{body}\n```\nthat is all"
    out = plan.parse_plan(fenced.format(body='{"files": [{"path": "a.py"}]}'))
    assert out["ok"] is True and out["plan"]["files"][0]["path"] == "a.py"
    found = explore.parse_explore_answer(
        fenced.format(body='[{"file": "a.py", "lines": [1, 2], "why": "x"}]'))
    assert found["ok"] is True and found["findings"][0]["file"] == "a.py"


def _module_aliases(tree: ast.Module) -> dict[str, str]:
    """Return the names one file refers to this package's modules by.

    Both import forms are read, so ``from codetwine_loom import config`` and
    ``import codetwine_loom.sandbox as sandbox_mod`` both come back. A name that is
    not a module (``from ...rlm import COMPLETE_KEY``) is in here too; the caller drops it
    when the import fails.

    Args:
        tree: The parsed file.

    Returns:
        {the name used in the file: the module's dotted path}.
    """
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(_PKG):
            for name in node.names:
                aliases[name.asname or name.name] = f"{node.module}.{name.name}"
        elif isinstance(node, ast.Import):
            for name in node.names:
                if name.name.startswith(_PKG):
                    aliases[name.asname or name.name.split(".")[0]] = name.name
    return aliases


def names_not_there(path: pathlib.Path) -> list[str]:
    """Return every ``<module>.<name>`` one file writes that its module does not have.

    Args:
        path: The Python file to read.

    Returns:
        One line per name missing, with its line number and module.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _module_aliases(tree)
    missing = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
            continue
        dotted = aliases.get(node.value.id)
        if not dotted:
            continue
        try:
            module = importlib.import_module(dotted)
        except ImportError:
            continue        # the name is something imported from a module, not a module
        if not hasattr(module, node.attr):
            missing.append(f"line {node.lineno}: {node.value.id}.{node.attr} "
                           f"({dotted} has no such name)")
    return missing


@pytest.mark.parametrize("path", _importing_files(), ids=lambda p: p.name)
def test_every_name_taken_from_a_module_of_this_package_is_there(path):
    """A rename that leaves a caller behind is not a failure until that line runs.

    Every ``<module>.<name>`` written against a module of this package is looked up on
    the module itself, so the caller a rename left behind fails here instead.
    """
    missing = names_not_there(path)
    assert not missing, f"{path.name} names what is not there: {missing}"


# The one file that asks git a question before any setting is read, and so before
# gitcmd.py and config.child_env() can be used: whether git tracks the .env file
_ASKS_BEFORE_THE_SETTINGS = "config.py"


def test_git_is_started_from_one_place_only():
    """A second way of starting git is a second set of arguments to keep in step.
    gitcmd.py is where the arguments and the failure handling live.
    config.py asks its one question about the .env file itself.
    """
    started = [(path.name, line.strip())
               for path in _source_files() if path.name != "gitcmd.py"
               for line in path.read_text(encoding="utf-8").splitlines()
               if '["git"' in line or "['git'" in line]
    assert [name for name, _ in started] == [_ASKS_BEFORE_THE_SETTINGS], \
        f"git is started outside gitcmd.py: {started}"


def test_whether_one_commit_reaches_another_is_asked_in_one_place_only():
    """Three callers ask it - what a merge need not merge again, which branch names may
    go, and whether a branch's work is already checked out - and a second copy is a
    second place to get the argument order the wrong way round. gitcmd.is_ancestor()
    is where it lives.
    """
    asked = [(path.name, line.strip())
             for path in _source_files() if path.name != "gitcmd.py"
             for line in path.read_text(encoding="utf-8").splitlines()
             if "--is-ancestor" in line]
    assert not asked, f"one commit reaching another is asked outside gitcmd.py: {asked}"


# The one file that starts a process which is not the target repository's code: the
# container engine and the broker are this tool's own, and the broker is where the
# credentials have to be
_STARTS_ITS_OWN = "sandbox.py"

# The file that starts this interpreter over one file of this package, under a time
# limit of its own, to read an index a container could write
_READS_ALONE = "store.py"


def _started_programs(path):
    """Yield every subprocess.run / subprocess.Popen call in one source file.

    Args:
        path: The file to read.

    Yields:
        The ast.Call nodes.
    """
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("run", "Popen")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"):
            yield node


def test_a_program_this_one_starts_is_given_the_environment_from_one_place():
    """The test command, the launcher and git all run code the target repository
    supplies. Handing one of them this process's own environment hands it the
    credentials, so what they are given is built in config.child_env() and nowhere else
    (config.project_env() and config.own_child_env() build on it).
    """
    loose = []
    for path in _source_files():
        if path.name in (_STARTS_ITS_OWN, _ASKS_BEFORE_THE_SETTINGS):
            continue
        for node in _started_programs(path):
            given = [k.value for k in node.keywords if k.arg == "env"]
            if not given or not all(
                    isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
                    and v.func.attr in ("child_env", "project_env", "own_child_env")
                    for v in given):
                loose.append((path.name, node.lineno))
    assert not loose, (f"a program is started without config.child_env(): {loose}")
    question_list = [node.lineno for path in _source_files()
                     if path.name == _ASKS_BEFORE_THE_SETTINGS
                     for node in _started_programs(path)]
    assert len(question_list) == 1, question_list


def test_the_scan_finds_the_programs_that_are_started():
    """The check above passes on its own if the scan matches nothing."""
    found = [path.name for path in _source_files()
             for _ in _started_programs(path)]
    assert set(found) >= {"gitcmd.py", "edit.py"}


def test_a_command_the_target_repository_supplies_is_started_in_one_place_only():
    """run_test, run_check and run_project start what the target repository supplies,
    and a command of its own can start children of its own. edit.run_in_project() starts
    it as the leader of a process group, which is what lets a time limit stop all of
    it, so it is the one place outside git and the container engine that starts one.
    """
    started = [(path.name, node.lineno) for path in _source_files()
               if path.name not in (_STARTS_ITS_OWN, _NEEDS_NO_LIMIT,
                                    _ASKS_BEFORE_THE_SETTINGS, _READS_ALONE)
               for node in _started_programs(path)]
    assert len(started) == 1 and started[0][0] == "edit.py", started
    node = next(_started_programs(pathlib.Path(edit.__file__)))
    assert node.func.attr == "Popen"
    assert any(k.arg == "start_new_session" and isinstance(k.value, ast.Constant)
               and k.value.value is True for k in node.keywords)


# The files that split a text which is not the target repository's: the settings
# template, a traceback, and the instruction files, whose every line break the model may
# read as one
_SPLITS_ITS_OWN_TUPLE = ("config.py", "rlm.py", "guide.py")


def test_the_lines_of_the_target_repository_are_split_in_one_place_only():
    """str.splitlines() also breaks at a form feed, U+2028 and the like, which the
    analysis does not count as line breaks, so a line number read one way and used the
    other lands on another line. textcut.line_list_of() is where lines are split.
    """
    split = [(path.name, n) for path in _source_files()
             if path.name not in _SPLITS_ITS_OWN_TUPLE
             for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
             if ".splitlines(" in line]
    assert not split, f"a text is split with str.splitlines(): {split}"


# The one file whose started program needs no time limit: git runs on this machine
# against a repository this process already holds, and a limit on it would end a slow
# commit rather than a hang somebody can act on
_NEEDS_NO_LIMIT = "gitcmd.py"


def test_every_program_this_one_waits_for_is_given_a_time_limit():
    """A wait with no limit is a run nothing ends. The agent's container had none, so a
    container that stopped answering held the run, the agent's lock and, in a
    dispatching run, one of DISPATCH_PARALLEL's slots, for as long as the engine kept it
    alive. Every subprocess.run passes timeout=, whatever setting it reads it from.
    """
    unlimited = []
    for path in _source_files():
        if path.name == _NEEDS_NO_LIMIT:
            continue
        for node in _started_programs(path):
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "run"
                    and not any(k.arg == "timeout" for k in node.keywords)):
                unlimited.append((path.name, node.lineno))
    assert not unlimited, (f"a program is waited for without a time limit: "
                           f"{unlimited}. Pass timeout= from a setting in config.py")


def test_the_atomic_write_is_written_in_one_place_only():
    """The temporary name, the sweep and the move into place belong together: a second
    copy of the sequence is a second place for a leftover to survive and for a failed
    clean-up to be handled differently. tmpfile.py is where it lives.
    """
    replaced = [(path.name, line.strip())
                for path in _source_files() if path.name != "tmpfile.py"
                for line in path.read_text(encoding="utf-8").splitlines()
                if "os.replace(" in line]
    assert not replaced, f"a file is moved into place outside tmpfile.py: {replaced}"


def test_the_name_of_a_temporary_file_is_built_in_one_place_only():
    """The sweep that removes the leftovers of killed writes has to know the name the
    writers build, so a second way of building it would leave those files behind.
    tmpfile.py is where the name lives.
    """
    built = [(path.name, line.strip())
             for path in _source_files() if path.name != "tmpfile.py"
             for line in path.read_text(encoding="utf-8").splitlines()
             if ".tmp" in line and "os.getpid()" in line]
    assert not built, f"a temporary name is built outside tmpfile.py: {built}"


def test_which_paths_hold_the_tests_is_decided_in_one_place_only():
    """Its readers - the run saying which test files it changed among them - answer the
    same for a language that names no test directory. paths.under_test_path() is where
    the answer lives.
    """
    asked = [(path.name, line.strip())
             for path in _source_files() if path.name != "paths.py"
             for line in path.read_text(encoding="utf-8").splitlines()
             if "TEST_PATH" in line and "startswith" in line]
    assert not asked, f"a path is placed under TEST_PATH outside paths.py: {asked}"


def test_no_pattern_for_a_test_run_is_written_twice():
    """A language whose build tool prints another's lines is registered as sharing it,
    not written out again, so a fix to one cannot leave the other behind.

    The check is on the source rather than on the compiled tables: re.compile hands back
    the same object for the same pattern, so two copies cannot be told apart afterwards.
    A language added by copying fails here; adding it to _SAME_REPORTER_DICT (or to a
    language's aliases) is what passes.
    """
    tree = ast.parse(pathlib.Path(edit.__file__).read_text(encoding="utf-8"))
    written = [node.args[0].value for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "compile"
               and node.args and isinstance(node.args[0], ast.Constant)]
    twice = {p for p in written if written.count(p) > 1}
    assert not twice, f"the same pattern is written more than once: {twice}"


def _every_refusal_of_a_budget_tool(monkeypatch) -> list[tuple[str, dict]]:
    """Return one refusal from every failure path of the three tools with a budget.

    Args:
        monkeypatch: pytest's, for the settings and the stand-in model.

    Returns:
        [(what the path is, the dict the tool returned), ...].
    """
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 0)
    monkeypatch.setattr(config, "SEARCH_AGAIN_MAX_CALL", 0)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    refusals = [
        ("explore: no calls left", explore.explore("error handling")),
        ("explore: out of turns", explore.parse_explore_answer(None)),
        ("explore: not JSON", explore.parse_explore_answer("nothing json-shaped here")),
        ("search_again: no calls left", source.search_again("config loading")),
        ("web_search: the question is too long",
         web.web_search("x" * (config.WEB_SEARCH_QUESTION_MAX_CHAR + 1))),
    ]
    monkeypatch.setattr(web, "_search", lambda question: [])
    refusals.append(("web_search: nothing found", web.web_search("what changed")))
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    refusals.append(("web_search: turned off", web.web_search("what changed")))

    def raising(question):
        raise RuntimeError("the search request failed")

    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    monkeypatch.setattr(web, "_search", raising)
    refusals.append(("web_search: the search failed",
                     web.web_search("what changed")))
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    refusals.append(("web_search: no calls left", web.web_search("what changed")))
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    refusals.append(("web_fetch: no search returned it", web.web_fetch("what changed")))
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 0)
    refusals.append(("web_fetch: turned off", web.web_fetch("https://a.invalid/")))
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    monkeypatch.setattr(web, "_fetch", raising)
    web._result_url_set.add(websearch.url_key("https://a.invalid/"))
    refusals.append(("web_fetch: the read failed", web.web_fetch("https://a.invalid/")))
    refusals.append(("web_fetch: no calls left", web.web_fetch("https://a.invalid/")))
    return refusals


def test_a_tool_with_a_budget_says_how_much_is_left_on_every_answer(monkeypatch):
    """A key that is only there on success costs the model the turn it first reads it
    without checking "ok". Every failure path is driven here, so a new
    refusal written without the budget fails.
    """
    refusals = _every_refusal_of_a_budget_tool(monkeypatch)
    assert len(refusals) == 13
    for what, answer in refusals:
        assert answer["ok"] is False, what
        assert isinstance(answer.get("calls_left"), int), f"{what}: no calls_left"
        assert answer["calls_left"] >= 0, what


def test_the_plan_json_budget_holds_for_a_handed_over_dict(monkeypatch):
    """The instructions ask for a JSON under N characters. The stage is told it may
    hand the dict itself over, and that path keeps the limit too.
    """
    monkeypatch.setattr(config, "PLAN_ANSWER_MAX_CHAR", 400)
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 8)
    monkeypatch.setattr(config, "PLAN_MAX_FACT", 12)
    handed_over = {
        "files": [{"path": f"src/f{i}.py", "change": "x" * 40, "anchor": "load_config"}
                  for i in range(8)],
        "order": [f"src/f{i}.py" for i in range(8)],
        "interfaces": ["y" * 40 for _ in range(8)],
        "facts": ["z" * 40 for _ in range(12)],
        "verify": "w" * 200,
    }
    out = plan.parse_plan(handed_over)
    assert out["ok"] is True
    import json
    assert len(json.dumps(out["plan"], ensure_ascii=False)) <= 400
    # What survives is the part the editing stage cannot work without
    assert out["plan"]["files"]


def test_the_plan_keys_that_go_into_every_prompt_are_all_defended(monkeypatch):
    """No plan key that goes into every prompt reaches the model as a repr of a dict."""
    monkeypatch.setattr(config, "PLAN_ANSWER_MAX_CHAR", 6000)
    out = plan.parse_plan({
        "files": [{"path": "src/a.py", "change": "c", "anchor": "load_config"}],
        "order": [{"nested": 1}, "src/a.py"],
        "interfaces": [["nested"], "one line"],
        "facts": [{"nested": 1}, "a fact\nsplit over lines"],
    })
    assert out["plan"]["order"] == ["src/a.py"]
    assert out["plan"]["interfaces"] == ["one line"]
    assert out["plan"]["facts"] == ["a fact split over lines"]


def _refactor_build(stage: str):
    """Return a build(ext) for one refactoring stage's instructions."""
    from codetwine_loom import refactor
    return lambda ext: refactor.build_refactor_instruction(ext, stage)


def _limit_case_list() -> list[tuple[str, tuple[str, ...], object]]:
    """Return (setting, the phrases that state it, build(ext)) for every instruction
    that states a limit whose 0 is no limit."""
    from codetwine_loom import integrate, testcreate
    output_phrase_tuple = ("; only the first {limit}\ncharacters come back",
                           ", only the first {limit} characters come back")
    output_build_list = [dispatch.build_dispatch_instruction,
                         answer_mod.build_answer_instruction,
                         answer_mod.build_combine_instruction,
                         plan.build_plan_instruction, explore.build_explore_instruction,
                         testcreate.build_test_create_instruction,
                         testcreate.build_test_case_instruction,
                         integrate.build_integrate_instruction,
                         _refactor_build(integrate.REFACTOR_STAGE),
                         _refactor_build(integrate.TEST_REFACTOR_STAGE)]
    return ([("RLM_MAX_OUTPUT_CHAR", output_phrase_tuple, build)
             for build in output_build_list]
            + [("DISPATCH_TASK_MAX_CHAR", (", {limit} characters each",),
                dispatch.build_dispatch_instruction),
               ("PLAN_ANSWER_MAX_CHAR",
                ("- Keep the whole JSON under {limit} characters.\n",),
                plan.build_plan_instruction),
               ("EXPLORE_ANSWER_MAX_CHAR",
                ("- Keep the whole JSON under {limit} characters.\n",),
                explore.build_explore_instruction),
               ("DISPATCH_ANSWER_MAX_CHAR", ("- Keep it under {limit} characters.",),
                answer_mod.build_answer_instruction),
               ("DISPATCH_ANSWER_MAX_CHAR", ("- Keep it under {limit} characters.",),
                answer_mod.build_combine_instruction)])


@pytest.mark.parametrize("setting, phrase_tuple, build", _limit_case_list())
def test_a_limit_that_is_off_is_left_out_and_nothing_else_changes(setting, phrase_tuple,
                                                                 build, monkeypatch):
    """0 is no limit for the cut and the guard, so the text names no limit then; with
    the limit on, the text is the one without it plus the phrase that states it."""
    limit = getattr(config, setting)
    text = build("py")
    phrase_list = [p.format(limit=limit) for p in phrase_tuple
                   if p.format(limit=limit) in text]
    assert len(phrase_list) == 1
    monkeypatch.setattr(config, setting, 0)
    off_text = build("py")
    assert off_text == text.replace(phrase_list[0], "")
    assert " 0 characters" not in off_text


def test_the_editing_stage_s_output_limit_line_is_left_out_when_it_is_off(monkeypatch):
    """The editing stage states the turn's output limit in a bullet of its own."""
    monkeypatch.setattr(config, "READ_MAX_CHAR", 1000)
    limit = config.RLM_MAX_OUTPUT_CHAR
    bullet = (f"- Only the first {limit} characters you print each turn come back to "
              "you.\n  The rest is replaced by a \"... (N chars omitted)\" marker and "
              "you never see it. Re-running\n  the same print does NOT show you the "
              "missing part.\n")
    text = agent.build_instruction("py")
    assert bullet in text
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 0)
    assert agent.build_instruction("py") == text.replace(bullet, "").replace(
        ", which is far less than the limit", "")


def test_a_feature_that_is_off_is_not_described_to_the_model(monkeypatch):
    """"You get 0 rounds" reads as a limit already spent, and a tool that can only
    refuse costs a turn to find that out.
    """
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 0)
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 0)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 0)
    text = agent.build_instruction("py")
    assert "edit-then-test rounds" not in text
    assert "call explore" not in text
    assert "web_search" not in text and "web_fetch" not in text
    assert "outside this repository" not in text
    assert explore.explore not in agent.main_tool_list()
    assert web.web_search not in agent.main_tool_list()
    assert web.web_fetch not in agent.main_tool_list()


class _IndexWith:
    """Stand in for the loaded index, answering only whether it holds summaries."""

    def __init__(self, summaries: bool):
        self.summaries = summaries

    def has_summary(self) -> bool:
        return self.summaries


_STAGES_THAT_READ_THE_SCOPE = [agent.build_instruction, plan.build_plan_instruction,
                               dispatch.build_dispatch_instruction,
                               explore.build_explore_instruction]


@pytest.mark.parametrize("build", _STAGES_THAT_READ_THE_SCOPE)
def test_an_index_without_summaries_does_not_send_the_model_to_read_them(build,
                                                                        monkeypatch):
    """A build with the documents off leaves scope["file_summaries"] empty, and a
    stage told to read it first spends its first turn printing an empty dict."""
    monkeypatch.setattr(scope_mod, "INDEX", _IndexWith(summaries=False))
    text = build("py")
    assert "file_summaries" not in text
    assert 'the "signature" of each scope["defs"] entry' in text


@pytest.mark.parametrize("build", _STAGES_THAT_READ_THE_SCOPE)
def test_an_index_with_summaries_is_read_as_it_always_was(build, monkeypatch):
    monkeypatch.setattr(scope_mod, "INDEX", _IndexWith(summaries=True))
    with_index = build("py")
    assert 'scope["file_summaries"] and the "signature"' in with_index
    # With no index loaded at all the stages say the same thing
    monkeypatch.setattr(scope_mod, "INDEX", None)
    assert build("py") == with_index


def test_the_instructions_never_name_a_tool_that_was_not_handed_over(monkeypatch,
                                                                    tmp_path):
    """The general form of the check above: the text and the tool list agree.

    It reads main_tool_list() rather than a list to keep up to date, so a tool switched off
    in a new way fails here instead of costing a turn in a run.
    """
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 0)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    # A repository with no tests: run_test could only refuse
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    handed = {tool.__name__ for tool in agent.main_tool_list()}
    absent = set(agent.TOOL_RESULT_KEY_DICT) - handed
    assert absent, "nothing is switched off, so this would pass on an empty set"
    text = agent.build_instruction("py")
    for name in absent:
        assert name not in text


def _absent_tool_name_list(text: str, tool_list: list, all_name_set: set[str]) -> list[str]:
    """Return the names of the tools a text names that are not among the ones handed."""
    hand_set = {tool.__name__ for tool in tool_list}
    return sorted(name for name in all_name_set - hand_set
                  if re.search(rf"\b{name}\b(?!\w)", text))


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_no_language_s_instructions_name_a_tool_the_stage_is_not_handed(ext, monkeypatch):
    """Every extension, with each tool a setting takes away on and off: the editing
    stage over every combination, and the stages after it over the tests alone, which
    is what their tool lists follow."""
    all_name_set = {tool.__name__ for tool in agent.all_tool_list()}
    all_name_set |= {"run_test", "run_project", "explore", "web_search", "web_fetch"}
    monkeypatch.setattr(config, "PRIMARY_EXT", ext)
    monkeypatch.setattr(websearch, "can_search", lambda: True)
    monkeypatch.setattr(websearch, "can_fetch", lambda: True)
    monkeypatch.setattr(config, "RUN_COMMAND", ["launcher"])
    build_count = 0
    for can_test, explore_max, run_max, search_max, fetch_max in itertools.product(
            (True, False), (3, 0), (3, 0), (5, 0), (5, 0)):
        monkeypatch.setattr(edit, "test_can_run", lambda value=can_test: value)
        monkeypatch.setattr(config, "EXPLORE_MAX_CALL", explore_max)
        monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", run_max)
        monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", search_max)
        monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", fetch_max)
        assert _absent_tool_name_list(agent.build_instruction(ext), agent.main_tool_list(),
                                      all_name_set) == []
        build_count += 1
    assert build_count == 32
    for can_test in (True, False):
        monkeypatch.setattr(edit, "test_can_run", lambda value=can_test: value)
        assert _absent_tool_name_list(integrate.build_integrate_instruction(ext),
                                      integrate.merge_tool_list(), all_name_set) == []
    # The stages after the merge call a model only where the tests run
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    assert _absent_tool_name_list(testcreate.build_test_create_instruction(ext),
                                  testcreate.test_create_tool_list(), all_name_set) == []
    for stage in refactor.STAGE_TUPLE:
        assert _absent_tool_name_list(refactor.build_refactor_instruction(ext, stage),
                                      refactor.stage_tool_list(stage), all_name_set) == []


def test_a_piece_of_a_language_s_note_that_names_a_tool_is_tied_to_that_tool():
    """What the table holds for the stages that write code names, outside the pieces
    tied to a tool, only tools every such stage is handed whatever the settings."""
    always_set = {tool.__name__ for tool in integrate.integrate_tool_list()} - {"run_test"}
    all_name_set = {tool.__name__ for tool in agent.all_tool_list()} | {
        "run_test", "run_project", "explore", "web_search", "web_fetch"}
    for ext, lang_config in LANG_DICT.items():
        for tool_name, text in lang_config.edit_note_tuple:
            name_set = {name for name in all_name_set
                        if re.search(rf"\b{name}\b(?!\w)", text)}
            assert name_set <= always_set | ({tool_name} if tool_name else set()), ext
        # With every tool handed, the note is every piece of it
        assert edit_note(ext, all_name_set) == "".join(
            text for _, text in lang_config.edit_note_tuple)


def test_the_note_on_c_tests_says_what_run_test_does_only_to_a_stage_handed_it():
    with_tool = edit_note("c", {"run_test"})
    assert with_tool.startswith(
        "## C and C++ tests\n\n- run_test() configures and builds the CMake project and "
        "runs its CTest tests, or, with no CMakeLists.txt, runs make's test target (check "
        "when there is none). A test\n  runs only when CMakeLists.txt builds its source "
        "file (add_executable) and registers\n")
    assert "do not add a CMakeLists.txt" in " ".join(with_tool.split())
    without_tool = edit_note("cpp", set())
    assert "run_test" not in without_tool
    assert without_tool == with_tool.replace(
        "run_test() configures and builds the CMake project and runs its CTest tests, or, "
        "with no CMakeLists.txt, runs make's test target (check when there is none). ", "")
    assert edit_paragraph("c", {"run_test"}) == "\n" + with_tool.rstrip("\n") + "\n"


def test_a_repository_without_tests_is_not_told_about_tests(monkeypatch, tmp_path):
    """run_test is the one tool no setting switches off."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 3)
    text = agent.build_instruction("py")
    assert edit.run_test not in agent.main_tool_list()
    assert "## Tests" not in text
    # A round closes in run_test, so without it there is no limit to state
    assert "edit-then-test rounds" not in text


def test_a_repository_with_tests_is_told_about_them(monkeypatch, tmp_path):
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 3)
    text = agent.build_instruction("py")
    assert edit.run_test in agent.main_tool_list()
    assert "## Tests" in text
    assert "You get 3 edit-then-test rounds" in text


def test_a_build_tool_that_finds_its_own_tests_is_left_alone(monkeypatch, tmp_path):
    """Only "path" puts TEST_PATH in argv; the others cannot be told there are none."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "TEST_PATH", "src/test/java")  # not in the repository
    for style in ("none", "separator"):
        monkeypatch.setattr(config, "TEST_TARGET_STYLE", style)
        assert edit.test_can_run() is True
        assert edit.run_test in agent.main_tool_list()


def test_run_tests_switched_off_leaves_the_tool_and_every_line_about_it_out(
        monkeypatch, tmp_path):
    """RUN_TEST off is the shape a repository with no tests already gets."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")  # otherwise always runnable
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 3)
    monkeypatch.setattr(config, "RUN_TEST", False)
    assert edit.test_can_run() is False
    assert edit.run_test not in agent.main_tool_list()
    text = agent.build_instruction("py")
    assert "run_test()" not in text
    assert "edit-then-test rounds" not in text
    assert "There are no tests in this repository to run" in text


def test_run_tests_left_on_is_what_it_has_always_been(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "RUN_TEST", True)
    assert edit.test_can_run() is True
    assert edit.run_test in agent.main_tool_list()


def test_a_feature_that_is_on_is_described_once(monkeypatch):
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 5)
    monkeypatch.setattr(config, "EXPLORE_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 3)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 4)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    text = agent.build_instruction("py")
    assert "You get 5 edit-then-test rounds" in text
    assert "You get\n  2 calls" in text
    assert "You get 3 calls" in text
    assert "You get 4 calls" in " ".join(text.split())
    assert text.count("- web_search \u2192 ") == 1
    assert text.count("- web_fetch \u2192 ") == 1
    assert text.count("## A fact from outside this repository") == 1
    assert explore.explore in agent.main_tool_list()
    assert web.web_search in agent.main_tool_list()
    assert web.web_fetch in agent.main_tool_list()


def test_the_history_is_read_and_written_in_one_place_only():
    """Reading it, adding to it and writing it back is one step, under one lock: a
    second copy of the sequence is a second place for one run's entry to take another
    run's away, which is what runs sharing an output directory do.
    history.open_to_update() is where the sequence lives.
    """
    written = [(path.name, line.strip())
               for path in _source_files() if path.name != "history.py"
               for line in path.read_text(encoding="utf-8").splitlines()
               if "history.save(" in line]
    inside = [node.name for node in ast.walk(
        ast.parse((_PACKAGE / "history.py").read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "save" for call in ast.walk(node))]
    assert not written, f"the history is written outside history.py: {written}"
    assert inside == ["open_to_update"], \
        f"the history is written outside history.open_to_update(): {inside}"


def test_the_porcelain_status_is_read_in_one_place_only():
    """Two readers strip the two-letter prefix off `git status --porcelain`, and a
    second copy of that parsing is a second place for the prefix to be got wrong.
    gitcmd.status_path_list() is where it lives.
    """
    parsed = [(path.name, line.strip())
              for path in _source_files() if path.name != "gitcmd.py"
              for line in path.read_text(encoding="utf-8").splitlines()
              if "--porcelain" in line]
    assert not parsed, f"a porcelain status is read outside gitcmd.py: {parsed}"


def test_the_name_and_kind_of_every_change_is_read_in_one_place_only():
    """Two readers turn `git diff --name-status` into added / modified / deleted - the
    run saying what it has changed and the report saying what a finished run changed -
    and a second copy of that parsing is a second place for git's one-letter code to be
    read wrong. gitcmd.name_status() is where it lives.
    """
    parsed = [(path.name, line.strip())
              for path in _source_files() if path.name != "gitcmd.py"
              for line in path.read_text(encoding="utf-8").splitlines()
              if "--name-status" in line]
    assert not parsed, f"a name-status diff is read outside gitcmd.py: {parsed}"


def _default_nodes(fn):
    """Yield every expression written as a default argument of one function.

    Args:
        fn: An ast.FunctionDef or ast.AsyncFunctionDef.

    Yields:
        The ast nodes, the keyword-only defaults included.
    """
    for default in list(fn.args.defaults) + list(fn.args.kw_defaults):
        if default is not None:
            yield from ast.walk(default)


def test_no_setting_is_frozen_into_a_default_argument():
    """Python evaluates a default argument once, when the module is imported, so
    `max_iters: int = config.RLM_MAX_TURN` freezes the value the environment had at
    import. Everything else reads config as an attribute per call, so a frozen default
    would miss a setting rewritten afterwards - by a caller driving several requests -
    that build_instruction() and cli.print_result() still read, and the instructions
    would state a limit the loop did not apply.
    """
    frozen = []
    for path in _source_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            frozen += [f"{path.name}:{node.name}(...={sub.attr})"
                       for sub in _default_nodes(node)
                       if isinstance(sub, ast.Attribute)
                       and isinstance(sub.value, ast.Name)
                       and sub.value.id == "config" and sub.attr.isupper()]
    assert not frozen, (f"a setting is read into a default argument, which freezes it "
                        f"at import: {frozen}. Default to None and read config inside "
                        f"the function")


def test_the_default_argument_scan_reaches_the_functions_that_have_defaults():
    """The check above passes on its own if the scan matches nothing."""
    seen = [node.name for path in _source_files()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for _ in _default_nodes(node)]
    assert len(seen) > 20 and "__init__" in seen


# What the image is built from. COPY . . puts whatever the repository has grown since -
# documents, reports, an earlier run's output directory - into the build context and
# into the layer, and .dockerignore is a list somebody has to remember to extend
_DOCKERFILE = pathlib.Path(__file__).resolve().parent.parent / "Dockerfile"

# The COPY sources that come from another image rather than from this directory
_NOT_FROM_HERE = "--from="


def _copied_paths() -> list[str]:
    """Return the sources of every COPY in the Dockerfile that reads this directory.

    Returns:
        The paths as they are written, relative to the repository root.
    """
    found = []
    for line in _DOCKERFILE.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != "COPY" or any(
                p.startswith(_NOT_FROM_HERE) for p in parts):
            continue
        found += parts[1:-1]        # the last item is the destination
    return found


def test_the_image_names_what_it_is_built_from():
    """A `COPY . .` sends the whole repository, and this one holds a few hundred MB of
    documents and audio that nothing in the image reads.
    """
    copied = _copied_paths()
    assert copied, "the Dockerfile copies nothing from this directory"
    assert "." not in copied, ("the Dockerfile copies the whole directory. Name what "
                               "the image needs instead")


@pytest.mark.parametrize("copied", _copied_paths())
def test_everything_the_image_copies_is_there(copied):
    """A name that no longer exists fails the build rather than the tests."""
    assert (_DOCKERFILE.parent / copied).exists(), \
        f"the Dockerfile copies {copied}, which is not in the repository"


# The files of the installed litellm that write a client that follows a redirect.
# providercall.py keeps the ones a completion goes through from following one
_FOLLOW_REDIRECT_FILE_SET = {
    "experimental_mcp_client/client.py", "litellm_core_utils/url_utils.py",
    "llms/custom_httpx/http_handler.py", "llms/openai/common_utils.py"}


def test_the_litellm_installed_holds_what_keeps_a_call_from_following_a_redirect():
    """Every function providercall replaces is there in the form it replaces, the
    replacements are in place, and no call is refused for them."""
    import aiohttp
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from openai._base_client import AsyncAPIClient, SyncAPIClient

    from codetwine_loom import providercall

    providercall.check_before_call()
    assert providercall._redirect_block_text == ""
    for handler_class in (HTTPHandler, AsyncHTTPHandler):
        assert hasattr(vars(handler_class)["create_client"], "__wrapped__")
        assert hasattr(vars(handler_class)["client"].fget, "__wrapped__")
    for api_class in (SyncAPIClient, AsyncAPIClient):
        assert hasattr(vars(api_class)["__init__"], "__wrapped__")
    assert hasattr(vars(aiohttp.ClientSession)["_request"], "__wrapped__")


def test_litellm_writes_a_client_that_follows_a_redirect_only_where_it_is_known_to():
    """A file of litellm that starts to write one is looked at before it is used."""
    import litellm

    root = pathlib.Path(litellm.__file__).resolve().parent
    found_set = {path.relative_to(root).as_posix() for path in root.rglob("*.py")
                 if re.search(r"""(follow|allow)_redirects["']?\s*[=,:]\s*True""",
                              path.read_text(encoding="utf-8"))}
    assert found_set == _FOLLOW_REDIRECT_FILE_SET
