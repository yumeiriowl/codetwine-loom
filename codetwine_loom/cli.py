"""Call every subcommand of codetwine-loom from the command line.

build, overview, search, run --here, answer, edit, run-tests, integrate --here,
test-create, refactor and test-refactor go through the same start-up. Every one of them
analyses a repository that has no analysis yet, without the design documents and without
asking; only build --doc-create writes the documents. Every one of them makes the
analysis again, without the documents, when it does not describe the commit and tree on
disk, integrate before it merges the branches; outside a container it stops instead
while a sandboxed run is going over the same output directory. What the analysis keeps
out is decided by the subcommands that take --exclude; the others read it as it was
made. Every one of them that calls a model checks the credentials first while LLM_BROKER
is empty or --doc-create is passed; at a terminal with no .env read, it may ask
questions and write .env in the current directory before going on (_ask_env_file).

setup, history, format, sandbox, run, integrate, warm and broker return before that
start-up: setup reads no target repository and writes .env, history reads one file
beside the index, format reads the working tree alone, sandbox, warm, and run and
integrate without --here do their work in a container rather than here, and broker reads
a repository only when it names one, to fill that repository's package cache. sandbox,
warm, and run and integrate without --here look for their container engine and ask it
for their image before the credentials are checked, and stop when either is not there,
except that the image is not asked for while a credential the run checks is missing;
the image holds the toolchains of every language unless one is named, and one that is
missing or stale may be built after a question at a terminal. A
subcommand stops when its repository is not a directory, and, format aside, when its
output directory is inside the repository or holds it; setup is not checked, and broker
only when it names a repository.
"""

import argparse
import contextlib
import functools
import importlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import NoReturn

# config is imported ahead of the modules below
from codetwine_loom import SETUP_COMMAND, SIGNAL_STATUS_BASE, config, keep_cwd_off_path
from codetwine_loom import answer as answer_mod
from codetwine_loom import broker as broker_mod
from codetwine_loom import dispatch as dispatch_mod
from codetwine_loom import image as image_mod
from codetwine_loom import integrate as integrate_mod
from codetwine_loom import knowledge as knowledge_mod
from codetwine_loom import refactor as refactor_mod
from codetwine_loom import report as report_mod
from codetwine_loom import (gitcmd, history, llm, overview, progress,
                            runrecord, tmpfile, tokencount, turnlog)
from codetwine_loom import sandbox as sandbox_mod
from codetwine_loom import testcreate as test_create_mod
from codetwine_loom.agent import check_start, create_agent, run_edit
from codetwine_loom.paths import list_text
from codetwine_loom.rlm import no_answer_reason
from codetwine_loom.textcut import command_line, error_part, one_line, with_command
from codetwine_loom.index import build, store
from codetwine_loom.knowledge import Knowledge, ensure_knowledge, exclude_of_analysis
from codetwine_loom.languages import LANG_NAME_DICT
from codetwine_loom.search import scope as scope_mod
from codetwine_loom.search.scope import search_scope
from codetwine_loom.tools import edit, source

logger = logging.getLogger(__name__)

# The subcommands a container runs that take the language the host hands it
_HOST_LANGUAGE_COMMAND_TUPLE = (integrate_mod.TEST_CREATE_STAGE, integrate_mod.REFACTOR_STAGE,
                                integrate_mod.TEST_REFACTOR_STAGE, "integrate")

# The loggers of the libraries called here, by the names the libraries give them
_LIBRARY_LOGGER_TUPLE = ("LiteLLM",)

# The subcommands that start containers from this machine when not given --here
_CONTAINER_START_COMMAND_TUPLE = ("sandbox", "integrate", "run", "warm")


# The settings that hold the key and the address of the model calls
_MODEL_KEY_SETTING = "LLM_API_KEY"
_MODEL_BASE_SETTING = "LLM_API_BASE"

# What the first run's questions say above a provider's own variable
_PROVIDER_VARIABLE_HELP = "The provider's own variable, written to .env under its name"

# What the first run's questions ask before the settings of _PROXY_SETTING_TUPLE
_PROXY_QUESTION = ("Is the internet (the model's API, the web APIs, the package "
                   "registries) reached through a proxy?")

# The setting the proxy's URL is written under, and the one naming the hosts reached
# without it
_PROXY_URL_SETTING = "HTTPS_PROXY"
_NO_PROXY_SETTING = "NO_PROXY"

# The settings asked for after a y to _PROXY_QUESTION, in the order they are asked
_PROXY_SETTING_TUPLE = (_PROXY_URL_SETTING, _NO_PROXY_SETTING, config.CA_FILE_ENV)

# The answer that takes out the value a setting holds in .env
_CLEAR_ANSWER = "-"

# The answers, in lower case, that say yes to a question answered with y or n
_YES_ANSWER_TUPLE = ("y", "yes")

# What a question shows in place of the value of a key .env holds, and in place of the
# password in a URL it holds
_SECRET_MARK = "set"
_PASSWORD_MARK = "****"

# The scheme a URL starts with, with "://" after it
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")

# What the questions for a .env not yet there say Enter does
_NEW_KEEP_TEXT = ("Enter keeps the value shown, and a line left as it is stays "
                  "commented out")

# Why a .env was not written when the input ends before the last question
_INPUT_END_TEXT = "the input ended before the last question"

# The subcommands whose request is what they search for or carry out. integrate takes
# one as well, but as the account of what the branches were made for
_REQUEST_COMMAND_TUPLE = ("search", "edit", "run", "answer", "sandbox")

# The subcommands that search in this process once the index is read
_SEARCH_COMMAND_TUPLE = ("search", "edit", "answer", "run")

# The subcommands that take --base, the commit the change HEAD holds was written against
_BASE_COMMAND_TUPLE = ("test-create", *refactor_mod.STAGE_TUPLE)


def print_build(index_dict: dict, index_path: str, knowledge_path: str,
                is_reuse: bool) -> None:
    """Print build's 6 lines.

    primary_ext is read from config, which load_index() filled in.

    Args:
        index_dict: The return value of build_index().
        index_path: The path of the index that was written.
        knowledge_path: The knowledge file path ensure_knowledge() returned.
        is_reuse: Whether an existing knowledge file was used as it was.
    """
    bm25_dict, file_bm25_dict = index_dict["index"]["def"], index_dict["index"]["file"]
    print(f"knowledge: {knowledge_path} "
          f"({'reused existing' if is_reuse else 'ran codetwine'})")
    print(f"defs: {len(index_dict['nodes']['defs'])} files: {len(index_dict['nodes']['files'])} "
          f"calls: {len(index_dict['edges']['calls'])}")
    print(f"terms: {len(bm25_dict['postings'])} avg_len: {bm25_dict['avg_len']:.1f}")
    print(f"file terms: {len(file_bm25_dict['postings'])} "
          f"file avg_len: {file_bm25_dict['avg_len']:.1f}")
    print(f"primary_ext: {config.PRIMARY_EXT}")
    print(f"index: {index_path} ({os.path.getsize(index_path) / 1024:.1f} KB)")


def _print_overview(result_dict: dict) -> None:
    """Print what making the overview did.

    Args:
        result_dict: The return value of overview.create().
    """
    print(f"overview: {result_dict['path']} ({result_dict['status']})")


def refuse_while_run_live(command: str, project: str, output_dir: str) -> None:
    """Stop a subcommand that would replace the analysis while a sandboxed run goes.

    The runs are looked for with sandbox.refuse_while_run_live().

    Args:
        command: The subcommand's name, for the message.
        project: The target repository path, as it was passed.
        output_dir: The output directory, as it was passed.

    Raises:
        SystemExit: With the runs that are going, when there are any.
    """
    try:
        sandbox_mod.refuse_while_run_live(output_dir, knowledge_mod.project_name_of(project))
    except RuntimeError as e:
        raise SystemExit(f"{command}: {e}") from None


def analysis_match_tree(project_dir: str, index_path: str, source_stamp: str,
                        exclude_list: list[str]) -> bool:
    """Say whether the analysis on disk describes the tree on disk.

    What is compared is the value of what an analysis of the tree reads now
    (knowledge.analysis_state()), not the commit: an analysis made over a tree that has
    changes is used again while those changes are what is on disk. An analysis with no
    record beside it, and one whose record names another codetwine than the one
    installed now (knowledge.same_codetwine()), describe no tree.

    Args:
        project_dir: The target repository, already normalised.
        index_path: The path to code_index.sqlite, beside the analysis.
        source_stamp: The analysis on disk (Knowledge.stamp()).
        exclude_list: The extra names this run keeps out of the analysis.

    Returns:
        True when the analysis was made by the codetwine installed now, describes the
        tree on disk, keeps out the same names and read its files with the encodings
        this run reads with. False otherwise, saying why in the log.
    """
    knowledge_path = knowledge_mod.knowledge_db_path_of_index(index_path)
    now_state = knowledge_mod.analysis_state(project_dir, exclude_list)
    record_dict = knowledge_mod.analysis_record(knowledge_path, source_stamp)
    if not record_dict:
        logger.info("nothing beside the analysis says what it describes, so it is made "
                    "again (no LLM is called)")
        return False
    if not knowledge_mod.same_codetwine(record_dict):
        logger.info("the analysis was made by codetwine %s and codetwine %s is "
                    "installed, so it is made again (no LLM is called)",
                    record_dict.get("codetwine") or "of an unknown version",
                    knowledge_mod.codetwine_id() or "of an unknown version")
        return False
    if not now_state or record_dict.get("state") != now_state:
        logger.info("the analysis describes %s and the repository holds %s, so it is "
                    "made again (no LLM is called)",
                    gitcmd.short(str(record_dict.get("state") or "")) or "no state",
                    gitcmd.short(now_state) or "a state git cannot tell")
        return False
    if not knowledge_mod.same_exclude(record_dict, exclude_list):
        logger.info("the analysis was made with --exclude %s and this run keeps out %s, "
                    "so it is made again (no LLM is called)",
                    record_dict.get("exclude", "unknown"), list(exclude_list))
        return False
    if not knowledge_mod.same_encoding(record_dict):
        logger.info("the analysis was made with SOURCE_ENCODING %s and this run reads "
                    "with %s, so it is made again (no LLM is called)",
                    record_dict.get("encoding"), list(config.SOURCE_ENCODING))
        return False
    return True


def union_analysed(project_dir: str, knowledge_path: str, project_name: str,
                   output_dir: str, exclude_list: list[str]) -> bool:
    """Say whether the analysis on disk already describes the union in the working tree.

    Args:
        project_dir: The target repository, holding the union.
        knowledge_path: The path to project_knowledge.sqlite.
        project_name: The repository's directory name.
        output_dir: Where the analysis is written.
        exclude_list: The extra names this run keeps out of the analysis.

    Returns:
        analysis_match_tree()'s answer, False when there is no analysis to read.
    """
    if not os.path.exists(knowledge_path):
        return False
    knowledge = Knowledge.open(knowledge_path)
    try:
        source_stamp = knowledge.stamp()
    finally:
        knowledge.close()
    return analysis_match_tree(project_dir, store.index_db_path(output_dir, project_name),
                               source_stamp, exclude_list)


def print_scope(scope_dict: dict) -> None:
    """Print search_scope()'s return value in blocks.

    The search terms come first: a request that was rewritten shows what it was searched
    with. A definition takes two lines: its line range and its first line.

    Args:
        scope_dict: The return value of search_scope().
    """
    trace = scope_dict["trace"]
    print(f"query: {trace['query']}")
    if scope_dict.get("note"):
        print(scope_dict["note"])
    print(f"confidence: {scope_dict['confidence']} "
          f"(raw {trace['top_raw']:.1f} / hit_terms {trace['top_hit_terms']} "
          f"/ spread {scope_dict['spread']})")
    print("files:")
    for file_path in scope_dict["files"]:
        print(f"  {file_path}")
    print("defs:")
    for def_dict in scope_dict["defs"]:
        start, end = def_dict["lines"]
        print(f"  {def_dict['key']}  [{start}-{end}]")
        print(f"    {def_dict['signature']}")
    if scope_dict.get("expanded"):
        print("expanded:")
        for def_dict in scope_dict["expanded"]:
            start, end = def_dict["lines"]
            print(f"  {def_dict['key']}  [{start}-{end}]  <- {def_dict['from']}")
    print("file_summaries:")
    for path, summary in scope_dict["file_summaries"].items():
        print(f"  {path}")
        print(f"    {summary}")
    print("callers:")
    for key, caller_list in scope_dict["callers"].items():
        if not caller_list:
            continue  # definitions with no caller at all are left out
        print(f"  {key}")
        for caller in caller_list:
            print(f"    <- {caller}")


def no_answer_line(result_dict: dict, max_turn: int, call: str) -> str:
    """Return the line for a run that handed nothing over, saying which of the two it is.

    Which of the two it is is decided in rlm.no_answer_reason().

    Args:
        result_dict: The return value of the stage, holding "error" when the loop stopped
            early and "outcome" when the stage ended itself. A result with neither key
            is read as a run that did neither.
        max_turn: That stage's turn limit, for the wording of a run that used them up.
        call: The call that would have ended the run, e.g. "COMPLETE".

    Returns:
        The single "answer: none (...)" line.
    """
    reason = no_answer_reason(result_dict.get("error") or "", max_turn, call,
                              result_dict.get("outcome"))
    return f"answer: none ({reason})"


# The lists the splitting stage can hand over, in the key the result holds them under
_SPLIT_KEY_TUPLE = dispatch_mod.SPLIT_KEY_TUPLE + (dispatch_mod.UNCLEAR_KEY,)


def _send_back_mark(send_back_count: int) -> str:
    """Return the mark put after the turns line when the ending call was sent back.

    Args:
        send_back_count: A stage's "refused_completes": how many times it was.

    Returns:
        ", N sent back", or "" when nothing was.
    """
    return f", {send_back_count} sent back" if send_back_count else ""


def _late_answer_mark(result_dict: dict) -> str:
    """Return the mark put after a line whose answer came from the one extra call.

    Args:
        result_dict: A stage's result, holding "harvested".

    Returns:
        The mark, or "" for an answer that came through COMPLETE().
    """
    return (" (harvested: the turns ran out before COMPLETE())"
            if result_dict.get("harvested") else "")


def print_split(result_dict: dict) -> None:
    """Print what the splitting stage decided.

    That is the questions or the tasks it split into, or what it found the request
    leaves open.

    Args:
        result_dict: The return value of dispatch_mod.run_dispatch().
    """
    print()
    print(f"iters: {result_dict['iters']} / {config.DISPATCH_MAX_TURN}"
          f"{_send_back_mark(result_dict.get('refused_completes', 0))}")
    late_mark = _late_answer_mark(result_dict)
    if not result_dict["ok"]:
        print(f"split: none{late_mark}")
        print(f"  {result_dict['error']}")
        return
    key = next(name for name in _SPLIT_KEY_TUPLE if name in result_dict)
    print(f"{key}: {len(result_dict[key])}{late_mark}")
    for number, entry_dict in enumerate(result_dict[key], 1):
        print(f"  {number}. {entry_dict}")


def print_answer(result_dict: dict) -> None:
    """Print what an answering or a combining stage handed over.

    Args:
        result_dict: The return value of answer_mod.run_answer() or run_combine().
    """
    print()
    max_turn = result_dict.get("max_turn")
    iters_text = f"{result_dict['iters']}" + (f"/{max_turn}" if max_turn else "")
    print(f"iters: {iters_text}"
          f"{_send_back_mark(result_dict.get('refused_completes', 0))}")
    if not result_dict["ok"]:
        print("answer: none")
        print(f"  {result_dict['error']}")
        return
    print(f"answer:{_late_answer_mark(result_dict)}")
    print(f"  {result_dict['answer']}")


def _write_out(path: str, what: object, label: str) -> bool:
    """Write one file with sandbox.write_across().

    Args:
        path: Where to write it.
        what: The object to write.
        label: What the file holds, for the log line.

    Returns:
        True when the file was written. False when it could not be, which is logged as
        an error naming the path.
    """
    try:
        sandbox_mod.write_across(path, what, label)
    except OSError as e:
        logger.error("%s could not be written to %s (%s)", label, path, e)
        return False
    return True


def write_split(path: str, result_dict: dict) -> bool:
    """Write what the splitting stage decided.

    Args:
        path: Where to write it.
        result_dict: The return value of dispatch_mod.run_dispatch().

    Returns:
        Whether the file was written (_write_out).
    """
    split_dict = {key: result_dict[key] for key in _SPLIT_KEY_TUPLE if key in result_dict} \
        or {"error": result_dict.get("error", "")}
    if "tasks" in split_dict and "defs" in result_dict:
        split_dict["defs"] = result_dict["defs"]
    return _write_out(path, split_dict, "the split")


def write_answer(path: str, result_dict: dict) -> bool:
    """Write what an answering or a combining stage handed over.

    Args:
        path: Where to write it.
        result_dict: The return value of answer_mod.run_answer() or run_combine().

    Returns:
        Whether the file was written (_write_out).
    """
    write_dict = {"answer": result_dict["answer"]} if result_dict["ok"] \
        else {"error": result_dict["error"]}
    return _write_out(path, write_dict, "the answer")


def read_answer_list(path: str) -> list[dict]:
    """Read the agents' answers from a file.

    Args:
        path: The file, as --answered names it.

    Returns:
        The entries, as answer.render_answer_list() takes them.

    Raises:
        SystemExit: When the file is not there, does not hold a list, or holds an entry
            of another shape.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            answer_list = json.load(handle)
    except (OSError, ValueError, RecursionError) as e:
        raise SystemExit(f"answer: --answered {path} cannot be read ({e})") from None
    if not isinstance(answer_list, list):
        raise SystemExit(f"answer: --answered {path} does not hold a list of answers")
    # The shape of every entry is checked before the run starts
    for i, entry_dict in enumerate(answer_list):
        if not isinstance(entry_dict, dict) or not isinstance(entry_dict.get("n"), int) \
                or not isinstance(entry_dict.get("request"), str):
            raise SystemExit(f"answer: --answered {path} entry {i} is not "
                             f"{{n, request, answer, error}}")
    return answer_list


def recheck_gap(path: str) -> dict:
    """Run the tests the cases in a file were recorded with again, over the tree as it is.

    Args:
        path: The file, as --gap-in names it: record_gap()'s entries.

    Returns:
        {"gap": the cases still not done, "gap_done": the ones whose test passes now},
        or {} when the file cannot be read or holds no list.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            gap_list = json.load(handle)
    except (OSError, ValueError, RecursionError) as e:
        logger.warning("the cases to run again could not be read from %s (%s), so none "
                       "is run", path, e)
        return {}
    if not isinstance(gap_list, list):
        return {}
    keep_list, pass_list = integrate_mod.recheck_gap_list(
        [entry_dict for entry_dict in gap_list if isinstance(entry_dict, dict)])
    return {"gap": keep_list, "gap_done": pass_list}


def _verdict_dict(result_dict: dict) -> dict:
    """Return a stage's verdict, what the gates still find and what the run said.

    Args:
        result_dict: The return value of integrate_mod.run_integration(),
            test_create_mod.run_test_create() or refactor_mod.run_refactor().

    Returns:
        {"verdict": integrate_mod.SETTLE_VERDICT or UNSETTLE_VERDICT, "why": what the
        gates still find, joined with " / ", "did": what the run said it did}.
    """
    return {
        "verdict": integrate_mod.SETTLE_VERDICT if result_dict["settled"]
        else integrate_mod.UNSETTLE_VERDICT,
        # What the gates still find; empty for a run that settled
        "why": " / ".join(result_dict["left"]),
        # What the run said it did; empty for a run that called no model
        "did": result_dict["answer"] if result_dict["called"] else "",
    }


def write_outcome(path: str, result_dict: dict, gap_dict: dict | None = None,
                  parent_list: list[str] | None = None,
                  separator_dict: dict | None = None,
                  no_merge_list: list | None = None) -> bool:
    """Write what the merging stage made of the merge.

    Only what the machine settled and what the run said about it, and the commits it
    merged.

    Args:
        path: Where to write it.
        result_dict: The return value of integrate_mod.run_integration(). Its
            "head_block", what keeps any merge over the same HEAD from being settled, is
            written as it is for a merge that did not settle.
        gap_dict: recheck_gap()'s return value, written as it is. None or {} writes
            neither of its keys.
        parent_list: integrate_mod.branch_commit_list() over the union, written as
            "parents". None writes no such key.
        separator_dict: integrate_mod.separator_record_dict() over the union,
            written as "separator" for a merge that did not settle: the separator lines
            the union may hold. None writes no such key.
        no_merge_list: integrate_mod.no_merge_record_list() over the union, written
            as "no_merge" for a merge that did not settle: the sides whose change to a
            file is not in the union. None writes no such key.

    Returns:
        Whether the file was written (_write_out).
    """
    return _write_out(path, {
        **_verdict_dict(result_dict),
        # The tests the merging run changed
        "tests_changed": sorted(edit.test_change_set()),
        # How the run ended and what it recorded with record_decision()
        "ended": integrate_mod.end_of(result_dict),
        "decisions": result_dict["work"]["decisions"],
        **(gap_dict or {}),
        **({"parents": parent_list} if parent_list is not None else {}),
        **({"separator": separator_dict}
           if separator_dict is not None and not result_dict["settled"] else {}),
        **({"no_merge": no_merge_list}
           if no_merge_list is not None and not result_dict["settled"] else {}),
        **({"head_block": result_dict["head_block"]}
           if result_dict.get("head_block") and not result_dict["settled"] else {}),
    }, "what the merge settled")


def read_test_task(path: str) -> dict:
    """Read the files one run of the stage that writes tests is handed.

    Args:
        path: The file, as --target-in names it.

    Returns:
        {"paths": [repository-relative path, ...], "test_paths": [the test files that
        refer to them, ...], "baseline": the baseline before the stage, as
        testcreate.list_target() returns it}, with "first_case" and "case_step"
        (positive integers) where the file holds them.

    Raises:
        SystemExit: When the file is not there or is of another shape.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            task_dict = json.load(handle)
    except (OSError, ValueError, RecursionError) as e:
        raise SystemExit(f"test-create: --target-in {path} cannot be read ({e})") from None
    if not isinstance(task_dict, dict) or not all(
            isinstance(task_dict.get(key), list) for key in ("paths", "test_paths", "baseline")) \
            or not all(type(task_dict.get(key, 1)) is int and task_dict.get(key, 1) > 0
                       for key in ("first_case", "case_step")):
        raise SystemExit(f"test-create: --target-in {path} is not "
                         f"{{paths, test_paths, baseline[, first_case, case_step]}}")
    return task_dict


def write_create_outcome(path: str, result_dict: dict) -> bool:
    """Write what the stage that writes tests made of them.

    The keys of _verdict_dict(), "tests_changed", "ended" and "decisions" are written as
    write_outcome() writes them, with the stage's own keys beside them.

    Args:
        path: Where to write it.
        result_dict: The return value of test_create_mod.run_test_create().

    Returns:
        Whether the file was written (_write_out).
    """
    return _write_out(path, {
        "stage": integrate_mod.TEST_CREATE_STAGE,
        **_verdict_dict(result_dict),
        "tests_changed": result_dict["tests_changed"],
        "non_test_changed": result_dict["non_test_changed"],
        "untooled": result_dict["untooled"],
        "tests_base": result_dict["tests_base"],
        "base_failed": result_dict["base_failed"],
        "skipped": result_dict["skipped"],
        # What the first stage decided from the request
        "cases": result_dict["cases"],
        "no_case": result_dict["no_case"],
        "uncovered": result_dict["uncovered"],
        "refused": result_dict["refused"],
        "ended": integrate_mod.end_of(result_dict),
        "decisions": result_dict["work"]["decisions"],
        "gap": result_dict["work"]["gap"],
    }, "what the tests came to")


def write_refactor_outcome(path: str, stage: str, result_dict: dict) -> bool:
    """Write what a stage that refactors made of its run.

    The keys of _verdict_dict(), "tests_changed", "ended" and "decisions" are written as
    write_outcome() writes them, with the stage's own keys beside them.

    Args:
        path: Where to write it.
        stage: One of refactor_mod.STAGE_TUPLE.
        result_dict: The return value of refactor_mod.run_refactor().

    Returns:
        Whether the file was written (_write_out).
    """
    return _write_out(path, {
        "stage": stage,
        **_verdict_dict(result_dict),
        "tests_changed": result_dict["tests_changed"],
        "non_test_changed": result_dict["non_test_changed"],
        "untooled": result_dict["untooled"],
        "skipped": result_dict["skipped"],
        "refused": result_dict["refused"],
        "ended": integrate_mod.end_of(result_dict),
        "decisions": result_dict["work"]["decisions"],
    }, "what the refactoring came to")


def print_dispatch(project_dir: str, result_dict: dict) -> None:
    """Print what one run of a request produced: the answer, or the agents and the union.

    For a question: each question with whether its agent answered it, whether the
    combining run put the answers together, and the answer. For a change: each task
    with what landed, and the union. For a request the splitting stage handed back:
    what it leaves open.

    What a run left to carry on from is not printed here.

    A request that was run more than once says so first, with the files the runs
    before the last put into the working tree; what follows is the last run. A
    request handed back by the last run says how many files those runs put in on the
    "unclear" line instead. The cases the tests of a run before the last found not
    done are printed where the last run put no union in ("earlier_gap").

    Args:
        project_dir: The target repository, as an absolute path.
        result_dict: The return value of sandbox.dispatch().
    """
    if result_dict.get("runs", 1) > 1:
        print(f"runs: {result_dict['runs']} (the runs before the last did not get through)")
        earlier_path_list = _land_path_list(
            {"earlier_land": result_dict.get("earlier_land") or []})
        if earlier_path_list and dispatch_mod.UNCLEAR_KEY not in result_dict:
            print(f"  the runs before the last put {len(earlier_path_list)} file(s) into "
                  f"the working tree, which are still there: "
                  f"{one_line(list_text(earlier_path_list))}")
    if result_dict.get("merge_runs", 1) > 1:
        print(f"merges: {result_dict['merge_runs']} (the merges before the last were not "
              "settled)")
    if result_dict.get("combine_runs", 1) > 1:
        print(f"combines: {result_dict['combine_runs']} (the combines before the last "
              "handed no answer back)")
    if dispatch_mod.UNCLEAR_KEY in result_dict:
        unclear_list = result_dict[dispatch_mod.UNCLEAR_KEY]
        earlier_count = len(_land_path_list(result_dict))
        print(f"unclear: {len(unclear_list)} (no agent was started and nothing was changed)"
              if not earlier_count else
              f"unclear: {len(unclear_list)} (no agent was started in the last run. The "
              f"runs before it put {earlier_count} file(s) into the working tree, which "
              f"are still there)")
        for number, text in enumerate(unclear_list, 1):
            print(f"  {number}. {text}")
        _print_gap(result_dict.get("earlier_gap") or [])
        _print_old_run_list(result_dict)
        return
    if "answer" in result_dict:
        print(f"questions: {len(result_dict['questions'])}")
        for entry_dict in result_dict["answers"]:
            again = f", try {entry_dict['attempt']}" if entry_dict["attempt"] > 1 else ""
            print(f"  {entry_dict['n']}. {'answered' if entry_dict['answer'] else 'no answer'}"
                  f"{again}")
            print(f"     {entry_dict['request']}")
            if entry_dict["exit_status"]:
                why = entry_dict["error"] or sandbox_mod.status_text(entry_dict["exit_status"])
                print(f"     it did not finish: {why}")
        if result_dict["combined"]["error"]:
            print(f"combined: nothing ({result_dict['combined']['error']}), so the "
                  f"agents' answers are printed as they came")
        print("answer:")
        print(f"  {result_dict['answer']}")
        _print_old_run_list(result_dict)
        return
    print(f"tasks: {len(result_dict['tasks'])}")
    unfinish_count = 0
    for entry_dict in result_dict["branches"]:
        land_text = f"{entry_dict['branch']} ({gitcmd.short(entry_dict['commit'])})" \
            if entry_dict["commit"] else "nothing taken in"
        again = f", try {entry_dict['attempt']}" if entry_dict["attempt"] > 1 else ""
        verdict = f", {entry_dict['verdict']}" if entry_dict["verdict"] else ""
        print(f"  {entry_dict['n']}. {land_text}{again}{verdict}")
        print(f"     {entry_dict['request']}")
        if entry_dict["exit_status"]:
            unfinish_count += 1
            why = entry_dict["error"] or sandbox_mod.status_text(entry_dict["exit_status"])
            print(f"     it did not finish: {why}")
    if result_dict["dropped"]:
        print(f"branches taken away (the union holds their work, or a later try did "
              f"it again): {', '.join(result_dict['dropped'])}")
    if not result_dict["integration"]:
        if unfinish_count:
            print(f"combined: nothing ({unfinish_count} task(s) did not finish, so what "
                  f"the others left is on their branches and nowhere else)")
        elif result_dict.get("token_stop"):
            print(f"combined: nothing ({result_dict['token_stop']}, so the merge was "
                  f"not started and what the tasks left is on their branches)")
        else:
            print("combined: nothing landed to combine")
    else:
        print_sandbox(project_dir, result_dict["integration"])
    _print_gap(result_dict.get("earlier_gap") or [])
    _print_old_run_list(result_dict)


def _print_old_run_list(result_dict: dict) -> None:
    """Print the runs this one took away, if it took any.

    Args:
        result_dict: The return value of sandbox.dispatch(), or of sandbox.run() or
            sandbox.integrate() with "swept" added.
    """
    if result_dict.get("swept"):
        print(f"older runs taken away (their branches and their output): "
              f"{', '.join(result_dict['swept'])}")


def print_report(project_dir: str, request: str, result_dict: dict) -> str:
    """Print what the run made of the request.

    Nothing is printed for a request that was a question, nor for a request the
    splitting stage handed back. What is printed for a change is what report.report()
    returns.

    Args:
        project_dir: The target repository, as an absolute path.
        request: The request as it was given.
        result_dict: The return value of sandbox.dispatch().

    Returns:
        What was printed, or "" when nothing was.
    """
    if "answer" in result_dict or dispatch_mod.UNCLEAR_KEY in result_dict:
        return ""
    with progress.step("write the report"):
        report_text = report_mod.report(project_dir, request, result_dict)
    if report_text:
        print()
        print(report_text)
    return report_text


def request_of_run(project_dir: str, request: str, resume_list: list[str]) -> str:
    """Return the request to split, read off the resume branches when none was passed.

    The request is read off the commits of the branches
    (integrate.request_in_branch_list). A request that was passed is used as it was
    passed.

    Args:
        project_dir: The target repository, as an absolute path.
        request: What --request was given, or "" when it was not.
        resume_list: The branches --resume named.

    Returns:
        The request as it was given, or, on one line, the one those branches carry.

    Raises:
        SystemExit: When no request was passed and none can be read: no branch was
            named, one of them is not in the repository, none of them carries one, or
            they were made for different requests.
    """
    if request:
        return request
    if not resume_list:
        raise SystemExit("run: pass --request, or --resume with a branch an earlier run "
                         "left, whose commits say what that run was asked for")
    try:
        text = integrate_mod.request_in_branch_list(project_dir, resume_list)
    except RuntimeError as e:
        raise SystemExit(f"run: {e}") from None
    if not text:
        raise SystemExit("run: pass --request: the branches passed with --resume do not "
                         "say between them what one request they were made for")
    progress.note("no request was passed, so the one %s carries is used: %s",
                  ", ".join(resume_list), text)
    return text


def request_of_integrate(project_dir: str, request: str,
                         name_list: list[str], resume: str = "") -> str:
    """Return the request a merge is for, read off the branches when none was passed.

    The request is read off the commits of the branches
    (integrate.request_in_branch_list), and branches that carry none, or that cannot be
    read, are not refused here. The branch an earlier merge left is read first and
    alone, and the branches are read together when it carries none, or when there is no
    such branch.

    Args:
        project_dir: The target repository, as an absolute path.
        request: What --request was given, or "" when it was not.
        name_list: The branches to read, the one an earlier merge left first.
        resume: The branch an earlier merge left, or "" when there is none.

    Returns:
        The request as it was given, the one those branches carry, or "" when there is
        neither. A branch that is not in the repository leaves it empty as well.
    """
    if request or not name_list:
        return request
    try:
        text = integrate_mod.request_in_branch_list(project_dir, [resume]) \
            if resume else ""
        text = text or integrate_mod.request_in_branch_list(project_dir, name_list)
    except RuntimeError:
        return ""
    if text:
        progress.note("no request was passed, so the one %s carries is used: %s",
                      ", ".join(name_list), text)
    return text


def _print_land_line(result_dict: dict | None) -> None:
    """Say that a change did not reach the working tree, and how to put it in.

    Args:
        result_dict: The union dispatch() combined, or None when there was none.
    """
    if result_dict and result_dict.get("land_why"):
        print(f"NOT applied to the working tree: "
              f"{sandbox_mod.land_why_text(result_dict)}")


def print_take_out_line(project_dir: str, result_dict: dict | None) -> None:
    """Say how to take back out what this command put into the working tree, or nothing.

    The command applies the diff the last of _land_dict_list() names ("last") in reverse
    with gitcmd.APPLY_TUPLE, in the target repository whatever directory it is run in.

    Args:
        project_dir: The target repository, as an absolute path.
        result_dict: The return value of sandbox.dispatch(), sandbox.run() or
            sandbox.integrate(), or None.
    """
    land_dict = (_land_dict_list(result_dict or {}) or [{}])[-1]
    if land_dict.get("last"):
        command = command_line("git", "-C", project_dir, *gitcmd.APPLY_TUPLE, "-R",
                               land_dict["last"])
        print(f"to take it back out of the working tree: {command}")


def print_resume_line(project_dir: str, end_dict: dict) -> None:
    """Say how to start again from what this run left, or nothing when it left nothing.

    The line is the subcommand and the flags to pass (resume_command_text).

    Args:
        project_dir: The target repository, as an absolute path.
        end_dict: What the command made, with "command" the subcommand, "request" the
            request it was given, "resume" the branch names as the run's record holds
            them, and "pass_branch" where the command was stopped (_stop_result_dict).
    """
    resume_list = end_dict.get("resume") or []
    if not resume_list:
        return
    command_text = resume_command_text(
        project_dir, end_dict.get("command"), resume_list, end_dict.get("request"),
        end_dict.get("pass_branch"))
    print(f"to carry on from what this run left: {command_text}")


def resume_command_text(project_dir: str, command: object, branch_list: list[str],
                        request: object = "", side_list: object = None) -> str:
    """Return the subcommand and the flags that carry on from branches.

    The subcommand is the one of _RESUME_COMMAND_DICT for command, or _RESUME_COMMAND
    for one it does not list.

    A command that was given a request other than the one the branches were made for
    (integrate.request_in_branch_list), or branches to combine besides them, is
    carried on from as it was given: that subcommand, --branch for each of those
    branches still there, and --request with that request.

    For any other, what the branches hold now is read over the subcommand:
    - _RESUME_COMMAND when every branch is a union the working tree holds whose tests
      found cases the change does not do (sandbox.is_gap_union), and
    - _LAND_COMMAND when the one branch is a settled union the working tree does not
      hold, whether or not its tests found such cases (sandbox.is_union_to_land).
    Where the subcommand is then _RESUME_COMMAND and neither the branches nor the
    command say what request they were made for, --request follows with
    _REQUEST_MARK in place of the request to pass.

    Args:
        project_dir: The target repository, as an absolute path.
        command: The subcommand that left the branches, as a record holds it.
        branch_list: The branch names.
        request: The request that command was given, as a record holds it. Anything
            but a string is read as none.
        side_list: The branches that command was passed to combine besides
            branch_list, as a record of a command stopped before it left a branch of
            its own holds them ("pass_branch"). Anything but a list is read as none.

    Returns:
        "<subcommand> --resume <branch> ...", as command_line() writes it, the request
        on one line, with " --request <mark>" after it where the request has to be
        passed.
    """
    resume_command = _RESUME_COMMAND_DICT.get(command, _RESUME_COMMAND) \
        if isinstance(command, str) else _RESUME_COMMAND
    own_request = one_line(request.replace("\0", "")) if isinstance(request, str) else ""
    try:
        branch_request = integrate_mod.request_in_branch_list(project_dir, branch_list)
        is_other_request = bool(own_request) and own_request != branch_request
        has_request = bool(own_request or branch_request)
    except RuntimeError:
        is_other_request, has_request = False, True
    side_list = _exist_branch_list(project_dir, side_list) \
        if isinstance(side_list, list) and resume_command != _RESUME_COMMAND else []
    word_list = [word for branch in branch_list for word in ("--resume", branch)]
    if is_other_request or side_list:
        word_list += [word for branch in side_list for word in ("--branch", branch)]
        word_list += ["--request", own_request] if is_other_request else []
        return command_line(resume_command, *word_list)
    if branch_list and all(sandbox_mod.is_gap_union(project_dir, branch)
                           for branch in branch_list):
        resume_command = _RESUME_COMMAND
    elif len(branch_list) == 1 \
            and sandbox_mod.is_union_to_land(project_dir, branch_list[0]):
        resume_command = _LAND_COMMAND
    text = command_line(resume_command, *word_list)
    if resume_command != _RESUME_COMMAND or has_request:
        return text
    return f"{text} --request {_REQUEST_MARK}"


def _print_unclear_line(result_dict: dict) -> None:
    """Say how to go on from a request the splitting stage handed back, or nothing.

    Args:
        result_dict: The return value of sandbox.dispatch().
    """
    if dispatch_mod.UNCLEAR_KEY not in result_dict:
        return
    print("to go on, run it again with --request holding the request and what settles "
          "the points above")


def _exist_branch_list(project_dir: str, branch_list: list) -> list[str]:
    """Return the branches among those named that are still in the repository.

    Args:
        project_dir: The target repository, as an absolute path.
        branch_list: The names, as a result or a record holds them. A name that is not
            a non-empty string is left out.

    Returns:
        The names that still point at a commit, in the order they were given.
    """
    return [branch for branch in branch_list
            if isinstance(branch, str) and branch
            and sandbox_mod.branch_tip(project_dir, branch)]


def open_branch_list(project_dir: str, branch_list: list) -> list[str]:
    """Return the branches among those named that a run is still carried on from.

    Args:
        project_dir: The target repository, as an absolute path.
        branch_list: The names, as a result or a record holds them. A name that is not
            a non-empty string is left out.

    Returns:
        The names that still point at a commit (_exist_branch_list) and that
        sandbox.open_branch_list() keeps: a branch that holds a result the working
        tree holds is left out, and so is a union left unsettled whose commits merged
        the working tree holds (sandbox.is_union_in_tree). They come in the order they
        were given.
    """
    return [branch for branch in sandbox_mod.open_branch_list(
                project_dir, _exist_branch_list(project_dir, branch_list))
            if not sandbox_mod.is_union_in_tree(project_dir, branch)]


def _land_dict_list(result_dict: dict) -> list[dict]:
    """Return what a command put into the working tree, each time it put something in.

    Args:
        result_dict: The return value of sandbox.dispatch(), sandbox.run() or
            sandbox.integrate().

    Returns:
        sandbox.land_in_tree()'s return values, oldest first: the ones of the runs
        before the last of a request that was run again ("earlier_land"), then the
        last run's. One that touched no file is left out.
    """
    holder_dict = result_dict.get("integration") or result_dict
    land_list = [*(result_dict.get("earlier_land") or []), holder_dict.get("land") or {}]
    return [land_dict for land_dict in land_list if land_dict.get("files")]


def _land_path_list(result_dict: dict) -> list[str]:
    """Return the files a command put into the working tree.

    Args:
        result_dict: The return value of sandbox.dispatch(), sandbox.run() or
            sandbox.integrate().

    Returns:
        The paths relative to the repository, each once, or [] when nothing was put in.
    """
    return list(dict.fromkeys(path for land_dict in _land_dict_list(result_dict)
                              for path in land_dict.get("paths") or []))


def write_run_record(args: argparse.Namespace, record_id: str,
                     result_dict: dict | None, exit_status: int,
                     report_text: str = "", error: str = "",
                     error_command: str = "") -> dict:
    """Write the record of one run, sandbox or integrate command, and remove old ones.

    The records beyond the newest DISPATCH_KEEP_RUN are removed unless a branch they
    name is still one to carry on from (open_branch_list, runrecord.remove_old).

    Args:
        args: The parsed arguments of the command.
        record_id: What the record is named, as sandbox.new_run_id() made it when the
            command started.
        result_dict: What the command returned, or None for one that was stopped
            before it returned.
        exit_status: The status the process ends with.
        report_text: The report that was printed, or "".
        error: What stopped the command, or "".
        error_command: The command that message ends with (textcut.error_part), or "".

    Returns:
        The record, with "path" set to its file, or to "" when it was not written.
        "resume" holds the branches still to carry on from (open_branch_list), "files"
        what was put into the working tree and "token" tokencount.total_dict(). For a
        command stopped before it returned, "result" is _stop_result_dict(): what it had
        put into the working tree and the branches its runs had left by then. For
        sandbox and integrate, whose result names no branches to carry on from, "resume"
        is what sandbox.resume_branch_list() makes of the result and exit_status, which
        is none for a change kept out of the working tree for the paths it touches. One
        that ended with another exit status than 0 and left no branch of its own is read
        as one stopped before it returned: "resume" and the result's "pass_branch",
        which is set in result_dict, are _stop_result_dict()'s. Where no union this
        process put into the working tree has yet stood for the unions it carried on
        from for their cases (sandbox.stand_for_list), the result names them under
        runrecord.STAND_FOR_KEY.
    """
    project_dir = os.path.abspath(args.project)
    out_dir = os.path.abspath(args.output_dir)
    name = knowledge_mod.project_name_of(args.project)
    if result_dict is None:
        result_dict = _stop_result_dict(args, record_id)
    resume_list = result_dict.get("resume")
    if resume_list is None:
        resume_list = sandbox_mod.resume_branch_list(
            [], result_dict, exit_status) if "branch" in result_dict else []
        if exit_status and not resume_list and not result_dict.get("branch"):
            stop_dict = _stop_result_dict(args, record_id)
            resume_list = stop_dict["resume"]
            if "pass_branch" in stop_dict:
                result_dict["pass_branch"] = stop_dict["pass_branch"]
    stand_list = sandbox_mod.stand_for_list(project_dir)
    if stand_list:
        result_dict = {**result_dict, runrecord.STAND_FOR_KEY: stand_list}
    start = progress.run_start()
    end = time.time()
    record_dict = {
        "id": record_id, "command": args.command, "request": args.request,
        "start": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
        "end": datetime.fromtimestamp(end).isoformat(timespec="seconds"),
        "elapsed_sec": round(end - start, 1),
        "exit_status": exit_status, "error": error, "error_command": error_command,
        "resume": open_branch_list(project_dir, resume_list),
        "files": _land_path_list(result_dict),
        "token": {**tokencount.total_dict(), "limit": config.MAX_TOKEN_PER_RUN},
        "report": report_text, "result": result_dict}
    directory = sandbox_mod.record_dir(out_dir, name)
    record_dict["path"] = runrecord.write(directory, record_dict)
    runrecord.remove_old(directory, config.DISPATCH_KEEP_RUN,
                         lambda old_dict: bool(open_branch_list(
                             project_dir, old_dict.get("resume") or [])))
    return record_dict


def _stop_result_dict(args: argparse.Namespace, record_id: str) -> dict:
    """Return what a command stopped before it returned had made by then.

    Args:
        args: The parsed arguments of the command.
        record_id: What the command's record is named.

    Returns:
        {"earlier_land": what the process put into the working tree
        (sandbox.land_list), "resume": the branches its runs had left to carry on
        from (sandbox.remain_branch_list), after the unions it was passed with
        --resume for the cases their tests found not done
        (sandbox.gap_union_list_of), or every branch it was passed with --resume
        when its runs had left none; the branches its runs had left alone, none
        included, once its last run had put its union into the working tree or left
        one still to go in (sandbox.last_run_has_union)}, and with "pass_branch",
        the branches it was passed with --branch, where "resume" is the branches it
        was passed with --resume and it was passed both.
    """
    project_dir = os.path.abspath(args.project)
    pass_list = [branch for branch in (*getattr(args, "resume_list", []),
                                       getattr(args, "resume", "")) if branch]
    remain_list = sandbox_mod.remain_branch_list(
        project_dir, os.path.abspath(args.output_dir), record_id)
    is_union = sandbox_mod.last_run_has_union(project_dir, remain_list)
    is_own = bool(remain_list) or is_union
    if remain_list and not is_union:
        remain_list = list(dict.fromkeys([*sandbox_mod.gap_union_list_of(
            project_dir, getattr(args, "resume_list", []), args.request), *remain_list]))
    side_list = [] if is_own or not pass_list else list(getattr(args, "branch_list", []))
    return {"earlier_land": sandbox_mod.land_list(),
            "resume": remain_list if is_own else pass_list,
            **({"pass_branch": side_list} if side_list else {})}


def _token_stop_status(result_dict: dict) -> int:
    """Say that the command's model calls reached MAX_TOKEN_PER_RUN, when they did.

    Args:
        result_dict: What the command returned. Its "token_stop" is set to what
            tokencount.limit_text() says when the limit is reached.

    Returns:
        1 when the limit is reached, else 0.
    """
    stop_text = tokencount.limit_text()
    if not stop_text:
        return 0
    result_dict["token_stop"] = stop_text
    print(f"token limit: {stop_text}, so the run ends with exit status 1")
    return 1


def _print_record_line(record_dict: dict) -> None:
    """Print the token counts of a command's model calls and where its record is.

    Args:
        record_dict: The return value of write_run_record().
    """
    token_dict = record_dict["token"]
    limit_text = f", of MAX_TOKEN_PER_RUN={token_dict['limit']:,}" if token_dict["limit"] \
        else ""
    print(f"tokens: {tokencount.line(token_dict)}{limit_text}")
    if record_dict["path"]:
        print(f"record: {record_dict['path']}")


# What the record of a command stopped by a signal says stopped it when the stop itself
# says nothing
_INTERRUPT_TEXT = "the run was stopped"

# The SIGTERM and SIGHUP that stopped the command, oldest first (_raise_interrupt)
_STOP_NUMBER_LIST: list[int] = []


# What the token counts of the processes a stopped command did not wait for are listed
# under in its record
_STOP_PART = "not waited for"

# The signals that stop a command the way Ctrl-C does, by their names in signal
_STOP_SIGNAL_NAME_TUPLE = ("SIGTERM", "SIGHUP")

# How often a broker serving a run looks whether that run's process is still there
# (seconds)
_RUN_WATCH_SEC = 1.0


def _raise_interrupt(signal_number: int, frame: object) -> None:
    """Stop the command on a signal the way Ctrl-C stops it.

    The signal is kept in _STOP_NUMBER_LIST.

    Args:
        signal_number: The signal that arrived.
        frame: The frame it arrived in, which is not read.

    Raises:
        KeyboardInterrupt: Always, with _INTERRUPT_TEXT and the signal's name.
    """
    _STOP_NUMBER_LIST.append(signal_number)
    raise KeyboardInterrupt(f"{_INTERRUPT_TEXT} ({signal.Signals(signal_number).name})")


def stop_number() -> signal.Signals:
    """Return the signal that stopped the command.

    Returns:
        The last SIGTERM or SIGHUP that stopped it (_raise_interrupt), or SIGINT when
        none did.
    """
    return signal.Signals(_STOP_NUMBER_LIST[-1]) if _STOP_NUMBER_LIST else signal.SIGINT


def _signal_stop_status() -> int:
    """Return the exit status a command stopped by a signal ends with.

    Examples:
        stopped by Ctrl-C -> 130
        stopped by SIGTERM -> 143
        stopped by SIGHUP -> 129

    Returns:
        SIGNAL_STATUS_BASE plus the number of stop_number()'s signal.
    """
    return SIGNAL_STATUS_BASE + stop_number()


def stop_on_signal() -> list[int]:
    """Have _STOP_SIGNAL_NAME_TUPLE's signals raise KeyboardInterrupt in the main thread.

    A command stopped by one ends the way one stopped by Ctrl-C does. A signal the
    process was started ignoring stays ignored, and one that already has a handler keeps
    it. Nothing is set from another thread than the main one, nor for a signal this
    system does not have.

    Returns:
        The numbers of the signals this call set the handler of.
    """
    signal_list = []
    if threading.current_thread() is not threading.main_thread():
        return signal_list
    for name in _STOP_SIGNAL_NAME_TUPLE:
        signal_number = getattr(signal, name, None)
        if signal_number is not None \
                and signal.getsignal(signal_number) is signal.SIG_DFL:
            signal.signal(signal_number, _raise_interrupt)
            signal_list.append(signal_number)
    return signal_list


def put_signal_back(signal_list: list[int]) -> None:
    """Give the signals stop_on_signal() set the handler of their default action again.

    A signal whose handler is no longer the one stop_on_signal() set is left as it is.

    Args:
        signal_list: What stop_on_signal() returned.
    """
    for signal_number in signal_list:
        if signal.getsignal(signal_number) is _raise_interrupt:
            signal.signal(signal_number, signal.SIG_DFL)


def _end_with_run(run_id: str) -> None:
    """End this process, as SIGTERM ends it, once the process of a run has ended.

    A thread of its own looks every _RUN_WATCH_SEC seconds whether the process the run
    id names is still there (the last "-" part of the id), and sends this process
    SIGTERM when it is not. An id that names no process starts no thread.

    Args:
        run_id: The run this process serves, as sandbox.new_run_id() names it.
    """
    tail = run_id.rsplit("-", 1)[-1]
    if not (tail.isascii() and tail.isdigit()):
        return
    run_process = int(tail)

    def watch() -> None:
        """Wait for the run's process to end, then send this process SIGTERM."""
        while not tmpfile.process_dead(run_process):
            time.sleep(_RUN_WATCH_SEC)
        logger.warning("the process of run %s has ended, so this broker ends too", run_id)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=watch, daemon=True).start()


def _error_text(error: Exception) -> str:
    """Return what the record of a command says an error that was not expected was.

    Examples:
        ValueError("no such target repository: /p")
            -> "ValueError: no such target repository: /p"

    Args:
        error: What the command raised.

    Returns:
        The exception's type name and its message.
    """
    return f"{type(error).__name__}: {error}"


def _write_stop_record(args: argparse.Namespace, record_id: str,
                       error: str | BaseException, exit_status: int = 1,
                       result_dict: dict | None = None) -> dict:
    """Write the record of a command that was stopped, and print the record's line.

    The line holds its token counts and where the record is (_print_record_line). The
    token counts of the processes the command had started and did not wait for are added
    first (tokencount.add_remain_file).

    Args:
        args: The parsed arguments of the command.
        record_id: What the record is named.
        error: What stopped the command: the message, or the exception, whose
            message is recorded apart from the command it ends with
            (textcut.error_part).
        exit_status: The status the process ends with.
        result_dict: What the command had returned when it was stopped, or None for
            one stopped before it returned.

    Returns:
        What the command had made when it was stopped: result_dict, or the record's
        "result" (_stop_result_dict) when result_dict is None, with "command" the
        subcommand, "request" the request it was given and "resume" the branches the
        record names.
    """
    tokencount.add_remain_file(_STOP_PART)
    why, command = error_part(error) if isinstance(error, BaseException) else (error, "")
    record_dict = write_run_record(args, record_id, result_dict, exit_status, error=why,
                                   error_command=command)
    _print_record_line(record_dict)
    return {**(result_dict if result_dict is not None else record_dict["result"]),
            "command": record_dict["command"], "request": record_dict["request"],
            "resume": record_dict["resume"]}


def print_result(result_dict: dict, stat_dict: dict) -> None:
    """Print edit's iters, answer and changed files.

    A run that ended itself without changing a file is labelled as such: it is the run
    saying the request needed no change. Under --dry-run what the tools would have
    written is printed in place of what the working tree holds. Which runs those are is
    read off history.judge()'s verdict, which is printed on the line above.

    Args:
        result_dict: The return value of run_edit().
        stat_dict: The return value of git_diff_stat().
    """
    print()
    print(f"iters: {result_dict['iters']} / {config.RLM_MAX_TURN}")
    print(f"verdict: {result_dict['verdict']} ({result_dict['why']})")
    for reason in result_dict.get("final_check") or []:
        print(f"  {reason}")
    if result_dict["answer"] is None:
        print(no_answer_line(result_dict, config.RLM_MAX_TURN, "COMPLETE"))
    elif result_dict["verdict"] == history.VERDICT_NO_CHANGE:
        print("answer: no change was made (it said the request needed none)")
        print(f"  {result_dict['answer']}")
    else:
        # A harvested answer never went through COMPLETE() and is labelled as such
        print(f"answer:{_late_answer_mark(result_dict)}")
        print(f"  {result_dict['answer']}")
    if config.EDIT_DRY_RUN:
        _print_dry_change()
        return
    print("changed:")
    for file_dict in stat_dict["files"]:
        print(f"  {file_dict['path']} +{file_dict['insertions']} -{file_dict['deletions']}")
    print(f"total: +{stat_dict['total_insertions']} -{stat_dict['total_deletions']}")
    print(f"untracked: {' '.join(stat_dict['untracked']) or '(none)'}")


def _print_dry_change() -> None:
    """Print what the editing tools would have written during a --dry-run."""
    change_list = edit.dry_change_list()
    print("dry run, so nothing was written. What the tools would have written:"
          if change_list else "dry run, and no tool was asked to write anything")
    for change in change_list:
        print(change, end="" if change.endswith("\n") else "\n")


def print_history() -> None:
    """Print each recorded attempt, newest last, of every edit history there is.

    The records are read from each agent's copy of the output directory and from the
    output directory itself (sandbox.output_dir_list), and printed the same way.

    The first line of each attempt shows both the verdict the run reached on its own and
    the one confirmed afterwards. A record in an agent's copy that is reached through a
    link leading out of the copy (sandbox.inside_path) is not read, which is logged.
    """
    path_list = [(agent, directory, os.path.join(directory, source.project_name or "",
                                                 history.HISTORY_FILE))
                 for agent, directory in sandbox_mod.output_dir_list(config.OUTPUT_DIR)]
    exist_list = [entry for entry in path_list if os.path.exists(entry[2])]
    for agent, directory, path in exist_list or path_list[:1]:
        print(f"history: {path}{f'  ({agent})' if agent else ''}")
        attempt_list = []
        try:
            if agent:
                sandbox_mod.inside_path(path, directory)
            attempt_list = history.read_at(path)["attempts"]
        except RuntimeError as e:
            logger.warning("the edit history at %s is not read: %s", path, e)
        _print_attempt_list(attempt_list)


# The subcommand that carries on from the branches a record names, by the subcommand
# that left the record. One not listed is carried on from with _RESUME_COMMAND
_RESUME_COMMAND_DICT = {"integrate": "integrate"}
_RESUME_COMMAND = "run"

# The subcommand that puts a settled union the working tree does not hold into it
_LAND_COMMAND = "integrate"

# What stands in a printed command for the request to pass with --request
_REQUEST_MARK = "<request>"


def _refusal_line(project_dir: str, record_dict: dict) -> str:
    """Say why a record's change was not applied to the working tree, or "".

    That is a change kept out of the working tree for the paths it touches, or a merge
    no merge over the same HEAD is settled after (sandbox.integrate()'s "head_block").

    Args:
        project_dir: The target repository, as an absolute path.
        record_dict: One record, as runrecord.read_list() returns it.

    Returns:
        Why the change was kept out, as one line, which says where its diff is, or
        what to change before the merge is carried on from, and the branch it is on
        while that is still in the repository, followed by the command that applies
        the diff by hand, as it was recorded. "" for a record of no such change, and
        for a merge of "head_block" whose branch a run is no longer carried on from
        (open_branch_list).
    """
    result_dict = record_dict.get("result") or {}
    holder_dict = result_dict.get("integration") or result_dict
    if not isinstance(holder_dict, dict):
        return ""
    if holder_dict.get("land_refuse"):
        branch_list = _exist_branch_list(project_dir, [holder_dict.get("branch")])
    elif holder_dict.get("head_block"):
        branch_list = open_branch_list(project_dir, [holder_dict.get("branch")])
        if not branch_list:
            return ""
    else:
        return ""
    branch_text = f", and on {branch_list[0]}" if branch_list else ""
    return with_command(f"NOT applied to the working tree{branch_text}: "
                        f"{one_line(str(holder_dict.get('land_why') or ''))}",
                        str(holder_dict.get("land_command") or ""))


def print_run_record_list(project_dir: str) -> None:
    """Print one entry per record left by run, sandbox and integrate, oldest first.

    Each entry says when the command started, which command it was, how it ended, how
    long it took and how many tokens its model calls used, then the request, then what
    kept its change out of the working tree for the paths it touches, or what to change
    before its merge is carried on from (_refusal_line), and then the subcommand and the
    flags that carry on from the branches it left that a run is still carried on from
    (open_branch_list, resume_command_text). Nothing is printed when there is no record.

    Args:
        project_dir: The target repository, as an absolute path.
    """
    directory = sandbox_mod.record_dir(os.path.abspath(config.OUTPUT_DIR),
                                       source.project_name or "")
    record_list = runrecord.read_list(directory)
    if not record_list:
        return
    print(f"runs: {directory}")
    for record_dict in record_list:
        token_dict = record_dict.get("token") or {}
        print(f"  {one_line(record_dict.get('id', ''))}  "
              f"{one_line(record_dict.get('command', ''))}  "
              f"exit {record_dict.get('exit_status', '?')}  "
              f"{progress.elapsed_text(record_dict.get('elapsed_sec') or 0)}  "
              f"{token_dict.get('total', 0):,} token(s)")
        if record_dict.get("request"):
            print(f"    {one_line(str(record_dict['request']))}")
        if record_dict.get("error"):
            print(with_command(f"    stopped: {one_line(str(record_dict['error']))}",
                               str(record_dict.get("error_command") or "")))
        refusal_line = _refusal_line(project_dir, record_dict)
        if refusal_line:
            print(f"    {refusal_line}")
        branch_list = open_branch_list(project_dir, record_dict.get("resume") or [])
        if branch_list:
            result_dict = record_dict.get("result")
            command_text = resume_command_text(
                project_dir, record_dict.get("command"), branch_list,
                record_dict.get("request"),
                result_dict.get("pass_branch") if isinstance(result_dict, dict) else None)
            print(f"    to carry on from it: {command_text}")


def _local_time_text(at: str) -> str:
    """Return a time an entry of the change history holds, in this machine's local time.

    Examples:
        "2026-10-27T16:32:39+00:00" (on a machine 9 hours ahead) -> "2026-10-28T01:32:39"
        "2026-10-27T16:32:39"                                     -> "2026-10-27T16:32:39"

    Args:
        at: The "at" of the entry, in ISO 8601.

    Returns:
        The time in local time, to the second and without its offset, when at holds an
        offset; at as it is otherwise.
    """
    try:
        time_value = datetime.fromisoformat(at)
    except ValueError:
        return at
    if time_value.tzinfo is None:
        return at
    return time_value.astimezone().replace(tzinfo=None).isoformat(timespec="seconds")


def _print_attempt_list(attempt_list: list[dict]) -> None:
    """Print the lines of one record: two per entry, each text of it as one line.

    Args:
        attempt_list: The entries, as read_at() returns them.
    """
    if not attempt_list:
        print("  (no attempt recorded yet)")
        return
    for attempt_dict in attempt_list:
        file_dict = attempt_dict.get("files", {})
        count_text = " ".join(f"{kind[:3]}{len(file_dict.get(kind, []))}"
                              for kind in gitcmd.CHANGE_KIND_TUPLE)
        confirm = attempt_dict.get("verified") or "-"
        print(f"  {attempt_dict['n']}  {one_line(_local_time_text(attempt_dict['at']))}  "
              f"{one_line(attempt_dict['verdict'])} (confirmed: {one_line(confirm)})  "
              f"{count_text}")
        print(f"    {one_line(attempt_dict.get('why', ''))}")


def print_sandbox(project_dir: str, result_dict: dict) -> None:
    """Print what one sandboxed run produced, and whether it is in the working tree.

    The branch is named unless its change was put into the working tree with nothing
    left open on it (sandbox.open_stage_key, "gap") and its name is one this tool made
    up (sandbox.is_own_branch). The command that applies a patch that was not taken in
    is sandbox.by_hand_guide()'s: it applies the patch the way this tool applies a diff
    (gitcmd.APPLY_TUPLE), in the target repository whatever directory it is run in, and
    checks out the commit the patch moves a submodule to. What the run wrote that the
    patch does not hold ("unseen_text") is named above that command, or in place of
    "changed: nothing" where the patch holds nothing, and a run whose container left no
    patch that is taken in is printed with "no_patch_why". A patch that was not taken in
    from a run with a non-zero exit status says that status (sandbox.status_text), and a
    patch that holds nothing is printed with "land_why", where there is one, in place of
    "changed: nothing".

    Args:
        project_dir: The target repository, as an absolute path.
        result_dict: The return value of sandbox.run() or sandbox.integrate().
    """
    print(f"base: {result_dict['base']}")
    if result_dict["patch"]:
        size_kb = os.path.getsize(result_dict["patch"]) / 1024
        print(f"patch: {result_dict['patch']} ({size_kb:.1f} KB, "
              f"{result_dict['files']} file(s))")
        if not result_dict["branch"]:
            if result_dict.get("unseen_text"):
                print(f"NOT in the patch, nor applied by the command below: "
                      f"{result_dict['unseen_text']}")
            move_text, command = sandbox_mod.by_hand_guide(project_dir,
                                                           result_dict["patch"])
            end_text = f", from a run that ended with " \
                f"{sandbox_mod.status_text(result_dict['exit_status'])}" \
                if result_dict.get("exit_status") else ""
            print(f"not taken in{end_text}. Applying it by hand skips the checks a change "
                  f"put into the working tree goes through; read it, and apply it by "
                  f"hand{move_text} with: {command}")
            return
    elif not result_dict.get("commit"):
        if result_dict.get("no_patch"):
            print(f"NOT taken in: {result_dict['no_patch_why']}")
        elif result_dict.get("unseen_text"):
            print(f"NOT in the patch, which holds nothing else: "
                  f"{result_dict['unseen_text']}")
        elif not result_dict.get("land_why"):
            print("changed: nothing")
        if result_dict.get("land_why") and not result_dict.get("no_patch"):
            print(f"NOT applied to the working tree: "
                  f"{sandbox_mod.land_why_text(result_dict)}")
        _print_stage_list(result_dict)
        return
    land_dict = result_dict.get("land")
    if not land_dict or sandbox_mod.open_stage_key(result_dict) or result_dict.get("gap") \
            or not sandbox_mod.is_own_branch(result_dict["branch"]):
        print(f"branch: {result_dict['branch']} ({gitcmd.short(result_dict['commit'])})")
    _print_stage_list(result_dict)
    if land_dict and land_dict["files"]:
        print(f"applied to the working tree: {land_dict['files']} file(s), over "
              f"{gitcmd.short(result_dict['base'])}. Commit them yourself")
    elif land_dict:
        print("nothing more to put into the working tree")
    else:
        print(f"NOT applied to the working tree: "
              f"{sandbox_mod.land_why_text(result_dict) or 'nothing was put in'}")


def _print_stage_list(result_dict: dict) -> None:
    """Print what each stage that followed a merging run left on its branch.

    Args:
        result_dict: The return value of sandbox.run() or sandbox.integrate().
    """
    _print_test_create(result_dict.get("test_create"))
    for stage in refactor_mod.STAGE_TUPLE:
        _print_refactor(stage, result_dict.get(integrate_mod.stage_key(stage)))
    _print_gap(result_dict.get("gap") or [])


def _print_gap(gap_list: list[dict]) -> None:
    """Print the cases the tests found the change does not do.

    Args:
        gap_list: [{"what", "why"}, ...], as record_gap() recorded them. Empty prints
            nothing.
    """
    if not gap_list:
        return
    print(f"NOT done by the change, as the tests found ({len(gap_list)}):")
    for entry_dict in gap_list:
        print(f"  {entry_dict['what']} — {entry_dict['why']}")


# What each stage that refactors is called where its result is printed
_REFACTOR_LABEL_DICT = {integrate_mod.REFACTOR_STAGE: "refactoring",
                        integrate_mod.TEST_REFACTOR_STAGE: "test refactoring"}


def _print_refactor(stage: str, stage_dict: dict | None) -> None:
    """Print what a stage that refactors left on a merging run's branch.

    The first line says "nothing done" and why for a stage that was skipped. For any
    other it names the commit that holds the stage's change, or says "nothing taken in"
    for a stage whose container left no patch that is taken in or could not be run to
    its end ("exit_status" 1), and "nothing changed" otherwise.

    Args:
        stage: One of refactor_mod.STAGE_TUPLE.
        stage_dict: integrate()'s entry for the stage, or None for a run without it.
    """
    if not stage_dict:
        return
    label = _REFACTOR_LABEL_DICT[stage]
    if stage_dict["skipped"]:
        print(f"{label}: nothing done ({stage_dict['skipped']})")
        return
    if stage_dict["commit"] and stage_dict["patch"]:
        land_text = f"{stage_dict['branch']} ({gitcmd.short(stage_dict['commit'])})"
    elif stage_dict.get("exit_status"):
        land_text = "nothing taken in"
    else:
        land_text = "nothing changed"
    print(f"{label}: {land_text}, {stage_dict['verdict'] or 'no verdict'}")
    change_list = stage_dict["tests_changed"] + stage_dict["non_test_changed"]
    if change_list:
        print(f"  changed: {', '.join(change_list)}")
    if stage_dict.get("refused"):
        print(f"  writes refused (not the stage's files): {', '.join(stage_dict['refused'])}")
    if sandbox_mod.is_stage_open(stage_dict):
        print(f"  NOT settled: {stage_dict['why'] or 'the run said nothing about why'}")


def _print_test_create(create_dict: dict | None) -> None:
    """Print what the stage that writes tests left on a merging run's branch.

    Args:
        create_dict: integrate()'s "test_create", or None for a run without the stage.
    """
    if not create_dict:
        return
    if create_dict["skipped"]:
        print(f"tests: none written ({create_dict['skipped']})")
        return
    land_text = f"{create_dict['branch']} ({gitcmd.short(create_dict['commit'])})" \
        if create_dict["commit"] and create_dict["patch"] else "nothing taken in"
    print(f"tests: {land_text}, {create_dict['verdict'] or 'no verdict'}")
    entry_list = create_dict.get("test_tasks") or []
    if len(entry_list) > 1:
        settle_count = sum(1 for entry_dict in entry_list
                           if entry_dict["verdict"] == integrate_mod.SETTLE_VERDICT
                           or entry_dict["skipped"])
        print(f"  test tasks: {len(entry_list)} ({settle_count} settled)")
    print(f"  cases decided from the request: {len(create_dict['cases'])}")
    if create_dict.get("uncovered"):
        print(f"  no case for: {len(create_dict['uncovered'])} definition(s)")
    if create_dict.get("refused"):
        print(f"  writes refused (not the test task's files): "
              f"{', '.join(create_dict['refused'])}")
    if create_dict["tests_changed"]:
        print(f"  written: {', '.join(create_dict['tests_changed'])}")
    if create_dict["non_test_changed"]:
        print(f"  also changed (not tests): {', '.join(create_dict['non_test_changed'])}")
    if create_dict["tests_base"]:
        print(f"  on the base: {create_dict['tests_base']}")
    if sandbox_mod.is_stage_open(create_dict):
        print(f"  NOT settled: {create_dict['why'] or 'the run said nothing about why'}")


# The repository integrate --here put a union into, integrate.prepare()'s return value
# and whether the command prints what the repository holds (_has_union_note), from the
# return of integrate.prepare() until the union is what the command leaves
_HERE_UNION_LIST: list[tuple[str, dict, bool]] = []


def _has_union_note(args: argparse.Namespace) -> bool:
    """Say whether integrate --here prints what it left in the repository.

    That is what the repository holds and how to put it back (print_union_in_place).

    Args:
        args: The parsed arguments of integrate --here.

    Returns:
        False when it writes --outcome-out, else True.
    """
    return not args.outcome_out


def _put_union_back() -> None:
    """Put back each repository integrate --here put a union into.

    Only the ones whose union is not yet what the command leaves are put back
    (integrate.put_back, whose error log names the line that puts it back by hand where
    git cannot).
    """
    while _HERE_UNION_LIST:
        project_dir, prepare_dict, _ = _HERE_UNION_LIST.pop()
        integrate_mod.put_back(project_dir, prepare_dict["was_on"])


def _print_union_remain() -> None:
    """Say what each repository integrate --here put a union into holds.

    Only the ones whose union is not yet what the command leaves are named, each with
    how to put it back (print_union_in_place), where the command prints that
    (_has_union_note).
    """
    while _HERE_UNION_LIST:
        project_dir, prepare_dict, has_note = _HERE_UNION_LIST.pop()
        if has_note:
            print_union_in_place(project_dir, prepare_dict)


def print_union_in_place(project_dir: str, prepare_dict: dict) -> None:
    """Say what state --here left the repository in, and how to put it back.

    The lines say that the working tree was written over with the union. Both commands
    of the line that puts it back name the repository, and work on it whatever directory
    they are run in.

    Args:
        project_dir: The target repository, as an absolute path.
        prepare_dict: The return value of integrate.prepare().
    """
    print(f"left in place: the working tree now holds the union, against "
          f"{gitcmd.short(prepare_dict['base'])}")
    if prepare_dict["was_on"]:
        print(f"  to put the repository back: "
              f"{integrate_mod.put_back_command(project_dir, prepare_dict['was_on'])}")


def _print_turn_line(result_dict: dict) -> None:
    """Print how many turns a stage's run took and how it ended, and what it said.

    Args:
        result_dict: The return value of integrate.run_integration(),
            test_create_mod.run_test_create() or refactor_mod.run_refactor().
    """
    print(f"turns: {result_dict['iters']}"
          f"{_send_back_mark(result_dict['refused_completes'])}, "
          f"ended: {integrate_mod.end_of(result_dict)}")
    if result_dict["answer"]:
        print(f"said: {result_dict['answer']}")


def _print_unsettle(result_dict: dict) -> None:
    """Print "NOT settled:" and what the gates still find, one line each.

    Args:
        result_dict: The return value of integrate.run_integration(),
            test_create_mod.run_test_create() or refactor_mod.run_refactor().
    """
    print("NOT settled:")
    for reason in result_dict["left"]:
        print(f"  {reason}")


def print_integration(result_dict: dict) -> None:
    """Print what one merging run settled and what it left.

    Args:
        result_dict: The return value of integrate.run_integration().
    """
    for entry_dict in result_dict["work"]["branches"]:
        conflict_count = len(entry_dict["conflicted"])
        print(f"merged: {entry_dict['branch']}"
              + (f" ({conflict_count} conflicted file(s))" if conflict_count else ""))
    if not result_dict["called"]:
        print("settled: by git alone, with no model called")
        return
    _print_turn_line(result_dict)
    if result_dict["settled"]:
        print("settled: everything the merge left")
    else:
        _print_unsettle(result_dict)


def print_test_create_stage(result_dict: dict) -> None:
    """Print what one run of the stage that writes tests wrote and what it left.

    Args:
        result_dict: The return value of test_create_mod.run_test_create().
    """
    if result_dict["skipped"]:
        print(f"skipped: {result_dict['skipped']}")
        return
    print(f"cases: {len(result_dict['cases'])} (decided in {result_dict['case_iters']} "
          f"turn(s), without the code)"
          + (f", {len(result_dict['uncovered'])} definition(s) with no case and no reason"
             if result_dict["uncovered"] else ""))
    if not result_dict["called"]:
        _print_unsettle(result_dict)
        return
    _print_turn_line(result_dict)
    print(f"tests written: {', '.join(result_dict['tests_changed']) or 'none'}")
    if result_dict["non_test_changed"]:
        print(f"also changed (not tests): {', '.join(result_dict['non_test_changed'])}")
    if result_dict["tests_base"]:
        print(f"on the base: {result_dict['tests_base']}")
    _print_gap(result_dict["work"]["gap"])
    if result_dict["settled"]:
        print("settled: the tests pass and every reference resolves")
    else:
        _print_unsettle(result_dict)


def print_refactor_stage(result_dict: dict) -> None:
    """Print what one run of a stage that refactors changed and what it left.

    Args:
        result_dict: The return value of refactor_mod.run_refactor().
    """
    if result_dict["skipped"]:
        print(f"skipped: {result_dict['skipped']}")
        return
    _print_turn_line(result_dict)
    change_list = result_dict["tests_changed"] + result_dict["non_test_changed"]
    print(f"changed: {', '.join(change_list) or 'nothing'}")
    if result_dict["settled"]:
        print("settled: the tests say what they said and every reference resolves")
    else:
        _print_unsettle(result_dict)


def print_format(result_dict: dict | None) -> None:
    """Print what one run of the formatting command did.

    Args:
        result_dict: The return value of edit.run_format().
    """
    if result_dict is None:
        print("format: not run (FORMAT_COMMAND is empty, or no file was changed)")
        return
    status = result_dict["exit_status"]
    print(f"format: {len(result_dict['files'])} file(s), "
          + (f"exit status {status}" if status is not None else "did not run"))
    if status != 0 and result_dict["output"]:
        print(result_dict["output"])


def print_test(result_dict: dict) -> None:
    """Print what one run of the test command did.

    The commands of the other languages that ran with it (edit.other_test_list) are
    named under the primary language's. A run that reached no verdict is followed by
    why, and by the line saying nothing was fetched when a command of it stopped
    short of its tests (edit.stop_short_run).

    Args:
        result_dict: The return value of edit.run_test().
    """
    print(f"test command: {shlex.join(edit.primary_test_command())}")
    for _, argv, directory in edit.other_test_list():
        print(f"  and: {shlex.join(argv)}{f' (in {directory})' if directory else ''}")
    print(f"status: {result_dict['status']}")
    if result_dict["status"] == "did not run":
        if edit.stop_short_run(result_dict):
            print("nothing was fetched. The cache is only filled by a command that ran")
        print(f"  {result_dict['failed_excerpt']}")


# The help of --doc-create of a subcommand that starts containers
_DOC_COST_HELP = ("codetwine calls the LLM several times for each file it writes a "
                  "document for, which costs money, following the template "
                  "DOC_TEMPLATE_PATH names (doc_template.json of this repository when "
                  "unset). Without "
                  "it only the definitions and the line numbers are made, and no LLM is "
                  "called. Inside the container the documents are never made")
_BUILD_DOC_HELP = ("when nothing is built in the output directory yet and the "
                   "repository is analysed here before the agents start, write the "
                   "design documents too: " + _DOC_COST_HELP)
_DOC_UPDATE_HELP = ("write the design documents before the agents start: when nothing "
                    "is built in the output directory yet, with the analysis made here; "
                    "otherwise the analysis there is made again here, writing the "
                    "documents of the files that have none or have changed since theirs "
                    "were written and of the files depending on them, and is refused "
                    "while other runs are going over the output directory. This is done "
                    "once, when the command starts, and not by a run that carries on "
                    "from a settled merge (--resume). " + _DOC_COST_HELP)
_RUN_ASK_HELP = (". Without it, and with a terminal, the run asks whether to make "
                 "them when nothing is built in the output directory yet")


def _help_value(value: object) -> str:
    """Return a setting's value in the form a help text holds it in.

    Each "%" is doubled, which argparse writes out as one "%".

    Args:
        value: The value of a setting.

    Returns:
        The value as text, shown in --help as it is.
    """
    return str(value).replace("%", "%%")


# The help of the container flags, of --exclude and of --preset
_ENGINE_HELP = ("the container engine (defaults to the CONTAINER_ENGINE environment "
                f"variable, and to whichever of {', '.join(sandbox_mod.ENGINE_TUPLE)} is on "
                "PATH when that is empty)")
_IMAGE_HELP = ("the image to run (defaults to the CONTAINER_IMAGE environment variable, "
               "and to codetwine-loom, which holds the toolchains of every language, "
               "when that is not set). An image that is missing, or was not built from "
               "this codetwine-loom, is built after a question at a terminal")
_EXCLUDE_HELP = ("a directory or file name to keep out of the analysis, on top of "
                 "codetwine's own list and of the BUILD_EXCLUDE environment variable. "
                 "A name is kept out wherever it is; one holding a / is a path from the "
                 "repository's root (apps/old, ./vendor for vendor at the root alone, "
                 "apps/*/gen), and keeps out that file or everything under that "
                 "directory. Takes one or more names, and can be given more than once. "
                 "A run over an analysis already in the output directory keeps out what "
                 "that analysis keeps out: given no names it takes the analysis's, given "
                 "other names it stops and says which build to run")
_PRESET_HELP = ("how hard the run tries: low, medium or high. It picks the column of "
                "the settings that raise accuracy against cost and time together "
                "(defaults to the PRESET environment variable, itself "
                f"{_help_value(config.PRESET)})")


class _SettingDefault(str):
    """The value of an option that was not passed: what its setting holds."""


# The setting each option of the command line gives a value to, by the name the parsed
# arguments hold the option under. An option not passed holds None or a _SettingDefault
_OPTION_SETTING_DICT = {
    "project": "PROJECT_DIR", "output_dir": "OUTPUT_DIR", "preset": "PRESET",
    "guide": "EDIT_GUIDE", "engine": "CONTAINER_ENGINE", "image": "CONTAINER_IMAGE",
    "endpoint": "LLM_BROKER", "overview": "OVERVIEW",
    "test_create": "TEST_CREATE", "refactor": "REFACTOR", "test_refactor": "TEST_REFACTOR",
}


def _option_setting_set(args: argparse.Namespace) -> set[str]:
    """Return the settings the command line gave a value to.

    Args:
        args: The arguments as argparse returned them, before _check_args() fills in
            the options that were not passed.

    Returns:
        The names of _OPTION_SETTING_DICT whose option was passed.
    """
    return {name for dest, name in _OPTION_SETTING_DICT.items()
            if getattr(args, dest, None) is not None
            and not isinstance(getattr(args, dest), _SettingDefault)}


def _add_preset(parser: argparse.ArgumentParser) -> None:
    """Add --preset to a parser whose subcommand calls a model."""
    parser.add_argument("--preset", default=None, choices=config.PRESET_LEVEL_TUPLE,
                        help=_PRESET_HELP)


# The stages after a settled merge that the command line can turn off, by the setting
# each one writes and what it does
_STAGE_FLAG_DICT = {
    "TEST_CREATE": "write tests for the combined change",
    "REFACTOR": "refactor the combined change",
    "TEST_REFACTOR": "refactor the tests of the combined change",
}


def _add_stage_flag(parser: argparse.ArgumentParser) -> None:
    """Add --no-test-create, --no-refactor and --no-test-refactor.

    Args:
        parser: The parser of a subcommand that merges branches.
    """
    for name, what in _STAGE_FLAG_DICT.items():
        parser.add_argument(
            f"--no-{name.lower().replace('_', '-')}", dest=name.lower(),
            action="store_const", const=False, default=None,
            help=f"do not {what} after the merge settles")


def _add_container_flag(parser: argparse.ArgumentParser) -> None:
    """Add --engine and --image.

    Both default to None, which _check_args() replaces with CONTAINER_ENGINE and
    CONTAINER_IMAGE once it has told whether either was passed.

    Args:
        parser: The parser of a subcommand that starts containers.
    """
    parser.add_argument("--engine", default=None, help=_ENGINE_HELP)
    parser.add_argument("--image", default=None, help=_IMAGE_HELP)


def _add_overview_flag(parser: argparse.ArgumentParser) -> None:
    """Add --no-overview.

    Args:
        parser: The subcommand's parser.
    """
    parser.add_argument(
        "--no-overview", dest="overview", action="store_const", const=False,
        default=None,
        help="neither make the repository's overview nor hand it to search step 1. "
             "Without it an overview that is missing or not current is made first on "
             "this machine (the LLM is called); inside a container one that is not "
             "current is not used. With QUERY_EXPAND=false no overview is made or "
             "used either way")


# What --request-file reads the standard input for
_STDIN_NAME = "-"

# The two flags one request is passed with
_REQUEST_FLAG_TUPLE = ("--request", "--request-file")


def _flag_set(parser: argparse.ArgumentParser) -> set[str]:
    """Return every flag the sub-commands of a parser take.

    Args:
        parser: The parser the sub-commands were added to.

    Returns:
        The option strings of all the sub-commands ("--project", "--no-resume", "-h").
    """
    flag_set = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child_parser in action.choices.values():
                flag_set.update(child_parser._option_string_actions)
    return flag_set


def _is_flag_word(word: str, flag_set: set[str]) -> bool:
    """Say whether one argument is a flag of a sub-command rather than a text.

    Examples:
        "--dry-run"                          -> True
        "--output-dir=out"                   -> True
        "--output-dir=out is where to write" -> False
        "--dry-run by default"               -> False

    Args:
        word: The argument.
        flag_set: The flags the sub-commands take (_flag_set()).

    Returns:
        True when word is one of flag_set, or one of them with "=" and a value behind
        it and no blank anywhere in it.
    """
    name, _, _ = word.partition("=")
    return name in flag_set and len(word.split()) == 1


def _join_request_flag(argv: list[str], flag_set: set[str]) -> list[str]:
    """Join --request and --request-file with the argument after each of them.

    Examples:
        ["edit", "--request", "-v as well"] -> ["edit", "--request=-v as well"]
        ["run", "--request-file", "-"]      -> ["run", "--request-file=-"]
        ["edit", "--request", "--dry-run"]  -> unchanged
        ["edit", "--request", "--guide=x goes first"]
                                            -> ["edit", "--request=--guide=x goes first"]

    Args:
        argv: The argument list.
        flag_set: The flags the sub-commands take (_flag_set()).

    Returns:
        The list with each of the two flags and the argument that follows it made one
        "--flag=value" argument, which is read as the flag's value whatever it starts
        with. A flag that comes last, or that is followed by a flag of a sub-command
        (_is_flag_word), is left as it is.
    """
    arg_list = []
    position = 0
    while position < len(argv):
        follow = argv[position + 1] if position + 1 < len(argv) else None
        if (argv[position] in _REQUEST_FLAG_TUPLE and follow is not None
                and not _is_flag_word(follow, flag_set)):
            arg_list.append(f"{argv[position]}={follow}")
            position += 2
        else:
            arg_list.append(argv[position])
            position += 1
    return arg_list


def _add_request(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add --request and --request-file, the two ways one request is passed.

    Args:
        parser: The subcommand's parser, or a parent of it.
        help_text: What the request is to this subcommand.
    """
    parser.add_argument("--request", default="", metavar="TEXT", help=help_text)
    parser.add_argument("--request-file", default="", metavar="PATH",
                        help="a UTF-8 text file holding the request, read in place of "
                             f"--request; {_STDIN_NAME} reads the standard input. The "
                             "blank space around the text is dropped")


@functools.lru_cache(maxsize=1)
def _stdin_text() -> str:
    """Return what the standard input holds, read once however often it is asked for.

    Returns:
        Its bytes read as UTF-8 with a BOM at their head dropped, or "" for a process
        started with the standard input closed.

    Raises:
        UnicodeDecodeError: When the bytes are not UTF-8.
    """
    if sys.stdin is None:
        return ""
    return sys.stdin.buffer.read().decode("utf-8-sig")


def _read_request_file(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Put the text --request-file names into args.request.

    Args:
        parser: The top parser, which says what is wrong and exits: when --request was
            passed too, when the file cannot be read, and when it holds no text.
        args: The parsed arguments. A subcommand without --request-file, and one it
            was not passed to, are left as they are.
    """
    path = getattr(args, "request_file", "")
    if not path:
        return
    if args.request:
        parser.error("--request and --request-file both hold the request; pass one of "
                     "them")
    try:
        if path == _STDIN_NAME:
            text = _stdin_text()
        else:
            with open(path, encoding="utf-8-sig") as handle:
                text = handle.read()
    except (OSError, UnicodeDecodeError) as e:
        parser.error(f"--request-file: {path} could not be read as UTF-8 text ({e})")
    if not text.strip():
        parser.error(f"--request-file: {path} holds no text")
    args.request = text.strip()


def _write_option_value(name: str, value: str) -> None:
    """Write a setting the command line gives into the environment.

    What the environment held for it before is kept (config.keep_option_value).

    Args:
        name: The setting's name.
        value: The value the command line gives.
    """
    config.keep_option_value(name, os.environ.get(name), value)
    os.environ[name] = value


def _apply_stage_flag(args: argparse.Namespace) -> None:
    """Write the --no- flags given on the command line into the environment.

    config is then read again. A flag not given leaves its setting as it is.

    Args:
        args: The parsed arguments.
    """
    flag_dict = {name: getattr(args, name.lower(), None)
                 for name in (*_STAGE_FLAG_DICT, "OVERVIEW")}
    flag_dict = {name: value for name, value in flag_dict.items() if value is not None}
    if not flag_dict:
        return
    for name, value in flag_dict.items():
        _write_option_value(name, "true" if value else "false")
    importlib.reload(config)


def _want_resume(entry_dict: dict, choice: bool | None) -> bool:
    """Decide whether to carry on from the unfinished change the previous run left.

    Args:
        entry_dict: The open attempt, as history.open_attempt() returns it.
        choice: True for --resume, False for --no-resume, None for neither, which
            asks at a terminal (progress.at_terminal) and starts fresh where there is
            none or the input ends before an answer.

    Returns:
        True to hand the unfinished change to the model, False to start fresh.
    """
    if choice is not None:
        return choice
    file_dict = entry_dict.get("files", {})
    total = sum(len(file_dict.get(kind, [])) for kind in gitcmd.CHANGE_KIND_TUPLE)
    print(f"The previous run left a change unfinished ({entry_dict.get('why', '')}). "
          f"It changed {total} file(s) and reported:")
    print(f"  {entry_dict.get('did') or history.NO_ANSWER_TEXT}")
    # With no terminal to ask on, the run starts fresh without asking
    if not progress.at_terminal():
        print("no terminal to ask on, so this run starts fresh. Pass --resume to carry "
              "the unfinished change over")
        return False
    try:
        answer = progress.ask("Carry on from there? [y/N] ")
    except EOFError:
        print()
        return False
    return answer.strip().lower() in _YES_ANSWER_TUPLE


def _print_help(entry: config.EnvTemplateEntry) -> None:
    """Print a blank line and the comment lines the template has above a setting."""
    print()
    for text in entry.help_list:
        print(f"# {text}")


def _show_value(name: str, current: str, template_value: str) -> str:
    """Return what a question shows in brackets after a setting's name.

    Examples:
        ("PRESET", "high", "medium")                       -> "high"
        ("PRESET", "", "medium")                           -> "medium"
        ("LLM_API_KEY", "sk-1", "")                        -> "set"
        ("HTTPS_PROXY", "http://u:pw@proxy:3128", "")      -> "http://u:****@proxy:3128"
        ("HTTPS_PROXY", "http://u@x:p@ss@proxy:3128", "")  -> "http://u@x:****@proxy:3128"
        ("HTTPS_PROXY", "u:pw@proxy:3128", "")             -> "u:****@proxy:3128"

    Args:
        name: The setting's name.
        current: The value .env holds for it; "" when it holds none.
        template_value: The value the template writes for it.

    Returns:
        current with the password of a URL in it written _PASSWORD_MARK, or
        _SECRET_MARK for a key (llm.is_credential_name); template_value when current
        is "". The user and the password are what comes before the last "@" after the
        scheme, and the password is what follows the first ":" in them.
    """
    if not current:
        return template_value
    if llm.is_credential_name(name):
        return _SECRET_MARK
    scheme_match = _URL_SCHEME_RE.match(current)
    scheme = scheme_match[0] if scheme_match else ""
    user_part, at, host_part = current[len(scheme):].rpartition("@")
    user, colon, _ = user_part.partition(":")
    if not (at and colon):
        return current
    return f"{scheme}{user}:{_PASSWORD_MARK}@{host_part}"


def _answer_text(answer: str, current: str) -> str:
    """Return what an answer gives a setting, to be checked as a typed answer is.

    Examples:
        ("", "high")    -> "high"
        ("-", "high")   -> ""
        ("low", "high") -> "low"

    Args:
        answer: What was typed, without the blanks around it.
        current: The value .env holds for the setting; "" when it holds none.

    Returns:
        current on Enter, "" on _CLEAR_ANSWER, and answer otherwise.
    """
    if not answer:
        return current
    return "" if answer == _CLEAR_ANSWER else answer


def _ask_by_name(name: str, current: str, template_value: str,
                 is_secret: bool = False) -> str:
    """Ask for a variable by name, once, showing the value in brackets (_show_value).

    Args:
        name: The variable's name.
        current: The value .env holds for it, which Enter keeps; "" when it holds none.
        template_value: The value the template writes for it; "" when it writes none.
        is_secret: Whether the answer is read without echo.

    Returns:
        _answer_text() of what was typed, without the blanks around it.
    """
    show_text = _show_value(name, current, template_value)
    answer = progress.ask(f"{name} [{show_text}]: ", is_secret)
    return _answer_text(answer.strip(), current)


def _ask_setting(entry: config.EnvTemplateEntry, current: str = "") -> str:
    """Print a setting's comment lines and ask for its value.

    A setting that takes one of a few values (config.choice_tuple_of()) is asked again
    until the answer is one of them.

    Args:
        entry: The setting, as the template names it. A key is read without echo.
        current: The value .env holds for it, which Enter keeps; "" when it holds none.

    Returns:
        What was typed, without the blanks around it, or the value it names for a
        setting of a few values. current on Enter, checked as a typed answer is; "" on
        _CLEAR_ANSWER.
    """
    _print_help(entry)
    is_secret = entry.name.endswith("_API_KEY")
    choice_tuple = config.choice_tuple_of(entry.name)
    while True:
        answer = _ask_by_name(entry.name, current, entry.value, is_secret)
        if not answer or not choice_tuple:
            return answer
        try:
            return config.choice_of(entry.name, answer, choice_tuple)
        except ValueError as e:
            print(e)


def _ask_yes_no(question: str) -> bool | None:
    """Ask a question answered with y or n, again until the answer is one of them or Enter.

    Examples:
        "y", "Yes" -> True
        "n", "NO"  -> False
        ""         -> None

    Args:
        question: The question, with what it takes in brackets after it.

    Returns:
        True on y, False on n, None on Enter.
    """
    while True:
        answer = progress.ask(question).strip().lower()
        if answer in _YES_ANSWER_TUPLE:
            return True
        if answer in ("n", "no"):
            return False
        if not answer:
            return None
        print("answer y or n, or press Enter")


def _ask_question(entry: config.EnvTemplateEntry, current: str = "") -> str:
    """Print a setting's comment lines and ask for it with the template's question.

    Examples:
        "n" -> entry.no_value
        "y" -> current, or "" when current is entry.no_value
        ""  -> current
        "x" -> asked again

    Args:
        entry: The setting, as the template names it, with a question.
        current: The value .env holds for it, which Enter keeps; "" when it holds none.

    Returns:
        entry.no_value on n, current on Enter, and on y current unless it is
        entry.no_value, which y turns into "".
    """
    _print_help(entry)
    if current == entry.no_value:
        choice_text = "y/N"
    elif not current:
        choice_text = "Y/n"
    else:
        choice_text = f"Y/n, Enter keeps {entry.name}={current}"
    is_yes = _ask_yes_no(f"{entry.question} [{choice_text}]: ")
    if is_yes is None:
        return current
    if not is_yes:
        return entry.no_value
    return "" if current == entry.no_value else current


def _ask_endpoint_choice(entry: config.EnvTemplateEntry, choice_list: list[str],
                         current: str = "") -> str:
    """Ask which API a web tool sends its requests to, as one numbered question.

    An answer is a number or a choice's name, read as the setting reads it
    (config.setting_form); anything else is asked again after a line saying so. Enter
    gives current, which is checked as a typed answer is.

    Args:
        entry: The tool's *_PROVIDER setting, as the template names it.
        choice_list: The values that setting takes.
        current: The value a run reads for it, which Enter keeps; "" when there is none.

    Returns:
        The choice's name, or "" on _CLEAR_ANSWER and on Enter with no current.
    """
    _print_help(entry)
    for number, choice in enumerate(choice_list, 1):
        print(f"  {number}) {choice}")
    enter_text = (f"Enter keeps {current}, {_CLEAR_ANSWER} sets none" if current
                  else "Enter sets none")
    while True:
        answer = config.setting_form(entry.name, _answer_text(progress.ask(
            f"{entry.name} [1-{len(choice_list)}, {enter_text}]: ").strip(), current))
        if answer.isdecimal() and 1 <= int(answer) <= len(choice_list):
            return choice_list[int(answer) - 1]
        if answer in choice_list or not answer:
            return answer
        print(f"{entry.name} {answer} is neither a number above nor a name there")


def _ask_web_endpoint(entry_dict: dict[str, config.EnvTemplateEntry], prefix: str,
                      choice_list: list[str], key_dict: dict[str, str],
                      answer_dict: dict[str, str]) -> dict[str, str]:
    """Ask where a web tool sends its requests, and for what that choice needs.

    The API and its key answer_dict holds are read as a run reads them
    (config.run_setting_value). An API is asked for its key next, and with no key
    nothing is set. The key answer_dict holds is the one Enter keeps when the API is the
    one answer_dict names. The model's own search, and an API key_dict already holds a
    key for, need no key of their own: theirs is asked for only when answer_dict holds
    one and names that API, and Enter keeps it. When answer_dict names no API and none
    is chosen, the key is left as answer_dict holds it.

    Args:
        entry_dict: The settings _ask_entry_list() returns, by name.
        prefix: The prefix the tool's settings share: "WEB_SEARCH" or "WEB_FETCH".
        choice_list: The values the tool's *_PROVIDER setting takes.
        key_dict: {API name: key} of the keys already answered, which an API in it uses
            when it has no key of its own.
        answer_dict: The settings answered so far, those .env holds among them.

    Returns:
        The tool's *_PROVIDER and *_API_KEY settings, each "" when it is not set. A
        setting left as answer_dict holds it is not in it.
    """
    provider_name, key_name = f"{prefix}_PROVIDER", f"{prefix}_API_KEY"
    current = config.run_setting_value(provider_name, answer_dict)
    choice = _ask_endpoint_choice(entry_dict[provider_name], choice_list, current)
    if not choice:
        return {provider_name: "", key_name: ""} if current else {}
    current_key = (config.run_setting_value(key_name, answer_dict) if choice == current
                   else "")
    if choice == config.WEB_SEARCH_BY_THE_MODEL or choice in key_dict:
        return {provider_name: choice,
                key_name: current_key and _ask_setting(entry_dict[key_name], current_key)}
    key = _ask_setting(entry_dict[key_name], current_key)
    if not key:
        print(f"no key, so {provider_name} is left unset")
        return {provider_name: "", key_name: ""}
    return {provider_name: choice, key_name: key}


def _ask_web_search(entry_dict: dict[str, config.EnvTemplateEntry],
                    answer_dict: dict[str, str]) -> dict[str, str]:
    """Ask where web_search searches, and for what that choice needs.

    Args:
        entry_dict: The settings _ask_entry_list() returns, by name.
        answer_dict: The settings answered so far, those .env holds among them.

    Returns:
        The WEB_SEARCH_* settings, each "" when it is not set.
    """
    return _ask_web_endpoint(
        entry_dict, "WEB_SEARCH",
        [config.WEB_SEARCH_BY_THE_MODEL, *config.WEB_SEARCH_PRESET_DICT], {}, answer_dict)


def _ask_web_fetch(entry_dict: dict[str, config.EnvTemplateEntry],
                   answer_dict: dict[str, str]) -> dict[str, str]:
    """Ask where web_fetch reads pages through, and for what that choice needs.

    The search and its key are the ones a run reads with answer_dict written
    (config.run_setting_value). It is not asked when no search is set: the
    WEB_FETCH_* settings are then taken out when answer_dict names a
    WEB_FETCH_PROVIDER, after a line saying so, and left as answer_dict holds them
    otherwise. The API of the search, when it has a key, uses that key unless it has
    one of its own (_ask_web_endpoint).

    Args:
        entry_dict: The settings _ask_entry_list() returns, by name.
        answer_dict: The settings answered so far, the search's among them.

    Returns:
        The WEB_FETCH_* settings, each "" when it is not set. A setting left as
        answer_dict holds it is not in it.
    """
    search = config.run_setting_value("WEB_SEARCH_PROVIDER", answer_dict)
    if not search:
        if not answer_dict.get("WEB_FETCH_PROVIDER"):
            return {}
        print("no WEB_SEARCH_PROVIDER, so WEB_FETCH_PROVIDER and WEB_FETCH_API_KEY are "
              "taken out")
        return {"WEB_FETCH_PROVIDER": "", "WEB_FETCH_API_KEY": ""}
    search_key = config.run_setting_value("WEB_SEARCH_API_KEY", answer_dict)
    key_dict = {search: search_key} if search_key else {}
    return _ask_web_endpoint(entry_dict, "WEB_FETCH", list(config.WEB_FETCH_PRESET_DICT),
                             key_dict, answer_dict)


def _ask_ca_file(entry: config.EnvTemplateEntry, current: str = "") -> str:
    """Print the comment lines of the certificate file's setting and ask for its path.

    An answer that names no file, a file whose path the container that fetches
    dependencies cannot be given (config.ca_file_path_error), or a file that holds no
    PEM certificate (config.ca_file_error), is asked again.

    Examples:
        "~/corp-ca.pem" -> "/home/u/corp-ca.pem"

    Args:
        entry: The setting, as the template names it.
        current: The value .env holds for it, which Enter keeps; "" when it holds none.

    Returns:
        The file, as an absolute path with a leading "~" replaced by the home
        directory. current on Enter, checked as a typed answer is; "" on
        _CLEAR_ANSWER.
    """
    _print_help(entry)
    while True:
        answer = _ask_by_name(entry.name, current, entry.value)
        if not answer:
            return ""
        path = os.path.abspath(os.path.expanduser(answer))
        if not os.path.isfile(path):
            print(f"{path} is not a file")
            continue
        error = config.ca_file_path_error(path)
        if error:
            print(f"{path}: {error}; move the file and give its new path")
            continue
        error = config.ca_file_error(path)
        if not error:
            return path
        print(f"{path} holds no PEM certificate ({error})")


def _ask_proxy_url(entry: config.EnvTemplateEntry, current: str = "") -> str:
    """Print the comment lines of the proxy's setting and ask for its URL.

    An answer config.proxy_url_error() refuses, one without "://" that names no port
    included, is asked again after a line naming why.

    Examples:
        "http://proxy.corp.example:3128"             -> "http://proxy.corp.example:3128"
        "proxy.corp.example:3128"                    -> "proxy.corp.example:3128"
        "y"                                          -> asked again
        "socks5://proxy.corp.example:1080"           -> asked again
        "http://user:pa#ss@proxy.corp.example:3128"  -> asked again

    Args:
        entry: The setting, as the template names it.
        current: The value .env holds for it, which Enter keeps; "" when it holds none.

    Returns:
        What was typed, without the blanks around it. current on Enter, checked as a
        typed answer is; "" on _CLEAR_ANSWER.
    """
    _print_help(entry)
    while True:
        answer = _ask_by_name(entry.name, current, entry.value)
        if not answer:
            return ""
        error = config.proxy_url_error(answer, port_needed="://" not in answer)
        if not error:
            return answer
        print(f"{entry.name} {error}")


def _ask_no_proxy(entry: config.EnvTemplateEntry, https_proxy: str,
                  api_base: str, current: str = "") -> str:
    """Print the comment lines of the setting of the hosts not proxied and ask for it.

    An answer config.no_proxy_error() refuses is asked again after a line naming why.

    Examples:
        "localhost,llm.corp.example"  -> "localhost,llm.corp.example"
        "localhost,*"                 -> asked again

    Args:
        entry: The setting, as the template names it.
        https_proxy: The proxy's URL answered before it; "" when none was.
        api_base: The LLM_API_BASE the calls will go to.
        current: The value .env holds for it, which Enter keeps; "" when it holds none.

    Returns:
        What was typed, without the blanks around it. current on Enter, checked as a
        typed answer is; "" on _CLEAR_ANSWER.
    """
    _print_help(entry)
    while True:
        answer = _ask_by_name(entry.name, current, entry.value)
        if not answer:
            return ""
        error = config.no_proxy_error(answer, api_base, https_proxy)
        if not error:
            return answer
        print(f"{entry.name} {error}")


def _in_environment(name: str) -> bool:
    """Say whether the shell this process was started from holds a value for a setting.

    The setting is one of the template's (config.shell_env). A value the command line
    gave is not the shell's.

    Args:
        name: The setting's name.

    Returns:
        True when name is set and not empty, and for a name of
        config.PROXY_NAME_TUPLE also when the same name in lower case is.
    """
    shell_dict = config.shell_env()
    return bool(shell_dict.get(name)
                or (name in config.PROXY_NAME_TUPLE and shell_dict.get(name.lower())))


def _environment_holds(name_tuple: tuple[str, ...]) -> bool:
    """Say whether settings asked for together are left as the environment has them.

    A line saying so is printed when they are.

    Examples:
        ("PROJECT_DIR",) with PROJECT_DIR set
            -> True, after "PROJECT_DIR is set in the environment and is left as it is"
        ("WEB_SEARCH_PROVIDER", "WEB_SEARCH_API_KEY") with only the key set
            -> True, after "WEB_SEARCH_API_KEY is set in the environment, so
               WEB_SEARCH_PROVIDER and WEB_SEARCH_API_KEY are left as they are"

    Args:
        name_tuple: The settings' names.

    Returns:
        True when the environment holds a value for any of them (_in_environment).
    """
    env_name_list = [name for name in name_tuple if _in_environment(name)]
    if not env_name_list:
        return False
    if len(name_tuple) == 1:
        print(f"{name_tuple[0]} is set in the environment and is left as it is")
    else:
        verb = "is" if len(env_name_list) == 1 else "are"
        print(f"{' and '.join(env_name_list)} {verb} set in the environment, so "
              f"{' and '.join(name_tuple)} are left as they are")
    return True


def _ask_proxy(entry_dict: dict[str, config.EnvTemplateEntry], api_base: str,
               current_dict: dict[str, str]) -> dict[str, str]:
    """Ask whether the APIs are reached through a proxy, and for its settings.

    After a y the settings of _PROXY_SETTING_TUPLE are asked for in turn. A setting
    the environment holds a value for is not asked for and is said to be left as it is
    (_environment_holds). A proxy variable shows the value a run's HTTP clients take
    for it from current_dict, under its name in either case, and each other setting
    the value a run reads (config.run_setting_value).
    A new value, or _CLEAR_ANSWER, takes out the same name in every other case
    current_dict holds; a value kept from another case is left on its line. The hosts
    reached without the proxy are asked for only when there is a proxy: the one the
    question before gives, one config.proxy_name_list() names, or one current_dict
    holds under another name (config.PROXY_URL_NAME_TUPLE, in either case) that is not
    taken out. They are checked against that proxy and api_base (_ask_no_proxy). The
    first question takes y, n or Enter (_ask_yes_no). When current_dict holds any of
    those settings or any proxy variable (config.PROXY_NAME_TUPLE, in either case),
    Enter to it is a y, and n takes them all out.

    Args:
        entry_dict: The settings _ask_entry_list() returns, by name.
        api_base: The LLM_API_BASE the calls will go to.
        current_dict: The settings answered so far, those .env holds among them, whose
            values Enter keeps.

    Returns:
        The settings of _PROXY_SETTING_TUPLE asked for, each "" when it is not set,
        with the other cases of a name given a new value, each "". On a no the
        settings of _PROXY_SETTING_TUPLE the environment holds no value for and the
        proxy variables current_dict holds, each "". A setting left as current_dict
        holds it is not in it.
    """
    other_list = [name for name in current_dict if name not in _PROXY_SETTING_TUPLE
                  and name.upper() in config.PROXY_NAME_TUPLE]
    name_list = [name for name in _PROXY_SETTING_TUPLE
                 if not _in_environment(name)] + other_list
    is_current = any(current_dict.get(name) for name in name_list)
    print()
    is_yes = _ask_yes_no(f"{_PROXY_QUESTION} [{'Y/n' if is_current else 'y/N'}]: ")
    if not (is_current if is_yes is None else is_yes):
        return dict.fromkeys(name_list, "")
    answer_dict, https_proxy = {}, ""
    for name in _PROXY_SETTING_TUPLE:
        if _environment_holds((name,)):
            continue
        if name == _NO_PROXY_SETTING and not (
                https_proxy or config.proxy_name_list()
                or any(other.upper() in config.PROXY_URL_NAME_TUPLE
                       and other not in answer_dict for other in other_list)):
            continue
        current = config.run_setting_value(name, current_dict)
        ask = {config.CA_FILE_ENV: _ask_ca_file,
               _PROXY_URL_SETTING: _ask_proxy_url,
               _NO_PROXY_SETTING: functools.partial(
                   _ask_no_proxy, https_proxy=https_proxy,
                   api_base=api_base)}.get(name, _ask_setting)
        value = ask(entry_dict[name], current=current)
        if name == _PROXY_URL_SETTING:
            https_proxy = value
        # A value kept from the same name in another case is left on its line
        if value == current and current_dict.get(name, "") != current:
            continue
        answer_dict[name] = value
        if value != current:
            answer_dict.update({other: "" for other in other_list if other.upper() == name})
    return answer_dict


@contextlib.contextmanager
def _apply_answer(answer_dict: dict[str, str]) -> Iterator[None]:
    """Read the settings again as a run reads them with answer_dict written to .env.

    This lasts for the time of the block. Each name takes the value a run reads for it
    (config.run_value), all of them worked out before any is set; a name a run reads no
    value for from the file is left as the environment holds it. config is then read
    again. The environment and config are put back as they were when the block ends.

    Args:
        answer_dict: The settings answered so far, by name.

    Yields:
        Nothing. The block runs with the settings read again.

    Raises:
        ValueError: When the settings refuse an answer.
    """
    keep_dict = {name: os.environ.get(name)
                 for name in (*answer_dict, config.REQUESTS_CA_FILE_ENV)}
    run_dict = {name: config.run_value(name, answer_dict) for name in answer_dict}
    os.environ.update({name: value for name, value in run_dict.items() if value})
    try:
        importlib.reload(config)
        yield
    finally:
        for name, value in keep_dict.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        importlib.reload(config)


def _lack_with_answer(answer_dict: dict[str, str]) -> list[llm.MissingCredential]:
    """Return what the models would lack with answer_dict written to .env (_apply_answer).

    Args:
        answer_dict: The settings answered so far, by name.

    Returns:
        llm.missing_credential_list() under those settings.

    Raises:
        ValueError: When the settings refuse an answer.
    """
    with _apply_answer(answer_dict):
        return llm.missing_credential_list()


def _check_no_proxy_again(entry_dict: dict[str, config.EnvTemplateEntry],
                          answer_dict: dict[str, str]) -> dict[str, str]:
    """Check once more the hosts a run reaches without the proxy, with every answer.

    The hosts are those a run reads with answer_dict written, under NO_PROXY in either
    case (config.run_setting_value), checked with every answer read as a run reads it
    (_apply_answer). They are not checked here when the environment holds them
    (_in_environment). When config.no_proxy_error() refuses them they are asked for
    again: the answer takes the place of NO_PROXY in every case answer_dict holds it in.

    Args:
        entry_dict: The settings _ask_entry_list() returns, by name.
        answer_dict: All the settings answered.

    Returns:
        answer_dict, with NO_PROXY as asked again and without the same name in any
        other case; without NO_PROXY in any case when Enter is the answer.

    Raises:
        ValueError: When the settings refuse an answer.
    """
    no_proxy = config.run_setting_value(_NO_PROXY_SETTING, answer_dict)
    if not no_proxy or _in_environment(_NO_PROXY_SETTING):
        return answer_dict
    with _apply_answer(answer_dict):
        error = config.no_proxy_error(no_proxy, config.LLM_API_BASE)
        if not error:
            return answer_dict
        print()
        print(f"{_NO_PROXY_SETTING} {error}")
        no_proxy = _ask_no_proxy(entry_dict[_NO_PROXY_SETTING], "", config.LLM_API_BASE)
    answer_dict = {name: value for name, value in answer_dict.items()
                   if name.upper() != _NO_PROXY_SETTING}
    if no_proxy:
        answer_dict[_NO_PROXY_SETTING] = no_proxy
    return answer_dict


def _ask_credential(entry_dict: dict[str, config.EnvTemplateEntry],
                    answer_dict: dict[str, str]) -> dict[str, str]:
    """Ask for what the models lack under the settings answered so far.

    Each missing variable is asked for by the name it is set under
    (llm.MissingCredential.setting), once per name, after the lines saying which model
    needs it. What is missing is worked out again after every answer, with that answer
    among the settings, and a name that is no longer missing then is not asked for. A
    name entry_dict holds is asked for as its setting there; any other is the provider's
    own variable, read without echo when its name marks a credential
    (llm.is_credential_name).

    Args:
        entry_dict: The settings _ask_entry_list() returns, by name.
        answer_dict: The settings answered so far, by name.

    Returns:
        The settings to write, by name. Empty when nothing is missing or nothing was
        typed.

    Raises:
        ValueError: When the settings refuse an answer in answer_dict.
        EOFError: When the input ends before the last question.
    """
    credential_dict = {}
    ask_set = set()
    while True:
        lack_list = _lack_with_answer({**answer_dict, **credential_dict})
        setting = next((lack.setting for lack in lack_list
                        if lack.setting not in ask_set), "")
        if not setting:
            return credential_dict
        ask_set.add(setting)
        print()
        print(llm.credential_text([lack for lack in lack_list
                                   if lack.setting == setting]))
        answer = _ask_variable(entry_dict, setting)
        if answer:
            credential_dict[setting] = answer


def _ask_variable(entry_dict: dict[str, config.EnvTemplateEntry], name: str,
                  current: str = "") -> str:
    """Ask for one variable.

    It is asked for as its setting in entry_dict when entry_dict holds it, and as the
    provider's own variable otherwise, read without echo when its name marks a
    credential (llm.is_credential_name).

    Args:
        entry_dict: The settings _ask_entry_list() returns, by name.
        name: The variable's name.
        current: The value .env holds for it, which Enter keeps; "" when it holds none.

    Returns:
        What was typed, without the blanks around it. current on Enter, "" on
        _CLEAR_ANSWER.
    """
    if name in entry_dict:
        return _ask_setting(entry_dict[name], current)
    print(f"# {_PROVIDER_VARIABLE_HELP}")
    return _ask_by_name(name, current, "", llm.is_credential_name(name))


def _ask_current_credential(entry_dict: dict[str, config.EnvTemplateEntry],
                            answer_dict: dict[str, str]) -> dict[str, str]:
    """Ask again for the models' key and the providers' own variables answer_dict holds.

    The variables are LLM_API_KEY and those llm.provider_variable_tuple() names for the
    models answer_dict sets (_apply_answer). One the environment holds a value for is
    not asked for and is said to be left as it is (_environment_holds).

    Args:
        entry_dict: The settings _ask_entry_list() returns, by name.
        answer_dict: The settings answered so far, those .env holds among them.

    Returns:
        The variables asked for, each "" when it was taken out.

    Raises:
        ValueError: When the settings refuse an answer in answer_dict.
    """
    with _apply_answer(answer_dict):
        name_tuple = (_MODEL_KEY_SETTING, *llm.provider_variable_tuple())
    credential_dict = {}
    for name in dict.fromkeys(name_tuple):
        if not answer_dict.get(name) or _environment_holds((name,)):
            continue
        print()
        credential_dict[name] = _ask_variable(entry_dict, name, answer_dict[name])
    return credential_dict


def _env_file_writable(path: str) -> bool:
    """Say whether a new .env can be written at path.

    Args:
        path: Where the file would be written, as an absolute path.

    Returns:
        False when anything is already at path, a directory or a link that leads
        nowhere as well, when the directory it is in cannot be written to, or when
        git tracks a file at path or does not say whether it does
        (config.git_index_state).
    """
    return not os.path.lexists(path) \
        and os.access(os.path.dirname(path), os.W_OK | os.X_OK) \
        and config.git_index_state(path) == config.INDEX_FREE


def _env_file_ask_path() -> str:
    """Return the .env _ask_env_file() asks for and writes, when it asks at all.

    Returns:
        The .env in the current directory, as an absolute path, when no .env was read,
        the loading of .env files is not turned off (config.is_env_file_off), a terminal
        is attached (progress.at_terminal) and the file can be written
        (_env_file_writable); "" otherwise.
    """
    if config.ENV_FILE_PATH or config.is_env_file_off() or not progress.at_terminal():
        return ""
    path = os.path.join(os.getcwd(), config.ENV_FILE_NAME)
    return path if _env_file_writable(path) else ""


def _ask_env_file(is_lack: bool) -> str:
    """Ask for the settings _ask_entry_list() returns and write .env.

    The .env is the one in the current directory, asked for and written only when no
    .env was read, the loading of .env files is not turned off (config.is_env_file_off),
    a terminal is attached (progress.at_terminal) and it can be written
    (_env_file_ask_path). With nothing missing, the first question is whether to answer
    now, and anything but y writes the template as it is. A setting the template gives a
    question for is asked for with it and [Y/n]. The proxy's settings are asked for
    after one [y/N] (_ask_proxy). The WEB_SEARCH_* settings are asked for as one
    question and the WEB_FETCH_* settings as another. A setting the shell holds a value
    for is not asked for and is said to be left as it is, and the WEB_SEARCH_* or
    WEB_FETCH_* settings are left together when it holds either of them
    (_environment_holds). A setting only the command line gave a value for (--preset) is
    asked for. What the models lack is asked for last (_ask_credential), and the hosts
    reached without the proxy answered here are then checked again with every answer
    (_check_no_proxy_again).

    Args:
        is_lack: Whether the credential check found something missing.

    Returns:
        The path written, or "" when nothing was asked.

    Raises:
        SystemExit: When the input ends before the last question, or when the settings
            refuse an answer; nothing is written. Also when the file cannot be
            written (config.write_env_file), which leaves nothing at the path.
    """
    path = _env_file_ask_path()
    if not path:
        return ""
    if is_lack:
        print(f"No .env was read. Answer these to write {path}. {_NEW_KEEP_TEXT}")
        is_answer = True
    else:
        print(f"No .env was read. {path} is written now, with what is answered "
              f"here or as the template has it")
        try:
            is_answer = progress.ask(
                "Answer the settings now? [y/N]: ").strip().lower() in _YES_ANSWER_TUPLE
        except EOFError:
            print()
            raise SystemExit(f"{path} was not written: {_INPUT_END_TEXT}") from None
        if is_answer:
            print(_NEW_KEEP_TEXT)
    answer_dict = _ask_template_for(path, {}) if is_answer else {}
    try:
        config.write_env_file(path, answer_dict)
    except (OSError, ValueError) as e:
        raise SystemExit(f"{path} was not written: {e}") from None
    print(f"wrote {path}. The setup subcommand asks for its settings again")
    return path


def _ask_template_for(path: str, current_dict: dict[str, str]) -> dict[str, str]:
    """Ask for the settings _ask_entry_list() returns (_ask_template), for the .env at path.

    Args:
        path: The .env the answers are written to, as an absolute path.
        current_dict: The settings it holds, by name. Empty for a file not yet there.

    Returns:
        The settings the file is to hold, by name.

    Raises:
        SystemExit: Naming path, when the input ends before the last question or when
            the settings refuse an answer.
    """
    try:
        return _ask_template(current_dict)
    except EOFError:
        print()
        raise SystemExit(f"{path} was not written: {_INPUT_END_TEXT}") from None
    except ValueError as e:
        raise SystemExit(f"{path} was not written: {e}") from None


def _refuse_setup_place(path: str) -> None:
    """Stop setup before its first question when the .env it writes cannot be used.

    That is a .env that could not be written or would not be read.

    Args:
        path: The .env in the current directory, as an absolute path.

    Raises:
        SystemExit: With a line naming why: no terminal to ask on
            (progress.at_terminal), the loading of .env files turned off
            (config.is_env_file_off), a directory or a symbolic link at path, a
            directory this user cannot write to, or a file git tracks at path or does
            not say whether it does (config.git_index_state).
    """
    directory = os.path.dirname(path)
    if not progress.at_terminal():
        raise SystemExit("setup: asks its questions at a terminal; start it with its "
                         "input and output on one")
    if config.is_env_file_off():
        raise SystemExit(f"setup: {config.NO_ENV_FILE_ENV} turns the loading of .env "
                         f"files off, so the .env written here would not be read. "
                         f"Unset it and run setup again")
    if os.path.islink(path) or (os.path.lexists(path) and not os.path.isfile(path)):
        raise SystemExit(f"setup: {path} is a directory or a symbolic link. Move it "
                         f"away, or run setup from another directory")
    if not os.access(directory, os.W_OK | os.X_OK):
        raise SystemExit(f"setup: {directory} cannot be written to. Run setup from a "
                         f"directory this user can write to")
    state = config.git_index_state(path)
    if state == config.INDEX_HOLD:
        raise SystemExit(f"setup: {path} is tracked by git and would not be read. Take "
                         f"it out of git (git rm --cached .env), or run setup from "
                         f"another directory")
    if state == config.INDEX_UNKNOWN:
        raise SystemExit(f"setup: {path} is in a git repository git did not answer for, "
                         f"and would not be read. Run setup from another directory")


def _ask_entry_list() -> list[config.EnvTemplateEntry]:
    """Return the template's settings without a "#? skip" line."""
    return [entry for entry in config.env_template_entry_list() if not entry.skip]


def _refuse_no_question_setting(path: str, current_dict: dict[str, str]) -> None:
    """Stop setup before asking when the settings refuse a value no question asks for.

    The value is one of a setting the .env at path holds and no question of the template
    asks for: one the template does not name, or one it names with a "#? skip" line.

    Args:
        path: The .env, as an absolute path.
        current_dict: The settings it holds, by name.

    Raises:
        SystemExit: Naming path and the setting refused.
    """
    ask_name_set = {entry.name for entry in _ask_entry_list()}
    try:
        with _apply_answer({name: value for name, value in current_dict.items()
                            if name not in ask_name_set}):
            pass
    except ValueError as e:
        raise SystemExit(f"setup: {path}: {e}. setup does not ask for this setting; "
                         f"change it in {path}") from None


def _run_setup(args: argparse.Namespace) -> None:
    """Ask for the settings _ask_entry_list() returns and write them to .env.

    The .env is the one in the current directory. A .env that is not there is written as
    the first run's questions write it (config.write_env_file), with no question whether
    to answer now. A .env that is there is read for the values it holds
    (config.env_file_value_dict), and the environment is put back as the shell holds it
    (config.unload_env_file) for the questions and the checks. A value the settings
    refuse of a setting no question asks for stops it before the first question
    (_refuse_no_question_setting). Each question shows the value the file holds, which
    Enter keeps and _CLEAR_ANSWER takes out (_ask_template). Only the lines of the
    settings whose value changes are written over (config.rewrite_env_file). What the
    models lack and a proxy variable the HTTP clients cannot read
    (config.proxy_error_text) are then checked with the settings the file holds, after
    the container image is checked under them (_setup_image).

    Args:
        args: The parsed arguments of setup.

    Raises:
        SystemExit: When _refuse_setup_place() refuses the place or
            _refuse_no_question_setting() a setting, when the input ends or the settings
            refuse an answer, when the file cannot be written, and after the file is
            written, with the lines naming what the models lack or the proxy variable.
    """
    path = os.path.join(os.getcwd(), config.ENV_FILE_NAME)
    _refuse_setup_place(path)
    is_there = os.path.lexists(path)
    current_dict = config.env_file_value_dict(path) if is_there else {}
    if config.ENV_FILE_PATH:
        config.unload_env_file()
        importlib.reload(config)
    _refuse_no_question_setting(path, current_dict)
    if is_there:
        print(f"Answer these to change {path}. Enter keeps the value in brackets, and "
              f"{_CLEAR_ANSWER} takes it out")
    else:
        print(f"Answer these to write {path}. {_NEW_KEEP_TEXT}")
    answer_dict = _ask_template_for(path, current_dict)
    try:
        if not is_there:
            config.write_env_file(path, answer_dict)
            print(f"wrote {path}")
        elif config.rewrite_env_file(path, current_dict, answer_dict):
            print(f"wrote {path}")
        else:
            print(f"{path} is left as it was")
    except (OSError, ValueError) as e:
        raise SystemExit(f"{path} was not written: {e}") from None
    with _apply_answer(answer_dict):
        _setup_image(args)
        text_list = [text for text in (llm.missing_credential_text(),
                                       config.proxy_error_text()) if text]
    if text_list:
        raise SystemExit("\n".join(text_list))


def _setup_image(args: argparse.Namespace) -> None:
    """Check the container image the runs use, and build it on y (_ready_image).

    The engine is the one CONTAINER_ENGINE names, or the first of the known engines, and
    the image the one CONTAINER_IMAGE names. What stops a run is printed here and setup
    goes on.

    Args:
        args: The parsed arguments of setup.
    """
    try:
        engine_path = sandbox_mod.pick_engine()
    except RuntimeError as e:
        print(f"no container image is checked: {e}")
        return
    image_name = config.CONTAINER_IMAGE
    try:
        state = _ready_image(engine_path, image_name, True, is_enter_build=False)
    except RuntimeError as e:
        print(e)
        return
    if state == image_mod.CURRENT_STATE:
        print(f"the container image {image_name} is built from this codetwine-loom")


def _put_answer(answer_dict: dict[str, str], value_dict: dict[str, str]) -> None:
    """Set in answer_dict the settings value_dict gives a value, and take out the rest.

    The rest are the ones value_dict gives "".

    Args:
        answer_dict: The settings answered so far, by name. Changed in place.
        value_dict: The values of the settings just asked for, by name.
    """
    for name, value in value_dict.items():
        if value:
            answer_dict[name] = value
        else:
            answer_dict.pop(name, None)


def _ask_until_accept(entry: config.EnvTemplateEntry, answer_dict: dict[str, str],
                      remain_set: set[str]) -> str:
    """Ask for a setting until the settings accept it with the others answered so far.

    The answer is read with them as a run reads it (_apply_answer), and each one the
    settings refuse is followed by a line naming why. A setting the template gives a
    question for is asked for with it (_ask_question), and any other by name
    (_ask_setting). The value answer_dict holds for it, as the setting reads it
    (config.setting_form), is the one Enter keeps.

    Args:
        entry: The setting, as the template names it.
        answer_dict: The settings answered so far, those .env holds among them.
        remain_set: The settings of _ask_entry_list() not reached yet, whose values in
            answer_dict are left out of the check.

    Returns:
        The value answered; "" when it is not set.
    """
    ask = _ask_question if entry.question else _ask_setting
    check_dict = {name: value for name, value in answer_dict.items()
                  if name not in remain_set}
    while True:
        value = ask(entry, config.setting_form(entry.name,
                                               answer_dict.get(entry.name, "")))
        try:
            with _apply_answer({**check_dict, entry.name: value}):
                return value
        except ValueError as e:
            print(e)


def _ask_template(current_dict: dict[str, str]) -> dict[str, str]:
    """Ask for the settings _ask_entry_list() returns, as _ask_env_file() describes.

    Each question shows the value current_dict holds for its setting, which Enter keeps
    and _CLEAR_ANSWER takes out. A setting asked for by name or with a question is asked
    again until the settings accept its answer (_ask_until_accept). The models' key and
    the providers' own variables current_dict holds are asked for again before what the
    models lack (_ask_current_credential).

    Args:
        current_dict: The settings .env holds, by name. Empty for a .env not yet there.

    Returns:
        The settings the file is to hold, by name: current_dict with the answers.

    Raises:
        ValueError: When the settings refuse an answer.
        EOFError: When the input ends before the last question.
    """
    answer_dict = dict(current_dict)
    entry_list = _ask_entry_list()
    entry_dict = {entry.name: entry for entry in entry_list}
    remain_set = set(entry_dict)
    for entry in entry_list:
        remain_set.discard(entry.name)
        # The models' key is asked for after everything else, with the models answered
        if entry.name == _MODEL_KEY_SETTING:
            continue
        # The settings of each web tool are asked for together, where the first of
        # them stands, and are left together when the environment holds either
        if entry.name.startswith(("WEB_SEARCH_", "WEB_FETCH_")):
            prefix = entry.name.removesuffix("_PROVIDER")
            if prefix in ("WEB_SEARCH", "WEB_FETCH") and not _environment_holds(
                    (entry.name, f"{prefix}_API_KEY")):
                _put_answer(answer_dict, _ask_web_search(entry_dict, answer_dict)
                            if prefix == "WEB_SEARCH"
                            else _ask_web_fetch(entry_dict, answer_dict))
            continue
        # The proxy's settings are asked for together, where the first of them stands
        if entry.name in _PROXY_SETTING_TUPLE:
            if entry.name == _PROXY_SETTING_TUPLE[0]:
                _put_answer(answer_dict, _ask_proxy(
                    entry_dict, config.run_setting_value(_MODEL_BASE_SETTING, answer_dict),
                    answer_dict))
            continue
        if _environment_holds((entry.name,)):
            continue
        _put_answer(answer_dict,
                    {entry.name: _ask_until_accept(entry, answer_dict, remain_set)})
    _put_answer(answer_dict, _ask_current_credential(entry_dict, answer_dict))
    answer_dict.update(_ask_credential(entry_dict, answer_dict))
    return _check_no_proxy_again(entry_dict, answer_dict)


def _common_parser(project_setting: bool = True) -> argparse.ArgumentParser:
    """Return the parent parser of --project and --output-dir.

    Args:
        project_setting: Whether --project defaults to PROJECT_DIR. False leaves it
            empty unless it is passed.

    Returns:
        The parser, made with add_help=False.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project",
                        default=_SettingDefault(config.PROJECT_DIR if project_setting else ""),
                        help="path to the target repository"
                             + (" (defaults to the PROJECT_DIR environment variable)"
                                if project_setting else ""))
    common.add_argument("--output-dir", default=_SettingDefault(config.OUTPUT_DIR),
                        help="where the knowledge file and the index live "
                             "(defaults to the OUTPUT_DIR environment variable, "
                             f"itself {_help_value(config.OUTPUT_DIR)})")
    return common


def _add_build_parser(sub_parser: argparse._SubParsersAction,
                      common: argparse.ArgumentParser) -> None:
    """Add build.

    Args:
        sub_parser: The subcommands of the top parser.
        common: The parent parser of --project and --output-dir.
    """
    p_build = sub_parser.add_parser(
        "build", parents=[common],
        help="analyse the repository and build the index. codetwine runs when there "
             "is no analysis yet, when the one there does not describe what is "
             "checked out (another commit, uncommitted changes, another --exclude "
             "list), and always with --doc-create; otherwise only the index is built. "
             "Definitions and line numbers are read from every file each run; the "
             "design documents are written only for the files that changed, and only "
             "with --doc-create")
    p_build.add_argument("--doc-create", dest="with_doc", action="store_true",
                         help="have codetwine write the design documents too, for "
                              "the files that have none or have changed and for the "
                              "files depending on them. It calls the LLM several times "
                              "for each of those files, following the template "
                              "DOC_TEMPLATE_PATH names (doc_template.json of this "
                              "repository when unset). "
                              "Without it no LLM is called")
    p_build.add_argument("--exclude", dest="exclude_list", action="extend",
                         nargs="+", metavar="NAME",
                         help="a directory or file name to keep out of the analysis, "
                              "on top of codetwine's own list and of the BUILD_EXCLUDE "
                              "environment variable. A name is kept out wherever it "
                              "is; one holding a / is a path from the repository's "
                              "root (apps/old, ./vendor for vendor at the root alone, "
                              "apps/*/gen), and keeps out that file or everything "
                              "under that directory. Takes one or more names, and "
                              "can be given more than once")


def _add_search_parser(sub_parser: argparse._SubParsersAction,
                       common: argparse.ArgumentParser) -> None:
    """Add overview and search.

    Args:
        sub_parser: The subcommands of the top parser.
        common: The parent parser of --project and --output-dir.
    """
    sub_parser.add_parser(
        "overview", parents=[common],
        help="make the analysis and the index as build makes them, then make the "
             "repository's overview when it is missing or not current, which calls "
             "the LLM. Ends with status 1 when the overview could not be made. This "
             "is what the container run and sandbox start for the overview runs "
             "inside itself")
    p_search = sub_parser.add_parser("search", parents=[common],
                                     help="show the candidates. No file of the repository is "
                                          "changed; the analysis, the index and the overview "
                                          "are made first when they are missing or not "
                                          "current, and the overview calls the LLM")
    _add_request(p_search, "the change request")
    _add_overview_flag(p_search)


def _add_run_parser(sub_parser: argparse._SubParsersAction,
                    common: argparse.ArgumentParser) -> None:
    """Add run.

    Args:
        sub_parser: The subcommands of the top parser.
        common: The parent parser of --project and --output-dir.
    """
    p_run = sub_parser.add_parser(
        "run", parents=[common],
        help="read the request in a container and split it: a question about the "
             "code into the questions the agents answer one each, whose answers a "
             "further run puts together; a change into the changes the agents are "
             "given one each, whose branches a further run combines and puts into "
             "the working tree. Every container starts from the working tree as it "
             "is, committed or not")
    _add_request(p_run, "the request, as a question or as a change. With --resume "
                        "it can be left out, and the request the branches say their "
                        "run was given is used")
    _add_preset(p_run)
    _add_stage_flag(p_run)
    _add_overview_flag(p_run)
    _add_container_flag(p_run)
    p_run.add_argument("--exclude", dest="exclude_list", action="extend",
                       nargs="+", metavar="NAME",
                       help=_EXCLUDE_HELP)
    p_run.add_argument("--resume", dest="resume_list", action="append", default=[],
                       metavar="NAME",
                       help="a branch an earlier run of this left its work on. "
                            "What the commits there say - what each agent was "
                            "asked for, what it said it did and how it ended - is "
                            "put in front of the request, so this run is split up "
                            "knowing what is already done. Repeatable, and the "
                            "line the earlier run ended with says which names to "
                            "pass. The branch is read, never checked out: to "
                            "build on that code, check it out first. One branch "
                            "holding a settled merge with a stage still open is "
                            "carried on from by running that stage, when "
                            "--request is left out or is the request the branch "
                            "was made for; another request is split up again")
    p_run.add_argument("--here", action="store_true",
                       help="read the request in this repository and write the "
                            "split out instead of starting a container. This is "
                            "what the container runs inside itself")
    p_run.add_argument("--tasks-out", dest="task_out", default="", metavar="PATH",
                       help="where --here writes the split it decided on. Without "
                            "it the split is printed and nothing is written")
    p_run.add_argument("--doc-create", dest="with_doc", action="store_true",
                       help=_DOC_UPDATE_HELP + _RUN_ASK_HELP)


def _add_answer_parser(sub_parser: argparse._SubParsersAction,
                       common: argparse.ArgumentParser) -> None:
    """Add answer.

    Args:
        sub_parser: The subcommands of the top parser.
        common: The parent parser of --project and --output-dir.
    """
    p_answer = sub_parser.add_parser(
        "answer", parents=[common],
        help="answer one question about the code, reading only. With --answered it "
             "puts the agents' answers together instead. This is what the container "
             "an agent of a run runs inside itself")
    _add_request(p_answer, "the question, or with --answered the request the agents' "
                           "answers are put together for")
    _add_preset(p_answer)
    _add_overview_flag(p_answer)
    p_answer.add_argument("--answered", dest="answer_in", default="", metavar="PATH",
                          help="the file holding the agents' answers, as a run "
                               "writes it. Given, the run reads the code only where "
                               "those answers disagree or leave something open")
    p_answer.add_argument("--answer-out", default="", metavar="PATH",
                          help="where to write the answer. Without it the answer is "
                               "printed and nothing is written")


def _add_change_parser(sub_parser: argparse._SubParsersAction,
                       common: argparse.ArgumentParser) -> None:
    """Add edit and sandbox, which make the same change.

    Args:
        sub_parser: The subcommands of the top parser.
        common: The parent parser of --project and --output-dir.
    """
    # edit and sandbox make the same change; they differ in where it is made
    change = argparse.ArgumentParser(add_help=False)
    _add_request(change, "the change request")
    _add_preset(change)
    change.add_argument("--dry-run", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="have the editing tools run their checks without writing "
                             "anything. With edit what they would have written is "
                             "printed at the end and the run is not recorded in the "
                             "history; with sandbox it also stops the patch being "
                             "taken in, so nothing lands on a branch or in the working "
                             "tree. --no-dry-run writes, as passing neither does")
    change.add_argument("--check", action=argparse.BooleanOptionalAction, default=True,
                        help="hold the change to the static check (CHECK_COMMAND) and "
                             "to the references the analysis resolves: each COMPLETE() "
                             "is sent back, up to INTEGRATE_MAX_REFUSAL times, while "
                             "either finds something that was not there before the "
                             "change, and what is still found once the run has ended "
                             "makes the verdict failed, and sandbox then leaves the "
                             "change on its branch. --no-check leaves the check out")
    change.add_argument("--guide", default=_SettingDefault(config.EDIT_GUIDE),
                        metavar="PATH",
                        help="the file saying how this repository is to be changed, "
                             "relative to it; an absolute path, or one that leads out "
                             "of the repository with .., names a file kept elsewhere "
                             "(defaults to the EDIT_GUIDE environment "
                             f"variable, itself {_help_value(config.EDIT_GUIDE or 'empty')}). "
                             "It is put in front of the model when it is there and "
                             "nothing is added when it is not. Pass an empty string to "
                             "ignore it")
    p_edit = sub_parser.add_parser("edit", parents=[common, change], help="make the change")
    _add_overview_flag(p_edit)
    # edit's alone: it is asked at a terminal
    resume_group = p_edit.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=None,
                              help="carry on from the change the previous run left "
                                   "unfinished")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false",
                              help="start fresh, ignoring any unfinished change")

    p_sandbox = sub_parser.add_parser(
        "sandbox", parents=[common, change],
        help="make the change in a container, against a read-only copy of the "
             "repository's history, and put the change it produces into the working tree. The "
             "container starts from the working tree as it is, committed or not; the "
             "change is applied to it when the run gets through, for you to commit")
    _add_container_flag(p_sandbox)
    p_sandbox.add_argument("--exclude", dest="exclude_list", action="extend",
                           nargs="+", metavar="NAME",
                           help=_EXCLUDE_HELP)
    p_sandbox.add_argument("--doc-create", dest="with_doc", action="store_true",
                           help=_DOC_UPDATE_HELP)
    _add_overview_flag(p_sandbox)


def _add_integrate_parser(sub_parser: argparse._SubParsersAction,
                          common: argparse.ArgumentParser) -> None:
    """Add integrate.

    Args:
        sub_parser: The subcommands of the top parser.
        common: The parent parser of --project and --output-dir.
    """
    p_integrate = sub_parser.add_parser(
        "integrate", parents=[common],
        help="combine branches into the working tree, in a container. What the tree "
             "holds, committed or not, is combined along with them, and the union is "
             "applied to the tree when the machine finds it settled, for you to commit")
    _add_preset(p_integrate)
    _add_stage_flag(p_integrate)
    p_integrate.add_argument("--branch", dest="branch_list", action="append", default=[],
                             metavar="NAME",
                             help="a branch to combine. Repeatable, and they are "
                                  "merged in the order they are passed. Two or more, "
                                  "unless --resume names the branch an earlier merge "
                                  "left")
    p_integrate.add_argument("--resume", default="", metavar="NAME",
                             help="the branch an earlier merge left. One left "
                                  "unsettled is combined as it is, along with the "
                                  "working tree, and the merging run carries on from "
                                  "there; every branch it reaches is read for what it "
                                  "was asked. One settled with a stage after it left "
                                  "open runs that stage over what it put into the "
                                  "tree, when no --branch is passed and a stage from "
                                  "there on is on. Otherwise the commits of stages "
                                  "nobody settled are taken off the branch and what is "
                                  "under them is combined. With --here the commit "
                                  "checked out is what it is combined onto")
    _add_container_flag(p_integrate)
    p_integrate.add_argument("--exclude", dest="exclude_list", action="extend",
                             nargs="+", metavar="NAME",
                             help=_EXCLUDE_HELP)
    p_integrate.add_argument("--onto", default="", metavar="NAME",
                             help="with --here, the branch the union is written "
                                  "against, combined along with the others. The host "
                                  "passes the branch pointing at what its working tree "
                                  "holds")
    p_integrate.add_argument("--onto-commit", default="", metavar="COMMIT",
                             help="with --here, the commit of --onto to write the "
                                  "union against. The host passes the commit it read "
                                  "before the container was started, so a branch that "
                                  "moves in between does not move what the union is "
                                  "written for")
    _add_request(p_integrate, "the request the branches were made for. It is put at "
                              "the head of what the merging stage is asked, so that it "
                              "knows what they were split out of")
    p_integrate.add_argument("--outcome-out", default="", metavar="PATH",
                             help="where --here writes what it made of the merge. "
                                  "Without it nothing is written and the commit "
                                  "carries no account of how the merge ended")
    p_integrate.add_argument("--gap-in", default="", metavar="PATH",
                             help="the cases runs of the stage that writes tests recorded "
                                  "as not done, with their tests, which --here runs "
                                  "again over the merge once it is settled. The ones "
                                  "whose test passes are left out of what "
                                  "--outcome-out records")
    p_integrate.add_argument("--here", action="store_true",
                             help="combine the branches in this repository instead of "
                                  "starting a container. This is what the container "
                                  "runs inside itself")
    p_integrate.add_argument("--doc-create", dest="with_doc", action="store_true",
                             help=_BUILD_DOC_HELP)


def _add_stage_parser(sub_parser: argparse._SubParsersAction,
                      common: argparse.ArgumentParser) -> None:
    """Add test-create, refactor and test-refactor, the stages after a merge.

    Args:
        sub_parser: The subcommands of the top parser.
        common: The parent parser of --project and --output-dir.
    """
    p_test_create = sub_parser.add_parser(
        "test-create", parents=[common],
        help="write tests for the definitions the change HEAD holds touched, in this "
             "repository. This is what the container started after a settled merge "
             "runs inside itself when TEST_CREATE is set")
    _add_preset(p_test_create)
    p_test_create.add_argument("--base", required=True, metavar="COMMIT",
                               help="the commit the change HEAD holds was written "
                                    "against. What HEAD changed against it is what the "
                                    "tests are written for")
    p_test_create.add_argument("--request", default="", metavar="TEXT",
                               help="the request the change was made for, which the "
                                    "tests take what they expect from")
    p_test_create.add_argument("--outcome-out", default="", metavar="PATH",
                               help="where what the stage made of the tests is written, "
                                    "or with --target-only what it listed. Without it "
                                    "nothing is written")
    p_test_create.add_argument("--target-in", default="", metavar="PATH",
                               help="a JSON file naming the files whose definitions "
                                    "this run writes tests for and the tests failing "
                                    "before it ({\"paths\", \"test_paths\", \"baseline\"}). "
                                    "Without it every file the change touched is taken "
                                    "and the tests are run to read which fail")
    p_test_create.add_argument("--target-only", action="store_true",
                               help="list the definitions the change touched, the "
                                    "tests failing now and the language taken as the "
                                    "primary one (empty for the index's), call no model "
                                    "and write nothing to the repository")

    for stage, what, target in (
            (integrate_mod.REFACTOR_STAGE, "refactor the files the change HEAD holds "
             "touched outside the tests", "REFACTOR"),
            (integrate_mod.TEST_REFACTOR_STAGE, "refactor the tests the change HEAD "
             "holds added or changed", "TEST_REFACTOR")):
        p_refactor = sub_parser.add_parser(
            stage, parents=[common],
            help=f"{what}, in this repository. This is what the container started "
                 f"after a settled merge runs inside itself when {target} is set")
        _add_preset(p_refactor)
        p_refactor.add_argument("--base", required=True, metavar="COMMIT",
                                help="the commit the change HEAD holds was written "
                                     "against. What HEAD changed against it is what is "
                                     "refactored")
        p_refactor.add_argument("--request", default="", metavar="TEXT",
                                help="the request the change was made for")
        p_refactor.add_argument("--outcome-out", default="", metavar="PATH",
                                help="where what the stage made of its run is written. "
                                     "Without it nothing is written")


def _add_tool_parser(sub_parser: argparse._SubParsersAction,
                     common: argparse.ArgumentParser) -> None:
    """Add warm, run-tests, broker, format, history and setup.

    Args:
        sub_parser: The subcommands of the top parser.
        common: The parent parser of --project and --output-dir.
    """
    sub_parser.add_parser(
        "warm", parents=[common],
        help="fill the package cache with a network, no key of this tool and no model, "
             "in a container of its own, so a sandboxed run afterwards finds its "
             "dependencies already there")

    sub_parser.add_parser("run-tests", parents=[common],
                          help="run the test commands once over the repository (the primary "
                               "language's, and those of the other languages whose tests "
                               "are there) and say what they did, making the tests' "
                               "directory where the tests are reached by path and it is not "
                               "there. Ends with status 1 when a command could not be "
                               "started, or reached no verdict and printed its language's "
                               "line for a tool that ended before it got to the tests")

    p_broker = sub_parser.add_parser(
        "broker", parents=[_common_parser(project_setting=False)],
        help="answer the model calls of a container that has no network and no "
             "credentials, using the LLM_API_KEY of this machine. With --project it "
             "also fills that repository's package cache again when a run asks")
    p_broker.add_argument("--endpoint", default=_SettingDefault(config.LLM_BROKER),
                          help="where to listen: unix://<path> or tcp://<host>:<port> "
                               "(defaults to the LLM_BROKER environment variable)")
    p_broker.add_argument("--agent", default="", metavar="NAME",
                          help="the agent this broker serves, which the sandbox "
                               "subcommand passes for the run it started: a refill "
                               "is made for that agent, and the broker's log is kept "
                               "with what this machine writes for it. Without it a "
                               "refill is an agent of its own and the log is kept "
                               "under --output-dir")
    p_broker.add_argument("--run", default="", metavar="ID",
                          help="the run this broker serves, which the sandbox "
                               "subcommand passes for the run it started: a refill "
                               "takes that run's dependency files, and the broker ends "
                               "when the process of that run has ended. Without it a "
                               "refill takes none of a running change's files, and "
                               "the broker serves until it is stopped")
    p_broker.add_argument("--base", default="", metavar="COMMIT",
                          help="the commit a refill checks its clone out at, which "
                               "the sandbox subcommand passes: the commit the working "
                               "tree made when its command started. Without it a "
                               "refill checks it out at the commit the working tree "
                               "makes then")
    p_broker.add_argument("--fill-ext", default="", metavar="LANGS",
                          help="the languages, comma-separated, whose default test "
                               "commands a refill runs over its clone as well, which "
                               "sandbox, run and integrate pass for the languages the "
                               "container's commit adds. Without it a refill runs none "
                               "of them")
    p_broker.add_argument("--token-out", default="", metavar="PATH",
                          help="the file the token counts of this broker's model calls "
                               "are written to after each call, which the run passes "
                               "to add them to its own. Without it none is written")
    p_broker.add_argument("--token-left", type=int, default=0, metavar="N",
                          help="how many tokens this broker's model calls and those of "
                               "the brokers writing their counts beside --token-out "
                               "may use between them, which the run passes for "
                               "MAX_TOKEN_PER_RUN. A completion or a search asked for "
                               "after that many is refused. 0 sets no limit")
    p_broker.add_argument("--label", default="", metavar="TEXT",
                          help="what every line this broker logs starts with, which "
                               "the run passes for the agent the broker serves")

    sub_parser.add_parser(
        "format", parents=[common],
        help="run FORMAT_COMMAND over the files changed since HEAD, from the "
             "repository root. This is what a container runs once its run has ended, "
             "before it writes its patch. It always ends with status 0")

    sub_parser.add_parser("history", parents=[common],
                          help="show the run, sandbox and integrate commands that left a "
                               "record, with what to pass to carry on from each, and what "
                               "the previous runs changed")

    sub_parser.add_parser(
        SETUP_COMMAND,
        help="ask for the settings of .env in the current directory, as the first run "
             "does, and write it. With a .env there, each question shows the value it "
             f"holds, Enter keeps it and {_CLEAR_ANSWER} takes it out, and only the "
             "lines of the settings changed are written over. The container image is "
             "then checked, and built when it is missing or stale and the answer is y")


def _check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse the combinations argparse cannot refuse on its own.

    Args:
        parser: The top parser, which says what is wrong and exits.
        args: The parsed arguments.
    """
    if args.command == "integrate" and (args.onto or args.onto_commit) and not args.here:
        parser.error("--onto and --onto-commit are what --here is told; without --here "
                     "the union is written against the working tree")
    if args.command == "run" and args.task_out and not args.here:
        parser.error("--tasks-out is where --here writes the split, so it needs --here; "
                     "without --here the split is printed")
    if args.command == "integrate" and not args.branch_list and not args.resume:
        parser.error("integrate needs --branch once for each branch to combine, or "
                     "--resume with the branch an earlier merge left")
    if args.command == "broker" and (args.run or args.base or args.agent
                                     or args.fill_ext) and not args.project:
        parser.error("--agent, --run, --base and --fill-ext say what a refill fills the "
                     "cache for, and a broker refills only with --project")
    if args.command == "broker" and args.agent and (
            args.agent.startswith(".")
            or args.agent != sandbox_mod.agent_name(args.agent, "")):
        parser.error("--agent names one directory under the output directory: letters, "
                     "digits, '-', '_' and '.', not starting with '.'")
    if args.command == "integrate" and (args.outcome_out or args.gap_in) \
            and not args.here:
        parser.error("--outcome-out and --gap-in are what --here is told; without --here "
                     "the run says how it ended on the terminal")
    if args.command == "integrate" and args.onto_commit and not args.onto:
        parser.error("--onto-commit is the commit of --onto, so it needs --onto")
    if args.command == "integrate" and args.gap_in and not args.outcome_out:
        parser.error("the cases --gap-in names are written back to --outcome-out, so it "
                     "needs --outcome-out")
    if args.command == "test-create" and args.target_only \
            and (args.target_in or args.request):
        parser.error("--target-only lists what there is to write tests for and writes "
                     "no test, so it reads neither --target-in nor --request")
    if getattr(args, "here", False):
        skip_flag_list = _here_skip_flag_list(args)
        if skip_flag_list:
            parser.error(f"{', '.join(skip_flag_list)} are read where the containers are "
                         f"started; --here runs in this repository and starts none"
                         + (", and runs no stage after the merge"
                            if args.command == "integrate" else ""))
    if hasattr(args, "engine"):
        args.engine = config.CONTAINER_ENGINE if args.engine is None else args.engine
        args.image = config.CONTAINER_IMAGE if args.image is None else args.image


def _here_skip_flag_list(args: argparse.Namespace) -> list[str]:
    """Return the flags passed with --here that --here does not read.

    Args:
        args: The parsed arguments of run or integrate with --here.

    Returns:
        The flags as they are written on the command line, in order.
    """
    skip_flag_list = [flag for flag, value in (("--engine", args.engine),
                                               ("--image", args.image))
                      if value is not None]
    if args.with_doc:
        skip_flag_list.append("--doc-create")
    if args.command == "integrate":
        skip_flag_list += [f"--no-{name.lower().replace('_', '-')}"
                           for name in _STAGE_FLAG_DICT
                           if getattr(args, name.lower()) is not None]
    return skip_flag_list


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the arguments of the subcommands.

    Not every argument is on every subcommand (--doc-create and --exclude are on build,
    run, sandbox and integrate, --dry-run and --guide on edit and sandbox). --doc-create
    is read as args.with_doc.

    The argument after --request or --request-file is its value whatever it starts with
    (_join_request_flag). What argparse cannot say on its own is checked afterwards
    (_check_args). The text --request-file names is then read into args.request
    (_read_request_file).

    Args:
        argv: The argument list. None uses sys.argv.

    Returns:
        argparse's Namespace, with the settings the command line gave a value to
        under "option_setting_set" (_option_setting_set).
    """
    parser = argparse.ArgumentParser(
        prog="codetwine-loom",
        description="Search the codetwine analysis for the scope of a change, "
                    "then make the change or answer the question")
    sub_parser = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    for add in (_add_build_parser, _add_search_parser, _add_run_parser,
                _add_answer_parser, _add_change_parser, _add_integrate_parser,
                _add_stage_parser, _add_tool_parser):
        add(sub_parser, common)

    args = parser.parse_args(_join_request_flag(
        sys.argv[1:] if argv is None else argv, _flag_set(parser)))
    args.option_setting_set = _option_setting_set(args)
    _check_args(parser, args)
    _read_request_file(parser, args)
    return args


# The subcommands that take --exclude, and decide what the analysis keeps out
_EXCLUDE_DECIDE_TUPLE = ("build", "sandbox", "run", "integrate")

# The subcommands that never call a model, whatever they are passed
_NO_MODEL_TUPLE = ("history", "warm", "run-tests", "format")


def calls_a_model(args: argparse.Namespace) -> bool:
    """Say whether this process, or a broker it starts, makes a model call for args.

    Args:
        args: The parsed arguments.

    Returns:
        False for the subcommands in _NO_MODEL_TUPLE, for build without --doc-create,
        for search with QUERY_EXPAND off, for overview when no overview is made
        (overview.is_on), and for test-create with --target-only; True otherwise.
    """
    if args.command in _NO_MODEL_TUPLE:
        return False
    if args.command == "build":
        return args.with_doc
    if args.command == "search":
        return config.QUERY_EXPAND
    if args.command == "overview":
        return overview.is_on()
    if args.command == "test-create":
        return not args.target_only
    return True


def exclude_name_list(args: argparse.Namespace) -> list[str]:
    """Return the names and paths kept out of the analysis, the setting plus the arguments.

    The setting comes first and an entry given twice appears once: the same list is
    built whether the entries arrived through BUILD_EXCLUDE or through --exclude. Each
    is put in the form knowledge.exclude_entry() returns.

    Args:
        args: The parsed arguments. A subcommand without --exclude contributes nothing.

    Returns:
        The entries, in the order they were first seen.

    Raises:
        SystemExit: When one of them is empty, starts at the file system's root, or
            holds a ".." part.
    """
    entry_list = []
    for text in [*config.BUILD_EXCLUDE, *(getattr(args, "exclude_list", None) or [])]:
        try:
            entry = knowledge_mod.exclude_entry(text)
        except ValueError as e:
            raise SystemExit(f"--exclude and BUILD_EXCLUDE take directory or file names, "
                             f"and paths relative to the repository: {e}") from None
        if entry not in entry_list:
            entry_list.append(entry)
    return entry_list


def _set_up_log() -> logging.Handler:
    """Send the log to stderr at AGENT_LOG_LEVEL, holding library lines back below DEBUG.

    Returns:
        The handler that writes to stderr. In a process started without standard error
        it is one that writes nothing.
    """
    handler = logging.StreamHandler() if sys.stderr is not None else logging.NullHandler()
    handler.setFormatter(progress.LabelFormatter(progress.TERM_LOG_FORMAT))
    logging.basicConfig(level=config.AGENT_LOG_LEVEL, handlers=[handler])
    # Below DEBUG the terminal shows the run's progress and warnings, and the libraries'
    # own lines are held back
    if config.AGENT_LOG_LEVEL != "DEBUG":
        handler.addFilter(progress.TermFilter())
        for name in _LIBRARY_LOGGER_TUPLE:
            logging.getLogger(name).setLevel(logging.WARNING)
    return handler


# How many setting names are written out before the rest are counted
_MAX_SHOW_SETTING = 8


def _refuse_skip_env_setting(args: argparse.Namespace) -> None:
    """Stop a command whose .env file was left unread and sets settings of this tool.

    A .env file left unread that sets none of them is the repository's own and stops
    nothing. A setting the command line gave a value to is not counted.

    Args:
        args: The parsed arguments, which hold those settings (option_setting_set).

    Raises:
        SystemExit: Naming the file, the settings (config.skip_env_setting_list())
            and what to do, when there are any.
    """
    name_list = [name for name in config.skip_env_setting_list()
                 if name not in args.option_setting_set]
    if not name_list:
        return
    name_text = ", ".join(name_list[:_MAX_SHOW_SETTING])
    if len(name_list) > _MAX_SHOW_SETTING:
        name_text += f" and {len(name_list) - _MAX_SHOW_SETTING} more"
    if config.ENV_FILE_SKIP_STATE == config.INDEX_HOLD:
        why = "git tracks the file"
        where = "keep them in a .env git does not track and run the command from its directory"
    else:
        why = ("the file is in a git repository, and git did not say whether it tracks "
               "it (git failed there, as it does in a repository another user owns, or "
               "did not answer in time)")
        where = "run the command from a directory outside the repository with the .env there"
    raise SystemExit(
        f"not started: {config.ENV_FILE_SKIP_PATH} sets {name_text}, and it is not "
        f"read: {why}. Set them in the shell, or {where}. To run without them, set "
        f"{config.NO_ENV_FILE_ENV}={config.NO_ENV_FILE_VALUE}")


def _refuse_empty_request(args: argparse.Namespace) -> None:
    """Stop a subcommand whose request to carry out is empty.

    Args:
        args: The parsed arguments.

    Raises:
        SystemExit: When the request is empty or blank.
    """
    # Every subcommand that is given a request to search for or carry out is refused an
    # empty one here. integrate takes one too, as what the branches were made for, and
    # an empty one is not refused. A run given --resume and no --request at all is
    # passed on
    is_read_later = (args.command == "run" and args.request == ""
                     and bool(args.resume_list))
    if args.command in _REQUEST_COMMAND_TUPLE and not args.request.strip() \
            and not is_read_later:
        raise SystemExit(f"{args.command}: the request is empty. Pass it with --request, "
                         f"or with --request-file as a file or the standard input")


def _refuse_unreadable_request(args: argparse.Namespace) -> None:
    """Stop a subcommand whose request holds what cannot be passed on as text.

    Args:
        args: The parsed arguments. A subcommand without a request is left alone.

    Raises:
        SystemExit: When the request holds a NUL character, or a character UTF-8
            cannot encode, which is what bytes that are not UTF-8 are read as.
    """
    request = getattr(args, "request", "")
    try:
        request.encode("utf-8")
    except UnicodeEncodeError:
        raise SystemExit(f"{args.command}: the request is not UTF-8 text. Pass it as "
                         f"UTF-8, with --request or --request-file") from None
    if "\0" in request:
        raise SystemExit(f"{args.command}: the request holds a NUL character. Take it "
                         f"out and pass the request again")


# How many bytes the request of a subcommand that starts containers may hold
_REQUEST_MAX_BYTE = 131072 - 64


def _refuse_long_request(args: argparse.Namespace) -> None:
    """Stop a subcommand that starts containers when its request is too long.

    Args:
        args: The parsed arguments.

    Raises:
        SystemExit: When the request holds more than _REQUEST_MAX_BYTE bytes as UTF-8.
    """
    byte_count = len(getattr(args, "request", "").encode("utf-8"))
    if _starts_container(args) and byte_count > _REQUEST_MAX_BYTE:
        raise SystemExit(
            f"{args.command}: the request holds {byte_count:,} bytes, and a container "
            f"is handed it on its command line, which takes at most "
            f"{_REQUEST_MAX_BYTE:,}. Shorten it, or keep the detail in a file of the "
            f"repository and name that file in the request")


def _refuse_no_project(args: argparse.Namespace) -> None:
    """Stop a subcommand that reads a repository when none is named.

    Args:
        args: The parsed arguments.

    Raises:
        SystemExit: When neither --project nor PROJECT_DIR names one, for every
            subcommand but broker.
    """
    if args.command != "broker" and not args.project:
        raise SystemExit(f"{args.command}: pass --project, or set PROJECT_DIR "
                         f"(--project . for the current directory)")


def _refuse_wrong_place(args: argparse.Namespace) -> None:
    """Stop a subcommand whose repository is not there or whose output directory cannot be.

    Every subcommand has to name a directory with --project, but a broker started
    without one, which is checked for nothing here. format, which reads the working tree
    alone, has to name the root of a git repository. Every other one is refused an
    output directory inside the repository or holding it (knowledge.check_output_place),
    the ones that start containers and broker a SANDBOX_CACHE_DIR in such a place
    (sandbox.check_cache_place), and every one of those but history and broker has the
    output directory made and is refused one that cannot be written to
    (knowledge.make_output_dir).

    Args:
        args: The parsed arguments.

    Raises:
        SystemExit: With "<command>: <what is wrong>".
    """
    if args.command == "broker" and not args.project:
        return
    try:
        knowledge_mod.check_project_dir(args.project)
        project_dir = os.path.abspath(args.project)
        if args.command == "format":
            knowledge_mod.check_git_root(project_dir)
            return
        knowledge_mod.check_output_place(project_dir, os.path.abspath(args.output_dir))
        if _starts_container(args) or args.command == "broker":
            sandbox_mod.check_cache_place(project_dir, os.path.abspath(args.output_dir))
        if args.command not in ("history", "broker"):
            knowledge_mod.make_output_dir(args.output_dir)
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f"{args.command}: {e}") from None


# The arguments that name a file a subcommand writes, each with the flag it is passed
# with
_OUT_FILE_FLAG_DICT = {"answer_out": "--answer-out", "task_out": "--tasks-out",
                       "outcome_out": "--outcome-out", "token_out": "--token-out"}


def _refuse_unwritable_out(args: argparse.Namespace) -> None:
    """Stop a subcommand told to write a file where none can be written.

    The directory of each file _OUT_FILE_FLAG_DICT names is made, before anything
    runs (sandbox.check_across_path).

    Args:
        args: The parsed arguments.

    Raises:
        SystemExit: With "<command>: <flag> <what is wrong>".
    """
    for name, flag in _OUT_FILE_FLAG_DICT.items():
        path = getattr(args, name, "")
        if not path:
            continue
        try:
            sandbox_mod.check_across_path(path)
        except RuntimeError as e:
            raise SystemExit(f"{args.command}: {flag} {e}") from None


def _refuse_no_base(args: argparse.Namespace) -> None:
    """Stop test-create, refactor and test-refactor when --base names no commit.

    A --project that is not the root of a git repository is not checked here.

    Args:
        args: The parsed arguments.

    Raises:
        SystemExit: When the repository holds no commit under the name --base was
            given.
    """
    if args.command not in _BASE_COMMAND_TUPLE:
        return
    project_dir = os.path.abspath(args.project)
    if gitcmd.is_work_tree_root(project_dir) \
            and not gitcmd.has_commit(project_dir, args.base):
        raise SystemExit(
            f"{args.command}: --base {args.base} names no commit in {project_dir}. "
            f"Pass the commit the change HEAD holds was written against; the commits "
            f"are listed by: {command_line('git', '-C', project_dir, 'log')}")


# The subcommands that never run the tests
_NO_TEST_RUN_TUPLE = ("broker", "history", "format", "build", "search", "overview",
                      "answer")


def _refuse_quiet_test_command(args: argparse.Namespace) -> None:
    """Stop a subcommand that may run the tests when a test command is too quiet.

    That is, with RUN_TEST on, a test command whose own command line keeps pytest from
    printing the line that counts its tests (edit.quiet_test_error).

    Args:
        args: The parsed arguments.

    Raises:
        SystemExit: With the line naming the command and what to change.
    """
    if args.command in _NO_TEST_RUN_TUPLE or not config.RUN_TEST:
        return
    error = edit.quiet_test_error()
    if error:
        raise SystemExit(f"{args.command}: {error}")


def _starts_container(args: argparse.Namespace) -> bool:
    """Return whether the subcommand starts containers from this machine.

    Args:
        args: The parsed arguments.

    Returns:
        True for sandbox and warm, and for run and integrate without --here.
    """
    return args.command in _CONTAINER_START_COMMAND_TUPLE \
        and not getattr(args, "here", False)


def _refuse_no_engine_or_image(args: argparse.Namespace) -> None:
    """Stop a subcommand that starts containers when its engine or its image is not there.

    The engine is looked for as sandbox.pick_engine() looks for it: the one --engine or
    CONTAINER_ENGINE names, or the first of the known engines. The image is the one
    --image or CONTAINER_IMAGE names. The questions of the start of a run are asked
    (_ask_before_build), and the engine is then asked for the image (_ready_image),
    which may build it, and, for every one of them but warm, whether
    the network SANDBOX_NETWORK names is internal (sandbox.refuse_open_network). Before
    the engine is looked for, the file of certificate authorities is checked for a path
    a container can be given (sandbox.refuse_ca_file_path). The image is not asked for
    while the repository is not a directory, which a later check stops on, and, for a
    run that checks the credentials (_needs_credential), while the first run's questions
    are still to write .env (_check_credential asks for it once the file is read) or a
    credential is missing (_check_credential stops on it).

    Args:
        args: The parsed arguments. args.image is set to the image picked.

    Raises:
        SystemExit: When the engine asked for is not on PATH, when none of the known
            engines is, when the engine does not have the image and it is not built
            here, or when one of the other checks stops.
    """
    if not _starts_container(args):
        return
    try:
        sandbox_mod.refuse_ca_file_path()
        engine_path = sandbox_mod.pick_engine(getattr(args, "engine", ""))
        args.image = getattr(args, "image", None) or config.CONTAINER_IMAGE
        if args.project and os.path.isdir(os.path.abspath(args.project)) and not (
                _needs_credential(args) and (
                    _env_file_ask_path() or llm.missing_credential_text())):
            _ask_before_build(args)
            _ready_image(engine_path, args.image, progress.at_terminal())
        # warm's container is not put on that network
        if args.command != "warm":
            sandbox_mod.refuse_open_network(engine_path)
    except RuntimeError as e:
        raise SystemExit(f"{args.command}: {e}") from None


# What says how to get the Dockerfile an image is built from
_NO_SOURCE_TEXT = ("the image is built from the Dockerfile of a clone of codetwine-loom's "
                   "repository, and there is none beside this installation. Clone the "
                   "repository and install codetwine-loom from the clone, or pull the "
                   "image, or set CONTAINER_IMAGE to an image the engine has")

# What says how an image was found stale
_STALE_TEXT = "it carries no mark of the files it was built from, or the mark of others"


def _ask_before_build(args: argparse.Namespace) -> None:
    """Ask the questions of the start of a run, and check what its answers need, first.

    run asks whether the first analysis makes the design documents too
    (knowledge.ask_doc), when there is no analysis yet and --doc-create is not passed,
    and run and integrate ask which dialect the static check reads COBOL in
    (knowledge.ask_cobol_dialect). The answers are kept, and the analysis made later
    asks nothing again. Nothing is asked with no terminal, or when the output directory
    is one the run refuses (knowledge.check_output_place). When the documents are to be
    made, by --doc-create or by the answer, what they need is checked
    (knowledge.check_doc_setting); integrate, which makes them only with an analysis it
    makes, checks nothing while the analysis is there.

    Args:
        args: The parsed arguments.

    Raises:
        RuntimeError: When the documents are to be made and what they need is not set.
    """
    if args.command not in ("run", "integrate", "sandbox"):
        return
    project_dir = os.path.abspath(args.project)
    out_dir = os.path.abspath(args.output_dir)
    try:
        knowledge_mod.check_output_place(project_dir, out_dir)
    except RuntimeError:
        return
    name = knowledge_mod.project_name_of(project_dir)
    knowledge_path = knowledge_mod.knowledge_db_path(out_dir, name)
    is_analysed = knowledge_mod.is_complete(knowledge_path)
    with_doc = getattr(args, "with_doc", False)
    if args.command in ("run", "integrate") and progress.at_terminal():
        if args.command == "run" and not with_doc and not is_analysed:
            with_doc = knowledge_mod.ask_doc(knowledge_path)
        knowledge_mod.ask_cobol_dialect(store.index_db_path(out_dir, name), project_dir,
                                        exclude_name_list(args))
    # integrate makes the documents only with the analysis it makes when there is none
    if with_doc and not (args.command == "integrate" and is_analysed):
        knowledge_mod.check_doc_setting()


def _is_default_image(image_name: str) -> bool:
    """Say whether an image name is the default of CONTAINER_IMAGE.

    Examples:
        "codetwine-loom", "localhost/codetwine-loom:latest" -> True
        "codetwine-loom:java"                               -> False

    Args:
        image_name: The image's name.

    Returns:
        True when the name, without a leading "localhost/" and a trailing ":latest", is
        config.default_of("CONTAINER_IMAGE").
    """
    name = image_name.removeprefix("localhost/").removesuffix(":latest")
    return name == config.default_of("CONTAINER_IMAGE")


def _ready_image(engine_path: str, image_name: str, is_ask: bool,
                 is_enter_build: bool = True) -> str | None:
    """Make sure the engine has the image, built from this installation of codetwine-loom.

    The image is missing when the engine does not show it, and stale when its mark
    (image.SOURCE_LABEL) is missing or is not image.source_mark() of this installation.
    A missing or stale image is built (_build_image) with image.DEFAULT_TOOLCHAIN, or,
    when it is not the default image (_is_default_image), with the TOOLCHAIN its label
    names where it has one. With is_ask, what is to be built is printed and [Y/n] asked
    first ([y/N] without is_enter_build). Without is_ask, or on n, a missing image stops
    and a stale one is warned about, with the TOOLCHAIN its label names. A stale image is
    passed as it is when there is no Dockerfile to build from (image.source_dir).

    Args:
        engine_path: The engine's executable, as pick_engine() returns it.
        image_name: The image's name.
        is_ask: Whether to ask before building.
        is_enter_build: Whether Enter is taken as y rather than as n.

    Returns:
        image.CURRENT_STATE when the image was there and current, the state it was in
        when it was built or passed, or None when the engine could not be asked.

    Raises:
        RuntimeError: When the image is missing and not built, when it cannot be built
            here, when the certificate authorities of config.CA_FILE_ENV cannot be
            passed to its build (image.ca_pem_file), or when the build ends with a
            non-zero status. The message says what to run.
    """
    repo_dir = image_mod.source_dir()
    mark = image_mod.source_mark(repo_dir) if repo_dir else ""
    try:
        label_dict = sandbox_mod.image_label_dict(engine_path, image_name)
        absent_text = ""
    except sandbox_mod.NoImageError as e:
        label_dict, absent_text = None, str(e)
        if not repo_dir:
            raise RuntimeError(f"{absent_text}\n{_NO_SOURCE_TEXT}") from None
    else:
        if label_dict is None:
            return None
    state = image_mod.image_state(label_dict, mark)
    if state == image_mod.CURRENT_STATE or not repo_dir:
        return state
    label_toolchain = (label_dict or {}).get(image_mod.TOOLCHAIN_LABEL, "")
    toolchain = (not _is_default_image(image_name) and label_toolchain) \
        or image_mod.DEFAULT_TOOLCHAIN
    try:
        command_text = image_mod.build_command_text(engine_path, image_name, toolchain,
                                                    mark, repo_dir)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"the image {image_name} cannot be built with the certificate "
                           f"authorities of {config.CA_FILE_ENV}: {e}") from None
    what_text = f"TOOLCHAIN={toolchain}"
    is_absent = state == image_mod.ABSENT_STATE
    if is_ask:
        if is_absent:
            print(f"{os.path.basename(engine_path)} does not have the image {image_name} "
                  f"({what_text}).")
        else:
            print(f"The image {image_name} ({what_text}) was not built from this "
                  f"codetwine-loom: {_STALE_TEXT}.")
        print(f"Building it takes a few minutes, fetches packages over the network and "
              f"takes several GB of disk:\n  {command_text}")
        if _ask_build(is_absent, is_enter_build):
            _build_image(engine_path, image_name, toolchain, mark, repo_dir, command_text)
            return state
    if is_absent:
        raise RuntimeError(f"{absent_text}\nBuild it ({what_text}) with:\n"
                           f"  {command_text}\n"
                           f"or set CONTAINER_IMAGE to an image the engine has")
    logger.warning("the image %s (TOOLCHAIN=%s) was not built from this codetwine-loom "
                   "(%s), and the containers run what it holds. Build it again with: %s",
                   image_name, label_toolchain or "not named", _STALE_TEXT, command_text)
    return state



def _ask_build(is_absent: bool, is_enter_build: bool = True) -> bool:
    """Ask whether to build the image.

    Args:
        is_absent: Whether the image is not there, rather than stale.
        is_enter_build: Whether Enter is taken as y rather than as n.

    Returns:
        True on y, and on Enter with is_enter_build. False on n, on Enter without it, and
        when the input ends.
    """
    question = ("Build it now?" if is_absent else "Build it again now?") \
        + (" [Y/n]: " if is_enter_build else " [y/N]: ")
    try:
        answer = _ask_yes_no(question)
        return answer is True or (answer is None and is_enter_build)
    except EOFError:
        print()
        return False


def _build_image(engine_path: str, image_name: str, toolchain: str, mark: str,
                 repo_dir: str, command_text: str) -> None:
    """Build the image (sandbox.build_image), saying so before and after.

    Args:
        engine_path: The engine's executable.
        image_name: The name the image is given.
        toolchain: The TOOLCHAIN it is built with.
        mark: image.source_mark() of repo_dir.
        repo_dir: codetwine-loom's repository.
        command_text: image.build_command_text() of the same build.

    Raises:
        RuntimeError: When the file of certificate authorities cannot be read, or the
            build cannot be started, does not end in IMAGE_BUILD_TIMEOUT_SEC or ends
            with a non-zero status.
    """
    engine_name = os.path.basename(engine_path)
    try:
        argv = image_mod.build_argv(engine_path, image_name, toolchain, mark, repo_dir)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"the certificate authorities of {config.CA_FILE_ENV} could not "
                           f"be passed ({e}), so the image {image_name} was not built"
                           ) from None
    print(f"building the image {image_name} with {engine_name}")
    try:
        status = sandbox_mod.build_image(argv)
    except OSError as e:
        raise RuntimeError(f"{engine_name} build could not be started ({e}). Build the "
                           f"image by hand with:\n  {command_text}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{engine_name} build did not end within "
                           f"{config.IMAGE_BUILD_TIMEOUT_SEC} seconds, and the image "
                           f"{image_name} was not built. Build it by hand with:\n"
                           f"  {command_text}") from None
    if status:
        raise RuntimeError(f"{engine_name} build ended with exit status {status}, and the "
                           f"image {image_name} was not built. Build it by hand with:\n"
                           f"  {command_text}")
    print(f"built the image {image_name}")


def _needs_credential(args: argparse.Namespace) -> bool:
    """Say whether this process checks the credentials and signs in for args.

    Args:
        args: The parsed arguments.

    Returns:
        True for a run that will call a model, or hold the credentials for one that
        does: broker whatever LLM_BROKER says, and any other subcommand that calls a
        model while LLM_BROKER is empty or under --doc-create.
    """
    return calls_a_model(args) and (args.command == "broker" or not config.LLM_BROKER
                                    or getattr(args, "with_doc", False))


def _check_credential(args: argparse.Namespace,
                      argv: list[str] | None) -> argparse.Namespace:
    """Stop a run that will call a model when the provider has nothing to call with.

    It is stopped as well when a proxy variable holds a URL the HTTP clients of this
    tool cannot read (config.proxy_error_text). The runs checked are those
    _needs_credential() names.

    Args:
        args: The parsed arguments.
        argv: The argument list, read again when .env was written here.

    Returns:
        args, or the arguments read again after .env was written.

    Raises:
        SystemExit: When a credential is still missing, or naming the proxy variable.
    """
    # A run that will call a model stops here, before the analysis, when the provider
    # has nothing to call with. At a terminal, with no .env read, it may first ask
    # questions and write .env in the current directory (_ask_env_file); when it does,
    # the check is made again under what that file holds
    if _needs_credential(args):
        lack_text = llm.missing_credential_text()
        path = _ask_env_file(bool(lack_text))
        if path:
            # The reload reads the file just written, and codetwine's settings are read
            # again (knowledge.read_codetwine_setting_again); the arguments are read
            # again, the ones whose default is a setting (--project) following it too,
            # and the check made before the file was there is made again under what it
            # holds
            config.accept_env_file(path)
            try:
                importlib.reload(config)
                knowledge_mod.read_codetwine_setting_again()
            except ValueError as e:
                raise SystemExit(f"{path}: {e}") from None
            args = _parse_args(argv)
            _refuse_no_engine_or_image(args)
            lack_text = llm.missing_credential_text()
        if lack_text:
            skip_text = (f" ({config.ENV_FILE_SKIP_PATH} {config.env_file_skip_text()})"
                         if config.ENV_FILE_SKIP_PATH else "")
            raise SystemExit(f"{lack_text}\n.env read: "
                             f"{config.ENV_FILE_PATH or 'none'}{skip_text}")
        proxy_text = config.proxy_error_text()
        if proxy_text:
            raise SystemExit(f"{args.command}: {proxy_text}")
    return args


def _sign_in(args: argparse.Namespace) -> None:
    """Have litellm sign in to the providers it signs in to (llm.sign_in_text).

    The runs signed in for are those _needs_credential() names.

    Args:
        args: The parsed arguments.

    Raises:
        SystemExit: Naming the models litellm did not sign in for.
    """
    if not _needs_credential(args):
        return
    sign_in_text = llm.sign_in_text()
    if sign_in_text:
        raise SystemExit(sign_in_text)


def _warn_no_partner_setting(args: argparse.Namespace) -> None:
    """Log one warning naming the settings set while their partner setting is empty.

    The settings are config.no_partner_setting_list(). It is logged by a subcommand that
    calls a model. broker and the run inside a container log none.

    Args:
        args: The parsed arguments.
    """
    if not calls_a_model(args) or args.command == "broker" or sandbox_mod.in_container():
        return
    no_partner_list = config.no_partner_setting_list()
    if no_partner_list:
        logger.warning("set but not used: %s", ", ".join(no_partner_list))


def _serve_broker(args: argparse.Namespace, handler: logging.Handler) -> None:
    """Answer the model calls of a container until the broker is stopped.

    A broker given --run ends when the process of that run has ended (_end_with_run).
    A broker stopped before it listens ends as one stopped while it listens does.
    When the broker ends, the container of a filling it has going is taken away
    (sandbox.stop_running_container), with what ended the broker as the reason the
    lines about it give ("the run was stopped" for Ctrl-C, "the broker was asked to
    end (<signal>)" for SIGTERM and SIGHUP), and the filling is waited for, for at most
    CONTAINER_COMMAND_TIMEOUT_SEC, while it takes its clone away. A stop signal that
    reaches it meanwhile is held until then (sandbox.stop_signal_hold).

    Args:
        args: The parsed arguments of broker.
        handler: The stderr handler, whose lines take --label at their head.

    Raises:
        SystemExit: When the broker cannot listen where it was told.
    """
    # Without a repository the broker only makes model calls. With one it also starts
    # the container that fills that repository's cache when a run asks for it
    def refill_under_label(spend: Callable[[], object]) -> dict:
        """Fill the cache again, under the label this broker was given.

        Args:
            spend: What sandbox.warm() calls before it fills the cache.

        Returns:
            The return value of sandbox.warm().
        """
        with fill_lock, progress.label(args.label):
            try:
                return sandbox_mod.warm(args.project, args.output_dir,
                                        live_dep_file=True, run_id=args.run, once=True,
                                        before_fill=spend, base=args.base,
                                        agent=args.agent,
                                        fill_ext_list=sandbox_mod.fill_ext_list(
                                            args.fill_ext))
            except KeyboardInterrupt as e:
                # The broker was stopped while the filling's container was going
                raise RuntimeError(str(e) or _INTERRUPT_TEXT) from None

    # Held for the length of a filling, which a broker that is ending waits for
    fill_lock = threading.Lock()
    refill = refill_under_label if args.project else None
    handler.setFormatter(progress.LabelFormatter(progress.TERM_LOG_FORMAT,
                                                 head=args.label))
    # What the broker adds to a call is taken out of every line it writes to stderr
    handler.addFilter(broker_mod.OwnValueFilter())
    # A broker given a repository keeps every line it logs in a file, without the values
    # it adds to a call: with what this machine writes for the agent it serves
    # (sandbox.host_dir), or under the output directory when it serves none
    if args.project:
        output_dir = os.path.abspath(args.output_dir)
        config.OUTPUT_DIR = sandbox_mod.host_dir(output_dir, args.agent) if args.agent \
            else output_dir
        source.project_name = knowledge_mod.project_name_of(args.project)
        turnlog.open_log(log_filter=broker_mod.OwnValueFilter())
    try:
        # A broker serving a run ends with that run's process, however that one ended
        if args.run:
            _end_with_run(args.run)
        broker_mod.serve(args.endpoint, refill_cache=refill,
                         token_out=os.path.abspath(args.token_out) if args.token_out
                         else "", token_left=args.token_left)
    except KeyboardInterrupt as e:
        logger.info("the broker was stopped before it listened (%s)",
                    str(e) or _INTERRUPT_TEXT)
    except (OSError, ValueError) as e:
        raise SystemExit(f"broker: {e}") from None
    finally:
        # A filling this broker started is not left going once the broker has ended:
        # its container is taken away, and the filling is given the time to take its
        # clone away. A stop signal that comes meanwhile is held until then. What
        # ended the broker is said: Ctrl-C stopped the run, any other signal asked
        # the broker to end
        with sandbox_mod.stop_signal_hold():
            number = stop_number()
            sandbox_mod.stop_running_container(
                _INTERRUPT_TEXT if number == signal.SIGINT
                else f"the broker was asked to end ({number.name})")
            if fill_lock.acquire(timeout=config.CONTAINER_COMMAND_TIMEOUT_SEC or -1):
                fill_lock.release()


def _start_sandbox(args: argparse.Namespace) -> NoReturn:
    """Make the change in a container, print what it produced and exit with its status.

    Args:
        args: The parsed arguments of sandbox.

    Raises:
        SystemExit: Always, with the run's exit status or what stopped it.
        KeyboardInterrupt: When the command was stopped, once its record is written.
        Exception: Whatever else the command raised, once its record is written.
    """
    config.EDIT_GUIDE = args.guide
    record_id = sandbox_mod.new_run_id()
    project_dir = os.path.abspath(args.project)
    result_dict = None
    try:
        result_dict = sandbox_mod.run(
            args.project, args.output_dir, args.request, engine=args.engine,
            image=args.image, build_doc=args.with_doc,
            dry_run=args.dry_run, guide=args.guide,
            exclude_list=exclude_name_list(args),
            with_overview=True, land=True, check=args.check)
        result_dict["swept"] = sandbox_mod.remove_old_run_in(
            args.project, args.output_dir, result_dict.get("engine", ""))
        exit_status = result_dict["exit_status"] or sandbox_mod.land_status(result_dict)
        print_sandbox(project_dir, result_dict)
        _print_old_run_list(result_dict)
        exit_status = max(exit_status, _token_stop_status(result_dict))
    except KeyboardInterrupt as e:
        _print_run_end(project_dir, _write_stop_record(
            args, record_id, str(e) or _INTERRUPT_TEXT, _signal_stop_status(), result_dict))
        raise
    except RuntimeError as e:
        # A RuntimeError ends the run with its message alone
        _print_run_end(project_dir, _write_stop_record(args, record_id, e,
                                                       result_dict=result_dict))
        raise SystemExit(f"{args.command}: {e}") from None
    except Exception as e:
        _print_run_end(project_dir, _write_stop_record(args, record_id, _error_text(e),
                                                       result_dict=result_dict))
        raise
    record_dict = write_run_record(args, record_id, result_dict, exit_status)
    _print_record_line(record_dict)
    _print_run_end(project_dir, {**result_dict, "command": args.command,
                                 "request": args.request,
                                 "resume": record_dict["resume"]})
    raise SystemExit(exit_status)


def _print_run_end(project_dir: str, end_dict: dict) -> None:
    """Print the closing lines of a sandbox or integrate command.

    They say how to take out what it put into the working tree, and what to carry on
    from.

    Args:
        project_dir: The target repository, as an absolute path.
        end_dict: What the command made, with "command" the subcommand, "request"
            the request it was given and "resume" the branches its record names:
            _write_stop_record()'s return value for a command that was stopped.
    """
    print_take_out_line(project_dir, end_dict)
    print_resume_line(project_dir, end_dict)


def _start_integrate(args: argparse.Namespace) -> NoReturn:
    """Combine the branches in a container, print the union and exit with its status.

    Args:
        args: The parsed arguments of integrate without --here.

    Raises:
        SystemExit: Always, with the union's exit status or what stopped it.
        KeyboardInterrupt: When the command was stopped, once its record is written.
        Exception: Whatever else the command raised, once its record is written.
    """
    record_id = sandbox_mod.new_run_id()
    project_dir = os.path.abspath(args.project)
    result_dict = None
    try:
        result_dict = sandbox_mod.integrate_into_tree(
            args.project, args.output_dir, args.branch_list, engine=args.engine,
            image=args.image, build_doc=args.with_doc,
            request=args.request, exclude_list=exclude_name_list(args),
            resume=args.resume)
        result_dict["swept"] = sandbox_mod.remove_old_run_in(
            args.project, args.output_dir, result_dict.get("engine", ""),
            protect_list=[*args.branch_list, args.resume])
        exit_status = sandbox_mod.union_status(result_dict) \
            or sandbox_mod.land_status(result_dict)
        print_sandbox(project_dir, result_dict)
        _print_old_run_list(result_dict)
        exit_status = max(exit_status, _token_stop_status(result_dict))
    except KeyboardInterrupt as e:
        _print_run_end(project_dir, _write_stop_record(
            args, record_id, str(e) or _INTERRUPT_TEXT, _signal_stop_status(), result_dict))
        raise
    except RuntimeError as e:
        _print_run_end(project_dir, _write_stop_record(args, record_id, e,
                                                       result_dict=result_dict))
        raise SystemExit(f"{args.command}: {e}") from None
    except Exception as e:
        _print_run_end(project_dir, _write_stop_record(args, record_id, _error_text(e),
                                                       result_dict=result_dict))
        raise
    record_dict = write_run_record(args, record_id, result_dict, exit_status)
    _print_record_line(record_dict)
    _print_run_end(project_dir, {**result_dict, "command": args.command,
                                 "request": args.request,
                                 "resume": record_dict["resume"]})
    raise SystemExit(exit_status)


def _start_dispatch(args: argparse.Namespace) -> NoReturn:
    """Split the request in a container, run the agents, print and exit with the status.

    Args:
        args: The parsed arguments of run without --here.

    Raises:
        SystemExit: Always, with the run's exit status or what stopped it. When what
            stopped it is a sandbox.ContainerEndError, the status is that container's
            exit status, and what stopped it is printed to stderr.
        KeyboardInterrupt: When the command was stopped, once its record is written.
        Exception: Whatever else the command raised, once its record is written.
    """
    record_id = sandbox_mod.new_run_id()
    project_dir = os.path.abspath(args.project)
    result_dict = None
    try:
        result_dict = sandbox_mod.dispatch(
            args.project, args.output_dir, args.request, engine=args.engine,
            image=args.image, build_doc=args.with_doc,
            resume_list=args.resume_list, exclude_list=exclude_name_list(args))
        exit_status = result_dict["exit_status"] \
            or sandbox_mod.land_status(result_dict.get("integration"))
        print_dispatch(project_dir, result_dict)
        report_text = print_report(os.path.abspath(args.project), args.request,
                                   result_dict)
        exit_status = max(exit_status, _token_stop_status(result_dict))
    except KeyboardInterrupt as e:
        _print_dispatch_end(project_dir, _write_stop_record(
            args, record_id, str(e) or _INTERRUPT_TEXT, _signal_stop_status(), result_dict))
        raise
    except RuntimeError as e:
        stop_status = e.exit_status if isinstance(e, sandbox_mod.ContainerEndError) else 1
        _print_dispatch_end(project_dir, _write_stop_record(args, record_id, e, stop_status,
                                                            result_dict=result_dict))
        if stop_status == 1:
            raise SystemExit(f"{args.command}: {e}") from None
        print(f"{args.command}: {e}", file=sys.stderr)
        raise SystemExit(stop_status) from None
    except Exception as e:
        _print_dispatch_end(project_dir, _write_stop_record(
            args, record_id, _error_text(e), result_dict=result_dict))
        raise
    record_dict = write_run_record(args, record_id, result_dict, exit_status, report_text)
    _print_record_line(record_dict)
    _print_dispatch_end(project_dir, {**result_dict, "command": args.command,
                                      "request": args.request,
                                      "resume": record_dict["resume"]})
    raise SystemExit(exit_status)


def _print_dispatch_end(project_dir: str, result_dict: dict) -> None:
    """Print the closing lines of a dispatching run.

    Args:
        project_dir: The target repository, as an absolute path.
        result_dict: The return value of sandbox.dispatch() with "command" the
            subcommand, "request" the request it was given and "resume" the branches the
            run's record names, or the return value of _write_stop_record() for a run
            that was stopped.
    """
    _print_land_line(result_dict.get("integration"))
    print_take_out_line(project_dir, result_dict)
    print_resume_line(project_dir, result_dict)
    _print_unclear_line(result_dict)


def _start_warm(args: argparse.Namespace) -> NoReturn:
    """Fill the package cache in a container and exit with its status.

    Args:
        args: The parsed arguments of warm.

    Raises:
        SystemExit: Always, with the filling's exit status or what stopped it.
    """
    try:
        result_dict = sandbox_mod.warm(args.project, args.output_dir,
                                       exclude_list=exclude_name_list(args))
    except RuntimeError as e:
        raise SystemExit(f"warm: {e}") from None
    print(f"cache: {result_dict['cache']}")
    print(f"base: {result_dict['base']}")
    raise SystemExit(result_dict["exit_status"])


def _finish_integrate(args: argparse.Namespace, prepare_dict: dict, path: str,
                      project_name: str, exclude_list: list[str]) -> NoReturn:
    """Run the merging stage over the union integrate --here prepared, and exit.

    Args:
        args: The parsed arguments of integrate --here.
        prepare_dict: The return value of integrate.prepare().
        path: The knowledge file of the union. A relative path, a relative
            --project and a relative --output-dir are resolved from here, before the
            run moves to a directory of its own.
        project_name: The repository's directory name.
        exclude_list: The names kept out of the analysis.

    Raises:
        SystemExit: Always: 0 when the merge settled and --outcome-out was written,
            1 otherwise.
    """
    path, output_dir = os.path.abspath(path), os.path.abspath(args.output_dir)
    project_dir = os.path.abspath(args.project)
    with turnlog.open_run():
        result_dict = integrate_mod.run_integration(prepare_dict, path, project_name,
                                                    output_dir,
                                                    exclude_list=exclude_list,
                                                    request=args.request)
    gap_dict = recheck_gap(args.gap_in) if args.gap_in and result_dict["settled"] \
        else {}
    # --outcome-out is written before anything is printed
    is_out = not args.outcome_out or write_outcome(
        args.outcome_out, result_dict, gap_dict,
        integrate_mod.branch_commit_list(prepare_dict["merged"]),
        integrate_mod.separator_record_dict(prepare_dict["merged"]),
        integrate_mod.no_merge_record_list(prepare_dict["merged"]))
    _HERE_UNION_LIST.clear()
    print_integration(result_dict)
    if _has_union_note(args):
        print_union_in_place(project_dir, prepare_dict)
    raise SystemExit(0 if result_dict["settled"] and is_out else 1)


def _take_change_language(base: str, path: str, project_name: str) -> str:
    """Make the language of the change the primary one, as testcreate.change_lang_ext says.

    Args:
        base: The commit the change HEAD holds was written against.
        path: The knowledge file of HEAD.
        project_name: The repository's directory name.

    Returns:
        The language made the primary one, or "" when the primary one is kept.
    """
    ext = test_create_mod.change_lang_ext(base, path, project_name)
    if not ext:
        return ""
    logger.info("the stage takes %s as the primary language in place of %s",
                LANG_NAME_DICT[ext], LANG_NAME_DICT[config.PRIMARY_EXT])
    hand_list = config.hand_env_list(config.RUN_SETTING_TUPLE)
    if hand_list:
        logger.warning("%s set by hand stays as it is with %s as the primary language",
                       " and ".join(hand_list), LANG_NAME_DICT[ext])
    scope_mod.set_primary(ext)
    return ext


def _host_language() -> str:
    """Return the language the host handed a container to take as the primary one.

    Returns:
        sandbox.primary_ext_of_env() in a container, "" outside one.
    """
    return sandbox_mod.primary_ext_of_env() if sandbox_mod.in_container() else ""


def _take_host_language(command: str) -> None:
    """Make the language the host handed a container the primary one, for a stage.

    Args:
        command: The subcommand. Only test-create, refactor, test-refactor and integrate
            take the language (_host_language).
    """
    ext = _host_language() if command in _HOST_LANGUAGE_COMMAND_TUPLE else ""
    if ext:
        scope_mod.set_primary(ext)


def _finish_test_create(args: argparse.Namespace, path: str, project_name: str,
                        exclude_list: list[str]) -> NoReturn:
    """List what to write tests for (--target-only), or write them, and exit.

    The language of the change is made the primary one first (_take_change_language),
    except with --target-in in a container, which takes the one it is handed, and
    --target-only writes it as "lang" beside what it lists.

    Args:
        args: The parsed arguments of test-create.
        path: The knowledge file. A relative path, a relative --output-dir and the
            file --target-in names are read from here, before the run moves to a
            directory of its own.
        project_name: The repository's directory name.
        exclude_list: The names kept out of the analysis.

    Raises:
        SystemExit: Always: 0 for --target-only and for a settled run, each with
            --outcome-out written, 1 otherwise.
    """
    lang = "" if args.target_in and sandbox_mod.in_container() \
        else _take_change_language(args.base, path, project_name)
    if args.target_only:
        target_dict = {**test_create_mod.list_target(args.base, path, project_name),
                       "lang": lang}
        is_out = not args.outcome_out or _write_out(args.outcome_out, target_dict,
                                                    "what there is to test")
        print(f"definitions to write tests for: "
              f"{sum(len(def_list) for def_list in target_dict['targets'].values())} in "
              f"{len(target_dict['targets'])} file(s)"
              + (f" (skipped: {target_dict['skipped']})" if target_dict["skipped"] else ""))
        raise SystemExit(0 if is_out else 1)
    path, output_dir = os.path.abspath(path), os.path.abspath(args.output_dir)
    task_dict = read_test_task(args.target_in) if args.target_in else None
    with turnlog.open_run():
        result_dict = test_create_mod.run_test_create(
            args.base, path, project_name, output_dir,
            exclude_list=exclude_list, request=args.request, task_dict=task_dict)
    is_out = not args.outcome_out or write_create_outcome(args.outcome_out, result_dict)
    print_test_create_stage(result_dict)
    raise SystemExit(0 if result_dict["settled"] and is_out else 1)


def _finish_refactor(args: argparse.Namespace, path: str, project_name: str,
                     exclude_list: list[str]) -> NoReturn:
    """Run one of the stages that refactor, and exit.

    The language of the change is made the primary one first (_take_change_language),
    unless a container was handed one (_take_host_language), and when it is, only the
    files its test command runs are refactored (runnable_only).

    Args:
        args: The parsed arguments of refactor or test-refactor.
        path: The knowledge file. A relative path, and a relative --output-dir, are
            resolved from here, before the run moves to a directory of its own.
        project_name: The repository's directory name.
        exclude_list: The names kept out of the analysis.

    Raises:
        SystemExit: Always: 0 when the stage settled and --outcome-out was written,
            1 otherwise.
    """
    lang = _host_language() or _take_change_language(args.base, path, project_name)
    path, output_dir = os.path.abspath(path), os.path.abspath(args.output_dir)
    with turnlog.open_run():
        result_dict = refactor_mod.run_refactor(
            args.command, args.base, path, project_name, output_dir,
            exclude_list=exclude_list, request=args.request, runnable_only=bool(lang))
    is_out = not args.outcome_out or write_refactor_outcome(
        args.outcome_out, args.command, result_dict)
    print_refactor_stage(result_dict)
    raise SystemExit(0 if result_dict["settled"] and is_out else 1)


def _finish_split(args: argparse.Namespace) -> NoReturn:
    """Split the request inside the container run --here is, and exit.

    Args:
        args: The parsed arguments of run --here.

    Raises:
        SystemExit: Always: 0 when the split was handed over and --tasks-out was
            written, 1 otherwise, and "run: <why>" when the splitting stage refused the
            run.
    """
    # Nothing is written but the turn log and the split itself: no automatic check, no
    # plan and no edit-history entry
    with turnlog.open_run():
        try:
            result_dict = dispatch_mod.run_dispatch(args.request,
                                                    resume_list=args.resume_list)
        except RuntimeError as e:
            raise SystemExit(f"run: {e}") from None
    # --tasks-out is written before anything is printed
    is_out = not args.task_out or write_split(args.task_out, result_dict)
    print_scope(result_dict["scope"])
    print_split(result_dict)
    raise SystemExit(0 if result_dict["ok"] and is_out else 1)


def _finish_answer(args: argparse.Namespace) -> NoReturn:
    """Answer the question, or combine the agents' answers with --answered, and exit.

    Args:
        args: The parsed arguments of answer.

    Raises:
        SystemExit: Always: 0 when an answer was handed over and --answer-out was
            written, 1 otherwise.
    """
    # One answering agent, or the combining run when --answered names the agents'
    # answers. Read-only either way, and both read the working tree as it is. The file
    # --answered names is read before the run moves to a directory of its own
    answer_list = read_answer_list(args.answer_in) if args.answer_in else None
    with turnlog.open_run():
        if answer_list is not None:
            result_dict = answer_mod.run_combine(args.request, answer_list)
        else:
            result_dict = answer_mod.run_answer(args.request)
    is_out = not args.answer_out or write_answer(args.answer_out, result_dict)
    print_scope(result_dict["scope"])
    print_answer(result_dict)
    raise SystemExit(0 if result_dict["ok"] and is_out else 1)


# What edit prints when --resume was passed and nothing is left to carry on from
_NO_OPEN_ATTEMPT_TEXT = ("--resume: no unfinished change is recorded to carry on from, "
                         "so this run starts as a new one")


def _finish_edit(args: argparse.Namespace, knowledge_path: str,
                 exclude_list: list[str]) -> NoReturn:
    """Make the change here, print what it did, and exit.

    With --resume and no unfinished change recorded, one line says so and the run
    starts as a new one. Under --dry-run an unfinished change that is not carried on
    from stays open in the history, and this run is not handed it.

    Args:
        args: The parsed arguments of edit.
        knowledge_path: The analysis of the working tree, which the change is checked
            against (agent.check_start) at each COMPLETE() and once the run has ended,
            unless --no-check or --dry-run was passed or the analysis cannot be read. A
            relative path is resolved from here, before the run moves to a directory of
            its own.
        exclude_list: The names kept out of the analysis.

    Raises:
        SystemExit: Always: 1 when history.judge() found the run failed, else 0.
    """
    if getattr(args, "resume", None) and not config.EDIT_HISTORY:
        raise SystemExit("edit: --resume carries on from what EDIT_HISTORY records, and "
                         "EDIT_HISTORY is false. Set it to true, or leave --resume out")
    # Asked before create_agent(): saying no closes the entry, before run_edit() reads
    # the history to build the prompt
    open_entry = history.open_attempt()
    if open_entry is None and getattr(args, "resume", None):
        print(_NO_OPEN_ATTEMPT_TEXT)
    if open_entry and not _want_resume(open_entry, getattr(args, "resume", None)):
        history.resolve("closed", "the user chose to start fresh")
        # Under a dry run the entry stays open, and this run alone is not handed it
        if config.EDIT_DRY_RUN:
            config.EDIT_HISTORY = False
    # The turn log is opened before the stages start, and the run works from a directory
    # of its own, not from the repository (turnlog.open_run)
    knowledge_path = os.path.abspath(knowledge_path)
    with turnlog.open_run():
        start_dict = check_start(knowledge_path, exclude_list) if args.check else None
        result_dict = run_edit(create_agent(), args.request, start_dict)
    print_scope(result_dict["scope"])
    print_result(result_dict, edit.git_diff_stat())
    # Exit 1 only when history.judge() found the run failed. Tests that ended red are
    # exit 0, as is a run whose changes were not verified. Files may have been changed
    # either way
    raise SystemExit(0 if result_dict["verdict"] != history.VERDICT_FAIL else 1)


def main(argv: list[str] | None = None) -> None:
    """The command-line entry point.

    SIGTERM and SIGHUP stop the command the way Ctrl-C does while it runs
    (stop_on_signal). The handlers set for that, the start time put into the
    environment (progress.mark_run_start) and where the containers' stdout goes
    (progress.keep_child_stdout) are taken back when the command returns or is
    stopped.

    What an earlier command of this process left is taken away first: what it put into a
    working tree (sandbox.reset_land_list), the commit its working tree made
    (sandbox.forget_start_tree), a stop it was given (sandbox.forget_stop and the
    signals stop_number() reads), its token counts (tokencount.reset) and the union
    integrate --here put into a repository (_HERE_UNION_LIST). The working directory is
    taken off sys.path (keep_cwd_off_path). A command stopped while integrate --here
    holds a union that is not yet what it leaves puts that repository back first
    (_put_union_back). One that ends there in any other way prints what the repository
    holds and how to put it back first, where it prints that at all
    (_print_union_remain).

    Args:
        argv: The argument list. None uses sys.argv.

    Raises:
        SystemExit: What _run_command() raises.
        KeyboardInterrupt: When the command was stopped (stop_number() names the
            signal).
    """
    keep_cwd_off_path()
    is_start_set = progress.mark_run_start()
    _STOP_NUMBER_LIST.clear()
    signal_list = stop_on_signal()
    sandbox_mod.reset_land_list()
    sandbox_mod.forget_start_tree()
    sandbox_mod.forget_stop()
    tokencount.reset()
    _HERE_UNION_LIST.clear()
    try:
        _run_command(argv)
    except KeyboardInterrupt:
        _put_union_back()
        raise
    except (SystemExit, Exception):
        _print_union_remain()
        raise
    finally:
        put_signal_back(signal_list)
        progress.keep_child_stdout(False)
        if is_start_set:
            progress.forget_run_start()


def _run_command(argv: list[str] | None) -> None:
    """Run one command line from its arguments to what it prints.

    Steps:
    0. setup asks for the settings of .env, writes it and returns (_run_setup). broker
       puts the model table entries it was handed into litellm's table
       (broker.load_model_table) before the checks. Every subcommand but setup goes
       through the checks of _check_before_start(), where the container engine and image
       of a subcommand that starts containers are looked for before the credentials are
       checked (the image only once they are there). broker, history and format
       return, and sandbox, warm, and run and integrate without --here start their
       containers and exit (_start_container_command), before any of the start-up below
    1. Read the arguments and prepare the knowledge file (step 0). Every subcommand
       makes it when it is missing, without the design documents and without asking.
       integrate analyses again when the analysis does not describe what is checked
       out, then merges and analyses the union (1b). Every other subcommand analyses
       again when the analysis does not describe what is checked out, and build does
       so with --doc-create either way; each stops first while a sandboxed run is
       going over the same output directory (1c). Only build --doc-create writes the
       design documents. A subcommand without --exclude keeps out what the analysis on
       disk kept out
    2. Build the index and write it out (steps 1-5): for build, overview and
       integrate, for a run that just made the analysis, and when there is no index
       built from the analysis on disk
    3. Read the index and hand it to the module variables
    4. Print whatever the subcommand prints

    Args:
        argv: The argument list. None uses sys.argv.

    Raises:
        SystemExit: With the message of what stopped the run, or with the exit status
            a subcommand ends with: every one but setup, build, search, history, format
            and broker ends with one, run-tests only when a test command stopped short
            of its tests, and overview only when the overview could not be made.
    """
    # == Step 0: checks, and what returns before start-up ====
    progress.set_up_output()
    handler = _set_up_log()
    args = _parse_args(argv)
    # setup reads no target repository and writes .env
    if args.command == SETUP_COMMAND:
        _run_setup(args)
        return

    # A broker takes the model table entries it was handed before anything here asks
    # litellm about a model
    if args.command == "broker":
        broker_mod.load_model_table(os.environ.get(broker_mod.MODEL_TABLE_ENV, ""))

    args = _check_before_start(args, argv)

    # broker answers model calls and returns before the start-up below
    if args.command == "broker":
        _serve_broker(args, handler)
        return

    # history reads one file beside the index and returns before the start-up below: it
    # works on a project that has never been built
    if args.command == "history":
        config.OUTPUT_DIR = args.output_dir
        source.project_name = knowledge_mod.project_name_of(args.project)
        print_run_record_list(os.path.abspath(args.project))
        print_history()
        return

    # format reads the working tree alone, and returns before the start-up below too
    if args.command == "format":
        config.PROJECT_DIR = os.path.abspath(args.project)
        print_format(edit.run_format())
        return

    _start_container_command(args)

    # == Step 1: prepare the knowledge file ==================
    # overview does what build does, without --doc-create, before it makes the overview
    is_build = args.command in ("build", "overview")
    with_doc = is_build and getattr(args, "with_doc", False)
    exclude_list = exclude_name_list(args)
    try:
        path, project_dir, project_name, is_reuse = ensure_knowledge(
            args.project, args.output_dir,
            with_doc=with_doc,
            exclude_list=exclude_list)
    except (ValueError, RuntimeError) as e:
        # The error ends the run with its message alone
        raise SystemExit(str(e)) from None
    config.PROJECT_DIR = project_dir
    # Absolute: a run of the stages works from a directory of its own (turnlog.open_run)
    config.OUTPUT_DIR = os.path.abspath(args.output_dir)
    # A subcommand without --dry-run writes, and one without --guide leaves the setting
    # as the environment set it
    config.EDIT_DRY_RUN = getattr(args, "dry_run", False)
    config.EDIT_GUIDE = getattr(args, "guide", config.EDIT_GUIDE)
    # == Step 1b: for integrate, merge and analyse the union =
    is_integrate = args.command == "integrate"
    prepare_dict = None
    if is_integrate:
        prepare_dict, path, is_reuse = _prepare_union(args, project_dir, path,
                                                      project_name, exclude_list, is_reuse)

    knowledge = Knowledge.open(path)
    source_stamp = knowledge.stamp()
    index_path = store.index_db_path(args.output_dir, project_name)
    # A subcommand without --exclude reads the analysis as it was made, keeping out what
    # the record says it keeps out
    if args.command not in _EXCLUDE_DECIDE_TUPLE:
        exclude_list = exclude_of_analysis(path, source_stamp, exclude_list)
    # run reaches here with --here alone, which keeps out what the analysis keeps out
    if args.command == "run":
        try:
            exclude_list = sandbox_mod.exclude_list_of_record(
                project_dir, os.path.abspath(args.output_dir), project_name, exclude_list)
        except RuntimeError as e:
            raise SystemExit(f"run: {e}") from None

    # == Step 1c: make the analysis describe what is on disk =
    # integrate did it in 1b. For every other subcommand an analysis of another commit,
    # of a tree that has changed since, made with another exclude list or made by another
    # codetwine than the one installed is made again here; build --doc-create runs codetwine either way. The documents already on disk
    # are kept, and only build writes new ones. build makes again as well an analysis
    # whose usages do not name the definition they lead to
    if not is_integrate and is_reuse and (
            with_doc or (is_build and not knowledge.names_target())
            or not analysis_match_tree(project_dir, index_path, source_stamp,
                                       exclude_list)):
        refuse_while_run_live(args.command, args.project, args.output_dir)
        knowledge.close()
        try:
            path, _, _, is_reuse = ensure_knowledge(
                args.project, args.output_dir, rebuild=True,
                with_doc=with_doc, exclude_list=exclude_list)
        except (ValueError, RuntimeError) as e:
            raise SystemExit(str(e)) from None
        knowledge = Knowledge.open(path)
        source_stamp = knowledge.stamp()
    if not knowledge.names_target():
        knowledge.close()
        raise SystemExit(
            f"{args.command}: the analysis at {path} is out of date: its usages do not "
            f"name the definition they lead to. Run `codetwine-loom build` over the "
            f"repository to make the analysis again")

    # == Step 2: build the index and write it out ============
    # The index is built before step 3 reads it: for build, overview and integrate, and
    # for a run with no index built from the analysis on disk (one that just made the
    # analysis, or one over an analysis made without this tool). Building it calls no
    # LLM
    index_dict = None
    need_index_build = is_build or is_integrate
    if not need_index_build and (not is_reuse
                                 or not store.is_from_analysis(index_path, source_stamp)):
        logger.info("no index built from this analysis at %s, so it is built now "
                    "(no LLM is called)", index_path)
        need_index_build = True
    if need_index_build:
        # The index records the commit the analysis describes, as the record beside the
        # analysis says
        index_dict = build.build_and_save(knowledge, project_name, index_path)

    # == Step 3: read the index into the module variables ====
    # run --here and integrate --here may ask which dialect the static check reads COBOL
    # in (knowledge.ask_cobol_dialect)
    if args.command in ("run", "integrate"):
        knowledge_mod.ask_cobol_dialect(index_path)
    scope_mod.load_index(index_path, project_name, source_stamp=source_stamp)
    _take_host_language(args.command)
    # In a container, what the tools of the index's languages write in the clone is left
    # out of its git
    if sandbox_mod.in_container():
        edit.exclude_build_output()
    source.set_project_data(knowledge)
    source.project_name = project_name
    # set_project_data holds the callee usages, and the analysis is not read again
    knowledge.close()
    overview_dict = _make_overview(args, project_dir, project_name)

    # == Step 4: print whatever the subcommand prints ========
    if is_build:
        print_build(index_dict, index_path, path, is_reuse)
        if overview_dict:
            _print_overview(overview_dict)
            # An overview that was not current and could not be made again ends the
            # process with status 1
            if overview_dict["status"] == overview.FAIL_STATUS:
                raise SystemExit(1)
    elif args.command == "search":
        print_scope(search_scope(args.request))  # no llm is passed
    elif args.command == "run-tests":
        _finish_run_tests()
    elif is_integrate:
        _finish_integrate(args, prepare_dict, path, project_name, exclude_list)
    elif args.command == "test-create":
        _finish_test_create(args, path, project_name, exclude_list)
    elif args.command in refactor_mod.STAGE_TUPLE:
        _finish_refactor(args, path, project_name, exclude_list)
    elif args.command == "run":
        _finish_split(args)
    elif args.command == "answer":
        _finish_answer(args)
    else:  # "edit"
        _finish_edit(args, path, exclude_list)


def _check_before_start(args: argparse.Namespace,
                        argv: list[str] | None) -> argparse.Namespace:
    """Run the checks of every subcommand but setup, in order, before anything starts.

    The .env file left unread and the request are checked, --preset and the stage flags
    are written into the environment, and then the test commands, the container engine
    and image, the credentials (which may write .env and read the arguments again), the
    repository, the output directory, the files to write and --base are checked. Last,
    litellm signs in to the providers it signs in to (_sign_in).

    Args:
        args: The parsed arguments.
        argv: The argument list, read again when .env was written here.

    Returns:
        args, or the arguments read again after .env was written (_check_credential).

    Raises:
        SystemExit: With the line of the first check that stops the command.
    """
    _refuse_skip_env_setting(args)
    _refuse_empty_request(args)
    _refuse_unreadable_request(args)
    _refuse_long_request(args)

    # --preset and the stage flags are written into the environment, and config is read
    # again
    if getattr(args, "preset", None):
        _write_option_value("PRESET", args.preset)
        importlib.reload(config)
    _apply_stage_flag(args)

    _refuse_quiet_test_command(args)
    _refuse_no_engine_or_image(args)
    args = _check_credential(args, argv)
    _warn_no_partner_setting(args)
    _refuse_no_project(args)
    _refuse_wrong_place(args)
    _refuse_unwritable_out(args)
    _refuse_no_base(args)
    _sign_in(args)
    return args


def _start_container_command(args: argparse.Namespace) -> None:
    """Start the containers of sandbox, warm, and run and integrate without --here.

    Those four exit here. Before that, the stdout of run's containers is sent to files
    below DEBUG, a command that starts containers opens its log under the output
    directory, and integrate and run read their request off the branches when none was
    passed (request_of_integrate, request_of_run). Every other subcommand returns, run
    --here and integrate --here with that request in args.request.

    Args:
        args: The parsed arguments. args.request is set for integrate and run.

    Raises:
        SystemExit: For sandbox, warm, and run and integrate without --here: what
            _start_sandbox(), _start_warm(), _start_dispatch() and _start_integrate()
            raise. For run, with --here as well, when no request was passed and none can
            be read off the branches (request_of_run).
    """
    # Below DEBUG what the containers of run print goes to files
    progress.keep_child_stdout(args.command == "run" and not getattr(args, "here", False)
                               and config.AGENT_LOG_LEVEL != "DEBUG")

    # A command that starts containers from here keeps every line it logs in a file
    # under the output directory
    if _starts_container(args):
        config.OUTPUT_DIR = os.path.abspath(args.output_dir)
        source.project_name = knowledge_mod.project_name_of(args.project)
        turnlog.open_log()

    # sandbox makes the change in a container, and nothing below is read here
    if args.command == "sandbox":
        _start_sandbox(args)

    # integrate given no request reads it off the branches here, before it starts a
    # container
    if args.command == "integrate":
        args.request = request_of_integrate(
            os.path.abspath(args.project), args.request,
            [name for name in (args.resume, *args.branch_list) if name],
            resume=args.resume)

    # integrate without --here starts a container too. With it, the start-up below
    # merges the branches
    if args.command == "integrate" and not args.here:
        _start_integrate(args)

    # run given the branches alone reads its request off them here, before it starts a
    # container
    if args.command == "run":
        args.request = request_of_run(os.path.abspath(args.project), args.request,
                                      args.resume_list)

    # run without --here starts a container too, and then one agent per task it split
    # the request into (sandbox.dispatch)
    if args.command == "run" and not args.here:
        _start_dispatch(args)

    # warm starts a container of its own, with a network and without the credentials,
    # and nothing below is read here either
    if args.command == "warm":
        _start_warm(args)


def _prepare_union(args: argparse.Namespace, project_dir: str, path: str,
                   project_name: str, exclude_list: list[str],
                   is_reuse: bool) -> tuple[dict, str, bool]:
    """Merge the branches of integrate --here and analyse the union.

    The branches are merged while the repository still holds the state they started
    from, and what is read first is that state; the analysis is then made again over
    what the merge left in the working tree. It stops first while a sandboxed run is
    going over the same output directory (refuse_while_run_live).

    Args:
        args: The parsed arguments of integrate --here.
        project_dir: The target repository, as an absolute path.
        path: The knowledge file of the state the branches started from.
        project_name: The repository's directory name.
        exclude_list: The names kept out of the analysis.
        is_reuse: Whether that knowledge file was there already.

    Returns:
        (integrate.prepare()'s return value, the knowledge file of the union, whether it
        was used as it was).

    Raises:
        SystemExit: "integrate: <why>" when the analysis or the merge cannot be made.
    """
    refuse_while_run_live("integrate", args.project, args.output_dir)
    try:
        # An analysis that was there already is made again, without the documents, when
        # it does not describe what is checked out
        if is_reuse and not union_analysed(project_dir, path, project_name,
                                           args.output_dir, exclude_list):
            ensure_knowledge(args.project, args.output_dir, rebuild=True,
                             with_doc=False, exclude_list=exclude_list)
        prepare_dict = integrate_mod.prepare(
            project_dir, path,
            ([args.resume] if args.resume else []) + args.branch_list,
            onto=args.onto, resume=args.resume, output_dir=args.output_dir,
            exclude_list=exclude_list, onto_commit=args.onto_commit)
        # The repository stays in _HERE_UNION_LIST until the union is what the command
        # leaves
        _HERE_UNION_LIST.append((project_dir, prepare_dict, _has_union_note(args)))
        # The sides not merged into a file are written to --outcome-out before the
        # analysis and the model
        if args.outcome_out:
            _write_out(args.outcome_out, {"no_merge": integrate_mod.no_merge_record_list(
                prepare_dict["merged"])}, "the sides not merged into a file")
        # The analysis prepare() left last is used as it is when it describes the union
        path, _, _, is_reuse = ensure_knowledge(
            args.project, args.output_dir,
            rebuild=not union_analysed(project_dir, path, project_name,
                                       args.output_dir, exclude_list),
            with_doc=False, exclude_list=exclude_list)
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f"integrate: {e}") from None
    return prepare_dict, path, is_reuse


def _make_overview(args: argparse.Namespace, project_dir: str,
                   project_name: str) -> dict | None:
    """Make or read the repository's overview.

    overview, and outside a container a subcommand that searches, make the overview
    first when it is missing or not current, unless overview.is_on() says no. A
    subcommand that searches and makes none reads the one there (overview.load).

    Args:
        args: The parsed arguments.
        project_dir: The target repository, as an absolute path.
        project_name: The repository's directory name.

    Returns:
        overview.create()'s return value when it was called, else None.
    """
    overview_dict = None
    if overview.is_on() and (args.command == "overview" or (
            args.command in _SEARCH_COMMAND_TUPLE and not sandbox_mod.in_container())):
        overview_dict = overview.create(project_dir, args.output_dir, project_name)
    elif args.command in _SEARCH_COMMAND_TUPLE:
        overview.load(project_dir, args.output_dir, project_name)
    return overview_dict


def _finish_run_tests() -> None:
    """Run the test commands once and print what they did.

    load_index() put the primary language's test command in config, and this runs
    whatever the target repository's language uses. In a container that fills the
    package cache for the languages sandbox.fill_ext_list() names, that run is made only
    when it has a test command to run (edit.test_can_run) and sandbox.fill_own_run()
    says it is made, and then each of those languages whose test command is on PATH is
    made the primary one with no other language and its test command is run the same
    way, from the repository root. The tests' directory is made where the tests are
    reached by path and it is not there. In a filling of the package cache, each run is
    made once more when fill tests are written after it (_run_test_with_fill_test).

    Raises:
        SystemExit: 1 when a test command of the index's own run is known to have
            stopped short of its tests (edit.stop_short_run), or one of the languages
            sandbox.fill_ext_list() names, or a run made again with fill tests, is known
            to have stopped short of them because what they need could not be fetched
            (edit.fetch_fail_run).
    """
    fill_ext_list = sandbox_mod.fill_ext_list()
    edit.make_test_path()
    is_own_run = not fill_ext_list or (sandbox_mod.fill_own_run() and edit.test_can_run())
    is_stop_short = False
    if is_own_run:
        test_dict, is_fetch_fail = _run_test_with_fill_test()
        is_stop_short = edit.stop_short_run(test_dict) or is_fetch_fail
    for ext in fill_ext_list:
        config.set_primary_ext(ext)
        config.set_other_ext([])
        if not config.TEST_COMMAND or shutil.which(config.TEST_COMMAND[0]) is None:
            print(f"the test command of .{ext} files is not in this image, so it is not "
                  f"run")
            continue
        edit.make_test_path()
        test_dict, is_fetch_fail = _run_test_with_fill_test()
        is_stop_short = edit.fetch_fail_run(test_dict) or is_fetch_fail or is_stop_short
    if is_stop_short:
        raise SystemExit(1)


def _run_test_with_fill_test() -> tuple[dict, bool]:
    """Run the test commands once, and in a filling once more when fill tests are written.

    The fill tests are written after the first run (edit.write_fill_test).

    Returns:
        (the first run's edit.run_test() value, whether the second run is known to have
        stopped short of its tests because what they need could not be fetched
        (edit.fetch_fail_run), False without one).
    """
    test_dict = _run_tests_once()
    if sandbox_mod.fill_run() and edit.write_fill_test():
        return test_dict, edit.fetch_fail_run(_run_tests_once())
    return test_dict, False


def _run_tests_once() -> dict:
    """Run the test commands config holds once, and print what they did.

    Returns:
        edit.run_test()'s return value.
    """
    test_dict = edit.run_test()
    print_test(test_dict)
    return test_dict
