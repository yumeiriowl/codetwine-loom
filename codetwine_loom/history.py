"""Carry one change over to the next run.

A change too large for one run is picked up by the next one: this module keeps what a
run did in a small JSON file, and renders what earlier runs left as a block for the next
run's request (render()).

The file lives in <OUTPUT_DIR>/<project_name>/edit_history.json.

Only machine signals decide whether a change is finished. Nothing here asks an LLM.

Of an entry, only the contents of the keys in _RENDER_KEY_TUPLE, and the paths under
"files" and "untooled" (_file_line()), can appear in the block render() builds. What is
known beyond the machine's signals is written through resolve(), into fields whose
contents render() never prints. "verified" is the one resolve() field render() looks at;
its value never appears in the block.

The block covers what earlier runs left in this repository, not one request alone: the
paths of every live attempt are merged rather than only the newest one's, an attempt
that landed no edit is not what the block is built from, and a request that is not the
recorded one is told so instead of being told it is a continuation of itself.
"""

import contextlib
import json
import logging
import os
from collections.abc import Iterator
from datetime import datetime

from codetwine_loom import config, filelock, gitcmd, tmpfile
from codetwine_loom.paths import list_text
from codetwine_loom.rlm import cut, cut_tail
from codetwine_loom.textcut import one_line
from codetwine_loom.tools import edit, run, source

logger = logging.getLogger(__name__)

# The schema version written into the file. A file carrying anything else is ignored
# rather than migrated
SCHEMA = 1

# The history file inside <output_dir>/<project name>/
HISTORY_FILE = "edit_history.json"

# The most bytes read of a history file
FILE_MAX_BYTE = 4 * 1024 * 1024

# What stands in for an answer a run never gave, and for a verdict nothing recorded
NO_ANSWER_TEXT = "(it never said)"
NO_VERDICT_TEXT = "(nothing was recorded)"

# The verdicts judge() can reach on its own. There is no "passed" among them
VERDICT_FAIL = "failed"
VERDICT_NO_PROOF = "unverified"
VERDICT_TEST_RED = "tests_red"
VERDICT_NO_CHANGE = "no_change"
JUDGE_VERDICT_TUPLE = (VERDICT_FAIL, VERDICT_NO_PROOF, VERDICT_TEST_RED, VERDICT_NO_CHANGE)

# The verdicts that mean the change is still open
_OPEN_VERDICT_TUPLE = (VERDICT_FAIL, VERDICT_NO_PROOF, VERDICT_TEST_RED)

# The two resolutions that close an entry (resolve()). A resolution of "failed" leaves
# it open
VERDICT_PASS = "passed"
VERDICT_CLOSE = "closed"
_CLOSE_VERDICT_TUPLE = (VERDICT_PASS, VERDICT_CLOSE)


def end_text(record_dict: dict) -> str:
    """Render how one run ended, from the machine's own record of it.

    The reason is cut to EDIT_HISTORY_DID_CHAR.

    Args:
        record_dict: A dict holding "verdict" and "why", as integrate.commit_record_list()
            returns them.

    Returns:
        "<verdict> - <why>", the verdict alone when nothing said why, or NO_VERDICT_TEXT.
    """
    verdict, why = record_dict["verdict"], record_dict["why"]
    if not verdict:
        return NO_VERDICT_TEXT
    return f"{verdict} - {cut(why, config.EDIT_HISTORY_DID_CHAR)}" if why else verdict


# The only entry keys whose contents may appear in the rendered block, beside the paths
# under "files" and "untooled" (_file_line())
_RENDER_KEY_TUPLE = ("request", "did", "verdict", "why", "last_run")

# The lines that show the entry's "last_run" (_last_run_output())
_LAST_RUN = """
The last time it ran the program itself, this is what came back:
{output}"""

# The skeleton of the block, filled in with one of the three states in _STATE_DICT
_BLOCK = """## {heading}
{opening}
It was asked to: {request}
It wrote: {files}
It reported: {did}{state}{last_run}
{closing}"""

# The line that says the work is not done yet. The finished state leaves it out
_NOT_FINISH_LINE = "\nIt is not finished: {why}"

# The closing sentences: one for a continuation of the same request, one for a request
# that is a new one
_GO_ON = ("That work is present already; verifying it again proves nothing. Find what is"
          " still\nmissing and write that.")
_READ_THE_NEW_ONE = ("That work is present already; verifying it again proves nothing. "
                     "The request below is\na new one - read it as it is written.")

# The three states a rendered block can be in. Every opening says who wrote the files
_STATE_DICT = {
    # The same request asked again while the work it left is unfinished
    "continued": {
        "heading": "Unfinished change from the previous run",
        "opening": ("This repository is part-way through the request below. The files "
                    "named here are NOT\ncode that was already here - an earlier run of "
                    "this same request wrote them and stopped\nbefore the work was "
                    "done."),
        "state": _NOT_FINISH_LINE,
        "closing": _GO_ON,
    },
    # A different request is being asked while earlier work is still unfinished
    "open_other": {
        "heading": "Unfinished work already in this repository",
        "opening": ("The files named here are NOT code that was already here - an "
                    "earlier run in this\nrepository wrote them for the request quoted "
                    "below, and stopped before it was done."),
        "state": _NOT_FINISH_LINE,
        "closing": _READ_THE_NEW_ONE,
    },
    # A different request is being asked after the earlier one was confirmed finished
    "done_other": {
        "heading": "Earlier work already in this repository",
        "opening": ("The files named here are NOT code that was already here - an "
                    "earlier run in this\nrepository wrote them for the request quoted "
                    "below, which is finished."),
        "state": "",
        "closing": _READ_THE_NEW_ONE,
    },
}


def path() -> str:
    """Return the history file's path.

    Returns:
        <OUTPUT_DIR>/<project_name>/edit_history.json.
    """
    return os.path.join(config.OUTPUT_DIR, source.project_name or "", HISTORY_FILE)


def _empty_dict() -> dict:
    """Return the skeleton used for a missing or unreadable file.

    Returns:
        {"schema": SCHEMA, "project": str, "attempts": []}.
    """
    return {"schema": SCHEMA, "project": source.project_name or "", "attempts": []}


def _is_path_list(value: object) -> bool:
    """Say whether a value is a list of strings."""
    return isinstance(value, list) and all(isinstance(file_rel, str) for file_rel in value)


def _is_text(value: object) -> bool:
    """Say whether a value is a string."""
    return isinstance(value, str)


# The keys of an entry whose value is checked, each with what says its value is of the
# right shape
_ENTRY_SHAPE_DICT = {
    "n": lambda value: type(value) is int,
    **dict.fromkeys(("at", "verdict", "request", "did", "why", "last_run"), _is_text),
    "verified": lambda value: value is None or isinstance(value, str),
    "files": lambda value: isinstance(value, dict) and all(
        _is_path_list(path_list) for path_list in value.values()),
    **dict.fromkeys(("untooled", "tests_changed"), _is_path_list),
}

# The keys of _ENTRY_SHAPE_DICT every entry holds. Any other may be missing
_REQUIRE_KEY_TUPLE = ("n", "at", "verdict")


def _is_entry(entry: object) -> bool:
    """Say whether one entry of "attempts" has the shape of an entry.

    Args:
        entry: What the file holds as one entry.

    Returns:
        True for an object that holds every key of _REQUIRE_KEY_TUPLE and whose keys
        named in _ENTRY_SHAPE_DICT each hold a value of that shape.
    """
    return isinstance(entry, dict) \
        and all(key in entry for key in _REQUIRE_KEY_TUPLE) \
        and all(is_shape(entry[key]) for key, is_shape in _ENTRY_SHAPE_DICT.items()
                if key in entry)


def read_at(file_path: str) -> dict:
    """Read one history file, or an empty skeleton when there is nothing usable.

    A broken or foreign file is logged and an empty history is returned. So is one of
    more than FILE_MAX_BYTE bytes and one whose "attempts" is not a list. An entry of
    "attempts" that is not of the right shape (_is_entry) is left out and counted in the
    log.

    Args:
        file_path: The history file to read, as an absolute path.

    Returns:
        {"schema": int, "project": str, "attempts": list[dict]}, every entry holding an
        int "n", a string "at" and a string "verdict", and its other keys of
        _ENTRY_SHAPE_DICT, where it has them, values of that shape.
    """
    try:
        history_dict = tmpfile.read_json(file_path, FILE_MAX_BYTE)
    except FileNotFoundError:
        return _empty_dict()
    except (OSError, ValueError) as e:
        logger.warning("the edit history at %s could not be read (%s). Starting with an "
                       "empty one", file_path, e)
        return _empty_dict()
    if not isinstance(history_dict, dict) or history_dict.get("schema") != SCHEMA:
        logger.warning("the edit history at %s is not schema %d. Starting with an empty "
                       "one", file_path, SCHEMA)
        return _empty_dict()
    entry_list = history_dict.setdefault("attempts", [])
    if not isinstance(entry_list, list):
        logger.warning('the "attempts" of the edit history at %s is not a list. Starting '
                       'with an empty one', file_path)
        return _empty_dict()
    history_dict["attempts"] = [entry for entry in entry_list if _is_entry(entry)]
    if len(history_dict["attempts"]) < len(entry_list):
        logger.warning('%d of the %d entries of the edit history at %s are not of the '
                       'shape an attempt is recorded in and are left out',
                       len(entry_list) - len(history_dict["attempts"]), len(entry_list),
                       file_path)
    return history_dict


def load() -> dict:
    """Read this project's history file, as read_at() reads any other.

    Returns:
        {"schema": int, "project": str, "attempts": list[dict]}.
    """
    return read_at(path())


def last_attempt(file_path: str) -> dict:
    """Return the newest attempt in one history file, or {} when it holds none.

    The path is given, not built by path().

    Args:
        file_path: The history file to read, as an absolute path.

    Returns:
        The entry, or {}.
    """
    attempt_list = read_at(file_path)["attempts"]
    return attempt_list[-1] if attempt_list else {}


def save(history_dict: dict) -> None:
    """Write the history file, creating its directory when needed.

    It goes through tmpfile.atomic_write(): a run interrupted mid-write leaves the
    previous file intact, and its temporary file is left to the clean-up atomic_write()
    does.

    This never raises: a history that could not be written is logged. A character UTF-8
    cannot encode is written as its backslash escape.

    Args:
        history_dict: The whole history, as load() returns it.
    """
    file_path = path()
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with tmpfile.atomic_write(file_path) as tmp_path:
            with open(tmp_path, "w", encoding="utf-8", errors="backslashreplace") as fp:
                json.dump(history_dict, fp, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning("the edit history could not be written to %s (%s)", file_path, e)


@contextlib.contextmanager
def open_to_update() -> Iterator[dict]:
    """Yield this project's history to be changed, and write it back afterwards.

    Reading, changing and writing are one step, under the lock filelock takes: writers
    in different processes never read the same file and number two entries alike. When
    the directory of the history file cannot be made, that is logged and nothing is
    written.

    Yields:
        The history, as read_at() returns it, or an empty one when that directory cannot
        be made. What the block leaves in the history read_at() returned is what is
        written.
    """
    file_path = path()
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    except OSError as e:
        logger.warning("the edit history could not be written to %s (%s)", file_path, e)
        yield _empty_dict()
        return
    with filelock.hold_lock(file_path, "the edit history"):
        history_dict = read_at(file_path)
        yield history_dict
        save(history_dict)


def clear() -> None:
    """Delete the history file."""
    file_path = path()
    if os.path.exists(file_path):
        os.remove(file_path)


def judge(signal_dict: dict) -> tuple[str, str]:
    """Decide whether the change is finished, from machine signals alone.

    The signals are asked in order and the first red one is what comes back.

    Whether ending with nothing changed behind it is "nothing needed changing" or "it
    never got anywhere" is settled here, from "completed".

    Args:
        signal_dict: {"completed": bool, "edits_landed": bool, "tests_ok": bool | None,
            "check_ok": bool | None, "final_ok": bool | None, "error": str}.
            "completed" is the run having ended itself, not the harvest that follows a
            run out of turns. None means the signal was never taken; for tests_ok, that
            no run of the tests reached a verdict. "final_ok" is False when the check
            made at the end of a run on its own (agent.final_check) found something.
            "error" is why the run stopped early, "" when it did not.

    Returns:
        (verdict, why). verdict is VERDICT_NO_CHANGE, VERDICT_FAIL, VERDICT_TEST_RED or
        VERDICT_NO_PROOF; why is one machine-written line. The error is asked first.
    """
    if signal_dict.get("error"):
        return VERDICT_FAIL, "it stopped before the end, so what it changed is part-way"
    if not signal_dict.get("edits_landed"):
        # Ending itself with nothing changed is the run saying the request needed no
        # change. Running out of turns with nothing changed is not the same thing, and
        # "completed" is what tells the two apart
        if signal_dict.get("completed"):
            return VERDICT_NO_CHANGE, ("it said the request needed no change and left the "
                                       "repository as it was")
        return VERDICT_FAIL, "no edit landed at all"
    if not signal_dict.get("completed"):
        return VERDICT_FAIL, "the turns ran out before it said the change was done"
    if signal_dict.get("final_ok") is False:
        return VERDICT_FAIL, ("the static check or the references found something that "
                              "was not there before the change")
    if signal_dict.get("check_ok") is False:
        return VERDICT_FAIL, "the program did not run when it stopped"
    if signal_dict.get("tests_ok") is False:
        return VERDICT_TEST_RED, ("a test was failing when it stopped, which is a result to "
                                  "read rather than a run that did not get there")
    return VERDICT_NO_PROOF, ("nothing here checks that the request itself is done, "
                              "so the work may still be part-way")


def _signal_dict(result_dict: dict, no_tool_path_list: list[str]) -> dict:
    """Read the six machine signals from run_edit()'s return value and the editing tools.

    "completed" is the run having ended itself.

    "edits_landed" is true for a write made without an editing tool as well as for the
    writes the tools landed, and under --dry-run for what the tools would have written.

    Args:
        result_dict: The return value of agent.run_edit().
        no_tool_path_list: The paths this run changed without an editing tool, as the
            entry records them.

    Returns:
        The dict judge() takes.
    """
    smoke_count_dict = result_dict.get("smoke") or {}
    final_list = result_dict.get("final_check")
    # The check is read only when its baseline ran green before any edit
    baseline_ok = smoke_count_dict.get("baseline_ok")
    return {
        # A harvested answer is not a completion: the run used every turn and was asked
        # once what it had changed (EDIT_LATE_ANSWER)
        "completed": (result_dict.get("answer") is not None
                      and not result_dict.get("harvested")),
        # What is in the repository now: a write that was reverted leaves nothing
        # behind. A dry run counts what it would have written
        "edits_landed": bool(edit.write_path_set()) or bool(no_tool_path_list)
                        or bool(edit.dry_change_list()),
        "tests_ok": edit.last_test_ok(),
        "check_ok": smoke_count_dict.get("last_ok") if baseline_ok else None,
        "final_ok": None if final_list is None else not final_list,
        "error": result_dict.get("error") or ""}


def _change_by_kind_dict() -> dict:
    """List what this run wrote, split into added, modified and deleted.

    git decides the kind of each change, and the editing tools' own record of where a
    write landed decides which of them belong to this run.

    Returns:
        edit.change_by_kind_dict()'s three lists, narrowed to the paths a write landed
        on. All empty when git could not be run, which is logged and not raised.
    """
    try:
        by_kind_dict = edit.change_by_kind_dict()
    except OSError as e:
        logger.warning("the edit history could not list the changed files (%s). "
                       "Recording the attempt without them", e)
        return {kind: [] for kind in gitcmd.CHANGE_KIND_TUPLE}
    write_path_set = edit.write_path_set()
    return {kind: [file_rel for file_rel in path_list if file_rel in write_path_set]
            for kind, path_list in by_kind_dict.items()}


def _cut(text: str, limit: int) -> str:
    """Shorten text to one line, cutting it to limit characters and a marker.

    Args:
        text: The text to shorten.
        limit: The maximum number of characters to keep.

    Returns:
        The text as one line, cut and marked (rlm.cut) when it was over the limit. The
        marker is folded onto the same line.
    """
    return one_line(cut(one_line(text or ""), limit))


def _last_run_output() -> str:
    """Return the tail of what the last run of the program that ran to its end printed.

    The output is what the tool itself recorded of its run (run.last_output()): a turn
    that only names run_project in a string or a comment is not read as a run, and the
    program's output is carried over, not everything that turn printed.

    Returns:
        The last EDIT_HISTORY_LAST_RUN_CHAR characters of that output, with the
        "... (N chars omitted)" marker in front when it was cut, or "" when
        run.last_output() holds no run that ran to its end or that output is only white
        space.
    """
    return cut_tail(run.last_output().strip(), config.EDIT_HISTORY_LAST_RUN_CHAR)


def record(request: str, result_dict: dict) -> dict:
    """Judge what this run did and, with the history on, append it to the history file.

    The verdict is reached whether or not it is written: a run with the history switched
    off is judged the same way as one with it on.

    Args:
        request: The change request as it was given, before any history block.
        result_dict: The return value of agent.run_edit().

    Returns:
        The entry, with "verdict" and "why" among its keys. With EDIT_HISTORY off, and
        under --dry-run, it is built the same way and written nowhere.
    """
    if not config.EDIT_HISTORY or config.EDIT_DRY_RUN:
        return _add_entry(_empty_dict(), request, result_dict)
    with open_to_update() as history_dict:
        entry_dict = _add_entry(history_dict, request, result_dict)
    logger.info("edit history: attempt %d recorded as %s (%s)",
                entry_dict["n"], entry_dict["verdict"], entry_dict["why"])
    return entry_dict


def _add_entry(history_dict: dict, request: str, result_dict: dict) -> dict:
    """Add one attempt to a history, and return the entry.

    Args:
        history_dict: The history to add to: what open_to_update() yields, or an
            empty skeleton (_empty_dict()) when nothing is written. The attempt is
            numbered from what it holds.
        request: The change request as it was given.
        result_dict: The return value of agent.run_edit().

    Returns:
        The entry that was added.
    """
    no_tool_path_list = sorted(edit.no_tool_path_set())
    signal_dict = _signal_dict(result_dict, no_tool_path_list)
    verdict, why = judge(signal_dict)
    file_dict = _change_by_kind_dict()
    test_file_list = sorted(edit.test_change_set())
    entry_dict = {
        "n": (history_dict["attempts"][-1]["n"] + 1) if history_dict["attempts"] else 1,
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "request": _cut(request, config.EDIT_HISTORY_REQUEST_CHAR),
        "did": _cut(result_dict.get("answer") or "", config.EDIT_HISTORY_DID_CHAR),
        "files": {kind: path_list[:config.EDIT_HISTORY_MAX_FILE]
                  for kind, path_list in file_dict.items()},
        "verdict": verdict,
        "signals": signal_dict,
        "why": why,
        "last_run": _last_run_output(),
        # Filled in by resolve() when someone knows more than the signals do
        "verified": None,
        "verified_note": "",
    }
    if result_dict.get("final_check"):
        entry_dict["final_check"] = list(result_dict["final_check"])
    # Only written when there is one
    if no_tool_path_list:
        entry_dict["untooled"] = no_tool_path_list[:config.EDIT_HISTORY_MAX_FILE]
    # Same: only written when the run changed a test
    if test_file_list:
        entry_dict["tests_changed"] = test_file_list[:config.EDIT_HISTORY_MAX_FILE]
    history_dict["attempts"] = (history_dict["attempts"] + [entry_dict])[-config.EDIT_HISTORY_KEEP:]
    return entry_dict


def resolve(verdict: str, note: str = "") -> None:
    """Record what something outside the run knows about the last entry.

    It writes into "verified" and "verified_note".

    Nothing is written with EDIT_HISTORY off, and nothing under --dry-run.

    Args:
        verdict: What the caller concluded, e.g. "passed", "failed" or "closed".
        note: Free text, written as "verified_note".
    """
    if not config.EDIT_HISTORY or config.EDIT_DRY_RUN:
        return
    with open_to_update() as history_dict:
        if history_dict["attempts"]:
            history_dict["attempts"][-1]["verified"] = verdict
            history_dict["attempts"][-1]["verified_note"] = note


def _live_attempt_list() -> list[dict]:
    """Return the live entries of the history, oldest first.

    A "closed" resolution ends what is live: nothing recorded up to it is live.
    Everything after it is, including entries somebody resolved as "passed".

    Returns:
        The entries after the last dismissed one, or every entry when there is none.
    """
    attempt_list = load()["attempts"]
    for index in range(len(attempt_list) - 1, -1, -1):
        if attempt_list[index].get("verified") == VERDICT_CLOSE:
            return attempt_list[index + 1:]
    return attempt_list


def _entry_path_list(entry_dict: dict) -> list[str]:
    """List every path one entry recorded, whatever kind of change it was.

    Args:
        entry_dict: One attempt entry.

    Returns:
        The added, modified and deleted paths and the paths changed without an editing
        tool ("untooled"), sorted and without duplicates.
    """
    file_dict = entry_dict.get("files") or {}
    path_set = {file_rel for kind in gitcmd.CHANGE_KIND_TUPLE
                for file_rel in file_dict.get(kind, [])}
    return sorted(path_set | set(entry_dict.get("untooled") or []))


def _last_write_entry(attempt_list: list[dict]) -> dict | None:
    """Return the most recent entry that actually changed a file.

    An attempt that landed no edit is passed over.

    Args:
        attempt_list: The live entries, oldest first.

    Returns:
        The entry, or None when not one of them changed a file.
    """
    for entry_dict in reversed(attempt_list):
        if _entry_path_list(entry_dict):
            return entry_dict
    return None


def _live_write_entry() -> tuple[list[dict], dict | None]:
    """Return the live entries and the last of them that wrote a file.

    Returns:
        (_live_attempt_list()'s entries, _last_write_entry()'s entry). ([], None) with
        EDIT_HISTORY off; the entry is None when no live attempt wrote a file.
    """
    if not config.EDIT_HISTORY:
        return [], None
    attempt_list = _live_attempt_list()
    return attempt_list, _last_write_entry(attempt_list)


def open_attempt() -> dict | None:
    """Return the last live entry that wrote a file while it is still open, or None.

    A resolution of "passed" or "closed" closes the change. Any other resolution -
    "failed" above all - leaves it open.

    Whether anything is open is asked of the last attempt that actually wrote something.

    Returns:
        The entry, or None when there is nothing left open, when no live attempt wrote
        anything, and with EDIT_HISTORY off.
    """
    entry_dict = _live_write_entry()[1]
    if entry_dict is None:
        return None
    if entry_dict.get("verified") in _CLOSE_VERDICT_TUPLE \
            or entry_dict.get("verdict") not in _OPEN_VERDICT_TUPLE:
        return None
    return entry_dict


def _file_line(attempt_list: list[dict]) -> str:
    """Render the paths every live attempt wrote as one line.

    Every live attempt's paths are merged, not only the newest entry's.

    Args:
        attempt_list: The live entries, oldest first.

    Returns:
        "a.py, b.py and 3 more", or "none" when nothing was changed at all.
    """
    path_list = sorted({file_rel for entry_dict in attempt_list
                        for file_rel in _entry_path_list(entry_dict)})
    return list_text(path_list) or "none"


def _state_key(entry_dict: dict, request: str) -> str | None:
    """Decide which of _STATE_DICT describes an entry for the request about to be made.

    Args:
        entry_dict: The entry the block is built from, as _last_write_entry() returned it.
        request: The request about to be made. An empty string means the caller is not
            saying, and is read as a continuation.

    Returns:
        A key of _STATE_DICT, or None when there is nothing worth saying - the earlier work
        is finished and what is being asked now is that same request.
    """
    # Read off the entry the block describes, not off the newest one, which may be an
    # attempt that changed nothing
    is_finish = entry_dict.get("verified") == VERDICT_PASS
    record_request = entry_dict.get("request") or ""
    is_same = not request or record_request == _cut(request,
                                                    config.EDIT_HISTORY_REQUEST_CHAR)
    if not is_finish:
        return "continued" if is_same else "open_other"
    return None if is_same else "done_other"


def render(request: str = "") -> str:
    """Build the block describing what earlier runs left, for the next run's request.

    Of an entry, only the contents of the keys in _RENDER_KEY_TUPLE can appear in the
    block, beside the paths the live attempts wrote (_file_line()): whatever resolve()
    stored stays out of it.

    Args:
        request: The request this block will be placed in front of. It decides the
            wording: a continuation of the same request is told the earlier run was
            working on it, and a follow-up request is told the work it is looking at
            belongs to a different one. Left out, the block reads as a continuation.

    Returns:
        The block, or "" when the history is off, when nothing is left to say, or when
        no live attempt wrote a file.
    """
    attempt_list, entry_dict = _live_write_entry()
    if entry_dict is None:
        return ""
    key = _state_key(entry_dict, request)
    if key is None:
        return ""
    state_dict = _STATE_DICT[key]
    show_dict = {field: entry_dict.get(field) for field in _RENDER_KEY_TUPLE}
    last_run = show_dict["last_run"]
    return _BLOCK.format(
        heading=state_dict["heading"],
        opening=state_dict["opening"],
        request=show_dict["request"] or "(not recorded)",
        files=_file_line(attempt_list),
        did=show_dict["did"] or NO_ANSWER_TEXT,
        state=state_dict["state"].format(why=show_dict["why"] or show_dict["verdict"]),
        last_run=_LAST_RUN.format(output=last_run) if last_run else "",
        closing=state_dict["closing"])
