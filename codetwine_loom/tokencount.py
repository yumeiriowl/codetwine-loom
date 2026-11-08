"""The token counts of the model calls one process made or was told about.

add() adds one call's counts to this process's total, write() writes that total to a
file, and add_file() adds a total another process wrote to this process's.

MAX_TOKEN_PER_RUN is held against that total, and limit_text() says when it is reached.
child_file() makes the files of the processes this one starts in one directory, which is
removed when this process ends. remain_token() says how many tokens those files may hold
between them, and total_beside() how many this process and the other files beside its
own file hold.
"""

import atexit
import json
import logging
import os
import shutil
import tempfile
import threading

from codetwine_loom import config, tmpfile

logger = logging.getLogger(__name__)

# The keys of one call's token counts, and what each is read from on litellm's usage
# object: (key, the attribute on usage, the attribute under usage.prompt_tokens_details)
_FIELD_TUPLE = (("input_token", "prompt_tokens", ""),
                ("output_token", "completion_tokens", ""),
                ("cache_read_token", "", "cached_tokens"),
                ("cache_write_token", "", "cache_creation_tokens"))

# The keys of the counts, in the order they are written
KEY_TUPLE = tuple(key for key, _, _ in _FIELD_TUPLE)

# The key of how many calls the counts are of
CALL_KEY = "call"

# The keys of a total, the number of calls first
_TOTAL_KEY_TUPLE = (CALL_KEY, *KEY_TUPLE)

# The keys whose sum is held against MAX_TOKEN_PER_RUN. The tokens read from the cache
# and written to it are part of the input
_SUM_KEY_TUPLE = ("input_token", "output_token")

# What the files child_file() makes end in
_FILE_SUFFIX = ".json"

_lock = threading.RLock()

# Held while write() writes a file: one thread writes it at a time
_write_lock = threading.Lock()
_total_dict: dict[str, int] = {}
_part_list: list[dict] = []

# How many of the total's tokens add_file() added, and the files it added them from
_child_token = 0
_add_path_set: set[str] = set()

# The directory child_file() makes its files in, "" until the first one is made
_child_dir = ""


def count_dict_of(usage: object) -> dict:
    """Read one call's token counts off litellm's usage object.

    Args:
        usage: The response's usage, or None when the reply carried none.

    Returns:
        {"input_token", "output_token", "cache_read_token", "cache_write_token"}, each
        an int; 0 where the provider reported nothing.
    """
    detail = getattr(usage, "prompt_tokens_details", None)
    count_dict = {}
    for key, name, detail_name in _FIELD_TUPLE:
        value = getattr(usage, name, None) if name else getattr(detail, detail_name, None)
        count_dict[key] = value if isinstance(value, int) else 0
    return count_dict


def _whole_number(value: object) -> int:
    """Return value when it is a whole number of 0 or more, and 0 otherwise.

    Args:
        value: What a file or a reply held under one of the keys.

    Returns:
        value, or 0.
    """
    return value if type(value) is int and value >= 0 else 0


def add(count_dict: dict, call_count: int = 1, part: str = "") -> None:
    """Add token counts to this process's total.

    Args:
        count_dict: The counts by KEY_TUPLE. A key that is missing or holds anything
            but a whole number of 0 or more adds nothing.
        call_count: How many calls the counts are of.
        part: What made the calls, which total_dict() lists the counts under. Empty
            lists nothing.
    """
    clean_dict = {key: _whole_number(count_dict.get(key)) for key in KEY_TUPLE}
    clean_dict[CALL_KEY] = _whole_number(call_count)
    with _lock:
        for key, value in clean_dict.items():
            _total_dict[key] = _total_dict.get(key, 0) + value
        if part:
            _part_list.append({"part": part, **clean_dict, "total": token_sum(clean_dict)})


def token_sum(count_dict: dict) -> int:
    """Return the number held against MAX_TOKEN_PER_RUN: the input and the output tokens.

    Args:
        count_dict: The counts by KEY_TUPLE.

    Returns:
        The sum of "input_token" and "output_token", each 0 when it is not a whole
        number of 0 or more.
    """
    return sum(_whole_number(count_dict.get(key)) for key in _SUM_KEY_TUPLE)


def total_dict() -> dict:
    """Return what this process has counted so far.

    Returns:
        {"call", "input_token", "output_token", "cache_read_token",
        "cache_write_token", "total" (token_sum() of them), "parts": [{"part", the
        same six keys}, ...] in the order they were added}.
    """
    with _lock:
        count_dict = {key: _total_dict.get(key, 0) for key in _TOTAL_KEY_TUPLE}
        part_list = [dict(part_dict) for part_dict in _part_list]
    return {**count_dict, "total": token_sum(count_dict), "parts": part_list}


def reset() -> None:
    """Start this process's total again at 0, and remove the files child_file() made."""
    global _child_token
    with _lock:
        _total_dict.clear()
        _part_list.clear()
        _add_path_set.clear()
        _child_token = 0
    _remove_child_dir()


def child_file() -> str:
    """Make the empty file a process this one starts writes its token counts to.

    The file is made in one directory of this process's own, in the temporary directory.
    The directory is removed when this process ends (_remove_child_dir).

    Returns:
        The path, as an absolute path, or "" when the file could not be made.
    """
    global _child_dir
    try:
        with _lock:
            if not _child_dir:
                _child_dir = tempfile.mkdtemp(prefix="ctloom-token-")
            token_fd, token_path = tempfile.mkstemp(suffix=_FILE_SUFFIX, dir=_child_dir)
    except OSError as e:
        logger.warning("no file for the token counts could be made (%s), so the calls "
                       "of this part are not counted", e)
        return ""
    os.close(token_fd)
    return token_path


def _remove_child_dir() -> None:
    """Remove the directory child_file() made, with the files in it."""
    global _child_dir
    with _lock:
        directory, _child_dir = _child_dir, ""
    if directory:
        shutil.rmtree(directory, ignore_errors=True)


atexit.register(_remove_child_dir)


def _file_count_dict(path: str) -> dict:
    """Read the total another process wrote with write().

    Args:
        path: The file.

    Returns:
        The counts by CALL_KEY and KEY_TUPLE; every one 0 for a file that is not
        there, is empty or does not hold a JSON object.
    """
    try:
        read_dict = tmpfile.read_json(path)
    except (OSError, ValueError):
        read_dict = {}
    if not isinstance(read_dict, dict):
        read_dict = {}
    return {key: _whole_number(read_dict.get(key)) for key in _TOTAL_KEY_TUPLE}


def _dir_total(directory: str, skip_name: str = "") -> int:
    """Return the tokens the files child_file() made in one directory hold between them.

    Args:
        directory: The directory. One that is not there, or "", holds none.
        skip_name: The name of a file in it to leave out, or "" for none.

    Returns:
        token_sum() over every file there that ends in _FILE_SUFFIX.
    """
    try:
        name_list = os.listdir(directory) if directory else []
    except OSError:
        return 0
    return sum(token_sum(_file_count_dict(os.path.join(directory, name)))
               for name in name_list
               if name.endswith(_FILE_SUFFIX) and name != skip_name)


def total_beside(out_path: str) -> int:
    """Return the tokens this process and the processes writing beside it have used.

    Args:
        out_path: The file this process writes its own total to (write). Empty counts
            this process alone.

    Returns:
        token_sum() of this process's total, plus that of every other file child_file()
        made in out_path's directory.
    """
    total = total_dict()["total"]
    if not out_path:
        return total
    return total + _dir_total(os.path.dirname(out_path), os.path.basename(out_path))


def _own_total() -> int:
    """Return the tokens this process's own calls have used, without what add_file() added.

    Returns:
        token_sum() of the total, less what add_file() added.
    """
    with _lock:
        return total_dict()["total"] - _child_token


def write(path: str) -> None:
    """Write this process's total to a file.

    A file that cannot be written is logged and nothing is raised.

    Args:
        path: Where to write it. Its directory has to exist already.
    """
    try:
        with _write_lock:
            count_dict = total_dict()
            del count_dict["parts"]
            with tmpfile.atomic_write(path) as tmp_path:
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    json.dump(count_dict, handle)
    except OSError as e:
        logger.warning("the token counts could not be written to %s (%s)", path, e)


def add_file(path: str, part: str = "") -> dict:
    """Add the total another process wrote with write() to this process's total.

    Args:
        path: The file. One that is not there, is empty or does not hold a JSON object
            adds nothing, and neither does one that was added before.
        part: What made the calls, as add() takes it.

    Returns:
        The counts that were added, by CALL_KEY and KEY_TUPLE; every one 0 when the
        file held nothing to add.
    """
    global _child_token
    file_path = os.path.abspath(path)
    with _lock:
        if file_path in _add_path_set:
            return {key: 0 for key in _TOTAL_KEY_TUPLE}
        count_dict = _file_count_dict(file_path)
        if any(count_dict.values()):
            _add_path_set.add(file_path)
            add(count_dict, count_dict[CALL_KEY], part)
            _child_token += token_sum(count_dict)
    return count_dict


def add_remain_file(part: str) -> None:
    """Add the files child_file() made that add_file() has not been given yet.

    Args:
        part: What the counts are listed under, as add() takes it.
    """
    with _lock:
        directory = _child_dir
    try:
        name_list = sorted(os.listdir(directory)) if directory else []
    except OSError:
        return
    for name in name_list:
        if name.endswith(_FILE_SUFFIX):
            add_file(os.path.join(directory, name), part)


def limit_text() -> str:
    """Say that the run's model calls have reached MAX_TOKEN_PER_RUN, or "" when not.

    What is held against the setting is this process's own calls and what the files
    child_file() made hold now, which takes in the processes still running.

    Returns:
        The sentence naming that total and the setting, or "" when the setting is 0 or
        the total is below it.
    """
    limit = config.MAX_TOKEN_PER_RUN
    if limit <= 0:
        return ""
    with _lock:
        child_dir = _child_dir
    total = max(total_dict()["total"], _own_total() + _dir_total(child_dir))
    if total < limit:
        return ""
    return (f"the model calls of this run have used {total} token(s), which reaches "
            f"MAX_TOKEN_PER_RUN={limit}")


def remain_token() -> int:
    """Return how many tokens the processes this one starts may use between them.

    Returns:
        MAX_TOKEN_PER_RUN minus what this process's own calls have used, which leaves
        out what add_file() added, at least 1; 0 when the setting is 0, which sets no
        limit.
    """
    limit = config.MAX_TOKEN_PER_RUN
    if limit <= 0:
        return 0
    return max(1, limit - _own_total())


def line(count_dict: dict) -> str:
    """Return the one line that says what the counts are.

    Examples:
        {"call": 3, "input_token": 1200, "output_token": 300, "cache_read_token": 800,
         "cache_write_token": 100, "total": 1500}
            -> "1,500 in 3 call(s): 1,200 input (800 read from the cache, 100 written
               to it), 300 output"

    Args:
        count_dict: The counts, as total_dict() returns them.

    Returns:
        The line, each number written with thousands separators.
    """
    return (f"{count_dict['total']:,} in {count_dict[CALL_KEY]:,} call(s): "
            f"{count_dict['input_token']:,} input ({count_dict['cache_read_token']:,} "
            f"read from the cache, {count_dict['cache_write_token']:,} written to it), "
            f"{count_dict['output_token']:,} output")
