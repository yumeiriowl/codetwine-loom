"""The record one command leaves of itself, one JSON file per command.

write() writes one command's record into the records' directory, in a file named after
its "id", and read_list() reads the records back, oldest first. remove_old() removes the
records beyond the newest few, keeping the ones its is_open says True for.
"""

import json
import logging
import math
import os
from collections.abc import Callable

from codetwine_loom import tmpfile

logger = logging.getLogger(__name__)

_SUFFIX = ".json"


def _path_of(directory: str, record_id: str) -> str:
    """Return the file one command's record is written to.

    Args:
        directory: The records' directory, as an absolute path.
        record_id: The command's identifier, which starts with the time it started.

    Returns:
        "<directory>/<record_id>.json".
    """
    return os.path.join(directory, f"{record_id}{_SUFFIX}")


def write(directory: str, record_dict: dict) -> str:
    """Write one command's record, making the directory when it is not there.

    A record that cannot be written is logged and nothing is raised.

    Args:
        directory: The records' directory, as an absolute path.
        record_dict: The record. "id" names its file. A value json cannot write is
            written as its str().

    Returns:
        The file's path, or "" when it could not be written.
    """
    path = _path_of(directory, record_dict["id"])
    try:
        os.makedirs(directory, exist_ok=True)
        with tmpfile.atomic_write(path) as tmp_path:
            with open(tmp_path, "w", encoding="utf-8", errors="backslashreplace") as handle:
                json.dump(record_dict, handle, ensure_ascii=False, indent=1, default=str)
    except OSError as e:
        logger.warning("the record of this run could not be written to %s (%s)", path, e)
        return ""
    return path


def _is_text(value: object) -> bool:
    """Say whether a value is a string."""
    return isinstance(value, str)


def _is_number(value: object) -> bool:
    """Say whether a value is a whole or a real number that is finite."""
    return type(value) in (int, float) and math.isfinite(value)


def _is_argument(value: object) -> bool:
    """Say whether a value is a string a program can be handed as one argument.

    Such a string holds no NUL character and UTF-8 can encode it.
    """
    if not isinstance(value, str) or "\0" in value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


# The key a command's result holds its union under, the key the result and the union
# hold a branch name under, and the key the result holds the branches a command was
# passed with --branch under
_UNION_KEY = "integration"
_BRANCH_KEY = "branch"
_PASS_BRANCH_KEY = "pass_branch"

# The key the result holds the unions under that the union of a branch "resume" names
# stands for once it is put into the working tree
STAND_FOR_KEY = "stand_for"


def _is_result(value: object) -> bool:
    """Say whether a value has the shape of a command's result.

    Args:
        value: What the record holds under "result".

    Returns:
        True for an object whose branch, and whose union's branch, is not a string
        or is one _is_argument() takes, and whose "pass_branch" and "stand_for",
        where it holds them, are lists of strings _is_argument() takes.
    """
    if not isinstance(value, dict):
        return False
    holder_list = [value, value.get(_UNION_KEY)]
    name_list_tuple = (value.get(_PASS_BRANCH_KEY, []), value.get(STAND_FOR_KEY, []))
    return all(isinstance(branch_list, list)
               and all(_is_argument(branch) for branch in branch_list)
               for branch_list in name_list_tuple) \
        and all(not isinstance(holder_dict.get(_BRANCH_KEY), str)
                or _is_argument(holder_dict[_BRANCH_KEY])
                for holder_dict in holder_list if isinstance(holder_dict, dict))


# The keys of a record whose value is checked, each with what says its value is of the
# right shape. Any of them may be missing
_SHAPE_DICT = {
    **dict.fromkeys(("id", "command", "request", "error", "error_command"), _is_text),
    "exit_status": lambda value: type(value) is int,
    "elapsed_sec": _is_number,
    "token": lambda value: isinstance(value, dict) and (
        "total" not in value or _is_number(value["total"])),
    "resume": lambda value: isinstance(value, list) and all(
        _is_argument(branch) for branch in value),
    "result": _is_result,
}


def _is_record(record: object) -> bool:
    """Say whether what one file holds has the shape of a record.

    Args:
        record: What the file holds.

    Returns:
        True for an object whose keys named in _SHAPE_DICT each hold a value of that
        shape.
    """
    return isinstance(record, dict) and all(
        is_shape(record[key]) for key, is_shape in _SHAPE_DICT.items() if key in record)


def read_list(directory: str) -> list[dict]:
    """Return every record in the directory, oldest first.

    A file that cannot be read, and one that does not hold a record of the right shape
    (_is_record), is left out and named in the log.

    Args:
        directory: The records' directory. One that is not there holds none.

    Returns:
        The records, each with "path" set to its file, in the order of their file names.
        Where a record holds them, "id", "command", "request", "error" and
        "error_command" are strings, "exit_status" an int, "elapsed_sec" and the "total"
        of "token" finite numbers, "token" and "result" objects and "resume" a list of
        strings _is_argument() takes, as is the branch of "result" and of its union
        where that is a string.
    """
    try:
        name_list = sorted(name for name in os.listdir(directory)
                           if name.endswith(_SUFFIX))
    except OSError:
        return []
    record_list = []
    for name in name_list:
        path = os.path.join(directory, name)
        try:
            record_dict = tmpfile.read_json(path)
        except (OSError, ValueError) as e:
            logger.warning("the record %s could not be read (%s)", path, e)
            continue
        if not _is_record(record_dict):
            logger.warning("the record %s is not of the shape a record is written in "
                           "and is left out", path)
            continue
        record_list.append({**record_dict, "path": path})
    return record_list


def remove_old(directory: str, keep_count: int,
               is_open: Callable[[dict], bool]) -> list[str]:
    """Remove the records beyond the newest few that nothing can be carried on from.

    Args:
        directory: The records' directory.
        keep_count: How many of the newest records stay whatever they hold. 0 or less
            removes nothing.
        is_open: (a record) -> whether what it names can still be carried on from. A
            record it says True for stays whatever its age.

    Returns:
        The paths of the files that were removed.
    """
    if keep_count <= 0:
        return []
    remove_list = []
    for record_dict in read_list(directory)[:-keep_count]:
        if is_open(record_dict):
            continue
        try:
            os.remove(record_dict["path"])
        except OSError as e:
            logger.warning("the record %s could not be removed (%s)",
                           record_dict["path"], e)
            continue
        remove_list.append(record_dict["path"])
    return remove_list
