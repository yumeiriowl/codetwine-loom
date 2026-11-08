"""Read which analysis an index was built from, in a process of its own.

Nothing of this package is imported here. main() reads one JSON object from the standard
input (the request) and writes one to the standard output (the answer).

The request holds
- "uri": the URI the index is opened with,
- "table": {name: the statement the table of that name is to be made with},
- "max_address_byte": how much address space this process may take,
- "max_answer_byte": how many bytes this process may write to a file,
- "max_second": how many seconds this process may run from when it read the request,
- "progress_step" and "max_progress_call": how many instructions SQLite runs between
  two counts, and how many counts the connection is given,
- "max_value_byte": how many bytes a text or a blob the connection reads may hold,
- "max_pattern_byte": how many bytes the pattern of a LIKE or a GLOB may hold.

The answer is {"stamp": the source stamp, "commit": the commit}, each "" where the
index holds no text under it, or {"error": why the file is no index to read}.
"""

import json
import resource
import signal
import sqlite3
import sys


def _hold_process(max_address_byte: int, max_answer_byte: int, max_second: float) -> None:
    """Limit this process's address space, the size of a file it writes and its run time.

    Neither of the first two limits is set higher than the soft and the hard one the
    process runs under. Once the time is up the process is ended by SIGALRM, whatever
    it is in the middle of and whether or not the process that started it is still
    there.

    Args:
        max_address_byte: The address space, in bytes.
        max_answer_byte: The size of a file, in bytes.
        max_second: The time, in seconds from now.
    """
    # The process ends on SIGALRM however it was started: the signal is neither
    # ignored nor held back
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
    signal.setitimer(signal.ITIMER_REAL, max_second)
    for name, limit in ((resource.RLIMIT_AS, max_address_byte),
                        (resource.RLIMIT_FSIZE, max_answer_byte)):
        for bound in resource.getrlimit(name):
            if bound != resource.RLIM_INFINITY:
                limit = min(limit, bound)
        resource.setrlimit(name, (limit, limit))


def _bound_read(connection: sqlite3.Connection, progress_step: int,
                max_progress_call: int, max_value_byte: int,
                max_pattern_byte: int) -> None:
    """Limit what a connection runs and what it reads.

    The connection is stopped once it has run progress_step * max_progress_call
    instructions in all. A query stopped by that count raises sqlite3.OperationalError,
    and one that meets a text or a blob of more than max_value_byte bytes
    sqlite3.DataError. A pattern of a LIKE or a GLOB holds max_pattern_byte bytes at
    most.

    Args:
        connection: The connection to bound.
        progress_step: How many instructions SQLite runs between two counts.
        max_progress_call: How many counts the connection is given.
        max_value_byte: How many bytes a text or a blob may hold.
        max_pattern_byte: How many bytes a pattern may hold.
    """
    call_count = 0

    def is_over() -> int:
        """Count one more call, and say whether the connection is over its bound."""
        nonlocal call_count
        call_count += 1
        return int(call_count > max_progress_call)

    connection.set_progress_handler(is_over, progress_step)
    connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max_value_byte)
    connection.setlimit(sqlite3.SQLITE_LIMIT_LIKE_PATTERN_LENGTH, max_pattern_byte)


def _refuse_other_table(connection: sqlite3.Connection, table_dict: dict) -> None:
    """Confirm that each name is the table its statement makes, and nothing else.

    What the database holds under the name, in whatever case, is compared with the
    statement: a view, a virtual table, and a table of other columns or with a column
    computed when it is read are all refused.

    Args:
        connection: The connection to ask.
        table_dict: {name: the statement as SQLite keeps it in sqlite_master}.

    Raises:
        sqlite3.DatabaseError: When a name is not that table.
    """
    for name, statement in table_dict.items():
        row_list = connection.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = ? COLLATE NOCASE",
            (name,)).fetchall()
        if row_list != [("table", statement)]:
            raise sqlite3.DatabaseError(
                f"{name} is not the table an index holds under that name")


def _read_analysis(request_dict: dict) -> dict:
    """Read the source stamp and the commit of the index a request names.

    The file is held to this: meta and calls are the tables their statements make, calls
    has the columns src and dst, and file_lang is the text of a JSON object.

    Args:
        request_dict: The request, as the module's docstring lays it out.

    Returns:
        {"stamp": the source stamp, "commit": the commit}, each "" where the index
        holds no text under it.

    Raises:
        sqlite3.DatabaseError: When the file is no such index, or reading it runs
            past a bound.
    """
    connection = sqlite3.connect(request_dict["uri"], uri=True)
    try:
        _bound_read(connection, request_dict["progress_step"],
                    request_dict["max_progress_call"], request_dict["max_value_byte"],
                    request_dict["max_pattern_byte"])
        _refuse_other_table(connection, request_dict["table"])
        meta_dict = dict(connection.execute(
            "SELECT key, value FROM meta WHERE key IN ('source_stamp', 'commit', "
            "'file_lang')"))
        connection.execute("SELECT src, dst FROM calls LIMIT 0")
    finally:
        connection.close()
    file_lang = meta_dict.get("file_lang", "{}")
    try:
        file_lang_dict = json.loads(file_lang) if isinstance(file_lang, str) else None
    except (ValueError, RecursionError):
        file_lang_dict = None
    if not isinstance(file_lang_dict, dict):
        raise sqlite3.DatabaseError("the index holds no JSON object under file_lang")
    stamp, commit = (meta_dict.get(key, "") for key in ("source_stamp", "commit"))
    return {"stamp": stamp if isinstance(stamp, str) else "",
            "commit": commit if isinstance(commit, str) else ""}


def main() -> None:
    """Read one request from standard input and write its answer to standard output."""
    request_dict = json.load(sys.stdin)
    _hold_process(request_dict["max_address_byte"], request_dict["max_answer_byte"],
                  request_dict["max_second"])
    try:
        answer_dict = _read_analysis(request_dict)
    except (sqlite3.Error, MemoryError) as e:
        answer_dict = {"error": str(e) or type(e).__name__}
    json.dump(answer_dict, sys.stdout)
