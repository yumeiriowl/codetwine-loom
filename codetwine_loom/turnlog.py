"""Keep what each turn of a run wrote and what came back, and what this tool logged.

Each turn goes into one JSONL file per run under <OUTPUT_DIR>/<project_name>/turns, one
line per turn. What this tool logged during that run goes into a .log file of the same
name beside it.

Lines are written as the run goes: a run that is killed still leaves what it did up to
that point. Each line says which stage wrote it, and the lines are in the order they
were written.

The .log file holds the lines the root logger is given while the run is going, with the
time and the level, and the traceback of a run that ended with an error.
"""

import contextlib
import json
import logging
import os
import secrets
import tempfile
from collections.abc import Iterator
from datetime import datetime

from codetwine_loom import config
from codetwine_loom.tools import source

logger = logging.getLogger(__name__)

# The directory the per-run files go in, inside <output_dir>/<project name>/
TURN_LOG_DIR = "turns"

# The two files of one run, told apart by their suffix
_TURN_SUFFIX = ".jsonl"
_LOG_SUFFIX = ".log"
_SUFFIX_TUPLE = (_TURN_SUFFIX, _LOG_SUFFIX)

# The line format of the .log file: the time, the level, the logger
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# The open turn file of the run that is going on, and whether writing to it has failed.
# A failed write stops the logging for the rest of the run rather than being reported
# once per turn
_turn_file = None
_is_write_fail = False

# The handler that writes the run's .log file, while one is open
_handler = None

# How many times each stage has started since the file was opened. {stage: count}
_stage_count_dict: dict[str, int] = {}


def directory() -> str:
    """Return the directory the per-run files go in.

    Returns:
        <OUTPUT_DIR>/<project_name>/turns.
    """
    return os.path.join(config.OUTPUT_DIR, source.project_name or "", TURN_LOG_DIR)


def _trim(keep: int) -> None:
    """Take away the oldest runs' files, leaving at most keep - 1 runs beside the new one.

    How old a run is is the time its files were last written, the later of the two.
    Runs written at the same time are ordered by name.

    Args:
        keep: How many runs the directory holds once the new one is written. A run is
            its .jsonl and .log files together.
    """
    try:
        name_list = os.listdir(directory())
    except OSError:
        return
    time_dict: dict[str, float] = {}
    for name in name_list:
        if not name.endswith(_SUFFIX_TUPLE):
            continue
        stem = os.path.splitext(name)[0]
        try:
            write_time = os.path.getmtime(os.path.join(directory(), name))
        except OSError:
            write_time = 0.0
        time_dict[stem] = max(time_dict.get(stem, 0.0), write_time)
    stem_list = sorted(time_dict, key=lambda stem: (time_dict[stem], stem))
    for stem in stem_list[:max(len(stem_list) - keep + 1, 0)]:
        for suffix in _SUFFIX_TUPLE:
            try:
                os.remove(os.path.join(directory(), stem + suffix))
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.info("an old turn log could not be taken away: %s", e)


def open_log(start_name: str = "", log_filter: logging.Filter | None = None) -> str:
    """Start the two files for this run, taking the oldest runs' files away first.

    The .log file takes every line the root logger is given from here until close().
    Both files take a character UTF-8 cannot encode as its backslash escape.

    Args:
        start_name: What names the file, without an extension. Empty names it after the
            time now and a random part that tells it from the files of another run
            started in the same second.
        log_filter: A filter every record passes before it is written to the .log
            file. None passes each record as it is.

    Returns:
        The path of the .jsonl file that was opened, or "" when the setting keeps none
        or making the directory or opening either file raised OSError.
    """
    global _turn_file, _is_write_fail, _handler
    close()
    _is_write_fail = False
    _stage_count_dict.clear()
    if config.TURN_LOG_KEEP <= 0:
        return ""
    name = start_name or (f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
                          f"{secrets.token_hex(3)}")
    stem = os.path.join(directory(), name)
    path = stem + _TURN_SUFFIX
    try:
        os.makedirs(directory(), exist_ok=True)
        _trim(config.TURN_LOG_KEEP)
        _turn_file = open(path, "a", encoding="utf-8", errors="backslashreplace")
        _handler = logging.FileHandler(stem + _LOG_SUFFIX, encoding="utf-8",
                                       errors="backslashreplace")
    except OSError as e:
        logger.warning("the turn log could not be opened at %s (%s). The run goes on "
                       "without one", path, e)
        close()
        return ""
    _handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    if log_filter is not None:
        _handler.addFilter(log_filter)
    logging.getLogger().addHandler(_handler)
    logger.info("writing what each turn did to %s", path)
    return path


def start_stage(stage: str) -> int:
    """Count one more start of a stage, for the "call" of the turns it writes.

    Args:
        stage: The stage that starts, as record() takes it.

    Returns:
        Which start of that stage it is since the file was opened, counting from 1.
    """
    _stage_count_dict[stage] = _stage_count_dict.get(stage, 0) + 1
    return _stage_count_dict[stage]


def record(stage: str, turn: int, code: str, output: str,
           usage_dict: dict | None = None, call: int = 0) -> None:
    """Write one turn, or do nothing when no file is open.

    Args:
        stage: Which stage the turn belongs to, written as the line's "stage".
        turn: Which turn of that stage it is, counting from 1.
        code: The block that was run.
        output: What came back.
        usage_dict: The token counts of the turn's calls added up, as
            broker.usage_dict_of() shapes them. None or empty writes no "usage".
        call: Which start of the stage the turn belongs to (start_stage()). 0 writes no
            "call".
    """
    global _is_write_fail
    if _turn_file is None or _is_write_fail:
        return
    line_dict = {"at": datetime.now().isoformat(timespec="seconds"), "stage": stage,
                 "turn": turn, "code": code, "output": output}
    if call:
        line_dict["call"] = call
    if usage_dict:
        line_dict["usage"] = usage_dict
    try:
        _turn_file.write(json.dumps(line_dict, ensure_ascii=False) + "\n")
        _turn_file.flush()
    except OSError as e:
        logger.warning("the turn log could not be written (%s). The rest of this run is "
                       "not logged", e)
        _is_write_fail = True


def close() -> None:
    """Close the files, whichever are open. Calling it twice is not an error."""
    global _turn_file, _handler
    if _turn_file is not None:
        try:
            _turn_file.close()
        except OSError:
            pass
        _turn_file = None
    if _handler is not None:
        logging.getLogger().removeHandler(_handler)
        _handler.close()
        _handler = None


@contextlib.contextmanager
def run_dir() -> Iterator[None]:
    """Work from a directory of this run's own for the length of the block.

    Yields:
        Nothing. The working directory is put back however the block ended.
    """
    before = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="ctloom-run-") as temp_dir:
        os.chdir(temp_dir)
        try:
            yield
        finally:
            os.chdir(before)


@contextlib.contextmanager
def open_run() -> Iterator[None]:
    """Keep the two files open around one run, and write how it ended with an error.

    The run works from a directory of its own (run_dir). A relative path is resolved
    from that directory inside the block: a caller makes the paths it reads there
    absolute before it enters.

    The traceback is logged once, and the process exits with status 1, as an uncaught
    exception would end it. SystemExit and KeyboardInterrupt pass through as they are.

    Yields:
        Nothing. The files are closed however the block ended.

    Raises:
        SystemExit: With status 1, in place of the exception the run raised.
    """
    open_log()
    try:
        with run_dir():
            yield
    except Exception as e:
        logger.exception("the run ended with %s: %s", type(e).__name__, e)
        raise SystemExit(1) from None
    finally:
        close()
