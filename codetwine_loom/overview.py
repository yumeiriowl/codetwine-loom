"""The repository's overview, made and handed to step 1 of the search.

The overview is the answer to QUESTION, written at the level of a requirements
document: what the repository is for, what each of its features takes in and puts out,
the steps the work goes through, how it is driven from outside and what it supports. It
names no files and no definitions. One answering agent answers it (answer.run_answer()
with CONTENT_RULE), and the answer is kept in OVERVIEW_FILE beside the index.

create() makes it when it is missing or not current. load() hands a current one to
search step 1 (query.OVERVIEW_TEXT) when is_on(); otherwise step 1 is handed nothing. An
overview is current when it answers QUESTION as it is worded now and the files that both
the index and the working tree hold are the ones they held when it was made. A change to
the files' contents leaves it current. load() also takes as current one that answers
QUESTION as it is worded now and was made over the file digest RUN_DIGEST_ENV holds (the
state a run started from), whatever files have been added or removed since.

OVERVIEW_FILE:
    {"schema": 3, "commit": the commit checked out when it was made,
     "file_digest": _file_digest() then, "question": QUESTION then,
     "answer": the answer}

This module imports answer.py only inside create().
"""

import hashlib
import json
import logging
import os
import sqlite3

from codetwine_loom import config, gitcmd, turnlog
from codetwine_loom.index import store
from codetwine_loom.paths import knowledge_to_repo
from codetwine_loom.search import query
from codetwine_loom.search import scope as scope_mod
from codetwine_loom.tmpfile import atomic_write, read_json

logger = logging.getLogger(__name__)

# The file the answer is kept in, inside <output_dir>/<project name>/
OVERVIEW_FILE = "overview.json"

# The most bytes of OVERVIEW_FILE that are read
RECORD_MAX_BYTE = 1024 * 1024

# The shape of OVERVIEW_FILE. A file of another shape is read as no overview
_SCHEMA = 3

# The question the answering agent is asked
QUESTION = (
    "Write this repository's requirements, as a requirements document would: what it "
    "is for and who uses it; each feature it offers its users, with what the feature "
    "takes in and what it puts out; the steps the work goes through from input to "
    "output; how it is used from outside, saying only the kinds - a command line, "
    "settings, files it reads and writes - and not their names; and what it supports "
    "and what it does not. Write it in English.")

# The line that replaces the answering agent's rule of naming files and definitions
CONTENT_RULE = """- Describe the repository as somebody who uses it sees it. Do not name files, directories,
  functions, classes, variables, command-line options or settings, and do not give
  numbers such as limits or defaults: say what the repository does, not where or with
  which values it does it."""

# The environment variable that holds the file digest of the overview made for the state
# a run started from
RUN_DIGEST_ENV = "CTLOOM_OVERVIEW_DIGEST"

# create()'s status: the answer was written, the one there is used as it is, or no
# answer came back and the file is left as it was
MAKE_STATUS = "made"
KEEP_STATUS = "kept"
FAIL_STATUS = "no answer"


def overview_path(out_dir: str, name: str) -> str:
    """Return where OVERVIEW_FILE is.

    Args:
        out_dir: The output directory.
        name: The repository's directory name.

    Returns:
        <out_dir>/<name>/OVERVIEW_FILE.
    """
    return os.path.join(out_dir, name, OVERVIEW_FILE)


def read_record(out_dir: str, name: str) -> dict:
    """Read OVERVIEW_FILE.

    Args:
        out_dir: The output directory.
        name: The repository's directory name.

    Returns:
        Its contents, or {} when there is none, it cannot be read, it holds more than
        RECORD_MAX_BYTE bytes or it is of another shape.
    """
    path = overview_path(out_dir, name)
    try:
        record_dict = read_json(path, RECORD_MAX_BYTE)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("the overview at %s cannot be read, so it is not used: %s", path, e)
        return {}
    if not isinstance(record_dict, dict) or record_dict.get("schema") != _SCHEMA \
            or not isinstance(record_dict.get("answer"), str):
        logger.warning("the overview at %s is of another shape, so it is not used", path)
        return {}
    return record_dict


def is_on() -> bool:
    """Say whether search step 1 is to be handed the overview.

    Returns:
        True when OVERVIEW and QUERY_EXPAND are both on.
    """
    return config.OVERVIEW and config.QUERY_EXPAND


def load(project_dir: str, out_dir: str, name: str) -> None:
    """Hand the overview to step 1 of the search, when is_on() and it is current.

    Current means it has an answer to QUESTION as it is worded now, made over the files
    that both the loaded index and the working tree hold now, or over the files the file
    digest in RUN_DIGEST_ENV stands for.

    Args:
        project_dir: The repository, as an absolute path.
        out_dir: The output directory.
        name: The repository's directory name.
    """
    query.OVERVIEW_TEXT = ""
    if not is_on():
        return
    record_dict = read_record(out_dir, name)
    if not record_dict:
        logger.info("there is no overview in %s, so search step 1 "
                    "runs without one", out_dir)
        return
    run_digest = os.environ.get(RUN_DIGEST_ENV, "")
    if not _is_current(record_dict, _file_digest(project_dir)) \
            and not (run_digest and _is_current(record_dict, run_digest)):
        logger.warning("the overview in %s was made over other files than the index "
                       "and the working tree hold now, and not for the state this run "
                       "started from, so search step 1 runs without it",
                       overview_path(out_dir, name))
        return
    query.OVERVIEW_TEXT = record_dict["answer"]
    logger.info("the overview made at %s is handed to search step 1 (%d characters)",
                gitcmd.short(str(record_dict.get("commit", ""))) or "?",
                len(query.OVERVIEW_TEXT))


# == Making the overview ======================================

def _file_digest(project_dir: str) -> str:
    """Return a digest of the files that both the loaded index and the working tree hold.

    The working tree's files are the ones git carries or would carry
    (gitcmd.work_tree_file_list): a file not yet committed counts.

    Args:
        project_dir: The repository.

    Returns:
        The sha256 hex digest of their knowledge paths, sorted and one per line.
    """
    return _digest_of(project_dir, scope_mod.INDEX.file_path_list(),
                      scope_mod.PROJECT_NAME)


def _digest_of(project_dir: str, index_path_list: list[str], name: str) -> str:
    """Return a digest of the files that both an index and the working tree hold.

    Args:
        project_dir: The repository.
        index_path_list: The knowledge paths of the index's files.
        name: The repository's directory name.

    Returns:
        The sha256 hex digest of the paths the working tree holds too, sorted and one
        per line.
    """
    tree_set = set(gitcmd.work_tree_file_list(project_dir) or [])
    path_list = sorted(knowledge_path for knowledge_path in index_path_list
                       if knowledge_to_repo(knowledge_path, name) in tree_set)
    return hashlib.sha256("\n".join(path_list).encode("utf-8")).hexdigest()


def index_digest(project_dir: str, out_dir: str, name: str) -> str:
    """Return the file digest over the index in an output directory, read from disk.

    Args:
        project_dir: The repository, as an absolute path.
        out_dir: The output directory.
        name: The repository's directory name.

    Returns:
        The digest, or "" when there is no index there or it cannot be read.
    """
    try:
        index = store.load(store.index_db_path(out_dir, name))
    except (FileNotFoundError, sqlite3.DatabaseError) as e:
        logger.info("no index to make the overview over in %s: %s", out_dir, e)
        return ""
    try:
        index_path_list = index.file_path_list()
    finally:
        index.close()
    return _digest_of(project_dir, index_path_list, name)


def is_current(out_dir: str, name: str, digest: str) -> bool:
    """Say whether the overview in an output directory is current for a file digest.

    Args:
        out_dir: The output directory.
        name: The repository's directory name.
        digest: A file digest, as _file_digest() makes it.

    Returns:
        What _is_current() says of the record there.
    """
    return _is_current(read_record(out_dir, name), digest)


def answer_of(out_dir: str, name: str) -> str:
    """Return the answer of the overview in an output directory.

    Args:
        out_dir: The output directory.
        name: The repository's directory name.

    Returns:
        The answer, or "" when there is no overview there or it answers another
        question than QUESTION as it is worded now.
    """
    record_dict = read_record(out_dir, name)
    return record_dict["answer"] if record_dict.get("question") == QUESTION else ""


def _is_current(record_dict: dict, digest: str) -> bool:
    """Say whether an overview answers QUESTION over the files a file digest stands for.

    Args:
        record_dict: The overview, as read_record() returns it.
        digest: A file digest, as _file_digest() makes it.

    Returns:
        True when it has an answer to QUESTION as it is worded now, made over the files
        digest stands for.
    """
    return bool(record_dict.get("answer")) and record_dict.get("question") == QUESTION \
        and record_dict.get("file_digest") == digest


def write_record(path: str, project_dir: str, digest: str, answer_text: str) -> None:
    """Write OVERVIEW_FILE in one atomic move, making its directory when needed.

    Args:
        path: overview_path()'s result.
        project_dir: The repository, whose checked-out commit is recorded.
        digest: _file_digest() over the files the answer was made from.
        answer_text: The answer to QUESTION.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with atomic_write(path) as tmp_path:
        with open(tmp_path, "w", encoding="utf-8", errors="backslashreplace") as fp:
            json.dump({"schema": _SCHEMA,
                       "commit": gitcmd.head_commit(project_dir),
                       "file_digest": digest,
                       "question": QUESTION,
                       "answer": answer_text}, fp,
                      ensure_ascii=False, indent=2)


def create(project_dir: str, out_dir: str, name: str) -> dict:
    """Answer QUESTION and write OVERVIEW_FILE, unless the overview there is current.

    The index has to be loaded (scope.load_index()) and the reading tools set up
    (source.set_project_data()). An answer that does not come back leaves the file as it
    was. It ends with load().

    The turn log is opened (turnlog.open_run()) around the making alone, which works
    from a directory of its own: an overview that is current opens none.

    Args:
        project_dir: The repository, as an absolute path.
        out_dir: The output directory.
        name: The repository's directory name.

    Returns:
        {"path": OVERVIEW_FILE's path, "status": MAKE_STATUS, KEEP_STATUS or
        FAIL_STATUS}.

    Raises:
        SystemExit: With status 1, when the making raised (turnlog.open_run()).
    """
    from codetwine_loom import answer as answer_mod

    path = overview_path(out_dir, name)
    record_abs = os.path.abspath(path)
    digest = _file_digest(project_dir)
    status = KEEP_STATUS
    if not is_current(out_dir, name, digest):
        with turnlog.open_run():
            # Step 1 of the search is handed no overview while the question is answered
            query.OVERVIEW_TEXT = ""
            logger.info("the overview asks: %s", QUESTION)
            answer_dict = answer_mod.run_answer(QUESTION, CONTENT_RULE)
            status = FAIL_STATUS
            if answer_dict["ok"]:
                status = MAKE_STATUS
                write_record(record_abs, project_dir, digest, answer_dict["answer"])
    logger.info("the overview at %s: %s", path, status)
    load(project_dir, out_dir, name)
    return {"path": path, "status": status}
