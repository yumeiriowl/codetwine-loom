"""Step 0: prepare the knowledge database project_knowledge.sqlite.

ensure_knowledge() uses a complete project_knowledge.sqlite in output_dir/project_name/
as it is unless rebuild is on. Otherwise codetwine is called to create one, except when
the design documents are not asked for and an analysis there can be used as it is for
the state of the target repository's working tree (_describe_state()). The design
documents are made only when they are asked for (with_doc).

The analysis is read one file at a time through Knowledge.

Beside the database, what the analysis describes is recorded in ANALYSIS_COMMIT_FILE
whenever codetwine runs here.
"""

import asyncio
import contextlib
import errno
import fnmatch
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
import sqlite3
import stat
import sys
from collections.abc import Iterator

import codetwine.extractors.dependency_graph as codetwine_walk
import codetwine.pipeline as codetwine_pipeline
from codetwine import doc_creator as codetwine_doc
from codetwine.config import settings as codetwine_setting
from codetwine import knowledge_db
from codetwine.llm.client import LLMClient
from codetwine.utils import file_utils as codetwine_file

from codetwine_loom import (SettingError, config, filelock, gitcmd, progress,
                            providercall, tmpfile)
from codetwine_loom.index import store
from codetwine_loom.languages import LANG_NAME_DICT, ext_of
from codetwine_loom.textcut import command_line, name_list_text

logger = logging.getLogger(__name__)

# The analysis file inside <output_dir>/<project name>/
KNOWLEDGE_FILE = "project_knowledge.sqlite"

# The record beside it: {"stamp": Knowledge.stamp(), "commit": the commit whose tree the
# analysis read, "state": the value of what it read (analysis_state()), "exclude": the
# extra names it was given, "exclude_all": every name that was in force, "encoding": the
# SOURCE_ENCODING the files were read with, "root": the commits the repository's history
# starts from, "codetwine": the codetwine that made it (codetwine_id())}
ANALYSIS_COMMIT_FILE = "analysed_commit.json"

# The most bytes read of ANALYSIS_COMMIT_FILE
RECORD_MAX_BYTE = 1024 * 1024

# The directory the agents' own copies of an output directory and this tool's own files
# of each repository are kept under, directly in it, beside the directories named after
# the repositories analysed there
AGENT_DIR = "agents"

# The names this tool uses directly under an output directory
_OUTPUT_OWN_NAME_TUPLE = (AGENT_DIR,)

# codetwine's own list of names to keep out, as it stood when this module was imported
_BASE_EXCLUDE_TUPLE = tuple(codetwine_walk.EXCLUDE_PATTERNS)

# The members of a file entry's file_dependencies
_DEPENDENCY_MEMBER_TUPLE = ("definitions", "callee_usages", "same_file_usages",
                            "caller_usages")

# The members whose entries name the definition they lead to ("target_name",
# "target_start_line")
_TARGET_MEMBER_TUPLE = ("callee_usages", "same_file_usages")


class Knowledge:
    """One codetwine analysis, read one file at a time."""

    def __init__(self, connection: sqlite3.Connection, output_dir: str | None = None) -> None:
        """Wrap an open database connection.

        Args:
            connection: A read-only connection to a knowledge database.
            output_dir: The output directory codetwine wrote the database and the
                design documents (doc.json) under, or None to read the design documents
                from the database only.
        """
        self._connection = connection
        self._output_dir = output_dir

    @classmethod
    def open(cls, path: str) -> "Knowledge":
        """Open a knowledge database for reading.

        Args:
            path: The path to project_knowledge.sqlite.

        Returns:
            A Knowledge the caller closes when it is done with it.

        Raises:
            FileNotFoundError: When there is no database at that path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"no knowledge database: {path}")
        return cls(store.open_read_only(path), os.path.dirname(os.path.dirname(path)))

    def file_count(self) -> int:
        """Return how many files the analysis covers.

        Returns:
            The number of analysed files.
        """
        return self._connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def iter_file(self) -> Iterator[dict]:
        """Yield every analysed file, one at a time.

        Each call starts a new pass; the same Knowledge can be walked any number of
        times.

        Yields:
            {"file": knowledge path, "file_dependencies": {"definitions",
            "callee_usages", "same_file_usages", "caller_usages"}, "doc": the design
            document}. Every file_dependencies member is present, empty where the
            analysis has none. "doc" is the design document the database holds, else
            the doc.json codetwine left in the file's output directory (one written from
            an earlier content of the file; its "source_hash" says which), and is absent
            when there is neither.

        Raises:
            RuntimeError: When a usage of a file does not name the definition it leads
                to, which is an analysis an older codetwine made.
        """
        for entry in knowledge_db.iter_files(self._connection):
            dependency_dict = entry.setdefault("file_dependencies", {})
            for member in _DEPENDENCY_MEMBER_TUPLE:
                dependency_dict.setdefault(member, [])
            if "doc" not in entry and self._output_dir is not None:
                doc = self._left_doc(entry["file"])
                if doc is not None:
                    entry["doc"] = doc
            for member in _TARGET_MEMBER_TUPLE:
                if any("target_name" not in usage for usage in dependency_dict[member]):
                    raise RuntimeError(
                        f"the analysis is out of date: the {member} of {entry['file']} "
                        f"do not name the definition they lead to. Run `codetwine-loom "
                        f"build` over the repository to make the analysis again")
            yield entry

    def _left_doc(self, knowledge_path: str) -> dict | None:
        """Return the doc.json codetwine left in a file's output directory.

        Args:
            knowledge_path: The "file" of the file's entry ("<project>/<copy path>").

        Returns:
            The design document, or None when doc.json is not a regular file (a
            symbolic link included) under the output directory, cannot be read as a
            JSON object, or does not have the members a design document has (str
            "summary" and "source_hash", "sections" of {"id": str, "content": str}).
        """
        doc_dir = os.path.join(self._output_dir, os.path.dirname(knowledge_path))
        doc_path = os.path.join(doc_dir, codetwine_file.DOC_JSON_NAME)
        real_output_dir = os.path.realpath(self._output_dir)
        if os.path.commonpath([real_output_dir, os.path.realpath(doc_dir)]) \
                != real_output_dir:
            return None
        try:
            if not stat.S_ISREG(os.lstat(doc_path).st_mode):
                return None
        except OSError:
            return None
        doc = codetwine_file.load_doc(doc_dir)
        if doc is None or not isinstance(doc.get("summary", ""), str) \
                or not isinstance(doc.get("source_hash", ""), str) \
                or not isinstance(doc.get("sections", []), list) \
                or not all(isinstance(section, dict)
                           and isinstance(section.get("id"), str)
                           and isinstance(section.get("content"), str)
                           for section in doc.get("sections", [])):
            return None
        return doc

    def names_target(self) -> bool:
        """Say whether the usages of the analysis name the definition they lead to.

        Returns:
            False when the first usage found carries no "target_name", which is an
            analysis an older codetwine made. True otherwise, also for an analysis
            without a usage.
        """
        for entry in knowledge_db.iter_files(self._connection):
            dependency_dict = entry.get("file_dependencies") or {}
            for member in _TARGET_MEMBER_TUPLE:
                for usage in dependency_dict.get(member) or ():
                    return "target_name" in usage
        return True

    def stamp(self) -> str:
        """Return the value identifying the analysis run this database holds.

        Returns:
            "<schema version>:<creation time>". A member the database does not carry
            comes back empty.
        """
        row_iter = self._connection.execute(
            "SELECT key, value FROM meta WHERE key IN ('schema_version', 'created_at')")
        meta_dict = {row["key"]: row["value"] for row in row_iter}
        return f'{meta_dict.get("schema_version", "")}:{meta_dict.get("created_at", "")}'

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()


def project_name_of(project: str) -> str:
    """Return the directory name a repository's analysis is filed under.

    The path is made absolute first: ".", ".." and a trailing separator all name the
    directory they stand for.

    Examples:
        ("/home/me/myapp") -> "myapp"
        (".") (in /home/me/myapp) -> "myapp"

    Args:
        project: The target repository path, as it was passed.

    Returns:
        The basename of its absolute path.

    Raises:
        ValueError: When the absolute path has no basename, which is the filesystem
            root.
    """
    name = os.path.basename(os.path.abspath(project))
    if not name:
        raise ValueError(f"the target repository has no directory name: {project}. "
                         f"Pass a repository rather than the filesystem root")
    return name


def _effective_exclude(exclude_list: list[str] | None) -> list[str]:
    """Return every name codetwine keeps out when it is given these extra ones.

    Args:
        exclude_list: The extra directory and file names, or None for none. A name
            given more than once is kept once.

    Returns:
        codetwine's own names followed by the extra ones.
    """
    base_list = list(_BASE_EXCLUDE_TUPLE)
    return base_list + [name for name in dict.fromkeys(exclude_list or [])
                        if name not in base_list]


# What an entry of the names to keep out holds when it is a path from the repository's
# root rather than a name, and what such a path of one part starts with
_PATH_MARK = "/"
_ROOT_PREFIX = "./"


def exclude_entry(text: str) -> str:
    """Return one entry of the names to keep out, in the form it is kept in.

    An entry holding no "/" is a directory or file name, matched wherever it is. One
    holding a "/" is a path from the repository's root, which keeps out that file or
    everything under that directory. A "/" at the end is dropped, and "./" at the
    head marks a path of one part.

    Examples:
        "vendor"          -> "vendor"            (every directory or file named vendor)
        "vendor/"         -> "vendor"
        "./vendor"        -> "./vendor"          (vendor at the root alone)
        "apps/old/"       -> "apps/old"
        "./apps//old"     -> "apps/old"
        "apps/*/gen"      -> "apps/*/gen"

    Args:
        text: The entry as it was given, with "/" or the system's own separator.

    Returns:
        The entry in the form above.

    Raises:
        ValueError: When the entry is empty, starts at the file system's root, or
            holds a ".." part.
    """
    path = text.replace(os.sep, _PATH_MARK)
    part_list = [part for part in path.split(_PATH_MARK) if part not in ("", ".")]
    if not part_list or path.startswith(_PATH_MARK) or ".." in part_list:
        raise ValueError(f"{text!r} is not a name or a path inside the repository")
    if len(part_list) == 1:
        return f"{_ROOT_PREFIX}{part_list[0]}" if path.startswith(_ROOT_PREFIX) \
            else part_list[0]
    return _PATH_MARK.join(part_list)


def _is_path_entry(entry: str) -> bool:
    """Say whether an entry of the names to keep out is a path from the root.

    Args:
        entry: The entry, as exclude_entry() returns it.

    Returns:
        True when the entry holds _PATH_MARK.
    """
    return _PATH_MARK in entry


def is_out_by_path(file_rel: str, entry_list: list[str]) -> bool:
    """Say whether a file is kept out by one of the path entries.

    Args:
        file_rel: The file's path relative to the repository, with "/" between parts.
        entry_list: The path entries, as exclude_entry() returns them. Each is matched
            as an fnmatch pattern against the whole path, and against every path under
            it.

    Returns:
        True when the file is the path an entry names, or is under it.
    """
    for entry in entry_list:
        pattern = entry.removeprefix(_ROOT_PREFIX)
        if fnmatch.fnmatch(file_rel, pattern) \
                or fnmatch.fnmatch(file_rel, f"{pattern}{_PATH_MARK}*"):
            return True
    return False


def _walk_file_list(project_dir: str) -> list[str]:
    """Return every file under a directory, relative to it, without following links.

    Args:
        project_dir: The directory to walk.

    Returns:
        The paths with "/" between their parts, in os.walk's order.
    """
    file_list = []
    for dir_path, _, file_name_list in os.walk(project_dir):
        for file_name in file_name_list:
            file_rel = os.path.relpath(os.path.join(dir_path, file_name), project_dir)
            file_list.append(file_rel.replace(os.sep, _PATH_MARK))
    return file_list


def _is_out_by_name(file_rel: str, name_list: list[str]) -> bool:
    """Say whether a file is kept out by one of the names, as codetwine keeps it out.

    Args:
        file_rel: The file's path relative to the repository, with "/" between parts.
        name_list: The entries that are names (no _PATH_MARK), each an fnmatch pattern.

    Returns:
        True when a part of the path matches one of them.
    """
    return any(fnmatch.fnmatch(part, name) for part in file_rel.split(_PATH_MARK)
               for name in name_list)


# What _file_form() says stands at a path of the files listed for codetwine: a regular
# file codetwine reads, something that cannot be read as a file, and nothing (no file,
# a directory, or a file reached through a link)
_READ_FORM = "read"
_UNREAD_FORM = "unread"
_NO_FORM = "none"

# The errors of lstat that say nothing stands at a path: none there, a part of it that
# is not a directory, and links that go round
_NO_FILE_ERRNO_TUPLE = (errno.ENOENT, errno.ENOTDIR, errno.ELOOP)


def _file_form(root: str, file_rel: str, own_dir_dict: dict[str, bool]) -> str:
    """Say what stands at a path of the files listed for codetwine.

    Args:
        root: The repository, resolved through realpath.
        file_rel: The path, relative to root.
        own_dir_dict: {directory relative to root: whether it is reached through no
            link}, read and written here so that each directory is resolved once.

    Returns:
        _READ_FORM for a regular file this process can open for reading and reaches
        through no link. _UNREAD_FORM for a named pipe, a device or a socket, a regular
        file it cannot open for reading and a path lstat refuses for another reason
        (one under a directory that cannot be searched). _NO_FORM for a path lstat
        finds nothing at (_NO_FILE_ERRNO_TUPLE), a directory, a symbolic link and a
        path under one.
    """
    full_path = os.path.join(root, file_rel)
    dir_rel = os.path.dirname(file_rel)
    try:
        mode = os.lstat(full_path).st_mode
        if dir_rel not in own_dir_dict:
            dir_path = os.path.join(root, dir_rel) if dir_rel else root
            own_dir_dict[dir_rel] = os.path.realpath(dir_path) == dir_path
        if stat.S_ISDIR(mode) or stat.S_ISLNK(mode) or not own_dir_dict[dir_rel]:
            return _NO_FORM
    except OSError as e:
        return _NO_FORM if e.errno in _NO_FILE_ERRNO_TUPLE else _UNREAD_FORM
    except RecursionError:
        return _NO_FORM
    if not stat.S_ISREG(mode) or not os.access(full_path, os.R_OK):
        return _UNREAD_FORM
    return _READ_FORM


def analysis_file_list(project_dir: str, exclude_list: list[str] | None,
                       walk: bool = False) -> tuple[list[str] | None, list[str], int]:
    """Return the files codetwine is handed for a repository, and what is kept out.

    The files are the ones git tracks or does not ignore (gitcmd.work_tree_file_list()).
    Taken off them are the ones the paths of exclude_list keep out (is_out_by_path), the
    ones its names and codetwine's own names keep out as codetwine keeps them out
    (_is_out_by_name), and every path where no regular file this process can read
    stands (_file_form). A project_dir that is not a git repository is walked here when
    exclude_list holds a path or walk is True (_walk_file_list), and is left to
    codetwine otherwise.

    Args:
        project_dir: The path to the target repository, already normalised.
        exclude_list: The extra names and paths to keep out, each in the form
            exclude_entry() returns, or None for none.
        walk: Whether a project_dir that is not a git repository is walked here whatever
            exclude_list holds.

    Returns:
        (the repository-relative paths of the files, in git's order, or None for a
        project_dir codetwine walks itself; the paths taken off where something stands
        that cannot be read as a file (_UNREAD_FORM), in the same order; how many paths
        the paths of exclude_list took off).
    """
    entry_list = _effective_exclude(exclude_list)
    path_entry_list = [entry for entry in entry_list if _is_path_entry(entry)]
    name_entry_list = [entry for entry in entry_list if not _is_path_entry(entry)]
    file_list = gitcmd.work_tree_file_list(project_dir)
    if file_list is None:
        logger.info("%s is not a git repository, so %s walks the directory "
                    "itself", project_dir,
                    "this tool" if path_entry_list or walk else "codetwine")
        if not path_entry_list and not walk:
            return None, [], 0
        file_list = _walk_file_list(project_dir)
    keep_list = [file_rel for file_rel in file_list
                 if not is_out_by_path(file_rel, path_entry_list)]
    out_count = len(file_list) - len(keep_list)
    root = os.path.realpath(project_dir)
    read_list, unread_list = [], []
    own_dir_dict: dict[str, bool] = {}
    for file_rel in keep_list:
        if _is_out_by_name(file_rel, name_entry_list):
            continue
        form = _file_form(root, file_rel, own_dir_dict)
        if form == _READ_FORM:
            read_list.append(file_rel)
        elif form == _UNREAD_FORM:
            unread_list.append(file_rel)
    return read_list, unread_list, out_count


def _state_of(blob_dict: dict[str, str] | None) -> str:
    """Return the value that stands for some files and their contents.

    Args:
        blob_dict: {repository-relative path: blob id}, or None when they could not be
            told.

    Returns:
        The sha256 hex digest of every path and its blob id, in the order of the paths,
        or "" for None.
    """
    if blob_dict is None:
        return ""
    state_hash = hashlib.sha256()
    for path in sorted(blob_dict):
        state_hash.update(os.fsencode(path) + b"\0" + blob_dict[path].encode() + b"\0")
    return state_hash.hexdigest()


def analysis_state(project_dir: str, exclude_list: list[str] | None) -> str:
    """Return the value of what an analysis of the working tree reads now.

    It is made of the files codetwine is handed (analysis_file_list()) and the blob git
    makes of each of them on disk (gitcmd.disk_blob_dict()). A clone checked out at a
    commit gives the value commit_state() gives for that commit, and a file that cannot
    be read, is not on disk or is kept out does not change it.

    Args:
        project_dir: The target repository, already normalised.
        exclude_list: The extra names and paths the analysis keeps out.

    Returns:
        The value, or "" for a directory codetwine walks itself and when the files
        cannot be told.
    """
    file_list = analysis_file_list(project_dir, exclude_list)[0]
    if file_list is None:
        return ""
    return _state_of(gitcmd.disk_blob_dict(project_dir, file_list))


def commit_state(project_dir: str, commit: str, exclude_list: list[str] | None) -> str:
    """Return the value analysis_state() gives for a tree holding what a commit holds.

    It is made of the regular files of the commit's tree (gitcmd.tree_blob_dict()) that
    exclude_list and codetwine's own names do not keep out (is_out_by_path,
    _is_out_by_name).

    Args:
        project_dir: The repository holding the commit.
        commit: The commit. "" gives "".
        exclude_list: The extra names and paths the analysis keeps out.

    Returns:
        The value, or "" when the commit's tree cannot be read.
    """
    if not commit:
        return ""
    blob_dict = gitcmd.tree_blob_dict(project_dir, commit)
    if blob_dict is None:
        return ""
    entry_list = _effective_exclude(exclude_list)
    path_entry_list = [entry for entry in entry_list if _is_path_entry(entry)]
    name_entry_list = [entry for entry in entry_list if not _is_path_entry(entry)]
    return _state_of({path: blob for path, blob in blob_dict.items()
                      if not is_out_by_path(path, path_entry_list)
                      and not _is_out_by_name(path, name_entry_list)})


def same_exclude(record_dict: dict, exclude_list: list[str] | None) -> bool:
    """Say whether an analysis keeps out what this run asks to keep out.

    The names are compared as sets, whatever order they were given in. Where the record
    holds every name in force (codetwine's own list included) those are compared; a
    record without them is compared on the extra names alone, and one that says nothing
    about either answers True for whatever the run asks.

    Args:
        record_dict: The record beside the analysis (analysis_record()), or {} when
            there is none.
        exclude_list: The extra names this run keeps out.

    Returns:
        True when the two keep out the same names.
    """
    record_all_list = record_dict.get("exclude_all")
    if isinstance(record_all_list, list):
        return set(record_all_list) == set(_effective_exclude(exclude_list))
    record_list = record_dict.get("exclude")
    if not isinstance(record_list, list):
        return True
    return set(record_list) == set(exclude_list or [])


def same_encoding(record_dict: dict) -> bool:
    """Say whether an analysis read its files with the encodings this run reads with.

    Args:
        record_dict: The record beside the analysis (analysis_record()), or {} when
            there is none.

    Returns:
        True when the record's encodings are config.SOURCE_ENCODING, in the same
        order, or the record says nothing about them.
    """
    record_list = record_dict.get("encoding")
    if not isinstance(record_list, list):
        return True
    return record_list == list(config.SOURCE_ENCODING)


def codetwine_id() -> str:
    """Return the value that tells the codetwine installed apart from another.

    Examples:
        a codetwine installed from git -> "0.4.5@917e5bd5452da1bd4a4ce00c999bcbbd9570fdf9"
        one installed from a directory -> "0.4.5"

    Returns:
        The version of the codetwine distribution found first on the import path,
        followed by "@" and the commit its install record (direct_url.json) names when
        it was installed from a version control system, or "" when no codetwine
        distribution is found.
    """
    try:
        distribution = importlib.metadata.distribution("codetwine")
    except importlib.metadata.PackageNotFoundError:
        return ""
    version = distribution.version or ""
    try:
        url_dict = json.loads(distribution.read_text("direct_url.json") or "{}")
    except (OSError, ValueError):
        url_dict = {}
    vcs_dict = url_dict.get("vcs_info") if isinstance(url_dict, dict) else None
    commit = vcs_dict.get("commit_id") if isinstance(vcs_dict, dict) else None
    return f"{version}@{commit}" if isinstance(commit, str) and commit else version


def same_codetwine(record_dict: dict) -> bool:
    """Say whether an analysis was made by the codetwine installed now.

    Args:
        record_dict: The record beside the analysis (analysis_record()), or {} when
            there is none.

    Returns:
        True when the record's "codetwine" is codetwine_id(). A record without it
        answers False.
    """
    return record_dict.get("codetwine") == codetwine_id()


def is_complete(path: str) -> bool:
    """Say whether the analysis at a path is one that can be used as it is.

    A database that has its tables but not the meta rows the stamp is read from is not
    complete.

    Args:
        path: The path to project_knowledge.sqlite.

    Returns:
        True when the file is there, opens, and both members of its stamp have a value.
    """
    if not os.path.exists(path):
        return False
    try:
        knowledge = Knowledge.open(path)
    except (OSError, sqlite3.Error) as e:
        logger.info("the analysis at %s cannot be opened (%s), so it is made again",
                    path, e)
        return False
    try:
        part_list = knowledge.stamp().split(":", 1)
    except sqlite3.Error as e:
        logger.info("the analysis at %s cannot be read (%s), so it is made again",
                    path, e)
        return False
    finally:
        knowledge.close()
    if len(part_list) == 2 and all(part_list):
        return True
    logger.info("the analysis at %s says nothing about the run that made it, so it is "
                "made again", path)
    return False


# The types a codetwine setting has to have to be handed over again
_CODETWINE_ENV_TYPE_TUPLE = (str, int, float, bool, list)


def read_codetwine_setting_again() -> list[str]:
    """Hand codetwine the settings it would read from the environment as it is now.

    codetwine's settings file is run once more here, with config.NO_ENV_FILE_ENV set,
    and every public value that came out different is put in the settings module and in
    each codetwine module still holding the value read at the import. A module this tool
    has given another value keeps it.

    Returns:
        The names of the settings that changed, sorted.

    Raises:
        SettingError: Naming the setting, when codetwine cannot read a value.
    """
    spec = importlib.util.find_spec(codetwine_setting.__name__)
    now_module = importlib.util.module_from_spec(spec)
    try:
        with config.env_file_off():
            spec.loader.exec_module(now_module)
    except ValueError as e:
        raise SettingError(f"codetwine could not read its settings: {e}") from None
    name_list = []
    for name, value in vars(now_module).items():
        before = getattr(codetwine_setting, name, None)
        if not name.isupper() or name.startswith("_") \
                or type(value) not in _CODETWINE_ENV_TYPE_TUPLE \
                or type(before) is not type(value) or before == value:
            continue
        name_list.append(name)
        _put_codetwine_value(name, before, value)
    return sorted(name_list)


def _put_codetwine_value(name: str, before: object, value: object) -> None:
    """Put a value of codetwine's settings in every codetwine module that holds it.

    Args:
        name: The setting's name.
        before: The value the modules hold now; a module holding another one keeps it.
        value: The value to put.
    """
    for module_name, module in list(sys.modules.items()):
        if module_name.split(".")[0] == "codetwine" and module is not None \
                and name in vars(module) and vars(module)[name] == before:
            setattr(module, name, value)


# The file name of the document template at the top of codetwine-loom's repository
_DOC_TEMPLATE_NAME = "doc_template.json"


def doc_template_path() -> str:
    """Return the template the design documents follow, and hand it to codetwine.

    It is the one DOC_TEMPLATE_PATH names when that is set to a value; otherwise
    doc_template.json at the top of the repository this tool was installed from
    (image.source_dir), when that file is there; otherwise codetwine's own default.

    Returns:
        The template's path.
    """
    from codetwine_loom import image
    path = codetwine_setting.DOC_TEMPLATE_PATH
    repo_dir = image.source_dir()
    own_path = os.path.join(repo_dir, _DOC_TEMPLATE_NAME) if repo_dir else ""
    if not os.environ.get("DOC_TEMPLATE_PATH") and own_path and os.path.isfile(own_path) \
            and path != own_path:
        _put_codetwine_value("DOC_TEMPLATE_PATH", path, own_path)
        path = own_path
    return path


def doc_call_count() -> int:
    """Return how many model calls one file's design document takes.

    One call for each section the template DOC_TEMPLATE_PATH names, plus the one that
    summarises the file. A file too long for the context window adds more.

    Returns:
        The number of calls, or 0 when the template cannot be read.
    """
    path = doc_template_path()
    try:
        with open(path, encoding="utf-8") as fp:
            template_dict = json.load(fp)
        return len(template_dict["sections"]) + 1
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
        logger.info("the document template at %s cannot be read (%s), so how many "
                    "calls it takes is not named", path, e)
        return 0


def check_doc_setting() -> None:
    """Say what the design documents need, before codetwine is asked for them.

    The template (doc_template_path) is looked for here, before codetwine runs, and
    codetwine's check of its code summary settings is made here
    (check_code_summary_setting).

    Raises:
        RuntimeError: When DOC_TEMPLATE_PATH names no file, or codetwine refuses a
            code summary setting.
    """
    path = doc_template_path()
    if not os.path.isfile(path):
        raise RuntimeError(f"the design documents follow the template DOC_TEMPLATE_PATH "
                           f"names, and there is no file at {path}. Point it at a "
                           f"document template, or leave the documents out")
    try:
        codetwine_doc.check_code_summary_setting()
    except ValueError as e:
        raise RuntimeError(str(e)) from None


def report_run(run_dict: dict, with_doc: bool) -> None:
    """Log what one codetwine run could not do, and refuse a run that wrote no document.

    Args:
        run_dict: What process_all_files() returned: {"file_count",
            "dependency_fail_list", "doc_count", "doc_fail_list",
            "detected_encoding_dict"}.
        with_doc: Whether the design documents were asked for.

    Raises:
        RuntimeError: When the documents were asked for and every one of them failed.
    """
    dependency_fail_list = run_dict["dependency_fail_list"]
    if dependency_fail_list:
        logger.warning("codetwine could not extract %d of %d file(s), which the analysis "
                       "carries no definitions of: %s", len(dependency_fail_list),
                       run_dict["file_count"], ", ".join(dependency_fail_list))
    detected_encoding_dict = run_dict["detected_encoding_dict"]
    if detected_encoding_dict:
        logger.warning(
            "codetwine guessed the encoding of %d file(s), which are read and written "
            "back in it: %s. Set SOURCE_ENCODING to the encoding they are stored in",
            len(detected_encoding_dict),
            ", ".join(f"{path} ({encoding or 'none, read as UTF-8 with replacement'})"
                      for path, encoding in detected_encoding_dict.items()))
    if not with_doc:
        return
    doc_fail_list = run_dict["doc_fail_list"]
    logger.info("docs: %d ok, %d failed", run_dict["doc_count"], len(doc_fail_list))
    if not doc_fail_list:
        return
    if not run_dict["doc_count"]:
        raise RuntimeError(
            f"every one of the {len(doc_fail_list)} design document(s) failed. The "
            f"analysis is written without them. Check LLM_API_KEY and DOC_CREATE_MODEL "
            f"({config.DOC_CREATE_MODEL}), then run it again")
    logger.warning("no complete design document for: %s", ", ".join(doc_fail_list))


def run_codetwine(project_dir: str, output_dir: str, with_doc: bool = True,
                  exclude_list: list[str] | None = None) -> None:
    """Run codetwine over the target repository.

    codetwine is handed the files analysis_file_list() gives: the ones git tracks or
    does not ignore, without what exclude_list and codetwine's own names keep out and
    without the paths where no regular file this process can read stands. The log
    names, in one warning, the paths where something stands that cannot be read as a
    file. A project_dir that is not a git repository is walked by codetwine itself, or
    here when exclude_list holds a path (_walk_file_list). The names of exclude_list
    are handed to codetwine as well.

    codetwine regenerates a design document only for a file that changed and for the
    files depending on it, comparing each source against the copy it left in output_dir
    last time. Extraction of definitions and dependencies, line numbers included, runs
    over every file on every call.

    With with_doc False the whole document step is skipped and no LLM is called at all.
    The documents written by an earlier run stay on disk.

    Args:
        project_dir: The path to the target repository, already normalised.
        output_dir: Where codetwine writes its output.
        with_doc: Whether to generate the design documents (the only part that calls the
            LLM).
        exclude_list: Extra directory and file names, and paths from the repository's
            root, to keep out of the analysis, on top of codetwine's own list, each in
            the form exclude_entry() returns. One given more than once is added once.
            None changes nothing.

    Raises:
        RuntimeError: When the documents were asked for and what they need is not set
            (check_doc_setting), when no model call may be made
            (providercall.check_before_call), or when every one of them failed
            (report_run).
    """
    if with_doc:
        check_doc_setting()
        providercall.check_before_call()
    entry_list = _effective_exclude(exclude_list)
    path_entry_list = [entry for entry in entry_list if _is_path_entry(entry)]
    file_list, unread_list, out_count = analysis_file_list(project_dir, exclude_list)
    if path_entry_list and file_list is not None:
        logger.info("%d file(s) under %s are kept out of the analysis", out_count,
                    ", ".join(path_entry_list))
    if unread_list:
        logger.warning("%d path(s) of %s cannot be read as files and are not in the "
                       "analysis: %s", len(unread_list), project_dir,
                       name_list_text(unread_list))
        logger.debug("the paths of %s that cannot be read as files: %s", project_dir,
                     ", ".join(unread_list))
    # Set the three settings in the codetwine modules that read them (pipeline and walk)
    # on each call, EXCLUDE_PATTERNS to the entries of entry_list that are not paths
    codetwine_pipeline.ENABLE_LLM_DOC = with_doc
    codetwine_pipeline.KNOWLEDGE_FORMAT = "sqlite"
    codetwine_walk.EXCLUDE_PATTERNS = [entry for entry in entry_list
                                       if not _is_path_entry(entry)]
    # codetwine's own lines and what it prints are what a run with the documents shows
    # as it goes; a run without them carries only what is a warning or worse, and
    # below DEBUG prints nothing
    with contextlib.ExitStack() as stack:
        if not with_doc:
            logger.info("analysing %s without the design documents (no LLM is called)",
                        project_dir)
            stack.enter_context(progress.quiet("codetwine"))
            if config.AGENT_LOG_LEVEL != "DEBUG":
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        run_dict = asyncio.run(codetwine_pipeline.process_all_files(
            project_dir, output_dir,
            LLMClient(model=config.DOC_CREATE_MODEL, effort=config.DOC_CREATE_EFFORT,
                      **config.credential_dict())
            if with_doc else None,
            max_workers=config.CODETWINE_MAX_WORKER, file_list=file_list))
    report_run(run_dict, with_doc)


def knowledge_db_path(output_dir: str, project_name: str) -> str:
    """Return the path to the knowledge database of one repository in an output directory.

    Args:
        output_dir: Where codetwine writes its output.
        project_name: The directory name the analysis is filed under.

    Returns:
        KNOWLEDGE_FILE joined to output_dir/project_name; relative when output_dir is.
    """
    return os.path.join(output_dir, project_name, KNOWLEDGE_FILE)


def knowledge_db_path_of_index(index_path: str) -> str:
    """Return the path to the knowledge database beside an index.

    Args:
        index_path: The path to code_index.sqlite.

    Returns:
        KNOWLEDGE_FILE joined to the directory of index_path.
    """
    return os.path.join(os.path.dirname(index_path), KNOWLEDGE_FILE)


def analysis_record_path(path: str) -> str:
    """Return the path of the record beside an analysis.

    Args:
        path: The path to project_knowledge.sqlite.

    Returns:
        ANALYSIS_COMMIT_FILE joined to the directory of path.
    """
    return os.path.join(os.path.dirname(path), ANALYSIS_COMMIT_FILE)


def record_analysis_commit(path: str, commit: str, state: str,
                           exclude_list: list[str] | None = None,
                           root_list: list[str] | None = None) -> None:
    """Write beside the analysis what it describes, keyed by the analysis' stamp.

    The record names the codetwine installed now (codetwine_id()) as the one that made
    the analysis. An OSError while the record is written is logged as a warning and not raised.

    Args:
        path: The path to project_knowledge.sqlite. The record goes in its directory.
        commit: The commit whose tree the analysis read (commit_state() of it is
            state): "" when it read another.
        state: The value of what the analysis read (analysis_state()): "" when it
            describes nothing.
        exclude_list: The extra names kept out of the analysis, as run_codetwine() took
            them.
        root_list: The commits the analysed repository's history starts from
            (gitcmd.root_commit_list()). None records none.
    """
    knowledge = Knowledge.open(path)
    try:
        stamp = knowledge.stamp()
    finally:
        knowledge.close()
    record_dict = {"stamp": stamp, "commit": commit, "state": state,
                   "exclude": list(exclude_list or []),
                   "exclude_all": _effective_exclude(exclude_list),
                   "encoding": list(config.SOURCE_ENCODING),
                   "root": list(root_list or []), "codetwine": codetwine_id()}
    record_path = analysis_record_path(path)
    try:
        with tmpfile.atomic_write(record_path) as tmp_path:
            with open(tmp_path, "w", encoding="utf-8") as fp:
                json.dump(record_dict, fp, indent=2)
    except OSError as e:
        logger.warning("the commit of the analysis could not be recorded in %s (%s)",
                       record_path, e)


def exclude_of_analysis(path: str, source_stamp: str,
                        fallback_list: list[str]) -> list[str]:
    """Return what the analysis on disk keeps out.

    Args:
        path: The path to project_knowledge.sqlite.
        source_stamp: The analysis on disk (Knowledge.stamp()).
        fallback_list: What to answer when the record says nothing about it: a record
            written before the exclude list was kept, or no record at all.

    Returns:
        The record's exclude list, or fallback_list.
    """
    record_dict = analysis_record(path, source_stamp)
    exclude_list = record_dict.get("exclude")
    return list(exclude_list) if isinstance(exclude_list, list) else list(fallback_list)


def analysis_record(path: str, source_stamp: str) -> dict:
    """Return the record beside the analysis, when it belongs to that analysis.

    Args:
        path: The path to project_knowledge.sqlite. The record is read from its
            directory.
        source_stamp: The analysis the record has to belong to (Knowledge.stamp()).

    Returns:
        {"stamp", "commit", "state", "exclude", "exclude_all", "encoding", "root",
        "codetwine"} as record_analysis_commit() wrote it (a record written before a member was kept
        does not have it), or {} when there is no record, it cannot be read as a
        regular file of at most RECORD_MAX_BYTE bytes (tmpfile.read_json), or it
        belongs to another analysis.
    """
    record_path = analysis_record_path(path)
    try:
        record_dict = tmpfile.read_json(record_path, RECORD_MAX_BYTE)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("the record of the analysis at %s is not read (%s), so the "
                       "analysis counts as one with no record", record_path, e)
        return {}
    if not isinstance(record_dict, dict) or record_dict.get("stamp") != source_stamp:
        logger.info("the record at %s belongs to another analysis, so what it says "
                    "is not used", record_path)
        return {}
    return record_dict


def analysis_commit(path: str, source_stamp: str) -> str:
    """Return the commit the record beside the analysis says it describes.

    Args:
        path: The path to project_knowledge.sqlite.
        source_stamp: The analysis the answer has to belong to (Knowledge.stamp()).

    Returns:
        The full commit hash, or "" when there is no such record for the analysis or it
        recorded none.
    """
    return str(analysis_record(path, source_stamp).get("commit") or "")


# The answers ask_doc() was given in this process, by the absolute path asked about
_DOC_ANSWER_DICT: dict[str, bool] = {}


def ask_doc(path: str) -> bool:
    """Say on the terminal that the repository is analysed now, and ask about the documents.

    The analysis is made either way; what is asked is whether the design documents are
    made with it. A path asked about once in this process is not asked about again: the
    answer given then comes back.

    Args:
        path: The knowledge database that is missing, as ensure_knowledge() joins it.

    Returns:
        True to generate the design documents as well. With no terminal to ask on
        (progress.at_terminal), when the input ends before an answer, or when the
        answer is no, False.
    """
    key = os.path.abspath(path)
    if key in _DOC_ANSWER_DICT:
        return _DOC_ANSWER_DICT[key]
    if not progress.at_terminal():
        return False
    print(f"There is no analysis of the repository yet: {path}. It is made now.")
    call_count = doc_call_count()
    call_text = (f"{call_count} times per file, one per section of the template plus "
                 f"the summary and more where a file exceeds the context window,"
                 if call_count else
                 "once per section of the template and once more per file,")
    try:
        answer = progress.ask(
            f"Generate the design documents as well? codetwine calls the LLM "
            f"{call_text} which costs money, following the template DOC_TEMPLATE_PATH "
            f"names. Without them the search sees only paths and definition names [y/N] ")
    except EOFError:
        print()
        answer = ""
    _DOC_ANSWER_DICT[key] = answer.strip().lower() in ("y", "yes")
    return _DOC_ANSWER_DICT[key]


# The name of the setting that holds the COBOL dialect
_COBOL_DIALECT_NAME = "COBOL_DIALECT"


def ask_cobol_dialect(index_path: str, project_dir: str = "",
                      exclude_list: list[str] | None = None) -> str:
    """Ask on the terminal which dialect the static check reads COBOL in.

    It is asked once: when the repository holds COBOL files, COBOL_DIALECT is not set
    (in the environment or a .env) and a terminal is attached (progress.at_terminal).
    The files are those of the index, or, while there is no index yet, those the
    analysis of project_dir is handed (analysis_file_list). The answer, Enter included, is
    added to the .env of the directory the command is run from, which is made when there
    is none, and put in the environment and in config for this run. The line ends with
    the line ending the file uses most (config._main_line_end), which is put after its
    last line too when that has none; the bytes already in the file are left as they
    are. A .env file git tracks is not written to, and none is while the loading of
    .env files is turned off (config.is_env_file_off). The end of the input writes
    nothing, and the default holds for this run without the question being asked again.

    Args:
        index_path: The index whose files decide.
        project_dir: The repository whose files decide while index_path does not open, as
            an absolute path. Empty decides nothing then.
        exclude_list: The names and paths kept out of its analysis, or None for none.

    Returns:
        The line for .env (COBOL_DIALECT=<answer>), whether or not it could be written
        there, or "" when nothing was asked or answered.
    """
    if _COBOL_DIALECT_NAME in os.environ or not progress.at_terminal():
        return ""
    try:
        index = store.load(index_path)
    except (FileNotFoundError, sqlite3.DatabaseError):
        file_ext_set = {ext_of(path) for path in (analysis_file_list(
            project_dir, exclude_list, walk=True)[0] or [] if project_dir else [])}
    else:
        try:
            file_ext_set = {ext_of(path) for path in index.file_path_list()}
        finally:
            index.close()
    if not any(LANG_NAME_DICT.get(ext) == "COBOL" for ext in file_ext_set):
        return ""
    choice_list = [name for name, _ in config.COBOL_DIALECT_CHOICE_TUPLE]
    print("This repository holds COBOL. Choose the dialect the static check reads it in")
    for number, (name, text) in enumerate(config.COBOL_DIALECT_CHOICE_TUPLE, 1):
        print(f"  {number}) {name:<5} ({text})")
    print("  Enter    GnuCOBOL's own default")
    while True:
        try:
            answer = progress.ask(f"{_COBOL_DIALECT_NAME} [1-{len(choice_list)}, Enter "
                                  f"keeps the default]: ").strip()
        except EOFError:
            print()
            logger.info("no answer, so the default dialect is used for this run and "
                        "nothing is written")
            _hold_dialect("")
            return ""
        if answer.isdecimal() and 1 <= int(answer) <= len(choice_list):
            answer = choice_list[int(answer) - 1]
        if answer in choice_list or not answer:
            break
    path = config.ENV_FILE_PATH or os.path.join(os.getcwd(), config.ENV_FILE_NAME)
    line = f"{_COBOL_DIALECT_NAME}={answer}"
    try:
        if config.is_env_file_off():
            raise PermissionError(f"{config.NO_ENV_FILE_ENV} turns the loading of "
                                  f".env files off, so it is not read")
        if not config.ENV_FILE_PATH \
                and config.git_index_state(path) != config.INDEX_FREE:
            raise PermissionError("git tracks it or does not say whether it does, so "
                                  "it is not read as the .env file")
        with open(path, "r+" if os.path.exists(path) else "w", encoding="utf-8",
                  newline="") as handle:
            text = handle.read() if handle.mode == "r+" else ""
            line_end = config._main_line_end(text)
            handle.write(("" if not text or config._line_end(text) else line_end)
                         + line + line_end)
    except OSError as e:
        logger.warning("%s could not be written to %s (%s); it holds for this run only",
                       line, path, e)
    else:
        print(f"wrote {line} to {path}")
    _hold_dialect(answer)
    return line


def _hold_dialect(answer: str) -> None:
    """Put the COBOL dialect answered in the environment and in config for this run.

    Args:
        answer: The dialect, "" for GnuCOBOL's own default.
    """
    os.environ[_COBOL_DIALECT_NAME] = answer
    config.COBOL_DIALECT = answer


def ensure_knowledge(project_dir: str, output_dir: str,
                     rebuild: bool = False,
                     with_doc: bool = False,
                     exclude_list: list[str] | None = None) -> tuple[str, str, str, bool]:
    """Prepare the knowledge database and return its path plus the normalised values.

    Args:
        project_dir: The path to the target repository.
        output_dir: Where codetwine writes its output.
        rebuild: True runs codetwine even when the database already exists, except when
            the design documents are not asked for and the analysis in output_dir can be
            used as it is for the state of the working tree (_describe_state()).
        with_doc: Whether codetwine generates the design documents when it runs. The
            default, False, skips them and calls no LLM (see run_codetwine).
        exclude_list: Extra directory and file names to keep out of the analysis. It reaches
            codetwine only when codetwine runs; an existing database that is used as it
            is keeps what it was made with.

    Returns:
        (path, project_dir, project_name, reused).

        path ... the path to the knowledge database. It is only a join with output_dir;
            a relative output_dir comes back relative. When codetwine ran, what the
            analysis describes is recorded beside it (ANALYSIS_COMMIT_FILE)
        project_dir ... the target repository path made absolute
        project_name ... its basename
        reused ... True when an existing database was used as it was, False when
            codetwine was run to create it. Without the design documents, an analysis
            another command made for the tree on disk while this one waited for its lock
            is used as it was too

    Raises:
        ValueError: When project_dir does not exist, is not a directory, or is the
            filesystem root.
        RuntimeError: When project_dir is not the root of a git repository, when
            output_dir sits inside it or holds it, when output_dir cannot be made or
            written to, when the analysis there is recorded as made over another
            repository, or when the database still does not exist after codetwine ran.
    """
    # == Step 1: check the target and the output directory ===
    project_dir, project_name = _check_target(project_dir, output_dir)

    # == Step 2: use the analysis that is there ==============
    path = knowledge_db_path(output_dir, project_name)
    _check_owner(path, project_dir)
    if is_complete(path) and not rebuild:
        logger.info("using the existing knowledge database: %s", path)
        return path, project_dir, project_name, True

    # == Step 3: run codetwine and record what it describes ==
    is_reuse = _analyse(project_dir, output_dir, project_name, with_doc, exclude_list)
    return path, project_dir, project_name, is_reuse


def check_project_dir(project_dir: str) -> None:
    """Refuse a target repository path that is not a directory.

    Args:
        project_dir: The path to the target repository, as it was passed.

    Raises:
        ValueError: When project_dir does not exist or is not a directory.
    """
    if not os.path.isdir(project_dir):
        raise ValueError(f"no such target repository: {project_dir}")


def check_git_root(project_dir: str) -> None:
    """Refuse a directory that is not the root of a git working tree.

    Args:
        project_dir: The target repository, as an absolute path.

    Raises:
        RuntimeError: When project_dir is inside a working tree rather than its root,
            or in no repository at all.
    """
    if not gitcmd.is_work_tree_root(project_dir):
        raise RuntimeError(
            f"not the root of a git repository: {project_dir}. Run "
            f"`{command_line('git', '-C', project_dir, 'init')}` first, or pass the "
            f"repository itself rather than a directory inside it")


def check_output_place(project_dir: str, output_dir: str) -> None:
    """Refuse an output directory the target repository holds or is inside of.

    A repository whose directory name this tool uses in the output directory is refused
    too. Both paths are compared with their links resolved. Neither has to exist.

    Args:
        project_dir: The path to the target repository.
        output_dir: The output directory.

    Raises:
        RuntimeError: When output_dir is project_dir, is inside it, or holds it, or
            when the repository's directory is named as one of _OUTPUT_OWN_NAME_TUPLE.
    """
    name = os.path.basename(os.path.abspath(project_dir))
    if name in _OUTPUT_OWN_NAME_TUPLE:
        raise RuntimeError(
            f"the repository's directory is named {name} ({project_dir}), and "
            f"{os.path.join(output_dir, name)} is where this tool keeps its agents' "
            f"own files, so the analysis of the repository cannot be written there. "
            f"Rename the repository's directory, or copy the repository to a "
            f"directory of another name and pass that")
    output_real = os.path.realpath(output_dir)
    project_real = os.path.realpath(project_dir)
    if os.path.commonpath([output_real, project_real]) == project_real:
        raise RuntimeError(
            f"OUTPUT_DIR is inside PROJECT_DIR ({output_dir} is inside {project_dir}), "
            f"so the analysis would cover what it writes. Pass an --output-dir outside "
            f"the repository")
    if os.path.commonpath([output_real, project_real]) == output_real:
        raise RuntimeError(
            f"OUTPUT_DIR holds PROJECT_DIR ({project_dir} is inside {output_dir}), so the "
            f"analysis would be written over the repository or every path in it would be "
            f"taken for output. Pass an --output-dir that neither holds the repository nor "
            f"is inside it")


def make_output_dir(output_dir: str) -> None:
    """Make the output directory when it is not there.

    An output directory that cannot be written to is refused.

    Args:
        output_dir: The output directory.

    Raises:
        RuntimeError: When output_dir cannot be made, is not a directory, or files
            cannot be made in it.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"the output directory {output_dir} cannot be made ({e}). Pass an "
            f"--output-dir that can be written to") from None
    if not os.access(output_dir, os.W_OK | os.X_OK):
        raise RuntimeError(
            f"the output directory {output_dir} cannot be written to. Pass an "
            f"--output-dir that can be written to")


def _check_target(project_dir: str, output_dir: str) -> tuple[str, str]:
    """Check the target repository and the output directory before anything is analysed.

    The output directory is made when it is not there (make_output_dir).

    Args:
        project_dir: The path to the target repository, as it was passed.
        output_dir: Where codetwine writes its output.

    Returns:
        (project_dir made absolute, project_name_of() of it).

    Raises:
        ValueError: When project_dir does not exist, is not a directory, or is the
            filesystem root.
        RuntimeError: When project_dir is not the root of a git repository, when
            output_dir sits inside it or holds it, or when output_dir cannot be made
            or written to.
    """
    check_project_dir(project_dir)
    project_name = project_name_of(project_dir)
    project_dir = os.path.abspath(project_dir)
    check_git_root(project_dir)
    check_output_place(project_dir, output_dir)
    make_output_dir(output_dir)
    return project_dir, project_name


def _check_owner(path: str, project_dir: str) -> None:
    """Refuse an analysis the record beside it says was made over another repository.

    The record names the commits the analysed repository's history starts from. When
    project_dir holds none of them, the analysis and its design documents are of
    another repository that has the same directory name. A record that names none is
    not refused.

    Args:
        path: The path to project_knowledge.sqlite.
        project_dir: The target repository, as an absolute path.

    Raises:
        RuntimeError: When the record names commits project_dir does not hold.
    """
    record_path = analysis_record_path(path)
    try:
        record_dict = tmpfile.read_json(record_path, RECORD_MAX_BYTE)
    except (OSError, ValueError):
        return
    root_list = record_dict.get("root") if isinstance(record_dict, dict) else None
    if not isinstance(root_list, list) or not root_list:
        return
    if any(gitcmd.has_commit(project_dir, str(root)) for root in root_list):
        return
    raise RuntimeError(
        f"{os.path.dirname(path)} holds the analysis of another repository with the "
        f"same name: the history it was made over starts at "
        f"{', '.join(gitcmd.short(str(root)) for root in root_list)}, which "
        f"{project_dir} does not hold. Pass another --output-dir, or remove "
        f"{os.path.dirname(path)} to analyse this repository there")


def _analyse(project_dir: str, output_dir: str, project_name: str, with_doc: bool,
             exclude_list: list[str] | None) -> bool:
    """Run codetwine under the analysis' lock and record what the analysis describes.

    The value of what an analysis reads (analysis_state()) is read before and after
    codetwine runs; when the two differ, the record says the analysis describes no state
    and no commit. The commit recorded is HEAD when the value is the one HEAD's tree
    gives (commit_state()), and none otherwise. Without the documents, an analysis that
    can be used as it is for the value read under the lock (_describe_state) is used
    and codetwine does not run.

    Args:
        project_dir: The target repository, as an absolute path.
        output_dir: Where codetwine writes its output.
        project_name: The directory name the analysis is filed under.
        with_doc: Whether codetwine generates the design documents.
        exclude_list: Extra directory and file names to keep out of the analysis.

    Returns:
        True when the analysis on disk was used as it was, False when codetwine ran.

    Raises:
        RuntimeError: When the database does not exist after codetwine ran, or
            run_codetwine() refuses the run.
    """
    path = knowledge_db_path(output_dir, project_name)
    os.makedirs(output_dir, exist_ok=True)
    with filelock.hold_lock(os.path.join(output_dir, project_name),
                            f"the analysis in {output_dir}"):
        state_before = analysis_state(project_dir, exclude_list)
        if not with_doc and _describe_state(path, state_before, exclude_list):
            logger.info("the analysis in %s already describes the tree of %s, so it is "
                        "used as it is", output_dir, project_dir)
            return True
        run_codetwine(project_dir, output_dir, with_doc=with_doc,
                      exclude_list=exclude_list)
        if not os.path.exists(path):
            raise RuntimeError(
                f"the database is still missing after codetwine ran: {path}")
        state = analysis_state(project_dir, exclude_list)
        if state != state_before:
            logger.warning("the tree of %s changed while codetwine ran, so the analysis "
                           "is recorded as describing no state and the next command "
                           "makes it again", project_dir)
            state = ""
        head = gitcmd.head_commit(project_dir) if state else ""
        commit = head if head and state == commit_state(project_dir, head,
                                                         exclude_list) else ""
        record_analysis_commit(path, commit, state, exclude_list,
                               gitcmd.root_commit_list(project_dir))
    return False


def _describe_state(path: str, state: str, exclude_list: list[str] | None) -> bool:
    """Say whether the analysis on disk can be used as it is for one state of the tree.

    Args:
        path: The path to project_knowledge.sqlite.
        state: The value of what an analysis of the tree reads (analysis_state()). ""
            matches nothing.
        exclude_list: The extra names the analysis has to keep out.

    Returns:
        True when the analysis is complete, its usages name the definition they lead
        to, and its record names that state, the same names kept out, the encodings
        this run reads with and the codetwine installed now (same_codetwine()).
    """
    if not state or not is_complete(path):
        return False
    knowledge = Knowledge.open(path)
    try:
        stamp = knowledge.stamp()
        has_target = knowledge.names_target()
    finally:
        knowledge.close()
    record_dict = analysis_record(path, stamp)
    return (has_target and record_dict.get("state") == state
            and same_exclude(record_dict, exclude_list) and same_encoding(record_dict)
            and same_codetwine(record_dict))
