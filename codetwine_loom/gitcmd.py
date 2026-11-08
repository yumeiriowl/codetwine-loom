"""Starting git and reading its answers.

_git_argv() builds the command line of a git command, and run() and three readers start
git with it. run() returns the finished process. The readers read the output as git
writes it: run_entry(), which keeps the entries asked for of a listing, diff_head(),
which keeps the start of a diff, and _write_side_list(), which writes the blobs git
cat-file gives into files.

It also answers questions about a repository, among them is_ancestor() (whether one
commit is already reached by another), name_status() (what changed, split by git's
one-letter code) and work_tree_id() (the tree the files on disk make).

A command that fails does not raise: run() returns its returncode and stdout, and a
directory that is not a repository is answered rather than being an exception.

git is given config.child_env() and config.GIT_SETTING_TUPLE. No hook runs: the
repository's own hooks and any core.hooksPath it sets are replaced by an empty path.
"""

import contextlib
import hashlib
import logging
import os
import re
import select
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import IO, NamedTuple

from codetwine_loom import config, tmpfile
from codetwine_loom.textcut import line_list_of, name_list_text

logger = logging.getLogger(__name__)

# The letter git's --name-status gives each kind of change. A letter missing here (T, a
# conflict's U) is a change to the contents
LETTER_BY_KIND_DICT = {"added": "A", "modified": "M", "deleted": "D"}
CHANGE_KIND_TUPLE = tuple(LETTER_BY_KIND_DICT)

# How many characters of a commit hash a log line or a message names
SHORT_COMMIT_CHAR = 12

# What a git command that writes a commit of this tool's own is started with. The commit
# is attributed to this tool, as its author and as its committer, whatever user.*,
# author.* and committer.* the git settings hold, and is not signed, the commits a merge
# names as parents are not checked for a signature, and the message is stored as UTF-8
# with every line of it kept, a line that starts with the comment character too. Passed
# as -c settings, which a repository with no identity configured commits with too
_OWN_NAME = "codetwine-loom"
_OWN_EMAIL = "codetwine-loom@localhost"
COMMITTER_TUPLE = (*(part for who in ("user", "author", "committer")
                     for part in ("-c", f"{who}.name={_OWN_NAME}",
                                  "-c", f"{who}.email={_OWN_EMAIL}")),
                   "-c", "commit.gpgsign=false", "-c", "merge.verifySignatures=false",
                   "-c", "i18n.commitEncoding=UTF-8", "-c", "commit.cleanup=whitespace")

# What a diff read as text or applied starts with, "diff" included, whatever the user's
# own git settings say about its shape: no colour, no external diff program, no text
# conversion, the prefixes a/ and b/, three lines of context, no context between hunks,
# the myers algorithm with the indent heuristic, no renames, paths from the top of the
# repository, a repository of its own shown as its commit whatever untracked files it
# holds, git's own order of the files, and a blank context line written with its leading
# space. An option given after it wins over it
PLAIN_DIFF_TUPLE = ("-c", "diff.suppressBlankEmpty=false", "diff",
                    "--no-color", "--no-ext-diff", "--no-textconv", "--src-prefix=a/",
                    "--dst-prefix=b/", "--unified=3", "--inter-hunk-context=0",
                    "--diff-algorithm=myers", "--indent-heuristic", "--no-renames",
                    "--no-relative", "--submodule=short",
                    "--ignore-submodules=untracked", "-O/dev/null")

# What a diff read for a list of paths, a count or a yes-or-no starts with, "diff"
# included, whatever the user's own git settings say about the answer: a rename as a
# deleted path and an added one, no external diff program, a repository of its own shown
# as its commit whatever untracked files it holds, and git's own order of the files
PATH_DIFF_TUPLE = ("diff", "--no-renames", "--no-ext-diff",
                   "--ignore-submodules=untracked", "-O/dev/null")

# What the "git status" read for the paths git reports as changed starts with: one entry
# to a path with NUL between them, every untracked file by itself, a rename as two
# entries, and a repository of its own listed whenever it differs
_STATUS_TUPLE = ("status", "--porcelain", "-z", "-uall", "--no-renames",
                 "--ignore-submodules=none")

# What the "git status" starts with that says which paths a user of the repository is
# shown: one entry to a path with NUL between them, every untracked file by itself and
# a rename as two entries. What is ignored, what counts as a change and which
# repositories of their own are listed is left to the git settings
_USER_STATUS_TUPLE = ("status", "--porcelain=v1", "-z", "--untracked-files=all",
                      "--no-renames")

# What a "git apply" starts with: binary patches are taken, the whitespace of a line is
# applied as the patch has it, whatever apply.whitespace says, and a context line is
# matched with its whitespace, whatever apply.ignoreWhitespace says
APPLY_TUPLE = ("apply", "--binary", "--whitespace=nowarn", "--no-ignore-whitespace")

# The address space, in bytes, to hold a git command to, as run()'s max_address_byte
MAX_PATCH_ADDRESS_BYTE = 4 << 30

# The settings a git command held to an address space is given besides: how much of a
# pack file it maps at a time, and how much of the pack files it keeps mapped in all
_PACK_MAP_TUPLE = ("-c", "core.packedGitWindowSize=64m", "-c", "core.packedGitLimit=256m")

# What a command held to an address space is started through: a Python process that
# takes the limit in bytes and the command line as its arguments, sets the limit on
# itself, no higher than the soft and the hard one it runs under, and becomes the
# command
_LIMIT_START = (
    "import os, resource, sys\n"
    "limit = int(sys.argv[1])\n"
    "for bound in resource.getrlimit(resource.RLIMIT_AS):\n"
    "    if bound != resource.RLIM_INFINITY:\n"
    "        limit = min(limit, bound)\n"
    "resource.setrlimit(resource.RLIMIT_AS, (limit, limit))\n"
    "os.execvp(sys.argv[2], sys.argv[2:])\n")

# The exit status a command that was stopped at its time limit comes back with
TIMEOUT_STATUS = 124

# The exit status a command that was stopped at its size limit comes back with
SIZE_STATUS = 125

# How much of a command's output is read at a time, in bytes and in characters
_READ_BLOCK_BYTE = 1 << 16

# What the files on disk are added to an index with: every change, deletion and new
# file, a file that cannot be read passed over, and a line ending core.safecrlf refuses
# taken as core.autocrlf converts it
_ADD_TUPLE = ("-c", "core.safecrlf=false", "add", "-A", "--ignore-errors")

# The option that makes "git add" take a path outside the definition of a sparse
# checkout too
_ADD_SPARSE = "--sparse"

# What a git command that works on a copy of the index (_index_copy) starts with: the
# copy is written whole, with nothing put in the repository's git directory
_INDEX_COPY_TUPLE = ("-c", "core.splitIndex=false")

# The exit status git ends with when it stops at an error it does not pass over, and
# the one it ends with when it does not know an option
_FATAL_STATUS = 128
_USAGE_STATUS = 129

# What the subject of a snapshot commit starts with
SNAPSHOT_SUBJECT = "codetwine-loom: the working tree"

# How much of a changed file is read at a time when it is hashed
_HASH_BLOCK_BYTE = 1 << 16

# One escape in a path git writes between double quotes: a backslash and either three
# octal digits or one character
_QUOTE_ESCAPE = re.compile(rb"\\([0-7]{3}|.)", re.DOTALL)

# The byte each one-character escape stands for. Any other character stands for itself
_ESCAPE_BYTE_DICT = {b"a": 7, b"b": 8, b"t": 9, b"n": 10, b"v": 11, b"f": 12, b"r": 13}

# The file names not in UTF-8 already named in the log, each named once
_NOT_UTF8_NAME_SET: set[str] = set()

# The keys of the warnings warn_once() has given in this process
_WARN_KEY_SET: set[tuple] = set()


def warn_once(key: tuple, message: str, *arg_tuple: object, detail: str = "",
              log: logging.Logger = logger) -> None:
    """Log a warning the first time its key comes in this process, and at DEBUG after.

    What git wrote (detail) is logged at DEBUG alone, after the warning.

    Args:
        key: What tells one warning from another: what was asked, of which repository
            and over which paths.
        message: The warning, with %-style placeholders.
        *arg_tuple: The values for the placeholders.
        detail: What git wrote on standard error, or "" for nothing.
        log: The logger the lines go to.
    """
    log.log(logging.DEBUG if key in _WARN_KEY_SET else logging.WARNING, message,
            *arg_tuple)
    _WARN_KEY_SET.add(key)
    if detail:
        log.debug("what git wrote: %s", detail)


def short(commit: str) -> str:
    """Return the first SHORT_COMMIT_CHAR characters of a commit hash.

    Args:
        commit: A full hash, or whatever stands in for one ("" stays "").

    Returns:
        The shortened hash.
    """
    return commit[:SHORT_COMMIT_CHAR]


def _git_argv(project_dir: str, *args: str) -> list[str]:
    """Return the command line of one git command run in a repository.

    Args:
        project_dir: The repository the command runs in.
        *args: The git arguments, without "git" itself.

    Returns:
        ["git", "-C", project_dir, *config.GIT_SETTING_TUPLE, *args].

    Raises:
        FileNotFoundError: When project_dir is empty, before git runs.
    """
    if not project_dir:
        raise FileNotFoundError("no repository to run git in. PROJECT_DIR is not set")
    return ["git", "-C", project_dir, *config.GIT_SETTING_TUPLE, *args]


def run(project_dir: str, *args: str, env_dict: dict[str, str] | None = None,
        input_text: str | bytes | None = None,
        text: bool = True, timeout_sec: float | None = None,
        max_address_byte: int | None = None,
        base_env_dict: Mapping[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run one git command in a repository and return the finished process.

    Examples:
        run(repo, "rev-parse", "HEAD").stdout.strip()

    Args:
        project_dir: The repository the command runs in.
        *args: The git arguments, without "git" itself.
        env_dict: Settings put in the command's environment on top of
            config.child_env(), such as the index file it works with. None adds none.
        input_text: What the command reads on its standard input, as bytes when text
            is False. None gives it none.
        text: False captures stdout and stderr as bytes, for reading a file's contents
            whatever its encoding and a path whatever its bytes.
        timeout_sec: How long the command may run, in seconds. A command still
            running then is killed. None sets no limit.
        max_address_byte: How much address space the command may take, in bytes. It
            is started through _LIMIT_START and with _PACK_MAP_TUPLE, and ends with
            an exit status that is not 0 once it asks for more. None sets no limit.
        base_env_dict: The environment config.child_env() starts from in place of
            this process's own. None starts from this process's own.

    Returns:
        The finished process, with stdout and stderr captured as text (as bytes when
        text is False). The exit status is not checked here. A command killed at
        timeout_sec comes back with TIMEOUT_STATUS as its exit status, no stdout, and
        a stderr saying so.

    Raises:
        FileNotFoundError: When project_dir is empty, before git runs.
    """
    argv = _git_argv(project_dir, *args)
    if max_address_byte is not None:
        argv = [sys.executable, "-I", "-S", "-c", _LIMIT_START, str(max_address_byte),
                *argv[:3], *_PACK_MAP_TUPLE, *argv[3:]]
    try:
        return subprocess.run(argv, capture_output=True, check=False, input=input_text,
                              env=config.child_env(env_dict, base_env_dict),
                              timeout=timeout_sec,
                              **(config.CHILD_TEXT_DICT if text else {}))
    except subprocess.TimeoutExpired:
        message = f"git did not end within {timeout_sec:g} seconds and was killed"
        return subprocess.CompletedProcess(
            argv, TIMEOUT_STATUS, "" if text else b"",
            message if text else message.encode("utf-8"))


def run_entry(project_dir: str, *args: str, keep: Callable[[str], bool],
              max_byte: int, timeout_sec: float | None = None
              ) -> subprocess.CompletedProcess:
    """Run one git command that writes entries ended by NUL, keeping the ones wanted.

    The output is read as git writes it and only the kept entries are held. A command
    that writes more than max_byte bytes is killed, and so is one still running at
    timeout_sec.

    Args:
        project_dir: The repository the command runs in.
        *args: The git arguments, without "git" itself.
        keep: Says of each entry, as text, whether it is kept.
        max_byte: The most bytes of output that are read.
        timeout_sec: How long the command may run, in seconds. None sets no limit.

    Returns:
        The finished process: stdout is the list of the kept entries, read as run()
        reads text, and stderr the first _READ_BLOCK_BYTE bytes of what git wrote
        there. A command killed at timeout_sec comes back with TIMEOUT_STATUS and one
        killed at max_byte with SIZE_STATUS, each with no entry and a stderr saying so.

    Raises:
        FileNotFoundError: When project_dir is empty, before git runs.
    """
    argv = _git_argv(project_dir, *args)
    deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
    entry_list, rest, read_count, stop_status, message = [], b"", 0, 0, ""
    with tempfile.TemporaryFile() as error_file, subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=error_file,
            env=config.child_env()) as proc:
        try:
            while not stop_status:
                wait_sec = None if deadline is None else deadline - time.monotonic()
                if wait_sec is not None and (wait_sec <= 0 or not select.select(
                        [proc.stdout], [], [], wait_sec)[0]):
                    stop_status = TIMEOUT_STATUS
                    message = (f"git did not end within {timeout_sec:g} seconds and "
                               f"was killed")
                    break
                block = os.read(proc.stdout.fileno(), _READ_BLOCK_BYTE)
                if not block:
                    break
                read_count += len(block)
                if read_count > max_byte:
                    stop_status = SIZE_STATUS
                    message = f"git wrote more than {max_byte} bytes and was killed"
                    break
                part_list = (rest + block).split(b"\0")
                rest = part_list.pop()
                entry_list += [entry for entry in (
                    part.decode("utf-8", "replace") for part in part_list)
                    if entry and keep(entry)]
        finally:
            if proc.poll() is None and (stop_status or sys.exc_info()[0] is not None):
                proc.kill()
        status = proc.wait()
        error_file.seek(0)
        error_text = error_file.read(_READ_BLOCK_BYTE).decode("utf-8", "replace")
    if stop_status:
        return subprocess.CompletedProcess(argv, stop_status, [], message)
    last_entry = rest.decode("utf-8", "replace")
    if last_entry and keep(last_entry):
        entry_list.append(last_entry)
    return subprocess.CompletedProcess(argv, status, entry_list, error_text)


def shown_text(text: bytes | str) -> str:
    """Return what git wrote as text.

    UTF-8 is decoded as it is, and every byte that is not UTF-8 is written as "\\xNN".

    Examples:
        b"caf\\xe9.txt"        -> "caf\\\\xe9.txt"
        os.fsdecode(b"caf\\xe9.txt") -> "caf\\\\xe9.txt"

    Args:
        text: git's output as bytes, or as os.fsdecode() gives it.

    Returns:
        The text.
    """
    raw = text if isinstance(text, bytes) else os.fsencode(text)
    return raw.decode("utf-8", "backslashreplace")


def unquote_path(text: str) -> str:
    """Return a path as it is, from the form git writes it in a patch's "---"/"+++" line.

    git writes a path holding a double quote, a backslash or a control character
    between double quotes with C-style escapes, and puts a tab after a path holding a
    space, quoted or not. Both are undone here.

    Examples:
        'b/src/main.py'           -> 'b/src/main.py'
        '"b/q\\"t.py"'            -> 'b/q"t.py'
        '"b/tab\\tname.py"'       -> 'b/tab<TAB>name.py'
        'b/space name.py<TAB>'    -> 'b/space name.py'
        '"b/a \\"b.py"<TAB>'      -> 'b/a "b.py'

    Args:
        text: What follows "--- " or "+++ ", decoded with os.fsdecode(), which keeps a
            byte that is not UTF-8.

    Returns:
        The path with git's quoting and trailing tab taken off, as shown_text() gives it.
    """
    return shown_text(_unquote_raw(os.fsencode(text)))


def _unquote_raw(raw: bytes) -> bytes:
    """Return the bytes of a path from its form in a patch's "---"/"+++" line.

    The form is the one git writes, read as unquote_path() reads it.

    Args:
        raw: What follows "--- " or "+++ ", as bytes.

    Returns:
        The path with git's quoting and trailing tab taken off, as bytes.
    """
    raw = raw.removesuffix(b"\t")
    if len(raw) < 2 or not (raw.startswith(b'"') and raw.endswith(b'"')):
        return raw

    def unescape(match: re.Match) -> bytes:
        """Return the byte one escape of _QUOTE_ESCAPE stands for."""
        escape = match.group(1)
        if len(escape) == 3:
            return bytes([int(escape, 8)])
        return bytes([_ESCAPE_BYTE_DICT.get(escape, escape[0])])

    return _QUOTE_ESCAPE.sub(unescape, raw[1:-1])


def status_path_list(project_dir: str, new_only: bool = False,
                     disk: bool = False) -> list[str]:
    """Return the paths git reports as changed, read from one porcelain status.

    Its two-letter prefix is stripped here. -z gives every path as it is, without
    quoting. -uall lists the files inside an untracked directory rather than the
    directory itself: a file created in a new directory comes back as the path it was
    given. --no-renames gives a staged rename as its deleted and its added path.

    Args:
        project_dir: The repository to look at.
        new_only: True keeps the "?? " lines alone.
        disk: True gives each path as os.fsdecode() does, which opens and removes the
            file whatever its bytes. False gives it as shown_text() does.

    Returns:
        Repository-relative paths, in git's order. Empty for a directory that is not a
        repository.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    status = run(project_dir, *_STATUS_TUPLE, text=False).stdout
    prefix = b"?? " if new_only else b""
    decode = os.fsdecode if disk else shown_text
    return [decode(entry[3:]) for entry in status.split(b"\0")
            if entry[3:] and entry.startswith(prefix)]


def new_file_list(project_dir: str, disk: bool = False) -> list[str]:
    """Return the paths git does not track, one per file.

    Args:
        project_dir: The repository to look at.
        disk: As status_path_list() takes it.

    Returns:
        Repository-relative paths, in git's order.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    return status_path_list(project_dir, new_only=True, disk=disk)


def status_path_set(project_dir: str) -> set[bytes] | None:
    """Return the paths "git status" names in a repository, under its git settings.

    Args:
        project_dir: The repository to look at.

    Returns:
        The paths relative to the repository, as bytes, read from one status made
        with _USER_STATUS_TUPLE: a repository of its own that git does not track is
        named as its directory with a "/" after it. None when git gives no answer,
        which is logged.
    """
    proc = run(project_dir, *_USER_STATUS_TUPLE, text=False)
    if proc.returncode != 0:
        logger.info("git status gave no answer in %s: %s", project_dir,
                    shown_text(proc.stderr).strip())
        return None
    return {entry[3:] for entry in proc.stdout.split(b"\0") if entry[3:]}


def work_tree_file_list(project_dir: str) -> list[str] | None:
    """Return the files of a working tree that git tracks or does not ignore.

    A symbolic link is left out, and so is a path whose real location is outside the
    repository (one under a directory that is a link) and a path that is not UTF-8,
    which the log names.

    Args:
        project_dir: The repository to look at.

    Returns:
        Repository-relative paths, in git's order: the tracked files, then the untracked
        ones no ignore rule covers. A tracked file deleted from the working tree is
        still listed. None for a directory that is not a repository.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    proc = run(project_dir, "ls-files", "--cached", "--others", "--exclude-standard", "-z",
               text=False)
    if proc.returncode != 0:
        return None
    root = os.path.realpath(project_dir)
    path_list, other_list = [], []
    own_dir_dict: dict[str, bool] = {}
    for raw_path in proc.stdout.split(b"\0"):
        if not raw_path:
            continue
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError:
            other_list.append(shown_text(raw_path))
            continue
        if _is_own_file(root, path, own_dir_dict):
            path_list.append(path)
    other_list = [path for path in other_list if path not in _NOT_UTF8_NAME_SET]
    if other_list:
        _NOT_UTF8_NAME_SET.update(other_list)
        logger.warning("%d file name(s) of %s are not UTF-8, so those files are left "
                       "out: %s", len(other_list), project_dir, ", ".join(other_list))
    return path_list


def tracked_path_list(project_dir: str, path_list: list[str]) -> list[str]:
    """Return the paths of a list that the index of a repository names.

    The whole index is listed once and path_list is looked up in it.

    Args:
        project_dir: The repository to look at.
        path_list: Repository-relative paths, each read as it is written.

    Returns:
        The paths of path_list the index names, in path_list's order. Empty for an empty
        path_list and when git gives no answer.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    if not path_list:
        return []
    proc = run(project_dir, "ls-files", "--cached", "-z", text=False)
    if proc.returncode != 0:
        return []
    name_set = {os.fsdecode(entry) for entry in proc.stdout.split(b"\0") if entry}
    return [path for path in path_list if path in name_set]


def _is_own_file(root: str, path: str, own_dir_dict: dict[str, bool]) -> bool:
    """Say whether a path of a working tree is a file of the repository, not a link.

    Args:
        root: The repository, resolved through realpath.
        path: The path, relative to root.
        own_dir_dict: {directory relative to root: whether its real location is root or
            inside it}, read and written here so that each directory is resolved once.

    Returns:
        False for a symbolic link and for a path whose real location is outside root.
        A path that does not exist on disk is True.
    """
    if os.path.islink(os.path.join(root, path)):
        return False
    dir_rel = os.path.dirname(path)
    if dir_rel not in own_dir_dict:
        dir_real = os.path.realpath(os.path.join(root, dir_rel))
        own_dir_dict[dir_rel] = dir_real == root or dir_real.startswith(root + os.sep)
    return own_dir_dict[dir_rel]


# What numstat_list() gives in place of the line counts of a path git could not read
UNREAD_COUNT = "?"


def numstat_list(project_dir: str, commit: str = "HEAD") -> list[dict]:
    """Return the tracked files the working tree changes against a commit, with line counts.

    git's tab-separated entry is read here, with -z giving the path as it is, without
    quoting. Naming the commit compares the working tree against it rather than against
    the index: staged and unstaged changes both count. --no-renames keeps a rename as
    separate paths instead of one bracketed entry.

    When git stops at a path of the working tree it cannot read (a named pipe, a file
    it may not open), the paths are counted again past the ones that cannot be read
    (work_tree_diff), and those come with UNREAD_COUNT for both counts. When git still
    stops, every changed path comes with UNREAD_COUNT, and the log says why.

    Args:
        project_dir: The repository to look at.
        commit: The commit the working tree is compared against.

    Returns:
        [{"path": str, "insertions": int, "deletions": int}, ...] in git's order, the
        path as shown_text() gives it. A binary file counts 0 lines either way. A path
        that could not be read has UNREAD_COUNT in place of both numbers.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    proc, unread_list = work_tree_diff(project_dir, commit, *PATH_DIFF_TUPLE, commit,
                                       "--numstat", "-z")
    if proc.returncode != 0:
        detail = shown_text(proc.stderr).strip()
        warn_once(("count", project_dir, commit, detail),
                  "git could not count the changed lines of %s against %s, so the "
                  "changed paths are named without counts", project_dir, commit,
                  detail=detail)
        unread_list = _diff_path_list(project_dir, commit)
    unread_set = set(unread_list)
    count_dict = {}
    for entry in proc.stdout.split(b"\0") if proc.returncode == 0 else []:
        if not entry:
            continue
        insert_text, delete_text, path = entry.split(b"\t", 2)
        if path not in unread_set:
            count_dict[path] = (int(insert_text) if insert_text != b"-" else 0,
                                int(delete_text) if delete_text != b"-" else 0)
    if unread_list:
        path_list = [path for path in _diff_path_list(project_dir, commit)
                     if path in count_dict or path in unread_set]
    else:
        path_list = list(count_dict)
    return [{"path": shown_text(path),
             "insertions": count_dict[path][0] if path in count_dict else UNREAD_COUNT,
             "deletions": count_dict[path][1] if path in count_dict else UNREAD_COUNT}
            for path in path_list]


def _diff_path_list(project_dir: str, commit: str) -> list[bytes]:
    """Return the tracked paths the working tree changes against a commit.

    Args:
        project_dir: The repository to look at.
        commit: The commit the working tree is compared against.

    Returns:
        The paths as git writes them, in git's order. Empty when git gives no answer.
    """
    proc = run(project_dir, *PATH_DIFF_TUPLE, commit, "--name-only", "-z", text=False)
    if proc.returncode != 0:
        return []
    return [path for path in proc.stdout.split(b"\0") if path]


def _unread_diff_path_list(project_dir: str, commit: str) -> list[bytes]:
    """Return the changed tracked paths whose working tree file git cannot read.

    Args:
        project_dir: The repository to look at.
        commit: The commit the working tree is compared against.

    Returns:
        The paths of _diff_path_list() where something other than a regular file, a
        symbolic link or a directory stands, or a regular file this process cannot open
        for reading.
    """
    return [path for path in _diff_path_list(project_dir, commit)
            if _is_unread_on_disk(project_dir, path)]


def _is_unread_on_disk(project_dir: str, path: bytes) -> bool:
    """Say whether what stands at a path of a working tree is a file git cannot read.

    Args:
        project_dir: The repository.
        path: The path, relative to project_dir, as git writes it.

    Returns:
        True for something other than a regular file, a symbolic link or a directory,
        and for a regular file this process cannot open for reading. False for a path
        where nothing stands or that cannot be looked at.
    """
    full_path = os.path.join(project_dir, os.fsdecode(path))
    if not os.path.lexists(full_path) or os.path.islink(full_path) \
            or os.path.isdir(full_path):
        return False
    return not os.path.isfile(full_path) or not os.access(full_path, os.R_OK)


def work_tree_diff(project_dir: str, commit: str, *args: str
                   ) -> tuple[subprocess.CompletedProcess, list[bytes]]:
    """Run one git diff of the working tree against a commit, past what git cannot read.

    When git ends with an exit status that is not 0, the changed paths whose working
    tree file cannot be read (_unread_diff_path_list) are handed on standard input to
    "git update-index --assume-unchanged" over a copy of the index (_index_copy), and
    the diff is run again over that copy: git takes those files as the index holds them
    and reads the others.

    Args:
        project_dir: The repository.
        commit: The commit the working tree is compared against.
        *args: The git arguments of the diff, without "git" itself.

    Returns:
        (the finished process, as run() returns it with text False, the paths that could
        not be read, as git writes them). The paths are empty when the first run ended
        with exit status 0 or none of the changed paths is one that cannot be read; the
        process is the first run's when the copy cannot be made or marked.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    proc = run(project_dir, *args, text=False)
    if proc.returncode == 0:
        return proc, []
    unread_list = _unread_diff_path_list(project_dir, commit)
    if not unread_list:
        return proc, []
    with _index_copy(project_dir) as env_dict:
        entry_list = index_entry_list(project_dir, env_dict) if env_dict else None
        if entry_list is None:
            return proc, unread_list
        mark_list = sorted(set(unread_list) & {path for _, _, path in entry_list})
        if mark_list:
            mark_proc = run(project_dir, *_INDEX_COPY_TUPLE, "update-index", "-z",
                            "--assume-unchanged", "--stdin", env_dict=env_dict,
                            input_text=b"\0".join(mark_list) + b"\0", text=False)
            if mark_proc.returncode != 0:
                logger.debug("cannot mark the paths that cannot be read in a copy of "
                             "the index of %s: %s", project_dir,
                             shown_text(mark_proc.stderr).strip())
                return proc, unread_list
        return run(project_dir, *_INDEX_COPY_TUPLE, *args, env_dict=env_dict,
                   text=False), unread_list


# The git arguments that end with exit status 0 when the commit after them is reached
# by the one after that
IS_ANCESTOR_TUPLE = ("merge-base", "--is-ancestor")


def is_ancestor(project_dir: str, older: str, newer: str) -> bool:
    """Say whether one commit is already reached by another.

    Args:
        project_dir: The repository to read.
        older: The commit looked for.
        newer: The commit looked from.

    Returns:
        True when older is newer or comes before it. A commit neither of them names,
        and a directory that is not a repository, both give False.
    """
    return run(project_dir, *IS_ANCESTOR_TUPLE, older, newer).returncode == 0


def move_dict(project_dir: str, commit: str, other_commit: str) -> dict[str, str] | None:
    """Return the files of one commit git reads as moved in another.

    Args:
        project_dir: The repository.
        commit: The commit the files are read from.
        other_commit: The commit they are looked for in.

    Returns:
        {the path in commit: the path in other_commit}, both as shown_text() gives
        them, or None when git could not compare the two.
    """
    proc = run(project_dir, "diff-tree", "-r", "-z", "-M", "--diff-filter=R",
               "--name-status", commit, other_commit, text=False)
    if proc.returncode != 0:
        logger.info("cannot read the files moved between %s and %s: %s",
                    short(commit), short(other_commit), shown_text(proc.stderr).strip())
        return None
    # One record is the status, the old path and the new path
    field_list = proc.stdout.split(b"\0")
    return {shown_text(old_path): shown_text(new_path)
            for old_path, new_path in zip(field_list[1::3], field_list[2::3])}


def add_by_path_dict(project_dir: str, base: str, commit: str) -> dict[bytes, bool]:
    """Return the paths one commit holds differently from another.

    Args:
        project_dir: The target repository, as an absolute path.
        base: The commit the change starts at.
        commit: The commit the change leads to.

    Returns:
        {the path relative to the repository, as git wrote it: True when base does
        not hold it}, in git's order, a file that was moved at both of its paths.

    Raises:
        RuntimeError: When git cannot compare the two.
    """
    proc = run(project_dir, *PATH_DIFF_TUPLE, "--name-status", "-z", base, commit,
               text=False)
    if proc.returncode != 0:
        raise RuntimeError(f"could not read what {short(commit)} changes against "
                           f"{short(base)}: {shown_text(proc.stderr).strip()}")
    field_list = proc.stdout.split(b"\0")[:-1]
    add_letter = LETTER_BY_KIND_DICT["added"].encode("ascii")
    return {raw: letter == add_letter
            for letter, raw in zip(field_list[::2], field_list[1::2])}


def name_status(project_dir: str, base: str = "", commit: str = "") -> dict:
    """Return the changed paths split into added, modified and deleted.

    git's one-letter code is read here, with -z giving each path as it is, without
    quoting. With no revisions named the working tree is compared against HEAD; naming
    both compares one commit against another.

    --no-renames keeps a rename as a deleted path and an added one rather than one
    bracketed entry: every path git reports is a path that exists on one side.

    Examples:
        (repo) -> {"added": ["new.py"], "modified": ["main.py"], "deleted": []}
        (repo, "abc123^", "abc123") -> what that one commit changed

    Args:
        project_dir: The repository to look at.
        base: The commit the range starts at. Empty compares against HEAD.
        commit: The commit the range ends at. Empty compares against HEAD.

    Returns:
        {"added": list[str], "modified": list[str], "deleted": list[str]}, each sorted
        and without repeats, the paths as shown_text() gives them. All three empty, and
        no exception, for a range git cannot read.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    range_list = [f"{base}..{commit}"] if base and commit else ["HEAD"]
    # -z gives the code and the path as two entries, one after the other
    entry_list = run(project_dir, *PATH_DIFF_TUPLE, "--name-status", "-z",
                     *range_list, text=False).stdout.split(b"\0")
    by_kind_dict = {kind: set() for kind in CHANGE_KIND_TUPLE}
    kind_by_letter_dict = {letter: kind for kind, letter in LETTER_BY_KIND_DICT.items()}
    for code, path in zip(entry_list[0::2], entry_list[1::2]):
        if not path:
            continue
        by_kind_dict[kind_by_letter_dict.get(code[:1].decode("ascii", "replace"),
                                             "modified")].add(shown_text(path))
    return {kind: sorted(path_set) for kind, path_set in by_kind_dict.items()}


def diff_text(project_dir: str, base: str, commit: str) -> str:
    """Return the diff of one commit against another, as text.

    Args:
        project_dir: The repository to look at.
        base: The commit the range starts at.
        commit: The commit the range ends at.

    Returns:
        git's unified diff, made with PLAIN_DIFF_TUPLE. "" for a range git cannot
        read.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    proc = run(project_dir, *PLAIN_DIFF_TUPLE, base, commit)
    return proc.stdout if proc.returncode == 0 else ""


def diff_head(project_dir: str, base: str, commit: str, max_char: int
              ) -> tuple[str, int]:
    """Return the start of the diff of one commit against another, and how much follows.

    The diff is read as git writes it, and what follows the start is counted and
    not held.

    Args:
        project_dir: The repository to look at.
        base: The commit the range starts at.
        commit: The commit the range ends at.
        max_char: How many characters of the diff are returned. 0 or less returns
            the whole of it.

    Returns:
        (the first max_char characters of what diff_text() returns, the number of
        characters after them). ("", 0) for a range git cannot read.

    Raises:
        FileNotFoundError: When project_dir is empty, before git runs.
    """
    argv = _git_argv(project_dir, *PLAIN_DIFF_TUPLE, base, commit)
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          env=config.child_env(), **config.CHILD_TEXT_DICT) as proc:
        head = proc.stdout.read(max_char if max_char > 0 else -1)
        rest_count = 0
        while block := proc.stdout.read(_READ_BLOCK_BYTE):
            rest_count += len(block)
    return (head, rest_count) if proc.returncode == 0 else ("", 0)


def write_patch(project_dir: str, base: str, commit: str,
                patch_path: str) -> subprocess.CompletedProcess:
    """Write the change one commit makes over another to a file, as a patch to apply.

    git writes the file itself: the bytes of the files are not read through this
    process's text decoding. The patch is made with PLAIN_DIFF_TUPLE and holds binary
    files. git is held to MAX_PATCH_ADDRESS_BYTE while it writes it.

    Args:
        project_dir: The repository to look at.
        base: The commit the change starts at.
        commit: The commit the change ends at.
        patch_path: Where the patch is written, as an absolute path. Its directory has
            to exist.

    Returns:
        The finished process. The file is empty when the two commits hold the same
        tree.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    return run(project_dir, *PLAIN_DIFF_TUPLE, "--binary",
               f"--output={patch_path}", base, commit,
               max_address_byte=MAX_PATCH_ADDRESS_BYTE)


def tree_diff_path_list(project_dir: str, commit: str,
                        env_dict: dict[str, str] | None = None) -> list[str]:
    """Return the paths the working tree holds differently from a commit's tree.

    Args:
        project_dir: The repository, whose working tree is read.
        commit: The commit, or the tree, compared against.
        env_dict: As run() takes it, such as the index file the working tree is read
            through.

    Returns:
        Repository-relative paths as os.fsdecode() gives them, in git's order: a file
        whose contents differ, and a file only one side holds. [] when git cannot
        compare.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    proc = run(project_dir, *PATH_DIFF_TUPLE, "--name-only", "-z", commit,
               env_dict=env_dict, text=False)
    if proc.returncode != 0:
        return []
    return [os.fsdecode(path) for path in proc.stdout.split(b"\0") if path]


def has_change_in_tree(project_dir: str, start: str, commit: str) -> bool | None:
    """Say whether the working tree holds the change one commit makes over another.

    The paths looked at are the ones the change touches and the ones each commit from
    commit down its first parents to start touches against its first parent: a path one
    commit changed and a later one changed back is among them (_path_change_dict). The
    working tree is read path by path against commit (_has_path_in_tree). Nothing is
    written into the repository.

    Args:
        project_dir: The repository, whose working tree is read.
        start: The commit the change starts at.
        commit: The commit the change ends at.

    Returns:
        True when the working tree holds every such path as _has_path_in_tree() reads
        it, and when there is none. False when it does not hold one, or one cannot be
        read. None when git cannot read the change, which is logged.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    change_dict = _path_change_dict(project_dir, start, commit)
    if change_dict is None:
        return None
    return _has_path_in_tree(project_dir, change_dict)


# The modes git records for a file that is not a link: not executable, and executable
_FILE_MODE_TUPLE = (b"100644", b"100755")

# The mode a raw diff gives the side of a path that is not there
_NO_MODE = b"000000"

# What git log is made to write ahead of each commit's entries of a raw diff
_COMMIT_MARK = b"\x01"

# The options of the raw diffs read for a range of commits: every file by itself, with
# NUL after each field, a rename as a deleted path and an added one, whole object ids,
# paths from the top of the repository, and a repository of its own listed whenever its
# commit differs
_RAW_DIFF_TUPLE = ("-r", "-z", "--raw", "--no-renames", "--no-abbrev", "--no-relative",
                   "--ignore-submodules=none")


class _PathChange(NamedTuple):
    """What a range of commits does to one path.

    An entry is (mode, object id) as a raw diff writes them, (_NO_MODE, the id of no
    object) where the path is not there.
    """
    start_entry: tuple[bytes, bytes]
    end_entry: tuple[bytes, bytes]
    base_list: tuple[tuple[bytes, bytes], ...]
    is_mode_change: bool


def _raw_group_list(output: bytes) -> list[list[tuple[bytes, ...]]]:
    """Read the entries of git's raw diff output written with -z.

    Args:
        output: What git wrote. A field that starts with _COMMIT_MARK starts the
            entries of another commit.

    Returns:
        The entries of each commit, in git's order, an entry being (old mode, new
        mode, old object id, new object id, path). One list for output with no mark.
    """
    group_list: list[list[tuple[bytes, ...]]] = [[]]
    field_list = output.split(b"\0")
    index = 0
    while index < len(field_list):
        field = field_list[index].lstrip(b"\n")
        index += 1
        if field.startswith(_COMMIT_MARK):
            group_list.append([])
        elif field.startswith(b":") and index < len(field_list):
            group_list[-1].append((*field[1:].split(b" ")[:4], field_list[index]))
            index += 1
    return [group for group in group_list if group]


def _path_change_dict(project_dir: str, start: str, commit: str
                      ) -> dict[bytes, _PathChange] | None:
    """Return what a range of commits does to each path it touches.

    Args:
        project_dir: The repository.
        start: The commit the range starts at.
        commit: The commit it ends at.

    Returns:
        {path as git writes it: _PathChange} for the paths the change from start to
        commit touches and the paths each commit from commit down its first parents,
        start not reached, touches against its first parent. base_list holds the
        entries the path is at before the end, each once, oldest first: its start
        entry, and the entry each of those commits found it at. is_mode_change is True
        where the change, or one of those commits, gives the path another mode, adding
        it included. None when git cannot list them, which is logged.
    """
    net_proc = run(project_dir, "diff-tree", *_RAW_DIFF_TUPLE, start, commit,
                   text=False)
    log_proc = run(project_dir, "log", "--first-parent", "--diff-merges=first-parent",
                   *_RAW_DIFF_TUPLE, f"--format={_COMMIT_MARK.decode()}%H",
                   f"{start}..{commit}", text=False)
    for proc in (net_proc, log_proc):
        if proc.returncode != 0:
            logger.warning("could not read what %s changes against %s: %s",
                           short(commit), short(start), shown_text(proc.stderr).strip())
            return None
    entry_dict = {path: ((old_mode, old_id), (new_mode, new_id))
                  for group in _raw_group_list(net_proc.stdout)
                  for old_mode, new_mode, old_id, new_id, path in group}
    touch_dict: dict[bytes, list[tuple[bytes, ...]]] = {}
    for group in _raw_group_list(log_proc.stdout):
        for entry in group:
            touch_dict.setdefault(entry[4], []).append(entry)
    change_dict = {}
    for path in dict.fromkeys([*entry_dict, *touch_dict]):
        touch_list = touch_dict.get(path, [])
        if path in entry_dict:
            start_entry, end_entry = entry_dict[path]
        else:
            # git log lists the newest commit first: its side is what commit holds of
            # a path the change as a whole leaves as it was
            start_entry = end_entry = (touch_list[0][1], touch_list[0][3])
        change_dict[path] = _PathChange(
            start_entry, end_entry,
            tuple(dict.fromkeys([start_entry, *((entry[0], entry[2])
                                                for entry in reversed(touch_list))])),
            start_entry[0] != end_entry[0]
            or any(entry[0] != entry[1] for entry in touch_list))
    return change_dict


def _has_path_in_tree(project_dir: str, change_dict: dict[bytes, _PathChange]) -> bool:
    """Say whether the working tree holds each path as a change leaves it.

    A path is held when the working tree holds it as the change's end does: the same
    kind of entry, nothing for a path the change takes away, a link to the same place,
    a repository of its own whose HEAD is the commit the change names, and a file
    whose contents git reads as the same object, through the conversions the
    attributes and core.autocrlf name and with no filter driver run
    (own_driver_tuple). A file's executable bit is compared where the change gives
    the file another mode and core.fileMode is not false. A file whose contents
    differ, and that is a file at every entry of its base_list, is held when it holds
    the change from each of them to the end with changes of the working tree's own
    around it (_has_change_by_merge). A directory on the way to a path that is a
    link, and an entry that cannot be read, are not held.

    Args:
        project_dir: The repository, whose working tree is read.
        change_dict: _path_change_dict()'s return value.

    Returns:
        True when every path is held, and when there is none.
    """
    top = os.fsencode(project_dir)
    compare_bit = run(project_dir, "config", "--type=bool", "--get",
                      "core.fileMode").stdout.strip() != "false"
    file_dict, link_dict = {}, {}
    for path, change in change_dict.items():
        mode, object_id = change.end_entry
        try:
            entry_stat = _tree_lstat(top, path)
        except OSError:
            return False
        if mode == _NO_MODE:
            is_held = entry_stat is None
        elif entry_stat is None:
            is_held = False
        elif mode == GITLINK_MODE:
            is_held = stat.S_ISDIR(entry_stat.st_mode) \
                and _repo_head(os.path.join(top, path)) == object_id
        elif mode == SYMLINK_MODE:
            is_held = stat.S_ISLNK(entry_stat.st_mode)
            link_dict[path] = object_id
        else:
            is_held = mode in _FILE_MODE_TUPLE and stat.S_ISREG(entry_stat.st_mode) \
                and not (compare_bit and change.is_mode_change and bool(
                    entry_stat.st_mode & stat.S_IXUSR) != (mode == _FILE_MODE_TUPLE[1]))
            file_dict[path] = change
        if not is_held:
            return False
    try:
        target_list = _blob_list(project_dir, list(link_dict.values()))
        if target_list is None or any(os.readlink(os.path.join(top, path)) != target
                                      for path, target in zip(link_dict, target_list)):
            return False
        id_list = _tree_id_list(project_dir, list(file_dict))
    except (OSError, RuntimeError) as e:
        logger.info("could not read the working tree of %s: %s", project_dir, e)
        return False
    if id_list is None:
        return False
    merge_dict = {path: change for (path, change), object_id in zip(file_dict.items(),
                                                                    id_list)
                  if object_id != change.end_entry[1]}
    if not all(mode in _FILE_MODE_TUPLE for change in merge_dict.values()
               for mode, _ in change.base_list):
        return False
    return not merge_dict or _has_change_by_merge(project_dir, merge_dict)


def _tree_lstat(top: bytes, path: bytes) -> os.stat_result | None:
    """Return what the working tree holds at a path, without following a link.

    Args:
        top: The top of the working tree.
        path: The path under it, as git writes it.

    Returns:
        The entry's status, or None when nothing is there, a file standing where a
        directory on the way would be included.

    Raises:
        OSError: When the entry cannot be read, or a directory on the way is a link.
    """
    directory = top
    part_list = path.split(b"/")
    for part in part_list[:-1]:
        directory = os.path.join(directory, part)
        try:
            mode = os.lstat(directory).st_mode
        except (FileNotFoundError, NotADirectoryError):
            return None
        if stat.S_ISLNK(mode):
            raise OSError(f"{os.fsdecode(directory)} is a link")
        if not stat.S_ISDIR(mode):
            return None
    try:
        return os.lstat(os.path.join(directory, part_list[-1]))
    except (FileNotFoundError, NotADirectoryError):
        return None


def _repo_head(directory: bytes) -> bytes:
    """Return the commit HEAD names in a repository of its own in the working tree.

    Args:
        directory: The directory, as an absolute path.

    Returns:
        The full hash, or b"" when the directory is not the top of a repository or
        git cannot tell.
    """
    proc = run(os.fsdecode(directory), "rev-parse", "--show-toplevel", "HEAD",
               text=False)
    line_list = [line for line in proc.stdout.split(b"\n") if line]
    if proc.returncode != 0 or len(line_list) != 2 \
            or os.path.realpath(line_list[0]) != os.path.realpath(directory):
        return b""
    return line_list[1]


def _blob_size_list(project_dir: str, id_list: list[bytes]) -> list[int] | None:
    """Return the sizes of objects, read through one git cat-file --batch-check.

    Args:
        project_dir: The repository.
        id_list: The objects' ids.

    Returns:
        The sizes in bytes, in the order of id_list, or None when git does not give
        one.
    """
    if not id_list:
        return []
    proc = run(project_dir, "cat-file", "--batch-check", text=False,
               input_text=b"".join(object_id + b"\n" for object_id in id_list))
    size_list = []
    for line in proc.stdout.split(b"\n")[:len(id_list)]:
        head = line.split(b" ")
        if len(head) != 3 or not head[2].isdigit():
            return None
        size_list.append(int(head[2]))
    return size_list if proc.returncode == 0 and len(size_list) == len(id_list) else None


def _blob_list(project_dir: str, id_list: list[bytes]) -> list[bytes] | None:
    """Return the contents of objects, read through one git cat-file --batch.

    Args:
        project_dir: The repository.
        id_list: The objects' ids.

    Returns:
        The contents, in the order of id_list, or None when git does not give one as
        a blob, and when one is larger than tmpfile.READ_MAX_BYTE.
    """
    if not id_list:
        return []
    size_list = _blob_size_list(project_dir, id_list)
    if size_list is None or any(size > tmpfile.READ_MAX_BYTE for size in size_list):
        return None
    proc = run(project_dir, "cat-file", "--batch", text=False,
               input_text=b"".join(object_id + b"\n" for object_id in id_list),
               max_address_byte=MAX_PATCH_ADDRESS_BYTE)
    content_list, place = [], 0
    for _ in id_list:
        end = proc.stdout.find(b"\n", place)
        head = proc.stdout[place:end].split(b" ") if end >= 0 else []
        if len(head) != 3 or head[1] != b"blob" or not head[2].isdigit():
            return None
        place = end + 1 + int(head[2])
        content_list.append(proc.stdout[end + 1:place])
        place += 1
    return content_list if proc.returncode == 0 else None


def _quote_path(path: bytes) -> bytes:
    r"""Write a path in git's C-style quotes, which git reads back byte for byte.

    Examples:
        a"b, then a line feed -> "a\"b\012"

    Args:
        path: The path, as git writes it.

    Returns:
        The quoted path.
    """
    return b'"' + b"".join(b"\\" + bytes([byte]) if byte in b'"\\'
                           else b"\\%03o" % byte if byte < 0x20 or byte >= 0x7f
                           else bytes([byte]) for byte in path) + b'"'


def _tree_id_list(project_dir: str, path_list: list[bytes]) -> list[bytes] | None:
    """Return the object ids git gives the working tree's files, as it would store them.

    Each file is read through the conversions the attributes and core.autocrlf name
    for its path, with no filter driver run (own_driver_tuple). Nothing is written.

    Args:
        project_dir: The repository, whose working tree is read.
        path_list: The files, as git writes their paths.

    Returns:
        The ids, in the order of path_list, or None when git did not read them all.

    Raises:
        RuntimeError: When the git settings could not be read.
    """
    if not path_list:
        return []
    option_list, env_dict = own_driver_tuple(project_dir)
    proc = run(project_dir, *option_list, "hash-object", "--stdin-paths",
               env_dict=env_dict, text=False,
               input_text=b"".join(_quote_path(path) + b"\n" for path in path_list))
    id_list = proc.stdout.split()
    return id_list if proc.returncode == 0 and len(id_list) == len(path_list) else None


def _has_change_by_merge(project_dir: str, change_dict: dict[bytes, _PathChange]
                         ) -> bool:
    """Say whether the working tree's files hold changes among changes of their own.

    For each file and each entry of its base_list, the change from that entry to the
    end is merged into the working tree's file as git merge-file merges it with its
    default settings, both versions read as git cat-file --filters writes them into
    the working tree at the path, with no filter driver run (own_driver_tuple). The
    file holds the change when every such merge finds no conflict and gives the file
    as it is: what the end holds is in it, and so is the taking away of what a version
    on the way held and the end does not. A file or a version larger than
    tmpfile.READ_MAX_BYTE is not read, and does not hold its change; neither does a
    file whose merge git merge-file refuses as binary (_is_binary).

    The versions are written through one git cat-file (_write_side_list), the changes
    of every merge are read from one git diff (_pair_script_dict), and each merge is
    worked out from them (_is_merge_same).

    Args:
        project_dir: The repository, whose working tree is read.
        change_dict: {path: _PathChange} for files whose end and every entry of whose
            base_list are files.

    Returns:
        True when every file holds its change. False when one does not, a binary file
        included, and when one cannot be read.
    """
    top = os.fsencode(project_dir)
    size_list = _blob_size_list(project_dir, [
        object_id for change in change_dict.values()
        for _, object_id in (change.end_entry, *change.base_list)])
    if size_list is None or any(size > tmpfile.READ_MAX_BYTE for size in size_list):
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="ctloom-held-") as directory:
            # == Step 1: the working tree's files and every version of them ==========
            side_list = []
            for path_index, (path, change) in enumerate(change_dict.items()):
                tree_path = os.path.join(top, path)
                if os.lstat(tree_path).st_size > tmpfile.READ_MAX_BYTE:
                    return False
                shutil.copyfile(tree_path, os.path.join(directory, f"t{path_index}"))
                side_list += [(object_id, path,
                               os.path.join(directory, f"v{path_index}-{index}"))
                              for index, (_, object_id) in enumerate((change.end_entry,
                                                                      *change.base_list))]
            if not _write_side_list(project_dir, side_list):
                return False

            # == Step 2: the changes of each merge, from one git diff ================
            # Pairs 2n and 2n+1 compare the n-th entry of the base_lists, counted over
            # every file in turn, with the working tree's file and with the end
            pair_dir_list = [os.path.join(directory, side) for side in ("a", "b")]
            for pair_dir in pair_dir_list:
                os.mkdir(pair_dir)
            pair_count = 0
            for path_index, change in enumerate(change_dict.values()):
                for index in range(1, 1 + len(change.base_list)):
                    for other in (f"t{path_index}", f"v{path_index}-0"):
                        os.link(os.path.join(directory, f"v{path_index}-{index}"),
                                os.path.join(pair_dir_list[0], str(pair_count)))
                        os.link(os.path.join(directory, other),
                                os.path.join(pair_dir_list[1], str(pair_count)))
                        pair_count += 1
            script_dict = _pair_script_dict(directory)
            if script_dict is None:
                return False

            # == Step 3: each merge ==================================================
            pair_count = 0
            for path_index, change in enumerate(change_dict.values()):
                content_list = []
                for name in (f"t{path_index}", f"v{path_index}-0"):
                    with open(os.path.join(directory, name), "rb") as handle:
                        content_list.append(handle.read())
                if any(_is_binary(content) for content in content_list):
                    return False
                tree_list, end_list = (_record_list(content) for content in content_list)
                for index in range(1, 1 + len(change.base_list)):
                    with open(os.path.join(directory, f"v{path_index}-{index}"),
                              "rb") as handle:
                        if _is_binary(handle.read(_BINARY_CHECK_BYTE)):
                            return False
                    if not _is_merge_same(tree_list, end_list,
                                          script_dict.get(pair_count, []),
                                          script_dict.get(pair_count + 1, [])):
                        return False
                    pair_count += 2
    except (OSError, RuntimeError, ValueError) as e:
        logger.info("could not merge into the working tree of %s: %s", project_dir, e)
        return False
    return True


# The first bytes of a file git merge-file looks for a NUL in, and refuses to merge the
# file as binary when it finds one
_BINARY_CHECK_BYTE = 8000


def _is_binary(content: bytes) -> bool:
    """Say whether git merge-file refuses to merge a file as binary.

    Args:
        content: The file's contents, or their first _BINARY_CHECK_BYTE bytes.

    Returns:
        True when there is a NUL in the first _BINARY_CHECK_BYTE bytes.
    """
    return b"\0" in content[:_BINARY_CHECK_BYTE]


def _record_list(content: bytes) -> list[bytes]:
    """Split a file into the lines git's diff and merge compare.

    Examples:
        b"a\\nb\\n" -> [b"a\\n", b"b\\n"]
        b"a\\r\\nb"  -> [b"a\\r\\n", b"b"]

    Args:
        content: The file's contents.

    Returns:
        Each line with the line feed that ends it, the last one without when the
        file does not end with one.
    """
    part_list = content.split(b"\n")
    return [part + b"\n" for part in part_list[:-1]] + ([part_list[-1]]
                                                        if part_list[-1] else [])


# A path git cat-file --batch reads back from its line of input as it is written: no
# space or tab at its start, no line feed in it and no carriage return at its end
_BATCH_PATH_RE = re.compile(rb"[^ \t\n][^\n]*(?<!\r)")


def _write_side_list(project_dir: str,
                     side_list: list[tuple[bytes, bytes, str]]) -> bool:
    """Write objects into files as git cat-file --filters writes them at a path.

    Each object is written as git writes it into the working tree at its path, with no
    filter driver run (own_driver_tuple). The objects are read through one git cat-file
    --batch --filters, each followed by the name of no object, made up for the call,
    that ends what git writes of it. An object whose path that command cannot read back
    as it is written is read through a git cat-file --filters of its own.

    Args:
        project_dir: The repository.
        side_list: (object id, path as git writes it, the file it is written to, as an
            absolute path) for each object.

    Returns:
        True when every object was written. False when git did not give one as a blob.

    Raises:
        RuntimeError: When the git settings could not be read.
        OSError: When a file cannot be written.
    """
    option_list, env_dict = own_driver_tuple(project_dir)
    batch_list = [side for side in side_list if _BATCH_PATH_RE.fullmatch(side[1])]
    for object_id, path, file_path in side_list:
        if _BATCH_PATH_RE.fullmatch(path):
            continue
        proc = run(project_dir, *option_list, "cat-file", "--filters",
                   f"--path={os.fsdecode(path)}", object_id.decode(),
                   env_dict=env_dict, text=False)
        if proc.returncode != 0:
            return False
        with open(file_path, "wb") as handle:
            handle.write(proc.stdout)
    if not batch_list:
        return True
    end_name = f"ctloom-end-{os.urandom(16).hex()}".encode()
    end_mark = b"\n" + end_name + b" missing\n"
    argv = _git_argv(project_dir, *option_list, "cat-file", "--batch", "--filters")
    with tempfile.TemporaryFile() as request_file:
        request_file.write(b"".join(object_id + b" " + path + b"\n" + end_name + b" -\n"
                                    for object_id, path, _ in batch_list))
        request_file.seek(0)
        with subprocess.Popen(argv, stdin=request_file, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL,
                              env=config.child_env(env_dict)) as proc:
            try:
                is_write = _read_side_stream(proc.stdout,
                                             [side[2] for side in batch_list], end_mark)
            finally:
                if proc.poll() is None:
                    proc.kill()
    return is_write and proc.returncode == 0


def _read_side_stream(stream: IO[bytes], file_path_list: list[str],
                      end_mark: bytes) -> bool:
    """Read what git cat-file --batch writes of blobs into files, one blob a file.

    Args:
        stream: git's standard output.
        file_path_list: The files, in the order the blobs were asked for.
        end_mark: What git writes after each blob's contents.

    Returns:
        True when each blob was read up to its end_mark. False when git wrote
        anything else.

    Raises:
        OSError: When a file cannot be written.
    """
    rest = b""
    for file_path in file_path_list:
        while b"\n" not in rest:
            block = stream.read1(_READ_BLOCK_BYTE)
            if not block:
                return False
            rest += block
        head, rest = rest.split(b"\n", 1)
        if head.split(b" ")[1:2] != [b"blob"]:
            return False
        with open(file_path, "wb") as handle:
            while (end := rest.find(end_mark)) < 0:
                keep = max(len(rest) - len(end_mark) + 1, 0)
                handle.write(rest[:keep])
                rest = rest[keep:]
                block = stream.read1(_READ_BLOCK_BYTE)
                if not block:
                    return False
                rest += block
            handle.write(rest[:end])
            rest = rest[end + len(end_mark):]
    return not rest and not stream.read(1)


# The options of the git diff --no-index that compares pairs of files: every pair
# compared as the bytes they hold, as text, by the diff git merge-file makes of its files
# (Myers, no indent heuristic), with one line of context, no conversion, no attribute
# read and fixed prefixes
_PAIR_DIFF_TUPLE = ("-c", "core.autocrlf=false", "-c", f"core.attributesFile={os.devnull}",
                    "-c", "diff.suppressBlankEmpty=false", "diff", "--no-index",
                    "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames",
                    "--text", "--diff-algorithm=myers", "--no-indent-heuristic",
                    "--unified=1", "--src-prefix=a/", "--dst-prefix=b/")

# The head of a hunk of a unified diff: the first line and the number of lines on each
# side, a number left out being 1
_HUNK_HEAD_RE = re.compile(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _pair_script_dict(directory: str) -> dict[int, list[tuple[int, int, int, int]]] | None:
    """Return the changes each pair of files of a directory makes, as git's diff does.

    The files of directory/a are compared with the files of the same name in
    directory/b through one git diff --no-index (_PAIR_DIFF_TUPLE), run in directory
    with no repository above it read and no system attributes. Its output is written
    to directory/pair.diff and read a line at a time.

    Args:
        directory: The directory, as an absolute path. Its files are named by number.

    Returns:
        {the number a pair is named by: [(the first line of a change in the first
        file, how many lines of it the change takes away, the first line of the change
        in the second file, how many lines it puts there), ...] in the files' order,
        lines counted from 0}. A pair of the same files is not in it. None when git
        does not compare them.
    """
    diff_path = os.path.join(directory, "pair.diff")
    proc = run(directory, *_PAIR_DIFF_TUPLE, f"--output={diff_path}", "a", "b",
               env_dict={"GIT_CEILING_DIRECTORIES": os.path.dirname(directory),
                         "GIT_ATTR_NOSYSTEM": "1"})
    if proc.returncode not in (0, 1):
        logger.info("could not compare the versions in %s: %s", directory,
                    proc.stderr.strip())
        return None
    script_dict: dict[int, list[tuple[int, int, int, int]]] = {}
    script_list: list[tuple[int, int, int, int]] = []
    with open(diff_path, "rb") as handle:
        for line in handle:
            # A file's part of the diff starts with a _PATCH_FILE_HEAD line, which ends
            # with the file's name after the last "/"
            if line.startswith(_PATCH_FILE_HEAD):
                script_list = script_dict.setdefault(
                    int(line.rstrip(b"\n").rsplit(b"/", 1)[1]), [])
                continue
            head = _HUNK_HEAD_RE.match(line)
            if head and not _read_hunk(handle, head, script_list):
                return None
    return script_dict


def _read_hunk(handle: IO[bytes], head: re.Match[bytes],
               script_list: list[tuple[int, int, int, int]]) -> bool:
    """Read the lines of one hunk, adding the changes it holds to a list.

    Args:
        handle: The diff, open for reading in binary mode, at the hunk's first line.
        head: _HUNK_HEAD_RE's match of the hunk's head.
        script_list: The changes of the hunk's pair of files, which this adds to as
            _pair_script_dict() returns them.

    Returns:
        True when the hunk held as many lines as its head says.
    """
    old_count, new_count = int(head.group(2) or b"1"), int(head.group(4) or b"1")
    old = int(head.group(1)) - (1 if old_count else 0)
    new = int(head.group(3)) - (1 if new_count else 0)
    change = None
    while old_count or new_count:
        kind = handle.readline()[:1]
        if kind == b"\\":
            continue
        if kind not in (b" ", b"-", b"+"):
            return False
        if kind == b" ":
            if not old_count or not new_count:
                return False
            if change:
                script_list.append(tuple(change))
                change = None
            old, new, old_count, new_count = old + 1, new + 1, old_count - 1, new_count - 1
            continue
        change = change or [old, 0, new, 0]
        if kind == b"-":
            if not old_count:
                return False
            change[1] += 1
            old, old_count = old + 1, old_count - 1
        else:
            if not new_count:
                return False
            change[3] += 1
            new, new_count = new + 1, new_count - 1
    if change:
        script_list.append(tuple(change))
    return True


# The kinds of a part of a merge, as git merge-file works them out: a conflict, a change
# only the working tree's file makes, and a change only the end makes
_MERGE_CONFLICT, _MERGE_TREE, _MERGE_END = 0, 1, 2


def _is_merge_same(tree_list: list[bytes], end_list: list[bytes],
                   tree_script: list[tuple[int, int, int, int]],
                   end_script: list[tuple[int, int, int, int]]) -> bool:
    """Say whether git merge-file finds no conflict and gives back the working tree's file.

    git merge-file merges the change from a version to the end into the working tree's
    file with its default settings. The changes of the two sides are joined where they
    touch or overlap as git merge-file joins them. A part both sides change the same way
    is the working tree's; a part they change in different ways is a conflict unless
    both sides hold the same lines there, and a part only the end changes is the end's.

    Args:
        tree_list: The working tree's file (_record_list).
        end_list: The end (_record_list).
        tree_script: The changes from the version to the working tree's file, as
            _pair_script_dict() gives them.
        end_script: The changes from the version to the end.

    Returns:
        True when the merge finds no conflict and gives tree_list.
    """
    if not tree_script:
        return end_list == tree_list
    if not end_script:
        return True
    part_list: list[list[int]] = []

    def add_part(kind: int, tree_start: int, tree_count: int, end_start: int,
                 end_count: int) -> None:
        """Add one part, joined to the last when they touch or overlap."""
        if part_list and (tree_start <= part_list[-1][1] + part_list[-1][2]
                          or end_start <= part_list[-1][3] + part_list[-1][4]):
            last_part = part_list[-1]
            last_part[0] = last_part[0] if last_part[0] == kind else _MERGE_CONFLICT
            last_part[2] = tree_start + tree_count - last_part[1]
            last_part[4] = end_start + end_count - last_part[3]
            return
        part_list.append([kind, tree_start, tree_count, end_start, end_count])

    # == Step 1: the parts, in the version's order ===========================
    tree_index = end_index = 0
    while tree_index < len(tree_script) and end_index < len(end_script):
        tree_change, end_change = tree_script[tree_index], end_script[end_index]
        if tree_change[0] + tree_change[1] < end_change[0]:
            add_part(_MERGE_TREE, tree_change[2], tree_change[3],
                     end_change[2] - end_change[0] + tree_change[0], tree_change[1])
            tree_index += 1
            continue
        if end_change[0] + end_change[1] < tree_change[0]:
            add_part(_MERGE_END, tree_change[2] - tree_change[0] + end_change[0],
                     end_change[1], end_change[2], end_change[3])
            end_index += 1
            continue
        if tree_change[:2] != end_change[:2] or tree_change[3] != end_change[3] \
                or tree_list[tree_change[2]:tree_change[2] + tree_change[3]] \
                != end_list[end_change[2]:end_change[2] + end_change[3]]:
            gap = tree_change[0] - end_change[0]
            end_gap = gap + tree_change[1] - end_change[1]
            tree_start, end_start = tree_change[2], end_change[2]
            if gap > 0:
                tree_start -= gap
            else:
                end_start += gap
            tree_count = tree_change[2] + tree_change[3] - tree_start
            end_count = end_change[2] + end_change[3] - end_start
            if end_gap < 0:
                tree_count -= end_gap
            else:
                end_count += end_gap
            add_part(_MERGE_CONFLICT, tree_start, tree_count, end_start, end_count)
        tree_stop = tree_change[0] + tree_change[1]
        end_stop = end_change[0] + end_change[1]
        if tree_stop >= end_stop:
            end_index += 1
        if end_stop >= tree_stop:
            tree_index += 1
    tree_shift = sum(change[3] - change[1] for change in tree_script)
    end_shift = sum(change[3] - change[1] for change in end_script)
    for tree_change in tree_script[tree_index:]:
        add_part(_MERGE_TREE, tree_change[2], tree_change[3],
                 tree_change[0] + end_shift, tree_change[1])
    for end_change in end_script[end_index:]:
        add_part(_MERGE_END, end_change[0] + tree_shift, end_change[1], end_change[2],
                 end_change[3])

    # == Step 2: what the merge gives ========================================
    merge_list, place = [], 0
    for kind, tree_start, tree_count, end_start, end_count in part_list:
        tree_part = tree_list[tree_start:tree_start + tree_count]
        if kind == _MERGE_CONFLICT:
            if not tree_count or not end_count \
                    or tree_part != end_list[end_start:end_start + end_count]:
                return False
            continue
        merge_list += tree_list[place:tree_start]
        merge_list += tree_part if kind == _MERGE_TREE \
            else end_list[end_start:end_start + end_count]
        place = tree_start + tree_count
    return merge_list + tree_list[place:] == tree_list


def root_commit_list(project_dir: str) -> list[str]:
    """Return the commits HEAD's history starts from.

    Args:
        project_dir: The repository. An empty path gives [].

    Returns:
        The full hashes of the commits with no parent that HEAD reaches, sorted. [] for a
        directory that is not a repository and one with no commit in it.
    """
    if not project_dir:
        return []
    proc = run(project_dir, "rev-list", "--max-parents=0", "HEAD")
    if proc.returncode != 0:
        return []
    return sorted(proc.stdout.split())


def has_commit(project_dir: str, commit: str) -> bool:
    """Say whether a repository holds a commit.

    Args:
        project_dir: The repository.
        commit: A full commit hash.

    Returns:
        True when the object is there and is a commit.
    """
    return run(project_dir, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0


def head_commit(project_dir: str) -> str:
    """Return the commit a repository is checked out at, or "" when it cannot be told.

    An empty string comes back for a directory that is not a repository and one with no
    commit in it.

    Args:
        project_dir: The repository. An empty path gives "".

    Returns:
        The full commit hash, or "".

    Raises:
        FileNotFoundError: When git cannot be started (no git on PATH).
    """
    if not project_dir:
        return ""
    proc = run(project_dir, "rev-parse", "HEAD")
    if proc.returncode != 0:
        logger.info("no commit for %s: %s", project_dir, proc.stderr.strip())
        return ""
    return proc.stdout.strip()


def clean_head(project_dir: str) -> str:
    """Return the commit whose tree is what is on disk, or "" when there is none.

    A working tree with changes git reports (modified, added, deleted or untracked
    paths) comes back as "".

    Args:
        project_dir: The repository. An empty path gives "".

    Returns:
        The full commit hash, or "" when the tree differs from HEAD or HEAD cannot be
        told (see head_commit).

    Raises:
        FileNotFoundError: When git cannot be started (no git on PATH).
    """
    commit = head_commit(project_dir)
    if not commit:
        return ""
    change_list = status_path_list(project_dir)
    if change_list:
        logger.info("%s has %d path(s) not committed, so the analysis describes no "
                    "commit", project_dir, len(change_list))
        return ""
    return commit


def version(project_dir: str) -> tuple[int, ...]:
    """Return the version of the git this machine runs.

    Args:
        project_dir: A repository to ask in, or any directory git runs in.

    Returns:
        The numbers of "git version X.Y.Z", e.g. (2, 43, 0), or () when git fails or
        answers something else.

    Raises:
        FileNotFoundError: When git cannot be started (no git on PATH), and when
            project_dir is empty (see run()).
    """
    proc = run(project_dir, "version")
    if proc.returncode != 0:
        return ()
    number_list = re.findall(r"\d+", proc.stdout)
    return tuple(int(number) for number in number_list[:3])


def add_exclude(project_dir: str, pattern_list: list[str]) -> list[str]:
    """Add patterns to the exclude file of a repository's own git directory.

    What they match is left out of everything git lists as new in that repository, as
    a .gitignore would leave it out, and no file of the working tree is written.

    Args:
        project_dir: The repository.
        pattern_list: .gitignore patterns, one per entry.

    Returns:
        The patterns that were added, in their order: the ones the file did not hold
        yet. Empty for a directory that is not a repository.

    Raises:
        OSError: When the file cannot be read or written.
    """
    proc = run(project_dir, "rev-parse", "--git-path", "info/exclude")
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    exclude_path = os.path.join(project_dir, proc.stdout.strip())
    try:
        with open(exclude_path, encoding="utf-8", errors="replace",
                  newline="") as handle:
            text = handle.read()
    except FileNotFoundError:
        text = ""
    have_set = set(line_list_of(text))
    add_list = [pattern for pattern in dict.fromkeys(pattern_list)
                if pattern not in have_set]
    if add_list:
        os.makedirs(os.path.dirname(exclude_path), exist_ok=True)
        # A file whose last line is not ended is ended first
        head = "\n" if text and not text.endswith("\n") else ""
        with open(exclude_path, "a", encoding="utf-8") as handle:
            handle.write(head + "".join(f"{pattern}\n" for pattern in add_list))
    return add_list


# What git's check-ignore ends with when a rule covers one of the paths it was asked
# for, and when none does
IGNORE_STATUS = 0
NO_IGNORE_STATUS = 1


def check_ignore(project_dir: str, raw_path_list: list[bytes], rule_dir: str = ""
                 ) -> subprocess.CompletedProcess:
    """Ask git which of the paths its ignore rules cover, whether or not it tracks them.

    Each path is handed over on the standard input as "./<path>", which git reads as
    that path and not as a pattern or a pathspec with magic.

    Args:
        project_dir: The repository the paths are in.
        raw_path_list: Paths relative to the repository, as bytes.
        rule_dir: A directory whose .gitignore files are read in place of the working
            tree's, as an absolute path. Empty reads the working tree's.

    Returns:
        The finished process, its output as bytes: the paths a rule covers, each as
        "./<path>" and ended with a NUL. It ends with IGNORE_STATUS, NO_IGNORE_STATUS,
        or another status when git gives no answer.
    """
    return run(project_dir, *([f"--work-tree={rule_dir}"] if rule_dir else []),
               "check-ignore", "--no-index", "-z", "--stdin", text=False,
               input_text=b"".join(b"./" + raw + b"\0" for raw in raw_path_list))


def is_ignored(project_dir: str, path: str) -> bool:
    """Say whether git leaves a path out of everything it carries.

    The path is asked for as it is written (check_ignore()), and one the index names is
    carried whatever rule covers it.

    Args:
        project_dir: The repository the path is in.
        path: The path, relative to project_dir.

    Returns:
        True when a rule covers the path and the index does not name it. False when
        git gives no answer; a directory that is not a repository ignores nothing.
    """
    if check_ignore(project_dir, [os.fsencode(path)]).returncode != IGNORE_STATUS:
        return False
    return run(project_dir, "ls-files", "--error-unmatch", "--",
               f":(literal){path}").returncode != 0


# The mode git's index and its trees give an entry that is another repository's commit
GITLINK_MODE = b"160000"

# The mode they give an entry that is a symbolic link
SYMLINK_MODE = b"120000"


def nest_dir_of(project_dir: str, raw_path: bytes,
                link_set: Collection[bytes]) -> bytes:
    """Return the outermost directory holding a path that is a repository of its own.

    A directory above the path is one when link_set names it or it holds a .git of its
    own on disk, as a file or as a directory. The path itself is one when link_set
    names it.

    Examples:
        (b"sub/pkg/a.py", {b"sub"})             -> b"sub"
        (b"sub", {b"sub"})                      -> b"sub"
        (b"nest/a.py", with nest/.git on disk)  -> b"nest"
        (b"a.py", {b"sub"})                     -> b""

    Args:
        project_dir: The repository the path is in.
        raw_path: The path, relative to project_dir, as bytes.
        link_set: The paths an index or a tree holds with GITLINK_MODE.

    Returns:
        That directory, or the path itself, relative to project_dir. b"" for a path
        under none.
    """
    part_list = raw_path.split(b"/")
    root = os.fsencode(project_dir)
    for part_count in range(1, len(part_list) + 1):
        above = b"/".join(part_list[:part_count])
        if above in link_set or (part_count < len(part_list) and os.path.lexists(
                os.path.join(root, above, b".git"))):
            return above
    return b""


def gitlink_of(project_dir: str, path: str) -> str:
    """Return the directory above a path that is a repository of its own, or the path.

    The path itself comes back when it is a submodule. git reports such a directory as
    one entry and carries nothing of what lies under it: a submodule, whether or not its
    files are checked out, and a directory holding a .git of its own (nest_dir_of).

    Examples:
        ("sub/pkg/a.py", with the submodule "sub") -> "sub"
        ("sub", with the submodule "sub")          -> "sub"
        ("a.py")                                   -> ""

    Args:
        project_dir: The repository the path is in.
        path: The path, relative to project_dir.

    Returns:
        The outermost such directory, relative to project_dir, or "" when the path is
        none and lies under none.
    """
    part_list = [part for part in path.split("/") if part not in ("", os.curdir)]
    # The path and every directory above it, the one nearest the repository root first
    name_list = ["/".join(part_list[:part_count])
                 for part_count in range(1, len(part_list) + 1)]
    if not name_list:
        return ""
    # Of the entries git lists, only one whose path is one of the names counts
    proc = run(project_dir, "ls-files", "--stage", "-z", "--",
               *(f":(literal){name}" for name in name_list), text=False)
    link_set = {entry.partition(b"\t")[2] for entry in proc.stdout.split(b"\0")
                if entry.startswith(GITLINK_MODE)}
    link_set &= {os.fsencode(name) for name in name_list}
    return os.fsdecode(nest_dir_of(project_dir, os.fsencode(path), link_set))


# What a patch starts the header of a file with, the starts of the lines git apply reads
# as the rest of that header, the starts of the ones among them a header that moves a
# repository of its own holds, the line of the header that gives GITLINK_MODE to both
# sides of the file, the line that names the file's new path, the head of a hunk with
# the numbers of lines it takes away and leaves, and the lines of a hunk that name the
# commit it moves a repository of its own from and to by its full id (40 or 64 hex
# digits)
_PATCH_FILE_HEAD = b"diff --git "
_PATCH_HEAD_LINE_TUPLE = (b"--- ", b"+++ ", b"old mode ", b"new mode ",
                          b"deleted file mode ", b"new file mode ", b"copy from ",
                          b"copy to ", b"rename old ", b"rename new ", b"rename from ",
                          b"rename to ", b"similarity index ", b"dissimilarity index ",
                          b"index ")
_PATCH_MOVE_HEAD_LINE_TUPLE = (b"--- ", b"+++ ", b"index ")
_PATCH_LINK_INDEX = re.compile(rb"index [0-9a-f]+\.\.[0-9a-f]+ " + GITLINK_MODE)
_PATCH_NEW_PATH_HEAD = b"+++ "
_PATCH_HUNK_HEAD = re.compile(rb"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_PATCH_LINK_FROM = re.compile(rb"-Subproject commit ([0-9a-f]{40}|[0-9a-f]{64})")
_PATCH_LINK_COMMIT = re.compile(rb"\+Subproject commit ([0-9a-f]{40}|[0-9a-f]{64})")

# A path of a "diff --git" line that git quotes, from its opening double quote to its
# closing one
_PATCH_QUOTE_NAME = re.compile(rb'"(?:[^"\\]|\\.)*"', re.DOTALL)

# The longest line of a patch read whole, in bytes. A longer line is read past in parts
# of this size and read as an empty line
_MAX_PATCH_LINE_BYTE = 65536


def submodule_move_list(patch_path: str, link_set: Collection[bytes]
                        ) -> list[tuple[bytes, str, str]]:
    """Return the repositories of their own whose commit a patch moves.

    Each file of the patch whose "diff --git" line starts a file that moves a
    repository of its own (_patch_link_move) is one, when link_set names its path. A
    path two such files of the patch move is left out, and so is a path with an empty
    part, "." or ".." in it.

    Args:
        patch_path: The patch, as git diff writes it.
        link_set: The paths, relative to the repository, as bytes, that are read.

    Returns:
        (the path relative to the repository, as bytes, the id of the commit the patch
        moves it from, the id of the one it moves it to) for each such file, in the
        order of the patch. Empty when the patch cannot be read, which is logged.
    """
    # Each path read, with the two commits, or None once a second file moves it
    link_move_dict: dict[bytes, tuple[str, str] | None] = {}
    try:
        with open(patch_path, "rb") as patch_file:
            line_iter = _patch_line_iter(patch_file)
            line = next(line_iter, None)
            while line is not None:
                if not line.startswith(_PATCH_FILE_HEAD):
                    line = next(line_iter, None)
                    continue
                move, line = _patch_link_move(line[len(_PATCH_FILE_HEAD):], line_iter)
                if move and move[0] in link_set and all(
                        part not in (b"", b".", b"..") for part in move[0].split(b"/")):
                    link_move_dict[move[0]] = (None if move[0] in link_move_dict
                                               else move[1:])
    except OSError as error:
        logger.info("could not read the patch %s: %s", patch_path, error)
        return []
    return [(path, *commit_tuple) for path, commit_tuple in link_move_dict.items()
            if commit_tuple]


def _patch_line_iter(patch_file: IO[bytes]) -> Iterator[bytes]:
    """Yield the lines of a patch without their newline.

    No more than _MAX_PATCH_LINE_BYTE bytes of a line are held at once. A longer line
    is read past and yielded as b"".

    Args:
        patch_file: The patch, open for reading in binary mode.

    Yields:
        Each line, in order.
    """
    while line := patch_file.readline(_MAX_PATCH_LINE_BYTE + 1):
        if line.endswith(b"\n"):
            yield line[:-1]
            continue
        if len(line) <= _MAX_PATCH_LINE_BYTE:
            yield line
            return
        while line and not line.endswith(b"\n"):
            line = patch_file.readline(_MAX_PATCH_LINE_BYTE)
        yield b""


def _patch_link_move(head_name: bytes, line_iter: Iterator[bytes]
                     ) -> tuple[tuple[bytes, str, str] | None, bytes | None]:
    """Read one file of a patch as the move of a repository of its own.

    The file is read from the line after its "diff --git" line. It is one when the lines
    git apply reads as its header (_PATCH_HEAD_LINE_TUPLE) are all of
    _PATCH_MOVE_HEAD_LINE_TUPLE and hold _PATCH_LINK_INDEX and a _PATCH_NEW_PATH_HEAD
    line whose path both names of the "diff --git" line give (_is_head_of), and the
    lines after them are the file's one hunk: a head that takes away one line and leaves
    one, a _PATCH_LINK_FROM line and a _PATCH_LINK_COMMIT line, which name the two
    commits by full ids of one length. Paths have the first directory taken off as git
    apply takes it off.

    Args:
        head_name: What follows "diff --git " on the file's first line.
        line_iter: The lines of the patch after that line (_patch_line_iter).

    Returns:
        ((the path, as bytes, the commit it moves from, the commit it moves to) when
        the file is one, else None; the first line not read as part of it, None at
        the end of the patch).
    """
    is_link, is_move_head, new_path = False, True, None
    line = next(line_iter, None)
    while line is not None and line.startswith(_PATCH_HEAD_LINE_TUPLE):
        if not line.startswith(_PATCH_MOVE_HEAD_LINE_TUPLE):
            is_move_head = False
        elif _PATCH_LINK_INDEX.fullmatch(line):
            is_link = True
        elif line.startswith(_PATCH_NEW_PATH_HEAD):
            new_path = _unquote_raw(line[len(_PATCH_NEW_PATH_HEAD):]).partition(b"/")[2]
        line = next(line_iter, None)
    hunk = _PATCH_HUNK_HEAD.match(line) if line is not None else None
    if not (hunk and is_link and is_move_head and new_path is not None
            and hunk.group(1) in (None, b"1") and hunk.group(2) in (None, b"1")
            and _is_head_of(head_name, new_path)):
        return None, line
    line = next(line_iter, None)
    from_commit = _PATCH_LINK_FROM.fullmatch(line) if line is not None else None
    if not from_commit:
        return None, line
    line = next(line_iter, None)
    commit = _PATCH_LINK_COMMIT.fullmatch(line) if line is not None else None
    if not commit or len(commit.group(1)) != len(from_commit.group(1)):
        return None, line
    # A second hunk of the file is not read as a move
    line = next(line_iter, None)
    if line is not None and line.startswith(b"@@ -"):
        return None, line
    return (new_path, from_commit.group(1).decode("ascii"),
            commit.group(1).decode("ascii")), line


def _is_head_of(head_name: bytes, path: bytes) -> bool:
    """Return whether both names of a "diff --git" line give one path.

    Each name may be quoted as git quotes a path in a patch (_unquote_raw), and has its
    first directory taken off as git apply takes it off.

    Examples:
        (b'a/sub b/sub', b'sub')            -> True
        (b'a/sub "b/s\\165b"', b'sub')      -> True
        (b'a/x y b/x y', b'x y')            -> True
        (b'a/sub b/lnk', b'lnk')            -> False

    Args:
        head_name: What follows "diff --git ".
        path: The path, with its first directory taken off.

    Returns:
        True when the two names are that path.
    """
    quote = _PATCH_QUOTE_NAME.match(head_name)
    if quote:
        split = quote.end()
    elif head_name.endswith(b'"'):
        split = head_name.find(b' "')
    else:
        split = head_name.find(b"/") + 1 + len(path)
    return head_name[split:split + 1] == b" " and all(
        _unquote_raw(name).partition(b"/")[2] == path
        for name in (head_name[:split], head_name[split + 1:]))


def index_entry_list(project_dir: str, env_dict: dict[str, str] | None = None
                     ) -> list[tuple[bytes, bytes, bytes]] | None:
    """Return the entries of an index, each with its flags and its mode.

    Args:
        project_dir: The repository.
        env_dict: As run() takes it, such as the index file that is read. None reads
            the repository's own.

    Returns:
        (the tag "git ls-files -v" gives the entry, its mode, its path relative to
        project_dir) for every entry, as bytes, or None when git cannot list the index,
        which is logged.
    """
    proc = run(project_dir, "ls-files", "-v", "--stage", "-z", env_dict=env_dict,
               text=False)
    if proc.returncode != 0:
        logger.info("cannot list the index of %s: %s", project_dir,
                    shown_text(proc.stderr).strip())
        return None
    entry_list = []
    for entry in proc.stdout.split(b"\0"):
        meta, _, path = entry.partition(b"\t")
        field_list = meta.split()
        if path and len(field_list) >= 2:
            entry_list.append((field_list[0], field_list[1], path))
    return entry_list


def is_assume_unchanged(tag: bytes) -> bool:
    """Say whether an index entry carries the assume-unchanged flag.

    Args:
        tag: The entry's tag, as index_entry_list() returns it.

    Returns:
        True for a tag in lower case.
    """
    return tag.islower()


def is_skip_worktree(tag: bytes) -> bool:
    """Say whether an index entry carries the skip-worktree flag.

    Args:
        tag: The entry's tag, as index_entry_list() returns it.

    Returns:
        True for the tag "S" in either case.
    """
    return tag.upper() == b"S"


# The attribute that names the filter driver git runs a file's contents through
_FILTER_ATTR = "filter"

# What check-attr answers for an attribute a path does not carry, carries unset, and
# carries with no value
_NO_ATTR_VALUE_TUPLE = (b"unspecified", b"unset", b"set")


def filter_by_path_dict(project_dir: str, raw_path_list: list[bytes],
                        env_dict: dict[str, str] | None = None,
                        attribute_file: str = "") -> dict[bytes, str] | None:
    """Ask git which filter driver the attributes name for each of the paths.

    Each path is handed over on the standard input as it is written, ended with a
    NUL. Nothing is run and no file of the paths is read.

    Args:
        project_dir: The repository the paths are in.
        raw_path_list: Paths relative to the repository, as bytes.
        env_dict: As run() takes it. Given the settings naming an index file, the
            .gitattributes files are read from that index alone; None reads the ones
            the working tree holds.
        attribute_file: The file git reads as core.attributesFile names it, whatever
            the settings say. Empty leaves the settings' own.

    Returns:
        {path: the driver's name, as shown_text() gives it} for the paths whose
        "filter" attribute names one, or None when git gives no answer.
    """
    if not raw_path_list:
        return {}
    option_list = ["-c", f"core.attributesFile={attribute_file}"] if attribute_file else []
    proc = run(project_dir, *option_list, "check-attr",
               *(["--cached"] if env_dict else []), "-z",
               "--stdin", _FILTER_ATTR, env_dict=env_dict, text=False,
               input_text=b"".join(raw + b"\0" for raw in raw_path_list))
    field_list = proc.stdout.split(b"\0")[:-1]
    if proc.returncode != 0 or len(field_list) != 3 * len(raw_path_list):
        logger.info("git did not say which filter %s gives %d path(s): %s", project_dir,
                    len(raw_path_list), shown_text(proc.stderr).strip())
        return None
    return {field_list[index]: shown_text(field_list[index + 2])
            for index in range(0, len(field_list), 3)
            if field_list[index + 2] not in _NO_ATTR_VALUE_TUPLE}


# The settings a filter driver and a merge driver of the user's own are defined by, as
# a pattern for git config --get-regexp, and the part of a key that names the driver
_DRIVER_KEY_PATTERN = r"^(filter\..*\.(clean|smudge|process|required)|merge\..*\.driver)$"
_FILTER_KEY = re.compile(r"filter\.(.*)\.(clean|smudge|process|required)", re.DOTALL)

# The settings of a filter driver that name a program
_FILTER_COMMAND_KEY_TUPLE = ("clean", "smudge", "process")

# The environment variables git is handed the drivers' replacements in: no command, the
# value that makes a filter driver one git may go without, the command standing in for a
# merge driver, and the driver a merge driver merges common ancestors with
_NO_COMMAND_ENV = "CTLOOM_NO_DRIVER"
_NO_REQUIRE_ENV = "CTLOOM_NO_REQUIRE"
_MERGE_DRIVER_ENV = "CTLOOM_MERGE_DRIVER"
_MERGE_RECURSIVE_ENV = "CTLOOM_MERGE_RECURSIVE"

# A merge driver's name no setting defines, which git reads as its own three-way merge
NO_MERGE_DRIVER = "ctloom/none"

# What ends the key of the setting that gives a merge driver its command, and what
# ends the key of the one naming the driver it merges common ancestors with
_DRIVER_KEY_END = ".driver"
_RECURSIVE_KEY_END = ".recursive"


def _driver_setting_dict(project_dir: str) -> dict[str, str]:
    """Return the settings that define a filter driver or a merge driver.

    Args:
        project_dir: The repository whose settings are read, the user's own and the
            system's with it.

    Returns:
        {key: value}, a key as git writes it and set more than once holding its last
        value.

    Raises:
        RuntimeError: When the settings could not be read.
    """
    proc = run(project_dir, "config", "-z", "--get-regexp", _DRIVER_KEY_PATTERN,
               text=False)
    # git config ends with 1 when no setting matched
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"could not read the drivers of the git settings: "
                           f"{shown_text(proc.stderr or proc.stdout).strip()}")
    setting_dict = {}
    for entry in proc.stdout.split(b"\0"):
        key, _, value = entry.partition(b"\n")
        if key:
            setting_dict[os.fsdecode(key)] = os.fsdecode(value)
    return setting_dict


def filter_name_set(project_dir: str) -> set[str]:
    """Return the names of the filter drivers the git settings give a program.

    Every setting git may read is looked at (_setting_group_list).

    Args:
        project_dir: The repository whose settings are read.

    Returns:
        The names, as shown_text() gives them, of the drivers with a clean, a smudge
        or a process command of which any value is not empty, a value a later one
        sets over as well.

    Raises:
        RuntimeError: When the settings could not be read.
    """
    group_list = _setting_group_list(project_dir)
    if group_list is None:
        raise RuntimeError(f"could not read the filter drivers of the git settings in "
                           f"{project_dir}")
    name_set = set()
    for setting_list in group_list:
        for _, key, value in setting_list:
            match = _FILTER_KEY.fullmatch(key)
            if match and match.group(2) in _FILTER_COMMAND_KEY_TUPLE and value:
                name_set.add(shown_text(match.group(1)))
    return name_set


def attribute_file_list(project_dir: str) -> list[str] | None:
    """Return the files git may read attributes from beside the repository's own.

    The repository's own are the ones in the working tree and in its git directory.

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.

    Returns:
        Each value core.attributesFile has among the settings of _setting_group_list()
        other than one config.GIT_SETTING_TUPLE hands git, "~" replaced by the home
        directory and taken from the top of the working tree when relative, then
        git/attributes under _own_config_dir() of each environment of
        _env_dict_list(), each once, whether or not a file is there; or None when git
        cannot list its settings, which is logged.
    """
    group_list = _setting_group_list(project_dir)
    if group_list is None:
        return None
    path_list = [os.path.join(project_dir, os.path.expanduser(value))
                 for setting_list in group_list for origin_path, key, value in setting_list
                 if key == _ATTRIBUTE_FILE_KEY and value
                 and not _is_own_setting(origin_path, key, value)]
    path_list += [os.path.join(project_dir, _own_config_dir(env_dict), "attributes")
                  for env_dict in _env_dict_list()]
    return list(dict.fromkeys(path_list))


def own_driver_tuple(project_dir: str, merge_command: str = ""
                     ) -> tuple[list[str], dict[str, str]]:
    """Return what makes one git command run none of the drivers the git settings define.

    Every filter driver is left with no command and as one git may go without: a file's
    contents pass as they are. Every merge driver is replaced by merge_command when one
    is given, and merges common ancestors with NO_MERGE_DRIVER.

    Args:
        project_dir: The repository whose settings are read.
        merge_command: The command that stands in for every merge driver, as git
            reads "merge.<name>.driver". Empty leaves the merge drivers as they are.

    Returns:
        (the git options that go before the subcommand, the environment they read
        their values from, as run()'s env_dict takes it), both empty when the
        settings define no driver that is replaced.

    Raises:
        RuntimeError: When the settings could not be read.
    """
    option_list, env_dict = [], {}
    filter_name_list = []
    for key in _driver_setting_dict(project_dir):
        match = _FILTER_KEY.fullmatch(key)
        if match and match.group(1) not in filter_name_list:
            filter_name_list.append(match.group(1))
        elif not match and merge_command:
            option_list.append(f"--config-env={key}={_MERGE_DRIVER_ENV}")
            option_list.append(f"--config-env={key.removesuffix(_DRIVER_KEY_END)}"
                               f"{_RECURSIVE_KEY_END}={_MERGE_RECURSIVE_ENV}")
            env_dict.update({_MERGE_DRIVER_ENV: merge_command,
                             _MERGE_RECURSIVE_ENV: NO_MERGE_DRIVER})
    for name in filter_name_list:
        option_list += [f"--config-env=filter.{name}.{key}={_NO_COMMAND_ENV}"
                        for key in _FILTER_COMMAND_KEY_TUPLE]
        option_list.append(f"--config-env=filter.{name}.required={_NO_REQUIRE_ENV}")
        env_dict.update({_NO_COMMAND_ENV: "", _NO_REQUIRE_ENV: "false"})
    return option_list, env_dict


# What the origin of a setting read from a file starts with in "git config
# --show-origin"
_FILE_ORIGIN = b"file:"

# The settings that name a file git reads more settings from, as a pattern for the
# whole key
_INCLUDE_KEY = re.compile(r"include\.path|includeif\..*\.path", re.DOTALL)

# The settings that name the file git reads attributes from, the file it reads ignore
# rules from and the directory it reads its hooks from, as git writes their keys
_ATTRIBUTE_FILE_KEY = "core.attributesfile"
_HOOK_PATH_KEY = "core.hookspath"
_PLACE_KEY_TUPLE = (_ATTRIBUTE_FILE_KEY, "core.excludesfile", _HOOK_PATH_KEY)

# The setting that names a repository's working tree, as git writes its key
_WORK_TREE_KEY = "core.worktree"

# Where git reads the system's settings from when GIT_CONFIG_SYSTEM names no file
_SYSTEM_CONFIG_PATH = "/etc/gitconfig"

# The settings whose value git runs as a command or names a program by, as a pattern
# for the whole key as git writes it
_COMMAND_KEY = re.compile(
    r"browser\..*\.(cmd|path)"
    r"|core\.(alternaterefscommand|askpass|editor|fsmonitor|gitproxy|pager|sshcommand)"
    r"|credential\.(.*\.)?helper|diff\.external|diff\..*\.(command|textconv)"
    r"|difftool\..*\.(cmd|path)|filter\..*\.(clean|smudge|process)"
    r"|gc\.recentobjectshook|gpg\.(.*\.)?program|gpg\.ssh\.defaultkeycommand"
    r"|guitool\..*\.cmd|imap\.tunnel|instaweb\.(browser|httpd)"
    r"|interactive\.difffilter|man\..*\.(cmd|path)|merge\..*\.driver"
    r"|mergetool\..*\.(cmd|path)|pager\..*|remote\..*\.(receivepack|uploadpack)"
    r"|sendemail\.(.*\.)?(sendmailcmd|smtpserver|tocmd|cccmd|headercmd)"
    r"|hook\..*\.command"
    r"|sequence\.editor|tar\..*\.command"
    r"|trailer\..*\.(command|cmd)|uploadpack\.packobjectshook", re.DOTALL)

# The settings whose value git runs as a command when it starts with _SHELL_MARK, as a
# pattern for the whole key, and what such a value starts with
_SHELL_COMMAND_KEY = re.compile(r"alias\..*|submodule\..*\.update", re.DOTALL)
_SHELL_MARK = "!"

# The settings whose command runs when its name is typed, as a pattern for the whole
# key
_ALIAS_KEY = re.compile(r"alias\..*", re.DOTALL)

# The variables of the environment whose value git runs as a command or names a
# program by
COMMAND_ENV_TUPLE = ("GIT_EXTERNAL_DIFF", "GIT_EDITOR", "GIT_SEQUENCE_EDITOR", "GIT_SSH",
                     "GIT_SSH_COMMAND", "GIT_PAGER", "GIT_ASKPASS", "GIT_PROXY_COMMAND",
                     "GIT_MAN_VIEWER", "EDITOR", "VISUAL", "PAGER", "SSH_ASKPASS")

# The variables of the environment that name the directory git takes its own programs
# from and the one it takes the hooks of a new repository from
DIR_ENV_TUPLE = ("GIT_EXEC_PATH", "GIT_TEMPLATE_DIR")

# What stands between the words of a command a shell reads, quotes and operators
# included
_COMMAND_WORD_GAP = re.compile(r"[\s;|&()<>'\"`$=]+")

# The programs that run what a directory holds when the directory is their first
# argument, as a pattern for the name of the program
_DIR_RUN_NAME = re.compile(r"(python|pypy)[0-9.]*|node|nodejs|deno|bun|sh|bash|dash|zsh"
                           r"|ksh|ruby|perl|php|run-parts")

# The words of a command whose argument is the directory the rest of it is run in or
# taken from, and the one of them that is written with its argument joined to it
_DIR_OPTION_TUPLE = ("-C", "cd", "--git-dir", "--work-tree")
_JOIN_DIR_OPTION = "-C"

# What a word of a command starts with ahead of an argument written joined to it: a
# short option of one letter or digit, and the mark of a file of arguments
_JOIN_HEAD = re.compile(r"(-[A-Za-z0-9]|@)(?=.)", re.DOTALL)

# What a word of a command starts with when its path is written from the top of the
# working tree: the command substitution that prints the top, and the variable that
# holds the directory the command is run in, each with the separator of a path
# behind it
_TOP_HEAD = re.compile(r"(?<![^\s;|&()<>'\"`=])"
                       r"(\$\(\s*git\s+rev-parse\s+--show-toplevel\s*\)|\$PWD|\$\{PWD\})"
                       r"(?=/)")

# The variable of the environment that lists the directories a program started by its
# name is looked for in
_PROGRAM_PATH_ENV = "PATH"


def _dir_word_list(word_list: list[str]) -> list[str]:
    """Return the words of a command that stand where a directory names what is run.

    They are the first word when it has a directory in it, the first word that
    does not start with "-" after a program of _DIR_RUN_NAME, and the one after a
    word of _DIR_OPTION_TUPLE, joined to it by "=" or to _JOIN_DIR_OPTION as well.

    Args:
        word_list: The words of the command, in order.

    Returns:
        Those words, in order. Nothing has to be at any of them.
    """
    flat_list = []
    for word in word_list:
        option, mark, tail = word.partition("=")
        if mark and option in _DIR_OPTION_TUPLE:
            flat_list += [option, tail]
        elif word.startswith(_JOIN_DIR_OPTION) and word != _JOIN_DIR_OPTION:
            flat_list += [_JOIN_DIR_OPTION, word[len(_JOIN_DIR_OPTION):]]
        else:
            flat_list.append(word)
    dir_word_list = [word for word in flat_list[:1] if os.sep in word]
    is_dir_next = False
    for word in flat_list:
        is_option = word.startswith("-")
        if is_dir_next and not is_option:
            dir_word_list.append(word)
        if word in _DIR_OPTION_TUPLE or _DIR_RUN_NAME.fullmatch(os.path.basename(word)):
            is_dir_next = True
        elif not is_option:
            is_dir_next = False
    return dir_word_list


def _value_place_list(project_dir: str, value: str, is_command: bool,
                      any_dir: bool = False) -> list[str]:
    """Return the files and directories a value names in a working tree.

    The value is one of the settings or of the environment. It is read as it is written
    (_text_place_list) and with each _TOP_HEAD in it written as ".", the top of the
    working tree.

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.
        value: The value.
        is_command: Whether git runs the value as a command or names a program by it.
        any_dir: Whether a directory is named by a word wherever the word stands.

    Returns:
        The paths, "~" replaced by the home directory, no symbolic link followed.
    """
    text = value.removeprefix(_SHELL_MARK).strip()
    place_list = []
    for form in dict.fromkeys([text, _TOP_HEAD.sub(os.curdir, text)]):
        place_list += _text_place_list(project_dir, form, is_command, any_dir)
    return list(dict.fromkeys(place_list))


def _text_place_list(project_dir: str, text: str, is_command: bool,
                     any_dir: bool) -> list[str]:
    """Return the files and directories one form of a value names in a working tree.

    The text is read as one path, as the words a shell splits it into, and as the parts
    _COMMAND_WORD_GAP leaves of it, with what follows _JOIN_HEAD in each of them as a
    word of its own, each taken from the top of the working tree when it is relative. A
    word names a file when a file or a symbolic link is at it, and a directory when a
    directory other than the top of the working tree is at it and the word stands where
    _dir_word_list() says or, with any_dir, anywhere but as a first word with no
    directory in it. With is_command a word nothing is at names a place as well, the top
    of the working tree left out, when it stands where _dir_word_list() says or, with
    any_dir, has a directory in it; and the text whole and the first word of it name the
    program besides: one with a directory in it is a place whether or not anything is
    there, the top of the working tree left out, and one without is looked up as a
    command (shutil.which).

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.
        text: The value, with what marks it as a shell command left out.
        is_command: Whether git runs the value as a command or names a program by it.
        any_dir: Whether a directory is named by a word wherever the word stands.

    Returns:
        The paths, "~" replaced by the home directory, no symbolic link followed.
    """
    try:
        word_list = shlex.split(text)
    except ValueError:
        word_list = []
    part_list = [part for part in _COMMAND_WORD_GAP.split(text) if part]
    top = os.path.realpath(project_dir)
    place_list = []
    if is_command:
        for word in dict.fromkeys([text, *word_list[:1], *part_list[:1]]):
            path = os.path.expanduser(word) if os.sep in word else shutil.which(word)
            if path and os.path.realpath(os.path.join(project_dir, path)) != top:
                place_list.append(os.path.join(project_dir, path))
    dir_word_set = {word for sequence in ([text], word_list, part_list)
                    for word in _dir_word_list(sequence)}
    first_word_set = {text, *word_list[:1], *part_list[:1]}
    tail_list = [word[match.end():] for word in (*word_list, *part_list)
                 if (match := _JOIN_HEAD.match(word))]
    for word in dict.fromkeys([text, *word_list, *part_list, *tail_list,
                               *sorted(dir_word_set)]):
        path = os.path.join(project_dir, os.path.expanduser(word))
        if os.path.lexists(path):
            is_place = not os.path.isdir(path) or word in dir_word_set or (
                any_dir and (os.sep in word or word not in first_word_set))
        else:
            is_place = is_command and (word in dir_word_set
                                       or (any_dir and os.sep in word))
        if is_place and os.path.realpath(path) != top:
            place_list.append(path)
    return list(dict.fromkeys(place_list))


def _env_place_list(project_dir: str, env_dict: Mapping[str, str]) -> list[str]:
    """Return the files and directories an environment names git's programs by.

    The directories it lists for a program started by its name are among them.

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.
        env_dict: The environment.

    Returns:
        What the values of COMMAND_ENV_TUPLE name as a command (_value_place_list, a
        directory named by a word wherever it stands), then the directories
        DIR_ENV_TUPLE names, taken from the top of the working tree when relative,
        whether or not anything is there, then the directories _PROGRAM_PATH_ENV lists
        (_program_dir_list).
    """
    place_list = []
    for name in COMMAND_ENV_TUPLE:
        place_list += _value_place_list(project_dir, env_dict.get(name, ""), True, True)
    place_list += [os.path.join(project_dir, env_dict[name])
                   for name in DIR_ENV_TUPLE if env_dict.get(name)]
    return place_list + _program_dir_list(project_dir, env_dict)


def _program_dir_list(project_dir: str, env_dict: Mapping[str, str] | None = None
                      ) -> list[str]:
    """Return the directories a program started by its name is looked for in.

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.
        env_dict: The environment. None reads this process's own.

    Returns:
        The entries of _PROGRAM_PATH_ENV, os.defpath's when the variable is not set,
        whether or not anything is there. A relative entry, an empty one among them,
        is taken from the top of the working tree and from the directory this
        process runs in.
    """
    env_dict = os.environ if env_dict is None else env_dict
    base_list = [project_dir]
    with contextlib.suppress(OSError):
        base_list.append(os.getcwd())
    return [os.path.join(base, entry)
            for entry in env_dict.get(_PROGRAM_PATH_ENV, os.defpath).split(os.pathsep)
            for base in dict.fromkeys(base_list)]


def _home_dir(env_dict: Mapping[str, str]) -> str:
    """Return the home directory of an environment.

    Args:
        env_dict: The environment.

    Returns:
        HOME, or os.path.expanduser("~") when it is unset or empty.
    """
    return env_dict.get("HOME") or os.path.expanduser("~")


def _own_config_dir(env_dict: Mapping[str, str]) -> str:
    """Return the directory git reads the user's own files under XDG_CONFIG_HOME from.

    Args:
        env_dict: The environment.

    Returns:
        git under XDG_CONFIG_HOME, or under .config in the home directory when it is
        unset or empty.
    """
    home = _home_dir(env_dict)
    return os.path.join(env_dict.get("XDG_CONFIG_HOME") or os.path.join(home, ".config"),
                        "git")


def _own_setting_place_list(project_dir: str, env_dict: Mapping[str, str]) -> list[str]:
    """Return where git reads the user's own and the system's settings from.

    The places are those of one environment, whether or not a file is there.

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.
            A relative path of the environment is taken from it.
        env_dict: The environment.

    Returns:
        Absolute paths: the configuration files (GIT_CONFIG_GLOBAL, or .gitconfig in
        the home directory and git/config under XDG_CONFIG_HOME; GIT_CONFIG_SYSTEM, or
        _SYSTEM_CONFIG_PATH), then the attributes file and the ignore file git reads
        when no setting names one (git/attributes and git/ignore under
        XDG_CONFIG_HOME).
    """
    home = _home_dir(env_dict)
    own_dir = _own_config_dir(env_dict)
    global_path = env_dict.get("GIT_CONFIG_GLOBAL")
    config_list = [global_path] if global_path else [os.path.join(home, ".gitconfig"),
                                                     os.path.join(own_dir, "config")]
    config_list.append(env_dict.get("GIT_CONFIG_SYSTEM") or _SYSTEM_CONFIG_PATH)
    return [os.path.normpath(os.path.join(project_dir, path)) for path in
            (*config_list, os.path.join(own_dir, "attributes"),
             os.path.join(own_dir, "ignore"))]


# The file of a working tree's own settings, as "git rev-parse --git-path" takes it
_WORKTREE_CONFIG_PATH = "config.worktree"


def _env_dict_list() -> list[dict[str, str]]:
    """Return the environments the git commands of a repository's user may run in.

    Returns:
        This process's own environment, then the one of the shell it was started from
        (config.shell_env()) when the two differ in a name other than the ones config
        wrote for itself and for the libraries it imports (config.own_name_set()).
    """
    process_dict = dict(os.environ)
    shell_dict = config.shell_env()
    own_set = config.own_name_set()
    if {name for name in process_dict.keys() | shell_dict.keys()
            if process_dict.get(name) != shell_dict.get(name)} <= own_set:
        return [process_dict]
    return [process_dict, shell_dict]


def _read_setting_list(project_dir: str, option_list: list[str],
                       env_dict: Mapping[str, str] | None = None
                       ) -> list[tuple[str, str, str]] | None:
    """Return the settings one "git config --list" reads, with the file of each.

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.
        option_list: The options of git config that say what it reads.
        env_dict: The environment git runs in. None is this process's own.

    Returns:
        (the file the setting is read from, as an absolute path with no symbolic link
        followed, or "" for a setting no file holds; the key as git writes it; the
        value), in git's order, or None when git cannot list them, which is logged.
    """
    proc = run(project_dir, "config", *option_list, "--list", "--show-origin", "-z",
               text=False, base_env_dict=env_dict)
    if proc.returncode != 0:
        logger.info("git config %s did not list its settings in %s: %s",
                    shlex.join(option_list), project_dir,
                    shown_text(proc.stderr).strip())
        return None
    setting_list = []
    field_list = proc.stdout.split(b"\0")[:-1]
    for origin, entry in zip(field_list[::2], field_list[1::2]):
        key, _, value = (os.fsdecode(part) for part in entry.partition(b"\n"))
        origin_path = ""
        if origin.startswith(_FILE_ORIGIN):
            origin_path = os.path.join(project_dir,
                                       os.fsdecode(origin[len(_FILE_ORIGIN):]))
        setting_list.append((origin_path, key, value))
    return setting_list


def _include_path(origin_path: str, key: str, value: str) -> str:
    """Return the file a setting has git read more settings from.

    Args:
        origin_path: The file the setting is read from, or "" for none.
        key: The key as git writes it.
        value: The value.

    Returns:
        For "include.path" and "includeIf.<condition>.path", whatever the condition,
        the value with "~" replaced by the home directory, taken from the directory
        of origin_path when relative; "" for another setting and for a relative value
        no file holds, which git does not read.
    """
    value = os.path.expanduser(value)
    if _INCLUDE_KEY.fullmatch(key) and value and (origin_path or os.path.isabs(value)):
        return os.path.join(os.path.dirname(origin_path), value)
    return ""


def _setting_group_list(project_dir: str) -> list[list[tuple[str, str, str]]] | None:
    """Return every setting git may read for a repository, with the file of each.

    The settings come in groups git reads in one go. The groups are what git reads now
    in each environment of _env_dict_list(), then what each file one of the groups
    includes holds (_include_path, whatever the condition) and what the working tree's
    _WORKTREE_CONFIG_PATH holds, whether or not extensions.worktreeConfig is on: each
    file a group of its own, read once and alone, the files it includes followed the
    same way. A file a group is already read from, nothing is at, or that is not a file
    is not read. One git cannot read is left out, which is logged.

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.

    Returns:
        The groups, each a list of _read_setting_list()'s tuples in git's order, or
        None when git cannot list its settings in one of the environments or say where
        _WORKTREE_CONFIG_PATH is, which is logged.
    """
    group_list = []
    for env_dict in _env_dict_list():
        setting_list = _read_setting_list(project_dir, [], env_dict)
        if setting_list is None:
            return None
        group_list.append(setting_list)
    proc = run(project_dir, "rev-parse", "--git-path", _WORKTREE_CONFIG_PATH, text=False)
    if proc.returncode != 0:
        logger.info("git did not say where %s is in %s: %s", _WORKTREE_CONFIG_PATH,
                    project_dir, shown_text(proc.stderr).strip())
        return None
    read_set = {os.path.realpath(origin_path) for setting_list in group_list
                for origin_path, _, _ in setting_list if origin_path}
    path_list = [os.path.join(project_dir, os.fsdecode(proc.stdout.rstrip(b"\n")))]
    path_list += [path for setting_list in group_list for path in
                  (_include_path(*setting) for setting in setting_list) if path]
    # path_list grows with the files each file read includes
    for path in path_list:
        if os.path.realpath(path) in read_set or not os.path.isfile(path):
            continue
        read_set.add(os.path.realpath(path))
        setting_list = _read_setting_list(project_dir, ["--file", path, "--no-includes"])
        if setting_list:
            group_list.append(setting_list)
            path_list += [include_path for include_path in
                          (_include_path(*setting) for setting in setting_list)
                          if include_path]
    return group_list


def _is_own_setting(origin_path: str, key: str, value: str) -> bool:
    """Say whether a setting is one config.GIT_SETTING_TUPLE hands git.

    Args:
        origin_path: The file the setting is read from, or "" for none.
        key: The key as git writes it.
        value: The value.

    Returns:
        True for a setting no file holds whose key and value GIT_SETTING_TUPLE has.
    """
    own_set = {(own_key.lower(), own_value) for own_key, _, own_value in
               (setting.partition("=") for setting in config.GIT_SETTING_TUPLE[1::2])}
    return not origin_path and (key, value) in own_set


def _is_command_setting(origin_path: str, key: str, value: str) -> bool:
    """Say whether git runs a setting's value as a command or names a program by it.

    Args:
        origin_path: The file the setting is read from, or "" for none.
        key: The key as git writes it.
        value: The value.

    Returns:
        True for a setting of _COMMAND_KEY, and for one of _SHELL_COMMAND_KEY whose
        value starts with _SHELL_MARK, that is not one config.GIT_SETTING_TUPLE hands
        git.
    """
    is_command = _COMMAND_KEY.fullmatch(key) or (
        _SHELL_COMMAND_KEY.fullmatch(key) and value.startswith(_SHELL_MARK))
    return bool(is_command) and not _is_own_setting(origin_path, key, value)


def setting_command_dict(project_dir: str) -> dict[str, bool] | None:
    """Return the values git may run as a command for a repository and its submodules.

    Each value comes with whether only an alias holds it.

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.

    Returns:
        The values of the settings of _setting_group_list() _is_command_setting()
        answers True for, as git holds them, for each repository of _repo_dir_list(),
        then the values each environment of _env_dict_list() has for
        COMMAND_ENV_TUPLE, each once, mapped to True when every setting holding it is
        of _ALIAS_KEY and no variable holds it, or None when git cannot list the
        settings of one of the repositories, which is logged.
    """
    group_list = []
    for repo_dir in _repo_dir_list(project_dir):
        repo_group_list = _setting_group_list(repo_dir)
        if repo_group_list is None:
            return None
        group_list += repo_group_list
    value_dict = {}
    for origin_path, key, value in (setting for setting_list in group_list
                                    for setting in setting_list):
        if _is_command_setting(origin_path, key, value):
            value_dict[value] = value_dict.get(value, True) \
                and bool(_ALIAS_KEY.fullmatch(key))
    for env_dict in _env_dict_list():
        for name in COMMAND_ENV_TUPLE:
            if env_dict.get(name):
                value_dict[env_dict[name]] = False
    return value_dict


def _hook_link_list(hook_dir: str) -> list[str]:
    """Return the entries of a directory git runs hooks from that are symbolic links.

    Args:
        hook_dir: The directory, as an absolute path. Nothing has to be there.

    Returns:
        The absolute path of each such entry, by name, or none when the directory
        cannot be listed.
    """
    try:
        name_list = sorted(os.listdir(hook_dir))
    except OSError:
        return []
    return [os.path.join(hook_dir, name) for name in name_list
            if os.path.islink(os.path.join(hook_dir, name))]


def _is_work_tree_top(directory: str) -> bool:
    """Say whether git takes a directory as the top of a working tree.

    Args:
        directory: The directory, as an absolute path.

    Returns:
        True when "git rev-parse --show-toplevel" run there names the directory
        itself, compared by real path.
    """
    proc = run(directory, "rev-parse", "--show-toplevel", text=False)
    return proc.returncode == 0 and os.path.realpath(
        os.fsdecode(proc.stdout.rstrip(b"\n"))) == os.path.realpath(directory)


def _repo_dir_list(project_dir: str) -> list[str]:
    """Return the top of a repository's working tree and of each submodule git walks into.

    The submodules are those a git command run there walks into, the submodules of a
    submodule as well. A submodule is at a path the index of the repository above it
    holds with GITLINK_MODE, with a .git at that path and git taking the path as the top
    of a working tree (_is_work_tree_top). A path whose real path is already one of them
    is not looked into again. An index git cannot list adds none (index_entry_list,
    which logs it).

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.

    Returns:
        project_dir, then the top of each submodule's working tree as an absolute
        path, a submodule ahead of the ones in it.
    """
    dir_list = [project_dir]
    real_set = {os.path.realpath(project_dir)}
    # dir_list grows with the submodules of each repository it holds
    for repo_dir in dir_list:
        for _, mode, path in index_entry_list(repo_dir) or []:
            sub_dir = os.path.join(repo_dir, os.fsdecode(path))
            if mode != GITLINK_MODE or os.path.realpath(sub_dir) in real_set \
                    or not os.path.lexists(os.path.join(sub_dir, ".git")) \
                    or not _is_work_tree_top(sub_dir):
                continue
            real_set.add(os.path.realpath(sub_dir))
            dir_list.append(sub_dir)
    return dir_list


def setting_place_list(project_dir: str) -> list[str] | None:
    """Return the files and directories git may read a repository's settings from.

    The settings of its submodules count as well. They are _repo_setting_place_list()'s
    for each repository of _repo_dir_list(), a relative value of a submodule's settings
    taken from the top of the submodule's working tree. A place of a submodule that lies
    in the submodule's working tree both as it is written, ".." folded, and with every
    symbolic link resolved is left out.

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.

    Returns:
        Absolute paths, each once, "~" replaced by the home directory, no symbolic
        link followed and no ".." folded, or None when git cannot list the settings of
        one of the repositories, which is logged.
    """
    place_list = []
    for repo_dir in _repo_dir_list(project_dir):
        repo_place_list = _repo_setting_place_list(repo_dir)
        if repo_place_list is None:
            return None
        if repo_dir != project_dir:
            repo_place_list = [place for place in repo_place_list
                               if not _is_in_tree(repo_dir, place)]
        place_list += repo_place_list
    return list(dict.fromkeys(place_list))


def _is_in_tree(top: str, place: str) -> bool:
    """Say whether a path lies at or under the top of a working tree in both forms.

    The forms are the path as it is written, ".." folded, and the path with every
    symbolic link resolved.

    Args:
        top: The top of the working tree, as an absolute path.
        place: The path, as an absolute path. Nothing has to be there.

    Returns:
        True when both forms are top or under it.
    """
    for form, form_top in ((os.path.normpath(place), os.path.normpath(top)),
                           (os.path.realpath(place), os.path.realpath(top))):
        if form != form_top and not form.startswith(form_top + os.sep):
            return False
    return True


def _repo_setting_place_list(project_dir: str) -> list[str] | None:
    """Return the files and directories git may read one repository's settings from.

    They are, for every setting of _setting_group_list() and each environment of
    _env_dict_list(),
    - every configuration file a setting is read from, and every file one of them
      includes (_include_path), whether or not the file is there and the condition
      holds,
    - where git reads the user's own and the system's settings from
      (_own_setting_place_list),
    - the files core.attributesFile and core.excludesFile name and the directory
      core.hooksPath names, by every value of them, a value a later one sets over
      as well,
    - the files and the directories the value of any setting other than
      _WORK_TREE_KEY names (_value_place_list), read as a command for the settings
      of _COMMAND_KEY and the ones of _SHELL_COMMAND_KEY whose value starts with
      _SHELL_MARK, a directory named by a word wherever it stands for those of them
      that are not of _ALIAS_KEY,
    - what the environment names a program of git's by, and the directories it lists
      for a program started by its name (_env_place_list),
    - info/attributes, info/exclude and hooks in the repository's own git directory,
      and
    - every entry of hooks and of a directory core.hooksPath names that is a
      symbolic link (_hook_link_list).
    A setting config.GIT_SETTING_TUPLE hands git is not one of them.

    Args:
        project_dir: The top of the repository's working tree, as an absolute path.
            A relative value is taken from it.

    Returns:
        Absolute paths, "~" replaced by the home directory, no symbolic link
        followed and no ".." folded, or None when git cannot list its settings,
        which is logged.
    """
    group_list = _setting_group_list(project_dir)
    if group_list is None:
        return None
    dir_proc = run(project_dir, "rev-parse", "--git-path", "info/attributes",
                   "--git-path", "info/exclude", "--git-common-dir", text=False)
    dir_line_list = dir_proc.stdout.split(b"\n")[:-1]
    if dir_proc.returncode != 0 or len(dir_line_list) != 3:
        logger.info("git did not list where it reads its settings from in %s: %s",
                    project_dir, shown_text(dir_proc.stderr).strip())
        return None
    place_list = []
    for env_dict in _env_dict_list():
        place_list += _own_setting_place_list(project_dir, env_dict)
        place_list += _env_place_list(project_dir, env_dict)
    for setting_list in group_list:
        for origin_path, key, value in setting_list:
            if origin_path:
                place_list.append(origin_path)
            is_own = _is_own_setting(origin_path, key, value)
            if not is_own and key != _WORK_TREE_KEY:
                is_command = _is_command_setting(origin_path, key, value)
                place_list += _value_place_list(
                    project_dir, value, is_command,
                    is_command and not _ALIAS_KEY.fullmatch(key))
            include_path = _include_path(origin_path, key, value)
            if include_path:
                place_list.append(include_path)
            elif key in _PLACE_KEY_TUPLE and value and not is_own:
                place = os.path.join(project_dir, os.path.expanduser(value))
                place_list.append(place)
                if key == _HOOK_PATH_KEY:
                    place_list += _hook_link_list(place)
    place_list += [os.path.join(project_dir, os.fsdecode(line))
                   for line in dir_line_list[:2]]
    hook_dir = os.path.join(project_dir, os.fsdecode(dir_line_list[2]), "hooks")
    return [*place_list, hook_dir, *_hook_link_list(hook_dir)]


def git_dir(project_dir: str) -> str:
    """Return the directory git keeps the repository itself in.

    It is not always <project_dir>/.git: a worktree and a submodule keep theirs
    elsewhere and name it in a .git file.

    Args:
        project_dir: The repository to ask.

    Returns:
        The path, resolved through realpath, or "" when git cannot say.
    """
    proc = run(project_dir, "rev-parse", "--absolute-git-dir")
    if proc.returncode != 0:
        return ""
    return os.path.realpath(proc.stdout.strip())


def is_work_tree_root(project_dir: str) -> bool:
    """Say whether a directory is the top of a git working tree.

    git's own answer is used: a git worktree and a submodule, which hold a .git file
    rather than a .git directory, count too.

    Args:
        project_dir: The directory to look at. An empty path gives False.

    Returns:
        True when git reports the directory itself as the top of a working tree, False
        for a directory inside one and for one that is not in a repository at all.
    """
    if not project_dir:
        return False
    proc = run(project_dir, "rev-parse", "--show-toplevel")
    if proc.returncode != 0:
        return False
    return os.path.realpath(proc.stdout.strip()) == os.path.realpath(project_dir)


def work_tree_id(project_dir: str, keep_object: bool = True) -> str:
    """Return the id of the git tree the files on disk make.

    The tree holds what git carries or would carry: the tracked files as they are on
    disk and the untracked files no ignore rule covers, without the ones deleted from
    the disk. It is made in an index file of its own (work_tree_index()); the
    repository's index and its working tree are left as they are. A file that
    cannot be read, and one git stops at, keeps what the index holds for it, and the
    log names it. core.safecrlf is not followed: a line ending it refuses is taken as
    core.autocrlf converts it.

    Args:
        project_dir: The repository. An empty path gives "".
        keep_object: False writes the blobs and trees into an object directory of
            this call's own, which reads the repository's objects as an alternate and is
            removed afterwards; the repository's object store is left as it is.

    Returns:
        The full tree hash, or "" for a directory that is not a repository, one the
        tree cannot be written for (a git directory this process cannot write to), and
        one git adds none of the changed files of.
    """
    with work_tree_index(project_dir, keep_object) as (tree, _):
        return tree


@contextlib.contextmanager
def work_tree_index(project_dir: str, keep_object: bool = True
                    ) -> Iterator[tuple[str, dict[str, str]]]:
    """Write the tree the files on disk make, and lend the index file that holds it.

    The index file is one of this call's own, started from a copy of the repository's
    index: files whose stat data is current are not read again. A git command given the
    environment this yields works on that index, and the repository's own index is left
    as it is. The file is removed when the block ends.

    Args:
        project_dir: The repository. An empty path yields ("", {}).
        keep_object: As work_tree_id() takes it.

    Yields:
        (the full tree hash, the settings naming the index file, as run()'s env_dict
        takes them), or ("", {}) for a directory that is not a repository or one the
        tree cannot be written for.
    """
    directory = git_dir(project_dir) if project_dir else ""
    index_path = _own_index_path(directory) if directory else ""
    if not index_path:
        yield "", {}
        return
    env_dict = {"GIT_INDEX_FILE": index_path}
    object_dir = ""
    try:
        if not keep_object:
            object_dir = _scratch_object_dir(project_dir, env_dict)
        tree = _index_tree(project_dir, index_path, env_dict) \
            if keep_object or object_dir else ""
        yield tree, (env_dict if tree else {})
    finally:
        try:
            os.unlink(index_path)
        except FileNotFoundError:
            pass
        if object_dir:
            shutil.rmtree(object_dir, ignore_errors=True)


def _own_index_path(directory: str) -> str:
    """Return a name for a separate index file, which does not exist yet.

    Args:
        directory: The directory git keeps the repository itself in.

    Returns:
        The path, or "" when no file can be made in that directory.
    """
    # Only the name is kept: the file must not exist when git first opens it
    try:
        handle, index_path = tempfile.mkstemp(prefix="ctloom-index-", dir=directory)
    except OSError as e:
        logger.info("cannot make an index of its own in %s: %s", directory, e)
        return ""
    os.close(handle)
    os.unlink(index_path)
    return index_path


def _index_tree(project_dir: str, index_path: str, env_dict: dict[str, str]) -> str:
    """Fill a separate index file with the files on disk and write its tree.

    Args:
        project_dir: The repository.
        index_path: The index file, which does not exist yet.
        env_dict: The environment naming that file.

    Returns:
        The full tree hash, or "" when the index cannot be filled, when git adds none
        of the files on disk to it, or when the tree cannot be written.
    """
    if not _start_index(project_dir, index_path, env_dict):
        return ""
    proc = _add_to_index(project_dir, env_dict)
    if proc.returncode == _FATAL_STATUS:
        if not _add_by_half(project_dir, env_dict, proc.stderr.strip()):
            return ""
    elif proc.returncode != 0:
        unread_list = _unread_add_path_list(project_dir, env_dict)
        warn_once(("unread", project_dir,
                   frozenset(unread_list) or proc.stderr.strip()),
                  "some files of %s could not be read, so what the index holds for "
                  "them is used%s", project_dir,
                  f": {name_list_text([shown_text(path) for path in unread_list])}"
                  if unread_list else "", detail=proc.stderr.strip())
    proc = run(project_dir, "write-tree", env_dict=env_dict)
    if proc.returncode != 0:
        logger.info("cannot write the tree of %s: %s", project_dir,
                    proc.stderr.strip())
        return ""
    return proc.stdout.strip()


def _unread_add_path_list(project_dir: str, env_dict: dict[str, str]) -> list[bytes]:
    """Return the paths an index left as they were because git could not read them.

    Args:
        project_dir: The repository.
        env_dict: The environment naming the separate index the files were added to.

    Returns:
        The paths git lists as changed or new against that index where a file git cannot
        read stands (_is_unread_on_disk), sorted. Empty when git gives no answer.
    """
    proc = run(project_dir, "ls-files", "-z", "--modified", "--others",
               "--exclude-standard", env_dict=env_dict, text=False)
    if proc.returncode != 0:
        return []
    return sorted({path for path in proc.stdout.split(b"\0")
                   if path and _is_unread_on_disk(project_dir, path)})


def _add_to_index(project_dir: str, env_dict: dict[str, str], *args: str,
                  **run_arg: object) -> subprocess.CompletedProcess:
    """Add the files on disk to an index, outside the sparse checkout's definition too.

    git is run with _ADD_TUPLE and _ADD_SPARSE, and without _ADD_SPARSE when it does
    not know that option.

    Args:
        project_dir: The repository.
        env_dict: The environment naming the separate index.
        *args: More arguments of "git add".
        **run_arg: What run() takes by name besides env_dict.

    Returns:
        The finished process, as run() returns it.
    """
    proc = run(project_dir, *_ADD_TUPLE, _ADD_SPARSE, *args, env_dict=env_dict,
               **run_arg)
    if proc.returncode == _USAGE_STATUS:
        proc = run(project_dir, *_ADD_TUPLE, *args, env_dict=env_dict, **run_arg)
    return proc


def _add_by_half(project_dir: str, env_dict: dict[str, str], reason: str) -> bool:
    """Add the files on disk to an index after git stopped at one of them and added none.

    The paths git lists as changed, deleted or new are added half by half
    (_stop_path_list). The paths git stops at keep what the index holds for them, and
    the log names them.

    Args:
        project_dir: The repository.
        env_dict: The environment naming the separate index.
        reason: What git said when it stopped, for the log.

    Returns:
        False when the paths cannot be listed or git stops at every one of them, which
        is logged with reason. True otherwise.
    """
    proc = run(project_dir, "ls-files", "-z", "--modified", "--deleted", "--others",
               "--exclude-standard", env_dict=env_dict, text=False)
    path_list = sorted({path for path in proc.stdout.split(b"\0") if path}) \
        if proc.returncode == 0 else []
    stop_list = _stop_path_list(project_dir, path_list, env_dict)
    if not path_list or len(stop_list) == len(path_list):
        warn_once(("none", project_dir, frozenset(path_list) or reason),
                  "git added none of the files of %s to an index%s", project_dir,
                  f": {name_list_text([shown_text(path) for path in path_list])}"
                  if path_list else "", detail=reason)
        return False
    if stop_list:
        warn_once(("stop", project_dir, frozenset(stop_list)),
                  "git stopped at %d path(s) of %s, so what the index holds for them "
                  "is used: %s", len(stop_list), project_dir,
                  name_list_text([shown_text(path) for path in stop_list]),
                  detail=reason)
    return True


def _stop_path_list(project_dir: str, path_list: list[bytes],
                    env_dict: dict[str, str]) -> list[bytes]:
    """Add paths to an index, and return the ones git stops at.

    The paths are added in one command. When git stops and adds none, each half of
    them is added on its own, down to one path.

    Args:
        project_dir: The repository.
        path_list: The paths, relative to project_dir, as git lists them.
        env_dict: The environment naming the separate index.

    Returns:
        The paths that were not added, in the order given.
    """
    if not path_list:
        return []
    proc = _add_to_index(project_dir, {**env_dict, "GIT_LITERAL_PATHSPECS": "1"},
                         "--pathspec-from-file=-", "--pathspec-file-nul",
                         input_text=b"\0".join(path_list) + b"\0", text=False)
    if proc.returncode != _FATAL_STATUS:
        return []
    if len(path_list) == 1:
        return path_list
    half = len(path_list) // 2
    return (_stop_path_list(project_dir, path_list[:half], env_dict)
            + _stop_path_list(project_dir, path_list[half:], env_dict))


def _scratch_object_dir(project_dir: str, env_dict: dict[str, str]) -> str:
    """Make an object directory git writes into and reads the repository's objects from.

    env_dict is given GIT_OBJECT_DIRECTORY and GIT_ALTERNATE_OBJECT_DIRECTORIES naming
    the two.

    Args:
        project_dir: The repository.
        env_dict: The environment git is run with, added to here.

    Returns:
        The directory made, or "" when the repository's object directory cannot be told
        or the directory cannot be made.
    """
    real_dir = run(project_dir, "rev-parse", "--git-path", "objects").stdout.strip()
    if not real_dir:
        logger.info("cannot tell the object directory of %s", project_dir)
        return ""
    real_dir = os.path.realpath(os.path.join(project_dir, real_dir))
    try:
        object_dir = tempfile.mkdtemp(prefix="ctloom-objects-")
    except OSError as e:
        logger.info("cannot make an object directory of its own: %s", e)
        return ""
    env_dict["GIT_OBJECT_DIRECTORY"] = object_dir
    env_dict["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = _alternate_entry(real_dir)
    return object_dir


def _alternate_entry(path: str) -> str:
    """Return a path as one entry of GIT_ALTERNATE_OBJECT_DIRECTORIES.

    The entries are separated by os.pathsep. A path holding the separator, a double
    quote or a backslash is written between double quotes with C-style escapes.

    Args:
        path: An absolute path.

    Returns:
        The entry.
    """
    if not any(char in path for char in (os.pathsep, '"', "\\")):
        return path
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _start_index(project_dir: str, index_path: str, env_dict: dict[str, str]) -> bool:
    """Fill a separate index file before the files on disk are added to it.

    The repository's own index is copied when there is one, and the two flags that keep
    a file's changes on disk out of what git adds are taken off the files that are on
    disk: assume-unchanged, and skip-worktree. A file absent from the disk keeps
    skip-worktree, as the files outside a sparse checkout do. Without an index of the
    repository's, the tree of HEAD is read.

    Args:
        project_dir: The repository.
        index_path: The index file of this call's own, which does not exist yet.
        env_dict: The environment naming that file.

    Returns:
        True when the index is ready for the files to be added.
    """
    return _copy_index(project_dir, index_path, env_dict) \
        and _clear_local_flag(project_dir, env_dict)


def _copy_index(project_dir: str, index_path: str, env_dict: dict[str, str]) -> bool:
    """Fill a separate index file with a copy of the repository's index.

    Without an index of the repository's, the tree of HEAD is read, and a repository
    with no commit leaves the file as it is.

    Args:
        project_dir: The repository.
        index_path: The index file, which does not exist yet.
        env_dict: The environment naming that file.

    Returns:
        True when the file holds the copy, or the repository has neither an index nor
        a commit. False when it cannot be made, which is logged.
    """
    real_path = run(project_dir, "rev-parse", "--git-path", "index").stdout.strip()
    if real_path and not os.path.isabs(real_path):
        real_path = os.path.join(project_dir, real_path)
    if real_path and os.path.isfile(real_path):
        try:
            shutil.copy2(real_path, index_path)
        except OSError as e:
            logger.info("cannot copy the index of %s: %s", project_dir, e)
            return False
        return True
    if not head_commit(project_dir):
        return True
    proc = run(project_dir, *_INDEX_COPY_TUPLE, "read-tree", "HEAD", env_dict=env_dict)
    if proc.returncode != 0:
        logger.info("cannot read the tree of HEAD in %s: %s", project_dir,
                    proc.stderr.strip())
        return False
    return True


@contextlib.contextmanager
def _index_copy(project_dir: str) -> Iterator[dict[str, str]]:
    """Lend a copy of a repository's index, in a directory of this call's own.

    The copy is filled as _copy_index() fills it, and the objects a command writes go to
    an object directory of this call's own (_scratch_object_dir): a git command given
    the settings this yields, with _INDEX_COPY_TUPLE, leaves the repository's index,
    object store and git directory as they are. Both are removed when the block ends.

    Args:
        project_dir: The repository.

    Yields:
        The settings naming the copy and the object directory, as run()'s env_dict
        takes them, or {} when they cannot be made, which is logged.
    """
    try:
        directory = tempfile.mkdtemp(prefix="ctloom-index-")
    except OSError as e:
        logger.info("cannot make a directory for a copy of the index of %s: %s",
                    project_dir, e)
        yield {}
        return
    env_dict = {"GIT_INDEX_FILE": os.path.join(directory, "index")}
    object_dir = ""
    try:
        object_dir = _scratch_object_dir(project_dir, env_dict)
        is_ready = bool(object_dir) and _copy_index(
            project_dir, env_dict["GIT_INDEX_FILE"], env_dict)
        yield env_dict if is_ready else {}
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        if object_dir:
            shutil.rmtree(object_dir, ignore_errors=True)


def disk_blob_dict(project_dir: str, path_list: list[str]) -> dict[str, str] | None:
    """Return the blob git makes of each of some files on disk.

    The files are handed on standard input to "git update-index --add" over a copy of
    the repository's index (_index_copy) whose entries for files on disk have
    assume-unchanged and skip-worktree taken off (_clear_local_flag): a file whose stat
    data is current is not read again, and one that is read is read as "git add" reads
    it. When git cannot give them, each blob is worked out from the file's bytes as they
    are (_raw_blob_dict).

    Args:
        project_dir: The repository.
        path_list: Paths of regular files, relative to project_dir.

    Returns:
        {path: full blob hash} for every path of path_list, or None when a file cannot
        be read either way, which is logged.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    with _index_copy(project_dir) as env_dict:
        if env_dict and _clear_local_flag(project_dir, env_dict):
            proc = run(project_dir, *_INDEX_COPY_TUPLE, "-c", "core.safecrlf=false",
                       "update-index", "-z", "--add", "--stdin", env_dict=env_dict,
                       input_text=b"".join(os.fsencode(path) + b"\0"
                                           for path in path_list),
                       text=False) if path_list else None
            if proc is None or proc.returncode == 0:
                blob_dict = _stage_blob_dict(project_dir, env_dict)
                if blob_dict is not None \
                        and all(path in blob_dict for path in path_list):
                    return {path: blob_dict[path] for path in path_list}
            else:
                logger.info("git gave no blobs of the files of %s: %s", project_dir,
                            shown_text(proc.stderr).strip())
    return _raw_blob_dict(project_dir, path_list)


def _stage_blob_dict(project_dir: str, env_dict: dict[str, str]
                     ) -> dict[str, str] | None:
    """Return the blob of every entry of an index that is not a conflict.

    Args:
        project_dir: The repository.
        env_dict: As run() takes it: the settings naming the index file.

    Returns:
        {path relative to project_dir, as os.fsdecode() gives it: full blob hash}, or
        None when git cannot list the index.
    """
    proc = run(project_dir, "ls-files", "--stage", "-z", env_dict=env_dict, text=False)
    if proc.returncode != 0:
        return None
    blob_dict = {}
    for entry in proc.stdout.split(b"\0"):
        meta, _, path = entry.partition(b"\t")
        field_list = meta.split()
        if path and len(field_list) == 3 and field_list[2] == b"0":
            blob_dict[os.fsdecode(path)] = field_list[1].decode("ascii")
    return blob_dict


def _raw_blob_dict(project_dir: str, path_list: list[str]) -> dict[str, str] | None:
    """Return the blob id each file's bytes as they are on disk make, without git.

    The id is hashed with the repository's object format ("git rev-parse
    --show-object-format"; sha1 when git does not say), over "blob <size>\\0" and the
    bytes: the id git gives a file that no filter or line ending setting converts.

    Args:
        project_dir: The repository.
        path_list: Paths of regular files, relative to project_dir.

    Returns:
        {path: blob id} for every path of path_list, or None when one cannot be read,
        which is logged.
    """
    algorithm = run(project_dir, "rev-parse", "--show-object-format").stdout.strip()
    if algorithm not in hashlib.algorithms_available:
        algorithm = "sha1"
    blob_dict = {}
    for path in path_list:
        file_path = os.path.join(project_dir, path)
        try:
            tmpfile.check_file(file_path, sys.maxsize)
            with open(file_path, "rb") as fp:
                size = os.fstat(fp.fileno()).st_size
                blob_hash = hashlib.new(algorithm, b"blob %d\0" % size)
                for block in iter(lambda: fp.read(_HASH_BLOCK_BYTE), b""):
                    blob_hash.update(block)
        except OSError as e:
            logger.info("cannot read %s to tell its blob: %s", file_path, e)
            return None
        blob_dict[path] = blob_hash.hexdigest()
    return blob_dict


def tree_blob_dict(project_dir: str, commit: str) -> dict[str, str] | None:
    """Return the regular files a commit's tree holds, each with its blob.

    Args:
        project_dir: The repository.
        commit: The commit.

    Returns:
        {path relative to the repository's root: full blob hash} for every entry of mode
        100644 or 100755 whose path is UTF-8: a symbolic link and a submodule are left
        out. None when the tree cannot be read.

    Raises:
        FileNotFoundError: When project_dir is empty (see run()).
    """
    proc = run(project_dir, "ls-tree", "-r", "-z", "--full-tree", commit, text=False)
    if proc.returncode != 0:
        logger.info("cannot read the tree of %s in %s: %s", short(commit), project_dir,
                    shown_text(proc.stderr).strip())
        return None
    blob_dict = {}
    for entry in proc.stdout.split(b"\0"):
        meta, _, raw_path = entry.partition(b"\t")
        field_list = meta.split()
        if not raw_path or len(field_list) != 3 \
                or field_list[0] not in _FILE_MODE_TUPLE:
            continue
        try:
            blob_dict[raw_path.decode("utf-8")] = field_list[2].decode("ascii")
        except UnicodeDecodeError:
            continue
    return blob_dict


def _clear_local_flag(project_dir: str, env_dict: dict[str, str]) -> bool:
    """Take assume-unchanged and skip-worktree off the entries whose file is on disk.

    Each of the two is taken off by a call of update-index of its own.

    Args:
        project_dir: The repository.
        env_dict: The environment naming the separate index.

    Returns:
        True when the flags could be read and taken off.
    """
    entry_list = index_entry_list(project_dir, env_dict)
    if entry_list is None:
        return False
    assume_list, skip_list = [], []
    for tag, _, path in entry_list:
        if not os.path.lexists(os.path.join(os.fsencode(project_dir), path)):
            continue
        if is_assume_unchanged(tag):
            assume_list.append(path)
        if is_skip_worktree(tag):
            skip_list.append(path)
    for flag, path_list in (("--no-assume-unchanged", assume_list),
                            ("--no-skip-worktree", skip_list)):
        if not path_list:
            continue
        proc = run(project_dir, *_INDEX_COPY_TUPLE, "update-index", "-z", flag,
                   "--stdin",
                   env_dict=env_dict, input_text=b"\0".join(path_list) + b"\0",
                   text=False)
        if proc.returncode != 0:
            logger.info("cannot take %s off in %s: %s", flag, project_dir,
                        shown_text(proc.stderr).strip())
            return False
    return True


def worktree_path_list(project_dir: str) -> list[str]:
    """Return the working trees registered with a repository.

    Args:
        project_dir: The repository to ask.

    Returns:
        The path of the main working tree and of every linked worktree, as
        os.fsdecode() gives them, whether or not the directory is still there. []
        when git cannot list them.
    """
    proc = run(project_dir, "worktree", "list", "--porcelain", "-z", text=False)
    if proc.returncode != 0:
        return []
    head = b"worktree "
    return [os.fsdecode(field[len(head):]) for field in proc.stdout.split(b"\0")
            if field.startswith(head)]


def tree_of(project_dir: str, commit: str) -> str:
    """Return the tree a commit holds.

    Args:
        project_dir: The repository to ask.
        commit: The commit.

    Returns:
        The full tree hash, or "" when the commit cannot be read.
    """
    proc = run(project_dir, "rev-parse", "--verify", "--quiet", f"{commit}^{{tree}}")
    return proc.stdout.strip() if proc.returncode == 0 else ""


def is_snapshot(project_dir: str, commit: str) -> bool:
    """Say whether a commit is one snapshot_commit() wrote of a working tree.

    Args:
        project_dir: The repository to read.
        commit: The commit.

    Returns:
        True when its subject starts with SNAPSHOT_SUBJECT. False for a commit that
        cannot be read.
    """
    proc = run(project_dir, "log", "-1", "--format=%s", commit)
    return proc.returncode == 0 and proc.stdout.startswith(SNAPSHOT_SUBJECT)


def snapshot_commit(project_dir: str, earlier: str = "") -> str:
    """Return a commit whose tree is the files on disk, making one when there is none.

    A working tree with nothing git reports is the tree of HEAD, and HEAD is what comes
    back. Otherwise a commit holding the tree on disk is written with HEAD as its
    parent. No ref is written and nothing is checked out.

    Given an earlier commit of the working tree, the commit is always written, with
    that commit as a second parent: the tree on disk is then what that tree became,
    and a merge of the two takes the tree on disk whole.

    Args:
        project_dir: The repository, as an absolute path.
        earlier: A commit this function wrote of the same working tree before, or "".

    Returns:
        The full commit hash.

    Raises:
        RuntimeError: When project_dir is not a git repository with a commit in it, or
            the tree cannot be written or committed.
    """
    head = require_head(project_dir)
    tree = work_tree_id(project_dir)
    if not tree:
        raise RuntimeError(f"the tree of {project_dir} could not be written, so there is "
                           f"nothing a run can start from")
    if not earlier and tree == tree_of(project_dir, head):
        return head
    proc = run(project_dir, *COMMITTER_TUPLE, "commit-tree", tree, "-p", head,
               *(["-p", earlier] if earlier else []),
               "-m", f"{SNAPSHOT_SUBJECT} at {short(head)}")
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"the tree of {project_dir} could not be committed: "
                           f"{(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip()


def require_head(project_dir: str) -> str:
    """Return the commit a repository is checked out at, refusing a repository with none.

    Args:
        project_dir: The repository, as an absolute path.

    Returns:
        The full commit hash.

    Raises:
        RuntimeError: When project_dir is not a git repository, or holds no commit.
    """
    commit = head_commit(project_dir)
    if not commit:
        raise RuntimeError(f"{project_dir} is not a git repository with a commit in it")
    return commit
