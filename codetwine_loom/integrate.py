"""Combine the branches several runs left behind into one.

The branches start from one commit. Two things can be wrong with the union of those
branches: git cannot decide the text of a line both of them changed, and a name one of
them renamed can be left with references the other wrote or kept. The first is left with
conflict markers, the second merges cleanly and shows up nowhere.

This module builds the union and checks it. Gates run in order and none of them calls a
model. Among them, git's own three-way merge decides the text, the analysis decides
whether a reference that resolved before still resolves, and the test command decides
whether a test that passed before fails now. What is left over after those - conflict
markers, references the merge broke, tests the merge broke - is the work list a model is
asked to settle. Beside the gates, the analysis and each branch's diff say where the
merge took two sides' changes to one definition, or one side's change to a definition
another side's change uses; those are on the work list as well, to be read. When nothing
is on the list, no model is called at all.

The merge itself never touches a working tree. merge-tree writes a tree whether or not
it conflicted: the branches are accumulated with merge-tree and commit-tree and the
result is put in the working tree once, at the end, with HEAD left at the commit the
union is written out against (union_base()).

This module imports nothing from agent.py.
"""

import collections
import contextlib
import functools
import inspect
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterator

from codetwine.utils.file_utils import is_text_file, read_source_text

from codetwine_loom import config, gitcmd, textfile, tmpfile
from codetwine_loom.index.graph import (build_def_node_dict, def_by_file_dict,
                                        key_by_file_name_dict, no_resolve_ref_dict,
                                        outer_def, resolve_usage, target_key_dict)
from codetwine_loom.knowledge import Knowledge, ensure_knowledge
from codetwine_loom.languages import (LANG_NAME_DICT, PATH_EXAMPLE_DICT, edit_paragraph,
                                      is_value_type, name_key, own_name)
from codetwine_loom.llm import make_llm
from codetwine_loom.paths import (knowledge_to_repo, list_text, path_text,
                                  repo_to_knowledge, resolve_in_project, to_repo_relative)
from codetwine_loom.rlm import (COMPLETE_KEY, OUTPUT_LIMIT_NOTE, RLM, END_CALL_DICT, cut,
                                make_deadline_note, output_limit_note, refuse_up_to,
                                run_stage)
from codetwine_loom.textcut import (CommandError, command_line, line_list_of, one_line,
                                    text_error)
from codetwine_loom.tools import edit, source

logger = logging.getLogger(__name__)

# The settings a merge made here is given, whatever the user's git settings say. The
# conflict style the markers are written in, whose middle section holds what the line
# was before either side changed it. Then the values git merges with when they are not
# set: its own three-way merge for a file with no merge attribute, renames detected up
# to the limit git documents as its default, a file added to a directory the other side
# renamed left as a conflict, no line ending conversion before a file is merged, and the
# conflicts of the merge itself alone among the messages
_MERGE_SETTING_TUPLE = ("-c", "merge.conflictStyle=zdiff3",
                        "-c", f"merge.default={gitcmd.NO_MERGE_DRIVER}",
                        "-c", "merge.renames=true",
                        "-c", f"merge.renameLimit={config.GIT_RENAME_LIMIT}",
                        "-c", "merge.directoryRenames=conflict",
                        "-c", "merge.renormalize=false", "-c", "merge.verbosity=2")

# What stands in for every merge driver of the user's own in a merge made here: git's own
# three-way merge of the three files git hands a driver, with the names the merge is
# made under as the labels of a conflict block. It ends with 0 for a clean merge and
# with 1 for any other
_DRIVER_COMMAND = ("git merge-file --marker-size=%L -L {onto} -L {base} -L {commit} "
                   "%A %O %B || exit 1")

# What a conflict block written by _DRIVER_COMMAND names its middle section with
_DRIVER_BASE_LABEL = "base"

# The ref that names what has been merged so far in the markers. It is made while the
# union is assembled and taken away again afterwards
UNION_REF = "refs/codetwine-loom/merged-so-far"

# Where each branch being merged is given a ref of this tool's own, which names it in
# the markers by the name it was passed under, in a clone as well
SIDE_REF_ROOT = "refs/codetwine-loom/side"

# The oldest git whose "merge-tree --write-tree" writes a tree
_MERGE_TREE_VERSION = (2, 38)

# One line of merge-tree's conflicted file info: "<mode> <object> <stage>\t<path>"
_STAGE_RECORD = re.compile(r"^[0-7]{6} [0-9a-f]{40,64} [123]\t")

# What an object name looks like, for telling a written tree from a refusal to merge
OBJECT_NAME = re.compile(r"^[0-9a-f]{40,64}$")

# The stable prefix of the conflict types merge-tree reports
_CONFLICT_TYPE = "CONFLICT"

# The types of the records merge-tree writes about what it merged on its own. In a
# merge that ended with a conflict, a record of any other type is about the conflict
_INFO_TYPE_TUPLE = ("Auto-merging", "Path updated due to directory rename",
                    "Directory rename skipped since directory was renamed on both sides",
                    "Fast forwarding submodule")

# What stands for git's words where a merge ended with a conflict and no record of it
_NO_WORD_CONFLICT = "git ended this merge with a conflict and said nothing more of it"

# The name git gives a file it could not place at its own path in a conflict: the path,
# "~", the ref of the side the file is from with "_" for every "/", and "_<number>"
# where that name is taken
_STAND_IN_NAME = re.compile(
    "~(?:{union}(?:_[0-9]+)?|{side}_[^/]+)$".format(
        union=re.escape(UNION_REF.replace("/", "_")),
        side=re.escape(SIDE_REF_ROOT.replace("/", "_"))))

# How long git writes a conflict marker in a file whose conflict-marker-size attribute
# gives no length
_MARKER_SIZE = 7

# The attribute a file's own marker length is read from
_MARKER_SIZE_ATTR = "conflict-marker-size"

# The number an attribute's value starts with, as git reads it
_LEAD_NUMBER = re.compile(rb"\s*[+-]?[0-9]{1,9}(?![0-9])")

# A line holding what starts or ends a conflict block git wrote in a merge made here,
# wherever in the line it stands and whatever the length of its markers: "<" or ">",
# white space, and one of the names this tool merges under. UNION_REF is followed by
# the end of the line, ":" or white space; a ref under SIDE_REF_ROOT by a character
# that is no carriage return. The line is read without the carriage returns it ends
# with, and searched, not matched from its start
_NAME_LINE = re.compile(
    rb"[<>][ \t]+(?:%s(?:[: \t\r]|$)|%s/[^\r])"
    % (re.escape(UNION_REF.encode()), re.escape(SIDE_REF_ROOT.encode())))

# The lines _NAME_LINE finds, as a pattern for git grep -E
_NAME_LINE_PATTERN = (f"[<>][ \t]+"
                      f"({UNION_REF}([: \t\r]|$)|{SIDE_REF_ROOT}/[^\r])")

# A line that starts a conflict block git wrote in a merge made here, as a pattern for
# git grep -E: a run of "<" from the start of the line, white space and UNION_REF
_BLOCK_START_PATTERN = f"^<+[ \t]+{UNION_REF}([: \t\r]|$)"

# A separator line of a conflict block as git tells one: {size} or more of "=" or "|",
# then white space or the end of the line, as a pattern for git grep -E
_SEPARATOR_PATTERN = "^([=]{{{size},}}|[|]{{{size},}})([ \t\r\f\v]|$)"

# A line that starts or ends a conflict block under whatever name, as a pattern for git
# grep -E: {size} or more of "<" or ">" from the start of the line, then white space
_EDGE_PATTERN = "^([<]{{{size},}}|[>]{{{size},}})[ \t]"

# What the name of a trailer of this tool starts with
TRAILER_PREFIX = "Codetwine-Loom-"

# A trailer line this tool wrote, which is the machine's record rather than anything the
# run was asked for: the one prefix, the rest of the name, ": " and the value
_TRAILER = re.compile(rf"^{TRAILER_PREFIX}[A-Za-z0-9-]*: ")

# The trailer a run's own account of what it did is written in and read back out of
DID_TRAILER = f"{TRAILER_PREFIX}Did"

# The trailers a run's verdict and the exit status its container ended with are
# written in
VERDICT_TRAILER = f"{TRAILER_PREFIX}Verdict"
EXIT_STATUS_TRAILER = f"{TRAILER_PREFIX}Exit-Status"

# What the subject of the commit a merging run leaves starts with, before ": "
MERGE_PREFIX = "integrate"

# The trailer the commit of a union left unsettled holds its separator lines in: how
# many of each separator line each file a conflict block was written in may hold with
# nothing left of a block, as separator_text() writes it. A line named with 0 is one
# the file may hold none of, and a separator line as long as it is read in that file
SEPARATOR_TRAILER = f"{TRAILER_PREFIX}Separator"

# The trailer the commit of a union left unsettled holds the sides in whose change to a
# file is not in the union, as no_merge_text() writes them
NO_MERGE_TRAILER = f"{TRAILER_PREFIX}No-Merge"

# The longest text SEPARATOR_TRAILER and NO_MERGE_TRAILER are each written with and read
# from, in characters
_MAX_SEPARATOR_CHAR = 65536

# The most of one separator line a record may allow one file
_MAX_SEPARATOR_COUNT = 1_000_000

# The longest path a record names, in characters
_MAX_SEPARATOR_PATH_CHAR = 4096

# What is read off one commit besides the request, in the order the fields come back, as
# {the name in the record: the trailer it is read from}. They are the machine's own
# record of how the run that made the commit ended. "tests_changed" is which of the
# changed files are tests. "run_request" is what the run as a whole was given; the
# message of a commit written for one task of a split request holds that task.
# "stand_on" is, on a union's merge, the last commit of an earlier union the commit the
# union was written against holds (union_part_list)
_READ_TRAILER_DICT = {"did": "Did", "verdict": "Verdict", "why": "Why",
                      "tests_changed": "Tests-Changed", "ended": "Ended",
                      "decisions": "Decisions", "run_request": "Run-Request",
                      "stage": "Stage", "non_test_changed": "Non-Test-Changed",
                      "tests_base": "Tests-Base", "test_open": "Test-Open", "gap": "Gap",
                      "gap_done": "Gap-Done", "stand_on": "Stand-On"}

# The stage names a commit records: the stage that writes tests, the one that refactors
# the change and the one that refactors the tests
TEST_CREATE_STAGE = "test-create"
REFACTOR_STAGE = "refactor"
TEST_REFACTOR_STAGE = "test-refactor"

# Those three stages, in the order they run
STAGE_TUPLE = (TEST_CREATE_STAGE, REFACTOR_STAGE, TEST_REFACTOR_STAGE)


def stage_key(stage: str) -> str:
    """Return the name one of those stages goes by where a hyphen cannot be used.

    Examples:
        "test-create"   -> "test_create"
        "test-refactor" -> "test_refactor"

    Args:
        stage: One of STAGE_TUPLE.

    Returns:
        The name.
    """
    return stage.replace("-", "_")


# What a merging run records as its verdict in the commit's trailer: whether the machine
# found nothing left, or found something left
SETTLE_VERDICT = "settled"
UNSETTLE_VERDICT = "unsettled"

# What a merging run records as how it ended: COMPLETE() accepted with the check finding
# nothing, COMPLETE() accepted with the check still finding something once the refusals
# were used up, the turns used up without COMPLETE(), or the loop stopped on an error. A
# run that called no model records none
END_COMPLETE = "complete"
END_REFUSAL_OUT = "refusals exhausted"
END_TURN_OUT = "turns exhausted"
END_ERROR = "stopped early"

# How "git log" is asked for the message of each commit it lists: whole, as UTF-8, a
# NUL after each
_MESSAGE_ARG_TUPLE = ("-z", "--encoding=UTF-8", "--format=%B")


def commit_of(project_dir: str, name: str) -> str:
    """Resolve a branch name to a commit.

    The remote-tracking name under "origin/", which is how a clone reaches the branches
    of what it was cloned from, is tried when there is no local branch of that name.

    Args:
        project_dir: The repository to resolve the name in.
        name: The branch name, as it was passed on the command line.

    Returns:
        The full commit hash.

    Raises:
        RuntimeError: When the repository holds no branch of that name.
    """
    for ref in (f"refs/heads/{name}", f"refs/remotes/origin/{name}", name):
        proc = gitcmd.run(project_dir, "rev-parse", "--verify", "--quiet",
                          f"{ref}^{{commit}}")
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    raise CommandError(f"{project_dir} has no branch named {name}. Pass a branch a run "
                       f"committed to; they are listed by:",
                       command_line("git", "-C", project_dir, "branch", "--list"))


def first_parent(project_dir: str, name: str) -> str:
    """Return the first parent of a branch's tip, which a merge was written against.

    Args:
        project_dir: The repository to read.
        name: The branch name, as it was passed on the command line.

    Returns:
        The full commit hash.

    Raises:
        RuntimeError: When the repository holds no branch of that name, or its tip has
            no parent.
    """
    commit = commit_of(project_dir, name)
    proc = gitcmd.run(project_dir, "rev-parse", "--verify", "--quiet", f"{commit}^1")
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"{name} holds {gitcmd.short(commit)}, which has no parent, so "
                           f"there is no commit a combined change on it was written against")
    return proc.stdout.strip()


def merge_base(project_dir: str, branch_list: list[str]) -> str:
    """Return the commit every one of these branches starts from.

    The names are turned into commits by commit_of(), which reads a name a clone holds
    under "origin/" as well.

    Args:
        project_dir: The repository to read.
        branch_list: Two or more branch names.

    Returns:
        The full commit hash.

    Raises:
        RuntimeError: When one of them is not in the repository, or when they share no
            starting commit.
    """
    commit_list = [commit_of(project_dir, name) for name in branch_list]
    proc = gitcmd.run(project_dir, "merge-base", "--octopus", *commit_list)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"these branches share no starting commit, so there is nothing to merge "
            f"them onto: {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip()


def _parse_record_list(record_list: list[str]) -> list[tuple[list[str], str, str]]:
    """Read merge-tree's informational records.

    One record is "<count>", then count paths, then the conflict type, then the message.
    An empty record separates this section from the one before it and is skipped. A
    record that cannot be read that way ends the reading: the ones already read are kept
    and how many were left is logged.

    Args:
        record_list: The NUL-separated records that follow the conflicted file info.

    Returns:
        [(the paths the record names, its type, its message in git's own words,
        stripped of the newline git ends it with), ...].
    """
    entry_list, index = [], 0
    while index < len(record_list):
        if record_list[index] == "":
            index += 1
            continue
        try:
            count = int(record_list[index])
        except ValueError:
            count = -1
        end = index + count + 3
        if count < 0 or end > len(record_list):
            logger.info("cannot read %d record(s) of the merge output as conflict "
                        "messages; leaving them out", len(record_list) - index)
            break
        entry_list.append((record_list[index + 1:end - 2], record_list[end - 2],
                           record_list[end - 1].strip()))
        index = end
    return entry_list


def _parse_merge_tree(stdout: str) -> tuple[str, list[str],
                                            list[tuple[list[str], str, str]]]:
    """Read what merge-tree --write-tree -z printed.

    The output is the merged tree, then one record per conflicted file, then the
    informational records. The two kinds of record are told apart by their shape: a
    conflicted file's record is a mode, an object and a stage before a tab.

    Args:
        stdout: merge-tree's standard output, decoded with os.fsdecode().

    Returns:
        (the merged tree, the conflicted paths without repeats, in the order given,
        _parse_record_list()'s entries), every path as os.fsdecode() gives it.

    Raises:
        RuntimeError: When the first record is not an object name.
    """
    record_list = stdout.split("\0")
    while record_list and record_list[-1] == "":
        record_list.pop()
    tree = record_list[0].strip() if record_list else ""
    if not OBJECT_NAME.match(tree):
        raise RuntimeError("merge-tree wrote no tree")
    rest = record_list[1:]
    path_dict, index = {}, 0
    while index < len(rest) and _STAGE_RECORD.match(rest[index]):
        path_dict[rest[index].split("\t", 1)[1]] = None
        index += 1
    return tree, list(path_dict), _parse_record_list(rest[index:])


def _merge_tree(project_dir: str, onto: str, commit: str
                ) -> tuple[str, list[str], list[str], list[str]]:
    """Merge one commit into another and return the tree, without a working tree.

    A tree is written whether or not the merge conflicted; a conflicted one holds the
    files with the markers in them. Whether it conflicted is what git's exit status
    and the types of its records say: a record whose type starts with _CONFLICT_TYPE,
    and in a merge git ended with a conflict every record whose type is not one of
    _INFO_TYPE_TUPLE. A conflict record naming no conflicted file is held to the
    files of the merged tree it names, or that lie under a directory it names
    (_tree_file_list). No merge driver and no filter driver of the user's git
    settings is run: git's own merge stands in for each merge driver, and a file's
    contents pass as they are (_own_driver_tuple). The settings of
    _MERGE_SETTING_TUPLE are the same for every merge.

    Args:
        project_dir: The repository to merge in.
        onto: The commit merged into.
        commit: The commit merged.

    Returns:
        (the merged tree, the conflicted paths sorted and without repeats, the
        conflict messages, those of them that name no file of the merged tree), the
        paths and the messages as gitcmd.shown_text() gives them. A merge git ended
        with a conflict holds at least one message.

    Raises:
        RuntimeError: When the merge could not be run at all. A refusal comes back with
            the same status as a conflict and no tree on standard output: the tree
            decides, not the status alone.
    """
    option_list, env_dict = _own_driver_tuple(project_dir, onto, commit)
    proc = gitcmd.run(project_dir, *_MERGE_SETTING_TUPLE, *option_list, "merge-tree",
                      "--write-tree", "-z", onto, commit, env_dict=env_dict, text=False)
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        raise RuntimeError(f"could not merge {commit} onto {onto}: "
                           f"{gitcmd.shown_text(proc.stderr or proc.stdout).strip()}")
    is_conflict = proc.returncode == 1
    tree, stage_path_list, record_list = _parse_merge_tree(os.fsdecode(proc.stdout))
    path_set = {gitcmd.shown_text(path) for path in stage_path_list}
    message_list, no_path_list = [], []
    for name_list, kind, message in record_list:
        if not kind.startswith(_CONFLICT_TYPE) \
                and (not is_conflict or kind in _INFO_TYPE_TUPLE):
            continue
        message_list.append(gitcmd.shown_text(message))
        if not set(name_list).isdisjoint(stage_path_list):
            continue
        file_list = _tree_file_list(project_dir, tree, name_list)
        path_set.update(file_list)
        if not file_list:
            no_path_list.append(message_list[-1])
    if is_conflict and not message_list:
        message_list.append(_NO_WORD_CONFLICT)
        if not path_set:
            no_path_list.append(_NO_WORD_CONFLICT)
    return tree, sorted(path_set), message_list, no_path_list


def _own_driver_tuple(project_dir: str, onto: str, commit: str
                      ) -> tuple[list[str], dict[str, str]]:
    """Return what makes git run its own merge in place of the user's merge drivers.

    No merge driver and no filter driver the user's git settings define is run.

    Args:
        project_dir: The repository whose settings are read.
        onto: The name the commit merged into is given in a conflict block.
        commit: The name the commit merged is given there.

    Returns:
        gitcmd.own_driver_tuple()'s return value with _DRIVER_COMMAND standing in
        for the merge drivers: (the git options, the environment they read their
        values from), both empty when the settings define no driver.

    Raises:
        RuntimeError: When the settings could not be read.
    """
    # A "%" is doubled for git, which reads the command for its own placeholders
    onto_label, base_label, commit_label = (
        shlex.quote(label).replace("%", "%%")
        for label in (onto, _DRIVER_BASE_LABEL, commit))
    command = _DRIVER_COMMAND.format(onto=onto_label, base=base_label, commit=commit_label)
    return gitcmd.own_driver_tuple(project_dir, command)


def _tree_file_list(project_dir: str, tree: str,
                    name_list: list[str] | None = None) -> list[str]:
    """Return the files a tree holds, or those at some paths and under them.

    A path that is a directory gives the files under it.

    Args:
        project_dir: The repository.
        tree: The commit or the tree to read.
        name_list: Repository-relative paths as os.fsdecode() gives them, each read
            as itself and not as a pattern. None reads every file of the tree.

    Returns:
        The paths of the files, as gitcmd.shown_text() gives them. [] for an empty
        name_list, when the tree holds none of them, and when the tree cannot be read,
        which is logged.
    """
    if name_list is not None and not name_list:
        return []
    proc = gitcmd.run(project_dir, "--literal-pathspecs", "ls-tree", "-r", "-z",
                      "--full-tree", "--name-only", tree, "--", *(name_list or []),
                      text=False)
    if proc.returncode != 0:
        logger.info("cannot read the files of %s: %s", gitcmd.short(tree),
                    gitcmd.shown_text(proc.stderr).strip())
        return []
    return [gitcmd.shown_text(path) for path in proc.stdout.split(b"\0") if path]


def _commit_tree(project_dir: str, tree: str, onto: str, commit: str,
                 name: str) -> str:
    """Make the merge commit of a tree merge-tree wrote.

    Args:
        project_dir: The repository to write the commit in.
        tree: The merged tree.
        onto: The first parent, which is what has been accumulated so far.
        commit: The second parent, which is the branch being merged.
        name: The branch's name, for the message.

    Returns:
        The full commit hash.

    Raises:
        RuntimeError: When the commit could not be written.
    """
    proc = gitcmd.run(project_dir, *gitcmd.COMMITTER_TUPLE, "commit-tree", tree,
                      "-p", onto, "-p", commit, "-m", f"merge {name}")
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"could not commit the merge of {name}: "
                           f"{(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip()


def message_list(project_dir: str, *rev_arg: str) -> list[str]:
    """Return the message of every commit "git log" lists, whole.

    The message comes back as git gives it in UTF-8, a byte that is not UTF-8 read
    as U+FFFD, and its lines end at "\\n" alone.

    Examples:
        message_list(repo, "-1", commit)                      -> [that commit's message]
        message_list(repo, "--reverse", f"{base}..{commit}")  -> oldest first

    Args:
        project_dir: The repository to read.
        *rev_arg: What "git log" is told to list: its options and revisions.

    Returns:
        The messages, in the order git lists the commits.

    Raises:
        RuntimeError: When git could not list them. The message holds git's own
            output.
    """
    proc = gitcmd.run(project_dir, "log", *_MESSAGE_ARG_TUPLE, *rev_arg, text=False)
    if proc.returncode != 0:
        raise RuntimeError(gitcmd.shown_text(proc.stderr or proc.stdout).strip())
    return [message_byte.decode("utf-8", "replace")
            for message_byte in proc.stdout.split(b"\0")[:-1]]


def _message_part_tuple(message: str) -> tuple[str, str]:
    """Split a commit message into what is above its record and the record.

    The record is the message's last paragraph: the lines after its last empty line.
    It is one only where a line of it reads as _TRAILER, and a message without an
    empty line holds none.

    Examples:
        "x: do it\\n\\ndo it\\n\\nCodetwine-Loom-Verdict: failed\\n"
            -> ("x: do it\\n\\ndo it", "Codetwine-Loom-Verdict: failed")
        "x: do it\\n\\ndo it\\n"  -> ("x: do it\\n\\ndo it", "")
        "Codetwine-Loom-Verdict: failed\\n" -> ("Codetwine-Loom-Verdict: failed", "")

    Args:
        message: The whole commit message.

    Returns:
        (the message above the record, the record's paragraph or ""), neither with a
        line break at its end.
    """
    text = message.rstrip("\n")
    above, separator, paragraph = text.rpartition("\n\n")
    if not separator or not any(_TRAILER.match(line) for line in paragraph.split("\n")):
        return text, ""
    return above.rstrip("\n"), paragraph


def record_line_list(message: str) -> list[str]:
    """Return the record lines a commit message ends with.

    They are the lines of the message's last paragraph that read as _TRAILER. Any other
    line of that paragraph is left out, and no line above it is read, whatever it holds.

    Args:
        message: The whole commit message, as message_list() returns it.

    Returns:
        The lines, each "Name: value", in the order they are written.
    """
    paragraph = _message_part_tuple(message)[1]
    return [line for line in paragraph.split("\n") if _TRAILER.match(line)]


def trailer_value(line_list: list[str], name: str) -> str:
    """Return the value record lines carry under one name.

    Examples:
        (["Codetwine-Loom-Verdict: failed", "Codetwine-Loom-Why: no edit"],
         "Codetwine-Loom-Why") -> "no edit"
        (["Codetwine-Loom-Verdict: failed"], "Codetwine-Loom-Why") -> ""

    Args:
        line_list: record_line_list()'s return value.
        name: The trailer's whole name, TRAILER_PREFIX included.

    Returns:
        The value of the first line under that name, or "" when there is none.
    """
    head = f"{name}: "
    return next((line[len(head):].strip() for line in line_list
                 if line.startswith(head)), "")


def _request_in_message(message: str) -> str:
    """Return the request out of one commit message a run wrote.

    The subject line and the record the message ends with are dropped, and what is
    left is the request as it was given, every line of it.

    Args:
        message: The whole commit message.

    Returns:
        The request, or "" for a message that carries none.
    """
    return _message_part_tuple(message)[0].partition("\n")[2].strip()


def commit_record_list(project_dir: str, base: str, commit: str,
                       own_only: bool = False) -> list[dict]:
    """Return what each run that committed to a branch was asked for and made of it.

    It is what the branch says about itself, read from the commits the runs left, in
    one pass: the request and the machine's own record out of each message.

    Args:
        project_dir: The repository to read.
        base: The commit the range starts after.
        commit: The commit the range ends at.
        own_only: True keeps the commits this tool wrote alone (a branch somebody else
            also commits to).

    Returns:
        [{"asked", "did", "verdict", "why", "tests_changed", "ended", "decisions",
        "run_request", "stage", "non_test_changed", "tests_base", "test_open", "gap",
        "gap_done"}, ...], oldest first, and a commit always after the commits it has as
        parents, uncut. "tests_changed" is the paths as the trailer holds them, one line
        separated by spaces; "decisions" is the JSON text the trailer holds;
        "run_request" is on one line, as every trailer is. A commit saying none of them
        is left out. A range that cannot be read comes back empty, which is logged.
    """
    try:
        text_list = message_list(project_dir, "--reverse", "--topo-order",
                                 f"{base}..{commit}")
    except RuntimeError as e:
        logger.info("cannot read what %s left: %s", gitcmd.short(commit), e)
        return []
    record_list = []
    for message in text_list:
        entry_dict = _record_entry_dict(message)
        if own_only and not any(entry_dict[key] for key in _READ_TRAILER_DICT):
            continue
        if any(entry_dict.values()):
            record_list.append(entry_dict)
    return record_list


def _record_entry_dict(message: str) -> dict:
    """Read one commit's record out of its message.

    Args:
        message: The whole commit message, as message_list() returns it.

    Returns:
        {"asked", then every key of _READ_TRAILER_DICT}, a key the message says
        nothing for holding "".
    """
    line_list = record_line_list(message)
    entry_dict = {"asked": _request_in_message(message)}
    entry_dict.update({key: trailer_value(line_list, f"{TRAILER_PREFIX}{name}")
                       for key, name in _READ_TRAILER_DICT.items()})
    return entry_dict


def own_record_at(project_dir: str, commit: str) -> dict:
    """Return what the run that wrote one commit recorded on it.

    The commit alone is read, and no commit it reaches: a merge commit says nothing of
    the commits it merged. "stage" is kept only where the commit's subject starts with
    that stage's name and ": ".

    Args:
        project_dir: The repository to read.
        commit: The commit.

    Returns:
        commit_record_list()'s entry for the commit, or {} for a commit that says
        none of it and for one that cannot be read, which is logged.
    """
    try:
        text_list = message_list(project_dir, "-1", commit)
    except RuntimeError as e:
        logger.info("cannot read what %s records: %s", gitcmd.short(commit), e)
        return {}
    message = text_list[0] if text_list else ""
    entry_dict = _record_entry_dict(message)
    if not any(entry_dict.values()):
        return {}
    subject = message.partition("\n")[0]
    if entry_dict["stage"] and not subject.startswith(f"{entry_dict['stage']}: "):
        entry_dict["stage"] = ""
    return entry_dict


def branch_record_dict(project_dir: str, name: str) -> dict:
    """Return what the run that last committed to a branch left there.

    The range starts after the first parent of below_stage()'s commit under the tip and
    ends at the tip: for a merge it holds the commits of the branches it combined as
    well. The earlier unions the merge stands on come before them (union_part_list).

    Args:
        project_dir: The repository to read.
        name: The branch's name, as it was passed.

    Returns:
        {"branch", "commit" (the tip), "held" (whether the working tree holds what that
        run made: _is_part_held), "told" (the records of the earlier unions' parts,
        then those of that range, oldest first), "earlier_list" ([{"held", "told"},
        ...]: one per earlier union, whether the working tree holds what it made as
        union_part_list() reads it and its records, in the order "told" holds them),
        "gap" (union_gap_list(): the cases of the range and of the earlier unions the
        working tree holds)}.

    Raises:
        RuntimeError: When the repository holds no branch of that name.
    """
    tip = commit_of(project_dir, name)
    merge = below_stage(project_dir, tip)
    part_list = union_part_list(project_dir, f"{merge}^", tip)
    return {"branch": name, "commit": tip,
            "held": part_list[-1]["held"],
            "told": [record_dict for part_dict in part_list
                     for record_dict in part_dict["told"]],
            "earlier_list": [{"held": part_dict["held"], "told": part_dict["told"]}
                             for part_dict in part_list[:-1]],
            "gap": union_gap_list(part_list)}


def _is_part_held(project_dir: str, tip: str, merge: str) -> bool:
    """Say whether the working tree holds what the run that ended at a commit made.

    Args:
        project_dir: The repository, whose working tree is read.
        tip: The last commit of the run.
        merge: below_stage()'s commit under it.

    Returns:
        True when what is checked out reaches tip, or the files hold what that run
        made (_has_run_change). False for a repository whose HEAD cannot be read.
    """
    head = gitcmd.head_commit(project_dir)
    return bool(head) and (gitcmd.is_ancestor(project_dir, tip, head)
                           or _has_run_change(project_dir, tip, merge))


def _has_run_change(project_dir: str, tip: str, merge: str) -> bool:
    """Say whether the working tree holds what the last run on a branch made.

    The run is the one that last committed to the branch. The change is read from the
    commit the run was written against, up to the tip and then up to each commit under
    it down to merge, which is what the working tree holds of a run whose last stages
    were not put into it.

    Args:
        project_dir: The repository, whose working tree is read.
        tip: The branch's tip.
        merge: below_stage()'s commit under the tip.

    Returns:
        True when gitcmd.has_change_in_tree() says so for one of those commits.
    """
    commit = tip
    while not gitcmd.has_change_in_tree(project_dir, f"{merge}^", commit):
        if commit == merge:
            return False
        commit = first_parent(project_dir, commit)
    return True


def below_stage(project_dir: str, commit: str) -> str:
    """Return the first commit at or under one that no stage after a merge wrote.

    The commits of the stages are passed over along their first parents.

    Args:
        project_dir: The repository to read.
        commit: The commit to start at.

    Returns:
        The commit: the merge the stages were run over, or commit itself when no stage
        wrote it.
    """
    while own_record_at(project_dir, commit).get("stage") in STAGE_TUPLE:
        commit = first_parent(project_dir, commit)
    return commit


def _run_request_at(project_dir: str, commit: str) -> str:
    """Return the request one commit says its run was given.

    Args:
        project_dir: The repository to read.
        commit: The commit to read, and no commit it reaches.

    Returns:
        The request on one line, or "" when the commit carries none or cannot be read.
    """
    try:
        line_list = own_trailer_list(project_dir, commit)
    except RuntimeError:
        return ""
    return trailer_value(line_list, f"{TRAILER_PREFIX}{_READ_TRAILER_DICT['run_request']}")


def request_in_branch_list(project_dir: str, branch_list: list[str]) -> str:
    """Return the request the run that left these branches was given.

    The request is read from the run-request trailer, not from the message, which for
    a commit written for one task of a split request holds that task. A branch says
    the request its tip carries; the commits a merge at the tip combined are read only
    when the tip carries none. Branches that carry different requests give none.

    Args:
        project_dir: The repository to read.
        branch_list: The branch names, as they were passed.

    Returns:
        The request on one line, or "" when none of those branches carries one and when
        they do not carry the same one.

    Raises:
        RuntimeError: When the repository holds no branch of one of those names.
    """
    text_list = []
    for name in branch_list:
        tip_text = _run_request_at(project_dir, commit_of(project_dir, name))
        request_list = [tip_text] if tip_text else [
            record_dict["run_request"]
            for record_dict in branch_record_dict(project_dir, name)["told"]]
        for text in request_list:
            if text and text not in text_list:
                text_list.append(text)
    if len(text_list) > 1:
        logger.info("%s were made for %d different requests, so none of them is read as "
                    "what they were all made for", list_text(branch_list), len(text_list))
    return text_list[0] if len(text_list) == 1 else ""


def decision_text(decision_list: list[dict], max_char: int = 0) -> str:
    """Render what a run recorded as the one line its trailer holds.

    Where the line would be over max_char, entries are left out from the end until it
    fits; what the trailer holds is still read back by decision_list_of().

    Examples:
        [{"what": "f.txt", "why": "kept a"}] -> '[{"what":"f.txt","why":"kept a"}]'
        []                                    -> ""

    Args:
        decision_list: [{"what", "why"}, ...], as decision_list() returns them. Any
            other key an entry holds (a case's test) is left out.
        max_char: The longest line to write. 0 or less writes every entry.

    Returns:
        The entries as JSON on one line, or "" when there are none or none fits.
    """
    entry_list = [{"what": entry_dict["what"], "why": entry_dict["why"]}
                  for entry_dict in decision_list]
    keep_count = len(entry_list)
    while keep_count:
        text = json.dumps(entry_list[:keep_count], ensure_ascii=False,
                          separators=(",", ":"))
        if max_char <= 0 or len(text) <= max_char:
            break
        keep_count -= 1
    if keep_count < len(entry_list):
        logger.info("the trailer holds %d of %d recorded entries, to stay within %d "
                    "characters; every entry is in what the run wrote", keep_count,
                    len(entry_list), max_char)
    return text if keep_count else ""


def decision_list_of(text: str) -> list[dict]:
    """Read a trailer's text back into the decisions it holds; decision_text()'s inverse.

    Args:
        text: The trailer's value, "" when the commit carries none.

    Returns:
        [{"what", "why"}, ...], with an entry that is not a dict of two strings left
        out. Text that is not JSON comes back empty, which is logged.
    """
    if not text:
        return []
    try:
        value = json.loads(text)
    except (ValueError, RecursionError) as e:
        logger.info("the decisions trailer cannot be read (%s): %s", e, cut(text, 80))
        return []
    if not isinstance(value, list):
        return []
    return [{"what": str(entry_dict["what"]), "why": str(entry_dict["why"])}
            for entry_dict in value
            if isinstance(entry_dict, dict) and "what" in entry_dict and "why" in entry_dict]


def gap_list_in(project_dir: str, base: str, commit: str) -> list[dict]:
    """Return every case the commits of a union say the change does not do.

    The union is read as the working tree holds it once it goes in, so every earlier
    union it stands on counts.

    Args:
        project_dir: The repository to read.
        base: The commit the union was written against, which the range starts after.
        commit: The commit the range ends at.

    Returns:
        union_gap_list() over union_part_list()'s parts, the union's own held.
    """
    return union_gap_list(union_part_list(project_dir, base, commit, held=True))


def union_part_list(project_dir: str, base: str, commit: str,
                    held: bool | None = None) -> list[dict]:
    """Return what the runs that made a union recorded, with the unions it stands on.

    The union's own part is commit_record_list()'s from base to commit. The merge under
    commit (below_stage) names the earlier union it stands on (stand_on_of): the commit
    of that union's branch that the commit the union was written against holds, which
    the merging run that wrote it recorded. That union's part is read from the commit its
    merge was written against up to that commit, and the union it stands on in turn.

    Whether the working tree holds a part is read from the union's own down: an earlier
    union's part is held when the part above it, whose merge names it, is held, and is
    read by itself (_is_part_held) only when that part is not.

    Args:
        project_dir: The repository to read, whose working tree says which parts it
            holds.
        base: The commit the union was written against, which the range starts after.
        commit: The commit the range ends at.
        held: Whether the working tree holds the union's own part. None reads it
            (_is_part_held).

    Returns:
        [{"tip": the commit the part ends at, "merge": below_stage() of it, "told":
        commit_record_list()'s entries, "held": whether the working tree holds what
        the part made}, ...], the earliest union's part first and the union's own last.
        The reading of earlier unions stops at a commit that cannot be read, which is
        logged.
    """
    merge = below_stage(project_dir, commit)
    if held is None:
        held = _is_part_held(project_dir, commit, merge)
    part_list = [{"tip": commit, "merge": merge, "held": held,
                  "told": commit_record_list(project_dir, base, commit)}]
    earlier = stand_on_of(project_dir, merge)
    while earlier:
        try:
            merge = below_stage(project_dir, earlier)
            start = first_parent(project_dir, merge)
        except RuntimeError as e:
            logger.info("cannot read what the union under %s left: %s",
                        gitcmd.short(earlier), e)
            break
        held = held or _is_part_held(project_dir, earlier, merge)
        part_list.insert(0, {"tip": earlier, "merge": merge, "held": held,
                             "told": commit_record_list(project_dir, start, earlier)})
        earlier = stand_on_of(project_dir, merge)
    return part_list


def stand_on_of(project_dir: str, merge: str) -> str:
    """Return the earlier union a union's merge was recorded to stand on.

    Args:
        project_dir: The repository to read.
        merge: The union's merge, as below_stage() finds it under the union's tip.

    Returns:
        The full hash its stand-on trailer names, or "" when it names none or holds
        anything but one full hash.
    """
    value = own_record_at(project_dir, merge).get("stand_on", "")
    return value if OBJECT_NAME.fullmatch(value) else ""


def union_gap_list(part_list: list[dict]) -> list[dict]:
    """Return the cases a union's parts say the change does not do.

    Args:
        part_list: union_part_list()'s return value.

    Returns:
        gap_list_of() over the records of the union's own part and of the earlier
        unions' parts the working tree holds, oldest first.
    """
    return gap_list_of([record_dict for part_dict in part_list
                        if part_dict is part_list[-1] or part_dict["held"]
                        for record_dict in part_dict["told"]])


def gap_list_of(record_list: list[dict]) -> list[dict]:
    """Return the cases a list of commit records says the change does not do.

    A case a later commit records as done ("gap_done") is left out.

    Args:
        record_list: commit_record_list()'s return value, oldest first.

    Returns:
        [{"what", "why"}, ...], each case once, in the order they were recorded.
    """
    gap_list = []
    for record_dict in record_list:
        for entry_dict in decision_list_of(record_dict.get("gap_done", "")):
            while entry_dict in gap_list:
                gap_list.remove(entry_dict)
        for entry_dict in decision_list_of(record_dict.get("gap", "")):
            if entry_dict not in gap_list:
                gap_list.append(entry_dict)
    return gap_list


def earlier_record_dict(project_dir: str, name: str) -> dict | None:
    """Return what the merging run that left a branch recorded.

    Args:
        project_dir: The repository to read.
        name: The branch the earlier merge left.

    Returns:
        {"branch", "ended", "why", "decisions": [{"what", "why"}, ...] cut to
        config.INTEGRATE_WORK_MAX_ITEM}, from the last record on the branch, or None
        when the branch carries no record.

    Raises:
        RuntimeError: When the repository holds no branch of that name.
    """
    record_list = branch_record_dict(project_dir, name)["told"]
    if not record_list:
        return None
    last_dict = record_list[-1]
    return {"branch": name, "ended": last_dict["ended"], "why": last_dict["why"],
            "decisions": decision_list_of(last_dict["decisions"])[:config.INTEGRATE_WORK_MAX_ITEM]}


def union_base(project_dir: str, start: str, onto: str, onto_commit: str = "") -> str:
    """Return the commit the union is written out against.

    Args:
        project_dir: The repository the branches are in.
        start: The commit every branch starts from.
        onto: The branch the result is to be merged into afterwards, or "".
        onto_commit: The commit of onto to write against. Given, it is used in place of
            whatever onto points at now.

    Returns:
        onto's commit when one was given and start does not already reach it, start
        otherwise.
    """
    if not onto:
        return start
    head = onto_commit or commit_of(project_dir, onto)
    if gitcmd.is_ancestor(project_dir, head, start):
        logger.info("%s is already in %s, so it is not merged again", onto,
                    gitcmd.short(start))
        return start
    return head


def assemble(project_dir: str, base: str, branch_list: list[str],
             onto: str = "", onto_commit: str = "", resume: str = "") -> dict:
    """Merge the branches onto the base one after another and put the result in place.

    The branches are merged in the order they were passed. Nothing is written to the
    working tree until every merge has been made; the repository then holds the whole
    union as an uncommitted change against whatever HEAD is left at.

    Where the result is going to be merged into a branch, that branch is merged here
    like any other and HEAD is left on it. The union then already holds everything that
    branch has, and the change written out is what the union adds on top of it: the
    merge afterwards is a fast-forward. A branch the base already reaches adds nothing
    and is not merged again.

    A branch that is the union an earlier merge left unsettled - resume, and any
    other branch whose merge carries this tool's record and is none a merging run
    settled - is merged for the tree it holds and is not one of the sides: the sides
    are the commits it was merged from (earlier_side_list), merged once more here
    onto the commit it was written against to read what git could not settle
    between them (_merge_side_again). The sides its commit names as not merged into
    a file of it (_no_merge_record_at) are kept with that file here as well
    (_carry_no_merge), and so is every commit it was made from, with each file
    whose change of it the union is not found to hold (_open_no_merge_list).

    What the union may hold of the lines a conflict block is told by is read here
    once: the name lines the start and the sides hold themselves (_name_frame_dict),
    and the separator lines of the files a block was written in (_carry_separator_dict):
    those of the blocks the merges made here wrote, what each unsettled union among
    the branches records it may hold, and those a side holds in a file its change
    was not merged into (_no_merge_separator_dict). A conflicted path is unmarked
    when the merge that reported it wrote no block in it.

    A file that holds a conflict block, or what an unsettled union holds of one, is
    merged no further: where a merge changes it, the merged tree is given the file
    as it was (_keep_block_file), and the file is kept with every side whose change
    to it is not in the tree, under that side's "in_block". The path a side holds
    such a file at is read through the commits the two merged parted at
    (_fork_path_dict).

    The flow:
    1. Order the merges and read which branches are unsettled unions.
    2. Read what the unsettled unions hold of conflict blocks (_more_size_by_tip_dict,
       _tip_line_dict).
    3. Merge the branches one after another (_merge_branch_list), then take the refs of
       the merges away.
    4. Read the marker lines of the union (_union_more_size_dict).
    5. Keep the sides each unsettled union names as not merged (_carry_no_merge).
    6. Put the union in place.

    Args:
        project_dir: The repository to merge in, holding nothing uncommitted.
        base: The commit every branch starts from.
        branch_list: The branch names, in the order they are to be merged.
        onto: The branch the result is to be merged into afterwards, or "" when the
            result is left as a branch of its own.
        onto_commit: The commit of onto to write against, as union_base() takes it.
        resume: The branch among branch_list an earlier merge left, or "". Its
            merge is read as a union whether or not it carries this tool's trailers.

    Returns:
        {"start" (the commit every branch starts from), "base" (the commit the union is
        written out against: onto's commit when one was merged, start otherwise),
        "commit" (the accumulated merge commit), "branches": [{"branch", "commit",
        "conflicted" (paths), "in_block" ({a file of the union that was kept as it was
        where the change of that branch to it was to be merged, or that git left without
        that branch's version in view (_hold_unmerged_file): the path that branch holds
        the file at, "" for none, None where it cannot be told}), "messages", "said"
        (what the runs on that branch were asked for), "did" (what they said they did),
        both read by commit_record_list() and cut to config.EDIT_HISTORY_REQUEST_CHAR
        and config.EDIT_HISTORY_DID_CHAR}, ...] (the sides: onto when it was merged,
        then every branch passed, the sides of an unsettled union standing in its place
        under side_name()'s names, then, with "carry" True, each commit an unsettled
        union names as not merged into a file that is none of those, under side_name()'s
        name), "parents" (the commit of every branch passed, in the order passed),
        "conflicted" (every conflicted path, sorted), "unmarked" (those git wrote no
        conflict marker in: a file git does not merge as text, or one side deleted),
        "messages" (every conflict message), "no_path" (those of them that name no file
        of the merged tree), "marker" (_name_frame_dict()'s "name_count" and "name_at",
        "separator" (_carry_separator_dict()'s mapping) and "size" (marker_size_dict()
        for the files of the merges made here and of the unsettled unions among the
        branches)), "stand_in" (the files of the union under a name git made up for a
        file it could not place: _stand_in_path_list)}.

    Raises:
        RuntimeError: When a branch cannot be resolved, merged or committed.
    """
    # == Step 1: the order of the merges and the unsettled unions =============
    order_list, head = list(branch_list), union_base(project_dir, base, onto, onto_commit)
    commit_by_name_dict = {name: commit_of(project_dir, name) for name in order_list}
    if head != base:
        order_list.insert(0, onto)
        commit_by_name_dict[onto] = head
    # The unsettled unions among the branches passed, each with the commits it was
    # merged from
    side_by_name_dict = {
        name: side_commit_list for name in branch_list
        if (side_commit_list := earlier_side_list(project_dir, commit_by_name_dict[name],
                                                  is_union=name == resume))}
    ref_name_list: list[str] = []

    # == Step 2: what the unsettled unions hold of conflict blocks ============
    name_frame_dict = _name_frame_dict(project_dir, base, [
        side_commit for name in order_list
        for side_commit in side_by_name_dict.get(name) or [commit_by_name_dict[name]]])
    # The marker lengths and the marker lines of the unsettled unions, and the files
    # of each that hold a line it left of a conflict block
    tip_list = list(dict.fromkeys(commit_by_name_dict[name] for name in side_by_name_dict))
    size_dict = marker_size_dict(project_dir, tip_list)
    more_size_by_tip_dict = _more_size_by_tip_dict(project_dir, tip_list)
    line_dict = _tip_line_dict(project_dir, tip_list, size_dict, more_size_by_tip_dict)
    tip_block_dict = {tip: _tip_block_path_set(project_dir, tip, line_dict[tip],
                                               name_frame_dict) for tip in tip_list}

    # == Step 3: merge the branches one after another =========================
    try:
        (union_commit, merge_list, commit_list, no_path_list, step_list, no_merge_list,
         now_by_tip_dict) = _merge_branch_list(project_dir, base, onto, order_list,
                                               commit_by_name_dict, side_by_name_dict,
                                               tip_block_dict, ref_name_list)
        # The marker lengths are read while the working tree is as the merges above
        # read it
        step_commit_list = list(dict.fromkeys(
            [union_commit, *[step_commit for step_commit, conflict_list in step_list
                             if conflict_list]]))
        size_dict = {**size_dict, **marker_size_dict(project_dir, step_commit_list)}
    finally:
        gitcmd.run(project_dir, "update-ref", "-d", UNION_REF)
        for name in ref_name_list:
            gitcmd.run(project_dir, "update-ref", "-d", f"{SIDE_REF_ROOT}/{name}")

    # == Step 4: the marker lines of the union ================================
    now_by_tip_dict = {tip: _union_path_dict(project_dir, tip, union_commit,
                                             now_by_tip_dict.get(tip, {}))
                       for tip in tip_list}
    more_size_dict = _union_more_size_dict(tip_list, more_size_by_tip_dict, now_by_tip_dict)
    line_dict = {**line_dict, **_marker_line_dict(project_dir, step_commit_list,
                                                  size_dict=size_dict,
                                                  more_size_dict=more_size_dict)}
    unmark_set = {path for step_commit, conflict_list in step_list if conflict_list
                  for path in set(conflict_list) - set(_block_separator_dict(
                      line_dict[step_commit], ref_name_list))}
    side_commit_list = [entry_dict["commit"] for entry_dict in merge_list]

    # == Step 5: the sides the unsettled unions name as not merged ============
    # The sides each unsettled union names as not merged into a file of it are kept
    # with those files in this run too
    for name in side_by_name_dict:
        tip = commit_by_name_dict[name]
        record_list = [
            *(_no_merge_record_at(project_dir, below_stage(project_dir, tip)) or []),
            *_open_no_merge_list(project_dir, tip, is_union=name == resume)]
        _carry_no_merge(project_dir, base, merge_list, record_list, now_by_tip_dict[tip])

    # == Step 6: put the union in place =======================================
    marker_dict = {**name_frame_dict,
                   "separator": _carry_separator_dict(
                       project_dir, union_commit, tip_list, line_dict, ref_name_list,
                       now_by_tip_dict,
                       _no_merge_separator_dict(project_dir, no_merge_list, size_dict)),
                   "size": size_dict}
    stand_in_list = _stand_in_path_list(project_dir, union_commit, [base, *side_commit_list])
    if head != base:
        _move_head(project_dir, head)
    place(project_dir, union_commit)
    conflict_list = sorted({path for entry_dict in merge_list
                            for path in entry_dict["conflicted"]})
    return {
        "start": base,
        "base": head,
        "commit": union_commit,
        "branches": merge_list,
        "parents": commit_list[1:] if head != base else commit_list,
        "conflicted": conflict_list,
        "unmarked": sorted(unmark_set),
        "messages": [message for entry_dict in merge_list
                     for message in entry_dict["messages"]],
        "no_path": no_path_list,
        "marker": marker_dict,
        "stand_in": stand_in_list,
    }


def _more_size_by_tip_dict(project_dir: str, tip_list: list[str]
                           ) -> dict[str, dict[str, set[int]]]:
    """Return the separator line lengths each unsettled union's own record names.

    Args:
        project_dir: The repository.
        tip_list: The tips of the unsettled unions.

    Returns:
        {tip: _line_size_dict()'s mapping over the lines the record of the commit under
        the tip's stages (_separator_record_at) names, {} for none}.

    Raises:
        RuntimeError: When the trailers of a union's commit could not be read.
    """
    return {
        tip: _line_size_dict([
            (path, line) for path, count_dict in (_separator_record_at(
                project_dir, below_stage(project_dir, tip)) or {}).items()
            for line in count_dict])
        for tip in tip_list}


def _tip_line_dict(project_dir: str, tip_list: list[str], size_dict: dict[str, int],
                   more_size_by_tip_dict: dict[str, dict[str, set[int]]]
                   ) -> dict[str, list[tuple[str, bytes]]]:
    """Return the marker lines of the unsettled unions, each tip read on its own.

    Args:
        project_dir: The repository.
        tip_list: The tips of the unsettled unions.
        size_dict: marker_size_dict()'s lengths for the tips.
        more_size_by_tip_dict: _more_size_by_tip_dict()'s mapping for the tips.

    Returns:
        {tip: _marker_line_dict()'s lines for it}.

    Raises:
        RuntimeError: When a tree could not be read.
    """
    line_dict: dict[str, list[tuple[str, bytes]]] = {}
    for tip in tip_list:
        line_dict.update(_marker_line_dict(project_dir, [tip], size_dict=size_dict,
                                           more_size_dict=more_size_by_tip_dict[tip]))
    return line_dict


def _merge_branch_list(project_dir: str, base: str, onto: str, order_list: list[str],
                       commit_by_name_dict: dict[str, str],
                       side_by_name_dict: dict[str, list[str]],
                       tip_block_dict: dict[str, set[str]], ref_name_list: list[str]
                       ) -> tuple[str, list[dict], list[str], list[str],
                                  list[tuple[str, list[str]]],
                                  list[tuple[str, str | None, str, str]],
                                  dict[str, dict[str, str]]]:
    """Merge the branches onto the base one after another.

    Each branch is merged into what has been merged so far (_merge_in). The sides of an
    unsettled union are merged once more onto the commit it was written against
    (_merge_side_again), and the files a merge kept as they were are kept with the sides
    whose change to them is not in the merged tree (_add_no_merge), and so is each
    conflicted file git left without a side's version, with that side
    (_hold_unmerged_file). The refs the merges write are left in place.

    Args:
        project_dir: The repository to merge in.
        base: The commit every branch starts from.
        onto: The branch the result is to be merged into afterwards, or "".
        order_list: The names of the branches, in the order they are merged.
        commit_by_name_dict: {name: the branch's commit} for every name of order_list.
        side_by_name_dict: {name: the commits it was merged from} for the unsettled
            unions among the branches.
        tip_block_dict: {an unsettled union's tip: _tip_block_path_set()'s files}.
        ref_name_list: As _merge_in() takes it.

    Returns:
        (the accumulated merge commit, assemble()'s "branches" so far, the commit of
        every name of order_list in that order, the conflict messages that name no file
        of the merged tree, (merge commit, conflicted paths) of every merge made, (side
        commit, path on the side, path in the merged tree, commit whose contents were
        kept) of every side whose change to a file was not merged,
        {an unsettled union's tip: {a path of the tip: its path in the merged tree}} for
        the files of tip_block_dict followed from merge to merge).

    Raises:
        RuntimeError: When a branch cannot be merged or committed.
    """
    union_commit, merge_list, commit_list = base, [], []
    no_path_list: list[str] = []
    # Every merge made here for the union and between the sides of an unsettled union,
    # as (its commit, the paths git could not settle in it)
    step_list: list[tuple[str, list[str]]] = []
    # The files of what has been merged so far that are merged no further, and every
    # (side commit, path on the side, path in the merged tree, commit whose contents
    # were kept) of a side whose change to one was not merged
    block_path_set: set[str] = set()
    no_merge_list: list[tuple[str, str | None, str, str]] = []
    # Where the merged tree holds each file an unsettled union left a line of a block
    # in, as {the union's tip: {the file's path in it: its path in the merged tree}}
    now_by_tip_dict: dict[str, dict[str, str]] = {}
    for name in order_list:
        commit = commit_by_name_dict[name]
        commit_list.append(commit)
        under_commit = union_commit
        step_dict = _merge_in(project_dir, union_commit, name, commit, ref_name_list,
                              block_path_set, tip_block_dict.get(commit))
        union_commit, conflict_list = step_dict["commit"], step_dict["conflicted"]
        step_list.append((union_commit, conflict_list))
        no_path_list += step_dict["no_path"]
        now_by_tip_dict = {
            tip: {path: step_dict["now_under"][now_path]
                  for path, now_path in now_dict.items()
                  if now_path in step_dict["now_under"]}
            for tip, now_dict in now_by_tip_dict.items()}
        if commit in tip_block_dict:
            now_by_tip_dict[commit] = step_dict["now_side"]
        no_merge_list += _add_no_merge(
            project_dir, merge_list, commit,
            {path: side_path
             for path, (side_path, _) in step_dict["keep_side"].items()})
        side_commit_list = side_by_name_dict.get(name, [])
        if side_commit_list:
            entry_list, again_no_path_list = _merge_side_again(
                project_dir, base,
                first_parent(project_dir, below_stage(project_dir, commit)),
                side_commit_list, ref_name_list, step_list)
            no_path_list += again_no_path_list
            # What git could not settle of the union itself is kept with the last of the
            # sides it was merged from
            last_dict = entry_list[-1]
            last_dict["conflicted"] = sorted({*last_dict["conflicted"], *conflict_list})
            last_dict["messages"] += step_dict["messages"]
            no_merge_list += _add_no_merge(
                project_dir, entry_list, under_commit,
                {path: under_path
                 for path, (under_path, _) in step_dict["keep_under"].items()})
            merge_list += entry_list
        else:
            # Of the branch the union lands on, only the commits this tool wrote are
            # read. A file put back with no path of the branch told is kept with the
            # branch as one whose path cannot be told
            merge_list.append(_side_entry(
                project_dir, base, name, commit, conflict_list, step_dict["messages"],
                own_only=name == onto,
                in_block_dict={path: side_path or None for path, (_, side_path)
                               in step_dict["keep_under"].items()}))
            no_merge_list += [(commit, side_path, path, under_commit) for path, side_path
                              in merge_list[-1]["in_block"].items()]
        _hold_unmerged_file(project_dir, merge_list, base, union_commit, under_commit,
                            commit, conflict_list, step_dict["block"])
        block_path_set = step_dict["block"]
        logger.info("merged %s: %d conflicted file(s), %d file(s) kept as they were",
                    name, len(conflict_list),
                    len(step_dict["keep_under"]) + len(step_dict["keep_side"]))
    return (union_commit, merge_list, commit_list, no_path_list, step_list, no_merge_list,
            now_by_tip_dict)


def _union_more_size_dict(tip_list: list[str],
                          more_size_by_tip_dict: dict[str, dict[str, set[int]]],
                          now_by_tip_dict: dict[str, dict[str, str]]
                          ) -> dict[str, set[int]]:
    """Return the lengths each union's record names for a file, by its path in the union.

    Args:
        tip_list: The tips of the unsettled unions.
        more_size_by_tip_dict: _more_size_by_tip_dict()'s mapping for the tips.
        now_by_tip_dict: {tip: _union_path_dict()'s mapping for it}.

    Returns:
        {the path in the merged tree: the lengths}, a file the merged tree holds at no
        path left out.
    """
    more_size_dict: dict[str, set[int]] = {}
    for tip in tip_list:
        for path, size_set in more_size_by_tip_dict[tip].items():
            if path in now_by_tip_dict[tip]:
                more_size_dict.setdefault(now_by_tip_dict[tip][path], set()).update(size_set)
    return more_size_dict


def _merge_in(project_dir: str, union_commit: str, name: str, commit: str,
              ref_name_list: list[str], block_path_set: set[str] | None = None,
              side_path_set: set[str] | None = None) -> dict:
    """Merge one more commit into what has been merged so far and commit the result.

    merge-tree writes the names it is given into the markers: a ref of this tool's own
    for what has been merged so far, and one for the commit under the name its side goes
    by. Both refs are left in place.

    A file that is merged no further and that the merge changed is put back as it
    was (_keep_block_file), and is no conflicted path of this merge.

    Args:
        project_dir: The repository to merge in.
        union_commit: What has been merged so far.
        name: The name the commit's side goes by.
        commit: The commit merged.
        ref_name_list: The names a ref under SIDE_REF_ROOT was written for, added to
            here once each.
        block_path_set: The files of union_commit that are merged no further. None
            holds none.
        side_path_set: The files of commit that are merged no further
            (_tip_block_path_set). None holds none.

    Returns:
        {"commit" (the new merge commit), "conflicted", "messages", "no_path" (as
        _merge_tree() returns them, the files put back left out of "conflicted"),
        "keep_under" and "keep_side" (the files of the merged tree put back as
        union_commit and as commit holds them, each {the file's path in the merged
        tree: (the path the one whose contents were kept holds it at, the path the
        other of the two holds it at as _one_path() gives it)}), "now_under" and
        "now_side" ({a path of union_commit, of commit: the path in the merged tree}
        for every file of block_path_set and side_path_set the merged tree holds),
        "block" (the files of the merged tree that are merged no further: those
        files, and the conflicted paths holding a line that starts a conflict
        block)}, every path as gitcmd.shown_text() gives it.

    Raises:
        RuntimeError: When a ref could not be written, or the merge could not be run
            or committed.
    """
    if name not in ref_name_list:
        ref_name_list.append(name)
    _write_ref(project_dir, UNION_REF, union_commit)
    _write_ref(project_dir, f"{SIDE_REF_ROOT}/{name}", commit)
    tree, conflict_list, message_list, no_path_list = _merge_tree(
        project_dir, UNION_REF, f"{SIDE_REF_ROOT}/{name}")
    tree, row_list = _keep_block_file(project_dir, tree, union_commit, commit,
                                      block_path_set or set(), side_path_set or set())
    keep_dict_tuple: tuple[dict, dict] = ({}, {})
    now_dict_tuple: tuple[dict[str, str], dict[str, str]] = ({}, {})
    for is_side, path, other_path, now_path, is_put_back in row_list:
        now_dict_tuple[is_side][path] = now_path
        # A file of commit at the path of a file of union_commit is found at that
        # file's path
        if not is_side and other_path in (side_path_set or ()):
            now_dict_tuple[1].setdefault(other_path, now_path)
        if is_put_back:
            keep_dict_tuple[is_side][now_path] = (path, other_path)
    conflict_list = [path for path in conflict_list
                     if path not in keep_dict_tuple[0] and path not in keep_dict_tuple[1]]
    now_block_set = {now_path for _, _, _, now_path, _ in row_list}
    if conflict_list:
        now_block_set |= set(conflict_list) & {
            gitcmd.shown_text(path) for path in _block_file_dict(project_dir, [tree])[tree]}
    merge_commit = _commit_tree(project_dir, tree, union_commit, commit, name)
    return {"commit": merge_commit, "conflicted": conflict_list, "messages": message_list,
            "no_path": no_path_list, "keep_under": keep_dict_tuple[0],
            "keep_side": keep_dict_tuple[1], "now_under": now_dict_tuple[0],
            "now_side": now_dict_tuple[1], "block": now_block_set}


def _block_file_dict(project_dir: str, tree_list: list[str]) -> dict[str, list[str]]:
    """Return the files of some trees that hold a line starting a conflict block.

    Args:
        project_dir: The repository.
        tree_list: The commits or the trees to read, each named without a ":".

    Returns:
        {tree: [repository-relative path as os.fsdecode() gives it, ...]}, a file
        holding a line _BLOCK_START_PATTERN matches.

    Raises:
        RuntimeError: When a tree could not be read.
    """
    file_dict: dict[str, list[str]] = {tree: [] for tree in tree_list}
    proc = gitcmd.run(project_dir, "grep", "-a", "-z", "-l", "--no-color", "-E", "-e",
                      _BLOCK_START_PATTERN, *tree_list, "--", text=False)
    # git grep ends with 1 when no line matched
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"could not read the conflict blocks of "
                           f"{list_text([gitcmd.short(tree) for tree in tree_list])}: "
                           f"{gitcmd.shown_text(proc.stderr or proc.stdout).strip()}")
    for record in (proc.stdout or b"").split(b"\0"):
        tree, _, raw_path = record.partition(b":")
        if raw_path:
            file_dict[os.fsdecode(tree)].append(os.fsdecode(raw_path))
    return file_dict


def _file_entry_dict(project_dir: str, tree: str) -> dict[str, tuple[bytes, bytes, bytes]]:
    """Return the files a tree holds, each with its mode and the object of its contents.

    Args:
        project_dir: The repository.
        tree: The commit or the tree to read.

    Returns:
        {repository-relative path as gitcmd.shown_text() gives it: (the mode, the
        object name, the path as git holds it)} for every file and symbolic link. A
        submodule has no entry.

    Raises:
        RuntimeError: When the tree could not be read.
    """
    proc = gitcmd.run(project_dir, "ls-tree", "-r", "-z", "--full-tree", tree, text=False)
    if proc.returncode != 0:
        raise RuntimeError(f"could not read the files of {gitcmd.short(tree)}: "
                           f"{gitcmd.shown_text(proc.stderr).strip()}")
    entry_dict = {}
    # One record is "<mode> <type> <object>", a tab and the path
    for record in proc.stdout.split(b"\0"):
        head, _, raw_path = record.partition(b"\t")
        field_list = head.split(b" ")
        if raw_path and len(field_list) == 3 and field_list[1] == b"blob":
            entry_dict[gitcmd.shown_text(raw_path)] = (field_list[0], field_list[2], raw_path)
    return entry_dict


# The mode a tree holds a symbolic link under
_LINK_MODE = b"120000"


def _keep_block_file(project_dir: str, tree: str, under_commit: str, commit: str,
                     under_path_set: set[str], side_path_set: set[str]
                     ) -> tuple[str, list[tuple[bool, str, str | None, str, bool]]]:
    """Put the files a merge is not to change back into the tree it made.

    Each file is looked for in the merged tree at its own path, then at the path
    the other of the two commits merged holds it at (_fork_path_dict). Where the
    merged tree holds other contents there, the entry is given the contents the
    file had, under the mode the merged tree holds it with, or under the mode it
    had where one of the two is a symbolic link. A file of commit at a path taken
    by one of under_commit is left as that one.

    Args:
        project_dir: The repository.
        tree: The tree git made of the merge of commit into under_commit.
        under_commit: What had been merged before.
        commit: The commit merged.
        under_path_set: The files of under_commit that are merged no further.
        side_path_set: The files of commit that are merged no further.

    Returns:
        (the tree with those files as they were, which is tree itself when no file
        was put back, [(False for a file of under_path_set and True for one of
        side_path_set, the path the commit it is of holds it at, the path the
        other commit holds it at as _one_path() gives it, its path in the merged
        tree, whether it was put back), ...] for every file of the two sets the
        merged tree holds, those of under_path_set first), the paths as
        gitcmd.shown_text() gives them.

    Raises:
        RuntimeError: When a tree could not be read or written.
    """
    if not under_path_set and not side_path_set:
        return tree, []
    now_dict = _file_entry_dict(project_dir, tree)
    write_list: list[tuple[bytes, bytes, bytes]] = []
    row_list: list[tuple[bool, str, str | None, str, bool]] = []
    now_path_set: set[str] = set()
    for is_side, holder, other, path_set in ((False, under_commit, commit, under_path_set),
                                             (True, commit, under_commit, side_path_set)):
        if not path_set:
            continue
        holder_dict = _file_entry_dict(project_dir, holder)
        fork_dict = _fork_path_dict(project_dir, holder, other)
        for path in sorted(path_set & set(holder_dict)):
            other_path = _one_path(fork_dict[path])
            now_path = path if path in now_dict else other_path
            if now_path not in now_dict or now_path in now_path_set:
                continue
            now_path_set.add(now_path)
            (mode, name, raw_path), (own_mode, own_name, _) = now_dict[now_path], holder_dict[path]
            if name != own_name:
                is_link_change = (mode == _LINK_MODE) != (own_mode == _LINK_MODE)
                write_list.append((own_mode if is_link_change else mode, own_name, raw_path))
            row_list.append((is_side, path, other_path, now_path, name != own_name))
    if write_list:
        tree = _tree_with_entry(project_dir, tree, write_list)
    return tree, row_list


def _tree_with_entry(project_dir: str, tree: str,
                     entry_list: list[tuple[bytes, bytes, bytes]]) -> str:
    """Write a tree that holds some files otherwise than another tree does.

    Args:
        project_dir: The repository to write in. Its own index and its working tree
            are not touched.
        tree: The tree to start from.
        entry_list: [(the mode, the object name, the path as git holds it), ...],
            each replacing what tree holds at that path.

    Returns:
        The new tree's object name.

    Raises:
        RuntimeError: When the tree could not be written.
    """
    with tempfile.TemporaryDirectory() as directory:
        env_dict = {"GIT_INDEX_FILE": os.path.join(directory, "index")}
        proc = gitcmd.run(project_dir, "read-tree", tree, env_dict=env_dict, text=False)
        if proc.returncode == 0:
            proc = gitcmd.run(
                project_dir, "update-index", "-z", "--index-info", env_dict=env_dict,
                input_text=b"".join(b"%s %s\t%s\0" % entry for entry in entry_list),
                text=False)
        if proc.returncode == 0:
            proc = gitcmd.run(project_dir, "write-tree", env_dict=env_dict, text=False)
    new_tree = os.fsdecode(proc.stdout).strip()
    if proc.returncode != 0 or not OBJECT_NAME.match(new_tree):
        raise RuntimeError(f"could not write the merged tree with {len(entry_list)} "
                           f"file(s) as they were before the merge: "
                           f"{gitcmd.shown_text(proc.stderr or proc.stdout).strip()}")
    return new_tree


def _tip_block_path_set(project_dir: str, tip: str, line_list: list[tuple[str, bytes]],
                        name_frame_dict: dict) -> set[str]:
    """Return the files of an unsettled union that hold a line left of a conflict block.

    Args:
        project_dir: The repository.
        tip: The tip of the unsettled union.
        line_list: _marker_line_dict()'s lines for that tip.
        name_frame_dict: _name_frame_dict()'s value for the start and the sides.

    Returns:
        The paths holding a name line over what name_frame_dict allows
        (_marker_path_set), and those holding a separator line over what the
        union's own commit records it may hold (_over_separator_dict).

    Raises:
        RuntimeError: When the trailers of the union's commit could not be read.
    """
    over_by_path_dict = _over_separator_dict(project_dir, tip, line_list)
    return _marker_path_set(collections.Counter(line_list), name_frame_dict) \
        | {path for path, over_dict in over_by_path_dict.items() if over_dict}


def _add_no_merge(project_dir: str, entry_list: list[dict], commit: str,
                  keep_dict: dict[str, str]
                  ) -> list[tuple[str, str | None, str, str]]:
    """Keep the files a merge left as one commit holds them, with the sides not merged in.

    The files are kept as one of the two commits merged holds them, with the sides of
    the other whose change to them is not in the merged tree. A side is one of those
    when it holds the file otherwise than a commit it and the commit whose contents were
    kept both reach last does (_is_side_change), the file read at the path each of those
    commits holds it at (_fork_path_dict). Where that holds for no side, every side that
    holds the file otherwise than the commit whose contents were kept is one, and where
    that holds for none either, every side is one that holds the file at a path that
    cannot be told.

    Args:
        project_dir: The repository.
        entry_list: assemble()'s "branches" for the sides of the commit whose
            contents were not kept, whose "in_block" is added to.
        commit: The commit whose contents were kept.
        keep_dict: {a file kept, by its path in the merged tree: the path commit
            holds it at}.

    Returns:
        [(the side's commit, the path the side holds the file at as _one_path()
        gives it, the file's path in the merged tree, commit), ...] for every
        "in_block" entry added.

    Raises:
        RuntimeError: When a tree could not be read.
    """
    if not keep_dict or not entry_list:
        if keep_dict:
            logger.warning("%s kept as %s holds them, and no side was merged with it for "
                           "the run to be held to", list_text(sorted(keep_dict)),
                           gitcmd.short(commit))
        return []
    entry_by_commit_dict: dict[str, dict] = {}
    no_merge_list: list[tuple[str, str | None, str, str]] = []

    def object_at(tree: str, path: str | None) -> bytes:
        """Return the object name of the file a tree holds at a path, b"" for none."""
        if tree not in entry_by_commit_dict:
            entry_by_commit_dict[tree] = _file_entry_dict(project_dir, tree)
        return entry_by_commit_dict[tree].get(path, (b"", b"", b""))[1]

    fork_by_side_dict = {entry_dict["commit"]: _fork_path_dict(project_dir, commit,
                                                               entry_dict["commit"])
                         for entry_dict in entry_list}
    for path, keep_path in keep_dict.items():
        side_list = [(entry_dict, fork_by_side_dict[entry_dict["commit"]].get(keep_path, []))
                     for entry_dict in entry_list]
        change_list = [(entry_dict, _one_path(fork_list))
                       for entry_dict, fork_list in side_list
                       if _is_side_change(fork_list, entry_dict["commit"], object_at)]
        if not change_list:
            change_list = [(entry_dict, side_path) for entry_dict, fork_list in side_list
                           if (side_path := _one_path(fork_list)) is None
                           or object_at(entry_dict["commit"], side_path)
                           != object_at(commit, keep_path)]
        if not change_list:
            logger.warning("%s is kept as %s holds it, and no side merged with it "
                           "holds it otherwise, so every one of them is held to it",
                           path, gitcmd.short(commit))
            change_list = [(entry_dict, None) for entry_dict in entry_list]
        for entry_dict, side_path in change_list:
            if path not in entry_dict["in_block"]:
                entry_dict["in_block"][path] = side_path
                no_merge_list.append((entry_dict["commit"], side_path, path, commit))
    return no_merge_list


def _hold_unmerged_file(project_dir: str, entry_list: list[dict], start: str,
                        merge_commit: str, under_commit: str, commit: str,
                        conflict_list: list[str], block_set: set[str]) -> None:
    """Keep each conflicted file left without a side's version with that side.

    The file is one a merge left without the side's version of it in view. Two kinds of
    file are kept. One git put at a path neither of the two commits merged holds a file
    at, moving it into a directory another side renamed or giving it a name of its own
    making, is kept with every side that holds the file otherwise than the start does.
    One git kept a single version of and wrote no conflict block in, a file git does not
    merge as text, is kept with every side that holds the file otherwise than both the
    start and the merged tree do. The path a side holds the file at is read through the
    merge commit (_fork_path_dict).

    Args:
        project_dir: The repository.
        entry_list: assemble()'s "branches" merged so far, those of commit among
            them, whose "in_block" is added to.
        start: The commit every branch starts from.
        merge_commit: The merge commit.
        under_commit: What had been merged before.
        commit: The commit merged.
        conflict_list: The paths git could not settle in the merge.
        block_set: The files of the merged tree holding a line that starts a
            conflict block.

    Raises:
        RuntimeError: When a tree could not be read, or git could not say which
            commits two commits share.
    """
    if not conflict_list:
        return
    merge_dict = _file_entry_dict(project_dir, merge_commit)
    hold_set = (_file_entry_dict(project_dir, under_commit).keys()
                | _file_entry_dict(project_dir, commit).keys())
    place_set = {path for path in conflict_list
                 if path in merge_dict and path not in hold_set}
    path_list = [path for path in conflict_list if path in merge_dict
                 and (path in place_set or path not in block_set)]
    if not path_list:
        return
    start_dict = _file_entry_dict(project_dir, start)
    for entry_dict in entry_list:
        side_dict = _file_entry_dict(project_dir, entry_dict["commit"])
        fork_dict = _fork_path_dict(project_dir, merge_commit, entry_dict["commit"])
        for path in path_list:
            side_path = _one_path(fork_dict[path])
            if side_path == "":
                continue
            if side_path is not None:
                side_entry = side_dict[side_path][:2]
                if side_entry == start_dict.get(side_path, ())[:2] or (
                        path not in place_set and side_entry == merge_dict[path][:2]):
                    continue
            logger.info("the merged tree holds %s as git left it in a conflict, without "
                        "%s's version in view, so %s is held to it", path,
                        entry_dict["branch"], entry_dict["branch"])
            entry_dict["in_block"].setdefault(path, side_path)


def _is_side_change(fork_list: list[tuple[str, str, str]], commit: str,
                    object_at: Callable[[str, str | None], bytes]) -> bool:
    """Say whether a side changed a file since it and another commit parted.

    Args:
        fork_list: _fork_path_dict()'s entries for the file, read from the other
            commit to the side.
        commit: The side's commit.
        object_at: (a commit, a path) -> the object name of the file the commit
            holds there, b"" for none.

    Returns:
        True when the side holds something else at its path than one of the commits
        the two both reach last holds at its own, one holding the file and the
        other not too, and when the two share no commit.
    """
    return any(not fork or object_at(fork, fork_path) != object_at(commit, side_path)
               for fork, fork_path, side_path in fork_list)


def _fork_path_dict(project_dir: str, commit: str, other_commit: str
                    ) -> dict[str, list[tuple[str, str, str]]]:
    """Return where each file of one commit is at the fork and in another commit.

    The fork is where the commit and the other commit parted. A file is followed from
    each commit the two both reach last (_merge_base_list) to either of them the way git
    follows it in a merge of the two (_from_fork_dict). A file of commit none of those
    commits holds is at its own path in other_commit where other_commit holds a file
    there.

    Args:
        project_dir: The repository.
        commit: The commit whose files are read.
        other_commit: The commit they are looked for in.

    Returns:
        {the path in commit: [(a commit the two both reach last, the file's path
        there or "", its path in other_commit or ""), ...]}, one entry for each of
        those commits in git's order, and one with "" for the commit where the two
        share none. Every path is as gitcmd.shown_text() gives it.

    Raises:
        RuntimeError: When a tree could not be read, or git could not say which
            commits the two share.
    """
    own_dict = _file_entry_dict(project_dir, commit)
    other_dict = _file_entry_dict(project_dir, other_commit)
    path_dict: dict[str, list[tuple[str, str, str]]] = {path: [] for path in own_dict}
    fork_list = _merge_base_list(project_dir, commit, other_commit)
    if not fork_list:
        for path in own_dict:
            path_dict[path].append(("", "", path if path in other_dict else ""))
    for fork in fork_list:
        fork_dict = _file_entry_dict(project_dir, fork)
        other_by_fork_dict = _from_fork_dict(project_dir, fork, other_commit, fork_dict,
                                             other_dict)
        fork_by_own_dict = {
            own_path: fork_path for fork_path, own_path
            in _from_fork_dict(project_dir, fork, commit, fork_dict, own_dict).items()}
        for path in own_dict:
            fork_path = fork_by_own_dict.get(path, "")
            if fork_path:
                other_path = other_by_fork_dict.get(fork_path, "")
            else:
                other_path = path if path in other_dict else ""
            path_dict[path].append((fork, fork_path, other_path))
    return path_dict


def _from_fork_dict(project_dir: str, fork: str, commit: str, fork_dict: dict,
                    commit_dict: dict) -> dict[str, str]:
    """Return where a commit holds each file of a commit it was made from.

    Args:
        project_dir: The repository.
        fork: The commit the files are read from.
        commit: The commit they are looked for in.
        fork_dict: _file_entry_dict()'s mapping for fork.
        commit_dict: _file_entry_dict()'s mapping for commit.

    Returns:
        {the path in fork: the path in commit}: the same path where commit holds a
        file there, else the path git reads the file as moved to (_move_dict). A
        file commit holds at neither has no entry.
    """
    move_dict: dict[str, str] | None = None
    path_dict = {}
    for path in fork_dict:
        if path in commit_dict:
            path_dict[path] = path
            continue
        if move_dict is None:
            move_dict = _move_dict(project_dir, fork, commit)
        if move_dict.get(path) in commit_dict:
            path_dict[path] = move_dict[path]
    return path_dict


def _one_path(fork_list: list[tuple[str, str, str]]) -> str | None:
    """Return the one path _fork_path_dict() gives for a file in the other commit.

    Examples:
        [("1f2e", "f.txt", "g.txt")]                          -> "g.txt"
        [("1f2e", "f.txt", "")]                               -> ""
        [("1f2e", "f.txt", "g.txt"), ("9a8b", "f.txt", "")]   -> None

    Args:
        fork_list: _fork_path_dict()'s entries for one file.

    Returns:
        The path every entry gives, "" where every entry says the other commit
        holds no such file, None where the entries give different paths.
    """
    path_set = {other_path for _, _, other_path in fork_list}
    return path_set.pop() if len(path_set) == 1 else None


def _union_path_dict(project_dir: str, tip: str, union_commit: str,
                     track_dict: dict[str, str]) -> dict[str, str]:
    """Return where a union holds each file of an unsettled union merged into it.

    Args:
        project_dir: The repository.
        tip: The tip of the unsettled union.
        union_commit: The union.
        track_dict: {a path of tip: the path the merges made here left the file
            at}, for the files followed from merge to merge.

    Returns:
        {the path in tip: the path in union_commit}: track_dict's path, and for any
        other file the one path _fork_path_dict() gives. A file with neither has no
        entry.

    Raises:
        RuntimeError: When a tree could not be read.
    """
    now_dict = {}
    for path, fork_list in _fork_path_dict(project_dir, tip, union_commit).items():
        now_path = track_dict.get(path) or _one_path(fork_list)
        if now_path:
            now_dict[path] = now_path
    return now_dict


def _carry_no_merge(project_dir: str, base: str, entry_list: list[dict],
                    record_list: list[tuple[str, str | None, str]],
                    now_dict: dict[str, str]) -> None:
    """Keep the files an unsettled union names as not merged for a side with it.

    Each file is kept with that side in this run.

    Args:
        project_dir: The repository.
        base: The commit every branch starts from.
        entry_list: assemble()'s "branches", whose "in_block" is added to. A commit
            no entry is of is given one under side_name()'s name, with "carry"
            True, at the end.
        record_list: [(the side's commit, the path it holds the file at as
            _one_path() gives it, the file's path in the unsettled union), ...].
        now_dict: _union_path_dict()'s mapping for the unsettled union. A file it
            does not hold is named by its path in the unsettled union.
    """
    for commit, side_path, path in record_list:
        entry_dict = next((entry_dict for entry_dict in entry_list
                           if entry_dict["commit"] == commit), None)
        if entry_dict is None:
            entry_dict = _side_entry(project_dir, base, side_name(commit), commit, [], [],
                                     own_only=True)
            entry_dict["carry"] = True
            entry_list.append(entry_dict)
        entry_dict["in_block"].setdefault(now_dict.get(path, path), side_path)


def _open_no_merge_list(project_dir: str, tip: str,
                        is_union: bool) -> list[tuple[str, str | None, str]]:
    """Return the sides held to the files of an unsettled union by what it holds.

    Every commit the union was merged from and every commit a union opened on the
    way was written against (_open_union) is one for each file it changed that the
    union's tip is not found to hold its change of (_unheld_path_dict), whether or
    not the tip holds a line of a conflict block there.

    Args:
        project_dir: The repository.
        tip: The tip of the unsettled union.
        is_union: As union_side_list() takes it.

    Returns:
        [(the commit, the path it holds the file at as _unheld_path_dict() gives
        it, the file's path in tip), ...], a file the commit holds at no path left
        out.

    Raises:
        RuntimeError: When a commit could not be read, written or merged.
    """
    side_commit_list, onto_list = _open_union(project_dir, tip, is_union)
    commit_list = list(dict.fromkeys([*side_commit_list, *onto_list]))
    record_list = [
        (commit, side_path, path) for commit in commit_list
        for path, side_path in _unheld_path_dict(project_dir, tip, commit,
                                                 commit_list).items()
        if side_path != ""]
    if record_list:
        logger.info("the commits %s was made from are held to the %d file(s) whose "
                    "change the union is not found to hold", gitcmd.short(tip),
                    len({path for _, _, path in record_list}))
    return record_list


def _unheld_path_dict(project_dir: str, tip: str, commit: str,
                      commit_list: list[str]) -> dict[str, str | None]:
    """Return the files of a union whose tip is not found to hold one commit's change.

    The change is what the commit holds over each commit it and another of
    commit_list both reach last. The commit is merged into the tip from each of
    those (_merge_tree, over a commit holding its tree whose one parent is that
    commit, which the tip reaches); a path the merge conflicts in or leaves other
    than the tip holds it is one. The path the commit holds a file of the tip at is
    read through the commit merged from, and that of a file the merge adds through a
    commit of the merged tree (_fork_path_dict).

    Args:
        project_dir: The repository.
        tip: The tip of the union, which reaches every commit of commit_list.
        commit: The commit whose change is looked for.
        commit_list: The commits the union was made from, commit among them.

    Returns:
        {the path in the merged tree (a file of the tip, or one the merge adds):
        the path the commit holds the file at as _unheld_side_path() gives it over
        every merge that names the path, which for a path no merged tree holds is the
        path itself where the commit holds a file there, and "" where the commit
        holds none}, in the order of the paths.

    Raises:
        RuntimeError: When a commit could not be read, written or merged.
    """
    fork_list = list(dict.fromkeys(
        fork for other in commit_list if other != commit
        for fork in _merge_base_list(project_dir, commit, other) if fork != commit))
    tip_dict = _file_entry_dict(project_dir, tip)
    commit_dict = _file_entry_dict(project_dir, commit)
    # The paths, each with _fork_path_dict()'s entries for it from every merge
    fork_by_path_dict: dict[str, list[tuple[str, str, str]]] = {}
    for fork in fork_list:
        proc = gitcmd.run(project_dir, *gitcmd.COMMITTER_TUPLE, "commit-tree",
                          f"{commit}^{{tree}}", "-p", fork, "-m",
                          f"{gitcmd.short(commit)} over {gitcmd.short(fork)}")
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(f"could not write {gitcmd.short(commit)} over "
                               f"{gitcmd.short(fork)}: {(proc.stderr or proc.stdout).strip()}")
        stand_commit = proc.stdout.strip()
        tree, conflict_list, _, _ = _merge_tree(project_dir, tip, stand_commit)
        tree_dict = _file_entry_dict(project_dir, tree)
        path_set = set(conflict_list) | {
            path for path in tip_dict.keys() | tree_dict.keys()
            if (tip_dict.get(path) or ())[:2] != (tree_dict.get(path) or ())[:2]}
        if not path_set:
            continue
        fork_dict = _fork_path_dict(project_dir, tip, stand_commit)
        if not path_set <= tip_dict.keys():
            # A file the merge adds is followed back from a commit of the merged tree
            fork_dict = {**_fork_path_dict(project_dir, _commit_tree(
                project_dir, tree, tip, stand_commit, gitcmd.short(commit)),
                stand_commit), **fork_dict}
        for path in path_set:
            fork_by_path_dict.setdefault(path, []).extend(fork_dict.get(path, []))
    return {path: _unheld_side_path(path, fork_entry_list) if fork_entry_list
            else (path if path in commit_dict else "")
            for path, fork_entry_list in sorted(fork_by_path_dict.items())}


def _unheld_side_path(path: str, fork_list: list[tuple[str, str, str]]) -> str | None:
    """Return the one path _unheld_path_dict()'s merges give for a file in the commit.

    The entries of the commits that hold the file at the path the tip holds it at
    are read first, and their one path is taken when it is a path. Otherwise the
    path the entries give is taken where they give one path at most, an entry saying
    the commit holds no such file not counting against one that gives a path.

    Examples:
        ("g.txt", [("1f2e", "g.txt", "g.txt"), ("9a8b", "f.txt", "")])     -> "g.txt"
        ("g.txt", [("1f2e", "f.txt", "h.txt"), ("9a8b", "f.txt", "")])     -> "h.txt"
        ("g.txt", [("1f2e", "f.txt", "h.txt"), ("9a8b", "f.txt", "k.txt")]) -> None

    Args:
        path: The file's path in the merged tree.
        fork_list: _fork_path_dict()'s entries for the file, from every merge.

    Returns:
        The path, "" where no entry gives one, None where the entries give
        different paths.
    """
    near_path = _one_path([entry for entry in fork_list if entry[1] == path])
    if near_path:
        return near_path
    path_set = {other_path for _, _, other_path in fork_list if other_path}
    if len(path_set) > 1:
        return None
    return path_set.pop() if path_set else ""


def _write_ref(project_dir: str, ref: str, commit: str) -> None:
    """Point one of this tool's own refs at a commit.

    Args:
        project_dir: The repository to write in.
        ref: UNION_REF or a ref under SIDE_REF_ROOT.
        commit: The commit the ref is to name.

    Raises:
        RuntimeError: When git could not write the ref.
    """
    proc = gitcmd.run(project_dir, "update-ref", ref, commit)
    if proc.returncode != 0:
        raise RuntimeError(f"could not point {ref} at {gitcmd.short(commit)}: "
                           f"{(proc.stderr or proc.stdout).strip()}")


def _side_entry(project_dir: str, base: str, name: str, commit: str,
                conflict_list: list[str], message_list: list[str],
                own_only: bool = False,
                in_block_dict: dict[str, str | None] | None = None) -> dict:
    """Build one entry of assemble()'s "branches".

    Args:
        project_dir: The repository to read.
        base: The commit every branch starts from.
        name: The name the side goes by.
        commit: The side's commit.
        conflict_list: The paths git could not settle when it was merged.
        message_list: What git said of them.
        own_only: As commit_record_list() takes it.
        in_block_dict: {a file of the merged tree kept as it was when the side was
            merged: the path the side holds it at as _one_path() gives it}. None
            holds none.

    Returns:
        {"branch", "commit", "conflicted", "in_block", "messages", "said", "did"}.
    """
    record_list = commit_record_list(project_dir, base, commit, own_only=own_only)
    return {"branch": name, "commit": commit,
            "conflicted": conflict_list,
            "in_block": dict(in_block_dict or {}),
            "messages": message_list,
            "said": [cut(record_dict["asked"], config.EDIT_HISTORY_REQUEST_CHAR)
                     for record_dict in record_list if record_dict["asked"]],
            "did": [cut(record_dict["did"], config.EDIT_HISTORY_DID_CHAR)
                    for record_dict in record_list if record_dict["did"]]}


def side_name(commit: str) -> str:
    """Return the name a commit an earlier union was merged from goes by as a side.

    Args:
        commit: The full commit hash.

    Returns:
        gitcmd.short()'s form of it.
    """
    return gitcmd.short(commit)


def earlier_side_list(project_dir: str, commit: str, is_union: bool = True) -> list[str]:
    """Return the commits the union an earlier merge left unsettled was merged from.

    Args:
        project_dir: The repository to read.
        commit: The tip of a branch.
        is_union: As union_side_list() takes it: True for the branch an earlier
            merge left, False for any other branch.

    Returns:
        union_side_list()'s commits, or [] when the tip is no union left unsettled:
        a merging run settled the merge under it, the tip is not a merge commit, or
        (is_union False) the merge carries no record of this tool's.
    """
    side_commit_list = union_side_list(project_dir, commit, is_union)
    return [] if side_commit_list == [commit] else side_commit_list


def earlier_fork_dict(project_dir: str, commit: str) -> dict[str, list[str]]:
    """Return where each side of an unsettled union parts from what it was written on.

    The sides are the commits a union an earlier merge left unsettled was merged from,
    and each parts from the commit the union was written against.

    Args:
        project_dir: The repository to read.
        commit: The tip of a branch an earlier merge left.

    Returns:
        {each of earlier_side_list()'s commits: the merge bases (_merge_base_list) of
        it and each commit a merge _open_union() replaces was written against, each
        once}, in the order of the commits, or {} when the tip is no union left
        unsettled.

    Raises:
        RuntimeError: When the parents or the record of a commit could not be read,
            or git could not name the merge bases.
    """
    side_commit_list, onto_list = _open_union(project_dir, commit)
    if side_commit_list == [commit]:
        return {}
    return {side: list(dict.fromkeys(fork for onto in onto_list
                                     for fork in _merge_base_list(project_dir, side, onto)))
            for side in side_commit_list}


def _merge_side_again(project_dir: str, base: str, under_commit: str,
                      side_commit_list: list[str], ref_name_list: list[str],
                      step_list: list[tuple[str, list[str]]]
                      ) -> tuple[list[dict], list[str]]:
    """Merge the commits an earlier union was merged from once more.

    The merges read what git could not settle between them. A file holding a conflict
    block is merged no further here either (_merge_in), and is no conflicted path of the
    commits merged after it.

    Args:
        project_dir: The repository to merge in.
        base: The commit every branch starts from.
        under_commit: The commit the earlier union was written against.
        side_commit_list: earlier_side_list()'s commits, in the order they are merged.
        ref_name_list: As _merge_in() takes it.
        step_list: assemble()'s (merge commit, conflicted paths) of every merge made,
            added to here.

    Returns:
        (one _side_entry() per commit, named by side_name(), with an empty
        "in_block", the conflict messages that name no file of the tree merged).
    """
    union_commit, entry_list, no_path_list = under_commit, [], []
    block_path_set: set[str] = set()
    for commit in side_commit_list:
        step_dict = _merge_in(project_dir, union_commit, side_name(commit), commit,
                              ref_name_list, block_path_set)
        union_commit, block_path_set = step_dict["commit"], step_dict["block"]
        step_list.append((union_commit, step_dict["conflicted"]))
        no_path_list += step_dict["no_path"]
        entry_list.append(_side_entry(
            project_dir, base, side_name(commit), commit, step_dict["conflicted"],
            step_dict["messages"]))
    return entry_list, no_path_list


def branch_commit_list(merge_dict: dict) -> list[str]:
    """Return the commits assemble() merged for the branches it was passed.

    Args:
        merge_dict: assemble()'s return value.

    Returns:
        The full commit hashes, in the order the branches were passed. The branch the
        union is written against is not one of them.
    """
    return list(merge_dict["parents"])


def marker_size_dict(project_dir: str, tree_list: list[str]) -> dict[str, int]:
    """Return how long a conflict marker is in the files of some trees.

    Only the files whose markers are not _MARKER_SIZE long are returned. The length is
    the one git writes a marker with when it merges the file now: the number the file's
    conflict-marker-size attribute starts with, as the attribute files in effect in the
    working tree give it, and _MARKER_SIZE where that is not above 0.

    Args:
        project_dir: The repository.
        tree_list: The commits or trees whose files are asked about.

    Returns:
        {repository-relative path as gitcmd.shown_text() gives it: the length}, holding
        only the files whose length is not _MARKER_SIZE.

    Raises:
        RuntimeError: When a tree or the attributes could not be read.
    """
    name_set: set[bytes] = set()
    for tree in tree_list:
        proc = gitcmd.run(project_dir, "ls-tree", "-r", "-z", "--full-tree", "--name-only",
                          tree, text=False)
        if proc.returncode != 0:
            raise RuntimeError(f"could not read the files of {gitcmd.short(tree)}: "
                               f"{gitcmd.shown_text(proc.stderr).strip()}")
        name_set.update(name for name in proc.stdout.split(b"\0") if name)
    if not name_set:
        return {}
    proc = gitcmd.run(project_dir, "check-attr", "-z", "--stdin", _MARKER_SIZE_ATTR,
                      input_text=b"\0".join(sorted(name_set)) + b"\0", text=False)
    if proc.returncode != 0:
        raise RuntimeError(f"could not read the {_MARKER_SIZE_ATTR} attribute: "
                           f"{gitcmd.shown_text(proc.stderr).strip()}")
    # One answer is the path, the attribute's name and its value
    field_list = proc.stdout.split(b"\0")
    size_dict = {}
    for path, value in zip(field_list[0::3], field_list[2::3]):
        number = _LEAD_NUMBER.match(value)
        size = int(number.group()) if number else 0
        if size > 0 and size != _MARKER_SIZE:
            size_dict[gitcmd.shown_text(path)] = size
    return size_dict


def _marker_line_dict(project_dir: str, tree_list: list[str],
                      env_dict: dict[str, str] | None = None,
                      size_dict: dict[str, int] | None = None,
                      more_size_dict: dict[str, set[int]] | None = None
                      ) -> dict[str, list[tuple[str, bytes]]]:
    """Return the lines of some trees a conflict block is told by, in the order they stand.

    The contents are read as git holds them, and every file as text, whatever git's
    attributes, the user's git settings or the contents say. A symbolic link holds none.
    Two kinds of line are returned: a name line (_NAME_LINE), wherever in the line the
    name stands, and a separator line, which is a run of "=" or "|", then white space or
    the end of the line. The run of a separator line is as long as the file's markers
    are, as one of the lengths more_size_dict gives the file, or, in a file of a tree
    that holds a name line starting with "<" or ">", as the run any line of that file
    starts with that is a run of "<" or ">" and white space. A line starting with such a
    run that is no name line is returned as a separator line too where its run is one of
    those lengths and not as long as the file's markers, which is how git writes the
    block of two merge bases inside the block of a merge. A line that is both is a name
    line.

    Args:
        project_dir: The repository.
        tree_list: The commits or the trees to read, each named without a ":".
        env_dict: As gitcmd.run() takes it: the settings gitcmd.work_tree_index()
            yields, for a tree it wrote.
        size_dict: marker_size_dict()'s lengths. A file it does not hold, and every
            file when it is None, has markers _MARKER_SIZE long.
        more_size_dict: {path: the other lengths a separator line of the file is
            read with}. None gives none.

    Returns:
        {tree: [(repository-relative path as gitcmd.shown_text() gives it, the line as
        bytes without its line ending), ...]}, the lines of one file together and
        from its first line to its last.

    Raises:
        RuntimeError: When a tree could not be read.
    """
    line_dict: dict[str, list[tuple[str, bytes]]] = {tree: [] for tree in tree_list}
    if not tree_list:
        return line_dict
    size_dict, more_size_dict = size_dict or {}, more_size_dict or {}
    least_size = min([_MARKER_SIZE, *size_dict.values(),
                      *[size for size_set in more_size_dict.values() for size in size_set]])
    # -z ends each "<tree>:<path>" with a NUL, unquoted, and the line with a newline
    proc = gitcmd.run(project_dir, "grep", "-a", "-z", "--no-color", "--no-line-number",
                      "--no-column", "-E", "-e", _NAME_LINE_PATTERN,
                      "-e", _SEPARATOR_PATTERN.format(size=least_size),
                      "-e", _EDGE_PATTERN.format(size=least_size),
                      *tree_list, "--", env_dict=env_dict, text=False)
    # git grep ends with 1 when no line matched
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"could not read the conflict marker lines of "
                           f"{list_text([gitcmd.short(tree) for tree in tree_list])}: "
                           f"{gitcmd.shown_text(proc.stderr or proc.stdout).strip()}")
    # Every line git found, by (tree, path)
    find_dict: dict[tuple[str, str], list[bytes]] = {}
    start = 0
    output = proc.stdout or b""
    while (path_end := output.find(b"\0", start)) != -1:
        line_end = output.find(b"\n", path_end)
        line_end = len(output) if line_end == -1 else line_end
        tree, _, raw_path = output[start:path_end].partition(b":")
        find_dict.setdefault((os.fsdecode(tree), gitcmd.shown_text(raw_path)), []).append(
            output[path_end + 1:line_end].rstrip(b"\r"))
        start = line_end + 1
    for (tree, path), file_line_list in find_dict.items():
        own_size = size_dict.get(path, _MARKER_SIZE)
        edge_list = [line for line in file_line_list if line[:1] in (b"<", b">")]
        size_set = {own_size, *more_size_dict.get(path, ())}
        if any(_is_name_line(line) for line in edge_list):
            size_set.update(_run_size(line) for line in edge_list)
        # A separator line is told by the run of the character it starts with
        line_dict[tree] += [
            (path, line) for line in file_line_list
            if _is_name_line(line) or (_run_size(line) in size_set and (
                line[:1] in (b"=", b"|") or _run_size(line) != own_size))]
    return line_dict


def _run_size(line: bytes) -> int:
    """Return how many times the character a line starts with stands at its start.

    Examples:
        b"<<<<<<<< refs/codetwine-loom/merged-so-far:a.txt" -> 8
        b"======="                               -> 7

    Args:
        line: The line.

    Returns:
        The length of that run, 0 for an empty line.
    """
    return len(line) - len(line.lstrip(line[:1]))


def _is_name_line(line: bytes) -> bool:
    """Say whether one of _marker_line_dict()'s lines is a name line and no separator.

    Args:
        line: The line.

    Returns:
        True for a line _NAME_LINE finds a name in.
    """
    return bool(_NAME_LINE.search(line))


def _block_separator_dict(line_list: list[tuple[str, bytes]], ref_name_list: list[str]
                          ) -> dict[str, collections.Counter]:
    """Return the separator lines of the conflict blocks git wrote, file by file.

    A block is told by the names this tool merges under, wherever its file is: it
    starts at a "<" name line naming UNION_REF and ends at a ">" name line naming a
    ref under SIDE_REF_ROOT, either with ":<path>" after the name or without. An
    end ends the block begun last, and the block of an end naming a side merged
    here is one git wrote. A name line that does not start with "<" or ">" is no
    line of a block.

    Args:
        line_list: _marker_line_dict()'s lines for one tree.
        ref_name_list: The names the sides were merged under, as _merge_in() keeps them.

    Returns:
        {path: {separator line: how many of it the blocks git wrote in that file
        hold}} (_file_block_separator), for every file holding the end of one.
    """
    end_label_set = {f"{SIDE_REF_ROOT}/{name}".encode() for name in ref_name_list}
    line_by_path_dict: dict[str, list[bytes]] = {}
    for path, line in line_list:
        line_by_path_dict.setdefault(path, []).append(line)
    by_path_dict: dict[str, collections.Counter] = {}
    for path, file_line_list in line_by_path_dict.items():
        block_dict = _file_block_separator(file_line_list, end_label_set)
        if block_dict is not None:
            by_path_dict[path] = block_dict
    return by_path_dict


def _file_block_separator(line_list: list[bytes], end_label_set: set[bytes]
                          ) -> collections.Counter | None:
    """Return the separator lines of the conflict blocks git wrote in one file.

    A block's markers are as long as the run of "<" it starts with. A separator
    line of a block is a "|" line, or an "=" line with nothing after it, whose run
    is that long. One read while a block is begun and not ended is that block's own
    until the block has its "=" line: every "|" line up to there, and that "="
    line. Between its first "|" line and its "=" line, every other line
    _marker_line_dict() returns that is no name line is the block's own as well:
    the lines of a block git wrote there under names of its own. The lines of a
    block git wrote are counted when the block ends. A block whose end names no
    side merged here is not counted. A block that ends with no "|" line or no "="
    line of its own is made up for out of the separator lines of its length that
    are no counted block's own: every such "|" line, and one such "=" line for
    each block without one.

    Args:
        line_list: _marker_line_dict()'s lines for the file, from its first to its last.
        end_label_set: The refs of the sides merged here, as a block's end names them.

    Returns:
        {separator line: how many of it those blocks hold}, or None for a file
        holding the end of no block git wrote.
    """
    start_label = UNION_REF.encode()
    side_label = f"{SIDE_REF_ROOT}/".encode()
    bar, equal = b"|", b"="
    # The blocks begun and not ended, each (how long its markers are, {separator
    # character: its own lines}, its other lines between its "|" and its "=" line)
    open_list: list[tuple[int, dict[bytes, list[bytes]], list[bytes]]] = []
    # By (the character a line starts with, the length of that run): the separator
    # lines that are no counted block's own, and how many blocks git wrote ended
    # with no line of their own
    spare_dict: dict[tuple[bytes, int], list[bytes]] = {}
    short_dict: collections.Counter = collections.Counter()
    block_dict: collections.Counter | None = None

    def spare(own_size: int, own_dict: dict[bytes, list[bytes]], nest_list: list[bytes]
              ) -> None:
        """Add the lines of a block that is not counted to spare_dict."""
        for key, own_list in own_dict.items():
            spare_dict.setdefault((key, own_size), []).extend(own_list)
        for nest_line in nest_list:
            spare_dict.setdefault((nest_line[:1], _run_size(nest_line)), []).append(nest_line)

    for line in line_list:
        char, size = line[:1], _run_size(line)
        if char in (b"<", b">") and _is_name_line(line):
            label = line.lstrip(char).strip().partition(b":")[0]
            if char == b"<":
                if label == start_label:
                    open_list.append((size, {bar: [], equal: []}, []))
                continue
            if not label.startswith(side_label):
                continue
            own_size, own_dict, nest_list = open_list.pop() if open_list \
                else (size, {bar: [], equal: []}, [])
            if label not in end_label_set:
                spare(own_size, own_dict, nest_list)
                continue
            block_dict = collections.Counter() if block_dict is None else block_dict
            block_dict.update(nest_list)
            for key, own_list in own_dict.items():
                block_dict.update(own_list)
                short_dict[(key, own_size)] += not own_list
        elif _is_name_line(line) or (char == equal and len(line) != size):
            continue
        elif open_list and not open_list[-1][1][equal] and size == open_list[-1][0] \
                and char in (bar, equal):
            open_list[-1][1][char].append(line)
        elif open_list and open_list[-1][1][bar] and not open_list[-1][1][equal]:
            open_list[-1][2].append(line)
        else:
            spare_dict.setdefault((char, size), []).append(line)
    if block_dict is None:
        return None
    for own_size, own_dict, nest_list in open_list:
        spare(own_size, own_dict, nest_list)
    for (key, size), count in short_dict.items():
        spare_list = spare_dict.get((key, size), [])
        if count:
            block_dict.update(spare_list if key == bar else spare_list[:count])
    return block_dict


def _separator_count_dict(line_list: list[tuple[str, bytes]]
                          ) -> dict[str, collections.Counter]:
    """Return the separator lines each file of a tree holds.

    Args:
        line_list: _marker_line_dict()'s lines for the tree.

    Returns:
        {path: {separator line: how many of it the file holds}}, for every file
        holding one.
    """
    count_by_path_dict: dict[str, collections.Counter] = {}
    for path, line in line_list:
        if not _is_name_line(line):
            count_by_path_dict.setdefault(path, collections.Counter())[line] += 1
    return count_by_path_dict


def _separator_frame_dict(line_list: list[tuple[str, bytes]], ref_name_list: list[str],
                          carry_by_path_dict: dict[str, collections.Counter] | None = None,
                          tip_path_set: set[str] | None = None
                          ) -> dict[str, collections.Counter]:
    """Return the separator lines the files of a merge may hold with no block left.

    The files are those a conflict block was written in.

    Args:
        line_list: _marker_line_dict()'s lines for a merge as git alone made it.
        ref_name_list: The names the sides were merged under.
        carry_by_path_dict: {path: {separator line: how many of it an unsettled union
            merged here brought to the file from a block written before}}. None
            holds none.
        tip_path_set: The files the merge holds as an unsettled union merged here
            holds them, in which no merge made here wrote a block. None holds none.

    Returns:
        {path: {separator line: how many of it the file holds, less
        _block_separator_dict()'s lines where the file is none of tip_path_set and
        less carry_by_path_dict's}}, for every file holding a block that is none
        of tip_path_set and every file of carry_by_path_dict, in the order of the
        paths. Any other file has no entry.
    """
    carry_by_path_dict = carry_by_path_dict or {}
    block_by_path_dict = {
        path: block_dict for path, block_dict in _block_separator_dict(
            line_list, ref_name_list).items()
        if path not in (tip_path_set or ())}
    count_by_path_dict = _separator_count_dict(line_list)
    none_dict: collections.Counter = collections.Counter()
    return {path: count_by_path_dict.get(path, none_dict)
            - block_by_path_dict.get(path, none_dict)
            - carry_by_path_dict.get(path, none_dict)
            for path in sorted({*block_by_path_dict, *carry_by_path_dict})}


def _carry_separator_dict(project_dir: str, union_commit: str, tip_list: list[str],
                          line_dict: dict[str, list[tuple[str, bytes]]],
                          ref_name_list: list[str],
                          now_by_tip_dict: dict[str, dict[str, str]],
                          more_by_path_dict: dict[str, collections.Counter] | None = None
                          ) -> dict[str, collections.Counter]:
    """Return the separator lines the files of a union may hold with no block left.

    The files are those a conflict block was written in. A block is one a merge made
    here wrote, or one written in a file of an unsettled union merged here before that
    union was left. A file the union holds with the contents such a union holds it with,
    at the path now_by_tip_dict gives for it, may hold what that union's own commit
    records for it (_record_separator_dict), the fewest of each line where several
    record it. A file such a record names that the union holds with other contents is
    counted like a file a merge made here wrote a block in. Where the union holds a file
    of a record at no path, the separator lines that unsettled union holds in it over
    the record are taken off every file the union holds and that unsettled union does
    not.

    Args:
        project_dir: The repository.
        union_commit: The union as git alone made it.
        tip_list: The tips of the unsettled unions merged into it.
        line_dict: _marker_line_dict()'s lines for union_commit and for every tip.
        ref_name_list: The names the sides were merged under.
        now_by_tip_dict: {tip: _union_path_dict()'s mapping for it}.
        more_by_path_dict: _no_merge_separator_dict()'s mapping, added to the
            files that have an entry. None adds none.

    Returns:
        {path: {separator line: how many of it the file may hold}}, in the order of
        the paths: _separator_frame_dict()'s mapping with the files held as an
        unsettled union holds them replaced by what its record allows. A file
        holding marker lines names the "=" line of every length their runs have,
        that of the file's markers too, with 0 where it may hold none.

    Raises:
        RuntimeError: When a tree could not be read.
    """
    carry_by_path_dict: dict[str, collections.Counter] = {}
    allow_by_path_dict: dict[str, collections.Counter] = {}
    tip_path_set: set[str] = set()
    none_dict: collections.Counter = collections.Counter()
    union_entry_dict = _file_entry_dict(project_dir, union_commit) if tip_list else {}
    for tip in tip_list:
        record_by_path_dict = _record_separator_dict(project_dir, tip, line_dict[tip])
        count_by_path_dict = _separator_count_dict(line_dict[tip])
        tip_entry_dict = _file_entry_dict(project_dir, tip)
        now_dict = now_by_tip_dict[tip]
        # The files the union holds with the contents the tip holds them with, by
        # their path in the tip
        same_path_dict = {
            path: now_dict[path] for path, entry in tip_entry_dict.items()
            if path in now_dict and union_entry_dict[now_dict[path]][1] == entry[1]}
        tip_path_set.update(same_path_dict.values())
        # The files of the union holding a separator line that the tip does not hold
        new_path_list: list[str] | None = None
        for path, record_dict in sorted(record_by_path_dict.items()):
            now_path = now_dict.get(path)
            if path in same_path_dict:
                allow_by_path_dict[now_path] = record_dict & allow_by_path_dict.get(
                    now_path, record_dict)
                continue
            if now_path is not None:
                carry_by_path_dict.setdefault(now_path, collections.Counter())
                continue
            over_dict = count_by_path_dict.get(path, none_dict) - record_dict
            if not over_dict:
                continue
            if new_path_list is None:
                new_path_list = sorted(
                    set(_separator_count_dict(line_dict[union_commit]))
                    - set(tip_entry_dict))
            logger.info("%s of %s is at no path of the merged tree, so the "
                        "separator lines left in it are looked for in %d new file(s)",
                        path, gitcmd.short(tip), len(new_path_list))
            for new_path in new_path_list:
                carry_by_path_dict[new_path] = carry_by_path_dict.get(
                    new_path, none_dict) + over_dict
    frame_dict = {**_separator_frame_dict(line_dict[union_commit], ref_name_list,
                                          carry_by_path_dict, tip_path_set),
                  **allow_by_path_dict}
    for path, more_dict in (more_by_path_dict or {}).items():
        if path in frame_dict:
            frame_dict[path] = frame_dict[path] + more_dict
    # Every length a separator line of a file is read with is named with the file, by
    # an "=" line of that length
    for path, size_set in _line_size_dict(line_dict[union_commit]).items():
        for size in size_set:
            if path in frame_dict:
                frame_dict[path].setdefault(b"=" * size, 0)
    return dict(sorted(frame_dict.items()))


def _line_size_dict(line_list: list[tuple[str, bytes]]) -> dict[str, set[int]]:
    """Return how long the runs are that the marker lines of each file start with.

    Args:
        line_list: [(path, a separator line or a name line), ...], as
            _marker_line_dict() returns them for one tree.

    Returns:
        {path: the lengths (_run_size) of its separator lines and of its name lines
        that start with "<" or ">"}.
    """
    size_by_path_dict: dict[str, set[int]] = {}
    for path, line in line_list:
        if line[:1] in (b"<", b">") or not _is_name_line(line):
            size_by_path_dict.setdefault(path, set()).add(_run_size(line))
    return size_by_path_dict


def _no_merge_separator_dict(project_dir: str,
                             no_merge_list: list[tuple[str, str | None, str, str]],
                             size_dict: dict[str, int]
                             ) -> dict[str, collections.Counter]:
    """Return the separator lines the sides add to the files left without their change.

    The files are those the side's change was not merged into, and the lines are counted
    over what the file held where the side parted from the commit whose contents were
    kept.

    Args:
        project_dir: The repository.
        no_merge_list: [(the side's commit, the path the side holds the file at as
            _one_path() gives it, the file's path in the merged tree, the commit
            whose contents were kept), ...]. An entry with no path of the side is
            left out.
        size_dict: marker_size_dict()'s lengths.

    Returns:
        {the path in the merged tree: {separator line: how many more of it the
        sides hold at their path than the commits the side and the kept commit
        both reach last hold at the path the file is at there (_fork_path_dict),
        the most any of those commits holds counted}}, the counts of several sides
        of one file added.

    Raises:
        RuntimeError: When a commit could not be read.
    """
    no_merge_list = [entry for entry in no_merge_list if entry[1]]
    if not no_merge_list:
        return {}
    fork_by_pair_dict = {
        pair: _fork_path_dict(project_dir, *pair)
        for pair in dict.fromkeys((commit, keep_commit)
                                  for commit, _, _, keep_commit in no_merge_list)}
    # Each entry with the commits its side and the kept commit both reach last and
    # the file's path at each
    fork_list_list = [
        [(fork, fork_path) for fork, fork_path, _
         in fork_by_pair_dict[(commit, keep_commit)].get(side_path, []) if fork]
        for commit, side_path, _, keep_commit in no_merge_list]
    commit_list = list(dict.fromkeys(
        [commit for commit, _ in fork_by_pair_dict]
        + [fork for fork_list in fork_list_list for fork, _ in fork_list]))
    line_dict = _marker_line_dict(project_dir, commit_list, size_dict=size_dict)
    count_by_commit_dict = {commit: _separator_count_dict(line_dict[commit])
                            for commit in commit_list}
    none_dict: collections.Counter = collections.Counter()
    more_by_path_dict: dict[str, collections.Counter] = {}
    for (commit, side_path, path, _), fork_list in zip(no_merge_list, fork_list_list):
        start_dict: collections.Counter = collections.Counter()
        for fork, fork_path in fork_list:
            start_dict |= count_by_commit_dict[fork].get(fork_path, none_dict)
        more_by_path_dict[path] = more_by_path_dict.get(path, none_dict) + (
            count_by_commit_dict[commit].get(side_path, none_dict) - start_dict)
    return more_by_path_dict


def _record_separator_dict(project_dir: str, commit: str,
                           line_list: list[tuple[str, bytes]]
                           ) -> dict[str, collections.Counter]:
    """Return the separator lines an unsettled union may hold with no block left.

    Args:
        project_dir: The repository.
        commit: The tip of the unsettled union.
        line_list: _marker_line_dict()'s lines for that tip.

    Returns:
        What its commit records (_separator_record_at): {path: {separator line: how
        many of it the file may hold}} for every file the record names. For a
        union whose commit records nothing that can be read, which is logged,
        every file holding a separator line with none allowed.

    Raises:
        RuntimeError: When the trailers of the union's commit could not be read.
    """
    record_dict = _separator_record_at(project_dir, below_stage(project_dir, commit))
    if record_dict is None:
        logger.info("%s records no separator lines it may hold, so every separator "
                    "line it holds is read as left of a conflict block",
                    gitcmd.short(commit))
        return {path: collections.Counter() for path in _separator_count_dict(line_list)}
    return record_dict


def _over_separator_dict(project_dir: str, commit: str,
                         line_list: list[tuple[str, bytes]]
                         ) -> dict[str, collections.Counter]:
    """Return the separator lines an unsettled union holds over what it may hold.

    What it may hold is what it holds with nothing left of a conflict block.

    Args:
        project_dir: The repository.
        commit: The tip of the unsettled union.
        line_list: _marker_line_dict()'s lines for that tip.

    Returns:
        {path: {separator line: how many of it the file holds over
        _record_separator_dict()'s count}}, for every file that mapping names, one
        holding none over it too.

    Raises:
        RuntimeError: When the trailers of the union's commit could not be read.
    """
    count_by_path_dict = _separator_count_dict(line_list)
    return {path: count_by_path_dict.get(path, collections.Counter()) - allow_dict
            for path, allow_dict in _record_separator_dict(project_dir, commit,
                                                           line_list).items()}


def _move_dict(project_dir: str, commit: str, union_commit: str) -> dict[str, str]:
    """Return the files of one commit git reads as moved in another.

    Args:
        project_dir: The repository.
        commit: The commit the files are read from.
        union_commit: The commit they are looked for in.

    Returns:
        {the path in commit: the path in union_commit} (gitcmd.move_dict). Empty when
        the two could not be compared.
    """
    return gitcmd.move_dict(project_dir, commit, union_commit) or {}


def separator_record_dict(merge_dict: dict) -> dict[str, dict[str, int]]:
    """Return the separator lines a union may hold, from assemble()'s "marker".

    Args:
        merge_dict: assemble()'s return value.

    Returns:
        {path: {separator line: how many of it the file may hold, 0 too}} over the
        "separator" of assemble()'s "marker", every byte of a line written as the
        character of the same number.
    """
    separator_by_path_dict = (merge_dict.get("marker") or {}).get("separator") or {}
    return {path: {line.decode("latin-1"): count
                   for line, count in sorted(count_dict.items()) if count >= 0}
            for path, count_dict in separator_by_path_dict.items()}


def is_separator_record(value: object) -> bool:
    """Say whether a value is of the shape separator_record_dict() returns.

    Args:
        value: What a run recorded, or what a trailer held.

    Returns:
        True for {path: {line: count}} whose every path is a string of 1 to
        _MAX_SEPARATOR_PATH_CHAR characters holding no NUL, whose every line is a string
        of characters under 256 that starts with "=" or "|", and whose every count is an
        int from 0 to _MAX_SEPARATOR_COUNT.
    """
    if not isinstance(value, dict):
        return False
    for path, count_dict in value.items():
        if not isinstance(path, str) or not 0 < len(path) <= _MAX_SEPARATOR_PATH_CHAR \
                or "\0" in path or not isinstance(count_dict, dict):
            return False
        for line, count in count_dict.items():
            if not isinstance(line, str) or line[:1] not in ("=", "|") \
                    or any(ord(char) > 255 for char in line):
                return False
            if type(count) is not int or not 0 <= count <= _MAX_SEPARATOR_COUNT:
                return False
    return True


def separator_text(record_dict: dict | None) -> str:
    """Render the separator lines a union may hold as the one line a trailer holds.

    Every character outside ASCII and every whitespace character is written as its JSON
    escape; the line holds no whitespace.

    Examples:
        {"doc.rst": {"=======": 1}, "f.py": {}} -> '{"doc.rst":{"=======":1},"f.py":{}}'
        {}                                       -> "{}"
        None                                     -> ""

    Args:
        record_dict: separator_record_dict()'s mapping, or None for a run that
            records none.

    Returns:
        The mapping as JSON on one line, or "" for None and for a line that would be
        over _MAX_SEPARATOR_CHAR characters, which is logged.
    """
    if record_dict is None:
        return ""
    text = _one_line_json(record_dict)
    if len(text) > _MAX_SEPARATOR_CHAR:
        logger.warning("the separator lines of %d file(s) take %d characters, over the "
                       "%d a commit records, so none is recorded", len(record_dict),
                       len(text), _MAX_SEPARATOR_CHAR)
        return ""
    return text


def _separator_record_at(project_dir: str, commit: str
                         ) -> dict[str, collections.Counter] | None:
    """Return the separator lines the union at one commit records it may hold.

    It is separator_text()'s inverse over the commit's trailer.

    Args:
        project_dir: The repository to read.
        commit: The commit to read, and no commit it reaches.

    Returns:
        {path: {separator line as bytes: how many of it the file may hold}}, or None
        when the commit does not carry SEPARATOR_TRAILER once, or carries a value that
        is over _MAX_SEPARATOR_CHAR characters, is not JSON or is of another shape
        (is_separator_record), which is logged.

    Raises:
        RuntimeError: When the commit's trailers could not be read.
    """
    head = f"{SEPARATOR_TRAILER}:"
    text_list = [line[len(head):].strip() for line in own_trailer_list(project_dir, commit)
                 if line.startswith(head)]
    if not text_list:
        return None
    value = None
    if len(text_list) == 1 and len(text_list[0]) <= _MAX_SEPARATOR_CHAR:
        try:
            value = json.loads(text_list[0])
        except (ValueError, RecursionError):
            value = None
    if not is_separator_record(value):
        logger.warning("the separator lines %s records cannot be read: %s",
                       gitcmd.short(commit), cut(" ".join(text_list), 200))
        return None
    return {path: collections.Counter({line.encode("latin-1"): count
                                       for line, count in count_dict.items()})
            for path, count_dict in value.items()}


def _one_line_json(value: object) -> str:
    """Render a value as JSON a trailer holds on one line.

    Args:
        value: What json.dumps() takes.

    Returns:
        The JSON, every character outside ASCII and every whitespace character
        written as its JSON escape.
    """
    text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return re.sub(r"\s", lambda match: f"\\u{ord(match.group()):04x}", text)


def no_merge_record_list(merge_dict: dict) -> list[list]:
    """Return the sides whose change to a file is not in a union.

    Args:
        merge_dict: assemble()'s return value.

    Returns:
        [[the side's commit, the path the side holds the file at ("" for none,
        None where it cannot be told), the file's path in the union], ...] over the
        "in_block" of every entry of assemble()'s "branches".
    """
    return [[entry_dict["commit"], side_path, path]
            for entry_dict in merge_dict.get("branches") or []
            for path, side_path in sorted((entry_dict.get("in_block") or {}).items())]


def is_no_merge_record(value: object) -> bool:
    """Say whether a value is of the shape no_merge_record_list() returns.

    Args:
        value: What a run recorded, or what a trailer held.

    Returns:
        True for a list of [commit, side path, path] whose every commit is a string
        OBJECT_NAME matches, whose every side path is None or a string of at most
        _MAX_SEPARATOR_PATH_CHAR characters holding no NUL, and whose every path is such
        a string of at least one character.
    """
    if not isinstance(value, list):
        return False
    for entry in value:
        if not isinstance(entry, list) or len(entry) != 3:
            return False
        commit, side_path, path = entry
        if not isinstance(commit, str) or not OBJECT_NAME.fullmatch(commit):
            return False
        for text in (path, *([] if side_path is None else [side_path])):
            if not isinstance(text, str) or len(text) > _MAX_SEPARATOR_PATH_CHAR \
                    or "\0" in text:
                return False
        if not path:
            return False
    return True


def no_merge_text(record_list: list | None) -> str:
    """Render the sides whose change to a file is not in a union as one trailer line.

    Examples:
        [["1f2e...", "f.txt", "g.txt"], ["9a8b...", None, "g.txt"]]
            -> '[["1f2e...","f.txt","g.txt"],["9a8b...",null,"g.txt"]]'
        []   -> "[]"
        None -> ""

    Args:
        record_list: no_merge_record_list()'s list, or None for a run that records
            none.

    Returns:
        The list as JSON on one line (_one_line_json), or "" for None and for a line
        that would be over _MAX_SEPARATOR_CHAR characters, which is logged.
    """
    if record_list is None:
        return ""
    text = _one_line_json(record_list)
    if len(text) > _MAX_SEPARATOR_CHAR:
        logger.warning("the %d file(s) a side's change was not merged into take %d "
                       "characters, over the %d a commit records, so none is recorded",
                       len(record_list), len(text), _MAX_SEPARATOR_CHAR)
        return ""
    return text


def _no_merge_record_at(project_dir: str, commit: str
                        ) -> list[tuple[str, str | None, str]] | None:
    """Return the sides the union at one commit records as not merged into a file.

    It is no_merge_text()'s inverse over the commit's trailer.

    Args:
        project_dir: The repository to read.
        commit: The commit to read, and no commit it reaches.

    Returns:
        [(the side's commit, the path it holds the file at, the file's path in the
        union), ...], or None when the commit does not carry NO_MERGE_TRAILER, or
        carries a value that is over _MAX_SEPARATOR_CHAR characters, is not JSON or is
        of another shape (is_no_merge_record), which is logged.

    Raises:
        RuntimeError: When the commit's record could not be read.
    """
    text = trailer_value(own_trailer_list(project_dir, commit), NO_MERGE_TRAILER)
    if not text:
        return None
    value = None
    if len(text) <= _MAX_SEPARATOR_CHAR:
        try:
            value = json.loads(text)
        except (ValueError, RecursionError):
            value = None
    if not is_no_merge_record(value):
        logger.warning("the sides %s records as not merged into a file cannot be read: "
                       "%s", gitcmd.short(commit), cut(text, 200))
        return None
    return [(side_commit, side_path, path) for side_commit, side_path, path in value]


def _name_frame_dict(project_dir: str, start: str, side_commit_list: list[str]) -> dict:
    """Return how many of each name line a union may hold with no block left.

    A name line the start or a side holds itself is none git wrote here. The union
    may hold as many of one as the start holds and, for each side, as many more as
    that side holds over the start. The lines are counted over all the files of a
    commit, wherever a file is.

    Args:
        project_dir: The repository.
        start: The commit every branch starts from.
        side_commit_list: The commits merged.

    Returns:
        {"name_count": {name line: that count}, "name_at": {(path, name line): the
        most of that line the start or one side holds at that path}}, both a
        collections.Counter.

    Raises:
        RuntimeError: When a commit could not be read.
    """
    commit_list = list(dict.fromkeys([start, *side_commit_list]))
    line_dict = _marker_line_dict(project_dir, commit_list)
    name_at_dict: collections.Counter = collections.Counter()
    count_list = []
    for commit in commit_list:
        at_dict = collections.Counter(entry for entry in line_dict[commit]
                                      if _is_name_line(entry[1]))
        name_at_dict |= at_dict
        count_list.append(_name_count_dict(at_dict))
    name_count_dict = collections.Counter(count_list[0])
    for count_dict in count_list[1:]:
        name_count_dict += count_dict - count_list[0]
    return {"name_count": name_count_dict, "name_at": name_at_dict}


def _name_count_dict(count_dict: collections.Counter) -> collections.Counter:
    """Return how many of each name line a tree holds in all its files.

    Args:
        count_dict: {(path, line): how many of that line the file holds}, over
            _marker_line_dict()'s lines.

    Returns:
        {name line: the count over every path}, the separator lines left out.
    """
    name_count_dict: collections.Counter = collections.Counter()
    for (_, line), count in count_dict.items():
        if _is_name_line(line):
            name_count_dict[line] += count
    return name_count_dict


def _marker_path_set(count_dict: collections.Counter, marker_dict: dict) -> set[str]:
    """Return the paths of a tree that hold a line left of a conflict block.

    A name line is left of a block when the tree holds more of it in all its files
    than "name_count" allows. It is named at the paths holding more of it than
    "name_at" holds there, and at every path holding it when there is none. A
    separator line is left of a block when a file "separator" has an entry for
    holds more of it than that entry.

    Args:
        count_dict: {(path, line): how many of that line the file holds}, over
            _marker_line_dict()'s lines for the tree.
        marker_dict: assemble()'s "marker". A key it does not hold allows nothing.

    Returns:
        The paths.
    """
    allow_dict = marker_dict.get("name_count") or {}
    name_at_dict = marker_dict.get("name_at") or {}
    separator_by_path_dict = marker_dict.get("separator") or {}
    over_line_set = {line for line, count in _name_count_dict(count_dict).items()
                     if count > allow_dict.get(line, 0)}
    path_set: set[str] = set()
    for over_line in over_line_set:
        hold_set = {path for path, line in count_dict if line == over_line}
        own_set = {path for path in hold_set
                   if count_dict[(path, over_line)] > name_at_dict.get((path, over_line), 0)}
        path_set |= own_set or hold_set
    for (path, line), count in count_dict.items():
        if not _is_name_line(line) and path in separator_by_path_dict \
                and count > separator_by_path_dict[path].get(line, 0):
            path_set.add(path)
    return path_set


def _stand_in_path_list(project_dir: str, commit: str, side_commit_list: list[str]
                        ) -> list[str]:
    """Return the files of a union that are under a name git made up in a conflict.

    Args:
        project_dir: The repository.
        commit: The union commit.
        side_commit_list: The start and the commits merged. A path one of them holds
            is a name of its own and is left out.

    Returns:
        The paths matching _STAND_IN_NAME that no commit of side_commit_list holds, as
        _tree_file_list() gives them.
    """
    path_list = [path for path in _tree_file_list(project_dir, commit)
                 if _STAND_IN_NAME.search(path)]
    return [path for path in path_list
            if not any(_has_path(project_dir, side_commit, path)
                       for side_commit in side_commit_list)]


def _has_path(project_dir: str, commit: str, path: str) -> bool:
    """Say whether a commit holds anything at a path.

    Anything is a file, a link, a directory or a submodule.

    Args:
        project_dir: The repository.
        commit: The commit to read.
        path: A repository-relative path.

    Returns:
        True when the commit's tree holds an entry there.
    """
    return gitcmd.run(project_dir, "cat-file", "-e", f"{commit}:{path}").returncode == 0


def _stand_in_now_list(project_dir: str, stand_in_list: list[str]) -> list[str]:
    """Return the stand-in names the working tree still holds something under.

    Args:
        project_dir: The repository.
        stand_in_list: assemble()'s "stand_in".

    Returns:
        Those of the paths the working tree holds, in the order given.
    """
    return [path for path in stand_in_list
            if os.path.lexists(os.path.join(project_dir, path))]


def _move_head(project_dir: str, commit: str) -> None:
    """Check a commit out, leaving the working tree to be written over afterwards.

    Args:
        project_dir: The repository to write in.
        commit: The commit HEAD is moved to.

    Raises:
        RuntimeError: When HEAD could not be moved.
    """
    proc = _run_tree_write(project_dir, gitcmd.tree_diff_path_list(project_dir, commit),
                           ("checkout", "--quiet", "--detach", commit))
    if proc.returncode != 0:
        raise RuntimeError(f"could not start from {gitcmd.short(commit)}: "
                           f"{(proc.stderr or proc.stdout).strip()}")


def place(project_dir: str, commit: str, env_dict: dict[str, str] | None = None) -> None:
    """Put a commit's tree in the working tree and the index, leaving HEAD where it is.

    A file whose contents the tree replaces is given a modification time and size the
    path has not had (tmpfile.move_mtime_on).

    Args:
        project_dir: The repository to write in.
        commit: The commit, or the tree, that is put in place.
        env_dict: As gitcmd.run() takes it. The settings gitcmd.work_tree_index()
            yields put the tree in that index file, which holds the working tree as it
            is, and leave the repository's own index as it is. None writes the
            repository's own.

    Raises:
        RuntimeError: When the working tree could not be written.
    """
    stamp_dict = {path: tmpfile.file_stamp(os.path.join(project_dir, path))
                  for path in gitcmd.tree_diff_path_list(project_dir, commit, env_dict)}
    proc = _run_tree_write(project_dir, list(stamp_dict),
                           ("read-tree", "-u", "--reset", commit), env_dict)
    if proc.returncode != 0:
        raise RuntimeError(f"could not put the merged tree in the working tree: "
                           f"{(proc.stderr or proc.stdout).strip()}")
    for path, stamp in stamp_dict.items():
        tmpfile.move_mtime_on(os.path.join(project_dir, path), stamp)


def _run_tree_write(project_dir: str, path_list: list[str], word_tuple: tuple[str, ...],
                    env_dict: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a git command that writes a tree into the working tree and the index.

    When git does not end with 0, or the run is stopped while it writes, the files it
    wrote at the paths of path_list that held nothing before it started and that the
    index does not hold are taken away (_remove_new_file).

    Args:
        project_dir: The repository to write in.
        path_list: The repository-relative paths the command may write, as
            gitcmd.tree_diff_path_list() gives them.
        word_tuple: The git arguments, without "git" itself.
        env_dict: As gitcmd.run() takes it, naming the index the command writes.

    Returns:
        gitcmd.run()'s finished process.

    Raises:
        KeyboardInterrupt: When the run is stopped while git writes, once those files
            are taken away.
    """
    absent_list = [path for path in path_list
                   if not os.path.lexists(os.path.join(project_dir, path))]
    try:
        proc = gitcmd.run(project_dir, *word_tuple, env_dict=env_dict)
    except BaseException:
        _remove_new_file(project_dir, absent_list, env_dict)
        raise
    if proc.returncode != 0:
        _remove_new_file(project_dir, absent_list, env_dict)
    return proc


def _remove_new_file(project_dir: str, path_list: list[str],
                     env_dict: dict[str, str] | None = None) -> None:
    """Take away the new files a git that wrote the working tree left behind.

    They are the files at paths that held nothing before it started and that the index
    does not hold. The directories this leaves empty are taken away as well. A path the
    index holds, one with nothing at it and a directory are left as they are, and a file
    that cannot be taken away is logged.

    Args:
        project_dir: The repository.
        path_list: Repository-relative paths, as os.fsdecode() gives them, that held
            nothing before git started.
        env_dict: As gitcmd.run() takes it, naming the index git wrote.
    """
    if not path_list:
        return
    proc = gitcmd.run(project_dir, "ls-files", "-z", env_dict=env_dict, text=False)
    if proc.returncode != 0:
        logger.warning("the files git left in %s are not taken away: the index could not "
                       "be read (%s)", project_dir, gitcmd.shown_text(proc.stderr).strip())
        return
    hold_set = {os.fsdecode(raw_path) for raw_path in proc.stdout.split(b"\0") if raw_path}
    for rel_path in path_list:
        path = os.path.join(project_dir, rel_path)
        if rel_path in hold_set or not os.path.lexists(path) \
                or (os.path.isdir(path) and not os.path.islink(path)):
            continue
        try:
            os.remove(path)
        except OSError as e:
            logger.warning("%s, which git left, could not be taken away (%s)", path, e)
            continue
        parent = os.path.dirname(rel_path)
        while parent:
            try:
                os.rmdir(os.path.join(project_dir, parent))
            except OSError:
                break
            parent = os.path.dirname(parent)


def marker_path_list(project_dir: str, marker_dict: dict | None = None) -> list[str]:
    """Return the paths the working tree holds a line left of a conflict block in.

    The lines are those _marker_line_dict() reads and _marker_path_set() tells: the
    name lines git writes at the start and the end of a block in a merge made here,
    and the separator lines of the files git wrote a block in, a separator line as
    long as one "separator" names for its file among them. The working tree is
    read as the tree git would make of it (gitcmd.work_tree_index): the tracked files
    as they are on disk and the files git does not track yet and no ignore rule
    covers, each in the form git holds it in.

    Args:
        project_dir: The repository to read.
        marker_dict: assemble()'s "marker": the lines the tree may hold. None
            allows no name line and reads no file for separator lines.

    Returns:
        The paths, sorted and without repeats.

    Raises:
        RuntimeError: When the working tree could not be read as a tree.
    """
    marker_dict = marker_dict or {}
    with gitcmd.work_tree_index(project_dir, keep_object=False) as (tree, env_dict):
        if not tree:
            raise RuntimeError(f"the tree of {project_dir} could not be written, so its "
                               f"conflict marker lines cannot be read")
        line_list = _marker_line_dict(
            project_dir, [tree], env_dict, marker_dict.get("size"),
            {path: {_run_size(line) for line in count_dict} for path, count_dict
             in (marker_dict.get("separator") or {}).items()})[tree]
    return sorted(_marker_path_set(collections.Counter(line_list), marker_dict))


def qual_name_dict(def_dict: dict) -> dict[str, str]:
    """Return each definition's name with the names of the definitions around it.

    A definition is around another when it is in the same file and its line range holds
    the other's and is longer. The name of the narrowest one around it, named the same
    way, goes in front with "." between; of two as narrow, the one written first.

    Examples:
        class A: def run (lines 2-3 of 1-3), class B: def run (lines 7-8 of 6-8)
            -> {"...::A": "A", "...::run": "A.run", "...::B": "B",
                "...::run#function_definition#7": "B.run"}

    Args:
        def_dict: build_def_node_dict()'s return value.

    Returns:
        {definition key: the name}, each part of it in the form name_key() returns.
    """
    qual_dict: dict[str, str] = {}
    for file, entry_list in def_by_file_dict(def_dict).items():
        # The definitions whose line range is still open, outermost first
        open_list: list[tuple[int, int, str]] = []
        for start, end, key in sorted(entry_list, key=lambda entry: (entry[0], -entry[1])):
            open_list = [entry for entry in open_list if entry[1] >= end]
            name = name_key(file, def_dict[key]["name"])
            # The narrowest of the longer ones, the first written of those as narrow
            outer_list = sorted((outer_end - outer_start, index, outer_key)
                                for index, (outer_start, outer_end, outer_key)
                                in enumerate(open_list)
                                if outer_end - outer_start > end - start)
            qual_dict[key] = f"{qual_dict[outer_list[0][2]]}.{name}" if outer_list else name
            open_list.append((start, end, key))
    return qual_dict


def same_name(name: str, other: str) -> bool:
    """Say whether two names written with "." between their parts name one definition.

    Examples:
        ("B.run", "B.run")              -> True
        ("B.run", "run")                -> True
        ("helpers.find", "find")        -> True
        ("B.run", "A.run")              -> False

    Args:
        name: One name.
        other: The other name.

    Returns:
        True when they are the same, or one is the other with parts in front of it.
    """
    return name == other or name.endswith(f".{other}") or other.endswith(f".{name}")


def analyse(knowledge_path: str) -> dict:
    """Read one state of the repository out of its analysis.

    A definition is named as qual_name_dict() names it: with the names of the
    definitions around it in front. Two definitions of one name in one file (two
    classes' methods) are two names.

    Args:
        knowledge_path: The path to the knowledge database of that state.

    Returns:
        {"no_resolve": the references the analysis cannot resolve, as
        no_resolve_ref_dict() returns them, "name": {knowledge path: the names that
        file defines, sorted}, "range": {knowledge path: [(start line, end line,
        name), ...]}, "ref": [(referring file, referenced file, the name the referenced
        file defines, [line, ...]), ...] for every reference that resolves, once for
        each definition it can be read as, "use": [(referring file, referenced file),
        ...] for every use of another file, whether or not it resolves to a
        definition (a module used as a value does not), sorted, "signature":
        {knowledge path: {name: the definition's first line}}, "own": {knowledge path:
        {name: the definition's own name, without the names of the definitions around
        it or of those it belongs to (own_name())}}, "define": {own name: [the
        knowledge path of each definition of that name, ...], sorted, with one entry
        for every definition: the length of a list is how many definitions of that name
        the state holds}}. "own" and "define" leave out a definition that gives a
        variable, a constant or a field a value holding no function (_value_key_set()).
        Every path is a knowledge path. Every name of a file whose language ignores case
        is in the form name_key() returns: the same name written in another case in
        another state is the same name.
    """
    knowledge = Knowledge.open(knowledge_path)
    try:
        def_dict = build_def_node_dict(knowledge)
        by_file_name_dict = key_by_file_name_dict(def_dict)
        target_dict = target_key_dict(by_file_name_dict, def_by_file_dict(def_dict))
        qual_dict = qual_name_dict(def_dict)
        value_key_set = _value_key_set(knowledge)
        name_dict: dict = {}
        range_dict: dict = {}
        signature_dict: dict = {}
        own_dict: dict = {}
        define_dict: dict = {}
        for key, node in def_dict.items():
            name = qual_dict[key]
            name_dict.setdefault(node["file"], set()).add(name)
            range_dict.setdefault(node["file"], []).append(
                (node["lines"][0], node["lines"][1], name))
            signature_dict.setdefault(node["file"], {})[name] = node["signature"]
            if (node["file"], node["name"], node["lines"][0]) in value_key_set:
                continue
            own = name_key(node["file"], own_name(node["file"], node["name"]))
            own_dict.setdefault(node["file"], {})[name] = own
            define_dict.setdefault(own, []).append(node["file"])
        ref_list = []
        use_set = set()
        for file_dict in knowledge.iter_file():
            for usage_dict in file_dict["file_dependencies"]["callee_usages"]:
                use_set.add((file_dict["file"], usage_dict["from"]))
                key_list = resolve_usage(target_dict, usage_dict)
                # The definitions the name written can be read as, all of them when
                # it does not tell them apart
                qual_list = sorted({qual_dict[key] for key in key_list})
                use_name = name_key(usage_dict["from"], usage_dict["name"])
                ref_list += [(file_dict["file"], usage_dict["from"], name,
                              sorted(usage_dict["lines"] or ()))
                             for name in [name for name in qual_list if same_name(name, use_name)]
                             or qual_list]
        no_resolve_dict: dict = {}
        for (from_file, to_file, name), line_list in no_resolve_ref_dict(
                knowledge, def_dict).items():
            no_resolve_dict.setdefault((from_file, to_file, name_key(to_file, name)),
                                       set()).update(line_list)
        return {"no_resolve": {key: sorted(line_set)
                               for key, line_set in no_resolve_dict.items()},
                "name": {file: sorted(name_set) for file, name_set in name_dict.items()},
                "range": range_dict, "ref": ref_list, "use": sorted(use_set),
                "signature": signature_dict, "own": own_dict,
                "define": {name: sorted(file_list)
                           for name, file_list in define_dict.items()}}
    finally:
        knowledge.close()


def _value_key_set(knowledge: Knowledge) -> set[tuple[str, str, int]]:
    """Return the definitions of an analysis that give a plain value to a name.

    The name is a variable, a constant or a field, and the value holds no function and
    no class. Each definition is read by its type and by its own text
    (languages.is_value_type).

    Args:
        knowledge: The analysis, after build_def_node_dict() has read it.

    Returns:
        {(knowledge path, the definition's name, its first line), ...}.
    """
    return {(file_dict["file"], def_entry["name"], def_entry["start_line"])
            for file_dict in knowledge.iter_file()
            for def_entry in file_dict["file_dependencies"]["definitions"]
            if is_value_type(file_dict["file"], def_entry["type"], def_entry["context"])}


def broken_ref_dict(before_dict: dict, after_dict: dict) -> dict:
    """Return the references that resolve in one state and no longer resolve in another.

    What the analysis could not resolve in before is in both and is not returned for
    that alone. A key is counted by its lines: a key that holds more lines in after than
    in before holds a reference that no longer resolves, though which of its lines that
    is cannot be told once the lines have moved.

    Args:
        before_dict: analyse()'s "no_resolve" for the commit the union is written out
            against.
        after_dict: analyse()'s "no_resolve" for the merged state.

    Returns:
        The entries of after that hold more lines than the same key holds in before,
        with all of after's lines.
    """
    return {key: line_list for key, line_list in after_dict.items()
            if _line_count(after_dict, key) > _line_count(before_dict, key)}


def add_no_resolve(into_dict: dict, from_dict: dict) -> None:
    """Put one state's unresolved references into those of others, by the most lines.

    Args:
        into_dict: analyse()'s "no_resolve" of the states read so far, written to.
        from_dict: analyse()'s "no_resolve" of one more state.
    """
    for key in from_dict:
        if _line_count(from_dict, key) > _line_count(into_dict, key):
            into_dict[key] = from_dict[key]


def _line_count(no_resolve_dict: dict, key: tuple) -> int:
    """Return how many references of one key a state leaves unresolved.

    Args:
        no_resolve_dict: analyse()'s "no_resolve".
        key: (referring file, referenced file, name).

    Returns:
        The number of its lines, 1 for a key held with no line, 0 for one not held.
    """
    if key not in no_resolve_dict:
        return 0
    return max(len(no_resolve_dict[key]), 1)


# == The definitions more than one side's change reaches ==

# One hunk header of a unified diff: "@@ -<line>[,<count>] +<line>[,<count>] @@"
_HUNK_HEAD = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")

# The path lines of one file's diff, which come after its "diff --git" line and before
# its first hunk. The paths start with git's own a/ and b/
_DIFF_FILE_HEAD = "diff --git "
_DIFF_PATH_PREFIX_LEN = len("b/")
_NO_FILE = "/dev/null"


def change_dict_of(project_dir: str, base: str, commit: str) -> dict:
    """Read which lines of the base a commit changed, file by file.

    Args:
        project_dir: The repository to read.
        base: The commit the branch started from.
        commit: The branch's tip. "" reads the working tree against base instead, past
            the files git cannot read there (gitcmd.work_tree_diff()), which are left
            out.

    Returns:
        {"add": [repository-relative path, ...] the commit added, "delete": [...] it
        removed, "line": {repository-relative path: [(first line, last line), ...]}}
        over the files it changed, the lines being the base's and the paths as
        gitcmd.shown_text() gives them. A hunk that only inserts lines names the line it
        follows and the one after it.

    Raises:
        RuntimeError: When the diff could not be read.
    """
    diff_tuple = ("-c", "core.quotePath=false", *gitcmd.PLAIN_DIFF_TUPLE, "--unified=0",
                  base)
    if commit:
        proc, unread_list = gitcmd.run(project_dir, *diff_tuple, commit, text=False), []
    else:
        proc, unread_list = gitcmd.work_tree_diff(project_dir, base, *diff_tuple)
    if proc.returncode != 0:
        what = gitcmd.short(commit) or "the working tree"
        raise RuntimeError(f"could not read what {what} changed: "
                           f"{gitcmd.shown_text(proc.stderr or proc.stdout).strip()}")
    unread_set = {gitcmd.shown_text(path) for path in unread_list}
    change_dict: dict = {"add": [], "delete": [], "line": {}}
    path, span_list, in_head = "", None, False
    for line in line_list_of(os.fsdecode(proc.stdout)):
        if line.startswith(_DIFF_FILE_HEAD):
            in_head, span_list = True, None
        elif in_head and line.startswith("--- "):
            head_path = gitcmd.unquote_path(line[4:])
            path = "" if head_path == _NO_FILE else head_path[_DIFF_PATH_PREFIX_LEN:]
        elif in_head and line.startswith("+++ "):
            # The old path is the removed file's, the new one every other file's
            head_path = gitcmd.unquote_path(line[4:])
            in_head = False
            if head_path == _NO_FILE:
                change_dict["delete"].append(path)
            elif not path:
                change_dict["add"].append(head_path[_DIFF_PATH_PREFIX_LEN:])
            else:
                span_list = change_dict["line"].setdefault(
                    head_path[_DIFF_PATH_PREFIX_LEN:], [])
        elif span_list is not None:
            head = _HUNK_HEAD.match(line)
            if head:
                first = int(head.group(1))
                count = int(head.group(2)) if head.group(2) is not None else 1
                span_list.append((first, first + count - 1) if count else (first, first + 1))
    if unread_set:
        change_dict = {
            "add": [path for path in change_dict["add"] if path not in unread_set],
            "delete": [path for path in change_dict["delete"]
                       if path not in unread_set],
            "line": {path: span_list for path, span_list in change_dict["line"].items()
                     if path not in unread_set}}
    return change_dict


def side_dict_of(change_dict: dict, range_dict: dict, project_name: str) -> dict:
    """Map what one side changed onto the base's definitions.

    Args:
        change_dict: change_dict_of()'s return value for that side.
        range_dict: analyse()'s "range" for the base.
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        {knowledge path: {definition name, ...}}, the names being the narrowest base
        definitions the changed lines fall in. "" stands for a line outside every
        definition and for a file the side added; a file the side removed holds every
        name it defined and "".
    """
    side_dict: dict = {}
    for path in change_dict["add"]:
        side_dict[repo_to_knowledge(path, project_name)] = {""}
    for path in change_dict["delete"]:
        file = repo_to_knowledge(path, project_name)
        side_dict[file] = {""} | {name for _, _, name in range_dict.get(file, ())}
    for path, span_list in change_dict["line"].items():
        file = repo_to_knowledge(path, project_name)
        name_set = side_dict.setdefault(file, set())
        for first, last in span_list:
            for line in range(first, last + 1):
                name_set.add(outer_def(range_dict, file, line) or "")
    return side_dict


def _change_name_set(side_dict: dict, own_dict: dict) -> set[str]:
    """Return the own names of the definitions one side's change falls in.

    Args:
        side_dict: side_dict_of()'s mapping for the side.
        own_dict: analyse()'s "own" for the commit the side starts from.

    Returns:
        The names, without the "" that stands for a line outside every definition.
    """
    return {own_dict[file][name] for file, name_set in side_dict.items()
            for name in name_set if name in own_dict.get(file, {})}


def keep_list_of(side_list: list[tuple[str, dict]], start_dict: dict,
                 tip_define_by_branch_dict: dict,
                 start_by_branch_dict: dict | None = None,
                 start_side_list: list[tuple[dict, dict]] | None = None,
                 fork_define_by_pair_dict: dict | None = None
                 ) -> list[tuple[str, dict[str, list[str]]]]:
    """Return the names each side added or changed that the union has to define.

    A definition is told by its own name alone, wherever its file is. A name a side's
    tip defines is one it added when the commit the side starts from defines it
    nowhere, and one it changed when the side's changed lines fall in a definition of
    that name there. Either is left out when another side's tip holds fewer
    definitions of it than the commit the two sides both reach last
    (fork_define_by_pair_dict; the commit that other side starts from where the
    pair has no entry): that side removed or renamed one.

    A side that starts from a commit of its own is read against the names of that
    commit, and the start the other sides share is one more side to it, read from
    that commit: a name of the side is left out when the start holds fewer
    definitions of it than that commit, and the names the start added or changed
    since that commit (start_side_list) are looked for as a side's are. A name of
    the start is left out when a side's tip holds fewer definitions of it than
    the commit that side starts from. A side with no analysis of its tip holds no
    name to look for and takes none away, which is logged.

    Examples:
        a changed foo in lib.py, b's tip defines no foo
            -> [("a", {}), ("b", {...})]
        a moved lib.py to core.py, b removed foo and left keep
            -> [("a", {"keep": ["p/core_py/core.py"]}), ("b", {})]
        a added new_fn to lib.py, whatever b's tip holds
            -> [("a", {"new_fn": ["p/lib_py/lib.py"]}), ("b", {...})]

    Args:
        side_list: [(branch, side_dict_of()'s mapping), ...] for the sides.
        start_dict: analyse()'s return value for the start, or a dict holding its
            "own" and its "define".
        tip_define_by_branch_dict: {branch: analyse()'s "define" at that branch's tip}.
        start_by_branch_dict: {branch: {"own", "define"} for the commit that side
            starts from}, for the sides that do not start from the start. None holds
            none.
        start_side_list: [(side_dict_of()'s mapping for what the start changed since
            a commit, that commit's {"own", "define"}), ...], one for every commit of
            start_by_branch_dict. None holds none.
        fork_define_by_pair_dict: {(branch, another branch): analyse()'s "define" at
            the commit both reach last}, as _fork_define_dict() returns it. None
            holds none.

    Returns:
        [(branch, {own name: [the knowledge path of each file the side's tip defines
        it in, sorted and without repeats]}), ...] in the order given, then one entry
        for the start under _START_BRANCH_TEXT when start_side_list holds any.
    """
    start_by_branch_dict = start_by_branch_dict or {}
    fork_define_by_pair_dict = fork_define_by_pair_dict or {}
    no_tip_list = [branch for branch, _ in side_list if branch not in tip_define_by_branch_dict]
    if no_tip_list:
        logger.info("%s have no analysis of their own tip, so what they added or changed "
                    "is not looked for in the union", list_text(no_tip_list))
    # What each side's tip defines, with what the commit it starts from does
    pair_by_branch_dict = {
        branch: (tip_define_dict, start_by_branch_dict.get(branch, start_dict)["define"])
        for branch, tip_define_dict in tip_define_by_branch_dict.items()}
    keep_list = []
    for branch, side_dict in side_list:
        own_start_dict = start_by_branch_dict.get(branch, start_dict)
        other_list = [(tip_define_dict,
                       fork_define_by_pair_dict.get((branch, other), from_define_dict))
                      for other, (tip_define_dict, from_define_dict)
                      in pair_by_branch_dict.items() if other != branch]
        if branch in start_by_branch_dict:
            other_list.append((start_dict["define"], own_start_dict["define"]))
        keep_list.append((branch, _keep_dict(
            tip_define_by_branch_dict.get(branch, {}), own_start_dict["define"],
            _change_name_set(side_dict, own_start_dict["own"]), other_list)))
    if start_side_list:
        start_keep_dict: dict[str, list[str]] = {}
        for side_dict, own_start_dict in start_side_list:
            start_keep_dict.update(_keep_dict(
                start_dict["define"], own_start_dict["define"],
                _change_name_set(side_dict, own_start_dict["own"]),
                list(pair_by_branch_dict.values())))
        keep_list.append((_START_BRANCH_TEXT, start_keep_dict))
    return keep_list


def _keep_dict(tip_define_dict: dict, from_define_dict: dict, change_set: set[str],
               other_list: list[tuple[dict, dict]]) -> dict[str, list[str]]:
    """Return the names one side added or changed that the union has to define.

    Args:
        tip_define_dict: analyse()'s "define" at the side's tip.
        from_define_dict: analyse()'s "define" at the commit the side starts from.
        change_set: _change_name_set()'s names for the side.
        other_list: [(analyse()'s "define" at another side's tip, that at the commit
            the two sides both reach last), ...]. A name the other side's tip holds
            fewer definitions of than that commit is left out.

    Returns:
        {own name: [the knowledge path of each file the tip defines it in, sorted and
        without repeats]}.
    """
    keep_dict: dict[str, list[str]] = {}
    for name, file_list in tip_define_dict.items():
        if name in from_define_dict and name not in change_set:
            continue
        if any(len(other_tip_dict.get(name, ())) < len(other_fork_dict.get(name, ()))
               for other_tip_dict, other_fork_dict in other_list):
            continue
        keep_dict[name] = sorted(set(file_list))
    return keep_dict


def overlap_entry_list(before_dict: dict, after_dict: dict, side_list: list[tuple[str, dict]],
                       project_name: str) -> list[dict]:
    """Return the definitions the merge took more than one side's change to.

    Args:
        before_dict: analyse()'s return value for the base.
        after_dict: analyse()'s return value for the merged tree.
        side_list: [(branch, side_dict_of()'s return value for it), ...].
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        [{"kind": "both", "file", "name", "branches": [branch, ...]}, ...] for every
        definition more than one side changed, then [{"kind": "use", "file", "name",
        "changed_by": branch, "used_in": file, "lines": [line, ...], "used_by":
        [branch, ...]}, ...] for every definition one side changed whose reference
        lies in what another side changed in the merged tree. Every path is relative
        to the repository root and every list is sorted.
    """
    by_def_dict: dict = {}
    for branch, side_dict in side_list:
        for file, name_set in side_dict.items():
            for name in name_set - {""}:
                by_def_dict.setdefault((file, name), []).append(branch)
    both_list = [{"kind": "both", "file": knowledge_to_repo(file, project_name), "name": name,
                  "branches": sorted(branch_list)}
                 for (file, name), branch_list in sorted(by_def_dict.items())
                 if len(branch_list) > 1]

    use_dict: dict = {}
    for from_file, to_file, name, line_list in after_dict["ref"]:
        change_list = by_def_dict.get((to_file, name), [])
        if not change_list:
            continue
        # The base's definitions each referring line lies in. A line in a definition
        # the merge added lies at the top level of the file, as an inserted line does
        base_name_set = set(before_dict["name"].get(from_file, ()))
        range_list = after_dict["range"].get(from_file, ())
        for line in line_list:
            key_set = {def_name for start, end, def_name in range_list
                       if start <= line <= end} & base_name_set or {""}
            for branch, side_dict in side_list:
                if branch in change_list or not key_set & side_dict.get(from_file, set()):
                    continue
                for changer in change_list:
                    entry_dict = use_dict.setdefault((to_file, name, changer, from_file),
                                                     {"lines": set(), "used_by": set()})
                    entry_dict["lines"].add(line)
                    entry_dict["used_by"].add(branch)
    use_list = [{"kind": "use", "file": knowledge_to_repo(to_file, project_name), "name": name,
                 "changed_by": changer, "used_in": knowledge_to_repo(from_file, project_name),
                 "lines": sorted(entry_dict["lines"]), "used_by": sorted(entry_dict["used_by"])}
                for (to_file, name, changer, from_file), entry_dict in sorted(use_dict.items())]
    return both_list + use_list


# == The work list and the stage the model runs ==========

INTEGRATE_INSTRUCTION_TEMPLATE = """You are combining changes several people made to
the same {lang_name} repository. They all started from the same commit and none of them
saw what the others did. Their changes have already been merged for you; what is left is
what the merge could not settle.

Each turn you write ONE ```python block. It runs, you see what it printed, then you write
the next block. print() what you want to see{output_limit}. Names you define stay \
available in later turns. Stop at the closing
fence — anything you write after it is thrown away and costs you the turn.

Your earlier turns stay in the prompt. After the request comes an "## At the start"
block: what the machine measured before the first turn, which does not change. Before
every turn you are also given a "## State" block that the machine rebuilds from the
repository. Write down what you settle as you settle it: record_decision(what, why) puts
it in the state, where it is read again after this run.

Your one job is to make the combined text say what every side meant. That is all:

- **Keep every side's intent.** A change you drop is somebody's work thrown away.
- **Do not make the changes again**, and do not add anything nobody asked for.
- **Touch only what the work list points at**: the files it names, and what a test it
  names runs. Everything else is already merged.

Where two sides changed the same lines, the file holds a conflict block:

```
<<<<<<< refs/codetwine-loom/merged-so-far
what the sides merged before this one wrote
||||||| the starting commit
what the line was before either of them touched it
=======
what the branch named after the last marker wrote
>>>>>>> refs/codetwine-loom/side/<branch>
```

Replace the whole block, every marker line included, with the one text that says
what both sides meant. Use replace_line_range when the block is long, replace_in_file when a
short exact string is easier. Read the file around the block first: a block settled
without reading what encloses it is usually settled wrong. work["branches"] says what
each side was asked to do and what it said it did — read that before you decide which
text to keep, and keep the one that leaves both requests satisfied rather than whichever
came first.

Where "## At the start" says the merged tree is not known to hold a side's change to a
file, the merge left that change out (the file held a conflict block of other sides, or
what an earlier run left of one), git put the side's file at a path of its own choosing
or kept another side's version of it, or an earlier run on these branches did not keep
the change. Read the file as that side has it and as the starting commit has it with
read_branch_file, and make the one text say what that side meant as well.

Where git could not settle a file and wrote no conflict block in it (a file git does not
merge as text, or one a side deleted while another changed it), the file holds the
version git kept, and "## At the start" says so. Read each side's version with
read_branch_file and make the file say what both sides meant, or delete it with
delete_file when that is what they meant.

Where the merge left a call to a name that is no longer defined, one side renamed or
removed it while another kept using it. Read what the file defines now and point the
call at it.

Where "## At the start" names a definition more than one side changed, or one side
changed while another side's change uses it, git merged the text without a conflict
block, and nobody has checked that the combined code does what every side was asked.
Read both sides' changes there against work["branches"], make the combined code satisfy
each request, and record with record_decision what you checked and what you found.

Every path you pass to a tool is relative to the repository root, e.g. "{path_example}".
{lang_note}{test_note}
When the work list is settled, call COMPLETE("<what you did>") and nothing else. Write
it in the language the requests in work["branches"] are written in. It is checked
before it is accepted: conflict markers left in the repository, calls that still do not
resolve{test_check} {refusal_text}"""

# How the instructions end the sentence on the check, by INTEGRATE_MAX_REFUSAL
_REFUSAL_TEXT_DICT = {0: "are recorded, and the call is accepted as it is.",
                      1: "send it back once."}
_REFUSAL_TEXT_MANY = "send it back, up to {count} times."

# The paragraph that is left out when there is no test command to run
_INTEGRATE_TEST_NOTE = """
Where a test the state names fails on the merged tree and passed before it, two changes
that each worked alone do not work together. Read the test and the code both sides
changed, and make the combined code pass it. Where the test asserts behaviour one side
was asked to change, update that assertion to what that side was asked for, nothing
else in that file. Never weaken or delete a test to make it pass — every test file you
change is recorded in the commit. Tests that test only code a side was asked to delete
go with it: remove those tests, and delete the file when nothing else is left in it.
run_test() runs `{test_command}` over the whole repository. Run it once you have settled
the work list, and read what it says before you finish.
"""
_INTEGRATE_TEST_CHECK = " and tests that fail when they passed before"

# The variables the stage is handed, in the shape rlm.build_prompt() shows them
INTEGRATE_VAR_DOC = """- work (dict) — what the "## At the start" and "## State" blocks
  show, so you can read it in code. It has "base", "branches" (a list of {"branch", "conflicted",
  "in_block", "said", "did"}, "in_block" being the files the merged tree is not known
  to hold the change of that branch to, "said" being
  what the runs on that branch were asked to do and "did"
  what they said they did), "markers" (the paths whose text still holds a conflict block,
  measured again every turn), "unmarked" (the conflicted paths git wrote no conflict
  block in, keeping one side's version), "changed" (a list of {"path", "insertions", "deletions",
  "new", "defs"} for every file this stage has changed since the branches were combined,
  measured again every turn; "new" is True for a file that did not exist and "defs"
  names the definitions the changed lines fall in), "read" (a list of {"path", "turn"}
  for every file this stage has read with read_source_file, read_branch_file,
  read_definition or get_use_of_file, "turn" being the last turn it read it on), "broken"
  (a list of {"file", "lines", "name", "defined_in", "now_defines"}, as measured before
  the first turn), "unresolved" (how many references
  the analysis could not resolve over the merged tree in all, the broken ones included,
  as measured at the last check),
  "overlap" (a list of {"kind": "both", "file", "name", "branches"} for a definition
  more than one side changed, and {"kind": "use", "file", "name", "changed_by",
  "used_in", "lines", "used_by"} for a definition one side changed that another
  side's change uses, as measured before the first turn),
  "tests" (what the whole test suite said, the last time it was run, that it did not
  say before the branches were combined, and "" when it said nothing new; a
  run_test() call over the whole suite, yours or the machine's, is what moves it),
  "tests_at" (the turn of that run, 0 for before the first turn), "checks" (what the
  static check found in the merged repository before the first turn that it did not
  find at HEAD), "already_failing" (the tests that were failing before the branches were
  combined and still fail on the merged tree, which are not counted against you, and
  None when that could not be read), "left" and "checked_at" (what the
  machine's last whole-repository check found, and the turn it ran on), "decisions" (what
  you have recorded), "earlier" ({"branch", "ended", "why", "decisions"}: how the merging
  run this one carries on from ended and what it recorded, and None when this run does
  not carry on from one) and "complete" (False when a list was too long to show whole, so
  settling what is there and running the check again is what finds the rest). Writing to
  it changes nothing: the machine rebuilds it before every turn."""

# Shown over the stage's last turns
_DEADLINE_WHAT = ("Settle what is left of the work list and call COMPLETE(). A merge "
                  "left half done is worse than either side on its own.")

# The late-answer note and its closing line, as run_stage() takes them
_LATE_ANSWER_NOTE = """## Out of turns

You have no turns left, so nothing more can be changed. Say what you settled and what
you left, and NOTHING else - no ```python fence."""

_LATE_ANSWER_CLOSE = "Answer with what you settled and what you left, and nothing else."

# What the guard sends back, with the reasons that are still open
_COMPLETE_REFUSAL = ("{call}() was not accepted. The merge is not settled:\n{reasons}\n"
                     "Settle these, or call {call}() again saying why they cannot be "
                     "settled. {left_text}")
_REFUSAL_LEFT_NONE = "This is asked once only."
_REFUSAL_LEFT_LAST = "This is the last time it is sent back."
_REFUSAL_LEFT_MORE = "It can be sent back {count} more time(s)."

# What prepare() says when the repository holds anything uncommitted, which the union
# written into the working tree would write over
_UNCOMMIT_REFUSAL = (
    "{project_dir} holds {count} uncommitted path(s) (changed tracked files, and files "
    "git neither tracks nor ignores), and combining branches writes the union straight "
    "into the working tree. Commit them, or stash them with the files git does not "
    "track ({stash}), or move those files out of the repository first; for a submodule "
    "among them, stash or commit what it holds inside it first ({submodule_stash}). "
    "Without --here the merge is made in a clone instead and this repository is only "
    "read: {paths}")

# What prepare() says when a tracked file git is told not to look at holds a change,
# which the union written into the working tree would write over
_MARK_REFUSAL = (
    "{project_dir} holds {count} tracked file(s) git is told not to look at "
    "(assume-unchanged or skip-worktree) that do not hold what the index holds, and "
    "combining branches writes the union straight into the working tree. Take the mark "
    "off ({unmark}), then commit or stash them first. Without --here the merge is made in a clone instead "
    "and this repository is only read: {paths}")

# What prepare() says when the working tree holds what git does not track where a
# branch to combine holds a file, which checking that branch out would write over
_IN_WAY_REFUSAL = (
    "{project_dir} holds {count} path(s) git does not track at {base} that a branch to "
    "combine, or the branch the union is written against, holds a file at or needs the "
    "place of, and combining branches checks those branches out in the working tree. "
    "Move them out of the repository first. Without --here the merge is made in a "
    "clone instead and this repository is only read: {paths}")

_MARKER_REASON = "- conflict markers are still in: {paths}"
_STAND_IN_REASON = ("- these files are still under the stand-in name git gave them where "
                    "it could not put them at their own path: {paths}")
_BROKEN_REASON = "- these calls still do not resolve: {calls}"
_ANALYSIS_REASON = "- the analysis could not be made again: {why}"
_DROP_REASON = ("- {name}, which {branch} added or changed in {file}, is not defined in "
                "any file now")
_NO_PATH_REASON = "- git left a conflict that names no file of the merged tree: {messages}"
_NO_READ_REASON = ("- {paths} hold what the sides both reached and nothing in this run "
                   "has read them")
_NO_READ_BRANCH_REASON = ("- {path} holds what the sides both reached, the merged tree "
                          "does not hold it and nothing in this run has read it. "
                          "read_branch_file reads it on {branches}")
_NO_TOOL_READ_REASON = ("- {paths} hold what the sides both reached and are no file of "
                        "the merged tree or of a branch for a tool of this run to read")
_NO_BRANCH_READ_REASON = ("- the merged tree is not known to hold the change of {branch} "
                          "to {path}, and nothing in this run has read the file as "
                          "{branch} has it. read_branch_file reads it on {branch} at "
                          "{side_path}")
_NO_SIDE_PATH_REASON = ("- the merged tree is not known to hold the change of {branch} "
                        "to {path}, and the path {branch} holds that file at could not "
                        "be told, so the merge is left unsettled. Look for the file on "
                        "{branch} with read_branch_file and say what you made of it")
_SIDE_MODE_REASON = ("- {branch} made {side_path} {side_kind}, and {path} in the working "
                     "tree is {tree_kind}, which reading the file does not show. Give "
                     "{path} the mode {branch} gave it")
# How _SIDE_MODE_REASON names a file by the mode a tree holds it under
_MODE_KIND_DICT = {b"100644": "a file that is not executable",
                   b"100755": "an executable file", b"120000": "a symbolic link"}
# How the commit every branch starts from is named
_START_BRANCH_TEXT = '"" (the starting commit)'
_TEST_REASON = "- `{command}` fails ({why})"
_TEST_EARLY_STOP_REASON = ("- `{command}` stopped before it had run every test ({why}). "
                           "The failing tests it named up to there: {tests}")
_NO_TEST_NAME_TEXT = "none"
_TEST_NO_RUN_REASON = "- `{command}` did not run ({why})"
_TEST_BREAK_REASON = ("- `{command}` fails {tests}, which passed before the branches "
                      "were combined ({why})")
_TEST_UNRUN_REASON = ("- `{command}` no longer runs {tests}, which ran before the branches "
                      "were combined: they are not found, are left out or skip now")
_TEST_FEWER_RUN_REASON = ("- `{command}` runs {now} of the {before} tests it ran before the "
                          "branches were combined: the others are not found, are left out "
                          "or skip now")
_TEST_UNREAD_REASON = ("- `{command}` no longer prints which of its tests ran, as it did "
                       "before the branches were combined, so they cannot be checked against "
                       "the tests that ran then")
_TEST_SHORT_REASON = ("- `{command}` does not print how many tests it passed and failed, or "
                      "its `PASSED` and `FAILED` lines do not name each test it counts, now "
                      "or before the branches were combined, so the tests it runs cannot be "
                      "checked against the tests that ran then")
_TEST_HEAD_STOP_REASON = ("- `{command}` stopped before it had run every test before the "
                          "branches were combined, so the tests it runs cannot be checked "
                          "against how they stood then")

# The block placed after the request: what was measured before the first turn and
# stays as it is for the whole run
_START_TITLE = "## At the start"

# The state block, rebuilt from the repository before every turn
_STATE_TITLE = "## State (turn {turn}/{max_turn})"
_UNMARKED_NOTE = ("of those, git wrote no conflict block in and kept one side's version "
                  "of")
_IN_BLOCK_NOTE = "the merged tree is not known to hold its change to these files"
_MARKER_NOW = "Conflict markers in the working tree, as of this turn:"
_NO_MARKER_NOW = "Conflict markers in the working tree, as of this turn: none."
_CHANGE_NOW = ("Files this stage has changed since the branches were combined, as of "
               "this turn:")
_NO_CHANGE_NOW = ("Files this stage has changed since the branches were combined, as of "
                  "this turn: none.")
_BROKEN_AT_START = ("Calls the merge left pointing at names that were not defined, "
                    "as measured before the first turn. Whether they resolve now is "
                    "settled by the check, not here:")
_NO_BROKEN = "The merge broke no references."
_NO_RESOLVE_NOTE = ("The state block counts the references the analysis could not resolve "
                    "in the merged tree in all. The ones named here are those the merge "
                    "broke, and the rest did not resolve before it either. A reference "
                    "the analysis could not follow to a file at all is in neither count, "
                    "so a call the merge broke can be outside both.")
_NO_RESOLVE_NOW = ("References the analysis could not resolve in the merged tree in all, "
                   "as measured at the last check: {count}.")
_OVERLAP_AT_START = ("Definitions git merged without a conflict block that more than one "
                     "side changed, or that one side changed while another side's change "
                     "uses them, as measured before the first turn. The machine does not "
                     "check that the combined code does what every side was asked; what "
                     "this stage has read and changed there since is in the state block:")
_NO_OVERLAP = ("No definition was changed by more than one side, or changed by one side "
               "and used by another side's change.")
_OVERLAP_BOTH = "  {name} in {file} — changed by {branches}"
_OVERLAP_USE = ("  {name} in {file} — changed by {changed_by}; used at {used_in} line(s) "
                "{lines}, which {used_by} changed")
# The state's line for each broken call and overlapping definition of the start block:
# what this stage has since read and changed in the files it names, from the "read" and
# "changed" entries measured on the same turn
_REACH_NOW = ("What this stage has read and changed in the files of the calls and "
              "definitions named under \"At the start\", as of this turn:")
_REACH_LINE = "  {name} in {file} — {notes}"
_NO_REACH_TEXT = "nothing yet"
_READ_NOW = "Files this stage has read since the branches were combined, as of this turn:"
_NO_READ_NOW = ("Files this stage has read since the branches were combined, as of this "
                "turn: none.")
_TEST_BREAK = ("What the whole test suite said the last time it was run ({when}) that "
               "it did not say before the branches were combined. Running the whole "
               "suite again is what moves this:")
_TEST_HEAD_BLOCK = ("What the whole test suite said the last time it was run ({when}). How "
                    "the tests ran before the branches were combined keeps the check from "
                    "settling them whatever they say now, so running the whole suite again "
                    "does not settle this:")
_NO_TEST_BREAK = ("The whole test suite, the last time it was run ({when}), said "
                  "nothing it did not say before the branches were combined, so a test "
                  "that fails now is one the settling broke.")
_TEST_BEFORE_FIRST_TURN = "before the first turn"
_TEST_ON_TURN = "on turn {turn}"
_ALREADY_RED = ("Tests that were failing before the branches were combined and "
                "still fail on the merged tree. They are not counted against what "
                "you settle; a test that fails and is not one of these is:")
_ALREADY_RED_AT_HEAD = ("Tests that were failing before the branches were combined. The "
                        "tests did not run on the merged tree when it was made. They are "
                        "not counted against what you settle; a test that fails and is "
                        "not one of these is:")
_ALREADY_RED_AT_STOP = ("The run of {commands} stopped before it had run every test before "
                        "the branches were combined, so the tests it runs cannot be checked "
                        "against how they stood then, and the check finds them unsettled "
                        "whatever they say now. The tests that were failing when it "
                        "stopped:")
_ALREADY_RED_AT_SHORT = ("{commands} did not print how many tests it passed and failed, or its "
                         "`PASSED` and `FAILED` lines did not name each test it counted, "
                         "before the branches were combined, so the tests it runs cannot be "
                         "checked against how they stood then, "
                         "and the check finds them unsettled whatever they say now. The tests "
                         "it named as failing then:")
_TEST_NO_COMPARE = ("What the tests say could not be compared with how they stood "
                    "before the branches were combined, so they are not part of the "
                    "check.")
_TEST_NO_COMPARE_PART = ("What the tests of {commands} say could not be compared with "
                         "how they stood before the branches were combined, so they "
                         "are not part of the check. The other tests are.")
_CHECK_AT = "The machine last checked the whole repository on turn {turn}. It found:"
_STATIC_CHECK_TITLE = ("What the static check found in the merged repository before the "
                       "first turn that it did not find at HEAD:")
_CHECK_FIND_NOTHING = "  nothing left"
_NO_CHECK_YET = ("The machine has not checked the whole repository yet. It does that "
                 "when you call COMPLETE().")
_EARLIER_TITLE = ("The merging run this one carries on from, on {branch}, ended with "
                  "\"{ended}\"{why}. What it recorded with record_decision():")
_EARLIER_WHY = ", the check finding: {why}"
_NO_EARLIER_DECISION = "  nothing"
_DECISION_TITLE = "What you have recorded with record_decision():"
_NO_DECISION = ("You have recorded nothing with record_decision() yet. What you record "
                "is kept in this block and with the merge, where a run continuing it "
                "reads it; a reason you only print is not.")


def test_note() -> str:
    """Return the run_test paragraph, or "" when there is no test command to run.

    Returns:
        The paragraph, ending in a newline, or an empty string.
    """
    if not edit.test_can_run():
        return ""
    return _INTEGRATE_TEST_NOTE.format(test_command=edit.test_command_text())


def _test_check() -> str:
    """Return the clause naming the test check, or "" when there is no test command.

    Returns:
        The clause, or an empty string.
    """
    return _INTEGRATE_TEST_CHECK if edit.test_can_run() else ""


def build_integrate_instruction(primary_ext: str) -> str:
    """Assemble the merging stage's instructions for the primary language.

    Args:
        primary_ext: The primary language (config.PRIMARY_EXT).

    Returns:
        The assembled instructions.
    """
    refusal_max = config.INTEGRATE_MAX_REFUSAL
    return INTEGRATE_INSTRUCTION_TEMPLATE.format(
        lang_name=LANG_NAME_DICT[primary_ext],
        path_example=PATH_EXAMPLE_DICT[primary_ext],
        lang_note=edit_paragraph(primary_ext,
                                 {tool.__name__ for tool in merge_tool_list()}),
        output_limit=output_limit_note(OUTPUT_LIMIT_NOTE),
        test_note=test_note(),
        test_check=_test_check(),
        refusal_text=_REFUSAL_TEXT_DICT.get(refusal_max,
                                            _REFUSAL_TEXT_MANY.format(count=refusal_max)))


def integrate_tool_list() -> list[Callable]:
    """Return the reading tools, the editing tools and record_decision, plus run_test.

    The editing tools are replace_in_file, replace_line_range and create_file, and
    run_test is there when edit.test_can_run() says so. run_project, explore,
    web_search, git_diff_stat and revert_file are left out.

    Returns:
        The tool callables.
    """
    tool_list = read_record_tool_list(
        [source.read_source_file, source.read_definition, source.get_use_of_file,
         source.graph_search, source.grep_source,
         edit.replace_in_file, edit.replace_line_range, edit.create_file,
         record_decision])
    if edit.test_can_run():
        tool_list.append(edit.run_test)
    return tool_list


def merge_tool_list() -> list[Callable]:
    """Return integrate_tool_list()'s tools, read_branch_file and delete_file.

    Returns:
        The tool callables.
    """
    tool_list = integrate_tool_list()
    tool_list.insert(3, _record_read(read_branch_file, _path_of_branch_file))
    tool_list.insert(tool_list.index(edit.create_file) + 1, edit.delete_file)
    return tool_list


def read_branch_file(branch: str, path: str, start_line: int = 1,
                     end_line: int | None = None) -> str:
    """Return a file as one merged branch has it, or as the starting commit has it.

    The working tree holds the merged file; this reads one side's own version of it,
    out of the commit. Of a symbolic link it returns the path the link holds.

    Args:
        branch: A branch named in work["branches"], or "" for the commit every branch
            started from.
        path: A repository-relative path, e.g. "myapp/config/settings.py".
        start_line: The line to start at (1-based, inclusive).
        end_line: The line to stop at (inclusive). None reads to the end of the file.

    Returns:
        The file as read_source_file returns it. A string starting with "Error: " when
        branch is not a string or not one being merged, the file is not there or is
        not text, or the line arguments are invalid. It never raises.
    """
    return _read_branch_result(branch, path, start_line, end_line)[0]


def _read_branch_result(branch: str, path: str, start_line: int = 1,
                        end_line: int | None = None) -> tuple[str, bool]:
    """Read a file on one merged branch, or at the starting commit.

    Args:
        branch: As read_branch_file() takes it.
        path: As read_branch_file() takes it.
        start_line: As read_branch_file() takes it.
        end_line: As read_branch_file() takes it.

    Returns:
        (read_branch_file()'s return value, True when the file was read on the
        branch: its text, the range asked for, or the answer that it is not text; False
        when it says why the file was not read).
    """
    type_error = text_error(branch=branch)
    if type_error:
        return f"Error: {type_error}", False
    commit_dict = _STAGE_DICT["commit_dict"]
    if branch not in commit_dict:
        name_text = list_text(sorted(name for name in commit_dict if name))
        return (f"Error: {branch!r} is not a branch being merged. Use one of "
                f"{name_text}, or \"\" for the starting commit"), False
    try:
        rel = to_repo_relative(path_text(path))
    except ValueError as e:
        return f"Error: {e}", False
    if "\0" in rel:
        return f"Error: a path holds no NUL character: {rel!r}", False
    where = _branch_where(branch)
    proc = gitcmd.run(config.PROJECT_DIR, "cat-file", "blob",
                      f"{commit_dict[branch]}:{rel}", text=False)
    if proc.returncode != 0:
        return f"Error: {rel} does not exist {where}", False
    if not proc.stdout:
        return "", True
    with tempfile.TemporaryDirectory() as directory:
        blob_path = os.path.join(directory, os.path.basename(rel))
        with open(blob_path, "wb") as handle:
            handle.write(proc.stdout)
        if not is_text_file(blob_path):
            return _NOT_TEXT_ERROR.format(path=rel, where=where), True
        text = read_source_text(blob_path)
    return source.text_range(text, rel, start_line, end_line)


# What read_branch_file() returns for a file a branch holds that is not text
_NOT_TEXT_ERROR = "Error: {path} {where} is not text"


def _branch_where(branch: str) -> str:
    """Say where read_branch_file() reads.

    Args:
        branch: A name read_branch_file() takes.

    Returns:
        "on <branch>", or "at the starting commit" for "".
    """
    return f"on {branch}" if branch else "at the starting commit"


# What one run of the stage keeps between turns: "decision_list" is what the model has
# recorded with record_decision(), "gap_list" what it has recorded with record_gap(),
# "read_dict" is {repository-relative path: the last turn this stage read it on},
# "branch_read_set" is the (branch, repository-relative path) pairs read_branch_file has
# read, "turn" is the turn that is running, "test_run" is the count of whole test runs
# the state last read (edit.last_whole_run()), "gap_check" is how record_gap() runs a
# case's test (set_gap_check), "commit_dict" is
# {a name read_branch_file takes: its commit}, "" for the starting commit,
# "head_baseline" says the tests not counted are HEAD's failures alone, the merged
# tree's tests not having run, "head_stop" holds the test commands whose run over HEAD's
# tree was red and ended before every test was run (edit.is_stop_before), "head_short"
# the ones whose run over HEAD's tree does not name each test it counts
# (_is_head_short), and "not_compared" holds the test commands the baseline leaves out
# of the check
_STAGE_DICT: dict = {"decision_list": [], "gap_list": [], "read_dict": {},
                     "branch_read_set": set(), "turn": 0,
                     "test_run": 0, "gap_check": None, "commit_dict": {},
                     "head_baseline": False, "head_stop": [], "head_short": [],
                     "not_compared": []}


def reset_read() -> None:
    """Empty the files read, the branch files read and the turn."""
    _STAGE_DICT["read_dict"].clear()
    _STAGE_DICT["branch_read_set"].clear()
    _STAGE_DICT["turn"] = 0


def reset_stage() -> None:
    """Empty what the stage has recorded and read."""
    _STAGE_DICT["decision_list"].clear()
    _STAGE_DICT["gap_list"].clear()
    reset_read()
    _STAGE_DICT["test_run"] = 0
    _STAGE_DICT["gap_check"] = None
    _STAGE_DICT["commit_dict"] = {}
    _STAGE_DICT["head_baseline"] = False
    _STAGE_DICT["head_stop"] = []
    _STAGE_DICT["head_short"] = []
    _STAGE_DICT["not_compared"] = []


# What keeps the check of the tests from finding a merge settled whatever the merged
# tree holds (head_block_list): a test command's run over HEAD's tree that ended before
# every test was run, and one that does not name each test it counts
HEAD_STOP_BLOCK = "stop"
HEAD_SHORT_BLOCK = "short"


def head_block_list() -> list[str]:
    """Return what keeps the check of the tests from finding any merge settled.

    It is read off the run over HEAD's tree the last merge was measured against.

    Returns:
        HEAD_STOP_BLOCK when a test command's run over HEAD's tree was red and ended
        before every test was run (_TEST_HEAD_STOP_REASON), then HEAD_SHORT_BLOCK
        when one does not name each test it counts (_TEST_SHORT_REASON), each at
        most once; [] when neither.
    """
    return [block for block, key in ((HEAD_STOP_BLOCK, "head_stop"),
                                     (HEAD_SHORT_BLOCK, "head_short"))
            if _STAGE_DICT[key]]


def set_gap_check(check: Callable[[str, str], bool | None]) -> None:
    """Say how record_gap() runs the test of a case it is given.

    Args:
        check: (the test's path, its text) -> gap_test_result()'s return value, its
            baseline_set already given.
    """
    _STAGE_DICT["gap_check"] = check


def set_turn(turn: int) -> None:
    """Keep the turn that is about to run.

    Args:
        turn: The turn (1-based).
    """
    _STAGE_DICT["turn"] = turn


def stage_turn() -> int:
    """Return the turn set_turn() last kept, 0 before the first one."""
    return _STAGE_DICT["turn"]


def read_list() -> list[dict]:
    """Return the files this stage has read, and the last turn each was read on.

    Returns:
        [{"path", "turn"}, ...] sorted by path.
    """
    return [{"path": path, "turn": turn}
            for path, turn in sorted(_STAGE_DICT["read_dict"].items())]


def _path_argument(name: str) -> Callable:
    """Build the reader of a tool's path argument, by the name the tool gives it.

    Args:
        name: The parameter holding the repository-relative path.

    Returns:
        (tool, args, kwargs, result) -> the path, or "" when the call did not bind.
    """
    def path_of(tool: Callable, args: tuple, kwargs: dict, result: object) -> str:
        """Return the argument under name of one call, or "" when it did not bind."""
        try:
            return str(inspect.signature(tool).bind(*args, **kwargs).arguments.get(name, ""))
        except TypeError:
            return ""

    return path_of


def _path_of_branch_file(tool: Callable, args: tuple, kwargs: dict, result: object) -> str:
    """Return the file read_branch_file() read, and keep the branch it was read on.

    Args:
        tool: read_branch_file.
        args: Its positional arguments.
        kwargs: Its keyword arguments.
        result: Its return value.

    Returns:
        The path as it was passed, "" when the call did not bind. A path other than ""
        is added to the stage's "branch_read_set" with its branch.
    """
    path = _path_argument("path")(tool, args, kwargs, result)
    branch = _path_argument("branch")(tool, args, kwargs, result)
    if not path:
        return ""
    _STAGE_DICT["branch_read_set"].add((branch, to_repo_relative(path_text(path))))
    return path


def _path_of_definition(tool: Callable, args: tuple, kwargs: dict, result: object) -> str:
    """Return the file read_definition() read, from what it returned.

    Args:
        tool: read_definition.
        args: Its positional arguments.
        kwargs: Its keyword arguments.
        result: Its return value.

    Returns:
        The file's path when the definition was found, "" otherwise.
    """
    return str(result.get("file", "")) if isinstance(result, dict) and result.get("ok") else ""


def _path_of_use(tool: Callable, args: tuple, kwargs: dict, result: object) -> str:
    """Return the file get_use_of_file() was asked about, when the repository holds it.

    Args:
        tool: get_use_of_file.
        args: Its positional arguments.
        kwargs: Its keyword arguments.
        result: Its return value.

    Returns:
        The path as it was passed, "" for a path that is not a file under PROJECT_DIR.
    """
    path = _path_argument("target_file")(tool, args, kwargs, result)
    try:
        return path if path and os.path.isfile(resolve_in_project(to_repo_relative(path))) \
            else ""
    except ValueError:
        return ""


def _record_read(tool: Callable, path_of: Callable) -> Callable:
    """Wrap a reading tool: the file it reads is kept with the turn it is read on.

    The wrapped tool takes the tool's name, docstring and signature (functools.wraps). A
    tool _READ_RESULT_DICT names is called through the function that says, apart from
    its answer, whether the call read the file, and a call that did not is not kept. A
    call whose arguments do not fit the tool's signature goes to the tool itself, which
    raises its own TypeError.

    Args:
        tool: The reading tool.
        path_of: (tool, args, kwargs, result) -> the repository-relative path read, or
            "" for a call that read no file.

    Returns:
        The wrapped tool.
    """
    read_of = _READ_RESULT_DICT.get(tool)

    @functools.wraps(tool)
    def read_tool(*args: object, **kwargs: object) -> object:
        """Call the tool and keep the file it read with the turn."""
        if read_of:
            try:
                inspect.signature(tool).bind(*args, **kwargs)
            except TypeError:
                # Arguments the tool does not take are refused by the tool itself
                return tool(*args, **kwargs)
        tool_result, is_read = read_of(*args, **kwargs) if read_of \
            else (tool(*args, **kwargs), True)
        path = path_of(tool, args, kwargs, tool_result) if is_read else ""
        if path:
            # Kept relative to the repository root, whether it was read by its index
            # path or by an absolute one
            _STAGE_DICT["read_dict"][to_repo_relative(path)] = _STAGE_DICT["turn"]
        return tool_result

    return read_tool


# How the file a reading tool read is taken from one call of it, by the tool's name
_READ_PATH_DICT = {"read_source_file": _path_argument("path"),
                   "read_definition": _path_of_definition,
                   "get_use_of_file": _path_of_use}

# The reading tools whose answer can be any text of a file, each with the function
# that reads as it does and says whether it read
_READ_RESULT_DICT = {source.read_source_file: source.read_source_result,
                     read_branch_file: _read_branch_result}


def read_record_tool_list(tool_list: list[Callable]) -> list[Callable]:
    """Wrap the tools _READ_PATH_DICT names: the file each reads is kept with the turn.

    Args:
        tool_list: The stage's tools.

    Returns:
        A new list in the same order, with read_source_file, read_definition and
        get_use_of_file wrapped by _record_read() and every other tool as it is.
    """
    return [_record_read(tool, _READ_PATH_DICT[tool.__name__])
            if tool.__name__ in _READ_PATH_DICT else tool for tool in tool_list]


def decision_list() -> list[dict]:
    """Return what has been recorded so far, oldest first.

    Returns:
        [{"what", "why"}, ...], at most config.INTEGRATE_WORK_MAX_ITEM of them.
    """
    return list(_STAGE_DICT["decision_list"])


def record_decision(what: str, why: str) -> dict:
    """Record one thing you have settled and why you settled it that way.

    The entry is shown in the state block on every turn after this one. It records what
    you decided; it does not settle anything on its own, and the checks are made over
    the repository either way.

    Args:
        what: What was settled, e.g. the path and which conflict in it.
        why: Why it was settled that way, e.g. which side's request it keeps.

    Returns:
        {"ok": bool, "recorded": how many are held now}. "ok" is False when there is no
        room for another one.
    """
    return _record_entry("decision_list", what, why)


def gap_list() -> list[dict]:
    """Return what record_gap() has recorded so far, oldest first.

    Returns:
        [{"what", "why"}, ...], at most config.INTEGRATE_WORK_MAX_ITEM of them.
    """
    return list(_STAGE_DICT["gap_list"])


# What record_gap() returns for a case given no test while the stage can run one
_GAP_NO_TEST = ("nothing was recorded: pass the test you would have written for the case "
                "in test_path and test_text, so that it is run. A case no test can reach "
                "is not one the change does not do: say why with record_decision().")


def record_gap(what: str, why: str, test_path: str = "", test_text: str = "") -> dict:
    """Record one case the change does not do, which you leave without a test in the suite.

    Call it where the code does not do what a case's sentence says and you do not change
    that code. Pass the test you would have written for the case in test_path and
    test_text: it is run once here, over the tree as it is, and taken back out; it is
    never left in the repository. A test that does not pass is kept with the case. A
    case given no test is not recorded while the stage can run one (set_gap_check).

    Args:
        what: The case's id, as work["cases"] spells it.
        why: Which file, and what it does that the sentence does not say.
        test_path: Where the test would go, relative to the repository root: a file
            that is not there yet, under the tests.
        test_text: The whole of that test file.

    Returns:
        {"ok": bool, "recorded": how many are held now, "test": what the test did:
        "failed", "passed" (the tree does this case already: write the test into the
        suite rather than record the gap), "unclear" (it did not run, or failed only
        as the suite already did) or "none" (no test was passed)}. "ok" is False when
        there is no room for another one, and when no test was passed while the stage
        can run one, which adds "error".
    """
    check = _STAGE_DICT["gap_check"]
    if check is not None and not (test_path and test_text):
        return {"ok": False, "recorded": len(_STAGE_DICT["gap_list"]), "test": "none",
                "error": _GAP_NO_TEST}
    test_dict = {}
    check_result = None
    if test_path and test_text and check is not None:
        check_result = check(test_path, test_text)
        # A test that passes is not kept with the case
        if check_result is not False:
            test_dict = {"test_path": test_path, "test_text": test_text}
    record_dict = _record_entry("gap_list", what, why)
    if record_dict["ok"] and test_dict:
        _STAGE_DICT["gap_list"][-1].update(test_dict)
    word = "none" if not (test_path and test_text and check is not None) \
        else {True: "failed", False: "passed", None: "unclear"}[check_result]
    return {**record_dict, "test": word}


def gap_test_result(test_path: str, test_text: str, baseline_set: frozenset[str]) -> bool | None:
    """Run a case's test, written into the tree for this run alone, and say what it did.

    The file is written as create_file writes a new file (textfile.new_file_form), the
    test command is run over it (the whole suite where the language's tool takes no
    paths), and the file and any directory made for it, with whatever the run wrote in
    such a directory, are taken away again; what cannot be taken away is logged and
    left. A file written where one was taken away before is given a modification time
    and size the path has not had (tmpfile.move_mtime_on). What the stage's own runs
    left behind is kept as it was.

    Args:
        test_path: Where the test goes, relative to the repository root.
        test_text: The whole of the test file.
        baseline_set: The set the failing names are held to, as testcreate.baseline_of()
            returns it or as a run's "failed" holds them: a failing test it names is not
            counted.

    Returns:
        True when the run fails with a test the baseline does not hold, or fails naming
        no test where the tool takes paths, False when it passes, None when neither can
        be told: the path is outside the repository or already there, it is a file
        another language's test command runs (edit.is_other_test_path), the text holds a
        character the file's encoding cannot write, the file or its directories could
        not be written, or the file could not be put back as it was after the run with
        a line added to it (edit.in_whole_run raised), the run did not come to a verdict,
        it failed naming only tests the baseline holds, or naming none where the tool
        takes no paths, or it passed where the tool takes no paths and
        edit.in_whole_run() does not say the whole suite builds the file.
    """
    try:
        full_path = resolve_in_project(test_path)
    except ValueError:
        return None
    if edit.is_other_test_path(test_path):
        return None
    if os.path.exists(full_path):
        return None
    try:
        content = textfile.encode_text(
            "", test_text, textfile.new_file_form(config.PROJECT_DIR, test_path))
    except ValueError as e:
        logger.info("the test of a case was not run: %s", e)
        return None
    new_dir_list = []
    directory = os.path.dirname(full_path)
    while not os.path.isdir(directory):
        new_dir_list.append(directory)
        directory = os.path.dirname(directory)
    try:
        for directory in reversed(new_dir_list):
            os.mkdir(directory)
        before_stamp = tmpfile.file_stamp(full_path)
        with open(full_path, "wb") as fp:
            fp.write(content)
        tmpfile.move_mtime_on(full_path, before_stamp)
        with edit.keep_test_state():
            result_dict = edit.run_test([test_path])
        # A pass where the tool takes no paths says the case is done only for a file
        # the whole suite builds
        is_build = result_dict["status"] != edit.TEST_STATUS_DICT[True] \
            or config.TEST_TARGET_STYLE != "none" or edit.in_whole_run(test_path, result_dict)
    except OSError as e:
        logger.warning("the test of a case at %s could not be written or put back as it "
                       "was, so what it does is not told: %s", test_path, e)
        return None
    finally:
        if os.path.exists(full_path):
            tmpfile.file_stamp(full_path)
            try:
                os.remove(full_path)
            except OSError as e:
                logger.warning("%s, where the test of a case was run, could not be taken "
                               "away again: %s", test_path, e)
        # The outermost directory made here goes with everything under it
        if new_dir_list:
            shutil.rmtree(new_dir_list[-1], ignore_errors=True)
    if result_dict["status"] == edit.TEST_STATUS_DICT[True]:
        return False if is_build else None
    if result_dict["status"] != edit.TEST_STATUS_DICT[False]:
        return None
    if set(result_dict["failed"]) - baseline_set:
        return True
    if not result_dict["failed"] and config.TEST_TARGET_STYLE != "none":
        return True
    return None


def recheck_gap_list(gap_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """Run the tests the cases were recorded with again, over the tree as it is now.

    Args:
        gap_list: record_gap()'s entries, as the runs of the stage recorded them.

    Returns:
        (the cases still not done, the cases whose test passes now). A case recorded
        without a test, and one whose test says nothing either way, stays in the first.
    """
    check_list = [entry_dict for entry_dict in gap_list if entry_dict.get("test_path")]
    baseline_set = frozenset()
    if check_list and config.TEST_TARGET_STYLE == "none":
        with edit.keep_test_state():
            baseline_set = frozenset(edit.run_test()["failed"])
    keep_list, drop_list = [], []
    for entry_dict in gap_list:
        is_pass = entry_dict.get("test_path") and gap_test_result(
            entry_dict["test_path"], entry_dict["test_text"], baseline_set) is False
        (drop_list if is_pass else keep_list).append(entry_dict)
    return keep_list, drop_list


def _record_entry(key: str, what: str, why: str) -> dict:
    """Add one {"what", "why"} entry to one of the stage's lists, within the limit.

    Args:
        key: The list in _STAGE_DICT, "decision_list" or "gap_list".
        what: The entry's "what".
        why: The entry's "why".

    Returns:
        {"ok": bool, "recorded": how many that list holds now}.
    """
    entry_list = _STAGE_DICT[key]
    limit = config.INTEGRATE_WORK_MAX_ITEM
    if len(entry_list) >= limit:
        logger.info("an entry was not recorded in %s: %d are already held", key, limit)
        return {"ok": False, "recorded": len(entry_list)}
    entry_list.append({"what": cut(str(what), config.EDIT_HISTORY_DID_CHAR),
                       "why": cut(str(why), config.EDIT_HISTORY_DID_CHAR)})
    return {"ok": True, "recorded": len(entry_list)}


def _broken_entry_list(broken_dict: dict, name_dict: dict, project_name: str) -> list[dict]:
    """Render the references the merge broke, as the work list shows them.

    The number of names offered for each is cut to config.INTEGRATE_WORK_MAX_ITEM. The
    references themselves are all returned.

    Args:
        broken_dict: broken_ref_dict()'s mapping.
        name_dict: analyse()'s "name" for the merged state.
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        [{"file", "lines", "name", "defined_in", "now_defines"}, ...], sorted, with
        every path relative to the repository root.
    """
    entry_list = []
    for (from_file, to_file, name), line_list in sorted(broken_dict.items()):
        entry_list.append({
            "file": knowledge_to_repo(from_file, project_name),
            "lines": line_list,
            "name": name,
            "defined_in": knowledge_to_repo(to_file, project_name),
            "now_defines": name_dict.get(to_file, [])[:config.INTEGRATE_WORK_MAX_ITEM],
        })
    return entry_list


def broken_list_of(before_dict: dict, after_dict: dict, project_name: str,
                   side_no_resolve_dict: dict | None = None) -> list[dict]:
    """Return the references the merge broke, as the work list shows them.

    A key is left out when the merged state holds no more of its lines than the start
    or one of the branches alone holds, which takes in one whose key a side moved by
    renaming or moving a file.

    Args:
        before_dict: analyse()'s return value for the commit the union is written out
            against.
        after_dict: analyse()'s return value for the merged state.
        project_name: The directory name of PROJECT_DIR, for the path conversion.
        side_no_resolve_dict: The references that do not resolve on the branches taken
            one at a time, keyed as analyse()'s "no_resolve" is. None counts them all.

    Returns:
        _broken_entry_list()'s list over the references that resolve in before_dict and
        on every side of the merge, and no longer resolve in after_dict.
    """
    before_no_resolve_dict = dict(before_dict["no_resolve"])
    add_no_resolve(before_no_resolve_dict, side_no_resolve_dict or {})
    return _broken_entry_list(
        broken_ref_dict(before_no_resolve_dict, after_dict["no_resolve"]),
        after_dict["name"], project_name)


def build_work_dict(merge_dict: dict, marker_list: list[str], broken_list: list[dict],
                    baseline_set: frozenset[str] | None, test_reason: str,
                    no_resolve_count: int, overlap_list: list[dict] | None = None,
                    earlier_dict: dict | None = None) -> dict:
    """Build the merging stage's work dict.

    The machine writes every field; "decisions" holds what the model recorded.

    Args:
        merge_dict: assemble()'s return value.
        marker_list: marker_path_list() over the merged working tree.
        broken_list: _broken_entry_list()'s list, uncut.
        baseline_set: _baseline_set()'s return value.
        test_reason: _test_reason() over the merged tree, or "" when the tests said nothing
            they did not say before.
        no_resolve_count: How many references the analysis could not resolve over the
            merged tree in all, the ones the merge broke included.
        overlap_list: overlap_entry_list()'s list, or None for an empty one.
        earlier_dict: earlier_record_dict()'s value for the branch this run carries on
            from, or None when it carries on from none.

    Returns:
        {"base", "branches": [{"branch", "commit", "conflicted", "in_block" (the paths
        of the merged tree alone), "said", "did"}, ...], "markers" (rebuilt every turn),
        "unmarked" (empty here), "changed" and "read" (what this stage has changed since
        the branches were combined, and what it has read, both rebuilt every turn),
        "broken", "unresolved", "overlap", "tests", "checks" (empty here), "tests_at"
        (the turn of the run "tests" comes from, 0 before the first), "already_failing"
        (the baseline's names, None where there is none), "left" (what the last full
        check found), "checked_at" (the turn that check ran on, 0 before the first one),
        "decisions" (what the model recorded), "earlier" (what the merging run this one
        carries on from recorded, or None) and "complete"}, with the lists cut to
        config.INTEGRATE_WORK_MAX_ITEM and "complete" False when anything was left out
        by that cut. "already_failing" is cut by the same count but does not set
        "complete".
    """
    limit = config.INTEGRATE_WORK_MAX_ITEM
    overlap_list = overlap_list or []
    work_dict = {
        "base": merge_dict["base"],
        "branches": [{"branch": entry_dict["branch"],
                      "commit": gitcmd.short(entry_dict["commit"]),
                      "conflicted": entry_dict["conflicted"][:limit],
                      "in_block": sorted(entry_dict.get("in_block") or {})[:limit],
                      "said": entry_dict["said"][:limit],
                      "did": entry_dict["did"][:limit]}
                     for entry_dict in merge_dict["branches"]],
        "markers": marker_list[:limit],
        "unmarked": [],
        "changed": [],
        "read": [],
        "broken": broken_list[:limit],
        "unresolved": no_resolve_count,
        "overlap": overlap_list[:limit],
        "tests": test_reason,
        "checks": [],
        "tests_at": 0,
        "already_failing": None if baseline_set is None
        else edit.baseline_name_list(baseline_set)[:limit],
        "left": [],
        "checked_at": 0,
        "decisions": [],
        "earlier": earlier_dict,
        "complete": True,
    }
    work_dict["complete"] = is_complete(work_dict, marker_list, broken_list,
                                        overlap_list, merge_dict["branches"])
    return work_dict


def is_complete(work_dict: dict, marker_list: list[str], broken_list: list[dict],
                overlap_list: list[dict], branch_list: list[dict]) -> bool:
    """Say whether the state holds everything there is, or was cut to the limit.

    Args:
        work_dict: The state as it stands.
        marker_list: The conflict markers found now, uncut.
        broken_list: The references the merge broke, uncut.
        overlap_list: The definitions more than one side's change reaches, uncut.
        branch_list: The merged branches, each with its uncut "conflicted" and
            "in_block".

    Returns:
        True when nothing was left out by the cut.
    """
    limit = config.INTEGRATE_WORK_MAX_ITEM
    over_list = [len(marker_list) > limit, len(broken_list) > limit,
                 len(overlap_list) > limit, len(work_dict["changed"]) > limit,
                 len(work_dict["read"]) > limit]
    over_list += [len(entry_dict["conflicted"]) > limit
                  or len(entry_dict.get("in_block") or {}) > limit
                  for entry_dict in branch_list]
    return not any(over_list)


def request_of(merge_dict: dict, request: str = "") -> str:
    """Build the sentence the merging stage is asked, from what the merge left behind.

    Only what stays true for the whole run is here.

    Args:
        merge_dict: assemble()'s return value.
        request: The request the branches were made for, as the user wrote it: what they
            were split out of.

    Returns:
        The request text, naming the branches and what git said about the merge.
    """
    branch_text = list_text([entry_dict["branch"] for entry_dict in merge_dict["branches"]])
    part_list = []
    if request.strip():
        part_list.append(f"The request these branches were made for: "
                         f"{cut(request.strip(), config.DISPATCH_TASK_MAX_CHAR)}")
    part_list.append(f"Combine the changes on {branch_text}.")
    part_list += merge_dict["messages"][:config.INTEGRATE_WORK_MAX_ITEM]
    return " ".join(part_list)


# == What the machine checks before a run is allowed to end ==

# What stands for the output of the tests over HEAD's tree where that tree was not put
# in the working tree
_NO_HEAD_TREE_TEXT = ("the tests were not run over HEAD's tree: a file of the working "
                      "tree in its way could not be kept aside")


def _at_head(project_dir: str, union: str, measure: Callable[[], object]) -> object:
    """Measure HEAD's tree, then put the union back in the working tree.

    Args:
        project_dir: The repository, holding the union uncommitted against HEAD.
        union: The commit holding the union, as assemble() returns it.
        measure: () -> what is read off the tree, e.g. edit.run_test or edit.run_check.

    Returns:
        measure()'s return value over HEAD's tree. The files of the working tree no tree
        holds that HEAD's tree is written over, and the tracked files the working tree
        holds otherwise than the union, come back as they were (keep_file). None,
        with HEAD's tree not put in place and nothing measured, when one of them cannot
        be kept (KeepError), which is logged.
    """
    try:
        with keep_file(project_dir, "HEAD", union):
            place(project_dir, "HEAD")
            try:
                return measure()
            finally:
                place(project_dir, union)
    except KeepError as e:
        logger.warning("HEAD's tree is not measured: %s", e)
        return None


def _head_test_dict(project_dir: str, union: str) -> dict:
    """Run the tests over HEAD's tree.

    Args:
        project_dir: The repository, holding the union uncommitted against HEAD.
        union: The commit holding the union, as assemble() returns it.

    Returns:
        edit.run_test()'s return value over HEAD's tree (_at_head), or a result
        saying the tests did not run when HEAD's tree was not put in place.
    """
    result_dict = _at_head(project_dir, union, edit.run_test)
    if result_dict is not None:
        return result_dict
    return {"status": edit.TEST_STATUS_DICT[None], "failed": [],
            "failed_excerpt": _NO_HEAD_TREE_TEXT, "targets": [],
            "output": _NO_HEAD_TREE_TEXT}


class KeepError(RuntimeError):
    """Raised when what putting a commit's tree in place takes away cannot be kept.

    It is something in the working tree. Nothing has been written when it is raised.
    """


@contextlib.contextmanager
def keep_file(project_dir: str, commit: str, tree: str,
              env_dict: dict[str, str] | None = None) -> Iterator[None]:
    """Keep the files of the working tree that putting two trees in place takes away.

    The trees are a commit's tree, put in place, and another tree put back afterwards.
    The files are what the working tree holds at a path the commit holds and the tree
    does not, or in the way of one (_in_way_path): ones no tree is made of, such as a
    file an ignore rule covers now and the commit tracks, or a file where the commit
    holds a directory. They are also the tracked files whose working-tree contents
    differ from the index (_work_change_path_list). Each file and link is copied aside
    when the block starts and written back when it ends, in place of whatever the block
    left at its path. One that cannot be written back is logged. A directory at such a
    path is not kept.

    Args:
        project_dir: The repository.
        commit: The commit whose tree is put in the working tree inside the block.
        tree: The commit or the tree put back before the block ends.
        env_dict: As gitcmd.run() takes it, for a tree gitcmd.work_tree_index() wrote.

    Yields:
        Nothing.

    Raises:
        KeepError: Before the block is entered, when one of them is no file and no
            link, or cannot be copied.
    """
    keep_dict: dict[str, None] = {}
    for rel_path in _tree_add_path_list(project_dir, commit, tree, env_dict):
        in_way_path = _in_way_of(project_dir, rel_path, tree, env_dict)
        if in_way_path:
            keep_dict[os.path.join(project_dir, in_way_path)] = None
    for rel_path in _work_change_path_list(project_dir, env_dict):
        keep_dict[os.path.join(project_dir, rel_path)] = None
    with tempfile.TemporaryDirectory() as directory:
        copy_list = []
        for index, path in enumerate(keep_dict):
            copy_path = os.path.join(directory, str(index))
            try:
                mode = os.lstat(path).st_mode
                if not stat.S_ISLNK(mode) and not stat.S_ISREG(mode):
                    raise OSError("it is neither a file nor a link")
                shutil.copy2(path, copy_path, follow_symlinks=False)
            except OSError as e:
                raise KeepError(f"{path} is in the way of the tree of "
                                f"{gitcmd.short(commit)} and could not be kept aside "
                                f"({e})") from e
            copy_list.append((path, copy_path))
        try:
            yield
        finally:
            for path, copy_path in copy_list:
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    if os.path.isdir(path) and not os.path.islink(path):
                        shutil.rmtree(path)
                    elif os.path.lexists(path):
                        os.remove(path)
                    shutil.copy2(copy_path, path, follow_symlinks=False)
                except OSError as e:
                    logger.error("%s could not be put back as it was before the tests "
                                 "were run over %s (%s)", path, gitcmd.short(commit), e)


def _work_change_path_list(project_dir: str,
                           env_dict: dict[str, str] | None = None) -> list[str]:
    """Return the tracked files whose working-tree contents differ from the index.

    Args:
        project_dir: The repository.
        env_dict: As gitcmd.run() takes it, for an index gitcmd.work_tree_index() wrote.

    Returns:
        Repository-relative paths as os.fsdecode() gives them, in git's order, of the
        files and links the working tree holds. [] when git cannot compare, which is
        logged.
    """
    proc = gitcmd.run(project_dir, *gitcmd.PATH_DIFF_TUPLE, "--ignore-submodules=all",
                      "--name-only", "-z", "--diff-filter=M", env_dict=env_dict,
                      text=False)
    if proc.returncode != 0:
        logger.warning("the changed files of the working tree could not be listed: %s",
                       gitcmd.shown_text(proc.stderr).strip())
        return []
    return [os.fsdecode(raw) for raw in proc.stdout.split(b"\0") if raw]


def _tree_add_path_list(project_dir: str, commit: str, tree: str,
                        env_dict: dict[str, str] | None = None) -> list[str]:
    """Return the paths a commit's tree holds and another tree does not.

    Args:
        project_dir: The repository.
        commit: The commit whose paths are listed.
        tree: The commit or the tree they are looked for in.
        env_dict: As gitcmd.run() takes it.

    Returns:
        Repository-relative paths as os.fsdecode() gives them, in git's order. [] when
        git cannot compare the two, which is logged.
    """
    proc = gitcmd.run(project_dir, *gitcmd.PATH_DIFF_TUPLE, "--name-only", "-z",
                      "--diff-filter=A", tree, commit, env_dict=env_dict, text=False)
    if proc.returncode != 0:
        logger.info("cannot read which files %s holds and %s does not: %s",
                    gitcmd.short(commit), gitcmd.short(tree),
                    gitcmd.shown_text(proc.stderr).strip())
        return []
    return [os.fsdecode(raw_path) for raw_path in proc.stdout.split(b"\0") if raw_path]


def _in_way_of(project_dir: str, rel_path: str, tree: str,
               env_dict: dict[str, str] | None = None) -> str:
    """Return what the working tree holds in the way of a file being written.

    The file is written at a path the tree does not hold, and what is in the way is not
    held by the tree either.

    Args:
        project_dir: The repository.
        rel_path: A repository-relative path the tree does not hold, as os.fsdecode()
            gives it.
        tree: The commit or the tree.
        env_dict: As gitcmd.run() takes it.

    Returns:
        _in_way_path()'s path, where it is rel_path itself or a directory above it the
        tree does not hold as well. "" otherwise.
    """
    in_way_path = _in_way_path(project_dir, rel_path)
    # Something above the path that the tree holds comes back with the tree
    if in_way_path and (in_way_path == rel_path or gitcmd.run(
            project_dir, "cat-file", "-e", f"{tree}:{in_way_path}",
            env_dict=env_dict).returncode != 0):
        return in_way_path
    return ""


def _user_in_way_list(project_dir: str, base: str, commit_list: list[str]) -> list[str]:
    """Return what checking the commits out writes over that base does not track.

    It is what the working tree holds that base does not track and that checking one of
    the commits out writes over or takes away: that is a file or a link at a path a
    commit holds and base does not, or in the way of one (_in_way_of), and what git does
    not track inside a directory at such a path: a file an ignore rule covers, such as
    one holding local settings, among them.

    Args:
        project_dir: The repository, checked out at base.
        base: The commit checked out.
        commit_list: The commits whose trees are put in the working tree.

    Returns:
        Repository-relative paths, a directory with "/" at its end, in the order found.
    """
    path_dict: dict[str, None] = {}
    for commit in commit_list:
        for rel_path in _tree_add_path_list(project_dir, commit, base):
            path = os.path.join(project_dir, rel_path)
            if os.path.isdir(path) and not os.path.islink(path):
                proc = gitcmd.run(project_dir, "ls-files", "-z", "--others", "--directory",
                                  "--", f":(literal){rel_path}", text=False)
                path_dict.update(dict.fromkeys(os.fsdecode(raw_path) for raw_path
                                               in proc.stdout.split(b"\0") if raw_path))
                continue
            in_way_path = _in_way_of(project_dir, rel_path, base)
            if in_way_path:
                path_dict[in_way_path] = None
    return list(path_dict)


def _in_way_path(project_dir: str, rel_path: str) -> str:
    """Return what the working tree holds in the way of a file being written at a path.

    Args:
        project_dir: The repository.
        rel_path: A repository-relative path, as os.fsdecode() gives it.

    Returns:
        The first of the path and the directories it is under, from the top, at
        which the working tree holds something that is no directory: a file, a link
        or anything else. "" when it holds nothing there, or a directory at the path
        itself.
    """
    part_list = rel_path.split("/")
    for count in range(1, len(part_list) + 1):
        in_way_path = "/".join(part_list[:count])
        path = os.path.join(project_dir, in_way_path)
        if not os.path.lexists(path):
            return ""
        if os.path.islink(path) or not os.path.isdir(path):
            return in_way_path
    return ""


# What check_reason() says when the static check finds more than it found when the stage
# started
_CHECK_REASON = ("- `{command}` finds what it did not find before you started: {findings}")


def check_reason(before: collections.Counter | None) -> list[str]:
    """Run the static check again and say what it finds that it did not find before.

    Args:
        before: edit.run_check() at the stage's start. None leaves the check out.

    Returns:
        One reason line, or [] when nothing is new or the check did not run.
    """
    if before is None:
        return []
    after = edit.run_check()
    if after is None:
        return []
    new_list = edit.new_finding_list(before, after)
    if not new_list:
        return []
    return [_CHECK_REASON.format(command=edit.check_command_text(),
                                 findings=list_text(new_list))]


def _baseline_set(test_dict: dict | None,
                  at_head: Callable[[], dict]) -> frozenset[str] | None:
    """Return the baseline the merged tree's tests are held to.

    The merged tree is held to the commit the branches were combined onto, by the names
    run_test() scrapes: a test red in both is not counted, and the tests HEAD's run
    ran are held to running still (edit.ran_entry_set). A merged tree whose tests did
    not run is held to HEAD's run: the tests failing there are the ones not counted.
    Where HEAD's run found no test to run, a red merged tree is held as if every test
    passed at HEAD, and any other is held by edit.NO_TEST_BASELINE.

    A run of several test commands is compared command by command
    (edit.part_dict_of), each with the same command's run over HEAD's tree, and the
    sets are put together (edit.join_baseline): a command that is left out leaves the
    others compared.

    Args:
        test_dict: run_test()'s return value over the merged tree, taken before a model was
            called, or None where there is no test command to run.
        at_head: () -> run_test()'s return value over HEAD's tree.

    Returns:
        The names of the tests failing in both states, each as many times as it failed
        in both (edit.fail_entry_set; in HEAD's run alone where the merged tree's did
        not run), none when nothing that fails on the merged tree failed before, with
        the entry of the tests HEAD's run ran, edit.NO_TEST_BASELINE (or its entry) for
        a command whose run found no test to run at HEAD and on the merged tree alike,
        HEAD's failing names with edit.stop_entry_set()'s entry where HEAD's run is red
        and ended before every test was run (edit.early_stop_run), or None when there is
        nothing to compare against and the tests are left out of the check: HEAD's tests
        did not run for another reason, or HEAD is red and its run or the merged tree's
        named no test. Where one of those two runs does not name each test it counts
        (edit.RunResult.is_name_short), the failing names of HEAD's run are not counted,
        with the entry of the tests HEAD's run ran. HEAD's run that says nothing of how
        many tests it ran (edit.RunResult.is_ran_unread) is not left out where it did
        not run either: the set is the entry of the tests it ran.
    """
    if test_dict is None:
        return None
    part_dict = edit.part_dict_of(test_dict)
    if not part_dict:
        return _part_baseline_set(test_dict, at_head)
    # HEAD's tree is measured once, when the first command's run asks for it
    head_list: list[dict] = []

    def head_part(command: str) -> dict:
        """Return one command's own result over HEAD's tree, measuring it once."""
        if not head_list:
            head_list.append(at_head())
        return edit.part_of(head_list[0], command)

    return edit.join_baseline({
        command: _part_baseline_set(part, functools.partial(head_part, command))
        for command, part in part_dict.items()}, getattr(test_dict, "primary_name", ""))


def _part_baseline_set(test_dict: dict,
                       at_head: Callable[[], dict]) -> frozenset[str] | None:
    """Return _baseline_set()'s set for the run of one test command.

    Args:
        test_dict: The command's own result over the merged tree.
        at_head: () -> the same command's own result over HEAD's tree.

    Returns:
        As _baseline_set() says, for that command alone.
    """
    before_dict = at_head()
    ran_set = edit.ran_entry_set(before_dict)
    if before_dict["status"] == "failed" and edit.early_stop_run(before_dict):
        # HEAD's run is red and ended before every test was run: the check says so
        # whatever the merged tree's run says (_TEST_HEAD_STOP_REASON), and a test the
        # merged tree's run names that HEAD's does not is one the merged tree broke
        return edit.fail_entry_set(before_dict["failed"]) | edit.stop_entry_set()
    if test_dict["status"] == "passed":
        return ran_set
    if edit.no_test_run(before_dict):
        # Where tests ran on the merged tree, HEAD's tree is held as all passing
        return frozenset() if test_dict["status"] == "failed" else edit.NO_TEST_BASELINE
    if before_dict["status"] == "passed":
        return ran_set
    if before_dict["status"] != "failed" and getattr(before_dict, "is_ran_unread", False):
        # HEAD's run says nothing of how many tests it ran: the check says so
        # (edit.unrun_reason)
        return ran_set
    if before_dict["status"] != "failed":
        logger.info("the tests are left out of the check: they did not run over what "
                    "the branches were combined onto")
        return None
    if before_dict["failed"] and test_dict["status"] != "failed":
        return edit.fail_entry_set(before_dict["failed"]) | ran_set
    if before_dict["failed"] and test_dict["failed"]:
        return (edit.fail_entry_set(before_dict["failed"])
                & edit.fail_entry_set(test_dict["failed"])) | ran_set
    if getattr(before_dict, "is_name_short", False) or getattr(test_dict, "is_name_short", False):
        # A run that does not name each test it counts: the check says so
        # (edit.unrun_reason), and a test the merged tree's run names that HEAD's does
        # not is one the merged tree broke
        return edit.fail_entry_set(before_dict["failed"]) | ran_set
    logger.info("the tests are left out of the check: what the branches were combined "
                "onto is red, and its run or the merged tree's named no test to compare by")
    return None


def _test_reason(result_dict: dict | None, baseline_set: frozenset[str] | None) -> str:
    """Return what the tests say that the baseline does not, or "" when nothing.

    Only a failure the baseline does not hold counts, and a run that did not run, a run
    that found no test to run included where the run the baseline was read from found
    tests (edit.compare_part_list).

    Args:
        result_dict: run_test()'s return value over the repository as it stands, or
            None where there is no test command to run.
        baseline_set: _baseline_set()'s set. None leaves the tests out of the check, and
            nothing is said.

    Returns:
        One reason line for every test command of the run that has one
        (edit.compare_part_list), or "".
    """
    if baseline_set is None:
        return ""
    return "\n".join(reason for reason in (
        _part_test_reason(part, baseline_set, command, has_own_name)
        for command, part, has_own_name
        in edit.compare_part_list(result_dict, baseline_set)) if reason)


def _part_test_reason(result_dict: dict, baseline_set: frozenset[str], command: str,
                      has_own_name: bool) -> str:
    """Return _test_reason()'s line for the run of one test command.

    Args:
        result_dict: The command's own result.
        baseline_set: _baseline_set()'s set.
        command: The command, as edit.compare_part_list() names it.
        has_own_name: Whether the baseline holds failing tests of this command.

    Returns:
        One reason line, or "". Where HEAD's run ended before every test was run
        (edit.is_stop_before), any run is a reason: one that did not run or names a
        failing test the baseline does not hold says so, any other says HEAD's run
        stopped. A failing run that ended before every test was run
        (edit.early_stop_run) is a reason whatever tests it named. A run that passed,
        or failed with no test the baseline does not hold, is held to the tests that
        ran over HEAD's tree (edit.unrun_reason).
    """
    shown = edit.command_text(command)
    unrun = functools.partial(edit.unrun_reason, result_dict, baseline_set, command,
                              _TEST_UNRUN_REASON, _TEST_FEWER_RUN_REASON,
                              _TEST_UNREAD_REASON, _TEST_SHORT_REASON)
    head_stop_reason = _TEST_HEAD_STOP_REASON.format(command=shown) \
        if edit.is_stop_before(baseline_set, command) else ""
    if result_dict["status"] == "passed":
        return head_stop_reason or unrun()
    if result_dict["status"] != "failed":
        return _TEST_NO_RUN_REASON.format(command=shown, why=result_dict["failed_excerpt"])
    break_list = sorted(set(edit.fail_name_list(
        edit.fail_entry_set(result_dict["failed"]) - baseline_set)))
    if break_list:
        return _TEST_BREAK_REASON.format(command=shown, tests=list_text(break_list),
                                         why=result_dict["failed_excerpt"])
    if head_stop_reason:
        return head_stop_reason
    if edit.early_stop_run(result_dict):
        return _TEST_EARLY_STOP_REASON.format(
            command=shown, why=edit.early_stop_why(result_dict),
            tests=list_text(result_dict["failed"]) or _NO_TEST_NAME_TEXT)
    if has_own_name:
        # Everything red now is in the baseline, or this run named nothing and the two
        # states cannot be told apart
        return unrun()
    return _TEST_REASON.format(
        command=shown, why=result_dict["failed_excerpt"] or list_text(result_dict["failed"]))


def _is_head_short(baseline_set: frozenset[str], command: str) -> bool:
    """Say whether the baseline holds a command's run over HEAD's tree as short of names.

    Short of names is a run that does not name each test it counts: what
    edit.unrun_reason() gives _TEST_SHORT_REASON for, however the merged tree's run
    names its tests.

    Args:
        baseline_set: The baseline.
        command: The command, as edit.compare_part_list() names it.

    Returns:
        True when the baseline holds that command's run over HEAD's tree so.
    """
    # A run the tests that ran were read off, that names each test it counts
    full_name_run = edit.RunResult({"status": edit.TEST_STATUS_DICT[True], "failed": []})
    full_name_run.is_ran_hold = True
    return bool(edit.unrun_reason(full_name_run, baseline_set, command, "", "", "",
                                  HEAD_SHORT_BLOCK))


def _reason_list(project_dir: str, output_dir: str, before_dict: dict,
                 baseline_set: frozenset[str] | None,
                 exclude_list: list[str] | None, side_dict: dict | None = None,
                 work_dict: dict | None = None,
                 side_list: list[tuple] | None = None,
                 check_before: collections.Counter | None = None,
                 meet_list: list[str] | None = None,
                 marker_dict: dict | None = None,
                 no_path_list: list[str] | None = None,
                 stand_in_list: list[str] | None = None,
                 in_block_list: list[tuple[str, str, str | None]] | None = None
                 ) -> list[str]:
    """Return what is still wrong with the merged repository, in the model's words.

    The analysis is run again over what is in the working tree now: a reference settled
    since the merge is no longer counted. Nothing here calls a model. What was measured
    is turned into reasons by _unsettle_reason_list().

    Args:
        project_dir: The repository holding the merged change.
        output_dir: Where the analysis is written.
        before_dict: analyse()'s return value for the commit the union is written out
            against.
        baseline_set: _baseline_set()'s set. None leaves the tests out of the check.
        exclude_list: The extra names kept out of the analysis, as the build takes them.
        side_dict: The references that do not resolve on the branches taken one at a
            time, as prepare() read them.
        work_dict: The state, whose "unresolved" is set to how many references do not
            resolve in the analysis made again. None leaves it out.
        side_list: keep_list_of()'s list for the sides, by which what each of them
            added or changed is looked for in the union.
        check_before: edit.run_check() over HEAD's tree. None leaves the static check out.
        meet_list: meet_path_list()'s paths, each of which this stage has to have read.
            None leaves that out.
        marker_dict: assemble()'s "marker", the lines the tree may hold.
        no_path_list: assemble()'s "no_path".
        stand_in_list: assemble()'s "stand_in".
        in_block_list: in_block_read_list()'s entries, each of which this stage has
            to have read on its branch. None leaves that out.

    Returns:
        One line per reason, empty when there is nothing left.
    """
    marker_list = marker_path_list(project_dir, marker_dict)
    stand_in_now_list = _stand_in_now_list(project_dir, stand_in_list or [])
    check_line_list = check_reason(check_before)

    # An analysis that cannot be made is returned as a reason like any other
    try:
        path, _, project_name, _ = ensure_knowledge(project_dir, output_dir,
                                                    rebuild=True, with_doc=False,
                                                    exclude_list=exclude_list)
    except (ValueError, RuntimeError) as e:
        logger.warning("the analysis could not be made again (%s), so what the "
                       "references say now is not known", e)
        return _unsettle_reason_list(marker_list, check_line_list, [], [], {}, "", [], "",
                                     no_path_list or [], stand_in_now_list) \
            + [_ANALYSIS_REASON.format(why=e)]
    after_dict = analyse(path)
    if work_dict is not None:
        work_dict["unresolved"] = len(after_dict["no_resolve"])
    test_reason = _test_reason(edit.run_test(), baseline_set) \
        if baseline_set is not None else ""
    return _unsettle_reason_list(
        marker_list, check_line_list,
        broken_list_of(before_dict, after_dict, project_name, side_dict),
        side_list or [], after_dict["define"], project_name, meet_list or [],
        test_reason, no_path_list or [], stand_in_now_list, in_block_list)


def _unsettle_reason_list(marker_list: list[str], check_line_list: list[str],
                          broken_list: list[dict], keep_list: list[tuple], define_dict: dict,
                          project_name: str, meet_list: list[str],
                          test_reason: str, no_path_list: list[str],
                          stand_in_list: list[str],
                          in_block_list: list[tuple[str, str, str | None]] | None = None
                          ) -> list[str]:
    """Return what keeps a merged repository from being settled, from what was measured.

    Args:
        marker_list: marker_path_list() over the working tree.
        check_line_list: check_reason()'s lines.
        broken_list: broken_list_of()'s list, uncut.
        keep_list: keep_list_of()'s list for the sides.
        define_dict: analyse()'s "define" for the working tree.
        project_name: The directory name of PROJECT_DIR, for the path conversion.
        meet_list: meet_path_list()'s paths, each of which this run has to have read.
        test_reason: _test_reason() over the working tree.
        no_path_list: assemble()'s "no_path": what git said of the conflicts that name
            no file of the merged tree.
        stand_in_list: _stand_in_now_list()'s paths: the stand-in names of
            assemble()'s "stand_in" the working tree still holds.
        in_block_list: in_block_read_list()'s entries, each of which this run has
            to have read on its branch. None holds none.

    Returns:
        One line per reason, empty when the merge is settled.
    """
    reason_list = []
    if no_path_list:
        reason_list.append(_NO_PATH_REASON.format(messages=" ".join(no_path_list)))
    if marker_list:
        reason_list.append(_MARKER_REASON.format(paths=list_text(marker_list)))
    if stand_in_list:
        reason_list.append(_STAND_IN_REASON.format(paths=list_text(stand_in_list)))
    reason_list += check_line_list
    if broken_list:
        reason_list.append(_BROKEN_REASON.format(
            calls=list_text([f"{entry_dict['name']} in {entry_dict['file']}"
                             for entry_dict in broken_list])))
    reason_list += _drop_reason_list(keep_list, define_dict, project_name)
    reason_list += _no_read_reason_list(meet_list)
    reason_list += _no_branch_read_reason_list(in_block_list or [])
    if test_reason:
        reason_list.append(test_reason)
    return reason_list


def _drop_reason_list(keep_list: list[tuple], define_dict: dict,
                      project_name: str) -> list[str]:
    """Return what a side added or changed and the union defines nowhere.

    Args:
        keep_list: keep_list_of()'s list for the branches merged.
        define_dict: analyse()'s "define" for the merged state.
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        One line per name no file of the union defines, naming the files the side's
        own tip defines it in.
    """
    return [_DROP_REASON.format(
        name=name, branch=branch,
        file=list_text([knowledge_to_repo(file, project_name) for file in file_list]))
        for branch, keep_dict in keep_list
        for name, file_list in sorted(keep_dict.items()) if name not in define_dict]


def meet_path_list(merge_dict: dict, overlap_list: list[dict]) -> list[str]:
    """Return the files the sides both reached.

    They are what git reported as conflicted, and where a definition more than one
    side's change reaches is.

    Args:
        merge_dict: assemble()'s return value.
        overlap_list: overlap_entry_list()'s list, uncut.

    Returns:
        The repository-relative paths, sorted and without repeats.
    """
    return sorted({entry_dict["file"] for entry_dict in overlap_list}
                  | {path for entry_dict in merge_dict["branches"]
                     for path in entry_dict["conflicted"]})


def in_block_read_list(merge_dict: dict) -> list[tuple[str, str, str | None]]:
    """Return the files kept as they were against a side's change, each with that side.

    A file is one kept as it was where the change of the side to it was to be merged.

    Args:
        merge_dict: assemble()'s return value.

    Returns:
        [(the side's name, the file's path in the merged tree, the path the side
        holds the file at, None where that path cannot be told), ...] over the
        "in_block" of every branch, in the order of the branches. A file the side
        holds at no path is left out.
    """
    return [(entry_dict["branch"], path, side_path)
            for entry_dict in merge_dict["branches"]
            for path, side_path in sorted((entry_dict.get("in_block") or {}).items())
            if side_path != ""]


def _no_branch_read_reason_list(in_block_list: list[tuple[str, str, str | None]]
                                ) -> list[str]:
    """Return the files left without a side's change that this run has not read there.

    The files are those the side's change was not merged into, and reading one counts
    only on that side. What read_branch_file has recorded up to now is read, the turn
    that is running included.

    Args:
        in_block_list: in_block_read_list()'s entries.

    Returns:
        One line for each entry whose (side, path on the side) read_branch_file has
        not read, one for each entry whose path on the side cannot be told,
        whatever was read, and _side_mode_reason_list()'s lines.
    """
    read_set = _STAGE_DICT["branch_read_set"]
    reason_list = []
    for branch, path, side_path in in_block_list:
        if side_path is None:
            reason_list.append(_NO_SIDE_PATH_REASON.format(path=path, branch=branch))
        elif (branch, side_path) not in read_set:
            reason_list.append(_NO_BRANCH_READ_REASON.format(
                path=path, branch=branch, side_path=side_path))
    return reason_list + _side_mode_reason_list(in_block_list)


def _side_mode_reason_list(in_block_list: list[tuple[str, str, str | None]]
                           ) -> list[str]:
    """Return the files a side changed the mode of that the working tree holds otherwise.

    The working tree holds such a file under another mode. A side changed the mode where
    it holds the file under a mode none of the versions it was made from holds it under:
    the file is followed from the side back to where the side and the starting commit
    parted (_fork_path_dict), a file the version there does not hold being read as one
    that is not executable.

    Args:
        in_block_list: in_block_read_list()'s entries.

    Returns:
        One line for each entry whose file the working tree holds as a file or a
        symbolic link under a mode other than the one the side changed it to.

    Raises:
        RuntimeError: When a tree could not be read, or git could not say which
            commits a side and the starting commit share.
    """
    commit_dict = _STAGE_DICT["commit_dict"]
    entry_list = [(branch, path, side_path) for branch, path, side_path in in_block_list
                  if side_path and branch in commit_dict]
    if not entry_list:
        return []
    start = commit_dict[""]
    entry_by_commit_dict: dict[str, dict] = {}
    fork_by_commit_dict: dict[str, dict] = {}

    def entry_dict_of(commit: str) -> dict:
        """Return _file_entry_dict()'s mapping for a commit, reading it once."""
        if commit not in entry_by_commit_dict:
            entry_by_commit_dict[commit] = _file_entry_dict(config.PROJECT_DIR, commit)
        return entry_by_commit_dict[commit]

    reason_list = []
    for branch, path, side_path in entry_list:
        commit = commit_dict[branch]
        if commit not in fork_by_commit_dict:
            fork_by_commit_dict[commit] = _fork_path_dict(config.PROJECT_DIR, commit, start)
        side_mode = entry_dict_of(commit).get(side_path, (b"",))[0]
        # Where the two share no commit, the version is the starting commit's own
        fork_mode_set = {
            entry_dict_of(fork or start).get(fork_path if fork else start_path,
                                             (b"100644",))[0]
            for fork, fork_path, start_path in fork_by_commit_dict[commit].get(side_path, [])}
        tree_mode = _tree_mode(path)
        if side_mode in _MODE_KIND_DICT and tree_mode \
                and side_mode not in fork_mode_set \
                and tree_mode != side_mode:
            reason_list.append(_SIDE_MODE_REASON.format(
                branch=branch, side_path=side_path, side_kind=_MODE_KIND_DICT[side_mode],
                path=path, tree_kind=_MODE_KIND_DICT[tree_mode]))
    return reason_list


def _tree_mode(path: str) -> bytes:
    """Return the mode git would hold a file of the working tree under.

    Args:
        path: A repository-relative path.

    Returns:
        b"120000" for a symbolic link, b"100755" for a file its owner may run,
        b"100644" for any other file, and b"" for anything else.
    """
    try:
        mode = os.lstat(os.path.join(config.PROJECT_DIR, path)).st_mode
    except (OSError, ValueError):
        return b""
    if stat.S_ISLNK(mode):
        return _LINK_MODE
    if not stat.S_ISREG(mode):
        return b""
    return b"100755" if mode & stat.S_IXUSR else b"100644"


def _no_read_reason_list(meet_list: list[str]) -> list[str]:
    """Return the files the sides both reached that nothing in this run has read.

    What the reading tools have recorded up to now is read, the turn that is running
    included. A path the working tree does not hold as a file is named with the
    branches read_branch_file reads it on (_holder_list), and one no branch holds
    as a file either is named as one no tool reads. A path neither the working tree
    nor a branch holds anything at is left out.

    Args:
        meet_list: meet_path_list()'s paths.

    Returns:
        One line for the files of the working tree, one for each path a branch holds
        and one for the paths no tool reads, or [] when every path was read.
    """
    read_dict = _STAGE_DICT["read_dict"]
    here_list, branch_line_list, no_tool_list = [], [], []
    for path in meet_list:
        if path in read_dict:
            continue
        if _is_tree_file(path):
            here_list.append(path)
            continue
        holder_list = _holder_list(path)
        if holder_list:
            branch_line_list.append(_NO_READ_BRANCH_REASON.format(
                path=path, branches=", ".join(holder_list)))
        elif _has_entry(path):
            no_tool_list.append(path)
    reason_list = [_NO_READ_REASON.format(paths=list_text(here_list))] if here_list else []
    reason_list += branch_line_list
    if no_tool_list:
        reason_list.append(_NO_TOOL_READ_REASON.format(paths=list_text(no_tool_list)))
    return reason_list


def _is_tree_file(path: str) -> bool:
    """Say whether the working tree holds a file at a path.

    Args:
        path: A repository-relative path.

    Returns:
        True for a file under PROJECT_DIR, False for a path outside it as well.
    """
    try:
        return os.path.isfile(resolve_in_project(path))
    except ValueError:
        return False


def _has_entry(path: str) -> bool:
    """Say whether the working tree or a branch being merged holds anything at a path.

    Args:
        path: A repository-relative path.

    Returns:
        True for a path the working tree holds, whatever is there, and for one the
        starting commit or a branch holds a file, a link, a directory or a submodule
        at.
    """
    return os.path.lexists(os.path.join(config.PROJECT_DIR, path)) or any(
        _has_path(config.PROJECT_DIR, commit, path)
        for commit in _STAGE_DICT["commit_dict"].values())


def _holder_list(path: str) -> list[str]:
    """Return the branches being merged that hold a file at a path.

    Args:
        path: A repository-relative path.

    Returns:
        The names read_branch_file takes, in the order the stage holds them, the
        starting commit as _START_BRANCH_TEXT.
    """
    holder_list = []
    for branch, commit in _STAGE_DICT["commit_dict"].items():
        proc = gitcmd.run(config.PROJECT_DIR, "cat-file", "-t", f"{commit}:{path}")
        if proc.returncode == 0 and proc.stdout.strip() == "blob":
            holder_list.append(branch or _START_BRANCH_TEXT)
    return holder_list


def _change_text(entry_dict: dict) -> str:
    """Say what this stage changed in one file, by definition where the lines fall in one.

    Args:
        entry_dict: One "changed" entry.

    Returns:
        E.g. "changed total, summary in report.py", "changed report.py +3 -1" for lines
        outside every definition, "changed x.py (new file)" for a file that did not exist.
    """
    if entry_dict["new"]:
        return f"changed {entry_dict['path']} (new file)"
    if entry_dict["defs"]:
        return f"changed {', '.join(entry_dict['defs'])} in {entry_dict['path']}"
    return f"changed {entry_dict['path']} +{entry_dict['insertions']} -{entry_dict['deletions']}"


def _this_stage_note(work_dict: dict, path_list: list[str]) -> str:
    """Say what this stage has read and changed in the files a start line names.

    Args:
        work_dict: The state, its "read" and "changed" as measured on this turn.
        path_list: The paths the line names, in the order it names them.

    Returns:
        The notes joined into one, _NO_REACH_TEXT when this stage has neither read nor
        changed any of them.
    """
    read_dict = {entry_dict["path"]: entry_dict["turn"] for entry_dict in work_dict["read"]}
    change_by_path_dict = {entry_dict["path"]: entry_dict for entry_dict in work_dict["changed"]}
    note_list = []
    for path in dict.fromkeys(path_list):
        if path in read_dict:
            note_list.append(f"read {path} on turn {read_dict[path]}")
        if path in change_by_path_dict:
            note_list.append(_change_text(change_by_path_dict[path]))
    return "; ".join(note_list) or _NO_REACH_TEXT


def _branch_line_list(work_dict: dict) -> list[str]:
    """Render what each merged branch was asked for, did and left unsettled.

    Args:
        work_dict: The state.

    Returns:
        The lines, starting with the commit the branches were merged onto.
    """
    line_list = [f"Merged onto {gitcmd.short(work_dict['base'])}:"]
    for entry_dict in work_dict["branches"]:
        line_list.append(f"  {entry_dict['branch']}")
        for label, record_list in (("asked", entry_dict["said"]), ("did", entry_dict["did"])):
            line_list += [f"    {label}: {line}" for line in record_list]
        if entry_dict["conflicted"]:
            line_list.append(f"    git could not settle: "
                             f"{list_text(entry_dict['conflicted'])}")
        unmark_list = [path for path in entry_dict["conflicted"]
                       if path in work_dict.get("unmarked", ())]
        if unmark_list:
            line_list.append(f"    {_UNMARKED_NOTE}: {list_text(unmark_list)}")
        if entry_dict.get("in_block"):
            line_list.append(f"    {_IN_BLOCK_NOTE}: {list_text(entry_dict['in_block'])}")
    return line_list


def _measure_line_list(work_dict: dict) -> list[str]:
    """Render the markers, the files this stage changed and the files it read.

    Args:
        work_dict: The state, its "markers", "changed" and "read" as measured on this turn.

    Returns:
        The lines of the three sections, each after an empty line.
    """
    line_list = ["", _MARKER_NOW if work_dict["markers"] else _NO_MARKER_NOW]
    line_list += [f"  {path}" for path in work_dict["markers"]]

    line_list += ["", _CHANGE_NOW if work_dict["changed"] else _NO_CHANGE_NOW]
    line_list += [f"  {entry_dict['path']}  "
                  + ("new file" if entry_dict["new"]
                     else f"+{entry_dict['insertions']} -{entry_dict['deletions']}")
                  + (f" ({', '.join(entry_dict['defs'])})" if entry_dict["defs"] else "")
                  for entry_dict in work_dict["changed"]]

    line_list += ["", _READ_NOW if work_dict["read"] else _NO_READ_NOW]
    line_list += [f"  {entry_dict['path']}  on turn {entry_dict['turn']}"
                  for entry_dict in work_dict["read"]]
    return line_list


def _broken_start_line_list(work_dict: dict) -> list[str]:
    """Render the references the merge broke, as they were measured before the first turn.

    Args:
        work_dict: The state.

    Returns:
        The lines of the section, after an empty line.
    """
    line_list = ["", _BROKEN_AT_START if work_dict["broken"] else _NO_BROKEN]
    for entry_dict in work_dict["broken"]:
        offer_text = list_text(entry_dict["now_defines"]) or "nothing"
        line_list.append(f"  {entry_dict['name']} in {entry_dict['file']} — "
                         f"{entry_dict['defined_in']} now defines {offer_text}")
    line_list.append(_NO_RESOLVE_NOTE)
    return line_list


def _overlap_start_line_list(work_dict: dict) -> list[str]:
    """Render the definitions more than one side's change reaches, at the start.

    They are as they were measured before the first turn.

    Args:
        work_dict: The state.

    Returns:
        The lines of the section, after an empty line.
    """
    line_list = ["", _OVERLAP_AT_START if work_dict["overlap"] else _NO_OVERLAP]
    for entry_dict in work_dict["overlap"]:
        if entry_dict["kind"] == "both":
            line_list.append(_OVERLAP_BOTH.format(
                name=entry_dict["name"], file=entry_dict["file"],
                branches=list_text(entry_dict["branches"])))
        else:
            line_list.append(_OVERLAP_USE.format(
                name=entry_dict["name"], file=entry_dict["file"],
                changed_by=entry_dict["changed_by"], used_in=entry_dict["used_in"],
                lines=list_text([str(line) for line in entry_dict["lines"]]) or "?",
                used_by=list_text(entry_dict["used_by"])))
    return line_list


def _reach_line_list(work_dict: dict) -> list[str]:
    """Render what this stage has read and changed where the start block points.

    The files are those of each broken call and overlapping definition, and the last
    section says how many references do not resolve in all.

    Args:
        work_dict: The state, its "read" and "changed" as measured on this turn.

    Returns:
        The lines of the sections, each after an empty line. The first is left out when
        the start block names neither a broken call nor an overlapping definition.
    """
    reach_list = [_REACH_LINE.format(
        name=entry_dict["name"], file=entry_dict["file"],
        notes=_this_stage_note(work_dict, [entry_dict["file"], entry_dict["defined_in"]]))
        for entry_dict in work_dict["broken"]]
    reach_list += [_REACH_LINE.format(
        name=entry_dict["name"], file=entry_dict["file"],
        notes=_this_stage_note(
            work_dict, [entry_dict["file"]] if entry_dict["kind"] == "both"
            else [entry_dict["file"], entry_dict["used_in"]]))
        for entry_dict in work_dict["overlap"]]
    line_list = ["", _REACH_NOW, *reach_list] if reach_list else []
    return line_list + ["", _NO_RESOLVE_NOW.format(count=work_dict["unresolved"])]


def _test_start_line_list(work_dict: dict) -> list[str]:
    """Render the tests that were failing at the start, or nothing without a test command.

    Args:
        work_dict: The state.

    Returns:
        The lines of the section, after an empty line, or [].
    """
    if not edit.test_can_run():
        return []
    if work_dict["already_failing"] is None:
        return ["", _TEST_NO_COMPARE]
    part_line_list = ["", _TEST_NO_COMPARE_PART.format(
        commands=_command_list_text(_STAGE_DICT["not_compared"]))] \
        if _STAGE_DICT["not_compared"] else []
    if not work_dict["already_failing"]:
        return part_line_list
    if _STAGE_DICT["head_stop"]:
        title = _ALREADY_RED_AT_STOP.format(
            commands=_command_list_text(_STAGE_DICT["head_stop"]))
    elif _STAGE_DICT["head_short"]:
        title = _ALREADY_RED_AT_SHORT.format(
            commands=_command_list_text(_STAGE_DICT["head_short"]))
    else:
        title = _ALREADY_RED_AT_HEAD if _STAGE_DICT["head_baseline"] else _ALREADY_RED
    return part_line_list + [
        "", title, *[f"  {name}" for name in work_dict["already_failing"]]]


def _command_list_text(command_list: list[str]) -> str:
    """Name test commands for a line of the start block.

    Examples:
        ["pytest -q", "npm test"] -> "`pytest -q`, `npm test`"

    Args:
        command_list: The commands, as edit.compare_part_list() names them.

    Returns:
        Each command as it runs (edit.command_text) in backquotes, joined with ", ".
    """
    return ", ".join(f"`{edit.command_text(command)}`" for command in command_list)


def _test_line_list(work_dict: dict) -> list[str]:
    """Render what the whole test suite last said against the start.

    Nothing is rendered when the tests are not part of the check.

    Args:
        work_dict: The state.

    Returns:
        The lines of the section, after an empty line, or []. What it said is headed by
        _TEST_HEAD_BLOCK where the run over HEAD's tree keeps any merge from being settled
        (head_block_list), else by _TEST_BREAK.
    """
    if not edit.test_can_run() or work_dict["already_failing"] is None:
        return []
    when = _TEST_ON_TURN.format(turn=work_dict["tests_at"]) if work_dict["tests_at"] \
        else _TEST_BEFORE_FIRST_TURN
    title = (_TEST_HEAD_BLOCK if head_block_list() else _TEST_BREAK) if work_dict["tests"] \
        else _NO_TEST_BREAK
    line_list = ["", title.format(when=when)]
    line_list += [f"  {work_dict['tests']}"] if work_dict["tests"] else []
    return line_list


def _check_start_line_list(work_dict: dict) -> list[str]:
    """Render what the static check found before the first turn.

    Args:
        work_dict: The state.

    Returns:
        The lines of the section, after an empty line, or [] when it found nothing.
    """
    if not work_dict.get("checks"):
        return []
    return ["", _STATIC_CHECK_TITLE, *[f"  {reason}" for reason in work_dict["checks"]]]


def _check_line_list(work_dict: dict) -> list[str]:
    """Render what the last check of the whole repository found.

    Args:
        work_dict: The state.

    Returns:
        The lines of the section, after an empty line.
    """
    if not work_dict["checked_at"]:
        return ["", _NO_CHECK_YET]
    return ["", _CHECK_AT.format(turn=work_dict["checked_at"]),
            *([f"  {reason}" for reason in work_dict["left"]] or [_CHECK_FIND_NOTHING])]


def _decision_line_list(decision_list: list[dict]) -> list[str]:
    """Render recorded decisions, one indented line each.

    Args:
        decision_list: [{"what", "why"}, ...].

    Returns:
        One line per entry.
    """
    return [f"  {entry_dict['what']} — {entry_dict['why']}" for entry_dict in decision_list]


def _earlier_line_list(work_dict: dict) -> list[str]:
    """Render what the merging run this one carries on from recorded.

    Args:
        work_dict: The state.

    Returns:
        The lines of the section, after an empty line, or [] for a run that carries on
        from none.
    """
    earlier_dict = work_dict["earlier"]
    if not earlier_dict:
        return []
    return ["", _EARLIER_TITLE.format(
        branch=earlier_dict["branch"], ended=earlier_dict["ended"],
        why=_EARLIER_WHY.format(why=earlier_dict["why"]) if earlier_dict["why"] else ""),
        *(_decision_line_list(earlier_dict["decisions"]) or [_NO_EARLIER_DECISION])]


def _record_line_list(work_dict: dict) -> list[str]:
    """Render what this run has recorded.

    Args:
        work_dict: The state.

    Returns:
        The lines of the section, after an empty line.
    """
    return ["", _DECISION_TITLE if work_dict["decisions"] else _NO_DECISION,
            *_decision_line_list(work_dict["decisions"])]


def start_text(work_dict: dict) -> str:
    """Render what was measured before the first turn: the block placed after the request.

    Args:
        work_dict: build_work_dict()'s return value, with "checks" set.

    Returns:
        The block, its heading first: the branches, the calls the merge broke, the
        definitions more than one side reaches, the tests that were failing, what the
        static check found and what an earlier merging run recorded.
    """
    return "\n".join([_START_TITLE, ""]
                     + _branch_line_list(work_dict)
                     + _broken_start_line_list(work_dict)
                     + _overlap_start_line_list(work_dict)
                     + _test_start_line_list(work_dict)
                     + _check_start_line_list(work_dict)
                     + _earlier_line_list(work_dict))


def _state_line_list(work_dict: dict, turn: int, max_turn: int) -> list[str]:
    """Render the state block for the turn about to run.

    Args:
        work_dict: build_work_dict()'s return value, as the turn before this one left it.
        turn: Which turn is about to run (1-based).
        max_turn: The stage's turn limit.

    Returns:
        The lines of the block, the heading first: what is measured again on every
        turn, what the tests and the check last said, and what was recorded.
    """
    return ([_STATE_TITLE.format(turn=turn, max_turn=max_turn)]
            + _measure_line_list(work_dict)
            + _reach_line_list(work_dict)
            + _test_line_list(work_dict)
            + _check_line_list(work_dict)
            + _record_line_list(work_dict))


def change_entry_list(project_dir: str, union: str, range_dict: dict | None = None,
                      project_name: str = "") -> list[dict]:
    """Return what the working tree changes against the union, tracked and new alike.

    Args:
        project_dir: The repository holding the merged change.
        union: The commit holding the union, as assemble() returns it.
        range_dict: analyse()'s "range" for the union, which the changed lines are
            mapped onto. None maps nothing.
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        [{"path", "insertions", "deletions", "new", "defs"}, ...], the tracked files in
        git's order and then the files git does not track, "new" True for those and
        their counts 0. "defs" is the sorted names of the union's definitions the
        changed lines fall in, empty for lines outside every definition, for a file git
        cannot read in the working tree (whose counts are gitcmd.UNREAD_COUNT) and for
        every file when git gives no diff, which is logged.
    """
    def_dict: dict = {}
    if range_dict is not None:
        try:
            change_dict = change_dict_of(project_dir, union, "")
        except RuntimeError as e:
            gitcmd.warn_once(("defs", project_dir, str(e)),
                             "the changed lines of %s are not named by definition, as "
                             "git gave no diff of them", project_dir, detail=str(e),
                             log=logger)
            change_dict = {"add": [], "delete": [], "line": {}}
        side_dict = side_dict_of(change_dict, range_dict, project_name)
        def_dict = {knowledge_to_repo(file, project_name): sorted(name_set - {""})
                    for file, name_set in side_dict.items()}
    entry_list = [dict(entry_dict, new=False, defs=def_dict.get(entry_dict["path"], []))
                  for entry_dict in gitcmd.numstat_list(project_dir, union)]
    entry_list += [{"path": path, "insertions": 0, "deletions": 0, "new": True, "defs": []}
                   for path in gitcmd.new_file_list(project_dir)]
    return entry_list


def make_state(project_dir: str, union: str, work_dict: dict,
               check: Callable[[], list[str]],
               baseline_set: frozenset[str] | None = None,
               range_dict: dict | None = None, project_name: str = "",
               marker_dict: dict | None = None,
               uncut_dict: dict | None = None
               ) -> tuple[Callable[[int, int], str], Callable[[], list[str]]]:
    """Build the state block's turn_note() and the check that keeps the block current.

    The block is the state the machine has measured. Only the markers, this stage's own
    changes and what it has read are measured again on every turn, which is
    marker_path_list()'s reading of the working tree, two diffs against the union and
    the reading tools' own record; what the references say is carried with the turn it
    was measured on and refreshed by the check. What the tests say follows the last run
    of the whole suite, whichever side ran it, with the turn it ran on.

    Args:
        project_dir: The repository holding the merged change.
        union: The commit holding the union, as assemble() returns it. What the working
            tree changes against it is what this stage has done.
        work_dict: build_work_dict()'s return value, which both return values write to.
        check: () -> [reason, ...], the machine's own reading of what is still wrong.
        baseline_set: _baseline_set()'s set, which the last whole run of the suite is
            read against. None leaves the tests out.
        range_dict: analyse()'s "range" for the union, by which what this stage changed
            is named by definition. None names files alone.
        project_name: The directory name of PROJECT_DIR, for the path conversion.
        marker_dict: assemble()'s "marker", the lines the tree may hold.
        uncut_dict: The lists build_work_dict() was given and cut, as they were before
            the cut: {"broken", "overlap", "branches" (assemble()'s)}. Whether the
            state is complete is read off them. None reads the state's own lists.

    Returns:
        (turn_note(turn, max_turn) -> str, a check that records what it found in the
        state before returning it). A turn past max_turn is headed and kept as max_turn.
    """
    uncut_dict = uncut_dict or work_dict
    _STAGE_DICT["test_run"] = (edit.last_whole_run() or {"n": 0})["n"]
    deadline = make_deadline_note(config.INTEGRATE_DEADLINE_TURN, _DEADLINE_WHAT)

    def turn_note(turn: int, max_turn: int) -> str:
        """Measure the state again and render the block shown before one turn."""
        set_turn(min(turn, max_turn))
        limit = config.INTEGRATE_WORK_MAX_ITEM
        marker_list = marker_path_list(project_dir, marker_dict)
        change_list = change_entry_list(project_dir, union, range_dict, project_name)
        work_dict["markers"] = marker_list[:limit]
        work_dict["changed"] = change_list[:limit]
        work_dict["read"] = read_list()[:limit]
        work_dict["decisions"] = decision_list()
        # Whether the cut left anything out is worked out again every turn
        work_dict["complete"] = is_complete(
            {**work_dict, "changed": change_list, "read": read_list()}, marker_list,
            uncut_dict["broken"], uncut_dict["overlap"], uncut_dict["branches"])
        run_dict = edit.last_whole_run()
        if baseline_set is not None and run_dict and run_dict["n"] != _STAGE_DICT["test_run"]:
            _STAGE_DICT["test_run"] = run_dict["n"]
            work_dict["tests"] = _test_reason(run_dict["result"], baseline_set)
            work_dict["tests_at"] = turn - 1
        line_list = _state_line_list(work_dict, min(turn, max_turn), max_turn)
        note = deadline(turn, max_turn)
        return "\n".join(line_list + (["", note] if note else []))

    def check_and_keep() -> list[str]:
        """Run the check and record what it found in the state."""
        work_dict["left"] = check()
        work_dict["checked_at"] = _STAGE_DICT["turn"]
        # What was recorded on this turn, the one that called COMPLETE() included
        work_dict["decisions"] = decision_list()
        return work_dict["left"]

    return turn_note, check_and_keep


def end_of(result_dict: dict) -> str:
    """Say how a merging run ended, as one of the END_* values.

    A COMPLETE() the check refused every time it was allowed to is accepted as it is: a
    run that ended with COMPLETE() while the check still finds something is one whose
    refusals were used up. What a stage adds to its "left" besides what the check found
    is not read here.

    Args:
        result_dict: The return value of a stage's run, run_integration()'s among them,
            its "work" holding under "left" what the check found the last time it was
            made.

    Returns:
        END_COMPLETE, END_REFUSAL_OUT, END_TURN_OUT or END_ERROR, or "" for a run that
        called no model.
    """
    if not result_dict["called"]:
        return ""
    if result_dict["outcome"] == COMPLETE_KEY:
        return END_REFUSAL_OUT if result_dict["work"]["left"] else END_COMPLETE
    if result_dict["error"]:
        return END_ERROR
    return END_TURN_OUT


def make_complete_guard(check: Callable[[], list[str]], refusal_max: int | None = None,
                        refusal_text: str = _COMPLETE_REFUSAL) -> Callable[[str], str]:
    """Build the guard that sends the call ending the run back while work is left.

    The call is sent back at most refusal_max times in one run.

    Args:
        check: () -> [reason, ...], the machine's own reading of what is still wrong.
        refusal_max: How many times the call is sent back. None reads
            config.INTEGRATE_MAX_REFUSAL.
        refusal_text: What is sent back, with {call}, {reasons} and {left_text}.

    Returns:
        on_complete(answer) -> str for RLM.run(): "" accepts, text refuses.
    """
    if refusal_max is None:
        refusal_max = config.INTEGRATE_MAX_REFUSAL

    def refusal(answer: object, refused_count: int) -> str:
        """Return the text sending the call back, or "" when nothing is left."""
        reason_list = check()
        if not reason_list:
            return ""
        left_count = refusal_max - refused_count - 1
        logger.info("%s() refused (%d of %d): %d thing(s) left to settle",
                    END_CALL_DICT[COMPLETE_KEY], refused_count + 1, refusal_max, len(reason_list))
        if refusal_max == 1:
            left_text = _REFUSAL_LEFT_NONE
        elif left_count == 0:
            left_text = _REFUSAL_LEFT_LAST
        else:
            left_text = _REFUSAL_LEFT_MORE.format(count=left_count)
        return refusal_text.format(call=END_CALL_DICT[COMPLETE_KEY],
                                   reasons="\n".join(reason_list), left_text=left_text)

    return refuse_up_to(refusal, refusal_max)


# == One run of the merging stage ========================


def was_on(project_dir: str) -> str:
    """Return the branch a repository is on, or its commit when it is on none.

    It is read before the union is put in place.

    Args:
        project_dir: The repository to read.

    Returns:
        The branch name, or the commit, or "" when neither could be read.
    """
    proc = gitcmd.run(project_dir, "symbolic-ref", "--quiet", "--short", "HEAD")
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    return gitcmd.head_commit(project_dir)


def _require_merge_tree(project_dir: str) -> None:
    """Refuse a git too old to write the trees this stage merges with.

    Args:
        project_dir: The repository this stage runs in.

    Raises:
        RuntimeError: When git is older than _MERGE_TREE_VERSION. Nothing is refused
            when gitcmd.version() returns ().
    """
    machine_version = gitcmd.version(project_dir)
    if machine_version and machine_version < _MERGE_TREE_VERSION:
        raise RuntimeError(
            f"git {'.'.join(str(part) for part in _MERGE_TREE_VERSION)} or later is needed "
            f"for merge-tree --write-tree, and this machine has "
            f"{'.'.join(str(part) for part in machine_version)}")


def _analyse_head(project_dir: str, output_dir: str,
                  exclude_list: list[str] | None) -> dict:
    """Make the analysis of what is checked out again and read it.

    Args:
        project_dir: The repository, checked out at the commit to analyse.
        output_dir: Where the analysis is written.
        exclude_list: The extra names kept out of the analysis.

    Returns:
        analyse()'s return value over that analysis.

    Raises:
        RuntimeError: When the analysis cannot be made.
    """
    path, _, _, _ = ensure_knowledge(project_dir, output_dir, rebuild=True,
                                     with_doc=False, exclude_list=exclude_list)
    return analyse(path)


def union_side_list(project_dir: str, commit: str, is_union: bool = True) -> list[str]:
    """Return the commits a union an earlier merge left unsettled was merged from.

    Args:
        project_dir: The repository to read.
        commit: The tip of the union.
        is_union: True takes commit for a union whether or not it carries trailers.

    Returns:
        _open_union()'s commits the union was merged from, which are [commit] itself
        when the merge under it is not replaced by its parents.

    Raises:
        RuntimeError: When the parents or the record of a commit could not be read.
    """
    return _open_union(project_dir, commit, is_union)[0]


def _open_union(project_dir: str, commit: str, is_union: bool = True
                ) -> tuple[list[str], list[str]]:
    """Replace a union an earlier merge left unsettled by the commits it was merged from.

    The commits of the stages after a merge are passed over first. The merge commit
    under them is replaced by its parents other than the first, and each of those whose
    merge is one this tool wrote (one carrying its record) in the same way: an unsettled
    union made from an unsettled union reaches the branches under both. A merge a
    merging run wrote and settled (is_settle_merge) is not replaced: it is one commit,
    its tip, with the commits of the stages on it.

    Args:
        project_dir: The repository to read.
        commit: The tip of the union.
        is_union: True takes commit for a union whether or not it carries a record.

    Returns:
        (the commits, the first parent of every merge replaced, which is the commit
        that merge was written against). ([commit], []) when the merge under commit
        is not replaced: it is not a merge commit, a merging run settled it, or
        (is_union False) it carries no record of this tool's.

    Raises:
        RuntimeError: When the parents or the record of a commit could not be read.
    """
    merge = below_stage(project_dir, commit)
    proc = gitcmd.run(project_dir, "rev-list", "--parents", "-n", "1", merge)
    if proc.returncode != 0:
        raise RuntimeError(f"could not read the parents of {gitcmd.short(merge)}: "
                           f"{(proc.stderr or proc.stdout).strip()}")
    parent_list = proc.stdout.split()[1:]
    if len(parent_list) < 2:
        return [commit], []
    has_record = bool(own_trailer_list(project_dir, merge))
    if is_settle_merge(project_dir, merge) or not (is_union or has_record):
        return [commit], []
    side_list: list[str] = []
    onto_list = [parent_list[0]]
    for parent in parent_list[1:]:
        more_side_list, more_onto_list = _open_union(project_dir, parent, is_union=False)
        side_list += more_side_list
        onto_list += more_onto_list
    return side_list, onto_list


def is_settle_merge(project_dir: str, commit: str) -> bool:
    """Say whether one commit is a merge a merging run wrote and settled.

    Args:
        project_dir: The repository to read.
        commit: The commit to read, and no commit it reaches.

    Returns:
        True when all of these hold: the commit's subject starts with MERGE_PREFIX and
        ": "; the commit has a second parent; it records SETTLE_VERDICT; and it records
        exit status 0 or none. False as well for a commit that cannot be read.
    """
    try:
        text_list = message_list(project_dir, "-1", commit)
    except RuntimeError:
        return False
    message = text_list[0] if text_list else ""
    line_list = record_line_list(message)
    return (message.startswith(f"{MERGE_PREFIX}: ")
            and trailer_value(line_list, VERDICT_TRAILER) == SETTLE_VERDICT
            and trailer_value(line_list, EXIT_STATUS_TRAILER) in ("", "0")
            and gitcmd.run(project_dir, "rev-parse", "--verify", "--quiet",
                           f"{commit}^2").returncode == 0)


def own_trailer_list(project_dir: str, commit: str) -> list[str]:
    """Return the record lines one commit carries.

    Args:
        project_dir: The repository to read.
        commit: The commit to read, and no commit it reaches.

    Returns:
        record_line_list()'s lines for the commit's message.

    Raises:
        RuntimeError: When the commit's message could not be read.
    """
    try:
        text_list = message_list(project_dir, "-1", commit)
    except RuntimeError as e:
        raise RuntimeError(
            f"could not read the trailers of {gitcmd.short(commit)}: {e}") from e
    return record_line_list(text_list[0]) if text_list else []


def _side_analysis(project_dir: str, output_dir: str, base: str, branch_list: list[str],
                   exclude_list: list[str] | None, resume: str = "",
                   start_by_commit_dict: dict | None = None) -> tuple[dict, dict, dict]:
    """Return what the branches taken one at a time leave unresolved and define.

    Each branch is checked out and analysed, and HEAD is put back at base afterwards.
    The references are left out of the references the merge broke: a reference that does
    not resolve on the branch that wrote it, and one whose key that branch moved by
    moving or renaming a file. A branch that is a union an earlier merge left unsettled
    (resume, and any other assemble() reads as one) is not one side: its references are
    read on the commits it was merged from (earlier_side_list), and what that merge
    broke is not left out. The names each side defines are what its change is looked for
    by in the union, and those of the commits such a union was merged from are kept
    under side_name()'s names. The commit a side starts from is analysed as well where
    it is not base (_side_start_dict).

    Args:
        project_dir: The repository, checked out at the commit the branches start from.
        output_dir: Where the analysis is written.
        base: The commit to come back to when every branch has been read.
        branch_list: The branch names.
        exclude_list: The extra names kept out of the analysis.
        resume: The branch among branch_list an earlier merge left, or "".
        start_by_commit_dict: As _side_start_dict() takes it. None starts an empty one.

    Returns:
        (analyse()'s "no_resolve" of every side put together by add_no_resolve(),
        {side: analyse()'s "define" at that side}, {side: _side_start_dict()'s value}
        for the sides that do not start from base). All are empty without branches
        or output_dir. The analysis left in output_dir is that of the last branch's
        tip.
    """
    if not branch_list or not output_dir:
        return {}, {}, {}
    start_by_commit_dict = {} if start_by_commit_dict is None else start_by_commit_dict
    no_resolve_dict: dict = {}
    define_by_branch_dict: dict = {}
    start_by_branch_dict: dict = {}
    for branch in branch_list:
        tip = commit_of(project_dir, branch)
        side_commit_list = earlier_side_list(project_dir, tip, is_union=branch == resume)
        for name, commit in [(side_name(commit), commit) for commit in side_commit_list] \
                or [(branch, tip)]:
            start_dict = _side_start_dict(project_dir, output_dir, base, commit,
                                          exclude_list, start_by_commit_dict)
            if start_dict:
                start_by_branch_dict[name] = start_dict
        for commit in side_commit_list:
            _move_head(project_dir, commit)
            side_dict = _analyse_head(project_dir, output_dir, exclude_list)
            add_no_resolve(no_resolve_dict, side_dict["no_resolve"])
            define_by_branch_dict[side_name(commit)] = side_dict["define"]
        # A tip is analysed after the commits it was merged from
        _move_head(project_dir, tip)
        tip_dict = _analyse_head(project_dir, output_dir, exclude_list)
        define_by_branch_dict[branch] = tip_dict["define"]
        if not side_commit_list:
            add_no_resolve(no_resolve_dict, tip_dict["no_resolve"])
    _move_head(project_dir, base)
    logger.info("read %d branch(es) on their own, which carry %d reference(s) that do "
                "not resolve there either", len(branch_list), len(no_resolve_dict))
    return no_resolve_dict, define_by_branch_dict, start_by_branch_dict


def _side_start_dict(project_dir: str, output_dir: str, base: str, commit: str,
                     exclude_list: list[str] | None, by_commit_dict: dict) -> dict | None:
    """Return the analysis of the commit a side starts from, where that is not base.

    The commit is the one base and the side both reach. It is checked out and analysed
    once for every side that starts from it, and HEAD is left there.

    Args:
        project_dir: The repository.
        output_dir: Where the analysis is written.
        base: The commit every branch is merged onto.
        commit: The side's commit.
        exclude_list: The extra names kept out of the analysis.
        by_commit_dict: {start commit: the value returned for it} over the sides read
            so far, added to here.

    Returns:
        {"commit" (the commit the side starts from), "range", "own", "define"
        (analyse()'s for that commit)}, or None for a side that starts from base,
        shares no commit with it, or holds the tree base holds. Where base and the
        side both reach more than one commit last, the first git names is the
        commit the side starts from, every one of them is analysed and "define" is
        _most_define_dict()'s over them all.

    Raises:
        RuntimeError: When git could not say which commit the two share.
    """
    start_list = _merge_base_list(project_dir, base, commit)
    if not start_list or start_list[0] == base \
            or gitcmd.tree_of(project_dir, commit) == gitcmd.tree_of(project_dir, base):
        return None
    for start in start_list:
        if start not in by_commit_dict:
            logger.info("%s starts from %s, not from %s, so that commit is analysed too",
                        gitcmd.short(commit), gitcmd.short(start), gitcmd.short(base))
    start_dict_list = [_analysis_at(project_dir, output_dir, start, exclude_list,
                                    by_commit_dict) for start in start_list]
    if len(start_dict_list) == 1:
        return start_dict_list[0]
    return {**start_dict_list[0], "define": _most_define_dict(
        [start_dict["define"] for start_dict in start_dict_list])}


def _analysis_at(project_dir: str, output_dir: str, commit: str,
                 exclude_list: list[str] | None, by_commit_dict: dict) -> dict:
    """Return the analysis of one commit, made once.

    A commit not read before is checked out and analysed, and HEAD is left there.

    Args:
        project_dir: The repository.
        output_dir: Where the analysis is written.
        commit: The commit to analyse.
        exclude_list: The extra names kept out of the analysis.
        by_commit_dict: {commit: the value returned for it} over the commits read so
            far, added to here.

    Returns:
        {"commit", "range", "own", "define" (analyse()'s for that commit)}.
    """
    if commit not in by_commit_dict:
        _move_head(project_dir, commit)
        commit_dict = _analyse_head(project_dir, output_dir, exclude_list)
        by_commit_dict[commit] = {"commit": commit, "range": commit_dict["range"],
                                  "own": commit_dict["own"],
                                  "define": commit_dict["define"]}
    return by_commit_dict[commit]


def _side_commit_dict(project_dir: str, branch_list: list[str], resume: str = ""
                      ) -> dict[str, str]:
    """Return the commit of every side the branches are read as.

    Args:
        project_dir: The repository.
        branch_list: The branch names.
        resume: The branch among branch_list an earlier merge left, or "".

    Returns:
        {side: commit}: a branch under its own name, and a branch that is a union an
        earlier merge left unsettled as the commits it was merged from
        (earlier_side_list), under side_name()'s names.
    """
    commit_by_side_dict = {}
    for branch in branch_list:
        tip = commit_of(project_dir, branch)
        side_commit_list = earlier_side_list(project_dir, tip, is_union=branch == resume)
        commit_by_side_dict.update(
            {side_name(commit): commit for commit in side_commit_list} or {branch: tip})
    return commit_by_side_dict


def _fork_define_dict(project_dir: str, output_dir: str, base: str,
                      commit_by_side_dict: dict[str, str], start_define_dict: dict,
                      exclude_list: list[str] | None, by_commit_dict: dict
                      ) -> dict[tuple[str, str], dict]:
    """Return what the commit two sides both reach last defines, pair by pair.

    A pair is left out where that commit is the one the second of them starts from. A
    commit not analysed yet is checked out and analysed (_analysis_at), and HEAD is put
    back at base afterwards.

    Args:
        project_dir: The repository.
        output_dir: Where the analysis is written. "" reads no pair.
        base: The commit every branch is merged onto.
        commit_by_side_dict: {side: commit} over every side.
        start_define_dict: analyse()'s "define" for base.
        exclude_list: The extra names kept out of the analysis.
        by_commit_dict: As _analysis_at() takes it, holding the commit each side
            that does not start from base starts from.

    Returns:
        {(side, another side): analyse()'s "define" at git's merge base of the two}.
        A pair that shares no commit has no entry. For a pair with more than one
        merge base, both of its entries hold _most_define_dict()'s mapping over
        them all.

    Raises:
        RuntimeError: When git could not say which commit two sides share.
    """
    if not output_dir:
        return {}
    start_by_side_dict = {}
    for side, commit in commit_by_side_dict.items():
        start_by_side_dict[side] = (_merge_base_list(project_dir, base, commit)
                                    or [base])[0]
    fork_define_dict: dict[tuple[str, str], dict] = {}
    is_head_moved = False
    side_list = list(commit_by_side_dict)
    for index, side in enumerate(side_list):
        for other in side_list[index + 1:]:
            fork_list = _merge_base_list(project_dir, commit_by_side_dict[side],
                                         commit_by_side_dict[other])
            if not fork_list or (len(fork_list) == 1 and all(
                    fork_list[0] == start_by_side_dict[name] for name in (side, other))):
                continue
            define_list = []
            for fork in fork_list:
                if fork == base:
                    define_list.append(start_define_dict)
                    continue
                is_head_moved = is_head_moved or fork not in by_commit_dict
                define_list.append(_analysis_at(project_dir, output_dir, fork,
                                                exclude_list, by_commit_dict)["define"])
            define_dict = define_list[0] if len(define_list) == 1 \
                else _most_define_dict(define_list)
            # Each of the two is read against the fork where it is not the one
            # commit the other one starts from
            for one, another in ((side, other), (other, side)):
                if len(fork_list) > 1 or fork_list[0] != start_by_side_dict[another]:
                    fork_define_dict[(one, another)] = define_dict
    if is_head_moved:
        _move_head(project_dir, base)
    return fork_define_dict


def _merge_base_list(project_dir: str, commit: str, other_commit: str) -> list[str]:
    """Return the commits two commits both reach last.

    Args:
        project_dir: The repository.
        commit: One commit.
        other_commit: The other.

    Returns:
        Every merge base git names for the two, in git's order: one commit, more
        where each of the two has merged the other, none when they share no commit.

    Raises:
        RuntimeError: When git could not say.
    """
    proc = gitcmd.run(project_dir, "merge-base", "--all", commit, other_commit)
    # git merge-base ends with 1 when the two share no commit
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"could not read where {gitcmd.short(other_commit)} starts "
                           f"from: {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.split()


def _most_define_dict(define_list: list[dict]) -> dict:
    """Return, for every name, the files of the commit that defines it most often.

    Args:
        define_list: analyse()'s "define" for each commit.

    Returns:
        {own name: the file list of the commit holding the most definitions of it}.
    """
    most_dict: dict = {}
    for define_dict in define_list:
        for name, file_list in define_dict.items():
            if len(file_list) > len(most_dict.get(name, ())):
                most_dict[name] = file_list
    return most_dict


def _start_commit_list(project_dir: str, base: str, head: str, branch_list: list[str],
                       resume: str = "") -> list[str]:
    """Return the commits checked out before the union is put in place.

    They are each branch's tip, the commits the sides are read as (_side_commit_dict),
    head where it is not base, and every commit git names as the merge base of two of
    those sides or of one of them and base.

    Args:
        project_dir: The repository.
        base: The commit every branch starts from.
        head: union_base()'s commit.
        branch_list: The branch names.
        resume: The branch among branch_list an earlier merge left, or "".

    Returns:
        Full commit hashes, each once.

    Raises:
        RuntimeError: When a branch cannot be resolved, or git cannot say which
            commit two of them share.
    """
    side_list = list(_side_commit_dict(project_dir, branch_list, resume).values())
    if head != base:
        side_list.append(head)
    commit_list = [commit_of(project_dir, branch) for branch in branch_list] + side_list
    for index, commit in enumerate(side_list):
        for other in (base, *side_list[index + 1:]):
            commit_list += _merge_base_list(project_dir, commit, other)
    return list(dict.fromkeys(commit_list))


def _mark_change_list(project_dir: str) -> list[str]:
    """Return the tracked files git is told not to look at that the index does not match.

    Their contents are not what the index holds. They are the entries marked
    assume-unchanged or skip-worktree (git ls-files -v), which git status does not show.
    A file is read as git add reads it (git hash-object), and a link by what it points
    to. One with nothing at its path is among them, except a skip-worktree entry, which
    a sparse checkout leaves off the disk. A submodule and an entry with a conflict are
    left out.

    Args:
        project_dir: The repository.

    Returns:
        Repository-relative paths, as gitcmd.shown_text() gives them.

    Raises:
        RuntimeError: When git cannot read the index or the files.
    """
    proc = gitcmd.run(project_dir, "ls-files", "-v", "-s", "-z", text=False)
    if proc.returncode != 0:
        raise RuntimeError(f"could not read the index of {project_dir}: "
                           f"{gitcmd.shown_text(proc.stderr).strip()}")
    change_list: list[bytes] = []
    object_by_path_dict: dict[bytes, bytes] = {}
    for entry in proc.stdout.split(b"\0"):
        head, _, raw_path = entry.partition(b"\t")
        part_list = head.split()
        # tag, mode, object, stage: a lowercase tag is assume-unchanged, S skip-worktree
        if len(part_list) != 4 or not (part_list[0].islower() or part_list[0] == b"S") \
                or part_list[3] != b"0" or part_list[1] == gitcmd.GITLINK_MODE:
            continue
        tag, mode, object_name = part_list[:3]
        path = os.path.join(project_dir, os.fsdecode(raw_path))
        if not os.path.lexists(path):
            if tag.upper() != b"S":
                change_list.append(raw_path)
        elif (mode == _LINK_MODE) != os.path.islink(path) \
                or not (os.path.islink(path) or os.path.isfile(path)):
            change_list.append(raw_path)
        elif mode == _LINK_MODE:
            link_proc = gitcmd.run(project_dir, "hash-object", "--stdin",
                                   input_text=os.fsencode(os.readlink(path)), text=False)
            if link_proc.stdout.strip() != object_name:
                change_list.append(raw_path)
        else:
            object_by_path_dict[raw_path] = object_name
    if object_by_path_dict:
        # Each path is quoted as git unquotes a line of --stdin-paths
        line_list = [b'"' + raw_path.replace(b"\\", b"\\\\").replace(b'"', b'\\"')
                     .replace(b"\n", b"\\n") + b'"\n' for raw_path in object_by_path_dict]
        hash_proc = gitcmd.run(project_dir, "hash-object", "--stdin-paths",
                               input_text=b"".join(line_list), text=False)
        if hash_proc.returncode != 0:
            raise RuntimeError(f"could not read the files of {project_dir} git is told "
                               f"not to look at: "
                               f"{gitcmd.shown_text(hash_proc.stderr).strip()}")
        for (raw_path, object_name), file_object in zip(object_by_path_dict.items(),
                                                        hash_proc.stdout.split()):
            if file_object != object_name:
                change_list.append(raw_path)
    return [gitcmd.shown_text(raw_path) for raw_path in change_list]


def put_back_command(project_dir: str, start_branch: str) -> str:
    """Return the shell line that puts the repository back on a branch or commit.

    Args:
        project_dir: The repository, as an absolute path.
        start_branch: The branch or commit was_on() read before anything moved.

    Returns:
        `git -C <repository> checkout <start_branch> && git -C <repository> reset
        --hard`, each word quoted as a shell reads it whole.
    """
    return (f"{command_line('git', '-C', project_dir, 'checkout', start_branch)} && "
            f"{command_line('git', '-C', project_dir, 'reset', '--hard')}")


def put_back(project_dir: str, start_branch: str) -> bool:
    """Put the repository back where it was, and say so when it cannot be.

    The index and the files git tracks are made what the commit HEAD is at holds
    (reset --hard), then start_branch is checked out. A file git does not track stays,
    and the log names the ones git status lists. Where git fails, the error log names
    put_back_command()'s line.

    Args:
        project_dir: The repository, as an absolute path, which held no file git does
            not track and does not ignore when the run started.
        start_branch: The branch or commit was_on() read before anything moved. ""
            leaves the repository as it is.

    Returns:
        True when the repository is back on start_branch.
    """
    if not start_branch:
        return False
    for word_tuple in (("reset", "--quiet", "--hard"), ("checkout", "--quiet", start_branch)):
        proc = gitcmd.run(project_dir, *word_tuple)
        if proc.returncode != 0:
            logger.error("%s could not be put back on %s (%s)%s; to put the repository "
                         "back: %s", project_dir, start_branch,
                         one_line((proc.stderr or proc.stdout).strip()),
                         _new_file_text(project_dir),
                         put_back_command(project_dir, start_branch))
            return False
    new_file_text = _new_file_text(project_dir)
    if new_file_text:
        logger.warning("%s is back on %s%s", project_dir, start_branch, new_file_text)
    return True


def _new_file_text(project_dir: str) -> str:
    """Return the words that name the files git does not track and does not ignore.

    Args:
        project_dir: The repository, which held none of them when the run started.

    Returns:
        "; the <count> file(s) made since the run started that git does not track
        stay: <the paths, as list_text() names them>", or "" for none.
    """
    new_list = gitcmd.new_file_list(project_dir)
    if not new_list:
        return ""
    return (f"; the {len(new_list)} file(s) made since the run started that git does not "
            f"track stay: {list_text(new_list)}")


def prepare(project_dir: str, knowledge_path: str, branch_list: list[str],
            onto: str = "", resume: str = "", output_dir: str = "",
            exclude_list: list[str] | None = None, onto_commit: str = "") -> dict:
    """Read the starting state, then merge the branches into the working tree.

    Called before the analysis is made again: what it reads is the state the branches
    all started from, and what it leaves in the working tree is their union. The caller
    analyses that union and hands both to run_integration().

    Where the result is to be merged into a branch whose tip is ahead of the start,
    that tip is checked out and analysed here as well, and it is that analysis the
    references the merge broke are read against.

    The union is written straight into the working tree. Before the first branch is
    read, the repository is refused as _check_start() refuses it.

    The flow:
    1. Refuse a repository the union would write over (_check_start).
    2. Read the starting state and where the repository is.
    3. Read the branches one at a time and merge them (assemble). Whatever is raised
       from here on is raised once the repository is put back (put_back).

    Args:
        project_dir: The repository, checked out at the commit the branches start from.
            Anything uncommitted in it is refused.
        knowledge_path: The analysis of that commit.
        branch_list: The branch names, in the order they are to be merged.
        onto: The branch the result is to be merged into afterwards, which is merged
            here as well: the merge afterwards is a fast-forward. "" leaves the result
            as a branch of its own.
        resume: The branch an earlier merge left, among branch_list, whose record is
            read for the run and whose sides, where it was left unsettled, are the
            commits it was merged from. "" reads none.
        output_dir: Where the analysis of onto's tip and of each branch is written.
        exclude_list: The extra names kept out of that analysis, as the build takes them.
        onto_commit: The commit of onto this union is to be written against. Given, it
            is used in place of whatever onto points at now.

    Returns:
        {"base" (what the change is written out against, which is onto's commit when
        one was given), "merged" (assemble()'s return value), "before" (analyse()'s
        return value for the starting state), "against" (analyse()'s return value for
        "base": the same dict as "before" when the two commits are one),
        "side_no_resolve" (the references that do not resolve on the branches taken one
        at a time), "side_define" ({side: analyse()'s "define" at that side}, onto
        among them when its tip was analysed), "side_start" ({side:
        _side_start_dict()'s value} for the sides that do not start from the starting
        state), "side_fork" (_fork_define_dict()'s mapping over the sides, onto
        among them when it was merged), "was_on" (where the repository was before the
        union was put in place), "earlier" (earlier_record_dict() for resume, None
        without one)}.

    Raises:
        RuntimeError: When the repository is refused as above, or when a branch
            cannot be resolved, merged or committed.
        KeyboardInterrupt: When the run is stopped. Past the refusals, this and the
            error above are raised once the repository is put back where it was
            (put_back).
    """
    # == Step 1: refuse a repository the union would write over ===============
    base, head = _check_start(project_dir, branch_list, onto, resume, onto_commit)

    # == Step 2: read the starting state ======================================
    start_branch = was_on(project_dir)
    earlier_dict = earlier_record_dict(project_dir, resume) if resume else None
    before_dict = analyse(knowledge_path)
    against_dict = before_dict

    # == Step 3: read the branches and merge them =============================
    # Whatever moves HEAD or writes the working tree from here on is put back before
    # anything is raised, a stop (KeyboardInterrupt) included
    try:
        # The branches are read one at a time first, and the tip being merged into last
        start_by_commit_dict: dict = {}
        side_dict, side_define_dict, side_start_dict = _side_analysis(
            project_dir, output_dir, base, branch_list, exclude_list, resume,
            start_by_commit_dict)
        commit_by_side_dict = _side_commit_dict(project_dir, branch_list, resume)
        if head != base:
            logger.info("%s is at %s, ahead of %s, so that commit is analysed too", onto,
                        gitcmd.short(head), gitcmd.short(base))
            commit_by_side_dict[onto] = head
            start_dict = _side_start_dict(project_dir, output_dir, base, head,
                                          exclude_list, start_by_commit_dict) \
                if output_dir else None
            if start_dict:
                side_start_dict[onto] = start_dict
        side_fork_dict = _fork_define_dict(
            project_dir, output_dir, base, commit_by_side_dict, before_dict["define"],
            exclude_list, start_by_commit_dict)
        if head != base:
            _move_head(project_dir, head)
            against_dict = _analyse_head(project_dir, output_dir, exclude_list)
            side_define_dict[onto] = against_dict["define"]
        merge_dict = assemble(project_dir, base, branch_list, onto=onto,
                              onto_commit=onto_commit, resume=resume)
    except BaseException:
        put_back(project_dir, start_branch)
        raise
    logger.info("merged %d branch(es) onto %s", len(merge_dict["branches"]), gitcmd.short(base))
    return {"base": merge_dict["base"], "merged": merge_dict, "before": before_dict,
            "against": against_dict, "side_no_resolve": side_dict,
            "side_define": side_define_dict, "side_start": side_start_dict,
            "side_fork": side_fork_dict,
            "was_on": start_branch, "earlier": earlier_dict}


def _check_start(project_dir: str, branch_list: list[str], onto: str, resume: str,
                 onto_commit: str) -> tuple[str, str]:
    """Refuse a repository the union would write over, and return where it starts.

    A git too old for merge-tree is refused (_require_merge_tree), then, in this order,
    a directory that is not a git repository with a commit in it (gitcmd.require_head),
    a repository holding anything uncommitted, a tracked file git is told not to look at
    that holds a change (_mark_change_list), and what git does not track where a commit
    _start_commit_list() returns holds a file (_user_in_way_list).

    Args:
        project_dir: The repository, checked out at the commit the branches start from.
        branch_list: The branch names.
        onto: As prepare() takes it.
        resume: As prepare() takes it.
        onto_commit: As prepare() takes it.

    Returns:
        (the commit HEAD is at, union_base()'s commit).

    Raises:
        RuntimeError: When the repository is refused, or a branch cannot be resolved.
    """
    _require_merge_tree(project_dir)
    base = gitcmd.require_head(project_dir)
    uncommit_list = gitcmd.status_path_list(project_dir)
    if uncommit_list:
        raise RuntimeError(_UNCOMMIT_REFUSAL.format(
            project_dir=project_dir, count=len(uncommit_list), paths=list_text(uncommit_list),
            stash=command_line("git", "-C", project_dir, "stash", "--include-untracked"),
            submodule_stash=command_line(
                "git", "-C", os.path.join(project_dir, "<submodule>"), "stash",
                "--include-untracked")))
    mark_list = _mark_change_list(project_dir)
    if mark_list:
        raise RuntimeError(_MARK_REFUSAL.format(
            project_dir=project_dir, count=len(mark_list), paths=list_text(mark_list),
            unmark=" && ".join(
                command_line("git", "-C", project_dir, "update-index", flag, "<file>")
                for flag in ("--no-assume-unchanged", "--no-skip-worktree"))))
    head = union_base(project_dir, base, onto, onto_commit)
    in_way_list = _user_in_way_list(
        project_dir, base, _start_commit_list(project_dir, base, head, branch_list, resume))
    if in_way_list:
        raise RuntimeError(_IN_WAY_REFUSAL.format(
            project_dir=project_dir, count=len(in_way_list), base=gitcmd.short(base),
            paths=list_text(in_way_list)))
    return base, head


def run_integration(prepare_dict: dict, knowledge_path: str, project_name: str,
                    output_dir: str, exclude_list: list[str] | None = None,
                    request: str = "") -> dict:
    """Check the merged repository and, where anything is left, have a model settle it.

    A merge with nothing left to settle ends here: no model is called and no turn is
    spent. What is left is what _unsettle_reason_list() makes of the measurement taken
    before any model is called, which is what the call ending the run and the check
    after the run are held to as well. A file the sides both reached (meet_path_list)
    is left to settle until the run has read it, and a file kept as it was where
    the change of a side to it was to be merged (in_block_read_list) until the run
    has read it on that side.

    The references the merge broke are read against the commit the union is written
    out against; the definitions each side changed are read against the start.

    Args:
        prepare_dict: prepare()'s return value.
        knowledge_path: The analysis of the merged working tree.
        project_name: The directory name of PROJECT_DIR.
        output_dir: Where the analysis is written when it is made again.
        exclude_list: The extra names kept out of the analysis, as the build takes them.
        request: The request the branches were made for, put at the head of what the
            stage is asked. "" leaves it out.

    Returns:
        The return value of RLM.run() with "work" (what the merge left behind),
        "called" (whether a model was called at all), "left" (what the machine still
        finds wrong, empty when there is nothing) and "settled" added, and, for a run
        that called a model, "head_block" (head_block_list()). A run that called no
        model has "answer" holding what was merged.
    """
    project_dir = config.PROJECT_DIR
    reset_stage()
    # The editing tools count what is changed from the union in the working tree on,
    # with no limit on the rounds
    edit.reset_patch_round(round_limit=0)
    merge_dict = prepare_dict["merged"]
    _STAGE_DICT["commit_dict"] = {"": merge_dict["start"],
                                  **{entry_dict["branch"]: entry_dict["commit"]
                                     for entry_dict in merge_dict["branches"]}}

    # == Step 1: measure the union before a model is called ==
    measure_dict = _measure_union(prepare_dict, knowledge_path, project_name, project_dir)
    work_dict = build_work_dict(merge_dict, measure_dict["markers"], measure_dict["broken"],
                                measure_dict["baseline"], measure_dict["test_reason"],
                                len(measure_dict["after"]["no_resolve"]),
                                measure_dict["overlap"],
                                earlier_dict=prepare_dict.get("earlier"))
    work_dict["checks"] = measure_dict["checks"]
    work_dict["unmarked"] = measure_dict["unmarked"]

    # == Step 2: end here when nothing is left to settle ======
    # Before a model is called no file has been read, and every file the sides both
    # reached is left to settle
    meet_list = meet_path_list(merge_dict, measure_dict["overlap"])
    in_block_list = in_block_read_list(merge_dict)
    no_path_list = merge_dict.get("no_path") or []
    stand_in_list = merge_dict.get("stand_in") or []
    if not _unsettle_reason_list(measure_dict["markers"], measure_dict["checks"],
                                 measure_dict["broken"], measure_dict["keep"],
                                 measure_dict["after"]["define"], project_name,
                                 meet_list, measure_dict["test_reason"], no_path_list,
                                 _stand_in_now_list(project_dir, stand_in_list),
                                 in_block_list):
        logger.info("the merge settled itself; no model was called")
        return {"work": work_dict, "called": False, "left": [], "settled": True,
                "answer": request_of(prepare_dict["merged"], request), "history": [],
                "iters": 0, "refused_completes": 0, "harvested": False,
                "error": "", "outcome": None}

    # == Step 3: have the stage settle the work list ==========
    marker_dict = merge_dict.get("marker")
    check = functools.partial(_reason_list, project_dir, output_dir,
                              prepare_dict["against"], measure_dict["baseline"],
                              exclude_list, prepare_dict.get("side_no_resolve"), work_dict,
                              measure_dict["keep"],
                              check_before=measure_dict["check_before"],
                              meet_list=meet_list,
                              marker_dict=marker_dict,
                              no_path_list=no_path_list, stand_in_list=stand_in_list,
                              in_block_list=in_block_list)
    stage = RLM(llm=make_llm(config.INTEGRATE_MODEL, config.INTEGRATE_MAX_TOKEN,
                             "INTEGRATE_MODEL", is_turn_reply=True,
                             think_level=config.INTEGRATE_THINK_LEVEL),
                tool_list=merge_tool_list(),
                instruction=build_integrate_instruction(config.PRIMARY_EXT),
                stage="integrate",
                max_turn=config.INTEGRATE_MAX_TURN)
    stage_request = "\n\n".join([request_of(prepare_dict["merged"], request),
                                 start_text(work_dict)])
    turn_note, check_and_keep = make_state(
        project_dir, merge_dict["commit"], work_dict, check, measure_dict["baseline"],
        range_dict=measure_dict["after"]["range"], project_name=project_name,
        marker_dict=marker_dict,
        uncut_dict={"broken": measure_dict["broken"], "overlap": measure_dict["overlap"],
                    "branches": merge_dict["branches"]})
    out_dict = run_stage(stage, stage_request, {"work": work_dict},
                         INTEGRATE_VAR_DOC,
                         _LATE_ANSWER_NOTE, _LATE_ANSWER_CLOSE,
                         late_state=turn_note, turn_note=turn_note, text_answer=True,
                         on_complete=make_complete_guard(check_and_keep))
    out_dict["work"] = work_dict
    out_dict["called"] = True

    # == Step 4: check what the stage left ====================
    # A run the guard accepted keeps what the guard's check found; a run that used up
    # its refusals or its turns is checked here
    has_guard_check = (out_dict["outcome"] == COMPLETE_KEY
                       and out_dict["refused_completes"] < config.INTEGRATE_MAX_REFUSAL)
    out_dict["left"] = work_dict["left"] if has_guard_check else check_and_keep()
    out_dict["settled"] = not out_dict["left"]
    out_dict["head_block"] = head_block_list()
    # What the last turn recorded
    work_dict["decisions"] = decision_list()
    return out_dict


def _measure_union(prepare_dict: dict, knowledge_path: str, project_name: str,
                   project_dir: str) -> dict:
    """Measure the union in the working tree, before any model is called.

    Args:
        prepare_dict: prepare()'s return value.
        knowledge_path: The analysis of the merged working tree.
        project_name: The directory name of PROJECT_DIR.
        project_dir: The repository holding the union.

    Returns:
        {"after" (analyse()'s return value for the union), "broken" (broken_list_of()'s
        list), "side" ([(branch, side_dict_of()'s mapping), ...] over own_side_list()'s
        branches, each read from the commit it starts from), "keep" (keep_list_of()'s
        list over them), "overlap" (overlap_entry_list()'s list), "markers"
        (marker_path_list()), "unmarked" (assemble()'s), "baseline" (_baseline_set()'s
        set), "test_reason" (_test_reason() over the union), "check_before"
        (edit.run_check() over HEAD's tree, None without a check command) and "checks"
        (check_reason()'s lines)}.
    """
    before_dict, merge_dict = prepare_dict["before"], prepare_dict["merged"]
    after_dict = analyse(knowledge_path)
    broken_list = broken_list_of(prepare_dict["against"], after_dict, project_name,
                                 prepare_dict.get("side_no_resolve"))
    side_start_dict = prepare_dict.get("side_start") or {}
    side_list = []
    for entry_dict in own_side_list(
            project_dir,
            [entry_dict for entry_dict in merge_dict["branches"]
             if not entry_dict.get("carry")],
            merge_dict["start"], set(side_start_dict)):
        start_dict = side_start_dict.get(entry_dict["branch"]) \
            or {"commit": merge_dict["start"], "range": before_dict["range"]}
        side_list.append((entry_dict["branch"], side_dict_of(
            change_dict_of(project_dir, start_dict["commit"], entry_dict["commit"]),
            start_dict["range"], project_name)))
    overlap_list = overlap_entry_list(before_dict, after_dict, side_list, project_name)
    side_define_dict = prepare_dict.get("side_define") or {}
    keep_list = keep_list_of(
        side_list, before_dict,
        {branch: side_define_dict[branch] for branch, _ in side_list
         if branch in side_define_dict}, side_start_dict,
        _start_side_list(project_dir, merge_dict["start"],
                         [side_start_dict[branch] for branch, _ in side_list
                          if branch in side_start_dict], project_name),
        prepare_dict.get("side_fork"))
    marker_list = marker_path_list(project_dir, merge_dict.get("marker"))
    test_dict = edit.run_test() if edit.test_can_run() else None
    baseline_set = _baseline_set(test_dict, functools.partial(
        _head_test_dict, project_dir, merge_dict["commit"]))
    test_reason = _test_reason(test_dict, baseline_set)
    _STAGE_DICT["head_baseline"] = bool(edit.baseline_name_list(baseline_set)) \
        and test_dict["status"] != "failed"
    _STAGE_DICT["head_stop"] = [] if baseline_set is None else [
        command for command, _, _ in edit.compare_part_list(test_dict, baseline_set)
        if edit.is_stop_before(baseline_set, command)]
    _STAGE_DICT["head_short"] = [] if baseline_set is None else [
        command for command, _, _ in edit.compare_part_list(test_dict, baseline_set)
        if _is_head_short(baseline_set, command)]
    _STAGE_DICT["not_compared"] = edit.not_compared_list(baseline_set)
    check_before = _at_head(project_dir, merge_dict["commit"], edit.run_check) \
        if edit.check_argv_list() else None
    return {"after": after_dict, "broken": broken_list, "side": side_list, "keep": keep_list,
            "overlap": overlap_list, "markers": marker_list,
            "unmarked": merge_dict.get("unmarked", []), "baseline": baseline_set,
            "test_reason": test_reason, "check_before": check_before,
            "checks": check_reason(check_before)}


def _start_side_list(project_dir: str, start: str, own_start_list: list[dict],
                     project_name: str) -> list[tuple[dict, dict]]:
    """Return what the start the sides share changed since each commit a side starts from.

    Args:
        project_dir: The repository.
        start: The commit the sides share.
        own_start_list: _side_start_dict()'s value for each side that starts from a
            commit of its own.
        project_name: The directory name of PROJECT_DIR, for the path conversion.

    Returns:
        [(side_dict_of()'s mapping for the start read from that commit, the commit's
        own value), ...], one for each commit.

    Raises:
        RuntimeError: When a diff could not be read.
    """
    by_commit_dict = {own_start_dict["commit"]: own_start_dict
                      for own_start_dict in own_start_list}
    return [(side_dict_of(change_dict_of(project_dir, commit, start),
                          own_start_dict["range"], project_name), own_start_dict)
            for commit, own_start_dict in by_commit_dict.items()]


def own_side_list(project_dir: str, branch_list: list[dict], start: str = "",
                  own_start_set: set[str] | frozenset[str] = frozenset()) -> list[dict]:
    """Return the merged branches whose own change is read as a side's.

    A branch whose commit another merged branch reaches is left out. Of branches at the
    same commit, the first is kept. A branch that does not start from start is left
    out as well, unless the commit it starts from was analysed: one holding the tree
    start holds adds nothing, and any other is logged.

    Args:
        project_dir: The repository.
        branch_list: assemble()'s "branches".
        start: The commit every branch starts from. "" leaves that out.
        own_start_set: The branches read from a commit of their own, as prepare()'s
            "side_start" names them.

    Returns:
        Those entries, in the order given.
    """
    keep_list = []
    for index, entry_dict in enumerate(branch_list):
        commit = entry_dict["commit"]
        if start and entry_dict["branch"] not in own_start_set \
                and not gitcmd.is_ancestor(project_dir, start, commit):
            if gitcmd.tree_of(project_dir, commit) == gitcmd.tree_of(project_dir, start):
                logger.info("%s holds what %s holds, so it is no side of its own",
                            entry_dict["branch"], gitcmd.short(start))
            else:
                logger.warning("%s does not start from %s and the commit it starts "
                               "from was not analysed, so what it added or changed "
                               "is not looked for in the union", entry_dict["branch"],
                               gitcmd.short(start))
            continue
        held_list = [other_index < index if other_dict["commit"] == commit
                     else gitcmd.is_ancestor(project_dir, commit, other_dict["commit"])
                     for other_index, other_dict in enumerate(branch_list)
                     if other_index != index]
        if any(held_list):
            logger.info("%s is already held by another branch merged, so what it changed "
                        "is read as that branch's", entry_dict["branch"])
            continue
        keep_list.append(entry_dict)
    return keep_list
