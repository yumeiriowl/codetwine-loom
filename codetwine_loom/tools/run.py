"""run_project: start the edited program and look at what it produced.

It is not a shell. The command line is config.RUN_COMMAND, with
edit.with_uv_project_word() applied, followed by arg_list. The files passed in file_dict
are written into a scratch workspace outside the repository. A path passed relative
rather than as "{workspace}/..." points into the repository instead: files a run creates
there are deleted afterwards and reported in "left_in_repo", and a sample no argument and
no environment value points at is refused before anything starts.

run_once() starts the same program with a workspace of its own (CHECK_SLOT), and its
runs are not counted.
"""

import atexit
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import NamedTuple

from codetwine.utils.file_utils import read_source

from codetwine_loom import config, gitcmd, tmpfile
from codetwine_loom.tools import edit, source
from codetwine_loom.languages import LANG_DICT, LANG_NAME_DICT, RUN_EXAMPLE_DICT
from codetwine_loom.textcut import cut, cut_tail
from codetwine_loom.textfile import lf_text

logger = logging.getLogger(__name__)

# The scratch directories, created on first use, one per slot (MODEL_SLOT, CHECK_SLOT),
# in the system temp area rather than inside PROJECT_DIR
_workspace_dict: dict[str, str] = {}

# The slot run_project uses, and the slot run_once() uses
MODEL_SLOT = "model"
CHECK_SLOT = "check"

# Whether deleting the scratch directories when the process ends has been registered
_is_exit_drop_on = False

# What {workspace} in an argument or an environment value is replaced with
_PLACEHOLDER = "{workspace}"

# The width launcher_note() wraps its lines to
_NOTE_WIDTH = 88

# git's own directory name: the listing of the repository lists it and does not walk
# into it
_GIT_DIR_NAME = ".git"

# How many run_project calls tried to run the program since reset_run_state()
_run_count = 0

# What the last of those calls whose program ran to its end printed, with the line
# edit.refill_cache_on_dep_change() returned for it in front. A run_once() run is
# neither kept here nor counted in _run_count
_last_output = ""


def _drop_workspace(slot: str | None = None) -> None:
    """Delete a scratch directory; the next call starts from an empty one.

    Args:
        slot: Which workspace to delete. None deletes every one of them.
    """
    for name in [slot] if slot else list(_workspace_dict):
        path = _workspace_dict.pop(name, None)
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def reset_run_state() -> None:
    """Drop every slot's scratch workspace, the call counter and the last output.

    All three are module level.
    """
    global _run_count, _last_output
    _drop_workspace()
    _run_count = 0
    _last_output = ""


def run_count() -> int:
    """Return how many run_project calls tried to run the program since the last reset.

    Returns:
        The count, 0 after reset_run_state().
    """
    return _run_count


def available() -> bool:
    """Say whether run_project is handed over: it has a launcher and calls are allowed.

    Returns:
        True when RUN_COMMAND names a command and RUN_PROJECT_MAX_CALL is above 0.
    """
    return bool(config.RUN_COMMAND) and config.RUN_PROJECT_MAX_CALL > 0


def launcher_note(indent: str) -> str:
    """Return what the primary language says of its launcher, where it is the one in force.

    Args:
        indent: What each line of the note starts with.

    Returns:
        The language's run_note wrapped to the instructions' width and ending in a
        newline, while RUN_COMMAND is the language's run_command_tuple; "" otherwise and
        for a language with no run_note.
    """
    lang_config = LANG_DICT.get(config.PRIMARY_EXT)
    if lang_config is None or not lang_config.run_note \
            or config.RUN_COMMAND != list(lang_config.run_command_tuple):
        return ""
    return textwrap.fill(lang_config.run_note, width=_NOTE_WIDTH, initial_indent=indent,
                         subsequent_indent=indent, break_on_hyphens=False) + "\n"


def call_remain_count() -> int:
    """Return how many run_project calls are left.

    Returns:
        config.RUN_PROJECT_MAX_CALL minus what has been used, never below 0.
    """
    return max(0, config.RUN_PROJECT_MAX_CALL - _run_count)


def last_output() -> str:
    """Return what the last run_project run printed.

    Returns:
        The tail of the output (RUN_OUTPUT_TAIL) of the last counted run that ran to its
        end, with the line edit.refill_cache_on_dep_change() returned for it in front,
        or "" when there is none since reset_run_state().
    """
    return _last_output


def _ensure_workspace(slot: str) -> str:
    """Return one slot's scratch directory, creating it on first use.

    The first directory made in this process registers _drop_workspace() to run when
    the process ends.

    Args:
        slot: MODEL_SLOT or CHECK_SLOT.

    Returns:
        The absolute path of that workspace.
    """
    global _is_exit_drop_on
    path = _workspace_dict.get(slot)
    if path is None or not os.path.isdir(path):
        path = tempfile.mkdtemp(prefix=f"codetwine-loom-{slot}-")
        _workspace_dict[slot] = path
        # The directories still there when the process ends are deleted then
        if not _is_exit_drop_on:
            atexit.register(_drop_workspace)
            _is_exit_drop_on = True
    return path


def _resolve_in_workspace(root: str, rel_path: str) -> str:
    """Make rel_path absolute and confirm it stays inside the workspace.

    Args:
        root: The workspace directory.
        rel_path: A path relative to the workspace.

    Returns:
        The normalised absolute path.

    Raises:
        ValueError: When the path escapes the workspace.
    """
    target = os.path.realpath(os.path.join(root, rel_path))
    if os.path.commonpath([os.path.realpath(root), target]) != os.path.realpath(root):
        raise ValueError(f"outside the workspace: {rel_path}. Write the path relative to "
                         f"the workspace, e.g. \"input.txt\"")
    return target


def _write_file_list(root: str, file_dict: dict[str, str]) -> list[str]:
    """Write the sample files the run needs into the workspace.

    Args:
        root: The workspace directory.
        file_dict: {path relative to the workspace: contents}.

    Returns:
        The paths written, relative to the workspace.

    Raises:
        ValueError: When a path escapes the workspace.
        OSError: When a file cannot be written, and when something other than a regular
            file stands at its path (tmpfile.check_file).
    """
    write_list = []
    for rel_path, text in file_dict.items():
        target = _resolve_in_workspace(root, rel_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            tmpfile.check_file(target, sys.maxsize)
        with open(target, "w", encoding="utf-8") as fp:
            fp.write(text)
        write_list.append(rel_path)
    return sorted(write_list)


def _list_workspace(root: str, skip_set: set[str]) -> list[str]:
    """List the files under the workspace that the run produced.

    Args:
        root: The workspace directory.
        skip_set: Relative paths that were already there before the run.

    Returns:
        The relative paths, sorted.
    """
    path_list = []
    for dir_path, _, file_name_list in os.walk(root):
        for name in file_name_list:
            file_rel = os.path.relpath(os.path.join(dir_path, name), root).replace("\\", "/")
            if file_rel not in skip_set:
                path_list.append(file_rel)
    return sorted(path_list)


def _fill(text: str, root: str) -> str:
    """Replace the {workspace} placeholder with the real path."""
    return text.replace(_PLACEHOLDER, root)


def _result_dict(ok: bool, workspace: str, returncode: int = -1, output: str = "",
                 output_list: list[str] | None = None,
                 show_dict: dict[str, str] | None = None,
                 error: str = "", in_repo_list: list[str] | None = None,
                 more_count: int = 0) -> dict:
    """Build the return value, with call_remain_count() as "calls_left".

    Args:
        ok: Whether the program exited 0.
        workspace: The scratch directory this call used; "" when there was none.
        returncode: The program's exit code; -1 when it never started or timed out.
        output: The tail of stdout+stderr, with the line
            edit.refill_cache_on_dep_change() returned in front.
        output_list: Files under the workspace that were not there before.
        more_count: How many more such files there are than output_list holds.
        show_dict: {path: contents} for the paths asked for in show_list.
        error: Why it could not run; "" when it ran.
        in_repo_list: Files the run wrote inside the repository, now deleted.

    Returns:
        The dict documented in run_project's Returns.
    """
    return {"ok": ok, "returncode": returncode, "output": output,
            "produced": output_list or [], "produced_more": more_count,
            "shown": show_dict or {},
            "workspace": workspace, "error": error,
            "left_in_repo": in_repo_list or [],
            "calls_left": call_remain_count()}


def _new_file_set() -> set[str]:
    """Return the paths git does not track in the target repository.

    Returns:
        Repository-relative paths as os.fsdecode() gives them, which remove the file
        whatever its bytes. Empty when PROJECT_DIR is not a git repository.
    """
    return set(gitcmd.new_file_list(config.PROJECT_DIR, disk=True))


class _PathRecord(NamedTuple):
    """What was under the target repository at one moment.

    Attributes:
        path_set: Every path that was listed: files, links and directories, relative
            to the repository, with "/" between their parts.
        no_list_dir_set: The directories whose entries could not be listed, or not all
            of them, as paths of the same form; "" for the repository's own.
    """

    path_set: set[str]
    no_list_dir_set: set[str]


def _path_record() -> _PathRecord:
    """List every path under the target repository: its files, links and directories.

    Returns:
        The paths, and the directories that could not be listed. A directory named
        _GIT_DIR_NAME is listed and not walked into, and neither is a symbolic link to
        a directory.
    """
    root = os.path.realpath(config.PROJECT_DIR)
    record = _PathRecord(set(), set())
    wait_list = [""]
    while wait_list:
        rel_dir = wait_list.pop()
        head = rel_dir + "/" if rel_dir else ""
        try:
            with os.scandir(os.path.join(root, rel_dir)) as entry_iter:
                for entry in entry_iter:
                    record.path_set.add(head + entry.name)
                    if entry.name != _GIT_DIR_NAME \
                            and entry.is_dir(follow_symlinks=False):
                        wait_list.append(head + entry.name)
        except OSError:
            record.no_list_dir_set.add(rel_dir)
    return record


def _under_dir(rel_path: str, dir_set: set[str]) -> bool:
    """Say whether a path lies under one of a set of directories.

    Args:
        rel_path: A repository-relative path.
        dir_set: Repository-relative paths of directories; "" is the repository's own.

    Returns:
        True when one of the directories above rel_path is in dir_set.
    """
    rel_dir = os.path.dirname(rel_path)
    while rel_dir and rel_dir not in dir_set:
        rel_dir = os.path.dirname(rel_dir)
    return rel_dir in dir_set


def _keep_program_dir(rel_path: str, path_before_set: set[str]) -> None:
    """Record the directories above a file that were made after a listing was taken.

    Args:
        rel_path: The repository-relative path of a file the program wrote.
        path_before_set: The paths that were there before the program started.
    """
    rel_dir = os.path.dirname(rel_path)
    while rel_dir and rel_dir not in path_before_set:
        edit.keep_new_dir(rel_dir)
        rel_dir = os.path.dirname(rel_dir)


def _clean_repo(base_set: set[str], record: _PathRecord) -> list[str]:
    """Delete the files the run just wrote inside the repository.

    A path passed relative rather than as "{workspace}/..." makes the program write into
    the target repository. A file is deleted when git lists it as untracked and did not
    before the program started, and nothing was at its path then either
    (record.path_set): a file that was already there stays, whatever the program did
    to the ignore rules or the index, and a tracked file the run modified is left
    alone. A path under a directory that could not be listed then
    (record.no_list_dir_set) stays too. A repository of its own the program made,
    which git lists as its directory, is deleted with all it holds. The directories the
    program made for the files go with them (edit.remove_empty_dir), and a directory
    that was there before it started stays.

    Args:
        base_set: The untracked paths as they were before the program started.
        record: _path_record() as it was before the program started.

    Returns:
        The repository-relative paths that were deleted, sorted, as gitcmd.shown_text()
        gives them. A path that could not be deleted is not among them.
    """
    delete_path_list = []
    for new_path in sorted(_new_file_set() - base_set):
        rel_path = new_path.rstrip("/")
        if rel_path in record.path_set or _under_dir(rel_path, record.no_list_dir_set):
            continue
        _keep_program_dir(rel_path, record.path_set)
        full_path = os.path.join(config.PROJECT_DIR, rel_path)
        try:
            if os.path.isdir(full_path) and not os.path.islink(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)
        except OSError as e:
            logger.info("%s, which the program wrote, could not be deleted (%s)",
                        gitcmd.shown_text(new_path), e)
            continue
        edit.remove_empty_dir(rel_path)
        delete_path_list.append(gitcmd.shown_text(new_path))
    return delete_path_list


# What comes back when a sample was written but no argument and no environment value
# points at it: file_dict= lands in the workspace while the program runs from the
# repository
_NO_PLACEHOLDER = (
    'the files you passed in file_dict= were not given to the program. They are written '
    'into the workspace, but none of your arguments and none of your env_dict= values '
    'contains "{workspace}", so the program ran from the repository and looked for '
    'those paths there. Pass the paths as "{workspace}/<path>", in whichever argument '
    'or env_dict= value this program takes them. Nothing was started')


# What comes back when args is empty or starts with a flag and the launcher does not end
# with "--". The launcher is prepended to args: `run_project(["--in", ...])` becomes
# `<launcher> --in ...`, where the launcher takes the flag as its own option
_NO_PROGRAM = (
    'the first item of args is "{first}", which is a flag, so {launcher} took it as its '
    'own option and the program never started. args goes after the launcher, so it has '
    'to begin with what to run. Nothing was started. A call that works here is:\n'
    '{call}')

# The call _call_example() returns when config.SMOKE_ARG is empty. It names no program
# of its own
_GENERIC_CALL = ('run_project(["<what starts this project>", "<its flags>"])  '
                 '# any path you pass it goes in as "{workspace}/<path>"')


def _call_example() -> str:
    """Return a run_project() call, as one string.

    It is built from config.SMOKE_ARG, SMOKE_FILE and SMOKE_ENV.

    Returns:
        The call to write, or _GENERIC_CALL when config.SMOKE_ARG is empty.
    """
    if not config.SMOKE_ARG:
        return _GENERIC_CALL
    part_list = [f"run_project({list(config.SMOKE_ARG)!r}"]
    if config.SMOKE_FILE:
        name_text = ", ".join(repr(name) + ": ..." for name in config.SMOKE_FILE)
        part_list.append(f"file_dict={{{name_text}}}")
    if config.SMOKE_ENV:
        part_list.append(f"env_dict={dict(config.SMOKE_ENV)!r}")
    return ", ".join(part_list) + ")"


def _argument_error(arg_list: object, file_dict: object, env_dict: object,
                    show_list: object) -> str:
    """Say which argument of one call is not of the type the tool takes.

    Args:
        arg_list: What was passed as arg_list.
        file_dict: What was passed as file_dict.
        env_dict: What was passed as env_dict.
        show_list: What was passed as show_list, a single path already made a list.

    Returns:
        One sentence naming the argument and what it has to be, or "" when every
        argument can be used: the types, a name of env_dict the system takes as an
        environment variable's, and no NUL character in an argument or a value.
    """
    if not isinstance(arg_list, (list, tuple)):
        return (f"arg_list must be a list of strings, one per argument, not "
                f"{type(arg_list).__name__}")
    if file_dict is not None and not (
            isinstance(file_dict, dict)
            and all(isinstance(key, str) and isinstance(value, str)
                    for key, value in file_dict.items())):
        return ("file_dict must be {path: contents} with both of them strings. Turn "
                "contents that are not text into text first (json.dumps() for a dict)")
    if env_dict is not None and not isinstance(env_dict, dict):
        return f"env_dict must be {{name: value}}, not {type(env_dict).__name__}"
    for name in env_dict or ():
        if not isinstance(name, str) or not name or "=" in name or "\0" in name:
            return (f"each name of env_dict must be a string that is not empty and "
                    f"holds no \"=\": {name!r}")
    if any("\0" in str(value) for value in [*arg_list, *(env_dict or {}).values()]):
        return "an argument and an env_dict value cannot hold a NUL character"
    if show_list is not None and not (
            isinstance(show_list, (list, tuple))
            and all(isinstance(path, str) for path in show_list)):
        return "show_list must be a list of paths, each a string"
    return ""


def run_project(arg_list: list[str], file_dict: dict[str, str] | None = None,
                env_dict: dict[str, str] | None = None, show_list: list[str] | None = None,
                reset: bool = False) -> dict:
    """Run the program with your arguments and report what it printed and produced.

    The command starts with the configured launcher (RUN_COMMAND); your args follow it.
    Anything you write as "{workspace}" in args or in env values is replaced with a
    scratch directory that only this tool owns. Files you pass in file_dict= are written
    there first: a small sample project can be built, run against and read back. The
    workspace survives between calls; pass reset=True to start from an empty one.

    It does not count towards MAX_PATCH_ROUND.

    Examples:
        run_project(["--in", "{workspace}/sample", "--out", "{workspace}/out"],
                    file_dict={"sample/one.txt": "a line\\n"},
                    env_dict={"SOME_FEATURE": "off"},
                    show_list=["out/result.json"])

    Args:
        arg_list: The arguments after the launcher, e.g. ["--flag", "value"]. Use
            "{workspace}" wherever the program needs a path it may write to.
        file_dict: {path relative to the workspace: contents} written before the run.
            Paths that escape the workspace are refused.
        env_dict: Extra environment variables for the run, on top of the current ones.
        show_list: Paths, relative to the workspace, whose contents you want back. Each is
            cut to what one read_source_file call returns at most (source.read_limit()),
            with the "... (N chars omitted)" marker after it. One path alone may be
            passed as a string.
        reset: True empties the workspace before writing files and running.

    Returns:
        {"ok": bool,           # whether the program exited 0
         "returncode": int,    # -1 when it could not start or timed out
         "output": str,        # the last RUN_OUTPUT_TAIL characters of stdout+stderr,
                               # after a "note:" line when the package cache was filled
                               # again or the call to fill it failed
         "produced": list[str],# the first RUN_MAX_FILE files under the workspace that
                               # were not there before
         "produced_more": int, # how many more such files there are than "produced"
                               # lists
         "shown": dict,        # {path: contents} for the paths asked for in show_list
         "workspace": str,     # the scratch directory this run used
         "error": str,         # why it could not run; "" when it ran
         "left_in_repo": list[str],  # files the run wrote into the repository by a
                               # relative path; they have been deleted
         "calls_left": int}    # how many run_project calls are left
    """
    return _run(arg_list, file_dict, env_dict, show_list, reset, MODEL_SLOT, count=True)


def run_once(arg_list: list[str], file_dict: dict[str, str] | None = None,
             env_dict: dict[str, str] | None = None, show_list: list[str] | None = None,
             reset: bool = True) -> dict:
    """Start the program with CHECK_SLOT's workspace, without counting the run.

    The workspace is emptied by default; reset=False keeps it.

    Args:
        arg_list: The arguments after the launcher (config.RUN_COMMAND).
        file_dict: {path relative to the workspace: contents} written before the run.
        env_dict: Extra environment variables for the run.
        show_list: Paths, relative to the workspace, whose contents come back in "shown".
        reset: True empties the check's workspace first.

    Returns:
        The same dict run_project returns, with every file the run wrote in "produced"
        whatever RUN_MAX_FILE is.
    """
    return _run(arg_list, file_dict, env_dict, show_list, reset=reset, slot=CHECK_SLOT, count=False)


def _refuse_reason(arg_list: list[str], file_dict: dict[str, str] | None,
                   env_dict: dict[str, str] | None, show_list: list[str] | None,
                   count: bool) -> str:
    """Say why one call is refused before anything is written or started.

    The checks run in this order: the argument types, the call limit of a counted run,
    RUN_COMMAND being set, a program name ahead of the flags, and a sample an argument
    or an environment value points at.

    Args:
        arg_list: The arguments after the launcher.
        file_dict: {path relative to the workspace: contents} written before the run.
        env_dict: Extra environment variables for the run.
        show_list: Paths whose contents come back in "shown", a single path already
            made a list.
        count: Whether the run counts towards run_count().

    Returns:
        The error the result carries, or "" when the call goes ahead.
    """
    argument_error = _argument_error(arg_list, file_dict, env_dict, show_list)
    if argument_error:
        return f"{argument_error}. Nothing was started"
    if count and _run_count >= config.RUN_PROJECT_MAX_CALL:
        logger.info("run_project hit its call limit: %d calls", _run_count)
        return (f"run_project hit its call limit "
                f"({config.RUN_PROJECT_MAX_CALL} calls)")
    if not config.RUN_COMMAND:
        return ("RUN_COMMAND is not set and this language has no default "
                "launcher. Set RUN_COMMAND to what starts this project "
                f"(for {LANG_NAME_DICT[config.PRIMARY_EXT]}"
                f" that tends to look like "
                f"\"{RUN_EXAMPLE_DICT[config.PRIMARY_EXT]}\")."
                " A call that works here is:\n" + _call_example())

    # A missing program name is caught first, ahead of an unreachable sample. A launcher
    # that ends with "--" has named the program itself, and everything after it is the
    # program's own
    if config.RUN_COMMAND[-1] != "--" and (not arg_list or str(arg_list[0]).startswith("-")):
        return _NO_PROGRAM.format(
            first="" if not arg_list else arg_list[0],
            launcher=" ".join(config.RUN_COMMAND), call=_call_example())

    # A sample no argument and no environment value points at is refused before
    # anything starts
    if file_dict and not any(_PLACEHOLDER in str(value)
                             for value in [*arg_list, *(env_dict or {}).values()]):
        return _NO_PLACEHOLDER
    return ""


def _show_dict(root: str, show_list: list[str] | None) -> dict[str, str]:
    """Read back the files of the workspace a call asked to see.

    Args:
        root: The workspace directory.
        show_list: Paths relative to the workspace. None reads none.

    Returns:
        {path: contents cut to source.read_limit(), line endings made "\\n"}, or
        {path: "(cannot read: <why>)"} for a path that cannot be read or decoded, and
        for one that is not a regular file of at most tmpfile.READ_MAX_BYTE bytes
        (tmpfile.check_file).
    """
    show_dict: dict[str, str] = {}
    for rel_path in show_list or []:
        try:
            full_path = _resolve_in_workspace(root, rel_path)
            tmpfile.check_file(full_path)
            text, encoding = read_source(full_path)
            if not encoding:
                raise ValueError("none of the encodings tried decodes it")
            show_dict[rel_path] = cut(lf_text(text), source.read_limit())
        except (OSError, ValueError) as e:
            show_dict[rel_path] = f"(cannot read: {e})"
    return show_dict


def _run(arg_list: list[str], file_dict: dict[str, str] | None,
         env_dict: dict[str, str] | None, show_list: list[str] | None,
         reset: bool, slot: str, count: bool) -> dict:
    """Start config.RUN_COMMAND from the repository root and collect the result.

    An argument of the wrong type is refused before anything is written or started. The
    model's slot lists the first RUN_MAX_FILE files the run wrote and says how many more
    there are; the check's slot lists every one.

    The flow:
    1. Refuse a call that cannot run (_refuse_reason).
    2. Prepare the slot's workspace and write the sample files into it.
    3. Start the program, deleting what it wrote in the repository when it cannot
       start, times out or is stopped.
    4. Collect the files it produced, the files asked for (_show_dict) and its output.

    Args:
        arg_list: The arguments after the launcher.
        file_dict: {path relative to the workspace: contents} written before the run.
        env_dict: Extra environment variables for the run.
        show_list: Paths whose contents come back in "shown".
        reset: True empties this slot's workspace first.
        slot: MODEL_SLOT or CHECK_SLOT.
        count: Whether the run counts towards run_count().

    Returns:
        The dict documented in run_project's Returns.
    """
    global _run_count, _last_output

    # == Step 1: refuse a call that cannot run ===========================
    if isinstance(show_list, str):
        show_list = [show_list]
    refuse_reason = _refuse_reason(arg_list, file_dict, env_dict, show_list, count)
    if refuse_reason:
        return _result_dict(False, "", error=refuse_reason)

    # == Step 2: prepare the workspace ===================================
    if reset:
        _drop_workspace(slot)
    root = _ensure_workspace(slot)

    try:
        base_set = set(_list_workspace(root, set()))
        _write_file_list(root, file_dict or {})
    except (OSError, ValueError) as e:
        return _result_dict(False, root, error=f"cannot prepare the workspace: {e}")

    # == Step 3: start the program =======================================
    argv = edit.with_uv_project_word(
        config.RUN_COMMAND, LANG_DICT[config.PRIMARY_EXT].run_command_tuple,
        config.PROJECT_DIR) + [_fill(str(arg), root) for arg in arg_list]
    extra_env_dict = {name: _fill(str(value), root)
                      for name, value in (env_dict or {}).items()}

    if count:
        _run_count += 1
    # The package cache is asked for again when a dependency file changed; what came of
    # that goes in front of the output
    note = edit.refill_cache_on_dep_change()
    edit.install_dep()
    # What the repository looked like before the program ran; _clean_repo() compares
    # against it afterwards
    before_tuple = (_new_file_set(), _path_record())
    try:
        proc = edit.run_in_project(argv, config.RUN_TIMEOUT_SEC, extra_env_dict,
                                   is_machine_run=slot != MODEL_SLOT)
    except edit.StartError as e:
        return _result_dict(False, root, in_repo_list=_clean_repo(*before_tuple),
                            error=f"{argv[0]} cannot be started ({e}). Set RUN_COMMAND to "
                                  f"something runnable. A call that works here is:\n"
                                  + _call_example())
    except subprocess.TimeoutExpired:
        return _result_dict(False, root, in_repo_list=_clean_repo(*before_tuple),
                            error=f"timeout: over {config.RUN_TIMEOUT_SEC} seconds")
    except BaseException:
        # Any other exception: delete the files the run wrote, then raise it again
        _clean_repo(*before_tuple)
        raise

    # == Step 4: collect the result ======================================
    output_list = _list_workspace(root, base_set | set(file_dict or {}))
    list_count = config.RUN_MAX_FILE if slot == MODEL_SLOT else len(output_list)
    show_dict = _show_dict(root, show_list)

    output = note + cut_tail(proc.stdout + proc.stderr, config.RUN_OUTPUT_TAIL)
    if count:
        _last_output = output
    return _result_dict(proc.returncode == 0, root, returncode=proc.returncode,
                        output=output, output_list=output_list[:list_count],
                        more_count=len(output_list[list_count:]), show_dict=show_dict,
                        in_repo_list=_clean_repo(*before_tuple))
