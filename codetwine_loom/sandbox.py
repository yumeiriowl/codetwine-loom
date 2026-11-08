"""Run a change inside a container and put what it changed into the working tree.

A bare copy of the target repository's branches and objects is mounted read-only, the
container clones it somewhere writable, the change is made to the clone, and the change
comes back as a git patch. The container is given no mount of the host repository.

A run starts from the working tree as it is. Its files, committed or not, are written as
a commit of their own (gitcmd.snapshot_commit) that a branch named after the run points
at for as long as the run goes, and that commit is what the clone is checked out at.
What the run changed lands on a branch first, in a worktree, as a commit carrying what
the run recorded about itself; the change is then applied to the working tree, with git
checking that it applies before anything is written, and the commit is the user's to
make. The branches stay where a run did not get through, as the account a later run
carries on from, and where its change did not go into the working tree.

Several containers started against the same read-only repository and the same base
commit each produce a patch, and each one is taken in the same way. Their union is built
and checked by a container of its own, started here the same way and bounded the same
way, which merges the branches they left and hands back one more patch (integrate).

The container is started with no network, or only the internal one SANDBOX_NETWORK
names, and the credentials are not among the settings forwarded to it. A broker started
here, outside the container, holds the credentials and is reached over a Unix socket
(start_broker).

A container without the network is told to work from a package cache alone
(build_run_argv). warm() fills that cache in a container of its own with the network on,
no key of this tool and no model, and the cache is kept between runs. A broker started
with a repository fills the cache again when a run asks (start_broker).
"""

import collections
import concurrent.futures
import contextlib
import functools
import hashlib
import itertools
import json
import logging
import os
import re
import shlex
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime

from codetwine_loom import (SIGNAL_STATUS_BASE, broker, config, filelock, gitcmd, guide,
                            history, knowledge, llm, overview, progress, runrecord,
                            tmpfile, tokencount, turnlog)
from codetwine_loom import answer as answer_mod
from codetwine_loom import dispatch as dispatch_mod
from codetwine_loom import image as image_mod
from codetwine_loom import integrate as integrate_mod
from codetwine_loom.index import build, store
from codetwine_loom.languages import (ALIAS_DICT, DEP_FILE_TUPLE, LANG_DICT,
                                      OFFLINE_ENV_DICT, ext_of, is_dep_file,
                                      split_output_tuple)
from codetwine_loom.paths import HOST_PROJECT_ENV, list_text
from codetwine_loom.textcut import (CommandError, command_line, cut, error_part,
                                    one_line, with_command)
from codetwine_loom.tools import edit as edit_mod
from codetwine_loom.tools import web

logger = logging.getLogger(__name__)

# The engines looked for, in order, when CONTAINER_ENGINE is empty
ENGINE_TUPLE = ("podman", "docker")

# Where the container sees the repository, the clone, the output directory and the
# caches. _SRC and _LIVE are mounted read-only and the others writable; the image's own
# filesystem is read-only
_SRC = "/src"
_WORK = "/work"
_OUT = "/out"
_TMP = "/tmp"
_CACHE = "/cache"
_SOCK = "/run/ctloom"
_LIVE = "/live"

# The socket the container reaches the broker on, inside _SOCK
_SOCKET_NAME = "llm.sock"

# The scripts the image runs instead of its usual entry point: one clones the repository
# and writes back what the run changed as a patch, the other clones it and writes no
# patch
_ENTRYPOINT = "/usr/local/bin/ctloom-sandbox"
_WARM_ENTRYPOINT = "/usr/local/bin/ctloom-warm"

# Settings whose value is a path on this machine. They are left out of what is forwarded
# to the container
_HOST_ONLY_TUPLE = ("PROJECT_DIR", "OUTPUT_DIR", "SANDBOX_CACHE_DIR")

# The credentials, LLM_API_BASE, and where and how a search or a page read is made. They
# are left out of what is forwarded to the container
_BROKER_ONLY_TUPLE = config.SECRET_NAME_TUPLE

# Settings the container is given a value for on its command line. They are not
# forwarded by name as well
_SET_BY_HAND_TUPLE = ("BUILD_EXCLUDE", "LLM_BROKER", "TOOL_OFFLINE",
                      "WEB_SEARCH_MAX_CALL", "WEB_FETCH_MAX_CALL", "EDIT_HISTORY",
                      "EDIT_GUIDE")

# What the entry point is given the clone in. A process with it in its environment is
# the run inside a container (in_container)
CTLOOM_CLONE_ENV = "CTLOOM_WORK"

# The variable that gives a container filling the package cache the languages whose
# default test commands it runs as well, comma-separated (fill_ext_list)
CTLOOM_FILL_EXT_ENV = "CTLOOM_FILL_EXT"

# The variable that tells a container filling the package cache, with "0", to leave out
# the run of the index's own test commands (fill_own_run)
CTLOOM_FILL_OWN_ENV = "CTLOOM_FILL_OWN"

# The variable that tells a run that it fills the package cache, with "1" (fill_run)
CTLOOM_FILL_ENV = "CTLOOM_FILL"

# The variable that names to a run in a container the language it takes as the primary
# one in place of the index's (primary_ext_of_env)
CTLOOM_PRIMARY_EXT_ENV = "CTLOOM_PRIMARY_EXT"

# Settings of what this tool depends on rather than its own, forwarded by name to the
# analysis made in a container
_DEPENDENCY_ENV_TUPLE = ("EXCLUDE_PATTERNS",)

# Where a container with the engine's default network reads the file this machine's
# config.CA_FILE_ENV names
_CA_FILE = "/etc/ctloom-ca.pem"

# The variables the tools in that container read the path of that file from
_CA_ENV_TUPLE = (config.CA_FILE_ENV, "CURL_CA_BUNDLE", "GIT_SSL_CAINFO")

# Where a run's patch and its summary go, under <output_dir>/<project name>/. The name
# carries the run that wrote it
_PATCH_DIR = "sandbox"
_PATCH_PREFIX = "patch-"
_PATCH_SUFFIX = ".diff"

# The file the change put into the working tree is written to, in the run's directory
# on this machine's side (host_dir)
_LAND_PREFIX = "land-"

# Where the records of the commands that start containers are kept, in repo_own_dir()
_RECORD_DIR = "record"

# What the copy of the diff last put into the working tree is named after, in place of
# a run
_LAND_LAST_NAME = "last"

# Where a dispatching run's split goes, beside the patches
_TASK_PREFIX = "tasks-"
_TASK_SUFFIX = ".json"

# Where a run says how it ended, or what it listed, beside its patch. Same shape as the
# split above
_OUTCOME_PREFIX = "outcome-"

# Where an answering run puts its answer, and where the combining run is handed the
# agents' answers to put together. Same shape as the split above; the first is written
# by the container and read here, the second written here and read there
_ANSWER_PREFIX = "answer-"
_ANSWER_IN_PREFIX = "answered-"

# What one container of the stage that writes tests is handed: the files whose
# definitions it writes tests for and the baseline before the stage. Written here and
# read there, in the same shape as the split above
_TARGET_PREFIX = "targets-"

# What a run combining the test tasks is handed: the cases they recorded as not done,
# with the tests to run them again over the combined tests
_GAP_PREFIX = "gaps-"

# Where a container's stdout is kept (run_the_container), on this machine's side
# (host_dir)
_STDOUT_PREFIX = "stdout-"

# What the worktree a patch is taken in through is named with, on this machine's side
_WORKTREE_PREFIX = "worktree-"

# The record of one run this machine started under an agent, on this machine's side
# (host_dir): "run-<run>.txt", holding what the run left to pick up by hand, one of the
# two words below, or nothing
_RUN_MARK_PREFIX = "run-"
_RUN_MARK_SUFFIX = ".txt"
_PATCH_KEEP = "patch"
_ANSWER_KEEP = "answer"

# How many bytes one such record may hold to be read
_RUN_MARK_MAX_BYTE = 64

# What is added to a patch's name to get the summary beside it. sandbox/entrypoint.sh
# writes the file under the same name ("$CTLOOM_PATCH.meta.json")
_META_SUFFIX = ".meta.json"

# Where the clone and the run's temporary files are written, under _PATCH_DIR on this
# machine's side (host_dir). It is made before the container starts and taken away when
# it ends
_SCRATCH_PREFIX = "scratch-"

# Where warm() puts its clone, beside the one above, apart from the run's own scratch
# directory that the change's container holds while a refill runs
_WARM_SCRATCH_PREFIX = "scratch-warm-"

# The directory inside a scratch directory the container's clone source is copied to
# (make_clone_source)
_SOURCE_DIR = "src"

# The directory inside a filling's scratch directory the dependency files of a commit
# are written into
_DEP_DIR = "dep"

# What the mode of a regular file starts with in git's tree listing
_REGULAR_MODE_PREFIX = "100"

# Where the toolchain's package cache is written, in repo_own_dir(). Unlike the scratch
# directory it is kept between runs
_CACHE_DIR = "cache"

# What the cache's path is followed by in the name of the file warm() keeps beside the
# cache (fill_mark_path), one line per set of dependency files and image a filling has
# run through for (_dep_fingerprint()). A filling whose line is in it starts no
# container unless it is asked to fill again
_CACHE_FILL_MARK = ".filled"

# The first part of the name of every branch a run makes
_BRANCH_ROOT = "codetwine-loom"

# The prefix of the branch a taken-in patch is committed to
_BRANCH_PREFIX = f"{_BRANCH_ROOT}/sandbox/"

# The branch that points at the commit a run starts from (gitcmd.snapshot_commit) for as
# long as the run goes, which the clone the container makes carries. It is under the
# run's own root
_TREE_BRANCH = _BRANCH_ROOT + "/run/{run_id}/tree"

# What the branch of one dispatched task is called. The run is in the name: a second run
# over the same repository never lands on the branches the first one left
_TASK_BRANCH = _BRANCH_ROOT + "/run/{run_id}/{n}"

# What is added to that name when a task whose agent did not finish is handed out again.
# The try gets a branch of its own
_RETRY_SUFFIX = "-retry{attempt}"

# Where all of those live, read off the name itself. The branch a union lands on is not
# under it
_TASK_BRANCH_ROOT = _TASK_BRANCH.split("/{run_id}/", 1)[0]

# The names the branches of a run are made under. git cannot make a branch under a name
# that is itself a branch
_BRANCH_ROOT_TUPLE = (_BRANCH_ROOT, _TASK_BRANCH_ROOT, _BRANCH_PREFIX.rstrip("/"))

# What a run this tool named looks like inside a name (new_run_id). Branches and output
# directories both carry it - "codetwine-loom/run/<run>/2",
# "codetwine-loom-run-<run>-2", "<run>" - and it names the run each of them belongs to
_RUN_IN_NAME = re.compile(r"\d{8}-\d{6}-\d+-\d+", re.ASCII)

# The prefix of the name the run's container answers to
_CONTAINER_PREFIX = "ctloom-"

# The exit status a container stopped by its time limit is reported with, the one
# timeout(1) uses for the same thing
_TIME_OUT_STATUS = 124

# The exit status a container killed by SIGKILL is reported with (128 + 9). It is what a
# container whose processes reach SANDBOX_MEMORY ends with
_KILL_STATUS = 137

# The highest exit status a process ends with
_MAX_EXIT_STATUS = 255

# The memory one job of a build in a container is given: CMake's and cargo's parallel
# jobs are SANDBOX_MEMORY over this, and 1 at least (_build_job_count)
_BUILD_JOB_BYTE = 1024 ** 3


def _build_job_count() -> int:
    """Return how many jobs a build in a container runs at once (_BUILD_JOB_BYTE)."""
    return max(1, config.SANDBOX_MEMORY_BYTE // _BUILD_JOB_BYTE)


def _is_memory_kill(result_dict: dict) -> bool:
    """Say whether a run ended with a container killed at SANDBOX_MEMORY.

    Args:
        result_dict: The run's result, with "exit_status" and, for a stage after the
            merge or one of its test tasks, "memory_kill".

    Returns:
        True when "exit_status" is _KILL_STATUS or "memory_kill" is set.
    """
    return result_dict.get("exit_status") == _KILL_STATUS \
        or bool(result_dict.get("memory_kill"))


# How many bytes a patch a container wrote may hold to be taken in
_MAX_PATCH_BYTE = 1 << 30

# How many bytes a JSON file a container wrote beside its patch for this machine may
# hold to be read
_MAX_ACROSS_BYTE = 4 * 1024 * 1024

# How many bytes git may write when it lists the files of a running change's copy
_MAX_LIST_BYTE = 32 * 1024 * 1024

# Where the agents' own copies of the output directory go, under it
_AGENT_DIR = knowledge.AGENT_DIR

# Where, under _AGENT_DIR, this machine keeps what it alone writes for each agent
# (host_dir). No container is given this directory or one holding it. No agent is named
# this: a name agent_name() returns starts with no dot
_HOST_DIR = ".host"

# Where, under _AGENT_DIR, this machine keeps this tool's own files of each repository
# (repo_own_dir): the package cache, the records of the commands, the copy of the diff
# last put into the working tree and the copies of what a diff's paths held before it
# went in. No container is given this directory or one holding it; every container is
# given the package cache in it. No agent is named this
_REPO_DIR = ".repo"

# What is not part of that copy: what this tool writes for itself, directly under
# <output_dir>/<project name>/. The file is HISTORY_FILE. Each directory named here
# holds this tool's own files and, beside them, codetwine's analysis of the directory of
# that name in the repository: every file directly in it is this tool's, and every
# directory in it is codetwine's. What is this tool's is not copied into an agent's
# directory and is what is left behind when the copy is taken away again
_RUN_OWN_FILE = history.HISTORY_FILE
_RUN_OWN_DIR_TUPLE = (_PATCH_DIR, turnlog.TURN_LOG_DIR)

# How much of the request the commit's subject line keeps
_MAX_SUBJECT_CHAR = 72

# The name every trailer this writes starts with, which tells what the tool put in a
# commit apart from the trailers whoever else touches the branch writes. It is decided
# in integrate.py
_TRAILER_PREFIX = integrate_mod.TRAILER_PREFIX


def pick_engine(name: str = "") -> str:
    """Return the container engine to start, looked up on PATH.

    Args:
        name: The engine asked for. Empty falls back to config.CONTAINER_ENGINE, and an
            empty setting tries the known engines in order.

    Returns:
        The path to the engine's executable.

    Raises:
        RuntimeError: When the engine asked for is not on PATH, or when none of the
            known engines is.
    """
    want = name or config.CONTAINER_ENGINE
    if want:
        engine_path = shutil.which(want)
        if not engine_path:
            raise RuntimeError(f"{want} is not on PATH. Install it, or set "
                               f"CONTAINER_ENGINE to an engine that is")
        return engine_path
    for candidate in ENGINE_TUPLE:
        engine_path = shutil.which(candidate)
        if engine_path:
            return engine_path
    raise RuntimeError(f"none of {', '.join(ENGINE_TUPLE)} is on PATH. Install one, or set "
                       f"CONTAINER_ENGINE to the engine to use")


@dataclass(frozen=True)
class MountLayout:
    """The directories on this machine one container is given.

    Attributes:
        src: The copy of the target repository's git history the container clones
            (make_clone_source). It is mounted read-only.
        out: The agent's own copy of the output directory (agent_dir), holding the
            index, the change history and the patch.
        scratch: Where the clone and the run's temporary files go, on this machine's
            side (host_dir). It holds "work" and "tmp", which are what is mounted, and
            is taken away when the run ends.
        sock: The directory holding the broker's socket. Empty leaves the mount out and
            passes LLM_BROKER empty.
        cache: Where the toolchain's package cache goes. It holds the directories
            _cache_env_dict() names and is kept between runs.
        live: The directory the dependency files are taken from, mounted read-only; they
            are read from the directory named after the repository in it. Empty leaves
            the mount out.
    """

    src: str
    out: str
    scratch: str
    cache: str
    sock: str = ""
    live: str = ""


def env_forward_argv() -> list[str]:
    """Return the -e arguments that forward this machine's settings to the container.

    The names are those config.env_name_tuple() returns. A name with no value on this
    machine is passed as a bare name, which both engines drop, and an unset setting
    keeps its default inside the container.

    Returns:
        ["-e", "NAME", "-e", "NAME", ...] for every setting but the credentials,
        LLM_API_BASE and where and how a search or a page read is made
        (_BROKER_ONLY_TUPLE), those naming a path on this machine (_HOST_ONLY_TUPLE),
        and those given a value on the command line (_SET_BY_HAND_TUPLE), followed by
        the settings of what this tool depends on.
    """
    argv = []
    for name in config.env_name_tuple():
        if name in _HOST_ONLY_TUPLE or name in _SET_BY_HAND_TUPLE or name in _BROKER_ONLY_TUPLE:
            continue
        argv += ["-e", name]
    for name in _DEPENDENCY_ENV_TUPLE:
        argv += ["-e", name]
    return argv


def build_run_argv(engine: str, image: str, mount: MountLayout, base: str, name: str,
                   change_argv: list[str], run_id: str = "",
                   rebuild_index: bool = False, exclude_list: list[str] | None = None,
                   network: bool = False,
                   entrypoint: str = _ENTRYPOINT,
                   live_dep_path_list: list[str] | None = None,
                   build_output_tuple: tuple[str, ...] = (),
                   fill_ext_list: Sequence[str] = (),
                   fill_own: bool = True, primary_ext: str = "",
                   host_project: str = "") -> list[str]:
    """Build the command line that starts one sandboxed run.

    The container gets the copy of the repository's history read-only, the agent's copy
    of the output directory, a scratch directory for the clone, the package cache, the
    broker's socket given mount.sock, and the directory the dependency files are taken
    from read-only given mount.live. The image's own filesystem is read-only and every
    capability is dropped. Without network, the container has no network but the one
    SANDBOX_NETWORK names, and this machine's proxy variables are kept out of it
    (_no_host_proxy_argv); with it, the container is given them, with the file of
    certificate authorities (_route_argv). The process and memory limits come from the
    settings. The engine's init process runs first and reaps the processes the run
    leaves behind.

    The scratch directory is a directory on this machine, not a temporary filesystem,
    and is not charged to the container's memory limit.

    Args:
        engine: The engine's executable, as pick_engine() returns it.
        image: The image to run.
        mount: The directories on this machine the container is given, as absolute
            paths. The directories inside them must exist before the container starts.
        base: The commit the clone is checked out at.
        name: The repository's directory name. The clone is made at /work/<name>, and
            the container's output goes under the same /out/<name>/ the index built on
            this machine uses.
        change_argv: What follows the image, i.e. the subcommand and its arguments.
        run_id: The run, as new_run_id() names it. It names the container and the patch.
            Empty names neither.
        rebuild_index: Whether the container rebuilds the index over the clone before
            the change is made. It calls no LLM and keeps the design documents that are
            already in the output directory.
        exclude_list: The directory and file names kept out of that rebuild, passed as
            BUILD_EXCLUDE.
        network: Whether the container is given the engine's default network. False
            gives no network, or the one SANDBOX_NETWORK names, and tells the build
            tools to work from the package cache alone.
        entrypoint: The script in the image the container runs. _ENTRYPOINT writes back
            what the run changed as a patch; _WARM_ENTRYPOINT writes no patch.
        live_dep_path_list: The dependency files taken from mount.live when it is given,
            relative to the directory named after the repository in it.
        build_output_tuple: The primary language's build_output_tuple entries that do
            not start with "/", as .gitignore patterns (build_output_tuple_of()), which
            the clone's git leaves out of what the run changed.
        fill_ext_list: The languages whose default test commands a container that fills
            the package cache runs as well, passed as CTLOOM_FILL_EXT_ENV. Empty passes
            nothing.
        fill_own: Whether that container runs the index's own test commands. False
            passes CTLOOM_FILL_OWN_ENV as "0".
        primary_ext: The language the run takes as the primary one, passed as
            CTLOOM_PRIMARY_EXT_ENV. Empty passes nothing.
        host_project: The target repository on this machine, which the run is told
            of (HOST_PROJECT_ENV) as an absolute path and with its links resolved.
            Empty tells it nothing.

    Returns:
        The full argument list, engine first.
    """
    work = f"{_WORK}/{name}"
    argv = [
        engine, "run", "--rm", "--init",
        "-v", f"{mount.src}:{_SRC}:ro",
        "-v", f"{mount.out}:{_OUT}",
        "-v", f"{os.path.join(mount.scratch, 'work')}:{_WORK}",
        "-v", f"{os.path.join(mount.scratch, 'tmp')}:{_TMP}",
        "-v", f"{mount.cache}:{_CACHE}",
    ]
    if mount.sock:
        argv += ["-v", f"{mount.sock}:{_SOCK}"]
    if mount.live:
        argv += ["-v", f"{mount.live}:{_LIVE}:ro"]
    if network:
        argv += _route_argv()
    else:
        argv.append(f"--network={config.SANDBOX_NETWORK or 'none'}")
        argv += _no_host_proxy_argv(engine)
    argv += [
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit", str(config.SANDBOX_MAX_PROCESS),
        "--memory", str(config.SANDBOX_MEMORY),
    ]
    if run_id:
        argv += ["--name", container_name(run_id)]
    argv += _engine_user_argv(engine)
    argv += [
        "-e", f"CTLOOM_SRC={_SRC}",
        "-e", f"{CTLOOM_CLONE_ENV}={work}",
        "-e", f"CTLOOM_BASE={base}",
        "-e", f"CTLOOM_OUT={_OUT}",
        "-e", f"CTLOOM_REBUILD_INDEX={'1' if rebuild_index else '0'}",
        "-e", f"CTLOOM_FORMAT={'1' if config.FORMAT_COMMAND else '0'}",
        # What the primary language's tools write, which the clone's git leaves out of
        # what the run changed
        "-e", f"CTLOOM_BUILD_OUTPUT={chr(10).join(build_output_tuple)}",
        # Where the run is started from, outside the clone: a relative path the model
        # writes outside a tool lands here and goes away with the scratch
        "-e", f"CTLOOM_RUNDIR={_TMP}",
        "-e", f"{progress.RUN_START_ENV}={progress.run_start()}",
        # The parallel jobs of a CMake build and of cargo (_build_job_count)
        "-e", f"CMAKE_BUILD_PARALLEL_LEVEL={_build_job_count()}",
        "-e", f"CARGO_BUILD_JOBS={_build_job_count()}",
    ]
    # The state the run started from, which the overview in the copy was made for
    if os.environ.get(overview.RUN_DIGEST_ENV):
        argv += ["-e", f"{overview.RUN_DIGEST_ENV}={os.environ[overview.RUN_DIGEST_ENV]}"]
    # The same file name this machine reads the patch back from, in the container's own
    # paths. The entry point that fills the cache writes no patch and is given no name
    # for one, however the container itself is named
    if run_id and entrypoint == _ENTRYPOINT:
        argv += ["-e",
                 f"CTLOOM_PATCH={_OUT}/{name}/{_PATCH_DIR}/{patch_name(run_id)}"]
    if fill_ext_list:
        argv += ["-e", f"{CTLOOM_FILL_EXT_ENV}={','.join(fill_ext_list)}"]
    if not fill_own:
        argv += ["-e", f"{CTLOOM_FILL_OWN_ENV}=0"]
    if primary_ext:
        argv += ["-e", f"{CTLOOM_PRIMARY_EXT_ENV}={primary_ext}"]
    if host_project:
        place_list = dict.fromkeys([os.path.abspath(host_project),
                                    os.path.realpath(host_project)])
        argv += ["-e", f"{HOST_PROJECT_ENV}={chr(10).join(place_list)}"]
    # HOME, the tools' caches and Maven's repository point into the cache mount, and
    # Gradle's home into the run's temporary directory
    argv += [arg for env_name, value in {**_cache_env_dict(_CACHE),
                                         **_jvm_tool_env_dict(_CACHE, _TMP)}.items()
             for arg in ("-e", f"{env_name}={value}")]
    # A container with no network has the build tools work from that cache alone. The
    # settings go in for the tool that has one; TOOL_OFFLINE is for a tool that takes it
    # on the command line
    argv += ["-e", f"TOOL_OFFLINE={'false' if network else 'true'}"]
    if not network:
        argv += [arg for env_name, value in OFFLINE_ENV_DICT.items()
                 for arg in ("-e", f"{env_name}={value}")]
    argv += env_forward_argv()
    # Written out, not forwarded by name
    argv += ["-e", f"BUILD_EXCLUDE={shlex.join(exclude_list or [])}"]
    # Likewise: the socket's address as the container sees it, or empty when no socket
    # is mounted
    argv += ["-e", f"LLM_BROKER={socket_url(_SOCK) if mount.sock else ''}"]
    # The budgets of the two web tools, as 0 where web.available() or
    # web.fetch_available() is False
    argv += ["-e", f"WEB_SEARCH_MAX_CALL="
                   f"{config.WEB_SEARCH_MAX_CALL if web.available() else 0}",
             "-e", f"WEB_FETCH_MAX_CALL="
                   f"{config.WEB_FETCH_MAX_CALL if web.fetch_available() else 0}"]
    # A change made inside is recorded whatever EDIT_HISTORY is here
    argv += ["-e", "EDIT_HISTORY=true"]
    # The dependency files taken from mount.live, which the entry point copies over the
    # clone
    if mount.live:
        argv += ["-e", f"CTLOOM_LIVE={_LIVE}/{name}",
                 "-e", f"CTLOOM_DEP_PATH={chr(10).join(live_dep_path_list or [])}"]
    argv += ["--entrypoint", entrypoint, image]
    return argv + change_argv


def _cache_env_dict(root: str) -> dict:
    """Return the variables pointing HOME, the XDG cache and uv's cache under root.

    Examples:
        "/cache" -> {"HOME": "/cache/home", ...}

    Args:
        root: The cache directory, as the process reading it sees it.

    Returns:
        {variable name: the directory it points at}.
    """
    return {"HOME": os.path.join(root, "home"),
            "XDG_CACHE_HOME": os.path.join(root, "xdg"),
            "UV_CACHE_DIR": os.path.join(root, "uv")}


def _jvm_tool_env_dict(cache: str, tmp: str) -> dict:
    """Return the variables that tell the JVM build tools where they keep things.

    The tools find their home from the directory the system names for the user, whatever
    HOME holds. Maven is handed its repository of fetched artifacts under the home
    directory _cache_env_dict() names. Gradle is handed a home in the run's own
    temporary directory, and the directory of the cache it reads fetched dependencies
    from without writing to it.

    Examples:
        ("/cache", "/tmp")
            -> {"MAVEN_OPTS": "-Dmaven.repo.local=/cache/home/.m2/repository",
                "GRADLE_USER_HOME": "/tmp/gradle-home",
                "GRADLE_RO_DEP_CACHE": "/cache/gradle"}

    Args:
        cache: The cache directory, as the process reading it sees it.
        tmp: The directory of the run's temporary files, as that process sees it.

    Returns:
        {variable name: the options or the directory it holds}.
    """
    repository = os.path.join(_cache_env_dict(cache)["HOME"], ".m2", "repository")
    return {"MAVEN_OPTS": f"-Dmaven.repo.local={repository}",
            "GRADLE_USER_HOME": os.path.join(tmp, "gradle-home"),
            "GRADLE_RO_DEP_CACHE": os.path.join(cache, "gradle")}


def socket_url(directory: str) -> str:
    """Return the address of the broker's socket in a directory.

    Examples:
        "/run/ctloom" -> "unix:///run/ctloom/llm.sock"

    Args:
        directory: The directory the socket sits in.

    Returns:
        The address, in the form config.LLM_BROKER takes.
    """
    return f"{broker.UNIX_SCHEME}{os.path.join(directory, _SOCKET_NAME)}"


def _no_host_proxy_argv(engine: str) -> list[str]:
    """Return the arguments that keep this machine's proxy variables out of the container.

    podman copies HTTP_PROXY, HTTPS_PROXY and the like from its own environment into
    every container unless it is told not to. An engine that is not podman is given
    nothing.

    Args:
        engine: The engine's executable, as pick_engine() returns it.

    Returns:
        The arguments, or an empty list.
    """
    return ["--http-proxy=false"] if os.path.basename(engine) == "podman" else []


def refuse_ca_file_path() -> None:
    """Stop when the file config.CA_FILE_ENV names cannot be mounted in a container.

    Raises:
        RuntimeError: Naming the file, when its path holds a ":", which ends a path
            in the engine's mount argument.
    """
    ca_file = os.environ.get(config.CA_FILE_ENV, "")
    if os.path.isfile(ca_file) and config.ca_file_path_error(os.path.abspath(ca_file)):
        raise RuntimeError(
            f"{config.CA_FILE_ENV} names {os.path.abspath(ca_file)}, and the container "
            f"that fetches dependencies cannot be given a file whose path holds a "
            f"':'. Move the file and set {config.CA_FILE_ENV} to where it is")


def _has_user_info(value: str) -> bool:
    """Say whether a proxy variable's value carries a user or a password.

    Examples:
        "http://me:pw@proxy.corp.example:3128"     -> True
        "me@proxy.corp.example:3128"               -> True
        "http://me:pa#ss@proxy.corp.example:3128"  -> True
        "http://proxy.corp.example:3128"           -> False

    Args:
        value: The value of one proxy variable.

    Returns:
        True when the value holds an "@", before which an HTTP client reads a user
        and a password, whether the rest can be read as a URL or not.
    """
    return "@" in value


def _route_argv() -> list[str]:
    """Return the arguments that give a container this machine's proxy variables.

    The file of certificate authorities goes with them.

    Examples:
        HTTPS_PROXY and no_proxy set, SSL_CERT_FILE=/home/u/ca.pem
            -> ["-e", "HTTPS_PROXY", "-e", "no_proxy",
                "-v", "/home/u/ca.pem:/etc/ctloom-ca.pem:ro",
                "-e", "SSL_CERT_FILE=/etc/ctloom-ca.pem",
                "-e", "CURL_CA_BUNDLE=/etc/ctloom-ca.pem",
                "-e", "GIT_SSL_CAINFO=/etc/ctloom-ca.pem"]

    Returns:
        "-e" and the name for each proxy variable this process holds
        (config.proxy_name_list()), which the engine fills from the environment it is
        started with. When config.CA_FILE_ENV names a file, they are followed by the
        file's read-only mount at _CA_FILE and by that path under each name of
        _CA_ENV_TUPLE. A config.CA_FILE_ENV that names no file is logged and adds
        nothing. A proxy variable whose value carries a user or a password is logged
        by name.

    Raises:
        RuntimeError: When the path of the file config.CA_FILE_ENV names holds a ":",
            which ends a path in the engine's mount argument. The message does not hold
            the path.
    """
    name_list = config.proxy_name_list()
    argv = [arg for name in name_list for arg in ("-e", name)]
    secret_name_list = [name for name in name_list if _has_user_info(os.environ[name])]
    if secret_name_list:
        logger.warning("%s carries a user or a password, and the container that "
                       "fetches dependencies is given it: the build files of the "
                       "repository run there with a network",
                       list_text(secret_name_list))
    ca_file = os.environ.get(config.CA_FILE_ENV, "")
    if not ca_file:
        return argv
    if not os.path.isfile(ca_file):
        logger.warning("%s names %s, which is not a file, so the container that "
                       "fetches dependencies is given no file of certificate "
                       "authorities", config.CA_FILE_ENV, ca_file)
        return argv
    ca_file = os.path.abspath(ca_file)
    if ":" in ca_file:
        raise RuntimeError(
            f"the file {config.CA_FILE_ENV} names cannot be given to the container "
            f"that fetches dependencies: its path holds a ':'")
    argv += ["-v", f"{ca_file}:{_CA_FILE}:ro"]
    return argv + [arg for name in _CA_ENV_TUPLE for arg in ("-e", f"{name}={_CA_FILE}")]


def _engine_user_argv(engine: str) -> list[str]:
    """Return the arguments that make the container write as the user running it.

    The two engines ask for it differently, and an engine that is neither is left alone.

    Args:
        engine: The engine's executable, as pick_engine() returns it.

    Returns:
        The arguments, or an empty list.
    """
    basename = os.path.basename(engine)
    getuid = getattr(os, "getuid", None)
    if basename == "podman":
        return ["--userns=keep-id"]
    if basename == "docker" and getuid is not None:
        return ["--user", f"{getuid()}:{os.getgid()}"]
    return []


def edit_argv(name: str, request: str, dry_run: bool = False,
              guide: str = "", check: bool = False) -> list[str]:
    """Build the edit subcommand the container runs, in the container's own paths.

    --no-resume is always passed.

    Args:
        name: The repository's directory name, the same one build_run_argv() is given.
        request: The change request.
        dry_run: Whether the run makes its checks and writes nothing.
        guide: The file saying how the repository is to be changed. Empty reads
            none. It goes on the command line whatever it holds, as dry_run does, and
            EDIT_GUIDE is not forwarded by name.
        check: Whether the run inside holds its change to the static check and the
            references when it ends. It goes on the command line as dry_run does.

    Returns:
        The argument list that follows the image.
    """
    return ["edit", "--project", f"{_WORK}/{name}", "--output-dir", _OUT,
            f"--request={request}", "--no-resume",
            "--dry-run" if dry_run else "--no-dry-run", f"--guide={guide}",
            "--check" if check else "--no-check"]


def integrate_argv(name: str, branch_list: list[str], onto: str = "",
                   outcome_out: str = "", resume: str = "", onto_commit: str = "",
                   request: str = "", gap_in: str = "") -> list[str]:
    """Build the integrate subcommand the container runs, in the container's own paths.

    --here tells the run inside the container to merge the branches in the clone rather
    than start a container of its own.

    Args:
        name: The repository's directory name, the same one build_run_argv() is given.
        branch_list: The branches to combine, in the order they are to be merged.
        onto: The branch the union is written against, which the run combines along with
            the others: the one pointing at what the working tree holds. "" writes the
            union against the commit the branches start from.
        outcome_out: Where inside the container the run says how it ended. Empty asks
            for nothing to be written.
        resume: The branch an earlier merge left, which the run combines first and reads
            the record of. "" passes none.
        onto_commit: The commit of onto this machine read before the container was
            started, which is what the union is written against inside it.
        request: The request the branches were made for. "" passes none.
        gap_in: Where inside the container the cases whose tests are run again over the
            settled merge are. "" passes none.

    Returns:
        The argument list that follows the image.
    """
    argv = ["integrate", "--here", "--project", f"{_WORK}/{name}", "--output-dir", _OUT]
    if resume:
        argv += ["--resume", resume]
    for branch in branch_list:
        argv += ["--branch", branch]
    if onto:
        argv += ["--onto", onto]
    if onto_commit:
        argv += ["--onto-commit", onto_commit]
    if request:
        argv += [f"--request={request}"]
    if outcome_out:
        argv += ["--outcome-out", outcome_out]
    if gap_in:
        argv += ["--gap-in", gap_in]
    return argv


def test_create_argv(name: str, base: str, request: str = "", outcome_out: str = "",
                     target_in: str = "", target_only: bool = False) -> list[str]:
    """Build the test-create subcommand the container runs, in the container's own paths.

    Args:
        name: The repository's directory name, the same one build_run_argv() is given.
        base: The commit the change the clone is checked out at was written against.
        request: The request the change was made for. "" passes none.
        outcome_out: Where inside the container the run says how it ended, or what it
            listed. Empty asks for nothing to be written.
        target_in: Where inside the container the files this run writes tests for are
            named. Empty takes every file the change touched.
        target_only: Whether the run lists what there is to write tests for and ends,
            calling no model.

    Returns:
        The argument list that follows the image.
    """
    argv = ["test-create", "--project", f"{_WORK}/{name}", "--output-dir", _OUT,
            "--base", base]
    if request:
        argv += [f"--request={request}"]
    if outcome_out:
        argv += ["--outcome-out", outcome_out]
    if target_in:
        argv += ["--target-in", target_in]
    if target_only:
        argv += ["--target-only"]
    return argv


def refactor_argv(name: str, stage: str, base: str, request: str = "",
                  outcome_out: str = "") -> list[str]:
    """Build the subcommand of a stage that refactors, in the container's own paths.

    Args:
        name: The repository's directory name, the same one build_run_argv() is given.
        stage: One of refactor.STAGE_TUPLE, which is the subcommand's name.
        base: The commit the change the container's HEAD holds was written against.
        request: The request the change was made for. "" leaves it out.
        outcome_out: Where the run writes what it made of its run, as a path inside
            the container. "" writes nothing.

    Returns:
        The argument list that follows the image.
    """
    argv = [stage, "--project", f"{_WORK}/{name}", "--output-dir", _OUT, "--base", base]
    if request:
        argv += [f"--request={request}"]
    if outcome_out:
        argv += ["--outcome-out", outcome_out]
    return argv


def answer_argv(name: str, question: str, answer_out: str,
                answer_in_path: str = "") -> list[str]:
    """Build the answer subcommand the container runs, in the container's own paths.

    Args:
        name: The repository's directory name, the same one build_run_argv() is given.
        question: The question, or the request the agents' answers are put together
            for.
        answer_out: Where inside the container the answer is written.
        answer_in_path: Where inside the container the agents' answers are, for the
            combining run. Empty makes it an answering run.

    Returns:
        The argument list that follows the image.
    """
    argv = ["answer", "--project", f"{_WORK}/{name}", "--output-dir", _OUT,
            f"--request={question}", "--answer-out", answer_out]
    if answer_in_path:
        argv += ["--answered", answer_in_path]
    return argv


def dispatch_argv(name: str, request: str, task_out: str,
                  resume_list: Sequence[str] = ()) -> list[str]:
    """Build the run subcommand the container runs, in the container's own paths.

    --here tells the run inside the container to read the request itself rather than
    start a container of its own.

    Args:
        name: The repository's directory name, the same one build_run_argv() is given.
        request: The request, as a question or as a change.
        task_out: Where inside the container the split is written.
        resume_list: The branches an earlier run left its work on. They are read in the
            clone, which the branches of what it was cloned from reach.

    Returns:
        The argument list that follows the image.
    """
    argv = ["run", "--here", "--project", f"{_WORK}/{name}",
            "--output-dir", _OUT, f"--request={request}", "--tasks-out", task_out]
    for branch in resume_list:
        argv += ["--resume", branch]
    return argv


@dataclass(frozen=True)
class RunFileLayout:
    """Where one run's own files go.

    They are in two directories. What the container reads or writes is under the agent's
    own copy of the output directory, which is what the container is given; what this
    machine alone writes is under host_dir(), which no container is given. Every name
    carries the run it belongs to, and several runs of one agent share the two.

    Attributes:
        out_dir: The agent's own copy of the output directory (agent_dir).
        directory: <out_dir>/<project name>/sandbox, which holds the files below up to
            gap.
        host_directory: The directory of that name on this machine's side, which holds
            stdout, scratch, warm_scratch, worktree, land and mark. The lock of the
            agent's runs is taken on it.
        patch: The patch the container writes, with its summary beside it.
        task: The split a dispatching run writes.
        task_inside: That same file as the container sees it, which is what the
            container is told to write. Both paths are built by one rule here.
        outcome: What a run says about how it ended, or what it listed.
        outcome_inside: That same file as the container sees it.
        answer: The answer an answering or a combining run hands back.
        answer_inside: That same file as the container sees it.
        answer_in: The agents' answers a combining run is handed, written here before
            the container starts.
        answer_in_inside: That same file as the container sees it.
        target: What one container of the stage that writes tests is handed, written
            here before the container starts.
        target_inside: That same file as the container sees it.
        gap: The cases a run combining the test tasks runs the tests of again, written
            here before the container starts.
        gap_inside: That same file as the container sees it.
        stdout: The file the container's stdout is kept in (run_the_container).
        scratch: The clone and the run's temporary files. It is taken away when the run
            ends.
        warm_scratch: The clone a refill makes while the run still holds scratch.
        worktree: Where the patch is applied while it is taken in on a branch.
        land: The diff between what the run started from and what it landed, which is
            what is applied to the working tree.
        mark: The record that this machine started the run, and of what the run left to
            pick up by hand (_write_run_mark).
    """

    out_dir: str
    directory: str
    host_directory: str
    patch: str
    task: str
    task_inside: str
    outcome: str
    outcome_inside: str
    answer: str
    answer_inside: str
    answer_in: str
    answer_in_inside: str
    target: str
    target_inside: str
    gap: str
    gap_inside: str
    stdout: str
    scratch: str
    warm_scratch: str
    worktree: str
    land: str
    mark: str


# How many runs this process has started. next() on it hands out a number of its own
# each time
_RUN_COUNT = itertools.count(1)


def _run_place(project: str, output_dir: str) -> tuple[str, str, str]:
    """Resolve where a run works.

    A setting of guide.SETTING_TUPLE that names a path outside the target repository
    (guide.is_outside()) is warned about, once per process for each setting and value
    (_warn_absolute_guide).

    Args:
        project: The target repository, as it was passed.
        output_dir: The output directory, as it was passed.

    Returns:
        (the repository as an absolute path, the output directory as an absolute
        path, the repository's directory name).

    Raises:
        RuntimeError: When the output directory is inside the repository or holds it
            (knowledge.check_output_place), or when the repository has a branch named
            after one of _BRANCH_ROOT_TUPLE, under which the run's branches cannot be
            made.
    """
    project_dir = os.path.abspath(project)
    knowledge.check_output_place(project_dir, os.path.abspath(output_dir))
    for setting in guide.SETTING_TUPLE:
        if guide.is_outside(getattr(config, setting)):
            _warn_absolute_guide(setting, getattr(config, setting))
    for root in _BRANCH_ROOT_TUPLE:
        if gitcmd.run(project_dir, "show-ref", "--verify", "--quiet",
                      f"refs/heads/{root}").returncode == 0:
            raise RuntimeError(
                f"{project_dir} has a branch named {root}, and the branches a run makes "
                f"are named {root}/..., which git cannot make while that branch is "
                f"there. Rename it ({command_line('git', '-C', project_dir, 'branch', '-m', root)} "
                f"<new name>) and run again")
    return project_dir, os.path.abspath(output_dir), knowledge.project_name_of(project)


@functools.cache
def _warn_absolute_guide(setting: str, value: str) -> None:
    """Say once per process that an instruction file is out of the containers' reach.

    Args:
        setting: The setting that names the file.
        value: Its value, a path outside the target repository (guide.is_outside()).
    """
    logger.warning("%s names %s, a path outside the target repository, which a "
                   "container cannot read, so the stages it is for run without it. Put "
                   "the file in the target repository and name it by its path from "
                   "there", setting, value)


def new_run_id() -> str:
    """Return a name that tells one run's files from every other run's.

    Besides the time, two things tell runs apart: the process, and a count of the runs
    this process has started, which tells apart the runs one process starts at once.

    The process stays last.

    Returns:
        "<the time this was called>-<the count>-<this process>".
    """
    return (f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{next(_RUN_COUNT)}"
            f"-{os.getpid()}")


def agent_name(branch: str, run_id: str) -> str:
    """Return the name the agent making this change is known by.

    An agent is the branch it commits to, which is what its own place in the output
    directory is named after. A run with no branch of its own is an agent of its own.

    Examples:
        "codetwine-loom/run/20260101-000000-1-42/2"
            -> "codetwine-loom-run-20260101-000000-1-42-2"

    Args:
        branch: The branch the patch is committed to. Empty names the agent after the
            run.
        run_id: The run, as new_run_id() names it.

    Returns:
        A name that can also be a directory. It starts with no dot: a run id starts
        with a digit, and git has no branch whose name starts with a dot.
    """
    return "".join(char if char.isalnum() or char in "-_." else "-"
                   for char in branch or run_id)


def agent_dir(out_dir: str, agent: str) -> str:
    """Return where one agent's own copy of the output directory goes.

    Args:
        out_dir: The output directory, as an absolute path.
        agent: The agent, as agent_name() names it.

    Returns:
        The path, as an absolute path.
    """
    return os.path.join(out_dir, _AGENT_DIR, agent)


def host_dir(out_dir: str, agent: str) -> str:
    """Return where this machine keeps what it alone writes for one agent.

    The directory is laid out as the agent's own copy of the output directory is:
    <project name>/sandbox/ holds the clones, the diffs put into the working tree, the
    containers' stdout, the worktrees patches are taken in through and the records of
    the runs started for the agent (_write_run_mark), and <project name>/turns/ the
    broker's log. No container is given it; the "src", "work" and "tmp" directories of a
    clone are what a container is given of it.

    Args:
        out_dir: The output directory, as an absolute path.
        agent: The agent, as agent_name() names it.

    Returns:
        The path, as an absolute path.
    """
    return os.path.join(out_dir, _AGENT_DIR, _HOST_DIR, agent)


def repo_own_dir(out_dir: str, name: str) -> str:
    """Return where this machine keeps this tool's own files of one repository.

    It holds the package cache (cache_dir), the records of the commands
    (record_dir), the copy of the diff last put into the working tree
    (last_land_path) and, beside it, the copies of what a diff's paths held before it
    went in (_copy_before). It is outside <out_dir>/<name>/, where codetwine writes the
    analysis, and outside every agent's own copy of the output directory.

    Args:
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.

    Returns:
        The path, as an absolute path.
    """
    return os.path.join(out_dir, _AGENT_DIR, _REPO_DIR, name)


def output_dir_list(out_dir: str) -> list[tuple[str, str]]:
    """Return the output directories under one, the agents' own copies included.

    Args:
        out_dir: The output directory, as it was passed.

    Returns:
        [("", the directory itself), (the agent, its own copy), ...], in order.
    """
    dir_list = [("", out_dir)]
    try:
        agent_list = sorted(os.listdir(os.path.join(out_dir, _AGENT_DIR)))
    except OSError:
        return dir_list
    return dir_list + [(agent, agent_dir(out_dir, agent)) for agent in agent_list
                       if agent not in (_HOST_DIR, _REPO_DIR)]


def host_dir_list(out_dir: str) -> list[tuple[str, str]]:
    """Return this machine's own directories of the agents under an output directory.

    Args:
        out_dir: The output directory, as an absolute path.

    Returns:
        [(the agent, host_dir() of it), ...], in order; empty when there is none.
    """
    try:
        agent_list = sorted(os.listdir(os.path.join(out_dir, _AGENT_DIR, _HOST_DIR)))
    except OSError:
        return []
    return [(agent, host_dir(out_dir, agent)) for agent in agent_list]


def inside_path(path: str, out_dir: str) -> str:
    """Return a path in an agent's copy of the output directory, refusing one outside it.

    A path that leads out of the directory through a link is refused. A container writes
    that directory.

    Args:
        path: The file or directory, as an absolute path under out_dir. It need not
            exist.
        out_dir: The agent's own copy of the output directory (agent_dir).

    Returns:
        path as it was given.

    Raises:
        RuntimeError: When path, with every link in it resolved, is not inside
            out_dir.
    """
    real_path, real_dir = os.path.realpath(path), os.path.realpath(out_dir)
    if real_path != real_dir and os.path.commonpath([real_path, real_dir]) != real_dir:
        raise RuntimeError(
            f"{path} is reached through a link that leads out of {out_dir} (to "
            f"{real_path}). Only a container writes that directory, so the file is "
            f"neither read nor written")
    return path


def analyse_if_none(project_dir: str, out_dir: str, name: str,
                    exclude_list: list[str] | None, with_doc: bool,
                    ask_doc: bool = False) -> bool:
    """Make the first analysis in an output directory nothing has been built in.

    The working tree is analysed as it is, and the record beside the analysis says which
    tree it describes (build.analyse_and_index).

    Making the analysis with the documents is a step of its own on the terminal
    (progress.step).

    The terminal is asked about the documents before the lock is taken; the answer is
    dropped when the lock comes and the analysis is there. An analysis already there
    whose index was not built from it under the rules in force (build.is_index_from)
    gets its index built anew, under the same lock (no LLM).

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        exclude_list: The extra names kept out of the analysis.
        with_doc: Whether codetwine generates the design documents too.
        ask_doc: Whether, when with_doc is False and the analysis is about to be made,
            the terminal is asked if the documents are made too (knowledge.ask_doc).
            With no terminal the answer is no.

    Returns:
        True when the analysis was made.
    """
    build_dir = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    knowledge_path = knowledge.knowledge_db_path(out_dir, name)
    if knowledge.is_complete(knowledge_path):
        _index_up_to_date(knowledge_path, out_dir, name)
        return False
    if ask_doc and not with_doc:
        with_doc = knowledge.ask_doc(knowledge_path)
    # Two first runs started at once make one analysis: the second waits here and
    # finds it made
    with filelock.hold_lock(build_dir, f"the first analysis in {out_dir}"):
        if knowledge.is_complete(knowledge_path):
            return False
        progress.note("nothing is built in %s, so the repository is analysed here first "
                      "%s", out_dir,
                      "with the design documents (the LLM is called)" if with_doc
                      else "without the design documents (no LLM is called)")
        doc_step = progress.step("make the design documents") if with_doc \
            else contextlib.nullcontext()
        with doc_step:
            build.analyse_and_index(project_dir, out_dir, exclude_list, with_doc=with_doc)
        return True


def _index_up_to_date(knowledge_path: str, out_dir: str, name: str) -> None:
    """Build the index of a complete analysis anew when it was not built from it.

    Args:
        knowledge_path: The analysis.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
    """
    index_path = store.index_db_path(out_dir, name)
    if build.is_index_from(knowledge_path, index_path):
        return
    with filelock.hold_lock(os.path.join(out_dir, name), f"the index in {out_dir}"):
        if not build.is_index_from(knowledge_path, index_path):
            logger.info("the index at %s was not built from the analysis under the rules "
                        "in force, so it is built again (no LLM is called)", index_path)
            build.index_again(knowledge_path, name, index_path)


# What a run is told when it names other names to keep out than the analysis on disk
# was made with
_EXCLUDE_MISMATCH = (
    "the analysis in {out_dir} keeps out {record} and this run asks to keep out {given}. "
    "Every container copies that analysis, so a run keeps out what it keeps out. Drop "
    "--exclude / BUILD_EXCLUDE, or make the analysis again with:")


def exclude_list_of_run(project_dir: str, out_dir: str, name: str,
                        exclude_list: list[str] | None, with_doc: bool,
                        ask_doc: bool = False, doc_update: bool = False,
                        engine: str = "") -> list[str] | None:
    """Return the names this run keeps out of the analysis.

    The analysis is made first when there is none (analyse_if_none). A run that names
    nothing beyond config.BUILD_EXCLUDE keeps out what the analysis keeps out; one that
    names the same names goes on; one that names other names is refused. With doc_update
    and with_doc, an analysis that was already there is made again with the design
    documents, keeping out the names returned (_update_doc).

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        exclude_list: The names this run was given, config.BUILD_EXCLUDE's among them,
            or None for none.
        with_doc: Whether a first analysis made here generates the design documents.
        ask_doc: Whether that analysis asks on the terminal about the documents, as
            analyse_if_none() takes it.
        doc_update: Whether, with with_doc, an analysis already in out_dir is made
            again with the design documents.
        engine: The container engine the run uses, as _update_doc() takes it.

    Returns:
        The record's names when the run named none beyond config.BUILD_EXCLUDE and they
        differ from exclude_list; exclude_list as it came otherwise.

    Raises:
        RuntimeError: When the run names other names than the analysis was made with,
            or _update_doc() refuses or fails.
    """
    is_new = analyse_if_none(project_dir, out_dir, name, exclude_list, with_doc, ask_doc)
    run_exclude_list = exclude_list_of_record(project_dir, out_dir, name, exclude_list)
    if doc_update and with_doc and not is_new:
        _update_doc(project_dir, out_dir, name, run_exclude_list, engine)
    return run_exclude_list


def _exclude_entry_or_text(text: str) -> str:
    """Return one entry of BUILD_EXCLUDE in the form the names to keep out are kept in.

    Args:
        text: The entry as the setting holds it.

    Returns:
        knowledge.exclude_entry() of it, or text itself when that refuses it.
    """
    try:
        return knowledge.exclude_entry(text)
    except ValueError:
        return text


def exclude_list_of_record(project_dir: str, out_dir: str, name: str,
                           exclude_list: list[str] | None) -> list[str] | None:
    """Match the names this run was given against the record of the analysis in out_dir.

    Args:
        project_dir, out_dir, name, exclude_list: As exclude_list_of_run() takes them.

    Returns:
        What exclude_list_of_run() returns.

    Raises:
        RuntimeError: When the run names other names than the analysis was made with.
    """
    given_list = list(exclude_list or [])
    setting_set = {_exclude_entry_or_text(text) for text in config.BUILD_EXCLUDE}
    want_list = given_list if set(given_list) - setting_set else []
    knowledge_path = knowledge.knowledge_db_path(out_dir, name)
    if not os.path.exists(knowledge_path):
        return exclude_list
    db = knowledge.Knowledge.open(knowledge_path)
    try:
        stamp = db.stamp()
    finally:
        db.close()
    record_dict = knowledge.analysis_record(knowledge_path, stamp)
    record_list = knowledge.exclude_of_analysis(knowledge_path, stamp, given_list)
    if want_list and not knowledge.same_exclude(record_dict, want_list):
        raise CommandError(
            _EXCLUDE_MISMATCH.format(
                out_dir=out_dir, given=list_text(want_list),
                record=list_text(record_list) if record_list else "nothing"),
            command_line("codetwine-loom", "build", "--project", project_dir,
                         "--output-dir", out_dir,
                         *(word for exclude_name in want_list
                           for word in ("--exclude", exclude_name))))
    if want_list or sorted(record_list) == sorted(given_list):
        return exclude_list
    logger.info("this run keeps out what the analysis in %s keeps out: %s", out_dir,
                list_text(record_list) if record_list else "nothing")
    return record_list


def _update_doc(project_dir: str, out_dir: str, name: str,
                exclude_list: list[str] | None, engine: str = "") -> None:
    """Make the analysis already in out_dir again with the design documents, and its index.

    codetwine writes the documents of the files that have none or whose source differs
    from the one their document was written over, and of the files depending on them;
    with none such, no LLM is called and only the definitions and the line numbers are
    made again. The analysis and the index are made under the lock of <out_dir>/<name>
    (filelock.hold_lock), and this is a step of its own on the terminal (progress.step).

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        exclude_list: The names the analysis keeps out.
        engine: The container engine the run uses, as refuse_while_run_live() takes
            it.

    Raises:
        RuntimeError: When runs are going over out_dir, when what the documents need is
            not set, or when every document failed (knowledge.run_codetwine()).
    """
    refuse_while_run_live(out_dir, name, "drop --doc-create, ", engine)
    build_dir = os.path.join(out_dir, name)
    with filelock.hold_lock(build_dir, f"the analysis in {out_dir}"):
        logger.info("making the analysis in %s again with the design documents of the "
                    "files that have none or have changed since theirs were written (the "
                    "LLM is called for those)", out_dir)
        with progress.step("make the design documents again"):
            build.analyse_and_index(project_dir, out_dir, exclude_list, with_doc=True)


def refuse_while_run_live(out_dir: str, name: str, other_way: str = "",
                          engine: str = "") -> None:
    """Refuse to replace the analysis in out_dir while runs are going over it.

    Nothing is looked at inside a container, where the analysis is the run's own copy,
    nor where no container engine is on PATH (live_run_list).

    Args:
        out_dir: The output directory, as the message names it.
        name: The repository's directory name.
        other_way: What else the message offers besides waiting, ending in ", ", put
            before "or pass another --output-dir". Empty offers nothing more.
        engine: The container engine the run asking uses, asked first
            (live_run_list), or "" for the configured one.

    Raises:
        RuntimeError: With the runs that are going, when there are any.
    """
    if in_container():
        return
    run_list = live_run_list(out_dir, name, engine=engine)
    if not run_list:
        return
    raise RuntimeError(
        f"{len(run_list)} run(s) are going over {out_dir} ({', '.join(run_list)}). Each "
        f"agent a run starts copies the analysis in {os.path.join(out_dir, name)}, which "
        f"this would replace. Wait for them to end, {other_way}or pass another "
        f"--output-dir")


def overview_argv(name: str) -> list[str]:
    """Build the overview subcommand the container runs, in the container's own paths.

    Args:
        name: The repository's directory name, the same one build_run_argv() is given.

    Returns:
        The argument list that follows the image.
    """
    return ["overview", "--project", f"{_WORK}/{name}", "--output-dir", _OUT]


def make_overview(project_dir: str, out_dir: str, name: str,
                  exclude_list: list[str] | None, base: str, engine: str = "",
                  image: str = "") -> None:
    """Make the overview when it is missing or not current, in a container.

    Steps:
    1. Make the analysis and the index in out_dir describe the working tree, in a
       process of its own on this machine that runs build (_describe_tree). It calls no
       model, is given no credential, and refuses while other runs are going over
       out_dir
    2. Read whether the overview in out_dir is current for that index and the working
       tree (overview.is_current). One that is has no container started for it
    3. Otherwise start one container - one broker, one copy of the output directory of
       its own - and have the overview subcommand make the overview in that copy
       (_overview_in_container)
    4. Write the answer the container left into out_dir, with the file digest read in
       step 2

    When the overview is current or was made, its file digest is set in this process's
    overview.RUN_DIGEST_ENV: it is the state the run starts from. A step that fails, and
    the two processes running past OVERVIEW_TIMEOUT_SEC between them, are logged as a
    warning, the variable is left unset, and nothing is raised. The token counts of the
    container's model calls are added to this process's (_start_container).

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        exclude_list: The names this run keeps out of the analysis.
        base: The commit the container's clone is checked out at, which a branch
            reaches: the commit the run starts from.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
    """
    # A digest a try before this one set stands for the state that try started from
    os.environ.pop(overview.RUN_DIGEST_ENV, None)
    limit = config.OVERVIEW_TIMEOUT_SEC
    deadline = time.monotonic() + limit if limit > 0 else None
    why = _describe_tree(project_dir, out_dir, exclude_list, deadline)
    if not why:
        digest = overview.index_digest(project_dir, out_dir, name)
        if digest and overview.is_current(out_dir, name, digest):
            os.environ[overview.RUN_DIGEST_ENV] = digest
            return
        why = _overview_in_container(project_dir, out_dir, name, exclude_list, base,
                                     engine, image, digest, deadline) if digest \
            else "found no index to make it over"
        if not why:
            os.environ[overview.RUN_DIGEST_ENV] = digest
            return
    logger.warning("making the overview %s, so the overview in %s may be missing or "
                   "not current; the run goes on, and an agent that finds it so searches "
                   "without it", why, out_dir)


def _overview_time_out_text() -> str:
    """Return what the warning says of an overview stopped at its time limit."""
    return f"was stopped at OVERVIEW_TIMEOUT_SEC ({config.OVERVIEW_TIMEOUT_SEC} s)"


def _describe_tree(project_dir: str, out_dir: str, exclude_list: list[str] | None,
                   deadline: float | None) -> str:
    """Run build over the repository in a process of its own, with no credential.

    The process is given config.own_child_env(), with which it reads no .env file,
    and the names to keep out as BUILD_EXCLUDE. Its printed lines go to stderr, or
    nowhere when this process was started with stderr closed, and no terminal is
    handed on.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        exclude_list: The names this run keeps out of the analysis.
        deadline: The time.monotonic() value the process is stopped at, or None for
            no limit.

    Returns:
        "" when the process ended with status 0, otherwise what the warning says of it.
    """
    argv = config.own_argv(["build", "--project", project_dir, "--output-dir", out_dir])
    try:
        proc = subprocess.run(
            argv, stdin=subprocess.DEVNULL, check=False,
            stdout=subprocess.DEVNULL if sys.stderr is None else sys.stderr,
            timeout=None if deadline is None else max(deadline - time.monotonic(), 0),
            env=config.own_child_env({"BUILD_EXCLUDE": shlex.join(exclude_list or [])}))
    except subprocess.TimeoutExpired:
        return _overview_time_out_text()
    return f"ended with status {proc.returncode} while the analysis was made" \
        if proc.returncode else ""


def _overview_in_container(project_dir: str, out_dir: str, name: str,
                           exclude_list: list[str] | None, base: str, engine: str,
                           image: str, digest: str, deadline: float | None) -> str:
    """Have a container make the overview, and write its answer into out_dir.

    The container is an agent of its own, named after its run: its copy of the output
    directory is made before it starts and taken away when it ends. The answer is read
    from that copy through inside_path(), held to DISPATCH_ANSWER_MAX_CHAR (_cap_text)
    and written into out_dir with digest (overview.write_record). A container that
    ended with a status other than 0 is taken at the overview it wrote, when the copy
    holds another than it held before the container started.

    Args:
        project_dir, out_dir, name, exclude_list, base, engine, image: As
            make_overview() takes them.
        digest: The file digest the overview is recorded with
            (overview.index_digest).
        deadline: The time.monotonic() value the container is stopped at, or None for
            no limit.

    Returns:
        "" when the overview was written, otherwise what the warning says of it.
    """
    limit = 0
    if deadline is not None:
        limit = int(deadline - time.monotonic())
        if limit <= 0:
            return _overview_time_out_text()
    run_id = new_run_id()
    agent = agent_name("", run_id)
    try:
        own_dir = make_agent_dir(out_dir, name, agent, project_dir, exclude_list, False)
        # What the copy holds before the container starts is no overview it wrote
        record_before_dict = overview.read_record(own_dir, name)
        _, _, status = _start_container(
            engine, image, project_dir, out_dir, own_dir, name, agent, base,
            run_file_layout(out_dir, agent, name, run_id), run_id, overview_argv(name),
            exclude_list, limit=limit)
        inside_path(overview.overview_path(own_dir, name), own_dir)
        answer_text = overview.answer_of(own_dir, name)
        # A container that ended with another status is taken at its overview when it
        # wrote one
        if status and not (answer_text
                           and overview.read_record(own_dir, name) != record_before_dict):
            return _overview_time_out_text() if status == _TIME_OUT_STATUS \
                else f"ended with status {status}"
        if status:
            logger.warning("the container ended with exit status %d after it wrote the "
                           "overview, and the overview is taken", status)
        if not answer_text:
            return "left no overview in the container's copy of the output directory"
        overview.write_record(
            overview.overview_path(out_dir, name), project_dir, digest,
            _cap_text(answer_text, config.DISPATCH_ANSWER_MAX_CHAR, "the overview"))
    except Exception as e:
        return f"could not be done in a container ({_error_text(e)})"
    finally:
        drop_the_copy(out_dir, name, agent)
    return ""


def _take_token_file(token_path: str, part: str) -> None:
    """Add the token counts a process wrote to this process's total.

    The file stays where it is.

    Args:
        token_path: What tokencount.child_file() returned.
        part: What made the calls, as tokencount.add() takes it.
    """
    if not token_path:
        return
    count_dict = tokencount.add_file(token_path, part)
    if count_dict[tokencount.CALL_KEY]:
        logger.info("%s used %s token(s) in %d call(s)", part,
                    f"{tokencount.token_sum(count_dict):,}",
                    count_dict[tokencount.CALL_KEY])


def make_agent_dir(out_dir: str, name: str, agent: str, project_dir: str,
                   exclude_list: list[str] | None, with_doc: bool,
                   ask_doc: bool = False) -> str:
    """Make the agent's own copy of the output directory, and return where it is.

    An output directory nothing has been built in gets its first analysis here
    (analyse_if_none) before the copy.

    The copy is made from <out_dir>/<name>/: the index, the analysis and the files
    codetwine left beside them. What this tool writes for itself (_is_run_own) - a
    patch, a turn log, a record - is not copied.

    An agent's directory that holds the analysis already is left as it is. One that
    holds none is copied into again: this tool's own files there stay, and everything
    else there, a link included, is taken away first (_remove_copy). Nothing is copied
    when any of that is still there afterwards, and the copy is made onto nothing that
    stands there (_copy_build).

    Args:
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        agent: The agent, as agent_name() names it.
        project_dir: The target repository, as an absolute path.
        exclude_list: The extra names kept out of the analysis.
        with_doc: Whether a first analysis made here generates the design documents.
        ask_doc: Whether that analysis asks on the terminal about the documents when
            with_doc is False, as analyse_if_none() takes it.

    Returns:
        The agent's output directory, as an absolute path.

    Raises:
        RuntimeError: When the copy cannot be made, or when the agent's directory
            holds what could not be taken away.
    """
    analyse_if_none(project_dir, out_dir, name, exclude_list, with_doc, ask_doc)
    own_dir = agent_dir(out_dir, agent)
    os.makedirs(own_dir, exist_ok=True)
    # The copy is looked for and made under the agent's lock: of two first rounds
    # started at once, the second waits here and finds it made
    with filelock.hold_lock(own_dir, f"{agent}'s copy of the output directory"):
        copy_dir = os.path.join(own_dir, name)
        if _is_real_dir(copy_dir) \
                and os.path.exists(knowledge.knowledge_db_path(own_dir, name)):
            return own_dir
        build_dir = os.path.join(out_dir, name)
        if not os.path.isdir(build_dir):
            logger.info("%s starts with an empty output directory, so the container "
                        "builds the index itself", agent)
            return own_dir
        logger.info("copying what is built in %s to %s, which is %s's from now on",
                    build_dir, own_dir, agent)
        stay_list = _remove_copy(copy_dir, agent)
        if stay_list:
            raise RuntimeError(
                f"{agent}'s copy of the output directory holds what could not be "
                f"taken away ({stay_list[0]}"
                f"{f' and {len(stay_list) - 1} more' if len(stay_list) > 1 else ''}), "
                f"so nothing is copied over it. Remove {copy_dir} by hand, then run "
                f"again")
        # Under the analysis' own lock as well
        try:
            with filelock.hold_lock(build_dir, f"the analysis in {out_dir}"):
                _copy_build(build_dir, copy_dir)
        except OSError as e:
            raise RuntimeError(f"could not make {agent}'s copy of {build_dir}: "
                               f"{e}") from None
    return own_dir


def _copy_build(build_dir: str, copy_dir: str) -> None:
    """Copy build_dir into an agent's copy, onto nothing that stands there.

    The copy itself and the directories of _RUN_OWN_DIR_TUPLE directly in it are used
    when they are directories of their own and made when they are not there. Every other
    entry is made new: a directory with what it holds (shutil.copytree), a file at a
    name nothing stands at (tmpfile.make_new_file). What is this tool's own in build_dir
    is not copied (_run_own_ignore). Only regular files and directories are copied, at
    any depth (_is_plain_entry): a link, wherever it leads, and anything else that is
    neither is left out, and the ones left out are named in the log. Nothing is read
    through a link.

    Args:
        build_dir: <output_dir>/<project name>/, as an absolute path.
        copy_dir: The agent's copy of it, as an absolute path.

    Raises:
        OSError: When something stands where an entry goes, a link included, when one
            of the directories that are used is a link or no directory, or when an
            entry cannot be copied.
    """
    own_ignore = _run_own_ignore(build_dir)
    drop_list = []

    def ignore(directory: str, name_list: list[str]) -> set[str]:
        """Return the names in one directory under build_dir that are not copied."""
        skip_set = own_ignore(directory, name_list)
        for entry in name_list:
            path = os.path.join(directory, entry)
            if entry not in skip_set and not _is_plain_entry(path):
                skip_set.add(entry)
                drop_list.append(path)
        return skip_set

    pair_list = [(build_dir, copy_dir)]
    while pair_list:
        source_dir, target_dir = pair_list.pop()
        try:
            os.mkdir(target_dir)
        except FileExistsError:
            if not _is_real_dir(target_dir):
                raise
        name_list = os.listdir(source_dir)
        skip_set = ignore(source_dir, name_list)
        for entry in name_list:
            if entry in skip_set:
                continue
            source = os.path.join(source_dir, entry)
            target = os.path.join(target_dir, entry)
            if source_dir == build_dir and entry in _RUN_OWN_DIR_TUPLE \
                    and _is_real_dir(source):
                pair_list.append((source, target))
            elif _is_real_dir(source):
                shutil.copytree(source, target, symlinks=True, ignore=ignore)
            else:
                tmpfile.make_new_file(target)
                shutil.copy2(source, target, follow_symlinks=False)
    if drop_list:
        logger.warning("what in %s is neither a regular file nor a directory, a link "
                       "included, is not copied into %s (%d: %s)", build_dir, copy_dir,
                       len(drop_list), one_line(list_text(drop_list)))


def _is_plain_entry(path: str) -> bool:
    """Say whether a path is a regular file or a directory, not reached through a link.

    Args:
        path: The path to look at.

    Returns:
        True for a regular file or a directory that is no symbolic link, False for
        anything else and for a path that cannot be looked at.
    """
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) or stat.S_ISDIR(mode)


def _is_real_dir(path: str) -> bool:
    """Say whether a path is a directory itself rather than a link to one.

    Args:
        path: The path to look at.

    Returns:
        True for a directory that is not a symbolic link.
    """
    return os.path.isdir(path) and not os.path.islink(path)


def _is_run_own(parent: str, entry: str, path: str) -> bool:
    """Say whether one entry of <output_dir>/<project name>/ is this tool's own.

    Args:
        parent: The directory the entry is in, relative to <output_dir>/<project name>/:
            "" for that directory itself, or one of the names in _RUN_OWN_DIR_TUPLE.
        entry: The entry's name.
        path: The entry's path.

    Returns:
        True for _RUN_OWN_FILE directly under the directory, and, in a directory of
        _RUN_OWN_DIR_TUPLE, for every entry that is no directory. A link is never this
        tool's own.
    """
    if os.path.islink(path):
        return False
    if not parent:
        return entry == _RUN_OWN_FILE and os.path.isfile(path)
    return not os.path.isdir(path)


def _run_own_ignore(build_dir: str) -> Callable[[str, list[str]], set[str]]:
    """Return what tells shutil.copytree() which entries of build_dir are not copied.

    Args:
        build_dir: <output_dir>/<project name>/, as an absolute path.

    Returns:
        (a directory under build_dir, the names in it) -> the names left out: this
        tool's own (_is_run_own) in build_dir and in the directories of
        _RUN_OWN_DIR_TUPLE directly under it, and such a directory itself when it holds
        nothing else. Nothing is left out of any other directory.
    """
    def ignore(directory: str, name_list: list[str]) -> set[str]:
        """Return the names in one directory under build_dir that are not copied."""
        parent = os.path.relpath(directory, build_dir)
        parent = "" if parent == os.curdir else parent
        if parent and parent not in _RUN_OWN_DIR_TUPLE:
            return set()
        skip_set = {entry for entry in name_list
                    if _is_run_own(parent, entry, os.path.join(directory, entry))}
        if parent:
            return skip_set
        for entry in name_list:
            path = os.path.join(directory, entry)
            if entry in _RUN_OWN_DIR_TUPLE and _is_real_dir(path) and all(
                    _is_run_own(entry, inner, os.path.join(path, inner))
                    for inner in os.listdir(path)):
                skip_set.add(entry)
        return skip_set

    return ignore


def _remove_entry(path: str, what: str) -> bool:
    """Take one entry of a directory away.

    A directory goes with what it holds; anything else, a link included, goes as the one
    file it is.

    Args:
        path: The entry, as an absolute path.
        what: What the entry held, for the message when it cannot be taken away.

    Returns:
        True when nothing is left at the path.
    """
    if _is_real_dir(path):
        return _remove_tree(path, what)
    return _remove_file(path, what)


def _entry_list(path: str, what: str) -> list[str] | None:
    """Return the names in a directory, saying so when it cannot be listed.

    Args:
        path: The directory, as an absolute path.
        what: What the directory holds, for the message.

    Returns:
        The names, or None for a directory that cannot be listed.
    """
    try:
        return os.listdir(path)
    except OSError as e:
        logger.warning("%s at %s could not be listed (%s). Remove it by hand; nothing "
                       "reads it", what, path, e)
        return None


def _remove_copy(copy_dir: str, agent: str) -> list[str]:
    """Take everything but this tool's own out of an agent's copy of its directory.

    The copy is the agent's copy of <output_dir>/<project name>/. What stays is this
    tool's own (_is_run_own). A link is taken away as the link it is, whether it is the
    copy itself or anything in it, and nothing is listed or removed through one. A
    directory of _RUN_OWN_DIR_TUPLE left holding nothing is taken away too.

    Args:
        copy_dir: The agent's copy, as an absolute path. One that is not there is left
            alone.
        agent: The agent, for the messages.

    Returns:
        The paths that were to go and are still there, a directory that could not be
        listed among them. Empty when everything went.
    """
    if os.path.islink(copy_dir):
        logger.warning("%s is a link, which no run of this tool makes there, so it is "
                       "taken away", copy_dir)
        is_clear = _remove_file(copy_dir, f"the link in place of {agent}'s copy")
        return [] if is_clear else [copy_dir]
    if not os.path.isdir(copy_dir):
        return []
    entry_list = _entry_list(copy_dir, f"{agent}'s copy")
    if entry_list is None:
        return [copy_dir]
    stay_list = []
    for entry in entry_list:
        path = os.path.join(copy_dir, entry)
        if entry in _RUN_OWN_DIR_TUPLE and _is_real_dir(path):
            inner_list = _entry_list(path, f"{agent}'s copy of {entry}")
            if inner_list is None:
                stay_list.append(path)
                continue
            for inner in inner_list:
                inner_path = os.path.join(path, inner)
                if not _is_run_own(entry, inner, inner_path) and not _remove_entry(
                        inner_path, f"{agent}'s copy of {entry}/{inner}"):
                    stay_list.append(inner_path)
            with contextlib.suppress(OSError):
                os.rmdir(path)
        elif not _is_run_own("", entry, path) \
                and not _remove_entry(path, f"{agent}'s copy of {entry}"):
            stay_list.append(path)
    return stay_list


def drop_the_copy(out_dir: str, name: str, agent: str) -> None:
    """Take away the copy of the index an agent whose run has ended left behind.

    What the run wrote is kept - the patch, the summary beside it, the turn log and the
    record - and the copy those were written beside is what goes (_remove_copy). An
    agent that wrote nothing at all leaves nothing in its own copy: the directory and
    the lock file standing for it go too, and so does the agent's directory on this
    machine's side when nothing is kept in it (_remove_empty_host_dir).

    Args:
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        agent: The agent, as agent_name() names it.
    """
    own_dir = agent_dir(out_dir, agent)
    copy_dir = os.path.join(own_dir, name)
    if os.path.islink(copy_dir) or os.path.isdir(copy_dir):
        logger.info("%s has ended, so its copy of the index is taken away", agent)
        _remove_copy(copy_dir, agent)
        for path in (copy_dir, own_dir):
            try:
                os.rmdir(path)
            except FileNotFoundError:
                continue
            except OSError:
                break
            _remove_file(filelock.path_for(path), f"the lock of {path}")
    _remove_empty_host_dir(out_dir, name, agent)


def _remove_empty_host_dir(out_dir: str, name: str, agent: str) -> None:
    """Take away an agent's directory on this machine's side when nothing is kept in it.

    The directory of the runs' files goes when it is empty, with the lock taken on it,
    then the directories above it up to host_dir(), each when it is empty. A diff put
    into the working tree, a container's stdout, the broker's log and the records of
    the runs this machine started (_write_run_mark) are kept.

    Args:
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        agent: The agent, as agent_name() names it.
    """
    own_host_dir = host_dir(out_dir, agent)
    run_dir = _sandbox_dir(own_host_dir, name)
    try:
        os.rmdir(run_dir)
    except FileNotFoundError:
        pass
    except OSError:
        return
    _remove_file(filelock.path_for(run_dir), f"the lock of {run_dir}")
    for path in (os.path.dirname(run_dir), own_host_dir):
        try:
            os.rmdir(path)
        except OSError:
            return


def container_name(run_id: str) -> str:
    """Return the name the run's container answers to.

    Args:
        run_id: The run, as new_run_id() names it.

    Returns:
        The name.
    """
    return f"{_CONTAINER_PREFIX}{run_id}"


# Set when the run was stopped (Ctrl-C, SIGTERM, SIGHUP) while agents were running. No
# container is started after it
_STOP_EVENT = threading.Event()

# What stopped the containers, as stop_running_container() was told it
_STOP_WHY_LIST: list[str] = []
_STOP_TEXT = "the run was stopped"

# The containers going now, as {name: (the engine that started it, the engine's client
# that is waited for)}
_LIVE_CONTAINER_DICT: dict[str, tuple[str, subprocess.Popen]] = {}
_LIVE_CONTAINER_LOCK = threading.Lock()


def forget_stop() -> None:
    """Let containers be started again after stop_running_container()."""
    _STOP_EVENT.clear()
    _STOP_WHY_LIST.clear()


def _stop_why() -> str:
    """Return what stop_running_container() was last told, or _STOP_TEXT."""
    return _STOP_WHY_LIST[-1] if _STOP_WHY_LIST else _STOP_TEXT


def stop_running_container(why: str = _STOP_TEXT) -> None:
    """Stop every container this process has going, and start no other.

    Each is stopped with the engine's client that started it (_stop_container). A stop
    signal that reaches the main thread meanwhile is held until every one of them was
    stopped, and then goes on (stop_signal_hold).

    Args:
        why: What stopped them.
    """
    _STOP_WHY_LIST[:] = [why]
    _STOP_EVENT.set()
    with _LIVE_CONTAINER_LOCK:
        live_list = list(_LIVE_CONTAINER_DICT.items())
    with stop_signal_hold():
        for name, (engine, process) in live_list:
            logger.warning("%s, so %s is taken away", why, name)
            _stop_container(engine, name, process)


def _start_client(argv: list[str], out_fd: int, err_fd: int) -> subprocess.Popen:
    """Start the engine's client that runs one container.

    The client is started with config.child_env(), without the credentials in its
    environment.

    Args:
        argv: The command line, as build_run_argv() returns it.
        out_fd: The descriptor the client's stdout goes to.
        err_fd: The descriptor its stderr goes to.

    Returns:
        The running client.

    Raises:
        OSError: When the engine could not be started.
    """
    return subprocess.Popen(argv, stdout=out_fd, stderr=err_fd, env=config.child_env())


def _stop_container(engine: str, name: str, process: subprocess.Popen) -> None:
    """Take one container away and end the engine's client that started it.

    The container is removed by name first, and the client is then ended. Where the
    engine did not remove a container - the client had not made it yet, the engine
    refused, or it did not answer in time - one the engine runs under that name, or
    one it cannot say about, once the client has ended is removed. A client still
    going CONTAINER_COMMAND_TIMEOUT_SEC after it was asked to end is killed. All of it
    takes at most 2 x CONTAINER_COMMAND_TIMEOUT_SEC: the engine is asked about the
    container and to remove it again within what is left of that. A container that
    may still be there afterwards - no time was left, or the second removal failed -
    is named with the command that removes it. A stop signal that reaches the main
    thread meanwhile is held until then, and then goes on (stop_signal_hold).

    Args:
        engine: The engine's executable.
        name: The container's name, as container_name() builds it.
        process: The engine's client, as _start_client() returned it.
    """
    limit = config.CONTAINER_COMMAND_TIMEOUT_SEC or None
    with stop_signal_hold():
        deadline = time.monotonic() + 2 * limit if limit else None
        is_away = _remove_container(engine, name)
        process.terminate()
        try:
            process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            logger.warning("the engine's client of %s did not end when asked, so it is "
                           "being killed", name)
            process.kill()
            process.wait()
        if is_away:
            return
        wait_sec = _remain_sec(deadline)
        if wait_sec != 0:
            if not _container_alive(engine, name[len(_CONTAINER_PREFIX):],
                                    wait_sec=wait_sec):
                return
            wait_sec = _remain_sec(deadline)
            if wait_sec != 0 and _remove_container(engine, name, wait_sec=wait_sec):
                return
        logger.warning("%s may still be there. Take it away with: %s", name,
                       command_line(engine, "rm", "--force", name))


def _remain_sec(deadline: float | None) -> float | None:
    """Return the seconds left until deadline.

    Args:
        deadline: A time.monotonic() value, or None for none.

    Returns:
        The seconds left, 0 once deadline has passed, or None where there is no
        deadline.
    """
    if deadline is None:
        return None
    return max(deadline - time.monotonic(), 0.0)


def run_the_container(argv: list[str], engine: str, run_id: str, limit: int,
                      out_path: str) -> int:
    """Start one container, wait for it, and return its exit status.

    The time limit is applied here. A container over the limit, and one this thread is
    waiting for when a stop reaches it, is taken away with the engine's client that
    started it (_stop_container). The client is held in _LIVE_CONTAINER_DICT from the
    moment it is started, and a stop from another thread (stop_running_container)
    reaches a container the engine has not made yet. What the container writes to stdout
    and stderr is handed on line by line, each line under the label of the work that
    started it. Its stdout goes to out_path instead of this process's stdout while
    progress.keep_child_stdout() has it on. The engine's client is started with
    config.child_env(), without the credentials in its environment.

    Args:
        argv: The command line, as build_run_argv() returns it.
        engine: The engine's executable, as pick_engine() returns it.
        run_id: The run, which names the container.
        limit: The seconds the container may run for. 0 or less sets no limit.
        out_path: The file the container's stdout is appended to (progress.relay).

    Returns:
        The container's exit status, or _TIME_OUT_STATUS when the limit was reached. An
        engine's client ended by a signal gives SIGNAL_STATUS_BASE plus the signal's
        number. _KILL_STATUS from the container is logged with the memory settings to
        change.

    Raises:
        KeyboardInterrupt: When the run was stopped (stop_running_container(), or a
            stop in this thread), before the container is started or once it was
            taken away.
        OSError: When the engine's client could not be started.
    """
    name = container_name(run_id)
    if _STOP_EVENT.is_set():
        raise KeyboardInterrupt(f"{_stop_why()}, so {name} is not started")
    try:
        with progress.relay(config.CONTAINER_COMMAND_TIMEOUT_SEC or None,
                            out_path) as (out_fd, err_fd):
            process = _start_client(argv, out_fd, err_fd)
            with _LIVE_CONTAINER_LOCK:
                _LIVE_CONTAINER_DICT[name] = (engine, process)
            try:
                # A stop made while the client was being started found none to end
                if _STOP_EVENT.is_set():
                    _stop_container(engine, name, process)
                status = process.wait(timeout=limit if limit > 0 else None)
            except subprocess.TimeoutExpired:
                logger.warning("the container ran for %d seconds without ending, so it "
                               "is being taken away. Raise the limit for this "
                               "repository, or look at what the run was waiting for",
                               limit)
                _stop_container(engine, name, process)
                return _TIME_OUT_STATUS
            except KeyboardInterrupt:
                # A stop that reaches the thread waiting here: the container and the
                # engine's client are taken away before the stop goes on
                logger.warning("the run was stopped, so %s is taken away", name)
                _stop_container(engine, name, process)
                raise
    finally:
        with _LIVE_CONTAINER_LOCK:
            _LIVE_CONTAINER_DICT.pop(name, None)
    if _STOP_EVENT.is_set():
        raise KeyboardInterrupt(f"{_stop_why()} while {name} was going")
    if status < 0:
        logger.warning("the engine's client of %s was ended by signal %d, so the "
                       "container is taken as ended with exit status %d", name, -status,
                       SIGNAL_STATUS_BASE - status)
        return SIGNAL_STATUS_BASE - status
    if status == _KILL_STATUS:
        logger.warning("the container was killed (exit status %d), which is how it ends "
                       "when its processes reach SANDBOX_MEMORY (%s). Raise "
                       "SANDBOX_MEMORY for this repository, and lower DISPATCH_PARALLEL "
                       "so that the agents running at once (%d) fit in this machine's "
                       "memory", status, config.SANDBOX_MEMORY, config.DISPATCH_PARALLEL)
    return status


def _engine_proc(argv: list[str], wait_sec: float | None = None
                 ) -> subprocess.CompletedProcess:
    """Run one short command of a container engine and return it as it ended.

    The engine is started with config.child_env(), its output captured as text.

    Args:
        argv: The command line, the engine's executable first.
        wait_sec: The seconds the engine is given to answer. None gives
            CONTAINER_COMMAND_TIMEOUT_SEC.

    Returns:
        The finished process.

    Raises:
        OSError: When the engine could not be started.
        subprocess.TimeoutExpired: When it did not answer in time.
    """
    return subprocess.run(argv, check=False, capture_output=True, text=True,
                          env=config.child_env(),
                          timeout=wait_sec or config.CONTAINER_COMMAND_TIMEOUT_SEC or None)


def _remove_container(engine: str, name: str, wait_sec: float | None = None) -> bool:
    """Take one container away, saying so when it cannot be taken away.

    Args:
        engine: The engine's executable.
        name: The container's name, as container_name() builds it.
        wait_sec: The seconds the engine is given to answer. None gives
            CONTAINER_COMMAND_TIMEOUT_SEC.

    Returns:
        True when the engine removed the container.
    """
    try:
        proc = _engine_proc([engine, "rm", "--force", name], wait_sec)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("%s could not be taken away (%s)", name, e)
        return False
    if proc.returncode:
        logger.warning("%s is still there: %s", name,
                       proc.stderr.strip()[-config.LOG_EXCERPT_CHAR:])
        return False
    return True


def patch_name(run_id: str) -> str:
    """Return the file name one run's patch is written under.

    This machine reads the patch under this name, and the container is told to write it
    under it.

    Args:
        run_id: The run, as new_run_id() names it.

    Returns:
        The file name, without a directory.
    """
    return f"{_PATCH_PREFIX}{run_id}{_PATCH_SUFFIX}"


def _sandbox_dir(root: str, name: str) -> str:
    """Return where the runs' own files go under one output directory.

    Args:
        root: An output directory, as this machine sees it or as the container does.
        name: The repository's directory name.

    Returns:
        The directory.
    """
    return os.path.join(root, name, _PATCH_DIR)


def stdout_path(directory: str, run_id: str) -> str:
    """Return the file a container's stdout is kept in.

    Args:
        directory: The runs' own directory, as _sandbox_dir() returns it.
        run_id: The run, as new_run_id() names it.

    Returns:
        The path, "<directory>/stdout-<run_id>.txt".
    """
    return os.path.join(directory, f"{_STDOUT_PREFIX}{run_id}.txt")


def run_file_layout(out_dir: str, agent: str, name: str, run_id: str) -> RunFileLayout:
    """Return where one run's own files go.

    What the container reads or writes goes under the agent's own copy of the output
    directory (agent_dir), and what this machine alone writes under host_dir().

    Args:
        out_dir: The output directory, as an absolute path.
        agent: The agent the run is one of, as agent_name() names it.
        name: The repository's directory name.
        run_id: The run, as new_run_id() names it.

    Returns:
        The paths, all absolute.
    """
    own_dir = agent_dir(out_dir, agent)
    directory = _sandbox_dir(own_dir, name)
    host_directory = _sandbox_dir(host_dir(out_dir, agent), name)
    inside = _sandbox_dir(_OUT, name)

    def both_side(prefix: str) -> tuple[str, str]:
        """Return one file's path on this machine and as the container sees it."""
        # The output directory is mounted at _OUT, and the same name read from there is
        # where the container writes or reads the file this machine reads or writes
        file_name = f"{prefix}{run_id}{_TASK_SUFFIX}"
        return os.path.join(directory, file_name), os.path.join(inside, file_name)

    task, task_inside = both_side(_TASK_PREFIX)
    outcome, outcome_inside = both_side(_OUTCOME_PREFIX)
    answer, answer_inside = both_side(_ANSWER_PREFIX)
    answer_in, answer_in_inside = both_side(_ANSWER_IN_PREFIX)
    target, target_inside = both_side(_TARGET_PREFIX)
    gap, gap_inside = both_side(_GAP_PREFIX)
    return RunFileLayout(
        out_dir=own_dir,
        directory=directory,
        host_directory=host_directory,
        patch=os.path.join(directory, patch_name(run_id)),
        task=task, task_inside=task_inside,
        outcome=outcome, outcome_inside=outcome_inside,
        answer=answer, answer_inside=answer_inside,
        answer_in=answer_in, answer_in_inside=answer_in_inside,
        target=target, target_inside=target_inside,
        gap=gap, gap_inside=gap_inside,
        stdout=stdout_path(host_directory, run_id),
        scratch=os.path.join(host_directory, f"{_SCRATCH_PREFIX}{run_id}"),
        warm_scratch=os.path.join(host_directory, f"{_WARM_SCRATCH_PREFIX}{run_id}"),
        worktree=os.path.join(host_directory, f"{_WORKTREE_PREFIX}{run_id}"),
        land=os.path.join(host_directory, f"{_LAND_PREFIX}{run_id}{_PATCH_SUFFIX}"),
        mark=os.path.join(host_directory, f"{_RUN_MARK_PREFIX}{run_id}{_RUN_MARK_SUFFIX}"))


def fill_mark_path(cache: str) -> str:
    """Return the file that says which sets of dependency files a cache was filled for.

    It is beside the cache, not in it.

    Examples:
        "/out/agents/.repo/repo/cache" -> "/out/agents/.repo/repo/cache.filled"

    Args:
        cache: The package cache, as an absolute path.

    Returns:
        The mark file's path.
    """
    return f"{cache.rstrip(os.sep)}{_CACHE_FILL_MARK}"


def cache_dir(out_dir: str, name: str) -> str:
    """Return where the toolchain's package cache is kept.

    It is the "cache" directory in this tool's own directory of the repository
    (repo_own_dir), or config.SANDBOX_CACHE_DIR when that is set.

    Args:
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.

    Returns:
        The directory, as an absolute path.
    """
    if config.SANDBOX_CACHE_DIR:
        return os.path.abspath(config.SANDBOX_CACHE_DIR)
    return os.path.join(repo_own_dir(out_dir, name), _CACHE_DIR)


def check_cache_place(project_dir: str, out_dir: str) -> None:
    """Refuse a SANDBOX_CACHE_DIR that would give a container more than the package cache.

    The paths are compared with their links resolved; none has to exist. Nothing is
    refused when SANDBOX_CACHE_DIR is not set, or names the place the cache has without
    it.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.

    Raises:
        RuntimeError: When SANDBOX_CACHE_DIR is the repository, is inside it or holds
            it, is the output directory or holds it, or is inside one of the
            directories this tool keeps its own files in there (the repository's
            name, and the agents' directory).
    """
    if not config.SANDBOX_CACHE_DIR:
        return
    name = knowledge.project_name_of(project_dir)
    cache_real = os.path.realpath(cache_dir(out_dir, name))
    project_real, out_real = os.path.realpath(project_dir), os.path.realpath(out_dir)
    if cache_real == os.path.join(repo_own_dir(out_real, name), _CACHE_DIR):
        return
    why = ""
    if os.path.commonpath([cache_real, project_real]) in (cache_real, project_real):
        why = f"is inside PROJECT_DIR or holds it ({project_dir})"
    elif os.path.commonpath([cache_real, out_real]) == cache_real:
        why = f"is OUTPUT_DIR or holds it ({out_dir})"
    else:
        for own_dir in (os.path.join(out_real, name), os.path.join(out_real, _AGENT_DIR)):
            if os.path.commonpath([cache_real, own_dir]) == own_dir:
                why = f"is inside {own_dir}, where this tool keeps its own files"
    if why:
        raise RuntimeError(
            f"SANDBOX_CACHE_DIR ({config.SANDBOX_CACHE_DIR}) {why}, and every container "
            f"is given that directory to write. Set it to a directory of its own "
            f"outside the repository and the output directory, or leave it unset")


def in_container() -> bool:
    """Say whether this process is the run inside a sandboxed container.

    A process with CTLOOM_CLONE_ENV in its environment, where the entry point names the
    clone it made, is taken as that run.

    Returns:
        True inside the container.
    """
    return bool(os.environ.get(CTLOOM_CLONE_ENV))


def fill_ext_list(text: str | None = None) -> list[str]:
    """Return the languages a filling runs the default test commands of.

    Args:
        text: The languages, comma-separated. None reads CTLOOM_FILL_EXT_ENV.

    Returns:
        The LANG_DICT keys text names, in its order, each once. Empty when it names
        none.
    """
    if text is None:
        text = os.environ.get(CTLOOM_FILL_EXT_ENV, "")
    ext_list = []
    for ext in text.split(","):
        if ext in LANG_DICT and ext not in ext_list:
            ext_list.append(ext)
    return ext_list


def primary_ext_of_env() -> str:
    """Return the language a run in a container is told to take as the primary one.

    Returns:
        The LANG_DICT key CTLOOM_PRIMARY_EXT_ENV names, or "" when it names none.
    """
    ext = os.environ.get(CTLOOM_PRIMARY_EXT_ENV, "")
    return ext if ext in LANG_DICT else ""


def fill_run() -> bool:
    """Say whether this run fills the package cache.

    Returns:
        True when CTLOOM_FILL_ENV is "1", which sandbox/warm.sh sets.
    """
    return os.environ.get(CTLOOM_FILL_ENV) == "1"


def fill_own_run() -> bool:
    """Say whether a filling's container runs the index's own test commands.

    Returns:
        False when CTLOOM_FILL_OWN_ENV is "0", True otherwise.
    """
    return os.environ.get(CTLOOM_FILL_OWN_ENV) != "0"


def _clone_list(out_dir: str, name: str) -> list[tuple[str, str, bool]]:
    """Return the clones of one agent's runs.

    Args:
        out_dir: The agent's directory on this machine's side (host_dir), as an
            absolute path.
        name: The repository's directory name.

    Returns:
        [(the path, the run it belongs to, whether it is a refill's), ...], in order.
    """
    directory = _sandbox_dir(out_dir, name)
    try:
        entry_list = sorted(os.listdir(directory))
    except OSError:
        return []
    clone_list = []
    for entry in entry_list:
        for prefix, refill in ((_WARM_SCRATCH_PREFIX, True), (_SCRATCH_PREFIX, False)):
            if entry.startswith(prefix):
                clone_list.append((os.path.join(directory, entry), entry[len(prefix):],
                                   refill))
                break
    return clone_list


def _run_dead(run_id: str, engine_path: str = "") -> bool:
    """Say whether the run a clone belongs to has ended.

    Two things can be holding it. The id ends in the process that started the run
    (new_run_id). Its container, which outlives a process that was killed, is asked
    about as well, of engine_path and every other engine on PATH
    (_other_engine_path_list).

    Args:
        run_id: The run, as new_run_id() names it.
        engine_path: The engine's executable, asked first. One that cannot be asked
            counts as holding the container. Empty asks the other engines alone.

    Returns:
        True when neither is there any more. Anything engine_path cannot tell counts as
        running.
    """
    tail = run_id.rsplit("-", 1)[-1]
    if not (tail.isascii() and tail.isdigit()) or not tmpfile.process_dead(int(tail)):
        return False
    if engine_path and _container_alive(engine_path, run_id):
        return False
    return not any(_container_alive(other, run_id, doubt_alive=False)
                   for other in _other_engine_path_list(engine_path))


# The engines that did not answer this process within CONTAINER_COMMAND_TIMEOUT_SEC
# when it asked them as an engine other than a run's own, by their resolved path
_NO_ANSWER_ENGINE_SET: set[str] = set()


def _note_no_answer(engine_path: str, what: str) -> None:
    """Record that an engine other than a run's own did not answer in time, and say so.

    Args:
        engine_path: The engine's executable.
        what: What it was asked, for the log.
    """
    _NO_ANSWER_ENGINE_SET.add(os.path.realpath(engine_path))
    logger.warning("%s did not answer %s within %g seconds, so it is taken to hold "
                   "nothing and is not asked again by this command",
                   os.path.basename(engine_path), what,
                   config.CONTAINER_COMMAND_TIMEOUT_SEC)


def _other_engine_path_list(engine_path: str) -> list[str]:
    """Return the container engines on PATH other than engine_path.

    The engine CONTAINER_ENGINE names comes first, then those of ENGINE_TUPLE. Paths
    that resolve to the same file count as one engine. An engine that did not answer
    this process in time before (_NO_ANSWER_ENGINE_SET) is left out.

    Args:
        engine_path: The engine to leave out, as a path or a name on PATH, or "" for
            none.

    Returns:
        The executables' paths.
    """
    real_set = {os.path.realpath(shutil.which(engine_path) or engine_path)} \
        if engine_path else set()
    real_set |= _NO_ANSWER_ENGINE_SET
    path_list = []
    for want in (config.CONTAINER_ENGINE, *ENGINE_TUPLE):
        other = shutil.which(want) if want else None
        if other and os.path.realpath(other) not in real_set:
            real_set.add(os.path.realpath(other))
            path_list.append(other)
    return path_list


# What the engine's image inspect prints for an image: the JSON of its labels
_IMAGE_LABEL_FORMAT = "{{json .Config.Labels}}"


def _image_proc(engine_path: str, image: str) -> subprocess.CompletedProcess:
    """Ask a container engine for an image's labels.

    Args:
        engine_path: The engine's executable, as pick_engine() returns it.
        image: The image's name.

    Returns:
        The engine's image inspect as it ended, with exit status 0 when the engine has
        the image, and the JSON of the image's labels on its stdout.

    Raises:
        OSError: When the engine could not be started.
        subprocess.TimeoutExpired: When it did not answer in
            CONTAINER_COMMAND_TIMEOUT_SEC.
    """
    return _engine_proc([engine_path, "image", "inspect", "--format", _IMAGE_LABEL_FORMAT,
                         image])


def _image_id(engine: str, image: str) -> str:
    """Return the id an engine gives the image a filling runs.

    Args:
        engine: The container engine. Empty picks one (pick_engine()).
        image: The image's name. Empty uses config.CONTAINER_IMAGE.

    Returns:
        The id the engine's image inspect prints, or the image's name when there is no
        engine, the engine cannot be asked or it does not show the image.
    """
    image_name = image or config.CONTAINER_IMAGE
    try:
        proc = _engine_proc([pick_engine(engine), "image", "inspect", "--format",
                             "{{.Id}}", image_name])
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        logger.info("could not ask for the id of the image %s (%s), so the package "
                    "cache's mark names it by its name", image_name, e)
        return image_name
    image_id = (proc.stdout or "").strip() if not proc.returncode else ""
    if not image_id:
        logger.info("the engine gave no id for the image %s (%s), so the package "
                    "cache's mark names it by its name", image_name,
                    _engine_text(proc) or f"exit status {proc.returncode}")
    return image_id or image_name


def _engine_text(proc: subprocess.CompletedProcess) -> str:
    """Return what an engine printed: its stderr, or its stdout when the stderr is empty.

    Args:
        proc: The engine's command as it ended.

    Returns:
        The text, stripped.
    """
    return (proc.stderr or "").strip() or (proc.stdout or "").strip()


def build_image(argv: list[str]) -> int:
    """Run a container engine's build of an image, as image.build_argv() writes it.

    The engine is started with config.child_env() and no input, and what it prints goes
    to this process's stdout and stderr.

    Args:
        argv: The command line, the engine's executable first.

    Returns:
        The engine's exit status.

    Raises:
        OSError: When the engine could not be started.
        subprocess.TimeoutExpired: When it did not end in IMAGE_BUILD_TIMEOUT_SEC.
    """
    return subprocess.run(argv, env=config.child_env(), stdin=subprocess.DEVNULL,
                          check=False, timeout=config.IMAGE_BUILD_TIMEOUT_SEC).returncode


class NoImageError(RuntimeError):
    """The engine does not have the image the containers are started from.

    The message names the engine, the image, the exit status and what the engine
    printed, and each other engine on PATH that has an image of that name.
    """


def image_label_dict(engine_path: str, image: str = "") -> dict[str, str] | None:
    """Return the labels of the image the containers are started from.

    An engine that could not be asked is passed with a warning.

    Args:
        engine_path: The engine's executable, as pick_engine() returns it.
        image: The image's name, or "" for config.CONTAINER_IMAGE.

    Returns:
        {label: value} as the engine reports them (image.label_dict_of), or None when
        the engine could not be asked.

    Raises:
        NoImageError: When the engine's image inspect ends with a non-zero status. The
            message holds what the engine printed (its stderr, or its stdout when the
            stderr is empty), and each other engine on PATH that has an image of that
            name.
    """
    image_name = image or config.CONTAINER_IMAGE
    engine_name = os.path.basename(engine_path)
    try:
        proc = _image_proc(engine_path, image_name)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("could not ask %s for the image %s (%s), so the run goes on "
                       "without knowing it is there", engine_name, image_name, e)
        return None
    if not proc.returncode:
        return image_mod.label_dict_of(proc.stdout or "")
    line_list = [f"{engine_name} does not show the image {image_name} (exit status "
                 f"{proc.returncode}): {_engine_text(proc)}"]
    for other_path in _other_engine_path_list(engine_path):
        other_name = os.path.basename(other_path)
        try:
            other_proc = _image_proc(other_path, image_name)
        except subprocess.TimeoutExpired:
            _note_no_answer(other_path, f"whether it has the image {image_name}")
            continue
        except OSError as e:
            logger.info("could not ask %s for the image %s (%s)", other_name,
                        image_name, e)
            continue
        if not other_proc.returncode:
            line_list.append(f"{other_name} has an image of that name: set "
                             f"CONTAINER_ENGINE={other_name} to use it")
    raise NoImageError("\n".join(line_list))


def _is_internal_network(inspect_text: str) -> bool | None:
    """Read whether a network is internal off what the engine's network inspect printed.

    Args:
        inspect_text: The JSON the engine printed: a list holding one object, whose
            "internal" or "Internal" says it.

    Returns:
        True or False as the engine reports it, or None when the text does not say.
    """
    try:
        entry_list = json.loads(inspect_text)
    except (ValueError, RecursionError):
        return None
    entry_dict = entry_list[0] if isinstance(entry_list, list) and entry_list else entry_list
    if not isinstance(entry_dict, dict):
        return None
    for key in ("internal", "Internal"):
        if isinstance(entry_dict.get(key), bool):
            return entry_dict[key]
    return None


def refuse_open_network(engine_path: str) -> None:
    """Stop when SANDBOX_NETWORK names a network the engine does not report as internal.

    Nothing is asked while SANDBOX_NETWORK is empty.

    Args:
        engine_path: The engine's executable, as pick_engine() returns it.

    Raises:
        RuntimeError: When the engine could not be asked, does not have the network,
            or does not report it as internal. The message says how to make one that
            is.
    """
    network = config.SANDBOX_NETWORK
    if not network:
        return
    engine_name = os.path.basename(engine_path)
    try:
        proc = _engine_proc([engine_path, "network", "inspect", network])
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"could not ask {engine_name} about the network {network} "
                           f"(SANDBOX_NETWORK): {e}. Nothing is started on a network "
                           f"that is not known to be internal. Run it again, or leave "
                           f"SANDBOX_NETWORK empty") from None
    make_text = (f"Make one with `{engine_name} network create --internal <name>`, start "
                 f"the service the tests need on it, and set SANDBOX_NETWORK to its name; "
                 f"or leave SANDBOX_NETWORK empty")
    if proc.returncode:
        engine_text = _engine_text(proc)
        raise RuntimeError(f"{engine_name} does not show the network {network} "
                           f"(SANDBOX_NETWORK; exit status {proc.returncode}): "
                           f"{engine_text}\n{make_text}")
    if _is_internal_network(proc.stdout) is not True:
        raise RuntimeError(f"{engine_name} does not report the network {network} "
                           f"(SANDBOX_NETWORK) as internal, so a container on it could "
                           f"reach outside this machine.\n{make_text}")


def _container_alive(engine_path: str, run_id: str, doubt_alive: bool = True,
                     wait_sec: float | None = None) -> bool:
    """Say whether the run's container is still known to the engine.

    Args:
        engine_path: The engine's executable, as pick_engine() returns it.
        run_id: The run, as new_run_id() names it.
        doubt_alive: What an engine that could not be asked, or whose ps ended with a
            non-zero status, counts as: True has the container, False has none.
            Either way it is logged, as a warning with True and as information with
            False. An engine asked with False that does not answer in time is logged
            as a warning and not asked again by this process (_note_no_answer).
        wait_sec: The seconds the engine is given to answer. None gives
            CONTAINER_COMMAND_TIMEOUT_SEC.

    Returns:
        True when the engine names one, or could not be asked and doubt_alive is True.
    """
    container = container_name(run_id)
    argv = [engine_path, "ps", "--quiet", "--filter", f"name={container}"]
    log = logger.warning if doubt_alive else logger.info
    outcome = "still going" if doubt_alive else "not there"
    try:
        proc = _engine_proc(argv, wait_sec)
    except (OSError, subprocess.TimeoutExpired) as e:
        if isinstance(e, subprocess.TimeoutExpired) and not doubt_alive:
            _note_no_answer(engine_path, f"whether {container} is running")
            return False
        log("could not ask %s what is running (%s), so %s counts as %s",
            os.path.basename(engine_path), e, container, outcome)
        return doubt_alive
    if proc.returncode:
        log("%s ps ended with exit status %d (%s), so %s counts as %s",
            os.path.basename(engine_path), proc.returncode,
            (proc.stderr or "").strip()[-config.LOG_EXCERPT_CHAR:], container, outcome)
        return doubt_alive
    return bool((proc.stdout or "").strip())


def other_live_run_list(out_dir: str, name: str, run_id: str,
                        engine_path: str = "") -> list[str]:
    """Return the other runs that are changing this repository right now.

    They are the runs whose clone is there and whose run has not ended (_run_dead). A
    refill's clone is not counted.

    Args:
        out_dir: One agent's directory on this machine's side (host_dir), as an
            absolute path.
        name: The repository's directory name.
        run_id: This run, which is passed over.
        engine_path: The engine's executable, for telling a run that is going from one
            that was killed.

    Returns:
        The other runs' ids, in order.
    """
    return [other for _, other, refill in _clone_list(out_dir, name)
            if not refill and other != run_id and not _run_dead(other, engine_path)]


def live_run_list(output_dir: str, name: str, engine: str = "") -> list[str]:
    """Return the runs changing this repository right now, whichever agent's they are.

    Every agent's directory on this machine's side (host_dir_list) is looked at.

    Args:
        output_dir: The output directory, as it was passed.
        name: The repository's directory name.
        engine: The container engine asked first (_run_dead), or "" for the configured
            one. When that one is not on PATH the other engines on PATH are asked alone,
            and with no engine on PATH nothing is looked at and the answer is empty.

    Returns:
        ["<agent>/<run id>", ...] in order.
    """
    try:
        engine_path = pick_engine(engine)
    except RuntimeError as e:
        if not _other_engine_path_list(""):
            logger.info("no container engine here (%s), so no run is looked for", e)
            return []
        engine_path = ""
    run_list = []
    for agent, directory in host_dir_list(os.path.abspath(output_dir)):
        run_list += [f"{agent}/{run_id}" for run_id in
                     other_live_run_list(directory, name, "", engine_path)]
    return run_list


def remove_dead_clone(out_dir: str, name: str, engine_path: str = "") -> None:
    """Take away the clones of runs that are no longer going.

    What this finds is what a run that was killed outright left behind: a whole copy of
    the repository. Each agent's directory on this machine's side (host_dir_list) is
    looked at, one at a time, each under the lock of its runs' own directory
    (_sandbox_dir).

    Args:
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        engine_path: The engine's executable, for telling a run that is going from one
            that was killed.
    """
    for agent, directory in host_dir_list(out_dir):
        if not _clone_list(directory, name):
            continue
        with filelock.hold_lock(_sandbox_dir(directory, name), f"the runs of {agent}"):
            for path, run_id, _ in _clone_list(directory, name):
                if _run_dead(run_id, engine_path):
                    logger.info("%s was left by a run that is no longer going, so it "
                                "is being taken away", path)
                    remove_scratch(path)


def work_tree(out_dir: str, agent: str, name: str, run_id: str) -> str:
    """Return where a running change's copy of the repository is, on this machine.

    It is the clone the container makes at /work/<name>, inside the "work" directory of
    the run's scratch directory, and it holds the change's files before they are
    committed anywhere.

    Args:
        out_dir: The output directory, as an absolute path.
        agent: The agent the run is one of, as agent_name() names it.
        name: The repository's directory name.
        run_id: The run holding the clone, as new_run_id() names it.

    Returns:
        The path. It exists only while that change is running.
    """
    return os.path.join(run_file_layout(out_dir, agent, name, run_id).scratch, "work",
                        name)


# The commit the working tree of a repository made when this process first read it, by
# the repository's path
_START_TREE_DICT: dict[str, str] = {}


def start_tree(project_dir: str) -> str:
    """Return the commit the working tree made when this command first read it.

    The first call for a repository returns the commit tree_branch() has made of its
    working tree in this process, or makes one (gitcmd.snapshot_commit). Every later
    call returns that commit, whatever the working tree holds by then. Called before
    this command writes to the working tree, it returns a commit that holds nothing this
    command put there.

    Args:
        project_dir: The target repository, as an absolute path.

    Returns:
        The full commit hash.

    Raises:
        RuntimeError: As gitcmd.snapshot_commit() raises.
    """
    if project_dir not in _START_TREE_DICT:
        _START_TREE_DICT[project_dir] = gitcmd.snapshot_commit(project_dir)
    return _START_TREE_DICT[project_dir]


def forget_start_tree() -> None:
    """Forget the commits start_tree() returns."""
    _START_TREE_DICT.clear()


@contextlib.contextmanager
def tree_branch(project_dir: str, run_id: str, earlier: str = "", commit: str = "",
                engine: str = "") -> Iterator[tuple[str, str]]:
    """Point a branch of the run's own at the commit the working tree makes, in the block.

    The commit is gitcmd.snapshot_commit()'s: HEAD when the files on disk are what HEAD
    holds, and a commit of the tree's own otherwise. The clone a container makes carries
    the branches of what it clones, and is checked out at that commit through the
    branch. The branch is taken away when the block ends. The first commit made here for
    a repository is kept in _START_TREE_DICT. Given a commit, the branch is pointed at
    that one and none is made.

    Args:
        project_dir: The target repository, as an absolute path.
        run_id: The run, as new_run_id() names it.
        earlier: As gitcmd.snapshot_commit() takes it.
        commit: A commit to point the branch at in place of the one the working tree
            makes now. Empty makes that one.
        engine: The container engine the run uses, as remove_dead_tree_branch()
            takes it.

    Yields:
        (the commit, the branch's name).

    Raises:
        RuntimeError: When project_dir is not a git repository with a commit in it,
            or the commit or the branch cannot be made.
    """
    is_new = not commit
    if is_new:
        commit = gitcmd.snapshot_commit(project_dir, earlier)
        _START_TREE_DICT.setdefault(project_dir, commit)
    branch = _TREE_BRANCH.format(run_id=run_id)
    with _branch_lock(project_dir):
        remove_dead_tree_branch(project_dir, engine)
        proc = gitcmd.run(project_dir, "branch", "-f", branch, commit)
    if proc.returncode != 0:
        raise RuntimeError(f"could not make {branch} at {gitcmd.short(commit)}, which "
                           f"the run starts from: {proc.stderr.strip()}")
    if is_new and commit != gitcmd.head_commit(project_dir):
        logger.info("%s holds changes that are not committed, so the run starts from "
                    "%s, a commit of the working tree as it is", project_dir,
                    gitcmd.short(commit))
    try:
        yield commit, branch
    finally:
        with _branch_lock(project_dir):
            _delete_branch(project_dir, branch)


def _engine_path_of(engine: str) -> str:
    """Return the engine's executable as pick_engine() finds it, or "" when it finds none.

    Args:
        engine: The engine asked for, as pick_engine() takes it.

    Returns:
        The path, or "".
    """
    try:
        return pick_engine(engine)
    except RuntimeError:
        return ""


def remove_dead_tree_branch(project_dir: str, engine: str = "") -> list[str]:
    """Take away the branches runs that are no longer going left on their start commit.

    What is found here is what a run that was killed left. The caller holds the lock the
    runs writing branches share.

    Args:
        project_dir: The target repository, as an absolute path.
        engine: The container engine the run looking uses, asked first (_run_dead),
            or "" for the configured one.

    Returns:
        The branches taken away.
    """
    engine_path = _engine_path_of(engine)
    drop_list = []
    for branch in _own_branch_list(project_dir, _TASK_BRANCH_ROOT):
        if not is_tree_branch(branch) or not _run_dead(run_in(branch), engine_path):
            continue
        proc = gitcmd.run(project_dir, "branch", "-D", branch)
        if proc.returncode == 0:
            drop_list.append(branch)
    if drop_list:
        logger.info("%d branch(es) runs that ended left on their start commit were "
                    "taken away: %s", len(drop_list), ", ".join(drop_list))
    return drop_list


def remove_dead_worktree(project_dir: str, engine: str = "") -> list[str]:
    """Take away the worktrees runs that are no longer going left registered.

    What is found here is what a run that was killed left. A worktree is one of them
    when its directory is named as run_file_layout() names a run's ("worktree-<run>" in
    a "sandbox" directory) and that run has ended (_run_dead). Its registration is
    removed whether or not the directory is still there, and the directory with it. The
    caller holds the lock the runs writing branches share.

    Args:
        project_dir: The target repository, as an absolute path.
        engine: The container engine the run looking uses, asked first (_run_dead),
            or "" for the configured one.

    Returns:
        The worktree directories whose registration was removed.
    """
    engine_path = _engine_path_of(engine)
    drop_list = []
    for path in gitcmd.worktree_path_list(project_dir):
        run_id = os.path.basename(path)[len(_WORKTREE_PREFIX):]
        if not os.path.basename(path).startswith(_WORKTREE_PREFIX) \
                or os.path.basename(os.path.dirname(path)) != _PATCH_DIR \
                or not _RUN_IN_NAME.fullmatch(run_id) \
                or not _run_dead(run_id, engine_path):
            continue
        remove_argv = ("worktree", "remove", "--force", path)
        if gitcmd.run(project_dir, *remove_argv).returncode != 0:
            _remove_tree(path, "the worktree of a run that ended")
            if gitcmd.run(project_dir, *remove_argv).returncode != 0:
                continue
        drop_list.append(path)
    if drop_list:
        logger.info("%d worktree(s) runs that ended left registered were taken away: "
                    "%s", len(drop_list), ", ".join(drop_list))
    return drop_list


def is_tree_branch(name: str) -> bool:
    """Say whether a branch is the one a run points at the commit it starts from.

    Args:
        name: A branch name.

    Returns:
        True for a name tree_branch() makes.
    """
    return bool(run_in(name)) and name == _TREE_BRANCH.format(run_id=run_in(name))


def record_dir(out_dir: str, name: str) -> str:
    """Return where the records of the commands that start containers are kept.

    Args:
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.

    Returns:
        The directory, as an absolute path.
    """
    return os.path.join(repo_own_dir(out_dir, name), _RECORD_DIR)


def last_land_path(out_dir: str, name: str) -> str:
    """Return where what the last command put into the working tree is kept, as a diff.

    It is in this tool's own directory of the repository (repo_own_dir), where
    land_in_tree() also makes the copies of what a diff's paths held before it went in.

    Args:
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.

    Returns:
        The path, as an absolute path.
    """
    return os.path.join(repo_own_dir(out_dir, name),
                        f"{_LAND_PREFIX}{_LAND_LAST_NAME}{_PATCH_SUFFIX}")


def _keep_last_land(project_dir: str, before_tree: str, land_pair: tuple[str, str],
                    patch_path: str, last_path: str) -> str:
    """Write what this process has put into a working tree to last_path, over its old copy.

    The diffs this process applied to project_dir before patch_path
    (_LAND_PATCH_DICT) are taken off before_tree, newest first (_earlier_tree), and the
    diff from the tree that leaves to the one patch_path makes of before_tree is
    written: applied in reverse to the working tree as patch_path left it, it takes out
    those diffs and patch_path together. A diff that does not come off whole at the
    lines it went in at (what it touches was written after it was applied) is left out
    of the copy with the ones before it, and the log says so. With no diff before
    patch_path, the copy is patch_path itself, or the diff from before_tree to the
    tree patch_path makes of it where before_tree was read. patch_path is added to the
    diffs the next call takes off, whether or not the copy is written: as that diff
    from before_tree where before_tree was read, and as land_pair otherwise.

    Args:
        project_dir: The target repository, as an absolute path.
        before_tree: The tree the working tree made just before patch_path was
            applied (gitcmd.work_tree_id), or "" when it was not read.
        land_pair: (the commit patch_path was written from, the commit it leads to).
        patch_path: The diff that was applied, as an absolute path.
        last_path: Where the copy goes (last_land_path()).

    Returns:
        last_path, or "" when the copy could not be written, and when the diffs it
        would hold change nothing taken together.
    """
    earlier_list = _LAND_PATCH_DICT.setdefault(project_dir, [])
    start, end, off_count = _earlier_tree(project_dir, before_tree, patch_path,
                                          earlier_list)
    if off_count < len(earlier_list):
        logger.warning("applying %s in reverse takes out the last %d of the %d change(s) "
                       "this command put into the working tree of %s: what the %d before "
                       "them touch changed after they were put in, and stays",
                       last_path, off_count + 1, len(earlier_list) + 1, project_dir,
                       len(earlier_list) - off_count)
        del earlier_list[:len(earlier_list) - off_count]
    earlier_list.append((before_tree, end) if end else land_pair)
    if off_count and gitcmd.run(project_dir, *gitcmd.PATH_DIFF_TUPLE, "--quiet", start,
                                end).returncode == 0:
        logger.info("the changes this command put into the working tree of %s change "
                    "nothing taken together, so there is nothing to take out",
                    project_dir)
        return ""
    if not off_count and end:
        start = before_tree
    try:
        os.makedirs(os.path.dirname(last_path), exist_ok=True)
        with tmpfile.atomic_write(last_path) as tmp_path:
            if not start:
                shutil.copyfile(patch_path, tmp_path)
            elif (proc := gitcmd.write_patch(project_dir, start, end,
                                             tmp_path)).returncode != 0:
                raise OSError(proc.stderr.strip())
    except OSError as e:
        logger.warning("could not keep a copy of %s at %s (%s)", patch_path, last_path, e)
        return ""
    return last_path


# The line "git apply -v" writes, in the C locale, for a hunk it applies at lines other
# than the ones the diff names, and the setting that puts git in that locale
_APPLY_OFFSET_RE = re.compile(r"^Hunk #\d+ succeeded at \d+ \(offset -?\d+ lines?\)\.$",
                              re.M)
_C_LOCALE_DICT = {"LC_ALL": "C"}


def _apply_proc(project_dir: str, *arg_tuple: str, env_dict: dict[str, str] | None = None
                ) -> subprocess.CompletedProcess:
    """Run git apply with gitcmd.APPLY_TUPLE, held to gitcmd.MAX_PATCH_ADDRESS_BYTE.

    Args:
        project_dir: The target repository, as an absolute path.
        *arg_tuple: What follows gitcmd.APPLY_TUPLE: options, then the diff's path.
        env_dict: As gitcmd.run() takes it. None puts nothing in git's environment.

    Returns:
        gitcmd.run()'s return value.
    """
    return gitcmd.run(project_dir, *gitcmd.APPLY_TUPLE, *arg_tuple, env_dict=env_dict,
                      max_address_byte=gitcmd.MAX_PATCH_ADDRESS_BYTE)


def _is_at_line(project_dir: str, patch_path: str, *arg_tuple: str,
                env_dict: dict[str, str] | None = None) -> bool:
    """Say whether git applies a diff whole with every hunk at the lines the diff names.

    git checks the diff (--check) with arg_tuple and gitcmd.APPLY_TUPLE, in the C
    locale, and writes a line of _APPLY_OFFSET_RE for a hunk it applies at other lines.

    Args:
        project_dir: The target repository, as an absolute path.
        patch_path: The diff, as an absolute path.
        *arg_tuple: Further options of git apply, such as "--cached" and "-R".
        env_dict: Settings put in git's environment, such as the index file it
            works with. None puts none.

    Returns:
        True when git ends with exit status 0 and writes no such line.
    """
    proc = _apply_proc(project_dir, *arg_tuple, "--check", "-v", patch_path,
                       env_dict={**(env_dict or {}), **_C_LOCALE_DICT})
    return proc.returncode == 0 and not _APPLY_OFFSET_RE.search(proc.stderr)


def _earlier_tree(project_dir: str, before_tree: str, patch_path: str,
                  pair_list: list[tuple[str, str]]) -> tuple[str, str, int]:
    """Take earlier diffs off a tree, newest first, and put a later diff on it.

    The work is done in an index of this call's own. The repository's own index and
    working tree are not touched. Each earlier diff is written beside that index from
    its pair (gitcmd.write_patch). Taking off stops at the first diff git does not take
    off whole with every hunk at the lines it names (_is_at_line).

    Args:
        project_dir: The target repository, as an absolute path.
        before_tree: The tree the diffs are taken off and patch_path goes on, or ""
            for none.
        patch_path: The later diff, as an absolute path.
        pair_list: The earlier diffs, oldest first, each as (the commit or tree it
            starts from, the one it leads to).

    Returns:
        (the tree left once the diffs that came off are off before_tree, or "" when
        none came off, the tree patch_path makes of before_tree, how many of the
        newest diffs of pair_list came off). Both trees are "" and the count 0 when
        before_tree is "" and when git fails on before_tree or patch_path.
    """
    if not before_tree:
        return "", "", 0
    try:
        with tempfile.TemporaryDirectory(prefix="ctloom-land-") as index_dir:
            env_dict = {"GIT_INDEX_FILE": os.path.join(index_dir, "index")}
            if gitcmd.run(project_dir, "read-tree", before_tree,
                          env_dict=env_dict).returncode \
                    or _apply_proc(project_dir, "--cached", patch_path,
                                   env_dict=env_dict).returncode:
                return "", "", 0
            end_proc = gitcmd.run(project_dir, "write-tree", env_dict=env_dict)
            if end_proc.returncode or gitcmd.run(project_dir, "read-tree", before_tree,
                                                 env_dict=env_dict).returncode:
                return "", "", 0
            off_count = 0
            for index, (from_id, to_id) in enumerate(reversed(pair_list)):
                earlier_path = os.path.join(index_dir, f"{index}{_PATCH_SUFFIX}")
                if gitcmd.write_patch(project_dir, from_id, to_id,
                                      earlier_path).returncode \
                        or not _is_at_line(project_dir, earlier_path, "--cached", "-R",
                                           env_dict=env_dict) \
                        or _apply_proc(project_dir, "--cached", "-R", earlier_path,
                                       env_dict=env_dict).returncode:
                    break
                off_count += 1
            proc = gitcmd.run(project_dir, "write-tree", env_dict=env_dict)
    except OSError as e:
        logger.info("could not take the earlier changes off %s: %s",
                    gitcmd.short(before_tree), e)
        return "", "", 0
    end = end_proc.stdout.strip()
    if not off_count or proc.returncode:
        return "", end, 0
    return proc.stdout.strip(), end, off_count


# What this process put into the working tree, oldest first: land_in_tree()'s return
# value for every diff it applied
_LAND_LIST: list[dict] = []

# The paths of those diffs by repository, relative to it and as git wrote them, each
# with whether one of the diffs added it
_LAND_PATH_DICT: dict[str, dict[bytes, bool]] = {}

# The diffs this process applied to each repository that have not been left out of a
# kept copy (_keep_last_land), oldest first, each as (the commit or tree it starts from,
# the one it leads to), with its hunks at the lines of the working tree it went into
_LAND_PATCH_DICT: dict[str, list[tuple[str, str]]] = {}

# The runs of a request dispatch() made again, oldest first, each as (a count of the
# runs this process had started, taken as it started, which the runs it started count
# above (new_run_id), the branches it was told to carry on from, the unions the cases
# the runs before it found not done are recorded on, how many diffs _LAND_LIST held as
# it started)
_RERUN_LIST: list[tuple[int, list[str], list[str], int]] = []

# The unions carried on from for the cases their tests found not done that the next
# union this process puts into a repository's working tree stands for, by repository,
# each as (the output directory, the repository's directory name, the container engine,
# the branches) (_hold_stand_for)
_STAND_FOR_DICT: dict[str, tuple[str, str, str, list[str]]] = {}


def _hold_stand_for(project_dir: str, out_dir: str, name: str, engine: str,
                    resume_list: Sequence[str], gap_union_list: Sequence[str],
                    request: str) -> None:
    """Keep the unions the next union this process puts into the working tree may stand for.

    They are gap_union_list, and the ones a record of an earlier command whose "resume"
    names a branch of resume_list holds under runrecord.STAND_FOR_KEY that are there
    under a name this tool made up (is_own_branch) and were made for request or carry
    none (_is_request_of).

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        engine: The container engine, or "" for the configured one.
        resume_list: The branches the command was told to carry on from.
        gap_union_list: gap_union_list_of()'s return value for them.
        request: The request the command was given.
    """
    resume_set = set(resume_list)
    record_list = runrecord.read_list(record_dir(out_dir, name)) if resume_set else []
    record_branch_list = [
        branch for record_dict in record_list
        if resume_set & set(record_dict.get("resume") or [])
        for branch in (record_dict.get("result") or {}).get(runrecord.STAND_FOR_KEY, [])]
    branch_list = [*gap_union_list, *(
        branch for branch in dict.fromkeys(record_branch_list)
        if branch not in gap_union_list and is_own_branch(branch)
        and branch_tip(project_dir, branch)
        and _is_request_of(project_dir, branch, request))]
    _STAND_FOR_DICT.pop(project_dir, None)
    if branch_list:
        _STAND_FOR_DICT[project_dir] = (out_dir, name, engine, branch_list)


def _drop_stand_for(project_dir: str, before_list: Sequence[str], commit: str
                    ) -> list[str]:
    """Delete the unions a union just put into the working tree stands for, and say which.

    They are the ones held (_hold_stand_for) that the working tree held just before
    the union went in (before_list) or holds now (is_gap_union), and whose branch does
    not reach the commit that went in, deleted as _drop_resume_branch_list() deletes
    them. Those of them it leaves are still held, and the other held unions no longer
    are. What is held is changed only once that is through, so a call that does not
    get through leaves every one of them held.

    Args:
        project_dir: The target repository, as an absolute path.
        before_list: The held unions is_gap_union() said yes for just before the union
            went in.
        commit: The commit holding what went in.

    Returns:
        The branches deleted, in the order they were held.
    """
    if project_dir not in _STAND_FOR_DICT:
        return []
    out_dir, name, engine, branch_list = _STAND_FOR_DICT[project_dir]
    hold_list = [branch for branch in branch_list
                 if (branch in before_list or is_gap_union(project_dir, branch))
                 and not gitcmd.is_ancestor(project_dir, commit, f"refs/heads/{branch}")]
    drop_list = _drop_resume_branch_list(project_dir, out_dir, name, engine, hold_list)
    remain_list = [branch for branch in hold_list
                   if branch not in drop_list and branch_tip(project_dir, branch)]
    if remain_list:
        _STAND_FOR_DICT[project_dir] = (out_dir, name, engine, remain_list)
    else:
        _STAND_FOR_DICT.pop(project_dir, None)
    return drop_list


def stand_for_list(project_dir: str) -> list[str]:
    """Return the unions the next union this process puts into the working tree stands for.

    They are the ones _hold_stand_for() kept.

    Args:
        project_dir: The target repository, as an absolute path.

    Returns:
        The branches, or [] for none.
    """
    return list(_STAND_FOR_DICT.get(project_dir, ("", "", "", []))[3])


def land_list() -> list[dict]:
    """Return what this process has put into the working tree so far.

    Returns:
        land_in_tree()'s return value for every diff it applied, oldest first.
    """
    return list(_LAND_LIST)


def reset_land_list() -> None:
    """Empty what land_list() returns and what this process keeps beside it.

    That is the paths and diffs kept beside it, the runs of a request made again
    (_RERUN_LIST) and the unions a union stands for (_STAND_FOR_DICT).
    """
    _LAND_LIST.clear()
    _LAND_PATH_DICT.clear()
    _LAND_PATCH_DICT.clear()
    _RERUN_LIST.clear()
    _STAND_FOR_DICT.clear()


def _setting_file_path_list(project_dir: str, path_list: list[str]) -> list[str]:
    """Return the paths among path_list that a command would read as its settings.

    They are the files named config.ENV_FILE_NAME, in whatever directory and whatever
    the case of the letters, that config.git_index_state() does not answer
    config.INDEX_HOLD for.

    Args:
        project_dir: The target repository, as an absolute path.
        path_list: Paths relative to the repository.

    Returns:
        Those paths, in the order of path_list.
    """
    return [path for path in path_list
            if os.path.basename(path).casefold() == config.ENV_FILE_NAME
            and config.git_index_state(os.path.join(project_dir, path))
            != config.INDEX_HOLD]


# How many symbolic links are followed from one place git reads its settings from
_MAX_LINK_HOP = 40


def _place_on_way_list(place: str) -> list[str]:
    """Return a path and every path it stands for on the way through symbolic links.

    Args:
        place: An absolute path. Nothing has to be there.

    Returns:
        place, then, for each symbolic link met from its start, the path with the
        link replaced by what the link points at, at most _MAX_LINK_HOP of them,
        then each of those links itself, and last the path with every link resolved.
    """
    way_list, link_list = [place], []
    while len(way_list) <= _MAX_LINK_HOP:
        part_list = way_list[-1].split(os.sep)
        for part_count in range(2, len(part_list) + 1):
            head = os.sep.join(part_list[:part_count])
            if os.path.islink(head):
                break
        else:
            break
        try:
            target = os.readlink(head)
        except OSError:
            break
        link_list.append(head)
        way_list.append(os.path.join(os.path.dirname(head), target,
                                     *part_list[part_count:]))
    return [*way_list, *link_list, os.path.realpath(place)]


# The bytes a shell reads as quoting, which are left out of a name and of a command
# when the one is looked for in the other
_SHELL_QUOTE_BYTE = b"\\'\""

# The bytes a word of a command ends at on either side: white space, the separator
# of a path, quotes, and what a shell reads as an operator or a separator
_WORD_END_BYTE = b" \t\n\r\v\f/'\"`;&|()<>=:,{}"

# What a shell variable is written with ahead of its one character, and the
# characters that stand for a variable by themselves: a word starts right behind one
_VARIABLE_MARK = b"$"
_VARIABLE_BYTE = b"0123456789@*?$!#-"

# What a value git hands a shell starts with, ahead of the command
_SHELL_MARK = b"!"

# What a word starts with ahead of an argument written joined to it: a short option,
# which is _OPTION_MARK and one letter or digit, and the mark of a file of arguments
_OPTION_MARK = b"-"
_ARGUMENT_FILE_MARK = b"@"


def _is_word_start(command: bytes, start: int) -> bool:
    """Say whether a word of a command, or an argument joined to a word's head, starts here.

    One does at the start of the command, behind a byte of _WORD_END_BYTE, behind a
    variable written as _VARIABLE_MARK and one byte of _VARIABLE_BYTE, and behind a
    short option or _ARGUMENT_FILE_MARK that itself starts the command or stands
    behind a byte of _WORD_END_BYTE.

    Args:
        command: The command.
        start: The place, as an index into command.

    Returns:
        True when one starts there.
    """
    def is_head(index: int) -> bool:
        """Say whether index is the start of the command or follows a word's end."""
        return index == 0 or command[index - 1:index] in _WORD_END_BYTE

    before = command[start - 1:start]
    if is_head(start):
        return True
    if start >= 2 and command[start - 2:start - 1] == _VARIABLE_MARK \
            and before in _VARIABLE_BYTE:
        return True
    if before == _ARGUMENT_FILE_MARK and is_head(start - 1):
        return True
    return start >= 2 and before.isalnum() and before.isascii() \
        and command[start - 2:start - 1] == _OPTION_MARK and is_head(start - 2)


def _has_word(command: bytes, name: bytes, root: bytes = b"",
              is_alias: bool = False) -> bool:
    """Say whether a name stands in a command as a word of its own.

    It does where a word starts at its first byte (_is_word_start) and the end of
    the command or a byte of _WORD_END_BYTE follows its last; with root, not where
    it stands behind a directory outside root (_is_outside_dir), and with is_alias,
    only where _is_alias_place() says.

    Args:
        command: The command.
        name: The name. An empty one stands nowhere.
        root: The top of a working tree, as an absolute path with no symbolic link
            in it, or b"" for none.
        is_alias: Whether the command is an alias's.

    Returns:
        True when one place the name is written at is bounded that way.
    """
    start = command.find(name) if name else -1
    while start >= 0:
        end = start + len(name)
        if _is_word_start(command, start) \
                and (end == len(command) or command[end:end + 1] in _WORD_END_BYTE) \
                and not (root and _is_outside_dir(command, start, root)) \
                and (not is_alias or _is_alias_place(command, start, end)):
            return True
        start = command.find(name, start + 1)
    return False


# The bytes a directory written as it is, with nothing a shell expands or reads as
# quoting in it, is made of, and the bytes that may stand ahead of one
_PLAIN_PATH_BYTE = (b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                    b"._+%@-/~")
_PLAIN_GAP_BYTE = b" \t\n\r\v\f;&|<>=:,"

# The quotes a directory written as it is may stand inside
_PLAIN_QUOTE_BYTE = b"'\""

# What a short option, or the mark of a file of arguments, joined ahead of such a
# directory is
_PLAIN_JOIN_HEAD = re.compile(rb"-[A-Za-z0-9]|@")

# What the home directory is written as at the head of a path
_HOME_MARK = b"~"


def _plain_dir(command: bytes, start: int) -> bytes | None:
    """Return the directory written as it is right ahead of a place of a command.

    The directory runs back from the "/" ahead of the place over _PLAIN_PATH_BYTE,
    to the start of the command, to a byte of _PLAIN_GAP_BYTE, or to a byte of
    _PLAIN_QUOTE_BYTE that starts the command or stands behind a byte of
    _PLAIN_GAP_BYTE. A short option or the mark of a file of arguments at its head
    (_PLAIN_JOIN_HEAD) is left out, and _HOME_MARK at its head, with or without a
    user's name, is that home directory.

    Examples:
        (b"vim /usr/lib/x", 9)   -> b"/usr/"
        (b"vim other/lib", 10)   -> b"other/"
        (b"vim $(pwd)/lib", 11)  -> None
        (b'vim "$PWD"/lib', 11)  -> None
        (b"vim ~+/lib", 7)       -> None
        (b"vim lib", 4)          -> None

    Args:
        command: The command.
        start: The place, as an index into command.

    Returns:
        The directory, ending with "/", or None when no "/" stands right ahead of
        the place, when anything else runs up to it, or when _HOME_MARK at its head
        names no home directory.
    """
    if start < 1 or command[start - 1:start] != b"/":
        return None
    head = start - 1
    while head > 0 and command[head - 1:head] in _PLAIN_PATH_BYTE:
        head -= 1
    if head > 0:
        before = command[head - 1:head]
        is_open_quote = before in _PLAIN_QUOTE_BYTE and (
            head == 1 or command[head - 2:head - 1] in _PLAIN_GAP_BYTE)
        if before not in _PLAIN_GAP_BYTE and not is_open_quote:
            return None
    directory = command[head:start]
    if match := _PLAIN_JOIN_HEAD.match(directory):
        directory = directory[match.end():]
    if directory.startswith(_HOME_MARK):
        directory = os.path.expanduser(directory)
        if directory.startswith(_HOME_MARK):
            return None
    return directory


def _is_outside_dir(command: bytes, start: int, root: bytes) -> bool:
    """Say whether a place of a command stands behind a directory outside a working tree.

    The directory is _plain_dir()'s. It lies outside when it starts with "/" and its
    real path is neither root nor under root.

    Examples:
        (b"vim /usr/lib/x", 9, b"/w")   -> True
        (b"vim $(pwd)/lib", 11, b"/w")  -> False
        (b'vim "$PWD"/lib', 11, b"/w")  -> False
        (b"vim /w/lib", 7, b"/w")       -> False
        (b"vim other/lib", 10, b"/w")   -> False

    Args:
        command: The command.
        start: The place, as an index into command.
        root: The top of the working tree, as an absolute path with no symbolic
            link in it.

    Returns:
        True for a place behind such a directory.
    """
    directory = _plain_dir(command, start)
    if directory is None or not directory.startswith(b"/"):
        return False
    real = os.path.realpath(directory)
    return real != root and not real.startswith(root + b"/")


# The bytes a word of an alias's command ends at: white space and what a shell reads
# as an operator
_ALIAS_WORD_GAP_BYTE = b" \t\n\r\v\f;&|<>"


def _is_alias_place(command: bytes, start: int, end: int) -> bool:
    """Say whether a name at a place of an alias's command counts as a head of a path.

    It does when the word it stands in, bounded by _ALIAS_WORD_GAP_BYTE, holds a "/"
    and the directory written as it is right ahead of it (_plain_dir), if any, is
    absolute or the top of the working tree.

    Examples:
        (b"git rebase origin/main", 18, 22)  -> False
        (b"git checkout main", 13, 17)       -> False
        (b"python3 tools/clean", 8, 13)      -> True
        (b'python3 "$PWD"/tools', 15, 20)    -> True
        (b"python3 ./tools", 10, 15)         -> True

    Args:
        command: The command.
        start: Where the name starts, as an index into command.
        end: Where it ends.

    Returns:
        True for such a place.
    """
    head = start
    while head > 0 and command[head - 1:head] not in _ALIAS_WORD_GAP_BYTE:
        head -= 1
    tail = end
    while tail < len(command) and command[tail:tail + 1] not in _ALIAS_WORD_GAP_BYTE:
        tail += 1
    if b"/" not in command[head:tail]:
        return False
    directory = _plain_dir(command, start)
    return directory is None or directory.startswith(b"/") \
        or os.path.normpath(directory) == os.curdir.encode()


def _path_name_list(raw: bytes) -> list[bytes]:
    """Return every head of a path, each followed by its forms without a dot's suffix.

    A form leaves out what a dot in the head's last part starts.

    Examples:
        b"tools/clean/__main__.py" -> [b"tools", b"tools/clean",
                                       b"tools/clean/__main__.py",
                                       b"tools/clean/__main__"]
        b"clean.d/run"             -> [b"clean.d", b"clean", b"clean.d/run"]
        b".gitmessage.txt"         -> [b".gitmessage.txt", b".gitmessage"]

    Args:
        raw: A path relative to the repository, as git wrote it.

    Returns:
        Those names, shortest head first.
    """
    part_list = raw.split(b"/")
    name_list = []
    for count, part in enumerate(part_list, 1):
        head = b"/".join(part_list[:count])
        name_list.append(head)
        name_list += [head[:len(head) - len(part) + index]
                      for index in range(1, len(part)) if part[index:index + 1] == b"."]
    return name_list


def _command_name_path_list(raw_path_list: Sequence[bytes],
                            command_list: Sequence[bytes],
                            root: bytes = b"",
                            alias_list: Sequence[bytes] = ()) -> list[bytes]:
    """Return the paths among raw_path_list whose name is written in a command.

    A path is one of them when the name of its last part stands as a word of its
    own (_has_word) in one of the commands, _SHELL_MARK at the head of the command
    left out, as it is or with _SHELL_QUOTE_BYTE left out of both. With root, a path
    is one of them as well when a name of _path_name_list() does, other than behind
    a directory outside root, and in a command of alias_list only at a place
    _is_alias_place() says.

    Args:
        raw_path_list: Paths relative to the repository, as git wrote them.
        command_list: The commands, each as the bytes of a setting's value.
        root: The top of the working tree, as an absolute path with no symbolic
            link in it, or b"" to look for the name of the last part alone.
        alias_list: The commands among command_list that only an alias holds.

    Returns:
        Those paths, in the order of raw_path_list.
    """
    alias_set = set(alias_list)
    form_list = [(text, text.translate(None, _SHELL_QUOTE_BYTE), command in alias_set)
                 for command in command_list
                 for text in [command.removeprefix(_SHELL_MARK)]]
    answer_dict = {}

    def is_in_command(name: bytes, outside_root: bytes) -> bool:
        """Say whether a name stands as a word in a command, in either form."""
        if (name, outside_root) not in answer_dict:
            bare_name = name.translate(None, _SHELL_QUOTE_BYTE)
            answer_dict[name, outside_root] = any(
                _has_word(text, name, outside_root, is_alias and bool(outside_root))
                or _has_word(bare, bare_name, outside_root,
                             is_alias and bool(outside_root))
                for text, bare, is_alias in form_list)
        return answer_dict[name, outside_root]

    return [raw for raw in raw_path_list
            if is_in_command(os.path.basename(raw), b"")
            or (root and any(is_in_command(name, root) for name in _path_name_list(raw)))]


def _git_setting_path_list(project_dir: str, raw_path_list: list[bytes],
                           link_path_list: Sequence[bytes] = ()) -> list[bytes] | None:
    """Return the paths among raw_path_list that are, or are near, a place of git settings.

    A path is one when git reads its settings from it, or when it lies above or below a
    place git reads them from, or beside one (_is_beside). The places are
    gitcmd.setting_place_list()'s and the symbolic links of link_path_list that lead to
    no directory and whose name is written in a command of the settings or of the
    environment (gitcmd.setting_command_dict, _command_name_path_list), each with the
    paths it stands for through symbolic links and with those links themselves
    (_place_on_way_list). One counts where the entry it names lies in the working tree,
    the directory above it resolved through its links. A place that is the top of the
    working tree makes every path one of them. A path whose own name is written in one
    of those commands is one of them wherever it lies, and so is one a head of which, or
    such a head with what a dot in its last part starts left out, is written there other
    than behind a directory outside the working tree, in a command only an alias holds
    only at a place _is_alias_place() says (_path_name_list, _is_outside_dir).

    Args:
        project_dir: The target repository, as an absolute path.
        raw_path_list: Paths relative to the repository, as git wrote them.
        link_path_list: The symbolic links the working tree holds, in the same form.

    Returns:
        Those paths, in the order of raw_path_list, or None when git cannot list the
        places or its settings.
    """
    place_list = gitcmd.setting_place_list(project_dir)
    value_dict = None if place_list is None else gitcmd.setting_command_dict(project_dir)
    if value_dict is None:
        return None
    command_list = [os.fsencode(value) for value in value_dict]
    alias_list = [os.fsencode(value) for value, is_alias in value_dict.items() if is_alias]
    place_list += [os.path.join(project_dir, os.fsdecode(raw))
                   for raw in _command_name_path_list(link_path_list, command_list)
                   if not os.path.isdir(os.path.join(os.fsencode(project_dir), raw))]
    root = os.fsencode(os.path.realpath(project_dir))
    name_set = set(_command_name_path_list(raw_path_list, command_list, root,
                                           alias_list))
    inside_set = set()
    for place in place_list:
        for way in _place_on_way_list(place):
            way = os.fsencode(way.rstrip(os.sep))
            name = os.path.basename(way)
            entry = os.path.realpath(way) if name in (b".", b"..") else os.path.join(
                os.path.realpath(os.path.dirname(way)), name)
            if entry.startswith(root + b"/"):
                inside_set.add(entry[len(root) + 1:])
            elif entry == root:
                return list(raw_path_list)
    return [raw for raw in raw_path_list
            if raw in name_set
            or any(raw == inside or raw.startswith(inside + b"/")
                   or inside.startswith(raw + b"/") or _is_beside(raw, inside)
                   for inside in inside_set)]


def _is_beside(raw: bytes, place: bytes) -> bool:
    """Say whether a path is, or lies under, an entry named as the place with a suffix.

    The entry is one of the place's directory whose name is the place's name with a dot
    and more after it.

    Examples:
        (b"scripts/clean.js", b"scripts/clean")     -> True
        (b"clean.d/run", b"clean")                  -> True
        (b"scripts/cleaner.js", b"scripts/clean")   -> False
        (b"other/clean.js", b"scripts/clean")       -> False

    Args:
        raw: A path relative to the repository.
        place: A place, relative to the repository.

    Returns:
        True for such a path.
    """
    parent, name = os.path.split(place)
    if parent:
        if not raw.startswith(parent + b"/"):
            return False
        raw = raw[len(parent) + 1:]
    return raw.split(b"/", 1)[0].startswith(name + b".")


# The names of the files git reads the ignore rules and the attributes of a directory
# from
_IGNORE_FILE_NAME = b".gitignore"
_ATTRIBUTE_FILE_NAME = b".gitattributes"


def _tree_entry_dict(project_dir: str, commit: str
                     ) -> dict[bytes, tuple[bytes, str]] | None:
    """Return every file a commit holds, with its mode and the id of its contents.

    Args:
        project_dir: The target repository, as an absolute path.
        commit: The commit.

    Returns:
        {the path relative to the repository, as git wrote it: (the mode, the object
        id)}, a repository of its own being one entry (gitcmd.GITLINK_MODE). None
        when git cannot list the commit's files.
    """
    proc = gitcmd.run(project_dir, "ls-tree", "-r", "-z", "--full-tree", commit,
                      text=False)
    if proc.returncode != 0:
        return None
    entry_dict = {}
    for entry in proc.stdout.split(b"\0"):
        meta, _, raw = entry.partition(b"\t")
        field_list = meta.split()
        if raw and len(field_list) == 3:
            entry_dict[raw] = (field_list[0], field_list[2].decode("ascii"))
    return entry_dict


def _ignore_answer(project_dir: str, raw_path_list: list[bytes], rule_dir: str = ""
                   ) -> tuple[set[bytes], list[bytes]]:
    """Ask git which of the paths its ignore rules cover, whether or not it tracks them.

    The rules are the ones git reads for the working tree: the .gitignore files on
    disk, the repository's info/exclude and core.excludesFile. When git stops at a
    path it gives no answer for (one below a symbolic link), every path is asked for
    alone.

    Args:
        project_dir: The target repository, as an absolute path.
        raw_path_list: Paths relative to the repository, as git wrote them.
        rule_dir: A directory whose .gitignore files are read in place of the working
            tree's, as an absolute path. Empty reads the working tree's.

    Returns:
        (the paths a rule covers, the paths git gave no answer for).
    """
    if not raw_path_list:
        return set(), []
    proc = gitcmd.check_ignore(project_dir, raw_path_list, rule_dir)
    if proc.returncode in (gitcmd.IGNORE_STATUS, gitcmd.NO_IGNORE_STATUS):
        return {raw[2:] for raw in proc.stdout.split(b"\0") if raw}, []
    ignore_set, no_answer_list = set(), []
    for raw in raw_path_list:
        proc = gitcmd.check_ignore(project_dir, [raw], rule_dir)
        if proc.returncode == gitcmd.IGNORE_STATUS:
            ignore_set.add(raw)
        elif proc.returncode != gitcmd.NO_IGNORE_STATUS:
            logger.info("git did not say whether it ignores %s in %s: %s",
                        gitcmd.shown_text(raw), project_dir,
                        gitcmd.shown_text(proc.stderr).strip())
            no_answer_list.append(raw)
    return ignore_set, no_answer_list


def _write_ignore_file(project_dir: str, entry_dict: dict[bytes, tuple[bytes, str]],
                       rule_dir: str) -> bool:
    """Write the .gitignore files a commit holds under a directory, each at its path.

    Only regular files are written. A path with a ".." part is passed over.

    Args:
        project_dir: The target repository, as an absolute path.
        entry_dict: The commit's files, as _tree_entry_dict() returns them.
        rule_dir: The directory they are written under, as an absolute path.

    Returns:
        False when git could not read one of them, or one holds more than
        tmpfile.READ_MAX_BYTE bytes.

    Raises:
        OSError: When one of them cannot be written.
    """
    for raw, (mode, object_id) in entry_dict.items():
        if os.path.basename(raw) != _IGNORE_FILE_NAME or b".." in raw.split(b"/") \
                or not mode.startswith(_REGULAR_MODE_PREFIX.encode("ascii")):
            continue
        size_text = gitcmd.run(project_dir, "cat-file", "-s", object_id).stdout.strip()
        if not size_text.isdigit() or int(size_text) > tmpfile.READ_MAX_BYTE:
            return False
        blob_proc = gitcmd.run(project_dir, "cat-file", "blob", object_id, text=False)
        if blob_proc.returncode != 0:
            return False
        target = os.path.join(os.fsencode(rule_dir), raw)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as handle:
            handle.write(blob_proc.stdout)
    return True


def _ignore_path_list(project_dir: str, entry_dict: dict[bytes, tuple[bytes, str]],
                      raw_path_list: list[bytes], after_raw_list: list[bytes],
                      index_set: set[bytes]) -> tuple[set[bytes], list[bytes]]:
    """Return the paths git would ignore once a diff is in the working tree.

    A path the diff touches is one of them when a rule the working tree holds as it
    is covers it. For a diff that touches a .gitignore file, a path of after_raw_list
    is one of them too when a rule of the .gitignore files the diff leads to covers
    it. A path the index names is none of them, whatever rule covers it.

    Args:
        project_dir: The target repository, as an absolute path.
        entry_dict: The files of the commit the diff leads to, as _tree_entry_dict()
            returns them.
        raw_path_list: The paths the diff touches, relative to the repository, as git
            wrote them.
        after_raw_list: The paths asked for under the rules the diff leads to: the
            ones the working tree holds once the diff is in it that it did not hold
            before this command, in the same form.
        index_set: The paths the index names.

    Returns:
        (the paths git would ignore, the paths git gave no answer for, in the order
        asked). When git cannot read the commit's .gitignore files, or they cannot be
        written where git reads them, every path is one git gave no answer for.
    """
    ignore_set, no_answer_list = _ignore_answer(project_dir, raw_path_list)
    if any(os.path.basename(raw) == _IGNORE_FILE_NAME for raw in raw_path_list):
        own_set, own_no_answer_list = set(), []
        try:
            with tempfile.TemporaryDirectory(prefix="ctloom-ignore-") as rule_dir:
                if not _write_ignore_file(project_dir, entry_dict, rule_dir):
                    no_answer_list = list(raw_path_list)
                own_set, own_no_answer_list = _ignore_answer(
                    project_dir, after_raw_list, rule_dir)
        except OSError as e:
            logger.info("could not write the .gitignore files the change leads to "
                        "where git reads them: %s", e)
            no_answer_list = list(raw_path_list)
        ignore_set |= own_set
        no_answer_list += own_no_answer_list
    ignore_set -= index_set
    return ignore_set, [raw for raw in dict.fromkeys(no_answer_list)
                        if raw not in ignore_set]


def _filter_path_dict(project_dir: str, commit: str, raw_path_list: list[bytes],
                      after_raw_list: list[bytes]) -> dict[bytes, str] | None:
    """Return the paths git would run a settings' filter driver on with a diff in the tree.

    A path is one of them when its "filter" attribute names a driver the git settings
    give a program (gitcmd.filter_name_set): by the attributes the working tree holds
    as it is, or with any file of gitcmd.attribute_file_list() that is there read as
    core.attributesFile names it, for a path the diff touches, or, for a diff that
    touches a .gitattributes file, by the attributes of the commit the diff leads to,
    for a path of after_raw_list. Settings that define no such driver make no path
    one.

    Args:
        project_dir: The target repository, as an absolute path.
        commit: The commit the diff leads to.
        raw_path_list: The paths the diff touches, relative to the repository, as git
            wrote them.
        after_raw_list: The paths asked for under the attributes the diff leads to,
            in the same form.

    Returns:
        {path: the driver's name}, or None when git gave no answer or the directory
        the commit's attributes are read through could not be made.
    """
    try:
        name_set = gitcmd.filter_name_set(project_dir)
    except RuntimeError as e:
        logger.info("%s", e)
        return None
    if not name_set:
        return {}
    filter_dict = gitcmd.filter_by_path_dict(project_dir, raw_path_list)
    attribute_list = gitcmd.attribute_file_list(project_dir)
    if filter_dict is None or attribute_list is None:
        return None
    file_dict_list = []
    for attribute_file in attribute_list:
        if os.path.isfile(attribute_file):
            file_dict_list.append(gitcmd.filter_by_path_dict(
                project_dir, raw_path_list, attribute_file=attribute_file))
            if file_dict_list[-1] is None:
                return None
    after_dict = {}
    if any(os.path.basename(raw) == _ATTRIBUTE_FILE_NAME for raw in raw_path_list):
        try:
            with tempfile.TemporaryDirectory(prefix="ctloom-attr-") as index_dir:
                env_dict = {"GIT_INDEX_FILE": os.path.join(index_dir, "index")}
                if gitcmd.run(project_dir, "read-tree", commit,
                              env_dict=env_dict).returncode != 0:
                    return None
                after_dict = gitcmd.filter_by_path_dict(project_dir, after_raw_list,
                                                        env_dict)
        except OSError as e:
            logger.info("could not read the attributes %s leads to: %s",
                        gitcmd.short(commit), e)
            return None
        if after_dict is None:
            return None
    # A path counts under any set of attributes; the working tree's driver is the
    # one named where more than one gives it one
    return {raw: name for one_dict in (after_dict, *file_dict_list, filter_dict)
            for raw, name in one_dict.items() if name in name_set}


class LandRefusal(CommandError):
    """A diff that is not put into a working tree for the paths it touches.

    The same diff is refused however often it is made or applied again, until the
    repository's own files, rules or settings change. Its command applies the diff
    by hand.
    """


def _path_text(raw_path_list: Sequence[bytes]) -> str:
    """Name paths on one line, each as gitcmd.shown_text() writes it.

    Args:
        raw_path_list: Paths as git wrote them.

    Returns:
        The line.
    """
    return one_line(list_text([gitcmd.shown_text(raw) for raw in raw_path_list]))


def _diff_path_dict(project_dir: str, base: str, commit: str) -> dict[bytes, bool]:
    """Return the paths one commit holds differently from another.

    Args:
        project_dir: The target repository, as an absolute path.
        base: The commit the change starts at.
        commit: The commit the change leads to.

    Returns:
        {the path relative to the repository, as git wrote it: True when base does
        not hold it} (gitcmd.add_by_path_dict).

    Raises:
        RuntimeError: When git cannot compare the two.
    """
    return gitcmd.add_by_path_dict(project_dir, base, commit)


def _refusal_text(project_dir: str, base: str, commit: str, path_dict: dict[bytes, bool],
                  earlier_dict: dict[bytes, bool] | None = None) -> str:
    """Say what keeps a diff out of the working tree for the paths it touches, or "".

    A diff is kept out when, once it is in the working tree,
    - a file a command would read as its settings is written (_setting_file_path_list),
    - a place git reads its settings from is written, or a path above or below one, or a
      file whose name, or the name of a symbolic link base holds that leads to it, is
      written in a command of the git settings or of the environment, or git cannot list
      those places or its settings (_git_setting_path_list),
    - a path it wrote does not show in "git status": one at or under a repository of
      its own (gitcmd.nest_dir_of, over the entries of that kind in base, in commit
      and in the index), one the index names with assume-unchanged or skip-worktree,
      one git ignores or does not say that for (_ignore_path_list), or
    - git runs a filter driver of its settings on a path it wrote, or which paths git
      runs one on cannot be told (_filter_path_dict).
    Under the .gitignore files the diff leads to, the paths asked for are the ones
    the diff adds and the ones an earlier diff added that commit holds; under the
    .gitattributes files it leads to, every path the diff touches and every path an
    earlier diff touched that commit holds.

    Args:
        project_dir: The target repository, as an absolute path.
        base: The commit the diff starts at.
        commit: The commit the diff leads to.
        path_dict: The paths the diff touches, as _diff_path_dict() returns them.
        earlier_dict: The paths this command, or the run carried on from, put into
            the working tree before, in the same form. None for none.

    Returns:
        A clause naming the paths and what they are, or "" when the diff is none of
        the above.
    """
    raw_path_list = list(path_dict)
    earlier_dict = earlier_dict or {}
    setting_path_list = _setting_file_path_list(
        project_dir, [raw.decode("utf-8", "replace") for raw in raw_path_list])
    if setting_path_list:
        return (f"the change writes {one_line(list_text(setting_path_list))}, which git "
                f"does not track in {project_dir} and a command run from its directory "
                f"would read as its settings")
    base_dict = _tree_entry_dict(project_dir, base)
    git_setting_list = _git_setting_path_list(
        project_dir, raw_path_list,
        [raw for raw, (mode, _) in (base_dict or {}).items()
         if mode == gitcmd.SYMLINK_MODE])
    if git_setting_list is None:
        return (f"the change writes {_path_text(raw_path_list)}, and git did not say "
                f"which files it reads its settings from in {project_dir}")
    if git_setting_list:
        return (f"the change writes {_path_text(git_setting_list)}, which git reads "
                f"its settings from in {project_dir} (a configuration file, a file or "
                f"a directory a setting of its configuration names, its hooks, a "
                f"program a setting or a variable of the environment names, or a "
                f"directory PATH lists)")
    no_answer_text = (f"the change writes {_path_text(raw_path_list)}, and git did not "
                      f"say whether it ignores that in {project_dir}")
    commit_dict = _tree_entry_dict(project_dir, commit)
    index_list = gitcmd.index_entry_list(project_dir)
    if base_dict is None or commit_dict is None or index_list is None:
        return no_answer_text
    # == A repository of its own, at or above a path ==========
    link_set = {raw for entry_dict in (base_dict, commit_dict)
                for raw, (mode, _) in entry_dict.items() if mode == gitcmd.GITLINK_MODE}
    link_set.update(raw for _, mode, raw in index_list if mode == gitcmd.GITLINK_MODE)
    nest_dict = {raw: gitcmd.nest_dir_of(project_dir, raw, link_set)
                 for raw in raw_path_list}
    nest_path_list = [raw for raw in raw_path_list if nest_dict[raw]]
    if nest_path_list:
        nest_dir_list = list(dict.fromkeys(nest_dict[raw] for raw in nest_path_list))
        return (f"the change writes {_path_text(nest_path_list)}, at or under "
                f"{_path_text(nest_dir_list)}, a repository of its own in "
                f"{project_dir}, so git status would not show it")
    # == A file git is set not to look at =====================
    flag_set = _flag_path_set(index_list)
    flag_path_list = [raw for raw in raw_path_list if raw in flag_set]
    if flag_path_list:
        return (f"the change writes {_path_text(flag_path_list)}, which git is set not "
                f"to look at in {project_dir} (assume-unchanged or skip-worktree), so "
                f"git status would not show it")
    # == A path git ignores ===================================
    add_list = [raw for raw, is_add in path_dict.items() if is_add]
    earlier_list = [raw for raw, is_add in earlier_dict.items()
                    if is_add and raw in commit_dict and raw not in add_list]
    ignore_set, no_answer_list = _ignore_path_list(
        project_dir, commit_dict, raw_path_list, add_list + earlier_list,
        {raw for _, _, raw in index_list})
    ignore_path_list = [raw for raw in raw_path_list if raw in ignore_set]
    if ignore_path_list:
        return (f"the change writes {_path_text(ignore_path_list)}, which git "
                f"ignores in {project_dir}, by the rules there now or by the "
                f".gitignore files the change itself writes, so git status would not "
                f"show it")
    ignore_path_list = [raw for raw in earlier_list if raw in ignore_set]
    if ignore_path_list:
        return (f"the change writes .gitignore files whose rules cover "
                f"{_path_text(ignore_path_list)}, which was put into the working tree "
                f"of {project_dir} before it, so git status would no longer show it")
    if no_answer_list:
        return (f"the change writes {_path_text(no_answer_list)}, and "
                f"git did not say whether it ignores that in {project_dir}")
    # == A path git runs a filter driver on ===================
    earlier_list = [raw for raw in earlier_dict
                    if raw in commit_dict and raw not in path_dict]
    filter_dict = _filter_path_dict(project_dir, commit, raw_path_list,
                                    raw_path_list + earlier_list)
    if filter_dict is None:
        return (f"the change writes {_path_text(raw_path_list)}, and git did not say "
                f"which filter driver of its settings it runs on that in {project_dir}")
    lead = "the change writes"
    filter_path_list = [raw for raw in raw_path_list if raw in filter_dict]
    if not filter_path_list:
        lead = "the change writes .gitattributes files that name a filter for"
        filter_path_list = [raw for raw in earlier_list if raw in filter_dict]
    if filter_path_list:
        name_list = sorted({filter_dict[raw] for raw in filter_path_list})
        return (f"{lead} {_path_text(filter_path_list)}, which the git settings of "
                f"{project_dir} run a program on (the filter "
                f"{one_line(list_text(name_list))}), by the .gitattributes files "
                f"there now or by the ones the change itself writes")
    return ""


def _flag_path_set(index_list: list[tuple[bytes, bytes, bytes]]) -> set[bytes]:
    """Return the paths the index names with assume-unchanged or skip-worktree.

    Args:
        index_list: gitcmd.index_entry_list()'s return value.

    Returns:
        The paths, as git wrote them.
    """
    return {raw for tag, _, raw in index_list
            if gitcmd.is_assume_unchanged(tag) or gitcmd.is_skip_worktree(tag)}


class LandUndoError(RuntimeError):
    """A working tree that could not be put back as it was before a diff was put in."""


# How git reports a path of a working tree (_view_dict): "git status" names it, neither
# the disk nor the index holds a file there, the index holds it with no flag and "git
# status" does not name it, and anything else
_VIEW_SHOW = "show"
_VIEW_NONE = "none"
_VIEW_SAME = "same"
_VIEW_HIDE = "hide"


def _view_dict(project_dir: str, raw_path_list: Sequence[bytes]
               ) -> dict[bytes, str] | None:
    """Say how git reports each of the paths in a working tree as it is.

    Args:
        project_dir: The target repository, as an absolute path.
        raw_path_list: Paths relative to the repository, as git wrote them.

    Returns:
        {path: _VIEW_SHOW, _VIEW_NONE, _VIEW_SAME or _VIEW_HIDE}, read from
        gitcmd.status_path_set() and gitcmd.index_entry_list(). A path that is a
        directory on disk holds no file there. None when git gives no answer.
    """
    status_set = gitcmd.status_path_set(project_dir)
    index_list = gitcmd.index_entry_list(project_dir)
    if status_set is None or index_list is None:
        return None
    index_set = {raw for _, _, raw in index_list}
    flag_set = _flag_path_set(index_list)
    root = os.fsencode(project_dir)
    view_dict = {}
    for raw in raw_path_list:
        if raw in status_set:
            view_dict[raw] = _VIEW_SHOW
        elif raw in index_set:
            view_dict[raw] = _VIEW_HIDE if raw in flag_set else _VIEW_SAME
        elif not _has_file_at(os.path.join(root, raw)):
            view_dict[raw] = _VIEW_NONE
        else:
            view_dict[raw] = _VIEW_HIDE
    return view_dict


def _has_file_at(path: bytes) -> bool:
    """Say whether the disk holds something other than a directory at a path.

    Args:
        path: The path, as an absolute path. A symbolic link at it is not followed.

    Returns:
        False for a path nothing is at and for a directory.
    """
    try:
        return not stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


# What the disk holds at a path (_disk_state): nothing, a directory, a regular file, a
# symbolic link, and anything else or a file that cannot be read
_DISK_NONE = "none"
_DISK_DIR = "dir"
_DISK_FILE = "file"
_DISK_LINK = "link"
_DISK_OTHER = "other"


def _disk_state(path: bytes) -> tuple:
    """Return what the disk holds at one path, no symbolic link at it followed.

    Args:
        path: The path, as an absolute path.

    Returns:
        (_DISK_NONE,), (_DISK_DIR, its mode), (_DISK_FILE, its mode, the sha256 of its
        contents), (_DISK_LINK, what it points at) or (_DISK_OTHER, its mode), a mode
        as os.lstat() gives it.
    """
    try:
        path_stat = os.lstat(path)
    except OSError:
        return (_DISK_NONE,)
    if stat.S_ISDIR(path_stat.st_mode):
        return (_DISK_DIR, path_stat.st_mode)
    try:
        if stat.S_ISLNK(path_stat.st_mode):
            return (_DISK_LINK, os.readlink(path))
        if stat.S_ISREG(path_stat.st_mode):
            with open(path, "rb") as handle:
                return (_DISK_FILE, path_stat.st_mode,
                        hashlib.file_digest(handle, "sha256").hexdigest())
    except OSError:
        pass
    return (_DISK_OTHER, path_stat.st_mode)


def _disk_state_dict(project_dir: str, raw_path_list: Iterable[bytes]
                     ) -> dict[bytes, tuple]:
    """Return what the disk holds at each path and at every directory above it.

    Args:
        project_dir: The target repository, as an absolute path.
        raw_path_list: Paths relative to the repository, as git wrote them.

    Returns:
        {path relative to the repository: _disk_state()'s return value}, a directory
        ahead of what lies under it.
    """
    root = os.fsencode(project_dir)
    state_dict = {}
    for raw in raw_path_list:
        part_list = raw.split(b"/")
        for part_count in range(1, len(part_list) + 1):
            above = b"/".join(part_list[:part_count])
            if above not in state_dict:
                state_dict[above] = _disk_state(os.path.join(root, above))
    return state_dict


# The signals stop_signal_hold() holds, by their names in signal
_HOLD_SIGNAL_NAME_TUPLE = ("SIGINT", "SIGTERM", "SIGHUP")


@contextlib.contextmanager
def stop_signal_hold() -> Iterator[list[int]]:
    """Hold the signals that stop the command for the length of the block.

    A signal of _HOLD_SIGNAL_NAME_TUPLE that arrives meanwhile runs no handler. Once
    the block ends, each signal has the handler it had and the ones that arrived are
    sent to this process again. A signal the process ignores stays ignored, and
    nothing is held outside the main thread.

    Yields:
        The numbers of the signals that have arrived so far, oldest first.
    """
    stop_list: list[int] = []
    handler_dict = {}
    try:
        if threading.current_thread() is threading.main_thread():
            for name in _HOLD_SIGNAL_NAME_TUPLE:
                number = getattr(signal, name, None)
                handler = signal.getsignal(number) if number is not None else None
                if handler is None or handler == signal.SIG_IGN:
                    continue
                handler_dict[number] = handler
                signal.signal(number, lambda stop_number, frame:
                              stop_list.append(stop_number))
        yield stop_list
    finally:
        for number, handler in handler_dict.items():
            signal.signal(number, handler)
        for number in dict.fromkeys(stop_list):
            signal.raise_signal(number)


# What the directory is named that holds what the paths of one diff held before the
# diff was put in ("<prefix><process ID>-<random>"), the directory in it the files are
# copied under, and the file in it written once every file is copied
_BEFORE_PREFIX = "land-before-"
_BEFORE_TREE_NAME = "tree"
_BEFORE_LIST_NAME = "before.json"
_BEFORE_NAME = re.compile(re.escape(_BEFORE_PREFIX) + r"(\d+)-")

# How many bytes of files are copied before one diff is put in
_MAX_BEFORE_BYTE = 4 << 30


def _copy_before(project_dir: str, state_dict: dict[bytes, tuple], keep_root: str) -> str:
    """Copy what the disk holds at the paths a diff touches to a directory of its own.

    Each regular file is copied under _BEFORE_TREE_NAME at its path, with its mode and
    the owner's read and write bits, and each directory is made there. A symbolic link
    and a path nothing is at are written to _BEFORE_LIST_NAME, with the repository,
    once every file is copied.

    Args:
        project_dir: The target repository, as an absolute path.
        state_dict: What the disk holds, as _disk_state_dict() returns it for the
            paths the diff touches.
        keep_root: The directory the copy is made under, as an absolute path. It is
            made when it is not there.

    Returns:
        The directory made, as an absolute path.

    Raises:
        OSError: When a path is neither a regular file, a symbolic link, a directory
            nor nothing, when a path leads out of the repository, when the files hold
            more than _MAX_BEFORE_BYTE bytes, and when a file cannot be copied. The
            directory is removed.
    """
    os.makedirs(keep_root, exist_ok=True)
    keep_dir = tempfile.mkdtemp(prefix=f"{_BEFORE_PREFIX}{os.getpid()}-", dir=keep_root)
    root = os.fsencode(project_dir)
    tree_dir = os.fsencode(os.path.join(keep_dir, _BEFORE_TREE_NAME))
    byte_count = 0
    link_dict, none_list = {}, []
    try:
        for raw, state in state_dict.items():
            name = gitcmd.shown_text(raw)
            if raw.startswith(b"/") or b".." in raw.split(b"/"):
                raise OSError(f"{name} leads out of {project_dir}")
            target = os.path.join(tree_dir, raw)
            if state[0] == _DISK_NONE:
                none_list.append(name)
            elif state[0] == _DISK_LINK:
                link_dict[name] = gitcmd.shown_text(state[1])
            elif state[0] == _DISK_DIR:
                os.makedirs(target, exist_ok=True)
            elif state[0] == _DISK_FILE:
                byte_count += os.lstat(os.path.join(root, raw)).st_size
                if byte_count > _MAX_BEFORE_BYTE:
                    raise OSError(f"the files hold more than {_MAX_BEFORE_BYTE} bytes")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copyfile(os.path.join(root, raw), target, follow_symlinks=False)
                os.chmod(target, stat.S_IMODE(state[1]) | stat.S_IRUSR | stat.S_IWUSR)
            else:
                raise OSError(f"{name} is not a file that can be read")
        with tmpfile.atomic_write(os.path.join(keep_dir, _BEFORE_LIST_NAME)) as tmp_path:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump({"project": project_dir, "link": link_dict,
                           "none": none_list}, handle)
    except BaseException:
        shutil.rmtree(keep_dir, ignore_errors=True)
        raise
    return keep_dir


def _warn_old_copy(project_dir: str, keep_root: str) -> None:
    """Name the copies _copy_before() made for a repository that are still there.

    A copy stays when the process that made it was killed before it removed it, and when
    the working tree could not be put back from it (_put_back). One with its
    _BEFORE_LIST_NAME naming project_dir is named in a warning and stays where it is.
    One without the file, whose process has ended, is removed. The caller holds the lock
    the runs writing to the repository share.

    Args:
        project_dir: The target repository, as an absolute path.
        keep_root: The directory the copies are made under, as an absolute path.
    """
    try:
        name_list = sorted(os.listdir(keep_root))
    except OSError:
        return
    for name in name_list:
        match = _BEFORE_NAME.match(name)
        keep_dir = os.path.join(keep_root, name)
        if not match or not os.path.isdir(keep_dir):
            continue
        list_path = os.path.join(keep_dir, _BEFORE_LIST_NAME)
        if not os.path.lexists(list_path):
            if tmpfile.process_dead(int(match.group(1))):
                _remove_tree(keep_dir, "what a change's files held before it")
            continue
        try:
            list_dict = tmpfile.read_json(list_path)
        except (OSError, ValueError):
            continue
        if isinstance(list_dict, dict) and list_dict.get("project") == project_dir:
            logger.warning(
                "a command ended before it was through putting a change into the working "
                "tree of %s, which may hold part of that change: what the files it "
                "touches held before is kept in %s, and %s names the symbolic links and "
                "the paths nothing was at. Put back what the working tree is missing "
                "from there by hand, then delete %s", project_dir,
                os.path.join(keep_dir, _BEFORE_TREE_NAME), list_path, keep_dir)


def _is_under_link(root: bytes, raw: bytes) -> bool:
    """Say whether a directory above a path is a symbolic link on disk.

    Args:
        root: The top of the working tree, as an absolute path.
        raw: The path, relative to root.

    Returns:
        True when one of the directories between root and the path's last part is a
        symbolic link.
    """
    part_list = raw.split(b"/")
    return any(os.path.islink(os.path.join(root, *part_list[:part_count]))
               for part_count in range(1, len(part_list)))


def _put_back(project_dir: str, state_dict: dict[bytes, tuple], keep_dir: str,
              patch_path: str, why: str) -> None:
    """Put the paths a diff touches back as they were before it was applied.

    The disk is then confirmed to hold what it held before. What is at a path now and
    was not is removed, a directory only when it is empty. A directory is made again, a
    regular file is copied back from keep_dir, a symbolic link is made again, and a
    directory and a regular file are given the mode they had. A path below a symbolic
    link is left as it is.

    Args:
        project_dir: The target repository, as an absolute path.
        state_dict: What the disk held before the diff was applied, as
            _disk_state_dict() returns it for the paths the diff touches.
        keep_dir: The copy of those paths, as _copy_before() returns it.
        patch_path: The diff, as an absolute path.
        why: What the paths are put back for, as a clause.

    Raises:
        LandUndoError: When a path of state_dict is not as it was. The message names
            the paths, keep_dir and patch_path.
    """
    root = os.fsencode(project_dir)
    tree_dir = os.fsencode(os.path.join(keep_dir, _BEFORE_TREE_NAME))
    error_list = []
    # == Step 1: what was not there, deepest first ============
    for raw in reversed(state_dict):
        path = os.path.join(root, raw)
        now = _disk_state(path)
        if now in (state_dict[raw], (_DISK_NONE,)) or _is_under_link(root, raw):
            continue
        try:
            if now[0] != _DISK_DIR:
                os.unlink(path)
            elif state_dict[raw][0] != _DISK_DIR:
                os.rmdir(path)
        except OSError as e:
            error_list.append(str(e))
    # == Step 2: what was there, directories first ============
    for raw, state in state_dict.items():
        path = os.path.join(root, raw)
        now = _disk_state(path)
        if now == state or state[0] in (_DISK_NONE, _DISK_OTHER) \
                or _is_under_link(root, raw):
            continue
        try:
            if state[0] == _DISK_DIR and now[0] == _DISK_NONE:
                os.mkdir(path)
            elif state[0] == _DISK_LINK and now[0] == _DISK_NONE:
                os.symlink(state[1], path)
            elif state[0] == _DISK_FILE and now[0] in (_DISK_NONE, _DISK_FILE):
                shutil.copyfile(os.path.join(tree_dir, raw), path, follow_symlinks=False)
                os.chmod(path, stat.S_IMODE(state[1]))
        except OSError as e:
            error_list.append(str(e))
    # == Step 3: the mode of each directory, deepest first ====
    for raw in reversed(state_dict):
        state = state_dict[raw]
        path = os.path.join(root, raw)
        if state[0] != _DISK_DIR or _is_under_link(root, raw):
            continue
        try:
            if os.lstat(path).st_mode != state[1]:
                os.chmod(path, stat.S_IMODE(state[1]))
        except OSError as e:
            error_list.append(str(e))
    # == Step 4: the disk against what it held ================
    now_dict = _disk_state_dict(project_dir, state_dict)
    remain_list = [raw for raw, state in state_dict.items() if now_dict[raw] != state]
    if remain_list:
        error_text = one_line("; ".join(error_list[:config.MAX_NAME_ITEM])) \
            or "no error was raised"
        raise LandUndoError(
            f"{why}, and the working tree could not be put back as it was: "
            f"{_path_text(remain_list)} in {project_dir} is not as it was before the "
            f"change was put in ({error_text}). What the paths the change touches held "
            f"before is kept in {keep_dir}, and the change is written to {patch_path}; "
            f"put the working tree right from the two by hand")


def _write_in_tree(project_dir: str, commit: str, patch_path: str,
                   path_dict: dict[bytes, bool]) -> str:
    """Apply a diff to the files on disk, and say what git did not write of it, or "".

    The diff is applied as it is written (gitcmd.APPLY_TUPLE). Once git ends without
    an error, every path the diff touches is looked at: one commit holds a file or a
    symbolic link at has something other than a directory there, and one commit
    holds nothing at has no file there or lies below a symbolic link
    (_is_under_link). A repository of its own commit holds is not looked at.

    Args:
        project_dir: The target repository, as an absolute path.
        commit: The commit the diff leads to.
        patch_path: The diff, as an absolute path.
        path_dict: The paths the diff touches, as _diff_path_dict() returns them.

    Returns:
        A clause saying that git ended with an error, that the files of commit could
        not be listed, or which paths are not as commit has them. "" when every path
        is.
    """
    proc = _apply_proc(project_dir, patch_path)
    git_text = (proc.stderr or proc.stdout).strip() or f"exit status {proc.returncode}"
    if proc.returncode != 0:
        return (f"git did not get through putting the change into the working tree of "
                f"{project_dir} ({git_text})")
    entry_dict = _tree_entry_dict(project_dir, commit)
    if entry_dict is None:
        return (f"with the change in the working tree of {project_dir}, git did not "
                f"list the files {gitcmd.short(commit)} holds")
    root = os.fsencode(project_dir)
    miss_list = []
    for raw in path_dict:
        has_file = _has_file_at(os.path.join(root, raw))
        if raw in entry_dict:
            is_miss = not has_file and entry_dict[raw][0] != gitcmd.GITLINK_MODE
        else:
            is_miss = has_file and not _is_under_link(root, raw)
        if is_miss:
            miss_list.append(raw)
    if miss_list:
        return (f"git ended without an error and did not write or did not delete "
                f"{_path_text(miss_list)} in the working tree of {project_dir} "
                f"({git_text})")
    return ""


def _no_show_text(project_dir: str, path_dict: dict[bytes, bool],
                  earlier_dict: dict[bytes, bool], before_dict: dict[bytes, str],
                  view_dict: dict[bytes, str] | None) -> str:
    """Say which paths "git status" leaves out with a diff in the working tree, or "".

    A path the diff touches is one when git answers _VIEW_HIDE for it, unless the
    working tree held it before this command and git answered _VIEW_SHOW for it
    before the diff was applied. A path an earlier diff added that this one does not
    touch is one when git answers _VIEW_HIDE for it and did not before.

    Args:
        project_dir: The target repository, as an absolute path.
        path_dict: The paths the diff touches, as _diff_path_dict() returns them.
        earlier_dict: The paths put into the working tree before, in the same form.
        before_dict: _view_dict()'s return value before the diff was applied, or {}
            when git gave no answer then.
        view_dict: _view_dict()'s return value for the same paths with the diff
            applied.

    Returns:
        A clause naming those paths, or saying that git gave no answer. "" when
        there is none.
    """
    if view_dict is None:
        return (f"with the change in the working tree of {project_dir}, git did not "
                f"say what git status shows of {_path_text(list(path_dict))}")
    add_set = {raw for one_dict in (path_dict, earlier_dict)
               for raw, is_add in one_dict.items() if is_add}
    hide_list = [raw for raw in path_dict if view_dict[raw] == _VIEW_HIDE
                 and (raw in add_set or before_dict.get(raw) != _VIEW_SHOW)]
    if hide_list:
        return (f"the change writes {_path_text(hide_list)}, which git status does not "
                f"show in {project_dir} once it is there")
    hide_list = [raw for raw, view in view_dict.items()
                 if raw not in path_dict and view == _VIEW_HIDE
                 and before_dict.get(raw, _VIEW_HIDE) != _VIEW_HIDE]
    if hide_list:
        return (f"with the change in the working tree, git status no longer shows "
                f"{_path_text(hide_list)}, which was put into the working tree of "
                f"{project_dir} before it")
    return ""


def by_hand_guide(project_dir: str, patch_path: str) -> tuple[str, str]:
    """Return what applying a patch by hand says for its submodules, and the command.

    The clause is for the submodules the patch moves, and the command applies the patch.
    git apply leaves a submodule at the commit it is at.

    Args:
        project_dir: The target repository, as an absolute path.
        patch_path: The patch, as an absolute path.

    Returns:
        (a clause naming the repositories of their own whose commit the patch moves
        (gitcmd.submodule_move_list) and saying the command checks out in each the
        commit the patch moves it to once it is at the one the patch moves it from,
        or "" for none; the command, its parts followed through "&&": in each of
        those directories, with the .git there as the repository, git with
        gitcmd.IS_ANCESTOR_TUPLE over HEAD and the commit the patch moves it from both
        ways, which ends with a non-zero exit status unless HEAD is that commit, and
        when one of them fails, a printf of one line to standard error naming those
        directories and commits, and false; git apply of the patch as
        gitcmd.APPLY_TUPLE applies it, in project_dir; and git checkout
        --detach of the commit the patch moves it to, in each of those directories).
        Only a path the index of project_dir holds with gitcmd.GITLINK_MODE, whose
        real path lies inside project_dir, is checked out.
    """
    root = os.path.realpath(project_dir)
    link_set = {path for _, mode, path in gitcmd.index_entry_list(project_dir) or []
                if mode == gitcmd.GITLINK_MODE}
    move_list = []
    for raw, from_commit, move_commit in gitcmd.submodule_move_list(patch_path, link_set):
        real = os.path.realpath(os.path.join(project_dir, os.fsdecode(raw)))
        if real != root and os.path.commonpath([root, real]) == root:
            move_list.append((raw, from_commit, move_commit))

    def sub_git(raw: bytes, *arg_tuple: str) -> str:
        """Return a git command run in the submodule at raw with its .git."""
        return command_line("git", "-C", os.path.join(project_dir, os.fsdecode(raw)),
                            "--git-dir=.git", *arg_tuple)

    apply_command = " && ".join([
        command_line("git", "-C", project_dir, *gitcmd.APPLY_TUPLE, patch_path),
        *(sub_git(raw, "checkout", "--detach", move_commit)
          for raw, _, move_commit in move_list)])
    if not move_list:
        return "", apply_command
    # The checks of each submodule's HEAD, which print one line and stop before git
    # apply when one fails
    check = " && ".join(
        sub_git(raw, *gitcmd.IS_ANCESTOR_TUPLE, *pair) for raw, from_commit, _ in move_list
        for pair in (("HEAD", from_commit), (from_commit, "HEAD")))
    stop_text = "nothing was applied: " + " or ".join(
        f"{_path_text([raw])} is not at {from_commit}" for raw, from_commit, _ in move_list
    ) + ", the commit the change moves it from"
    stop = command_line("printf", r"%s\n", stop_text)
    return (f" (git apply moves no submodule, so the command first stops unless "
            f"{_path_text([raw for raw, _, _ in move_list])} is at the commit the change "
            f"moves it from, and then checks out there the commit the change moves it "
            f"to, which has to be there)",
            f"{check} || {{ {stop} >&2; false; }} && {apply_command}")


def land_in_tree(project_dir: str, base: str, commit: str, patch_path: str,
                 last_path: str, earlier_base: str = "") -> dict:
    """Apply what a run landed to the working tree, and say what was applied.

    The diff from base to commit is written to patch_path (gitcmd.write_patch) and
    applied to the files on disk as it is written (gitcmd.APPLY_TUPLE), under the lock
    the runs writing to one repository share. A diff _refusal_text() names a reason for
    is not applied: one that writes a file a command or git would read as its settings,
    one that leaves a path out of what "git status" shows, and one git would run a
    filter driver of its settings on. The paths this process put into project_dir
    before, and the ones the change from earlier_base to base touches, are checked with
    it. git checks that it applies before it writes anything, and a tree that no longer
    holds what the change was written against is left as it is.

    The diff ends either whole in the working tree or with the working tree as it
    was. What the paths it touches hold is copied beside last_path before it is
    applied (_copy_before) and removed once this call is through with it. The paths
    are put back from the copy and confirmed against what the disk held
    (_put_back), and the diff counts as one that is not applied, when
    - git ends with an error, or leaves a path other than commit has it
      (_write_in_tree),
    - "git status" leaves out a path the diff touches or an earlier diff added
      (_view_dict, read before the diff is applied and once it is, and _no_show_text),
    - a signal that stops the command arrives while the diff is put in
      (stop_signal_hold, which holds it until the paths are back), or
    - anything is raised meanwhile.
    A path that stays applied and "git status" does not name - one the index holds
    as the disk now does, and one the working tree held before this command that a
    rule now covers - is named in the log. A copy a killed process left behind is
    named in the log too (_warn_old_copy).

    The index is not touched: what was applied shows as changes git reports, for the
    user to commit. Once a diff is applied, last_path is written over with the diffs
    this process has applied to project_dir, taken together and read against the tree
    the working tree made just before (_keep_last_land); applying that file in reverse
    takes them back out. That tree is read when this process applied a diff to
    project_dir before, and when git does not apply this one with every hunk at the
    lines it names (_is_at_line). What was applied is added to what land_list() returns,
    and the unions the command carried on from for their cases that the change stands
    for (_hold_stand_for), that the working tree held just before the diff went in or
    holds now and whose branch does not reach commit are deleted (_drop_stand_for),
    while the signals are still held. The commit start_tree() returns is made before the
    first diff is applied. The git commands that read the diff are held to
    gitcmd.MAX_PATCH_ADDRESS_BYTE.

    Args:
        project_dir: The target repository, as an absolute path.
        base: The commit the working tree held when the run started.
        commit: The commit holding what the run landed.
        patch_path: Where the diff is written, as an absolute path.
        last_path: Where the copy of an applied diff is kept (last_land_path()).
        earlier_base: The commit a run carried on from started at, when the working
            tree holds that run's change up to base. Empty for none.

    Returns:
        {"files": how many files the diff touches, "paths": those files, relative to
        the repository, a file that was moved counted at both of its paths, "patch":
        patch_path, "last": last_path, or "" when no copy was written, "no_status":
        the files among "paths" that are there and "git status" does not name}, with
        "stand_drop": the unions _drop_stand_for() deleted, where it deleted any.
        "files" is 0 and nothing is written when the two commits hold the same tree;
        the unions the change stands for that the working tree holds are deleted then
        too, with the signals held as well.

    The command a LandRefusal and a CommandError end with applies the diff by hand,
    and then checks out the commit the diff moves a submodule to, once the submodule
    is at the commit the diff moves it from (by_hand_guide).

    Raises:
        LandRefusal: When the diff touches a path _refusal_text() names, or was
            taken back out for a path git reports in no way.
        LandUndoError: When the working tree could not be put back as it was.
        CommandError: When the diff does not apply, when what its paths hold could
            not be copied, and when it was taken back out for what git did not write
            or for a signal. The diff stays at patch_path, and the command applies it
            by hand, as the one of a LandRefusal does.
        RuntimeError: When the diff cannot be read.
    """
    path_dict = _diff_path_dict(project_dir, base, commit)
    path_list = [raw.decode("utf-8", "replace") for raw in path_dict]
    file_count = len(path_list)
    if not file_count:
        land_dict = {"files": 0, "paths": [], "patch": patch_path, "last": "",
                     "no_status": []}
        # The working tree holds what would go in: the unions this command carried on
        # from that it stands for go as they do once a diff is applied
        with _branch_lock(project_dir), stop_signal_hold():
            stand_drop_list = _drop_stand_for(project_dir, [], commit)
        if stand_drop_list:
            land_dict["stand_drop"] = stand_drop_list
        return land_dict
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    proc = gitcmd.write_patch(project_dir, base, commit, patch_path)
    if proc.returncode != 0:
        raise RuntimeError(f"could not write what {gitcmd.short(commit)} changes against "
                           f"{gitcmd.short(base)} to {patch_path}: {proc.stderr.strip()}")
    own_dict = _LAND_PATH_DICT.setdefault(project_dir, {})
    earlier_dict = dict(own_dict)
    if earlier_base:
        for raw, is_add in _diff_path_dict(project_dir, earlier_base, base).items():
            earlier_dict[raw] = is_add or earlier_dict.get(raw, False)
    refusal_text = _refusal_text(project_dir, base, commit, path_dict, earlier_dict)
    move_text, by_hand_command = by_hand_guide(project_dir, patch_path)
    by_hand_text = (f"The change is written to {patch_path}; read it, and apply it by "
                    f"hand{move_text} with:")
    if refusal_text:
        raise LandRefusal(f"{refusal_text}, so nothing is put into the working tree. "
                          f"{by_hand_text}", by_hand_command)
    # The commit start_tree() returns is made before anything is written here
    start_tree(project_dir)
    view_path_list = [*path_dict, *(raw for raw, is_add in earlier_dict.items()
                                    if is_add and raw not in path_dict)]
    keep_root = os.path.dirname(last_path)
    with _branch_lock(project_dir):
        _warn_old_copy(project_dir, keep_root)
        before_dict = _view_dict(project_dir, view_path_list) or {}
        state_dict = _disk_state_dict(project_dir, path_dict)
        proc = _apply_proc(project_dir, "--check", patch_path)
        if proc.returncode != 0:
            raise CommandError(
                f"the change does not apply to the working tree of {project_dir} "
                f"({(proc.stderr or proc.stdout).strip()}). It is written to "
                f"{patch_path}; apply it by hand{move_text} with:", by_hand_command)
        # What the diffs this process put in before are taken off for the kept copy,
        # and what the diff is read against where its hunks go in at other lines
        before_tree = gitcmd.work_tree_id(project_dir) \
            if _LAND_PATCH_DICT.get(project_dir) \
            or not _is_at_line(project_dir, patch_path) else ""
        # The unions this command carried on from that the working tree holds before
        # the diff goes in
        stand_before_list = [branch for branch in stand_for_list(project_dir)
                             if is_gap_union(project_dir, branch)]
        with stop_signal_hold() as stop_list:
            try:
                keep_dir = _copy_before(project_dir, state_dict, keep_root)
            except OSError as e:
                raise CommandError(
                    f"what the change touches in the working tree of {project_dir} "
                    f"could not be copied before the change is put in ({e}), so "
                    f"nothing is put into the working tree. {by_hand_text}",
                    by_hand_command) from None
            is_no_show = False
            try:
                back_text = _write_in_tree(project_dir, commit, patch_path, path_dict)
                if not back_text:
                    view_dict = _view_dict(project_dir, view_path_list)
                    back_text = _no_show_text(project_dir, path_dict, earlier_dict,
                                              before_dict, view_dict)
                    is_no_show = bool(back_text)
                if stop_list and not back_text:
                    back_text = (f"the command was stopped while the change was put "
                                 f"into the working tree of {project_dir}")
            except BaseException as e:
                _put_back(project_dir, state_dict, keep_dir, patch_path,
                          f"{type(e).__name__} was raised while the change was put "
                          f"into the working tree of {project_dir}")
                _remove_tree(keep_dir, "what a change's files held before it")
                raise
            if back_text:
                _put_back(project_dir, state_dict, keep_dir, patch_path, back_text)
            _remove_tree(keep_dir, "what a change's files held before it")
            if back_text:
                raise (LandRefusal if is_no_show else CommandError)(
                    f"{back_text}, so it was taken back out and nothing is put into "
                    f"the working tree. {by_hand_text}", by_hand_command)
            keep_path = _keep_last_land(project_dir, before_tree, (base, commit),
                                        patch_path, last_path)
            no_status_list = [raw.decode("utf-8", "replace") for raw in path_dict
                              if view_dict[raw] in (_VIEW_SAME, _VIEW_HIDE)]
            land_dict = {"files": file_count, "paths": path_list, "patch": patch_path,
                         "last": keep_path, "no_status": no_status_list}
            _LAND_LIST.append(land_dict)
            for raw, is_add in path_dict.items():
                own_dict[raw] = is_add or own_dict.get(raw, False)
            # The unions this command carried on from that what went in stands for
            stand_drop_list = _drop_stand_for(project_dir, stand_before_list, commit)
            if stand_drop_list:
                land_dict["stand_drop"] = stand_drop_list
    logger.info("%d file(s) of what the run changed were put into the working tree of "
                "%s; commit them when they are what was asked for", file_count,
                project_dir)
    if no_status_list:
        logger.warning("git status does not name %s, which the run wrote in the working "
                       "tree of %s: as git reads the working tree, the index holds the "
                       "same there, or a rule now covers a file git named before",
                       one_line(list_text(no_status_list)), project_dir)
    return land_dict


def land_status(result_dict: dict | None) -> int:
    """Return the exit status a run adds for a change that did not reach the working tree.

    It is apart from the run's own exit status.

    Args:
        result_dict: run()'s or integrate()'s return value, or None.

    Returns:
        1 when the change was to go into the working tree and did not, else 0.
    """
    return 1 if result_dict and result_dict.get("land_why") else 0


def land_why_text(result_dict: dict | None) -> str:
    """Return why a change did not reach the working tree, with its command.

    Args:
        result_dict: run()'s or integrate()'s return value, or None.

    Returns:
        "land_why" followed by "land_command" (with_command), or "" when the change
        went in or there is no result.
    """
    result_dict = result_dict or {}
    return with_command(str(result_dict.get("land_why") or ""),
                        str(result_dict.get("land_command") or ""))


# What keeps any merge over the same HEAD from being settled, and what to change for one
# to be, by the entries of integrate.head_block_list()
_HEAD_BLOCK_TEXT_DICT = {
    integrate_mod.HEAD_STOP_BLOCK: (
        "the tests run over HEAD stop before every test has run",
        "let the test command run every test when one fails, or make the tests that "
        "fail over HEAD pass"),
    integrate_mod.HEAD_SHORT_BLOCK: (
        "the tests run over HEAD do not print how many tests they pass and fail (or a "
        "test ends their process before they print it), or do not name each test they "
        "pass and fail",
        "have the test command print how many tests it passes and fails (for pytest, "
        "one -q at most, no plugin or conftest.py hook that takes that line away, and "
        "no test that ends the test process) and the name of each test it passes and "
        "fails"),
}


def _head_block_text(block_list: list[str] | None, branch: str) -> str:
    """Return what a merge's reason adds for what keeps merges over its HEAD unsettled.

    Nothing keeps them unsettled when no entry of block_list is known.

    Examples:
        ["stop"], "codetwine-loom/sandbox/x"
            -> ". No merge over this HEAD is settled while the tests run over HEAD
               stop before every test has run: let the test command run every test
               when one fails, or make the tests that fail over HEAD pass, then carry
               on from codetwine-loom/sandbox/x"

    Args:
        block_list: The merge's "head_block" (integrate.head_block_list()), or None.
        branch: The branch the merge left.

    Returns:
        The text, starting with ". ", or "" when nothing keeps them unsettled.
    """
    pair_list = [_HEAD_BLOCK_TEXT_DICT[block] for block in block_list or []
                 if block in _HEAD_BLOCK_TEXT_DICT]
    if not pair_list:
        return ""
    return (f". No merge over this HEAD is settled while "
            f"{' and '.join(why for why, _ in pair_list)}: "
            f"{'; '.join(change for _, change in pair_list)}, then carry on from {branch}")


def held_base(project_dir: str, start: str, settled: str, patch_path: str) -> str:
    """Return what the working tree holds of what an earlier run put into it.

    The earlier run put in the change from start to settled. How far the working tree
    still holds it is told by _held_commit(), which reads settled and the commits
    under it down to the merge. Whether it holds it up to a commit under settled is
    told as well by whether the change from that commit to settled applies forward the
    way git applies a patch, by its context; whether it holds none of it, by whether
    the change from start does.

    Args:
        project_dir: The target repository, as an absolute path.
        start: The commit the earlier run started from.
        settled: The commit whose change the earlier run put in.
        patch_path: Where the change read is written, as an absolute path. Its
            directory is made when it is not there.

    Returns:
        The commit _held_commit() finds when the working tree holds the change up to
        settled, or up to a commit under it and the change from there to settled
        applies to it, start when it holds none of it, and "" when it holds part of it
        or git cannot tell.
    """
    os.makedirs(os.path.dirname(patch_path), exist_ok=True)
    with _branch_lock(project_dir):
        try:
            held = _held_commit(project_dir, start, settled)
        except RuntimeError as e:
            logger.warning("%s", e)
            return ""
        if held == settled:
            return held
        if held:
            if gitcmd.write_patch(project_dir, held, settled, patch_path).returncode == 0 \
                    and _apply_proc(project_dir, "--check", patch_path).returncode == 0:
                return held
            logger.info("the working tree holds what %s put in before up to %s, and what "
                        "is above it does not apply to the working tree",
                        gitcmd.short(settled), gitcmd.short(held))
            return ""
        if gitcmd.write_patch(project_dir, start, settled, patch_path).returncode == 0 \
                and _apply_proc(project_dir, "--check", patch_path).returncode == 0:
            logger.warning("the working tree does not hold what %s put in before, so it "
                           "is put in again along with what is new",
                           gitcmd.short(settled))
            return start
    return ""


def _held_commit(project_dir: str, start: str, top: str) -> str:
    """Return the last commit of an earlier merge's branch whose change the tree holds.

    The tree is the working tree. The commits are read from top down its first parents
    to the merge under every commit of a stage (integrate.below_stage), the change of
    each being the one from start (gitcmd.has_change_in_tree). A commit whose change git
    cannot read is passed over.

    Args:
        project_dir: The target repository, as an absolute path.
        start: The commit the merge was written against.
        top: The commit the reading starts at.

    Returns:
        The full hash of the first such commit from top, or "" when the working tree
        holds the change of none of them.

    Raises:
        RuntimeError: When the commits cannot be read.
    """
    commit = top
    merge = integrate_mod.below_stage(project_dir, commit)
    while not gitcmd.has_change_in_tree(project_dir, start, commit):
        if commit == merge:
            return ""
        commit = integrate_mod.first_parent(project_dir, commit)
    return commit


def try_land(project_dir: str, base: str, commit: str, patch_path: str,
             last_path: str, earlier_base: str = "") -> dict:
    """Put what a run landed into the working tree, and say what happened either way.

    Args:
        As land_in_tree() takes them.

    Returns:
        {"land": land_in_tree()'s return value, or None when it did not apply,
        "land_why": why it did not, or "", "land_command": the command that message ends
        with (error_part), or "", "land_refuse": True when it did not for the paths the
        diff touches (LandRefusal)}.

    Raises:
        LandUndoError: As land_in_tree() raises it.
    """
    try:
        return {"land": land_in_tree(project_dir, base, commit, patch_path, last_path,
                                     earlier_base),
                "land_why": "", "land_command": "", "land_refuse": False}
    except LandUndoError:
        raise
    except RuntimeError as e:
        logger.warning("%s", e)
        why, command = error_part(e)
        return {"land": None, "land_why": why, "land_command": command,
                "land_refuse": isinstance(e, LandRefusal)}


def drop_branch_under(project_dir: str, branch_list: list[str],
                      commit: str) -> list[str]:
    """Delete the branches a commit already reaches, and say which were deleted.

    A branch whose tip a commit reaches holds nothing that commit does not. A branch the
    commit does not reach is left as it is.

    Args:
        project_dir: The repository, as an absolute path.
        branch_list: The branches to consider.
        commit: The commit that has to reach a branch for it to go.

    Returns:
        The branches that were deleted, in the order they were given.
    """
    drop_list = []
    for branch in branch_list:
        tip = branch_tip(project_dir, branch)
        if not tip or not gitcmd.is_ancestor(project_dir, tip, commit):
            continue
        if _delete_branch(project_dir, branch):
            drop_list.append(branch)
    if drop_list:
        logger.info("%d branch(es) this run made are in %s, so their names were taken "
                    "away: %s", len(drop_list), gitcmd.short(commit), ", ".join(drop_list))
    return drop_list


def drop_try_branch_list(project_dir: str, run_id: str, keep_list: Sequence[str]) -> list[str]:
    """Delete the branches of one run's tries that were handed out again.

    The branches are those under the task branches' root that carry run_id, the branch
    of its tree and those of keep_list left out.

    Args:
        project_dir: The repository, as an absolute path.
        run_id: The run, as new_run_id() names it.
        keep_list: The branches of the tries the run is reported with, which are left
            alone whether or not they are still there.

    Returns:
        The branches that were deleted, in git's order.
    """
    drop_list = []
    for branch in _own_branch_list(project_dir, _TASK_BRANCH_ROOT):
        if run_in(branch) != run_id or branch in keep_list or is_tree_branch(branch):
            continue
        if _delete_branch(project_dir, branch):
            drop_list.append(branch)
    if drop_list:
        logger.info("%d branch(es) of tries that were done again were taken away: %s",
                    len(drop_list), ", ".join(drop_list))
    return drop_list


def run_in(name: str) -> str:
    """Return the run a branch or an output directory belongs to, or "".

    Examples:
        run_in("codetwine-loom/run/20260101-000000-1-42/2-retry1")
            -> "20260101-000000-1-42"

    Args:
        name: A branch name or a directory name, as this tool made it.

    Returns:
        The run id, or "" for a name nothing here made.
    """
    match = _RUN_IN_NAME.search(name)
    return match.group(0) if match else ""


def is_own_branch(name: str) -> bool:
    """Say whether a branch's name is one this tool made up.

    Args:
        name: The branch's name.

    Returns:
        True for a name under one of this tool's own roots that carries a run (run_in).
    """
    return bool(run_in(name)) and name.startswith(f"{_BRANCH_ROOT}/")


def _own_branch_list(project_dir: str, root: str) -> list[str]:
    """Return the branches under one of this tool's own roots.

    Args:
        project_dir: The target repository, as an absolute path.
        root: The first part of the name, without a slash.

    Returns:
        The branch names, in git's order; empty as well when git does not list them.
    """
    try:
        return _read_branch_list(project_dir, root)
    except RuntimeError:
        return []


def _read_branch_list(project_dir: str, root: str) -> list[str]:
    """Return the branches under one of this tool's own roots, as git lists them.

    Args:
        project_dir: The target repository, as an absolute path.
        root: The first part of the name, without a slash.

    Returns:
        The branch names, in git's order.

    Raises:
        RuntimeError: When git does not list them.
    """
    proc = gitcmd.run(project_dir, "for-each-ref", "--format=%(refname:short)",
                      f"refs/heads/{root}/**")
    if proc.returncode != 0:
        raise RuntimeError(f"git did not list the branches under {root} in "
                           f"{project_dir}: {proc.stderr.strip()}")
    return proc.stdout.split()


def _run_remain_dict(project_dir: str, out_dir: str) -> dict[str, dict[str, list[str]]]:
    """Return what each run left, told apart by what it is.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.

    Returns:
        {run id: {"tasks": [branch, ...], "products": [branch, ...], "tree": [branch],
        "dirs": [directory, ...], "runs": [run id, ...], "answers": [file, ...],
        "patches": [file, ...]}}. "dirs" holds each agent's own copy of the output
        directory and what this machine keeps for that agent (host_dir), for the agents
        whose directory holds something for this repository (_is_agent_dir_of): an
        output directory several repositories share holds the agents of each. "runs",
        "answers" and "patches" are read off the records this machine alone wrote for
        the repository's directory name on its side of the agents (_run_mark_dict), and
        nothing the agents' own copies hold names or decides any of them: "runs" are the
        runs this machine started there, "answers" the answers it read back and
        "patches" the patches that landed in no commit, each named where that run of
        that agent wrote it (run_file_layout). "tree" is the branch a run that was
        killed left pointing at the commit it started from (tree_branch). A name
        carrying no run is left out: a branch somebody else made and an agent named
        after a branch are never in here.

    Raises:
        RuntimeError: When git does not list the branches (_read_branch_list).
    """
    name = knowledge.project_name_of(project_dir)
    remain_dict: dict[str, dict[str, list[str]]] = {}

    def list_of(run_id: str, kind: str) -> list[str]:
        """Return the list of one kind for a run, adding the run's entry when missing."""
        return remain_dict.setdefault(run_id, {"tasks": [], "products": [], "tree": [],
                                               "dirs": [], "runs": [], "answers": [],
                                               "patches": []})[kind]

    for branch in _read_branch_list(project_dir, _TASK_BRANCH_ROOT):
        if run_in(branch):
            list_of(run_in(branch), "tree" if is_tree_branch(branch) else "tasks"
                    ).append(branch)
    for branch in _read_branch_list(project_dir, _BRANCH_PREFIX.rstrip("/")):
        if run_in(branch):
            list_of(run_in(branch), "products").append(branch)
    for agent, path in output_dir_list(out_dir)[1:]:
        if run_in(agent) and _is_agent_dir_of(path, name):
            list_of(run_in(agent), "dirs").append(path)
    # What this machine keeps for an agent goes with the agent's own directory, and
    # says which runs it started there and what each of them left to pick up
    for agent, path in host_dir_list(out_dir):
        if not run_in(agent) or not _is_agent_dir_of(path, name):
            continue
        list_of(run_in(agent), "dirs").append(path)
        for one_run, keep in _run_mark_dict(_sandbox_dir(path, name)).items():
            list_of(run_in(agent), "runs").append(one_run)
            run_file = run_file_layout(out_dir, agent, name, one_run)
            if keep == _PATCH_KEEP:
                list_of(run_in(agent), "patches").append(run_file.patch)
            elif keep == _ANSWER_KEEP:
                list_of(run_in(agent), "answers").append(run_file.answer)
    return remain_dict


def _write_run_mark(run_file: RunFileLayout, keep: str) -> None:
    """Record on this machine's side that it started a run, and what the run left.

    What the run left is what is to be picked up by hand.

    A record that cannot be written is said in the log.

    Args:
        run_file: The run's paths; the record is run_file.mark.
        keep: _PATCH_KEEP for a patch that landed in no commit, _ANSWER_KEEP for an
            answer read back, or "" for nothing.
    """
    try:
        os.makedirs(run_file.host_directory, exist_ok=True)
        with tmpfile.atomic_write(run_file.mark) as tmp_path:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                handle.write(keep)
    except OSError as e:
        logger.warning("could not write %s: %s", run_file.mark, e)


def _mark_run_of_branch(project_dir: str, out_dir: str, name: str,
                        branch_list: Sequence[str], run_id: str) -> None:
    """Record on this machine's side that a command goes on from the runs of some branches.

    For each branch whose name this tool made up (is_own_branch), an empty record of
    run_id is written for the agent named after the branch (agent_name). A branch of
    another name is passed over. The records are written under _branch_lock(), waited
    for with no limit.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        branch_list: The branches the command was told to go on from.
        run_id: The command's run, as new_run_id() names it.
    """
    own_list = [branch for branch in dict.fromkeys(branch_list) if is_own_branch(branch)]
    if not own_list:
        return
    with _branch_lock(project_dir):
        for branch in own_list:
            _write_run_mark(run_file_layout(out_dir, agent_name(branch, ""), name, run_id),
                            "")


def _run_mark_dict(directory: str) -> dict[str, str]:
    """Return the runs this machine recorded starting in one directory on its side.

    Each comes with what it left to pick up.

    Args:
        directory: The runs' own directory on this machine's side of an agent
            (_sandbox_dir of host_dir).

    Returns:
        {run id: _PATCH_KEEP, _ANSWER_KEEP or ""}, for the entries named
        "run-<run id>.txt" around a whole run id (_RUN_IN_NAME); empty when the
        directory cannot be listed. A record that cannot be read is taken as one of a
        patch left to pick up, which is logged; its run is not taken away for it.
    """
    try:
        entry_list = sorted(os.listdir(directory))
    except OSError:
        return {}
    mark_dict = {}
    for entry in entry_list:
        if not (entry.startswith(_RUN_MARK_PREFIX) and entry.endswith(_RUN_MARK_SUFFIX)):
            continue
        run_id = entry[len(_RUN_MARK_PREFIX):-len(_RUN_MARK_SUFFIX)]
        if not _RUN_IN_NAME.fullmatch(run_id):
            continue
        path = os.path.join(directory, entry)
        try:
            keep = tmpfile.read_file(path, _RUN_MARK_MAX_BYTE).decode("utf-8", "replace")
        except (OSError, ValueError) as e:
            logger.info("could not read %s (%s), so the run is taken to have left a "
                        "patch to pick up", path, e)
            keep = _PATCH_KEEP
        mark_dict[run_id] = keep if keep in (_PATCH_KEEP, _ANSWER_KEEP) else ""
    return mark_dict


def _is_agent_dir_of(agent_path: str, name: str) -> bool:
    """Say whether an agent's directory holds what a run wrote for one repository.

    Args:
        agent_path: An agent's own copy of the output directory, or what this machine
            keeps for the agent (host_dir), as an absolute path.
        name: The repository's directory name.

    Returns:
        True for a directory that holds a directory of that name, and for one that
        holds nothing at all. False for anything else: one that holds the directories
        of other repositories alone, a link, and a path that cannot be listed.
    """
    if not _is_real_dir(agent_path):
        return False
    try:
        is_empty = not os.listdir(agent_path)
    except OSError:
        return False
    return is_empty or _is_real_dir(os.path.join(agent_path, name))


def _remove_agent_dir(agent_path: str, name: str, what: str) -> None:
    """Take away what an agent's directory holds for one repository.

    The directory named after the repository goes with the lock taken on it, then the
    agent's directory and its lock when nothing is left in it. What the directory
    holds for another repository stays.

    Args:
        agent_path: An agent's own copy of the output directory, or what this machine
            keeps for the agent (host_dir), as an absolute path.
        name: The repository's directory name.
        what: What the directory held, for the message when it cannot be taken away.
    """
    own_dir = os.path.join(agent_path, name)
    if _is_real_dir(own_dir):
        _remove_tree(own_dir, what)
        _remove_file(filelock.path_for(own_dir), f"the lock of {own_dir}")
    try:
        os.rmdir(agent_path)
    except OSError:
        return
    _remove_file(filelock.path_for(agent_path), f"the lock of {agent_path}")


def _run_time(run_id: str) -> str:
    """Return the time a run's name starts with.

    Examples:
        "20260101-000000-1-42" -> "20260101-000000"

    Args:
        run_id: The run, as new_run_id() names it.

    Returns:
        The date and the time, joined by "-".
    """
    return "-".join(run_id.split("-")[:2])


def _run_process(run_id: str) -> str:
    """Return the process a run's name ends with.

    Examples:
        "20260101-000000-1-42" -> "42"

    Args:
        run_id: The run, as new_run_id() names it.

    Returns:
        The process ID, as text.
    """
    return run_id.rsplit("-", 1)[-1]


def remain_branch_list(project_dir: str, out_dir: str, since_id: str) -> list[str]:
    """Return the branches the runs this process started have left to carry on from.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        since_id: A name new_run_id() made. A run named for an earlier time is not
            one of this command's.

    Returns:
        The task branches of those runs, the last try alone where a task was handed
        out again (_last_try_list), and their unions a run is carried on from
        (open_branch_list), oldest run first; empty when they left none, and when
        git does not list the branches. A union among them stands for what it holds:
        the branches named under it for the tests of its stage (_TEST_BRANCH) are
        left out, and so are a task branch a union of those runs reaches and a union
        another union of those runs was made over (reaches, and is not reached by),
        whether that union is one to carry on from or one the working tree holds.
        Where dispatch() ran the request again (_RERUN_LIST), the last run of
        it stands for the runs before it: once it has put its union into the working
        tree or left one still to go in (last_run_has_union), its own branches alone
        are named; else once it has left a branch of its own, its branches are named
        after the unions the cases a run before it found not done are recorded on,
        and until it has, the branches it was told to carry on from are named.
    """
    since_time = _run_time(since_id)
    process_tail = f"-{os.getpid()}"
    try:
        remain_dict = _run_remain_dict(project_dir, out_dir)
    except RuntimeError:
        return []
    # (the branch, whether it is a task's, the run that left it) for every branch to
    # carry on from
    triple_list, product_list = [], []
    for run_id in sorted(remain_dict):
        if not run_id.endswith(process_tail) or _run_time(run_id) < since_time:
            continue
        by_kind_dict = remain_dict[run_id]
        triple_list += [(branch, True, run_id)
                        for branch in _last_try_list(by_kind_dict["tasks"])]
        triple_list += [(branch, False, run_id)
                        for branch in open_branch_list(project_dir, by_kind_dict["products"])]
        product_list += by_kind_dict["products"]
    open_list = [branch for branch, is_task, _ in triple_list if not is_task]
    name_list = [branch for branch in open_list if not _is_test_branch(branch, open_list)]
    union_list = [branch for branch in product_list
                  if not _is_test_branch(branch, product_list)]
    remain_list = [(branch, run_id) for branch, is_task, run_id in triple_list
                   if (is_task or branch in name_list) and not any(
                       gitcmd.is_ancestor(project_dir, branch, other)
                       and (is_task or not gitcmd.is_ancestor(project_dir, other, branch))
                       for other in (product_list if is_task else union_list)
                       if other != branch)]
    if not _RERUN_LIST:
        return [branch for branch, _ in remain_list]
    start_count, resume_list, gap_list, _ = _RERUN_LIST[-1]
    own_list = [branch for branch, run_id in remain_list
                if _run_count(run_id) > start_count]
    if last_run_has_union(project_dir, own_list):
        return own_list
    return list(dict.fromkeys([*gap_list, *own_list])) if own_list else list(resume_list)


def last_run_has_union(project_dir: str, branch_list: Sequence[str]) -> bool:
    """Say whether the last run of this command has put its union in or left one to go in.

    The union goes into the working tree.

    Args:
        project_dir: The target repository, as an absolute path.
        branch_list: The branches that run left to carry on from.

    Returns:
        True when land_list() has grown since that run started (the last entry of
        _RERUN_LIST, or the start of the command), and when one of branch_list is a
        union whose merge was settled and that the working tree does not hold
        (_is_union_ahead).
    """
    land_count = _RERUN_LIST[-1][3] if _RERUN_LIST else 0
    return len(_LAND_LIST) > land_count or any(
        _is_union_ahead(project_dir, branch) for branch in branch_list)


def _is_union_ahead(project_dir: str, branch: str) -> bool:
    """Say whether a branch is a settled union that the working tree does not hold.

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.

    Returns:
        True when the commit under the commits of the branch's stages
        (integrate.below_stage) is a merge a merging run wrote and settled
        (integrate.is_settle_merge) and the working tree does not hold all of what
        the branch made (_is_in_tree). False as well for a branch that is not there
        or cannot be read.
    """
    if not branch_tip(project_dir, branch):
        return False
    try:
        merge = integrate_mod.below_stage(
            project_dir, integrate_mod.commit_of(project_dir, branch))
        return integrate_mod.is_settle_merge(project_dir, merge) \
            and not _is_in_tree(project_dir, branch)
    except RuntimeError:
        return False


def _run_count(run_id: str) -> int:
    """Return the count of the runs its process had started that a run's name holds.

    Examples:
        "20260101-000000-12-42" -> 12

    Args:
        run_id: The run, as new_run_id() names it.

    Returns:
        The count.
    """
    return int(run_id.split("-")[2])


def _is_test_branch(branch: str, union_list: Sequence[str]) -> bool:
    """Say whether a branch is one named under a union for the tests of its stage.

    Args:
        branch: The branch's name.
        union_list: The unions it may be named under.

    Returns:
        True when it is named as _TEST_BRANCH names one under any of them.
    """
    return any(branch.startswith(_TEST_BRANCH.format(branch=union, n=""))
               for union in union_list)


# The trailers a commit records its run's exit status and its verdict under
_EXIT_STATUS_TRAILER = integrate_mod.EXIT_STATUS_TRAILER
_VERDICT_TRAILER = integrate_mod.VERDICT_TRAILER

# What the subject of the commit of a merge starts with
_MERGE_PREFIX = integrate_mod.MERGE_PREFIX

# The exit status a commit records, as _hold_meta_number() holds it
_EXIT_STATUS_RE = re.compile(r"[0-9]{1,3}")


def _merge_exit_status(project_dir: str, branch: str) -> int:
    """Return the exit status the run that left a union recorded for it.

    It is read off the commit under the ones the stages wrote (integrate.below_stage).

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.

    Returns:
        The status, or 0 when the commit records none or the branch cannot be read.
    """
    try:
        commit = integrate_mod.below_stage(
            project_dir, integrate_mod.commit_of(project_dir, branch))
        return _exit_status_at(project_dir, commit)
    except RuntimeError:
        return 0


def _exit_status_at(project_dir: str, commit: str) -> int:
    """Return the exit status one commit records, as run_trailer_dict() wrote it.

    Args:
        project_dir: The repository, as an absolute path.
        commit: The commit to read, and no commit it reaches.

    Returns:
        The status, or 0 when the commit records none.

    Raises:
        RuntimeError: When the commit's message could not be read.
    """
    text = integrate_mod.trailer_value(
        integrate_mod.own_trailer_list(project_dir, commit), _EXIT_STATUS_TRAILER)
    return int(text) if _EXIT_STATUS_RE.fullmatch(text) else 0


def open_branch_list(project_dir: str, branch_list: Sequence[str]) -> list[str]:
    """Return the branches among those named that a run is carried on from.

    They are a branch that holds no result (_settle_branch_list), one whose run recorded
    another exit status than 0 (_merge_exit_status), whatever verdict it carries, and
    one not all of whose change the working tree holds (_is_in_tree). A branch that
    cannot be read is one of them.

    Args:
        project_dir: The repository, as an absolute path.
        branch_list: The names of branches the repository holds.

    Returns:
        The branches among them, in their order.
    """
    open_list = []
    for branch in branch_list:
        try:
            is_open = not _settle_branch_list(project_dir, [branch]) \
                or bool(_merge_exit_status(project_dir, branch)) \
                or not _is_in_tree(project_dir, branch)
        except RuntimeError:
            is_open = True
        if is_open:
            open_list.append(branch)
    return open_list


def _result_in_tree_list(project_dir: str, branch_list: Sequence[str]) -> list[str]:
    """Return the branches among those named that hold a result the working tree holds.

    Args:
        project_dir: The repository, as an absolute path.
        branch_list: The branch names, as a command was passed them.

    Returns:
        The branches whose name this tool made up (is_own_branch), that are in the
        repository and that open_branch_list() leaves out, in their order.
    """
    own_list = [branch for branch in branch_list
                if is_own_branch(branch) and branch_tip(project_dir, branch)]
    open_list = open_branch_list(project_dir, own_list)
    return [branch for branch in own_list if branch not in open_list]


def _is_in_tree(project_dir: str, branch: str) -> bool:
    """Say whether the working tree holds all of what the last run on a branch made.

    The run is the one that last committed to the branch. A working tree that holds the
    merge and not the commits the stages left on it does not hold the branch.

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.

    Returns:
        True when what is checked out reaches the branch's tip, or the working tree
        holds the change from the commit the run was written against (the first
        parent of integrate.below_stage()'s commit) up to the tip, each commit of it
        included (gitcmd.has_change_in_tree). False when it does not, and for a branch
        or a change that cannot be read.
    """
    try:
        tip = integrate_mod.commit_of(project_dir, branch)
        merge = integrate_mod.below_stage(project_dir, tip)
    except RuntimeError:
        return False
    head = gitcmd.head_commit(project_dir)
    if not head:
        return False
    if gitcmd.is_ancestor(project_dir, tip, head):
        return True
    return bool(gitcmd.has_change_in_tree(project_dir, f"{merge}^", tip))


def is_union_in_tree(project_dir: str, branch: str) -> bool:
    """Say whether a branch is an unsettled union whose merged commits the tree holds.

    The union is one an earlier merge left unsettled, and the working tree holds every
    one of the commits it merged.

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.

    Returns:
        True for a branch whose name this tool made up (is_own_branch), whose tip is
        a union left unsettled (integrate.earlier_fork_dict() names the commits it
        was merged from, opening every unsettled union under it too) and of whose
        commits each is held by the working tree (_is_side_in_tree). False otherwise,
        and for a branch that cannot be read.
    """
    if not is_own_branch(branch):
        return False
    try:
        fork_dict = integrate_mod.earlier_fork_dict(
            project_dir, integrate_mod.commit_of(project_dir, branch))
    except RuntimeError:
        return False
    return bool(fork_dict) and all(_is_side_in_tree(project_dir, side, fork_list)
                                   for side, fork_list in fork_dict.items())


def _is_side_in_tree(project_dir: str, side: str, fork_list: list[str]) -> bool:
    """Say whether the working tree holds all of what one side of an unsettled union brings.

    The side is one commit the union was merged from.

    A commit whose merge under the stages (integrate.below_stage) is one a merging
    run settled is read as _is_in_tree() reads a branch. Any other commit - the tip
    of a branch somebody made, the last task branch of a chain - is read from where
    it parts from what the union was written against.

    Args:
        project_dir: The repository, as an absolute path.
        side: The commit.
        fork_list: integrate.earlier_fork_dict()'s merge bases for it.

    Returns:
        _is_in_tree() for a commit over a settled merge. For any other, True when
        what is checked out reaches it, or there is a merge base and the working
        tree holds the change from each of them up to it, each commit included
        (gitcmd.has_change_in_tree). False when it does not, and for a commit or a
        change that cannot be read.
    """
    try:
        is_settle = integrate_mod.is_settle_merge(
            project_dir, integrate_mod.below_stage(project_dir, side))
    except RuntimeError:
        return False
    if is_settle:
        return _is_in_tree(project_dir, side)
    head = gitcmd.head_commit(project_dir)
    if not head:
        return False
    if gitcmd.is_ancestor(project_dir, side, head):
        return True
    return bool(fork_list) and all(gitcmd.has_change_in_tree(project_dir, fork, side)
                                   for fork in fork_list)


# What _RETRY_SUFFIX adds to a task branch's name, with the try it names
_RETRY_IN_NAME = re.compile(re.escape(_RETRY_SUFFIX).replace(r"\{attempt\}", r"(\d+)") + "$")


def _last_try_list(branch_list: list[str]) -> list[str]:
    """Return the task branches of one run without the tries that were done again.

    Examples:
        ["codetwine-loom/run/r/1", "codetwine-loom/run/r/2",
         "codetwine-loom/run/r/2-retry1"]
            -> ["codetwine-loom/run/r/1", "codetwine-loom/run/r/2-retry1"]

    Args:
        branch_list: The task branches of one run, as _run_remain_dict() lists them.

    Returns:
        For each task, the branch of its last try that is there, in the order of the
        tasks' names.
    """
    last_dict: dict[str, tuple[int, str]] = {}
    for branch in branch_list:
        match = _RETRY_IN_NAME.search(branch)
        task, retry = (branch[:match.start()], int(match.group(1))) if match \
            else (branch, 0)
        if task not in last_dict or retry > last_dict[task][0]:
            last_dict[task] = (retry, branch)
    return [branch for _, (_, branch) in sorted(last_dict.items())]


def remove_old_run(project_dir: str, out_dir: str, keep_count: int, engine_path: str = "",
                   protect_list: Sequence[str] = ()) -> list[str]:
    """Take away what every run but the newest few left behind.

    What is counted is the runs that still hold task branches, a union that holds no
    result (_settle_branch_list: one the merging run recorded as unsettled, one a stage
    is still ahead of, one whose tests recorded a case the change does not do), the
    branch of a run made on its own that recorded "failed", an answer read back, or a
    patch that landed in no commit (the last two as this machine recorded them,
    _run_remain_dict). They are counted by the command that made them: the runs one
    process started (_run_process) - the runs of a request that was run again, each of
    which left branches of its own - take up one place between them. A run a later
    command carried on from is counted by the newest run this machine recorded under its
    agents ("runs"). A run is still going while the process of the run itself, or of the
    newest run of any process recorded under its agents, or a container of one of those
    runs, is there (_run_dead). The runs of the newest few commands are kept whole and
    the rest go, branches, unions and output alike; a run that holds none of the above
    takes up none of those places and leaves no output behind.

    Two things are never taken away here. The branch a union landed on that holds a
    result (_settle_branch_list) is the result of the run: while it is there, that run's
    output stays with it. A union that cannot be read is held to be one
    (_keep_branch_list). And a name this tool did not make up carries no run: a branch
    somebody else wrote is not looked at.

    The runs are those of this repository: its branches, and under the output
    directory what the agents' directories hold under the repository's directory
    name (_run_remain_dict). What an agent's directory holds for another repository
    is neither counted nor taken away (_remove_agent_dir). Nothing is taken away
    when git does not list the branches.

    The run id starts with the time, and the names sort the way the runs happened.

    The worktrees runs that ended left registered are taken away first
    (remove_dead_worktree). That, reading the runs and deleting their branches are done
    under the lock the runs writing branches share (_branch_lock). The lock is not
    waited for: a sweep that finds another process holding it takes nothing away, which
    is logged. The output directories of the runs whose branches were deleted are taken
    away once the lock is given up.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        keep_count: How many commands' runs that can still be carried on from are kept,
            newest first. 0 or less takes nothing away.
        engine_path: The engine's executable, for telling a run that is still going from
            one that ended. A run still going is left alone whatever its age.
        protect_list: Runs to leave alone whatever their age, named either by their id
            or by one of their branches.

    Returns:
        The runs whose leftovers were taken away, newest first.
    """
    if keep_count <= 0:
        return []
    try:
        with _branch_lock(project_dir, wait=False):
            remove_dead_worktree(project_dir, engine_path)
            dir_dict = _remove_old_branch(project_dir, out_dir, keep_count, engine_path,
                                          protect_list)
    except filelock.LockBusy as e:
        logger.info("%s, so what the older runs left is left where it is", e)
        return []
    name = knowledge.project_name_of(project_dir)
    for run_id, path_list in dir_dict.items():
        for path in path_list:
            _remove_agent_dir(path, name, f"what {run_id} wrote")
    remove_list = list(dir_dict)
    if remove_list:
        logger.info("%d run(s) beyond those of the newest %d command(s) that can be "
                    "carried on from were taken away: %s", len(remove_list), keep_count,
                    ", ".join(remove_list))
    return remove_list


def _remove_old_branch(project_dir: str, out_dir: str, keep_count: int, engine_path: str,
                       protect_list: Sequence[str]) -> dict[str, list[str]]:
    """Delete the branches of the runs remove_old_run() takes away, and say which.

    Args:
        As remove_old_run() takes them, keep_count being 1 or more.

    Returns:
        {run id: the output directories it left, as _run_remain_dict() lists them},
        newest first, for the runs whose branches were deleted; empty when git does
        not list the branches.
    """
    try:
        remain_dict = _run_remain_dict(project_dir, out_dir)
    except RuntimeError as e:
        logger.warning("%s, so what the older runs left is left where it is", e)
        return {}
    last_dict = {run_id: max([run_id, *remain_dict[run_id]["runs"]])
                 for run_id in remain_dict}
    order_list = sorted(remain_dict, key=lambda run_id: last_dict[run_id], reverse=True)
    settle_dict = {run_id: _keep_branch_list(project_dir, remain_dict[run_id]["products"])
                   for run_id in order_list}
    # A question leaves no branch; a run of one is kept for the answer read back
    resume_run_list = [run_id for run_id in order_list
                       if remain_dict[run_id]["tasks"]
                       or remain_dict[run_id]["answers"]
                       or remain_dict[run_id]["patches"]
                       or len(remain_dict[run_id]["products"]) > len(settle_dict[run_id])]
    # One place per command: the runs one process started, or last carried on from,
    # are counted as one
    process_list = list(dict.fromkeys(_run_process(last_dict[run_id])
                                      for run_id in resume_run_list))
    keep_set = {run_id for run_id in resume_run_list
                if _run_process(last_dict[run_id]) in process_list[:keep_count]}
    keep_set |= {run_in(name) for name in protect_list}
    dir_dict = {}
    for run_id in order_list:
        by_kind_dict = remain_dict[run_id]
        if run_id in keep_set or settle_dict[run_id]:
            continue
        # The newest run of each process recorded under the run's agents
        newest_dict = {_run_process(one_run): one_run
                       for one_run in sorted(by_kind_dict["runs"])}
        if not all(_run_dead(one_run, engine_path)
                   for one_run in dict.fromkeys([run_id, *newest_dict.values()])):
            logger.info("%s is still going, so what it has written is left where it is",
                        run_id)
            continue
        for branch in by_kind_dict["tasks"] + by_kind_dict["products"] + by_kind_dict["tree"]:
            _delete_branch(project_dir, branch)
        dir_dict[run_id] = by_kind_dict["dirs"]
    return dir_dict


def _keep_branch_list(project_dir: str, branch_list: list[str]) -> list[str]:
    """Return the unions among those one run left that hold a result or cannot be read.

    Args:
        project_dir: The repository, as an absolute path.
        branch_list: The union branches one run left, as _run_remain_dict() lists them.

    Returns:
        The branches that hold a result (_settle_branch_list), and the ones that
        cannot be read, which is logged: a branch another command took away after
        the listing is one of those.
    """
    keep_list = []
    for branch in branch_list:
        try:
            keep_list += _settle_branch_list(project_dir, [branch])
        except RuntimeError as e:
            logger.info("%s cannot be read (%s), so what its run left is left where "
                        "it is", branch, e)
            keep_list.append(branch)
    return keep_list


def remove_old_run_in(project: str, output_dir: str, engine_path: str = "",
                      protect_list: Sequence[str] = ()) -> list[str]:
    """Take away what the runs beyond the newest DISPATCH_KEEP_RUN left behind.

    Args:
        project: The target repository. A relative path is resolved from here.
        output_dir: The output directory. A relative path is resolved from here.
        engine_path: The engine's executable, as remove_old_run() takes it.
        protect_list: Runs to leave alone, as remove_old_run() takes them.

    Returns:
        remove_old_run()'s return value.
    """
    project_dir, root_out_dir, _ = _run_place(project, output_dir)
    return remove_old_run(project_dir, root_out_dir, config.DISPATCH_KEEP_RUN,
                          engine_path, protect_list=protect_list)


def _settle_branch_list(project_dir: str, branch_list: list[str]) -> list[str]:
    """Return the branches under this tool's own root that hold a result.

    The verdict is read off the branch's tip as integrate.branch_record_dict() reads
    it; the last record over that range is the newest commit's own. A branch holds a
    result unless it is one a run is carried on from: that verdict is a merging run's
    "unsettled" or an editing run's "failed", integrate.branch_record_dict() reads a
    case the change does not do ("gap"), or a stage is still ahead of the settled merge
    under its tip (stage_resume_dict). A branch carrying no verdict at all holds one.

    Args:
        project_dir: The repository, as an absolute path.
        branch_list: The union branches one run left, as _run_remain_dict() lists them.

    Returns:
        The branches among them to keep whatever the run's age.
    """
    keep_list = []
    for branch in branch_list:
        branch_dict = integrate_mod.branch_record_dict(project_dir, branch)
        record_list = branch_dict["told"]
        verdict = record_list[-1]["verdict"] if record_list else ""
        if verdict in (integrate_mod.UNSETTLE_VERDICT, history.VERDICT_FAIL) \
                or branch_dict["gap"] \
                or _is_stage_ahead(project_dir, branch):
            continue
        keep_list.append(branch)
    return keep_list


def _is_stage_ahead(project_dir: str, branch: str) -> bool:
    """Say whether a branch holds a settled merge a stage is still to run over.

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.

    Returns:
        True when stage_resume_dict() names a stage for it. False as well for a
        branch it cannot read.
    """
    try:
        return bool(stage_resume_dict(project_dir, branch))
    except RuntimeError:
        return False


def branch_tip(project_dir: str, branch: str) -> str:
    """Return the commit a branch points at, or "" when there is no such branch.

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.

    Returns:
        The full commit hash, or "".
    """
    proc = gitcmd.run(project_dir, "rev-parse", "--verify", "--quiet",
                      f"refs/heads/{branch}")
    return proc.stdout.strip() if proc.returncode == 0 else ""


def index_describe(project_dir: str, out_dir: str, name: str, commit: str) -> bool:
    """Say whether the index in an output directory describes a commit.

    It does when the record beside the index's analysis names the codetwine installed
    now (knowledge.same_codetwine()) and either the commit the index's analysis records
    is that commit, or the value the record holds of what it read is the one the
    commit's tree gives with the names and paths the record keeps out
    (knowledge.commit_state()). The second is how an analysis made over a working tree
    with changes, which records no commit, is found to describe the commit a run made of
    that tree. An analysis with no record beside it, a missing index, an unreadable
    one, one that is not a regular file, one built from another analysis than the
    record's and one reached through a link that leads out of out_dir (inside_path) all
    count as not describing it, and so do an index that holds something of another
    shape where store.analysis_of_index() reads it and a record whose "exclude" is not
    a list of texts. The index is not loaded: what is read of it is what
    store.analysis_of_index() reads.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name, which is the directory under out_dir.
        commit: The commit a run starts at.

    Returns:
        True when the analysis the index was built from describes commit's tree.
    """
    index_path = store.index_db_path(out_dir, name)
    knowledge_path = knowledge.knowledge_db_path(out_dir, name)
    try:
        for path in (index_path, knowledge_path,
                     knowledge.analysis_record_path(knowledge_path)):
            inside_path(path, out_dir)
        if os.path.lexists(index_path) and not os.path.isfile(index_path):
            raise RuntimeError(f"{index_path} is not a regular file")
        stamp, index_commit = store.analysis_of_index(index_path)
    except (FileNotFoundError, sqlite3.DatabaseError, RuntimeError) as e:
        logger.info("could not read what %s describes: %s", index_path, e)
        return False
    record_dict = knowledge.analysis_record(knowledge_path, stamp)
    if not knowledge.same_codetwine(record_dict):
        return False
    if index_commit and index_commit == commit:
        return True
    state = str(record_dict.get("state") or "")
    exclude_list = record_dict.get("exclude", [])
    if not isinstance(exclude_list, list) \
            or not all(isinstance(entry, str) for entry in exclude_list):
        return False
    return bool(state) and state == knowledge.commit_state(project_dir, commit,
                                                           exclude_list)


def path_list_text(path_list: Sequence[str]) -> str:
    """Render paths as the one line a trailer holds them on.

    Every whitespace character is written as its JSON escape: the line holds none.

    Examples:
        ["a.py", "my lib/b.py"] -> '["a.py","my\\u0020lib/b.py"]'
        []                      -> ""

    Args:
        path_list: The paths.

    Returns:
        The paths as a JSON list on one line, or "" when there are none.
    """
    if not path_list:
        return ""
    text = json.dumps(list(path_list), ensure_ascii=False, separators=(",", ":"))
    return re.sub(r"\s", lambda match: f"\\u{ord(match.group()):04x}", text)


def path_list_of(text: str) -> list[str] | None:
    """Read a trailer's text back into the paths it holds; path_list_text()'s inverse.

    Args:
        text: The trailer's value, "" when the commit carries none.

    Returns:
        The paths, [] for "", and None for text that is not a JSON list of strings,
        which is logged.
    """
    if not text:
        return []
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        value = None
    if not isinstance(value, list) or not all(isinstance(path, str) for path in value):
        logger.warning("the paths a trailer holds cannot be read: %s", cut(text, 200))
        return None
    return value


# What commit_message() writes as the record of a commit that records nothing
_NO_RECORD_LINE = f"{_TRAILER_PREFIX}Record: none"


def run_trailer_dict(meta_dict: dict, attempt_dict: dict, request: str = "") -> dict:
    """Return what the machine knows about a finished run, as commit trailers.

    The values are what the machine decided about the run. An empty attempt gives every
    trailer that comes from it None or "".

    The trailers of a run that changed a file without going through an editing tool
    carry the paths of those files, and the trailers of a run that changed test files
    name them.

    Args:
        meta_dict: The summary the container wrote beside the patch.
        attempt_dict: The history entry this run recorded, or {} when it recorded none.
        request: What the run was given, whole: for one task of a split request, the
            request the tasks were split out of. Empty gives its trailer "".

    Returns:
        {trailer name: value}; None or "" for a trailer with no value.
    """
    return {
        # The request the run was given, whole
        f"{_TRAILER_PREFIX}Run-Request": request,
        f"{_TRAILER_PREFIX}Attempt": attempt_dict.get("n"),
        _VERDICT_TRAILER: attempt_dict.get("verdict"),
        f"{_TRAILER_PREFIX}Why": attempt_dict.get("why"),
        # What the run said it did
        f"{_TRAILER_PREFIX}Did": attempt_dict.get("did"),
        _EXIT_STATUS_TRAILER: meta_dict.get("exit_status"),
        # Empty when the run wrote through the tools alone
        f"{_TRAILER_PREFIX}Untooled": " ".join(attempt_dict.get("untooled") or ()),
        # The tests this run changed, a red test the run turned green by changing the
        # test among them
        f"{_TRAILER_PREFIX}Tests-Changed": " ".join(attempt_dict.get("tests_changed") or ()),
        # How a merging run ended and what it recorded with record_decision(). A change
        # run records neither and its commit carries neither
        f"{_TRAILER_PREFIX}Ended": attempt_dict.get("ended"),
        f"{_TRAILER_PREFIX}Decisions": integrate_mod.decision_text(
            attempt_dict.get("decisions") or [], config.DISPATCH_ANSWER_MAX_CHAR),
        # The separator lines a merging run that left its union unsettled says the union
        # may hold
        integrate_mod.SEPARATOR_TRAILER: integrate_mod.separator_text(
            attempt_dict.get("separator")),
        # The sides whose change to a file that run kept out of the union
        integrate_mod.NO_MERGE_TRAILER: integrate_mod.no_merge_text(
            attempt_dict.get("no_merge")),
        # Which stage the commit is of, the files a stage changed that are not tests,
        # and what the tests the stage that writes tests wrote say over the commit the
        # change was written against
        f"{_TRAILER_PREFIX}Stage": attempt_dict.get("stage"),
        f"{_TRAILER_PREFIX}Non-Test-Changed": " ".join(
            attempt_dict.get("non_test_changed") or ()),
        f"{_TRAILER_PREFIX}Tests-Base": attempt_dict.get("tests_base"),
        # The files whose tests were left unsettled when that stage's test tasks were
        # combined
        f"{_TRAILER_PREFIX}Test-Open": path_list_text(attempt_dict.get("test_open") or ()),
        # The cases the stage that writes tests found the change does not do, as it
        # recorded them with record_gap()
        f"{_TRAILER_PREFIX}Gap": integrate_mod.decision_text(attempt_dict.get("gap") or []),
        # The cases the test tasks recorded that the combined tests found done
        f"{_TRAILER_PREFIX}Gap-Done": integrate_mod.decision_text(
            attempt_dict.get("gap_done") or []),
        # The last commit of an earlier union the commit a merge was written against
        # holds (integrate.union_part_list)
        f"{_TRAILER_PREFIX}Stand-On": attempt_dict.get("stand_on"),
    }


def commit_message(request: str, trailer_dict: dict, prefix: str = "sandbox") -> str:
    """Build the message one taken-in patch is committed with.

    The subject is prefix, then the request cut to _MAX_SUBJECT_CHAR; the request
    follows in full, and the record comes last, as a paragraph of its own: one line per
    trailer, each value on one line with the whitespace inside it collapsed. The message
    always ends with that paragraph. Where no trailer holds a value, the paragraph is
    _NO_RECORD_LINE. Every NUL character is left out, the request's and a value's alike.

    Examples:
        ("make it stop\\ncrashing", {"Codetwine-Loom-Verdict": "unverified"})
            -> "sandbox: make it stop crashing\\n\\nmake it stop\\ncrashing\\n\\n"
               "Codetwine-Loom-Verdict: unverified"
        ("", {})
            -> "sandbox: (no request)\\n\\n(no request)\\n\\n"
               "Codetwine-Loom-Record: none"

    Args:
        request: The change request as it was given.
        trailer_dict: {trailer name: value}, as run_trailer_dict() returns them. A value
            of None or "" is left out; the name of one that is kept must hold no
            whitespace.
        prefix: What the subject starts with, naming the subcommand that produced it.

    Returns:
        The whole message, subject first.
    """
    body = request.replace("\0", "").strip() or "(no request)"
    record_list = []
    for name, value in trailer_dict.items():
        value_text = "" if value is None else one_line(str(value).replace("\0", ""))
        if value_text:
            record_list.append(f"{name}: {value_text}")
    return "\n".join([f"{prefix}: {one_line(body)[:_MAX_SUBJECT_CHAR]}", "", body, "",
                      *(record_list or [_NO_RECORD_LINE])])


def _take_in_lock(project_dir: str) -> str:
    """Return what the runs taking a patch in to one repository lock on.

    It goes in git's common directory. The main working tree and every linked worktree
    of one repository lock on the same path.

    Args:
        project_dir: The repository, as an absolute path.

    Returns:
        The path filelock takes its lock beside.
    """
    proc = gitcmd.run(project_dir, "rev-parse", "--git-common-dir")
    git_dir = proc.stdout.strip() if proc.returncode == 0 else ""
    return os.path.join(project_dir, git_dir or ".git", "codetwine-loom-take-in")


def _branch_lock(project_dir: str, wait: bool = True) -> contextlib.AbstractContextManager:
    """Return the lock the runs writing branches or the working tree of a repository share.

    Args:
        project_dir: The repository, as an absolute path.
        wait: As filelock.hold_lock() takes it.

    Returns:
        filelock.hold_lock()'s context manager on _take_in_lock().
    """
    return filelock.hold_lock(_take_in_lock(project_dir), "a branch in this repository",
                              wait=wait)


def _delete_branch(project_dir: str, branch: str) -> bool:
    """Delete one branch, logging git's message when it is left where it is.

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.

    Returns:
        True when the branch was deleted.
    """
    proc = gitcmd.run(project_dir, "branch", "-D", branch)
    if proc.returncode != 0:
        logger.info("%s was left where it is: %s", branch, proc.stderr.strip())
        return False
    return True


def take_in_patch(project_dir: str, patch_path: str, base: str, branch: str,
                  worktree_dir: str, message: str, parent_list: Sequence[str] = (),
                  reset: bool = False, engine: str = "") -> dict:
    """Take a patch in, on a branch of its own, without touching the working tree.

    Steps:
    1. Add a worktree at base without checking its files out, making the branch
       when it is not there yet
    2. Where the commit is to have more parents, record those of them that are
       independent of the branch's tip and of each other, keeping the tip's tree
    3. Check the patch applies there, then apply it to the worktree's index alone
       (_commit_on_index): the commit holds every file of the patch as the patch
       writes it, whatever this machine's ignore rules, attributes and filter drivers
       say, and nothing else. Without a patch nothing is applied and the commit
       holds its parent's tree
    4. Commit, and remove the worktree again. Every one of these commands is started
       with gitcmd.COMMITTER_TUPLE

    A branch that is already there is committed onto, and a run that continues an
    earlier one stacks on what the earlier one left, unless reset asks for it to be made
    again at base. A branch made here, and the worktree either way, are taken away again
    when the commit does not happen: a failure leaves the repository as it was and the
    patch file still on disk.

    One patch is taken in at a time, under the lock filelock takes on the repository;
    runs finishing at once wait here. The worktrees runs that ended left registered are
    taken away first (remove_dead_worktree).

    Args:
        project_dir: The repository the branch is made in, as an absolute path.
        patch_path: The patch the container wrote, as an absolute path, or "" for a
            commit that changes nothing and carries the message alone.
        base: The commit the branch starts at, which is the commit the patch was made
            against. It is ignored for a branch that already exists, whose tip is where
            the commit lands, unless reset is given.
        branch: The branch's name.
        worktree_dir: Where the worktree is put while the patch is applied. It must not
            exist yet.
        message: The commit message.
        parent_list: The commits the commit has as parents besides base, which a patch
            combining several branches names them by. One base already reaches is left
            out by git.
        reset: Whether a branch that already exists is moved to base first rather than
            committed onto. Its old tip is not kept.
        engine: The container engine the run uses, as remove_dead_worktree() takes it.

    Returns:
        {"branch": the branch's name, "commit": the commit's hash}.

    Raises:
        RuntimeError: When the worktree cannot be made, when the patch does not apply,
            or when the commit fails. The message holds git's own output.
    """
    with _branch_lock(project_dir):
        return _apply_on_branch(project_dir, patch_path, base, branch, worktree_dir,
                                message, parent_list, reset, engine)


def _apply_on_branch(project_dir: str, patch_path: str, base: str, branch: str,
                     worktree_dir: str, message: str, parent_list: Sequence[str] = (),
                     reset: bool = False, engine: str = "") -> dict:
    """Add the worktree, apply the patch to its index and commit it.

    It is run under _branch_lock().

    Args:
        project_dir: The repository the branch is made in, as an absolute path.
        patch_path: The patch the container wrote, as an absolute path, or "".
        base: The commit the branch starts at.
        branch: The branch's name.
        worktree_dir: Where the worktree is put while the patch is applied.
        message: The commit message.
        parent_list: The commit's parents besides base, as take_in_patch() says.
        reset: Whether an existing branch is moved to base first, as take_in_patch() says.
        engine: The container engine the run uses, as take_in_patch() says.

    Returns:
        {"branch": the branch's name, "commit": the commit's hash}.

    Raises:
        RuntimeError: As take_in_patch() says.
    """
    remove_dead_worktree(project_dir, engine)
    is_new_branch = not branch_tip(project_dir, branch)
    add_argv = ["worktree", "add", "--quiet", "--no-checkout"]
    add_argv += ["-B" if reset else "-b", branch, worktree_dir, base] \
        if is_new_branch or reset else [worktree_dir, branch]
    proc = gitcmd.run(project_dir, *add_argv)
    if proc.returncode != 0:
        raise RuntimeError(f"could not make a worktree for {branch}: "
                           f"{proc.stderr.strip()}")
    is_commit = False
    try:
        commit = _commit_on_index(worktree_dir, patch_path, message, parent_list)
        is_commit = True
    finally:
        gitcmd.run(project_dir, "worktree", "remove", "--force", worktree_dir)
        # "worktree add -b" makes the branch before the patch is read; a branch that
        # never got its commit is taken away again along with the worktree. One that was
        # already there is left as it was
        if not is_commit and is_new_branch:
            gitcmd.run(project_dir, "branch", "-D", branch)
    return {"branch": branch, "commit": commit}


def _commit_on_index(worktree_dir: str, patch_path: str, message: str,
                     parent_list: Sequence[str]) -> str:
    """Commit a patch on the branch a worktree is on, through its index alone.

    Steps:
    1. Read the tree of HEAD into the worktree's index
    2. Check the patch applies to the index, then apply it there
    3. Write the index as a tree, and commit it with HEAD and those of parent_list
       that are independent of HEAD and of each other as its parents
    4. Move the branch to the commit

    No file is written to the worktree or read from it. The message is stored as
    "git stripspace" leaves it. The git commands that read the patch are held to
    gitcmd.MAX_PATCH_ADDRESS_BYTE.

    Args:
        worktree_dir: The worktree, whose files are not checked out.
        patch_path: The patch, as an absolute path, or "" for a commit that holds its
            first parent's tree.
        message: The commit message.
        parent_list: The commit's parents besides HEAD, as take_in_patch() says.

    Returns:
        The commit's hash.

    Raises:
        RuntimeError: When a git command fails, when the patch leaves the tree as it
            was in a commit with one parent, or when the message is empty. The
            message holds git's own output.
    """
    what = patch_path or "a commit that changes nothing"

    def git(*args: str, input_text: str | None = None,
            max_address_byte: int | None = None) -> str:
        """Run one git command in the worktree and return what it wrote."""
        proc = gitcmd.run(worktree_dir, *gitcmd.COMMITTER_TUPLE, *args,
                          input_text=input_text, max_address_byte=max_address_byte)
        if proc.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed while taking in {what}: "
                               f"{(proc.stderr or proc.stdout).strip()}")
        return proc.stdout

    # == Step 1: the index of HEAD ============================
    head = git("rev-parse", "--verify", "HEAD^{commit}").strip()
    git("read-tree", "HEAD")
    # == Step 2: the patch, on the index alone ================
    if patch_path:
        for argv in ([*gitcmd.APPLY_TUPLE, "--cached", "--check", patch_path],
                     [*gitcmd.APPLY_TUPLE, "--cached", patch_path]):
            git(*argv, max_address_byte=gitcmd.MAX_PATCH_ADDRESS_BYTE)
    # == Step 3: the commit ===================================
    tree = git("write-tree").strip()
    more_list = []
    if parent_list:
        tip_set = set(git("merge-base", "--independent", head, *parent_list).split())
        for parent in parent_list:
            parent = git("rev-parse", "--verify", f"{parent}^{{commit}}").strip()
            if parent in tip_set and parent != head and parent not in more_list:
                more_list.append(parent)
    if patch_path and not more_list and tree == git("rev-parse", "HEAD^{tree}").strip():
        raise RuntimeError(f"git commit failed while taking in {what}: the patch leaves "
                           f"the tree as it was")
    message_text = git("stripspace", input_text=message)
    if not message_text:
        raise RuntimeError(f"git commit failed while taking in {what}: the commit "
                           f"message is empty")
    parent_argv = [arg for parent in (head, *more_list) for arg in ("-p", parent)]
    commit = git("commit-tree", tree, *parent_argv, "-F", "-",
                 input_text=message_text).strip()
    # == Step 4: the branch ===================================
    git("update-ref", "-m", "commit: a patch taken in", "HEAD", commit, head)
    return commit


def run(project: str, output_dir: str, request: str, engine: str = "", image: str = "",
        branch: str = "", build_doc: bool = False, dry_run: bool = False, guide: str = "",
        exclude_list: list[str] | None = None, run_request: str = "", start: str = "",
        with_overview: bool = False, land: bool = False, check: bool = False,
        request_note: str = "") -> dict:
    """Make one change inside a container, take the patch in, and put it in the tree.

    Steps:
    1. Work out where the run starts: the commit it was told, or the commit the
       working tree makes (tree_branch)
    2. Make the scratch directory this run's clone goes in, and start the container over
       a read-only copy of the repository's history
    3. Read the patch it wrote, take it in on the branch, and put it into the working
       tree when asked

    The commit the patch lands in carries what the run recorded about itself as git
    trailers.

    Everything this run writes for itself is named after the run (run_file_layout).

    Args:
        project: The target repository. A relative path is resolved from here.
        output_dir: Where the index and the patch live. A relative path is resolved
            from here.
        request: The change request.
        engine: The container engine. Empty picks one.
        image: The image to run. Empty uses config.CONTAINER_IMAGE.
        branch: The branch a taken-in patch is committed to. Empty names a new one
            after the run itself.
        start: The commit the container starts at and the patch lands on, which the
            caller has made, with a branch reaching it. Empty makes them here.
        build_doc: Whether the analysis made here generates the design documents too:
            the first one when nothing is built in the output directory yet
            (analyse_if_none), or the one already there made again (_update_doc).
        dry_run: Whether the editing tools inside the container write anything. A
            dry run writes the patch file, takes nothing in on a branch and changes
            nothing in the repository.
        guide: The file saying how the repository is to be changed.
        exclude_list: The names this run was given, as exclude_list_of_run() takes them.
        run_request: What the run this is one agent of was given, which the commit
            carries as a trailer. Empty writes the request above there.
        request_note: What the agent is given in front of the request. The commit
            carries the request alone.
        with_overview: Whether the overview is made first when it is missing or not
            current (make_overview), in a container of its own, before the agent's
            copy of the output directory.
        land: Whether what landed on the branch is then put into the working tree
            (land_in_tree), when the run got through, and the branch taken away once the
            change is in.
        check: Whether the run inside holds its change to the static check and the
            references when it ends (edit_argv).

    Returns:
        {"engine", "image", "base", "verdict" (what history.judge() made of the run,
        "" when it recorded nothing), "patch" (the path, or None when nothing
        changed), "files" (how many the run changed), "exit_status" (the run's own),
        "branch" and "commit" (both None when the patch was not taken in), "no_patch",
        "no_patch_why" and "unseen_text" (as _land_patch() returns them), "land"
        (what was put into the working tree, as land_in_tree() returns it, or None),
        "land_why" (why nothing was, or "": with land, also for a run with a non-zero
        exit status whose patch holds nothing), "land_command" and "land_refuse" (as
        try_land() returns them)}.

    Raises:
        RuntimeError: When there is no engine, when project is not a git repository,
            or when the patch does not apply on the branch.
    """
    # == Step 1: read what the container starts from =========
    project_dir, root_out_dir, name = _run_place(project, output_dir)
    run_id = new_run_id()
    on_branch = branch or f"{_BRANCH_PREFIX}{run_id}"
    # A dry run stops before anything is taken in
    take_in = not dry_run
    if dry_run:
        logger.info("this is a dry run, so the patch is written and nothing is taken in")
    # The agent works in a copy of its own (make_agent_dir)
    agent = agent_name(branch, run_id)
    exclude_list = exclude_list_of_run(project_dir, root_out_dir, name, exclude_list,
                                       build_doc, doc_update=True, engine=engine)
    with contextlib.ExitStack() as stack:
        # A run told where to start has a branch pointing at that commit already. One
        # that is not starts from the working tree, and points the branch at it here
        base = start or stack.enter_context(
            tree_branch(project_dir, run_id, engine=engine))[0]
        # The agent's copy takes the overview made here
        if with_overview and overview.is_on():
            with progress.step("make the overview"):
                make_overview(project_dir, root_out_dir, name, exclude_list, base,
                              engine, image)
        out_dir = make_agent_dir(root_out_dir, name, agent, project_dir, exclude_list,
                                 build_doc)
        run_file = run_file_layout(root_out_dir, agent, name, run_id)
        history_path = os.path.join(out_dir, name, history.HISTORY_FILE)
        try:
            record_before_count = _last_attempt(history_path, out_dir).get("n", 0)
            # == Step 2: start the broker, then the container ========
            engine_path, image_name, status = _start_container(
                engine, image, project_dir, root_out_dir, out_dir, name, agent, base,
                run_file, run_id,
                edit_argv(name, request_note + request, dry_run, guide, check),
                exclude_list)

            # == Step 3: take the patch in ===========================
            result_dict = {"engine": engine_path, "image": image_name, "base": base,
                           "land": None, "land_why": "", "land_command": "",
                           "land_refuse": False}
            # Read once: it is written into the commit as trailers and handed back as
            # the verdict
            attempt_dict = _own_record_dict(history_path, out_dir, record_before_count)
            result_dict["verdict"] = attempt_dict.get("verdict", "")
            result_dict.update(_land_patch(
                project_dir, run_file, base, on_branch, status, take_in,
                lambda meta_dict: commit_message(
                    request, run_trailer_dict(meta_dict, attempt_dict,
                                              run_request or request)),
                engine=engine))
            if land and result_dict["commit"]:
                result_dict.update(_land_run(project_dir, base, result_dict, run_file.land,
                                             last_land_path(root_out_dir, name)))
            elif land and result_dict["exit_status"] and not result_dict["patch"] \
                    and not result_dict.get("no_patch"):
                # A run that did not get through and left a patch that holds nothing
                result_dict["land_why"] = (
                    f"the run {_empty_patch_text(result_dict['exit_status'])}, so nothing "
                    f"was put into the working tree")
        finally:
            drop_the_copy(root_out_dir, name, agent)
    return result_dict


# What the "git diff" a user is given to read a branch with starts with, "diff"
# included: no external diff program and no text conversion of the git settings
_READ_DIFF_TUPLE = ("diff", "--no-textconv", "--no-ext-diff")


def _land_run(project_dir: str, base: str, result_dict: dict, patch_path: str,
              last_path: str) -> dict:
    """Put what one run landed on its branch into the working tree, when it got through.

    A run that did not get through - stopped early, used every turn, ended with a check
    or the program still failing - still landed what it had changed by then, and that
    stays on the branch for reading and does not go into the working tree. One that got
    through has its change put in when it applies (try_land), and the branch, a name
    this tool made up, is then taken away.

    Args:
        project_dir: The target repository, as an absolute path.
        base: The commit the run started from.
        result_dict: run()'s return value so far, with "branch", "commit" and
            "exit_status".
        patch_path: Where the diff put into the working tree is written.
        last_path: Where the copy of an applied diff is kept (last_land_path()).

    Returns:
        {"land", "land_why", "land_command", "land_refuse"}, as try_land() returns
        them.
    """
    branch, commit = result_dict["branch"], result_dict["commit"]
    if result_dict["exit_status"]:
        why = (f"the run ended with {status_text(result_dict['exit_status'])}, so what "
               f"it changed stays on {branch} ({gitcmd.short(commit)}) and the working "
               f"tree is as it was. Read it with:")
        command = command_line("git", "-C", project_dir, *_READ_DIFF_TUPLE,
                               gitcmd.short(base), branch)
        logger.warning("%s %s", why, command)
        return {"land": None, "land_why": why, "land_command": command,
                "land_refuse": False}
    land_dict = try_land(project_dir, base, commit, patch_path, last_path)
    if land_dict["land"]:
        drop_branch_under(project_dir, [branch], commit)
    return land_dict


def _start_container(engine: str, image: str, project_dir: str, root_out_dir: str,
                     out_dir: str, name: str, agent: str, base: str,
                     run_file: RunFileLayout, run_id: str, argv_tail: list[str],
                     exclude_list: list[str] | None,
                     limit: int | None = None,
                     tip_list: Sequence[str] = (),
                     primary_ext: str = "") -> tuple[str, str, int]:
    """Start the broker, then the container, and wait for the container to finish.

    No container is started once the token counts of this process have reached
    MAX_TOKEN_PER_RUN (tokencount.limit_text). The broker is told how many tokens the
    processes this one starts may use between them (tokencount.remain_token), and it
    counts the model calls of the brokers running beside it against that as well as its
    own. The counts of the broker's model calls are added to this process's when the
    container has ended.

    The run is recorded on this machine's side of the agent (host_dir) as one this
    machine started (_write_run_mark). The clones that runs which ended left on this
    machine's side of every agent are swept first, each under its own lock
    (remove_dead_clone). This run's clone directory is then made there under the agent's
    lock, and the copy of the repository's history the container clones is made in it
    (make_clone_source). The container is given the agent's own copy of the output
    directory, the "src", "work" and "tmp" directories of that clone directory, the
    package cache and the broker's socket directory.

    Args:
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        project_dir: The target repository.
        root_out_dir: The output directory as it was passed, which holds the cache.
        out_dir: This agent's own place inside it.
        name: The target repository's directory name.
        agent: The agent's name, for the lock and the messages.
        base: The commit the clone is checked out at.
        run_file: This run's paths.
        run_id: This run's identifier.
        argv_tail: What the tool is run with inside the container.
        exclude_list: The extra names kept out of the analysis.
        limit: The seconds the container may run for, 0 or less for no limit. None
            is SANDBOX_TIMEOUT_SEC.
        tip_list: The commits the container combines onto base, whose files the
            package cache is filled for as well (new_test_ext_list).
        primary_ext: The language the run takes as the primary one (build_run_argv).

    Returns:
        (the engine's path, the image's name, the container's exit status).

    Raises:
        RuntimeError: When no engine is on PATH, when this process lacks what a model
            call needs (llm.missing_credential_text), when MAX_TOKEN_PER_RUN is reached,
            when the copy of the history cannot be made, when the directory of the run's
            files cannot be made in the agent's copy of the output directory, or when
            that copy is reached through a link that leads out of it (inside_path); and
            as make_cache(), fill_cache_once(), start_tree() and start_broker() raise.
        OSError: When the agent's directory on this machine (host_dir) or the clone
            directory cannot be made.
    """
    engine_path = pick_engine(engine)
    image_name = image or config.CONTAINER_IMAGE
    # The credentials of this process, which the broker runs with, are checked before
    # anything is made or started
    lack_text = llm.missing_credential_text()
    if lack_text:
        raise RuntimeError(f"the broker would start without what a model call needs:\n"
                           f"{lack_text}")
    over_text = tokencount.limit_text()
    if over_text:
        raise RuntimeError(f"{over_text}, so no further container is started")
    try:
        os.makedirs(inside_path(run_file.directory, run_file.out_dir), exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"{agent}'s copy of the output directory holds what a run cannot be started "
            f"in at {run_file.directory} ({e.strerror or e}). Remove it by hand, then "
            f"run again") from None
    os.makedirs(run_file.host_directory, exist_ok=True)
    _write_run_mark(run_file, "")
    remove_dead_clone(root_out_dir, name, engine_path)
    # The cache is not in the agent's copy (cache_dir). A cache nothing has filled yet
    # is filled now, in a container with a network and no key of this tool. The index is
    # read after the filling, whose run-tests may make the analysis in the agent's copy
    # again
    cache = make_cache(cache_dir(root_out_dir, name))
    fill_ext_list = new_test_ext_list(project_dir, root_out_dir, base, tip_list)
    refill_count = fill_cache_once(project_dir, root_out_dir, agent, cache, engine_path,
                                   image_name, exclude_list, base, fill_ext_list)
    with filelock.hold_lock(run_file.host_directory, f"the runs of {agent}"):
        index_old = not index_describe(project_dir, out_dir, name, base)
        scratch = make_scratch(run_file.scratch)
    try:
        source = make_clone_source(project_dir, scratch)
        # The socket is in a short path of its own, outside the output directory
        sock_dir = tempfile.mkdtemp(prefix="ctloom-broker-")
        token_path = ""
        try:
            token_path = tokencount.child_file()
            argv = build_run_argv(
                engine_path, image_name,
                MountLayout(src=source, out=out_dir, scratch=scratch, cache=cache,
                            sock=sock_dir),
                base, name, argv_tail, run_id=run_id, rebuild_index=index_old,
                exclude_list=exclude_list,
                build_output_tuple=build_output_tuple_of(root_out_dir, name),
                primary_ext=primary_ext, host_project=project_dir)
            if index_old:
                logger.info("the index does not describe %s, so the container rebuilds "
                            "it first (no LLM is called)", gitcmd.short(base))
            logger.info("starting %s over a read-only copy of the history of %s at %s, "
                        "with no network",
                        os.path.basename(engine_path), project_dir, gitcmd.short(base))
            broker_process = start_broker(
                sock_dir, project_dir, root_out_dir, run_id, cache, engine_path,
                image_name, token_out=token_path, token_left=tokencount.remain_token(),
                agent=agent,
                max_refill=max(0, config.WARM_MAX_REFILL - refill_count),
                base=start_tree(project_dir), fill_ext_list=fill_ext_list)
            try:
                status = run_the_container(
                    argv, engine_path, run_id,
                    config.SANDBOX_TIMEOUT_SEC if limit is None else limit,
                    run_file.stdout)
            finally:
                stop_broker(broker_process, sock_dir)
        finally:
            _remove_tree(sock_dir, "the broker's socket directory")
            _take_token_file(token_path, f"{argv_tail[0]} {run_id}")
    finally:
        remove_scratch(scratch)
    return engine_path, image_name, status


def _land_patch(project_dir: str, run_file: RunFileLayout, base: str, branch: str,
                status: int, take_in: bool, message_of: Callable[[dict], str],
                parent_list: Sequence[str] = (), reset: bool = False,
                keep_empty: bool = False, engine: str = "") -> dict:
    """Read the patch the container wrote and put it on a branch.

    A patch that holds something is recorded on this machine's side as what the run
    left to pick up by hand (_write_run_mark) until a commit holds it.

    Args:
        project_dir: The target repository.
        run_file: This run's paths, which say where the patch is.
        base: The commit the branch starts at.
        branch: The branch the commit lands on.
        status: The container's exit status. The exit status the patch's own summary
            names stands in its place, unless that one is 0 and the container's is
            not (_hold_meta_number).
        take_in: Whether the patch is committed at all.
        message_of: (the patch's summary) -> the commit message.
        parent_list: The commit's parents besides base, as take_in_patch() takes them.
        reset: Whether an existing branch is moved to base first, as take_in_patch() takes
            it.
        keep_empty: Whether a patch the container wrote that holds nothing is
            committed too, as a commit that changes nothing and carries the message.
        engine: The container engine the run uses, as take_in_patch() takes it.

    Returns:
        {"patch", "files", "exit_status", "branch", "commit", "no_patch",
        "no_patch_why", "unseen_text"}, "branch" and "commit" None when nothing was
        taken in. The container left one of three things: no patch ("no_patch" True,
        "patch" None), a patch that holds nothing ("no_patch" False, "patch" None) or a
        patch with a change in it ("patch" its path). No patch is a patch that is not
        there as a regular file, a summary beside the patch that _read_meta() returns {}
        for, and a patch of more than _MAX_PATCH_BYTE bytes: nothing is taken in, and
        "exit_status" is the container's, or 1 where that is 0. "files" and
        "exit_status" are otherwise the summary's where it holds one this machine takes
        (_hold_meta_number). With take_in, a summary that names a directory under
        "unseen" (_warn_patch_gap) is no patch too, its "exit_status" 1 where the run's
        is 0. "no_patch_why" says why there is no patch, with the container's own exit
        status where it ended before it wrote one, and where a patch it wrote stays; it
        is "" when there is one. "unseen_text" names the directories under "unseen" and
        what is in them, whether or not the patch is taken in, and is "" when the
        summary names none.

    Raises:
        RuntimeError: When the patch does not apply, or when the patch or its summary
            is reached through a link that leads out of the agent's copy of the
            output directory (inside_path).
    """
    patch_path = inside_path(run_file.patch, run_file.out_dir)
    result_dict = {"patch": None, "files": 0, "exit_status": status,
                   "branch": None, "commit": None, "no_patch": False, "no_patch_why": "",
                   "unseen_text": ""}
    # A patch that holds something is left to pick up by hand until a commit holds it
    has_patch = os.path.isfile(patch_path) and os.path.getsize(patch_path) > 0
    if has_patch:
        _write_run_mark(run_file, _PATCH_KEEP)
    meta_dict = _read_meta(patch_path, run_file.out_dir)
    # The summary is written after the patch: a run that left none did not finish
    # writing its patch
    if not meta_dict or not os.path.isfile(patch_path):
        return _no_patch_dict(
            result_dict, status or 1,
            f"the container ended ({status_text(status)}) before it wrote a patch and the "
            f"summary beside it"
            + (f". What it wrote of the patch stays at {patch_path}"
               if has_patch else ""))
    if os.path.getsize(patch_path) > _MAX_PATCH_BYTE:
        return _no_patch_dict(
            result_dict, status or 1,
            f"the patch the run wrote holds {os.path.getsize(patch_path)} bytes, more "
            f"than the {_MAX_PATCH_BYTE} that are taken in. It stays at {patch_path}")
    _hold_meta_number(meta_dict, status)
    result_dict.update({key: meta_dict[key] for key in ("files", "exit_status")
                        if key in meta_dict})
    _warn_patch_gap(meta_dict)
    unseen_list = _meta_path_list(meta_dict, "unseen")
    if unseen_list:
        result_dict["unseen_text"] = (
            f"what the run wrote into {len(unseen_list)} submodule directory(ies) of "
            f"its clone that git shows as their commit alone: {list_text(unseen_list)}")
    # Files the run wrote that are in no patch keep the whole of what it changed out
    if take_in and unseen_list:
        return _no_patch_dict(
            result_dict, result_dict["exit_status"] or 1,
            f"the patch does not hold {result_dict['unseen_text']}"
            + (f". It stays at {patch_path}" if has_patch else
               ", and it holds nothing else"))
    if os.path.getsize(patch_path) == 0:
        logger.info("the run changed nothing, so there is no patch to take in")
        if take_in and keep_empty:
            result_dict.update(take_in_patch(
                project_dir, "", base, branch, run_file.worktree, message_of(meta_dict),
                parent_list, reset, engine))
        return result_dict
    result_dict["patch"] = patch_path
    if not take_in:
        return result_dict
    commit_dict = take_in_patch(project_dir, patch_path, base, branch, run_file.worktree,
                                message_of(meta_dict), parent_list, reset, engine)
    _write_land_meta(patch_path, run_file.out_dir, commit_dict)
    if commit_dict.get("commit"):
        _write_run_mark(run_file, "")
    result_dict.update(commit_dict)
    return result_dict


def _no_patch_dict(result_dict: dict, exit_status: int, why: str) -> dict:
    """Return _land_patch()'s result for a container that left no patch taken in.

    why is logged as a warning.

    Args:
        result_dict: The result built so far.
        exit_status: The exit status the run is reported with.
        why: Why there is no patch, and where a patch it wrote stays.

    Returns:
        result_dict with "no_patch" True, "exit_status" and "no_patch_why" why.
    """
    logger.warning("nothing of what the run changed is taken in: %s", why)
    return {**result_dict, "no_patch": True, "exit_status": exit_status,
            "no_patch_why": why}


def status_text(status: int) -> str:
    """Return a container's exit status as the reasons this machine gives name it.

    Examples:
        124 -> "exit status 124, the status of a container taken away at
               SANDBOX_TIMEOUT_SEC (3600 s)"
        137 -> "exit status 137, the status of a container killed when its processes
               reach SANDBOX_MEMORY (4g)"
        1   -> "exit status 1"

    Args:
        status: The exit status run_the_container() returned.

    Returns:
        The status, followed by what it is the status of where it is _TIME_OUT_STATUS
        with a SANDBOX_TIMEOUT_SEC set, or _KILL_STATUS.
    """
    if status == _TIME_OUT_STATUS and config.SANDBOX_TIMEOUT_SEC > 0:
        return (f"exit status {status}, the status of a container taken away at "
                f"SANDBOX_TIMEOUT_SEC ({config.SANDBOX_TIMEOUT_SEC} s)")
    if status == _KILL_STATUS:
        return (f"exit status {status}, the status of a container killed when its "
                f"processes reach SANDBOX_MEMORY ({config.SANDBOX_MEMORY})")
    return f"exit status {status}"


class ContainerEndError(RuntimeError):
    """A container that ended with a non-zero exit status and left nothing to read.

    Attributes:
        exit_status: The container's exit status.
        error: What reading the container's file raised.
    """

    def __init__(self, error: RuntimeError, status: int) -> None:
        """Hold what reading failed with, after how the container ended.

        Args:
            error: What reading the container's file raised.
            status: The container's exit status.
        """
        super().__init__(f"the container ended ({status_text(status)}), and {error}")
        self.exit_status = status
        self.error = error


def _read_after_end(read: Callable[[], object], status: int) -> object:
    """Read what a container wrote, saying how it ended when that cannot be read.

    Args:
        read: Reads the file and returns what it holds, raising RuntimeError when it
            cannot.
        status: The container's exit status.

    Returns:
        What read returns.

    Raises:
        ContainerEndError: When read raises RuntimeError and status is not 0.
        RuntimeError: As read raises it, when status is 0.
    """
    try:
        return read()
    except RuntimeError as e:
        if not status:
            raise
        raise ContainerEndError(e, status) from None


def _hold_meta_number(meta_dict: dict, status: int) -> None:
    """Hold the numbers of a patch's summary to the ones the container's entrypoint writes.

    "exit_status" is kept when it is a whole number from 0 to _MAX_EXIT_STATUS; any
    other value is replaced by the container's own exit status, or by 1 where that is 0.
    A 0 beside a container that ended with another status is replaced by that status.
    "files" is kept when it is a whole number of 0 or more, and taken out otherwise. A
    summary without one of the two is left without it.

    Args:
        meta_dict: The summary beside the patch, as _read_meta() returns it. It is
            changed in place.
        status: The container's exit status.
    """
    exit_status = meta_dict.get("exit_status")
    if "exit_status" in meta_dict and not (
            type(exit_status) is int and 0 <= exit_status <= _MAX_EXIT_STATUS):
        logger.warning("the summary beside the patch names %s as the run's exit "
                       "status, which no process ends with, so the run is read as "
                       "one that did not get through", one_line(repr(exit_status))[:60])
        meta_dict["exit_status"] = status or 1
    elif exit_status == 0 and status:
        logger.warning("the summary beside the patch names 0 as the run's exit status "
                       "and its container ended with %d, so the run is read as one "
                       "that ended with %d", status, status)
        meta_dict["exit_status"] = status
    file_count = meta_dict.get("files")
    if "files" in meta_dict and not (type(file_count) is int and file_count >= 0):
        del meta_dict["files"]


# The keys of a patch's summary that name what the patch does not hold, each with what
# the warning says of the paths under it
_PATCH_GAP_DICT = {
    "unadded": "git could not add %d path(s) of the run's clone, so they are not in "
               "the patch: %s",
    "nested": "the run left %d repository(ies) of their own inside its clone, and the "
              "files in them are not in the patch: %s",
    "unseen": "the run wrote files into %d submodule directory(ies) of its clone that "
              "git shows as their commit alone, and they are not in the patch: %s"}


def _meta_path_list(meta_dict: dict, key: str) -> list[str]:
    """Return the paths a patch's summary lists under a key, the strings alone.

    Args:
        meta_dict: The summary beside the patch, as _read_meta() returns it.
        key: One of _PATCH_GAP_DICT.

    Returns:
        The strings of the list under key, or [] when there is no list there.
    """
    value = meta_dict.get(key)
    return [path for path in value if isinstance(path, str)] \
        if isinstance(value, list) else []


def _warn_patch_gap(meta_dict: dict) -> None:
    """Log what the container said its patch does not hold.

    Args:
        meta_dict: The summary beside the patch, as _read_meta() returns it. Its
            "unadded", "nested" and "unseen" are lists of paths as git shows them,
            and its "moved_head" the commit HEAD was at when the run ended, written
            when the run moved it off the commit it started at.
    """
    for key, message in _PATCH_GAP_DICT.items():
        path_list = _meta_path_list(meta_dict, key)
        if path_list:
            logger.warning(message, len(path_list), list_text(path_list))
    end_head = meta_dict.get("moved_head")
    if isinstance(end_head, str) and end_head:
        logger.warning("the run committed in its clone (HEAD ended at %s), and what it "
                       "committed is in the patch with the rest of what it changed",
                       gitcmd.short(end_head))


def dispatch(project: str, output_dir: str, request: str, engine: str = "",
             image: str = "", build_doc: bool = False,
             resume_list: Sequence[str] = (),
             exclude_list: list[str] | None = None) -> dict:
    """Read one request, run an agent per question or per task, and combine what they did.

    One dispatching run is _dispatch_once(). A run that did not get through, which is
    a non-zero exit status, is run again up to DISPATCH_MAX_RERUN times more, each
    time as a whole run of its own that carries on from the branches the run before it
    left, named as the run prints them for --resume. What the last run returned is
    what is returned. A question whose combining run handed nothing back while some
    agent answered is not run again, and its agents' answers are returned as they came.
    A request the reading run handed back for the person who asked to settle is not
    run again either, nor is one whose union was kept out of the working tree for the
    paths it touches (try_land's "land_refuse"), one whose run of the tests over HEAD
    keeps any merge from being settled (integrate()'s "head_block") or one that ended
    with exit status _KILL_STATUS, and nothing is run again once MAX_TOKEN_PER_RUN is
    reached.

    Args:
        As _dispatch_once() takes them. resume_list names the branches the first run is
        told about; every run after it is told the branches the run before it left, or,
        when that run left none, the branches that run was told about. The branches
        resume_list names are handed to every run as keep_list, and their runs are
        recorded as this command's before any of them is read (_mark_run_of_branch).
        build_doc is handed to the first run only.

    Returns:
        _dispatch_once()'s return value for the last run, with "runs": how many runs
        were made, counting from 1, and "earlier_land": what each run before the last
        put into the working tree (land_in_tree()'s return value), oldest first, a run
        that put nothing in left out.

        The cases the tests of a run before the last found not done stay with the result
        where no run after it put a union into the working tree (_open_after_run):
        "earlier_gap" holds them ([] otherwise), "resume" names the union they are
        recorded on ahead of the last run's own branches, and "exit_status" is non-zero
        (_hold_to_open_gap). The cases recorded on the unions resume_list names that
        this request carries on from (gap_union_list_of) are counted the same way, as
        those of a run before the first.

    Raises:
        RuntimeError: As _dispatch_once() raises it, from the first run or from a run
            made again.
    """
    once = functools.partial(_dispatch_once, project, output_dir, request,
                             engine=engine, image=image, exclude_list=exclude_list,
                             keep_list=resume_list)
    run_resume_list = list(resume_list)
    project_dir, root_out_dir, name = _run_place(project, output_dir)
    # Recorded before they are read
    run_id = new_run_id()
    _mark_run_of_branch(project_dir, root_out_dir, name, resume_list, run_id)
    # The cases recorded on the unions this request carries on from, read before any
    # run changes what the working tree holds
    gap_union_list = gap_union_list_of(project_dir, resume_list, request)
    _hold_stand_for(project_dir, root_out_dir, name, engine, resume_list, gap_union_list,
                    request)
    start_dict = {"gap": [entry_dict for branch in gap_union_list
                          for entry_dict in integrate_mod.branch_record_dict(
                              project_dir, branch)["gap"]],
                  "resume": gap_union_list}
    result_dict = once(resume_list=run_resume_list, build_doc=build_doc, run_id=run_id)
    run_count = 1
    earlier_land_list = []
    # The cases the tests of the runs so far found not done in what the working tree
    # holds, and the union they are recorded on
    open_dict = _open_after_run(start_dict, result_dict)
    result_dict = _hold_to_open_gap(result_dict, open_dict)
    while result_dict["exit_status"] and run_count <= config.DISPATCH_MAX_RERUN:
        # A request handed back for the person who asked to settle is not run again
        if dispatch_mod.UNCLEAR_KEY in result_dict:
            break
        # A question some agent answered is not asked again: the run hands back those
        # answers
        union_dict = result_dict.get("combined") or {}
        is_combine_failure = "answers" in result_dict and (
            not union_dict.get("answer") or bool(union_dict.get("error")))
        if is_combine_failure and any(not entry_dict["exit_status"]
                                      for entry_dict in result_dict["answers"]):
            logger.warning("the combining run handed nothing back, and the request is "
                           "not run again: the agents' answers are given as they came")
            break
        # Nothing is run again once MAX_TOKEN_PER_RUN is reached
        if tokencount.limit_text():
            logger.warning("%s, so the request is not run again", tokencount.limit_text())
            break
        # A union kept out of the working tree for the paths it touches is not made
        # again
        if (result_dict.get("integration") or {}).get("land_refuse"):
            logger.warning("the change is kept out of the working tree for the paths "
                           "it touches, so the request is not run again")
            break
        # Nor is a union whose run of the tests over HEAD keeps any merge from being
        # settled
        if (result_dict.get("integration") or {}).get("head_block"):
            logger.warning("no merge over this HEAD is settled while its tests run as "
                           "they do, so the request is not run again")
            break
        # Nor is a run a container of which was killed at SANDBOX_MEMORY
        if _is_memory_kill(result_dict):
            logger.warning("the run did not get through (%s), so the request is not run "
                           "again; raise SANDBOX_MEMORY to run it",
                           status_text(_KILL_STATUS))
            break
        carry_list = result_dict.get("resume") or run_resume_list
        run_count += 1
        logger.warning("the run did not get through (exit status %d), so the request "
                       "is run again%s (run %d of %d)", result_dict["exit_status"],
                       f", carrying on from {', '.join(carry_list)}" if carry_list else "",
                       run_count, config.DISPATCH_MAX_RERUN + 1)
        run_resume_list = carry_list
        # Recorded before the run starts, for a command stopped while it goes on
        _RERUN_LIST.append((next(_RUN_COUNT), list(run_resume_list), list(open_dict["resume"]),
                            len(_LAND_LIST)))
        next_dict = once(resume_list=run_resume_list)
        # What the run before put into the working tree is handed back with the result
        # of the last run
        land_dict = (result_dict.get("integration") or {}).get("land") or {}
        earlier_land_list += [land_dict] if land_dict.get("files") else []
        open_dict = _open_after_run(open_dict, next_dict)
        result_dict = _hold_to_open_gap(next_dict, open_dict)
    earlier_gap_list = [] if _has_union_in_tree(result_dict) \
        or "questions" in result_dict else open_dict["gap"]
    return {**result_dict, "runs": run_count, "earlier_land": earlier_land_list,
            "earlier_gap": earlier_gap_list}


def _has_union_in_tree(run_dict: dict) -> bool:
    """Say whether one dispatching run put its union into the working tree.

    Args:
        run_dict: _dispatch_once()'s return value.

    Returns:
        True when the run combined its branches and what it settled was put in.
    """
    return bool((run_dict.get("integration") or {}).get("land"))


def _hold_to_open_gap(run_dict: dict, open_dict: dict) -> dict:
    """Return one dispatching run's result, held to the cases left not done before it.

    The cases keep the request from having got through, whatever a run that put no
    union into the working tree ended with. A run that answered a question is
    returned as it is.

    Args:
        run_dict: _dispatch_once()'s return value.
        open_dict: _open_after_run()'s return value for the runs up to that one.

    Returns:
        run_dict as it is when no case is left, when the run put its union in, or
        when it answered a question. Otherwise run_dict with "exit_status" non-zero
        and "resume" naming the unions the cases are recorded on ahead of its own.
    """
    if not open_dict["gap"] or _has_union_in_tree(run_dict) or "questions" in run_dict:
        return run_dict
    return {**run_dict, "exit_status": run_dict["exit_status"] or 1,
            "resume": list(dict.fromkeys(
                [*open_dict["resume"], *(run_dict.get("resume") or [])]))}


def _open_after_run(open_dict: dict, run_dict: dict) -> dict:
    """Return the cases the tests found not done in what the working tree holds.

    run_dict is the run that ended. A run that put its union into the working tree
    stands for the runs before it: the cases are the ones its own tests found. A run
    that put none in leaves them as they were.

    Args:
        open_dict: {"gap": the cases as they were before the run that ended, "resume":
            the branches naming the union they are recorded on}.
        run_dict: _dispatch_once()'s return value for the run that ended.

    Returns:
        The same two keys, for the runs up to that one. "resume" is empty where
        "gap" is.
    """
    if not _has_union_in_tree(run_dict):
        return open_dict
    gap_list = list(run_dict["integration"].get("gap") or [])
    return {"gap": gap_list,
            "resume": list(run_dict.get("resume") or []) if gap_list else []}


def _dispatch_once(project: str, output_dir: str, request: str, engine: str = "",
                   image: str = "", build_doc: bool = False,
                   resume_list: Sequence[str] = (),
                   exclude_list: list[str] | None = None,
                   keep_list: Sequence[str] = (), run_id: str = "") -> dict:
    """Make one dispatching run: read the request, run the agents, combine, sweep.

    Steps:
    1. Start a container that reads the repository and hands back one split: the
       questions the agents are to answer, or the change requests they are given. The
       run starts from the commit the working tree makes (tree_branch): what is on disk
       is what is read and changed, committed or not. When it hands back what the
       request leaves open instead, the run ends there with no agent started (step 5 is
       still made)
    2. For questions, start one answering agent per question, at most DISPATCH_PARALLEL
       at once, hand a question whose agent answered nothing out again, then start the
       combining run over every answer and hand its answer back
    3. For tasks, start one editing agent per task, at most DISPATCH_PARALLEL at once,
       each on a branch of its own, and hand a task whose agent did not finish out again
    4. Combine the branches that ended up holding something, when every task finished.
       One branch alone is combined too, and held to the merging run's gates. A union
       the gates passed is put into the working tree (integrate with land). With
       MAX_TOKEN_PER_RUN reached by then, nothing is combined and the run ends with a
       non-zero status
    5. Take away what every run but the newest DISPATCH_KEEP_RUN left behind: the
       branches and the output of the runs that did not finish

    The reading run, the answering agents and the combining run are built the same way
    an editing agent is - one broker, one container, one place in the output directory
    of its own - and are asked to read, not to change. What this machine reads back from
    each is one file: the split, or the answer.

    The agents are not told which files to touch. Two of them may write the same file,
    and the combining run afterwards combines what they wrote. The tasks whose text
    names the same definition (task_chain_list) run one after another, each starting at
    what the one before it left, and the last branch of such a chain is what is
    combined.

    A run this one is carrying on from is named by its branches. They are read, never
    checked out: what the commits there say - what each agent was asked for, what it
    said it did and how the machine judged it - goes in front of the request the reading
    run is given.

    A run that stops part-way still writes what it had changed by then and that patch
    still lands on its branch. The exit status follows history.judge(): a run that
    stopped early, used every turn without COMPLETE() or ended with a check or the
    program still failing comes back non-zero, and a task whose agent did not finish is
    handed out again, up to DISPATCH_MAX_RETRY times. One still unfinished after that
    leaves nothing combined at all. The branches stay where they are, with what each
    agent did get to. A run that got through leaves nothing to read: the branch of a try
    that was done again is taken away with the rest.

    Args:
        project: The target repository. A relative path is resolved from here.
        output_dir: Where the index and the patches live.
        request: The request, as a question or as a change.
        engine: The container engine. Empty picks one.
        image: The image to run. Empty uses config.CONTAINER_IMAGE.
        build_doc: Whether the analysis made before the reading run generates the design
            documents too: the first one when nothing is built in the output directory
            yet, or the one already there made again (exclude_list_of_run with
            doc_update). A run that carries on from a settled merge makes no analysis
            again. Without it, and with a terminal, the first analysis asks
            (analyse_if_none with ask_doc).
        resume_list: The branches an earlier run of this left its work on, which the
            reading run is told about. They are checked before anything is started.
        exclude_list: The directory and file names kept out of the analysis.
        keep_list: More branches the sweep leaves alone, with the runs they belong to,
            besides those of resume_list.
        run_id: The run, as new_run_id() names it. Empty names a new one.

    Returns:
        {"questions": [str, ...], "answers": [{"n", "request", "attempt", "answer",
        "exit_status", "error"}, ...], "combined": {"answer", "exit_status", "error"}
        (what the combining run made of them), "answer": the one answer to hand back,
        which is the combining run's, or the agents' answers as they came when that run
        handed nothing back, "combine_runs": how many combining runs were made over the
        same answers, counting from 1, "swept" and "exit_status"} when the request was a
        question,
        or {"tasks": [str, ...], "branches": [{"n", "request", "branch", "attempt",
        "commit", "exit_status", "error", "verdict"}, ...], "integration": the return
        value of integrate() for the last merge, or None when nothing landed or a task
        did not finish, "merge_runs": how many merges were made, counting from 1, and 0
        when none was, "dropped": the branches this run made that were taken away,
        because the merge reaches them or because the try they held was done again,
        the unions the merges before the last left included, the unions a run that
        put a union of its own into the working tree was carrying on from for their
        cases, or that union stands for (_hold_stand_for, "stand_drop" of
        land_in_tree()), and the branches a run that got through was carrying on
        from that hold a result the working tree holds (_drop_resume_branch_list),
        "resume": the branches to name to carry on from what this run left
        (resume_branch_list), empty when it got through and left nothing to carry on
        from, "token_stop": what tokencount.limit_text() said when it kept the merge
        from starting, "" otherwise, "swept": the older runs whose leftovers were taken
        away (remove_old_run), "exit_status": int}. The exit status is the worst of what
        the tasks and the merge reported, and non-zero as well where a task's tests were
        left red with no merge to read them. The reading run's container counts for
        nothing in it once its split was read.
        {"unclear": [str, ...], "resume": the branches resume_list named, "swept" and
        "exit_status": 1} when the reading run handed the request back for the person
        who asked to settle.

    Raises:
        RuntimeError: When there is no engine, when project is not a git repository,
            when a branch named in resume_list is not there, or when the reading run
            handed no split back: a ContainerEndError, with the container's exit status,
            when its container ended with one that is not 0.
    """
    project_dir, root_out_dir, name = _run_place(project, output_dir)
    run_id = run_id or new_run_id()
    # Named before anything is paid for. The reading run reads them in its clone, which
    # reaches them as the branches of what it cloned
    for branch in resume_list:
        integrate_mod.commit_of(project_dir, branch)
    # One branch holding a settled merge of this request which a stage is still ahead
    # of is carried on from by running that stage, with no task handed out
    if len(resume_list) == 1 \
            and stage_resume_dict(project_dir, resume_list[0]) \
            and one_line(request) == integrate_mod.request_in_branch_list(
                project_dir, resume_list):
        union_dict = integrate(project, output_dir, [], engine=engine, image=image,
                               build_doc=build_doc, request=request,
                               exclude_list=exclude_list, resume=resume_list[0],
                               land=True, ask_doc=True)
        status = union_status(union_dict)
        return {"tasks": [], "branches": [], "integration": union_dict, "merge_runs": 0,
                "dropped": list((union_dict.get("land") or {}).get("stand_drop", [])),
                "swept": [],
                "resume": resume_branch_list([], union_dict, status),
                "exit_status": status}
    with tree_branch(project_dir, run_id, engine=engine) as (base, _):
        return _dispatch_from(project_dir, root_out_dir, name, run_id, base, request,
                              engine, image, build_doc, resume_list, exclude_list,
                              keep_list)


def _dispatch_from(project_dir: str, root_out_dir: str, name: str, run_id: str, base: str,
                   request: str, engine: str, image: str, build_doc: bool,
                   resume_list: Sequence[str], exclude_list: list[str] | None,
                   keep_list: Sequence[str] = ()) -> dict:
    """Make one dispatching run from one commit, as _dispatch_once() says.

    Args:
        project_dir: The target repository, as an absolute path.
        root_out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        run_id: The run, as new_run_id() names it.
        base: The commit the working tree makes, which the run starts from and the union
            is written against.
        request, engine, image, build_doc, resume_list, exclude_list, keep_list: As
            _dispatch_once() takes them.

    Returns:
        What _dispatch_once() returns.

    Raises:
        RuntimeError: As _dispatch_once() raises.
    """
    output_dir = root_out_dir
    # Read before this run puts anything into the working tree
    gap_union_list = gap_union_list_of(project_dir, resume_list, request)
    hold_list = _result_in_tree_list(project_dir, resume_list)
    # The reading run is an agent of its own, named after itself: it commits to no
    # branch
    agent = agent_name("", run_id)
    with progress.step("prepare the analysis"):
        exclude_list = exclude_list_of_run(project_dir, root_out_dir, name, exclude_list,
                                           build_doc, ask_doc=True, doc_update=True,
                                           engine=engine)
        # Asked before any container starts: each is given the settings as they are then
        knowledge.ask_cobol_dialect(store.index_db_path(root_out_dir, name))
        # The agents' copies take the overview made here
        if overview.is_on():
            with progress.step("make the overview"):
                make_overview(project_dir, root_out_dir, name, exclude_list, base,
                              engine, image)
        out_dir = make_agent_dir(root_out_dir, name, agent, project_dir, exclude_list,
                                 build_doc, ask_doc=True)
    run_file = run_file_layout(root_out_dir, agent, name, run_id)

    # == Step 1: read the request in a container =============
    try:
        with progress.step("split the request"):
            engine_path, _, status = _start_container(
                engine, image, project_dir, root_out_dir, out_dir, name, agent, base,
                run_file, run_id,
                dispatch_argv(name, request, run_file.task_inside, resume_list),
                exclude_list)
            split_dict = _read_after_end(
                lambda: _read_split(run_file.task, run_file.out_dir), status)
        # A reading run that handed its split back read the request, whatever its
        # container ended with afterwards
        if status:
            logger.warning("the container ended with exit status %d after it wrote the "
                           "split, and the split is taken", status)
    finally:
        # The reading run is named after itself and changes nothing: what it leaves is
        # the split and the turn log saying how it was arrived at
        drop_the_copy(root_out_dir, name, agent)
    # Called last: what every older run left goes and what this one left stays. This run
    # and the runs it was told to read are named and kept whatever the ordering
    remove_old = functools.partial(remove_old_run, project_dir, root_out_dir,
                                   config.DISPATCH_KEEP_RUN, engine_path,
                                   protect_list=[run_id, *resume_list, *keep_list])
    # A request handed back for the person who asked to settle starts no agent
    if dispatch_mod.UNCLEAR_KEY in split_dict:
        unclear_list = split_dict[dispatch_mod.UNCLEAR_KEY]
        progress.note("the request leaves %d thing(s) open for the person who asked; "
                      "no agent is started", len(unclear_list))
        _log_request_list(unclear_list, "open point")
        return {dispatch_mod.UNCLEAR_KEY: unclear_list, "resume": list(resume_list),
                "swept": remove_old(), "exit_status": 1}
    # == Step 2: questions: an agent each, then combine ======
    if "questions" in split_dict:
        question_list = split_dict["questions"]
        progress.note("the request is %d question(s); starting %d agent(s) at once",
                      len(question_list), min(len(question_list), config.DISPATCH_PARALLEL))
        _log_request_list(question_list, "question")
        answer_list, union_dict, combine_count = _answer_question_list(
            project_dir, output_dir, request, engine, image, exclude_list, base, run_id,
            question_list)
        # A run whose combining run handed an answer back got through, whatever an agent
        # answered
        has_answer = bool(union_dict["answer"]) and not union_dict["error"]
        return {"questions": question_list, "answers": answer_list, "combined": union_dict,
                "answer": union_dict["answer"], "combine_runs": combine_count,
                "swept": remove_old(), "exit_status": 0 if has_answer else 1}
    task_list = split_dict["tasks"]

    # == Step 3: an editing agent per task, in parallel ======
    # The tasks that name the same definition are one chain and run one after another
    chain_list = task_chain_list(split_dict["defs"])
    chain_dict = {chain: new_chain_dict(base) for chain in chain_list if chain}
    progress.note("the request is %d task(s); starting %d agent(s) at once",
                  len(task_list), start_count_of(chain_list, config.DISPATCH_PARALLEL))
    _log_request_list(task_list, "task")
    for chain in sorted(chain_dict):
        progress.note("tasks %s change the same definition, so they run one after another",
                      ", ".join(str(n) for n, task_chain in enumerate(chain_list, 1)
                                if task_chain == chain))
    with progress.step("run the tasks"):
        entry_list = _run_agent_list(task_list, functools.partial(
            _edit_agent, project_dir, output_dir, run_id, engine, image,
            exclude_list, request, base, chain_list, chain_dict, task_list), "task")

    # == Step 4: combine what they left ======================
    # With MAX_TOKEN_PER_RUN reached the merging run is not started, and what every task
    # finished stays on its branch
    is_token_stop = bool(tokencount.limit_text()) \
        and not any(entry_dict["exit_status"] for entry_dict in entry_list) \
        and any(entry_dict["commit"] for entry_dict in entry_list)
    if is_token_stop:
        logger.error("%s, so nothing is combined and the repository is left as it is. "
                     "What each agent did is on its branch: %s", tokencount.limit_text(),
                     ", ".join(entry_dict["branch"] for entry_dict in entry_list
                               if entry_dict["commit"]))
        union_dict, drop_list, merge_count = None, [], 0
    else:
        union_dict, drop_list, merge_count = _combine_task_branch(
            project_dir, output_dir, engine, image, build_doc, request, exclude_list,
            base, entry_list, chain_list, chain_dict)
    # A task whose test was left red is not handed out again. Where no merge was made,
    # nothing read those tests, which is logged
    no_check_list = [entry_dict for entry_dict in entry_list
                     if entry_dict["verdict"] == history.VERDICT_TEST_RED]
    if no_check_list and not union_dict:
        logger.error("%d task(s) ended with a test failing and nothing combined them "
                     "afterwards, so nothing read those tests: %s", len(no_check_list),
                     "; ".join(_task_place_text(entry_dict) for entry_dict in no_check_list))
    worst = max([0] + [entry_dict["exit_status"] for entry_dict in entry_list]
                + ([union_status(union_dict)] if union_dict else [])
                + ([1] if (no_check_list and not union_dict) or is_token_stop else []))
    # A run that got through leaves nothing to carry on from: the branch of a try that
    # was done again, which holds the half a change the next try started over from, goes
    # too
    if not worst:
        drop_list += drop_try_branch_list(
            project_dir, run_id, [entry_dict["branch"] for entry_dict in entry_list])
    # A union this run put in stands for the unions it carried on from, whether or not
    # its own tests found cases not done (_open_after_run). A run that got through
    # stands for the branches it carried on from that hold a result the working tree
    # held before it or holds now
    drop_resume_list = list(gap_union_list) if union_dict and union_dict.get("land") else []
    if not worst and not land_status(union_dict):
        drop_resume_list += [branch for branch in resume_list
                             if branch in hold_list
                             or _result_in_tree_list(project_dir, [branch])]
    drop_list += [*((union_dict or {}).get("land") or {}).get("stand_drop", []),
                  *_drop_resume_branch_list(project_dir, root_out_dir, name, engine,
                                            list(dict.fromkeys(drop_resume_list)))]
    return {"tasks": task_list, "branches": entry_list, "integration": union_dict,
            "merge_runs": merge_count, "dropped": drop_list,
            "resume": resume_branch_list(entry_list, union_dict, worst),
            "token_stop": tokencount.limit_text() if is_token_stop else "",
            "swept": remove_old(), "exit_status": worst}


def is_gap_union(project_dir: str, branch: str) -> bool:
    """Say whether a branch is a union the working tree holds whose tests found cases.

    The cases are ones the change does not do.

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.

    Returns:
        True when the name is one this tool made up (is_own_branch), the branch holds
        a result but for those cases (_is_gap_result) and the working tree holds all
        of what it made (_is_in_tree). False as well for a branch that is not there or
        cannot be read.
    """
    if not is_own_branch(branch) or not branch_tip(project_dir, branch):
        return False
    try:
        return _is_gap_result(project_dir, branch) and _is_in_tree(project_dir, branch)
    except RuntimeError:
        return False


def _is_gap_result(project_dir: str, branch: str) -> bool:
    """Say whether a branch holds a result but for the cases its tests found.

    The cases are ones the change does not do.

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.

    Returns:
        True when integrate.branch_record_dict() reads such cases ("gap": the branch's
        own and those of the earlier unions the working tree holds), the newest commit
        it reads carries integrate.SETTLE_VERDICT and no stage is still ahead
        (_is_stage_ahead).

    Raises:
        RuntimeError: When the branch cannot be read.
    """
    branch_dict = integrate_mod.branch_record_dict(project_dir, branch)
    record_list = branch_dict["told"]
    verdict = record_list[-1]["verdict"] if record_list else ""
    return bool(branch_dict["gap"]
                and verdict == integrate_mod.SETTLE_VERDICT
                and not _is_stage_ahead(project_dir, branch))


def is_union_to_land(project_dir: str, branch: str) -> bool:
    """Say whether a branch is a settled union the working tree does not hold.

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.

    Returns:
        True when the commit under the commits of the branch's stages
        (integrate.below_stage) is a merge a merging run wrote and settled
        (integrate.is_settle_merge), the branch holds a result (_settle_branch_list),
        or one but for the cases its tests found the change does not do
        (_is_gap_result), and the working tree does not hold all of what it made
        (_is_in_tree). False as well for a branch that is not there or cannot be read.
    """
    if not branch_tip(project_dir, branch):
        return False
    try:
        merge = integrate_mod.below_stage(
            project_dir, integrate_mod.commit_of(project_dir, branch))
        return (integrate_mod.is_settle_merge(project_dir, merge)
                and bool(_settle_branch_list(project_dir, [branch])
                         or _is_gap_result(project_dir, branch))
                and not _is_in_tree(project_dir, branch))
    except RuntimeError:
        return False


def gap_union_list_of(project_dir: str, branch_list: Sequence[str], request: str
                      ) -> list[str]:
    """Return the unions a request carries on from for the cases their tests found.

    The cases are ones the tests found not done. A union is one of them when
    is_gap_union() says so and it was made for the same request or carries none
    (_is_request_of).

    Args:
        project_dir: The repository, as an absolute path.
        branch_list: The branches the run was told to carry on from.
        request: The request the run was given.

    Returns:
        Those of branch_list, in the order they were given.
    """
    return [branch for branch in branch_list
            if is_gap_union(project_dir, branch)
            and _is_request_of(project_dir, branch, request)]


def _is_request_of(project_dir: str, branch: str, request: str) -> bool:
    """Say whether a branch was made for a request or carries none.

    Args:
        project_dir: The repository, as an absolute path.
        branch: The branch's name.
        request: The request.

    Returns:
        True when integrate.request_in_branch_list() gives "" or the request on one
        line. False as well for a branch that cannot be read.
    """
    try:
        return integrate_mod.request_in_branch_list(project_dir, [branch]) \
            in ("", one_line(request))
    except RuntimeError:
        return False


def _drop_resume_branch_list(project_dir: str, out_dir: str, name: str, engine: str,
                             branch_list: Sequence[str]) -> list[str]:
    """Delete the branches a run was carrying on from that the working tree stands for.

    None is deleted while another process has a run going over the output directory
    (live_run_list).

    Args:
        project_dir: The repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        engine: The container engine, or "" for the configured one.
        branch_list: The branches to delete.

    Returns:
        The branches that were deleted, in the order they were given.
    """
    if not branch_list:
        return []
    own_tail = f"-{os.getpid()}"
    other_list = [run_name for run_name in live_run_list(out_dir, name, engine)
                  if not run_name.endswith(own_tail)]
    if other_list:
        logger.info("the branch(es) the run carried on from are left where they are "
                    "while other runs are going: %s", ", ".join(other_list))
        return []
    drop_list = [branch for branch in branch_list
                 if branch_tip(project_dir, branch) and _delete_branch(project_dir, branch)]
    if drop_list:
        logger.info("the working tree holds what the run carried on from, so the "
                    "branch(es) it was passed to carry on from were taken away: %s",
                    ", ".join(drop_list))
    return drop_list


def _answer_question_list(project_dir: str, output_dir: str, request: str, engine: str,
                          image: str, exclude_list: list[str] | None, base: str,
                          run_id: str, question_list: list[str]
                          ) -> tuple[list[dict], dict, int]:
    """Answer each question in an agent of its own, then combine the answers.

    The combining run is made again over the same answers, up to INTEGRATE_MAX_RERUN
    times more, while it hands nothing back and MAX_TOKEN_PER_RUN is not reached.

    Args:
        project_dir: The target repository, as an absolute path.
        output_dir: The output directory, as an absolute path.
        request: The request, as the user wrote it.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        exclude_list: The extra names kept out of the analysis.
        base: The commit the agents and the combining run read.
        run_id: The dispatching run, which every agent's output directory is named
            under.
        question_list: The questions the reading run split the request into.

    Returns:
        (one _answer_agent() entry per question, _combine_answer()'s return value for
        the last combining run, how many combining runs were made, counting from 1).
    """
    with progress.step("answer the questions"):
        answer_list = _run_agent_list(question_list, functools.partial(
            _answer_agent, project_dir, output_dir, engine, image, exclude_list,
            base, run_id), "question")
    combine = functools.partial(_combine_answer, project_dir, output_dir, request,
                                answer_list, engine, image, exclude_list, base, run_id)
    union_dict, combine_count = combine(), 1
    # A combining run that handed nothing back is run again over the answers already in
    # hand. The agents are not asked again
    while (not union_dict["answer"] or union_dict["error"]) \
            and combine_count <= config.INTEGRATE_MAX_RERUN \
            and not tokencount.limit_text():
        combine_count += 1
        logger.warning("the combining run handed nothing back, so it is run again "
                       "over the same answers (combine %d of %d)", combine_count,
                       config.INTEGRATE_MAX_RERUN + 1)
        union_dict = combine()
    return answer_list, union_dict, combine_count


def _empty_patch_text(exit_status: int) -> str:
    """Return what a run whose patch holds nothing did, said after the run's name.

    Examples:
        0   -> "changed nothing"
        1   -> "ended with exit status 1 and changed nothing"
        137 -> "ended with exit status 137, the status of a container killed when its
               processes reach SANDBOX_MEMORY (4g), and changed nothing"

    Args:
        exit_status: The exit status the run is reported with.

    Returns:
        "changed nothing", led by the exit status where it is not 0 (_end_with_text).
    """
    if not exit_status:
        return "changed nothing"
    return f"{_end_with_text(exit_status)} and changed nothing"


def _end_with_text(status: int) -> str:
    """Return "ended with" and a container's exit status, for a reason followed by "and".

    Examples:
        1   -> "ended with exit status 1"
        137 -> "ended with exit status 137, the status of a container killed when its
               processes reach SANDBOX_MEMORY (4g),"

    Args:
        status: The exit status.

    Returns:
        "ended with" and status_text(status), closed with a comma where that says what
        the status is the status of.
    """
    text = status_text(status)
    return f"ended with {text}" + ("," if text != f"exit status {status}" else "")


def _task_place_text(entry_dict: dict) -> str:
    """Return where what one task's agent got to is, as a log line names it.

    Examples:
        a commit on codetwine-loom/run/r/1
                                -> "what task 1's agent got to is on
                                   codetwine-loom/run/r/1"
        no commit               -> "nothing of task 2 was taken in (<its error>)"
        an empty patch, 0       -> "nothing of task 2 was taken in (its run changed
                                   nothing)"

    Args:
        entry_dict: The task's entry, as _edit_task() returns it.

    Returns:
        The branch, where the agent's commit landed on one, or else why nothing of the
        task was taken in: its "error", or, where it has none, that its run changed
        nothing (_empty_patch_text).
    """
    if entry_dict["commit"]:
        return f"what task {entry_dict['n']}'s agent got to is on {entry_dict['branch']}"
    why = entry_dict["error"] or f"its run {_empty_patch_text(entry_dict['exit_status'])}"
    return f"nothing of task {entry_dict['n']} was taken in ({why})"


def _combine_task_branch(project_dir: str, output_dir: str, engine: str, image: str,
                         build_doc: bool, request: str, exclude_list: list[str] | None,
                         base: str, entry_list: list[dict], chain_list: list[int],
                         chain_dict: dict) -> tuple[dict | None, list[str], int]:
    """Combine the task branches of one dispatching run, when every task finished.

    The last branch of a chain stands for the chain. A merge left unsettled is made
    again from the union it left, up to INTEGRATE_MAX_RERUN times more, and not once
    MAX_TOKEN_PER_RUN is reached, nor for one whose run of the tests over HEAD keeps any
    merge from being settled ("head_block") or whose container ended with exit status
    _KILL_STATUS. A merge made again that landed no commit ends the merges, and the
    union it was made from stays the one the run is carried on from. A first merge that
    ended with exit status 0 and landed no commit is returned as one left unsettled with
    exit status 1, its "land_why" saying so.

    Args:
        project_dir: The target repository, as an absolute path.
        output_dir: The output directory, as an absolute path.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        build_doc: As integrate() takes it.
        request: The request, as the user wrote it.
        exclude_list: The extra names kept out of the analysis.
        base: The commit the tasks started from, which the union is written against.
        entry_list: One _edit_task() entry per task, in the tasks' order.
        chain_list: Each task's chain number, in the tasks' order, 0 for none.
        chain_dict: {a chain's number: new_chain_dict()'s dict}.

    Returns:
        (integrate()'s return value for the last merge, or None when nothing was
        combined, the branches taken away because a merge reaches them, how many
        merges were made, 0 when none was). For a last merge that landed no commit,
        whether it wrote no patch or one that holds nothing, "branch", "commit" and
        "verdict" are those of the union it was made again from, "exit_status" is
        its own, or that union's (1 at the least) where its own is 0, and "land_why"
        says what it ended with and where that union stays.
    """
    union_dict, drop_list, merge_count = None, [], 0
    unfinish_list = [entry_dict for entry_dict in entry_list if entry_dict["exit_status"]]
    if unfinish_list:
        # The exit status, not the verdict, says a task did not finish: unverified is a
        # way to finish. A branch is named only where the agent's commit landed on it
        logger.error("%d of %d task(s) did not finish, so nothing is combined and the "
                     "repository is left as it is: %s", len(unfinish_list),
                     len(entry_list), "; ".join(_task_place_text(entry_dict)
                                                for entry_dict in unfinish_list))
        return union_dict, drop_list, merge_count
    # The last branch of a chain reaches the rest of it and stands for them
    tip_set = {chain["branch"] for chain in chain_dict.values() if chain["branch"]}
    combine_branch_list = [entry_dict["branch"] for entry_dict in entry_list
                           if entry_dict["commit"] and (not chain_list[entry_dict["n"] - 1]
                                                        or entry_dict["branch"] in tip_set)]
    if not combine_branch_list:
        logger.info("no branch holds anything, so there is nothing to combine")
        return union_dict, drop_list, merge_count
    # Every chain started at base, and one branch is combined against it too
    combine = functools.partial(integrate, project_dir, output_dir, engine=engine,
                                image=image, build_doc=build_doc,
                                request=request, exclude_list=exclude_list,
                                base=base, land=True)
    union_dict = combine(combine_branch_list)
    merge_count = 1
    # A merge that ended with exit status 0 and landed no commit left the tree the
    # branches start from: nothing of what they hold is combined, and the run did not
    # get through and is carried on from the branches
    if not union_dict["commit"] and not union_dict["exit_status"]:
        union_dict = {
            **union_dict, "verdict": integrate_mod.UNSETTLE_VERDICT, "exit_status": 1,
            "land_why": "the merge ended with exit status 0 and changed nothing, so "
                        "there is no union and nothing was put into the working tree. "
                        "The branches it was to combine are as they were"}
        logger.warning("%s", union_dict["land_why"])
    # A merge the run did not settle is run again from the union it left, up to
    # INTEGRATE_MAX_RERUN times more, unless what keeps it from being settled is the
    # run of the tests over HEAD ("head_block") or its container was killed at
    # SANDBOX_MEMORY (_KILL_STATUS). The union before it is a parent of the
    # new one and its name is taken away as the task branches' names are below
    while (union_dict.get("verdict") == integrate_mod.UNSETTLE_VERDICT
           and union_dict.get("branch") and merge_count <= config.INTEGRATE_MAX_RERUN
           and not tokencount.limit_text() and not union_dict.get("head_block")
           and not _is_memory_kill(union_dict)):
        merge_count += 1
        logger.warning("the merge was not settled, so it is run again from %s "
                       "(merge %d of %d)", union_dict["branch"], merge_count,
                       config.INTEGRATE_MAX_RERUN + 1)
        again_dict = combine([], resume=union_dict["branch"])
        if not again_dict["commit"]:
            # The union before it is still the unsettled one the run is carried on
            # from, whatever the merge that landed nothing said of itself
            why = again_dict["land_why"] or (
                f"the merge made again ended with exit status "
                f"{again_dict['exit_status']} and changed nothing, so there is nothing "
                f"of it to put into the working tree")
            union_dict = {
                **again_dict, "branch": union_dict["branch"],
                "commit": union_dict["commit"], "verdict": union_dict["verdict"],
                "exit_status": again_dict["exit_status"] or union_dict["exit_status"] or 1,
                "land_why": f"{why}; the union the merge before it left stays on "
                            f"{union_dict['branch']}"}
            break
        union_dict = again_dict
        drop_list += drop_branch_under(project_dir, [union_dict["branches"][0]],
                                       union_dict["commit"])
    # The branches this run made are parents of what the merge landed on, and what they
    # hold is reached from it; their names are taken away
    drop_list += drop_branch_under(
        project_dir, [entry_dict["branch"] for entry_dict in entry_list],
        union_dict["commit"]) if union_dict["commit"] else []
    return union_dict, drop_list, merge_count


def union_status(union_dict: dict) -> int:
    """Return the exit status a run reports for its union.

    Args:
        union_dict: integrate()'s return value.

    Returns:
        The union's own exit status, or 1 when that is 0 and the stages that followed
        it recorded a case the change does not do.
    """
    if not union_dict["exit_status"] and union_dict.get("gap"):
        logger.warning("the tests found %d case(s) the change does not do: %s",
                       len(union_dict["gap"]),
                       "; ".join(entry_dict["what"] for entry_dict in union_dict["gap"]))
        return 1
    return union_dict["exit_status"]


def resume_branch_list(entry_list: list[dict], union_dict: dict | None,
                       exit_status: int) -> list[str]:
    """Return the branches naming what this run left, for a run carrying on from it.

    A run that got through, as the exit status this run worked out says, leaves nothing
    to carry on from. Of the rest, what is named is whatever holds work nobody has
    finished with. Where the branches were combined, the merge commit reaches every one
    of them and its branch alone is named; where they were not, the branches that a
    patch landed on are named.

    Args:
        entry_list: One per task, as _run_agent_list() returns them.
        union_dict: integrate()'s return value, or None when nothing was combined.
        exit_status: What this run is reporting.

    Returns:
        The branch names, empty when the run finished and nothing is left to carry on
        from. A merge with a stage after it that was not settled names its branch
        whatever the exit status is, and so does one that was to go into the working
        tree and did not (land_status). A change kept out of the working tree for the
        paths it touches (try_land's "land_refuse") names none, whatever is left on
        its branch.
    """
    if union_dict and union_dict.get("land_refuse"):
        return []
    if union_dict and union_dict["branch"] \
            and (open_stage_key(union_dict) or land_status(union_dict)):
        return [union_dict["branch"]]
    if not exit_status:
        return []
    if union_dict and union_dict["branch"]:
        return [union_dict["branch"]]
    return [entry["branch"] for entry in entry_list if entry["commit"]]


def check_across_path(path: str) -> None:
    """Make the directory a file is to be written in.

    A path no file can be written at is refused.

    Args:
        path: Where the file is to be written.

    Raises:
        RuntimeError: When the directory cannot be made or written to, or path is a
            directory.
    """
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"{path} cannot be written: its directory cannot be made "
                           f"({e})") from None
    if os.path.isdir(path):
        raise RuntimeError(f"{path} cannot be written: it is a directory")
    if not os.access(directory, os.W_OK | os.X_OK):
        raise RuntimeError(f"{path} cannot be written: {directory} cannot be written to")


def write_across(path: str, what: object, label: str) -> None:
    """Write one object as JSON to a file, making its directory when it is not there.

    It is written in one step (tmpfile.atomic_write): none is left half-written by a run
    killed mid-write.

    Args:
        path: Where to write it.
        what: The object to write. A value in it that JSON has no form for is written
            as the text str() gives of it.
        label: What the file holds, for the log line.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tmpfile.atomic_write(path) as tmp_path:
        with open(tmp_path, "w", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(json.dumps(what, ensure_ascii=False, indent=1, default=str))
    logger.info("%s is in %s", label, path)


def _read_across(path: str, out_dir: str, who: str, label: str) -> object:
    """Read one object a container wrote for this machine.

    Args:
        path: The file the container was told to write.
        out_dir: The agent's copy of the output directory the file is in.
        who: The run that was to write it, for the error.
        label: What the file holds, for the error.

    Returns:
        Whatever the file holds, its shape not checked.

    Raises:
        RuntimeError: When the file is not there or cannot be read (_json_across), or
            is reached through a link that leads out of out_dir (inside_path).
    """
    try:
        return _json_across(path, out_dir)
    except (OSError, ValueError) as e:
        raise RuntimeError(
            f"{who} left no {label} in {path} ({e}). What it did is in the turn log "
            f"beside it") from None


def _json_across(path: str, out_dir: str) -> object:
    """Read one JSON file a container wrote into an agent's copy of the output directory.

    Args:
        path: The file, as an absolute path under out_dir.
        out_dir: The agent's copy of the output directory the file is in.

    Returns:
        What the file holds.

    Raises:
        RuntimeError: When the file is reached through a link that leads out of
            out_dir (inside_path).
        OSError, ValueError: As tmpfile.read_json() raises them: no such file, not a
            regular file, larger than _MAX_ACROSS_BYTE, or not JSON it can parse.
    """
    return tmpfile.read_json(inside_path(path, out_dir), _MAX_ACROSS_BYTE)


def _is_text_list(value: object) -> bool:
    """Say whether a value is a list of strings."""
    return isinstance(value, list) and all(isinstance(entry, str) for entry in value)


def _is_entry_list(value: object) -> bool:
    """Say whether a value is a list of {"what": str, "why": str} entries."""
    return isinstance(value, list) and all(
        isinstance(entry, dict) and isinstance(entry.get("what"), str)
        and isinstance(entry.get("why"), str) for entry in value)


def _is_list_by_path(value: object, is_entry_list: Callable[[object], bool]) -> bool:
    """Say whether a value is {path: list}, every list one is_entry_list() accepts."""
    return isinstance(value, dict) and all(
        isinstance(path, str) and is_entry_list(entry_list)
        for path, entry_list in value.items())


# The keys of what a run recorded of itself that this machine reads - an outcome file,
# an entry of the history - each with what says its value is of the shape read
_RECORD_SHAPE_DICT: dict[str, Callable[[object], bool]] = {
    **dict.fromkeys(("verdict", "why", "did", "ended", "stage", "skipped", "tests_base"),
                    lambda value: isinstance(value, str)),
    **dict.fromkeys(("untooled", "tests_changed", "non_test_changed", "refused",
                     "baseline", "test_open", "head_block"), _is_text_list),
    **dict.fromkeys(("decisions", "gap", "gap_done"), _is_entry_list),
    "separator": integrate_mod.is_separator_record,
    "no_merge": integrate_mod.is_no_merge_record,
    **dict.fromkeys(("cases", "uncovered"), lambda value: isinstance(value, list)),
    "targets": lambda value: _is_list_by_path(
        value, lambda target_list: isinstance(target_list, list)),
    "tests": lambda value: _is_list_by_path(value, _is_text_list),
}


# The keys of what a run recorded of itself that reach the trailers of its commit
# (run_trailer_dict)
_TRAILER_KEY_TUPLE = ("n", "verdict", "why", "did", "untooled", "tests_changed", "ended",
                      "decisions", "separator", "no_merge", "stage", "non_test_changed",
                      "tests_base", "test_open", "gap", "gap_done", "stand_on")

# The verdicts a merging run and a stage after it record
_MERGE_VERDICT_TUPLE = (integrate_mod.SETTLE_VERDICT, integrate_mod.UNSETTLE_VERDICT)

# The kinds of run whose own record this machine reads, besides the stages of
# integrate.STAGE_TUPLE: a change run, a merge, the merge that combines test tasks, and
# the run that lists what there is to write tests for
_EDIT_KIND = "edit"
_MERGE_KIND = "merge"
_TEST_MERGE_KIND = "test-merge"
_TARGET_KIND = "target"

# The keys of _TRAILER_KEY_TUPLE a merge records
_MERGE_KEY_TUPLE = ("verdict", "why", "did", "tests_changed", "ended", "decisions",
                    "separator", "no_merge")

# The keys of _TRAILER_KEY_TUPLE either refactoring stage records
_REFACTOR_KEY_TUPLE = ("verdict", "why", "did", "untooled", "tests_changed",
                       "non_test_changed", "ended", "decisions")

# What each kind of run records of itself, as {kind: (the keys of _TRAILER_KEY_TUPLE it
# records, the verdicts it records)}. A key of _TRAILER_KEY_TUPLE outside the kind's is
# not read from what a run of that kind wrote. "stage", "test_open" and "stand_on" are
# in none: which stage a commit is of, the files whose tests it leaves open and the
# earlier union a merge stands on are this machine's alone to decide
_RECORD_KIND_DICT: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    _EDIT_KIND: (("n", "verdict", "why", "did", "untooled", "tests_changed"),
                 history.JUDGE_VERDICT_TUPLE),
    _MERGE_KIND: (_MERGE_KEY_TUPLE, _MERGE_VERDICT_TUPLE),
    _TEST_MERGE_KIND: ((*_MERGE_KEY_TUPLE, "gap", "gap_done"), _MERGE_VERDICT_TUPLE),
    integrate_mod.TEST_CREATE_STAGE: ((*_REFACTOR_KEY_TUPLE, "tests_base", "gap"),
                                      _MERGE_VERDICT_TUPLE),
    integrate_mod.REFACTOR_STAGE: (_REFACTOR_KEY_TUPLE, _MERGE_VERDICT_TUPLE),
    integrate_mod.TEST_REFACTOR_STAGE: (_REFACTOR_KEY_TUPLE, _MERGE_VERDICT_TUPLE),
    _TARGET_KIND: ((), ()),
}

# The keys of a commit's record this machine alone decides: which stage the commit is
# of, and the files whose tests it leaves open
_OWN_RECORD_KEY_TUPLE = ("stage", "test_open")


# The keys of what a merge's commit was to record that only a merge the run settled
# carries: a verdict and a reason in place of the run's own, the cases recorded as
# done, and _OWN_RECORD_KEY_TUPLE
_SETTLE_RECORD_KEY_TUPLE = ("verdict", "why", "gap_done", *_OWN_RECORD_KEY_TUPLE)


def _merge_record_dict(record_dict: dict, outcome_dict: dict, status: int,
                       meta_dict: dict) -> dict:
    """Return what the commit of a merge records besides what the run said of itself.

    A merge is one the run settled when the run's outcome says so and neither the
    container nor the patch's summary carries a non-zero exit status. A run that
    wrote no outcome settled nothing.

    Args:
        record_dict: What the commit was to record, by the keys run_trailer_dict()
            reads.
        outcome_dict: What the merging run wrote of itself, or {} when it wrote
            nothing.
        status: The container's exit status.
        meta_dict: The summary the container wrote beside the patch.

    Returns:
        record_dict as it is for a merge the run settled. For any other, record_dict
        without _SETTLE_RECORD_KEY_TUPLE: the commit keeps the run's own verdict and
        reason, where it gave any, is no commit of a stage and names no file as open.
    """
    if outcome_dict.get("verdict") == integrate_mod.SETTLE_VERDICT \
            and not status and not meta_dict.get("exit_status"):
        return record_dict
    return {key: value for key, value in record_dict.items()
            if key not in _SETTLE_RECORD_KEY_TUPLE}


def _record_dict_of(record_obj: object, path: str, kind: str) -> dict:
    """Return what a run recorded of itself when it is of the shape this machine reads.

    Args:
        record_obj: What the file held: an outcome, or one entry of the history.
        path: The file it was read from, for the log.
        kind: The kind of run that wrote it, a key of _RECORD_KIND_DICT.

    Returns:
        record_obj without the keys of _TRAILER_KEY_TUPLE a run of that kind does not
        record and without the keys named in _RECORD_SHAPE_DICT that hold None, when
        it is an object whose other keys named there each hold a value of that shape
        and whose verdict, where it gives one, is one a run of that kind records.
        Otherwise {}, which is logged.
    """
    if not isinstance(record_obj, dict):
        logger.warning("what a run recorded in %s is not an object, so it is not read",
                       path)
        return {}
    key_tuple, verdict_tuple = _RECORD_KIND_DICT[kind]
    read_dict = {key: value for key, value in record_obj.items()
                 if key in key_tuple or key not in _TRAILER_KEY_TUPLE}
    wrong_list = [key for key, is_shape in _RECORD_SHAPE_DICT.items()
                  if read_dict.get(key) is not None and not is_shape(read_dict[key])]
    if not wrong_list and read_dict.get("verdict") \
            and read_dict["verdict"] not in verdict_tuple:
        wrong_list = ["verdict"]
    if wrong_list:
        logger.warning("what a run recorded in %s holds a value of another shape under "
                       "%s, so none of it is read", path, list_text(wrong_list))
        return {}
    return {key: value for key, value in read_dict.items()
            if value is not None or key not in _RECORD_SHAPE_DICT}


def _read_split(path: str, out_dir: str) -> dict:
    """Read what the reading run decided, as it wrote it.

    Args:
        path: The split file the container wrote.
        out_dir: The agent's copy of the output directory the file is in.

    Returns:
        {"questions": [str, ...]}, {"tasks": [str, ...], "defs": [[str, ...], ...]} or
        {"unclear": [str, ...]}. "defs" holds the definition keys each task names, one
        list per task; every list is empty for a split written without them. A key
        holding anything but a list is read as not there. Of its entries, those
        dispatch.is_entry_text() takes are read, at most dispatch.max_entry_of() of
        them, each held to DISPATCH_TASK_MAX_CHAR (_cap_text).

    Raises:
        RuntimeError: When the file is not there, cannot be read, or holds none of
            them.
    """
    split_dict = _read_across(path, out_dir, "the run that reads the request", "split")
    if isinstance(split_dict, dict):
        for key in dispatch_mod.split_key_tuple():
            max_entry = dispatch_mod.max_entry_of(key)
            raw_list = split_dict.get(key)
            if not isinstance(raw_list, list):
                raw_list = []
            def_list_list = split_dict.get("defs")
            if not isinstance(def_list_list, list) or len(def_list_list) != len(raw_list):
                def_list_list = [[] for _ in raw_list]
            pair_list = [(entry_text, def_value)
                         for entry_text, def_value in zip(raw_list, def_list_list)
                         if dispatch_mod.is_entry_text(entry_text)]
            if len(pair_list) > max_entry:
                logger.warning("the split holds %d %s and at most %d are taken; the "
                               "rest are dropped", len(pair_list), key, max_entry)
            pair_list = [(_cap_text(entry_text, config.DISPATCH_TASK_MAX_CHAR,
                                    f"entry {n} of the {key}"), def_value)
                         for n, (entry_text, def_value)
                         in enumerate(pair_list[:max_entry], start=1)]
            if pair_list and key != "tasks":
                return {key: [entry_text for entry_text, _ in pair_list]}
            if pair_list:
                return {key: [entry_text for entry_text, _ in pair_list],
                        "defs": [[def_key for def_key in def_value
                                  if isinstance(def_key, str)]
                                 if isinstance(def_value, list) else []
                                 for _, def_value in pair_list]}
    why = split_dict.get("error", "") if isinstance(split_dict, dict) else ""
    raise RuntimeError(
        "the run that reads the request handed neither any question nor any task back"
        + (f" ({why})" if why else "") + ". Nothing was started")


# A character the argument of a command cannot hold: half of a surrogate pair, and NUL
_NO_ARGUMENT_CHAR = re.compile("[\ud800-\udfff\0]")


def _cap_text(text: str, max_char: int, what: str) -> str:
    """Hold a text a container handed back to the limit the container was to keep.

    A text the container cut itself, max_char characters followed by textcut.cut()'s
    marker, is taken as it is (textcut.cut). Every character an argument of a command
    cannot hold (_NO_ARGUMENT_CHAR) is replaced by U+FFFD first.

    Args:
        text: The text as the container wrote it.
        max_char: The limit. 0 or less is no limit.
        what: What the text is, for the warning.

    Returns:
        The text, or textcut.cut()'s cut of it, with a warning, when it is longer.
    """
    text = _NO_ARGUMENT_CHAR.sub("\ufffd", text)
    keep = cut(text, max_char)
    if keep != text:
        logger.warning("%s is %d characters and at most %d are taken; the rest is cut",
                       what, len(text), max_char)
    return keep


class _NoAnswerError(RuntimeError):
    """An answer file that was read and holds no answer.

    Attributes:
        why: What the run that wrote it gave as the reason, or "".
    """

    def __init__(self, why: str) -> None:
        """Hold the reason the answer file gives.

        Args:
            why: The file's "error", held to DISPATCH_ANSWER_MAX_CHAR, or "".
        """
        super().__init__("the run that answers handed no answer back"
                         + (f" ({why})" if why else ""))
        self.why = why


def _read_answer(path: str, out_dir: str) -> str:
    """Read the answer an answering or a combining run wrote.

    Args:
        path: The answer file the container wrote.
        out_dir: The agent's copy of the output directory the file is in.

    Returns:
        The answer, held to DISPATCH_ANSWER_MAX_CHAR (_cap_text).

    Raises:
        _NoAnswerError: When the file holds no answer, with the reason it gives held
            to DISPATCH_ANSWER_MAX_CHAR.
        RuntimeError: When the file is not there or cannot be read.
    """
    answer_obj = _read_across(path, out_dir, "the run that answers", "answer")
    answer_text = answer_obj.get("answer") if isinstance(answer_obj, dict) else None
    if isinstance(answer_text, str) and answer_text.strip():
        return _cap_text(answer_text, config.DISPATCH_ANSWER_MAX_CHAR, "the answer")
    why = answer_obj.get("error", "") if isinstance(answer_obj, dict) else ""
    raise _NoAnswerError(_cap_text(str(why), config.DISPATCH_ANSWER_MAX_CHAR,
                                   "the reason the run that answers gave") if why else "")


def _log_request_list(task_list: list[str], what: str) -> None:
    """Log every request a run was split into, one line each with its number.

    Args:
        task_list: The requests.
        what: What one request is called, for the log.
    """
    for n, task in enumerate(task_list, 1):
        logger.info("%s %d of %d: %s", what, n, len(task_list), one_line(task))


def _error_text(error: Exception) -> str:
    """Return what an agent's entry says about the exception that ended it.

    Examples:
        RuntimeError("the patch does not apply") -> "the patch does not apply"
        OSError(28, "No space left on device")   -> "OSError: [Errno 28] No space left on device"

    Args:
        error: The exception one agent raised.

    Returns:
        The message, led by the exception's type for an exception that is not a
        RuntimeError.
    """
    if isinstance(error, RuntimeError):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _run_agent_list(task_list: list[str], one: Callable[[int, str, int], dict],
                    what: str) -> list[dict]:
    """Start one agent per task, hand out again what did not finish, and return each.

    A task is handed out at most DISPATCH_MAX_RETRY times more, each time to an agent of
    its own, and what the last try left is what the task is reported as. None is handed
    out again once MAX_TOKEN_PER_RUN is reached, nor is one that ended with a container
    killed at SANDBOX_MEMORY (_is_memory_kill). An agent that raises does not stop the
    others: what it was asked for is reported with the reason, and the tasks that did
    finish are still reported as finished.

    Every agent works under a label of its own, "[task 2/5] " or "[task 2/5 try 2] ",
    and its start and its end are logged with how many of the tasks have finished and
    how many agents are going.

    Args:
        task_list: The requests, one per agent.
        one: (the task's number, the request, which try this is counting from 1) -> the
            entry for that try. Every entry is built by it and holds "n", "request",
            "attempt", "exit_status" and "error".
        what: What one request is called in the label and the log.

    Returns:
        One entry per task, in the tasks' order.
    """
    total = len(task_list)
    count_dict = {"finish": 0, "run": 0}
    lock = threading.Lock()

    def track(n: int, task: str, attempt: int) -> dict:
        """Run one agent under its label, logging its start and its end.

        Args:
            n: The task's number.
            task: The request.
            attempt: Which try this is, counting from 1.

        Returns:
            The entry one() built.
        """
        try_text = f" try {attempt}" if attempt > 1 else ""
        with progress.label(f"[{what} {n}/{total}{try_text}] "):
            with lock:
                count_dict["run"] += 1
            logger.info("%s %d of %d starts", what, n, total)
            start = time.monotonic()
            entry_dict = one(n, task, attempt)
            with lock:
                count_dict["run"] -= 1
                count_dict["finish"] += not entry_dict["exit_status"]
                finish_count, run_count = count_dict["finish"], count_dict["run"]
            if entry_dict["exit_status"]:
                end_text = "did not finish"
            else:
                verdict = entry_dict.get("verdict")
                end_text = f"finished ({verdict})" if verdict else "finished"
            progress.note("%s %d of %d %s in %s: %d of %d finished, %d running", what,
                          n, total, end_text,
                          progress.elapsed_text(time.monotonic() - start), finish_count,
                          total, run_count)
            return entry_dict

    entry_list = _run_agent_batch([(n, task) for n, task in enumerate(task_list, 1)], 1,
                                  track)
    kill_set: set[int] = set()
    for attempt in range(1, config.DISPATCH_MAX_RETRY + 1):
        # A task whose container was killed at SANDBOX_MEMORY is not handed out again
        kill_list = [entry_dict["n"] for entry_dict in entry_list
                     if entry_dict["exit_status"] and _is_memory_kill(entry_dict)
                     and entry_dict["n"] not in kill_set]
        if kill_list:
            kill_set.update(kill_list)
            logger.warning("%d %s(s) did not finish (%s), so they are not handed out "
                           "again; raise SANDBOX_MEMORY to run them", len(kill_list),
                           what, status_text(_KILL_STATUS))
        retry_list = [entry_dict for entry_dict in entry_list
                      if entry_dict["exit_status"] and entry_dict["n"] not in kill_set]
        if not retry_list:
            break
        if tokencount.limit_text():
            logger.warning("%s, so the %d task(s) that did not finish are not handed "
                           "out again", tokencount.limit_text(), len(retry_list))
            break
        logger.warning("%d task(s) did not finish, so they are handed out again, each "
                       "to an agent of its own (try %d of %d)", len(retry_list),
                       attempt + 1, config.DISPATCH_MAX_RETRY + 1)
        retry_dict = {entry_dict["n"]: entry_dict for entry_dict in _run_agent_batch(
            [(entry_dict["n"], entry_dict["request"]) for entry_dict in retry_list],
            attempt + 1, track)}
        entry_list = [retry_dict.get(entry_dict["n"], entry_dict)
                      for entry_dict in entry_list]
    return entry_list


def _run_agent_batch(job_list: list[tuple[int, str]], attempt: int,
                     one: Callable[[int, str, int], dict]) -> list[dict]:
    """Run one round of agents, at most DISPATCH_PARALLEL at a time.

    Each has a place of its own in the output directory.

    Args:
        job_list: [(the task's number, the request), ...].
        attempt: Which try these are, counting from 1.
        one: What runs one agent, as _run_agent_list() takes it.

    Returns:
        One entry per job, in the jobs' order.
    """
    worker_count = max(1, min(len(job_list), config.DISPATCH_PARALLEL))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
    try:
        entry_list = list(pool.map(lambda job: one(job[0], job[1], attempt), job_list))
    except BaseException:
        # A stop (Ctrl-C) reaches this thread alone: the jobs not started are dropped
        # and the containers going are taken away, which ends the agents. They are
        # waited for before the stop is handed on, and so is a stop that came while the
        # containers were taken away
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            stop_running_container()
        finally:
            pool.shutdown(wait=True)
        raise
    pool.shutdown(wait=True)
    return entry_list


def _edit_agent(project_dir: str, output_dir: str, run_id: str, engine: str,
                image: str, exclude_list: list[str] | None,
                request: str, base: str, chain_list: list[int], chain_dict: dict,
                task_list: list[str], n: int, task: str, attempt: int) -> dict:
    """Run one editing agent, after the others of its chain, and return its entry.

    A task of no chain starts at the commit the dispatching run started from. The tasks
    of a chain run in their order (run_in_chain()): each starts at what the last of
    them that finished left, and, when it finishes too, is what the next one starts at.
    A task that starts at the work of tasks before it is told what they were asked for
    (_CHAIN_NOTE).

    Args:
        project_dir, output_dir, run_id, engine, image, exclude_list, request, n,
            task, attempt: As _edit_task() takes them.
        base: The commit the dispatching run started from.
        chain_list: Each task's chain number, in the tasks' order, 0 for none.
        chain_dict: {a chain's number: new_chain_dict()'s dict}.
        task_list: Every task of the dispatching run, in their order.

    Returns:
        _edit_task()'s return value.
    """
    chain = chain_dict.get(chain_list[n - 1])
    edit = functools.partial(_edit_task, project_dir, output_dir, run_id, engine, image,
                             exclude_list, request, n, task, attempt)
    if chain is None:
        return edit(base)
    member_list = [member for member, chain_number in enumerate(chain_list, 1)
                   if chain_number == chain_list[n - 1]]
    return run_in_chain(
        chain, member_list, n, attempt,
        lambda start: edit(start, [task_list[member - 1] for member in chain["land"]
                                   if member != n]),
        lambda entry_dict: bool(entry_dict["commit"]) and not entry_dict["exit_status"],
        f"task {n} waits for the tasks before it that change the same definition")


# What a task of a chain is told in front of its request when it starts at the work of
# tasks before it: {list} holds one line for each of them, "- " and its request
_CHAIN_NOTE = ("This change starts from the repository as earlier changes of the same "
               "request left it, so what the request below says of the code may describe "
               "it as it was before them. Those changes were:\n{list}\n\nThe request:\n")


def _edit_task(project_dir: str, output_dir: str, run_id: str, engine: str,
               image: str, exclude_list: list[str] | None,
               request: str, n: int, task: str, attempt: int, start: str,
               before_list: Sequence[str] = ()) -> dict:
    """Run one editing agent on a branch of its own and return its entry.

    The branch carries the dispatching run and the try: each try starts on a branch of
    its own.

    Args:
        project_dir: The target repository, as an absolute path.
        output_dir: Where the index and the patches live, as it was passed.
        run_id: The dispatching run, which the branch name is built from.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        exclude_list: The extra names kept out of the analysis.
        request: What the dispatching run was given, which the commit carries as a
            trailer.
        n: The task's number.
        task: The change request.
        attempt: Which try this is, counting from 1.
        start: The commit the agent starts at.
        before_list: The requests of the tasks whose work start holds, which the agent
            is told of in front of task (_CHAIN_NOTE).

    Returns:
        {"n", "request", "branch", "attempt", "commit", "exit_status", "error",
        "verdict"}. "commit" is "" for an agent that landed nothing, "error" says why
        for one that did not finish or whose container left no patch that is taken in
        (run()'s "no_patch_why") and is "" otherwise, and "verdict" is what
        history.judge() made of the run.
    """
    branch = _TASK_BRANCH.format(run_id=run_id, n=n)
    if attempt > 1:
        branch += _RETRY_SUFFIX.format(attempt=attempt - 1)
    entry_dict = {"n": n, "request": task, "branch": branch, "attempt": attempt,
                  "commit": "", "exit_status": 0, "error": "", "verdict": ""}
    try:
        note = _CHAIN_NOTE.format(list="\n".join(
            f"- {before}" for before in before_list)) if before_list else ""
        out_dict = run(project_dir, output_dir, task, engine=engine, image=image,
                       branch=branch, exclude_list=exclude_list, run_request=request,
                       guide=config.EDIT_GUIDE, start=start, request_note=note)
        entry_dict = {**entry_dict, "commit": out_dict["commit"] or "",
                      "exit_status": out_dict["exit_status"], "verdict": out_dict["verdict"],
                      "error": out_dict.get("no_patch_why") or ""}
    except Exception as e:
        logger.warning("the agent on %s did not finish: %s", branch, _error_text(e))
        entry_dict = {**entry_dict, "exit_status": 1, "error": _error_text(e)}
    return entry_dict


# What the combining stage is told about an agent that handed nothing back, in place of
# the message the machine logs
_NO_ANSWER_BACK = "the agent handed no answer back"


def _answer_agent(project_dir: str, output_dir: str, engine: str, image: str,
                  exclude_list: list[str] | None, start: str, owner: str, n: int,
                  question: str, attempt: int) -> dict:
    """Run one answering agent and return its entry.

    Args:
        project_dir: The target repository, as an absolute path.
        output_dir: Where the index lives, as it was passed.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        exclude_list: The extra names kept out of the analysis.
        start: The commit the agent reads, which the dispatching run started from.
        owner: The dispatching run's id, which the agent's output directory is named
            under.
        n: The question's number.
        question: The question.
        attempt: Which try this is, counting from 1.

    Returns:
        {"n", "request", "attempt", "answer", "exit_status", "error"}. "answer" is ""
        and "error" says why for an agent that handed no answer back, with the
        container's exit status and what it is the status of (status_text) where the
        container ended with one that is not 0, and the reason the answer file gives
        where it gives one (_NoAnswerError). That reason says what happened and not
        where this machine keeps its files; the whole message is in the log.
    """
    entry_dict = {"n": n, "request": question, "attempt": attempt, "answer": "",
                  "exit_status": 0, "error": ""}
    try:
        out_dict = answer(project_dir, output_dir, question, engine=engine, image=image,
                          exclude_list=exclude_list, start=start, owner=owner)
        entry_dict.update(answer=out_dict["answer"], exit_status=out_dict["exit_status"])
    except Exception as e:
        logger.warning("the agent answering question %d did not finish: %s", n,
                       _error_text(e))
        # How the container ended, then the reason the answer file gives; no other
        # reading error is said
        why_list, read_error = [], e
        if isinstance(e, ContainerEndError):
            why_list.append(f"the container ended ({status_text(e.exit_status)})")
            read_error = e.error
        if isinstance(read_error, _NoAnswerError) and read_error.why:
            why_list.append(f"the run that answers said: {read_error.why}")
        entry_dict.update(exit_status=1, error=f"{_NO_ANSWER_BACK}: {'; '.join(why_list)}"
                          if why_list else _NO_ANSWER_BACK)
    return entry_dict


def _combine_answer(project_dir: str, output_dir: str, request: str,
                    answer_list: list[dict], engine: str, image: str,
                    exclude_list: list[str] | None, start: str, owner: str) -> dict:
    """Run the combining stage over the agents' answers, or list them as they came.

    A question no agent answered is not left out: the combining run is handed it with
    the reason, and says which part of the request that leaves open. A combining run
    that hands no answer back leaves the agents' answers to be read as they came.

    Args:
        project_dir: The target repository, as an absolute path.
        output_dir: Where the index lives, as it was passed.
        request: The request, as the user wrote it.
        answer_list: The agents' entries, as _answer_agent() builds them.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        exclude_list: The extra names kept out of the analysis.
        start: The commit the run reads, which the dispatching run started from.
        owner: The dispatching run's id, which the run's output directory is named
            under.

    Returns:
        {"answer": str, "exit_status": int, "error": str}. "error" says why the
        combining run handed nothing back, and "" when it did.
    """
    no_answer_list = [entry_dict["n"] for entry_dict in answer_list
                      if entry_dict["exit_status"]]
    if no_answer_list:
        logger.error("%d of %d question(s) got no answer; the combining run is told "
                     "so and says which part of the request that leaves open: %s",
                     len(no_answer_list), len(answer_list),
                     ", ".join(str(n) for n in no_answer_list))
    # With no answer to put together, the combining stage answers the request itself
    is_no_answer = len(no_answer_list) == len(answer_list)
    if is_no_answer:
        logger.error("no question was answered, so the request is answered by one run "
                     "of its own rather than put together from nothing")
    try:
        with progress.step("combine the answers"):
            out_dict = answer(project_dir, output_dir, request, engine=engine,
                              image=image, exclude_list=exclude_list,
                              answer_list=None if is_no_answer else answer_list, start=start,
                              owner=owner)
    except Exception as e:
        logger.error("the combining run handed no answer back (%s), so the agents' "
                     "answers are given as they came", _error_text(e))
        return {"answer": answer_mod.render_answer_list(answer_list, for_user=True),
                "exit_status": 1, "error": _error_text(e)}
    return {"answer": out_dict["answer"], "exit_status": out_dict["exit_status"], "error": ""}


def answer(project: str, output_dir: str, question: str, engine: str = "",
           image: str = "", exclude_list: list[str] | None = None,
           answer_list: list[dict] | None = None, start: str = "",
           owner: str = "") -> dict:
    """Answer one question inside a container, or put the agents' answers together.

    The run has one broker, one container and one place of its own in the output
    directory. The place is named after the run itself, or after owner and the run when
    owner is given. What this machine reads back is the answer file, and the copy made
    for the run is taken away when it ends. An answer read back is recorded on this
    machine's side as what the run left (_write_run_mark). It reads the working tree as
    it is: the commit it was told, or the one the tree makes (tree_branch).

    Args:
        project: The target repository. A relative path is resolved from here.
        output_dir: Where the index lives.
        question: The question, or for the combining run the request the agents'
            answers are put together for.
        engine: The container engine. Empty picks one.
        image: The image to run. Empty uses config.CONTAINER_IMAGE.
        exclude_list: The directory and file names kept out of the analysis.
        answer_list: The agents' answers, as answer.render_answer_list() takes them. Given,
            this is the combining run: they are written where the container reads them,
            and the run is asked for the one answer to question.
        start: The commit the container reads. Empty makes the commit of the working
            tree here.
        owner: The id of the dispatching run this answers for. Empty for a run of its
            own.

    Returns:
        {"answer": str, "exit_status": 0}. A run that handed an answer back answered,
        whatever exit status its container ended with.

    Raises:
        RuntimeError: When there is no engine, when project is not a git repository, or
            when the run handed no answer back: a ContainerEndError when its container
            ended with an exit status that is not 0.
    """
    project_dir, root_out_dir, name = _run_place(project, output_dir)
    run_id = new_run_id()
    agent = agent_name(f"{owner}/{run_id}" if owner else "", run_id)
    exclude_list = exclude_list_of_run(project_dir, root_out_dir, name, exclude_list, False)
    out_dir = make_agent_dir(root_out_dir, name, agent, project_dir, exclude_list,
                             False)
    run_file = run_file_layout(root_out_dir, agent, name, run_id)
    answer_in_path = ""
    if answer_list is not None:
        write_across(inside_path(run_file.answer_in, run_file.out_dir), answer_list,
                     "what the agents answered")
        answer_in_path = run_file.answer_in_inside
    with contextlib.ExitStack() as stack:
        base = start or stack.enter_context(
            tree_branch(project_dir, run_id, engine=engine))[0]
        try:
            _, _, status = _start_container(
                engine, image, project_dir, root_out_dir, out_dir, name, agent, base,
                run_file, run_id,
                answer_argv(name, question, run_file.answer_inside, answer_in_path),
                exclude_list)
            answer_text = _read_after_end(
                lambda: _read_answer(run_file.answer, run_file.out_dir), status)
            _write_run_mark(run_file, _ANSWER_KEEP)
        finally:
            drop_the_copy(root_out_dir, name, agent)
    if status:
        logger.warning("the container ended with exit status %d after it wrote its "
                       "answer, and the answer is taken", status)
    return {"answer": answer_text, "exit_status": 0}


# The branch one test task lands on, when there are several: the branch the combined
# change is on, then the test task's number
_TEST_BRANCH = "{branch}-test/{n}"

# The branch the run combining the test tasks' branches lands on, named beside them
_TEST_UNION_BRANCH = _TEST_BRANCH.format(branch="{branch}", n="union")

# What the stage records about the tests written over the commit the change was written
# against, where the test tasks did not all say the same
_TESTS_BASE_MIX = "mixed"


def _fill_list(unit_list: list[list[str]], def_count_dict: dict,
               def_max: int) -> list[list[str]]:
    """Put units of files into lists, each filled up to the even share of the definitions.

    The units are taken in the order they come and put into one list after another. A
    unit is never split: one holding more than def_max definitions is a list of its own.

    Args:
        unit_list: [[a file, ...], ...], at least one unit, none of them empty.
        def_count_dict: {a file: how many of its definitions the change touched}.
        def_max: How many definitions one list holds at most. 0 or less makes one.

    Returns:
        [[a file, ...], ...], every file of unit_list in one of them. Units that hold
        no definition between them make one list.
    """
    def_count = sum(def_count_dict[path] for file_list in unit_list for path in file_list)
    list_count = max(1, -(-def_count // def_max)) if def_max > 0 else 1
    share = -(-def_count // list_count)
    path_list_list: list[list[str]] = [[]]
    fill_count = 0
    for file_list in unit_list:
        unit_def_count = sum(def_count_dict[path] for path in file_list)
        if path_list_list[-1] and fill_count + unit_def_count > share:
            path_list_list.append([])
            fill_count = 0
        path_list_list[-1] += file_list
        fill_count += unit_def_count
    return path_list_list


def join_set_list(link_dict: dict) -> list[list]:
    """Return the keys joined through a value they share, one sorted list per set.

    Examples:
        {"a": ["t1"], "b": [], "c": ["t1"]} -> [["a", "c"], ["b"]]

    Args:
        link_dict: {key: the values it is linked by}. Two keys with a value in common
            are in one set, however many keys that chains together.

    Returns:
        The sets, each a sorted list of keys, sorted.
    """
    set_of_dict = {key: {key} for key in link_dict}
    first_dict: dict = {}
    for key in sorted(link_dict):
        for value in link_dict[key]:
            first = first_dict.setdefault(value, key)
            if set_of_dict[first] is not set_of_dict[key]:
                join_set = set_of_dict[first] | set_of_dict[key]
                for member in join_set:
                    set_of_dict[member] = join_set
    return sorted({id(key_set): sorted(key_set) for key_set in set_of_dict.values()}
                  .values())


def start_count_of(chain_list: list[int], parallel: int) -> int:
    """Return how many agents start at once, of tasks run parallel at a time in order.

    A task of a chain after its first one waits in its place for the task before it
    (_edit_agent), so of the first parallel tasks only those of no chain and the first
    task of each chain start.

    Examples:
        ([1, 1, 1, 0, 0], 3) -> 1
        ([0, 1, 1, 0], 3)    -> 2
        ([0, 0, 0, 0], 3)    -> 3

    Args:
        chain_list: Each task's chain number, as task_chain_list() returns them.
        parallel: How many tasks are run at a time (DISPATCH_PARALLEL).

    Returns:
        The number of agents that start at once.
    """
    head_list = chain_list[:max(1, parallel)]
    return sum(1 for index, chain in enumerate(head_list)
               if not chain or chain not in head_list[:index])


def task_chain_list(def_list_list: list[list[str]]) -> list[int]:
    """Return each task's chain number, 0 for no chain.

    The tasks that name a definition in common are one chain, numbered from 1 in the
    order of their first task.

    Examples:
        [["a::f"], ["b::g"], ["a::f", "c::h"], ["c::h"], []] -> [1, 0, 1, 1, 0]

    Args:
        def_list_list: The definition keys each task names, in the tasks' order.

    Returns:
        One chain number per task, in the tasks' order.
    """
    chain_list = [0] * len(def_list_list)
    chain = 0
    for n_list in join_set_list({n: def_list for n, def_list in enumerate(def_list_list, 1)}):
        if len(n_list) < 2:
            continue
        chain += 1
        for n in n_list:
            chain_list[n - 1] = chain
    return chain_list


def new_chain_dict(tip: str) -> dict:
    """Return the state one chain of tasks shares.

    Args:
        tip: The commit the first task of the chain starts at.

    Returns:
        {"cond": the condition the tasks wait on, "tip": the commit the next one starts
         at, "branch": the branch that commit is on, or "", "ran": {a task's number:
         {"attempt": the last try it ran, "ok": whether that try finished, "kill":
         whether its container was killed at SANDBOX_MEMORY (_is_memory_kill), which
         _run_agent_list() hands out no more}}, "land":
         the numbers of the tasks whose work "tip" holds, in their order}.
    """
    return {"cond": threading.Condition(), "tip": tip, "branch": "", "ran": {}, "land": []}


def run_in_chain(chain_dict: dict, member_list: list[int], n: int, attempt: int,
                 run_task: Callable[[str], dict], advance: Callable[[dict], bool],
                 wait_text: str) -> dict:
    """Run one task of a chain after the tasks before it in the chain, and return its entry.

    A task waits until every task of the chain with a smaller number has run in this
    try, or finished or was killed at SANDBOX_MEMORY in an earlier one and so is not
    run again. The tasks of one chain therefore run one at a time and in their order,
    in every try.

    Args:
        chain_dict: new_chain_dict()'s dict for the chain.
        member_list: The numbers of the chain's tasks.
        n: This task's number.
        attempt: Which try this is, counting from 1.
        run_task: (the commit to start at) -> the task's entry, with "commit", "branch"
            and "exit_status".
        advance: (the entry) -> whether the next task starts at this one's commit.
        wait_text: The line logged when the task has to wait.

    Returns:
        run_task()'s return value.
    """
    cond = chain_dict["cond"]

    def is_past(member: int) -> bool:
        """Say whether a task of the chain ran in this try or is not run again."""
        ran_dict = chain_dict["ran"].get(member)
        return ran_dict is not None and (ran_dict["attempt"] >= attempt or ran_dict["ok"]
                                         or ran_dict["kill"])

    def before_are_past() -> bool:
        """Say whether every task of the chain before this one is past."""
        return all(is_past(member) for member in member_list if member < n)

    with cond:
        if not before_are_past():
            logger.info(wait_text)
            cond.wait_for(before_are_past)
        start = chain_dict["tip"]
    entry_dict = None
    try:
        entry_dict = run_task(start)
        return entry_dict
    finally:
        with cond:
            if entry_dict is not None and advance(entry_dict):
                chain_dict.update({"tip": entry_dict["commit"], "branch": entry_dict["branch"],
                                   "land": [*chain_dict["land"], n]})
            chain_dict["ran"][n] = {
                "attempt": attempt,
                "ok": entry_dict is not None and not entry_dict["exit_status"],
                "kill": entry_dict is not None and bool(entry_dict["exit_status"])
                and _is_memory_kill(entry_dict)}
            cond.notify_all()


def test_task_list(test_by_file_dict: dict, def_count_dict: dict,
                   def_max: int) -> list[dict]:
    """Split the files tests are written for into test tasks.

    Files whose definitions one test file refers to are one set, however many files that
    chains together: no test file that is already there is written by two test tasks at
    once. The sets holding at most def_max of the definitions the change touched are
    taken in the order of their first path and put into one test task after another,
    each filled up to the even share of their definitions; such a set is never split.

    A set holding more is split, in the order of its paths, into parts of an even share
    of its definitions, a file's definitions staying together. Each part is a test task
    of its own, and the parts of one set are one chain: they are written one after
    another, each over what the one before it left (_test_task_agent), and every one of
    them writes to the test files the set shares.

    The test tasks come in the order of their place in their chain and then of their
    first path: the first part of every chain and the test tasks of no chain come before
    any second part.

    Examples:
        ({"a.py": ["t/1.py"], "b.py": [], "c.py": ["t/1.py"]},
         {"a.py": 1, "b.py": 2, "c.py": 1}, 2)
            -> [{"paths": ["a.py", "c.py"], "test_paths": ["t/1.py"], "chain": 0},
                {"paths": ["b.py"], "test_paths": [], "chain": 0}]
        ({"a.py": ["t/1.py"], "b.py": ["t/1.py"], "c.py": ["t/1.py"]},
         {"a.py": 1, "b.py": 1, "c.py": 1}, 2)
            -> [{"paths": ["a.py", "b.py"], "test_paths": ["t/1.py"], "chain": 1},
                {"paths": ["c.py"], "test_paths": ["t/1.py"], "chain": 1}]

    Args:
        test_by_file_dict: {a file tests are written for: the test files that refer to
            its definitions}, as the listing run read them.
        def_count_dict: {a file tests are written for: how many of its definitions the
            change touched}.
        def_max: How many definitions one test task holds at most. 0 or less makes one.

    Returns:
        [{"paths": the files, sorted, "test_paths": the test files that refer to them,
        sorted, "chain": the number of the chain the test task is a part of, counting
        from 1, or 0 for none}, ...]. [] for no file.
    """
    # The sets: files joined through a test file they share
    set_list = join_set_list(test_by_file_dict)

    # The chains: the parts of each set that holds more than def_max
    whole_list: list[list[str]] = []
    place_list: list[tuple[int, list[str], int]] = []
    for file_list in set_list:
        if len(file_list) > 1 \
                and 0 < def_max < sum(def_count_dict[path] for path in file_list):
            chain = 1 + len({chain for _, _, chain in place_list})
            part_list = _fill_list([[path] for path in file_list], def_count_dict, def_max)
            place_list += [(n, path_list, chain) for n, path_list in enumerate(part_list)]
        else:
            whole_list.append(file_list)
    if whole_list:
        place_list += [(0, path_list, 0)
                       for path_list in _fill_list(whole_list, def_count_dict, def_max)]
    return [{"paths": sorted(path_list),
             "test_paths": sorted({test_path for path in path_list
                                   for test_path in test_by_file_dict[path]}),
             "chain": chain}
            for _, path_list, chain in sorted(place_list)]


def add_case_number(task_dict_list: list[dict], first_case: object) -> None:
    """Give each test task the case numbers it hands out, no two of them the same.

    Examples:
        (3 test tasks, 5) -> "first_case" 5, 6 and 7, each with "case_step" 3

    Args:
        task_dict_list: test_task_list()'s test tasks. Each gets "first_case" and
            "case_step".
        first_case: The listing run's "first_case", the number after the highest case
            number the test files hold. Anything but a positive integer adds nothing.
    """
    if type(first_case) is not int or first_case < 1:
        return
    for n, task_dict in enumerate(task_dict_list):
        task_dict.update({"first_case": first_case + n, "case_step": len(task_dict_list)})


def _test_task_agent(project_dir: str, root_out_dir: str, name: str, engine: str,
                     image: str, union: str, base: str, request: str,
                     exclude_list: list[str] | None, baseline_list: list[str],
                     task_dict_list: list[dict], place_dict: dict, n: int, task: str,
                     attempt: int) -> dict:
    """Write the tests of one test task, after the others of its chain, and take them in.

    A test task of no chain is written over the commit holding the combined change. The
    test tasks of a chain are written in their order (run_in_chain()): each over what
    the last of them that settled left, and, when it settles too, it is what the next
    one is written over.

    Args:
        project_dir, root_out_dir, name, engine, image, union, base, request,
        exclude_list, baseline_list, task_dict_list, n, task, attempt: As
            _write_test_task() takes them.
        place_dict: As _write_test_task() takes it, with "chain_dict": {a chain's
            number: new_chain_dict()'s dict}.

    Returns:
        _write_test_task()'s return value.
    """
    chain = task_dict_list[n - 1]["chain"]
    chain_dict = place_dict["chain_dict"].get(chain)
    write = functools.partial(
        _write_test_task, project_dir, root_out_dir, name, engine, image, union, base,
        request, exclude_list, baseline_list, task_dict_list, place_dict, n, task, attempt)
    if chain_dict is None:
        return write(union)
    member_list = [member for member, task_dict in enumerate(task_dict_list, 1)
                   if task_dict["chain"] == chain]
    return run_in_chain(
        chain_dict, member_list, n, attempt, write,
        lambda entry_dict: bool(entry_dict["commit"]) and not _is_test_task_open(entry_dict),
        f"test task {n} waits for the test tasks before it that write the same test files")


def _write_test_task(project_dir: str, root_out_dir: str, name: str, engine: str,
                     image: str, union: str, base: str, request: str,
                     exclude_list: list[str] | None, baseline_list: list[str],
                     task_dict_list: list[dict], place_dict: dict, n: int, task: str,
                     attempt: int, start: str) -> dict:
    """Write the tests of one test task, in a container, and take them in.

    The container starts at start. The patch it writes lands as one commit on top of
    that commit, recording what _stage_record_dict() makes of what the container said.
    A lone test task that was not left unsettled lands a commit for a patch that holds
    nothing too.

    Args:
        project_dir: The target repository, as an absolute path.
        root_out_dir: The output directory as it was passed.
        name: The repository's directory name.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        union: The commit holding the combined change.
        base: The commit the combined change was written against.
        request: The request the change was made for, or "".
        exclude_list: The extra names kept out of the analysis.
        baseline_list: The baseline before the stage, as the listing run read it.
        task_dict_list: test_task_list()'s test tasks.
        place_dict: {"branch": the branch a lone test task lands on, "agent" and
            "out_dir": the agent it runs as and its copy of the output directory,
            "gap_done": the cases the commit of a lone test task that settled records
            as done (_create_test), "lang": the language the listing run took as the
            primary one, which the container takes too, or ""}. Several test tasks each
            land on a branch and run as an agent of their own.
        n: The test task's number, counting from 1.
        task: The test task's files on one line, for the log.
        attempt: Which try this is, counting from 1.
        start: The commit the tests are written over: union, or what a test task of the
            same chain left on top of it.

    Returns:
        {"n", "request", "attempt", "exit_status", "error", "paths", "chain", "branch",
        "commit", "patch", "files", "verdict", "why", "skipped", "cases", "uncovered",
        "refused", "tests_changed", "non_test_changed", "tests_base", "ended", "gap"}.
        "gap" is what the test task recorded with record_gap(). "chain" is the test
        task's, as test_task_list() numbers it. "exit_status" is 1 only for a test task
        whose container could not be run to its end or left no patch that is taken in
        (_no_patch_record_dict); one that left its tests unsettled says so in "verdict".
        "memory_kill" is added when the container ran, set when it was killed at
        SANDBOX_MEMORY.
    """
    path_list = task_dict_list[n - 1]["paths"]
    is_alone = len(task_dict_list) == 1
    branch = place_dict["branch"] if is_alone \
        else _TEST_BRANCH.format(branch=place_dict["branch"], n=n)
    entry_dict = {"n": n, "request": task, "attempt": attempt, "exit_status": 0, "error": "",
                  "paths": path_list, "chain": task_dict_list[n - 1]["chain"],
                  "branch": branch, "commit": None, "patch": None,
                  "files": 0, "verdict": "", "why": "", "skipped": "", "cases": [],
                  "uncovered": [], "refused": [], "tests_changed": [],
                  "non_test_changed": [], "tests_base": "", "ended": "", "gap": []}
    run_id = new_run_id()
    agent = place_dict["agent"] if is_alone else agent_name(branch, run_id)
    try:
        out_dir = place_dict["out_dir"] if is_alone else make_agent_dir(
            root_out_dir, name, agent, project_dir, exclude_list, False)
        run_file = run_file_layout(root_out_dir, agent, name, run_id)
        write_across(inside_path(run_file.target, run_file.out_dir),
                     {"paths": path_list, "test_paths": task_dict_list[n - 1]["test_paths"],
                      "baseline": baseline_list,
                      **{key: task_dict_list[n - 1][key] for key in ("first_case", "case_step")
                         if key in task_dict_list[n - 1]}},
                     "the files to write tests for")
        _, _, status = _start_container(
            engine, image, project_dir, root_out_dir, out_dir, name, agent, start,
            run_file, run_id,
            test_create_argv(name, base, request, run_file.outcome_inside,
                             target_in=run_file.target_inside), exclude_list,
            primary_ext=place_dict["lang"])
        outcome_dict = _stage_record_dict(
            integrate_mod.TEST_CREATE_STAGE,
            _read_outcome(run_file.outcome, run_file.out_dir,
                          integrate_mod.TEST_CREATE_STAGE), status)
        if is_alone and outcome_dict["verdict"] == integrate_mod.SETTLE_VERDICT \
                and not outcome_dict["skipped"]:
            outcome_dict = {**outcome_dict, "gap_done": place_dict["gap_done"]}
        land_dict = _land_patch(
            project_dir, run_file, start, branch, status, True,
            lambda meta_dict: commit_message(
                _test_commit_request(request, union),
                run_trailer_dict(meta_dict, outcome_dict, request),
                prefix=integrate_mod.TEST_CREATE_STAGE),
            reset=not is_alone,
            keep_empty=is_alone and not is_stage_open(outcome_dict), engine=engine)
        outcome_dict = _no_patch_record_dict(outcome_dict, land_dict)
    except Exception as e:
        logger.warning("the tests of test task %d were not written: %s", n,
                       _error_text(e))
        return {**entry_dict, "exit_status": 1, "error": _error_text(e),
                "why": _error_text(e)}
    finally:
        if not is_alone:
            drop_the_copy(root_out_dir, name, agent)
    _add_stage_result(entry_dict, outcome_dict, land_dict, (
        "verdict", "why", "skipped", "cases", "uncovered", "refused",
        "tests_changed", "non_test_changed", "tests_base", "ended", "gap"), status)
    return entry_dict


def _add_stage_result(result_dict: dict, outcome_dict: dict, land_dict: dict,
                      key_tuple: tuple[str, ...], status: int) -> None:
    """Add what a stage's container recorded and what of its patch landed to a result.

    Args:
        result_dict: The result, changed in place.
        outcome_dict: What the stage recorded, as _no_patch_record_dict() returns it.
        land_dict: _land_patch()'s return value for the stage's container.
        key_tuple: The keys of outcome_dict taken, where it holds them.
        status: The container's exit status. "memory_kill" is set when it is
            _KILL_STATUS.
    """
    result_dict.update({key: outcome_dict[key] for key in key_tuple
                        if key in outcome_dict})
    result_dict.update({key: land_dict[key] for key in ("patch", "files", "commit")})
    result_dict["exit_status"] = int(land_dict["no_patch"])
    result_dict["memory_kill"] = status == _KILL_STATUS


def _is_test_task_open(entry_dict: dict) -> bool:
    """Say whether one test task's tests were left unsettled.

    Args:
        entry_dict: _test_task_agent()'s return value.

    Returns:
        True unless the test task settled, or had nothing to write a test for.
    """
    return not entry_dict["skipped"] \
        and entry_dict["verdict"] != integrate_mod.SETTLE_VERDICT


def _create_test(project_dir: str, root_out_dir: str, out_dir: str, name: str, agent: str,
                 engine: str, image: str, union: str, base: str, branch: str,
                 request: str, exclude_list: list[str] | None,
                 only_path_list: Sequence[str] = ()) -> dict:
    """Write tests for a combined change, in containers, and take them in on its branch.

    Steps:
    1. One container lists the definitions the change touched and the tests failing
       now, and the language it took as the primary one, and calls no model. The
       containers of steps 3 and 4 take that language too
    2. The files are split into test tasks of at most config.TEST_CREATE_MAX_DEFINITION
       definitions, the parts of files that share a test file being one chain
       (test_task_list)
    3. One container per test task writes its tests, config.DISPATCH_PARALLEL of them
       at once and those of one chain one after another. A lone test task lands on the
       branch itself; several land on branches of their own, which start at the commit
       holding the combined change, or at what the test task of the same chain before
       them left
    4. The branches of the test tasks that settled are combined onto that commit by
       integrate(), and the branch is moved to what it lands
    5. What the test tasks recorded is put together into the stage's result
       (_add_task_result)

    A stage that had nothing to write tests for lands a commit that changes nothing
    and records the stage as settled (_land_stage_record). A stage carrying on for
    files (only_path_list) is left unsettled instead, and starts no test task, when
    the listing has nothing to write tests for or names none of those files.

    The cases the commits the change was combined from record as not done
    (integrate.gap_list_in() from base to the merge under union) are recorded as done
    by the commit of this stage that leaves no file open, where a test task wrote
    tests: what the change does not do is from then on what this stage's own tests
    found.

    Args:
        project_dir: The target repository, as an absolute path.
        root_out_dir: The output directory as it was passed.
        out_dir: The agent's own copy of the output directory.
        name: The repository's directory name.
        agent: The agent, as agent_name() names it.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        union: The commit the tests are written over.
        base: The commit the combined change was written against.
        branch: The branch the tests are committed to, whose tip is union.
        request: The request the change was made for, or "".
        exclude_list: The extra names kept out of the analysis.
        only_path_list: The files to write tests for. Empty takes every file the
            listing names.

    Returns:
        {"verdict", "why", "skipped", "cases", "uncovered", "tests_changed",
        "non_test_changed", "tests_base", "ended", "exit_status", "patch", "files",
        "branch", "commit", "test_tasks", "open_paths", "refused", "gap", "lang"}.
        "lang" is the language the listing run took as the primary one, which the stages
        that refactor are handed too, or "". "commit" is None when nothing landed on
        branch, "patch" is None when no test did, "skipped" says why there was nothing
        to write, or "", "test_tasks" holds
        _test_task_agent()'s entries, "open_paths" the files of the test tasks whose
        tests were not settled and "refused" the paths a test task was refused a write
        to, "gap" the cases still not done: as the combined tests leave them where they
        were run again, and as the test tasks recorded them with record_gap() otherwise.
        "verdict" is settled only when every test task and the combining run settled.
        "memory_kill" is set when the combining run was killed at SANDBOX_MEMORY, or
        every test task left unsettled was (_add_task_result); a listing run killed so
        gives "exit_status" _KILL_STATUS.
    """
    result_dict = {"verdict": "", "why": "", "skipped": "", "cases": [],
                   "uncovered": [], "tests_changed": [], "non_test_changed": [],
                   "tests_base": "", "ended": "", "exit_status": 0, "patch": None,
                   "files": 0, "branch": branch, "commit": None, "test_tasks": [],
                   "open_paths": [], "refused": [], "gap": [], "lang": ""}
    # The branches an earlier run of the stage left its open test tasks on, and the one
    # it left their tests combined on (_TEST_UNION_BRANCH), are taken away, under a
    # branch this tool named (is_own_branch) alone
    for old_branch in _own_branch_list(project_dir, _TEST_BRANCH.format(
            branch=branch, n="").rstrip("/")) if is_own_branch(branch) else []:
        gitcmd.run(project_dir, "branch", "-D", old_branch)

    with progress.step("write the tests"):
        # == Step 1: what there is to write tests for ============
        run_id = new_run_id()
        run_file = run_file_layout(root_out_dir, agent, name, run_id)
        try:
            _, _, status = _start_container(
                engine, image, project_dir, root_out_dir, out_dir, name, agent, union,
                run_file, run_id,
                test_create_argv(name, base, outcome_out=run_file.outcome_inside,
                                 target_only=True), exclude_list)
            target_dict = _read_outcome(run_file.outcome, run_file.out_dir, _TARGET_KIND)
        except Exception as e:
            result_dict.update({"verdict": integrate_mod.UNSETTLE_VERDICT,
                                "exit_status": 1, "why": _error_text(e)})
            _warn_stage_open(integrate_mod.TEST_CREATE_STAGE, result_dict)
            return result_dict
        if not isinstance(target_dict.get("targets"), dict):
            result_dict.update({"verdict": integrate_mod.UNSETTLE_VERDICT, "exit_status": status,
                                "why": f"what there is to write tests for could not be "
                                       f"listed ({status_text(status)})"})
            _warn_stage_open(integrate_mod.TEST_CREATE_STAGE, result_dict)
            return result_dict
        # A listing that was handed back is the listing, whatever its container ended
        # with afterwards
        if status:
            logger.warning("the container ended with exit status %d after it wrote what "
                           "there is to write tests for, and the listing is taken", status)
        lang = result_dict["lang"] = target_dict.get("lang") \
            if target_dict.get("lang") in LANG_DICT else ""
        # A file the listing names no definition of is none to write tests for
        test_by_file_dict = {path: (target_dict.get("tests") or {}).get(path, [])
                             for path, def_list in target_dict["targets"].items()
                             if def_list
                             and (not only_path_list or path in set(only_path_list))}
        open_why = _open_why(only_path_list, target_dict, test_by_file_dict)
        if open_why:
            open_path_list = sorted(set(only_path_list))
            result_dict.update({
                "verdict": integrate_mod.UNSETTLE_VERDICT, "open_paths": open_path_list,
                "why": f"no test was written for the files left to write tests for "
                       f"({list_text(open_path_list)}): {open_why}"})
            _warn_stage_open(integrate_mod.TEST_CREATE_STAGE, result_dict)
            return result_dict
        skip_record_dict = {"stage": integrate_mod.TEST_CREATE_STAGE,
                            "verdict": integrate_mod.SETTLE_VERDICT}
        if target_dict.get("skipped"):
            result_dict.update({"skipped": target_dict["skipped"], "commit": _land_stage_record(
                project_dir, run_file.worktree, branch, skip_record_dict,
                _test_commit_request(request, union), request, engine)})
            return result_dict

        # == Step 2: the test tasks ==============================
        miss_list = sorted(set(only_path_list) - set(test_by_file_dict))
        if miss_list:
            logger.warning("%d of the file(s) left to write tests for are not among the "
                           "files the change touched, so no test is written for them: %s",
                           len(miss_list), list_text(miss_list))
        def_count_dict = {path: len(target_dict["targets"][path])
                          for path in test_by_file_dict}
        task_dict_list = test_task_list(test_by_file_dict, def_count_dict,
                                        config.TEST_CREATE_MAX_DEFINITION)
        if not task_dict_list:
            result_dict.update({
                "skipped": "none of the files left to write tests for was changed",
                "commit": _land_stage_record(
                    project_dir, run_file.worktree, branch, skip_record_dict,
                    _test_commit_request(request, union), request, engine)})
            return result_dict
        add_case_number(task_dict_list, target_dict.get("first_case"))
        logger.info("tests are written for %d definition(s) of %d file(s) in %d test "
                    "task(s)", sum(def_count_dict.values()), len(test_by_file_dict),
                    len(task_dict_list))

        # == Step 3: one container per test task =================
        chain_dict = {task_dict["chain"]: new_chain_dict(union)
                      for task_dict in task_dict_list if task_dict["chain"]}
        earlier_gap_list = integrate_mod.gap_list_in(
            project_dir, base, integrate_mod.below_stage(project_dir, union))
        one = functools.partial(
            _test_task_agent, project_dir, root_out_dir, name, engine, image, union, base,
            request, exclude_list, list(target_dict.get("baseline") or []), task_dict_list,
            {"branch": branch, "agent": agent, "out_dir": out_dir,
             "chain_dict": chain_dict, "gap_done": earlier_gap_list, "lang": lang})
        task_list = [" ".join(task_dict["paths"]) for task_dict in task_dict_list]
        entry_list = [one(1, task_list[0], 1)] if len(task_dict_list) == 1 \
            else _run_agent_list(task_list, one, "test task")
        result_dict["test_tasks"] = entry_list

        # == Step 4: one commit on the branch ====================
        # The cases recorded as not done, as the combined tests leave them where they
        # were run again, and as the test tasks recorded them otherwise
        gap_list = [entry for entry_dict in entry_list for entry in entry_dict["gap"]]
        if len(task_dict_list) == 1:
            result_dict.update({key: entry_list[0][key] for key in ("patch", "files", "commit")})
        else:
            combine_dict = _combine_test_task(
                project_dir, root_out_dir, engine, image, union, branch, request,
                exclude_list, entry_list,
                [chain["branch"] for chain in chain_dict.values() if chain["branch"]],
                run_file.worktree, earlier_gap_list, lang)
            gap_list = combine_dict.pop("gap", gap_list)
            result_dict.update(combine_dict)

    # == Step 5: the stage's result ==========================
    _add_task_result(result_dict, entry_list, gap_list)
    if is_stage_open(result_dict):
        _warn_stage_open(integrate_mod.TEST_CREATE_STAGE, result_dict)
    return result_dict


def _open_why(only_path_list: Sequence[str], target_dict: dict,
              test_by_file_dict: dict) -> str:
    """Return why the files a run carrying on was told to write tests for stay open.

    They stay open when the listing has nothing to write tests for, or names none of
    them.

    Args:
        only_path_list: The files the run was told to write tests for, empty for all.
        target_dict: What the listing run recorded, with "targets".
        test_by_file_dict: The files of the listing to write tests for.

    Returns:
        The reason, or "" when they do not stay open or no file was named.
    """
    open_why = ""
    if only_path_list and target_dict.get("skipped"):
        open_why = target_dict["skipped"]
    elif only_path_list and not test_by_file_dict:
        open_why = "none of them is among the files the change touched"
    return open_why


def _add_task_result(result_dict: dict, entry_list: list[dict], gap_list: list[dict]
                     ) -> None:
    """Put what the test tasks recorded together into the result of the stage.

    Args:
        result_dict: _create_test()'s result so far, changed in place. Its "why" is
            added after the test tasks' reasons. "memory_kill" is set when the run
            combining the test tasks was killed at SANDBOX_MEMORY, or every test task
            left unsettled was.
        entry_list: _test_task_agent()'s entries.
        gap_list: The cases recorded as not done.
    """
    open_list = [entry_dict for entry_dict in entry_list if _is_test_task_open(entry_dict)]
    why_list = [f"test task {entry_dict['n']} ({' '.join(entry_dict['paths'])}): "
                f"{entry_dict['why'] or 'nothing said why'}"
                if len(entry_list) > 1 else entry_dict["why"]
                for entry_dict in open_list]
    why_list += [result_dict["why"]] if result_dict["why"] else []
    result_dict.update({
        "verdict": integrate_mod.UNSETTLE_VERDICT if why_list or open_list
        else integrate_mod.SETTLE_VERDICT,
        "why": " / ".join(why_list),
        "skipped": entry_list[0]["skipped"]
        if all(entry_dict["skipped"] for entry_dict in entry_list) else "",
        "cases": [case for entry_dict in entry_list for case in entry_dict["cases"]],
        "uncovered": [entry for entry_dict in entry_list
                      for entry in entry_dict["uncovered"]],
        "refused": sorted({path for entry_dict in entry_list
                           for path in entry_dict["refused"]}),
        "tests_changed": sorted({path for entry_dict in entry_list
                                 for path in entry_dict["tests_changed"]}),
        "non_test_changed": sorted({path for entry_dict in entry_list
                                    for path in entry_dict["non_test_changed"]}),
        "tests_base": _tests_base_of(entry_list),
        "ended": entry_list[0]["ended"] if len(entry_list) == 1 else "",
        "exit_status": max(entry_dict["exit_status"] for entry_dict in entry_list),
        "memory_kill": _is_memory_kill(result_dict)
        or (bool(open_list)
            and all(_is_memory_kill(entry_dict) for entry_dict in open_list)),
        "open_paths": sorted(path for entry_dict in open_list
                             for path in entry_dict["paths"]),
        "gap": gap_list})


def _tests_base_of(entry_list: list[dict]) -> str:
    """Return what test tasks' tests said over the commit the change was written against.

    Args:
        entry_list: _test_task_agent()'s entries.

    Returns:
        The one value the test tasks that wrote tests all recorded, _TESTS_BASE_MIX when
        they recorded different ones, and "" when none wrote a test.
    """
    base_set = {entry_dict["tests_base"] for entry_dict in entry_list
                if entry_dict["tests_base"]}
    if len(base_set) > 1:
        return _TESTS_BASE_MIX
    return base_set.pop() if base_set else ""


def _combine_test_task(project_dir: str, root_out_dir: str, engine: str, image: str,
                       union: str, branch: str, request: str,
                       exclude_list: list[str] | None, entry_list: list[dict],
                       chain_branch_list: list[str], worktree_dir: str,
                       earlier_gap_list: list[dict], primary_ext: str = "") -> dict:
    """Combine the test tasks that settled into one commit on the combined change's branch.

    The test tasks' branches start at the commit holding the combined change, or at a
    branch of their chain that does, and are combined onto it by integrate(), with the
    same gates, on _TEST_UNION_BRANCH. Of a chain the last branch alone is combined,
    which holds the ones before it. The branch is then moved to what that landed, and
    the branches it reaches are taken away.

    Where no test task left a branch to combine and one of them was not left
    unsettled, what they recorded lands as a commit that changes nothing
    (_land_stage_record). The commit records what the test tasks that settled
    recorded: the files they changed and the cases they found not done.

    Args:
        project_dir: The target repository, as an absolute path.
        root_out_dir: The output directory as it was passed.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        union: The commit holding the combined change.
        branch: The branch that commit is on.
        request: The request the change was made for, or "".
        exclude_list: The extra names kept out of the analysis.
        entry_list: _test_task_agent()'s entries.
        chain_branch_list: The last branch of every chain that settled a test task.
        worktree_dir: Where the worktree of a commit that changes nothing is put. It
            must not exist yet.
        earlier_gap_list: The cases the commit records as done when no test task was
            left unsettled and one of them wrote tests (_create_test).
        primary_ext: The language the listing took as the primary one, which the
            container combining the tests takes as well. Empty keeps the index's.

    Returns:
        {"patch", "files", "commit", "why"}: what landed on branch, "commit" None when
        nothing did, and "why" saying what stopped it, or "". "memory_kill" is added,
        set, when the combining run's container was killed at SANDBOX_MEMORY.
    """
    settle_list = [entry_dict for entry_dict in entry_list
                   if entry_dict["commit"] and not _is_test_task_open(entry_dict)]
    combine_branch_list = [entry_dict["branch"] for entry_dict in settle_list
                           if not entry_dict["chain"]] + chain_branch_list
    open_list = [entry_dict for entry_dict in entry_list if _is_test_task_open(entry_dict)]
    if not combine_branch_list and len(open_list) == len(entry_list):
        return {"patch": None, "files": 0, "commit": None, "why": ""}
    record_dict = {
        "stage": integrate_mod.TEST_CREATE_STAGE,
        "tests_base": _tests_base_of(settle_list),
        "test_open": sorted(path for entry_dict in open_list
                            for path in entry_dict["paths"]),
        "tests_changed": sorted({path for entry_dict in entry_list
                                 if not _is_test_task_open(entry_dict)
                                 for path in entry_dict["tests_changed"]}),
        "non_test_changed": sorted({path for entry_dict in entry_list
                                    if not _is_test_task_open(entry_dict)
                                    for path in entry_dict["non_test_changed"]}),
        "gap": [entry for entry_dict in entry_list if not _is_test_task_open(entry_dict)
                for entry in entry_dict["gap"]]}
    if open_list:
        record_dict.update({"verdict": integrate_mod.UNSETTLE_VERDICT,
                            "why": f"{len(open_list)} test task(s) were not settled"})
    elif any(not entry_dict["skipped"] for entry_dict in entry_list):
        record_dict["gap_done"] = earlier_gap_list
    if not combine_branch_list:
        return {"patch": None, "files": 0, "why": "", "commit": _land_stage_record(
            project_dir, worktree_dir, branch,
            {"verdict": integrate_mod.SETTLE_VERDICT, **record_dict},
            _test_commit_request(request, union), request, engine)}
    try:
        union_dict = integrate(project_dir, root_out_dir, combine_branch_list,
                               engine=engine, image=image, request=request,
                               exclude_list=exclude_list, base=union, stage_after=False,
                               record_dict=record_dict,
                               step_name="combine the test branches",
                               gap_list=record_dict["gap"],
                               union_branch=_TEST_UNION_BRANCH.format(branch=branch),
                               primary_ext=primary_ext)
    except Exception as e:
        return {"patch": None, "files": 0, "commit": None,
                "why": f"the test tasks' tests could not be combined: {_error_text(e)}"}
    if not union_dict.get("commit"):
        return {"patch": None, "files": 0, "commit": None,
                "memory_kill": _is_memory_kill(union_dict),
                "why": f"the run combining the test tasks' tests left no patch that is "
                       f"taken in, so they were not combined: {union_dict['no_patch_why']}"
                if union_dict.get("no_patch") else
                f"the run combining the test tasks' tests "
                f"{_empty_patch_text(union_dict['exit_status'])}, so they were not "
                f"combined"}
    if union_dict["exit_status"]:
        return {"patch": union_dict.get("patch"), "files": union_dict.get("files", 0),
                "commit": None, "memory_kill": _is_memory_kill(union_dict),
                "why": f"the test tasks' tests were combined on {union_dict['branch']} and "
                       f"the merge was not settled "
                       f"({status_text(union_dict['exit_status'])}): "
                       f"{union_dict['merge_why']}"}
    with _branch_lock(project_dir):
        proc = gitcmd.run(project_dir, "branch", "-f", branch, union_dict["commit"])
    if proc.returncode != 0:
        return {"patch": union_dict["patch"], "files": union_dict["files"], "commit": None,
                "why": f"{branch} could not be moved to the combined tests on "
                       f"{union_dict['branch']}: {proc.stderr.strip()}"}
    drop_branch_under(project_dir, [union_dict["branch"],
                                    *(entry_dict["branch"] for entry_dict in settle_list)],
                      union_dict["commit"])
    return {"patch": union_dict["patch"], "files": union_dict["files"],
            "commit": union_dict["commit"], "why": "",
            **({"gap": union_dict["gap"]} if "gap" in union_dict else {})}


# What each stage that follows a merging run works on, as a log line names it, and how
# the warning about a stage left unsettled starts
_STAGE_WHAT_DICT = {integrate_mod.TEST_CREATE_STAGE: "the tests",
                    integrate_mod.REFACTOR_STAGE: "the refactoring",
                    integrate_mod.TEST_REFACTOR_STAGE: "the refactoring of the tests"}
_STAGE_OPEN_DICT = {integrate_mod.TEST_CREATE_STAGE: "the tests were not settled",
                    integrate_mod.REFACTOR_STAGE: "the refactoring was not settled",
                    integrate_mod.TEST_REFACTOR_STAGE: "the refactoring of the tests was "
                                                       "not settled"}

# The name of the step each stage that refactors is shown under while it runs
_STAGE_STEP_DICT = {integrate_mod.REFACTOR_STAGE: "refactor the change",
                    integrate_mod.TEST_REFACTOR_STAGE: "refactor the tests"}


def _no_patch_record_dict(record_dict: dict, land_dict: dict) -> dict:
    """Return what a stage recorded as unsettled when its container wrote no patch.

    Args:
        record_dict: What the stage recorded, as _stage_record_dict() returns it.
        land_dict: _land_patch()'s return value for that container.

    Returns:
        record_dict as it is when the container wrote a patch. Otherwise record_dict
        with "verdict" unsettled, "skipped" empty and "why" saying that the container
        left no patch that is taken in, followed by _land_patch()'s "no_patch_why".
    """
    if not land_dict["no_patch"]:
        return record_dict
    return {**record_dict, "verdict": integrate_mod.UNSETTLE_VERDICT, "skipped": "",
            "why": f"the container left no patch that is taken in, so nothing of the "
                   f"stage was taken in: {land_dict['no_patch_why']}"}


def _stage_record_dict(stage: str, outcome_dict: dict, status: int) -> dict:
    """Return what the commit of a stage records, whatever its container said.

    Args:
        stage: The stage this process started, one of integrate.STAGE_TUPLE.
        outcome_dict: What the container wrote, as _read_outcome() reads it.
        status: The container's exit status.

    Returns:
        outcome_dict with "stage" set to stage and "skipped" always there. A container
        that named something as skipped is recorded as settled where it gave no
        verdict; any other that gave none is recorded as unsettled, "why" saying that
        it left no account where the container gave no reason.
    """
    record_dict = {"skipped": "", **outcome_dict, "stage": stage}
    if record_dict.get("verdict"):
        return record_dict
    if record_dict["skipped"]:
        return {**record_dict, "verdict": integrate_mod.SETTLE_VERDICT}
    return {**record_dict, "verdict": integrate_mod.UNSETTLE_VERDICT,
            "why": record_dict.get("why") or f"the run {_end_with_text(status)} and left "
                                             f"no account"}


def _test_commit_request(request: str, union: str) -> str:
    """Return what the message of a commit of the stage that writes tests starts with.

    Args:
        request: The request the change was made for, or "".
        union: The commit holding the combined change.

    Returns:
        request, or a line naming the commit when there is none.
    """
    return request or f"write tests for {gitcmd.short(union)}"


def _land_stage_record(project_dir: str, worktree_dir: str, branch: str,
                       record_dict: dict, commit_request: str, request: str,
                       engine: str = "") -> str | None:
    """Commit what a stage that left no file recorded, onto the tip of its branch.

    The commit changes nothing and carries the record as its trailers
    (take_in_patch() without a patch).

    Args:
        project_dir: The target repository, as an absolute path.
        worktree_dir: Where the worktree is put while the commit is made. It must not
            exist yet.
        branch: The branch the stages commit to, which is there.
        record_dict: What the commit records, by the keys run_trailer_dict() reads,
            with "stage".
        commit_request: What the commit's message starts with.
        request: The request the change was made for, or "".
        engine: The container engine the run uses, as take_in_patch() takes it.

    Returns:
        The commit's hash, or None when git did not make it, which is logged.
    """
    stage = record_dict["stage"]
    try:
        return take_in_patch(
            project_dir, "", branch, branch, worktree_dir,
            commit_message(commit_request, run_trailer_dict({}, record_dict, request),
                           prefix=stage), engine=engine)["commit"]
    except RuntimeError as e:
        logger.warning("what %s recorded could not be put on %s: %s",
                       _STAGE_WHAT_DICT[stage], branch, e)
        return None


def _stage_on_list() -> list[str]:
    """Return the stages that follow a settled merge, in the order they run.

    Returns:
        The ones of integrate.STAGE_TUPLE whose setting is on: config.TEST_CREATE,
        config.REFACTOR and config.TEST_REFACTOR.
    """
    on_tuple = (config.TEST_CREATE, config.REFACTOR, config.TEST_REFACTOR)
    return [stage for stage, is_on in zip(integrate_mod.STAGE_TUPLE, on_tuple) if is_on]


def _warn_stage_open(stage: str, stage_dict: dict) -> None:
    """Log that a stage following a merging run was left unsettled, and how to carry on.

    Args:
        stage: One of integrate.STAGE_TUPLE.
        stage_dict: The stage's result.
    """
    branch = stage_dict["branch"]
    logger.warning("%s: %s. %s. Carry on from it with integrate %s",
                   _STAGE_OPEN_DICT[stage],
                   stage_dict["why"] or "the run said nothing about why",
                   f"What was settled is on {branch} as {gitcmd.short(stage_dict['commit'])}"
                   if stage_dict["commit"] else f"Nothing was added to {branch}",
                   command_line("--resume", branch))


def is_stage_open(stage_dict: dict | None) -> bool:
    """Say whether one stage following a merging run was started and left unsettled.

    Args:
        stage_dict: The stage's result, or None for a run without the stage.

    Returns:
        True when the stage ran and did not settle. A run without the stage, and one
        whose stage called no model, are not.
    """
    return bool(stage_dict) and not stage_dict["skipped"] \
        and stage_dict["verdict"] != integrate_mod.SETTLE_VERDICT


def open_stage_key(result_dict: dict) -> str:
    """Return the key of the stage a merging run left unsettled, or "" for none.

    Args:
        result_dict: integrate()'s return value.

    Returns:
        integrate.stage_key() of the first stage that ran and did not settle.
    """
    for stage in integrate_mod.STAGE_TUPLE:
        if is_stage_open(result_dict.get(integrate_mod.stage_key(stage))):
            return integrate_mod.stage_key(stage)
    return ""


def _land_target(result_dict: dict) -> str:
    """Return the commit a settled merging run puts into the working tree.

    Args:
        result_dict: integrate()'s return value so far, with "branch" the combined
            change's and "settle_commit" as _run_stage_list() returns it.

    Returns:
        The branch's tip, or the last commit a stage settled - the combined change's
        own when none did - when a stage was left unsettled.
    """
    if open_stage_key(result_dict):
        return result_dict["settle_commit"]
    return result_dict["branch"]


def _land_union(project_dir: str, base: str, result_dict: dict, patch_path: str,
                branch: str, last_path: str, earlier_base: str = "") -> dict:
    """Put what a merging run settled into the working tree.

    Its branch is taken away when nothing is left on it. What goes in is up to the last
    commit a stage settled (_land_target): the union itself, or the last stage settled
    on top of it. The branch stays where a stage was left unsettled or the tests found a
    case the change does not do, and where its name is not one this tool made up
    (is_own_branch), which is a branch passed to carry on from; otherwise its name is
    taken away.

    Args:
        project_dir: The target repository, as an absolute path.
        base: The commit the working tree holds, which the change is applied over.
        result_dict: integrate()'s return value so far.
        patch_path: Where the diff put into the working tree is written.
        branch: The branch the union and the stages landed on.
        last_path: Where the copy of an applied diff is kept (last_land_path()).
        earlier_base: As land_in_tree() takes it.

    Returns:
        {"land", "land_why", "land_command", "land_refuse"}, as try_land() returns
        them.
    """
    target = _land_target(result_dict)
    land_dict = try_land(project_dir, base, target, patch_path, last_path, earlier_base)
    if land_dict["land"] and not open_stage_key(result_dict) \
            and not result_dict.get("gap"):
        if is_own_branch(branch):
            drop_branch_under(project_dir, [branch], target)
        else:
            logger.info("%s is not a name this tool made up, so the branch is left "
                        "where it is", branch)
    return land_dict


def _own_record_at(project_dir: str, commit: str) -> dict:
    """Return what the run that wrote one commit recorded on it, or {} for none.

    Args:
        project_dir: The repository to read.
        commit: The commit.

    Returns:
        integrate.own_record_at()'s entry for that commit.
    """
    return integrate_mod.own_record_at(project_dir, commit)


def _pass_unsettled_stage(project_dir: str, commit: str) -> tuple[str, dict, str]:
    """Pass over the commits of stages nobody settled, from commit down.

    A commit of tests that names files still open is not passed over: it holds the
    test tasks that were settled.

    Args:
        project_dir: The repository to read.
        commit: The commit to start from.

    Returns:
        (the first commit that is not passed over, its record as _own_record_at()
        reads it, the stage of the last commit passed over or "" when none was).
    """
    record_dict = _own_record_at(project_dir, commit)
    open_stage = ""
    while record_dict.get("stage") in integrate_mod.STAGE_TUPLE \
            and record_dict["verdict"] != integrate_mod.SETTLE_VERDICT \
            and not record_dict["test_open"]:
        open_stage = record_dict["stage"]
        commit = integrate_mod.first_parent(project_dir, commit)
        record_dict = _own_record_at(project_dir, commit)
    return commit, record_dict, open_stage


def _merge_under_stage(project_dir: str, commit: str, record_dict: dict
                       ) -> tuple[str, dict]:
    """Go down from a commit of a stage to the merge every commit of a stage is over.

    Args:
        project_dir: The repository to read.
        commit: The commit to start from.
        record_dict: Its record, as _own_record_at() reads it.

    Returns:
        (the first commit that is no commit of a stage, its record). commit itself
        when it is none.
    """
    while record_dict.get("stage") in integrate_mod.STAGE_TUPLE:
        commit = integrate_mod.first_parent(project_dir, commit)
        record_dict = _own_record_at(project_dir, commit)
    return commit, record_dict


def resume_head(project_dir: str, branch: str) -> tuple[str, str]:
    """Return the commit an earlier merge's branch is carried on from, and its base.

    The base is the commit its change was written against. The commits of stages nobody
    settled are passed over from the tip down (_pass_unsettled_stage). The change was
    written against the first parent of the merge under every commit of a stage
    (_merge_under_stage).

    Args:
        project_dir: The repository to read.
        branch: The branch an earlier merge left.

    Returns:
        (the commit carried on from, the commit the change was written against).

    Raises:
        RuntimeError: When the repository holds no branch of that name.
    """
    head, record_dict, _ = _pass_unsettled_stage(
        project_dir, integrate_mod.commit_of(project_dir, branch))
    merge, _ = _merge_under_stage(project_dir, head, record_dict)
    return head, integrate_mod.first_parent(project_dir, merge)


def _held_union_commit(project_dir: str, branch: str) -> str:
    """Return the last commit of an earlier merge's branch whose change the tree holds.

    The tree is the working tree. The commits are read from the one the branch is
    carried on from (resume_head) as _held_commit() reads them, the change of each being
    the one from the commit the merge was written against.

    Args:
        project_dir: The target repository, as an absolute path.
        branch: The branch an earlier merge left.

    Returns:
        The full hash of the first such commit from the top, or "" when the working
        tree holds the change of none of them, and for a branch whose merge is not
        one a merging run settled (integrate.is_settle_merge).

    Raises:
        RuntimeError: When the repository holds no branch of that name.
    """
    commit, base = resume_head(project_dir, branch)
    if not integrate_mod.is_settle_merge(
            project_dir, integrate_mod.below_stage(project_dir, commit)):
        return ""
    return _held_commit(project_dir, base, commit)


def stage_resume_dict(project_dir: str, branch: str) -> dict:
    """Return which stage a branch is carried on from and over which commit, or {}.

    The commits of stages nobody settled are passed over from the tip down. What is
    reached is a commit of tests that names the files still open, a commit a stage
    settled, or the merge that was settled. The first says the tests are written again
    for those files, the second that the stage after it is next, and the third that the
    first stage is. When a commit of a stage left open was passed over, that stage is
    next instead. A stage that was settled left a commit, one that changes nothing when
    the stage wrote nothing, and the stages up to the commit reached are through. Of the
    stages from there on, the first whose setting is on is the one started. The commit
    the change was written against is the first parent of the merge, which lies under
    every commit of a stage.

    Args:
        project_dir: The repository to read.
        branch: The branch a merging run left.

    Returns:
        {"stage": the stage to start with, "union": the commit it runs over, "settled":
        the last commit that was settled, "merge": the merge that was settled, "base":
        the commit the change was written against, "paths": the files to write tests
        for, [] for all and for open files that cannot be read (path_list_of)}, or {}
        for a branch no stage is left to run on, for one whose open tests have their
        stage turned off and for one no settled merge is under
        (integrate.is_settle_merge).

    Raises:
        RuntimeError: When the repository holds no branch of that name.
    """
    union, record_dict, open_stage = _pass_unsettled_stage(
        project_dir, integrate_mod.commit_of(project_dir, branch))
    is_test_open = bool(record_dict.get("test_open"))
    path_list = path_list_of(record_dict.get("test_open", "")) or []
    stage_tuple = integrate_mod.STAGE_TUPLE
    # The stage left open comes first, and the stages before it are through
    if open_stage:
        stage_tuple = stage_tuple[stage_tuple.index(open_stage):]
    elif record_dict.get("stage") in stage_tuple and not is_test_open:
        stage_tuple = stage_tuple[stage_tuple.index(record_dict["stage"]) + 1:]
    next_list = [stage for stage in stage_tuple if stage in _stage_on_list()]
    if not next_list or (is_test_open and next_list[0] != integrate_mod.TEST_CREATE_STAGE):
        return {}
    commit, _ = _merge_under_stage(project_dir, union, record_dict)
    if not integrate_mod.is_settle_merge(project_dir, commit):
        return {}
    return {"stage": next_list[0], "union": union,
            "settled": commit if is_test_open else union, "merge": commit,
            "base": integrate_mod.first_parent(project_dir, commit), "paths": path_list}


def _settle_stage_of(project_dir: str, branch: str) -> str:
    """Return the stage a branch records as the last one settled over a settled merge.

    The commits of stages nobody settled are passed over from the tip down
    (_pass_unsettled_stage).

    Args:
        project_dir: The repository to read.
        branch: The branch a merging run left.

    Returns:
        The stage of the commit reached, when that is a commit a stage settled that
        names no open tests and the merge under it is one a merging run settled
        (integrate.is_settle_merge); "" otherwise.

    Raises:
        RuntimeError: When the repository holds no branch of that name.
    """
    head, record_dict, _ = _pass_unsettled_stage(
        project_dir, integrate_mod.commit_of(project_dir, branch))
    if record_dict.get("stage") not in integrate_mod.STAGE_TUPLE \
            or record_dict["verdict"] != integrate_mod.SETTLE_VERDICT \
            or record_dict["test_open"]:
        return ""
    merge, _ = _merge_under_stage(project_dir, head, record_dict)
    return record_dict["stage"] if integrate_mod.is_settle_merge(project_dir, merge) else ""


def _carry_stage_of(project_dir: str, branch: str) -> str:
    """Return the stage a branch combined alone carries as the last one settled.

    That is the branch's own (_settle_stage_of), or, for a union an earlier merge left
    unsettled, the one of the commit it was merged from when that is one commit alone
    (integrate.earlier_side_list, which opens every unsettled union under it too): a
    union whose stages were all settled stays so through the merges of it alone that
    were left unsettled.

    Args:
        project_dir: The repository to read.
        branch: The branch an earlier merge left.

    Returns:
        The stage, or "" when neither names one.

    Raises:
        RuntimeError: When the repository holds no branch of that name, or the parents
            of a commit could not be read.
    """
    stage = _settle_stage_of(project_dir, branch)
    if stage:
        return stage
    side_list = integrate_mod.earlier_side_list(
        project_dir, integrate_mod.commit_of(project_dir, branch))
    return _settle_stage_of(project_dir, side_list[0]) if len(side_list) == 1 else ""


def _refactor_stage(project_dir: str, root_out_dir: str, out_dir: str, name: str,
                    agent: str, engine: str, image: str, stage: str, start: str,
                    base: str, branch: str, request: str,
                    exclude_list: list[str] | None, primary_ext: str = "") -> dict:
    """Run one stage that refactors, in a container, and take what it left in.

    The container starts at start. The patch it writes lands as one commit on top of
    that commit, recording what _stage_record_dict() makes of what the container said.
    A stage that was not left unsettled lands a commit for a patch that holds nothing
    too.

    Args:
        project_dir: The target repository, as an absolute path.
        root_out_dir: The output directory as it was passed.
        out_dir: The agent's own copy of the output directory.
        name: The repository's directory name.
        agent: The agent, as agent_name() names it.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        stage: One of refactor.STAGE_TUPLE.
        start: The commit the stage runs over, which is the branch's tip.
        base: The commit the combined change was written against.
        branch: The branch the stage's commit lands on.
        request: The request the change was made for, or "".
        exclude_list: The extra names kept out of the analysis.
        primary_ext: The language the container takes as the primary one in place of
            the index's (build_run_argv). Empty keeps the index's.

    Returns:
        {"verdict", "why", "skipped", "refused", "tests_changed", "non_test_changed",
        "ended", "exit_status", "patch", "files", "branch", "commit"}. "commit" is None
        when nothing landed on branch, and "patch" is None when the stage changed
        nothing. "exit_status" is 1 only for a stage whose container could not be run
        to its end or left no patch that is taken in (_no_patch_record_dict); one that
        left its work unsettled says so in "verdict". "memory_kill" is added when the
        container ran, set when it was killed at SANDBOX_MEMORY.
    """
    stage_dict = {"verdict": "", "why": "", "skipped": "", "refused": [],
                  "tests_changed": [], "non_test_changed": [], "ended": "",
                  "exit_status": 0, "patch": None, "files": 0, "branch": branch,
                  "commit": None}
    what = _STAGE_WHAT_DICT[stage]
    with progress.step(_STAGE_STEP_DICT[stage]):
        run_id = new_run_id()
        run_file = run_file_layout(root_out_dir, agent, name, run_id)
        try:
            _, _, status = _start_container(
                engine, image, project_dir, root_out_dir, out_dir, name, agent, start,
                run_file, run_id,
                refactor_argv(name, stage, base, request, run_file.outcome_inside),
                exclude_list, primary_ext=primary_ext)
            outcome_dict = _stage_record_dict(
                stage, _read_outcome(run_file.outcome, run_file.out_dir, stage), status)
            commit_request = request or f"{stage} {gitcmd.short(start)}"
            land_dict = _land_patch(
                project_dir, run_file, start, branch, status, True,
                lambda meta_dict: commit_message(
                    commit_request, run_trailer_dict(meta_dict, outcome_dict, request),
                    prefix=stage),
                keep_empty=not is_stage_open(outcome_dict), engine=engine)
            outcome_dict = _no_patch_record_dict(outcome_dict, land_dict)
        except Exception as e:
            logger.warning("%s did not run to its end: %s", what, _error_text(e))
            stage_dict.update({"verdict": integrate_mod.UNSETTLE_VERDICT, "exit_status": 1,
                               "why": _error_text(e)})
            _warn_stage_open(stage, stage_dict)
            return stage_dict
    _add_stage_result(stage_dict, outcome_dict, land_dict, (
        "verdict", "why", "skipped", "refused", "tests_changed", "non_test_changed",
        "ended"), status)
    if is_stage_open(stage_dict):
        _warn_stage_open(stage, stage_dict)
    return stage_dict


def _run_stage_list(project_dir: str, root_out_dir: str, out_dir: str, name: str,
                    agent: str, engine: str, image: str, union: str, base: str,
                    branch: str, request: str, exclude_list: list[str] | None,
                    first_stage: str = "", settle_commit: str = "",
                    only_path_list: Sequence[str] = ()) -> dict:
    """Run the stages that follow a settled merge, one after another, on its branch.

    Each stage whose setting is on starts at the branch's tip, which is what the stage
    before it landed, and lands a commit of its own there. A stage that was left
    unsettled is run again from where _rerun_start() puts the branch back, up to its
    setting's times more; it is the last one started, and it is not run again once
    MAX_TOKEN_PER_RUN is reached, when a container of it was killed at SANDBOX_MEMORY
    (_is_memory_kill), nor when what it says of why a test command did not run names a
    command that could not be found (_missing_command_list). The language the
    test-writing stage took as the primary one (_create_test's "lang") is handed to the
    containers of the stages that refactor; without one, each of them works it out
    itself.

    Args:
        project_dir: The target repository, as an absolute path.
        root_out_dir: The output directory as it was passed.
        out_dir: The agent's own copy of the output directory.
        name: The repository's directory name.
        agent: The agent, as agent_name() names it.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        union: The commit the first stage runs over, which is the branch's tip.
        base: The commit the combined change was written against.
        branch: The branch the stages commit to.
        request: The request the change was made for, or "".
        exclude_list: The extra names kept out of the analysis.
        first_stage: The stage to start with. "" starts with the first one that is on.
        settle_commit: The last commit that was settled before these stages. "" is
            union.
        only_path_list: The files the stage that writes tests writes them for. Empty
            takes every file it lists.

    Returns:
        {integrate.stage_key(stage): the stage's result, for every stage that was started,
        "settle_commit": the last commit a stage settled, or the one passed when none
        landed anything, "gap": integrate.gap_list_in() from base to "settle_commit"}.
    """
    stage_list = _stage_on_list()
    if first_stage:
        stage_list = [stage for stage in stage_list
                      if integrate_mod.STAGE_TUPLE.index(stage)
                      >= integrate_mod.STAGE_TUPLE.index(first_stage)]
    result_dict: dict = {"settle_commit": settle_commit or union}
    tip, lang = union, ""
    for stage in stage_list:
        is_test = stage == integrate_mod.TEST_CREATE_STAGE
        max_rerun = config.TEST_CREATE_MAX_RERUN if is_test else config.REFACTOR_MAX_RERUN
        path_list = only_path_list
        stage_dict: dict = {}
        for run_count in range(max_rerun + 1):
            if run_count:
                if tokencount.limit_text():
                    break
                rerun_dict = _rerun_start(project_dir, branch, stage, stage_dict, tip,
                                          path_list, run_count + 1, max_rerun + 1)
                if not rerun_dict:
                    break
                tip, path_list = rerun_dict["union"], rerun_dict["paths"]
            if is_test:
                stage_dict = _create_test(project_dir, root_out_dir, out_dir, name, agent,
                                          engine, image, tip, base, branch, request,
                                          exclude_list, only_path_list=path_list)
                lang = stage_dict.get("lang", "")
            else:
                stage_dict = _refactor_stage(project_dir, root_out_dir, out_dir, name,
                                             agent, engine, image, stage, tip, base, branch,
                                             request, exclude_list, primary_ext=lang)
            if not is_stage_open(stage_dict):
                break
            # A stage held back by a command that is not in the image is not run again
            missing_list = _missing_command_list(stage_dict)
            if missing_list:
                if run_count < max_rerun:
                    logger.warning("%s did not settle because %s is not in the image, so "
                                   "the stage is not run again; use an image that has it "
                                   "(CONTAINER_IMAGE), or set TEST_COMMAND",
                                   _STAGE_WHAT_DICT[stage], ", ".join(missing_list))
                break
            # A stage a container of which was killed at SANDBOX_MEMORY is not run again
            if _is_memory_kill(stage_dict):
                if run_count < max_rerun:
                    logger.warning("%s did not settle (%s), so the stage is not run "
                                   "again; raise SANDBOX_MEMORY to run it",
                                   _STAGE_WHAT_DICT[stage], status_text(_KILL_STATUS))
                break
        result_dict[integrate_mod.stage_key(stage)] = stage_dict
        if is_stage_open(stage_dict):
            break
        if stage_dict["commit"]:
            tip = result_dict["settle_commit"] = stage_dict["commit"]
    result_dict["gap"] = integrate_mod.gap_list_in(project_dir, base,
                                                   result_dict["settle_commit"])
    return result_dict


def _missing_command_list(stage_dict: dict) -> list[str]:
    """Return the commands a stage left unsettled was held back for lack of.

    Args:
        stage_dict: The stage's result.

    Returns:
        edit.missing_command_list() of the stage's "why", or, for a stage of several
        test tasks, of every unsettled test task's when each of them names one; []
        otherwise.
    """
    task_list = stage_dict.get("test_tasks") or []
    if len(task_list) <= 1:
        return edit_mod.missing_command_list(stage_dict.get("why") or "")
    by_task_list = [edit_mod.missing_command_list(entry_dict.get("why") or "")
                    for entry_dict in task_list if _is_test_task_open(entry_dict)]
    if not by_task_list or not all(by_task_list):
        return []
    return list(dict.fromkeys(command for one in by_task_list for command in one))


def _rerun_start(project_dir: str, branch: str, stage: str, stage_dict: dict,
                 start: str, path_list: Sequence[str], run_number: int,
                 max_run: int) -> dict:
    """Put a branch back where the stage just left unsettled is run again from.

    Where several test tasks left a commit holding the ones that settled and naming
    the files still open, the stage is run again over that commit, for those files.
    Any other stage is run again over the commit it started at, which takes what it
    left off the branch, for the files it was run for.

    Args:
        project_dir: The target repository, as an absolute path.
        branch: The branch the stages commit to.
        stage: The stage that was left unsettled.
        stage_dict: That stage's result.
        start: The commit that run of the stage started at.
        path_list: The files that run of the stage wrote tests for, empty for all.
        run_number: Which run of the stage this is, counting from 1.
        max_run: How many runs of the stage there are at most.

    Returns:
        {"union": the commit the stage is run again over, "paths": the files it
        writes tests for, empty for all}, or {} when the branch cannot be put back.
    """
    is_part_settle = len(stage_dict.get("test_tasks") or ()) > 1 \
        and bool(stage_dict["commit"]) and bool(stage_dict["open_paths"])
    union, path_list = (stage_dict["commit"], stage_dict["open_paths"]) \
        if is_part_settle else (start, list(path_list))
    try:
        _put_branch_back(project_dir, branch, union)
    except RuntimeError as e:
        logger.warning("%s is not run again: %s", _STAGE_WHAT_DICT[stage], e)
        return {}
    logger.warning("%s, so the stage is run again over %s%s (run %d of %d)",
                   _STAGE_OPEN_DICT[stage], gitcmd.short(union),
                   f", for {' '.join(path_list)}" if path_list else "",
                   run_number, max_run)
    return {"union": union, "paths": path_list}


def _resume_stage(project_dir: str, root_out_dir: str, name: str, branch: str,
                  resume_dict: dict, engine: str, image: str, build_doc: bool,
                  request: str, exclude_list: list[str] | None, land: bool,
                  ask_doc: bool = False) -> dict:
    """Run the stages a settled merge still has ahead of it.

    What they settle is put into the working tree when land asks for it. The working
    tree holds what the run that settled the merge put into it, which is up to the last
    commit a stage settled (resume_dict["settled"]); what the stages settle from there
    on is what is put in now.

    Args:
        project_dir: The target repository, as an absolute path.
        root_out_dir: The output directory as it was passed.
        name: The repository's directory name.
        branch: The branch the merge is on, which the stages commit to.
        resume_dict: stage_resume_dict()'s return value for that branch.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        build_doc: As integrate() takes it.
        request: The request the change was made for, or "".
        exclude_list: The extra names kept out of the analysis.
        land: Whether what the stages settle is put into the working tree
            (_land_union).
        ask_doc: As integrate() takes it.

    Returns:
        What integrate() returns, "branches" holding the branch alone, "commit" the
        merge that was settled and each stage's result under integrate.stage_key().

    Raises:
        RuntimeError: As integrate() raises, and when the branch cannot be put back on
            the commit the stages run over.
    """
    union = resume_dict["union"]
    what = _STAGE_WHAT_DICT[resume_dict["stage"]]
    run_id = new_run_id()
    agent = agent_name(branch, run_id)
    exclude_list = exclude_list_of_run(project_dir, root_out_dir, name, exclude_list,
                                       build_doc, ask_doc)
    out_dir = make_agent_dir(root_out_dir, name, agent, project_dir, exclude_list,
                             build_doc)
    run_file = run_file_layout(root_out_dir, agent, name, run_id)
    logger.info("the merge on %s was settled, so the run carries on with %s, over %s%s",
                branch, what, gitcmd.short(union),
                f", for {' '.join(resume_dict['paths'])}" if resume_dict["paths"] else "")
    result_dict = {"engine": pick_engine(engine), "image": image or config.CONTAINER_IMAGE,
                   "base": resume_dict["base"], "branches": [branch],
                   "verdict": integrate_mod.SETTLE_VERDICT, "exit_status": 0,
                   "patch": None, "files": 0, "branch": branch,
                   "commit": resume_dict["merge"], "land": None, "land_why": "",
                   "land_command": "", "land_refuse": False}
    try:
        # The branch is put back on the commit the stages run over before anything is
        # started, which takes away what nobody settled above it
        _put_branch_back(project_dir, branch, union)
        result_dict.update(_run_stage_list(
            project_dir, root_out_dir, out_dir, name, agent, engine, image, union,
            resume_dict["base"], branch, request, exclude_list,
            first_stage=resume_dict["stage"], settle_commit=resume_dict["settled"],
            only_path_list=resume_dict["paths"]))
    finally:
        drop_the_copy(root_out_dir, name, agent)
    create_dict = result_dict.get(integrate_mod.stage_key(integrate_mod.TEST_CREATE_STAGE))
    if create_dict:
        result_dict.update({"patch": create_dict["patch"], "files": create_dict["files"]})
    if land:
        result_dict.update(_land_resume(project_dir, resume_dict, result_dict,
                                        run_file.land, branch,
                                        last_land_path(root_out_dir, name)))
    return result_dict


def _land_union_as_is(project_dir: str, root_out_dir: str, name: str, branch: str,
                      engine: str, image: str) -> dict | None:
    """Put a union whose stages were all settled into the working tree, as it is.

    No merge is made and no stage is run again. What nobody settled above the commit the
    branch is carried on from is taken off it (resume_head). What the working tree holds
    of the change is read by held_base(), and what is above that goes in (_land_resume).
    A working tree that holds none of it goes on only from the commit the change was
    written against or a commit after it, or with that commit one snapshot_commit()
    wrote of the working tree.

    Args:
        project_dir: The target repository, as an absolute path.
        root_out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        branch: The branch an earlier merge left, for which _settle_stage_of() names
            a stage.
        engine: The container engine, or "" to pick one.
        image: The image, or "" for the configured one.

    Returns:
        What _resume_stage() returns, with no stage's result and "gap" the cases the
        branch's commits record as not done (integrate.gap_list_in). None, with
        nothing changed, when the working tree holds part of the change, or holds none
        of it and does not go on from that commit, which the branch is then merged
        again for.
    """
    head, base = resume_head(project_dir, branch)
    run_id = new_run_id()
    run_file = run_file_layout(root_out_dir, agent_name(branch, run_id), name, run_id)
    held = held_base(project_dir, base, head, f"{run_file.land}.held")
    if not held:
        logger.info("the working tree holds part of what %s holds, so it is merged again "
                    "with the working tree", branch)
        return None
    if held == base and not gitcmd.is_snapshot(project_dir, base) \
            and not gitcmd.is_ancestor(project_dir, base,
                                       gitcmd.head_commit(project_dir)):
        logger.info("the working tree holds none of what %s holds and is not on %s or a "
                    "commit after it, so it is merged again with the working tree",
                    branch, gitcmd.short(base))
        return None
    _put_branch_back(project_dir, branch, head)
    result_dict = {"engine": pick_engine(engine), "image": image or config.CONTAINER_IMAGE,
                   "base": base, "branches": [branch],
                   "verdict": integrate_mod.SETTLE_VERDICT, "exit_status": 0,
                   "patch": None, "files": 0, "branch": branch,
                   "commit": integrate_mod.below_stage(project_dir, head),
                   "land": None, "land_why": "", "land_command": "", "land_refuse": False,
                   "settle_commit": head,
                   "gap": integrate_mod.gap_list_in(project_dir, base, head)}
    result_dict.update(_land_resume(project_dir, {"base": base, "settled": head},
                                    result_dict, run_file.land, branch,
                                    last_land_path(root_out_dir, name), held=held))
    return result_dict


def _land_resume(project_dir: str, resume_dict: dict, result_dict: dict,
                 patch_path: str, branch: str, last_path: str,
                 held: str | None = None) -> dict:
    """Put what the stages carried on with settled into the working tree.

    The run that settled the merge put in the change up to resume_dict["settled"]. When
    the working tree still holds it, or holds it up to a commit under it (the merge, or
    a commit of a stage above the merge, whose own landing was stopped) and what is
    above that commit applies to it, that goes in (held_base); when it holds none of it,
    the change from the commit the merge was written against goes in whole; when it
    holds part of it, nothing is written, the branch stays and the command reads the
    change, which no longer applies to the working tree whole. Once the change is in
    with no stage left open and no case left not done, the task branches the process
    that made the union made (_union_task_branch_list) are taken away where the union
    reaches them (drop_branch_under).

    Args:
        project_dir: The target repository, as an absolute path.
        resume_dict: stage_resume_dict()'s return value for the branch, of which
            "base" and "settled" are read.
        result_dict: What _resume_stage() has made of the stages so far.
        patch_path: Where the diff put into the working tree is written.
        branch: The branch the stages landed on.
        last_path: Where the copy of an applied diff is kept (last_land_path()).
        held: held_base()'s return value when the caller read it already. None reads
            it here.

    Returns:
        {"land", "land_why", "land_command", "land_refuse"}, as try_land() returns
        them.
    """
    base = held_base(project_dir, resume_dict["base"], resume_dict["settled"],
                     f"{patch_path}.held") if held is None else held
    if base:
        # Read before the union's own name is taken away
        tip = branch_tip(project_dir, branch)
        land_dict = _land_union(project_dir, base, result_dict, patch_path, branch,
                                last_path, earlier_base=resume_dict["base"])
        if land_dict["land"] and tip and not open_stage_key(result_dict) \
                and not result_dict.get("gap"):
            drop_branch_under(project_dir, _union_task_branch_list(project_dir, branch),
                              tip)
        return land_dict
    why = (f"the working tree holds part of what {gitcmd.short(resume_dict['settled'])} "
           f"put in before, so nothing was written: the change from "
           f"{gitcmd.short(resume_dict['base'])} no longer applies to it whole. What "
           f"the stages settled is on {branch}; read it, and put in by hand what the "
           f"working tree does not hold yet, with:")
    command = command_line("git", "-C", project_dir, *_READ_DIFF_TUPLE,
                           resume_dict["base"], _land_target(result_dict))
    logger.warning("%s %s", why, command)
    return {"land": None, "land_why": why, "land_command": command,
            "land_refuse": False}


def _union_task_branch_list(project_dir: str, branch: str) -> list[str]:
    """Return the task branches made by the process that made a union's branch.

    Args:
        project_dir: The target repository, as an absolute path.
        branch: The union's branch.

    Returns:
        The branches under the task branches' root whose name this tool made up
        (is_own_branch), which carry a run of the process the union's name carries
        (_run_process) and are not the branch of a tree (is_tree_branch), in git's
        order. Empty for a union whose name this tool did not make up.
    """
    if not is_own_branch(branch):
        return []
    process = _run_process(run_in(branch))
    return [task_branch for task_branch in _own_branch_list(project_dir, _TASK_BRANCH_ROOT)
            if is_own_branch(task_branch) and not is_tree_branch(task_branch)
            and _run_process(run_in(task_branch)) == process]


def _put_branch_back(project_dir: str, branch: str, commit: str) -> None:
    """Put a branch back on one of the commits under its tip, taking away what is above.

    Nothing is done when the branch is on that commit already.

    Args:
        project_dir: The target repository, as an absolute path.
        branch: The branch.
        commit: The commit it is put on.

    Raises:
        RuntimeError: When git does not move it.
    """
    tip = integrate_mod.commit_of(project_dir, branch)
    if tip == commit:
        return
    logger.info("the commits above %s on %s were left by stages nobody settled, so they "
                "are taken off the branch", gitcmd.short(commit), branch)
    with _branch_lock(project_dir):
        proc = gitcmd.run(project_dir, "branch", "-f", branch, commit)
    if proc.returncode != 0:
        raise RuntimeError(f"{branch} could not be put back on {gitcmd.short(commit)}: "
                           f"{proc.stderr.strip()}")


def integrate(project: str, output_dir: str, branch_list: list[str], engine: str = "",
              image: str = "", build_doc: bool = False, request: str = "",
              exclude_list: list[str] | None = None, base: str = "",
              resume: str = "", stage_after: bool = True,
              record_dict: dict | None = None,
              step_name: str = "combine the branches", land: bool = False,
              onto: str = "", gap_list: list[dict] | None = None,
              ask_doc: bool = False, union_branch: str = "",
              primary_ext: str = "") -> dict:
    """Combine the branches several runs left behind, in a container, and take it in.

    The container is started through _start_container(); it is asked to merge, not to
    change, and it starts at the commit the branches have in common, not at a branch's
    tip.

    A merge the machine still finds unsettled - conflict markers left in the text, calls
    that no longer resolve, tests that were passing and now fail - comes back with a
    non-zero exit status. It is still committed to its branch, where it can be read and
    finished by hand, and it is never put into the working tree. A run whose container
    left no patch that is taken in (_land_patch) lands on no branch: it is unsettled
    with a non-zero exit status whatever it said of itself, and, with land, "land_why"
    says so.

    The commit is a merge commit whose parents are the commit the change was written
    against and every branch that was combined: what those branches hold is reached from
    it, and merging one of them again changes nothing.

    With onto, the branch pointing at what the working tree holds is combined along with
    the others: the run settles what the tree holds against them too, and what is put
    into the tree is what the union adds on top of it. With resume as well, a merge
    that adds nothing on top of it lands too, as a commit that changes nothing, which
    reaches resume. The change lands on a branch named after the run, or on
    union_branch.

    The flow:
    1. A settled merge with a stage still ahead of it is carried on from by running
       that stage (_resume_stage), and nothing else is done
    2. The base and the branches are worked out, an earlier merge's branch first
    3. The merge runs in a container, and its patch is taken in on a branch
       (_land_patch)
    4. The stages that follow run over a settled merge (_run_stage_list), or the last
       stage the earlier branch settled is recorded on top of it
    5. What the gates passed is put into the working tree (_land_union)
    6. Why a merge was not settled is added to the result (_add_unsettle_why)

    Args:
        project: The path to the target repository.
        output_dir: Where the index and the analysis live.
        branch_list: The branches to combine, in the order they are to be merged.
        engine: The container engine, or "" to pick one.
        image: The image to run, or "" for the configured one.
        build_doc: Whether a first analysis made here, when nothing is built in the
            output directory yet, generates the design documents too (analyse_if_none).
        request: What the whole thing was asked for, which the commit is written with.
            Empty names the branches instead.
        exclude_list: The extra names kept out of the analysis.
        base: The commit the branches started from, when the caller knows it. Empty
            works it out as what the branches have in common, which takes two or more of
            them. With it one branch alone is combined, which is that branch held to the
            gates.
        resume: The branch an earlier merge left, or "". It is combined first, as
            one more branch, onto the first parent of its tip when base is not
            given. Combined with no other branch, a branch whose stages were all
            settled, or a union left unsettled by a merge of such a branch alone
            (_carry_stage_of), has no stage run again over the union: a commit that
            changes nothing and records the last of them as settled goes on top of
            it instead (_land_stage_record), and only the stages after that one whose
            setting is on run, over that commit. When that commit is not made, no
            stage runs, nothing goes into the working tree, the union's branch is
            taken away, and the result names resume as its branch and its tip as its
            commit, with no patch and "land_why" saying so.
        stage_after: Whether the stages whose setting is on (_stage_on_list) follow a
            settled merge.
        record_dict: What the commit records besides what the merging run said of
            itself, by the keys run_trailer_dict() reads; its "stage" is what the
            commit's subject starts with. Only a merge the run settled records all
            of it (_merge_record_dict). None records nothing more.
        step_name: The name the start and end of the container are logged under
            (progress.step).
        land: Whether what the gates passed is put into the working tree
            (_land_union), over the commit the union was written against, and the branch
            taken away when nothing is left on it.
        onto: The branch pointing at the commit the working tree makes (tree_branch).
            It is combined first and the union is written against its commit, read here
            before the container starts. Empty writes the union against base.
        gap_list: The cases the test tasks recorded as not done, with the tests they
            were recorded with. The run inside runs those tests again over the merge
            once it is settled, and the commit records the cases still not done in
            place of record_dict's "gap". None runs nothing.
        ask_doc: Whether a first analysis made here without build_doc asks on the
            terminal whether to make the design documents too (analyse_if_none).
        union_branch: The branch the union is committed to. One that is there already
            is made again at the commit the union is written against, and the agent is
            named after it. Empty names a new branch, and the agent, after the run
            itself.
        primary_ext: The language the container takes as the primary one in place of
            the index's (build_run_argv). Empty keeps the index's.

    Returns:
        What run() returns, with "branches" holding what was combined, "verdict"
        saying whether the run settled what the gates found, "head_block" where the
        run said what of the tests run over HEAD keeps any merge from being settled
        (integrate.head_block_list(), which "land_why" says what to change for),
        "merge_why" for a merge that landed on its branch with a non-zero exit status
        (the reason the run gave, or that it gave none), and "gap" and
        "gap_done" (the cases still not done and the ones whose test passes over the
        merge) when gap_list was passed and the run read them.

    Raises:
        RuntimeError: When fewer than two branches were passed with no base, when one
            of them is not in the repository, when they share no starting commit, when
            onto points at a commit behind the one they start from, when no engine is
            on PATH, or when the patch does not apply on the branch.
    """
    if len(branch_list) < 2 and not base and not resume:
        raise RuntimeError(f"there is nothing to combine: {len(branch_list)} branch(es) "
                           f"were passed. Pass --branch once for each branch to merge, "
                           f"or --resume with the branch an earlier merge left")
    project_dir, root_out_dir, name = _run_place(project, output_dir)

    # == Step 1: carry on with a stage still ahead ===========
    # A branch whose merge was settled and which a stage is still ahead of is carried
    # on from by running that stage, over what the branch holds
    resume_dict = stage_resume_dict(project_dir, resume) \
        if stage_after and resume and not branch_list else {}
    if resume_dict:
        return _resume_stage(project_dir, root_out_dir, name, resume, resume_dict,
                             engine, image, build_doc, request, exclude_list, land,
                             ask_doc)

    # == Step 2: the base and the branches ===================
    run_id = new_run_id()
    on_branch = union_branch or f"{_BRANCH_PREFIX}{run_id}"
    other_list = list(branch_list)
    carry_stage = ""
    # An earlier merge's branch is combined as one more branch, onto the commit it was
    # written against, and its tree and every branch it reaches come along. What nobody
    # settled above it is taken off the branch first (resume_head)
    if resume:
        head, resume_base = resume_head(project_dir, resume)
        # Combined alone, a branch whose stages were all settled has none run again
        carry_stage = "" if other_list else _carry_stage_of(project_dir, resume)
        _put_branch_back(project_dir, resume, head)
        base = base or resume_base
    base = base or integrate_mod.merge_base(project_dir,
                                            other_list + ([onto] if onto else []))
    # What the run writes is against what the working tree holds, not against the commit
    # the clone starts at. The same commit is passed into the container: the union is
    # written against what was read here
    base_commit = branch_tip(project_dir, onto) if onto else base
    if onto and integrate_mod.union_base(project_dir, base, onto, base_commit) != base_commit:
        raise RuntimeError(
            f"the working tree is at {gitcmd.short(base_commit)}, behind "
            f"{gitcmd.short(base)}, which the branches start from, so the union would be "
            f"written against a commit the working tree does not hold. Check out "
            f"{gitcmd.short(base)} or a commit after it, then run again")
    # The earlier merge's branch goes first; the run inside is told which one it is,
    # combines it first and reads its record
    branch_list = [resume, *other_list] if resume else other_list
    branch_commit_list = [branch_tip(project_dir, branch) or branch
                          for branch in branch_list]
    logger.info("combining %s onto %s", ", ".join(branch_list), gitcmd.short(base))

    # == Step 3: the merge in a container ====================
    agent = agent_name(union_branch, run_id)
    exclude_list = exclude_list_of_run(project_dir, root_out_dir, name, exclude_list,
                                       build_doc, ask_doc)
    knowledge.ask_cobol_dialect(store.index_db_path(root_out_dir, name))
    out_dir = make_agent_dir(root_out_dir, name, agent, project_dir, exclude_list,
                             build_doc)
    try:
        run_file = run_file_layout(root_out_dir, agent, name, run_id)
        if gap_list is not None:
            write_across(inside_path(run_file.gap, run_file.out_dir), gap_list,
                         "the cases whose tests are run again")

        with progress.step(step_name):
            engine_path, image_name, status = _start_container(
                engine, image, project_dir, root_out_dir, out_dir, name, agent, base,
                run_file, run_id,
                integrate_argv(name, other_list, onto, run_file.outcome_inside,
                               resume=resume, onto_commit=base_commit if onto else "",
                               request=request,
                               gap_in=run_file.gap_inside if gap_list is not None else ""),
                exclude_list, tip_list=branch_commit_list, primary_ext=primary_ext)

        # What the run made of the merge, which the commit carries as trailers
        outcome_dict = _read_outcome(
            run_file.outcome, run_file.out_dir,
            _MERGE_KIND if gap_list is None else _TEST_MERGE_KIND)
        result_dict = {"engine": engine_path, "image": image_name, "base": base_commit,
                       "branches": list(branch_list),
                       "verdict": outcome_dict.get("verdict", ""),
                       "land": None, "land_why": "", "land_command": "",
                       "land_refuse": False}
        # What keeps any merge over the same HEAD from being settled, as the run said
        head_block_list = [block for block in outcome_dict.get("head_block") or []
                           if block in _HEAD_BLOCK_TEXT_DICT]
        if head_block_list:
            result_dict["head_block"] = head_block_list
        commit_request = request or f"combine {', '.join(branch_list)}"
        record_dict = dict(record_dict or {})
        if gap_list is not None and isinstance(outcome_dict.get("gap"), list):
            _take_gap_done(record_dict, result_dict, outcome_dict)

        def message_of(meta_dict: dict) -> str:
            """Return the message of the merge's commit, from the patch's summary."""
            own_dict = _merge_record_dict(record_dict, outcome_dict, status, meta_dict)
            return commit_message(
                commit_request,
                run_trailer_dict(meta_dict, {**outcome_dict, **own_dict}, request),
                prefix=own_dict.get("stage") or _MERGE_PREFIX)

        result_dict.update(_land_patch(
            project_dir, run_file, base_commit, on_branch, status, True, message_of,
            parent_list=_parent_list_of(outcome_dict, branch_commit_list),
            reset=bool(union_branch), keep_empty=bool(onto and resume), engine=engine))
        # One exit status stands for the merge from here on: what _land_patch() made of
        # the container's own and the one its summary names
        status = result_dict["exit_status"]
        # A run that wrote no patch settled nothing, whatever it said of itself
        if result_dict.get("no_patch"):
            result_dict["verdict"] = integrate_mod.UNSETTLE_VERDICT

        # == Step 4: the stages that follow ======================
        # The stages that follow run over a merge that was settled and landed, each in
        # a container of its own, and each lands as a commit of its own on top of it
        is_carry_stop = False
        if stage_after and not carry_stage and status == 0 and result_dict.get("commit"):
            result_dict.update(_run_stage_list(
                project_dir, root_out_dir, out_dir, name, agent, engine, image,
                result_dict["commit"], base_commit, on_branch, request, exclude_list))
        elif carry_stage and status == 0 and result_dict.get("commit"):
            # The last stage the earlier branch settled is recorded on top of the union
            tip = _land_stage_record(
                project_dir, run_file.worktree, on_branch,
                {"stage": carry_stage, "verdict": integrate_mod.SETTLE_VERDICT},
                request or f"{carry_stage} {gitcmd.short(head)}", request, engine)
            # The stages after it whose setting is on run over that commit
            later_tuple = integrate_mod.STAGE_TUPLE[
                integrate_mod.STAGE_TUPLE.index(carry_stage) + 1:]
            if not tip:
                # A union without the record is taken away: no stage runs over it,
                # nothing goes into the working tree, and the branch carried on from
                # is the one named again
                _delete_branch(project_dir, on_branch)
                is_carry_stop = True
                result_dict.update({
                    "branch": resume, "commit": head, "patch": None, "files": 0,
                    "land_why": f"what {_STAGE_WHAT_DICT[carry_stage]} settled could "
                                f"not be recorded on the union, so no stage was run over "
                                f"it and nothing was put into the working tree. "
                                f"{on_branch} was taken away and {resume} is as it was"})
                logger.warning("%s", result_dict["land_why"])
            elif stage_after and set(later_tuple) & set(_stage_on_list()):
                result_dict.update(_run_stage_list(
                    project_dir, root_out_dir, out_dir, name, agent, engine, image, tip,
                    base_commit, on_branch, request, exclude_list,
                    first_stage=later_tuple[0]))
            else:
                result_dict["gap"] = integrate_mod.gap_list_in(project_dir, base_commit,
                                                               tip)

        # == Step 5: the union into the working tree =============
        # A merge the run itself could not settle is not put in the working tree
        if land and status == 0 and result_dict.get("commit") and not is_carry_stop:
            result_dict.update(_land_union(project_dir, base_commit, result_dict,
                                           run_file.land, on_branch,
                                           last_land_path(root_out_dir, name)))
    finally:
        drop_the_copy(root_out_dir, name, agent)

    # == Step 6: why a merge was not settled =================
    _add_unsettle_why(result_dict, outcome_dict, status, land, on_branch)
    return result_dict


def _take_gap_done(record_dict: dict, result_dict: dict, outcome_dict: dict) -> None:
    """Take the cases a merging run found done over the settled merge from its outcome.

    The cases still not done replace the "gap" of the commit's record and are the
    result's "gap"; the ones found done are the result's "gap_done" and are added to the
    record's, and each is logged. They are not recorded as not done again.

    Args:
        record_dict: What the merge's commit records, changed in place.
        result_dict: integrate()'s result so far, changed in place.
        outcome_dict: What the merging run said, as _read_outcome() reads it, with a
            list under "gap".
    """
    record_dict["gap"] = result_dict["gap"] = outcome_dict["gap"]
    result_dict["gap_done"] = outcome_dict.get("gap_done") or []
    record_dict["gap_done"] = [*(record_dict.get("gap_done") or []),
                               *result_dict["gap_done"]]
    for entry_dict in result_dict["gap_done"]:
        logger.info("%s is done over the combined tests: the test it was "
                    "recorded with passes there, so it is no longer counted",
                    entry_dict["what"])


def _add_unsettle_why(result_dict: dict, outcome_dict: dict, status: int, land: bool,
                      on_branch: str) -> None:
    """Add why a merge with a non-zero exit status was not settled to its result.

    "merge_why" is what the run said of a merge that landed on its branch and was not
    settled. With land, "land_why" says that nothing was put into the working tree, and
    is logged.

    Args:
        result_dict: integrate()'s result, changed in place.
        outcome_dict: What the merging run said, as _read_outcome() reads it.
        status: The exit status that stands for the merge.
        land: Whether what the gates passed was to be put into the working tree.
        on_branch: The branch the union lands on.
    """
    # What the run said of a merge that landed on its branch and was not settled
    if status != 0 and result_dict.get("commit"):
        result_dict["merge_why"] = outcome_dict.get("why") or "the run said nothing about why"
    if land and status != 0:
        # A branch is named only when the merge landed on one
        result_dict["land_why"] = (
            f"the merge was not settled ({status_text(status)}), so nothing was put into "
            f"the working tree. What it left is on {on_branch}: {result_dict['merge_why']}"
            f"{_head_block_text(result_dict.get('head_block'), on_branch)}"
            if result_dict.get("commit") else
            f"the merging run left no patch that is taken in, so there is no union and "
            f"nothing was put into the working tree, and the branches it was to "
            f"combine are as they were: {result_dict['no_patch_why']}"
            if result_dict.get("no_patch") else
            f"the merging run ended ({status_text(status)}) and left no patch to take "
            f"in, so there is no union and nothing was put into the working tree. The "
            f"branches it was to combine are as they were")
        logger.warning("%s", result_dict["land_why"])


def integrate_into_tree(project: str, output_dir: str, branch_list: list[str],
                        engine: str = "", image: str = "", build_doc: bool = False,
                        request: str = "", exclude_list: list[str] | None = None,
                        resume: str = "") -> dict:
    """Combine branches into the working tree: integrate() written against the tree.

    The commit the working tree makes is pointed at by a branch of the run's own
    (tree_branch) and combined along with the branches, as the first of them: the union
    holds what the tree holds and what is put into the tree is what the union adds on
    top of it. A branch an earlier merge left whose stages alone are left to run is
    carried on from over what that merge put into the tree, and no branch of the tree is
    made for it. One whose stages were all settled (_settle_stage_of) is put into the
    tree as it is, with no merge and no stage run again, unless the working tree holds
    part of it, or holds none of it away from the commit it was written against
    (_land_union_as_is); then it is merged again with no stage run over the union but
    the ones after the last settled whose setting is on (integrate), and so is a union
    such a merge left unsettled. For a merge that was settled, the commit the tree makes
    now is written as what the last commit of the earlier merge's branch the working
    tree holds became (_held_union_commit): a commit above it that takes away what is
    under it is merged as a change. That holds as well where the earlier merge was
    written against a commit of the working tree, the commit found being one above it.
    Where the working tree holds none of those commits and the earlier merge was written
    against a commit of the working tree (gitcmd.is_snapshot), the commit the tree makes
    now is written as what that commit became: what is on disk now stands in place of
    what was on disk then. The new union's merge records the earlier union the commit of
    the tree stands on ("stand_on", which integrate.union_part_list() reads the cases
    of): the commit _held_union_commit() found, or else the one the earlier merge
    recorded, where integrate.union_part_list() reads the working tree as holding it.
    A branch an earlier merge left is taken away once the working tree holds what it
    holds: the new union reaches it and was put in, the new union reaches it and was
    kept out of the working tree for the paths it touches (land_refuse; the new union
    holds what it held, and its record says why and how to put it in by hand), or the
    merge got through and the branch holds a result the working tree holds
    (_drop_resume_union). The runs the branches belong to are recorded first, before
    any of them is read (_mark_run_of_branch).

    Args:
        As integrate() takes them.

    Returns:
        What integrate() returns.

    Raises:
        RuntimeError: As integrate() and tree_branch() raise.
    """
    project_dir, root_out_dir, name = _run_place(project, output_dir)
    # Recorded before they are read
    _mark_run_of_branch(project_dir, root_out_dir, name, [resume, *branch_list],
                        new_run_id())
    # The union carried on from for its cases, read before any run changes what the
    # working tree holds
    resume_list = [resume] if resume else []
    _hold_stand_for(project_dir, root_out_dir, name, engine, resume_list,
                    gap_union_list_of(project_dir, resume_list, request), request)
    if resume and not branch_list and stage_resume_dict(project_dir, resume):
        return integrate(project, output_dir, [], engine=engine, image=image,
                         build_doc=build_doc, request=request, exclude_list=exclude_list,
                         resume=resume, land=True)
    # A union whose stages were all settled goes in as it is where the working tree
    # holds none of it, or a part up to one of its commits
    if resume and not branch_list and _settle_stage_of(project_dir, resume):
        result_dict = _land_union_as_is(project_dir, root_out_dir, name, resume, engine,
                                        image)
        if result_dict is not None:
            return result_dict
    head, earlier = resume_head(project_dir, resume) if resume else ("", "")
    stand_on = ""
    if earlier:
        stand_on = _held_union_commit(project_dir, resume)
        if stand_on:
            earlier = stand_on
        elif gitcmd.is_snapshot(project_dir, earlier):
            part_list = integrate_mod.union_part_list(project_dir, earlier, head)
            stand_on = part_list[-2]["tip"] \
                if len(part_list) > 1 and part_list[-2]["held"] else ""
        else:
            earlier = ""
    with tree_branch(project_dir, new_run_id(), earlier, engine=engine) as (_, branch):
        result_dict = integrate(project, output_dir, branch_list, engine=engine,
                                image=image, build_doc=build_doc, request=request,
                                exclude_list=exclude_list, resume=resume,
                                record_dict={"stand_on": stand_on}, land=True,
                                onto=branch)
    _drop_resume_union(project_dir, root_out_dir, name, engine, resume, result_dict)
    return result_dict


def _drop_resume_union(project_dir: str, out_dir: str, name: str, engine: str,
                       resume: str, result_dict: dict) -> None:
    """Delete the union a merge was carried on from, once the working tree holds it.

    Nothing is deleted unless the union's name is one this tool made up
    (is_own_branch). The branch goes
    - when the new union was put into the working tree, or kept out of it for the
      paths it touches (land_refuse), and reaches it (drop_branch_under), whatever
      is left on the new union's own branch, and
    - when the merge got through - exit status 0, no stage left unsettled, no case
      its tests found not done, and its union put into the working tree or no commit
      left - and the branch holds a result the working tree holds
      (_result_in_tree_list), which is deleted as _drop_resume_branch_list() deletes
      one.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.
        engine: The container engine, or "" for the configured one.
        resume: The branch the earlier merge left, or "".
        result_dict: integrate()'s return value for the merge carried on from it.
    """
    if not resume or not is_own_branch(resume):
        return
    is_land = bool(result_dict.get("commit") and result_dict.get("land"))
    if is_land or (result_dict.get("commit") and result_dict.get("land_refuse")):
        drop_branch_under(project_dir, [resume], result_dict["commit"])
    if (is_land or not result_dict.get("commit")) and not open_stage_key(result_dict) \
            and not result_dict.get("gap") and not result_dict.get("exit_status") \
            and not land_status(result_dict):
        _drop_resume_branch_list(project_dir, out_dir, name, engine,
                                 _result_in_tree_list(project_dir, [resume]))


def _parent_list_of(outcome_dict: dict, branch_commit_list: list[str]) -> list[str]:
    """Return the commits a merge commit is given as its parents besides its base.

    Args:
        outcome_dict: What the merging run said, as _read_outcome() reads it.
        branch_commit_list: The commits the branches pointed at before the container
            started, in the order they were passed, each the branch's name where it is
            not a local branch.

    Returns:
        The outcome's "parents" (the commits the run merged) when it holds one full
        commit hash for each branch, branch_commit_list otherwise.
    """
    parent_list = outcome_dict.get("parents")
    if isinstance(parent_list, list) and len(parent_list) == len(branch_commit_list) \
            and all(isinstance(parent, str) and integrate_mod.OBJECT_NAME.match(parent)
                    for parent in parent_list):
        return parent_list
    return branch_commit_list


def _read_outcome(path: str, out_dir: str, kind: str) -> dict:
    """Read what a merging run or a stage said about how it ended, or {} for nothing.

    A run that was killed before it wrote the file leaves nothing here.

    Args:
        path: The outcome file the container wrote.
        out_dir: The agent's copy of the output directory the file is in.
        kind: The kind of run that wrote it, a key of _RECORD_KIND_DICT.

    Returns:
        What the file holds as _record_dict_of() reads it for that kind - for a merge
        {"verdict", "why", "did", "tests_changed", "ended", "decisions", "parents",
        "separator", "no_merge", "head_block"} - or {} for a file that is not there,
        cannot be read (_json_across), which is logged as a warning, or holds a value of
        another shape.

    Raises:
        RuntimeError: When the file is reached through a link that leads out of
            out_dir (inside_path).
    """
    try:
        outcome_obj = _json_across(path, out_dir)
    except FileNotFoundError as e:
        logger.info("the merging run left no account of how it ended in %s (%s)",
                    path, e)
        return {}
    except (OSError, ValueError) as e:
        logger.warning("what a run wrote of how it ended in %s is not read (%s), so "
                       "the run counts as one that left no account of it", path, e)
        return {}
    return _record_dict_of(outcome_obj, path, kind)


def warm(project: str, output_dir: str, engine: str = "", image: str = "",
         exclude_list: list[str] | None = None, live_dep_file: bool = False,
         run_id: str = "", once: bool = False, cache: str = "",
         before_fill: Callable[[], object] | None = None, base: str = "",
         agent: str = "", dep_commit: str = "",
         fill_ext_list: Sequence[str] = (), fill_own: bool = True) -> dict:
    """Fill the package cache, in a container with the engine's default network.

    Steps:
    1. Work out where the cache goes and make it, and, for a filling no agent asked
       for, make a copy of the output directory of its own for the container
       (make_agent_dir), which is taken away again when the filling ends
    2. Run the four steps in a container of its own: clone the copy of the
       repository's history (make_clone_source), check the clone out at the base
       commit, copy in the manifests of a change that is running or of dep_commit,
       and run the test command once over the clone, then the default test command of
       each language of fill_ext_list
    3. Add the filling's set of dependency files, image and fill_ext_list to the mark
       beside the cache (fill_mark_path) when the filling ran through

    Nothing a model wrote runs here besides those manifests, and no key of this tool
    takes part: the proxy variables, with the user and the password a proxy's URL
    carries, are what the container is given (_route_argv).

    Args:
        project: The target repository. A relative path is resolved from here.
        output_dir: Where the index and the cache live. A relative path is resolved from
            here.
        engine: The container engine. Empty picks one.
        image: The image to run. Empty uses config.CONTAINER_IMAGE.
        exclude_list: The directory and file names kept out of the analysis, passed as
            BUILD_EXCLUDE. Unused by the test command itself.
        live_dep_file: Whether to take the dependency files of a change that is running
            from its own copy. Only those files are taken; the code that runs is the
            clone's, and nothing else a change wrote runs with a network. False, and a
            copy that is not there, both leave the clone as it is.
        run_id: The run whose copy those files are taken from, as new_run_id() names it.
            Empty takes no file from any running change.
        once: Whether a cache whose mark already holds this filling's set of dependency
            files and image is left as it is.
        cache: The package cache to fill, as an absolute path. Empty works it out from
            output_dir (cache_dir).
        before_fill: Called under the lock once it is settled that a container is
            started, before it is; a filling that once leaves out does not call it. An
            exception it raises ends the filling before any container starts.
        base: The commit the clone is checked out at. Empty makes the commit of the
            working tree here: the dependency files as they are on disk are what the
            cache is filled for. A branch of the filling's own points at the commit
            while the filling goes (tree_branch).
        agent: The agent the filling is for, as agent_name() names it: the container is
            given that agent's copy of the output directory as /out, where anything the
            run-tests inside it makes again lands, and run_id's clone is looked for
            among that agent's. Empty makes an agent of the filling's own.
        dep_commit: A commit whose dependency files are copied over the clone's: the
            regular files of it that base does not hold with the same content
            (_dep_change_dict). Only those files are taken; the code that runs is
            base's. Empty takes none. With it, no file is taken from a running change.
        fill_ext_list: The languages whose default test commands are run over the clone
            as well, as new_test_ext_list() returns them. Empty runs none.
        fill_own: Whether the test command of the index is run over the clone. False
            runs only those of fill_ext_list.

    Returns:
        {"engine", "image", "base", "cache" (where it was filled), "exit_status",
        "skip"}. A filling that once left out has "skip" True, exit_status 0, and
        "engine" and "image" empty.

    Raises:
        RuntimeError: When project is not a git repository, when there is no engine,
            when git cannot list the files of the running change's copy, or when the
            copy of the repository's history cannot be made; and as make_cache(),
            _write_dep_file() and _dep_fingerprint() raise.
        OSError: When the filling's scratch directory cannot be made (make_scratch).
    """
    # == Step 1: work out where the cache goes ===============
    project_dir, out_dir, name = _run_place(project, output_dir)
    cache = make_cache(cache or cache_dir(out_dir, name))
    # A clone of its own, named after this filling and not after the change, beside the
    # change's clone that the change's container still holds during a refill
    fill_id = new_run_id()
    with contextlib.ExitStack() as stack:
        base = stack.enter_context(
            tree_branch(project_dir, fill_id, commit=base, engine=engine))[0]
        if not agent:
            agent = agent_name("", fill_id)
            stack.callback(drop_the_copy, out_dir, name, agent)
            make_agent_dir(out_dir, name, agent, project_dir, exclude_list, False)
        return _warm_from(project_dir, out_dir, agent, name, base, cache, fill_id,
                          engine, image, exclude_list, live_dep_file, run_id, once,
                          before_fill, dep_commit, fill_ext_list, fill_own)


def _warm_from(project_dir: str, out_dir: str, agent: str, name: str, base: str,
               cache: str, fill_id: str, engine: str, image: str,
               exclude_list: list[str] | None, live_dep_file: bool, run_id: str,
               once: bool, before_fill: Callable[[], object] | None,
               dep_commit: str = "", fill_ext_list: Sequence[str] = (),
               fill_own: bool = True) -> dict:
    """Fill the package cache from one commit, as warm() says.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        agent: The agent whose copy of the output directory the container is given.
        name: The repository's directory name.
        base: The commit the clone is checked out at.
        cache: The package cache, as an absolute path, already made.
        fill_id: This filling, as new_run_id() names it.
        engine, image, exclude_list, live_dep_file, run_id, once, before_fill,
            dep_commit, fill_ext_list, fill_own: As warm() takes them.

    Returns:
        What warm() returns.
    """
    fill_file = run_file_layout(out_dir, agent, name, fill_id)
    scratch = make_scratch(fill_file.warm_scratch)

    # == Step 2: run the four steps in a container ===========
    # A link in the place of the running change's working tree is not taken as one
    live = work_tree(out_dir, agent, name, run_id) \
        if live_dep_file and run_id and not dep_commit else ""
    if live_dep_file and not dep_commit and not (live and _is_real_dir(live)):
        logger.info("no change is running%s, so the clone's own build files are what "
                    "the cache is filled for", f" at {live}" if live else "")
        live = ""
    # One filling at a time, under a lock on the cache itself
    mark = fill_mark_path(cache)
    try:
        with filelock.hold_lock(cache, "the shared package cache"):
            # Read under the lock: runs started at once fill the cache once between
            # them, not once each
            if dep_commit:
                # Written by this machine into a directory of the filling's own, which
                # the container is given in the running change's place
                live = os.path.join(scratch, _DEP_DIR, name)
                dep_path_list = _write_dep_file(project_dir, base, dep_commit, live)
                live = live if dep_path_list else ""
            else:
                dep_path_list = live_dep_path_list_of(
                    project_dir, live, build_output_tuple_of(out_dir, name),
                    store.primary_ext_of(store.index_db_path(out_dir, name))) \
                    if live else []
            fingerprint = _dep_fingerprint(project_dir, base, live, dep_path_list,
                                           _image_id(engine, image), fill_ext_list)
            if once and fingerprint in _fill_mark_set(mark):
                logger.info("%s already holds what these dependency files fetch in "
                            "this image, so no container is started", cache)
                return {"engine": "", "image": "", "base": base, "cache": cache,
                        "exit_status": 0, "skip": True}
            if before_fill is not None:
                before_fill()
            logger.info("filling %s from %s at %s%s%s, in a container with a network "
                        "and no key of this tool%s", cache, project_dir,
                        gitcmd.short(base),
                        f", with the build files of {gitcmd.short(dep_commit)}"
                        if live and dep_commit
                        else ", with the running change's build files" if live else "",
                        f", and the tests of {', '.join(fill_ext_list)} as well"
                        if fill_ext_list else "",
                        ": nothing has filled it for these dependency files in this "
                        "image yet" if once else "")
            # What is mounted is the directory this machine made, which holds the clone
            mount = MountLayout(src=make_clone_source(project_dir, scratch),
                                out=agent_dir(out_dir, agent), scratch=scratch,
                                cache=cache, live=os.path.dirname(live))
            engine_path, image_name, status = _fill_cache(
                mount, base, name, engine, image, exclude_list, fill_id, dep_path_list,
                fill_file.stdout, build_output_tuple_of(out_dir, name), fill_ext_list,
                fill_own)
            if status == 0:
                with open(mark, "a", encoding="utf-8") as handle:
                    handle.write(f"{fingerprint}\n")
    finally:
        remove_scratch(scratch)
    return {"engine": engine_path, "image": image_name, "base": base,
            "cache": cache, "exit_status": status, "skip": False}


# The number of the layout the tools' fetched packages are kept in. A fingerprint made
# under another number does not match, and a cache of another layout is filled again
_CACHE_LAYOUT = 2


def _dep_fingerprint(project_dir: str, base: str, live: str,
                     dep_path_list: list[str], image_id: str,
                     fill_ext_list: Sequence[str] = ()) -> str:
    """Return one line for the dependency files, image and languages of a filling.

    It follows what sandbox/warm.sh makes of the clone: the files of base whose names
    match DEP_FILE_TUPLE, each overwritten by the file of the same path in live. Each file
    is taken as its git blob id: the same files give the same line whatever commit base
    is. The files of dep_path_list are read up to tmpfile.READ_MAX_BYTE bytes between
    them.

    Args:
        project_dir: The target repository, as an absolute path.
        base: The commit the clone is checked out at.
        live: The directory the dependency files are taken from, as an absolute path, or
            "" for none.
        dep_path_list: The dependency files taken from live, relative to it.
        image_id: The image the filling runs, as _image_id() returns it.
        fill_ext_list: The languages whose default test commands the filling runs as
            well (warm's fill_ext_list).

    Returns:
        The sha256 of _CACHE_LAYOUT, image_id and the sorted (path, blob id) pairs, with
        fill_ext_list after them when it is not empty, in hex.

    Raises:
        RuntimeError: When a file of dep_path_list cannot be read as a regular file
            (tmpfile.read_file), or the files hold more than tmpfile.READ_MAX_BYTE
            bytes between them.
    """
    blob_dict = _dep_blob_dict(project_dir, base)
    remain_byte = tmpfile.READ_MAX_BYTE
    for path in dep_path_list:
        try:
            content = tmpfile.read_file(os.path.join(live, path), remain_byte)
        except ValueError:
            raise RuntimeError(f"the dependency files of the running change at {live} "
                               f"hold more than {tmpfile.READ_MAX_BYTE} bytes between "
                               f"them, so they are not read") from None
        except OSError as e:
            raise RuntimeError(f"the dependency file {path} of the running change at "
                               f"{live} could not be read: {e}") from None
        remain_byte -= len(content)
        header = f"blob {len(content)}\0".encode("ascii")
        blob_dict[path] = hashlib.sha1(header + content).hexdigest()
    pair_list = [_CACHE_LAYOUT, image_id, sorted(blob_dict.items())]
    if fill_ext_list:
        pair_list.append(list(fill_ext_list))
    pair_text = json.dumps(pair_list)
    return hashlib.sha256(pair_text.encode("utf-8")).hexdigest()


# What a path handed to a filling's container, one path to a line, does not hold
_CONTROL_CHAR = re.compile(r"[\x00-\x1f\x7f]")


def _is_line_path(path: str, where: str) -> bool:
    """Say whether a dependency file's path can be handed on as one line.

    A path holding a control character, a line break among them, cannot, and is
    logged.

    Args:
        path: The path, relative to the repository.
        where: What holds the file, for the log.

    Returns:
        True when the path holds no control character.
    """
    if not _CONTROL_CHAR.search(path):
        return True
    logger.warning("the dependency file %r of %s holds a control character in its name, "
                   "so no filling takes it", path, where)
    return False


def _dep_blob_dict(project_dir: str, commit: str, regular_only: bool = False
                   ) -> dict[str, str]:
    """Return the dependency files a commit holds, each with its git blob id.

    Args:
        project_dir: The target repository, as an absolute path.
        commit: The commit.
        regular_only: Whether the entries that are not regular files (a link, a
            repository of its own) are left out.

    Returns:
        {path: blob id} for the files whose names match DEP_FILE_TUPLE and hold no
        control character (_is_line_path). Empty when the commit cannot be read.
    """
    blob_dict: dict[str, str] = {}
    tree_text = gitcmd.run(project_dir, "ls-tree", "-r", "-z", commit).stdout
    for entry in tree_text.split("\0"):
        meta, _, path = entry.partition("\t")
        if not path or not is_dep_file(path, DEP_FILE_TUPLE) \
                or not _is_line_path(path, gitcmd.short(commit)):
            continue
        mode, _, blob = meta.split()
        if regular_only and not mode.startswith(_REGULAR_MODE_PREFIX):
            continue
        blob_dict[path] = blob
    return blob_dict


def _dep_change_dict(project_dir: str, base: str, dep_commit: str) -> dict[str, str]:
    """Return the dependency files of one commit that another does not hold as they are.

    Args:
        project_dir: The target repository, as an absolute path.
        base: The commit the files are compared against.
        dep_commit: The commit whose files are returned.

    Returns:
        {path: blob id} for the regular dependency files of dep_commit that base holds
        with another content or not at all.
    """
    base_dict = _dep_blob_dict(project_dir, base)
    return {path: blob
            for path, blob in _dep_blob_dict(project_dir, dep_commit, True).items()
            if base_dict.get(path) != blob}


def _write_dep_file(project_dir: str, base: str, dep_commit: str,
                    directory: str) -> list[str]:
    """Write the dependency files of a commit that base does not hold as they are.

    Args:
        project_dir: The target repository, as an absolute path.
        base: The commit a filling's clone is checked out at.
        dep_commit: The commit whose dependency files are written (_dep_change_dict).
        directory: Where they are written, each at its path in the repository. It is
            made here.

    Returns:
        The paths written, relative to directory, sorted. Empty writes nothing.

    Raises:
        RuntimeError: When git cannot read one of the files, or it cannot be written.
    """
    path_list = []
    for path, blob in sorted(_dep_change_dict(project_dir, base, dep_commit).items()):
        proc = gitcmd.run(project_dir, "cat-file", "blob", blob, text=False)
        if proc.returncode:
            raise RuntimeError(f"the dependency file {path} of {gitcmd.short(dep_commit)}"
                               f" could not be read: {gitcmd.shown_text(proc.stderr)}")
        target = os.path.join(directory, path)
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as handle:
                handle.write(proc.stdout)
        except OSError as e:
            raise RuntimeError(f"the dependency file {path} of {gitcmd.short(dep_commit)}"
                               f" could not be written under {directory}: {e}") from None
        path_list.append(path)
    return path_list


def live_dep_path_list_of(project_dir: str, live: str,
                          output_tuple: tuple[str, ...] = (),
                          primary_ext: str | None = None) -> list[str]:
    """Return the dependency files a filling takes from the running change's copy.

    The files git carries or would carry there: tracked, or new and neither ignored nor
    matched by output_tuple. Regular files only, none reached through a link, in the
    path or at its end, that leads out of the copy, and none whose name holds a control
    character (_is_line_path). With primary_ext, what the tools of the other languages
    write beside their project files in the copy is left out as well
    (tools.edit.output_place_list_of).

    The copy is read as a work tree of the target repository's own git directory. The
    copy's own git directory is not read. Each listing is given
    CONTAINER_COMMAND_TIMEOUT_SEC seconds and _MAX_LIST_BYTE bytes, and of what git
    lists only the paths whose names match DEP_FILE_TUPLE are held.

    Args:
        project_dir: The target repository, as an absolute path.
        live: The running change's copy, as an absolute path.
        output_tuple: .gitignore patterns of what the language's tools write, left out
            of the new files (build_output_tuple_of()).
        primary_ext: The index's primary language, "" when it has none. None leaves
            the other languages' patterns out of this.

    Returns:
        Paths relative to live, sorted.

    Raises:
        RuntimeError: When git cannot list the files of live, does not end within
            the time it is given, or lists more than it is given to.
    """
    def file_list_without(pattern_list: Sequence[str]) -> list[str]:
        """Return the dependency files git lists in live, leaving out pattern_list."""
        proc = gitcmd.run_entry(
            project_dir, "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false",
            f"--work-tree={live}", "ls-files", "-z", "--cached", "--others",
            "--exclude-standard", *[f"--exclude={pattern}" for pattern in pattern_list],
            keep=lambda path: is_dep_file(path, DEP_FILE_TUPLE),
            max_byte=_MAX_LIST_BYTE,
            timeout_sec=config.CONTAINER_COMMAND_TIMEOUT_SEC or None)
        if proc.returncode:
            raise RuntimeError(f"git could not list the files of the running change at "
                               f"{live}: {proc.stderr.strip()}")
        return proc.stdout

    path_list = file_list_without(output_tuple)
    if primary_ext is not None:
        other_pattern_list = edit_mod.output_pattern_list_of(edit_mod.output_place_list_of(
            _other_output_ext_list(primary_ext), path_list, live))
        if other_pattern_list:
            path_list = file_list_without([*output_tuple, *other_pattern_list])
    real_live = os.path.realpath(live)
    return sorted({path for path in path_list
                   if is_dep_file(path, DEP_FILE_TUPLE)
                   and _is_line_path(path, "the running change")
                   and not os.path.islink(os.path.join(live, path))
                   and os.path.isfile(os.path.join(live, path))
                   and os.path.realpath(os.path.join(live, path))
                   == os.path.join(real_live, os.path.normpath(path))})


def _other_output_ext_list(primary_ext: str) -> list[str]:
    """Return one LANG_DICT key for every other language whose tools write build output.

    The primary language is left out.

    Args:
        primary_ext: The primary language's LANG_DICT key, or "" for none.

    Returns:
        The keys, in LANG_DICT's order, one per language (the first of its
        extensions), without the primary language's.
    """
    primary_config = LANG_DICT.get(primary_ext)
    ext_list, pick_config_list = [], [primary_config]
    for ext, lang_config in LANG_DICT.items():
        if lang_config.build_output_tuple and not any(
                lang_config is pick_config for pick_config in pick_config_list):
            ext_list.append(ext)
            pick_config_list.append(lang_config)
    return ext_list


def _fill_mark_set(mark: str) -> set[str]:
    """Return the sets of dependency files and images the cache has been filled for.

    Args:
        mark: The mark file beside the cache, as an absolute path (fill_mark_path).

    Returns:
        One _dep_fingerprint() line per filling that ran through, or an empty set when
        nothing has filled the cache yet.
    """
    try:
        with open(mark, encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}
    except FileNotFoundError:
        return set()


def _commit_path_list(project_dir: str, commit: str) -> list[str]:
    """Return the paths of the files a commit holds.

    Args:
        project_dir: The target repository, as an absolute path.
        commit: The commit.

    Returns:
        The paths relative to the repository, a repository of its own being one path.
        Empty when git cannot list the commit's files.
    """
    return [os.fsdecode(path) for path in _tree_entry_dict(project_dir, commit) or {}]


def new_test_ext_list(project_dir: str, out_dir: str, base: str,
                      tip_list: Sequence[str] = ()) -> list[str]:
    """Return the languages base adds whose tests the start's index does not run.

    They are the languages of the files base or a commit of tip_list holds and the
    commit the working tree made when the command started (start_tree) does not, that
    have a default test command other than those of the languages whose tests a run over
    the index in out_dir runs (_start_test_lang_set), the first of them by name where
    several have the same one. A language is one LANG_DICT key, with the extensions
    ALIAS_DICT gives it.

    Examples:
        the start holds "MAIN.cbl" and "tools/gen.py" and no pyproject.toml,
        base adds "main.py"                                  -> ["py"]
        TEST_COMMAND="make test"                             -> []

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        base: The commit the container starts at.
        tip_list: The commits the container combines onto base.

    Returns:
        The languages' LANG_DICT keys, sorted. Empty when RUN_TEST is off, while
        TEST_COMMAND in the environment names a command, which is then the one command a
        run runs, and when base and every commit of tip_list are that start commit.
    """
    start = start_tree(project_dir)
    commit_set = {base, *tip_list} - {start}
    if not config.RUN_TEST or config.shell_list_env("TEST_COMMAND") or not commit_set:
        return []
    start_path_set = set(_commit_path_list(project_dir, start))
    path_set = {path for commit in commit_set
                for path in _commit_path_list(project_dir, commit)}
    lang_set = {ALIAS_DICT.get(ext, ext) for ext in map(ext_of, path_set - start_path_set)}
    lang_list = sorted(lang for lang in lang_set
                       if lang in LANG_DICT and LANG_DICT[lang].test_command_tuple)
    if not lang_list:
        return []
    run_command_set = {LANG_DICT[lang].test_command_tuple for lang in _start_test_lang_set(
        project_dir, out_dir, knowledge.project_name_of(project_dir)) or ()
        if lang in LANG_DICT}
    ext_list = []
    for lang in lang_list:
        if LANG_DICT[lang].test_command_tuple not in run_command_set:
            ext_list.append(lang)
            run_command_set.add(LANG_DICT[lang].test_command_tuple)
    return ext_list


def _start_test_lang_set(project_dir: str, out_dir: str, name: str) -> set[str] | None:
    """Return the languages whose tests a run over the index in an output directory runs.

    They are the index's primary language when it has a test command
    (config.test_command_of), and each other language the index holds a file of whose
    default test command (config.other_test_list_of) has what it takes to run in the
    repository: its dependency file and its tests (tools.edit.other_test_ext_list).
    Whether the image has the other language's command is not looked at.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.

    Returns:
        The languages' LANG_DICT keys, with the extensions ALIAS_DICT gives each counted
        as it. None when there is no index there or it cannot be read.
    """
    try:
        index = store.load(store.index_db_path(out_dir, name))
    except (FileNotFoundError, sqlite3.DatabaseError):
        return None
    try:
        primary_ext = index.primary_ext
        ext_list = [index.file_lang_dict.get(path) or ext_of(path)
                    for path in index.file_path_list()]
    finally:
        index.close()
    run_list = [primary_ext] if config.test_command_of(primary_ext) else []
    other_ext_list = [ext for ext, _ in config.other_test_list_of(primary_ext, ext_list)]
    if other_ext_list:
        run_list += edit_mod.other_test_ext_list(
            other_ext_list, gitcmd.work_tree_file_list(project_dir) or [], project_dir)
    return {ALIAS_DICT.get(ext, ext) for ext in run_list}


def _has_test_to_run(project_dir: str, out_dir: str, name: str) -> bool:
    """Say whether a run over the index in an output directory has a test to run.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.

    Returns:
        False when _start_test_lang_set() names no language. True when it names one,
        and when there is no index there or it cannot be read.
    """
    lang_set = _start_test_lang_set(project_dir, out_dir, name)
    return lang_set is None or bool(lang_set)


def build_output_tuple_of(out_dir: str, name: str) -> tuple[str, ...]:
    """Return what the tools of a repository's language write, which is not a change.

    Args:
        out_dir: The output directory, as an absolute path.
        name: The repository's directory name.

    Returns:
        The build_output_tuple entries of the index's primary language that do not start
        with "/"; empty when there is no index or the language has none.
    """
    lang_config = LANG_DICT.get(store.primary_ext_of(store.index_db_path(out_dir, name)))
    return split_output_tuple(lang_config.build_output_tuple)[0] \
        if lang_config is not None else ()


# How many times filling each fingerprint (_dep_fingerprint) ended with a non-zero exit
# status in this process, and how many such endings stop it being filled again until the
# next command
_fill_fail_count_dict: collections.Counter = collections.Counter()
_FILL_MAX_FAIL = 2

# Held from the second look at the mark and at _fill_fail_count_dict to the end of the
# filling: one filling runs at a time
_fill_lock = threading.Lock()


def fill_cache_once(project_dir: str, out_dir: str, agent: str, cache: str, engine: str,
                    image: str, exclude_list: list[str] | None, base: str,
                    fill_ext_list: Sequence[str] = ()) -> int:
    """Fill the package cache unless it is filled for base's dependency files and image.

    A filling that ends with a non-zero exit status is logged as a warning and not
    raised.

    Steps:
    1. Nothing is filled when a run over the index in out_dir has no test to run
       (_has_test_to_run) and fill_ext_list is empty, when the cache's mark already
       holds base's set of dependency files, the image and fill_ext_list, or when
       filling them ended with a non-zero exit status _FILL_MAX_FAIL times in this
       process (_fill_fail_count_dict)
    2. The filling's clone is checked out at the commit the working tree made when
       the command started (start_tree), whatever base is and whatever the command
       has put into the working tree since, and the default test commands of
       fill_ext_list run over it as well (warm). When the mark already holds base's
       set of dependency files and the image without fill_ext_list, only those run
    3. Where base holds dependency files that commit does not hold as they are
       (_dep_change_dict), the filling is a refill: with WARM_MAX_REFILL at 0 those
       files are left out and the cache is filled for that commit's own, and
       otherwise they are copied over the clone's (warm with dep_commit) and the
       filling is counted

    The container that fills it is given the agent's own output directory and the cache
    by name (warm with agent and cache). The run-tests inside it reads the index there,
    and where it finds the analysis not describing the clone it makes it again there, in
    the agent's copy.

    Args:
        project_dir: The target repository, as an absolute path.
        out_dir: The output directory, as an absolute path.
        agent: The agent the run is one of, as agent_name() names it.
        cache: The package cache, as an absolute path (make_cache()'s return value).
        engine: The engine's executable, as pick_engine() returns it.
        image: The image to run.
        exclude_list: The extra names kept out of the analysis.
        base: The commit the run's container starts at, whose dependency files the
            cache is filled for.
        fill_ext_list: The languages whose default test commands the filling runs as
            well, as new_test_ext_list() returns them for base.

    Returns:
        How many refills were made: 1 when a container was started with dependency
        files of base the commit of step 2 does not hold, otherwise 0.

    Raises:
        RuntimeError: As start_tree() and warm() raise.
    """
    # == Step 1: tell whether there is anything to fill ======
    if not fill_ext_list and not _has_test_to_run(
            project_dir, out_dir, knowledge.project_name_of(project_dir)):
        logger.debug("no test command, so the package cache is not filled")
        return 0
    fingerprint = _dep_fingerprint(project_dir, base, "", [], _image_id(engine, image),
                                   fill_ext_list)
    if _is_fill_known(fingerprint, cache, base):
        return 0
    with _fill_lock:
        if _is_fill_known(fingerprint, cache, base):
            return 0
        fill_own = not fill_ext_list or _dep_fingerprint(
            project_dir, base, "", [], _image_id(engine, image)) not in _fill_mark_set(
            fill_mark_path(cache))
        return _fill_now(project_dir, out_dir, agent, cache, engine, image, exclude_list,
                         base, fingerprint, fill_ext_list, fill_own)


def _is_fill_known(fingerprint: str, cache: str, base: str) -> bool:
    """Say whether a filling is not to be made.

    It is not when the cache's mark holds its fingerprint, or when filling it failed
    _FILL_MAX_FAIL times in this process (_fill_fail_count_dict).

    Args:
        fingerprint: The filling's _dep_fingerprint().
        cache: The package cache, as an absolute path.
        base: The commit whose dependency files the fingerprint stands for.

    Returns:
        True when the filling is not to be made.
    """
    if fingerprint in _fill_mark_set(fill_mark_path(cache)):
        logger.debug("%s already holds what the dependency files of %s fetch in this "
                     "image", cache, gitcmd.short(base))
        return True
    if _fill_fail_count_dict[fingerprint] >= _FILL_MAX_FAIL:
        logger.debug("filling the package cache for the dependency files of %s failed %d "
                     "times in this command, so it is not tried again",
                     gitcmd.short(base), _fill_fail_count_dict[fingerprint])
        return True
    return False


def _fill_now(project_dir: str, out_dir: str, agent: str, cache: str, engine: str,
              image: str, exclude_list: list[str] | None, base: str,
              fingerprint: str, fill_ext_list: Sequence[str], fill_own: bool) -> int:
    """Run steps 2 and 3 of fill_cache_once().

    It takes fill_cache_once()'s arguments, with the filling's _dep_fingerprint() as
    fingerprint and whether the index's own test command runs in it as fill_own, and
    returns what fill_cache_once() returns.
    """
    # == Step 2: the working tree's commit at the start ======
    tree_commit = start_tree(project_dir)
    # == Step 3: fill, base's dependency files as a refill ===
    dep_commit = base if _dep_change_dict(project_dir, tree_commit, base) else ""
    if dep_commit and config.WARM_MAX_REFILL <= 0:
        logger.info("%s holds dependency files the working tree did not when the "
                    "command started, and WARM_MAX_REFILL is 0, so the package cache "
                    "is not filled for them", gitcmd.short(base))
        dep_commit = ""
    fill_dict = warm(project_dir, out_dir, engine=engine, image=image,
                     exclude_list=exclude_list, once=True, cache=cache,
                     base=tree_commit, agent=agent, dep_commit=dep_commit,
                     fill_ext_list=fill_ext_list, fill_own=fill_own)
    if fill_dict["exit_status"]:
        _fill_fail_count_dict[fingerprint] += 1
        logger.warning("the package cache could not be filled (exit status %d), so the "
                       "run goes on with the cache as it is%s", fill_dict["exit_status"],
                       ", and the next command fills it again"
                       if _fill_fail_count_dict[fingerprint] >= _FILL_MAX_FAIL
                       else ", and the next container fills it again")
    return 1 if dep_commit and not fill_dict["skip"] else 0


def _fill_cache(mount: MountLayout, base: str, name: str, engine: str, image: str,
                exclude_list: list[str] | None, run_id: str,
                live_dep_path_list: list[str], out_path: str,
                build_output_tuple: tuple[str, ...],
                fill_ext_list: Sequence[str] = (),
                fill_own: bool = True) -> tuple[str, str, int]:
    """Fill the package cache in a container of its own, with the network on.

    It holds no key of this tool and no model takes part in it.

    Args:
        mount: The directories the container is given, as absolute paths.
        base: The commit the clone is checked out at.
        name: The repository's directory name.
        engine: The container engine. Empty picks one.
        image: The image to run. Empty uses config.CONTAINER_IMAGE.
        exclude_list: The directory and file names kept out of the analysis.
        run_id: The filling, as new_run_id() names it. It names the container.
        live_dep_path_list: The dependency files the container takes from mount.live,
            relative to the directory named after the repository in it.
        out_path: The file the container's stdout is kept in (run_the_container).
        build_output_tuple: The primary language's build_output_tuple entries that do
            not start with "/", as build_run_argv() takes them.
        fill_ext_list, fill_own: What run-tests runs besides or in place of the index's
            own test commands, as build_run_argv() takes them.

    Returns:
        (the engine's executable, the image, the container's exit status).

    Raises:
        RuntimeError: When there is no engine.
    """
    engine_path = pick_engine(engine)
    image_name = image or config.CONTAINER_IMAGE
    argv = build_run_argv(
        engine_path, image_name, mount, base, name,
        ["run-tests", "--project", f"{_WORK}/{name}", "--output-dir", _OUT],
        run_id=run_id, exclude_list=exclude_list, network=True, entrypoint=_WARM_ENTRYPOINT,
        live_dep_path_list=live_dep_path_list, build_output_tuple=build_output_tuple,
        fill_ext_list=fill_ext_list, fill_own=fill_own)
    return engine_path, image_name, run_the_container(
        argv, engine_path, run_id, config.WARM_TIMEOUT_SEC, out_path)


def _last_attempt(history_path: str, out_dir: str) -> dict:
    """Return the newest entry of the history in an agent's copy of the output directory.

    Args:
        history_path: The history file inside out_dir, as an absolute path.
        out_dir: The agent's copy of the output directory.

    Returns:
        What history.last_attempt() returns.

    Raises:
        RuntimeError: When the file is reached through a link that leads out of
            out_dir (inside_path).
    """
    return history.last_attempt(inside_path(history_path, out_dir))


def _own_record_dict(history_path: str, out_dir: str, record_before_count: int) -> dict:
    """Return the history entry this run wrote, or {} when it wrote none.

    The run appends its entry at the end. The newest one is this run's only when it is
    newer than what was there before the container started. A run that stopped before it
    recorded anything, and one with the history switched off, both leave the entry an
    earlier run wrote as the newest, and {} is returned for both.

    The clocks of the container and this machine are not compared.

    Args:
        history_path: The history file inside the output directory, as an absolute path.
        out_dir: The agent's copy of the output directory the file is in.
        record_before_count: The "n" of the newest entry before the container started, or 0
            when there was none.

    Returns:
        The entry as _record_dict_of() reads what a change run records, or {} for
        none and for one that holds a value of another shape.

    Raises:
        RuntimeError: As _last_attempt() raises.
    """
    attempt_dict = _last_attempt(history_path, out_dir)
    if attempt_dict.get("n", 0) <= record_before_count:
        return {}
    return _record_dict_of(attempt_dict, history_path, _EDIT_KIND)


def make_scratch(scratch: str) -> str:
    """Make the scratch directory the container writes the clone into.

    The name carries the run, and nothing of another run's is taken away here. The two
    directories in it, "work" and "tmp", are made here, by this process and not by the
    engine.

    Args:
        scratch: The scratch directory, as an absolute path.

    Returns:
        The path it was given.
    """
    for inside in ("work", "tmp"):
        os.makedirs(os.path.join(scratch, inside), exist_ok=True)
    return scratch


def make_clone_source(project_dir: str, scratch: str) -> str:
    """Copy the repository's git history into the scratch directory, for the container.

    The copy is a bare repository holding the branches, the tags and the objects. None
    of the working tree, the files git ignores, the repository's config or its hooks is
    in it, and objects the repository borrows from another one are copied in. The
    objects are hard-linked where the file system allows it.

    Args:
        project_dir: The target repository, as an absolute path: a working tree, a
            linked worktree or a submodule.
        scratch: The scratch directory make_scratch() made.

    Returns:
        The copy, as an absolute path.

    Raises:
        RuntimeError: When git cannot make the copy.
    """
    source = os.path.join(scratch, _SOURCE_DIR)
    proc = gitcmd.run(project_dir, "clone", "--bare", "--local", "--dissociate",
                      "--quiet", project_dir, source)
    if proc.returncode:
        raise RuntimeError(f"could not copy the history of {project_dir} to {source}: "
                           f"{proc.stderr.strip()[-config.LOG_EXCERPT_CHAR:]}")
    return source


# The name Gradle reads fetched dependencies under in the directory GRADLE_RO_DEP_CACHE
# names
_GRADLE_MODULE_NAME = "modules-2"


def make_cache(cache: str) -> str:
    """Make the directories the toolchain's package cache is written into.

    Anything an earlier run left is kept.

    Each of the directories _cache_env_dict() names, and the one Gradle reads fetched
    dependencies from, is made, or found, as a directory of its own and not a link. That
    one is given an empty _GRADLE_MODULE_NAME when it holds nothing under the name.

    Args:
        cache: The cache directory, as an absolute path.

    Returns:
        The path it was given.

    Raises:
        RuntimeError: When one of the directories is a link, or something else than a
            directory is in its place, or when one cannot be made.
    """
    gradle_share = _jvm_tool_env_dict(cache, "")["GRADLE_RO_DEP_CACHE"]
    for directory in (*_cache_env_dict(cache).values(), gradle_share):
        if os.path.lexists(directory) and not _is_real_dir(directory):
            raise RuntimeError(
                f"{directory} in the package cache is not a directory of its own: a "
                f"link or a file is in its place. Remove it, then run again")
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            raise _cache_dir_error(directory, e) from None
    # Whatever is under the name already, a link or what another caller made at the
    # same moment, stays as it is
    module_dir = os.path.join(gradle_share, _GRADLE_MODULE_NAME)
    try:
        os.mkdir(module_dir)
    except FileExistsError:
        pass
    except OSError as e:
        raise _cache_dir_error(module_dir, e) from None
    return cache


def _cache_dir_error(directory: str, error: OSError) -> RuntimeError:
    """Return what says a directory of the package cache cannot be made.

    Args:
        directory: The directory, as an absolute path.
        error: What making it raised.

    Returns:
        The error, naming the directory and what to do.
    """
    return RuntimeError(
        f"{directory} in the package cache cannot be made ({error.strerror or error}). "
        f"Make the directory above it writable, or remove the cache, then run again")


# The variable that has litellm read the table of Anthropic beta headers it ships with,
# with none fetched before the first call that sends such a header
_LOCAL_BETA_HEADER_ENV = "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS"


def start_broker(sock_dir: str, project_dir: str = "", output_dir: str = "",
                 run_id: str = "", cache: str = "", engine: str = "",
                 image: str = "", token_out: str = "",
                 token_left: int = 0, agent: str = "",
                 max_refill: int | None = None, base: str = "",
                 fill_ext_list: Sequence[str] = ()) -> subprocess.Popen:
    """Start the broker on a socket in sock_dir and wait for it to be listening.

    The broker runs with this process's environment, which is where the credentials are,
    listens on the address it is passed whatever LLM_BROKER says, and is handed no
    terminal to ask anything on. It reads no .env file of its own
    (config.NO_ENV_FILE_ENV): the settings it has are the ones this process read. It
    reads the model table litellm ships with and is handed the entries this process's
    table holds for its models (broker.model_table_text), and it reads the table of
    Anthropic beta headers litellm ships with (_LOCAL_BETA_HEADER_ENV). It logs under
    the label of the work that started it.

    Args:
        sock_dir: The directory the socket is made in, as an absolute path.
        project_dir: The repository a refill fills the cache from, as an absolute path.
            Empty starts a broker that refuses to fill any.
        output_dir: The output directory, as an absolute path.
        run_id: The run whose dependency files a refill takes, as new_run_id() names it.
            Empty leaves a refill filling the cache for what is committed.
        cache: The package cache a refill fills, as an absolute path. Empty leaves the
            broker to work it out from output_dir.
        engine: The container engine a refill starts, handed to the broker as
            CONTAINER_ENGINE. Empty leaves the broker the configured one.
        image: The image a refill runs, handed to the broker as CONTAINER_IMAGE. Empty
            leaves the broker the configured one.
        token_out: The file the broker writes the token counts of its model calls to, as
            an absolute path (tokencount.child_file). Empty has it write none.
        token_left: How many tokens the broker's model calls and those of the processes
            writing beside token_out may use between them before it refuses a model
            call. 0 sets no limit.
        agent: The agent the broker serves, as agent_name() names it: a refill is made
            for that agent (warm), and the broker's log goes under host_dir() of it.
            Empty names none.
        max_refill: How many refills the broker serves, handed to it as WARM_MAX_REFILL.
            None leaves the broker the configured number.
        base: The commit a refill checks its clone out at. Empty leaves a refill to make
            the commit of the working tree as it is then.
        fill_ext_list: The languages whose default test commands a refill runs as well
            (warm's fill_ext_list).

    Returns:
        The running process.

    Raises:
        RuntimeError: When the broker ends before it listens, or has not listened
            within BROKER_START_TIMEOUT_SEC.
        KeyboardInterrupt: When the run is stopped while the broker starts. The broker
            is ended (stop_broker) before this and the error above are raised.
    """
    url = socket_url(sock_dir)
    argv = config.own_argv(["broker", "--endpoint", url])
    if project_dir:
        argv += ["--project", project_dir, "--output-dir", output_dir]
    if agent:
        argv += ["--agent", agent]
    if run_id:
        argv += ["--run", run_id]
    if project_dir and base:
        argv += ["--base", base]
    if project_dir and fill_ext_list:
        argv += ["--fill-ext", ",".join(fill_ext_list)]
    if token_out:
        argv += ["--token-out", token_out]
    if token_left > 0:
        argv += ["--token-left", str(token_left)]
    if progress.label_text():
        argv += ["--label", progress.label_text()]
    # The broker, which holds the credentials, is given this process's environment, not
    # config.child_env()
    add_env_dict = {key: value for key, value in (
        ("SANDBOX_CACHE_DIR", cache), ("CONTAINER_ENGINE", engine),
        ("CONTAINER_IMAGE", image)) if value}
    if max_refill is not None:
        add_env_dict["WARM_MAX_REFILL"] = str(max_refill)
    add_env_dict[config.NO_ENV_FILE_ENV] = config.NO_ENV_FILE_VALUE
    add_env_dict[broker.LOCAL_MODEL_TABLE_ENV] = "True"
    add_env_dict[_LOCAL_BETA_HEADER_ENV] = "True"
    add_env_dict[broker.MODEL_TABLE_ENV] = broker.model_table_text()
    process = subprocess.Popen(argv, env={**os.environ, **add_env_dict},
                               stdin=subprocess.DEVNULL)
    socket_path = os.path.join(sock_dir, _SOCKET_NAME)
    deadline = time.monotonic() + config.BROKER_START_TIMEOUT_SEC
    # Whatever ends the wait other than the broker listening, a stop included, ends the
    # broker before it is raised on
    try:
        while time.monotonic() < deadline:
            if os.path.exists(socket_path):
                logger.info("the broker is listening on %s", url)
                return process
            if process.poll() is not None:
                raise RuntimeError(f"the broker ended before it listened on {url} "
                                   f"(exit status {process.returncode})")
            time.sleep(0.05)
        raise RuntimeError(f"the broker did not listen on {url} within "
                           f"{config.BROKER_START_TIMEOUT_SEC} seconds")
    except BaseException:
        stop_broker(process, sock_dir)
        raise


def stop_broker(process: subprocess.Popen, sock_dir: str) -> None:
    """End the broker and take its socket directory away.

    Args:
        process: What start_broker() returned.
        sock_dir: The directory the socket was made in.
    """
    process.terminate()
    try:
        process.wait(timeout=config.BROKER_START_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        logger.warning("the broker did not end when asked, so it is being killed")
        process.kill()
        process.wait()
    _remove_tree(sock_dir, "the broker's socket directory")


def remove_scratch(scratch: str) -> None:
    """Take the scratch directory away, saying so when it cannot be taken away.

    The clone is not kept. A directory an engine wrote as root, when it was not told who
    to run as, is reported as still there.

    Args:
        scratch: The scratch directory, as an absolute path.
    """
    _remove_tree(scratch, "the clone")


def _remove_tree(path: str, what: str) -> bool:
    """Take a directory away, saying so when it cannot be taken away.

    Args:
        path: The directory, as an absolute path. One that is not there is not
            reported.
        what: What the directory held, for the message.

    Returns:
        True when nothing is left at the path.
    """
    return _remove_with(shutil.rmtree, path, what)


def _remove_file(path: str, what: str) -> bool:
    """Take a file away, saying so when it cannot be taken away.

    Args:
        path: The file, as an absolute path. One that is not there is not reported.
        what: What the file held, for the message.

    Returns:
        True when nothing is left at the path.
    """
    return _remove_with(os.remove, path, what)


def _remove_with(remove: Callable[[str], object], path: str, what: str) -> bool:
    """Take what is at a path away with remove, saying so when it cannot be taken away.

    Args:
        remove: What takes it away: shutil.rmtree for a directory, os.remove for a
            file.
        path: The path, as an absolute path. One that is not there is not reported.
        what: What the path held, for the message.

    Returns:
        True when nothing is left at the path.
    """
    try:
        remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("%s at %s could not be taken away (%s). Remove it by hand; "
                       "nothing reads it", what, path, e)
    return not os.path.lexists(path)


def meta_path(patch_path: str) -> str:
    """Return the path of the summary written beside one patch.

    Args:
        patch_path: The patch's path.

    Returns:
        That path with _META_SUFFIX added.
    """
    return f"{patch_path}{_META_SUFFIX}"


def _read_meta(patch_path: str, out_dir: str) -> dict:
    """Return the summary written beside a patch, or {} when there is none to read.

    Args:
        patch_path: The patch's path, or "" when there is no patch to read. The
            summary sits next to it, with .meta.json added to the name.
        out_dir: The agent's copy of the output directory the patch is in.

    Returns:
        The parsed summary, or {} when it is not there, cannot be read (_json_across)
        or is not an object.

    Raises:
        RuntimeError: When the summary is reached through a link that leads out of
            out_dir (inside_path).
    """
    if not patch_path:
        return {}
    path = meta_path(patch_path)
    try:
        meta_dict = _json_across(path, out_dir)
    except FileNotFoundError:
        logger.warning("no summary beside %s", patch_path)
        return {}
    except (OSError, ValueError) as e:
        logger.warning("could not read %s: %s", path, e)
        return {}
    return meta_dict if isinstance(meta_dict, dict) else {}


def _write_land_meta(patch_path: str, out_dir: str, commit_dict: dict) -> None:
    """Add the branch and the commit a patch landed on to the summary beside it.

    Args:
        patch_path: The patch's path.
        out_dir: The agent's copy of the output directory the patch is in.
        commit_dict: take_in_patch()'s return value.

    Raises:
        RuntimeError: When the summary is reached through a link that leads out of
            out_dir (inside_path).
    """
    path = meta_path(patch_path)
    meta_dict = _read_meta(patch_path, out_dir)
    meta_dict.update(commit_dict)
    try:
        with tmpfile.atomic_write(path) as tmp_path:
            with open(tmp_path, "w", encoding="utf-8", errors="backslashreplace") as handle:
                json.dump(meta_dict, handle, ensure_ascii=False)
    except OSError as e:
        logger.warning("could not write %s: %s", path, e)
