#!/usr/bin/env python3
"""Keep the dependencies Gradle fetched where every container reads them.

Usage:
    ctloom-gradle-share seed <share> <home>
    ctloom-gradle-share publish <home> <share>

<home> is a Gradle home one container has to itself (GRADLE_USER_HOME) and <share> the
directory the containers read fetched dependencies from without writing to it
(GRADLE_RO_DEP_CACHE). What is kept is <home>/caches/modules-2.

seed copies <share>/modules-2 to <home>/caches/modules-2, so a run with a network
fetches only what the share does not hold. Nothing is copied when the share holds
none, when <share>/modules-2 is neither a directory of the share nor a link to one,
or when the home already has the directory. A copy that could not be finished is
removed.

Only the regular files and the directories are copied, in either direction.

publish copies <home>/caches/modules-2 to a new directory in <share>, without the lock
files and gc.properties, points <share>/modules-2 at it in one step, and removes the
directories it pointed at before. Nothing is done when the home has no such directory.

Exits 0 when there was nothing to do or it was done, 1 with one line on standard error
when it could not be done, and 2 when the arguments are not one of the two forms.
"""

import os
import shutil
import stat
import sys
import time

# The directory of a Gradle home that holds the fetched dependencies, the name the
# share is read under, and what the directories that name points at start with
_MODULE_DIR = os.path.join("caches", "modules-2")
_LINK_NAME = "modules-2"
_SNAP_PREFIX = "snap."

# The files of a Gradle home's dependency directory a share does not hold
_SKIP_NAME = "gc.properties"
_SKIP_SUFFIX = ".lock"


def _say(text: str) -> None:
    """Write one line to standard error."""
    print(f"ctloom-gradle-share: {text}", file=sys.stderr)


def seed(share: str, home: str) -> bool:
    """Copy the dependencies the share holds into a Gradle home.

    Args:
        share: The directory the containers read fetched dependencies from.
        home: The Gradle home to start.

    Returns:
        True when they were copied. False when the share holds none, when what it
        holds under the name leads out of the share, or when the home already has
        its dependency directory.

    Raises:
        OSError: When the copy could not be finished. What was copied is removed.
    """
    source = os.path.realpath(os.path.join(share, _LINK_NAME))
    target = os.path.join(home, _MODULE_DIR)
    if not os.path.isdir(source) or os.path.lexists(target):
        return False
    if os.path.dirname(source) != os.path.realpath(share):
        _say(f"{os.path.join(share, _LINK_NAME)} leads to {source}, which is not a "
             f"directory of {share}, so no home is started from it")
        return False
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        shutil.copytree(source, target, ignore=_skip_name_list)
    except OSError:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return True


def _skip_name_list(directory: str, name_list: list[str]) -> list[str]:
    """Return the names of one directory that are not copied (shutil.copytree's ignore).

    Args:
        directory: The directory being copied.
        name_list: The names in it.

    Returns:
        The lock files, gc.properties, and whatever is neither a regular file nor a
        directory (a link, a pipe, a device).
    """
    skip_list = []
    for name in name_list:
        mode = os.lstat(os.path.join(directory, name)).st_mode
        if name == _SKIP_NAME or name.endswith(_SKIP_SUFFIX) \
                or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            skip_list.append(name)
    return skip_list


def publish(home: str, share: str) -> bool:
    """Put the dependencies of a Gradle home where the containers read them.

    Args:
        home: The Gradle home that fetched.
        share: The directory the containers read fetched dependencies from.

    Returns:
        True when the share now points at a copy of the home's dependency directory.
        False when the home has none.
    """
    source = os.path.join(home, _MODULE_DIR)
    if not os.path.isdir(source):
        return False
    os.makedirs(share, exist_ok=True)
    snap_name = f"{_SNAP_PREFIX}{time.time_ns()}"
    snap = os.path.join(share, snap_name)
    link = os.path.join(share, _LINK_NAME)
    new_link = f"{link}.new"
    for path in (snap, new_link):
        _remove(path)
    shutil.copytree(source, snap, ignore=_skip_name_list)
    # Something else than a link under the name cannot be replaced in one step
    if os.path.lexists(link) and not os.path.islink(link):
        _remove(link)
    os.symlink(snap_name, new_link)
    os.replace(new_link, link)
    for name in os.listdir(share):
        if name.startswith(_SNAP_PREFIX) and name != snap_name:
            _remove(os.path.join(share, name))
    return True


def _remove(path: str) -> None:
    """Remove a file, a link or a directory with what it holds. A path nothing is at is
    left alone."""
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def main(argv: list[str]) -> int:
    """Run one of the two forms.

    Args:
        argv: The program's name, "seed" or "publish", and the two directories.

    Returns:
        The exit status the module's description names.
    """
    if len(argv) != 4 or argv[1] not in ("seed", "publish"):
        _say("usage: ctloom-gradle-share seed <share> <home> | publish <home> <share>")
        return 2
    try:
        if argv[1] == "seed":
            seed(argv[2], argv[3])
        else:
            publish(argv[2], argv[3])
    except (OSError, shutil.Error) as e:
        _say(f"{argv[1]} of {argv[2]} to {argv[3]} could not be done: "
             f"{type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
