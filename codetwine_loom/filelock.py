"""The lock taken when the writers of one thing are not all in one process.

The lock is a file of its own beside what it stands for.

Where there is no flock to take, the block still runs and says so in the log.
"""

import contextlib
import logging
import os
import threading
from collections.abc import Iterator

try:
    import fcntl
except ImportError:
    # A system with no file locks
    fcntl = None

logger = logging.getLogger(__name__)

# What the lock file is called, beside the file or directory it stands for
_SUFFIX = ".lock"

# The lock files a thread holds. A block that takes one it is already inside of runs on
# the lock it has, without a second flock. Two threads still wait for one another
_HOLD_STATE = threading.local()


def _hold_set() -> set[str]:
    """Return the lock files this thread holds.

    Returns:
        The set, made on this thread's first call.
    """
    if not hasattr(_HOLD_STATE, "path_set"):
        _HOLD_STATE.path_set = set()
    return _HOLD_STATE.path_set


def path_for(path: str) -> str:
    """Return the lock file that stands for one path.

    Args:
        path: The file or directory the lock is about.

    Returns:
        The lock file's path.
    """
    return f"{path}{_SUFFIX}"


class LockBusy(RuntimeError):
    """The lock hold_lock() was told not to wait for is held by another process."""


@contextlib.contextmanager
def hold_lock(path: str, what: str, wait: bool = True) -> Iterator[None]:
    """Hold the lock for one path for the length of the block.

    Args:
        path: The file or directory the lock is about. Its directory must exist.
        what: What is being held, for the message when there is no lock to take and
            the one when another process holds it and this one waits.
        wait: Whether to wait, with no limit, while another process holds the lock.
            False raises LockBusy instead.

    Yields:
        Nothing. The block runs either way; what differs is whether it ran alone.

    Raises:
        LockBusy: When wait is False and another process holds the lock; the block
            does not run.
    """
    lock_path = os.path.realpath(path_for(path))
    if lock_path in _hold_set():
        yield
        return
    if fcntl is None:
        logger.warning("this system has no file locks, so two runs writing %s at once "
                       "can undo one another. Give each run an output directory of its "
                       "own", what)
        yield
        return
    try:
        handle = open(lock_path, "a+", encoding="utf-8")
    except OSError as e:
        logger.warning("%s is being written without its lock (%s)", what, e)
        yield
        return
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                raise LockBusy(f"another run of this tool holds {what}") from None
            logger.info("waiting for %s, which another run of this tool holds", what)
            fcntl.flock(handle, fcntl.LOCK_EX)
        _hold_set().add(lock_path)
        yield
    finally:
        _hold_set().discard(lock_path)
        # Closing the file is what gives the lock up, and it is given up however the
        # block ended
        handle.close()
