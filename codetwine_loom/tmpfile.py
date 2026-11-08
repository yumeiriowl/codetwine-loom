"""The temporary file an atomic write goes through, and the leftovers of killed writes.

A file written through atomic_write() goes to a temporary name beside the real one and
is moved into place with os.replace, which is atomic within a file system. The name
carries the writer's process ID; two processes writing to the same output directory
never share it.

A write that is killed outright leaves its temporary file behind. remove_dead_tmp()
takes those away; it reads the same name temp_path() builds.

atomic_write() is the whole sequence - name, removal of leftovers, making of the empty
temporary file at a name nothing stands at (make_new_file), move into place, removal
of what never got moved. The writer produces the bytes in the block.

read_file() reads a regular file whole, without waiting on anything that is none, and
read_json() reads a JSON file that way. check_file() confirms that a path names such a
file without opening it.

file_stamp() and move_mtime_on() give a file written at a path a modification time and
size the path has not had.
"""

import contextlib
import errno
import json
import logging
import os
import stat
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_SUFFIX = ".tmp"

# The most bytes a file read whole may hold
READ_MAX_BYTE = 64 * 1024 * 1024


def check_file(path: str | bytes, max_byte: int | None = None) -> None:
    """Confirm that path names a regular file of at most max_byte bytes.

    The file is not opened. A named pipe, a device and a directory are refused.

    Args:
        path: The file. A symbolic link is followed.
        max_byte: The most bytes the file may hold. None is READ_MAX_BYTE.

    Raises:
        FileNotFoundError: When there is no such file.
        OSError: When the file cannot be looked at or is not a regular file.
        ValueError: When the file is larger than max_byte.
    """
    if max_byte is None:
        max_byte = READ_MAX_BYTE
    file_stat = os.stat(path)
    if not stat.S_ISREG(file_stat.st_mode):
        raise OSError(f"{os.fsdecode(path)} is not a regular file")
    if file_stat.st_size > max_byte:
        raise ValueError(f"{os.fsdecode(path)} is {file_stat.st_size} bytes, over the "
                         f"{max_byte} read of one file")


def read_file(path: str, max_byte: int | None = None) -> bytes:
    """Read one file whole, which is to be a regular file of at most max_byte bytes.

    The file is opened without waiting for a writer. A named pipe, a device and a
    directory are refused.

    Args:
        path: The file to read. A symbolic link is followed.
        max_byte: The most bytes the file may hold. None is READ_MAX_BYTE.

    Returns:
        The file's bytes.

    Raises:
        FileNotFoundError: When there is no such file.
        OSError: When the file cannot be opened or is not a regular file.
        ValueError: When the file is larger than max_byte.
    """
    if max_byte is None:
        max_byte = READ_MAX_BYTE
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OSError(f"{path} is not a regular file")
        if file_stat.st_size > max_byte:
            raise ValueError(f"{path} is {file_stat.st_size} bytes, over the "
                             f"{max_byte} read of one file")
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "rb") as handle:
        raw = handle.read(max_byte + 1)
    if len(raw) > max_byte:
        raise ValueError(f"{path} is over the {max_byte} bytes read of one file")
    return raw


def read_json(path: str, max_byte: int | None = None) -> object:
    """Read one JSON file, as read_file() reads a file.

    Args:
        path: The file to read. A symbolic link is followed.
        max_byte: The most bytes the file may hold. None is READ_MAX_BYTE.

    Returns:
        What the file holds.

    Raises:
        FileNotFoundError: When there is no such file.
        OSError: When the file cannot be opened or is not a regular file.
        ValueError: When the file is larger than max_byte, is not UTF-8, is not JSON,
            is nested deeper than the parser reads, or holds more values than this
            process has the memory for.
    """
    raw = read_file(path, max_byte)
    try:
        return json.loads(raw.decode("utf-8"))
    except RecursionError:
        raise ValueError(f"{path} is nested deeper than is read") from None
    except MemoryError:
        raise ValueError(f"{path} holds more values than there is memory to read "
                         f"them into") from None


# The stamps kept for each path, by the path with its symbolic links resolved
# (_stamp_key). A stamp is (the second the file was last modified in, its size in bytes)
_stamp_set_dict: dict[str, set[tuple[int, int]]] = {}


def _stamp_key(file_path: str) -> str:
    """Return the path the stamps of a file are kept under.

    Args:
        file_path: The file, as an absolute path.

    Returns:
        The path with its symbolic links resolved, or the path as it is when the links
        go deeper than they can be resolved (RecursionError).
    """
    try:
        return os.path.realpath(file_path)
    except RecursionError:
        return os.path.abspath(file_path)


def _stamp_of(file_path: str) -> tuple[int, int] | None:
    """Return the stamp of the file at a path as it stands.

    Args:
        file_path: The file, as an absolute path.

    Returns:
        (the modification time in whole seconds, the size in bytes), or None for a
        path that is not a regular file or is not there.
    """
    try:
        file_stat = os.lstat(file_path)
    except OSError:
        return None
    if not stat.S_ISREG(file_stat.st_mode):
        return None
    return int(file_stat.st_mtime), file_stat.st_size


def file_stamp(file_path: str) -> tuple[int, int] | None:
    """Return the stamp of a file before it is written, and keep it for the path.

    Args:
        file_path: The file, as an absolute path.

    Returns:
        (the modification time in whole seconds, the size in bytes) of a regular file.
        For a path with no file, the latest stamp kept for it: what a file held there
        when this was last asked, such as right before the file was removed. None for
        a path that is anything else, and for one no stamp was kept for.
    """
    stamp_set = _stamp_set_dict.setdefault(_stamp_key(file_path), set())
    stamp = _stamp_of(file_path)
    if stamp is not None:
        stamp_set.add(stamp)
        return stamp
    return None if os.path.lexists(file_path) else max(stamp_set, default=None)


def move_mtime_on(file_path: str, before_stamp: tuple[int, int] | None) -> None:
    """Give a file that was just written a stamp the path has not had.

    A file of the size before_stamp holds whose modification time does not fall in a
    later second than before_stamp's is given the second after it, and a file whose
    second and size are both those of a stamp kept for the path is given the next
    second no kept stamp of that size holds. A file of a size the path has not had is
    left as it is. The stamp the file ends with is kept.

    Args:
        file_path: The file, as an absolute path.
        before_stamp: file_stamp() of the path before it was written. None leaves the
            file as it is.
    """
    stamp = _stamp_of(file_path)
    if before_stamp is None or stamp is None:
        return
    stamp_set = _stamp_set_dict.setdefault(_stamp_key(file_path), set())
    stamp_set.add(before_stamp)
    second, size = stamp
    if size == before_stamp[1] and second <= before_stamp[0]:
        second = before_stamp[0] + 1
    while (second, size) in stamp_set:
        second += 1
    if second != stamp[0]:
        try:
            os.utime(file_path, (second, second))
        except OSError as e:
            logger.info("the modification time of %s could not be moved on: %s",
                        file_path, e)
            return
    stamp_set.add((second, size))


def temp_path(path: str) -> str:
    """Return the temporary name to write path through.

    Args:
        path: Where the file is to end up.

    Returns:
        The path with this process's ID and .tmp added.
    """
    return f"{path}.{os.getpid()}{_SUFFIX}"


def process_dead(pid: int) -> bool:
    """Say whether the process with an ID has ended.

    Args:
        pid: The process ID.

    Returns:
        True when no process holds that ID. A process this one may not signal counts as
        running, and so does a number no process ID can be.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (OSError, OverflowError):
        return False
    return False


def remove_dead_tmp(path: str) -> list[str]:
    """Remove the temporary files of writes to path that nothing is going to finish.

    What is taken away is this process's own temporary file and the files of processes
    that have ended. A file whose writer is still running is left alone.

    Anything written beside a temporary file goes with it (sqlite's "-journal", "-wal"
    and "-shm"): the name is matched up to the suffix rather than at its end.

    A file whose process ID a running process holds again is kept as well. Nothing here
    raises: a file that cannot be removed is logged, and a directory that cannot be read
    gives [].

    Args:
        path: Where the file is to end up. Its directory is the one looked through.

    Returns:
        The paths that were to go and are still there.
    """
    directory = os.path.dirname(path) or "."
    prefix = f"{os.path.basename(path)}."
    try:
        name_list = os.listdir(directory)
    except OSError:
        return []
    stay_list = []
    for name in name_list:
        if not name.startswith(prefix):
            continue
        # "<pid>.tmp" on its own, or with what another tool appended to it
        pid, suffix, _ = name[len(prefix):].partition(_SUFFIX)
        if not suffix or not (pid.isascii() and pid.isdigit()):
            continue
        if int(pid) != os.getpid() and not process_dead(int(pid)):
            continue
        if not _remove(os.path.join(directory, name)):
            stay_list.append(os.path.join(directory, name))
    return stay_list


def _remove(path: str) -> bool:
    """Remove one temporary file, saying so in the log when it will not go.

    Nothing here raises.

    Args:
        path: The temporary file to remove. One that is not there is not an error.

    Returns:
        True when nothing is left at the path.
    """
    try:
        os.remove(path)
    except FileNotFoundError:
        return True
    except OSError as e:
        logger.warning("the leftover %s could not be removed (%s)",
                       os.path.basename(path), e)
        return False
    return True


def make_new_file(path: str, mode: int = 0o666) -> None:
    """Make an empty regular file at a name nothing stands at.

    Args:
        path: The file to make. Its directory has to exist already.
        mode: The permission bits the file is made with, before the umask.

    Raises:
        FileExistsError: When something stands at the name, a symbolic link included.
        OSError: When the file cannot be made.
    """
    flag = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    os.close(os.open(path, flag, mode))


@contextlib.contextmanager
def atomic_write(path: str, mode: int = 0o666) -> Iterator[str]:
    """Yield the name to write path through, then move it into place.

    The block writes the whole file under the yielded name. Leaving the block normally
    is what moves it onto path; a run that stops part-way leaves whatever was there
    before untouched. When os.replace has not taken the temporary file, the clean-up
    removes it; one the clean-up cannot remove stays, logged unless the directory
    holding it cannot be read.

    remove_dead_tmp() runs at both ends: before the block it takes away what writes
    that were killed outright left, and after it what this write itself left, the
    temporary file and whatever a tool wrote beside it alike.

    The temporary file is made here, empty, before the block (make_new_file), and the
    block writes the file it finds under the yielded name. Nothing is written where
    something stands at that name, or beside it under a name that starts with it,
    that could not be removed.

    Args:
        path: Where the file is to end up. Its directory has to exist already.
        mode: The permission bits the temporary file is made with, before the umask.

    Yields:
        The temporary path to write.

    Raises:
        FileExistsError: When what stands at the temporary name, a symbolic link
            included, or beside it could not be removed.
        OSError: When the temporary file cannot be made or moved onto path.
    """
    tmp_path = temp_path(path)
    own_name = os.path.basename(tmp_path)
    own_stay_list = [stay for stay in remove_dead_tmp(path)
                     if os.path.basename(stay).startswith(own_name)]
    if own_stay_list:
        raise FileExistsError(errno.EEXIST, "what stands at the temporary name of the "
                              "write could not be removed", own_stay_list[0])
    make_new_file(tmp_path, mode)
    try:
        yield tmp_path
        os.replace(tmp_path, path)
    finally:
        # Anything of ours is still there only when the write stopped before
        # os.replace
        remove_dead_tmp(path)
