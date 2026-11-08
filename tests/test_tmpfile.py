"""The temporary file an atomic write goes through, and the leftovers of killed writes.

The checks run against files under tmp_path. Where a process has to be dead, one is
started and waited for; where the operating system's answer has to be a refusal,
os.kill and os.remove are substituted.
"""

import os
import re
import stat
import subprocess
import sys
import types

import pytest

from codetwine_loom import tmpfile


def _dead_pid() -> int:
    """Return the id of a process that has ended and been waited for."""
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


# == process_dead ==============================================
def test_an_id_of_0_or_less_is_not_dead():
    assert tmpfile.process_dead(0) is False
    assert tmpfile.process_dead(-1) is False


def test_this_process_is_not_dead():
    assert tmpfile.process_dead(os.getpid()) is False


def test_a_process_that_ended_is_dead():
    assert tmpfile.process_dead(_dead_pid()) is True


def test_a_process_that_cannot_be_signalled_counts_as_running(monkeypatch):
    def refuse(pid, sig):
        raise PermissionError("not yours")

    monkeypatch.setattr(os, "kill", refuse)
    assert tmpfile.process_dead(12345) is False


def test_a_number_no_process_id_can_be_counts_as_running():
    assert tmpfile.process_dead(2 ** 31) is False
    assert tmpfile.process_dead(10 ** 30) is False


# == remove_dead_tmp ===========================================
def test_this_processs_own_temporary_file_is_taken_away(tmp_path):
    target = tmp_path / "index.sqlite"
    own = tmp_path / f"index.sqlite.{os.getpid()}.tmp"
    own.write_text("half written", encoding="utf-8")
    tmpfile.remove_dead_tmp(str(target))
    assert not own.exists()


def test_the_temporary_file_of_a_dead_process_is_taken_away(tmp_path):
    target = tmp_path / "index.sqlite"
    dead = tmp_path / f"index.sqlite.{_dead_pid()}.tmp"
    dead.write_text("half written", encoding="utf-8")
    tmpfile.remove_dead_tmp(str(target))
    assert not dead.exists()


@pytest.mark.parametrize("pid_text", ["\u00b2", "\u2460", "\uff11\uff12", str(2 ** 31),
                                      "9" * 40])
def test_a_name_whose_number_names_no_process_is_left_and_the_write_goes_on(tmp_path,
                                                                           pid_text):
    target = tmp_path / "index.sqlite"
    left = tmp_path / f"index.sqlite.{pid_text}.tmp"
    left.write_text("half written", encoding="utf-8")
    with tmpfile.atomic_write(str(target)) as tmp_path_text:
        with open(tmp_path_text, "w", encoding="utf-8") as tmp_file:
            tmp_file.write("whole")
    assert target.read_text(encoding="utf-8") == "whole"
    assert left.exists()


def test_what_a_tool_wrote_beside_the_temporary_file_goes_with_it(tmp_path):
    target = tmp_path / "index.sqlite"
    pid = _dead_pid()
    journal = tmp_path / f"index.sqlite.{pid}.tmp-journal"
    journal.write_text("rollback", encoding="utf-8")
    tmpfile.remove_dead_tmp(str(target))
    assert not journal.exists()


def test_the_temporary_file_of_a_running_process_is_left_alone(tmp_path, monkeypatch):
    target = tmp_path / "index.sqlite"
    other = tmp_path / "index.sqlite.99999.tmp"
    other.write_text("being written", encoding="utf-8")
    monkeypatch.setattr(tmpfile, "process_dead", lambda pid: False)
    tmpfile.remove_dead_tmp(str(target))
    assert other.exists()


@pytest.mark.parametrize("name", ["index.sqlite.abc.tmp", "index.sqlite.123", "index.sqlite"])
def test_a_name_that_is_not_a_temporary_file_is_left_alone(tmp_path, name):
    target = tmp_path / "index.sqlite"
    other = tmp_path / name
    other.write_text("kept", encoding="utf-8")
    tmpfile.remove_dead_tmp(str(target))
    assert other.exists()


def test_a_directory_that_cannot_be_read_is_passed_over(tmp_path):
    tmpfile.remove_dead_tmp(str(tmp_path / "nowhere" / "index.sqlite"))


def test_a_file_that_is_already_gone_is_not_an_error(tmp_path):
    tmpfile._remove(str(tmp_path / "gone.tmp"))


def test_a_file_that_will_not_go_is_said_in_the_log(tmp_path, monkeypatch, caplog):
    stay = tmp_path / "stay.tmp"
    stay.write_text("x", encoding="utf-8")

    def refuse(path):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(os, "remove", refuse)
    tmpfile._remove(str(stay))
    assert "could not be removed" in caplog.text


# == atomic_write ==============================================
def test_the_block_writes_the_temporary_name_and_leaving_it_moves_it_into_place(tmp_path):
    target = tmp_path / "out.json"
    with tmpfile.atomic_write(str(target)) as tmp:
        assert tmp == tmpfile.temp_path(str(target))
        with open(tmp, "w", encoding="utf-8") as fp:
            fp.write("whole")
    assert target.read_text(encoding="utf-8") == "whole"
    assert not os.path.exists(tmp)


def test_a_write_that_fails_leaves_the_old_file_and_no_temporary_file(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("before", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with tmpfile.atomic_write(str(target)) as tmp:
            with open(tmp, "w", encoding="utf-8") as fp:
                fp.write("half")
            raise RuntimeError("stopped part-way")
    assert target.read_text(encoding="utf-8") == "before"
    assert not os.path.exists(tmp)


def test_the_block_finds_an_empty_file_of_its_own_under_the_temporary_name(tmp_path):
    target = tmp_path / "out.json"
    with tmpfile.atomic_write(str(target)) as tmp:
        assert os.path.isfile(tmp) and not os.path.islink(tmp)
        assert os.path.getsize(tmp) == 0
    assert target.read_bytes() == b""


def test_what_could_not_be_removed_is_handed_back(tmp_path, monkeypatch):
    target = tmp_path / "index.sqlite"
    own = tmp_path / f"index.sqlite.{os.getpid()}.tmp"
    own.write_text("half written", encoding="utf-8")
    assert tmpfile.remove_dead_tmp(str(target)) == []
    own.write_text("half written", encoding="utf-8")

    def refuse(path):
        raise PermissionError("the directory cannot be written to")

    monkeypatch.setattr(os, "remove", refuse)
    assert tmpfile.remove_dead_tmp(str(target)) == [str(own)]


@pytest.mark.parametrize("left", ["link", "link to nothing", "file", "beside"])
def test_what_stays_at_the_temporary_name_stops_the_write_before_anything_is_written(
        tmp_path, monkeypatch, left):
    """A link left at the name the write goes through, in a directory nothing can be
    removed from, is not written through."""
    directory = tmp_path / "copy"
    directory.mkdir()
    target = directory / "out.json"
    target.write_text("before", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    if left != "link to nothing":
        outside.write_text("the user's own", encoding="utf-8")
    name = f"out.json.{os.getpid()}.tmp"
    if left == "file":
        (directory / name).write_text("left", encoding="utf-8")
    elif left == "beside":
        os.symlink(outside, directory / f"{name}-journal")
    else:
        os.symlink(outside, directory / name)

    def refuse(path):
        raise PermissionError("the directory cannot be written to")

    monkeypatch.setattr(os, "remove", refuse)
    block_count = 0
    with pytest.raises(FileExistsError, match="could not be removed"):
        with tmpfile.atomic_write(str(target)) as tmp:
            block_count += 1
            with open(tmp, "w", encoding="utf-8") as fp:
                fp.write("whole")
    assert block_count == 0
    assert target.read_text(encoding="utf-8") == "before"
    if left == "link to nothing":
        assert not outside.exists()
    else:
        assert outside.read_text(encoding="utf-8") == "the user's own"


@pytest.mark.parametrize("to_nothing", [False, True])
def test_a_link_at_the_temporary_name_is_not_followed_where_nothing_was_listed(
        tmp_path, monkeypatch, to_nothing):
    """The temporary file is made as a new file: a link standing at its name that the
    removal did not see is not opened."""
    target = tmp_path / "out.json"
    outside = tmp_path / "outside.txt"
    if not to_nothing:
        outside.write_text("the user's own", encoding="utf-8")
    os.symlink(outside, tmpfile.temp_path(str(target)))
    monkeypatch.setattr(tmpfile, "remove_dead_tmp", lambda path: [])
    with pytest.raises(FileExistsError):
        with tmpfile.atomic_write(str(target)) as tmp:
            with open(tmp, "w", encoding="utf-8") as fp:
                fp.write("whole")
    assert not target.exists()
    assert not outside.exists() if to_nothing \
        else outside.read_text(encoding="utf-8") == "the user's own"


def test_a_leftover_of_another_write_that_stays_does_not_stop_this_one(tmp_path, caplog):
    target = tmp_path / "out.json"
    stay = tmp_path / f"out.json.{_dead_pid()}.tmp.d"
    stay.mkdir()
    with tmpfile.atomic_write(str(target)) as tmp:
        with open(tmp, "w", encoding="utf-8") as fp:
            fp.write("whole")
    assert target.read_text(encoding="utf-8") == "whole"
    assert stay.is_dir()
    assert "could not be removed" in caplog.text


def test_a_new_file_is_made_only_where_nothing_stands(tmp_path):
    path = tmp_path / "new.txt"
    tmpfile.make_new_file(str(path))
    assert path.read_bytes() == b""
    link = tmp_path / "link.txt"
    os.symlink(tmp_path / "nowhere.txt", link)
    for taken in (path, link, tmp_path):
        with pytest.raises(FileExistsError):
            tmpfile.make_new_file(str(taken))
    assert not (tmp_path / "nowhere.txt").exists()


def test_a_new_file_is_made_with_the_mode_it_is_given(tmp_path):
    path = tmp_path / "secret.txt"
    tmpfile.make_new_file(str(path), 0o600)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    with tmpfile.atomic_write(str(tmp_path / "a.txt"), 0o600) as tmp:
        assert stat.S_IMODE(os.stat(tmp).st_mode) == 0o600


# == read_json =================================================
def test_a_json_file_is_read_back(tmp_path):
    path = tmp_path / "a.json"
    path.write_text('{"a": ["あ", 1]}', encoding="utf-8")
    assert tmpfile.read_json(str(path)) == {"a": ["あ", 1]}


def test_a_file_that_is_not_there_is_said_apart_from_one_that_cannot_be_read(tmp_path):
    with pytest.raises(FileNotFoundError):
        tmpfile.read_json(str(tmp_path / "none.json"))


@pytest.mark.parametrize("make", [os.mkfifo, os.mkdir])
def test_what_is_no_regular_file_is_refused_without_waiting_on_it(tmp_path, make):
    path = tmp_path / "a.json"
    make(path)
    with pytest.raises(OSError, match="is not a regular file"):
        tmpfile.read_json(str(path))


@pytest.mark.parametrize("raw, why", [
    (b"[" * 200000, "nested deeper"), (b"{", "Expecting"), (b"\xff\xfe", "decode"),
    (b"[1, 2, 3]", "over the 4 ")])
def test_a_file_that_cannot_be_parsed_or_is_too_large_is_a_value_error(
        tmp_path, monkeypatch, raw, why):
    path = tmp_path / "a.json"
    path.write_bytes(raw)
    if why.startswith("over"):
        monkeypatch.setattr(tmpfile, "READ_MAX_BYTE", 4)
    with pytest.raises(ValueError, match=why):
        tmpfile.read_json(str(path))


def test_a_json_file_is_read_up_to_the_size_its_reader_names(tmp_path):
    path = tmp_path / "a.json"
    path.write_bytes(b'{"a": 1}' + b" " * 92)
    assert tmpfile.read_json(str(path), 100) == {"a": 1}
    assert tmpfile.read_file(str(path), 100) == path.read_bytes()
    for read in (tmpfile.read_json, tmpfile.read_file):
        with pytest.raises(ValueError, match="is 100 bytes, over the 99 read of one file"):
            read(str(path), 99)


def test_a_file_is_checked_against_the_size_its_reader_names(tmp_path, monkeypatch):
    path = tmp_path / "a.txt"
    path.write_bytes(b"x" * 100)
    monkeypatch.setattr(tmpfile, "READ_MAX_BYTE", 99)
    with pytest.raises(ValueError, match="is 100 bytes, over the 99 read of one file"):
        tmpfile.check_file(str(path))
    tmpfile.check_file(str(path), 100)
    tmpfile.check_file(os.fsencode(path), sys.maxsize)
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(OSError, match=f"^{re.escape(str(tmp_path))}/pipe is not a regular "
                                      f"file$"):
        tmpfile.check_file(os.fsencode(tmp_path / "pipe"), sys.maxsize)
    with pytest.raises(FileNotFoundError):
        tmpfile.check_file(str(tmp_path / "none"), sys.maxsize)


def test_a_json_file_there_is_no_memory_to_read_is_a_value_error(tmp_path, monkeypatch):
    path = tmp_path / "a.json"
    path.write_text("[{}, {}]", encoding="utf-8")

    def no_memory(text):
        raise MemoryError

    monkeypatch.setattr(tmpfile, "json", types.SimpleNamespace(loads=no_memory))
    with pytest.raises(ValueError, match="holds more values than there is memory"):
        tmpfile.read_json(str(path))


# == the stamp of a file a tool writes ==========================
def _write_at(path, text: str, second: int) -> None:
    """Write a file and date it as a write made within that second is."""
    path.write_text(text)
    os.utime(path, (second, second))


def test_a_write_of_the_same_size_in_the_same_second_is_dated_a_second_on(tmp_path):
    path = tmp_path / "m.py"
    _write_at(path, "x = 1\n", 1000)
    before_stamp = tmpfile.file_stamp(str(path))
    assert before_stamp == (1000, 6)
    _write_at(path, "x = 2\n", 1000)
    tmpfile.move_mtime_on(str(path), before_stamp)
    assert tmpfile.file_stamp(str(path)) == (1001, 6)


def test_a_write_of_another_size_is_left_with_the_time_it_was_written_at(tmp_path):
    """However many times the file is written, its time does not run ahead."""
    path = tmp_path / "m.py"
    _write_at(path, "x = 1\n", 1000)
    for count in range(2, 12):
        before_stamp = tmpfile.file_stamp(str(path))
        _write_at(path, "x = 1\n" + "y = 2\n" * count, 1000)
        tmpfile.move_mtime_on(str(path), before_stamp)
    assert tmpfile.file_stamp(str(path)) == (1000, 6 + 6 * 11)


def test_a_write_that_gives_the_file_a_stamp_it_had_before_the_last_one_is_dated_on(
        tmp_path):
    """Three writes in one second, the third as long as the first."""
    path = tmp_path / "m.py"
    _write_at(path, "x = 1\n", 1000)
    for text in ("x = 10\n", "x = 3\n"):
        before_stamp = tmpfile.file_stamp(str(path))
        _write_at(path, text, 1000)
        tmpfile.move_mtime_on(str(path), before_stamp)
    assert tmpfile.file_stamp(str(path)) == (1001, 6)


def test_a_file_written_where_one_was_removed_is_not_left_with_that_file_s_stamp(tmp_path):
    path = tmp_path / "m.py"
    assert tmpfile.file_stamp(str(path)) is None
    _write_at(path, "x = 1\n", 1000)
    tmpfile.move_mtime_on(str(path), None)
    assert tmpfile.file_stamp(str(path)) == (1000, 6)
    path.unlink()
    before_stamp = tmpfile.file_stamp(str(path))
    assert before_stamp == (1000, 6)
    _write_at(path, "x = 2\n", 1000)
    tmpfile.move_mtime_on(str(path), before_stamp)
    assert tmpfile.file_stamp(str(path)) == (1001, 6)
    # What stands at the path and is no regular file has no stamp
    path.unlink()
    path.mkdir()
    assert tmpfile.file_stamp(str(path)) is None


def test_a_write_in_a_later_second_is_left_as_it_is(tmp_path):
    path = tmp_path / "m.py"
    _write_at(path, "x = 1\n", 1000)
    before_stamp = tmpfile.file_stamp(str(path))
    _write_at(path, "x = 2\n", 1003)
    tmpfile.move_mtime_on(str(path), before_stamp)
    assert tmpfile.file_stamp(str(path)) == (1003, 6)
