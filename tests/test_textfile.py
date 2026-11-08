"""Writing a file back in the encoding it is stored in (textfile.py)."""

import codecs
import os
import random
import subprocess

import pytest
from codetwine.utils import file_utils
from conftest import git_repo

from codetwine_loom import textcut, textfile, tmpfile


@pytest.fixture(autouse=True)
def source_encoding(monkeypatch):
    """Decode what is not UTF-8 as Shift_JIS, whatever the environment says."""
    monkeypatch.setattr(file_utils, "SOURCE_ENCODING", ["cp932"])


def _write(tmp_path, raw: bytes) -> str:
    """Write raw to a file and return its path."""
    path = tmp_path / "f.txt"
    path.write_bytes(raw)
    return str(path)


def _edit(tmp_path, raw: bytes, edit) -> bytes:
    """Read raw back as a file, apply edit to its text and return what would be written."""
    old_text, form = textfile.read_for_edit(_write(tmp_path, raw))
    return textfile.encode_text(old_text, edit(old_text), form)


@pytest.mark.parametrize("raw, edit, want", [
    # The last line has no line ending, before and after
    (b"a\nb", lambda t: t.replace("b", "c"), b"a\nc"),
    # A line that loses its line ending, and one that gains one
    (b"a\nb", lambda t: "a", b"a"),
    (b"a\nb", lambda t: t + "\nc", b"a\nb\nc"),
    (b"a\nb\n", lambda t: t + "c\n", b"a\nb\nc\n"),
    # Everything goes
    (b"a\nb\n", lambda t: "", b""),
    # The line endings of the lines not touched are the file's own
    (b"a\r\nb\nc\r\n", lambda t: t.replace("b", "B"), b"a\r\nB\r\nc\r\n"),
    (b"a\nb\r\n", lambda t: "z\n" + t, b"z\r\na\nb\r\n"),
    # A lone \r ends a line, and stays where the edit did not reach
    (b"a\rb\nc\n", lambda t: t.replace("c", "d"), b"a\rb\nd\n"),
    (b"a\rb\nc\n", lambda t: t.replace("a\n", "z\n"), b"z\nb\nc\n"),
    # A file whose line breaks are all a lone \r gets a lone \r for the lines written
    (b"a\rb\r", lambda t: t.replace("b\n", "b\nc\n"), b"a\rb\rc\r"),
    # A \r in front of a \r\n ends a line of its own
    (b"a\r\r\nb\r\r\nc\r\r\n", lambda t: t.replace("b", "B"),
     b"a\r\r\nB\r\n\r\nc\r\r\n"),
    # An empty line that comes to stand behind a lone \r stays a line of its own: one
    # the file held, one the edit wrote, one at the end, and several in a row
    (b"a = 1\rb = 2\n\nc = 3\n", lambda t: t.replace("b = 2\n", ""), b"a = 1\r\rc = 3\n"),
    (b"x = 1\ry = 2\nz = 3\n", lambda t: t.replace("y = 2\n", "\ny = 2\n"),
     b"x = 1\r\ry = 2\nz = 3\n"),
    (b"a\r", lambda t: t + "\nQ", b"a\r\rQ"),
    (b"a\rb\n\n\nc\n", lambda t: t.replace("b\n", ""), b"a\r\r\rc\n"),
    (b"a\r\r\r", lambda t: t + "\n", b"a\r\r\r\r"),
])
def test_what_an_edit_writes(tmp_path, raw, edit, want):
    assert _edit(tmp_path, raw, edit) == want


def test_a_byte_python_writes_another_way_stays_where_the_edit_did_not_reach(tmp_path):
    """0xfbfc is 髙 as Windows writes it; Python's cp932 writes it as 0xeee0."""
    raw = "# 表\n".encode("cp932") + b"# \xfb\xfc\n" + b"x = 1\n"
    assert _edit(tmp_path, raw, lambda t: t.replace("x = 1", "x = 2")) == \
        raw.replace(b"x = 1", b"x = 2")


def test_a_file_encoded_whole_that_does_not_come_back_the_same_is_refused(tmp_path,
                                                                           monkeypatch):
    monkeypatch.setattr(textfile, "_raw_line_tuple", lambda body, text, codec: ())
    raw = b"# \xfb\xfc\n"
    with pytest.raises(ValueError, match="does not give back the same bytes"):
        textfile.read_for_edit(_write(tmp_path, raw))


def test_a_utf_16_file_is_written_line_by_line(tmp_path):
    raw = codecs.BOM_UTF16_BE + "a\r\n表\r\n".encode("utf-16-be")
    old_text, form = textfile.read_for_edit(_write(tmp_path, raw))
    assert len(form.raw_line_tuple) == 3 and old_text == "a\n表\n"
    assert textfile.encode_text(old_text, old_text.replace("a", "b"), form) == \
        codecs.BOM_UTF16_BE + "b\r\n表\r\n".encode("utf-16-be")


@pytest.mark.parametrize("bom, codec", [(codecs.BOM_UTF16_LE, "utf-16-le"),
                                        (codecs.BOM_UTF16_BE, "utf-16-be"),
                                        (codecs.BOM_UTF32_LE, "utf-32-le")])
def test_the_lines_of_a_utf_16_or_32_file_an_edit_did_not_reach_keep_their_endings(
        tmp_path, bom, codec):
    raw = bom + "a = 1\r\nb = 2\nc = 3\n".encode(codec)
    old_text, form = textfile.read_for_edit(_write(tmp_path, raw))
    assert textfile.encode_text(old_text, old_text, form) == raw
    assert textfile.encode_text(old_text, old_text.replace("c = 3", "c = 4"), form) == \
        bom + "a = 1\r\nb = 2\nc = 4\r\n".encode(codec)


def test_random_edits_keep_every_line_they_did_not_touch(tmp_path):
    """Whatever the edit, the text comes back and the lines around it keep their bytes."""
    rng = random.Random(0)
    word_list = ["a", "表", "能", "x = 1", "", "  return 0"]
    for _ in range(300):
        line_list = [rng.choice(word_list) for _ in range(rng.randint(0, 6))]
        ending_list = [rng.choice(["\n", "\r\n"]) for _ in line_list]
        raw_text = "".join(line + end for line, end in zip(line_list, ending_list))
        if rng.random() < 0.3 and line_list:
            raw_text = raw_text[:-len(ending_list[-1])]
        raw = raw_text.encode("cp932")
        old_text, form = textfile.read_for_edit(_write(tmp_path, raw))
        old_list = old_text.split("\n")
        start = rng.randint(0, len(old_list))
        end = rng.randint(start, len(old_list))
        new_list = old_list[:start] + [rng.choice(word_list) for _ in range(
            rng.randint(0, 3))] + old_list[end:]
        new_text = "\n".join(new_list)
        out = textfile.encode_text(old_text, new_text, form)
        assert out.decode(form.codec).replace("\r\n", "\n") == new_text
        # The lines in front of the edit are the file's own bytes, endings included,
        # unless the line was the last of either text, which has no ending
        piece_list = raw.split(b"\n")
        keep_count = min(start, len(piece_list) - 1, new_text.count("\n"))
        assert out.startswith(b"".join(piece + b"\n" for piece in piece_list[:keep_count]))


@pytest.mark.parametrize("ending_tuple", [
    ("\n",), ("\r\n",), ("\r",), ("\n", "\r\n"), ("\r", "\r\n"), ("\r", "\n"),
    ("\r", "\n", "\r\n")])
@pytest.mark.parametrize("codec", ["utf-8", "cp932", "utf-16-le"])
def test_an_edit_reads_back_line_for_line_whatever_the_line_endings(tmp_path, ending_tuple,
                                                                    codec):
    """Read, edit anywhere, write and read again: the text is the edited one, and the
    lines in front of the edit and behind it hold the bytes they held."""
    rng = random.Random(2)
    bom = codecs.BOM_UTF16_LE if codec == "utf-16-le" else b""
    for _ in range(150):
        raw_text = "".join(rng.choice(["def a():", "", "x = '表'", "\tz"])
                           + rng.choice(ending_tuple) for _ in range(rng.randint(0, 7)))
        if rng.random() < 0.3:
            raw_text += "y = 2"
        path = _write(tmp_path, bom + raw_text.encode(codec))
        old_text, form = textfile.read_for_edit(path)
        start = rng.randint(0, len(old_text))
        end = rng.randint(start, min(len(old_text), start + 12))
        new_text = old_text[:start] + rng.choice(
            ["", "Q", "\n", "Q\n", "\nQ", "\n\n", "Q\nR\n"]) + old_text[end:]
        out = textfile.encode_text(old_text, new_text, form)
        _write(tmp_path, out)
        assert out.startswith(bom)
        assert file_utils.read_source_text(path) == new_text, (raw_text, new_text, out)
        new_form = textfile.read_for_edit(path)[1]
        old_list, new_list = old_text.split("\n"), new_text.split("\n")
        assert len(new_form.raw_line_tuple) == len(new_list)
        head = 0
        while head < min(len(old_list), len(new_list)) and old_list[head] == new_list[head]:
            head += 1
        tail = 0
        while tail < min(len(old_list), len(new_list)) - head \
                and old_list[-1 - tail] == new_list[-1 - tail]:
            tail += 1
        old_body_list = [line for line, _ in form.raw_line_tuple]
        new_body_list = [line for line, _ in new_form.raw_line_tuple]
        assert new_body_list[:head] == old_body_list[:head]
        assert new_body_list[len(new_list) - tail:] == old_body_list[len(old_list) - tail:]


@pytest.mark.parametrize("codec", ["utf-8", "cp932", "utf-16-le", "utf-32-be"])
def test_the_text_to_edit_has_the_lines_the_reading_side_counts(tmp_path, codec):
    """Whatever the line breaks, the text and its stored lines are split the same way."""
    rng = random.Random(1)
    bom = {"utf-16-le": codecs.BOM_UTF16_LE, "utf-32-be": codecs.BOM_UTF32_BE}.get(codec, b"")
    for _ in range(200):
        raw_text = "".join(rng.choice(["a", "表", "", "x = 1"])
                           + rng.choice(["\n", "\r\n", "\r", "\r\r\n", ""])
                           for _ in range(rng.randint(0, 8)))
        raw = bom + raw_text.encode(codec)
        path = _write(tmp_path, raw)
        old_text, form = textfile.read_for_edit(path)
        assert old_text == file_utils.read_source_text(path)
        assert "\r" not in old_text
        assert textcut.line_list_of(old_text) == file_utils.line_list_of(raw_text)
        assert len(form.raw_line_tuple) == len(old_text.split("\n"))
        assert b"".join(line + ending for line, ending in form.raw_line_tuple) == \
            raw[len(bom):]
        assert textfile.encode_text(old_text, old_text, form) == raw


def test_a_file_that_is_not_there_yet_follows_nothing_without_an_extension(tmp_path):
    assert textfile.new_file_form(str(tmp_path), "Makefile") == textfile.UTF8_FORM


def test_what_is_no_regular_file_or_too_large_is_not_read_for_edit(tmp_path,
                                                                    monkeypatch):
    os.mkfifo(tmp_path / "pipe.py")
    with pytest.raises(OSError, match="is not a regular file"):
        textfile.read_for_edit(str(tmp_path / "pipe.py"))
    monkeypatch.setattr(tmpfile, "READ_MAX_BYTE", 4)
    with pytest.raises(ValueError, match="is 5 bytes, over the 4 read of one file"):
        textfile.read_for_edit(_write(tmp_path, b"x = 1"))
    assert textfile.read_for_edit(_write(tmp_path, b"x=1"))[0] == "x=1"


def test_a_new_file_follows_the_files_of_its_language_that_can_be_read(tmp_path,
                                                                       monkeypatch):
    """A tracked path that is a named pipe now is passed over without waiting on it,
    and so is a file larger than the read of one file."""
    project = tmp_path / "p"
    git_repo(project)
    (project / "sjis.py").write_bytes("# あ\n".encode("cp932"))
    (project / "pipe.py").write_text("x = 1\n")
    (project / "large.py").write_bytes("# é\n".encode() * 9)
    subprocess.run(["git", "-C", str(project), "add", "-A"], check=True,
                   capture_output=True)
    (project / "pipe.py").unlink()
    os.mkfifo(project / "pipe.py")
    monkeypatch.setattr(tmpfile, "READ_MAX_BYTE", 40)
    assert textfile.new_file_form(str(project), "new.py") == textfile.TextForm(
        "cp932", b"", "\n")
    monkeypatch.setattr(tmpfile, "READ_MAX_BYTE", 64)
    assert textfile.new_file_form(str(project), "new.py") == textfile.UTF8_FORM
