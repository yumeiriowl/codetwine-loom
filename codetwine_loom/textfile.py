"""Writing the target repository's files in the encoding each is stored in.

A file is decoded by codetwine's read_source(): the codec its BOM names, UTF-8, the
encodings of SOURCE_ENCODING, a detected encoding, and last UTF-8 with each invalid byte
replaced. What is written goes back out in the encoding the file was read in, with its
BOM, and the lines an edit did not touch keep their bytes and their line endings.
"""

import codecs
import os
import re
from dataclasses import dataclass

from codetwine.utils.file_utils import is_text_file, lone_cr_to_lf, read_source

from codetwine_loom import gitcmd, tmpfile
from codetwine_loom.languages import ALIAS_DICT, ext_of

# The codecs read_source() names for a file it decoded by its BOM, each with the BOMs
# it may have been and the codec that writes the body after that BOM
_BOM_CODEC_DICT: dict[str, tuple[tuple[bytes, str], ...]] = {
    "utf-8-sig": ((codecs.BOM_UTF8, "utf-8"),),
    "utf-16": ((codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be")),
    "utf-32": ((codecs.BOM_UTF32_LE, "utf-32-le"), (codecs.BOM_UTF32_BE, "utf-32-be")),
}

# A line break ("\r\n", a lone "\r" or "\n"), in a text and in the bytes of a codec that
# writes "\r" and "\n" as the bytes 0x0d and 0x0a
_BREAK_RE = re.compile(r"\r\n|\r|\n")
_BYTE_BREAK_RE = re.compile(rb"\r\n|\r|\n")


@dataclass(frozen=True)
class TextForm:
    """How one file's text is turned into its bytes."""

    codec: str      # what the text is encoded with
    bom: bytes      # what goes in front of it
    newline: str    # what a "\n" of a line the edit wrote is written as
    # The file's lines as they are stored, as (the line without its ending, the ending),
    # the ending b"" on the last one. Empty for a file whose lines are not lines of its
    # bytes one for one and for a file that is not there yet: those are encoded whole
    raw_line_tuple: tuple[tuple[bytes, bytes], ...] = ()


UTF8_FORM = TextForm("utf-8", b"", "\n")


def _write_codec(raw: bytes, encoding: str) -> tuple[str, bytes]:
    """Return the codec that writes a file's body and the BOM that goes in front of it.

    Examples:
        (b"\\xef\\xbb\\xbfx", "utf-8-sig") -> ("utf-8", b"\\xef\\xbb\\xbf")
        (b"\\xfe\\xff\\x00x", "utf-16")    -> ("utf-16-be", b"\\xfe\\xff")
        (b"x", "cp1252")                   -> ("cp1252", b"")

    Args:
        raw: The file's bytes.
        encoding: The encoding read_source() decoded them with.

    Returns:
        (codec name, BOM), the BOM empty for a file that has none.
    """
    name = codecs.lookup(encoding).name
    for bom, codec in _BOM_CODEC_DICT.get(name, ()):
        if raw.startswith(bom):
            return codec, bom
    return name, b""


def _line_pair_list(content: str | bytes, break_re: re.Pattern) -> list[tuple]:
    """Split a text or its bytes into its lines, each with the line break that ends it.

    Examples:
        ("a\\r\\nb\\rc\\n", _BREAK_RE) -> [("a", "\\r\\n"), ("b", "\\r"), ("c", "\\n"), ("", "")]
        ("a\\r\\r\\nb", _BREAK_RE)    -> [("a", "\\r"), ("", "\\r\\n"), ("b", "")]
        ("", _BREAK_RE)             -> [("", "")]

    Args:
        content: A str, or the bytes of a codec that writes "\\r" and "\\n" as 0x0d and
            0x0a.
        break_re: _BREAK_RE for a str, _BYTE_BREAK_RE for bytes.

    Returns:
        (the line without its ending, the ending) for each line. The last pair is what
        follows the last line break, its ending empty.
    """
    pair_list = []
    at = 0
    for match in break_re.finditer(content):
        pair_list.append((content[at:match.start()], match.group()))
        at = match.end()
    pair_list.append((content[at:], content[:0]))
    return pair_list


def _raw_line_tuple(body: bytes, text: str, codec: str) -> tuple[tuple[bytes, bytes], ...]:
    """Split a file's bytes into its lines when each decodes to the same line of its text.

    A line ends at "\\n", at "\\r\\n" and at a lone "\\r". Where the codec writes "\\r"
    and "\\n" as the bytes 0x0d and 0x0a, the bytes are split at those bytes and each
    piece has to decode to its line. Otherwise (UTF-16, UTF-32) each line is encoded
    alone and the pieces have to make the bytes again.

    Args:
        body: The file's bytes after its BOM.
        text: What they decode to, line endings as they are.
        codec: What they were decoded with.

    Returns:
        (the line without its ending, the ending) for each line, the ending b"" on the
        last one, or () when a line does not decode to the same line alone, or the
        lines encoded one by one do not give back the bytes.
    """
    text_pair_list = _line_pair_list(text, _BREAK_RE)
    if "\r\n".encode(codec) == b"\r\n":
        byte_pair_list = _line_pair_list(body, _BYTE_BREAK_RE)
        if len(byte_pair_list) != len(text_pair_list):
            return ()
        try:
            if any(line.decode(codec) != text_line or ending.decode(codec) != text_ending
                   for (line, ending), (text_line, text_ending)
                   in zip(byte_pair_list, text_pair_list)):
                return ()
        except UnicodeDecodeError:
            return ()
        return tuple(byte_pair_list)
    try:
        byte_pair_list = [(text_line.encode(codec), text_ending.encode(codec))
                          for text_line, text_ending in text_pair_list]
    except UnicodeEncodeError:
        return ()
    if b"".join(line + ending for line, ending in byte_pair_list) != body:
        return ()
    return tuple(byte_pair_list)


def lf_text(text: str) -> str:
    """Return a text with "\\n" as its only line break.

    Examples:
        "a\\r\\nb\\rc\\n" -> "a\\nb\\nc\\n"

    Args:
        text: A text with any of the three line breaks.

    Returns:
        The text, each "\\r\\n" and each lone "\\r" turned into "\\n".
    """
    return lone_cr_to_lf(text).replace("\r\n", "\n")


def read_for_edit(path: str) -> tuple[str, TextForm]:
    """Read a file that is to be written back, with how to write it.

    Args:
        path: The file's absolute path.

    Returns:
        (the text as lf_text() gives it, its TextForm). newline is "\\r\\n" when the
        file holds one, "\\r" when its line breaks are all a lone "\\r", and "\\n"
        otherwise.

    Raises:
        OSError: When the file cannot be read or is not a regular file
            (tmpfile.read_file).
        ValueError: When the file is larger than tmpfile.READ_MAX_BYTE, when no encoding
            tried decodes it, or when it is encoded whole and encoding the decoded text
            again does not give back the file's bytes.
    """
    raw = tmpfile.read_file(path)
    text, encoding = read_source(path)
    if not encoding:
        raise ValueError("none of the encodings tried decodes it. Set SOURCE_ENCODING to "
                         "the encoding it is stored in")
    codec, bom = _write_codec(raw, encoding)
    raw_line_tuple = _raw_line_tuple(raw[len(bom):], text, codec)
    if not raw_line_tuple and bom + text.encode(codec, errors="replace") != raw:
        raise ValueError(f"it was decoded as {encoding}, which does not give back the same "
                         f"bytes when it is written. Set SOURCE_ENCODING to the encoding "
                         f"it is stored in")
    if "\r\n" in text:
        newline = "\r\n"
    elif "\r" in text and "\n" not in text:
        newline = "\r"
    else:
        newline = "\n"
    return lf_text(text), TextForm(codec, bom, newline, raw_line_tuple)


def _encode_line(line: str, form: TextForm, text: str) -> bytes:
    """Encode one line an edit wrote.

    Args:
        line: The line, without its ending.
        form: How the file is stored.
        text: The whole text after the edit, for the line number of a character the
            codec cannot write.

    Returns:
        The line's bytes.

    Raises:
        ValueError: When the text holds a character the codec cannot write. The message
            names every such character in it and the line of the first.
    """
    try:
        return line.encode(form.codec)
    except UnicodeEncodeError:
        pass
    bad_set = set()
    for char in set(text):
        try:
            char.encode(form.codec)
        except UnicodeEncodeError:
            bad_set.add(char)
    first = min(text.index(char) for char in bad_set)
    raise ValueError(f"{', '.join(repr(char) for char in sorted(bad_set))} cannot be written in "
                     f"{form.codec}, the encoding this file is stored in (first at line "
                     f"{text.count(chr(10), 0, first) + 1})")


def encode_text(old_text: str, new_text: str, form: TextForm) -> bytes:
    """Turn the text an edit left into the bytes the file is to hold.

    The lines in front of the first line the edit changed and after the last are the
    file's own bytes, line endings included. The lines between are encoded with the
    file's codec and end with form.newline. An empty line ending in "\\n" that comes to
    stand behind a line ending in a lone "\\r" is written with a lone "\\r" itself, and
    the two line breaks stay two. A file with no raw_line_tuple is encoded whole.

    Args:
        old_text: The text before the edit, with "\\n" line breaks, as read_for_edit()
            returned it. "" for a new file.
        new_text: The text after the edit, with "\\n" line breaks.
        form: How the file is stored.

    Returns:
        The BOM followed by the encoded text.

    Raises:
        ValueError: When a line the edit wrote holds a character the codec cannot write.
    """
    newline = form.newline.encode(form.codec)
    new_list = new_text.split("\n")
    if not form.raw_line_tuple:
        return form.bom + newline.join(_encode_line(line, form, new_text)
                                       for line in new_list)
    old_list = old_text.split("\n")
    limit = min(len(old_list), len(new_list))
    head = 0
    while head < limit and old_list[head] == new_list[head]:
        head += 1
    tail = 0
    while tail < limit - head and old_list[-1 - tail] == new_list[-1 - tail]:
        tail += 1
    pair_list = []
    for new_index, line in enumerate(new_list):
        if new_index < head:
            old_index = new_index
        elif new_index >= len(new_list) - tail:
            old_index = new_index - len(new_list) + len(old_list)
        else:
            old_index = None
        body = form.raw_line_tuple[old_index][0] if old_index is not None \
            else _encode_line(line, form, new_text)
        if new_index == len(new_list) - 1:
            ending = b""
        elif old_index is not None and old_index < len(old_list) - 1:
            ending = form.raw_line_tuple[old_index][1]
        else:
            ending = newline
        pair_list.append((body, ending))
    # An empty line ending in "\n" behind a lone "\r" is written with a lone "\r"
    lone_cr, empty_line = "\r".encode(form.codec), (b"", "\n".encode(form.codec))
    part_list = []
    before_ending = b""
    for body, ending in pair_list:
        if before_ending == lone_cr and (body, ending) == empty_line:
            ending = lone_cr
        part_list.append(body + ending)
        before_ending = ending
    return form.bom + b"".join(part_list)


def new_file_form(project_dir: str, path: str) -> TextForm:
    """Return how a file that is not there yet is to be stored.

    The files git tracks or does not ignore with the same extension, or with another
    extension of the same language (ALIAS_DICT), are read, and those holding a byte
    outside ASCII say how they are stored. A path that is not a regular file of at most
    tmpfile.READ_MAX_BYTE bytes is not read (tmpfile.read_file) and says nothing. When
    they all say the same, the new file is stored that way; otherwise, and for a path
    with no extension, as UTF-8. The line ending is "\\n" either way.

    Args:
        project_dir: The repository.
        path: The new file's repository-relative path.

    Returns:
        The TextForm to write the new file with.
    """
    ext = ext_of(os.path.basename(path))
    if not ext:
        return UTF8_FORM
    lang = ALIAS_DICT.get(ext, ext)
    form_set = set()
    for file_rel in gitcmd.work_tree_file_list(project_dir) or []:
        other_ext = ext_of(os.path.basename(file_rel))
        if not other_ext or ALIAS_DICT.get(other_ext, other_ext) != lang:
            continue
        file_abs = os.path.join(project_dir, file_rel)
        try:
            raw = tmpfile.read_file(file_abs)
            if raw.isascii() or not is_text_file(file_abs):
                continue
            encoding = read_source(file_abs)[1]
        except (OSError, ValueError):
            continue
        if encoding:
            form_set.add(_write_codec(raw, encoding))
    if len(form_set) != 1:
        return UTF8_FORM
    codec, bom = form_set.pop()
    return TextForm(codec, bom, "\n")
