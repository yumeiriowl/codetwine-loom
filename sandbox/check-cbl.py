#!/usr/bin/env python3
"""The default static check for COBOL.

Checks every program among the .cbl and .cob files git tracks or does not ignore in the
current directory with cobc -fsyntax-only, one program at a time, and prints the
compiler's error lines, one "<path>:<line>: error: <message>" line each. Prints nothing
when there is no program.

The files are checked as copies, each line and column kept, in which:
- each EXEC ... END-EXEC block is replaced by CONTINUE where it is a statement, by a COPY
  statement where it is an EXEC SQL INCLUDE of a file the repository holds, and by
  spaces otherwise
- DFHRESP(name) and DFHVALUE(name) are replaced by the number 0
- the file a COPY statement names is found among the COBOL files by its name without
  extension, in either case, and the file an EXEC SQL INCLUDE names among every file,
  the COBOL files first
- a COPY statement that names no COBOL file finds a file of another extension or of none
  that it names whole (a literal with the extension, or a name without one), or else one
  whose name without extension it is and that holds a data description or a file
  description. The COPY statements of such a file are followed as well
- a COPY statement that names no COBOL file and names the mapset of a BMS source (.bms)
  of the repository copies the symbolic map made from that source
Every directory holding such a file is a copybook directory (-I). A copybook the
repository does not hold and a name that is not defined are not reported when their
names start with what the CICS, IMS, DB2 and MQ copybooks and translators use (DFH, EIB,
DIB, SQL, MQ, CMQ, DSN). In a program that copies another copybook the repository does
not hold, no name that is not defined is reported. In a program that copies a symbolic
map made here, a name that is not defined is not reported when it is a name of that map
(a field, a group or a map) followed by one character.

The settings COBOL_DIALECT (the dialect, cobc's -std=) and COBOL_COPYBOOK_DIR (copybook
directories separated by commas, relative to the repository) are passed to cobc, and so
are the arguments after them: "ctloom-check-cbl -std=ibm -I copylib" checks in the IBM
dialect with the directory copylib. The copybook directories are searched after those
of the repository.

Exits 127 when cobc is not installed, or when cobc does not take what it is given.
"""

import concurrent.futures
import dataclasses
import os
import re
import shutil
import subprocess
import sys
import tempfile

_CANNOT_RUN = 127

# The extensions of the COBOL files, and of the programs among them
_COBOL_EXT_SET = frozenset({".cbl", ".cob", ".cpy"})
_PROGRAM_EXT_SET = frozenset({".cbl", ".cob"})

# The columns (0-based) of a fixed-format line: the indicator, the start of the code and
# of area B, and the end of the code
_INDICATOR_COLUMN = 6
_CODE_COLUMN = 7
_AREA_B_COLUMN = 11
_CODE_END_COLUMN = 72
_INDICATOR_CHAR_SET = frozenset(" -*/dD$")
_COMMENT_CHAR_SET = frozenset("*/")

# A word of the code, a literal, or a separator period
_TOKEN_RE = re.compile(r"""(?P<literal>"[^"]*"?|'[^']*'?)|(?P<word>[\w-]+)|(?P<period>\.(?=\s|$))""")

# What starts a program, and the headers of the divisions that change how EXEC is replaced
_PROGRAM_ID_RE = re.compile(r"(?<![\w-])PROGRAM-ID(?![\w-])", re.IGNORECASE)
_FREE_DIRECTIVE_RE = re.compile(r">>\s*SOURCE\s+(?:FORMAT\s+)?(?:IS\s+)?FREE|"
                                r"\$\s*SET\s+SOURCEFORMAT\s*\(?\s*[\"']?FREE",
                                re.IGNORECASE)

# The EXEC SQL statements that declare rather than do something
_SQL_DECLARE_WORD_SET = frozenset({"INCLUDE", "DECLARE", "BEGIN", "END"})

# One error line of cobc
_ERROR_RE = re.compile(r"^(?P<path>[^:\s][^:]*):(?P<line>\d+): error: (?P<message>.*)$")
_NO_COPYBOOK_RE = re.compile(r"^(?P<name>\S+): No such file or directory$")
_NOT_DEFINED_RE = re.compile(r"^'(?P<name>[^']+)' is not defined$")

# The name starts of the copybooks and names the CICS, IMS, DB2 and MQ translators and
# copybooks supply
_PRODUCT_PREFIX_TUPLE = ("DFH", "EIB", "DIB", "SQL", "MQ", "CMQ", "DSN")

# A CICS condition or value the translator turns into a number
_CICS_VALUE_RE = re.compile(r"(?<![\w-])DFH(?:RESP|VALUE)\s*\(\s*[\w-]+\s*\)", re.IGNORECASE)

# The extension a copybook found under another name is written with
_COPYBOOK_EXT = ".cpy"

# A level number that starts a data description, and the words that start a file
# description
_LEVEL_RE = re.compile(r"0?[1-9]|[1-4][0-9]|66|77|88")
_FILE_WORD_SET = frozenset({"FD", "SD"})

# The kinds of the names _copy_name_list() returns: a COPY of a name, a COPY of a
# literal, an EXEC SQL INCLUDE
_COPY_KIND = "copy"
_LITERAL_KIND = "literal"
_INCLUDE_KIND = "include"

# How long one program may take
_TIMEOUT_SEC = 120

# == BMS =====================================================================
# The extension of a BMS source
_BMS_EXT = ".bms"

# The column (0-based) that marks an assembler line as continued, and where the
# continuation starts
_BMS_CONTINUE_COLUMN = 71
_BMS_CONTINUE_START = 15

# The extended attributes of a field and the letter their field of the symbolic map
# ends with, in the order the map holds them
_BMS_ATTRIBUTE_TUPLE = (("COLOR", "C"), ("PS", "P"), ("HILIGHT", "H"), ("VALIDN", "V"),
                        ("OUTLINE", "U"), ("SOSI", "M"), ("TRANSP", "T"))

# The attributes EXTATT=YES gives every field when DSATTS names none
_BMS_EXTATT_TUPLE = ("COLOR", "PS", "HILIGHT", "VALIDN")

# The length of the prefix TIOAPFX=YES puts in front of a map
_BMS_PREFIX_LENGTH = 12

# A name BMS gives the map that is also a COBOL word
_BMS_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")

# The program the arguments are tried on
_TRY_PROGRAM = """       IDENTIFICATION DIVISION.
       PROGRAM-ID. CTLOOMCK.
       PROCEDURE DIVISION.
           STOP RUN.
"""


def _git_file_list() -> list[str]:
    """Return the files git tracks or does not ignore, that exist."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                         capture_output=True, check=True).stdout.decode("utf-8", "surrogateescape")
    return sorted({path for path in out.split("\0") if path and os.path.isfile(path)})


def _is_free(line_list: list[str]) -> bool:
    """Say whether a file is in free format: a FREE directive, or a line fixed format cannot hold."""
    for line in line_list:
        if _FREE_DIRECTIVE_RE.search(line):
            return True
        if line.strip() and len(line) > _INDICATOR_COLUMN \
                and line[_INDICATOR_COLUMN] not in _INDICATOR_CHAR_SET:
            return True
    return False


def _code_span(line: str, is_free: bool) -> tuple[int, int]:
    """Return where the code of a line starts and ends, an empty span for a comment line."""
    if is_free:
        comment_at = line.find("*>")
        return 0, len(line) if comment_at < 0 else comment_at
    if len(line) <= _INDICATOR_COLUMN or line[_INDICATOR_COLUMN] in _COMMENT_CHAR_SET:
        return 0, 0
    comment_at = line.find("*>", _CODE_COLUMN)
    return _CODE_COLUMN, len(line) if comment_at < 0 else comment_at


def _token_list(line_list: list[str], is_free: bool,
                keep_literal: bool = False) -> list[tuple[str, str, int, int, int]]:
    """Return the words and periods of the code, and the literals when keep_literal is
    true, as (kind, text as written, line, start, end)."""
    token_list = []
    for i, line in enumerate(line_list):
        start, end = _code_span(line, is_free)
        for match in _TOKEN_RE.finditer(line, start, end):
            if keep_literal or match.lastgroup != "literal":
                token_list.append((match.lastgroup, match.group(), i, match.start(),
                                   match.end()))
    return token_list


def _blank(line_list: list[str], first: tuple[int, int], last: tuple[int, int],
           is_free: bool) -> None:
    """Put spaces over the code from first (line, column) up to last (line, column)."""
    for i in range(first[0], last[0] + 1):
        start, end = _code_span(line_list[i], is_free)
        start = max(start, first[1]) if i == first[0] else start
        end = min(end, last[1]) if i == last[0] else end
        if start < end:
            line_list[i] = line_list[i][:start] + " " * (end - start) + line_list[i][end:]


def _put(line_list: list[str], line: int, column: int, text: str) -> None:
    """Write text over one line from a column on."""
    line_list[line] = line_list[line][:column] + text + line_list[line][column + len(text):]


def _put_statement(line_list: list[str], first: tuple[int, int], last_line: int,
                   is_free: bool) -> None:
    """Write CONTINUE where a block that was a statement started.

    In fixed format, where CONTINUE does not end by column 72 there, it goes to column
    12 of the next line of the block that holds code, or is left out when there is none.
    """
    line, column = first
    if is_free or column + len("CONTINUE") <= _CODE_END_COLUMN:
        _put(line_list, line, column, "CONTINUE")
        return
    for i in range(line + 1, last_line + 1):
        if _code_span(line_list[i], is_free) != (0, 0):
            _put(line_list, i, _AREA_B_COLUMN, "CONTINUE")
            return


def replace_exec(text: str, copybook_set: frozenset[str]) -> str:
    """Replace every EXEC ... END-EXEC block of a COBOL file, keeping its lines and columns.

    Args:
        text: The file, read as Latin-1 (one character per byte).
        copybook_set: The upper-case names, without extension, of the files the
            repository holds.

    Returns:
        The file with each block replaced: by CONTINUE where it is a statement, by
        "COPY <name>" where it is an EXEC SQL INCLUDE of a name in copybook_set, by
        spaces otherwise, the separator period after a block that is not a statement
        included. DFHRESP(name) and DFHVALUE(name) become 0.
    """
    line_list = text.split("\n")
    is_free = _is_free(line_list)
    for i, line in enumerate(line_list):
        start, end = _code_span(line, is_free)
        code = _CICS_VALUE_RE.sub(lambda match: "0".ljust(len(match.group())),
                                  line[start:end])
        line_list[i] = line[:start] + code + line[end:]
    token_list = _token_list(line_list, is_free)
    in_procedure = False
    k = 0
    upper_list = [token[1].upper() for token in token_list]
    while k < len(token_list):
        _, _, line, start, _ = token_list[k]
        word = upper_list[k]
        if word in ("IDENTIFICATION", "ID") and upper_list[k + 1:k + 2] == ["DIVISION"]:
            in_procedure = False
        elif word == "PROCEDURE" and upper_list[k + 1:k + 2] == ["DIVISION"]:
            in_procedure = True
        if word != "EXEC":
            k += 1
            continue
        end_k = next((j for j in range(k + 1, len(token_list))
                      if upper_list[j] == "END-EXEC"), None)
        if end_k is None:
            break
        raw_word_list = [token[1] for token in token_list[k + 1:end_k] if token[0] == "word"]
        body_list = [written.upper() for written in raw_word_list]
        is_declare = len(body_list) > 1 and body_list[0] == "SQL" \
            and body_list[1] in _SQL_DECLARE_WORD_SET
        _, _, end_line, _, end_column = token_list[end_k]
        is_statement = not is_declare and (in_procedure or body_list[:1] != ["SQL"])
        if not is_statement and end_k + 1 < len(token_list) \
                and token_list[end_k + 1][0] == "period":
            _, _, end_line, _, end_column = token_list[end_k + 1]
            end_k += 1
        _blank(line_list, (line, start), (end_line, end_column), is_free)
        if is_statement:
            _put_statement(line_list, (line, start), end_line, is_free)
        elif body_list[:2] == ["SQL", "INCLUDE"] and len(body_list) > 2 \
                and body_list[2] in copybook_set:
            _put(line_list, line, start, f"COPY {raw_word_list[2]}.")
        k = end_k + 1
    return "\n".join(line_list)


def _stem(path: str) -> str:
    """Return a file's name without its extension, in upper case."""
    return os.path.splitext(os.path.basename(path))[0].upper()


def _write_copy(path: str, target: str, copybook_set: frozenset[str]) -> None:
    """Write one file of the repository to target, with replace_exec() applied."""
    with open(path, encoding="latin-1", newline="") as fp:
        text = fp.read()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="latin-1", newline="") as fp:
        fp.write(replace_exec(text, copybook_set))


def _read_line_list(path: str) -> list[str]:
    """Return the lines of a file read as Latin-1."""
    with open(path, encoding="latin-1") as fp:
        return fp.read().split("\n")


def _copy_name_list(path: str) -> list[tuple[str, str]]:
    """Return what a file's COPY statements and EXEC SQL INCLUDEs name.

    Returns:
        (name, kind) in the order they are written: the name after COPY (_COPY_KIND),
        the literal after COPY without its quotes (_LITERAL_KIND), the name after EXEC
        SQL INCLUDE (_INCLUDE_KIND).
    """
    line_list = _read_line_list(path)
    token_list = _token_list(line_list, _is_free(line_list), keep_literal=True)
    upper_list = [text.upper() if kind == "word" else "" for kind, text, *_ in token_list]
    name_list = []
    for k in range(len(token_list) - 1):
        kind, text = token_list[k + 1][:2]
        if upper_list[k] == "COPY" and kind == "word":
            name_list.append((text, _COPY_KIND))
        elif upper_list[k] == "COPY" and kind == "literal" and len(text) > 2 \
                and text[-1] == text[0]:
            name_list.append((text[1:-1], _LITERAL_KIND))
        elif upper_list[k - 1:k + 1] == ["SQL", "INCLUDE"] and kind == "word":
            name_list.append((text, _INCLUDE_KIND))
    return name_list


def _has_data_item(path: str) -> bool:
    """Say whether a file read as COBOL holds a data description or a file description."""
    try:
        line_list = _read_line_list(path)
    except OSError:
        return False
    token_list = _token_list(line_list, _is_free(line_list))
    previous_kind = "period"
    for k, (kind, text, *_) in enumerate(token_list):
        if previous_kind == "period" and kind == "word" and k + 1 < len(token_list) \
                and token_list[k + 1][0] == "word" \
                and (_LEVEL_RE.fullmatch(text) or text.upper() in _FILE_WORD_SET):
            return True
        previous_kind = kind
    return False


@dataclasses.dataclass(frozen=True)
class _FileLookup:
    """The tables the file a COPY statement or an EXEC SQL INCLUDE names is found in."""

    cobol_stem_dict: dict[str, str]   # {upper-case name without extension: COBOL file}
    any_stem_dict: dict[str, str]     # the same over every file, the COBOL files first
    # {upper-case file name: path} of the files other than BMS sources, the COBOL files
    # first
    base_dict: dict[str, str]
    # {upper-case name without extension: path} of the files other than COBOL files and
    # BMS sources
    other_stem_dict: dict[str, str]


def _file_lookup(git_file_list: list[str], cobol_file_list: list[str]) -> _FileLookup:
    """Build the tables _mirror() finds the copied files in."""
    cobol_stem_dict: dict[str, str] = {}
    base_dict: dict[str, str] = {}
    for path in cobol_file_list:
        cobol_stem_dict.setdefault(_stem(path), path)
        base_dict.setdefault(os.path.basename(path).upper(), path)
    any_stem_dict = dict(cobol_stem_dict)
    other_stem_dict: dict[str, str] = {}
    cobol_file_set = set(cobol_file_list)
    for path in git_file_list:
        any_stem_dict.setdefault(_stem(path), path)
        if path not in cobol_file_set and os.path.splitext(path)[1].lower() != _BMS_EXT:
            base_dict.setdefault(os.path.basename(path).upper(), path)
            other_stem_dict.setdefault(_stem(path), path)
    return _FileLookup(cobol_stem_dict, any_stem_dict, base_dict, other_stem_dict)


def _named_copybook(name: str, kind: str, lookup: _FileLookup) -> str | None:
    """Return the file a COPY statement or an EXEC SQL INCLUDE names, or None.

    An EXEC SQL INCLUDE takes any file by its name without extension. A COPY of a name
    takes a COBOL file by its name without extension, else a file whose whole name it
    is, else a file other than a COBOL file by its name without extension when that
    file holds a data description or a file description. A COPY of a literal takes the
    file whose whole name the literal ends with.

    Args:
        name: The name or the literal, as _copy_name_list() returns it.
        kind: Its kind, as _copy_name_list() returns it.
        lookup: The tables of the repository's files.
    """
    base = os.path.basename(name).upper()
    if kind == _INCLUDE_KIND:
        return lookup.any_stem_dict.get(base)
    if kind == _LITERAL_KIND:
        return lookup.base_dict.get(base)
    real = lookup.cobol_stem_dict.get(base) or lookup.base_dict.get(base)
    if real is None:
        real = lookup.other_stem_dict.get(base)
        if real is not None and not _has_data_item(real):
            real = None
    return real


def _copy_file_name(name: str, kind: str, real: str) -> str:
    """Return the file name the copy of a copied file is written under.

    It is the name cobc looks for: the literal as written, the file's own name when that
    is the COPY name with a COBOL extension or none, otherwise the name with .cpy.
    """
    if kind == _LITERAL_KIND:
        return os.path.basename(name)
    if os.path.basename(real) in (name + ext for ext in (".cpy", ".CPY", ".cbl", ".CBL",
                                                         ".cob", ".COB", "")):
        return os.path.basename(real)
    return name + _COPYBOOK_EXT


def _write_symbolic_map(name: str, bms_path: str, map_list: list, root: str,
                        alias_dict: dict[str, str]) -> None:
    """Write the symbolic map of a BMS source under the name a COPY statement uses.

    Args:
        name: The name after COPY, the mapset's name.
        bms_path: The BMS source.
        map_list: Its maps, as read_bms() returns them.
        root: The directory the copy is written under.
        alias_dict: {path of a copy written under another name: the file it is}. It is
            added to.
    """
    alias = os.path.join(os.path.dirname(bms_path), name + _COPYBOOK_EXT)
    target = os.path.join(root, alias)
    if os.path.exists(target):
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="latin-1") as fp:
        fp.write(symbolic_map_text(map_list))
    alias_dict[alias] = bms_path


def _mirror(git_file_list: list[str], cobol_file_list: list[str],
            root: str) -> tuple[list[str], dict[str, str], dict[str, set[str]]]:
    """Write the COBOL files and the files they copy under root, ready for cobc.

    Processing flow:
    1. Write every COBOL file
    2. Follow the COPY statements and EXEC SQL INCLUDEs of the COBOL files, and of each
       other file they lead to: write the symbolic map of a mapset, or the file found
       under the name cobc looks for

    Args:
        git_file_list: Every file git tracks or does not ignore.
        cobol_file_list: The COBOL files among them.
        root: The directory the copy is written under.

    Returns:
        (the directories, relative to the repository, holding a file that is copied
        from, {path of a copy written under the name a COPY statement uses: the file it
        is}, {upper-case name of a symbolic map written here: bms_name_set() of it}).
    """
    # == Step 1: the COBOL files ==============================================
    lookup = _file_lookup(git_file_list, cobol_file_list)
    copybook_set = frozenset(lookup.any_stem_dict)
    for path in cobol_file_list:
        _write_copy(path, os.path.join(root, path), copybook_set)
    dir_set = {os.path.dirname(path) for path in cobol_file_list}

    # == Step 2: the files they copy ==========================================
    cobol_file_set = set(cobol_file_list)
    alias_dict: dict[str, str] = {}
    map_name_dict: dict[str, set[str]] = {}
    mapset_dict = _read_mapset_dict(git_file_list)
    queue = sorted({item for path in cobol_file_list for item in _copy_name_list(path)})
    seen_set = set(queue)
    while queue:
        name, kind = queue.pop(0)
        if kind == _COPY_KIND and name.upper() not in lookup.cobol_stem_dict \
                and name.upper() in mapset_dict:
            bms_path, map_list = mapset_dict[name.upper()]
            _write_symbolic_map(name, bms_path, map_list, root, alias_dict)
            dir_set.add(os.path.dirname(bms_path))
            map_name_dict[name.upper()] = bms_name_set(map_list)
            continue
        real = _named_copybook(name, kind, lookup)
        if real is None:
            continue
        dir_set.add(os.path.dirname(real))
        copy_path = os.path.join(os.path.dirname(real), _copy_file_name(name, kind, real))
        if not os.path.exists(os.path.join(root, copy_path)):
            _write_copy(real, os.path.join(root, copy_path), copybook_set)
            if copy_path != real:
                alias_dict[copy_path] = real
        if real not in cobol_file_set:
            for item in _copy_name_list(real):
                if item not in seen_set:
                    seen_set.add(item)
                    queue.append(item)
    return sorted(directory or "." for directory in dir_set), alias_dict, map_name_dict


@dataclasses.dataclass
class _BmsField:
    """One named field of a map, or a group of fields (GRPNAME)."""

    name: str
    occurs: int = 0                   # OCCURS; 0 when the field is not repeated
    picin: str = ""                   # the picture of the input field
    picout: str = ""                  # the picture of the output field
    member_list: list = dataclasses.field(default_factory=list)  # a group's fields


@dataclasses.dataclass
class _BmsMap:
    """One map (DFHMDI) of a mapset."""

    name: str
    prefix: bool                      # TIOAPFX=YES
    attribute_list: list[str]         # the letters of the extended attributes
    field_list: list[_BmsField]


def _bms_statement_list(text: str) -> list[tuple[str, str, str]]:
    """Read the assembler statements of a BMS source.

    A line with "*" in column 1 is a comment. A statement is continued while column 72
    holds a character, on the next line from column 16. The operands end at the first
    blank outside quotes; what follows is a remark.

    Returns:
        (label, operation, operands) per statement, the label "" where there is none.
    """
    statement_list = []
    label = operation = operand_text = ""
    is_continued = in_quote = is_done = False
    for line in text.expandtabs(8).split("\n"):
        line = line.rstrip("\r")
        if not is_continued:
            if not line.strip() or line.startswith("*"):
                continue
            head = re.match(r"(\S*)\s+(\S+)\s*", line[:_BMS_CONTINUE_COLUMN])
            if head is None:
                continue
            label, operation = head.group(1), head.group(2).upper()
            body = line[head.end():_BMS_CONTINUE_COLUMN]
            operand_text, in_quote, is_done = "", False, False
        else:
            body = line[_BMS_CONTINUE_START:_BMS_CONTINUE_COLUMN]
            if not in_quote:
                body = body.lstrip()
        for char in body:
            if is_done:
                break
            if char == "'":
                in_quote = not in_quote
            elif char == " " and not in_quote:
                is_done = True
                break
            operand_text += char
        if in_quote:
            is_done = False
        is_continued = len(line) > _BMS_CONTINUE_COLUMN and line[_BMS_CONTINUE_COLUMN] != " "
        if is_continued:
            is_done = False
        else:
            statement_list.append((label, operation, operand_text))
    return statement_list


def _bms_operand_dict(operand_text: str) -> dict[str, str]:
    """Split "KEY=value,..." at the commas outside quotes and parentheses.

    Returns:
        {upper-case key: value as written}, the value of a key with no "=" being "".
    """
    part_list, part, depth, in_quote = [], "", 0, False
    for char in operand_text:
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
        elif not in_quote and depth == 0 and char == ",":
            part_list.append(part)
            part = ""
            continue
        part += char
    part_list.append(part)
    operand_dict = {}
    for part in part_list:
        key, _, value = part.partition("=")
        if key.strip():
            operand_dict[key.strip().upper()] = value.strip()
    return operand_dict


def _bms_value_list(value: str) -> list[str]:
    """Return the upper-case items of a value such as "(COLOR,HILIGHT)" or "YES"."""
    return [item.strip().upper() for item in value.strip("()").split(",") if item.strip()]


def _bms_picture(value: str) -> str:
    """Return the picture a PICIN or PICOUT value gives, without its quotes."""
    return value[1:-1].replace("''", "'") if value.startswith("'") else value


def read_bms(text: str) -> dict[str, list[_BmsMap]]:
    """Read the mapsets of a BMS source.

    Args:
        text: The BMS source.

    Returns:
        {upper-case mapset name: its maps}. A mapset with a map or a field whose name is
        not a COBOL word is left out.
    """
    mapset_dict: dict[str, list[_BmsMap]] = {}
    mapset_name, set_option_dict, map_list = "", {}, []
    bad_set = set()
    group_dict: dict[str, _BmsField] = {}
    for label, operation, operand_text in _bms_statement_list(text):
        operand_dict = _bms_operand_dict(operand_text)
        if operation == "DFHMSD":
            if operand_dict.get("TYPE", "").upper() == "FINAL":
                if mapset_name and mapset_name not in bad_set:
                    mapset_dict[mapset_name] = map_list
                mapset_name, map_list = "", []
                continue
            mapset_name, set_option_dict, map_list = label.upper(), operand_dict, []
        elif operation == "DFHMDI" and mapset_name:
            option_dict = {**set_option_dict, **operand_dict}
            if "DSATTS" in option_dict:
                attribute_set = set(_bms_value_list(option_dict["DSATTS"]))
            elif option_dict.get("EXTATT", "NO").upper() == "YES":
                attribute_set = set(_BMS_EXTATT_TUPLE)
            else:
                attribute_set = set()
            if not _BMS_NAME_RE.fullmatch(label):
                bad_set.add(mapset_name)
            map_list.append(_BmsMap(label.upper(), option_dict.get("TIOAPFX", "NO").upper() == "YES",
                                    [letter for attribute, letter in _BMS_ATTRIBUTE_TUPLE
                                     if attribute in attribute_set], []))
            group_dict = {}
        elif operation == "DFHMDF" and map_list and label:
            if not _BMS_NAME_RE.fullmatch(label):
                bad_set.add(mapset_name)
                continue
            length = operand_dict.get("LENGTH", "1")
            field = _BmsField(label.upper(),
                              int(operand_dict["OCCURS"]) if operand_dict.get("OCCURS", "").isdigit() else 0,
                              _bms_picture(operand_dict.get("PICIN", "")) or f"X({length})",
                              _bms_picture(operand_dict.get("PICOUT", "")) or f"X({length})")
            group = operand_dict.get("GRPNAME", "").upper()
            if group:
                if group not in group_dict:
                    group_dict[group] = _BmsField(group)
                    map_list[-1].field_list.append(group_dict[group])
                group_dict[group].member_list.append(field)
            else:
                map_list[-1].field_list.append(field)
    if mapset_name and mapset_name not in bad_set:
        mapset_dict[mapset_name] = map_list
    return mapset_dict


def _symbolic_line_list(field: _BmsField, attribute_list: list[str], level: int,
                        is_input: bool) -> list[str]:
    """Return the lines one field adds to the input or the output record of its map."""
    pad = " " * (11 + (level - 2) * 2)
    this, under = f"{level:02d}", f"{level + 1:02d}"
    name = field.name
    if is_input:
        line_list = [f"{pad}{this}  {name}L COMP PIC S9(4).", f"{pad}{this}  {name}F PIC X.",
                     f"{pad}{this}  FILLER REDEFINES {name}F.",
                     f"{pad}  {under}  {name}A PIC X."]
        if attribute_list:
            line_list.append(f"{pad}{this}  FILLER PIC X({len(attribute_list)}).")
    else:
        line_list = [f"{pad}{this}  FILLER PIC X(3)."]
        line_list += [f"{pad}{this}  {name}{letter} PIC X." for letter in attribute_list]
    suffix = "I" if is_input else "O"
    if not field.member_list:
        return line_list + [f"{pad}{this}  {name}{suffix}",
                            f"{pad}    PIC {field.picin if is_input else field.picout}."]
    line_list.append(f"{pad}{this}  {name}{suffix}.")
    for member in field.member_list:
        line_list += [f"{pad}  {under}  {member.name}{suffix}",
                      f"{pad}      PIC {member.picin if is_input else member.picout}."]
    return line_list


def symbolic_map_text(map_list: list[_BmsMap]) -> str:
    """Write the symbolic map of a mapset as a COBOL copybook in fixed format.

    Each map is an input record <map>I and an output record <map>O redefining it. A
    field is <field>L, <field>F, <field>A and <field>I in the input record and the
    letters of its extended attributes and <field>O in the output one; a field with
    OCCURS is a group of those repeated; a group (GRPNAME) holds the I or O field of
    each of its fields.

    Args:
        map_list: The maps of the mapset, as read_bms() returns them.

    Returns:
        The copybook's text.
    """
    line_list = []
    for bms_map in map_list:
        for is_input in (True, False):
            record = f"{bms_map.name}I" if is_input else f"{bms_map.name}O REDEFINES {bms_map.name}I"
            line_list.append(f"       01  {record}.")
            body_list = []
            if bms_map.prefix:
                body_list.append(f"           02  FILLER PIC X({_BMS_PREFIX_LENGTH}).")
            for field in bms_map.field_list:
                if field.occurs:
                    body_list.append(f"           02  FILLER OCCURS {field.occurs} TIMES.")
                    body_list += _symbolic_line_list(field, bms_map.attribute_list, 3, is_input)
                else:
                    body_list += _symbolic_line_list(field, bms_map.attribute_list, 2, is_input)
            line_list += body_list or ["           02  FILLER PIC X."]
    return "\n".join(line_list) + "\n"


def bms_name_set(map_list: list[_BmsMap]) -> set[str]:
    """Return the upper-case names of a mapset's maps, fields and groups."""
    name_set = set()
    for bms_map in map_list:
        name_set.add(bms_map.name)
        for field in bms_map.field_list:
            name_set.add(field.name)
            name_set.update(member.name for member in field.member_list)
    return name_set


def _read_mapset_dict(git_file_list: list[str]) -> dict[str, tuple[str, list[_BmsMap]]]:
    """Read the mapsets of every BMS source of the repository.

    Returns:
        {upper-case mapset name: (the BMS source's path, its maps)}. A source that
        cannot be read or read as BMS gives nothing.
    """
    mapset_dict: dict[str, tuple[str, list[_BmsMap]]] = {}
    for path in git_file_list:
        if os.path.splitext(path)[1].lower() != _BMS_EXT:
            continue
        try:
            with open(path, encoding="latin-1") as fp:
                text = fp.read()
            for name, map_list in read_bms(text).items():
                mapset_dict.setdefault(name, (path, map_list))
        except (OSError, ValueError) as e:
            print(f"ctloom-check-cbl: {path} is not read as BMS: {e}", file=sys.stderr)
    return mapset_dict


def _is_program(path: str) -> bool:
    """Say whether a COBOL file holds PROGRAM-ID outside its comment lines."""
    if os.path.splitext(path)[1].lower() not in _PROGRAM_EXT_SET:
        return False
    with open(path, encoding="latin-1") as fp:
        line_list = fp.read().split("\n")
    is_free = _is_free(line_list)
    for line in line_list:
        start, end = _code_span(line, is_free)
        if _PROGRAM_ID_RE.search(line, start, end):
            return True
    return False


def _is_product_error(message: str) -> bool:
    """Say whether an error message is about a copybook or a name a product supplies."""
    for pattern in (_NO_COPYBOOK_RE, _NOT_DEFINED_RE):
        match = pattern.match(message)
        if match and match.group("name").upper().startswith(_PRODUCT_PREFIX_TUPLE):
            return True
    return False


def _check(path: str, root: str, option_tuple: tuple[list[str], list[str]],
           dir_list: list[str], alias_dict: dict[str, str],
           map_name_set: set[str]) -> list[str]:
    """Run cobc over one program of the copy under root and return its error lines.

    Args:
        path: The program, relative to the repository.
        root: The copy of the repository.
        option_tuple: _option_tuple()'s return value: the options come before the
            copybook directories of the repository, the -I arguments after them.
        dir_list: The copybook directories, relative to the repository.
        alias_dict: _mirror()'s copies written under another name; a line about one
            names the file it is.
        map_name_set: The names of the symbolic maps made here that the program copies.
            A name that is not defined is not reported when it is one of these followed
            by one character.

    Returns:
        The error lines.
    """
    argv = ["cobc", "-fsyntax-only"] + option_tuple[0]
    for directory in dir_list:
        argv += ["-I", directory]
    argv += option_tuple[1]
    try:
        proc = subprocess.run(argv + [path], cwd=root, capture_output=True,
                              timeout=_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return [f"{path}:1: error: cobc ran over {_TIMEOUT_SEC} seconds"]
    out = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    match_list = [match for match in map(_ERROR_RE.match, out.splitlines())
                  if match and not _is_product_error(match.group("message"))]
    lack_copybook = any(_NO_COPYBOOK_RE.match(match.group("message")) for match in match_list)
    line_list = []
    for match in match_list:
        name_match = _NOT_DEFINED_RE.match(match.group("message"))
        if name_match and (lack_copybook or name_match.group("name")[:-1].upper() in map_name_set):
            continue
        line_list.append(f"{alias_dict.get(match.group('path'), match.group('path'))}:"
                         f"{match.group('line')}: error: {match.group('message')}")
    return line_list


def _option_tuple(arg_list: list[str], repo: str) -> tuple[list[str], list[str]]:
    """Split the arguments into the options for cobc and the copybook directories.

    Examples:
        (["-std=ibm", "-I", "lib", "-I/opt/copy"], "/r")
            -> (["-std=ibm"], ["-I", "/r/lib", "-I", "/opt/copy"])

    Returns:
        (the other arguments, the -I arguments with each directory made absolute under
        repo).
    """
    option_list, include_list, k = [], [], 0
    while k < len(arg_list):
        arg = arg_list[k]
        if arg == "-I" and k + 1 < len(arg_list):
            include_list += ["-I", os.path.join(repo, arg_list[k + 1])]
            k += 2
            continue
        if arg.startswith("-I") and len(arg) > 2:
            include_list += ["-I", os.path.join(repo, arg[2:])]
        else:
            option_list.append(arg)
        k += 1
    return option_list, include_list


def _try_option(option_list: list[str]) -> str:
    """Check a small program with the arguments.

    Returns:
        "" when cobc takes them, else what cobc said.
    """
    with tempfile.TemporaryDirectory(prefix="ctloom-check-cbl-try-") as directory:
        with open(os.path.join(directory, "CTLOOMCK.cbl"), "w", encoding="ascii") as fp:
            fp.write(_TRY_PROGRAM)
        try:
            proc = subprocess.run(["cobc", "-fsyntax-only"] + option_list + ["CTLOOMCK.cbl"],
                                  cwd=directory, capture_output=True, timeout=_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            return f"cobc ran over {_TIMEOUT_SEC} seconds"
    if proc.returncode == 0:
        return ""
    return (proc.stdout + proc.stderr).decode("utf-8", "replace").strip() or \
        f"cobc ended with {proc.returncode}"


def _setting_arg_list() -> list[str]:
    """Return the cobc arguments COBOL_DIALECT and COBOL_COPYBOOK_DIR stand for.

    Examples:
        COBOL_DIALECT=ibm, COBOL_COPYBOOK_DIR="copylib, lib2"
            -> ["-std=ibm", "-I", "copylib", "-I", "lib2"]
    """
    arg_list = []
    dialect = os.environ.get("COBOL_DIALECT", "").strip()
    if dialect:
        arg_list.append(f"-std={dialect}")
    for directory in os.environ.get("COBOL_COPYBOOK_DIR", "").split(","):
        if directory.strip():
            arg_list += ["-I", directory.strip()]
    return arg_list


def main(arg_list: list[str]) -> int:
    """Check the repository in the current directory and print the error lines.

    Args:
        arg_list: The arguments, passed to cobc after those the settings stand for.
    """
    if shutil.which("cobc") is None:
        print("ctloom-check-cbl: cobc does not exist", file=sys.stderr)
        return _CANNOT_RUN
    arg_list = _setting_arg_list() + arg_list
    option_tuple = _option_tuple(arg_list, os.getcwd())
    refusal = _try_option(option_tuple[0] + option_tuple[1])
    if refusal:
        print(f"ctloom-check-cbl: cobc does not take {' '.join(arg_list)} (COBOL_DIALECT, "
              f"COBOL_COPYBOOK_DIR and the arguments): {refusal}", file=sys.stderr)
        return _CANNOT_RUN
    git_file_list = _git_file_list()
    cobol_file_list = [path for path in git_file_list
                       if os.path.splitext(path)[1].lower() in _COBOL_EXT_SET]
    program_list = [path for path in cobol_file_list if _is_program(path)]
    if not program_list:
        return 0
    with tempfile.TemporaryDirectory(prefix="ctloom-check-cbl-") as root:
        dir_list, alias_dict, map_name_dict = _mirror(git_file_list, cobol_file_list, root)

        def check_one(path: str) -> list[str]:
            """Check one program with the names of the symbolic maps it copies."""
            map_name_set = set()
            for name, kind in _copy_name_list(path):
                if kind == _COPY_KIND:
                    map_name_set |= map_name_dict.get(name.upper(), set())
            return _check(path, root, option_tuple, dir_list, alias_dict, map_name_set)

        with concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 1) as pool:
            for line_list in pool.map(check_one, program_list):
                for line in line_list:
                    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
