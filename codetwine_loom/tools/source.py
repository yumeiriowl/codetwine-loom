"""The reading tools.

This module keeps the project name and the usage table set_project_data() takes out of
the codetwine analysis. graph_search also queries the index, through search.scope.

The index's line numbers are those of the state the run started from. A definition in a
file this run has written is looked for in the file as it stands now (fresh_line_range).
"""

import logging
import os
import pathlib
import re
from collections import Counter
from collections.abc import Iterator

from codetwine.utils.file_utils import copy_path_to_rel, is_text_file, read_source_text

from codetwine_loom import cobol, config, tmpfile
from codetwine_loom.index.indexer import tokenize
from codetwine_loom.knowledge import Knowledge
from codetwine_loom.languages import (LANG_DICT, lang_of, name_only_dir_set,
                                      name_without_suffix, split_def_key)
from codetwine_loom.paths import (has_name_of, host_path_of, knowledge_to_repo, path_text,
                                  repo_to_knowledge, resolve_in_project, to_repo_relative)
from codetwine_loom.search import scope as scope_mod
from codetwine_loom.search.query import has_english_word
from codetwine_loom.textcut import (cut, line_list_of, no_lone_surrogate, one_line,
                                    text_error, whole_number_error)
from codetwine_loom.tools import edit as edit_mod

logger = logging.getLogger(__name__)

project_name: str | None = None    # the directory name of PROJECT_DIR

# What a read leaves free inside one turn's output: room for the header line, the line
# saying where to continue from, and the text around them
_READ_TURN_ROOM_CHAR = 400

# Where each file is referred to from, taken out of the analysis by set_project_data().
# {knowledge path of the file being used: [(knowledge path of the referring file, the
# name being used, its lines), ...]}. Nothing else of the analysis is kept
_file_use_dict: dict[str, list[tuple[str, str, list[int]]]] = {}

# How many search_again calls were not refused since reset_search_count()
_search_again_count = 0

# How to read the range and the total line count out of the header _slice_line_range
# writes, applied to the first line only (with match)
SLICE_HEADER = re.compile(r"# .*: lines (\d{1,15})-(\d{1,15}) of (\d{1,15})\n")


def set_project_data(knowledge: Knowledge) -> None:
    """Take what get_use_of_file needs out of the codetwine analysis, drop the rest.

    Each call replaces what the previous one kept. Only callee_usages is kept; the
    definitions' code and design documents are dropped.

    The referring file's path is stored as the knowledge path it comes in as: this may
    run before project_name is assigned.

    Args:
        knowledge: The analysis to read the files from.
    """
    global _file_use_dict
    use_dict: dict[str, list[tuple[str, str, list[int]]]] = {}
    for file_entry in knowledge.iter_file():
        for usage in file_entry["file_dependencies"]["callee_usages"]:
            use_dict.setdefault(usage["from"], []).append(
                (file_entry["file"], usage["name"], usage["lines"]))
    _file_use_dict = use_dict


def read_limit() -> int:
    """The cap on what one read returns, kept under the turn's own output limit.

    Returns:
        config.READ_MAX_CHAR, lowered to leave room for the header and the
        continuation line inside config.RLM_MAX_OUTPUT_CHAR. 0 when neither has a cap.
    """
    if config.RLM_MAX_OUTPUT_CHAR <= 0:
        return config.READ_MAX_CHAR
    room = max(1, config.RLM_MAX_OUTPUT_CHAR - _READ_TURN_ROOM_CHAR)
    if config.READ_MAX_CHAR <= 0:
        return room
    return min(config.READ_MAX_CHAR, room)


def read_source_file(path: str, start_line: int = 1,
                     end_line: int | None = None) -> str:
    """Return a file of the target repository, cut before READ_MAX_CHAR when long.

    It reads the real file under PROJECT_DIR, not the copy in the output directory. A
    path that is not a regular file of at most tmpfile.READ_MAX_BYTE bytes is not read
    (tmpfile.check_file).

    The amount returned is capped by read_limit(), which is config.READ_MAX_CHAR kept
    under the turn's own output limit. A file over the cap comes back up to the last
    line that fits, with a note on how to continue. With both caps off, a call without
    the optional arguments returns the whole file byte for byte.

    Args:
        path: A repository-relative path, e.g. "myapp/config/settings.py".
        start_line: The line to start at (1-based, inclusive).
        end_line: The line to stop at (inclusive). None reads to the end of the file.

    Returns:
        When the whole file fits and no line arguments were given: the file contents
        themselves. Otherwise: that range prefixed with a "# <path>: lines A-B of N"
        line, plus a trailing line on how to continue when it was cut at the cap.
        A string starting with "Error: " when the file could not be read or the line
        arguments are invalid. It never raises.
    """
    return read_source_result(path, start_line, end_line)[0]


def read_source_result(path: str, start_line: int = 1,
                       end_line: int | None = None) -> tuple[str, bool]:
    """Read a file of the target repository as read_source_file() reads it.

    Args:
        path: As read_source_file() takes it.
        start_line: As read_source_file() takes it.
        end_line: As read_source_file() takes it.

    Returns:
        (read_source_file()'s return value, True when the file was read and that is
        the text or the range asked for, False when it says why not).
    """
    try:
        path = path_text(path)
        file_abs = resolve_in_project(path)
    except ValueError as e:
        return f"Error: {e}", False
    try:
        tmpfile.check_file(file_abs)
        text = read_source_text(file_abs)
    except OSError as e:
        # A path with the shape of a knowledge path gets its own message
        try:
            repo_path = knowledge_to_repo(path, project_name)
        except ValueError:
            return f"Error: cannot read {path}: {e}", False
        return (f"Error: {path} is an index path. Tools take repository-relative "
                f"paths; in this case {repo_path}"), False
    except ValueError as e:
        return f"Error: cannot read {path}: {e}", False

    return text_range(text, path, start_line, end_line)


def text_range(text: str, path: str, start_line: int = 1,
               end_line: int | None = None) -> tuple[str, bool]:
    """Return a file's text as read_source_file() returns it.

    Args:
        text: The whole file, decoded.
        path: The repository-relative path the header line names.
        start_line: The line to start at (1-based, inclusive).
        end_line: The line to stop at (inclusive). None reads to the end of the file.

    Returns:
        (the text itself when it fits in read_limit() and no line arguments were
        given, the range with its header line otherwise, or a string starting with
        "Error: " for invalid line arguments; True for the text or the range, False
        for that string).
    """
    argument_error = whole_number_error("start_line", start_line) or (
        whole_number_error("end_line", end_line) if end_line is not None else "")
    if argument_error:
        return f"Error: {argument_error}", False
    # Called with the defaults and fitting in the cap: the text is returned unsplit, a
    # missing trailing newline included
    if start_line == 1 and end_line is None \
            and not (0 < read_limit() < len(text)):
        return text, True
    return _slice_line_range(text, path, start_line, end_line)


def _fit_line_list(line_list: list[str], limit: int) -> list[str]:
    """Return the leading lines that fit in limit characters, never cutting mid-line.

    The first line is always kept, even when it alone is over the limit.

    Args:
        line_list: The lines, newlines included.
        limit: The maximum number of characters. 0 or less means no cut.

    Returns:
        As many leading lines as fit.
    """
    if limit <= 0:
        return line_list
    keep_list, use_count = [], 0
    for line in line_list:
        if keep_list and use_count + len(line) > limit:
            break
        keep_list.append(line)
        use_count += len(line)
    return keep_list


def _range_head(path: str, start_line: int, last_line: int, line_count: int) -> str:
    """Return the start of the header line SLICE_HEADER reads, without its line end.

    Args:
        path: A repository-relative path.
        start_line: The first line of the range.
        last_line: The last line of the range.
        line_count: The total number of lines in the file.

    Returns:
        "# <path>: lines A-B of N".
    """
    return f"# {path}: lines {start_line}-{last_line} of {line_count}"


def _continue_hint(path: str, start_line: int, last_show_line: int,
                   line_count: int) -> str:
    """The trailing line explaining how to continue after a cut.

    Args:
        path: A repository-relative path.
        start_line: The first line of the range returned.
        last_show_line: The last line of the range returned.
        line_count: The total number of lines in the file.

    Returns:
        A single "# <path>: lines A-B of N returned. Continue with ..." line.
    """
    return (_range_head(path, start_line, last_show_line, line_count)
            + f' returned. Continue with read_source_file("{path}", {last_show_line + 1})\n')


def _slice_line_range(text: str, path: str, start_line: int,
                      end_line: int | None) -> tuple[str, bool]:
    """Slice out read_source_file's line range and prefix the header line.

    Anything beyond read_limit() is dropped at a line boundary. The header's end line
    reflects what was actually returned, and the note on how to continue is appended
    only when something was dropped.

    The cap applies to the **whole return value**: the header and the note come out of
    the body's budget. The first line of the range is returned even when it does not fit
    in what the header and the note leave of the cap.

    Args:
        text: The whole file.
        path: The repository-relative path to write in the header.
        start_line: The line to start at (1-based).
        end_line: The line to stop at. None reads to the end.

    Returns:
        ("# <path>: lines A-B of N" followed by that range, with a "# <path>: lines
        A-B of N returned. Continue with ..." line appended when it was cut at the
        cap, and True; or a string starting with "Error: " when the arguments are
        invalid, and False).
    """
    line_list = line_list_of(text, keepends=True)
    line_count = len(line_list)
    if start_line < 1:
        return f"Error: start_line must be 1 or more: {start_line}", False
    if start_line > line_count:
        return f"Error: {path} has only {line_count} lines. start_line={start_line}", False
    last_line = line_count if end_line is None else min(end_line, line_count)
    if last_line < start_line:
        return (f"Error: end_line must be start_line or more: "
                f"start_line={start_line} end_line={end_line}"), False

    # The header and the note are sized with last_line, the longest they can get
    body_line_list = line_list[start_line - 1:last_line]
    head_line = _range_head(path, start_line, last_line, line_count) + "\n"
    head_len = len(head_line)
    limit = read_limit()
    keep_list = _fit_line_list(body_line_list, max(1, limit - head_len) if limit > 0 else 0)
    if len(keep_list) == len(body_line_list):
        # Everything fit: no continuation note is added
        return head_line + "".join(keep_list), True

    # Cutting. Take the note's line out of the budget too, then refill
    hint_len = len(_continue_hint(path, start_line, last_line, line_count))
    keep_list = _fit_line_list(body_line_list, max(1, limit - head_len - hint_len))
    last_show_line = start_line + len(keep_list) - 1
    # The body reaching this point always ends with a newline: only the file's last line
    # can lack one, and that case returned above
    return (_range_head(path, start_line, last_show_line, line_count) + "\n"
            + "".join(keep_list)
            + _continue_hint(path, start_line, last_show_line, line_count)), True


def is_codetwine_copy(part_tuple: tuple[str, ...]) -> bool:
    """Whether this is a copy codetwine wrote, judged by its last two components.

    The shape is the one codetwine's copy_path_to_rel() recognises
    (<stem>_<ext>/<basename>, or <name>/<name> for a file without an extension). A copy
    that was committed to the repository has the shape too.

    Args:
        part_tuple: pathlib.Path.parts.

    Returns:
        True when it has the shape of a copy.
    """
    if len(part_tuple) < 2:
        return False
    tail = "/".join(part_tuple[-2:])
    return copy_path_to_rel(tail) != tail


def skip_dir_set() -> frozenset[str]:
    """The directory names of GREP_SKIP_DIR and of the primary language's build output.

    Read per call: a setting rewritten at run time is followed.

    Returns:
        The names of config.GREP_SKIP_DIR, with blanks and surrounding spaces removed,
        and the directories the primary language's build_output_tuple gives by name
        alone ("obj/" -> "obj"; "/dist/" and "uv.lock" are left out), each read as an
        fnmatch pattern (paths.has_name_of).
    """
    lang_config = LANG_DICT.get(config.PRIMARY_EXT)
    output_tuple = lang_config.build_output_tuple if lang_config is not None else ()
    return frozenset(
        name.strip() for name in config.GREP_SKIP_DIR.split(",") if name.strip()
    ) | name_only_dir_set(output_tuple)


def _walk_glob(glob: str) -> str:
    """Return the pattern pathlib walks for one glob of a call.

    A glob with no "/" in it, a "/" at its end aside, is tried in every directory. "."
    is the repository root.

    Examples:
        "*.py"     -> "**/*.py"
        "src/"     -> "**/src/"
        "./*.py"   -> "./*.py"
        "src/**"   -> "src/**"
        "."        -> "**"

    Args:
        glob: A glob relative to PROJECT_DIR.

    Returns:
        The pattern to hand to pathlib.Path.glob().
    """
    name = glob.rstrip("/") or glob
    if name == ".":
        return "**"
    if "/" not in name and name not in ("**", ".."):
        return f"**/{glob}"
    return glob


def _file_under(directory: pathlib.Path, skip_set: frozenset[str],
                walk_set: set[pathlib.Path]) -> list[pathlib.Path]:
    """Return every file under a directory, and record the directories walked.

    A directory skip_set names, one walk_set already holds and a link to a directory
    are not walked into.

    Args:
        directory: The directory to walk.
        skip_set: The directory names not walked into (skip_dir_set).
        walk_set: The directories walked so far. Each one walked here is added.

    Returns:
        The files, sorted as paths.
    """
    file_list = []
    for dir_path, dir_name_list, file_name_list in os.walk(directory):
        walk_set.add(pathlib.Path(dir_path))
        dir_name_list[:] = [name for name in dir_name_list
                            if not has_name_of([name], skip_set)
                            and pathlib.Path(dir_path, name) not in walk_set]
        file_list += [pathlib.Path(dir_path, name) for name in file_name_list]
    return sorted(file_list)


# What a glob writes in place of any number of directories
_ANY_DEPTH_PART = "**"


def _link_spell(glob: str) -> tuple[tuple[str | None, ...], frozenset[str]]:
    """Return where a glob writes the name of a directory it walks into.

    Examples:
        "link/**"      -> (("link",), {})
        "a/link/*.py"  -> (("a", "link"), {})
        "src/*/x.py"   -> (("src", None), {})
        "**/link/**"   -> ((), {"link"})
        "link"         -> ((), {})

    Args:
        glob: A glob relative to PROJECT_DIR.

    Returns:
        (the parts of the glob in front of its first "**", each as the name it writes
        or None for one written with "*", "?" or "[", the names the parts behind that
        "**" write). The last part of the glob is in neither, and "" and "." are no
        parts.
    """
    part_list = [part for part in glob.rstrip("/").split("/") if part not in ("", ".")][:-1]
    name_list = [None if any(mark in part for mark in "*?[") else part
                 for part in part_list]
    if _ANY_DEPTH_PART not in part_list:
        return tuple(name_list), frozenset()
    lead_count = part_list.index(_ANY_DEPTH_PART)
    return tuple(name_list[:lead_count]), \
        frozenset(name for name in name_list[lead_count + 1:] if name)


def _is_open_dir(directory: pathlib.Path, root: pathlib.Path,
                 spell_tuple: tuple[tuple[str | None, ...], frozenset[str]],
                 open_dict: dict[pathlib.Path, bool]) -> bool:
    """Say whether a glob walks into a directory below the repository root.

    Args:
        directory: The directory, as the glob reached it below root.
        root: PROJECT_DIR, resolved.
        spell_tuple: _link_spell() of the glob.
        open_dict: The answers given so far for the glob, by directory.

    Returns:
        False when the directory, or one between it and root, is a link whose name
        the glob does not write where the link stands: as the part of the glob at the
        link's depth in front of its first "**", or as any part behind that "**".
    """
    if directory == root or directory.parent == directory:
        return True
    if directory not in open_dict:
        lead_tuple, rest_set = spell_tuple
        depth = len(directory.relative_to(root).parts) - 1
        has_name = lead_tuple[depth] == directory.name if depth < len(lead_tuple) \
            else directory.name in rest_set
        open_dict[directory] = \
            _is_open_dir(directory.parent, root, spell_tuple, open_dict) \
            and (has_name or not directory.is_symlink())
    return open_dict[directory]


def _iter_file(root: pathlib.Path, glob_list: list[str]) -> Iterator[pathlib.Path]:
    """Yield the files the globs name, each once.

    A directory a glob matches stands for every file under it (_file_under), unless it
    is under a directory skip_dir_set() names or resolve_in_project() refuses it. A
    link to a directory is walked into only where the glob writes its name at the
    place the link stands, with more of the glob behind it (_link_spell): what a glob
    matches in or below any other link to a directory, and such a link itself, is
    passed over.

    Args:
        root: PROJECT_DIR, resolved.
        glob_list: Patterns relative to PROJECT_DIR, as pathlib.Path.glob() reads them.

    Yields:
        The path of each regular file, below root as the glob reached it.

    Raises:
        NotImplementedError, ValueError: When pathlib cannot read a glob (an absolute
            pattern, "**" inside a component).
    """
    skip_set = skip_dir_set()
    file_set: set[pathlib.Path] = set()
    walk_set: set[pathlib.Path] = set()
    for glob in glob_list:
        spell_tuple = _link_spell(glob)
        open_dict: dict[pathlib.Path, bool] = {}
        for match in sorted(root.glob(glob)):
            under_list = [match]
            if match.is_dir():
                if not _is_open_dir(match, root, spell_tuple, open_dict) \
                        or match in walk_set \
                        or has_name_of(match.relative_to(root).parts, skip_set):
                    continue
                try:
                    resolve_in_project(str(match))
                except ValueError:
                    continue
                under_list = _file_under(match, skip_set, walk_set)
            elif not _is_open_dir(match.parent, root, spell_tuple, open_dict):
                continue
            for file_path in under_list:
                if file_path.is_file() and file_path not in file_set:
                    file_set.add(file_path)
                    yield file_path


def _iter_match(rx: re.Pattern, glob_list: list[str], unread_list: list[str],
                read_list: list[str]) -> Iterator[tuple[str, int, list[str]]]:
    """Walk the files the globs name and yield every line the regex matched.

    Files that are not text, files under a directory skip_dir_set() names, what the
    other languages' tools write (edit.in_other_build_output) and source copies
    codetwine wrote are skipped, and so is a file resolve_in_project() refuses: one a
    glob or a symbolic link reaches outside PROJECT_DIR, and one inside the git
    directory. In a file of a language that ignores case, the regex matches in either
    case.

    Args:
        rx: A compiled regular expression.
        glob_list: Patterns relative to PROJECT_DIR, read as _iter_file() reads them.
        unread_list: Where the repository-relative path of each file that could not be
            read is added, and of each that is larger than tmpfile.READ_MAX_BYTE
            (tmpfile.check_file).
        read_list: Where the repository-relative path of each file whose lines were
            read is added.

    Yields:
        (repository-relative path, line number (1-based), all lines of that file).

    Raises:
        NotImplementedError, ValueError: When pathlib cannot read a glob (an absolute
            pattern, "**" inside a component).
    """
    root = pathlib.Path(resolve_in_project("."))
    # The files under an OUTPUT_DIR inside PROJECT_DIR are dropped
    out_dir = pathlib.Path(os.path.realpath(config.OUTPUT_DIR))
    skip_set = skip_dir_set()
    place_list = edit_mod.other_output_place_list()
    rx_fold = re.compile(rx.pattern, rx.flags | re.IGNORECASE)
    for file_path in _iter_file(root, glob_list):
        if file_path.is_relative_to(out_dir):
            continue
        # Drop the files under a directory skip_set names and the source copies
        # (is_codetwine_copy), both judged by the components below PROJECT_DIR
        part_tuple = file_path.relative_to(root).parts
        if has_name_of(part_tuple[:-1], skip_set) or is_codetwine_copy(part_tuple):
            continue
        try:
            resolve_in_project(str(file_path))
        except ValueError:
            continue
        path_rel = str(file_path.relative_to(root))
        # What another language's tools write is dropped where they write it
        if edit_mod.in_other_build_output(file_path.relative_to(root).as_posix(), place_list):
            continue
        if not os.access(file_path, os.R_OK):
            unread_list.append(path_rel)
            continue
        try:
            tmpfile.check_file(str(file_path))
        except (OSError, ValueError):
            unread_list.append(path_rel)
            continue
        if not is_text_file(str(file_path)):
            continue
        try:
            line_list = line_list_of(read_source_text(str(file_path)))
        except OSError:
            unread_list.append(path_rel)
            continue
        read_list.append(path_rel)
        lang_config = lang_of(path_rel)
        file_rx = rx_fold if lang_config is not None and lang_config.ignore_case else rx
        for line_no, line in enumerate(line_list, 1):
            if file_rx.search(line):
                yield path_rel, line_no, line_list


def _hit_entry(path_rel: str, line_no: int, line_list: list[str],
               with_context: bool) -> dict:
    """Shape one hit into the returned form.

    Args:
        path_rel: A repository-relative path.
        line_no: The matching line number (1-based).
        line_list: All lines of that file (without newlines).
        with_context: True adds GREP_CONTEXT_LINE lines before and after.

    Returns:
        {"path", "line", "text"}, plus "before" and "after" with context.
    """
    lang_config = lang_of(path_rel)
    keep_space = lang_config is not None and lang_config.column_layout
    entry_dict = {"path": path_rel, "line": line_no,
                  "text": _grep_line(line_list[line_no - 1], keep_space)}
    context_count = config.GREP_CONTEXT_LINE
    if with_context and context_count > 0:
        entry_dict["before"] = [
            _grep_line(line, keep_space)
            for line in line_list[max(0, line_no - 1 - context_count):line_no - 1]]
        entry_dict["after"] = [_grep_line(line, keep_space)
                               for line in line_list[line_no:line_no + context_count]]
    return entry_dict


def _grep_line(line: str, keep_space: bool = False) -> str:
    """Cut one line to GREP_LINE_MAX_CHAR, the marker on the line.

    Args:
        line: The line as it stands in the file, without its newline.
        keep_space: True keeps the spaces of the line as they are, for a language whose
            columns carry meaning. False folds every run of whitespace into one space.

    Returns:
        The line when it fits, otherwise its head with the "... (N chars omitted)"
        marker on the same line.
    """
    text = cut(line, config.GREP_LINE_MAX_CHAR)
    return text.replace("\n", " ") if keep_space else one_line(text)


def _entry_char_count(entry_dict: dict) -> int:
    """Estimate how many characters one hit takes when it is printed.

    Args:
        entry_dict: A dict built by _hit_entry().

    Returns:
        The characters of the text and the context lines plus the fixed cost of the
        key names.
    """
    body = len(entry_dict["text"]) + sum(len(line) for line in entry_dict.get("before", []))
    body += sum(len(line) for line in entry_dict.get("after", []))
    return body + config.GREP_HIT_OVERHEAD_CHAR


def hit_limit() -> int:
    """The cap on the number of hits, from GREP_MAX_HIT and the per-turn output limit.

    Returns:
        The cap on the number of hits, never below 1: one hit always fits. GREP_MAX_HIT
        alone when RLM_MAX_OUTPUT_CHAR is 0.
    """
    if config.RLM_MAX_OUTPUT_CHAR <= 0:
        return max(1, config.GREP_MAX_HIT)
    per_hit = config.GREP_HIT_OVERHEAD_CHAR + config.GREP_LINE_MAX_CHAR
    return max(1, min(config.GREP_MAX_HIT,
                      config.RLM_MAX_OUTPUT_CHAR // per_hit))


def _grep_error_dict(error: str, glob_list: list[str]) -> dict:
    """Build what grep_source returns for a pattern or a glob it cannot read.

    Args:
        error: What was wrong with it.
        glob_list: The globs to put in "globs".

    Returns:
        The keys of a count_only=False result, "globs" as glob_list, "counts_complete"
        True and every other value 0, False or empty, plus "error".
    """
    return {"error": error, "total": 0, "shown": 0, "truncated": False,
            "counts_complete": True, "globs": glob_list, "unread": [], "counts": [],
            "hits": []}


def grep_source(pattern: str, glob: str | None = None,
                count_only: bool = False) -> dict:
    """Grep the target repository: per-file counts plus the first hits with context.

    A pattern re cannot read, a glob pathlib cannot read, and a glob that names no file
    to search (none read and none unreadable) each come back with no match and "error"
    on them.

    A glob with no "/" in it is tried in every directory, and a directory a glob
    matches stands for every file under it (_walk_glob, _iter_file).

    The return value has two levels. "counts" and "total" cover every match, while
    "hits" holds only as much as fits in one turn (RLM_MAX_OUTPUT_CHAR). The first
    GREP_CONTEXT_HIT entries carry GREP_CONTEXT_LINE lines of context.

    count_only=True returns the per-file counts alone, with no line contents.

    A file of a language that ignores case is matched in either case. A line of a
    language that can be in fixed format (COBOL: a sequence area, an indicator column
    and code up to column 72) comes back with its spaces as they are; every other line
    comes back with each run of whitespace folded into one space, with none left at
    either end.

    Args:
        pattern: A regular expression as understood by Python's re, as a string or
            compiled from one.
        glob: A glob relative to PROJECT_DIR, or a path under it (an absolute one
            under the repository's place on the starting machine is read from there,
            host_path_of). None or an empty
            string walks every file under it.
        count_only: True returns only the per-file match counts.

    Returns:
        count_only=False:
            {"total": int,        # matching lines (up to GREP_MAX_HIT_COUNT)
             "shown": int,        # how many went into "hits"
             "truncated": bool,   # "hits" is not the full set
             "counts_complete": bool,  # False when the walk stopped before the end
             "globs": list[str],  # the globs actually walked, as _walk_glob() gives them
             "unread": list[str], # files the walk could not read and did not search
             "counts": [{"path": str, "count": int}],  # most hits first
             "hits": [{"path": str, "line": int, "text": str,
                       "before": list[str], "after": list[str]}]}
                                  # before / after only on the first GREP_CONTEXT_HIT
        count_only=True:
            {"counts": [...], "total": int, "globs": list[str], "unread": list[str],
             "truncated": bool,   # matches reached GREP_MAX_HIT_COUNT
             "counts_complete": bool}
        A pattern or a glob that cannot be read, or a glob that names no file to search:
            the count_only=False keys, "globs" holding glob (config.DEFAULT_GLOB_TUPLE
            when it is None or empty, and as _walk_glob() gives it when it names no file
            to search), "counts_complete" True and every other value 0, False or empty,
            plus {"error": str}
    """
    if isinstance(glob, str) and host_path_of(glob) != glob:
        # Kept to the path itself, where a glob with no "/" is tried in every directory
        glob = host_path_of(glob)
        glob = glob if "/" in glob or glob == "." else f"./{glob}"
    glob_list = [glob] if glob else list(config.DEFAULT_GLOB_TUPLE)
    is_pattern = isinstance(pattern, str) or (
        isinstance(pattern, re.Pattern) and isinstance(pattern.pattern, str))
    if not is_pattern or not isinstance(glob, (str, type(None))):
        return _grep_error_dict("pattern and glob must be strings", glob_list)
    # re raises more than re.error over a pattern: a repeat count too large, flags
    # that cannot go together, groups nested too deep
    try:
        rx = re.compile(pattern)
    except (re.error, ValueError, OverflowError, RecursionError) as e:
        return _grep_error_dict(f"invalid regex: {e}", glob_list)
    limit = hit_limit()
    count_by_path_dict = Counter()
    hit_list, use_char_count, total_count = [], 0, 0
    is_count_cut = False
    is_hit_full = count_only   # with count_only no line contents are collected at all
    unread_list: list[str] = []
    read_list: list[str] = []
    walk_list = [_walk_glob(glob_entry) for glob_entry in glob_list]

    # One walk. Every match is counted; line contents are collected within the budget
    try:
        for path_rel, line_no, line_list in _iter_match(rx, walk_list, unread_list,
                                                        read_list):
            total_count += 1
            count_by_path_dict[path_rel] += 1
            if not is_hit_full:
                entry_dict = _hit_entry(
                    path_rel, line_no, line_list,
                    with_context=len(hit_list) < config.GREP_CONTEXT_HIT)
                char_count = _entry_char_count(entry_dict)
                # The first hit comes back even when it is over budget, and every hit
                # up to the limit does when RLM_MAX_OUTPUT_CHAR sets no budget
                if not hit_list or config.RLM_MAX_OUTPUT_CHAR <= 0 \
                        or use_char_count + char_count <= config.RLM_MAX_OUTPUT_CHAR:
                    hit_list.append(entry_dict)
                    use_char_count += char_count
                    is_hit_full = len(hit_list) >= limit
                else:
                    is_hit_full = True
            if total_count >= config.GREP_MAX_HIT_COUNT:
                is_count_cut = True
                break
    except (NotImplementedError, ValueError) as e:
        return _grep_error_dict(f"invalid glob: {e}", glob_list)
    if glob and not read_list and not unread_list:
        return _grep_error_dict(
            f"the glob names no file to search: {glob}. A glob is relative to the "
            f"repository root, one with no / in it is tried in every directory, and "
            f"a directory stands for every file under it", walk_list)

    count_list = [{"path": path, "count": count}
                  for path, count in count_by_path_dict.most_common()]
    # The walk stops where the count limit was reached: the files after that point are
    # not in "counts" at all, and "counts_complete" is False
    if count_only:
        return {"counts": count_list, "total": total_count, "globs": walk_list,
                "unread": unread_list, "truncated": is_count_cut,
                "counts_complete": not is_count_cut}
    return {"total": total_count, "shown": len(hit_list),
            "truncated": len(hit_list) < total_count,
            "counts_complete": not is_count_cut, "globs": walk_list,
            "unread": unread_list, "counts": count_list, "hits": hit_list}


def _repo_path(path: str) -> str:
    """Return the repository-relative path a given path names.

    Examples:
        ("myapp/myapp/a_py/a.py") (project name myapp) -> "myapp/a.py"
        ("./myapp/a.py"), ("tests/../myapp/a.py")      -> "myapp/a.py"

    Args:
        path: A knowledge path of project_name, or a path in any notation
            to_repo_relative() reads.

    Returns:
        The repository-relative path.
    """
    try:
        return knowledge_to_repo(path, project_name)
    except ValueError:
        return to_repo_relative(path)


def get_use_of_file(target_file: str) -> list[dict]:
    """Return the files referring to this file and where they refer to it.

    The argument is converted with repo_to_knowledge() to the knowledge path the keys of
    _file_use_dict are, and looked up by equality, never as a substring.

    Args:
        target_file: A repository-relative path, e.g. "myapp/config/settings.py".
            A knowledge path is accepted too, and so is any notation _repo_path()
            reads.

    Returns:
        [{"file": str,        # repository-relative path of the referring side
          "name": str,        # the name being used; can be of the form "obj.attr"
          "lines": list[int]}, ...]
        An empty list when nothing refers to it, and for a target that is not a
        path. It never raises.
    """
    try:
        target_file = path_text(target_file)
    except ValueError:
        return []
    knowledge_path = repo_to_knowledge(_repo_path(target_file), project_name)
    return [{"file": knowledge_to_repo(use_path, project_name), "name": name,
             "lines": line_list}
            for use_path, name, line_list in _file_use_dict.get(knowledge_path, ())]


def _inner_def_list(name: str) -> list[dict]:
    """Return the definitions a name written with "." between its parts stands for.

    Examples:
        "B.run" -> the definitions named run whose lines lie inside a definition named
                   B of the same file

    Args:
        name: The name, as integrate.analyse() writes a definition inside another.

    Returns:
        [{"key", "file"}, ...] as def_list_by_name() returns them, empty for a name
        without ".".
    """
    outer, _, inner = name.rpartition(".")
    if not outer or not inner:
        return []
    span_list = [(def_entry["file"], scope_mod.INDEX.definition(def_entry["key"])["lines"])
                 for def_entry in scope_mod.INDEX.def_list_by_name(outer.rpartition(".")[2])]
    inner_list = []
    for def_entry in scope_mod.INDEX.def_list_by_name(inner):
        start, end = scope_mod.INDEX.definition(def_entry["key"])["lines"]
        if any(outer_file == def_entry["file"] and outer_span[0] <= start
               and end <= outer_span[1] and outer_span != [start, end]
               for outer_file, outer_span in span_list):
            inner_list.append(def_entry)
    return inner_list


def _resolve_start(name: str, file: str | None,
                   on_ambiguous: str) -> tuple[str | None, dict | None]:
    """Settle a name, and a file when one is given, on exactly one definition key.

    Args:
        name: The definition name, or the "<path>::<name>" key the tools print. A key
            the index holds, with its "#<kind>#<start line>" suffix when it has one, is
            that definition whatever file and on_ambiguous say. A "#" a name holds
            anywhere else is part of the name.
        file: A repository-relative path, in any notation _repo_path() reads. None
            does not narrow.
        on_ambiguous: What to do with several candidates: "return_candidates" / "first".

    Returns:
        (the definition key to start from, None), or
        (None, a dict with "error" or with "ambiguous" and "candidates"). A name or a
        file that is not a string gets such a dict, and so does a name holding half of a
        surrogate pair, which is no definition's name.
    """
    if not isinstance(name, str) or not isinstance(file, (str, type(None))):
        return None, {"error": "name and file must be strings"}
    if no_lone_surrogate(name) != name:
        return None, {"error": f"no such definition: {name}"}
    # The key the tools print is "<path>::<name>", with "#<kind>#<start line>" behind
    # it where a file holds several definitions of one name. A string that is a key of
    # the index as it stands names that one definition
    if "::" in name:
        path, label = split_def_key(name)
        key = f"{repo_to_knowledge(_repo_path(path), project_name)}::{label}"
        if scope_mod.INDEX.definition(key) is not None:
            return key, None
    # Any other string with "::" is read as a path and a name, the suffix dropped. A
    # name that holds "::" itself and is a definition as it stands is taken as the name
    if "::" in name and not scope_mod.INDEX.def_list_by_name(name_without_suffix(name)):
        path, name = split_def_key(name)
        file = path if file is None else file
    name = name_without_suffix(name)
    def_list = scope_mod.INDEX.def_list_by_name(name) or _inner_def_list(name)
    key_list = [def_entry["key"] for def_entry in def_list]
    if not key_list:
        return None, {"error": f"no such definition: {name}"}
    if file is not None:
        file = _repo_path(file)
        want_path = repo_to_knowledge(file, project_name)
        in_file_list = [def_entry["key"] for def_entry in def_list
                        if def_entry["file"] == want_path]
        # "the name exists but not in that file" gets its own message, with the
        # candidates that do carry the name
        if not in_file_list:
            candidate_list = [scope_mod.to_repo(def_key) for def_key in sorted(key_list)]
            return None, {"error": f"{name} exists but not in {file}. Candidates: "
                                   f"{candidate_list}"}
        key_list = in_file_list
    if len(key_list) > 1 and on_ambiguous != "first":
        return None, {"ambiguous": True,
                      "candidates": [scope_mod.to_repo(def_key) for def_key in sorted(key_list)]}
    return key_list[0], None


def fresh_line_range(path: str, lines: list[int], signature: str,
                     signature_line: int | None = None) -> tuple[list[int], bool]:
    """Return the line range a definition stands on in the file as it is now.

    lines are those of the state the run started from. A file this run has written
    (edit.write_path_set(), edit.own_change_set()) is read again and the definition's
    own declaration line is looked for; of several such lines, the one nearest where the
    index holds it is taken. The range keeps its length and the declaration keeps its
    place in it. In a language whose lines can be in fixed format, two lines are the
    same declaration when their code is the same words in either case
    (cobol.line_key_list()), whatever the sequence and identification areas hold.

    Args:
        path: The definition's file, relative to PROJECT_DIR.
        lines: [start, end] as the index holds them.
        signature: The definition's declaration line, as the index holds it.
        signature_line: The line the index holds the declaration on. None takes it to
            be the first line of the range.

    Returns:
        (the line range now, whether it may be out of date). The range is the argument
        unchanged when this run has not written that file; the second member is True
        when the file was written and the declaration line was not found in it.
    """
    if path not in edit_mod.write_path_set() | edit_mod.own_change_set():
        return lines, False
    want_key = signature.strip()
    if not want_key:
        return lines, True
    try:
        full_path = resolve_in_project(path)
        tmpfile.check_file(full_path)
        line_list = line_list_of(read_source_text(full_path))
    except (OSError, ValueError) as e:
        logger.info("%s cannot be read to place %s again (%s)", path, want_key, e)
        return lines, True
    start, end = lines
    offset = 0 if signature_line is None else signature_line - start
    lang_config = lang_of(path)
    if lang_config is not None and lang_config.column_layout:
        key_list = cobol.line_key_list(line_list + [signature])
        want_key = key_list.pop()
    else:
        key_list = [text.strip() for text in line_list]
    hit_list = [line_no for line_no, key in enumerate(key_list, 1)
                if want_key and key == want_key]
    if not hit_list:
        return lines, True
    new_start = max(1, min(hit_list, key=lambda line_no: abs(line_no - start - offset))
                    - offset)
    return [new_start, new_start + (end - start)], False


def read_definition(name: str, file: str | None = None) -> dict:
    """Read one definition from the real file in the target repository, as "code".

    The index's node["code"] is never returned: only the line range comes from the
    index, and the contents are read from the real file by read_source_file.

    A definition longer than read_limit() is cut by read_source_file: "code" ends with
    the note on how to continue while "lines" still holds the whole range.

    Args:
        name: The definition name, or the "<path>::<name>" key the tools print. A key
            with the "#<kind>#<start line>" suffix names that one of the same-named
            definitions of its file, and one without it the first of them.
        file: A repository-relative path, to settle on one of several same-named
            definitions. A knowledge path is accepted too.

    Returns:
        When found:
            {"ok": True, "key": "<relative path>::<name>", "file": str,
             "lines": [start line, end line], "code": str}. "lines_stale" is True as
            well when this run has written that file and the definition could not be
            found in it any more
        When there is no such definition, or none in that file:
            {"ok": False, "error": str}
        When the name matched several definitions:
            {"ok": False, "ambiguous": True, "candidates": [...]}
    """
    key, early_dict = _resolve_start(name, file, on_ambiguous="return_candidates")
    if key is None:
        # _resolve_start only fills its second return value on failure
        return {"ok": False, **early_dict}

    node_dict = scope_mod.INDEX.definition(key)
    path = knowledge_to_repo(node_dict["file"], project_name)
    (start, end), is_stale = fresh_line_range(path, node_dict["lines"],
                                              node_dict["signature"],
                                              node_dict["signature_line"])
    code, is_read = read_source_result(path, start, end)
    if not is_read:
        return {"ok": False, "error": code[len("Error: "):]}

    # A file shorter than the index's line range is logged. It is detected from the
    # header's total line count, not from its end line, which also shrinks on a cut
    header_match = SLICE_HEADER.match(code)
    if header_match and int(header_match.group(3)) < end:
        logger.warning("the index line range is past the end of the file: %s [%d, %d]",
                       scope_mod.to_repo(key), start, end)
    answer_dict = {"ok": True, "key": scope_mod.to_repo(key), "file": path,
                   "lines": [start, end], "code": code}
    if is_stale:
        answer_dict["lines_stale"] = True
    return answer_dict


def _call_remain_count() -> int:
    """Return how many search_again calls are left.

    Returns:
        SEARCH_AGAIN_MAX_CALL minus what has been used, never below 0.
    """
    return max(0, config.SEARCH_AGAIN_MAX_CALL - _search_again_count)


def _refuse_dict(error: str) -> dict:
    """Build what search_again returns for a call it refuses.

    Args:
        error: Why the call was refused.

    Returns:
        {"ok": False, "error": error, "calls_left": _call_remain_count()}.
    """
    return {"ok": False, "error": error, "calls_left": _call_remain_count()}


def reset_search_count() -> None:
    """Reset the search_again call counter to 0.

    The counter is a module variable.
    """
    global _search_again_count
    _search_again_count = 0


def _count_argument(value: int | None, name: str, limit: int) -> tuple[int | None, str]:
    """Check one of search_again's counts and bring it inside what the settings allow.

    Args:
        value: The count passed. None leaves the setting's value.
        name: The argument's name, for the message.
        limit: The largest value the settings allow.

    Returns:
        (the count to use, ""), or (None, the reason) for a value that is not a count.
        A count above the limit comes back as the limit, with the note saying so.
    """
    if value is None:
        return None, ""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None, f"{name} must be a whole number of 1 or more, not {value!r}"
    if value > limit:
        return limit, f"{name} was cut to {limit}, which is what the settings allow"
    return value, ""


def search_again(query: str, def_k: int | None = None,
                 file_k: int | None = None) -> dict:
    """Query the index again with different ENGLISH terms and counts (step 5).

    No query expansion happens here: query is matched as it is against the index.

    A query that is not a string, one with no English word in it and none of whose
    words is in the index, and a count that is not a count, are refused without
    spending one of the calls.

    Args:
        query: English search terms, space separated.
        def_k: How many definitions BM25 keeps. None uses config.DEF_K.
        file_k: How many files to keep in files. None uses config.SCOPE_FILE_K.

    Returns:
        The dict from search_scope_from_query() with "ok" and "calls_left" added, and
        each count that was cut added to the end of "note", or
        {"ok": False, "error": str, "calls_left": int}
        when the call limit is reached or an argument is refused. "calls_left" is on
        the answer either way.
    """
    global _search_again_count
    if _search_again_count >= config.SEARCH_AGAIN_MAX_CALL:
        logger.info("search_again hit its call limit: %d calls", _search_again_count)
        return _refuse_dict(f"search_again hit its call limit "
                            f"({config.SEARCH_AGAIN_MAX_CALL} calls)")
    query_error = text_error(query=query)
    if query_error:
        return _refuse_dict(query_error)
    if not has_english_word(query) and not any(
            scope_mod.INDEX.has_term("def", term) for term in tokenize(query)):
        return _refuse_dict("no English word in the query. Pass English search terms: the "
                            "index is built from English design documents and code")
    note_list = []
    count_list = []
    for value, name, limit in ((def_k, "def_k", config.DEF_K),
                               (file_k, "file_k", config.SCOPE_FILE_K)):
        count, note = _count_argument(value, name, limit)
        if count is None and value is not None:
            return _refuse_dict(note)
        count_list.append(count)
        if note:
            note_list.append(note)

    _search_again_count += 1
    scope_dict = scope_mod.search_scope_from_query(
        query, def_k=count_list[0], file_k=count_list[1])
    for entry_dict in scope_dict["defs"] + scope_dict["expanded"]:
        path, name = split_def_key(entry_dict["key"])
        node_dict = scope_mod.INDEX.definition(
            f"{repo_to_knowledge(path, project_name)}::{name}")
        entry_dict["lines"], is_stale = fresh_line_range(
            path, entry_dict["lines"], entry_dict["signature"],
            node_dict["signature_line"] if node_dict else None)
        if is_stale:
            entry_dict["lines_stale"] = True
    scope_dict["ok"] = True
    scope_dict["calls_left"] = _call_remain_count()
    # What the search itself noted stays in front of the counts that were cut
    if note_list:
        scope_dict["note"] = " ".join(
            text for text in (scope_dict.get("note"), "; ".join(note_list)) if text)
    return scope_dict


def graph_search(name: str,
                 hop_count: int = 1,
                 direction: str = "both",
                 file: str | None = None,
                 on_ambiguous: str = "return_candidates") -> dict:
    """Follow the index's calls edges from a definition name.

    Args:
        name: The definition name, or the "<path>::<name>" key the tools print. A key
            with the "#<kind>#<start line>" suffix names that one of the same-named
            definitions of its file, and one without it the first of them.
        hop_count: How many hops to follow. 1 stays with the directly connected definitions.
        direction: "outgoing" (what this definition uses) / "incoming" (what uses this
            definition) / "both" (default).
        file: A repository-relative path, to settle on one of several same-named
            definitions. A knowledge path is accepted too.
        on_ambiguous: What to do when several candidates match.
            "return_candidates" (default) ... return the candidates and stop
            "first" ... take the first one and go on

    Returns:
        When found:
            {"start": "<relative path>::<name>", "hops": int, "direction": str,
             "nodes": [{"key", "file", "name", "type", "hop", "via"}],
             "edges": [{"source", "target", "hop"}]}
        With several candidates and on_ambiguous "return_candidates":
            {"ambiguous": True, "candidates": ["<relative path>::<name>", ...]}
        When nothing was found, or an argument is invalid:
            {"error": str}
    """
    if direction not in ("outgoing", "incoming", "both"):
        return {"error": f"direction must be outgoing / incoming / both: {direction}"}
    # on_ambiguous is checked as well: a value like "First" is refused
    if on_ambiguous not in ("return_candidates", "first"):
        return {"error": f"on_ambiguous must be return_candidates / first: "
                         f"{on_ambiguous}"}

    hop_error = whole_number_error("hop_count", hop_count)
    if hop_error:
        return {"error": hop_error}

    start, early_dict = _resolve_start(name, file, on_ambiguous)
    if early_dict is not None:
        return early_dict

    # The adjacency tables of search.scope, {} when it holds none
    out_adj_dict = scope_mod.OUT_ADJ_DICT or {}
    in_adj_dict = scope_mod.IN_ADJ_DICT or {}

    known_set, queue_list = {start}, [(start, 0)]
    node_list, edge_list, known_edge_set = [], [], set()
    while queue_list:
        key, hop = queue_list.pop(0)
        if hop >= hop_count:
            continue
        for via, adj_dict in (("outgoing", out_adj_dict), ("incoming", in_adj_dict)):
            if direction not in (via, "both"):
                continue
            for neighbor in adj_dict.get(key, []):
                source_key, target_key = (key, neighbor) if via == "outgoing" \
                    else (neighbor, key)
                if (source_key, target_key) not in known_edge_set:
                    known_edge_set.add((source_key, target_key))
                    edge_list.append({"source": scope_mod.to_repo(source_key),
                                      "target": scope_mod.to_repo(target_key),
                                      "hop": hop + 1})
                if neighbor in known_set:
                    continue
                known_set.add(neighbor)
                def_entry = scope_mod.INDEX.definition(neighbor)
                node_list.append({"key": scope_mod.to_repo(neighbor),
                                  "file": knowledge_to_repo(def_entry["file"], project_name),
                                  "name": def_entry["name"], "type": def_entry["type"],
                                  "hop": hop + 1, "via": via})
                queue_list.append((neighbor, hop + 1))
    return {"start": scope_mod.to_repo(start), "hops": hop_count,
            "direction": direction, "nodes": node_list, "edges": edge_list}
