"""Index steps 1-2: definition and file nodes, and the calls edges."""

import logging
import re
from collections.abc import Iterable

from codetwine_loom.index.store import is_constant
from codetwine_loom.knowledge import Knowledge
from codetwine_loom.languages import (LANG_DICT, add_file_lang, def_key_suffix, ext_of,
                                      file_lang, lang_of)

logger = logging.getLogger(__name__)

# The fields of a data item the analysis gives that the definition nodes keep
_ITEM_FIELD_TUPLE = ("level", "is_group")

# How a line starts that is a decorator, an annotation or an attribute: "@" (Python,
# Java, Kotlin, TypeScript), "[" (C#, C++ "[[...]]") and "#[" (Rust)
_ANNOTATION_START_TUPLE = ("@", "[", "#[")

# The brackets an annotation's arguments are written in, which carry it onto the next
# lines until they close
_OPEN_BRACKET = "([{"
_CLOSE_BRACKET = ")]}"


def _annotation_line_count(line_list: list[str]) -> int:
    """Return how many lines at the top of a definition's code are its annotations.

    A line starting with one of _ANNOTATION_START_TUPLE begins an annotation, and the
    lines after it belong to it while a bracket it opened is still open.

    Examples:
        ["@app.route(", "    '/login')", "def login():"] -> 2

    Args:
        line_list: The definition's code, one line each.

    Returns:
        The number of leading lines that are annotations.
    """
    depth = 0
    for index, line in enumerate(line_list):
        if depth <= 0 and not line.lstrip().startswith(_ANNOTATION_START_TUPLE):
            return index
        depth += (sum(line.count(char) for char in _OPEN_BRACKET)
                  - sum(line.count(char) for char in _CLOSE_BRACKET))
    return len(line_list)


def signature_index(context: str, name: str, hyphen_name: bool = False) -> int:
    """Return which line of a definition's code declares it, counted from 0 (step 1).

    A decorator or an annotation is part of the definition's line range, and the first
    line of the code is not always the declaration. The first line after the annotations
    (_annotation_line_count()) holding the definition's own name as a whole word is.
    Failing that, the first line anywhere holding it as a whole word, then the first
    line holding it at all.

    Examples:
        ("@property\\ndef size(self):", "size") -> 1
        ("@property\\ndef prop(self):", "prop") -> 1
        ("@app.route('/login')\\ndef login():", "login") -> 1

    Args:
        context: The definition's code, as the analysis carries it.
        name: The definition's name.
        hyphen_name: Whether "-" is part of a word, as in a language whose names join
            words with it (LangConfig.hyphen_name).

    Returns:
        The index of the line picked, or 0 when no line holds the name.
    """
    line_list = context.split("\n")
    word_char = r"[\w-]" if hyphen_name else r"\w"
    word = re.compile(rf"(?<!{word_char}){re.escape(name)}(?!{word_char})")
    word_index_list = [index for index, line in enumerate(line_list) if word.search(line)]
    annotation_count = _annotation_line_count(line_list)
    for index in word_index_list:
        if index >= annotation_count:
            return index
    if word_index_list:
        return word_index_list[0]
    return next((index for index, line in enumerate(line_list) if name in line), 0)


def signature_of(context: str, name: str) -> str:
    """Return the line of a definition's code that declares it (step 1).

    Examples:
        ("@property\\ndef size(self):", "size") -> "def size(self):"

    Args:
        context: The definition's code, as the analysis carries it.
        name: The definition's name.

    Returns:
        The line signature_index() picks.
    """
    return context.split("\n")[signature_index(context, name)]


def add_def_node(def_dict: dict, file_entry: dict) -> None:
    """Add one file's definitions to the definition nodes (step 1).

    The ID is "<file path>::<definition name>". When one file holds the same name more
    than once, every occurrence after the first gets "#<type>#<start line>" appended.
    The code the signature is taken from is not kept; "signature_line" says which line
    of the file the signature is: the analysis's "name_line" where it gives one within
    the definition's line range, otherwise the line signature_index() picks. The
    analysis's "level" and "is_group" of a data item are kept.

    Args:
        def_dict: The nodes built so far. It is added to.
        file_entry: One file entry of the analysis.
    """
    lang_config = LANG_DICT.get(file_lang(file_entry) or ext_of(file_entry["file"]))
    hyphen_name = lang_config is not None and lang_config.hyphen_name
    for def_entry in file_entry["file_dependencies"]["definitions"]:
        # A definition whose line range is reversed is skipped, not stored
        if def_entry["start_line"] > def_entry["end_line"]:
            logger.warning("reversed line range, skipping: %s::%s [%d, %d]",
                           file_entry["file"], def_entry["name"], def_entry["start_line"],
                           def_entry["end_line"])
            continue
        key = file_entry["file"] + "::" + def_entry["name"]
        if key in def_dict:
            key += def_key_suffix(def_entry["type"], def_entry["start_line"])
            logger.debug("same-named definitions split by type and line: %s", key)
        if (def_entry["start_line"] <= def_entry.get("name_line", -1)
                <= def_entry["end_line"]):
            index = def_entry["name_line"] - def_entry["start_line"]
        else:
            index = signature_index(def_entry["context"], def_entry["name"], hyphen_name)
        def_dict[key] = {
            "name": def_entry["name"],
            "type": def_entry["type"],
            "file": file_entry["file"],
            "lines": [def_entry["start_line"], def_entry["end_line"]],
            "signature": def_entry["context"].split("\n")[index],
            "signature_line": def_entry["start_line"] + index,
            "doc": "",
        }
        def_dict[key].update({field: def_entry[field] for field in _ITEM_FIELD_TUPLE
                              if field in def_entry})


def build_def_node_dict(knowledge: Knowledge) -> dict:
    """Turn every definition of an analysis into one node (step 1).

    The files the analysis read in a language their extension does not say are handed to
    languages.add_file_lang().

    Args:
        knowledge: The analysis to read the files from.

    Returns:
        {definition key: {"name", "type", "file", "lines", "signature",
        "signature_line", "doc"}}, with "level" and "is_group" where the analysis gives
        them. "doc" is empty.
    """
    def_dict: dict = {}
    lang_dict: dict = {}
    for file_entry in knowledge.iter_file():
        add_def_node(def_dict, file_entry)
        if lang := file_lang(file_entry):
            lang_dict[file_entry["file"]] = lang
    add_file_lang(lang_dict)
    return def_dict


def def_by_file_dict(def_dict: dict) -> dict:
    """Group the definition nodes as {file path: [(start, end, definition key), ...]}.

    The order is defs' insertion order.

    Args:
        def_dict: The result of build_def_node_dict().

    Returns:
        (start line, end line, definition key) lists keyed by file path.
    """
    by_file_dict = {}
    for key, node_dict in def_dict.items():
        start, end = node_dict["lines"]
        by_file_dict.setdefault(node_dict["file"], []).append((start, end, key))
    return by_file_dict


def mark_record(def_dict: dict, by_file_dict: dict) -> None:
    """Mark which data items of a language with a record_type are records (step 1).

    An item of that type is a record when the analysis says it is a group ("is_group")
    and no item of that type in its file with a lower level number encloses its line
    range. Every item of that type gets "record" set, True or False. In a file whose
    items the analysis gives no level, none of them is a record, and how many such files
    there are is logged once.

    Args:
        def_dict: The definition nodes. They are added to.
        by_file_dict: The result of def_by_file_dict().
    """
    no_level_path_list = []
    for path, entry_list in by_file_dict.items():
        lang_config = lang_of(path)
        if lang_config is None or not lang_config.record_type:
            continue
        item_list = sorted((entry for entry in entry_list
                            if def_dict[entry[2]]["type"] == lang_config.record_type),
                           key=lambda entry: (entry[0], -entry[1]))
        if any("level" not in def_dict[key] for _, _, key in item_list):
            no_level_path_list.append(path)
            for _, _, key in item_list:
                def_dict[key]["record"] = False
            continue
        # (end line, level) of the items before this one whose line range can hold it
        open_list: list[tuple[int, int]] = []
        for start, end, key in item_list:
            node_dict = def_dict[key]
            open_list = [(open_end, level) for open_end, level in open_list
                         if open_end >= start]
            is_top = not any(open_end >= end and level < node_dict["level"]
                             for open_end, level in open_list)
            node_dict["record"] = is_top and node_dict.get("is_group", False)
            open_list.append((end, node_dict["level"]))
    if no_level_path_list:
        logger.warning("the analysis gives the data items of %d file(s) no level, e.g. %s; "
                       "none of them is taken as a record. Remove the analysis and build "
                       "it again to have them", len(no_level_path_list),
                       no_level_path_list[0])


def add_hold_name(def_dict: dict, by_file_dict: dict) -> None:
    """Give each non-constant definition the names of the constants it holds (step 1).

    In a language with a record_type, the name of each definition is_constant() calls a
    constant goes to the narrowest non-constant definition of its file whose line range
    contains it. A constant that no non-constant definition contains goes nowhere.
    mark_record() runs first.

    Args:
        def_dict: The definition nodes. Each definition that holds a name gets
            "hold_name_list", the names in the order they are written.
        by_file_dict: The result of def_by_file_dict().
    """
    for path, entry_list in by_file_dict.items():
        lang_config = lang_of(path)
        if lang_config is None or not lang_config.record_type:
            continue
        search_list, out_list = [], []
        for entry in entry_list:
            (out_list if is_constant(entry[2], def_dict[entry[2]])
             else search_list).append(entry)
        for start, end, key in sorted(out_list):
            best_tuple = narrowest_entry(search_list, start, end)
            if best_tuple is not None:
                def_dict[best_tuple[2]].setdefault("hold_name_list", []).append(
                    def_dict[key]["name"])


def key_by_file_name_dict(def_dict: dict) -> dict:
    """Build the "file -> definition name -> definition keys" mapping in one pass.

    Keys that step 1 disambiguated with "#<type>#<start line>" (Java constructors,
    overloads) are collected under the same name, in the order step 1 built them, which
    is the order they are written in the file.

    Args:
        def_dict: The result of build_def_node_dict().

    Returns:
        {file path: {definition name: [definition key, ...]}}.
    """
    key_dict: dict = {}
    for key, node_dict in def_dict.items():
        key_dict.setdefault(node_dict["file"], {}).setdefault(
            node_dict["name"], []).append(key)
    return key_dict


def outer_def(by_file_dict: dict, file_path: str, line: int) -> str | None:
    """Return the narrowest definition whose line range contains a line (step 2).

    The first argument is the result of def_by_file_dict(defs), not defs itself.

    What comes back is the definition key, including any "#<type>#<start line>" suffix,
    not the definition name.

    Args:
        by_file_dict: The result of def_by_file_dict().
        file_path: The referring side's file path (a knowledge path).
        line: The referring line number.

    Returns:
        The definition key, or None when no definition encloses that line.
    """
    best_tuple = narrowest_entry(by_file_dict.get(file_path, ()), line, line)
    return best_tuple[2] if best_tuple else None


def narrowest_entry(entry_iter: Iterable[tuple[int, int, str]], start: int,
                    end: int) -> tuple[int, int, str] | None:
    """Return the entry with the narrowest line range that contains a range.

    Args:
        entry_iter: (start line, end line, definition key) entries, as
            def_by_file_dict() holds them.
        start: The first line of the range to contain.
        end: The last line of the range to contain.

    Returns:
        The first entry of the narrowest such range, or None when no entry contains it.
    """
    best_tuple = None
    for entry_start, entry_end, key in entry_iter:
        if entry_start <= start and end <= entry_end and (
                best_tuple is None
                or entry_end - entry_start < best_tuple[1] - best_tuple[0]):
            best_tuple = (entry_start, entry_end, key)
    return best_tuple


def _owner_key_dict(entry_list: list[tuple[int, int, str]]) -> dict[str, str | None]:
    """Return which definition each definition of one file is written directly inside.

    Args:
        entry_list: The (start line, end line, definition key) entries of one file, as
            def_by_file_dict() holds them.

    Returns:
        {definition key: the key of the narrowest definition whose line range contains
        its range and is wider than it, or None when there is none}.
    """
    owner_dict: dict[str, str | None] = {}
    # (start line, end line, key) of the definitions around the one being looked at
    open_list: list[tuple[int, int, str]] = []
    for start, end, key in sorted(entry_list, key=lambda entry: (entry[0], -entry[1])):
        while open_list and open_list[-1][1] < end:
            open_list.pop()
        # A definition of the same line range (int a, b;) is beside it, not around it
        owner = next((entry for entry in reversed(open_list)
                      if (entry[0], entry[1]) != (start, end)), None)
        owner_dict[key] = owner[2] if owner else None
        open_list.append((start, end, key))
    return owner_dict


def target_key_dict(by_file_name_dict: dict, by_file_dict: dict) -> dict:
    """Build the "(file, definition name, start line) -> definition keys" mapping (step 2).

    The keys are the definition that starts on the line and the definitions of the same
    name it is not told apart from: the ones written directly inside the same
    definition as it (the overloads of a method, a property and its setter), and the
    ones written directly inside one of those (the constructors of a class). They come
    in the order they are written.

    Args:
        by_file_name_dict: The result of key_by_file_name_dict().
        by_file_dict: The result of def_by_file_dict().

    Returns:
        {(file path, definition name, start line): [definition key, ...]}.
    """
    target_dict: dict = {}
    for file_path, entry_list in by_file_dict.items():
        owner_dict = _owner_key_dict(entry_list)
        start_dict = {key: start for start, _, key in entry_list}
        for name, key_list in by_file_name_dict.get(file_path, {}).items():
            key_set = set(key_list)
            # The definition a key is grouped under: its owner, or the owner of that
            # one while the owner is a definition of the same name
            group_owner_dict: dict = {}
            for key in key_list:
                owner = owner_dict[key]
                while owner in key_set:
                    owner = owner_dict[owner]
                group_owner_dict[key] = owner
            group_dict: dict = {}
            for key in key_list:
                group_dict.setdefault(group_owner_dict[key], []).append(key)
            for key in key_list:
                target_dict[(file_path, name, start_dict[key])] = group_dict[group_owner_dict[key]]
    return target_dict


def resolve_usage(target_dict: dict, usage: dict) -> list[str]:
    """Resolve one usage of the analysis to the definitions it leads to (step 2).

    A usage is resolved by its "target_name" and "target_start_line", the definition the
    analysis names (target_key_dict()); one whose "target_name" is None leads to no
    definition.

    Args:
        target_dict: The result of target_key_dict().
        usage: A callee_usages entry, or a same_file_usages entry with "from" set to
            its own file.

    Returns:
        The definition keys, or [] when the usage leads to no definition node.
    """
    if usage["target_name"] is None:
        return []
    return list(target_dict.get(
        (usage["from"], usage["target_name"], usage["target_start_line"]), ()))


def no_resolve_ref_dict(knowledge: Knowledge, def_dict: dict) -> dict:
    """Return the references that lead to no definition of the file they point at.

    These are the callee_usages entries resolve_usage() finds no definition for in the
    referenced file.

    Args:
        knowledge: The analysis to read the files from.
        def_dict: The result of build_def_node_dict(), for resolving a referenced name.

    Returns:
        {(referring file, referenced file, name): [line, ...]}, the lines being where
        the reference is written, sorted and without repeats. Both paths are knowledge
        paths.
    """
    by_file_name_dict = key_by_file_name_dict(def_dict)
    target_dict = target_key_dict(by_file_name_dict, def_by_file_dict(def_dict))
    ref_dict: dict = {}
    for file_entry in knowledge.iter_file():
        for usage in file_entry["file_dependencies"]["callee_usages"]:
            if resolve_usage(target_dict, usage):
                continue
            key = (file_entry["file"], usage["from"], usage["name"])
            ref_dict.setdefault(key, set()).update(usage["lines"] or ())
    return {key: sorted(line_set) for key, line_set in ref_dict.items()}


def build_call_list(usage_dict: dict, by_file_name_dict: dict,
                    by_file_dict: dict) -> list[tuple[str, str]]:
    """Build the calls edges (step 2).

    A reference is one entry per (referred file, name) with every line it is written on,
    and every one of those lines is looked at: a name used from more than one definition
    of the file is an edge from each of them.

    Args:
        usage_dict: {referring file: [usage, ...]}: its callee_usages entries and its
            same_file_usages entries with "from" set to the file.
        by_file_name_dict: The result of key_by_file_name_dict(), for resolving a
            reference (resolve_usage()).
        by_file_dict: The result of def_by_file_dict(), for finding the definition
            around a referring line.

    Returns:
        [(calling definition key, called definition key), ...], sorted and without
        repeats: several references from one definition to the same target are one edge.
    """
    call_set = set()
    target_dict = target_key_dict(by_file_name_dict, by_file_dict)
    for path, usage_list in usage_dict.items():
        for usage in usage_list:
            target_list = resolve_usage(target_dict, usage)
            if not target_list:
                logger.debug("cannot resolve the target: %s -> %s",
                             usage["name"], usage["from"])
                continue
            for line in usage["lines"] or [0]:
                caller_key = outer_def(by_file_dict, path, line)
                if caller_key is None:
                    # A line no definition encloses. There is no node to connect
                    logger.debug("no definition encloses the line: %s:%d", path, line)
                    continue
                call_set.update((caller_key, target) for target in target_list)
    return sorted(call_set)
