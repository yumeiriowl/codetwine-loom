"""Index step 3: attach the design docs to the definitions.

The attachment goes (1) split by heading -> (2) match the rest by backquoted names ->
(3) append the file summary. (3) appends the summary to every definition of the file; a
definition that gets no text from (1) or (2) and whose file has no summary keeps an
empty doc.
"""

import logging
import re
from collections.abc import Callable

from codetwine_loom.index.graph import narrowest_entry
from codetwine_loom.languages import name_fold
from codetwine_loom.rlm import one_line

logger = logging.getLogger(__name__)

# A heading of a section that is a backquoted name, at any depth from 1 to 4 #s
_TITLE_RE = re.compile(r'^#{1,4}\s+`([^`]+)`\s*$', re.M)

# A backquoted name inside a paragraph
_BACKQUOTE = re.compile(r'`([^`\n]{1,80})`')

# What joins an owner and a member in a heading such as `Store.close` or `ns::Store::close`
_QUALIFIER = re.compile(r'::|\.|#')


def file_doc(file_entry: dict) -> dict:
    """Return one file entry's design document, empty where it has none.

    A file entry with no "doc" key comes back with every member empty.

    Args:
        file_entry: One file entry of the analysis.

    Returns:
        {"summary": str, "sections": list, "source_hash": str}. "source_hash" is the
        sha256 of the source the document was written from. A missing member comes
        back empty.
    """
    doc = file_entry.get("doc") or {}
    return {"summary": doc.get("summary") or "",
            "sections": doc.get("sections") or [],
            "source_hash": doc.get("source_hash") or ""}


def split_by_title(content: str) -> list[tuple[str, str]]:
    """Split one section of a design document by its ``## `name``` headings.

    Args:
        content: The content of one doc.sections[] entry.

    Returns:
        [(the heading's name, the body from that heading up to the next one), ...], in
        the order the headings are written. The name is the text in backquotes with any
        "(...)" after it removed.
    """
    body_list, cut_list = [], [(match.start(), match.group(1).split("(")[0].strip())
                               for match in _TITLE_RE.finditer(content)]
    for i, (pos, name) in enumerate(cut_list):
        end = cut_list[i + 1][0] if i + 1 < len(cut_list) else len(content)
        body_list.append((name, content[pos:end].strip()))
    return body_list


def heading_name(title: str, name_dict: dict[str, str],
                 fold: Callable[[str], str] = str) -> tuple[str, str]:
    """Return the definition name a heading stands for, and the owner it names.

    A heading that is a definition name of the file is taken as it is. Otherwise its
    last "."-, "::"- or "#"-separated part is taken when that part is one, and the part
    before it is the owner.

    Examples:
        ("load", {"load": "load"})                   -> ("load", "")
        ("Store.close", {"close": "close"})          -> ("close", "Store")
        ("ns::Store::close", {"close": "close"})     -> ("close", "Store")
        ("Store.close", {"open": "open"})            -> ("Store.close", "")
        ("main-para", {"MAIN-PARA": "Main-Para"}, str.upper) -> ("Main-Para", "")

    Args:
        title: The heading's name, as split_by_title() returns it.
        name_dict: {a definition name of the file the document describes, as fold
            returns it: that name}.
        fold: What a name is compared through: str, or str.upper for a language that
            ignores case.

    Returns:
        (definition name, owner name). An owner that is a definition name comes back
        as that name. The owner is "" when the heading names none, and the name is the
        title unchanged when no part of it is a definition name.
    """
    if fold(title) in name_dict:
        return name_dict[fold(title)], ""
    part_list = [part for part in _QUALIFIER.split(title) if part]
    if len(part_list) > 1 and fold(part_list[-1]) in name_dict:
        return (name_dict[fold(part_list[-1])],
                name_dict.get(fold(part_list[-2]), part_list[-2]))
    return title, ""


def _owner_name(key: str, def_dict: dict, by_file_dict: dict) -> str:
    """Return the name of the narrowest other definition enclosing one definition.

    Args:
        key: The definition key.
        def_dict: The definition nodes.
        by_file_dict: The result of def_by_file_dict().

    Returns:
        The enclosing definition's name, or "" when none encloses it.
    """
    node_dict = def_dict[key]
    start, end = node_dict["lines"]
    best_tuple = narrowest_entry((entry for entry in by_file_dict.get(node_dict["file"], ())
                                  if entry[2] != key), start, end)
    return def_dict[best_tuple[2]]["name"] if best_tuple else ""


def _pair_body(entry_list: list[tuple[str, str]], key_list: list[str],
               owner_of: Callable[[str], str]) -> dict:
    """Pair the headings of one name with the definitions of that name.

    The headings naming one owner go to the definitions that owner encloses, in the
    order both are written, when there are as many of each. The headings left are
    matched one to one, in the order both are written, with the definitions left;
    definitions past the last of those headings take that last one.

    Args:
        entry_list: [(owner name, body), ...] for the headings of one name, in order.
        key_list: The definition keys of that name, in the order they are written.
        owner_of: (definition key) -> the name of the definition enclosing it.

    Returns:
        {definition key: body}.
    """
    body_by_key_dict: dict = {}
    free_key_list = list(key_list)
    use_set: set = set()
    for owner in dict.fromkeys(owner for owner, _ in entry_list if owner):
        index_list = [i for i, (entry_owner, _) in enumerate(entry_list)
                      if entry_owner == owner]
        hit_list = [key for key in free_key_list if owner_of(key) == owner]
        if len(index_list) != len(hit_list):
            continue
        for i, key in zip(index_list, hit_list):
            body_by_key_dict[key] = entry_list[i][1]
            free_key_list.remove(key)
            use_set.add(i)
    rest_body_list = [body for i, (_, body) in enumerate(entry_list) if i not in use_set]
    if not rest_body_list:
        return body_by_key_dict
    for i, key in enumerate(free_key_list):
        body_by_key_dict[key] = rest_body_list[min(i, len(rest_body_list) - 1)]
    return body_by_key_dict


def attach_doc(doc_dict: dict, def_dict: dict, by_file_name_dict: dict,
               by_file_dict: dict) -> dict:
    """Fill in the definition nodes' "doc" (step 3), rewriting defs in place.

    Every section is split by its headings, whatever its id. The headings are paired
    with the definitions by _pair_body().

    Args:
        doc_dict: {file path: file_doc()'s return value}.
        def_dict: The result of build_def_node_dict(). This dict is rewritten.
        by_file_name_dict: The result of key_by_file_name_dict().
        by_file_dict: The result of def_by_file_dict().

    Returns:
        The rewritten defs (the same dict as the argument).
    """
    summary_only_count = 0  # definitions whose doc is (3)'s line alone
    for path, doc in doc_dict.items():
        by_name_dict = by_file_name_dict.get(path, {})
        fold = name_fold(path)
        name_dict = {fold(name): name for name in by_name_dict}
        section_dict = {section["id"]: section["content"] for section in doc["sections"]}

        # (1) split every section by its headings and pair them with the definitions
        entry_by_name_dict: dict = {}
        for content in section_dict.values():
            for title, body in split_by_title(content):
                name, owner = heading_name(title, name_dict, fold)
                entry_by_name_dict.setdefault(name, []).append((owner, body))
        for name, entry_list in entry_by_name_dict.items():
            body_by_key_dict = _pair_body(entry_list, by_name_dict.get(name, []),
                                          lambda key: _owner_name(key, def_dict, by_file_dict))
            for key, body in body_by_key_dict.items():
                def_dict[key]["doc"] = body

        # (2) fill definitions still without a doc from paragraphs containing their
        # backquoted name. Only names that file defines take part, and a paragraph that
        # is a heading and nothing else is skipped
        for content in section_dict.values():
            for para in content.split("\n\n"):
                if _TITLE_RE.fullmatch(para.strip()):
                    continue
                hit_set = {name_dict[fold(quote.split("(")[0].strip())]
                           for quote in _BACKQUOTE.findall(para)
                           if fold(quote.split("(")[0].strip()) in name_dict}
                for name in hit_set:
                    for key in by_name_dict[name]:
                        if not def_dict[key]["doc"]:
                            def_dict[key]["doc"] = para.strip() + "\n\n"

        # (3) append the whole file summary, folding newlines into single spaces. This
        # runs over every definition in by_name_dict, not only the ones (1)(2) missed
        head = one_line(doc["summary"])
        for key_list in by_name_dict.values():
            for key in key_list:
                if not def_dict[key]["doc"]:
                    summary_only_count += 1
                def_dict[key]["doc"] = (def_dict[key]["doc"] + "\n" + head).strip()

    # One line for the whole build, not one per file
    logger.info("definitions whose doc is the file summary alone: %d / %d",
                summary_only_count, len(def_dict))
    return def_dict
