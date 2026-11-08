"""Search steps 2-4: find definitions, follow dependencies, build the scope.

The index lives in the module variable INDEX, which load_index() fills. This module does
not import tools/.
"""

import copy
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass

from codetwine.utils.file_utils import compute_file_hash

from codetwine_loom import config, tmpfile
from codetwine_loom.index import store
from codetwine_loom.index.indexer import bm25_rank
from codetwine_loom.languages import lang_ext, set_file_lang, split_def_key
from codetwine_loom.llm import make_llm
from codetwine_loom.paths import knowledge_to_repo, resolve_in_project, under_test_path
from codetwine_loom.search.query import (NAME_WORD, PATH_MARK_TUPLE, WordRun, expand_query,
                                         need_expand, word_run_list)
from codetwine_loom.textcut import cut

logger = logging.getLogger(__name__)

INDEX = None           # the Index read from code_index.sqlite
NEIGHBOR_DICT = None           # {definition key: adjacent definition keys}, built once from calls
# The same edges with their direction kept: {calling key: called keys} and the other way
# round, built once by load_index()
OUT_ADJ_DICT = None
IN_ADJ_DICT = None
PROJECT_NAME = None  # the directory name the knowledge paths start with

# What the instructions tell the model to read before anything else, with the file
# summaries and without them
_SCOPE_FIRST = 'scope["file_summaries"] and the "signature" of each scope["defs"] entry'
_SCOPE_FIRST_NO_SUMMARY = 'the "signature" of each scope["defs"] entry'

# The description of the scope variable, in the shape rlm.build_prompt() shows it
SCOPE_VAR_DOC = """- scope (dict) — the result of the search that ran before you
  started. It has "files", "defs" (a list of {"key", "lines", "signature"}),
  "expanded" (definitions reached from those, each with the "from" it was reached
  through), "file_summaries", "callers", "confidence", "spread" and "note" (a line to
  read when the search found nothing or ran on the request untranslated)."""

# The name of the tool that searches the index again, as a stage's tools are keyed
_SEARCH_AGAIN_TOOL = "search_again"

# What "note" says when the search found nothing. {advice} is _EMPTY_ADVICE for a stage
# that has _SEARCH_AGAIN_TOOL and "" for one that does not
_EMPTY_NOTE = "The search matched nothing{why}{advice}."
_EMPTY_ADVICE = f"; {_SEARCH_AGAIN_TOOL} with English terms first"
# What "note" says when the query expansion failed and the search, run on the request as
# it was written, still found something. {advice} is _EXPAND_FAIL_ADVICE or ""
_EXPAND_FAIL_NOTE = ("The query expansion failed, so the search ran on the request as "
                     "it was written{advice}.")
_EXPAND_FAIL_ADVICE = (f"; {_SEARCH_AGAIN_TOOL} with English terms if these candidates "
                       f"do not fit")


def scope_note(scope_dict: dict, with_search_again: bool = True) -> str:
    """Return the line that says this search found nothing or ran untranslated, or "".

    Args:
        scope_dict: The result of search_scope_from_query(), its trace filled in.
        with_search_again: Whether the line is read where _SEARCH_AGAIN_TOOL can be
            called. False leaves the tool out of the line.

    Returns:
        One line, or "" when the search found something with the query expansion
        working or not asked for.
    """
    trace_dict = scope_dict.get("trace", {})
    if not trace_dict.get("empty"):
        if not trace_dict.get("expand_failed"):
            return ""
        return _EXPAND_FAIL_NOTE.format(
            advice=_EXPAND_FAIL_ADVICE if with_search_again else "")
    return _EMPTY_NOTE.format(
        why=" (the query expansion failed)" if trace_dict.get("expand_failed") else "",
        advice=_EMPTY_ADVICE if with_search_again else "")


def scope_first(brief: bool = False) -> str:
    """Return what the instructions send the model to read first.

    Args:
        brief: True for a stage whose scope was rendered without the file summaries.

    Returns:
        The phrase naming the file summaries and the signatures, or the signatures
        alone when the scope carries no summary or the index holds none at all.
    """
    if brief or (INDEX is not None and not INDEX.has_summary()):
        return _SCOPE_FIRST_NO_SUMMARY
    return _SCOPE_FIRST


def stage_variable_dict(scope_dict: dict | None,
                        tool_dict: dict[str, Callable]) -> dict[str, object]:
    """Return the variables a stage's REPL is handed for one search result.

    The stage gets a deep copy: what its code does to "scope" does not reach scope_dict.
    Where scope_dict carries a "note", the copy of a stage that is not handed
    _SEARCH_AGAIN_TOOL carries the one that does not name it (scope_note()).

    Args:
        scope_dict: The return value of search_scope().
        tool_dict: The stage's tools, as {function name: function}.

    Returns:
        {"scope": a deep copy of scope_dict}.
    """
    stage_scope_dict = copy.deepcopy(scope_dict)
    if stage_scope_dict and stage_scope_dict.get("note") \
            and _SEARCH_AGAIN_TOOL not in tool_dict:
        stage_scope_dict["note"] = scope_note(stage_scope_dict, with_search_again=False)
    return {"scope": stage_scope_dict}


def load_index(path: str, project_name: str,
               source_stamp: str | None = None) -> None:
    """Read the index, build the adjacency tables, and set the index's languages.

    It does not build an index. Starting from nothing means running build first.

    Args:
        path: The path to code_index.sqlite.
        project_name: The directory name of PROJECT_DIR (what step 0 returned).
        source_stamp: What identifies the analysis on disk (Knowledge.stamp()). When
            given, it is compared with the one the index was built from; a mismatch
            only warns and the search goes on.

    Raises:
        FileNotFoundError: When there is no index (raised by store.load()).
        ValueError: When the index's project differs from project_name.
    """
    global INDEX, NEIGHBOR_DICT, OUT_ADJ_DICT, IN_ADJ_DICT, PROJECT_NAME
    if INDEX is not None:
        INDEX.close()
    INDEX = store.load(path)
    PROJECT_NAME = project_name

    # An index built for another project is refused here, before any knowledge path is
    # converted
    if INDEX.project != project_name:
        raise ValueError(
            f"the index is for a different project: index={INDEX.project} "
            f"given={project_name}. Check the --project and --output-dir pair")

    # An analysis that was run again since the index was built only warns
    if source_stamp and source_stamp != INDEX.source_stamp:
        logger.warning("the index may be stale; re-run build: index=%s actual=%s",
                       INDEX.source_stamp, source_stamp)

    NEIGHBOR_DICT, OUT_ADJ_DICT, IN_ADJ_DICT = {}, {}, {}
    for caller_key, callee_key in INDEX.call_list:
        NEIGHBOR_DICT.setdefault(caller_key, []).append(callee_key)
        NEIGHBOR_DICT.setdefault(callee_key, []).append(caller_key)
        OUT_ADJ_DICT.setdefault(caller_key, []).append(callee_key)
        IN_ADJ_DICT.setdefault(callee_key, []).append(caller_key)

    # Hand the primary language and the languages of the files from the index to
    # config.py, and the languages of its files to languages.py; this runs before any
    # tool is called
    set_file_lang(INDEX.file_lang_dict)
    set_primary(INDEX.primary_ext)


def set_primary(ext: str) -> None:
    """Make a language the primary one, with the other languages of the loaded index.

    Args:
        ext: The language's LANG_DICT key (config.set_primary_ext).
    """
    config.set_primary_ext(ext)
    config.set_other_ext(lang_ext(knowledge_path) for knowledge_path in INDEX.file_path_list())


def _is_test_file(knowledge_path: str) -> bool:
    """Whether a file is a test file (paths.under_test_path()).

    Args:
        knowledge_path: The file's path as the index holds it.

    Returns:
        True when the file is a test.
    """
    return bool(under_test_path([knowledge_to_repo(knowledge_path, PROJECT_NAME)]))


def _is_test_def(def_key: str) -> bool:
    """Whether a definition is in a test file (_is_test_file()).

    Args:
        def_key: A definition key from the index.

    Returns:
        True when the file it is in is a test.
    """
    return _is_test_file(split_def_key(def_key)[0])


def search_def_list(query: str, k: int | None = None) -> list[dict]:
    """Search definitions with BM25 (search step 2). query must be English.

    config values are read on every call, not taken as default arguments.

    A definition in a test file goes behind every definition that is not. Within each
    of the two the order is the score's.

    Args:
        query: English search terms, space separated.
        k: How many definitions to keep. None reads config.DEF_K.

    Returns:
        Up to k entries of [{"key", "score", "raw", "hit_terms"}, ...], the definitions
        outside the tests first and each group score descending.
    """
    k = config.DEF_K if k is None else k
    entry_list = bm25_rank(query, INDEX, "def")
    entry_list.sort(key=lambda entry: _is_test_def(entry["key"]))
    return entry_list[:k]


def expand(entry_list: list[dict],
           depth: int | None = None) -> tuple[dict[str, float], dict[str, str]]:
    """Follow the calls edges breadth-first (search step 3).

    A neighbour's score is the score it was reached from times the decay, divided by
    how many definitions call the neighbour. Definitions that step 2 hit keep their
    own score whatever their in-degree, and none of them is added as reached from
    another. A definition in a test file is neither added nor followed, an entry of
    entry_list included.

    The decay per hop is read from config on each call.

    Args:
        entry_list: The result of search_def_list().
        depth: How many hops to follow. None reads config.EXPAND_DEPTH.

    Returns:
        ({definition key: score}, {definition key added: the entry-point key it was
        reached from}). The scores include the keys of entries; the second dict holds
        only what was added.
    """
    depth = config.EXPAND_DEPTH if depth is None else depth
    score_dict = {entry["key"]: entry["score"] for entry in entry_list}
    entry_key_set = set(score_dict)
    from_dict: dict[str, str] = {}
    queue_list = [(entry["key"], entry["score"], 0, entry["key"]) for entry in entry_list
                  if not _is_test_def(entry["key"])]
    while queue_list:
        key, score, hop, entry_key = queue_list.pop(0)
        if hop >= depth:
            continue
        for neighbor in NEIGHBOR_DICT.get(key, []):
            if neighbor in entry_key_set or _is_test_def(neighbor):
                continue
            new_score = (score * config.EXPAND_DECAY
                         / max(1, INDEX.indegree_dict.get(neighbor, 0)))
            if new_score > score_dict.get(neighbor, 0):
                score_dict[neighbor] = new_score
                from_dict[neighbor] = entry_key
                queue_list.append((neighbor, new_score, hop + 1, entry_key))
    return score_dict, from_dict


def to_repo(def_key: str) -> str:
    """Turn "knowledge path::definition name" into "repo-relative path::definition name".

    The key is split with split_def_key(): a key that the index build disambiguated with
    "#<type>#<start line>" keeps that suffix on the way out. The suffix is a label for
    the definition, not part of a path.

    Examples:
        "myapp/myapp/config/settings_py/settings.py::load_config"
            -> "myapp/config/settings.py::load_config"

    Args:
        def_key: A definition key from the index.

    Returns:
        The definition key with a repository-relative path.
    """
    path, name = split_def_key(def_key)
    return f"{knowledge_to_repo(path, PROJECT_NAME)}::{name}"


def _path_word_list(text: str) -> list[str]:
    """Split a path or a query into lower-case runs of word characters and hyphens.

    Examples:
        "pyproject.toml"            -> ["pyproject", "toml"]
        "Add src/Pyproject.toml"    -> ["add", "src", "pyproject", "toml"]
        "`store/memory.go:42`,"     -> ["store", "memory", "go", "42"]
        "my-file.js __init__.py"    -> ["my-file", "js", "__init__", "py"]

    Args:
        text: A file name, a path or a line of search terms.

    Returns:
        The words, in order.
    """
    return NAME_WORD.findall(text.lower())


def _part_word_list(path: str) -> list[list[str]]:
    """Split a path into its parts and each part into words (_path_word_list()).

    Examples:
        "app/store/memory.go" -> [["app"], ["store"], ["memory", "go"]]
        "./Store//memory.go:42" -> [["store"], ["memory", "go", "42"]]

    Args:
        path: A path with "/" between its parts.

    Returns:
        The words of each part, in order. A part with no word in it is left out.
    """
    return [word_list for word_list in map(_path_word_list, path.split("/"))
            if word_list]


@dataclass(frozen=True, eq=False)
class _FileName:
    """One file of the index, as a written name is compared with it.

    Attributes:
        order: Its place in the index.
        knowledge_path: Its path as the index holds it.
        repo_path: Its repository-relative path.
        part_list: The parts of repo_path that hold a word, in order.
        word_part_list: The words of each of those parts (_part_word_list()).
    """

    order: int
    knowledge_path: str
    repo_path: str
    part_list: list[str]
    word_part_list: list[list[str]]


@dataclass(eq=False)
class _TailNode:
    """The files of the index whose paths end with the same parts.

    Attributes:
        depth: How many trailing path parts the files share, the file's name counted.
        file_list: Those files, in index order.
        name_list: Those parts as each file spells them, joined with "/", each
            spelling once, in index order. None until _tail_name_list() fills it in.
        child_dict: {the words of the part before those parts: the node of the files
            that have it there}. None until _tail_child_dict() fills it in.
        count_tuple: How many words such a part holds, each number once, ascending.
    """

    depth: int
    file_list: list[_FileName]
    name_list: list[str] | None = None
    child_dict: dict[tuple[str, ...], "_TailNode"] | None = None
    count_tuple: tuple[int, ...] = ()


def _tail_name_list(node: _TailNode) -> list[str]:
    """Return the trailing parts the files of a node share, as each file spells them."""
    if node.name_list is None:
        node.name_list = list(dict.fromkeys("/".join(file.part_list[-node.depth:])
                                            for file in node.file_list))
    return node.name_list


def _tail_child_dict(node: _TailNode) -> dict[tuple[str, ...], _TailNode]:
    """Return the nodes one path part further back than a node, built on the first call.

    Args:
        node: A node of the files of the index.

    Returns:
        {the words of the path part before the parts the node's files share: the node
        of the files that have that part there}. A file with no part there is in none.
    """
    if node.child_dict is None:
        file_by_part_dict: dict[tuple[str, ...], list[_FileName]] = {}
        for file in node.file_list:
            if len(file.word_part_list) > node.depth:
                file_by_part_dict.setdefault(
                    tuple(file.word_part_list[-node.depth - 1]), []).append(file)
        node.child_dict = {key: _TailNode(node.depth + 1, file_list)
                           for key, file_list in file_by_part_dict.items()}
        node.count_tuple = tuple(sorted({len(key) for key in file_by_part_dict}))
    return node.child_dict


@dataclass(eq=False)
class _SpaceNode:
    """The files of one name with a space in it whose paths end with the same parts.

    Attributes:
        depth: How many trailing path parts the files share, the file's name counted.
        file_list: Those files, in index order.
        child_dict: {(the length of the part before those parts, that part in lower
            case): the node of the files that have it there}. None until
            _space_child_dict() fills it in.
        length_tuple: The lengths such a part has, each once, ascending.
    """

    depth: int
    file_list: list[_FileName]
    child_dict: dict[tuple[int, str], "_SpaceNode"] | None = None
    length_tuple: tuple[int, ...] = ()


def _space_child_dict(node: _SpaceNode) -> dict[tuple[int, str], _SpaceNode]:
    """Return the nodes one path part further back than a node, built on the first call.

    Args:
        node: A node of the files of one name with a space in it.

    Returns:
        {(the length of the path part before the parts the node's files share, that
        part in lower case): the node of the files that have it there}. A file with no
        part there is in none.
    """
    if node.child_dict is None:
        file_by_part_dict: dict[tuple[int, str], list[_FileName]] = {}
        for file in node.file_list:
            part_list = file.repo_path.split("/")
            if len(part_list) > node.depth:
                part = part_list[-node.depth - 1]
                file_by_part_dict.setdefault((len(part), part.lower()), []).append(file)
        node.child_dict = {key: _SpaceNode(node.depth + 1, file_list)
                           for key, file_list in file_by_part_dict.items()}
        node.length_tuple = tuple(sorted({key[0] for key in file_by_part_dict}))
    return node.child_dict


@dataclass(frozen=True)
class _FileTable:
    """The files of one index, keyed by their name.

    Attributes:
        index: The Index the table was built from.
        project_name: The PROJECT_NAME it was built with.
        node_by_base_dict: {the words of a file's last path part: the node of the
            files named so}.
        dot_node_by_base_dict: The same for the files among them whose name starts
            with a ".", a key filled in when _dot_node() is first asked for it.
        word_count_tuple: How many words a last path part holds, each number once,
            ascending.
        base_name_set: The last path part of every file, in lower case.
        text_file_list: The files whose last path part holds a character outside ASCII
            and at least _MIN_TEXT_NAME_CHAR characters, in index order.
        space_node_dict: {a last path part that is ASCII and holds a space, in lower
            case: the node of the files named so}.
        space_mark_dict: {the first run of ASCII letters, digits and "_" of such a
            part, in lower case: ((where the run starts in the part, the length of the
            part), ...), each pair once}.
    """

    index: object
    project_name: str
    node_by_base_dict: dict[tuple[str, ...], _TailNode]
    dot_node_by_base_dict: dict[tuple[str, ...], _TailNode]
    word_count_tuple: tuple[int, ...]
    base_name_set: frozenset[str]
    text_file_list: list[_FileName]
    space_node_dict: dict[str, _SpaceNode]
    space_mark_dict: dict[str, tuple[tuple[int, int], ...]]


# The table of the index in INDEX, built once for each index
_FILE_TABLE: _FileTable | None = None


def _file_table() -> _FileTable:
    """Return the table of the loaded index's files, built on the first call for it.

    Returns:
        The _FileTable of INDEX and PROJECT_NAME. A file whose last path part holds no
        word is in base_name_set only.
    """
    global _FILE_TABLE
    if _FILE_TABLE is not None and _FILE_TABLE.index is INDEX \
            and _FILE_TABLE.project_name == PROJECT_NAME:
        return _FILE_TABLE
    file_by_base_dict: dict[tuple[str, ...], list[_FileName]] = {}
    base_name_set, text_file_list = set(), []
    space_file_dict: dict[str, list[_FileName]] = {}
    space_mark_dict: dict[str, dict[tuple[int, int], None]] = {}
    for order, knowledge_path in enumerate(INDEX.file_path_list()):
        repo_path = knowledge_to_repo(knowledge_path, PROJECT_NAME)
        base_name = repo_path.rsplit("/", 1)[-1]
        base_name_set.add(base_name.lower())
        part_list = [part for part in repo_path.split("/") if _path_word_list(part)]
        if not part_list or part_list[-1] != base_name:
            continue
        file = _FileName(order, knowledge_path, repo_path, part_list,
                         [_path_word_list(part) for part in part_list])
        file_by_base_dict.setdefault(tuple(file.word_part_list[-1]), []).append(file)
        if not base_name.isascii() and len(base_name) >= _MIN_TEXT_NAME_CHAR:
            text_file_list.append(file)
        elif _SPACE in base_name:
            space_file_dict.setdefault(base_name.lower(), []).append(file)
            first_word = _ASCII_WORD_RUN.search(base_name)
            space_mark_dict.setdefault(first_word.group(0).lower(), {})[
                (first_word.start(), len(base_name))] = None
    _FILE_TABLE = _FileTable(
        INDEX, PROJECT_NAME,
        {key: _TailNode(1, file_list) for key, file_list in file_by_base_dict.items()},
        {}, tuple(sorted({len(key) for key in file_by_base_dict})),
        frozenset(base_name_set), text_file_list,
        {name: _SpaceNode(1, file_list) for name, file_list in space_file_dict.items()},
        {word: tuple(mark_dict) for word, mark_dict in space_mark_dict.items()})
    return _FILE_TABLE


def name_hit_list(name_list: list[str]) -> tuple[list[str], list[str]]:
    """List the files one of the names names (step 4).

    A file's name is the last part of its repository-relative path, split into words
    (_path_word_list()). A name names the file when its last part, split the same way,
    holds those words. The parts before it count for as long as each holds the words of
    the file's part at the same place (_tail_child_dict()). Every file of the analysis
    is looked at, whether or not it is in the file index.

    Files with the same name are one group. Within a group only the files whose path
    is written the furthest back are kept, so "store/memory.go" keeps store/memory.go
    and drops cache/memory.go. A group one file is left in is a sure hit; a group
    several files are left in is a same-name hit.

    Examples:
        name "store/memory.go",    file app/store/memory.go -> 2 parts written
        name "memory.go",          file app/store/memory.go -> 1
        name "memory",             file app/store/memory.go -> not named
        name "pkg/run/main.py",    file bin/run             -> not named
        name "server.run",         file bin/run             -> not named

    Args:
        name_list: The names written in a request or in search terms
            (name_list_of()).

    Returns:
        (the sure hits, the same-name hits), each a list of knowledge paths grouped by
        file name: the groups in the index order of the first file in the index with
        that name, and the files of a group in index order. Both empty when
        config.PATH_WEIGHT is 0.
    """
    if config.PATH_WEIGHT <= 0:
        return [], []
    table = _file_table()
    # {the words of a file name: the nodes of its files written the furthest back}
    hit_node_dict: dict[tuple[str, ...], dict[_TailNode, None]] = {}
    for name_part_list in map(_part_word_list, name_list):
        base_key = tuple(name_part_list[-1]) if name_part_list else ()
        node = table.node_by_base_dict.get(base_key)
        if node is None:
            continue
        for word_list in reversed(name_part_list[:-1]):
            child = _tail_child_dict(node).get(tuple(word_list))
            if child is None:
                break
            node = child
        node_dict = hit_node_dict.setdefault(base_key, {})
        depth = next(iter(node_dict)).depth if node_dict else 0
        if node.depth > depth:
            node_dict.clear()
        if node.depth >= depth:
            node_dict[node] = None
    sure_list, same_name_list = [], []
    for base_key in sorted(hit_node_dict,
                           key=lambda key: table.node_by_base_dict[key].file_list[0].order):
        file_list = sorted((file for node in hit_node_dict[base_key]
                            for file in node.file_list), key=lambda file: file.order)
        path_list = [file.knowledge_path for file in file_list]
        (sure_list if len(path_list) == 1 else same_name_list).extend(path_list)
    return sure_list, same_name_list


def _pin_name_hit_list(rank_list: list[str], sure_list: list[str],
                       same_name_list: list[str], k: int) -> list[str]:
    """Put the files named in the query in front of the fused ranking (step 4).

    Args:
        rank_list: The ranking rrf_merge() returned, best first.
        sure_list: The sure hits of name_hit_list().
        same_name_list: The same-name hits of name_hit_list().
        k: How many files to keep.

    Returns:
        The sure hits in their order, then the same-name hits that are in rank_list in
        its order followed by the ones that are not in their order, cut to k // 2, then
        the rest of rank_list; each file once, at most k in all.
    """
    same_name_set = set(same_name_list)
    same_rank_list = [file_path for file_path in rank_list if file_path in same_name_set]
    same_rank_set = set(same_rank_list)
    same_rank_list += [file_path for file_path in same_name_list
                       if file_path not in same_rank_set]
    pin_list = sure_list + same_rank_list[:k // 2]
    if pin_list:
        logger.debug("files named in the query: %s", " ".join(pin_list))
    pin_set = set(pin_list)
    return (pin_list + [file_path for file_path in rank_list
                        if file_path not in pin_set])[:k]


def search_file_list(query: str, k: int | None = None) -> list[str]:
    """Search the file-summary index with BM25 (step 4).

    The file index holds the file summaries and the path words; it also finds files no
    definition matches.

    Args:
        query: English search terms, space separated.
        k: How many to take. None uses config.FILE_K.

    Returns:
        Knowledge paths, at most k of them, score descending.
    """
    k = config.FILE_K if k is None else k
    return [entry["key"] for entry in bm25_rank(query, INDEX, "file")][:k]


def rrf_merge(rank_list_a: list[str], rank_list_b: list[str],
              rrf_k: int | None = None, rrf_k_b: int | None = None) -> list[str]:
    """Fuse two rankings into one with Reciprocal Rank Fusion (step 4).

    They are fused by rank, not by score. Each ranking has a constant of its own.

    Args:
        rank_list_a: The first ranking, best first.
        rank_list_b: The second ranking.
        rrf_k: The constant for rank_list_a. None uses config.RRF_K. The smaller it is,
            the heavier that ranking's top entry weighs.
        rrf_k_b: The constant for rank_list_b. None uses config.RRF_K_FILE.

    Returns:
        The fused ranking. Entries appearing in both rise on their combined score.
    """
    rrf_k = config.RRF_K if rrf_k is None else rrf_k
    rrf_k_b = config.RRF_K_FILE if rrf_k_b is None else rrf_k_b
    score_dict = {}
    for rank_list, constant in ((rank_list_a, rrf_k), (rank_list_b, rrf_k_b)):
        for rank, key in enumerate(rank_list, 1):
            score_dict[key] = score_dict.get(key, 0.0) + 1.0 / (constant + rank)
    return [key for key, _ in sorted(score_dict.items(), key=lambda pair: -pair[1])]


def def_detail(def_key: str) -> dict:
    """Build a dict with the line range and the definition's first line (step 4).

    The line range and the first line come from the index; the file is not opened.

    Examples:
        "myapp/myapp/config/settings_py/settings.py::load_config"
            -> {"key": "myapp/config/settings.py::load_config",
                "lines": [412, 478],
                "signature": "def load_config(key, default=None):"}

    Args:
        def_key: A definition key from the index (knowledge path::definition name).

    Returns:
        {"key": repo-relative path::definition name, "lines": [start, end],
         "signature": the definition's first line}.
    """
    node_dict = INDEX.definition(def_key)
    return {"key": to_repo(def_key),
            "lines": node_dict["lines"],
            "signature": node_dict["signature"]}


def doc_is_current(knowledge_path: str) -> bool:
    """Say whether a file's design document describes the file as it is now.

    The sha256 of the file in PROJECT_DIR is compared with the one the design document
    was written from.

    Args:
        knowledge_path: The file's path as the index holds it.

    Returns:
        True when the hashes match. False when the file has no design document, the
        document carries no hash, or the file cannot be read or is not a regular file
        (tmpfile.check_file).
    """
    doc_hash = INDEX.file_doc_hash(knowledge_path)
    if not doc_hash:
        return False
    try:
        file_abs = resolve_in_project(knowledge_to_repo(knowledge_path, PROJECT_NAME))
        tmpfile.check_file(file_abs, sys.maxsize)
        return compute_file_hash(file_abs) == doc_hash
    except (OSError, ValueError) as e:
        logger.info("the design document of %s is not used: %s", knowledge_path, e)
        return False


def file_summary_dict(file_list: list[str]) -> dict[str, str]:
    """Take the summaries of the candidate files (step 4).

    Empty summaries, and those of files whose design document does not describe the
    file as it is now (doc_is_current()), are left out.

    Args:
        file_list: Knowledge paths of the candidate files.

    Returns:
        {repository-relative path: summary}, each cut to config.SUMMARY_MAX_CHAR with
        the "... (N chars omitted)" marker when it was longer.
    """
    summary_dict = {}
    for knowledge_path in file_list:
        summary = INDEX.file_summary(knowledge_path).strip()
        if summary and doc_is_current(knowledge_path):
            summary_dict[knowledge_to_repo(knowledge_path, PROJECT_NAME)] = \
                cut(summary, config.SUMMARY_MAX_CHAR)
    return summary_dict


# The shortest name outside ASCII that is looked for in a text
_MIN_TEXT_NAME_CHAR = 2
# What a word run of a request ends at, and a file name may hold
_SPACE = " "
# A letter, a digit or a "_" of ASCII; a run of them; and the place with none right
# behind it or behind a single "." behind it
_ASCII_WORD = re.compile(r"[A-Za-z0-9_]")
_ASCII_WORD_RUN = re.compile(r"[A-Za-z0-9_]+")
_NO_WORD_BEHIND = re.compile(rf"(?!{_ASCII_WORD.pattern}|\.{_ASCII_WORD.pattern})")


def text_name_list(text: str) -> list[str]:
    """Return the index's definition names outside ASCII that a text holds.

    A name is found wherever it stands in the text, the words around it joined to it
    or not, in either case.

    Examples:
        "widen café_total" -> ["café_total"]  # where the index defines café_total

    Args:
        text: A request or a task, as it was written.

    Returns:
        The names of at least _MIN_TEXT_NAME_CHAR characters the text holds, sorted.
        Empty when no index is loaded or the text is ASCII only.
    """
    if INDEX is None or text.isascii():
        return []
    upper_text = text.upper()
    return [name for name in INDEX.non_ascii_name_list()
            if len(name) >= _MIN_TEXT_NAME_CHAR and name.upper() in upper_text]


def is_index_name(name: str) -> bool:
    """Whether the index has a file whose last path part is a name, or a definition of it.

    Args:
        name: A name as it was written. Files are matched in either case.

    Returns:
        False when no index is loaded.
    """
    if INDEX is None:
        return False
    return name.lower() in _file_table().base_name_set \
        or bool(INDEX.def_list_by_name(name))


def text_file_name_list(text: str) -> list[str]:
    """Return the index's file names outside ASCII that a text holds, with their paths.

    A file whose last path part holds a character outside ASCII counts when that part
    stands anywhere in the text, the words around it joined to it or not, in either
    case. What comes back for it is the longest end of its repository-relative path,
    cut at a "/", that the text holds.

    Examples:
        "fix docs/résumé.md" -> ["docs/résumé.md"]  # where the index has docs/résumé.md

    Args:
        text: A request or search terms, as they were written.

    Returns:
        The names of at least _MIN_TEXT_NAME_CHAR characters, in index order. Empty when
        no index is loaded or the text is ASCII only.
    """
    if INDEX is None or text.isascii():
        return []
    upper_text = text.upper()
    name_list = []
    for file in _file_table().text_file_list:
        part_list = file.repo_path.split("/")
        for start in range(len(part_list)):
            name = "/".join(part_list[start:])
            if name.upper() in upper_text:
                if name not in name_list:
                    name_list.append(name)
                break
    return name_list


def _space_span(text: str, start: int, end: int,
                node: _SpaceNode) -> tuple[int, int, str]:
    """Return what a text writes of a file whose name with a space stands in it.

    The directories of a file written right before its name, each followed by a path
    separator, in either case and with no ASCII letter, digit or "_" right before
    them, are part of what is written.

    Args:
        text: A request or search terms, as they were written.
        start: Where the name starts in text.
        end: Where it ends.
        node: The node of the files named so.

    Returns:
        (where the directories written before the name start, end, the end of the
        repository-relative path written there, spelled as the index spells it): of the
        files of the node, the one whose written part starts the earliest in the text,
        and of several that start at the same place the first in index order.
    """
    best_start, best_node = start, node
    wait_list = [(start, node)]
    while wait_list:
        at, node = wait_list.pop()
        if (at, node.file_list[0].order) < (best_start, best_node.file_list[0].order):
            best_start, best_node = at, node
        if not at or text[at - 1] not in PATH_MARK_TUPLE:
            continue
        child_dict = _space_child_dict(node)
        for length in node.length_tuple:
            part_start = at - 1 - length
            if part_start < 0 or (part_start and _ASCII_WORD.match(text[part_start - 1])):
                continue
            child = child_dict.get((length, text[part_start:at - 1].lower()))
            if child is not None:
                wait_list.append((part_start, child))
    part_list = best_node.file_list[0].repo_path.split("/")
    return best_start, end, "/".join(part_list[-best_node.depth:])


def _space_name_span_list(text: str, table: _FileTable) -> list[tuple[int, int, str]]:
    """Return where a text writes the files of the index whose name holds a space.

    The name of such a file counts where it stands in the text in either case, with no
    ASCII letter, digit or "_" right before it, and none right behind it or behind a
    single "." behind it. The directories of the file written right before it, each
    followed by a path separator, are part of what is written (_space_span()). The
    text is gone through once: each run of ASCII letters, digits and "_" in it is
    looked up as the first such run of a name (space_mark_dict), and what stands
    where that name would is looked up as the name (space_node_dict). Of two places
    one name is written at that overlap, the first alone counts.

    Examples:
        "fix `docs/User Guide.md`." -> [(5, 23, "docs/User Guide.md")]
            # where the index has docs/User Guide.md
        "see User Guide.md.orig"     -> []

    Args:
        text: A request or search terms, as they were written.
        table: The table of the index's files.

    Returns:
        (start, end, the end of the file's repository-relative path written there,
        spelled as the index spells it), in the order written. Of two places one of
        which lies inside the other, the longer alone.
    """
    span_list = []
    # {a name in lower case: where the last place it is written at ends}
    end_by_name_dict: dict[str, int] = {}
    for word in _ASCII_WORD_RUN.finditer(text):
        for offset, length in table.space_mark_dict.get(word.group(0).lower(), ()):
            start = word.start() - offset
            end = start + length
            piece = text[start:end] if start >= 0 else ""
            if len(piece) < length or not piece.isascii():
                continue
            name = piece.lower()
            node = table.space_node_dict.get(name)
            if node is None or start < end_by_name_dict.get(name, 0) \
                    or (start and _ASCII_WORD.match(text[start - 1])) \
                    or not _NO_WORD_BEHIND.match(text, end):
                continue
            end_by_name_dict[name] = end
            span_list.append(_space_span(text, start, end, node))
    span_list.sort(key=lambda span: (span[0], -span[1]))
    keep_list: list[tuple[int, int, str]] = []
    for span in span_list:
        if not keep_list or span[1] > keep_list[-1][1]:
            keep_list.append(span)
    return keep_list


# What tells a bare word that stands alone from a word of a sentence: an underscore, a
# hyphen or a lower-to-upper case change
_WORD_MARK = re.compile(r"_|-|[a-z][A-Z]")
# What a file name is written with before its extension, and a path before each part
_EXTENSION_MARK = "."
# What a path that starts at the directory it is read from is written with
_HERE_MARK_TUPLE = tuple(_EXTENSION_MARK + mark for mark in PATH_MARK_TUPLE)


def _is_path_gap(gap: str) -> bool:
    """Whether what stands between two words holds a path separator."""
    return any(mark in gap for mark in PATH_MARK_TUPLE)


def _is_plain_path_gap(gap: str) -> bool:
    """Whether a gap is a path separator with nothing but separators and dots in it."""
    return _is_path_gap(gap) and all(char in PATH_MARK_TUPLE or char == _EXTENSION_MARK
                                     for char in gap)


def _is_name_start(run: WordRun, position: int) -> bool:
    """Whether a name may start with a word of a run.

    Args:
        run: A stretch of word_run_list().
        position: Which word of run.

    Returns:
        False when a single "." joins the word to the word before it and that word is
        no number, or to a word outside ASCII the run stands behind. True otherwise.
    """
    if run.gap_list[position] != _EXTENSION_MARK:
        return True
    if position == 0:
        return not run.after_text
    return run.word_list[position - 1].isdigit()


def _is_dot_start(run: WordRun, position: int) -> bool:
    """Whether a word stands right behind a "." no word of the run is joined to."""
    gap = run.gap_list[position]
    return gap.endswith(_EXTENSION_MARK) and (position == 0 or gap != _EXTENSION_MARK)


def _is_name_end(run: WordRun, position: int) -> bool:
    """Whether a name may end with a word of a run: no single "." joins it to the next."""
    return position + 1 == len(run.word_list) \
        or run.gap_list[position + 1] != _EXTENSION_MARK


def _is_name_shape(run: WordRun, start: int, end: int, lone_position: int) -> bool:
    """Whether the words of one path part are written the way a file is named.

    Args:
        run: A stretch of word_run_list().
        start: The first word of the part's words looked at.
        end: The last one.
        lone_position: Which word of run stands alone (_lone_position()), or -1.

    Returns:
        True when the last word is written as an extension (behind a "."). Without
        one: False where a plain path separator follows; True where one stands in
        front, behind a word of the run or starting with a ".", and where what stands
        in front ends with "./" or ".\\", whatever it starts with; and otherwise True
        only for the word that stands alone.
    """
    if end > start and run.gap_list[end].endswith(_EXTENSION_MARK):
        return True
    if _is_plain_path_gap(run.gap_list[end + 1]):
        return False
    gap = run.gap_list[start]
    if gap.endswith(_HERE_MARK_TUPLE):
        return True
    if _is_plain_path_gap(gap) and (start > 0 or gap.startswith(_EXTENSION_MARK)):
        return True
    return start == end == lone_position


def _lone_position(run: WordRun) -> int:
    """Return which word of a run stands alone.

    Args:
        run: A stretch of word_run_list().

    Returns:
        The position of the only word of the run that is no number, or -1 when there
        are several, none, or the run is bare text and the word has no _WORD_MARK.
    """
    position_list = [position for position, word in enumerate(run.word_list)
                     if not word.isdigit()]
    if len(position_list) != 1:
        return -1
    if not run.in_quote and not _WORD_MARK.search(run.word_list[position_list[0]]):
        return -1
    return position_list[0]


def _dot_node(table: _FileTable, base_key: tuple[str, ...]) -> _TailNode:
    """Return the node of the files of one name whose name starts with a ".".

    Args:
        table: The table of the index's files.
        base_key: The words of a last path part the table has files of.

    Returns:
        The node of those of table.node_by_base_dict[base_key] whose last path part
        starts with a ".", built on the first call for the key. It may hold no file.
    """
    if base_key not in table.dot_node_by_base_dict:
        table.dot_node_by_base_dict[base_key] = _TailNode(1, [
            file for file in table.node_by_base_dict[base_key].file_list
            if file.part_list[-1].startswith(_EXTENSION_MARK)])
    return table.dot_node_by_base_dict[base_key]


def _write_node_list(run: WordRun, word_list: list[str], part_start_list: list[int],
                     start: int, node: _TailNode) -> list[_TailNode]:
    """Return the nodes of the files a run writes the furthest back in front of a word.

    A directory counts when it is written right before what is already counted, with
    the words a file's path has there. The furthest one may start inside a part of
    the run, where a name may start (_is_name_start()).

    Args:
        run: A stretch of word_run_list().
        word_list: Its words in lower case.
        part_start_list: For each word, the first word of the path part it is in.
        start: The first word of the file's name in the run.
        node: The node of the files whose name the run holds from start.

    Returns:
        The nodes of the greatest depth the run writes: node itself when no directory
        of its files is written in front of the name.
    """
    node_list, open_node, part_end = [node], node, start
    while open_node is not None and part_end > 0 and _is_path_gap(run.gap_list[part_end]):
        part_start = part_start_list[part_end - 1]
        child_dict = _tail_child_dict(open_node)
        next_list, next_node, next_end = [], None, part_end
        for word_count in open_node.count_tuple:
            first = part_end - word_count
            if first < part_start:
                break
            child = child_dict.get(tuple(word_list[first:part_end]))
            if child is None or not _is_name_start(run, first):
                continue
            next_list.append(child)
            if first == part_start:
                next_node, next_end = child, first
        if not next_list:
            break
        node_list, open_node, part_end = next_list, next_node, next_end
    return node_list


def _run_name_list(run: WordRun, table: _FileTable) -> list[str]:
    """List the files of the index a word run names.

    Every stretch of the run's words that may be a file name is looked up: one that
    holds a letter, stays within a path part, starts where a name may start
    (_is_name_start()), ends where one may end (_is_name_end()) and is written the way
    a file is named (_is_name_shape()). A stretch right behind a "." names only a file
    whose name starts with a "." when it is one word or is none of these.

    Examples:
        "<b>src/b.py(new).</b>"   -> ["src/b.py"]     # where the index has src/b.py
        "a.md/b.md"               -> ["a.md", "b.md"] # where it has both
        "pkg/run/main.py"         -> ["pkg/run/main.py"]  # and not bin/run

    Args:
        run: A stretch of word_run_list().
        table: The table of the index's files.

    Returns:
        For each stretch the index has a file of, the end of that file's path the run
        writes (_write_node_list()), spelled as the index spells it; of several files
        with the name, the ones written the furthest back. In the order written.
    """
    word_list = [word.lower() for word in run.word_list]
    part_start_list: list[int] = []
    for position, gap in enumerate(run.gap_list[:-1]):
        part_start_list.append(position if position == 0 or _is_path_gap(gap)
                               else part_start_list[-1])
    lone_position = _lone_position(run)
    name_list = []
    for end in range(len(word_list)):
        if not _is_name_end(run, end):
            continue
        for word_count in table.word_count_tuple:
            start = end - word_count + 1
            if start < part_start_list[end]:
                break
            base_key = tuple(word_list[start:end + 1])
            node = table.node_by_base_dict.get(base_key)
            if node is None or not any(char.isalpha() for char in "".join(base_key)):
                continue
            is_plain = _is_name_start(run, start) \
                and _is_name_shape(run, start, end, lone_position)
            if _is_dot_start(run, start) and (start == end or not is_plain):
                node = _dot_node(table, base_key)
            elif not is_plain:
                continue
            node_list = _write_node_list(run, word_list, part_start_list, start, node)
            if len(node_list) == 1:
                name_list += _tail_name_list(node_list[0])
                continue
            # Several directories of one depth: the names in the order of their files
            file_list = sorted((file for node in node_list for file in node.file_list),
                               key=lambda file: file.order)
            name_list += ["/".join(file.part_list[-node_list[0].depth:])
                          for file in file_list]
    return name_list


def name_list_of(text: str) -> list[str]:
    """Return the names of the index's files that a text writes.

    Examples:
        "fix `src/a.py#L10` and b.py(new)." -> ["src/a.py", "b.py"]  # the index has both
        "fix new.py and https://x.y/a.py"   -> []                    # and no new.py

    Args:
        text: A request or search terms, as they were written.

    Returns:
        The files whose name holds a space, where the text writes them
        (_space_name_span_list()), then the files of the index each stretch of
        word_run_list() names in the rest of the text (_run_name_list()), then
        text_file_name_list(text), each once. Empty when no index is loaded.
    """
    if INDEX is None:
        return []
    table = _file_table()
    name_list = []
    rest_text = text
    if table.space_node_dict:
        # What is written of a name with a space in it is read as that name alone
        rest_list, position = [], 0
        for start, end, name in _space_name_span_list(text, table):
            name_list.append(name)
            rest_list.append(text[position:start])
            position = end
        rest_text = _SPACE.join(rest_list + [text[position:]])
    for run in word_run_list(rest_text):
        name_list += _run_name_list(run, table)
    return list(dict.fromkeys(name_list + text_file_name_list(text)))


def with_text_name(query: str, request: str) -> str:
    """Append to search terms the names outside ASCII the request holds and they lack.

    Args:
        query: The search terms step 1 returned.
        request: The change request as the user wrote it.

    Returns:
        query followed by text_name_list(request) minus the names already among its
        words, compared case-insensitively.
    """
    word_set = {word.upper() for word in query.split()}
    add_list = [name for name in text_name_list(request) if name.upper() not in word_set]
    if add_list:
        logger.debug("names of the index written in the request are kept: %s",
                     " ".join(add_list))
    return " ".join([query] + add_list)


def _file_rank_list(score_dict: dict[str, float], file_by_def_dict: dict[str, str],
                    query: str, name_list: list[str], file_k: int) -> list[str]:
    """Rank the candidate files (step 4).

    Steps:
    1. Group the scored definitions by file; a file takes its highest score, and the
       test files go behind every file that is not one
    2. Fuse that ranking with the file index's via RRF, which also brings in files no
       definition matched
    3. Put the files named in name_list in front and cut to file_k

    Args:
        score_dict: {definition key: score}, as expand() returned it.
        file_by_def_dict: {definition key: knowledge path} for every key of score_dict.
        query: English search terms, space separated.
        name_list: The file names to put in front, as name_list_of() returns them.
        file_k: How many files to keep.

    Returns:
        Knowledge paths, best first, at most file_k of them.
    """
    file_score_dict = {}
    for key, score in score_dict.items():
        file_path = file_by_def_dict[key]
        file_score_dict[file_path] = max(file_score_dict.get(file_path, 0.0), score)
    file_list = [file_path for file_path, _ in sorted(file_score_dict.items(),
                                                      key=lambda pair: -pair[1])]
    file_list.sort(key=_is_test_file)

    file_list = rrf_merge(file_list, search_file_list(query))
    return _pin_name_hit_list(file_list, *name_hit_list(name_list), file_k)


def search_scope(request: str, llm: Callable[[str], str] | None = None) -> dict:
    """Turn a change request into candidate files and definitions (search steps 1-4).

    Steps:
    1. Rewrite the request into English search terms (step 1), and add the index's
       names outside ASCII the request holds
    2. Search definitions with BM25 (step 2)
    3. Expand along the calls edges (step 3)
    4. Group by file, separate the callers and attach a confidence (step 4). The files
       put in front are the ones whose name is among the names name_list_of() takes
       from the request

    Args:
        request: The change request as the user wrote it, in any language.
        llm: A callable taking a string and returning a string. When it is None and
            need_expand() says an expansion is needed, one is built with make_llm().
            Omitting it never skips the query expansion.

    Returns:
        The dict search_scope_from_query() returns (see there for the keys), with
        trace's "request" set to request and "expand_failed" to whether the query
        expansion failed, and "note" made again by scope_note().

    Raises:
        ValueError: When request is empty or whitespace only.
    """
    if not request or not request.strip():
        raise ValueError("empty request. Describe the change in --request")

    if llm is None and need_expand(request):
        llm = make_llm(config.QUERY_EXPAND_MODEL, config.QUERY_EXPAND_MAX_TOKEN,
                       "QUERY_EXPAND_MODEL", think_level="off")
    query, is_ok = expand_query(request, llm, is_index_name)
    scope_dict = search_scope_from_query(with_text_name(query, request),
                                         name_list=name_list_of(request))
    scope_dict["trace"]["request"] = request
    scope_dict["trace"]["expand_failed"] = not is_ok
    scope_dict["note"] = scope_note(scope_dict)
    return scope_dict


def search_scope_from_query(query: str, def_k: int | None = None,
                            file_k: int | None = None,
                            name_list: list[str] | None = None) -> dict:
    """Build the scope from English search terms (search steps 2-4).

    Steps:
    1. Search definitions with BM25 (step 2)
    2. Expand along the calls edges (step 3)
    3. Group by file, fuse with the file index's ranking via RRF and put the files
       named in name_list in front (step 4)
    4. Separate the callers and attach confidence and spread (step 4)

    Args:
        query: English search terms, space separated.
        def_k: How many definitions BM25 keeps. None uses config.DEF_K.
        file_k: How many files to keep in files. None uses config.SCOPE_FILE_K.
        name_list: The file names to put in front, as name_list_of() returns them.
            None uses name_list_of(query).

    Returns:
        {"files": list[str],   # the files to read (rank descending, at most file_k)
         "defs": list[dict],   # {"key", "lines", "signature"} (at most def_k).
                               # Not cut by files: files not in "files" appear here
         "expanded": list[dict],  # {"key", "lines", "signature", "from"} for the
                               # definitions reached from those along the calls edges
                               # (the top def_k by score)
         "file_summaries": dict,  # {repository-relative path: file summary}
         "callers": dict,      # {definition key: the definition keys calling it,
                               # at most def_k each}
         "confidence": str,    # "high" or "low"
         "spread": int,        # how many files the entry-point definitions scattered over
         "note": str,          # the line to read when the search found nothing or
                               # ran on the request untranslated, or ""
         "trace": dict}        # request / query / expanded / empty / expand_failed /
                               # top_score / top_raw / top_hit_terms. "request" is empty
    """
    def_k = config.DEF_K if def_k is None else def_k
    file_k = config.SCOPE_FILE_K if file_k is None else file_k

    logger.info("query: %s", query)
    entry_list = search_def_list(query, k=def_k)
    if not entry_list:
        logger.info("no match: %s", query)
    score_dict, from_dict = expand(entry_list)
    # "expanded nothing" is for when there were entry points but nothing grew from them
    if entry_list and len(score_dict) == len(entry_list):
        logger.debug("expanded nothing")

    file_by_def_dict = INDEX.file_by_def_dict(score_dict)
    file_list = _file_rank_list(score_dict, file_by_def_dict, query,
                                name_list_of(query) if name_list is None else name_list,
                                file_k)

    entry_key_set = {entry["key"] for entry in entry_list}
    caller_dict = {key: INDEX.caller_list(key) for key in entry_key_set}

    # confidence is decided from the pre-normalisation raw and hit_terms of the top
    # definition. The normalised score and the file index's scores are not used
    top = entry_list[0] if entry_list else None
    is_high = (top is not None
               and top["hit_terms"] >= config.CONFIDENCE_MIN_TERM
               and top["raw"] >= config.CONFIDENCE_MIN_SCORE)
    scope_dict = {
        # Everything past this point is a repository-relative path
        "files": [knowledge_to_repo(file_path, PROJECT_NAME) for file_path in file_list],
        "defs": [def_detail(entry["key"]) for entry in entry_list],
        # Both lists are bounded by def_k: the reached definitions by score, and each
        # entry point's callers sorted by key
        "expanded": [dict(def_detail(key), **{"from": to_repo(from_dict[key])})
                     for key in sorted(from_dict, key=score_dict.get, reverse=True)[:def_k]],
        "file_summaries": file_summary_dict(file_list),
        "callers": {to_repo(key): [to_repo(caller_key) for caller_key in caller_list[:def_k]]
                    for key, caller_list in caller_dict.items()},
        "confidence": "high" if is_high else "low",
        # How many files the entry-point definitions scattered over
        "spread": len({file_by_def_dict[key] for key in entry_key_set}),
        "trace": {"request": "", "query": query,
                  "expanded": sorted(to_repo(key) for key in set(score_dict) - entry_key_set),
                  "empty": not entry_list and not file_list,
                  "expand_failed": False,
                  "top_score": top["score"] if top else 0.0,
                  "top_raw": top["raw"] if top else 0.0,
                  "top_hit_terms": top["hit_terms"] if top else 0},
    }
    scope_dict["note"] = scope_note(scope_dict)
    return scope_dict
