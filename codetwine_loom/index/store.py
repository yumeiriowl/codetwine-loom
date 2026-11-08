"""Index step 5: read and write code_index.sqlite.

The index is written as tables and read one query at a time. Index() reads the call
edges and the settings when it is made.

The BM25 postings keep integer IDs. save() assigns them in the order the definition
nodes and the file nodes were built, and Index.key_dict() turns them back into keys.
"""

import functools
import hashlib
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import urllib.parse
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone

from codetwine_loom import config, tmpfile
from codetwine_loom.index import metaread
from codetwine_loom.languages import (ALIAS_DICT, HEADER_OWNER_TUPLE, LANG_DICT,
                                      STOPWORD_BASE_SET, header_owner, is_dep_file,
                                      lang_ext, lang_of, project_file_pattern_tuple,
                                      split_def_key)
from codetwine_loom.paths import knowledge_to_repo

logger = logging.getLogger(__name__)

# The index file inside <output_dir>/<project name>/
INDEX_FILE = "code_index.sqlite"

# The version of the table layout below and of the meta rows an index is read with
_SCHEMA_VERSION = "9"

# How many values one "IN (...)" statement is given at a time
_MAX_SQL_VARIABLE = 500

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE defs (
    id          INTEGER PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,   -- "<file path>::<definition name>"
    name        TEXT NOT NULL,
    fold_name   TEXT,                   -- the name in upper case where the language
                                        -- ignores case, NULL elsewhere
    type        TEXT,
    file        TEXT NOT NULL,
    start_line  INTEGER,
    end_line    INTEGER,
    signature   TEXT,
    signature_line INTEGER,             -- the line the signature is on
    is_constant INTEGER NOT NULL,       -- 1 for a definition the search leaves out
    len         INTEGER NOT NULL        -- BM25 token count
);
CREATE INDEX idx_defs_name ON defs(name);
CREATE INDEX idx_defs_fold_name ON defs(fold_name);

CREATE TABLE files (
    id      INTEGER PRIMARY KEY,
    path    TEXT NOT NULL UNIQUE,
    summary  TEXT,
    doc_hash TEXT NOT NULL,             -- sha256 of the source the design document describes, or ""
    len      INTEGER NOT NULL
);

CREATE TABLE calls (
    src  TEXT NOT NULL,                 -- the calling definition key
    dst  TEXT NOT NULL,                 -- the called definition key
    PRIMARY KEY (src, dst)
);
CREATE INDEX idx_calls_dst ON calls(dst);

CREATE TABLE postings (
    kind TEXT NOT NULL,                 -- "def" or "file"
    term TEXT NOT NULL,
    id   INTEGER NOT NULL,
    cnt  INTEGER NOT NULL,
    PRIMARY KEY (kind, term, id)
) WITHOUT ROWID;
"""


def _lang_ext_of_def(def_key: str) -> str:
    """Take the language of the file out of a "<file path>::<definition name>" key.

    Examples:
        "p/settings_py/settings.py::get" -> "py"

    Args:
        def_key: A definition key from the index.

    Returns:
        languages.lang_ext() of the file: a LANG_DICT key, or the file's lower-case
        extension, or "" when it has none.
    """
    return lang_ext(split_def_key(def_key)[0])


def is_constant(def_key: str, node_dict: dict) -> bool:
    """Say whether a definition counts as a constant, per its own language.

    type, a tree-sitter node name that differs per language, is looked up through
    lang_of(). A definition of the language's record_type counts unless its node carries
    a true "record".

    Args:
        def_key: The definition key, which starts with the file path.
        node_dict: That definition's node (a dict with "type", and "record" where
            mark_record() set it).

    Returns:
        True when the type is in that language's exclude_type_set, or is its
        record_type and the definition is not a record.
    """
    lang_config = lang_of(split_def_key(def_key)[0])
    if lang_config is None:
        return False
    if lang_config.record_type and node_dict["type"] == lang_config.record_type:
        return not node_dict.get("record", False)
    return node_dict["type"] in lang_config.exclude_type_set


def primary_ext(def_dict: dict, file_list: Iterable[str] = (), project_name: str = "") -> str:
    """Return the extension of the language most files carry a non-constant definition in.

    One index can hold several languages; this settles which one is the primary
    language. Files rather than definitions are counted, and the definitions
    is_constant() says are constants are left out. The extensions of one language
    (ALIAS_DICT) are counted together, and of that language the extension the most of
    those files carry is returned. A header extension counts as the language
    header_owner() says it stands for. When a language counted has a file of its
    projects at the repository root (_root_project_lang_set), a language with
    need_root_project that has none there is left out while another language is
    counted.

    Examples:
        {"cbl": 3 files, "cpy": 5 files, "sql": 4 files} -> "cpy"
        {"c": 4 files, "h": 9 files}                     -> "c"

    Where no definition is left once the constants are left out, the files of
    file_list whose language is a LANG_DICT key are counted the same way.

    Args:
        def_dict: The definition nodes after steps 1-3.
        file_list: The paths of the files the index holds.
        project_name: The directory name of PROJECT_DIR, which the knowledge paths of
            file_list start with.

    Returns:
        An extension such as "py", or "" when there is neither such a definition nor
        such a file.
    """
    file_list = list(file_list)
    file_set_dict: dict[str, set] = {}
    for key, node_dict in def_dict.items():
        if is_constant(key, node_dict):
            continue
        file_set_dict.setdefault(_lang_ext_of_def(key), set()).add(node_dict["file"])
    ext_count_dict = Counter({ext: len(file_set)
                              for ext, file_set in file_set_dict.items()})
    if not ext_count_dict:
        ext_count_dict = Counter(ext for ext in map(lang_ext, file_list)
                                 if ext in LANG_DICT)
    if not ext_count_dict:
        return ""
    lang_count_dict: Counter = Counter()
    own_count_dict: Counter = Counter()
    for ext, count in ext_count_dict.items():
        own_ext = header_owner(ext, ext_count_dict)
        lang_count_dict[ALIAS_DICT.get(own_ext, own_ext)] += count
        own_count_dict[own_ext] += count
    root_lang_set = _root_project_lang_set(lang_count_dict, file_list, project_name)
    keep_count_dict = Counter({lang: count for lang, count in lang_count_dict.items()
                               if lang in root_lang_set or not _need_root_project(lang)}
                              ) if root_lang_set else Counter()
    lang = (keep_count_dict or lang_count_dict).most_common(1)[0][0]
    return max((ext for ext in own_count_dict if ALIAS_DICT.get(ext, ext) == lang),
               key=lambda ext: own_count_dict[ext])


def _need_root_project(lang: str) -> bool:
    """Say whether a language is passed over without a project file at the root.

    Args:
        lang: A LANG_DICT key or another extension.

    Returns:
        The language's need_root_project, False for an extension not in LANG_DICT.
    """
    return lang in LANG_DICT and LANG_DICT[lang].need_root_project


def _root_project_lang_set(lang_iter: Iterable[str], file_list: list[str],
                           project_name: str) -> set[str]:
    """Return the languages that have a file of their projects at the repository root.

    Args:
        lang_iter: LANG_DICT keys.
        file_list: The paths of the files the index holds, as knowledge paths or as
            repository-relative paths.
        project_name: The directory name of PROJECT_DIR.

    Returns:
        The keys of lang_iter one of whose project_file_pattern_tuple() patterns
        matches a file at the root.
    """
    root_list = []
    for path in file_list:
        try:
            path = knowledge_to_repo(path, project_name)
        except ValueError:
            pass
        if "/" not in path:
            root_list.append(path)
    return {lang for lang in lang_iter if lang in LANG_DICT and any(
        is_dep_file(path, project_file_pattern_tuple(LANG_DICT[lang])) for path in root_list)}


def _build_recipe() -> str:
    """Return the value standing for the rules an index is built under now.

    The value covers what the settings and the tables say now: how heavily a path
    counts, which words are dropped, how an identifier is split, which languages fold a
    name's case, and the rules the primary language is chosen by.

    Returns:
        A sha256 hex string of those rules.
    """
    from codetwine_loom.index import indexer

    part_list = [repr(config.PATH_WEIGHT), repr(indexer.MIN_TOKEN_CHAR),
                 indexer.WORD.pattern, indexer.CAMEL.pattern, indexer.JOIN_WORD.pattern,
                 repr(sorted(STOPWORD_BASE_SET)), repr(HEADER_OWNER_TUPLE),
                 repr(sorted(ALIAS_DICT.items()))]
    for ext, lang_config in sorted(LANG_DICT.items()):
        part_list += [ext, repr(sorted(lang_config.stopword_set)),
                      repr(sorted(lang_config.exclude_type_set)),
                      repr(lang_config.hyphen_name), lang_config.record_type,
                      repr(lang_config.ignore_case),
                      repr(project_file_pattern_tuple(lang_config)),
                      repr(lang_config.need_root_project)]
    return hashlib.sha256("\n".join(part_list).encode()).hexdigest()


# == writing ================================================================

def _meta_row_list(index_dict: dict, def_dict: dict, source_stamp: str,
                   project_name: str, commit: str) -> list[tuple[str, str]]:
    """Assemble the meta table's rows.

    Args:
        index_dict: build_index()'s return value.
        def_dict: The definition nodes of steps 1-3.
        source_stamp: What identifies the analysis the index is built from.
        project_name: The directory name of PROJECT_DIR.
        commit: The commit the analysis this index is built from describes.

    Returns:
        (key, value) pairs. Numbers and lists are stored as text.
    """
    return [
        ("schema_version", _SCHEMA_VERSION),
        ("project", project_name),
        ("built_at", datetime.now(timezone.utc).isoformat()),
        ("source_stamp", source_stamp),
        # Which state of the repository this index describes
        ("commit", commit),
        # What the postings were built under
        ("recipe", _build_recipe()),
        ("primary_ext", primary_ext(def_dict, index_dict["nodes"]["files"], project_name)),
        # {knowledge path: extension} of the files the analysis read in a language
        # their extension does not say
        ("file_lang", json.dumps(index_dict.get("file_lang", {}), sort_keys=True)),
        ("def_avg_len", repr(index_dict["index"]["def"]["avg_len"])),
        # The same average over the definitions that are not constants
        ("def_avg_len_no_constant",
         repr(_avg_len_no_constant(index_dict["index"]["def"]["len"], def_dict))),
        ("file_avg_len", repr(index_dict["index"]["file"]["avg_len"])),
    ]


def _avg_len_no_constant(len_dict: dict, def_dict: dict) -> float:
    """Return the average document length over the definitions that are not constants.

    Args:
        len_dict: {definition key: its token count}, from build_bm25().
        def_dict: The definition nodes, for telling which are constants.

    Returns:
        The average, or 0.0 when every definition is one.
    """
    length_list = [length for key, length in len_dict.items()
                   if key in def_dict and not is_constant(key, def_dict[key])]
    return sum(length_list) / len(length_list) if length_list else 0.0


def _fold_name(def_key: str, name: str) -> str | None:
    """Return what the fold_name column holds for one definition.

    Args:
        def_key: The definition key, which starts with the file path.
        name: The definition's name.

    Returns:
        The name in upper case where the file's language ignores case, else None.
    """
    lang_config = lang_of(split_def_key(def_key)[0])
    return name.upper() if lang_config is not None and lang_config.ignore_case else None


def _iter_posting_row(bm25_dict: dict, id_dict: dict, kind: str) -> Iterator[tuple]:
    """Yield the postings table's rows for one BM25 index.

    Args:
        bm25_dict: One build_bm25() result, its postings still keyed by key string.
        id_dict: {key: integer ID}.
        kind: "def" or "file".

    Yields:
        (kind, term, ID, occurrences) tuples.
    """
    for term, posting_list in bm25_dict["postings"].items():
        for key, count in posting_list:
            yield (kind, term, id_dict[key], count)


def index_db_path(output_dir: str, project_name: str) -> str:
    """Return the path to the index database of one repository in an output directory.

    Args:
        output_dir: Where codetwine writes its output.
        project_name: The directory name the analysis is filed under.

    Returns:
        INDEX_FILE joined to output_dir/project_name; relative when output_dir is.
    """
    return os.path.join(output_dir, project_name, INDEX_FILE)


def save(path: str, index_dict: dict, def_dict: dict, source_stamp: str,
         project_name: str, commit: str) -> None:
    """Write the index out as tables (step 5).

    Any index already at that path is replaced. It goes through tmpfile.atomic_write():
    a build that stops mid-write leaves the previous index intact, and its temporary
    file is left to the clean-up atomic_write() does.

    The definition nodes are written without their "doc" field. is_constant is settled
    here and stored with each definition.

    Args:
        path: Where to write.
        index_dict: The dict holding the results of steps 1-4 (build_index()'s return
            value).
        def_dict: The definition nodes of steps 1-3. Their order is the definition index's
            integer ID.
        source_stamp: What identifies the analysis run the index is built from
            (Knowledge.stamp()).
        project_name: The directory name of PROJECT_DIR.
        commit: The commit the analysis this index is built from describes, which is
            not the same as what is checked out now when the analysis was reused rather
            than made. "" says it cannot be told.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_dict = index_dict["nodes"]["files"]
    def_id_dict = {key: i for i, key in enumerate(def_dict)}
    file_id_dict = {file_path: i for i, file_path in enumerate(file_dict)}
    def_len_dict = index_dict["index"]["def"]["len"]
    file_len_dict = index_dict["index"]["file"]["len"]

    # The connection is closed inside the block, before atomic_write() moves the
    # database onto path
    with tmpfile.atomic_write(path) as tmp_path:
        connection = sqlite3.connect(tmp_path)
        try:
            connection.executescript(_SCHEMA)
            connection.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                                   _meta_row_list(index_dict, def_dict, source_stamp,
                                                  project_name, commit))
            connection.executemany(
                "INSERT INTO defs (id, key, name, fold_name, type, file, start_line, "
                "end_line, signature, signature_line, is_constant, len) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(def_id_dict[key], key, node_dict["name"], _fold_name(key, node_dict["name"]),
                  node_dict["type"], node_dict["file"],
                  node_dict["lines"][0], node_dict["lines"][1], node_dict["signature"],
                  node_dict.get("signature_line", node_dict["lines"][0]),
                  int(is_constant(key, node_dict)), def_len_dict.get(key, 0))
                 for key, node_dict in def_dict.items()])
            connection.executemany(
                "INSERT INTO files (id, path, summary, doc_hash, len) VALUES (?, ?, ?, ?, ?)",
                [(file_id_dict[file_path], file_path, node_dict["summary"],
                  node_dict.get("doc_hash", ""), file_len_dict.get(file_path, 0))
                 for file_path, node_dict in file_dict.items()])
            connection.executemany("INSERT OR IGNORE INTO calls (src, dst) VALUES (?, ?)",
                                   index_dict["edges"]["calls"])
            connection.executemany(
                "INSERT INTO postings (kind, term, id, cnt) VALUES (?, ?, ?, ?)",
                _iter_posting_row(index_dict["index"]["def"], def_id_dict, "def"))
            connection.executemany(
                "INSERT INTO postings (kind, term, id, cnt) VALUES (?, ?, ?, ?)",
                _iter_posting_row(index_dict["index"]["file"], file_id_dict, "file"))
            connection.commit()
        finally:
            connection.close()


# == reading ================================================================

def _file_lang_dict(value: object) -> dict[str, str]:
    """Read the languages of the files as an index records them.

    Args:
        value: What the meta table holds under file_lang.

    Returns:
        {file: language}.

    Raises:
        sqlite3.DatabaseError: When value is not the text of a JSON object.
    """
    try:
        file_lang_dict = json.loads(value) if isinstance(value, str) else None
    except (ValueError, RecursionError):
        file_lang_dict = None
    if not isinstance(file_lang_dict, dict):
        raise sqlite3.DatabaseError("the index holds no JSON object under file_lang")
    return file_lang_dict


class Index:
    """One built index, read one query at a time.

    The call edges and the settings are read when it is made; the definitions, the file
    summaries and the postings stay in the database and are queried on each call.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Read the meta rows and the call edges, and keep the connection for the rest.

        Args:
            connection: A read-only connection to a built index.

        Raises:
            sqlite3.DatabaseError: When the tables lack what is read here, or
                file_lang is not a JSON object (_file_lang_dict).
        """
        self._connection = connection
        self._meta_dict = {row["key"]: row["value"]
                           for row in connection.execute("SELECT key, value FROM meta")}
        self.project = self._meta_dict.get("project", "")
        self.source_stamp = self._meta_dict.get("source_stamp", "")
        self.commit = self._meta_dict.get("commit", "")
        self.primary_ext = self._meta_dict.get("primary_ext", "")
        self.file_lang_dict = _file_lang_dict(self._meta_dict.get("file_lang", "{}"))
        self.schema_version = self._meta_dict.get("schema_version", "")
        self.recipe = self._meta_dict.get("recipe", "")
        self.call_list = [(row["src"], row["dst"]) for row in connection.execute(
            "SELECT src, dst FROM calls ORDER BY src, dst")]
        self.indegree_dict: dict[str, int] = {}
        for _, dst in self.call_list:
            self.indegree_dict[dst] = self.indegree_dict.get(dst, 0) + 1
        self._doc_count_dict: dict[tuple[str, bool], int] = {}

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    # -- BM25 ---------------------------------------------------------------

    def avg_len(self, kind: str) -> float:
        """Return the average document length over the documents of one kind.

        The "def" average leaves out the definitions that are constants unless
        config.INCLUDE_CONSTANT is on.

        Args:
            kind: "def" or "file".

        Returns:
            The average token count.
        """
        if kind == "def" and not config.INCLUDE_CONSTANT:
            no_constant = self._meta_dict.get("def_avg_len_no_constant")
            if no_constant is not None:
                return float(no_constant)
        return float(self._meta_dict.get(f"{kind}_avg_len", 0.0))

    def doc_count(self, kind: str) -> int:
        """Return how many documents of one kind there are.

        Definitions the language calls constants are left out unless
        config.INCLUDE_CONSTANT is on.

        Args:
            kind: "def" or "file".

        Returns:
            The number of documents.
        """
        cache_key = (kind, config.INCLUDE_CONSTANT)
        count_cache = self._doc_count_dict.get(cache_key)
        if count_cache is not None:
            return count_cache
        if kind == "file":
            sql = "SELECT COUNT(*) FROM files"
        elif config.INCLUDE_CONSTANT:
            sql = "SELECT COUNT(*) FROM defs"
        else:
            sql = "SELECT COUNT(*) FROM defs WHERE is_constant = 0"
        count = self._connection.execute(sql).fetchone()[0]
        self._doc_count_dict[cache_key] = count
        return count

    def posting_list(self, kind: str, term: str) -> list[tuple[int, int, int]]:
        """Return one term's postings, ID ascending, with each document's length.

        Args:
            kind: "def" or "file".
            term: The search term.

        Returns:
            [(ID, occurrences, that document's token count), ...].
        """
        if kind == "file":
            sql = ("SELECT p.id, p.cnt, f.len FROM postings p "
                   "JOIN files f ON f.id = p.id "
                   "WHERE p.kind = 'file' AND p.term = ? ORDER BY p.id")
        elif config.INCLUDE_CONSTANT:
            sql = ("SELECT p.id, p.cnt, d.len FROM postings p "
                   "JOIN defs d ON d.id = p.id "
                   "WHERE p.kind = 'def' AND p.term = ? ORDER BY p.id")
        else:
            sql = ("SELECT p.id, p.cnt, d.len FROM postings p "
                   "JOIN defs d ON d.id = p.id "
                   "WHERE p.kind = 'def' AND p.term = ? AND d.is_constant = 0 "
                   "ORDER BY p.id")
        return [(row[0], row[1], row[2])
                for row in self._connection.execute(sql, (term,))]

    def has_term(self, kind: str, term: str) -> bool:
        """Say whether a term is in that index at all.

        Args:
            kind: "def" or "file".
            term: The search term.

        Returns:
            True when at least one document carries it.
        """
        row = self._connection.execute(
            "SELECT 1 FROM postings WHERE kind = ? AND term = ? LIMIT 1",
            (kind, term)).fetchone()
        return row is not None

    def key_dict(self, kind: str, id_list: list[int]) -> dict[int, str]:
        """Turn integer IDs back into keys.

        The IDs are asked for in blocks of _MAX_SQL_VARIABLE.

        Args:
            kind: "def" or "file".
            id_list: The IDs to look up.

        Returns:
            {ID: definition key or file path}, missing IDs left out.
        """
        column, table = ("key", "defs") if kind == "def" else ("path", "files")
        return {row[0]: row[1] for row in self._in_row_list(
            f"SELECT id, {column} FROM {table} WHERE id", list(id_list))}

    def _in_row_list(self, select_text: str, value_list: list) -> list[sqlite3.Row]:
        """Run "<select_text> IN (...)" over values, in blocks of _MAX_SQL_VARIABLE.

        Args:
            select_text: The statement up to the column the values are matched with,
                such as "SELECT key, file FROM defs WHERE key".
            value_list: The values to match.

        Returns:
            The rows of every block, in the order the blocks ran.
        """
        row_list = []
        for start in range(0, len(value_list), _MAX_SQL_VARIABLE):
            block_list = value_list[start:start + _MAX_SQL_VARIABLE]
            mark_text = ",".join("?" * len(block_list))
            row_list += self._connection.execute(
                f"{select_text} IN ({mark_text})", block_list).fetchall()
        return row_list

    # -- nodes and edges ----------------------------------------------------

    def definition(self, def_key: str) -> dict | None:
        """Return one definition node.

        Args:
            def_key: A definition key from the index.

        Returns:
            {"name", "type", "file", "lines", "signature", "signature_line"}, or None
            when the index holds no such definition.
        """
        row = self._connection.execute(
            "SELECT name, type, file, start_line, end_line, signature, signature_line "
            "FROM defs WHERE key = ?", (def_key,)).fetchone()
        return None if row is None else {
            "name": row["name"], "type": row["type"], "file": row["file"],
            "lines": [row["start_line"], row["end_line"]],
            "signature": row["signature"], "signature_line": row["signature_line"]}

    def def_list_by_name(self, name: str) -> list[dict]:
        """Return every definition of a given name.

        A definition of a language that ignores case matches the name in either case.

        Args:
            name: The definition name, without any "#<type>#<start line>" suffix.

        Returns:
            [{"key", "file"}, ...] in key order.
        """
        row_iter = self._connection.execute(
            "SELECT key, file FROM defs WHERE name = ? OR fold_name = ? ORDER BY key",
            (name, name.upper()))
        return [{"key": row["key"], "file": row["file"]} for row in row_iter]

    def non_ascii_name_list(self) -> list[str]:
        """Return the definition names that hold a character outside ASCII.

        Returns:
            The distinct names, sorted.
        """
        return [row["name"] for row in self._connection.execute(
            "SELECT DISTINCT name FROM defs WHERE name GLOB '*[^ -~]*' ORDER BY name")]

    def file_by_def_dict(self, def_key_iter: Iterable[str]) -> dict[str, str]:
        """Return which file each definition is in.

        The keys are asked for in blocks of _MAX_SQL_VARIABLE.

        Args:
            def_key_iter: The definition keys to look up.

        Returns:
            {definition key: file path}, missing keys left out.
        """
        return {row["key"]: row["file"] for row in self._in_row_list(
            "SELECT key, file FROM defs WHERE key", list(def_key_iter))}

    def file_summary(self, path: str) -> str:
        """Return one file's summary.

        Args:
            path: A file path as the index holds it.

        Returns:
            The summary, empty when the file has none or is not in the index.
        """
        row = self._connection.execute(
            "SELECT summary FROM files WHERE path = ?", (path,)).fetchone()
        return (row["summary"] or "") if row else ""

    def file_doc_hash(self, path: str) -> str:
        """Return the sha256 of the source one file's design document describes.

        Args:
            path: A file path as the index holds it.

        Returns:
            The hex digest, empty when the file has no design document, the document
            carries no hash, or the file is not in the index.
        """
        row = self._connection.execute(
            "SELECT doc_hash FROM files WHERE path = ?", (path,)).fetchone()
        return (row["doc_hash"] or "") if row else ""

    def has_summary(self) -> bool:
        """Say whether any file in the index carries a summary.

        Returns:
            True when at least one file's summary is not empty.
        """
        return self._connection.execute(
            "SELECT 1 FROM files WHERE summary != '' LIMIT 1").fetchone() is not None

    def file_path_list(self) -> list[str]:
        """Return every file path in the index, in the order the index was built in.

        Returns:
            The file paths.
        """
        return [row["path"] for row in
                self._connection.execute("SELECT path FROM files ORDER BY id")]

    def caller_list(self, def_key: str) -> list[str]:
        """Return the definitions that call one definition.

        Args:
            def_key: A definition key from the index.

        Returns:
            The calling definitions' keys, sorted.
        """
        row_iter = self._connection.execute(
            "SELECT src FROM calls WHERE dst = ? ORDER BY src", (def_key,))
        return [row["src"] for row in row_iter]


def read_only_uri(path: str, alone: bool = False) -> str:
    """Return the URI that opens one SQLite file for reading alone.

    The path is made absolute and escaped, and the URI names no host: a "#", a "?" or
    a "%" in the path stays part of the file name, and so do the slashes it starts
    with.

    Examples:
        "/out/app/x.sqlite"        -> "file:///out/app/x.sqlite?mode=ro"
        "//out/C#/x.sqlite"        -> "file:////out/C%23/x.sqlite?mode=ro"
        ("/out/app/x.sqlite", True) -> "file:///out/app/x.sqlite?mode=ro&immutable=1"

    Args:
        path: The path to the database file, absolute or relative to the working
            directory.
        alone: Whether the file is read as one nothing writes while it is open:
            SQLite takes no lock on it and opens none of the files beside it (its
            journal, its write-ahead log and the index of that log).

    Returns:
        The URI to pass to sqlite3.connect() with uri=True.
    """
    path_abs = os.path.abspath(path)
    return (f"file://{urllib.parse.quote(os.fsencode(path_abs))}?mode=ro"
            + ("&immutable=1" if alone else ""))


def open_read_only(path: str) -> sqlite3.Connection:
    """Open a SQLite file for reading alone.

    Args:
        path: The path to the database file.

    Returns:
        A read-only connection whose rows come back as sqlite3.Row.
    """
    connection = sqlite3.connect(read_only_uri(path), uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load(path: str) -> Index:
    """Open a built index for reading (step 5).

    Args:
        path: The path to code_index.sqlite.

    Returns:
        An Index over a read-only connection to the file.

    Raises:
        FileNotFoundError: When there is no index, with the command that builds one.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no index: {path}. Run "
            f"`codetwine-loom build --project ... --output-dir ...` "
            f"first")
    return Index(open_read_only(path))


# How many instructions of its virtual machine SQLite runs between two counts while an
# index is read in a process of its own, and how many of those counts the read is given
# before its query is stopped
_PROGRESS_STEP = 1000
_MAX_PROGRESS_CALL = 1000

# How many bytes the pattern of a LIKE or a GLOB holds at most in that read
_MAX_LIKE_PATTERN_BYTE = 100

# The tables of an index read in a process of its own
_ANALYSIS_TABLE_TUPLE = ("meta", "calls")

# What the process an index is read in is held to: its address space in bytes, how long
# it may run in seconds, and how many bytes its answer holds
_READ_ADDRESS_BYTE = 1 << 30
_READ_TIMEOUT_SEC = 20
_MAX_ANSWER_BYTE = 1 << 20


@functools.lru_cache(maxsize=1)
def _schema_sql_dict() -> dict[str, str]:
    """Return the statement _SCHEMA makes each table of an index with.

    Returns:
        {the table's name: the statement as SQLite keeps it in sqlite_master}.
    """
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(_SCHEMA)
        return dict(connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'"))
    finally:
        connection.close()


# What the interpreter is given to run metaread's file: the file is read by its path
# and its main() is called
_METAREAD_START = "import runpy, sys; runpy.run_path(sys.argv[1])['main']()"


def _answer_alone(path: str, request_dict: dict) -> dict:
    """Have metaread answer one request in a process of its own, and return the answer.

    The process is this interpreter in its isolated mode, running metaread's file with
    config.own_child_env(). It is given _READ_TIMEOUT_SEC seconds, which it counts
    itself as well (the request's "max_second"), and of what it writes
    _MAX_ANSWER_BYTE bytes are read.

    Args:
        path: The file the request is about, for the messages.
        request_dict: The request, as metaread lays it out.

    Returns:
        The answer, as metaread lays it out.

    Raises:
        sqlite3.DatabaseError: When the process cannot be started, does not end in
            time, or ends without an answer of that form.
    """
    try:
        with tempfile.TemporaryFile() as answer_file:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _METAREAD_START, metaread.__file__],
                check=False,
                input=json.dumps(request_dict).encode("utf-8"), stdout=answer_file,
                stderr=subprocess.DEVNULL, env=config.own_child_env(),
                timeout=_READ_TIMEOUT_SEC)
            answer_file.seek(0)
            answer_text = answer_file.read(_MAX_ANSWER_BYTE + 1)
        # An end by SIGALRM is the time running out as well
        is_time_up = hasattr(signal, "SIGALRM") and proc.returncode == -signal.SIGALRM
    except subprocess.TimeoutExpired:
        is_time_up = True
    except OSError as e:
        raise sqlite3.DatabaseError(f"the process that reads {path} could not be "
                                    f"started: {e}") from None
    if is_time_up:
        raise sqlite3.DatabaseError(f"reading {path} did not end within "
                                    f"{_READ_TIMEOUT_SEC} seconds")
    try:
        answer_dict = json.loads(answer_text) if len(answer_text) <= _MAX_ANSWER_BYTE \
            else None
    except (ValueError, RecursionError):
        answer_dict = None
    if not isinstance(answer_dict, dict) or not all(
            isinstance(value, str) for value in answer_dict.values()):
        raise sqlite3.DatabaseError(f"the process that reads {path} ended with exit "
                                    f"status {proc.returncode} and no answer")
    return answer_dict


def analysis_of_index(path: str) -> tuple[str, str]:
    """Return the analysis an index was built from and the commit it records.

    The call edges, the definitions and the postings are not read. The index is held to
    the columns of meta and of calls and to a file_lang that is a JSON object. The file
    is read by metaread in a process of its own (_answer_alone), held to
    _READ_ADDRESS_BYTE of address space: it is opened as one nothing writes meanwhile,
    without the files beside it (read_only_uri), meta and calls are read only where they
    are the tables _SCHEMA makes (_schema_sql_dict), and what is run over the file is
    bounded (_PROGRESS_STEP, _MAX_PROGRESS_CALL, tmpfile.READ_MAX_BYTE,
    _MAX_LIKE_PATTERN_BYTE).

    Args:
        path: The path to code_index.sqlite.

    Returns:
        (the source stamp, the commit), each "" where the index holds no text under it.

    Raises:
        FileNotFoundError: When there is no index.
        sqlite3.DatabaseError: When the file does not hold those tables and a file_lang
            of that form, or reading it runs past a bound.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no index: {path}")
    answer_dict = _answer_alone(path, {
        "uri": read_only_uri(path, alone=True),
        "table": {name: _schema_sql_dict()[name] for name in _ANALYSIS_TABLE_TUPLE},
        "max_address_byte": _READ_ADDRESS_BYTE, "max_answer_byte": _MAX_ANSWER_BYTE,
        "max_second": _READ_TIMEOUT_SEC,
        "progress_step": _PROGRESS_STEP, "max_progress_call": _MAX_PROGRESS_CALL,
        "max_value_byte": tmpfile.READ_MAX_BYTE,
        "max_pattern_byte": _MAX_LIKE_PATTERN_BYTE})
    if "error" in answer_dict:
        raise sqlite3.DatabaseError(answer_dict["error"])
    return answer_dict.get("stamp", ""), answer_dict.get("commit", "")


def commit_in_index(path: str, source_stamp: str = "") -> str:
    """Return the commit an index already at path says its analysis describes.

    Args:
        path: The path to code_index.sqlite.
        source_stamp: The analysis the answer has to belong to (Knowledge.stamp()).
            The commit only comes back when the index was built from that same
            analysis; "" asks for whatever the index recorded, from whichever analysis.

    Returns:
        The full commit hash, or "" when there is no index there, it cannot be read, it
        was built from a different analysis, or it recorded no commit.
    """
    try:
        index = load(path)
    except (FileNotFoundError, sqlite3.DatabaseError) as e:
        logger.info("could not read the commit of %s: %s", path, e)
        return ""
    try:
        if source_stamp and index.source_stamp != source_stamp:
            logger.info("the index at %s was built from another analysis (%s, not %s), "
                        "so what it says about the commit is not carried over",
                        path, index.source_stamp, source_stamp)
            return ""
        return index.commit
    finally:
        index.close()


def primary_ext_of(path: str) -> str:
    """Return the primary language an index already at path settled on.

    Args:
        path: The path to code_index.sqlite.

    Returns:
        The index's primary_ext, or "" when there is no index there, it cannot be read,
        or it settled on none.
    """
    try:
        index = load(path)
    except (FileNotFoundError, sqlite3.DatabaseError) as e:
        logger.info("could not read the primary language of %s: %s", path, e)
        return ""
    try:
        return index.primary_ext or ""
    finally:
        index.close()


def is_from_analysis(path: str, source_stamp: str) -> bool:
    """Return whether the index at path was built from the analysis with this stamp.

    Args:
        path: The path to code_index.sqlite.
        source_stamp: The analysis on disk (Knowledge.stamp()).

    Returns:
        True when the index there was built from that analysis, holds this version's
        tables and was built under the rules in force now. False when there is no index
        there, it cannot be read, it was built from another analysis or another version,
        or a setting or a table the postings depend on has changed since.
    """
    # A missing index is not logged
    if not os.path.exists(path):
        return False
    try:
        index = load(path)
    except sqlite3.DatabaseError as e:
        logger.info("could not read the index at %s: %s", path, e)
        return False
    try:
        if index.source_stamp != source_stamp or index.schema_version != _SCHEMA_VERSION:
            return False
        if index.recipe != _build_recipe():
            logger.info("the index at %s was built with other settings or another way "
                        "of splitting words than this run uses, so it is built again",
                        path)
            return False
        return True
    finally:
        index.close()
