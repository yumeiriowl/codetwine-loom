"""Index step 5: what save() writes into the tables and what load() reads back."""

import dataclasses
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time

import pytest

from conftest import write_index, write_knowledge_db
from codetwine_loom import config, gitcmd
from codetwine_loom import knowledge as knowledge_mod
from codetwine_loom.index import build, metaread, store
from codetwine_loom.index.indexer import build_bm25
from codetwine_loom.index.store import is_constant


@pytest.fixture
def idx_and_defs():
    """An index and definition nodes for 2 definitions in 1 file."""
    defs = {
        "p/a_py/a.py::f": {"name": "f", "type": "function_definition",
                           "file": "p/a_py/a.py", "lines": [1, 3],
                           "signature": "def f():",
                           "doc": "parse config value"},
        "p/a_py/a.py::g": {"name": "g", "type": "expression_statement",
                           "file": "p/a_py/a.py", "lines": [5, 7],
                           "signature": "g = 1",
                           "doc": "unrelated text"},
    }
    files = {"p/a_py/a.py": {"summary": "config helpers for the app",
                             "defs": list(defs)}}
    idx = {"nodes": {"defs": defs, "files": files},
           "edges": {"calls": [("p/a_py/a.py::f", "p/a_py/a.py::g")],
                     },
           "index": {"def": build_bm25({k: (d["signature"], d["doc"])
                                        for k, d in defs.items()}),
                     "file": build_bm25({p: ("", n["summary"])
                                         for p, n in files.items()})}}
    return idx, defs


def _saved(tmp_path, idx: dict, defs: dict, stamp: str = "stamp",
           project: str = "p", commit: str = "") -> store.Index:
    """Write the index and open it for reading."""
    path = str(tmp_path / store.INDEX_FILE)
    store.save(path, idx, defs, stamp, project, commit)
    return store.load(path)


def test_save_does_not_modify_the_idx_argument(tmp_path, idx_and_defs):
    """A caller that builds and then saves must get its own dict back unchanged."""
    idx, defs = idx_and_defs
    before = {k: list(v) for k, v in idx["index"]["def"]["postings"].items()}
    _saved(tmp_path, idx, defs).close()
    assert idx["index"]["def"]["postings"] == before
    # what the caller holds is untouched, so a build that saves can go on searching
    assert "doc" in defs[next(iter(defs))]


def test_every_definition_is_written_with_its_node_fields(tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    index = _saved(tmp_path, idx, defs)
    node = index.definition("p/a_py/a.py::f")
    assert node == {"name": "f", "type": "function_definition",
                    "file": "p/a_py/a.py", "lines": [1, 3],
                    "signature": "def f():", "signature_line": 1}
    assert index.definition("p/a_py/a.py::nosuch") is None
    index.close()


def test_the_in_degree_and_the_constants_are_settled_when_the_index_is_written(
        tmp_path, idx_and_defs):
    """The search side reads both as columns rather than working them out per query."""
    idx, defs = idx_and_defs
    index = _saved(tmp_path, idx, defs)
    assert index.indegree_dict == {"p/a_py/a.py::g": 1}
    # g is an expression_statement, which Python counts as a constant
    assert index.doc_count("def") == 1
    assert index.doc_count("file") == 1
    index.close()


def test_a_constant_takes_part_when_include_constants_is_on(tmp_path, idx_and_defs,
                                                            monkeypatch):
    idx, defs = idx_and_defs
    monkeypatch.setattr(config, "INCLUDE_CONSTANT", True)
    index = _saved(tmp_path, idx, defs)
    assert index.doc_count("def") == 2
    assert [i for i, _, _ in index.posting_list("def", "unrelated")]
    index.close()


def test_include_constants_is_followed_after_the_index_was_opened(
        tmp_path, idx_and_defs, monkeypatch):
    """Every other setting is read per call, so the count cannot be cached by kind."""
    idx, defs = idx_and_defs
    index = _saved(tmp_path, idx, defs)
    assert index.doc_count("def") == 1
    monkeypatch.setattr(config, "INCLUDE_CONSTANT", True)
    assert index.doc_count("def") == 2
    index.close()


def test_a_save_load_round_trip_keeps_every_posting(tmp_path, idx_and_defs,
                                                    monkeypatch):
    """What comes back has to score against the keys that went in (step 5)."""
    idx, defs = idx_and_defs
    monkeypatch.setattr(config, "INCLUDE_CONSTANT", True)
    index = _saved(tmp_path, idx, defs)
    ids = index.key_dict("def", [0, 1])
    read_back = {}
    for term in idx["index"]["def"]["postings"]:
        read_back[term] = [[ids[i], n] for i, n, _ in index.posting_list("def", term)]
    assert read_back == idx["index"]["def"]["postings"]
    assert index.avg_len("def") == idx["index"]["def"]["avg_len"]
    index.close()


def test_the_postings_come_back_id_ascending(tmp_path, idx_and_defs, monkeypatch):
    """The order decides which of two equally scored definitions comes first."""
    idx, defs = idx_and_defs
    monkeypatch.setattr(config, "INCLUDE_CONSTANT", True)
    index = _saved(tmp_path, idx, defs)
    for term in idx["index"]["def"]["postings"]:
        ids = [i for i, _, _ in index.posting_list("def", term)]
        assert ids == sorted(ids)
    index.close()


def test_the_postings_carry_each_documents_length(tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    index = _saved(tmp_path, idx, defs)
    lengths = {index.key_dict("def", [i])[i]: dl
               for i, _, dl in index.posting_list("def", "config")}
    assert lengths == {"p/a_py/a.py::f": idx["index"]["def"]["len"]["p/a_py/a.py::f"]}
    index.close()


def test_has_term_answers_for_both_indexes(tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    index = _saved(tmp_path, idx, defs)
    assert index.has_term("def", "config") is True
    assert index.has_term("def", "nonexistentword") is False
    assert index.has_term("file", "helpers") is True
    index.close()


def test_the_file_nodes_and_the_call_edges_come_back(tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    index = _saved(tmp_path, idx, defs)
    assert index.file_path_list() == ["p/a_py/a.py"]
    assert index.file_summary("p/a_py/a.py") == "config helpers for the app"
    assert index.file_summary("p/nosuch_py/nosuch.py") == ""
    assert index.call_list == [("p/a_py/a.py::f", "p/a_py/a.py::g")]
    assert index.caller_list("p/a_py/a.py::g") == ["p/a_py/a.py::f"]
    assert index.caller_list("p/a_py/a.py::f") == []
    index.close()


def test_definitions_named_finds_every_file_carrying_the_name(tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    index = _saved(tmp_path, idx, defs)
    assert index.def_list_by_name("f") == [{"key": "p/a_py/a.py::f",
                                             "file": "p/a_py/a.py"}]
    assert index.def_list_by_name("nosuch") == []
    assert index.file_by_def_dict(defs) == {k: "p/a_py/a.py" for k in defs}
    index.close()


def test_a_failed_write_leaves_the_previous_index_in_place(tmp_path, idx_and_defs,
                                                           monkeypatch):
    """A build killed mid-write must not take the index that was already there."""
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    store.save(path, idx, defs, "stamp", "p", "")
    before = open(path, "rb").read()

    def die(*a, **k):
        raise OSError("no space left on device")

    monkeypatch.setattr(store, "_iter_posting_row", die)
    with pytest.raises(OSError):
        store.save(path, idx, defs, "stamp", "p", "")
    assert open(path, "rb").read() == before
    # and it takes its temporary file with it: no later run removes another process's
    # leftovers, so one per failed build would stay in the output directory
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
    index = store.load(path)
    assert index.project == "p"
    index.close()


def test_a_leftover_of_a_killed_build_is_taken_away_by_the_next_one(tmp_path,
                                                                   idx_and_defs):
    """A build killed outright cannot run its own cleanup, so the next build removes
    the file it left.

    The file of a build that is still running is kept: that is what the process ID in
    the name is there for.

    What sqlite wrote beside the temporary file goes with it. A killed write leaves its
    rollback journal at "<the temporary file>-journal", and the sweep removes it as
    well as the names ending in ".tmp".
    """
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()
    killed = tmp_path / f"{store.INDEX_FILE}.{dead.pid}.tmp"
    killed_journal = tmp_path / f"{killed.name}-journal"
    running = tmp_path / f"{store.INDEX_FILE}.{os.getppid()}.tmp"
    running_journal = tmp_path / f"{running.name}-journal"
    for leftover in (killed, killed_journal, running, running_journal):
        leftover.write_bytes(b"half an index")

    store.save(path, idx, defs, "stamp", "p", "")

    assert not killed.exists()
    assert not killed_journal.exists()
    # another build may still be writing them
    assert running.exists() and running_journal.exists()
    store.load(path).close()


def test_save_records_the_project_the_stamp_and_the_settings(tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    index = _saved(tmp_path, idx, defs, stamp="1:2026-01-01T00:00:00+00:00",
                   project="myproj")
    assert index.project == "myproj"
    assert index.source_stamp == "1:2026-01-01T00:00:00+00:00"
    assert index.primary_ext == "py"
    index.close()


def test_the_index_is_opened_read_only(tmp_path, idx_and_defs):
    """Nothing but a build writes an index."""
    idx, defs = idx_and_defs
    index = _saved(tmp_path, idx, defs)
    with pytest.raises(sqlite3.OperationalError):
        index._connection.execute("DELETE FROM defs")
    index.close()


@pytest.mark.parametrize("name", ["C#", "what?", "a%41b", "with space"])
def test_an_index_under_a_path_a_uri_reads_marks_in_is_opened_where_it_is(
        tmp_path, idx_and_defs, name):
    idx, defs = idx_and_defs
    (tmp_path / name).mkdir()
    index = _saved(tmp_path / name, idx, defs)
    assert index.project == "p"
    with pytest.raises(sqlite3.OperationalError):
        index._connection.execute("DELETE FROM defs")
    index.close()
    assert os.listdir(tmp_path) == [name]


def test_an_index_under_a_path_written_with_two_slashes_in_front_is_opened(
        tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    _saved(tmp_path, idx, defs).close()
    path = str(tmp_path / store.INDEX_FILE)
    for written in ("/" + path, "//" + path):
        index = store.load(written)
        assert index.project == "p"
        with pytest.raises(sqlite3.OperationalError):
            index._connection.execute("DELETE FROM defs")
        index.close()


def test_an_index_is_opened_by_a_path_relative_to_the_working_directory(
        tmp_path, idx_and_defs, monkeypatch):
    idx, defs = idx_and_defs
    (tmp_path / "a#b").mkdir()
    _saved(tmp_path / "a#b", idx, defs).close()
    monkeypatch.chdir(tmp_path)
    index = store.load(os.path.join("a#b", store.INDEX_FILE))
    assert index.project == "p"
    index.close()
    assert os.listdir(tmp_path) == ["a#b"]


def test_the_uri_of_a_database_names_no_host_and_keeps_every_mark_in_the_path():
    assert store.read_only_uri("/out/app/x.sqlite") == "file:///out/app/x.sqlite?mode=ro"
    assert store.read_only_uri("/out/app/x.sqlite", alone=True) == (
        "file:///out/app/x.sqlite?mode=ro&immutable=1")
    assert store.read_only_uri("//out/C#/x?.sqlite") == (
        "file:////out/C%23/x%3F.sqlite?mode=ro")


def test_load_fails_with_the_build_instructions_when_there_is_no_index(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        store.load(str(tmp_path / "missing.sqlite"))
    assert "build" in str(e.value)


def _node(file: str, type_: str = "function_definition") -> dict:
    """One definition node with what primary_ext() reads."""
    return {"file": file, "type": type_}


def test_primary_ext_is_the_extension_the_most_files_define_in():
    defs = {"a.java::x": _node("a.java"), "b.java::y": _node("b.java"),
            "c.py::z": _node("c.py")}
    assert store.primary_ext(defs) == "java"


def test_primary_ext_counts_files_rather_than_definitions():
    """One large bundled file does not outweigh the sources."""
    defs = {f"vendor.js::f{i}": _node("vendor.js") for i in range(20)}
    defs.update({"a.py::x": _node("a.py"), "b.py::y": _node("b.py")})
    assert store.primary_ext(defs) == "py"


def test_primary_ext_leaves_out_what_the_language_calls_a_constant():
    defs = {"a.js::A": _node("a.js", "lexical_declaration"),
            "b.js::B": _node("b.js", "lexical_declaration"),
            "c.py::z": _node("c.py")}
    assert store.primary_ext(defs) == "py"


def test_primary_ext_reads_headers_as_the_language_whose_sources_are_there():
    defs = {"a.h::x": _node("a.h"), "b.h::y": _node("b.h"), "c.c::z": _node("c.c")}
    assert store.primary_ext(defs) == "c"
    defs["d.cpp::w"] = _node("d.cpp")
    assert store.primary_ext(defs) == "h"


@pytest.mark.parametrize("file_list, want", [
    # Only the language whose project is at the root is taken
    (["app/pom_xml/pom.xml", "app/native/a_c/a.c"], "java"),
    (["pom.xml"], "java"),
    # A project file below the root does not count
    (["app/native/Makefile/Makefile", "app/pom_xml/pom.xml"], "java"),
    # With the projects of both, or of neither, at the root, files are counted
    (["app/pom_xml/pom.xml", "app/Makefile/Makefile"], "c"),
    (["app/README_md/README.md"], "c"),
    # A file of a language that is not counted is passed over
    (["app/package_json/package.json"], "c")])
def test_primary_ext_takes_the_language_whose_project_is_at_the_repository_root(
        file_list, want):
    defs = {"native/a.c::x": _node("native/a.c"), "native/a.h::y": _node("native/a.h"),
            "src/A.java::z": _node("src/A.java")}
    assert store.primary_ext(defs, file_list, "app") == want


@pytest.mark.parametrize("defs_spec, file_list, want", [
    # Gradle's files make a Java project as well
    ({"java": 3, "kt": 1}, ["app/build_gradle_kts/build.gradle.kts"], "java"),
    # A language whose tests need no project file is never left out
    ({"py": 3, "js": 1}, ["app/package_json/package.json"], "py"),
    ({"cbl": 3, "c": 1}, ["app/Makefile/Makefile"], "cbl"),
    # Nothing is left out while no language counted has a project file at the root
    ({"cpp": 3, "py": 1}, ["app/meson_build/meson.build"], "cpp")])
def test_primary_ext_leaves_out_only_a_language_whose_tests_need_a_root_project(
        defs_spec, file_list, want):
    defs = {f"d/f{i}.{ext}::x": _node(f"d/f{i}.{ext}")
            for ext, count in defs_spec.items() for i in range(count)}
    assert store.primary_ext(defs, file_list, "app") == want


def test_primary_ext_is_empty_when_there_are_no_definitions():
    """It names no language of its own; config.set_primary_ext() decides and warns."""
    assert store.primary_ext({}) == ""
    assert store.primary_ext({}, ["README.md", "dbt_project.yml"]) == ""


def test_primary_ext_counts_the_files_where_no_file_defines_anything():
    """A repository of queries with no definitions in them is a SQL repository."""
    file_list = ["models/a.sql", "models/b.sql", "models/schema.yml", "scripts/c.py"]
    assert store.primary_ext({}, file_list) == "sql"
    defs = {"scripts/c.py::main": _node("scripts/c.py")}
    assert store.primary_ext(defs, file_list) == "py"


# == is_constant ============================================================

def test_is_constant_recognises_a_js_constant(js_defs):
    excluded = [k for k, d in js_defs.items() if is_constant(k, d)]
    assert len(excluded) == 2
    assert all(js_defs[k]["type"] == "lexical_declaration" for k in excluded)


def test_is_constant_does_not_exclude_a_function(js_defs):
    key = "app/src/config_js/config.js::loadConfig"
    assert is_constant(key, js_defs[key]) is False


def test_is_constant_excludes_pythons_expression_statement():
    key = "p/s_py/s.py::API_BASE"
    assert is_constant(key, {"type": "expression_statement"}) is True


def test_is_constant_looks_at_a_different_type_per_language():
    """An expression_statement is not a constant in Java."""
    assert is_constant("p/C_java/C.java::X", {"type": "expression_statement"}) is False
    assert is_constant("p/C_java/C.java::X", {"type": "field_declaration"}) is True


def test_a_namespace_is_kept_in_a_c_source_file():
    """namespace_definition is left out for the C++ extensions and headers alone."""
    assert is_constant("p/c_c/c.c::X", {"type": "namespace_definition"}) is False


def test_is_constant_is_false_for_an_extension_not_covered():
    assert is_constant("p/docs_md/docs.md::X", {"type": "declaration"}) is False


@pytest.mark.parametrize("path, constant_type, kept_type", [
    ("p/C_java/C.java", "enum_constant", "record_declaration"),
    ("p/C_java/C.java", "constant_declaration", "interface_declaration"),
    ("p/c_cpp/c.cpp", "namespace_definition", "class_specifier"),
    ("p/c_h/c.h", "namespace_definition", "function_definition"),
    ("p/c_hpp/c.hpp", "namespace_definition", "struct_specifier"),
    ("p/c_rs/c.rs", "mod_item", "function_item"),
    ("p/c_rs/c.rs", "associated_type", "trait_item"),
    ("p/C_kt/C.kt", "enum_entry", "type_alias"),
    ("p/c_c/c.c", "enumerator", "preproc_function_def"),
    ("p/c_h/c.h", "enumerator", "type_definition"),
    ("p/c_ts/c.ts", "property_signature", "method_signature"),
    ("p/c_ts/c.ts", "enum_assignment", "function_signature"),
    ("p/c_ts/c.ts", "property_identifier", "abstract_class_declaration"),
    ("p/C_cs/C.cs", "indexer_declaration", "operator_declaration"),
    ("p/c_js/c.js", "variable_declaration", "expression_statement"),
    ("p/c_r/c.r", "binary_operator", "function_definition"),
])
def test_is_constant_tells_a_value_from_a_function_or_a_type(path, constant_type, kept_type):
    """An enum member, a field of an interface and a variable are left out; a type, a
    function-like macro, a signature and a function assigned to a member are not."""
    assert is_constant(f"{path}::X", {"type": constant_type}) is True
    assert is_constant(f"{path}::X", {"type": kept_type}) is False


def test_an_index_name_of_cobol_is_left_out_of_the_search():
    assert is_constant("p/A_cbl/A.cbl::WS-IX", {"type": "occurs_indexed"}) is True


# == the commit the index says its analysis describes ========

def _git_repo(path):
    """Make a git repository with one commit and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    for args in (("init", "--quiet", "-b", "main"),
                 ("config", "user.email", "store@example.invalid"),
                 ("config", "user.name", "store")):
        subprocess.run(["git", "-C", str(path), *args], check=True,
                       capture_output=True)
    (path / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "--quiet", "-m", "first"],
                   check=True, capture_output=True)
    return path


def test_recorded_commit_gives_what_the_index_says(tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    store.save(path, idx, defs, "stamp", "p", "abc123")
    assert store.commit_in_index(path) == "abc123"
    assert store.commit_in_index(path, "stamp") == "abc123"


def test_recorded_commit_says_nothing_for_another_analysis(tmp_path, idx_and_defs):
    """The commit belongs to the analysis the index was built from, not to the path."""
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    store.save(path, idx, defs, "stamp", "p", "abc123")
    assert store.commit_in_index(path, "another stamp") == ""


@pytest.mark.parametrize("written", [None, "not a database"])
def test_recorded_commit_says_nothing_when_it_cannot_read(tmp_path, written):
    path = str(tmp_path / store.INDEX_FILE)
    if written is not None:
        open(path, "w", encoding="utf-8").write(written)
    assert store.commit_in_index(path) == ""


# One change to a built index each: what the meta table holds under file_lang, or a
# table every reader of the index is made over
_OTHER_SHAPE_SQL_LIST = [
    "UPDATE meta SET value = '{' WHERE key = 'file_lang'",
    "UPDATE meta SET value = X'FFFE' WHERE key = 'file_lang'",
    "UPDATE meta SET value = '5' WHERE key = 'file_lang'",
    "UPDATE meta SET value = '[\"a.py\"]' WHERE key = 'file_lang'",
    f"UPDATE meta SET value = '{'[' * 100000}' WHERE key = 'file_lang'",
    "ALTER TABLE calls RENAME COLUMN dst TO other",
    "DROP TABLE calls",
    "ALTER TABLE meta RENAME COLUMN key TO other"]


def _index_with(path, idx, defs, sql):
    """Save an index and run one statement over it."""
    store.save(path, idx, defs, "stamp", "p", "abc123")
    connection = sqlite3.connect(path)
    connection.execute(sql)
    connection.commit()
    connection.close()


@pytest.mark.parametrize("sql", _OTHER_SHAPE_SQL_LIST)
def test_an_index_of_another_shape_is_one_that_cannot_be_read(tmp_path, idx_and_defs, sql):
    """Every reader takes it as an index that is not there to read, and none ends on
    another error."""
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    _index_with(path, idx, defs, sql)
    with pytest.raises(sqlite3.DatabaseError):
        store.load(path)
    with pytest.raises(sqlite3.DatabaseError):
        store.analysis_of_index(path)
    assert store.commit_in_index(path) == ""
    assert store.is_from_analysis(path, "stamp") is False


def test_the_analysis_and_the_commit_of_an_index_are_read_without_its_rows(
        tmp_path, idx_and_defs, monkeypatch):
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    store.save(path, idx, defs, "stamp", "p", "abc123")
    monkeypatch.setattr(store, "Index", lambda connection: pytest.fail("loaded"))
    assert store.analysis_of_index(path) == ("stamp", "abc123")
    # A value that is no text is none
    _index_with(path, idx, defs, "UPDATE meta SET value = X'00FF' WHERE key = 'commit'")
    assert store.analysis_of_index(path) == ("stamp", "")
    with pytest.raises(FileNotFoundError):
        store.analysis_of_index(str(tmp_path / "none.sqlite"))


# What stands in the place of a table the analysis of an index is read from: a view
# whose rows never end, a view over the rows that counts first, a view of the table's
# rows under the name in another case, a table with a column computed when it is read,
# and a table of the same columns in another order
_NO_TABLE_SQL_LIST = [
    ["ALTER TABLE meta RENAME TO meta_real",
     "CREATE VIEW meta AS WITH RECURSIVE c(key, value) AS (SELECT 'x', 'y' UNION ALL "
     "SELECT key, value FROM c) SELECT key, value FROM c"],
    ["ALTER TABLE meta RENAME TO meta_real",
     "CREATE VIEW meta AS SELECT key, value FROM meta_real WHERE (WITH RECURSIVE "
     "n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 40000000) "
     "SELECT count(*) FROM n) > 0"],
    ["ALTER TABLE meta RENAME TO meta_real",
     "CREATE VIEW META AS SELECT key, value FROM meta_real"],
    ["ALTER TABLE calls RENAME TO calls_real",
     "CREATE VIEW calls AS SELECT src, dst FROM calls_real"],
    ["ALTER TABLE meta RENAME TO meta_real",
     "CREATE TABLE meta (key TEXT PRIMARY KEY, real_value TEXT, value TEXT GENERATED "
     "ALWAYS AS (upper(real_value)) VIRTUAL)",
     "INSERT INTO meta (key, real_value) SELECT key, value FROM meta_real"],
    ["ALTER TABLE meta RENAME TO meta_real",
     "CREATE TABLE meta (value TEXT, key TEXT PRIMARY KEY)",
     "INSERT INTO meta (key, value) SELECT key, value FROM meta_real"],
]


@pytest.mark.parametrize("sql_list", _NO_TABLE_SQL_LIST)
def test_the_analysis_of_an_index_is_read_off_tables_alone(tmp_path, idx_and_defs,
                                                          sql_list):
    """What is under the name of a table the analysis is read from is queried only
    where it is the table an index is saved with."""
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    store.save(path, idx, defs, "stamp", "p", "abc123")
    connection = sqlite3.connect(path)
    for sql in sql_list:
        connection.execute(sql)
    connection.commit()
    connection.close()
    start = time.monotonic()
    with pytest.raises(sqlite3.DatabaseError,
                       match="is not the table an index holds under that name"):
        store.analysis_of_index(path)
    assert time.monotonic() - start < 5


def test_reading_the_analysis_of_an_index_stops_at_its_bound(tmp_path, idx_and_defs,
                                                             monkeypatch):
    """The instructions run over the file are counted, and a value and the answer
    are held to a size."""
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    _index_with(path, idx, defs, f"UPDATE meta SET value = '{'c' * 5000}' "
                                 f"WHERE key = 'commit'")
    assert store.analysis_of_index(path) == ("stamp", "c" * 5000)
    with monkeypatch.context() as patch:
        patch.setattr(store.tmpfile, "READ_MAX_BYTE", 4000)
        with pytest.raises(sqlite3.DatabaseError, match="too big"):
            store.analysis_of_index(path)
    with monkeypatch.context() as patch:
        patch.setattr(store, "_MAX_ANSWER_BYTE", 4000)
        with pytest.raises(sqlite3.DatabaseError, match="and no answer"):
            store.analysis_of_index(path)
    monkeypatch.setattr(store, "_PROGRESS_STEP", 1)
    monkeypatch.setattr(store, "_MAX_PROGRESS_CALL", 2)
    with pytest.raises(sqlite3.DatabaseError, match="interrupted"):
        store.analysis_of_index(path)


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_the_analysis_of_an_index_is_read_without_the_files_beside_it(
        tmp_path, idx_and_defs, suffix):
    """A pipe under the name of one of them is not opened, and nothing is made there."""
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    store.save(path, idx, defs, "stamp", "p", "abc123")
    os.mkfifo(path + suffix)
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; from codetwine_loom.index import store; "
                               "print(store.analysis_of_index(sys.argv[1]))", path],
        capture_output=True, text=True, timeout=60, check=False)
    assert proc.stdout.strip() == "('stamp', 'abc123')", proc.stderr
    assert sorted(os.listdir(tmp_path)) == sorted(
        [store.INDEX_FILE, store.INDEX_FILE + suffix])


def test_a_bounded_connection_stops_a_query_that_never_ends(tmp_path):
    connection = sqlite3.connect(":memory:")
    metaread._bound_read(connection, store._PROGRESS_STEP, store._MAX_PROGRESS_CALL,
                         store.tmpfile.READ_MAX_BYTE, store._MAX_LIKE_PATTERN_BYTE)
    start = time.monotonic()
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        connection.execute("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 "
                           "FROM n) SELECT count(*) FROM n").fetchall()
    assert time.monotonic() - start < 5
    connection.close()


def _run_each(path, sql_list_list):
    """Run each list of statements over an index, on a connection of its own."""
    for sql_list in sql_list_list:
        connection = sqlite3.connect(path)
        for sql in sql_list:
            connection.execute(sql)
        connection.commit()
        connection.close()


def _compute_as_the_schema_is_read(path):
    """Give the table SQLite keeps its statistics in, which it reads along with the
    schema, a column computed when it is read, and rows to compute it for."""
    table = ("CREATE TABLE sqlite_stat1(tbl,idx,real_stat,stat GENERATED ALWAYS AS "
             "(CASE WHEN (printf('%.*c', 2000000, 'a') || tbl) LIKE '%' || "
             "printf('%.*c', 90, 'a') || 'b' THEN '1' ELSE real_stat END) VIRTUAL)")
    _run_each(path, [[
        "ANALYZE",
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 400) "
        "INSERT INTO sqlite_stat1 (tbl, idx, stat) SELECT 't' || i, NULL, '1' FROM n",
        "PRAGMA writable_schema=ON",
        "UPDATE sqlite_master SET sql = '{}' WHERE name = 'sqlite_stat1'".format(
            table.replace("'", "''"))]])


def _put_a_pipe_in_its_place(path):
    os.remove(path)
    os.mkfifo(path)


def _hold_many_wide_tables(path):
    column_text = ", ".join(f"c{n} TEXT" for n in range(2000))
    _run_each(path, [[f"CREATE TABLE wide_{n} ({column_text})" for n in range(250)]])


@pytest.mark.parametrize("change, bound_dict, why", [
    (_compute_as_the_schema_is_read, {"_READ_TIMEOUT_SEC": 2},
     "did not end within 2 seconds"),
    (_put_a_pipe_in_its_place, {"_READ_TIMEOUT_SEC": 2}, "did not end within 2 seconds"),
    (_hold_many_wide_tables, {"_READ_ADDRESS_BYTE": 32 << 20},
     "MemoryError|out of memory|and no answer"),
])
def test_an_index_that_costs_its_reader_time_or_memory_is_one_that_cannot_be_read(
        tmp_path, idx_and_defs, monkeypatch, change, bound_dict, why):
    """What SQLite computes and holds when it reads the schema is outside what a
    connection is bounded by. The process that reads the index is stopped at its
    time or its address space, and this one goes on; with the bounds as they are set,
    an index as it is saved is read."""
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    store.save(path, idx, defs, "stamp", "p", "abc123")
    with monkeypatch.context() as patch:
        for name, value in bound_dict.items():
            patch.setattr(store, name, value)
        assert store.analysis_of_index(path) == ("stamp", "abc123")
        change(path)
        start = time.monotonic()
        with pytest.raises(sqlite3.DatabaseError, match=why):
            store.analysis_of_index(path)
        assert time.monotonic() - start < 15


@pytest.mark.parametrize("change", [_compute_as_the_schema_is_read,
                                    _put_a_pipe_in_its_place])
def test_the_process_that_reads_an_index_ends_itself_at_the_same_number_of_seconds(
        tmp_path, idx_and_defs, monkeypatch, change):
    """The process that started it waits far longer here, as one that is no longer
    there does: the reader counts the seconds itself and ends with SIGALRM."""
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    store.save(path, idx, defs, "stamp", "p", "abc123")
    change(path)
    seen = {}
    real_run = subprocess.run

    def run(argv, **kwargs):
        seen["request"] = json.loads(kwargs["input"])
        proc = real_run(argv, **{**kwargs, "timeout": 60})
        seen["status"] = proc.returncode
        return proc

    monkeypatch.setattr(store, "_READ_TIMEOUT_SEC", 1)
    monkeypatch.setattr(store.subprocess, "run", run)
    start = time.monotonic()
    with pytest.raises(sqlite3.DatabaseError, match="did not end within 1 seconds"):
        store.analysis_of_index(path)
    assert time.monotonic() - start < 15
    assert seen["request"]["max_second"] == 1 and seen["status"] == -signal.SIGALRM


def test_the_process_that_reads_an_index_is_given_no_setting_of_this_one(
        tmp_path, idx_and_defs, monkeypatch):
    """It is the interpreter in its isolated mode on metaread's own file, with neither
    a credential nor a .env file to read."""
    idx, defs = idx_and_defs
    path = str(tmp_path / store.INDEX_FILE)
    store.save(path, idx, defs, "stamp", "p", "abc123")
    seen = {}
    real_run = subprocess.run

    def run(argv, **kwargs):
        seen.update(argv=argv, env=kwargs["env"], timeout=kwargs["timeout"])
        return real_run(argv, **kwargs)

    def no_start(argv, **kwargs):
        raise OSError("no interpreter")

    monkeypatch.setenv("LLM_API_KEY", "sk-not-handed-on")
    monkeypatch.setattr(store.subprocess, "run", run)
    assert store.analysis_of_index(path) == ("stamp", "abc123")
    assert seen["argv"] == [sys.executable, "-I", "-c", store._METAREAD_START,
                            metaread.__file__]
    assert "LLM_API_KEY" not in seen["env"]
    assert seen["env"][config.NO_ENV_FILE_ENV] == config.NO_ENV_FILE_VALUE
    assert seen["timeout"] == store._READ_TIMEOUT_SEC
    monkeypatch.setattr(store.subprocess, "run", no_start)
    with pytest.raises(sqlite3.DatabaseError, match="could not be started"):
        store.analysis_of_index(path)


def test_the_process_that_reads_an_index_keeps_a_lower_limit_it_runs_under():
    """The process runs under soft limits below the ones it is to hold itself to."""
    code = ("import importlib.util, resource, sys\n"
            "spec = importlib.util.spec_from_file_location('metaread', sys.argv[1])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "for name, soft in ((resource.RLIMIT_AS, 1 << 30), (resource.RLIMIT_FSIZE, 1 << 20)):\n"
            "    resource.setrlimit(name, (soft, resource.getrlimit(name)[1]))\n"
            "module._hold_process(4 << 30, 8 << 20, 5)\n"
            "print(*resource.getrlimit(resource.RLIMIT_AS),\n"
            "      *resource.getrlimit(resource.RLIMIT_FSIZE))\n")
    proc = subprocess.run([sys.executable, "-I", "-c", code, metaread.__file__],
                          capture_output=True, text=True, check=False)
    assert proc.stdout.split() == [str(1 << 30)] * 2 + [str(1 << 20)] * 2, proc.stderr


def _record_beside(path: str, stamp: str, commit: str) -> None:
    """Write the analysis' commit record in the index's directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(os.path.join(os.path.dirname(path), knowledge_mod.ANALYSIS_COMMIT_FILE),
              "w", encoding="utf-8") as fp:
        json.dump({"stamp": stamp, "commit": commit}, fp)


def test_a_reused_analysis_keeps_the_commit_it_was_made_at(tmp_path, knowledge_of):
    """The index must not claim a state its analysis was not made at.

    Building again without re-analysing is what a `build` does while the analysis
    describes what is checked out. When the commit checked out has moved on, the index
    keeps the commit the analysis was made at, so a sandboxed run does not trust an
    analysis of some earlier state.
    """
    repo = _git_repo(tmp_path / "repo")
    path = str(tmp_path / "out" / store.INDEX_FILE)
    knowledge = knowledge_of({"files": []})
    analysed_at = gitcmd.head_commit(str(repo))
    _record_beside(path, knowledge.stamp(), analysed_at)
    build.build_and_save(knowledge, "p", path)

    subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "--allow-empty",
                    "-m", "later"], check=True, capture_output=True)
    build.build_and_save(knowledge, "p", path)

    assert store.commit_in_index(path) == analysed_at
    assert store.commit_in_index(path) != gitcmd.head_commit(str(repo))


def test_a_reused_analysis_says_nothing_when_there_is_nothing_to_carry(tmp_path, knowledge_of):
    """No index to read the commit off. "" is what every reader treats as no match."""
    path = str(tmp_path / "out" / store.INDEX_FILE)
    build.build_and_save(knowledge_of({"files": []}), "p", path)
    assert store.commit_in_index(path) == ""


def test_a_reused_analysis_says_nothing_when_the_index_is_of_another_analysis(
        tmp_path, monkeypatch, knowledge_of, idx_and_defs):
    idx, defs = idx_and_defs
    path = str(tmp_path / "out" / store.INDEX_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store.save(path, idx, defs, "an analysis this one is not", "p", "abc123")
    build.build_and_save(knowledge_of({"files": []}), "p", path)
    assert store.commit_in_index(path) == ""


# == whether the index holds any summary at all ==============
def test_an_index_with_a_summary_says_so(tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    path = str(tmp_path / "i.sqlite")
    store.save(path, idx, defs, "1:now", "p", "")
    assert store.load(path).has_summary() is True


def test_an_index_built_without_documents_says_it_holds_no_summary(tmp_path,
                                                                    idx_and_defs):
    """A build with the documents off, and the container's rebuild, leave every
    summary empty; the stages read this before sending the model there."""
    idx, defs = idx_and_defs
    idx["nodes"]["files"]["p/a_py/a.py"]["summary"] = ""
    path = str(tmp_path / "i.sqlite")
    store.save(path, idx, defs, "1:now", "p", "")
    assert store.load(path).has_summary() is False


# == is_from_analysis: whether the index describes the analysis on disk ===========
def test_built_from_says_yes_for_the_same_analysis(tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    _saved(tmp_path, idx, defs, stamp="stamp").close()
    assert store.is_from_analysis(str(tmp_path / store.INDEX_FILE), "stamp") is True


def test_built_from_says_no_for_another_analysis(tmp_path, idx_and_defs):
    idx, defs = idx_and_defs
    _saved(tmp_path, idx, defs, stamp="stamp").close()
    assert store.is_from_analysis(str(tmp_path / store.INDEX_FILE), "another stamp") is False


def test_built_from_says_no_for_another_version_of_the_tables(tmp_path, idx_and_defs):
    """An index whose tables are of an earlier version is made again, as one from
    another analysis is, rather than read until a query fails on a column."""
    idx, defs = idx_and_defs
    _saved(tmp_path, idx, defs, stamp="stamp").close()
    path = str(tmp_path / store.INDEX_FILE)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
    assert store.is_from_analysis(path, "stamp") is False


@pytest.mark.parametrize("change", ["ignore_case", "header_owner", "alias"])
def test_built_from_says_no_when_a_table_the_index_follows_changes(
        tmp_path, idx_and_defs, monkeypatch, change):
    idx, defs = idx_and_defs
    _saved(tmp_path, idx, defs, stamp="stamp").close()
    path = str(tmp_path / store.INDEX_FILE)
    if change == "ignore_case":
        monkeypatch.setitem(store.LANG_DICT, "py", dataclasses.replace(
            store.LANG_DICT["py"], ignore_case=not store.LANG_DICT["py"].ignore_case))
    elif change == "header_owner":
        monkeypatch.setattr(store, "HEADER_OWNER_TUPLE", ())
    else:
        monkeypatch.setattr(store, "ALIAS_DICT", {})
    assert store.is_from_analysis(path, "stamp") is False


def test_built_from_says_no_when_there_is_no_index_or_it_cannot_be_read(tmp_path):
    path = str(tmp_path / store.INDEX_FILE)
    assert store.is_from_analysis(path, "stamp") is False
    (tmp_path / store.INDEX_FILE).write_text("not a database")
    assert store.is_from_analysis(path, "stamp") is False


# == commit_of_analysis: the record beside the analysis, then the index ======
def test_a_reused_analysis_takes_the_commit_recorded_beside_it(tmp_path, knowledge_of):
    path = str(tmp_path / "out" / store.INDEX_FILE)
    knowledge = knowledge_of({"files": []})
    _record_beside(path, knowledge.stamp(), "recorded1")
    build.build_and_save(knowledge, "p", path)
    assert store.commit_in_index(path) == "recorded1"


def test_the_record_beside_the_analysis_wins_over_the_index(tmp_path, knowledge_of,
                                                            idx_and_defs):
    idx, defs = idx_and_defs
    path = str(tmp_path / "out" / store.INDEX_FILE)
    knowledge = knowledge_of({"files": []})
    _record_beside(path, knowledge.stamp(), "recorded1")
    store.save(path, idx, defs, knowledge.stamp(), "p", "fromindex")
    assert build.commit_of_analysis(path, knowledge.stamp()) == "recorded1"


def test_a_record_of_another_analysis_falls_back_to_the_index(tmp_path, knowledge_of,
                                                              idx_and_defs):
    idx, defs = idx_and_defs
    path = str(tmp_path / "out" / store.INDEX_FILE)
    knowledge = knowledge_of({"files": []})
    _record_beside(path, "another stamp", "recorded1")
    store.save(path, idx, defs, knowledge.stamp(), "p", "fromindex")
    build.build_and_save(knowledge, "p", path)
    assert store.commit_in_index(path) == "fromindex"


# == analyse_and_index: codetwine without the documents, then the index ======
def test_analyse_and_index_records_the_commit_and_builds_the_index(tmp_path, monkeypatch):
    """What a run that moved the repository does: no LLM, and the index says HEAD."""
    repo = _git_repo(tmp_path / "repo")
    out = tmp_path / "out"
    call_list = []

    def fake_run_codetwine(project_dir, output_dir, with_doc=True, exclude_list=None):
        call_list.append((with_doc, exclude_list))
        write_knowledge_db({"files": []}, str(out / "repo" / knowledge_mod.KNOWLEDGE_FILE),
                           "repo")

    monkeypatch.setattr(knowledge_mod, "run_codetwine", fake_run_codetwine)
    path = build.analyse_and_index(str(repo), str(out), ["vendor"])
    build.analyse_and_index(str(repo), str(out), with_doc=True)

    assert call_list == [(False, ["vendor"]), (True, None)]
    assert path == str(out / "repo" / knowledge_mod.KNOWLEDGE_FILE)
    head = gitcmd.head_commit(str(repo))
    index_path = str(out / "repo" / store.INDEX_FILE)
    assert store.commit_in_index(index_path) == head
    assert knowledge_mod.analysis_commit(path, store.load(index_path).source_stamp) == head


def test_a_file_with_neither_a_definition_nor_a_summary_is_a_node_and_not_a_document(
        tmp_path, knowledge_of):
    """An analysis lists every text file; one with no language has no definitions. Its
    path words alone would outscore a described file, so it is not searched."""
    knowledge = knowledge_of({
        "project_name": "p",
        "files": [{"file": "p/Makefile/Makefile",
                   "file_dependencies": {"definitions": [], "callee_usages": [],
                                         "caller_usages": []}},
                  {"file": "p/notes_md/notes.md",
                   "file_dependencies": {"definitions": [], "callee_usages": [],
                                         "caller_usages": []},
                   "doc": {"sections": [], "summary": "how the build works"}}],
    })
    idx = build.build_index(knowledge, "p")
    assert idx["nodes"]["defs"] == {}
    assert idx["nodes"]["files"]["p/Makefile/Makefile"] == {"summary": "", "doc_hash": "",
                                                            "defs": []}
    assert "makefile" not in idx["index"]["file"]["postings"]
    assert "notes" in idx["index"]["file"]["postings"]
    index = _saved(tmp_path, idx, {}, project="p")
    assert index.primary_ext == ""
    index.close()


def test_an_index_with_no_definition_records_the_language_its_files_are_in(tmp_path,
                                                                           knowledge_of):
    no_def = {"definitions": [], "callee_usages": [], "caller_usages": []}
    knowledge = knowledge_of({
        "project_name": "p",
        "files": [{"file": "p/models_sql/a.sql", "file_dependencies": no_def},
                  {"file": "p/models_sql/b.sql", "file_dependencies": no_def},
                  {"file": "p/models_yml/schema.yml", "file_dependencies": no_def}],
    })
    index = _saved(tmp_path / "idx", build.build_index(knowledge, "p"), {}, project="p")
    assert index.primary_ext == "sql"
    index.close()


def test_looking_up_no_definition_at_all_asks_the_database_nothing(fake_index):
    assert fake_index.file_by_def_dict([]) == {}


# == what a search covers ====================================================

def test_the_average_length_follows_whether_constants_are_searched(tmp_path,
                                                                   monkeypatch):
    """BM25 normalises length against the documents the search actually scores."""
    defs = {
        "app/src/config_js/config.js::API_BASE": {
            "name": "API_BASE", "type": "lexical_declaration",
            "file": "app/src/config_js/config.js", "lines": [1, 1],
            "signature": "const API_BASE = 1;", "doc": ""},
        "app/src/config_js/config.js::loadConfig": {
            "name": "loadConfig", "type": "function_declaration",
            "file": "app/src/config_js/config.js", "lines": [3, 9],
            "signature": "function loadConfig() {", "doc": "reads and returns it"},
    }
    index = write_index(str(tmp_path / "i.sqlite"), defs, {}, project="app")
    monkeypatch.setattr(config, "INCLUDE_CONSTANT", False)
    without = index.avg_len("def")
    monkeypatch.setattr(config, "INCLUDE_CONSTANT", True)
    assert index.avg_len("def") != without
    index.close()


def test_the_keys_of_a_search_that_hit_everything_come_back(tmp_path, monkeypatch):
    """SQLite takes a bounded number of values in one statement."""
    monkeypatch.setattr(store, "_MAX_SQL_VARIABLE", 7)
    defs = {f"p/a_py/a.py::f{i}": {
        "name": f"f{i}", "type": "function_definition", "file": "p/a_py/a.py",
        "lines": [1, 2], "signature": "", "doc": "text"} for i in range(50)}
    index = write_index(str(tmp_path / "many.sqlite"), defs, {})
    assert len(index.key_dict("def", list(range(0, 50)))) == 50
    index.close()


def test_the_files_of_more_definitions_than_one_statement_takes_come_back(tmp_path,
                                                                          monkeypatch):
    """file_by_def_dict() asks in blocks too, under a connection's variable limit."""
    monkeypatch.setattr(store, "_MAX_SQL_VARIABLE", 7)
    defs = {f"p/a_py/a.py::f{i}": {
        "name": f"f{i}", "type": "function_definition", "file": "p/a_py/a.py",
        "lines": [1, 2], "signature": "", "doc": "text"} for i in range(50)}
    index = write_index(str(tmp_path / "many.sqlite"), defs, {})
    index._connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 10)
    assert index.file_by_def_dict(defs) == {key: "p/a_py/a.py" for key in defs}
    index.close()


def test_the_design_documents_hash_is_written(tmp_path, java_pk, knowledge_of):
    java_pk["files"][0]["doc"]["source_hash"] = "ab12"
    idx = build.build_index(knowledge_of(java_pk), "app")
    file = java_pk["files"][0]["file"]
    index = _saved(tmp_path, idx, idx["nodes"]["defs"], project="app")
    assert index.file_doc_hash(file) == "ab12"
    assert index.file_doc_hash("app/nosuch_java/nosuch.java") == ""
    index.close()
