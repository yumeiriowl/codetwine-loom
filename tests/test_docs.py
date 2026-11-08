"""Index step 3.

The checks use the synthetic Java data. Python data has no same-name collisions, so a
bare key and a #<type>#<start line> key coincide there and the defect of rebuilding keys
as f["file"] + "::" + name instead of using the name lookup cannot be detected.
"""

import pytest

from codetwine_loom.index.docs import attach_doc, file_doc, heading_name, split_by_title
from codetwine_loom.index.graph import (build_def_node_dict, def_by_file_dict,
                                         key_by_file_name_dict)


def _attach(knowledge, defs):
    """Run step 3 over an analysis, as build_index does."""
    doc_dict = {f["file"]: file_doc(f) for f in knowledge.iter_file()}
    return attach_doc(doc_dict, defs, key_by_file_name_dict(defs), def_by_file_dict(defs))


def test_split_by_heading_splits_at_every_heading():
    content = "## `foo`\nwhat foo does\n\n### `bar(x)`\nwhat bar does"
    body_dict = dict(split_by_title(content))
    assert list(body_dict) == ["foo", "bar"]
    assert "what foo does" in body_dict["foo"]
    assert "what bar does" in body_dict["bar"]


def test_split_by_heading_accepts_heading_levels_1_to_4():
    content = "# `a`\nA\n\n#### `d`\nD"
    assert [name for name, _ in split_by_title(content)] == ["a", "d"]


def test_keys_by_name_collects_the_split_keys_under_the_same_name(
        java_pk, knowledge_of):
    defs = build_def_node_dict(knowledge_of(java_pk))
    file = java_pk["files"][0]["file"]
    by_name = key_by_file_name_dict(defs)[file]
    assert len(by_name["Config"]) == 3  # 1 class + 2 constructors


def test_a_heading_match_reaches_every_split_key(java_pk, knowledge_of):
    """An implementation rebuilding f["file"] + "::" + name fails on this one."""
    knowledge = knowledge_of(java_pk)
    defs = build_def_node_dict(knowledge)
    _attach(knowledge, defs)
    file = java_pk["files"][0]["file"]
    for key in (f"{file}::Config",
                f"{file}::Config#constructor_declaration#5",
                f"{file}::Config#constructor_declaration#9"):
        assert "configuration holder" in defs[key]["doc"]


def test_the_backquote_match_picks_up_a_definition_without_a_heading(
        java_pk, knowledge_of):
    """API_BASE has no heading in definitions and is picked up from overview."""
    knowledge = knowledge_of(java_pk)
    defs = build_def_node_dict(knowledge)
    _attach(knowledge, defs)
    doc = defs[f"{java_pk['files'][0]['file']}::API_BASE"]["doc"]
    assert "default endpoint" in doc


def test_a_definition_nothing_matched_gets_the_whole_file_summary(
        java_pk, knowledge_of):
    """The whole file summary is given, not cut on the first line; newlines fold to
    single spaces.
    """
    java_pk["files"][0]["file_dependencies"]["definitions"].append(
        {"name": "orphan", "type": "method_declaration", "start_line": 17,
         "end_line": 18, "context": "void orphan() {"})
    knowledge = knowledge_of(java_pk)
    defs = build_def_node_dict(knowledge)
    _attach(knowledge, defs)
    doc = defs[f"{java_pk['files'][0]['file']}::orphan"]["doc"]
    assert doc == "**Config.java** holds client settings. Second line."


def test_no_definition_is_left_with_an_empty_doc(java_pk, knowledge_of):
    knowledge = knowledge_of(java_pk)
    defs = build_def_node_dict(knowledge)
    _attach(knowledge, defs)
    assert all(d["doc"] for d in defs.values())


def test_the_backquote_match_is_limited_to_that_files_definitions(
        java_pk, knowledge_of):
    """Definition names from other files (common words like str or None) stay out."""
    java_pk["files"][0]["doc"]["sections"][0]["content"] += \
        "\n\nThe helper `unrelated_name` lives elsewhere."
    knowledge = knowledge_of(java_pk)
    defs = build_def_node_dict(knowledge)
    _attach(knowledge, defs)
    assert not any("unrelated_name" in d["doc"] for d in defs.values())


def test_file_doc_is_empty_for_a_file_whose_design_document_was_not_generated():
    """A build with docs switched off writes no "doc" key at all."""
    assert file_doc({"file": "a.py"}) == {"summary": "", "sections": [], "source_hash": ""}


def test_file_doc_fills_in_whichever_member_is_missing():
    assert file_doc({"doc": {"summary": "s"}}) == {"summary": "s", "sections": [],
                                                   "source_hash": ""}
    assert file_doc({"doc": {"sections": []}}) == {"summary": "", "sections": [],
                                                   "source_hash": ""}


def test_file_doc_carries_the_source_hash():
    assert file_doc({"doc": {"source_hash": "ab"}})["source_hash"] == "ab"


def test_attach_docs_runs_over_a_file_that_has_no_design_document(
        java_pk, knowledge_of):
    """A file with no "doc" key, as a build made with docs switched off leaves, is
    passed over without an error."""
    del java_pk["files"][0]["doc"]
    knowledge = knowledge_of(java_pk)
    defs = build_def_node_dict(knowledge)
    _attach(knowledge, defs)
    assert len(defs) == 5
    assert all(d["doc"] == "" for d in defs.values())


# == headings that name the owner =========================================

@pytest.mark.parametrize("title, name_set, want", [
    ("load", {"load"}, ("load", "")),
    ("Store.close", {"close"}, ("close", "Store")),
    ("ns::Store::close", {"close"}, ("close", "Store")),
    ("Store#close", {"close"}, ("close", "Store")),
    ("Store.close", {"open"}, ("Store.close", "")),
    ("pkg.name", {"pkg.name", "name"}, ("pkg.name", "")),
])
def test_heading_name_reads_the_owner_off_a_qualified_heading(title, name_set, want):
    assert heading_name(title, {name: name for name in name_set}) == want


@pytest.mark.parametrize("title, want", [
    ("main-para", ("Main-Para", "")),
    ("MAIN-PARA", ("Main-Para", "")),
    ("cust-record.cust-id", ("CUST-ID", "CUST-RECORD")),
    ("other", ("other", "")),
])
def test_heading_name_takes_a_name_in_either_case_where_the_language_ignores_it(title, want):
    """A design document of a COBOL file names a paragraph in whatever case it likes."""
    name_dict = {name.upper(): name for name in ("Main-Para", "CUST-RECORD", "CUST-ID")}
    assert heading_name(title, name_dict, str.upper) == want


def _two_owner_pk(heading_list: list[str], sep: str = ".") -> dict:
    """One file with classes A and B that both define close(), and the given headings."""
    def _d(name, kind, start, end):
        return {"name": name, "type": kind, "start_line": start, "end_line": end,
                "context": f"{name} {{"}
    content = "\n\n".join(f"## `{h}`\nabout {h.replace(sep, ' ')}" for h in heading_list)
    return {"project_name": "app", "files": [{
        "file": "app/src/s_ts/s.ts",
        "file_dependencies": {
            "definitions": [_d("A", "class_declaration", 1, 5),
                            _d("close", "method_definition", 2, 4),
                            _d("B", "class_declaration", 7, 11),
                            _d("close", "method_definition", 8, 10)],
            "callee_usages": [], "caller_usages": []},
        "doc": {"sections": [{"id": "design", "title": "Design", "content": content}],
                "summary": "two stores"}}]}


def _close_doc_by_line(knowledge) -> dict:
    """Run step 3 and return {start line: doc} for the two close() definitions."""
    defs = build_def_node_dict(knowledge)
    _attach(knowledge, defs)
    return {d["lines"][0]: d["doc"] for d in defs.values() if d["name"] == "close"}


def test_an_owner_in_the_heading_picks_the_definition_that_owner_encloses(knowledge_of):
    """The headings are written in the other order than the definitions."""
    by_line = _close_doc_by_line(knowledge_of(_two_owner_pk(["B.close", "A.close"])))
    assert "about A close" in by_line[2] and "about B close" not in by_line[2]
    assert "about B close" in by_line[8] and "about A close" not in by_line[8]


def test_a_cpp_style_owner_is_read_as_well(knowledge_of):
    by_line = _close_doc_by_line(
        knowledge_of(_two_owner_pk(["ns::B::close", "ns::A::close"], sep="::")))
    assert "about ns A close" in by_line[2] and "about ns B close" not in by_line[2]
    assert "about ns B close" in by_line[8] and "about ns A close" not in by_line[8]


def test_one_bare_heading_over_two_same_named_definitions_reaches_both(knowledge_of):
    by_line = _close_doc_by_line(knowledge_of(_two_owner_pk(["close"])))
    assert all("about close" in doc for doc in by_line.values())


def test_several_headings_of_one_owner_pair_in_order_with_its_definitions(knowledge_of):
    """Two overloads inside one class, each with its own qualified heading."""
    def _d(name, kind, start, end):
        return {"name": name, "type": kind, "start_line": start, "end_line": end,
                "context": f"{name} {{"}
    content = ("## `Config`\nthe holder\n\n## `Config.Config`\nthe default one\n\n"
               "## `Config.Config`\nthe one taking a timeout")
    pk = {"project_name": "app", "files": [{
        "file": "app/Config_java/Config.java",
        "file_dependencies": {
            "definitions": [_d("Config", "class_declaration", 1, 20),
                            _d("Config", "constructor_declaration", 3, 5),
                            _d("Config", "constructor_declaration", 7, 9)],
            "callee_usages": [], "caller_usages": []},
        "doc": {"sections": [{"id": "design", "title": "Design", "content": content}],
                "summary": "config"}}]}
    knowledge = knowledge_of(pk)
    defs = build_def_node_dict(knowledge)
    _attach(knowledge, defs)
    by_line = {d["lines"][0]: d["doc"] for d in defs.values()}
    assert "the holder" in by_line[1]
    assert "the default one" in by_line[3] and "timeout" not in by_line[3]
    assert "the one taking a timeout" in by_line[7] and "default" not in by_line[7]
