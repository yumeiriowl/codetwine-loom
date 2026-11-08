"""COBOL: the line reader, the names, the index, the search and the reading tools.

The definitions are the ones codetwine reads from fixed-format lines with sequence
numbers and an identification area: a program whose range starts at IDENTIFICATION
DIVISION, records with the items under them, and names in any case. The uses between
them are synthetic.
"""

import importlib.util
import os
import pathlib
import re
import subprocess

import pytest

from codetwine.config.settings import EXT_TO_LANGUAGE_DICT
from codetwine.extractors.bms_source import read_bms_source
from codetwine.extractors.cobol_source import read_cobol_source

from codetwine_loom import cobol, config, progress
from codetwine_loom.index import build, store
from codetwine_loom.index.graph import (add_def_node, add_hold_name, build_def_node_dict,
                                         def_by_file_dict, mark_record)
from codetwine_loom.index.indexer import build_bm25, name_token_list, tokenize
from codetwine_loom.languages import (DEF_EXAMPLE_DICT, LANG_DICT, edit_paragraph,
                                       file_lang, lang_of, name_fold, name_key,
                                       set_file_lang)
from codetwine_loom.search import query
from codetwine_loom.search import scope as scope_mod
from codetwine_loom.tools import edit, source

from conftest import git_repo

# The identification area of a fixed-format line, from column 73 on
_IDENT = " " * 10


def _fixed(sequence: str, code: str, ident: str = "MAIN0001") -> str:
    """Build one fixed-format line: sequence area, indicator blank, code to column 72."""
    return f"{sequence} {code}".ljust(72) + ident


_MAIN_LINE_LIST = [
    _fixed("000100", "IDENTIFICATION DIVISION.", "MAIN"),
    _fixed("000200", "PROGRAM-ID. MAIN."),
    _fixed("000300", "DATA DIVISION."),
    _fixed("000400", "WORKING-STORAGE SECTION."),
    _fixed("000500", "COPY CUSTREC."),
    _fixed("000600", "01  WS-A                PIC 9."),
    _fixed("000700", "01  WS-REC."),
    _fixed("000800", "    05  WS-X            PIC X."),
    _fixed("000900", "PROCEDURE DIVISION."),
    _fixed("001000", "MAIN-PARA."),
    _fixed("001100", "    PERFORM Load-Cust"),
    _fixed("001200", "    MOVE 5 TO cust-acct-id"),
    _fixed("001300", "    STOP RUN."),
    _fixed("001400", "Load-Cust."),
    _fixed("001500", "    MOVE SPACES TO WS-X."),
]

_CUSTREC_LINE_LIST = [
    "       01  CUST-RECORD.",
    "           05  CUST-ACCT-ID        PIC 9(11).",
    "           05  顧客番号            PIC X(10).",
]

_MAIN_PATH = "p/src/MAIN_cbl/MAIN.cbl"
_CUSTREC_PATH = "p/copy/CUSTREC_cpy/CUSTREC.cpy"


def _definition_list(line_list: list[str], ext: str = "cbl") -> list[dict]:
    """Return the definitions[] codetwine writes for a COBOL file or a BMS source."""
    text = "\n".join(line_list) + "\n"
    cobol_source = (read_bms_source(text) if ext == "bms"
                    else read_cobol_source(text, EXT_TO_LANGUAGE_DICT[ext]))
    entry_list = []
    for d in cobol_source.definition_list:
        entry = {"name": d.name, "type": d.type, "start_line": d.start_line,
                 "end_line": d.end_line}
        entry.update({key: getattr(d, key) for key in ("name_line", "level", "is_group")
                      if getattr(d, key) is not None})
        entry["context"] = "\n".join(line_list[d.start_line - 1:d.end_line])
        entry_list.append(entry)
    return entry_list


def _usage(line: int, name: str, definition_list: list[dict]) -> dict:
    """Return the usage entry codetwine writes for a name used on one line."""
    target = next(d for d in definition_list if d["name"].upper() == name.upper())
    return {"lines": [line], "name": name, "target_name": target["name"],
            "target_start_line": target["start_line"]}


def _pk() -> dict:
    """The analysis of a program and the copybook it copies."""
    main, rec = _MAIN_LINE_LIST, _CUSTREC_LINE_LIST
    main_definition_list = _definition_list(main)
    return {"project_name": "p", "files": [
        {"file": _MAIN_PATH, "file_dependencies": {
            "definitions": main_definition_list,
            "callee_usages": [{**_usage(12, "CUST-ACCT-ID", _definition_list(rec)),
                               "from": _CUSTREC_PATH}],
            "same_file_usages": [_usage(11, "Load-Cust", main_definition_list),
                                 _usage(15, "WS-X", main_definition_list)],
            "caller_usages": []}},
        {"file": _CUSTREC_PATH, "file_dependencies": {
            "definitions": _definition_list(rec),
            "callee_usages": [], "same_file_usages": [], "caller_usages": []}},
    ]}


@pytest.fixture
def cobol_index(tmp_path, monkeypatch, knowledge_of):
    """The index of _pk(), loaded for the search and the tools, over real files."""
    root = tmp_path / "p"
    (root / "src").mkdir(parents=True)
    (root / "copy").mkdir()
    (root / "src" / "MAIN.cbl").write_text("\n".join(_MAIN_LINE_LIST) + "\n",
                                           encoding="utf-8")
    (root / "copy" / "CUSTREC.cpy").write_text("\n".join(_CUSTREC_LINE_LIST) + "\n",
                                               encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    index_path = str(tmp_path / "code_index.sqlite")
    index_dict = build.build_and_save(knowledge_of(_pk()), "p", index_path)
    scope_mod.load_index(index_path, "p")
    monkeypatch.setattr(source, "project_name", "p")
    yield index_dict
    scope_mod.INDEX.close()
    config.set_primary_ext("py")


# == the line reader ==========================================================

def test_the_code_of_a_fixed_format_line_leaves_out_the_sequence_and_identification():
    line_list = [_fixed("000200", "PROGRAM-ID. MAIN."), "000300* a comment line"]
    assert [text.strip() for text in cobol.code_text_list(line_list)] == \
        ["PROGRAM-ID. MAIN.", ""]


def test_a_free_format_line_is_read_whole():
    assert cobol.code_text_list(["MOVE A TO B.", "MAIN-PARA."]) == \
        ["MOVE A TO B.", "MAIN-PARA."]


def test_two_lines_are_the_same_declaration_whatever_the_numbers_case_and_spaces():
    before = _fixed("001000", "MAIN-PARA.", "MAIN0010")
    after = _fixed("002000", "Main-Para.", "CHG00001")
    assert cobol.line_key_list([before]) == cobol.line_key_list([after]) == ["MAIN-PARA."]


# == names ====================================================================

def test_a_cobol_name_is_compared_in_upper_case_and_a_python_name_as_it_is():
    assert name_key("src/MAIN.cbl", "Main-Para") == "MAIN-PARA"
    assert name_key("src/MAIN.CBL", "Main-Para") == "MAIN-PARA"
    assert name_key("app/main.py", "Main") == "Main"
    assert name_fold("copy/X.cpy") is str.upper and name_fold("a.java") is str


def test_only_cobol_csharp_c_and_cpp_have_a_note_for_the_stages_that_write_code():
    assert "## COBOL source" in edit_paragraph("cbl", set())
    assert "## C# source" in edit_paragraph("cs", set())
    assert "## C and C++ tests" in edit_paragraph("c", set()) == edit_paragraph("cpp", set())
    assert all(edit_paragraph(ext, {"run_test"}) == ""
               for ext, lang_config in LANG_DICT.items()
               if not lang_config.column_layout
               and lang_config not in (LANG_DICT["cs"], LANG_DICT["c"], LANG_DICT["cpp"]))
    assert DEF_EXAMPLE_DICT["cbl"] == "1000-LOAD-CONFIG"


# == the words of the index ===================================================

def test_a_hyphen_joined_cobol_name_is_one_word_besides_its_parts():
    assert tokenize("       05 CUST-ACCT-ID   PIC 9(11).", "cbl") == \
        ["cust-acct-id", "cust", "acct", "id"]


def test_a_query_keeps_a_hyphen_joined_name_whole_and_python_code_does_not():
    assert tokenize("CUST-ACCT-ID") == ["cust-acct-id", "cust", "acct", "id"]
    assert tokenize("read-only mode", "py") == ["read", "mode"]


def test_a_name_outside_ascii_is_one_word_whatever_its_length():
    assert tokenize("       05 顧客番号 PIC X(10).", "cbl") == ["顧客番号"]
    assert tokenize("       05 額 PIC 9.", "cbl") == ["額"]


def test_the_cobol_keywords_are_left_out_of_the_code_and_kept_in_the_prose():
    assert tokenize("PROGRAM-ID. MAIN. WORKING-STORAGE", "cbl") == ["program", "id", "main",
                                                                    "working", "storage"]
    assert "working-storage" in tokenize("WORKING-STORAGE", "cbl", keep_keyword=True)


def test_the_names_a_definition_holds_are_whole_words_that_do_not_count_toward_its_length():
    assert name_token_list(["CUST-ACCT-ID", "WS-A", "FILLER"], "cbl") == ["cust-acct-id",
                                                                         "ws-a"]
    index_dict = build_bm25({"p/a_cbl/a.cbl::REC": ("01 REC.", "", ["CUST-ACCT-ID"])})
    assert index_dict["len"]["p/a_cbl/a.cbl::REC"] == 1
    assert ["p/a_cbl/a.cbl::REC", 1] in index_dict["postings"]["cust-acct-id"]
    assert "cust" not in index_dict["postings"]


# == the index ================================================================

def test_the_analysis_gives_the_declaration_line_level_and_group():
    assert [(d["name"], d.get("name_line"), d.get("level"), d.get("is_group"))
            for d in _definition_list(_MAIN_LINE_LIST)] == [
        ("MAIN", 2, None, None), ("WS-A", 6, 1, False), ("WS-REC", 7, 1, True),
        ("WS-X", 8, 5, False), ("MAIN-PARA", 10, None, None),
        ("Load-Cust", 14, None, None)]


def _record_dict(path: str, line_list: list[str], ext: str = "cbl") -> dict:
    """Return {name: whether mark_record() takes it as a record} for one file."""
    def_dict: dict = {}
    add_def_node(def_dict, {"file": path, "file_dependencies": {
        "definitions": _definition_list(line_list, ext)}})
    mark_record(def_dict, def_by_file_dict(def_dict))
    return {node["name"]: node["record"] for node in def_dict.values() if "record" in node}


def test_a_record_is_a_group_no_item_of_a_lower_level_holds():
    line_list = ["       01  REC-A.", "           05  GRP-B.",
                 "               10  FLD-C  PIC X.", "           05  PTR-D  USAGE POINTER.",
                 "       01  REC-E REDEFINES REC-A.", "           05  FLD-F  PIC X(2).",
                 "       77  WS-COUNT  PIC 9.", "       01  WS-PTR  USAGE POINTER."]
    assert _record_dict("p/copy/R_cpy/R.cpy", line_list) == {
        "REC-A": True, "GRP-B": False, "FLD-C": False, "PTR-D": False, "REC-E": True,
        "FLD-F": False, "WS-COUNT": False, "WS-PTR": False}


def test_a_copybook_of_items_under_a_record_elsewhere_has_its_own_records():
    line_list = ["           05  GRP-A.", "               10  FLD-B  PIC X.",
                 "           05  FLD-C  PIC X."]
    assert _record_dict("p/copy/F_cpy/F.cpy", line_list) == {
        "GRP-A": True, "FLD-B": False, "FLD-C": False}


def test_the_maps_of_a_bms_source_are_records_that_hold_their_fields():
    path = "p/bms/TESTSET_bms/TESTSET.bms"
    def_dict: dict = {}
    add_def_node(def_dict, {"file": path, "file_dependencies": {
        "definitions": _definition_list(_BMS_TEXT.split("\n"), "bms")}})
    by_file_dict = def_by_file_dict(def_dict)
    mark_record(def_dict, by_file_dict)
    add_hold_name(def_dict, by_file_dict)
    searchable = {node["name"] for key, node in def_dict.items()
                  if not store.is_constant(key, node)}
    assert searchable == {"TESTMAPI", "TESTMAPO"}
    map_node = def_dict[f"{path}::TESTMAPI"]
    assert map_node["signature"].startswith("TESTMAP  DFHMDI")
    assert {"CUSTNOI", "DAYI", "DATEO"} <= set(map_node["hold_name_list"])
    assert name_key(path, "custnoi") == "CUSTNOI"


def test_the_bms_sources_count_as_cobol_when_there_is_cobol():
    def node(file):
        return {"file": file, "type": "data_description", "name": "X", "record": True}
    def_dict = {f"p/m{i}_bms/m{i}.bms::X": node(f"p/m{i}_bms/m{i}.bms") for i in range(3)}
    assert store.primary_ext(def_dict) == "bms"
    def_dict.update({"p/r_cpy/r.cpy::X": node("p/r_cpy/r.cpy"),
                     "p/s_sql/s.sql::X": {"file": "p/s_sql/s.sql", "name": "X",
                                          "type": "create_table"},
                     "p/t_sql/t.sql::X": {"file": "p/t_sql/t.sql", "name": "X",
                                          "type": "create_table"}})
    assert store.primary_ext(def_dict) == "cbl"


def test_jcl_and_csd_files_count_as_cobol_when_there_is_cobol():
    def node(file, def_type):
        return {"file": file, "type": def_type, "name": "X"}
    def_dict = {f"p/j{i}_jcl/j{i}.jcl::X": node(f"p/j{i}_jcl/j{i}.jcl", "job_statement")
                for i in range(3)}
    def_dict.update({f"p/c{i}_prc/c{i}.prc::X": node(f"p/c{i}_prc/c{i}.prc", "proc_statement")
                     for i in range(2)})
    def_dict["p/a_csd/a.csd::X"] = node("p/a_csd/a.csd", "define_transaction")
    def_dict["p/j0_jcl/j0.jcl::HLQ"] = node("p/j0_jcl/j0.jcl", "symbolic_parameter")
    assert store.primary_ext(def_dict) == "jcl"
    assert store.is_constant("p/j0_jcl/j0.jcl::HLQ", def_dict["p/j0_jcl/j0.jcl::HLQ"])
    def_dict["p/m_cbl/m.cbl::M"] = node("p/m_cbl/m.cbl", "program_definition")
    assert store.primary_ext(def_dict) == "cbl"
    assert name_key("p/j0_jcl/j0.jcl", "nightly") == "NIGHTLY"
    assert name_key("p/a_csd/a.csd", "tx01") == "TX01"


def test_an_analysis_without_levels_has_no_record_and_says_so_once(caplog):
    def_dict: dict = {}
    for path in (_CUSTREC_PATH, "p/copy/OTHER_cpy/OTHER.cpy"):
        add_def_node(def_dict, {"file": path, "file_dependencies": {"definitions": [
            {key: value for key, value in d.items() if key not in ("level", "is_group")}
            for d in _definition_list(_CUSTREC_LINE_LIST, "cpy")]}})
    mark_record(def_dict, def_by_file_dict(def_dict))
    assert not any(node["record"] for node in def_dict.values())
    assert [record.getMessage() for record in caplog.records
            if "no level" in record.getMessage()] == [
        ("the analysis gives the data items of 2 file(s) no level, e.g. "
         f"{_CUSTREC_PATH}; none of them is taken as a record. Remove the analysis and "
         "build it again to have them")]


def test_a_name_line_outside_the_definition_is_not_its_declaration():
    def_dict: dict = {}
    entry = {"name": "SIZE", "type": "paragraph_header", "start_line": 3, "end_line": 4,
             "name_line": 9, "context": "* a comment\n       SIZE."}
    add_def_node(def_dict, {"file": "p/a_cbl/a.cbl",
                            "file_dependencies": {"definitions": [entry]}})
    assert def_dict["p/a_cbl/a.cbl::SIZE"]["signature_line"] == 4


def test_the_records_are_searched_and_the_other_data_items_are_not(cobol_index):
    def_dict = cobol_index["nodes"]["defs"]
    searchable = {def_dict[key]["name"] for key in def_dict
                  if not store.is_constant(key, def_dict[key])}
    assert searchable == {"MAIN", "WS-REC", "MAIN-PARA", "Load-Cust", "CUST-RECORD"}


def test_a_left_out_item_goes_to_the_searchable_definition_around_it(cobol_index):
    def_dict = cobol_index["nodes"]["defs"]
    assert def_dict[f"{_CUSTREC_PATH}::CUST-RECORD"]["hold_name_list"] == \
        ["CUST-ACCT-ID", "顧客番号"]
    assert def_dict[f"{_MAIN_PATH}::WS-REC"]["hold_name_list"] == ["WS-X"]
    assert def_dict[f"{_MAIN_PATH}::MAIN"]["hold_name_list"] == ["WS-A"]


def test_the_declaration_line_is_recorded(cobol_index):
    node = scope_mod.INDEX.definition(f"{_MAIN_PATH}::MAIN")
    assert node["lines"] == [1, 15] and node["signature_line"] == 2
    assert node["signature"].strip().startswith("000200 PROGRAM-ID. MAIN.")


def test_the_primary_language_is_cobol(cobol_index):
    assert scope_mod.INDEX.primary_ext in ("cbl", "cpy")
    assert config.PRIMARY_EXT == scope_mod.INDEX.primary_ext


def test_the_extensions_of_one_language_are_counted_together():
    def node(file, type_="paragraph_header"):
        return {"file": file, "type": type_, "name": "X"}
    def_dict = {"p/a_cbl/a.cbl::X": node("p/a_cbl/a.cbl"),
                "p/b_cpy/b.cpy::X": node("p/b_cpy/b.cpy", "program_definition"),
                "p/c_cpy/c.cpy::X": node("p/c_cpy/c.cpy", "program_definition"),
                "p/d_sql/d.sql::X": node("p/d_sql/d.sql", "create_table"),
                "p/e_sql/e.sql::X": node("p/e_sql/e.sql", "create_table")}
    assert store.primary_ext(def_dict) == "cpy"


def test_a_cobol_name_is_found_in_either_case_and_a_python_name_is_not(cobol_index):
    assert [entry["key"] for entry in scope_mod.INDEX.def_list_by_name("main-para")] == \
        [f"{_MAIN_PATH}::MAIN-PARA"]
    assert [entry["key"] for entry in scope_mod.INDEX.def_list_by_name("LOAD-CUST")] == \
        [f"{_MAIN_PATH}::Load-Cust"]
    assert scope_mod.INDEX.non_ascii_name_list() == ["顧客番号"]


def test_a_python_name_in_another_case_is_another_name(tmp_path, knowledge_of):
    pk = {"project_name": "p", "files": [{"file": "p/a_py/a.py", "file_dependencies": {
        "definitions": [{"name": "Load", "type": "function_definition", "start_line": 1,
                         "end_line": 2, "context": "def Load():\n    pass"}],
        "callee_usages": [], "same_file_usages": [], "caller_usages": []}}]}
    path = str(tmp_path / "py_index.sqlite")
    build.build_and_save(knowledge_of(pk), "p", path)
    index = store.load(path)
    try:
        assert index.def_list_by_name("load") == []
        assert len(index.def_list_by_name("Load")) == 1
    finally:
        index.close()


_DCL_PATH = "p/dcl/DCLX_dcl/DCLX.dcl"

_DCL_LINE_LIST = ["       01  DCL-X.", "           05  DCL-A  PIC X."]


@pytest.fixture
def dcl_index(tmp_path, monkeypatch, knowledge_of):
    """The index of a program and a DCLGEN member the analysis read as a copybook."""
    root = tmp_path / "p"
    (root / "dcl").mkdir(parents=True)
    (root / "dcl" / "DCLX.dcl").write_text("\n".join(_DCL_LINE_LIST) + "\n",
                                           encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    pk = {"project_name": "p", "files": [
        {"file": _DCL_PATH, "file_dependencies": {
            "language": "cpy", "definitions": _definition_list(_DCL_LINE_LIST, "cpy"),
            "callee_usages": [], "same_file_usages": [], "caller_usages": []}},
        {"file": "p/src/A_cbl/A.Cbl", "file_dependencies": {
            "language": "cbl", "definitions": [], "callee_usages": [],
            "same_file_usages": [], "caller_usages": []}}]}
    index_path = str(tmp_path / "code_index.sqlite")
    index_dict = build.build_and_save(knowledge_of(pk), "p", index_path)
    set_file_lang({})
    scope_mod.load_index(index_path, "p")
    monkeypatch.setattr(source, "project_name", "p")
    yield index_dict
    scope_mod.INDEX.close()
    set_file_lang({})
    config.set_primary_ext("py")


def test_a_file_the_analysis_read_as_cobol_is_cobol_whatever_its_extension(dcl_index):
    assert dcl_index["file_lang"] == {_DCL_PATH: "cpy"}
    assert lang_of("dcl/DCLX.dcl") is lang_of(_DCL_PATH) is LANG_DICT["cpy"]
    assert lang_of("dcl/OTHER.dcl") is None
    assert scope_mod.INDEX.primary_ext == "cpy"
    def_dict = dcl_index["nodes"]["defs"]
    assert [def_dict[key]["name"] for key in def_dict
            if not store.is_constant(key, def_dict[key])] == ["DCL-X"]
    result = source.read_definition("dcl-a")
    assert result["ok"] and result["key"] == "dcl/DCLX.dcl::DCL-A"


def test_the_definitions_of_an_analysis_read_its_files_in_its_languages(knowledge_of):
    set_file_lang({})
    pk = {"project_name": "p", "files": [{"file": _DCL_PATH, "file_dependencies": {
        "language": "cpy", "definitions": _definition_list(_DCL_LINE_LIST, "cpy"),
        "callee_usages": [], "same_file_usages": [], "caller_usages": []}}]}
    try:
        build_def_node_dict(knowledge_of(pk))
        assert name_key("dcl/DCLX.dcl", "dcl-a") == name_key(_DCL_PATH, "dcl-a") == "DCL-A"
    finally:
        set_file_lang({})


def test_an_analysis_without_the_language_reads_a_file_by_its_extension():
    assert file_lang({"file": _DCL_PATH, "file_dependencies": {}}) == ""
    assert file_lang({"file": "p/src/A_cbl/A.Cbl",
                      "file_dependencies": {"language": "cbl"}}) == ""
    assert file_lang({"file": _DCL_PATH, "file_dependencies": {"language": "cpy"}}) == "cpy"


# == the search ===============================================================

@pytest.mark.parametrize("query_text", ["CUST-ACCT-ID", "cust-acct-id", "顧客番号"])
def test_a_field_name_finds_the_record_that_holds_it(cobol_index, query_text):
    scope_dict = scope_mod.search_scope_from_query(query_text)
    assert scope_dict["defs"][0]["key"] == "copy/CUSTREC.cpy::CUST-RECORD"


def test_a_backquoted_name_outside_ascii_is_a_name_of_the_request():
    assert query.literal_term_list("`顧客番号` の桁を 12 にして") == ["顧客番号"]
    assert query.literal_term_list("顧客番号の桁を 12 にして") == []


def test_a_name_of_the_index_written_in_a_japanese_request_is_searched_for(cobol_index):
    assert scope_mod.text_name_list("顧客番号の桁を 12 にして") == ["顧客番号"]
    assert scope_mod.with_text_name("customer number digits", "顧客番号の桁を直す") == \
        "customer number digits 顧客番号"
    assert scope_mod.with_text_name("顧客番号 digits", "顧客番号の桁を直す") == \
        "顧客番号 digits"


def test_search_again_takes_a_name_outside_ascii_that_the_index_holds(cobol_index):
    assert source.search_again("顧客番号")["ok"] is True
    assert source.search_again("存在しない語")["ok"] is False


# == the reading tools ========================================================

def test_read_definition_takes_a_cobol_name_in_either_case(cobol_index):
    result = source.read_definition("load-cust")
    assert result["ok"] and result["key"] == "src/MAIN.cbl::Load-Cust"
    assert "Load-Cust." in result["code"]


def test_graph_search_starts_from_a_cobol_name_in_either_case(cobol_index):
    result = source.graph_search("main-para", direction="outgoing")
    assert result["start"] == "src/MAIN.cbl::MAIN-PARA"
    assert "src/MAIN.cbl::Load-Cust" in {node["key"] for node in result["nodes"]}


def test_a_renumbered_program_is_placed_again_from_its_program_id_line(cobol_index,
                                                                       monkeypatch):
    path = "src/MAIN.cbl"
    monkeypatch.setattr(source.edit_mod, "write_path_set", lambda: {path})
    monkeypatch.setattr(source.edit_mod, "own_change_set", set)
    line_list = [_fixed("000050", "* a line added before it")] + [
        _fixed(f"{(i + 1) * 200:06d}", line[7:72].rstrip(), "CHANGED ")
        for i, line in enumerate(_MAIN_LINE_LIST)]
    (pathlib.Path(config.PROJECT_DIR) / path).write_text("\n".join(line_list) + "\n",
                                                          encoding="utf-8")
    node = scope_mod.INDEX.definition(f"{_MAIN_PATH}::MAIN")
    assert source.fresh_line_range(path, node["lines"], node["signature"],
                                   node["signature_line"]) == ([2, 16], False)


def test_a_decorated_python_function_keeps_its_decorator_in_the_range(tmp_path,
                                                                      monkeypatch):
    path = "cfg.py"
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(source.edit_mod, "write_path_set", lambda: {path})
    monkeypatch.setattr(source.edit_mod, "own_change_set", set)
    (tmp_path / path).write_text("# added\n@property\ndef size(self):\n    return 1\n",
                                 encoding="utf-8")
    assert source.fresh_line_range(path, [1, 3], "def size(self):", 2) == ([2, 4], False)


def test_grep_matches_a_cobol_file_in_either_case_and_keeps_its_columns(cobol_index):
    result = source.grep_source("load-cust")
    assert {hit["line"] for hit in result["hits"]} == {11, 14}
    assert result["hits"][0]["text"] == _MAIN_LINE_LIST[10]


def test_grep_keeps_matching_other_files_in_their_own_case(cobol_index):
    (pathlib.Path(config.PROJECT_DIR) / "notes.md").write_text("Load-Cust\nload-cust\n",
                                                                encoding="utf-8")
    result = source.grep_source("load-cust", glob="*.md")
    assert [hit["line"] for hit in result["hits"]] == [2]


def test_old_text_retyped_without_its_sequence_numbers_is_said_to_differ_there():
    src = "\n".join(_MAIN_LINE_LIST) + "\n"
    old_text = "       MAIN-PARA.\n           PERFORM Load-Cust"
    hint = edit._no_match_hint(src, old_text, "src/MAIN.cbl")
    assert "columns 1-6" in hint
    assert "columns 1-6" not in edit._no_match_hint(src, "       OTHER-PARA.", "src/MAIN.cbl")


def test_a_cobol_mismatch_in_spaces_is_said_to_move_the_columns():
    src = "       MOVE A TO B.\n"
    assert "columns" in edit._no_match_hint(src, "      MOVE A TO B.", "src/MAIN.cbl")
    assert "whitespace" in edit._no_match_hint(src, "      MOVE A TO B.", "a.py")


# == the static check (sandbox/check-cbl.py) ==================================

_CHECK_CBL_PATH = pathlib.Path(__file__).resolve().parent.parent / "sandbox" / "check-cbl.py"


@pytest.fixture(scope="module")
def check_cbl():
    """The check script, loaded as a module."""
    spec = importlib.util.spec_from_file_location("check_cbl", _CHECK_CBL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_exec_statement_becomes_continue_in_its_own_columns(check_cbl):
    text = "\n".join([
        "       PROCEDURE DIVISION.",
        "           EXEC CICS RETURN",
        "                TRANSID (WS-TRANID)",
        "           END-EXEC.",
        "           IF WS-RESP = DFHRESP(NOTFND)",
        "              CONTINUE",
        "           END-IF."])
    out = check_cbl.replace_exec(text, frozenset())
    assert out.split("\n") == [
        "       PROCEDURE DIVISION.",
        "           CONTINUE        ",
        "                                   ",
        "                   .",
        "           IF WS-RESP = 0              ",
        "              CONTINUE",
        "           END-IF."]


def test_an_sql_include_becomes_a_copy_of_the_file_the_repository_holds(check_cbl):
    text = "\n".join([
        "       WORKING-STORAGE SECTION.",
        "           EXEC SQL INCLUDE DCLTRTYP END-EXEC.",
        "           EXEC SQL INCLUDE SQLCA END-EXEC.",
        "           EXEC SQL DECLARE C1 CURSOR FOR",
        "                SELECT A FROM T END-EXEC.",
        "       01  WS-A PIC X."])
    out = check_cbl.replace_exec(text, frozenset({"DCLTRTYP"})).split("\n")
    assert out[1].strip() == "COPY DCLTRTYP." and len(out[1]) == len(text.split("\n")[1])
    assert out[2].strip() == out[3].strip() == out[4].strip() == ""
    assert out[5] == "       01  WS-A PIC X."


def test_a_copybook_of_another_extension_is_copied_where_cobc_finds_it(check_cbl,
                                                                     tmp_path, monkeypatch):
    file_dict = {
        "src/MAIN.cbl": "       COPY 'Incx.inc'.\n       COPY DATA1.\n       COPY NOTE.\n"
                        "       COPY 'custrec.cpy'.\n",
        "copy/CUSTREC.cpy": "       01  CUST-REC  PIC X.\n",
        "copy/INCX.inc": "       01  INC-X.\n           COPY NOEXT.\n",
        "copy/NOEXT": "       05  NOEXT-A  PIC X.\n",
        "copy/DATA1.txt": "       01  DATA1-A  PIC X.\n",
        "doc/NOTE.md": "# a note\n",
        "copy/UNUSED.inc": "       01  UNUSED-A  PIC X.\n"}
    for path, text in file_dict.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "mirror"
    dir_list, alias_dict, _ = check_cbl._mirror(sorted(file_dict),
                                                ["copy/CUSTREC.cpy", "src/MAIN.cbl"],
                                                str(root))
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) == [
        "copy/CUSTREC.cpy", "copy/DATA1.cpy", "copy/Incx.inc", "copy/NOEXT",
        "copy/custrec.cpy", "src/MAIN.cbl"]
    assert alias_dict == {"copy/Incx.inc": "copy/INCX.inc", "copy/DATA1.cpy": "copy/DATA1.txt",
                          "copy/custrec.cpy": "copy/CUSTREC.cpy"}
    assert dir_list == ["copy", "src"]


def test_a_literal_that_goes_on_to_the_next_line_is_not_a_copybook_name(check_cbl,
                                                                        tmp_path):
    path = tmp_path / "MAIN.cbl"
    path.write_text("       COPY 'OPEN\n      -    'X.cpy'.\n       COPY 'SHUT.cpy'.\n",
                    encoding="utf-8")
    assert [item for item in check_cbl._copy_name_list(str(path))
            if item[1] == check_cbl._LITERAL_KIND] == [("SHUT.cpy", check_cbl._LITERAL_KIND)]


def test_a_comment_and_a_literal_are_not_read_as_exec(check_cbl):
    text = "\n".join([
        "       PROCEDURE DIVISION.",
        "      *    EXEC CICS RETURN END-EXEC.",
        "           DISPLAY 'EXEC SQL END-EXEC'."])
    assert check_cbl.replace_exec(text, frozenset()) == text


@pytest.fixture
def cobol_repo(tmp_path, monkeypatch):
    """A repository of one program and one copybook, and a stand-in cobc that prints
    $COBC_OUT and exits 1. The trial of the arguments on CTLOOMCK.cbl prints
    $COBC_TRY_OUT, exits $COBC_TRY_STATUS (0 when unset) and is not written to
    $COBC_ARGS."""
    repo = pathlib.Path(git_repo(tmp_path / "repo", commit=False))
    (repo / "src").mkdir()
    (repo / "copy").mkdir()
    (repo / "src" / "MAIN.cbl").write_text("       IDENTIFICATION DIVISION.\n"
                                          "       PROGRAM-ID. MAIN.\n"
                                          "       DATA DIVISION.\n"
                                          "       WORKING-STORAGE SECTION.\n"
                                          "       COPY custrec.\n", encoding="utf-8")
    (repo / "copy" / "CUSTREC.cpy").write_text("       01  CUST-RECORD PIC X.\n",
                                               encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "cobc").write_text(
        '#!/bin/sh\n'
        'for arg; do last="$arg"; done\n'
        'if [ "$last" = CTLOOMCK.cbl ]; then\n'
        '    printf "%s\\n" "${COBC_TRY_OUT:-}" >&2; exit "${COBC_TRY_STATUS:-0}"\n'
        'fi\n'
        'echo "$@" >> "$COBC_ARGS"\n'
        'printf "%s\\n" "$COBC_OUT" >&2\nexit 1\n')
    (bin_dir / "cobc").chmod(0o755)
    monkeypatch.setattr(config, "PROJECT_DIR", str(repo))
    monkeypatch.setattr(config, "CHECK_COMMAND", ["python3", str(_CHECK_CBL_PATH)])
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("COBC_ARGS", str(tmp_path / "cobc_args.txt"))
    return repo


_RUN_CHECK = edit.run_check


def test_the_cobol_check_reports_the_errors_and_not_the_products_names(cobol_repo,
                                                                       monkeypatch):
    monkeypatch.setenv("COBC_OUT", "\n".join([
        "src/MAIN.cbl:9: error: 'WS-TYPO' is not defined",
        "src/MAIN.cbl:10: error: 'EIBCALEN' is not defined",
        "src/MAIN.cbl:11: error: DFHAID: No such file or directory",
        "src/MAIN.cbl:12: warning: something",
        "   12 | the quoted source line",
        "copy/custrec.cpy:1: error: syntax error, unexpected ."]))
    assert _RUN_CHECK() == {"src/MAIN.cbl: error: 'WS-TYPO' is not defined": 1,
                            "copy/CUSTREC.cpy: error: syntax error, unexpected .": 1}


def test_the_cobol_check_takes_no_undefined_name_where_a_copybook_is_missing(cobol_repo,
                                                                             monkeypatch):
    monkeypatch.setenv("COBC_OUT", "\n".join([
        "src/MAIN.cbl:5: error: SSMAP: No such file or directory",
        "src/MAIN.cbl:9: error: 'SSMAPO' is not defined"]))
    assert _RUN_CHECK() == {"src/MAIN.cbl: error: SSMAP: No such file or directory": 1}


def test_the_cobol_check_checks_the_programs_with_every_copybook_directory(cobol_repo,
                                                                           monkeypatch,
                                                                           tmp_path):
    monkeypatch.setenv("COBC_OUT", "")
    assert _RUN_CHECK() == {}
    argv = (tmp_path / "cobc_args.txt").read_text(encoding="utf-8").split()
    assert argv == ["-fsyntax-only", "-I", "copy", "-I", "src", "src/MAIN.cbl"]


def test_the_arguments_go_to_cobc_and_their_directories_after_the_repository_s(
        cobol_repo, monkeypatch, tmp_path):
    monkeypatch.setenv("COBC_OUT", "")
    monkeypatch.setattr(config, "CHECK_COMMAND", ["python3", str(_CHECK_CBL_PATH),
                                                  "-std=ibm", "-I", "lib", "-I/opt/copy"])
    assert _RUN_CHECK() == {}
    argv = (tmp_path / "cobc_args.txt").read_text(encoding="utf-8").split()
    assert argv == ["-fsyntax-only", "-std=ibm", "-I", "copy", "-I", "src",
                    "-I", f"{cobol_repo}/lib", "-I", "/opt/copy", "src/MAIN.cbl"]


def test_arguments_cobc_does_not_take_run_no_check(cobol_repo, monkeypatch):
    """A misspelt option would otherwise leave every program with no error line."""
    monkeypatch.setenv("COBC_TRY_STATUS", "1")
    monkeypatch.setenv("COBC_TRY_OUT", "configuration error: ibmm.conf: No such file")
    monkeypatch.setattr(config, "CHECK_COMMAND", ["python3", str(_CHECK_CBL_PATH),
                                                  "-std=ibmm"])
    assert _RUN_CHECK() is None


def _continued(text: str, mark: str = "X") -> str:
    """Return a BMS line continued on the next: the mark in column 72."""
    return text.ljust(71) + mark


_BMS_TEXT = "\n".join([
    "* a mapset",
    _continued("TESTSET  DFHMSD TYPE=&&SYSPARM,MODE=INOUT,LANG=COBOL,TIOAPFX=YES,"),
    "               DSATTS=(COLOR,HILIGHT)",
    "TESTMAP  DFHMDI SIZE=(24,80)",
    _continued("         DFHMDF POS=(1,1),LENGTH=20,INITIAL='It''s a title,"),
    "               here'",
    _continued("CUSTNO   DFHMDF POS=(2,1),LENGTH=8,PICIN='99999999',PICOUT='ZZZZZZZ9',", "*"),
    "               ATTRB=(UNPROT,NUM)       a remark",
    "ROW      DFHMDF POS=(3,1),LENGTH=10,OCCURS=5",
    "DAY      DFHMDF POS=(9,1),LENGTH=2,GRPNAME=DATE",
    "MONTH    DFHMDF POS=(9,3),LENGTH=2,GRPNAME=DATE",
    "         DFHMSD TYPE=FINAL",
    "         END"])


def test_a_bms_source_is_read_with_its_continuations_quotes_and_remarks(check_cbl):
    statement_list = check_cbl._bms_statement_list(_BMS_TEXT)
    assert statement_list[0] == ("TESTSET", "DFHMSD", "TYPE=&&SYSPARM,MODE=INOUT,LANG=COBOL,"
                                 "TIOAPFX=YES,DSATTS=(COLOR,HILIGHT)")
    quoted = "INITIAL='It''s a title,"
    assert statement_list[2][2] == ("POS=(1,1),LENGTH=20," + quoted
                                    + " " * (71 - 36 - len(quoted)) + "here'")
    assert statement_list[3][2].endswith("ATTRB=(UNPROT,NUM)")
    (bms_map,) = check_cbl.read_bms(_BMS_TEXT)["TESTSET"]
    assert (bms_map.name, bms_map.prefix, bms_map.attribute_list) == ("TESTMAP", True, ["C", "H"])
    assert [(f.name, f.occurs, f.picin, f.picout, [m.name for m in f.member_list])
            for f in bms_map.field_list] == [
        ("CUSTNO", 0, "99999999", "ZZZZZZZ9", []), ("ROW", 5, "X(10)", "X(10)", []),
        ("DATE", 0, "", "", ["DAY", "MONTH"])]


def test_the_symbolic_map_holds_the_names_a_program_uses(check_cbl):
    text = check_cbl.symbolic_map_text(check_cbl.read_bms(_BMS_TEXT)["TESTSET"])
    name_set = set(re.findall(r"^ {7,}\d\d  ([A-Z0-9-]+)", text, re.M)) - {"FILLER"}
    assert name_set == {"TESTMAPI", "TESTMAPO", "CUSTNOL", "CUSTNOF", "CUSTNOA", "CUSTNOI",
                        "CUSTNOC", "CUSTNOH", "CUSTNOO", "ROWL", "ROWF", "ROWA", "ROWI",
                        "ROWC", "ROWH", "ROWO", "DATEL", "DATEF", "DATEA", "DATEI", "DAYI",
                        "MONTHI", "DATEC", "DATEH", "DATEO", "DAYO", "MONTHO"}
    assert "FILLER OCCURS 5 TIMES." in text and "PIC 99999999." in text
    assert max(len(line) for line in text.splitlines()) <= 72


def test_a_mapset_with_a_name_cobol_cannot_hold_is_not_read(check_cbl):
    assert check_cbl.read_bms(_BMS_TEXT.replace("ROW   ", "RO#W  ")) == {}


def test_a_copy_of_a_mapset_only_the_bms_source_holds_copies_its_symbolic_map(
        cobol_repo, monkeypatch, tmp_path):
    (cobol_repo / "bms").mkdir()
    (cobol_repo / "bms" / "TESTSET.bms").write_text(_BMS_TEXT, encoding="utf-8")
    with (cobol_repo / "src" / "MAIN.cbl").open("a", encoding="utf-8") as fp:
        fp.write("       COPY TESTSET.\n")
    monkeypatch.setenv("COBC_OUT", "\n".join([
        "src/MAIN.cbl:9: error: 'CUSTNOZ' is not defined",
        "src/MAIN.cbl:10: error: 'NOT-THERE' is not defined",
        "bms/TESTSET.cpy:3: error: syntax error"]))
    assert _RUN_CHECK() == {"src/MAIN.cbl: error: 'NOT-THERE' is not defined": 1,
                            "bms/TESTSET.bms: error: syntax error": 1}
    argv = (tmp_path / "cobc_args.txt").read_text(encoding="utf-8").split()
    assert argv == ["-fsyntax-only", "-I", "bms", "-I", "copy", "-I", "src", "src/MAIN.cbl"]


def test_the_cobol_check_does_not_run_without_cobc(cobol_repo, tmp_path):
    (tmp_path / "bin" / "cobc").unlink()
    assert _RUN_CHECK() is None


def test_the_cobol_check_finds_nothing_without_a_program(cobol_repo, monkeypatch):
    (cobol_repo / "src" / "MAIN.cbl").unlink()
    monkeypatch.setenv("COBC_OUT", "copy/CUSTREC.cpy:1: error: PROGRAM-ID header missing")
    assert _RUN_CHECK() == {}


# == COBOL_DIALECT / COBOL_COPYBOOK_DIR ========================================

def test_the_settings_go_to_cobc_before_the_arguments(cobol_repo, monkeypatch, tmp_path):
    monkeypatch.setenv("COBC_OUT", "")
    monkeypatch.setenv("COBOL_DIALECT", "ibm")
    monkeypatch.setenv("COBOL_COPYBOOK_DIR", "lib, /opt/copy")
    monkeypatch.setattr(config, "CHECK_COMMAND", ["python3", str(_CHECK_CBL_PATH), "-w"])
    assert _RUN_CHECK() == {}
    argv = (tmp_path / "cobc_args.txt").read_text(encoding="utf-8").split()
    assert argv == ["-fsyntax-only", "-std=ibm", "-w", "-I", "copy", "-I", "src",
                    "-I", f"{cobol_repo}/lib", "-I", "/opt/copy", "src/MAIN.cbl"]


def test_a_dialect_cobc_does_not_know_runs_no_check(cobol_repo, monkeypatch):
    monkeypatch.setenv("COBOL_DIALECT", "ibmm")
    monkeypatch.setenv("COBC_TRY_STATUS", "1")
    assert _RUN_CHECK() is None


def test_the_settings_reach_a_container():
    from codetwine_loom import sandbox
    argv = sandbox.env_forward_argv()
    assert "COBOL_DIALECT" in argv and "COBOL_COPYBOOK_DIR" in argv


@pytest.fixture
def dialect_ask(tmp_path, monkeypatch):
    """A terminal, a directory to run from with no .env, and an index of one language.

    Yields:
        make(ext, answer_list) -> the index path; input() gives answer_list in turn.
    """
    from conftest import write_index
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COBOL_DIALECT", raising=False)
    monkeypatch.setattr(config, "ENV_FILE_PATH", "")
    monkeypatch.setattr(config, "COBOL_DIALECT", "")
    monkeypatch.setattr(progress, "at_terminal", lambda: True)

    def make(ext: str, answer_list: list[str]) -> str:
        file = f"p/a_{ext}/a.{ext}"
        defs = {f"{file}::X": {"name": "X", "type": "paragraph_header" if ext != "py"
                               else "function_definition", "file": file, "lines": [1, 2],
                               "signature": "X", "doc": ""}}
        path = str(tmp_path / f"index_{ext}.sqlite")
        write_index(path, defs, {file: {"summary": "", "defs": list(defs)}}).close()
        answer_iter = iter(answer_list)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answer_iter))
        return path

    return make


def test_the_dialect_is_asked_once_and_written_to_the_env_file(dialect_ask, tmp_path):
    from codetwine_loom import knowledge
    path = dialect_ask("cbl", ["x", "1"])
    assert knowledge.ask_cobol_dialect(path) == "COBOL_DIALECT=ibm"
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "COBOL_DIALECT=ibm\n"
    assert os.environ["COBOL_DIALECT"] == "ibm" and config.COBOL_DIALECT == "ibm"
    assert knowledge.ask_cobol_dialect(path) == ""


def test_a_digit_that_is_no_number_is_asked_again(dialect_ask, tmp_path):
    from codetwine_loom import knowledge
    path = dialect_ask("cbl", ["\u2460", "\u00b2", "\uff11"])
    assert knowledge.ask_cobol_dialect(path) == "COBOL_DIALECT=ibm"


def test_enter_keeps_the_default_and_is_not_asked_again(dialect_ask, tmp_path):
    from codetwine_loom import knowledge
    (tmp_path / ".env").write_text("LLM_API_KEY=k", encoding="utf-8")
    path = dialect_ask("cpy", [""])
    assert knowledge.ask_cobol_dialect(path) == "COBOL_DIALECT="
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "LLM_API_KEY=k\nCOBOL_DIALECT=\n"
    assert os.environ["COBOL_DIALECT"] == ""
    assert knowledge.ask_cobol_dialect(path) == ""


@pytest.mark.parametrize("before, after", [
    (b"A=1\r\nB=2\r\n", b"A=1\r\nB=2\r\nCOBOL_DIALECT=ibm\r\n"),
    (b"\xef\xbb\xbfA=1\r\nB=2", b"\xef\xbb\xbfA=1\r\nB=2\r\nCOBOL_DIALECT=ibm\r\n"),
    (b"A=1\rB=2\r", b"A=1\rB=2\rCOBOL_DIALECT=ibm\r"),
    (b"A=1\r\nB=2\nC=3\n", b"A=1\r\nB=2\nC=3\nCOBOL_DIALECT=ibm\n"),
    (b"", b"COBOL_DIALECT=ibm\n")])
def test_the_dialect_line_takes_the_line_ending_the_env_file_uses_most(
        dialect_ask, tmp_path, before, after):
    from codetwine_loom import knowledge
    (tmp_path / ".env").write_bytes(before)
    path = dialect_ask("cbl", ["1"])
    assert knowledge.ask_cobol_dialect(path) == "COBOL_DIALECT=ibm"
    assert (tmp_path / ".env").read_bytes() == after


@pytest.mark.parametrize("has_file", [False, True])
def test_the_dialect_is_not_written_while_no_dot_env_file_is_loaded(
        dialect_ask, tmp_path, monkeypatch, caplog, has_file):
    from codetwine_loom import knowledge
    if has_file:
        (tmp_path / ".env").write_text("LLM_API_KEY=k\n", encoding="utf-8")
    monkeypatch.setenv(config.NO_ENV_FILE_ENV, config.NO_ENV_FILE_VALUE)
    path = dialect_ask("cbl", ["1"])
    with caplog.at_level("WARNING"):
        assert knowledge.ask_cobol_dialect(path) == "COBOL_DIALECT=ibm"
    assert (tmp_path / ".env").exists() is has_file
    if has_file:
        assert (tmp_path / ".env").read_text(encoding="utf-8") == "LLM_API_KEY=k\n"
    assert "it holds for this run only" in caplog.text
    assert os.environ["COBOL_DIALECT"] == "ibm" and config.COBOL_DIALECT == "ibm"


def test_the_dialect_is_not_written_to_a_dot_env_file_git_tracks(dialect_ask, tmp_path):
    from codetwine_loom import knowledge
    git_repo(tmp_path)
    (tmp_path / ".env").write_text("LLM_API_KEY=k\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", ".env"], check=True,
                   capture_output=True)
    path = dialect_ask("cbl", ["1"])
    assert knowledge.ask_cobol_dialect(path) == "COBOL_DIALECT=ibm"
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "LLM_API_KEY=k\n"
    assert os.environ["COBOL_DIALECT"] == "ibm" and config.COBOL_DIALECT == "ibm"


def test_the_dialect_is_not_asked_for_another_language_or_with_no_terminal(dialect_ask,
                                                                          monkeypatch):
    from codetwine_loom import knowledge
    assert knowledge.ask_cobol_dialect(dialect_ask("py", [])) == ""
    path = dialect_ask("cbl", [])
    monkeypatch.setattr(progress, "at_terminal", lambda: False)
    assert knowledge.ask_cobol_dialect(path) == ""
    assert "COBOL_DIALECT" not in os.environ


def test_with_no_index_yet_the_files_of_the_repository_decide(dialect_ask, tmp_path):
    """A run asks before the analysis and the image build, when there is no index to read
    the primary language off: a COBOL file the analysis of the repository is handed
    decides, and one in a directory kept out of it does not."""
    from codetwine_loom import knowledge
    dialect_ask("py", ["1"])
    missing = str(tmp_path / "no_index.sqlite")
    project = tmp_path / "repo"
    (project / "copy").mkdir(parents=True)
    (project / "a.py").write_text("x = 1\n", encoding="utf-8")
    (project / "README.md").write_text("# r\n", encoding="utf-8")
    assert knowledge.ask_cobol_dialect(missing) == ""
    assert knowledge.ask_cobol_dialect(missing, str(project)) == ""
    (project / "copy" / "REC.cpy").write_text("       01 REC PIC X.\n", encoding="utf-8")
    assert knowledge.ask_cobol_dialect(missing, str(project), ["copy"]) == ""
    assert "COBOL_DIALECT" not in os.environ
    assert knowledge.ask_cobol_dialect(missing, str(project)) == "COBOL_DIALECT=ibm"


def test_the_dialect_is_asked_when_cobol_is_not_the_primary_language(dialect_ask, tmp_path):
    """The static check of COBOL runs for any COBOL file of the repository, the primary
    language or not."""
    from codetwine_loom import knowledge
    from conftest import write_index
    dialect_ask("py", ["1"])
    defs = {f"p/m{n}_py/m{n}.py::f": {"name": "f", "type": "function_definition",
                                   "file": f"p/m{n}_py/m{n}.py", "lines": [1, 2],
                                   "signature": "def f():", "doc": ""} for n in range(3)}
    path = str(tmp_path / "index_mix.sqlite")
    write_index(path, defs, {**{node["file"]: {"summary": "", "defs": [key]}
                                for key, node in defs.items()},
                             "p/a_cbl/a.cbl": {"summary": "", "defs": []}}).close()
    assert knowledge.ask_cobol_dialect(path) == "COBOL_DIALECT=ibm"


def test_the_end_of_the_input_keeps_the_default_for_the_run_and_is_not_asked_again(
        dialect_ask, tmp_path, monkeypatch):
    """The run asks before the image and again after the analysis and before the merge;
    an answer of no input is not asked for at the later ones."""
    from codetwine_loom import knowledge
    path = dialect_ask("cbl", [])

    def no_more(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_more)
    assert knowledge.ask_cobol_dialect(path) == ""
    assert os.environ["COBOL_DIALECT"] == "" and config.COBOL_DIALECT == ""
    assert not (tmp_path / ".env").exists()
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("asked again"))
    assert knowledge.ask_cobol_dialect(path) == ""

