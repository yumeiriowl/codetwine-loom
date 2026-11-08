"""The reading tools: line-range reads, read_definition, grep_source and search_again.

read_source_file's line range lets files longer than the per-turn output limit
(RLM_MAX_OUTPUT_CHAR) be read in pieces. That a call without those arguments still
returns the whole file is pinned down here.

read_definition, grep_source's count_only and search_again are checked against
conftest's fake_index.

graph_search and get_use_of_file also need project_data and are not checked here.
"""

import dataclasses
import os
import pathlib
import re

import pytest
from codetwine.utils import file_utils

from codetwine_loom import agent, config, paths, tmpfile
from conftest import fake_defs, fake_files
from codetwine_loom.languages import LANG_DICT
from codetwine_loom.textcut import line_list_of
from codetwine_loom.tools import source

TEXT = "".join(f"line {i}\n" for i in range(1, 11))  # 10 lines


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """Point PROJECT_DIR at a directory holding one 10-line file."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "m.py").write_text(TEXT)
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    return root


@pytest.fixture
def def_proj(tmp_path, monkeypatch, fake_index):
    """A PROJECT_DIR holding the 2 files of fake_index as real files.

    read_definition takes only the line range from the index and reads the contents
    from the real file (the index's code disagrees with reality after an edit), so
    exercising that path needs both the index and the files.
    """
    root = tmp_path / "proj"
    root.mkdir()
    (root / "cfg.py").write_text("".join(f"cfg {i}\n" for i in range(1, 101)))
    (root / "util.py").write_text("".join(f"util {i}\n" for i in range(1, 11)))
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    return root


def test_omitting_the_arguments_returns_the_whole_file(proj):
    """read_source_file(path) returns the file byte for byte."""
    assert source.read_source_file("m.py") == TEXT


def test_a_file_without_a_trailing_newline_also_comes_back_whole(proj):
    (proj / "m.py").write_text("a\nb")
    assert source.read_source_file("m.py") == "a\nb"


def test_a_line_range_returns_that_range_with_a_header_line(proj):
    r = source.read_source_file("m.py", 3, 5)
    assert r == "# m.py: lines 3-5 of 10\nline 3\nline 4\nline 5\n"


def test_omitting_end_line_reads_to_the_end(proj):
    r = source.read_source_file("m.py", 9)
    assert r == "# m.py: lines 9-10 of 10\nline 9\nline 10\n"


def test_an_end_line_past_the_file_stops_at_the_end(proj):
    r = source.read_source_file("m.py", 8, 999)
    assert r.startswith("# m.py: lines 8-10 of 10\n")
    assert r.endswith("line 10\n")


def test_a_start_line_past_the_file_is_an_error(proj):
    r = source.read_source_file("m.py", 11)
    assert r.startswith("Error: ")
    assert "has only 10 lines" in r


def test_a_start_line_of_0_or_less_is_an_error(proj):
    assert source.read_source_file("m.py", 0).startswith("Error: ")


def test_an_end_line_below_start_line_is_an_error(proj):
    assert source.read_source_file("m.py", 5, 3).startswith("Error: ")


# A form feed on a line of its own, as C and older Python sources carry between sections.
# The analysis numbers lines at "\n" alone, so "b = 2" is line 3
FORM_FEED_TEXT = "a = 1\n\x0c\nb = 2\nc = 3\n"


@pytest.mark.parametrize("text, keepends, expect", [
    ("a\n\x0c\nb\n", False, ["a", "\x0c", "b"]),
    ("a\n\x0cb", True, ["a\n", "\x0cb"]),
    ("a\u2028b\x1c\x85c\x0bd\n", False, ["a\u2028b\x1c\x85c\x0bd"]),
    ("a\r\nb\rc", True, ["a\r\n", "b\r", "c"]),
    ("a\r\nb\rc", False, ["a", "b", "c"]),
    ("\n\n", False, ["", ""]),
    ("", False, []),
])
def test_lines_are_split_at_the_line_breaks_alone(text, keepends, expect):
    assert line_list_of(text, keepends=keepends) == expect


def test_a_form_feed_line_is_numbered_as_the_analysis_numbers_it(proj):
    (proj / "ff.py").write_text(FORM_FEED_TEXT)
    assert source.read_source_file("ff.py", 3, 3) == "# ff.py: lines 3-3 of 4\nb = 2\n"
    assert [hit["line"] for hit in source.grep_source("b = 2")["hits"]] == [3]


def test_a_line_range_cannot_read_outside_project_dir_either(proj):
    r = source.read_source_file("../../etc/passwd", 1, 3)
    assert r.startswith("Error: ")


def test_an_unreadable_file_is_an_error_with_a_line_range_too(proj):
    r = source.read_source_file("nosuch.py", 1, 3)
    assert r.startswith("Error: ")


# == Tying the reading side to the per-turn limit ==

def test_grep_source_reads_a_repository_under_a_directory_named_like_a_skip_name(
        tmp_path, monkeypatch):
    root = tmp_path / "build" / "main_py"
    root.mkdir(parents=True)
    (root / "main.py").write_text("needle = 1\n")
    (root / "build").mkdir()
    (root / "build" / "made.py").write_text("needle = 2\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(config, "GREP_SKIP_DIR", "build")
    out_dict = source.grep_source("needle")
    assert [hit["path"] for hit in out_dict["hits"]] == ["main.py"]


def test_grep_source_does_not_read_a_directory_a_build_output_pattern_names(
        tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "src" / "calc.egg-info").mkdir(parents=True)
    (root / "src" / "calc.py").write_text("needle = 1\n")
    (root / "src" / "calc.egg-info" / "PKG-INFO").write_text("needle = 2\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    for glob in (None, "src", "**/*"):
        out_dict = source.grep_source("needle", glob)
        assert [hit["path"] for hit in out_dict["hits"]] == ["src/calc.py"], glob


@pytest.fixture
def big_proj(proj, monkeypatch):
    """A 100-line file (9 characters per line) and a 200-character cap.

    The header (27 characters) and the continuation note (about 81) come out of the
    body's budget, so one call returns 10 lines.
    """
    (proj / "m.py").write_text("".join(f"line {i:03d}\n" for i in range(1, 101)))
    monkeypatch.setattr(config, "READ_MAX_CHAR", 200)
    return proj


def test_a_file_over_the_cap_is_cut_at_a_line_with_a_continuation_note(big_proj):
    r = source.read_source_file("m.py")
    assert r.startswith("# m.py: lines 1-10 of 100\nline 001\nline 002\n")
    assert r.endswith('# m.py: lines 1-10 of 100 returned. '
                      'Continue with read_source_file("m.py", 11)\n')
    assert "line 011" not in r


def test_the_whole_return_value_fits_in_the_cap(big_proj):
    """The header and note count toward the budget, so the REPL keeps the note."""
    assert len(source.read_source_file("m.py")) <= 200
    assert len(source.read_source_file("m.py", 11)) <= 200
    assert len(source.read_source_file("m.py", 1, 100)) <= 200


def test_reading_again_from_the_next_line_returns_the_rest(big_proj):
    r = source.read_source_file("m.py", 11)
    assert r.startswith("# m.py: lines 11-20 of 100\nline 011\n")
    assert 'read_source_file("m.py", 21)' in r


def test_the_same_cap_applies_with_an_explicit_line_range(big_proj):
    r = source.read_source_file("m.py", 1, 100)
    assert r.startswith("# m.py: lines 1-10 of 100\n")
    assert "line 011" not in r


@pytest.mark.parametrize("line_tuple, want", [
    (("2",), "Error: start_line must be a whole number, not '2'"),
    ((2.0,), "Error: start_line must be a whole number, not 2.0"),
    ((True,), "Error: start_line must be a whole number, not True"),
    ((1, "3"), "Error: end_line must be a whole number, not '3'"),
])
def test_a_line_number_that_is_not_a_whole_number_is_an_error_string(proj, line_tuple,
                                                                     want):
    assert source.read_source_file("m.py", *line_tuple) == want


def test_a_path_that_is_not_a_string_is_an_error_string(proj):
    assert source.read_source_file(None) == "Error: a path must be a string, not NoneType"
    assert source.read_source_file(["m.py"]) == "Error: a path must be a string, not list"


def test_a_path_like_object_is_read_as_its_string(proj):
    assert source.read_source_file(pathlib.Path("m.py")) == TEXT
    assert source.read_source_file(pathlib.Path("nosuch.py")).startswith(
        "Error: cannot read nosuch.py")


def test_a_reading_tool_handed_the_wrong_type_answers_and_does_not_raise(def_proj):
    assert source.grep_source(None)["error"] == "pattern and glob must be strings"
    assert source.grep_source("cfg", glob=3)["error"] == "pattern and glob must be strings"
    assert source.read_definition(None) == {"ok": False,
                                            "error": "name and file must be strings"}
    assert source.read_definition("save_config", file=3)["ok"] is False
    assert source.graph_search(["save_config"]) == {"error": "name and file must be strings"}
    assert source.graph_search("save_config", hop_count="2") == {
        "error": "hop_count must be a whole number, not '2'"}
    assert source.get_use_of_file(None) == []


def test_a_file_within_the_cap_comes_back_whole(proj, monkeypatch):
    monkeypatch.setattr(config, "READ_MAX_CHAR", 4000)
    assert source.read_source_file("m.py") == TEXT


def test_read_max_chars_of_0_never_cuts(proj, monkeypatch):
    monkeypatch.setattr(config, "READ_MAX_CHAR", 0)
    assert source.read_source_file("m.py") == TEXT
    assert source.read_source_file("m.py", 1, 10) == \
        "# m.py: lines 1-10 of 10\n" + TEXT


def test_one_line_comes_back_even_when_it_exceeds_the_cap(proj, monkeypatch):
    """Returning nothing would read as "the file is empty" (as with grep's first hit)."""
    (proj / "m.py").write_text("x" * 100 + "\nsecond\n")
    monkeypatch.setattr(config, "READ_MAX_CHAR", 10)
    r = source.read_source_file("m.py")
    assert r.startswith("# m.py: lines 1-1 of 2\n" + "x" * 100 + "\n")
    assert "second" not in r


def test_no_note_is_added_when_the_end_was_reached(big_proj):
    """Saying "there is more" without cutting sends the model after lines that do not
    exist."""
    r = source.read_source_file("m.py", 96)
    assert r == ("# m.py: lines 96-100 of 100\n"
                 + "".join(f"line {i:03d}\n" for i in range(96, 101)))


# == read_definition ===============================================

def test_read_definition_reads_the_real_file_at_the_indexs_line_range(def_proj):
    r = source.read_definition("save_config")
    assert r["ok"] is True
    assert r["key"] == "cfg.py::save_config"
    assert r["file"] == "cfg.py"
    assert r["lines"] == [45, 60]
    assert r["code"].startswith("# cfg.py: lines 45-60 of 100\n")
    assert "cfg 45\n" in r["code"] and "cfg 60\n" in r["code"]
    assert "cfg 61" not in r["code"]


def test_read_definition_does_not_return_the_indexs_code(def_proj, fake_index):
    """It reads the real file, so an edit cannot desync it. The index holds no code."""
    node = fake_index.definition("p/cfg_py/cfg.py::save_config")
    assert "code" not in node
    assert "cfg 45\n" in source.read_definition("save_config")["code"]


def test_read_definition_is_ok_false_for_an_unknown_name(def_proj):
    r = source.read_definition("nosuch_function")
    assert r["ok"] is False
    assert "no such definition" in r["error"]


def test_read_definition_returns_the_candidates_for_a_duplicated_name(def_proj):
    r = source.read_definition("load_config")
    assert r["ok"] is False
    assert r["ambiguous"] is True
    assert sorted(r["candidates"]) == ["cfg.py::load_config",
                                       "util.py::load_config"]


def test_read_definition_takes_the_key_the_tools_print(def_proj):
    """scope["defs"] and every tool answer name a definition as "<path>::<name>", so
    handing that back must not cost a turn."""
    r = source.read_definition("util.py::load_config")
    assert r["ok"] is True and r["key"] == "util.py::load_config"
    # The index's own label, with the suffix a duplicated name carries, is read too
    r = source.read_definition("cfg.py::load_config#function_definition#10")
    assert r["ok"] is True and r["key"] == "cfg.py::load_config"


def test_a_candidate_of_several_same_named_definitions_in_one_file_picks_that_one(
        tmp_path, monkeypatch, index_of):
    from conftest import _def_node
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "Box.java").write_text("".join(f"box {i}\n" for i in range(1, 21)))
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    file = "p/src/Box_java/Box.java"
    index_of({f"{file}::Box": _def_node("Box", file, [3, 20], "class Box {", ""),
              f"{file}::Box#constructor_declaration#5": _def_node(
                  "Box", file, [5, 7], "Box() {", ""),
              f"{file}::Box#constructor_declaration#9": _def_node(
                  "Box", file, [9, 11], "Box(int a) {", "")},
             {file: {"summary": "", "defs": []}})
    r = source.read_definition("Box")
    assert r["ambiguous"] is True
    assert r["candidates"] == ["src/Box.java::Box",
                               "src/Box.java::Box#constructor_declaration#5",
                               "src/Box.java::Box#constructor_declaration#9"]
    assert source.read_definition("Box", file="src/Box.java")["candidates"] == \
        r["candidates"]
    # Each candidate handed back as it was printed is that definition
    assert [source.read_definition(key)["lines"] for key in r["candidates"]] == [
        [3, 20], [5, 7], [9, 11]]
    assert source.read_definition(r["candidates"][2])["key"] == r["candidates"][2]
    for on_ambiguous in ("return_candidates", "first"):
        assert [source.graph_search(key, on_ambiguous=on_ambiguous)["start"]
                for key in r["candidates"]] == r["candidates"]
    # A suffix the index does not hold is dropped and the name looked up
    assert source.read_definition("src/Box.java::Box#method_declaration#99")[
        "ambiguous"] is True


def test_a_name_that_starts_with_a_hash_is_read_as_the_name_it_is(tmp_path, monkeypatch,
                                                                 index_of):
    from conftest import _def_node
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.js").write_text("".join(f"js {i}\n" for i in range(1, 21)))
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    file = "p/src/a_js/a.js"
    index_of({f"{file}::#bump": _def_node("#bump", file, [3, 5], "#bump() {", ""),
              f"{file}::#bump#method_definition#9": _def_node(
                  "#bump", file, [9, 11], "#bump(step) {", ""),
              f"{file}::#count": _def_node("#count", file, [2, 2], "#count = 0;", "")},
             {file: {"summary": "", "defs": []}})
    assert source.read_definition("#count")["key"] == "src/a.js::#count"
    assert source.read_definition("#count", file="src/a.js")["lines"] == [2, 2]
    assert source.read_definition("src/a.js::#count")["lines"] == [2, 2]
    assert source.graph_search("#count")["start"] == "src/a.js::#count"
    assert source.read_definition("#bump")["candidates"] == [
        "src/a.js::#bump", "src/a.js::#bump#method_definition#9"]
    assert source.read_definition("src/a.js::#bump#method_definition#9")["lines"] == [9, 11]
    # A suffix the index does not hold is dropped, and the name keeps its own "#"
    assert source.read_definition("src/a.js::#count#field_definition#99")["lines"] == [2, 2]
    assert source.read_definition("#count#field_definition#99")["lines"] == [2, 2]


def test_the_file_of_a_definition_is_taken_in_the_notations_the_other_tools_read(def_proj):
    want = source.read_definition("load_config", file="util.py")
    assert want["ok"] is True
    for file in ("./util.py", "sub/../util.py", str(def_proj / "util.py")):
        assert source.read_definition("load_config", file=file) == want
        assert source.read_definition(f"{file}::load_config") == want


def test_read_definition_narrows_to_one_with_file(def_proj):
    r = source.read_definition("load_config", file="util.py")
    assert r["ok"] is True
    assert r["key"] == "util.py::load_config"


def test_read_definition_takes_a_method_by_its_class_and_its_own_name(tmp_path,
                                                                     monkeypatch, index_of):
    """integrate.analyse() names a definition inside another "<outer>.<name>"."""
    from conftest import _def_node
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text("".join(f"calc {i}\n" for i in range(1, 11)))
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    file = "p/calc_py/calc.py"
    index_of({f"{file}::A": _def_node("A", file, [1, 3], "class A:", ""),
              f"{file}::run": _def_node("run", file, [2, 3], "def run(self, a):", ""),
              f"{file}::B": _def_node("B", file, [6, 8], "class B:", ""),
              f"{file}::run#function_definition#7": _def_node(
                  "run", file, [7, 8], "def run(self, b, c):", "")},
             {file: {"summary": "", "defs": []}})
    r = source.read_definition("B.run")
    assert r["ok"] is True and r["lines"] == [7, 8]
    assert source.read_definition("A.run", file="calc.py")["lines"] == [2, 3]
    assert source.read_definition("run")["ambiguous"] is True
    assert "no such definition" in source.read_definition("C.run")["error"]


def test_read_definition_says_something_else_when_the_name_is_in_another_file(def_proj):
    r = source.read_definition("save_config", file="util.py")
    assert r["ok"] is False
    assert "exists but not in" in r["error"]


def test_read_definition_stops_at_the_end_when_the_range_runs_past_it(
        def_proj, index_of, caplog):
    """The file got shorter after the index was built. lines keeps the index's value."""
    defs = fake_defs()
    defs["p/cfg_py/cfg.py::save_config"]["lines"] = [95, 200]
    index_of(defs, fake_files())
    r = source.read_definition("save_config")
    assert r["ok"] is True
    assert r["lines"] == [95, 200]
    assert r["code"].startswith("# cfg.py: lines 95-100 of 100\n")
    assert "the index line range is past the end of the file" in caplog.text


def test_read_definition_does_not_warn_about_a_cut_at_the_cap(
        def_proj, monkeypatch, caplog):
    """An end line shrunk by READ_MAX_CHAR is not an index mismatch."""
    monkeypatch.setattr(config, "READ_MAX_CHAR", 20)
    r = source.read_definition("save_config")
    assert r["ok"] is True
    assert "returned. Continue with read_source_file" in r["code"]
    assert "the index line range is past the end of the file" not in caplog.text


# == grep_source's count mode =======================================

def test_grep_source_count_only_returns_the_per_file_counts(def_proj):
    r = source.grep_source(r"^cfg ", count_only=True)
    assert r["counts"] == [{"path": "cfg.py", "count": 100}]
    assert r["total"] == 100
    assert r["truncated"] is False


def test_grep_source_count_only_orders_by_count(def_proj):
    r = source.grep_source(r"\d", count_only=True)
    assert [c["path"] for c in r["counts"]] == ["cfg.py", "util.py"]


def test_grep_source_count_only_returns_no_line_contents(def_proj):
    r = source.grep_source(r"\d", count_only=True)
    assert "hits" not in r
    assert len(str(r)) < config.RLM_MAX_OUTPUT_CHAR


def test_grep_source_count_only_uses_its_own_limit(def_proj, monkeypatch):
    """It must stop at GREP_MAX_HIT_COUNT, not at GREP_MAX_HIT."""
    monkeypatch.setattr(config, "GREP_MAX_HIT", 5)
    monkeypatch.setattr(config, "GREP_MAX_HIT_COUNT", 30)
    r = source.grep_source(r"^cfg ", count_only=True)
    assert r["total"] == 30
    assert r["truncated"] is True


# == grep_source's context lines and the output limit ================

def test_the_return_keys_in_the_instructions_match_the_real_ones(def_proj):
    """The return keys the instructions name are the keys read_definition returns."""
    found = source.read_definition("save_config")
    assert set(found) == set(agent.TOOL_RESULT_KEY_DICT["read_definition"][0])
    hits = source.grep_source(r"^util 3$")
    assert set(hits) == set(agent.TOOL_RESULT_KEY_DICT["grep_source"][0])
    assert set(hits["hits"][0]) == set(agent.TOOL_RESULT_KEY_DICT["grep_source"][1]["hits"])
    assert set(hits["counts"][0]) == set(
        agent.TOOL_RESULT_KEY_DICT["grep_source"][1]["counts"])


def test_grep_source_adds_3_context_lines_to_the_first_hits(def_proj):
    r = source.grep_source(r"^util 3$")
    hit = r["hits"][0]
    assert hit["path"] == "util.py" and hit["line"] == 3 and hit["text"] == "util 3"
    assert hit["before"] == ["util 1", "util 2"]      # only 2 exist near the top
    assert hit["after"] == ["util 4", "util 5", "util 6"]
    assert r["total"] == 1 and r["shown"] == 1 and r["truncated"] is False


def test_grep_source_adds_context_to_the_first_grep_context_hits_only(
        def_proj, monkeypatch):
    monkeypatch.setattr(config, "GREP_CONTEXT_HIT", 2)
    r = source.grep_source(r"^util ")
    assert all("before" in h for h in r["hits"][:2])
    assert all("before" not in h for h in r["hits"][2:])


def test_grep_source_counts_every_match_in_counts(def_proj, monkeypatch):
    """Even with hits cut, counts and total must cover every match."""
    monkeypatch.setattr(config, "GREP_MAX_HIT", 3)
    r = source.grep_source(r"^cfg ")
    assert r["counts"] == [{"path": "cfg.py", "count": 100}]
    assert r["total"] == 100
    assert r["shown"] == 3
    assert r["truncated"] is True


def test_counts_say_when_the_walk_stopped_before_the_last_file(def_proj, monkeypatch):
    """The walk stops at GREP_MAX_HIT_COUNT part-way through, so the files after that
    point are missing from "counts" rather than matching nothing.
    """
    monkeypatch.setattr(config, "GREP_MAX_HIT_COUNT", 50)
    r = source.grep_source(r"^(cfg|util) ")
    assert r["counts_complete"] is False
    # util.py has 10 matches but was never reached, so it is absent, not 0
    assert [c["path"] for c in r["counts"]] == ["cfg.py"]
    counted = source.grep_source(r"^(cfg|util) ", count_only=True)
    assert counted["counts_complete"] is False

    monkeypatch.setattr(config, "GREP_MAX_HIT_COUNT", 5000)
    whole = source.grep_source(r"^(cfg|util) ", count_only=True)
    assert whole["counts_complete"] is True
    assert {c["path"] for c in whole["counts"]} == {"cfg.py", "util.py"}


def test_the_number_of_hits_follows_the_output_limit(def_proj, monkeypatch):
    """Lowering RLM_MAX_OUTPUT_CHAR must return fewer hits."""
    monkeypatch.setattr(config, "GREP_CONTEXT_LINE", 0)
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 4000)
    many = source.grep_source(r"^cfg ")
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 400)
    few = source.grep_source(r"^cfg ")
    assert few["shown"] < many["shown"]
    assert few["total"] == many["total"] == 100
    assert source.hit_limit() == 400 // (config.GREP_HIT_OVERHEAD_CHAR
                                         + config.GREP_LINE_MAX_CHAR)


def test_a_grep_line_over_the_cap_says_how_much_was_omitted(proj, monkeypatch):
    monkeypatch.setattr(config, "GREP_LINE_MAX_CHAR", 6)
    (proj / "m.py").write_text("line 1\nline two is long\nline 3\n")
    r = source.grep_source(r"^line two")
    assert r["hits"][0]["text"] == "line t ... (10 chars omitted)"
    assert r["hits"][0]["before"] == ["line 1"]


def test_hit_limit_follows_grep_max_hit_alone_when_the_output_limit_is_off(monkeypatch):
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 0)
    monkeypatch.setattr(config, "GREP_MAX_HIT", 77)
    assert source.hit_limit() == 77


def test_no_output_limit_leaves_the_number_of_hits_to_grep_max_hit(def_proj, monkeypatch):
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 0)
    r = source.grep_source(r"^cfg ")
    assert (r["total"], r["shown"], r["truncated"]) == (100, 100, False)
    monkeypatch.setattr(config, "GREP_MAX_HIT", 7)
    r = source.grep_source(r"^cfg ")
    assert (r["total"], r["shown"], r["truncated"]) == (100, 7, True)


def test_grep_source_returns_one_hit_even_below_the_limit(def_proj, monkeypatch):
    """0 hits would read as "there is no such name"."""
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 1)
    r = source.grep_source(r"^util 3$")
    assert r["shown"] == 1


# == search_again ===================================================

def test_search_again_finds_a_file_named_in_the_terms(index_of, tmp_path, monkeypatch):
    index_of(fake_defs(), {**fake_files(), "p/pyproject_toml/pyproject.toml":
                           {"summary": "", "defs": []}})
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    assert "pyproject.toml" in source.search_again("pyproject.toml")["files"]


def test_search_again_returns_calls_left(def_proj):
    r = source.search_again("configuration file")
    assert r["ok"] is True
    assert r["calls_left"] == config.SEARCH_AGAIN_MAX_CALL - 1
    assert "defs" in r and "files" in r


def test_search_again_is_ok_false_once_the_limit_is_reached(def_proj, monkeypatch):
    monkeypatch.setattr(config, "SEARCH_AGAIN_MAX_CALL", 2)
    assert source.search_again("configuration")["calls_left"] == 1
    assert source.search_again("configuration")["calls_left"] == 0
    r = source.search_again("configuration")
    assert r["ok"] is False
    assert "call limit" in r["error"]


@pytest.mark.parametrize("query", ["  ", "設定を直して", "123 456"])
def test_search_again_refuses_a_query_with_no_english_word_in_it(def_proj, query):
    """It would spend one of the calls on terms an English index matches nothing of."""
    before = source._call_remain_count()
    out = source.search_again(query)
    assert out["ok"] is False and "English" in out["error"]
    assert source._call_remain_count() == before


@pytest.mark.parametrize("query", [["load", "config"], 3, None])
def test_search_again_refuses_a_query_that_is_not_a_string(def_proj, query):
    before = source._call_remain_count()
    assert source.search_again(query) == {
        "ok": False, "error": f"query must be a string, not {type(query).__name__}",
        "calls_left": before}
    assert source._call_remain_count() == before


def test_reset_search_calls_restores_the_budget(def_proj, monkeypatch):
    monkeypatch.setattr(config, "SEARCH_AGAIN_MAX_CALL", 1)
    source.search_again("configuration")
    assert source.search_again("configuration")["ok"] is False
    source.reset_search_count()
    assert source.search_again("configuration")["ok"] is True


# == The usage table: set_project_data / get_use_of_file ====================

PK = {"files": [
    {"file": "proj/a_py/a.py",
     "file_dependencies": {"callee_usages": [
         {"from": "proj/util_py/util.py", "name": "helper", "lines": [3],
          "target_name": "helper", "target_start_line": 1},
         {"from": "proj/b_py/b.py", "name": "other", "lines": [9],
          "target_name": "other", "target_start_line": 1}]}},
    {"file": "proj/c_py/c.py",
     "file_dependencies": {"callee_usages": [
         {"from": "proj/util_py/util.py", "name": "helper.attr", "lines": [4, 5],
          "target_name": "helper", "target_start_line": 1}]}},
]}


@pytest.fixture
def usages(monkeypatch, knowledge_of):
    """Load PK into the usage table with proj as the project name."""
    monkeypatch.setattr(source, "project_name", "proj")
    source.set_project_data(knowledge_of(PK))
    yield
    source.set_project_data(knowledge_of({"files": []}))


def test_get_files_using_returns_the_referring_files_in_repository_paths(usages):
    assert source.get_use_of_file("util.py") == [
        {"file": "a.py", "name": "helper", "lines": [3]},
        {"file": "c.py", "name": "helper.attr", "lines": [4, 5]}]


def test_get_files_using_takes_a_knowledge_path_too(usages):
    assert source.get_use_of_file("proj/util_py/util.py") == \
        source.get_use_of_file("util.py")


def test_get_files_using_takes_the_path_in_the_notations_the_other_tools_read(
        usages, tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    want = source.get_use_of_file("util.py")
    assert len(want) == 2
    for path in ("./util.py", "tests/../util.py", str(root / "util.py"),
                 pathlib.Path("util.py")):
        assert source.get_use_of_file(path) == want
    assert source.get_use_of_file(str(tmp_path / "util.py")) == []


def test_get_files_using_matches_the_whole_path_not_a_substring(usages):
    assert source.get_use_of_file("til.py") == []


def test_get_files_using_returns_an_empty_list_for_a_file_nothing_uses(usages):
    assert source.get_use_of_file("a.py") == []


def test_set_project_data_keeps_nothing_but_the_usages(usages, knowledge_of):
    """The definitions and the design documents must not stay in the process."""
    pk = {"files": [{"file": "proj/a_py/a.py",
                     "doc": {"sections": ["x" * 1000]},
                     "file_dependencies": {
                         "definitions": [
                             {"name": "big", "type": "function_definition",
                              "start_line": 1, "end_line": 2,
                              "context": "y" * 1000}],
                         "callee_usages": []}}]}
    source.set_project_data(knowledge_of(pk))
    assert source._file_use_dict == {}


def test_skip_dirs_reads_the_setting(monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "sql")
    monkeypatch.setattr(config, "GREP_SKIP_DIR", "build, target ,,node_modules")
    assert source.skip_dir_set() == frozenset({"build", "target", "node_modules"})


def test_skip_dirs_adds_the_build_output_of_the_primary_language_only(monkeypatch):
    monkeypatch.setattr(config, "GREP_SKIP_DIR", "node_modules")
    monkeypatch.setattr(config, "PRIMARY_EXT", "cs")
    assert source.skip_dir_set() == frozenset({"node_modules", "bin", "obj"})
    monkeypatch.setattr(config, "PRIMARY_EXT", "kt")
    assert source.skip_dir_set() == frozenset({"node_modules", "build", ".gradle", ".kotlin"})
    # A file name (uv.lock) is not a directory name
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    assert source.skip_dir_set() == frozenset({"node_modules", "__pycache__", "*.egg-info"})
    # Nor is a pattern anchored in the root
    monkeypatch.setitem(LANG_DICT, "yy", dataclasses.replace(
        LANG_DICT["py"], build_output_tuple=("/dist/", "out/")))
    monkeypatch.setattr(config, "PRIMARY_EXT", "yy")
    assert source.skip_dir_set() == frozenset({"node_modules", "out"})
    monkeypatch.setattr(config, "PRIMARY_EXT", "sql")
    assert source.skip_dir_set() == frozenset({"node_modules"})
    monkeypatch.setattr(config, "PRIMARY_EXT", "zz")
    assert source.skip_dir_set() == frozenset({"node_modules"})


def test_skip_dirs_covers_the_build_output_of_the_other_languages():
    """grep_source walking a jar's unpacked classes or a Gradle build finds code the
    index has never seen."""
    names = source.skip_dir_set()
    assert {"target", "build", "dist", "vendor", "node_modules", ".venv"} <= names


@pytest.mark.parametrize("part_tuple, expected", [
    (("out", "p", "a_py", "a.py"), True),
    (("out", "Makefile", "Makefile"), True),        # no extension: wrapped in its own name
    ((".gitignore", ".gitignore"), True),
    (("docs", "Makefile"), False),
    (("a.py",), False),
])
def test_is_codetwine_copy_follows_codetwine_s_copy_path_shape(part_tuple, expected):
    assert source.is_codetwine_copy(part_tuple) is expected


# == What read_source_file says about a path it cannot read ================
def test_a_missing_file_is_an_error_naming_the_path(proj):
    r = source.read_source_file("nosuch.py")
    assert r.startswith("Error: cannot read nosuch.py")


def test_an_index_path_is_told_apart_from_a_missing_file(def_proj):
    r = source.read_source_file("p/cfg_py/cfg.py")
    assert "is an index path" in r and "in this case cfg.py" in r


# == What grep_source walks past ============================================
def test_grep_source_passes_over_the_output_directory_inside_the_repository(
        def_proj, monkeypatch):
    out = def_proj / "out"
    out.mkdir()
    (out / "copy.py").write_text("cfg 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "OUTPUT_DIR", str(out))
    path_list = [c["path"] for c in source.grep_source("^cfg 1$", count_only=True)["counts"]]
    assert path_list == ["cfg.py"]


def test_grep_source_passes_over_a_directory_whose_name_matches_the_glob(def_proj):
    (def_proj / "d.py").mkdir()
    assert source.grep_source("^cfg 1$", glob="*.py", count_only=True)["total"] == 1


def test_grep_source_passes_over_the_directories_it_is_told_to_skip(def_proj, monkeypatch):
    monkeypatch.setattr(config, "GREP_SKIP_DIR", "vendor")
    (def_proj / "vendor").mkdir()
    (def_proj / "vendor" / "v.py").write_text("cfg 1\n", encoding="utf-8")
    assert source.grep_source("^cfg 1$", count_only=True)["total"] == 1


def test_grep_source_reads_a_file_named_like_a_directory_it_skips(def_proj, monkeypatch):
    monkeypatch.setattr(config, "GREP_SKIP_DIR", "vendor,build")
    (def_proj / "build").write_text("cfg 1\n", encoding="utf-8")
    (def_proj / "scripts").mkdir()
    (def_proj / "scripts" / "vendor").write_text("cfg 1\n", encoding="utf-8")
    (def_proj / "vendor").mkdir()
    (def_proj / "vendor" / "build").write_text("cfg 1\n", encoding="utf-8")
    out_dict = source.grep_source("^cfg 1$", count_only=True)
    assert sorted(entry["path"] for entry in out_dict["counts"]) == [
        "build", "cfg.py", "scripts/vendor"]
    assert source.grep_source("^cfg 1$", glob="build", count_only=True)["total"] == 1


@pytest.fixture
def tree_proj(def_proj):
    """def_proj with the same line in files two and three directories down."""
    for name in ("src/s.py", "src/a/deep.py", "pkg/src/p.py", "pkg/q.txt"):
        (def_proj / name).parent.mkdir(parents=True, exist_ok=True)
        (def_proj / name).write_text("cfg 1\n", encoding="utf-8")
    return def_proj


@pytest.mark.parametrize("glob, walk, path_list", [
    ("*.py", "**/*.py", ["cfg.py", "pkg/src/p.py", "src/a/deep.py", "src/s.py"]),
    ("./*.py", "./*.py", ["cfg.py"]),
    ("s.py", "**/s.py", ["src/s.py"]),
    ("src", "**/src", ["pkg/src/p.py", "src/a/deep.py", "src/s.py"]),
    ("src/", "**/src/", ["pkg/src/p.py", "src/a/deep.py", "src/s.py"]),
    ("./src", "./src", ["src/a/deep.py", "src/s.py"]),
    ("src/**", "src/**", ["src/a/deep.py", "src/s.py"]),
    ("src/*", "src/*", ["src/a/deep.py", "src/s.py"]),
    ("pkg/**/*.py", "pkg/**/*.py", ["pkg/src/p.py"]),
    ("**", "**", ["cfg.py", "pkg/q.txt", "pkg/src/p.py", "src/a/deep.py", "src/s.py"]),
    (".", "**", ["cfg.py", "pkg/q.txt", "pkg/src/p.py", "src/a/deep.py", "src/s.py"]),
    ("./", "**", ["cfg.py", "pkg/q.txt", "pkg/src/p.py", "src/a/deep.py", "src/s.py"]),
])
def test_a_glob_without_a_slash_is_tried_in_every_directory_and_a_directory_is_its_files(
        tree_proj, glob, walk, path_list):
    for count_only in (False, True):
        out = source.grep_source("^cfg 1$", glob=glob, count_only=count_only)
        assert "error" not in out and out["globs"] == [walk]
        assert sorted(entry["path"] for entry in out["counts"]) == path_list
        assert out["total"] == len(path_list)


def test_the_walk_with_no_glob_reads_each_file_once_in_the_order_of_the_paths(tree_proj):
    out = source.grep_source("^cfg 1$")
    assert [hit["path"] for hit in out["hits"]] == [
        "cfg.py", "pkg/q.txt", "pkg/src/p.py", "src/a/deep.py", "src/s.py"]
    assert out["globs"] == ["**/*"] and out["total"] == 5


@pytest.mark.parametrize("glob", ["nope", "nope/", "src/nope/**", "*.md", "..", "vendor",
                                  "link_dir", "blob.bin"])
def test_a_glob_that_names_no_file_to_search_says_so(tree_proj, monkeypatch, glob):
    """A name that is not there, a directory the walk skips, a link to a directory and
    a file that is not text are each no file to search."""
    monkeypatch.setattr(config, "GREP_SKIP_DIR", "vendor")
    (tree_proj / "vendor").mkdir()
    (tree_proj / "vendor" / "v.py").write_text("cfg 1\n", encoding="utf-8")
    (tree_proj / "link_dir").symlink_to("src")
    (tree_proj / "blob.bin").write_bytes(b"cfg 1\n\x00\xff")
    for count_only in (False, True):
        out = source.grep_source("^cfg 1$", glob=glob, count_only=count_only)
        assert out["error"].startswith(f"the glob names no file to search: {glob}. ")
        assert (out["total"], out["counts"], out["hits"]) == (0, [], [])
    assert set(out) == set(source.grep_source("(", glob="*.py"))


@pytest.mark.parametrize("glob, path_list", [
    ("link_src/**", ["link_src/a/deep.py", "link_src/s.py"]),
    ("link_src/*", ["link_src/a/deep.py", "link_src/s.py"]),
    ("link_src/**/*", ["link_src/a/deep.py", "link_src/s.py"]),
    ("link_src/*.py", ["link_src/s.py"]),
    ("./link_src/**", ["link_src/a/deep.py", "link_src/s.py"]),
    ("pkg/link_up/src/**", ["pkg/link_up/src/a/deep.py", "pkg/link_up/src/s.py"]),
    ("src/**", ["src/a/deep.py", "src/s.py"]),
    ("**", ["cfg.py", "pkg/q.txt", "pkg/src/p.py", "src/a/deep.py", "src/s.py"]),
    ("*/**", ["pkg/q.txt", "pkg/src/p.py", "src/a/deep.py", "src/s.py"]),
    ("*/*.py", ["src/s.py"]),
    ("s*/**", ["src/a/deep.py", "src/s.py"]),
])
def test_a_link_to_a_directory_is_walked_into_where_the_glob_writes_its_name(
        tree_proj, glob, path_list):
    """Under such a link the files right in it and the files below it are read alike;
    a link a wildcard reaches is not walked into at any depth."""
    (tree_proj / "link_src").symlink_to("src")
    (tree_proj / "pkg" / "link_up").symlink_to("..")
    for count_only in (False, True):
        out = source.grep_source("^cfg 1$", glob=glob, count_only=count_only)
        assert "error" not in out and out["counts_complete"] is True
        assert sorted(entry["path"] for entry in out["counts"]) == path_list


@pytest.mark.parametrize("glob", ["link_src", "link_src/", "link_*/**", "link_*/*",
                                  "l*/**/*.py", "link_far/**", "link_far/*"])
def test_a_link_to_a_directory_the_glob_does_not_walk_into_is_no_file_to_search(
        tree_proj, glob):
    (tree_proj / "link_src").symlink_to("src")
    (tree_proj.parent / "far").mkdir()
    (tree_proj.parent / "far" / "deep.py").write_text("cfg 1\n", encoding="utf-8")
    (tree_proj / "link_far").symlink_to(tree_proj.parent / "far")
    out = source.grep_source("^cfg 1$", glob=glob)
    assert out["error"].startswith(f"the glob names no file to search: {glob}. ")
    assert (out["total"], out["counts"]) == (0, [])


@pytest.mark.parametrize("glob, path_list", [
    ("src/*", ["src/a/deep.py", "src/s.py"]),
    ("src/*/**", ["src/a/deep.py"]),
    ("src/*/*.py", ["src/a/deep.py"]),
    ("src/**", ["src/a/deep.py", "src/s.py"]),
    ("src", ["pkg/src/p.py", "src/a/deep.py", "src/s.py"]),
    ("pkg/*", ["pkg/q.txt", "pkg/src/p.py"]),
    ("./pkg/*/p.py", ["pkg/src/p.py"]),
    ("src/src/**", ["src/src/q.txt", "src/src/src/p.py"]),
    ("./src/src/*.txt", ["src/src/q.txt"]),
    ("src/other/**", ["src/other/a/deep.py", "src/other/s.py"]),
    ("**/other/**", ["src/other/a/deep.py", "src/other/s.py"]),
])
def test_a_link_named_like_another_part_of_the_glob_is_walked_into_only_where_it_is_written(
        tree_proj, glob, path_list):
    """src/src points at pkg, src/other at src, and pkg/pkg at pkg itself: a wildcard
    that reaches one of them does not walk into it, whatever the glob writes elsewhere."""
    (tree_proj / "src" / "src").symlink_to("../pkg")
    (tree_proj / "src" / "other").symlink_to("../src")
    (tree_proj / "pkg" / "pkg").symlink_to(".")
    for count_only in (False, True):
        out = source.grep_source("^cfg 1$", glob=glob, count_only=count_only)
        assert "error" not in out and out["counts_complete"] is True
        assert sorted(entry["path"] for entry in out["counts"]) == path_list
        assert out["total"] == len(path_list)


@pytest.mark.parametrize("glob, spell_tuple", [
    ("link/**", (("link",), frozenset())),
    ("a/link/*.py", (("a", "link"), frozenset())),
    ("./a//link/*.py", (("a", "link"), frozenset())),
    ("src/*/x.py", (("src", None), frozenset())),
    ("**/link/**", ((), frozenset({"link"}))),
    ("a/**/b/c*/d/e", (("a",), frozenset({"b", "d"}))),
    ("link", ((), frozenset())),
    ("link/", ((), frozenset())),
])
def test_where_a_glob_writes_a_directory_s_name_is_read_part_by_part(glob, spell_tuple):
    assert source._link_spell(glob) == spell_tuple


def test_the_header_of_a_slice_is_read_with_a_count_of_a_few_digits_alone():
    assert source.SLICE_HEADER.match("# a.py: lines 1-2 of 3\nx\n")[3] == "3"
    assert source.SLICE_HEADER.match(f"# a.py: lines 1-2 of {'9' * 5000}\nx\n") is None


def test_a_glob_whose_files_hold_no_match_is_a_walk_that_ran_to_its_end(tree_proj):
    out = source.grep_source("no such line", glob="src")
    assert "error" not in out and (out["total"], out["counts_complete"]) == (0, True)
    assert "error" not in source.grep_source("no such line")


def test_a_glob_whose_only_file_cannot_be_read_names_the_file(tree_proj):
    (tree_proj / "pkg" / "q.txt").chmod(0o000)
    try:
        out = source.grep_source("cfg", glob="*.txt")
    finally:
        (tree_proj / "pkg" / "q.txt").chmod(0o644)
    assert "error" not in out and out["unread"] == ["pkg/q.txt"]


@pytest.mark.parametrize("pattern", ["(", "a{99999999999}", "(?u)(?a)x",
                                     "(" * 5000 + ")" * 5000])
def test_a_pattern_re_cannot_compile_comes_back_as_an_error(def_proj, pattern):
    out = source.grep_source(pattern)
    assert out["error"].startswith("invalid regex: ") and out["total"] == 0


def test_grep_source_takes_a_pattern_that_is_already_compiled(def_proj):
    assert source.grep_source(re.compile("^CFG 1$", re.IGNORECASE),
                              count_only=True)["total"] == 1
    assert source.grep_source(re.compile("^CFG 1$"), count_only=True)["total"] == 0
    assert source.grep_source(re.compile(b"cfg"))["error"] == \
        "pattern and glob must be strings"


def test_grep_source_passes_over_a_copy_codetwine_wrote(def_proj):
    (def_proj / "cfg_py").mkdir()
    (def_proj / "cfg_py" / "cfg.py").write_text("cfg 1\n", encoding="utf-8")
    assert source.grep_source("^cfg 1$", count_only=True)["total"] == 1


def test_grep_source_reads_no_file_outside_the_repository(def_proj):
    (def_proj.parent / "outside.txt").write_text("cfg 1\n", encoding="utf-8")
    (def_proj.parent / "far").mkdir()
    (def_proj.parent / "far" / "deep.py").write_text("cfg 1\n", encoding="utf-8")
    (def_proj / "link_out.py").symlink_to(def_proj.parent / "outside.txt")
    (def_proj / "link_dir").symlink_to(def_proj.parent / "far")
    (def_proj / "link_in.py").symlink_to("cfg.py")
    root = def_proj.resolve()
    for glob, want_count in ((None, 2), ("../*", 2), ("../**/*", 2), ("link_dir/*", 0),
                             ("link_dir", 0), ("..", 0), ("../far", 0),
                             ("link_out.py", 0), ("link_in.py", 1)):
        out = source.grep_source("^cfg 1$", glob=glob, count_only=True)
        assert out["total"] == want_count, glob
        assert all((def_proj / c["path"]).resolve().is_relative_to(root)
                   for c in out["counts"]), (glob, out["counts"])
    # A link that stays inside the repository is read, as read_source_file reads it
    out = source.grep_source("^cfg 1$", count_only=True)
    assert sorted(c["path"] for c in out["counts"]) == ["cfg.py", "link_in.py"]


@pytest.mark.parametrize("glob", ["/etc/host*", "src/**x/*.py"])
def test_a_glob_that_cannot_be_read_comes_back_as_an_error(def_proj, glob):
    for count_only in (False, True):
        out = source.grep_source("cfg", glob=glob, count_only=count_only)
        assert out["error"].startswith("invalid glob: ")
        assert (out["total"], out["counts"], out["hits"], out["globs"]) == (0, [], [], [glob])
    assert set(out) - {"error"} == set(source.grep_source("(", glob="*.py")) - {"error"}


def test_grep_source_passes_over_a_file_that_is_not_text(def_proj):
    (def_proj / "blob.py").write_bytes(b"cfg 1\n\x00\xff")
    out = source.grep_source("^cfg 1$", count_only=True)
    assert out["total"] == 1 and out["unread"] == []


# == Files that are not UTF-8 ==================================
# 表 and 能 end in 0x5c ("\\") in Shift_JIS, the byte that ends a string literal early
# when the file is read as anything but Shift_JIS
_SJIS_TEXT = "# 表の能力を足す\ndef add(a, b):\n    return a + b\n"


@pytest.fixture
def sjis_proj(proj, monkeypatch):
    """The proj directory with a Shift_JIS file and SOURCE_ENCODING naming it."""
    monkeypatch.setattr(file_utils, "SOURCE_ENCODING", ["cp932"])
    (proj / "sj.py").write_bytes(_SJIS_TEXT.encode("cp932"))
    return proj


def test_a_file_that_is_not_utf_8_is_read_in_its_own_encoding(sjis_proj):
    assert source.read_source_file("sj.py") == _SJIS_TEXT


def test_grep_source_finds_what_a_file_that_is_not_utf_8_holds(sjis_proj):
    out = source.grep_source("表の能力", glob="*.py")
    assert out["counts"] == [{"path": "sj.py", "count": 1}] and out["unread"] == []


def test_grep_source_names_a_file_it_could_not_read(proj):
    (proj / "locked.py").write_text("x = 1\n")
    (proj / "locked.py").chmod(0o000)
    try:
        out = source.grep_source("x = 1")
        counted = source.grep_source("x = 1", count_only=True)
    finally:
        (proj / "locked.py").chmod(0o644)
    assert out["unread"] == ["locked.py"] and counted["unread"] == ["locked.py"]
    assert out["total"] == 0


def test_a_named_pipe_and_a_file_too_large_are_not_read(proj, monkeypatch):
    """Neither is opened: the pipe is not waited on, the large file is not read whole."""
    os.mkfifo(proj / "pipe.py")
    assert source.read_source_file("pipe.py") == \
        f"Error: cannot read pipe.py: {proj / 'pipe.py'} is not a regular file"
    monkeypatch.setattr(source.edit_mod, "own_change_set", lambda: {"pipe.py", "m.py"})
    assert source.fresh_line_range("pipe.py", [1, 1], "x = 1") == ([1, 1], True)
    monkeypatch.setattr(tmpfile, "READ_MAX_BYTE", 40)
    assert source.read_source_file("m.py") == \
        f"Error: cannot read m.py: {proj / 'm.py'} is 71 bytes, over the 40 read of one file"
    assert source.fresh_line_range("m.py", [2, 2], "line 2") == ([2, 2], True)
    out = source.grep_source("line")
    assert out["unread"] == ["m.py"] and out["total"] == 0
    monkeypatch.setattr(tmpfile, "READ_MAX_BYTE", 71)
    assert source.grep_source("line")["total"] == 10
    assert source.fresh_line_range("m.py", [2, 2], "line 2") == ([2, 2], False)


def test_a_file_utf_8_cannot_decode_is_read_in_the_encoding_detected(proj, monkeypatch):
    monkeypatch.setattr(file_utils, "SOURCE_ENCODING", [])
    body = "# 日本語のコメントがここに入ります。表示と機能の説明。\n" * 20
    (proj / "sj.py").write_bytes(body.encode("cp932"))
    assert source.read_source_file("sj.py") == body


def test_a_crlf_file_that_is_not_utf_8_reads_with_newlines_only(sjis_proj):
    (sjis_proj / "sj.py").write_bytes(_SJIS_TEXT.replace("\n", "\r\n").encode("cp932"))
    assert source.read_source_file("sj.py") == _SJIS_TEXT


def test_a_second_hit_that_does_not_fit_the_turn_is_left_out(def_proj, monkeypatch):
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 1)
    out = source.grep_source("cfg [12]$", glob="cfg.py")
    assert out["total"] == 2 and out["shown"] == 1 and out["truncated"] is True


# == read_definition on a file that is not there ============================
def test_read_definition_says_when_the_file_cannot_be_read(def_proj):
    (def_proj / "cfg.py").unlink()
    out = source.read_definition("save_config")
    assert out["ok"] is False and "cannot read" in out["error"]


# == graph_search ===========================================================
_A = "p/cfg_py/cfg.py::load_config"
_B = "p/cfg_py/cfg.py::save_config"
_C = "p/util_py/util.py::load_config"


@pytest.fixture
def graph_index(index_of):
    """fake_defs with the calls B -> A and C -> B."""
    return index_of(fake_defs(), fake_files(), calls=[(_B, _A), (_C, _B)])


def test_graph_search_refuses_a_direction_it_does_not_know(graph_index):
    assert "direction must be" in source.graph_search("save_config", direction="up")["error"]


def test_graph_search_refuses_an_on_ambiguous_it_does_not_know(graph_index):
    out = source.graph_search("save_config", on_ambiguous="First")
    assert "on_ambiguous must be" in out["error"]


def test_graph_search_returns_the_candidates_of_a_name_in_two_files(graph_index):
    out = source.graph_search("load_config")
    assert out["ambiguous"] is True
    assert out["candidates"] == ["cfg.py::load_config", "util.py::load_config"]


def test_graph_search_takes_the_first_candidate_when_told_to(graph_index):
    out = source.graph_search("load_config", on_ambiguous="first")
    assert out["start"] == "cfg.py::load_config"


def test_graph_search_settles_a_name_with_the_file(graph_index):
    out = source.graph_search("load_config", file="util.py")
    assert out["start"] == "util.py::load_config"


def test_graph_search_says_when_nothing_has_the_name(graph_index):
    assert "error" in source.graph_search("nosuch")


def test_outgoing_follows_what_the_definition_uses(graph_index):
    out = source.graph_search("save_config", direction="outgoing")
    assert [n["key"] for n in out["nodes"]] == ["cfg.py::load_config"]
    assert out["edges"] == [{"source": "cfg.py::save_config",
                             "target": "cfg.py::load_config", "hop": 1}]
    assert out["nodes"][0]["via"] == "outgoing" and out["nodes"][0]["file"] == "cfg.py"


def test_incoming_follows_what_uses_the_definition(graph_index):
    out = source.graph_search("save_config", direction="incoming")
    assert [n["key"] for n in out["nodes"]] == ["util.py::load_config"]
    assert out["edges"][0]["source"] == "util.py::load_config"


def test_both_follows_both_ways_and_says_which_way_each_came(graph_index):
    out = source.graph_search("save_config")
    assert {n["key"]: n["via"] for n in out["nodes"]} == {
        "cfg.py::load_config": "outgoing", "util.py::load_config": "incoming"}
    assert out["hops"] == 1 and out["direction"] == "both"


def test_a_second_hop_reaches_the_definition_behind_the_first(graph_index):
    out = source.graph_search("load_config", file="util.py", hop_count=2,
                              direction="outgoing")
    assert [(n["key"], n["hop"]) for n in out["nodes"]] == [
        ("cfg.py::save_config", 1), ("cfg.py::load_config", 2)]


def test_an_edge_is_listed_once_however_many_ways_it_is_reached(graph_index):
    out = source.graph_search("save_config", hop_count=3)
    edge_set = {(e["source"], e["target"]) for e in out["edges"]}
    assert len(edge_set) == len(out["edges"]) == 2


# == the line ranges of a file this run has written =========================

def test_a_definition_moves_with_what_was_written_before_it(def_proj, monkeypatch):
    """The index keeps the line numbers of the state the run started from."""
    path = "cfg.py"
    monkeypatch.setattr(source.edit_mod, "write_path_set", lambda: {path})
    monkeypatch.setattr(source.edit_mod, "own_change_set", set)
    target = pathlib.Path(config.PROJECT_DIR) / path
    target.write_text("# added\n# added\n# added\ndef load_config():\n    return 1\n",
                      encoding="utf-8")
    assert source.fresh_line_range(path, [1, 2], "def load_config():") == ([4, 5], False)


def test_a_definition_that_is_no_longer_there_says_the_lines_may_be_out_of_date(
        def_proj, monkeypatch):
    path = "cfg.py"
    monkeypatch.setattr(source.edit_mod, "write_path_set", lambda: {path})
    monkeypatch.setattr(source.edit_mod, "own_change_set", set)
    (pathlib.Path(config.PROJECT_DIR) / path).write_text("# gone\n", encoding="utf-8")
    assert source.fresh_line_range(path, [1, 2], "def load_config():") == ([1, 2], True)


def test_a_definition_is_placed_again_in_a_file_that_is_not_utf_8(def_proj, monkeypatch):
    path = "cfg.py"
    monkeypatch.setattr(file_utils, "SOURCE_ENCODING", ["cp932"])
    monkeypatch.setattr(source.edit_mod, "write_path_set", lambda: {path})
    monkeypatch.setattr(source.edit_mod, "own_change_set", set)
    (pathlib.Path(config.PROJECT_DIR) / path).write_bytes(
        "# 表\n# 能\ndef load_config():  # 設定\n    return 1\n".encode("cp932"))
    assert source.fresh_line_range(path, [1, 2], "def load_config():  # 設定") == \
        ([3, 4], False)


def test_a_file_this_run_has_not_written_keeps_the_index_line_range(def_proj,
                                                                    monkeypatch):
    monkeypatch.setattr(source.edit_mod, "write_path_set", set)
    monkeypatch.setattr(source.edit_mod, "own_change_set", set)
    assert source.fresh_line_range("cfg.py", [1, 2], "def load_config():") == ([1, 2], False)


# == the counts search_again takes ==========================================

@pytest.mark.parametrize("kwargs", [{"def_k": -1}, {"def_k": 0}, {"file_k": "3"},
                                    {"file_k": True}])
def test_search_again_refuses_a_count_that_is_not_one(def_proj, kwargs):
    before = source._call_remain_count()
    out = source.search_again("configuration", **kwargs)
    assert out["ok"] is False and "whole number" in out["error"]
    assert source._call_remain_count() == before


def test_a_count_above_what_the_settings_allow_is_cut(def_proj, monkeypatch):
    monkeypatch.setattr(config, "DEF_K", 2)
    out = source.search_again("configuration", def_k=999)
    assert out["ok"] is True and len(out["defs"]) <= 2
    assert "def_k was cut to 2" in out["note"]


def test_a_cut_count_is_said_after_what_the_search_itself_noted(def_proj, monkeypatch):
    monkeypatch.setattr(config, "DEF_K", 2)
    empty = source.search_again("zzzqqq")["note"]
    assert empty and "cut" not in empty
    out = source.search_again("zzzqqq", def_k=999, file_k=999)
    assert out["note"] == (f"{empty} def_k was cut to 2, which is what the settings allow; "
                           f"file_k was cut to {config.SCOPE_FILE_K}, which is what the "
                           f"settings allow")
    # A search that noted nothing carries the cut alone
    assert source.search_again("configuration", def_k=999)["note"] == \
        "def_k was cut to 2, which is what the settings allow"


def test_search_again_takes_no_count_of_hops():
    import inspect
    assert list(inspect.signature(source.search_again).parameters) == [
        "query", "def_k", "file_k"]


def test_the_line_the_model_sees_says_the_terms_are_english():
    """The stages are shown the signature and the first docstring line, nothing else."""
    from codetwine_loom.rlm import tool_line
    assert "ENGLISH" in tool_line(source.search_again)


def test_a_read_leaves_room_for_the_line_saying_where_to_continue(proj, monkeypatch):
    """Both limits are 40,000 by default, so a read at the cap would be cut by the
    turn's limit and the continuation line is what goes first."""
    monkeypatch.setattr(config, "READ_MAX_CHAR", 200)
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 200)
    (proj / "big.py").write_text("".join(f"line {i}\n" for i in range(1, 200)))
    out = source.read_source_file("big.py")
    assert len(out) < config.RLM_MAX_OUTPUT_CHAR
    assert "Continue with" in out.splitlines()[-1]


def test_a_glob_under_the_repository_s_place_on_the_starting_machine_is_read_from_there(
        monkeypatch, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("needle\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv(paths.HOST_PROJECT_ENV, "/home/me/myapp")
    result = source.grep_source("needle", glob="/home/me/myapp/src/*.py", count_only=True)
    assert result["counts"] == [{"path": "src/a.py", "count": 1}]


def test_a_glob_of_a_root_file_on_the_starting_machine_names_that_file_alone(
        monkeypatch, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "sub" / "a.py").write_text("needle\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv(paths.HOST_PROJECT_ENV, "/home/me/myapp")
    result = source.grep_source("needle", glob="/home/me/myapp/a.py", count_only=True)
    assert [entry["path"] for entry in result["counts"]] == ["a.py"]
