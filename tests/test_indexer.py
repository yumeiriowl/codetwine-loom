"""Index step 4 and the scoring of search step 2."""

import pytest

from codetwine_loom import config
from codetwine_loom.index.build import bm25_doc_dict, file_bm25_doc_dict, path_term_text
from conftest import scoring_index
from codetwine_loom.index.store import INDEX_FILE


def _index_path(tmp_path) -> str:
    """Where a test writes the index it scores against."""
    return str(tmp_path / INDEX_FILE)
from codetwine_loom.index.indexer import bm25_rank, build_bm25, tokenize


def test_tokenize_also_splits_identifiers_into_parts():
    assert tokenize("LLM_API_BASE") == ["llm_api_base", "llm", "api", "base"]


def test_tokenize_drops_single_character_words():
    assert tokenize("parse_file(x)") == ["parse_file", "parse", "file"]


def test_tokenize_drops_the_english_stopwords():
    assert tokenize("the value and the key") == ["value", "key"]


def test_hyphen_joined_words_are_taken_whole_wherever_they_stand():
    assert tokenize("x9 CUST-ACCT-ID, (a-b_c-d) ab -e- 12-34 f--g") == [
        "cust-acct-id", "a-b_c-d", "x9", "cust", "acct", "id", "b_c", "ab"]


def test_a_long_run_is_split_in_time_that_grows_with_its_length():
    import time
    for piece in ("a", "a_", "a-", "-", "aB", "あ", "0f", "a."):
        for ext in (None, "cbl", "py"):
            start = time.monotonic()
            tokenize("fix " + piece * 100_000 + " now", ext)
            assert time.monotonic() - start < 5, (piece, ext)


def test_passing_ext_also_drops_that_languages_keywords():
    assert "dict" in tokenize("dict of values")
    assert "dict" not in tokenize("dict of values", "py")


def test_the_stopwords_differ_per_language():
    """interface is a keyword in Java and an ordinary English word in Python."""
    assert "interface" not in tokenize("public interface Foo", "java")
    assert "interface" in tokenize("the public interface", "py")


def test_an_unknown_extension_uses_the_base_stopwords_only():
    assert "dict" in tokenize("dict of values", "xyz")


def test_path_terms_converts_to_repo_relative_before_splitting():
    """A directory the index added for its own purposes, such as settings_py, stays out."""
    assert path_term_text("p/sub/settings_py/settings.py", "p") == "sub settings py "


def _repeat_doc_dict(doc_dict: dict, weight: int) -> dict:
    """Return the documents with the path words written out weight times before the
    code, which is the text the path words stand for."""
    return {key: ((path_text * weight) + code_text, prose_text, name_list)
            for key, (code_text, prose_text, name_list, path_text) in doc_dict.items()}


@pytest.mark.parametrize("weight", [0, 1, 3, 10])
def test_the_path_words_count_path_weight_times_as_if_they_were_written_out(monkeypatch,
                                                                         weight):
    monkeypatch.setattr(config, "PATH_WEIGHT", weight)
    defs = {
        "p/lib/CUST-ACCT_cbl/CUST-ACCT.cbl::REC": {
            "file": "p/lib/CUST-ACCT_cbl/CUST-ACCT.cbl", "signature": "01 CUST-ACCT.",
            "doc": "the account record", "hold_name_list": ["CUST-ACCT-ID"]},
        "p/src/parse_file_py/parse_file.py::load": {
            "file": "p/src/parse_file_py/parse_file.py", "signature": "def load(file):",
            "doc": "parse the file"}}
    files = {"p/src/parse_file_py/parse_file.py": {"summary": "parse helpers", "defs": []}}
    for doc_dict in (bm25_doc_dict(defs, "p"), file_bm25_doc_dict(files, "p")):
        index_dict = build_bm25(doc_dict)
        write_out_dict = build_bm25(_repeat_doc_dict(doc_dict, weight))
        assert index_dict == write_out_dict
        assert list(index_dict["postings"]) == list(write_out_dict["postings"])
    index_dict = build_bm25(bm25_doc_dict(defs, "p"))
    key = "p/src/parse_file_py/parse_file.py::load"
    assert dict(map(tuple, index_dict["postings"].get("parse_file", []))).get(key) \
        == (weight or None)


def test_a_path_weight_of_the_most_a_setting_takes_builds_an_index(monkeypatch):
    defs = {"p/a_py/a.py::f": {"file": "p/a_py/a.py", "signature": "def f():",
                               "doc": "does something"}}
    monkeypatch.setattr(config, "PATH_WEIGHT", 0)
    own_length = build_bm25(bm25_doc_dict(defs, "p"))["len"]["p/a_py/a.py::f"]
    monkeypatch.setattr(config, "PATH_WEIGHT", config.NUMBER_MAX)
    index_dict = build_bm25(bm25_doc_dict(defs, "p"))
    assert index_dict["postings"]["py"] == [["p/a_py/a.py::f", config.NUMBER_MAX]]
    assert index_dict["len"]["p/a_py/a.py::f"] == config.NUMBER_MAX + own_length


def test_bm25_docs_keeps_the_code_and_the_prose_apart():
    defs = {"p/a_py/a.py::f": {"file": "p/a_py/a.py", "signature": "def f():",
                               "doc": "does something"}}
    assert bm25_doc_dict(defs, "p") == {
        "p/a_py/a.py::f": ("def f():", "does something", [], "a py ")}


def test_file_bm25_docs_keeps_the_path_words_and_the_file_summary_apart():
    files = {"p/a_py/a.py": {"summary": "config helpers", "defs": []}}
    assert file_bm25_doc_dict(files, "p") == {
        "p/a_py/a.py": ("", "config helpers", [], "a py ")}


def test_a_file_with_neither_a_summary_nor_a_definition_stays_out_of_the_file_index(
        monkeypatch):
    """Its path words alone are a document short enough to outscore a described file."""
    monkeypatch.setattr(config, "PATH_WEIGHT", 1)
    files = {"p/README_md/README.md": {"summary": "", "defs": []},
             "p/a_py/a.py": {"summary": "", "defs": ["p/a_py/a.py::f"]},
             "p/b_py/b.py": {"summary": "helpers", "defs": []}}
    assert set(file_bm25_doc_dict(files, "p")) == {"p/a_py/a.py", "p/b_py/b.py"}


def test_build_bm25_gets_the_extension_from_a_file_path_key_too():
    """A file index key holds no "::"; ext_of still has to find py."""
    idx = build_bm25({"p/a_py/a.py": ("the dict of values", "")})
    # dict, a py keyword, being gone proves the extension was passed as py
    assert "dict" not in idx["postings"]
    assert "values" in idx["postings"]


def test_build_bm25_builds_the_inverted_index_and_the_average_length():
    idx = build_bm25({"a.py::f": ("parse config value", ""),
                      "a.py::g": ("parse config", "")})
    assert idx["postings"]["parse"] == [["a.py::f", 1], ["a.py::g", 1]]
    assert idx["len"] == {"a.py::f": 3, "a.py::g": 2}
    assert idx["avg_len"] == 2.5


def test_build_bm25_drops_keywords_by_the_definition_keys_extension():
    """The key starts with the file path, so ext comes from there into tokenize."""
    idx = build_bm25({"a.py::f": ("dict of values", "")})
    assert "dict" not in idx["postings"]


def test_build_bm25_keeps_a_keyword_that_is_an_english_word_in_the_prose():
    """"import" and "type" are Python keywords and ordinary words in a description."""
    idx = build_bm25({"a.py::f": ("def resolve(import_path):",
                                  "resolves an import to a file type")})
    assert "import" in idx["postings"]
    assert "type" in idx["postings"]


def test_bm25_rank_returns_results_score_descending(tmp_path):
    idx = scoring_index({"a.py::f": "parse config value config",
                         "a.py::g": "unrelated text here"}, _index_path(tmp_path))
    hits = bm25_rank("config", idx, "def")
    assert hits[0]["key"] == "a.py::f"
    assert hits[0]["hit_terms"] == 1
    idx.close()


def test_bm25_rank_leaves_out_what_the_language_calls_a_constant(tmp_path):
    """A constant is dropped when the index is read, not when it is written."""
    idx = scoring_index({"a.py::f": "config", "a.py::g": "config"},
                        _index_path(tmp_path),
                        types={"a.py::g": "expression_statement"})
    hits = bm25_rank("config", idx, "def")
    assert [h["key"] for h in hits] == ["a.py::f"]
    idx.close()


def test_score_sums_to_1_and_raw_is_the_value_before_the_division(tmp_path):
    idx = scoring_index({"a.py::f": "config value", "a.py::g": "config"},
                        _index_path(tmp_path))
    hits = bm25_rank("config", idx, "def")
    assert abs(sum(h["score"] for h in hits) - 1.0) < 1e-9
    total_raw = sum(h["raw"] for h in hits)
    assert all(abs(h["raw"] / total_raw - h["score"]) < 1e-9 for h in hits)
    idx.close()


def test_documents_with_no_word_of_their_own_are_scored_by_the_names_they_hold(tmp_path):
    """The average length is 0 where no document that takes part holds a word."""
    idx = scoring_index({"x.cpy::A": "", "x.cpy::B": ""}, _index_path(tmp_path),
                        names={"x.cpy::A": ["bb", "cc-dd"], "x.cpy::B": ["bb", "bb", "ee"]})
    assert idx.avg_len("def") == 0.0
    hits = bm25_rank("bb", idx, "def")
    assert [(h["key"], h["hit_terms"]) for h in hits] == [("x.cpy::B", 0), ("x.cpy::A", 0)]
    assert abs(sum(h["score"] for h in hits) - 1.0) < 1e-9
    assert [h["key"] for h in bm25_rank("cc-dd", idx, "def")] == ["x.cpy::A"]
    assert bm25_rank("nothing", idx, "def") == []
    idx.close()


def test_bm25_rank_returns_an_empty_list_when_nothing_matches(tmp_path):
    idx = scoring_index({"a.py::f": "config"}, _index_path(tmp_path))
    assert bm25_rank("nonexistentword", idx, "def") == []
    idx.close()


def test_a_term_most_documents_carry_is_not_counted_as_a_hit(tmp_path):
    """"to" and "of" tell documents apart barely at all, whatever they score."""
    doc_dict = {f"a.py::f{i}": f"config value to read {i}" for i in range(10)}
    doc_dict["a.py::special"] = "config parser to read"
    idx = scoring_index(doc_dict, _index_path(tmp_path))
    hits = bm25_rank("parser to read", idx, "def")
    # "to" and "read" are in every document, "parser" in one
    assert hits[0]["key"] == "a.py::special"
    assert hits[0]["hit_terms"] == 1
    idx.close()


def test_raw_is_per_term_that_hit(tmp_path):
    """One term hitting hard and five hitting weakly are told apart."""
    idx = scoring_index({"a.py::f": "config config config config",
                         "a.py::g": "alpha beta gamma delta epsilon"},
                        _index_path(tmp_path))
    one = bm25_rank("config", idx, "def")[0]
    many = bm25_rank("alpha beta gamma delta epsilon", idx, "def")[0]
    assert many["hit_terms"] == 5
    assert many["raw"] < one["raw"] * many["hit_terms"]
    idx.close()
