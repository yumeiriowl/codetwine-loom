"""The scope fields added in search steps 1 and 4.

Step 1 (def_detail, file_summaries) and step 4 (search_file_list, rrf_merge, spread) are
checked with conftest's fake_index installed as IDX.
"""

import hashlib
import logging
import os

import pytest

from codetwine_loom import config
from conftest import fake_defs, fake_files, write_index
from codetwine_loom.paths import repo_to_knowledge
from codetwine_loom.search import scope as scope_mod
from codetwine_loom.search.scope import (def_detail, file_summary_dict, name_hit_list,
                                            rrf_merge, search_file_list, search_scope,
                                            search_scope_from_query)


# == Step 1: line ranges, first lines and file summaries in the scope ========

def test_def_detail_adds_the_line_range_and_the_first_line(fake_index):
    d = def_detail("p/cfg_py/cfg.py::load_config")
    assert d == {"key": "cfg.py::load_config", "lines": [10, 42],
                 "signature": "def load_config(path):"}


@pytest.fixture
def doc_proj(tmp_path, monkeypatch, index_of):
    """fake_index over a PROJECT_DIR holding its 2 files, both design documents current.

    Returns:
        The PROJECT_DIR path.
    """
    root = tmp_path / "proj"
    root.mkdir()
    file_dict = fake_files()
    for name in ("cfg.py", "util.py"):
        (root / name).write_text(f"# {name}\n")
        file_dict[f"p/{name.replace('.', '_')}/{name}"]["doc_hash"] = \
            hashlib.sha256(f"# {name}\n".encode()).hexdigest()
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    index_of(fake_defs(), file_dict)
    return root


def test_file_summaries_is_keyed_by_repository_relative_path(doc_proj):
    out = file_summary_dict(["p/cfg_py/cfg.py"])
    assert out == {"cfg.py": "This file reads and writes settings."}


def test_file_summaries_leaves_out_empty_summaries(doc_proj):
    """A file with an empty summary stays in files but not in file_summaries."""
    out = file_summary_dict(["p/cfg_py/cfg.py", "p/util_py/util.py"])
    assert list(out) == ["cfg.py"]


def test_file_summaries_cuts_at_summary_max_chars(doc_proj, monkeypatch):
    monkeypatch.setattr(config, "SUMMARY_MAX_CHAR", 10)
    summary = file_summary_dict(["p/cfg_py/cfg.py"])["cfg.py"]
    assert summary.startswith("This file \n... (") and summary.endswith(" chars omitted)")


def test_file_summaries_leaves_out_a_file_changed_since_its_design_document(doc_proj):
    (doc_proj / "cfg.py").write_text("# cfg.py\nchanged = 1\n")
    assert file_summary_dict(["p/cfg_py/cfg.py"]) == {}


def test_file_summaries_leaves_out_a_design_document_without_a_hash(tmp_path, monkeypatch,
                                                                    fake_index):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "cfg.py").write_text("# cfg.py\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    assert file_summary_dict(["p/cfg_py/cfg.py"]) == {}


def test_file_summaries_leaves_out_a_file_that_is_gone(doc_proj):
    (doc_proj / "cfg.py").unlink()
    assert file_summary_dict(["p/cfg_py/cfg.py"]) == {}


def test_a_current_design_document_is_told_apart_from_one_that_is_not(doc_proj):
    assert scope_mod.doc_is_current("p/cfg_py/cfg.py")
    assert not scope_mod.doc_is_current("p/nosuch_py/nosuch.py")


def test_a_file_a_named_pipe_stands_in_for_has_no_current_design_document(doc_proj):
    (doc_proj / "cfg.py").unlink()
    os.mkfifo(doc_proj / "cfg.py")
    assert not scope_mod.doc_is_current("p/cfg_py/cfg.py")
    assert scope_mod.doc_is_current("p/util_py/util.py")


def test_a_stage_is_handed_a_copy_of_the_search_result():
    scope = {"files": ["a.py"], "defs": [{"key": "a.py::f", "lines": [1, 2]}],
             "confidence": "high"}
    variable_dict = scope_mod.stage_variable_dict(scope, {})
    assert variable_dict == {"scope": scope}
    variable_dict["scope"]["files"].append("b.py")
    variable_dict["scope"]["defs"][0]["lines"][0] = 9
    del variable_dict["scope"]["confidence"]
    assert scope == {"files": ["a.py"], "defs": [{"key": "a.py::f", "lines": [1, 2]}],
                     "confidence": "high"}
    assert scope_mod.stage_variable_dict(None, {}) == {"scope": None}


def test_scope_defs_is_a_list_of_dicts(fake_index):
    """It went from a list of strings to a list of dicts; no compatibility key."""
    scope = search_scope_from_query("configuration file")
    assert scope["defs"]
    assert all(set(d) == {"key", "lines", "signature"} for d in scope["defs"])


# == Step 4: the file index, RRF and spread =================================

def test_search_files_finds_a_file_by_its_summary(fake_index):
    assert search_file_list("settings") == ["p/cfg_py/cfg.py"]


def test_search_files_finds_a_file_by_a_word_of_its_name(fake_index):
    """Path words are searchable, so a word absent from the summary hits."""
    assert "p/util_py/util.py" in search_file_list("util")


# == Step 4: a file named in the query =======================================

@pytest.fixture
def named_index(index_of):
    """fake_index plus a file with neither a summary nor a definition."""
    return index_of(fake_defs(), {**fake_files(), "p/pyproject_toml/pyproject.toml":
                                  {"summary": "", "defs": []}})


def _same_name_files() -> dict:
    """3 files of one name in 3 directories, one of them with a summary."""
    return {"p/a/x_py/x.py": {"summary": "", "defs": []},
            "p/b/x_py/x.py": {"summary": "parses the settings file", "defs": []},
            "p/c/x_py/x.py": {"summary": "", "defs": []}}


@pytest.fixture
def same_name_index(index_of):
    """fake_index plus 3 files named x.py."""
    return index_of(fake_defs(), {**fake_files(), **_same_name_files()})


def test_a_file_named_in_the_query_is_a_hit_even_when_it_is_not_in_the_file_index(named_index):
    assert named_index.posting_list("file", "pyproject") == []
    assert search_scope_from_query("add a dependency to pyproject.toml")["files"][0] == \
        "pyproject.toml"


def test_the_file_index_itself_does_not_put_a_name_first(named_index):
    assert "p/pyproject_toml/pyproject.toml" not in \
        search_file_list("add a dependency to pyproject.toml")


@pytest.mark.parametrize("query", ["toml", "project", "pyproject", "toml pyproject"])
def test_a_word_of_a_file_name_alone_is_not_its_name(named_index, query):
    assert "pyproject.toml" not in search_scope_from_query(query)["files"]


@pytest.mark.parametrize("query", ["PYPROJECT.TOML", "pyproject toml", "src/pyproject.toml"])
def test_a_name_is_matched_whatever_the_case_and_the_separators(named_index, query):
    assert name_hit_list([query]) == (["p/pyproject_toml/pyproject.toml"], [])


@pytest.mark.parametrize("request_text", [
    "add a dependency to `pyproject.toml`",
    "add a dependency to pyproject.toml, then lock it",
    "add a dependency at pyproject.toml:42",
    "add a dependency (pyproject.toml)",
])
def test_a_name_with_marks_around_it_is_a_sure_hit(named_index, request_text):
    assert name_hit_list(scope_mod.name_list_of(request_text)) == \
        (["p/pyproject_toml/pyproject.toml"], [])
    assert search_scope(request_text)["files"][0] == "pyproject.toml"


def test_a_name_is_split_at_every_mark_and_kept_whole_at_a_hyphen():
    assert scope_mod._path_word_list("`store/memory.go:42`,") == ["store", "memory", "go", "42"]
    assert scope_mod._path_word_list("my-file.js __init__.py") == \
        ["my-file", "js", "__init__", "py"]


@pytest.fixture
def plain_name_index(index_of):
    """fake_index plus scripts/test, a file whose name is one plain word."""
    return index_of(fake_defs(), {**fake_files(),
                                  "p/scripts/test/test": {"summary": "", "defs": []}})


def test_a_plain_word_the_expansion_returns_is_not_a_file_name(plain_name_index,
                                                              monkeypatch):
    monkeypatch.setattr(config, "QUERY_EXPAND", True)
    scope = search_scope("設定の読み込みを直す", llm=lambda _: "configuration test")
    assert scope["trace"]["query"] == "configuration test"
    assert "scripts/test" not in scope["files"][:1]


def test_a_plain_word_in_an_english_request_is_not_a_file_name(plain_name_index):
    assert "scripts/test" not in search_scope("add a test for the configuration")["files"][:1]


@pytest.mark.parametrize("request_text", ["`scripts/test` を直す", "fix scripts/test"])
def test_a_plain_word_file_written_as_a_name_is_a_sure_hit(plain_name_index, monkeypatch,
                                                          request_text):
    monkeypatch.setattr(config, "QUERY_EXPAND", True)
    scope = search_scope(request_text, llm=lambda _: "configuration")
    assert scope["files"][0] == "scripts/test"


def test_search_again_terms_take_file_names_the_way_a_request_does(plain_name_index):
    """search_scope_from_query() without a name text looks for the names written in its
    query, so a plain word is no file name there either."""
    assert search_scope_from_query("configuration test")["files"][0] != "scripts/test"
    assert search_scope_from_query("configuration scripts/test")["files"][0] == \
        "scripts/test"


@pytest.fixture
def path_word_index(index_of):
    """fake_index plus bin/run, a file of one word, and pkg/run/main.py."""
    return index_of(fake_defs(), {**fake_files(),
                                  "p/bin/run/run": {"summary": "", "defs": []},
                                  "p/pkg/run/main_py/main.py": {"summary": "", "defs": []}})


@pytest.mark.parametrize("request_text, hit_list", [
    ("fix the port check in pkg/run/main.py", ["p/pkg/run/main_py/main.py"]),
    ("update config.run.timeout handling", []),
    ("make server.run retry", []),
    ("`pkg` and `main.py` are two names", ["p/pkg/run/main_py/main.py"]),
    ("fix bin/run", ["p/bin/run/run"]),
    ("fix `run`", ["p/bin/run/run"]),
])
def test_a_word_inside_a_written_name_is_not_a_file_name(path_word_index, request_text,
                                                        hit_list):
    assert name_hit_list(scope_mod.name_list_of(request_text)) == (hit_list, [])


@pytest.mark.parametrize("name", ["run/main.py:42", "pkg/run/main.py:3-9",
                                  "main.py::start_server", "./pkg/run/main.py",
                                  "pkg/run/main.py:42:7", "pkg/run/main.py:line",
                                  "pkg/run/main.py:42:def", "run/main.py:start_server",
                                  "C:/app/pkg/run/main.py:42"])
def test_a_name_with_line_numbers_or_a_definition_behind_it_names_the_file(
        path_word_index, name):
    assert name_hit_list(scope_mod.name_list_of(name)) == (["p/pkg/run/main_py/main.py"], [])
    assert search_scope_from_query(f"configuration {name}")["files"][0] == \
        "pkg/run/main.py"


@pytest.mark.parametrize("request_text, hit_list", [
    ("at App.Run.Main() in /src/pkg/run/main.py:line 42", ["p/pkg/run/main_py/main.py"]),
    ("pkg/run/main.py:42:def start_server(port):", ["p/pkg/run/main_py/main.py"]),
    ("pkg/run/main.py:start_server is wrong", ["p/pkg/run/main_py/main.py"]),
    ("pkg/run/main.py:42: error: no port", ["p/pkg/run/main_py/main.py"]),
    ("bin/run:7 fails", ["p/bin/run/run"]),
    ("listen on `run.2`", []),
    ("see `main.py.orig`", []),
])
def test_what_is_written_behind_the_colon_after_a_file_name_is_left_out(
        path_word_index, request_text, hit_list):
    assert name_hit_list(scope_mod.name_list_of(request_text)) == (hit_list, [])


_MAIN_PATH, _CFG_PATH = "p/pkg/run/main_py/main.py", "p/cfg_py/cfg.py"


@pytest.mark.parametrize("request_text, hit_list", [
    ("`pkg/run/main.py#L10` を直す", [_MAIN_PATH]),
    ("`pkg/run/main.py#L10-L20` を直す", [_MAIN_PATH]),
    ("`pkg/run/main.py#start` を直す", [_MAIN_PATH]),
    ("`pkg/run/main.py(new)` を足す", [_MAIN_PATH]),
    ("`pkg/run/main.py（新規）` を足す", [_MAIN_PATH]),
    ("`pkg/run/main.py@abc123` を見る", [_MAIN_PATH]),
    ("`pkg/run/main.py?plain=1` を見る", [_MAIN_PATH]),
    ("`pkg/run/main.py,cfg.py` を直す", [_CFG_PATH, _MAIN_PATH]),
    ("`pkg/run/main.py、cfg.py` を直す", [_CFG_PATH, _MAIN_PATH]),
    ("`pkg/run/main.py|cfg.py` を直す", [_CFG_PATH, _MAIN_PATH]),
    ("`pkg/run/main.py;cfg.py` を直す", [_CFG_PATH, _MAIN_PATH]),
    ("pkg/run/main.py#L10 と pkg/run/main.py(new)", [_MAIN_PATH]),
    ("`[run]/main.py` を直す", [_MAIN_PATH]),
    ("`(run)/main.py` を直す", [_MAIN_PATH]),
    ("`{run}/main.py` を直す", [_MAIN_PATH]),
    ("`<project>/pkg/run/main.py` を直す", [_MAIN_PATH]),
    ("`${HOME}/main.py` と `$(pwd)/cfg.py`", [_CFG_PATH, _MAIN_PATH]),
    ("`[the entry](pkg/run/main.py)` を直す", [_MAIN_PATH]),
    ("[run]/main.py を直す", [_MAIN_PATH]),
    ("C:\\app\\pkg\\run\\main.py を直す", [_MAIN_PATH]),
])
def test_a_file_written_with_something_around_or_behind_it_is_a_sure_hit(
        path_word_index, request_text, hit_list):
    assert name_hit_list(scope_mod.name_list_of(request_text)) == (hit_list, [])

# The files of the index the names of _NAME_TABLE are looked up in: the files its
# requests name, and files of one plain word
_NAME_TABLE_FILE_LIST = [
    "store/memory.go", "cache/memory.go", "store/memory.py", "pkg/run/main.py", "bin/run",
    "settings.py", "a.py", "src/a.py", "src/b.py", "src/main.py", "main.py", "cfg.py",
    "util.py", "Makefile", "src/Makefile", "2to3.py", "pyproject.toml", "LICENSE",
    "docs/設計.md", "src/routes/+page.svelte", "app/[slug]/page.tsx", "app/(shop)/page.tsx",
    "app/(shop)/{id}/@modal/~x/$y/page.tsx", "src/x.py", "src/x/y.py", "src/x.c", "src/x.h",
    "assets/icon@2x.png", "docs/report(1).pdf", "config.py", "include/stdio.h",
    "include/sys/stat.h", ".bashrc", "a/b.py", "c/d.py", "pkg/mod.py", "src/app.yaml",
    "docs/x.md", "docs/b.md", "path/to/file.py", "src/+page.js", "memory.py", "Store.cs",
    "foo.py", "c.py", "handler.py", "request.json", "path/x.py", "README.md", "CHANGELOG.md",
    "src/models/User.php", "os/path.py", "d/e.py", "f.py", "x.toml", ".go", "src/[name].py",
    "docs/User Guide.md", "Guide.md", "docs/Release Notes.md", "src/Controller.cs",
    "Assets/Scripts/Player Controller.cs",
    "new", "main", "text", "open", "x", "config", "exclude", "HOME", "name", "L10", "plain",
    "line", "def", "load", "host", "origin", "abc123", "slug", "shop", "pkg", "src", "app",
    "foo", "Vec", "b", "z", "y", "std", "key", "value", "note", "error", "C", "pwd",
    "project", "doc", "html", "py", "go", "path", "os", "code", "WIP", "1", "42", "user",
]

# What a request names, by how it is written: a line is (the request, the names taken
# from it against the index of _NAME_TABLE_FILE_LIST)
_NAME_TABLE = [
    # a path, a file name and an identifier, bare
    ("store/memory.go を直す", ["store/memory.go"]),
    ("store/memory.py を直す", ["store/memory.py"]),
    ("pkg/run/main.py を直す", ["pkg/run/main.py"]),
    ("tree-sitter-go を足す", []),
    ("resolve_module_to_project_path を呼ぶ", []),
    ("loadConfig() を直す", []),
    ("settings.py. を見る", ["settings.py"]),
    ("a.py と a.py の両方", ["a.py"]),
    ("Makefile を直す", []),
    ("2026-09-22 に 1e-5 を", []),
    ("Python 3.10 と v1.2 で動かす", []),
    ("src/ と a/b/ を見る", []),
    ("loadConfig(path) と foo(a/b)", ["b"]),
    ("src/routes/+page.svelteを直す", ["src/routes/+page.svelte"]),
    ("import os.path; from a.b import c", []),
    # the same in backquotes, where a word that stands alone needs no mark of a name
    ("`Makefile` を直す", ["Makefile"]),
    ("`.go` ファイルを対象にする", [".go"]),
    ("`2to3.py` と 2to3.py", ["2to3.py"]),
    ("`loadConfig()` を呼ぶ", []),
    ("`顧客番号` の桁を 12 にして", []),
    ("`CUST-ACCT-ID` を直す", []),
    ("`docs/設計.md` を直す", ["docs/設計.md"]),
    ("`pyproject.toml と `LICENSE`", ["pyproject.toml", "LICENSE"]),
    ("`two words` を見る", []),
    ("`git log --oneline` then `rm -rf build/`", []),
    ("`!=` と `->` を見る", []),
    ("`foo(bar)[0]` と `Vec<String>`", []),
    ("`x=y/z` を見る", ["z"]),
    ("`--no-overview` を足す", []),
    ("`--config=src/app.yaml` と `--exclude=build/`", ["src/app.yaml"]),
    # the marks in front of a name are no part of it
    ("`/store/memory.py` を直す", ["store/memory.py"]),
    ("`@app.route` と `$HOME/x.toml`", ["x.toml"]),
    ('`"config.py"` と `<stdio.h>` と `~/.bashrc`', ["config.py", "stdio.h", ".bashrc"]),
    ('`<sys/stat.h>` と `"a/b.py"`', ["sys/stat.h", "a/b.py"]),
    ("//pkg:mod.py と ./src/a.py", ["mod.py", "src/a.py"]),
    # a part of a path is compared by its words, whatever marks it is written with
    ("`app/[slug]/page.tsx` を直す", ["app/[slug]/page.tsx"]),
    ("app/[slug]/page.tsx を直す", ["app/[slug]/page.tsx"]),
    ("`src/routes/+page.svelte` を直す", ["src/routes/+page.svelte"]),
    ("`app/(shop)/{id}/@modal/~x/$y/page.tsx`", ["app/(shop)/{id}/@modal/~x/$y/page.tsx"]),
    ("app/(shop)/{id}/@modal/~x/$y/page.tsx を", ["app/(shop)/{id}/@modal/~x/$y/page.tsx"]),
    ("`C:\\src\\x.py` を直す", ["src/x.py"]),
    ("C:\\src\\x.py を直す", ["src/x.py"]),
    ("assets/icon@2x.png と src/<name>.py", ["assets/icon@2x.png", "src/[name].py"]),
    ("docs/report(1).pdf を見る", ["docs/report(1).pdf"]),
    ("`[slug]/page.tsx` と `(shop)/page.tsx`", ["[slug]/page.tsx", "(shop)/page.tsx"]),
    ("[slug]/page.tsx と (shop)/page.tsx", ["[slug]/page.tsx", "(shop)/page.tsx"]),
    ("`<project>/src/main.py` を直す", ["src/main.py"]),
    ("`${HOME}/cfg.py` と `$(pwd)/util.py`", ["cfg.py", "util.py"]),
    ("`{name}/cfg.py` を直す", ["cfg.py"]),
    # what is written around a path or behind it is no part of it
    ("fix (app/[slug]/page.tsx).", ["app/[slug]/page.tsx"]),
    ('「cfg.py」 と （util.py） と "a/b.py" と \'c/d.py\'',
     ["cfg.py", "util.py", "a/b.py", "c/d.py"]),
    ("see [the doc](docs/x.md) and [src/a.py](docs/b.md)",
     ["docs/x.md", "src/a.py", "docs/b.md"]),
    ("[text](path/to/file.py) を見る", ["path/to/file.py"]),
    ("`[text](path/to/file.py)` を見る", ["path/to/file.py"]),
    ("`open(src/main.py)` と open(src/main.py)", ["src/main.py"]),
    ("cfg.py#load_config と MemoryStore#loadConfig", ["cfg.py"]),
    ("UserStore#load_config と UserStore#save_config", []),
    ("isReady?loadConfig():saveConfig()", []),
    ("`src/main.py#L10` を直す", ["src/main.py"]),
    ("src/main.py#L10 を直す", ["src/main.py"]),
    ("`src/main.py#L10-L20` と `src/b.py#main`", ["src/main.py", "src/b.py"]),
    ("`main.py#L10` を直す", ["main.py"]),
    ("`app/[slug]/page.tsx#L3` を直す", ["app/[slug]/page.tsx"]),
    ("`src/main.py?plain=1` を直す", ["src/main.py"]),
    ("`src/main.py(new)` と src/b.py(new)", ["src/main.py", "src/b.py"]),
    ("`src/main.py（新規）` を足す", ["src/main.py"]),
    ("`src/Makefile(new)` を足す", ["src/Makefile"]),
    ("`src/main.py@abc123` を見る", ["src/main.py"]),
    ("`src/main.py*` と `src/b.py$`", ["src/main.py", "src/b.py"]),
    ("`src/main.py:main()` を直す", ["src/main.py"]),
    ("src/x/y.py:42:config.load() で落ちる", ["src/x/y.py"]),
    ("store/memory.py::load_config#function#4 を直す", ["store/memory.py"]),
    # a comma or another separator stands between two names
    ("`src/main.py,cfg.py` を直す", ["src/main.py", "cfg.py"]),
    ("`src/main.py、cfg.py` と `a/b.py|c.py` と `d/e.py;f.py`",
     ["src/main.py", "cfg.py", "a/b.py", "c.py", "d/e.py", "f.py"]),
    ("app/(shop)/page.tsx, then src/+page.js,src/b.py",
     ["app/(shop)/page.tsx", "src/+page.js", "src/b.py"]),
    ("src/a.py#L1,src/b.py を直す", ["src/a.py", "src/b.py"]),
    ("テーブル|src/main.py|cfg.py|", ["src/main.py", "cfg.py"]),
    # a colon stands between two names, or between a name and what is none
    ("memory.py:42 と Store.cs:line 42", ["memory.py", "Store.cs"]),
    ("store/memory.py:42:def load を直す", ["store/memory.py"]),
    ("`store/memory.py::load_config` を直す", ["store/memory.py"]),
    ("`Makefile:12` と Makefile:12", ["Makefile"]),
    ("git show main:a.py と origin/main:a.py", ["a.py", "main"]),
    ("host:a.py と HEAD:src/x.py と a.py:b.py", ["a.py", "src/x.py", "b.py"]),
    ("store/memory.py:42:a/b と user@host:path/x.py", ["store/memory.py", "b", "path/x.py"]),
    ("error: foo.py not found と note:main.py", ["foo.py", "main.py"]),
    ("`std::vector` と std::vector と `key:value`", []),
    ("a:b:c.py と handler.py:request.json", ["c.py", "handler.py", "request.json"]),
    ("C: と 12:30 を見る", []),
    # a URL names no file
    ("https://x.y/doc.html を読む", []),
    ("`https://x.y/z` を見る", []),
    ("https://x.y/doc.html?a=b_c#frag-ment を読む", []),
    ("(https://x.y/a/main.py), <https://x.y/cfg.py>. と http://host:8080/x.py", []),
    ("[src/a.py](https://x.y/z) を見る", ["src/a.py"]),
    ("https://x.y/z、src/a.py を見る", ["src/a.py"]),
    # a note in brackets behind a path, followed by a mark that ends the sentence
    ("Add src/b.py(new).", ["src/b.py"]),
    ("Add src/a.py(new) and src/b.py(new).", ["src/a.py", "src/b.py"]),
    ("src/b.py(new): the loader", ["src/b.py"]),
    ("src\\x.c(10): error C2143: syntax error", ["src/x.c"]),
    ("C:\\proj\\src\\x.c(10): error C2143", ["src/x.c"]),
    ("#0 /var/www/src/models/User.php(42): load()", ["src/models/User.php"]),
    ("> /home/u/proj/src/main.py(42)load()", ["src/main.py"]),
    ("`src/main.py(new).` と `src/main.py(42):`", ["src/main.py"]),
    ("app/(shop)/page.tsx(new).", ["app/(shop)/page.tsx"]),
    # two names joined by a path separator or a mark
    ("README.md/CHANGELOG.md を更新", ["README.md", "CHANGELOG.md"]),
    ("x.c/x.h と src/x.c/x.h", ["x.c", "x.h", "src/x.c"]),
    ("src/main.py/cfg.py の両方", ["src/main.py", "cfg.py"]),
    ("cfg.py/load_config と src/main.py/Loader/load", ["cfg.py", "src/main.py", "load"]),
    ("src/main.py+cfg.py と src/x.c~src/x.h",
     ["src/main.py", "cfg.py", "src/x.c", "src/x.h"]),
    # a tag or a number in front of a path
    ("<code>src/main.py</code> を直す", ["src/main.py"]),
    ("<b>cfg.py</b> と <li>src/a.py</li>", ["cfg.py", "src/a.py"]),
    ("(1)src/main.py (2)cfg.py", ["src/main.py", "cfg.py"]),
    ("[WIP]src/main.py: crash", ["src/main.py"]),
    ("1.src/main.py 2.cfg.py を直す", ["src/main.py", "cfg.py"]),
    ("cfg.py#load_config と Store.cs#L10", ["cfg.py", "Store.cs"]),
    ("src/main.py->cfg.py", ["src/main.py", "cfg.py"]),
    # a name joined to a word by a dot, a file of one plain word, a file named with a dot
    ("see `main.py.orig` and foo.main.py", []),
    ("config.run.timeout と server.run", []),
    ("fix `run` and `run:7`, not run or `run:start`", ["run"]),
    ("./run と bin/run:7 と pkg/run", ["run", "bin/run"]),
    ("の.go と 設定.py と .bashrc", [".go", ".bashrc"]),
    ("fix README.md.", ["README.md"]),
    ("`*.py` と `.go` と *.go", [".go"]),
    ("a/b と 1/2 と and/or と `/x`", ["b", "x"]),
    # a mark that ends the sentence behind a name, with more written right behind it
    ("<li>Fix src/a.py.</li> と <p>Fix cfg.py.</p>", ["src/a.py", "cfg.py"]),
    ('{"message": "error in src/main.py.\\nTraceback (most recent call last)"}',
     ["src/main.py"]),
    ("- cfg.py.\\n- util.py.\\r\\n", ["cfg.py", "util.py"]),
    ("see cfg.py.[1] and util.py.[^1]", ["cfg.py", "util.py"]),
    ("Fix cfg.py.(see README.md) と a.py...and CHANGELOG.md",
     ["cfg.py", "README.md", "a.py", "CHANGELOG.md"]),
    ("Fix cfg.py.**Important** と src/a.py.,util.py と f.py.<br>",
     ["cfg.py", "src/a.py", "util.py", "f.py"]),
    ("Fix bin/run.</p> と `Makefile`.</li>", ["bin/run", "Makefile"]),
    ("Fix cfg.py.Then と util.py._ と `f.py.bak` と main.py.orig.</p>", []),
    # a file with no extension written from the directory it is read in, behind a mark
    ('"./Makefile" と \'./LICENSE\' と `./run`', ["Makefile", "LICENSE", "run"]),
    ("[Makefile](./Makefile) の build を直す", ["Makefile"]),
    ("run (./Makefile) と $(./LICENSE) と CMD=./run", ["Makefile", "LICENSE", "run"]),
    ('<code>./Makefile</code> と open("./LICENSE") と --file=./run',
     ["Makefile", "LICENSE", "run"]),
    ('".\\Makefile" と "../LICENSE" と "./src/Makefile" と "./bin/run"',
     ["Makefile", "LICENSE", "src/Makefile", "bin/run"]),
    ('"Makefile" と (LICENSE) と </main> と "/config" と =/run と [new](x)', []),
    # a file whose name holds a space
    ("docs/User Guide.md を直す", ["docs/User Guide.md"]),
    ('`docs/User Guide.md` と "Assets/Scripts/Player Controller.cs"',
     ["docs/User Guide.md", "Assets/Scripts/Player Controller.cs"]),
    ("Fix User Guide.md and Guide.md, then release notes.md.",
     ["User Guide.md", "Release Notes.md", "Guide.md"]),
    ("assets\\scripts\\player controller.cs:42 で落ちる",
     ["Assets/Scripts/Player Controller.cs"]),
    ("old/docs/User Guide.md, xdocs/Release Notes.md を直す",
     ["docs/User Guide.md", "Release Notes.md"]),
    ("User Guide.md.orig と AUser Guide.md と Release Notes.mdx", ["Guide.md"]),
    ("the User と Guide.md と Player\nController.cs", ["Guide.md", "Controller.cs"]),
    # backquotes around words with a space between them hold no name, and the closing
    # one of the two opens no word
    ("`make test`が落ちるので`Makefile`を直して", ["Makefile"]),
    ("`docker build .`で使う`Makefile`と`LICENSE`を直す", ["Makefile", "LICENSE"]),
    ("`a b`,`Makefile` と see `make test`(`LICENSE`)", ["Makefile", "LICENSE"]),
    ("`docs/User Guide.md`と`Makefile`を直す", ["docs/User Guide.md", "Makefile"]),
    ("```make test``` then `Makefile`", ["Makefile"]),
    ("`a.py`\n`x y``cfg.py`\n` stray `LICENSE`", ["a.py", "cfg.py", "LICENSE"]),
    # a run of two, three or four backquotes is one mark around a word
    ("Fix ``Makefile`` and ``LICENSE``.", ["Makefile", "LICENSE"]),
    ("``Makefile``と````LICENSE````を直して", ["Makefile", "LICENSE"]),
    ("``main`` を `a.py` から呼ぶ", ["main", "a.py"]),
    ("``` `Makefile` ``` と `` `LICENSE` `` を直す", ["Makefile", "LICENSE"]),
    ("`make test` と ``Makefile``、``make test`` と `LICENSE`", ["Makefile", "LICENSE"]),
    ("``make test``が落ちるので``Makefile``を直して", ["Makefile"]),
    # a run of backquotes nothing closes, beside names in doubled ones
    ("``Makefile`` と ``LICENSE を直して", ["Makefile"]),
    ("``Makefile`` and ``LICENSE need a fix", ["Makefile"]),
    ("``Makefile`` を直す。`` は空のまま", ["Makefile"]),
    ("```` で始まるブロックの ``Makefile`` を直す", ["Makefile"]),
    ("```sh のあとの ` と ``Makefile`` を直す", ["Makefile"]),
    ("``main`` が止まらない。``Makefile`` も `` で囲む", ["main", "Makefile"]),
    ("``Makefile`` と ``LICENSE` を直して", ["Makefile", "LICENSE"]),
    ("``docs/設計.md`` を更新", ["docs/設計.md"]),
]


@pytest.fixture
def name_table_index(index_of):
    """An index of the files of _NAME_TABLE_FILE_LIST, none with a summary or a definition."""
    return index_of(fake_defs(), {repo_to_knowledge(path, "p"): {"summary": "", "defs": []}
                                  for path in _NAME_TABLE_FILE_LIST})


@pytest.mark.parametrize("request_text, name_list", _NAME_TABLE)
def test_the_names_of_a_request_are_the_files_of_the_index_it_writes(name_table_index,
                                                                    request_text, name_list):
    assert scope_mod.name_list_of(request_text) == name_list


def test_a_file_whose_name_holds_a_space_comes_before_the_file_named_like_its_last_word(
        name_table_index):
    files = search_scope("docs/User Guide.md と Player Controller.cs を直す",
                         llm=lambda prompt: "guide")["files"]
    assert files[:2] == ["docs/User Guide.md", "Assets/Scripts/Player Controller.cs"]
    assert "Guide.md" not in files and "src/Controller.cs" not in files
    assert name_hit_list(scope_mod.name_list_of("Guide.md と User Guide.md")) == (
        [repo_to_knowledge("docs/User Guide.md", "p"), repo_to_knowledge("Guide.md", "p")], [])


def test_a_text_is_searched_for_names_with_a_space_only_where_the_index_has_one(
        index_of, monkeypatch):
    index_of(fake_defs(), {**fake_files(), "p/docs/b_md/b.md": {"summary": "", "defs": []}})
    monkeypatch.setattr(scope_mod, "_space_name_span_list", None)
    assert scope_mod.name_list_of("docs/a b.md を直す") == ["b.md"]


@pytest.mark.parametrize("request_text, span_list", [
    ("fix a b.txt", [(4, 11, "a b.txt")]),
    ("x/a b.txt と Y\\A B.TXT", [(0, 9, "x/a b.txt"), (12, 21, "y/a b.txt")]),
    ("x/y/a b.txt を直す", [(0, 11, "x/y/A B.TXT")]),
    ("see my docs/a b.txt.", [(4, 19, "my docs/a b.txt")]),
    ("zdocs/a b.txt", [(6, 13, "a b.txt")]),
    ("a a a", [(0, 3, "a a")]),
    ("z/a a a a", [(0, 5, "z/a a"), (6, 9, "a a")]),
    ("(- notes.md)", [(1, 11, "- notes.md")]),
    ("x- notes.md", []),
    ("a b.txt.orig と a b.txt_x と a b.txt.", [(27, 34, "a b.txt")]),
    ("data b.txt と A  B.TXT", []),
])
def test_a_name_with_a_space_is_found_with_the_directories_written_before_it(
        index_of, request_text, span_list):
    """Of the files of one name, the one the most directories are written of is named,
    and of several the first of the index."""
    index_of(fake_defs(), {repo_to_knowledge(path, "p"): {"summary": "", "defs": []}
                           for path in ("x/a b.txt", "y/a b.txt", "x/y/A B.TXT",
                                        "my docs/a b.txt", "docs/a b.txt", "a a", "z/a a",
                                        "- notes.md", "b.txt")})
    assert scope_mod._space_name_span_list(request_text, scope_mod._file_table()) \
        == span_list


def test_names_with_a_space_are_read_in_time_that_grows_with_the_text_alone(index_of):
    """Neither how many such files a text names nor how many files have one such name
    multiplies the time."""
    import time
    file_count = 3000
    path_list = [f"Assets/Mod{number % 53}/Player Controller {number}.cs"
                 for number in range(file_count)] \
        + [f"pkg/p{number}/Shared Name.txt" for number in range(file_count)]
    index_of(fake_defs(), {repo_to_knowledge(path, "p"): {"summary": "", "defs": []}
                           for path in path_list})
    for line_list, name_count in (
            ([f"{path}(3,5): error: no name\n" for path in path_list[:file_count]],
             file_count),
            ([f"{path}: unused\n" for path in path_list[file_count:]], file_count),
            (["fix Shared Name.txt and "] * file_count, 1)):
        start = time.monotonic()
        name_list = scope_mod.name_list_of("".join(line_list))
        assert time.monotonic() - start < 5, line_list[0]
        assert len(name_list) == name_count


def test_a_name_is_spelled_as_the_index_spells_it_however_it_is_written(name_table_index):
    assert scope_mod.name_list_of("fix SRC\\Models\\user.PHP and <slug>/Page.tsx") == \
        ["src/models/User.php", "[slug]/page.tsx"]


def test_a_text_is_read_without_listing_the_files_of_the_index_again(name_table_index,
                                                                    monkeypatch):
    """The table of the index's files is built once, whatever the text holds."""
    call_list = []
    path_list = name_table_index.file_path_list()
    monkeypatch.setattr(name_table_index, "file_path_list",
                        lambda: call_list.append(1) or path_list)
    text = "".join(f"第{number}章の仕様.md と項目_v2 と cfg.py を直す\n" for number in range(300))
    assert scope_mod.name_list_of(text) == ["cfg.py"]
    assert scope_mod.is_index_name(".go") and not scope_mod.is_index_name(".md")
    name_hit_list(["cfg.py", "src/main.py"])
    assert len(call_list) <= 1


def test_a_long_text_is_read_for_names_in_time_that_grows_with_its_length(name_table_index):
    """Each of these is read once from end to end, whatever it repeats."""
    import time
    for piece in ("a/b)", "x/", "[a]/", "`a/b`", "a:b/", "\\", "a.", "/a.b", "a,", "cfg.py,",
                  "src/main.py/", "run,", "a-", "://", ".go", "main.", "x.c(1):"):
        start = time.monotonic()
        scope_mod.name_list_of("fix " + piece * 50_000 + " now")
        assert time.monotonic() - start < 5, piece


@pytest.mark.parametrize("name, hit_list", [
    ("main:main.py", ["p/pkg/run/main_py/main.py"]),
    ("origin/main:pkg/run/main.py", ["p/pkg/run/main_py/main.py"]),
    ("HEAD:pkg/run/main.py", ["p/pkg/run/main_py/main.py"]),
    ("//pkg:main.py", ["p/pkg/run/main_py/main.py"]),
    ("host:main.py", ["p/pkg/run/main_py/main.py"]),
    ("main.py:cfg.py", ["p/cfg_py/cfg.py", "p/pkg/run/main_py/main.py"]),
    ("pkg/run/main.py:42:bin/run", ["p/bin/run/run", "p/pkg/run/main_py/main.py"]),
    ("C:\\app\\pkg\\run\\main.py", ["p/pkg/run/main_py/main.py"]),
    ("`run:7`", ["p/bin/run/run"]),
    ("`run:start`", []),
    ("start:run", []),
    ("C:/x", []),
    ("12:30", []),
])
def test_a_file_written_on_either_side_of_a_colon_is_named(path_word_index, name,
                                                          hit_list):
    assert name_hit_list(scope_mod.name_list_of(name)) == (hit_list, [])


def test_a_path_behind_a_colon_keeps_its_directories(index_of):
    """Two files of one name: the directory written behind the colon tells them apart."""
    index_of(fake_defs(), {**fake_files(),
                           "p/a/x_py/x.py": {"summary": "", "defs": []},
                           "p/b/x_py/x.py": {"summary": "", "defs": []}})
    assert name_hit_list(scope_mod.name_list_of("HEAD:b/x.py")) == (["p/b/x_py/x.py"], [])
    assert name_hit_list(scope_mod.name_list_of("HEAD:x.py")) == \
        ([], ["p/a/x_py/x.py", "p/b/x_py/x.py"])


_PART_FILE_LIST = ["app/store/memory.go", "cache/memory.go", "log.1", "b/x.py", "c/x.py"]


@pytest.mark.parametrize("name, sure_list, same_name_list", [
    ("store/memory.go", ["app/store/memory.go"], []),
    ("x/store/memory.go", ["app/store/memory.go"], []),
    ("zzz/app/store/memory.go", ["app/store/memory.go"], []),
    ("memory.go", [], ["app/store/memory.go", "cache/memory.go"]),
    ("memory.go.42", [], []),
    ("memory", [], []),
    ("x.memory.go", [], []),
    ("memory.go.bak", [], []),
    ("log.1", ["log.1"], []),
    ("b/x.py", ["b/x.py"], []),
    ("a.b/x.py", [], ["b/x.py", "c/x.py"]),
    ("", [], []),
    ("///", [], []),
])
def test_a_name_is_matched_part_by_part_from_the_file_name_backwards(
        index_of, name, sure_list, same_name_list):
    index_of(fake_defs(), {repo_to_knowledge(path, "p"): {"summary": "", "defs": []}
                           for path in _PART_FILE_LIST})
    assert name_hit_list([name]) == (
        [repo_to_knowledge(path, "p") for path in sure_list],
        [repo_to_knowledge(path, "p") for path in same_name_list])


def test_a_directory_written_before_a_name_is_looked_up_part_by_part(index_of):
    """Files of one name: the directories written in front of it pick among them."""
    path_list = ["a.py", "src/a.py", "lib/src/a.py", "app/src/a.py", "foo+src/a.py", ".env",
                 "src/env", "src/.env"]
    index_of(fake_defs(), {repo_to_knowledge(path, "p"): {"summary": "", "defs": []}
                           for path in path_list})
    for text, name_list in [
            ("a.py", ["a.py"]), ("src/a.py", ["src/a.py"]),
            ("lib/src/a.py", ["lib/src/a.py"]), ("x/lib/src/a.py:3", ["lib/src/a.py"]),
            ("x/src/a.py", ["src/a.py"]), ("lib/a.py", ["a.py"]),
            ("foo+src/a.py", ["src/a.py", "foo+src/a.py"]), ("bar+src/a.py", ["src/a.py"]),
            ("bar.src/a.py", ["a.py"]), ("LIB\\SRC\\A.PY", ["lib/src/a.py"]),
            ("src/.env", ["src/.env"]), ("x/.env", [".env"]),
            ("`src/env`", ["src/env", "src/.env"])]:
        assert scope_mod.name_list_of(text) == name_list, text


def test_files_of_one_name_are_not_gone_through_each_time_the_name_is_written(index_of):
    import time
    file_count, write_count = 2000, 500
    index_of(fake_defs(), {repo_to_knowledge(f"pkg/p{number}/mod.py", "p"):
                           {"summary": "", "defs": []} for number in range(file_count)})
    for line in ("pkg/p{n}/mod.py:{n}:5: unused\n", "mod.py:{n}: unused\n",
                 "lib/q{n}/mod.py:{n}: unused\n"):
        text = "".join(line.replace("{n}", str(number)) for number in range(write_count))
        start = time.monotonic()
        hit_tuple = name_hit_list(scope_mod.name_list_of(text))
        assert time.monotonic() - start < 5, line
        assert len(hit_tuple[1]) == (write_count if line.startswith("pkg") else file_count)


def test_of_the_names_of_one_file_name_the_one_written_the_furthest_back_decides(index_of):
    index_of(fake_defs(), {repo_to_knowledge(path, "p"): {"summary": "", "defs": []}
                           for path in _PART_FILE_LIST})
    hit_tuple = ([repo_to_knowledge("app/store/memory.go", "p")], [])
    assert name_hit_list(["memory.go", "store/memory.go", "nowhere/memory.go"]) == hit_tuple
    assert name_hit_list(["store/memory.go", "memory.go"]) == hit_tuple
    assert name_hit_list(["b/x.py", "c/x.py", "x.py"]) == (
        [], [repo_to_knowledge("b/x.py", "p"), repo_to_knowledge("c/x.py", "p")])


@pytest.fixture
def non_ascii_name_index(index_of):
    """fake_index plus 2 files whose names hold characters outside ASCII."""
    return index_of(fake_defs(), {**fake_files(),
                                  "p/設定_py/設定.py": {"summary": "", "defs": []},
                                  "p/docs/設計_md/設計.md": {"summary": "", "defs": []},
                                  "p/old/設計_md/設計.md": {"summary": "", "defs": []}})


@pytest.mark.parametrize("request_text, expected", [
    ("設定.py を直して", "設定.py"),
    ("設定.pyの読み込みを直して", "設定.py"),
    ("docs/設計.md を直して", "docs/設計.md"),
])
def test_a_file_name_outside_ascii_written_bare_is_a_sure_hit(non_ascii_name_index,
                                                              monkeypatch, request_text,
                                                              expected):
    monkeypatch.setattr(config, "QUERY_EXPAND", True)
    assert search_scope(request_text, llm=lambda _: "configuration")["files"][0] == expected


def test_a_piece_of_a_name_outside_ascii_is_left_out_of_the_names(non_ascii_name_index):
    assert scope_mod.name_list_of("設定.py を直して") == ["設定.py"]
    assert scope_mod.name_list_of("新規.py を足して") == []


def test_a_piece_the_index_has_a_file_of_is_kept_among_the_names(index_of):
    index_of(fake_defs(), {**fake_files(), "p/.env/.env": {"summary": "", "defs": []}})
    assert scope_mod.name_list_of("の.env を消して") == [".env"]
    assert scope_mod.is_index_name(".ENV") and scope_mod.is_index_name("load_config")
    assert not scope_mod.is_index_name(".py")


def test_the_file_names_outside_ascii_a_text_holds_come_with_their_written_path(
        non_ascii_name_index):
    assert {"設定.py", "docs/設計.md"} <= set(
        scope_mod.text_file_name_list("docs/設計.md と 設定.py"))
    assert scope_mod.text_file_name_list("設計.md") == ["設計.md"]
    assert scope_mod.text_file_name_list("configuration") == []


def test_a_name_hit_comes_first_and_only_once(named_index):
    out = search_scope_from_query("cfg.py settings")["files"]
    assert out[0] == "cfg.py" and out.count("cfg.py") == 1


def test_a_named_file_reaches_the_scope(named_index):
    scope = search_scope_from_query("add a dependency to pyproject.toml")
    assert "pyproject.toml" in scope["files"]
    assert "pyproject.toml" not in scope["file_summaries"]


def test_with_path_weight_0_a_name_is_not_searched(named_index, monkeypatch):
    monkeypatch.setattr(config, "PATH_WEIGHT", 0)
    assert name_hit_list(["pyproject.toml"]) == ([], [])
    assert "pyproject.toml" not in \
        search_scope_from_query("add a dependency to pyproject.toml")["files"]


def test_the_name_hits_are_cut_at_file_k_too(named_index):
    assert len(search_scope_from_query("cfg.py pyproject.toml", file_k=1)["files"]) == 1


def test_a_sure_hit_is_first_even_with_file_k_1(named_index):
    assert search_scope_from_query("settings pyproject.toml", file_k=1)["files"] == \
        ["pyproject.toml"]


def test_a_sure_hit_is_first_below_a_long_definition_derived_ranking(index_of):
    """9 files each holding a definition the query hits, and a tenth only named."""
    from conftest import _def_node
    defs, files = {}, {"p/pyproject_toml/pyproject.toml": {"summary": "", "defs": []}}
    for i in range(9):
        path, key = f"p/m{i}_py/m{i}.py", f"p/m{i}_py/m{i}.py::load_settings"
        defs[key] = _def_node("load_settings", path, [1, 2], "def load_settings():",
                              "reads the settings")
        files[path] = {"summary": "settings", "defs": [key]}
    index_of(defs, files)
    out = search_scope_from_query("load settings pyproject.toml", def_k=9)["files"]
    assert out[0] == "pyproject.toml" and len(out) == 10


def test_files_of_one_name_are_same_name_hits(same_name_index):
    assert name_hit_list(["x.py"]) == \
        ([], ["p/a/x_py/x.py", "p/b/x_py/x.py", "p/c/x_py/x.py"])


def test_a_directory_written_before_the_name_keeps_that_file_alone(same_name_index):
    assert name_hit_list(["b/x.py"]) == (["p/b/x_py/x.py"], [])


def test_same_name_hits_take_half_of_file_k_and_the_ranked_one_comes_first(same_name_index):
    out = search_scope_from_query("settings x.py", file_k=4)["files"]
    assert out[:2] == ["b/x.py", "a/x.py"]
    assert set(out[2:]) == {"cfg.py", "util.py"}


def test_same_name_hits_take_no_slot_when_file_k_is_1(same_name_index):
    out = search_scope_from_query("settings x.py", file_k=1)["files"]
    assert len(out) == 1 and not out[0].endswith("x.py")


def test_rrf_merge_puts_an_entry_present_in_both_on_top():
    """Last in one ranking still rises on the combined score when it is in both."""
    assert rrf_merge(["a", "b", "c"], ["c"]) == ["c", "a", "b"]


def test_rrf_merge_keeps_the_order_when_one_side_is_empty():
    assert rrf_merge(["a", "b"], []) == ["a", "b"]


def test_a_smaller_rrf_k_weighs_rank_1_more_heavily():
    # with rrf_k=1, rank 1 (1/2) outweighs ranks 2 and 3 together (1/3 + 1/4)
    assert rrf_merge(["x"], ["y", "z"], rrf_k=1)[0] == "x"


def test_rrf_merge_takes_a_separate_constant_per_ranking():
    """Weighting only the second (the file index) must lift its top entry."""
    assert rrf_merge(["a"], ["b"], rrf_k=200, rrf_k_b=1) == ["b", "a"]
    assert rrf_merge(["a"], ["b"], rrf_k=1, rrf_k_b=200) == ["a", "b"]


def test_rrf_merge_defaults_to_rrf_k_and_rrf_k_file(monkeypatch):
    """Omitting the arguments must read both config values."""
    monkeypatch.setattr(config, "RRF_K", 200)
    monkeypatch.setattr(config, "RRF_K_FILE", 1)
    assert rrf_merge(["a"], ["b"]) == ["b", "a"]


def test_spread_is_how_many_files_the_entry_points_scattered_over(fake_index):
    scope = search_scope_from_query("load_config")
    assert scope["spread"] == 2      # cfg.py and util.py


def test_spread_is_between_1_and_def_k(fake_index):
    scope = search_scope_from_query("configuration")
    assert 1 <= scope["spread"] <= config.DEF_K


@pytest.mark.parametrize("key", ["files", "defs", "file_summaries", "callers",
                                 "confidence", "spread", "trace"])
def test_the_scope_returns_its_7_keys(fake_index, key):
    assert key in search_scope_from_query("configuration file")


def test_the_trace_only_names_a_request_when_one_was_given(fake_index):
    """search_again() comes straight into search_scope_from_query() with terms of its
    own, and terms are not a request, so the trace of a search_again call names no
    request.
    """
    assert search_scope_from_query("configuration")["trace"]["request"] == ""
    scope = search_scope("configuration file")
    assert scope["trace"]["request"] == "configuration file"
    assert scope["trace"]["query"] == "configuration file"


def test_search_scope_from_query_takes_the_counts_as_arguments(fake_index):
    assert len(search_scope_from_query("configuration", def_k=1)["defs"]) == 1
    assert len(search_scope_from_query("configuration", file_k=1)["files"]) == 1


# == Reading the index in ====================================================

def test_an_index_of_another_project_is_refused(tmp_path):
    path = str(tmp_path / "index.sqlite")
    write_index(path, fake_defs(), fake_files()).close()
    with pytest.raises(ValueError, match="different project"):
        scope_mod.load_index(path, "other")


def test_an_index_built_from_an_earlier_analysis_only_warns(tmp_path, caplog):
    path = str(tmp_path / "index.sqlite")
    write_index(path, fake_defs(), fake_files()).close()
    scope_mod.load_index(path, "p", source_stamp="2:2026-02-02T00:00:00+00:00")
    assert "may be stale" in caplog.text
    scope_mod.INDEX.close()


def test_the_neighbour_table_holds_both_ends_of_every_call(tmp_path):
    path = str(tmp_path / "index.sqlite")
    defs = fake_defs()
    write_index(path, defs, fake_files()).close()
    scope_mod.load_index(path, "p")
    assert scope_mod.NEIGHBOR_DICT == {}
    scope_mod.INDEX.close()


def test_a_language_made_the_primary_one_keeps_the_index_s_other_languages(tmp_path,
                                                                           monkeypatch):
    for name in ("PRIMARY_EXT", "TEST_COMMAND", "TEST_PATH", "TEST_TARGET_STYLE",
                 "RUN_COMMAND", "CHECK_COMMAND", "OTHER_CHECK_LIST", "OTHER_TEST_LIST"):
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    path = str(tmp_path / "index.sqlite")
    write_index(path, fake_defs(), fake_files()).close()
    scope_mod.load_index(path, "p")
    assert config.PRIMARY_EXT == "py"
    scope_mod.set_primary("rs")
    assert config.PRIMARY_EXT == "rs" and config.TEST_COMMAND[0] == "cargo"
    assert [ext for ext, _ in config.OTHER_TEST_LIST] == ["py"]
    scope_mod.INDEX.close()


# == Step 3: following the calls edges =======================================
_A = "p/cfg_py/cfg.py::load_config"
_B = "p/cfg_py/cfg.py::save_config"
_C = "p/util_py/util.py::load_config"


@pytest.fixture
def chain_index(index_of):
    """fake_defs with the calls A -> B -> C."""
    return index_of(fake_defs(), fake_files(), calls=[(_A, _B), (_B, _C)])


def test_a_depth_of_0_adds_nothing(chain_index):
    assert scope_mod.expand([{"key": _A, "score": 1.0}], depth=0) == ({_A: 1.0}, {})


def test_each_hop_multiplies_the_score_by_the_decay(chain_index, monkeypatch):
    monkeypatch.setattr(config, "EXPAND_DECAY", 0.5)
    out, _ = scope_mod.expand([{"key": _A, "score": 1.0}], depth=2)
    assert out == {_A: 1.0, _B: 0.5, _C: 0.25}


def test_a_test_that_calls_a_match_is_neither_added_nor_followed(index_of, monkeypatch):
    """A test function is called by nothing, so it would take the decayed score whole
    and bring its test file into files."""
    from conftest import _def_node
    test_key = "p/tests/test_cfg_py/test_cfg.py::test_load"
    helper_key = "p/tests/test_cfg_py/test_cfg.py::make_file"
    defs = dict(fake_defs(), **{
        test_key: _def_node("test_load", "p/tests/test_cfg_py/test_cfg.py", [1, 4],
                            "def test_load():", "checks loading"),
        helper_key: _def_node("make_file", "p/tests/test_cfg_py/test_cfg.py", [6, 8],
                              "def make_file():", "writes a file")})
    index_of(defs, fake_files(), calls=[(test_key, _A), (test_key, helper_key), (_A, _B)])
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    out, from_dict = scope_mod.expand([{"key": _A, "score": 1.0}], depth=2)
    assert set(out) == {_A, _B} and from_dict == {_B: _A}


def test_a_neighbour_is_discounted_by_how_many_call_it(index_of, monkeypatch):
    """C is called by A, B and D, so one hop hands it the decayed score over 3. An
    entry point is never discounted, whatever its in-degree."""
    from conftest import _def_node
    _D = "p/util_py/util.py::run"
    defs = dict(fake_defs(), **{_D: _def_node("run", "p/util_py/util.py", [7, 9],
                                              "def run():", "runs")})
    index_of(defs, fake_files(), calls=[(_A, _C), (_B, _C), (_D, _C)])
    monkeypatch.setattr(config, "EXPAND_DECAY", 0.5)
    out, from_dict = scope_mod.expand([{"key": _A, "score": 1.0}], depth=1)
    assert out[_C] == pytest.approx(0.5 / 3) and from_dict == {_C: _A}
    out, _ = scope_mod.expand([{"key": _C, "score": 1.0}], depth=1)
    assert out[_C] == 1.0 and out[_A] == 0.5


def test_a_lower_score_by_another_route_does_not_replace_a_higher_one(chain_index,
                                                                      monkeypatch):
    monkeypatch.setattr(config, "EXPAND_DECAY", 0.5)
    out, _ = scope_mod.expand([{"key": _A, "score": 1.0},
                               {"key": _B, "score": 0.8}], depth=1)
    assert out[_B] == 0.8


def test_an_entry_point_reached_from_another_keeps_its_score_and_is_not_added(
        chain_index, monkeypatch):
    """A (1.0) calls B (0.1): the hop would hand B 0.5, more than its own score."""
    monkeypatch.setattr(config, "EXPAND_DECAY", 0.5)
    out, from_dict = scope_mod.expand([{"key": _A, "score": 1.0},
                                       {"key": _B, "score": 0.1}], depth=1)
    assert out[_B] == 0.1
    assert from_dict == {_C: _B}


def test_expanded_holds_no_definition_that_defs_holds(chain_index, monkeypatch):
    monkeypatch.setattr(scope_mod, "search_def_list",
                        lambda query, k: [{"key": _A, "score": 1.0, "raw": 1.0, "hit_terms": 1},
                                          {"key": _B, "score": 0.1, "raw": 0.1, "hit_terms": 1}])
    scope = search_scope_from_query("load_config")
    def_key_set = {d["key"] for d in scope["defs"]}
    assert def_key_set == {"cfg.py::load_config", "cfg.py::save_config"}
    assert [d["key"] for d in scope["expanded"]] == ["util.py::load_config"]


# == The entrance of search_scope ============================================
def test_an_empty_request_is_refused(fake_index):
    with pytest.raises(ValueError, match="empty request"):
        search_scope("   ")


def test_a_request_that_needs_expanding_builds_the_model_itself(fake_index, monkeypatch):
    built = []

    def make(model, max_token, setting_name, think_level=""):
        built.append((model, max_token, setting_name, think_level))
        return lambda prompt: "configuration file"

    monkeypatch.setattr(scope_mod, "make_llm", make)
    monkeypatch.setattr(config, "QUERY_EXPAND", True)
    scope = search_scope("設定の読み込み")
    # Thinking is off for the rewrite, so the whole output limit is the reply's
    assert built == [(config.QUERY_EXPAND_MODEL, config.QUERY_EXPAND_MAX_TOKEN,
                      "QUERY_EXPAND_MODEL", "off")]
    assert scope["trace"]["query"] == "configuration file"
    assert scope["trace"]["request"] == "設定の読み込み"


def test_an_ascii_request_builds_no_model(fake_index, monkeypatch):
    monkeypatch.setattr(scope_mod, "make_llm",
                        lambda *a, **k: pytest.fail("a model was built"))
    assert search_scope("configuration file")["files"]


def test_a_query_that_matches_nothing_is_said_in_the_log(fake_index, caplog):
    with caplog.at_level(logging.INFO):
        scope = search_scope("zzzz qqqq")
    assert scope["files"] == [] and "no match" in caplog.text


def test_entry_points_that_grow_nothing_are_said_in_the_log(fake_index, caplog):
    with caplog.at_level(logging.DEBUG, logger="codetwine_loom.search.scope"):
        search_scope("configuration file")
    assert "expanded nothing" in caplog.text


# == the note a scope that found nothing carries ============================

def test_a_scope_that_found_nothing_says_so(fake_index):
    scope = search_scope_from_query("nothingmatchesthisword")
    assert scope["files"] == [] and scope["defs"] == []
    assert scope["note"].startswith("The search matched nothing;")


@pytest.mark.parametrize("trace_dict, note", [
    ({"empty": True}, "The search matched nothing."),
    ({"empty": True, "expand_failed": True},
     "The search matched nothing (the query expansion failed)."),
    ({"expand_failed": True},
     "The query expansion failed, so the search ran on the request as it was written."),
])
def test_a_stage_without_search_again_is_handed_a_note_that_does_not_name_it(
        trace_dict, note):
    scope = {"trace": trace_dict, "note": scope_mod.scope_note({"trace": trace_dict})}
    assert "search_again" in scope["note"]
    with_tool = scope_mod.stage_variable_dict(scope, {"search_again": print})["scope"]
    assert with_tool["note"] == scope["note"]
    without = scope_mod.stage_variable_dict(scope, {"read_definition": print})["scope"]
    assert without["note"] == note


def test_a_scope_that_found_something_carries_no_note(fake_index):
    assert search_scope_from_query("configuration file")["note"] == ""


def test_a_failed_expansion_is_named_in_the_note(fake_index, monkeypatch):
    """The stages read the note, so a run that starts from nothing is told why."""
    monkeypatch.setattr(config, "QUERY_EXPAND", True)
    scope = search_scope("設定を直して", llm=lambda _: "")
    assert scope["trace"]["expand_failed"] is True
    assert "the query expansion failed" in scope["note"]


def test_a_failed_expansion_is_named_in_the_note_when_something_was_found(fake_index,
                                                                          monkeypatch):
    """A name written in the request can still hit; the stages are told the rest of
    the request was not searched with."""
    monkeypatch.setattr(config, "QUERY_EXPAND", True)

    def llm(_prompt):
        raise RuntimeError("down")

    scope = search_scope("load_config の再試行を直して", llm=llm)
    assert scope["files"] and scope["trace"]["expand_failed"] is True
    assert scope["note"] == ("The query expansion failed, so the search ran on the "
                             "request as it was written; search_again with English "
                             "terms if these candidates do not fit.")


# == step 3: what the expansion reached and where from ======================

def test_the_expansion_says_which_match_it_was_reached_from(chain_index):
    scope = search_scope_from_query("load_config")
    assert scope["defs"]
    for entry_dict in scope["expanded"]:
        assert set(entry_dict) == {"key", "lines", "signature", "from"}
        assert entry_dict["from"] in [d["key"] for d in scope["defs"]]


def test_expanded_and_each_callers_list_are_capped_at_def_k(index_of, monkeypatch):
    """C is reached from, and called by, A, B and D; def_k bounds both lists."""
    from conftest import _def_node
    _D = "p/util_py/util.py::run"
    defs = dict(fake_defs(), **{_D: _def_node("run", "p/util_py/util.py", [7, 9],
                                              "def run():", "runs")})
    index_of(defs, fake_files(), calls=[(_A, _C), (_B, _C), (_D, _C)])
    monkeypatch.setattr(config, "EXPAND_DECAY", 0.5)
    wide = search_scope_from_query("load_config", def_k=3)
    assert len(wide["expanded"]) + max(len(v) for v in wide["callers"].values()) >= 3
    narrow = search_scope_from_query("load_config", def_k=1)
    assert len(narrow["expanded"]) <= 1
    assert all(len(v) <= 1 for v in narrow["callers"].values())


def test_expanded_keeps_the_highest_scores_when_cut(index_of, monkeypatch):
    """B is reached from the entry with the higher score, so it stays when 1 is kept."""
    from conftest import _def_node
    _D = "p/util_py/util.py::run"
    defs = dict(fake_defs(), **{_D: _def_node("run", "p/util_py/util.py", [7, 9],
                                              "def run():", "runs")})
    index_of(defs, fake_files(), calls=[(_A, _B), (_C, _D)])
    monkeypatch.setattr(scope_mod, "search_def_list",
                        lambda query, k: [{"key": _A, "score": 1.0, "raw": 1.0, "hit_terms": 1},
                                          {"key": _C, "score": 0.2, "raw": 0.2, "hit_terms": 1}])
    scope = search_scope_from_query("load_config", def_k=1)
    assert [d["key"] for d in scope["expanded"]] == ["cfg.py::save_config"]


def test_a_test_that_matched_is_not_followed(index_of, monkeypatch):
    from conftest import _def_node
    test_key = "p/tests/test_cfg_py/test_cfg.py::test_load"
    helper_key = "p/util_py/util.py::only_the_test_calls_it"
    defs = dict(fake_defs(), **{
        test_key: _def_node("test_load", "p/tests/test_cfg_py/test_cfg.py", [1, 4],
                            "def test_load():", "checks loading"),
        helper_key: _def_node("only_the_test_calls_it", "p/util_py/util.py", [20, 22],
                              "def only_the_test_calls_it():", "helps")})
    index_of(defs, fake_files(), calls=[(test_key, helper_key), (_A, _B)])
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    out, from_dict = scope_mod.expand([{"key": _A, "score": 1.0},
                                       {"key": test_key, "score": 0.9}], depth=1)
    assert set(out) == {_A, _B, test_key} and from_dict == {_B: _A}


# == step 2: the definitions to change come before the tests =================

def test_a_test_definition_goes_behind_the_definitions_that_are_not(index_of,
                                                                    monkeypatch):
    """The k the caller asked for is spent on the places to change."""
    from conftest import _def_node
    defs = {
        "p/tests/test_cfg_py/test_cfg.py::test_load_config": _def_node(
            "test_load_config", "p/tests/test_cfg_py/test_cfg.py", [1, 5],
            "def test_load_config():", "loads the configuration file"),
        "p/cfg_py/cfg.py::load_config": _def_node(
            "load_config", "p/cfg_py/cfg.py", [1, 5],
            "def load_config():", "loads the configuration file"),
    }
    index_of(defs, {"p/tests/test_cfg_py/test_cfg.py": {"summary": "", "defs": list(defs)[:1]},
                    "p/cfg_py/cfg.py": {"summary": "", "defs": list(defs)[1:]}})
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    key_list = [d["key"] for d in search_scope_from_query("configuration")["defs"]]
    assert key_list[0] == "cfg.py::load_config"


def test_a_test_file_goes_behind_the_files_that_are_not(monkeypatch):
    """The file of a lower definition comes before the test file of a higher one."""
    monkeypatch.setattr(scope_mod, "PROJECT_NAME", "p")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(scope_mod, "search_file_list", lambda query: [])
    monkeypatch.setattr(scope_mod, "name_hit_list", lambda name_list: ([], []))
    file_by_def_dict = {"t::a": "p/tests/test_cfg_py/test_cfg.py",
                        "c::b": "p/cfg_py/cfg.py", "u::c": "p/util_py/util.py"}
    assert scope_mod._file_rank_list({"t::a": 0.9, "c::b": 0.5, "u::c": 0.7},
                                     file_by_def_dict, "q", [], 5) == \
        ["p/util_py/util.py", "p/cfg_py/cfg.py", "p/tests/test_cfg_py/test_cfg.py"]
