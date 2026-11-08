"""Search step 1: the request rewritten into English search terms.

The model is a stand-in returning a fixed line, so what is checked is the three gates
in front of it, the word cap behind it, and what a failed call leaves the search with.
"""

import itertools
import logging
import time

import pytest

from codetwine_loom import config
from codetwine_loom.search import query


def _llm_of(reply: str):
    """Return a stand-in model that returns reply and records the prompts."""
    prompt_list = []

    def call(prompt: str) -> str:
        prompt_list.append(prompt)
        return reply

    call.prompt_list = prompt_list
    return call


def test_the_expansion_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(config, "QUERY_EXPAND", False)
    llm = _llm_of("never")
    assert query.expand_query("設定を直して", llm) == ("設定を直して", True)
    assert llm.prompt_list == []


def test_an_ascii_request_is_searched_as_it_is():
    llm = _llm_of("never")
    assert query.expand_query("fix the config loader", llm) == \
        ("fix the config loader", True)
    assert llm.prompt_list == []


def test_without_a_model_the_request_is_searched_as_it_is_and_the_log_says_so(caplog):
    assert query.expand_query("設定を直して", None) == ("設定を直して", True)
    assert "no model is available for the expansion" in caplog.text


def test_the_request_and_the_word_range_go_into_the_prompt(monkeypatch):
    monkeypatch.setattr(config, "QUERY_EXPAND_WORD_RANGE", "5-9")
    llm = _llm_of("config load")
    assert query.expand_query("設定を直して", llm) == ("config load", True)
    assert "設定を直して" in llm.prompt_list[0] and "5-9 words" in llm.prompt_list[0]


def test_more_words_than_the_range_allows_are_cut(monkeypatch, caplog):
    monkeypatch.setattr(config, "QUERY_EXPAND_WORD_RANGE", "2-3")
    llm = _llm_of("one two three four five")
    with caplog.at_level(logging.INFO):
        assert query.expand_query("設定を直して", llm) == ("one two three", True)
    assert "searching with the first 3" in caplog.text


def test_a_range_with_no_number_turns_the_cut_off(monkeypatch):
    monkeypatch.setattr(config, "QUERY_EXPAND_WORD_RANGE", "a few")
    assert query.word_cap() == 0
    llm = _llm_of("one two three four five")
    assert query.expand_query("設定を直して", llm) == \
        ("one two three four five", True)


def test_the_reply_is_split_on_any_whitespace():
    llm = _llm_of("  config\nload \t reader ")
    assert query.expand_query("設定を直して", llm) == ("config load reader", True)


def test_a_call_that_fails_leaves_the_request_as_it_is(caplog):
    def fail(prompt: str) -> str:
        raise RuntimeError("no key")

    assert query.expand_query("設定を直して", fail) == ("設定を直して", False)
    assert "query expansion failed" in caplog.text and "RuntimeError: no key" in caplog.text


def test_a_reply_with_no_words_is_asked_for_once_more(caplog):
    """make_llm turns a reply with no content in it into "", which splits to nothing."""
    reply_list = ["", "config load"]
    llm = lambda prompt: reply_list.pop(0)                          # noqa: E731
    assert query.expand_query("設定を直して", llm) == ("config load", True)
    assert "came back with no English word in it" in caplog.text
    assert reply_list == []


@pytest.mark.parametrize("reply", ["設定 読み込み 修正", "... - - -"])
def test_a_reply_with_no_english_word_is_asked_for_once_more(reply):
    """A reply the English index cannot match is no better than an empty one."""
    reply_list = [reply, "config load"]
    llm = lambda prompt: reply_list.pop(0)                          # noqa: E731
    assert query.expand_query("設定を直して", llm) == ("config load", True)
    assert reply_list == []


def test_a_reply_with_no_english_word_twice_leaves_the_request_as_it_is():
    assert query.expand_query("設定を直して", lambda _: "設定 読み込み") == \
        ("設定を直して", False)


@pytest.mark.parametrize("text, expected", [
    ("load config", True), ("_x", True), ("設定 読み込み", False), ("... - 12", False),
    ("", False)])
def test_an_english_word_is_an_ascii_letter_or_an_underscore(text, expected):
    assert query.has_english_word(text) is expected


def test_a_reply_with_no_words_twice_leaves_the_request_as_it_is(caplog):
    assert query.expand_query("設定を直して", lambda _: "  ") == ("設定を直して", False)
    assert "searching with the request as it is" in caplog.text


# == The names written in the request ======================================

def test_a_name_written_in_the_request_is_kept_after_the_rewrite():
    llm = _llm_of("dependency package")
    assert query.expand_query("`pyproject.toml` に依存を足して", llm) == \
        ("dependency package pyproject.toml", True)


def test_a_name_the_rewrite_already_holds_is_not_added_twice():
    llm = _llm_of("Pyproject.toml dependency")
    assert query.expand_query("`pyproject.toml` に依存を足して", llm) == \
        ("Pyproject.toml dependency", True)


def test_a_url_in_the_request_is_not_a_name():
    llm = _llm_of("config")
    assert query.expand_query("https://example.com/doc.html を見て直して", llm) == \
        ("config", True)


def test_names_are_kept_past_the_word_cap(monkeypatch):
    monkeypatch.setattr(config, "QUERY_EXPAND_WORD_RANGE", "2-3")
    llm = _llm_of("one two three four")
    assert query.expand_query("`load_config` を直して", llm) == \
        ("one two three load_config", True)


# The search terms a request holds, by how it is written: a line is (the request, the
# terms taken from it)
_TERM_TABLE = [
    # a path, a file name and an identifier, bare
    ("store/memory.go を直す", ["store/memory.go"]),
    ("store/memory.py を直す", ["store/memory.py"]),
    ("pkg/run/main.py を直す", ["pkg/run/main.py"]),
    ("tree-sitter-go を足す", ["tree-sitter-go"]),
    ("resolve_module_to_project_path を呼ぶ", ["resolve_module_to_project_path"]),
    ("loadConfig() を直す", ["loadConfig"]),
    ("settings.py. を見る", ["settings.py"]),
    ("a.py と a.py の両方", ["a.py"]),
    ("Makefile を直す", []),
    ("2026-09-22 に 1e-5 を", ["1e-5"]),
    ("Python 3.10 と v1.2 で動かす", []),
    ("src/ と a/b/ を見る", ["a/b"]),
    ("loadConfig(path) と foo(a/b)", ["loadConfig", "a/b"]),
    ("src/routes/+page.svelteを直す", ["src/routes", "page.svelte"]),
    ("import os.path; from a.b import c", ["os.path", "a.b"]),
    # the same in backquotes, where a word needs no mark of a name and may be written
    # outside ASCII
    ("`Makefile` を直す", ["Makefile"]),
    ("`.go` ファイルを対象にする", [".go"]),
    ("`2to3.py` と 2to3.py", ["2to3.py"]),
    ("`loadConfig()` を呼ぶ", ["loadConfig"]),
    ("`顧客番号` の桁を 12 にして", ["顧客番号"]),
    ("`CUST-ACCT-ID` を直す", ["CUST-ACCT-ID"]),
    ("`docs/設計.md` を直す", ["docs/設計.md"]),
    ("`pyproject.toml と `LICENSE`", ["pyproject.toml", "LICENSE"]),
    ("`two words` を見る", []),
    ("`git log --oneline` then `rm -rf build/`", []),
    ("`make test`が落ちるので`Makefile`を直して", ["Makefile"]),
    ("`go test ./...`で`retry`が落ちる", ["retry"]),
    ("`a b`,`Makefile` と see `make test`(`LICENSE`)", ["Makefile", "LICENSE"]),
    ("```make test``` then `Makefile`", ["Makefile"]),
    ("`a.py`\n`x y``cfg.py`\n` stray `LICENSE`", ["a.py", "cfg.py", "LICENSE"]),
    ("Fix ``Makefile`` and ``LICENSE``.", ["Makefile", "LICENSE"]),
    ("``retry`` が 3 回で止まらない", ["retry"]),
    ("````Makefile```` と ``` `LICENSE` ```", ["Makefile", "LICENSE"]),
    ("「``顧客番号``」の桁を直す", ["顧客番号"]),
    ("`make test` と ``Makefile``", ["Makefile"]),
    ("``make test``が落ちるので``Makefile``を直す", ["Makefile"]),
    ("``Makefile`` と ``LICENSE を直して", ["Makefile"]),
    ("``retry`` が止まらない。``Makefile`` も `` で囲む", ["retry", "Makefile"]),
    ("```sh のあとの ` と ``Makefile`` を直す", ["Makefile"]),
    ("`!=` と `->` を見る", []),
    ("`foo(bar)[0]` と `Vec<String>`", ["foo", "Vec"]),
    ("`x=y/z` を見る", ["x", "y/z"]),
    ("`--no-overview` を足す", ["no-overview"]),
    ("`--config=src/app.yaml` と `--exclude=build/`", ["config", "src/app.yaml",
                                                    "exclude"]),
    # the marks in front of a name are passed over
    ("`/store/memory.py` を直す", ["store/memory.py"]),
    ("`@app.route` と `$HOME/x.toml`", ["app.route", "HOME/x.toml"]),
    ("`\"config.py\"` と `<stdio.h>` と `~/.bashrc`", ["config.py", "stdio.h", ".bashrc"]),
    ("`<sys/stat.h>` と `\"a/b.py\"`", ["sys/stat.h", "a/b.py"]),
    ("//pkg:mod.py と ./src/a.py", ["pkg:mod.py", "./src/a.py"]),
    # a mark a directory or a file is named with ends a term
    ("`app/[slug]/page.tsx` を直す", ["app", "page.tsx"]),
    ("app/[slug]/page.tsx を直す", ["page.tsx"]),
    ("`src/routes/+page.svelte` を直す", ["src/routes", "page.svelte"]),
    ("`app/(shop)/{id}/@modal/~x/$y/page.tsx`", ["app", "y/page.tsx"]),
    ("app/(shop)/{id}/@modal/~x/$y/page.tsx を", ["y/page.tsx"]),
    ("`C:\\src\\x.py` を直す", ["C:\\src\\x.py"]),
    ("C:\\src\\x.py を直す", ["C:\\src\\x.py"]),
    ("assets/icon@2x.png と src/<name>.py", ["assets/icon", "2x.png", ".py"]),
    ("docs/report(1).pdf を見る", ["docs/report", ".pdf"]),
    # a bracket ends a term like any other mark
    ("`[slug]/page.tsx` と `(shop)/page.tsx`", ["slug", "page.tsx", "shop"]),
    ("[slug]/page.tsx と (shop)/page.tsx", ["page.tsx"]),
    ("`<project>/src/main.py` を直す", ["project", "src/main.py"]),
    ("`${HOME}/cfg.py` と `$(pwd)/util.py`", ["HOME", "cfg.py", "pwd", "util.py"]),
    ("`{name}/cfg.py` を直す", ["name", "cfg.py"]),
    # what is written around a path is no part of it
    ("fix (app/[slug]/page.tsx).", ["page.tsx"]),
    ("「cfg.py」 と （util.py） と \"a/b.py\" と 'c/d.py'",
     ["cfg.py", "util.py", "a/b.py", "c/d.py"]),
    ("see [the doc](docs/x.md) and [src/a.py](docs/b.md)",
     ["docs/x.md", "src/a.py", "docs/b.md"]),
    ("[text](path/to/file.py) を見る", ["path/to/file.py"]),
    ("`[text](path/to/file.py)` を見る", ["text", "path/to/file.py"]),
    ("`open(src/main.py)` と open(src/main.py)", ["open", "src/main.py"]),
    # what is written behind a name is a term of its own when it has the mark of a name
    ("cfg.py#load_config と MemoryStore#loadConfig", ["cfg.py", "load_config", "MemoryStore",
                                                    "loadConfig"]),
    ("UserStore#load_config と UserStore#save_config", ["UserStore", "load_config",
                                                      "save_config"]),
    ("isReady?loadConfig():saveConfig()", ["isReady", "loadConfig", "saveConfig"]),
    ("`src/main.py#L10` を直す", ["src/main.py"]),
    ("src/main.py#L10 を直す", ["src/main.py"]),
    ("`src/main.py#L10-L20` と `src/b.py#main`", ["src/main.py", "L10-L20", "src/b.py"]),
    ("`main.py#L10` を直す", ["main.py"]),
    ("`app/[slug]/page.tsx#L3` を直す", ["app", "page.tsx"]),
    ("`src/main.py?plain=1` を直す", ["src/main.py"]),
    ("`src/main.py(new)` と src/b.py(new)", ["src/main.py", "src/b.py"]),
    ("`src/main.py（新規）` を足す", ["src/main.py"]),
    ("`src/Makefile(new)` を足す", ["src/Makefile"]),
    ("`src/main.py@abc123` を見る", ["src/main.py"]),
    ("`src/main.py*` と `src/b.py$`", ["src/main.py", "src/b.py"]),
    ("`src/main.py:main()` を直す", ["src/main.py:main"]),
    ("src/x/y.py:42:config.load() で落ちる", ["src/x/y.py:42:config.load"]),
    ("store/memory.py::load_config#function#4 を直す", ["store/memory.py::load_config"]),
    # a comma or another separator starts the next name
    ("`src/main.py,cfg.py` を直す", ["src/main.py", "cfg.py"]),
    ("`src/main.py、cfg.py` と `a/b.py|c.py` と `d/e.py;f.py`",
     ["src/main.py", "cfg.py", "a/b.py", "c.py", "d/e.py", "f.py"]),
    ("app/(shop)/page.tsx, then src/+page.js,src/b.py", ["page.tsx", "page.js", "src/b.py"]),
    ("src/a.py#L1,src/b.py を直す", ["src/a.py", "src/b.py"]),
    ("テーブル|src/main.py|cfg.py|", ["src/main.py", "cfg.py"]),
    # a colon stays in the term
    ("memory.py:42 と Store.cs:line 42", ["memory.py:42", "Store.cs:line"]),
    ("store/memory.py:42:def load を直す", ["store/memory.py:42:def"]),
    ("`store/memory.py::load_config` を直す", ["store/memory.py::load_config"]),
    ("`Makefile:12` と Makefile:12", ["Makefile:12"]),
    ("git show main:a.py と origin/main:a.py", ["main:a.py", "origin/main:a.py"]),
    ("host:a.py と HEAD:src/x.py と a.py:b.py", ["host:a.py", "HEAD:src/x.py",
                                                "a.py:b.py"]),
    ("store/memory.py:42:a/b と user@host:path/x.py",
     ["store/memory.py:42:a/b", "host:path/x.py"]),
    ("error: foo.py not found と note:main.py", ["foo.py", "note:main.py"]),
    ("`std::vector` と std::vector と `key:value`", ["std::vector", "key:value"]),
    ("a:b:c.py と handler.py:request.json", ["a:b:c.py", "handler.py:request.json"]),
    ("C: と 12:30 を見る", []),
    # a URL is no name, and neither is what is written behind it
    ("https://x.y/doc.html を読む", []),
    ("`https://x.y/z` を見る", []),
    ("https://x.y/doc.html?a=b_c#frag-ment を読む", []),
    ("(https://x.y/a/main.py), <https://x.y/cfg.py>. と http://host:8080/x.py", []),
    ("[src/a.py](https://x.y/z) を見る", ["src/a.py"]),
    ("https://x.y/z、src/a.py を見る", ["src/a.py"]),
]


@pytest.mark.parametrize("request_text, name_list", _TERM_TABLE)
def test_literal_term_list(request_text, name_list):
    assert query.literal_term_list(request_text) == name_list


@pytest.mark.parametrize("request_text, span_list", [
    ("`a.py` と `b c` と `d.py`", [(0, 6), (17, 23)]),
    ("`make test`が落ちるので`Makefile`を直す", [(17, 27)]),
    ("`a b`,`c`(`d e`)`f`", [(6, 9), (16, 19)]),
    ("`` と ```x y``` と `z`", [(17, 20)]),
    ("`pyproject.toml と `LICENSE`", [(18, 27)]),
    ("`a b`c`d", [(4, 7)]),
    ("`a b`c`\n`d`\r\n`e f``g`", [(4, 7), (8, 11), (18, 21)]),
    ("no backquote", []),
    ("Fix ``Makefile`` and ``LICENSE``.", [(5, 15), (22, 31)]),
    ("````Makefile````", [(3, 13)]),
    ("`` `Makefile` `` と ``a b``", [(3, 13)]),
    ("``` `Makefile` ```", [(4, 14)]),
    ("`a``b`", [(0, 3), (3, 6)]),
    ("``make test``が落ちるので``Makefile``を直す", [(20, 30)]),
    ("`make test` と ``Makefile``", [(15, 25)]),
    ("``Makefile`` と ` を使う", [(1, 11)]),
    ("`` と `make test`が`Makefile`", [(17, 27)]),
    ("``a`b``", []),
    # A run nothing closes leaves the words the other runs pair to
    ("``Makefile`` と ``LICENSE を直して", [(1, 11)]),
    ("``Makefile`` を直す。`` は空のまま", [(1, 11)]),
    ("```` で始まるブロックの ``Makefile`` を直す", [(16, 26)]),
    ("```sh のあとの ` と ``Makefile`` を直す", [(16, 26)]),
    ("``retry`` が止まらない。``Makefile`` も `` で囲む", [(1, 8), (18, 28)]),
    ("```` と ``b.py`` と `c d` と `e.py` と ``f", [(8, 14), (26, 32)]),
    # What is read the other way keeps its place
    ("``Makefile`` と ``LICENSE` を直して", [(1, 11), (16, 25)]),
    ("`a b`c``d.py`` と `e", [(4, 7), (7, 13)]),
    ("it`s `Makefile` isn`t", []),
])
def test_the_runs_of_backquotes_of_a_line_are_paired_as_marks_from_its_start(
        request_text, span_list):
    assert query.quote_span_list(request_text) == span_list


def test_the_words_in_backquotes_of_a_line_are_in_order_and_share_no_place():
    """Every line of two to four runs of one to three backquotes, with a word, a space
    or a phrase between them."""
    for run_count in (2, 3, 4):
        for length_tuple in itertools.product((1, 2, 3), repeat=run_count):
            for gap_tuple in itertools.product(("w", " ", " x y "), repeat=run_count - 1):
                line = "".join("`" * length + gap
                               for length, gap in zip(length_tuple, (*gap_tuple, "")))
                span_list = query.quote_span_list(line)
                assert span_list == sorted(span_list), line
                assert all(first[1] <= second[0]
                           for first, second in zip(span_list, span_list[1:])), line
                assert all(line[start] == line[end - 1] == "`" and "`" not in
                           line[start + 1:end - 1] for start, end in span_list), line


@pytest.mark.parametrize("request_text", [
    "` ``` ` の `Makefile` を `` ` `` で囲む",
    "`` ` `` と ` ``` ` のあと ``Makefile`` を直す",
    "` ``` ` のあと ``Makefile`` を直す",
    "`` ` `` を `Makefile` に直す ```python"])
def test_a_name_beside_code_spans_that_hold_backquotes_is_read(request_text):
    """A run pairs with the next run of its own length where the line leaves a run
    open."""
    assert query.literal_term_list(request_text) == ["Makefile"]


def test_the_runs_of_the_same_length_pair_over_the_runs_between_them():
    line = "` ``` ` と `a` と ``b`` と ```c d``` と ````"
    run_list = [(run.start(), len(run.group(0))) for run in query._QUOTE_RUN.finditer(line)]
    assert [line[start + 1:end - 1]
            for start, end in query._same_run_pair_list(line, run_list)] == ["a", "b"]
    assert query._same_run_pair_list("`a``b`", [(0, 1), (2, 2), (5, 1)]) == []


def _pair_list_level_by_level(line: str, run_list: list) -> list:
    """Pair the runs of a line one level at a time: the runs of the line, then the
    runs inside each pair that holds no word, each list read from its start."""
    span_list = []
    wait_list = [run_list]
    while wait_list:
        part_list = wait_list.pop()
        word_list, inside_list, inner_list = [], [], []
        open_length, open_at = 0, 0
        for index, (start, length) in enumerate(part_list):
            if open_length and length < open_length:
                inner_list.append((start, length))
                continue
            rest_length = length - open_length
            if open_length and "`" not in line[open_at + 1:start] \
                    and line[open_at + 1:start] and not any(
                        character.isspace() for character in line[open_at + 1:start]):
                word_list.append((open_at, start + 1))
            elif open_length and inner_list:
                inside_list.append(inner_list)
            if part_list is run_list and rest_length > max(
                    (length for _, length in run_list[index + 1:]), default=0):
                rest_length = 0
            open_length, open_at, inner_list = rest_length, start + length - 1, []
        if not open_length:
            span_list += word_list
            wait_list += inside_list
    return sorted(span_list)


def test_runs_nested_in_runs_are_paired_as_they_are_level_by_level():
    """Every line of two to six runs of one to four backquotes, a letter, a space or
    nothing between them."""
    for run_count in (2, 3, 4, 5, 6):
        for length_tuple in itertools.product((1, 2, 3, 4), repeat=run_count):
            for gap_tuple in (("w",) * run_count, (" ",) * run_count, ("",) * run_count,
                              tuple("w " [index % 2] for index in range(run_count))):
                line = "".join("`" * length + gap
                               for length, gap in zip(length_tuple, gap_tuple))
                run_list = [(run.start(), len(run.group(0)))
                            for run in query._QUOTE_RUN.finditer(line)]
                assert query._run_pair_list(line, run_list)[0] \
                    == _pair_list_level_by_level(line, run_list), line


def test_a_line_of_runs_nested_as_deep_as_it_is_long_is_paired_in_one_pass():
    depth = 400
    line = ("".join("`" * length + "a" for length in range(depth, 1, -1))
            + "`b" * 150000 + "".join("a" + "`" * length for length in range(2, depth + 1)))
    start = time.monotonic()
    query.quote_span_list(line)
    assert time.monotonic() - start < 2


def test_a_request_is_cut_into_word_runs_at_spaces_backquotes_and_urls():
    run_list = query.word_run_list("`a/b.py`と設定.py see(https://x.y/a.py),b-c.py")
    assert [(run.text, run.in_quote, run.after_text) for run in run_list] == [
        ("a/b.py", True, False), (".py", False, True), ("see(", False, False),
        ("),b-c.py", False, False)]
    assert run_list[0].word_list == ["a", "b", "py"]
    assert run_list[0].gap_list == ["", "/", ".", ""]
    assert run_list[3].word_list == ["b-c", "py"] and run_list[3].gap_list == ["),", ".", ""]


@pytest.mark.parametrize("text, word_list", [
    ("`設定_v2.py`", ["設定_v2", "py"]), ("src/main.py->cfg.py", ["src", "main", "py", "cfg", "py"]),
    ("--no-overview", ["no-overview"]), ("a--b", ["a--b"])])
def test_a_word_of_a_run_keeps_its_hyphens_between_letters_only(text, word_list):
    assert [word for run in query.word_run_list(text) for word in run.word_list] == word_list


def test_a_long_run_is_read_in_time_that_grows_with_its_length():
    """Each of these is read once from end to end, whatever it repeats."""
    import time
    for piece in ("a/b)", "((", "a(", "x/", "[a]/", "`a/b`", "a/b#", "a:b/", "\\", "a@",
                  "a.", "/a.b", "a.b@", "x/()", "x/a()", "aB", "a_", "a-", "_", "://",
                  "a/b?", "`a`", "`"):
        start = time.monotonic()
        query.literal_term_list("fix " + piece * 50_000 + " now")
        assert time.monotonic() - start < 5, piece


@pytest.mark.parametrize("request_text", ["設定.py を直す", "app/設定.py を直す",
                                          "設定_v2.py を直す"])
def test_a_piece_of_a_name_outside_ascii_is_no_name_of_its_own(request_text):
    assert query.literal_term_list(request_text) == []


def test_a_piece_the_index_knows_is_a_name():
    known_set = {".env"}
    assert query.literal_term_list("の.env を消す", known_set.__contains__) == [".env"]
    assert query.literal_term_list("設定.py を直す", known_set.__contains__) == []


def test_a_name_written_after_a_word_outside_ascii_stays_a_name():
    assert query.literal_term_list("のloadConfig を直す") == ["loadConfig"]


def test_a_piece_of_a_name_outside_ascii_is_not_added_to_the_terms():
    assert query.expand_query("設定.py を直して", _llm_of("config")) == ("config", True)


@pytest.mark.parametrize("value, cap", [("20-40", 40), ("9" * 5000, 0), ("20-" + "9" * 12, 20),
                                        ("up to 999999999", 999999999), ("", 0),
                                        ("\uff12\uff10-\uff14\uff10", 0),
                                        ("20-\u0664\u0660", 20)])
def test_a_run_of_digits_too_long_for_a_count_is_no_number_of_the_word_range(
        monkeypatch, value, cap):
    monkeypatch.setattr(config, "QUERY_EXPAND_WORD_RANGE", value)
    assert query.word_cap() == cap
