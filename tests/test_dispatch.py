"""The stage that reads the request and splits it into questions or tasks.

run_dispatch() builds and runs a fresh read-only RLM, so the LLM is substituted for
these checks, the same way tests/test_plan.py substitutes the planning stage's.
"""

import json
import re

import pytest

from codetwine_loom import config, dispatch, testcreate, textcut
from codetwine_loom.languages import LANG_DICT


def _scripted_llm(*replies: str):
    """Build an LLM returning the given replies in order, repeating the last one."""
    seq = list(replies)

    def call(prompt: str) -> str:
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return call


def _complete(obj) -> str:
    """Return a turn that hands the object over as the JSON string COMPLETE() takes."""
    return ("```python\nCOMPLETE("
            + repr(json.dumps(obj, ensure_ascii=False)) + ")\n```")


@pytest.fixture
def stub_search(monkeypatch):
    """Skip the search: these tests are about what happens after it."""
    scope = {"files": [], "defs": [], "file_summaries": {}, "callers": {},
             "confidence": "high", "spread": 0}
    monkeypatch.setattr(dispatch, "search_scope", lambda request: scope)
    return scope


@pytest.fixture
def questioning_llm(monkeypatch):
    """Set up a stage that splits a question into two on its first turn."""
    llm = _scripted_llm(_complete({"questions": ["where is it read?",
                                                 "what happens without it?"]}))
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    return llm


# == the instructions =======================================
def test_the_instructions_build_for_all_11_extensions():
    for ext in LANG_DICT:
        assert dispatch.build_dispatch_instruction(ext)


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_no_injection_is_left_unresolved(ext):
    """A leftover {max_output_char} means a setting stopped reaching the model."""
    assert not re.search(r"\{[a-z_]+\}",
                         dispatch.build_dispatch_instruction(ext))


def test_the_instructions_say_nothing_is_edited():
    text = dispatch.build_dispatch_instruction("py")
    assert "no editing tools" in text
    assert "COMPLETE" in text


def test_the_instructions_show_both_ways_the_request_can_end():
    """A stage told only how to split a change would split a question the same way."""
    text = dispatch.build_dispatch_instruction("py")
    assert '"questions"' in text and '"tasks"' in text


def test_the_entries_are_asked_for_in_the_language_of_the_request():
    """An entry is what an agent is asked, so its language sets the answer's."""
    text = dispatch.build_dispatch_instruction("py")
    assert "Write each entry in the language of the request" in text


def test_the_instructions_tell_the_stage_not_to_answer():
    """The answering is the agents'; an answer written here reaches nobody."""
    text = dispatch.build_dispatch_instruction("py")
    assert "Do not answer a question yourself" in text
    assert '"answer"' not in text


def test_the_instructions_say_not_to_divide_the_files_up():
    """The overlap is what the combining run is for; dividing here would be a guess."""
    text = dispatch.build_dispatch_instruction("py")
    assert "NOT by which files it touches" in text


def test_the_instructions_never_offer_a_second_ending():
    """One terminal, so the split is read out of the object rather than out of which
    call the model happened to make."""
    assert "ANSWER(" not in dispatch.build_dispatch_instruction("py")


def test_the_output_limit_is_injected_from_the_settings(monkeypatch):
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 1234)
    assert "1234" in dispatch.build_dispatch_instruction("py")


def test_the_task_limits_are_injected_from_the_settings(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 7)
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 321)
    text = dispatch.build_dispatch_instruction("py")
    assert "7 entries" in text and "321 characters" in text


def test_the_split_keeps_tests_out_when_a_stage_writes_them(monkeypatch):
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", ["pytest"])
    assert "writing tests" in dispatch.build_dispatch_instruction("py")
    monkeypatch.setattr(config, "TEST_CREATE", False)
    assert dispatch.dispatch_test_note() == ""
    assert "writing tests" not in dispatch.build_dispatch_instruction("py")


@pytest.mark.parametrize("ext, with_test_file, is_on", [
    ("cbl", False, False), ("sql", False, False), ("cbl", True, True),
    ("py", False, True), ("py", True, True)])
def test_the_split_is_told_a_stage_writes_the_tests_only_where_the_stage_would_run(
        tmp_path, monkeypatch, ext, with_test_file, is_on):
    """A language with no dependency file the stage could write, in a repository with
    no test file yet: the stage calls no model, so the first test is an entry."""
    import subprocess
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / f"main.{ext}").write_text("x\n")
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    if with_test_file:
        (repo / "tests" / f"check.{ext}").write_text("x\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(repo))
    monkeypatch.setattr(config, "PRIMARY_EXT", ext)
    for name, value in (("TEST_CREATE", True), ("RUN_TEST", True), ("TEST_PATH", "tests"),
                        ("TEST_COMMAND", ["./run-tests.sh"]), ("OTHER_TEST_LIST", [])):
        monkeypatch.setattr(config, name, value)
    assert testcreate.is_stage_on() is is_on
    assert (testcreate.skip_reason(str(repo), {"main": ["f"]}) == "") is is_on
    assert (dispatch._TEST_CREATE_NOTE in dispatch.build_dispatch_instruction(ext)) is is_on
    # With no repository named, the settings alone decide
    monkeypatch.setattr(config, "PROJECT_DIR", "")
    assert testcreate.is_stage_on() is True


@pytest.mark.parametrize("name, value", [("RUN_TEST", False), ("TEST_COMMAND", []),
                                         ("TEST_CREATE", False)])
def test_the_split_is_not_told_a_stage_writes_the_tests_when_none_will(monkeypatch, name,
                                                                      value):
    """With the tests switched off, and in a language with no test command, the stage
    that writes tests calls no model, so a test the request asks for is an entry."""
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", ["pytest"])
    with_note = dispatch.build_dispatch_instruction("py")
    assert testcreate.is_stage_on() is True and dispatch._TEST_CREATE_NOTE in with_note
    monkeypatch.setattr(config, name, value)
    assert testcreate.is_stage_on() is False and dispatch.dispatch_test_note() == ""
    assert dispatch.build_dispatch_instruction("py") == \
        with_note.replace(dispatch._TEST_CREATE_NOTE, "")
    assert bool(testcreate.skip_reason(".", {"a.py": ["f"]})) is (name != "TEST_CREATE")


def test_the_split_is_told_how_many_files_a_change_plans_only_when_it_plans(monkeypatch):
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 7)
    monkeypatch.setattr(config, "TWO_STAGE", True)
    with_note = dispatch.build_dispatch_instruction("py")
    assert ("with the work half done, and an agent making a change plans at most\n"
            "  7 files. A request written as one sentence") in with_note
    monkeypatch.setattr(config, "TWO_STAGE", False)
    without_note = dispatch.build_dispatch_instruction("py")
    assert "plans at most" not in without_note and "7 files" not in without_note
    assert "with the work half done. A request written as one sentence" in without_note
    assert dispatch.dispatch_plan_note() == ""
    assert without_note == with_note.replace(dispatch._PLAN_NOTE.format(plan_max_file=7), "")


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_no_injection_is_left_unresolved_when_a_stage_writes_tests(ext, monkeypatch):
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", ["pytest"])
    assert dispatch._TEST_CREATE_NOTE in dispatch.build_dispatch_instruction(ext)
    assert not re.search(r"\{[a-z_]+\}", dispatch.build_dispatch_instruction(ext))


def test_the_run_note_is_left_out_without_a_launcher(monkeypatch):
    """Describing a tool that could only refuse costs a turn to find out."""
    monkeypatch.setattr(config, "RUN_COMMAND", [])
    assert dispatch.dispatch_run_note(dispatch.dispatch_tool_list()) == ""
    assert "run_project" not in dispatch.build_dispatch_instruction("py")


def test_a_run_budget_of_0_takes_run_project_away_from_the_splitting_stage(monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    assert "run_project" in dispatch.build_dispatch_instruction("py")
    monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", 0)
    assert "run_project" not in [t.__name__ for t in dispatch.dispatch_tool_list()]
    assert "run_project" not in dispatch.build_dispatch_instruction("py")


def test_the_run_note_names_the_launcher(monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    assert "`uv run`" in dispatch.build_dispatch_instruction("py")


def test_the_run_note_is_left_out_for_a_stage_not_handed_run_project(monkeypatch):
    from codetwine_loom import plan
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    assert dispatch.dispatch_run_note(plan.answer_tool_list()) == ""
    assert "run_project" not in dispatch.read_rule("py", plan.answer_tool_list())
    assert "run_project" in dispatch.read_rule("py", dispatch.dispatch_tool_list())


def test_the_web_question_reaches_it_only_when_it_is_switched_on(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    assert "web_search" not in dispatch.build_dispatch_instruction("py")
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    text = dispatch.build_dispatch_instruction("py")
    assert "web_search" in text and "2 calls" in text


# == the tools ==============================================
def test_not_one_editing_tool_is_handed_over():
    names = {t.__name__ for t in dispatch.dispatch_tool_list()}
    assert not names & {"replace_in_file", "replace_line_range", "create_file",
                        "run_test", "revert_file", "git_diff_stat"}
    assert "read_definition" in names


def test_the_tools_are_the_read_only_ones_the_planning_stage_gets():
    """One list, so a tool added for reading reaches both stages."""
    from codetwine_loom import plan
    assert dispatch.dispatch_tool_list() == plan.plan_tool_list()


# == the definitions each task names ========================
def test_a_definition_is_taken_only_when_the_task_names_its_file_too(fake_index):
    """save_config has one definition, but a name alone is not enough."""
    assert dispatch.task_def_list(["change save_config"]) == [[]]
    assert dispatch.task_def_list(["change save_config in p/cfg.py"]) == \
        [["p/cfg_py/cfg.py::save_config"]]


def test_a_definition_named_by_its_file_name_alone_is_taken(fake_index):
    """load_config is in two files; the file name says which."""
    assert dispatch.task_def_list(["load_config in cfg.py", "load_config in util.py",
                                   "load_config"]) == \
        [["p/cfg_py/cfg.py::load_config"], ["p/util_py/util.py::load_config"], []]


def test_a_file_name_inside_a_longer_name_does_not_count(fake_index):
    assert dispatch.task_def_list(["save_config in mycfg.py"]) == [[]]


def test_a_path_written_against_japanese_text_still_names_its_file(fake_index):
    assert dispatch.task_def_list(["p/cfg.pyのsave_configを直す"]) == \
        [["p/cfg_py/cfg.py::save_config"]]


def test_a_cobol_name_and_a_name_outside_ascii_are_taken_from_a_task(index_of):
    def node(name, file):
        return {"name": name, "type": "paragraph_header", "file": file, "lines": [1, 2],
                "signature": f"       {name}.", "doc": ""}
    index_of({"p/src/MAIN_cbl/MAIN.cbl::MAIN-PARA": node("MAIN-PARA", "p/src/MAIN_cbl/MAIN.cbl"),
              "p/src/MAIN_cbl/MAIN.cbl::顧客処理": node("顧客処理", "p/src/MAIN_cbl/MAIN.cbl")},
             {})
    assert dispatch.task_def_list(["MAIN.cblのmain-paraを直す", "MAIN.cblの顧客処理を直す",
                                   "main-para の外"]) == \
        [["p/src/MAIN_cbl/MAIN.cbl::MAIN-PARA"], ["p/src/MAIN_cbl/MAIN.cbl::顧客処理"], []]


# == a definition one task adds and a later task uses ========
def _chain(task_list: list[str]) -> list[int]:
    """Return the chain numbers the host gives the tasks."""
    from codetwine_loom import sandbox
    return sandbox.task_chain_list(dispatch.task_def_list(task_list))


def test_a_definition_one_task_adds_and_another_uses_chains_them(fake_index):
    task_list = ["Add a new function parse_path(text) to cfg.py.",
                 "In util.py, call parse_path (defined in cfg.py) before returning.",
                 "Rename the log lines in util.py."]
    assert dispatch.task_def_list(task_list) == [
        ["p/cfg_py/cfg.py::parse_path"], ["p/cfg_py/cfg.py::parse_path"], []]
    assert _chain(task_list) == [1, 1, 0]


@pytest.mark.parametrize("name", ["parsePath", "PARSE-PATH", "`parse`", "`parse()`",
                                  "`解析する`"])
def test_a_new_name_written_as_in_code_or_in_backticks_chains(fake_index, name):
    task_list = [f"Add {name} to cfg.py.", f"Call {name} of cfg.py from util.py."]
    assert _chain(task_list) == [1, 1]


@pytest.mark.parametrize("signature", [
    "`parse(text: str) -> list[str]`", "`def parse(text):`",
    "`export function parse(text: string): string[]`",
    "`public static String[] parse(String text)`", "`fn parse(text: &str) -> Vec<String>`",
    "`func (c *Calc) parse(text string) []string`", "`parse <- function(text) text`",
    "`export const parse = (text) => [text]`", "`const parse: Fn = (text) => [text]`",
    "`parse<T>(xs: T[]): T[]`", "`int Calc::parse(int a)`", "`fun String.parse(): Int`",
    "`def parse(text: str | None = None, *, sep: str = \",\", limit: int = -1) -> list`",
    "`export const parse = (text: string | null): string[] => [text]`",
    "`const parse = ({ text, sep }: Options) => [text]`",
    "`pub fn parse<'a>(text: &'a str) -> Vec<&'a str>`",
    "`Map<K, V> parse(Map<K, V> m, List<String> xs)`"])
def test_a_new_name_written_as_a_signature_in_backticks_chains(fake_index, signature):
    """One task writes the definition it adds as its signature, the other names it."""
    task_list = [f"Add {signature} to cfg.py.", "Register `parse` of cfg.py in util.py."]
    assert _chain(task_list) == [1, 1]


@pytest.mark.parametrize("code", [
    "`a - b`", "`is None`", "`x[\"parse\"]`", "`if (b === 0) throw new Error(\"x\")`",
    "`throw new Error(\"x\")`", "`while (i < n)`", "`sub(5, 2) == 3`",
    "`for (int i = 0; i < n; i++)`", "`expect(sub(5, 2)).toBe(3)`", "`if(b === 0)`",
    "`assert(x == y)`", "`for(int i = 0; i < n; i++)`", "`f(None)`", "`f(!ok)`"])
def test_code_in_backticks_that_defines_nothing_names_nothing(code):
    assert dispatch._quote_name_set(f"return {code} from cfg.py") == set()


@pytest.mark.parametrize("code", [
    "`parse(text: (s: string) => void): void`", "`def parse(node: \"Node\") -> None:`",
    "`fn parse<F: Fn(i32) -> i32>(f: F)`", "`if(x>0) parse(x)`"])
def test_a_name_inside_any_code_chains_with_a_task_that_names_it_outright(fake_index, code):
    """The task that adds a definition may write it in any code; the task that uses it
    names it in backticks."""
    task_list = [f"Add {code} to cfg.py.", "Register `parse` of cfg.py in util.py."]
    assert _chain(task_list) == [1, 1]


def test_two_tasks_that_name_a_word_only_inside_code_do_not_chain(fake_index):
    """A word neither task names outright, such as "if" or "Error", chains nothing."""
    task_list = ["Add div to cfg.py: `if(b === 0) throw new Error(\"zero\")`.",
                 "Add sqrt to cfg.py: `if(x < 0) throw new Error(\"negative\")`."]
    assert _chain(task_list) == [0, 0]


def test_two_tasks_that_write_the_same_statement_in_one_file_do_not_chain(fake_index):
    """A condition or a call both tasks write is no definition either of them adds."""
    task_list = ["Add div to cfg.py: `if (b === 0) throw new Error(\"zero\")`.",
                 "Add sqrt to cfg.py: `if (x < 0) throw new Error(\"negative\")`."]
    assert _chain(task_list) == [0, 0]


def test_a_new_name_in_a_language_that_ignores_case_chains_across_cases(index_of):
    index_of({"p/src/MAIN_cbl/MAIN.cbl::MAIN-PARA": {
        "name": "MAIN-PARA", "type": "paragraph_header", "file": "p/src/MAIN_cbl/MAIN.cbl",
        "lines": [1, 2], "signature": "       MAIN-PARA.", "doc": ""}}, {})
    task_list = ["Add PARSE-PATH to MAIN.cbl.", "Perform `parse-path` of MAIN.cbl from MAIN-PARA.",
                 "Add parse_path to cfg.py.", "Call Parse_Path of cfg.py."]
    key_list_list = dispatch.task_def_list(task_list)
    assert "MAIN.cbl::PARSE-PATH" in key_list_list[0] and "MAIN.cbl::PARSE-PATH" in key_list_list[1]
    assert _chain(task_list) == [1, 1, 0, 0]


def test_a_new_file_named_by_both_tasks_is_the_file(fake_index):
    task_list = ["Create src/paths.py holding split_path(text).",
                 "Make util.py use split_path from paths.py."]
    assert dispatch.task_def_list(task_list) == [["paths.py::split_path"]] * 2
    assert _chain(task_list) == [1, 1]


@pytest.mark.parametrize("task_list", [
    # words, not names in code, on the same file
    ["Make the parse step in cfg.py stricter.", "Speed up the parse step in cfg.py."],
    # a name in code, but in no file both name
    ["Add parse_path to cfg.py.", "Call parse_path from util.py."],
    # a file name with "_" is not a definition name
    ["Fix the header of new_util.py.", "Fix the footer of new_util.py."],
    # a name only one task names
    ["Add parse_path to cfg.py.", "Add split_path to cfg.py."],
])
def test_tasks_that_share_no_new_definition_and_file_do_not_chain(fake_index, task_list):
    assert _chain(task_list) == [0, 0]


@pytest.mark.parametrize("first", [
    "cfg.py で`make test`が落ちるので`retry`を足す",
    "cfg.py: `go test ./...`で`retry`が落ちるので直す",
    "In cfg.py `make test`and`retry`fail",
    "`make test` が落ちるので cfg.py に `retry` を足す",
    "cfg.py に ``retry`` を足す",
    "cfg.py に `retry()` を足す",
    "cfg.py: ``retry`` を足す（`` は使わない）",
    "cfg.py: ``retry`` を足す。```` で囲まない",
])
def test_a_new_name_in_backticks_is_read_as_the_request_s_words_in_backticks_are(
        fake_index, first):
    """The backtick that closes a phrase with a space in it opens no name."""
    task_list = [first, "cfg.py の `retry` を util.py から呼ぶ"]
    assert dispatch.task_def_list(task_list)[0] == ["p/cfg_py/cfg.py::retry"]
    assert _chain(task_list) == [1, 1]
    assert dispatch._quote_name_set(first) == {"retry"}


def test_a_word_in_backticks_that_is_no_name_is_none(fake_index):
    assert dispatch._quote_name_set("`a.b` と `1st` と `two words` と `x-y` と `f()`") \
        == {"x-y", "f"}


def test_a_name_the_index_holds_in_a_named_file_is_its_key_alone(fake_index):
    task_list = ["Change save_config in cfg.py.", "Call save_config of cfg.py in util.py."]
    assert dispatch.task_def_list(task_list) == [["p/cfg_py/cfg.py::save_config"]] * 2


def test_no_definition_is_named_without_an_index(monkeypatch):
    monkeypatch.setattr(dispatch.scope_mod, "INDEX", None)
    assert dispatch.task_def_list(["one", "two"]) == [[], []]


# == parse_dispatch =========================================
def test_questions_are_read_as_questions():
    out = dispatch.parse_dispatch('{"questions": ["where?", "how?"]}')
    assert out == {"ok": True, "questions": ["where?", "how?"]}


def test_an_answer_alone_is_refused():
    """A reply holding only an answer and no tasks hands nothing over."""
    out = dispatch.parse_dispatch('{"answer": "it is read in config.py"}')
    assert out["ok"] is False and "no entry under" in out["error"]


def test_tasks_are_read_as_tasks():
    out = dispatch.parse_dispatch('{"tasks": ["do this", "do that"]}')
    assert out == {"ok": True, "tasks": ["do this", "do that"]}


def test_the_object_is_taken_as_it_is_when_it_is_not_a_string():
    """COMPLETE() is called from a Python REPL, so a dict is the natural thing to pass."""
    assert dispatch.parse_dispatch({"tasks": ["do this"]}) == {"ok": True,
                                                               "tasks": ["do this"]}


def test_holding_both_is_refused():
    out = dispatch.parse_dispatch('{"questions": ["a"], "tasks": ["b"]}')
    assert out["ok"] is False and "exactly one of" in out["error"]


def test_holding_neither_is_refused():
    out = dispatch.parse_dispatch('{"facts": []}')
    assert out["ok"] is False and "no entry under" in out["error"]


def test_an_empty_task_list_is_refused():
    """A request read as a change with nothing to do would start no agent at all."""
    out = dispatch.parse_dispatch('{"tasks": []}')
    assert out["ok"] is False and "no entry under" in out["error"]


def test_every_entry_the_instructions_show_is_an_example():
    assert dispatch._EXAMPLE_ENTRY_SET == {
        "one whole question", "one whole change request", "another",
        "one thing to settle, naming the readings it is between"}


def test_an_entry_with_nothing_written_out_is_not_handed_out():
    out = dispatch.parse_dispatch(json.dumps(
        {"tasks": ["...", "fix a in a.py", "-", "設定を直す", "42", "One whole change request"]}))
    assert out["ok"] is True and out["tasks"] == ["fix a in a.py", "設定を直す", "42"]


def test_text_that_is_not_json_is_refused_with_what_came_back():
    out = dispatch.parse_dispatch("I think we should split this in two")
    assert out["ok"] is False and "raw" in out


def test_nothing_handed_over_says_which_of_the_two_it_was():
    assert "turns" in dispatch.parse_dispatch(None)["error"]
    assert "broker" in dispatch.parse_dispatch(None, "the broker did not answer")["error"]


def test_the_task_count_is_cut_to_the_setting(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 2)
    out = dispatch.parse_dispatch('{"tasks": ["a", "b", "c", "d"]}')
    assert out["tasks"] == ["a", "b"]


def test_one_task_is_cut_to_the_setting(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 10)
    out = dispatch.parse_dispatch(json.dumps({"tasks": ["x" * 200]}))
    assert len(out["tasks"][0]) <= 10 + len("\n... (190 chars omitted)")


def test_questions_are_held_to_the_same_limits_as_tasks(monkeypatch):
    """One rule for both lists, so a limit raised for one is raised for the other."""
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 2)
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 10)
    out = dispatch.parse_dispatch(json.dumps({"questions": ["y" * 100, "b", "c"]}))
    assert len(out["questions"]) == 2
    assert len(out["questions"][0]) <= 10 + len("\n... (90 chars omitted)")


def test_a_task_that_is_not_text_is_dropped():
    """A dict would reach the agent as a Python repr, which is not a request."""
    out = dispatch.parse_dispatch('{"tasks": [{"path": "a.py"}, "do this"]}')
    assert out["tasks"] == ["do this"]


@pytest.mark.parametrize("answer, entry_dict", [
    ('{"tasks": ["do this", null, 7, true, 1.5, ["x"], {"why": "' + "y" * 30 + '"}]}',
     {"tasks": ["do this"]}),
    ('{"tasks": [null], "questions": ["what is a()?"]}', {"questions": ["what is a()?"]}),
    ('{"questions": [null, false]}', None),
    ('{"tasks": ["  ", ""], "questions": [0]}', None)])
def test_an_entry_that_is_no_string_is_no_entry_to_hand_out(monkeypatch, answer, entry_dict):
    """The guard and the parser read the lists alike."""
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 20)
    out = dispatch.parse_dispatch(answer)
    text = dispatch._over_the_limit(answer)
    if entry_dict is None:
        assert "no entry under" in out["error"] and f"- {out['error']}" in text
    else:
        assert out == {"ok": True, **entry_dict}
        assert text == ""


# == run_dispatch ===========================================
def test_what_the_stage_does_to_scope_does_not_reach_the_search_result(stub_search,
                                                                      monkeypatch):
    stub_search.update(files=["a.py"], defs=[{"key": "a.py::f"}])
    turn = ('```python\nscope["files"].append("x.py")\nscope["defs"].clear()\nscope.pop("confidence")\n'
            'COMPLETE({"questions": ["where is it read?"]})\n```')
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(turn))
    out = dispatch.run_dispatch("設定はどこで読んでいる?")
    assert out["questions"] == ["where is it read?"] and out["scope"] is stub_search
    assert stub_search == {"files": ["a.py"], "defs": [{"key": "a.py::f"}],
                           "file_summaries": {}, "callers": {}, "confidence": "high",
                           "spread": 0}


def test_the_questions_come_back_with_the_search_result(stub_search, questioning_llm):
    out = dispatch.run_dispatch("設定はどこで読んでいる?")
    assert out["questions"] == ["where is it read?", "what happens without it?"]
    assert "tasks" not in out and "answer" not in out
    assert out["scope"] is stub_search
    assert out["harvested"] is False and out["iters"] == 1


def test_a_change_comes_back_as_the_tasks_the_agents_are_given(stub_search,
                                                               monkeypatch):
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(
        _complete({"tasks": ["rename read_config", "update its callers"]})))
    out = dispatch.run_dispatch("設定の読み込みを整理して")
    assert out["tasks"] == ["rename read_config", "update its callers"]
    assert "answer" not in out


def test_a_split_of_tasks_carries_the_definitions_they_name(stub_search, monkeypatch):
    monkeypatch.setattr(dispatch.scope_mod, "INDEX", None)
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(
        _complete({"tasks": ["rename read_config", "update its callers"]})))
    assert dispatch.run_dispatch("設定の読み込みを整理して")["defs"] == [[], []]


def test_a_split_of_questions_carries_no_definition(stub_search, questioning_llm):
    assert "defs" not in dispatch.run_dispatch("設定はどこで読んでいる?")


def test_the_request_is_what_the_model_reads(stub_search, monkeypatch):
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return _complete({"questions": ["here?"]})

    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    dispatch.run_dispatch("設定はどこで読んでいる?")
    assert "設定はどこで読んでいる?" in prompts[0]


def test_a_stage_that_runs_out_of_turns_is_asked_once_for_the_object(stub_search,
                                                                     monkeypatch):
    """The reading is already paid for; only the hand-over is missing."""
    monkeypatch.setattr(config, "DISPATCH_MAX_TURN", 2)
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(
        "```python\nprint('still reading')\n```",
        "```python\nprint('still reading')\n```",
        '{"questions": ["where is it read?"]}'))
    out = dispatch.run_dispatch("設定はどこで読んでいる?")
    assert out["harvested"] is True and out["questions"] == ["where is it read?"]


def test_the_harvest_shows_both_shapes(stub_search, monkeypatch):
    """A question that ran out of turns has to come back as questions in that one
    call, or it comes back as a split of something that was never a change."""
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return "```python\nprint('still reading')\n```"

    monkeypatch.setattr(config, "DISPATCH_MAX_TURN", 1)
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    dispatch.run_dispatch("設定はどこで読んでいる?")
    assert '"questions"' in prompts[-1] and '"tasks"' in prompts[-1]


def _harvest_prompt(monkeypatch) -> str:
    """Run a stage that never calls COMPLETE() and return the harvest's prompt."""
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return "```python\nprint('still reading')\n```"

    monkeypatch.setattr(config, "DISPATCH_MAX_TURN", 1)
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    dispatch.run_dispatch("q")
    return prompts[-1]


def test_a_task_length_that_is_off_is_not_stated(stub_search, monkeypatch):
    """0 is no limit for the guard and the cut; the two texts that state the length
    lose that phrase and nothing else."""
    phrase = f", {config.DISPATCH_TASK_MAX_CHAR} characters each"
    count = config.DISPATCH_MAX_TASK
    text = dispatch.build_dispatch_instruction("py")
    late = _harvest_prompt(monkeypatch)
    assert f"- At most {count} entries{phrase}.\n" in text
    assert f"At most {count} entries{phrase}. Hand\n" in late
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 0)
    assert dispatch.build_dispatch_instruction("py") == text.replace(phrase, "")
    assert _harvest_prompt(monkeypatch) == late.replace(phrase, "")


def test_a_run_whose_call_never_went_through_is_not_asked_again(stub_search,
                                                                monkeypatch):
    """The same call would not go through for the harvest either, and the failure
    would leave run_dispatch instead of the run reporting what it had."""
    calls = []

    def dead(prompt):
        calls.append(prompt)
        raise RuntimeError("the broker did not answer")

    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: dead)
    out = dispatch.run_dispatch("設定はどこで読んでいる?")
    assert len(calls) == 1
    assert out["ok"] is False and out["harvested"] is False
    assert "the broker did not answer" in out["error"]


def test_the_turn_limit_is_its_own_setting(stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_TURN", 3)
    monkeypatch.setattr(config, "RLM_MAX_TURN", 99)
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return _complete({"questions": ["here?"]})

    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    dispatch.run_dispatch("設定はどこで読んでいる?")
    assert "## Turn 1/3" in prompts[0]


def test_the_stage_calls_the_model_its_own_setting_names(stub_search, monkeypatch):
    """The splitting stage may be a stronger model than the agents it hands to."""
    monkeypatch.setattr(config, "DISPATCH_MODEL", "provider/splitter")
    monkeypatch.setattr(config, "ANSWER_MODEL", "provider/agent")
    asked = []

    def make(model, max_tokens, setting_name="MEDIUM_MODEL", **kw):
        asked.append((model, setting_name))
        return _scripted_llm(_complete({"questions": ["here?"]}))

    monkeypatch.setattr(dispatch, "make_llm", make)
    dispatch.run_dispatch("設定はどこで読んでいる?")
    assert asked == [("provider/splitter", "DISPATCH_MODEL")]


# == Carrying on from what an earlier run left ==============

def _left(branch="dispatch/1/1", held=False, **told):
    """Return one integrate.branch_record_dict() result with a single run on it."""
    run = {"asked": "add the flag", "did": "added it", "verdict": "failed",
           "why": "the turns ran out"}
    run.update(told)
    return {"branch": branch, "commit": "c" * 40, "held": held, "told": [run],
            "earlier_list": []}


def test_what_the_tests_found_not_done_is_shown_under_the_run_that_recorded_it():
    block = dispatch.render_earlier([_left(
        gap='[{"what":"case_8","why":"imports are not resolved"}]')])
    assert "  not done: case_8 — imports are not resolved" in block


def test_a_case_a_later_commit_found_done_is_not_shown_and_one_left_is_shown_once(
        monkeypatch):
    """Two test tasks each recorded a case; the commit combining them found the first
    done and recorded the second again."""
    case_1 = '[{"what":"case-1","why":"f(-1) raises"}]'
    case_2 = '[{"what":"case-2","why":"g() returns None"}]'
    run = {"asked": "change f", "did": "", "verdict": "settled", "why": ""}
    told = [{**run, "did": "merged"}, {**run, "gap": case_1}, {**run, "gap": case_2},
            {**run, "did": "combined", "gap": case_2, "gap_done": case_1}]
    left = {"branch": "codetwine-loom/u", "commit": "c" * 40, "held": False, "told": told,
            "earlier_list": []}
    block = dispatch.render_earlier([left])
    assert "case-1" not in block
    assert block.count("  not done: case-2 — g() returns None") == 1
    # Under the newest run that recorded it
    assert "  did: combined\n  ended: settled\n  not done: case-2" in block
    # A case only a run that is not shown recorded goes under the newest run shown
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 1)
    block = dispatch.render_earlier([{**left, "told": [{**run, "gap": case_2},
                                                       {**run, "did": "refactored"}]}])
    assert "  did: refactored\n  ended: settled\n  not done: case-2" in block


def test_what_an_earlier_run_left_goes_in_front_of_the_request(stub_search,
                                                               monkeypatch):
    """The stage reads it before it has read anything, so it stands where the change
    history's own block stands for a single agent."""
    monkeypatch.setattr(dispatch.integrate, "branch_record_dict", lambda *a: _left())
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return _complete({"tasks": ["finish the flag"]})

    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    dispatch.run_dispatch("--verbose を足して", resume_list=["dispatch/1/1"])
    block = prompts[0].index("What an earlier run left on dispatch/1/1")
    assert block < prompts[0].index("--verbose を足して")
    for said in ("add the flag", "added it", "failed", "the turns ran out"):
        assert said in prompts[0]


def test_the_search_is_made_on_the_request_alone(stub_search, monkeypatch):
    """What an earlier run left says what is done, not what this one is about."""
    monkeypatch.setattr(dispatch.integrate, "branch_record_dict", lambda *a: _left())
    searched = []
    monkeypatch.setattr(dispatch, "search_scope",
                        lambda request: searched.append(request) or stub_search)
    monkeypatch.setattr(dispatch, "make_llm",
                        lambda *a, **k: _scripted_llm(_complete({"tasks": ["fix a.py"]})))
    dispatch.run_dispatch("--verbose を足して", resume_list=["dispatch/1/1"])
    assert searched == ["--verbose を足して"]


def test_the_request_is_the_whole_prompt_when_nothing_is_carried_on_from(stub_search,
                                                                         monkeypatch):
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return _complete({"tasks": ["fix a.py"]})

    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    dispatch.run_dispatch("設定はどこで読んでいる?")
    assert "What an earlier run left" not in prompts[0]


@pytest.mark.parametrize("held, said", [
    (True, "already holds this work"),
    (False, "does NOT hold this work"),
])
def test_whether_the_repository_holds_that_work_is_said(held, said):
    assert said in dispatch.render_earlier([_left(held=held)])


@pytest.mark.parametrize("earlier_held, own_held", [
    (True, False), (False, True), (True, True)])
def test_whether_the_repository_holds_the_work_is_said_for_each_union_a_branch_stands_on(
        earlier_held, own_held):
    """The runs of the earlier union a branch's merge stands on come first, under what
    is said of that union; the sentence is said again only where it changes."""
    entry = _left(held=own_held, asked="the new union")
    earlier_run = dict(entry["told"][0], asked="the earlier union",
                       gap='[{"what":"case_1","why":"a.txt does not do it"}]')
    entry["told"] = [earlier_run, *entry["told"]]
    entry["earlier_list"] = [{"held": earlier_held, "told": [earlier_run]}]
    block = dispatch.render_earlier([entry])
    held_text = {True: "already holds this work", False: "does NOT hold this work"}
    assert block.index(held_text[earlier_held]) < block.index("the earlier union") \
        < block.index("not done: case_1") < block.index("the new union")
    if earlier_held == own_held:
        assert block.count("this work") == 1
    else:
        assert block.index("not done: case_1") < block.index(held_text[own_held]) \
            < block.index("the new union")


@pytest.mark.parametrize("limit", [1, 2, 40])
@pytest.mark.parametrize("earlier_held, own_held", [(False, True), (True, False)])
def test_a_case_whose_run_is_cut_is_shown_under_what_is_said_of_the_run_that_recorded_it(
        monkeypatch, limit, earlier_held, own_held):
    """The earlier union's runs are cut: its case goes under the line saying how many
    runs are not shown, after what is said of the earlier union, and the runs shown come
    under what is said of them."""
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", limit)
    run = _left()["told"][0]
    earlier_told = [dict(run, asked="the earlier merge"),
                    dict(run, asked="the earlier test",
                         gap='[{"what":"case_1","why":"a.txt does not do it"}]')]
    entry = _left(held=own_held)
    entry["told"] = [*earlier_told, dict(run, asked="the new merge"),
                     dict(run, asked="the new stage")]
    entry["earlier_list"] = [{"held": earlier_held, "told": earlier_told}]
    block = dispatch.render_earlier([entry])
    held_text = {True: "already holds this work", False: "does NOT hold this work"}

    def said_before(text):
        head = block[:block.index(text)]
        return max(held_text, key=lambda held: head.rfind(held_text[held]))

    assert block.count("not done: case_1") == 1
    assert said_before("not done: case_1") is earlier_held
    assert said_before("the new stage") is own_held
    if limit < len(entry["told"]):
        assert block.index("not shown here") < block.index("not done: case_1") \
            < block.index("the new stage")


def test_a_run_the_machine_never_judged_is_said_to_be_that(stub_search):
    """A commit carrying no verdict is one no version of this wrote, and reading a
    blank as "it ended fine" is worse than saying nothing was recorded."""
    block = dispatch.render_earlier([_left(verdict="", why="")])
    assert "(nothing was recorded)" in block


def test_a_run_that_said_nothing_about_what_it_did_is_said_to_be_that(stub_search):
    assert "(it never said)" in dispatch.render_earlier([_left(did="")])


def test_a_branch_that_says_nothing_at_all_is_left_out(stub_search, caplog):
    """A block naming a branch and describing nothing tells the stage less than no
    block does."""
    empty = {"branch": "dispatch/1/1", "commit": "c" * 40, "held": False, "told": [],
             "earlier_list": []}
    assert dispatch.render_earlier([empty]) == ""


def test_every_branch_named_is_described(stub_search):
    block = dispatch.render_earlier([_left("dispatch/1/1"), _left("dispatch/1/2")])
    assert block.count("What an earlier run left on") == 2


def test_of_more_runs_than_are_shown_the_newest_are_kept_and_the_rest_counted(
        monkeypatch):
    """The runs come oldest first: the agents', then the merge's, then the stages'."""
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    entry = _left()
    entry["told"] = [dict(entry["told"][0], asked=f"task {n}") for n in (1, 2)] + [
        dict(entry["told"][0], asked="the whole request", verdict="unsettled",
             why="conflict markers are still in: a.py")]
    block = dispatch.render_earlier([entry])
    assert "task 1" not in block and "task 2" in block
    assert "ended: unsettled - conflict markers are still in: a.py" in block
    assert block.index("(1 run(s) before the ones below are on this branch and are not "
                       "shown here)") < block.index("task 2")
    # A branch within the limit says nothing of the kind
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 3)
    assert "not shown here" not in dispatch.render_earlier([entry])


def test_of_more_branches_than_are_shown_the_rest_are_named(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    block = dispatch.render_earlier([_left(f"dispatch/1/{n}") for n in (1, 2, 3, 4)])
    assert block.count("What an earlier run left on") == 2
    assert ("2 more branch(es) were named and are not shown here: dispatch/1/3, "
            "dispatch/1/4") in block
    assert block.index("not shown here") < block.index("Read the request below")


def test_what_a_run_was_asked_for_is_cut_to_what_a_task_is_cut_to(stub_search,
                                                                  monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 20)
    block = dispatch.render_earlier([_left(asked="y" * 500)])
    assert "y" * 500 not in block and "y" * 20 in block


def test_why_a_run_ended_that_way_is_cut_to_what_an_account_of_it_is_cut_to(
        stub_search, monkeypatch):
    """A merging run writes everything its gates found into that one trailer, so it is
    the field a branch can make arbitrarily long."""
    monkeypatch.setattr(config, "EDIT_HISTORY_DID_CHAR", 20)
    block = dispatch.render_earlier([_left(verdict="unsettled", why="w" * 500)])
    assert "w" * 20 in block and "w" * 21 not in block


def test_nothing_is_read_when_no_branch_is_named(monkeypatch):
    monkeypatch.setattr(dispatch.integrate, "branch_record_dict",
                        lambda *a: pytest.fail("a branch was read that nobody named"))
    assert dispatch.earlier_run_text([]) == ""


def test_what_to_do_with_the_block_is_said_once_however_many_branches(stub_search):
    """Three branches saying the same paragraph three times is three chances to read
    past it."""
    block = dispatch.render_earlier([_left("dispatch/1/1"), _left("dispatch/1/2"),
                                     _left("dispatch/1/3")])
    assert block.count("Read the request below") == 1


# == an entry over the limit is sent back once before it is cut ==
def _over(chars: int) -> str:
    return _complete({"questions": ["short one", "y" * chars]})


def test_an_entry_over_the_limit_is_sent_back_once(stub_search, monkeypatch):
    """The model shortens it itself; the cut is only for what is still over."""
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 50)
    prompts = []
    replies = iter([_over(80), _complete({"questions": ["short one", "y" * 40]})])

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    out = dispatch.run_dispatch("q")
    assert out["questions"] == ["short one", "y" * 40]
    assert out["refused_completes"] == 1 and out["iters"] == 2
    assert "question 2 (80 characters)" in prompts[1]
    assert "over the 50 characters" in prompts[1]


def test_an_entry_still_over_after_being_sent_back_is_cut(stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 50)
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(_over(80)))
    out = dispatch.run_dispatch("q")
    assert out["refused_completes"] == 1 and out["iters"] == 2
    assert len(out["questions"][1]) < 80 and "chars omitted" in out["questions"][1]


def test_entries_within_the_limit_are_not_sent_back(stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 50)
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(_over(50)))
    out = dispatch.run_dispatch("q")
    assert out["refused_completes"] == 0 and out["iters"] == 1


def test_a_limit_that_is_off_sends_nothing_back(stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 0)
    monkeypatch.setattr(dispatch, "make_llm",
                        lambda *a, **k: _scripted_llm(_over(5000)))
    out = dispatch.run_dispatch("q")
    assert out["refused_completes"] == 0 and len(out["questions"][1]) == 5000


def test_a_split_that_cannot_be_read_is_sent_back_once(stub_search, monkeypatch):
    """A JSON string written inside a Python string loses its escapes in the REPL, so
    the model is told once to hand the object itself over; a split still unreadable
    after that is the parser's refusal and ends the run."""
    prompts = []
    # The first turn's REPL string resolves the escaped quotes, leaving broken JSON; the
    # second hands the dict over as the refusal says
    first = "```python\nCOMPLETE('{\"tasks\": [\"say \\\"hi\\\" here\"]}')\n```"
    second = "```python\nCOMPLETE({\"tasks\": ['say \"hi\" here']})\n```"
    replies = iter([first, second])

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    out = dispatch.run_dispatch("q")
    assert out["ok"] is True and out["tasks"] == ['say "hi" here']
    assert out["refused_completes"] == 1 and out["iters"] == 2
    assert "could not be read as JSON" in prompts[1] and "COMPLETE({...})" in prompts[1]


def test_a_split_that_stays_unreadable_is_sent_back_each_turn_and_ends_the_run(
        stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_TURN", 4)
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(
        "```python\nCOMPLETE('not json at all')\n```"))
    out = dispatch.run_dispatch("q")
    assert out["ok"] is False and out["refused_completes"] >= 4


def test_an_unreadable_split_after_the_one_refusal_is_still_sent_back(stub_search,
                                                                      monkeypatch):
    """The one refusal of a split with the wrong shape does not use up the sending back of
    one that cannot be read: taking it would end the run with nothing to start."""
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(
        _complete({"tasks": []}),
        "```python\nCOMPLETE('{\"tasks\": [\"say \"hi\" in a.py\"]}')\n```",
        _complete({"tasks": ["fix a.py"]})))
    out = dispatch.run_dispatch("q")
    assert out["ok"] is True and out["tasks"] == ["fix a.py"]
    assert out["refused_completes"] == 2


# == more entries than the instructions allow ==================
def test_one_entry_over_the_count_is_read_not_refused_as_broken_json(monkeypatch):
    """A split holding one entry too many, each within its length, is parsed as JSON
    and keeps the first entries up to the count."""
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 4)
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 1000)
    out = dispatch.parse_dispatch(json.dumps({"tasks": ["x" * 900] * 5}))
    assert out["ok"] is True and len(out["tasks"]) == 4


def test_a_dropped_entry_is_said_in_the_log(monkeypatch, caplog):
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 2)
    with caplog.at_level("WARNING", logger="codetwine_loom.dispatch"):
        dispatch.parse_dispatch('{"tasks": ["a", "b", "c"]}')
    assert "holds 3 entries and at most 2" in caplog.text


def test_a_cut_entry_is_said_in_the_log(monkeypatch, caplog):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 10)
    with caplog.at_level("WARNING", logger="codetwine_loom.dispatch"):
        dispatch.parse_dispatch(json.dumps({"tasks": ["short", "x" * 25]}))
    assert "entry 2 is 25 characters and at most 10" in caplog.text
    assert "entry 1" not in caplog.text


def test_an_entry_within_the_limit_says_nothing(monkeypatch, caplog):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 10)
    with caplog.at_level("WARNING", logger="codetwine_loom.dispatch"):
        dispatch.parse_dispatch(json.dumps({"tasks": ["x" * 10]}))
    assert caplog.text == ""


def test_too_many_entries_are_sent_back_once(stub_search, monkeypatch):
    """The model puts them together itself, as it shortens an entry itself."""
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 2)
    prompts = []
    replies = iter([_complete({"tasks": ["a", "b", "c"]}),
                    _complete({"tasks": ["a and c", "b"]})])

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    out = dispatch.run_dispatch("q")
    assert out["tasks"] == ["a and c", "b"]
    assert out["refused_completes"] == 1 and out["iters"] == 2
    assert '"tasks" holds 3 entries and at most 2' in prompts[1]


def test_still_too_many_after_being_sent_back_is_cut_to_the_count(stub_search,
                                                                  monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 2)
    monkeypatch.setattr(dispatch, "make_llm",
                        lambda *a, **k: _scripted_llm(_complete({"tasks": ["a", "b", "c"]})))
    out = dispatch.run_dispatch("q")
    assert out["refused_completes"] == 1 and out["tasks"] == ["a", "b"]


def test_the_count_and_the_length_are_said_in_one_refusal(monkeypatch):
    """The guard refuses once, so everything wrong with the split goes into that one."""
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 1)
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 5)
    text = dispatch._over_the_limit(json.dumps({"questions": ["short", "y" * 9]}))
    assert '"questions" holds 2 entries and at most 1' in text
    assert "question 2 (9 characters)" in text and "over the 5 characters" in text


@pytest.mark.parametrize("entry", ["  " * 9, "  short  " + " " * 9,
                                   "short" + textcut.CUT_NOTE.format(omitted=4)])
def test_the_length_is_measured_on_what_the_parser_hands_out(monkeypatch, entry):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 5)
    assert dispatch._over_the_limit(json.dumps({"tasks": ["fix a", entry]})) == ""


@pytest.mark.parametrize("extra", [None, "", "  ", {"note": "x"}, 7, ["x"]])
def test_an_entry_that_is_no_text_is_not_counted_against_the_limit(monkeypatch, extra):
    """The guard counts what the parser hands out."""
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 2)
    answer = json.dumps({"tasks": ["fix a in a.py", "fix b in b.py", extra]})
    assert dispatch._over_the_limit(answer) == ""
    assert dispatch.parse_dispatch(answer)["tasks"] == ["fix a in a.py", "fix b in b.py"]
    over = json.dumps({"tasks": ["fix a in a.py", extra, "fix b in b.py", "fix c in c.py"]})
    assert '"tasks" holds 3 entries and at most 2' in dispatch._over_the_limit(over)


# == A split that reads as JSON but has another shape =======
@pytest.mark.parametrize("answer, error", [
    ('["fix a in a.py", "fix b in b.py"]', "not a JSON object"),
    ('{"tasks": []}', "no entry under"),
    ('{"tasks": "fix a in a.py"}', "no entry under"),
    ('{"tasks": ["..."]}', 'no entry written out under "tasks"'),
    ('{"tasks": ["…", " - "]}', 'no entry written out under "tasks"'),
    ('{"tasks": ["<one whole change request>", "another"]}',
     'no entry written out under "tasks"'),
    ('{"unclear": ["<one thing to settle, naming the readings it is between>"]}',
     'no entry written out under "unclear"'),
    ('{"task": ["fix a in a.py"]}', "no entry under"),
    ('{"questions": ["q"], "tasks": ["t"]}', "exactly one of"),
    ('{"tasks": ["t"], "unclear": ["u"]}', "exactly one of"),
])
def test_a_split_of_another_shape_is_sent_back_with_what_the_parser_says(answer, error):
    text = dispatch._over_the_limit(answer)
    parse_error = dispatch.parse_dispatch(answer)["error"]
    assert error in parse_error
    assert text.startswith("COMPLETE() was not accepted.") and f"- {parse_error}" in text
    assert ("write out each entry" in text) == ("no entry written out" in error)


def test_a_split_with_nothing_written_out_is_sent_back_with_what_to_do():
    error = dispatch.parse_dispatch('{"tasks": ["..."]}')["error"]
    assert error == ('the split holds no entry written out under "tasks", only marks such '
                     'as "..." or the placeholders the instructions show')
    text = dispatch._over_the_limit('{"tasks": ["..."]}')
    assert f"\n- {error}\n- write out each entry in full in place of" in text


def test_an_unclear_example_out_of_turns_is_no_entry_under_the_lists_taken():
    answer = '{"unclear": ["<one thing to settle, naming the readings it is between>"]}'
    error = dispatch.parse_dispatch(answer, is_late_answer=True)["error"]
    assert "no entry under" in error and "unclear" not in error


def test_a_hollow_list_beside_a_list_with_entries_is_passed_over():
    answer = '{"tasks": ["fix a in a.py"], "questions": ["<another>"]}'
    assert dispatch.parse_dispatch(answer)["tasks"] == ["fix a in a.py"]
    assert dispatch._over_the_limit(answer) == ""
    assert dispatch._over_the_limit('{"tasks": ["fix a in a.py", "..."]}') == ""
    text = dispatch._over_the_limit(json.dumps({
        "questions": ["<one whole question>", "<another>"],
        "tasks": ["<one whole change request>", "<another>"]}))
    assert 'under exactly one of "questions", "tasks"' in text
    assert dispatch.parse_dispatch('{"questions": ["<another>"], "tasks": ["..."]}')[
        "error"].startswith('the split holds no entry written out under "questions" and "tasks"')


def test_an_entry_wrapped_in_angle_brackets_is_handed_out_without_them():
    out = dispatch.parse_dispatch(json.dumps({"tasks": [
        "<Add a --verbose flag to src/cli.py>", "Make List<T> generic", "Wrap it in <pre>"]}))
    assert out["tasks"] == ["Add a --verbose flag to src/cli.py", "Make List<T> generic",
                            "Wrap it in <pre>"]


def test_the_shape_and_the_length_are_said_in_one_refusal(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 5)
    text = dispatch._over_the_limit(json.dumps({"questions": ["q"], "tasks": ["y" * 9]}))
    assert 'holds "questions" and "tasks"; it holds exactly one of' in text
    assert "task 1 (9 characters)" in text


def test_a_split_of_another_shape_is_sent_back_once_and_then_read(stub_search,
                                                                   monkeypatch):
    prompts = []
    replies = iter([_complete(["fix a in a.py", "fix b in b.py"]),
                    _complete({"tasks": ["fix a in a.py", "fix b in b.py"]})])

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    out = dispatch.run_dispatch("q")
    assert out["ok"] is True and out["tasks"] == ["fix a in a.py", "fix b in b.py"]
    assert out["refused_completes"] == 1 and out["iters"] == 2
    assert "the split is not a JSON object" in prompts[1]


def test_a_split_of_another_shape_twice_ends_the_run(stub_search, monkeypatch):
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(
        _complete({"questions": ["q"], "tasks": ["t"]})))
    out = dispatch.run_dispatch("q")
    assert out["ok"] is False and out["refused_completes"] == 1
    assert "exactly one of" in out["error"]


# == A request handed back for the person who asked to settle ==
def test_what_the_request_leaves_open_is_read_as_unclear():
    out = dispatch.parse_dispatch('{"unclear": ["JSON or YAML?"]}')
    assert out == {"ok": True, "unclear": ["JSON or YAML?"]}


def test_the_instructions_show_the_third_object_with_its_count(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_UNCLEAR", 2)
    text = dispatch.build_dispatch_instruction("py")
    assert '{"unclear": [' in text and "At most 2 entries" in text
    assert 'exactly one of\n"questions", "tasks" and "unclear"' in text


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_no_injection_is_left_unresolved_without_the_third_object(ext, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_UNCLEAR", 0)
    text = dispatch.build_dispatch_instruction(ext)
    assert not re.search(r"\{[a-z_]+\}", text)
    assert "unclear" not in text
    assert 'exactly one of\n"questions" and "tasks"' in text


def test_the_third_object_is_not_taken_when_it_is_switched_off(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_UNCLEAR", 0)
    out = dispatch.parse_dispatch('{"unclear": ["JSON or YAML?"]}')
    assert out["ok"] is False and '"questions" and "tasks"' in out["error"]


def test_more_open_points_than_the_setting_are_sent_back_once(stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_UNCLEAR", 3)
    prompts = []
    replies = iter([_complete({"unclear": ["a?", "b?", "c?", "d?"]}),
                    _complete({"unclear": ["a?", "b?"]})])

    def llm(prompt):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: llm)
    out = dispatch.run_dispatch("q")
    assert out["refused_completes"] == 1 and out["unclear"] == ["a?", "b?"]
    assert '"unclear" holds 4 entries and at most 3' in prompts[1]
    assert "defs" not in out


def test_still_too_many_open_points_are_cut_to_the_setting(stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_UNCLEAR", 3)
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(
        _complete({"unclear": ["a?", "b?", "c?", "d?"]})))
    out = dispatch.run_dispatch("q")
    assert out["refused_completes"] == 1 and out["unclear"] == ["a?", "b?", "c?"]


def test_an_open_point_over_the_length_is_named_in_the_refusal(monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 5)
    text = dispatch._over_the_limit(json.dumps({"unclear": ["y" * 9]}))
    assert "unclear entry 1 (9 characters)" in text


def test_the_call_after_the_last_turn_says_the_third_object_is_not_taken(stub_search,
                                                                         monkeypatch):
    assert '"unclear" is not taken now' in _harvest_prompt(monkeypatch)


def test_the_call_after_the_last_turn_is_silent_about_a_third_object_switched_off(
        stub_search, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_UNCLEAR", 0)
    assert "unclear" not in _harvest_prompt(monkeypatch)


def test_the_third_object_is_not_taken_from_the_call_after_the_last_turn(stub_search,
                                                                         monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_MAX_TURN", 1)
    monkeypatch.setattr(dispatch, "make_llm", lambda *a, **k: _scripted_llm(
        "```python\nprint('still reading')\n```",
        '{"unclear": ["JSON or YAML?"]}'))
    out = dispatch.run_dispatch("q")
    assert out["harvested"] is True and out["ok"] is False
    assert "unclear" not in out


def test_a_split_that_is_not_an_object_is_refused_when_read():
    out = dispatch.parse_dispatch('["a", "b"]')
    assert out["ok"] is False and "not a JSON object" in out["error"]
    assert out["raw"] == '["a", "b"]'


def test_the_split_is_told_the_plan_s_file_limit_and_no_turn_count(monkeypatch):
    """The split cannot tell how many turns a piece of work takes, so no count is
    given; the files an entry names it can count."""
    monkeypatch.setattr(config, "RLM_MAX_TURN", 17)
    monkeypatch.setattr(config, "PLAN_MAX_FILE", 5)
    text = dispatch.build_dispatch_instruction("py")
    assert "at most\n  5 files" in text
    assert "17" not in text
    assert "hand over one entry" not in text
    assert "one entry, not several" not in text


def test_the_split_is_told_how_entries_on_one_definition_run():
    text = dispatch.build_dispatch_instruction("py")
    assert "one after\n  another in the order you list them" in text
    assert "write the same definition\n  name in backticks and its file in both" in text


def test_the_page_read_reaches_it_only_when_it_is_switched_on(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 0)
    assert "web_fetch" not in dispatch.build_dispatch_instruction("py")
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 3)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    assert "web_fetch" not in dispatch.build_dispatch_instruction("py")
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    text = dispatch.build_dispatch_instruction("py")
    assert "web_fetch" in text and "3 calls" in text and "web_search" in text
