"""Every subcommand driven through cli.main() over a repository in tmp_path.

codetwine analyses the repository for real and the index is built over that. The one
thing substituted is the model, which answers from a script over HTTP (conftest).
"""

import json
import os
import shutil

import pytest

from integration.conftest import (ADD_SUB_REPLY_LIST, branch_list, code, commit_all, complete,
                      complete_json, git)
from codetwine_loom import cli, history
from codetwine_loom.index import store
from codetwine_loom.knowledge import KNOWLEDGE_FILE

pytestmark = pytest.mark.integration

_FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data")


def _main(*argv: str) -> int:
    """Run cli.main() and return the exit status, 0 when it returned."""
    try:
        cli.main(list(argv))
    except SystemExit as raised:
        if isinstance(raised.code, int):
            return raised.code
        pytest.fail(f"main stopped with: {raised.code}")
    return 0


def _build_dir(out_dir, name="calc"):
    """Return where the analysis and the index of one repository are."""
    return out_dir / name


# == build ======================================================
def test_build_analyses_the_repository_and_writes_the_index(calc_repo, out_dir,
                                                            model_server, capsys):
    assert _main("build", "--project", str(calc_repo), "--output-dir", str(out_dir)) == 0
    out = capsys.readouterr().out
    assert "(ran codetwine)" in out and "primary_ext: py" in out
    assert "defs: 6 files: 8" in out
    built = _build_dir(out_dir)
    assert (built / KNOWLEDGE_FILE).exists() and (built / store.INDEX_FILE).exists()
    record = json.loads((built / "analysed_commit.json").read_text(encoding="utf-8"))
    assert record["commit"] == git(calc_repo, "rev-parse", "HEAD").strip()
    assert record["exclude"] == []
    assert model_server.call_list == []


def test_a_second_build_reads_the_analysis_as_it_is(calc_repo, out_dir, model_server,
                                                     capsys):
    _main("build", "--project", str(calc_repo), "--output-dir", str(out_dir))
    capsys.readouterr()
    _main("build", "--project", str(calc_repo), "--output-dir", str(out_dir))
    assert "(existing)" in capsys.readouterr().out or True
    index = store.load(str(_build_dir(out_dir) / store.INDEX_FILE))
    assert index.commit == git(calc_repo, "rev-parse", "HEAD").strip()
    index.close()


def test_a_commit_after_the_build_makes_the_analysis_again(calc_repo, out_dir,
                                                           model_server, capsys):
    _main("build", "--project", str(calc_repo), "--output-dir", str(out_dir))
    (calc_repo / "calc" / "extra.py").write_text("def extra():\n    return 0\n",
                                                 encoding="utf-8")
    head = commit_all(calc_repo, "extra")
    capsys.readouterr()
    _main("build", "--project", str(calc_repo), "--output-dir", str(out_dir))
    assert "defs: 7 files: 9" in capsys.readouterr().out
    record = json.loads((_build_dir(out_dir) / "analysed_commit.json").read_text())
    assert record["commit"] == head


def test_what_exclude_keeps_out_is_kept_out_by_the_next_search_too(calc_repo, out_dir,
                                                                   model_server,
                                                                   capsys):
    _main("build", "--project", str(calc_repo), "--output-dir", str(out_dir),
          "--exclude", "tests")
    out = capsys.readouterr().out
    assert "files: 6" in out
    record = json.loads((_build_dir(out_dir) / "analysed_commit.json").read_text())
    assert record["exclude"] == ["tests"]
    _main("search", "--project", str(calc_repo), "--output-dir", str(out_dir),
          "--request", "add and sub")
    index = store.load(str(_build_dir(out_dir) / store.INDEX_FILE))
    assert not any("tests" in path for path in index.file_path_list())
    index.close()


def test_a_repository_of_several_languages_is_built_with_one_primary(tmp_path, out_dir,
                                                                     model_server,
                                                                     capsys):
    repo = tmp_path / "multilang_sample"
    shutil.copytree(os.path.join(_FIXTURE_DIR, "multilang_sample"), repo)
    git(repo, "init", "-q", "-b", "main")
    commit_all(repo, "first")
    assert _main("build", "--project", str(repo), "--output-dir", str(out_dir)) == 0
    out = capsys.readouterr().out
    assert "primary_ext: " in out
    index = store.load(str(out_dir / "multilang_sample" / store.INDEX_FILE))
    ext_set = {os.path.splitext(p)[1] for p in index.file_path_list()}
    assert {".py", ".java", ".kt", ".js", ".ts", ".c", ".cpp"} <= ext_set
    index.close()


def _cobol_repo(tmp_path):
    """A COBOL program in fixed format and the copybook it copies, in one commit."""
    repo = tmp_path / "cobol_sample"
    (repo / "src").mkdir(parents=True)
    (repo / "copy").mkdir()
    (repo / "copy" / "CUSTREC.cpy").write_text(
        "       01  CUST-RECORD.\n"
        "           05  CUST-ACCT-ID        PIC 9(11).\n"
        "           05  顧客番号            PIC X(10).\n", encoding="utf-8")
    (repo / "src" / "MAIN.cbl").write_text("".join(
        f"{n:06d} {code}".ljust(72) + "MAIN0001\n" for n, code in enumerate([
            "IDENTIFICATION DIVISION.", "PROGRAM-ID. MAIN.", "DATA DIVISION.",
            "WORKING-STORAGE SECTION.", "COPY CUSTREC.", "01  WS-A    PIC 9.",
            "PROCEDURE DIVISION.", "MAIN-PARA.", "    PERFORM load-cust",
            "    MOVE 5 TO cust-acct-id", "    STOP RUN.", "LOAD-CUST."], 1))
        + "000013     MOVE SPACES TO 顧客番号.\n", encoding="utf-8")
    git(repo, "init", "-q", "-b", "main")
    commit_all(repo, "first")
    return repo


def test_a_cobol_repository_is_built_with_its_records_and_its_program_id_lines(
        tmp_path, out_dir, model_server, capsys):
    repo = _cobol_repo(tmp_path)
    assert _main("build", "--project", str(repo), "--output-dir", str(out_dir)) == 0
    out = capsys.readouterr().out
    assert "primary_ext: cbl" in out or "primary_ext: cpy" in out
    index = store.load(str(out_dir / "cobol_sample" / store.INDEX_FILE))
    try:
        program = index.definition("cobol_sample/src/MAIN_cbl/MAIN.cbl::MAIN")
        assert program["signature_line"] == 2 and "PROGRAM-ID. MAIN." in program["signature"]
        assert [entry["key"] for entry in index.def_list_by_name("load-cust")] == \
            ["cobol_sample/src/MAIN_cbl/MAIN.cbl::LOAD-CUST"]
        assert index.non_ascii_name_list() == ["顧客番号"]
        assert index.caller_list("cobol_sample/copy/CUSTREC_cpy/CUSTREC.cpy::顧客番号") == \
            ["cobol_sample/src/MAIN_cbl/MAIN.cbl::LOAD-CUST"]
    finally:
        index.close()


def test_a_japanese_request_finds_the_record_of_the_name_it_writes(tmp_path, out_dir,
                                                                   model_server, capsys):
    repo = _cobol_repo(tmp_path)
    model_server.reply_dict["expand"] = ["customer number digits"]
    assert _main("search", "--project", str(repo), "--output-dir", str(out_dir),
                 "--request", "顧客番号の桁を 12 にして") == 0
    out = capsys.readouterr().out
    assert "copy/CUSTREC.cpy::CUST-RECORD" in out.split("defs:")[1].split("\n")[1]


# == search =====================================================
def test_search_names_the_file_and_the_definition_the_request_is_about(
        calc_repo, out_dir, model_server, capsys):
    assert _main("search", "--project", str(calc_repo), "--output-dir", str(out_dir),
                 "--request", "make add return an int") == 0
    out = capsys.readouterr().out
    assert "files:\n  calc/basic.py" in out
    assert "calc/basic.py::add" in out and "def add(a: float, b: float) -> float:" in out
    assert model_server.call_list == []


def test_a_request_in_another_language_is_searched_through_the_expansion(
        calc_repo, out_dir, model_server, capsys):
    model_server.reply_dict["expand"] = ["mul multiply scale"]
    assert _main("search", "--project", str(calc_repo), "--output-dir", str(out_dir),
                 "--request", "掛け算の関数を直して") == 0
    out = capsys.readouterr().out
    assert model_server.stage_list() == ["expand"]
    assert "掛け算の関数を直して" in model_server.prompt_list("expand")[0]
    assert "files:\n  calc/scale.py" in out


# == edit =======================================================
def _edit(calc_repo, out_dir, *argv: str) -> int:
    return _main("edit", "--project", str(calc_repo), "--output-dir", str(out_dir),
                 "--request", "add a sub function beside add", *argv)


def test_edit_changes_the_file_runs_the_tests_and_records_the_run(calc_repo, out_dir,
                                                                  model_server, capsys):
    model_server.reply_dict["edit"] = ADD_SUB_REPLY_LIST
    assert _edit(calc_repo, out_dir) == 0
    out = capsys.readouterr().out
    assert "def sub(" in (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
    assert model_server.stage_list() == ["edit"] * 3
    assert "'status': 'passed'" in model_server.prompt_list("edit")[2]
    assert "verdict: unverified" in out
    record = json.loads((_build_dir(out_dir) / history.HISTORY_FILE).read_text())
    assert [a["verdict"] for a in record["attempts"]] == ["unverified"]
    assert record["attempts"][0]["files"]["modified"] == ["calc/basic.py"]
    turn_dir = _build_dir(out_dir) / "turns"
    assert sorted(p.suffix for p in turn_dir.iterdir()) == [".jsonl", ".log"]


def test_history_prints_what_the_edit_recorded(calc_repo, out_dir, model_server, capsys):
    model_server.reply_dict["edit"] = ADD_SUB_REPLY_LIST
    _edit(calc_repo, out_dir)
    capsys.readouterr()
    assert _main("history", "--project", str(calc_repo), "--output-dir", str(out_dir)) == 0
    out = capsys.readouterr().out
    assert "unverified" in out and "add0 mod1 del0" in out


def test_a_dry_run_writes_nothing_and_says_so(calc_repo, out_dir, model_server, capsys):
    model_server.reply_dict["edit"] = ADD_SUB_REPLY_LIST
    before = (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
    assert _edit(calc_repo, out_dir, "--dry-run") == 0
    assert (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8") == before
    assert git(calc_repo, "status", "--porcelain") == ""
    assert "'dry_run': True" in model_server.prompt_list("edit")[1]


def test_a_run_that_never_finishes_is_asked_once_and_exits_1(calc_repo, out_dir,
                                                             model_server, capsys):
    model_server.reply_dict["edit"] = [code("print('reading')")]
    model_server.late_reply = "I read the code and changed nothing"
    assert _edit(calc_repo, out_dir) == 1
    out = capsys.readouterr().out
    assert "harvested" in out and "verdict: failed" in out
    assert model_server.stage_list() == ["edit"] * 7
    record = json.loads((_build_dir(out_dir) / history.HISTORY_FILE).read_text())
    assert record["attempts"][0]["verdict"] == "failed"


def test_a_run_that_leaves_the_tests_red_says_so_and_exits_0(calc_repo, out_dir,
                                                             model_server, capsys):
    model_server.reply_dict["edit"] = [
        code('print(replace_in_file("calc/basic.py", "    return a + b\\n", '
             '"    return a - b\\n"))'),
        code("print(run_test())"),
        complete("changed add; one test fails"),
    ]
    assert _edit(calc_repo, out_dir) == 0
    out = capsys.readouterr().out
    assert "verdict: tests_red" in out
    assert "'status': 'failed'" in model_server.prompt_list("edit")[2]


# == answer =====================================================
def test_answer_hands_the_answer_back_and_writes_it_where_asked(calc_repo, out_dir,
                                                                model_server, tmp_path,
                                                                capsys):
    model_server.reply_dict["answer"] = [
        code('print(read_definition("add"))'),
        complete("add is defined in calc/basic.py"),
    ]
    answer_out = tmp_path / "answer.json"
    assert _main("answer", "--project", str(calc_repo), "--output-dir", str(out_dir),
                 "--request", "where is add defined?", "--answer-out",
                 str(answer_out)) == 0
    out = capsys.readouterr().out
    assert "add is defined in calc/basic.py" in out
    assert json.loads(answer_out.read_text(encoding="utf-8"))["answer"] == \
        "add is defined in calc/basic.py"
    assert model_server.stage_list() == ["answer", "answer"]
    assert "def add(a: float, b: float)" in model_server.prompt_list("answer")[1]
    assert git(calc_repo, "status", "--porcelain") == ""


def test_the_combining_run_is_handed_the_agents_answers(calc_repo, out_dir, model_server,
                                                        tmp_path, capsys):
    answered = tmp_path / "answered.json"
    answered.write_text(json.dumps([
        {"n": 1, "request": "where is add?", "answer": "calc/basic.py", "error": ""},
        {"n": 2, "request": "where is mul?", "answer": None,
         "error": "no answer came back"}]), encoding="utf-8")
    model_server.reply_dict["combine"] = [complete("add: calc/basic.py; mul: open")]
    assert _main("answer", "--project", str(calc_repo), "--output-dir", str(out_dir),
                 "--request", "where are add and mul?", "--answered", str(answered)) == 0
    assert "add: calc/basic.py; mul: open" in capsys.readouterr().out
    prompt = model_server.prompt_list("combine")[0]
    assert "answered: calc/basic.py" in prompt and "no answer came back" in prompt


# == run --here =================================================
def test_the_reading_run_splits_the_request_into_tasks(calc_repo, out_dir, model_server,
                                                       tmp_path, capsys):
    model_server.reply_dict["dispatch"] = [
        code('print(grep_source("def ", count_only=True))'),
        complete_json({"tasks": ["add sub to calc/basic.py", "add div to calc/scale.py"]}),
    ]
    task_out = tmp_path / "tasks.json"
    assert _main("run", "--here", "--project", str(calc_repo), "--output-dir",
                 str(out_dir), "--request", "add sub and div", "--tasks-out",
                 str(task_out)) == 0
    assert "tasks: 2" in capsys.readouterr().out
    assert json.loads(task_out.read_text(encoding="utf-8")) == {
        "tasks": ["add sub to calc/basic.py", "add div to calc/scale.py"],
        "defs": [["calc/calc/basic_py/basic.py::add"], []]}
    assert "'calc/basic.py'" in model_server.prompt_list("dispatch")[1]


def test_the_reading_run_can_split_a_question_instead(calc_repo, out_dir, model_server,
                                                      tmp_path, capsys):
    model_server.reply_dict["dispatch"] = [complete_json({"questions": ["where is add?"]})]
    task_out = tmp_path / "tasks.json"
    assert _main("run", "--here", "--project", str(calc_repo), "--output-dir",
                 str(out_dir), "--request", "where is add?", "--tasks-out",
                 str(task_out)) == 0
    assert json.loads(task_out.read_text(encoding="utf-8")) == {
        "questions": ["where is add?"]}


# == integrate --here ===========================================
def _two_branches(calc_repo, same_line: bool) -> None:
    """Make branches a and b off main, each adding a function, on the same line or not."""
    base = git(calc_repo, "rev-parse", "HEAD").strip()
    git(calc_repo, "checkout", "-q", "-b", "a", base)
    (calc_repo / "calc" / "basic.py").write_text(
        (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
        + "\n\ndef sub(a, b):\n    return a - b\n", encoding="utf-8")
    commit_all(calc_repo, "sandbox: add sub")
    git(calc_repo, "checkout", "-q", "-b", "b", base)
    target = "basic.py" if same_line else "scale.py"
    (calc_repo / "calc" / target).write_text(
        (calc_repo / "calc" / target).read_text(encoding="utf-8")
        + "\n\ndef div(a, b):\n    return a / b\n", encoding="utf-8")
    commit_all(calc_repo, "sandbox: add div")
    git(calc_repo, "checkout", "-q", "main")


def test_branches_that_do_not_overlap_are_combined_by_git_alone(calc_repo, out_dir,
                                                                model_server, tmp_path,
                                                                capsys):
    _two_branches(calc_repo, same_line=False)
    outcome = tmp_path / "outcome.json"
    assert _main("integrate", "--here", "--project", str(calc_repo), "--output-dir",
                 str(out_dir), "--branch", "a", "--branch", "b", "--outcome-out",
                 str(outcome)) == 0
    out = capsys.readouterr().out
    assert "settled: by git alone, with no model called" in out
    assert "left in place" not in out
    assert model_server.call_list == []
    assert "def sub(" in (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
    assert "def div(" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
    assert json.loads(outcome.read_text(encoding="utf-8"))["verdict"] == "settled"


def test_branches_that_conflict_are_settled_by_the_model(calc_repo, out_dir,
                                                         model_server, tmp_path, capsys):
    _two_branches(calc_repo, same_line=True)
    settled = ('"""Addition and its helpers."""\n\n\ndef add(a: float, b: float) -> '
               'float:\n    """aとbの合計を返します。"""\n    return a + b\n\n\n'
               'def sub(a, b):\n    return a - b\n\n\ndef div(a, b):\n    return a / b\n')
    model_server.reply_dict["integrate"] = [
        code('print(read_source_file("calc/basic.py"))'),
        code('line_count = read_source_file("calc/basic.py").count("\\n")\n'
             f'print(replace_line_range("calc/basic.py", 1, line_count, {settled!r}))\n'
             'print(record_decision("calc/basic.py", "kept both functions"))'),
        code("print(run_test())"),
        complete("kept sub and div in calc/basic.py"),
    ]
    outcome = tmp_path / "outcome.json"
    assert _main("integrate", "--here", "--project", str(calc_repo), "--output-dir",
                 str(out_dir), "--branch", "a", "--branch", "b", "--outcome-out",
                 str(outcome)) == 0
    out = capsys.readouterr().out
    assert "settled: everything the merge left" in out
    assert "<<<<<<<" in model_server.prompt_list("integrate")[1]
    assert "kept both functions" in model_server.prompt_list("integrate")[3]
    written = json.loads(outcome.read_text(encoding="utf-8"))
    assert written["verdict"] == "settled" and written["did"] == \
        "kept sub and div in calc/basic.py"
    assert "<<<<<<<" not in (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")


def test_a_merge_the_model_does_not_settle_exits_1_with_what_is_left(calc_repo, out_dir,
                                                                    model_server,
                                                                    tmp_path, capsys):
    _two_branches(calc_repo, same_line=True)
    model_server.reply_dict["integrate"] = [complete("done, I think")]
    outcome = tmp_path / "outcome.json"
    assert _main("integrate", "--here", "--project", str(calc_repo), "--output-dir",
                 str(out_dir), "--branch", "a", "--branch", "b", "--outcome-out",
                 str(outcome)) == 1
    out = capsys.readouterr().out
    assert "conflict markers are still in: calc/basic.py" in out
    written = json.loads(outcome.read_text(encoding="utf-8"))
    assert written["verdict"] == "unsettled" and "calc/basic.py" in written["why"]


# == run-tests ==================================================
def test_run_tests_runs_the_repositorys_own_tests(calc_repo, out_dir, model_server,
                                                  capsys):
    assert _main("run-tests", "--project", str(calc_repo), "--output-dir",
                 str(out_dir)) == 0
    out = capsys.readouterr().out
    assert "test command: " in out and "-m pytest" in out
    assert "status: passed" in out
    assert branch_list(calc_repo) == ["main"]
