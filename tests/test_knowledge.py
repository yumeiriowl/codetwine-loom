"""Step 0: the reader the rest of the tool takes its analysis from.

The analysis is read one file at a time, so the index build walks the same reader
several times and the members a file has none of still have to be there.
"""

import json
import logging
import os
import sqlite3
import subprocess

import pytest
from codetwine.utils import file_utils as codetwine_file_utils

from conftest import git_repo, write_knowledge_db
from codetwine_loom import SettingError, gitcmd, knowledge, progress
from codetwine_loom.knowledge import KNOWLEDGE_FILE, Knowledge
from codetwine_loom.textcut import LOG_NAME_MAX_CHAR, name_list_text

_ONE_FILE = {
    "project_name": "proj",
    "files": [{
        "file": "proj/a_py/a.py",
        "file_dependencies": {
            "definitions": [{"name": "f", "type": "function_definition",
                             "start_line": 1, "end_line": 2, "context": "def f():"}],
            "callee_usages": [], "caller_usages": []},
        "doc": {"sections": [], "summary": "a"},
    }],
}


# What codetwine's process_all_files() returns for a run in which nothing failed
_RUN_DICT = {"file_count": 1, "dependency_fail_list": [], "doc_count": 0,
             "doc_fail_list": [], "detected_encoding_dict": {}}


@pytest.fixture
def one_file(knowledge_of) -> Knowledge:
    """The reader over a single analysed file."""
    return knowledge_of(_ONE_FILE)


def test_iter_files_can_be_walked_more_than_once(one_file):
    """The index build walks the files 4 times over one reader."""
    assert [f["file"] for f in one_file.iter_file()] == \
        [f["file"] for f in one_file.iter_file()]


def test_every_dependency_member_is_there_for_a_file_that_has_none(knowledge_of):
    """A file with no dependencies at all has no file_dependencies in the database."""
    pk = {"files": [{"file": "proj/a_py/a.py", "doc": {"summary": "a"}}]}
    entry = next(knowledge_of(pk).iter_file())
    assert entry["file_dependencies"] == {"definitions": [], "callee_usages": [],
                                          "same_file_usages": [], "caller_usages": []}


def test_a_file_without_a_design_document_carries_no_doc(knowledge_of):
    """file_doc() answers for the missing key; nothing is invented here."""
    pk = {"files": [{"file": "proj/a_py/a.py",
                     "file_dependencies": {"definitions": [], "callee_usages": [],
                                           "caller_usages": []}}]}
    assert "doc" not in next(knowledge_of(pk).iter_file())


def test_file_count_is_how_many_files_were_analysed(one_file, knowledge_of):
    assert one_file.file_count() == 1
    assert knowledge_of({"files": []}).file_count() == 0


def test_the_stamp_is_the_same_every_time_it_is_read(one_file):
    assert one_file.stamp() == one_file.stamp()
    assert one_file.stamp().startswith("1:")


def test_the_stamp_differs_between_two_runs_of_the_analysis(knowledge_of):
    """Writing the analysis again is what the index is warned about."""
    assert knowledge_of(_ONE_FILE).stamp() != knowledge_of(_ONE_FILE).stamp()


def test_opening_a_database_that_is_not_there_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        Knowledge.open(str(tmp_path / KNOWLEDGE_FILE))


@pytest.mark.parametrize("name", ["C#", "what?", "a%41b", "with space"])
def test_a_database_under_a_path_a_uri_reads_marks_in_is_opened_where_it_is(tmp_path,
                                                                           name):
    path = str(tmp_path / name / KNOWLEDGE_FILE)
    write_knowledge_db(_ONE_FILE, path, "proj")
    opened = Knowledge.open(path)
    assert opened.file_count() == 1
    with pytest.raises(sqlite3.OperationalError):
        opened._connection.execute("DELETE FROM files")
    opened.close()
    assert os.listdir(tmp_path) == [name]


# == the design document left in the output directory =======================
_LEFT_DOC = {"file": "proj/a.py", "summary": "earlier summary", "source_hash": "old",
             "sections": [{"id": "overview", "title": "Overview", "content": "zebra"}]}


def _no_doc_knowledge(out) -> str:
    """Write a database of one file without a design document; return its path."""
    entry = {**_ONE_FILE["files"][0]}
    del entry["doc"]
    path = str(out / "proj" / KNOWLEDGE_FILE)
    write_knowledge_db({"project_name": "proj", "files": [entry]}, path, "proj")
    return path


def _left_doc_of(path: str) -> dict | None:
    reader = Knowledge.open(path)
    try:
        return next(reader.iter_file()).get("doc")
    finally:
        reader.close()


def test_a_file_without_a_document_in_the_database_takes_the_one_left_beside_it(
        tmp_path):
    """A document written from an earlier content of the file comes with its hash."""
    out = tmp_path / "out"
    path = _no_doc_knowledge(out)
    (out / "proj" / "a_py").mkdir(parents=True)
    (out / "proj" / "a_py" / "doc.json").write_text(json.dumps(_LEFT_DOC),
                                                   encoding="utf-8")
    assert _left_doc_of(path) == _LEFT_DOC


def test_the_document_in_the_database_is_taken_before_the_one_left_beside_it(tmp_path):
    out = tmp_path / "out"
    path = str(out / "proj" / KNOWLEDGE_FILE)
    write_knowledge_db(_ONE_FILE, path, "proj")
    (out / "proj" / "a_py").mkdir(parents=True)
    (out / "proj" / "a_py" / "doc.json").write_text(json.dumps(_LEFT_DOC),
                                                   encoding="utf-8")
    assert _left_doc_of(path)["summary"] == "a"


@pytest.mark.parametrize("form", ["link", "not_a_document", "not_json", "none"])
def test_a_left_document_that_is_no_regular_design_document_is_not_taken(tmp_path,
                                                                        form):
    out = tmp_path / "out"
    path = _no_doc_knowledge(out)
    doc_dir = out / "proj" / "a_py"
    doc_dir.mkdir(parents=True)
    if form == "link":
        (tmp_path / "secret.json").write_text(json.dumps(_LEFT_DOC), encoding="utf-8")
        (doc_dir / "doc.json").symlink_to(tmp_path / "secret.json")
    elif form == "not_a_document":
        (doc_dir / "doc.json").write_text(json.dumps({"sections": [{"id": 1}]}),
                                          encoding="utf-8")
    elif form == "not_json":
        (doc_dir / "doc.json").write_text("{", encoding="utf-8")
    assert _left_doc_of(path) is None


def test_a_left_document_outside_the_output_directory_is_not_taken(tmp_path):
    out = tmp_path / "out"
    entry = {**_ONE_FILE["files"][0], "file": "../outside/a_py/a.py"}
    del entry["doc"]
    path = str(out / "proj" / KNOWLEDGE_FILE)
    write_knowledge_db({"project_name": "proj", "files": [entry]}, path, "proj")
    (tmp_path / "outside" / "a_py").mkdir(parents=True)
    (tmp_path / "outside" / "a_py" / "doc.json").write_text(json.dumps(_LEFT_DOC),
                                                            encoding="utf-8")
    assert _left_doc_of(path) is None


# == running the analysis ===================================================

def test_run_codetwine_asks_for_the_form_the_reader_takes(monkeypatch, tmp_path):
    """The setting is replaced in the pipeline's namespace, not in the environment."""
    seen = {}

    async def fake_process_all_files(project_dir, output_dir, client, max_workers,
                                     file_list=None):
        seen["format"] = knowledge.codetwine_pipeline.KNOWLEDGE_FORMAT
        seen["with_doc"] = knowledge.codetwine_pipeline.ENABLE_LLM_DOC
        return _RUN_DICT

    monkeypatch.setattr(knowledge.codetwine_pipeline, "process_all_files",
                        fake_process_all_files)
    knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=False)
    assert seen == {"format": "sqlite", "with_doc": False}


def test_codetwine_is_given_the_files_git_does_not_ignore(tmp_path):
    """The analysis is made by codetwine itself, without the documents."""
    repo = tmp_path / "proj"
    git_repo(repo)
    (repo / ".gitignore").write_text("local.py\n", encoding="utf-8")
    (repo / "local.py").write_text("def hidden():\n    return 1\n", encoding="utf-8")
    (repo / "new.py").write_text("def shown():\n    return 1\n", encoding="utf-8")
    out = tmp_path / "out"
    knowledge.run_codetwine(str(repo), str(out), with_doc=False)
    reader = Knowledge.open(str(out / "proj" / KNOWLEDGE_FILE))
    try:
        file_list = sorted(f["file"] for f in reader.iter_file())
    finally:
        reader.close()
    assert file_list == ["proj/.gitignore/.gitignore", "proj/a_py/a.py",
                         "proj/new_py/new.py"]


def test_a_symbolic_link_is_not_analysed_in_or_out_of_the_repository(tmp_path):
    """A link to a file outside and a link to a file inside, both tracked."""
    (tmp_path / "outside.py").write_text("API_TOKEN = 1\n", encoding="utf-8")
    repo = tmp_path / "proj"
    git_repo(repo)
    (repo / "out.py").symlink_to(tmp_path / "outside.py")
    (repo / "in.py").symlink_to("a.py")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    out = tmp_path / "out"
    knowledge.run_codetwine(str(repo), str(out), with_doc=False)
    reader = Knowledge.open(str(out / "proj" / KNOWLEDGE_FILE))
    try:
        file_list = sorted(f["file"] for f in reader.iter_file())
    finally:
        reader.close()
    assert file_list == ["proj/a_py/a.py"]


def test_a_directory_that_is_not_a_repository_is_walked_by_codetwine(monkeypatch,
                                                                    tmp_path):
    given = {}

    async def fake_process_all_files(*a, **k):
        given.update(k)
        return _RUN_DICT

    monkeypatch.setattr(knowledge.codetwine_pipeline, "process_all_files",
                        fake_process_all_files)
    knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=False)
    assert given["file_list"] is None


def test_a_listed_path_that_holds_no_file_and_no_directory_is_not_handed_to_codetwine(
        monkeypatch, tmp_path, caplog):
    """A tracked file a named pipe stands in for is kept out and named in one warning;
    a deleted file is kept out without a word."""
    project = tmp_path / "proj"
    git_repo(project)
    (project / "d").mkdir()
    (project / "d" / "x.py").write_text("X = 1\n")
    (project / "p.py").write_text("y = 1\n")
    subprocess.run(["git", "-C", str(project), "add", "-A"], check=True,
                   capture_output=True)
    (project / "p.py").unlink()
    os.mkfifo(project / "p.py")
    (project / "d" / "x.py").unlink()
    given = {}

    async def fake_process_all_files(*a, **k):
        given.update(k)
        return _RUN_DICT

    monkeypatch.setattr(knowledge.codetwine_pipeline, "process_all_files",
                        fake_process_all_files)
    with caplog.at_level("DEBUG", logger=knowledge.logger.name):
        knowledge.run_codetwine(str(project), str(tmp_path / "out"), with_doc=False)
    assert given["file_list"] == ["a.py"]
    assert [record.getMessage() for record in caplog.records
            if record.levelname == "WARNING"] == [
        f"1 path(s) of {project} cannot be read as files and are not in the analysis: "
        f"p.py"]
    assert f"the paths of {project} that cannot be read as files: p.py" in caplog.text


def test_the_names_of_one_log_line_stop_at_the_characters_it_holds():
    assert name_list_text(["a.py", "b.py"], 100) == "a.py, b.py"
    assert name_list_text(["a.py", "b.py", "c.py"], 10) == "a.py, b.py and 1 more"
    assert name_list_text(["a" * 50, "b.py"], 10) == "a" * 50 + " and 1 more"
    assert name_list_text([], 10) == ""
    text = name_list_text([f"s/f{number:05d}_{'x' * 56}.txt" for number in range(10000)])
    assert len(text) <= LOG_NAME_MAX_CHAR + len(" and 10000 more")
    assert text.endswith(f" and {10000 - text.count(', ') - 1} more")


@pytest.mark.parametrize("level, printed", [("INFO", ""), ("DEBUG", "Analyzing...\n")])
def test_what_codetwine_prints_without_the_documents_is_held_back_below_debug(
        monkeypatch, tmp_path, capsys, level, printed):
    async def fake_process_all_files(*a, **k):
        print("Analyzing...")
        return _RUN_DICT

    monkeypatch.setattr(knowledge.codetwine_pipeline, "process_all_files",
                        fake_process_all_files)
    monkeypatch.setattr(knowledge.config, "AGENT_LOG_LEVEL", level)
    knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=False)
    assert capsys.readouterr().out == printed


# == what a run of codetwine could not do ===================================
def test_a_file_codetwine_could_not_extract_is_named(caplog):
    run_dict = dict(_RUN_DICT, file_count=3, dependency_fail_list=["a.py", "b/c.py"])
    with caplog.at_level("WARNING", logger=knowledge.logger.name):
        knowledge.report_run(run_dict, with_doc=False)
    assert "2 of 3 file(s)" in caplog.text
    assert "a.py, b/c.py" in caplog.text


def test_a_file_whose_encoding_codetwine_guessed_is_named_with_source_encoding(caplog):
    run_dict = dict(_RUN_DICT, detected_encoding_dict={"a.py": "big5hkscs", "b.c": ""})
    with caplog.at_level("WARNING", logger=knowledge.logger.name):
        knowledge.report_run(run_dict, with_doc=False)
    assert "2 file(s)" in caplog.text
    assert "a.py (big5hkscs)" in caplog.text
    assert "b.c (none, read as UTF-8 with replacement)" in caplog.text
    assert "SOURCE_ENCODING" in caplog.text


def test_a_short_cp932_file_is_named_by_a_real_run(tmp_path, monkeypatch, caplog):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.py").write_bytes("# 設定\nX = 1\n".encode("cp932"))
    (project / "b.py").write_text("Y = 2\n", encoding="utf-8")
    monkeypatch.setattr(codetwine_file_utils, "SOURCE_ENCODING", [])
    with caplog.at_level("WARNING", logger=knowledge.logger.name):
        knowledge.run_codetwine(str(project), str(tmp_path / "out"), with_doc=False)
    warning_list = [r.getMessage() for r in caplog.records
                    if r.name == knowledge.logger.name and "guessed" in r.getMessage()]
    assert len(warning_list) == 1
    assert "1 file(s)" in warning_list[0] and "a.py (" in warning_list[0]
    assert "b.py" not in warning_list[0]


def test_a_run_in_which_nothing_failed_warns_about_nothing(caplog):
    with caplog.at_level("INFO", logger=knowledge.logger.name):
        knowledge.report_run(dict(_RUN_DICT, doc_count=1), with_doc=True)
    assert [r.levelname for r in caplog.records] == ["INFO"]
    assert "docs: 1 ok, 0 failed" in caplog.text


def test_a_document_that_failed_is_named_and_the_run_goes_on(caplog):
    run_dict = dict(_RUN_DICT, doc_count=2, doc_fail_list=["a.py"])
    with caplog.at_level("INFO", logger=knowledge.logger.name):
        knowledge.report_run(run_dict, with_doc=True)
    assert "docs: 2 ok, 1 failed" in caplog.text
    assert "no complete design document for: a.py" in caplog.text


def test_a_run_that_wrote_no_document_at_all_is_refused():
    run_dict = dict(_RUN_DICT, doc_count=0, doc_fail_list=["a.py", "b.py"])
    with pytest.raises(RuntimeError, match="every one of the 2 design document"):
        knowledge.report_run(run_dict, with_doc=True)


def test_the_documents_are_not_counted_when_they_were_not_asked_for(caplog):
    with caplog.at_level("INFO", logger=knowledge.logger.name):
        knowledge.report_run(_RUN_DICT, with_doc=False)
    assert caplog.text == ""


def _never(_prompt):
    """An input() a test says must not be reached."""
    raise AssertionError("the question was asked")


@pytest.fixture
def repo_and_analysis(tmp_path, monkeypatch):
    """A git repository with no analysis, and a run_codetwine that writes one.

    Returns:
        (repo, out, calls): calls collects the docs argument of every run_codetwine.
    """
    repo = tmp_path / "proj"
    git_repo(repo)
    out = tmp_path / "out"
    calls = []

    def fake_run_codetwine(project_dir, output_dir, with_doc=True, exclude_list=None):
        calls.append(with_doc)
        write_knowledge_db(_ONE_FILE, str(out / "proj" / KNOWLEDGE_FILE), "proj")

    monkeypatch.setattr(knowledge, "run_codetwine", fake_run_codetwine)
    return repo, out, calls


def _answers(monkeypatch, *answers: str) -> None:
    """Make input() return the answers in turn, and a terminal to ask on.

    A question asked past the answers given fails the test, so a test that gives one
    answer also says that only one question is asked.
    """
    given = iter(answers)
    monkeypatch.setattr(progress, "at_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: next(given))


def test_a_missing_database_is_made_without_asking(repo_and_analysis, monkeypatch):
    """Every run reads the analysis and without the documents it costs nothing, so
    nothing here asks: input() is never reached, with or without a terminal."""
    repo, out, calls = repo_and_analysis
    monkeypatch.setattr(progress, "at_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", _never)
    path, _, project_name, reused = knowledge.ensure_knowledge(str(repo), str(out))
    assert (project_name, reused) == ("proj", False)
    assert path == str(out / "proj" / KNOWLEDGE_FILE)
    assert calls == [False]


def test_the_documents_come_from_the_caller(repo_and_analysis, monkeypatch):
    repo, out, calls = repo_and_analysis
    monkeypatch.setattr("builtins.input", _never)
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), with_doc=True)
    assert reused is False and calls == [True]


# == what the documents need is checked before codetwine runs ================
def test_the_documents_need_a_template_before_codetwine_runs(tmp_path, monkeypatch):
    """A missing template comes up inside codetwine as a FileNotFoundError after the
    extraction; what the documents need is read first, and said as one line."""
    monkeypatch.setattr(knowledge.codetwine_pipeline, "process_all_files",
                        lambda *a, **k: pytest.fail("codetwine was run"))
    monkeypatch.setattr(knowledge.codetwine_setting, "DOC_TEMPLATE_PATH",
                        str(tmp_path / "no_template.json"))
    monkeypatch.setenv("DOC_TEMPLATE_PATH", str(tmp_path / "no_template.json"))
    with pytest.raises(RuntimeError, match="DOC_TEMPLATE_PATH"):
        knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=True)


def test_a_code_summary_setting_codetwine_refuses_stops_the_documents(tmp_path,
                                                                    monkeypatch):
    template = tmp_path / "template.json"
    template.write_text('{"sections": [], "summary_prompt": ""}', encoding="utf-8")
    monkeypatch.setattr(knowledge.codetwine_setting, "DOC_TEMPLATE_PATH", str(template))
    monkeypatch.setenv("DOC_TEMPLATE_PATH", str(template))
    monkeypatch.setattr(knowledge.codetwine_doc, "ENABLE_CODE_SUMMARY", True)
    monkeypatch.setattr(knowledge.codetwine_doc, "CODE_SUMMARY_PIECE_LINES", 0)
    with pytest.raises(RuntimeError, match="CODE_SUMMARY_PIECE_LINES must be 1 or more"):
        knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=True)
    monkeypatch.setattr(knowledge.codetwine_doc, "CODE_SUMMARY_PIECE_LINES", 200)
    knowledge.check_doc_setting()


def test_the_documents_are_written_by_the_model_this_tool_names(tmp_path, monkeypatch):
    """codetwine's own LLM_MODEL and DOC_EFFORT are not read: the client is given
    DOC_CREATE_MODEL and DOC_CREATE_EFFORT."""
    monkeypatch.setattr(knowledge, "check_doc_setting", lambda: None)
    monkeypatch.setattr(knowledge.config, "DOC_CREATE_MODEL", "provider/doc-model")
    monkeypatch.setattr(knowledge.config, "LLM_API_KEY", "key-for-the-documents")
    monkeypatch.setattr(knowledge.config, "DOC_CREATE_EFFORT", "medium")
    given = {}
    monkeypatch.setattr(knowledge, "LLMClient",
                        lambda **kw: given.update(kw) or "client")
    ran = []
    monkeypatch.setattr(knowledge.codetwine_pipeline, "process_all_files",
                        lambda *a, **k: ran.append(a) or _RUN_DICT)
    monkeypatch.setattr(knowledge.asyncio, "run", lambda coroutine: coroutine)
    knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=True)
    assert given["model"] == "provider/doc-model"
    assert given["effort"] == "medium"
    assert given["api_key"] == "key-for-the-documents"
    assert ran[0][2] == "client"


def test_the_documents_are_not_asked_for_while_no_model_call_may_be_made(tmp_path,
                                                                         monkeypatch):
    monkeypatch.setattr(knowledge, "check_doc_setting", lambda: None)
    monkeypatch.setattr(knowledge.codetwine_pipeline, "process_all_files",
                        lambda *a, **k: pytest.fail("codetwine was run"))
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")
    with pytest.raises(RuntimeError, match="^no call is made to a model: ALL_PROXY "):
        knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=True)


def test_an_analysis_without_the_documents_checks_nothing_about_them(tmp_path,
                                                                    monkeypatch):
    monkeypatch.setattr(knowledge, "check_doc_setting",
                        lambda: pytest.fail("the documents' settings were checked"))
    ran = []
    monkeypatch.setattr(knowledge.codetwine_pipeline, "process_all_files",
                        lambda *a, **k: ran.append(a) or _RUN_DICT)
    monkeypatch.setattr(knowledge.asyncio, "run", lambda coroutine: coroutine)
    knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=False)
    assert len(ran) == 1


# == the question dispatch asks about the documents =========================
def test_no_terminal_means_no_documents(monkeypatch):
    """With no terminal to ask on, the answer is no rather than a blocked input()."""
    monkeypatch.setattr(progress, "at_terminal", lambda: False)
    monkeypatch.setattr("builtins.input", _never)
    assert knowledge.ask_doc("out/proj/k.sqlite") is False


@pytest.mark.parametrize("answer, expected", [("y", True), ("", False)])
def test_a_path_asked_about_once_is_not_asked_about_again(monkeypatch, answer, expected):
    """The run asks before an image is built, and the analysis made after it takes that
    answer without asking again."""
    asked_list = []
    monkeypatch.setattr(progress, "at_terminal", lambda: True)
    monkeypatch.setattr(progress, "ask", lambda question: asked_list.append(question)
                        or answer)
    assert knowledge.ask_doc("out/proj/k.sqlite") is expected
    assert knowledge.ask_doc(os.path.abspath("out/proj/k.sqlite")) is expected
    assert len(asked_list) == 1


@pytest.mark.parametrize("answer, expected", [("y", True), ("yes", True), ("n", False),
                                              ("", False)])
def test_the_one_question_is_about_the_documents(monkeypatch, answer, expected):
    """Enter is no, as it is for build without --doc-create."""
    _answers(monkeypatch, answer)
    assert knowledge.ask_doc("out/proj/k.sqlite") is expected


def test_the_end_of_the_input_at_the_question_means_no_documents(monkeypatch):
    def no_more(_prompt):
        raise EOFError

    monkeypatch.setattr(progress, "at_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", no_more)
    assert knowledge.ask_doc("out/proj/k.sqlite") is False


def test_the_question_says_where_the_analysis_is_missing(monkeypatch, capsys):
    _answers(monkeypatch, "")
    knowledge.ask_doc("out/proj/k.sqlite")
    assert "out/proj/k.sqlite" in capsys.readouterr().out


def test_ensure_knowledge_reuses_the_database_that_is_already_there(tmp_path):
    repo = tmp_path / "proj"
    git_repo(repo)
    db = tmp_path / "out" / "proj" / KNOWLEDGE_FILE
    write_knowledge_db(_ONE_FILE, str(db), "proj")
    path, _, project_name, reused = knowledge.ensure_knowledge(
        str(repo), str(tmp_path / "out"))
    assert (path, project_name, reused) == (str(db), "proj", True)


# == the commit recorded beside the analysis ===============================
def _committed_repo(path):
    """Make a git repository with one commit and return its path."""
    git_repo(path)
    return path


def _record_path(out) -> str:
    return str(out / "proj" / knowledge.ANALYSIS_COMMIT_FILE)


def test_running_codetwine_records_the_commit_beside_the_analysis(tmp_path,
                                                                  monkeypatch):
    repo = _committed_repo(tmp_path / "proj")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: write_knowledge_db(_ONE_FILE, db_path, "proj"))
    path, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out))
    assert reused is False
    record = json.loads(open(_record_path(out), encoding="utf-8").read())
    assert record["commit"] == gitcmd.head_commit(str(repo))
    assert record["stamp"] == Knowledge.open(path).stamp()
    assert knowledge.analysis_commit(path, record["stamp"]) == record["commit"]


def test_an_analysis_of_a_changed_tree_records_no_commit(tmp_path, monkeypatch):
    repo = _committed_repo(tmp_path / "proj")
    (repo / "a.py").write_text("y = 2\n", encoding="utf-8")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: write_knowledge_db(_ONE_FILE, db_path, "proj"))
    path, _, _, _ = knowledge.ensure_knowledge(str(repo), str(out))
    assert knowledge.analysis_commit(path, Knowledge.open(path).stamp()) == ""


def test_reusing_the_analysis_leaves_the_record_alone(tmp_path, monkeypatch):
    repo = _committed_repo(tmp_path / "proj")
    out = tmp_path / "out"
    write_knowledge_db(_ONE_FILE, str(out / "proj" / KNOWLEDGE_FILE), "proj")
    monkeypatch.setattr(knowledge, "run_codetwine", lambda *a, **k: _never(""))
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out))
    assert reused is True
    assert not (out / "proj" / knowledge.ANALYSIS_COMMIT_FILE).exists()


@pytest.mark.parametrize("written", [None, "not json", '{"stamp": "other", '
                                                        '"commit": "abc"}', "[1]"])
def test_analysed_commit_says_nothing_without_a_record_of_this_analysis(tmp_path,
                                                                        written):
    (tmp_path / "proj").mkdir()
    if written is not None:
        (tmp_path / "proj" / knowledge.ANALYSIS_COMMIT_FILE).write_text(
            written, encoding="utf-8")
    assert knowledge.analysis_commit(str(tmp_path / "proj" / KNOWLEDGE_FILE),
                                     "1:now") == ""


@pytest.mark.parametrize("leave", ["named pipe", "directory", "deep", "large"])
def test_a_record_that_is_no_regular_json_file_is_no_record(tmp_path, caplog, leave):
    """A named pipe in the record's place is not waited on, and a file of more bytes
    than a record is read in is not opened as JSON."""
    (tmp_path / "proj").mkdir()
    record = tmp_path / "proj" / knowledge.ANALYSIS_COMMIT_FILE
    if leave == "named pipe":
        os.mkfifo(record)
    elif leave == "directory":
        record.mkdir()
    elif leave == "large":
        record_text = '{"stamp": "1:now", "commit": "c"}'
        record.write_text(record_text + " " * (knowledge.RECORD_MAX_BYTE - len(record_text)),
                          encoding="utf-8")
        assert knowledge.analysis_record(str(tmp_path / "proj" / KNOWLEDGE_FILE),
                                         "1:now") == {"stamp": "1:now", "commit": "c"}
        record.write_text(
            record_text + " " * (knowledge.RECORD_MAX_BYTE - len(record_text) + 1),
            encoding="utf-8")
    else:
        record.write_text("[" * 200000, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert knowledge.analysis_record(str(tmp_path / "proj" / KNOWLEDGE_FILE),
                                         "1:now") == {}
    assert [entry.levelname for entry in caplog.records] == ["WARNING"]
    assert str(record) in caplog.text
    if leave == "large":
        assert (f"is {knowledge.RECORD_MAX_BYTE + 1} bytes, over the "
                f"{knowledge.RECORD_MAX_BYTE} read of one file") in caplog.text


# == what the record beside the analysis does when it cannot be written ======
def test_a_record_that_cannot_be_written_is_said_in_the_log(tmp_path, monkeypatch, caplog):
    path = str(tmp_path / "out" / "proj" / KNOWLEDGE_FILE)
    write_knowledge_db(_ONE_FILE, path, "proj")

    def refuse(record_path):
        raise OSError("read-only file system")

    monkeypatch.setattr(knowledge.tmpfile, "atomic_write", refuse)
    knowledge.record_analysis_commit(path, "abc123", "abc123")
    assert "could not be recorded" in caplog.text


# == what ensure_knowledge refuses before codetwine runs =====================
def test_a_target_that_is_not_a_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no such target repository"):
        knowledge.ensure_knowledge(str(tmp_path / "nowhere"), str(tmp_path / "out"))


def test_a_target_that_is_not_a_git_repository_is_refused(tmp_path):
    (tmp_path / "proj").mkdir()
    with pytest.raises(RuntimeError, match="Run `git -C .* init` first"):
        knowledge.ensure_knowledge(str(tmp_path / "proj"), str(tmp_path / "out"))


def test_a_database_still_missing_after_codetwine_ran_is_an_error(tmp_path, monkeypatch):
    git_repo(tmp_path / "proj")
    monkeypatch.setattr(knowledge, "run_codetwine", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="still missing"):
        knowledge.ensure_knowledge(str(tmp_path / "proj"), str(tmp_path / "out"))


# == the names kept out of the analysis =====================================
@pytest.fixture
def one_own_name(monkeypatch):
    """Make codetwine's own list one name, and let run_codetwine run over nothing."""
    monkeypatch.setattr(knowledge, "_BASE_EXCLUDE_TUPLE", ("__pycache__",))

    async def run_nothing(*a, **k):
        return _RUN_DICT

    monkeypatch.setattr(knowledge.codetwine_pipeline, "process_all_files", run_nothing)


def test_the_extra_names_are_added_to_codetwines_own_list_without_repeating_it(
        tmp_path, one_own_name):
    knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=False,
                            exclude_list=["vendor", "__pycache__"])
    assert knowledge.codetwine_walk.EXCLUDE_PATTERNS == ["__pycache__", "vendor"]


def test_a_name_given_more_than_once_is_added_once(tmp_path, one_own_name):
    knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=False,
                            exclude_list=["vendor", "vendor", "__pycache__"])
    assert knowledge.codetwine_walk.EXCLUDE_PATTERNS == ["__pycache__", "vendor"]


def test_the_names_of_one_call_are_not_still_there_for_the_next(tmp_path, one_own_name):
    """The list is written whole every time, so what one run keeps out is its own."""
    knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=False,
                            exclude_list=["vendor"])
    knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=False,
                            exclude_list=["build"])
    assert knowledge.codetwine_walk.EXCLUDE_PATTERNS == ["__pycache__", "build"]


@pytest.mark.parametrize("text, entry", [
    ("vendor", "vendor"), ("vendor/", "vendor"), ("./vendor", "./vendor"),
    ("apps/old/", "apps/old"), ("./apps//old", "apps/old"), ("apps/*/gen", "apps/*/gen"),
    ("./apps/old", "apps/old")])
def test_an_entry_is_kept_in_one_form(text, entry):
    assert knowledge.exclude_entry(text) == entry
    # The form is the one a container is handed and reads again
    assert knowledge.exclude_entry(entry) == entry


@pytest.mark.parametrize("text", ["", "/", "./", "/abs", "../up", "a/../b"])
def test_an_entry_that_is_no_name_or_path_inside_the_repository_is_refused(text):
    with pytest.raises(ValueError, match="not a name or a path inside the repository"):
        knowledge.exclude_entry(text)


@pytest.mark.parametrize("file_rel, entry_list, is_out", [
    ("apps/old/a.py", ["apps/old"], True), ("apps/old", ["apps/old"], True),
    ("apps/older/a.py", ["apps/old"], False), ("lib/apps/old/a.py", ["apps/old"], False),
    ("vendor/a.py", ["./vendor"], True), ("lib/vendor/a.py", ["./vendor"], False),
    ("apps/x/gen/a.py", ["apps/*/gen"], True), ("apps/x/src/a.py", ["apps/*/gen"], False),
    ("docs/a.md", ["apps/old", "docs/*.md"], True), ("a.py", [], False)])
def test_a_path_entry_keeps_out_the_path_and_what_is_under_it(file_rel, entry_list,
                                                               is_out):
    assert knowledge.is_out_by_path(file_rel, entry_list) is is_out


def _files_handed_to_codetwine(monkeypatch) -> list:
    """Stand in for codetwine and return the list its file_list arguments land in."""
    seen = []

    async def run_nothing(*a, file_list=None, **k):
        seen.append(file_list)
        return _RUN_DICT

    monkeypatch.setattr(knowledge.codetwine_pipeline, "process_all_files", run_nothing)
    return seen


def test_the_files_a_path_keeps_out_are_not_handed_to_codetwine(tmp_path, one_own_name,
                                                                monkeypatch):
    seen = _files_handed_to_codetwine(monkeypatch)
    rel_list = ["a.py", "apps/old/b.py", "apps/new/c.py", "lib/vendor/d.py", "vendor/e.py"]
    for rel in rel_list:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(knowledge.gitcmd, "work_tree_file_list",
                        lambda project_dir: rel_list)
    knowledge.run_codetwine(str(tmp_path), str(tmp_path / "out"), with_doc=False,
                            exclude_list=["apps/old", "./vendor", "logs"])
    assert seen == [["a.py", "apps/new/c.py", "lib/vendor/d.py"]]
    # codetwine is handed the names alone
    assert knowledge.codetwine_walk.EXCLUDE_PATTERNS == ["__pycache__", "logs"]


def test_a_directory_that_is_no_repository_is_walked_here_when_a_path_is_kept_out(
        tmp_path, one_own_name, monkeypatch):
    seen = _files_handed_to_codetwine(monkeypatch)
    project = tmp_path / "proj"
    (project / "apps" / "old").mkdir(parents=True)
    (project / "apps" / "old" / "b.py").write_text("x = 1\n", encoding="utf-8")
    (project / "a.py").write_text("x = 1\n", encoding="utf-8")
    knowledge.run_codetwine(str(project), str(tmp_path / "out"), with_doc=False,
                            exclude_list=["apps/old"])
    knowledge.run_codetwine(str(project), str(tmp_path / "out"), with_doc=False,
                            exclude_list=["old"])
    # With names alone codetwine walks the directory itself
    assert seen == [["a.py"], None]


# == the name the analysis is filed under ===================================
def test_the_name_is_the_same_however_the_repository_is_named(tmp_path, monkeypatch):
    """--project . and --project <path> analyse into the same output directory."""
    repo = tmp_path / "proj"
    repo.mkdir()
    monkeypatch.chdir(repo)
    assert knowledge.project_name_of(".") == "proj"
    assert knowledge.project_name_of(str(repo) + "/") == "proj"
    assert knowledge.project_name_of("..") == tmp_path.name
    assert knowledge.project_name_of(str(repo)) == "proj"


def test_the_filesystem_root_has_no_name_to_file_the_analysis_under():
    with pytest.raises(ValueError, match="no directory name"):
        knowledge.project_name_of("/")


# == what ensure_knowledge refuses about the output directory ================
def test_an_output_directory_inside_the_repository_is_refused(tmp_path, monkeypatch):
    repo = tmp_path / "proj"
    git_repo(repo)
    monkeypatch.setattr(knowledge, "run_codetwine", lambda *a, **k: _never(""))
    with pytest.raises(RuntimeError, match="OUTPUT_DIR is inside PROJECT_DIR"):
        knowledge.ensure_knowledge(str(repo), str(repo / "out"))


@pytest.mark.parametrize("level", [0, 1, 2])
def test_an_output_directory_holding_the_repository_is_refused(tmp_path, monkeypatch, level):
    """The parent would put the analysis inside the repository itself, and any directory
    above it would make every path in the repository output."""
    repo = tmp_path / "a" / "b" / "proj"
    repo.parent.mkdir(parents=True)
    git_repo(repo)
    monkeypatch.setattr(knowledge, "run_codetwine", lambda *a, **k: _never(""))
    with pytest.raises(RuntimeError, match="OUTPUT_DIR holds PROJECT_DIR"):
        knowledge.ensure_knowledge(str(repo), str(repo.parents[level]))


@pytest.mark.parametrize("project, output_dir, message", [
    ("proj", "proj", "is inside PROJECT_DIR"), ("proj", "proj/a/out", "is inside"),
    ("a/proj", "a", "holds PROJECT_DIR"), ("a/proj", ".", "holds PROJECT_DIR")])
def test_the_place_of_the_output_directory_is_checked_on_its_own(tmp_path, project,
                                                               output_dir, message):
    """Neither directory has to be there, and nothing is made."""
    with pytest.raises(RuntimeError, match=message):
        knowledge.check_output_place(str(tmp_path / project), str(tmp_path / output_dir))
    knowledge.check_output_place(str(tmp_path / project), str(tmp_path / "out"))
    assert list(tmp_path.iterdir()) == []


def test_a_repository_named_as_the_directory_of_the_agents_is_refused(tmp_path):
    """Its analysis would be written where the agents' own copies are kept."""
    with pytest.raises(RuntimeError) as raised:
        knowledge.check_output_place(str(tmp_path / "agents"), str(tmp_path / "out"))
    text = str(raised.value)
    assert "is named agents" in text and str(tmp_path / "out" / "agents") in text
    assert "Rename the repository's directory" in text and "copy the repository" in text
    knowledge.check_output_place(str(tmp_path / "agents" / "inner"), str(tmp_path / "out"))
    knowledge.check_output_place(str(tmp_path / "agents2"), str(tmp_path / "out"))
    assert list(tmp_path.iterdir()) == []


def test_an_output_directory_reached_through_a_link_is_placed_by_where_it_leads(tmp_path):
    (tmp_path / "proj" / "inside").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "proj" / "inside")
    with pytest.raises(RuntimeError, match="is inside PROJECT_DIR"):
        knowledge.check_output_place(str(tmp_path / "proj"), str(tmp_path / "link"))


def test_an_output_directory_that_is_a_file_is_refused(tmp_path, monkeypatch):
    repo = tmp_path / "proj"
    git_repo(repo)
    (tmp_path / "out").write_text("", encoding="utf-8")
    monkeypatch.setattr(knowledge, "run_codetwine", lambda *a, **k: _never(""))
    with pytest.raises(RuntimeError, match="cannot be made"):
        knowledge.ensure_knowledge(str(repo), str(tmp_path / "out"))


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes to any directory")
@pytest.mark.parametrize("output_dir, message", [
    ("locked/out", "cannot be made"), ("locked", "cannot be written to")])
def test_an_output_directory_that_cannot_be_written_to_is_refused(tmp_path, output_dir,
                                                                 message):
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked").chmod(0o555)
    try:
        with pytest.raises(RuntimeError, match=message):
            knowledge.make_output_dir(str(tmp_path / output_dir))
    finally:
        (tmp_path / "locked").chmod(0o755)


def test_an_output_directory_that_is_not_there_is_made(tmp_path):
    knowledge.make_output_dir(str(tmp_path / "a" / "out"))
    assert (tmp_path / "a" / "out").is_dir()


def test_a_directory_inside_a_repository_is_not_its_root(tmp_path):
    repo = tmp_path / "proj"
    git_repo(repo)
    (repo / "below").mkdir()
    knowledge.check_git_root(str(repo))
    for path in (repo / "below", tmp_path):
        with pytest.raises(RuntimeError, match="not the root of a git repository"):
            knowledge.check_git_root(str(path))


def test_an_output_directory_beside_the_repository_goes_on(tmp_path, monkeypatch):
    repo = tmp_path / "proj"
    git_repo(repo)
    db_path = str(tmp_path / "out" / "proj" / KNOWLEDGE_FILE)
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: write_knowledge_db(_ONE_FILE, db_path, "proj"))
    path, _, _, _ = knowledge.ensure_knowledge(str(repo), str(tmp_path / "out"))
    assert path == db_path


# == an analysis stopped while it was written ===============================
def _db_without_meta(path: str) -> None:
    """Write a database with the tables of an analysis and no meta rows in it."""
    write_knowledge_db(_ONE_FILE, path, "proj")
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM meta")


def test_an_analysis_with_no_meta_is_made_again(tmp_path, monkeypatch):
    """codetwine stopped while it wrote leaves the tables without the meta rows."""
    repo = tmp_path / "proj"
    git_repo(repo)
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    _db_without_meta(db_path)
    assert knowledge.is_complete(db_path) is False
    calls = []
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: calls.append(1) or write_knowledge_db(
                            _ONE_FILE, db_path, "proj"))
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out))
    assert (calls, reused) == ([1], False)
    assert knowledge.is_complete(db_path) is True


# == an analysis an older codetwine made =====================================
def _one_file_with_usage(usage: dict, member: str) -> dict:
    """Return _ONE_FILE with one usage in a member of its file."""
    file_dict = _ONE_FILE["files"][0]
    return {**_ONE_FILE, "files": [{
        **file_dict,
        "file_dependencies": {**file_dict["file_dependencies"], member: [usage]},
    }]}


@pytest.mark.parametrize("member", ["callee_usages", "same_file_usages"])
def test_an_analysis_whose_usages_name_no_definition_says_so(member, tmp_path):
    """Such an analysis is complete all the same: it is read, and iter_file() refuses it."""
    db_path = str(tmp_path / "out" / "proj" / KNOWLEDGE_FILE)
    usage = {"lines": [2], "name": "f", "from": "proj/a_py/a.py", "target_context": ""}
    write_knowledge_db(_one_file_with_usage(usage, member), db_path, "proj")
    assert knowledge.is_complete(db_path) is True
    old = Knowledge.open(db_path)
    assert old.names_target() is False
    old.close()


@pytest.mark.parametrize("pk", [
    _ONE_FILE,
    _one_file_with_usage({"lines": [2], "name": "f", "from": "proj/a_py/a.py",
                          "target_context": "", "target_name": None,
                          "target_start_line": None}, "callee_usages"),
])
def test_an_analysis_without_a_usage_or_with_a_usage_that_leads_nowhere_names_its_targets(
        pk, tmp_path):
    """A usage codetwine could tie to no definition carries the key with None."""
    db_path = str(tmp_path / "out" / "proj" / KNOWLEDGE_FILE)
    write_knowledge_db(pk, db_path, "proj")
    opened = Knowledge.open(db_path)
    assert opened.names_target() is True
    opened.close()


def test_an_analysis_whose_usages_name_no_definition_is_made_again_for_the_same_tree(
        tmp_path, monkeypatch):
    repo = _committed_repo(tmp_path / "proj")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    old_usage = {"lines": [2], "name": "f", "from": "proj/a_py/a.py",
                 "target_context": ""}
    pk_list = [_one_file_with_usage(old_usage, "callee_usages"),
               _one_file_with_usage({**old_usage, "target_name": "f",
                                     "target_start_line": 1}, "callee_usages")]
    call_list = []

    def write(*a, **k):
        write_knowledge_db(pk_list[len(call_list)], db_path, "proj")
        call_list.append(1)

    monkeypatch.setattr(knowledge, "run_codetwine", write)
    knowledge.ensure_knowledge(str(repo), str(out))
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert (len(call_list), reused) == (2, False)
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert (len(call_list), reused) == (2, True)


# == the encodings the analysis read its files with ==========================
def test_an_analysis_read_with_other_encodings_is_made_again_for_the_same_tree(
        tmp_path, monkeypatch):
    repo = _committed_repo(tmp_path / "proj")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    call_list = []
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: call_list.append(1) or write_knowledge_db(
                            _ONE_FILE, db_path, "proj"))
    monkeypatch.setattr(knowledge.config, "SOURCE_ENCODING", [])
    knowledge.ensure_knowledge(str(repo), str(out))
    record = json.loads(open(_record_path(out), encoding="utf-8").read())
    assert record["encoding"] == []
    monkeypatch.setattr(knowledge.config, "SOURCE_ENCODING", ["euc_jp", "cp932"])
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert (len(call_list), reused) == (2, False)
    record = json.loads(open(_record_path(out), encoding="utf-8").read())
    assert record["encoding"] == ["euc_jp", "cp932"]
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert (len(call_list), reused) == (2, True)


def test_the_encodings_are_compared_in_order_where_the_record_holds_them(monkeypatch):
    monkeypatch.setattr(knowledge.config, "SOURCE_ENCODING", ["euc_jp", "cp932"])
    assert knowledge.same_encoding({"encoding": ["euc_jp", "cp932"]}) is True
    assert knowledge.same_encoding({"encoding": ["cp932", "euc_jp"]}) is False
    assert knowledge.same_encoding({"encoding": []}) is False
    assert knowledge.same_encoding({}) is True


# == the codetwine that made the analysis ====================================
def test_an_analysis_made_by_another_codetwine_is_made_again_for_the_same_tree(
        tmp_path, monkeypatch):
    repo = _committed_repo(tmp_path / "proj")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    call_list = []
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: call_list.append(1) or write_knowledge_db(
                            _ONE_FILE, db_path, "proj"))
    monkeypatch.setattr(knowledge, "codetwine_id", lambda: "0.4.1@old")
    knowledge.ensure_knowledge(str(repo), str(out))
    assert _record_of(out)["codetwine"] == "0.4.1@old"
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert (len(call_list), reused) == (1, True)
    monkeypatch.setattr(knowledge, "codetwine_id", lambda: "0.4.5@new")
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert (len(call_list), reused) == (2, False)
    assert _record_of(out)["codetwine"] == "0.4.5@new"
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert (len(call_list), reused) == (2, True)


def test_a_record_that_names_no_codetwine_is_made_again(tmp_path, monkeypatch):
    repo = _committed_repo(tmp_path / "proj")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    call_list = []
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: call_list.append(1) or write_knowledge_db(
                            _ONE_FILE, db_path, "proj"))
    knowledge.ensure_knowledge(str(repo), str(out))
    record_dict = _record_of(out)
    del record_dict["codetwine"]
    with open(_record_path(out), "w", encoding="utf-8") as fp:
        json.dump(record_dict, fp)
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert (len(call_list), reused) == (2, False)
    assert _record_of(out)["codetwine"] == knowledge.codetwine_id()


class _Distribution:
    """A stand-in for an installed distribution: its version and direct_url.json."""

    def __init__(self, version, url_text):
        self.version = version
        self._url_text = url_text

    def read_text(self, name):
        assert name == "direct_url.json"
        if isinstance(self._url_text, Exception):
            raise self._url_text
        return self._url_text


@pytest.mark.parametrize("version, url_text, expect", [
    ("0.4.5", '{"url": "https://x/c.git", "vcs_info": {"vcs": "git", '
              '"commit_id": "917e5bd"}}', "0.4.5@917e5bd"),
    ("0.4.5", '{"url": "file:///src/c", "dir_info": {"editable": true}}', "0.4.5"),
    ("0.4.5", None, "0.4.5"),
    ("0.4.5", "not json", "0.4.5"),
    ("0.4.5", "[1]", "0.4.5"),
    ("0.4.5", '{"vcs_info": {"commit_id": 5}}', "0.4.5"),
    ("0.4.5", PermissionError(13, "denied"), "0.4.5"),
    (None, None, ""),
])
def test_the_codetwine_is_told_apart_by_its_version_and_the_commit_it_came_from(
        monkeypatch, version, url_text, expect):
    monkeypatch.setattr(knowledge.importlib.metadata, "distribution",
                        lambda name: _Distribution(version, url_text))
    assert knowledge.codetwine_id() == expect


def test_no_codetwine_distribution_is_told_apart_by_nothing(monkeypatch):
    def not_found(name):
        raise knowledge.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(knowledge.importlib.metadata, "distribution", not_found)
    assert knowledge.codetwine_id() == ""
    assert knowledge.same_codetwine({"codetwine": ""}) is True
    assert knowledge.same_codetwine({}) is False


# == the tree that changed while codetwine ran ==============================
def test_a_tree_that_changed_while_codetwine_ran_records_no_state(tmp_path, monkeypatch,
                                                                  caplog):
    repo = _committed_repo(tmp_path / "proj")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)

    def write_and_change(*a, **k):
        write_knowledge_db(_ONE_FILE, db_path, "proj")
        (repo / "a.py").write_text("y = 2\n", encoding="utf-8")

    monkeypatch.setattr(knowledge, "run_codetwine", write_and_change)
    path, _, _, _ = knowledge.ensure_knowledge(str(repo), str(out))
    record = json.loads(open(_record_path(out), encoding="utf-8").read())
    assert (record["commit"], record["state"]) == ("", "")
    assert "changed while codetwine ran" in caplog.text


def test_an_analysis_of_a_still_tree_records_the_state_it_describes(tmp_path,
                                                                    monkeypatch):
    repo = _committed_repo(tmp_path / "proj")
    (repo / "a.py").write_text("y = 2\n", encoding="utf-8")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: write_knowledge_db(_ONE_FILE, db_path, "proj"))
    knowledge.ensure_knowledge(str(repo), str(out))
    record = json.loads(open(_record_path(out), encoding="utf-8").read())
    assert record["state"] == knowledge.analysis_state(str(repo), None) != ""
    assert record["commit"] == ""


def _analysed_file_list(path: str) -> list[str]:
    """Return the files an analysis on disk holds."""
    knowledge_db = Knowledge.open(path)
    try:
        return sorted(file_dict["file"] for file_dict in knowledge_db.iter_file())
    finally:
        knowledge_db.close()


def _git_in(repo, *args) -> str:
    """Run git in a repository and return what it wrote."""
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout


def _record_of(out) -> dict:
    """Return the record beside the analysis in out."""
    return json.loads(open(_record_path(out), encoding="utf-8").read())


@pytest.mark.parametrize("leave", [
    "named pipe", "no read permission", "directory that cannot be searched"])
def test_a_tracked_file_that_cannot_be_read_is_left_out_while_it_stays_so(
        tmp_path, caplog, leave):
    """The analysis is used again while the file cannot be read, and made again with it
    once it can."""
    if leave != "named pipe" and os.geteuid() == 0:
        pytest.skip("root reads any file")
    repo = _committed_repo(tmp_path / "proj")
    rel = "s/p.py" if leave.startswith("directory") else "p.py"
    (repo / "s").mkdir()
    (repo / rel).write_text("from a import x\n\ndef p():\n    return x\n",
                            encoding="utf-8")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "p")
    head = gitcmd.head_commit(str(repo))
    if leave == "named pipe":
        (repo / rel).unlink()
        os.mkfifo(repo / rel)
    else:
        (repo / rel.split("/")[0]).chmod(0o600 if "/" in rel else 0)
    out = tmp_path / "out"
    try:
        with caplog.at_level("WARNING", logger=knowledge.logger.name):
            path, _, _, _ = knowledge.ensure_knowledge(str(repo), str(out))
        record = _record_of(out)
        assert record["commit"] == ""
        assert record["state"] == knowledge.analysis_state(str(repo), None) != ""
        assert record["state"] != knowledge.commit_state(str(repo), head, None)
        assert f"1 path(s) of {repo} cannot be read as files and are not in the " \
               f"analysis: {rel}" in caplog.text
        assert _analysed_file_list(path) == ["proj/a_py/a.py"]
        _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
        assert reused is True
    finally:
        (repo / rel.split("/")[0]).chmod(0o755 if "/" in rel else 0o644)
    if leave == "named pipe":
        (repo / rel).unlink()
    _git_in(repo, "checkout", "--", rel)
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert reused is False
    assert _analysed_file_list(path) == ["proj/a_py/a.py", f"proj/{rel[:-3]}_py/p.py"]
    assert _record_of(out)["commit"] == head
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert reused is True


def test_an_untracked_file_that_cannot_be_read_leaves_the_state_recorded(tmp_path):
    """The state does not hold the file either, so the analysis is used as it is."""
    repo = _committed_repo(tmp_path / "proj")
    os.mkfifo(repo / "u.py")
    out = tmp_path / "out"
    knowledge.ensure_knowledge(str(repo), str(out))
    record = _record_of(out)
    assert record["state"] == knowledge.analysis_state(str(repo), None) != ""
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert reused is True


def test_an_analysis_of_a_sparse_checkout_is_made_again_once_the_checkout_widens(
        tmp_path):
    """The files the checkout leaves out are not on disk and not in the analysis, which
    describes no commit."""
    repo = _committed_repo(tmp_path / "proj")
    for rel in ("x/b.py", "y/c.py"):
        (repo / rel).parent.mkdir()
        (repo / rel).write_text("v = 1\n", encoding="utf-8")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "two directories")
    head = gitcmd.head_commit(str(repo))
    _git_in(repo, "sparse-checkout", "set", "x")
    assert _git_in(repo, "status", "--porcelain") == ""
    out = tmp_path / "out"
    path, _, _, _ = knowledge.ensure_knowledge(str(repo), str(out))
    assert _analysed_file_list(path) == ["proj/a_py/a.py", "proj/x/b_py/b.py"]
    record = _record_of(out)
    assert record["commit"] == ""
    assert record["state"] != knowledge.commit_state(str(repo), head, None)
    _git_in(repo, "sparse-checkout", "disable")
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert reused is False
    assert _analysed_file_list(path) == ["proj/a_py/a.py", "proj/x/b_py/b.py",
                                         "proj/y/c_py/c.py"]
    assert _record_of(out)["commit"] == head


@pytest.mark.parametrize("leave", ["file", "link to itself"])
def test_a_tracked_file_whose_directory_is_gone_is_left_out_without_a_word(
        tmp_path, caplog, leave):
    """git counts the file as deleted; what is read in its place is what the analysis
    describes, and it is used again."""
    repo = _committed_repo(tmp_path / "proj")
    (repo / "q").mkdir()
    (repo / "q" / "z.py").write_text("z = 1\n", encoding="utf-8")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "q")
    (repo / "q" / "z.py").unlink()
    (repo / "q").rmdir()
    if leave == "file":
        (repo / "q").write_text("q = 1\n", encoding="utf-8")
    else:
        (repo / "q").symlink_to("q")
    out = tmp_path / "out"
    with caplog.at_level("WARNING", logger=knowledge.logger.name):
        knowledge.ensure_knowledge(str(repo), str(out))
    assert "cannot be read" not in caplog.text
    assert _record_of(out)["state"] == knowledge.analysis_state(str(repo), None) != ""
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert reused is True


@pytest.mark.parametrize("exclude_list, rel", [
    (["sec"], "sec/k.py"), (["k.py"], "sec/k.py"), (None, ".github/k.py")])
def test_a_file_a_name_keeps_out_does_not_change_what_the_analysis_reads(
        tmp_path, caplog, exclude_list, rel):
    """Neither its contents nor something that cannot be read in its place."""
    repo = _committed_repo(tmp_path / "proj")
    (repo / rel).parent.mkdir()
    (repo / rel).write_text("k = 1\n", encoding="utf-8")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "k")
    head = gitcmd.head_commit(str(repo))
    state = knowledge.analysis_state(str(repo), exclude_list)
    assert state == knowledge.commit_state(str(repo), head, exclude_list) != ""
    (repo / rel).write_text("k = 2\n", encoding="utf-8")
    assert knowledge.analysis_state(str(repo), exclude_list) == state
    (repo / rel).unlink()
    os.mkfifo(repo / rel)
    assert knowledge.analysis_state(str(repo), exclude_list) == state
    with caplog.at_level("WARNING", logger=knowledge.logger.name):
        knowledge.ensure_knowledge(str(repo), str(tmp_path / "out"),
                                   exclude_list=exclude_list)
    assert "cannot be read" not in caplog.text


def test_a_clone_checked_out_at_the_commit_of_a_tree_gives_the_value_of_that_tree(
        tmp_path):
    """The value of what is read holds the files git tracks or does not ignore, and
    leaves out what git ignores, links and what is kept out."""
    repo = _committed_repo(tmp_path / "proj")
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (repo / "vendor").mkdir()
    (repo / "vendor" / "v.py").write_text("v = 1\n", encoding="utf-8")
    (repo / "in.py").symlink_to("a.py")
    _git_in(repo, "add", "-A")
    _git_in(repo, "commit", "-qm", "more")
    (repo / "new.py").write_text("n = 1\n", encoding="utf-8")
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    exclude_list = ["./vendor"]
    state = knowledge.analysis_state(str(repo), exclude_list)
    (repo / "run.log").write_text("x\n", encoding="utf-8")
    (repo / "vendor" / "v.py").write_text("v = 2\n", encoding="utf-8")
    assert knowledge.analysis_state(str(repo), exclude_list) == state
    snapshot = gitcmd.snapshot_commit(str(repo))
    assert knowledge.commit_state(str(repo), snapshot, exclude_list) == state
    _git_in(repo, "branch", "snap", snapshot)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--quiet", "--no-checkout", str(repo), str(clone)],
                   check=True, capture_output=True)
    _git_in(clone, "checkout", "--quiet", "--detach", snapshot)
    assert knowledge.analysis_state(str(clone), exclude_list) == state
    (repo / "new.py").unlink()
    assert knowledge.analysis_state(str(repo), exclude_list) != state
    assert knowledge.commit_state(str(repo), "", exclude_list) == ""
    assert knowledge.commit_state(str(repo), "no-such-commit", exclude_list) == ""


# == an analysis another command made while this one waited ================
def test_an_analysis_made_for_the_tree_while_waiting_is_used_as_it_is(tmp_path,
                                                                      monkeypatch):
    repo = _committed_repo(tmp_path / "proj")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    call_list = []

    def write(*a, **k):
        call_list.append(k.get("with_doc"))
        write_knowledge_db(_ONE_FILE, db_path, "proj")

    monkeypatch.setattr(knowledge, "run_codetwine", write)
    knowledge.ensure_knowledge(str(repo), str(out))
    _, _, _, reused = knowledge.ensure_knowledge(str(repo), str(out), rebuild=True)
    assert (call_list, reused) == ([False], True)
    knowledge.ensure_knowledge(str(repo), str(out), rebuild=True, exclude_list=["x"])
    knowledge.ensure_knowledge(str(repo), str(out), rebuild=True, exclude_list=["x"],
                               with_doc=True)
    (repo / "a.py").write_text("y = 2\n", encoding="utf-8")
    knowledge.ensure_knowledge(str(repo), str(out), rebuild=True, exclude_list=["x"])
    assert call_list == [False, False, True, False]


# == an analysis made over another repository with the same name ============
def test_the_record_names_the_commits_the_history_starts_from(tmp_path, monkeypatch):
    repo = _committed_repo(tmp_path / "proj")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: write_knowledge_db(_ONE_FILE, db_path, "proj"))
    knowledge.ensure_knowledge(str(repo), str(out))
    record = json.loads(open(_record_path(out), encoding="utf-8").read())
    assert record["root"] == [gitcmd.head_commit(str(repo))]


def test_an_analysis_of_another_repository_with_the_same_name_is_refused(tmp_path,
                                                                        monkeypatch):
    first = _committed_repo(tmp_path / "a" / "proj")
    out = tmp_path / "out"
    db_path = str(out / "proj" / KNOWLEDGE_FILE)
    call_list = []

    def write(*a, **k):
        call_list.append(1)
        write_knowledge_db(_ONE_FILE, db_path, "proj")

    monkeypatch.setattr(knowledge, "run_codetwine", write)
    knowledge.ensure_knowledge(str(first), str(out))
    second = tmp_path / "b" / "proj"
    git_repo(second, commit=False)
    (second / "b.py").write_text("y = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(second), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(second), "commit", "-qm", "other"], check=True)
    for rebuild in (False, True):
        with pytest.raises(RuntimeError, match="another repository with the same name"):
            knowledge.ensure_knowledge(str(second), str(out), rebuild=rebuild)
    clone = tmp_path / "c" / "proj"
    subprocess.run(["git", "clone", "-q", str(first), str(clone)], check=True)
    knowledge.ensure_knowledge(str(clone), str(out), rebuild=True)
    assert call_list == [1]


# == the names kept out are compared as a set ===============================
def test_the_same_names_in_another_order_are_the_same_names():
    assert knowledge.same_exclude({"exclude": ["a", "b"]}, ["b", "a"]) is True
    assert knowledge.same_exclude({"exclude": ["a"]}, ["b"]) is False
    assert knowledge.same_exclude({}, ["a"]) is True


def test_the_names_in_force_are_compared_where_the_record_holds_them(monkeypatch):
    monkeypatch.setattr(knowledge, "_BASE_EXCLUDE_TUPLE", ("__pycache__",))
    record_dict = {"exclude": ["vendor"], "exclude_all": ["__pycache__", "vendor"]}
    assert knowledge.same_exclude(record_dict, ["vendor"]) is True
    monkeypatch.setattr(knowledge, "_BASE_EXCLUDE_TUPLE", ("__pycache__", "node_modules"))
    assert knowledge.same_exclude(record_dict, ["vendor"]) is False


# == what the question about the documents says ==============================
def test_the_question_says_how_many_calls_the_template_takes(tmp_path, monkeypatch,
                                                             capsys):
    template = tmp_path / "doc_template.json"
    template.write_text(json.dumps({"sections": [{"id": "a"}, {"id": "b"}]}),
                        encoding="utf-8")
    monkeypatch.setattr(knowledge.codetwine_setting, "DOC_TEMPLATE_PATH", str(template))
    monkeypatch.setenv("DOC_TEMPLATE_PATH", str(template))
    assert knowledge.doc_call_count() == 3
    asked = {}
    monkeypatch.setattr(progress, "at_terminal", lambda: True)
    monkeypatch.setattr(progress, "ask", lambda question: asked.setdefault("p", question))
    knowledge.ask_doc("k.sqlite")
    assert "3 times per file" in asked["p"]


def test_the_question_asks_without_the_number_when_the_template_cannot_be_read(
        tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge.codetwine_setting, "DOC_TEMPLATE_PATH",
                        str(tmp_path / "nowhere.json"))
    monkeypatch.setenv("DOC_TEMPLATE_PATH", str(tmp_path / "nowhere.json"))
    assert knowledge.doc_call_count() == 0
    asked = {}
    monkeypatch.setattr(progress, "at_terminal", lambda: True)
    monkeypatch.setattr(progress, "ask", lambda question: asked.setdefault("p", question))
    knowledge.ask_doc("k.sqlite")
    assert "once per section of the template" in asked["p"]


# == codetwine's settings after the .env file is written ==========================
def test_codetwine_is_handed_what_it_would_read_from_the_environment_now(monkeypatch):
    """A module holding the value read at the import takes the new one; a module this
    tool gave another value keeps it."""
    from codetwine import doc_creator
    from codetwine.config import settings
    from codetwine.extractors import dependency_graph
    for module, name in ((settings, "DOC_TEMPLATE_PATH"), (doc_creator, "DOC_TEMPLATE_PATH"),
                         (settings, "ENABLE_CODE_SUMMARY"),
                         (doc_creator, "ENABLE_CODE_SUMMARY"),
                         (settings, "EXCLUDE_PATTERNS")):
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(dependency_graph, "EXCLUDE_PATTERNS", ["set-by-this-tool"])
    monkeypatch.setenv("DOC_TEMPLATE_PATH", "/somewhere/template.json")
    monkeypatch.setenv("ENABLE_CODE_SUMMARY", "false")
    monkeypatch.setenv("EXCLUDE_PATTERNS", "one,two")
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    assert knowledge.read_codetwine_setting_again() == [
        "DOC_TEMPLATE_PATH", "ENABLE_CODE_SUMMARY", "EXCLUDE_PATTERNS"]
    assert settings.DOC_TEMPLATE_PATH == doc_creator.DOC_TEMPLATE_PATH \
        == "/somewhere/template.json"
    assert settings.ENABLE_CODE_SUMMARY is doc_creator.ENABLE_CODE_SUMMARY is False
    assert settings.EXCLUDE_PATTERNS[-2:] == ["one", "two"]
    assert dependency_graph.EXCLUDE_PATTERNS == ["set-by-this-tool"]
    assert "PYTHON_DOTENV_DISABLED" not in os.environ
    assert knowledge.read_codetwine_setting_again() == []


def test_a_codetwine_setting_it_cannot_read_stops_naming_what_is_wrong(monkeypatch):
    monkeypatch.setenv("MAX_WORKERS", "many")
    with pytest.raises(SettingError, match="codetwine could not read its settings"):
        knowledge.read_codetwine_setting_again()


_DEEP_JSON = "[" * 100_000 + "]" * 100_000


def test_a_record_or_a_template_nested_too_deep_to_read_is_one_that_cannot_be_read(
        tmp_path, monkeypatch):
    """Neither the record of an analysis nor the document template stops the command."""
    (tmp_path / knowledge.ANALYSIS_COMMIT_FILE).write_text(_DEEP_JSON)
    knowledge._check_owner(str(tmp_path / "project_knowledge.sqlite"), str(tmp_path))
    (tmp_path / "template.json").write_text(_DEEP_JSON)
    monkeypatch.setattr(knowledge.codetwine_setting, "DOC_TEMPLATE_PATH",
                        str(tmp_path / "template.json"))
    monkeypatch.setenv("DOC_TEMPLATE_PATH", str(tmp_path / "template.json"))
    assert knowledge.doc_call_count() == 0


def test_the_template_is_the_repository_s_own_unless_one_is_set(monkeypatch, tmp_path):
    """With DOC_TEMPLATE_PATH unset the documents follow doc_template.json at the top of
    this tool's repository, and codetwine is handed it; a value that is set wins."""
    from codetwine import doc_creator
    from codetwine_loom import image
    (tmp_path / "doc_template.json").write_text('{"sections": [{"id": "a"}]}',
                                                encoding="utf-8")
    monkeypatch.setattr(image, "source_dir", lambda: str(tmp_path))
    monkeypatch.delenv("DOC_TEMPLATE_PATH", raising=False)
    monkeypatch.setattr(knowledge.codetwine_setting, "DOC_TEMPLATE_PATH", "/nowhere.json")
    monkeypatch.setattr(doc_creator, "DOC_TEMPLATE_PATH", "/nowhere.json")
    assert knowledge.doc_template_path() == str(tmp_path / "doc_template.json")
    assert doc_creator.DOC_TEMPLATE_PATH == str(tmp_path / "doc_template.json")
    assert knowledge.doc_call_count() == 2
    monkeypatch.setenv("DOC_TEMPLATE_PATH", "/set.json")
    monkeypatch.setattr(knowledge.codetwine_setting, "DOC_TEMPLATE_PATH", "/set.json")
    assert knowledge.doc_template_path() == "/set.json"

