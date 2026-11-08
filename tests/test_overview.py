"""The repository's overview: when search step 1 is handed it, when it is made again,
and the command line that turns it off."""

import importlib
import json
import logging
import os
import pathlib
import subprocess

import pytest

from conftest import git_repo
from codetwine_loom import answer as answer_mod
from codetwine_loom import (cli, config, dispatch, gitcmd, overview, plan, sandbox,
                             turnlog)
from codetwine_loom.index import build
from codetwine_loom.search import query
from codetwine_loom.search import scope as scope_mod
from codetwine_loom.tools import source


@pytest.fixture
def repo(tmp_path):
    """A repository whose HEAD holds the two files of fake_index, cfg.py and util.py."""
    path = git_repo(tmp_path / "repo", commit=False)
    for name in ("cfg.py", "util.py"):
        (tmp_path / "repo" / name).write_text("a = 1\n", encoding="utf-8")
    _commit_all(path, "first")
    return path


@pytest.fixture
def overview_on(monkeypatch):
    """Turn OVERVIEW on."""
    monkeypatch.setattr(config, "OVERVIEW", True)


def _commit_all(repo: str, message: str) -> None:
    """Commit everything in repo."""
    for argv in (("add", "-A"), ("commit", "--quiet", "-m", message)):
        subprocess.run(["git", "-C", repo, *argv], check=True, capture_output=True)


def _record(repo: str, answer: str = "the requirements") -> dict:
    """An overview answering QUESTION over the files there are now."""
    return {"schema": 3, "commit": gitcmd.head_commit(repo),
            "file_digest": overview._file_digest(repo), "question": overview.QUESTION,
            "answer": answer}


def _write(out_dir: str, record_dict: dict, name: str = "p") -> None:
    """Write an overview for the project name."""
    path = pathlib.Path(overview.overview_path(out_dir, name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record_dict), encoding="utf-8")


def _answer_with(text: str, call_list: list):
    """A run_answer that records the rule it was given and answers text."""
    def run_answer(question: str, content_rule: str) -> dict:
        call_list.append(content_rule)
        return {"ok": True, "answer": text}
    return run_answer


# == What search step 1 is handed ============================

def test_no_overview_leaves_step_1_as_it_is(fake_index, overview_on):
    assert query.OVERVIEW_TEXT == ""
    assert not query.need_expand("add a flag")
    assert query.expand_query("add a flag", lambda _: "never")[0] == "add a flag"


def test_a_current_overview_goes_to_step_1_alone(fake_index, repo, overview_on, tmp_path):
    out_dir = str(tmp_path / "out")
    _write(out_dir, _record(repo))
    overview.load(repo, out_dir, "p")
    assert query.OVERVIEW_TEXT == "the requirements"
    assert query.need_expand("add a flag")
    prompt_list = []

    def llm(prompt: str) -> str:
        prompt_list.append(prompt)
        return "flag option"

    assert query.expand_query("add a flag", llm)[0] == "flag option"
    assert "the requirements" in prompt_list[0]
    # The stages' instructions are left as they are
    assert "the requirements" not in dispatch.build_dispatch_instruction("py")
    assert "the requirements" not in plan.build_plan_instruction("py")


@pytest.mark.parametrize("overview_setting, expand", [(False, True), (True, False)])
def test_the_overview_is_used_only_when_asked(fake_index, repo, tmp_path, monkeypatch,
                                             overview_setting, expand):
    monkeypatch.setattr(config, "OVERVIEW", overview_setting)
    monkeypatch.setattr(config, "QUERY_EXPAND", expand)
    out_dir = str(tmp_path / "out")
    _write(out_dir, _record(repo))
    overview.load(repo, out_dir, "p")
    assert query.OVERVIEW_TEXT == ""


@pytest.mark.parametrize("key, value", [("question", "an older wording"),
                                        ("file_digest", "other files"),
                                        ("answer", "")])
def test_an_overview_that_is_not_current_is_not_used(fake_index, repo, overview_on,
                                                     tmp_path, key, value):
    out_dir = str(tmp_path / "out")
    _write(out_dir, {**_record(repo), key: value})
    overview.load(repo, out_dir, "p")
    assert query.OVERVIEW_TEXT == ""


def test_one_made_for_the_state_the_run_started_from_is_used(fake_index, repo, overview_on,
                                                             tmp_path, monkeypatch):
    """What the run's own agents add does not put the run's overview out of use."""
    out_dir = str(tmp_path / "out")
    _write(out_dir, {**_record(repo), "file_digest": "the start"})
    monkeypatch.setenv(overview.RUN_DIGEST_ENV, "the start")
    overview.load(repo, out_dir, "p")
    assert query.OVERVIEW_TEXT == "the requirements"
    monkeypatch.setenv(overview.RUN_DIGEST_ENV, "another start")
    overview.load(repo, out_dir, "p")
    assert query.OVERVIEW_TEXT == ""
    monkeypatch.setenv(overview.RUN_DIGEST_ENV, "the start")
    _write(out_dir, {**_record(repo), "file_digest": "the start", "question": "older"})
    overview.load(repo, out_dir, "p")
    assert query.OVERVIEW_TEXT == ""


@pytest.mark.parametrize("content", ["not json", json.dumps({"schema": 2, "items": []}),
                                     json.dumps({"schema": 3, "answer": None}),
                                     json.dumps([1])])
def test_an_overview_of_another_shape_is_not_used(tmp_path, content):
    path = tmp_path / "out" / "p" / overview.OVERVIEW_FILE
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    assert overview.read_record(str(tmp_path / "out"), "p") == {}


@pytest.mark.parametrize("leave", ["pipe", "directory", "deep", "large"])
def test_an_overview_that_is_no_regular_file_or_too_deep_is_not_used(tmp_path, caplog,
                                                                     leave):
    """A named pipe is not waited on, and a file of more bytes than an overview is
    read in is not opened as JSON."""
    path = tmp_path / "out" / "p" / overview.OVERVIEW_FILE
    path.parent.mkdir(parents=True)
    if leave == "deep":
        path.write_text("[" * 200000, encoding="utf-8")
    elif leave == "large":
        record_text = json.dumps({"schema": 3, "question": overview.QUESTION,
                                  "answer": "it analyses code"})
        path.write_text(record_text + " " * (overview.RECORD_MAX_BYTE - len(record_text)),
                        encoding="utf-8")
        assert overview.answer_of(str(tmp_path / "out"), "p") == "it analyses code"
        path.write_text(record_text + " " * (overview.RECORD_MAX_BYTE - len(record_text) + 1),
                        encoding="utf-8")
    else:
        (os.mkfifo if leave == "pipe" else os.mkdir)(path)
    with caplog.at_level(logging.WARNING):
        assert overview.read_record(str(tmp_path / "out"), "p") == {}
    assert [entry.levelname for entry in caplog.records] == ["WARNING"]
    assert str(path) in caplog.text
    if leave == "large":
        assert (f"is {overview.RECORD_MAX_BYTE + 1} bytes, over the "
                f"{overview.RECORD_MAX_BYTE} read of one file") in caplog.text
    assert overview.answer_of(str(tmp_path / "out"), "p") == ""


# == Which files the overview stands for ======================

def test_the_digest_follows_the_set_of_files(fake_index, repo, monkeypatch):
    digest = overview._file_digest(repo)
    path_list = scope_mod.INDEX.file_path_list()
    monkeypatch.setattr(scope_mod.INDEX, "file_path_list", lambda: path_list[::-1])
    assert overview._file_digest(repo) == digest
    monkeypatch.setattr(scope_mod.INDEX, "file_path_list", lambda: path_list[1:])
    assert overview._file_digest(repo) != digest


def test_a_file_not_yet_committed_counts_once_the_index_holds_it(fake_index, repo,
                                                                tmp_path, monkeypatch):
    """A run starts from the working tree as it is, so a file on disk that the index
    holds is one of the files the overview stands for, committed or not."""
    digest = overview._file_digest(repo)
    path_list = scope_mod.INDEX.file_path_list()
    (tmp_path / "repo" / "new.py").write_text("b = 1\n", encoding="utf-8")
    assert overview._file_digest(repo) == digest
    monkeypatch.setattr(scope_mod.INDEX, "file_path_list",
                        lambda: [*path_list, "p/new_py/new.py"])
    assert overview._file_digest(repo) != digest
    (tmp_path / "repo" / ".gitignore").write_text("new.py\n", encoding="utf-8")
    assert overview._file_digest(repo) == digest


def test_a_change_to_a_file_leaves_it_current(fake_index, repo, tmp_path):
    digest = overview._file_digest(repo)
    (tmp_path / "repo" / "util.py").write_text("b = 2\n", encoding="utf-8")
    _commit_all(repo, "second")
    assert overview._file_digest(repo) == digest


# == The answering agent's rule ===============================

def test_the_overview_asks_for_no_file_names():
    text = answer_mod.build_answer_instruction("py", overview.CONTENT_RULE)
    assert overview.CONTENT_RULE in text
    assert answer_mod.CONTENT_RULE not in text


def test_the_answering_agent_still_names_files_by_default():
    assert answer_mod.CONTENT_RULE in answer_mod.build_answer_instruction("py")
    assert answer_mod.CONTENT_RULE in answer_mod.build_combine_instruction("py")


# == Making it ================================================

def test_create_writes_the_answer(fake_index, repo, overview_on, tmp_path, monkeypatch):
    call_list = []
    monkeypatch.setattr(answer_mod, "run_answer",
                        _answer_with("it analyses code", call_list))
    out_dir = str(tmp_path / "out")
    assert overview.create(repo, out_dir, "p")["status"] == overview.MAKE_STATUS
    assert call_list == [overview.CONTENT_RULE]
    assert overview.read_record(out_dir, "p") == _record(repo, "it analyses code")
    # The overview it wrote is the one the run goes on with
    assert query.OVERVIEW_TEXT == "it analyses code"

    # Over the same files nothing is asked
    assert overview.create(repo, out_dir, "p")["status"] == overview.KEEP_STATUS
    assert len(call_list) == 1


def test_the_overview_is_made_from_a_directory_of_its_own(fake_index, repo, tmp_path,
                                                        monkeypatch):
    """The answer is written where a relative output directory names from the directory
    the command was run in."""
    seen_list = []

    def run_answer(question: str, content_rule: str) -> dict:
        seen_list.append(os.getcwd())
        return {"ok": True, "answer": "it analyses code"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(answer_mod, "run_answer", run_answer)
    assert overview.create(repo, "out", "p") == {
        "path": overview.overview_path("out", "p"), "status": overview.MAKE_STATUS}
    assert os.path.basename(seen_list[0]).startswith("ctloom-run-")
    assert os.getcwd() == str(tmp_path)
    assert overview.read_record(str(tmp_path / "out"), "p")["answer"] == "it analyses code"


def test_an_answer_with_a_character_utf8_cannot_encode_is_written_as_its_escape(
        fake_index, repo, tmp_path, monkeypatch):
    monkeypatch.setattr(answer_mod, "run_answer", _answer_with("it reads \udce3", []))
    out_dir = str(tmp_path / "out")
    assert overview.create(repo, out_dir, "p")["status"] == overview.MAKE_STATUS
    assert overview.read_record(out_dir, "p")["answer"] == "it reads \udce3"


def test_the_turn_log_is_opened_only_while_the_overview_is_made(
        fake_index, repo, overview_on, tmp_path, monkeypatch):
    monkeypatch.setattr(source, "project_name", "p")
    monkeypatch.setattr(config, "TURN_LOG_KEEP", 20)
    open_list = []

    def run_answer(question: str, content_rule: str) -> dict:
        open_list.append(turnlog._turn_file is not None)
        return {"ok": True, "answer": "it analyses code"}

    monkeypatch.setattr(answer_mod, "run_answer", run_answer)
    out_dir = str(tmp_path / "out")
    turn_dir = pathlib.Path(turnlog.directory())
    assert overview.create(repo, out_dir, "p")["status"] == overview.MAKE_STATUS
    assert open_list == [True] and turnlog._turn_file is None
    name_list = sorted(path.name for path in turn_dir.iterdir())
    assert [name.rsplit(".", 1)[1] for name in name_list] == ["jsonl", "log"]
    for _ in range(3):
        assert overview.create(repo, out_dir, "p")["status"] == overview.KEEP_STATUS
    assert sorted(path.name for path in turn_dir.iterdir()) == name_list
    assert open_list == [True]


@pytest.mark.parametrize("key, value", [("question", "an older wording"),
                                        ("file_digest", "other files"),
                                        ("answer", "")])
def test_one_that_is_not_current_is_made_again(fake_index, repo, tmp_path, monkeypatch,
                                               key, value):
    out_dir = str(tmp_path / "out")
    _write(out_dir, {**_record(repo), key: value})
    monkeypatch.setattr(answer_mod, "run_answer", _answer_with("new", []))
    assert overview.create(repo, out_dir, "p")["status"] == overview.MAKE_STATUS
    assert overview.read_record(out_dir, "p")["answer"] == "new"


def test_an_answer_that_did_not_come_back_leaves_the_file_and_hands_nothing(
        fake_index, repo, overview_on, tmp_path, monkeypatch):
    out_dir = str(tmp_path / "out")
    record_dict = {**_record(repo), "file_digest": "other files"}
    _write(out_dir, record_dict)
    monkeypatch.setattr(answer_mod, "run_answer",
                        lambda *a: {"ok": False, "answer": None})
    assert overview.create(repo, out_dir, "p")["status"] == overview.FAIL_STATUS
    assert overview.read_record(out_dir, "p") == record_dict
    assert query.OVERVIEW_TEXT == ""


# == The command line =========================================

def test_build_calls_no_model_unless_it_is_asked_for_the_documents(monkeypatch):
    assert not cli.calls_a_model(cli._parse_args(["build"]))
    assert cli.calls_a_model(cli._parse_args(["build", "--doc-create"]))
    monkeypatch.setattr(config, "OVERVIEW", True)
    monkeypatch.setattr(config, "QUERY_EXPAND", True)
    assert cli.calls_a_model(cli._parse_args(["overview"]))


@pytest.mark.parametrize("command", ["search", "run", "answer", "edit", "sandbox"])
def test_the_subcommands_that_use_the_overview_take_only_the_no_form(command):
    argv = [command, "--request", "x"]
    assert cli._parse_args(argv).overview is None
    assert cli._parse_args([*argv, "--no-overview"]).overview is False
    with pytest.raises(SystemExit):
        cli._parse_args([*argv, "--overview"])


@pytest.mark.parametrize("argv", [["build", "--no-overview"], ["build", "--overview-create"],
                                  ["integrate", "--branch", "b", "--no-overview"]])
def test_the_subcommands_that_do_not_use_the_overview_take_no_overview_flag(argv):
    with pytest.raises(SystemExit):
        cli._parse_args(argv)


@pytest.fixture
def analysed_repo(tmp_path, monkeypatch):
    """A one-file repository analysed into tmp_path/out, and search step 1's model."""
    repo = git_repo(tmp_path / "repo")
    build.analyse_and_index(repo, str(tmp_path / "out"))
    prompt_list = []
    monkeypatch.setattr(scope_mod, "make_llm",
                        lambda *a, **k: lambda prompt: prompt_list.append(prompt) or "x")
    # Set first so that the teardown takes the names away, whatever main() writes
    monkeypatch.delenv("OVERVIEW")
    monkeypatch.setattr(config, "OVERVIEW", True)
    monkeypatch.setenv("LLM_API_KEY", "a key")
    yield repo, str(tmp_path / "out"), prompt_list
    importlib.reload(config)


def _search(repo: str, out_dir: str, *flag: str) -> None:
    """Run the search subcommand, with the flags given."""
    cli.main(["search", "--project", repo, "--output-dir", out_dir,
              "--request", "add a flag", *flag])


def test_a_search_here_makes_the_overview_first(analysed_repo, monkeypatch):
    repo, out_dir, prompt_list = analysed_repo
    call_list = []
    monkeypatch.setattr(answer_mod, "run_answer",
                        _answer_with("it sets a value", call_list))
    _search(repo, out_dir)
    assert len(call_list) == 1 and "it sets a value" in prompt_list[-1]
    # Over the same files it is used as it is
    _search(repo, out_dir)
    assert len(call_list) == 1 and "it sets a value" in prompt_list[-1]


def test_a_search_in_a_container_makes_none_and_uses_none_that_is_not_current(
        analysed_repo, monkeypatch):
    repo, out_dir, prompt_list = analysed_repo
    _write(out_dir, {"schema": 3, "commit": "", "file_digest": "other files",
                     "question": overview.QUESTION, "answer": "it sets a value"}, "repo")
    monkeypatch.setenv(sandbox.CTLOOM_CLONE_ENV, "/clone")
    monkeypatch.setattr(answer_mod, "run_answer",
                        lambda *a: pytest.fail("nothing is to be made in a container"))
    _search(repo, out_dir)
    # With no overview an ASCII request is searched with as it is
    assert prompt_list == [] and query.OVERVIEW_TEXT == ""


def test_a_search_with_no_overview_makes_none_and_hands_none(analysed_repo, monkeypatch):
    repo, out_dir, prompt_list = analysed_repo
    monkeypatch.setattr(answer_mod, "run_answer",
                        lambda *a: pytest.fail("no overview is to be made"))
    _search(repo, out_dir, "--no-overview")
    assert not pathlib.Path(overview.overview_path(out_dir, "repo")).exists()
    assert prompt_list == [] and query.OVERVIEW_TEXT == ""


def test_build_makes_no_overview(analysed_repo, monkeypatch, capsys):
    repo, out_dir, _ = analysed_repo
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.setattr(answer_mod, "run_answer",
                        lambda *a: pytest.fail("build makes no overview"))
    cli.main(["build", "--project", repo, "--output-dir", out_dir])
    assert not pathlib.Path(overview.overview_path(out_dir, "repo")).exists()
    assert "overview:" not in capsys.readouterr().out


def test_a_subcommand_that_does_not_search_does_not_read_the_overview(
        analysed_repo, monkeypatch, caplog):
    repo, out_dir, _ = analysed_repo
    _write(out_dir, {"schema": 3, "commit": "", "file_digest": "other files",
                     "question": overview.QUESTION, "answer": "it sets a value"}, "repo")
    monkeypatch.setenv(sandbox.CTLOOM_CLONE_ENV, "/clone")
    monkeypatch.setattr(overview, "load",
                        lambda *a: pytest.fail("the overview is not to be read"))
    cli.main(["build", "--project", repo, "--output-dir", out_dir])
    assert "overview" not in caplog.text


@pytest.mark.parametrize("clone", ["", "/clone"])
def test_the_subcommand_run_starts_does_what_build_does_and_makes_the_overview(
        analysed_repo, monkeypatch, capsys, clone):
    """It makes the overview in a container as well as outside one."""
    repo, out_dir, _ = analysed_repo
    call_list = []
    monkeypatch.setattr(answer_mod, "run_answer",
                        _answer_with("it sets a value", call_list))
    if clone:
        monkeypatch.setenv(sandbox.CTLOOM_CLONE_ENV, clone)
    cli.main(["overview", "--project", repo, "--output-dir", out_dir])
    out = capsys.readouterr().out
    assert len(call_list) == 1 and "index: " in out
    assert f"overview: {overview.overview_path(out_dir, 'repo')} (made)" in out
    cli.main(["overview", "--project", repo, "--output-dir", out_dir])
    assert len(call_list) == 1
    assert f"overview: {overview.overview_path(out_dir, 'repo')} (kept)" \
        in capsys.readouterr().out


def _stale_record() -> dict:
    """An overview made over other files than there are now."""
    return {"schema": 3, "commit": "", "file_digest": "other files",
            "question": overview.QUESTION, "answer": "OLD REQUIREMENTS"}


def test_the_subcommand_ends_with_status_1_when_an_overview_could_not_be_made_again(
        analysed_repo, monkeypatch, capsys):
    """The caller takes the overview only after status 0, so the file left there is
    not taken for the state the run starts from."""
    repo, out_dir, _ = analysed_repo
    _write(out_dir, _stale_record(), "repo")
    monkeypatch.setattr(answer_mod, "run_answer", lambda *a: {"ok": False, "answer": None})
    with pytest.raises(SystemExit) as info:
        cli.main(["overview", "--project", repo, "--output-dir", out_dir])
    assert info.value.code == 1
    assert "(no answer)" in capsys.readouterr().out
    assert overview.read_record(out_dir, "repo") == _stale_record()


def test_a_search_goes_on_when_an_overview_could_not_be_made_again(analysed_repo,
                                                                   monkeypatch):
    repo, out_dir, _ = analysed_repo
    _write(out_dir, _stale_record(), "repo")
    monkeypatch.setattr(answer_mod, "run_answer", lambda *a: {"ok": False, "answer": None})
    _search(repo, out_dir)
    assert query.OVERVIEW_TEXT == ""


def test_the_subcommand_makes_none_with_the_overview_off(analysed_repo, monkeypatch):
    repo, out_dir, _ = analysed_repo
    monkeypatch.setattr(config, "OVERVIEW", False)
    monkeypatch.setattr(answer_mod, "run_answer",
                        lambda *a: pytest.fail("the overview is off"))
    cli.main(["overview", "--project", repo, "--output-dir", out_dir])
    assert not pathlib.Path(overview.overview_path(out_dir, "repo")).exists()


def test_the_digest_read_from_disk_is_the_one_the_loaded_index_gives(analysed_repo):
    """What the host records an overview a container made with is what the run inside
    finds it current for."""
    repo, out_dir, _ = analysed_repo
    cli.main(["build", "--project", repo, "--output-dir", out_dir])
    digest = overview.index_digest(repo, out_dir, "repo")
    assert digest and digest == overview._file_digest(repo)
    assert overview.index_digest(repo, out_dir, "another") == ""
    assert not overview.is_current(out_dir, "repo", digest)
    overview.write_record(overview.overview_path(out_dir, "repo"), repo, digest, "it adds")
    assert overview.is_current(out_dir, "repo", digest)
    assert overview.answer_of(out_dir, "repo") == "it adds"
    _write(out_dir, {**_stale_record(), "question": "another question"}, "repo")
    assert overview.answer_of(out_dir, "repo") == ""
