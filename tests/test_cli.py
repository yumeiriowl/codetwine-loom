"""The command-line entry point.

The package is started through __main__.py alone: python -m codetwine_loom and the
codetwine-loom command. No other module of the package starts the command line.
"""

import argparse
import ast
import contextlib
import importlib
import io
import json
import logging
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

import codetwine_loom
from codetwine_loom import cli, config, gitcmd, history, knowledge, progress
from codetwine_loom.index import store
from codetwine_loom.index.indexer import build_bm25
from codetwine_loom.rlm import COMPLETE_KEY
from codetwine_loom import sandbox as sandbox_mod
from codetwine_loom.tools import source
from conftest import CERTIFICATE_PEM, git_repo, write_knowledge_db

# main()'s check for a container engine and its image, as it is without the fixture below
_REFUSE_NO_ENGINE_OR_IMAGE = cli._refuse_no_engine_or_image

# setup's check of the container image, as it is without the fixture below
_SETUP_IMAGE = cli._setup_image

# The check of where main() prints to, as it is without the fixture below
_OUTPUT_IS_TERMINAL = progress.output_is_terminal

# main()'s check of the repository and the output directory, as it is without the
# fixture below
_REFUSE_WRONG_PLACE = cli._refuse_wrong_place


@pytest.fixture(autouse=True)
def _an_engine_and_an_image_are_there(monkeypatch):
    """Let main() past its check for an engine and an image, whatever is installed here,
    and setup past its check of the image."""
    monkeypatch.setattr(cli, "_refuse_no_engine_or_image", lambda args: None)
    monkeypatch.setattr(cli, "_setup_image", lambda args: None)


@pytest.fixture(autouse=True)
def _no_older_run_is_taken_away(monkeypatch):
    """Leave what older runs left alone at the end of sandbox and integrate; the tests
    about that set it themselves."""
    monkeypatch.setattr(cli.sandbox_mod, "remove_old_run_in", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def _the_output_is_a_terminal(monkeypatch):
    """Have what main() prints go to a terminal, so sys.stdin alone decides the questions."""
    monkeypatch.setattr(progress, "output_is_terminal", lambda: True)


@pytest.fixture(autouse=True)
def _the_repository_and_the_output_directory_are_in_place(monkeypatch):
    """Let main() past its check of the repository and the output directory, so a test
    names them without making them; the tests about that check put it back."""
    monkeypatch.setattr(cli, "_refuse_wrong_place", lambda args: None)


def _run_module(module: str) -> subprocess.CompletedProcess:
    """Start a module with python -m.

    Args:
        module: The module name to start.

    Returns:
        A CompletedProcess holding the exit code and the output.
    """
    return subprocess.run([sys.executable, "-m", module],
                          capture_output=True, text=True, timeout=60)


def _is_main_guard(node: ast.stmt) -> bool:
    """Say whether a statement is an "if __name__ == '__main__':" block."""
    return isinstance(node, ast.If) and isinstance(node.test, ast.Compare) \
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"


def test_no_module_but_the_package_s_main_starts_the_command_line():
    """A module started with python -m as codetwine_loom.<module> runs no command:
    the check of the working directory and the one line for a setting that is not
    accepted are in __main__.main(), which only the two entry points reach."""
    package_dir = pathlib.Path(cli.__file__).parent
    guard_list = [str(path.relative_to(package_dir))
                  for path in sorted(package_dir.rglob("*.py"))
                  if any(_is_main_guard(node) for node in ast.parse(
                      path.read_text(encoding="utf-8")).body)]
    assert guard_list == ["__main__.py"]


def test_running_the_package_gives_the_same_message():
    """The package started as a module asks for a subcommand as the command does."""
    r = _run_module("codetwine_loom")
    assert r.returncode == 2
    assert "usage: codetwine-loom" in r.stderr


# == the resume question =======================
@pytest.fixture
def unfinished(tmp_path, monkeypatch):
    """Write one unfinished attempt and return it."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(source, "project_name", "codetwine")
    monkeypatch.setattr(config, "EDIT_HISTORY", True)
    entry = {"n": 1, "at": "2026-08-23T10:00:00", "request": "Go を解析できるように",
             "did": "registered go", "verdict": history.VERDICT_NO_PROOF,
             "why": "nothing checks the request itself",
             "files": {"added": [], "modified": ["a.py"], "deleted": []},
             "signals": {}, "verified": None, "verified_note": ""}
    history.save({"schema": history.SCHEMA, "project": "codetwine",
                  "attempts": [entry]})
    return entry


def test_resume_wins_over_the_question(unfinished, capsys):
    assert cli._want_resume(unfinished, True) is True


def test_no_resume_wins_over_the_question(unfinished):
    assert cli._want_resume(unfinished, False) is False


def test_a_run_with_no_terminal_starts_fresh(unfinished, monkeypatch):
    """A run nobody can answer must not block on an input() that never returns."""
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli._want_resume(unfinished, None) is False


def test_yes_at_the_prompt_carries_the_change_over(unfinished, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert cli._want_resume(unfinished, None) is True


def test_anything_else_at_the_prompt_starts_fresh(unfinished, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert cli._want_resume(unfinished, None) is False


def test_the_question_says_what_was_left_behind(unfinished, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    cli._want_resume(unfinished, None)
    out = capsys.readouterr().out
    assert "registered go" in out and "1 file(s)" in out


# == the history subcommand =================================
def test_history_prints_every_attempt(unfinished, capsys):
    cli.print_history()
    out = capsys.readouterr().out
    assert "edit_history.json" in out
    assert "unverified (confirmed: -)" in out and "mod1" in out


def test_history_says_so_when_nothing_ran(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(source, "project_name", "codetwine")
    cli.print_history()
    assert "no attempt recorded yet" in capsys.readouterr().out


def test_an_attempt_is_two_lines_whatever_its_texts_hold(capsys):
    """A line break in a text of an entry does not start a line of the listing."""
    cli._print_attempt_list([{
        "n": 1, "at": "2026\n  9", "verdict": "passed\n  2  2026  failed",
        "verified": "passed\r\nx", "files": {},
        "why": "w\n  2  2026  passed (confirmed: passed)  add0 mod0 del0"}])
    assert capsys.readouterr().out.splitlines() == [
        "  1  2026 9  passed 2 2026 failed (confirmed: passed x)  add0 mod0 del0",
        "    w 2 2026 passed (confirmed: passed) add0 mod0 del0"]


def test_an_attempt_written_in_a_container_is_listed_in_local_time(monkeypatch, capsys):
    """A container writes its entries with the offset of its own clock; the listing shows
    them in the time of the machine it runs on, as it shows the runs' names."""
    monkeypatch.setenv("TZ", "JST-9")
    time.tzset()
    try:
        cli._print_attempt_list([
            {"n": 1, "at": "2026-10-27T16:32:39+00:00", "verdict": "passed", "files": {}},
            {"n": 2, "at": "2026-10-28T01:40:00", "verdict": "passed", "files": {}}])
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
    line_list = capsys.readouterr().out.splitlines()
    assert line_list[0].startswith("  1  2026-10-28T01:32:39  passed")
    assert line_list[2].startswith("  2  2026-10-28T01:40:00  passed")


def test_the_listing_of_the_records_keeps_each_text_on_its_line(tmp_path, monkeypatch,
                                                                 capsys):
    """The id, the command, the request and the error of a record are one line each."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(source, "project_name", "p")
    directory = sandbox_mod.record_dir(str(tmp_path / "out"), "p")
    os.makedirs(directory)
    with open(os.path.join(directory, "20260101-000000-1-1.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"id": "x\n  y", "command": "run\n  z", "request": "a\nb",
                   "error": "e\n    f", "exit_status": 1, "elapsed_sec": 1,
                   "token": {"total": 2}, "resume": [], "result": {}}, handle)
    cli.print_run_record_list(str(tmp_path))
    assert capsys.readouterr().out.splitlines()[1:] == [
        "  x y  run z  exit 1  1s  2 token(s)", "    a b", "    stopped: e f"]


# == build --exclude ========================================================

def test_build_passes_exclude_through_to_codetwine(tmp_path, monkeypatch):
    """--exclude is what keeps a large repository's design documents affordable."""
    seen = {}

    def fake_run_codetwine(project_dir, output_dir, with_doc=True, exclude_list=None):
        seen["with_doc"] = with_doc
        seen["exclude_list"] = exclude_list
        path = tmp_path / "out" / "proj" / knowledge.KNOWLEDGE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        write_knowledge_db({"files": []}, str(path), "proj")

    repo = tmp_path / "proj"
    git_repo(repo)
    monkeypatch.setattr(knowledge, "run_codetwine", fake_run_codetwine)
    path, _, _, reused = knowledge.ensure_knowledge(
        str(repo), str(tmp_path / "out"), with_doc=False,
        exclude_list=["vendor", "migrations"])

    assert seen == {"with_doc": False, "exclude_list": ["vendor", "migrations"]}
    assert reused is False
    assert path.endswith(knowledge.KNOWLEDGE_FILE)


def test_build_runs_codetwine_only_when_the_analysis_does_not_describe_the_tree(
        tmp_path, monkeypatch, capsys):
    """build reads the analysis as it is while it describes what is
    checked out, and analyses again when the commit moved, without the documents."""
    repo = tmp_path / "proj"
    out = tmp_path / "out"

    def fake_run_codetwine(project_dir, output_dir, with_doc=True, exclude_list=None):
        path = out / "proj" / knowledge.KNOWLEDGE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        write_knowledge_db({"files": []}, str(path), "proj")

    monkeypatch.setattr(knowledge, "run_codetwine", fake_run_codetwine)
    repo.mkdir(parents=True)
    for args in (("init", "--quiet", "-b", "main"),
                 ("config", "user.email", "cli@example.invalid"),
                 ("config", "user.name", "cli")):
        subprocess.run(["git", "-C", str(repo), *args], check=True,
                       capture_output=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "-m", "first"],
                   check=True, capture_output=True)

    call_list = []
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: call_list.append(k["with_doc"]) or
                        fake_run_codetwine(*a, **k))
    monkeypatch.setattr(cli.sandbox_mod, "live_run_list", lambda *a, **k: [])
    argv = ["build", "--project", str(repo), "--output-dir", str(out)]
    cli.main(argv)
    index_path = str(out / "proj" / store.INDEX_FILE)
    analysed_at = store.commit_in_index(index_path)
    assert analysed_at == gitcmd.head_commit(str(repo))
    cli.main(argv)                       # the analysis describes the tree: kept
    assert call_list == [False]

    subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "--allow-empty",
                    "-m", "later"], check=True, capture_output=True)
    cli.main(argv)                       # the commit moved, the tree did not: kept
    assert call_list == [False]

    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "-a", "-m", "changed"],
                   check=True, capture_output=True)
    cli.main(argv)                       # the tree moved: analysed again
    assert call_list == [False, False]
    assert store.commit_in_index(index_path) == gitcmd.head_commit(str(repo))
    cli.main(argv + ["--doc-create"])          # --doc-create runs codetwine either way
    assert call_list == [False, False, True]
    capsys.readouterr()


def test_exclude_is_repeatable_on_build_only():
    args = cli._parse_args(["build", "--project", "p", "--exclude", "vendor",
                            "--exclude", "migrations"])
    assert args.exclude_list == ["vendor", "migrations"]
    assert cli._parse_args(["build", "--project", "p"]).exclude_list is None
    # search and edit do not take it; getattr in main() is what covers that
    assert not hasattr(cli._parse_args(["search", "--project", "p", "--request", "r"]),
                       "exclude")


@pytest.mark.parametrize("head", [["build"], ["run", "--request", "r"],
                                  ["sandbox", "--request", "r"],
                                  ["integrate", "--branch", "b"]])
def test_exclude_takes_several_names_at_once_and_can_be_given_again(head):
    args = cli._parse_args([*head, "--exclude", "vendor", "migrations",
                            "--exclude", "logs"])
    assert args.exclude_list == ["vendor", "migrations", "logs"]
    with pytest.raises(SystemExit):
        cli._parse_args([*head, "--exclude"])


@pytest.mark.parametrize("argv", [
    ["run", "--request", "r", "--into", "x"], ["run", "--request", "r", "--no-take-in"],
    ["sandbox", "--request", "r", "--no-take-in"],
    ["integrate", "--branch", "b", "--into", "x"],
    ["integrate", "--branch", "b", "--no-take-in"]])
def test_the_landing_branch_and_leaving_the_patch_are_not_on_the_command_line(argv):
    with pytest.raises(SystemExit):
        cli._parse_args(argv)


def test_tasks_out_needs_here():
    with pytest.raises(SystemExit):
        cli._parse_args(["run", "--request", "r", "--tasks-out", "t.json"])
    assert cli._parse_args(["run", "--here", "--request", "r", "--tasks-out",
                            "t.json"]).task_out == "t.json"


def test_the_sandbox_subcommand_asks_for_the_overview(monkeypatch, tmp_path):
    call_list = []
    monkeypatch.setattr(cli.sandbox_mod, "run", lambda *a, **k: call_list.append(k) or {
        "exit_status": 0})
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result_dict: None)
    monkeypatch.setenv("LLM_API_KEY", "a key")
    with pytest.raises(SystemExit):
        cli.main(["sandbox", "--project", str(tmp_path), "--request", "r"])
    assert call_list[0]["with_overview"] is True


# == the repository's own instruction file =====
def test_the_instruction_file_can_be_pointed_somewhere_else_on_the_command_line():
    args = cli._parse_args(["edit", "--project", "p", "--request", "r",
                            "--guide", "docs/agent.md"])
    assert args.guide == "docs/agent.md"


def test_the_instruction_file_defaults_to_the_setting():
    args = cli._parse_args(["edit", "--project", "p", "--request", "r"])
    assert args.guide == config.EDIT_GUIDE


def test_only_edit_takes_it():
    """search and run change nothing here, so a rule about changing things has
    nothing to apply to. main() reads it with getattr for that reason."""
    for argv in (["search", "--project", "p", "--request", "r"],
                 ["run", "--project", "p", "--request", "r"]):
        assert not hasattr(cli._parse_args(argv), "guide")


# == the container engine and the image are looked for before the credentials ==
def _check_put_back(monkeypatch) -> None:
    """Put main()'s check back, and fail the test if the run gets past it."""
    monkeypatch.setattr(cli, "_refuse_no_engine_or_image", _REFUSE_NO_ENGINE_OR_IMAGE)
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "")
    monkeypatch.setattr(cli, "_check_credential",
                        lambda *a, **k: pytest.fail("the credentials were checked"))


def _no_engine_on_path(monkeypatch) -> None:
    """Put main()'s check back with no engine on PATH."""
    _check_put_back(monkeypatch)
    monkeypatch.setattr(sandbox_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(sandbox_mod, "image_label_dict",
                        lambda *a: pytest.fail("the image was asked for"))


@pytest.mark.parametrize("argv", [
    ["sandbox", "--request", "r"], ["run", "--request", "r"],
    ["integrate", "--branch", "a", "--branch", "b"], ["warm"]])
def test_a_subcommand_that_starts_containers_stops_with_no_engine(argv, monkeypatch):
    _no_engine_on_path(monkeypatch)
    with pytest.raises(SystemExit, match=f"{argv[0]}: none of podman, docker is on PATH"):
        cli.main([*argv, "--project", "p"])


def test_the_engine_named_on_the_command_line_is_the_one_looked_for(monkeypatch):
    _no_engine_on_path(monkeypatch)
    monkeypatch.setattr(sandbox_mod.shutil, "which",
                        lambda name: "/usr/bin/podman" if name == "podman" else None)
    with pytest.raises(SystemExit, match="run: docker is not on PATH"):
        cli.main(["run", "--request", "r", "--project", "p", "--engine", "docker"])


@pytest.mark.parametrize("argv", [
    ["sandbox", "--request", "r"], ["run", "--request", "r"],
    ["integrate", "--branch", "a", "--branch", "b"], ["warm"]])
def test_a_subcommand_that_starts_containers_stops_with_no_image(argv, monkeypatch,
                                                                 tmp_path):
    _check_put_back(monkeypatch)
    monkeypatch.setattr(sandbox_mod, "pick_engine", lambda name="": "/usr/bin/podman")

    def refuse(engine_path, image):
        raise sandbox_mod.NoImageError("podman does not show the image i")
    monkeypatch.setattr(sandbox_mod, "image_label_dict", refuse)
    with pytest.raises(SystemExit, match=f"{argv[0]}: podman does not show the image i"):
        cli.main([*argv, "--project", str(tmp_path)])


@pytest.mark.parametrize("argv", [
    ["run", "--here", "--request", "r"], ["integrate", "--here", "--branch", "a"],
    ["edit", "--request", "r"], ["answer", "--request", "r"], ["build"],
    ["search", "--request", "r"], ["history"], ["broker", "--endpoint", "unix:///s"]])
def test_a_subcommand_that_starts_no_container_is_not_asked_for_either(argv, monkeypatch):
    monkeypatch.setattr(sandbox_mod, "pick_engine",
                        lambda name="": pytest.fail("an engine was looked for"))
    monkeypatch.setattr(cli, "_ready_image",
                        lambda *a: pytest.fail("the image was asked for"))
    _REFUSE_NO_ENGINE_OR_IMAGE(cli._parse_args(argv))


@pytest.mark.parametrize("argv, env_image, engine, image", [
    (["sandbox", "--request", "r"], "codetwine-loom:set", "", "codetwine-loom:set"),
    (["warm"], "", "", "codetwine-loom"),
    (["sandbox", "--request", "r"], "", "", "codetwine-loom"),
    (["warm"], "codetwine-loom:set", "", "codetwine-loom:set"),
    (["run", "--request", "r", "--engine", "docker", "--image", "mine"],
     "codetwine-loom:set", "docker", "mine")])
def test_the_engine_found_is_asked_for_the_image_of_the_run(argv, env_image, engine, image,
                                                           monkeypatch, tmp_path):
    """The image named by --image or CONTAINER_IMAGE, or else codetwine-loom whatever the
    repository's languages are, is the one asked for and the one the run is left with."""
    (tmp_path / "App.java").write_text("class App {}\n", encoding="utf-8")
    ask_list = []
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "")
    if env_image:
        monkeypatch.setenv("CONTAINER_IMAGE", env_image)
    else:
        monkeypatch.delenv("CONTAINER_IMAGE", raising=False)
    monkeypatch.setattr(config, "CONTAINER_IMAGE", env_image or "codetwine-loom")
    monkeypatch.setattr(sandbox_mod, "pick_engine",
                        lambda name="": ask_list.append(name) or "/usr/bin/engine")
    monkeypatch.setattr(cli, "_ready_image", lambda engine_path, image_name, *a:
                        ask_list.extend([engine_path, image_name]))
    monkeypatch.setattr(sandbox_mod, "refuse_open_network", lambda engine_path: None)
    monkeypatch.setattr(cli, "_env_file_ask_path", lambda: "")
    monkeypatch.setattr(cli.llm, "missing_credential_text", lambda: "")
    args = cli._parse_args([*argv, "--project", str(tmp_path)])
    _REFUSE_NO_ENGINE_OR_IMAGE(args)
    assert ask_list == [engine, "/usr/bin/engine", image]
    assert args.image == image and config.CONTAINER_IMAGE == (env_image or "codetwine-loom")


@pytest.mark.parametrize("argv, asked", [
    (["run", "--request", "r"], ["doc", "dialect", "image"]),
    (["run", "--request", "r", "--doc-create"], ["dialect", "image"]),
    (["integrate", "--branch", "a"], ["dialect", "image"]),
    (["sandbox", "--request", "r"], ["image"])])
def test_the_questions_of_a_run_come_before_the_image_is_built(argv, asked, monkeypatch,
                                                                tmp_path):
    """No question waits behind an image build: the run asks about the documents and the
    COBOL dialect first, and the analysis made later asks nothing again."""
    project = tmp_path / "repo"
    project.mkdir()
    (project / "a.cbl").write_text("       IDENTIFICATION DIVISION.\n", encoding="utf-8")
    order_list = []
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "")
    monkeypatch.setattr(progress, "at_terminal", lambda: True)
    monkeypatch.setattr(sandbox_mod, "pick_engine", lambda name="": "/usr/bin/engine")
    monkeypatch.setattr(cli.knowledge_mod, "ask_doc",
                        lambda path: order_list.append("doc") or False)
    monkeypatch.setattr(cli.knowledge_mod, "ask_cobol_dialect",
                        lambda path, project_dir="", exclude_list=None:
                        order_list.append("dialect") or "")
    monkeypatch.setattr(cli, "_ready_image", lambda *a: order_list.append("image"))
    monkeypatch.setattr(sandbox_mod, "refuse_open_network", lambda engine_path: None)
    monkeypatch.setattr(cli, "_env_file_ask_path", lambda: "")
    _REFUSE_NO_ENGINE_OR_IMAGE(cli._parse_args(
        [*argv, "--project", str(project), "--output-dir", str(tmp_path / "out")]))
    assert order_list == asked


@pytest.mark.parametrize("argv, answer", [
    (["run", "--request", "r"], True), (["run", "--request", "r", "--doc-create"], False),
    (["sandbox", "--request", "r", "--doc-create"], False)])
def test_documents_that_cannot_be_made_stop_the_run_before_the_image(argv, answer,
                                                                     monkeypatch, tmp_path):
    """A yes to the documents, or --doc-create, with no template stops at once: not after
    an image was built."""
    project = tmp_path / "repo"
    project.mkdir()
    (project / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "")
    monkeypatch.setattr(progress, "at_terminal", lambda: True)
    monkeypatch.setattr(sandbox_mod, "pick_engine", lambda name="": "/usr/bin/engine")
    monkeypatch.setattr(cli.knowledge_mod, "ask_doc", lambda path: answer)
    monkeypatch.setattr(cli.knowledge_mod, "doc_template_path",
                        lambda: str(tmp_path / "no_template.json"))
    monkeypatch.setattr(cli, "_ready_image", lambda *a: pytest.fail("the image was asked for"))
    monkeypatch.setattr(cli, "_env_file_ask_path", lambda: "")
    with pytest.raises(SystemExit, match="no file at .*no_template.json"):
        _REFUSE_NO_ENGINE_OR_IMAGE(cli._parse_args(
            [*argv, "--project", str(project), "--output-dir", str(tmp_path / "out")]))


def test_integrate_with_an_analysis_makes_no_documents_and_is_not_stopped_for_them(
        monkeypatch, tmp_path):
    """integrate makes the documents only with an analysis it makes: with one there,
    --doc-create with no template goes on to the image."""
    project = tmp_path / "repo"
    project.mkdir()
    (project / "a.py").write_text("x = 1\n", encoding="utf-8")
    asked_list = []
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "")
    monkeypatch.setattr(sandbox_mod, "pick_engine", lambda name="": "/usr/bin/engine")
    monkeypatch.setattr(sandbox_mod, "refuse_open_network", lambda engine_path: None)
    monkeypatch.setattr(cli.knowledge_mod, "is_complete", lambda path: True)
    monkeypatch.setattr(cli.knowledge_mod, "doc_template_path",
                        lambda: str(tmp_path / "no_template.json"))
    monkeypatch.setattr(cli, "_ready_image", lambda *a: asked_list.append("image"))
    monkeypatch.setattr(cli, "_env_file_ask_path", lambda: "")
    _REFUSE_NO_ENGINE_OR_IMAGE(cli._parse_args(
        ["integrate", "--branch", "a", "--doc-create", "--project", str(project),
         "--output-dir", str(tmp_path / "out")]))
    assert asked_list == ["image"]


# == the image is built when it is missing or stale ===========
_MARK = "m" * 64


@pytest.fixture
def image_engine(monkeypatch, tmp_path):
    """Stand in for the engine's answers about the image and for its build.

    Yields:
        {"label": the labels image_label_dict() gives (None: the engine could not be
        asked; "missing": the image is not there), "status": the build's exit status,
        "build_list": the arguments of each build, "repo": the repository with a
        Dockerfile, "project": a Java repository}. CONTAINER_IMAGE is not set.
    """
    repo = tmp_path / "ctloom"
    repo.mkdir()
    project = tmp_path / "proj"
    project.mkdir()
    (project / "App.java").write_text("class App {}\n", encoding="utf-8")
    state = {"label": "missing", "status": 0, "build_list": [], "repo": str(repo),
             "project": str(project)}

    def label(engine_path, image_name=""):
        if state["label"] == "missing":
            raise sandbox_mod.NoImageError(f"podman does not show the image {image_name} "
                                           f"(exit status 125): image not known")
        return state["label"]

    def build(argv):
        state["build_list"].append(argv)
        return state["status"]
    monkeypatch.setattr(cli.image_mod, "source_dir", lambda: state["repo"])
    monkeypatch.setattr(cli.image_mod, "source_mark", lambda repo_dir: _MARK)
    monkeypatch.setattr(cli.sandbox_mod, "image_label_dict", label)
    monkeypatch.setattr(cli.sandbox_mod, "build_image", build)
    monkeypatch.setattr(cli.sandbox_mod, "pick_engine", lambda name="": "/usr/bin/podman")
    monkeypatch.delenv("CONTAINER_IMAGE", raising=False)
    monkeypatch.setattr(config, "CONTAINER_IMAGE", "codetwine-loom")
    yield state


# The TOOLCHAIN an image is built with unless it is named and its label names one
_EVERY_TOOLCHAIN = cli.image_mod.DEFAULT_TOOLCHAIN


def _ready(state: dict, is_ask: bool, image_name: str = "codetwine-loom",
           is_enter_build: bool = True):
    """Run the image check of image_engine's engine."""
    return cli._ready_image("/usr/bin/podman", image_name, is_ask, is_enter_build)


def _build_argv(state: dict, image_name: str = "codetwine-loom",
                toolchain: str = _EVERY_TOOLCHAIN) -> list[str]:
    return ["/usr/bin/podman", "build", "--build-arg", f"TOOLCHAIN={toolchain}",
            "--label", f"io.codetwine-loom.source={_MARK}",
            "--label", f"io.codetwine-loom.toolchain={toolchain}", "-t", image_name, state["repo"]]


def _build_command(state: dict, image_name: str = "codetwine-loom",
                   toolchain: str = _EVERY_TOOLCHAIN) -> str:
    return (f"podman build --build-arg {shlex.quote(f'TOOLCHAIN={toolchain}')} --label "
            f"io.codetwine-loom.source={_MARK} --label "
            f"{shlex.quote(f'io.codetwine-loom.toolchain={toolchain}')} -t {image_name} "
            f"{state['repo']}")


def test_a_current_image_is_neither_asked_about_nor_built(image_engine, capsys):
    image_engine["label"] = {cli.image_mod.SOURCE_LABEL: _MARK}
    assert _ready(image_engine, True) == cli.image_mod.CURRENT_STATE
    assert image_engine["build_list"] == [] and capsys.readouterr().out == ""


def test_an_engine_that_cannot_be_asked_builds_nothing(image_engine):
    image_engine["label"] = None
    assert _ready(image_engine, True) is None and image_engine["build_list"] == []


def test_a_missing_image_stops_away_from_a_terminal_with_the_command_that_builds_it(
        image_engine):
    with pytest.raises(RuntimeError) as raised:
        _ready(image_engine, False)
    assert str(raised.value).splitlines() == [
        "podman does not show the image codetwine-loom (exit status 125): image not known",
        f"Build it (TOOLCHAIN={_EVERY_TOOLCHAIN}) with:",
        f"  {_build_command(image_engine)}",
        "or set CONTAINER_IMAGE to an image the engine has"]
    assert image_engine["build_list"] == []


def test_a_stale_image_is_warned_about_away_from_a_terminal(image_engine, caplog):
    image_engine["label"] = {cli.image_mod.SOURCE_LABEL: "old"}
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.cli"):
        assert _ready(image_engine, False) == cli.image_mod.STALE_STATE
    assert "the image codetwine-loom (TOOLCHAIN=not named) was not built from this " \
        "codetwine-loom" in caplog.text and _build_command(image_engine) in caplog.text
    assert image_engine["build_list"] == []


@pytest.mark.parametrize("answer", ["y", "", "Yes"])
def test_a_missing_image_is_built_at_a_terminal_on_a_yes(image_engine, monkeypatch,
                                                        capsys, answer):
    asked = []
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: asked.append(question) or answer)
    assert _ready(image_engine, True) == cli.image_mod.ABSENT_STATE
    assert asked == ["Build it now? [Y/n]: "]
    assert image_engine["build_list"] == [_build_argv(image_engine)]
    out = capsys.readouterr().out
    assert f"podman does not have the image codetwine-loom (TOOLCHAIN={_EVERY_TOOLCHAIN})." \
        in out
    assert "a few minutes" in out and "network" in out and "GB of disk" in out
    assert f"  {_build_command(image_engine)}\n" in out
    assert "built the image codetwine-loom" in out


def test_a_missing_image_stops_at_a_terminal_on_a_no(image_engine, monkeypatch):
    monkeypatch.setattr(cli.progress, "ask", lambda question, is_secret=False: "n")
    with pytest.raises(RuntimeError, match="Build it .* with:"):
        _ready(image_engine, True)
    assert image_engine["build_list"] == []


def test_the_end_of_the_input_is_a_no(image_engine, monkeypatch):
    def ask(question, is_secret=False):
        raise EOFError
    monkeypatch.setattr(cli.progress, "ask", ask)
    with pytest.raises(RuntimeError):
        _ready(image_engine, True)
    assert image_engine["build_list"] == []


def test_a_stale_image_goes_on_as_it_is_at_a_terminal_on_a_no(image_engine, monkeypatch,
                                                              caplog, capsys):
    image_engine["label"] = {}
    asked = []
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: asked.append(question) or "n")
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.cli"):
        assert _ready(image_engine, True) == cli.image_mod.STALE_STATE
    assert asked == ["Build it again now? [Y/n]: "]
    assert "was not built from this codetwine-loom" in capsys.readouterr().out
    assert "Build it again with" in caplog.text and image_engine["build_list"] == []


def test_a_stale_image_of_another_name_is_built_again_with_the_toolchain_its_label_names(
        image_engine, monkeypatch):
    """An image that is not the default keeps its label's TOOLCHAIN, and is built with
    every toolchain when its label names none."""
    image_engine["label"] = {cli.image_mod.SOURCE_LABEL: "old",
                             cli.image_mod.TOOLCHAIN_LABEL: "java ts"}
    monkeypatch.setattr(cli.progress, "ask", lambda question, is_secret=False: "y")
    _ready(image_engine, True, "mine")
    image_engine["label"] = {cli.image_mod.SOURCE_LABEL: "old"}
    _ready(image_engine, True, "mine")
    assert image_engine["build_list"] == [_build_argv(image_engine, "mine", "java ts"),
                                          _build_argv(image_engine, "mine")]


@pytest.mark.parametrize("image_name", [
    "codetwine-loom", "codetwine-loom:latest", "localhost/codetwine-loom"])
def test_a_stale_default_image_is_built_again_with_every_toolchain(image_engine,
                                                                    monkeypatch, capsys,
                                                                    image_name):
    """The default image is built with every toolchain whatever its label names, named
    or not."""
    image_engine["label"] = {cli.image_mod.SOURCE_LABEL: "old",
                             cli.image_mod.TOOLCHAIN_LABEL: "java"}
    monkeypatch.setattr(cli.progress, "ask", lambda question, is_secret=False: "y")
    assert _ready(image_engine, True, image_name) == cli.image_mod.STALE_STATE
    assert image_engine["build_list"] == [_build_argv(image_engine, image_name)]
    assert f"The image {image_name} (TOOLCHAIN={_EVERY_TOOLCHAIN}) was not built" \
        in capsys.readouterr().out


@pytest.mark.parametrize("answer, is_built", [("", False), ("y", True), ("n", False)])
def test_enter_builds_nothing_where_it_is_taken_as_no(image_engine, monkeypatch, answer,
                                                      is_built):
    asked = []
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: asked.append(question) or answer)
    if is_built:
        assert _ready(image_engine, True, is_enter_build=False) == cli.image_mod.ABSENT_STATE
    else:
        with pytest.raises(RuntimeError, match="Build it .* with:"):
            _ready(image_engine, True, is_enter_build=False)
    assert asked == ["Build it now? [y/N]: "]
    assert bool(image_engine["build_list"]) is is_built


def test_a_build_that_fails_stops_with_the_engine_s_exit_status(image_engine, monkeypatch):
    image_engine["status"] = 3
    monkeypatch.setattr(cli.progress, "ask", lambda question, is_secret=False: "y")
    with pytest.raises(RuntimeError) as raised:
        _ready(image_engine, True)
    assert str(raised.value).splitlines() == [
        "podman build ended with exit status 3, and the image codetwine-loom was not "
        "built. Build it by hand with:", f"  {_build_command(image_engine)}"]


def test_a_build_that_does_not_end_in_time_stops(image_engine, monkeypatch):
    def build(argv):
        raise subprocess.TimeoutExpired(argv, config.IMAGE_BUILD_TIMEOUT_SEC)
    monkeypatch.setattr(cli.sandbox_mod, "build_image", build)
    monkeypatch.setattr(cli.progress, "ask", lambda question, is_secret=False: "y")
    with pytest.raises(RuntimeError, match=f"did not end within "
                                           f"{config.IMAGE_BUILD_TIMEOUT_SEC} seconds"):
        _ready(image_engine, True)


def test_with_no_dockerfile_a_missing_image_says_to_install_from_a_clone(image_engine):
    image_engine["repo"] = ""
    with pytest.raises(RuntimeError) as raised:
        _ready(image_engine, True)
    assert "Clone the repository and install codetwine-loom from the clone" \
        in str(raised.value)
    image_engine["label"] = {}
    assert _ready(image_engine, True) == cli.image_mod.STALE_STATE
    assert image_engine["build_list"] == []


def test_a_run_away_from_a_terminal_stops_at_a_missing_image_before_the_credentials(
        image_engine, monkeypatch):
    _check_put_back(monkeypatch)
    with pytest.raises(SystemExit) as raised:
        cli.main(["run", "--request", "r", "--project", image_engine["project"]])
    assert str(raised.value).startswith(
        "run: podman does not show the image codetwine-loom (exit status 125)")
    assert _build_command(image_engine) in str(raised.value)


def test_an_image_named_by_the_environment_is_the_one_built(image_engine, monkeypatch):
    monkeypatch.setenv("CONTAINER_IMAGE", "corp/ctl:1")
    monkeypatch.setenv("LLM_API_KEY", config.LLM_API_KEY)
    importlib.reload(config)
    args = cli._parse_args(["sandbox", "--request", "r", "--project",
                            image_engine["project"]])
    with pytest.raises(SystemExit) as raised:
        _REFUSE_NO_ENGINE_OR_IMAGE(args)
    assert _build_command(image_engine, "corp/ctl:1") in str(raised.value)
    assert args.image == "corp/ctl:1"


def test_the_image_is_checked_after_the_first_run_s_questions(image_engine, monkeypatch):
    """With .env still to be written at a terminal, the image is asked for once the file
    is read (_check_credential asks for it again)."""
    monkeypatch.setattr(cli, "_env_file_ask_path", lambda: "/somewhere/.env")
    monkeypatch.setattr(cli, "_ready_image",
                        lambda *a: pytest.fail("the image was asked for"))
    args = cli._parse_args(["run", "--request", "r", "--project", image_engine["project"]])
    _REFUSE_NO_ENGINE_OR_IMAGE(args)
    assert args.image == "codetwine-loom"
    # warm calls no model and asks no question first
    monkeypatch.setattr(cli, "_ready_image", lambda *a: "checked")
    _REFUSE_NO_ENGINE_OR_IMAGE(cli._parse_args(
        ["warm", "--project", image_engine["project"]]))


def test_a_missing_key_stops_the_run_before_the_image(image_engine, monkeypatch):
    """A run with no key, whether the first run's questions left it out or .env never had
    one, stops on it without building the image."""
    monkeypatch.setattr(cli, "_refuse_no_engine_or_image", _REFUSE_NO_ENGINE_OR_IMAGE)
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    _key_is_what_is_missing(monkeypatch)
    monkeypatch.setattr(cli, "_env_file_ask_path", lambda: "")
    monkeypatch.setattr(cli, "_ready_image",
                        lambda *a: pytest.fail("the image was asked for"))
    with pytest.raises(SystemExit, match="LLM_API_KEY"):
        cli.main(["run", "--request", "r", "--project", image_engine["project"]])
    # sandbox's containers call the model through the broker LLM_BROKER names
    monkeypatch.setattr(config, "LLM_BROKER", "http://broker")
    monkeypatch.setattr(cli, "_ready_image", lambda *a: "checked")
    _REFUSE_NO_ENGINE_OR_IMAGE(cli._parse_args(
        ["sandbox", "--request", "r", "--project", image_engine["project"]]))


def test_the_first_run_s_questions_come_before_the_image_s(empty_dir, image_engine,
                                                           monkeypatch):
    monkeypatch.setattr(cli, "_refuse_no_engine_or_image", _REFUSE_NO_ENGINE_OR_IMAGE)
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "")
    _key_is_what_is_missing(monkeypatch)
    asked = _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1"})
    monkeypatch.setattr(cli, "_start_container_command",
                        lambda args: (_ for _ in ()).throw(SystemExit("started")))
    with pytest.raises(SystemExit, match="started"):
        cli.main(["sandbox", "--request", "r", "--project", image_engine["project"]])
    assert asked.index("LLM_API_KEY") < asked.index("Build it now?") == len(asked) - 1
    assert len(image_engine["build_list"]) == 1


def test_a_repository_that_is_not_there_is_left_to_the_later_check(image_engine,
                                                                   monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_ready_image",
                        lambda *a: pytest.fail("the image was asked for"))
    _REFUSE_NO_ENGINE_OR_IMAGE(cli._parse_args(
        ["warm", "--project", str(tmp_path / "none")]))


# == setup checks the image ===================================
def test_setup_builds_the_image_with_no_repository_on_a_yes(empty_dir, image_engine,
                                                            monkeypatch, capsys):
    monkeypatch.setattr(cli, "_setup_image", _SETUP_IMAGE)
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    monkeypatch.delenv("PROJECT_DIR", raising=False)
    prompt_list = _answer_setup(monkeypatch, {"Build it now?": "y"})
    cli.main(["setup"])
    assert "Build it now? [y/N]: " in prompt_list
    assert image_engine["build_list"] == [_build_argv(image_engine)]
    assert "built the image codetwine-loom" in capsys.readouterr().out


def test_setup_prints_why_a_missing_image_is_not_built_and_goes_on(empty_dir, image_engine,
                                                                   monkeypatch, capsys):
    monkeypatch.setattr(cli, "_setup_image", _SETUP_IMAGE)
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    _answer_setup(monkeypatch, {"Build it now?": "n"})
    cli.main(["setup"])
    assert _build_command(image_engine) in capsys.readouterr().out


def test_setup_says_a_current_image_is_built_from_this_codetwine_loom(empty_dir,
                                                                      image_engine,
                                                                      monkeypatch, capsys):
    monkeypatch.setattr(cli, "_setup_image", _SETUP_IMAGE)
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    image_engine["label"] = {cli.image_mod.SOURCE_LABEL: _MARK}
    _answer_setup(monkeypatch, {})
    cli.main(["setup"])
    assert "the container image codetwine-loom is built from this codetwine-loom" \
        in capsys.readouterr().out
    assert image_engine["build_list"] == []


def test_setup_checks_the_image_container_image_names(image_engine, monkeypatch):
    """setup checks CONTAINER_IMAGE's image, and asks with Enter taken as no."""
    ready_list = []
    monkeypatch.setattr(cli, "_ready_image",
                        lambda engine_path, image_name, is_ask, is_enter_build=True:
                        ready_list.append((image_name, is_ask, is_enter_build)) or "checked")
    monkeypatch.setenv("CONTAINER_IMAGE", "corp/ctl:1")
    monkeypatch.setattr(config, "CONTAINER_IMAGE", "corp/ctl:1")
    _SETUP_IMAGE(cli._parse_args(["setup"]))
    monkeypatch.delenv("CONTAINER_IMAGE")
    monkeypatch.setattr(config, "CONTAINER_IMAGE", "codetwine-loom")
    _SETUP_IMAGE(cli._parse_args(["setup"]))
    assert ready_list == [("corp/ctl:1", True, False), ("codetwine-loom", True, False)]


def test_setup_takes_no_repository(capsys):
    with pytest.raises(SystemExit):
        cli._parse_args(["setup", "--project", "p"])
    assert "unrecognized arguments: --project" in capsys.readouterr().err


# == the credentials are checked before the start-up ==========
def _missing_list(setting: str = "LLM_API_KEY") -> list[cli.llm.MissingCredential]:
    """Return one model's missing key, set under the name given."""
    return [cli.llm.MissingCredential("m", ["MEDIUM_MODEL"], "X_API_KEY", setting)]


def _nothing_to_call_with(monkeypatch) -> None:
    """Have the check find a key missing, and fail the test if the start-up is reached."""
    monkeypatch.setattr(cli.llm, "missing_credential_list", _missing_list)
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: pytest.fail("the start-up was reached"))


@pytest.mark.parametrize("argv", [
    ["edit", "--request", "r"], ["answer", "--request", "r"], ["run", "--request", "r"],
    ["sandbox", "--request", "r"], ["integrate", "--branch", "b"],
    ["search", "--request", "r"], ["build", "--doc-create"]])
def test_a_subcommand_that_calls_a_model_stops_with_what_is_missing(argv, monkeypatch):
    _nothing_to_call_with(monkeypatch)
    monkeypatch.setattr(config, "QUERY_EXPAND", True)
    with pytest.raises(SystemExit, match="LLM_API_KEY"):
        cli.main([*argv, "--project", "p"])


@pytest.mark.parametrize("argv", [
    ["build"], ["run-tests"], ["history"], ["warm"], ["search", "--request", "r"]])
def test_a_subcommand_that_calls_no_model_is_not_stopped(argv, monkeypatch):
    """The check is not what stops these: the start-up is, at the missing repository."""
    monkeypatch.setattr(cli.llm, "missing_credential_text",
                        lambda: pytest.fail("the credentials were checked"))
    monkeypatch.setattr(config, "QUERY_EXPAND", False)
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no repo")))
    monkeypatch.setattr(cli.sandbox_mod, "warm",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no repo")))
    monkeypatch.setattr(cli, "print_history", lambda: None)
    try:
        cli.main([*argv, "--project", "p"])
    except SystemExit as e:
        assert "no repo" in str(e)


def test_a_run_with_a_broker_leaves_the_credentials_to_it(monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(cli.llm, "missing_credential_text",
                        lambda: pytest.fail("the credentials were checked"))
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no repo")))
    with pytest.raises(SystemExit, match="no repo"):
        cli.main(["edit", "--project", "p", "--request", "r"])


@pytest.mark.parametrize("argv", [
    ["build"], ["run", "--request", "r"], ["sandbox", "--request", "r"],
    ["integrate", "--branch", "a", "--branch", "b"]])
def test_the_design_documents_are_checked_for_the_credentials_whatever_the_broker(
        argv, monkeypatch):
    """codetwine calls the provider itself, never through the broker."""
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    _nothing_to_call_with(monkeypatch)
    with pytest.raises(SystemExit, match="LLM_API_KEY"):
        cli.main([*argv, "--project", "p", "--doc-create"])


def test_the_broker_itself_is_checked_whatever_its_address(monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    _nothing_to_call_with(monkeypatch)
    monkeypatch.setattr(cli.broker_mod, "serve",
                        lambda *a, **k: pytest.fail("the broker was started"))
    with pytest.raises(SystemExit, match="LLM_API_KEY"):
        cli.main(["broker"])


# == a setting that is set while the one it works with is empty ==
def _no_partner_warning_list(monkeypatch, caplog, argv: list[str],
                             **attribute) -> list[str]:
    """Return what _warn_no_partner_setting() logs for argv, with two settings that have
    no partner."""
    monkeypatch.setattr(config, "no_partner_setting_list",
                        lambda: ["SMOKE_FILE (SMOKE_ARG is empty)",
                                 "SMOKE_ENV (SMOKE_ARG is empty)"])
    args = cli._parse_args(argv)
    for name, value in attribute.items():
        setattr(args, name, value)
    caplog.clear()
    with caplog.at_level("WARNING", logger="codetwine_loom.cli"):
        cli._warn_no_partner_setting(args)
    return [record.getMessage() for record in caplog.records]


@pytest.mark.parametrize("argv", [
    ["edit", "--project", "p", "--request", "r"],
    ["run", "--project", "p", "--request", "r"],
    ["sandbox", "--project", "p", "--request", "r"]])
def test_the_settings_without_a_partner_are_named_in_one_warning(monkeypatch, caplog,
                                                                 argv):
    monkeypatch.delenv(cli.sandbox_mod.CTLOOM_CLONE_ENV, raising=False)
    assert _no_partner_warning_list(monkeypatch, caplog, argv) == [
        "set but not used: SMOKE_FILE (SMOKE_ARG is empty), "
        "SMOKE_ENV (SMOKE_ARG is empty)"]


def test_no_setting_without_a_partner_logs_no_warning(monkeypatch, caplog):
    monkeypatch.delenv(cli.sandbox_mod.CTLOOM_CLONE_ENV, raising=False)
    monkeypatch.setattr(config, "no_partner_setting_list", lambda: [])
    args = cli._parse_args(["edit", "--project", "p", "--request", "r"])
    with caplog.at_level("WARNING", logger="codetwine_loom.cli"):
        cli._warn_no_partner_setting(args)
    assert caplog.records == []


def test_the_settings_without_a_partner_are_not_named_by_a_process_a_run_starts(
        monkeypatch, caplog):
    edit_argv = ["edit", "--project", "p", "--request", "r"]
    monkeypatch.delenv(cli.sandbox_mod.CTLOOM_CLONE_ENV, raising=False)
    assert _no_partner_warning_list(monkeypatch, caplog, ["broker"]) == []
    assert _no_partner_warning_list(monkeypatch, caplog,
                                    ["history", "--project", "p"]) == []
    monkeypatch.setenv(cli.sandbox_mod.CTLOOM_CLONE_ENV, "/work/p")
    assert _no_partner_warning_list(monkeypatch, caplog, edit_argv) == []
    assert _no_partner_warning_list(monkeypatch, caplog,
                                    ["overview", "--project", "p"]) == []


def test_main_names_the_settings_without_a_partner_once_after_the_credentials(
        monkeypatch, caplog):
    monkeypatch.delenv(cli.sandbox_mod.CTLOOM_CLONE_ENV, raising=False)
    monkeypatch.setattr(config, "no_partner_setting_list",
                        lambda: ["SMOKE_FILE (SMOKE_ARG is empty)"])
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no repo")))
    with caplog.at_level("WARNING", logger="codetwine_loom.cli"):
        with pytest.raises(SystemExit, match="no repo"):
            cli.main(["edit", "--project", "p", "--request", "r"])
    assert [record.getMessage() for record in caplog.records
            if "set but not used" in record.getMessage()] == [
        "set but not used: SMOKE_FILE (SMOKE_ARG is empty)"]


# == with no .env read, a terminal is asked and .env is written ==
@pytest.fixture
def empty_dir(tmp_path, monkeypatch):
    """Run in an empty directory with no key and no .env read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    importlib.reload(config)
    assert config.ENV_FILE_PATH == ""
    return tmp_path


def _key_is_what_is_missing(monkeypatch) -> None:
    """Have the check find the key missing until one is set, and stop at the start-up."""
    monkeypatch.setattr(cli.llm, "missing_credential_list",
                        lambda: [] if config.LLM_API_KEY else _missing_list())
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("start-up")))


def _answer_with(monkeypatch, answer_dict: dict) -> list[str]:
    """Answer each prompt from answer_dict by the setting name it starts with."""
    asked = []

    def answer(prompt):
        name = prompt.split(" [")[0].strip()
        asked.append(name)
        return answer_dict.get(name, "")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: answer(question))
    return asked


def test_the_answers_are_written_and_the_run_goes_on_with_them(empty_dir, monkeypatch, capsys):
    _key_is_what_is_missing(monkeypatch)
    asked = _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1", "PRESET": "high"})
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    text = (empty_dir / ".env").read_text(encoding="utf-8")
    assert "\nLLM_API_KEY=sk-1\n" in text and "\nPRESET=high\n" in text
    assert asked[0] == "MEDIUM_MODEL" and asked[-1] == "LLM_API_KEY"
    assert not {"DOC_CREATE_MODEL", "DOC_CREATE_EFFORT"} & set(asked)
    assert "\n# DOC_CREATE_MODEL=\n" in text and "\n# DOC_CREATE_EFFORT=low\n" in text
    # The file just written is what the run reads from here on
    assert config.LLM_API_KEY == "sk-1" and config.PRESET == "high"
    assert config.ENV_FILE_PATH == str(empty_dir / ".env")
    out = capsys.readouterr().out
    assert "needs X_API_KEY" in out and f"wrote {empty_dir / '.env'}" in out


@pytest.mark.parametrize("name, value, why", [
    ("ALL_PROXY", "socks5://svcname:secretpw@127.0.0.1:1080", "points at a SOCKS proxy"),
    ("https_proxy", "http://svcname:pa#secretpw@127.0.0.1:3128",
     "holds a '#', '/' or '?' in the user or the password"),
    ("HTTP_PROXY", "http://127.0.0.1:abc", "names a port that is not a number from 1 to 65535"),
    ("HTTPS_PROXY", "http://svcname:secr[et]pw@127.0.0.1:3128", "holds a user or a password not every HTTP client of this tool can read"),
    ("HTTPS_PROXY", "http://svcname:secretpw@127.1:3128",
     "names a host not every HTTP client of this tool can connect to"),
    ("HTTPS_PROXY", "http://svcname:secretpw@straße.example:3128",
     "names a host the HTTP clients of this tool encode differently")])
def test_a_proxy_variable_no_client_reads_stops_a_subcommand_that_calls_a_model(
        empty_dir, monkeypatch, name, value, why):
    """In one line naming the variable and nothing of its value, before the run starts;
    a subcommand that calls no model is not stopped."""
    monkeypatch.setattr(cli.llm, "missing_credential_text", lambda: "")
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("start-up")))
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as caught:
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert str(caught.value).startswith(f"edit: {name} {why}")
    assert "secretpw" not in str(caught.value) and "svcname" not in str(caught.value)
    monkeypatch.setenv(name, "http://127.0.0.1:3128")
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    monkeypatch.setenv(name, value)
    for argv in (["history"], ["format"], ["build", "--project", "p"]):
        args = cli._parse_args(argv)
        assert cli._check_credential(args, argv) is args


def test_a_subcommand_that_calls_a_model_signs_in_once_its_checks_pass(empty_dir,
                                                                       monkeypatch):
    """After every other check of the start-up; a failed sign-in stops it with litellm's
    lines; one that calls no model, a container's run and a run stopped by a check do
    not sign in."""
    sign_in_list = []

    def sign_in_text():
        sign_in_list.append("signed")
        return "github_copilot/m (EDIT_MODEL): litellm did not sign in to its provider (x)"

    monkeypatch.setattr(cli.llm, "sign_in_text", sign_in_text)
    monkeypatch.setattr(cli.llm, "missing_credential_text", lambda: "")
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("start-up")))
    monkeypatch.setattr(cli, "_refuse_wrong_place", _REFUSE_WRONG_PLACE)
    repo = _git_repo(empty_dir / "p")
    (empty_dir / "f").write_text("", encoding="utf-8")
    for stop_argv, why in (
            (["edit", "--request", "r"], "edit: pass --project"),
            (["edit", "--project", "none", "--request", "r"],
             "edit: no such target repository"),
            (["edit", "--project", repo, "--output-dir", f"{repo}/out", "--request", "r"],
             "edit: OUTPUT_DIR is inside PROJECT_DIR"),
            (["answer", "--project", repo, "--request", "r", "--answer-out", "f/x.json"],
             "answer: --answer-out"),
            (["test-create", "--project", repo, "--request", "r", "--base", "nosuch"],
             "test-create: --base nosuch names no commit")):
        with pytest.raises(SystemExit) as caught:
            cli.main(stop_argv)
        assert str(caught.value).startswith(why)
    assert sign_in_list == []
    argv = ["edit", "--project", repo, "--request", "r"]
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert str(caught.value) == ("github_copilot/m (EDIT_MODEL): litellm did not sign "
                                 "in to its provider (x)")
    assert sign_in_list == ["signed"]
    for other_argv in (["history", "--project", repo], ["build", "--project", repo]):
        cli._sign_in(cli._parse_args(other_argv))
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///tmp/ctloom-b.sock")
    cli._sign_in(cli._parse_args(argv))
    monkeypatch.setattr(config, "LLM_BROKER", "")
    monkeypatch.setattr(cli.llm, "missing_credential_text", lambda: "m needs ONE_API_KEY")
    with pytest.raises(SystemExit, match="m needs ONE_API_KEY"):
        cli.main(argv)
    assert sign_in_list == ["signed"]


def test_a_no_proxy_the_clients_read_apart_stops_a_subcommand_that_calls_a_model(
        empty_dir, monkeypatch):
    """In one line naming the variable and the kind of entry, nothing of its value."""
    monkeypatch.setattr(cli.llm, "missing_credential_text", lambda: "")
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("start-up")))
    monkeypatch.setattr(config, "LLM_API_BASE", "http://llm.corp.example:8080/v1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:3128")
    monkeypatch.setenv("NO_PROXY", "localhost,llm.corp.example:8080")
    with pytest.raises(SystemExit) as caught:
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert str(caught.value).startswith(
        "edit: NO_PROXY is not read alike by every HTTP client of this tool for "
        "LLM_API_BASE")
    assert "an entry with a port (host:port)" in str(caught.value)
    assert "llm.corp" not in str(caught.value)
    monkeypatch.setenv("NO_PROXY", "localhost,llm.corp.example")
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])


def test_the_dot_env_just_written_is_read_whatever_git_says_of_it_afterwards(empty_dir,
                                                                           monkeypatch):
    git_repo(empty_dir)
    _key_is_what_is_missing(monkeypatch)
    _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1"})
    ask = cli._ask_env_file
    state_list = []

    def ask_then_lose_git(is_absent):
        path = ask(is_absent)
        monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
        state_list.append(config.git_index_state(path))
        return path
    monkeypatch.setattr(cli, "_ask_env_file", ask_then_lose_git)
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert state_list == [config.INDEX_UNKNOWN]
    assert config.LLM_API_KEY == "sk-1" and config.ENV_FILE_PATH == str(empty_dir / ".env")


def test_the_template_answered_is_the_one_the_documents_are_made_from(empty_dir,
                                                                     monkeypatch):
    """codetwine read its settings before the file was written."""
    from codetwine import doc_creator
    from codetwine.config import settings
    for module in (settings, doc_creator):
        monkeypatch.setattr(module, "DOC_TEMPLATE_PATH", module.DOC_TEMPLATE_PATH)
    monkeypatch.delenv("DOC_TEMPLATE_PATH", raising=False)
    template = empty_dir / "template.json"
    template.write_text('{"sections": [], "summary_prompt": ""}', encoding="utf-8")
    _key_is_what_is_missing(monkeypatch)
    _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1", "DOC_TEMPLATE_PATH": str(template)})
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["build", "--doc-create", "--project", "p"])
    assert settings.DOC_TEMPLATE_PATH == doc_creator.DOC_TEMPLATE_PATH == str(template)
    assert cli.knowledge_mod.doc_call_count() == 1


@pytest.mark.parametrize("name, answer, line", [
    ("WARM_MAX_REFILL", "n", "\nWARM_MAX_REFILL=0\n"),
    ("WARM_MAX_REFILL", "", "\n# WARM_MAX_REFILL=3\n"),
    ("WARM_MAX_REFILL", "y", "\n# WARM_MAX_REFILL=3\n")])
def test_a_setting_with_a_question_is_asked_for_with_y_or_n(empty_dir, monkeypatch, name,
                                                            answer, line):
    _key_is_what_is_missing(monkeypatch)
    question = {entry.name: entry.question
                for entry in config.env_template_entry_list()}[name]
    asked = _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1", question: answer})
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert question in asked and name not in asked
    assert line in (empty_dir / ".env").read_text(encoding="utf-8")
    monkeypatch.delenv(name, raising=False)
    importlib.reload(config)


@pytest.mark.parametrize("current, answer, choice, value", [
    ("", "", "Y/n", ""), ("", "y", "Y/n", ""), ("", "n", "Y/n", "0"),
    ("0", "", "y/N", "0"), ("0", "y", "y/N", ""), ("0", "n", "y/N", "0"),
    ("5", "", "Y/n, Enter keeps WARM_MAX_REFILL=5", "5"),
    ("5", "y", "Y/n, Enter keeps WARM_MAX_REFILL=5", "5"),
    ("5", "n", "Y/n, Enter keeps WARM_MAX_REFILL=5", "0")])
def test_a_question_sets_its_value_on_n_and_keeps_any_other_on_y(monkeypatch, current,
                                                                 answer, choice, value):
    entry = config.EnvTemplateEntry("WARM_MAX_REFILL", "3", [], "Download them?", "0")
    asked = []

    def ask(question, is_secret=False):
        asked.append(question)
        return answer
    monkeypatch.setattr(cli.progress, "ask", ask)
    assert cli._ask_question(entry, current) == value
    assert asked == [f"Download them? [{choice}]: "]


def test_a_value_out_of_the_choices_is_asked_again(empty_dir, monkeypatch, capsys):
    _key_is_what_is_missing(monkeypatch)
    preset_iter = iter(["hgih", " High"])
    asked = []

    def answer(prompt):
        name = prompt.split(" [")[0].strip()
        asked.append(name)
        return {"LLM_API_KEY": "sk-1"}.get(name) or \
            (next(preset_iter) if name == "PRESET" else "")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: answer(question))
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert asked.count("PRESET") == 2
    assert "PRESET must be one of low / medium / high: hgih" in capsys.readouterr().out
    assert "\nPRESET=high\n" in (empty_dir / ".env").read_text(encoding="utf-8")
    assert config.PRESET == "high"


def test_a_value_the_settings_refuse_is_asked_again(empty_dir, monkeypatch, capsys):
    _key_is_what_is_missing(monkeypatch)
    test_path_list = ["/abs/tests", "tests"]
    asked = _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1"})
    monkeypatch.setattr(
        cli.progress, "ask", lambda question, is_secret=False: (
            asked.append(question.split(" [")[0]) or test_path_list.pop(0)
            if question.startswith("TEST_PATH [")
            else {"LLM_API_KEY": "sk-1"}.get(question.split(" [")[0], "")))
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert asked.count("TEST_PATH") == 2
    assert "TEST_PATH " in capsys.readouterr().out
    assert "\nTEST_PATH=tests\n" in (empty_dir / ".env").read_text(encoding="utf-8")


def test_the_input_ending_writes_nothing(empty_dir, monkeypatch):
    _key_is_what_is_missing(monkeypatch)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    def end(prompt):
        raise EOFError
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr(cli.progress.getpass, "getpass", end)
    with pytest.raises(SystemExit, match="was not written: the input ended"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert not (empty_dir / ".env").exists()


def test_the_project_answered_is_the_one_the_run_is_over(empty_dir, monkeypatch):
    _key_is_what_is_missing(monkeypatch)
    _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1", "PROJECT_DIR": "/repo"})
    seen = {}
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda project, *a, **k: seen.update(project=project) or
                        (_ for _ in ()).throw(RuntimeError("start-up")))
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--request", "r"])
    assert seen["project"] == "/repo"


def test_a_run_with_no_terminal_is_left_to_the_message(empty_dir, monkeypatch):
    _key_is_what_is_missing(monkeypatch)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("a terminal was asked"))
    with pytest.raises(SystemExit, match=r"Set LLM_API_KEY\n\.env read: none"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert not (empty_dir / ".env").exists()


def test_a_run_that_read_a_dot_env_is_told_which(empty_dir, monkeypatch):
    (empty_dir / ".env").write_text("DEF_K=99\n")
    importlib.reload(config)
    _nothing_to_call_with(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("a terminal was asked"))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    with pytest.raises(SystemExit, match=f"env read: {empty_dir / '.env'}"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert (empty_dir / ".env").read_text() == "DEF_K=99\n"


def test_enter_on_everything_writes_the_template_and_stops(empty_dir, monkeypatch):
    _key_is_what_is_missing(monkeypatch)
    _answer_with(monkeypatch, {})
    with pytest.raises(SystemExit, match=f"env read: {empty_dir / '.env'}"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    text = (empty_dir / ".env").read_text(encoding="utf-8")
    assert "\nLLM_API_KEY=\n" in text and "# PRESET=medium" in text



# == with nothing missing, the first run asks whether to answer now ==
_ANSWER_NOW = "Answer the settings now?"


def _key_in_the_shell(monkeypatch) -> None:
    """Have the shell hold the key, and stop at the start-up."""
    monkeypatch.setenv("LLM_API_KEY", "sk-shell")
    importlib.reload(config)
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("start-up")))


def test_a_run_with_its_key_is_asked_whether_to_answer_now(empty_dir, monkeypatch):
    _key_in_the_shell(monkeypatch)
    asked = _answer_with(monkeypatch, {_ANSWER_NOW: "y", "PRESET": "high"})
    try:
        with pytest.raises(SystemExit, match="start-up"):
            cli.main(["edit", "--project", "p", "--request", "r"])
        text = (empty_dir / ".env").read_text(encoding="utf-8")
        assert "\nPRESET=high\n" in text and "\nLLM_API_KEY=\n" in text
        assert asked[:2] == [_ANSWER_NOW, "MEDIUM_MODEL"] and "LLM_API_KEY" not in asked
        assert config.LLM_API_KEY == "sk-shell" and config.PRESET == "high"
    finally:
        monkeypatch.delenv("PRESET", raising=False)
        importlib.reload(config)


@pytest.mark.parametrize("answer", ["", "n"])
def test_anything_but_y_writes_the_template_and_the_run_goes_on(empty_dir, monkeypatch,
                                                              answer, capsys):
    _key_in_the_shell(monkeypatch)
    asked = _answer_with(monkeypatch, {_ANSWER_NOW: answer})
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    with open(config.ENV_TEMPLATE_PATH, encoding="utf-8") as handle:
        assert (empty_dir / ".env").read_text(encoding="utf-8") == handle.read()
    assert asked == [_ANSWER_NOW]
    assert config.LLM_API_KEY == "sk-shell"
    assert f"wrote {empty_dir / '.env'}" in capsys.readouterr().out


def test_a_run_with_its_key_and_no_terminal_is_not_asked(empty_dir, monkeypatch):
    _key_in_the_shell(monkeypatch)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("a terminal was asked"))
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert not (empty_dir / ".env").exists()


def test_a_run_with_its_key_and_a_dot_env_is_not_asked(empty_dir, monkeypatch):
    (empty_dir / ".env").write_text("DEF_K=99\n")
    _key_in_the_shell(monkeypatch)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("a terminal was asked"))
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])


def test_the_input_ending_at_the_first_question_writes_nothing(empty_dir, monkeypatch):
    _key_in_the_shell(monkeypatch)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    def end(prompt):
        raise EOFError
    monkeypatch.setattr("builtins.input", end)
    with pytest.raises(SystemExit, match="was not written: the input ended"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert not (empty_dir / ".env").exists()


# == the questions are asked only where they can be read and .env can be written ==
def _nobody_is_asked(monkeypatch) -> None:
    """Have the input be a terminal, and fail the test when anything is asked on it."""
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("a terminal was asked"))
    monkeypatch.setattr(cli.progress.getpass, "getpass",
                        lambda _: pytest.fail("a terminal was asked"))


def test_a_run_whose_output_is_no_terminal_is_not_asked(empty_dir, monkeypatch):
    _key_in_the_shell(monkeypatch)
    _nobody_is_asked(monkeypatch)
    monkeypatch.setattr(progress, "output_is_terminal", lambda: False)
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert not (empty_dir / ".env").exists()


def test_a_run_without_its_key_whose_output_is_no_terminal_is_left_to_the_message(
        empty_dir, monkeypatch):
    _key_is_what_is_missing(monkeypatch)
    _nobody_is_asked(monkeypatch)
    monkeypatch.setattr(progress, "output_is_terminal", lambda: False)
    with pytest.raises(SystemExit, match=r"Set LLM_API_KEY\n\.env read: none"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert not (empty_dir / ".env").exists()


@pytest.mark.parametrize("name", ["stdin", "stdout"])
def test_a_run_started_with_its_input_or_output_closed_is_not_asked(empty_dir,
                                                                   monkeypatch, name):
    _key_in_the_shell(monkeypatch)
    _nobody_is_asked(monkeypatch)
    monkeypatch.setattr(progress, "output_is_terminal", _OUTPUT_IS_TERMINAL)
    monkeypatch.setattr(cli.sys, name, None)
    assert progress.at_terminal() is False
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert not (empty_dir / ".env").exists()


def test_the_output_is_a_terminal_when_sys_stdout_is_one(monkeypatch):
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    assert _OUTPUT_IS_TERMINAL() is True
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False, raising=False)
    assert _OUTPUT_IS_TERMINAL() is False


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes to any directory")
def test_a_directory_that_cannot_be_written_to_is_not_asked(empty_dir, monkeypatch):
    _key_in_the_shell(monkeypatch)
    _nobody_is_asked(monkeypatch)
    empty_dir.chmod(0o555)
    try:
        with pytest.raises(SystemExit, match="start-up"):
            cli.main(["edit", "--project", "p", "--request", "r"])
    finally:
        empty_dir.chmod(0o755)
    assert not (empty_dir / ".env").exists()


@pytest.mark.parametrize("has_file", [False, True])
def test_a_run_told_to_load_no_dot_env_file_is_not_asked_and_writes_none(
        empty_dir, monkeypatch, has_file):
    if has_file:
        (empty_dir / ".env").write_text("LLM_API_KEY=sk-in-the-file\n")
    monkeypatch.setenv(config.NO_ENV_FILE_ENV, config.NO_ENV_FILE_VALUE)
    importlib.reload(config)
    _key_is_what_is_missing(monkeypatch)
    _nobody_is_asked(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert (empty_dir / ".env").exists() is has_file
    assert "needs X_API_KEY" in str(stop.value)
    last_line = str(stop.value).splitlines()[-1]
    assert last_line == (".env read: none" if not has_file else (
        f".env read: none ({empty_dir / '.env'} was not read: PYTHON_DOTENV_DISABLED "
        f"turns the loading of .env files off)"))
    assert "sk-in-the-file" not in str(stop.value)


@pytest.mark.parametrize("kind", ["directory", "link to nowhere"])
def test_something_else_at_dot_env_is_not_asked_and_is_left_as_it_is(empty_dir,
                                                                    monkeypatch, kind):
    if kind == "directory":
        (empty_dir / ".env").mkdir()
    else:
        (empty_dir / ".env").symlink_to(empty_dir / "nowhere")
    _key_in_the_shell(monkeypatch)
    _nobody_is_asked(monkeypatch)
    with pytest.raises(SystemExit, match="start-up"):
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert config.ENV_FILE_PATH == "" and not (empty_dir / ".env").is_file()


@pytest.mark.parametrize("in_tree", [True, False])
def test_a_dot_env_git_tracks_is_not_read_and_not_written_over(empty_dir, monkeypatch,
                                                               capsys, in_tree):
    """The tracked file is left as it is, in the working tree or only in the index, and
    the missing key's message says it was not read."""
    git_repo(empty_dir)
    (empty_dir / ".env").write_text("LLM_API_KEY=sk-from-the-repository\n")
    subprocess.run(["git", "-C", str(empty_dir), "add", "-f", ".env"], check=True,
                   capture_output=True)
    if not in_tree:
        (empty_dir / ".env").unlink()
    importlib.reload(config)
    _key_is_what_is_missing(monkeypatch)
    _nobody_is_asked(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert config.LLM_API_KEY == "" and config.ENV_FILE_PATH == ""
    assert "LLM_API_KEY" in str(stop.value)
    if in_tree:
        assert str(stop.value).startswith(f"not started: {empty_dir / '.env'} sets LLM_API_KEY")
        assert "git tracks the file" in str(stop.value)
        assert "PYTHON_DOTENV_DISABLED=1" in str(stop.value)
    else:
        assert ".env read: none" in str(stop.value)
    assert "sk-from-the-repository" not in str(stop.value)
    assert (empty_dir / ".env").exists() is in_tree


@pytest.mark.parametrize("argv, name_set", [
    (["history"], set()),
    (["history", "--project", "p", "--output-dir", "o"], {"PROJECT_DIR", "OUTPUT_DIR"}),
    (["history", "--proj", "p", "--output-dir=./out"], {"PROJECT_DIR", "OUTPUT_DIR"}),
    (["build", "--exclude", "vendor"], set()),
    (["sandbox", "--request", "r"], set()),
    (["sandbox", "--request", "r", "--preset", "high", "--no-overview", "--image", "i",
      "--engine", "e", "--guide", "g.md", "--no-check", "--dry-run"],
     {"PRESET", "OVERVIEW", "CONTAINER_IMAGE", "CONTAINER_ENGINE", "EDIT_GUIDE"}),
    (["integrate", "--branch", "b", "--no-test-create", "--no-refactor",
      "--no-test-refactor"], {"TEST_CREATE", "REFACTOR", "TEST_REFACTOR"}),
    (["broker"], set()),
    (["broker", "--endpoint", "unix:///tmp/ctloom.sock"], {"LLM_BROKER"}),
])
def test_the_settings_the_command_line_gave_a_value_to_are_told_from_the_rest(
        argv, name_set):
    """An option passed with the value its setting already holds was passed too."""
    assert cli._parse_args(argv).option_setting_set == name_set
    assert set(cli._OPTION_SETTING_DICT.values()) <= set(config.env_name_tuple())


@pytest.mark.parametrize("env_text, command, name_text", [
    ("OUTPUT_DIR=./dist\nPROJECT_DIR=.\n",
     ["history", "--project", "p", "--output-dir", "o"], ""),
    ("OUTPUT_DIR=./dist\nPROJECT_DIR=.\n", ["history", "--project", "p"], "OUTPUT_DIR"),
    ("OUTPUT_DIR=./dist\nPROJECT_DIR=.\n", ["history"], "OUTPUT_DIR, PROJECT_DIR"),
    ("PRESET=low\nOVERVIEW=true\nCONTAINER_IMAGE=x\nEDIT_GUIDE=g.md\n",
     ["sandbox", "--project", "p", "--output-dir", "o", "--request", "r", "--preset",
      "high", "--no-overview", "--image", "i", "--guide", "h.md"], ""),
    ("PRESET=low\nOVERVIEW=true\nDEF_K=3\n",
     ["sandbox", "--project", "p", "--request", "r", "--preset", "high",
      "--no-overview"], "DEF_K"),
    ("TEST_CREATE=true\nREFACTOR=true\nTEST_REFACTOR=true\n",
     ["integrate", "--project", "p", "--branch", "b", "--no-refactor"],
     "TEST_CREATE, TEST_REFACTOR"),
])
def test_a_setting_the_command_line_gave_a_value_to_does_not_stop_the_command(
        empty_dir, monkeypatch, env_text, command, name_text):
    """The unread .env file would not set it if it were read."""
    class _Past(Exception):
        """What stands for the command having got past the check."""

    def past(args):
        raise _Past()

    git_repo(empty_dir)
    (empty_dir / ".env").write_text(env_text)
    subprocess.run(["git", "add", "-f", ".env"], cwd=empty_dir, check=True)
    # A name the shell holds a value for is not counted either, so none is held
    for name in cli._OPTION_SETTING_DICT.values():
        monkeypatch.delenv(name, raising=False)
    importlib.reload(config)
    monkeypatch.setattr(cli, "_refuse_empty_request", past)
    if not name_text:
        with pytest.raises(_Past):
            cli.main(command)
        return
    with pytest.raises(SystemExit) as stop:
        cli.main(command)
    assert f".env sets {name_text}, and it is not read: git tracks the file" \
        in str(stop.value)


@pytest.mark.parametrize("command", [["history", "--project", "p"],
                                     ["search", "--project", "p", "--request", "r"],
                                     ["sandbox", "--project", "p", "--request", "r"]])
def test_an_unread_dot_env_that_sets_this_tool_s_settings_stops_every_subcommand(
        empty_dir, monkeypatch, command):
    """Whether git tracks the file or did not say: the limits it sets are not left out."""
    git_repo(empty_dir)
    (empty_dir / ".env").write_text("MAX_TOKEN_PER_RUN=5000\nWARM_MAX_REFILL=0\n")
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    importlib.reload(config)
    _key_in_the_shell(monkeypatch)
    _nobody_is_asked(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        cli.main(command)
    assert "sets MAX_TOKEN_PER_RUN, WARM_MAX_REFILL, and it is not read" in str(stop.value)
    assert "git did not say whether it tracks it" in str(stop.value)
    # With loading turned off the same command gets past it
    monkeypatch.setenv(config.NO_ENV_FILE_ENV, config.NO_ENV_FILE_VALUE)
    importlib.reload(config)
    assert config.skip_env_setting_list() == []


def test_an_unread_dot_env_of_the_repository_s_own_stops_nothing_and_says_nothing(
        empty_dir, monkeypatch, caplog, capsys):
    git_repo(empty_dir)
    (empty_dir / ".env").write_text("DATABASE_URL=postgres://x\nPORT=3000\n")
    subprocess.run(["git", "-C", str(empty_dir), "add", "-f", ".env"], check=True,
                   capture_output=True)
    importlib.reload(config)
    _key_in_the_shell(monkeypatch)
    _nobody_is_asked(monkeypatch)
    with caplog.at_level("WARNING"):
        with pytest.raises(SystemExit, match="start-up"):
            cli.main(["edit", "--project", "p", "--request", "r"])
    assert config.ENV_FILE_SKIP_STATE == config.INDEX_HOLD
    assert ".env" not in caplog.text and ".env" not in capsys.readouterr().err


def test_a_dot_env_that_cannot_be_written_stops_the_run_in_one_line(empty_dir,
                                                                   monkeypatch):
    _key_in_the_shell(monkeypatch)
    _answer_with(monkeypatch, {_ANSWER_NOW: ""})

    def refuse(path, answer_dict):
        raise PermissionError(13, "Permission denied", path)
    monkeypatch.setattr(config, "write_env_file", refuse)
    with pytest.raises(SystemExit) as e:
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert str(e.value).startswith(f"{empty_dir / '.env'} was not written: ")
    assert "Permission denied" in str(e.value)


# == what the models lack is asked for last, by the name it is set under ==
def test_the_key_is_asked_for_under_the_settings_answered(empty_dir, monkeypatch):
    """The models answered are what the check is made over, and are put back after."""
    seen = []

    def missing_list():
        seen.append(config.HIGH_MODEL)
        return [] if config.LLM_API_KEY else _missing_list()
    _key_is_what_is_missing(monkeypatch)
    monkeypatch.setattr(cli.llm, "missing_credential_list", missing_list)
    _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1", "HIGH_MODEL": "provider/two"})
    default = config.HIGH_MODEL
    try:
        answer_dict = cli._ask_template({})
    finally:
        assert "HIGH_MODEL" not in os.environ and config.HIGH_MODEL == default
        assert "LLM_API_KEY" not in os.environ and config.LLM_API_KEY == ""
    # Once before the key is asked for, and once more with the key answered
    assert seen == ["provider/two", "provider/two"]
    assert answer_dict == {"HIGH_MODEL": "provider/two", "LLM_API_KEY": "sk-1"}


def test_an_answer_does_not_stand_in_for_what_the_shell_holds(empty_dir, monkeypatch):
    monkeypatch.setenv("HIGH_MODEL", "provider/shell")
    seen = []
    monkeypatch.setattr(cli.llm, "missing_credential_list",
                        lambda: seen.append(config.HIGH_MODEL) or [])
    try:
        cli._lack_with_answer({"HIGH_MODEL": "provider/two"})
        assert seen == ["provider/shell"] and os.environ["HIGH_MODEL"] == "provider/shell"
    finally:
        monkeypatch.delenv("HIGH_MODEL")
        importlib.reload(config)


def test_a_providers_own_variable_is_asked_for_by_name_and_written(empty_dir, monkeypatch,
                                                                   capsys):
    """Two models that lack different keys are not asked for the one LLM_API_KEY."""
    two_list = [cli.llm.MissingCredential("one/m", ["MEDIUM_MODEL"], "ONE_API_KEY",
                                          "ONE_API_KEY"),
                cli.llm.MissingCredential("two/m", ["HIGH_MODEL", "PLAN_MODEL"],
                                          "TWO_API_KEY", "TWO_API_KEY")]
    monkeypatch.setattr(
        cli.llm, "missing_credential_list",
        lambda: [missing for missing in two_list if missing.variable not in os.environ])
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("start-up")))
    asked = _answer_with(monkeypatch, {"ONE_API_KEY": "k-1", "TWO_API_KEY": "k-2"})
    try:
        with pytest.raises(SystemExit, match="start-up"):
            cli.main(["edit", "--project", "p", "--request", "r"])
        text = (empty_dir / ".env").read_text(encoding="utf-8")
        assert text.endswith("\nONE_API_KEY=k-1\nTWO_API_KEY=k-2\n")
        assert "\nLLM_API_KEY=\n" in text
        assert asked[-2:] == ["ONE_API_KEY", "TWO_API_KEY"] and "LLM_API_KEY" not in asked
        assert os.environ["ONE_API_KEY"] == "k-1" and os.environ["TWO_API_KEY"] == "k-2"
        out = capsys.readouterr().out
        assert "two/m (HIGH_MODEL, PLAN_MODEL) needs TWO_API_KEY" in out
        assert cli._PROVIDER_VARIABLE_HELP in out
    finally:
        monkeypatch.delenv("ONE_API_KEY", raising=False)
        monkeypatch.delenv("TWO_API_KEY", raising=False)


def _either_name_is_what_is_missing(monkeypatch) -> None:
    """Have one model lack a key its provider reads under either of 2 names."""
    either_list = [cli.llm.MissingCredential("one/m", ["MEDIUM_MODEL"], name, name)
                   for name in ("ONE_API_KEY", "ONE_OTHER_KEY")]
    monkeypatch.setattr(
        cli.llm, "missing_credential_list",
        lambda: [] if any(missing.variable in os.environ for missing in either_list)
        else either_list)


def test_a_name_an_answer_made_up_for_is_not_asked_for(empty_dir, monkeypatch):
    _either_name_is_what_is_missing(monkeypatch)
    asked = _answer_with(monkeypatch, {"ONE_API_KEY": "k-1"})
    answer_dict = cli._ask_template({})
    assert answer_dict == {"ONE_API_KEY": "k-1"}
    assert "ONE_API_KEY" in asked and "ONE_OTHER_KEY" not in asked
    assert "ONE_API_KEY" not in os.environ


def test_a_name_left_empty_is_followed_by_the_other_name(empty_dir, monkeypatch):
    _either_name_is_what_is_missing(monkeypatch)
    asked = _answer_with(monkeypatch, {"ONE_OTHER_KEY": "k-2"})
    answer_dict = cli._ask_template({})
    assert answer_dict == {"ONE_OTHER_KEY": "k-2"}
    assert asked[-2:] == ["ONE_API_KEY", "ONE_OTHER_KEY"]


def test_a_providers_variable_that_is_no_credential_is_read_with_echo(empty_dir,
                                                                      monkeypatch):
    two_list = [cli.llm.MissingCredential("one/m", ["MEDIUM_MODEL"], name, name)
                for name in ("ONE_API_VERSION", "ONE_ACCESS_KEY_ID")]
    monkeypatch.setattr(
        cli.llm, "missing_credential_list",
        lambda: [missing for missing in two_list if missing.variable not in os.environ])
    echo_list, no_echo_list = [], []

    def answer_into(name_list: list[str]):
        def answer(prompt):
            name_list.append(prompt.split(" [")[0].strip())
            return "v-1"
        return answer
    monkeypatch.setattr(cli.progress, "ask", lambda question, is_secret=False: (
        answer_into(no_echo_list if is_secret else echo_list)(question)))
    assert cli._ask_credential({}, {}) == {"ONE_API_VERSION": "v-1",
                                           "ONE_ACCESS_KEY_ID": "v-1"}
    assert echo_list == ["ONE_API_VERSION"] and no_echo_list == ["ONE_ACCESS_KEY_ID"]


def test_a_key_left_empty_stops_the_run_with_the_file_written(empty_dir, monkeypatch):
    two_list = [cli.llm.MissingCredential("two/m", ["HIGH_MODEL"], "TWO_API_KEY",
                                          "TWO_API_KEY")]
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: two_list)
    _answer_with(monkeypatch, {})
    with pytest.raises(SystemExit) as e:
        cli.main(["edit", "--project", "p", "--request", "r"])
    assert str(e.value) == (f"two/m (HIGH_MODEL) needs TWO_API_KEY, which is not set. "
                            f"Set TWO_API_KEY\n.env read: {empty_dir / '.env'}")
    assert "TWO_API_KEY" not in (empty_dir / ".env").read_text(encoding="utf-8")


# == the first run's question about where web_search searches ==
_SEARCH_NAME_TUPLE = ("WEB_SEARCH_PROVIDER", "WEB_SEARCH_API_KEY",
                      "WEB_FETCH_PROVIDER", "WEB_FETCH_API_KEY")


def _first_run_env_text(empty_dir, monkeypatch, answer_dict: dict) -> tuple[str, list[str]]:
    """Answer the first run's questions and return the .env written and what was asked."""
    _key_is_what_is_missing(monkeypatch)
    asked = _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1", **answer_dict})
    try:
        with pytest.raises(SystemExit, match="start-up"):
            cli.main(["edit", "--project", "p", "--request", "r"])
    finally:
        for name in (*_SEARCH_NAME_TUPLE, "WARM_MAX_REFILL"):
            monkeypatch.delenv(name, raising=False)
        importlib.reload(config)
    return (empty_dir / ".env").read_text(encoding="utf-8"), asked


def _search_choice_list() -> list[str]:
    return [config.WEB_SEARCH_BY_THE_MODEL, *config.WEB_SEARCH_PRESET_DICT]


def test_the_search_question_offers_every_value_the_setting_takes(empty_dir, monkeypatch,
                                                                  capsys):
    _first_run_env_text(empty_dir, monkeypatch, {})
    out = capsys.readouterr().out
    for number, choice in enumerate(_search_choice_list(), 1):
        assert f"\n  {number}) {choice}" in out


def test_enter_on_the_search_question_sets_nothing_and_asks_no_more(empty_dir, monkeypatch):
    text, asked = _first_run_env_text(empty_dir, monkeypatch, {})
    assert "\n# WEB_SEARCH_PROVIDER=\n" in text
    assert [name for name in asked if name.startswith("WEB_SEARCH_")] == ["WEB_SEARCH_PROVIDER"]


def test_the_model_s_own_search_needs_no_more_than_the_choice(empty_dir, monkeypatch):
    text, asked = _first_run_env_text(empty_dir, monkeypatch, {"WEB_SEARCH_PROVIDER": "1"})
    assert f"\nWEB_SEARCH_PROVIDER={config.WEB_SEARCH_BY_THE_MODEL}\n" in text
    assert "WEB_SEARCH_API_KEY" not in asked and "\n# WEB_SEARCH_API_KEY=\n" in text


@pytest.mark.parametrize("preset", sorted(config.WEB_SEARCH_PRESET_DICT))
def test_a_search_api_is_asked_for_its_key(empty_dir, monkeypatch, preset):
    text, asked = _first_run_env_text(
        empty_dir, monkeypatch, {"WEB_SEARCH_PROVIDER": preset, "WEB_SEARCH_API_KEY": "k-1"})
    assert f"\nWEB_SEARCH_PROVIDER={preset}\n" in text and "\nWEB_SEARCH_API_KEY=k-1\n" in text


def test_a_search_api_with_no_key_is_left_unset(empty_dir, monkeypatch):
    preset = sorted(config.WEB_SEARCH_PRESET_DICT)[0]
    text, _ = _first_run_env_text(empty_dir, monkeypatch, {"WEB_SEARCH_PROVIDER": preset})
    assert "\n# WEB_SEARCH_PROVIDER=\n" in text and "\n# WEB_SEARCH_API_KEY=\n" in text


def test_an_answer_that_is_no_choice_is_asked_again(empty_dir, monkeypatch):
    _key_is_what_is_missing(monkeypatch)
    _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1"})
    answer_iter = iter(["99", "google", "1"])
    entry = {e.name: e for e in config.env_template_entry_list()}["WEB_SEARCH_PROVIDER"]
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: next(answer_iter))
    assert cli._ask_endpoint_choice(entry, _search_choice_list()) == \
        config.WEB_SEARCH_BY_THE_MODEL


def test_a_digit_that_is_no_number_is_asked_again(empty_dir, monkeypatch):
    _key_is_what_is_missing(monkeypatch)
    _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1"})
    answer_iter = iter(["\u2460", "\u00b2", "\uff11"])
    entry = {e.name: e for e in config.env_template_entry_list()}["WEB_SEARCH_PROVIDER"]
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: next(answer_iter))
    assert cli._ask_endpoint_choice(entry, _search_choice_list()) == \
        config.WEB_SEARCH_BY_THE_MODEL


def test_a_run_that_fetches_no_package_is_still_asked_where_to_search(empty_dir,
                                                                      monkeypatch):
    question = {entry.name: entry.question
                for entry in config.env_template_entry_list()}["WARM_MAX_REFILL"]
    text, asked = _first_run_env_text(empty_dir, monkeypatch,
                                      {question: "n", "WEB_SEARCH_PROVIDER": "1"})
    assert "\nWARM_MAX_REFILL=0\n" in text
    assert f"\nWEB_SEARCH_PROVIDER={config.WEB_SEARCH_BY_THE_MODEL}\n" in text
    assert "WEB_FETCH_PROVIDER" in asked


# == the first run's question about where web_fetch reads pages through ==
def _fetch_choice_list() -> list[str]:
    return list(config.WEB_FETCH_PRESET_DICT)


def test_the_fetch_question_offers_every_value_the_setting_takes(empty_dir, monkeypatch,
                                                                 capsys):
    _first_run_env_text(empty_dir, monkeypatch, {"WEB_SEARCH_PROVIDER": "1"})
    out = capsys.readouterr().out.split("# Where web_fetch reads", 1)[1]
    for number, choice in enumerate(_fetch_choice_list(), 1):
        assert f"\n  {number}) {choice}" in out
    assert ") model" not in out


def test_enter_on_the_fetch_question_sets_nothing_and_asks_no_more(empty_dir, monkeypatch):
    text, asked = _first_run_env_text(empty_dir, monkeypatch, {"WEB_SEARCH_PROVIDER": "1"})
    assert "\n# WEB_FETCH_PROVIDER=\n" in text
    assert [name for name in asked if name.startswith("WEB_FETCH_")] == ["WEB_FETCH_PROVIDER"]


def test_the_fetch_question_is_not_asked_without_a_search(empty_dir, monkeypatch):
    """web_fetch reads only what web_search returned."""
    text, asked = _first_run_env_text(empty_dir, monkeypatch, {})
    assert "\n# WEB_FETCH_PROVIDER=\n" in text
    assert not [name for name in asked if name.startswith("WEB_FETCH_")]


def test_an_api_chosen_for_both_is_asked_for_its_key_once(empty_dir, monkeypatch):
    """WEB_FETCH_API_KEY follows WEB_SEARCH_API_KEY when the two name the same API."""
    both = sorted(set(config.WEB_SEARCH_PRESET_DICT) & set(config.WEB_FETCH_PRESET_DICT))[0]
    text, asked = _first_run_env_text(
        empty_dir, monkeypatch,
        {"WEB_SEARCH_PROVIDER": both, "WEB_SEARCH_API_KEY": "k-1", "WEB_FETCH_PROVIDER": both})
    assert f"\nWEB_SEARCH_PROVIDER={both}\n" in text and "\nWEB_SEARCH_API_KEY=k-1\n" in text
    assert f"\nWEB_FETCH_PROVIDER={both}\n" in text and "\n# WEB_FETCH_API_KEY=\n" in text
    assert "WEB_FETCH_API_KEY" not in asked


def test_a_page_reading_api_the_search_did_not_choose_is_asked_for_its_key(empty_dir,
                                                                             monkeypatch):
    preset = sorted(config.WEB_FETCH_PRESET_DICT)[0]
    text, asked = _first_run_env_text(
        empty_dir, monkeypatch,
        {"WEB_SEARCH_PROVIDER": "1", "WEB_FETCH_PROVIDER": preset, "WEB_FETCH_API_KEY": "k-2"})
    assert f"\nWEB_FETCH_PROVIDER={preset}\n" in text and "\nWEB_FETCH_API_KEY=k-2\n" in text
    assert "WEB_FETCH_API_KEY" in asked


def test_enter_on_the_proxy_question_asks_for_none_of_its_settings(empty_dir, monkeypatch):
    text, asked = _first_run_env_text(empty_dir, monkeypatch, {})
    assert cli._PROXY_QUESTION in asked
    assert not set(cli._PROXY_SETTING_TUPLE) & set(asked)
    assert "\n# HTTPS_PROXY=\n# \n" not in text
    for name in cli._PROXY_SETTING_TUPLE:
        assert f"\n# {name}=\n" in text


def test_a_y_to_the_proxy_question_asks_for_its_settings_and_writes_them(empty_dir,
                                                                        monkeypatch):
    ca_file = empty_dir / "ctloom-ca.pem"
    ca_file.write_text(CERTIFICATE_PEM, encoding="utf-8")
    text, asked = _first_run_env_text(
        empty_dir, monkeypatch,
        {cli._PROXY_QUESTION: "y", "HTTPS_PROXY": "http://ctloom-proxy.corp.example:3128",
         "NO_PROXY": "localhost,.corp.example", "SSL_CERT_FILE": "ctloom-ca.pem"})
    start = asked.index(cli._PROXY_QUESTION)
    assert asked[start + 1:start + 4] == ["HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE"]
    assert "\nHTTPS_PROXY=http://ctloom-proxy.corp.example:3128\n" in text
    assert "\nNO_PROXY=localhost,.corp.example\n" in text
    assert f"\nSSL_CERT_FILE={ca_file}\n" in text


def test_with_no_proxy_answered_the_hosts_reached_without_it_are_not_asked_for(
        empty_dir, monkeypatch):
    text, asked = _first_run_env_text(empty_dir, monkeypatch, {cli._PROXY_QUESTION: "y"})
    assert "HTTPS_PROXY" in asked and "SSL_CERT_FILE" in asked and "NO_PROXY" not in asked
    for name in cli._PROXY_SETTING_TUPLE:
        assert f"\n# {name}=\n" in text


def test_a_proxy_setting_the_environment_holds_is_not_asked_for(empty_dir, monkeypatch,
                                                                capsys):
    """.env fills only the names the shell leaves unset, so an answer would set nothing."""
    monkeypatch.setenv("https_proxy", "http://ctloom-proxy.corp.example:3128")
    text, asked = _first_run_env_text(
        empty_dir, monkeypatch, {cli._PROXY_QUESTION: "y", "NO_PROXY": "localhost"})
    assert "HTTPS_PROXY" not in asked and "NO_PROXY" in asked and "SSL_CERT_FILE" in asked
    assert "\n# HTTPS_PROXY=\n" in text and "\nNO_PROXY=localhost\n" in text
    assert "HTTPS_PROXY is set in the environment and is left as it is" in (
        capsys.readouterr().out)


@pytest.mark.parametrize("answer, why", [
    ("y", "names no port"), ("yes", "names no port"), ("proxy.corp.example", "names no port"),
    ("http://", "names no host"), ("http://:3128", "names no host"),
    ("http://proxy corp.example:3128", "holds a blank"),
    ("socks5://proxy.corp.example:1080", "points at a SOCKS proxy"),
    ("ftp://proxy.corp.example:3128", "is not an http:// or https:// URL"),
    ("http://svcname:pa#secretpw@proxy.corp.example:3128", "holds a '#', '/' or '?'"),
    ("http://svcname:pa/secretpw@proxy.corp.example:3128", "holds a '#', '/' or '?'"),
    ("http://alice:8080#secretpw@proxy.corp.example:3128", "holds a '#', '/' or '?'"),
    ("http://proxy.corp.example:abc", "names a port that is not a number from 1 to 65535"),
    ("http://proxy.corp.example:99999", "names a port that is not a number from 1 to 65535"),
    ("svcname:[secretpw@proxy.corp.example:3128", "holds a user or a password not every HTTP client of this tool can read"),
    ("svcname:secretpw]@proxy.corp.example:3128", "holds a user or a password not every HTTP client of this tool can read"),
    ("svc[name:secretpw@proxy.corp.example:3128", "holds a user or a password not every HTTP client of this tool can read"),
    ("http://☃.example:3128", "names a host not every HTTP client of this tool can"),
    ("http://straße.example:3128", "names a host the HTTP clients of this tool encode"),
    ("http://svcname:pässecretpw@proxy.corp.example:3128",
     "holds a user or a password with a character outside ASCII"),
    ("http://svcname:%E6%97%A5secretpw@proxy.corp.example:3128",
     "holds a user or a password with a character outside ASCII"),
    ("http://proxy.corp.example:0", "names a port that is not a number from 1 to 65535"),
    ("http://proxy.corp.example:3128?secretpw", "holds a path, a query or a fragment")])
def test_an_answer_that_is_no_proxy_s_url_is_asked_again(empty_dir, monkeypatch, capsys,
                                                         answer, why):
    """With why, and nothing of the answer."""
    answer_iter = iter([answer, "proxy.corp.example:3128"])
    entry = {e.name: e for e in config.env_template_entry_list()}["HTTPS_PROXY"]
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: next(answer_iter))
    assert cli._ask_proxy_url(entry) == "proxy.corp.example:3128"
    out = capsys.readouterr().out
    assert f"HTTPS_PROXY {why}" in out
    assert "secretpw" not in out and "svcname" not in out


@pytest.mark.parametrize("answer, why", [
    ("llm.corp.example:8443", "an entry with a port (host:port)"),
    ("https://llm.corp.example", "an entry with a scheme (scheme://host)"),
    ("localhost,*", "'*' among other entries"),
    ("[::1]", "holds an entry not every HTTP client of this tool can read")])
def test_hosts_reached_without_the_proxy_the_clients_read_apart_are_asked_again(
        empty_dir, monkeypatch, capsys, answer, why):
    """Checked against the proxy and the LLM_API_BASE answered before them."""
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    answer_iter = iter([answer, "localhost,llm.corp.example"])
    entry = {e.name: e for e in config.env_template_entry_list()}["NO_PROXY"]
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: next(answer_iter))
    assert cli._ask_no_proxy(entry, "http://proxy.corp.example:3128",
                             "https://llm.corp.example:8443/v1") == (
        "localhost,llm.corp.example")
    out = capsys.readouterr().out
    assert "\nNO_PROXY " in out and why in out and "llm.corp" not in out


def _answer_setup(monkeypatch, answer_dict: dict) -> list[str]:
    """Answer each prompt from answer_dict by the text it starts with, a list giving one
    answer per time it is asked, and keep the prompts."""
    prompt_list = []

    def answer(prompt):
        prompt_list.append(prompt)
        key = next((key for key in answer_dict if prompt.startswith(key)), None)
        value = answer_dict.get(key, "")
        return value.pop(0) if isinstance(value, list) else value
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: answer(question))
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.delenv(config.NO_ENV_FILE_ENV, raising=False)
    return prompt_list


def test_setup_writes_a_dot_env_not_there_without_asking_whether_to(empty_dir,
                                                                     monkeypatch, capsys):
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    prompt_list = _answer_setup(monkeypatch, {"PRESET": "high"})
    cli.main(["setup"])
    assert prompt_list[0].startswith("MEDIUM_MODEL [")
    assert not any(prompt.startswith("Answer the settings now") for prompt in prompt_list)
    assert "\nPRESET=high\n" in (empty_dir / ".env").read_text(encoding="utf-8")
    assert f"wrote {empty_dir / '.env'}" in capsys.readouterr().out


def test_setup_shows_what_the_dot_env_holds_and_writes_over_what_changes(
        empty_dir, monkeypatch, capsys):
    """Enter keeps a value, - takes it out, and the key is asked for again although
    nothing is missing."""
    _key_is_what_is_missing(monkeypatch)
    config.write_env_file(str(empty_dir / ".env"), {
        "LLM_API_KEY": "sk-1", "PRESET": "high", "WEB_SEARCH_PROVIDER": "tavily",
        "WEB_SEARCH_API_KEY": "tv-1"})
    importlib.reload(config)
    assert config.LLM_API_KEY == "sk-1"
    prompt_list = _answer_setup(monkeypatch, {"PRESET": "-", "LLM_API_KEY": "sk-2"})
    cli.main(["setup"])
    assert "PRESET [high]: " in prompt_list and "LLM_API_KEY [set]: " in prompt_list
    assert "WEB_SEARCH_API_KEY [set]: " in prompt_list
    assert any("Enter keeps tavily" in prompt for prompt in prompt_list)
    text = (empty_dir / ".env").read_text(encoding="utf-8")
    assert "\nLLM_API_KEY=sk-2\n" in text and "\n# PRESET=medium\n" in text
    assert "\nWEB_SEARCH_PROVIDER=tavily\n" in text and "\nWEB_SEARCH_API_KEY=tv-1\n" in text
    out = capsys.readouterr().out
    assert "Enter keeps the value in brackets" in out and "sk-1" not in out


def test_setup_checks_the_settings_the_dot_env_holds_after_it(empty_dir, monkeypatch,
                                                              capsys):
    """A key taken out is what the models lack, whatever the file read held."""
    _key_is_what_is_missing(monkeypatch)
    config.write_env_file(str(empty_dir / ".env"), {"LLM_API_KEY": "sk-1"})
    importlib.reload(config)
    prompt_list = _answer_setup(monkeypatch, {"LLM_API_KEY": ["-", ""]})
    with pytest.raises(SystemExit, match="needs X_API_KEY"):
        cli.main(["setup"])
    assert prompt_list.count("LLM_API_KEY [set]: ") == 1
    assert prompt_list.count("LLM_API_KEY []: ") == 1
    assert "\nLLM_API_KEY=\n" in (empty_dir / ".env").read_text(encoding="utf-8")
    assert f"wrote {empty_dir / '.env'}" in capsys.readouterr().out


def test_setup_leaves_a_dot_env_nothing_changes_as_it_was(empty_dir, monkeypatch, capsys):
    _key_is_what_is_missing(monkeypatch)
    path = empty_dir / ".env"
    path.write_text("LLM_API_KEY=sk-1 # mine\n", encoding="utf-8")
    importlib.reload(config)
    _answer_setup(monkeypatch, {})
    cli.main(["setup"])
    assert path.read_text(encoding="utf-8") == "LLM_API_KEY=sk-1 # mine\n"
    assert f"{path} is left as it was" in capsys.readouterr().out


def test_setup_asks_nothing_where_the_dot_env_would_not_be_read(empty_dir, monkeypatch):
    prompt_list = _answer_setup(monkeypatch, {})
    git_repo(empty_dir)
    (empty_dir / ".env").write_text("LLM_API_KEY=sk-1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", ".env"], cwd=empty_dir, check=True)
    with pytest.raises(SystemExit, match=r"is tracked by git and would not be read"):
        cli.main(["setup"])
    monkeypatch.setenv(config.NO_ENV_FILE_ENV, "1")
    with pytest.raises(SystemExit, match="turns the loading of .env files off"):
        cli.main(["setup"])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="asks its questions at a terminal"):
        cli.main(["setup"])
    assert prompt_list == []


def test_setup_starts_where_the_settings_refuse_what_the_dot_env_holds(tmp_path):
    """The .env every other subcommand stops at is left unread for setup alone."""
    (tmp_path / ".env").write_text("PRESET=hgih\n", encoding="utf-8")
    env_dict = {name: value for name, value in config.own_child_env(
        {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}).items() if name != config.NO_ENV_FILE_ENV}
    proc_dict = {command: subprocess.run(
        config.own_argv(argv), cwd=tmp_path, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=120, env=env_dict)
        for command, argv in (("setup", ["setup"]),
                              ("history", ["history", "--project", "."]))}
    assert proc_dict["setup"].returncode == 1
    assert "setup: asks its questions at a terminal" in proc_dict["setup"].stderr
    assert "PRESET must be one of" in proc_dict["history"].stderr


def _held_dot_env(path: pathlib.Path, text: str) -> None:
    """Write a .env config leaves unread, as it does for setup."""
    path.write_text(text, encoding="utf-8")
    codetwine_loom.hold_env_file(str(path))
    importlib.reload(config)
    assert config.ENV_FILE_SKIP_STATE == config.ENV_FILE_HOLD


def test_setup_asks_again_for_a_value_of_the_dot_env_the_settings_refuse(
        empty_dir, monkeypatch, capsys):
    """Enter keeps the value as a typed answer would give it, checked the same way."""
    _key_is_what_is_missing(monkeypatch)
    _held_dot_env(empty_dir / ".env", "LLM_API_KEY=sk-1\nPRESET=hgih\nTEST_PATH=/abs\n")
    prompt_list = _answer_setup(monkeypatch, {"PRESET": ["", "low"],
                                              "TEST_PATH": ["", "tests"]})
    cli.main(["setup"])
    assert prompt_list.count("PRESET [hgih]: ") == 2
    assert prompt_list.count("TEST_PATH [/abs]: ") == 2
    out = capsys.readouterr().out
    assert "PRESET must be one of low / medium / high: hgih" in out and "TEST_PATH " in out
    text = (empty_dir / ".env").read_text(encoding="utf-8")
    assert "PRESET=low\n" in text and "TEST_PATH=tests\n" in text


@pytest.mark.parametrize("name, value", [("AGENT_LOG_LEVEL", "loud"),
                                         ("DOC_CREATE_EFFORT", "max")])
def test_setup_stops_before_asking_at_a_refused_setting_no_question_asks_for(
        empty_dir, monkeypatch, name, value):
    """A setting the template names with a skip line counts as one no question asks for."""
    _held_dot_env(empty_dir / ".env", f"LLM_API_KEY=sk-1\n{name}={value}\n")
    prompt_list = _answer_setup(monkeypatch, {})
    with pytest.raises(SystemExit) as e:
        cli.main(["setup"])
    assert str(e.value).startswith(f"setup: {empty_dir / '.env'}: {name} must be")
    assert prompt_list == []


def test_setup_takes_out_every_proxy_variable_of_the_dot_env_on_a_no(empty_dir,
                                                                      monkeypatch):
    """A proxy under another name counts as one: Enter is a y and the hosts reached
    without it are asked for."""
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    path = empty_dir / ".env"
    _held_dot_env(path, "LLM_API_KEY=sk-1\nhttp_proxy=http://proxy.example:3128\n")
    prompt_list = _answer_setup(monkeypatch, {"NO_PROXY": "localhost"})
    cli.main(["setup"])
    assert any(prompt.endswith("[Y/n]: ") for prompt in prompt_list)
    text = path.read_text(encoding="utf-8")
    assert "\nNO_PROXY=localhost\n" in text and "http_proxy=" in text
    prompt_list = _answer_setup(monkeypatch, {"Is the internet": ["x", "n"]})
    cli.main(["setup"])
    assert sum(prompt.startswith("Is the internet") for prompt in prompt_list) == 2
    text = path.read_text(encoding="utf-8")
    assert "http_proxy" not in text and "\nNO_PROXY=" not in text
    assert "\n# NO_PROXY=\n" in text


@pytest.mark.parametrize("answer, after, run_proxy", [
    ("", "LLM_API_KEY=sk-1\nhttps_proxy=http://old.example:3128\nno_proxy=old.example\n",
     "http://old.example:3128"),
    ("http://new.example:3128", "LLM_API_KEY=sk-1\nno_proxy=old.example\n"
                                "HTTPS_PROXY=http://new.example:3128\n",
     "http://new.example:3128"),
    ("-", "LLM_API_KEY=sk-1\nno_proxy=old.example\n", "")])
def test_setup_shows_and_writes_the_proxy_a_run_reads_in_either_case(
        empty_dir, monkeypatch, answer, after, run_proxy):
    """A new value or - takes out the name in the other case, and Enter leaves its line."""
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    path = empty_dir / ".env"
    _held_dot_env(path, "LLM_API_KEY=sk-1\nhttps_proxy=http://old.example:3128\n"
                        "no_proxy=old.example\n")
    prompt_list = _answer_setup(monkeypatch, {"HTTPS_PROXY": answer})
    cli.main(["setup"])
    assert "HTTPS_PROXY [http://old.example:3128]: " in prompt_list
    assert ("NO_PROXY [old.example]: " in prompt_list) == bool(run_proxy)
    assert path.read_text(encoding="utf-8") == after
    # A run reading the file takes the proxy answered
    for name in ("LLM_API_KEY", "HTTPS_PROXY", "https_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_ENV_FILE_FILL_DICT", {})
    config._load_env_file(str(path))
    assert urllib.request.getproxies().get("https", "") == run_proxy


def test_setup_checks_the_proxy_a_run_reads_with_the_shell_s(empty_dir, monkeypatch):
    """The name in the other case of the one the shell holds is not read from .env."""
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    path = empty_dir / ".env"
    _held_dot_env(path, "LLM_API_KEY=sk-1\nhttps_proxy=socks5://bad.example:1080\n")
    _answer_setup(monkeypatch, {})
    monkeypatch.setenv("HTTPS_PROXY", "http://good.example:3128")
    cli.main(["setup"])
    assert path.read_text(encoding="utf-8") == (
        "LLM_API_KEY=sk-1\nhttps_proxy=socks5://bad.example:1080\n")


@pytest.mark.parametrize("text, key_line", [
    ("WEB_SEARCH_PROVIDER=tavily\nWEB_SEARCH_API_KEY=tv-search\n"
     "WEB_FETCH_PROVIDER=tavily\nWEB_FETCH_API_KEY=tv-fetch\n",
     "WEB_FETCH_API_KEY=tv-fetch"),
    ("WEB_SEARCH_PROVIDER=model\nWEB_SEARCH_API_KEY=tv-search\n",
     "WEB_SEARCH_API_KEY=tv-search"),
    ("WEB_SEARCH_API_KEY=tv-search\nWEB_FETCH_API_KEY=tv-fetch\n", "")])
def test_setup_keeps_on_enter_the_key_of_a_web_tool_that_needs_none_of_its_own(
        empty_dir, monkeypatch, text, key_line):
    """It is asked for with [set] when the tool names an API, and - takes out the key
    alone."""
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    path = empty_dir / ".env"
    _held_dot_env(path, f"LLM_API_KEY=sk-1\n{text}")
    prompt_list = _answer_setup(monkeypatch, {})
    cli.main(["setup"])
    assert path.read_text(encoding="utf-8") == f"LLM_API_KEY=sk-1\n{text}"
    if not key_line:
        assert not any(prompt.startswith("WEB_") and "_API_KEY [" in prompt
                       for prompt in prompt_list)
        return
    key_name = key_line.partition("=")[0]
    assert f"{key_name} [set]: " in prompt_list
    _answer_setup(monkeypatch, {f"{key_name} [set]": "-"})
    cli.main(["setup"])
    assert path.read_text(encoding="utf-8") == f"LLM_API_KEY=sk-1\n{text}".replace(
        key_line, f"# {key_name}=")


def test_setup_takes_out_a_web_fetch_with_no_search_after_a_line_saying_so(
        empty_dir, monkeypatch, capsys):
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    path = empty_dir / ".env"
    _held_dot_env(path, "LLM_API_KEY=sk-1\nWEB_FETCH_PROVIDER=jina\n"
                        "WEB_FETCH_API_KEY=jn-1\n")
    _answer_setup(monkeypatch, {})
    cli.main(["setup"])
    assert "no WEB_SEARCH_PROVIDER, so WEB_FETCH_PROVIDER and WEB_FETCH_API_KEY are " \
           "taken out" in capsys.readouterr().out
    assert "jn-1" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("text, fetch_key_prompt", [
    ("WEB_SEARCH_PROVIDER=Tavily\nWEB_SEARCH_API_KEY=tv-search\nWEB_FETCH_PROVIDER=tavily\n",
     False),
    ('WEB_SEARCH_PROVIDER=" tavily "\nWEB_SEARCH_API_KEY=tv-search\n'
     "WEB_FETCH_PROVIDER=TAVILY\nWEB_FETCH_API_KEY=tv-fetch\nPRESET=High\n", True)])
@pytest.mark.parametrize("search_answer", ["", "2", "TAVILY"])
def test_setup_reads_a_value_written_in_another_case_as_a_run_reads_it(
        empty_dir, monkeypatch, text, fetch_key_prompt, search_answer):
    """Enter keeps it, the same API answered again keeps its key, and nothing is
    written."""
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    path = empty_dir / ".env"
    _held_dot_env(path, f"LLM_API_KEY=sk-1\n{text}")
    prompt_list = _answer_setup(monkeypatch, {"WEB_SEARCH_PROVIDER [": [search_answer]})
    cli.main(["setup"])
    assert path.read_text(encoding="utf-8") == f"LLM_API_KEY=sk-1\n{text}"
    assert "WEB_SEARCH_PROVIDER [1-4, Enter keeps tavily, - sets none]: " in prompt_list
    assert "WEB_SEARCH_API_KEY [set]: " in prompt_list
    assert "WEB_FETCH_PROVIDER [1-3, Enter keeps tavily, - sets none]: " in prompt_list
    assert ("WEB_FETCH_API_KEY [set]: " in prompt_list) is fetch_key_prompt
    assert not any(prompt.startswith("WEB_") and "[]" in prompt for prompt in prompt_list)


def test_setup_reads_the_search_the_shell_holds_in_another_case_as_a_run_reads_it(
        empty_dir, monkeypatch):
    """The fetch through the same API uses the search's key: no key of its own is asked
    for and its line is kept."""
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    path = empty_dir / ".env"
    _held_dot_env(path, "LLM_API_KEY=sk-1\nWEB_FETCH_PROVIDER=tavily\n")
    prompt_list = _answer_setup(monkeypatch, {})
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "Tavily")
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "tv-search")
    cli.main(["setup"])
    assert path.read_text(encoding="utf-8") == "LLM_API_KEY=sk-1\nWEB_FETCH_PROVIDER=tavily\n"
    assert not any(prompt.startswith("WEB_FETCH_API_KEY") for prompt in prompt_list)


@pytest.mark.parametrize("text, shell_no_proxy", [
    ("HTTPS_PROXY=http://proxy.example:3128\nNO_PROXY=llm.corp.example:8443\n"
     "no_proxy=llm.corp.example\n", ""),
    ("HTTPS_PROXY=http://proxy.example:3128\nNO_PROXY=llm.corp.example:8443\n",
     "llm.corp.example")])
def test_setup_checks_again_the_no_proxy_a_run_reads(empty_dir, monkeypatch, text,
                                                     shell_no_proxy):
    """A NO_PROXY a run does not read is not refused, and on Enter its line is kept."""
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    path = empty_dir / ".env"
    _held_dot_env(path, f"LLM_API_KEY=sk-1\n{text}")
    prompt_list = _answer_setup(monkeypatch, {})
    monkeypatch.setenv("LLM_API_BASE", "https://llm.corp.example:8443/v1")
    if shell_no_proxy:
        monkeypatch.setenv("NO_PROXY", shell_no_proxy)
    cli.main(["setup"])
    assert path.read_text(encoding="utf-8") == f"LLM_API_KEY=sk-1\n{text}"
    assert "NO_PROXY []: " not in prompt_list


@pytest.mark.parametrize("answer, after_dict", [
    ("localhost,llm.corp.example", {"NO_PROXY": "localhost,llm.corp.example"}),
    ("", {})])
def test_the_no_proxy_asked_again_takes_the_place_of_the_name_in_every_case(
        empty_dir, monkeypatch, answer, after_dict):
    entry_dict = {entry.name: entry for entry in cli._ask_entry_list()}
    _answer_setup(monkeypatch, {"NO_PROXY []": answer})
    monkeypatch.setenv("LLM_API_BASE", "https://llm.corp.example:8443/v1")
    answer_dict = {"LLM_API_KEY": "sk-1", "HTTPS_PROXY": "http://proxy.example:3128",
                   "NO_PROXY": "localhost", "no_proxy": "llm.corp.example:8443",
                   "No_Proxy": "x.example"}
    assert cli._check_no_proxy_again(entry_dict, answer_dict) == {
        "LLM_API_KEY": "sk-1", "HTTPS_PROXY": "http://proxy.example:3128", **after_dict}


@pytest.mark.parametrize("name, value, shown", [
    ("HTTPS_PROXY", "http://svc:secretpw@proxy:3128", "http://svc:****@proxy:3128"),
    ("HTTPS_PROXY", "svc:secretpw@proxy:3128", "svc:****@proxy:3128"),
    ("HTTPS_PROXY", "http://u:sec@retpw@proxy:3128", "http://u:****@proxy:3128"),
    ("HTTPS_PROXY", "http://u@x:secretpw@proxy:3128", "http://u@x:****@proxy:3128"),
    ("HTTPS_PROXY", "http://u:sec%40retpw@proxy:3128", "http://u:****@proxy:3128"),
    ("HTTPS_PROXY", "HTTPS://u:secretpw@proxy:3128/", "HTTPS://u:****@proxy:3128/"),
    ("HTTPS_PROXY", "http://svc@proxy:3128", "http://svc@proxy:3128"),
    ("HTTPS_PROXY", "http://proxy:3128", "http://proxy:3128"),
    ("WEB_SEARCH_API_KEY", "tv-1", "set"),
    ("PRESET", "", "medium")])
def test_a_question_shows_no_key_and_no_password_of_the_dot_env(name, value, shown):
    assert cli._show_value(name, value, "medium") == shown


def test_the_first_run_checks_the_hosts_reached_without_the_proxy_with_its_answers(
        empty_dir, monkeypatch):
    """The proxy and the LLM_API_BASE answered before are the ones checked against."""
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    seen = []
    monkeypatch.setattr(cli, "_ask_no_proxy",
                        lambda entry, https_proxy, api_base, current: seen.append(
                            (https_proxy, api_base)) or "")
    monkeypatch.setattr(cli, "_ask_proxy_url",
                        lambda entry, current: "http://ctloom-proxy:3128")
    monkeypatch.setattr(cli, "_ask_ca_file", lambda entry, current: "")
    monkeypatch.setattr(cli.progress, "ask", lambda question, is_secret=False: "y")
    entry_dict = {e.name: e for e in config.env_template_entry_list()}
    assert cli._ask_proxy(entry_dict, "http://llm.corp.example/v1", {}) == {
        "HTTPS_PROXY": "http://ctloom-proxy:3128", "NO_PROXY": "", "SSL_CERT_FILE": ""}
    assert seen == [("http://ctloom-proxy:3128", "http://llm.corp.example/v1")]


@pytest.mark.parametrize("shell", ["http://127.0.0.1:9", ""])
def test_the_first_run_checks_the_hosts_reached_without_the_proxy_with_what_a_run_reads(
        empty_dir, monkeypatch, shell):
    """LLM_API_BASE the shell holds is the one a run reads, whatever was answered."""
    monkeypatch.setenv("LLM_API_BASE", shell)
    seen = []
    monkeypatch.setattr(cli, "_ask_proxy",
                        lambda entry_dict, api_base, current_dict:
                        seen.append(api_base) or {})
    _key_is_what_is_missing(monkeypatch)
    _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1",
                               "LLM_API_BASE": "http://llm.corp.example:8080/v1"})
    cli._ask_template({})
    assert seen == [shell or "http://llm.corp.example:8080/v1"]


def test_the_hosts_reached_without_the_proxy_are_checked_again_after_the_last_question(
        empty_dir, monkeypatch, capsys):
    """LLM_API_BASE answered only when the models are found to need it makes an answer
    taken before it refused, and that answer is asked for again."""
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    for name in list(os.environ):
        if config._ENDPOINT_NAME_RE.fullmatch(name):
            monkeypatch.delenv(name)
    monkeypatch.delenv("LLM_API_BASE", raising=False)
    monkeypatch.setattr(cli.llm, "missing_credential_list",
                        lambda: [] if config.LLM_API_BASE else _missing_list("LLM_API_BASE"))
    answer_dict = {"LLM_API_BASE": ["", "https://llm.corp.example:8443/v1"],
                   cli._PROXY_QUESTION: ["y"], "HTTPS_PROXY": ["http://ctloom-proxy:3128"],
                   "NO_PROXY": ["localhost,llm.corp.example:8443", "localhost,llm.corp.example"]}
    asked = []

    def answer(question, is_secret=False):
        name = question.split(" [")[0].strip()
        asked.append(name)
        return answer_dict[name].pop(0) if answer_dict.get(name) else ""
    monkeypatch.setattr(cli.progress, "ask", answer)
    try:
        written = cli._ask_template({})
    finally:
        importlib.reload(config)
    assert written["LLM_API_BASE"] == "https://llm.corp.example:8443/v1"
    assert written["NO_PROXY"] == "localhost,llm.corp.example"
    assert asked.count("NO_PROXY") == 2 and asked.count("LLM_API_BASE") == 2
    assert asked[-2:] == ["LLM_API_BASE", "NO_PROXY"]
    out = capsys.readouterr().out
    assert "\nNO_PROXY is not read alike by every HTTP client of this tool for " \
           "LLM_API_BASE" in out and "llm.corp" not in out
    assert "LLM_API_BASE" not in os.environ and "NO_PROXY" not in os.environ
    assert config.LLM_API_BASE == ""


@pytest.mark.parametrize("shell_search, shell_key, is_key_asked", [
    ("model", "", True), ("tavily", "k-shell", False)])
def test_the_fetch_key_is_asked_by_the_search_a_run_reads(
        empty_dir, monkeypatch, shell_search, shell_key, is_key_asked):
    """The search the shell holds is the one a run reads: the page reading API is asked
    for its key unless that search names it with a key."""
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", shell_search)
    monkeypatch.setenv("WEB_SEARCH_API_KEY", shell_key)
    _key_is_what_is_missing(monkeypatch)
    asked = _answer_with(monkeypatch, {
        "LLM_API_KEY": "sk-1",
        "WEB_SEARCH_PROVIDER": "tavily" if shell_search == "model" else "1",
        "WEB_SEARCH_API_KEY": "k-answer", "WEB_FETCH_PROVIDER": "tavily",
        "WEB_FETCH_API_KEY": "k-fetch"})
    try:
        with pytest.raises(SystemExit, match="start-up"):
            cli.main(["edit", "--project", "p", "--request", "r"])
        text = (empty_dir / ".env").read_text(encoding="utf-8")
    finally:
        (empty_dir / ".env").unlink(missing_ok=True)
        for name in (*_SEARCH_NAME_TUPLE, "LLM_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        importlib.reload(config)
    assert ("WEB_FETCH_API_KEY" in asked) == is_key_asked
    assert ("\nWEB_FETCH_API_KEY=k-fetch\n" in text) == is_key_asked


_WARM_QUESTION = {entry.name: entry for entry in config.env_template_entry_list()}[
    "WARM_MAX_REFILL"].question


@pytest.mark.parametrize("shell_dict, left_tuple, line", [
    ({"PROJECT_DIR": "/shell/repo"}, ("PROJECT_DIR",),
     "PROJECT_DIR is set in the environment and is left as it is"),
    ({"WARM_MAX_REFILL": "2"}, ("WARM_MAX_REFILL",),
     "WARM_MAX_REFILL is set in the environment and is left as it is"),
    ({"WEB_SEARCH_API_KEY": "k-shell"}, ("WEB_SEARCH_PROVIDER", "WEB_SEARCH_API_KEY"),
     "WEB_SEARCH_API_KEY is set in the environment, so WEB_SEARCH_PROVIDER and "
     "WEB_SEARCH_API_KEY are left as they are"),
    ({"WEB_SEARCH_PROVIDER": "model"}, ("WEB_SEARCH_PROVIDER", "WEB_SEARCH_API_KEY"),
     "WEB_SEARCH_PROVIDER is set in the environment, so WEB_SEARCH_PROVIDER and "
     "WEB_SEARCH_API_KEY are left as they are"),
    ({"WEB_FETCH_PROVIDER": "jina", "WEB_FETCH_API_KEY": "k-shell"},
     ("WEB_FETCH_PROVIDER", "WEB_FETCH_API_KEY"),
     "WEB_FETCH_PROVIDER and WEB_FETCH_API_KEY are set in the environment, so "
     "WEB_FETCH_PROVIDER and WEB_FETCH_API_KEY are left as they are")])
def test_the_first_run_does_not_ask_for_a_setting_the_shell_holds(empty_dir, monkeypatch, capsys,
                                                    shell_dict, left_tuple, line):
    """A run reads the shell's value over the .env's, so the setting is not asked for
    nor written, and a web tool's settings are left together."""
    for name, value in shell_dict.items():
        monkeypatch.setenv(name, value)
    try:
        importlib.reload(config)
        text, asked = _first_run_env_text(empty_dir, monkeypatch, {
            "PROJECT_DIR": "/answer/repo", _WARM_QUESTION: "n",
            "WEB_SEARCH_PROVIDER": "tavily", "WEB_SEARCH_API_KEY": "k-answer",
            "WEB_FETCH_PROVIDER": "jina", "WEB_FETCH_API_KEY": "k-fetch"})
    finally:
        monkeypatch.delenv("PROJECT_DIR", raising=False)
        importlib.reload(config)
    assert line in capsys.readouterr().out
    for name in left_tuple:
        assert name not in asked and f"\n{name}=" not in text
    assert (_WARM_QUESTION in asked) == ("WARM_MAX_REFILL" not in left_tuple)
    assert ("\nPROJECT_DIR=/answer/repo\n" in text) == ("PROJECT_DIR" not in left_tuple)
    assert "MEDIUM_MODEL" in asked and "\nLLM_API_KEY=sk-1\n" in text


@pytest.mark.parametrize("shell_preset", ["", "medium"])
def test_the_first_run_asks_for_a_preset_only_the_command_line_gives(
        empty_dir, monkeypatch, capsys, shell_preset):
    """--preset is not a value the shell holds: PRESET is asked for and the answer
    written unless the shell holds one, and the command runs with --preset."""
    if shell_preset:
        monkeypatch.setenv("PRESET", shell_preset)
    else:
        monkeypatch.delenv("PRESET", raising=False)
    _key_is_what_is_missing(monkeypatch)
    asked = _answer_with(monkeypatch, {"LLM_API_KEY": "sk-1", "PRESET": "low"})
    try:
        importlib.reload(config)
        with pytest.raises(SystemExit, match="start-up"):
            cli.main(["edit", "--preset", "high", "--no-overview", "--project", "p",
                      "--request", "r"])
        preset = config.PRESET
        text = (empty_dir / ".env").read_text(encoding="utf-8")
    finally:
        importlib.reload(config)
    assert preset == "high"
    assert ("PRESET" in asked) == (not shell_preset)
    assert ("\nPRESET=low\n" in text) == (not shell_preset)
    assert ("PRESET is set in the environment and is left as it is"
            in capsys.readouterr().out) == bool(shell_preset)


@pytest.mark.parametrize("answer", [
    "http://proxy.corp.example:3128", "http://me:pw@proxy.corp.example:3128/",
    "http://proxy.corp.example", "https://proxy.corp.example:3128", "10.1.2.3:8080", "",
    "http://me:p%23w@proxy.corp.example:3128", "http://[::1]:3128", "[::1]:3128",
    "HTTP://Proxy.Corp.Example:3128", "https://proxy.corp.example",
    "http://proxy.bücher.example:3128", "http://xn--bcher-kva.example:3128",
    "http://xn--zz.example:3128", "me:p%5Bw%5D@proxy.corp.example:3128"])
def test_a_proxy_s_url_or_host_and_port_is_taken_as_it_is_typed(empty_dir, monkeypatch,
                                                              answer):
    entry = {e.name: e for e in config.env_template_entry_list()}["HTTPS_PROXY"]
    monkeypatch.setattr(cli.progress, "ask", lambda question, is_secret=False: answer)
    assert cli._ask_proxy_url(entry) == answer


def test_a_ca_file_no_container_can_be_given_is_asked_again(empty_dir, monkeypatch,
                                                            capsys):
    """The same check the commands make before they start a container."""
    ca_dir = empty_dir / "ca:dir"
    ca_dir.mkdir()
    (ca_dir / "ca.pem").write_text(CERTIFICATE_PEM, encoding="utf-8")
    good_file = empty_dir / "ca.pem"
    good_file.write_text(CERTIFICATE_PEM, encoding="utf-8")
    entry = {e.name: e for e in config.env_template_entry_list()}["SSL_CERT_FILE"]
    answer_list = ["ca:dir/ca.pem", "ca.pem"]
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: answer_list.pop(0))
    assert cli._ask_ca_file(entry) == str(good_file)
    assert "cannot be given to the container that fetches dependencies" in (
        capsys.readouterr().out)


def test_the_answers_tried_out_leave_the_environment_as_it_was(empty_dir, monkeypatch):
    """The certificate file among them, and the variable that follows it."""
    ca_file = empty_dir / "ctloom-ca.pem"
    ca_file.write_text(CERTIFICATE_PEM, encoding="utf-8")
    monkeypatch.setattr(cli.llm, "missing_credential_list", lambda: [])
    for name in (config.CA_FILE_ENV, config.REQUESTS_CA_FILE_ENV, "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
    environ_before = dict(os.environ)
    assert cli._lack_with_answer({config.CA_FILE_ENV: str(ca_file),
                                  "HTTPS_PROXY": "http://proxy.corp.example:3128"}) == []
    assert dict(os.environ) == environ_before


def test_the_proxy_question_names_what_is_reached_through_it():
    for word in ("model's API", "web APIs", "package registries"):
        assert word in cli._PROXY_QUESTION


def test_a_file_of_certificate_authorities_that_is_not_there_is_asked_again(
        empty_dir, monkeypatch, capsys):
    ca_file = empty_dir / "home" / "ctloom-ca.pem"
    ca_file.parent.mkdir()
    ca_file.write_text(CERTIFICATE_PEM, encoding="utf-8")
    (empty_dir / "ctloom-ca.cer").write_bytes(b"0\x82\x01\x82\x30\x82")
    monkeypatch.setenv("HOME", str(ca_file.parent))
    answer_iter = iter(["no-such.pem", "ctloom-ca.cer", "~/ctloom-ca.pem"])
    entry = {e.name: e for e in config.env_template_entry_list()}[config.CA_FILE_ENV]
    monkeypatch.setattr(cli.progress, "ask",
                        lambda question, is_secret=False: next(answer_iter))
    assert cli._ask_ca_file(entry) == str(ca_file)
    out = capsys.readouterr().out
    assert f"{empty_dir / 'no-such.pem'} is not a file" in out
    assert f"{empty_dir / 'ctloom-ca.cer'} holds no PEM certificate (" in out


def test_a_page_reading_api_with_no_key_is_left_unset(empty_dir, monkeypatch):
    preset = sorted(config.WEB_FETCH_PRESET_DICT)[0]
    text, _ = _first_run_env_text(empty_dir, monkeypatch,
                                  {"WEB_SEARCH_PROVIDER": "1", "WEB_FETCH_PROVIDER": preset})
    assert "\n# WEB_FETCH_PROVIDER=\n" in text and "\n# WEB_FETCH_API_KEY=\n" in text

# == --preset: which column of the settings the run reads ==
def test_the_subcommands_that_call_a_model_take_preset():
    for argv in (["edit", "--project", "p", "--request", "r"],
                 ["sandbox", "--project", "p", "--request", "r"],
                 ["run", "--project", "p", "--request", "r"],
                 ["answer", "--project", "p", "--request", "r"],
                 ["integrate", "--project", "p", "--branch", "b"]):
        assert cli._parse_args(argv).preset is None
        assert cli._parse_args(argv + ["--preset", "high"]).preset == "high"


def test_build_takes_no_preset():
    assert not hasattr(cli._parse_args(["build", "--project", "p"]), "preset")


def test_a_preset_outside_the_3_values_is_refused():
    with pytest.raises(SystemExit):
        cli._parse_args(["edit", "--project", "p", "--request", "r", "--preset", "max"])


def test_preset_is_written_into_the_environment_and_read_back(tmp_path, monkeypatch):
    """It goes through the environment so that a container the run starts reads the
    same column. The run stops at the missing repository, after the settings are read
    again."""
    # Set first so that the teardown takes the name away, whatever main() wrote into it
    monkeypatch.setenv("PRESET", "medium")
    monkeypatch.delenv("MAX_PATCH_ROUND", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["edit", "--project", str(tmp_path / "none"), "--request", "r",
                  "--preset", "low"])
    assert os.environ["PRESET"] == "low"
    assert config.MAX_PATCH_ROUND == 3
    monkeypatch.delenv("PRESET")
    importlib.reload(config)
    assert config.MAX_PATCH_ROUND == 5


def test_the_merging_subcommands_take_only_the_no_forms_of_the_stage_flags():
    for argv in (["run", "--project", "p", "--request", "r"],
                 ["integrate", "--project", "p", "--branch", "b"]):
        args = cli._parse_args(argv)
        assert (args.test_create, args.refactor, args.test_refactor) == (None, None, None)
        args = cli._parse_args(argv + ["--no-test-create", "--no-test-refactor"])
        assert (args.test_create, args.refactor, args.test_refactor) == (False, None, False)
        for flag in ("--test-create", "--refactor", "--test-refactor"):
            with pytest.raises(SystemExit):
                cli._parse_args(argv + [flag])


def test_a_subcommand_that_merges_nothing_takes_no_stage_flag():
    assert not hasattr(cli._parse_args(["sandbox", "--project", "p", "--request", "r"]),
                       "test_create")


def test_the_stage_flags_are_written_into_the_environment_and_read_back(tmp_path,
                                                                        monkeypatch):
    """A flag given turns its setting off; one not given leaves it as it was.
    The run stops at the missing repository, after the settings are read again."""
    # Set first so that the teardown takes the names away, whatever main() wrote
    for name in ("TEST_CREATE", "REFACTOR", "TEST_REFACTOR"):
        monkeypatch.delenv(name)
    importlib.reload(config)
    with pytest.raises(SystemExit):
        cli.main(["run", "--project", str(tmp_path / "none"), "--request", "r",
                  "--output-dir", str(tmp_path / "out"), "--no-refactor"])
    assert os.environ["REFACTOR"] == "false"
    assert "TEST_CREATE" not in os.environ and "TEST_REFACTOR" not in os.environ
    assert (config.TEST_CREATE, config.REFACTOR, config.TEST_REFACTOR) == (True, False, True)
    monkeypatch.delenv("REFACTOR")
    importlib.reload(config)


# == run: reading the request and handing it out ========
def test_run_needs_a_request_or_a_branch_that_says_what_it_was(tmp_path):
    """It can be left out only when there are branches to read it off."""
    args = cli._parse_args(["run", "--project", "p"])
    assert args.request == ""
    with pytest.raises(SystemExit):
        cli.request_of_run(str(tmp_path), "", [])


def test_the_request_passed_is_the_one_used(monkeypatch, tmp_path):
    """A run carrying on from what an earlier one left can be given a narrower
    request than that one was."""
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda *a: "the whole of it")
    assert cli.request_of_run(str(tmp_path), "just the rest", ["run/1/1"]) == \
        "just the rest"


def test_the_request_is_read_off_the_branches_when_none_was_passed(monkeypatch,
                                                                  tmp_path):
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda directory, branch_list: "add --version to main.py")
    assert cli.request_of_run(str(tmp_path), "", ["run/1/1"]) == "add --version to main.py"


def test_branches_that_say_nothing_about_the_request_are_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list", lambda *a: "")
    with pytest.raises(SystemExit):
        cli.request_of_run(str(tmp_path), "", ["mine"])


def test_a_branch_that_is_not_there_is_refused_where_it_is_read(monkeypatch, tmp_path):
    """Nothing is started before this, as the same name checked inside the run is."""
    def missing(*a):
        raise RuntimeError("no branch named gone")

    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list", missing)
    with pytest.raises(SystemExit):
        cli.request_of_run(str(tmp_path), "", ["gone"])


def test_a_run_given_only_the_branches_gets_as_far_as_the_agents(monkeypatch):
    """The whole command line, not the reading alone: the request read off the branches
    is what the dispatching run is handed."""
    seen = {}
    result = {"tasks": [], "exit_status": 0, "swept": [], "integration": None,
              "branches": [], "dropped": []}
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda directory, branch_list: "add --version to main.py")
    monkeypatch.setattr(cli.sandbox_mod, "dispatch",
                        lambda *a, **k: seen.update(args=a, resume=k["resume_list"])
                        or result)
    monkeypatch.setattr(cli.report_mod, "report", lambda *a, **k: "")
    with pytest.raises(SystemExit) as left:
        cli.main(["run", "--project", ".", "--resume", "run/1/1"])
    assert left.value.code == 0
    assert seen["args"][2] == "add --version to main.py"
    assert seen["resume"] == ["run/1/1"]


def test_a_run_given_a_blank_request_is_refused_whatever_branches_it_names(monkeypatch):
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda *a: pytest.fail("the branches were read"))
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--project", ".", "--request", "  ", "--resume", "run/1/1"])
    assert str(e.value).startswith("run: the request is empty. Pass it with --request")


def test_run_takes_no_dry_run():
    """It writes nothing itself, so there is nothing to hold back. Each agent it
    starts is given that on its own."""
    args = cli._parse_args(["run", "--request", "どこで読んでいる?"])
    assert args.command == "run" and args.request == "どこで読んでいる?"
    assert not hasattr(args, "dry_run")


def test_the_branches_to_carry_on_from_are_named_one_by_one():
    args = cli._parse_args(["run", "--request", "r", "--resume", "codetwine-loom/one",
                            "--resume", "codetwine-loom/two"])
    assert args.resume_list == ["codetwine-loom/one", "codetwine-loom/two"]


def test_carrying_on_from_nothing_is_the_default():
    assert cli._parse_args(["run", "--request", "r"]).resume_list == []


def test_a_change_in_a_container_is_not_asked_whether_to_resume():
    """The question is asked at a terminal, and there is none in a container. What an
    earlier run left reaches one through run's --resume instead."""
    with pytest.raises(SystemExit):
        cli._parse_args(["sandbox", "--request", "r", "--resume"])
    assert not hasattr(cli._parse_args(["sandbox", "--request", "r"]), "resume")
    assert cli._parse_args(["edit", "--request", "r", "--resume"]).resume is True


def test_a_question_is_its_own_report(capsys, monkeypatch):
    """The answer was printed above it, so nothing is said about it a second time and
    no model is called to say it."""
    monkeypatch.setattr(cli.report_mod, "report",
                        lambda *a, **k: pytest.fail("a report was built"))
    cli.print_report("/repo", "where is it read?", {"answer": "config.py",
                                                    "exit_status": 0})
    assert capsys.readouterr().out == ""


def test_a_report_that_could_not_be_written_prints_nothing(capsys, monkeypatch):
    monkeypatch.setattr(cli.report_mod, "report", lambda *a, **k: "")
    cli.print_report("/repo", "change it", {"branches": []})
    assert capsys.readouterr().out == ""


def test_a_run_that_failed_is_reported_like_any_other(capsys, monkeypatch):
    """The exit status says what the agents did; it is not moved by whether the run
    could be described, and a failed run is what a report is most needed for."""
    result = {"tasks": ["one"], "dropped": [], "integration": None, "resume": [],
              "swept": [], "exit_status": 1,
              "branches": [{"n": 1, "request": "one", "branch": "run/1/1",
                            "attempt": 1, "commit": "", "exit_status": 1,
                            "error": "the container was killed", "verdict": "failed"}]}
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", lambda *a, **k: result)
    monkeypatch.setattr(cli.report_mod, "report", lambda *a, **k: "nothing landed")
    with pytest.raises(SystemExit) as left:
        cli.main(["run", "--request", "one", "--project", "."])
    assert left.value.code == 1
    assert "nothing landed" in capsys.readouterr().out


def test_the_report_is_given_the_repository_the_run_was_against(monkeypatch,
                                                               tmp_path):
    """It reads the branches the run left, so it needs that repository as a path git
    can be run in rather than the relative one the command line was given."""
    seen = {}
    result = {"tasks": ["one"], "dropped": [], "integration": None, "resume": [],
              "swept": [], "exit_status": 0,
              "branches": [{"n": 1, "request": "one", "branch": "run/1/1",
                            "attempt": 1, "commit": "abc", "exit_status": 0,
                            "error": "", "verdict": "passed"}]}
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", lambda *a, **k: result)
    monkeypatch.setattr(cli.report_mod, "report",
                        lambda *a, **k: seen.update(args=a) or "")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["run", "--request", "change it", "--project", "repo"])
    assert seen["args"] == (str(tmp_path / "repo"), "change it", result)


def test_the_line_that_says_how_to_carry_on_names_every_branch(capsys, tmp_path):
    cli.print_resume_line(str(tmp_path), {"command": "run",
                                          "resume": ["run/1/1", "run/1/2"]})
    out = capsys.readouterr().out
    assert out == ("to carry on from what this run left: run --resume run/1/1 "
                   "--resume run/1/2\n")


def test_a_run_that_left_nothing_to_carry_on_from_says_nothing(capsys, tmp_path):
    cli.print_resume_line(str(tmp_path), {"command": "run", "resume": []})
    assert capsys.readouterr().out == ""


def test_answer_takes_the_agents_answers_and_where_to_write_its_own():
    args = cli._parse_args(["answer", "--request", "q"])
    assert args.answer_in == "" and args.answer_out == ""
    args = cli._parse_args(["answer", "--request", "q", "--answered", "in.json",
                            "--answer-out", "out.json"])
    assert args.answer_in == "in.json" and args.answer_out == "out.json"


def test_a_question_is_printed_with_every_agent_and_the_answer(capsys):
    cli.print_dispatch("/repo", {
        "questions": ["where?", "how?"], "answer": "both, together",
        "combined": {"answer": "both, together", "exit_status": 0, "error": ""},
        "swept": [], "exit_status": 1,
        "answers": [{"n": 1, "request": "where?", "attempt": 1, "answer": "here",
                     "exit_status": 0, "error": ""},
                    {"n": 2, "request": "how?", "attempt": 2, "answer": "",
                     "exit_status": 1, "error": "handed no answer back"}]})
    out = capsys.readouterr().out
    assert "questions: 2" in out
    assert "1. answered" in out and "where?" in out
    assert "2. no answer, try 2" in out and "it did not finish: handed no answer back" in out
    assert out.rstrip().endswith("answer:\n  both, together")
    assert "combined: nothing" not in out


def test_a_question_whose_combining_run_failed_says_so_before_the_answers(capsys):
    cli.print_dispatch("/repo", {
        "questions": ["where?"], "answer": "1. where?\n   answered: here",
        "combined": {"answer": "", "exit_status": 1,
                     "error": "the run that answers handed no answer back"},
        "swept": [], "exit_status": 1,
        "answers": [{"n": 1, "request": "where?", "attempt": 1, "answer": "here",
                     "exit_status": 0, "error": ""}]})
    out = capsys.readouterr().out
    assert "combined: nothing (the run that answers handed no answer back)" in out
    assert "printed as they came" in out and "answered: here" in out


def _one_task_result(runs: int) -> dict:
    """Return a dispatch result for one task that got through, made in runs runs."""
    return {"tasks": ["one"], "dropped": [], "integration": None, "resume": [],
            "swept": [], "exit_status": 0, "runs": runs,
            "branches": [{"n": 1, "request": "one", "branch": "run/1/1",
                          "attempt": 1, "commit": "c" * 40, "exit_status": 0,
                          "error": "", "verdict": "unverified"}]}


def test_a_request_that_was_run_again_says_how_many_runs_first(capsys):
    cli.print_dispatch("/repo", _one_task_result(2))
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "runs: 2 (the runs before the last did not get through)"
    assert lines[1] == "tasks: 1"


def test_a_request_run_again_names_the_files_the_runs_before_the_last_put_in(capsys):
    earlier_list = [{"files": 2, "paths": ["a.py", "b.py"], "patch": "p1",
                     "last": "/out/land-last.diff"},
                    {"files": 1, "paths": ["b.py"], "patch": "p2",
                     "last": "/out/land-last.diff"}]
    cli.print_dispatch("/repo", {**_one_task_result(3), "earlier_land": earlier_list})
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "runs: 3 (the runs before the last did not get through)",
        "  the runs before the last put 2 file(s) into the working tree, which are "
        "still there: a.py, b.py",
        "tasks: 1"]
    cli.print_dispatch("/repo", {"unclear": ["which?"], "runs": 3,
                                 "earlier_land": earlier_list})
    assert "the runs before the last put" not in capsys.readouterr().out


def test_a_request_run_once_says_nothing_about_runs(capsys):
    cli.print_dispatch("/repo", _one_task_result(1))
    assert "runs:" not in capsys.readouterr().out


@pytest.mark.parametrize("status, why", [
    (1, "exit status 1"),
    (137, "exit status 137, the status of a container killed when its processes reach "
          "SANDBOX_MEMORY (2g)"),
    (124, "exit status 124, the status of a container taken away at SANDBOX_TIMEOUT_SEC "
          "(3600 s)")])
def test_a_task_or_question_that_did_not_finish_says_what_its_exit_status_is(
        capsys, monkeypatch, status, why):
    monkeypatch.setattr(config, "SANDBOX_MEMORY", "2g")
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SEC", 3600)
    task_dict = _one_task_result(1)
    task_dict["branches"][0]["exit_status"] = status
    cli.print_dispatch("/repo", task_dict)
    cli.print_dispatch("/repo", {
        "questions": ["where?"], "answer": "1. where?\n   no answer came back",
        "combined": {"answer": "", "exit_status": 0, "error": ""},
        "swept": [], "exit_status": status,
        "answers": [{"n": 1, "request": "where?", "attempt": 1, "answer": "",
                     "exit_status": status, "error": ""}]})
    out = capsys.readouterr().out
    assert out.count(f"     it did not finish: {why}\n") == 2


def test_run_runs_in_a_container_unless_here_is_passed():
    """--here is what the container itself is given, as it is for integrate."""
    assert cli._parse_args(["run", "--request", "r"]).here is False
    assert cli._parse_args(["run", "--request", "r", "--here"]).here is True


def test_the_split_is_written_where_the_host_reads_it(tmp_path):
    """The container and the host share this one file, as the change runs share the
    patch, so it is written even when the directory is not there yet."""
    out = tmp_path / "sandbox" / "tasks-1.json"
    cli.write_split(str(out), {"ok": True, "tasks": ["one", "two"]})
    assert json.loads(out.read_text(encoding="utf-8")) == {"tasks": ["one", "two"]}


def test_the_definitions_each_task_names_are_written_beside_the_tasks(tmp_path):
    out = tmp_path / "tasks-1.json"
    cli.write_split(str(out), {"ok": True, "tasks": ["one"], "defs": [["a.py::f"]]})
    assert json.loads(out.read_text(encoding="utf-8")) == {"tasks": ["one"],
                                                           "defs": [["a.py::f"]]}


def test_questions_are_written_as_questions(tmp_path):
    out = tmp_path / "tasks-1.json"
    cli.write_split(str(out), {"ok": True, "questions": ["where?"]})
    assert json.loads(out.read_text(encoding="utf-8")) == {"questions": ["where?"]}


def test_an_answer_is_written_where_the_host_reads_it(tmp_path):
    out = tmp_path / "sandbox" / "answer-1.json"
    cli.write_answer(str(out), {"ok": True, "answer": "config.py", "error": ""})
    assert json.loads(out.read_text(encoding="utf-8")) == {"answer": "config.py"}


def test_a_run_that_answered_nothing_is_written_with_the_reason(tmp_path):
    out = tmp_path / "answer-1.json"
    cli.write_answer(str(out), {"ok": False, "answer": None,
                                "error": "no COMPLETE() within 20 turns"})
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "error": "no COMPLETE() within 20 turns"}


def test_the_agents_answers_are_read_back_as_a_list(tmp_path):
    path = tmp_path / "answered-1.json"
    path.write_text('[{"n": 1, "request": "where?", "answer": "here"}]',
                    encoding="utf-8")
    assert cli.read_answer_list(str(path)) == [{"n": 1, "request": "where?",
                                             "answer": "here"}]
    path.write_text('{"answer": "here"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="does not hold a list"):
        cli.read_answer_list(str(path))
    with pytest.raises(SystemExit, match="cannot be read"):
        cli.read_answer_list(str(tmp_path / "missing.json"))


def test_a_split_that_failed_is_written_with_the_reason(tmp_path):
    """The host raises on it, and the reason is what says why nothing was started."""
    out = tmp_path / "tasks-1.json"
    cli.write_split(str(out), {"ok": False, "error": "not valid JSON"})
    assert json.loads(out.read_text(encoding="utf-8")) == {"error": "not valid JSON"}


def test_the_questions_are_printed_one_per_line_with_the_turns_it_took(capsys):
    cli.print_split({"ok": True, "iters": 4, "harvested": False,
                     "questions": ["where is it read?", "what happens without it?"]})
    out = capsys.readouterr().out
    assert "questions: 2" in out and "1. where is it read?" in out
    assert "iters: 4" in out


def test_an_answer_is_printed_with_the_turns_it_took(capsys):
    cli.print_answer({"ok": True, "iters": 4, "harvested": False,
                      "answer": "config.py, in read_config", "error": ""})
    out = capsys.readouterr().out
    assert "config.py, in read_config" in out and "iters: 4" in out
    assert "harvested" not in out


def test_a_harvested_answer_says_so(capsys):
    cli.print_answer({"ok": True, "iters": 20, "harvested": True,
                      "answer": "config.py", "error": ""})
    assert "harvested" in capsys.readouterr().out


def test_a_run_that_answered_nothing_says_why(capsys):
    cli.print_answer({"ok": False, "iters": 20, "harvested": False, "answer": None,
                      "error": "the answer stage handed nothing over: no COMPLETE() "
                               "within 20 turns"})
    out = capsys.readouterr().out
    assert "answer: none" in out and "no COMPLETE() within 20 turns" in out


def test_the_tasks_are_printed_one_per_line(capsys):
    cli.print_split({"ok": True, "iters": 6, "harvested": False,
                     "tasks": ["rename read_config", "update its callers"]})
    out = capsys.readouterr().out
    assert "tasks: 2" in out and "1. rename read_config" in out


def test_a_harvested_split_says_so(capsys):
    cli.print_split({"ok": True, "iters": 20, "questions": ["where?"],
                     "harvested": True})
    assert "harvested" in capsys.readouterr().out


# == what a run that handed nothing over is told to have been ==
def test_a_run_that_handed_nothing_over_says_why(capsys):
    cli.print_split({"ok": False, "iters": 20, "harvested": False,
                     "error": "the splitting stage handed nothing over: "
                              "no COMPLETE() within 20 turns"})
    out = capsys.readouterr().out
    assert "split: none" in out and "no COMPLETE() within 20 turns" in out


def test_a_run_that_used_every_turn_is_reported_as_that(capsys):
    print(cli.no_answer_line({"error": ""}, 20, "COMPLETE"))
    assert "no COMPLETE() within" in capsys.readouterr().out


def test_a_run_that_stopped_early_is_not_reported_as_out_of_turns(capsys):
    """The loop tells the two apart in "error"; reporting the one as the other sends
    whoever reads it after a turn limit that was never reached."""
    print(cli.no_answer_line({"error": "RuntimeError: the broker did not answer"},
                             20, "COMPLETE"))
    out = capsys.readouterr().out
    assert "the broker did not answer" in out and "within" not in out


def _judged(verdict: str, why: str = "one machine-written line") -> dict:
    """Return the two keys run_edit() adds from history.judge()."""
    return {"verdict": verdict, "why": why}


def test_the_editing_run_says_the_same_two_things(capsys):
    stat = {"files": [], "untracked": [], "total_insertions": 0, "total_deletions": 0}
    cli.print_result({"iters": 2, "answer": None, "harvested": False,
                      "error": "RuntimeError: the broker did not answer",
                      **_judged(history.VERDICT_FAIL)}, stat)
    assert "the broker did not answer" in capsys.readouterr().out
    cli.print_result({"iters": 20, "answer": None, "harvested": False, "error": "",
                      **_judged(history.VERDICT_FAIL)}, stat)
    assert "no COMPLETE() within" in capsys.readouterr().out


# == --dry-run: the flag alone ===============================
@pytest.mark.parametrize("command", ["edit", "sandbox"])
def test_the_flag_is_what_makes_a_dry_run(command, monkeypatch):
    """What config holds from an earlier run of main() does not turn it on."""
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    base = [command, "--project", "p", "--request", "r"]
    assert cli._parse_args(base).dry_run is False
    assert cli._parse_args([*base, "--dry-run"]).dry_run is True
    assert cli._parse_args([*base, "--no-dry-run"]).dry_run is False


def test_a_dry_run_prints_what_would_have_been_written(capsys, monkeypatch):
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    monkeypatch.setattr(cli.edit, "dry_change_list",
                        lambda: ["--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
                                 "deleted b.py\n"])
    cli.print_result({"iters": 3, "answer": "renamed it", "outcome": COMPLETE_KEY,
                      **_judged(history.VERDICT_NO_PROOF)},
                     {"files": [{"path": "mine.py", "insertions": 1, "deletions": 0}],
                      "untracked": [], "total_insertions": 1, "total_deletions": 0})
    out = capsys.readouterr().out
    assert "nothing was written" in out and "+y" in out and "deleted b.py" in out
    assert "mine.py" not in out


def test_edit_says_when_the_run_ended_without_a_change(capsys):
    """An empty "changed" list would otherwise read as a run that did nothing."""
    cli.print_result({"iters": 3, "answer": "it already returns early",
                      "outcome": COMPLETE_KEY, **_judged(history.VERDICT_NO_CHANGE)},
                     {"files": [], "untracked": [], "total_insertions": 0,
                      "total_deletions": 0})
    out = capsys.readouterr().out
    assert "no change was made" in out and "it already returns early" in out


def test_edit_does_not_say_that_when_a_file_was_written(capsys):
    """It is the verdict that makes it a "no change", not the answer's wording."""
    cli.print_result({"iters": 3, "answer": "renamed it", "outcome": COMPLETE_KEY,
                      **_judged(history.VERDICT_NO_PROOF)},
                     {"files": [{"path": "a.py", "insertions": 2, "deletions": 1}],
                      "untracked": [], "total_insertions": 2, "total_deletions": 1})
    assert "no change was made" not in capsys.readouterr().out


def test_edit_does_not_say_that_against_its_own_verdict(capsys):
    """A run that wrote a file and put it back leaves an empty diff behind, and the
    screen does not say "no change" a line under a verdict saying otherwise. judge() is
    the one that decides."""
    cli.print_result({"iters": 3, "answer": "put it back", "outcome": COMPLETE_KEY,
                      **_judged(history.VERDICT_NO_PROOF)},
                     {"files": [], "untracked": [], "total_insertions": 0,
                      "total_deletions": 0})
    out = capsys.readouterr().out
    assert "no change was made" not in out and "verdict: unverified" in out


def test_the_verdict_is_printed_with_its_reason(capsys):
    """It is what the exit status follows, so the screen says it."""
    cli.print_result({"iters": 3, "answer": "renamed it", "outcome": COMPLETE_KEY,
                      **_judged(history.VERDICT_FAIL, "the tests were failing")},
                     {"files": [], "untracked": [], "total_insertions": 0,
                      "total_deletions": 0})
    assert "verdict: failed (the tests were failing)" in capsys.readouterr().out


def _run_result(result_dict: dict) -> "cli.edit.RunResult":
    """Return a stand-in for what edit.run_test() returns when its command was started."""
    run_result = cli.edit.RunResult(result_dict)
    run_result.is_start = True
    return run_result


def _past_the_start_up(tmp_path, monkeypatch) -> str:
    """Substitute the start-up main() runs before a subcommand, and return the project.

    Returns:
        The project directory main() is to be given.
    """
    class _Knowledge:
        def stamp(self):
            return "1:now"

        def file_count(self):
            return 3

        def names_target(self):
            return True

        def close(self):
            pass

    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: ("k.sqlite", str(project), "proj", True))
    monkeypatch.setattr(cli.Knowledge, "open", staticmethod(lambda path: _Knowledge()))
    monkeypatch.setattr(cli.scope_mod, "load_index", lambda *a, **k: None)
    monkeypatch.setattr(cli.source, "set_project_data", lambda knowledge: None)
    monkeypatch.setattr(cli.turnlog, "open_log", lambda: None)
    monkeypatch.setattr(cli.turnlog, "close", lambda: None)
    monkeypatch.setattr(cli, "print_scope", lambda scope: None)
    # Nothing here writes an index, so main() would build one over _Knowledge
    monkeypatch.setattr(cli.build, "build_and_save", lambda *a, **k: {})
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: True)
    monkeypatch.setattr(cli.sandbox_mod, "live_run_list", lambda *a, **k: [])
    return str(project)


# == the start-up when the analysis or the index is missing =================
def test_a_repository_the_analysis_refuses_stops_with_the_message_not_a_traceback(
        tmp_path, monkeypatch):
    _past_the_start_up(tmp_path, monkeypatch)

    def refuse(*a, **k):
        raise RuntimeError("not a git repository: p. Run `git init` first")

    monkeypatch.setattr(cli, "ensure_knowledge", refuse)
    with pytest.raises(SystemExit) as e:
        cli.main(["search", "--project", "p", "--output-dir", str(tmp_path),
                  "--request", "x"])
    assert "git init" in str(e.value)


def _index_builds(tmp_path, monkeypatch, reused: bool) -> list:
    """Run search past the start-up and return how often the index was built."""
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: ("k.sqlite", project, "proj", reused))
    built = []
    monkeypatch.setattr(cli.build, "build_and_save",
                        lambda *a, **k: built.append(a) or {})
    monkeypatch.setattr("builtins.input",
                        lambda _: pytest.fail("the question was asked"))
    monkeypatch.setattr(cli, "search_scope", lambda request: {})
    cli.main(["search", "--project", project, "--output-dir", str(tmp_path / "out"),
              "--request", "x"])
    return built


def test_a_run_that_just_analysed_builds_the_index_too(tmp_path, monkeypatch):
    assert len(_index_builds(tmp_path, monkeypatch, reused=False)) == 1


def test_a_missing_index_is_built_without_asking(tmp_path, monkeypatch):
    assert len(_index_builds(tmp_path, monkeypatch, reused=True)) == 1


def _index_over(tmp_path, stamp: str) -> None:
    """Write an empty index under out/proj, saying it was built from that analysis."""
    (tmp_path / "out" / "proj").mkdir(parents=True)
    empty = {"nodes": {"defs": {}, "files": {}},
             "edges": {"calls": []},
             "index": {"def": build_bm25({}), "file": build_bm25({})}}
    store.save(str(tmp_path / "out" / "proj" / store.INDEX_FILE), empty, {}, stamp,
               "proj", "")


def test_an_index_that_is_there_is_read_as_it_is(tmp_path, monkeypatch):
    _index_over(tmp_path, "1:now")            # what _past_the_start_up's stamp() says
    assert _index_builds(tmp_path, monkeypatch, reused=True) == []


def test_an_index_from_an_earlier_analysis_is_built_again_without_asking(tmp_path,
                                                                        monkeypatch):
    """analyse leaves the index of the analysis before it, which describes nothing."""
    _index_over(tmp_path, "1:then")
    assert len(_index_builds(tmp_path, monkeypatch, reused=True)) == 1


def _committed(repo) -> None:
    """Turn repo into a git repository with one commit."""
    repo.mkdir(parents=True, exist_ok=True)
    for args in (("init", "--quiet", "-b", "main"),
                 ("config", "user.email", "cli@example.invalid"),
                 ("config", "user.name", "cli")):
        subprocess.run(["git", "-C", str(repo), *args], check=True,
                       capture_output=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "-m", "first"],
                   check=True, capture_output=True)


def test_another_exclude_list_makes_the_analysis_again(tmp_path, monkeypatch, capsys):
    """What --exclude keeps out reaches codetwine only when it runs, so an analysis
    made with another list does not describe what this run asks for."""
    repo = tmp_path / "proj"
    out = tmp_path / "out"
    _committed(repo)
    call_list = []
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: call_list.append(k["exclude_list"]) or
                        write_knowledge_db({"files": []},
                                           str(out / "proj" / knowledge.KNOWLEDGE_FILE),
                                           "proj"))
    monkeypatch.setattr(cli.sandbox_mod, "live_run_list", lambda *a, **k: [])
    monkeypatch.setattr(cli, "print_build", lambda *a: None)
    argv = ["build", "--project", str(repo), "--output-dir", str(out)]
    cli.main(argv + ["--exclude", "vendor"])
    cli.main(argv + ["--exclude", "vendor"])
    cli.main(argv)
    assert call_list == [["vendor"], []]
    capsys.readouterr()


def test_a_subcommand_without_exclude_keeps_out_what_the_analysis_kept_out(
        tmp_path, monkeypatch, capsys):
    """search has no --exclude, so what it keeps out is the record's: a build made
    with --exclude is not made again without it by the next search, and not made
    again with it by the next build either."""
    repo = tmp_path / "proj"
    out = tmp_path / "out"
    _committed(repo)
    call_list = []
    monkeypatch.setattr(knowledge, "run_codetwine",
                        lambda *a, **k: call_list.append(k["exclude_list"]) or
                        write_knowledge_db({"files": []},
                                           str(out / "proj" / knowledge.KNOWLEDGE_FILE),
                                           "proj"))
    monkeypatch.setattr(cli.sandbox_mod, "live_run_list", lambda *a, **k: [])
    monkeypatch.setattr(cli, "print_build", lambda *a: None)
    monkeypatch.setattr(cli, "search_scope", lambda request: {})
    monkeypatch.setattr(cli, "print_scope", lambda scope: None)
    where = ["--project", str(repo), "--output-dir", str(out)]
    cli.main(["build", *where, "--exclude", "vendor"])
    cli.main(["search", *where, "--request", "x"])
    cli.main(["build", *where, "--exclude", "vendor"])
    assert call_list == [["vendor"]]
    record = json.loads((out / "proj" / knowledge.ANALYSIS_COMMIT_FILE).read_text())
    assert record["exclude"] == ["vendor"]
    capsys.readouterr()


def test_the_exclude_list_of_an_analysis_is_read_off_its_record(monkeypatch):
    monkeypatch.setattr(cli.knowledge_mod, "analysis_record",
                        lambda *a: {"commit": "c", "exclude": ["vendor"]})
    assert cli.exclude_of_analysis("k.sqlite", "1:now", ["env"]) == ["vendor"]
    # a record written before the exclude list was kept, and no record at all
    monkeypatch.setattr(cli.knowledge_mod, "analysis_record", lambda *a: {"commit": "c"})
    assert cli.exclude_of_analysis("k.sqlite", "1:now", ["env"]) == ["env"]
    monkeypatch.setattr(cli.knowledge_mod, "analysis_record", lambda *a: {})
    assert cli.exclude_of_analysis("k.sqlite", "1:now", []) == []


# == a build that would replace the analysis stops while a sandboxed run is going ==
@pytest.mark.parametrize("argv, describes", [(["build", "--doc-create"], True),
                                             (["build"], False)])
def test_replacing_the_analysis_beside_a_run_that_is_going_is_refused(tmp_path,
                                                                     monkeypatch,
                                                                     argv, describes):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: describes)
    monkeypatch.setattr(cli.sandbox_mod, "live_run_list",
                        lambda output_dir, name, **k: ["codetwine-loom-one/20260101-000000-1-1"])
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: ("k.sqlite", project, "proj", True)
                        if not k.get("rebuild") else pytest.fail("codetwine was run"))
    with pytest.raises(SystemExit) as e:
        cli.main([*argv, "--project", project, "--output-dir", str(tmp_path / "out")])
    assert "codetwine-loom-one/20260101-000000-1-1" in str(e.value)
    assert "--output-dir" in str(e.value)


@pytest.mark.parametrize("argv", [["search", "--request", "x"], ["run-tests"]])
def test_every_subcommand_that_would_replace_the_analysis_stops_the_same_way(
        tmp_path, monkeypatch, argv):
    """Every agent a run starts copies the analysis, so whoever would replace it waits,
    not build alone."""
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: False)
    monkeypatch.setattr(cli.sandbox_mod, "live_run_list",
                        lambda output_dir, name, **k: ["codetwine-loom-one/20260101-000000-1-1"])
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: ("k.sqlite", project, "proj", True)
                        if not k.get("rebuild") else pytest.fail("codetwine was run"))
    with pytest.raises(SystemExit) as e:
        cli.main([*argv, "--project", project, "--output-dir", str(tmp_path / "out")])
    assert "codetwine-loom-one/20260101-000000-1-1" in str(e.value)


def test_a_build_that_reads_the_analysis_as_it_is_does_not_look_for_runs(
        tmp_path, monkeypatch, capsys):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: True)
    monkeypatch.setattr(cli.sandbox_mod, "live_run_list",
                        lambda *a, **k: pytest.fail("the runs were looked for"))
    monkeypatch.setattr(cli, "print_build", lambda *a: None)
    cli.main(["build", "--project", project, "--output-dir", str(tmp_path / "out")])
    capsys.readouterr()


# == an analysis whose usages name no definition ==============================
class _OldKnowledge:
    """An analysis an older codetwine made, until codetwine is run again."""

    is_old = True

    def stamp(self):
        return "1:now"

    def file_count(self):
        return 3

    def names_target(self):
        return not _OldKnowledge.is_old

    def close(self):
        pass


@pytest.mark.parametrize("argv", [["search", "--request", "x"], ["run-tests"]])
def test_an_analysis_whose_usages_name_no_definition_stops_with_the_message(
        tmp_path, monkeypatch, argv):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(_OldKnowledge, "is_old", True)
    monkeypatch.setattr(cli.Knowledge, "open", staticmethod(lambda path: _OldKnowledge()))
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: ("k.sqlite", project, "proj", True)
                        if not k.get("rebuild") else pytest.fail("codetwine was run"))
    monkeypatch.setattr(cli.build, "build_and_save",
                        lambda *a, **k: pytest.fail("the index was built"))
    with pytest.raises(SystemExit) as e:
        cli.main([*argv, "--project", project, "--output-dir", str(tmp_path / "out")])
    assert str(e.value).startswith(f"{argv[0]}: the analysis at k.sqlite is out of date")
    assert "Run `codetwine-loom build`" in str(e.value)


def test_build_makes_an_analysis_whose_usages_name_no_definition_again(
        tmp_path, monkeypatch, capsys):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(_OldKnowledge, "is_old", True)
    monkeypatch.setattr(cli.Knowledge, "open", staticmethod(lambda path: _OldKnowledge()))
    calls = []

    def fake_ensure(project_dir, output_dir, **kwargs):
        calls.append(kwargs)
        if kwargs.get("rebuild"):
            monkeypatch.setattr(_OldKnowledge, "is_old", False)
        return ("k.sqlite", project, "proj", not kwargs.get("rebuild"))

    monkeypatch.setattr(cli, "ensure_knowledge", fake_ensure)
    monkeypatch.setattr(cli, "print_build", lambda *a: None)
    cli.main(["build", "--project", project, "--output-dir", str(tmp_path / "out")])
    capsys.readouterr()
    assert [kwargs.get("rebuild", False) for kwargs in calls] == [False, True]
    assert calls[1]["with_doc"] is False


# == the analysis is made again when it does not describe the tree ============
def _analysed_again(tmp_path, monkeypatch, argv: list[str], describes: bool) -> list:
    """Run argv past the start-up and return the kwargs of every ensure_knowledge."""
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: describes)
    calls = []

    def fake_ensure(project_dir, output_dir, **kwargs):
        calls.append(kwargs)
        return ("k.sqlite", project, "proj", not kwargs.get("rebuild"))

    monkeypatch.setattr(cli, "ensure_knowledge", fake_ensure)
    monkeypatch.setattr(cli, "search_scope", lambda request: {})
    monkeypatch.setattr(cli, "print_build", lambda *a: None)
    monkeypatch.setattr(cli.edit, "run_test", lambda: _run_result({
        "status": "passed", "output": ""}))
    monkeypatch.setattr(cli, "print_test", lambda *a: None)
    cli.main([*argv, "--project", project, "--output-dir", str(tmp_path / "out")])
    return calls


@pytest.mark.parametrize("is_inside", [True, False])
def test_the_languages_build_output_is_left_out_inside_a_container_alone(
        tmp_path, monkeypatch, is_inside):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: True)
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: ("k.sqlite", project, "proj", True))
    monkeypatch.setattr(cli.edit, "run_test", lambda: _run_result({
        "status": "passed", "failed": [], "failed_excerpt": "", "targets": [],
        "output": ""}))
    monkeypatch.setattr(cli, "refuse_while_run_live", lambda *a, **k: None)
    if is_inside:
        monkeypatch.setenv(sandbox_mod.CTLOOM_CLONE_ENV, project)
    else:
        monkeypatch.delenv(sandbox_mod.CTLOOM_CLONE_ENV, raising=False)
    seen = []
    monkeypatch.setattr(cli.edit, "exclude_build_output",
                        lambda: seen.append(config.PROJECT_DIR) or [])
    cli.main(["run-tests", "--project", project, "--output-dir", str(tmp_path / "out")])
    assert seen == ([project] if is_inside else [])


def test_run_tests_makes_the_tests_directory_and_says_when_nothing_could_run(
        tmp_path, monkeypatch, capsys):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: True)
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: ("k.sqlite", project, "proj", True))
    seen_dict = {}

    def run_test():
        seen_dict["dir"] = os.path.isdir(os.path.join(project, config.TEST_PATH))
        return _run_result({"status": "did not run", "failed": [],
                            "failed_excerpt": "nothing", "targets": [], "output": ""})

    monkeypatch.setattr(cli.edit, "run_test", run_test)
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "spec")
    monkeypatch.setattr(config, "TEST_COMMAND", ["run"])
    monkeypatch.setattr(config, "RUN_TEST", True)
    assert not os.path.isdir(os.path.join(project, "spec"))
    cli.main(["run-tests", "--project", project, "--output-dir", str(tmp_path / "out")])
    assert seen_dict["dir"] is True
    assert "status: did not run" in capsys.readouterr().out


@pytest.mark.parametrize("can_run, own, expect_ext_list", [
    (False, "", ["py", "ts"]), (True, "", ["cbl", "py", "ts"]),
    (True, "0", ["py", "ts"])])
def test_run_tests_in_a_filling_runs_the_tests_of_each_language_it_is_told_of(
        tmp_path, monkeypatch, can_run, own, expect_ext_list):
    """The index's own run is left out when it has nothing to run: a filling is not
    made to fail by a language with no test command."""
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: True)
    # What set_primary_ext() sets is put back after the test
    for name in ("TEST_PATH", "TEST_TARGET_STYLE", "RUN_COMMAND", "CHECK_COMMAND",
                 "OTHER_CHECK_LIST"):
        monkeypatch.setattr(config, name, getattr(config, name))
    for name, value in {"PRIMARY_EXT": "cbl", "TEST_COMMAND": [], "RUN_TEST": True,
                        "OTHER_TEST_LIST": [], "LLM_BROKER": ""}.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setenv(sandbox_mod.CTLOOM_FILL_EXT_ENV, "py,ts")
    monkeypatch.setenv(sandbox_mod.CTLOOM_FILL_OWN_ENV, own)
    monkeypatch.setattr(cli.edit, "test_can_run", lambda: can_run)
    monkeypatch.setattr(cli.shutil, "which", lambda name: name)
    seen = []
    monkeypatch.setattr(cli.edit, "run_test", lambda: seen.append(
        (config.PRIMARY_EXT, list(config.OTHER_TEST_LIST))) or _run_result({
            "status": "passed", "failed": [], "failed_excerpt": "", "targets": [],
            "output": ""}))
    cli.main(["run-tests", "--project", project, "--output-dir", str(tmp_path / "out")])
    assert [ext for ext, _ in seen] == expect_ext_list
    assert all(other == [] for ext, other in seen if ext != "cbl")


@pytest.mark.parametrize("fill, fill_ext, expect_list", [
    ("1", "", [("run", "cbl"), ("write", "cbl")]), ("", "", [("run", "cbl")]),
    ("1", "java", [("run", "cbl"), ("write", "cbl"), ("run", "java"), ("write", "java")])])
def test_run_tests_writes_the_fill_test_only_in_a_filling(tmp_path, monkeypatch, fill,
                                                          fill_ext, expect_list):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: True)
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: ("k.sqlite", project, "proj", True))
    for name in ("TEST_PATH", "TEST_TARGET_STYLE", "RUN_COMMAND", "CHECK_COMMAND",
                 "OTHER_CHECK_LIST", "OTHER_TEST_LIST"):
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.setattr(config, "PRIMARY_EXT", "cbl")
    monkeypatch.setenv(sandbox_mod.CTLOOM_FILL_ENV, fill)
    monkeypatch.setenv(sandbox_mod.CTLOOM_FILL_EXT_ENV, fill_ext)
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: name)
    monkeypatch.setattr(cli.edit, "test_can_run", lambda: True)
    seen = []
    monkeypatch.setattr(cli.edit, "write_fill_test",
                        lambda: seen.append(("write", config.PRIMARY_EXT)) or [])
    monkeypatch.setattr(cli.edit, "run_test", lambda: seen.append(
        ("run", config.PRIMARY_EXT)) or _run_result({
            "status": "passed", "failed": [], "failed_excerpt": "", "targets": [],
            "output": ""}))
    monkeypatch.setattr(config, "TEST_COMMAND", ["run"])
    monkeypatch.setattr(config, "RUN_TEST", True)
    cli.main(["run-tests", "--project", project, "--output-dir", str(tmp_path / "out")])
    assert seen == expect_list


@pytest.mark.parametrize("fill_ext", ["", "java"])
@pytest.mark.parametrize("second_output, exit_status", [
    ("stop", None), ("fetch", 1), ("", None)])
def test_a_filling_runs_the_tests_again_when_it_writes_fill_tests(
        tmp_path, monkeypatch, fill_ext, second_output, exit_status):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: True)
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: ("k.sqlite", project, "proj", True))
    for name in ("TEST_PATH", "TEST_TARGET_STYLE", "RUN_COMMAND", "CHECK_COMMAND",
                 "OTHER_CHECK_LIST", "OTHER_TEST_LIST", "PRIMARY_EXT"):
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.setenv(sandbox_mod.CTLOOM_FILL_ENV, "1")
    monkeypatch.setenv(sandbox_mod.CTLOOM_FILL_EXT_ENV, fill_ext)
    monkeypatch.setenv(sandbox_mod.CTLOOM_FILL_OWN_ENV, "0" if fill_ext else "")
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: name)
    monkeypatch.setattr(cli.edit, "write_fill_test", lambda: ["src/test/java/CtloomFillTest.java"])
    output_list = ["first", second_output]
    monkeypatch.setattr(cli.edit, "run_test", lambda: _run_result({
        "status": "passed", "failed": [], "failed_excerpt": "", "targets": [],
        "output": output_list.pop(0)}))
    monkeypatch.setattr(cli.edit, "stop_short_run",
                        lambda test_dict: test_dict["output"] in ("stop", "fetch"))
    monkeypatch.setattr(cli.edit, "fetch_fail_run",
                        lambda test_dict: test_dict["output"] == "fetch")
    monkeypatch.setattr(config, "TEST_COMMAND", ["run"])
    monkeypatch.setattr(config, "RUN_TEST", True)
    argv = ["run-tests", "--project", project, "--output-dir", str(tmp_path / "out")]
    if exit_status is None:
        cli.main(argv)
    else:
        with pytest.raises(SystemExit) as raised:
            cli.main(argv)
        assert raised.value.code == exit_status
    assert output_list == []


@pytest.mark.parametrize("fill_ext, fetch_fail, exit_status", [
    ("py", False, None), ("py", True, 1), ("", False, 1)])
def test_run_tests_in_a_filling_counts_a_language_it_is_told_of_only_when_it_fetched_not(
        tmp_path, monkeypatch, fill_ext, fetch_fail, exit_status):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: True)
    for name in ("TEST_PATH", "TEST_TARGET_STYLE", "RUN_COMMAND", "CHECK_COMMAND",
                 "OTHER_CHECK_LIST"):
        monkeypatch.setattr(config, name, getattr(config, name))
    for name, value in {"PRIMARY_EXT": "c", "TEST_COMMAND": ["ctloom-test-c"],
                        "RUN_TEST": True, "OTHER_TEST_LIST": [], "LLM_BROKER": ""}.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.setenv(sandbox_mod.CTLOOM_FILL_EXT_ENV, fill_ext)
    monkeypatch.setattr(cli.edit, "test_can_run", lambda: True)
    monkeypatch.setattr(cli.shutil, "which", lambda name: name)
    stop_ext = "py" if fill_ext else "c"
    monkeypatch.setattr(cli.edit, "run_test", lambda: _run_result({
        "status": "did not run", "failed": [], "failed_excerpt": "", "targets": [],
        "output": config.PRIMARY_EXT}))
    monkeypatch.setattr(cli.edit, "stop_short_run",
                        lambda test_dict: test_dict["output"] == stop_ext)
    monkeypatch.setattr(cli.edit, "fetch_fail_run",
                        lambda test_dict: fetch_fail and test_dict["output"] == stop_ext)
    argv = ["run-tests", "--project", project, "--output-dir", str(tmp_path / "out")]
    if exit_status is None:
        cli.main(argv)
    else:
        with pytest.raises(SystemExit) as raised:
            cli.main(argv)
        assert raised.value.code == exit_status


def test_run_tests_in_a_filling_passes_over_a_language_whose_command_the_image_lacks(
        tmp_path, monkeypatch, capsys):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: True)
    for name in ("TEST_PATH", "TEST_TARGET_STYLE", "RUN_COMMAND", "CHECK_COMMAND",
                 "OTHER_CHECK_LIST"):
        monkeypatch.setattr(config, name, getattr(config, name))
    for name, value in {"PRIMARY_EXT": "cbl", "TEST_COMMAND": [], "RUN_TEST": True,
                        "OTHER_TEST_LIST": [], "LLM_BROKER": ""}.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.setenv(sandbox_mod.CTLOOM_FILL_EXT_ENV, "rs,py")
    monkeypatch.setattr(cli.edit, "test_can_run", lambda: False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None if name == "cargo" else name)
    seen = []
    monkeypatch.setattr(cli.edit, "run_test", lambda: seen.append(config.PRIMARY_EXT)
                        or _run_result({"status": "passed", "failed": [],
                                        "failed_excerpt": "", "targets": [],
                                        "output": ""}))
    cli.main(["run-tests", "--project", project, "--output-dir", str(tmp_path / "out")])
    assert seen == ["py"]
    assert "the test command of .rs files is not in this image" in capsys.readouterr().out


@pytest.mark.parametrize("setting_dict, status, exit_status", [
    ({"TEST_COMMAND": [sys.executable, "-c", "pass"]}, "passed", None),
    ({"TEST_COMMAND": [sys.executable, "-c", "import time; time.sleep(30)"],
      "TEST_TIMEOUT_SEC": 1}, "did not run", None),
    ({"TEST_COMMAND": [sys.executable, "-c",
                       "import sys; print('no tests ran in 0.01s'); sys.exit(5)"]},
     "did not run", None),
    ({"TEST_COMMAND": [sys.executable, "-c",
                       "import sys; print('=== short test summary info ===');"
                       "print('FAILED t.py::t_x - boom'); sys.exit(1)"]},
     "failed", None),
    ({"TEST_COMMAND": [sys.executable, "-c",
                       "import sys; print('error: Request failed after 3 retries in 2s');"
                       "print('  Caused by: Failed to fetch: `https://pypi.org/x/`');"
                       "sys.exit(2)"]},
     "did not run", 1),
    ({"TEST_COMMAND": [sys.executable, "-c",
                       "import sys; print('error: something else went wrong'); "
                       "sys.exit(2)"]},
     "did not run", None),
    ({"TEST_COMMAND": [sys.executable, "-c",
                       "import sys; print('BTest > one() FAILED'); sys.exit(1)"]},
     "did not run", None),
    ({"TEST_COMMAND": [sys.executable, "-c",
                       "import sys; print('Ran 2 tests in 0.001s'); "
                       "print('FAILED (failures=1, errors=1)'); sys.exit(1)"]},
     "failed", None),
    ({"TEST_COMMAND": [sys.executable, "-c",
                       "import sys; print('1 failed in 0.1s'); "
                       "print('error: Failed to fetch: `x`'); sys.exit(1)"]},
     "failed", None),
    ({"TEST_COMMAND": ["ctloom-no-such-test-program"]}, "did not run", 1),
    ({"TEST_COMMAND": [sys.executable, "-c", "pass"], "RUN_TEST": False},
     "did not run", 1),
    ({"TEST_COMMAND": []}, "did not run", 1),
])
def test_run_tests_ends_with_status_1_when_a_test_command_stopped_short_of_its_tests(
        tmp_path, monkeypatch, capsys, setting_dict, status, exit_status):
    """The line saying nothing was fetched is printed for the runs that end with 1:
    a command that could not be started, and one that reached no verdict and printed
    the language's line for a tool that ended before the tests."""
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: ("k.sqlite", project, "proj", True))
    for name, value in {"PRIMARY_EXT": "py", "TEST_TARGET_STYLE": "separator",
                        "RUN_TEST": True, "OTHER_TEST_LIST": [], "LLM_BROKER": "",
                        **setting_dict}.items():
        monkeypatch.setattr(config, name, value)
    argv = ["run-tests", "--project", project, "--output-dir", str(tmp_path / "out")]
    if exit_status is None:
        cli.main(argv)
    else:
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == exit_status
    out = capsys.readouterr().out
    assert f"status: {status}" in out
    assert ("nothing was fetched" in out) is (exit_status == 1)


@pytest.mark.parametrize("argv", [["search", "--request", "x"], ["run-tests"]])
def test_an_analysis_of_another_state_is_made_again_without_the_documents(
        tmp_path, monkeypatch, argv):
    calls = _analysed_again(tmp_path, monkeypatch, argv, describes=False)
    assert len(calls) == 2
    assert calls[1]["rebuild"] is True and calls[1]["with_doc"] is False


def test_an_analysis_that_describes_the_tree_is_read_as_it_is(tmp_path, monkeypatch):
    calls = _analysed_again(tmp_path, monkeypatch, ["search", "--request", "x"],
                            describes=True)
    assert len(calls) == 1


def test_build_analyses_again_the_same_way(tmp_path, monkeypatch):
    calls = _analysed_again(tmp_path, monkeypatch, ["build"], describes=False)
    assert len(calls) == 2 and calls[1]["with_doc"] is False


def test_build_with_the_documents_runs_codetwine_either_way(tmp_path, monkeypatch):
    calls = _analysed_again(tmp_path, monkeypatch, ["build", "--doc-create"], describes=True)
    assert len(calls) == 2 and calls[1]["with_doc"] is True


def test_the_analysis_describes_the_tree_only_at_the_commit_it_recorded(tmp_path,
                                                                      monkeypatch):
    repo = tmp_path / "proj"
    _committed(repo)
    state = cli.knowledge_mod.analysis_state(str(repo), [])
    monkeypatch.setattr(cli.build, "commit_of_analysis", lambda *a: state)
    monkeypatch.setattr(cli.knowledge_mod, "analysis_record",
                        lambda *a: {"commit": state, "state": state,
                                    "exclude": ["vendor"],
                                    "codetwine": cli.knowledge_mod.codetwine_id()})
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", ["vendor"]) is True
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", []) is False
    (repo / "a.py").write_text("y = 2\n", encoding="utf-8")
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", ["vendor"]) is False
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", "--", "a.py"],
                   check=True, capture_output=True)
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", ["vendor"]) is True


def test_an_analysis_of_a_tree_with_changes_is_used_again_for_that_same_tree(
        tmp_path, monkeypatch):
    """Report: the state of the tree, not the commit, is what the analysis describes,
    so a dirty tree is not analysed again by every command."""
    repo = tmp_path / "proj"
    _committed(repo)
    (repo / "a.py").write_text("y = 2\n", encoding="utf-8")
    record_dict = {"commit": "", "state": cli.knowledge_mod.analysis_state(str(repo), []), "exclude": [],
                   "codetwine": cli.knowledge_mod.codetwine_id()}
    monkeypatch.setattr(cli.knowledge_mod, "analysis_record", lambda *a: record_dict)
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", []) is True
    (repo / "a.py").write_text("y = 3\n", encoding="utf-8")
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", []) is False


def test_an_analysis_read_with_other_encodings_does_not_describe_the_tree(
        tmp_path, monkeypatch):
    repo = tmp_path / "proj"
    _committed(repo)
    record_dict = {"commit": "", "state": cli.knowledge_mod.analysis_state(str(repo), []), "exclude": [],
                   "encoding": [], "codetwine": cli.knowledge_mod.codetwine_id()}
    monkeypatch.setattr(cli.knowledge_mod, "analysis_record", lambda *a: record_dict)
    monkeypatch.setattr(cli.config, "SOURCE_ENCODING", [])
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", []) is True
    monkeypatch.setattr(cli.config, "SOURCE_ENCODING", ["euc_jp"])
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", []) is False
    record_dict["encoding"] = ["euc_jp"]
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", []) is True


def test_without_a_record_the_analysis_describes_no_tree(tmp_path, monkeypatch, caplog):
    """Nothing says which codetwine made it, even where the index built from it names
    the commit checked out, and nothing is written for it."""
    caplog.set_level(logging.INFO, logger="codetwine_loom.cli")
    repo = tmp_path / "proj"
    _committed(repo)
    head = gitcmd.head_commit(str(repo))
    monkeypatch.setattr(cli.build, "commit_of_analysis", lambda *a: head)
    monkeypatch.setattr(cli.knowledge_mod, "analysis_record", lambda *a: {})
    monkeypatch.setattr(cli.knowledge_mod, "record_analysis_commit",
                        lambda *a: pytest.fail("a record was written"))
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", []) is False
    assert "nothing beside the analysis says what it describes" in caplog.text


@pytest.mark.parametrize("named", ["0.4.1@b60ae34", "", None])
def test_an_analysis_made_by_another_codetwine_does_not_describe_the_tree(
        tmp_path, monkeypatch, caplog, named):
    caplog.set_level(logging.INFO, logger="codetwine_loom.cli")
    repo = tmp_path / "proj"
    _committed(repo)
    record_dict = {"commit": "", "state": cli.knowledge_mod.analysis_state(str(repo), []),
                   "exclude": []}
    if named is not None:
        record_dict["codetwine"] = named
    monkeypatch.setattr(cli.knowledge_mod, "analysis_record", lambda *a: record_dict)
    monkeypatch.setattr(cli.knowledge_mod, "codetwine_id", lambda: "0.4.5@917e5bd")
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", []) is False
    assert "and codetwine 0.4.5@917e5bd is installed" in caplog.text
    record_dict["codetwine"] = "0.4.5@917e5bd"
    assert cli.analysis_match_tree(str(repo), "i.sqlite", "1:now", []) is True


# == the answer subcommand ==================================
def _answered_result(ok: bool = True) -> dict:
    """Build a run_answer() / run_combine() result with the keys main() reads."""
    return {"ok": ok, "answer": "here" if ok else None, "iters": 2,
            "harvested": False, "error": "" if ok else "handed nothing over",
            "scope": {}, "history": []}


def test_answer_answers_the_question_and_writes_it_out(tmp_path, monkeypatch):
    project = _past_the_start_up(tmp_path, monkeypatch)
    asked = []
    monkeypatch.setattr(cli.answer_mod, "run_answer",
                        lambda q: asked.append(q) or _answered_result())
    monkeypatch.setattr(cli.answer_mod, "run_combine",
                        lambda *a: pytest.fail("the combining run was started"))
    out = tmp_path / "sandbox" / "answer-1.json"
    with pytest.raises(SystemExit) as raised:
        cli.main(["answer", "--project", project, "--output-dir", str(tmp_path),
                  "--request", "where?", "--answer-out", str(out)])
    assert raised.value.code == 0 and asked == ["where?"]
    assert json.loads(out.read_text(encoding="utf-8")) == {"answer": "here"}


@pytest.mark.parametrize("argv, flag", [
    (["answer", "--request", "q"], "--answer-out"),
    (["run", "--here", "--request", "q"], "--tasks-out"),
    (["refactor", "--base", "HEAD"], "--outcome-out"),
    (["test-create", "--base", "HEAD", "--target-only"], "--outcome-out"),
    (["integrate", "--here", "--branch", "a", "--branch", "b"], "--outcome-out"),
    (["broker"], "--token-out")])
def test_a_file_to_write_that_cannot_be_written_stops_the_run_before_it_starts(
        tmp_path, monkeypatch, argv, flag):
    """The directory the file goes in is a regular file, and then the path a directory."""
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: pytest.fail("the run was started"))
    monkeypatch.setattr(cli, "_serve_broker", lambda *a: pytest.fail("the broker started"))
    (tmp_path / "plain.txt").write_text("x", encoding="utf-8")
    for path, why in ((tmp_path / "plain.txt" / "out.json", "its directory cannot be made"),
                      (tmp_path, "it is a directory")):
        with pytest.raises(SystemExit) as raised:
            cli.main([*argv, "--project", project, "--output-dir", str(tmp_path / "o"),
                      flag, str(path)])
        assert str(raised.value).startswith(f"{argv[0]}: {flag} {path} cannot be written: {why}")


def test_an_answer_that_could_not_be_written_out_is_still_printed_and_exits_1(
        tmp_path, monkeypatch, capsys, caplog):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.answer_mod, "run_answer", lambda q: _answered_result())

    def no_room(path, what, label):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cli.sandbox_mod, "write_across", no_room)
    out = tmp_path / "answer-1.json"
    with pytest.raises(SystemExit) as raised:
        cli.main(["answer", "--project", project, "--output-dir", str(tmp_path),
                  "--request", "where?", "--answer-out", str(out)])
    assert raised.value.code == 1
    assert "here" in capsys.readouterr().out
    assert f"the answer could not be written to {out}" in caplog.text


def test_answer_with_the_agents_answers_is_the_combining_run(tmp_path, monkeypatch):
    project = _past_the_start_up(tmp_path, monkeypatch)
    answered = tmp_path / "answered-1.json"
    answered.write_text('[{"n": 1, "request": "where?", "answer": "here"}]',
                        encoding="utf-8")
    seen = {}

    def combine(request, entry_list):
        seen.update(request=request, entry_list=entry_list)
        return _answered_result()

    monkeypatch.setattr(cli.answer_mod, "run_combine", combine)
    monkeypatch.setattr(cli.answer_mod, "run_answer",
                        lambda q: pytest.fail("an answering run was started"))
    with pytest.raises(SystemExit) as raised:
        cli.main(["answer", "--project", project, "--output-dir", str(tmp_path),
                  "--request", "q", "--answered", str(answered)])
    assert raised.value.code == 0
    assert seen == {"request": "q", "entry_list": [{"n": 1, "request": "where?",
                                                 "answer": "here"}]}


def test_answer_exits_1_when_nothing_was_handed_over(tmp_path, monkeypatch):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.answer_mod, "run_answer",
                        lambda q: _answered_result(ok=False))
    out = tmp_path / "answer-1.json"
    with pytest.raises(SystemExit) as raised:
        cli.main(["answer", "--project", project, "--output-dir", str(tmp_path),
                  "--request", "q", "--answer-out", str(out)])
    assert raised.value.code == 1
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "error": "handed nothing over"}


# == the exit status of edit follows history.judge() ========
@pytest.fixture
def _edit_main(tmp_path, monkeypatch):
    """Drive main(["edit", ...]) past the start-up with the run itself substituted.

    Returns a function taking the run_edit() result and returning the exit status.
    """
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.history, "open_attempt", lambda: None)
    monkeypatch.setattr(cli, "create_agent", lambda: None)
    monkeypatch.setattr(cli.edit, "git_diff_stat", lambda: {
        "files": [], "untracked": [], "total_insertions": 0, "total_deletions": 0})

    seen_list = []
    monkeypatch.setattr(cli, "check_start",
                        lambda path, exclude_list: seen_list.append(path) or {"at": path})

    def exit_status(result: dict) -> int:
        monkeypatch.setattr(cli, "run_edit", lambda rlm, request, start_dict: {
            **result, "start": start_dict})
        exit_status.result = result
        with pytest.raises(SystemExit) as raised:
            cli.main(["edit", "--project", project, "--output-dir",
                      str(tmp_path / "out"), "--request", "r"])
        return raised.value.code

    return exit_status


def _ran(answer, harvested: bool, verdict: str) -> dict:
    """Build a run_edit() result with the keys main() reads."""
    return {"iters": 20, "answer": answer, "harvested": harvested, "error": "",
            "outcome": None if harvested or answer is None else COMPLETE_KEY,
            "scope": {}, **_judged(verdict)}


def test_a_run_judged_failed_exits_1_even_with_a_harvested_answer(_edit_main):
    """A run out of turns is asked what it changed, and that answer is not a
    completion: judge() says failed, and the exit status is what dispatch reads."""
    assert _edit_main(_ran("edited a.py", True, history.VERDICT_FAIL)) == 1


def test_a_run_judged_unverified_exits_0(_edit_main):
    assert _edit_main(_ran("renamed it", False, history.VERDICT_NO_PROOF)) == 0


def test_a_run_judged_no_change_exits_0(_edit_main):
    assert _edit_main(_ran("it already does that", False, history.VERDICT_NO_CHANGE)) == 0


def test_a_run_with_no_answer_at_all_still_exits_1(_edit_main):
    assert _edit_main(_ran(None, False, history.VERDICT_FAIL)) == 1


def test_a_run_whose_tests_went_red_exits_0(_edit_main):
    """It got through what it was given, so handing the task out again would only
    repeat it. What reads the red test is the merging stage, or a person."""
    assert _edit_main(_ran("changed the signature", False, history.VERDICT_TEST_RED)) == 0


# == What is kept out of the analysis =========================

def test_the_setting_and_the_arguments_are_both_kept(monkeypatch):
    monkeypatch.setattr(config, "BUILD_EXCLUDE", ["vendor"])
    args = cli._parse_args(["build", "--project", "p", "--exclude", "migrations"])
    assert cli.exclude_name_list(args) == ["vendor", "migrations"]


def test_a_name_given_twice_is_kept_once(monkeypatch):
    monkeypatch.setattr(config, "BUILD_EXCLUDE", ["vendor"])
    args = cli._parse_args(["build", "--project", "p", "--exclude", "vendor"])
    assert cli.exclude_name_list(args) == ["vendor"]


def test_a_subcommand_without_the_argument_contributes_nothing(monkeypatch):
    monkeypatch.setattr(config, "BUILD_EXCLUDE", ["vendor"])
    args = cli._parse_args(["search", "--project", "p", "--request", "x"])
    assert cli.exclude_name_list(args) == ["vendor"]


def test_the_sandbox_takes_the_same_argument(monkeypatch):
    monkeypatch.setattr(config, "BUILD_EXCLUDE", [])
    args = cli._parse_args(["sandbox", "--project", "p", "--request", "x",
                            "--exclude", "vendor", "--exclude", "migrations"])
    assert cli.exclude_name_list(args) == ["vendor", "migrations"]


def test_the_broker_is_given_the_run_whose_files_a_refill_takes():
    """The sandbox subcommand passes it for the run it started."""
    args = cli._parse_args(["broker", "--endpoint", "unix:///tmp/s", "--project", "p",
                            "--run", "20260101-000000-1"])
    assert args.run == "20260101-000000-1"
    assert cli._parse_args(["broker", "--endpoint", "unix:///tmp/s"]).run == ""


def test_the_history_shows_every_agent_that_has_one(tmp_path, monkeypatch, capsys):
    """A sandboxed run records in the agent's own copy of the output directory, so
    what one output directory holds is a record per agent."""
    out = tmp_path / "out"
    for agent in ("codetwine-loom-agent1", "codetwine-loom-agent2"):
        directory = pathlib.Path(sandbox_mod.agent_dir(str(out), agent)) / "codetwine"
        directory.mkdir(parents=True)
        (directory / history.HISTORY_FILE).write_text(json.dumps(
            {"schema": history.SCHEMA, "project": "codetwine",
             "attempts": [{"n": 1, "at": "2026-09-01T00:00:00", "verdict": "failed",
                           "why": f"{agent} said so", "files": {}}]}), encoding="utf-8")
    monkeypatch.setattr(config, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(source, "project_name", "codetwine")
    cli.print_history()
    said = capsys.readouterr().out
    assert "codetwine-loom-agent1" in said and "codetwine-loom-agent2" in said
    assert said.count("history: ") == 2


def test_the_history_of_an_agent_s_copy_that_leads_out_of_the_copy_is_not_read(
        tmp_path, monkeypatch, capsys, caplog):
    """The record of an agent's copy is a file a container writes: one that is a
    link to a file outside the copy is not followed, and one that stays inside is
    read."""
    out = tmp_path / "out"
    record_text = json.dumps(
        {"schema": history.SCHEMA, "project": "codetwine",
         "attempts": [{"n": 7, "at": "2026-09-01T00:00:00", "verdict": "failed",
                       "why": "read from where the link leads", "files": {}}]})
    (tmp_path / "outside.json").write_text(record_text, encoding="utf-8")
    for agent, target in (("codetwine-loom-agent1", tmp_path / "outside.json"),
                          ("codetwine-loom-agent2", "inside.json")):
        directory = pathlib.Path(sandbox_mod.agent_dir(str(out), agent)) / "codetwine"
        directory.mkdir(parents=True)
        (directory / "inside.json").write_text(record_text, encoding="utf-8")
        (directory / history.HISTORY_FILE).symlink_to(target)
    monkeypatch.setattr(config, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(source, "project_name", "codetwine")
    with caplog.at_level(logging.WARNING):
        cli.print_history()
    said = capsys.readouterr().out
    assert said.count("history: ") == 2
    assert said.count("read from where the link leads") == 1
    assert said.count("(no attempt recorded yet)") == 1
    assert "is reached through a link that leads out of" in caplog.text


def test_an_output_directory_with_no_record_says_so_once(tmp_path, monkeypatch,
                                                          capsys):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(source, "project_name", "codetwine")
    cli.print_history()
    said = capsys.readouterr().out
    assert said.count("history: ") == 1 and "no attempt recorded yet" in said


# == what the merging run hands back to the host ============
def _merge_result(**over) -> dict:
    """Build run_integration()'s return value for a merge a model settled."""
    result = {"settled": True, "left": [], "called": True, "answer": "kept both sides in a.py",
              "outcome": "complete", "refused_completes": 0, "error": "",
              "work": {"decisions": [], "left": []}}
    result.update(over)
    return result


def test_a_settled_merge_says_so_and_carries_what_it_did(tmp_path, monkeypatch):
    out = tmp_path / "outcome.json"
    monkeypatch.setattr(cli.edit, "test_change_set", set)
    cli.write_outcome(str(out), _merge_result())
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "verdict": "settled", "why": "", "did": "kept both sides in a.py",
        "tests_changed": [], "ended": "complete", "decisions": []}


def test_the_outcome_carries_the_commits_the_run_merged(tmp_path, monkeypatch):
    out = tmp_path / "outcome.json"
    monkeypatch.setattr(cli.edit, "test_change_set", set)
    cli.write_outcome(str(out), _merge_result(), None, ["a" * 40, "b" * 40])
    assert json.loads(out.read_text(encoding="utf-8"))["parents"] == ["a" * 40, "b" * 40]


def test_the_outcome_of_a_merge_left_unsettled_carries_the_separator_lines_it_may_hold(
        tmp_path, monkeypatch):
    out = tmp_path / "outcome.json"
    monkeypatch.setattr(cli.edit, "test_change_set", set)
    separator_dict = {"doc.rst": {"=======": 1}, "f.py": {}}
    cli.write_outcome(str(out), _merge_result(settled=False, left=["- markers in f.py"]),
                      None, None, separator_dict)
    assert json.loads(out.read_text(encoding="utf-8"))["separator"] == separator_dict
    cli.write_outcome(str(out), _merge_result(settled=False, left=["- x"]), None, None, {})
    assert json.loads(out.read_text(encoding="utf-8"))["separator"] == {}
    cli.write_outcome(str(out), _merge_result(), None, None, separator_dict)
    assert "separator" not in json.loads(out.read_text(encoding="utf-8"))


def test_the_outcome_of_a_merge_left_unsettled_says_what_of_head_s_run_keeps_it_so(
        tmp_path, monkeypatch):
    out = tmp_path / "outcome.json"
    monkeypatch.setattr(cli.edit, "test_change_set", set)
    cli.write_outcome(str(out), _merge_result(settled=False, left=["- x"],
                                              head_block=["stop"]))
    assert json.loads(out.read_text(encoding="utf-8"))["head_block"] == ["stop"]
    for result_dict in (_merge_result(settled=False, left=["- x"], head_block=[]),
                        _merge_result(head_block=["stop"])):
        cli.write_outcome(str(out), result_dict)
        assert "head_block" not in json.loads(out.read_text(encoding="utf-8"))



def test_the_cases_run_again_over_the_merge_are_written_beside_its_outcome(tmp_path,
                                                                         monkeypatch):
    gap_in = tmp_path / "gaps.json"
    gap_list = [{"what": "case_1", "why": "a", "test_path": "t", "test_text": "x"},
                {"what": "case_2", "why": "b"}]
    gap_in.write_text(json.dumps(gap_list), encoding="utf-8")
    monkeypatch.setattr(cli.integrate_mod, "recheck_gap_list",
                        lambda entry_list: (entry_list[1:], entry_list[:1]))
    monkeypatch.setattr(cli.edit, "test_change_set", set)
    gap_dict = cli.recheck_gap(str(gap_in))
    assert gap_dict == {"gap": gap_list[1:], "gap_done": gap_list[:1]}
    out = tmp_path / "outcome.json"
    cli.write_outcome(str(out), _merge_result(), gap_dict)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["gap"] == gap_list[1:] and written["gap_done"] == gap_list[:1]
    assert cli.recheck_gap(str(tmp_path / "none.json")) == {}

def test_the_outcome_carries_how_the_run_ended_and_what_it_recorded(tmp_path, monkeypatch):
    """Both are read back by the merge that carries on from this one."""
    out = tmp_path / "outcome.json"
    monkeypatch.setattr(cli.edit, "test_change_set", set)
    cli.write_outcome(str(out), _merge_result(
        settled=False, left=["- markers in a.py"], refused_completes=2,
        work={"decisions": [{"what": "a.py", "why": "kept a's argument"}],
              "left": ["- markers in a.py"]}))
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["ended"] == "refusals exhausted"
    assert written["decisions"] == [{"what": "a.py", "why": "kept a's argument"}]


def test_a_merge_that_changed_a_test_says_which(tmp_path, monkeypatch):
    """The merging stage is told the tests it changes are recorded in the commit, and
    a merge settles a broken test by changing the code or by changing the test. Both
    stages count it the same way, so both reach the commit through run_trailer_dict()."""
    out = tmp_path / "outcome.json"
    monkeypatch.setattr(cli.edit, "test_change_set",
                        lambda: {"tests/test_b.py", "tests/test_a.py"})
    cli.write_outcome(str(out), _merge_result(answer="updated the assertion b() asserts"))
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["tests_changed"] == ["tests/test_a.py", "tests/test_b.py"]
    message = sandbox_mod.commit_message(
        "combine a, b", sandbox_mod.run_trailer_dict({"exit_status": 0}, written),
        prefix="integrate")
    assert "Codetwine-Loom-Tests-Changed: tests/test_a.py tests/test_b.py" in message


@pytest.mark.parametrize("answer, want", [
    ({"calc.py", }, "{'calc.py'}"), (b"bytes", "b'bytes'"), (ValueError("x"), "x")])
def test_an_outcome_is_written_whatever_the_run_said_it_did(tmp_path, monkeypatch, answer,
                                                           want):
    """A value JSON has no form for is written as text, by each of the three writers."""
    monkeypatch.setattr(cli.edit, "test_change_set", set)
    result = _merge_result(
        answer=answer, tests_changed=[], non_test_changed=[], untooled=[], skipped="",
        refused=[], tests_base="", base_failed=[], cases=[], no_case=[], uncovered=[],
        work={"decisions": [], "gap": [], "left": []})
    for name, write in (("merge", cli.write_outcome), ("create", cli.write_create_outcome),
                        ("refactor", lambda path, result_dict: cli.write_refactor_outcome(
                            path, "refactor", result_dict))):
        out = tmp_path / f"{name}.json"
        write(str(out), result)
        outcome_dict = json.loads(out.read_text(encoding="utf-8"))
        assert outcome_dict["did"] == want and outcome_dict["verdict"] == "settled"


def test_the_two_streams_write_what_they_cannot_encode_as_its_escape(monkeypatch):
    """A lone surrogate a model's reply holds is printed and logged, not raised."""
    raw_dict = {name: io.BytesIO() for name in ("stdout", "stderr")}
    for name, raw in raw_dict.items():
        monkeypatch.setattr(sys, name, io.TextIOWrapper(raw, encoding="utf-8",
                                                        errors="strict"))
    with pytest.raises(UnicodeEncodeError):
        print("said: \ud83d")
    progress.set_up_output()
    print("said: \ud83d", flush=True)
    print("warned: \udce3", file=sys.stderr, flush=True)
    assert raw_dict["stdout"].getvalue().endswith(b"said: \\ud83d\n")
    assert raw_dict["stderr"].getvalue() == b"warned: \\udce3\n"
    # A stream that cannot be set to escape is written to as it is
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    progress.set_up_output()
    print("said: plain")
    assert sys.stdout.getvalue() == "said: plain\n"


def test_the_log_of_a_process_started_without_standard_error_is_written_nowhere(
        monkeypatch):
    """A line logged while standard error is pointed somewhere for a time, as it is
    while a turn's block runs, puts nothing there: no line and no traceback."""
    config_list = []
    monkeypatch.setattr(cli.logging, "basicConfig",
                        lambda **kwargs: config_list.append(kwargs))
    monkeypatch.setattr(sys, "stderr", None)
    handler = cli._set_up_log()
    assert config_list[-1]["handlers"] == [handler]
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        handler.handle(logging.LogRecord("x", logging.WARNING, __file__, 1, "said", None,
                                         None))
    assert buf.getvalue() == ""
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    handler = cli._set_up_log()
    handler.handle(logging.LogRecord("x", logging.WARNING, __file__, 1, "said", None, None))
    assert sys.stderr.getvalue().endswith("WARNING x: said\n")


class _PrintStop(Exception):
    """What a test stops a finishing function's first print with."""


def _stop_print(*args, **kwargs):
    raise _PrintStop()


@pytest.mark.parametrize("finish", ["answer", "split", "refactor", "create", "merge"])
def test_what_the_host_reads_is_written_before_anything_is_printed(tmp_path, monkeypatch,
                                                                   finish):
    out = tmp_path / "out.json"
    monkeypatch.setattr(cli.turnlog, "open_run", contextlib.nullcontext)
    monkeypatch.setattr(cli.edit, "test_change_set", set)
    monkeypatch.setattr(cli.test_create_mod, "change_lang_ext", lambda *a: "")
    stage_dict = _merge_result(
        tests_changed=[], non_test_changed=[], untooled=[], skipped="", refused=[],
        tests_base="", base_failed=[], cases=[], no_case=[], uncovered=[],
        work={"decisions": [], "gap": [], "left": []})
    for name in ("print_scope", "print_answer", "print_split", "print_refactor_stage",
                 "print_test_create_stage", "print_integration", "print_union_in_place"):
        monkeypatch.setattr(cli, name, _stop_print)
    args = argparse.Namespace(
        request="r", answer_in=None, answer_out=str(out), task_out=str(out),
        outcome_out=str(out), resume_list=[], command="refactor", base="b",
        project=str(tmp_path), output_dir=str(tmp_path), target_only=False,
        target_in=None, gap_in=None)
    with pytest.raises(_PrintStop):
        if finish == "answer":
            monkeypatch.setattr(cli.answer_mod, "run_answer", lambda request: {
                "ok": True, "answer": "the answer", "scope": {}})
            cli._finish_answer(args)
        elif finish == "split":
            monkeypatch.setattr(cli.dispatch_mod, "run_dispatch", lambda request, **k: {
                "ok": True, "tasks": ["one task"], "scope": {}})
            cli._finish_split(args)
        elif finish == "refactor":
            monkeypatch.setattr(cli.refactor_mod, "run_refactor",
                                lambda *a, **k: stage_dict)
            cli._finish_refactor(args, "k", "p", [])
        elif finish == "create":
            monkeypatch.setattr(cli.test_create_mod, "run_test_create",
                                lambda *a, **k: stage_dict)
            cli._finish_test_create(args, "k", "p", [])
        else:
            monkeypatch.setattr(cli.integrate_mod, "run_integration",
                                lambda *a, **k: stage_dict)
            monkeypatch.setattr(cli.integrate_mod, "branch_commit_list", lambda merged: [])
            cli._finish_integrate(args, {"merged": {}}, "k", "p", [])
    assert json.loads(out.read_text(encoding="utf-8"))


def test_an_unsettled_merge_carries_what_the_machine_still_finds(tmp_path):
    out = tmp_path / "outcome.json"
    cli.write_outcome(str(out), _merge_result(
        settled=False, left=["- markers in a.py", "- b() does not resolve"],
        answer="could not settle b()"))
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["verdict"] == "unsettled"
    assert "markers in a.py" in written["why"] and "b()" in written["why"]


def test_a_merge_git_settled_alone_says_nobody_spoke(tmp_path):
    """No model was called, so the request text is not somebody's account of it."""
    out = tmp_path / "outcome.json"
    cli.write_outcome(str(out), _merge_result(called=False, outcome=None,
                                              answer="Combine the changes on a and b."))
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["did"] == "" and written["ended"] == ""


def test_what_the_branches_were_made_for_reaches_the_merge(monkeypatch):
    """It is what the merging stage is asked with and what the union commits, and the
    merge is where both are written, so leaving it here leaves the merge without it."""
    asked = {}
    monkeypatch.setattr(cli.sandbox_mod, "integrate_into_tree",
                        lambda *a, **k: asked.update(k) or {"exit_status": 0})
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit):
        cli.main(["integrate", "--project", "p", "--branch", "a", "--branch", "b",
                  "--request", "設定を整理して"])
    assert asked["request"] == "設定を整理して"


@pytest.mark.parametrize("argv", [
    ["sandbox", "--request", "r"],
    ["integrate", "--branch", "a", "--branch", "b"],
    ["run", "--request", "r"],
])
@pytest.mark.parametrize("flag", ["--merge", "--merge-unsettled", "--merged-doc-create",
                                  "--branch"])
def test_there_is_no_flag_to_merge_or_to_land_on_a_branch(argv, flag):
    """The change goes into the working tree, and the commit is the user's."""
    if flag == "--branch" and argv[0] == "integrate":
        return
    with pytest.raises(SystemExit):
        cli._parse_args(argv + [flag, "x"])


@pytest.mark.parametrize("argv", [
    ["sandbox", "--request", "r"],
    ["integrate", "--branch", "a", "--branch", "b"],
    ["run", "--request", "r"],
])
def test_the_documents_are_off_unless_asked_for(argv):
    assert cli._parse_args(argv).with_doc is False
    assert cli._parse_args(argv + ["--doc-create"]).with_doc is True


@pytest.mark.parametrize("flag", ["--onto", "--onto-commit"])
def test_what_the_union_is_written_against_is_told_to_here_alone(flag):
    """Without --here the union is written against the working tree."""
    argv = ["integrate", "--branch", "a", "--branch", "b", flag, "x"]
    with pytest.raises(SystemExit):
        cli._parse_args(argv)
    onto_list = ["--onto", "y"] if flag == "--onto-commit" else []
    assert getattr(cli._parse_args(argv + ["--here", *onto_list]),
                   flag[2:].replace("-", "_")) == "x"


@pytest.mark.parametrize("argv, message", [
    (["integrate", "--branch", "a", "--branch", "b", "--outcome-out", "o"],
     "what --here is told"),
    (["integrate", "--branch", "a", "--branch", "b", "--gap-in", "g"],
     "what --here is told"),
    (["integrate", "--here", "--branch", "a", "--branch", "b", "--onto-commit", "c"],
     "needs --onto"),
    (["integrate", "--here", "--branch", "a", "--branch", "b", "--gap-in", "g"],
     "needs --outcome-out"),
    (["test-create", "--base", "HEAD", "--target-only", "--target-in", "t"],
     "--target-only"),
    (["test-create", "--base", "HEAD", "--target-only", "--request", "r"],
     "--target-only"),
    (["run", "--here", "--request", "r", "--engine", "docker"], "--engine"),
    (["run", "--here", "--request", "r", "--image", "i", "--doc-create"],
     "--image, --doc-create"),
    (["integrate", "--here", "--branch", "a", "--branch", "b", "--no-refactor"],
     "runs no stage"),
])
def test_flags_that_would_not_be_read_are_refused(argv, message, capsys):
    with pytest.raises(SystemExit):
        cli._parse_args(argv)
    assert message in capsys.readouterr().err


def test_the_container_flags_default_to_their_settings(monkeypatch):
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "docker")
    monkeypatch.setattr(config, "CONTAINER_IMAGE", "codetwine-loom:js")
    args = cli._parse_args(["run", "--request", "r"])
    assert (args.engine, args.image) == ("docker", "codetwine-loom:js")
    args = cli._parse_args(["run", "--request", "r", "--engine", "podman"])
    assert args.engine == "podman"
    assert cli._parse_args(["run", "--here", "--request", "r"]).engine == "docker"


@pytest.mark.parametrize("overview_on, query_expand, calls", [
    (True, True, True), (False, True, False), (True, False, False)])
def test_the_overview_subcommand_calls_a_model_only_when_it_makes_one(
        monkeypatch, overview_on, query_expand, calls):
    monkeypatch.setattr(config, "OVERVIEW", overview_on)
    monkeypatch.setattr(config, "QUERY_EXPAND", query_expand)
    assert cli.calls_a_model(cli._parse_args(["overview", "--project", "p"])) is calls


def test_target_only_calls_no_model():
    assert cli.calls_a_model(cli._parse_args(["test-create", "--base", "HEAD",
                                              "--target-only"])) is False
    assert cli.calls_a_model(cli._parse_args(["test-create", "--base", "HEAD"])) is True


@pytest.mark.parametrize("subcommand, argv", [
    ("run", ["sandbox", "--request", "r"]),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"]),
    ("dispatch", ["run", "--request", "r"]),
])
def test_the_documents_reach_the_run(subcommand, argv, monkeypatch):
    asked = {}
    monkeypatch.setattr(cli.sandbox_mod, subcommand,
                        lambda *a, **k: asked.update(k) or {"exit_status": 0, "resume": []})
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    monkeypatch.setattr(cli, "print_dispatch", lambda project_dir, result: None)
    with pytest.raises(SystemExit):
        cli.main(argv + ["--project", "p", "--doc-create"])
    assert asked["build_doc"] is True


def test_the_sandbox_subcommand_lands_in_the_working_tree(monkeypatch):
    asked = {}
    monkeypatch.setattr(cli.sandbox_mod, "run",
                        lambda *a, **k: asked.update(k) or {"exit_status": 0})
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit):
        cli.main(["sandbox", "--project", "p", "--request", "r"])
    assert asked["land"] is True and asked["with_overview"] is True


@pytest.mark.parametrize("name, argv, protect", [
    ("run", ["sandbox", "--request", "r"], []),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"],
     ["a", "b", ""])])
def test_sandbox_and_integrate_end_by_taking_older_runs_away(monkeypatch, capsys, name,
                                                            argv, protect):
    asked = {}

    def remove(project, output_dir, engine_path, protect_list=()):
        asked.update(engine=engine_path, protect=list(protect_list))
        return ["20260101-000000-1-42"]

    monkeypatch.setattr(cli.sandbox_mod, name,
                        lambda *a, **k: {"exit_status": 0, "engine": "/bin/podman"})
    monkeypatch.setattr(cli.sandbox_mod, "remove_old_run_in", remove)
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit):
        cli.main(argv + ["--project", "p"])
    assert asked == {"engine": "/bin/podman", "protect": protect}
    assert "older runs taken away" in capsys.readouterr().out


def test_a_change_put_into_the_tree_ends_with_how_to_take_it_back_out(monkeypatch,
                                                                    capsys):
    monkeypatch.setattr(cli.sandbox_mod, "run", lambda *a, **k: {
        "exit_status": 0, "land": {"files": 2, "patch": "p", "last": "/out/land-last.diff"},
        "land_why": ""})
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit):
        cli.main(["sandbox", "--project", "p", "--request", "r"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == (f"to take it back out of the working tree: git -C "
                         f"{os.path.abspath('p')} apply --binary --whitespace=nowarn --no-ignore-whitespace -R "
                         f"/out/land-last.diff")


def test_the_line_that_takes_a_change_back_out_names_a_path_a_shell_reads_whole(capsys):
    cli.print_take_out_line("/a repo; $(x)", {"exit_status": 0, "land_why": "", "land": {
        "files": 1, "patch": "p", "last": "/an out's dir/land-last.diff"}})
    assert capsys.readouterr().out == (
        "to take it back out of the working tree: git -C '/a repo; $(x)' apply "
        "--binary --whitespace=nowarn --no-ignore-whitespace -R '/an out'\"'\"'s dir/land-last.diff'\n")


@pytest.mark.parametrize("land", [
    None, {"files": 0, "patch": "p", "last": ""}, {"files": 2, "patch": "p", "last": ""}])
def test_nothing_is_said_about_taking_out_what_was_not_put_in_or_not_kept(capsys, land):
    cli.print_take_out_line("/repo", {"land": land})
    cli.print_take_out_line("/repo", None)
    assert capsys.readouterr().out == ""


def test_a_request_handed_back_prints_what_is_open_and_how_to_go_on(capsys, monkeypatch):
    result = {"unclear": ["JSON or YAML?", "which command?"], "resume": [], "swept": [],
              "exit_status": 1, "runs": 1}
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", lambda *a, **k: result)
    monkeypatch.setattr(cli.report_mod, "report",
                        lambda *a, **k: pytest.fail("no report is written"))
    with pytest.raises(SystemExit) as raised:
        cli.main(["run", "--request", "one", "--project", "."])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert raised.value.code == 1
    assert lines[0] == "unclear: 2 (no agent was started and nothing was changed)"
    assert lines[1:3] == ["  1. JSON or YAML?", "  2. which command?"]
    assert lines[-1].startswith("to go on, run it again with --request")


def test_what_a_run_before_the_last_put_into_the_tree_is_printed_and_recorded(
        capsys, monkeypatch, tmp_path):
    land = {"files": 2, "paths": ["a.py", "b.py"], "patch": "p",
            "last": "/out/land-last.diff"}
    result = {"unclear": ["JSON or YAML?"], "resume": [], "swept": [], "exit_status": 1,
              "runs": 2, "earlier_land": [land]}
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", lambda *a, **k: result)
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    with pytest.raises(SystemExit):
        cli.main(["run", "--request", "one", "--project", ".",
                  "--output-dir", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert ("unclear: 1 (no agent was started in the last run. The runs before it put "
            "2 file(s) into the working tree, which are still there)") in out
    assert (f"to take it back out of the working tree: git -C {os.path.abspath('.')} "
            f"apply --binary --whitespace=nowarn --no-ignore-whitespace -R /out/land-last.diff") in out
    record_path = next(line for line in out.splitlines()
                       if line.startswith("record: ")).split(": ", 1)[1]
    with open(record_path, encoding="utf-8") as handle:
        assert json.load(handle)["files"] == ["a.py", "b.py"]


@pytest.mark.parametrize("last_land, path_list, last", [
    (None, ["a.py"], "/out/one.diff"),
    ({"files": 0, "paths": [], "patch": "p", "last": ""}, ["a.py"], "/out/one.diff"),
    ({"files": 2, "paths": ["b.py", "a.py"], "patch": "p", "last": "/out/two.diff"},
     ["a.py", "b.py"], "/out/two.diff")])
def test_the_files_put_in_are_those_of_every_run_and_the_diff_kept_is_the_last_one_s(
        capsys, last_land, path_list, last):
    result = {"tasks": ["t"], "branches": [], "runs": 2,
              "integration": {"land": last_land, "land_why": ""},
              "earlier_land": [{"files": 1, "paths": ["a.py"], "patch": "p",
                                "last": "/out/one.diff"}]}
    assert cli._land_path_list(result) == path_list
    cli.print_take_out_line("/repo", result)
    assert capsys.readouterr().out.strip().endswith(f"--whitespace=nowarn --no-ignore-whitespace -R {last}")


def test_the_cases_a_run_before_the_last_left_not_done_are_printed(capsys):
    gap_list = [{"what": "case-1", "why": "g(-1) is not handled"}]
    for result in ({"tasks": ["t"], "branches": [], "dropped": [], "integration": None,
                    "runs": 2, "earlier_gap": gap_list},
                   {"unclear": ["which?"], "runs": 2, "earlier_gap": gap_list}):
        cli.print_dispatch("/repo", result)
        out = capsys.readouterr().out
        assert "NOT done by the change, as the tests found (1):\n" \
               "  case-1 — g(-1) is not handled" in out


def test_what_a_dispatching_run_prints_ends_with_how_to_carry_on(capsys, monkeypatch,
                                                                tmp_path):
    """The report goes between what it is about and the line somebody acts on, so the
    order the three are printed in is main()'s and is checked through it."""
    result = {"tasks": ["one"], "dropped": [], "integration": None,
              "resume": ["run/1/1"], "swept": ["20260101-000000-1-42"],
              "exit_status": 1,
              "branches": [{"n": 1, "request": "one", "branch": "run/1/1",
                            "attempt": 1, "commit": "", "exit_status": 1,
                            "error": "", "verdict": "failed"}]}
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", lambda *a, **k: result)
    monkeypatch.setattr(cli.report_mod, "report", lambda *a, **k: "it did not finish")
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "run/1/1", "one")
    with pytest.raises(SystemExit):
        cli.main(["run", "--request", "one", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[-1] == "to carry on from what this run left: run --resume run/1/1"
    assert lines[-2].startswith("record: ")
    assert lines[-3].startswith("tokens: ")
    assert lines[-4] == "it did not finish"
    assert "older runs taken away" in lines[-5]


@pytest.mark.parametrize("name, argv", [
    ("run", ["sandbox", "--request", "r"]),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"])])
def test_a_change_that_did_not_reach_the_tree_ends_with_status_1(monkeypatch, name, argv,
                                                               tmp_path):
    monkeypatch.setattr(cli.sandbox_mod, name, lambda *a, **k: {
        "exit_status": 0, "land": None, "land_why": "it does not apply"})
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit) as raised:
        cli.main(argv + ["--project", str(tmp_path)])
    assert raised.value.code == 1


def test_a_dispatching_run_whose_union_did_not_reach_the_tree_says_so_last(
        capsys, monkeypatch):
    result = {"tasks": ["one"], "dropped": [], "resume": [], "swept": [],
              "exit_status": 0, "branches": [],
              "integration": {"base": "b" * 40, "patch": None, "commit": None,
                              "branch": "sandbox/x", "land": None,
                              "land_why": "it does not apply; apply it by hand"}}
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", lambda *a, **k: result)
    monkeypatch.setattr(cli.report_mod, "report", lambda *a, **k: "the report")
    with pytest.raises(SystemExit) as raised:
        cli.main(["run", "--request", "one", "--project", "."])
    assert raised.value.code == 1
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines[-1] == ("NOT applied to the working tree: it does not apply; "
                         "apply it by hand")


def test_the_broker_passes_the_run_s_start_to_a_refill(monkeypatch, tmp_path):
    asked = {}
    monkeypatch.setattr(cli.sandbox_mod, "warm", lambda *a, **k: asked.update(k) or {})
    monkeypatch.setattr(cli.broker_mod, "serve",
                        lambda endpoint, refill_cache, token_out="", token_left=0:
                        refill_cache(lambda: None))
    monkeypatch.setattr(cli.turnlog, "open_log", lambda *a, **k: None)
    cli.main(["broker", "--endpoint", "unix:///tmp/s", "--project", str(tmp_path / "repo"),
              "--output-dir", str(tmp_path / "out"), "--run", "r", "--base", "c" * 40,
              "--fill-ext", "py,ts"])
    assert asked["base"] == "c" * 40 and asked["run_id"] == "r"
    assert asked["fill_ext_list"] == ["py", "ts"]
    assert cli._parse_args(["broker", "--endpoint", "unix:///tmp/s"]).base == ""
    assert cli._parse_args(["broker", "--endpoint", "unix:///tmp/s"]).fill_ext == ""
    with pytest.raises(SystemExit):
        cli._parse_args(["broker", "--endpoint", "unix:///tmp/s", "--fill-ext", "py"])


def test_the_line_to_carry_on_says_what_to_resume(capsys, tmp_path):
    cli.print_resume_line(str(tmp_path), {"command": "integrate",
                                          "resume": ["run/1/1", "run/1/2"]})
    assert capsys.readouterr().out.strip().endswith(
        ": integrate --resume run/1/1 --resume run/1/2")


def test_a_merge_reads_the_request_off_the_branches_it_combines(monkeypatch, tmp_path):
    """The branch an earlier merge left is read first: it is the one whose request the
    merge carrying on from it is made for."""
    read = []
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda directory, name_list: read.append(name_list) or "設定を整理して")
    assert cli.request_of_integrate(str(tmp_path), "", ["sandbox/1", "run/1/1"]) == \
        "設定を整理して"
    assert read == [["sandbox/1", "run/1/1"]]


def test_a_merge_carrying_on_is_made_for_what_the_earlier_merge_was(monkeypatch,
                                                                   tmp_path):
    """A branch added beside it was made for a request of its own, which is not what
    the merge being carried on was made for."""
    read = []
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda directory, name_list: read.append(name_list)
                        or "設定とログをまとめて")
    assert cli.request_of_integrate(str(tmp_path), "", ["sandbox/1", "c"],
                                    resume="sandbox/1") == "設定とログをまとめて"
    assert read == [["sandbox/1"]]


def test_an_earlier_merge_saying_nothing_leaves_the_branches_read_together(monkeypatch,
                                                                          tmp_path):
    read = []
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda directory, name_list: read.append(name_list)
                        or ("" if name_list == ["sandbox/1"] else "丸めを足して"))
    assert cli.request_of_integrate(str(tmp_path), "", ["sandbox/1", "c"],
                                    resume="sandbox/1") == "丸めを足して"
    assert read == [["sandbox/1"], ["sandbox/1", "c"]]


def test_branches_merged_by_hand_leave_a_merge_without_a_request(monkeypatch, tmp_path):
    """A merge is made for them all the same, so nothing is refused over it."""
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list", lambda *a: "")
    assert cli.request_of_integrate(str(tmp_path), "", ["mine", "yours"]) == ""


def test_a_merge_given_a_request_is_made_for_that_one(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda *a: "the earlier one")
    assert cli.request_of_integrate(str(tmp_path), "this one", ["sandbox/1"]) == \
        "this one"


def test_a_branch_a_merge_cannot_read_leaves_it_to_the_merging_run_to_say_so(
        monkeypatch, tmp_path):
    def missing(*a):
        raise RuntimeError("no branch named gone")

    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list", missing)
    assert cli.request_of_integrate(str(tmp_path), "", ["gone"]) == ""


def test_the_turns_line_says_how_many_endings_were_sent_back(capsys):
    cli.print_split({"ok": True, "iters": 3, "harvested": False, "refused_completes": 1,
                     "questions": ["where?"]})
    cli.print_answer({"ok": True, "iters": 2, "harvested": False, "refused_completes": 1,
                      "answer": "here", "error": ""})
    out = capsys.readouterr().out
    assert f"iters: 3 / {config.DISPATCH_MAX_TURN}, 1 sent back" in out
    assert "iters: 2, 1 sent back" in out


# == what the printers say ==================================
def test_the_scope_is_printed_with_its_files_definitions_summaries_and_callers(capsys):
    cli.print_scope({
        "confidence": "high", "spread": 2, "note": "",
        "trace": {"top_raw": 3.14159, "top_hit_terms": 4, "query": "config loader"},
        "files": ["a.py", "b.py"],
        "defs": [{"key": "a.py::f", "lines": [3, 9], "signature": "def f():"}],
        "expanded": [{"key": "b.py::g", "lines": [2, 4], "signature": "def g():",
                      "from": "a.py::f"}],
        "file_summaries": {"a.py": "reads things"},
        "callers": {"a.py::f": ["b.py::g"], "b.py::g": []}})
    out = capsys.readouterr().out
    assert out.startswith("query: config loader\n")
    assert "confidence: high (raw 3.1 / hit_terms 4 / spread 2)" in out
    assert "  a.py::f  [3-9]\n    def f():" in out
    assert "  b.py::g  [2-4]  <- a.py::f" in out
    assert "  a.py\n    reads things" in out
    assert "  a.py::f\n    <- b.py::g" in out


def test_the_scope_says_when_the_search_found_nothing(capsys):
    cli.print_scope({
        "confidence": "low", "spread": 0,
        "note": "The search matched nothing; search_again with English terms first.",
        "trace": {"top_raw": 0.0, "top_hit_terms": 0, "query": "設定"},
        "files": [], "defs": [], "expanded": [], "file_summaries": {}, "callers": {}})
    out = capsys.readouterr().out
    assert "The search matched nothing" in out


# A branch named the way sandbox.run() and sandbox.integrate() name the ones they make
_RUN_BRANCH = "codetwine-loom/sandbox/20260101-000000-1-1"


def _sandbox_result(tmp_path, **override) -> dict:
    """Build sandbox.run()'s return value with a patch file that is there."""
    patch = tmp_path / "patch.diff"
    patch.write_text("diff\n" * 40, encoding="utf-8")
    result = {"base": "b" * 40, "patch": str(patch), "files": 2, "branch": _RUN_BRANCH,
              "commit": "c" * 40, "land": {"files": 2, "patch": str(patch)},
              "land_why": ""}
    result.update(override)
    return result


def test_a_change_put_into_the_working_tree_says_so_and_names_no_branch(tmp_path, capsys):
    """The branch is gone by the time this prints, so naming it would send the user
    to a name that is not there."""
    cli.print_sandbox("/repo", _sandbox_result(tmp_path))
    out = capsys.readouterr().out
    assert "patch: " in out and "(0.2 KB, 2 file(s))" in out
    assert "applied to the working tree: 2 file(s), over bbbbbbbbbbbb. Commit them" in out
    assert "branch:" not in out and "merge" not in out


def test_a_branch_this_tool_did_not_name_is_named_beside_what_was_applied(tmp_path,
                                                                          capsys):
    """It was passed to carry on from, and is still there once the change is put in."""
    cli.print_sandbox("/repo", _sandbox_result(tmp_path, branch="feature-x"))
    out = capsys.readouterr().out
    assert "branch: feature-x (cccccccccccc)" in out
    assert "applied to the working tree: 2 file(s)" in out


def test_a_change_not_put_into_the_working_tree_names_the_branch_and_why(tmp_path,
                                                                          capsys):
    cli.print_sandbox("/repo", _sandbox_result(
        tmp_path, land=None, land_why="the run ended with exit status 1, so what it "
                                      "changed stays on sandbox/one"))
    out = capsys.readouterr().out
    assert f"branch: {_RUN_BRANCH} (cccccccccccc)" in out
    assert "NOT applied to the working tree: the run ended with exit status 1" in out


def test_a_branch_left_with_an_open_stage_is_named_beside_what_was_applied(tmp_path,
                                                                           capsys):
    cli.print_sandbox("/repo", _sandbox_result(
        tmp_path, test_create=_create_result(verdict="unsettled", why="t::new fails")))
    out = capsys.readouterr().out
    assert f"branch: {_RUN_BRANCH} (cccccccccccc)" in out
    assert "applied to the working tree: 2 file(s)" in out


def _create_result(**override) -> dict:
    """Build what sandbox.integrate() returns under "test_create"."""
    create_dict = {"verdict": "settled", "why": "", "skipped": "",
                   "cases": [{"id": "case_1"}, {"id": "case_2"}],
                   "tests_changed": ["tests/test_a.py"], "non_test_changed": [],
                   "tests_base": "red", "ended": "complete", "exit_status": 0,
                   "patch": "p", "files": 1, "branch": "sandbox/one", "commit": "e" * 40}
    create_dict.update(override)
    return create_dict


def test_the_tests_a_merging_run_wrote_are_printed_under_its_branch(tmp_path, capsys):
    cli.print_sandbox("/repo", _sandbox_result(tmp_path, test_create=_create_result()))
    out = capsys.readouterr().out
    assert "tests: sandbox/one (eeeeeeeeeeee), settled" in out
    assert "cases decided from the request: 2" in out
    assert "written: tests/test_a.py" in out and "on the base: red" in out
    assert "NOT settled" not in out and "also changed" not in out


def test_what_the_tests_found_the_change_does_not_do_is_printed(tmp_path, capsys):
    cli.print_sandbox("/repo", _sandbox_result(
        tmp_path, test_create=_create_result(),
        gap=[{"what": "case_8", "why": "import_to_path.py makes no edge"}]))
    out = capsys.readouterr().out
    assert "NOT done by the change, as the tests found (1):" in out
    assert "  case_8 — import_to_path.py makes no edge" in out


def test_nothing_is_said_of_gaps_when_the_tests_found_none(tmp_path, capsys):
    cli.print_sandbox("/repo", _sandbox_result(tmp_path, test_create=_create_result(), gap=[]))
    assert "NOT done" not in capsys.readouterr().out


def test_tests_left_open_say_why_and_what_else_was_changed(tmp_path, capsys):
    cli.print_sandbox("/repo", _sandbox_result(tmp_path, test_create=_create_result(
        verdict="unsettled", why="- `run them` fails t::new",
        non_test_changed=["src/a.py"])))
    out = capsys.readouterr().out
    assert "also changed (not tests): src/a.py" in out
    assert "NOT settled: - `run them` fails t::new" in out


def test_several_test_tasks_and_what_has_no_case_are_printed(tmp_path, capsys):
    entry_list = [{"verdict": "settled", "skipped": ""}, {"verdict": "unsettled", "skipped": ""},
                  {"verdict": "", "skipped": "no case was decided"}]
    cli.print_sandbox("/repo", _sandbox_result(tmp_path, test_create=_create_result(
        verdict="unsettled", why="test task 2 (b.py): - red", test_tasks=entry_list,
        uncovered=[{"file": "c.py", "name": "h"}], refused=["shared/util.py"])))
    out = capsys.readouterr().out
    assert "test tasks: 3 (2 settled)" in out
    assert "writes refused (not the test task's files): shared/util.py" in out
    assert "cases decided from the request: 2\n" in out
    assert "no case for: 1 definition(s)" in out
    assert "NOT settled: test task 2 (b.py): - red" in out


def test_the_subcommand_lists_what_there_is_to_test_or_takes_the_files_it_is_handed(
        tmp_path):
    args = cli._parse_args(["test-create", "--project", "p", "--base", "a" * 40,
                            "--target-only", "--outcome-out", "o.json"])
    assert args.target_only is True and args.target_in == ""
    group = tmp_path / "t.json"
    group.write_text('{"paths": ["a.py"], "test_paths": [], "baseline": ["t::x"]}',
                     encoding="utf-8")
    assert cli.read_test_task(str(group)) == {"paths": ["a.py"], "test_paths": [],
                                              "baseline": ["t::x"]}
    group.write_text('{"paths": ["a.py"], "test_paths": [], "baseline": [], '
                     '"first_case": 4, "case_step": 3}', encoding="utf-8")
    assert cli.read_test_task(str(group))["first_case"] == 4
    for text in ('{"paths": "a.py"}',
                 '{"paths": [], "test_paths": [], "baseline": [], "case_step": 0}',
                 '{"paths": [], "test_paths": [], "baseline": [], "first_case": "4"}'):
        group.write_text(text, encoding="utf-8")
        with pytest.raises(SystemExit, match="is not"):
            cli.read_test_task(str(group))
    with pytest.raises(SystemExit, match="cannot be read"):
        cli.read_test_task(str(tmp_path / "none.json"))


def test_a_stage_that_wrote_no_test_says_why(tmp_path, capsys):
    cli.print_sandbox("/repo", _sandbox_result(tmp_path, test_create=_create_result(
        skipped="the repository holds no test file", commit=None, tests_changed=[])))
    assert "tests: none written (the repository holds no test file)" in \
        capsys.readouterr().out


def test_the_subcommand_that_writes_tests_takes_the_base_it_is_written_against():
    args = cli._parse_args(["test-create", "--project", "p", "--base", "a" * 40,
                            "--request", "r", "--outcome-out", "o.json"])
    assert (args.command, args.base, args.request, args.outcome_out) == \
        ("test-create", "a" * 40, "r", "o.json")
    assert not hasattr(args, "guide")
    with pytest.raises(SystemExit):
        cli._parse_args(["test-create", "--project", "p"])


def test_what_the_tests_came_to_is_written_with_the_keys_the_commit_is_made_from(tmp_path):
    out = tmp_path / "o.json"
    cli.write_create_outcome(str(out), {
        "settled": False, "left": ["- a", "- b"], "answer": "wrote two", "called": True,
        "tests_changed": ["tests/test_a.py"], "non_test_changed": ["src/a.py"],
        "untooled": [], "tests_base": "green", "base_failed": [], "skipped": "",
        "cases": [{"id": "case_1", "expect": "raises ValueError"}],
        "no_case": [{"file": "b.py", "name": "g", "why": "checked"}],
        "uncovered": [{"file": "c.py", "name": "h"}], "refused": ["x.py"],
        "outcome": "complete", "error": "",
        "work": {"decisions": [], "gap": [{"what": "case_2", "why": "a.py"}],
                 "left": ["- a"]}})
    outcome_dict = json.loads(out.read_text(encoding="utf-8"))
    assert outcome_dict["stage"] == "test-create" and outcome_dict["verdict"] == "unsettled"
    assert outcome_dict["why"] == "- a / - b" and outcome_dict["did"] == "wrote two"
    assert outcome_dict["ended"] == "refusals exhausted"
    assert outcome_dict["non_test_changed"] == ["src/a.py"]
    assert outcome_dict["tests_base"] == "green"
    assert outcome_dict["cases"] == [{"id": "case_1", "expect": "raises ValueError"}]
    assert outcome_dict["uncovered"] == [{"file": "c.py", "name": "h"}]
    assert "cut" not in outcome_dict and outcome_dict["no_case"][0]["name"] == "g"
    assert outcome_dict["gap"] == [{"what": "case_2", "why": "a.py"}]
    assert outcome_dict["refused"] == ["x.py"]


def _refactor_result(**override) -> dict:
    """Build what sandbox.integrate() returns under "refactor" and "test_refactor"."""
    stage_dict = {"verdict": "settled", "why": "", "skipped": "", "refused": [],
                  "tests_changed": [], "non_test_changed": ["src/a.py"],
                  "ended": "complete", "exit_status": 0, "patch": "p", "files": 1,
                  "branch": "sandbox/one", "commit": "f" * 40}
    stage_dict.update(override)
    return stage_dict


def test_what_each_stage_that_refactors_left_is_printed_under_the_tests(tmp_path, capsys):
    cli.print_sandbox("/repo", _sandbox_result(
        tmp_path, test_create=_create_result(), refactor=_refactor_result(),
        test_refactor=_refactor_result(
            verdict="unsettled", why="- case_2 is in no test file", refused=["src/a.py"],
            tests_changed=["tests/test_a.py"], non_test_changed=[])))
    out = capsys.readouterr().out
    assert out.index("tests: sandbox/one") < out.index("refactoring: sandbox/one") \
        < out.index("test refactoring: sandbox/one")
    assert "refactoring: sandbox/one (ffffffffffff), settled\n  changed: src/a.py" in out
    assert "test refactoring: sandbox/one (ffffffffffff), unsettled" in out
    assert "writes refused (not the stage's files): src/a.py" in out
    assert out.count("NOT settled") == 1
    assert "NOT settled: - case_2 is in no test file" in out


def test_a_stage_that_refactored_nothing_says_so(tmp_path, capsys):
    cli.print_sandbox("/repo", _sandbox_result(
        tmp_path, refactor=_refactor_result(patch=None, files=0, non_test_changed=[]),
        test_refactor=_refactor_result(skipped="the change added or changed no test file",
                                       patch=None, files=0, non_test_changed=[])))
    out = capsys.readouterr().out
    assert "refactoring: nothing changed, settled" in out
    assert "test refactoring: nothing done (the change added or changed no test file)" in out


def test_a_stage_whose_patch_was_not_taken_in_says_nothing_was_taken_in(tmp_path, capsys):
    why = ("the container left no patch that is taken in, so nothing of the stage was "
           "taken in: the patch does not hold what the run wrote into 1 submodule "
           "directory(ies) of its clone that git shows as their commit alone: sub. It "
           "stays at /o/p.diff")
    cli.print_sandbox("/repo", _sandbox_result(
        tmp_path, refactor=_refactor_result(
            verdict="unsettled", why=why, exit_status=1, patch=None, files=0,
            commit=None, non_test_changed=["c.txt"]),
        test_refactor=_refactor_result(
            verdict="unsettled", why="- t fails", patch=None, files=0, commit=None,
            non_test_changed=[])))
    out = capsys.readouterr().out
    assert f"refactoring: nothing taken in, unsettled\n  changed: c.txt\n  NOT settled: " \
           f"{why}\n" in out
    assert "test refactoring: nothing changed, unsettled" in out


@pytest.mark.parametrize("stage", ["refactor", "test-refactor"])
def test_the_subcommands_that_refactor_take_the_base_they_refactor_against(stage):
    args = cli._parse_args([stage, "--project", "p", "--base", "a" * 40, "--request", "r",
                            "--outcome-out", "o.json"])
    assert (args.command, args.base, args.request, args.outcome_out) == \
        (stage, "a" * 40, "r", "o.json")
    assert not hasattr(args, "guide")
    with pytest.raises(SystemExit):
        cli._parse_args([stage, "--project", "p"])


def test_what_the_refactoring_came_to_is_written_with_the_keys_the_commit_is_made_from(
        tmp_path):
    out = tmp_path / "o.json"
    cli.write_refactor_outcome(str(out), "test-refactor", {
        "settled": False, "left": ["- a", "- b"], "answer": "moved the set-up",
        "called": True, "tests_changed": ["tests/test_a.py"], "non_test_changed": [],
        "untooled": ["tests/test_a.py"], "skipped": "", "refused": ["x.py"],
        "outcome": "complete", "error": "",
        "work": {"decisions": [], "left": ["- a", "- b"]}})
    outcome_dict = json.loads(out.read_text(encoding="utf-8"))
    assert outcome_dict["stage"] == "test-refactor"
    assert outcome_dict["verdict"] == "unsettled" and outcome_dict["why"] == "- a / - b"
    assert outcome_dict["did"] == "moved the set-up"
    assert outcome_dict["ended"] == "refusals exhausted"
    assert outcome_dict["tests_changed"] == ["tests/test_a.py"]
    assert outcome_dict["untooled"] == ["tests/test_a.py"]
    assert outcome_dict["refused"] == ["x.py"]


def test_a_patch_not_taken_in_says_how_to_apply_it(tmp_path, capsys):
    cli.print_sandbox("/repo", _sandbox_result(tmp_path, branch="", commit=""))
    assert ("not taken in. Applying it by hand skips the checks a change put into the "
            "working tree goes through; read it, and apply it by hand with: git -C "
            "/repo apply ") in capsys.readouterr().out


def test_a_patch_not_taken_in_that_moves_a_submodule_says_how_to_move_it(tmp_path,
                                                                          capsys):
    """The command checks the submodule out at the commit the patch moves it to, as
    the one a change the working tree refused ends with does."""
    one, two = "1" * 40, "2" * 40
    repo = _git_repo(tmp_path / "repo")
    subprocess.run(["git", "-C", repo, "update-index", "--add", "--cacheinfo",
                    f"160000,{one},lib/sub"], check=True)
    result_dict = _sandbox_result(tmp_path, branch="", commit="")
    patch = tmp_path / "patch.diff"
    patch.write_text(f"diff --git a/lib/sub b/lib/sub\nindex {one}..{two} 160000\n"
                     f"--- a/lib/sub\n+++ b/lib/sub\n@@ -1 +1 @@\n"
                     f"-Subproject commit {one}\n+Subproject commit {two}\n",
                     encoding="utf-8")
    cli.print_sandbox(repo, result_dict)
    line = capsys.readouterr().out.splitlines()[-1]
    assert "(git apply moves no submodule, so the command first stops unless lib/sub " \
           "is at the commit the change moves it from, and then checks out there the " \
           "commit the change moves it to, which has to be there) with: " \
           f"git -C {repo}/lib/sub --git-dir=.git merge-base --is-ancestor HEAD {one} " \
           f"&& git -C {repo}/lib/sub --git-dir=.git merge-base --is-ancestor {one} " \
           f"HEAD || {{ printf '%s\\n' 'nothing was applied: lib/sub is not at {one}, " \
           f"the commit the change moves it from' >&2; false; }} && git -C {repo} " in line
    assert line.endswith(f"{patch} && git -C {repo}/lib/sub --git-dir=.git checkout "
                         f"--detach {two}")


def test_a_run_that_changed_nothing_says_so(tmp_path, capsys):
    cli.print_sandbox("/repo", _sandbox_result(tmp_path, patch="", commit=None, branch=None,
                                      land=None))
    out = capsys.readouterr().out
    assert "changed: nothing" in out and "patch:" not in out and "applied" not in out


def test_a_run_whose_container_wrote_no_patch_is_not_said_to_have_changed_nothing(
        tmp_path, capsys):
    """The line says why, as the run returned it, whatever the warnings above say."""
    why = "the patch the run wrote holds 9 bytes, more than the 8 that are taken in. " \
          "It stays at /out/patch.diff"
    cli.print_sandbox("/repo", {**_sandbox_result(tmp_path, patch="", commit=None, branch=None,
                                         land=None), "no_patch": True,
                                "no_patch_why": why})
    out = capsys.readouterr().out
    assert f"NOT taken in: {why}\n" in out
    assert "changed: nothing" not in out


def test_a_dry_run_patch_that_does_not_hold_all_the_run_wrote_says_so_above_the_command(
        tmp_path, capsys, monkeypatch):
    """What the summary names as not in the patch is printed before the command that
    applies it, and a patch with nothing of the kind prints no such line."""
    monkeypatch.setattr(cli.sandbox_mod, "by_hand_guide",
                        lambda project_dir, patch_path: ("", f"git apply {patch_path}"))
    unseen_text = "what the run wrote into 1 submodule directory(ies): sub"
    for text in (unseen_text, ""):
        result = {**_sandbox_result(tmp_path, commit=None, branch=None, land=None),
                  "unseen_text": text}
        cli.print_sandbox("/repo", result)
        line_list = capsys.readouterr().out.splitlines()
        command_index = [n for n, line in enumerate(line_list)
                         if line.startswith("not taken in.")]
        assert command_index and line_list[command_index[0]].endswith(
            f"git apply {result['patch']}")
        assert line_list[:command_index[0]][-1:] == (
            [f"NOT in the patch, nor applied by the command below: {unseen_text}"]
            if text else [f"patch: {result['patch']} (0.2 KB, 2 file(s))"])


def test_a_dry_run_patch_that_holds_nothing_names_what_the_run_wrote_outside_it(
        tmp_path, capsys):
    """A run whose patch holds nothing and whose summary names what is not in it is
    not printed as having changed nothing."""
    unseen_text = "what the run wrote into 1 submodule directory(ies): sub"
    for text in (unseen_text, ""):
        cli.print_sandbox("/repo", {
            **_sandbox_result(tmp_path, patch=None, commit=None, branch=None, land=None),
            "no_patch": False, "unseen_text": text})
        assert capsys.readouterr().out.splitlines()[1:] == (
            [f"NOT in the patch, which holds nothing else: {unseen_text}"] if text
            else ["changed: nothing"])


def test_a_run_that_did_not_get_through_is_not_said_to_have_only_changed_nothing(
        tmp_path, capsys, monkeypatch):
    """A patch that holds nothing is printed with why nothing was put in, and a patch
    not taken in names the exit status of the run that wrote it."""
    monkeypatch.setattr(config, "SANDBOX_MEMORY", "2g")
    monkeypatch.setattr(cli.sandbox_mod, "by_hand_guide",
                        lambda project_dir, patch_path: ("", f"git apply {patch_path}"))
    why = "the merging run ended (exit status 137, …) and left no patch to take in"
    cli.print_sandbox("/repo", {
        **_sandbox_result(tmp_path, patch=None, commit=None, branch=None, land=None),
        "no_patch": False, "exit_status": 137, "land_why": why})
    assert capsys.readouterr().out.splitlines()[1:] == [
        f"NOT applied to the working tree: {why}"]
    for status, text in ((137, ", from a run that ended with exit status 137, the status "
                               "of a container killed when its processes reach "
                               "SANDBOX_MEMORY (2g)"), (0, "")):
        result = {**_sandbox_result(tmp_path, commit=None, branch=None, land=None),
                  "exit_status": status}
        cli.print_sandbox("/repo", result)
        assert capsys.readouterr().out.splitlines()[-1].startswith(
            f"not taken in{text}. Applying it by hand skips")


def test_a_stage_carried_on_over_what_is_already_in_the_tree_says_what_it_added(
        tmp_path, capsys):
    """A run carrying on with a stage lands no patch of its own; what the stage
    settled is what goes in."""
    cli.print_sandbox("/repo", _sandbox_result(tmp_path, patch=None, land={"files": 1, "patch": "p"}))
    out = capsys.readouterr().out
    assert "changed: nothing" not in out and "branch:" not in out
    assert "applied to the working tree: 1 file(s)" in out
    cli.print_sandbox("/repo", _sandbox_result(tmp_path, patch=None, land={"files": 0, "patch": "p"}))
    assert "nothing more to put into the working tree" in capsys.readouterr().out


def test_a_test_command_that_did_not_run_says_nothing_was_fetched(capsys, monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    cli.print_test({"status": "did not run", "failed_excerpt": "run: not found",
                    "passed": 0, "failed": []})
    out = capsys.readouterr().out
    assert "test command: run them" in out and "status: did not run" in out
    assert "nothing was fetched" in out and "run: not found" in out


# == the subcommands that hand over to a container ==========
def test_the_broker_subcommand_serves_on_the_endpoint_it_is_given(monkeypatch):
    served = {}
    monkeypatch.setattr(cli.broker_mod, "serve",
                        lambda endpoint, refill_cache=None, token_out="", token_left=0: served.update(
                            endpoint=endpoint, refill=refill_cache, token_out=token_out))
    cli.main(["broker", "--endpoint", "unix:///run/ctloom/llm.sock"])
    assert served == {"endpoint": "unix:///run/ctloom/llm.sock", "refill": None,
                      "token_out": ""}


def test_the_broker_is_told_where_to_write_its_token_counts(monkeypatch, tmp_path):
    served = {}
    monkeypatch.setattr(cli.broker_mod, "serve",
                        lambda endpoint, refill_cache=None, token_out="", token_left=0: served.update(
                            token_out=token_out, token_left=token_left))
    cli.main(["broker", "--endpoint", "unix:///run/ctloom/llm.sock", "--token-out",
              str(tmp_path / "token.json"), "--token-left", "500"])
    assert served == {"token_out": str(tmp_path / "token.json"), "token_left": 500}


def test_the_broker_given_a_repository_can_fill_its_cache(monkeypatch, tmp_path):
    served = {}
    monkeypatch.setattr(cli.broker_mod, "serve",
                        lambda endpoint, refill_cache=None, token_out="", token_left=0: served.update(
                            refill=refill_cache))
    filled = {}
    monkeypatch.setattr(cli.sandbox_mod, "warm",
                        lambda *a, **k: filled.update(args=a, kwargs=k) or {})
    project, out = str(tmp_path / "p"), str(tmp_path / "o")
    cli.main(["broker", "--endpoint", "unix:///run/ctloom/llm.sock", "--project", project,
              "--output-dir", out, "--agent", "codetwine-loom-one", "--run", "20260101-000000-1-1"])
    spend = object()
    served["refill"](spend)
    assert filled["args"] == (project, out)
    assert filled["kwargs"] == {"live_dep_file": True, "run_id": "20260101-000000-1-1",
                                "once": True, "before_fill": spend, "base": "",
                                "agent": "codetwine-loom-one", "fill_ext_list": []}
    # The broker of an agent keeps its log with what this machine alone writes for it
    assert list((tmp_path / "o" / "agents" / ".host" / "codetwine-loom-one" / "p" / "turns")
                .glob("*.log"))
    assert not (tmp_path / "o" / "agents" / "codetwine-loom-one").exists()


def test_a_broker_given_a_label_fills_the_cache_under_it(monkeypatch, tmp_path):
    served = {}
    monkeypatch.setattr(cli.broker_mod, "serve",
                        lambda endpoint, refill_cache=None, token_out="", token_left=0: served.update(
                            refill=refill_cache))
    label_list = []
    monkeypatch.setattr(cli.sandbox_mod, "warm",
                        lambda *a, **k: label_list.append(cli.progress.label_text()) or {})
    cli.main(["broker", "--endpoint", "unix:///run/ctloom/llm.sock", "--project",
              str(tmp_path / "p"), "--output-dir", str(tmp_path / "o"),
              "--label", "[task 2/5] "])
    served["refill"](object())
    assert label_list == ["[task 2/5] "]
    assert cli.progress.label_text() == ""
    # A broker serving no agent keeps its log under the output directory
    assert list((tmp_path / "o" / "p" / "turns").glob("*.log"))


# == a command stopped by a signal, and a broker whose run has ended ================
@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_a_signal_that_stops_a_command_is_raised_the_way_ctrl_c_is(name):
    """What takes the containers away, stops the broker and writes the record of a
    stopped command is the handling of KeyboardInterrupt, so the two signals arrive
    as one."""
    number = getattr(signal, name)
    signal.signal(number, signal.SIG_DFL)
    cli.stop_on_signal()
    with pytest.raises(KeyboardInterrupt, match=rf"the run was stopped \({name}\)"):
        os.kill(os.getpid(), number)
        time.sleep(5)


@pytest.mark.parametrize("name", ["SIGINT", "SIGTERM", "SIGHUP", "while importing"])
def test_a_command_stopped_by_a_signal_ends_with_one_line_and_the_signal_s_status(
        monkeypatch, capsys, tmp_path, name):
    """Its record and the lines after it are written where it was stopped; the entry
    point adds one line and no traceback."""
    import codetwine_loom
    from codetwine_loom import __main__ as main_mod

    def stop(*a, **k):
        signal.raise_signal(getattr(signal, name))
        time.sleep(5)

    def stop_import(attribute):
        raise KeyboardInterrupt

    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(main_mod, "cwd_module_list", lambda: [])
    monkeypatch.setattr(cli.sandbox_mod, "run", stop)
    monkeypatch.setattr(sys, "argv", ["codetwine-loom", "sandbox", "--request", "r",
                                      "--project", repo, "--output-dir",
                                      str(tmp_path / "out")])
    if name == "while importing":
        monkeypatch.delattr(codetwine_loom, "cli")
        monkeypatch.setattr(codetwine_loom, "__getattr__", stop_import)
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, signal.SIG_DFL)
    with pytest.raises(SystemExit) as stop_end:
        main_mod.main()
    number = signal.SIGINT if name == "while importing" else getattr(signal, name)
    err = capsys.readouterr().err
    assert stop_end.value.code == 128 + number
    assert err.splitlines()[-1:] == [f"stopped by {number.name}"]
    assert "Traceback" not in err
    if name == "while importing":
        assert not (tmp_path / "out").exists()
        return
    record_dict, = _record_list(tmp_path)
    assert record_dict["exit_status"] == 128 + number
    assert record_dict["error"] == ("the run was stopped" if name == "SIGINT"
                                    else f"the run was stopped ({name})")


def test_a_signal_the_process_was_started_ignoring_stays_ignored():
    """A command started under nohup is not stopped when the terminal goes."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    own = lambda number, frame: None   # noqa: E731
    signal.signal(signal.SIGTERM, own)
    cli.stop_on_signal()
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    assert signal.getsignal(signal.SIGTERM) is own


def test_the_signals_are_left_alone_outside_the_main_thread():
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    thread = threading.Thread(target=cli.stop_on_signal)
    thread.start()
    thread.join()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_main_sets_the_signals_before_it_reads_its_arguments(monkeypatch):
    call_list = []
    monkeypatch.setattr(cli, "stop_on_signal", lambda: call_list.append("signal") or [])
    monkeypatch.setattr(cli, "_parse_args",
                        lambda argv: call_list.append("args") or pytest.fail("stop here"))
    with pytest.raises(pytest.fail.Exception):
        cli.main(["build"])
    assert call_list == ["signal", "args"]


def test_main_takes_back_the_handlers_and_the_start_time_it_set(monkeypatch):
    """A program that calls main() is stopped by a signal the way it was before."""
    own = lambda number, frame: None   # noqa: E731
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, own)
    monkeypatch.delenv(cli.progress.RUN_START_ENV, raising=False)
    seen = {}

    def parse(argv):
        seen["handler"] = signal.getsignal(signal.SIGTERM)
        seen["start"] = cli.progress.RUN_START_ENV in os.environ
        raise SystemExit(2)

    monkeypatch.setattr(cli, "_parse_args", parse)
    with pytest.raises(SystemExit):
        cli.main(["build"])
    assert seen == {"handler": cli._raise_interrupt, "start": True}
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    assert signal.getsignal(signal.SIGHUP) is own
    assert cli.progress.RUN_START_ENV not in os.environ


def test_main_leaves_a_handler_and_a_start_time_that_were_there_before_it(monkeypatch):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    cli.stop_on_signal()
    monkeypatch.setenv(cli.progress.RUN_START_ENV, "1000")
    monkeypatch.setattr(cli, "_parse_args", lambda argv: sys.exit(2))
    with pytest.raises(SystemExit):
        cli.main(["build"])
    assert signal.getsignal(signal.SIGTERM) is cli._raise_interrupt
    assert os.environ[cli.progress.RUN_START_ENV] == "1000"


def test_a_handler_set_while_the_command_ran_is_not_taken_away():
    own = lambda number, frame: None   # noqa: E731
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal_list = cli.stop_on_signal()
    assert signal_list == [signal.SIGTERM] or signal.SIGTERM in signal_list
    signal.signal(signal.SIGTERM, own)
    cli.put_signal_back(signal_list)
    assert signal.getsignal(signal.SIGTERM) is own


def test_a_broker_serving_a_run_ends_once_the_process_of_that_run_has_ended(monkeypatch):
    """A host process that was killed outright stops nothing itself; the broker it
    left behind holds the credentials and would go on answering."""
    ended = subprocess.Popen([sys.executable, "-c", ""])
    ended.wait()
    monkeypatch.setattr(cli, "_RUN_WATCH_SEC", 0.05)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    cli.stop_on_signal()
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        cli._end_with_run(f"20260101-000000-1-{ended.pid}")
        time.sleep(5)
    # The process of a run that is going, and an id that names none, end nothing
    cli._end_with_run(f"20260101-000000-1-{os.getpid()}")
    cli._end_with_run("not-a-run")
    time.sleep(0.3)


def test_a_broker_is_told_to_watch_the_run_it_serves_and_no_other(monkeypatch, tmp_path):
    watch_list = []
    monkeypatch.setattr(cli, "_end_with_run", watch_list.append)
    monkeypatch.setattr(cli.broker_mod, "serve", lambda *a, **k: None)
    monkeypatch.setattr(cli.sandbox_mod, "stop_running_container", lambda why: None)
    cli.main(["broker", "--endpoint", "unix:///run/ctloom/llm.sock"])
    cli.main(["broker", "--endpoint", "unix:///run/ctloom/llm.sock", "--project",
              str(tmp_path / "p"), "--output-dir", str(tmp_path / "o"), "--run",
              "20260101-000000-1-1"])
    assert watch_list == ["20260101-000000-1-1"]


def test_a_broker_that_ends_takes_the_filling_it_started_away(monkeypatch, tmp_path):
    """The filling's container has a network, and nothing else knows it is going."""
    event_list = []
    served = {}

    def serve(endpoint, refill_cache=None, **kwargs):
        """Stand in for a broker that is stopped while a refill it was asked for is
        going."""
        served["refill"] = refill_cache
        threading.Thread(target=one_refill, daemon=True).start()
        going.wait(10)

    going, stop = threading.Event(), threading.Event()

    def one_refill():
        try:
            served["refill"](object())
        except RuntimeError as e:
            event_list.append(("refill ended", str(e)))

    def warm(*a, **k):
        going.set()
        stop.wait(10)
        time.sleep(0.2)
        event_list.append(("clone taken away", ""))
        raise KeyboardInterrupt("the run was stopped while ctloom-x was going")

    monkeypatch.setattr(cli.broker_mod, "serve", serve)
    monkeypatch.setattr(cli.sandbox_mod, "warm", warm)
    monkeypatch.setattr(cli.sandbox_mod, "stop_running_container",
                        lambda why: event_list.append(("stop", why)) or stop.set())
    cli.main(["broker", "--endpoint", "unix:///run/ctloom/llm.sock", "--project",
              str(tmp_path / "p"), "--output-dir", str(tmp_path / "o")])
    # main() came back only once the filling had ended
    assert event_list == [("stop", "the run was stopped"), ("clone taken away", ""),
                          ("refill ended", "the run was stopped while ctloom-x was going")]


@pytest.mark.parametrize("name, why", [
    ("SIGINT", "the run was stopped"),
    ("SIGTERM", "the broker was asked to end (SIGTERM)"),
    ("SIGHUP", "the broker was asked to end (SIGHUP)")])
def test_a_broker_says_what_ended_it_when_it_takes_a_filling_away(monkeypatch, name, why):
    """The SIGTERM a run sends the broker once the run has ended stops nobody's run."""
    seen = []

    def serve(endpoint, refill_cache=None, **kwargs):
        """Stand in for broker.serve(), which a stop ends quietly."""
        try:
            signal.raise_signal(getattr(signal, name))
            time.sleep(5)
        except KeyboardInterrupt:
            pass

    monkeypatch.setattr(cli.broker_mod, "serve", serve)
    monkeypatch.setattr(cli.sandbox_mod, "stop_running_container", seen.append)
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, signal.SIG_DFL)
    cli.main(["broker", "--endpoint", "unix:///run/ctloom/llm.sock"])
    assert seen == [why]


@pytest.mark.parametrize("when", ["while the container is taken away",
                                  "while the filling is waited for"])
def test_a_second_stop_while_a_broker_ends_still_lets_the_filling_take_its_clone_away(
        monkeypatch, capsys, tmp_path, when):
    """The SIGTERM the run sends a broker it stops, after the Ctrl-C that reached both
    of them, is held until the filling has taken its clone away. The broker then ends
    with one line and the status of SIGTERM, as the command does."""
    from codetwine_loom import __main__ as main_mod
    event_list = []
    going, stop = threading.Event(), threading.Event()

    def serve(endpoint, refill_cache=None, **kwargs):
        threading.Thread(target=lambda: one_refill(refill_cache),
                         daemon=True).start()
        going.wait(10)

    def one_refill(refill_cache):
        try:
            refill_cache(object())
        except RuntimeError:
            pass

    def warm(*a, **k):
        going.set()
        stop.wait(10)
        time.sleep(0.5)
        event_list.append("clone taken away")
        raise KeyboardInterrupt("the run was stopped while ctloom-x was going")

    def stop_running_container(why):
        event_list.append("stop")
        if when == "while the container is taken away":
            signal.raise_signal(signal.SIGTERM)
        else:
            threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGTERM)).start()
        stop.set()

    monkeypatch.setattr(cli.broker_mod, "serve", serve)
    monkeypatch.setattr(cli.sandbox_mod, "warm", warm)
    monkeypatch.setattr(cli.sandbox_mod, "stop_running_container", stop_running_container)
    monkeypatch.setattr(main_mod, "cwd_module_list", lambda: [])
    monkeypatch.setattr(sys, "argv", [
        "codetwine-loom", "broker", "--endpoint", "unix:///run/ctloom/llm.sock",
        "--project", str(tmp_path / "p"), "--output-dir", str(tmp_path / "o")])
    handler_before = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        with pytest.raises(SystemExit) as stop_end:
            main_mod.main()
    finally:
        signal.signal(signal.SIGTERM, handler_before)
    assert event_list == ["stop", "clone taken away"]
    assert stop_end.value.code == 128 + signal.SIGTERM
    err = capsys.readouterr().err
    assert err.splitlines()[-1:] == ["stopped by SIGTERM"] and "Traceback" not in err


@pytest.mark.parametrize("error", [OSError("address in use"), ValueError("bad address")])
def test_a_broker_that_cannot_listen_stops_with_the_message(monkeypatch, error):
    def refuse(endpoint, refill_cache=None, token_out="", token_left=0):
        raise error

    monkeypatch.setattr(cli.broker_mod, "serve", refuse)
    with pytest.raises(SystemExit, match=f"broker: {error}"):
        cli.main(["broker", "--endpoint", "unix:///run/ctloom/llm.sock"])


@pytest.mark.parametrize("argv, name", [
    (["sandbox", "--request", "r"], "run"),
    (["integrate", "--branch", "a"], "integrate_into_tree"),
    (["run", "--request", "r"], "dispatch"),
    (["warm"], "warm")])
def test_a_container_run_that_cannot_start_stops_with_the_message(monkeypatch, argv,
                                                                  name):
    def refuse(*a, **k):
        raise RuntimeError("podman is not on PATH")

    monkeypatch.setattr(cli.sandbox_mod, name, refuse)
    with pytest.raises(SystemExit, match=f"{argv[0]}: podman is not on PATH"):
        cli.main([*argv, "--project", "p"])


def test_warm_prints_the_cache_and_the_base_and_exits_with_its_status(monkeypatch,
                                                                       capsys):
    monkeypatch.setattr(cli.sandbox_mod, "warm", lambda *a, **k: {
        "cache": "/out/p/sandbox/cache", "base": "b" * 40, "exit_status": 3})
    with pytest.raises(SystemExit) as raised:
        cli.main(["warm", "--project", "p"])
    assert raised.value.code == 3
    out = capsys.readouterr().out
    assert "cache: /out/p/sandbox/cache" in out and f"base: {'b' * 40}" in out


# == the runs inside the container ==========================
def _settled(settled: bool) -> dict:
    """Build run_integration()'s return value."""
    return {"work": {"base": "b" * 40, "branches": [], "markers": [], "broken": [],
                     "unresolved": 0, "tests": "", "tests_at": 0,
                     "already_failing": None, "left": [], "checked_at": 0,
                     "decisions": [], "complete": True},
            "called": True, "left": [] if settled else ["- markers are still in: a.py"],
            "settled": settled, "answer": "settled f.txt", "history": [], "iters": 2,
            "refused_completes": 0, "harvested": False, "error": "", "outcome": None}


@pytest.fixture
def _integrate_here(tmp_path, monkeypatch):
    """Drive main(["integrate", "--here", ...]) with the merge itself substituted.

    Returns a function taking run_integration()'s result and returning the exit status.
    """
    project = _past_the_start_up(tmp_path, monkeypatch)
    prepared = {"base": "b" * 40,
                "merged": {"start": "b" * 40, "base": "b" * 40,
                           "branches": [{"branch": "a", "commit": "c" * 40}],
                           "parents": ["c" * 40]},
                "before": {}, "against": {},
                "was_on": "main"}
    seen = {}
    monkeypatch.setattr(cli.integrate_mod, "prepare",
                        lambda *a, **k: seen.update(prepare=(a, k)) or prepared)
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: seen.update(ensure=k, ensure_output_dir=a[1]) or
                        ("k.sqlite", project, "proj", True))
    monkeypatch.setattr(cli.edit, "test_change_set", lambda: set())

    def exit_status(result: dict, *argv: str) -> int:
        monkeypatch.setattr(cli.integrate_mod, "run_integration",
                            lambda *a, **k: seen.update(run=(a, k)) or result)
        with pytest.raises(SystemExit) as raised:
            cli.main(["integrate", "--here", "--project", project, "--output-dir",
                      str(tmp_path / "out"), "--branch", "a", "--branch", "b", *argv])
        return raised.value.code

    exit_status.seen = seen
    return exit_status


def test_integrate_here_merges_analyses_again_and_exits_0_when_settled(_integrate_here,
                                                                        capsys):
    assert _integrate_here(_settled(True), "--onto", "main") == 0
    seen = _integrate_here.seen
    assert seen["prepare"][0][2] == ["a", "b"]
    assert seen["prepare"][1]["onto"] == "main" and seen["prepare"][1]["resume"] == ""
    assert seen["prepare"][1]["output_dir"] == seen["ensure_output_dir"]
    assert "exclude_list" in seen["prepare"][1]
    assert seen["ensure"]["rebuild"] is True and seen["ensure"]["with_doc"] is False
    out = capsys.readouterr().out
    assert "settled: everything the merge left" in out
    assert "left in place: the working tree now holds the union" in out
    assert (f"git -C {os.path.abspath(seen['prepare'][0][0])} checkout main && "
            f"git -C {os.path.abspath(seen['prepare'][0][0])} reset --hard") in out


@pytest.mark.parametrize("stop_at", ["the index", "the merging stage"])
def test_integrate_here_stopped_before_its_union_is_what_it_leaves_puts_the_repository_back(
        _integrate_here, monkeypatch, tmp_path, stop_at):
    put_list = []
    monkeypatch.setattr(cli.integrate_mod, "put_back",
                        lambda *a: put_list.append(a) or True)
    assert _integrate_here(_settled(True)) == 0
    assert put_list == [] and cli._HERE_UNION_LIST == []
    project = os.path.abspath(_integrate_here.seen["prepare"][0][0])

    def stop(*a, **k):
        raise KeyboardInterrupt("the run was stopped (SIGTERM)")

    if stop_at == "the index":
        monkeypatch.setattr(cli.build, "build_and_save", stop)
    else:
        monkeypatch.setattr(cli.integrate_mod, "run_integration", stop)
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        cli.main(["integrate", "--here", "--project", project, "--output-dir",
                  str(tmp_path / "out"), "--branch", "a", "--branch", "b"])
    assert put_list == [(_integrate_here.seen["prepare"][0][0], "main")]
    assert cli._HERE_UNION_LIST == []


@pytest.mark.parametrize("fail_at", ["the analysis of the union", "the index"])
def test_integrate_here_that_fails_once_its_union_is_in_place_says_what_it_left(
        _integrate_here, monkeypatch, tmp_path, capsys, fail_at):
    put_list = []
    monkeypatch.setattr(cli.integrate_mod, "put_back",
                        lambda *a: put_list.append(a) or True)
    assert _integrate_here(_settled(True)) == 0
    capsys.readouterr()
    project = _integrate_here.seen["prepare"][0][0]
    if fail_at == "the index":
        def fail(*a, **k):
            raise RuntimeError("the index could not be built")

        monkeypatch.setattr(cli.build, "build_and_save", fail)
    else:
        real_ensure = cli.ensure_knowledge

        def ensure(*a, **k):
            if cli._HERE_UNION_LIST:
                raise RuntimeError("the analysis of the union failed")
            return real_ensure(*a, **k)

        monkeypatch.setattr(cli, "ensure_knowledge", ensure)
    with pytest.raises((SystemExit, RuntimeError)):
        cli.main(["integrate", "--here", "--project", project, "--output-dir",
                  str(tmp_path / "out"), "--branch", "a", "--branch", "b"])
    out = capsys.readouterr().out
    assert f"left in place: the working tree now holds the union, against {'b' * 12}" in out
    assert (f"to put the repository back: "
            f"{cli.integrate_mod.put_back_command(project, 'main')}") in out
    assert put_list == [] and cli._HERE_UNION_LIST == []


@pytest.mark.parametrize("end", ["settled", "failed", "stopped"])
def test_integrate_here_that_writes_an_outcome_does_not_say_what_it_left(
        _integrate_here, monkeypatch, tmp_path, capsys, end):
    """The run that writes --outcome-out prints neither what the repository holds nor the
    line that puts it back, settled or failing, and still puts it back when stopped."""
    put_list = []
    monkeypatch.setattr(cli.integrate_mod, "put_back",
                        lambda *a: put_list.append(a) or True)
    outcome = str(tmp_path / "outcome.json")
    if end == "settled":
        assert _integrate_here(_settled(True), "--outcome-out", outcome) == 0
    else:
        assert _integrate_here(_settled(True)) == 0
        project = _integrate_here.seen["prepare"][0][0]

        def fail(*a, **k):
            raise (RuntimeError("the merge failed") if end == "failed"
                   else KeyboardInterrupt("the run was stopped (SIGTERM)"))

        monkeypatch.setattr(cli.integrate_mod, "run_integration", fail)
        capsys.readouterr()
        with pytest.raises((SystemExit, KeyboardInterrupt)):
            cli.main(["integrate", "--here", "--project", project, "--output-dir",
                      str(tmp_path / "out"), "--branch", "a", "--branch", "b",
                      "--outcome-out", outcome])
    out = capsys.readouterr().out
    assert "left in place" not in out and "to put the repository back" not in out
    assert put_list == ([(project, "main")] if end == "stopped" else [])
    assert cli._HERE_UNION_LIST == []


def test_integrate_here_keeps_an_analysis_that_already_describes_the_union(
        _integrate_here, monkeypatch):
    """With one branch the union is that branch's tip, which prepare() analysed last."""
    seen = {}
    monkeypatch.setattr(cli, "union_analysed", lambda *a: seen.update(args=a) or True)
    _integrate_here(_settled(True))
    assert _integrate_here.seen["ensure"]["rebuild"] is False
    assert seen["args"][1] == "k.sqlite" and seen["args"][2] == "proj"


def test_the_union_is_analysed_again_where_there_is_no_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "analysis_match_tree",
                        lambda *a: pytest.fail("there is no analysis to compare"))
    assert cli.union_analysed(str(tmp_path), str(tmp_path / "none.sqlite"), "proj",
                              str(tmp_path / "out"), []) is False


def test_integrate_here_exits_1_when_the_merge_was_not_settled(_integrate_here):
    assert _integrate_here(_settled(False)) == 1


def test_integrate_here_writes_what_it_made_of_the_merge_where_asked(_integrate_here,
                                                                     tmp_path):
    outcome = tmp_path / "outcome.json"
    _integrate_here(_settled(False), "--outcome-out", str(outcome))
    assert json.loads(outcome.read_text(encoding="utf-8")) == {
        "verdict": "unsettled", "why": "- markers are still in: a.py",
        "did": "settled f.txt", "tests_changed": [], "ended": "turns exhausted",
        "decisions": [], "parents": ["c" * 40], "separator": {}, "no_merge": []}


def test_integrate_here_writes_the_sides_not_merged_before_the_model_is_called(
        _integrate_here, tmp_path, monkeypatch):
    """What a container ending in the model leaves for the host; the end of the run
    writes the file again whole."""
    outcome = tmp_path / "outcome.json"
    record_list = [["c" * 40, "g.txt", "f.txt"]]
    monkeypatch.setattr(cli.integrate_mod, "no_merge_record_list", lambda merged: record_list)
    seen = {}

    @contextlib.contextmanager
    def open_run():
        seen["outcome"] = json.loads(outcome.read_text(encoding="utf-8"))
        yield

    monkeypatch.setattr(cli.turnlog, "open_run", open_run)
    assert _integrate_here(_settled(True), "--outcome-out", str(outcome)) == 0
    assert seen["outcome"] == {"no_merge": record_list}
    assert "no_merge" not in json.loads(outcome.read_text(encoding="utf-8"))
    assert _integrate_here(_settled(False), "--outcome-out", str(outcome)) == 1
    assert seen["outcome"] == {"no_merge": record_list}
    assert json.loads(outcome.read_text(encoding="utf-8"))["no_merge"] == record_list


def _split(ok: bool, **override) -> dict:
    """Build run_dispatch()'s return value."""
    result = {"ok": ok, "scope": {}, "iters": 3, "refused_completes": 0,
              "harvested": False, "history": [], "error": "", "outcome": None}
    result.update(override)
    return result


@pytest.fixture
def _run_here(tmp_path, monkeypatch):
    """Drive main(["run", "--here", ...]) with the splitting stage substituted.

    Returns a function taking run_dispatch()'s result and returning the exit status.
    """
    project = _past_the_start_up(tmp_path, monkeypatch)
    seen = {}

    def exit_status(result: dict, *argv: str) -> int:
        monkeypatch.setattr(cli.dispatch_mod, "run_dispatch",
                            lambda request, resume_list=None: seen.update(
                                request=request, resume=resume_list) or result)
        with pytest.raises(SystemExit) as raised:
            cli.main(["run", "--here", "--project", project, "--output-dir",
                      str(tmp_path / "out"), "--request", "split me", *argv])
        return raised.value.code

    exit_status.seen = seen
    return exit_status


def test_run_here_prints_the_tasks_and_writes_them_where_asked(_run_here, tmp_path,
                                                               capsys):
    task_out = tmp_path / "tasks.json"
    status = _run_here(_split(True, tasks=["do this", "do that"]), "--tasks-out",
                       str(task_out), "--resume", "run/old/1")
    assert status == 0
    assert _run_here.seen == {"request": "split me", "resume": ["run/old/1"]}
    out = capsys.readouterr().out
    assert "tasks: 2" in out and "  1. do this" in out and "  2. do that" in out
    assert json.loads(task_out.read_text(encoding="utf-8")) == {
        "tasks": ["do this", "do that"]}


def test_run_here_prints_the_questions(_run_here, capsys):
    assert _run_here(_split(True, questions=["where?"], refused_completes=1,
                            harvested=True)) == 0
    out = capsys.readouterr().out
    assert "1 sent back" in out and "questions: 1 (harvested" in out


def test_run_here_exits_1_when_the_split_did_not_come_back(_run_here, capsys):
    assert _run_here(_split(False, error="the splitting stage handed nothing over")) == 1
    out = capsys.readouterr().out
    assert "split: none" in out and "handed nothing over" in out


def test_run_here_keeps_out_what_the_analysis_keeps_out(_run_here, monkeypatch):
    seen = []

    def record(project_dir, out_dir, name, exclude_list):
        seen.append(exclude_list)
        raise RuntimeError("the analysis keeps out vendor and this run asks to keep "
                           "out other")

    monkeypatch.setattr(cli.sandbox_mod, "exclude_list_of_record", record)
    status = _run_here(_split(True, tasks=["x"]), "--exclude", "other")
    assert str(status).startswith("run: the analysis keeps out vendor")
    assert seen and "other" in seen[0] and _run_here.seen == {}


# == an analysis that cannot be made again ==================
def test_an_analysis_that_cannot_be_made_again_stops_with_the_message(tmp_path,
                                                                      monkeypatch):
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: False)
    call_count = []

    def ensure(*a, **k):
        call_count.append(k)
        if k.get("rebuild"):
            raise RuntimeError("the documents follow the template DOC_TEMPLATE_PATH names")
        return ("k.sqlite", project, "proj", True)

    monkeypatch.setattr(cli, "ensure_knowledge", ensure)
    monkeypatch.setattr(cli, "search_scope", lambda request: {})
    with pytest.raises(SystemExit, match="DOC_TEMPLATE_PATH"):
        cli.main(["search", "--project", project, "--output-dir", str(tmp_path / "out"),
                  "--request", "x"])
    assert len(call_count) == 2


# == declining to carry on from an unfinished run ===========
def test_declining_to_carry_on_closes_the_earlier_attempt(_edit_main, monkeypatch):
    monkeypatch.setattr(cli.history, "open_attempt", lambda: {"n": 1, "request": "r"})
    monkeypatch.setattr(cli, "_want_resume", lambda entry_dict, choice: False)
    closed = []
    monkeypatch.setattr(cli.history, "resolve",
                        lambda verdict, note="": closed.append((verdict, note)))
    _edit_main(_ran("done", False, history.VERDICT_NO_PROOF))
    assert closed == [("closed", "the user chose to start fresh")]


def test_integrate_hands_the_branch_to_carry_on_from_to_the_container_side(monkeypatch):
    asked = {}
    monkeypatch.setattr(cli.sandbox_mod, "integrate_into_tree",
                        lambda *a, **k: asked.update(k, branch_list=a[2]) or {"exit_status": 0})
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit):
        cli.main(["integrate", "--project", "p", "--resume", "sandbox/earlier"])
    assert asked["resume"] == "sandbox/earlier" and asked["branch_list"] == []


def test_integrate_without_a_branch_or_a_branch_to_carry_on_from_is_refused(capsys):
    with pytest.raises(SystemExit):
        cli.main(["integrate", "--project", "p"])
    assert "--resume" in capsys.readouterr().err


def test_integrate_here_combines_the_branch_to_carry_on_from_first(_integrate_here):
    _integrate_here(_settled(True), "--resume", "sandbox/earlier")
    assert _integrate_here.seen["prepare"][0][2] == ["sandbox/earlier", "a", "b"]
    assert _integrate_here.seen["prepare"][1]["resume"] == "sandbox/earlier"


# == the names and the paths kept out =======================================
@pytest.mark.parametrize("name", ["/abs/vendor", "../logs", "src/../logs", "", "./"])
def test_a_path_outside_the_repository_cannot_be_kept_out(name):
    with pytest.raises(SystemExit) as e:
        cli.exclude_name_list(argparse.Namespace(exclude_list=[name]))
    assert "paths relative to the repository" in str(e.value)


def test_a_path_from_the_root_is_kept_out_in_one_form(monkeypatch):
    monkeypatch.setattr(cli.config, "BUILD_EXCLUDE", ["apps/old/"])
    assert cli.exclude_name_list(argparse.Namespace(
        exclude_list=["./apps//old", "./vendor", "vendor/", "apps/*/gen"])) == [
        "apps/old", "./vendor", "vendor", "apps/*/gen"]


def test_the_names_kept_out_are_the_setting_and_the_arguments(monkeypatch):
    monkeypatch.setattr(cli.config, "BUILD_EXCLUDE", ["logs"])
    assert cli.exclude_name_list(
        argparse.Namespace(exclude_list=["vendor", "logs"])) == ["logs", "vendor"]


# == what the design documents need is checked before anything is analysed ===
def test_writing_the_documents_needs_the_credentials_even_with_a_broker(tmp_path,
                                                                        monkeypatch):
    """codetwine calls the provider itself, so a broker does not answer for it."""
    monkeypatch.setattr(cli.config, "LLM_BROKER", "unix:///tmp/ctloom.sock")
    monkeypatch.setattr(cli.llm, "missing_credential_text", lambda: "no key")
    monkeypatch.setattr(cli, "_ask_env_file", lambda missing: False)
    with pytest.raises(SystemExit) as e:
        cli.main(["build", "--doc-create", "--project", str(tmp_path),
                  "--output-dir", str(tmp_path / "out")])
    assert "no key" in str(e.value)


def test_a_build_without_the_documents_leaves_the_calls_to_the_broker(tmp_path,
                                                                      monkeypatch):
    monkeypatch.setattr(cli.config, "LLM_BROKER", "unix:///tmp/ctloom.sock")
    monkeypatch.setattr(cli.llm, "missing_credential_text",
                        lambda: pytest.fail("the credentials were asked for"))
    monkeypatch.setattr(cli, "ensure_knowledge",
                        lambda *a, **k: (_ for _ in ()).throw(SystemExit("past it")))
    with pytest.raises(SystemExit) as e:
        cli.main(["build", "--project", str(tmp_path),
                  "--output-dir", str(tmp_path / "out")])
    assert "past it" in str(e.value)


def test_a_run_inside_a_container_does_not_find_its_own_run(tmp_path, monkeypatch):
    """The analysis in a container is that run's own copy, which nothing else copies."""
    monkeypatch.setenv(sandbox_mod.CTLOOM_CLONE_ENV, "/work/proj")
    monkeypatch.setattr(cli.sandbox_mod, "live_run_list",
                        lambda *a, **k: pytest.fail("the runs were looked for"))
    cli.refuse_while_run_live("integrate", str(tmp_path), str(tmp_path / "out"))


# == the arguments every subcommand is checked on ===========
@pytest.mark.parametrize("command", ["search", "edit", "run", "answer", "sandbox"])
def test_a_request_of_whitespace_is_one_line_not_a_traceback(tmp_path, command):
    with pytest.raises(SystemExit) as e:
        cli.main([command, "--project", str(tmp_path), "--request", "   "])
    assert str(e.value).startswith(f"{command}: the request is empty. Pass it with ")


_PROJECT_ARG_DICT = {
    "build": [], "overview": [], "search": ["--request", "x"], "answer": ["--request", "x"],
    "edit": ["--request", "x"], "sandbox": ["--request", "x"], "run": ["--request", "x"],
    "integrate": ["--branch", "a", "--branch", "b"], "test-create": ["--base", "HEAD"],
    "refactor": ["--base", "HEAD"], "test-refactor": ["--base", "HEAD"], "warm": [],
    "run-tests": [], "history": [], "format": []}


@pytest.mark.parametrize("command", sorted(_PROJECT_ARG_DICT))
def test_a_subcommand_without_a_target_says_how_to_name_one(monkeypatch, command):
    monkeypatch.setattr(config, "PROJECT_DIR", "")
    monkeypatch.setattr(cli, "_check_credential", lambda args, argv: args)
    with pytest.raises(SystemExit) as e:
        cli.main([command, *_PROJECT_ARG_DICT[command]])
    assert str(e.value) == (f"{command}: pass --project, or set PROJECT_DIR "
                            f"(--project . for the current directory)")


def test_every_subcommand_but_broker_and_setup_is_asked_for_a_target():
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest="command")
    common = cli._common_parser()
    for add in (cli._add_build_parser, cli._add_search_parser, cli._add_run_parser,
                cli._add_answer_parser, cli._add_change_parser, cli._add_integrate_parser,
                cli._add_stage_parser, cli._add_tool_parser):
        add(sub_parser, common)
    assert set(sub_parser.choices) == {*_PROJECT_ARG_DICT, "broker", "setup"}


def test_broker_needs_no_target(monkeypatch):
    cli._refuse_no_project(argparse.Namespace(command="broker", project=""))


def test_the_current_directory_is_named_with_a_dot():
    cli._refuse_no_project(argparse.Namespace(command="history", project="."))


def test_an_answered_file_of_another_shape_is_one_line(tmp_path):
    path = tmp_path / "answered.json"
    path.write_text(json.dumps([{"request": "q"}]), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli.read_answer_list(str(path))
    assert "entry 0 is not" in str(e.value)


def test_an_answered_file_of_the_right_shape_is_read(tmp_path):
    path = tmp_path / "answered.json"
    entry_list = [{"n": 1, "request": "q", "answer": "a", "error": ""}]
    path.write_text(json.dumps(entry_list), encoding="utf-8")
    assert cli.read_answer_list(str(path)) == entry_list


@pytest.mark.parametrize("value, shown", [("", "itself empty)"),
                                          ("HOW.md", "itself HOW.md)")])
def test_the_guide_s_help_says_what_the_setting_holds(monkeypatch, capsys, value, shown):
    monkeypatch.setattr(config, "EDIT_GUIDE", value)
    with pytest.raises(SystemExit):
        cli._parse_args(["edit", "--help"])
    assert shown in " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("name, value", [("PRESET", "bogus"), ("TEST_PATH", "/abs")])
def test_a_setting_config_refuses_stops_the_command_with_one_line(tmp_path, name, value):
    proc = subprocess.run(
        [sys.executable, "-m", "codetwine_loom", "history", "--project", str(tmp_path)],
        cwd=tmp_path, env={**os.environ, name: value}, capture_output=True, text=True,
        check=False)
    assert proc.returncode == 1
    assert proc.stderr.strip().splitlines() == [proc.stderr.strip()]
    assert proc.stderr.startswith(f"{name} must be")


def test_importing_the_package_reads_no_setting():
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, codetwine_loom; print('codetwine_loom.config' in sys.modules)"],
        env={**os.environ, "PRESET": "bogus"}, capture_output=True, text=True, check=False)
    assert proc.stdout.strip() == "False"


@pytest.mark.parametrize("guide", ["/abs/guide.md", "rel.md"])
def test_sandbox_warns_about_the_guide_it_hands_the_container(tmp_path, monkeypatch, guide):
    monkeypatch.setattr(config, "EDIT_GUIDE", "/elsewhere/env.md")
    seen = []

    def run(*a, **k):
        seen.append((config.EDIT_GUIDE, k["guide"]))
        raise RuntimeError("stop")

    monkeypatch.setattr(cli.sandbox_mod, "run", run)
    with pytest.raises(SystemExit, match="stop"):
        cli._start_sandbox(cli._parse_args(["sandbox", "--project", str(tmp_path),
                                            "--request", "x", "--guide", guide]))
    assert seen == [(guide, guide)]


def test_edit_resume_with_the_history_off_says_it_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EDIT_HISTORY", False)
    args = cli._parse_args(["edit", "--project", str(tmp_path), "--request", "r",
                            "--resume"])
    with pytest.raises(SystemExit, match="EDIT_HISTORY is false"):
        cli._finish_edit(args, str(tmp_path / "k.sqlite"), [])


def _finish_edit_with(monkeypatch, tmp_path, flag_list, during=None):
    """Run cli._finish_edit() over a stand-in run and return what it printed with.

    Args:
        flag_list: The flags of edit after --project and --request.
        during: Called in place of the run, for what the run would be handed.
    """
    monkeypatch.setattr(cli, "create_agent", lambda: None)
    monkeypatch.setattr(cli.edit, "git_diff_stat", lambda: {
        "files": [], "untracked": [], "total_insertions": 0, "total_deletions": 0})
    monkeypatch.setattr(cli, "print_scope", lambda scope_dict: None)
    monkeypatch.setattr(cli, "check_start", lambda path, exclude_list: None)
    monkeypatch.setattr(cli, "run_edit", lambda rlm, request, start_dict: (
        during() if during else None) or _ran("renamed it", False,
                                              history.VERDICT_NO_PROOF))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    args = cli._parse_args(["edit", "--project", str(tmp_path), "--request", "r",
                            *flag_list])
    with pytest.raises(SystemExit) as caught:
        cli._finish_edit(args, str(tmp_path / "k.sqlite"), [])
    return caught.value.code


def test_edit_resume_with_nothing_to_carry_on_from_says_so_and_goes_on(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "EDIT_HISTORY", True)
    monkeypatch.setattr(source, "project_name", "codetwine")
    line = ("--resume: no unfinished change is recorded to carry on from, so this run "
            "starts as a new one")
    assert _finish_edit_with(monkeypatch, tmp_path, ["--resume"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == line
    for flag_list in ([], ["--no-resume"]):
        assert _finish_edit_with(monkeypatch, tmp_path, flag_list) == 0
        assert line not in capsys.readouterr().out


def test_edit_resume_with_something_to_carry_on_from_says_nothing_of_it(
        unfinished, tmp_path, monkeypatch, capsys):
    assert _finish_edit_with(monkeypatch, tmp_path, ["--resume"]) == 0
    assert "no unfinished change" not in capsys.readouterr().out
    assert history.load()["attempts"][-1]["verified"] is None


def test_a_run_that_starts_fresh_closes_the_unfinished_change(unfinished, tmp_path,
                                                              monkeypatch):
    seen = []
    _finish_edit_with(monkeypatch, tmp_path, [],
                      during=lambda: seen.append(history.render("r")))
    assert seen == [""]
    assert history.load()["attempts"][-1]["verified"] == history.VERDICT_CLOSE
    assert history.open_attempt() is None


def test_a_dry_run_that_starts_fresh_leaves_the_unfinished_change_open(
        unfinished, tmp_path, monkeypatch):
    """This run is not handed it, and the run after this one is."""
    seen = []
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    _finish_edit_with(monkeypatch, tmp_path, [],
                      during=lambda: seen.append(history.render("r")))
    assert seen == [""]
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    monkeypatch.setattr(config, "EDIT_HISTORY", True)
    assert history.load()["attempts"][-1]["verified"] is None
    assert history.open_attempt()["n"] == 1 and history.render("r") != ""


def test_a_dry_run_that_carries_on_is_handed_the_unfinished_change(
        unfinished, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    _finish_edit_with(monkeypatch, tmp_path, ["--resume"],
                      during=lambda: seen.append(history.render("r")))
    assert seen[0].startswith("## ") and config.EDIT_HISTORY is True


def test_a_broker_refills_only_for_the_project_it_is_given(monkeypatch, capsys):
    monkeypatch.setattr(config, "PROJECT_DIR", "/some/repo")
    assert cli._parse_args(["broker", "--endpoint", "unix:///tmp/x.sock"]).project == ""
    assert cli._parse_args(["broker", "--endpoint", "unix:///tmp/x.sock", "--project",
                            "/p"]).project == "/p"
    with pytest.raises(SystemExit):
        cli._parse_args(["broker", "--endpoint", "unix:///tmp/x.sock", "--run", "r"])
    assert "only with --project" in capsys.readouterr().err


# == the record of a run and the token counts =================================

def _record_list(tmp_path, name="repo"):
    """Return the records the commands of a test wrote, oldest first."""
    return cli.runrecord.read_list(
        str(tmp_path / "out" / "agents" / ".repo" / name / "record"))


def _git_repo(path):
    """Make a git repository with one commit at path and return it as a str."""
    path.mkdir()
    for argv in (["init", "-q", "-b", "main"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one",
                  "--allow-empty"]):
        subprocess.run(["git", "-C", str(path), *argv], check=True)
    return str(path)


def _work_branch(repo, name, request="r"):
    """Point a branch at a commit over HEAD that adds a file the working tree does not
    hold and says what request it was made for, the index and the working tree left
    as they are."""
    env = {**os.environ, "GIT_INDEX_FILE": os.path.join(repo, ".git", "work-index")}
    blob = subprocess.run(["git", "-C", repo, "hash-object", "-w", "--stdin"],
                          input="work\n", capture_output=True, text=True,
                          check=True).stdout.strip()
    subprocess.run(["git", "-C", repo, "update-index", "--add", "--cacheinfo",
                    f"100644,{blob},work.txt"], env=env, check=True)
    tree = subprocess.run(["git", "-C", repo, "write-tree"], env=env, capture_output=True,
                          text=True, check=True).stdout.strip()
    os.remove(env["GIT_INDEX_FILE"])
    commit = subprocess.run(
        ["git", "-C", repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit-tree",
         tree, "-p", "HEAD", "-m", f"work\n\nCodetwine-Loom-Run-Request: {request}"],
        capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "-C", repo, "branch", "-f", name, commit], check=True)


def test_a_dispatching_run_leaves_a_record_of_itself(capsys, monkeypatch, tmp_path):
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/run/r/1")
    result = {"tasks": ["one"], "dropped": [], "swept": [], "exit_status": 1,
              "integration": None,
              "resume": ["codetwine-loom/run/r/1", "codetwine-loom/run/r/gone"],
              "branches": [{"n": 1, "request": "one", "branch": "codetwine-loom/run/r/1",
                            "attempt": 1, "commit": "c" * 40, "exit_status": 1,
                            "error": "", "verdict": "failed"}]}

    def dispatch(*a, **k):
        cli.tokencount.add({"input_token": 1200, "output_token": 300}, part="edit r")
        return result

    monkeypatch.setattr(cli.sandbox_mod, "dispatch", dispatch)
    monkeypatch.setattr(cli.report_mod, "report", lambda *a, **k: "it did not finish")
    with pytest.raises(SystemExit):
        cli.main(["run", "--request", "依頼", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    out = capsys.readouterr().out
    record_dict, = _record_list(tmp_path)
    assert (record_dict["command"], record_dict["request"]) == ("run", "依頼")
    assert (record_dict["exit_status"], record_dict["error"]) == (1, "")
    assert record_dict["report"] == "it did not finish"
    # Only a branch that is still there is named to carry on from
    assert record_dict["resume"] == ["codetwine-loom/run/r/1"]
    assert record_dict["token"]["total"] == 1500 and record_dict["token"]["limit"] == 0
    assert record_dict["token"]["parts"][0]["part"] == "edit r"
    assert record_dict["result"]["branches"][0]["verdict"] == "failed"
    assert record_dict["elapsed_sec"] >= 0 and record_dict["start"] <= record_dict["end"]
    assert "tokens: 1,500 in 1 call(s): 1,200 input" in out
    assert f"record: {record_dict['path']}" in out


@pytest.mark.parametrize("status", [124, 137, 1, 0])
def test_a_run_whose_reading_container_left_no_split_ends_with_that_container_s_status(
        capsys, monkeypatch, tmp_path, status):
    """The record, the line printed and the exit status say how the container ended."""
    repo = _git_repo(tmp_path / "repo")
    read_error = RuntimeError("the run that reads the request left no split in /x")

    def dispatch(*a, **k):
        if status:
            raise cli.sandbox_mod.ContainerEndError(read_error, status)
        raise read_error

    monkeypatch.setattr(cli.sandbox_mod, "dispatch", dispatch)
    with pytest.raises(SystemExit) as raised:
        cli.main(["run", "--request", "r", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    why = str(read_error) if not status else \
        f"the container ended ({cli.sandbox_mod.status_text(status)}), and {read_error}"
    assert (record_dict["exit_status"], record_dict["error"]) == (status or 1, why)
    if status in (0, 1):
        assert raised.value.code == f"run: {why}"
    else:
        assert raised.value.code == status
        assert f"run: {why}\n" in capsys.readouterr().err


@pytest.mark.parametrize("name, argv", [
    ("run", ["sandbox", "--request", "r"]),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"])])
def test_a_sandboxed_run_and_a_merge_leave_a_record_with_what_reached_the_tree(
        monkeypatch, name, argv, tmp_path, capsys):
    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.sandbox_mod, name, lambda *a, **k: {
        "exit_status": 0, "land_why": "", "branch": None,
        "land": {"files": 2, "paths": ["a.py", "b.py"], "patch": "p", "last": "l"}})
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 5000)
    with pytest.raises(SystemExit) as raised:
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert raised.value.code == 0 and record_dict["exit_status"] == 0
    assert record_dict["command"] == argv[0]
    assert record_dict["files"] == ["a.py", "b.py"] and record_dict["resume"] == []
    lines = capsys.readouterr().out.splitlines()
    assert lines[-3].endswith(", of MAX_TOKEN_PER_RUN=5,000")
    assert lines[-2].startswith("record: ") and lines[-1].startswith("to take it back")


class _GoneReader:
    """A standard output whose reader has gone: every write raises BrokenPipeError."""

    def write(self, text):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


@pytest.mark.parametrize("name, argv, result_dict", [
    ("dispatch", ["run", "--request", "r"],
     {"exit_status": 3, "tasks": [], "branches": [], "resume": [], "dropped": [],
      "integration": None}),
    ("run", ["sandbox", "--request", "r"],
     {"exit_status": 3, "land_why": "", "branch": None, "base": "b" * 40, "patch": None,
      "commit": None, "land": None}),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"],
     {"exit_status": 3, "land_why": "", "branch": None, "base": "b" * 40, "patch": None,
      "commit": None, "land": None})])
def test_a_command_whose_reader_went_away_is_recorded_as_it_ended(
        monkeypatch, name, argv, result_dict, tmp_path):
    """Lines nobody reads are no stop: the record holds the command's own status and
    no error, and the process ends with that status."""
    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.sandbox_mod, name, lambda *a, **k: dict(result_dict))
    monkeypatch.setattr(cli.report_mod, "report", lambda *a, **k: "the report")
    monkeypatch.setattr(sys, "stdout", _GoneReader())
    with pytest.raises(SystemExit) as raised:
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert raised.value.code == 3
    assert (record_dict["exit_status"], record_dict["error"]) == (3, "")


def test_a_listing_whose_reader_went_away_ends_as_it_would_have(monkeypatch, tmp_path):
    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(sys, "stdout", _GoneReader())
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])


@pytest.mark.parametrize("name, argv", [
    ("dispatch", ["run", "--request", "r"]), ("run", ["sandbox", "--request", "r"]),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"])])
def test_a_command_that_was_stopped_leaves_a_record_saying_why(monkeypatch, name, argv,
                                                              tmp_path):
    def stop(*a, **k):
        raise RuntimeError("no engine")

    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.sandbox_mod, name, stop)
    with pytest.raises(SystemExit, match=f"{argv[0]}: no engine"):
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert (record_dict["exit_status"], record_dict["error"]) == (1, "no engine")
    assert record_dict["result"] == {"earlier_land": [], "resume": []}


@pytest.mark.parametrize("name, argv", [
    ("dispatch", ["run", "--request", "r"]), ("run", ["sandbox", "--request", "r"]),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"])])
def test_a_command_starts_without_what_an_earlier_one_left_in_the_process(
        monkeypatch, name, argv, tmp_path):
    """What an earlier command put into a working tree, a stop it was given and its
    token counts are not this command's."""
    seen = {}

    def stop(*a, **k):
        seen["stop"] = cli.sandbox_mod._STOP_EVENT.is_set()
        cli.tokencount.add({"input_token": 70, "output_token": 5})
        raise RuntimeError("no engine")

    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.sandbox_mod, "_LAND_LIST", [
        {"files": 1, "paths": ["earlier.py"], "patch": "p", "last": "l"}])
    cli.sandbox_mod._STOP_EVENT.set()
    cli.tokencount.add({"input_token": 1000, "output_token": 1000})
    monkeypatch.setattr(cli.sandbox_mod, name, stop)
    with pytest.raises(SystemExit, match="no engine"):
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert record_dict["files"] == [] and record_dict["result"]["earlier_land"] == []
    assert seen["stop"] is False and record_dict["token"]["total"] == 75


def test_where_the_containers_stdout_goes_is_put_back_when_the_command_ends(
        monkeypatch, tmp_path):
    seen = {}

    def dispatch(*a, **k):
        seen["to_file"] = cli.progress._CHILD_STDOUT_TO_FILE
        raise RuntimeError("no engine")

    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", dispatch)
    monkeypatch.setattr(cli.progress, "_CHILD_STDOUT_TO_FILE", False)
    with pytest.raises(SystemExit, match="no engine"):
        cli.main(["run", "--request", "r", "--project", repo,
                  "--output-dir", str(tmp_path / "out")])
    assert seen["to_file"] is True
    assert cli.progress._CHILD_STDOUT_TO_FILE is False


@pytest.mark.parametrize("name, argv", [
    ("dispatch", ["run", "--request", "r"]), ("run", ["sandbox", "--request", "r"]),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"])])
@pytest.mark.parametrize("text, error", [("", "the run was stopped"),
                                         ("stopped while x was going",
                                          "stopped while x was going")])
def test_a_command_stopped_from_the_keyboard_leaves_a_record_too(monkeypatch, name, argv,
                                                                tmp_path, text, error):
    def stop(*a, **k):
        cli.tokencount.add({"input_token": 70, "output_token": 5})
        raise KeyboardInterrupt(text)

    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.sandbox_mod, name, stop)
    with pytest.raises(KeyboardInterrupt):
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert (record_dict["exit_status"], record_dict["error"]) == (130, error)
    assert record_dict["token"]["total"] == 75


def test_a_run_stopped_while_its_report_is_written_leaves_its_record_and_its_lines(
        monkeypatch, tmp_path, capsys):
    def stop(*a, **k):
        raise KeyboardInterrupt

    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/run/r/1")
    result = {"tasks": ["one"], "dropped": [], "swept": [], "exit_status": 0,
              "resume": ["codetwine-loom/run/r/1"], "branches": [],
              "integration": {"land_why": "", "branch": None, "land": {
                  "files": 1, "paths": ["a.py"], "patch": "p", "last": "l.diff"}}}
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", lambda *a, **k: result)
    monkeypatch.setattr(cli, "print_dispatch", lambda project_dir, result_dict: None)
    monkeypatch.setattr(cli.report_mod, "report", stop)
    monkeypatch.setattr(cli.sandbox_mod, "remain_branch_list",
                        lambda *a: pytest.fail("the result names the branches"))
    with pytest.raises(KeyboardInterrupt):
        cli.main(["run", "--request", "r", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert (record_dict["exit_status"], record_dict["error"]) == (130, "the run was stopped")
    assert record_dict["files"] == ["a.py"] and record_dict["resume"] == ["codetwine-loom/run/r/1"]
    out = capsys.readouterr().out
    assert "to take it back out of the working tree" in out
    assert "to carry on from what this run left: run --resume codetwine-loom/run/r/1" in out


@pytest.mark.parametrize("name, argv", [
    ("run", ["sandbox", "--request", "r"]),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"])])
@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("no engine")])
def test_a_command_stopped_after_it_returned_keeps_what_it_returned_in_its_record(
        monkeypatch, name, argv, tmp_path, error, capsys):
    def stop(*a, **k):
        raise error

    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.sandbox_mod, name, lambda *a, **k: {
        "exit_status": 0, "land_why": "", "branch": None,
        "land": {"files": 1, "paths": ["a.py"], "patch": "p", "last": "l"}})
    monkeypatch.setattr(cli.sandbox_mod, "remove_old_run_in", stop)
    with pytest.raises((KeyboardInterrupt, SystemExit)):
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert record_dict["files"] == ["a.py"] and record_dict["error"]
    assert record_dict["exit_status"] == (130 if isinstance(error, KeyboardInterrupt) else 1)
    assert "to take it back out of the working tree" in capsys.readouterr().out


@pytest.mark.parametrize("name, argv", [
    ("dispatch", ["run", "--request", "r"]), ("run", ["sandbox", "--request", "r"]),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"])])
@pytest.mark.parametrize("token_count, status", [(1100, 1), (999, 0)])
def test_a_command_that_got_through_with_the_token_limit_reached_ends_with_status_1(
        monkeypatch, name, argv, tmp_path, capsys, token_count, status):
    def reach(*a, **k):
        cli.tokencount.add({"input_token": token_count - 200, "output_token": 200})
        return {"exit_status": 0, "land_why": "", "branch": None, "land": None,
                "tasks": [], "branches": [], "integration": None, "resume": []}

    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 1000)
    monkeypatch.setattr(cli.sandbox_mod, name, reach)
    monkeypatch.setattr(cli.sandbox_mod, "remove_old_run_in", lambda *a, **k: [])
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result_dict: None)
    monkeypatch.setattr(cli, "print_dispatch", lambda project_dir, result_dict: None)
    monkeypatch.setattr(cli.report_mod, "report", lambda *a, **k: "")
    with pytest.raises(SystemExit) as raised:
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    assert raised.value.code == status
    out = capsys.readouterr().out
    line = ("token limit: the model calls of this run have used 1100 token(s), which "
            "reaches MAX_TOKEN_PER_RUN=1000, so the run ends with exit status 1")
    assert (line in out.splitlines()) == bool(status)
    record_dict, = _record_list(tmp_path)
    assert record_dict["exit_status"] == status
    assert ("MAX_TOKEN_PER_RUN=1000" in record_dict["result"].get("token_stop", "")) \
        == bool(status)


def test_the_token_counts_of_processes_not_waited_for_are_in_the_stop_record(
        monkeypatch, tmp_path):
    def stop(*a, **k):
        path = cli.tokencount.child_file()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"call": 2, "input_token": 30, "output_token": 5}, handle)
        raise KeyboardInterrupt

    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", stop)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["run", "--request", "r", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert record_dict["token"]["total"] == 35
    assert record_dict["token"]["parts"][0]["part"] == "not waited for"


@pytest.mark.parametrize("name, argv, told", [
    ("dispatch", ["run", "--resume", "codetwine-loom/run/r/1", "--resume",
                  "codetwine-loom/run/r/gone"],
     ["codetwine-loom/run/r/1"]),
    ("integrate_into_tree", ["integrate", "--resume", "codetwine-loom/run/r/1"],
     ["codetwine-loom/run/r/1"]),
    ("run", ["sandbox", "--request", "r"], [])])
def test_a_command_stopped_before_it_left_a_branch_names_the_ones_it_was_passed(
        monkeypatch, tmp_path, name, argv, told):
    def stop(*a, **k):
        raise KeyboardInterrupt

    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/run/r/1")
    monkeypatch.setattr(cli.sandbox_mod, name, stop)
    monkeypatch.setattr(cli, "request_of_run", lambda project_dir, request, resume: "r")
    monkeypatch.setattr(cli, "request_of_integrate", lambda *a, **k: "r")
    with pytest.raises(KeyboardInterrupt):
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert record_dict["resume"] == told


@pytest.mark.parametrize("name, argv", [
    ("dispatch", ["run", "--request", "r"]), ("run", ["sandbox", "--request", "r"]),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"])])
def test_a_command_that_raised_something_else_leaves_a_record_and_raises_it(
        monkeypatch, name, argv, tmp_path):
    def stop(*a, **k):
        raise ValueError("no such target repository: /p")

    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.sandbox_mod, name, stop)
    with pytest.raises(ValueError, match="no such target repository"):
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert (record_dict["exit_status"], record_dict["error"]) == (
        1, "ValueError: no such target repository: /p")


@pytest.mark.parametrize("resume_list, resume", [
    (["codetwine-loom/sandbox/u0", "codetwine-loom/run/old/1"], ""),
    ([], "codetwine-loom/sandbox/u0")])
def test_a_stopped_command_whose_runs_left_no_branch_names_the_ones_it_was_passed(
        monkeypatch, tmp_path, resume_list, resume):
    """A command stopped before its runs left a branch of their own names the
    branches it was passed to carry on from, by run (--resume given more than once)
    and by integrate alike."""
    monkeypatch.setattr(cli.sandbox_mod, "remain_branch_list",
                        lambda project_dir, out_dir, since_id: [])
    monkeypatch.setattr(cli.sandbox_mod, "land_list", lambda: [])
    args = argparse.Namespace(project=str(tmp_path), output_dir=str(tmp_path / "out"),
                              resume_list=resume_list, resume=resume, request="r")
    assert cli._stop_result_dict(args, "20260101-000000-1-1") == {
        "earlier_land": [], "resume": [*resume_list, *([resume] if resume else [])]}
    # The branches a merge was passed to combine are kept beside them
    args.branch_list = ["fa", "fb"]
    assert cli._stop_result_dict(args, "20260101-000000-1-1")["pass_branch"] == [
        "fa", "fb"]
    args.resume_list, args.resume = [], ""
    assert "pass_branch" not in cli._stop_result_dict(args, "20260101-000000-1-1")


@pytest.mark.parametrize("remain_list", [[], ["codetwine-loom/sandbox/u1"]])
def test_a_stopped_command_whose_last_run_put_its_union_in_names_that_run_s_own_alone(
        monkeypatch, tmp_path, remain_list):
    """Once its last run had put its union into the working tree, or left a settled
    union still to go in, a stopped command names neither the branches it was passed
    nor the unions among them holding cases not done, as one that returned does."""
    asked = []
    monkeypatch.setattr(cli.sandbox_mod, "remain_branch_list",
                        lambda project_dir, out_dir, since_id: list(remain_list))
    monkeypatch.setattr(cli.sandbox_mod, "last_run_has_union",
                        lambda project_dir, branch_list: asked.append(branch_list) or True)
    monkeypatch.setattr(cli.sandbox_mod, "gap_union_list_of",
                        lambda *a: pytest.fail("no union is named ahead"))
    monkeypatch.setattr(cli.sandbox_mod, "land_list", lambda: [])
    args = argparse.Namespace(project=str(tmp_path), output_dir=str(tmp_path / "out"),
                              resume_list=["codetwine-loom/sandbox/u0"], resume="", request="r",
                              branch_list=["fa"])
    assert cli._stop_result_dict(args, "20260101-000000-1-1") == {
        "earlier_land": [], "resume": remain_list}
    assert asked == [remain_list]


def test_the_record_of_a_run_that_was_stopped_names_the_branches_it_had_left(
        monkeypatch, tmp_path):
    def stop(*a, **k):
        raise KeyboardInterrupt

    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/run/r/1")
    seen = []
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", stop)
    monkeypatch.setattr(cli.sandbox_mod, "remain_branch_list",
                        lambda project_dir, out_dir, since_id: seen.append(
                            (project_dir, out_dir)) or ["codetwine-loom/run/r/1",
                                                        "codetwine-loom/run/r/gone"])
    with pytest.raises(KeyboardInterrupt):
        cli.main(["run", "--request", "r", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert record_dict["resume"] == ["codetwine-loom/run/r/1"]
    assert record_dict["result"] == {"earlier_land": [],
                                     "resume": ["codetwine-loom/run/r/1",
                                                "codetwine-loom/run/r/gone"]}
    assert seen == [(repo, str(tmp_path / "out"))]


@pytest.mark.parametrize("remain_list, resume", [
    (["codetwine-loom/run/r/1"], ["codetwine-loom/sandbox/u0", "codetwine-loom/run/r/1"]),
    (["codetwine-loom/sandbox/u0", "codetwine-loom/run/r/1"],
     ["codetwine-loom/sandbox/u0", "codetwine-loom/run/r/1"]),
    ([], ["codetwine-loom/sandbox/u0", "codetwine-loom/run/old/1"])])
def test_the_record_of_a_run_that_was_stopped_names_the_union_it_was_passed_for_its_cases(
        monkeypatch, tmp_path, capsys, remain_list, resume):
    """The run was passed a union whose tests found a case not done and a task's
    branch. Stopped after its own runs left a branch, it names the union ahead of
    that branch, as a run that failed without being stopped does; stopped before
    they left any, it names what it was passed."""
    def stop(*a, **k):
        raise KeyboardInterrupt

    repo = _git_repo(tmp_path / "repo")
    for branch in ("codetwine-loom/sandbox/u0", "codetwine-loom/run/old/1",
                   "codetwine-loom/run/r/1"):
        _work_branch(repo, branch, "add a")
    asked = []
    monkeypatch.setattr(cli.sandbox_mod, "dispatch", stop)
    monkeypatch.setattr(cli.sandbox_mod, "remain_branch_list", lambda *a: remain_list)
    monkeypatch.setattr(cli.sandbox_mod, "gap_union_list_of",
                        lambda project_dir, branch_list, request: asked.append(
                            (project_dir, list(branch_list), request))
                        or ["codetwine-loom/sandbox/u0"])
    with pytest.raises(KeyboardInterrupt):
        cli.main(["run", "--request", "add a", "--resume", "codetwine-loom/sandbox/u0", "--resume",
                  "codetwine-loom/run/old/1", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert record_dict["resume"] == resume
    assert asked == ([(repo, ["codetwine-loom/sandbox/u0", "codetwine-loom/run/old/1"], "add a")]
                     if remain_list else [])
    assert capsys.readouterr().out.splitlines()[-1] == (
        "to carry on from what this run left: run "
        + " ".join(f"--resume {branch}" for branch in resume))


@pytest.mark.parametrize("name, argv", [
    ("dispatch", ["run", "--request", "r"]), ("run", ["sandbox", "--request", "r"]),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"])])
@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError("no engine"),
                                   ValueError("x")])
def test_a_command_stopped_after_it_put_a_change_in_says_so_in_its_record_and_its_lines(
        monkeypatch, tmp_path, capsys, name, argv, error):
    """The stop came before the command returned: what it had put into the working
    tree and the branch it had left are read off the process and the repository."""
    land_dict = {"files": 2, "paths": ["a.py", "t.py"], "patch": "p",
                 "last": "/out/land-last.diff"}

    def stop(*a, **k):
        monkeypatch.setattr(cli.sandbox_mod, "land_list", lambda: [land_dict])
        raise error

    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/sandbox/u1")
    monkeypatch.setattr(cli.sandbox_mod, name, stop)
    monkeypatch.setattr(cli.sandbox_mod, "remain_branch_list",
                        lambda project_dir, out_dir, since_id: ["codetwine-loom/sandbox/u1"])
    with pytest.raises((KeyboardInterrupt, SystemExit, ValueError)):
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    assert record_dict["files"] == ["a.py", "t.py"]
    assert record_dict["resume"] == ["codetwine-loom/sandbox/u1"]
    line_list = capsys.readouterr().out.splitlines()
    # The subcommand that carries on is the one the history subcommand prints for the
    # same record
    resume_command = "integrate" if argv[0] == "integrate" else "run"
    assert line_list[-2:] == [
        f"to take it back out of the working tree: git -C {repo} apply --binary "
        f"--whitespace=nowarn --no-ignore-whitespace -R "
        f"/out/land-last.diff",
        f"to carry on from what this run left: {resume_command} --resume codetwine-loom/sandbox/u1"]
    assert line_list[-3].startswith("record: ")


@pytest.mark.parametrize("override, exit_status, resume", [
    ({}, 0, []), ({}, 1, ["feature-x"]),
    ({"refactor": {"verdict": "unsettled", "skipped": ""}}, 0, ["feature-x"])])
def test_the_record_of_a_merge_names_its_branch_only_while_something_is_left_on_it(
        monkeypatch, tmp_path, override, exit_status, resume):
    """A branch this tool did not name stays where it is when the merge got through,
    and is not what to carry on from."""
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "feature-x")
    args = argparse.Namespace(project=repo, output_dir=str(tmp_path / "out"),
                              command="integrate", request="r")
    result_dict = {"branch": "feature-x", "commit": "c" * 40, "exit_status": 0,
                   "land": {"files": 1, "paths": ["a.py"]}, **override}
    record_dict = cli.write_run_record(args, "20260101-000000-1-1", result_dict,
                                       exit_status)
    assert record_dict["resume"] == resume and record_dict["files"] == ["a.py"]


def test_the_records_beyond_the_newest_few_go_unless_they_can_be_carried_on_from(
        monkeypatch, tmp_path):
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/sandbox/old")
    directory = tmp_path / "out" / "agents" / ".repo" / "repo" / "record"
    cli.runrecord.write(str(directory), {"id": "20250101-000000-1-1",
                                         "resume": ["codetwine-loom/sandbox/old"]})
    cli.runrecord.write(str(directory), {"id": "20250102-000000-1-1",
                                         "resume": ["codetwine-loom/sandbox/gone"]})
    cli.runrecord.write(str(directory), {"id": "20250103-000000-1-1"})
    monkeypatch.setattr(config, "DISPATCH_KEEP_RUN", 2)
    monkeypatch.setattr(cli.sandbox_mod, "run", lambda *a, **k: {
        "exit_status": 0, "land": None, "land_why": ""})
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit):
        cli.main(["sandbox", "--request", "r", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    id_list = [record_dict["id"] for record_dict in _record_list(tmp_path)]
    assert id_list[:2] == ["20250101-000000-1-1", "20250103-000000-1-1"]
    assert len(id_list) == 3


def test_history_lists_the_records_with_what_to_pass_to_carry_on(capsys, tmp_path):
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/run/r/1", "line one line two")
    directory = tmp_path / "out" / "agents" / ".repo" / "repo" / "record"
    cli.runrecord.write(str(directory), {
        "id": "20260101-000000-1-1", "command": "run", "request": "line one\nline two",
        "exit_status": 1, "elapsed_sec": 372.0, "error": "",
        "resume": ["codetwine-loom/run/r/1", "codetwine-loom/run/r/gone"],
        "token": {"total": 12345}})
    cli.runrecord.write(str(directory), {
        "id": "20260102-000000-1-1", "command": "sandbox", "request": "r",
        "exit_status": 1, "elapsed_sec": 2, "error": "no engine", "resume": []})
    cli.runrecord.write(str(directory), {
        "id": "20260103-000000-1-1", "command": "sandbox", "exit_status": 1,
        "resume": ["codetwine-loom/run/r/1"]})
    cli.runrecord.write(str(directory), {
        "id": "20260104-000000-1-1", "command": "integrate", "exit_status": 1,
        "resume": ["codetwine-loom/run/r/1"]})
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"runs: {directory}"
    assert lines[1] == "  20260101-000000-1-1  run  exit 1  6m12s  12,345 token(s)"
    assert lines[2] == "    line one line two"
    assert lines[3] == "    to carry on from it: run --resume codetwine-loom/run/r/1"
    assert lines[4] == "  20260102-000000-1-1  sandbox  exit 1  2s  0 token(s)"
    assert lines[5:7] == ["    r", "    stopped: no engine"]
    # The branch a sandboxed run left is carried on from with run, a merge's with
    # integrate
    assert lines[8] == "    to carry on from it: run --resume codetwine-loom/run/r/1"
    assert lines[10] == "    to carry on from it: integrate --resume codetwine-loom/run/r/1"
    assert lines[11].startswith("history: ")


def test_history_passes_over_a_record_of_another_shape(capsys, tmp_path, caplog):
    repo = _git_repo(tmp_path / "repo")
    directory = tmp_path / "out" / "agents" / ".repo" / "repo" / "record"
    cli.runrecord.write(str(directory), {
        "id": "20260101-000000-1-1", "command": "run", "exit_status": 0})
    for n, field_dict in enumerate(({"token": [1]}, {"elapsed_sec": "1.0"},
                                    {"command": ["run"]}, {"resume": 5}), 2):
        (directory / f"2026010{n}-000000-1-1.json").write_text(
            json.dumps({"id": f"2026010{n}-000000-1-1", **field_dict}), encoding="utf-8")
    (directory / "20260109-000000-1-1.json").write_text("[" * 200000, encoding="utf-8")
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [f"runs: {directory}",
                         "  20260101-000000-1-1  run  exit 0  0s  0 token(s)"]
    assert lines[2].startswith("history: ")
    assert caplog.text.count("is not of the shape a record is written in") == 4


def test_a_command_ends_with_its_lines_beside_a_record_of_another_shape(
        monkeypatch, tmp_path, capsys):
    """The record that cannot be read is older than the newest DISPATCH_KEEP_RUN."""
    repo = _git_repo(tmp_path / "repo")
    directory = tmp_path / "out" / "agents" / ".repo" / "repo" / "record"
    directory.mkdir(parents=True)
    (directory / "20200101-000000-1-1.json").write_text(json.dumps(
        {"id": "20200101-000000-1-1", "command": "run", "resume": 5}), encoding="utf-8")
    (directory / "20200102-000000-1-1.json").write_text("[" * 200000, encoding="utf-8")
    monkeypatch.setattr(config, "DISPATCH_KEEP_RUN", 1)
    monkeypatch.setattr(cli.sandbox_mod, "run", lambda *a, **k: {
        "exit_status": 0, "land_why": "", "branch": None,
        "land": {"files": 1, "paths": ["a.py"], "patch": "p", "last": "l"}})
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit) as raised:
        cli.main(["sandbox", "--request", "r", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    assert raised.value.code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2].startswith("record: ") and lines[-1].startswith("to take it back")
    assert len(list(directory.iterdir())) == 3


@pytest.mark.parametrize("command, result_dict", [
    ("sandbox", {"branch": "codetwine-loom/sandbox/u", "commit": "c" * 40, "exit_status": 0,
                 "land": None, "land_refuse": True,
                 "land_why": "the change writes .venv/x.pth, which git ignores"}),
    ("integrate", {"branch": "codetwine-loom/sandbox/u", "commit": "c" * 40, "exit_status": 0,
                   "land": None, "land_refuse": True, "gap": [{"what": "case-1"}],
                   "land_why": "the change writes .venv/x.pth, which git ignores"}),
    ("run", {"exit_status": 1, "resume": [], "integration": {
        "branch": "codetwine-loom/sandbox/u", "land": None, "land_refuse": True,
        "land_why": "the change writes .venv/x.pth, which git ignores"}}),
])
def test_a_change_kept_out_for_its_paths_is_recorded_with_nothing_to_carry_on_from(
        tmp_path, capsys, command, result_dict):
    """The record names no branch to pass with --resume; history says what kept the
    change out and the branch it is on while that is there."""
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/sandbox/u")
    args = argparse.Namespace(project=repo, output_dir=str(tmp_path / "out"),
                              command=command, request="r")
    record_dict = cli.write_run_record(args, "20260101-000000-1-1", result_dict, 1)
    assert record_dict["resume"] == []
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[3] == ("    NOT applied to the working tree, and on codetwine-loom/sandbox/u: the "
                        "change writes .venv/x.pth, which git ignores")
    assert lines[4].startswith("history: ")
    subprocess.run(["git", "-C", repo, "branch", "-D", "codetwine-loom/sandbox/u"], check=True,
                   capture_output=True)
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    assert capsys.readouterr().out.splitlines()[3] == (
        "    NOT applied to the working tree: the change writes .venv/x.pth, which git "
        "ignores")


def test_a_record_names_the_unions_a_union_still_to_go_in_stands_for(tmp_path,
                                                                     monkeypatch):
    """The unions held to be deleted once a union goes in are in the record's result;
    with none held, the result has no such key."""
    repo = _git_repo(tmp_path / "repo")
    args = argparse.Namespace(project=repo, output_dir=str(tmp_path / "out"),
                              command="run", request="r")
    record_dict = cli.write_run_record(args, "20260101-000000-1-1", None, 143)
    assert cli.runrecord.STAND_FOR_KEY not in record_dict["result"]
    monkeypatch.setattr(cli.sandbox_mod, "stand_for_list",
                        lambda project_dir: ["codetwine-loom/sandbox/20260101-000000-5-1"])
    record_dict = cli.write_run_record(args, "20260101-000000-2-1", None, 143)
    assert record_dict["result"][cli.runrecord.STAND_FOR_KEY] == [
        "codetwine-loom/sandbox/20260101-000000-5-1"]
    assert [one_dict["result"].get(cli.runrecord.STAND_FOR_KEY) for one_dict in
            cli.runrecord.read_list(cli.sandbox_mod.record_dir(
                str(tmp_path / "out"), "repo"))] == [
        None, ["codetwine-loom/sandbox/20260101-000000-5-1"]]


# Branch names git takes that a shell splits, expands or runs when they are written
# into a command as they are
_SHELL_BRANCH_TUPLE = ("x$(touch${IFS}MARK)", "a&b", "a;touch${IFS}MARK", "it's",
                       "back`touch${IFS}MARK`tick", 'say"so')


def _run_in_shell(command: str, work_dir) -> list[str]:
    """Run a line as a shell reads it, with a stand-in that prints the words it was
    given in place of the first word, and return those words.

    Nothing the line names is run: the directory holds the stand-in alone, and what
    is in it afterwards is checked by the caller.
    """
    work_dir.mkdir(exist_ok=True)
    show = work_dir / "show"
    show.write_text('#!/bin/sh\nfor word in "$@"; do printf "%s\\0" "$word"; done\n',
                    encoding="utf-8")
    show.chmod(0o755)
    proc = subprocess.run(["sh", "-c", f"./show {command}"], cwd=work_dir,
                          capture_output=True, text=True, check=True)
    assert sorted(os.listdir(work_dir)) == ["show"]
    return proc.stdout.split("\0")[:-1]


@pytest.mark.parametrize("branch", _SHELL_BRANCH_TUPLE)
def test_the_flags_that_carry_on_from_a_branch_are_read_whole_by_a_shell(
        capsys, tmp_path, branch):
    cli.print_resume_line(str(tmp_path), {"command": "run",
                                          "resume": [branch, "codetwine-loom/run/r/1"]})
    line = capsys.readouterr().out.rstrip("\n")
    head = "to carry on from what this run left: "
    assert line.startswith(head)
    assert _run_in_shell(line[len(head):], tmp_path / "work") == [
        "run", "--resume", branch, "--resume", "codetwine-loom/run/r/1"]


@pytest.mark.parametrize("branch", _SHELL_BRANCH_TUPLE)
def test_the_line_that_puts_a_repository_back_is_read_whole_by_a_shell(
        capsys, tmp_path, branch):
    repo = str(tmp_path / "a repo; $(x)")
    cli.print_union_in_place(repo, {"base": "a" * 40, "was_on": branch})
    line = capsys.readouterr().out.splitlines()[-1]
    head = "  to put the repository back: "
    assert line.startswith(head)
    first, second = line[len(head):].split(" && ")
    assert _run_in_shell(first, tmp_path / "work") == ["git", "-C", repo, "checkout",
                                                       branch]
    assert _run_in_shell(second, tmp_path / "work2") == ["git", "-C", repo, "reset",
                                                         "--hard"]


@pytest.mark.parametrize("branch", _SHELL_BRANCH_TUPLE)
def test_history_names_the_branches_to_carry_on_from_as_a_shell_reads_them(
        capsys, tmp_path, branch):
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, branch)
    directory = tmp_path / "out" / "agents" / ".repo" / "repo" / "record"
    cli.runrecord.write(str(directory), {
        "id": "20260101-000000-1-1", "command": "integrate", "exit_status": 1,
        "resume": [branch]})
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    line = capsys.readouterr().out.splitlines()[2]
    head = "    to carry on from it: "
    assert line.startswith(head)
    assert _run_in_shell(line[len(head):], tmp_path / "work") == [
        "integrate", "--resume", branch]


def test_history_says_what_to_change_before_a_merge_head_s_tests_keep_unsettled_goes_on(
        tmp_path, capsys):
    """The union is still what the run is carried on from, once HEAD's run is changed."""
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/sandbox/u")
    args = argparse.Namespace(project=repo, output_dir=str(tmp_path / "out"),
                              command="run", request="r")
    why = ("the merge was not settled (exit status 1), so nothing was put into the "
           "working tree. What it left is on codetwine-loom/sandbox/u: - `t` stopped. No "
           "merge over this HEAD is settled while the tests run over HEAD stop before "
           "every test has run: let the test command run every test when one fails, or "
           "make the tests that fail over HEAD pass, then carry on from "
           "codetwine-loom/sandbox/u")
    record_dict = cli.write_run_record(args, "20260101-000000-1-1", {
        "exit_status": 1, "resume": ["codetwine-loom/sandbox/u"], "integration": {
            "branch": "codetwine-loom/sandbox/u", "commit": "c" * 40, "exit_status": 1,
            "land": None, "land_refuse": False, "land_why": why, "head_block": ["stop"]}}, 1)
    assert record_dict["resume"] == ["codetwine-loom/sandbox/u"]
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    lines = capsys.readouterr().out.splitlines()
    assert lines[3] == \
        f"    NOT applied to the working tree, and on codetwine-loom/sandbox/u: {why}"
    assert lines[4].endswith("run --resume codetwine-loom/sandbox/u")


@pytest.mark.parametrize("out_name", ["w two  sp", "a\ttab", "it's \"quoted\"; $(x)"])
def test_history_writes_the_command_of_a_change_kept_out_as_it_was_recorded(
        tmp_path, capsys, out_name):
    """The reason is folded onto one line; the command that follows it is not."""
    repo = _git_repo(tmp_path / "repo")
    patch_path = str(tmp_path / out_name / "land.diff")
    command = cli.command_line("git", "-C", repo, "apply", "--binary", patch_path)
    args = argparse.Namespace(project=repo, output_dir=str(tmp_path / "out"),
                              command="sandbox", request="r")
    cli.write_run_record(args, "20260101-000000-1-1", {
        "branch": None, "commit": "c" * 40, "exit_status": 0, "land": None,
        "land_refuse": True, "land_command": command,
        "land_why": f"the change writes a  b.env,\nso nothing is put in. The change is "
                    f"written to {patch_path}; read it, and apply it by hand with:"}, 1)
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    line = capsys.readouterr().out.splitlines()[3]
    head = "apply it by hand with: "
    assert line.startswith("    NOT applied to the working tree: the change writes a b.env, "
                           "so nothing is put in. The change is written to ")
    assert line.endswith(f"{head}{command}")
    assert _run_in_shell(line.split(head)[1], tmp_path / "work") == [
        "git", "-C", repo, "apply", "--binary", patch_path]


def test_a_command_stopped_by_an_error_that_ends_with_a_command_records_the_two_apart(
        tmp_path, capsys, monkeypatch):
    repo = _git_repo(tmp_path / "repo")
    command = cli.command_line("codetwine-loom", "build", "--project", "/a  b",
                               "--exclude", "x y")

    def stop(*a, **k):
        raise cli.sandbox_mod.CommandError("the analysis keeps out\nvendor. Make it "
                                           "again with:", command)

    monkeypatch.setattr(cli.sandbox_mod, "run", stop)
    with pytest.raises(SystemExit) as raised:
        cli.main(["sandbox", "--request", "r", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    assert str(raised.value) == (f"sandbox: the analysis keeps out\nvendor. Make it "
                                 f"again with: {command}")
    record_dict, = _record_list(tmp_path)
    assert record_dict["error"] == "the analysis keeps out\nvendor. Make it again with:"
    assert record_dict["error_command"] == command
    capsys.readouterr()
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    assert capsys.readouterr().out.splitlines()[3] == (
        f"    stopped: the analysis keeps out vendor. Make it again with: {command}")


def test_a_patch_that_was_not_taken_in_is_named_in_a_command_a_shell_reads_whole(
        tmp_path, capsys):
    patch_path = tmp_path / "an out's; $(x)" / "run.patch"
    patch_path.parent.mkdir()
    patch_path.write_text("diff\n", encoding="utf-8")
    repo = str(tmp_path / "a repo; $(x)")
    cli.print_sandbox(repo, {"base": "b" * 40, "patch": str(patch_path), "files": 1,
                             "branch": None})
    line = capsys.readouterr().out.splitlines()[-1]
    head = ("not taken in. Applying it by hand skips the checks a change put into the "
            "working tree goes through; read it, and apply it by hand with: ")
    assert line.startswith(head)
    assert _run_in_shell(line[len(head):], tmp_path / "work") == [
        "git", "-C", repo, "apply", "--binary", "--whitespace=nowarn", "--no-ignore-whitespace",
        str(patch_path)]


def test_a_base_that_names_no_commit_is_refused_with_the_command_that_lists_them(
        tmp_path):
    repo = _git_repo(tmp_path / "a repo; $(x)")
    with pytest.raises(SystemExit) as raised:
        cli.main(["test-create", "--project", repo, "--base", "nope",
                  "--output-dir", str(tmp_path / "out")])
    assert str(raised.value).endswith(
        f"the commits are listed by: git -C '{repo}' log")


def test_history_says_nothing_of_runs_when_there_is_no_record(capsys, tmp_path):
    cli.main(["history", "--project", str(tmp_path), "--output-dir",
              str(tmp_path / "out")])
    assert capsys.readouterr().out.startswith("history: ")


# == the request read from a file or the standard input =======================

@pytest.fixture
def _stdin_unread():
    """Have the standard input read again by the test, and by the tests after it."""
    cli._stdin_text.cache_clear()
    yield
    cli._stdin_text.cache_clear()


@pytest.mark.parametrize("command", ["search", "edit", "run", "answer", "sandbox",
                                     "integrate"])
def test_the_request_is_read_from_the_file_it_is_passed_in(tmp_path, command):
    path = tmp_path / "request.txt"
    path.write_text("﻿\n  設定を整理して\n二行目\n\n", encoding="utf-8")
    argv = [command, "--request-file", str(path)]
    if command == "integrate":
        argv += ["--branch", "a", "--branch", "b"]
    assert cli._parse_args(argv).request == "設定を整理して\n二行目"


def _stdin_of(data: bytes):
    """Return a stand-in for sys.stdin that holds these bytes."""
    import io
    return io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="surrogateescape")


def test_the_request_is_read_from_the_standard_input_once(monkeypatch, _stdin_unread):
    monkeypatch.setattr(sys, "stdin", _stdin_of("\ufeff名前を変えて\n".encode("utf-8")))
    argv = ["run", "--request-file", "-"]
    assert cli._parse_args(argv).request == "名前を変えて"
    # The arguments are read again after the first run's questions
    assert cli._parse_args(argv).request == "名前を変えて"


@pytest.mark.parametrize("argv, message", [
    (["--request", "a", "--request-file", "{path}"], "both hold the request"),
    (["--request-file", "{missing}"], "could not be read as UTF-8 text"),
    (["--request-file", "{blank}"], "holds no text"),
    (["--request-file", "{binary}"], "could not be read as UTF-8 text")])
def test_a_request_file_that_cannot_be_used_is_refused(tmp_path, capsys, argv, message):
    (tmp_path / "request.txt").write_text("r", encoding="utf-8")
    (tmp_path / "blank.txt").write_text(" \n", encoding="utf-8")
    (tmp_path / "binary.txt").write_bytes(b"\xff\xfe\x00")
    name_dict = {"path": tmp_path / "request.txt", "missing": tmp_path / "none.txt",
                 "blank": tmp_path / "blank.txt", "binary": tmp_path / "binary.txt"}
    with pytest.raises(SystemExit):
        cli._parse_args(["sandbox", *(arg.format(**name_dict) for arg in argv)])
    assert message in capsys.readouterr().err


def test_a_request_passed_neither_way_is_one_line(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["sandbox", "--project", str(tmp_path)])
    assert "--request-file" in str(e.value)


@pytest.mark.parametrize("command", ["sandbox", "run", "integrate"])
def test_a_request_too_long_for_a_container_s_command_line_is_refused(tmp_path, command):
    path = tmp_path / "request.txt"
    path.write_text("あ" * (cli._REQUEST_MAX_BYTE // 3 + 1), encoding="utf-8")
    argv = [command, "--request-file", str(path), "--project", str(tmp_path)]
    if command == "integrate":
        argv += ["--branch", "a", "--branch", "b"]
    with pytest.raises(SystemExit, match="on its command line, which takes at most"):
        cli.main(argv)


def test_one_argument_of_that_many_bytes_is_taken_by_a_command_line():
    proc = subprocess.run(["true", "--request=" + "x" * cli._REQUEST_MAX_BYTE],
                          check=False)
    assert proc.returncode == 0


@pytest.mark.parametrize("argv", [["edit"], ["run", "--here"]])
def test_a_long_request_is_passed_on_where_no_container_is_started(argv):
    args = cli._parse_args(argv + ["--request", "x" * (cli._REQUEST_MAX_BYTE + 1)])
    cli._refuse_long_request(args)


# == the formatting command ====================================================

def test_format_runs_the_command_over_the_repository_it_is_given(monkeypatch, tmp_path,
                                                                 capsys):
    seen = []

    def run_format():
        seen.append(config.PROJECT_DIR)
        return {"files": ["a.py", "b.py"], "exit_status": 2, "output": "bad syntax"}

    monkeypatch.setattr(cli.edit, "run_format", run_format)
    cli.main(["format", "--project", str(tmp_path)])
    assert seen == [str(tmp_path)]
    assert capsys.readouterr().out.splitlines() == [
        "format: 2 file(s), exit status 2", "bad syntax"]


@pytest.mark.parametrize("result, line", [
    (None, "format: not run (FORMAT_COMMAND is empty, or no file was changed)"),
    ({"files": ["a.py"], "exit_status": 0, "output": "1 file reformatted"},
     "format: 1 file(s), exit status 0"),
    ({"files": ["a.py"], "exit_status": None, "output": ""},
     "format: 1 file(s), did not run")])
def test_format_says_what_the_command_did_in_one_line(capsys, result, line):
    cli.print_format(result)
    assert capsys.readouterr().out.splitlines() == [line]


def test_format_needs_no_credentials(monkeypatch, tmp_path):
    _nothing_to_call_with(monkeypatch)
    monkeypatch.setattr(cli.edit, "run_format", lambda: None)
    cli.main(["format", "--project", str(tmp_path)])


# == the check at the end of a run on its own ==================================

def test_edit_hands_the_run_what_the_change_is_compared_against(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.history, "open_attempt", lambda: None)
    monkeypatch.setattr(cli, "create_agent", lambda: None)
    monkeypatch.setattr(cli.edit, "git_diff_stat", lambda: {
        "files": [], "untracked": [], "total_insertions": 0, "total_deletions": 0})
    monkeypatch.setattr(cli, "print_scope", lambda scope_dict: None)
    monkeypatch.setattr(cli, "check_start", lambda path, exclude_list: {"at": path})
    monkeypatch.setattr(cli, "run_edit", lambda rlm, request, start_dict: seen.update(
        start=start_dict) or _ran("renamed it", False, history.VERDICT_NO_PROOF))
    # A relative path is the one it was where the command started, not in the
    # directory the run moves to
    monkeypatch.chdir(tmp_path)
    for flag, start in ([], {"at": str(tmp_path / "out" / "k.sqlite")}), \
            (["--no-check"], None):
        args = cli._parse_args(["edit", "--project", str(tmp_path), "--request", "r",
                                *flag])
        with pytest.raises(SystemExit):
            cli._finish_edit(args, os.path.join("out", "k.sqlite"), [])
        assert seen["start"] == start


def test_what_the_final_check_found_is_printed_under_the_verdict(capsys):
    result = {**_ran("renamed it", False, history.VERDICT_FAIL),
              "final_check": ["- these calls resolved before you started and no longer "
                              "do: f in a.py"]}
    cli.print_result(result, {"files": [], "untracked": [], "total_insertions": 0,
                              "total_deletions": 0})
    lines = capsys.readouterr().out.splitlines()
    assert lines[2].startswith("verdict: failed")
    assert lines[3] == ("  - these calls resolved before you started and no longer do: "
                        "f in a.py")


@pytest.mark.parametrize("flag, check", [([], True), (["--check"], True),
                                         (["--no-check"], False)])
def test_sandbox_passes_on_whether_the_change_is_checked(monkeypatch, tmp_path, flag,
                                                         check):
    seen = {}

    def run(*a, **k):
        seen.update(k)
        return {"exit_status": 0, "land": None, "land_why": ""}

    monkeypatch.setattr(cli.sandbox_mod, "run", run)
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit):
        cli.main(["sandbox", "--request", "r", "--project", str(tmp_path), *flag])
    assert seen["check"] is check


# == the network a run's containers join ======================================

def test_a_network_that_is_not_internal_stops_the_run_before_anything_starts(monkeypatch,
                                                                            tmp_path):
    def refuse(engine_path):
        raise RuntimeError("podman does not report the network n (SANDBOX_NETWORK) as "
                           "internal")

    _check_put_back(monkeypatch)
    monkeypatch.setattr(cli.sandbox_mod, "pick_engine", lambda name="": "/usr/bin/podman")
    monkeypatch.setattr(cli, "_ready_image", lambda *a: None)
    monkeypatch.setattr(cli.sandbox_mod, "refuse_open_network", refuse)
    monkeypatch.setattr(cli.sandbox_mod, "run", lambda *a, **k: pytest.fail("it started"))
    with pytest.raises(SystemExit, match="sandbox: podman does not report the network"):
        cli.main(["sandbox", "--request", "r", "--project", str(tmp_path)])
    # warm's container is not put on that network, so it is not asked about
    cli._refuse_no_engine_or_image(cli._parse_args(["warm", "--project", str(tmp_path)]))
    with pytest.raises(SystemExit, match="integrate: podman does not report"):
        cli._refuse_no_engine_or_image(cli._parse_args(
            ["integrate", "--project", str(tmp_path), "--branch", "a", "--branch", "b"]))



@pytest.mark.parametrize("argv", [
    ["sandbox", "--request", "r"], ["warm"], ["run", "--request", "r"],
    ["integrate", "--branch", "a", "--branch", "b"]])
def test_a_ca_file_no_container_can_be_given_stops_the_command_before_anything_starts(
        monkeypatch, tmp_path, argv):
    """Before the engine is looked for: a refill asked for later would stop at it."""
    ca_dir = tmp_path / "ca:dir"
    ca_dir.mkdir()
    (ca_dir / "ca.pem").write_text(CERTIFICATE_PEM, encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_dir / "ca.pem"))
    _check_put_back(monkeypatch)
    monkeypatch.setattr(cli.sandbox_mod, "pick_engine",
                        lambda name="": pytest.fail("the engine was looked for"))
    args = cli._parse_args([*argv, "--project", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        cli._refuse_no_engine_or_image(args)
    assert str(error.value) == (
        f"{argv[0]}: SSL_CERT_FILE names {ca_dir / 'ca.pem'}, and the container that "
        f"fetches dependencies cannot be given a file whose path holds a ':'. Move the "
        f"file and set SSL_CERT_FILE to where it is")
    # A command that starts no container is not stopped
    cli._refuse_no_engine_or_image(cli._parse_args(
        ["run", "--here", "--request", "r", "--project", str(tmp_path)]))


# == the repository and the output directory are checked before anything is written ==

@pytest.fixture
def _place_check_on(monkeypatch):
    """Put main()'s check of the repository and the output directory back."""
    monkeypatch.setattr(cli, "_refuse_wrong_place", _REFUSE_WRONG_PLACE)


# Every subcommand but broker, with the arguments each cannot do without
_PLACE_ARGV_DICT = {
    "build": [], "overview": [], "search": ["--request", "r"], "edit": ["--request", "r"],
    "answer": ["--request", "r"], "run": ["--request", "r"],
    "run --here": ["--request", "r"], "sandbox": ["--request", "r"],
    "integrate": ["--branch", "a", "--branch", "b"],
    "integrate --here": ["--branch", "a", "--branch", "b"],
    "test-create": ["--base", "HEAD"], "refactor": ["--base", "HEAD"],
    "test-refactor": ["--base", "HEAD"], "warm": [], "run-tests": [], "history": [],
    "format": []}


def _place_argv(command: str, project: str, output_dir: str) -> list[str]:
    """Return the argument list of one subcommand over a repository and an output directory."""
    return [*command.split(), *_PLACE_ARGV_DICT[command], "--project", project,
            "--output-dir", output_dir]


@pytest.mark.parametrize("command", _PLACE_ARGV_DICT)
def test_a_repository_that_is_not_there_stops_every_subcommand_in_one_line(
        command, tmp_path, _place_check_on):
    (tmp_path / "a-file").write_text("", encoding="utf-8")
    for project in (tmp_path / "nowhere", tmp_path / "a-file"):
        with pytest.raises(SystemExit) as raised:
            cli.main(_place_argv(command, str(project), str(tmp_path / "out")))
        assert str(raised.value) == (f"{command.split()[0]}: no such target repository: "
                                     f"{project}")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("command", [name for name in _PLACE_ARGV_DICT if name != "format"])
@pytest.mark.parametrize("output_dir, message", [
    ("repo/out", "OUTPUT_DIR is inside PROJECT_DIR"), ("", "OUTPUT_DIR holds PROJECT_DIR")])
def test_an_output_directory_in_the_wrong_place_stops_before_anything_is_written(
        command, output_dir, message, tmp_path, _place_check_on):
    repo = _git_repo(tmp_path / "repo")
    before_list = sorted(str(path) for path in tmp_path.rglob("*"))
    with pytest.raises(SystemExit, match=f"{command.split()[0]}: {message}"):
        cli.main(_place_argv(command, repo, str(tmp_path / output_dir)))
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before_list


@pytest.mark.parametrize("command", ["run", "sandbox", "integrate", "warm"])
@pytest.mark.parametrize("cache_dir, message", [
    ("repo", "is inside PROJECT_DIR or holds it"),
    ("repo/.cache", "is inside PROJECT_DIR or holds it"),
    ("", "is inside PROJECT_DIR or holds it"),
    ("out", "is OUTPUT_DIR or holds it"),
    ("out/agents/.host", "where this tool keeps its own files"),
    ("out/agents/.repo/repo", "where this tool keeps its own files"),
    ("out/repo/sandbox", "where this tool keeps its own files"),
    ("out/repo/sandbox/cache", "where this tool keeps its own files")])
def test_a_cache_directory_in_the_wrong_place_stops_before_anything_is_written(
        command, cache_dir, message, tmp_path, monkeypatch, _place_check_on):
    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(config, "SANDBOX_CACHE_DIR", str(tmp_path / cache_dir))
    monkeypatch.setattr(cli, "_refuse_no_engine_or_image", lambda args: None)
    before_list = sorted(str(path) for path in tmp_path.rglob("*"))
    with pytest.raises(SystemExit, match=f"{command}: SANDBOX_CACHE_DIR .* {message}"):
        cli.main(_place_argv(command, repo, str(tmp_path / "out")))
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before_list


@pytest.mark.parametrize("cache_dir", ["shared", "out/agents/.repo/repo/cache",
                                       "out/shared"])
def test_a_cache_directory_of_its_own_and_the_one_it_has_unset_are_not_refused(
        cache_dir, tmp_path, monkeypatch):
    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(config, "SANDBOX_CACHE_DIR", str(tmp_path / cache_dir))
    cli.sandbox_mod.check_cache_place(repo, str(tmp_path / "out"))
    # A subcommand that starts no container reads no cache and is not held to it
    monkeypatch.setattr(config, "SANDBOX_CACHE_DIR", repo)
    _REFUSE_WRONG_PLACE(cli._parse_args(_place_argv("search", repo, str(tmp_path / "out"))))


@pytest.mark.parametrize("command", [name for name in _PLACE_ARGV_DICT if name != "format"])
def test_a_repository_named_as_the_directory_of_the_agents_stops_before_anything_is_written(
        command, tmp_path, _place_check_on):
    repo = _git_repo(tmp_path / "agents")
    before_list = sorted(str(path) for path in tmp_path.rglob("*"))
    with pytest.raises(SystemExit) as raised:
        cli.main(_place_argv(command, repo, str(tmp_path / "out")))
    assert str(raised.value).startswith(
        f"{command.split()[0]}: the repository's directory is named agents")
    assert "\n" not in str(raised.value)
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before_list


def test_format_reads_no_output_directory_and_takes_the_root_of_a_repository_alone(
        tmp_path, monkeypatch, _place_check_on):
    repo = _git_repo(tmp_path / "repo")
    (tmp_path / "repo" / "below").mkdir()
    monkeypatch.setattr(cli.edit, "run_format", lambda: None)
    cli.main(_place_argv("format", repo, str(tmp_path / "repo" / "out")))
    for project in (tmp_path / "repo" / "below", tmp_path):
        with pytest.raises(SystemExit, match="format: not the root of a git repository"):
            cli.main(_place_argv("format", str(project), str(tmp_path / "out")))


def test_history_makes_no_output_directory_and_the_others_make_theirs(
        tmp_path, capsys, _place_check_on):
    repo = _git_repo(tmp_path / "repo")
    cli.main(_place_argv("history", repo, str(tmp_path / "none")))
    assert not (tmp_path / "none").exists()
    cli._refuse_wrong_place(cli._parse_args(_place_argv("build", repo,
                                                        str(tmp_path / "made"))))
    assert (tmp_path / "made").is_dir()


@pytest.mark.parametrize("command", ["build", "search", "sandbox", "run", "warm",
                                     "run-tests"])
def test_an_output_directory_that_cannot_be_made_stops_in_one_line(command, tmp_path,
                                                                 _place_check_on):
    repo = _git_repo(tmp_path / "repo")
    (tmp_path / "a-file").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        cli.main(_place_argv(command, repo, str(tmp_path / "a-file")))
    assert str(raised.value).startswith(
        f"{command}: the output directory {tmp_path / 'a-file'} cannot be made")


def test_a_broker_given_a_repository_is_refused_an_output_directory_inside_it(
        tmp_path, monkeypatch, _place_check_on):
    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.broker_mod, "serve",
                        lambda *a, **k: pytest.fail("the broker was started"))
    monkeypatch.setattr(cli.turnlog, "open_log",
                        lambda *a: pytest.fail("a log was opened"))
    with pytest.raises(SystemExit, match="broker: OUTPUT_DIR is inside PROJECT_DIR"):
        cli.main(["broker", "--endpoint", "unix:///s", "--project", repo,
                  "--output-dir", str(tmp_path / "repo" / "out")])


def test_a_broker_given_a_repository_that_is_not_there_stops_before_it_logs(
        tmp_path, monkeypatch, _place_check_on):
    monkeypatch.setattr(cli.broker_mod, "serve",
                        lambda *a, **k: pytest.fail("the broker was started"))
    monkeypatch.setattr(cli.turnlog, "open_log",
                        lambda *a, **k: pytest.fail("a log was opened"))
    with pytest.raises(SystemExit, match="broker: no such target repository: "):
        cli.main(["broker", "--endpoint", "unix:///s", "--project",
                  str(tmp_path / "nope"), "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("agent", ["../../../esc", "..", ".", "a/b", ".host", "a b"])
def test_a_broker_is_refused_an_agent_that_names_no_directory_of_its_own(
        tmp_path, monkeypatch, capsys, agent):
    """The broker's log goes under the output directory, by the agent's name."""
    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.broker_mod, "serve",
                        lambda *a, **k: pytest.fail("the broker was started"))
    monkeypatch.setattr(cli.turnlog, "open_log",
                        lambda *a, **k: pytest.fail("a log was opened"))
    with pytest.raises(SystemExit) as raised:
        cli.main(["broker", "--endpoint", "unix:///s", "--project", repo,
                  "--output-dir", str(tmp_path / "out"), "--agent", agent])
    assert raised.value.code == 2
    assert "--agent names one directory under the output directory" in \
        capsys.readouterr().err


def test_a_broker_stopped_before_it_listens_ends_without_a_traceback(monkeypatch, caplog):
    """The process of the run it serves has ended by the time it starts."""
    def stop_at_once(run_id):
        raise KeyboardInterrupt("the run was stopped (SIGTERM)")

    monkeypatch.setattr(cli, "_end_with_run", stop_at_once)
    monkeypatch.setattr(cli.broker_mod, "serve",
                        lambda *a, **k: pytest.fail("the broker was started"))
    monkeypatch.setattr(cli.turnlog, "open_log", lambda **k: None)
    stopped = []
    monkeypatch.setattr(cli.sandbox_mod, "stop_running_container",
                        lambda why: stopped.append(True))
    with caplog.at_level(logging.INFO, logger="codetwine_loom.cli"):
        cli.main(["broker", "--endpoint", "unix:///s", "--project", ".",
                  "--output-dir", "o", "--run", "20260101-000000-1-1"])
    assert "the broker was stopped before it listened (the run was stopped (SIGTERM))" \
        in caplog.text
    assert stopped == [True]


def test_a_broker_given_a_repository_opens_its_log_with_the_filter_of_its_values(
        tmp_path, monkeypatch):
    repo = _git_repo(tmp_path / "repo")
    seen = {}
    monkeypatch.setattr(cli.broker_mod, "serve", lambda *a, **k: None)
    monkeypatch.setattr(cli.turnlog, "open_log", lambda **k: seen.update(k))
    cli.main(["broker", "--endpoint", "unix:///s", "--project", repo,
              "--output-dir", str(tmp_path / "out")])
    assert isinstance(seen["log_filter"], cli.broker_mod.OwnValueFilter)


def test_what_a_broker_writes_to_stderr_passes_the_filter_of_its_values(monkeypatch):
    seen = {}
    set_up_log = cli._set_up_log

    def keep_handler():
        seen["handler"] = set_up_log()
        return seen["handler"]

    monkeypatch.setattr(cli, "_set_up_log", keep_handler)
    monkeypatch.setattr(cli.broker_mod, "serve", lambda *a, **k: None)
    cli.main(["broker", "--endpoint", "unix:///s"])
    assert [type(log_filter) for log_filter in seen["handler"].filters
            if isinstance(log_filter, cli.broker_mod.OwnValueFilter)] \
        == [cli.broker_mod.OwnValueFilter]


# == --base is a commit of the repository ====================================

@pytest.mark.parametrize("command", ["test-create", "refactor", "test-refactor"])
def test_a_base_that_names_no_commit_stops_the_three_stages_alike(command, tmp_path):
    repo = _git_repo(tmp_path / "repo")
    argv = [command, "--project", repo, "--output-dir", str(tmp_path / "out"), "--base"]
    with pytest.raises(SystemExit) as raised:
        cli.main([*argv, "nosuch"])
    assert str(raised.value).startswith(
        f"{command}: --base nosuch names no commit in {repo}.")
    assert not (tmp_path / "out").exists()
    for base in ("HEAD", "main"):
        cli._refuse_no_base(cli._parse_args([*argv, base]))
    # A directory that is no repository is the start-up's to refuse
    cli._refuse_no_base(cli._parse_args([command, "--project", str(tmp_path),
                                         "--base", "nosuch"]))
    cli._refuse_no_base(cli._parse_args(["build", "--project", repo]))


def test_a_test_command_that_keeps_pytest_from_counting_its_tests_stops_the_start_up(
        monkeypatch, tmp_path, capsys):
    """A subcommand that may run the tests stops before anything else in one line; the
    ones that run none, and --help, are not stopped."""
    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "TEST_COMMAND", ["pytest", "-qq"])
    for command, *rest in (("run-tests",), ("test-create", "--base", "nosuch")):
        with pytest.raises(SystemExit) as raised:
            cli.main([command, "--project", repo, "--output-dir", str(tmp_path / "out"),
                      *rest])
        assert str(raised.value) == f"{command}: {cli.edit.quiet_test_error()}"
        assert "one -q at most" in str(raised.value)
    assert not (tmp_path / "out").exists()
    for command in cli._NO_TEST_RUN_TUPLE:
        cli._refuse_quiet_test_command(argparse.Namespace(command=command))
    monkeypatch.setattr(cli.edit, "run_format", lambda: None)
    cli.main(["format", "--project", repo])
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as raised:
        cli.main(["run-tests", "--help"])
    assert raised.value.code == 0
    capsys.readouterr()


def test_a_start_up_that_runs_no_test_is_not_stopped_by_a_quiet_test_command(
        monkeypatch):
    """answer, and every subcommand while RUN_TEST is off, pass the check of a test
    command that keeps pytest from counting its tests; with RUN_TEST on the others
    are stopped."""
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "TEST_COMMAND", ["pytest", "-qq"])
    monkeypatch.setattr(config, "RUN_TEST", True)
    assert "answer" in cli._NO_TEST_RUN_TUPLE
    cli._refuse_quiet_test_command(argparse.Namespace(command="answer"))
    for command in ("run", "edit", "run-tests", "warm", "test-create"):
        with pytest.raises(SystemExit) as raised:
            cli._refuse_quiet_test_command(argparse.Namespace(command=command))
        assert str(raised.value) == f"{command}: {cli.edit.quiet_test_error()}"
    monkeypatch.setattr(config, "RUN_TEST", False)
    for command in ("run", "edit", "run-tests", "warm", "test-create", "answer"):
        cli._refuse_quiet_test_command(argparse.Namespace(command=command))


# == what the start-up refuses is one line ===================================

def test_a_file_the_analysis_cannot_read_stops_the_run_in_one_line(tmp_path,
                                                                  monkeypatch):
    project = _past_the_start_up(tmp_path, monkeypatch)

    def ensure(*a, rebuild=False, **k):
        if rebuild:
            raise json.JSONDecodeError("Expecting value", "", 0)
        return "k.sqlite", project, "proj", True

    monkeypatch.setattr(cli, "ensure_knowledge", ensure)
    monkeypatch.setattr(cli, "analysis_match_tree", lambda *a: False)
    with pytest.raises(SystemExit, match="Expecting value"):
        cli.main(["search", "--project", project, "--output-dir", str(tmp_path / "out"),
                  "--request", "x"])


@pytest.mark.parametrize("error", [
    RuntimeError("/p holds 1 uncommitted path(s)"), RuntimeError("/p has no branch named b"),
    ValueError("Expecting value")])
def test_integrate_here_stops_in_one_line_when_the_branches_cannot_be_combined(
        _integrate_here, tmp_path, monkeypatch, error):
    def refuse(*a, **k):
        raise error

    monkeypatch.setattr(cli.integrate_mod, "prepare", refuse)
    with pytest.raises(SystemExit) as raised:
        cli.main(["integrate", "--here", "--project", str(tmp_path / "proj"),
                  "--output-dir", str(tmp_path / "out"), "--branch", "a", "--branch", "b"])
    assert str(raised.value) == f"integrate: {error}"


def test_run_here_stops_in_one_line_when_a_branch_to_carry_on_from_is_not_there(
        tmp_path, monkeypatch):
    project = _past_the_start_up(tmp_path, monkeypatch)

    def refuse(request, resume_list=None):
        raise RuntimeError(f"/p has no branch named {resume_list[0]}")

    monkeypatch.setattr(cli.dispatch_mod, "run_dispatch", refuse)
    with pytest.raises(SystemExit) as raised:
        cli.main(["run", "--here", "--project", project, "--output-dir",
                  str(tmp_path / "out"), "--request", "r", "--resume", "gone"])
    assert str(raised.value) == "run: /p has no branch named gone"


# == integrate --here reads the starting state off an analysis of what is checked out ==

@pytest.mark.parametrize("is_match, rebuild_list", [(False, [True, True]),
                                                    (True, [False])])
def test_integrate_here_makes_an_analysis_of_another_tree_again_before_it_merges(
        _integrate_here, tmp_path, monkeypatch, is_match, rebuild_list):
    call_list = []
    monkeypatch.setattr(cli, "union_analysed", lambda *a: is_match)
    monkeypatch.setattr(cli, "ensure_knowledge", lambda *a, rebuild=False, **k: (
        call_list.append(("ensure", rebuild, k.get("with_doc"))) or
        ("k.sqlite", str(tmp_path / "proj"), "proj", True)))
    prepare = cli.integrate_mod.prepare
    monkeypatch.setattr(cli.integrate_mod, "prepare",
                        lambda *a, **k: call_list.append(("prepare",)) or prepare(*a, **k))
    _integrate_here(_settled(True))
    # The first call is the start-up's own, which uses what is there
    assert call_list[0] == ("ensure", False, False)
    before_list = call_list[1:call_list.index(("prepare",))]
    assert before_list == ([] if is_match else [("ensure", True, False)])
    assert [call[1] for call in call_list[1:] if call[0] == "ensure"] == rebuild_list


# == every stage run here works from a directory of its own ==================

def test_the_stages_run_here_work_from_a_directory_of_their_own(tmp_path, monkeypatch):
    """What each stage is handed by a relative path is read from where the command was
    started, and what it is to write there is written there."""
    project = _past_the_start_up(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.test_create_mod, "change_lang_ext", lambda *a: "")
    start = tmp_path / "start"
    start.mkdir()
    monkeypatch.chdir(start)
    (start / "answered.json").write_text('[{"n": 1, "request": "q", "answer": "a"}]',
                                         encoding="utf-8")
    (start / "target.json").write_text(
        '{"paths": ["a.py"], "test_paths": [], "baseline": []}', encoding="utf-8")
    seen_dict = {}

    def stage(name: str, result_dict: dict):
        def run(*a, **k):
            seen_dict[name] = (os.getcwd(), a, k)
            return result_dict
        return run

    create_dict = {"settled": True, "skipped": "", "work": {}, "left": [], "answer": "",
                   "outcome": None, "error": "", "tests_changed": [],
                   "non_test_changed": [], "tests": ""}
    monkeypatch.setattr(cli.answer_mod, "run_answer", stage("answer", _answered_result()))
    monkeypatch.setattr(cli.answer_mod, "run_combine", stage("combine", _answered_result()))
    monkeypatch.setattr(cli.dispatch_mod, "run_dispatch", stage("run", _split(True)))
    monkeypatch.setattr(cli.test_create_mod, "run_test_create",
                        stage("test-create", create_dict))
    monkeypatch.setattr(cli.refactor_mod, "run_refactor", stage("refactor", create_dict))
    monkeypatch.setattr(cli.integrate_mod, "run_integration",
                        stage("integrate", _settled(True)))
    for name in ("print_split", "print_answer", "print_test_create_stage",
                 "print_refactor_stage", "print_integration", "print_union_in_place",
                 "write_create_outcome", "write_refactor_outcome"):
        monkeypatch.setattr(cli, name, lambda *a, **k: None)
    monkeypatch.setattr(cli, "_refuse_no_base", lambda args: None)
    monkeypatch.setattr(cli.integrate_mod, "prepare", lambda *a, **k: {
        "merged": {"branches": []}, "was_on": "main"})
    monkeypatch.setattr(cli.integrate_mod, "branch_commit_list", lambda merged: [])
    monkeypatch.setattr(cli, "union_analysed", lambda *a: True)
    argv_dict = {
        "answer": ["answer", "--request", "q", "--answer-out", "answer.json"],
        "combine": ["answer", "--request", "q", "--answered", "answered.json"],
        "run": ["run", "--here", "--request", "r", "--tasks-out", "tasks.json"],
        "test-create": ["test-create", "--base", "HEAD", "--target-in", "target.json"],
        "refactor": ["refactor", "--base", "HEAD"],
        "integrate": ["integrate", "--here", "--branch", "a", "--branch", "b",
                      "--outcome-out", "outcome.json"]}
    for name, argv in argv_dict.items():
        with pytest.raises(SystemExit) as raised:
            cli.main([*argv, "--project", project, "--output-dir", "out"])
        assert raised.value.code == 0, name
        assert os.path.basename(seen_dict[name][0]).startswith("ctloom-run-"), name
        assert os.getcwd() == str(start)
    assert seen_dict["combine"][1][1] == [{"n": 1, "request": "q", "answer": "a"}]
    assert seen_dict["test-create"][2]["task_dict"]["paths"] == ["a.py"]
    # A relative path a stage is handed is the one it was where the command started
    abs_path, abs_out = str(start / "k.sqlite"), str(start / "out")
    assert seen_dict["test-create"][1][1:4] == (abs_path, "proj", abs_out)
    assert seen_dict["refactor"][1][2:5] == (abs_path, "proj", abs_out)
    assert seen_dict["integrate"][1][1:4] == (abs_path, "proj", abs_out)
    assert sorted(path.name for path in start.iterdir()) == [
        "answer.json", "answered.json", "outcome.json", "target.json", "tasks.json"]


# == the questions are asked at a terminal alone ==============================

@pytest.mark.parametrize("name", ["stdin", "stdout"])
def test_a_run_started_with_its_input_or_output_closed_starts_fresh(unfinished,
                                                                   monkeypatch, name):
    monkeypatch.setattr(progress, "output_is_terminal", _OUTPUT_IS_TERMINAL)
    monkeypatch.setattr(cli.sys, name, None)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("a terminal was asked"))
    assert cli._want_resume(unfinished, None) is False


def test_a_run_whose_output_is_no_terminal_starts_fresh_and_says_so(unfinished,
                                                                   monkeypatch, capsys):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(progress, "output_is_terminal", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("a terminal was asked"))
    assert cli._want_resume(unfinished, None) is False
    assert capsys.readouterr().out.splitlines()[-1].startswith(
        "no terminal to ask on, so this run starts fresh. Pass --resume")


def test_the_end_of_the_input_at_the_prompt_starts_fresh(unfinished, monkeypatch):
    def no_more(_prompt):
        raise EOFError

    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", no_more)
    assert cli._want_resume(unfinished, None) is False


# == the usage document's table of arguments ===================================

_USAGE_DOC = pathlib.Path(__file__).resolve().parent.parent / "docs" / "usage.md"

# The heading row of docs/usage.md's table of every subcommand's arguments
_ARGUMENT_TABLE_HEAD = "| Argument | Subcommands |"


def _usage_doc_flag_set() -> set[str]:
    """Return the flags the first column of docs/usage.md's tables of arguments names.

    The tables are those whose head starts with _ARGUMENT_TABLE_HEAD, each ending at the
    first line that is not a row.
    """
    line_list = _USAGE_DOC.read_text(encoding="utf-8").splitlines()
    flag_set = set()
    head_list = [index for index, line in enumerate(line_list)
                 if line.startswith(_ARGUMENT_TABLE_HEAD)]
    for head in head_list:
        position = head + 2
        while position < len(line_list) and line_list[position].startswith("|"):
            flag_set.update(re.findall(r"`(--[a-z-]+)", line_list[position].split("|")[1]))
            position += 1
    return flag_set


def test_the_usage_doc_s_table_of_arguments_names_the_flags_the_parser_takes(monkeypatch):
    """A flag added to a subcommand, or taken away from one, is added to the table or
    taken out of it."""
    seen = {}
    flag_set_of = cli._flag_set
    monkeypatch.setattr(cli, "_flag_set",
                        lambda parser: seen.setdefault("flag_set", flag_set_of(parser)))
    cli._parse_args(["history", "--project", "p"])
    parser_flag_set = {flag for flag in seen["flag_set"] if flag.startswith("--")}
    assert parser_flag_set - {"--help"} == _usage_doc_flag_set()


# == the request as it is passed ===============================================

@pytest.mark.parametrize("text", ["--versionフラグを足して", "-vを足して", "--", "-"])
def test_a_request_that_starts_with_a_dash_is_read_as_the_request(text):
    args = cli._parse_args(["search", "--project", "p", "--request", text])
    assert (args.request, args.project) == (text, "p")
    args = cli._parse_args(["edit", "--request", text, "--dry-run", "--project", "p"])
    assert (args.request, args.dry_run) == (text, True)


def test_a_request_file_whose_name_starts_with_a_dash_is_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "-request.txt").write_text("名前を変えて\n", encoding="utf-8")
    assert cli._parse_args(["run", "--request-file", "-request.txt"]).request \
        == "名前を変えて"


def test_the_two_request_flags_are_joined_with_what_follows_them():
    flag_set = {"--project", "--dry-run", "--request", "--request-file"}
    assert cli._join_request_flag(["edit", "--request", "-v", "--request-file", "-"],
                                  flag_set) \
        == ["edit", "--request=-v", "--request-file=-"]
    assert cli._join_request_flag(["edit", "--request=a", "--project", "--request"],
                                  flag_set) \
        == ["edit", "--request=a", "--project", "--request"]


def test_a_request_flag_followed_by_another_flag_is_not_joined_with_it():
    flag_set = {"--project", "--dry-run", "--request", "--request-file"}
    assert cli._join_request_flag(["edit", "--request", "--dry-run"], flag_set) \
        == ["edit", "--request", "--dry-run"]
    assert cli._join_request_flag(["edit", "--request", "--project=p"], flag_set) \
        == ["edit", "--request", "--project=p"]


@pytest.mark.parametrize("text", ["--output-dir=x に書くようにして",
                                  "--project=q を既定にして", "--guide=a b.md",
                                  "--dry-run\u3000を既定にして",
                                  "--exclude=dist\nも外して"])
def test_a_text_that_starts_with_a_flag_and_holds_a_blank_is_read_as_the_request(text):
    args = cli._parse_args(["sandbox", "--project", "p", "--request", text])
    assert (args.request, args.project, args.guide) == (text.strip(), "p",
                                                        config.EDIT_GUIDE)
    assert cli._is_flag_word(text, {"--output-dir", "--project", "--guide", "--dry-run",
                                    "--exclude"}) is False


@pytest.mark.parametrize("word", ["--dry-run", "--output-dir=x", "--project=", "--guide=a"])
def test_one_word_that_is_a_flag_alone_or_with_a_value_is_read_as_the_flag(word):
    flag_set = {"--output-dir", "--project", "--guide", "--dry-run"}
    assert cli._is_flag_word(word, flag_set) is True
    assert cli._join_request_flag(["edit", "--request", word], flag_set) \
        == ["edit", "--request", word]


def test_a_request_flag_with_no_text_before_the_next_flag_stops(capsys):
    with pytest.raises(SystemExit):
        cli._parse_args(["edit", "--request", "--dry-run", "--project", "p"])
    assert "expected one argument" in capsys.readouterr().err


def test_a_standard_input_that_is_closed_holds_no_text(monkeypatch, capsys, _stdin_unread):
    monkeypatch.setattr(sys, "stdin", None)
    with pytest.raises(SystemExit):
        cli._parse_args(["search", "--request-file", "-"])
    assert "--request-file: - holds no text" in capsys.readouterr().err


def test_a_standard_input_that_is_not_utf8_is_refused_as_a_file_is(monkeypatch, capsys,
                                                                  _stdin_unread):
    monkeypatch.setattr(sys, "stdin", _stdin_of("依頼".encode("cp932")))
    with pytest.raises(SystemExit):
        cli._parse_args(["search", "--request-file", "-"])
    assert "--request-file: - could not be read as UTF-8 text" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["search", "edit", "run", "answer", "sandbox",
                                     "integrate"])
@pytest.mark.parametrize("text, message", [
    ("add mul\0 please", "the request holds a NUL character"),
    ("a\udc90\udcddb", "the request is not UTF-8 text")])
def test_a_request_that_cannot_be_passed_on_as_text_is_refused(command, text, message,
                                                              monkeypatch):
    monkeypatch.setattr(cli, "_check_credential",
                        lambda *a, **k: pytest.fail("the run went on"))
    argv = [command, "--project", "p", f"--request={text}"]
    if command == "integrate":
        argv += ["--branch", "a", "--branch", "b"]
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)
    assert str(raised.value).startswith(f"{command}: {message}")


# == what a stopped command and a request read off a branch put on the terminal ==

@pytest.mark.parametrize("error", [KeyboardInterrupt(), RuntimeError("no engine"),
                                   ValueError("boom")])
def test_a_stopped_command_prints_its_token_counts_and_where_its_record_is(
        monkeypatch, tmp_path, capsys, error):
    def stop(*a, **k):
        cli.tokencount.add({"input_token": 70, "output_token": 5})
        raise error

    repo = _git_repo(tmp_path / "repo")
    monkeypatch.setattr(cli.sandbox_mod, "run", stop)
    with pytest.raises((KeyboardInterrupt, SystemExit, ValueError)):
        cli.main(["sandbox", "--request", "r", "--project", repo, "--output-dir",
                  str(tmp_path / "out")])
    record_dict, = _record_list(tmp_path)
    line_list = capsys.readouterr().out.splitlines()
    assert line_list[-2].startswith("tokens: 75 in 1 call(s)")
    assert line_list[-1] == f"record: {record_dict['path']}"


def test_the_request_read_off_a_branch_is_said_on_the_terminal(monkeypatch, caplog):
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda project_dir, name_list: "rename it")
    with caplog.at_level("INFO"):
        assert cli.request_of_run("/p", "", ["codetwine-loom/run/r/1"]) == "rename it"
        assert cli.request_of_integrate("/p", "", ["a", "b"]) == "rename it"
    assert [(record.name, record.getMessage()) for record in caplog.records] == [
        ("codetwine_loom.progress",
         "no request was passed, so the one codetwine-loom/run/r/1 carries is used: rename it"),
        ("codetwine_loom.progress",
         "no request was passed, so the one a, b carries is used: rename it")]


# == the working directory is no place modules are imported from ==============
def test_the_working_directory_is_taken_off_the_import_path(tmp_path, monkeypatch):
    import codetwine_loom
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", ["", str(other), ".", str(tmp_path),
                                      f"{tmp_path}/other/..", "/usr/lib/python3"])
    codetwine_loom.keep_cwd_off_path()
    assert sys.path == [str(other), "/usr/lib/python3"]


def test_the_directory_this_package_is_run_from_stays_on_the_import_path(monkeypatch):
    """Run from the directory the package is in, as "python -m" from a checkout is."""
    import codetwine_loom
    package_dir = os.path.dirname(os.path.dirname(codetwine_loom.__file__))
    monkeypatch.chdir(package_dir)
    monkeypatch.setattr(sys, "path", ["", package_dir, "/usr/lib/python3"])
    codetwine_loom.keep_cwd_off_path()
    assert sys.path == ["", package_dir, "/usr/lib/python3"]


@pytest.mark.parametrize("start", [
    # as "python -m codetwine_loom" starts, the working directory first on the path
    "import os, sys; sys.path.insert(0, os.getcwd())",
    # as the installed command starts, the working directory not on the path
    "import os, sys; sys.path[:] = [p for p in sys.path if p]"])
def test_a_module_in_the_working_directory_is_not_imported_by_a_command(tmp_path, start):
    """The names are ones the libraries look for and this installation does not have,
    and one looked for after the start-up."""
    import codetwine_loom
    target = pathlib.Path(git_repo(tmp_path / "target"))
    for name in ("rich", "opik", "brotli", "newrelic", "late_probe"):
        (target / f"{name}.py").write_text(
            f"open({str(tmp_path / ('ran_' + name))!r}, 'w').close()\n", encoding="utf-8")
    code = (f"{start}\n"
            "sys.argv = ['codetwine-loom', 'history', '--project', '.',\n"
            f"            '--output-dir', {str(tmp_path / 'out')!r}]\n"
            "from codetwine_loom.__main__ import main\n"
            "main()\n"
            "import importlib.util\n"
            "print('late:', importlib.util.find_spec('late_probe'))\n")
    env_dict = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "PYTHONPATH": os.path.dirname(os.path.dirname(codetwine_loom.__file__))}
    proc = subprocess.run([sys.executable, "-c", code], cwd=target, env=env_dict,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "no attempt recorded yet" in proc.stdout and "late: None" in proc.stdout
    assert sorted(path.name for path in tmp_path.glob("ran_*")) == []


def _module_at(name, file="", path_list=None):
    """Return a module object that says it was loaded from a file or a directory."""
    import types
    module = types.ModuleType(name)
    if file:
        module.__file__ = file
    if path_list is not None:
        module.__path__ = path_list
    return module


def test_the_modules_imported_from_the_working_directory_are_named(tmp_path, monkeypatch):
    import codetwine_loom
    monkeypatch.chdir(tmp_path)
    for name, file, path_list in (
            ("single", f"{tmp_path}/single.py", None),
            ("pack", f"{tmp_path}/pack/__init__.py", [f"{tmp_path}/pack"]),
            ("pack.part", f"{tmp_path}/pack/part.py", None),
            ("spread", "", ["/usr/lib/python3/spread", f"{tmp_path}/spread"]),
            ("below", f"{tmp_path}/lib/below.py", None),
            ("below_pack", f"{tmp_path}/lib/below_pack/__init__.py",
             [f"{tmp_path}/lib/below_pack"]),
            ("elsewhere", "/usr/lib/python3/elsewhere.py", None),
            ("built_in", "", None)):
        monkeypatch.setitem(sys.modules, name, _module_at(name, file, path_list))
    # The program the interpreter was started on is no module it imported
    monkeypatch.setitem(sys.modules, "__main__", _module_at("__main__",
                                                            f"{tmp_path}/start.py"))
    assert codetwine_loom.cwd_module_list() == ["pack", "single", "spread"]


def test_no_module_is_named_where_the_package_itself_is_run_from(monkeypatch):
    """Run from the directory the package is in, as "python -m" from a checkout is."""
    import codetwine_loom
    package_dir = os.path.dirname(os.path.dirname(codetwine_loom.__file__))
    monkeypatch.chdir(package_dir)
    monkeypatch.setitem(sys.modules, "single", _module_at("single",
                                                          f"{package_dir}/single.py"))
    assert codetwine_loom.cwd_module_list() == []


def test_a_command_stops_when_a_module_came_from_the_working_directory(tmp_path):
    import codetwine_loom
    target = pathlib.Path(git_repo(tmp_path / "target"))
    (target / "shadow_probe.py").write_text("", encoding="utf-8")
    code = ("import os, sys; sys.path.insert(0, os.getcwd())\n"
            "import shadow_probe\n"
            "sys.argv = ['codetwine-loom', 'history', '--project', '.']\n"
            "from codetwine_loom.__main__ import main\n"
            "main()\n")
    env_dict = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "PYTHONPATH": os.path.dirname(os.path.dirname(codetwine_loom.__file__))}
    proc = subprocess.run([sys.executable, "-c", code], cwd=target, env=env_dict,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1 and proc.stdout == ""
    assert proc.stderr.strip().splitlines() == [proc.stderr.strip()]
    assert proc.stderr.startswith("not started: this process imported shadow_probe from "
                                  f"the working directory, {target}, ")


@pytest.mark.parametrize("command", ["setup", "history", "--help"])
def test_a_command_started_from_a_removed_working_directory_stops_with_one_line(
        tmp_path, command):
    import codetwine_loom
    code = ("import os, sys\n"
            "from codetwine_loom.__main__ import main\n"
            "os.mkdir(sys.argv[1]); os.chdir(sys.argv[1]); os.rmdir(sys.argv[1])\n"
            f"sys.argv = ['codetwine-loom', {command!r}]\n"
            "main()\n")
    env_dict = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "PYTHONPATH": os.path.dirname(os.path.dirname(codetwine_loom.__file__))}
    proc = subprocess.run([sys.executable, "-c", code, str(tmp_path / "gone")],
                          cwd=tmp_path, env=env_dict, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1 and proc.stdout == ""
    assert proc.stderr.splitlines() == [
        "not started: the working directory cannot be read (No such file or directory). "
        "Start it from a directory that is there"]



@pytest.mark.parametrize("command", ["history", "edit", "sandbox", "build"])
def test_a_setting_with_a_percent_sign_in_it_is_shown_in_the_help_as_it_is(
        monkeypatch, capsys, command):
    monkeypatch.setattr(config, "OUTPUT_DIR", "./out 100%")
    monkeypatch.setattr(config, "EDIT_GUIDE", "100%_guide%(prog)s.md")
    with pytest.raises(SystemExit) as stop:
        cli._parse_args([command, "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert stop.value.code == 0 and "itself ./out 100%)" in out
    if command in ("edit", "sandbox"):
        assert "itself 100%_guide%(prog)s.md)" in out
    assert cli._help_value("img%d:1 50%") == "img%%d:1 50%%"


def test_an_image_name_with_a_percent_sign_in_it_does_not_stop_the_help(tmp_path):
    """The image's help is put together when the command line is imported."""
    proc = subprocess.run(
        config.own_argv(["sandbox", "--help"]), cwd=tmp_path, capture_output=True,
        text=True, env=config.own_child_env({"CONTAINER_IMAGE": "img%(prog)s:50%",
                                             "CONTAINER_ENGINE": "/nonexistent/engine"}))
    assert proc.returncode == 0, proc.stderr
    assert "and to codetwine-loom, which holds the toolchains of every language" in " ".join(
        proc.stdout.split())


def test_a_file_handed_to_a_subcommand_nested_too_deep_to_read_is_one_line(tmp_path):
    path = tmp_path / "deep.json"
    path.write_text("[" * 100_000 + "]" * 100_000)
    with pytest.raises(SystemExit, match="cannot be read"):
        cli.read_answer_list(str(path))
    with pytest.raises(SystemExit, match="cannot be read"):
        cli.read_test_task(str(path))
    assert cli.recheck_gap(str(path)) == {}


def _file_dict(repo) -> dict:
    """Return {path from repo: the bytes of the file}, without what .git holds."""
    return {str(path.relative_to(repo)): path.read_bytes()
            for path in sorted(pathlib.Path(repo).rglob("*"))
            if path.is_file() and ".git" not in path.relative_to(repo).parts}


@pytest.mark.parametrize("whitespace", ["", "fix", "error"])
def test_the_commands_printed_for_a_diff_do_what_this_tool_s_own_apply_does(
        tmp_path, capsys, whitespace):
    """The lines that take a change back out and that apply one by hand are pasted
    in another repository whose apply.whitespace is set: the target repository ends
    as this tool's own apply leaves it, and the other one is not touched."""
    repo_dict = {}
    for name in ("repo", "other"):
        repo = pathlib.Path(_git_repo(tmp_path / name))
        (repo / "doc.md").write_text("title  \nline one  \nkeep\t\n", encoding="utf-8")
        for argv in (["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t",
                                     "commit", "-qm", "two"],
                     *([["config", "apply.whitespace", whitespace]] if whitespace else [])):
            subprocess.run(["git", "-C", str(repo), *argv], check=True)
        repo_dict[name] = repo
    repo, other = repo_dict["repo"], repo_dict["other"]
    (other / "doc.md").write_text("title  \nmine  \nkeep\t\n", encoding="utf-8")
    other_before = _file_dict(other)
    base = gitcmd.head_commit(str(repo))
    (repo / "doc.md").write_text("title  \nline 1  \nkeep\t\n", encoding="utf-8")
    (repo / "new.md").write_text("added  \nplain\n", encoding="utf-8")
    for argv in (["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit",
                                 "-qm", "change"], ["branch", "change"],
                 ["reset", "-q", "--hard", base]):
        subprocess.run(["git", "-C", str(repo), *argv], check=True)
    before = _file_dict(repo)
    sandbox_mod.reset_land_list()
    land_dict = sandbox_mod.try_land(str(repo), base, "change", str(tmp_path / "land.diff"),
                                     str(tmp_path / "land-last.diff"))
    after = _file_dict(repo)
    assert land_dict["land"] and after != before
    capsys.readouterr()
    # Taken back out
    cli.print_take_out_line(str(repo), {"land": land_dict["land"]})
    command = capsys.readouterr().out.split(": ", 1)[1].strip()
    assert subprocess.run(["sh", "-c", command], cwd=other).returncode == 0
    assert _file_dict(repo) == before and _file_dict(other) == other_before
    # Applied by hand, as a change that was not taken in is
    cli.print_sandbox(str(repo), {"base": base, "patch": str(tmp_path / "land.diff"),
                                  "files": 2, "branch": None})
    command = capsys.readouterr().out.splitlines()[-1].split("with: ", 1)[1]
    assert subprocess.run(["sh", "-c", command], cwd=other).returncode == 0
    assert _file_dict(repo) == after and _file_dict(other) == other_before


def test_the_line_that_puts_a_repository_back_works_on_it_from_another_repository(
        tmp_path, capsys):
    repo, other = (pathlib.Path(_git_repo(tmp_path / name)) for name in ("repo", "other"))
    (other / "mine.txt").write_text("not committed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(other), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "union"], check=True)
    (repo / "left.txt").write_text("the union\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    cli.print_union_in_place(str(repo), {"base": "a" * 40, "was_on": "main"})
    command = capsys.readouterr().out.splitlines()[-1].split("back: ", 1)[1]
    assert subprocess.run(["sh", "-c", command], cwd=other).returncode == 0
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "-b"],
                          capture_output=True, text=True).stdout == "## main\n"
    assert (other / "mine.txt").read_text(encoding="utf-8") == "not committed\n"


@pytest.mark.parametrize("command, shown", [
    ("run", "run"), ("sandbox", "run"), ("integrate", "integrate"), (None, "run"),
    (["run"], "run")])
def test_the_subcommand_that_carries_on_is_the_one_history_prints(
        tmp_path, command, shown):
    assert cli.resume_command_text(str(tmp_path), command, ["codetwine-loom/sandbox/u 1"]) == (
        f"{shown} --resume 'codetwine-loom/sandbox/u 1'")


@pytest.mark.parametrize("command, branch_list, gap_list, land_list, told, shown", [
    ("integrate", ["u"], ["u"], [], "add a", "run --resume u"),
    ("integrate", ["u"], ["u"], [], "", "run --resume u --request <request>"),
    ("run", ["u", "v"], ["u", "v"], [], "add a", "run --resume u --resume v"),
    ("integrate", ["u"], [], [], "", "integrate --resume u"),
    ("run", ["u"], [], ["u"], "add a", "integrate --resume u"),
    ("sandbox", ["u"], [], ["u"], "", "integrate --resume u"),
    ("run", ["u", "v"], ["u"], ["v"], "add a", "run --resume u --resume v"),
    ("run", ["u"], [], [], "", "run --resume u --request <request>"),
    ("sandbox", ["u"], [], [], "add a", "run --resume u"),
])
def test_the_subcommand_that_carries_on_follows_what_the_branches_hold_now(
        monkeypatch, tmp_path, command, branch_list, gap_list, land_list, told, shown):
    """A union the working tree holds whose tests found cases not done is carried on
    from with run, a settled union the working tree does not hold with integrate,
    whatever subcommand left the record. run is told to pass the request where the
    branches do not say what it is."""
    monkeypatch.setattr(cli.sandbox_mod, "is_gap_union",
                        lambda project_dir, branch: branch in gap_list)
    monkeypatch.setattr(cli.sandbox_mod, "is_union_to_land",
                        lambda project_dir, branch: branch in land_list)
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda project_dir, name_list: told)
    assert cli.resume_command_text(str(tmp_path), command, branch_list) == shown


@pytest.mark.parametrize("command, gap_list, land_list, told, own, side_list, shown", [
    # A request other than the one the branches were made for
    ("run", [], [], "add a and b", "only add a", None, "run --resume u --request 'only add a'"),
    ("run", [], [], "", "x y", None, "run --resume u --request 'x y'"),
    ("sandbox", [], [], "add a", "it's $HOME", None,
     """run --resume u --request 'it'"'"'s $HOME'"""),
    ("run", ["u"], [], "add a", "only\0 add\na", None, "run --resume u --request 'only add a'"),
    # What the branches hold now is not read over a command given a request of its own
    ("run", [], ["u"], "add a", "new", None, "run --resume u --request new"),
    ("integrate", ["u"], [], "add a", "new", None, "integrate --resume u --request new"),
    # The same request, as it was typed and as a commit holds it
    ("run", [], [], "add a and b", "add a\n and\tb ", None, "run --resume u"),
    ("run", [], ["u"], "add a", "add a", None, "integrate --resume u"),
    ("integrate", ["u"], [], "add a", "add a", None, "run --resume u"),
    ("run", [], [], "", "", None, "run --resume u --request <request>"),
    ("run", [], [], "", None, None, "run --resume u --request <request>"),
    ("run", [], [], "", ["add a"], None, "run --resume u --request <request>"),
    ("run", [], [], "add a", 5, None, "run --resume u"),
    # The branches a merge was passed besides the one it carries on from
    ("integrate", [], [], "", "put them together", ["fc", "gone", "f d"],
     "integrate --resume u --branch fc --branch 'f d' --request 'put them together'"),
    ("integrate", ["u"], [], "add a", "add a", ["fc"], "integrate --resume u --branch fc"),
    ("integrate", [], ["u"], "", "", ["fc"], "integrate --resume u --branch fc"),
    ("integrate", ["u"], [], "add a", "", ["gone"], "run --resume u"),
    ("integrate", [], [], "", "", "fc", "integrate --resume u"),
    ("run", [], [], "add a", "add a", ["fc"], "run --resume u"),
])
def test_a_command_given_a_request_or_branches_of_its_own_is_carried_on_from_as_given(
        monkeypatch, tmp_path, command, gap_list, land_list, told, own, side_list,
        shown):
    """A command stopped before it left a branch of its own names the branches it was
    passed, and what it was given besides them goes into the line with them."""
    monkeypatch.setattr(cli.sandbox_mod, "is_gap_union",
                        lambda project_dir, branch: branch in gap_list)
    monkeypatch.setattr(cli.sandbox_mod, "is_union_to_land",
                        lambda project_dir, branch: branch in land_list)
    monkeypatch.setattr(cli.integrate_mod, "request_in_branch_list",
                        lambda project_dir, name_list: told)
    monkeypatch.setattr(cli.sandbox_mod, "branch_tip",
                        lambda project_dir, branch: "" if branch == "gone" else "c")
    text = cli.resume_command_text(str(tmp_path), command, ["u"], own, side_list)
    assert text == shown
    if "<request>" not in text:
        word_list = _run_in_shell(text, tmp_path / "work")
        assert "\n" not in text and word_list[:3] == [shown.split()[0], "--resume", "u"]
        if isinstance(own, str) and "--request" in word_list:
            assert word_list[-1] == " ".join(own.replace("\0", "").split())


@pytest.mark.parametrize("argv, result_key, shown", [
    (["run", "--request", "only add a.txt", "--resume", "codetwine-loom/run/r/1"], None,
     "run --resume codetwine-loom/run/r/1 --request 'only add a.txt'"),
    (["run", "--request", "add a.txt and b.txt", "--resume", "codetwine-loom/run/r/1"], None,
     "run --resume codetwine-loom/run/r/1"),
    (["run", "--request", "only add a.txt", "--resume", "codetwine-loom/run/r/1"], "unclear",
     "run --resume codetwine-loom/run/r/1 --request 'only add a.txt'"),
    (["integrate", "--resume", "codetwine-loom/run/r/1", "--branch", "fc", "--branch", "gone",
      "--request", "put a b and c together"], None,
     "integrate --resume codetwine-loom/run/r/1 --branch fc --request 'put a b and c together'"),
    (["integrate", "--resume", "codetwine-loom/run/r/1", "--branch", "fc"], None,
     "integrate --resume codetwine-loom/run/r/1 --branch fc")])
def test_a_command_stopped_before_it_left_a_branch_is_carried_on_from_as_it_was_given(
        monkeypatch, tmp_path, capsys, argv, result_key, shown):
    """The line the command ends with and the line history prints for its record
    are the same, and neither drops the request or the branches the command was
    given."""
    def stop(*a, **k):
        if result_key:
            return {result_key: ["which file?"], "resume": ["codetwine-loom/run/r/1"], "swept": [],
                    "exit_status": 1}
        raise KeyboardInterrupt

    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/run/r/1", "add a.txt and b.txt")
    _work_branch(repo, "fc", "")
    for name in ("dispatch", "integrate_into_tree"):
        monkeypatch.setattr(cli.sandbox_mod, name, stop)
    monkeypatch.setattr(cli.report_mod, "report", lambda *a, **k: "")
    with pytest.raises((KeyboardInterrupt, SystemExit)):
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    end_list = [line for line in capsys.readouterr().out.splitlines()
                if line.startswith("to carry on from what this run left: ")]
    assert end_list == [f"to carry on from what this run left: {shown}"]
    record_dict, = _record_list(tmp_path)
    assert record_dict["resume"] == ["codetwine-loom/run/r/1"]
    assert record_dict["result"].get("pass_branch") == (
        ["fc", "gone"] if "gone" in argv else ["fc"] if "fc" in argv else None)
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    assert [line.strip() for line in capsys.readouterr().out.splitlines()
            if "to carry on from it" in line] == [f"to carry on from it: {shown}"]


@pytest.mark.parametrize("argv, shown", [
    (["integrate", "--resume", "codetwine-loom/run/r/1", "--branch", "fc", "--request",
      "put a b and c together"],
     "integrate --resume codetwine-loom/run/r/1 --branch fc --request 'put a b and c together'"),
    (["integrate", "--resume", "codetwine-loom/run/r/1"],
     "integrate --resume codetwine-loom/run/r/1"),
    (["integrate", "--branch", "fc", "--branch", "fd"], None)])
def test_a_command_that_returned_leaving_no_branch_is_carried_on_from_as_it_was_given(
        monkeypatch, tmp_path, capsys, argv, shown):
    """A merging run whose container ended before it wrote a patch left no union: the
    command is named as it was given, as it is when it was stopped."""
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/run/r/1", "add a.txt and b.txt")
    _work_branch(repo, "fc", "")
    _work_branch(repo, "fd", "")
    monkeypatch.setattr(cli.sandbox_mod, "integrate_into_tree", lambda *a, **k: {
        "exit_status": 137, "branch": None, "commit": None, "no_patch": True,
        "land": None, "land_why": "the merging run ended (exit status 137) and left no "
        "patch to take in", "land_command": "", "land_refuse": False})
    monkeypatch.setattr(cli.sandbox_mod, "remove_old_run_in", lambda *a, **k: [])
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit) as raised:
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    assert raised.value.code == 137
    end_list = [line for line in capsys.readouterr().out.splitlines()
                if line.startswith("to carry on from what this run left: ")]
    assert end_list == ([f"to carry on from what this run left: {shown}"] if shown
                        else [])
    record_dict, = _record_list(tmp_path)
    assert record_dict["resume"] == (["codetwine-loom/run/r/1"] if shown else [])
    assert record_dict["result"].get("pass_branch") == (
        ["fc"] if shown and "--branch" in shown else None)
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    assert [line.strip() for line in capsys.readouterr().out.splitlines()
            if "to carry on from it" in line] == (
        [f"to carry on from it: {shown}"] if shown else [])


def test_a_branch_whose_result_the_working_tree_holds_is_named_by_no_record_line(
        capsys, tmp_path):
    """The record names a branch that is still in the repository and whose change
    the working tree holds: history prints no line to carry on from it, the record is
    one the records beyond the newest few go with, and the branch is still named for
    a change that was kept out of the working tree."""
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/sandbox/u1")
    _work_branch(repo, "codetwine-loom/sandbox/u2")
    directory = tmp_path / "out" / "agents" / ".repo" / "repo" / "record"
    for day, branch in ((1, "codetwine-loom/sandbox/u1"), (2, "codetwine-loom/sandbox/u2")):
        cli.runrecord.write(str(directory), {
            "id": f"2026010{day}-000000-1-1", "command": "run", "request": "r",
            "exit_status": 130, "resume": [branch],
            "result": {"branch": branch, "land_refuse": True, "land_why": "why"}})
    assert cli.open_branch_list(repo, ["codetwine-loom/sandbox/u1",
                                       "codetwine-loom/sandbox/gone", "", None]) \
        == ["codetwine-loom/sandbox/u1"]
    subprocess.run(["git", "-C", repo, "checkout", "-q", "codetwine-loom/sandbox/u1", "--",
                    "work.txt"], check=True)
    assert cli.open_branch_list(repo, ["codetwine-loom/sandbox/u1"]) == []
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    line_list = capsys.readouterr().out.splitlines()
    assert [line for line in line_list if "to carry on from it" in line] == []
    assert sum("NOT applied to the working tree, and on codetwine-loom/sandbox/u" in line
               for line in line_list) == 2
    cli.runrecord.remove_old(str(directory), 1, lambda record_dict: bool(
        cli.open_branch_list(repo, record_dict.get("resume") or [])))
    assert [record_dict["id"] for record_dict in _record_list(tmp_path)] == [
        "20260102-000000-1-1"]


def _unsettled_union(repo, name, side_list):
    """Point a branch at a merge commit a merging run left unsettled, over HEAD, whose
    other parents are the tips of the branches named."""
    git_list = ["git", "-C", repo, "-c", "user.email=t@t", "-c", "user.name=t"]
    parent_list = [word for branch in ["HEAD", *side_list] for word in ("-p", branch)]
    commit = subprocess.run(
        [*git_list, "commit-tree", f"{side_list[0]}^{{tree}}", *parent_list, "-m",
         "integrate: r\n\nCodetwine-Loom-Verdict: unsettled\nCodetwine-Loom-Exit-Status: 1"],
        capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run([*git_list, "branch", "-f", name, commit], check=True)


@pytest.mark.parametrize("carry_on", ["the working tree holds it", "its branch is gone"])
def test_history_says_no_more_what_to_change_once_a_merge_head_s_tests_kept_unsettled_is_carried_on(
        tmp_path, capsys, carry_on):
    """The line that says what to change before a union is carried on from goes with
    the union being one a run is carried on from."""
    side, union = (f"codetwine-loom/sandbox/20260101-000000-{n}-1" for n in (4, 5))
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, side)
    _unsettled_union(repo, union, [side])
    args = argparse.Namespace(project=repo, output_dir=str(tmp_path / "out"),
                              command="run", request="r")
    why = (f"the merge was not settled (exit status 1). No merge over this HEAD is "
           f"settled while …, then carry on from {union}")
    cli.write_run_record(args, "20260101-000000-1-1", {
        "exit_status": 1, "resume": [union], "integration": {
            "branch": union, "commit": "c" * 40, "exit_status": 1, "land": None,
            "land_refuse": False, "land_why": why, "head_block": ["stop"]}}, 1)
    history_argv = ["history", "--project", repo, "--output-dir", str(tmp_path / "out")]
    cli.main(history_argv)
    run_text = capsys.readouterr().out.split("\nhistory:")[0]
    assert [line.strip() for line in run_text.splitlines()[3:]] == [
        f"NOT applied to the working tree, and on {union}: {why}",
        f"to carry on from it: run --resume {union}"]
    if carry_on == "the working tree holds it":
        subprocess.run(["git", "-C", repo, "checkout", "-q", side, "--", "work.txt"],
                       check=True)
    else:
        subprocess.run(["git", "-C", repo, "branch", "-q", "-D", union], check=True)
    cli.main(history_argv)
    assert capsys.readouterr().out.split("\nhistory:")[0].splitlines()[3:] == []


def test_an_unsettled_union_whose_merged_commits_the_working_tree_holds_is_named_by_no_line(
        capsys, monkeypatch, tmp_path):
    """A union left unsettled is named to carry on from until the working tree holds
    every commit it merged, at the end of the command that left it and in history.
    It is still open for what deletes a branch, and the branch is left where it is."""
    u1, n1, n2, n3, gone = (f"codetwine-loom/sandbox/20260101-000000-{n}-1" for n in range(4, 9))
    task = "codetwine-loom/run/20260101-000000-2-1/1"
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, u1)
    (pathlib.Path(repo) / "other.txt").write_text("other\n")
    subprocess.run(["git", "-C", repo, "add", "other.txt"], check=True)
    subprocess.run(["git", "-C", repo, "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "other"], check=True)
    subprocess.run(["git", "-C", repo, "branch", "-f", task, "HEAD"], check=True)
    subprocess.run(["git", "-C", repo, "reset", "-q", "--hard", "HEAD^"], check=True)
    _unsettled_union(repo, n1, [u1])
    _unsettled_union(repo, n2, [n1])
    _unsettled_union(repo, n3, [u1, task])
    _unsettled_union(repo, "mine/union", [u1])
    name_list = [n1, n2, n3, "mine/union"]
    assert cli.open_branch_list(repo, name_list) == name_list
    # The working tree takes what the first union holds
    subprocess.run(["git", "-C", repo, "checkout", "-q", u1, "--", "work.txt"], check=True)
    assert cli.open_branch_list(repo, name_list) == [n3, "mine/union"]
    assert [sandbox_mod.is_union_in_tree(repo, branch)
            for branch in [*name_list, u1, gone]] == [True, True, False, False, False, False]
    assert sandbox_mod.open_branch_list(repo, name_list) == name_list
    assert sandbox_mod._result_in_tree_list(repo, name_list) == []
    directory = tmp_path / "out" / "agents" / ".repo" / "repo" / "record"
    for day, branch in enumerate(name_list[:3], 1):
        cli.runrecord.write(str(directory), {
            "id": f"2026010{day}-000000-1-1", "command": "integrate", "request": "r",
            "exit_status": 1, "resume": [branch], "result": {"branch": branch}})
    cli.main(["history", "--project", repo, "--output-dir", str(tmp_path / "out")])
    assert [line.strip() for line in capsys.readouterr().out.splitlines()
            if "to carry on from it" in line] == [f"to carry on from it: integrate --resume {n3}"]
    # A command that leaves such a union ends with no line to carry on from it
    monkeypatch.setattr(cli.sandbox_mod, "integrate_into_tree", lambda *a, **k: {
        "exit_status": 1, "branch": n2, "commit": "c" * 40, "land": None,
        "land_why": "it did not get through", "land_command": "", "land_refuse": False})
    monkeypatch.setattr(cli.sandbox_mod, "remove_old_run_in", lambda *a, **k: [])
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit) as raised:
        cli.main(["integrate", "--resume", n1, "--project", repo,
                  "--output-dir", str(tmp_path / "out")])
    assert raised.value.code == 1
    assert "to carry on from what this run left" not in capsys.readouterr().out
    assert _record_list(tmp_path)[-1]["resume"] == []
    assert subprocess.run(["git", "-C", repo, "rev-parse", "--verify", "-q", n2],
                          capture_output=True).returncode == 0


@pytest.mark.parametrize("name, argv, shown", [
    ("run", ["sandbox", "--request", "r"], "run"),
    ("integrate_into_tree", ["integrate", "--branch", "a", "--branch", "b"], "integrate")])
def test_a_command_that_returned_ends_with_the_line_history_prints_for_its_record(
        monkeypatch, tmp_path, capsys, name, argv, shown):
    """A sandbox or integrate command that left a branch to carry on from names it
    with the subcommand that takes --resume, as the history subcommand does."""
    repo = _git_repo(tmp_path / "repo")
    _work_branch(repo, "codetwine-loom/sandbox/u1")
    monkeypatch.setattr(cli.sandbox_mod, name, lambda *a, **k: {
        "exit_status": 1, "branch": "codetwine-loom/sandbox/u1", "commit": "c" * 40, "land": None,
        "land_why": "it did not get through", "land_command": "", "land_refuse": False})
    monkeypatch.setattr(cli.sandbox_mod, "remove_old_run_in", lambda *a, **k: [])
    monkeypatch.setattr(cli, "print_sandbox", lambda project_dir, result: None)
    with pytest.raises(SystemExit) as raised:
        cli.main(argv + ["--project", repo, "--output-dir", str(tmp_path / "out")])
    assert raised.value.code == 1
    end_line = capsys.readouterr().out.splitlines()[-1]
    assert end_line == \
        f"to carry on from what this run left: {shown} --resume codetwine-loom/sandbox/u1"
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(source, "project_name", "repo")
    cli.print_run_record_list(repo)
    assert capsys.readouterr().out.splitlines()[-1] == (
        f"    to carry on from it: {shown} --resume codetwine-loom/sandbox/u1")


def test_a_command_set_by_hand_is_said_to_stay_when_the_language_changes(monkeypatch,
                                                                         caplog):
    monkeypatch.setattr(config, "PRIMARY_EXT", "cbl")
    monkeypatch.setenv("RUN_COMMAND", "cobc -x src/MAIN.cbl")
    monkeypatch.delenv("CHECK_COMMAND", raising=False)
    monkeypatch.setattr(cli.test_create_mod, "change_lang_ext", lambda *a: "java")
    monkeypatch.setattr(cli.scope_mod, "set_primary", lambda ext: None)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.cli"):
        cli._take_change_language("b" * 40, "k.sqlite", "repo")
    assert "RUN_COMMAND set by hand stays as it is with Java" in caplog.text


@pytest.mark.parametrize("change_ext, primary, expect_list", [
    ("java", "cbl", ["java"]), ("", "cbl", [])])
def test_the_stages_after_the_merge_take_the_language_of_the_change(
        monkeypatch, change_ext, primary, expect_list):
    seen = []
    monkeypatch.setattr(config, "PRIMARY_EXT", primary)
    monkeypatch.setattr(cli.test_create_mod, "change_lang_ext", lambda *a: change_ext)
    monkeypatch.setattr(cli.scope_mod, "set_primary", seen.append)
    assert cli._take_change_language("b" * 40, "k.sqlite", "repo") == "".join(expect_list)
    assert seen == expect_list


@pytest.mark.parametrize("command", ["refactor", "test-refactor"])
@pytest.mark.parametrize("lang", ["java", ""])
def test_the_refactoring_stages_take_the_language_of_the_change_first(monkeypatch,
                                                                     command, lang):
    seen = []
    monkeypatch.setattr(cli, "_take_change_language",
                        lambda *a: seen.append("lang") or lang)
    monkeypatch.setattr(cli.turnlog, "open_run", contextlib.nullcontext)

    def refactor(*a, **k):
        seen.append(("refactor", k["runnable_only"]))
        raise _PrintStop

    monkeypatch.setattr(cli.refactor_mod, "run_refactor", refactor)
    args = argparse.Namespace(base="b", command=command, output_dir="out", request="r",
                              outcome_out="")
    with pytest.raises(_PrintStop):
        cli._finish_refactor(args, "k", "p", [])
    assert seen == ["lang", ("refactor", bool(lang))]


@pytest.mark.parametrize("lang", ["java", ""])
def test_the_listing_hands_on_the_language_it_took(tmp_path, monkeypatch, capsys, lang):
    out = tmp_path / "targets.json"
    seen = []
    monkeypatch.setattr(cli, "_take_change_language", lambda *a: seen.append("lang") or lang)
    monkeypatch.setattr(cli.test_create_mod, "list_target", lambda *a: seen.append("list") or {
        "targets": {"a.java": ["f"]}, "tests": {}, "skipped": "", "baseline": []})
    args = argparse.Namespace(base="b", target_only=True, target_in="",
                              outcome_out=str(out))
    with pytest.raises(SystemExit) as raised:
        cli._finish_test_create(args, "k", "p", [])
    assert raised.value.code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["lang"] == lang
    assert seen == ["lang", "list"]


def test_a_run_writing_tests_on_its_own_takes_the_language_of_the_change(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "_take_change_language", lambda *a: seen.append(a) or "")
    monkeypatch.setattr(cli.turnlog, "open_run", contextlib.nullcontext)
    monkeypatch.setattr(cli.test_create_mod, "run_test_create",
                        lambda *a, **k: (_ for _ in ()).throw(_PrintStop))
    args = argparse.Namespace(base="b", target_only=False, target_in="",
                              outcome_out="", output_dir="out", request="r")
    with pytest.raises(_PrintStop):
        cli._finish_test_create(args, "k", "p", [])
    assert seen == [("b", "k", "p")]


def test_a_refactoring_container_handed_a_language_does_not_work_it_out_again(
        monkeypatch):
    seen = []
    monkeypatch.setattr(cli.sandbox_mod, "in_container", lambda: True)
    monkeypatch.setattr(cli.sandbox_mod, "primary_ext_of_env", lambda: "java")
    monkeypatch.setattr(cli, "_take_change_language",
                        lambda *a: pytest.fail("the change was read"))
    monkeypatch.setattr(cli.turnlog, "open_run", contextlib.nullcontext)

    def refactor(*a, **k):
        seen.append(k["runnable_only"])
        raise _PrintStop

    monkeypatch.setattr(cli.refactor_mod, "run_refactor", refactor)
    args = argparse.Namespace(base="b", command="refactor", output_dir="out", request="r",
                              outcome_out="")
    with pytest.raises(_PrintStop):
        cli._finish_refactor(args, "k", "p", [])
    assert seen == [True]


def test_a_test_task_leaves_the_language_to_the_listing(monkeypatch):
    monkeypatch.setattr(cli.sandbox_mod, "in_container", lambda: True)
    monkeypatch.setattr(cli, "_take_change_language",
                        lambda *a: pytest.fail("the change was read"))
    monkeypatch.setattr(cli.turnlog, "open_run", contextlib.nullcontext)
    monkeypatch.setattr(cli, "read_test_task", lambda path: {"paths": []})
    monkeypatch.setattr(cli.test_create_mod, "run_test_create",
                        lambda *a, **k: (_ for _ in ()).throw(_PrintStop))
    args = argparse.Namespace(base="b", target_only=False, target_in="t.json",
                              outcome_out="", output_dir="out", request="r")
    with pytest.raises(_PrintStop):
        cli._finish_test_create(args, "k", "p", [])


@pytest.mark.parametrize("given, inside, expect_list", [
    ("java", True, ["java"]), ("", True, []), ("java", False, [])])
def test_a_container_takes_the_language_it_is_handed(monkeypatch, given, inside,
                                                    expect_list):
    seen = []
    monkeypatch.setattr(cli.sandbox_mod, "in_container", lambda: inside)
    monkeypatch.setattr(cli.sandbox_mod, "primary_ext_of_env", lambda: given)
    monkeypatch.setattr(cli.scope_mod, "set_primary", seen.append)
    cli._take_host_language("refactor")
    cli._take_host_language("build")
    assert seen == expect_list
