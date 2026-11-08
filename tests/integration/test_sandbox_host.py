"""The subcommands that start a container, with the container's scripts run here.

The engine is the one under tests/integration/host_engine.py: it runs entrypoint.sh
and warm.sh on this machine with the container's paths written as this machine's. The
broker is the real one, started as a process of its own and reached over its socket;
the model behind it answers from the script (conftest).
"""

import importlib
import json
import logging
import os
import re

import pytest

from integration.conftest import (ADD_SUB_REPLY_LIST, branch_list, code, commit_all, complete,
                      complete_json, engine_run_list, git)
from codetwine_loom import cli, config, history, image, rlm

pytestmark = pytest.mark.integration

DIV_TEXT = '''    return a * b


def div(a: float, b: float) -> float:
    """Return a divided by b."""
    return a / b
'''

# One agent's run: add div() beside mul(), run the tests, and finish
ADD_DIV_REPLY_LIST = [
    code('print(replace_in_file("calc/scale.py", "    return a * b\\n", '
         + repr(DIV_TEXT) + "))"),
    code("print(run_test())"),
    complete("added div() to calc/scale.py; the tests passed"),
]

_SUB_TASK = "add sub to calc/basic.py"
_DIV_TASK = "add div to calc/scale.py"


def _main(*argv: str) -> int:
    """Run cli.main() and return the exit status, 0 when it returned."""
    try:
        cli.main(list(argv))
    except SystemExit as raised:
        if isinstance(raised.code, int):
            return raised.code
        pytest.fail(f"main stopped with: {raised.code}")
    return 0


def _sandbox(calc_repo, out_dir, *argv: str) -> int:
    return _main("sandbox", "--project", str(calc_repo), "--output-dir", str(out_dir),
                 "--request", "add a sub function beside add", *argv)


def _branch_of(out: str, label: str) -> str:
    """Read the branch a printed line names, e.g. "branch: codetwine-loom/sandbox/x (abc)"."""
    found = re.search(rf"^{label}: (\S+)", out, re.M)
    assert found, out
    return found.group(1)


def _commit_of(out: str, label: str) -> str:
    """Read the short commit a line names, e.g. "tests: codetwine-loom/sandbox/x (abc), settled"."""
    found = re.search(rf"^{label}: \S+ \(([0-9a-f]+)\)", out, re.M)
    assert found, out
    return found.group(1)


def _base_of(run: dict) -> str:
    """Return the commit one engine run was told to start its clone at."""
    return run["env"]["CTLOOM_BASE"]


def _agent_out(out_dir, name_part: str):
    """Return the one agent output directory whose name holds name_part."""
    match_list = [p for p in (out_dir / "agents").iterdir()
                  if p.is_dir() and name_part in p.name and not p.name.startswith(".")]
    assert len(match_list) == 1, match_list
    return match_list[0]


# == sandbox ====================================================
def test_sandbox_makes_the_change_in_a_clone_and_puts_it_into_the_working_tree(
        calc_repo, out_dir, model_server, host_engine, capfd):
    model_server.reply_dict["edit"] = ADD_SUB_REPLY_LIST
    head = git(calc_repo, "rev-parse", "HEAD").strip()
    assert _sandbox(calc_repo, out_dir) == 0
    out = capfd.readouterr().out
    assert "applied to the working tree: 1 file(s)" in out and "Commit them" in out
    assert "branch:" not in out and "merge" not in out
    assert "verdict: unverified" in out
    # The change is in the working tree as a change to commit; HEAD is as it was, and
    # the branch it landed on first is gone
    assert git(calc_repo, "status", "--porcelain").strip() == "M calc/basic.py"
    assert git(calc_repo, "rev-parse", "HEAD").strip() == head
    assert "def sub(" in (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
    assert branch_list(calc_repo) == ["main"]
    # The container started at HEAD, the tree being what HEAD holds
    assert _base_of(engine_run_list(host_engine)[-1]) == head
    # The patch, the turn log and the record stay with the agent; its copy of the index
    # is gone, the run having ended
    agent_out = _agent_out(out_dir, "")
    patch = _branch_of(out, "patch")
    assert patch.startswith(str(agent_out)) and os.path.exists(patch)
    record = json.loads((agent_out / "calc" / history.HISTORY_FILE).read_text())
    assert record["attempts"][0]["verdict"] == "unverified"
    assert (agent_out / "calc" / "turns").is_dir()
    assert not (agent_out / "calc" / "project_knowledge.sqlite").exists()
    # The cache was filled first, then the change was made, each in a run of its own
    run_list = engine_run_list(host_engine)
    assert [r["tool_args"][0] for r in run_list] == ["run-tests", "edit"]
    assert (out_dir / "agents" / ".repo" / "calc" / "cache.filled").exists()


def test_sandbox_makes_the_overview_in_a_container_of_its_own_before_the_change(
        calc_repo, out_dir, model_server, host_engine, monkeypatch, capfd):
    """The stage that writes the overview runs the model's code, so it runs where the
    change does: behind the broker, with neither the key nor a network."""
    monkeypatch.setenv("OVERVIEW", "true")
    importlib.reload(config)
    model_server.reply_dict["answer"] = [complete("It offers arithmetic on two numbers.")]
    model_server.reply_dict["edit"] = ADD_SUB_REPLY_LIST
    assert _sandbox(calc_repo, out_dir) == 0
    run_list = engine_run_list(host_engine)
    assert [r["tool_args"][0] for r in run_list] == ["run-tests", "overview", "edit"]
    overview_env = run_list[1]["env"]
    assert "LLM_API_KEY" not in overview_env and "LLM_API_BASE" not in overview_env
    assert overview_env["LLM_BROKER"].startswith("unix:///")
    # What the container left in its copy was written into the output directory, and
    # the copy made for it is gone
    record = json.loads((out_dir / "calc" / "overview.json").read_text(encoding="utf-8"))
    assert record["answer"] == "It offers arithmetic on two numbers."
    overview_out = run_list[1]["tool_args"][run_list[1]["tool_args"].index("--output-dir") + 1]
    assert overview_out != str(out_dir) and not os.path.exists(
        os.path.join(overview_out, "calc", "overview.json"))
    # The container that makes the change is handed the state the overview was made for
    assert run_list[2]["env"]["CTLOOM_OVERVIEW_DIGEST"] == record["file_digest"]
    assert model_server.stage_list()[0] == "answer"
    assert git(calc_repo, "status", "--porcelain").strip() == "M calc/basic.py"


def test_the_container_is_given_neither_the_key_nor_the_address(calc_repo, out_dir,
                                                                model_server,
                                                                host_engine):
    model_server.reply_dict["edit"] = ADD_SUB_REPLY_LIST
    _sandbox(calc_repo, out_dir)
    for run in engine_run_list(host_engine):
        env = run["env"]
        assert "LLM_API_KEY" not in env and "LLM_API_BASE" not in env
        assert not any("scripted-key" in value for value in env.values())
    edit_env = engine_run_list(host_engine)[-1]["env"]
    assert edit_env["LLM_BROKER"].startswith("unix:///")
    assert edit_env["TOOL_OFFLINE"] == "true"
    # The container clones a bare copy of the history in its scratch directory, not
    # the repository
    src = edit_env["CTLOOM_SRC"]
    assert os.path.basename(src) == "src"
    assert os.path.basename(os.path.dirname(src)).startswith("scratch-")
    assert src.startswith(str(out_dir))
    # Every model call went through the broker's socket: the container held no address
    assert model_server.stage_list() == ["edit"] * 3


def test_sandbox_starts_from_changes_that_are_not_committed_and_lands_over_them(
        calc_repo, out_dir, model_server, host_engine, capfd):
    """A file added to the tree and a change to a tracked file, neither committed,
    are what the container reads; the change it makes goes in over them, and they
    stay as they were."""
    model_server.reply_dict["edit"] = ADD_SUB_REPLY_LIST
    head = git(calc_repo, "rev-parse", "HEAD").strip()
    (calc_repo / "docs").mkdir()
    (calc_repo / "docs" / "design.md").write_text("# sub\n", encoding="utf-8")
    scale_text = (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8") + "# note\n"
    (calc_repo / "calc" / "scale.py").write_text(scale_text, encoding="utf-8")
    assert _sandbox(calc_repo, out_dir) == 0
    out = capfd.readouterr().out
    assert "applied to the working tree: 1 file(s)" in out
    base = _base_of(engine_run_list(host_engine)[-1])
    assert base != head and git(calc_repo, "rev-parse", f"{base}^").strip() == head
    assert git(calc_repo, "show", f"{base}:docs/design.md") == "# sub\n"
    assert git(calc_repo, "show", f"{base}:calc/scale.py") == scale_text
    assert "def sub(" in (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
    assert (calc_repo / "docs" / "design.md").read_text(encoding="utf-8") == "# sub\n"
    assert (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8") == scale_text
    assert sorted(git(calc_repo, "status", "--porcelain").splitlines()) == \
        [" M calc/basic.py", " M calc/scale.py", "?? docs/"]
    assert git(calc_repo, "rev-parse", "HEAD").strip() == head
    assert branch_list(calc_repo) == ["main"]
    # The analysis made here describes the tree, so no container analysed it again
    assert all(run["env"]["CTLOOM_REBUILD_INDEX"] == "0" for run in engine_run_list(host_engine))


def test_a_dry_run_in_the_container_changes_nothing(calc_repo, out_dir, model_server,
                                                    host_engine, capfd):
    model_server.reply_dict["edit"] = ADD_SUB_REPLY_LIST
    assert _sandbox(calc_repo, out_dir, "--dry-run") == 0
    out = capfd.readouterr().out
    assert "changed: nothing" in out
    assert branch_list(calc_repo) == ["main"]
    assert "--dry-run" in engine_run_list(host_engine)[-1]["tool_args"]


def test_a_change_in_the_container_is_recorded_with_the_history_switched_off_here(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    monkeypatch.setenv("EDIT_HISTORY", "false")
    importlib.reload(config)
    model_server.reply_dict["edit"] = ADD_SUB_REPLY_LIST
    assert _sandbox(calc_repo, out_dir) == 0
    # The verdict this machine prints and writes into the commit is read off the record
    assert "verdict: unverified" in capfd.readouterr().out
    record = json.loads((_agent_out(out_dir, "") / "calc" / history.HISTORY_FILE).read_text())
    assert record["attempts"][0]["verdict"] == "unverified"
    assert engine_run_list(host_engine)[-1]["env"]["EDIT_HISTORY"] == "true"


# == the image ==================================================
def test_sandbox_stops_at_an_image_that_is_not_there_with_the_command_that_builds_it(
        calc_repo, out_dir, model_server, host_engine, monkeypatch, tmp_path):
    monkeypatch.setenv("CTLOOM_HOST_ENGINE_IMAGE_FILE", str(tmp_path / "images.json"))
    with pytest.raises(SystemExit) as raised:
        cli.main(["sandbox", "--project", str(calc_repo), "--output-dir", str(out_dir),
                  "--request", "add a sub function beside add"])
    mark = image.source_mark(image.source_dir())
    assert str(raised.value).startswith("sandbox: podman does not show the image "
                                        "codetwine-loom (exit status 125)")
    assert image.build_command_text("podman", "codetwine-loom", image.DEFAULT_TOOLCHAIN,
                                    mark, image.source_dir()) in str(raised.value)
    assert engine_run_list(host_engine) == []


def test_sandbox_builds_the_image_on_a_yes_and_goes_on(calc_repo, out_dir, model_server,
                                                       host_engine, monkeypatch, tmp_path,
                                                       capfd):
    image_file = tmp_path / "images.json"
    monkeypatch.setenv("CTLOOM_HOST_ENGINE_IMAGE_FILE", str(image_file))
    monkeypatch.setattr(cli.progress, "at_terminal", lambda: True)
    monkeypatch.setattr(cli, "_env_file_ask_path", lambda: "")
    asked = []
    monkeypatch.setattr(cli.progress, "ask", lambda question, is_secret=False: (
        asked.append(question) or ("y" if question.startswith("Build") else "n")))
    model_server.reply_dict["edit"] = ADD_SUB_REPLY_LIST
    assert _sandbox(calc_repo, out_dir) == 0
    record = json.loads(image_file.read_text(encoding="utf-8"))
    assert record["image_dict"]["codetwine-loom"]["io.codetwine-loom.source"] == \
        image.source_mark(image.source_dir())
    assert len(record["build_list"]) == 1
    out = capfd.readouterr().out
    assert "built the image codetwine-loom" in out
    assert "applied to the working tree: 1 file(s)" in out
    assert asked[0] == "Build it now? [Y/n]: "
    assert any("codetwine-loom" in run["argv"] for run in engine_run_list(host_engine))


# == warm =======================================================
def test_warm_fills_the_cache_and_leaves_its_mark(calc_repo, out_dir, model_server,
                                                  host_engine, capfd):
    assert _main("warm", "--project", str(calc_repo), "--output-dir", str(out_dir)) == 0
    out = capfd.readouterr().out
    cache = _branch_of(out, "cache")
    assert cache == str(out_dir / "agents" / ".repo" / "calc" / "cache")
    assert re.fullmatch(r"[0-9a-f]{64}\n", (out_dir / "agents" / ".repo" / "calc"
                                            / "cache.filled").read_text())
    run = engine_run_list(host_engine)[0]
    assert run["tool_args"][0] == "run-tests" and run["env"]["TOOL_OFFLINE"] == "false"
    assert "LLM_BROKER" not in run["env"] or run["env"]["LLM_BROKER"] == ""
    assert model_server.call_list == []


# == integrate ==================================================
def _two_branches(calc_repo) -> None:
    """Make branches a and b off main, each adding a function to a file of its own."""
    base = git(calc_repo, "rev-parse", "HEAD").strip()
    git(calc_repo, "checkout", "-q", "-b", "a", base)
    (calc_repo / "calc" / "basic.py").write_text(
        (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
        + "\n\ndef sub(a, b):\n    return a - b\n", encoding="utf-8")
    commit_all(calc_repo, "sandbox: add sub")
    git(calc_repo, "checkout", "-q", "-b", "b", base)
    (calc_repo / "calc" / "scale.py").write_text(
        (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
        + "\n\ndef div(a, b):\n    return a / b\n", encoding="utf-8")
    commit_all(calc_repo, "sandbox: add div")
    git(calc_repo, "checkout", "-q", "main")


def test_integrate_combines_two_branches_in_a_clone_and_puts_the_union_in_the_tree(
        calc_repo, out_dir, model_server, host_engine, capfd):
    _two_branches(calc_repo)
    head = git(calc_repo, "rev-parse", "HEAD").strip()
    assert _main("integrate", "--project", str(calc_repo), "--output-dir", str(out_dir),
                 "--branch", "a", "--branch", "b") == 0
    out = capfd.readouterr().out
    assert "settled: by git alone, with no model called" in out
    assert "applied to the working tree: 2 file(s)" in out
    assert "def sub(" in (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
    assert "def div(" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
    assert model_server.call_list == []
    assert sorted(git(calc_repo, "status", "--porcelain").splitlines()) == \
        [" M calc/basic.py", " M calc/scale.py"]
    assert git(calc_repo, "rev-parse", "HEAD").strip() == head
    assert branch_list(calc_repo) == ["a", "b", "main"]
    # The container was told the branch pointing at the tree, and combined it first
    run = engine_run_list(host_engine)[-1]
    assert run["tool_args"][run["tool_args"].index("--onto") + 1].endswith("/tree")
    assert run["tool_args"][run["tool_args"].index("--onto-commit") + 1] == head


# == run ========================================================
def _run(calc_repo, out_dir, request: str, *argv: str) -> int:
    return _main("run", "--project", str(calc_repo), "--output-dir", str(out_dir),
                 "--request", request, *argv)


def _split_into_two_tasks(model_server) -> None:
    """Have the reading run split the request into the sub task and the div task."""
    model_server.reply_dict["dispatch"] = [complete_json({"tasks": [_SUB_TASK, _DIV_TASK]})]
    model_server.add("edit", _SUB_TASK, ADD_SUB_REPLY_LIST)
    model_server.add("edit", _DIV_TASK, ADD_DIV_REPLY_LIST)
    model_server.reply_dict["report"] = ["Both functions were added and the tests passed."]


def test_run_splits_the_request_runs_an_agent_per_task_and_combines_the_branches(
        calc_repo, out_dir, model_server, host_engine, capfd):
    _split_into_two_tasks(model_server)
    head = git(calc_repo, "rev-parse", "HEAD").strip()
    assert _run(calc_repo, out_dir, "add sub and div") == 0
    out = capfd.readouterr().out
    assert "tasks: 2" in out
    assert "applied to the working tree: 2 file(s)" in out and "branch:" not in out
    assert "Both functions were added and the tests passed." in out
    assert "changed:" in out and "M calc/basic.py" in out and "M calc/scale.py" in out
    # The tree holds both changes, to commit; the task branches and the union's are gone
    assert "def sub(" in (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
    assert "def div(" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
    assert branch_list(calc_repo) == ["main"]
    assert sorted(git(calc_repo, "status", "--porcelain").splitlines()) == \
        [" M calc/basic.py", " M calc/scale.py"]
    assert git(calc_repo, "rev-parse", "HEAD").strip() == head
    stage_list = model_server.stage_list()
    assert stage_list.count("dispatch") == 1 and stage_list.count("edit") == 6
    assert stage_list.count("report") == 1 and "integrate" not in stage_list
    assert [r["tool_args"][0] for r in engine_run_list(host_engine)] == \
        ["run-tests", "run", "edit", "edit", "integrate"]


def test_run_says_what_every_agent_is_doing_and_how_many_tasks_have_finished(
        calc_repo, out_dir, model_server, host_engine, capfd, caplog):
    _split_into_two_tasks(model_server)
    with caplog.at_level(logging.INFO):
        assert _run(calc_repo, out_dir, "add sub and div") == 0
    out, err = capfd.readouterr()
    # What the containers write to stderr carries no clock time of its own
    assert not re.search(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", err)
    # What the containers log and print stays in files
    assert "codetwine_loom.rlm" not in err and "codetwine_loom.sandbox" not in err
    assert not re.search(r"^\[task ", out, re.M)
    kept = "".join(path.read_text(encoding="utf-8")
                   for path in out_dir.rglob("stdout-*.txt"))
    for n in (1, 2):
        assert re.search(rf"^\[task {n}/2\] verdict: ", kept, re.M), kept
    assert list(out_dir.rglob("turns/*.log"))
    # A turn, a call's token counts and the libraries' own lines are below what is shown
    assert not re.search(r" turn \d+ of at most ", err) and "input token(s)" not in err
    assert "LiteLLM" not in err
    message_list = [r.getMessage() for r in caplog.records]
    for name in ("prepare the analysis", "split the request", "run the tasks",
                 "combine the branches", "write the report"):
        assert f"{name}: start" in message_list and any(
            m.startswith(f"{name}: end (") for m in message_list), name
    assert any(m.endswith(": 1 of 2 finished, 1 running")
               or m.endswith(": 1 of 2 finished, 0 running") for m in message_list)
    assert any(m.endswith(": 2 of 2 finished, 0 running") for m in message_list)
    # What the run prints at its end is as it was
    assert "tasks: 2" in out and "applied to the working tree" in out


def test_a_run_of_one_agent_has_no_label(calc_repo, out_dir, model_server, host_engine,
                                         capfd):
    model_server.reply_dict["edit"] = list(ADD_SUB_REPLY_LIST)
    assert _sandbox(calc_repo, out_dir) == 0
    out, err = capfd.readouterr()
    assert re.search(r"^verdict: unverified", out, re.M)
    assert "codetwine_loom.rlm" not in err
    assert not [line for line in (out + err).splitlines() if line.startswith("[")]


def test_run_starts_from_changes_that_are_not_committed_and_lands_over_them(
        calc_repo, out_dir, model_server, host_engine, capfd):
    """Every container of the run - the reading run, the agents, the merging run -
    starts at a commit of the tree as it is, and the union goes in over it."""
    _split_into_two_tasks(model_server)
    head = git(calc_repo, "rev-parse", "HEAD").strip()
    (calc_repo / "NOTES.md").write_text("read me\n", encoding="utf-8")
    assert _run(calc_repo, out_dir, "add sub and div") == 0
    out = capfd.readouterr().out
    assert "applied to the working tree: 2 file(s)" in out
    run_list = engine_run_list(host_engine)
    base_set = {_base_of(run) for run in run_list if run["tool_args"][0] != "run-tests"}
    assert len(base_set) == 1 and head not in base_set
    base = base_set.pop()
    assert git(calc_repo, "show", f"{base}:NOTES.md") == "read me\n"
    assert "def sub(" in (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
    assert "def div(" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
    assert (calc_repo / "NOTES.md").read_text(encoding="utf-8") == "read me\n"
    assert branch_list(calc_repo) == ["main"]
    assert git(calc_repo, "rev-parse", "HEAD").strip() == head


def test_a_task_that_does_not_finish_leaves_its_branch_to_carry_on_from(
        calc_repo, out_dir, model_server, host_engine, capfd):
    model_server.reply_dict["dispatch"] = [complete_json({"tasks": [_SUB_TASK, _DIV_TASK]})]
    model_server.add("edit", _SUB_TASK, ADD_SUB_REPLY_LIST)
    model_server.add("edit", _DIV_TASK, [code("print('still reading')")])
    model_server.late_reply = "I changed nothing"
    model_server.reply_dict["report"] = ["One task finished; the other ran out of turns."]
    assert _run(calc_repo, out_dir, "add sub and div") == 1
    out = capfd.readouterr().out
    assert "it did not finish" in out and "combined: nothing (1 task(s) did not finish" in out
    resume_line = re.search(r"to carry on from what this run left: run (.*)$", out, re.M)
    assert resume_line, out
    resume_branch = resume_line.group(1).split()[-1]
    assert resume_branch in branch_list(calc_repo)
    assert "def sub(" in git(calc_repo, "show", f"{resume_branch}:calc/basic.py")
    assert "codetwine-loom/sandbox/" not in "".join(branch_list(calc_repo))
    # Nothing went into the tree, and no branch of the tree is left
    assert git(calc_repo, "status", "--porcelain") == ""
    assert "/tree" not in "".join(branch_list(calc_repo))
    # The next run is told what the earlier one left, in front of the request
    model_server.call_list.clear()
    model_server.rule_list.clear()
    model_server.add("edit", _DIV_TASK, ADD_DIV_REPLY_LIST)
    model_server.reply_dict["dispatch"] = [complete_json({"tasks": [_DIV_TASK]})]
    assert _run(calc_repo, out_dir, "add sub and div", "--resume", resume_branch) == 0
    prompt = model_server.prompt_list("dispatch")[0]
    assert f"## What an earlier run left on {resume_branch}" in prompt
    assert "asked: " + _SUB_TASK in prompt
    assert "def div(" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")



# == the stage that writes tests ================================
_TEST_DIV_TEXT = """from calc.scale import div


def test_div():  # case_1
    assert div(6, 3) == 2
"""

_DIV_CASE = {"file": "calc/scale.py", "name": "div", "input": "div(6, 3)",
             "expect": "returns 2", "from": "add div to calc/scale.py"}
_SUB_CASE = {"file": "calc/basic.py", "name": "sub", "input": "sub(5, 3)",
             "expect": "returns 2", "from": "add sub to calc/basic.py"}

# The first stage's run: one case for div(), taken from the task, and a reason for sub()
_DIV_CASE_REPLY_LIST = [complete_json({
    "cases": [_DIV_CASE],
    "no_case": [{"file": "calc/basic.py", "name": "sub", "why": "left to another run"}]})]

_TEST_SUB_TEXT = """from calc.basic import sub


def test_sub():  # case_1
    assert sub(5, 3) == 2
"""

# The stage's run: write a test for div(), run it alone, and finish
_WRITE_DIV_TEST_REPLY_LIST = [
    code('print(create_file("tests/test_div.py", ' + repr(_TEST_DIV_TEXT) + "))"),
    code('print(run_test(["tests/test_div.py"]))'),
    complete("wrote tests/test_div.py for div()"),
]

# The same run with a test that does not pass, handed over every time it is sent back
_WRITE_RED_TEST_REPLY_LIST = [
    code('print(create_file("tests/test_div.py", '
         + repr(_TEST_DIV_TEXT.replace("== 2", "== 3")) + "))"),
    complete("wrote tests/test_div.py for div()"),
]


def _with_test_create(monkeypatch, model_server) -> None:
    monkeypatch.setenv("TEST_CREATE", "true")
    importlib.reload(config)
    model_server.reply_dict["test_case"] = list(_DIV_CASE_REPLY_LIST)


def test_run_writes_tests_on_top_of_the_union_and_puts_both_in_the_tree(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    _with_test_create(monkeypatch, model_server)
    _split_into_two_tasks(model_server)
    model_server.reply_dict["test_create"] = list(_WRITE_DIV_TEST_REPLY_LIST)
    assert _run(calc_repo, out_dir, "add sub and div") == 0
    out = capfd.readouterr().out
    # The run that lists what there is to test, then the run that writes the tests
    run_list = engine_run_list(host_engine)
    assert [r["tool_args"][0] for r in run_list][-3:] == \
        ["integrate", "test-create", "test-create"]
    assert "--target-only" in run_list[-2]["tool_args"]
    assert "--target-in" in run_list[-1]["tool_args"]
    assert "written: tests/test_div.py" in out and "on the base: red" in out
    assert "to carry on from" not in out
    assert "applied to the working tree: 3 file(s)" in out
    test_commit = _commit_of(out, "tests")
    message = git(calc_repo, "log", "-1", "--format=%B", test_commit)
    assert message.startswith("test-create: add sub and div")
    for line in ("Codetwine-Loom-Stage: test-create", "Codetwine-Loom-Verdict: settled",
                 "Codetwine-Loom-Tests-Base: red",
                 "Codetwine-Loom-Tests-Changed: tests/test_div.py"):
        assert line in message
    assert "Codetwine-Loom-Non-Test-Changed" not in message
    assert git(calc_repo, "log", "-1", "--format=%s", f"{test_commit}^").startswith("integrate: ")
    assert (calc_repo / "tests" / "test_div.py").read_text(encoding="utf-8") == \
        _TEST_DIV_TEXT
    assert branch_list(calc_repo) == ["main"]
    assert sorted(git(calc_repo, "status", "--porcelain").splitlines()) == \
        [" M calc/basic.py", " M calc/scale.py", "?? tests/test_div.py"]
    # The cases are decided first, from the tasks and the first line of each definition,
    # by a stage that is shown neither the code nor what the agents said they did
    stage_list = model_server.stage_list()
    assert stage_list.index("test_case") < stage_list.index("test_create")
    case_prompt = model_server.prompt_list("test_case")[0]
    assert _DIV_TASK in case_prompt and "def div(a: float, b: float) -> float:" in case_prompt
    assert "return a / b" not in case_prompt and "added div()" not in case_prompt
    assert "replace_in_file" not in case_prompt
    # The stage that writes is handed the case, and is not told what the agents said
    prompt = model_server.prompt_list("test_create")[0]
    assert "asked: " + _DIV_TASK in prompt and "case_1 div in calc/scale.py" in prompt
    assert "added div()" not in prompt
    # It is shown where the tests are, and told its own file access does not reach them
    assert "  tests/test_scale.py" in prompt and rlm.REPL_REACH_NOTE in prompt
    assert "cases decided from the request: 1" in out


def test_tests_that_do_not_pass_stay_on_the_branch_and_are_written_again_on_resume(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    _with_test_create(monkeypatch, model_server)
    _split_into_two_tasks(model_server)
    model_server.reply_dict["test_create"] = list(_WRITE_RED_TEST_REPLY_LIST)
    head = git(calc_repo, "rev-parse", "HEAD").strip()
    assert _run(calc_repo, out_dir, "add sub and div") == 0
    out = capfd.readouterr().out
    assert "NOT settled" in out and "applied to the working tree: 2 file(s)" in out
    resume_line = re.search(r"to carry on from what this run left: run (.*)$", out, re.M)
    assert resume_line, out
    resume_branch = resume_line.group(1).split()[-1]
    # The tree holds the union alone, and the branch holds the tests over it
    assert "def div(" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
    assert not (calc_repo / "tests" / "test_div.py").exists()
    message = git(calc_repo, "log", "-1", "--format=%B", resume_branch)
    assert "Codetwine-Loom-Stage: test-create" in message \
        and "Codetwine-Loom-Verdict: unsettled" in message
    assert git(calc_repo, "log", "-1", "--format=%s", f"{resume_branch}^").startswith("integrate: ")
    assert git(calc_repo, "rev-parse", f"{resume_branch}^^").strip() == head

    # Carrying on starts the one container that writes the tests, over the same union,
    # and puts the tests in over what is already in the tree
    model_server.call_list.clear()
    model_server.reply_dict["test_create"] = list(_WRITE_DIV_TEST_REPLY_LIST)
    run_count = len(engine_run_list(host_engine))
    assert _run(calc_repo, out_dir, "", "--resume", resume_branch) == 0
    out = capfd.readouterr().out
    assert [r["tool_args"][0] for r in engine_run_list(host_engine)][run_count:] == \
        ["test-create", "test-create"]
    assert "dispatch" not in model_server.stage_list()
    assert "applied to the working tree: 1 file(s)" in out
    assert git(calc_repo, "log", "-1", "--format=%s", _commit_of(out, "tests")).startswith(
        "test-create: ")
    assert (calc_repo / "tests" / "test_div.py").read_text(encoding="utf-8") == \
        _TEST_DIV_TEXT
    assert "def div(" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
    assert branch_list(calc_repo) == ["main"]


# Two test tasks number their cases apart: the second one's first case is case_2
_TEST_DIV_SECOND_TEXT = _TEST_DIV_TEXT.replace("# case_1", "# case_2")

_WRITE_DIV_SECOND_TEST_REPLY_LIST = [
    code('print(create_file("tests/test_div.py", ' + repr(_TEST_DIV_SECOND_TEXT) + "))"),
    code('print(run_test(["tests/test_div.py"]))'),
    complete("wrote tests/test_div.py for div()"),
]


def test_two_test_tasks_write_their_tests_at_once_and_are_combined_into_one_commit(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    _with_test_create(monkeypatch, model_server)
    monkeypatch.setenv("TEST_CREATE_MAX_DEFINITION", "1")
    importlib.reload(config)
    _split_into_two_tasks(model_server)
    # Each test task is shown its own definition alone, and answers for it alone
    model_server.add("test_case", "sub in calc/basic.py", [complete_json({"cases": [_SUB_CASE]})])
    model_server.add("test_case", "div in calc/scale.py", [complete_json({"cases": [_DIV_CASE]})])
    model_server.add("test_create", "case_1 sub in calc/basic.py", [
        code('print(create_file("tests/test_sub.py", ' + repr(_TEST_SUB_TEXT) + "))"),
        complete("wrote tests/test_sub.py for sub()")])
    model_server.add("test_create", "case_2 div in calc/scale.py",
                     list(_WRITE_DIV_SECOND_TEST_REPLY_LIST))
    assert _run(calc_repo, out_dir, "add sub and div") == 0
    out = capfd.readouterr().out
    tail_list = [r["tool_args"][0] for r in engine_run_list(host_engine)][-5:]
    assert tail_list == ["integrate", "test-create", "test-create", "test-create", "integrate"]
    assert "test tasks: 2 (2 settled)" in out and "cases decided from the request: 2" in out
    assert "to carry on from" not in out
    test_commit = _commit_of(out, "tests")
    message = git(calc_repo, "log", "-1", "--format=%B", test_commit)
    assert message.startswith("test-create: add sub and div")
    assert "Codetwine-Loom-Stage: test-create" in message \
        and "Codetwine-Loom-Verdict: settled" in message
    assert "Codetwine-Loom-Test-Open" not in message
    parent_list = git(calc_repo, "rev-list", "--parents", "-n", "1", test_commit).split()[1:]
    assert len(parent_list) == 3
    assert git(calc_repo, "log", "-1", "--format=%s", parent_list[0]).startswith("integrate: ")
    for name in ("test_sub.py", "test_div.py"):
        assert (calc_repo / "tests" / name).exists()
    assert branch_list(calc_repo) == ["main"]
    assert "applied to the working tree: 4 file(s)" in out
    # The test task writing for sub() was not shown div(), and the other way round
    for prompt in model_server.prompt_list("test_case"):
        assert ("sub in calc/basic.py" in prompt) != ("div in calc/scale.py" in prompt)
    # Each is told which files it can write to: its own, and the test that refers to it
    for prompt in model_server.prompt_list("test_create"):
        own, other = ("basic", "scale") if "case_1 sub in" in prompt else ("scale", "basic")
        block = prompt.split("The files you can write to")[1].split("\n\n")[0]
        assert f"calc/{own}.py" in block and f"tests/test_{own}.py" in block
        assert f"calc/{other}.py" not in block and f"tests/test_{other}.py" not in block


def test_a_test_task_that_reaches_into_anothers_file_is_refused_and_left_open(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    _with_test_create(monkeypatch, model_server)
    monkeypatch.setenv("TEST_CREATE_MAX_DEFINITION", "1")
    # The test task left open is what is looked at, so it is not written again
    monkeypatch.setenv("TEST_CREATE_MAX_RERUN", "0")
    importlib.reload(config)
    _split_into_two_tasks(model_server)
    model_server.add("test_case", "sub in calc/basic.py", [complete_json({"cases": [_SUB_CASE]})])
    model_server.add("test_case", "div in calc/scale.py", [complete_json({"cases": [_DIV_CASE]})])
    # The test task for sub() tries to change calc/scale.py, which is the other one's
    model_server.add("test_create", "case_1 sub in calc/basic.py", [
        code('print(replace_in_file("calc/scale.py", "return a * b", "return b * a"))'),
        code('print(create_file("tests/test_sub.py", ' + repr(_TEST_SUB_TEXT) + "))"),
        complete("wrote tests/test_sub.py for sub()")])
    model_server.add("test_create", "case_2 div in calc/scale.py",
                     list(_WRITE_DIV_SECOND_TEST_REPLY_LIST))
    assert _run(calc_repo, out_dir, "add sub and div") == 0
    out = capfd.readouterr().out
    assert "test tasks: 2 (1 settled)" in out
    assert "writes refused (not the test task's files): calc/scale.py" in out
    assert "a change outside this run's files was refused: calc/scale.py" in out
    # The tree holds the union alone, and calc/scale.py is as the union left it
    assert "applied to the working tree: 2 file(s)" in out
    assert "return a * b" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
    assert not (calc_repo / "tests" / "test_sub.py").exists()
    resume_line = re.search(r"to carry on from what this run left: run (.*)$", out, re.M)
    assert resume_line, out
    message = git(calc_repo, "log", "-1", "--format=%B", resume_line.group(1).split()[-1])
    assert 'Codetwine-Loom-Test-Open: ["calc/basic.py"]' in message


def test_a_repository_without_tests_gets_its_first_ones(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    _with_test_create(monkeypatch, model_server)
    model_server.reply_dict["test_create"] = list(_WRITE_DIV_TEST_REPLY_LIST)
    git(calc_repo, "rm", "-r", "-q", "tests")
    commit_all(calc_repo, "drop the tests")
    _two_branches(calc_repo)
    assert _main("integrate", "--project", str(calc_repo), "--output-dir", str(out_dir),
                 "--branch", "a", "--branch", "b") == 0
    out = capfd.readouterr().out
    assert [r["tool_args"][0] for r in engine_run_list(host_engine)][-3:] == \
        ["integrate", "test-create", "test-create"]
    assert "written: tests/test_div.py" in out and "applied to the working tree" in out
    message = git(calc_repo, "log", "-1", "--format=%B", _commit_of(out, "tests"))
    assert message.startswith("test-create: ")
    assert "Codetwine-Loom-Tests-Changed: tests/test_div.py" in message
    assert (calc_repo / "tests" / "test_div.py").read_text(encoding="utf-8") == \
        _TEST_DIV_TEXT
    # The stage started with no test file to read and an empty baseline, and was told
    # it can write the language's dependency file
    case_prompt = model_server.prompt_list("test_case")[0]
    assert "which read_source_file reads:" in case_prompt and "  tests/" not in case_prompt
    prompt = model_server.prompt_list("test_create")[0]
    assert "a dependency file (such as pyproject.toml)" in prompt
    assert "pyproject.toml  (dependency file)" in prompt
    assert "Tests that were failing before you started" not in prompt


# == the stages that refactor ===================================
_HOUSE_TEXT = "Every public function has a one-line docstring.\n"
_TIDY_TEXT = "Say what kind of division a function does in its docstring.\n"
_TEST_RULE_TEXT = "A test file starts with a docstring naming what it tests.\n"
_TEST_TIDY_TEXT = "Set-up two tests share goes in a fixture.\n"

# The stage's run over the change: reword div()'s docstring, run the tests, and finish
_REFACTOR_DIV_REPLY_LIST = [
    code('print(replace_in_file("calc/scale.py", "Return a divided by b.", '
         '"Return a divided by b, as a float."))'),
    code("print(run_test())"),
    complete("reworded the docstring of div()"),
]

# The same run breaking what div() does, handed over every time it is sent back
_REFACTOR_BREAK_REPLY_LIST = [
    code('print(replace_in_file("calc/scale.py", ' + repr("    return a / b\n") + ", "
         + repr("    return a // b + 1\n") + "))"),
    complete("simplified div()"),
]

# The stage's run over the tests: give the test file written for div() a docstring
_REFACTOR_TEST_REPLY_LIST = [
    code('print(replace_in_file("tests/test_div.py", '
         + repr("from calc.scale import div\n") + ", "
         + repr('"""The tests of div()."""\nfrom calc.scale import div\n') + "))"),
    code('print(run_test(["tests/test_div.py"]))'),
    complete("gave tests/test_div.py a docstring"),
]


def _with_every_stage(monkeypatch, model_server) -> None:
    monkeypatch.setenv("REFACTOR", "true")
    monkeypatch.setenv("TEST_REFACTOR", "true")
    _with_test_create(monkeypatch, model_server)
    model_server.reply_dict["test_create"] = list(_WRITE_DIV_TEST_REPLY_LIST)


def test_run_refactors_the_change_and_then_its_tests_each_in_a_commit_of_its_own(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    (calc_repo / "CODETWINE_LOOM.md").write_text(_HOUSE_TEXT, encoding="utf-8")
    (calc_repo / "CODETWINE_LOOM_REFACTOR.md").write_text(_TIDY_TEXT, encoding="utf-8")
    (calc_repo / "CODETWINE_LOOM_TEST.md").write_text(_TEST_RULE_TEXT, encoding="utf-8")
    (calc_repo / "CODETWINE_LOOM_TEST_REFACTOR.md").write_text(_TEST_TIDY_TEXT, encoding="utf-8")
    commit_all(calc_repo, "say how this repository is changed, tested and refactored")
    _with_every_stage(monkeypatch, model_server)
    _split_into_two_tasks(model_server)
    model_server.reply_dict["refactor"] = list(_REFACTOR_DIV_REPLY_LIST)
    model_server.reply_dict["test_refactor"] = list(_REFACTOR_TEST_REPLY_LIST)
    assert _run(calc_repo, out_dir, "add sub and div") == 0
    out = capfd.readouterr().out
    run_list = engine_run_list(host_engine)
    assert [r["tool_args"][0] for r in run_list][-5:] == \
        ["integrate", "test-create", "test-create", "refactor", "test-refactor"]
    # One commit per stage under what went into the tree, the newest first
    last = _commit_of(out, "test refactoring")
    subject_list = git(calc_repo, "log", "-4", "--first-parent", "--format=%s",
                       last).splitlines()
    assert [subject.split(":")[0] for subject in subject_list] == \
        ["test-refactor", "refactor", "test-create", "integrate"]
    message = git(calc_repo, "log", "-1", "--format=%B", f"{last}^")
    for line in ("Codetwine-Loom-Stage: refactor", "Codetwine-Loom-Verdict: settled",
                 "Codetwine-Loom-Non-Test-Changed: calc/scale.py"):
        assert line in message
    message = git(calc_repo, "log", "-1", "--format=%B", last)
    for line in ("Codetwine-Loom-Stage: test-refactor", "Codetwine-Loom-Verdict: settled",
                 "Codetwine-Loom-Tests-Changed: tests/test_div.py"):
        assert line in message
    assert "as a float" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
    assert (calc_repo / "tests" / "test_div.py").read_text(encoding="utf-8").startswith(
        '"""The tests of div()."""')
    assert branch_list(calc_repo) == ["main"]
    assert "applied to the working tree: 3 file(s)" in out
    assert "refactoring: " in out and "test refactoring: " in out
    assert "to carry on from" not in out
    # Each stage is handed the one file written for it, and the stage that decides the
    # cases none
    house, tests, tidy, test_tidy = (text.strip() for text in (
        _HOUSE_TEXT, _TEST_RULE_TEXT, _TIDY_TEXT, _TEST_TIDY_TEXT))
    hand_dict = {"edit": {house}, "test_case": set(), "test_create": {tests},
                 "refactor": {tidy}, "test_refactor": {test_tidy}}
    for stage, hand_set in hand_dict.items():
        prompt_list = model_server.prompt_list(stage)
        assert prompt_list, stage
        for prompt in prompt_list:
            assert {text for text in (house, tests, tidy, test_tidy)
                    if text in prompt} == hand_set, stage
    for stage in ("refactor", "test_refactor"):
        assert "asked: " + _DIV_TASK in model_server.prompt_list(stage)[0]
    assert "  calc/scale.py — div" in model_server.prompt_list("refactor")[0]
    assert "  tests/test_div.py" in model_server.prompt_list("test_refactor")[0]


def test_a_refactoring_that_breaks_a_test_stays_on_the_branch_and_is_made_again_on_resume(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    _with_every_stage(monkeypatch, model_server)
    _split_into_two_tasks(model_server)
    model_server.reply_dict["refactor"] = list(_REFACTOR_BREAK_REPLY_LIST)
    model_server.reply_dict["test_refactor"] = list(_REFACTOR_TEST_REPLY_LIST)
    assert _run(calc_repo, out_dir, "add sub and div") == 0
    out = capfd.readouterr().out
    assert "NOT settled" in out and "test_refactor" not in model_server.stage_list()
    resume_line = re.search(r"to carry on from what this run left: run (.*)$", out, re.M)
    assert resume_line, out
    resume_branch = resume_line.group(1).split()[-1]
    # The tree holds the change and the tests, and the branch holds the refactoring
    assert "applied to the working tree: 3 file(s)" in out
    assert (calc_repo / "tests" / "test_div.py").exists()
    assert "return a / b" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
    message = git(calc_repo, "log", "-1", "--format=%B", resume_branch)
    assert "Codetwine-Loom-Stage: refactor" in message \
        and "Codetwine-Loom-Verdict: unsettled" in message
    assert git(calc_repo, "rev-parse", f"{resume_branch}^").strip() == \
        git(calc_repo, "rev-parse", _commit_of(out, "tests")).strip()

    # Carrying on starts with the stage that was left open, over the tests, and puts
    # in what the two stages changed
    model_server.call_list.clear()
    model_server.reply_dict["refactor"] = list(_REFACTOR_DIV_REPLY_LIST)
    run_count = len(engine_run_list(host_engine))
    assert _run(calc_repo, out_dir, "", "--resume", resume_branch) == 0
    out = capfd.readouterr().out
    assert [r["tool_args"][0] for r in engine_run_list(host_engine)][run_count:] == \
        ["refactor", "test-refactor"]
    assert "dispatch" not in model_server.stage_list()
    assert "test_create" not in model_server.stage_list()
    assert git(calc_repo, "log", "-1", "--format=%s",
               _commit_of(out, "test refactoring")).startswith("test-refactor: ")
    assert "applied to the working tree: 2 file(s)" in out
    assert "as a float" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")
    assert branch_list(calc_repo) == ["main"]


def test_a_repository_whose_tests_cannot_be_run_is_not_refactored_and_no_model_called(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    monkeypatch.setenv("REFACTOR", "true")
    monkeypatch.setenv("TEST_REFACTOR", "true")
    importlib.reload(config)
    git(calc_repo, "rm", "-r", "-q", "tests")
    commit_all(calc_repo, "drop the tests")
    _two_branches(calc_repo)
    assert _main("integrate", "--project", str(calc_repo), "--output-dir", str(out_dir),
                 "--branch", "a", "--branch", "b") == 0
    out = capfd.readouterr().out
    assert [r["tool_args"][0] for r in engine_run_list(host_engine)][-3:] == \
        ["integrate", "refactor", "test-refactor"]
    assert "refactoring: nothing done (" in out and "test refactoring: nothing done (" in out
    assert "applied to the working tree: 2 file(s)" in out and model_server.call_list == []
    assert "def sub(" in (calc_repo / "calc" / "basic.py").read_text(encoding="utf-8")
    assert "def div(" in (calc_repo / "calc" / "scale.py").read_text(encoding="utf-8")


_DIV_IN_BASIC_TASK = "put div into calc/basic.py"
_DIV_IN_BASIC_TEXT = '''    return a + b


def div(a: float, b: float) -> float:
    """Return a divided by b."""
    return a / b
'''

# One agent's run: add div() beside add(), on the line the sub task writes on too
_ADD_DIV_IN_BASIC_REPLY_LIST = [
    code('print(replace_in_file("calc/basic.py", "    return a + b\\n", '
         + repr(_DIV_IN_BASIC_TEXT) + "))"),
    code("print(run_test())"),
    complete("added div() to calc/basic.py; the tests passed"),
]


def test_a_merge_that_was_not_settled_is_run_again_from_its_union_told_what_it_recorded(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    """Both tasks finish and write on the same line, so the merge is what fails; it is
    run again from the union the first merge left, in a container told which branch
    that is, and its state block carries what the first merge recorded."""
    monkeypatch.setenv("INTEGRATE_MAX_REFUSAL", "1")
    monkeypatch.setenv("INTEGRATE_MAX_RERUN", "1")
    monkeypatch.setenv("DISPATCH_MAX_RERUN", "0")
    importlib.reload(config)
    model_server.reply_dict["dispatch"] = [
        complete_json({"tasks": [_SUB_TASK, _DIV_IN_BASIC_TASK]})]
    model_server.add("edit", _SUB_TASK, ADD_SUB_REPLY_LIST)
    model_server.add("edit", _DIV_IN_BASIC_TASK, _ADD_DIV_IN_BASIC_REPLY_LIST)
    model_server.reply_dict["integrate"] = [
        code('print(record_decision("calc/basic.py", "kept add as it was"))'),
        complete("done, I think")]
    model_server.add("integrate", "carries on from, on codetwine-loom/sandbox/", [complete("still not settled")])
    model_server.reply_dict["report"] = ["The merge was not settled twice."]
    assert _run(calc_repo, out_dir, "add sub and div") == 1
    out = capfd.readouterr().out
    assert "merges: 2 (the merges before the last were not settled)" in out
    run_list = [r["tool_args"] for r in engine_run_list(host_engine)]
    assert [r[0] for r in run_list] == ["run-tests", "run", "edit", "edit", "integrate", "integrate"]
    assert "--resume" not in run_list[4] and "--resume" in run_list[5]
    first_union = run_list[5][run_list[5].index("--resume") + 1]
    assert "--branch" not in run_list[5]
    # The second merge is told how the first ended and what it recorded
    second_prompt = model_server.prompt_list("integrate")[-1]
    assert f'carries on from, on {first_union}, ended with "refusals exhausted"' in second_prompt
    assert "calc/basic.py — kept add as it was" in second_prompt
    # The first union is a parent of the second, so its name is gone; the second is
    # what the next run is told to carry on from
    union_list = [b for b in branch_list(calc_repo) if b.startswith("codetwine-loom/sandbox/")]
    assert len(union_list) == 1 and first_union not in union_list
    resume_line = re.search(r"to carry on from what this run left: run (.*)$", out, re.M)
    assert resume_line and resume_line.group(1).split()[-1] == union_list[0]
    message = git(calc_repo, "log", "-1", "--format=%B", union_list[0])
    assert "Codetwine-Loom-Ended: refusals exhausted" in message
    # Both unions were left with the block of calc/basic.py in them, and each records
    # that the file may hold no separator line of it
    assert 'Codetwine-Loom-Separator: {"calc/basic.py":{"=======":0}}' in message.splitlines()
    assert 'Codetwine-Loom-Separator: {"calc/basic.py":{"=======":0}}' in git(
        calc_repo, "log", "-1", "--format=%B", f"{union_list[0]}^2").splitlines()


def test_a_question_is_answered_by_the_agents_and_put_together_without_a_branch(
        calc_repo, out_dir, model_server, host_engine, capfd):
    model_server.reply_dict["dispatch"] = [
        complete_json({"questions": ["where is add?", "where is mul?"]})]
    model_server.add("answer", "where is add?", [complete("add: calc/basic.py")])
    model_server.add("answer", "where is mul?", [complete("mul: calc/scale.py")])
    model_server.reply_dict["combine"] = [
        complete("add is in calc/basic.py and mul in calc/scale.py")]
    assert _run(calc_repo, out_dir, "where are add and mul?") == 0
    out = capfd.readouterr().out
    assert "questions: 2" in out
    assert "add is in calc/basic.py and mul in calc/scale.py" in out
    assert branch_list(calc_repo) == ["main"]
    assert git(calc_repo, "status", "--porcelain") == ""
    stage_list = model_server.stage_list()
    assert stage_list.count("answer") == 2 and stage_list.count("combine") == 1
    assert "report" not in stage_list
    combine_prompt = model_server.prompt_list("combine")[0]
    assert "answered: add: calc/basic.py" in combine_prompt
    assert [r["tool_args"][0] for r in engine_run_list(host_engine)] == \
        ["run-tests", "run", "answer", "answer", "answer"]


# == a request handed back, and the token limit of one run =========
def _record(out_dir, calc_repo) -> dict:
    """Return the newest record the commands of a test left."""
    return cli.runrecord.read_list(
        cli.sandbox_mod.record_dir(str(out_dir), calc_repo.name))[-1]


def test_a_request_handed_back_starts_no_agent_and_changes_nothing(
        calc_repo, out_dir, model_server, host_engine, capfd):
    model_server.reply_dict["dispatch"] = [
        complete_json({"unclear": ["which rounding rule does mul follow?"]})]
    assert _run(calc_repo, out_dir, "round what mul returns by the house rule") == 1
    out = capfd.readouterr().out
    assert "unclear: 1 (no agent was started and nothing was changed)" in out
    assert "  1. which rounding rule does mul follow?" in out
    assert out.rstrip().splitlines()[-1].startswith("to go on, run it again with")
    assert branch_list(calc_repo) == ["main"]
    assert git(calc_repo, "status", "--porcelain") == ""
    assert model_server.stage_list().count("edit") == 0
    assert [r["tool_args"][0] for r in engine_run_list(host_engine)] == ["run-tests", "run"]
    record_dict = _record(out_dir, calc_repo)
    assert record_dict["exit_status"] == 1 and record_dict["resume"] == []
    assert record_dict["result"]["unclear"] == ["which rounding rule does mul follow?"]


def test_the_token_limit_is_held_against_the_agents_running_at_once_together(
        calc_repo, out_dir, model_server, host_engine, capfd, monkeypatch):
    """Every call the scripted model answers counts 2 tokens."""
    # What the reading run uses on its own, read off a run that starts no agent
    model_server.reply_dict["dispatch"] = [complete_json({"unclear": ["which one?"]})]
    assert _run(calc_repo, out_dir, "add sub and div") == 1
    read_token = _record(out_dir, calc_repo)["token"]["total"]
    assert read_token > 0

    # Two calls' worth is left for the two agents between them
    cli.tokencount.reset()
    model_server.call_list.clear()
    model_server.rule_list.clear()
    _split_into_two_tasks(model_server)
    monkeypatch.setenv("MAX_TOKEN_PER_RUN", str(read_token + 4))
    importlib.reload(config)
    assert _run(calc_repo, out_dir, "add sub and div") == 1
    record_dict = _record(out_dir, calc_repo)
    container_token = sum(part["total"] for part in record_dict["token"]["parts"])
    # The limit, and at most the one call that was going when it was reached
    assert read_token + 4 <= container_token <= read_token + 6
    assert "MAX_TOKEN_PER_RUN" in capfd.readouterr().err
    assert git(calc_repo, "status", "--porcelain") == ""

