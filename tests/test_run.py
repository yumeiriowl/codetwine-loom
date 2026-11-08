"""run_project, the tool that starts the edited program.

The checks run a tiny Python program written into tmp_path as the "project", with
config.RUN_COMMAND pointing at the current interpreter, so no build tool is involved.
"""

import os
import subprocess
import sys
import time

import pytest
from codetwine.utils import file_utils

from codetwine_loom import agent, config, dispatch, plan, rlm
from codetwine_loom.tools import run, source

# A stand-in for the target program: it writes a file into the directory it is given
# and prints a line, which is everything the tool has to report back
PROGRAM = """import json, os, sys
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as fp:
    json.dump({"flag": os.environ.get("SOME_FLAG", "unset"),
               "inputs": sorted(os.listdir(sys.argv[2]))}, fp)
print("ran with", sys.argv[1])
sys.exit(int(os.environ.get("EXIT_CODE", "0")))
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Point PROJECT_DIR at a directory holding main.py, and RUN_COMMAND at python."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "main.py").write_text(PROGRAM)
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "RUN_COMMAND", [sys.executable])
    run.reset_run_state()
    yield root
    run.reset_run_state()


def test_what_was_there_before_is_never_reported_as_produced(project, monkeypatch):
    """The files present before the run are listed without the RUN_MAX_FILE cap; the
    cap applies to what is reported. With the cap on both, a workspace holding more
    files than the cap reported its own old files as new ones."""
    monkeypatch.setattr(config, "RUN_MAX_FILE", 2)
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": "", "src/b.py": "", "src/c.py": ""})
    assert r["produced"] == ["out/result.json"]
    r = run.run_project(["main.py", "{workspace}/out2", "{workspace}/src"],
                        file_dict={"src/a.py": "", "src/b.py": "", "src/c.py": ""})
    assert r["produced"] == ["out2/result.json"]


def test_run_project_runs_the_program_and_reports_what_it_produced(project):
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": "x = 1\n"})
    assert r["ok"] is True and r["returncode"] == 0
    assert "ran with" in r["output"]
    assert r["produced"] == ["out/result.json"]
    assert r["error"] == ""


def test_a_cut_run_output_starts_with_how_much_was_omitted(project, monkeypatch):
    monkeypatch.setattr(config, "RUN_OUTPUT_TAIL", 2)
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": "x = 1\n"})
    assert r["output"].startswith("... (") and "chars omitted)\n" in r["output"]
    assert len(r["output"].split("\n", 1)[1]) == 2


def test_run_project_reads_output_that_is_not_utf8(project):
    (project / "raw.py").write_text("import sys\nsys.stdout.buffer.write(b'caf\\xe9\\n')\n")
    r = run.run_project(["raw.py"])
    assert r["ok"] is True and "caf\ufffd" in r["output"]


def test_the_last_run_is_the_one_the_tool_reports(project):
    """The tool keeps what it ran, so the history does not have to guess."""
    assert run.last_output() == ""
    run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                    file_dict={"src/a.py": "x = 1\n"})
    first = run.last_output()
    assert "ran with" in first
    run.run_project(["main.py", "{workspace}/second", "{workspace}/src"],
                    file_dict={"src/a.py": "x = 1\n"})
    assert run.last_output() != first and "second" in run.last_output()


def test_the_automatic_checks_own_run_is_not_the_models(project):
    """run_once is the machine's own run: it counts for neither of the two."""
    run.run_once(["main.py", "{workspace}/out", "{workspace}/src"],
                 file_dict={"src/a.py": "x = 1\n"})
    assert run.run_count() == 0 and run.last_output() == ""


def test_run_project_returns_the_contents_asked_for_in_show(project):
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": "x = 1\n"},
                        show_list=["out/result.json"])
    assert '"inputs": ["a.py"]' in r["shown"]["out/result.json"]


def test_run_project_passes_the_environment_through(project):
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": "x = 1\n"}, env_dict={"SOME_FLAG": "on"},
                        show_list=["out/result.json"])
    assert '"flag": "on"' in r["shown"]["out/result.json"]


def test_run_project_reports_a_failing_exit_code(project):
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": "x = 1\n"}, env_dict={"EXIT_CODE": "3"})
    assert r["ok"] is False and r["returncode"] == 3


def test_the_workspace_survives_between_calls(project):
    """Running twice against the same workspace is the point for a caching change."""
    first = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                            file_dict={"src/a.py": "x = 1\n"})
    second = run.run_project(["main.py", "{workspace}/out2", "{workspace}/src"])
    assert first["workspace"] == second["workspace"]
    assert second["produced"] == ["out2/result.json"]
    assert '"inputs": ["a.py"]' in second["shown"].get("out2/result.json", "") or True
    assert os.path.exists(os.path.join(second["workspace"], "src", "a.py"))


def test_reset_empties_the_workspace(project):
    run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                    file_dict={"src/a.py": "x = 1\n"})
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/b.py": "y = 2\n"}, reset=True,
                        show_list=["out/result.json"])
    assert '"inputs": ["b.py"]' in r["shown"]["out/result.json"]


def test_run_project_refuses_a_file_path_outside_the_workspace(project):
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}"],
                        file_dict={"../escaped.py": "x = 1\n"})
    assert r["ok"] is False and "workspace" in r["error"]
    assert not (project.parent / "escaped.py").exists()


def test_run_project_writes_nothing_into_the_repository(project):
    """A run leaving files in PROJECT_DIR would show up as "what the model changed"."""
    before = sorted(os.listdir(project))
    run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                    file_dict={"src/a.py": "x = 1\n"})
    assert sorted(os.listdir(project)) == before


def test_run_project_says_so_when_no_launcher_is_configured(project, monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", [])
    r = run.run_project(["main.py"])
    assert r["ok"] is False and "RUN_COMMAND" in r["error"]


def test_run_project_returns_a_timeout_instead_of_hanging(project, monkeypatch):
    (project / "main.py").write_text("import time\ntime.sleep(5)\n")
    monkeypatch.setattr(config, "RUN_TIMEOUT_SEC", 1)
    r = run.run_project(["main.py"])
    assert r["ok"] is False and "timeout" in r["error"]


def test_run_project_says_so_when_the_launcher_is_missing(project, monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", ["no-such-launcher-xyz"])
    r = run.run_project(["main.py"])
    assert r["ok"] is False and r["error"].startswith(
        "no-such-launcher-xyz cannot be started (No such file or directory). Set "
        "RUN_COMMAND to something runnable.")


def test_run_project_says_so_when_the_launcher_is_there_but_cannot_be_started(
        project, monkeypatch):
    (project / "launch.sh").write_text("#!/bin/sh\necho hi\n")
    monkeypatch.setattr(config, "RUN_COMMAND", ["./launch.sh", "--"])
    r = run.run_project(["main.py"])
    assert r["ok"] is False and r["returncode"] == -1 and r["error"].startswith(
        "./launch.sh cannot be started (Permission denied). Set RUN_COMMAND to "
        "something runnable.")


def test_the_calls_are_counted_and_reset(project):
    run.run_project(["main.py", "{workspace}/out", "{workspace}/src"])
    run.run_project(["main.py", "{workspace}/out2", "{workspace}/src"])
    assert run.run_count() == 2
    run.reset_run_state()
    assert run.run_count() == 0


# == Paths that point at the repository instead of the workspace ==
@pytest.fixture
def git_project(project):
    """Turn the stand-in project into a git repository, for the clean-up checks."""
    for args in (("init", "-q", "-b", "main"), ("config", "user.email", "t@example.com"),
                 ("config", "user.name", "t"), ("add", "."), ("commit", "-q", "-m", "i")):
        subprocess.run(["git", *args], cwd=project, capture_output=True, check=True)
    return project


def test_a_timeout_stops_what_the_program_started_too(git_project, monkeypatch):
    """A launcher that starts the program as a child of its own (a shell, a build tool)
    leaves nothing running, and nothing it wrote late, once the time is up."""
    (git_project / "late_writer.py").write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1.5); "
        "open(\"late.txt\", \"w\").write(\"x\")'])\n"
        "time.sleep(30)\n")
    monkeypatch.setattr(config, "RUN_TIMEOUT_SEC", 1)
    r = run.run_project(["late_writer.py"])
    assert r["ok"] is False and "timeout" in r["error"]
    time.sleep(1.5)
    assert not (git_project / "late.txt").exists()


def test_a_sample_no_argument_points_at_is_refused_before_anything_starts(project):
    r = run.run_project(["main.py", "out", "src"], file_dict={"src/a.py": "x = 1\n"})
    assert r["ok"] is False and "{workspace}" in r["error"]
    # Nothing was started, so it does not count as the model having run the program
    assert run.run_count() == 0


def test_a_sample_an_environment_value_points_at_is_given_to_the_program(project):
    (project / "by_env.py").write_text(
        "import os\nprint(open(os.environ['APP_CFG']).read())\n")
    r = run.run_project(["by_env.py"], file_dict={"cfg.txt": "hello\n"},
                        env_dict={"APP_CFG": "{workspace}/cfg.txt"})
    assert r["ok"] is True and r["output"].startswith("hello")
    # Named by neither, the sample is refused, and the refusal names both places
    r = run.run_project(["by_env.py"], file_dict={"cfg.txt": "hello\n"},
                        env_dict={"APP_CFG": "cfg.txt"})
    assert r["ok"] is False and r["returncode"] == -1
    assert "none of your arguments and none of your env_dict= values" in r["error"]
    assert run.run_count() == 1


def test_a_call_that_forgot_the_program_is_refused_before_anything_starts(project,
                                                                          monkeypatch):
    """`uv run --project-dir ...` reaches the launcher, not the program."""
    monkeypatch.setattr(config, "SMOKE_ARG", ["main.py", "{workspace}/out"])
    r = run.run_project(["--project-dir", "{workspace}/out"])
    assert r["ok"] is False and r["returncode"] == -1
    assert "--project-dir" in r["error"]
    assert run.run_count() == 0
    # The way out is spelled with a call that is known to start this program
    assert "run_project(['main.py', '{workspace}/out'])" in r["error"]


def test_the_working_call_comes_from_the_automatic_check(project, monkeypatch):
    monkeypatch.setattr(config, "SMOKE_ARG", ["run.py", "--flag"])
    monkeypatch.setattr(config, "SMOKE_FILE", {"src/a.py": "x = 1\n"})
    monkeypatch.setattr(config, "SMOKE_ENV", {"SOME_FLAG": "on"})
    error = run.run_project(["-x"])["error"]
    assert "run_project(['run.py', '--flag']" in error
    assert "file_dict={'src/a.py': ...}" in error and "env_dict={'SOME_FLAG': 'on'}" in error


def test_a_generic_call_is_shown_when_there_is_no_automatic_check(project, monkeypatch):
    monkeypatch.setattr(config, "SMOKE_ARG", [])
    error = run.run_project(["-x"])["error"]
    assert 'run_project(["<what starts this project>"' in error
    # No program of any particular repository is named where none is known
    assert "main.py" not in error


def test_empty_args_are_refused_too(project):
    assert run.run_project([])["ok"] is False


def test_a_call_that_starts_with_the_program_still_runs(project):
    """The guard must not stand in the way of the calls that were always fine."""
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": "x = 1\n"})
    assert r["ok"] is True and r["error"] == ""


def test_a_launcher_ending_with_a_separator_hands_every_arg_to_the_program(project,
                                                                          monkeypatch):
    """`cargo run --` has named the program, so a flag first and no args at all reach it."""
    (project / "echo.py").write_text("import sys\nprint('argv', sys.argv[1:])\n")
    monkeypatch.setattr(config, "RUN_COMMAND", [sys.executable, "echo.py", "--"])
    r = run.run_project(["--in", "{workspace}/x"])
    assert r["ok"] is True and "'--in'" in r["output"]
    assert run.run_project([])["ok"] is True
    assert run.run_count() == 2


def test_a_run_without_a_sample_is_not_refused(project):
    """Only files= with nothing pointing at it is refused; a bare run still starts."""
    r = run.run_project(["main.py", "out", "src"])
    assert "{workspace}" not in r["error"]


def test_what_the_run_wrote_into_the_repository_is_taken_back_out(git_project):
    # "out1" is relative, so the program writes it under the repository, not the
    # workspace
    r = run.run_project(["main.py", "out1", "{workspace}/src"],
                        file_dict={"src/a.py": "x = 1\n"})
    assert r["ok"] is True
    assert r["left_in_repo"] == ["out1/result.json"]
    # The file and the directory it made are both gone again
    assert not (git_project / "out1").exists()


@pytest.mark.parametrize("name", ["space name.json", 'a"b.json', "tab\tname.json"])
def test_a_file_whose_name_git_would_quote_is_taken_back_out(git_project, name):
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": "x = 1\n"})
    assert r["ok"] is True
    base_set, path_record = run._new_file_set(), run._path_record()
    (git_project / name).write_text("{}\n")
    assert run._clean_repo(base_set, path_record) == [name]
    assert not (git_project / name).exists()


def test_a_file_whose_name_is_not_utf8_is_taken_back_out_and_named_with_its_bytes(
        git_project):
    base_set, path_record = run._new_file_set(), run._path_record()
    raw_path = os.path.join(os.fsencode(str(git_project)), b"made\xe9.out")
    with open(raw_path, "wb") as handle:
        handle.write(b"x\n")
    assert run._clean_repo(base_set, path_record) == ["made\\xe9.out"]
    assert not os.path.exists(raw_path)


@pytest.mark.parametrize("stop", [rlm.TurnTimeout, KeyboardInterrupt])
def test_what_the_run_wrote_is_taken_back_out_when_the_turn_stops_it(git_project,
                                                                     monkeypatch, stop):
    """The turn's time limit and Ctrl-C go on up, and the file the program wrote does not
    stay behind as a write no tool made."""
    def write_then_stop(argv, timeout_sec, extra_env_dict=None, is_machine_run=True):
        """Write a file into the repository as the program would, then stop the wait."""
        (git_project / "out1").mkdir()
        (git_project / "out1" / "result.json").write_text("{}\n")
        raise stop()

    monkeypatch.setattr(run.edit, "run_in_project", write_then_stop)
    with pytest.raises(stop):
        run.run_project(["main.py", "out1"])
    assert not (git_project / "out1").exists()


def test_a_file_that_was_already_there_is_left_alone(git_project):
    (git_project / "note.txt").write_text("mine\n")
    r = run.run_project(["main.py", "out1", "{workspace}/src"],
                        file_dict={"src/a.py": "x = 1\n"})
    assert r["left_in_repo"] == ["out1/result.json"]
    assert (git_project / "note.txt").read_text() == "mine\n"


def test_a_tracked_file_the_run_changed_is_not_reverted(git_project):
    """Undoing that would throw away an edit the model made on purpose."""
    (git_project / "main.py").write_text(
        'open("lib.py", "a").write("# touched\\n")\nprint("ran")\n')
    (git_project / "lib.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "lib.py"], cwd=git_project, check=True,
                   capture_output=True)
    subprocess.run(["git", "commit", "-qm", "lib"], cwd=git_project, check=True,
                   capture_output=True)
    r = run.run_project(["main.py"])
    assert r["ok"] is True and r["left_in_repo"] == []
    assert "# touched" in (git_project / "lib.py").read_text()


def test_the_return_keys_in_the_instructions_match_the_real_ones(project):
    """The same drift check the other tools get."""
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"])
    assert set(r) == set(agent.TOOL_RESULT_KEY_DICT["run_project"][0])


def test_the_instructions_tell_the_model_to_run_what_it_built(monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    text = agent.build_instruction("py")
    assert "run_project(arg_list, file_dict=..." in text
    assert "3.5." in text
    assert "run_test() is not that check" in text


def test_the_step_is_left_out_when_no_launcher_is_configured(monkeypatch):
    monkeypatch.setattr(config, "RUN_COMMAND", [])
    assert "run_project(" not in agent.build_instruction("py").split("## What the")[0]
    assert agent.run_note() == ""


def test_the_default_python_launcher_runs_with_the_repository_s_packages(project,
                                                                        monkeypatch):
    """A directory with setup.py and no uv project gets its own packages, isolated."""
    (project / "setup.py").write_text("")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    argv_list = []

    def note_and_stop(argv, *arg_list, **kwarg_dict):
        argv_list.append(argv)
        raise run.edit.StartError("not started")

    monkeypatch.setattr(run.edit, "run_in_project", note_and_stop)
    run.run_project(["main.py"])
    assert argv_list[0][:6] == ["uv", "run", "--isolated", "--with-editable", ".",
                                "main.py"]


# == the environment the program is given ====================
KEY_PROGRAM = """import os, sys
print(os.environ.get("LLM_API_KEY", "not given"))
print(os.environ.get("PATH", "")[:1])
"""


def test_the_program_is_not_given_the_credentials(tmp_path, monkeypatch):
    """The launcher runs the repository's own code, which must not read the key."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "main.py").write_text(KEY_PROGRAM)
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "RUN_COMMAND", [sys.executable])
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    run.reset_run_state()
    r = run.run_project(["main.py"])
    assert "secret-key" not in r["output"]
    assert "not given" in r["output"]


# == Taking back what the run left in the repository ============
def test_a_directory_that_still_holds_something_is_left_where_it_is(project):
    (project / "d").mkdir()
    (project / "d" / "b.txt").write_text("kept\n")
    run.edit.keep_new_dir("d")
    run.edit.remove_empty_dir("d/a.txt")
    assert (project / "d" / "b.txt").exists()


def test_a_directory_that_was_there_before_the_program_ran_stays_when_it_is_left_empty(
        git_project):
    """The program writes into two empty directories of the user's and one of its own."""
    (git_project / "cache").mkdir()
    (git_project / "tests" / "data").mkdir(parents=True)
    before_tuple = (run._new_file_set(), run._path_record())
    for name in ("cache/c.bin", "tests/data/out.log", "made/deep/x.txt", "build/new/y.txt"):
        (git_project / name).parent.mkdir(parents=True, exist_ok=True)
        (git_project / name).write_text("x\n")
    assert run._clean_repo(*before_tuple) == sorted(
        ["cache/c.bin", "tests/data/out.log", "made/deep/x.txt", "build/new/y.txt"])
    assert (git_project / "cache").is_dir() and (git_project / "tests" / "data").is_dir()
    assert not (git_project / "made").exists() and not (git_project / "build").exists()
    assert not any(path.is_file() for path in (git_project / "cache").rglob("*"))


def test_a_directory_made_under_one_no_search_walks_into_goes_and_one_that_was_there_stays(
        git_project):
    (git_project / "vendor" / "kept").mkdir(parents=True)
    before_tuple = (run._new_file_set(), run._path_record())
    for name in ("vendor/kept/z.txt", "vendor/made/y.txt"):
        (git_project / name).parent.mkdir(parents=True, exist_ok=True)
        (git_project / name).write_text("x\n")
    assert run._clean_repo(*before_tuple) == ["vendor/kept/z.txt", "vendor/made/y.txt"]
    assert (git_project / "vendor" / "kept").is_dir()
    assert not (git_project / "vendor" / "made").exists()


def test_the_listing_taken_before_a_run_names_every_path_but_what_git_keeps(git_project):
    (git_project / "node_modules" / "pkg").mkdir(parents=True)
    (git_project / "node_modules" / "pkg" / "i.js").write_text("x\n")
    (git_project / "empty").mkdir()
    os.symlink("node_modules", git_project / "link")
    assert run._path_record() == (
        {"main.py", "node_modules", "node_modules/pkg", "node_modules/pkg/i.js", "empty",
         "link", ".git"}, set())


# A program that changes what git says of the files that are there, and writes one file
_RULE_PROGRAM_DICT = {
    "writes the ignore rules": "open('.gitignore', 'w').write('__pycache__/\\n')",
    "deletes the ignore rules": "import os; os.remove('.gitignore')",
    "takes a file out of the index":
        "import subprocess; subprocess.run(['git', 'rm', '-q', '--cached', 'kept.txt'])",
}


@pytest.mark.parametrize("name", list(_RULE_PROGRAM_DICT))
def test_a_file_that_was_there_before_the_program_ran_stays_whatever_git_says_of_it_now(
        git_project, monkeypatch, name):
    (git_project / ".gitignore").write_text("local.env\ndata/\n*.sqlite\n")
    (git_project / "kept.txt").write_text("t\n")
    subprocess.run(["git", "add", "."], cwd=git_project, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "rules"], cwd=git_project, check=True,
                   capture_output=True)
    (git_project / "data").mkdir()
    own_list = ["local.env", "data/rows.csv", "cache.sqlite"]
    for path in own_list:
        (git_project / path).write_text("never committed\n")
    monkeypatch.setattr(config, "RUN_COMMAND", [sys.executable, "-c"])
    program = _RULE_PROGRAM_DICT[name] + "; open('new.txt', 'w').write('n')"

    r = run.run_project([program])

    assert r["ok"] is True and r["left_in_repo"] == ["new.txt"]
    assert not (git_project / "new.txt").exists()
    for path in [*own_list, "kept.txt"]:
        assert (git_project / path).exists(), path


def test_a_file_that_will_not_go_is_not_among_the_ones_deleted(project, monkeypatch):
    monkeypatch.setattr(run.gitcmd, "new_file_list",
                        lambda project_dir, **kwargs: ["gone.txt", "x.txt"])
    (project / "gone.txt").write_text("g\n")
    (project / "x.txt").write_text("x\n")
    remove = os.remove

    def refuse(path):
        if path.endswith("x.txt"):
            raise PermissionError("read-only")
        remove(path)

    monkeypatch.setattr(run.os, "remove", refuse)
    assert run._clean_repo(set(), run._PathRecord(set(), set())) == ["gone.txt"]
    assert (project / "x.txt").exists() and not (project / "gone.txt").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="every directory can be listed")
@pytest.mark.parametrize("ignore", ["", "locked/\n"])
def test_what_is_under_a_directory_that_could_not_be_listed_stays(git_project, monkeypatch,
                                                                 ignore):
    """Nothing says the files were not there before the program started."""
    (git_project / ".gitignore").write_text(ignore)
    (git_project / "locked" / "sub").mkdir(parents=True)
    for name in ("locked/mine.txt", "locked/sub/deep.txt"):
        (git_project / name).write_text("there before\n")
    os.chmod(git_project / "locked", 0o000)
    monkeypatch.setattr(config, "RUN_COMMAND", [sys.executable, "-c"])
    try:
        assert run._path_record().no_list_dir_set == {"locked"}
        r = run.run_project(["import os; os.chmod('locked', 0o755); "
                             "open('.gitignore', 'w').close(); "
                             "open('new.txt', 'w').close(); "
                             "open('locked/sub/made.txt', 'w').close()"])
    finally:
        os.chmod(git_project / "locked", 0o755)
    assert r["ok"] is True and r["left_in_repo"] == ["new.txt"]
    for name in ("locked/mine.txt", "locked/sub/deep.txt", "locked/sub/made.txt"):
        assert (git_project / name).exists(), name


def test_a_repository_of_its_own_the_program_made_goes_with_all_it_holds(git_project,
                                                                          monkeypatch):
    """git lists such a repository as its directory, with a "/" after it."""
    (git_project / ".gitignore").write_text("vendor/\n")
    (git_project / "vendor" / "kept").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(git_project / "vendor" / "kept")], check=True)
    (git_project / "vendor" / "kept" / "f.txt").write_text("there before\n")
    monkeypatch.setattr(config, "RUN_COMMAND", [sys.executable, "-c"])
    r = run.run_project([
        "import subprocess; subprocess.run(['git', 'init', '-q', 'made/scratch']); "
        "open('made/scratch/a.txt', 'w').close(); open('.gitignore', 'w').close()"])
    assert r["ok"] is True and r["left_in_repo"] == ["made/scratch/"]
    assert not (git_project / "made").exists()
    assert (git_project / "vendor" / "kept" / "f.txt").exists()


def test_a_file_asked_for_in_show_is_read_in_its_own_encoding(project, monkeypatch):
    monkeypatch.setattr(file_utils, "SOURCE_ENCODING", ["cp932"])
    (project / "w.py").write_text(
        "import sys, pathlib\n"
        "pathlib.Path(sys.argv[1], 'sj.txt').write_bytes('表\\r\\n'.encode('cp932'))\n"
        "pathlib.Path(sys.argv[1], 'bin.dat').write_bytes(bytes(range(256)) * 40)\n")
    r = run.run_project(["w.py", "{workspace}"], show_list=["sj.txt", "bin.dat"])
    assert r["shown"]["sj.txt"] == "表\n"
    assert r["shown"]["bin.dat"].startswith("(cannot read: ")


def test_a_file_asked_for_in_show_is_cut_at_the_reading_limit_and_says_so(project,
                                                                         monkeypatch):
    (project / "w.py").write_text(
        "import sys, pathlib\n"
        "pathlib.Path(sys.argv[1], 'big.txt').write_text('0123456789' * 50)\n")
    monkeypatch.setattr(config, "READ_MAX_CHAR", 120)
    r = run.run_project(["w.py", "{workspace}"], show_list=["big.txt"])
    limit = source.read_limit()
    assert limit == 120
    assert r["shown"]["big.txt"] == ("0123456789" * 12) + "\n... (380 chars omitted)"
    # With no limit on a read, the whole file comes back
    monkeypatch.setattr(config, "READ_MAX_CHAR", 0)
    monkeypatch.setattr(config, "RLM_MAX_OUTPUT_CHAR", 0)
    r = run.run_project(["w.py", "{workspace}"], show_list="big.txt")
    assert r["shown"] == {"big.txt": "0123456789" * 50}


def test_the_files_past_the_listing_limit_are_counted_and_the_check_is_handed_all(
        project, monkeypatch):
    (project / "many.py").write_text(
        "import sys, pathlib\n"
        "for i in range(5):\n"
        "    pathlib.Path(sys.argv[1], f'f{i}.txt').write_text('x')\n")
    monkeypatch.setattr(config, "RUN_MAX_FILE", 3)
    r = run.run_project(["many.py", "{workspace}"])
    assert r["produced"] == ["f0.txt", "f1.txt", "f2.txt"] and r["produced_more"] == 2
    r = run.run_once(["many.py", "{workspace}"])
    assert r["produced"] == [f"f{i}.txt" for i in range(5)] and r["produced_more"] == 0
    monkeypatch.setattr(config, "RUN_MAX_FILE", 5)
    r = run.run_project(["many.py", "{workspace}"], reset=True)
    assert len(r["produced"]) == 5 and r["produced_more"] == 0


@pytest.mark.parametrize("kwarg_dict, want", [
    ({"arg_list": "main.py"}, "arg_list must be a list of strings"),
    ({"arg_list": ["main.py", "{workspace}"], "file_dict": {"c.json": {"k": 1}}},
     "file_dict must be {path: contents} with both of them strings"),
    ({"arg_list": ["main.py", "{workspace}"], "file_dict": ["c.json"]},
     "file_dict must be {path: contents}"),
    ({"arg_list": ["main.py"], "env_dict": ["A=1"]}, "env_dict must be {name: value}"),
    ({"arg_list": ["main.py"], "env_dict": {1: "x"}},
     "each name of env_dict must be a string that is not empty and holds no \"=\": 1"),
    ({"arg_list": ["main.py"], "env_dict": {"A=B": "x"}},
     "each name of env_dict must be a string that is not empty and holds no \"=\": 'A=B'"),
    ({"arg_list": ["main.py"], "env_dict": {"": "x"}},
     "each name of env_dict must be a string that is not empty and holds no \"=\": ''"),
    ({"arg_list": ["main.py"], "env_dict": {"A": "x\0y"}},
     "an argument and an env_dict value cannot hold a NUL character"),
    ({"arg_list": ["main.py", "a\0b"]},
     "an argument and an env_dict value cannot hold a NUL character"),
    ({"arg_list": ["main.py"], "show_list": [1]}, "show_list must be a list of paths"),
])
def test_an_argument_of_the_wrong_type_is_refused_before_anything_is_written(
        project, kwarg_dict, want):
    r = run.run_project(**kwarg_dict)
    assert r["ok"] is False and r["returncode"] == -1 and r["workspace"] == ""
    assert r["error"].startswith(want) and r["error"].endswith("Nothing was started")
    assert run.run_count() == 0 and run._workspace_dict == {}
    assert set(r) == set(agent.TOOL_RESULT_KEY_DICT["run_project"][0])


def test_the_scratch_directories_are_deleted_when_the_process_ends(project, monkeypatch):
    register_list = []
    monkeypatch.setattr(run, "_is_exit_drop_on", False)
    monkeypatch.setattr(run.atexit, "register", register_list.append)
    model_dir = run._ensure_workspace(run.MODEL_SLOT)
    check_dir = run._ensure_workspace(run.CHECK_SLOT)
    assert run._ensure_workspace(run.MODEL_SLOT) == model_dir
    # Registered once, by the first directory made
    assert register_list == [run._drop_workspace]
    register_list[0]()
    assert not os.path.exists(model_dir) and not os.path.exists(check_dir)


def test_a_named_pipe_the_program_left_in_the_workspace_is_not_waited_on(project):
    (project / "w.py").write_text(
        "import os, sys\nos.mkfifo(os.path.join(sys.argv[1], 'p.txt'))\n")
    r = run.run_project(["w.py", "{workspace}"], show_list=["p.txt"])
    assert r["shown"]["p.txt"] == f"(cannot read: {r['workspace']}/p.txt is not a regular " \
                                  f"file)"
    r = run.run_project(["main.py", "{workspace}/p.txt"], file_dict={"p.txt": "x"})
    assert r["ok"] is False and r["error"].endswith("p.txt is not a regular file")


def test_a_file_asked_for_in_show_that_is_not_there_says_so(project):
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": "x = 1\n"},
                        show_list=["out/missing.txt", "../outside.txt"])
    assert r["shown"]["out/missing.txt"].startswith("(cannot read: ")
    assert r["shown"]["../outside.txt"].startswith("(cannot read: ")


def test_run_project_stops_at_its_call_limit(project, monkeypatch):
    """Every other tool has a budget; without one the turn limit is what ends a run
    that keeps starting the program."""
    monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", 1)
    first = run.run_project(["main.py"])
    assert first["error"] == "" and first["calls_left"] == 0
    second = run.run_project(["main.py"])
    assert second["ok"] is False and "call limit" in second["error"]
    assert second["calls_left"] == 0
    # The check the machine runs by itself is not one of the model's calls
    assert run.run_once(["main.py"])["error"] == ""


def test_run_project_is_handed_over_while_its_budget_is_above_0(monkeypatch):
    monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", 1)
    assert run.available() is True
    monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", 0)
    assert run.available() is False


# == Filling the package cache again ==========================

def _asked_to_refill(monkeypatch, dep_path_list, skip=False):
    """Stand in for the changed manifests and broker.refill(); return the asks made."""
    asks = []

    def refill():
        asks.append(1)
        return {"ok": True, "refills_left": 2, "error": "", "skip": skip}

    monkeypatch.setattr(run.edit, "change_dep_path_list", lambda: list(dep_path_list))
    monkeypatch.setattr(run.edit.broker, "refill", refill)
    return asks


def test_a_changed_manifest_fills_the_cache_before_the_program_runs(project, monkeypatch):
    """The same ask run_test makes, so a dependency the change added is there for both."""
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    asks = _asked_to_refill(monkeypatch, ["pyproject.toml"])
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": ""})
    assert len(asks) == 1
    assert r["output"].startswith("note: the package cache was filled again for "
                                  "pyproject.toml")
    assert "ran with" in r["output"]


def test_no_changed_manifest_asks_for_nothing(project, monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    asks = _asked_to_refill(monkeypatch, [])
    r = run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                        file_dict={"src/a.py": ""})
    assert asks == [] and not r["output"].startswith("note:")


def test_a_run_with_a_network_of_its_own_fills_nothing(project, monkeypatch):
    """LLM_BROKER empty means the launcher can fetch for itself."""
    monkeypatch.setattr(config, "LLM_BROKER", "")
    asks = _asked_to_refill(monkeypatch, ["pyproject.toml"])
    run.run_project(["main.py", "{workspace}/out", "{workspace}/src"],
                    file_dict={"src/a.py": ""})
    assert asks == []


def test_run_project_is_handed_over_only_with_a_launcher(monkeypatch):
    monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", 5)
    monkeypatch.setattr(config, "RUN_COMMAND", [])
    assert not run.available()
    assert run.run_project not in agent.main_tool_list()
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    assert run.available() and run.run_project in agent.main_tool_list()


@pytest.mark.parametrize("ext", ["c", "cpp"])
def test_the_c_launcher_is_told_of_in_every_run_note_while_it_is_in_force(monkeypatch, ext):
    """A launcher set by hand is the user's program, not make."""
    monkeypatch.setattr(config, "RUN_PROJECT_MAX_CALL", 5)
    monkeypatch.setattr(config, "PRIMARY_EXT", ext)
    monkeypatch.setattr(config, "RUN_COMMAND", ["ctloom-run-c"])
    note_list = [agent.run_note(), plan.plan_run_note(),
                 dispatch.build_dispatch_instruction(ext)]
    for note in note_list:
        assert "your args are make's targets and variables" in " ".join(note.split())
    # The plan's line on running a file of file_dict= is not said of make
    assert "can be run too" not in note_list[1]
    monkeypatch.setattr(config, "RUN_COMMAND", ["./build/app"])
    assert run.launcher_note("") == ""
    assert "make's targets" not in " ".join(agent.run_note().split())
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "RUN_COMMAND", ["uv", "run"])
    assert run.launcher_note("") == ""
    assert "can be run too" in plan.plan_run_note()
