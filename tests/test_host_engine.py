"""The engine the integration tests run the container's scripts with.

Only the reading of the engine's command line and the writing of the container's paths
as this machine's are checked here; what the scripts do is checked where they run.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "integration"))

import host_engine                                              # noqa: E402

_ARGV = ["--rm", "-v", "/tmp/repo:/src:ro", "-v", "/tmp/o:/out", "-v", "/tmp/s/work:/work",
         "-v", "/tmp/s/tmp:/tmp", "-v", "/tmp/c:/cache", "-v", "/tmp/k:/run/ctloom",
         "--network=none", "--read-only", "--pids-limit", "512", "--memory", "2g",
         "--name", "ctloom-x", "--userns=keep-id", "-e", "CTLOOM_SRC=/src",
         "-e", "CTLOOM_WORK=/work/repo", "-e", "TEST_COMMAND", "-e", "NOT_SET_HERE",
         "-e", "LLM_BROKER=unix:///run/ctloom/llm.sock",
         "--entrypoint", "/usr/local/bin/ctloom-sandbox", "an-image",
         "edit", "--project", "/work/repo", "--output-dir", "/out", "--request", "r"]


def test_the_mounts_the_environment_and_the_tool_arguments_are_read_apart():
    parsed = host_engine.parse_run_argv(_ARGV)
    assert parsed["mount_dict"]["/src"] == "/tmp/repo"
    assert parsed["mount_dict"]["/run/ctloom"] == "/tmp/k"
    assert parsed["env_dict"] == {"CTLOOM_SRC": "/src", "CTLOOM_WORK": "/work/repo",
                                  "LLM_BROKER": "unix:///run/ctloom/llm.sock"}
    assert parsed["env_name_list"] == ["TEST_COMMAND", "NOT_SET_HERE"]
    assert parsed["entrypoint"] == "/usr/local/bin/ctloom-sandbox"
    assert parsed["image"] == "an-image"
    assert parsed["arg_list"] == ["edit", "--project", "/work/repo", "--output-dir",
                                  "/out", "--request", "r"]


def test_a_path_inside_the_container_becomes_the_path_here():
    mount_dict = {"/out": "/tmp/o", "/work": "/tmp/s/work", "/run/ctloom": "/tmp/k"}
    assert host_engine.to_host("/out/repo/x", mount_dict) == "/tmp/o/repo/x"
    assert host_engine.to_host("/out", mount_dict) == "/tmp/o"
    assert host_engine.to_host("/output/x", mount_dict) == "/output/x"
    assert host_engine.to_host("unix:///run/ctloom/llm.sock", mount_dict) == \
        "unix:///tmp/k/llm.sock"
    assert host_engine.to_host("--request", mount_dict) == "--request"


def test_the_longest_mount_wins():
    mount_dict = {"/work": "/tmp/w", "/work/repo": "/tmp/r"}
    assert host_engine.to_host("/work/repo/a", mount_dict) == "/tmp/r/a"
    assert host_engine.to_host("/work/other", mount_dict) == "/tmp/w/other"


def test_the_environment_holds_only_what_the_engine_was_given(monkeypatch):
    monkeypatch.setenv("TEST_COMMAND", "run them")
    monkeypatch.delenv("NOT_SET_HERE", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "must not reach the container")
    env_dict = host_engine.child_env_dict(host_engine.parse_run_argv(_ARGV), "/bin-dir")
    assert env_dict["PATH"].startswith("/bin-dir:")
    assert env_dict["TEST_COMMAND"] == "run them"
    assert "NOT_SET_HERE" not in env_dict and "LLM_API_KEY" not in env_dict
    assert env_dict["CTLOOM_WORK"] == "/tmp/s/work/repo"
    assert env_dict["LLM_BROKER"] == "unix:///tmp/k/llm.sock"
    assert env_dict["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert env_dict["LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS"] == "True"


def test_the_engine_answers_what_is_running_and_what_to_remove_with_nothing(capsys):
    assert host_engine.main(["ps", "--quiet", "--filter", "name=ctloom-x"]) == 0
    assert host_engine.main(["rm", "--force", "ctloom-x"]) == 0
    assert capsys.readouterr().out == ""


def test_a_command_the_engine_does_not_know_is_refused():
    assert host_engine.main([]) == 2
    assert host_engine.main(["pull", "img"]) == 2


def test_every_image_is_there_with_the_mark_it_is_told(monkeypatch, capsys):
    monkeypatch.delenv("CTLOOM_HOST_ENGINE_IMAGE_FILE", raising=False)
    monkeypatch.setenv("CTLOOM_HOST_ENGINE_SOURCE", "m1")
    assert host_engine.main(["image", "inspect", "--format", "{{json .Config.Labels}}",
                             "img"]) == 0
    assert json.loads(capsys.readouterr().out) == {"io.codetwine-loom.source": "m1"}


def test_a_build_puts_the_image_in_the_file_of_images(monkeypatch, tmp_path, capsys):
    path = tmp_path / "images.json"
    monkeypatch.setenv("CTLOOM_HOST_ENGINE_IMAGE_FILE", str(path))
    inspect = ["image", "inspect", "--format", "{{json .Config.Labels}}", "img:java"]
    assert host_engine.main(inspect) == 125
    build = ["build", "--build-arg", "TOOLCHAIN=java", "--label", "io.codetwine-loom.source=m1",
             "--label", "io.codetwine-loom.toolchain=java", "--build-arg", "HTTPS_PROXY",
             "-t", "img:java", "/opt/ctl"]
    assert host_engine.main(build) == 0
    capsys.readouterr()
    assert host_engine.main(inspect) == 0
    assert json.loads(capsys.readouterr().out) == {
        "io.codetwine-loom.source": "m1", "io.codetwine-loom.toolchain": "java"}
    assert json.loads(path.read_text(encoding="utf-8"))["build_list"] == [build[1:]]


def test_an_entrypoint_the_image_does_not_have_is_refused():
    argv = ["-v", "/tmp/s/tmp:/tmp", "--entrypoint", "/usr/local/bin/other", "img"]
    assert host_engine.run(argv) == 125


def test_the_clone_and_the_temporary_files_are_moved_under_the_work_root(tmp_path,
                                                                         monkeypatch):
    monkeypatch.setenv("CTLOOM_HOST_ENGINE_WORK", str(tmp_path / "w"))
    mount_dict = {"/out": "/tmp/o", "/work": "/tmp/o/scratch-1/work",
                  "/tmp": "/tmp/o/scratch-1/tmp"}
    moved_dict = host_engine._moved_mount_dict(mount_dict)
    assert moved_dict["/out"] == "/tmp/o"
    assert moved_dict["/work"] == str(tmp_path / "w" / "scratch-1" / "work")
    assert moved_dict["/tmp"] == str(tmp_path / "w" / "scratch-1" / "tmp")
    assert (tmp_path / "w" / "scratch-1" / "tmp").is_dir()


def test_without_a_work_root_the_mounts_are_left_as_they_are(monkeypatch):
    monkeypatch.delenv("CTLOOM_HOST_ENGINE_WORK", raising=False)
    mount_dict = {"/work": "/tmp/o/scratch-1/work"}
    assert host_engine._moved_mount_dict(mount_dict) == mount_dict
