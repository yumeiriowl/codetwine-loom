"""The sandbox: the container's command line, and taking its patch in.

Nothing here starts a container. What is checked is the argument list that would be
handed to one - the mount is read-only, the image's own filesystem is not writable, the
whole configuration is forwarded - and the git side of taking a patch in, against real
repositories made in tmp_path.
"""

import ast
import contextlib
import hashlib
import json
import logging
import os
import pathlib
import random
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import types

import pytest

from conftest import fake_defs, fake_files, write_index, write_knowledge_db
from codetwine_loom import config, gitcmd, history, knowledge, paths, progress, sandbox
from codetwine_loom import integrate as integrate_mod
from codetwine_loom.languages import LANG_DICT, OFFLINE_ENV_DICT
from codetwine_loom.index import store

# What analyses a repository nothing is built for. Every test but its own replaces it,
# so make_agent_dir() never runs codetwine over the test's repository
_analyse_if_none = sandbox.analyse_if_none


@pytest.fixture(autouse=True)
def _no_first_analysis(monkeypatch):
    monkeypatch.setattr(sandbox, "analyse_if_none", lambda *a, **k: False)


# What finds the engines on this machine. Every test but its own sees none, so no test
# asks a real engine about a run
_other_engine_path_list = sandbox._other_engine_path_list


@pytest.fixture(autouse=True)
def _no_other_engine(monkeypatch):
    monkeypatch.setattr(sandbox, "_other_engine_path_list", lambda engine_path: [])

_CONFIG = pathlib.Path(sandbox.__file__).resolve().parent / "config.py"


# == What the containers keep out of the analysis =============

class _StampOnly:
    """A stand-in for an open knowledge database that only says its stamp."""

    def stamp(self):
        return "1:now"

    def close(self):
        pass


def _stamp_only(monkeypatch):
    """Let the analysis a test wrote as plain text be opened for its stamp alone."""
    monkeypatch.setattr(sandbox.knowledge.Knowledge, "open", staticmethod(lambda p: _StampOnly()))


def _analysis_on_disk(tmp_path, monkeypatch, record_list):
    """Put an analysis with this exclude list in tmp_path / out, and return that dir."""
    out_dir = tmp_path / "out"
    (out_dir / "repo").mkdir(parents=True)
    (out_dir / "repo" / sandbox.knowledge.KNOWLEDGE_FILE).write_text("")
    _stamp_only(monkeypatch)
    monkeypatch.setattr(sandbox.knowledge, "exclude_of_analysis",
                        lambda p, stamp, fallback: list(record_list) if record_list is not None
                        else list(fallback))
    monkeypatch.setattr(sandbox.knowledge, "analysis_record",
                        lambda p, stamp: {"exclude": list(record_list)}
                        if record_list is not None else {})
    return str(out_dir)


def test_a_run_naming_nothing_keeps_out_what_the_analysis_keeps_out(tmp_path, monkeypatch,
                                                                     caplog):
    caplog.set_level(logging.INFO, logger="codetwine_loom.sandbox")
    out_dir = _analysis_on_disk(tmp_path, monkeypatch, ["sample_output", "logs"])
    assert sandbox.exclude_list_of_run("/p", out_dir, "repo", None, False) == \
        ["sample_output", "logs"]
    assert "keeps out what the analysis" in caplog.text


def test_a_run_naming_the_same_names_goes_on_with_them_as_given(tmp_path, monkeypatch):
    out_dir = _analysis_on_disk(tmp_path, monkeypatch, ["logs", "sample_output"])
    assert sandbox.exclude_list_of_run("/p", out_dir, "repo", ["sample_output", "logs"],
                                       False) == ["sample_output", "logs"]


def test_a_run_naming_other_names_is_refused_with_the_build_to_run(tmp_path, monkeypatch):
    out_dir = _analysis_on_disk(tmp_path, monkeypatch, ["sample_output"])
    with pytest.raises(sandbox.CommandError) as caught:
        sandbox.exclude_list_of_run("/a  p; $(x)", out_dir, "repo", ["vendor", "a b"],
                                    False)
    text = caught.value.why
    assert "keeps out sample_output" in text and "asks to keep out vendor, a b" in text
    assert text.endswith("or make the analysis again with:")
    assert shlex.split(caught.value.command) == [
        "codetwine-loom", "build", "--project", "/a  p; $(x)", "--output-dir", out_dir,
        "--exclude", "vendor", "--exclude", "a b"]
    assert str(caught.value) == f"{text} {caught.value.command}"


@pytest.mark.parametrize("setting", ["apps/old", "apps/old/", "./apps//old"])
def test_the_setting_is_compared_in_the_form_its_entries_are_kept_in(tmp_path, monkeypatch,
                                                                    setting):
    """cli.exclude_name_list() hands every entry on as knowledge.exclude_entry() has it."""
    monkeypatch.setattr(sandbox.config, "BUILD_EXCLUDE", [setting])
    out_dir = _analysis_on_disk(tmp_path, monkeypatch, [])
    assert sandbox.exclude_list_of_run("/p", out_dir, "repo", ["apps/old"], False) == []


def test_an_analysis_that_keeps_out_nothing_refuses_a_run_that_names_something(tmp_path,
                                                                                 monkeypatch):
    out_dir = _analysis_on_disk(tmp_path, monkeypatch, [])
    with pytest.raises(RuntimeError, match="keeps out nothing"):
        sandbox.exclude_list_of_run("/p", out_dir, "repo", ["vendor"], False)


def test_the_names_of_the_setting_alone_are_no_names_given(tmp_path, monkeypatch):
    """BUILD_EXCLUDE=vendor, then build --exclude extra, then run with no --exclude."""
    monkeypatch.setattr(sandbox.config, "BUILD_EXCLUDE", ["vendor"])
    out_dir = _analysis_on_disk(tmp_path, monkeypatch, ["vendor", "extra"])
    assert sandbox.exclude_list_of_run("/p", out_dir, "repo", ["vendor"], False) == \
        ["vendor", "extra"]
    with pytest.raises(RuntimeError, match="asks to keep out vendor, other"):
        sandbox.exclude_list_of_run("/p", out_dir, "repo", ["vendor", "other"], False)


def test_a_first_analysis_keeps_out_the_names_of_the_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox.config, "BUILD_EXCLUDE", ["vendor"])
    seen = []
    monkeypatch.setattr(sandbox, "analyse_if_none",
                        lambda *a, **k: seen.append(a[3]) or False)
    assert sandbox.exclude_list_of_run("/p", str(tmp_path / "out"), "repo", ["vendor"],
                                       False) == ["vendor"]
    assert seen == [["vendor"]]


def test_without_a_record_or_an_analysis_the_run_keeps_what_it_was_given(tmp_path, monkeypatch):
    out_dir = _analysis_on_disk(tmp_path, monkeypatch, None)
    assert sandbox.exclude_list_of_run("/p", out_dir, "repo", ["vendor"], False) == ["vendor"]
    assert sandbox.exclude_list_of_run("/p", out_dir, "repo", None, False) is None
    assert sandbox.exclude_list_of_run("/p", str(tmp_path / "none"), "repo", ["v"], False) == ["v"]


def test_the_analysis_is_made_first_when_there_is_none(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(sandbox, "analyse_if_none",
                        lambda *a, **k: seen.append((a, k)) or False)
    sandbox.exclude_list_of_run("/p", str(tmp_path / "out"), "repo", ["v"], True, ask_doc=True)
    assert seen == [(("/p", str(tmp_path / "out"), "repo", ["v"], True, True), {})]


def test_an_index_not_built_from_the_analysis_there_is_built_again(tmp_path, monkeypatch):
    """No LLM: only the index is built, and only when it is not from that analysis."""
    out = tmp_path / "out"
    monkeypatch.setattr(knowledge, "is_complete", lambda path: True)
    fresh_list = []
    monkeypatch.setattr(sandbox.build, "is_index_from", lambda k, i: bool(fresh_list))
    build_list = []
    monkeypatch.setattr(sandbox.build, "index_again",
                        lambda k, name, i: build_list.append((name, i))
                        or fresh_list.append(1))
    assert _analyse_if_none("/p", str(out), "repo", None, False) is False
    assert build_list == [("repo", store.index_db_path(str(out), "repo"))]
    assert _analyse_if_none("/p", str(out), "repo", None, False) is False
    assert len(build_list) == 1


def _update_doc_seen(monkeypatch):
    """Note the exclude list of every _update_doc() call instead of making documents."""
    seen = []
    monkeypatch.setattr(sandbox, "_update_doc", lambda *a: seen.append(a[3]))
    return seen


def test_doc_update_makes_the_documents_over_an_analysis_already_there(tmp_path, monkeypatch):
    out_dir = _analysis_on_disk(tmp_path, monkeypatch, ["sample_output"])
    seen = _update_doc_seen(monkeypatch)
    assert sandbox.exclude_list_of_run("/p", out_dir, "repo", None, True,
                                       doc_update=True) == ["sample_output"]
    assert seen == [["sample_output"]]


@pytest.mark.parametrize("with_doc, doc_update, is_new", [
    (True, False, False), (False, True, False), (True, True, True)])
def test_the_documents_are_made_again_only_when_asked_and_not_just_made(
        tmp_path, monkeypatch, with_doc, doc_update, is_new):
    out_dir = _analysis_on_disk(tmp_path, monkeypatch, [])
    monkeypatch.setattr(sandbox, "analyse_if_none", lambda *a, **k: is_new)
    seen = _update_doc_seen(monkeypatch)
    sandbox.exclude_list_of_run("/p", out_dir, "repo", None, with_doc, doc_update=doc_update)
    assert seen == []


def test_doc_update_with_other_names_is_refused_before_the_documents(tmp_path, monkeypatch):
    out_dir = _analysis_on_disk(tmp_path, monkeypatch, ["sample_output"])
    seen = _update_doc_seen(monkeypatch)
    with pytest.raises(RuntimeError, match="asks to keep out vendor"):
        sandbox.exclude_list_of_run("/p", out_dir, "repo", ["vendor"], True, doc_update=True)
    assert seen == []


def test_update_doc_makes_the_analysis_and_the_index_with_the_documents(tmp_path,
                                                                         monkeypatch, caplog):
    """Both are made under the lock make_agent_dir() copies them under."""
    caplog.set_level(logging.INFO, logger="codetwine_loom.progress")
    (tmp_path / "repo").mkdir()
    lock_path = os.path.realpath(sandbox.filelock.path_for(str(tmp_path / "repo")))
    seen = []
    monkeypatch.setattr(sandbox, "live_run_list", lambda out_dir, name, engine="": [])
    monkeypatch.setattr(sandbox.build, "analyse_and_index",
                        lambda *a, **k: seen.append(
                            (a, k, lock_path in sandbox.filelock._hold_set())))
    sandbox._update_doc("/p", str(tmp_path), "repo", ["v"])
    assert seen == [(("/p", str(tmp_path), ["v"]), {"with_doc": True}, True)]
    assert [r.getMessage().split(" (")[0] for r in caplog.records
            if r.name == "codetwine_loom.progress"] == \
        ["make the design documents again: start", "make the design documents again: end"]


def test_update_doc_is_refused_while_runs_are_going(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "live_run_list", lambda out_dir, name, engine="": ["a/1"])
    monkeypatch.setattr(sandbox.build, "analyse_and_index", _never("runs are going"))
    with pytest.raises(RuntimeError, match=r"1 run\(s\) are going .*\(a/1\).*Wait for "
                       r"them to end, drop --doc-create, or pass another --output-dir"):
        sandbox._update_doc("/p", str(tmp_path), "repo", None)

# The real one, kept before the fixture below stands in for it
_START_BROKER = sandbox.start_broker
_STOP_BROKER = sandbox.stop_broker


@pytest.fixture(autouse=True)
def started_brokers(monkeypatch):
    """Stand in for the broker and record the socket directories it was started on.

    start_broker() launches a process and waits for its socket to appear, so without
    this every test that calls run() starts and stops a real one. test_broker.py is
    where a real one is served.

    Yields:
        The list the stand-in appends each socket directory to.
    """
    started = []

    def start(sock_dir, project_dir="", output_dir="", run_id="", cache="",
              engine="", image="", token_out="", token_left=0, agent="",
              max_refill=None, base="", fill_ext_list=()):
        started.append(sock_dir)
        return None

    monkeypatch.setattr(sandbox, "start_broker", start)
    monkeypatch.setattr(sandbox, "stop_broker",
                        lambda process, sock_dir: shutil.rmtree(sock_dir,
                                                                ignore_errors=True))
    yield started


# What starts the engine's client of one container. Every test but its own has the
# client stood in for by whatever the test put in place of subprocess.run
_START_CLIENT = sandbox._start_client


class _ClientThroughRun:
    """A stand-in for the engine's client that is run by subprocess.run when it is
    waited for, with the arguments the client is started with."""

    def __init__(self, argv, out_fd, err_fd):
        self._argv, self._out_fd, self._err_fd = argv, out_fd, err_fd
        self.returncode = None
        self.stop_list = []

    def wait(self, timeout=None):
        if self.returncode is None:
            # A wait that raised is not made again: the client counts as ended
            self.returncode = -1
            self.returncode = sandbox.subprocess.run(
                self._argv, check=False, stdout=self._out_fd, stderr=self._err_fd,
                env=config.child_env(), timeout=timeout).returncode
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.stop_list.append("terminate")

    def kill(self):
        self.stop_list.append("kill")


@pytest.fixture(autouse=True)
def _client_through_run(monkeypatch):
    monkeypatch.setattr(sandbox, "_start_client", _ClientThroughRun)


@pytest.fixture(autouse=True)
def _stages_run_once(monkeypatch):
    """A stage after the merge left unsettled is run once here; the tests of running it
    again set the two settings themselves."""
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 0)


_REAL_IMAGE_ID = sandbox._image_id


@pytest.fixture(autouse=True)
def _no_engine_is_asked_for_an_image(monkeypatch):
    """The id of an image is answered without asking an engine. The stand-in for the
    engine (_engine_stand_in) and the tests of the id itself put the real one back."""
    monkeypatch.setattr(sandbox, "_image_id", lambda engine, image:
                        f"sha256:{image or config.CONTAINER_IMAGE}")


@pytest.fixture(autouse=True)
def _the_cache_was_filled_already(monkeypatch):
    """The first run fills the cache before its container; every test here is about
    what comes after, so the filling is answered as one that already happened. The
    tests of the filling itself put the real one back."""
    monkeypatch.setattr(sandbox, "fill_cache_once", lambda *a, **k: 0)


def _git(cwd, *args):
    """Run one git command in cwd and return its stdout, failing the test on error."""
    done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                          text=True, check=False)
    assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr}"
    return done.stdout


def _is_git(argv):
    """Say whether a command line starts git, as it is or held to an address space
    (gitcmd.run with max_address_byte)."""
    return argv[0] == "git" or (gitcmd._LIMIT_START in argv and "git" in argv)


def _is_image_inspect(argv):
    """Say whether a command line asks an engine for an image."""
    return list(argv[1:3]) == ["image", "inspect"]


def _engine_stand_in(monkeypatch, on_container,
                     image_id=lambda engine, name: f"sha256:stand-in-{engine}-{name}"):
    """Stand in for the engine: let the git calls through, answer an image inspect for
    the id with image_id(<the engine asked>, <the image's name>), and answer every other
    command line with on_container(argv).

    subprocess.run is read before it is replaced, so the stand-in does not call itself.
    pick_engine() returns the engine it is given, or /usr/bin/true for none, and
    sandbox._image_id is the real one.

    Args:
        monkeypatch: pytest's fixture.
        on_container: Called with each command line but git's and the image inspect's;
            what it returns is what the command ended as.
        image_id: Called with the engine and the image's name for the id each image
            inspect prints.
    """
    real_run = subprocess.run
    picked_set = set()

    def pick(name=""):
        picked_set.add(name or "/usr/bin/true")
        return name or "/usr/bin/true"

    def run(argv, **kwargs):
        if _is_git(argv):
            return real_run(argv, **kwargs)
        if _is_image_inspect(argv):
            assert argv[0] in picked_set, argv
            assert argv[3:5] == ["--format", "{{.Id}}"], argv
            assert len(argv) == 6 and argv[5], argv
            return subprocess.CompletedProcess(
                argv, 0, stdout=f"{image_id(argv[0], argv[5])}\n")
        return on_container(argv)

    monkeypatch.setattr(sandbox, "_image_id", _REAL_IMAGE_ID)
    monkeypatch.setattr(sandbox, "pick_engine", pick)
    monkeypatch.setattr(sandbox.subprocess, "run", run)


def _stand_in_for_the_container(monkeypatch, seen=None):
    """Stand in for the engine (_engine_stand_in), answering every container as one
    that ran.

    Args:
        monkeypatch: pytest's fixture.
        seen: A list each container's argv is appended to, or None to drop them.
    """
    def ran(argv):
        if seen is not None:
            seen.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    _engine_stand_in(monkeypatch, ran)


def _repo(path):
    """Make a git repository at path with one committed file and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "--quiet", "-b", "main")
    _git(path, "config", "user.email", "sandbox@example.invalid")
    _git(path, "config", "user.name", "sandbox")
    (path / "kept.txt").write_text("one\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "--quiet", "-m", "first")
    return path


# A run that went through here before this one, as its files are named
_EARLIER_RUN = "20260101-000000-1"


def _a_run_that_ended():
    """Return the id of a run whose process is no longer there.

    The id ends in the process that made it, which is how a clone left by a run that
    was killed is told from one a run is still writing. A process that has ended and
    been waited for is one no id of a run now going can name.

    Returns:
        The run id.
    """
    ended = subprocess.Popen([sys.executable, "-c", ""])
    ended.wait()
    return f"20260101-000000-{ended.pid}"


def _patch_of(project_out, run_id=_EARLIER_RUN):
    """Return where the patch of one run goes, making the directory it goes in.

    Args:
        project_out: <output dir>/<repository name>.
        run_id: The run, as new_run_id() names it.

    Returns:
        The patch's path.
    """
    directory = pathlib.Path(project_out) / "sandbox"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / sandbox.patch_name(run_id)


def _make_agent(out, agent="20260101-000000-1-42"):
    """Make an agent's own copy of the output directory, and return the output
    directory as run_file_layout() takes it.

    Args:
        out: The output directory.
        agent: The agent, as agent_name() names it.

    Returns:
        out as a string.
    """
    sandbox.make_agent_dir(str(out), "repo", agent, "", None, False)
    return str(out)


def _mounted_at(argv, inside):
    """Return what a container given argv sees at one of its mount points.

    The run works in the agent's own copy of the output directory, so where /out is on
    this machine is what argv says it is, not what the caller passed run().

    Args:
        argv: What the container would have been started with.
        inside: The mount point, e.g. "/out".

    Returns:
        The path on this machine.
    """
    for i, arg in enumerate(argv):
        if arg == "-v" and argv[i + 1].split(":")[1] == inside:
            return pathlib.Path(argv[i + 1].split(":")[0])
    raise AssertionError(f"nothing is mounted at {inside}: {argv}")


def _told_to_write(argv):
    """Return where a container given argv would write its patch, on this machine.

    The run names the patch after itself and tells the container where it goes, so a
    test standing in for the container reads it back the same way the container does.

    Args:
        argv: What the container would have been started with.

    Returns:
        The patch's path.
    """
    inside = next(a.split("=", 1)[1] for a in argv if a.startswith("CTLOOM_PATCH="))
    return _mounted_at(argv, "/out") / inside[len("/out/"):]


def _patch_from(repo, write):
    """Write changes into a clone of repo and return the patch they make.

    Args:
        repo: The repository to clone.
        write: Called with the clone's path; it makes the changes.

    Returns:
        The patch text, as "git diff --cached --binary" produces it.
    """
    clone = repo.parent / f"{repo.name}-clone"
    _git(repo.parent, "clone", "--quiet", "--no-hardlinks", str(repo), str(clone))
    write(clone)
    _git(clone, "add", "-A")
    return _git(clone, "diff", "--cached", "--binary")


# == The engine ===============================================

def test_the_engine_asked_for_is_looked_up_on_path(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name == "docker" else None)
    assert sandbox.pick_engine("docker") == "/usr/bin/docker"


def test_an_engine_that_is_not_installed_says_so(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="podman is not on PATH"):
        sandbox.pick_engine("podman")


def test_the_engines_are_tried_in_order(monkeypatch):
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "")
    monkeypatch.setattr(sandbox.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name == "docker" else None)
    assert sandbox.pick_engine() == "/usr/bin/docker"


def test_no_engine_at_all_names_the_ones_that_were_tried(monkeypatch):
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="podman, docker"):
        sandbox.pick_engine()


# == the image is asked for before anything is started =======
def _image_answer(monkeypatch, status_dict: dict[str, int | Exception],
                   other_list: list[str] | None = None, engine_stdout: str = "",
                   engine_stderr: str = "Error: image not known\n") -> list[list[str]]:
    """Have each engine end its image inspect as told, and return the commands asked.

    Args:
        status_dict: The exit status by the engine's path, or the exception to raise.
        other_list: The other engines on PATH.
        engine_stdout: What each engine prints to its stdout.
        engine_stderr: What each engine prints to its stderr.
    """
    argv_list = []

    def run(argv, **kwargs):
        argv_list.append(argv)
        status = status_dict[argv[0]]
        if isinstance(status, Exception):
            raise status
        return subprocess.CompletedProcess(argv, status, engine_stdout,
                                           engine_stderr)

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    monkeypatch.setattr(sandbox, "_other_engine_path_list",
                        lambda engine_path: other_list or [])
    return argv_list


_LABEL_FORMAT_ARG_LIST = ["--format", "{{json .Config.Labels}}"]


def test_an_engine_that_has_the_image_gives_its_labels(monkeypatch):
    monkeypatch.setattr(config, "CONTAINER_IMAGE", "codetwine-loom:set")
    argv_list = _image_answer(monkeypatch, {"/usr/bin/podman": 0},
                              engine_stdout='{"io.codetwine-loom.source":"ab"}\n')
    assert sandbox.image_label_dict("/usr/bin/podman", "mine") == {
        "io.codetwine-loom.source": "ab"}
    sandbox.image_label_dict("/usr/bin/podman")
    assert argv_list == [
        ["/usr/bin/podman", "image", "inspect", *_LABEL_FORMAT_ARG_LIST, "mine"],
        ["/usr/bin/podman", "image", "inspect", *_LABEL_FORMAT_ARG_LIST,
         "codetwine-loom:set"]]


@pytest.mark.parametrize("stdout", ["null\n", "", "not json"])
def test_an_image_without_labels_gives_none(monkeypatch, stdout):
    _image_answer(monkeypatch, {"/usr/bin/podman": 0}, engine_stdout=stdout)
    assert sandbox.image_label_dict("/usr/bin/podman", "mine") == {}


def test_an_engine_without_the_image_stops_the_run_with_what_it_printed(monkeypatch):
    _image_answer(monkeypatch, {"/usr/bin/podman": 125})
    with pytest.raises(sandbox.NoImageError) as raised:
        sandbox.image_label_dict("/usr/bin/podman", "mine")
    assert str(raised.value).splitlines() == [
        "podman does not show the image mine (exit status 125): Error: image not known"]


def test_an_engine_that_says_why_on_its_stdout_alone_is_read_there(monkeypatch):
    _image_answer(monkeypatch, {"/usr/bin/docker": 1}, engine_stdout="not set up here\n",
                  engine_stderr="")
    with pytest.raises(RuntimeError) as raised:
        sandbox.image_label_dict("/usr/bin/docker", "mine")
    assert str(raised.value).splitlines()[0] == \
        "docker does not show the image mine (exit status 1): not set up here"


def test_another_engine_that_has_the_image_is_named(monkeypatch):
    argv_list = _image_answer(
        monkeypatch, {"/usr/bin/podman": 125, "/usr/bin/docker": 0, "/opt/nerdctl": 1,
                      "/opt/gone": OSError("no such file")},
        ["/usr/bin/docker", "/opt/nerdctl", "/opt/gone"])
    with pytest.raises(RuntimeError) as raised:
        sandbox.image_label_dict("/usr/bin/podman", "mine")
    assert str(raised.value).splitlines()[1:] == [
        "docker has an image of that name: set CONTAINER_ENGINE=docker to use it"]
    assert [argv[0] for argv in argv_list] == ["/usr/bin/podman", "/usr/bin/docker",
                                               "/opt/nerdctl", "/opt/gone"]


@pytest.mark.parametrize("error", [
    OSError("no such file"), subprocess.TimeoutExpired(["podman"], 30)])
def test_an_engine_that_cannot_be_asked_for_the_image_is_passed_with_a_warning(
        error, monkeypatch, caplog):
    _image_answer(monkeypatch, {"/usr/bin/podman": error})
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        assert sandbox.image_label_dict("/usr/bin/podman", "mine") is None
    assert "could not ask podman for the image mine" in caplog.text


# == What is forwarded ========================================

def test_every_setting_is_forwarded_by_name():
    """A setting added to config.py reaches the container without being listed twice."""
    forwarded = sandbox.env_forward_argv()
    assert forwarded[0::2] == ["-e"] * (len(forwarded) // 2)
    names = forwarded[1::2]
    for name in ("MEDIUM_MODEL", "RLM_MAX_TURN", "RUN_COMMAND",
                 "TEST_PATH", "SMOKE_ARG"):
        assert name in names
    assert len(names) > 50


def test_the_credentials_are_not_forwarded():
    """The broker holds them, so the container is given neither."""
    names = sandbox.env_forward_argv()[1::2]
    assert "LLM_API_KEY" not in names and "LLM_API_BASE" not in names


def test_where_and_how_a_search_is_made_stays_outside_the_container():
    """The container asks a question; the process with the network knows the rest."""
    names = sandbox.env_forward_argv()[1::2]
    for name in ("WEB_SEARCH_PROVIDER", "WEB_SEARCH_API_KEY", "WEB_SEARCH_MODEL",
                 "WEB_SEARCH_MAX_TOKEN", "WEB_SEARCH_CONTEXT_SIZE",
                 "WEB_SEARCH_TIMEOUT_SEC", "WEB_SEARCH_MIN_INTERVAL_SEC",
                 "WEB_FETCH_PROVIDER", "WEB_FETCH_API_KEY", "WEB_FETCH_TIMEOUT_SEC",
                 "WEB_FETCH_MIN_INTERVAL_SEC"):
        assert name not in names, name


def test_what_goes_on_the_command_line_is_not_forwarded_by_name():
    names = sandbox.env_forward_argv()[1::2]
    assert "EDIT_DRY_RUN" not in names and "EDIT_GUIDE" not in names


def test_the_caps_the_model_is_told_about_do_reach_the_container():
    """They are in the instructions, so the run has to hold the same numbers."""
    names = sandbox.env_forward_argv()[1::2]
    for name in ("WEB_SEARCH_QUESTION_MAX_CHAR", "WEB_SEARCH_RESULT_MAX_CHAR",
                 "WEB_FETCH_RESULT_MAX_CHAR"):
        assert name in names, name


def test_the_web_budgets_are_written_out_as_0_without_an_endpoint(monkeypatch):
    """The endpoint's settings stay with the broker, so the budget is how the run
    learns whether the tool is on."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 7)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 4)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "")
    monkeypatch.setattr(config, "WEB_SEARCH_PROVIDER", "")
    monkeypatch.setattr(config, "WEB_FETCH_URL", "")
    monkeypatch.setattr(config, "LLM_BROKER", "")
    argv = _argv()
    assert "WEB_SEARCH_MAX_CALL=0" in argv and "WEB_FETCH_MAX_CALL=0" in argv
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    argv = _argv()
    assert "WEB_SEARCH_MAX_CALL=7" in argv and "WEB_FETCH_MAX_CALL=4" in argv
    names = sandbox.env_forward_argv()[1::2]
    assert "WEB_SEARCH_MAX_CALL" not in names and "WEB_FETCH_MAX_CALL" not in names


def test_the_page_read_budget_is_written_out_as_0_when_the_search_is_off(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 4)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    monkeypatch.setattr(config, "LLM_BROKER", "")
    argv = _argv()
    assert "WEB_SEARCH_MAX_CALL=0" in argv and "WEB_FETCH_MAX_CALL=0" in argv


def test_no_credential_reaches_the_command_line(monkeypatch):
    """Not by name and not by value: neither -e form appears anywhere in the argv."""
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")
    monkeypatch.setattr(config, "LLM_API_KEY", "not-a-real-key")
    argv = _argv()
    assert "not-a-real-key" not in " ".join(argv)
    assert not [a for a in argv if a.startswith(("LLM_API_KEY", "LLM_API_BASE"))]


def test_the_paths_of_this_machine_are_not_forwarded():
    names = sandbox.env_forward_argv()[1::2]
    assert "PROJECT_DIR" not in names and "OUTPUT_DIR" not in names


def test_a_setting_written_out_by_hand_is_not_also_forwarded_by_name():
    """Two -e for one name would leave which value survives up to the engine."""
    assert "BUILD_EXCLUDE" not in sandbox.env_forward_argv()[1::2]
    argv = sandbox.build_run_argv("/usr/bin/podman", "an-image", _MOUNTS, "abc123",
                                  "repo", ["edit"], exclude_list=["vendor"])
    assert len([a for a in argv if a.startswith("BUILD_EXCLUDE")]) == 1


def test_the_container_and_a_started_program_are_kept_from_the_same_names():
    """Named once, so adding a credential does not have to be remembered twice."""
    assert tuple(sandbox._BROKER_ONLY_TUPLE) == tuple(config.SECRET_NAME_TUPLE)


# The words a setting holding a credential is named with. env_forward_argv() forwards
# every setting it is not told to keep back, so a credential added to config.py and
# forgotten in SECRET_NAME_TUPLE reaches the container by default. The checks above are
# lists of names somebody has to extend; this one fails on the new name itself.
#
# They are matched as whole words of the name rather than as substrings, and a word
# right after MAX counts things (AGENT_MAX_TOKEN) rather than holding one
_CREDENTIAL_WORD_SET = {"KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH",
                        "CREDENTIAL", "CREDENTIALS"}


def _named_like_a_credential() -> list[str]:
    """Return the settings whose name holds one of _CREDENTIAL_WORD_SET."""
    found = []
    for name in config.env_name_tuple():
        word_list = name.split("_")
        if any(word in _CREDENTIAL_WORD_SET and (i == 0 or word_list[i - 1] != "MAX")
               for i, word in enumerate(word_list)):
            found.append(name)
    return found


def test_a_setting_named_like_a_credential_is_kept_from_the_container():
    """Forwarding is the default, so a new credential has to be named in SECRET_NAME_TUPLE
    or it is handed to a container that is meant to hold none.
    """
    loose = [name for name in _named_like_a_credential()
             if name not in config.SECRET_NAME_TUPLE]
    assert not loose, (f"{loose} read like credentials and are forwarded to the "
                       f"container. Add them to config.SECRET_NAME_TUPLE")


def test_the_credential_word_scan_finds_the_setting_that_is_there():
    """The check above passes on its own if no setting matches the words."""
    assert set(_named_like_a_credential()) == {"LLM_API_KEY", "WEB_SEARCH_API_KEY",
                                              "WEB_FETCH_API_KEY"}


def test_a_setting_is_forwarded_by_name_and_not_by_value(monkeypatch):
    """A bare -e NAME is dropped by the engine when the name is unset here."""
    monkeypatch.setenv("MEDIUM_MODEL", "some/model")
    assert "some/model" not in " ".join(sandbox.env_forward_argv())


def test_config_reads_the_environment_only_through_its_helpers():
    """A read written straight into config.py would not be forwarded at all."""
    tree = ast.parse(_CONFIG.read_text(encoding="utf-8"))
    inside = set()
    for node in ast.walk(tree):
        # child_env(), shell_env() and git_index_state() copy the environment,
        # _load_env_file() fills it from .env and unload_env_file() puts that back,
        # skip_env_setting_list() and
        # proxy_name_list() look at
        # whether a name is set, _proxy_source_name() names the proxy variable a value
        # came from, model_endpoint_dict() reads the providers' endpoints,
        # run_value() reads what a run reads of a name the first run's questions
        # answer, _settle_ca_file() and _drop_empty_proxy_name() settle
        # the variables of the HTTP clients, and is_env_file_off(), env_file_off() and
        # _import_without_env_search() read and set python-dotenv's switch, rather than
        # reading a setting out of it, so nothing they touch belongs in env_name_tuple()
        if isinstance(node, ast.FunctionDef) and node.name in (
                "env", "json_env", "str_env", "child_env", "shell_env", "git_index_state",
                "_load_env_file", "unload_env_file", "skip_env_setting_list",
                "is_env_file_off", "env_file_off",
                "_import_without_env_search", "proxy_name_list", "_proxy_source_name",
                "model_endpoint_dict", "run_value", "_settle_ca_file",
                "_drop_empty_proxy_name"):
            inside.update(id(n) for n in ast.walk(node))
    loose = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and n.attr == "environ"
             and id(n) not in inside]
    assert not loose, (f"config.py line(s) {loose} read the environment outside env() / "
                       f"json_env() / str_env(), so env_name_tuple() does not see them")


def test_the_broker_is_started_and_its_socket_is_what_gets_mounted(
        monkeypatch, tmp_path, started_brokers):
    """The container is given the directory the broker just started listening in."""
    repo = _repo(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    sandbox.run(str(repo), str(tmp_path / "out"), "x")
    assert len(started_brokers) == 1
    assert f"{started_brokers[0]}:/run/ctloom" in seen[0]
    assert "--network=none" in seen[0]


@pytest.mark.parametrize("flag", [True, False])
def test_a_sandbox_run_makes_the_overview_before_the_agent_copies_the_output_dir(
        monkeypatch, tmp_path, flag):
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    event_list = []
    monkeypatch.setattr(sandbox, "make_overview",
                        lambda *a: event_list.append(("overview", a[:2])))
    make = sandbox.make_agent_dir
    monkeypatch.setattr(sandbox, "make_agent_dir",
                        lambda *a, **k: event_list.append(("copy",)) or make(*a, **k))
    monkeypatch.setattr(sandbox.overview, "is_on", lambda: flag)
    _stand_in_for_the_container(monkeypatch)
    sandbox.run(str(repo), str(out), "x", with_overview=True)
    if flag:
        assert event_list[:2] == [("overview", (str(repo), str(out))), ("copy",)]
    else:
        assert event_list == [("copy",)]


def test_an_agent_of_a_dispatching_run_makes_no_overview(tmp_path, monkeypatch):
    """dispatch() made it before the agents' copies, once for the whole run."""
    call_list = []
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: call_list.append(k) or {
        "commit": "", "exit_status": 0, "verdict": ""})
    sandbox._edit_task(str(tmp_path), str(tmp_path / "out"), "20260101-000000-1-42", "",
                       "", None, "r", 1, "one", 1, "")
    assert not call_list[0].get("with_overview", False)


def test_the_socket_directory_is_gone_afterwards(monkeypatch, tmp_path,
                                                 started_brokers):
    repo = _repo(tmp_path / "repo")
    _stand_in_for_the_container(monkeypatch)
    sandbox.run(str(repo), str(tmp_path / "out"), "x")
    assert not os.path.exists(started_brokers[0])


def test_the_cache_outlives_the_run_the_clone_does_not(monkeypatch, tmp_path):
    """A container with no network cannot fetch again, so the cache is kept."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    _stand_in_for_the_container(monkeypatch)
    sandbox.run(str(repo), str(out), "x")
    cache = out / "agents" / ".repo" / "repo" / "cache"
    assert cache.is_dir()
    assert sorted(p.name for p in cache.iterdir()) == ["gradle", "home", "uv", "xdg"]
    assert list(out.glob("agents/.host/*/repo/sandbox/scratch-*")) == []


def test_a_run_whose_broker_would_have_no_key_stops_before_the_container(monkeypatch,
                                                                       tmp_path):
    """The broker is the only process holding a key, so a key it lacks the whole run
    lacks. What the check says comes from llm.missing_credential_text()."""
    repo = _repo(tmp_path / "repo")
    monkeypatch.setattr(sandbox.llm, "missing_credential_text",
                        lambda: "m (MEDIUM_MODEL) needs X_API_KEY. Set LLM_API_KEY")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        sandbox.run(str(repo), str(tmp_path / "out"), "x")
    assert seen == []


@pytest.mark.parametrize("make", [os.mkfifo, lambda path: path.write_text("x")])
def test_a_run_directory_a_container_left_something_else_in_place_of_stops_in_one_line(
        tmp_path, monkeypatch, make):
    """What an earlier container of the agent left where the run's files go is named
    with the agent, and no engine is asked anything."""
    monkeypatch.setattr(sandbox, "pick_engine", lambda engine="": "/usr/bin/true")
    monkeypatch.setattr(sandbox.llm, "missing_credential_text", lambda: "")
    monkeypatch.setattr(sandbox, "remove_dead_clone",
                        lambda *a, **k: pytest.fail("went past the run's directory"))
    out = str(tmp_path / "out")
    run_file = sandbox.run_file_layout(out, "codetwine-loom-agent1", "repo", "20260101-000000-1-1")
    pathlib.Path(run_file.directory).parent.mkdir(parents=True)
    make(pathlib.Path(run_file.directory))
    with pytest.raises(RuntimeError, match=r"codetwine-loom-agent1's copy of the output directory "
                                           r"holds what a run cannot be started in at "
                                           r".*/sandbox \(File exists\)\. Remove it by hand"):
        sandbox._start_container("", "img", str(tmp_path), out, run_file.out_dir, "repo",
                                 "codetwine-loom-agent1", "0" * 40, run_file, "20260101-000000-1-1",
                                 ["edit"], [])


def _start_failing(tmp_path, monkeypatch, fill=None, broker_error=None):
    """Call _start_container() on a real repository with the cache filling and the
    broker standing in, and return (the clone directory, the new entries of TMPDIR
    besides the directory of the token counts, which stays until the process ends)."""
    repo = _repo(tmp_path / "repo")
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_dir))
    monkeypatch.setattr(sandbox, "pick_engine", lambda engine="": "/usr/bin/true")
    monkeypatch.setattr(sandbox.llm, "missing_credential_text", lambda: "")
    monkeypatch.setattr(sandbox, "index_describe", lambda *a: True)
    monkeypatch.setattr(sandbox, "fill_cache_once", fill or (lambda *a, **k: 0))

    def start(sock_dir, *a, **k):
        raise broker_error

    monkeypatch.setattr(sandbox, "start_broker", start)
    out = str(tmp_path / "out")
    run_file = sandbox.run_file_layout(out, "a", "repo", "20260101-000000-1-1")
    with pytest.raises(BaseException):
        sandbox._start_container("", "img", str(repo), out, run_file.out_dir, "repo",
                                 "a", "0" * 40, run_file, "20260101-000000-1-1",
                                 ["edit"], [])
    sandbox.tokencount.reset()
    return run_file.scratch, os.listdir(tmp_dir)


def test_the_broker_is_started_with_the_run_s_engine_and_image(tmp_path, monkeypatch):
    seen = []
    _start_failing(tmp_path, monkeypatch, broker_error=RuntimeError("stop"))
    monkeypatch.setattr(sandbox, "start_broker",
                        lambda sock_dir, *a, **k: seen.append(a) or (_ for _ in ()).throw(
                            RuntimeError("stop")))
    repo = tmp_path / "repo"
    out = str(tmp_path / "out")
    run_file = sandbox.run_file_layout(out, "a", "repo", "20260101-000000-1-2")
    with pytest.raises(RuntimeError, match="stop"):
        sandbox._start_container("", "codetwine-loom:js", str(repo), out,
                                 run_file.out_dir, "repo", "a", "0" * 40, run_file,
                                 "20260101-000000-1-2", ["edit"], [])
    assert seen[0][-2:] == ("/usr/bin/true", "codetwine-loom:js")


def test_the_broker_is_handed_the_engine_and_image_a_refill_starts(tmp_path, monkeypatch,
                                                                  real_broker_start):
    sock_dir = tempfile.mkdtemp(prefix="ctloom-test-")
    pathlib.Path(sock_dir, "llm.sock").touch()
    seen = {}

    class Going:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(sandbox.subprocess, "Popen",
                        lambda argv, env=None, stdin=None: seen.update(env=env) or Going())
    sandbox.start_broker(sock_dir, str(tmp_path), str(tmp_path / "out"), "r", "",
                         "/usr/bin/docker", "codetwine-loom:js")
    assert seen["env"]["CONTAINER_ENGINE"] == "/usr/bin/docker"
    assert seen["env"]["CONTAINER_IMAGE"] == "codetwine-loom:js"
    sandbox.start_broker(sock_dir)
    assert seen["env"].get("CONTAINER_IMAGE") == os.environ.get("CONTAINER_IMAGE")
    shutil.rmtree(sock_dir)


def test_the_broker_is_handed_the_model_table_entries_and_told_to_fetch_none(
        tmp_path, monkeypatch, real_broker_start):
    sock_dir = tempfile.mkdtemp(prefix="ctloom-test-")
    pathlib.Path(sock_dir, "llm.sock").touch()
    seen = {}

    class Going:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(sandbox.subprocess, "Popen",
                        lambda argv, env=None, stdin=None: seen.update(env=env) or Going())
    monkeypatch.setattr(config, "EDIT_MODEL", "anthropic/made-up-model")
    monkeypatch.setitem(sandbox.broker.litellm.model_cost, "made-up-model",
                        {"litellm_provider": "anthropic", "max_tokens": 10})
    monkeypatch.setenv("LLM_API_KEY", "the key")
    sandbox.start_broker(sock_dir)
    assert seen["env"]["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert seen["env"]["LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS"] == "True"
    assert json.loads(seen["env"]["CTLOOM_MODEL_TABLE"])["made-up-model"] == {
        "litellm_provider": "anthropic", "max_tokens": 10}
    # The rest of this process's environment reaches it as before
    assert seen["env"]["LLM_API_KEY"] == "the key"
    shutil.rmtree(sock_dir)


def test_a_broker_that_ends_before_it_listens_leaves_no_clone_and_no_socket(
        tmp_path, monkeypatch):
    scratch, tmp_list = _start_failing(tmp_path, monkeypatch,
                                       broker_error=RuntimeError("the broker ended"))
    assert not os.path.exists(scratch) and tmp_list == []


def test_a_stop_while_the_cache_is_filled_leaves_no_clone(tmp_path, monkeypatch):
    def fill(*a, **k):
        raise KeyboardInterrupt

    scratch, tmp_list = _start_failing(tmp_path, monkeypatch, fill=fill)
    assert not os.path.exists(scratch) and tmp_list == []


def _drop_noted(monkeypatch):
    """Note the agents whose copy drop_the_copy() is called for, and return the list."""
    drop_list = []
    drop = sandbox.drop_the_copy
    monkeypatch.setattr(sandbox, "drop_the_copy",
                        lambda out_dir, name, agent: drop_list.append(agent)
                        or drop(out_dir, name, agent))
    return drop_list


def test_a_merging_run_whose_container_cannot_start_leaves_no_copy(_dispatch_repo,
                                                                   monkeypatch):
    repo, out = _dispatch_repo
    _git(repo, "branch", "a")
    _git(repo, "branch", "b")

    def start(*a, **k):
        raise RuntimeError("the broker ended")

    monkeypatch.setattr(sandbox, "_start_container", start)
    drop_list = _drop_noted(monkeypatch)
    with pytest.raises(RuntimeError, match="the broker ended"):
        sandbox.integrate(str(repo), str(out), ["a", "b"])
    assert drop_list == [sandbox.agent_name("", "20260101-000000-1-42")]


def test_a_merging_container_is_given_the_tips_of_the_branches_it_combines(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    tip_list = []
    for branch in ("a", "b"):
        _git(repo, "checkout", "--quiet", "-b", branch, "main")
        tip_list.append(_commit_file(repo, f"{branch}.py"))
    _git(repo, "checkout", "--quiet", "main")
    seen = {}

    def start(*a, **k):
        seen.update(k)
        raise RuntimeError("stop")

    monkeypatch.setattr(sandbox, "_start_container", start)
    with pytest.raises(RuntimeError, match="stop"):
        sandbox.integrate(str(repo), str(out), ["a", "b"])
    assert seen["tip_list"] == tip_list


def test_a_stage_carried_on_from_a_branch_that_cannot_be_moved_leaves_no_copy(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    head = _git(repo, "rev-parse", "HEAD").strip()
    resume_dict = {"union": "f" * 40, "stage": integrate_mod.REFACTOR_STAGE,
                   "base": head, "paths": [], "merge": head, "settled": head}
    drop_list = _drop_noted(monkeypatch)
    with pytest.raises(RuntimeError, match="could not be put back"):
        sandbox._resume_stage(str(repo), str(out), "repo", "main", resume_dict, "", "",
                              False, "", None, False)
    assert drop_list == ["main"]


def test_the_engine_is_never_started_with_the_credentials(monkeypatch):
    """podman set to hand the host's environment on (env_host) would otherwise put the
    keys in the container."""
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setattr(sandbox, "_STOP_EVENT", threading.Event())
    env_list = []

    def run(argv, **kwargs):
        env_list.append(kwargs.get("env"))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    sandbox.run_the_container(["podman", "run"], "podman", "r1", 0, os.devnull)
    sandbox._container_alive("podman", "r1")
    sandbox._remove_container("podman", sandbox.container_name("r1"))
    sandbox._image_proc("podman", "i")
    assert len(env_list) == 4
    assert all(env is not None and "LLM_API_KEY" not in env for env in env_list)
    # The client itself, started as it is outside the tests
    env_path = os.path.join(tempfile.mkdtemp(prefix="ctloom-test-"), "env")
    client = _START_CLIENT(["sh", "-c", f"env > {shlex.quote(env_path)}"], None, None)
    assert client.wait(timeout=30) == 0
    with open(env_path, encoding="utf-8") as fp:
        assert "LLM_API_KEY=" not in fp.read()
    shutil.rmtree(os.path.dirname(env_path))


def test_a_linked_worktree_takes_in_under_the_lock_of_its_repository(tmp_path):
    repo = _repo(tmp_path / "repo")
    _git(repo, "worktree", "add", "-q", str(tmp_path / "linked"))
    assert os.path.realpath(sandbox._take_in_lock(str(tmp_path / "linked"))) \
        == os.path.realpath(sandbox._take_in_lock(str(repo))) \
        == os.path.realpath(repo / ".git" / "codetwine-loom-take-in")


@pytest.mark.parametrize("root", ["codetwine-loom", "codetwine-loom/run", "codetwine-loom/sandbox"])
def test_a_branch_the_runs_branches_would_go_under_stops_the_run_first(
        monkeypatch, tmp_path, root):
    repo = _repo(tmp_path / "repo")
    _git(repo, "branch", root)
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    with pytest.raises(RuntimeError, match=f"branch named {root}.*git -C .* branch -m {root} <new name>"):
        sandbox.run(str(repo), str(tmp_path / "out"), "x")
    assert seen == []


@pytest.mark.parametrize("name", ["run", "sandbox"])
def test_a_branch_with_a_common_name_does_not_stop_the_run(monkeypatch, tmp_path, name):
    repo = _repo(tmp_path / "repo")
    _git(repo, "branch", name)
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    sandbox.run(str(repo), str(tmp_path / "out"), "x")
    assert seen


def test_an_instruction_file_named_by_an_absolute_path_is_warned_about_once(
        monkeypatch, tmp_path, caplog):
    repo = _repo(tmp_path / "repo")
    monkeypatch.setattr(config, "TEST_GUIDE", "/elsewhere/TEST.md")
    sandbox._warn_absolute_guide.cache_clear()
    caplog.set_level(logging.WARNING, logger="codetwine_loom.sandbox")
    sandbox._run_place(str(repo), str(tmp_path / "out"))
    sandbox._run_place(str(repo), str(tmp_path / "out"))
    sandbox._warn_absolute_guide.cache_clear()
    assert caplog.text.count("TEST_GUIDE names /elsewhere/TEST.md") == 1
    assert "EDIT_GUIDE" not in caplog.text


def test_an_instruction_file_named_by_a_path_out_of_the_repository_is_warned_about(
        monkeypatch, tmp_path, caplog):
    repo = _repo(tmp_path / "repo")
    monkeypatch.setattr(config, "REFACTOR_GUIDE", "../shared/RULES.md")
    sandbox._warn_absolute_guide.cache_clear()
    caplog.set_level(logging.WARNING, logger="codetwine_loom.sandbox")
    sandbox._run_place(str(repo), str(tmp_path / "out"))
    sandbox._warn_absolute_guide.cache_clear()
    assert caplog.text.count("REFACTOR_GUIDE names ../shared/RULES.md, a path outside "
                             "the target repository") == 1


# == The container's command line =============================

_MOUNTS = sandbox.MountLayout(src="/host/repo", out="/host/out", scratch="/host/scratch",
                         cache="/host/cache", sock="/host/sock")


def _argv(engine="/usr/bin/podman", **kwargs):
    return sandbox.build_run_argv(engine, "an-image", _MOUNTS, "abc123", "repo",
                                  ["edit", "--request", "x"], **kwargs)


def test_the_engines_init_reaps_what_the_run_leaves():
    assert "--init" in _argv()[:_argv().index("an-image")]


def test_the_repository_is_mounted_read_only():
    assert "/host/repo:/src:ro" in _argv()


def test_the_repository_and_the_output_are_the_only_mounts_of_this_machine():
    mounts = [a for a in _argv() if a.startswith("/host/")]
    assert mounts == ["/host/repo:/src:ro", "/host/out:/out",
                      "/host/scratch/work:/work", "/host/scratch/tmp:/tmp",
                      "/host/cache:/cache", "/host/sock:/run/ctloom"]


def test_podman_is_told_not_to_hand_on_this_machine_s_proxy_variables():
    """It copies HTTP_PROXY and the like into every container by default, whatever is
    forwarded with -e. The container that fills the cache fetches, so it keeps them."""
    assert "--http-proxy=false" in _argv()
    assert "--http-proxy=false" not in _argv(network=True)
    assert not any("http-proxy" in arg for arg in _argv(engine="/usr/bin/docker"))


def _pair_list(argv: list[str], flag: str) -> list[str]:
    """Return what follows each flag in argv."""
    return [value for name, value in zip(argv, argv[1:]) if name == flag]


@pytest.mark.parametrize("engine", ["/usr/bin/podman", "/usr/bin/docker"])
def test_the_container_that_fetches_is_given_the_proxy_variables_by_name(monkeypatch,
                                                                         engine):
    """By name alone, under either engine: the value is the engine's to fill in."""
    monkeypatch.setenv("HTTPS_PROXY", "http://ctloom-proxy.corp.example:3128")
    monkeypatch.setenv("no_proxy", "localhost,.corp.example")
    name_list = _pair_list(_argv(engine=engine, network=True), "-e")
    assert {"HTTPS_PROXY", "no_proxy"} <= set(name_list)
    assert not {"HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "https_proxy"} & set(name_list)
    argv = _argv(engine=engine)
    assert not {"HTTPS_PROXY", "no_proxy"} & set(_pair_list(argv, "-e"))
    assert not any("ctloom-proxy.corp.example" in arg
                   for arg in argv + _argv(engine=engine, network=True))


@pytest.mark.parametrize("name, value, is_said", [
    ("HTTPS_PROXY", "http://puser:proxy-pw-123@ctloom-proxy.corp.example:3128", True),
    ("all_proxy", "puser@ctloom-proxy.corp.example:3128", True),
    ("HTTPS_PROXY", "http://ctloom-proxy.corp.example:3128", False),
    ("NO_PROXY", "localhost,.corp.example", False),
    ("HTTP_PROXY", "http://[broken", False),
    ("HTTPS_PROXY", "http://puser:pa#ss@ctloom-proxy.corp.example:3128", True),
    ("https_proxy", "http://puser:pa/ss@ctloom-proxy.corp.example:3128", True)])
def test_a_proxy_s_user_and_password_handed_to_the_container_that_fetches_are_said(
        monkeypatch, caplog, name, value, is_said):
    """By the variable's name, and never by its value."""
    monkeypatch.setenv(name, value)
    with caplog.at_level(logging.WARNING):
        _argv(network=True)
        fetch_list = [record.getMessage() for record in caplog.records]
        caplog.clear()
        _argv()
        assert caplog.records == []
    assert fetch_list == ([
        f"{name} carries a user or a password, and the container that fetches "
        f"dependencies is given it: the build files of the repository run there with "
        f"a network"] if is_said else [])


@pytest.mark.parametrize("engine", ["/usr/bin/podman", "/usr/bin/docker"])
def test_the_container_that_fetches_is_given_the_file_of_certificate_authorities(
        monkeypatch, tmp_path, engine):
    ca_file = tmp_path / "ctloom-ca.pem"
    ca_file.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SSL_CERT_FILE", "ctloom-ca.pem")
    argv = _argv(engine=engine, network=True)
    assert f"{ca_file}:/etc/ctloom-ca.pem:ro" in _pair_list(argv, "-v")
    assert {"SSL_CERT_FILE=/etc/ctloom-ca.pem", "CURL_CA_BUNDLE=/etc/ctloom-ca.pem",
            "GIT_SSL_CAINFO=/etc/ctloom-ca.pem"} <= set(_pair_list(argv, "-e"))
    assert argv.index(f"{ca_file}:/etc/ctloom-ca.pem:ro") < argv.index("an-image")
    # The container a change is made in gets neither the file nor its name
    assert not any("ctloom-ca.pem" in arg or arg.startswith(
        ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO")) for arg in _argv(engine=engine))


def test_a_file_of_certificate_authorities_whose_path_holds_a_colon_stops_in_one_line(
        monkeypatch, tmp_path):
    """The engine reads a ':' in what it mounts as the end of the path."""
    ca_dir = tmp_path / "ca:dir"
    ca_dir.mkdir()
    (ca_dir / "ca.pem").write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_dir / "ca.pem"))
    with pytest.raises(RuntimeError) as error:
        sandbox.refuse_ca_file_path()
    assert str(error.value) == (
        f"SSL_CERT_FILE names {ca_dir / 'ca.pem'}, and the container that fetches "
        f"dependencies cannot be given a file whose path holds a ':'. Move the file "
        f"and set SSL_CERT_FILE to where it is")
    # What a container asking for a refill is answered with holds no path of this machine
    with pytest.raises(RuntimeError) as error:
        _argv(network=True)
    assert str(error.value) == ("the file SSL_CERT_FILE names cannot be given to the "
                                "container that fetches dependencies: its path holds a ':'")
    for value in ("", str(tmp_path / "no-such:dir" / "ca.pem")):
        monkeypatch.setenv("SSL_CERT_FILE", value)
        sandbox.refuse_ca_file_path()
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_dir / "ca.pem"))
    # The container a change is made in is given no file, whatever its path
    assert not any("ca:dir" in arg for arg in _argv())


def test_a_file_of_certificate_authorities_that_is_not_there_is_said_and_not_mounted(
        monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "no-such.pem"))
    with caplog.at_level(logging.WARNING):
        argv = _argv(network=True)
    assert not any("no-such.pem" in arg or arg.startswith(
        ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO")) for arg in argv)
    assert [record.getMessage() for record in caplog.records] == [
        f"SSL_CERT_FILE names {tmp_path / 'no-such.pem'}, which is not a file, so the "
        f"container that fetches dependencies is given no file of certificate authorities"]


def test_a_build_in_the_container_runs_as_many_jobs_as_its_memory_holds(monkeypatch):
    """One job a GiB of SANDBOX_MEMORY, and one at least."""
    assert "CMAKE_BUILD_PARALLEL_LEVEL=2" in _argv() and "CARGO_BUILD_JOBS=2" in _argv()
    monkeypatch.setattr(config, "SANDBOX_MEMORY_BYTE", 512 * 1024 ** 2)
    assert "CMAKE_BUILD_PARALLEL_LEVEL=1" in _argv() and "CARGO_BUILD_JOBS=1" in _argv()


def test_the_container_is_told_where_to_start_the_run_from():
    """Not the clone: a relative write from the REPL must not reach the repository."""
    argv = _argv()
    assert "CTLOOM_RUNDIR=/tmp" in argv
    assert "CTLOOM_WORK=/work/repo" in argv


def test_the_container_is_told_when_the_run_started(monkeypatch):
    monkeypatch.setenv(progress.RUN_START_ENV, "1000.5")
    assert f"{progress.RUN_START_ENV}=1000.5" in _argv()


def test_the_container_has_no_network():
    assert "--network=none" in _argv()


def test_the_container_is_told_where_the_broker_is():
    assert "LLM_BROKER=unix:///run/ctloom/llm.sock" in _argv()


def test_a_container_with_no_socket_is_told_of_no_broker():
    """The cache-filling run calls no model, so it is given neither."""
    argv = sandbox.build_run_argv(
        "/usr/bin/podman", "an-image",
        sandbox.MountLayout(src="/host/repo", out="/host/out", scratch="/host/scratch",
                       cache="/host/cache"),
        "abc123", "repo", ["run-tests"], network=True,
        entrypoint="/usr/local/bin/ctloom-warm")
    assert "LLM_BROKER=" in argv
    assert "/run/ctloom" not in " ".join(argv)
    assert "--network=none" not in argv
    assert argv[argv.index("--entrypoint") + 1] == "/usr/local/bin/ctloom-warm"


def test_what_the_run_writes_is_not_charged_to_the_memory_limit():
    """A tmpfs would be, and then a clone larger than the limit is an OOM kill."""
    argv = _argv()
    assert "--tmpfs" not in argv
    assert "/host/scratch/work:/work" in argv and "/host/scratch/tmp:/tmp" in argv


def test_the_image_itself_cannot_be_written_to():
    assert "--read-only" in _argv()


def test_the_container_is_stripped_of_what_it_does_not_need():
    argv = _argv()
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert argv[argv.index("--pids-limit") + 1] == str(config.SANDBOX_MAX_PROCESS)
    assert argv[argv.index("--memory") + 1] == str(config.SANDBOX_MEMORY)


def test_the_clone_is_named_after_the_repository():
    """The output lands under /out/<name>/, the same place an index built here uses."""
    argv = _argv(run_id="20260101-000000-1")
    assert "CTLOOM_WORK=/work/repo" in argv
    assert "CTLOOM_PATCH=/out/repo/sandbox/patch-20260101-000000-1.diff" in argv
    assert "CTLOOM_BASE=abc123" in argv


def test_a_container_that_writes_no_patch_is_told_of_none():
    """The entry point that fills the cache writes nothing back, so it is given no
    name to write it under."""
    assert "CTLOOM_PATCH" not in " ".join(_argv())


def test_what_is_kept_out_of_the_rebuild_travels_with_it():
    """The rebuild inside the container covers the files the index was built over."""
    argv = sandbox.build_run_argv("/usr/bin/podman", "an-image", _MOUNTS, "abc123",
                                  "repo", ["edit"], rebuild_index=True,
                                  exclude_list=["vendor", "two words"])
    assert "BUILD_EXCLUDE=vendor 'two words'" in argv


def test_nothing_kept_out_says_so_rather_than_saying_nothing():
    argv = sandbox.build_run_argv("/usr/bin/podman", "an-image", _MOUNTS, "abc123",
                                  "repo", ["edit"])
    assert "BUILD_EXCLUDE=" in argv


@pytest.mark.parametrize("rebuild, expected", [(True, "1"), (False, "0")])
def test_the_container_is_told_whether_to_rebuild_the_index(rebuild, expected):
    argv = sandbox.build_run_argv("/usr/bin/podman", "an-image", _MOUNTS, "abc123",
                                  "repo", ["edit"], rebuild_index=rebuild)
    assert f"CTLOOM_REBUILD_INDEX={expected}" in argv
    assert "CTLOOM_OUT=/out" in argv


def test_the_entry_point_is_the_sandbox_script():
    argv = _argv()
    assert argv[argv.index("--entrypoint") + 1] == "/usr/local/bin/ctloom-sandbox"
    assert argv[-3:] == ["edit", "--request", "x"]
    assert argv[argv.index("--entrypoint") + 2] == "an-image"


def test_podman_keeps_the_user_it_was_started_by():
    assert "--userns=keep-id" in _argv("/usr/bin/podman")


def test_docker_is_told_the_user_by_number():
    argv = _argv("/usr/bin/docker")
    assert "--user" in argv
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


def test_an_engine_that_is_neither_is_left_alone():
    argv = _argv("/usr/bin/something-else")
    assert "--userns=keep-id" not in argv and "--user" not in argv


def test_the_writable_places_are_where_a_home_and_a_cache_are_pointed():
    """They are on the mount that is kept, so a run with no network still finds them."""
    argv = _argv()
    assert "HOME=/cache/home" in argv
    assert "XDG_CACHE_HOME=/cache/xdg" in argv
    assert "UV_CACHE_DIR=/cache/uv" in argv


@pytest.mark.parametrize("network", [False, True])
def test_the_jvm_build_tools_are_told_where_they_keep_things(network):
    """They read the user's home from the system and not from HOME: what Maven fetched
    goes to the mount that is kept, and Gradle has a home in the run's own temporary
    directory and reads what was fetched from the mount that is kept."""
    env_list = _pair_list(_argv(network=network), "-e")
    assert "MAVEN_OPTS=-Dmaven.repo.local=/cache/home/.m2/repository" in env_list
    assert "GRADLE_USER_HOME=/tmp/gradle-home" in env_list
    assert "GRADLE_RO_DEP_CACHE=/cache/gradle" in env_list
    assert not any(name.startswith("GRADLE_OPTS=") for name in env_list)


# == The change the container is asked for ====================

def test_the_change_runs_against_the_clone():
    argv = sandbox.edit_argv("repo", "make it stop crashing")
    assert argv == ["edit", "--project", "/work/repo", "--output-dir", "/out",
                    "--request=make it stop crashing", "--no-resume",
                    "--no-dry-run", "--guide=", "--no-check"]
    assert sandbox.edit_argv("repo", "r", check=True)[-1] == "--check"


@pytest.mark.parametrize("argv", [
    sandbox.edit_argv("repo", "-fix", guide="-g.md"),
    sandbox.integrate_argv("repo", ["a", "b"], request="-fix"),
    sandbox.test_create_argv("repo", "a" * 40, "-fix"),
    sandbox.refactor_argv("repo", "refactor", "a" * 40, "-fix"),
    sandbox.answer_argv("repo", "-fix", "/out/a.json"),
    sandbox.dispatch_argv("repo", "-fix", "/out/t.json"),
])
def test_a_request_starting_with_a_dash_reaches_the_container_as_it_is(argv):
    from codetwine_loom import cli
    args = cli._parse_args(argv)
    assert args.request == "-fix"
    assert getattr(args, "guide", "-g.md") == "-g.md"


def test_a_change_in_a_container_never_carries_an_earlier_record_over():
    """There is no terminal in there to ask the question on, and what an earlier run
    left reaches a container through dispatch's --resume, off the branches."""
    assert "--no-resume" in sandbox.edit_argv("repo", "x")
    assert "--resume" not in sandbox.edit_argv("repo", "x", dry_run=True)


def test_what_was_decided_here_is_on_the_command_line_either_way():
    """The host's choice is written out whatever it is: a guide of "" reads none in the
    container too, and --no-dry-run writes even where this machine's environment says
    not to.
    """
    argv = sandbox.edit_argv("repo", "x", dry_run=True, guide="GUIDE.md")
    assert "--dry-run" in argv and "--no-dry-run" not in argv
    assert "--guide=GUIDE.md" in argv
    argv = sandbox.edit_argv("repo", "x", dry_run=False, guide="")
    assert "--no-dry-run" in argv and "--dry-run" not in argv
    assert "--guide=" in argv


# == Reading the repository ===================================

def test_the_head_commit_is_what_the_container_starts_from(tmp_path):
    repo = _repo(tmp_path / "repo")
    assert gitcmd.require_head(str(repo)) == _git(repo, "rev-parse", "HEAD").strip()


def test_a_directory_that_is_not_a_repository_says_so(tmp_path):
    (tmp_path / "plain").mkdir()
    with pytest.raises(RuntimeError, match="not a git repository"):
        gitcmd.require_head(str(tmp_path / "plain"))


def test_what_the_clone_would_not_carry_over_is_listed(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "kept.txt").write_text("changed\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    assert sorted(gitcmd.status_path_list(str(repo))) == ["kept.txt", "new.txt"]


def test_a_clean_repository_leaves_nothing_behind(tmp_path):
    assert gitcmd.status_path_list(str(_repo(tmp_path / "repo"))) == []


# == Taking the patch in ======================================

def _take_in(tmp_path, write, branch="codetwine-loom/sandbox/one"):
    """Make a repository, make a patch against it and take that patch in."""
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    patch = tmp_path / "patch.diff"
    patch.write_text(_patch_from(repo, write), encoding="utf-8")
    taken = sandbox.take_in_patch(str(repo), str(patch), base, branch,
                            str(tmp_path / "wt"), "sandbox: a change")
    return repo, base, taken


def test_a_patch_is_taken_in_where_git_knows_no_one_to_commit_as(tmp_path, monkeypatch):
    """The commit is this tool's, as the commit of the working tree a run starts from
    is, so a machine with no identity set does not lose the run at its last step."""
    git_config = tmp_path / "gitconfig"
    git_config.write_text("[user]\n\tuseConfigOnly = true\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(git_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                 "GIT_COMMITTER_EMAIL", "EMAIL"):
        monkeypatch.delenv(name, raising=False)
    repo = _repo(tmp_path / "repo")
    patch = tmp_path / "patch.diff"
    patch.write_text(_patch_from(repo, lambda clone: (clone / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8")), encoding="utf-8")
    _git(repo, "branch", "side")
    for key in ("user.name", "user.email"):
        _git(repo, "config", "--unset", key)
    base = gitcmd.require_head(str(repo))
    for n, parent_list in enumerate(((), ("side",))):
        taken = sandbox.take_in_patch(str(repo), str(patch), base, f"codetwine-loom/sandbox/{n}",
                                      str(tmp_path / f"wt{n}"), "sandbox: x", parent_list)
        assert _git(repo, "log", "-1", "--format=%an <%ae>|%cn <%ce>",
                    taken["commit"]).strip() \
            == "|".join(["codetwine-loom <codetwine-loom@localhost>"] * 2)


def test_a_commit_of_this_tool_s_is_its_own_whoever_the_git_settings_name(
        tmp_path, monkeypatch):
    """The settings name an author and a committer apart from the user: the commit a
    patch is taken in with and the commit of the working tree are this tool's all the
    same."""
    git_config = tmp_path / "gitconfig"
    git_config.write_text(
        "[user]\n\tname = U\n\temail = u@example.invalid\n"
        "[author]\n\tname = Auth Or\n\temail = a@example.invalid\n"
        "[committer]\n\tname = Com Mitter\n\temail = c@example.invalid\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(git_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                 "GIT_COMMITTER_EMAIL", "EMAIL"):
        monkeypatch.delenv(name, raising=False)
    repo = _repo(tmp_path / "repo")
    _git(repo, "config", "committer.name", "Of The Repository")
    patch = tmp_path / "patch.diff"
    patch.write_text(_patch_from(repo, lambda clone: (clone / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8")), encoding="utf-8")
    base = gitcmd.require_head(str(repo))
    taken = sandbox.take_in_patch(str(repo), str(patch), base, "codetwine-loom/sandbox/0",
                                  str(tmp_path / "wt"), "sandbox: x")
    (repo / "draft.txt").write_text("x\n", encoding="utf-8")
    for commit in (taken["commit"], gitcmd.snapshot_commit(str(repo))):
        assert _git(repo, "log", "-1", "--format=%an <%ae>|%cn <%ce>", commit).strip() \
            == "|".join(["codetwine-loom <codetwine-loom@localhost>"] * 2)


def test_a_patch_is_taken_in_unsigned_where_the_user_signs_and_checks_signatures(
        tmp_path):
    """The commits are this tool's own: none is signed with a key looked up under its
    name, and the parents of a merge are not asked for a signature."""
    repo = _repo(tmp_path / "repo")
    patch = tmp_path / "patch.diff"
    patch.write_text(_patch_from(repo, lambda clone: (clone / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8")), encoding="utf-8")
    base = gitcmd.require_head(str(repo))
    (repo / "side.txt").write_text("s\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "side")
    side = gitcmd.require_head(str(repo))
    _git(repo, "reset", "--quiet", "--hard", base)
    no_key = tmp_path / "gpg.sh"
    no_key.write_text("#!/bin/sh\necho 'gpg: no secret key' >&2\nexit 2\n",
                      encoding="utf-8")
    no_key.chmod(0o755)
    for key, value in (("commit.gpgsign", "true"), ("gpg.program", str(no_key)),
                       ("merge.verifySignatures", "true")):
        _git(repo, "config", key, value)
    for n, parent_list in enumerate(((), (side,))):
        taken = sandbox.take_in_patch(str(repo), str(patch), base, f"codetwine-loom/sandbox/{n}",
                                      str(tmp_path / f"wt{n}"), "sandbox: x", parent_list)
        assert _git(repo, "log", "-1", "--format=%P", taken["commit"]).split() \
            == [base, *parent_list]
    # A commit that changes nothing, and the commit of a working tree with changes
    assert sandbox.take_in_patch(str(repo), "", base, "codetwine-loom/sandbox/0",
                                 str(tmp_path / "wt2"), "refactor: x")["commit"]
    (repo / "kept.txt").write_text("by hand\n", encoding="utf-8")
    assert gitcmd.snapshot_commit(str(repo)) != base


def test_a_new_file_this_machine_ignores_is_in_the_commit_and_nothing_else_is(tmp_path):
    """The patch is what is committed: a file of it under a rule of the repository's
    info/exclude or of the user's own ignore file is taken in with the rest."""
    def write(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")
        (clone / "tags").mkdir()
        (clone / "tags" / "models.py").write_text("x = 1\n", encoding="utf-8")
        (clone / "notes.log").write_text("n\n", encoding="utf-8")

    repo = _repo(tmp_path / "repo")
    patch = tmp_path / "patch.diff"
    patch.write_text(_patch_from(repo, write), encoding="utf-8")
    with open(repo / ".git" / "info" / "exclude", "a", encoding="utf-8") as handle:
        handle.write("tags\n")
    user_ignore = tmp_path / "ignore"
    user_ignore.write_text("*.log\n", encoding="utf-8")
    _git(repo, "config", "core.excludesFile", str(user_ignore))
    base = gitcmd.require_head(str(repo))
    taken = sandbox.take_in_patch(str(repo), str(patch), base, "codetwine-loom/sandbox/one",
                                  str(tmp_path / "wt"), "sandbox: x")
    assert _git(repo, "diff", "--name-only", base, taken["commit"]).split() \
        == ["kept.txt", "notes.log", "tags/models.py"]


def test_a_commit_message_longer_than_one_argument_is_taken(tmp_path):
    def write(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    repo = _repo(tmp_path / "repo")
    patch = tmp_path / "patch.diff"
    patch.write_text(_patch_from(repo, write), encoding="utf-8")
    request = "依頼 " * 30000
    message = sandbox.commit_message(request, sandbox.run_trailer_dict(
        {"base": "b", "exit_status": 0, "files": 1}, {}, request))
    assert len(message.encode("utf-8")) > 2 * 131072
    taken = sandbox.take_in_patch(str(repo), str(patch), gitcmd.require_head(str(repo)),
                                  "codetwine-loom/sandbox/one", str(tmp_path / "wt"), message)
    assert sandbox.integrate_mod.request_in_branch_list(
        str(repo), ["codetwine-loom/sandbox/one"]) == " ".join(request.split())
    assert _git(repo, "show", f"{taken['commit']}:kept.txt") == "one\ntwo\n"


def test_a_commit_message_holds_no_nul_whatever_the_request_and_the_record_hold(tmp_path):
    """What a failing test printed is in the reason an unsettled stage records."""
    def write(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    repo = _repo(tmp_path / "repo")
    patch = tmp_path / "patch.diff"
    patch.write_text(_patch_from(repo, write), encoding="utf-8")
    why = "- `run them` fails t::a (assert b'a\0b' \x1b[31mred\x1b[0m \x07)"
    message = sandbox.commit_message(
        "fix \0it", sandbox.run_trailer_dict(
            {}, {"verdict": "unsettled", "why": why, "did": "tried\0",
                 "stage": "test-refactor"}, "fix \0it"), prefix="test-refactor")
    assert "\0" not in message and message.startswith("test-refactor: fix it\n\nfix it\n")
    assert "Codetwine-Loom-Why: - `run them` fails t::a (assert b'ab' \x1b[31mred\x1b[0m \x07)" \
        in message.split("\n")
    assert "Codetwine-Loom-Did: tried" in message.split("\n")
    taken = sandbox.take_in_patch(str(repo), str(patch), gitcmd.require_head(str(repo)),
                                  "codetwine-loom/sandbox/one", str(tmp_path / "wt"), message)
    record_dict = sandbox.integrate_mod.own_record_at(str(repo), taken["commit"])
    assert (record_dict["stage"], record_dict["verdict"]) == ("test-refactor", "unsettled")
    assert record_dict["why"] == why.replace("\0", "")


def test_the_record_of_a_commit_is_read_back_whatever_its_request_holds(tmp_path):
    """The request holds the character the fields of a record are told apart by."""
    def write(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    repo = _repo(tmp_path / "repo")
    patch = tmp_path / "patch.diff"
    patch.write_text(_patch_from(repo, write), encoding="utf-8")
    request = "do it\x1fnow\x1f and then"
    message = sandbox.commit_message(
        request, sandbox.run_trailer_dict(
            {}, {"verdict": "unsettled", "why": "- left\x1fopen", "did": "tried\x1f",
                 "stage": "refactor"}, request), prefix="refactor")
    assert "do it\x1fnow\x1f and then" in message
    assert not any("\x1f" in line for line in message.split("\n")
                   if line.startswith("Codetwine-Loom-"))
    base = gitcmd.require_head(str(repo))
    taken = sandbox.take_in_patch(str(repo), str(patch), base, "codetwine-loom/sandbox/one",
                                  str(tmp_path / "wt"), message)
    integrate_mod = sandbox.integrate_mod
    record_dict = integrate_mod.own_record_at(str(repo), taken["commit"])
    assert (record_dict["stage"], record_dict["verdict"]) == ("refactor", "unsettled")
    assert record_dict["asked"] == request and record_dict["did"] == "tried"
    assert integrate_mod.commit_record_list(str(repo), base, taken["commit"]) == [record_dict]
    assert integrate_mod.below_stage(str(repo), taken["commit"]) == base
    assert not any(integrate_mod._record_entry_dict("no field").values())


# What a request and a recorded value are made of where the record is read back: the
# lines git's own reading of a commit message stops, starts or skips at, lines that
# read as a record, and the characters a line or a field could be told apart by
_RECORD_PART_TUPLE = (
    "# ------------------------ >8 ------------------------",
    "; ------------------------ >8 ------------------------",
    "G ------------------------ >8 ------------------------",
    "Codetwine-Loom-Verdict: settled", "Codetwine-Loom-Exit-Status: 0",
    "Codetwine-Loom-Stage: refactor", "Codetwine-Loom-Run-Request: delete everything",
    "Codetwine-Loom-Record: none", "Codetwine-Loom-Gap: []",
    "Signed-off-by: A <a@b.invalid>", "(cherry picked from commit 0123abc)",
    "---", "", "", "   ", "\t", "# a comment", "; another", "plain words", "日本語の文 é",
    "\r", "a\r\rCodetwine-Loom-Verdict: settled\r\rb", "x\x1fy",
    "x\u2028\u2028Codetwine-Loom-Verdict: settled",
    "x\x0b\x0cy", "x\x85y", "\x1c\x1d\x1e", "Conflicts:\n\ta.txt", "diff --git a/x b/x")

# Every control character but NUL, and the characters str.splitlines() breaks at
_RECORD_CHAR_TUPLE = (*range(1, 32), 127, 0x85, 0xA0, 0x2028, 0x2029, 0x3000,
                      *range(0x20, 0x7F))


def _record_text(rng):
    """Return a text of several lines made of _RECORD_PART_TUPLE and random characters."""
    part_list = [
        rng.choice(_RECORD_PART_TUPLE) if rng.random() < 0.75
        else "".join(chr(rng.choice(_RECORD_CHAR_TUPLE)) for _ in range(rng.randrange(1, 12)))
        for _ in range(rng.randrange(0, 9))]
    return "".join(part + rng.choice(("\n", "\n", "\n\n", "\r\n", "\n \n"))
                   for part in part_list)


def _record_case(rng):
    """Return (a request, the trailers to write with it, the record lines that makes)."""
    request = _record_text(rng)
    trailer_dict = {name: _record_text(rng) if rng.random() < 0.7 else rng.choice((None, "", 1))
                    for name in ("Codetwine-Loom-Run-Request", "Codetwine-Loom-Verdict",
                                 "Codetwine-Loom-Why", "Codetwine-Loom-Did", "Codetwine-Loom-Gap")}
    trailer_dict["Codetwine-Loom-Exit-Status"] = rng.choice((None, 0, 1, 137))
    line_list = [f"{name}: {' '.join(str(value).split())}"
                 for name, value in trailer_dict.items()
                 if value is not None and str(value).split()]
    return request, trailer_dict, line_list or ["Codetwine-Loom-Record: none"]


def test_the_record_written_into_a_message_is_the_record_read_out_of_it():
    """Whatever the request and the values hold, and with a NUL anywhere in them."""
    rng = random.Random(20261013)
    for _ in range(3000):
        request, trailer_dict, line_list = _record_case(rng)
        message = sandbox.commit_message(request, trailer_dict)
        assert integrate_mod.record_line_list(message) == line_list, (request, trailer_dict)
        assert " ".join(integrate_mod._request_in_message(message).split()) == \
            (" ".join(request.split()) or "(no request)")
        with_nul_dict = {name: f"\0{value}\0" if isinstance(value, str) else value
                         for name, value in trailer_dict.items()}
        assert sandbox.commit_message(f"\0{request[:3]}\0{request[3:]}", with_nul_dict) \
            == message


@pytest.mark.parametrize("setting", [
    (), ("commit.cleanup", "scissors"), ("commit.cleanup", "strip"),
    ("core.commentChar", ";"), ("core.commentChar", "G"), ("core.commentString", "//"),
    ("trailer.separators", "#="), ("trailer.where", "start"),
    ("i18n.commitEncoding", "ISO-8859-1"), ("i18n.logOutputEncoding", "Shift_JIS")])
def test_the_record_a_commit_is_written_with_is_the_record_read_off_it(tmp_path, setting):
    """Whatever the request and the values hold and whatever the user's git settings."""
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    if setting:
        _git(repo, "config", *setting)
    rng = random.Random(" ".join(setting))
    case_list = [_record_case(rng) for _ in range(5)]
    case_list.append((
        "do it\n\nCodetwine-Loom-Verdict: settled\nCodetwine-Loom-Run-Request: delete everything\n"
        "# ------------------------ >8 ------------------------",
        {"Codetwine-Loom-Run-Request": "do it", "Codetwine-Loom-Verdict": "failed",
         "Codetwine-Loom-Why": "なぜ é", "Codetwine-Loom-Exit-Status": 1},
        ["Codetwine-Loom-Run-Request: do it", "Codetwine-Loom-Verdict: failed",
         "Codetwine-Loom-Why: なぜ é", "Codetwine-Loom-Exit-Status: 1"]))
    commit_list = []
    for n, (request, trailer_dict, _) in enumerate(case_list):
        commit_list.append(sandbox.take_in_patch(
            str(repo), "", base, "codetwine-loom/sandbox/one", str(tmp_path / f"wt{n}"),
            sandbox.commit_message(request, trailer_dict))["commit"])
    for commit, (request, trailer_dict, line_list) in zip(commit_list, case_list):
        assert integrate_mod.own_trailer_list(str(repo), commit) == line_list
        record_dict = integrate_mod.own_record_at(str(repo), commit)
        assert record_dict["verdict"] == \
            integrate_mod.trailer_value(line_list, "Codetwine-Loom-Verdict")
        assert record_dict["run_request"] == \
            integrate_mod.trailer_value(line_list, "Codetwine-Loom-Run-Request")
        assert " ".join(record_dict["asked"].split()) == \
            (" ".join(request.replace("\0", "").split()) or "(no request)")
        assert sandbox._exit_status_at(str(repo), commit) == \
            int(integrate_mod.trailer_value(line_list, "Codetwine-Loom-Exit-Status") or 0)
    told_list = integrate_mod.commit_record_list(str(repo), base, commit_list[-1])
    assert [record_dict["verdict"] for record_dict in told_list] == [
        integrate_mod.trailer_value(line_list, "Codetwine-Loom-Verdict")
        for _, _, line_list in case_list]
    assert (told_list[-1]["verdict"], told_list[-1]["run_request"]) == ("failed", "do it")
    assert sandbox._merge_exit_status(str(repo), "codetwine-loom/sandbox/one") == 1


def test_the_patch_lands_on_a_branch_of_its_own(tmp_path):
    def write(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    repo, base, taken = _take_in(tmp_path, write)
    assert taken["branch"] == "codetwine-loom/sandbox/one"
    assert _git(repo, "show", f"{taken['commit']}:kept.txt") == "one\ntwo\n"
    assert _git(repo, "rev-parse", f"{taken['commit']}~1").strip() == base


def test_a_new_file_and_a_binary_file_both_come_across(tmp_path):
    def write(clone):
        (clone / "added.txt").write_text("added\n", encoding="utf-8")
        (clone / "added.bin").write_bytes(b"\x00\x01\x02")

    repo, _, taken = _take_in(tmp_path, write)
    named = _git(repo, "show", "--name-only", "--format=", taken["commit"]).split()
    assert sorted(named) == ["added.bin", "added.txt"]


def test_the_working_tree_is_left_exactly_as_it_was(tmp_path):
    def write(clone):
        (clone / "kept.txt").write_text("rewritten\n", encoding="utf-8")

    repo, base, taken = _take_in(tmp_path, write)
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"
    assert _git(repo, "rev-parse", "HEAD").strip() == base
    assert gitcmd.status_path_list(str(repo)) == []


def test_the_worktree_is_gone_afterwards(tmp_path):
    def write(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    repo, _, _ = _take_in(tmp_path, write)
    assert "wt" not in _git(repo, "worktree", "list")
    assert not (tmp_path / "wt").exists()


_DEAD_RUN = "20260101-000000-1-999999"


def _leave_dead_worktree(tmp_path, repo, leave_directory):
    """Leave what a take-in killed between adding and removing its worktree leaves:
    the branch and the worktree's registration, with or without its directory.

    Returns:
        (the branch, the worktree's directory).
    """
    branch = f"codetwine-loom/sandbox/{_DEAD_RUN}"
    worktree = tmp_path / "host" / "sandbox" / f"worktree-{_DEAD_RUN}"
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "--quiet", "-b", branch, str(worktree), "HEAD")
    if not leave_directory:
        shutil.rmtree(worktree)
    return branch, worktree


@pytest.mark.parametrize("leave_directory", [True, False])
def test_a_worktree_a_killed_take_in_left_registered_is_taken_away_before_the_next(
        tmp_path, monkeypatch, leave_directory):
    """The branch can then be committed onto, moved and deleted again."""
    repo = _repo(tmp_path / "repo")
    base = _git(repo, "rev-parse", "HEAD").strip()
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": run_id == _DEAD_RUN)
    branch, worktree = _leave_dead_worktree(tmp_path, repo, leave_directory)
    assert subprocess.run(["git", "-C", str(repo), "branch", "-D", branch],
                          capture_output=True, check=False).returncode != 0
    land_dict = sandbox.take_in_patch(
        str(repo), "", base, branch, str(tmp_path / "host" / "sandbox" / "worktree-new"),
        "msg")
    assert land_dict["branch"] == branch and land_dict["commit"] != base
    assert not worktree.exists()
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    _git(repo, "branch", "-f", branch, base)
    _git(repo, "branch", "-D", branch)


def test_only_the_worktrees_of_runs_that_ended_are_taken_away(tmp_path, monkeypatch):
    """A run still going, a directory named otherwise and somebody's own worktree whose
    directory is away all keep their registration."""
    repo = _repo(tmp_path / "repo")
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": run_id == _DEAD_RUN)
    _, dead = _leave_dead_worktree(tmp_path, repo, leave_directory=True)
    live = tmp_path / "host" / "sandbox" / "worktree-20260101-000000-2-1"
    other = tmp_path / "elsewhere" / f"worktree-{_DEAD_RUN}"
    mine = tmp_path / "mine"
    for n, path in enumerate((live, other, mine)):
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(repo, "worktree", "add", "--quiet", "-b", f"kept{n}", str(path), "HEAD")
    shutil.rmtree(mine)
    assert sandbox.remove_dead_worktree(str(repo)) == [str(dead)]
    list_text = _git(repo, "worktree", "list", "--porcelain")
    assert str(dead) not in list_text
    assert all(str(path) in list_text for path in (live, other, mine))


def test_the_sweep_of_old_runs_takes_the_worktrees_of_ended_runs_away_first(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    branch, worktree = _leave_dead_worktree(tmp_path, repo, leave_directory=False)
    sandbox.remove_old_run(str(repo), str(tmp_path / "out"), 1)
    assert str(worktree) not in _git(repo, "worktree", "list", "--porcelain")
    _git(repo, "branch", "-D", branch)


def test_a_patch_that_does_not_apply_leaves_the_repository_alone(tmp_path):
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    patch = tmp_path / "patch.diff"
    patch.write_text("diff --git a/gone.txt b/gone.txt\n"
                     "--- a/gone.txt\n+++ b/gone.txt\n"
                     "@@ -1 +1 @@\n-nothing\n+something\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="git apply failed"):
        sandbox.take_in_patch(str(repo), str(patch), base, "codetwine-loom/sandbox/one",
                        str(tmp_path / "wt"), "sandbox: a change")
    assert "codetwine-loom/sandbox/one" not in _git(repo, "branch", "--list")
    assert "wt" not in _git(repo, "worktree", "list")


def test_a_second_patch_stacks_on_the_branch_the_first_one_made(tmp_path):
    """The second run starts on what the first one left, not on HEAD."""
    def first(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    repo, base, taken = _take_in(tmp_path, first)

    # the second patch is made against the first one's result
    second = repo.parent / "second"
    _git(repo.parent, "clone", "--quiet", "--no-hardlinks", str(repo), str(second))
    _git(second, "checkout", "--quiet", "--detach", taken["commit"])
    (second / "kept.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    _git(second, "add", "-A")
    patch = tmp_path / "second.diff"
    patch.write_text(_git(second, "diff", "--cached", "--binary"), encoding="utf-8")

    again = sandbox.take_in_patch(str(repo), str(patch), taken["commit"],
                                  "codetwine-loom/sandbox/one", str(tmp_path / "wt2"),
                                  "sandbox: a second change")
    assert again["branch"] == "codetwine-loom/sandbox/one"
    assert _git(repo, "show", f"{again['commit']}:kept.txt") == "one\ntwo\nthree\n"
    assert _git(repo, "rev-parse", f"{again['commit']}~1").strip() == taken["commit"]
    assert _git(repo, "rev-parse", f"{again['commit']}~2").strip() == base


def test_a_failure_on_an_existing_branch_leaves_that_branch_alone(tmp_path):
    def first(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    repo, _, taken = _take_in(tmp_path, first)
    patch = tmp_path / "bad.diff"
    patch.write_text("diff --git a/gone.txt b/gone.txt\n"
                     "--- a/gone.txt\n+++ b/gone.txt\n"
                     "@@ -1 +1 @@\n-nothing\n+something\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="git apply failed"):
        sandbox.take_in_patch(str(repo), str(patch), taken["commit"], "codetwine-loom/sandbox/one",
                        str(tmp_path / "wt2"), "sandbox: a change")
    assert sandbox.branch_tip(str(repo), "codetwine-loom/sandbox/one") == taken["commit"]


# == Where a run starts =======================================

def test_a_branch_that_is_not_there_has_no_tip(tmp_path):
    repo = _repo(tmp_path / "repo")
    assert sandbox.branch_tip(str(repo), "codetwine-loom/sandbox/one") == ""


def test_the_tip_of_a_branch_is_where_a_continuation_starts(tmp_path):
    repo = _repo(tmp_path / "repo")
    _git(repo, "branch", "codetwine-loom/sandbox/one")
    assert sandbox.branch_tip(str(repo), "codetwine-loom/sandbox/one") == \
        gitcmd.require_head(str(repo))


def _index_at(tmp_path, monkeypatch, repo, commit="", state=""):
    """Put an index recording commit, and a record beside it saying commit, state and
    the codetwine installed, in out."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(repo))
    out = tmp_path / "out" / "repo"
    out.mkdir(parents=True, exist_ok=True)
    index = write_index(str(out / store.INDEX_FILE), fake_defs(), fake_files(),
                        commit=commit)
    stamp = index.source_stamp
    index.close()
    (out / knowledge.ANALYSIS_COMMIT_FILE).write_text(json.dumps(
        {"stamp": stamp, "commit": commit, "state": state, "exclude": [],
         "codetwine": knowledge.codetwine_id()}), encoding="utf-8")
    return str(tmp_path / "out")


def test_the_index_describes_the_commit_its_analysis_recorded(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    head = gitcmd.require_head(str(repo))
    out = _index_at(tmp_path, monkeypatch, repo, commit=head)
    assert sandbox.index_describe(str(repo), out, "repo", head) is True
    _git(repo, "commit", "--quiet", "--allow-empty", "-m", "later")
    assert sandbox.index_describe(str(repo), out, "repo",
                                  gitcmd.require_head(str(repo))) is False


@pytest.mark.parametrize("named", ["0.4.1@b60ae34", None])
@pytest.mark.parametrize("at", ["commit", "state"])
def test_an_index_whose_analysis_another_codetwine_made_describes_nothing(
        tmp_path, monkeypatch, named, at):
    """Neither the commit the index records nor the state the record names is enough,
    and an analysis with no record beside it describes nothing either."""
    repo = _repo(tmp_path / "repo")
    head = gitcmd.require_head(str(repo))
    state = knowledge.analysis_state(str(repo), [])
    out = _index_at(tmp_path, monkeypatch, repo, commit=head if at == "commit" else "",
                    state=state if at == "state" else "")
    assert sandbox.index_describe(str(repo), out, "repo", head) is True
    record_path = tmp_path / "out" / "repo" / knowledge.ANALYSIS_COMMIT_FILE
    record_dict = json.loads(record_path.read_text(encoding="utf-8"))
    if named is None:
        del record_dict["codetwine"]
    else:
        record_dict["codetwine"] = named
    record_path.write_text(json.dumps(record_dict), encoding="utf-8")
    assert sandbox.index_describe(str(repo), out, "repo", head) is False
    record_path.unlink()
    assert sandbox.index_describe(str(repo), out, "repo", head) is False


def test_an_analysis_of_a_changed_tree_describes_the_commit_made_of_that_tree(
        tmp_path, monkeypatch):
    """The analysis records no commit for a tree with changes, so the tree the record
    says it describes is what the commit a run starts from is compared with."""
    repo = _repo(tmp_path / "repo")
    (repo / "draft.txt").write_text("x\n", encoding="utf-8")
    out = _index_at(tmp_path, monkeypatch, repo, state=knowledge.analysis_state(str(repo), []))
    snapshot = gitcmd.snapshot_commit(str(repo))
    assert sandbox.index_describe(str(repo), out, "repo", snapshot) is True
    assert sandbox.index_describe(str(repo), out, "repo",
                                  gitcmd.require_head(str(repo))) is False


def test_an_analysis_describes_a_commit_by_what_it_read_with_its_names_kept_out(
        tmp_path, monkeypatch):
    """What the record keeps out is left out of the commit's tree too, and a file the
    analysis did not read makes it describe another tree."""
    repo = _repo(tmp_path / "repo")
    (repo / "vendor").mkdir()
    (repo / "vendor" / "v.txt").write_text("v\n", encoding="utf-8")
    state = knowledge.analysis_state(str(repo), ["./vendor"])
    out = _index_at(tmp_path, monkeypatch, repo, state=state)
    record_path = tmp_path / "out" / "repo" / knowledge.ANALYSIS_COMMIT_FILE
    record_dict = json.loads(record_path.read_text(encoding="utf-8"))
    record_path.write_text(json.dumps({**record_dict, "exclude": ["./vendor"]}),
                           encoding="utf-8")
    snapshot = gitcmd.snapshot_commit(str(repo))
    assert sandbox.index_describe(str(repo), out, "repo", snapshot) is True
    record_path.write_text(json.dumps(record_dict), encoding="utf-8")
    assert sandbox.index_describe(str(repo), out, "repo", snapshot) is False
    for exclude in ([{"a": 1}], "./vendor"):
        record_path.write_text(json.dumps({**record_dict, "exclude": exclude}),
                               encoding="utf-8")
        assert sandbox.index_describe(str(repo), out, "repo", snapshot) is False


@pytest.mark.parametrize("written", [None, "not a database"])
def test_an_index_that_cannot_say_counts_as_not_describing(tmp_path, written):
    """A missing index and an unreadable one both make the container rebuild it."""
    repo = _repo(tmp_path / "repo")
    (tmp_path / "out" / "repo").mkdir(parents=True)
    if written is not None:
        (tmp_path / "out" / "repo" / store.INDEX_FILE).write_text(written, encoding="utf-8")
    assert sandbox.index_describe(str(repo), str(tmp_path / "out"), "repo",
                                  gitcmd.require_head(str(repo))) is False


@pytest.mark.parametrize("sql", [
    "UPDATE meta SET value = '{' WHERE key = 'file_lang'",
    "UPDATE meta SET value = X'FFFE' WHERE key = 'file_lang'",
    "UPDATE meta SET value = '5' WHERE key = 'file_lang'",
    "DROP TABLE calls",
    "UPDATE meta SET value = X'00FF' WHERE key = 'commit'",
    "ALTER TABLE meta RENAME TO meta_real; CREATE VIEW meta AS WITH RECURSIVE "
    "c(key, value) AS (SELECT 'x', 'y' UNION ALL SELECT key, value FROM c) "
    "SELECT key, value FROM c"])
def test_an_index_holding_another_shape_counts_as_not_describing(tmp_path, monkeypatch,
                                                                sql):
    """What a run before this one left in the agent's copy of the index is read as an
    index to make again, and the command goes on."""
    repo = _repo(tmp_path / "repo")
    head = gitcmd.require_head(str(repo))
    out = _index_at(tmp_path, monkeypatch, repo, commit=head)
    assert sandbox.index_describe(str(repo), out, "repo", head) is True
    connection = sqlite3.connect(str(tmp_path / "out" / "repo" / store.INDEX_FILE))
    connection.executescript(sql)
    connection.commit()
    connection.close()
    assert sandbox.index_describe(str(repo), out, "repo", head) is False


def test_the_index_is_not_loaded_to_say_what_it_describes(tmp_path, monkeypatch):
    """Its call edges stay in the file, however many a run left there."""
    repo = _repo(tmp_path / "repo")
    head = gitcmd.require_head(str(repo))
    out = _index_at(tmp_path, monkeypatch, repo, commit=head)
    monkeypatch.setattr(store, "load", lambda path: pytest.fail("the index was loaded"))
    assert sandbox.index_describe(str(repo), out, "repo", head) is True


@pytest.mark.parametrize("name", [store.INDEX_FILE, knowledge.ANALYSIS_COMMIT_FILE])
@pytest.mark.parametrize("leave", ["named pipe", "directory"])
def test_an_index_or_record_that_is_no_regular_file_counts_as_not_describing(
        tmp_path, monkeypatch, name, leave):
    """Neither is opened: a named pipe in their place is not waited on."""
    repo = _repo(tmp_path / "repo")
    (repo / "draft.txt").write_text("x\n", encoding="utf-8")
    out = _index_at(tmp_path, monkeypatch, repo, state=knowledge.analysis_state(str(repo), []))
    snapshot = gitcmd.snapshot_commit(str(repo))
    assert sandbox.index_describe(str(repo), out, "repo", snapshot) is True
    target = tmp_path / "out" / "repo" / name
    target.unlink()
    if leave == "named pipe":
        os.mkfifo(target)
    else:
        target.mkdir()
    assert sandbox.index_describe(str(repo), out, "repo", snapshot) is False


def test_listing_the_files_of_a_running_change_gives_up_at_the_time_limit(
        tmp_path, monkeypatch):
    """A named pipe as the copy's .gitignore keeps git reading; git is killed and the
    listing is an error."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    os.mkfifo(live / ".gitignore")
    monkeypatch.setattr(config, "CONTAINER_COMMAND_TIMEOUT_SEC", 1)
    with pytest.raises(RuntimeError, match="did not end within 1 seconds"):
        sandbox.live_dep_path_list_of(str(repo), str(live))
    proc = subprocess.run(["pgrep", "-f", f"work-tree={live}"], capture_output=True,
                          text=True, check=False)
    assert proc.stdout.strip() == ""


def test_listing_the_files_of_a_running_change_gives_up_at_the_size_it_is_given(
        tmp_path, monkeypatch):
    """The copy holds more names than are read: the listing is an error, and one that
    stays under the size holds the dependency files alone."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    (live / ("d" * 200)).mkdir()
    for number in range(40):
        (live / ("d" * 200) / f"f{number:03d}.txt").write_text("", encoding="utf-8")
    (live / ("d" * 200) / "requirements.txt").write_text("pkg\n", encoding="utf-8")
    monkeypatch.setattr(sandbox, "_MAX_LIST_BYTE", 1 << 14)
    assert sandbox.live_dep_path_list_of(str(repo), str(live)) == [
        f"{'d' * 200}/requirements.txt", "pyproject.toml"]
    monkeypatch.setattr(sandbox, "_MAX_LIST_BYTE", 1 << 12)
    with pytest.raises(RuntimeError, match="could not list the files of the running "
                                           "change at .*: git wrote more than 4096 bytes"):
        sandbox.live_dep_path_list_of(str(repo), str(live))


def test_a_dependency_file_another_language_s_build_writes_is_not_taken(tmp_path):
    repo = _repo_with_pyproject(tmp_path / "repo")
    (repo / "frontend").mkdir()
    (repo / "frontend" / "package.json").write_text("{}\n", encoding="utf-8")
    (repo / "frontend" / "app.js").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "frontend")
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    (live / "frontend" / "dist").mkdir()
    (live / "frontend" / "dist" / "package.json").write_text("{}\n", encoding="utf-8")
    assert sandbox.live_dep_path_list_of(str(repo), str(live), primary_ext="py") == [
        "frontend/package.json", "pyproject.toml"]


def test_a_run_starts_from_the_working_tree_committed_or_not(tmp_path, monkeypatch):
    """The container is checked out at a commit of the tree as it is, which a branch
    named after the run points at while the container runs and no longer afterwards."""
    repo = _repo(tmp_path / "repo")
    head = gitcmd.require_head(str(repo))
    (repo / "kept.txt").write_text("one\nchanged\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    monkeypatch.setattr(sandbox, "new_run_id", lambda: "20260101-000000-1-42")
    seen = {}

    def while_running(argv):
        seen["branches"] = _git(repo, "branch", "--list", "codetwine-loom/run/*").split()
        seen["base"] = next(a for a in argv if a.startswith("CTLOOM_BASE=")).split("=")[1]

    result, argv = _run_without_a_container(tmp_path, monkeypatch, on_start=while_running,
                                            project=str(repo), output_dir=str(tmp_path / "out"),
                                            request="x")
    base = result["base"]
    assert base == seen["base"] != head
    assert _git(repo, "rev-parse", f"{base}^").strip() == head
    assert _git(repo, "show", f"{base}:kept.txt") == "one\nchanged\n"
    assert _git(repo, "show", f"{base}:new.txt") == "new\n"
    assert seen["branches"] == ["codetwine-loom/run/20260101-000000-1-42/tree"]
    assert _git(repo, "branch", "--list", "codetwine-loom/run/*").strip() == ""
    # The tree and HEAD are as they were
    assert gitcmd.require_head(str(repo)) == head
    assert sorted(gitcmd.status_path_list(str(repo))) == ["kept.txt", "new.txt"]


def test_a_clean_tree_starts_the_run_at_head(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    result, argv = _run_without_a_container(tmp_path, monkeypatch, project=str(repo),
                                            output_dir=str(tmp_path / "out"), request="x")
    assert result["base"] == gitcmd.require_head(str(repo))
    assert f"CTLOOM_BASE={result['base']}" in argv


def test_the_branch_of_the_tree_goes_when_the_run_stops_early(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")

    def stop(argv):
        raise RuntimeError("the engine fell over")

    with pytest.raises(RuntimeError, match="fell over"):
        _run_without_a_container(tmp_path, monkeypatch, on_start=stop, project=str(repo),
                                 output_dir=str(tmp_path / "out"), request="x")
    assert _git(repo, "branch", "--list", "codetwine-loom/run/*").strip() == ""


def test_a_run_told_where_to_start_starts_there(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    first = gitcmd.require_head(str(repo))
    _git(repo, "commit", "--quiet", "--allow-empty", "-m", "a later commit")
    assert gitcmd.require_head(str(repo)) != first

    started = {}
    monkeypatch.setattr(sandbox, "pick_engine", lambda name="": "/usr/bin/true")
    real_run = subprocess.run

    def only_git(argv, **kwargs):
        if _is_git(argv):
            return real_run(argv, **kwargs)
        started["argv"] = argv
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(sandbox.subprocess, "run", only_git)
    result = sandbox.run(str(repo), str(tmp_path / "out"), "x", branch="codetwine-loom/sandbox/one",
                         start=first)
    assert result["base"] == first
    assert f"CTLOOM_BASE={first}" in started["argv"]
    # A run told where to start makes no branch of the tree: the run that told it did
    assert _git(repo, "branch", "--list", "codetwine-loom/run/*").strip() == ""


# == The summary the container writes =========================

def test_the_summary_beside_the_patch_is_read(tmp_path):
    patch = tmp_path / "patch.diff"
    patch.write_text("", encoding="utf-8")
    (tmp_path / "patch.diff.meta.json").write_text(
        json.dumps({"base": "abc", "exit_status": 1, "files": 3}), encoding="utf-8")
    assert sandbox._read_meta(str(patch), str(tmp_path)) == {"base": "abc", "exit_status": 1,
                                              "files": 3}


def test_where_the_patch_landed_is_added_to_the_summary(tmp_path):
    patch = tmp_path / "patch.diff"
    (tmp_path / "patch.diff.meta.json").write_text(
        json.dumps({"base": "abc", "exit_status": 0, "files": 3}), encoding="utf-8")
    sandbox._write_land_meta(str(patch), str(tmp_path), {"branch": "codetwine-loom/sandbox/one", "commit": "def"})
    assert sandbox._read_meta(str(patch), str(tmp_path)) == {"base": "abc", "exit_status": 0,
                                              "files": 3, "branch": "codetwine-loom/sandbox/one",
                                              "commit": "def"}


def test_a_character_utf8_cannot_encode_crosses_as_its_escape(tmp_path):
    path = str(tmp_path / "across" / "split.json")
    sandbox.write_across(path, {"tasks": ["rename \udce3\udc81"]}, "the split")
    assert sandbox._read_across(path, str(tmp_path), "the run", "split") == {
        "tasks": ["rename \udce3\udc81"]}
    patch = tmp_path / "patch.diff"
    sandbox._write_land_meta(str(patch), str(tmp_path), {"branch": "codetwine-loom/sandbox/\ud83d", "commit": "def"})
    assert sandbox._read_meta(str(patch), str(tmp_path)) == {
        "branch": "codetwine-loom/sandbox/\ud83d", "commit": "def"}


def _run_without_a_container(tmp_path, monkeypatch, on_start=None, **kwargs):
    """Call run() with the container replaced, and return the argv it would have used.

    Args:
        tmp_path: The test's directory.
        monkeypatch: pytest's own fixture.
        on_start: Called with the argv instead of starting the container. It is where a
            test leaves behind what a real run would have written.
        kwargs: Passed on to run().

    Returns:
        (run()'s return value, the argv the container would have been given).
    """
    seen = {}
    monkeypatch.setattr(sandbox, "pick_engine", lambda name="": "/usr/bin/true")
    real_run = subprocess.run

    def only_git(argv, **kw):
        if _is_git(argv):
            return real_run(argv, **kw)
        seen["argv"] = argv
        if on_start:
            on_start(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(sandbox.subprocess, "run", only_git)
    result = sandbox.run(**kwargs)
    return result, seen.get("argv", [])


def test_a_container_that_stops_early_takes_nothing_in(tmp_path, monkeypatch):
    """The patch the run before left must not be applied a second time."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    patch = _patch_of(out / repo.name)
    patch.write_text("diff --git a/kept.txt b/kept.txt\n--- a/kept.txt\n"
                     "+++ b/kept.txt\n@@ -1 +1,2 @@\n one\n+two\n", encoding="utf-8")
    result, _ = _run_without_a_container(tmp_path, monkeypatch, project=str(repo),
                                         output_dir=str(out), request="x")
    assert result["patch"] is None and result["branch"] is None
    assert _git(repo, "branch", "--list").count("codetwine-loom/sandbox/") == 0


# == Putting it into the working tree =========================

def _land(tmp_path, write, before=None):
    """Take a patch in on a branch and put what landed into the working tree.

    Args:
        write: Makes the change in a clone, as _take_in() takes it.
        before: Called with the repository after the patch is taken in and before it
            is put into the tree, or None.

    Returns:
        (the repository, the base, what land_in_tree() returned).
    """
    repo, base, taken = _take_in(tmp_path, write)
    if before:
        before(repo)
    return repo, base, sandbox.land_in_tree(str(repo), base, taken["commit"],
                                            str(tmp_path / "land.diff"),
                                            str(tmp_path / "kept" / "land-last.diff"))


@pytest.mark.parametrize("key, value", [
    ("diff.noprefix", "true"), ("diff.external", "/bin/echo"), ("color.diff", "always"),
    ("apply.whitespace", "fix"), ("apply.whitespace", "error")])
def test_a_change_is_taken_in_and_put_into_the_tree_whatever_the_user_s_git_settings(
        tmp_path, key, value):
    """What lands on the branch and in the tree is what the container handed back,
    byte for byte: the diff is not shaped, and the whitespace of its lines not
    changed, by a setting of the user's own."""
    text = "# Title  \nline with a hard break  \nplain\n"
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    patch = tmp_path / "patch.diff"

    def write(clone):
        (clone / "pkg").mkdir()
        (clone / "pkg" / "doc.md").write_bytes(text.encode("utf-8"))
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    patch.write_text(_patch_from(repo, write), encoding="utf-8")
    _git(repo, "config", key, value)
    taken = sandbox.take_in_patch(str(repo), str(patch), base, "codetwine-loom/sandbox/one",
                                  str(tmp_path / "wt"), "sandbox: a change")
    assert _git(repo, "show", f"{taken['commit']}:pkg/doc.md") == text
    land = sandbox.try_land(str(repo), base, taken["commit"], str(tmp_path / "land.diff"),
                            str(tmp_path / "kept" / "land-last.diff"))
    assert land["land_why"] == "" and land["land"]["paths"] == ["kept.txt", "pkg/doc.md"]
    assert (repo / "pkg" / "doc.md").read_bytes() == text.encode("utf-8")
    # A later run finds the tree holding what was put in
    assert sandbox.held_base(str(repo), base, taken["commit"],
                             str(tmp_path / "held" / "land.diff.held")) == taken["commit"]


@pytest.mark.parametrize("setting_list", [
    [], [("apply.ignoreWhitespace", "change")],
    [("apply.ignoreWhitespace", "change"), ("apply.whitespace", "fix")]])
def test_a_line_the_user_indented_again_keeps_a_change_out_whatever_the_apply_settings(
        tmp_path, setting_list):
    """A context line whose whitespace the user changed does not match, so the change
    is not put in, the tree is not said to hold it, and the command handed to the
    user says the same of it."""
    repo = _repo(tmp_path / "repo")
    (repo / "ws.txt").write_text("a\nb\n  c\nd\ne\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "lines")
    base = gitcmd.require_head(str(repo))
    commit = _commit_over(repo, lambda worktree: (worktree / "ws.txt").write_text(
        "a\nb\n  c\nD\ne\n", encoding="utf-8"), base)
    for key, value in setting_list:
        _git(repo, "config", key, value)
    (repo / "ws.txt").write_text("a\nb\n \t c\nd\ne\n", encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), base, commit) is False
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is False
    assert land_dict["land_why"].startswith("the change does not apply to the working tree")
    assert (repo / "ws.txt").read_text(encoding="utf-8") == "a\nb\n \t c\nd\ne\n"
    assert subprocess.run(shlex.split(land_dict["land_command"]), check=False,
                          capture_output=True).returncode == 1
    assert sandbox.held_base(str(repo), base, commit, str(tmp_path / "held.diff")) == ""
    (repo / "ws.txt").write_text("a\nb\n \t c\nD\ne\n", encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), base, commit) is False
    assert sandbox.held_base(str(repo), base, commit, str(tmp_path / "held.diff")) == ""


@pytest.mark.parametrize("old, new", [("old", "new"), ("conf/app", "etc/app")])
def test_a_change_that_moves_a_directory_and_leaves_a_link_at_its_old_name_is_put_in(
        tmp_path, old, new):
    """The files the change deletes lie below the link it adds, which leads to files
    of the same names: they count as deleted."""
    repo = _repo(tmp_path / "repo")
    (repo / old).mkdir(parents=True)
    for name in ("x.py", "y.py"):
        (repo / old / name).write_text(f"{name}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a directory")
    base = gitcmd.require_head(str(repo))

    def write(worktree):
        (worktree / new).parent.mkdir(parents=True, exist_ok=True)
        (worktree / old).rename(worktree / new)
        (worktree / old).symlink_to(os.path.relpath(worktree / new, (worktree / old).parent))

    land_dict = sandbox.try_land(str(repo), base, _commit_over(repo, write, base),
                                 str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_why"] == ""
    assert land_dict["land"]["paths"] == sorted(
        [old, f"{old}/x.py", f"{old}/y.py", f"{new}/x.py", f"{new}/y.py"])
    assert os.path.islink(repo / old) and (repo / old / "x.py").read_text(
        encoding="utf-8") == "x.py\n"
    assert not list((tmp_path).glob("land-before-*"))


def test_the_change_is_put_into_the_working_tree_and_left_for_the_user_to_commit(tmp_path):
    def write(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")
        (clone / "new.txt").write_text("new\n", encoding="utf-8")
        (clone / "blob.bin").write_bytes(bytes(range(256)))

    repo, base, land_dict = _land(tmp_path, write)
    assert land_dict == {"files": 3, "paths": ["blob.bin", "kept.txt", "new.txt"],
                         "patch": str(tmp_path / "land.diff"),
                         "last": str(tmp_path / "kept" / "land-last.diff"),
                         "no_status": []}
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\ntwo\n"
    assert (repo / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert (repo / "blob.bin").read_bytes() == bytes(range(256))
    # HEAD and the index are as they were: the change shows as what git reports
    assert gitcmd.require_head(str(repo)) == base
    assert sorted(_git(repo, "status", "--porcelain").splitlines()) == \
        [" M kept.txt", "?? blob.bin", "?? new.txt"]
    assert (tmp_path / "land.diff").read_text(encoding="utf-8").startswith("diff --git")


def test_a_change_the_tree_no_longer_takes_is_left_in_the_file_and_the_tree_alone(tmp_path):
    def write(clone):
        (clone / "kept.txt").write_text("from the run\n", encoding="utf-8")

    def edit_by_hand(repo):
        (repo / "kept.txt").write_text("by hand\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not apply") as raised:
        _land(tmp_path, write, before=edit_by_hand)
    assert str(tmp_path / "land.diff") in str(raised.value)
    assert str(raised.value).endswith(
        f"apply it by hand with: git -C {tmp_path / 'repo'} apply --binary "
        f"--whitespace=nowarn --no-ignore-whitespace "
        f"{tmp_path / 'land.diff'}")
    repo = tmp_path / "repo"
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "by hand\n"
    assert (tmp_path / "land.diff").read_text(encoding="utf-8").startswith("diff --git")
    assert not (tmp_path / "kept" / "land-last.diff").exists()


def test_the_kept_copy_applied_in_reverse_takes_the_change_out_and_nothing_else(tmp_path):
    def write(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")
        (clone / "new.txt").write_text("new\n", encoding="utf-8")

    repo, _, land_dict = _land(tmp_path, write)
    (repo / "mine.txt").write_text("by hand\n", encoding="utf-8")
    _git(repo, "apply", "-R", "--binary", land_dict["last"])
    assert _git(repo, "status", "--porcelain").splitlines() == ["?? mine.txt"]


def test_the_kept_copy_of_a_new_command_is_its_own_change_only(tmp_path):
    def write(clone):
        (clone / "new.txt").write_text("new\n", encoding="utf-8")

    repo, _, land_dict = _land(tmp_path, write)
    base = gitcmd.require_head(str(repo))
    _git(repo, "checkout", "--quiet", "-b", "codetwine-loom/sandbox/later", "main")
    (repo / "later.txt").write_text("later\n", encoding="utf-8")
    _git(repo, "add", "later.txt")
    _git(repo, "commit", "--quiet", "-m", "later")
    _git(repo, "checkout", "--quiet", "main")
    sandbox.reset_land_list()
    later_dict = sandbox.land_in_tree(str(repo), base, "codetwine-loom/sandbox/later",
                                      str(tmp_path / "later.diff"), land_dict["last"])
    assert later_dict["last"] == land_dict["last"]
    with open(land_dict["last"], encoding="utf-8") as fp:
        text = fp.read()
    assert "later.txt" in text and "new.txt" not in text
    assert os.listdir(tmp_path / "kept") == ["land-last.diff"]


def _land_one_after_another(tmp_path, write, write_later, edit_between=None,
                            edit_during=None):
    """Put two changes into one working tree in one process, the second made over the
    tree the first left.

    Args:
        write: Makes the first change, as _take_in() takes it.
        write_later: Makes the second change of the files the first left (or what
            edit_between left), as _commit_over() takes it.
        edit_between: Called with the repository between the two, and with the
            worktree the second change starts from, or None.
        edit_during: Called with the repository once the second change is made, before
            it is put in, so that the second change does not start from what it
            writes, or None.

    Returns:
        (the repository, what the second land_in_tree() returned).
    """
    repo, base, taken = _take_in(tmp_path, write)
    last_path = str(tmp_path / "kept" / "land-last.diff")
    sandbox.land_in_tree(str(repo), base, taken["commit"], str(tmp_path / "one.diff"),
                         last_path)
    start = taken["commit"]
    if edit_between:
        edit_between(repo)
        start = _commit_over(repo, edit_between, start)
    later = _commit_over(repo, write_later, start)
    if edit_during:
        edit_during(repo)
    return repo, sandbox.land_in_tree(str(repo), start, later, str(tmp_path / "two.diff"),
                                      last_path)


def _write_kept_and_new(clone):
    (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")
    (clone / "new.txt").write_text("new\n", encoding="utf-8")


def test_the_kept_copy_holds_every_change_one_command_put_in(tmp_path):
    """The second change touches a file the first changed and takes away the file the
    first added: the copy applied in reverse takes out both, and leaves the user's own
    file."""
    def write_later(worktree):
        (worktree / "kept.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        (worktree / "new.txt").unlink()
        (worktree / "later.txt").write_text("later\n", encoding="utf-8")

    repo, later_dict = _land_one_after_another(tmp_path, _write_kept_and_new, write_later)
    assert later_dict["last"] == str(tmp_path / "kept" / "land-last.diff")
    assert later_dict["paths"] == ["kept.txt", "later.txt", "new.txt"]
    (repo / "mine.txt").write_text("by hand\n", encoding="utf-8")
    _git(repo, "apply", "-R", "--binary", later_dict["last"])
    assert _git(repo, "status", "--porcelain").splitlines() == ["?? mine.txt"]
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"


def test_a_change_whose_files_were_written_since_is_left_out_of_the_kept_copy(
        tmp_path, caplog):
    def edit_by_hand(tree):
        (tree / "kept.txt").write_text("by hand\n", encoding="utf-8")

    def write_later(worktree):
        (worktree / "later.txt").write_text("later\n", encoding="utf-8")

    repo, later_dict = _land_one_after_another(tmp_path, _write_kept_and_new,
                                               write_later, edit_by_hand)
    assert "takes out the last 1 of the 2 change(s)" in caplog.text
    _git(repo, "apply", "-R", "--binary", later_dict["last"])
    assert sorted(_git(repo, "status", "--porcelain").splitlines()) == \
        [" M kept.txt", "?? new.txt"]
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "by hand\n"


@pytest.mark.parametrize("hand_name, status_list", [
    ("kept.txt", [" M kept.txt", "?? new.txt"]), ("mine.txt", ["?? mine.txt"])])
def test_what_is_written_over_the_first_change_while_the_second_run_goes_on_is_seen(
        tmp_path, caplog, hand_name, status_list):
    """The user writes after the second run took its start from the working tree. Over
    the lines the first change put in, the copy holds the second change alone and,
    applied in reverse, takes it out and leaves the user's lines and the first change;
    elsewhere, the copy holds both changes and leaves the user's file."""
    def write_later(worktree):
        (worktree / "later.txt").write_text("later\n", encoding="utf-8")

    def edit_by_hand(repo):
        (repo / hand_name).write_text("by hand\n", encoding="utf-8")

    repo, later_dict = _land_one_after_another(tmp_path, _write_kept_and_new,
                                               write_later, edit_during=edit_by_hand)
    assert ("takes out the last 1 of the 2 change(s)" in caplog.text) is (
        hand_name == "kept.txt")
    _git(repo, "apply", "-R", "--binary", later_dict["last"])
    assert sorted(_git(repo, "status", "--porcelain").splitlines()) == status_list
    assert (repo / hand_name).read_text(encoding="utf-8") == "by hand\n"


@pytest.mark.parametrize("is_edit", [True, False])
def test_a_first_change_whose_lines_are_found_elsewhere_alone_is_left_out(
        tmp_path, caplog, is_edit):
    """The first change writes the lower of two blocks of the same lines. The user
    writes over that line while the second run goes on: the copy holds the second
    change alone and, applied in reverse, leaves the upper block and the user's line.
    With no such writing, the copy holds both changes."""
    block = "a\nb\nc\nX = {}\nd\ne\nf\n"
    fill = "".join(f"fill {i}\n" for i in range(10))
    repo = _repo(tmp_path / "repo")
    (repo / "a.txt").write_text(block.format(2) + fill + block.format(1) + fill,
                                encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "--quiet", "-m", "two blocks")
    base = gitcmd.require_head(str(repo))
    last_path = str(tmp_path / "kept" / "land-last.diff")

    def write(worktree):
        (worktree / "a.txt").write_text(block.format(2) + fill + block.format(2) + fill,
                                        encoding="utf-8")

    def write_later(worktree):
        (worktree / "later.txt").write_text("later\n", encoding="utf-8")

    first = _commit_over(repo, write, base)
    sandbox.land_in_tree(str(repo), base, first, str(tmp_path / "one.diff"), last_path)
    later = _commit_over(repo, write_later, first)
    user_text = block.format(2) + fill + block.format(3 if is_edit else 2) + fill
    (repo / "a.txt").write_text(user_text, encoding="utf-8")
    later_dict = sandbox.land_in_tree(str(repo), first, later,
                                      str(tmp_path / "two.diff"), last_path)
    assert ("takes out the last 1 of the 2 change(s)" in caplog.text) is is_edit
    _git(repo, "apply", "-R", "--binary", later_dict["last"])
    assert not (repo / "later.txt").exists()
    assert (repo / "a.txt").read_text(encoding="utf-8") == (
        user_text if is_edit else block.format(2) + fill + block.format(1) + fill)


@pytest.mark.parametrize("is_above", [True, False])
def test_a_first_change_that_went_in_at_other_lines_comes_out_with_the_later_one(
        tmp_path, caplog, is_above):
    """The user adds a line to the file the first run writes while that run goes on,
    above or below the line it writes, so the first change goes in at the lines git
    finds for it. The copy holds both changes and, applied in reverse, leaves the
    user's line alone."""
    text = "".join(f"line {i}\n" for i in range(1, 41))
    repo = _repo(tmp_path / "repo")
    (repo / "a.txt").write_text(text, encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "--quiet", "-m", "forty lines")
    base = gitcmd.require_head(str(repo))
    last_path = str(tmp_path / "kept" / "land-last.diff")

    def write(worktree):
        (worktree / "a.txt").write_text(text.replace("line 5\n", "line five\n"),
                                        encoding="utf-8")

    def write_later(worktree):
        (worktree / "later.txt").write_text("later\n", encoding="utf-8")

    first = _commit_over(repo, write, base)
    user_text = "by hand\n" + text if is_above else text + "by hand\n"
    (repo / "a.txt").write_text(user_text, encoding="utf-8")
    sandbox.land_in_tree(str(repo), base, first, str(tmp_path / "one.diff"), last_path)
    later = _commit_over(repo, write_later, first)
    later_dict = sandbox.land_in_tree(str(repo), first, later, str(tmp_path / "two.diff"),
                                      last_path)
    assert "takes out the last" not in caplog.text
    _git(repo, "apply", "-R", "--binary", later_dict["last"])
    assert not (repo / "later.txt").exists()
    assert (repo / "a.txt").read_text(encoding="utf-8") == user_text


def test_a_change_put_in_takes_away_the_unions_it_stands_for(tmp_path, monkeypatch):
    """A record whose "resume" names the branch carried on from names a union it stands
    for: that union is held when it was made for the command's request, deleted once a
    change is put in over a working tree that holds it and named under "stand_drop";
    a name this tool did not make is not held. A command carrying on from no such
    branch, or given another request, holds none."""
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    old_union = "codetwine-loom/sandbox/20260101-000000-5-1"
    new_union = "codetwine-loom/sandbox/20260101-000000-9-1"
    for branch in (old_union, new_union, "mine"):
        _git(repo, "branch", branch)
    out_dir = str(tmp_path / "out")
    sandbox.runrecord.write(sandbox.record_dir(out_dir, "repo"), {
        "id": "20260101-000000-1-1", "resume": [new_union],
        "result": {sandbox.runrecord.STAND_FOR_KEY: [old_union, "mine"]}})
    monkeypatch.setattr(sandbox, "live_run_list", lambda out_dir, name, engine="": [])
    monkeypatch.setattr(sandbox.integrate_mod, "request_in_branch_list",
                        lambda project_dir, branch_list: "edit both")
    monkeypatch.setattr(sandbox, "is_gap_union",
                        lambda project_dir, branch: branch == old_union)
    sandbox._hold_stand_for(str(repo), out_dir, "repo", "", ["codetwine-loom/sandbox/other"], [],
                            "edit both")
    assert sandbox.stand_for_list(str(repo)) == []
    sandbox._hold_stand_for(str(repo), out_dir, "repo", "", [new_union], [], "rename add")
    assert sandbox.stand_for_list(str(repo)) == []
    sandbox._hold_stand_for(str(repo), out_dir, "repo", "", [new_union], [], "edit both")
    assert sandbox.stand_for_list(str(repo)) == [old_union]

    def write(worktree):
        (worktree / "new.txt").write_text("new\n", encoding="utf-8")

    land_dict = sandbox.land_in_tree(str(repo), base, _commit_over(repo, write, base),
                                     str(tmp_path / "one.diff"),
                                     str(tmp_path / "kept" / "land-last.diff"))
    assert land_dict["stand_drop"] == [old_union]
    assert not sandbox.branch_tip(str(repo), old_union)
    assert sandbox.branch_tip(str(repo), new_union) and sandbox.branch_tip(str(repo), "mine")
    assert sandbox.stand_for_list(str(repo)) == []


@pytest.mark.parametrize("is_before, is_after, is_drop", [
    (True, False, True), (False, True, True), (False, False, False)])
def test_a_union_held_goes_where_the_working_tree_held_it_before_or_holds_it_after(
        tmp_path, monkeypatch, is_before, is_after, is_drop):
    """The change put in writes over what the held union put in, puts it in again, or
    goes into a working tree that no longer holds it: the union is deleted in the first
    two, and in the third it stays and is held no more."""
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    union = "codetwine-loom/sandbox/20260101-000000-5-1"
    _git(repo, "branch", union)
    monkeypatch.setattr(sandbox, "live_run_list", lambda out_dir, name, engine="": [])
    monkeypatch.setattr(sandbox, "is_gap_union", lambda project_dir, branch: (
        is_after if (repo / "new.txt").exists() else is_before))
    sandbox._hold_stand_for(str(repo), str(tmp_path / "out"), "repo", "", [union],
                            [union], "edit both")

    def write(worktree):
        (worktree / "new.txt").write_text("new\n", encoding="utf-8")

    land_dict = sandbox.land_in_tree(str(repo), base, _commit_over(repo, write, base),
                                     str(tmp_path / "one.diff"),
                                     str(tmp_path / "kept" / "land-last.diff"))
    assert land_dict.get("stand_drop", []) == ([union] if is_drop else [])
    assert bool(sandbox.branch_tip(str(repo), union)) is not is_drop
    assert sandbox.stand_for_list(str(repo)) == []


@pytest.mark.parametrize("is_gap, is_drop", [(True, True), (False, False)])
def test_a_change_the_working_tree_holds_already_takes_away_the_unions_it_stands_for(
        tmp_path, monkeypatch, is_gap, is_drop):
    """Nothing is left to put in: a held union the working tree holds is deleted and
    named under "stand_drop" as it is once a diff is put in, and one it does not hold
    stays and is held no more."""
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    union = "codetwine-loom/sandbox/20260101-000000-5-1"
    _git(repo, "branch", union)
    monkeypatch.setattr(sandbox, "live_run_list", lambda out_dir, name, engine="": [])
    monkeypatch.setattr(sandbox, "is_gap_union", lambda project_dir, branch: is_gap)
    sandbox._hold_stand_for(str(repo), str(tmp_path / "out"), "repo", "", [union],
                            [union], "edit both")
    same = _git(repo, "commit-tree", gitcmd.tree_of(str(repo), base), "-p", base, "-m",
                "nothing").strip()
    land_dict = sandbox.land_in_tree(str(repo), base, same, str(tmp_path / "one.diff"),
                                     str(tmp_path / "kept" / "land-last.diff"))
    assert land_dict["files"] == 0
    assert land_dict.get("stand_drop", []) == ([union] if is_drop else [])
    assert bool(sandbox.branch_tip(str(repo), union)) is not is_drop
    assert sandbox.stand_for_list(str(repo)) == []


@pytest.mark.parametrize("is_empty", [True, False])
def test_a_stop_while_the_unions_a_change_stands_for_go_loses_none_of_them(
        tmp_path, monkeypatch, is_empty):
    """A stop that arrives while the held unions are deleted is held until that is
    through, whether a diff went in or nothing was left to put in; deleting them, once
    stopped with no signal held, leaves every one of them held."""
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    union = "codetwine-loom/sandbox/20260101-000000-5-1"
    _git(repo, "branch", union)
    monkeypatch.setattr(sandbox, "is_gap_union", lambda project_dir, branch: True)
    stop_list = []

    def live(out_dir, name, engine=""):
        stop_list.append(name)
        signal.raise_signal(signal.SIGINT)
        return []

    def write(worktree):
        (worktree / "new.txt").write_text("new\n", encoding="utf-8")

    def stop(number, frame):
        raise KeyboardInterrupt(f"signal {number}")

    monkeypatch.setattr(sandbox, "live_run_list", live)
    sandbox._hold_stand_for(str(repo), str(tmp_path / "out"), "repo", "", [union],
                            [union], "edit both")
    commit = _git(repo, "commit-tree", gitcmd.tree_of(str(repo), base), "-p", base, "-m",
                  "nothing").strip() if is_empty else _commit_over(repo, write, base)
    handler_before = signal.signal(signal.SIGINT, stop)
    try:
        with pytest.raises(KeyboardInterrupt, match=f"signal {int(signal.SIGINT)}"):
            sandbox.land_in_tree(str(repo), base, commit, str(tmp_path / "one.diff"),
                                 str(tmp_path / "kept" / "land-last.diff"))
    finally:
        signal.signal(signal.SIGINT, handler_before)
    assert stop_list == ["repo"] and not sandbox.branch_tip(str(repo), union)
    assert sandbox.stand_for_list(str(repo)) == []
    _git(repo, "branch", union, base)
    sandbox._hold_stand_for(str(repo), str(tmp_path / "out"), "repo", "", [union],
                            [union], "edit both")

    def live_stop(out_dir, name, engine=""):
        raise KeyboardInterrupt("the run was stopped (SIGINT)")

    monkeypatch.setattr(sandbox, "live_run_list", live_stop)
    with pytest.raises(KeyboardInterrupt):
        sandbox._drop_stand_for(str(repo), [], commit)
    assert sandbox.branch_tip(str(repo), union)
    assert sandbox.stand_for_list(str(repo)) == [union]
    sandbox.reset_land_list()


@pytest.mark.parametrize("tip_at, is_drop", [
    ("commit", False), ("name", False), ("above", False), ("below", True)])
def test_a_union_held_stays_when_the_change_put_in_is_its_own(
        tmp_path, monkeypatch, tip_at, is_drop):
    """The change put in is the held union's own - its tip, named by the commit or by
    the branch, or a commit under its tip - and the working tree holds the union once
    it went in: the union stays and is held no more. A union under the change put in
    is deleted."""
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    union = "codetwine-loom/sandbox/20260101-000000-5-1"

    def write(worktree):
        (worktree / "new.txt").write_text("new\n", encoding="utf-8")

    def write_more(worktree):
        (worktree / "more.txt").write_text("more\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    tip = _commit_over(repo, write_more, commit) if tip_at == "above" \
        else base if tip_at == "below" else commit
    _git(repo, "branch", union, tip)
    monkeypatch.setattr(sandbox, "live_run_list", lambda out_dir, name, engine="": [])
    monkeypatch.setattr(sandbox, "is_gap_union",
                        lambda project_dir, branch: (repo / "new.txt").exists())
    sandbox._hold_stand_for(str(repo), str(tmp_path / "out"), "repo", "", [union],
                            [union], "edit both")
    land_dict = sandbox.land_in_tree(str(repo), base,
                                     union if tip_at == "name" else commit,
                                     str(tmp_path / "one.diff"),
                                     str(tmp_path / "kept" / "land-last.diff"))
    assert land_dict["files"] == 1
    assert land_dict.get("stand_drop", []) == ([union] if is_drop else [])
    assert bool(sandbox.branch_tip(str(repo), union)) is not is_drop
    assert sandbox.stand_for_list(str(repo)) == []


def test_changes_that_come_to_nothing_together_name_no_copy(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="codetwine_loom.sandbox")

    def write(clone):
        (clone / "new.txt").write_text("new\n", encoding="utf-8")

    def write_later(worktree):
        (worktree / "new.txt").unlink()

    repo, later_dict = _land_one_after_another(tmp_path, write, write_later)
    assert later_dict["files"] == 1 and later_dict["last"] == ""
    assert "change nothing taken together" in caplog.text
    assert _git(repo, "status", "--porcelain") == ""


def test_a_copy_that_cannot_be_kept_is_said_in_the_log(tmp_path, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file\n", encoding="utf-8")
    assert sandbox._keep_last_land(str(tmp_path), "a" * 40, ("b" * 40, "c" * 40),
                                   str(tmp_path / "land.diff"),
                                   str(blocker / "land-last.diff")) == ""
    assert "could not keep a copy" in caplog.text


def test_the_kept_copy_goes_beside_the_cache_of_the_repository(tmp_path):
    """Both are this tool's own, outside the directory codetwine writes the analysis in
    and outside every agent's copy; so are the records."""
    own_dir = tmp_path / "agents" / ".repo" / "repo"
    assert sandbox.repo_own_dir(str(tmp_path), "repo") == str(own_dir)
    assert sandbox.last_land_path(str(tmp_path), "repo") == str(own_dir / "land-last.diff")
    assert sandbox.cache_dir(str(tmp_path), "repo") == str(own_dir / "cache")
    assert sandbox.record_dir(str(tmp_path), "repo") == str(own_dir / "record")


def test_a_commit_holding_the_same_tree_puts_nothing_in(tmp_path):
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    _git(repo, "commit", "--quiet", "--allow-empty", "-m", "same tree")
    land_dict = sandbox.land_in_tree(str(repo), base, gitcmd.require_head(str(repo)),
                                     str(tmp_path / "land.diff"),
                                     str(tmp_path / "land-last.diff"))
    assert land_dict == {"files": 0, "paths": [], "patch": str(tmp_path / "land.diff"),
                         "last": "", "no_status": []}
    assert not (tmp_path / "land.diff").exists()
    assert not (tmp_path / "land-last.diff").exists()


def _commit_of(repo, write, base):
    """Commit what write() makes of the files at base, put the repository back at
    base, and return the commit."""
    write(repo)
    _git(repo, "add", "-A", "--force")
    _git(repo, "commit", "--quiet", "-m", "what a run landed")
    commit = gitcmd.require_head(str(repo))
    _git(repo, "reset", "--quiet", "--hard", base)
    return commit


@pytest.mark.parametrize("rel", [".env", "sub/.env", ".ENV", "sub/deep/.Env"])
@pytest.mark.parametrize("place", ["none there", "the user's own", "the user's own, ignored"])
def test_a_change_to_a_settings_file_git_does_not_track_is_not_put_into_the_tree(
        tmp_path, rel, place):
    """The change adds the file, or rewrites the one the user put there; the rest of
    the change is not put in either, and stays for the user to read."""
    repo = _repo(tmp_path / "repo")
    if place.endswith("ignored"):
        (repo / ".gitignore").write_text(".env\n.ENV\n.Env\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", "ignore")
    base = gitcmd.require_head(str(repo))

    def write(clone):
        (clone / rel).parent.mkdir(parents=True, exist_ok=True)
        (clone / rel).write_text("DEF_K=7\n", encoding="utf-8")
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    commit = _commit_of(repo, write, base)
    if place != "none there":
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("DEF_K=5\n", encoding="utf-8")
        if not place.endswith("ignored"):
            # The commit the run starts from holds the file, and the run rewrote it
            base = gitcmd.snapshot_commit(str(repo))
    with pytest.raises(RuntimeError) as refusal:
        sandbox.land_in_tree(str(repo), base, commit, str(tmp_path / "land.diff"),
                             str(tmp_path / "land-last.diff"))
    text, command = refusal.value.why, refusal.value.command
    assert str(refusal.value) == f"{text} {command}"
    assert command == (f"git -C {repo} apply --binary --whitespace=nowarn --no-ignore-whitespace "
                       f"{tmp_path / 'land.diff'}")
    assert text.startswith(f"the change writes {rel}, which git does not track in {repo}")
    assert text.splitlines() == [text] and str(tmp_path / "land.diff") in text
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"
    assert (repo / rel).exists() is (place != "none there")
    if place != "none there":
        assert (repo / rel).read_text(encoding="utf-8") == "DEF_K=5\n"
    assert (tmp_path / "land.diff").is_file() and sandbox.land_list() == []
    assert sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                            str(tmp_path / "land-last.diff")) == {
        "land": None, "land_why": text, "land_command": command, "land_refuse": True}


def test_a_change_to_a_settings_file_git_tracks_is_put_into_the_tree(tmp_path):
    """One the repository holds is no file a command reads its settings from; a
    removed one and one of another name are changes as any other is."""
    repo = _repo(tmp_path / "repo")
    for rel in (".env", "sub/.env", "gone/.env"):
        (repo / rel).parent.mkdir(exist_ok=True)
        (repo / rel).write_text("DEF_K=5\n", encoding="utf-8")
    _git(repo, "add", "-A", "--force")
    _git(repo, "commit", "--quiet", "-m", "settings of the repository's own")
    base = gitcmd.require_head(str(repo))

    def write(clone):
        (clone / ".env").write_text("DEF_K=7\n", encoding="utf-8")
        (clone / "sub" / ".env").write_text("DEF_K=8\n", encoding="utf-8")
        (clone / "gone" / ".env").unlink()
        (clone / ".env.example").write_text("DEF_K=\n", encoding="utf-8")
        (clone / "env").write_text("x\n", encoding="utf-8")

    land_dict = sandbox.land_in_tree(str(repo), base, _commit_of(repo, write, base),
                                     str(tmp_path / "land.diff"),
                                     str(tmp_path / "land-last.diff"))
    assert land_dict["paths"] == [".env", ".env.example", "env", "gone/.env", "sub/.env"]
    assert (repo / ".env").read_text(encoding="utf-8") == "DEF_K=7\n"
    assert not (repo / "gone" / ".env").exists()


def test_a_settings_file_taken_out_of_the_index_is_one_git_does_not_track(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / ".env").write_text("DEF_K=5\n", encoding="utf-8")
    _git(repo, "add", "-A", "--force")
    _git(repo, "commit", "--quiet", "-m", "tracked")
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: (clone / ".env").write_text(
        "DEF_K=7\n", encoding="utf-8"), base)
    assert sandbox._setting_file_path_list(str(repo), ["a.py", ".env"]) == []
    _git(repo, "rm", "--quiet", "--cached", ".env")
    assert sandbox._setting_file_path_list(str(repo), ["a.py", ".env"]) == [".env"]
    with pytest.raises(RuntimeError, match="the change writes .env, which git does not"):
        sandbox.land_in_tree(str(repo), base, commit, str(tmp_path / "land.diff"),
                             str(tmp_path / "land-last.diff"))
    assert (repo / ".env").read_text(encoding="utf-8") == "DEF_K=5\n"


def _include_a_file_of_the_tree(repo, outside):
    _git(repo, "config", "include.path", "../.gitconfig")


def _include_a_file_not_there_yet(repo, outside):
    _git(repo, "config", "includeIf.gitdir:/nowhere/.path", "../conf/more.cfg")


def _read_attributes_in_the_tree(repo, outside):
    _git(repo, "config", "core.attributesFile", str(repo / "conf" / "attributes"))


def _read_rules_in_the_tree(repo, outside):
    _git(repo, "config", "core.excludesFile", "conf/ignore")


def _read_rules_through_a_link(repo, outside):
    outside.mkdir()
    (outside / "to_conf").symlink_to(repo / "conf")
    (outside / "ignore").symlink_to("to_conf/ignore")
    _git(repo, "config", "core.excludesFile", str(outside / "ignore"))


def _read_hooks_in_the_tree(repo, outside):
    _git(repo, "config", "core.hooksPath", "conf")


def _link_the_hooks_into_the_tree(repo, outside):
    shutil.rmtree(repo / ".git" / "hooks")
    (repo / ".git" / "hooks").symlink_to("../conf")


def _link_a_hook_into_the_tree(repo, outside):
    (repo / ".git" / "hooks" / "pre-commit").symlink_to("../../conf/pre-commit")


def _link_a_hook_to_a_directory_of_the_tree(repo, outside):
    (repo / ".git" / "hooks" / "lib").symlink_to("../../conf")


def _link_a_hook_of_the_hooks_path_into_the_tree(repo, outside):
    outside.mkdir()
    (outside / "pre-commit").symlink_to(repo / "conf" / "pre-commit")
    _git(repo, "config", "core.hooksPath", str(outside))


def _add_a_submodule(repo, outside):
    """Add the submodule sub, cloned from a repository made at outside, and return the
    top of its working tree."""
    outside.mkdir()
    _git(outside, "init", "--quiet", "-b", "main")
    _git(outside, "-c", "user.name=a", "-c", "user.email=a@b", "commit", "--quiet",
         "--allow-empty", "-m", "one")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", "--quiet",
         str(outside), "sub")
    return repo / "sub"


def _read_hooks_of_a_submodule_in_the_tree(repo, outside):
    _git(_add_a_submodule(repo, outside), "config", "core.hooksPath", "../conf")


def _link_a_hook_of_a_submodule_into_the_tree(repo, outside):
    hook_dir = os.path.join(_git(_add_a_submodule(repo, outside), "rev-parse",
                                 "--path-format=absolute", "--git-common-dir").strip(),
                            "hooks")
    os.symlink(repo / "conf" / "pre-commit", os.path.join(hook_dir, "pre-commit"))


def _read_the_user_s_settings_in_the_tree(repo, outside):
    os.environ["GIT_CONFIG_GLOBAL"] = str(repo / "conf" / "more.cfg")


def _read_attributes_named_by_count(repo, outside):
    os.environ.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.attributesFile",
                       "GIT_CONFIG_VALUE_0": str(repo / "conf" / "attributes")})


def _read_rules_named_by_count(repo, outside):
    os.environ.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.excludesFile",
                       "GIT_CONFIG_VALUE_0": "conf/ignore"})


def _read_hooks_named_by_parameters(repo, outside):
    os.environ["GIT_CONFIG_PARAMETERS"] = "'core.hooksPath'='conf'"


# The variables git reads settings from in its environment
_GIT_SETTING_ENV_TUPLE = ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0",
                          "GIT_CONFIG_PARAMETERS")


@pytest.mark.parametrize("prepare, rel", [
    (_include_a_file_of_the_tree, ".gitconfig"),
    (_include_a_file_not_there_yet, "conf/more.cfg"),
    (_read_attributes_in_the_tree, "conf/attributes"),
    (_read_rules_in_the_tree, "conf/ignore"),
    (_read_rules_through_a_link, "conf/ignore"),
    (_read_hooks_in_the_tree, "conf/pre-commit"),
    (_link_the_hooks_into_the_tree, "conf/pre-commit"),
    (_link_a_hook_into_the_tree, "conf/pre-commit"),
    (_link_a_hook_to_a_directory_of_the_tree, "conf/pre-commit"),
    (_link_a_hook_of_the_hooks_path_into_the_tree, "conf/pre-commit"),
    (_read_hooks_of_a_submodule_in_the_tree, "conf/pre-commit"),
    (_link_a_hook_of_a_submodule_into_the_tree, "conf/pre-commit"),
    (_read_the_user_s_settings_in_the_tree, "conf/more.cfg"),
    (_read_attributes_named_by_count, "conf/attributes"),
    (_read_rules_named_by_count, "conf/ignore"),
    (_read_hooks_named_by_parameters, "conf/pre-commit"),
])
def test_a_change_to_a_place_git_reads_its_settings_from_is_not_put_into_the_tree(
        tmp_path, monkeypatch, prepare, rel):
    """The place is in the working tree, named by a setting of a file or of the
    environment, a submodule's among them, reached through links, or a file a
    setting includes that is not
    there yet. The program the change names there is run by no git command this tool
    starts afterwards."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    for name in _GIT_SETTING_ENV_TUPLE:
        monkeypatch.delenv(name, raising=False)
    repo = _repo(tmp_path / "repo")
    (repo / "conf").mkdir()
    for name, text in ((".gitconfig", "[alias]\n\tst = status\n"),
                       ("conf/attributes", "*.bin binary\n"), ("conf/ignore", "*.swp\n")):
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "settings kept in the tree")
    prepare(repo, tmp_path / "outside")
    base = gitcmd.snapshot_commit(str(repo))
    mark = tmp_path / "ran"

    def write(worktree):
        (worktree / rel).write_text(
            f"[filter \"mark\"]\n\tclean = \"sh -c 'echo ran >> {mark}; cat'\"\n"
            if rel.endswith(("config", ".cfg")) else "tools/\n* filter=mark\n",
            encoding="utf-8")
        (worktree / ".gitattributes").write_text("* filter=mark\n", encoding="utf-8")
        (worktree / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    before = _git(repo, "status", "--porcelain", "-uall", "--ignored")
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes {rel}, which git reads its settings from in {repo} (a "
        f"configuration file, a file or a directory a setting of its configuration "
        f"names, its hooks, a program a setting or a variable of the environment "
        f"names, or a directory PATH lists), so nothing is put into the working tree.")
    gitcmd.snapshot_commit(str(repo))
    assert _git(repo, "status", "--porcelain", "-uall", "--ignored") == before
    assert not mark.exists() and sandbox.land_list() == []


def test_a_change_beside_what_a_hook_links_to_is_put_into_the_tree(tmp_path, monkeypatch):
    """A hook that is a file, and a file beside the one a hook links to, are no
    place git reads its settings from."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "pre-commit").write_text("#!/bin/sh\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "hooks kept in the tree")
    (repo / ".git" / "hooks" / "pre-commit").symlink_to("../../scripts/pre-commit")
    (repo / ".git" / "hooks" / "commit-msg").write_text("#!/bin/sh\n", encoding="utf-8")
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / "scripts" / "lint.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (worktree / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"]["paths"] == ["kept.txt", "scripts/lint.sh"]


@pytest.mark.parametrize("key, value, rel, text", [
    ("core.hooksPath", "conf/hooks", "hooks/pre-commit", "#!/bin/sh\necho ran >> {mark}\n"),
    ("core.attributesFile", "conf/attributes", "attributes", "* filter=mark\n"),
    ("include.path", "../conf/more.cfg", "more.cfg",
     "[filter \"mark\"]\n\tclean = \"sh -c 'echo ran >> {mark}; cat'\"\n"),
])
def test_a_change_to_a_link_on_the_way_to_a_place_git_reads_its_settings_from_is_not_put_in(
        tmp_path, monkeypatch, key, value, rel, text):
    """A directory above the place is a symbolic link the tree tracks, and the change
    points the link at a directory it writes."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    mark = tmp_path / "ran"
    _name_a_filter(repo, mark)
    (repo / "one" / "hooks").mkdir(parents=True)
    (repo / "one" / rel).write_text("", encoding="utf-8")
    (repo / "conf").symlink_to("one")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a link on the way")
    _git(repo, "config", key, value)
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / "two" / "hooks").mkdir(parents=True)
        (worktree / "two" / rel).write_text(text.format(mark=mark), encoding="utf-8")
        (worktree / "two" / rel).chmod(0o755)
        (worktree / "conf").unlink()
        (worktree / "conf").symlink_to("two")
        (worktree / ".gitattributes").write_text("* filter=mark\n", encoding="utf-8")
        (worktree / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    mark.unlink(missing_ok=True)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes conf, which git reads its settings from in {repo} (")
    gitcmd.snapshot_commit(str(repo))
    assert os.readlink(repo / "conf") == "one" and not mark.exists()


@pytest.mark.parametrize("key, value", [
    ("filter.mark.clean", "tools/clean.sh"),
    ("filter.mark.clean", "sh ./tools/clean.sh %f"),
    ("filter.mark.clean", "sh -c 'tools/clean.sh \"$1\" | cat' -"),
    ("filter.mark.clean", "{repo}/tools/clean.sh"),
    ("diff.mark.textconv", "tools/clean.sh"),
    ("merge.mark.driver", "sh tools/clean.sh %A %O %B"),
    ("alias.tidy", "!tools/clean.sh"),
    ("core.pager", "tools/clean.sh | less"),
    ("credential.helper", "!f() {{ tools/clean.sh; }}; f"),
    ("core.sshCommand", "tools/clean.sh -i key"),
    ("filter.mark.clean", '"$(git rev-parse --show-toplevel)/tools/clean.sh"'),
    ("filter.mark.clean", "$(git rev-parse --show-toplevel)/tools/clean.sh %f"),
    ("filter.mark.clean", '"$PWD/tools/clean.sh"'),
    ("filter.mark.clean", "cd tools && ./clean.sh"),
    ("filter.mark.clean", "${{GIT_WORK_TREE:-.}}/tools/clean.sh"),
    ("filter.mark.clean", "cd tools && ./cl\\ean'.'sh"),
    ("hook.mine.command", "tools/clean.sh"),
    ("hook.mine.command", '"$PWD/tools/clean.sh"'),
    ("later.key.program", "tools/clean.sh --flag"),
    ("alias.open", "checkout tools/clean.sh"),
    ("user.signingKey", "tools/clean.sh"),
    ("filter.mark.clean", "sed -ftools/clean.sh"),
    ("filter.mark.clean", "java @tools/clean.sh"),
])
def test_a_change_to_a_program_the_git_settings_name_is_not_put_into_the_tree(
        tmp_path, monkeypatch, key, value):
    """The setting runs a file of the working tree as a command, by itself, in a line
    a shell reads, or behind what a shell expands, or is any setting whose value has
    a word that file is at, and the change rewrites that file alone."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    (repo / "tools").mkdir()
    (repo / "tools" / "clean.sh").write_text("#!/bin/sh\ncat\n", encoding="utf-8")
    (repo / "tools" / "clean.sh").chmod(0o755)
    (repo / ".gitattributes").write_text("*.dat filter=mark\n", encoding="utf-8")
    (repo / "data.dat").write_text("d = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a program in the tree")
    _git(repo, "config", key, value.format(repo=repo))
    (repo / "data.dat").write_text("d = 1\nnot committed\n", encoding="utf-8")
    base = gitcmd.snapshot_commit(str(repo))
    mark = tmp_path / "ran"
    commit = _commit_over(repo, lambda worktree: (worktree / "tools" / "clean.sh").write_text(
        f"#!/bin/sh\necho ran >> {mark}\ncat\n", encoding="utf-8"), base)
    mark.unlink(missing_ok=True)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes tools/clean.sh, which git reads its settings from in {repo} (")
    gitcmd.snapshot_commit(str(repo))
    assert (repo / "tools" / "clean.sh").read_text(encoding="utf-8") == "#!/bin/sh\ncat\n"
    assert not mark.exists()


def test_a_file_git_config_alone_is_handed_hides_no_program_of_the_settings(
        tmp_path, monkeypatch):
    """GIT_CONFIG names a file only "git config" reads; the gate reads the settings
    every other git command reads, and the change to the program a filter of them runs
    is not put in."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    (repo / "tools").mkdir()
    (repo / "tools" / "clean.sh").write_text("#!/bin/sh\ncat\n", encoding="utf-8")
    (repo / "tools" / "clean.sh").chmod(0o755)
    (repo / ".gitattributes").write_text("*.dat filter=mark\n", encoding="utf-8")
    (repo / "data.dat").write_text("d = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a program in the tree")
    _git(repo, "config", "filter.mark.clean", "tools/clean.sh")
    (repo / "data.dat").write_text("d = 1\nnot committed\n", encoding="utf-8")
    base = gitcmd.snapshot_commit(str(repo))
    mark = tmp_path / "ran"
    commit = _commit_over(repo, lambda worktree: (worktree / "tools" / "clean.sh").write_text(
        f"#!/bin/sh\necho ran >> {mark}\ncat\n", encoding="utf-8"), base)
    mark.unlink(missing_ok=True)
    (tmp_path / "only.gitconfig").write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG", str(tmp_path / "only.gitconfig"))
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert "--config-env=filter.mark.clean=CTLOOM_NO_DRIVER" in gitcmd.own_driver_tuple(
        str(repo))[0]
    gitcmd.snapshot_commit(str(repo))
    assert not mark.exists()
    assert (tmp_path / "only.gitconfig").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("where", ["onbranch", "onbranch-glob", "hasconfig", "gitdir",
                                   "worktree", "environment", "nested"])
@pytest.mark.parametrize("what", ["program", "data"])
def test_a_filter_git_reads_only_in_another_state_keeps_the_change_out(
        tmp_path, monkeypatch, where, what):
    """A filter set where git reads it only once a branch, a remote, the repository's
    place or the worktree extension says so is read as it is now: neither the change
    to the program it runs nor the change to a file it filters is put in, and the
    user's next git, once the state says so, runs nothing the change holds."""
    user = tmp_path / "user.gitconfig"
    user.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(user))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    (repo / "tools").mkdir()
    (repo / "tools" / "clean.sh").write_text("#!/bin/sh\ncat\n", encoding="utf-8")
    (repo / "tools" / "clean.sh").chmod(0o755)
    (repo / ".gitattributes").write_text("*.dat filter=mark\n", encoding="utf-8")
    (repo / "data.dat").write_text("d = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a program in the tree")
    base = gitcmd.snapshot_commit(str(repo))
    mark = tmp_path / "ran"
    run_text = f"#!/bin/sh\necho ran >> {mark}\ncat\n"
    if what == "program":
        program = "tools/clean.sh"
        commit = _commit_over(repo, lambda worktree: (
            worktree / "tools" / "clean.sh").write_text(run_text, encoding="utf-8"), base)
    else:
        (tmp_path / "user.sh").write_text(run_text, encoding="utf-8")
        (tmp_path / "user.sh").chmod(0o755)
        program = str(tmp_path / "user.sh")
        commit = _commit_over(repo, lambda worktree: (worktree / "data.dat").write_text(
            "d = 2\n", encoding="utf-8"), base)
    include = tmp_path / "mark.gitconfig"
    include.write_text(f'[filter "mark"]\n\tclean = {program}\n', encoding="utf-8")
    condition, path = {
        "onbranch": ("onbranch:feat", "feat"), "onbranch-glob": ("onbranch:rel/**",
                                                                  "rel/1"),
        "hasconfig": ("hasconfig:remote.*.url:https://corp.invalid/**", ""),
        "gitdir": (f"gitdir:{tmp_path}/corp/", ""), "worktree": ("", ""),
        "environment": ("onbranch:feat", "feat"), "nested": ("onbranch:feat", "feat"),
    }[where]
    if where == "worktree":
        (repo / ".git" / "config.worktree").write_text(include.read_text(encoding="utf-8"),
                                                       encoding="utf-8")
    elif where == "environment":
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", f"includeIf.{condition}.path")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(include))
    elif where == "nested":
        (tmp_path / "team.gitconfig").write_text(
            f'[includeIf "{condition}"]\n\tpath = mark.gitconfig\n', encoding="utf-8")
        user.write_text("[include]\n\tpath = team.gitconfig\n", encoding="utf-8")
    else:
        user.write_text(f'[includeIf "{condition}"]\n\tpath = {include}\n',
                        encoding="utf-8")
    assert gitcmd.own_driver_tuple(str(repo)) == ([], {})
    mark.unlink(missing_ok=True)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    if path:
        _git(repo, "checkout", "--quiet", "-b", path)
    elif where == "hasconfig":
        _git(repo, "remote", "add", "origin", "https://corp.invalid/team/x.git")
    elif where == "gitdir":
        (tmp_path / "corp").mkdir()
        repo = repo.rename(tmp_path / "corp" / "repo")
    else:
        _git(repo, "config", "extensions.worktreeConfig", "true")
    assert gitcmd.own_driver_tuple(str(repo))[0]
    (repo / "data.dat").write_text("d = 1\nthe user's own line\n", encoding="utf-8")
    gitcmd.snapshot_commit(str(repo))
    assert (repo / "tools" / "clean.sh").read_text(encoding="utf-8") == "#!/bin/sh\ncat\n"
    assert not mark.exists() or what == "data"
    assert (repo / "data.dat").read_text(encoding="utf-8").startswith("d = 1\n")


@pytest.mark.parametrize("where", ["include", "dot-env"])
def test_an_attributes_file_git_reads_only_in_another_state_keeps_a_filtered_file_out(
        tmp_path, monkeypatch, where):
    """A filter attribute an attributes file gives a path counts when git reads that
    file only once a branch says so, or only in the user's shell: the file the filter
    runs on is not put in."""
    user = tmp_path / "user.gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(user))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    mark = tmp_path / "ran"
    (tmp_path / "user.sh").write_text(f"#!/bin/sh\necho ran >> {mark}\ncat\n",
                                      encoding="utf-8")
    (tmp_path / "user.sh").chmod(0o755)
    user.write_text(f'[filter "bin"]\n\tclean = {tmp_path / "user.sh"}\n', encoding="utf-8")
    (tmp_path / "attrs").write_text("*.bin filter=bin\n", encoding="utf-8")
    repo = _repo(tmp_path / "repo")
    base = gitcmd.snapshot_commit(str(repo))
    commit = _commit_over(repo, lambda worktree: (worktree / "blob.bin").write_text(
        "b = 2\n", encoding="utf-8"), base)
    if where == "include":
        (tmp_path / "inc.gitconfig").write_text(
            f"[core]\n\tattributesFile = {tmp_path / 'attrs'}\n", encoding="utf-8")
        with user.open("a", encoding="utf-8") as handle:
            handle.write(f'[includeIf "onbranch:feat"]\n\tpath = {tmp_path / "inc.gitconfig"}\n')
    else:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        (tmp_path / "home" / ".config" / "git").mkdir(parents=True)
        shutil.copy(tmp_path / "attrs", tmp_path / "home" / ".config" / "git" / "attributes")
        (tmp_path / "dot.env").write_text(f"XDG_CONFIG_HOME={tmp_path / 'other'}\n",
                                          encoding="utf-8")
        monkeypatch.setattr(config, "_ENV_FILE_FILL_DICT", {})
        config._load_env_file(str(tmp_path / "dot.env"))
    assert gitcmd.filter_by_path_dict(str(repo), [b"blob.bin"]) == {}
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert not (repo / "blob.bin").exists() and not mark.exists()


@pytest.mark.parametrize("name, value, rel", [
    ("GIT_EXTERNAL_DIFF", "{repo}/tools/clean.sh", "tools/clean.sh"),
    ("GIT_EDITOR", "tools/clean.sh --wait", "tools/clean.sh"),
    ("GIT_SSH_COMMAND", '"$(git rev-parse --show-toplevel)/tools/clean.sh" -F none',
     "tools/clean.sh"),
    ("GIT_PAGER", "cd tools && ./clean.sh", "tools/clean.sh"),
    ("VISUAL", "python3 tools", "tools/clean.sh"),
    ("GIT_EDITOR", "npm --prefix tools run edit", "tools/clean.sh"),
    ("SSH_ASKPASS", "tools/ask.sh", "tools/ask.sh"),
    ("GIT_EXEC_PATH", "tools", "tools/git-new"),
    ("GIT_TEMPLATE_DIR", "{repo}/tools", "tools/hooks/pre-commit"),
])
def test_a_change_to_a_program_the_environment_names_is_not_put_into_the_tree(
        tmp_path, monkeypatch, name, value, rel):
    """A variable git runs as a command names a file of the working tree, there or
    not, or a directory of programs, and the change writes that file or one under
    that directory."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    (repo / "tools").mkdir()
    (repo / "tools" / "clean.sh").write_text("#!/bin/sh\ncat\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a program in the tree")
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / rel).parent.mkdir(parents=True, exist_ok=True)
        (worktree / rel).write_text("#!/bin/sh\necho ran\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    before = _git(repo, "status", "--porcelain", "-uall", "--ignored")
    monkeypatch.setenv(name, value.format(repo=repo))
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    monkeypatch.delenv(name)
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes {rel}, which git reads its settings from in {repo} (")
    assert _git(repo, "status", "--porcelain", "-uall", "--ignored") == before


@pytest.mark.parametrize("entry, rel, is_land", [
    ("{repo}/tools", "tools/git", False), ("{repo}/tools", "tools/no-such-program", False),
    ("{repo}/tools", "tools/sub/inner.sh", False), ("{repo}/bin", "bin/git", False),
    ("tools", "tools/git", False), ("./tools/", "tools/docker", False),
    ("{link}", "tools/git", False), ("", "src/new.py", False), (".", "src/new.py", False),
    ("{repo}/tools", "src/new.py", True), ("{outside}", "tools/git", True),
    ("{repo}/tools/sub", "tools/git", True),
])
def test_a_change_under_a_directory_programs_are_looked_for_in_is_not_put_in(
        tmp_path, monkeypatch, entry, rel, is_land):
    """PATH lists a directory of the working tree, there or not, by its path, by a
    relative entry, through a link or as the top itself, and the change writes a
    file under it, whatever the file is named. A change anywhere else is put in."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    (repo / "tools").mkdir()
    (repo / "tools" / "build.sh").write_text("#!/bin/sh\necho build\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a directory of programs in the tree")
    (tmp_path / "outside").mkdir()
    (tmp_path / "link").symlink_to(repo / "tools")
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / rel).parent.mkdir(parents=True, exist_ok=True)
        (worktree / rel).write_text("#!/bin/sh\necho ran\n", encoding="utf-8")
        (worktree / rel).chmod(0o755)

    commit = _commit_over(repo, write, base)
    before = _git(repo, "status", "--porcelain", "-uall", "--ignored")
    monkeypatch.setenv("PATH", os.pathsep.join([
        entry.format(repo=repo, link=tmp_path / "link", outside=tmp_path / "outside"),
        os.environ["PATH"]]))
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    if is_land:
        assert land_dict["land"]["paths"] == [rel]
        return
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes {rel}, which git reads its settings from in {repo} (")
    assert "or a directory PATH lists)" in land_dict["land_why"]
    assert _git(repo, "status", "--porcelain", "-uall", "--ignored") == before


@pytest.mark.parametrize("key, value, rel_dir, is_there", [
    ("filter.mark.clean", "python3 tools/clean", "tools/clean", False),
    ("filter.mark.clean", "python3 -u tools/clean", "tools/clean", False),
    ("filter.mark.clean", "sh -c 'cd tools/clean && python3 .'", "tools/clean", False),
    ("filter.mark.clean", "run-parts tools/clean", "tools/clean", False),
    ("filter.mark.clean", "grep -v x tools/clean", "tools/clean", False),
    ("filter.mark.clean", "python3 clean", "clean", False),
    ("filter.mark.clean", 'python3 "$(git rev-parse --show-toplevel)/tools/clean"',
     "tools/clean", True),
    ("filter.mark.clean", 'python3 "$PWD/tools/clean"', "tools/clean", True),
    ("filter.mark.clean", "python3 ${PWD}/tools/clean", "tools/clean", True),
    ("filter.mark.clean", "python3 $PWD/tools/clean", "tools/clean", False),
    ("core.fsmonitor", "$PWD/tools/clean/__main__.py", "tools/clean", False),
    ("alias.tidy", "!python3 tools/clean", "tools/clean", False),
    ("alias.tidy", "!cd $PWD/tools/clean && make", "tools/clean", True),
    ("filter.mark.clean", 'python3 "$(pwd)/tools/clean"', "tools/clean", False),
    ("filter.mark.clean", 'python3 "$PWD"/tools/clean', "tools/clean", True),
    ("filter.mark.clean", 'python3 "${PWD:-.}/tools/clean"', "tools/clean", False),
    ("filter.mark.clean", 'python3 "$(cd . && pwd)/tools/clean"', "tools/clean", True),
    ("filter.mark.clean", "bash -c 'python3 ~+/tools/clean'", "tools/clean", False),
    ("filter.mark.clean", 'python3 "`pwd`/tools/clean"', "tools/clean", False),
    ("filter.mark.clean", "python3 -W ignore clean", "clean", False),
    ("filter.mark.clean", "python3 -X utf8 clean", "clean", True),
])
def test_a_change_that_makes_a_directory_a_command_of_the_git_settings_runs_is_not_put_in(
        tmp_path, monkeypatch, key, value, rel_dir, is_there):
    """The command names a directory of the working tree nothing is at yet, or one
    written behind the top of the working tree, and the change writes a file under
    it. The program the change writes there is run by no git command this tool
    starts afterwards."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    main_text = "import sys\nsys.stdout.write(sys.stdin.read())\n"
    if is_there:
        (repo / rel_dir).mkdir(parents=True)
        (repo / rel_dir / "__main__.py").write_text(main_text, encoding="utf-8")
    (repo / ".gitattributes").write_text("*.dat filter=mark\n", encoding="utf-8")
    (repo / "data.dat").write_text("d = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a filter the settings name")
    _git(repo, "config", key, value)
    (repo / "data.dat").write_text("d = 1\nnot committed\n", encoding="utf-8")
    base = gitcmd.snapshot_commit(str(repo))
    mark = tmp_path / "ran"

    def write(worktree):
        (worktree / rel_dir).mkdir(parents=True, exist_ok=True)
        (worktree / rel_dir / "__main__.py").write_text(
            f"import sys\nopen({str(mark)!r}, 'a').write('ran')\n{main_text}",
            encoding="utf-8")

    commit = _commit_over(repo, write, base)
    mark.unlink(missing_ok=True)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes {rel_dir}/__main__.py, which git reads its settings from "
        f"in {repo} (")
    (repo / "data.dat").write_text("d = 1\nnot committed\nand more\n", encoding="utf-8")
    gitcmd.snapshot_commit(str(repo))
    assert not mark.exists()
    assert (repo / rel_dir / "__main__.py").exists() is is_there


@pytest.mark.parametrize("key, value", [
    ("user.name", "python3 tools/clean"), ("alias.tidy", "-C tools/clean status"),
    ("alias.tidy", "status clean tools"),
    ("filter.mark.clean", "python3 /opt/tools/clean"),
    ("filter.mark.clean", "python3 -I/opt/tools/clean"),
    ("filter.mark.clean", "python3 ~/tools/clean"),
    ("filter.mark.clean", "python3 other/cleaner"),
])
def test_a_change_that_makes_a_directory_no_command_runs_is_put_in(
        tmp_path, monkeypatch, key, value):
    """The directory the change makes is named by a setting git runs nothing by, or
    by a command only behind a directory written as an absolute path outside the
    working tree, or is not named by the command at all."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    _git(repo, "config", key, value)
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        for rel in ("tools/clean/__main__.py", "clean/tidy.py"):
            (worktree / rel).parent.mkdir(parents=True)
            (worktree / rel).write_text("x = 1\n", encoding="utf-8")

    land_dict = sandbox.try_land(str(repo), base, _commit_over(repo, write, base),
                                 str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"]["paths"] == ["clean/tidy.py", "tools/clean/__main__.py"]


@pytest.mark.parametrize("key, value, out_list", [
    ("filter.mark.clean", 'python3 "$(pwd)/tools/clean"', "tools clean"),
    ("filter.mark.clean", 'python3 "$PWD"/tools/clean', "tools clean"),
    ("filter.mark.clean", 'python3 "${{PWD:-.}}/tools/clean"', "tools clean"),
    ("filter.mark.clean", 'python3 "$(cd . && pwd)/tools/clean"', "tools clean"),
    ("filter.mark.clean", "bash -c 'python3 ~+/tools/clean'", "tools clean"),
    ("filter.mark.clean", 'python3 "`pwd`/tools/clean"', "tools clean"),
    ("filter.mark.clean", "python3 $HOME/tools/clean", "tools clean"),
    ("filter.mark.clean", "python3 x$PWD/tools/clean", "tools clean"),
    ("filter.mark.clean", "python3 {repo}/tools/clean", "tools clean"),
    ("filter.mark.clean", "python3 -W ignore clean", "clean"),
    ("filter.mark.clean", "python3 -X utf8 -m clean", "clean"),
    ("filter.mark.clean", 'node "$(pwd)/scripts/clean"', "scripts clean"),
    ("filter.mark.clean", 'PYTHONPATH="$PWD"/tools python3 -m clean', "tools clean"),
    ("filter.mark.clean", "python3 other/clean", "clean"),
    ("filter.mark.clean", "git-lfs clean -- %f", "clean"),
    ("alias.tidy", "!python3 -W ignore tools/clean", "tools"),
    ("alias.tidy", "!grep -rn TODO tools/clean", "tools"),
    ("alias.tidy", '!python3 "$PWD"/tools/clean', "tools clean"),
    ("core.pager", "less -R", "less"),
    ("filter.mark.clean", "python3 /opt/tools/clean", ""),
    ("filter.mark.clean", 'python3 "/opt/tools"/clean', "clean"),
    ("filter.mark.clean", "python3 /opt/scripts/clean.js", "scripts"),
])
def test_a_path_a_head_of_which_a_command_of_the_git_settings_names_is_kept_out(
        tmp_path, monkeypatch, key, value, out_list):
    """A head of the path, or such a head with what a dot in its last part starts left
    out, stands as a word in a command git runs, in the settings or in the
    environment, however the directory ahead of it is written, unless that directory
    is written as an absolute path outside the working tree."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    group_dict = {"tools": ["tools/clean/__main__.py", "tools/other.py"],
                  "clean": ["clean/tidy.py", "clean.py"], "scripts": ["scripts/clean.js"],
                  "less": ["less/x", "less.txt"], "": ["src/a.py", "cleaner.py"]}
    raw_list = [os.fsencode(rel) for rel_list in group_dict.values() for rel in rel_list]
    want_list = [os.fsencode(rel) for rel_list in group_dict.values() for rel in rel_list
                 if rel in {rel for group in out_list.split() for rel in group_dict[group]}]
    _git(repo, "config", key, value.format(repo=repo))
    assert sandbox._git_setting_path_list(str(repo), raw_list) == want_list
    if key.startswith("alias."):
        return
    _git(repo, "config", "--unset", key)
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", value.format(repo=repo).removeprefix("!"))
    assert sandbox._git_setting_path_list(str(repo), raw_list) == want_list


@pytest.mark.parametrize("value, rel, is_land", [
    ("python3 -m clean", "clean.py", False),
    ("sh -c 'cd tools && python3 -m clean'", "tools/clean.py", False),
    ("env -C tools python3 -m clean", "tools/clean.py", False),
    ("node scripts/clean", "scripts/clean.js", False),
    ("node scripts/clean", "scripts/clean.mjs", False),
    ("node scripts/clean", "scripts/clean/index.js", False),
    ("python3 -m clean", "cleaner.py", True), ("python3 -m clean", "src/clean.py", True),
    ("node scripts/clean", "scripts/cleanup.js", False),
    ("node scripts/clean", "other/clean.js", True),
    ("node /opt/scripts/clean", "scripts/cleanup.js", True),
    ("python3 -W ignore -m clean", "clean.py", False),
    ('node "$(pwd)/scripts/clean"', "scripts/clean.js", False),
])
def test_a_file_a_program_opens_by_the_name_with_an_ending_added_is_not_put_in(
        tmp_path, monkeypatch, value, rel, is_land):
    """The command names a path the program it runs reads with an ending added, or as
    a directory; the change writes the file it reads, or another file under a
    directory the command names. The same name in another directory, or a longer
    name, is put in."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    (repo / ".gitattributes").write_text("*.dat filter=mark\n", encoding="utf-8")
    (repo / "data.dat").write_text("d = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a filter the settings name")
    _git(repo, "config", "filter.mark.clean", value)
    base = gitcmd.snapshot_commit(str(repo))
    mark = tmp_path / "ran"

    def write(worktree):
        (worktree / rel).parent.mkdir(parents=True, exist_ok=True)
        (worktree / rel).write_text(
            f"import sys\nopen({str(mark)!r}, 'a').write('ran')\n"
            f"sys.stdout.write(sys.stdin.read())\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    mark.unlink(missing_ok=True)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    if is_land:
        assert land_dict["land"]["paths"] == [rel]
        return
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes {rel}, which git reads its settings from in {repo} (")
    (repo / "data.dat").write_text("d = 1\nnot committed\n", encoding="utf-8")
    gitcmd.snapshot_commit(str(repo))
    assert not mark.exists()


@pytest.mark.parametrize("raw, place, is_beside", [
    (b"scripts/clean.js", b"scripts/clean", True), (b"clean.d/run", b"clean", True),
    (b"clean.py", b"clean", True), (b"scripts/cleaner.js", b"scripts/clean", False),
    (b"other/clean.js", b"scripts/clean", False), (b"scripts/clean", b"scripts/clean", False),
    (b"x/scripts/clean.js", b"scripts/clean", False), (b"clean", b"clean.py", False),
])
def test_an_entry_beside_a_place_is_one_named_after_it_with_a_dot(raw, place, is_beside):
    assert sandbox._is_beside(raw, place) is is_beside


@pytest.mark.parametrize("command, name, is_place", [
    ("git rebase origin/main", "main", False), ("git checkout main", "main", False),
    ("git fetch && git rebase origin/main", "origin", True),
    ("python3 tools/clean", "tools", True), ("python3 tools/clean", "clean", False),
    ('python3 "$PWD"/tools/clean', "clean", True), ("python3 ./tools", "tools", True),
    ("python3 $(pwd)/tools", "tools", True), ("python3 /opt/tools", "tools", True),
    ("sh -c 'cd sub/x && make'", "sub", True), ("grep -rn TODO src", "src", False),
    ("python3 ~+/tools", "tools", True), ("make -C ../tools", "tools", False),
])
def test_a_name_in_an_alias_counts_in_a_word_with_a_slash_unless_a_relative_dir_leads(
        command, name, is_place):
    raw = os.fsencode(command)
    start = raw.rindex(os.fsencode(name))
    assert sandbox._is_alias_place(raw, start, start + len(name)) is is_place


@pytest.mark.parametrize("key, value, rel, is_land", [
    ("alias.sync", "!git fetch && git rebase origin/main", "main.go", True),
    ("alias.sync", "!git fetch && git rebase origin/main", "main.py", True),
    ("alias.sync", "!git fetch && git rebase origin/main", "main/x.py", True),
    ("alias.co", "!git checkout main", "main.go", True),
    ("alias.co", "!git checkout main", "main.py", True),
    ("alias.co", "!git checkout main", "main/x.py", True),
    ("alias.co", "!git checkout main && git pull", "pull.py", True),
    ("alias.t", "!python3 tools/clean", "tools/clean/__main__.py", False),
    ("alias.t", "!python3 tools/clean", "tools/other.py", False),
    ("alias.t", "!python3 tools/clean", "tools/clean.py", False),
    ("alias.t", "!sh scripts/run.sh", "scripts/run.sh", False),
    ("alias.t", "!sh scripts/run.sh", "scripts/x.sh", False),
    ("alias.t", "!sh scripts/run.sh", "scripts/run.py", False),
    ("alias.t", '!python3 "$(pwd)/tools/clean"', "tools/x.py", False),
    ("filter.mark.clean", "git checkout main", "main.go", False),
])
def test_a_name_an_alias_holds_only_as_an_argument_keeps_no_change_out(
        tmp_path, monkeypatch, key, value, rel, is_land):
    """A name in a word of an alias with no "/" in it, or behind a relative
    directory, is no head of a path; one in a word with a "/" is. The same command
    in a setting git runs by itself counts every word."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    _git(repo, "config", key, value)
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / rel).parent.mkdir(parents=True, exist_ok=True)
        (worktree / rel).write_text("x = 1\n", encoding="utf-8")

    land_dict = sandbox.try_land(str(repo), base, _commit_over(repo, write, base),
                                 str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    if is_land:
        assert land_dict["land"]["paths"] == [rel]
    else:
        assert land_dict["land"] is None and land_dict["land_refuse"] is True


def test_a_command_an_alias_and_another_setting_both_hold_counts_every_word(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    _git(repo, "config", "alias.co", "!git checkout main")
    raw_list = [b"main.go", b"src/a.py"]
    assert sandbox._git_setting_path_list(str(repo), raw_list) == []
    _git(repo, "config", "core.pager", "!git checkout main")
    assert sandbox._git_setting_path_list(str(repo), raw_list) == [b"main.go"]


@pytest.mark.parametrize("raw, name_list", [
    (b"tools/clean/__main__.py", [b"tools", b"tools/clean", b"tools/clean/__main__.py",
                                  b"tools/clean/__main__"]),
    (b"clean.d/run", [b"clean.d", b"clean", b"clean.d/run"]),
    (b".gitmessage.txt", [b".gitmessage.txt", b".gitmessage"]),
    (b"a.b.c", [b"a.b.c", b"a", b"a.b"]), (b"main", [b"main"]),
])
def test_the_names_of_a_path_are_its_heads_and_their_names_ahead_of_a_dot(raw, name_list):
    assert sandbox._path_name_list(raw) == name_list


@pytest.mark.parametrize("command, is_outside", [
    ("python3 /opt/lib/x", True), ('python3 "/opt/lib"/x', True), ("cc -I/opt/lib/x", True),
    ("x --dir=/opt/lib/x", True), ("cat a:/opt/lib/x", True), ("vim ~/lib/x", True),
    ("'/opt/lib/x'", True), ("/lib/x", True),
    ("python3 $(pwd)/lib/x", False), ('python3 "$PWD"/lib/x', False),
    ("python3 ${{PWD:-.}}/lib/x", False), ("python3 `pwd`/lib/x", False),
    ("python3 ~+/lib/x", False), ("python3 $HOME/lib/x", False), ("python3 x/lib/x", False),
    ("python3 ./lib/x", False), ("python3 /op*/lib/x", False), ("python3 {top}/lib/x", False),
    ("python3 {top}/sub/lib/x", False), ("python3 {top}/../top/lib/x", False),
    ("python3 {link}/lib/x", False), ('python3 x"/opt"/lib/x', False),
    ("python3 lib/x", False),
])
def test_a_name_behind_an_absolute_directory_outside_the_tree_is_told(
        tmp_path, monkeypatch, command, is_outside):
    """Only a directory written as an absolute path with nothing a shell expands in
    it, whose real path is outside the working tree, counts."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "top").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "top")
    raw = os.fsencode(command.format(top=tmp_path / "top", link=tmp_path / "link"))
    start = raw.rindex(b"lib")
    root = os.fsencode(os.path.realpath(tmp_path / "top"))
    assert sandbox._is_outside_dir(raw, start, root) is is_outside


@pytest.mark.parametrize("value", [
    "python3 tools/clean", "python3 -u ./tools/clean/ %f", "env python3 tools/clean",
    "sh -c 'cd tools/clean && make -s'", "make -s -C tools/clean",
    "make -s -Ctools/clean", "git --work-tree=tools/clean --git-dir tools/clean/.git log",
    "{repo}/tools/clean", "cd tools && make -s", "python3 -W ignore tools/clean",
    "make -s --directory=tools/clean", "npm --prefix tools/clean run tidy",
    "go run tools/clean", "cc -Itools/clean -E -",
])
def test_a_change_under_a_directory_a_command_of_the_git_settings_runs_is_not_put_in(
        tmp_path, monkeypatch, value):
    """The command, one git runs by itself, names a directory of the working tree
    in any of its words, and the change rewrites a file under it. The program the
    change writes there is run by no git command this tool starts afterwards."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    (repo / "tools" / "clean").mkdir(parents=True)
    main_text = "import sys\nsys.stdout.write(sys.stdin.read())\n"
    (repo / "tools" / "clean" / "__main__.py").write_text(main_text, encoding="utf-8")
    (repo / ".gitattributes").write_text("*.dat filter=mark\n", encoding="utf-8")
    (repo / "data.dat").write_text("d = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a program in a directory of the tree")
    _git(repo, "config", "filter.mark.clean", value.format(repo=repo))
    (repo / "data.dat").write_text("d = 1\nnot committed\n", encoding="utf-8")
    base = gitcmd.snapshot_commit(str(repo))
    mark = tmp_path / "ran"
    commit = _commit_over(
        repo, lambda worktree: (worktree / "tools" / "clean" / "__main__.py").write_text(
            f"import sys\nopen({str(mark)!r}, 'a').write('ran')\n{main_text}",
            encoding="utf-8"), base)
    mark.unlink(missing_ok=True)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes tools/clean/__main__.py, which git reads its settings from "
        f"in {repo} (")
    gitcmd.snapshot_commit(str(repo))
    assert (repo / "tools" / "clean" / "__main__.py").read_text(
        encoding="utf-8") == main_text
    assert not mark.exists()


@pytest.mark.parametrize("key, value", [
    ("filter.mark.clean", "/bin/cat"), ("user.name", "tools"),
    ("alias.tidy", "status tools"), ("core.pager", "/usr/bin/less -R"),
    ("alias.todo", "!grep -rn TODO /opt/tools /opt/cat /opt/less"),
    ("alias.here", "!cd . && make -C ./ && git --work-tree=. status"),
    ("init.defaultBranch", "less"),
    ("alias.wait", "!python3 -W ignore /opt/tools; npm --prefix /opt/cat run tidy"),
    ("core.editor", "/usr/bin/less --wait"), ("user.name", "npm --prefix tools"),
])
def test_a_change_to_a_file_no_setting_runs_as_a_program_is_put_in(
        tmp_path, monkeypatch, key, value):
    """The settings name a command outside the working tree by its absolute path, a
    directory of the working tree only in an alias that is no shell command or in a
    setting git runs nothing by, or the top of the working tree."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("EDITOR", "vim")
    monkeypatch.setenv("GIT_EXEC_PATH", str(tmp_path / "libexec"))
    repo = _repo(tmp_path / "repo")
    for name in ("tools", "cat", "less"):
        (repo / name).mkdir()
        (repo / name / "clean.sh").write_text("#!/bin/sh\ncat\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "files beside the settings")
    _git(repo, "config", key, value)
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        for name in ("tools", "cat", "less"):
            (worktree / name / "clean.sh").write_text("#!/bin/sh\ncat -\n", encoding="utf-8")

    land_dict = sandbox.try_land(str(repo), base, _commit_over(repo, write, base),
                                 str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"]["paths"] == ["cat/clean.sh", "less/clean.sh", "tools/clean.sh"]


@pytest.mark.parametrize("key, value, rel, is_land", [
    ("alias.tidy", "!./run.sh", "sub/run.sh", False),
    ("alias.tidy", "!sh sub/my\\ run.sh", "sub/my run.sh", False),
    ("alias.tidy", "!make test", "Makefile", True),
    ("alias.tidy", "status run.sh", "sub/run.sh", True),
    ("user.name", "run.sh", "sub/run.sh", True),
    ("alias.tidy", "!./run.sh", "sub/run.sh.txt", True),
    ("filter.lfs.clean", "git-lfs clean -- %f", "scripts/clean", False),
    ("filter.lfs.clean", "git-lfs clean -- %f", "testdata/a", True),
    ("filter.lfs.clean", "git-lfs clean -- %f", "testdata/lfs", True),
    ("alias.tidy", "!sh tools/run.sh", "sub/run", True),
    ("alias.tidy", "!sh $1run.sh", "sub/run.sh", False),
    ("alias.tidy", "!sh $DIRrun.sh", "sub/run.sh", True),
    ("alias.tidy", "!run.sh --now", "sub/run.sh", False),
    ("alias.tidy", "!sed -frun.sh", "sub/run.sh", False),
    ("alias.tidy", "!java @run.sh", "sub/run.sh", False),
    ("core.pager", "less -FRX", "sub/RX", False),
    ("core.pager", "less -FRX", "sub/s", True),
])
def test_a_file_whose_name_is_written_in_a_command_of_the_git_settings_is_not_put_in(
        tmp_path, monkeypatch, key, value, rel, is_land):
    """The name of the file the change writes stands as a word of its own in the
    value of a setting git runs as a command, whatever directory the file is in and
    with or without what a shell reads as quoting. A name that is part of a longer
    word keeps no change out."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    _git(repo, "config", key, value)
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / rel).parent.mkdir(parents=True, exist_ok=True)
        (worktree / rel).write_text("echo ran\n", encoding="utf-8")

    land_dict = sandbox.try_land(str(repo), base, _commit_over(repo, write, base),
                                 str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    if is_land:
        assert land_dict["land"]["paths"] == [rel]
    else:
        assert land_dict["land"] is None and land_dict["land_refuse"] is True
        assert land_dict["land_why"].startswith(
            f"the change writes {rel}, which git reads its settings from in {repo} (")
        assert not (repo / rel).exists()


@pytest.mark.parametrize("command, name, is_word", [
    ("vi", "vi", True), ("vi", "v", False), ("vi", "i", False),
    ("git-lfs clean -- %f", "clean", True), ("git-lfs clean -- %f", "git", False),
    ("git-lfs clean -- %f", "lfs", False), ("git-lfs clean -- %f", "f", False),
    ("git-lfs clean -- %f", "a", False), ("git-lfs clean -- %f", "git-lfs", True),
    ("git-lfs smudge -- %f", "smudge", True), ("git-lfs filter-process", "process", False),
    ("git-lfs filter-process", "filter", False),
    ("git-lfs filter-process", "filter-process", True),
    ("sh $1clean.sh", "clean.sh", True), ("sh $@clean.sh", "clean.sh", True),
    ("sh ${D}clean.sh", "clean.sh", True), ("sh $(pwd)clean.sh", "clean.sh", True),
    ("sh `pwd`clean.sh", "clean.sh", True), ("sh $(pwd)/clean.sh", "clean.sh", True),
    ("sh $DIRclean.sh", "clean.sh", False), ("sh $clean.sh", "clean.sh", False),
    ('sh "my file.sh"', "my file.sh", True), ("sh my\\ file.sh", "my file.sh", True),
    ('sh "my file".sh', "my file.sh", True), ("sh ./cl\\ean'.'sh", "clean.sh", True),
    ("sh tools/clean.sh", "clean", False), ("sh tools/clean.sh", "clean.sh", True),
    ("sh tools/clean.sh", "tools", True), ("sh tools/clean.sh.bak", "clean.sh", False),
    ("!run.sh --now", "run.sh", True), ("!f() { run.sh; }; f", "run.sh", True),
    ("x=1;run.sh|cat>out,log", "run.sh", True), ("x=1;run.sh|cat>out,log", "out", True),
    ("ssh -oIdentityFile=id_key -F a:b", "id_key", True),
    ("ssh -oIdentityFile=id_key -F a:b", "b", True), ("less -FRX", "less", True),
    ("less -FRX", "s", False), ("less -FRX", "FRX", False), ("less -FRX", "Less", False),
    ("sed -ftidy.sed", "tidy.sed", True), ("sed -f tidy.sed", "tidy.sed", True),
    ("sed -n -ftidy.sed", "tidy.sed", True), ("-ftidy.sed", "tidy.sed", True),
    ("sed x-ftidy.sed", "tidy.sed", False), ("sed --ftidy.sed", "tidy.sed", False),
    ("sed -fxtidy.sed", "tidy.sed", False), ("cc -I./inc", "inc", True),
    ("java @args.txt", "args.txt", True), ("@args.txt", "args.txt", True),
    ("mail a@args.txt", "args.txt", False), ("less -FRX", "RX", True),
    ("less -FRX", "X", False), ("ls -la", "a", True), ("emacs -nw", "w", True),
    ("git-lfs clean -- %f", "fs", False), ("tar +x.tar %x.tar ~x.tar", "x.tar", False),
    ("cat a.txt\nb.txt", "b.txt", True), ("", "x", False),
])
def test_a_name_counts_in_a_command_where_it_stands_as_a_word_of_its_own(
        command, name, is_word):
    """The word ends are the ends of the command, white space, "/", quotes and what
    a shell reads as an operator or a separator, and on the left a variable written
    with one character, a short option at the head of a word and "@" at the head of
    a word; elsewhere a letter, a digit, "-", ".", "@" and "%" are part of a word."""
    raw = os.fsencode(f"sub/{name}")
    assert sandbox._command_name_path_list([raw], [os.fsencode(command)]) == (
        [raw] if is_word else [])


@pytest.mark.parametrize("name, is_land", [("v", True), ("i", True), ("vi", False)])
def test_a_name_inside_a_word_of_a_variable_git_runs_keeps_no_change_out(
        tmp_path, monkeypatch, name, is_land):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("EDITOR", "vi")
    repo = _repo(tmp_path / "repo")
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / "testdata").mkdir()
        (worktree / "testdata" / name).write_text("x\n", encoding="utf-8")

    land_dict = sandbox.try_land(str(repo), base, _commit_over(repo, write, base),
                                 str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert bool(land_dict["land"]) is is_land
    assert land_dict["land_refuse"] is not is_land


def test_a_change_to_what_a_link_a_command_of_the_git_settings_names_leads_to_is_not_put_in(
        tmp_path, monkeypatch):
    """The setting names a symbolic link the tree holds behind what a shell expands,
    and the change rewrites the file the link leads to. A link that leads to a
    directory keeps no change out for its name."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    for name in ("tools", "bin", "src"):
        (repo / name).mkdir()
    (repo / "tools" / "clean.sh").write_text("#!/bin/sh\ncat\n", encoding="utf-8")
    (repo / "tools" / "clean.sh").chmod(0o755)
    (repo / "bin" / "tidy").symlink_to("../tools/clean.sh")
    (repo / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (repo / "lib").symlink_to("src")
    (repo / ".gitattributes").write_text("*.dat filter=mark\n", encoding="utf-8")
    (repo / "data.dat").write_text("d = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a program behind a link")
    _git(repo, "config", "filter.mark.clean", '"$(git rev-parse --show-toplevel)/bin/tidy"')
    _git(repo, "config", "credential.helper", "/usr/lib/git-core/git-credential-store")
    (repo / "data.dat").write_text("d = 1\nnot committed\n", encoding="utf-8")
    base = gitcmd.snapshot_commit(str(repo))
    mark = tmp_path / "ran"
    commit = _commit_over(repo, lambda worktree: (worktree / "tools" / "clean.sh").write_text(
        f"#!/bin/sh\necho ran >> {mark}\ncat\n", encoding="utf-8"), base)
    mark.unlink(missing_ok=True)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes tools/clean.sh, which git reads its settings from in {repo} (")
    gitcmd.snapshot_commit(str(repo))
    assert not mark.exists()
    commit = _commit_over(repo, lambda worktree: (worktree / "src" / "a.py").write_text(
        "a = 2\n", encoding="utf-8"), base)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"]["paths"] == ["src/a.py"]


def test_a_change_beside_the_places_git_reads_its_settings_from_is_put_in(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repo = _repo(tmp_path / "repo")
    (repo / "conf").mkdir()
    (repo / "conf" / "ignore").write_text("*.swp\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "settings kept in the tree")
    _read_rules_in_the_tree(repo, None)
    base = gitcmd.require_head(str(repo))

    def write(worktree):
        (worktree / "conf" / "ignored.txt").write_text("x\n", encoding="utf-8")
        (worktree / "conf.py").write_text("c = 1\n", encoding="utf-8")

    land_dict = sandbox.try_land(str(repo), base, _commit_over(repo, write, base),
                                 str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"]["paths"] == ["conf.py", "conf/ignored.txt"]


def test_settings_git_cannot_list_the_places_of_keep_a_change_out(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: (clone / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8"), base)
    monkeypatch.setattr(gitcmd, "setting_place_list", lambda project_dir: None)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_refuse"] is True and land_dict["land_why"].startswith(
        f"the change writes kept.txt, and git did not say which files it reads its "
        f"settings from in {repo}")


def test_a_place_is_followed_through_every_link_on_its_way(tmp_path):
    (tmp_path / "real" / "deep").mkdir(parents=True)
    (tmp_path / "one").symlink_to("real")
    (tmp_path / "real" / "deep" / "two").symlink_to("../file")
    place = str(tmp_path / "one" / "deep" / "two")
    assert sandbox._place_on_way_list(place) == [
        place, str(tmp_path / "real" / "deep" / "two"),
        str(tmp_path / "real" / "deep" / ".." / "file"), str(tmp_path / "one"),
        str(tmp_path / "real" / "deep" / "two"), str(tmp_path / "real" / "file")]
    (tmp_path / "loop").symlink_to("loop")
    assert len(sandbox._place_on_way_list(str(tmp_path / "loop" / "x"))) \
        == 2 * sandbox._MAX_LINK_HOP + 2


def _ignore_in_exclude(repo):
    with open(repo / ".git" / "info" / "exclude", "a", encoding="utf-8") as handle:
        handle.write("secret/\n")


def _ignore_in_file_of_its_own(repo):
    (repo / "ignore_rules").write_text("far/\n", encoding="utf-8")
    _git(repo, "config", "core.excludesFile", str(repo / "ignore_rules"))


def _ignore_below(repo):
    (repo / "sub").mkdir()
    (repo / "sub" / ".gitignore").write_text("*.tmp\n", encoding="utf-8")


def _drop_the_rule(clone):
    (clone / ".gitignore").write_text("*.log\n", encoding="utf-8")


def _bring_a_rule(clone):
    (clone / "gen").mkdir(exist_ok=True)
    (clone / "gen" / ".gitignore").write_text("*\n", encoding="utf-8")


def _bring_a_rule_at_the_top(clone):
    (clone / ".gitignore").write_text(".venv/\n*.log\nwarnings.py\n", encoding="utf-8")


@pytest.mark.parametrize("rel, prepare, also_write, name_list", [
    (".venv/lib/python3.12/site-packages/new.pth", None, None, None),
    ("debug.log", None, None, None),
    ("*.log", None, None, None),
    ("secret/key.txt", _ignore_in_exclude, None, None),
    ("far/x.py", _ignore_in_file_of_its_own, None, None),
    ("sub/new.tmp", _ignore_below, None, None),
    (".venv/lib/new.pth", None, _drop_the_rule, None),
    ("gen/x.py", None, _bring_a_rule, ["gen/.gitignore", "gen/x.py"]),
    ("warnings.py", None, _bring_a_rule_at_the_top, None),
])
def test_a_change_to_a_path_git_ignores_is_not_put_into_the_tree(
        tmp_path, rel, prepare, also_write, name_list):
    """The rule is one the working tree holds, wherever git reads it from and
    whatever the change makes of it, or one the change brings along; the rest of the
    change is not put in either."""
    repo = _repo(tmp_path / "repo")
    (repo / ".gitignore").write_text(".venv/\n*.log\n", encoding="utf-8")
    (repo / "tracked.log").write_text("one\n", encoding="utf-8")
    if prepare:
        prepare(repo)
    _git(repo, "add", "-A", "--force")
    _git(repo, "commit", "--quiet", "-m", "rules")
    base = gitcmd.require_head(str(repo))

    def write(clone):
        (clone / rel).parent.mkdir(parents=True, exist_ok=True)
        (clone / rel).write_text("x\n", encoding="utf-8")
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")
        if also_write:
            also_write(clone)

    commit = _commit_of(repo, write, base)
    before = _git(repo, "status", "--porcelain", "-uall", "--ignored")
    with pytest.raises(sandbox.LandRefusal) as refusal:
        sandbox.land_in_tree(str(repo), base, commit, str(tmp_path / "land.diff"),
                             str(tmp_path / "land-last.diff"))
    text, command = refusal.value.why, refusal.value.command
    assert command == (f"git -C {repo} apply --binary --whitespace=nowarn --no-ignore-whitespace "
                       f"{tmp_path / 'land.diff'}")
    assert text.startswith(f"the change writes {', '.join(name_list or [rel])}, which "
                           f"git ignores in {repo}")
    assert text.splitlines() == [text] and str(tmp_path / "land.diff") in text
    assert _git(repo, "status", "--porcelain", "-uall", "--ignored") == before
    assert (tmp_path / "land.diff").is_file() and sandbox.land_list() == []
    assert sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                            str(tmp_path / "land-last.diff")) == {
        "land": None, "land_why": text, "land_command": command, "land_refuse": True}


def test_a_change_to_a_path_whose_name_is_not_utf8_is_asked_for_by_its_bytes(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / ".gitignore").write_bytes(b"\xff*.pth\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "rules")
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: (clone / os.fsdecode(b"\xffa.pth")).write_text(
        "x\n", encoding="utf-8"), base)
    assert sandbox._ignore_path_list(str(repo), {}, ["\ufffda.pth".encode("utf-8")], [],
                                     set()) == (set(), [])
    with pytest.raises(sandbox.LandRefusal, match=r"the change writes \\xffa\.pth, which "
                                                  r"git ignores"):
        sandbox.land_in_tree(str(repo), base, commit, str(tmp_path / "land.diff"),
                             str(tmp_path / "land-last.diff"))


def test_a_change_git_gives_no_answer_for_is_not_put_into_the_tree(tmp_path):
    """git says nothing of a path below a link; the change that makes the link a
    directory is refused as one it does not answer for."""
    repo = _repo(tmp_path / "repo")
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (repo / "link").symlink_to("src")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a link")
    base = gitcmd.require_head(str(repo))

    def write(clone):
        (clone / "link").unlink()
        (clone / "link").mkdir()
        (clone / "link" / "inner.py").write_text("i = 1\n", encoding="utf-8")

    commit = _commit_of(repo, write, base)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes link/inner.py, and git did not say whether it ignores that "
        f"in {repo}, so nothing is put into the working tree.")
    assert (repo / "link").is_symlink()


def test_a_change_to_what_git_tracks_or_does_not_ignore_is_put_into_the_tree(tmp_path):
    """A file the index names is none git ignores, whatever rule covers its name; a
    new file no rule covers, the rules themselves and a rule file that leaves itself
    out of what it covers are changes as any other is."""
    repo = _repo(tmp_path / "repo")
    (repo / ".gitignore").write_text(".venv/\n*.log\n", encoding="utf-8")
    for rel in ("tracked.log", "gone.log", ".venv/pin.txt"):
        (repo / rel).parent.mkdir(exist_ok=True)
        (repo / rel).write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A", "--force")
    _git(repo, "commit", "--quiet", "-m", "rules")
    base = gitcmd.require_head(str(repo))

    def write(clone):
        (clone / "tracked.log").write_text("two\n", encoding="utf-8")
        (clone / ".venv" / "pin.txt").write_text("two\n", encoding="utf-8")
        (clone / "gone.log").unlink()
        (clone / "new.py").write_text("n = 1\n", encoding="utf-8")
        (clone / ":(top)new.py").write_text("n = 1\n", encoding="utf-8")
        (clone / ".gitignore").write_text(".venv/\n*.log\ndist/\n", encoding="utf-8")
        (clone / "logs").mkdir()
        (clone / "logs" / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")

    commit = _commit_of(repo, write, base)
    # The user's own file under a rule, which no commit holds
    (repo / ".venv" / "own.py").write_text("own = 1\n", encoding="utf-8")
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_why"] == "" and land_dict["land_refuse"] is False
    assert land_dict["land"]["paths"] == [
        ".gitignore", ".venv/pin.txt", ":(top)new.py", "gone.log", "logs/.gitignore",
        "new.py", "tracked.log"]
    assert _git(repo, "status", "--porcelain", "-uall").splitlines() == [
        " M .gitignore", " M .venv/pin.txt", " D gone.log", " M tracked.log",
        "?? :(top)new.py", "?? logs/.gitignore", "?? new.py"]


def _commit_over(repo, write, base):
    """Commit what write() makes of the files a commit holds, in a worktree of its
    own, and return the commit. The repository's working tree is left as it is."""
    worktree = repo.parent / "over"
    _git(repo, "worktree", "add", "--quiet", "--detach", str(worktree), base)
    write(worktree)
    _git(worktree, "add", "-A", "--force")
    _git(worktree, "commit", "--quiet", "-m", "what a run landed")
    commit = gitcmd.require_head(str(worktree))
    _git(repo, "worktree", "remove", "--force", str(worktree))
    return commit


def _edit_beside_a_rule(clone):
    (clone / "pkg" / "a.py").write_text("a = 2\n", encoding="utf-8")
    (clone / "pkg" / "gen" / "a.out").write_text("two\n", encoding="utf-8")
    (clone / ".gitignore").write_text("gen/\n", encoding="utf-8")


def _rule_alone(clone):
    (clone / ".gitignore").write_text("gen/\n", encoding="utf-8")


def _remove_under_a_rule(clone):
    (clone / "pkg" / "gen" / "a.out").unlink()
    (clone / ".gitignore").write_text("gen/\n", encoding="utf-8")


def _edit_alone(clone):
    (clone / "pkg" / "a.py").write_text("a = 2\n", encoding="utf-8")
    (clone / "pkg" / "gen" / "a.out").write_text("two\n", encoding="utf-8")


@pytest.mark.parametrize("write, path_list, no_status_list", [
    (_edit_beside_a_rule, [".gitignore", "pkg/a.py", "pkg/gen/a.out"], ["pkg/gen/a.out"]),
    (_rule_alone, [".gitignore"], []),
    (_remove_under_a_rule, [".gitignore", "pkg/gen/a.out"], []),
    (_edit_alone, ["pkg/a.py", "pkg/gen/a.out"], []),
])
def test_a_rule_a_change_brings_is_not_held_against_what_the_tree_held_before_it(
        tmp_path, write, path_list, no_status_list):
    """The file the rule covers was in the working tree before the run, which git
    did not track, and the change rewrites it, removes it or leaves it. One it
    rewrites, which git status named before and does not name now, is named."""
    repo = _repo(tmp_path / "repo")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "a.py").write_text("a = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a package")
    (repo / "pkg" / "gen").mkdir()
    (repo / "pkg" / "gen" / "a.out").write_text("one\n", encoding="utf-8")
    base = gitcmd.snapshot_commit(str(repo))
    commit = _commit_over(repo, write, base)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_why"] == "" and land_dict["land"]["paths"] == path_list
    assert land_dict["land"]["no_status"] == no_status_list


def _land_twice(tmp_path, first, second, resume=False, before_second=None):
    """Put the change first() makes into a working tree, then the one second() makes
    of the tree that holds it. With resume the second is put in by a process that did
    not put in the first, and is told where the first started. before_second is
    called with the repository once both commits are made."""
    repo = _repo(tmp_path / "repo")
    start = gitcmd.require_head(str(repo))
    one = _commit_of(repo, first, start)
    assert sandbox.try_land(str(repo), start, one, str(tmp_path / "one.diff"),
                            str(tmp_path / "land-last.diff"))["land"]
    base = gitcmd.snapshot_commit(str(repo))
    two = _commit_over(repo, second, base)
    if resume:
        sandbox.reset_land_list()
    before = _git(repo, "status", "--porcelain", "-uall", "--ignored")
    if before_second:
        before_second(repo)
    land_dict = sandbox.try_land(str(repo), base, two, str(tmp_path / "two.diff"),
                                 str(tmp_path / "land-last.diff"),
                                 earlier_base=start if resume else "")
    return repo, before, land_dict


def _add_a_tool(clone):
    (clone / "tools").mkdir()
    (clone / "tools" / "conf.py").write_text("c = 1\n", encoding="utf-8")


def _cover_the_tool(clone):
    (clone / ".gitignore").write_text("tools/\n", encoding="utf-8")
    (clone / "b.py").write_text("b = 1\n", encoding="utf-8")


def _cover_the_tool_and_edit_it(clone):
    (clone / ".gitignore").write_text("/tools/\n", encoding="utf-8")
    (clone / "tools" / "conf.py").write_text("c = 2\n", encoding="utf-8")


@pytest.mark.parametrize("second", [_cover_the_tool, _cover_the_tool_and_edit_it])
@pytest.mark.parametrize("resume", [False, True])
def test_a_rule_that_covers_what_an_earlier_change_added_keeps_the_later_one_out(
        tmp_path, second, resume):
    """The file was put in by this process, or by the run carried on from."""
    repo, before, land_dict = _land_twice(tmp_path, _add_a_tool, second, resume)
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    lead = ("the change writes tools/conf.py, which git ignores"
            if second is _cover_the_tool_and_edit_it else
            "the change writes .gitignore files whose rules cover tools/conf.py, which "
            f"was put into the working tree of {repo} before it")
    assert land_dict["land_why"].startswith(lead)
    assert _git(repo, "status", "--porcelain", "-uall", "--ignored") == before
    assert before.splitlines() == ["?? tools/conf.py"]


def test_a_rule_that_covers_what_an_earlier_change_only_rewrote_is_put_in(tmp_path):
    """A file git tracks shows whatever rule covers it."""
    def rewrite(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    def cover(clone):
        (clone / ".gitignore").write_text("kept.txt\n", encoding="utf-8")

    repo, _, land_dict = _land_twice(tmp_path, rewrite, cover)
    assert land_dict["land"]["paths"] == [".gitignore"]
    assert _git(repo, "status", "--porcelain", "-uall").splitlines() == [
        " M kept.txt", "?? .gitignore"]


def _nest_untracked(repo):
    return ["?? vendor/lib/"]


def _nest_in_the_index(repo):
    _git(repo, "-c", "advice.addEmbeddedRepo=false", "add", "vendor/lib")
    _git(repo, "commit", "--quiet", "-m", "a repository of its own")
    return []


def _write_inside(worktree):
    _git(worktree, "rm", "--quiet", "--cached", "vendor/lib")
    (worktree / "vendor" / "lib").mkdir(parents=True, exist_ok=True)
    (worktree / "vendor" / "lib" / "new.py").write_text("n = 1\n", encoding="utf-8")


def _remove_the_entry(worktree):
    _git(worktree, "rm", "--quiet", "--cached", "vendor/lib")


@pytest.mark.parametrize("prepare", [_nest_untracked, _nest_in_the_index])
@pytest.mark.parametrize("write, name", [
    (_write_inside, "vendor/lib, vendor/lib/new.py"), (_remove_the_entry, "vendor/lib")])
def test_a_change_at_or_under_a_repository_of_its_own_is_not_put_into_the_tree(
        tmp_path, prepare, write, name):
    """The directory holds a .git of its own, whether or not the index names it."""
    repo = _repo(tmp_path / "repo")
    inner = _repo(repo / "vendor" / "lib")
    status_line_list = prepare(repo)
    base = gitcmd.snapshot_commit(str(repo))
    commit = _commit_over(repo, write, base)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes {name}, at or under vendor/lib, a repository of its own in "
        f"{repo}, so git status would not show it, so nothing is put")
    assert not (inner / "new.py").exists() and (inner / "kept.txt").is_file()
    assert _git(repo, "status", "--porcelain", "-uall").splitlines() == status_line_list
    assert _git(inner, "status", "--porcelain") == ""


def test_a_new_repository_of_its_own_a_change_names_is_not_put_into_the_tree(tmp_path):
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))

    def write(worktree):
        _git(worktree, "update-index", "--add", "--cacheinfo", f"160000,{base},dep")
        (worktree / "dep").mkdir()

    commit = _commit_over(repo, write, base)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith("the change writes dep, at or under dep, ")
    assert not (repo / "dep").exists()


def test_the_command_for_a_change_that_moves_a_submodule_moves_it(tmp_path):
    """git apply leaves a submodule at its commit; the command checks out the commit
    the change moves it to once the diff is applied."""
    repo = _repo(tmp_path / "repo")
    inner = _repo(repo / "sub")
    first = gitcmd.require_head(str(inner))
    (inner / "kept.txt").write_text("two\n", encoding="utf-8")
    _git(inner, "commit", "--quiet", "-am", "second")
    second = gitcmd.require_head(str(inner))
    _git(inner, "checkout", "--quiet", "--detach", first)
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first},sub")
    _git(repo, "commit", "--quiet", "-m", "a submodule")
    base = gitcmd.require_head(str(repo))

    def write(worktree):
        _git(worktree, "update-index", "--cacheinfo", f"160000,{second},sub")
        (worktree / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_refuse"] is True
    assert land_dict["land_why"].endswith(
        "apply it by hand (git apply moves no submodule, so the command first stops "
        "unless sub is at the commit the change moves it from, and then checks out "
        "there the commit the change moves it to, which has to be there) with:")
    sub_git = ["git", "-C", str(repo / "sub"), "--git-dir=.git"]
    stop = shlex.join(["printf", r"%s\n", f"nothing was applied: sub is not at {first}, "
                       "the commit the change moves it from"])
    assert land_dict["land_command"] == " && ".join([
        shlex.join([*sub_git, "merge-base", "--is-ancestor", "HEAD", first]),
        shlex.join([*sub_git, "merge-base", "--is-ancestor", first, "HEAD"])
        + f" || {{ {stop} >&2; false; }}",
        shlex.join(["git", "-C", str(repo), *gitcmd.APPLY_TUPLE,
                    str(tmp_path / "land.diff")]),
        shlex.join([*sub_git, "checkout", "--detach", second])])
    subprocess.run(["sh", "-c", land_dict["land_command"]], check=True,
                   capture_output=True)
    assert gitcmd.require_head(str(inner)) == second
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\ntwo\n"


def test_the_command_for_a_submodule_with_no_repository_leaves_the_outer_head(
        tmp_path):
    """A submodule directory with no .git stops the checkout instead of moving the
    repository around it."""
    repo = _repo(tmp_path / "repo")
    inner = _repo(tmp_path / "inner")
    first = gitcmd.require_head(str(inner))
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first},sub")
    _git(repo, "commit", "--quiet", "-m", "a submodule")
    (repo / "sub").mkdir()
    base = gitcmd.require_head(str(repo))

    def write(worktree):
        _git(worktree, "update-index", "--cacheinfo", f"160000,{base},sub")

    commit = _commit_over(repo, write, base)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_refuse"] is True
    proc = subprocess.run(["sh", "-c", land_dict["land_command"]],
                          capture_output=True)
    assert proc.returncode != 0
    assert gitcmd.require_head(str(repo)) == base
    assert _git(repo, "symbolic-ref", "--quiet", "HEAD").strip()


def test_the_command_checks_out_only_the_submodules_the_index_holds(tmp_path):
    """A patch that names a path the index holds no submodule at, or one that leads
    out of the repository through a link, gets no checkout in the command."""
    repo = _repo(tmp_path / "repo")
    inner = _repo(repo / "sub")
    first = gitcmd.require_head(str(inner))
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first},sub")
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first},link/sub")
    _git(repo, "commit", "--quiet", "-m", "submodules")
    other = _repo(tmp_path / "other")
    (other / "sub").mkdir()
    (repo / "link").symlink_to(other)
    second = "1" * 40
    patch = "".join(
        f"diff --git a/{path} b/{path}\nindex {first[:7]}..{second[:7]} 160000\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-Subproject commit {first}\n"
        f"+Subproject commit {second}\n"
        for path in ("sub", "plain", "link/sub"))
    patch_path = tmp_path / "out.diff"
    patch_path.write_text(patch, encoding="utf-8")
    move_text, command = sandbox.by_hand_guide(str(repo), str(patch_path))
    assert move_text.startswith(" (git apply moves no submodule")
    sub_git = ["git", "-C", str(repo / "sub"), "--git-dir=.git"]
    stop = shlex.join(["printf", r"%s\n", f"nothing was applied: sub is not at {first}, "
                       "the commit the change moves it from"])
    assert command == " && ".join([
        shlex.join([*sub_git, "merge-base", "--is-ancestor", "HEAD", first]),
        shlex.join([*sub_git, "merge-base", "--is-ancestor", first, "HEAD"])
        + f" || {{ {stop} >&2; false; }}",
        shlex.join(["git", "-C", str(repo), *gitcmd.APPLY_TUPLE, str(patch_path)]),
        shlex.join([*sub_git, "checkout", "--detach", second])])


def test_the_command_stops_for_a_submodule_off_the_commit_the_change_moves_from(
        tmp_path):
    """A submodule at another commit than the one the change moves it from, ahead of
    it or behind it, stops the command before anything is applied; one at that
    commit is moved."""
    repo = _repo(tmp_path / "repo")
    inner = _repo(repo / "sub")
    commit_list = [gitcmd.require_head(str(inner))]
    for text in ("two\n", "three\n"):
        (inner / "kept.txt").write_text(text, encoding="utf-8")
        _git(inner, "commit", "--quiet", "-am", text.strip())
        commit_list.append(gitcmd.require_head(str(inner)))
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit_list[1]},sub")
    _git(repo, "commit", "--quiet", "-m", "a submodule")
    patch_path = tmp_path / "move.diff"
    patch_path.write_text(
        f"diff --git a/sub b/sub\nindex {commit_list[1][:7]}..{commit_list[2][:7]} "
        f"160000\n--- a/sub\n+++ b/sub\n@@ -1 +1 @@\n-Subproject commit {commit_list[1]}"
        f"\n+Subproject commit {commit_list[2]}\ndiff --git a/kept.txt b/kept.txt\n"
        f"--- a/kept.txt\n+++ b/kept.txt\n@@ -1 +1 @@\n-one\n+changed\n",
        encoding="utf-8")
    _, command = sandbox.by_hand_guide(str(repo), str(patch_path))
    for at in (commit_list[0], commit_list[2]):
        _git(inner, "checkout", "--quiet", "--detach", at)
        proc = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        assert proc.returncode != 0 and proc.stdout == ""
        assert proc.stderr == (f"nothing was applied: sub is not at {commit_list[1]}, "
                               "the commit the change moves it from\n")
        assert gitcmd.require_head(str(inner)) == at
        assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"
    _git(inner, "checkout", "--quiet", "--detach", commit_list[1])
    proc = subprocess.run(["sh", "-c", command], check=True, capture_output=True,
                          text=True)
    assert "nothing was applied" not in proc.stderr
    assert gitcmd.require_head(str(inner)) == commit_list[2]
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "changed\n"


def test_the_command_stopped_by_one_of_two_submodules_says_so_in_one_line(tmp_path):
    """With two submodules the change moves, one off its commit stops the command
    with one line naming both and moves neither."""
    repo = _repo(tmp_path / "repo")
    move_dict = {}
    for name in ("lib", "sub"):
        inner = _repo(repo / name)
        first = gitcmd.require_head(str(inner))
        (inner / "kept.txt").write_text("two\n", encoding="utf-8")
        _git(inner, "commit", "--quiet", "-am", "second")
        move_dict[name] = (first, gitcmd.require_head(str(inner)))
        _git(inner, "checkout", "--quiet", "--detach", first)
        _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first},{name}")
    _git(repo, "commit", "--quiet", "-m", "submodules")
    patch_path = tmp_path / "move.diff"
    patch_path.write_text("".join(
        f"diff --git a/{name} b/{name}\nindex {one[:7]}..{two[:7]} 160000\n--- a/{name}\n"
        f"+++ b/{name}\n@@ -1 +1 @@\n-Subproject commit {one}\n+Subproject commit {two}\n"
        for name, (one, two) in move_dict.items()), encoding="utf-8")
    _, command = sandbox.by_hand_guide(str(repo), str(patch_path))
    _git(repo / "sub", "checkout", "--quiet", "--detach", move_dict["sub"][1])
    proc = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
    assert proc.returncode != 0 and proc.stdout == ""
    assert proc.stderr == (f"nothing was applied: lib is not at {move_dict['lib'][0]} or "
                           f"sub is not at {move_dict['sub'][0]}, the commit the change "
                           "moves it from\n")
    assert gitcmd.require_head(str(repo / "lib")) == move_dict["lib"][0]
    assert gitcmd.require_head(str(repo / "sub")) == move_dict["sub"][1]


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
@pytest.mark.parametrize("change", ["rewrite", "remove"])
def test_a_change_to_a_file_git_is_set_not_to_look_at_is_not_put_into_the_tree(
        tmp_path, flag, change):
    repo = _repo(tmp_path / "repo")
    (repo / "local.yml").write_text("a: 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a file kept for this machine")
    _git(repo, "update-index", flag, "local.yml")
    (repo / "local.yml").write_text("a: 1\nown: 1\n", encoding="utf-8")
    assert _git(repo, "status", "--porcelain") == ""
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        if change == "remove":
            (worktree / "local.yml").unlink()
        else:
            (worktree / "local.yml").write_text("a: 2\nown: 1\n", encoding="utf-8")
        (worktree / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes local.yml, which git is set not to look at in {repo} "
        f"(assume-unchanged or skip-worktree), so git status would not show it")
    assert (repo / "local.yml").read_text(encoding="utf-8") == "a: 1\nown: 1\n"
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"


def test_a_change_beside_a_file_git_is_set_not_to_look_at_is_put_into_the_tree(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "local.yml").write_text("a: 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a file kept for this machine")
    _git(repo, "update-index", "--skip-worktree", "local.yml")
    base = gitcmd.snapshot_commit(str(repo))
    commit = _commit_over(repo, lambda worktree: (worktree / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8"), base)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"]["paths"] == ["kept.txt"]


def _tree_state(repo):
    """Return every entry of a working tree but .git: its kind, its mode, and what a
    file holds or a link points at."""
    state_dict = {}
    for root, dir_list, file_list in os.walk(repo):
        if ".git" in dir_list and root == str(repo):
            dir_list.remove(".git")
        for name in dir_list + file_list:
            path = os.path.join(root, name)
            mode = os.lstat(path).st_mode
            state_dict[os.path.relpath(path, repo)] = (
                mode, os.readlink(path) if os.path.islink(path)
                else None if os.path.isdir(path) else open(path, "rb").read())
    return state_dict


def _hide_under_a_rule(repo):
    (repo / ".gitignore").write_text("hid/\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a rule")
    return "hid/new.py", lambda worktree: (worktree / "hid").mkdir() or (
        worktree / "hid" / "new.py").write_text("n = 1\n", encoding="utf-8")


def _hide_under_a_flag(repo):
    _git(repo, "update-index", "--skip-worktree", "kept.txt")
    return "kept.txt", lambda worktree: None


def _hide_in_a_repository_of_its_own(repo):
    _repo(repo / "vendor" / "lib")
    return "vendor/lib/new.py", lambda worktree: (
        _git(worktree, "rm", "--quiet", "--cached", "vendor/lib"),
        (worktree / "vendor" / "lib").mkdir(parents=True, exist_ok=True),
        (worktree / "vendor" / "lib" / "new.py").write_text("n = 1\n", encoding="utf-8"))


def _hide_under_the_rules_of_a_setting(repo):
    (repo / "rules").write_text("*.swp\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "rules")
    _git(repo, "config", "core.excludesFile", "rules")
    return "tools/x.py", lambda worktree: (
        (worktree / "rules").write_text("*.swp\ntools/\n", encoding="utf-8"),
        (worktree / "tools").mkdir(),
        (worktree / "tools" / "x.py").write_text("x = 1\n", encoding="utf-8"))


@pytest.mark.parametrize("prepare", [
    _hide_under_a_rule, _hide_under_a_flag, _hide_in_a_repository_of_its_own,
    _hide_under_the_rules_of_a_setting])
def test_a_change_git_status_leaves_a_path_of_out_is_taken_back_out(
        tmp_path, monkeypatch, prepare):
    """Whatever keeps the path out of what git status shows, and with none of the
    checks made before the change is applied naming it: the working tree is put back
    entry by entry, the user's own files, modes and directories with it."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    repo = _repo(tmp_path / "repo")
    for rel in ("only/last.txt", "run.sh", "moved/from.txt"):
        (repo / rel).parent.mkdir(exist_ok=True)
        (repo / rel).write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / "link").symlink_to("kept.txt")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "files of every kind")
    name, write_hidden = prepare(repo)
    (repo / "mine").mkdir()
    (repo / "mine" / "notes.txt").write_text("the user's own\n", encoding="utf-8")
    (repo / "mine" / "notes.txt").chmod(0o600)
    (repo / "empty" / "inner").mkdir(parents=True)
    (repo / "only").chmod(0o700)
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")
        (worktree / "mine" / "notes.txt").write_text("rewritten\n", encoding="utf-8")
        (worktree / "only" / "last.txt").unlink()
        (worktree / "run.sh").chmod(0o755)
        (worktree / "link").unlink()
        (worktree / "link").symlink_to("run.sh")
        (worktree / "new" / "deep").mkdir(parents=True)
        (worktree / "new" / "deep" / "file.py").write_text("n = 1\n", encoding="utf-8")
        (worktree / "empty" / "inner").mkdir(parents=True)
        (worktree / "empty" / "inner" / "x.py").write_text("x = 1\n", encoding="utf-8")
        os.rename(worktree / "moved" / "from.txt", worktree / "to.txt")
        (worktree / "blob.bin").write_bytes(bytes(range(256)))
        write_hidden(worktree)

    commit = _commit_over(repo, write, base)
    before = _git(repo, "status", "--porcelain", "-uall", "--ignored")
    state_dict = _tree_state(repo)
    monkeypatch.setattr(sandbox, "_refusal_text", lambda *args, **kwargs: "")
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"] == (
        f"the change writes {name}, which git status does not show in {repo} once it "
        f"is there, so it was taken back out and nothing is put into the working tree. "
        f"The change is written to {tmp_path / 'land.diff'}; read it, and apply it by "
        f"hand with:")
    assert land_dict["land_command"] == (
        f"git -C {repo} apply --binary --whitespace=nowarn --no-ignore-whitespace {tmp_path / 'land.diff'}")
    assert _tree_state(repo) == state_dict
    assert _git(repo, "status", "--porcelain", "-uall", "--ignored") == before
    assert sandbox.land_list() == [] and not (tmp_path / "land-last.diff").exists()


@pytest.mark.parametrize("resume", [False, True])
def test_a_change_that_hides_what_an_earlier_one_added_is_taken_back_out(
        tmp_path, monkeypatch, resume):
    """With none of the checks made before the change is applied naming it."""
    repo, before, land_dict = _land_twice(
        tmp_path, _add_a_tool, _cover_the_tool, resume,
        before_second=lambda repo: monkeypatch.setattr(
            sandbox, "_refusal_text", lambda *args, **kwargs: ""))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"with the change in the working tree, git status no longer shows "
        f"tools/conf.py, which was put into the working tree of {repo} before it, so it "
        f"was taken back out and nothing is put into the working tree.")
    assert _git(repo, "status", "--porcelain", "-uall", "--ignored") == before
    assert not (repo / "b.py").exists() and not (repo / ".gitignore").exists()


def test_a_path_the_user_hid_is_not_held_against_a_later_change(tmp_path):
    """The earlier change's file is under a rule of the user's own by the time the
    later change, which does not touch it, is put in."""
    def hide(repo):
        with open(repo / ".git" / "info" / "exclude", "a", encoding="utf-8") as handle:
            handle.write("tools/\n")

    def add_another(worktree):
        (worktree / "b.py").write_text("b = 1\n", encoding="utf-8")

    repo, _, land_dict = _land_twice(tmp_path, _add_a_tool, add_another, resume=True,
                                     before_second=hide)
    assert land_dict["land"]["paths"] == ["b.py"]


def test_a_change_git_does_not_answer_for_once_it_is_in_is_taken_back_out(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: (clone / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8"), base)
    answer_list = [set()]
    monkeypatch.setattr(gitcmd, "status_path_set",
                        lambda project_dir: answer_list.pop() if answer_list else None)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_refuse"] is True and land_dict["land_why"].startswith(
        f"with the change in the working tree of {repo}, git did not say what git "
        f"status shows of kept.txt, so it was taken back out")
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"


def test_a_change_that_cannot_be_taken_back_out_stops_with_what_is_left(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("hid/\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a rule")
    base = gitcmd.require_head(str(repo))

    def write(clone):
        (clone / "hid").mkdir()
        (clone / "hid" / "new.py").write_text("n = 1\n", encoding="utf-8")
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    commit = _commit_of(repo, write, base)
    real_copy = shutil.copyfile

    def no_copy_back(source, target, **kwargs):
        if os.fsdecode(target) == str(repo / "kept.txt"):
            raise OSError("stand-in")
        return real_copy(source, target, **kwargs)

    monkeypatch.setattr(sandbox, "_refusal_text", lambda *args, **kwargs: "")
    monkeypatch.setattr(shutil, "copyfile", no_copy_back)
    with pytest.raises(sandbox.LandUndoError) as stop:
        sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                         str(tmp_path / "kept" / "land-last.diff"))
    keep_dir, = (tmp_path / "kept").iterdir()
    assert keep_dir.name.startswith(f"land-before-{os.getpid()}-")
    assert str(stop.value) == (
        f"the change writes hid/new.py, which git status does not show in {repo} once "
        f"it is there, and the working tree could not be put back as it was: kept.txt "
        f"in {repo} is not as it was before the change was put in (stand-in). What the "
        f"paths the change touches held before is kept in {keep_dir}, and the change is "
        f"written to {tmp_path / 'land.diff'}; put the working tree right from the two "
        f"by hand")
    assert (keep_dir / "tree" / "kept.txt").read_text(encoding="utf-8") == "one\n"
    assert json.loads((keep_dir / "before.json").read_text(encoding="utf-8")) == {
        "project": str(repo), "link": {}, "none": ["hid", "hid/new.py"]}
    assert not (repo / "hid").exists()
    assert sandbox.land_list() == []


def test_what_a_change_s_paths_held_before_is_kept_where_no_agent_s_copy_reads_it(
        tmp_path, monkeypatch):
    """The copy that stays is in this tool's own directory of the repository, which no
    agent's copy of the output directory is made from."""
    repo = _repo(tmp_path / "repo")
    out, built = _built(tmp_path)
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: (clone / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8"), base)
    monkeypatch.setattr(sandbox, "_no_show_text", lambda *args, **kwargs: "a stand-in")

    def no_put_back(project_dir, state_dict, keep_dir, patch_path, why):
        raise sandbox.LandUndoError(why)

    monkeypatch.setattr(sandbox, "_put_back", no_put_back)
    with pytest.raises(sandbox.LandUndoError):
        sandbox.land_in_tree(str(repo), base, commit, str(tmp_path / "land.diff"),
                             sandbox.last_land_path(str(out), "repo"))
    keep_dir, = pathlib.Path(sandbox.repo_own_dir(str(out), "repo")).iterdir()
    assert keep_dir.name.startswith("land-before-") and not list(built.rglob("land-*"))
    mine = pathlib.Path(sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "", None,
                                               False))
    assert not [path for path in mine.rglob("*") if path.name.startswith("land-")]


def _rewrite_under_a_directory_nothing_is_written_in(repo):
    (repo / "ro").mkdir()
    (repo / "ro" / "c.py").write_text("c = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a directory")
    (repo / "ro").chmod(0o555)
    return (lambda worktree: (worktree / "ro" / "c.py").write_text(
        "c = 2\n", encoding="utf-8")), lambda: None


def _add_a_file_where_ignored_files_are(repo):
    (repo / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a rule")
    (repo / "gen" / "cache").mkdir(parents=True)
    (repo / "gen" / "cache" / "x.pyc").write_text("x", encoding="utf-8")
    return (lambda worktree: (worktree / "gen").write_text(
        "#!/bin/sh\n", encoding="utf-8")), lambda: None


def _add_a_file_under_a_file_made_since(repo):
    return (lambda worktree: (worktree / "notes").mkdir() or (
        worktree / "notes" / "n.md").write_text("n\n", encoding="utf-8")), \
        lambda: (repo / "notes").write_text("the user's own\n", encoding="utf-8")


@pytest.mark.parametrize("prepare", [
    _rewrite_under_a_directory_nothing_is_written_in,
    _add_a_file_where_ignored_files_are, _add_a_file_under_a_file_made_since])
def test_a_change_git_stops_part_way_through_leaves_the_tree_as_it_was(tmp_path, prepare):
    """git passes the change, deletes the files it rewrites and stops at one it
    cannot write: the files, a line of the user's own that is not committed with
    them, are put back."""
    repo = _repo(tmp_path / "repo")
    for name in ("a.py", "z.py"):
        (repo / name).write_text("v = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "two files")
    write_more, change_since = prepare(repo)
    (repo / "z.py").write_text("v = 1\n# not committed\n", encoding="utf-8")
    (repo / "z.py").chmod(0o600)
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / "a.py").write_text("v = 2\n", encoding="utf-8")
        (worktree / "z.py").write_text("v = 2\n# not committed\n", encoding="utf-8")
        (worktree / "kept.txt").unlink()
        write_more(worktree)

    commit = _commit_over(repo, write, base)
    change_since()
    state_dict = _tree_state(repo)
    try:
        land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                     str(tmp_path / "kept" / "land-last.diff"))
        assert _tree_state(repo) == state_dict
    finally:
        if (repo / "ro").is_dir():
            (repo / "ro").chmod(0o755)
    assert land_dict["land"] is None and land_dict["land_refuse"] is False
    assert land_dict["land_why"].startswith(
        f"git did not get through putting the change into the working tree of {repo} (")
    assert land_dict["land_why"].endswith(
        f"), so it was taken back out and nothing is put into the working tree. The "
        f"change is written to {tmp_path / 'land.diff'}; read it, and apply it by hand "
        f"with:")
    assert sandbox.land_list() == [] and os.listdir(tmp_path / "kept") == []


def _add_under_a_directory_nothing_is_written_in(worktree):
    (worktree / "ro" / "sub").mkdir()
    (worktree / "ro" / "sub" / "n.py").write_text("n = 1\n", encoding="utf-8")


def _remove_under_a_directory_nothing_is_written_in(worktree):
    (worktree / "ro" / "c.py").unlink()
    (worktree / "ro" / "mine.txt").unlink()


@pytest.mark.parametrize("write_more, name", [
    (_add_under_a_directory_nothing_is_written_in, "ro/sub/n.py"),
    (_remove_under_a_directory_nothing_is_written_in, "ro/c.py, ro/mine.txt")])
def test_a_change_git_ends_without_writing_whole_is_taken_back_out(
        tmp_path, write_more, name):
    """git ends without an error, having passed over a file it could not make or
    one it could not delete."""
    repo = _repo(tmp_path / "repo")
    (repo / "ro").mkdir()
    (repo / "ro" / "c.py").write_text("c = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a directory")
    (repo / "ro" / "mine.txt").write_text("not tracked\n", encoding="utf-8")
    (repo / "ro").chmod(0o555)
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")
        write_more(worktree)

    commit = _commit_over(repo, write, base)
    state_dict = _tree_state(repo)
    try:
        land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                     str(tmp_path / "kept" / "land-last.diff"))
        assert _tree_state(repo) == state_dict
    finally:
        (repo / "ro").chmod(0o755)
    assert land_dict["land"] is None and land_dict["land_refuse"] is False
    assert land_dict["land_why"].startswith(
        f"git ended without an error and did not write or did not delete {name} in "
        f"the working tree of {repo} (")
    assert ", so it was taken back out and nothing is put into the working tree. " \
        in land_dict["land_why"]
    assert sandbox.land_list() == [] and os.listdir(tmp_path / "kept") == []


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("when", ["before git writes", "once git wrote",
                                  "while the tree is put back"])
def test_a_stop_while_a_change_is_put_in_leaves_the_tree_as_it_was(
        tmp_path, monkeypatch, stop_signal, when):
    """The signal is held until the working tree is as it was, then stops the
    command as it does anywhere else."""
    repo = _repo(tmp_path / "repo")
    (repo / "kept.txt").write_text("one\n# not committed\n", encoding="utf-8")
    (repo / "mine.txt").write_text("not tracked\n", encoding="utf-8")
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / "kept.txt").write_text("one\ntwo\n# not committed\n", encoding="utf-8")
        (worktree / "mine.txt").unlink()
        (worktree / "new" / "deep").mkdir(parents=True)
        (worktree / "new" / "deep" / "file.py").write_text("n = 1\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    state_dict = _tree_state(repo)
    real_write, real_back = sandbox._write_in_tree, sandbox._put_back
    seen_list = []

    def write_in_tree(*args):
        if when == "before git writes":
            signal.raise_signal(stop_signal)
        text = real_write(*args)
        assert _tree_state(repo) != state_dict
        if when == "once git wrote":
            signal.raise_signal(stop_signal)
        return "a stand-in for what git did not write" \
            if when == "while the tree is put back" else text

    def put_back(*args):
        if when == "while the tree is put back":
            signal.raise_signal(stop_signal)
        real_back(*args)
        seen_list.append(_tree_state(repo) == state_dict)

    def stop(number, frame):
        raise KeyboardInterrupt(f"signal {number}")

    monkeypatch.setattr(sandbox, "_write_in_tree", write_in_tree)
    monkeypatch.setattr(sandbox, "_put_back", put_back)
    handler_before = signal.signal(stop_signal, stop)
    try:
        with pytest.raises(KeyboardInterrupt, match=f"signal {int(stop_signal)}"):
            sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                             str(tmp_path / "kept" / "land-last.diff"))
        assert signal.getsignal(stop_signal) is stop
    finally:
        signal.signal(stop_signal, handler_before)
    assert seen_list == [True] and _tree_state(repo) == state_dict
    assert sandbox.land_list() == [] and os.listdir(tmp_path / "kept") == []


def test_an_error_raised_while_a_change_is_put_in_leaves_the_tree_as_it_was(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    (repo / "kept.txt").write_text("one\n# not committed\n", encoding="utf-8")
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        (worktree / "kept.txt").write_text("one\ntwo\n# not committed\n", encoding="utf-8")
        (worktree / "new.py").write_text("n = 1\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    state_dict = _tree_state(repo)
    real_view = sandbox._view_dict
    call_list = []

    def view_dict(*args):
        call_list.append(args)
        if len(call_list) == 2:
            assert _tree_state(repo) != state_dict
            raise RuntimeError("a stand-in for an error")
        return real_view(*args)

    monkeypatch.setattr(sandbox, "_view_dict", view_dict)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "kept" / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_why"] == "a stand-in for an error"
    assert _tree_state(repo) == state_dict
    assert sandbox.land_list() == [] and os.listdir(tmp_path / "kept") == []


def test_a_change_whose_files_cannot_be_copied_first_is_not_put_in(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: (clone / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8"), base)
    monkeypatch.setattr(sandbox, "_MAX_BEFORE_BYTE", 3)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "kept" / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is False
    assert land_dict["land_why"] == (
        f"what the change touches in the working tree of {repo} could not be copied "
        f"before the change is put in (the files hold more than 3 bytes), so nothing "
        f"is put into the working tree. The change is written to "
        f"{tmp_path / 'land.diff'}; read it, and apply it by hand with:")
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"
    assert os.listdir(tmp_path / "kept") == []


def test_a_copy_a_killed_process_left_is_named_and_stays_where_it_is(tmp_path, caplog):
    """The copy of a process that ended with its list written is named; one of
    another repository is passed over; one without its list is removed once its
    process has ended."""
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: (clone / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8"), base)
    ended = _a_run_that_ended().rsplit("-", 1)[1]
    keep_root = tmp_path / "kept"
    own, other, half, going = (keep_root / f"land-before-{pid}-{tail}" for pid, tail in (
        (ended, "own"), (ended, "other"), (ended, "half"), (os.getpid(), "going")))
    for keep_dir in (own, other, half, going):
        (keep_dir / "tree").mkdir(parents=True)
    (own / "before.json").write_text(json.dumps({"project": str(repo)}), encoding="utf-8")
    (other / "before.json").write_text(json.dumps({"project": "/another"}),
                                       encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        sandbox.land_in_tree(str(repo), base, commit, str(tmp_path / "land.diff"),
                             str(keep_root / "land-last.diff"))
    assert sorted(os.listdir(keep_root)) == sorted([
        own.name, other.name, going.name, "land-last.diff"])
    assert (f"a command ended before it was through putting a change into the working "
            f"tree of {repo}, which may hold part of that change: what the files it "
            f"touches held before is kept in {own / 'tree'}") in caplog.text
    assert str(other) not in caplog.text and str(going) not in caplog.text


def _leave_the_mode_to_the_index(repo):
    _git(repo, "config", "core.fileMode", "false")
    return lambda worktree: _git(worktree, "update-index", "--chmod=+x", "kept.txt")


def _edit_without_staging(repo):
    (repo / "kept.txt").write_text("one\nin the middle of this\n", encoding="utf-8")
    return lambda worktree: (worktree / "kept.txt").write_text("one\n", encoding="utf-8")


def _remove_without_staging(repo):
    (repo / "kept.txt").unlink()
    return lambda worktree: (worktree / "kept.txt").write_text("one\n", encoding="utf-8")


@pytest.mark.parametrize("prepare", [
    _leave_the_mode_to_the_index, _edit_without_staging, _remove_without_staging])
def test_a_file_git_reads_as_its_index_holds_it_is_put_in_and_named(
        tmp_path, caplog, prepare):
    """The mode alone changed where git does not compare it, and a change of the
    user's own that was not staged is put back or written over."""
    repo = _repo(tmp_path / "repo")
    change = prepare(repo)
    base = gitcmd.snapshot_commit(str(repo))

    def write(worktree):
        change(worktree)
        (worktree / "new.py").write_text("n = 1\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                     str(tmp_path / "land-last.diff"))
    assert land_dict["land"]["paths"] == ["kept.txt", "new.py"]
    assert land_dict["land"]["no_status"] == ["kept.txt"]
    assert _git(repo, "status", "--porcelain", "-uall") == "?? new.py\n"
    assert os.access(repo / "kept.txt", os.X_OK) is (prepare is _leave_the_mode_to_the_index)
    assert f"git status does not name kept.txt, which the run wrote in the working " \
           f"tree of {repo}" in caplog.text


def _name_a_filter(repo, mark):
    """Give the repository's git settings a filter driver named mark that leaves a
    line in the file mark each time it runs."""
    for kind in ("clean", "smudge"):
        _git(repo, "config", f"filter.mark.{kind}", f"sh -c 'echo {kind} %f >> {mark}; cat'")
    _git(repo, "config", "filter.mark.required", "true")


def _filter_repo(tmp_path, attribute_text):
    repo = _repo(tmp_path / "repo")
    (repo / ".gitattributes").write_text(attribute_text, encoding="utf-8")
    (repo / "old.dat").write_text("old\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "attributes")
    return repo


def _add_a_dat(worktree):
    (worktree / "new.dat").write_text("new\n", encoding="utf-8")


def _rewrite_a_dat(worktree):
    (worktree / "old.dat").write_text("old\nmore\n", encoding="utf-8")
    (worktree / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")


def _remove_a_dat(worktree):
    (worktree / "old.dat").unlink()


def _name_the_filter_and_add_a_dat(worktree):
    (worktree / "sub").mkdir()
    (worktree / "sub" / ".gitattributes").write_text("*.dat filter=mark\n",
                                                     encoding="utf-8")
    (worktree / "sub" / "new.dat").write_text("new\n", encoding="utf-8")


def _name_the_filter_in_place_of_another(worktree):
    (worktree / ".gitattributes").write_text("*.dat filter=mark\n", encoding="utf-8")
    (worktree / "old.dat").write_text("old\nmore\n", encoding="utf-8")
    (worktree / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")


def _name_the_filter_alone(worktree):
    (worktree / ".gitattributes").write_text("*.dat filter=mark\n*.cfg filter=mark\n",
                                             encoding="utf-8")
    (worktree / ".lfsconfig").write_text("[lfs]\n\turl = https://example.invalid\n",
                                         encoding="utf-8")


@pytest.mark.parametrize("attribute_text, write, name", [
    ("*.dat filter=mark\n", _add_a_dat, "new.dat"),
    ("*.dat filter=mark\n", _rewrite_a_dat, "old.dat"),
    ("*.dat filter=mark\n", _remove_a_dat, "old.dat"),
    ("*.dat -filter\n", _name_the_filter_and_add_a_dat, "sub/new.dat"),
    ("* filter=mark\n", _name_the_filter_alone, ".gitattributes, .lfsconfig"),
    ("*.dat filter=other\n", _name_the_filter_in_place_of_another, "old.dat"),
])
def test_a_change_git_would_run_a_filter_of_its_settings_on_is_not_put_into_the_tree(
        tmp_path, attribute_text, write, name):
    """The attribute is one the working tree holds or one the change brings along,
    over one that names a driver the settings give no program too; the driver is not
    run, and the rest of the change is not put in either."""
    repo = _filter_repo(tmp_path, attribute_text)
    base = gitcmd.require_head(str(repo))
    commit = _commit_over(repo, write, base)
    mark = tmp_path / "ran"
    _name_a_filter(repo, mark)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes {name}, which the git settings of {repo} run a program on "
        f"(the filter mark), by the .gitattributes files there now or by the ones the "
        f"change itself writes, so nothing is put into the working tree.")
    assert not mark.exists()
    assert (repo / "old.dat").read_text(encoding="utf-8") == "old\n"
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"


@pytest.mark.parametrize("attribute_text, write, path_list", [
    ("*.dat filter=mark\n", lambda worktree: (worktree / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8"), ["kept.txt"]),
    ("*.dat -filter\n", _name_the_filter_alone, [".gitattributes", ".lfsconfig"]),
    ("*.dat filter=other\n", _rewrite_a_dat, ["kept.txt", "old.dat"]),
])
def test_a_change_no_filter_of_the_settings_is_run_on_is_put_into_the_tree(
        tmp_path, attribute_text, write, path_list):
    """A path the attribute does not cover, the attribute files themselves, and an
    attribute naming a driver the settings give no program. The driver runs on none
    of the paths the change writes."""
    repo = _filter_repo(tmp_path, attribute_text)
    base = gitcmd.require_head(str(repo))
    commit = _commit_over(repo, write, base)
    mark = tmp_path / "ran"
    _name_a_filter(repo, mark)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_why"] == "" and land_dict["land"]["paths"] == path_list
    mark_text = mark.read_text(encoding="utf-8") if mark.exists() else ""
    assert not any(f" {path}\n" in mark_text for path in path_list)


def test_a_filter_named_for_what_an_earlier_change_put_in_keeps_the_later_one_out(
        tmp_path):
    def name_the_filter(worktree):
        (worktree / ".gitattributes").write_text("*.dat filter=mark\n", encoding="utf-8")

    mark = tmp_path / "ran"
    repo, _, land_dict = _land_twice(
        tmp_path, _add_a_dat, name_the_filter,
        before_second=lambda repo: _name_a_filter(repo, mark))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes .gitattributes files that name a filter for new.dat, which "
        f"the git settings of {repo} run a program on (the filter mark)")
    assert not mark.exists() and not (repo / ".gitattributes").exists()


def _no_room(*args, **kwargs):
    raise OSError(36, "File name too long")


@pytest.mark.parametrize("name, stand_in, what", [
    (".gitignore", (sandbox, "_write_ignore_file"), "whether it ignores that"),
    (".gitignore", (tempfile, "TemporaryDirectory"), "whether it ignores that"),
    (".gitattributes", (tempfile, "TemporaryDirectory"),
     "which filter driver of its settings it runs on that"),
])
def test_rules_that_cannot_be_written_where_git_reads_them_keep_a_change_out(
        tmp_path, monkeypatch, name, stand_in, what):
    repo = _filter_repo(tmp_path, "*.dat -filter\n")
    base = gitcmd.require_head(str(repo))

    def write(worktree):
        (worktree / "sub").mkdir()
        (worktree / "sub" / name).write_text("*.tmp filter=mark\n", encoding="utf-8")
        (worktree / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    commit = _commit_over(repo, write, base)
    _name_a_filter(repo, tmp_path / "ran")
    monkeypatch.setattr(*stand_in, _no_room)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is True
    assert land_dict["land_why"].startswith(
        f"the change writes kept.txt, sub/{name}, and git did not say {what} in {repo}")
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"


def test_a_rule_file_larger_than_one_file_is_read_keeps_a_change_out(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))

    def write(clone):
        (clone / ".gitignore").write_text("*.tmp\n" * 40, encoding="utf-8")
        (clone / "new.py").write_text("n = 1\n", encoding="utf-8")

    commit = _commit_of(repo, write, base)
    monkeypatch.setattr(sandbox.tmpfile, "READ_MAX_BYTE", 100)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_refuse"] is True and land_dict["land_why"].startswith(
        f"the change writes .gitignore, new.py, and git did not say whether it ignores "
        f"that in {repo}")


def test_a_dependency_file_that_cannot_be_written_is_said_in_one_line(tmp_path):
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: (clone / "requirements.txt").write_text(
        "pkg\n", encoding="utf-8"), base)
    (tmp_path / "blocker").write_text("a file\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"the dependency file requirements\.txt of "
                                           r"\w+ could not be written under "):
        sandbox._write_dep_file(str(repo), base, commit, str(tmp_path / "blocker" / "dep"))


def _patch_of_zero_file(tmp_path, repo, byte_count):
    """Write the patch that adds one file of byte_count zero bytes to a repository."""
    clone = tmp_path / "zero-clone"
    _git(tmp_path, "clone", "--quiet", str(repo), str(clone))
    with open(clone / "big.bin", "wb") as handle:
        handle.truncate(byte_count)
    _git(clone, "add", "-A")
    patch = tmp_path / "zero.diff"
    patch.write_bytes(subprocess.run(
        ["git", "-C", str(clone), "diff", "--cached", "--binary"], capture_output=True,
        check=True).stdout)
    shutil.rmtree(clone)
    return patch


def test_a_patch_git_cannot_hold_in_its_address_space_is_not_taken_in_or_put_in(
        tmp_path, monkeypatch):
    """The patch is a fraction of the size of the file it writes. git stops by itself,
    the branch is not left behind, and the working tree is as it was."""
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    patch = _patch_of_zero_file(tmp_path, repo, 96 << 20)
    assert patch.stat().st_size < (1 << 20)
    full_limit = gitcmd.MAX_PATCH_ADDRESS_BYTE
    monkeypatch.setattr(gitcmd, "MAX_PATCH_ADDRESS_BYTE", 64 << 20)
    with pytest.raises(RuntimeError, match="git apply failed while taking in .*"
                                           "Out of memory"):
        sandbox.take_in_patch(str(repo), str(patch), base, "codetwine-loom/sandbox/one",
                              str(tmp_path / "wt"), "sandbox: a change")
    assert _git(repo, "branch", "--list", "codetwine-loom/*") == ""
    monkeypatch.setattr(gitcmd, "MAX_PATCH_ADDRESS_BYTE", full_limit)
    taken = sandbox.take_in_patch(str(repo), str(patch), base, "codetwine-loom/sandbox/one",
                                  str(tmp_path / "wt"), "sandbox: a change")
    monkeypatch.setattr(gitcmd, "MAX_PATCH_ADDRESS_BYTE", 64 << 20)
    land_dict = sandbox.try_land(str(repo), base, taken["commit"],
                                 str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is False
    assert "Out of memory" in land_dict["land_why"]
    assert _git(repo, "status", "--porcelain", "-uall") == ""
    assert sandbox.held_base(str(repo), base, taken["commit"],
                             str(tmp_path / "held" / "land.diff.held")) == ""


def test_settings_git_cannot_read_the_filters_of_keep_a_change_out(tmp_path, monkeypatch):
    repo = _filter_repo(tmp_path, "*.dat filter=mark\n")
    base = gitcmd.require_head(str(repo))
    commit = _commit_over(repo, _add_a_dat, base)

    def refuse(project_dir):
        raise RuntimeError("could not read the drivers of the git settings: bad config")

    monkeypatch.setattr(gitcmd, "filter_name_set", refuse)
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_refuse"] is True and land_dict["land_why"].startswith(
        f"the change writes new.dat, and git did not say which filter driver of its "
        f"settings it runs on that in {repo}")


def test_a_patch_is_taken_in_without_a_filter_of_the_settings_being_run(tmp_path):
    """The commit holds what the patch writes, on a new branch and on one that is
    there, and no file of it is checked out."""
    repo = _filter_repo(tmp_path, "*.dat filter=mark\n")
    base = gitcmd.require_head(str(repo))
    patch = tmp_path / "patch.diff"

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "--quiet", str(repo), str(clone))
    (clone / "new.dat").write_bytes(b"new\r\n")
    (clone / "old.dat").write_text("old\nmore\n", encoding="utf-8")
    _git(clone, "add", "-A")
    patch.write_bytes(subprocess.run(
        ["git", "-C", str(clone), "diff", "--cached", "--binary"], capture_output=True,
        check=True).stdout)
    mark = tmp_path / "ran"
    _name_a_filter(repo, mark)
    _git(repo, "config", "core.autocrlf", "true")
    _git(repo, "config", "merge.autoStash", "true")
    taken = sandbox.take_in_patch(str(repo), str(patch), base, "codetwine-loom/sandbox/one",
                                  str(tmp_path / "wt"), "sandbox: a change")
    again = sandbox.take_in_patch(str(repo), "", base, "codetwine-loom/sandbox/one",
                                  str(tmp_path / "wt2"), "stage: nothing", [base])
    assert not mark.exists()
    assert not (tmp_path / "wt").exists() and not (tmp_path / "wt2").exists()
    tree = _git(repo, "rev-parse", f"{again['commit']}^{{tree}}")
    assert tree == _git(repo, "rev-parse", f"{taken['commit']}^{{tree}}")
    assert subprocess.run(["git", "-C", str(repo), "cat-file", "blob",
                           f"{tree.strip()}:new.dat"], capture_output=True,
                          check=True).stdout == b"new\r\n"
    assert _git(repo, "cat-file", "blob", f"{tree.strip()}:old.dat") == "old\nmore\n"
    assert _git(repo, "log", "--format=%P", "-1", again["commit"]).split() == [
        taken["commit"]]
    assert _git(repo, "stash", "list") == ""


@pytest.mark.parametrize("parent_list, more_list", [
    (["side", "main", "side", "side~0"], ["side"]), (["other", "side"], ["other", "side"]),
    (["main"], [])])
def test_a_commit_taken_in_names_the_parents_that_stand_apart(tmp_path, parent_list,
                                                              more_list):
    """A parent the branch's tip reaches, and one named twice, is left out."""
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    for branch in ("side", "other"):
        _git(repo, "checkout", "--quiet", "-b", branch, "main")
        (repo / f"{branch}.txt").write_text("x\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", branch)
    _git(repo, "checkout", "--quiet", "main")
    taken = sandbox.take_in_patch(str(repo), "", base, "codetwine-loom/sandbox/one",
                                  str(tmp_path / "wt"), "\n\n integrate: x \n\n\nbody\n\n",
                                  parent_list)
    want_list = [base, *(_git(repo, "rev-parse", name).strip() for name in more_list)]
    assert _git(repo, "log", "--format=%P", "-1", taken["commit"]).split() == want_list
    assert _git(repo, "log", "--format=%B", "-1", taken["commit"]) \
        == " integrate: x\n\nbody\n\n"
    assert _git(repo, "rev-parse", f"{taken['commit']}^{{tree}}") \
        == _git(repo, "rev-parse", f"{base}^{{tree}}")


# Names a shell splits, expands or runs when they are written into a command as they are
_SHELL_NAME_TUPLE = ("a repo; $(touch MARK)", "two  blanks\tand a tab", "it's \"quoted\"",
                     "back`touch MARK`tick", "a line\nbreak", "&& touch MARK #")


@pytest.mark.parametrize("repo_name", _SHELL_NAME_TUPLE)
@pytest.mark.parametrize("out_name", _SHELL_NAME_TUPLE[:3])
def test_the_command_a_change_kept_out_is_applied_with_runs_as_it_is_printed(
        tmp_path, repo_name, out_name):
    """The repository and the diff are at paths a shell splits, expands or runs."""
    repo = _repo(tmp_path / repo_name)
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: (clone / ".env").write_text(
        "DEF_K=7\n", encoding="utf-8"), base)
    patch_path = tmp_path / "out" / out_name / "land.diff"
    land_dict = sandbox.try_land(str(repo), base, commit, str(patch_path),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_refuse"] is True
    command = land_dict["land_command"]
    assert land_dict["land_why"].endswith("apply it by hand with:")
    assert shlex.split(command) == ["git", "-C", str(repo), "apply", "--binary",
                                    "--whitespace=nowarn", "--no-ignore-whitespace",
        str(patch_path)]
    assert sandbox.land_why_text(land_dict) == f"{land_dict['land_why']} {command}"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    proc = subprocess.run(["sh", "-c", command], cwd=work_dir, capture_output=True,
                          text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert (repo / ".env").read_text(encoding="utf-8") == "DEF_K=7\n"
    assert os.listdir(work_dir) == []
    # A change that no longer applies names the same command
    _git(repo, "add", "-f", ".env")
    land_dict = sandbox.try_land(str(repo), base, commit, str(patch_path),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land_refuse"] is False
    assert land_dict["land_why"].endswith("apply it by hand with:")
    assert land_dict["land_command"] == command


def test_a_change_that_does_not_apply_is_not_one_refused_for_its_paths(tmp_path):
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: (clone / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8"), base)
    (repo / "kept.txt").write_text("another\n", encoding="utf-8")
    land_dict = sandbox.try_land(str(repo), base, commit, str(tmp_path / "land.diff"),
                                 str(tmp_path / "land-last.diff"))
    assert land_dict["land"] is None and land_dict["land_refuse"] is False
    assert "does not apply to the working tree" in land_dict["land_why"]


def test_a_file_a_run_moved_is_counted_at_the_path_it_left_and_the_one_it_went_to(
        tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "sub").mkdir()
    (repo / "sub" / "moved.py").write_text("x = 1\n" * 20, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "a file to move")
    base = gitcmd.require_head(str(repo))
    commit = _commit_of(repo, lambda clone: os.rename(clone / "sub" / "moved.py",
                                                      clone / "sub" / "renamed.py"), base)
    _git(repo, "config", "diff.renames", "copies")
    land_dict = sandbox.land_in_tree(str(repo), base, commit, str(tmp_path / "land.diff"),
                                     str(tmp_path / "land-last.diff"))
    assert (land_dict["files"], land_dict["paths"]) == (2, ["sub/moved.py",
                                                            "sub/renamed.py"])
    assert not (repo / "sub" / "moved.py").exists()
    assert (repo / "sub" / "renamed.py").is_file()


def test_the_run_lands_in_the_tree_only_when_it_is_asked_to(tmp_path, monkeypatch):
    """A task of a dispatching run lands on its branch and nowhere else: the union
    is what goes into the tree."""
    repo = _repo(tmp_path / "repo")
    result, _ = _run_without_a_container(tmp_path, monkeypatch, project=str(repo),
                                         output_dir=str(tmp_path / "out"), request="x")
    assert result["land"] is None and result["land_why"] == ""


def _landed_run(tmp_path, monkeypatch, on_repo=None, exit_status=0, **kwargs):
    """Take one run's patch in with land and return the repository and the branches.

    Args:
        on_repo: Called with the repository's path before the run, for a test that
            leaves something in it.
        exit_status: What the container's summary says the run ended with.
        kwargs: Passed on to run().
    """
    repo = _repo(tmp_path / "repo")
    if on_repo:
        on_repo(repo)
    out = tmp_path / "out"
    run_id = "20260101-000000-1-42"
    monkeypatch.setattr(sandbox, "new_run_id", lambda: run_id)
    mine = pathlib.Path(sandbox.agent_dir(str(out), sandbox.agent_name("", run_id)))

    def wrote(argv):
        patch = _patch_of(mine / repo.name, run_id)
        patch.write_text(
            "diff --git a/kept.txt b/kept.txt\n--- a/kept.txt\n+++ b/kept.txt\n"
            "@@ -1 +1,2 @@\n one\n+two\n", encoding="utf-8")
        patch.with_suffix(f"{patch.suffix}.meta.json").write_text(
            json.dumps({"exit_status": exit_status, "files": 1}), encoding="utf-8")

    result, _ = _run_without_a_container(
        tmp_path, monkeypatch, on_start=wrote, project=str(repo), output_dir=str(out),
        request="x", land=True, **kwargs)
    return repo, result, _git(repo, "for-each-ref", "--format=%(refname:short)",
                              "refs/heads").split()


@pytest.mark.parametrize("exit_status, land", [(137, True), (1, True), (0, True),
                                               (137, False)])
def test_a_run_whose_patch_holds_nothing_says_why_by_its_exit_status(
        tmp_path, monkeypatch, exit_status, land):
    """A run that changed nothing is put in as nothing; one that did not get through
    says how it ended where the sandbox subcommand prints and records it."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    run_id = "20260101-000000-1-42"
    monkeypatch.setattr(sandbox, "new_run_id", lambda: run_id)
    mine = pathlib.Path(sandbox.agent_dir(str(out), sandbox.agent_name("", run_id)))

    def wrote(argv):
        patch = _patch_of(mine / repo.name, run_id)
        patch.write_text("", encoding="utf-8")
        patch.with_suffix(f"{patch.suffix}.meta.json").write_text(
            json.dumps({"exit_status": exit_status, "files": 0}), encoding="utf-8")

    result, _ = _run_without_a_container(
        tmp_path, monkeypatch, on_start=wrote, project=str(repo), output_dir=str(out),
        request="x", land=land)
    assert result["patch"] is None and result["commit"] is None and not result["no_patch"]
    assert result["land_why"] == (
        f"the run {sandbox._empty_patch_text(exit_status)}, so nothing was put into the "
        f"working tree" if exit_status and land else "")


def test_a_run_that_got_through_is_put_into_the_tree_and_its_name_goes(tmp_path,
                                                                        monkeypatch):
    """The tree holds the change and the commit is the user's, so the name this tool
    made up would be a second copy of what is already on disk."""
    repo, result, branches = _landed_run(tmp_path, monkeypatch)
    assert result["land"] == {"files": 1, "paths": ["kept.txt"],
                              "patch": result["land"]["patch"],
                              "last": str(tmp_path / "out" / "agents" / ".repo" / "repo"
                                          / "land-last.diff"),
                              "no_status": []}
    assert result["land"]["patch"].endswith("land-20260101-000000-1-42.diff")
    with open(result["land"]["patch"], "rb") as fp, \
            open(result["land"]["last"], "rb") as last_fp:
        assert fp.read() == last_fp.read()
    assert result["land_why"] == "" and result["commit"]
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\ntwo\n"
    assert branches == ["main"]
    assert gitcmd.require_head(str(repo)) == result["base"]


def test_a_run_that_did_not_get_through_stays_on_its_branch(tmp_path, monkeypatch,
                                                            caplog):
    """What it changed is however far it got, which is for reading, not for the tree."""
    repo, result, branches = _landed_run(tmp_path, monkeypatch, exit_status=1)
    assert result["land"] is None
    assert "exit status 1" in result["land_why"] and "codetwine-loom/sandbox/" in result["land_why"]
    assert result["land_why"].endswith("Read it with:")
    assert result["land_command"] == (
        f"git -C {repo} diff --no-textconv --no-ext-diff "
        f"{gitcmd.short(result['base'])} {result['branch']}")
    word_list = shlex.split(result["land_command"])
    assert word_list[:3] == ["git", "-C", str(repo)]
    assert "+two" in _git(repo, *word_list[3:])
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"
    assert sorted(branches) == sorted(["main", result["branch"]])
    assert sandbox.land_why_text(result) in caplog.text


def test_the_command_to_read_a_branch_starts_no_program_of_the_git_settings(tmp_path,
                                                                           monkeypatch):
    """A text conversion and an external diff program the repository's settings name
    for the changed file do not run when the user types the command given."""
    repo, result, _branch_list = _landed_run(tmp_path, monkeypatch, exit_status=1)
    mark = tmp_path / "mark"
    program = tmp_path / "program.sh"
    program.write_text(f"#!/bin/sh\necho ran >> {shlex.quote(str(mark))}\ncat \"$1\"\n",
                       encoding="utf-8")
    program.chmod(0o755)
    (repo / ".git" / "info" / "attributes").write_text("*.txt diff=mark\n",
                                                       encoding="utf-8")
    _git(repo, "config", "diff.mark.textconv", str(program))
    _git(repo, "config", "diff.external", str(program))
    word_list = shlex.split(result["land_command"])
    assert word_list[:3] == ["git", "-C", str(repo)]
    assert "+two" in _git(repo, *word_list[3:])
    assert not mark.exists()
    _git(repo, "diff", *word_list[-2:])
    assert mark.exists()


def test_a_run_over_a_changed_tree_lands_on_top_of_the_changes(tmp_path, monkeypatch):
    """The run starts from the tree as it is, so what it changed applies over what
    was there, committed or not, and what was there stays."""
    def leave_changes(repo):
        (repo / "draft.txt").write_text("draft\n", encoding="utf-8")

    repo, result, branches = _landed_run(tmp_path, monkeypatch, on_repo=leave_changes)
    assert result["base"] != gitcmd.require_head(str(repo))
    assert result["land"]["files"] == 1
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\ntwo\n"
    assert (repo / "draft.txt").read_text(encoding="utf-8") == "draft\n"
    assert branches == ["main"]
    assert sorted(gitcmd.status_path_list(str(repo))) == ["draft.txt", "kept.txt"]


def test_a_change_the_tree_stopped_taking_during_the_run_is_reported_not_forced(
        tmp_path, monkeypatch, caplog):
    def edit_meanwhile(argv):
        (tmp_path / "repo" / "kept.txt").write_text("by hand\n", encoding="utf-8")

    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    run_id = "20260101-000000-1-42"
    monkeypatch.setattr(sandbox, "new_run_id", lambda: run_id)
    mine = pathlib.Path(sandbox.agent_dir(str(out), sandbox.agent_name("", run_id)))

    def wrote(argv):
        patch = _patch_of(mine / repo.name, run_id)
        patch.write_text(
            "diff --git a/kept.txt b/kept.txt\n--- a/kept.txt\n+++ b/kept.txt\n"
            "@@ -1 +1,2 @@\n one\n+two\n", encoding="utf-8")
        patch.with_suffix(f"{patch.suffix}.meta.json").write_text(
            json.dumps({"exit_status": 0, "files": 1}), encoding="utf-8")
        edit_meanwhile(argv)

    result, _ = _run_without_a_container(
        tmp_path, monkeypatch, on_start=wrote, project=str(repo), output_dir=str(out),
        request="x", land=True)
    assert result["land"] is None and "does not apply" in result["land_why"]
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "by hand\n"
    # The branch stays, as the only place the change is
    assert result["branch"] in _git(repo, "branch", "--list", "codetwine-loom/sandbox/*")


# == the analysis every agent copies =========================
def _analysis_in(out, name="repo"):
    """Put an analysis where build writes it."""
    write_knowledge_db({"files": []}, str(out / name / knowledge.KNOWLEDGE_FILE), name)


def _analysis_call_list(monkeypatch, fail=None) -> list:
    """Replace what makes the analysis, and return what it is called with.

    Args:
        fail: An exception to raise instead, or None.
    """
    call_list = []

    def fake(*args, with_doc=False):
        call_list.append((*args, with_doc))
        if fail is not None:
            raise fail
        return "k.sqlite"

    monkeypatch.setattr(sandbox.build, "analyse_and_index", fake)
    return call_list


def test_a_run_leaves_the_analysis_as_it_is(tmp_path, monkeypatch):
    """The tree the run changed is analysed by the next run that reads it."""
    _analysis_in(tmp_path / "out")
    call_list = _analysis_call_list(monkeypatch)
    repo = _repo(tmp_path / "repo")
    _run_without_a_container(tmp_path, monkeypatch, project=str(repo),
                             output_dir=str(tmp_path / "out"), request="x")
    assert call_list == []


# == the first analysis, before the agents start ==============
def test_nothing_built_is_analysed_here_before_the_agent_copies(tmp_path, monkeypatch):
    """One analysis on the host is what every agent copies, instead of each container
    analysing the repository for itself."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    call_list = _analysis_call_list(monkeypatch)
    monkeypatch.setattr(sandbox, "analyse_if_none", _analyse_if_none)
    sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-a", str(repo), ["vendor"], True)
    assert call_list == [(str(repo), str(out), ["vendor"], True)]


def test_what_is_built_already_is_not_analysed_again_at_the_start(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    _analysis_in(tmp_path / "out")
    call_list = _analysis_call_list(monkeypatch)
    monkeypatch.setattr(sandbox, "analyse_if_none", _analyse_if_none)
    sandbox.make_agent_dir(str(tmp_path / "out"), "repo", "codetwine-loom-a", str(repo),
                           None, False)
    assert call_list == []


def _one_answer(monkeypatch, answer: str) -> list:
    """A terminal whose one answer is this; a second question fails the test."""
    given = iter([answer])
    monkeypatch.setattr(progress, "at_terminal", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: next(given))


def _never_asked(monkeypatch, tty: bool) -> None:
    """A terminal, or none, on which no question may be asked."""
    monkeypatch.setattr(progress, "at_terminal", lambda: tty)

    def refuse(_prompt):
        raise AssertionError("the question was asked")
    monkeypatch.setattr("builtins.input", refuse)


@pytest.mark.parametrize("answer, expected", [("y", True), ("", False)])
def test_dispatch_asks_about_the_documents_before_the_first_analysis(
        tmp_path, monkeypatch, answer, expected):
    """Without --doc-create, and with a terminal, the one question is the documents."""
    repo = _repo(tmp_path / "repo")
    call_list = _analysis_call_list(monkeypatch)
    _one_answer(monkeypatch, answer)
    assert _analyse_if_none(str(repo), str(tmp_path / "out"), "repo", None,
                                   False, ask_doc=True) is True
    assert call_list == [(str(repo), str(tmp_path / "out"), None, expected)]


def test_the_documents_asked_for_on_the_command_line_are_not_asked_again(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    call_list = _analysis_call_list(monkeypatch)
    _never_asked(monkeypatch, tty=True)
    _analyse_if_none(str(repo), str(tmp_path / "out"), "repo", None, True,
                            ask_doc=True)
    assert call_list == [(str(repo), str(tmp_path / "out"), None, True)]


@pytest.mark.parametrize("with_doc, line_count", [(True, 2), (False, 0)])
def test_a_first_analysis_with_the_documents_is_a_step_on_the_terminal(
        tmp_path, monkeypatch, caplog, with_doc, line_count):
    caplog.set_level(logging.INFO, logger="codetwine_loom.progress")
    repo = _repo(tmp_path / "repo")
    _analysis_call_list(monkeypatch)
    _analyse_if_none(str(repo), str(tmp_path / "out"), "repo", None, with_doc)
    assert len([r for r in caplog.records
                if r.getMessage().startswith("make the design documents:")]) == line_count
    # That the repository is analysed here is said where the terminal shows it
    assert [r.name for r in caplog.records
            if "the repository is analysed here first" in r.getMessage()] == \
        ["codetwine_loom.progress"]


def test_no_terminal_means_the_first_analysis_has_no_documents(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    call_list = _analysis_call_list(monkeypatch)
    _never_asked(monkeypatch, tty=False)
    _analyse_if_none(str(repo), str(tmp_path / "out"), "repo", None, False,
                            ask_doc=True)
    assert call_list == [(str(repo), str(tmp_path / "out"), None, False)]


def test_nothing_is_asked_when_the_analysis_is_there(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    _analysis_in(tmp_path / "out")
    _never_asked(monkeypatch, tty=True)
    assert _analyse_if_none(str(repo), str(tmp_path / "out"), "repo", None,
                                   False, ask_doc=True) is False


def test_the_other_runs_do_not_ask(tmp_path, monkeypatch):
    """sandbox and integrate take --doc-create or nothing; only dispatch asks."""
    repo = _repo(tmp_path / "repo")
    call_list = _analysis_call_list(monkeypatch)
    _never_asked(monkeypatch, tty=True)
    monkeypatch.setattr(sandbox, "analyse_if_none", _analyse_if_none)
    sandbox.make_agent_dir(str(tmp_path / "out"), "repo", "codetwine-loom-a", str(repo),
                           None, False)
    assert call_list == [(str(repo), str(tmp_path / "out"), None, False)]


def test_a_tree_with_changes_is_analysed_as_it_is(tmp_path, monkeypatch):
    """The record says which tree the analysis describes, and the run starts from a
    commit of that tree, so the agents read it without analysing again."""
    repo = _repo(tmp_path / "repo")
    (repo / "draft.txt").write_text("x", encoding="utf-8")
    call_list = _analysis_call_list(monkeypatch)
    monkeypatch.setattr(sandbox, "analyse_if_none", _analyse_if_none)
    sandbox.make_agent_dir(str(tmp_path / "out"), "repo", "codetwine-loom-a", str(repo),
                           None, False)
    assert call_list == [(str(repo), str(tmp_path / "out"), None, False)]


def test_a_missing_summary_is_not_an_error(tmp_path):
    assert sandbox._read_meta(str(tmp_path / "patch.diff"), str(tmp_path)) == {}


def test_a_summary_that_is_not_json_is_not_an_error(tmp_path):
    (tmp_path / "patch.diff.meta.json").write_text("{", encoding="utf-8")
    assert sandbox._read_meta(str(tmp_path / "patch.diff"), str(tmp_path)) == {}


def test_a_file_a_container_wrote_is_read_up_to_the_size_one_is_written_in(tmp_path,
                                                                           caplog):
    """The summary beside a patch, and what a run wrote for this machine, of more
    bytes than that are not opened as JSON, and one warning names each with its size
    and the size read."""
    record_text = '{"exit_status": 0}'
    for name in ("patch.diff.meta.json", "outcome.json", "tasks.json"):
        (tmp_path / name).write_text(
            record_text + " " * (sandbox._MAX_ACROSS_BYTE - len(record_text)),
            encoding="utf-8")
    assert sandbox._read_meta(str(tmp_path / "patch.diff"), str(tmp_path)) == {
        "exit_status": 0}
    assert sandbox._read_across(str(tmp_path / "tasks.json"), str(tmp_path), "the run",
                                "split") == {"exit_status": 0}
    for name in ("patch.diff.meta.json", "outcome.json", "tasks.json"):
        with open(tmp_path / name, "a", encoding="utf-8") as handle:
            handle.write(" ")
    with caplog.at_level(logging.WARNING):
        assert sandbox._read_meta(str(tmp_path / "patch.diff"), str(tmp_path)) == {}
        assert sandbox._read_outcome(str(tmp_path / "outcome.json"), str(tmp_path),
                                     sandbox._MERGE_KIND) == {}
    for name in ("patch.diff.meta.json", "outcome.json"):
        assert [record.levelname for record in caplog.records
                if f"{tmp_path / name} is {sandbox._MAX_ACROSS_BYTE + 1} bytes, over the "
                f"{sandbox._MAX_ACROSS_BYTE} read of one file" in record.getMessage()
                ] == ["WARNING"]
    with pytest.raises(RuntimeError, match="the run left no split in .* over the "
                                           f"{sandbox._MAX_ACROSS_BYTE} read of one file"):
        sandbox._read_across(str(tmp_path / "tasks.json"), str(tmp_path), "the run",
                             "split")


def test_a_record_there_is_no_memory_to_read_counts_as_not_describing(
        tmp_path, monkeypatch):
    """The record beside the index of an agent's copy is one a container writes."""
    repo = _repo(tmp_path / "repo")
    (repo / "draft.txt").write_text("x\n", encoding="utf-8")
    out = _index_at(tmp_path, monkeypatch, repo, state=knowledge.analysis_state(str(repo), []))
    snapshot = gitcmd.snapshot_commit(str(repo))
    assert sandbox.index_describe(str(repo), out, "repo", snapshot) is True

    def no_memory(text):
        raise MemoryError

    monkeypatch.setattr(sandbox.tmpfile, "json", types.SimpleNamespace(loads=no_memory))
    assert sandbox.index_describe(str(repo), out, "repo", snapshot) is False


# == What the commit says about the run =======================

def _attempt(n, verdict="unverified", why="nothing here checks that it is done",
             did="changed one file"):
    """Return one history entry, with the keys the trailers are taken from."""
    return {"n": n, "at": "2026-09-01T00:00:00", "verdict": verdict, "why": why,
            "did": did}


def _write_history(project_out, attempt_list):
    """Write one history file under the output directory of a project."""
    project_out.mkdir(parents=True, exist_ok=True)
    (project_out / "edit_history.json").write_text(
        json.dumps({"schema": history.SCHEMA, "project": project_out.name,
                    "attempts": attempt_list}), encoding="utf-8")


def _committed(tmp_path, monkeypatch, request, attempt_list, before=None):
    """Take one run's patch in and return the message it was committed with.

    Args:
        tmp_path: The test's directory.
        monkeypatch: pytest's own fixture.
        request: The change request run() is given.
        attempt_list: The history the container leaves behind, or None to leave none.
        before: The history that is already there when the run starts, or None.

    Returns:
        (the repository, the commit the patch landed in, the message it carries).
    """
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    project_out = out / repo.name
    if before is not None:
        _write_history(project_out, before)

    def write(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    patch_text = _patch_from(repo, write)

    def leave_behind(argv):
        patch = _told_to_write(argv)
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(patch_text, encoding="utf-8")
        patch.with_suffix(f"{patch.suffix}.meta.json").write_text(
            json.dumps({"base": gitcmd.require_head(str(repo)), "exit_status": 0,
                        "files": 1}), encoding="utf-8")
        if attempt_list is not None:
            _write_history(_mounted_at(argv, "/out") / repo.name, attempt_list)

    result, _ = _run_without_a_container(
        tmp_path, monkeypatch, on_start=leave_behind, project=str(repo),
        output_dir=str(out), request=request)
    assert result["commit"], "the patch was not taken in"
    return repo, result["commit"], _git(repo, "log", "-1", "--format=%B",
                                        result["commit"])


def _trailer(repo, commit, name):
    """Return one trailer of a commit, as git itself reads it back."""
    return _git(repo, "log", "-1", f"--format=%(trailers:key={name},valueonly)",
                commit).strip()


def test_the_commit_carries_what_the_run_recorded_about_itself(tmp_path, monkeypatch):
    _, _, message = _committed(
        tmp_path, monkeypatch, "make it stop crashing",
        [_attempt(1, verdict="failed", why="no edit landed at all")])
    assert "Codetwine-Loom-Attempt: 1" in message
    assert "Codetwine-Loom-Verdict: failed" in message
    assert "Codetwine-Loom-Why: no edit landed at all" in message
    assert "Codetwine-Loom-Exit-Status: 0" in message


def test_the_commit_carries_what_the_run_said_it_did(tmp_path, monkeypatch):
    """A run combining several branches reads this off the commit; it has no other way
    to reach it, the record it is written from being in that run's own output."""
    repo, commit, message = _committed(
        tmp_path, monkeypatch, "rename the helper",
        [_attempt(1, did="renamed helper() to parse_line() and fixed both callers")])
    assert ("Codetwine-Loom-Did: renamed helper() to parse_line() and fixed both callers"
            in message)
    assert (_trailer(repo, commit, integrate_mod.DID_TRAILER)
            == "renamed helper() to parse_line() and fixed both callers")


def test_a_run_that_said_nothing_carries_no_account_of_what_it_did(tmp_path,
                                                                   monkeypatch):
    _, _, message = _committed(tmp_path, monkeypatch, "x", [_attempt(1, did="")])
    assert "Codetwine-Loom-Did" not in message


def test_an_account_of_several_lines_is_written_as_one_trailer(tmp_path, monkeypatch):
    """git reads the last paragraph as trailers only while every line of it is one."""
    repo, commit, _ = _committed(
        tmp_path, monkeypatch, "x",
        [_attempt(1, did="renamed the helper\nand fixed   the callers")])
    assert (_trailer(repo, commit, integrate_mod.DID_TRAILER)
            == "renamed the helper and fixed the callers")
    assert _trailer(repo, commit, "Codetwine-Loom-Exit-Status") == "0"


def test_git_reads_the_trailers_back_as_trailers(tmp_path, monkeypatch):
    """The last paragraph is what git parses, so the format is checked with git."""
    repo, commit, _ = _committed(tmp_path, monkeypatch, "make it stop crashing",
                                 [_attempt(3, verdict="unverified")])
    assert _trailer(repo, commit, "Codetwine-Loom-Verdict") == "unverified"
    assert _trailer(repo, commit, "Codetwine-Loom-Attempt") == "3"


def test_a_request_of_several_lines_still_leaves_one_subject_line(tmp_path,
                                                                  monkeypatch):
    request = "make it stop crashing\n\nthe second paragraph of the request\n"
    repo, commit, message = _committed(tmp_path, monkeypatch, request, [_attempt(1)])
    subject, blank = message.splitlines()[0], message.splitlines()[1]
    assert subject == ("sandbox: make it stop crashing the second paragraph of the "
                       "request")
    assert blank == ""
    assert "the second paragraph of the request" in message
    assert _trailer(repo, commit, "Codetwine-Loom-Verdict") == "unverified"


def test_a_request_longer_than_a_subject_line_is_cut_there(tmp_path, monkeypatch):
    request = "make " * 30
    _, _, message = _committed(tmp_path, monkeypatch, request, [_attempt(1)])
    subject = message.splitlines()[0]
    assert subject == "sandbox: " + ("make " * 30).strip()[:sandbox._MAX_SUBJECT_CHAR]
    assert len(subject.splitlines()) == 1


def test_an_entry_an_earlier_run_wrote_is_not_what_this_commit_says(tmp_path,
                                                                    monkeypatch):
    """A run that records nothing must not be described by the run before it."""
    _, _, message = _committed(tmp_path, monkeypatch, "x", attempt_list=None,
                               before=[_attempt(4, verdict="failed")])
    assert "Codetwine-Loom-Verdict" not in message and "Codetwine-Loom-Attempt" not in message
    assert "Codetwine-Loom-Exit-Status: 0" in message


def test_a_run_that_recorded_nothing_still_says_how_it_ended(tmp_path, monkeypatch):
    _, _, message = _committed(tmp_path, monkeypatch, "x", attempt_list=None)
    assert "Codetwine-Loom-Verdict" not in message
    assert "Codetwine-Loom-Exit-Status: 0" in message


def test_a_write_made_without_a_tool_is_named_in_the_commit(tmp_path, monkeypatch):
    """The patch is what git holds, so what it carries is readable from the branch."""
    attempt = dict(_attempt(1), untooled=["by_hand.py", "sub/other.py"])
    _, _, message = _committed(tmp_path, monkeypatch, "x", [attempt])
    assert "Codetwine-Loom-Untooled: by_hand.py sub/other.py" in message


def test_a_run_that_went_through_the_tools_carries_no_such_trailer(tmp_path,
                                                                   monkeypatch):
    _, _, message = _committed(tmp_path, monkeypatch, "x", [_attempt(1)])
    assert "Codetwine-Loom-Untooled" not in message


def test_a_value_of_several_lines_is_flattened_into_one_trailer():
    message = sandbox.commit_message("x", {"Codetwine-Loom-Why": "one\ntwo   three"})
    assert message.endswith("\nCodetwine-Loom-Why: one two three")


def test_a_trailer_with_nothing_to_say_is_left_out():
    message = sandbox.commit_message("x", {"Codetwine-Loom-Verdict": None, "Codetwine-Loom-Why": "",
                                           "Codetwine-Loom-Exit-Status": 0})
    assert message.endswith("\nCodetwine-Loom-Exit-Status: 0")
    assert "Codetwine-Loom-Verdict" not in message and "Codetwine-Loom-Why" not in message


# == The scratch directory the clone goes in ==================

def _linked_and_submodule(tmp_path):
    """Make a repository with a linked worktree and a submodule, and return the three.

    Returns:
        {"main": the repository, "linked": its linked worktree, "submodule": the
        submodule's working tree inside it}.
    """
    main = _repo(tmp_path / "main")
    _git(main, "worktree", "add", "--quiet", "-b", "side", str(tmp_path / "linked"))
    sub = _repo(tmp_path / "sub")
    _git(main, "-c", "protocol.file.allow=always", "submodule", "add", "--quiet",
         str(sub), "sm")
    _git(main, "commit", "--quiet", "-m", "submodule")
    return {"main": main, "linked": tmp_path / "linked", "submodule": main / "sm"}


@pytest.mark.parametrize("kind", ["main", "linked", "submodule"])
def test_the_copy_of_the_history_clones_where_no_path_of_this_machine_is_there(
        tmp_path, kind):
    """A linked worktree's and a submodule's .git name a directory by a path on this
    machine, which the container does not have."""
    project = _linked_and_submodule(tmp_path)[kind]
    head = _git(project, "rev-parse", "HEAD").strip()
    scratch = sandbox.make_scratch(str(tmp_path / "scratch"))
    source = sandbox.make_clone_source(str(project), scratch)
    assert source == os.path.join(scratch, "src")
    # The container's view: none of the repositories this machine has is there
    hide = tmp_path / "hidden"
    hide.mkdir()
    for name in ("main", "linked", "sub"):
        (tmp_path / name).rename(hide / name)
    work = tmp_path / "work"
    proc = subprocess.run(["git", "clone", "--no-hardlinks", "--quiet", source, str(work)],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    _git(work, "cat-file", "-e", f"{head}^{{commit}}")


def test_the_copy_of_the_history_holds_no_ignored_file_config_or_hook(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / ".gitignore").write_text(".env\n", encoding="utf-8")
    (repo / ".env").write_text("LLM_API_KEY=secret-value\n", encoding="utf-8")
    _git(repo, "config", "remote.origin.url", "https://user:secret-token@example.invalid/r")
    (repo / ".git" / "hooks" / "post-checkout").write_text("#!/bin/sh\n", encoding="utf-8")
    scratch = sandbox.make_scratch(str(tmp_path / "scratch"))
    source = pathlib.Path(sandbox.make_clone_source(str(repo), scratch))
    assert _git(source, "rev-parse", "--is-bare-repository").strip() == "true"
    name_list = [p.name for p in source.rglob("*")]
    assert ".env" not in name_list and "kept.txt" not in name_list
    assert "post-checkout" not in name_list
    assert "secret-token" not in (source / "config").read_text(encoding="utf-8")


def test_the_copy_of_the_history_borrows_no_object_from_another_repository(tmp_path):
    """A clone made with --shared names the repository it reads objects from by a path
    on this machine."""
    repo = _repo(tmp_path / "repo")
    shared = tmp_path / "shared"
    _git(tmp_path, "clone", "--quiet", "--shared", str(repo), str(shared))
    scratch = sandbox.make_scratch(str(tmp_path / "scratch"))
    source = pathlib.Path(sandbox.make_clone_source(str(shared), scratch))
    assert not (source / "objects" / "info" / "alternates").exists()
    head = _git(shared, "rev-parse", "HEAD").strip()
    repo.rename(tmp_path / "hidden")
    _git(source, "cat-file", "-e", f"{head}^{{commit}}")


def test_a_run_over_a_linked_worktree_mounts_a_copy_the_container_can_clone(
        tmp_path, monkeypatch):
    project = _linked_and_submodule(tmp_path)["linked"]
    (project / "new.txt").write_text("not committed\n", encoding="utf-8")
    seen = {}

    def look(argv):
        source = _mounted_at(argv, "/src")
        base = next(a.split("=", 1)[1] for a in argv if a.startswith("CTLOOM_BASE="))
        work = tmp_path / "work"
        subprocess.run(["git", "clone", "--no-hardlinks", "--quiet", str(source),
                        str(work)], check=True, capture_output=True)
        seen["source"] = source
        seen["new"] = _git(work, "show", f"{base}:new.txt")

    _run_without_a_container(tmp_path, monkeypatch, on_start=look,
                             project=str(project), output_dir=str(tmp_path / "out"),
                             request="x", dry_run=True)
    assert seen["source"].name == "src" and seen["source"] != project
    assert seen["new"] == "not committed\n"


def _scratch_of(out, repo_name):
    """Return the clone of the one run going in an output directory.

    The clone is named after the run and sits on this machine's side of the agent
    (sandbox.host_dir), so a test that started one run finds it rather than builds it.

    Args:
        out: The output directory run() was given.
        repo_name: The repository's directory name.

    Returns:
        The path, or None when no run is holding one.
    """
    found = [p for p in sorted(pathlib.Path(out).glob(
        f"agents/.host/*/{repo_name}/sandbox/scratch-*"))
        if not p.name.startswith("scratch-warm-")]
    assert len(found) < 2, f"more than one run is holding a clone: {found}"
    return found[0] if found else None


def test_the_clone_has_somewhere_to_go_when_the_container_starts(tmp_path, monkeypatch):
    """The engine would make a missing mount root's, so the run makes it first."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    seen = {}

    def look(argv):
        scratch = _scratch_of(out, repo.name)
        seen["work"] = (scratch / "work").is_dir()
        seen["tmp"] = (scratch / "tmp").is_dir()
        seen["mounts"] = [a for a in argv if str(scratch) in a]
        seen["scratch"] = scratch

    _run_without_a_container(tmp_path, monkeypatch, on_start=look, project=str(repo),
                             output_dir=str(out), request="x")
    assert seen["work"] and seen["tmp"]
    scratch = seen["scratch"]
    assert seen["mounts"] == [f"{scratch}/src:/src:ro", f"{scratch}/work:/work",
                              f"{scratch}/tmp:/tmp"]


def test_the_clone_is_taken_away_when_the_run_ends(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    _run_without_a_container(tmp_path, monkeypatch, project=str(repo),
                             output_dir=str(out), request="x")
    assert _scratch_of(out, repo.name) is None


def test_a_clone_an_earlier_run_left_is_not_what_this_one_starts_on(tmp_path,
                                                                    monkeypatch):
    """A host that was killed mid-run leaves one behind; it is not carried over.

    The agent is named after its branch, so what an earlier round of the same agent
    left is in the same place as this round's own clone.
    """
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    mine = sandbox.host_dir(str(out), "sandbox-one")
    left = (pathlib.Path(mine) / repo.name / "sandbox"
            / f"scratch-{_a_run_that_ended()}" / "work" / "repo")
    left.mkdir(parents=True)
    (left / "half-written.py").write_text("x", encoding="utf-8")
    seen = {}
    _run_without_a_container(
        tmp_path, monkeypatch,
        on_start=lambda argv: seen.update(
            mine=(_scratch_of(out, repo.name) / "work").is_dir(),
            theirs=(left / "half-written.py").exists()),
        project=str(repo), output_dir=str(out), request="x", branch="codetwine-loom/sandbox/one")
    assert seen["mine"] is True and seen["theirs"] is False


def test_a_clone_that_cannot_be_taken_away_is_reported_and_not_raised(tmp_path, caplog,
                                                                     monkeypatch):
    """An engine that was not told who to run as writes as root."""
    def refuse(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(sandbox.shutil, "rmtree", refuse)
    sandbox.remove_scratch(str(tmp_path / "scratch"))
    assert "could not be taken away" in caplog.text


def test_the_container_failing_still_takes_the_clone_away(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    monkeypatch.setattr(sandbox, "pick_engine", lambda name="": "/usr/bin/true")
    real_run = subprocess.run

    def blow_up(argv, **kw):
        if _is_git(argv):
            return real_run(argv, **kw)
        raise OSError("the engine is not there")

    monkeypatch.setattr(sandbox.subprocess, "run", blow_up)
    with pytest.raises(OSError):
        sandbox.run(str(repo), str(out), "x")
    assert _scratch_of(out, repo.name) is None


# == The build files of a change that is running =============

_WARM_SH = pathlib.Path(sandbox.__file__).resolve().parent.parent / "sandbox" / "warm.sh"


def test_the_running_change_is_mounted_read_only_and_named():
    argv = sandbox.build_run_argv(
        "/usr/bin/podman", "an-image",
        sandbox.MountLayout(src="/host/repo", out="/host/out", scratch="/host/scratch",
                       cache="/host/cache", live="/host/live"),
        "abc123", "repo", ["run-tests"], network=True,
        entrypoint="/usr/local/bin/ctloom-warm")
    assert "/host/live:/live:ro" in argv
    assert "CTLOOM_LIVE=/live/repo" in argv
    assert "CTLOOM_DEP_PATH=" in argv


def test_the_dependency_files_taken_are_named_one_per_line():
    argv = sandbox.build_run_argv(
        "/usr/bin/podman", "an-image",
        sandbox.MountLayout(src="/host/repo", out="/host/out", scratch="/host/scratch",
                       cache="/host/cache", live="/host/live"),
        "abc123", "repo", ["run-tests"], network=True,
        entrypoint="/usr/local/bin/ctloom-warm",
        live_dep_path_list=["pyproject.toml", "sub dir/requirements.txt"])
    assert "CTLOOM_DEP_PATH=pyproject.toml\nsub dir/requirements.txt" in argv


def test_a_container_with_no_running_change_is_told_of_none():
    argv = _argv()
    assert "CTLOOM_LIVE" not in " ".join(argv)
    assert "/live" not in " ".join(argv)


def test_where_a_running_change_keeps_its_copy():
    assert sandbox.work_tree("/host/out", "a1", "repo", "20260101-000000-1") == \
        "/host/out/agents/.host/a1/repo/sandbox/scratch-20260101-000000-1/work/repo"


def _warm_sh(tmp_path, live=None, dep_path_list=()):
    """Run sandbox/warm.sh over a repository, with python standing in for the tool.

    Args:
        tmp_path: pytest's directory.
        live: The running change's copy, or None for a run without one.
        dep_path_list: The files the host names for it to take from live.

    Returns:
        The clone the script left behind, as a pathlib.Path.
    """
    src = _repo(tmp_path / "src")
    (src / "pyproject.toml").write_text("[project]\nname = 'p'\n", encoding="utf-8")
    (src / "app.py").write_text("original\n", encoding="utf-8")
    _git(src, "add", "-A")
    _git(src, "commit", "--quiet", "-m", "second")
    # The script ends in "exec python -m ...", which is not what is under test
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "python").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (stub / "python").chmod(0o755)
    work = tmp_path / "work" / "repo"
    env = {**os.environ,
           "PATH": f"{stub}:{os.environ['PATH']}",
           "CTLOOM_SRC": str(src), "CTLOOM_WORK": str(work),
           "CTLOOM_BASE": _git(src, "rev-parse", "HEAD").strip(),
           "CTLOOM_LIVE": str(live) if live else "",
           "CTLOOM_DEP_PATH": "\n".join(dep_path_list)}
    done = subprocess.run(["sh", str(_WARM_SH), "run-tests"], env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return work


def test_warm_sh_takes_the_manifests_of_a_running_change_and_nothing_else(tmp_path):
    """The code the networked container runs is the clone's, whatever the change wrote."""
    live = tmp_path / "live"
    (live / "sub").mkdir(parents=True)
    (live / "pyproject.toml").write_text("[project]\nname = 'p'\ndeps = 'new'\n",
                                         encoding="utf-8")
    (live / "sub" / "requirements.txt").write_text("cowsay\n", encoding="utf-8")
    (live / "app.py").write_text("what the change wrote\n", encoding="utf-8")
    work = _warm_sh(tmp_path, live, ["pyproject.toml", "sub/requirements.txt"])
    assert "deps = 'new'" in (work / "pyproject.toml").read_text(encoding="utf-8")
    assert (work / "sub" / "requirements.txt").read_text(encoding="utf-8") == "cowsay\n"
    assert (work / "app.py").read_text(encoding="utf-8") == "original\n"


def test_warm_sh_leaves_the_clone_alone_when_no_change_is_running(tmp_path):
    work = _warm_sh(tmp_path)
    assert (work / "app.py").read_text(encoding="utf-8") == "original\n"
    assert (work / "pyproject.toml").read_text(encoding="utf-8") == \
        "[project]\nname = 'p'\n"


def _warm_sh_jvm_option(tmp_path, route_text: str | None, option: str = "",
                        route_status: int = 0) -> str:
    """Run sandbox/warm.sh and return the JAVA_TOOL_OPTIONS the tool was started with.

    Args:
        tmp_path: pytest's directory.
        route_text: What a ctloom-jvm-route on PATH prints, or None for a PATH without one.
        option: The JAVA_TOOL_OPTIONS the script itself is started with.
        route_status: The exit status of that ctloom-jvm-route.

    Returns:
        The value, or "unset" when the tool was started without the variable.
    """
    src = _repo(tmp_path / "src")
    stub = tmp_path / "bin"
    stub.mkdir()
    seen = tmp_path / "seen.txt"
    (stub / "python").write_text(
        f'#!/bin/sh\nprintf %s "${{JAVA_TOOL_OPTIONS-unset}}" > {shlex.quote(str(seen))}\n',
        encoding="utf-8")
    (stub / "python").chmod(0o755)
    if route_text is not None:
        (stub / "ctloom-jvm-route").write_text(
            f'#!/bin/sh\n[ "$1" = "$CTLOOM_RUNDIR" ] || exit 9\nprintf "%s\\n" '
            f'{shlex.quote(route_text)}\nexit {route_status}\n', encoding="utf-8")
        (stub / "ctloom-jvm-route").chmod(0o755)
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}",
           "CTLOOM_SRC": str(src), "CTLOOM_WORK": str(tmp_path / "work" / "repo"),
           "CTLOOM_BASE": _git(src, "rev-parse", "HEAD").strip(),
           "CTLOOM_RUNDIR": str(tmp_path)}
    env.pop("JAVA_TOOL_OPTIONS", None)
    if option:
        env["JAVA_TOOL_OPTIONS"] = option
    proc = subprocess.run(["sh", str(_WARM_SH), "run-tests"], env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return seen.read_text(encoding="utf-8")


@pytest.mark.parametrize("route_text, option, want", [
    ("-Dhttps.proxyHost=p -Dhttps.proxyPort=3128", "",
     "-Dhttps.proxyHost=p -Dhttps.proxyPort=3128"),
    ("-Dhttps.proxyHost=p", "-Xmx1g", "-Xmx1g -Dhttps.proxyHost=p"),
    ("", "", "unset"),
    (None, "", "unset")])
def test_warm_sh_hands_the_jvm_what_ctloom_jvm_route_prints(tmp_path, route_text, option,
                                                         want):
    assert _warm_sh_jvm_option(tmp_path, route_text, option) == want


def test_warm_sh_tells_the_tool_it_fills_the_package_cache(tmp_path):
    """The default test commands read it (a makefile is not built there)."""
    src = _repo(tmp_path / "src")
    stub = tmp_path / "bin"
    stub.mkdir()
    fill_path = tmp_path / "seen.txt"
    (stub / "python").write_text(
        f'#!/bin/sh\nprintf %s "${{CTLOOM_FILL-unset}}" > {shlex.quote(str(fill_path))}\n',
        encoding="utf-8")
    (stub / "python").chmod(0o755)
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}",
           "CTLOOM_SRC": str(src), "CTLOOM_WORK": str(tmp_path / "work" / "repo"),
           "CTLOOM_BASE": _git(src, "rev-parse", "HEAD").strip(),
           "CTLOOM_RUNDIR": str(tmp_path)}
    env.pop("CTLOOM_FILL", None)
    proc = subprocess.run(["sh", str(_WARM_SH), "run-tests"], env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert fill_path.read_text(encoding="utf-8") == "1"


def _warm_sh_gradle(tmp_path, tool_status: int = 0, seed_status: int = 0,
                    publish_status: int = 0, share: str = "/cache/gradle") -> tuple[int, list[str]]:
    """Run sandbox/warm.sh with a ctloom-gradle-share on PATH and return what was started.

    Args:
        tmp_path: pytest's directory.
        tool_status: The exit status of the tool the script runs.
        seed_status: The exit status of ctloom-gradle-share seed.
        publish_status: The exit status of ctloom-gradle-share publish.
        share: The GRADLE_RO_DEP_CACHE the script is started with. "" leaves it unset.

    Returns:
        (the script's exit status, one line per program started: its name, its
        arguments and the GRADLE_RO_DEP_CACHE it saw).
    """
    src = _repo(tmp_path / "src")
    stub = tmp_path / "bin"
    stub.mkdir()
    seen = tmp_path / "seen.txt"
    record = f'echo "$0 $* ro=${{GRADLE_RO_DEP_CACHE-unset}}" >> {shlex.quote(str(seen))}'
    (stub / "python").write_text(f"#!/bin/sh\n{record}\nexit {tool_status}\n",
                                 encoding="utf-8")
    (stub / "ctloom-gradle-share").write_text(
        f'#!/bin/sh\n{record}\n[ "$1" = seed ] && exit {seed_status}\n'
        f"exit {publish_status}\n", encoding="utf-8")
    for name in ("python", "ctloom-gradle-share"):
        (stub / name).chmod(0o755)
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}",
           "CTLOOM_SRC": str(src), "CTLOOM_WORK": str(tmp_path / "work" / "repo"),
           "CTLOOM_BASE": _git(src, "rev-parse", "HEAD").strip(),
           "CTLOOM_RUNDIR": str(tmp_path), "GRADLE_USER_HOME": "/tmp/gradle-home"}
    env.pop("GRADLE_RO_DEP_CACHE", None)
    if share:
        env["GRADLE_RO_DEP_CACHE"] = share
    proc = subprocess.run(["sh", str(_WARM_SH), "run-tests"], env=env,
                          capture_output=True, text=True)
    line_list = seen.read_text(encoding="utf-8").splitlines() if seen.exists() else []
    return proc.returncode, [line.replace(f"{stub}/", "") for line in line_list]


def test_warm_sh_starts_gradle_s_home_from_the_share_and_puts_what_it_fetched_there(tmp_path):
    """The tool is run without the share named, so what it fetches is all in its home."""
    assert _warm_sh_gradle(tmp_path) == (0, [
        "ctloom-gradle-share seed /cache/gradle /tmp/gradle-home ro=unset",
        "python -P -m codetwine_loom run-tests ro=unset",
        "ctloom-gradle-share publish /tmp/gradle-home /cache/gradle ro=unset"])


@pytest.mark.parametrize("tool_status, seed_status, publish_status, want, count", [
    (3, 0, 0, 3, 3), (0, 0, 1, 1, 3), (3, 0, 1, 3, 3),
    # A home that could not be started from the share is not put there
    (0, 1, 0, 1, 2), (3, 1, 0, 3, 2)])
def test_warm_sh_ends_with_the_tool_s_status_unless_only_the_share_failed(
        tmp_path, tool_status, seed_status, publish_status, want, count):
    status, line_list = _warm_sh_gradle(tmp_path, tool_status, seed_status, publish_status)
    assert status == want and len(line_list) == count
    assert [line.split()[1] for line in line_list if line.startswith("ctloom-gradle-share")] \
        == (["seed", "publish"] if count == 3 else ["seed"])


def test_warm_sh_with_no_share_named_runs_the_tool_alone(tmp_path):
    assert _warm_sh_gradle(tmp_path, tool_status=4, share="") == (4, [
        "python -P -m codetwine_loom run-tests ro=unset"])


@pytest.mark.parametrize("env_dict, want", [
    ({"ALL_PROXY": "http://all.example:3128"},
     "https=http://all.example:3128 http=http://all.example:3128"),
    ({"all_proxy": "all.example:3128"},
     "https=http://all.example:3128 http=http://all.example:3128"),
    ({"ALL_PROXY": "http://all.example:3128", "https_proxy": "http://own.example:1"},
     "https=http://own.example:1 http=http://all.example:3128"),
    ({"ALL_PROXY": "http://all.example:3128", "HTTP_PROXY": "http://own.example:1"},
     "https=http://all.example:3128 http=http://own.example:1"),
    ({"ALL_PROXY": "socks5://all.example:1080"}, "https= http="),
    ({}, "https= http=")])
def test_warm_sh_has_all_proxy_stand_in_for_the_proxy_variables_that_are_not_set(
        tmp_path, env_dict, want):
    """npm reads HTTPS_PROXY and HTTP_PROXY alone."""
    src = _repo(tmp_path / "src")
    stub = tmp_path / "bin"
    stub.mkdir()
    seen = tmp_path / "seen.txt"
    (stub / "python").write_text(
        '#!/bin/sh\nprintf "https=%s http=%s" "${https_proxy:-${HTTPS_PROXY:-}}" '
        f'"${{http_proxy:-${{HTTP_PROXY:-}}}}" > {shlex.quote(str(seen))}\n',
        encoding="utf-8")
    (stub / "python").chmod(0o755)
    env = {name: value for name, value in os.environ.items()
           if name.upper() not in config.PROXY_NAME_TUPLE}
    env.update(PATH=f"{stub}:{os.environ['PATH']}", CTLOOM_SRC=str(src),
               CTLOOM_WORK=str(tmp_path / "work" / "repo"),
               CTLOOM_BASE=_git(src, "rev-parse", "HEAD").strip(), CTLOOM_RUNDIR=str(tmp_path),
               **env_dict)
    proc = subprocess.run(["sh", str(_WARM_SH), "run-tests"], env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert seen.read_text(encoding="utf-8") == want


def test_warm_sh_goes_on_without_the_options_when_ctloom_jvm_route_fails(tmp_path):
    assert _warm_sh_jvm_option(tmp_path, "-Dhttps.proxyHost=p", route_status=1) == "unset"


def test_warm_does_not_mount_a_copy_that_is_not_there(tmp_path, monkeypatch, caplog):
    """warm run by hand, before any change has started, has none to take from."""
    caplog.set_level("INFO", logger="codetwine_loom.sandbox")
    repo = _repo(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    sandbox.warm(str(repo), str(tmp_path / "out"), live_dep_file=True)
    assert "no change is running" in caplog.text
    assert "CTLOOM_LIVE" not in " ".join(seen[0])


def test_warm_uses_a_scratch_of_its_own(tmp_path, monkeypatch):
    """A refill runs while the change's container still holds the other one."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    sandbox.warm(str(repo), str(out))
    mounts = [a for a in seen[0] if ":/work" in a]
    assert len(mounts) == 1
    assert mounts[0].startswith(f"{out}/agents/.host/")
    assert "/repo/sandbox/scratch-warm-" in mounts[0]
    assert mounts[0].endswith("/work:/work")


def test_warm_names_its_container_after_the_filling_that_names_its_clone(tmp_path,
                                                                         monkeypatch):
    """A clone whose host process was killed is only swept once the engine says the
    container is gone too, and the engine is asked by that name."""
    repo = _repo(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    sandbox.warm(str(repo), str(tmp_path / "out"))
    argv = seen[0]
    name = argv[argv.index("--name") + 1]
    assert name.startswith("ctloom-")
    mount = next(a for a in argv if ":/work" in a)
    assert f"scratch-warm-{name[len('ctloom-'):]}/" in mount


def test_the_cache_is_filled_under_a_lock_on_the_cache_itself(tmp_path, monkeypatch):
    """The cache is the one thing every agent shares, so two fillings - a warm and a
    refill, or two agents' refills - go one after the other."""
    repo = _repo(tmp_path / "repo")
    events = []
    real_held = sandbox.filelock.hold_lock

    @contextlib.contextmanager
    def noted(path, what, wait=True):
        events.append(("enter", what))
        try:
            with real_held(path, what, wait=wait):
                yield
        finally:
            events.append(("exit", what))

    monkeypatch.setattr(sandbox.filelock, "hold_lock", noted)

    def ran(argv):
        events.append(("container", ""))
        return subprocess.CompletedProcess(argv, 0)

    _engine_stand_in(monkeypatch, ran)
    sandbox.warm(str(repo), str(tmp_path / "out"))
    what = "the shared package cache"
    inside = events[events.index(("enter", what)) + 1:events.index(("exit", what))]
    assert ("container", "") in inside


# == Working from the package cache alone =====================


def test_a_container_with_no_network_is_told_to_work_from_the_cache():
    """Without it the build tool asks a registry whose name it cannot resolve."""
    argv = _argv()
    assert "--network=none" in argv
    for name, value in OFFLINE_ENV_DICT.items():
        assert f"{name}={value}" in argv


def test_the_container_that_fills_the_cache_is_not_told_that():
    """It is the one with a network, and it is there to fetch."""
    argv = _argv(network=True)
    assert not [a for a in argv if a.partition("=")[0] in OFFLINE_ENV_DICT]


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_a_language_that_fetches_says_how_to_work_from_the_cache(ext):
    """A language with a manifest has a tool that fetches, and that tool has to be
    told, one way or the other, or a sandboxed run of its tests fails however full the
    cache is.
    """
    cfg = LANG_DICT[ext]
    assert bool(cfg.offline_env_tuple or cfg.offline_arg_tuple) == bool(cfg.dep_file_tuple), \
        f"{ext} has manifests={cfg.dep_file_tuple} and nothing that says to work offline"


def test_every_language_offline_setting_reaches_the_container():
    """The table is read off LANG, so a setting added to one language travels."""
    argv = _argv()
    for cfg in LANG_DICT.values():
        for name, value in cfg.offline_env_tuple:
            assert f"{name}={value}" in argv


def test_the_tool_that_takes_it_on_the_command_line_is_told_too():
    """gradle has no setting of its own, so the run adds it where it builds argv.

    What the container does with it is test_config's; here it is that it travels.
    """
    assert "TOOL_OFFLINE=true" in _argv()
    assert "TOOL_OFFLINE=false" in _argv(network=True)


def test_the_setting_is_written_out_rather_than_forwarded_by_name():
    """Forwarding it as well would leave which of the two wins up to the engine."""
    assert "TOOL_OFFLINE" in sandbox._SET_BY_HAND_TUPLE
    assert "TOOL_OFFLINE" not in sandbox.env_forward_argv()


@pytest.mark.parametrize("value", ["false", "true", None])
def test_a_change_made_in_a_container_is_recorded_whatever_the_setting_is_here(
        monkeypatch, value):
    """The record is what the commit's trailers are written from."""
    if value is None:
        monkeypatch.delenv("EDIT_HISTORY", raising=False)
    else:
        monkeypatch.setenv("EDIT_HISTORY", value)
    monkeypatch.setattr(config, "EDIT_HISTORY", value != "false")
    assert [arg for arg in _argv() if arg.split("=")[0] == "EDIT_HISTORY"] == \
        ["EDIT_HISTORY=true"]
    assert "EDIT_HISTORY" not in sandbox.env_forward_argv()


# == What the container is not told ===========================


def test_the_search_endpoint_stays_with_the_broker(monkeypatch):
    """WEB_SEARCH_PROVIDER is in config.SECRET_NAME_TUPLE, so the container never sees
    where the search goes. It does not need to: the search is made where the network
    is."""
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    argv = _argv()
    assert not [a for a in argv if a.startswith(("WEB_SEARCH_URL", "WEB_SEARCH_PROVIDER"))]
    assert "WEB_SEARCH_PROVIDER" not in sandbox.env_forward_argv()


# == The time limit on the container ==========================


def _engine_run(recorded, raise_timeout_for):
    """Return a stand-in for subprocess.run that records what it was given.

    Args:
        recorded: The list each call's (argv, timeout) goes into.
        raise_timeout_for: The argv[0] whose call raises TimeoutExpired, or "" for none.

    Returns:
        The stand-in.
    """
    def run(argv, **kwargs):
        recorded.append((list(argv), kwargs.get("timeout")))
        if raise_timeout_for and argv[0] == raise_timeout_for and "rm" not in argv:
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout") or 0)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    return run


def test_the_container_is_given_a_pipe_for_each_of_its_streams(monkeypatch):
    given = {}

    def run(argv, **kwargs):
        given.update(kwargs)
        os.write(kwargs["stdout"], b"verdict: unverified\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    assert sandbox.run_the_container(["podman", "run"], "podman", "r1", 0, "") == 0
    assert isinstance(given["stdout"], int) and isinstance(given["stderr"], int)
    assert given["stdout"] != given["stderr"]


def _write_both_stream(argv, **kwargs):
    """Stand in for subprocess.run, writing one line to each of the container's streams."""
    os.write(kwargs["stdout"], b"verdict: unverified\n")
    os.write(kwargs["stderr"], b"edit turn 1 of at most 6: read_file\n")
    return subprocess.CompletedProcess(argv, 0)


def test_what_the_container_writes_comes_out_under_the_label(monkeypatch, capsys,
                                                               tmp_path):
    monkeypatch.setattr(progress, "_CHILD_STDOUT_TO_FILE", False)
    monkeypatch.setattr(sandbox.subprocess, "run", _write_both_stream)
    with progress.label("[task 2/5] "):
        sandbox.run_the_container(["podman", "run"], "podman", "r1", 0,
                                  str(tmp_path / "stdout.txt"))
    out, err = capsys.readouterr()
    assert out == "[task 2/5] verdict: unverified\n"
    assert err == "[task 2/5] edit turn 1 of at most 6: read_file\n"
    assert not (tmp_path / "stdout.txt").exists()


def test_a_kept_container_stdout_goes_to_its_file(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(progress, "_CHILD_STDOUT_TO_FILE", False)
    progress.keep_child_stdout(True)
    monkeypatch.setattr(sandbox.subprocess, "run", _write_both_stream)
    out_path = tmp_path / "sandbox" / "stdout-r1.txt"
    with progress.label("[task 2/5] "):
        sandbox.run_the_container(["podman", "run"], "podman", "r1", 0, str(out_path))
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "[task 2/5] edit turn 1 of at most 6: read_file\n"
    assert out_path.read_text(encoding="utf-8") == "[task 2/5] verdict: unverified\n"


def test_the_stdout_file_is_named_after_the_run():
    assert sandbox.stdout_path("/out/repo/sandbox", "r1") == "/out/repo/sandbox/stdout-r1.txt"


def test_a_container_over_its_limit_is_taken_away_and_counts_as_unfinished(monkeypatch):
    """Killing the engine's client does not always end what it started, and a container
    still holding the clone stops the next run of the same agent."""
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SEC", 30)
    seen = []
    monkeypatch.setattr(sandbox.subprocess, "run",
                        _engine_run(seen, "/usr/bin/podman"))
    status = sandbox.run_the_container(["/usr/bin/podman", "run"], "/usr/bin/podman",
                                       "20260101-000000-1-2", 30, "")
    assert status == sandbox._TIME_OUT_STATUS
    assert seen[0][1] == 30
    assert seen[-1][0] == ["/usr/bin/podman", "rm", "--force",
                           sandbox.container_name("20260101-000000-1-2")]


def test_no_limit_waits_for_the_container_as_it_always_did(monkeypatch):
    seen = []
    monkeypatch.setattr(sandbox.subprocess, "run", _engine_run(seen, ""))
    assert sandbox.run_the_container(["/usr/bin/podman", "run"], "/usr/bin/podman",
                                     "20260101-000000-1-2", 0, "") == 0
    assert seen == [(["/usr/bin/podman", "run"], None)]


def test_a_killed_container_names_the_memory_settings(monkeypatch, caplog):
    monkeypatch.setattr(config, "SANDBOX_MEMORY", "2g")
    monkeypatch.setattr(sandbox.subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, sandbox._KILL_STATUS))
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        status = sandbox.run_the_container(["/usr/bin/podman", "run"], "/usr/bin/podman",
                                           "20260101-000000-1-2", 0, "")
    assert status == sandbox._KILL_STATUS
    assert "SANDBOX_MEMORY (2g)" in caplog.text
    assert "DISPATCH_PARALLEL" in caplog.text


def test_a_client_ended_by_a_signal_gives_the_status_a_shell_gives_it(tmp_path,
                                                                     monkeypatch, caplog):
    monkeypatch.setattr(sandbox.subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, -15))
    assert sandbox.run_the_container(["/usr/bin/podman", "run"], "/usr/bin/podman",
                                     "20260101-000000-1-2", 0, "") == 143
    monkeypatch.setattr(sandbox, "_start_client", _START_CLIENT)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        status = sandbox.run_the_container(["sh", "-c", "kill -KILL $$"], "sh",
                                           "20260101-000000-1-2", 0,
                                           str(tmp_path / "out.txt"))
    assert status == sandbox._KILL_STATUS
    assert "client of ctloom-20260101-000000-1-2 was ended by signal 9" in caplog.text
    assert "exit status 137" in caplog.text and "SANDBOX_MEMORY" not in caplog.text
    assert sandbox._LIVE_CONTAINER_DICT == {}


def test_another_failure_says_nothing_about_memory(monkeypatch, caplog):
    monkeypatch.setattr(sandbox.subprocess, "run", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 1))
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        assert sandbox.run_the_container(["/usr/bin/podman", "run"], "/usr/bin/podman",
                                         "20260101-000000-1-2", 0, "") == 1
    assert "SANDBOX_MEMORY" not in caplog.text


# == Where the cache is filled ================================


def test_the_cache_is_filled_in_a_container_of_its_own(tmp_path, monkeypatch):
    """The build files it runs are the repository's code, so they do not run in the
    process holding the credentials."""
    repo = _repo(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    result = sandbox.warm(str(repo), str(tmp_path / "out"))
    assert seen[0][0] == result["engine"] and result["image"] == config.CONTAINER_IMAGE
    assert "--network=none" not in seen[0]


def _index_in(out_dir, type_, file):
    """Write an index of one definition under out_dir/repo, whose language file says."""
    from conftest import write_index
    (out_dir / "repo").mkdir(parents=True)
    defs = {f"{file}::X": {"name": "X", "type": type_, "file": file, "lines": [1, 2],
                           "signature": "X", "doc": ""}}
    write_index(str(out_dir / "repo" / store.INDEX_FILE), defs, {}, "repo").close()


def test_no_cache_is_filled_where_there_is_no_test_command(tmp_path, monkeypatch):
    """The filling runs the tests; a COBOL or SQL repository with no TEST_COMMAND has
    none, and a container started for it could only fail."""
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    _index_in(tmp_path / "out", "paragraph_header", "repo/src/MAIN_cbl/MAIN.cbl")
    repo = _repo(tmp_path / "repo")
    monkeypatch.setattr(sandbox, "warm", lambda *a, **k: pytest.fail("a filling started"))
    _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", str(tmp_path / "cache"),
               "podman", "image", None, _commit_file(repo, "MAIN.cbl"))


def test_the_cache_is_filled_where_the_index_names_a_test_command(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    _index_in(tmp_path / "out", "function_definition", "repo/a_py/a.py")
    repo = _repo(tmp_path / "repo")
    call_list = []
    monkeypatch.setattr(sandbox, "warm",
                        lambda *a, **k: call_list.append(a) or {"exit_status": 0})
    _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", str(tmp_path / "cache"),
               "podman", "image", None, _git(repo, "rev-parse", "HEAD").strip())
    assert len(call_list) == 1


def _index_of_two_languages(tmp_path, primary_file, primary_type, other_file):
    """Write an index whose one definition is in primary_file and which holds
    other_file too, and make the repository both are in.

    Returns:
        (the repository, the output directory), both as str.
    """
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"

    def knowledge_path(rel):
        stem = pathlib.PurePosixPath(rel)
        return f"repo/{stem.parent}/{stem.name.replace('.', '_')}/{stem.name}".replace(
            "/./", "/")

    key = knowledge_path(primary_file)
    files = {knowledge_path(rel): {"summary": "", "defs": []}
             for rel in (primary_file, other_file)}
    defs = {f"{key}::X": {"name": "X", "type": primary_type, "file": key, "lines": [1, 2],
                          "signature": "X", "doc": "", "calls": [], "called_by": []}}
    write_index(str(out / "repo" / store.INDEX_FILE), defs, files, project="repo").close()
    return str(repo), str(out)


# The files of a repository whose primary language has no test command, with whether a
# run over it has a test to run
_OTHER_TEST_CASE_LIST = [
    ("src/MAIN.cbl", "paragraph_header", "tools/gen.py", [], False),
    ("src/MAIN.cbl", "paragraph_header", "tools/gen.py", ["pyproject.toml"], False),
    ("src/MAIN.cbl", "paragraph_header", "tools/gen.py",
     ["pyproject.toml", "tests/test_gen.py"], True),
    ("src/MAIN.cbl", "paragraph_header", "tools/gen.py",
     ["tools/pyproject.toml", "tests/test_gen.py"], False),
    ("db/schema.sql", "create_table", "web/app.js", ["web/package.json"], True),
    ("db/schema.sql", "create_table", "web/app.js", ["web/readme.md"], False)]


@pytest.mark.parametrize("primary_file, primary_type, other_file, file_list, expected",
                         _OTHER_TEST_CASE_LIST)
def test_a_first_filling_is_made_for_the_tests_of_another_language_too(
        tmp_path, monkeypatch, primary_file, primary_type, other_file, file_list,
        expected):
    """The primary language of a COBOL or SQL repository has no test command, and the
    run inside still runs the tests of the other languages whose dependency file and
    tests the repository holds; the cache is filled for exactly those runs."""
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    repo, out = _index_of_two_languages(tmp_path, primary_file, primary_type, other_file)
    for rel in [other_file, *file_list]:
        path = pathlib.Path(repo, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    assert sandbox._has_test_to_run(repo, out, "repo") is expected
    call_list = []
    monkeypatch.setattr(sandbox, "warm",
                        lambda *a, **k: call_list.append(a) or {"exit_status": 0})
    _REAL_FILL(repo, out, "a1", str(tmp_path / "cache"), "podman", "image", None, "base")
    assert len(call_list) == (1 if expected else 0)
    # What the run inside the container works out over the same index and repository,
    # with every language's command on PATH
    primary_ext, other_ext = (pathlib.PurePosixPath(name).suffix[1:]
                              for name in (primary_file, other_file))
    for name in ("PRIMARY_EXT", "TEST_COMMAND", "TEST_PATH", "TEST_TARGET_STYLE",
                 "RUN_COMMAND", "CHECK_COMMAND", "OTHER_CHECK_LIST", "OTHER_TEST_LIST"):
        monkeypatch.setattr(config, name, getattr(config, name))
    monkeypatch.setattr(config, "PROJECT_DIR", repo)
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(sandbox.edit_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    config.set_primary_ext(primary_ext)
    config.set_other_ext([primary_ext, other_ext])
    assert sandbox.edit_mod.test_can_run() is expected


def test_a_test_command_set_by_hand_is_a_test_to_run_whatever_the_language(tmp_path,
                                                                           monkeypatch):
    _index_in(tmp_path / "out", "paragraph_header", "repo/src/MAIN_cbl/MAIN.cbl")
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    assert sandbox._has_test_to_run(str(tmp_path), str(tmp_path / "out"), "repo") is False
    monkeypatch.setenv("TEST_COMMAND", "sh tests/run.sh")
    assert sandbox._has_test_to_run(str(tmp_path), str(tmp_path / "out"), "repo") is True
    # No index to read: the filling is left to find out
    assert sandbox._has_test_to_run(str(tmp_path), str(tmp_path / "none"), "repo") is True


def test_the_cache_variables_reach_the_container(tmp_path, monkeypatch):
    """One place decides where a home and a cache go, so nothing can disagree."""
    names = sorted(sandbox._cache_env_dict("/cache"))
    assert sorted(a.partition("=")[0] for a in _argv()
                  if a.partition("=")[0] in names) == names


def test_the_directories_made_are_the_ones_the_variables_name(tmp_path):
    """A variable pointing somewhere nothing made is a cache that is not written."""
    cache = sandbox.make_cache(str(tmp_path / "cache"))
    for directory in sandbox._cache_env_dict(cache).values():
        assert os.path.isdir(directory)


def test_two_agents_making_the_cache_at_the_same_moment_both_go_on(tmp_path, monkeypatch):
    """Both are held where the last directory is made until the other is there too."""
    cache = str(tmp_path / "cache")
    module_dir = os.path.join(cache, "gradle", "modules-2")
    barrier = threading.Barrier(2, timeout=30)
    mkdir = os.mkdir

    def mkdir_together(path, *args, **kwargs):
        if str(path) == module_dir:
            barrier.wait()
        return mkdir(path, *args, **kwargs)
    monkeypatch.setattr(sandbox.os, "mkdir", mkdir_together)
    error_list = []

    def make():
        try:
            sandbox.make_cache(cache)
        except Exception as e:
            error_list.append(e)
    thread_list = [threading.Thread(target=make) for _ in range(2)]
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join(60)
    assert error_list == [] and os.path.isdir(module_dir)


def test_the_directory_gradle_reads_fetched_dependencies_from_is_made_with_the_cache(
        tmp_path):
    """Gradle writes lines about a directory it is pointed at that is not there."""
    cache = sandbox.make_cache(str(tmp_path / "cache"))
    share = sandbox._jvm_tool_env_dict(cache, "/tmp")["GRADLE_RO_DEP_CACHE"]
    assert share == os.path.join(cache, "gradle")
    assert os.listdir(share) == ["modules-2"] and os.listdir(f"{share}/modules-2") == []
    # What a container that fetched put under the name is left as it is
    os.rmdir(f"{share}/modules-2")
    os.mkdir(f"{share}/snap.1")
    pathlib.Path(share, "snap.1", "a.jar").write_text("jar", encoding="utf-8")
    os.symlink("snap.1", f"{share}/modules-2")
    sandbox.make_cache(cache)
    assert os.readlink(f"{share}/modules-2") == "snap.1"
    assert sorted(os.listdir(share)) == ["modules-2", "snap.1"]
    # A link that leads nowhere under the name is left as well
    os.unlink(f"{share}/modules-2")
    os.symlink("snap.gone", f"{share}/modules-2")
    sandbox.make_cache(cache)
    assert os.readlink(f"{share}/modules-2") == "snap.gone"
    # A link under the name of the directory itself stops the run
    link_cache = tmp_path / "other"
    (link_cache / "elsewhere").mkdir(parents=True)
    os.symlink(link_cache / "elsewhere", link_cache / "gradle")
    with pytest.raises(RuntimeError, match="is not a directory of its own"):
        sandbox.make_cache(str(link_cache))
    assert os.listdir(link_cache / "elsewhere") == []


@pytest.mark.parametrize("where", ["modules-2", "home"])
def test_a_directory_of_the_cache_that_cannot_be_made_stops_the_run_in_one_line(
        tmp_path, monkeypatch, where):
    """Any container may leave the directory above it one nothing can be made in."""
    cache = sandbox.make_cache(str(tmp_path / "cache"))
    target = os.path.join(cache, "gradle", "modules-2") if where == "modules-2" \
        else os.path.join(cache, "home")
    os.rmdir(target)

    def refuse(path, *args, **kwargs):
        if str(path) == target:
            raise PermissionError(13, "Permission denied", str(path))
        return real_mkdir(path, *args, **kwargs)

    real_mkdir = os.mkdir
    monkeypatch.setattr(sandbox.os, "mkdir", refuse)
    with pytest.raises(RuntimeError, match=f"{target} in the package cache cannot be made "
                                           f"\\(Permission denied\\)"):
        sandbox.make_cache(cache)


# == Where the run is started from ============================

_ENTRYPOINT_SH = pathlib.Path(sandbox.__file__).resolve().parent.parent / "sandbox" \
    / "entrypoint.sh"


def _home_with_git_setting(tmp_path):
    """Make a HOME whose git settings name a template with a hook and a file system
    monitor, each of which writes a line to the file returned when it runs.

    Returns:
        (the home directory, the file the planted programs write to).
    """
    home = tmp_path / "cache" / "home"
    ran = tmp_path / "planted-ran.txt"
    hook = home / "tpl" / "hooks" / "post-checkout"
    hook.parent.mkdir(parents=True)
    hook.write_text(f'#!/bin/sh\necho hook >> "{ran}"\n', encoding="utf-8")
    hook.chmod(0o755)
    monitor = home / "monitor.sh"
    monitor.write_text(f'#!/bin/sh\necho monitor >> "{ran}"\n', encoding="utf-8")
    monitor.chmod(0o755)
    setting = f"[init]\n\ttemplateDir = {home / 'tpl'}\n[core]\n\tfsmonitor = {monitor}\n"
    (home / ".gitconfig").write_text(setting, encoding="utf-8")
    (home / ".config" / "git").mkdir(parents=True)
    (home / ".config" / "git" / "config").write_text(setting, encoding="utf-8")
    return home, ran


# What the stand-in python runs: a git command in the clone, and the settings it was
# started with
_GIT_IN_TOOL = ('git -C "$CTLOOM_WORK" status --short >/dev/null; '
                'env | sort > "$CTLOOM_RUNDIR/tool-env.txt"')


def test_the_entry_point_s_git_reads_no_setting_and_runs_no_hook_of_the_home_it_is_given(
        tmp_path):
    home, ran = _home_with_git_setting(tmp_path)
    _entrypoint_sh(tmp_path, f'pwd > "$CTLOOM_OUT/started-from.txt"\n{_GIT_IN_TOOL}',
                   {"HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config")})
    assert (tmp_path / "out" / "exit-status.txt").read_text() == "0"
    assert not ran.exists()
    assert not (tmp_path / "work" / "repo" / ".git" / "hooks").exists()
    tool_env = (tmp_path / "rundir" / "tool-env.txt").read_text(encoding="utf-8")
    assert "GIT_CONFIG_GLOBAL=/dev/null\n" in tool_env


# The lines of its environment with which a git started in a container reads no ignore
# file and no attributes file of the user's
_GIT_FILE_PIN_TUPLE = ("GIT_CONFIG_COUNT=4",
                       "GIT_CONFIG_KEY_2=core.excludesFile", "GIT_CONFIG_VALUE_2=/dev/null",
                       "GIT_CONFIG_KEY_3=core.attributesFile", "GIT_CONFIG_VALUE_3=/dev/null")


@pytest.mark.parametrize("config_home", ["", ".config"])
def test_the_entry_point_s_git_reads_no_ignore_and_no_attributes_file_of_its_home(
        tmp_path, monkeypatch, config_home):
    """The home holds an ignore file naming every .py file and an attributes file that
    merges every file by keeping both sides, where git looks for them with
    XDG_CONFIG_HOME unset and with it set."""
    home = tmp_path / "cache" / "home"
    (home / ".config" / "git").mkdir(parents=True)
    (home / ".config" / "git" / "ignore").write_text("*.py\n", encoding="utf-8")
    (home / ".config" / "git" / "attributes").write_text("* merge=union\n",
                                                         encoding="utf-8")
    script = """
pwd > "$CTLOOM_OUT/started-from.txt"
echo x > "$CTLOOM_WORK/new_module.py"; echo x > "$CTLOOM_WORK/new_note.txt"
git -C "$CTLOOM_WORK" check-attr merge -- kept.txt > "$CTLOOM_OUT/attr.txt"
git -C "$CTLOOM_WORK" status --short > "$CTLOOM_OUT/status.txt"
env | sort > "$CTLOOM_RUNDIR/tool-env.txt"
"""
    extra_env_dict = {"HOME": str(home)}
    if config_home:
        extra_env_dict["XDG_CONFIG_HOME"] = str(home / config_home)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    _entrypoint_sh(tmp_path, script, extra_env_dict)
    out = tmp_path / "out"
    assert (out / "exit-status.txt").read_text() == "0"
    # The tool's own git sees both new files and no merge attribute
    assert (out / "status.txt").read_text().split() == ["??", "new_module.py", "??",
                                                        "new_note.txt"]
    assert (out / "attr.txt").read_text().strip() == "kept.txt: merge: unspecified"
    # So does the git that writes the patch
    patch = (out / "patch.diff").read_text(encoding="utf-8")
    assert "b/new_module.py" in patch and "b/new_note.txt" in patch
    meta_dict = json.loads((out / "patch.diff.meta.json").read_text())
    assert (meta_dict["files"], meta_dict["unadded"]) == (2, [])
    tool_env = (tmp_path / "rundir" / "tool-env.txt").read_text(encoding="utf-8")
    for line in _GIT_FILE_PIN_TUPLE:
        assert f"{line}\n" in tool_env


def test_warm_sh_reads_no_setting_of_the_home_it_is_given(tmp_path):
    home, ran = _home_with_git_setting(tmp_path)
    src = _repo(tmp_path / "src")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "python").write_text(f"#!/bin/sh\n{_GIT_IN_TOOL}\n", encoding="utf-8")
    (stub / "python").chmod(0o755)
    rundir = tmp_path / "rundir"
    rundir.mkdir()
    work = tmp_path / "work" / "repo"
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "HOME": str(home),
           "XDG_CONFIG_HOME": str(home / ".config"), "CTLOOM_SRC": str(src),
           "CTLOOM_WORK": str(work), "CTLOOM_BASE": _git(src, "rev-parse", "HEAD").strip(),
           "CTLOOM_RUNDIR": str(rundir)}
    done = subprocess.run(["sh", str(_WARM_SH), "run-tests"], env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert not ran.exists()
    assert not (work / ".git" / "hooks").exists()
    tool_env = (rundir / "tool-env.txt").read_text(encoding="utf-8")
    for line in ("GIT_CONFIG_GLOBAL=/dev/null", "XDG_CONFIG_HOME=/nonexistent",
                 "PIP_CONFIG_FILE=/dev/null", "NPM_CONFIG_USERCONFIG=/dev/null",
                 "MAVEN_SKIP_RC=1", *_GIT_FILE_PIN_TUPLE):
        assert f"{line}\n" in tool_env
    # The home itself is still the one it was given
    assert f"HOME={home}\n" in tool_env


def _entrypoint_sh(tmp_path, script, extra_env_dict=None, command="edit"):
    """Run sandbox/entrypoint.sh over a repository, with python standing in for the tool.

    Args:
        tmp_path: pytest's directory.
        script: The shell body the stand-in python runs instead of the tool.
        extra_env_dict: More variables for the script, or None.
        command: The subcommand the entry point is given.

    Returns:
        The directory the run was started from, as the stand-in recorded it.
    """
    src = _repo(tmp_path / "src")
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "python").write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    (stub / "python").chmod(0o755)
    rundir = tmp_path / "rundir"
    rundir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    env = {**os.environ,
           "PATH": f"{stub}:{os.environ['PATH']}",
           "CTLOOM_SRC": str(src), "CTLOOM_WORK": str(tmp_path / "work" / "repo"),
           "CTLOOM_BASE": _git(src, "rev-parse", "HEAD").strip(),
           "CTLOOM_OUT": str(out), "CTLOOM_PATCH": str(out / "patch.diff"),
           "CTLOOM_RUNDIR": str(rundir), **(extra_env_dict or {})}
    proc = subprocess.run(["sh", str(_ENTRYPOINT_SH), command], env=env, check=False,
                          capture_output=True, text=True)
    (out / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    (out / "exit-status.txt").write_text(str(proc.returncode), encoding="utf-8")
    return (out / "started-from.txt").read_text(encoding="utf-8").strip()


def _write_patch_with(tmp_path, script, command="edit"):
    """Run the entry point with the tool stood in for by script, and return what it
    left: (the patch's text, its summary, the entry point's stderr and exit status)."""
    _entrypoint_sh(tmp_path, f'pwd > "$CTLOOM_OUT/started-from.txt"\n{script}',
                   command=command)
    out = tmp_path / "out"
    return ((out / "patch.diff").read_text(encoding="utf-8"),
            json.loads((out / "patch.diff.meta.json").read_text(encoding="utf-8")),
            (out / "stderr.txt").read_text(encoding="utf-8"),
            int((out / "exit-status.txt").read_text()))


def test_a_path_git_stops_at_in_the_clone_keeps_only_itself_out_of_the_patch(tmp_path):
    """A file its attribute calls UTF-16 with no byte order mark stops "git add" and
    nothing is added; the other paths are then added one at a time."""
    patch, meta, stderr, status = _write_patch_with(
        tmp_path, 'echo two >> "$CTLOOM_WORK/kept.txt"\n'
                  'echo "*.u16 working-tree-encoding=UTF-16" > "$CTLOOM_WORK/.gitattributes"\n'
                  'echo plain > "$CTLOOM_WORK/bad.u16"\necho new > "$CTLOOM_WORK/-n.txt"\n'
                  'git -C "$CTLOOM_WORK" config core.safecrlf true\n'
                  'git -C "$CTLOOM_WORK" config core.autocrlf input\n'
                  'printf "a\\r\\nb\\r\\n" > "$CTLOOM_WORK/crlf.txt"')
    assert status == 0 and meta["files"] == 4
    assert "+two" in patch and "b/.gitattributes" in patch and "b/-n.txt" in patch
    assert "b/crlf.txt" in patch and "bad.u16" not in patch
    assert meta["unadded"] == ["bad.u16"]


@pytest.mark.parametrize("leave, path", [
    ('git init -q "$CTLOOM_WORK/tmp_repo"', "tmp_repo/"),
    ('echo x > "$CTLOOM_WORK/no read.bin"; chmod 000 "$CTLOOM_WORK/no read.bin"', "no read.bin"),
    ('echo x > "$CTLOOM_WORK/a\\"b.txt"; chmod 000 "$CTLOOM_WORK/a\\"b.txt"', '"a\\"b.txt"')])
def test_a_path_git_cannot_add_keeps_nothing_else_out_of_the_patch(tmp_path, leave, path):
    """The patch and its summary are written all the same, and the summary names what
    is not in the patch."""
    patch, meta, stderr, status = _write_patch_with(
        tmp_path, f'echo two >> "$CTLOOM_WORK/kept.txt"\n{leave}\nexit 3')
    assert "+two" in patch and status == 3
    assert meta["exit_status"] == 3 and meta["files"] == 1
    assert meta["unadded"] == [path] and meta["nested"] == []
    assert "git could not add these paths" in stderr and path in stderr
    assert "moved_head" not in meta


def test_a_repository_left_inside_the_clone_is_named_in_the_summary(tmp_path):
    patch, meta, stderr, _ = _write_patch_with(tmp_path, """
echo two >> "$CTLOOM_WORK/kept.txt"
git init -q "$CTLOOM_WORK/vendor/lib"; echo v > "$CTLOOM_WORK/vendor/lib/f.c"
git -C "$CTLOOM_WORK/vendor/lib" add -A
git -C "$CTLOOM_WORK/vendor/lib" -c user.name=m -c user.email=m@e commit -qm v""")
    assert "+two" in patch and meta["nested"] == ["vendor/lib"] and meta["unadded"] == []
    assert "repositories of their own" in stderr and "vendor/lib" in stderr


def test_a_submodule_gitmodules_ignores_is_in_the_patch_as_its_commit(tmp_path):
    """.gitmodules setting a submodule's ignore to all changes neither the patch, its
    count of files nor the list of repositories of their own."""
    link = "1" * 40
    patch, meta, _, status = _write_patch_with(tmp_path, f"""
printf '[submodule "sub"]\\n\\tpath = sub\\n\\turl = ./sub\\n\\tignore = all\\n' \\
    > "$CTLOOM_WORK/.gitmodules"
mkdir "$CTLOOM_WORK/sub"
git -C "$CTLOOM_WORK" update-index --add --cacheinfo 160000,{link},sub""")
    assert status == 0 and f"+Subproject commit {link}" in patch
    assert meta["files"] == 2 and meta["nested"] == ["sub"]
    assert meta["unseen"] == []


@pytest.mark.parametrize("write, unseen_list", [
    ('echo x > "$CTLOOM_WORK/sub/new.py"', ["sub"]),
    ('mkdir -p "$CTLOOM_WORK/süb x/deep"; echo x > "$CTLOOM_WORK/süb x/deep/new.py"',
     ['"s\\303\\274b x"']),
    ('mkdir -p "$CTLOOM_WORK/sub/one/two"', []),
    ("", []),
    ('git -C "$CTLOOM_WORK/sub" init -q', []),
    ('git -C "$CTLOOM_WORK/sub" init -q; echo x > "$CTLOOM_WORK/sub/new.py"', ["sub"]),
    ('git -C "$CTLOOM_WORK/sub" init -q; echo x > "$CTLOOM_WORK/sub/new.py"\n'
     'git -C "$CTLOOM_WORK/sub" add -A\n'
     'git -C "$CTLOOM_WORK/sub" -c user.name=m -c user.email=m@e commit -qm x', []),
    ('git -C "$CTLOOM_WORK/sub" init -q; echo x > "$CTLOOM_WORK/sub/new.py"\n'
     'git -C "$CTLOOM_WORK/sub" add -A\n'
     'git -C "$CTLOOM_WORK/sub" -c user.name=m -c user.email=m@e commit -qm x\n'
     'echo y > "$CTLOOM_WORK/sub/more.py"', ["sub"])])
def test_a_file_written_into_a_submodule_with_no_repository_is_named_in_the_summary(
        tmp_path, write, unseen_list):
    """A submodule directory of the clone that holds a file and no .git, or a .git
    whose status is not clean, is named under "unseen"; one that holds directories
    alone, or a repository whose commit holds what it does, is not."""
    src = tmp_path / "src"
    for name in ("sub", "süb x"):
        (src / name).mkdir(parents=True)
    _git(src, "init", "--quiet", "-b", "main")
    for name in ("sub", "süb x"):
        _git(src, "update-index", "--add", "--cacheinfo", f"160000,{'1' * 40},{name}")
    _git(src, "-c", "user.name=a", "-c", "user.email=a@b", "commit", "--quiet", "-m",
         "links")
    patch, meta, stderr, status = _write_patch_with(
        tmp_path, f'echo two >> "$CTLOOM_WORK/kept.txt"\n{write}')
    assert status == 0 and "+two" in patch and "new.py" not in patch
    assert meta["unseen"] == unseen_list and meta["nested"] == []
    assert ("git shows these submodule directories as their commit alone" in stderr) \
        == bool(unseen_list)


def test_what_the_run_committed_in_its_clone_is_in_the_patch(tmp_path):
    """The patch is written against the commit the run started at, wherever HEAD is
    when the run ends."""
    patch, meta, stderr, _ = _write_patch_with(tmp_path, """
echo two >> "$CTLOOM_WORK/kept.txt"
git -C "$CTLOOM_WORK" -c user.name=m -c user.email=m@e commit -qam "by the run"
echo three >> "$CTLOOM_WORK/kept.txt"
echo new > "$CTLOOM_WORK/new.txt"
""")
    assert "+two" in patch and "+three" in patch and "new.txt" in patch
    assert meta["files"] == 2 and len(meta["moved_head"]) == 40
    assert meta["moved_head"] != meta["base"] and "HEAD is at" in stderr


def test_a_merging_run_s_patch_is_written_against_the_head_it_left(tmp_path):
    """integrate decides the commit its union is written against and leaves HEAD
    there, so what lies between the start and that commit is not in its patch."""
    patch, meta, _, _ = _write_patch_with(tmp_path, """
echo onto >> "$CTLOOM_WORK/kept.txt"
git -C "$CTLOOM_WORK" -c user.name=m -c user.email=m@e commit -qam "what the tree holds"
echo union > "$CTLOOM_WORK/union.txt"
""", command="integrate")
    assert "union.txt" in patch and "+onto" not in patch
    assert meta["files"] == 1 and "moved_head" not in meta


@pytest.mark.parametrize("tool_status, status", [(0, 1), (3, 3)])
def test_a_patch_that_cannot_be_written_leaves_no_summary_and_a_failing_status(
        tmp_path, tool_status, status):
    _entrypoint_sh(tmp_path, f'pwd > "$CTLOOM_OUT/started-from.txt"\n'
                             f'echo two >> "$CTLOOM_WORK/kept.txt"\nmkdir "$CTLOOM_PATCH"\n'
                             f'exit {tool_status}')
    out = tmp_path / "out"
    assert int((out / "exit-status.txt").read_text()) == status
    assert not (out / "patch.diff.meta.json").exists()
    assert not (out / "patch.diff").is_file()
    assert "git could not write the patch, so no patch is left" in \
        (out / "stderr.txt").read_text(encoding="utf-8")


def test_the_formatting_command_runs_after_the_tool_and_before_the_patch(tmp_path):
    """The stand-in appends to a file when it is asked to format, so the patch holds
    what the formatting left."""
    script = """
pwd > "$CTLOOM_OUT/started-from.txt"
echo "$4" >> "$CTLOOM_OUT/asked.txt"
if [ "$4" = "format" ]; then echo formatted >> "$CTLOOM_WORK/kept.txt"; exit 7; fi
echo changed >> "$CTLOOM_WORK/kept.txt"
"""
    _entrypoint_sh(tmp_path, script, {"CTLOOM_FORMAT": "1"})
    out = tmp_path / "out"
    assert (out / "asked.txt").read_text().split() == ["edit", "format"]
    patch = (out / "patch.diff").read_text()
    assert "+changed" in patch and "+formatted" in patch
    # The run's exit status is the tool's, not the formatting command's
    assert json.loads((out / "patch.diff.meta.json").read_text())["exit_status"] == 0


def test_no_formatting_step_is_made_without_the_setting(tmp_path):
    script = 'pwd > "$CTLOOM_OUT/started-from.txt"; echo "$4" >> "$CTLOOM_OUT/asked.txt"'
    _entrypoint_sh(tmp_path, script, {"CTLOOM_FORMAT": "0"})
    assert (tmp_path / "out" / "asked.txt").read_text().split() == ["edit"]


def test_the_tool_is_started_in_the_container_off_its_working_directory(tmp_path):
    """Every start of the tool the two scripts make: the rebuild, the run, the
    formatting, and the filling."""
    script = 'pwd > "$CTLOOM_OUT/started-from.txt"; echo "$*" >> "$CTLOOM_OUT/asked.txt"'
    _entrypoint_sh(tmp_path, script, {"CTLOOM_FORMAT": "1", "CTLOOM_REBUILD_INDEX": "1"})
    work, out = tmp_path / "work" / "repo", tmp_path / "out"
    assert (out / "asked.txt").read_text().splitlines() == [
        f"-P -m codetwine_loom build --project {work} --output-dir {out}",
        "-P -m codetwine_loom edit",
        f"-P -m codetwine_loom format --project {work} --output-dir {out}"]
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
           "CTLOOM_SRC": str(tmp_path / "src"), "CTLOOM_WORK": str(tmp_path / "warm" / "repo"),
           "CTLOOM_BASE": _git(tmp_path / "src", "rev-parse", "HEAD").strip(),
           "CTLOOM_OUT": str(out), "CTLOOM_RUNDIR": str(tmp_path / "rundir")}
    subprocess.run(["sh", str(_WARM_SH), "run-tests"], env=env, check=True,
                   capture_output=True, text=True)
    assert (out / "asked.txt").read_text().splitlines()[-1] == \
        "-P -m codetwine_loom run-tests"


def test_the_run_is_not_started_from_inside_the_clone(tmp_path):
    """The instructions tell the model a relative open() misses the repository."""
    started_from = _entrypoint_sh(tmp_path, 'pwd > "$CTLOOM_OUT/started-from.txt"')
    assert started_from == str(tmp_path / "rundir")
    assert started_from != str(tmp_path / "work" / "repo")


def test_a_relative_write_lands_outside_the_clone(tmp_path):
    """What the model writes with open("app.py", "w") must not reach the repository."""
    _entrypoint_sh(tmp_path, 'pwd > "$CTLOOM_OUT/started-from.txt"; echo x > kept.txt')
    assert (tmp_path / "rundir" / "kept.txt").exists()
    assert (tmp_path / "work" / "repo" / "kept.txt").read_text() == "one\n"


def test_the_build_output_of_the_primary_language_is_not_in_the_patch(tmp_path):
    work = tmp_path / "work" / "repo"
    _entrypoint_sh(tmp_path, f"""pwd > "$CTLOOM_OUT/started-from.txt"
mkdir -p {work}/src/App/bin/Debug {work}/src/App/obj {work}/robj
echo x > {work}/src/App/bin/Debug/App.dll
echo x > {work}/src/App/obj/project.assets.json
echo x > {work}/robj/kept.cs
echo x > {work}/src/App/New.cs""", {"CTLOOM_BUILD_OUTPUT": "bin/\nobj/"})
    patch = (tmp_path / "out" / "patch.diff").read_text(encoding="utf-8")
    assert "src/App/New.cs" in patch and "robj/kept.cs" in patch
    assert "/bin/" not in patch and "/obj/" not in patch
    # The repository's own ignore file is left as it was
    assert not (work / ".gitignore").exists()


def test_without_build_output_every_new_file_is_in_the_patch(tmp_path):
    work = tmp_path / "work" / "repo"
    _entrypoint_sh(tmp_path, f"""pwd > "$CTLOOM_OUT/started-from.txt"
mkdir -p {work}/bin && echo x > {work}/bin/run.sh""", {"CTLOOM_BUILD_OUTPUT": ""})
    assert "bin/run.sh" in (tmp_path / "out" / "patch.diff").read_text(encoding="utf-8")


def test_the_build_output_named_leaves_the_entries_placed_in_directories_out(tmp_path):
    out = tmp_path / "out"
    _index_in(out, "class_declaration", "repo/src/a_ts/a.ts")
    output_tuple = sandbox.build_output_tuple_of(str(out), "repo")
    assert "node_modules/" in output_tuple
    assert not any(pattern.startswith("/") for pattern in output_tuple)


def test_the_build_output_named_is_the_index_s_primary_language_s(tmp_path):
    out = tmp_path / "out"
    assert sandbox.build_output_tuple_of(str(out), "repo") == ()
    _index_in(out, "class_declaration", "repo/src/A_cs/A.cs")
    assert sandbox.build_output_tuple_of(str(out), "repo") == ("bin/", "obj/")
    assert "CTLOOM_BUILD_OUTPUT=bin/\nobj/" in sandbox.build_run_argv(
        "/usr/bin/podman", "an-image", sandbox.MountLayout(
            src="/r", out="/o", scratch="/s", cache="/c"), "abc", "repo", ["edit"],
        build_output_tuple=sandbox.build_output_tuple_of(str(out), "repo"))
    assert "CTLOOM_BUILD_OUTPUT=" in sandbox.build_run_argv(
        "/usr/bin/podman", "an-image", sandbox.MountLayout(
            src="/r", out=str(out), scratch="/s", cache="/c"), "abc", "repo", ["edit"])


def test_the_container_is_handed_the_primary_language_s_build_output_alone(tmp_path):
    """The other languages' is left out by the run inside the container."""
    out = tmp_path / "out"
    files = {"repo/src/A_cs/A.cs": {"summary": "", "defs": []},
             "repo/web/view_js/view.js": {"summary": "", "defs": []},
             "repo/tools/gen_py/gen.py": {"summary": "", "defs": []},
             "repo/svc/Cargo_toml/Cargo.toml": {"summary": "", "defs": []},
             "repo/svc/src/lib_rs/lib.rs": {"summary": "", "defs": []}}
    defs = {"repo/src/A_cs/A.cs::A": {
        "name": "A", "type": "class_declaration", "file": "repo/src/A_cs/A.cs",
        "lines": [1, 2], "signature": "class A", "doc": "", "calls": [], "called_by": []}}
    write_index(str(out / "repo" / store.INDEX_FILE), defs, files, project="repo")
    assert sandbox.build_output_tuple_of(str(out), "repo") == ("bin/", "obj/")


def test_a_language_without_build_output_names_none(tmp_path):
    _index_in(tmp_path / "out", "paragraph_header", "repo/src/MAIN_cbl/MAIN.cbl")
    assert sandbox.build_output_tuple_of(str(tmp_path / "out"), "repo") == ()


def test_a_file_pattern_and_a_root_pattern_are_left_out_of_the_patch(tmp_path):
    work = tmp_path / "work" / "repo"
    _entrypoint_sh(tmp_path, f"""pwd > "$CTLOOM_OUT/started-from.txt"
mkdir -p {work}/Testing/Temporary {work}/tests/Testing/Temporary {work}/sub
echo x > {work}/Testing/Temporary/LastTest.log
echo x > {work}/tests/Testing/Temporary/kept.log
echo x > {work}/uv.lock
echo x > {work}/sub/uv.lock""", {"CTLOOM_BUILD_OUTPUT": "/Testing/Temporary/\nuv.lock"})
    patch = (tmp_path / "out" / "patch.diff").read_text(encoding="utf-8")
    assert "tests/Testing/Temporary/kept.log" in patch
    assert "LastTest.log" not in patch and "uv.lock" not in patch


# == A dry run takes nothing in ===============================

@pytest.mark.parametrize("kwargs", [{}, {"land": True}])
def test_a_dry_run_leaves_the_repository_alone(tmp_path, monkeypatch, caplog, kwargs):
    """The patch is the clone's git state, so a write no tool made would still land."""
    caplog.set_level("INFO", logger="codetwine_loom.sandbox")
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    written = {}

    def plant(argv, **kw):
        """Stand in for a container whose run wrote without going through a tool."""
        patch = _told_to_write(argv)
        written["patch"] = patch
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(_patch_from(repo, lambda c: (c / "kept.txt").write_text("two\n")),
                         encoding="utf-8")
        patch.with_suffix(f"{patch.suffix}.meta.json").write_text(
            json.dumps({"files": 1, "exit_status": 0}), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    real_run = subprocess.run
    monkeypatch.setattr(sandbox, "pick_engine", lambda name="": "/usr/bin/true")
    monkeypatch.setattr(sandbox.subprocess, "run",
                        lambda argv, **kw: real_run(argv, **kw) if _is_git(argv)
                        else plant(argv, **kw))
    before = _git(repo, "rev-parse", "HEAD").strip()
    result = sandbox.run(str(repo), str(out), "x", dry_run=True, **kwargs)
    assert "nothing is taken in" in caplog.text
    assert result["patch"] == str(written["patch"])   # the patch is still written out
    assert result["branch"] is None and result["commit"] is None
    assert result["land"] is None
    assert _git(repo, "rev-parse", "HEAD").strip() == before
    assert _git(repo, "branch", "--list", "codetwine-loom/sandbox/*").strip() == ""
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\n"


def test_this_machine_records_each_run_it_starts_and_the_patch_it_left_to_pick_up(
        tmp_path, monkeypatch):
    """A dry run leaves its patch to pick up by hand; a run whose patch a commit holds
    leaves none. Both are runs this machine started for their agents."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    text = _patch_from(repo, lambda c: (c / "kept.txt").write_text("two\n"))

    def write(argv):
        patch = _told_to_write(argv)
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(text, encoding="utf-8")
        patch.with_suffix(f"{patch.suffix}.meta.json").write_text(
            json.dumps({"files": 1, "exit_status": 0}), encoding="utf-8")

    run_list = [f"2026010{n}-000000-1-{os.getpid()}" for n in (1, 2)]
    for run_id, dry_run in zip(run_list, (True, False)):
        monkeypatch.setattr(sandbox, "new_run_id", lambda run_id=run_id: run_id)
        result, _ = _run_without_a_container(tmp_path, monkeypatch, on_start=write,
                                             project=str(repo), output_dir=str(out),
                                             request="x", dry_run=dry_run)
        assert bool(result["commit"]) is not dry_run
    remain_dict = sandbox._run_remain_dict(str(repo), str(out))
    dry, took = run_list
    assert remain_dict[dry]["runs"] == [dry] and remain_dict[took]["runs"] == [took]
    assert remain_dict[dry]["patches"] == [sandbox.run_file_layout(
        str(out), dry, "repo", dry).patch]
    assert remain_dict[took]["patches"] == [] and remain_dict[dry]["answers"] == []
    assert pathlib.Path(sandbox.host_dir(str(out), dry), "repo", "sandbox",
                        f"run-{dry}.txt").read_text(encoding="utf-8") == "patch"


# == Several runs in one output directory =====================

def _two_runs(tmp_path, monkeypatch, held):
    """Start one run, and start a second one while the first one's clone is there.

    The first run's container is stood in for by something that leaves its clone where
    it is and calls back, which is where the second run is started from: what the two
    then have of their own is what a run going beside another one keeps.

    Args:
        tmp_path: The test's directory.
        monkeypatch: pytest's own fixture.
        held: Called with the first run's clone while the second run is going.

    Returns:
        (the first run's return value, the second run's).
    """
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    monkeypatch.setattr(sandbox, "index_describe", lambda *a: True)
    second = {}

    def alongside(argv):
        clone = _clone_from(argv)
        _leave_a_patch(argv, repo)
        monkeypatch.setattr(sandbox, "new_run_id",
                            lambda: f"20260102-000000-{os.getpid()}")
        second["result"], second["argv"] = _run_without_a_container(
            tmp_path, monkeypatch, on_start=lambda a: _leave_a_patch(a, repo),
            project=str(repo), output_dir=str(out), request="the second one",
            dry_run=True)
        held(clone)

    monkeypatch.setattr(sandbox, "new_run_id",
                        lambda: f"20260101-000000-{os.getpid()}")
    first, _ = _run_without_a_container(tmp_path, monkeypatch, on_start=alongside,
                                        project=str(repo), output_dir=str(out),
                                        request="the first one", dry_run=True)
    return first, second["result"]


def _clone_from(argv):
    """Return the clone a container given argv would be writing into."""
    mount = next(a for a in argv if a.endswith(":/work"))
    return pathlib.Path(mount.split(":")[0]).parent


def _leave_a_patch(argv, repo):
    """Write the patch the container given argv was told to write."""
    patch = _told_to_write(argv)
    patch.parent.mkdir(parents=True, exist_ok=True)
    patch.write_text(f"the patch of {patch.name}\n", encoding="utf-8")
    patch.with_suffix(f"{patch.suffix}.meta.json").write_text(
        json.dumps({"base": gitcmd.require_head(str(repo)), "exit_status": 0,
                    "files": 1}), encoding="utf-8")


def test_two_runs_in_one_output_directory_keep_their_own_patch(tmp_path, monkeypatch):
    """Each run's patch has a name of its own, so the two do not overwrite one another."""
    first, second = _two_runs(tmp_path, monkeypatch, held=lambda clone: None)
    assert first["patch"] != second["patch"]
    for result in (first, second):
        assert pathlib.Path(result["patch"]).name in \
            pathlib.Path(result["patch"]).read_text(encoding="utf-8")


def test_two_runs_in_one_output_directory_are_agents_of_their_own(tmp_path,
                                                                  monkeypatch):
    """Nobody has to remember to pass a different --output-dir: a run with no branch
    of its own is an agent of its own, and the tool makes it a place."""
    first, second = _two_runs(tmp_path, monkeypatch, held=lambda clone: None)
    mine, theirs = (pathlib.Path(r["patch"]).parents[2] for r in (first, second))
    assert mine != theirs
    assert mine.parent.name == "agents" and theirs.parent.name == "agents"
    assert mine.is_dir() and theirs.is_dir()


def test_a_branch_a_run_that_ended_left_on_its_start_is_taken_away(tmp_path,
                                                                   monkeypatch):
    """The next run to make one sweeps it; one of a run still going stays."""
    repo = _repo(tmp_path / "repo")
    monkeypatch.setattr(sandbox, "pick_engine", lambda name="": "")
    ended = "20260101-000000-1-" + _a_run_that_ended().rsplit("-", 1)[-1]
    going = f"20260101-000000-2-{os.getpid()}"
    for run_id in (ended, going):
        _git(repo, "branch", f"codetwine-loom/run/{run_id}/tree")
    _git(repo, "branch", f"codetwine-loom/run/{ended}/1")
    with sandbox.tree_branch(str(repo), f"20260102-000000-1-{os.getpid()}"):
        branches = _git(repo, "branch", "--list", "codetwine-loom/run/*").split()
    assert f"codetwine-loom/run/{ended}/tree" not in branches
    assert f"codetwine-loom/run/{going}/tree" in branches \
        and f"codetwine-loom/run/{ended}/1" in branches


def test_a_change_that_did_not_reach_the_tree_is_a_status_of_its_own():
    """It is not what decides whether a dispatching run is made again."""
    assert sandbox.land_status({"land": None, "land_why": "it does not apply"}) == 1
    assert sandbox.land_status({"land": {"files": 1}, "land_why": ""}) == 0
    assert sandbox.land_status(None) == 0 and sandbox.land_status({}) == 0


def test_a_run_starting_beside_another_leaves_that_one_s_clone_alone(tmp_path,
                                                                     monkeypatch):
    """make_scratch() makes a directory of the second run's own and leaves the running
    one's clone where it is."""
    seen = {}
    _two_runs(tmp_path, monkeypatch,
              held=lambda clone: seen.update(there=(clone / "work").is_dir()))
    assert seen["there"] is True


# == Which runs are going, and which were killed ==============

def _host_out(tmp_path, agent="a1"):
    """Return this machine's side of one agent under tmp_path / "out"
    (sandbox.host_dir)."""
    return pathlib.Path(sandbox.host_dir(str(tmp_path / "out"), agent))


def _clone_of(project_out, run_id, refill=False):
    """Make the clone one run would be holding and return its directory.

    Args:
        project_out: <this machine's side of an agent>/<repository name>.
        run_id: The run.
        refill: Whether the clone is a refill's.
    """
    prefix = "scratch-warm-" if refill else "scratch-"
    clone = pathlib.Path(project_out) / "sandbox" / f"{prefix}{run_id}" / "work"
    clone.mkdir(parents=True)
    return clone.parent


def test_a_run_that_is_going_is_named_as_one(tmp_path):
    project_out = _host_out(tmp_path) / "repo"
    _clone_of(project_out, f"20260101-000000-{os.getpid()}")
    assert sandbox.other_live_run_list(str(_host_out(tmp_path)), "repo", "mine") == \
        [f"20260101-000000-{os.getpid()}"]


def test_the_clone_of_a_run_that_ended_is_not_a_run_that_is_going(tmp_path):
    """Otherwise a machine that was rebooted mid-run would refuse every run after."""
    project_out = _host_out(tmp_path) / "repo"
    _clone_of(project_out, _a_run_that_ended())
    assert sandbox.other_live_run_list(str(_host_out(tmp_path)), "repo", "mine") == []


def test_a_refill_is_not_counted_as_a_run_of_its_own(tmp_path):
    """It belongs to the run that asked for it, which is already counted."""
    project_out = _host_out(tmp_path) / "repo"
    _clone_of(project_out, f"20260101-000000-{os.getpid()}", refill=True)
    assert sandbox.other_live_run_list(str(_host_out(tmp_path)), "repo", "mine") == []


def test_this_run_is_not_one_of_the_others(tmp_path):
    project_out = _host_out(tmp_path) / "repo"
    _clone_of(project_out, "mine")
    assert sandbox.other_live_run_list(str(_host_out(tmp_path)), "repo", "mine") == []


def _going(tmp_path, monkeypatch, engine="/bin/true"):
    """Ask live_run_list() over tmp_path/out with the engine found where it is said."""
    if engine:
        monkeypatch.setattr(sandbox, "pick_engine", lambda name="": engine)
    else:
        def none(name=""):
            raise RuntimeError("none of podman, docker is on PATH")
        monkeypatch.setattr(sandbox, "pick_engine", none)
    return sandbox.live_run_list(str(tmp_path / "out"), "repo")


def test_runs_going_looks_under_every_agents_directory_on_this_machines_side(
        tmp_path, monkeypatch):
    """A run's clone is under what this machine keeps for its agent."""
    mine = f"20260101-000000-1-{os.getpid()}"
    _clone_of(_host_out(tmp_path) / "repo", mine)
    _clone_of(_host_out(tmp_path, "codetwine-loom-one") / "repo", mine)
    assert _going(tmp_path, monkeypatch) == [f"a1/{mine}", f"codetwine-loom-one/{mine}"]


def test_runs_going_leaves_out_refills_and_runs_that_ended(tmp_path, monkeypatch):
    agent_out = _host_out(tmp_path, "codetwine-loom-one") / "repo"
    _clone_of(agent_out, f"20260101-000000-1-{os.getpid()}", refill=True)
    _clone_of(agent_out, _a_run_that_ended())
    assert _going(tmp_path, monkeypatch) == []


def test_runs_going_asks_the_other_engines_when_the_set_one_is_not_on_path(tmp_path,
                                                                           monkeypatch):
    mine = f"20260101-000000-1-{os.getpid()}"
    _clone_of(_host_out(tmp_path) / "repo", mine)
    monkeypatch.setattr(sandbox, "_other_engine_path_list",
                        lambda engine_path: ["/usr/bin/docker"])
    assert _going(tmp_path, monkeypatch, engine="") == [f"a1/{mine}"]


def test_runs_going_is_empty_where_no_engine_is_on_path(tmp_path, monkeypatch):
    """A machine with no engine started no container, and a container has no engine."""
    _clone_of(_host_out(tmp_path, "codetwine-loom-one") / "repo",
              f"20260101-000000-1-{os.getpid()}")
    assert _going(tmp_path, monkeypatch, engine="") == []


def test_the_clone_of_a_run_that_ended_is_swept_up(tmp_path):
    """Nothing else would: a clone is named after the run, so the run that follows
    cannot tell it apart from one that is being written."""
    project_out = _host_out(tmp_path) / "repo"
    ended = _clone_of(project_out, _a_run_that_ended())
    going = _clone_of(project_out, f"20260101-000000-{os.getpid()}")
    sandbox.remove_dead_clone(str(tmp_path / "out"), "repo")
    assert not ended.exists() and going.exists()


def test_a_refill_that_ended_is_swept_up_as_well(tmp_path):
    project_out = _host_out(tmp_path) / "repo"
    ended = _clone_of(project_out, _a_run_that_ended(), refill=True)
    sandbox.remove_dead_clone(str(tmp_path / "out"), "repo")
    assert not ended.exists()


# == The index a run starts over =============================

def test_a_run_starting_where_the_index_describes_only_reads_it(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    monkeypatch.setattr(sandbox, "index_describe", lambda *a: True)
    _, argv = _run_without_a_container(tmp_path, monkeypatch, project=str(repo),
                                       output_dir=str(tmp_path / "out"), request="x",
                                       dry_run=True)
    assert "CTLOOM_REBUILD_INDEX=0" in argv


def test_a_run_starting_elsewhere_has_the_container_rebuild_the_index(tmp_path,
                                                                     monkeypatch):
    repo = _repo(tmp_path / "repo")
    monkeypatch.setattr(sandbox, "index_describe", lambda *a: False)
    _, argv = _run_without_a_container(tmp_path, monkeypatch, project=str(repo),
                                       output_dir=str(tmp_path / "out"), request="x",
                                       dry_run=True)
    assert "CTLOOM_REBUILD_INDEX=1" in argv


def test_a_directory_is_swept_under_the_lock_its_clone_directories_are_made_under(
        tmp_path, monkeypatch):
    """A clone being made is never swept as one a run left behind: the clone left in
    another agent's directory is taken away under that directory's lock, this run's
    clone directory is made under its own, and no two of those locks are held at
    once, so two agents starting together never wait for one another."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    dead_clone = _clone_of(out / "agents" / ".host" / "codetwine-loom-old" / "repo",
                           _a_run_that_ended())
    event_list = []
    real_held = sandbox.filelock.hold_lock

    @contextlib.contextmanager
    def noted(path, what, wait=True):
        event_list.append(("enter", path))
        try:
            with real_held(path, what, wait=wait):
                yield
        finally:
            event_list.append(("exit", path))

    monkeypatch.setattr(sandbox.filelock, "hold_lock", noted)
    real_remove = sandbox.remove_scratch

    def note_remove(scratch):
        event_list.append(("remove", scratch))
        return real_remove(scratch)

    monkeypatch.setattr(sandbox, "remove_scratch", note_remove)
    real_scratch = sandbox.make_scratch

    def made(scratch):
        event_list.append(("scratch", scratch))
        return real_scratch(scratch)

    monkeypatch.setattr(sandbox, "make_scratch", made)
    _run_without_a_container(tmp_path, monkeypatch, project=str(repo),
                             output_dir=str(out), request="x", dry_run=True)
    assert not dead_clone.exists()

    def inside(path):
        return event_list[event_list.index(("enter", path)) + 1:
                          event_list.index(("exit", path))]

    assert ("remove", str(dead_clone)) in inside(str(dead_clone.parent))
    scratch = next(p for e, p in event_list if e == "scratch")
    assert ("scratch", scratch) in inside(os.path.dirname(scratch))
    # The run directories' locks are taken one at a time
    depth = 0
    for event, path in event_list:
        if event in ("enter", "exit") and path.endswith(os.sep + "sandbox"):
            depth += 1 if event == "enter" else -1
            assert depth <= 1


def test_the_sweep_looks_under_every_agents_copy(tmp_path):
    """Each run is an agent with a directory of its own, so what a killed run left is
    in a directory no later run makes its clone in."""
    out = tmp_path / "out"
    dead_clone = _clone_of(out / "agents" / ".host" / "codetwine-loom-old" / "repo",
                           _a_run_that_ended())
    live_clone = _clone_of(out / "agents" / ".host" / "codetwine-loom-now" / "repo",
                      f"20260101-000000-{os.getpid()}")
    top_clone = _clone_of(out / "agents" / ".host" / "codetwine-loom-top" / "repo",
                          _a_run_that_ended())
    sandbox.remove_dead_clone(str(out), "repo")
    assert not dead_clone.exists() and not top_clone.exists() and live_clone.exists()


def test_an_engine_whose_ps_fails_keeps_the_clone_and_says_why(tmp_path, monkeypatch,
                                                                caplog):
    """The run counts as going, as one that cannot be asked about does, and the log
    names what the engine said."""
    clone = _clone_of(_host_out(tmp_path) / "repo", _a_run_that_ended())

    def ps_fail(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 125, "", "Error: cannot connect")

    monkeypatch.setattr(sandbox.subprocess, "run", ps_fail)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        sandbox.remove_dead_clone(str(tmp_path / "out"), "repo", "/usr/bin/podman")
    assert clone.exists()
    assert any("exit status 125" in r.getMessage()
               and "Error: cannot connect" in r.getMessage() for r in caplog.records)


# == Where the package cache goes =============================

def test_the_cache_is_under_the_output_directory_unless_it_is_told_otherwise():
    assert sandbox.cache_dir("/host/out", "repo") == "/host/out/agents/.repo/repo/cache"


def test_the_setting_puts_one_cache_where_several_output_directories_read_it(
        monkeypatch):
    """Filling it needs a network, so it is what agents are meant to share."""
    monkeypatch.setattr(config, "SANDBOX_CACHE_DIR", "/somewhere/else")
    assert sandbox.cache_dir("/host/out", "repo") == "/somewhere/else"


def test_the_run_and_the_filling_read_the_cache_from_the_same_place(tmp_path,
                                                                    monkeypatch):
    """A run reading one cache while warm fills another is a run with no cache."""
    monkeypatch.setattr(config, "SANDBOX_CACHE_DIR", str(tmp_path / "shared"))
    repo = _repo(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    sandbox.run(str(repo), str(tmp_path / "out"), "x")
    sandbox.warm(str(repo), str(tmp_path / "out2"))
    assert [f"{tmp_path}/shared:/cache" in argv for argv in seen] == [True, True]


@pytest.mark.parametrize("output_dir, message", [
    ("repo/out", "OUTPUT_DIR is inside PROJECT_DIR"), ("", "OUTPUT_DIR holds PROJECT_DIR")])
def test_a_filling_is_refused_an_output_directory_inside_the_repository_or_holding_it(
        tmp_path, monkeypatch, output_dir, message):
    repo = _repo(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    before_list = sorted(str(path) for path in tmp_path.rglob("*"))
    with pytest.raises(RuntimeError, match=message):
        sandbox.warm(str(repo), str(tmp_path / output_dir))
    assert seen == [] and sorted(str(path) for path in tmp_path.rglob("*")) == before_list


def test_the_cache_this_machine_names_is_not_forwarded_to_the_container(tmp_path,
                                                                        monkeypatch):
    """The container reads it at /cache, wherever this machine keeps it."""
    monkeypatch.setattr(config, "SANDBOX_CACHE_DIR", str(tmp_path / "shared"))
    assert "SANDBOX_CACHE_DIR" not in " ".join(sandbox.env_forward_argv())


def test_the_broker_is_told_which_run_it_serves(tmp_path, monkeypatch):
    """A refill takes the dependency files out of the running change's clone, and an
    output directory holds one clone per run going in it."""
    seen = {}

    class _Listening:
        """A broker that is up, as start_broker() asks about one."""

        def poll(self):
            return None

    def popen(argv, **kwargs):
        seen["argv"] = argv
        pathlib.Path(tmp_path / "llm.sock").touch()
        return _Listening()

    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    _START_BROKER(str(tmp_path), "/host/repo", "/host/out", "20260101-000000-1")
    assert seen["argv"][-4:] == ["--output-dir", "/host/out",
                                 "--run", "20260101-000000-1"]


def test_the_broker_is_handed_no_terminal_and_listens_where_it_is_told(tmp_path,
                                                                      monkeypatch):
    """A broker started here has nobody to ask, and an address in LLM_BROKER is not
    where it listens."""
    seen = {}

    class _Listening:
        def poll(self):
            return None

    def popen(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        pathlib.Path(tmp_path / "llm.sock").touch()
        return _Listening()

    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    monkeypatch.setenv("LLM_BROKER", "tcp://127.0.0.1:9")
    _START_BROKER(str(tmp_path))
    assert seen["kwargs"]["stdin"] == subprocess.DEVNULL
    assert seen["argv"][:-2] == config.own_argv(["broker"])
    assert seen["argv"][-2:] == ["--endpoint", sandbox.socket_url(str(tmp_path))]
    # It is given this process's environment, the credentials included, and reads no
    # .env file of its own
    assert seen["kwargs"]["env"][config.NO_ENV_FILE_ENV] == "1"
    assert seen["kwargs"]["env"]["LLM_BROKER"] == "tcp://127.0.0.1:9"


def test_the_broker_is_told_the_label_of_the_agent_it_serves(tmp_path, monkeypatch):
    seen_list = []

    class _Listening:
        def poll(self):
            return None

    def popen(argv, **kwargs):
        seen_list.append(argv)
        pathlib.Path(tmp_path / "llm.sock").touch()
        return _Listening()

    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    with progress.label("[task 2/5] "):
        _START_BROKER(str(tmp_path))
    _START_BROKER(str(tmp_path))
    assert seen_list[0][-2:] == ["--label", "[task 2/5] "]
    assert "--label" not in seen_list[1]


def test_a_broker_started_by_hand_serves_no_run_in_particular(tmp_path, monkeypatch):
    seen = {}

    class _Listening:
        def poll(self):
            return None

    def popen(argv, **kwargs):
        seen["argv"] = argv
        pathlib.Path(tmp_path / "llm.sock").touch()
        return _Listening()

    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    _START_BROKER(str(tmp_path))
    assert "--run" not in seen["argv"]


# == Telling a run that is going from one that was killed =====

def test_the_container_is_named_after_the_run():
    """The name is what says whose container it is, both to the sweep below and to
    whoever runs podman ps while several agents are going."""
    argv = _argv(run_id="20260101-000000-1")
    assert argv[argv.index("--name") + 1] == "ctloom-20260101-000000-1"


def test_a_container_of_no_run_answers_to_no_name():
    assert "--name" not in _argv()


def test_the_cache_filling_container_answers_to_a_name_but_writes_no_patch():
    """The name is what lets the sweep ask the engine about a filling whose host
    process was killed; the patch stays the change entry point's alone."""
    argv = sandbox.build_run_argv(
        "/usr/bin/podman", "an-image",
        sandbox.MountLayout(src="/host/repo", out="/host/out", scratch="/host/scratch",
                       cache="/host/cache"),
        "abc123", "repo", ["run-tests"], run_id="20260101-000000-1", network=True,
        entrypoint="/usr/local/bin/ctloom-warm")
    assert argv[argv.index("--name") + 1] == "ctloom-20260101-000000-1"
    assert "CTLOOM_PATCH" not in " ".join(argv)


def _asked_the_engine(monkeypatch, running):
    """Stand in for the engine's "ps", saying whether it names a container.

    Args:
        monkeypatch: pytest's fixture.
        running: True to answer as an engine that has one.

    Returns:
        The list each argv is appended to.
    """
    seen = []
    real_run = subprocess.run

    def run(argv, **kwargs):
        if _is_git(argv):
            return real_run(argv, **kwargs)
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "abc123\n" if running else "")

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    return seen


def test_a_clone_whose_container_is_still_there_is_left_alone(tmp_path, monkeypatch):
    """Killing the run's process does not stop its container, and that container is
    still writing the clone."""
    project_out = _host_out(tmp_path) / "repo"
    ended = _a_run_that_ended()
    clone = _clone_of(project_out, ended)
    seen = _asked_the_engine(monkeypatch, running=True)
    sandbox.remove_dead_clone(str(tmp_path / "out"), "repo", "/usr/bin/podman")
    assert clone.exists()
    assert seen[0][:3] == ["/usr/bin/podman", "ps", "--quiet"]
    assert f"name=ctloom-{ended}" in seen[0]
    assert sandbox.other_live_run_list(str(_host_out(tmp_path)), "repo", "mine",
                                     "/usr/bin/podman") == [ended]


def test_a_clone_whose_container_has_gone_too_is_swept_up(tmp_path, monkeypatch):
    project_out = _host_out(tmp_path) / "repo"
    clone = _clone_of(project_out, _a_run_that_ended())
    _asked_the_engine(monkeypatch, running=False)
    sandbox.remove_dead_clone(str(tmp_path / "out"), "repo", "/usr/bin/podman")
    assert not clone.exists()


def _engine_answer(monkeypatch, answer_dict):
    """Stand in for every engine's "ps": answer_dict maps an engine to True (has the
    container), False (has none) or an int (the exit status of a ps that failed)."""
    def run(argv, **kwargs):
        answer = answer_dict[argv[0]]
        if answer is True or answer is False:
            return subprocess.CompletedProcess(argv, 0, "abc123\n" if answer else "", "")
        return subprocess.CompletedProcess(argv, answer, "", "Error: cannot connect")

    monkeypatch.setattr(sandbox.subprocess, "run", run)


def test_a_container_under_another_engine_keeps_the_run_going(tmp_path, monkeypatch):
    """A run started with --engine docker, looked at by one that uses podman."""
    ended = _a_run_that_ended()
    clone = _clone_of(_host_out(tmp_path) / "repo", ended)
    monkeypatch.setattr(sandbox, "_other_engine_path_list",
                        lambda engine_path: ["/usr/bin/docker"])
    _engine_answer(monkeypatch, {"/usr/bin/podman": False, "/usr/bin/docker": True})
    assert sandbox.other_live_run_list(str(_host_out(tmp_path)), "repo", "mine",
                                       "/usr/bin/podman") == [ended]
    sandbox.remove_dead_clone(str(tmp_path / "out"), "repo", "/usr/bin/podman")
    assert clone.exists()
    assert sandbox._run_dead(ended) is False


@pytest.mark.parametrize("tail", ["\u00b2", "\u2460", "\uff11", str(2 ** 31)])
def test_a_run_id_whose_tail_names_no_process_counts_as_running(tail):
    assert sandbox._run_dead(f"20260101-000000-1-{tail}") is False


def test_another_engine_that_cannot_be_asked_holds_no_container(tmp_path, monkeypatch,
                                                                caplog):
    """An engine that is on PATH but not working, beside the one the run uses."""
    ended = _a_run_that_ended()
    clone = _clone_of(_host_out(tmp_path) / "repo", ended)
    monkeypatch.setattr(sandbox, "_other_engine_path_list",
                        lambda engine_path: ["/usr/bin/podman"])
    _engine_answer(monkeypatch, {"/usr/bin/docker": False, "/usr/bin/podman": 125})
    with caplog.at_level(logging.INFO, logger="codetwine_loom.sandbox"):
        sandbox.remove_dead_clone(str(tmp_path / "out"), "repo", "/usr/bin/docker")
    assert not clone.exists()
    assert any(r.levelno == logging.INFO and "exit status 125" in r.getMessage()
               and "counts as not there" in r.getMessage() for r in caplog.records)


def _engine_bin(tmp_path, monkeypatch):
    """Put stand-ins named podman and docker on a PATH of their own, with no engine
    set and none recorded as not answering. Return their directory."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("podman", "docker"):
        (bin_dir / name).write_text("#!/bin/sh\n")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "")
    monkeypatch.setattr(sandbox, "_NO_ANSWER_ENGINE_SET", set())
    monkeypatch.setattr(sandbox, "_other_engine_path_list", _other_engine_path_list)
    return bin_dir


def test_another_engine_that_does_not_answer_is_said_so_once_and_not_asked_again(
        tmp_path, monkeypatch, caplog):
    """docker is on PATH and does not answer, beside the podman the run uses."""
    bin_dir = _engine_bin(tmp_path, monkeypatch)
    podman, docker = str(bin_dir / "podman"), str(bin_dir / "docker")
    monkeypatch.setattr(config, "CONTAINER_COMMAND_TIMEOUT_SEC", 30)
    argv_list = []

    def run(argv, **kwargs):
        argv_list.append(argv)
        if argv[0] == docker:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 125 if argv[1] == "image" else 0, "", "no")

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    first, second = _a_run_that_ended(), _a_run_that_ended()
    with caplog.at_level(logging.INFO, logger="codetwine_loom.sandbox"):
        assert sandbox._run_dead(first, podman) is True
        assert sandbox._run_dead(second, podman) is True
        with pytest.raises(RuntimeError, match="podman does not show the image mine"):
            sandbox.image_label_dict(podman, "mine")
    assert [argv[0] for argv in argv_list] == [podman, docker, podman, podman]
    warn_list = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warn_list == [
        f"docker did not answer whether ctloom-{first} is running within 30 seconds, so it "
        f"is taken to hold nothing and is not asked again by this command"]
    assert _other_engine_path_list(podman) == []
    # The image is asked for the same way
    sandbox._NO_ANSWER_ENGINE_SET.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        with pytest.raises(RuntimeError):
            sandbox.image_label_dict(podman, "mine")
    assert "docker did not answer whether it has the image mine within 30 seconds" \
        in caplog.text
    assert sandbox._run_dead(first, podman) is True and argv_list[-1][0] == podman


def test_the_engine_a_run_uses_is_asked_again_after_it_did_not_answer(tmp_path,
                                                                     monkeypatch):
    """Only an engine other than the run's own is left out after it did not answer:
    the run's own that cannot be asked holds the container."""
    bin_dir = _engine_bin(tmp_path, monkeypatch)
    podman = str(bin_dir / "podman")
    asked_list = []

    def run(argv, **kwargs):
        asked_list.append(argv[0])
        if argv[0] == podman:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    ended = _a_run_that_ended()
    assert sandbox._run_dead(ended, podman) is False
    assert sandbox._run_dead(ended, podman) is False
    assert asked_list == [podman, podman] and sandbox._NO_ANSWER_ENGINE_SET == set()


def test_what_a_killed_run_left_is_asked_about_with_the_engine_the_run_looking_uses(
        tmp_path, monkeypatch):
    """A run started with --engine docker asks docker first about the start branch, the
    worktree and the clones a killed run left, whatever CONTAINER_ENGINE names."""
    repo = _repo(tmp_path / "repo")
    asked_list = []
    monkeypatch.setattr(sandbox, "pick_engine", lambda name="": f"/bin/{name or 'podman'}")
    monkeypatch.setattr(sandbox, "_run_dead",
                        lambda run_id, engine_path="": asked_list.append(engine_path) or False)
    _git(repo, "branch", f"codetwine-loom/run/{_DEAD_RUN}/tree", "HEAD")
    monkeypatch.setattr(gitcmd, "worktree_path_list",
                        lambda project_dir: [f"/gone/sandbox/worktree-{_DEAD_RUN}"])
    for engine, engine_path in (("", "/bin/podman"), ("docker", "/bin/docker")):
        asked_list.clear()
        sandbox.remove_dead_tree_branch(str(repo), *([engine] if engine else []))
        sandbox.remove_dead_worktree(str(repo), *([engine] if engine else []))
        with sandbox.tree_branch(str(repo), "20260102-000000-1-7", engine=engine):
            pass
        sandbox.take_in_patch(str(repo), "", "HEAD", f"codetwine-loom/sandbox/x{len(engine)}",
                              str(tmp_path / f"wt{len(engine)}"), "m\n\nm", engine=engine)
        assert asked_list == [engine_path] * 4
    # The analysis is not made again while a run the engine names is going
    seen_list = []
    monkeypatch.setattr(sandbox, "live_run_list",
                        lambda out_dir, name, engine="": seen_list.append(engine) or [])
    monkeypatch.setattr(sandbox.build, "analyse_and_index", lambda *a, **k: None)
    sandbox._update_doc(str(repo), str(tmp_path / "out"), "repo", None, "docker")
    sandbox.refuse_while_run_live(str(tmp_path / "out"), "repo")
    assert seen_list == ["docker", ""]


def test_a_run_passes_its_engine_to_what_looks_for_a_killed_run_s_leftovers(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    seen_dict = {}
    for name in ("remove_dead_tree_branch", "remove_dead_worktree"):
        monkeypatch.setattr(sandbox, name, lambda project_dir, engine="", name=name:
                            seen_dict.setdefault(name, []).append(engine) or [])
    monkeypatch.setattr(sandbox, "refuse_while_run_live",
                        lambda out_dir, name, other_way="", engine="":
                        seen_dict.setdefault("refuse", []).append(engine))
    monkeypatch.setattr(sandbox, "analyse_if_none", lambda *a, **k: False)
    monkeypatch.setattr(sandbox, "exclude_list_of_record", lambda *a, **k: None)
    monkeypatch.setattr(sandbox.build, "analyse_and_index", lambda *a, **k: None)

    def write(clone):
        (clone / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")

    patch_text = _patch_from(repo, write)

    def leave_behind(argv):
        patch = _told_to_write(argv)
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(patch_text, encoding="utf-8")
        patch.with_suffix(f"{patch.suffix}.meta.json").write_text(
            json.dumps({"base": "b", "exit_status": 0, "files": 1}), encoding="utf-8")

    result, _ = _run_without_a_container(
        tmp_path, monkeypatch, on_start=leave_behind, project=str(repo),
        output_dir=str(tmp_path / "out"), request="r", engine="docker", build_doc=True)
    assert result["commit"]
    assert seen_dict == {"refuse": ["docker"], "remove_dead_tree_branch": ["docker"],
                         "remove_dead_worktree": ["docker"]}


def test_the_other_engines_are_the_setting_s_then_the_known_ones_each_once(tmp_path,
                                                                          monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("podman", "mine"):
        (bin_dir / name).write_text("#!/bin/sh\n")
        (bin_dir / name).chmod(0o755)
    (bin_dir / "docker").symlink_to(bin_dir / "podman")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "mine")
    assert _other_engine_path_list("") == [str(bin_dir / "mine"), str(bin_dir / "podman")]
    assert _other_engine_path_list(str(bin_dir / "docker")) == [str(bin_dir / "mine")]
    assert _other_engine_path_list("docker") == [str(bin_dir / "mine")]
    monkeypatch.setattr(config, "CONTAINER_ENGINE", "")
    assert _other_engine_path_list(str(bin_dir / "mine")) == [str(bin_dir / "podman")]


def test_an_engine_that_cannot_be_asked_takes_nothing_away(tmp_path, monkeypatch):
    """Nothing is removed on the strength of a question that failed."""
    project_out = _host_out(tmp_path) / "repo"
    clone = _clone_of(project_out, _a_run_that_ended())

    def blow_up(argv, **kwargs):
        raise OSError("the engine is not there")

    monkeypatch.setattr(sandbox.subprocess, "run", blow_up)
    sandbox.remove_dead_clone(str(tmp_path / "out"), "repo", "/usr/bin/podman")
    assert clone.exists()


# == Taking two patches in at once ============================

_TAKES_IN = '''
import sys, time
sys.path.insert(0, {package!r})
from codetwine_loom import sandbox

def say(what):
    with open({said!r}, "a") as fh:
        fh.write("%s %s %.4f\\n" % (what, {tag!r}, time.monotonic()))

# What collect() does under the lock, made long enough that two runs doing it at once
# would be seen doing it at once
def slowly(*args, **kwargs):
    say("in")
    time.sleep(0.5)
    say("out")
    return real(*args, **kwargs)

real = sandbox._apply_on_branch
sandbox._apply_on_branch = slowly
sandbox.take_in_patch({repo!r}, {patch!r}, {base!r}, {branch!r}, {worktree!r}, "sandbox: x")
'''


def _take_in_at_once(tmp_path):
    """Take two patches in to one repository from two processes at once.

    Args:
        tmp_path: The test's directory.

    Returns:
        (the repository, the lines the two wrote as they went in and out).
    """
    repo = _repo(tmp_path / "repo")
    base = gitcmd.require_head(str(repo))
    said = tmp_path / "said.txt"
    package = str(pathlib.Path(sandbox.__file__).resolve().parent.parent)
    writers = []
    for tag, line in (("one", "two\n"), ("two", "three\n")):
        patch = tmp_path / f"{tag}.diff"
        patch.write_text(_patch_from(repo, lambda c, line=line:
                                     (c / "kept.txt").write_text(f"one\n{line}")),
                         encoding="utf-8")
        shutil.rmtree(tmp_path / f"{repo.name}-clone")
        script = tmp_path / f"{tag}.py"
        script.write_text(_TAKES_IN.format(
            package=package, repo=str(repo), patch=str(patch), base=base, tag=tag,
            said=str(said), branch=f"codetwine-loom/sandbox/{tag}",
            worktree=str(tmp_path / f"wt-{tag}")), encoding="utf-8")
        writers.append(subprocess.Popen([sys.executable, str(script)],
                                        stderr=subprocess.PIPE, text=True))
    for writer in writers:
        _, complaint = writer.communicate()
        assert writer.returncode == 0, complaint
    return repo, said.read_text(encoding="utf-8").split("\n")


def test_two_runs_finishing_at_once_take_their_patches_in_one_at_a_time(tmp_path):
    """git keeps one index per repository and refuses a second worktree command while
    one is running, so the two wait for each other rather than racing."""
    _, said = _take_in_at_once(tmp_path)
    order = [line.split()[0] for line in said if line.strip()]
    assert order == ["in", "out", "in", "out"]


def test_two_runs_finishing_at_once_both_land_on_their_branch(tmp_path):
    repo, _ = _take_in_at_once(tmp_path)
    branches = _git(repo, "branch", "--list")
    assert "codetwine-loom/sandbox/one" in branches and "codetwine-loom/sandbox/two" in branches
    assert gitcmd.status_path_list(str(repo)) == []


_MERGES = '''
import sys, time
sys.path.insert(0, {package!r})
from codetwine_loom import gitcmd, sandbox

def say(what):
    with open({said!r}, "a") as fh:
        fh.write("%s %s %.4f\\n" % (what, {tag!r}, time.monotonic()))

# The check and the apply land_in_tree() makes under the lock, made long enough that
# two runs making them at once would be seen making them at once
real = gitcmd.run
def slowly(project_dir, *args, **kwargs):
    if args[:1] == ("apply",) and "--check" in args and "-v" not in args:
        say("in")
        time.sleep(0.5)
        return real(project_dir, *args, **kwargs)
    if args[:1] == ("apply",) and "--check" not in args:
        done = real(project_dir, *args, **kwargs)
        say("out")
        return done
    return real(project_dir, *args, **kwargs)

gitcmd.run = slowly
sandbox.land_in_tree({repo!r}, gitcmd.require_head({repo!r}), {branch!r},
                     {repo!r} + "-" + {tag!r} + ".diff", {repo!r} + "-last.diff")
'''


def test_two_runs_landing_at_once_go_one_after_the_other(tmp_path):
    """The working tree is one, so two changes written into it at once would be
    checked against a tree the other is changing; under the take-in lock they wait
    instead."""
    repo = _repo(tmp_path / "repo")
    for tag in ("one", "two"):
        _git(repo, "checkout", "--quiet", "-b", f"codetwine-loom/sandbox/{tag}", "main")
        (repo / f"{tag}.txt").write_text("x\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", tag)
    _git(repo, "checkout", "--quiet", "main")
    said = tmp_path / "said.txt"
    package = str(pathlib.Path(sandbox.__file__).resolve().parent.parent)
    writers = []
    for tag in ("one", "two"):
        script = tmp_path / f"merge-{tag}.py"
        script.write_text(_MERGES.format(package=package, repo=str(repo), tag=tag,
                                         said=str(said), branch=f"codetwine-loom/sandbox/{tag}"),
                          encoding="utf-8")
        writers.append(subprocess.Popen([sys.executable, str(script)],
                                        stderr=subprocess.PIPE, text=True))
    for writer in writers:
        _, complaint = writer.communicate()
        assert writer.returncode == 0, complaint
    order = [line.split()[0]
             for line in said.read_text(encoding="utf-8").split("\n") if line.strip()]
    assert order == ["in", "out", "in", "out"]
    assert (repo / "one.txt").exists() and (repo / "two.txt").exists()
    assert sorted(gitcmd.status_path_list(str(repo))) == ["one.txt", "two.txt"]


# == The agent's own copy of the output directory =============

def test_the_branch_is_what_names_the_agent():
    """Passing the same --branch again is what makes a run the next round of one."""
    assert sandbox.agent_name("codetwine-loom/agent1", "20260101-000000-1") == \
        "codetwine-loom-agent1"


def test_a_run_with_no_branch_is_an_agent_of_its_own():
    assert sandbox.agent_name("", "20260101-000000-1") == "20260101-000000-1"


def _built(tmp_path, name="repo"):
    """Leave behind what build writes, plus what runs write, and return the two paths."""
    built = tmp_path / "out" / name
    (built / "sandbox").mkdir(parents=True)
    (built / "turns").mkdir()
    (built / "codetwine_py").mkdir()
    (built / "code_index.sqlite").write_text("an index", encoding="utf-8")
    (built / "project_knowledge.sqlite").write_text("an analysis", encoding="utf-8")
    (built / "codetwine_py" / "codetwine.py").write_text("a copy", encoding="utf-8")
    (built / "edit_history.json").write_text("{}", encoding="utf-8")
    (built / "turns" / "20260101-000000-aaaaaa.jsonl").write_text("", encoding="utf-8")
    (built / "sandbox" / "patch-20260101-000000-1.diff").write_text("x",
                                                                   encoding="utf-8")
    return tmp_path / "out", built


def test_the_agent_gets_what_build_made(tmp_path):
    """A rebuild in an empty directory would leave the summaries out, so the copy is
    what makes the agent's own index keep them."""
    out, _ = _built(tmp_path)
    mine = pathlib.Path(sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "",
                                               None, False))
    assert mine == pathlib.Path(sandbox.agent_dir(str(out), "codetwine-loom-agent1"))
    assert sorted(p.name for p in (mine / "repo").iterdir()) == \
        ["code_index.sqlite", "codetwine_py", "project_knowledge.sqlite"]


def test_what_runs_wrote_is_not_the_agent_s_to_start_with(tmp_path):
    """A clone, a patch, a turn log and a record belong to whoever wrote them."""
    out, _ = _built(tmp_path)
    mine = pathlib.Path(sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "",
                                               None, False))
    for left in ("sandbox", "turns", "edit_history.json"):
        assert not (mine / "repo" / left).exists()


def test_the_analysis_of_a_directory_named_as_one_of_this_tool_s_own_is_copied(tmp_path):
    """codetwine files the analysis of the repository's sandbox/ and turns/ under the
    names this tool keeps its own files under; every directory in them is the agent's
    to start with, at any depth, and this tool's own files beside them are not."""
    out, built = _built(tmp_path)
    for rel in ("sandbox/entry_sh", "turns/next_py", "pkg/sandbox/tool_py", "pkg/turns/t_py",
                "pkg/edit_history.json", "sandbox/cache/x_py", "sandbox/record/y_py",
                "sandbox/land-before-1-x/z_py"):
        (built / rel).mkdir(parents=True)
        (built / rel / "doc.json").write_text("{}", encoding="utf-8")
    (built / "sandbox" / "land-last.diff").write_text("d", encoding="utf-8")
    (built / "sandbox" / "cache.lock").write_text("", encoding="utf-8")
    mine = pathlib.Path(sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "", None,
                                               False)) / "repo"
    copy_list = sorted(str(path.relative_to(mine)) for path in mine.rglob("*")
                       if path.is_file())
    assert copy_list == ["code_index.sqlite", "codetwine_py/codetwine.py",
                    "pkg/edit_history.json/doc.json", "pkg/sandbox/tool_py/doc.json",
                    "pkg/turns/t_py/doc.json", "project_knowledge.sqlite",
                    "sandbox/cache/x_py/doc.json", "sandbox/entry_sh/doc.json",
                    "sandbox/land-before-1-x/z_py/doc.json",
                    "sandbox/record/y_py/doc.json", "turns/next_py/doc.json"]
    assert sorted(p.name for p in (mine / "sandbox").iterdir()) == [
        "cache", "entry_sh", "land-before-1-x", "record"]
    # What the run then writes beside them stays when the copy goes, and they go
    (mine / "sandbox" / "patch-1.diff").write_text("d", encoding="utf-8")
    (mine / "turns" / "1.jsonl").write_text("", encoding="utf-8")
    sandbox.drop_the_copy(str(out), "repo", "codetwine-loom-agent1")
    assert sorted(str(path.relative_to(mine)) for path in mine.rglob("*")) == [
        "sandbox", "sandbox/patch-1.diff", "turns", "turns/1.jsonl"]


def test_an_agent_named_before_is_copied_the_analysis_again(tmp_path):
    """The copy is taken away when a run ends and what the run wrote stays, so the
    agent's directory being there does not say the analysis is."""
    out, built = _built(tmp_path)
    make = lambda: pathlib.Path(sandbox.make_agent_dir(   # noqa: E731
        str(out), "repo", "codetwine-loom-agent1", "", None, False))
    mine = make()
    _what_a_run_wrote(mine)
    sandbox.drop_the_copy(str(out), "repo", "codetwine-loom-agent1")
    assert not (mine / "repo" / "project_knowledge.sqlite").exists()
    make()
    assert sorted(p.name for p in (mine / "repo").iterdir()) == [
        "code_index.sqlite", "codetwine_py", history.HISTORY_FILE,
        "project_knowledge.sqlite", "sandbox", "turns"]
    assert (mine / "repo" / "sandbox" / "patch-1.diff").read_text(encoding="utf-8") == "d"
    assert (mine / "repo" / "turns" / "1.jsonl").exists()


def test_the_copy_is_not_made_through_a_link_a_run_left(tmp_path):
    """A container writes the agent's directory, so what it left there that is not a
    run's own file is taken away before anything is copied in."""
    out, built = _built(tmp_path)
    (built / "sandbox" / "entry_sh").mkdir()
    (built / "sandbox" / "entry_sh" / "doc.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside"
    for name in ("a", "b", "c"):
        (outside / name).mkdir(parents=True)
        (outside / name / "kept.txt").write_text("the user's", encoding="utf-8")
    mine = pathlib.Path(sandbox.agent_dir(str(out), "codetwine-loom-agent1")) / "repo"
    (mine / "sandbox").mkdir(parents=True)
    (mine / "sandbox" / "patch-1.diff").write_text("d", encoding="utf-8")
    (mine / "sandbox" / "entry_sh").symlink_to(outside / "a")
    (mine / "codetwine_py").symlink_to(outside / "b")
    (mine / "turns").symlink_to(outside / "c")
    sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "", None, False)
    assert sorted(str(p.relative_to(outside)) for p in outside.rglob("*") if p.is_file()) \
        == ["a/kept.txt", "b/kept.txt", "c/kept.txt"]
    assert not any(path.is_symlink() for path in mine.rglob("*"))
    assert (mine / "sandbox" / "entry_sh" / "doc.json").is_file()
    assert (mine / "codetwine_py" / "codetwine.py").is_file()
    assert (mine / "sandbox" / "patch-1.diff").read_text(encoding="utf-8") == "d"


@pytest.mark.parametrize("kind", ["file", "dir", "broken", "self", "fifo"])
@pytest.mark.parametrize("where", ["sandbox/cache/home", "codetwine_py", ""])
def test_what_the_copy_is_made_from_is_not_read_through_a_link(tmp_path, caplog, kind,
                                                               where):
    """A link in what the agent's copy is made from - the package cache an older layout
    kept in <name>/sandbox/cache/, which containers wrote, among it - is left out
    wherever it leads, and so is anything else that is no file or directory; the rest
    is copied."""
    out, built = _built(tmp_path)
    outside = tmp_path / "outside"
    (outside / "dir").mkdir(parents=True)
    (outside / "secret.txt").write_text("HOST-SECRET", encoding="utf-8")
    (outside / "dir" / "id_key").write_text("HOST-SECRET", encoding="utf-8")
    place = built / where
    place.mkdir(parents=True, exist_ok=True)
    planted = place / "planted"
    if kind == "fifo":
        os.mkfifo(planted)
    else:
        planted.symlink_to({"file": outside / "secret.txt", "dir": outside / "dir",
                            "broken": outside / "none.txt", "self": planted}[kind])
    caplog.set_level(logging.WARNING, logger=sandbox.__name__)
    mine = pathlib.Path(sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "",
                                               None, False)) / "repo"
    assert not any(path.is_symlink() or path.is_fifo() for path in mine.rglob("*"))
    assert not any("HOST-SECRET" in path.read_text(encoding="utf-8")
                   for path in mine.rglob("*") if path.is_file())
    assert not (mine / where / "planted").exists()
    assert (mine / "codetwine_py" / "codetwine.py").read_text(encoding="utf-8") == "a copy"
    assert (mine / "project_knowledge.sqlite").is_file()
    assert str(planted) in caplog.text


def test_a_copy_replaced_by_a_link_is_taken_away_and_nothing_behind_it(tmp_path, caplog):
    out, _ = _built(tmp_path)
    outside = tmp_path / "outside"
    (outside / "work").mkdir(parents=True)
    (outside / "work" / "kept.txt").write_text("the user's", encoding="utf-8")
    (outside / "notes.txt").write_text("the user's", encoding="utf-8")
    own_dir = pathlib.Path(sandbox.agent_dir(str(out), "codetwine-loom-agent1"))
    own_dir.mkdir(parents=True)
    (own_dir / "repo").symlink_to(outside)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        sandbox.drop_the_copy(str(out), "repo", "codetwine-loom-agent1")
    assert "is a link" in caplog.text
    assert sorted(str(p.relative_to(outside)) for p in outside.rglob("*")) == [
        "notes.txt", "work", "work/kept.txt"]
    assert not own_dir.exists()


def _copy_a_run_left(tmp_path, out, left: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Make an agent's directory as a container that ran in it left it, with a link
    that leads out of it, and return (the agent's copy, what the link leads to)."""
    outside = tmp_path / "outside"
    (outside / "dir").mkdir(parents=True)
    (outside / "file.txt").write_text("the user's", encoding="utf-8")
    own_dir = pathlib.Path(sandbox.agent_dir(str(out), "codetwine-loom-agent1"))
    mine = own_dir / "repo"
    if left == "the copy as a link":
        own_dir.mkdir(parents=True)
        mine.symlink_to(outside / "dir")
        return mine, outside
    (mine / "sandbox").mkdir(parents=True)
    (mine / "sandbox" / "patch-1.diff").write_text("d", encoding="utf-8")
    if left == "a link in a directory":
        (mine / "codetwine_py").mkdir()
        (mine / "codetwine_py" / "codetwine.py").symlink_to(outside / "file.txt")
    elif left == "a link to a directory":
        (mine / "codetwine_py").symlink_to(outside / "dir")
    elif left == "a link to a file":
        (mine / "code_index.sqlite").symlink_to(outside / "file.txt")
    elif left == "a link to no file":
        (mine / "code_index.sqlite").symlink_to(outside / "none.txt")
    else:
        (mine / "turns").symlink_to(outside / "dir")
    return mine, outside


def _outside_as_it_was(outside: pathlib.Path) -> bool:
    """Say whether what the links of _copy_a_run_left() lead to is as it was made."""
    return sorted(str(p.relative_to(outside)) for p in outside.rglob("*")) \
        == ["dir", "file.txt"] \
        and (outside / "file.txt").read_text(encoding="utf-8") == "the user's"


@pytest.mark.parametrize("left", ["a link in a directory", "a link to a directory",
                                  "a link to a file", "the copy as a link"])
def test_nothing_is_copied_over_what_could_not_be_taken_away(tmp_path, monkeypatch, caplog,
                                                             left):
    """A container that ran in the agent's directory may leave a link there in a
    directory nothing can be removed from. Nothing is then copied into it."""
    out, _ = _built(tmp_path)
    mine, outside = _copy_a_run_left(tmp_path, out, left)

    def refuse(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(path))

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.shutil, "rmtree", refuse)
        patch.setattr(sandbox.os, "remove", refuse)
        with pytest.raises(RuntimeError,
                           match="codetwine-loom-agent1's copy of the output directory "
                                 "holds what could not be taken away") as error_info:
            sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "", None, False)
    assert str(mine) in str(error_info.value) and "by hand" in str(error_info.value)
    assert _outside_as_it_was(outside)
    assert not (mine / "project_knowledge.sqlite").exists()
    assert "could not be taken away" in caplog.text


@pytest.mark.parametrize("left", ["a link in a directory", "a link to a directory",
                                  "a link to a file", "a link to no file",
                                  "one of this tool's directories as a link",
                                  "the copy as a link"])
def test_the_copy_is_made_onto_nothing_that_stands_there(tmp_path, monkeypatch, left):
    """Each entry of the copy is made new: a link standing where one goes is neither
    written through nor copied into, whatever the removal before it found."""
    out, built = _built(tmp_path)
    (built / "turns" / "next_py").mkdir()
    (built / "turns" / "next_py" / "doc.json").write_text("{}", encoding="utf-8")
    mine, outside = _copy_a_run_left(tmp_path, out, left)
    monkeypatch.setattr(sandbox, "_remove_copy", lambda copy_dir, agent: [])
    with pytest.raises(RuntimeError, match="could not make codetwine-loom-agent1's copy"):
        sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "", None, False)
    assert _outside_as_it_was(outside)


def test_a_file_a_run_left_where_a_directory_of_the_copy_goes_stops_the_copy(tmp_path):
    """Every file in one of this tool's directories is a run's own and stays, so the
    analysis of a directory of that name cannot be copied beside it."""
    out, built = _built(tmp_path)
    (built / "sandbox" / "entry_sh").mkdir()
    (built / "sandbox" / "entry_sh" / "doc.json").write_text("{}", encoding="utf-8")
    mine = pathlib.Path(sandbox.agent_dir(str(out), "codetwine-loom-agent1")) / "repo"
    (mine / "sandbox").mkdir(parents=True)
    (mine / "sandbox" / "entry_sh").write_text("what a run wrote", encoding="utf-8")
    with pytest.raises(RuntimeError, match="could not make codetwine-loom-agent1's copy"):
        sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "", None, False)
    assert (mine / "sandbox" / "entry_sh").read_text(encoding="utf-8") == "what a run wrote"


def test_what_stays_in_a_copy_is_handed_back(tmp_path, monkeypatch, caplog):
    mine = tmp_path / "repo"
    for rel in ("sandbox/entry_sh", "codetwine_py", "turns"):
        (mine / rel).mkdir(parents=True)
    (mine / "sandbox" / "patch-1.diff").write_text("d", encoding="utf-8")
    (mine / "code_index.sqlite").write_text("an index", encoding="utf-8")
    real_rmtree, real_listdir = shutil.rmtree, os.listdir

    def rmtree(path, *args, **kwargs):
        if os.path.basename(path) == "codetwine_py":
            raise PermissionError(13, "Permission denied", str(path))
        return real_rmtree(path, *args, **kwargs)

    def listdir(path="."):
        if os.path.basename(path) == "turns":
            raise PermissionError(13, "Permission denied", str(path))
        return real_listdir(path)

    with monkeypatch.context() as patch:
        patch.setattr(sandbox.shutil, "rmtree", rmtree)
        patch.setattr(sandbox.os, "listdir", listdir)
        stay_list = sandbox._remove_copy(str(mine), "codetwine-loom-agent1")
    assert sorted(stay_list) == [str(mine / "codetwine_py"), str(mine / "turns")]
    assert sorted(p.name for p in mine.iterdir()) == ["codetwine_py", "sandbox", "turns"]
    assert "could not be listed" in caplog.text
    assert sandbox._remove_copy(str(mine), "codetwine-loom-agent1") == []
    assert sandbox._remove_copy(str(tmp_path / "none"), "codetwine-loom-agent1") == []


def test_a_copy_that_cannot_be_listed_is_left_and_said_when_the_run_ends(tmp_path,
                                                                        monkeypatch, caplog):
    out, _ = _built(tmp_path)
    mine = pathlib.Path(sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "",
                                               None, False))
    real_listdir = os.listdir

    def listdir(path="."):
        if str(path) == str(mine / "repo"):
            raise PermissionError(13, "Permission denied", str(path))
        return real_listdir(path)

    monkeypatch.setattr(sandbox.os, "listdir", listdir)
    sandbox.drop_the_copy(str(out), "repo", "codetwine-loom-agent1")
    assert "codetwine-loom-agent1's copy at" in caplog.text and "could not be listed" in caplog.text
    (mine / "repo" / "project_knowledge.sqlite").unlink()
    with pytest.raises(RuntimeError, match="holds what could not be taken away"):
        sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "", None, False)


def test_the_copy_is_looked_for_and_made_under_the_agents_lock(tmp_path, monkeypatch):
    """Two first rounds started at once would otherwise each find no copy and both
    make one; the second waits at the lock and finds it made."""
    out, _ = _built(tmp_path)
    events = []
    real_held = sandbox.filelock.hold_lock

    @contextlib.contextmanager
    def noted(path, what, wait=True):
        events.append(("enter", what))
        try:
            with real_held(path, what, wait=wait):
                yield
        finally:
            events.append(("exit", what))

    monkeypatch.setattr(sandbox.filelock, "hold_lock", noted)
    real_copytree = shutil.copytree

    def copied(*args, **kwargs):
        events.append(("copy", ""))
        return real_copytree(*args, **kwargs)

    monkeypatch.setattr(sandbox.shutil, "copytree", copied)
    sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "", None, False)
    what = "codetwine-loom-agent1's copy of the output directory"
    inside = events[events.index(("enter", what)) + 1:events.index(("exit", what))]
    assert ("copy", "") in inside


def test_the_copy_is_made_once_and_the_agent_keeps_it(tmp_path):
    """Its next round rebuilds the index in it rather than starting again from build."""
    out, built = _built(tmp_path)
    mine = pathlib.Path(sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "",
                                               None, False))
    (mine / "repo" / "code_index.sqlite").write_text("a later index",
                                                         encoding="utf-8")
    sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "", None, False)
    assert (mine / "repo" / "code_index.sqlite").read_text(encoding="utf-8") == \
        "a later index"


def test_an_output_directory_with_nothing_built_still_gets_a_place(tmp_path):
    """The container builds the index itself when there is none to copy."""
    mine = pathlib.Path(sandbox.make_agent_dir(str(tmp_path / "out"), "repo", "a1", "", None, False))
    assert mine.is_dir() and not (mine / "repo").exists()


def test_a_copy_that_cannot_be_made_stops_the_run(tmp_path, monkeypatch):
    """Going on would build a second index over a directory the agent does not own."""
    out, _ = _built(tmp_path)

    def refuse(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sandbox.shutil, "copytree", refuse)
    with pytest.raises(RuntimeError, match="could not make codetwine-loom-agent1's copy"):
        sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "", None, False)


def test_the_agents_read_one_cache_and_it_stays_where_build_put_it(tmp_path,
                                                                   monkeypatch):
    """Filling it is the one step that needs a network, so it is not copied per agent."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    sandbox.run(str(repo), str(out), "x", branch="codetwine-loom/agent1")
    sandbox.run(str(repo), str(out), "x", branch="codetwine-loom/agent2")
    mounted = {a for argv in seen for a in argv if a.endswith(":/cache")}
    assert mounted == {f"{out}/agents/.repo/repo/cache:/cache"}


def test_the_broker_is_told_where_the_shared_cache_is(tmp_path, monkeypatch):
    """It fills that cache for the run, and the run works in a copy of its own."""
    seen = {}

    class _Listening:
        def poll(self):
            return None

    def popen(argv, **kwargs):
        seen["env"] = kwargs.get("env") or {}
        pathlib.Path(tmp_path / "llm.sock").touch()
        return _Listening()

    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    _START_BROKER(str(tmp_path), "/host/repo", "/host/out/agents/a1", "20260101-1",
                  "/host/out/repo/sandbox/cache")
    assert seen["env"]["SANDBOX_CACHE_DIR"] == "/host/out/repo/sandbox/cache"


# == the copy an agent with no next round leaves behind ======
def _what_a_run_wrote(mine: pathlib.Path) -> None:
    """Write into an agent's own directory what a run of it leaves there."""
    (mine / "repo" / "sandbox").mkdir(parents=True)
    (mine / "repo" / "sandbox" / "patch-1.diff").write_text("d", encoding="utf-8")
    (mine / "repo" / "turns").mkdir()
    (mine / "repo" / "turns" / "1.jsonl").write_text("", encoding="utf-8")
    (mine / "repo" / history.HISTORY_FILE).write_text("{}", encoding="utf-8")


def test_the_copy_goes_and_what_the_run_wrote_stays(tmp_path):
    """The copy is there for the agent's next round. What the run wrote is the only
    account of what happened, and nothing else holds it."""
    out, built = _built(tmp_path)
    mine = pathlib.Path(sandbox.make_agent_dir(str(out), "repo", "codetwine-loom-agent1", "",
                                               None, False))
    _what_a_run_wrote(mine)
    sandbox.drop_the_copy(str(out), "repo", "codetwine-loom-agent1")
    assert sorted(p.name for p in (mine / "repo").iterdir()) == [
        history.HISTORY_FILE, "sandbox", "turns"]
    assert (mine / "repo" / "sandbox" / "patch-1.diff").exists()
    # What build wrote is where it was: the copy was made from it, not moved
    assert (built / "code_index.sqlite").exists()


def test_an_agent_that_wrote_nothing_leaves_nothing(tmp_path):
    out, _ = _built(tmp_path)
    mine = pathlib.Path(sandbox.make_agent_dir(str(out), "repo", "one-round", "", None, False))
    sandbox.drop_the_copy(str(out), "repo", "one-round")
    assert not mine.exists()
    assert not pathlib.Path(f"{mine}.lock").exists()


def test_a_run_given_a_branch_leaves_no_copy_either(tmp_path, monkeypatch):
    """A task of a dispatching run is given its branch, and nothing asks for that
    agent again once the task has run."""
    out, _ = _built(tmp_path)
    repo = _repo(tmp_path / "repo")
    _stamp_only(monkeypatch)
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: ("/usr/bin/podman", "img", 0))
    sandbox.run(str(repo), str(out), "直して", branch="codetwine-loom/agent1", dry_run=True)
    assert not (out / "agents" / "codetwine-loom-agent1" / "repo" / "code_index.sqlite").exists()


def test_a_run_given_no_branch_leaves_no_copy(tmp_path, monkeypatch):
    """The name it was given is the run's own, so no later run asks for that copy."""
    out, _ = _built(tmp_path)
    repo = _repo(tmp_path / "repo")
    _stamp_only(monkeypatch)
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: ("/usr/bin/podman", "img", 0))
    sandbox.run(str(repo), str(out), "直して", dry_run=True)
    assert not list((out / "agents").glob("*/repo/code_index.sqlite"))


def test_the_container_keeps_out_what_the_analysis_it_copies_keeps_out(tmp_path, monkeypatch):
    """A run given no --exclude, over an analysis made with one."""
    out, _ = _built(tmp_path)
    repo = _repo(tmp_path / "repo")
    (out / "repo" / sandbox.knowledge.ANALYSIS_COMMIT_FILE).write_text(json.dumps(
        {"stamp": "1:now", "commit": "c", "exclude": ["sample_output", "logs"]}),
        encoding="utf-8")
    _stamp_only(monkeypatch)
    seen = []
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: seen.append(a) or ("/usr/bin/podman", "img", 0))
    sandbox.run(str(repo), str(out), "直して", dry_run=True)
    assert seen[0][-1] == ["sample_output", "logs"]   # _start_container's exclude_list


# == dispatch: one request read, then one agent per task ====
def _never(why: str):
    """Return a stand-in that fails the test if anything calls it."""
    def called(*a, **k):
        raise AssertionError(why)

    return called


def test_a_run_id_is_new_every_time_within_one_process():
    """dispatch starts several agents from one process, and two of them sharing a name
    would share a container and a clone."""
    ids = [sandbox.new_run_id() for _ in range(5)]
    assert len(set(ids)) == 5


def test_the_process_is_still_the_last_part_of_a_run_id():
    """_run_is_gone() reads it from there to decide whether a clone can be swept."""
    assert sandbox.new_run_id().rsplit("-", 1)[-1] == str(os.getpid())


def test_a_clone_of_this_process_is_never_swept():
    """A run id this process makes is read as the id of a live run."""
    assert sandbox._run_dead(sandbox.new_run_id()) is False


def test_the_split_file_is_named_after_the_run():
    files = sandbox.run_file_layout("/out", "a1", "repo", "20260101-000000-1-42")
    assert files.task.endswith("tasks-20260101-000000-1-42.json")
    assert os.path.dirname(files.task) == os.path.dirname(files.patch)


def test_the_answer_files_are_named_after_the_run_by_the_same_rule():
    """What an answering run hands back, and what the combining run is handed, cross
    the boundary the way the split does."""
    files = sandbox.run_file_layout("/out", "a1", "repo", "20260101-000000-1-42")
    assert files.answer.endswith("answer-20260101-000000-1-42.json")
    assert files.answer_in.endswith("answered-20260101-000000-1-42.json")
    assert os.path.dirname(files.answer) == os.path.dirname(files.task)
    for outside, inside in ((files.answer, files.answer_inside),
                            (files.answer_in, files.answer_in_inside)):
        assert inside == f"{sandbox._OUT}/repo/sandbox/{os.path.basename(outside)}"


def test_the_split_is_named_once_for_both_sides_of_the_boundary():
    """The container writes the file this machine reads, and one rule builds both
    paths: a second rule would let one of them move on its own."""
    files = sandbox.run_file_layout("/out", "a1", "repo", "20260101-000000-1-42")
    assert files.task_inside == f"{sandbox._OUT}/repo/sandbox/" \
                                 f"{os.path.basename(files.task)}"
    assert files.task_inside.startswith(sandbox._OUT)


def test_what_this_machine_alone_writes_is_kept_where_no_container_is_given():
    """The agent's copy of the output directory is what the container writes, so the
    clone, the diff put into the tree, the stdout and the worktree are not in it."""
    files = sandbox.run_file_layout("/out", "a1", "repo", "20260101-000000-1-42")
    assert files.out_dir == "/out/agents/a1"
    assert files.directory == "/out/agents/a1/repo/sandbox"
    assert files.host_directory == "/out/agents/.host/a1/repo/sandbox"
    for path in (files.patch, files.task, files.outcome, files.answer, files.answer_in,
                 files.target, files.gap):
        assert os.path.dirname(path) == files.directory
    assert files.scratch == f"{files.host_directory}/scratch-20260101-000000-1-42"
    assert files.warm_scratch == f"{files.host_directory}/scratch-warm-20260101-000000-1-42"
    assert files.worktree == f"{files.host_directory}/worktree-20260101-000000-1-42"
    assert files.land == f"{files.host_directory}/land-20260101-000000-1-42.diff"
    assert files.stdout == f"{files.host_directory}/stdout-20260101-000000-1-42.txt"
    assert sandbox.host_dir("/out", "a1") == "/out/agents/.host/a1"


def test_this_machine_s_side_is_no_agent_s_copy(tmp_path):
    out = tmp_path / "out"
    os.makedirs(sandbox.agent_dir(str(out), "codetwine-loom-one"))
    os.makedirs(sandbox.host_dir(str(out), "codetwine-loom-one"))
    os.makedirs(sandbox.host_dir(str(out), "codetwine-loom-two"))
    os.makedirs(sandbox.repo_own_dir(str(out), "repo"))
    assert sandbox.output_dir_list(str(out)) == [
        ("", str(out)), ("codetwine-loom-one", sandbox.agent_dir(str(out), "codetwine-loom-one"))]
    assert sandbox.host_dir_list(str(out)) == [
        ("codetwine-loom-one", sandbox.host_dir(str(out), "codetwine-loom-one")),
        ("codetwine-loom-two", sandbox.host_dir(str(out), "codetwine-loom-two"))]
    assert sandbox.host_dir_list(str(tmp_path / "none")) == []


def test_a_container_is_given_the_agent_s_copy_and_its_clone_and_nothing_above_them(
        tmp_path, monkeypatch):
    """No mount holds the output directory, another agent's copy, or what this machine
    keeps for the agent beside the three directories of the clone."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    _, argv = _run_without_a_container(tmp_path, monkeypatch, project=str(repo),
                                       output_dir=str(out), request="x",
                                       branch="codetwine-loom/agent1")
    mount_dict = {argv[i + 1].split(":")[1]: pathlib.Path(argv[i + 1].split(":")[0])
                  for i, arg in enumerate(argv) if arg == "-v"}
    host_side = pathlib.Path(sandbox.host_dir(str(out), "codetwine-loom-agent1"))
    assert mount_dict["/out"] == out / "agents" / "codetwine-loom-agent1"
    scratch = mount_dict["/work"].parent
    assert scratch.parent == host_side / "repo" / "sandbox"
    assert (mount_dict["/src"], mount_dict["/tmp"]) == (scratch / "src", scratch / "tmp")
    assert mount_dict["/cache"] == out / "agents" / ".repo" / "repo" / "cache"
    for inside, path in mount_dict.items():
        for kept in (out / "repo" / "code_index.sqlite", out / "agents" / "codetwine-loom-agent2",
                     host_side / "repo" / "sandbox" / "land-x.diff",
                     host_side / "repo" / "turns", scratch,
                     out / "agents" / ".repo" / "repo" / "land-last.diff",
                     out / "agents" / ".repo" / "repo" / "land-before-1-x",
                     out / "agents" / ".repo" / "repo" / "record"):
            assert path not in (kept, *kept.parents), (inside, path)


def test_the_runs_of_an_agent_are_locked_and_logged_on_this_machine_s_side(tmp_path,
                                                                           monkeypatch):
    lock_list, start_list = [], []
    real_held = sandbox.filelock.hold_lock

    @contextlib.contextmanager
    def noted(path, what, wait=True):
        lock_list.append(path)
        with real_held(path, what, wait=wait):
            yield

    monkeypatch.setattr(sandbox.filelock, "hold_lock", noted)
    arg_tuple, _ = _container_ready(
        tmp_path, monkeypatch, lambda sock_dir, *a, **k: start_list.append((a, k)))
    seen = []
    monkeypatch.setattr(sandbox, "run_the_container",
                        lambda argv, engine, run_id, limit, out_path:
                        seen.append((limit, out_path)) or 0)
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SEC", 77)
    sandbox._start_container(*arg_tuple)
    sandbox._start_container(*arg_tuple, limit=5)
    run_file = arg_tuple[8]
    assert run_file.host_directory in lock_list and run_file.directory not in lock_list
    assert seen == [(77, run_file.stdout), (5, run_file.stdout)]
    # The broker is told the output directory and the agent, not the agent's copy
    (position, keyword) = start_list[0]
    assert position[1] == str(tmp_path / "out") and keyword["agent"] == "a"
    sandbox.tokencount.reset()


def test_a_run_whose_directory_of_files_leads_out_of_the_copy_starts_nothing(tmp_path,
                                                                             monkeypatch):
    arg_tuple, _ = _container_ready(tmp_path, monkeypatch, _never("the broker started"))
    run_file = arg_tuple[8]
    outside = tmp_path / "outside"
    outside.mkdir()
    os.makedirs(os.path.dirname(run_file.directory))
    os.symlink(outside, run_file.directory)
    with pytest.raises(RuntimeError, match="leads out of"):
        sandbox._start_container(*arg_tuple)
    assert os.listdir(outside) == [] and not os.path.exists(run_file.scratch)


@pytest.mark.parametrize("link", ["file", "directory", "copy"])
def test_a_file_that_crosses_is_read_and_written_only_inside_the_agent_s_copy(
        tmp_path, link):
    """The patch, its summary, the split, the outcome, the answer and the history are
    what a container writes; a link among them, or above them, is not followed."""
    outside = tmp_path / "outside"
    outside.mkdir()
    out = str(tmp_path / "out")
    run_file = sandbox.run_file_layout(out, "a1", "repo", "20260101-000000-1-42")
    history_path = os.path.join(run_file.out_dir, "repo", history.HISTORY_FILE)
    secret = {"verdict": "settled", "answer": "a secret", "tasks": ["a secret"],
              "exit_status": 0, "files": 3, "schema": 1,
              "attempts": [{"n": 9, "verdict": "a secret"}]}
    name_list = [os.path.basename(path) for path in (
        run_file.patch, sandbox.meta_path(run_file.patch), run_file.task,
        run_file.outcome, run_file.answer, run_file.target)]
    for name in (*name_list, history.HISTORY_FILE):
        (outside / name).write_text(json.dumps(secret), encoding="utf-8")
    if link == "copy":
        os.makedirs(run_file.out_dir)
        os.symlink(outside, os.path.join(run_file.out_dir, "repo"))
        (outside / "sandbox").symlink_to(outside)
    elif link == "directory":
        os.makedirs(os.path.dirname(run_file.directory))
        os.symlink(outside, run_file.directory)
        os.symlink(outside / history.HISTORY_FILE, history_path)
    else:
        os.makedirs(run_file.directory)
        for name in name_list:
            os.symlink(outside / name, os.path.join(run_file.directory, name))
        os.symlink(outside / history.HISTORY_FILE, history_path)
    before = {path.name: path.read_text(encoding="utf-8") for path in outside.iterdir()
              if path.is_file()}
    for read in (lambda: sandbox._read_split(run_file.task, run_file.out_dir),
                 lambda: sandbox._read_outcome(run_file.outcome, run_file.out_dir, "merge"),
                 lambda: sandbox._read_answer(run_file.answer, run_file.out_dir),
                 lambda: sandbox._read_meta(run_file.patch, run_file.out_dir),
                 lambda: sandbox._write_land_meta(run_file.patch, run_file.out_dir,
                                                  {"commit": "c"}),
                 lambda: sandbox._last_attempt(history_path, run_file.out_dir),
                 lambda: sandbox._own_record_dict(history_path, run_file.out_dir, 0),
                 lambda: sandbox._land_patch("/repo", run_file, "b", "codetwine-loom/x", 0, True,
                                             lambda meta_dict: "m"),
                 lambda: sandbox.write_across(
                     sandbox.inside_path(run_file.target, run_file.out_dir), {}, "t")):
        with pytest.raises(RuntimeError, match="leads out of"):
            read()
    assert {path.name: path.read_text(encoding="utf-8") for path in outside.iterdir()
            if path.is_file()} == before


def test_a_path_inside_the_agent_s_copy_is_handed_back_as_it_came(tmp_path):
    own_dir = tmp_path / "out" / "agents" / "a1"
    (own_dir / "repo" / "sandbox").mkdir(parents=True)
    (own_dir / "repo" / "sandbox" / "real.json").write_text("{}", encoding="utf-8")
    (own_dir / "repo" / "sandbox" / "inner.json").symlink_to("real.json")
    for name in ("real.json", "inner.json", "not-there.json"):
        path = str(own_dir / "repo" / "sandbox" / name)
        assert sandbox.inside_path(path, str(own_dir)) == path
    assert sandbox.inside_path(str(own_dir), str(own_dir)) == str(own_dir)
    with pytest.raises(RuntimeError, match="leads out of"):
        sandbox.inside_path(str(tmp_path / "out" / "agents" / "a12" / "x"), str(own_dir))


def test_an_index_reached_through_a_link_out_of_the_copy_describes_nothing(tmp_path,
                                                                           monkeypatch):
    repo = _repo(tmp_path / "repo")
    commit = gitcmd.require_head(str(repo))
    elsewhere = _index_at(tmp_path / "elsewhere", monkeypatch, repo, commit=commit)
    assert sandbox.index_describe(str(repo), elsewhere, "repo", commit) is True
    own_dir = tmp_path / "out" / "agents" / "a1"
    own_dir.mkdir(parents=True)
    (own_dir / "repo").symlink_to(pathlib.Path(elsewhere) / "repo")
    assert sandbox.index_describe(str(repo), str(own_dir), "repo", commit) is False


def test_the_summary_of_a_patch_is_taken_only_as_whole_numbers(tmp_path, monkeypatch):
    """A container writes the summary, so what it says is held to the shape read."""
    run_file = sandbox.run_file_layout(str(tmp_path / "out"), "a1", "repo",
                                       "20260101-000000-1-42")
    os.makedirs(run_file.directory)
    for meta in ({"exit_status": "0", "files": [1]}, ["not", "an", "object"]):
        pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(json.dumps(meta),
                                                                   encoding="utf-8")
        pathlib.Path(run_file.patch).write_text("", encoding="utf-8")
        assert sandbox._land_patch("/repo", run_file, "b", "codetwine-loom/x", 7, True,
                                   lambda meta_dict: "m") == {
            "patch": None, "files": 0, "exit_status": 7, "branch": None, "commit": None,
            "no_patch": isinstance(meta, list),
            "no_patch_why": "the container ended (exit status 7) before it wrote a patch "
                            "and the summary beside it" if isinstance(meta, list) else "",
            "unseen_text": ""}


@pytest.mark.parametrize("meta_dict, status, want_dict", [
    ({"exit_status": 256, "files": -3}, 0, {"exit_status": 1}),
    ({"exit_status": 256, "files": 2}, 5, {"exit_status": 5, "files": 2}),
    ({"exit_status": -1}, 0, {"exit_status": 1}),
    ({"exit_status": True, "files": True}, 0, {"exit_status": 1}),
    ({"exit_status": 10 ** 30}, 0, {"exit_status": 1}),
    ({"exit_status": None, "files": 1.5}, 0, {"exit_status": 1}),
    ({"exit_status": 255, "files": 0}, 0, {"exit_status": 255, "files": 0}),
    ({"exit_status": 0, "files": 3}, 9, {"exit_status": 9, "files": 3}),
    ({"exit_status": 3, "files": 3}, 137, {"exit_status": 3, "files": 3}),
    ({"exit_status": 3, "files": 3}, 0, {"exit_status": 3, "files": 3}),
    ({"exit_status": 0, "files": 3}, 0, {"exit_status": 0, "files": 3}),
    ({"files": 2, "nested": ["x"]}, 0, {"files": 2, "nested": ["x"]}),
    ({"files": 2}, 7, {"files": 2}),
])
def test_the_numbers_of_a_summary_are_held_to_what_a_process_ends_with(
        tmp_path, meta_dict, status, want_dict):
    """Another value under "exit_status" is read as a run that did not get through,
    and a 0 there beside a container that ended with another status is that status."""
    run_file = sandbox.run_file_layout(str(tmp_path / "out"), "a1", "repo",
                                       "20260101-000000-1-42")
    os.makedirs(run_file.directory)
    pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(json.dumps(meta_dict),
                                                               encoding="utf-8")
    pathlib.Path(run_file.patch).write_text("", encoding="utf-8")
    result = sandbox._land_patch("/repo", run_file, "b", "codetwine-loom/x", status, True,
                                 lambda meta_dict: "m")
    assert (result["exit_status"], result["files"]) == (
        want_dict.get("exit_status", status), want_dict.get("files", 0))
    sandbox._hold_meta_number(meta_dict, status)
    assert meta_dict == want_dict


def test_a_patch_over_the_size_taken_in_is_no_patch_and_a_failure(tmp_path, monkeypatch,
                                                                 caplog):
    monkeypatch.setattr(sandbox, "_MAX_PATCH_BYTE", 16)
    run_file = sandbox.run_file_layout(str(tmp_path / "out"), "a1", "repo",
                                       "20260101-000000-1-42")
    os.makedirs(run_file.directory)
    pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(
        '{"exit_status": 0, "files": 1}', encoding="utf-8")
    pathlib.Path(run_file.patch).write_text("diff --git a/a b/a\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        assert sandbox._land_patch("/repo", run_file, "b", "codetwine-loom/x", 0, True,
                                   _never("a message was built")) == {
            "patch": None, "files": 0, "exit_status": 1, "branch": None, "commit": None,
            "no_patch": True,
            "no_patch_why": f"the patch the run wrote holds 19 bytes, more than the 16 "
                            f"that are taken in. It stays at {run_file.patch}",
            "unseen_text": ""}
    assert "nothing of what the run changed is taken in: the patch the run wrote holds " \
           "19 bytes, more than the 16 that are taken in" in caplog.text


@pytest.mark.parametrize("patch_text", ["diff --git a/a b/a\n", ""])
def test_a_patch_whose_summary_names_an_unseen_directory_is_not_taken_in(
        tmp_path, caplog, patch_text):
    """Files the run wrote into a submodule directory with no repository keep the
    whole of what it changed out, and a dry run still hands the patch back."""
    run_file = sandbox.run_file_layout(str(tmp_path / "out"), "a1", "repo",
                                       "20260101-000000-1-42")
    os.makedirs(run_file.directory)
    pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(
        '{"exit_status": 0, "files": 1, "unseen": ["sub", 3]}', encoding="utf-8")
    pathlib.Path(run_file.patch).write_text(patch_text, encoding="utf-8")
    unseen_text = ("what the run wrote into 1 submodule directory(ies) of its clone "
                   "that git shows as their commit alone: sub")
    # A patch that holds nothing is not named as a place where something stays
    stay_text = f". It stays at {run_file.patch}" if patch_text else \
        ", and it holds nothing else"
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        assert sandbox._land_patch("/repo", run_file, "b", "codetwine-loom/x", 0, True,
                                   _never("a message was built")) == {
            "patch": None, "files": 1, "exit_status": 1, "branch": None, "commit": None,
            "no_patch": True,
            "no_patch_why": f"the patch does not hold {unseen_text}{stay_text}",
            "unseen_text": unseen_text}
    assert "the run wrote files into 1 submodule directory(ies) of its clone that git " \
           "shows as their commit alone, and they are not in the patch: sub" \
        in caplog.text
    assert f"nothing of what the run changed is taken in: the patch does not hold " \
           f"{unseen_text}{stay_text}" in caplog.text
    dry_dict = sandbox._land_patch("/repo", run_file, "b", "codetwine-loom/x", 0, False,
                                   _never("a message was built"))
    assert (dry_dict["no_patch"], dry_dict["no_patch_why"], dry_dict["unseen_text"]) \
        == (False, "", unseen_text)


def test_no_patch_and_an_empty_patch_are_said_apart_and_what_is_not_in_it_is_named(
        tmp_path, caplog):
    run_file = sandbox.run_file_layout(str(tmp_path / "out"), "a1", "repo",
                                       "20260101-000000-1-42")
    os.makedirs(run_file.directory)
    land = lambda: sandbox._land_patch("/repo", run_file, "b", "codetwine-loom/x",  # noqa: E731
                                       137, True, lambda meta_dict: "m")
    # The container ended before it wrote anything
    with caplog.at_level(logging.INFO, logger="codetwine_loom.sandbox"):
        assert land() == {"patch": None, "files": 0, "exit_status": 137, "branch": None,
                          "commit": None, "no_patch": True,
                          "no_patch_why": f"the container ended (exit status 137, the "
                                          f"status of a container killed when its "
                                          f"processes reach SANDBOX_MEMORY "
                                          f"({config.SANDBOX_MEMORY})) before it wrote a "
                                          f"patch and the summary beside it",
                          "unseen_text": ""}
    assert "ended (exit status 137, the status of a container killed" in caplog.text
    assert "changed nothing" not in caplog.text
    caplog.clear()
    # The run changed nothing git could add, and says what it could not
    pathlib.Path(run_file.patch).write_text("", encoding="utf-8")
    pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(json.dumps({
        "exit_status": 0, "files": 0, "unadded": ["tmp_repo/", '"a\\"b.txt"', 7],
        "nested": ["vendor/lib"], "moved_head": "c" * 40}), encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="codetwine_loom.sandbox"):
        assert land() == {"patch": None, "files": 0, "exit_status": 137, "branch": None,
                          "commit": None, "no_patch": False, "no_patch_why": "",
                          "unseen_text": ""}
    warning_list = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_list == [
        "the summary beside the patch names 0 as the run's exit status and its "
        "container ended with 137, so the run is read as one that ended with 137",
        "git could not add 2 path(s) of the run's clone, so they are not in the patch: "
        'tmp_repo/, "a\\"b.txt"',
        "the run left 1 repository(ies) of their own inside its clone, and the files in "
        "them are not in the patch: vendor/lib",
        "the run committed in its clone (HEAD ended at cccccccccccc), and what it "
        "committed is in the patch with the rest of what it changed"]
    assert "the run changed nothing" in caplog.text
    assert "before it wrote a patch" not in caplog.text


@pytest.mark.parametrize("leave", ["nothing", "patch alone", "pipe", "directory",
                                   "summary alone", "summary unread"])
def test_a_patch_or_a_summary_that_is_not_there_is_no_patch_and_a_failure(tmp_path, leave):
    """The summary is written after the patch, so a run without both did not finish."""
    run_file = sandbox.run_file_layout(str(tmp_path / "out"), "a1", "repo",
                                       "20260101-000000-1-42")
    os.makedirs(run_file.directory)
    meta_path = pathlib.Path(sandbox.meta_path(run_file.patch))
    if leave in ("patch alone", "summary unread"):
        pathlib.Path(run_file.patch).write_text("diff --git a/a b/a\n", encoding="utf-8")
    if leave == "pipe":
        os.mkfifo(run_file.patch)
    if leave == "directory":
        os.mkdir(run_file.patch)
    if leave in ("pipe", "directory", "summary alone"):
        meta_path.write_text('{"exit_status": 0, "files": 1}', encoding="utf-8")
    if leave == "summary unread":
        os.mkfifo(meta_path)
    stay = f". What it wrote of the patch stays at {run_file.patch}" \
        if leave in ("patch alone", "summary unread") else ""
    assert sandbox._land_patch("/repo", run_file, "b", "codetwine-loom/x", 0, True,
                               _never("a message was built")) == {
        "patch": None, "files": 0, "exit_status": 1, "branch": None, "commit": None,
        "no_patch": True,
        "no_patch_why": f"the container ended (exit status 0) before it wrote a patch and "
                        f"the summary beside it{stay}",
        "unseen_text": ""}


@pytest.mark.parametrize("text", [
    '{"verdict": "settled", "decisions": "x"}', '{"decisions": [1]}',
    '{"verdict": {"a": 1}}', '{"skipped": ["x"]}', '{"non_test_changed": 5}',
    '{"tests_changed": [1]}', '{"gap": [{"what": "c"}]}', '{"cases": 5}',
    '{"targets": {"x.py": 3}}', '{"tests": {"x.py": [3]}}', '["verdict"]', '"settled"',
    '{"test_open": 5}', '{"test_open": [5]}', "[" * 200000])
def test_an_outcome_of_another_shape_is_read_as_no_outcome(tmp_path, text):
    """A container writes the file, so every value read from it is held to its shape."""
    path = tmp_path / "outcome.json"
    path.write_text(text, encoding="utf-8")
    assert sandbox._read_outcome(str(path), str(tmp_path), "test-create") == {}


def test_an_outcome_holding_another_key_is_read_as_it_is(tmp_path):
    path = tmp_path / "outcome.json"
    outcome_dict = {"verdict": "settled", "decisions": [], "parents": 5, "other": None,
                    "gap": [{"what": "c", "why": "w", "test_path": "t.py"}],
                    "targets": {"x.py": ["f"]}, "tests": {"x.py": ["t.py"]}}
    path.write_text(json.dumps(outcome_dict), encoding="utf-8")
    assert sandbox._read_outcome(str(path), str(tmp_path), "test-create") == outcome_dict


def test_a_null_under_a_key_this_machine_reads_is_read_as_the_key_not_there(tmp_path):
    """Every reader then takes its own default: no list is read out of a null."""
    path = tmp_path / "outcome.json"
    null_dict = dict.fromkeys(sandbox._RECORD_SHAPE_DICT)
    path.write_text(json.dumps({**null_dict, "verdict": "settled"}), encoding="utf-8")
    assert sandbox._read_outcome(str(path), str(tmp_path), "test-create") == \
        {"verdict": "settled"}
    record_dict = sandbox._stage_record_dict(
        "test-create", sandbox._read_outcome(str(path), str(tmp_path), "test-create"), 0)
    assert record_dict["skipped"] == "" and "cases" not in record_dict
    # The trailers are built from it as from an outcome without those keys
    assert "Codetwine-Loom-Gap" not in sandbox.commit_message(
        "r", sandbox.run_trailer_dict({}, record_dict, "r"))


@pytest.mark.parametrize("make", [os.mkfifo, os.mkdir])
def test_a_file_a_container_left_that_is_no_regular_file_is_not_opened(tmp_path, make):
    """A named pipe is not waited on, and a directory is not an error of its own."""
    run_file = sandbox.run_file_layout(str(tmp_path / "out"), "a1", "repo",
                                       "20260101-000000-1-42")
    os.makedirs(os.path.join(run_file.out_dir, "repo"))
    os.makedirs(run_file.directory)
    history_path = os.path.join(run_file.out_dir, "repo", history.HISTORY_FILE)
    for path in (run_file.task, run_file.answer, run_file.outcome,
                 sandbox.meta_path(run_file.patch), history_path):
        make(path)
    with pytest.raises(RuntimeError, match="is not a regular file"):
        sandbox._read_split(run_file.task, run_file.out_dir)
    with pytest.raises(RuntimeError, match="is not a regular file"):
        sandbox._read_answer(run_file.answer, run_file.out_dir)
    assert sandbox._read_outcome(run_file.outcome, run_file.out_dir, "merge") == {}
    assert sandbox._read_meta(run_file.patch, run_file.out_dir) == {}
    assert sandbox._last_attempt(history_path, run_file.out_dir) == {}


def test_a_history_entry_of_another_shape_is_not_what_the_commit_records(tmp_path):
    out_dir = tmp_path / "out"
    (out_dir / "repo").mkdir(parents=True)
    history_path = out_dir / "repo" / history.HISTORY_FILE
    at = "2026-09-01T00:00:00"
    for entry_dict, expect in (
            ({"n": 2, "at": at, "verdict": "failed", "untooled": 5}, {}),
            ({"n": 2, "at": at, "verdict": ["failed"]}, {}),
            ({"n": 2, "verdict": "failed"}, {}),
            ({"n": 2, "at": at, "verdict": "passed", "untooled": ["a.py"]}, {}),
            ({"n": 2, "at": at, "verdict": "failed", "untooled": ["a.py"]},
             {"n": 2, "at": at, "verdict": "failed", "untooled": ["a.py"]})):
        history_path.write_text(json.dumps(
            {"schema": history.SCHEMA, "project": "repo", "attempts": [entry_dict]}),
            encoding="utf-8")
        assert sandbox._own_record_dict(str(history_path), str(out_dir), 1) == expect


def test_what_this_machine_kept_for_an_agent_goes_with_the_copy_when_it_holds_nothing(
        tmp_path):
    out, _ = _built(tmp_path)
    for agent, kept in (("codetwine-loom-none", ""),
                        ("codetwine-loom-land", "land-20260101-000000-1-42.diff")):
        sandbox.make_agent_dir(str(out), "repo", agent, "", None, False)
        run_file = sandbox.run_file_layout(str(out), agent, "repo", "20260101-000000-1-42")
        os.makedirs(run_file.host_directory)
        pathlib.Path(sandbox.filelock.path_for(run_file.host_directory)).write_text("")
        if kept:
            pathlib.Path(run_file.host_directory, kept).write_text("d", encoding="utf-8")
        sandbox.drop_the_copy(str(out), "repo", agent)
    assert not os.path.exists(sandbox.host_dir(str(out), "codetwine-loom-none"))
    assert os.listdir(sandbox.host_dir(str(out), "codetwine-loom-land") + "/repo/sandbox") == [
        "land-20260101-000000-1-42.diff"]
    assert (out / "agents" / ".host").is_dir()


def test_the_container_is_told_where_the_split_goes_in_its_own_paths(_dispatch_repo,
                                                                    monkeypatch):
    repo, out = _dispatch_repo
    files = sandbox.run_file_layout(_make_agent(out), "20260101-000000-1-42", "repo",
                                    "20260101-000000-1-42")
    told = []

    def start(*a, **k):
        told.append(a[-2])
        return _split_written(files.task, {"questions": ["where?"]})(*a, **k)

    monkeypatch.setattr(sandbox, "_start_container", start)
    monkeypatch.setattr(sandbox, "answer", _answered_with("it is a.py"))
    sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert files.task_inside in told[0]


@pytest.mark.parametrize("flag", [True, False])
def test_the_overview_is_made_before_the_agents_copy_the_output_dir(_dispatch_repo,
                                                                    monkeypatch, flag):
    repo, out = _dispatch_repo
    event_list = []
    monkeypatch.setattr(sandbox, "make_overview",
                        lambda *a: event_list.append(("overview", a[:2])))
    make = sandbox.make_agent_dir
    monkeypatch.setattr(sandbox, "make_agent_dir",
                        lambda *a, **k: event_list.append(("copy",)) or make(*a, **k))
    make(str(out), "repo", "20260101-000000-1-42", "", None, False)
    files = sandbox.run_file_layout(str(out), "20260101-000000-1-42", "repo",
                                    "20260101-000000-1-42")
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(files.task, {"questions": ["where?"]}))
    monkeypatch.setattr(sandbox, "answer", _answered_with("it is a.py"))
    monkeypatch.setattr(sandbox.overview, "is_on", lambda: flag)
    sandbox.dispatch(str(repo), str(out), "どこ?")
    if flag:
        assert event_list[0] == ("overview", (str(repo), str(out)))
        assert event_list[1] == ("copy",)
    else:
        assert all(e[0] != "overview" for e in event_list)


def _make_overview_with(monkeypatch, tmp_path, build_status=0, current=False,
                      container=None):
    """Call make_overview() with the build and the container stood in for.

    Args:
        build_status: The exit status of the build, or None for one that runs past
            its time limit.
        current: What overview.is_current() answers.
        container: Called with _start_container()'s arguments; it returns the
            container's exit status. None fails the test when a container is started.

    Returns:
        (the build's calls as (argv, kwargs), the containers' arguments, the output
        directory).
    """
    build_list, container_list = [], []
    out_dir = str(tmp_path / "out")

    def run(argv, **kwargs):
        build_list.append((argv, kwargs))
        if build_status is None:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, build_status)

    def start(*arg_tuple, **kwargs):
        container_list.append((arg_tuple, kwargs))
        if container is None:
            raise AssertionError("a container was started")
        return "/usr/bin/podman", "img", container(*arg_tuple)

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    monkeypatch.setattr(sandbox, "_start_container", start)
    monkeypatch.setattr(sandbox.overview, "index_digest", lambda *a: "d1")
    monkeypatch.setattr(sandbox.overview, "is_current", lambda *a: current)
    monkeypatch.setattr(sandbox.gitcmd, "head_commit", lambda project_dir: "c" * 40)
    monkeypatch.setenv(sandbox.overview.RUN_DIGEST_ENV, "left by a try before")
    monkeypatch.setenv("LLM_API_KEY", "the key")
    sandbox.make_overview("/repo", out_dir, "repo", ["vendor", "a b"], "b" * 40)
    return build_list, container_list, out_dir


def _leave_overview(answer="what it is for", question=None):
    """Return a stand-in for the container that leaves an overview in its copy."""
    def container(*arg_tuple):
        own_dir = arg_tuple[4]
        path = pathlib.Path(sandbox.overview.overview_path(own_dir, "repo"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": 3, "commit": "x", "file_digest": "the container's own",
            "question": sandbox.overview.QUESTION if question is None else question,
            "answer": answer}), encoding="utf-8")
        return 0

    return container


def test_the_overview_is_made_in_a_container_after_a_build_with_no_credential(
        monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(config, "OVERVIEW_TIMEOUT_SEC", 50)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        build_list, container_list, out_dir = _make_overview_with(
            monkeypatch, tmp_path, container=_leave_overview())
    # The build runs on this machine without the credentials and without a terminal
    (argv, kwargs), = build_list
    assert argv == config.own_argv(["build", "--project", "/repo", "--output-dir",
                                    out_dir])
    assert shlex.split(kwargs["env"]["BUILD_EXCLUDE"]) == ["vendor", "a b"]
    assert "LLM_API_KEY" not in kwargs["env"]
    # It reads no .env file either: what it holds is what it is handed
    assert kwargs["env"][config.NO_ENV_FILE_ENV] == "1"
    assert 0 < kwargs["timeout"] <= 50 and kwargs["stdin"] == subprocess.DEVNULL
    # The overview is made by a container of its own, started as every other is
    (arg_tuple, start_kwargs), = container_list
    assert arg_tuple[3] == out_dir and arg_tuple[7] == "b" * 40
    assert arg_tuple[4] == sandbox.agent_dir(out_dir, arg_tuple[6])
    assert arg_tuple[10] == ["overview", "--project", "/work/repo", "--output-dir", "/out"]
    assert arg_tuple[11] == ["vendor", "a b"] and 0 < start_kwargs["limit"] <= 50
    # What it left is written into the output directory with the digest read here
    record = sandbox.overview.read_record(out_dir, "repo")
    assert record["answer"] == "what it is for" and record["file_digest"] == "d1"
    assert record["question"] == sandbox.overview.QUESTION
    assert os.environ.get(sandbox.overview.RUN_DIGEST_ENV) == "d1"
    assert "overview" not in caplog.text
    # The copy made for the container is gone
    assert not os.path.exists(arg_tuple[4])


@pytest.mark.parametrize("is_closed", [False, True])
def test_the_lines_the_build_prints_go_to_stderr_or_nowhere(monkeypatch, tmp_path,
                                                          is_closed):
    """A process started with stderr closed has no stderr to hand the build, and its
    stdout is not what the build's lines are written to."""
    if is_closed:
        monkeypatch.setattr(sandbox.sys, "stderr", None)
    stream = sandbox.sys.stderr
    build_list, _, _ = _make_overview_with(monkeypatch, tmp_path, current=True)
    (_, kwargs), = build_list
    assert kwargs["stdout"] == (subprocess.DEVNULL if is_closed else stream)
    assert kwargs["stdout"] is not None and kwargs["stdin"] == subprocess.DEVNULL


def test_an_overview_that_is_current_starts_no_container(monkeypatch, tmp_path):
    build_list, container_list, _ = _make_overview_with(monkeypatch, tmp_path, current=True)
    assert len(build_list) == 1 and container_list == []
    assert os.environ.get(sandbox.overview.RUN_DIGEST_ENV) == "d1"


@pytest.mark.parametrize("build_status, container, warn", [
    (1, None, "ended with status 1 while the analysis was made"),
    (None, None, "was stopped at OVERVIEW_TIMEOUT_SEC"),
    (0, lambda *a: 1, "ended with status 1"),
    (0, lambda *a: 124, "was stopped at OVERVIEW_TIMEOUT_SEC"),
    (0, lambda *a: 0, "left no overview"),
    (0, _leave_overview(question="another question"), "left no overview"),
    (0, lambda *a: (_ for _ in ()).throw(RuntimeError("no engine")),
     "could not be done in a container (no engine)"),
    (0, lambda *a: (_ for _ in ()).throw(OSError(28, "No space left on device")),
     "could not be done in a container (OSError: [Errno 28] No space left on device)"),
    (0, lambda *a: (_ for _ in ()).throw(ValueError("embedded null byte")),
     "could not be done in a container (ValueError: embedded null byte)")])
def test_an_overview_that_could_not_be_made_is_logged_and_the_run_goes_on(
        monkeypatch, tmp_path, caplog, build_status, container, warn):
    monkeypatch.setattr(config, "OVERVIEW_TIMEOUT_SEC", 5)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        _, container_list, out_dir = _make_overview_with(
            monkeypatch, tmp_path, build_status=build_status, container=container)
    assert warn in caplog.text and "the run goes on" in caplog.text
    assert os.environ.get(sandbox.overview.RUN_DIGEST_ENV) is None
    assert sandbox.overview.read_record(out_dir, "repo") == {}
    assert len(container_list) == (0 if build_status != 0 else 1)


@pytest.mark.parametrize("status", [1, 124])
def test_an_overview_a_container_wrote_before_it_ended_with_another_status_is_taken(
        monkeypatch, tmp_path, caplog, status):
    leave = _leave_overview()
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        _, _, out_dir = _make_overview_with(
            monkeypatch, tmp_path, container=lambda *arg_tuple: leave(*arg_tuple) or status)
    record = sandbox.overview.read_record(out_dir, "repo")
    assert record["answer"] == "what it is for" and record["file_digest"] == "d1"
    assert os.environ.get(sandbox.overview.RUN_DIGEST_ENV) == "d1"
    assert f"ended with exit status {status} after it wrote the overview" in caplog.text
    assert "the run goes on" not in caplog.text


@pytest.mark.parametrize("status, warn", [(1, "ended with status 1"),
                                          (124, "was stopped at OVERVIEW_TIMEOUT_SEC")])
def test_an_overview_the_copy_held_before_the_container_is_none_the_container_wrote(
        monkeypatch, tmp_path, caplog, status, warn):
    """The output directory holds an overview made over other files, which the
    container's copy takes; the container ends without writing one."""
    old_dict = {"schema": 3, "commit": "x", "file_digest": "d0",
                "question": sandbox.overview.QUESTION, "answer": "an older one"}
    path = pathlib.Path(sandbox.overview.overview_path(str(tmp_path / "out"), "repo"))
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(old_dict), encoding="utf-8")
    monkeypatch.setattr(config, "OVERVIEW_TIMEOUT_SEC", 5)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        _, container_list, out_dir = _make_overview_with(
            monkeypatch, tmp_path, container=lambda *arg_tuple: status)
    assert len(container_list) == 1
    assert warn in caplog.text and "the run goes on" in caplog.text
    assert sandbox.overview.read_record(out_dir, "repo")["file_digest"] == "d0"
    assert os.environ.get(sandbox.overview.RUN_DIGEST_ENV) is None


def test_an_overview_left_as_a_link_out_of_the_copy_is_not_read(monkeypatch, tmp_path,
                                                             caplog):
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps({"schema": 3, "question": sandbox.overview.QUESTION,
                                  "file_digest": "d", "answer": "a secret"}),
                      encoding="utf-8")

    def container(*arg_tuple):
        path = pathlib.Path(sandbox.overview.overview_path(arg_tuple[4], "repo"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(secret)
        return 0

    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        _, _, out_dir = _make_overview_with(monkeypatch, tmp_path, container=container)
    assert "leads out of" in caplog.text
    assert sandbox.overview.read_record(out_dir, "repo") == {}


def test_the_overview_taken_from_a_container_is_held_to_the_answer_s_limit(
        monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 10)
    _, _, out_dir = _make_overview_with(monkeypatch, tmp_path,
                                      container=_leave_overview(answer="a" * 500))
    assert sandbox.overview.read_record(out_dir, "repo")["answer"] == \
        sandbox.cut("a" * 500, 10)


def test_every_container_is_handed_the_state_the_run_started_from(monkeypatch):
    monkeypatch.delenv(sandbox.overview.RUN_DIGEST_ENV, raising=False)
    assert not any(a.startswith(sandbox.overview.RUN_DIGEST_ENV) for a in _argv())
    monkeypatch.setenv(sandbox.overview.RUN_DIGEST_ENV, "d1")
    assert f"{sandbox.overview.RUN_DIGEST_ENV}=d1" in _argv()


def test_the_container_is_asked_to_read_the_request_here():
    argv = sandbox.dispatch_argv("repo", "設定を整理して", "/out/repo/sandbox/t.json")
    assert argv[:2] == ["run", "--here"]
    assert "--request=設定を整理して" in argv and "/out/repo/sandbox/t.json" in argv
    # The clone, never the read-only mount of the repository
    assert f"{sandbox._WORK}/repo" in argv


def _split_written(path: str, split: dict):
    """Return a _start_the_container stand-in that writes the split the test wants."""
    def start(*a, **k):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(split, handle)
        return "/usr/bin/podman", "img", 0

    return start


@pytest.fixture
def _dispatch_repo(tmp_path, monkeypatch):
    """A repository with one commit, and a run id that does not move."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one")
    monkeypatch.setattr(sandbox, "new_run_id", lambda: "20260101-000000-1-42")
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 1)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 0)
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 0)
    return repo, tmp_path / "out"


def _tasks_file(out):
    """Return where the split of the run _dispatch_repo pins is written."""
    return sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task


def test_the_branches_to_carry_on_from_reach_the_reading_run(_dispatch_repo,
                                                             monkeypatch):
    """The reading run reads them in its clone, which reaches them as the branches of
    what it cloned."""
    repo, out = _dispatch_repo
    _git(repo, "branch", "codetwine-loom/run/old/1")
    told = []

    def start(*a, **k):
        told.append(a[-2])
        return _split_written(_tasks_file(out), {"questions": ["where?"]})(*a, **k)

    monkeypatch.setattr(sandbox, "_start_container", start)
    monkeypatch.setattr(sandbox, "answer", _answered_with("it is a.py"))
    sandbox.dispatch(str(repo), str(out), "続きを", resume_list=["codetwine-loom/run/old/1"])
    assert told[0][-2:] == ["--resume", "codetwine-loom/run/old/1"]


def test_a_branch_named_before_the_rename_is_carried_on_from_as_it_is(_dispatch_repo,
                                                                     monkeypatch):
    """--resume reads the name it is given: a dispatch/... branch an earlier version
    left is read like any other."""
    repo, out = _dispatch_repo
    _git(repo, "branch", "dispatch/old/1")
    told = []

    def start(*a, **k):
        told.append(a[-2])
        return _split_written(_tasks_file(out), {"questions": ["where?"]})(*a, **k)

    monkeypatch.setattr(sandbox, "_start_container", start)
    monkeypatch.setattr(sandbox, "answer", _answered_with("it is a.py"))
    sandbox.dispatch(str(repo), str(out), "続きを", resume_list=["dispatch/old/1"])
    assert told[0][-2:] == ["--resume", "dispatch/old/1"]


def test_a_branch_that_is_not_there_stops_the_run_before_it_is_paid_for(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _never("a container was started for a branch that is not there"))
    with pytest.raises(RuntimeError, match="no branch named"):
        sandbox.dispatch(str(repo), str(out), "続きを", resume_list=["codetwine-loom/run/gone/1"])


def test_the_first_analysis_is_not_made_for_a_run_the_entrance_refuses(
        _dispatch_repo, monkeypatch):
    """The first analysis can cost the documents and a question on the terminal, so
    it comes after every check that can stop the run before anything is paid for."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "analyse_if_none",
                        _never("the repository was analysed for a run that stops"))
    monkeypatch.setattr(sandbox, "_start_container", _never("a container was started"))
    with pytest.raises(RuntimeError, match="no branch named"):
        sandbox.dispatch(str(repo), str(out), "続きを", resume_list=["codetwine-loom/run/gone/1"])
    assert not (out / "agents").exists()
    assert _git(repo, "branch", "--list", "codetwine-loom/run/*").strip() == ""


def test_a_dispatching_run_starts_every_container_from_the_working_tree(
        _dispatch_repo, monkeypatch):
    """The reading run and the agents are given the commit the tree makes, and the
    branch pointing at it is there for them and gone afterwards."""
    repo, out = _dispatch_repo
    (repo / "draft.py").write_text("y = 2\n", encoding="utf-8")
    seen = {}

    def start(*a, **k):
        seen["base"] = a[7]
        seen["branches"] = _git(repo, "branch", "--list", "codetwine-loom/run/*").split()
        return _split_written(_tasks_file(out), {"tasks": ["one"]})(*a, **k)

    monkeypatch.setattr(sandbox, "_start_container", start)
    run, run_list, _ = _starting_where_told(repo)
    monkeypatch.setattr(sandbox, "run", run)
    called = []
    monkeypatch.setattr(sandbox, "integrate", _combining_stand_in(called))
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    sandbox.dispatch(str(repo), str(out), "設定を整理して")
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert seen["base"] != head and _git(repo, "show", f"{seen['base']}:draft.py") == "y = 2\n"
    assert seen["branches"] == ["codetwine-loom/run/20260101-000000-1-42/tree"]
    assert run_list[0]["start"] == seen["base"]
    assert _git(repo, "branch", "--list", "codetwine-loom/run/*/tree").strip() == ""
    assert _git(repo, "rev-parse", "HEAD").strip() == head
    assert (repo / "draft.py").read_text(encoding="utf-8") == "y = 2\n"


def test_the_first_analysis_is_not_made_for_a_merge_the_entrance_refuses(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    monkeypatch.setattr(sandbox, "analyse_if_none",
                        _never("the repository was analysed for a run that stops"))
    monkeypatch.setattr(sandbox, "_start_container", _never("a container was started"))
    with pytest.raises(RuntimeError, match="no branch named"):
        sandbox.integrate(str(repo), str(tmp_path / "out"), ["gone/a", "gone/b"])
    assert not (tmp_path / "out" / "agents").exists()


def test_the_branches_a_run_left_are_what_it_says_to_carry_on_from(_dispatch_repo,
                                                                   monkeypatch):
    """A task nobody finished is only on its own branch, so that is what the next run
    is told to read."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 1, "verdict": "failed"})
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["resume"] == ["codetwine-loom/run/20260101-000000-1-42/1",
                                "codetwine-loom/run/20260101-000000-1-42/2"]


def test_a_merge_that_was_not_settled_is_the_one_branch_to_carry_on_from(
        _dispatch_repo, monkeypatch):
    """It reaches every task's commit, so its own name says everything they hold."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 0, "verdict": "unverified"})
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: {"exit_status": 1, "commit": "u" * 40,
                                         "branch": "codetwine-loom/combined"})
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["resume"] == ["codetwine-loom/combined"]


def test_a_run_that_got_through_leaves_nothing_to_carry_on_from(_dispatch_repo,
                                                                monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 0, "verdict": "unverified"})
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: {"exit_status": 0, "commit": "u" * 40,
                                         "branch": "codetwine-loom/combined"})
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["resume"] == []


def test_a_task_that_landed_nothing_is_not_named_to_carry_on_from(_dispatch_repo,
                                                                  monkeypatch):
    """A patch is what makes the branch, so an agent that wrote none left no branch to
    read, and naming it would stop the next run before it started."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40 if k["branch"].endswith("/1") else "",
        "exit_status": 1, "verdict": "failed"})
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["resume"] == ["codetwine-loom/run/20260101-000000-1-42/1"]


def test_one_task_that_got_through_leaves_nothing_to_carry_on_from(_dispatch_repo,
                                                                   monkeypatch):
    """Nothing is combined when there is one branch, and that is not the run having
    left something: it got through, so there is nothing to read again."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 0, "verdict": "unverified"})
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
                            "exit_status": 0, "commit": "u" * 40,
                            "branch": "codetwine-loom/sandbox/u"})
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["resume"] == [] and result["exit_status"] == 0


# == the tasks that name the same definition run one after another ==

def test_the_keys_joined_through_a_value_they_share_are_one_set():
    assert sandbox.join_set_list({"a": ["t1"], "b": [], "c": ["t1"], "d": ["t2"],
                                  "e": ["t2", "t3"], "f": ["t3"]}) == \
        [["a", "c"], ["b"], ["d", "e", "f"]]


def _run_chain_in_threads(chain_dict, member_list, start_order, attempt, ok_set):
    """Start one thread per task in start_order and return the order they ran in."""
    ran_list = []

    def one(n):
        def run(start):
            ran_list.append((n, start))
            return {"commit": f"c{n}", "branch": f"b{n}",
                    "exit_status": 0 if n in ok_set else 1}
        return sandbox.run_in_chain(chain_dict, member_list, n, attempt, run,
                                    lambda e: not e["exit_status"], f"{n} waits")

    thread_list = []
    for n in start_order:
        thread = threading.Thread(target=one, args=(n,))
        thread.start()
        thread_list.append(thread)
        time.sleep(0.05)
    for thread in thread_list:
        thread.join(timeout=5)
    return ran_list


def test_the_tasks_of_a_chain_run_in_their_order_whichever_starts_first():
    chain_dict = sandbox.new_chain_dict("base")
    ran_list = _run_chain_in_threads(chain_dict, [1, 3, 4], [4, 3, 1], 1, {1, 3, 4})
    # Each starts at the commit the one before it left
    assert ran_list == [(1, "base"), (3, "c1"), (4, "c3")]


def test_a_try_again_waits_only_for_the_tasks_run_again_before_it():
    chain_dict = sandbox.new_chain_dict("base")
    # Task 2 did not finish: 3 still starts at 1's commit, and 2 is run again
    assert _run_chain_in_threads(chain_dict, [1, 2, 3], [1, 2, 3], 1, {1, 3}) == [
        (1, "base"), (2, "c1"), (3, "c1")]
    # The second try runs 2 alone, and 1 and 3, which finished, are not waited for
    assert _run_chain_in_threads(chain_dict, [1, 2, 3], [2], 2, {2}) == [(2, "c3")]


def test_tasks_sharing_a_definition_are_one_chain_and_the_rest_are_none():
    assert sandbox.task_chain_list([["a::f"], ["b::g"], ["a::f", "c::h"], ["c::h"], []]) \
        == [1, 0, 1, 1, 0]
    assert sandbox.task_chain_list([["a::f"], ["b::g"], ["b::g"], ["a::f"]]) == [1, 2, 2, 1]
    assert sandbox.task_chain_list([[], []]) == [0, 0]


@pytest.mark.parametrize("split_obj", [
    {"tasks": 5}, {"tasks": "add g"}, {"tasks": {"1": "add g"}}, {"tasks": None},
    {"questions": 1.5}, {"questions": "why?"}, {"unclear": True}, {"unclear": "what?"}])
def test_a_split_whose_entries_are_no_list_is_no_split(tmp_path, split_obj):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(split_obj), encoding="utf-8")
    with pytest.raises(RuntimeError, match="neither any question nor any task"):
        sandbox._read_split(str(path), str(path.parent))


def test_a_split_entry_with_nothing_written_is_not_read(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": ["...", "add g"], "defs": [["x"], ["a.py::g"]]}),
                    encoding="utf-8")
    split_dict = sandbox._read_split(str(path), str(path.parent))
    assert split_dict["tasks"] == ["add g"] and split_dict["defs"] == [["a.py::g"]]
    path.write_text(json.dumps({"tasks": ["..."]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="neither any question nor any task"):
        sandbox._read_split(str(path), str(path.parent))


def test_a_split_key_that_is_no_list_leaves_the_other_keys_read(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": "add g", "questions": 5, "unclear": ["what?"]}),
                    encoding="utf-8")
    assert sandbox._read_split(str(path), str(path.parent)) == {"unclear": ["what?"]}


@pytest.mark.parametrize("key", ["tasks", "questions"])
def test_a_text_handed_back_holds_nothing_a_command_line_cannot(tmp_path, key):
    """Half of a surrogate pair and NUL are replaced where the split is read, so the
    container of that task or question can be started."""
    path = tmp_path / "tasks.json"
    sandbox.write_across(str(path), {key: ["add \ud83d a\0 function \udc80", "テストを足す"]},
                         "split")
    entry_list = sandbox._read_split(str(path), str(path.parent))[key]
    assert entry_list == ["add \ufffd a\ufffd function \ufffd", "テストを足す"]
    for argv in (sandbox.edit_argv("repo", entry_list[0]),
                 sandbox.answer_argv("repo", entry_list[0], "/out/a.json")):
        assert subprocess.run(["true", *argv], check=False).returncode == 0


def test_an_answer_handed_back_holds_nothing_a_command_line_cannot(tmp_path):
    path = tmp_path / "answer.json"
    sandbox.write_across(str(path), {"answer": "it \udfff is\0 so"}, "answer")
    assert sandbox._read_answer(str(path), str(path.parent)) == "it \ufffd is\ufffd so"


def test_a_split_without_definitions_reads_as_tasks_of_no_definition(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": ["one", "", "two"]}), encoding="utf-8")
    assert sandbox._read_split(str(path), str(path.parent)) == {"tasks": ["one", "two"], "defs": [[], []]}
    path.write_text(json.dumps({"tasks": ["one", "", "two"],
                                "defs": [["a::f"], ["b::g"], ["c::h", 3]]}),
                    encoding="utf-8")
    assert sandbox._read_split(str(path), str(path.parent)) == {"tasks": ["one", "two"],
                                              "defs": [["a::f"], ["c::h"]]}
    path.write_text(json.dumps({"tasks": ["one"], "defs": [["a::f"], ["b::g"]]}),
                    encoding="utf-8")
    assert sandbox._read_split(str(path), str(path.parent)) == {"tasks": ["one"], "defs": [[]]}


@pytest.mark.parametrize("key", ["tasks", "questions"])
def test_the_split_is_held_to_its_limits_here_too(tmp_path, monkeypatch, caplog, key):
    """The limits are kept inside the container, where the model's code runs."""
    monkeypatch.setattr(config, "DISPATCH_MAX_TASK", 2)
    monkeypatch.setattr(config, "DISPATCH_TASK_MAX_CHAR", 10)
    caplog.set_level(logging.WARNING, logger="codetwine_loom.sandbox")
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({key: ["x" * 30, sandbox.cut("y" * 30, 10), "z"]}),
                    encoding="utf-8")
    entry_list = sandbox._read_split(str(path), str(path.parent))[key]
    assert entry_list == [sandbox.cut("x" * 30, 10), sandbox.cut("y" * 30, 10)]
    assert "holds 3" in caplog.text and "entry 1" in caplog.text
    assert "entry 2" not in caplog.text


def test_the_answer_is_held_to_its_limit_here_too(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 10)
    path = tmp_path / "answer.json"
    path.write_text(json.dumps({"answer": "a" * 30}), encoding="utf-8")
    assert sandbox._read_answer(str(path), str(path.parent)) == sandbox.cut("a" * 30, 10)
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 0)
    assert sandbox._read_answer(str(path), str(path.parent)) == "a" * 30


def _starting_where_told(repo, exit_status_dict=None):
    """Return a sandbox.run() stand-in that makes the branch it was given at the commit
    it was told to start at (or at HEAD), lands one empty commit there, notes the
    start and how many were going at once, and ends the way exit_status_dict says for
    that branch (0 when it says nothing)."""
    run_list = []
    going = {"count": 0, "most": 0}
    lock = threading.Lock()

    def run(project, output_dir, request, **kw):
        with lock:
            going["count"] += 1
            going["most"] = max(going["most"], going["count"])
        time.sleep(0.05)
        start = kw.get("start") or _git(repo, "rev-parse", "HEAD").strip()
        _git(repo, "branch", kw["branch"], start)
        _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit-tree",
             f"{start}^{{tree}}", "-p", start, "-m", kw["branch"])
        commit = _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit-tree",
                      f"{start}^{{tree}}", "-p", start, "-m", kw["branch"]).strip()
        _git(repo, "branch", "-f", kw["branch"], commit)
        status = (exit_status_dict or {}).get(kw["branch"], 0)
        run_list.append({"branch": kw["branch"], "start": kw.get("start", ""),
                         "commit": commit})
        with lock:
            going["count"] -= 1
        return {"commit": commit, "exit_status": status,
                "verdict": "failed" if status else "unverified"}

    return run, run_list, going


def _combining_stand_in(called: list):
    """Return a sandbox.integrate() stand-in that notes the branches it was given."""
    def integrate(project, output_dir, branch_list, **kw):
        called.append(list(branch_list))
        return {"exit_status": 0, "commit": "u" * 40, "branch": "codetwine-loom/combined",
                "verdict": "settled"}

    return integrate


def test_tasks_that_change_the_same_definition_run_one_after_the_other(
        _dispatch_repo, monkeypatch, caplog):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container", _split_written(
        _tasks_file(out), {"tasks": ["one", "two", "three"],
                           "defs": [["s::r"], ["s::r"], ["o::q"]]}))
    run, run_list, going = _starting_where_told(repo)
    monkeypatch.setattr(sandbox, "run", run)
    called = []
    monkeypatch.setattr(sandbox, "integrate", _combining_stand_in(called))
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    with caplog.at_level(logging.INFO):
        result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["exit_status"] == 0
    assert "tasks 1, 2 change the same definition, so they run one after another" \
        in caplog.text
    assert "task 2 waits for the tasks before it that change the same definition" in caplog.text
    by_branch = {run["branch"][-1]: run for run in run_list}
    base = _git(repo, "rev-parse", "main").strip()
    assert by_branch["1"]["start"] == base and by_branch["3"]["start"] == base
    assert by_branch["2"]["start"] == by_branch["1"]["commit"]
    assert _git(repo, "rev-parse", f"{by_branch['2']['commit']}^").strip() \
        == by_branch["1"]["commit"]
    # Task 3 ran beside the chain, and the chain's last branch alone stands for it
    assert going["most"] == 2
    assert called == [[by_branch["2"]["branch"], by_branch["3"]["branch"]]]


def test_the_next_of_a_chain_starts_where_the_last_that_finished_left(
        _dispatch_repo, monkeypatch):
    """Task 1 does not get through on its first try, so task 2 starts at the base and
    the try that gets task 1 through starts at what task 2 left."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container", _split_written(
        _tasks_file(out), {"tasks": ["one", "two"], "defs": [["s::r"], ["s::r"]]}))
    run, run_list, _ = _starting_where_told(
        repo, {"codetwine-loom/run/20260101-000000-1-42/1": 1})
    monkeypatch.setattr(sandbox, "run", run)
    called = []
    monkeypatch.setattr(sandbox, "integrate", _combining_stand_in(called))
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["exit_status"] == 0
    assert [run["branch"] for run in run_list] == [
        "codetwine-loom/run/20260101-000000-1-42/1", "codetwine-loom/run/20260101-000000-1-42/2",
        "codetwine-loom/run/20260101-000000-1-42/1-retry1"]
    base = _git(repo, "rev-parse", "main").strip()
    assert run_list[1]["start"] == base
    assert run_list[2]["start"] == run_list[1]["commit"]
    assert called == [["codetwine-loom/run/20260101-000000-1-42/1-retry1"]]


def test_a_split_without_definitions_runs_every_task_at_once_as_before(
        _dispatch_repo, monkeypatch, caplog):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    run, run_list, going = _starting_where_told(repo)
    monkeypatch.setattr(sandbox, "run", run)
    called = []
    monkeypatch.setattr(sandbox, "integrate", _combining_stand_in(called))
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    with caplog.at_level(logging.INFO):
        sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert "change the same definition" not in caplog.text
    base = _git(repo, "rev-parse", "main").strip()
    assert going["most"] == 2 and all(run["start"] == base for run in run_list)
    assert called == [[
        "codetwine-loom/run/20260101-000000-1-42/1", "codetwine-loom/run/20260101-000000-1-42/2"]]


def test_a_union_that_did_not_reach_the_tree_is_not_run_again(_dispatch_repo,
                                                             monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    monkeypatch.setattr(sandbox, "_start_container", _split_written(
        _tasks_file(out), {"tasks": ["one"]}))
    run, _, _ = _starting_where_told(repo)
    monkeypatch.setattr(sandbox, "run", run)
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        "exit_status": 0, "commit": "u" * 40, "branch": "codetwine-loom/combined",
        "verdict": "settled", "land": None, "land_why": "it does not apply"})
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["runs"] == 1 and result["exit_status"] == 0
    assert sandbox.land_status(result["integration"]) == 1


# == running the merge again from the union it left (INTEGRATE_MAX_RERUN) ========

def _merge_stand_in(repo, verdict_list: list[str], called: list):
    """Return a sandbox.integrate() stand-in that lands a union on a branch of its own
    per call, with the verdict verdict_list gives for that call, and notes what it was
    called with."""
    def integrate(project, output_dir, branch_list, **kw):
        n = len(called) + 1
        called.append({"branches": list(branch_list), "resume": kw.get("resume", "")})
        branch = f"codetwine-loom/combined-{n}"
        _git(repo, "branch", branch)
        verdict = verdict_list[n - 1]
        return {"exit_status": 0 if verdict == "settled" else 1, "commit": "u" * 40,
                "branch": branch, "verdict": verdict,
                "branches": [kw["resume"]] if kw.get("resume") else list(branch_list)}

    return integrate


@pytest.fixture
def _two_tasks_land(_dispatch_repo, monkeypatch):
    """A dispatching run whose two tasks both finish, so what follows is the merge."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one", "two"]}, []))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 0))
    return repo, out


def test_a_merge_that_was_not_settled_is_run_again_from_the_union_it_left(
        _two_tasks_land, monkeypatch):
    """Every task got through, so what is run again is the merge alone, carrying on
    from the union the merge before it left; it stops as soon as one is settled."""
    repo, out = _two_tasks_land
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 3)
    called = []
    monkeypatch.setattr(sandbox, "integrate",
                        _merge_stand_in(repo, ["unsettled", "unsettled", "settled"], called))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert [c["resume"] for c in called] == [
        "", "codetwine-loom/combined-1", "codetwine-loom/combined-2"]
    assert called[1]["branches"] == [] and called[2]["branches"] == []
    assert result["merge_runs"] == 3 and result["runs"] == 1
    assert result["exit_status"] == 0 and result["resume"] == []


def test_the_unions_the_merges_before_the_last_left_are_taken_away(_two_tasks_land,
                                                                   monkeypatch):
    """Each is a parent of the union after it, as the task branches are."""
    repo, out = _two_tasks_land
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 1)
    called, dropped = [], []
    monkeypatch.setattr(sandbox, "integrate",
                        _merge_stand_in(repo, ["unsettled", "settled"], called))
    monkeypatch.setattr(sandbox, "drop_branch_under",
                        lambda project_dir, branch_list, commit: dropped.append(list(branch_list)) or [])
    sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert ["codetwine-loom/combined-1"] in dropped


def test_the_merge_is_run_again_no_more_than_the_setting_says(_two_tasks_land, monkeypatch):
    """What is left after the last one is the union to carry on from, which is what a
    whole run again (DISPATCH_MAX_RERUN) is told."""
    repo, out = _two_tasks_land
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 1)
    called = []
    monkeypatch.setattr(sandbox, "integrate",
                        _merge_stand_in(repo, ["unsettled", "unsettled", "unsettled"], called))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert len(called) == 2 and result["merge_runs"] == 2
    assert result["exit_status"] == 1 and result["resume"] == ["codetwine-loom/combined-2"]


def test_a_merge_that_head_s_run_of_the_tests_keeps_unsettled_is_not_made_again(
        _two_tasks_land, monkeypatch, caplog):
    """No merge over the same HEAD is settled, nor is the request run again over it;
    the union is still the one to carry on from once HEAD's run is changed."""
    repo, out = _two_tasks_land
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 3)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    called = []
    merge = _merge_stand_in(repo, ["unsettled"] * 4, called)
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        **merge(*a, **k), "head_block": ["stop"]})
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert len(called) == 1 and result["merge_runs"] == 1 and result["runs"] == 1
    assert result["exit_status"] == 1 and result["resume"] == ["codetwine-loom/combined-1"]
    assert "so the request is not run again" in caplog.text
    assert "is run again from" not in caplog.text


def test_a_merge_killed_at_the_memory_limit_is_not_made_again(_two_tasks_land, monkeypatch,
                                                               caplog):
    """Neither the merge nor the request is run again."""
    repo, out = _two_tasks_land
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 3)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    call_list = []
    merge = _merge_stand_in(repo, ["unsettled"] * 4, call_list)
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        **merge(*a, **k), "exit_status": 137})
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert len(call_list) == 1 and result["runs"] == 1
    assert "raise SANDBOX_MEMORY to run it" in caplog.text
    assert "is run again" not in caplog.text


def test_the_merge_is_run_once_when_the_setting_is_zero(_two_tasks_land, monkeypatch):
    repo, out = _two_tasks_land
    called = []
    monkeypatch.setattr(sandbox, "integrate", _merge_stand_in(repo, ["unsettled"], called))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert len(called) == 1 and result["merge_runs"] == 1 and result["exit_status"] == 1


def test_a_merge_made_again_that_landed_nothing_leaves_the_union_before_it_to_carry_on_from(
        _two_tasks_land, monkeypatch):
    """The container of the merge made again ended before it wrote a patch: the
    merges end there, and the run names the union the first merge left."""
    repo, out = _two_tasks_land
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 2)
    called = []
    first = _merge_stand_in(repo, ["unsettled"], called)

    def integrate(project, output_dir, branch_list, **kw):
        if not kw.get("resume"):
            return first(project, output_dir, branch_list, **kw)
        called.append({"branches": list(branch_list), "resume": kw["resume"]})
        return {"exit_status": 137, "commit": None, "branch": None, "no_patch": True,
                "verdict": "unsettled", "branches": [kw["resume"]], "land": None,
                "land_why": "the merging run ended (exit status 137) and left no patch"}

    monkeypatch.setattr(sandbox, "integrate", integrate)
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert [c["resume"] for c in called] == ["", "codetwine-loom/combined-1"]
    assert result["merge_runs"] == 2 and result["exit_status"] == 137
    assert result["resume"] == ["codetwine-loom/combined-1"]
    union_dict = result["integration"]
    assert (union_dict["branch"], union_dict["commit"]) == ("codetwine-loom/combined-1", "u" * 40)
    assert union_dict["land_why"] == (
        "the merging run ended (exit status 137) and left no patch; the union the merge "
        "before it left stays on codetwine-loom/combined-1")
    assert sandbox.resume_branch_list([], union_dict, 137) == ["codetwine-loom/combined-1"]


def test_a_merge_made_again_that_changed_nothing_leaves_the_union_before_it_unsettled(
        _two_tasks_land, monkeypatch):
    """The merge made again ended with exit status 0 and a patch that holds nothing:
    the run is still one that did not get through, and names the union the first
    merge left."""
    repo, out = _two_tasks_land
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 2)
    called = []
    first = _merge_stand_in(repo, ["unsettled"], called)

    def integrate(project, output_dir, branch_list, **kw):
        if not kw.get("resume"):
            return first(project, output_dir, branch_list, **kw)
        called.append({"branches": list(branch_list), "resume": kw["resume"]})
        return {"exit_status": 0, "commit": None, "branch": None, "no_patch": False,
                "patch": None, "files": 0, "verdict": "settled",
                "branches": [kw["resume"]], "land": None, "land_why": "",
                "land_command": "", "land_refuse": False}

    monkeypatch.setattr(sandbox, "integrate", integrate)
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert [c["resume"] for c in called] == ["", "codetwine-loom/combined-1"]
    assert result["merge_runs"] == 2 and result["exit_status"] == 1
    assert result["resume"] == ["codetwine-loom/combined-1"]
    union_dict = result["integration"]
    assert (union_dict["branch"], union_dict["commit"], union_dict["verdict"],
            union_dict["exit_status"]) == ("codetwine-loom/combined-1", "u" * 40, "unsettled", 1)
    assert union_dict["land_why"] == (
        "the merge made again ended with exit status 0 and changed nothing, so there is "
        "nothing of it to put into the working tree; the union the merge before it left "
        "stays on codetwine-loom/combined-1")
    assert sandbox.land_status(union_dict) == 1


def test_a_first_merge_that_changed_nothing_is_a_run_that_did_not_get_through(
        _two_tasks_land, monkeypatch):
    """The first merge ended with exit status 0 and a patch that holds nothing: no
    union was made, the task branches stay, and the run names them to carry on from."""
    repo, out = _two_tasks_land
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 2)
    called = []

    def integrate(project, output_dir, branch_list, **kw):
        called.append(list(branch_list))
        return {"exit_status": 0, "commit": None, "branch": None, "no_patch": False,
                "patch": None, "files": 0, "verdict": "settled",
                "branches": list(branch_list), "land": None, "land_why": "",
                "land_command": "", "land_refuse": False}

    monkeypatch.setattr(sandbox, "integrate", integrate)
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert len(called) == 1 and result["merge_runs"] == 1
    task_list = [entry_dict["branch"] for entry_dict in result["branches"]]
    assert result["exit_status"] == 1 and result["resume"] == task_list
    assert result["dropped"] == [] and set(task_list) <= set(_branches(repo))
    union_dict = result["integration"]
    assert (union_dict["verdict"], union_dict["exit_status"], union_dict["commit"]) == (
        "unsettled", 1, None)
    assert union_dict["land_why"] == (
        "the merge ended with exit status 0 and changed nothing, so there is no union "
        "and nothing was put into the working tree. The branches it was to combine are "
        "as they were")
    assert sandbox.land_status(union_dict) == 1


def test_a_run_whose_union_was_not_put_in_names_the_union_to_carry_on_from(
        _two_tasks_land, monkeypatch):
    """The merge settled and its union did not go into the working tree: the run is
    not made again, and what it names to carry on from is the union."""
    repo, out = _two_tasks_land
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    called = []
    first = _merge_stand_in(repo, ["settled"], called)

    def integrate(project, output_dir, branch_list, **kw):
        return {**first(project, output_dir, branch_list, **kw), "land": None,
                "land_why": "the change does not apply to the working tree",
                "land_command": "git apply x", "land_refuse": False}

    monkeypatch.setattr(sandbox, "integrate", integrate)
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert len(called) == 1 and result["runs"] == 1 and result["exit_status"] == 0
    assert result["resume"] == ["codetwine-loom/combined-1"]


@pytest.mark.parametrize("union_dict, exit_status, expect_list", [
    ({"branch": "u", "land": None, "land_why": "it does not apply"}, 0, ["u"]),
    ({"branch": "u", "land": None, "land_why": "kept out", "land_refuse": True}, 1, []),
    ({"branch": "u", "land": {"files": 1}, "land_why": ""}, 0, []),
    ({"branch": None, "land": None, "land_why": "no union"}, 0, []),
    ({"branch": None, "land": None, "land_why": "no union"}, 1, ["t"])])
def test_a_union_that_was_not_put_into_the_tree_is_what_a_run_is_carried_on_from(
        union_dict, exit_status, expect_list):
    entry_list = [{"branch": "t", "commit": "c" * 40}]
    assert sandbox.resume_branch_list(entry_list, union_dict, exit_status) == expect_list


def _change_on_new_branch(repo, branch, start, name, message):
    """Make a branch at start and commit one new file on it, leaving the working tree
    as it was."""
    _git(repo, "checkout", "--quiet", "-b", branch, start)
    (repo / name).write_text("the change\n", encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m",
         message)
    _git(repo, "checkout", "--quiet", "main")


@pytest.mark.parametrize("root", ["codetwine-loom/sandbox", "codetwine-loom/run"])
def test_a_stopped_command_names_a_settled_branch_the_working_tree_does_not_hold(
        _dispatch_repo, monkeypatch, root):
    """A command stopped before or while its change was put into the working tree
    left the whole change on a branch: the branch is named until the working tree
    holds what it made, and the task branches it reaches are named in neither case."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    own = f"20260102-000000-2-{os.getpid()}"
    union = f"codetwine-loom/sandbox/20260102-000001-5-{os.getpid()}"
    task = f"codetwine-loom/run/{own}/1"
    _change_on_new_branch(repo, task, "main", "task.txt", "sandbox: one task")
    _change_on_new_branch(repo, union, task, "union.txt",
                          "integrate: x\n\ncombine a\n\nCodetwine-Loom-Verdict: settled\n")
    assert sandbox._settle_branch_list(str(repo), [union]) == [union]
    assert not sandbox._is_in_tree(str(repo), union)
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [union]
    # A task branch no union reaches is named beside it
    other = f"{root}/{own}{'' if root == 'codetwine-loom/sandbox' else '/2'}"
    _change_on_new_branch(repo, other, "main", "other.txt", "sandbox: another")
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [other, union]
    # With the change of the union in the working tree, neither it nor the task
    # branch it reaches is named
    (repo / "union.txt").write_text("the change\n", encoding="utf-8")
    assert sandbox._is_in_tree(str(repo), union)
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [other]
    assert not sandbox._is_in_tree(str(repo), "codetwine-loom/sandbox/gone")


def test_a_merge_that_left_no_verdict_is_not_run_again(_two_tasks_land, monkeypatch):
    """No verdict is a run that did not say how it ended, not one the check found
    unsettled, so there is nothing a merge again would carry on from."""
    repo, out = _two_tasks_land
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 2)
    called = []
    monkeypatch.setattr(sandbox, "integrate", _merge_stand_in(repo, [""], called))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert len(called) == 1 and result["exit_status"] == 1


def test_the_merge_again_comes_before_the_whole_run_again(_two_tasks_land, monkeypatch):
    """The merges again use up their count first; only then is the request run again
    as a whole, told the last union."""
    repo, out = _two_tasks_land
    _run_ids(monkeypatch, 2)
    told = []
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one", "two"]}, told))
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 1)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    called = []
    monkeypatch.setattr(sandbox, "integrate", _merge_stand_in(
        repo, ["unsettled", "unsettled", "unsettled", "settled"], called))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert [c["resume"] for c in called] == [
        "", "codetwine-loom/combined-1", "", "codetwine-loom/combined-3"]
    assert told[1][-2:] == ["--resume", "codetwine-loom/combined-2"]
    assert result["runs"] == 2 and result["merge_runs"] == 2 and result["exit_status"] == 0


# == running the request again (DISPATCH_MAX_RERUN) ==============================

def _run_ids(monkeypatch, count: int):
    """Pin new_run_id to count names in order, differing in the serial number alone."""
    ids = iter(f"20260101-000000-{i}-42" for i in range(1, count + 1))
    monkeypatch.setattr(sandbox, "new_run_id", ids.__next__)


def _split_written_where_asked(split: dict, told: list):
    """Return a _start_the_container stand-in that writes the split where the run it
    was started for reads it, and notes what each reading run was told."""
    def start(*a, **k):
        told.append(a[10])
        return _split_written(a[8].task, split)(*a, **k)

    return start


def _landing_on_its_branch(repo, exit_status):
    """Return a sandbox.run() stand-in that makes the branch it was given, so a later
    run can be told to read it, and ends the way exit_status says. A callable
    exit_status is asked per branch."""
    def run(project, output_dir, request, **kw):
        _git(repo, "branch", kw["branch"])
        status = exit_status(kw["branch"]) if callable(exit_status) else exit_status
        return {"commit": "c" * 40, "exit_status": status,
                "verdict": "failed" if status else "unverified"}

    return run


def test_a_run_that_did_not_get_through_is_run_once_unless_the_setting_says(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    told = []
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one", "two"]}, told))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 1))
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 0)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert len(told) == 1
    assert result["runs"] == 1 and result["exit_status"] == 1


def test_a_run_that_did_not_get_through_is_run_again_from_what_it_left(
        _dispatch_repo, monkeypatch):
    """Every run after the first is told the branches the run before it left: the
    same names the run prints for --resume. The count is the setting's."""
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 3)
    told = []
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one", "two"]}, told))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 1))
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["runs"] == 3 and result["exit_status"] == 1
    assert "--resume" not in told[0]
    assert told[1][-4:] == ["--resume", "codetwine-loom/run/20260101-000000-1-42/1",
                            "--resume", "codetwine-loom/run/20260101-000000-1-42/2"]
    assert told[2][-4:] == ["--resume", "codetwine-loom/run/20260101-000000-2-42/1",
                            "--resume", "codetwine-loom/run/20260101-000000-2-42/2"]
    assert result["resume"] == ["codetwine-loom/run/20260101-000000-3-42/1",
                                "codetwine-loom/run/20260101-000000-3-42/2"]


def test_only_the_first_run_makes_the_documents_again(_dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 2)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one"]}, []))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 1))
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    seen = _update_doc_seen(monkeypatch)
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して", build_doc=True)
    assert result["runs"] == 2
    assert len(seen) == 1


def test_a_sandbox_run_makes_the_documents_again_at_its_start(_dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    seen = []

    def note(*a, **k):
        seen.append((a[4], k))
        raise RuntimeError("stop here")

    monkeypatch.setattr(sandbox, "exclude_list_of_run", note)
    with pytest.raises(RuntimeError, match="stop here"):
        sandbox.run(str(repo), str(out), "設定を整理して", build_doc=True)
    assert seen == [(True, {"doc_update": True, "engine": ""})]


def test_a_run_carrying_on_from_a_settled_merge_asks_about_the_documents(_dispatch_repo,
                                                                         monkeypatch):
    repo, out = _dispatch_repo
    _git(repo, "branch", "codetwine-loom/sandbox/old")
    monkeypatch.setattr(sandbox, "stage_resume_dict", lambda *a: {"stage": "refactor"})
    monkeypatch.setattr(sandbox.integrate_mod, "request_in_branch_list",
                        lambda *a: "add a")
    seen = []

    def integrate(*a, **k):
        seen.append(k)
        raise RuntimeError("stop")

    monkeypatch.setattr(sandbox, "integrate", integrate)
    with pytest.raises(RuntimeError, match="stop"):
        sandbox.dispatch(str(repo), str(out), "add a", resume_list=["codetwine-loom/sandbox/old"])
    assert seen[0]["ask_doc"] is True and seen[0]["resume"] == "codetwine-loom/sandbox/old"


def test_a_stage_carried_on_from_hands_the_question_to_the_first_analysis(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    seen = []

    def exclude(*a, **k):
        seen.append(a[4:])
        raise RuntimeError("stop")

    monkeypatch.setattr(sandbox, "exclude_list_of_run", exclude)
    resume_dict = {"union": "f" * 40, "stage": integrate_mod.REFACTOR_STAGE,
                   "base": "", "paths": [], "merge": "", "settled": ""}
    with pytest.raises(RuntimeError, match="stop"):
        sandbox._resume_stage(str(repo), str(out), "repo", "main", resume_dict, "", "",
                              False, "", None, False, True)
    assert seen == [(False, True)]


def test_every_run_again_leaves_the_branches_the_first_was_told_about(_dispatch_repo,
                                                                     monkeypatch):
    """The sweep of a run made again would otherwise count them as an old run's."""
    repo, out = _dispatch_repo
    _git(repo, "branch", "codetwine-loom/run/20250101-000000-1-1/1")
    _run_ids(monkeypatch, 3)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one"]}, []))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 1))
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    protect = []
    monkeypatch.setattr(sandbox, "remove_old_run",
                        lambda *a, protect_list=(), **k: protect.append(protect_list) or [])
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して",
                              resume_list=["codetwine-loom/run/20250101-000000-1-1/1"])
    assert result["runs"] == 3 and len(protect) == 3
    assert all("codetwine-loom/run/20250101-000000-1-1/1" in each for each in protect)


def test_a_run_that_left_no_branch_hands_on_what_it_was_told(_dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    _git(repo, "branch", "codetwine-loom/run/20250101-000000-1-1/1")
    _run_ids(monkeypatch, 2)
    told = []
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one"]}, told))
    monkeypatch.setattr(sandbox, "run", lambda project, output_dir, request, **kw: {
        "commit": "", "exit_status": 1, "verdict": "failed"})
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して",
                              resume_list=["codetwine-loom/run/20250101-000000-1-1/1"])
    assert result["runs"] == 2
    assert [argv[-2:] for argv in told] == \
        [["--resume", "codetwine-loom/run/20250101-000000-1-1/1"]] * 2


def test_a_merge_that_was_not_settled_is_what_the_run_again_is_told(_dispatch_repo,
                                                                    monkeypatch):
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 2)
    told = []
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one", "two"]}, told))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 0))
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    unions = []

    def unsettled(*a, **k):
        unions.append(f"codetwine-loom/combined-{len(unions) + 1}")
        _git(repo, "branch", unions[-1])
        return {"exit_status": 1, "commit": "u" * 40, "branch": unions[-1]}

    monkeypatch.setattr(sandbox, "integrate", unsettled)
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert told[1][-2:] == ["--resume", "codetwine-loom/combined-1"]
    assert result["runs"] == 2 and result["resume"] == ["codetwine-loom/combined-2"]
    # What a command stopped while the run made again goes on reads
    assert [entry[1:] for entry in sandbox._RERUN_LIST] == [(["codetwine-loom/combined-1"], [], 0)]


def test_a_gap_the_tests_found_runs_the_request_again_with_a_new_split(_dispatch_repo,
                                                                        monkeypatch):
    """The union settled, so the run carrying on from it is split again rather than
    sent to a stage."""
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 2)
    told = []
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one"]}, told))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 0))
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    unions = []

    def gapped(*a, **k):
        unions.append(f"codetwine-loom/combined-{len(unions) + 1}")
        _git(repo, "branch", unions[-1])
        return {"exit_status": 0, "commit": "u" * 40, "branch": unions[-1],
                "gap": [{"what": "case_1", "why": "a.py does not do it"}]}

    monkeypatch.setattr(sandbox, "integrate", gapped)
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert len(told) == 2 and told[1][-2:] == ["--resume", "codetwine-loom/combined-1"]
    assert result["runs"] == 2 and result["exit_status"] == 1
    assert result["resume"] == ["codetwine-loom/combined-2"]


def test_what_a_run_put_into_the_tree_is_handed_back_with_the_run_made_again(monkeypatch):
    land = {"files": 1, "paths": ["a.py"], "patch": "p", "last": "/out/land-last.diff"}
    run_list = [
        {"tasks": ["t"], "integration": {"land": land}, "resume": ["codetwine-loom/u"],
         "exit_status": 1},
        {"tasks": ["t"], "integration": {"land": {"files": 0, "paths": []}},
         "resume": ["codetwine-loom/u"], "exit_status": 1},
        {"unclear": ["which?"], "resume": ["codetwine-loom/u"], "exit_status": 1}]
    monkeypatch.setattr(sandbox, "_dispatch_once", lambda *a, **k: run_list.pop(0))
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    result = sandbox.dispatch("repo", "out", "add a")
    assert result["runs"] == 3 and "unclear" in result
    assert result["earlier_land"] == [land]


def test_a_run_made_again_that_stops_stops_the_command_as_the_first_run_does(monkeypatch):
    """What the run before it put in is what the command's stop record reads
    (land_list), and the branches both runs left are what it names to carry on from."""
    land = {"files": 1, "paths": ["a.py"], "patch": "p", "last": "/out/land-last.diff"}
    first = {"tasks": ["t"], "integration": {"land": land}, "resume": ["codetwine-loom/u"],
             "exit_status": 1}

    def once(*a, **k):
        if "build_doc" in k:
            return first
        raise RuntimeError("no engine")

    monkeypatch.setattr(sandbox, "_dispatch_once", once)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    with pytest.raises(RuntimeError, match="no engine"):
        sandbox.dispatch("repo", "out", "add a")


def test_the_runs_stop_as_soon_as_one_gets_through(_dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 2)
    told = []
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one"]}, told))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(
        repo, lambda branch: 1 if "-1-42/" in branch else 0))
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 3)
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
                            "exit_status": 0, "commit": "u" * 40,
                            "branch": "codetwine-loom/sandbox/u"})
    result = sandbox.dispatch(str(repo), str(out), "名前を変えて")
    assert len(told) == 2
    assert result["runs"] == 2 and result["exit_status"] == 0
    assert result["resume"] == []


def test_a_run_again_that_cannot_start_stops_the_command_as_a_first_run_does(
        _dispatch_repo, monkeypatch, caplog):
    """The run again stops the way a run somebody started by hand does."""
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 2)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one", "two"]}, []))
    # A commit is reported, but no branch is made, so the next run is told a name
    # that is not there
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 1, "verdict": "failed"})
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    with pytest.raises(RuntimeError, match="no branch named"):
        sandbox.dispatch(str(repo), str(out), "設定を整理して")


def test_a_run_again_that_raises_stops_the_command_with_the_branches_still_there(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 2)
    told = []
    split_start = _split_written_where_asked({"tasks": ["one", "two"]}, told)

    def start(*a, **k):
        if told:
            raise RuntimeError("the reading run handed back no split")
        return split_start(*a, **k)

    monkeypatch.setattr(sandbox, "_start_container", start)
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 1))
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    with pytest.raises(RuntimeError, match="handed back no split"):
        sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert _git(repo, "branch", "--list", "codetwine-loom/run/20260101-000000-1-42/*").split() == \
        ["codetwine-loom/run/20260101-000000-1-42/1", "codetwine-loom/run/20260101-000000-1-42/2"]


def test_a_first_run_that_raises_still_stops_the_whole_thing(_dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo

    def start(*a, **k):
        raise RuntimeError("the reading run handed back no split")

    monkeypatch.setattr(sandbox, "_start_container", start)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    with pytest.raises(RuntimeError, match="no split"):
        sandbox.dispatch(str(repo), str(out), "設定を整理して")


def test_a_question_that_was_not_answered_is_asked_again_as_a_whole(_dispatch_repo,
                                                                    monkeypatch):
    """A question leaves no branch, so the run again is told nothing and asks again."""
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 2)
    told = []
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"questions": ["where?"]}, told))
    monkeypatch.setattr(sandbox, "answer", lambda project, output_dir, question, **kw: {
        "answer": "", "exit_status": 1})
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    result = sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert len(told) == 2 and "--resume" not in told[1]
    assert result["runs"] == 2 and result["exit_status"] == 1


def test_a_question_the_agents_answered_is_not_asked_again_as_a_whole(_dispatch_repo,
                                                                      monkeypatch):
    """The run made again starts without the answers this one holds."""
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 2)
    told = []
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"questions": ["where?"]}, told))

    def answer(project, output_dir, question, answer_list=None, **kw):
        if answer_list is not None:
            raise RuntimeError("the combining run handed no answer back")
        return {"answer": "in config.py", "exit_status": 0}

    monkeypatch.setattr(sandbox, "answer", answer)
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 1)
    result = sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert len(told) == 1
    assert result["runs"] == 1 and result["exit_status"] == 1
    assert result["combine_runs"] == 2
    assert "in config.py" in result["answer"]


def test_a_request_handed_back_starts_no_agent_and_is_not_run_again(_dispatch_repo,
                                                                   monkeypatch):
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 2)
    told = []
    monkeypatch.setattr(sandbox, "_start_container", _split_written_where_asked(
        {"unclear": ["JSON or YAML?"]}, told))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: pytest.fail("an agent started"))
    monkeypatch.setattr(sandbox, "answer", lambda *a, **k: pytest.fail("an agent started"))
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    result = sandbox.dispatch(str(repo), str(out), "設定を書き出せるようにして")
    assert len(told) == 1
    assert result["unclear"] == ["JSON or YAML?"] and result["runs"] == 1
    assert result["exit_status"] == 1 and result["resume"] == []
    assert _git(repo, "status", "--porcelain") == ""
    assert _git(repo, "for-each-ref", "--format=%(refname:short)",
                "refs/heads").split() == ["main"]


def test_the_open_points_read_back_are_held_to_the_setting(tmp_path, monkeypatch):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"unclear": ["a?", "b?", "c?", "d?"]}), encoding="utf-8")
    monkeypatch.setattr(config, "DISPATCH_MAX_UNCLEAR", 3)
    assert sandbox._read_split(str(path), str(path.parent)) == {"unclear": ["a?", "b?", "c?"]}


def test_open_points_are_not_read_back_when_they_are_switched_off(tmp_path, monkeypatch):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"unclear": ["a?"]}), encoding="utf-8")
    monkeypatch.setattr(config, "DISPATCH_MAX_UNCLEAR", 0)
    with pytest.raises(RuntimeError, match="Nothing was started"):
        sandbox._read_split(str(path), str(path.parent))


# == taking away what older runs left ====================


@pytest.fixture
def _runs_left_behind(_dispatch_repo, monkeypatch):
    """Three requests nobody finished, oldest first, with every run counted as ended.

    That is what a run leaves when it did not reach a union: a branch per task and an
    output directory per agent, and no union branch at all.
    """
    repo, out = _dispatch_repo
    runs = ["20260101-000000-1-11", "20260102-000000-1-22", "20260103-000000-1-33"]
    for run_id in runs:
        for n in (1, 2):
            _git(repo, "branch", f"codetwine-loom/run/{run_id}/{n}")
            os.makedirs(sandbox.agent_dir(str(out), f"run-{run_id}-{n}"))
        os.makedirs(sandbox.agent_dir(str(out), run_id))
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    return repo, out, runs


def _branches(repo) -> list[str]:
    """Return the repository's branch names."""
    return _git(repo, "for-each-ref", "--format=%(refname:short)",
                "refs/heads").split()


def test_what_all_but_the_newest_runs_left_is_taken_away(_runs_left_behind):
    """A run that did not reach a union leaves its branches and its output where they
    are, and nothing else ever takes them away."""
    repo, out, runs = _runs_left_behind
    assert sandbox.remove_old_run(str(repo), str(out), 2) == [runs[0]]
    assert f"codetwine-loom/run/{runs[0]}/1" not in _branches(repo)
    assert f"codetwine-loom/run/{runs[1]}/1" in _branches(repo)
    assert not os.path.exists(sandbox.agent_dir(str(out), f"run-{runs[0]}-1"))
    assert os.path.exists(sandbox.agent_dir(str(out), f"run-{runs[1]}-1"))


def test_what_this_machine_kept_for_the_agents_of_an_old_run_goes_with_them(
        _runs_left_behind):
    repo, out, runs = _runs_left_behind
    for run_id in runs:
        os.makedirs(os.path.join(sandbox.host_dir(str(out), f"run-{run_id}-1"), "repo",
                                 "turns"))
    alone = "20260104-000000-1-44"
    os.makedirs(sandbox.host_dir(str(out), alone))
    assert sandbox.remove_old_run(str(repo), str(out), 2) == [alone, runs[0]]
    assert not os.path.exists(sandbox.host_dir(str(out), f"run-{runs[0]}-1"))
    assert not os.path.exists(sandbox.host_dir(str(out), alone))
    assert os.path.exists(sandbox.host_dir(str(out), f"run-{runs[1]}-1"))


def test_only_the_requests_nobody_finished_are_counted(_runs_left_behind):
    """A run that got through leaves nothing to carry on from, so it neither takes up
    one of the places nor keeps its output."""
    repo, out, runs = _runs_left_behind
    got_through = "20260104-000000-1-44"
    os.makedirs(sandbox.agent_dir(str(out), got_through))
    assert sandbox.remove_old_run(str(repo), str(out), 3) == [got_through]
    assert f"codetwine-loom/run/{runs[0]}/1" in _branches(repo)
    assert not os.path.exists(sandbox.agent_dir(str(out), got_through))


def _leave_patch(out, run_id, text="diff --git a/a.txt b/a.txt\n", meta_dict=None):
    """Write the patch and the summary one run of its own leaves in its agent's copy,
    and the record this machine writes of the run on its side: a patch that holds
    something and names no commit is one left to pick up."""
    run_file = sandbox.run_file_layout(str(out), sandbox.agent_name("", run_id), "repo",
                                       run_id)
    os.makedirs(run_file.directory, exist_ok=True)
    pathlib.Path(run_file.patch).write_text(text, encoding="utf-8")
    pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(json.dumps(
        {"base": "b" * 40, "exit_status": 0, "files": 1, **(meta_dict or {})}),
        encoding="utf-8")
    sandbox._write_run_mark(run_file, sandbox._PATCH_KEEP
                            if text and "commit" not in (meta_dict or {}) else "")
    return run_file.patch


def test_a_run_whose_patch_landed_in_no_commit_takes_up_a_place_and_keeps_it(
        _runs_left_behind):
    """A dry run's patch, and one that could not be taken in, is the one place what
    that run changed is kept."""
    repo, out, runs = _runs_left_behind
    dry_run_id, land_run_id, empty_run_id = (
        "20260104-000000-1-44", "20260105-000000-1-55", "20260106-000000-1-66")
    patch = _leave_patch(out, dry_run_id)
    _leave_patch(out, land_run_id, meta_dict={"branch": "codetwine-loom/x", "commit": "c" * 40})
    _leave_patch(out, empty_run_id, text="")
    # The newest three that hold something to pick up: the dry run and two requests
    assert sandbox.remove_old_run(str(repo), str(out), 3) == [empty_run_id, land_run_id,
                                                              runs[0]]
    assert os.path.exists(patch) and f"codetwine-loom/run/{runs[1]}/1" in _branches(repo)
    # It is the newest of them, so it is the one kept when one place is left
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [runs[2], runs[1]]
    assert os.path.exists(patch)


def test_a_patch_that_landed_in_no_commit_goes_once_newer_runs_fill_the_places(
        _runs_left_behind):
    repo, out, runs = _runs_left_behind
    old_run = "20251231-000000-1-44"
    patch = _leave_patch(out, old_run)
    assert sandbox.remove_old_run(str(repo), str(out), 3) == [old_run]
    assert not os.path.exists(patch)


def test_the_records_of_the_runs_are_read_off_names_this_machine_makes(tmp_path):
    directory = tmp_path / "sandbox"
    directory.mkdir()
    wide = "\uff12\uff10\uff12\uff16\uff10\uff11\uff10\uff11-\uff10\uff10\uff10\uff10\uff10\uff10-1-1"
    for name, text in (("run-20260101-000000-1-2.txt", "patch"),
                       ("run-20260101-000000-1-3.txt", "answer"),
                       ("run-20260101-000000-1-4.txt", ""),
                       ("run-20260101-000000-1-5.txt", "else"),
                       ("run-x20260101-000000-1-6.txt", "patch"),
                       ("run-20260101-000000-1-7.txt.1230.tmp", "patch"),
                       (f"run-{wide}.txt", "patch"),
                       ("stdout-20260101-000000-1-8.txt", "")):
        (directory / name).write_text(text, encoding="utf-8")
    (directory / "run-20260101-000000-1-9.txt").mkdir()
    assert sandbox._run_mark_dict(str(directory)) == {
        "20260101-000000-1-2": "patch", "20260101-000000-1-3": "answer",
        "20260101-000000-1-4": "", "20260101-000000-1-5": "",
        "20260101-000000-1-9": "patch"}
    assert sandbox._run_mark_dict(str(tmp_path / "gone")) == {}
    assert sandbox.run_in(f"codetwine-loom/run/{wide}/1") == ""
    assert not sandbox.is_own_branch(f"codetwine-loom/run/{wide}/1")


# What a program inside a container can write into its own /out: a name carrying a
# run of another time or of process 1 (one in full-width digits too), a patch or an
# answer of no run this machine started, the answer file of its own run, and the
# same under the directory name of another repository
_PLANT_LIST = ["stdout-99991231-235959-1-1.txt", "stdout-20260101-000000-1-1.txt",
               "stdout-\uff12\uff10\uff12\uff16\uff10\uff11\uff10\uff11-"
               "\uff10\uff10\uff10\uff10\uff10\uff10-1-1.txt",
               "land-99991231-235959-1-1.diff", "outcome-99991231-235959-1-1.json",
               "patch-x.diff", "patch-99991231-235959-1-1.diff", "answer-x"]


@pytest.mark.parametrize("plant", [*_PLANT_LIST, "own answer", "other repository"])
def test_what_a_container_leaves_in_its_own_copy_decides_nothing_the_sweep_does(
        _runs_left_behind, plant):
    """The runs are told apart, put in order and counted by what this machine recorded
    on its side alone, so a run that got through stays one that holds nothing to pick
    up, and the oldest of the runs to carry on from is still the one that goes."""
    repo, out, runs = _runs_left_behind
    through = "20260104-000000-1-44"
    _leave_patch(out, through, meta_dict={"branch": "codetwine-loom/x", "commit": "c" * 40})
    agent = sandbox.agent_dir(str(out), sandbox.agent_name("", through))
    for run_id, place in ((runs[0], sandbox.agent_dir(str(out), runs[0])),
                          (through, agent)):
        name = "other" if plant == "other repository" else "repo"
        directory = os.path.join(place, name, "sandbox")
        os.makedirs(directory, exist_ok=True)
        file_name = {"own answer": f"answer-{run_id}.json",
                     "other repository": "patch-x.diff"}.get(plant, plant)
        pathlib.Path(directory, file_name).write_text(
            "" if file_name == "answer-x" else "x\n", encoding="utf-8")
    before = {run_id: {key: what[key] for key in ("runs", "answers", "patches")}
              for run_id, what in sandbox._run_remain_dict(str(repo), str(out)).items()}
    assert before[through] == {"runs": [through], "answers": [], "patches": []}
    assert before[runs[0]] == {"runs": [], "answers": [], "patches": []}
    assert sandbox.remove_old_run(str(repo), str(out), 2) == [through, runs[0]]
    assert f"codetwine-loom/run/{runs[1]}/1" in _branches(repo)
    if plant == "other repository":
        # Nor is it a run of the other repository that holds something to pick up
        other = _repo(out.parent / "other")
        assert all(not (what["runs"] or what["answers"] or what["patches"])
                   for what in sandbox._run_remain_dict(str(other), str(out)).values())


def _leave_in_other_repo(out, run_id, name="other", answer=False):
    """Write what one run leaves in its agent's directories for another repository of
    the same output directory: a patch that landed in no commit, the diff this
    machine keeps, and an answer when asked. Returns (the agent's directory, this
    machine's directory for the agent)."""
    agent = sandbox.agent_name("", run_id)
    run_file = sandbox.run_file_layout(str(out), agent, name, run_id)
    os.makedirs(run_file.directory)
    os.makedirs(run_file.host_directory)
    pathlib.Path(run_file.patch).write_text("diff --git a/a b/a\n", encoding="utf-8")
    pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(
        json.dumps({"exit_status": 0, "files": 1}), encoding="utf-8")
    pathlib.Path(run_file.land).write_text("diff --git a/a b/a\n", encoding="utf-8")
    sandbox._write_run_mark(run_file, sandbox._PATCH_KEEP)
    if answer:
        pathlib.Path(run_file.answer).write_text('{"answer": "x"}', encoding="utf-8")
        answer_file = sandbox.run_file_layout(str(out), agent, name, f"{run_id[:-2]}66")
        pathlib.Path(answer_file.answer).write_text('{"answer": "x"}', encoding="utf-8")
        sandbox._write_run_mark(answer_file, sandbox._ANSWER_KEEP)
    return sandbox.agent_dir(str(out), agent), sandbox.host_dir(str(out), agent)


def test_the_runs_one_command_made_take_up_one_of_the_places_between_them(
        _runs_left_behind):
    """A request that was run again leaves the branches of every one of its runs. The
    request before it is still one of the newest few."""
    repo, out, runs = _runs_left_behind
    again_list = ["20260103-000001-6-33", "20260103-000002-11-33"]
    for run_id in again_list:
        _git(repo, "branch", f"codetwine-loom/run/{run_id}/1")
        os.makedirs(sandbox.agent_dir(str(out), f"run-{run_id}-1"))
    # Three commands left runs to carry on from: processes 11, 22 and 33
    assert sandbox.remove_old_run(str(repo), str(out), 3) == []
    assert sandbox.remove_old_run(str(repo), str(out), 2) == [runs[0]]
    assert f"codetwine-loom/run/{runs[1]}/1" in _branches(repo)
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [runs[1]]
    assert [branch for branch in _branches(repo) if branch.startswith("codetwine-loom/run/")] == [
        f"codetwine-loom/run/{run_id}/{n}" for run_id in (runs[2], *again_list)
        for n in ((1, 2) if run_id == runs[2] else (1,))]
    for run_id in again_list:
        assert os.path.isdir(sandbox.agent_dir(str(out), f"run-{run_id}-1"))


def test_a_union_gone_between_the_listing_and_its_reading_stops_no_sweep(
        _runs_left_behind, monkeypatch, caplog):
    """Another command took a union's name away after the branches were listed. Its
    run is left where it is, the others go as they would, and nothing is raised."""
    repo, out, runs = _runs_left_behind
    gone = f"codetwine-loom/sandbox/{runs[0]}"
    _union_left(repo, runs[0], "unsettled")
    real_list = sandbox._run_remain_dict

    def listing(project_dir, out_dir):
        remain_dict = real_list(project_dir, out_dir)
        _git(repo, "branch", "-D", gone)
        return remain_dict

    monkeypatch.setattr(sandbox, "_run_remain_dict", listing)
    with caplog.at_level(logging.INFO, logger="codetwine_loom.sandbox"):
        assert sandbox.remove_old_run(str(repo), str(out), 1) == [runs[1]]
    assert f"{gone} cannot be read" in caplog.text
    assert f"codetwine-loom/run/{runs[0]}/1" in _branches(repo)
    assert sandbox._keep_branch_list(str(repo), ["codetwine-loom/sandbox/never-there"]) == [
        "codetwine-loom/sandbox/never-there"]


def test_what_a_run_wrote_for_another_repository_is_neither_counted_nor_taken_away(
        _runs_left_behind):
    """An output directory two repositories share holds the agents of both. The runs
    of the other one hold no branch here, and what they wrote stays whatever their
    age and however many places are kept."""
    repo, out, runs = _runs_left_behind
    old_pair = _leave_in_other_repo(out, "20251231-000000-1-44")
    new_pair_list = [_leave_in_other_repo(out, f"2026020{n}-000000-1-55", answer=True)
                     for n in (1, 2, 3)]
    remain_dict = sandbox._run_remain_dict(str(repo), str(out))
    assert sorted(remain_dict) == runs
    # The newer runs of the other repository take none of this repository's places
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [runs[1], runs[0]]
    assert f"codetwine-loom/run/{runs[2]}/1" in _branches(repo)
    for agent_dir, host_dir in [old_pair, *new_pair_list]:
        assert os.listdir(os.path.join(agent_dir, "other", "sandbox"))
        assert os.listdir(os.path.join(host_dir, "other", "sandbox"))


def test_an_agent_s_directory_two_repositories_wrote_in_loses_this_one_s_alone(
        _runs_left_behind):
    repo, out, runs = _runs_left_behind
    run_id = "20251231-000000-1-44"
    agent_dir, host_dir = _leave_in_other_repo(out, run_id)
    patch = _leave_patch(out, run_id)
    pathlib.Path(sandbox.filelock.path_for(os.path.join(agent_dir, "repo"))).touch()
    pathlib.Path(sandbox.filelock.path_for(agent_dir)).touch()
    os.makedirs(os.path.join(host_dir, "repo", "sandbox"), exist_ok=True)
    assert sandbox.remove_old_run(str(repo), str(out), 3) == [run_id]
    assert not os.path.exists(patch)
    assert sorted(os.listdir(agent_dir)) == ["other"] and os.listdir(host_dir) == ["other"]
    assert os.path.exists(sandbox.filelock.path_for(agent_dir))


def test_an_agent_s_directory_of_this_repository_alone_goes_with_its_locks(
        _runs_left_behind):
    repo, out, runs = _runs_left_behind
    run_id = "20251231-000000-1-44"
    patch = _leave_patch(out, run_id)
    agent_dir = sandbox.agent_dir(str(out), sandbox.agent_name("", run_id))
    lock_list = [sandbox.filelock.path_for(os.path.join(agent_dir, "repo")),
                 sandbox.filelock.path_for(agent_dir)]
    for lock in lock_list:
        pathlib.Path(lock).touch()
    assert sandbox.remove_old_run(str(repo), str(out), 3) == [run_id]
    assert not os.path.exists(patch) and not os.path.exists(agent_dir)
    assert not any(os.path.exists(lock) for lock in lock_list)


@pytest.mark.parametrize("entry", ["file", "link", "other"])
def test_a_directory_that_holds_nothing_for_the_repository_is_not_an_agent_s_of_it(
        tmp_path, entry):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    assert sandbox._is_agent_dir_of(str(agent_dir), "repo")
    (agent_dir / "repo").mkdir()
    assert sandbox._is_agent_dir_of(str(agent_dir), "repo")
    assert not sandbox._is_agent_dir_of(str(agent_dir), "other")
    assert not sandbox._is_agent_dir_of(str(tmp_path / "gone"), "repo")
    (agent_dir / "repo").rmdir()
    if entry == "file":
        (agent_dir / "repo").write_text("x", encoding="utf-8")
    elif entry == "link":
        (tmp_path / "elsewhere").mkdir()
        os.symlink(tmp_path / "elsewhere", agent_dir / "repo")
    else:
        (agent_dir / "other").mkdir()
    assert not sandbox._is_agent_dir_of(str(agent_dir), "repo")
    sandbox._remove_agent_dir(str(agent_dir), "repo", "what it held")
    assert os.listdir(agent_dir)
    link = tmp_path / "agent-link"
    os.symlink(agent_dir, link)
    assert not sandbox._is_agent_dir_of(str(link), "repo")


def test_nothing_is_taken_away_while_git_does_not_list_the_branches(_runs_left_behind,
                                                                   monkeypatch, caplog):
    """Without the branches every run looks like one nobody carries on from."""
    repo, out, runs = _runs_left_behind
    patch = _leave_patch(out, "20251231-000000-1-44")
    run = sandbox.gitcmd.run

    def no_list(project_dir, *args, **kwargs):
        if args[0] == "for-each-ref":
            return subprocess.CompletedProcess(args, 128, "", "fatal: no answer")
        return run(project_dir, *args, **kwargs)

    monkeypatch.setattr(sandbox.gitcmd, "run", no_list)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        assert sandbox.remove_old_run(str(repo), str(out), 1) == []
    assert "git did not list the branches" in caplog.text
    assert sandbox.remain_branch_list(str(repo), str(out), runs[0]) == []
    monkeypatch.setattr(sandbox.gitcmd, "run", run)
    assert os.path.exists(patch)
    assert all(f"codetwine-loom/run/{run_id}/1" in _branches(repo) for run_id in runs)
    assert os.path.exists(sandbox.agent_dir(str(out), runs[0]))


def test_the_branch_a_union_landed_on_is_never_taken_away(_runs_left_behind):
    """It is what the run produced, not scaffolding: one nobody has merged yet is the
    only copy of that result, and its account stays with it."""
    repo, out, runs = _runs_left_behind
    _git(repo, "branch", f"codetwine-loom/sandbox/{runs[0]}")
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [runs[1]]
    assert f"codetwine-loom/sandbox/{runs[0]}" in _branches(repo) and "main" in _branches(repo)
    assert os.path.exists(sandbox.agent_dir(str(out), runs[0]))


def test_a_run_this_one_was_told_to_read_is_left_alone(_runs_left_behind):
    """It was named on the command line, branch and all, so it is not housekeeping to
    take it away underneath whoever named it."""
    repo, out, runs = _runs_left_behind
    assert sandbox.remove_old_run(str(repo), str(out), 1,
                              protect_list=[f"codetwine-loom/run/{runs[0]}/1"]) == [runs[1]]
    assert f"codetwine-loom/run/{runs[0]}/2" in _branches(repo)
    assert os.path.exists(sandbox.agent_dir(str(out), f"run-{runs[0]}-1"))


def test_a_run_that_is_still_going_is_left_alone(_runs_left_behind, monkeypatch):
    """Its container is still writing in there, and its branches are still being made."""
    repo, out, runs = _runs_left_behind
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": False)
    assert sandbox.remove_old_run(str(repo), str(out), 1) == []
    assert f"codetwine-loom/run/{runs[0]}/1" in _branches(repo)


def test_the_agents_of_one_question_take_up_one_place(_runs_left_behind, monkeypatch):
    """Every answering agent and the combining run are named under the dispatching run,
    so a question split in parts is one run to the sweep and older unfinished requests
    keep their places."""
    repo, out, runs = _runs_left_behind
    owner = "20260201-000000-1-55"
    id_iter = iter(f"20260201-000001-{n}-55" for n in range(2, 10))
    monkeypatch.setattr(sandbox, "new_run_id", lambda: next(id_iter))

    def start(*a, **k):
        return _split_written(a[8].answer, {"answer": "in a.py"})(*a, **k)

    monkeypatch.setattr(sandbox, "_start_container", start)
    for question in ("one?", "two?", "together?"):
        assert sandbox.answer(str(repo), str(out), question, start="HEAD",
                              owner=owner)["answer"] == "in a.py"
    assert len(sandbox._run_remain_dict(str(repo), str(out))[owner]["answers"]) == 3
    run_list = {sandbox.run_in(agent) for agent, _ in sandbox.output_dir_list(str(out))[1:]}
    assert owner in run_list and not any(r.startswith("20260201-000001") for r in run_list)
    assert sandbox.remove_old_run(str(repo), str(out), 3, protect_list=[owner]) == [runs[0]]
    assert f"codetwine-loom/run/{runs[1]}/1" in _branches(repo)


def test_keeping_nothing_is_the_way_to_switch_the_sweep_off(_runs_left_behind):
    repo, out, runs = _runs_left_behind
    assert sandbox.remove_old_run(str(repo), str(out), 0) == []
    assert f"codetwine-loom/run/{runs[0]}/1" in _branches(repo)


def test_a_dispatching_run_takes_away_what_older_runs_left(_runs_left_behind,
                                                           monkeypatch):
    """What this run leaves is the newest, so it survives its own sweep."""
    repo, out, runs = _runs_left_behind
    monkeypatch.setattr(config, "DISPATCH_KEEP_RUN", 1)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 0, "verdict": "unverified"})
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
                            "exit_status": 0, "commit": "u" * 40,
                            "branch": "codetwine-loom/sandbox/u"})
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["swept"] == runs[1::-1]
    assert not any(b.startswith(f"codetwine-loom/run/{runs[0]}/") for b in _branches(repo))
    # The run doing the sweeping is named rather than left to the ordering
    assert os.path.exists(sandbox.agent_dir(str(out), "20260101-000000-1-42"))


def test_what_the_whole_thing_was_asked_for_is_what_the_merge_commits(_dispatch_repo,
                                                                      monkeypatch):
    """A run carrying on from the merge reads its request back off that commit, and
    "combine A, B" is not a request anybody made."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 0, "verdict": "unverified"})
    asked = {}
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: asked.update(k) or {
        "exit_status": 0, "commit": "u" * 40, "branch": "codetwine-loom/combined"})
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert asked["request"] == "設定を整理して"


def test_what_the_whole_thing_was_asked_for_reaches_every_agents_commit(_dispatch_repo,
                                                                        monkeypatch):
    """Each agent commits the task it was handed, so a run carrying on from one of
    those branches reads the request off the trailer this puts there."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    asked = []
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: asked.append(k["run_request"]) or {
        "commit": "c" * 40, "exit_status": 0, "verdict": "unverified"})
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        "exit_status": 0, "commit": "u" * 40, "branch": "codetwine-loom/combined"})
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert asked == ["設定を整理して", "設定を整理して"]


def test_every_agent_is_handed_the_instruction_file_this_machine_is_set_to(_dispatch_repo,
                                                                          monkeypatch):
    """EDIT_GUIDE is not forwarded by name, so the flag each agent's container gets is
    the only way the setting reaches it."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox.config, "EDIT_GUIDE", "docs/HOUSE.md")
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    asked = []
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: asked.append(k["guide"]) or {
        "commit": "c" * 40, "exit_status": 0, "verdict": "unverified"})
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        "exit_status": 0, "commit": "u" * 40, "branch": "codetwine-loom/combined"})
    monkeypatch.setattr(sandbox, "drop_branch_under", lambda *a, **k: [])
    sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert asked == ["docs/HOUSE.md", "docs/HOUSE.md"]


def _answered_with(*answers: str):
    """Return a sandbox.answer() stand-in that answers each question with the next
    of these, and the combining run with the last."""
    seq = list(answers)

    def start(project, output_dir, question, **kw):
        said = seq.pop(0) if len(seq) > 1 else seq[0]
        return {"answer": said, "exit_status": 0}

    return start


def test_a_question_starts_one_answering_agent_each_and_then_the_combining_run(
        _dispatch_repo, monkeypatch):
    """The agents see their question alone; the combining run sees the request and
    every answer, and what it says is what comes back."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"questions": ["where?",
                                                                        "how?"]}))
    monkeypatch.setattr(sandbox, "run", _never("an editing agent was started"))
    asked = []

    def answering(project, output_dir, question, answer_list=None, **kw):
        asked.append((question, answer_list))
        if answer_list is None:
            return {"answer": f"answer to {question}", "exit_status": 0}
        return {"answer": "both, together", "exit_status": 0}

    monkeypatch.setattr(sandbox, "answer", answering)
    result = sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert [q for q, a in asked[:2]] == ["where?", "how?"]
    assert asked[2][0] == "どこで読んでいる?"
    assert [e["answer"] for e in asked[2][1]] == ["answer to where?", "answer to how?"]
    assert result["answer"] == "both, together" and result["exit_status"] == 0
    assert result["questions"] == ["where?", "how?"]
    assert result["combined"] == {"answer": "both, together", "exit_status": 0,
                                  "error": ""}
    # Nothing was committed anywhere: an answer is not a change
    assert _git(repo, "branch", "--list").strip() == "* main"


def test_a_combining_run_that_handed_nothing_back_is_run_again_over_the_same_answers(
        _dispatch_repo, monkeypatch):
    """The agents are not asked again: what they answered is in hand, and asking them
    again would pay for every one of them."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 1)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"questions": ["where?"]}))
    monkeypatch.setattr(sandbox, "run", _never("an editing agent was started"))
    asked = []

    def answering(project, output_dir, question, answer_list=None, **kw):
        asked.append(answer_list is not None)
        if answer_list is None:
            return {"answer": "answer to where?", "exit_status": 0}
        # The first combining run hands nothing back; the second one answers
        return {"answer": "" if asked.count(True) == 1 else "both, together",
                "exit_status": 1 if asked.count(True) == 1 else 0}

    monkeypatch.setattr(sandbox, "answer", answering)
    result = sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert asked == [False, True, True]
    assert result["answer"] == "both, together" and result["exit_status"] == 0
    assert result["combine_runs"] == 2


def test_the_combining_run_is_run_again_no_more_than_it_is_allowed(_dispatch_repo,
                                                                  monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "INTEGRATE_MAX_RERUN", 2)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"questions": ["where?"]}))
    monkeypatch.setattr(sandbox, "run", _never("an editing agent was started"))
    count_list = []

    def answering(project, output_dir, question, answer_list=None, **kw):
        if answer_list is None:
            return {"answer": "answer to where?", "exit_status": 0}
        count_list.append(1)
        return {"answer": "", "exit_status": 1}

    monkeypatch.setattr(sandbox, "answer", answering)
    result = sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert len(count_list) == 3 and result["combine_runs"] == 3
    assert result["exit_status"] == 1


def test_an_agent_that_answered_nothing_is_asked_again(_dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 1)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"questions": ["where?"]}))
    tries = []

    def answering(project, output_dir, question, answer_list=None, **kw):
        if answer_list is not None:
            return {"answer": "together", "exit_status": 0}
        tries.append(question)
        if len(tries) == 1:
            raise RuntimeError("the run that answers handed no answer back")
        return {"answer": "second time", "exit_status": 0}

    monkeypatch.setattr(sandbox, "answer", answering)
    result = sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert tries == ["where?", "where?"]
    assert result["answers"][0]["attempt"] == 2
    assert result["answers"][0]["answer"] == "second time"
    assert result["exit_status"] == 0


def test_a_question_still_unanswered_reaches_the_combining_run_as_such(_dispatch_repo,
                                                                       monkeypatch):
    """A part of the request nobody answered is said to be open, not left out."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"questions": ["where?",
                                                                        "how?"]}))
    handed = {}

    def answering(project, output_dir, question, answer_list=None, **kw):
        if answer_list is not None:
            handed["answer_list"] = answer_list
            return {"answer": "where is answered; how is open", "exit_status": 0}
        if question == "how?":
            raise RuntimeError("the run that answers handed no answer back")
        return {"answer": "here", "exit_status": 0}

    monkeypatch.setattr(sandbox, "answer", answering)
    result = sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert handed["answer_list"][1]["answer"] == ""
    assert "handed no answer back" in handed["answer_list"][1]["error"]
    assert result["answer"] == "where is answered; how is open"
    # There is an answer to read, so the run got through: asking the whole request
    # again would pay for every agent that did answer
    assert result["exit_status"] == 0


@pytest.mark.parametrize("error, why", [
    (RuntimeError("the run that answers handed no answer back"),
     "the run that answers handed no answer back"),
    (OSError(28, "No space left on device"),
     "OSError: [Errno 28] No space left on device")])
def test_a_combining_run_that_hands_nothing_back_leaves_the_answers_as_they_came(
        _dispatch_repo, monkeypatch, error, why):
    """Whatever the combining run raised, the agents' answers are what comes back."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"questions": ["where?"]}))

    def answering(project, output_dir, question, answer_list=None, **kw):
        if answer_list is not None:
            raise error
        return {"answer": "in a.py", "exit_status": 0}

    monkeypatch.setattr(sandbox, "answer", answering)
    result = sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert "in a.py" in result["answer"] and "where?" in result["answer"]
    assert result["combined"]["error"] == why
    assert result["exit_status"] == 1


def test_a_split_in_the_old_answer_shape_starts_nothing(_dispatch_repo, monkeypatch):
    """The reading run no longer answers, so an answer from it is not a split."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"answer": "it is a.py"}))
    monkeypatch.setattr(sandbox, "answer", _never("an answering agent was started"))
    monkeypatch.setattr(sandbox, "run", _never("an editing agent was started"))
    with pytest.raises(RuntimeError, match="neither any question nor any task"):
        sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")


# == the answering run itself ================================
def test_the_container_is_asked_to_answer_and_told_where_the_answer_goes():
    argv = sandbox.answer_argv("repo", "どこで?", "/out/repo/sandbox/a.json")
    assert argv[0] == "answer" and "--here" not in argv
    assert "--request=どこで?" in argv and "/out/repo/sandbox/a.json" in argv
    assert f"{sandbox._WORK}/repo" in argv
    assert "--answered" not in argv
    combining = sandbox.answer_argv("repo", "q", "/out/a.json", "/out/answered.json")
    assert combining[-2:] == ["--answered", "/out/answered.json"]


def test_an_answering_run_reads_the_answer_the_container_wrote(_dispatch_repo,
                                                               monkeypatch):
    repo, out = _dispatch_repo
    files = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42")
    told = []

    def start(*a, **k):
        told.append(a[-2])
        return _split_written(files.answer, {"answer": "in a.py"})(*a, **k)

    monkeypatch.setattr(sandbox, "_start_container", start)
    result = sandbox.answer(str(repo), str(out), "どこで?")
    assert result == {"answer": "in a.py", "exit_status": 0}
    assert told[0][0] == "answer" and files.answer_inside in told[0]
    assert not os.path.exists(files.answer_in)


def test_a_run_that_wrote_its_answer_and_then_ended_non_zero_answered(_dispatch_repo,
                                                                     monkeypatch):
    """The answer is what the run was for, so it is neither handed out again nor
    counted as a question nobody answered."""
    repo, out = _dispatch_repo
    files = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42")
    write = _split_written(files.answer, {"answer": "in a.py"})

    def start(*a, **k):
        write(*a, **k)
        return "/usr/bin/podman", "img", 137

    monkeypatch.setattr(sandbox, "_start_container", start)
    assert sandbox.answer(str(repo), str(out), "どこで?") == {"answer": "in a.py",
                                                            "exit_status": 0}


def test_a_combining_run_is_handed_the_answers_before_it_starts(_dispatch_repo,
                                                                monkeypatch):
    repo, out = _dispatch_repo
    files = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42")
    seen = {}

    def start(*a, **k):
        with open(files.answer_in, encoding="utf-8") as handle:
            seen["answer_list"] = json.load(handle)
        seen["argv"] = a[-2]
        return _split_written(files.answer, {"answer": "together"})(*a, **k)

    monkeypatch.setattr(sandbox, "_start_container", start)
    answer_list = [{"n": 1, "request": "where?", "attempt": 1, "answer": "here",
                 "exit_status": 0, "error": ""}]
    result = sandbox.answer(str(repo), str(out), "q", answer_list=answer_list)
    assert result["answer"] == "together"
    assert seen["answer_list"] == answer_list
    assert seen["argv"][-2:] == ["--answered", files.answer_in_inside]


def test_a_run_that_wrote_no_answer_is_reported_with_its_reason(_dispatch_repo,
                                                                monkeypatch):
    repo, out = _dispatch_repo
    files = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42")
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: ("/usr/bin/podman", "img", 1))
    with pytest.raises(RuntimeError, match="left no answer"):
        sandbox.answer(str(repo), str(out), "q")
    monkeypatch.setattr(sandbox, "_start_container", _split_written(
        files.answer, {"error": "the answer stage handed nothing over"}))
    with pytest.raises(RuntimeError, match="handed nothing over"):
        sandbox.answer(str(repo), str(out), "q")


def test_one_agent_is_started_per_task_each_on_a_branch_of_its_own(_dispatch_repo,
                                                                   monkeypatch):
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["rename it", "fix callers"]}))
    started = []

    def fake_run(project, output_dir, request, **kw):
        started.append((request, kw["branch"]))
        return {"commit": "c" * 40, "exit_status": 0, "verdict": "unverified"}

    monkeypatch.setattr(sandbox, "run", fake_run)
    combined = []
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: combined.append(a[2]) or {
                            "exit_status": 0, "commit": "u" * 40,
                            "branch": "codetwine-loom/sandbox/u"})
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert [r for r, _ in started] == ["rename it", "fix callers"]
    assert [b for _, b in started] == ["codetwine-loom/run/20260101-000000-1-42/1",
                                       "codetwine-loom/run/20260101-000000-1-42/2"]
    # The branches carry the dispatching run, so a second dispatch never continues these
    assert combined == [[b for _, b in started]]
    assert result["tasks"] == ["rename it", "fix callers"]


def test_one_task_is_combined_too_against_the_commit_it_started_from(_dispatch_repo,
                                                                     monkeypatch):
    """One branch is still held to the merging run's gates, which is what reads a test
    the agent left red; the base is what every agent started at."""
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["rename it"]}))
    monkeypatch.setattr(sandbox, "run",
                        lambda *a, **k: {"commit": "c" * 40, "exit_status": 0,
                                     "verdict": "unverified"})
    seen = {}
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: seen.update(branches=a[2], base=k["base"]) or {
                            "exit_status": 0, "commit": "u" * 40,
                            "branch": "codetwine-loom/sandbox/u"})
    result = sandbox.dispatch(str(repo), str(out), "名前を変えて")
    assert seen["branches"] == ["codetwine-loom/run/20260101-000000-1-42/1"]
    assert seen["base"] == _git(repo, "rev-parse", "HEAD").strip()
    assert result["integration"]["exit_status"] == 0


def test_an_agent_that_did_not_finish_does_not_stop_the_others(_dispatch_repo,
                                                               monkeypatch):
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one", "two"]}))

    def fake_run(project, output_dir, request, **kw):
        if request == "one":
            raise RuntimeError("the patch does not apply")
        return {"commit": "c" * 40, "exit_status": 0, "verdict": "unverified"}

    monkeypatch.setattr(sandbox, "run", fake_run)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["branches"][0]["error"] == "the patch does not apply"
    assert result["branches"][1]["commit"] == "c" * 40
    # The failure is reported rather than swallowed
    assert result["exit_status"] == 1


def test_a_task_that_did_not_finish_is_handed_out_again_on_a_branch_of_its_own(
        _dispatch_repo, monkeypatch):
    """Continuing the branch that failed would start the next try on half a change."""
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one", "two"]}))
    started = []

    def fake_run(project, output_dir, request, **kw):
        started.append((request, kw["branch"]))
        # The first try of "one" ends without saying anything, as a run stopped
        # part-way does, and still leaves a commit on its branch
        died = request == "one" and len(started) < 3
        return {"commit": "c" * 40, "exit_status": 1 if died else 0,
                "verdict": "unverified"}

    monkeypatch.setattr(sandbox, "run", fake_run)
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: {"exit_status": 0, "branches": a[2],
                                     "commit": "u" * 40, "branch": "codetwine-loom/sandbox/u"})
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert ("one", "codetwine-loom/run/20260101-000000-1-42/1-retry1") in started
    assert result["branches"][0]["attempt"] == 2
    assert result["branches"][0]["exit_status"] == 0
    # The second try is what the task is reported as, and it is what is combined
    assert result["integration"]["branches"] == [
        "codetwine-loom/run/20260101-000000-1-42/1-retry1",
        "codetwine-loom/run/20260101-000000-1-42/2"]
    assert result["exit_status"] == 0


def test_every_agent_works_under_a_label_naming_its_task_and_its_try(
        _dispatch_repo, monkeypatch, caplog):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    label_list = []

    def fake_run(project, output_dir, request, **kw):
        label_list.append(progress.label_text())
        is_first_try_of_one = request == "one" and not kw["branch"].endswith("-retry1")
        return {"commit": "c" * 40, "exit_status": 1 if is_first_try_of_one else 0,
                "verdict": "unverified"}

    monkeypatch.setattr(sandbox, "run", fake_run)
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: {"exit_status": 0, "branches": a[2],
                                         "commit": "u" * 40, "branch": "codetwine-loom/sandbox/u"})
    with caplog.at_level("INFO", logger="codetwine_loom.sandbox"), \
            caplog.at_level("INFO", logger="codetwine_loom.progress"):
        sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert sorted(label_list) == ["[task 1/2 try 2] ", "[task 1/2] ", "[task 2/2] "]
    assert progress.label_text() == ""
    message_list = [r.getMessage() for r in caplog.records]
    assert "task 1 of 2: one" in message_list and "task 2 of 2: two" in message_list
    assert message_list.count("task 1 of 2 starts") == 2
    assert any(m.startswith("task 1 of 2 did not finish in ") for m in message_list)
    end_list = [m for m in message_list if " finished (unverified) in " in m]
    assert len(end_list) == 2
    assert end_list[-1].startswith("task 1 of 2 finished (unverified) in ")
    assert end_list[-1].endswith(": 2 of 2 finished, 0 running")


def test_a_question_is_counted_as_a_question(_dispatch_repo, monkeypatch, caplog):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"questions": ["where?"]}))
    monkeypatch.setattr(sandbox, "answer", _answered_with("it is a.py"))
    with caplog.at_level("INFO", logger="codetwine_loom.sandbox"), \
            caplog.at_level("INFO", logger="codetwine_loom.progress"):
        sandbox.dispatch(str(repo), str(out), "どこ？")
    message_list = [r.getMessage() for r in caplog.records]
    assert "question 1 of 1: where?" in message_list
    assert any(m.startswith("question 1 of 1 finished in ")
               and m.endswith(": 1 of 1 finished, 0 running") for m in message_list)


def test_the_branch_of_a_try_done_again_goes_once_the_run_got_through(_dispatch_repo,
                                                                      monkeypatch):
    """It held half a change the next try started over from, and left there it would
    count a run that got through as one to carry on from."""
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one"]}))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(
        repo, lambda branch: 0 if branch.endswith("-retry1") else 1))
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        "exit_status": 0, "commit": "u" * 40, "branch": "codetwine-loom/sandbox/u"})
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["exit_status"] == 0
    assert result["dropped"] == ["codetwine-loom/run/20260101-000000-1-42/1"]
    assert "codetwine-loom/run/20260101-000000-1-42/1" not in _branches(repo)
    assert "codetwine-loom/run/20260101-000000-1-42/1-retry1" in _branches(repo)


def test_the_branch_of_a_try_done_again_stays_while_the_run_did_not_get_through(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one"]}))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 1))
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 1)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["exit_status"] == 1 and result["dropped"] == []
    assert "codetwine-loom/run/20260101-000000-1-42/1" in _branches(repo)


def test_a_task_still_unfinished_after_the_last_try_leaves_nothing_combined(
        _dispatch_repo, monkeypatch):
    """A run stopped part-way still writes what it had, and that commit is on its
    branch: a union missing one of the tasks would read as the whole request done."""
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one", "two"]}))
    monkeypatch.setattr(
        sandbox, "run",
        lambda project, output_dir, request, **kw: {
            "commit": "c" * 40, "exit_status": 1 if request == "one" else 0,
            "verdict": "unverified"})
    monkeypatch.setattr(sandbox, "integrate",
                        _never("a task that did not finish was combined in"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["integration"] is None
    assert result["exit_status"] == 1
    # The branches are left where they are, so what each agent got to can be read
    assert [e["branch"] for e in result["branches"]] == [
        "codetwine-loom/run/20260101-000000-1-42/1-retry1",
        "codetwine-loom/run/20260101-000000-1-42/2"]


def test_a_task_asked_for_again_is_handed_out_no_more_than_the_setting_says(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one"]}))
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 2)
    started = []
    monkeypatch.setattr(sandbox, "run",
                        lambda project, output_dir, request, **kw:
                        started.append(kw["branch"]) or {"commit": "", "exit_status": 1,
                                                         "verdict": "failed"})
    result = sandbox.dispatch(str(repo), str(out), "名前を変えて")
    assert started == ["codetwine-loom/run/20260101-000000-1-42/1",
                       "codetwine-loom/run/20260101-000000-1-42/1-retry1",
                       "codetwine-loom/run/20260101-000000-1-42/1-retry2"]
    assert result["branches"][0]["attempt"] == 3


def test_a_task_killed_at_the_memory_limit_is_not_handed_out_again(_dispatch_repo,
                                                                   monkeypatch, caplog):
    repo, out = _dispatch_repo
    task_path = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(task_path, {"tasks": ["one"]}))
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 2)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    branch_list = []
    monkeypatch.setattr(sandbox, "run",
                        lambda project, output_dir, request, **kw:
                        branch_list.append(kw["branch"])
                        or {"commit": "", "exit_status": 137, "verdict": "failed"})
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        result = sandbox.dispatch(str(repo), str(out), "名前を変えて")
    assert branch_list == ["codetwine-loom/run/20260101-000000-1-42/1"]
    assert result["exit_status"] == 137 and result["runs"] == 1
    assert "1 task(s) did not finish (exit status 137" in caplog.text


def test_an_agent_that_landed_nothing_and_finished_is_not_asked_again(_dispatch_repo,
                                                                     monkeypatch):
    """Ending with nothing changed is the run saying the task needed no change."""
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one", "two"]}))
    started = []
    monkeypatch.setattr(
        sandbox, "run",
        lambda project, output_dir, request, **kw: started.append(request) or {
            "commit": "" if request == "one" else "c" * 40, "exit_status": 0,
            "verdict": "unverified"})
    combined = []
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: combined.append(a[2]) or {
                            "exit_status": 0, "commit": "u" * 40,
                            "branch": "codetwine-loom/sandbox/u"})
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert started == ["one", "two"]
    assert result["exit_status"] == 0
    # The branch that landed something is combined on its own
    assert combined == [["codetwine-loom/run/20260101-000000-1-42/2"]]


def test_a_red_test_is_not_handed_out_again_and_reaches_the_merge(_dispatch_repo,
                                                                  monkeypatch):
    """A change meant to change behaviour leaves old tests red; handing it out again
    would only repeat it. The merging stage holds the union against the base."""
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one", "two"]}))
    started = []
    monkeypatch.setattr(
        sandbox, "run",
        lambda project, output_dir, request, **kw: started.append(kw["branch"]) or {
            "commit": "c" * 40, "exit_status": 0, "verdict": "tests_red"})
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: {"exit_status": 0, "branches": a[2],
                                     "commit": "u" * 40, "branch": "codetwine-loom/sandbox/u"})
    result = sandbox.dispatch(str(repo), str(out), "戻り値を変えて")
    # No retry branch among them, and both were combined
    assert started == ["codetwine-loom/run/20260101-000000-1-42/1",
                       "codetwine-loom/run/20260101-000000-1-42/2"]
    assert result["integration"]["branches"] == started
    assert result["exit_status"] == 0


def test_a_red_test_on_one_task_is_read_by_the_merging_run(_dispatch_repo,
                                                           monkeypatch, caplog):
    """One task alone is combined too, so its red test is held against the commit it
    started from rather than reported as one nothing read."""
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 0, "verdict": "tests_red"})
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        "exit_status": 0, "commit": "u" * 40, "branch": "codetwine-loom/sandbox/u"})
    result = sandbox.dispatch(str(repo), str(out), "戻り値を変えて")
    assert result["exit_status"] == 0
    assert "nothing read those tests" not in caplog.text


def test_a_red_test_with_nothing_combined_afterwards_is_reported(_dispatch_repo,
                                                                 monkeypatch, caplog):
    """No branch holds anything, so no merging run read the test."""
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "", "exit_status": 0, "verdict": "tests_red"})
    monkeypatch.setattr(sandbox, "integrate", _never("no branch holds anything"))
    result = sandbox.dispatch(str(repo), str(out), "戻り値を変えて")
    assert result["exit_status"] == 1
    assert "nothing read those tests" in caplog.text


def test_a_split_that_never_arrived_stops_the_run(_dispatch_repo, monkeypatch):
    """Starting agents on a guess at what was asked would be worse than stopping."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: ("/usr/bin/podman", "img", 1))
    monkeypatch.setattr(sandbox, "run", _never("an agent was started with no task"))
    with pytest.raises(RuntimeError, match="no split"):
        sandbox.dispatch(str(repo), str(out), "設定を整理して")


# == the cache is filled by the first run that needs it =====
_REAL_FILL = sandbox.fill_cache_once
def test_a_cache_nothing_has_filled_is_filled_before_the_container_starts(
        tmp_path, monkeypatch):
    """Nobody has to remember to run warm first: a run finding no mark of a filling
    fills the cache itself, before the container that cannot fetch anything."""
    repo = _repo(tmp_path / "repo")
    seen, fillings = [], []
    _stand_in_for_the_container(monkeypatch, seen)
    monkeypatch.setattr(sandbox, "fill_cache_once", _REAL_FILL)
    monkeypatch.setattr(sandbox, "warm", lambda *a, **k: fillings.append(
        (a, k, len(seen))) or {"exit_status": 0})
    sandbox.run(str(repo), str(tmp_path / "out"), "x")
    assert len(fillings) == 1
    (args, kwargs, containers_before) = fillings[0]
    assert kwargs["once"] is True and args[0] == str(repo)
    assert kwargs["base"] == gitcmd.require_head(str(repo))
    assert containers_before == 0          # before the change container was started


def test_a_filling_that_could_not_be_made_does_not_stop_the_run(tmp_path,
                                                                monkeypatch, caplog):
    """It is tried again once for the same dependency files in the same command; other
    dependency files are filled whatever failed before."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    call_list = []
    monkeypatch.setattr(sandbox, "warm", lambda *a, **k: call_list.append(a)
                        or {"exit_status": 125, "skip": False})

    def fill():
        return _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", str(tmp_path / "cache"),
                          "/usr/bin/true", "img", None,
                          _git(repo, "rev-parse", "HEAD").strip())
    for _ in range(3):
        assert fill() == 0
    assert "could not be filled" in caplog.text
    assert len(call_list) == 2
    (repo / "pyproject.toml").write_text("[project]\nname = 'changed'\n", encoding="utf-8")
    _git(repo, "commit", "--quiet", "-am", "changed")
    fill()
    assert len(call_list) == 3


def test_containers_started_at_once_do_not_each_repeat_a_failing_filling(tmp_path,
                                                                         monkeypatch):
    repo = _repo(tmp_path / "repo")
    call_list = []

    def slow_failure(*a, **k):
        call_list.append(a)
        time.sleep(0.3)
        return {"exit_status": 1}
    monkeypatch.setattr(sandbox, "warm", slow_failure)
    head = _git(repo, "rev-parse", "HEAD").strip()
    thread_list = [threading.Thread(target=_REAL_FILL, args=(
        str(repo), str(tmp_path / "out"), "a1", str(tmp_path / "cache"), "/usr/bin/true",
        "img", None, head)) for _ in range(4)]
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
    assert len(call_list) == sandbox._FILL_MAX_FAIL


def test_a_container_whose_cache_is_filled_does_not_wait_for_another_filling(tmp_path,
                                                                             monkeypatch):
    repo = _repo(tmp_path / "repo")
    head = _git(repo, "rev-parse", "HEAD").strip()
    cache = str(tmp_path / "cache")
    call_list = []

    def slow_failure(*a, **k):
        call_list.append(k.get("image"))
        time.sleep(1.0)
        return {"exit_status": 1}
    monkeypatch.setattr(sandbox, "warm", slow_failure)
    fill_fingerprint = sandbox._dep_fingerprint(str(repo), head, "", [],
                                      sandbox._image_id("/usr/bin/true", "filled"))
    os.makedirs(cache, exist_ok=True)
    pathlib.Path(sandbox.fill_mark_path(cache)).write_text(fill_fingerprint + "\n",
                                                           encoding="utf-8")
    thread = threading.Thread(target=_REAL_FILL, args=(
        str(repo), str(tmp_path / "out"), "a1", cache, "/usr/bin/true", "other", None,
        head))
    thread.start()
    time.sleep(0.2)
    start_time = time.monotonic()
    assert _REAL_FILL(str(repo), str(tmp_path / "out"), "a2", cache, "/usr/bin/true",
                      "filled", None, head) == 0
    assert time.monotonic() - start_time < 0.5
    thread.join()
    assert call_list == ["other"]


def _commit_file(repo, name, text="x\n"):
    """Commit one file in repo and return the new commit."""
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", f"add {name}")
    return _git(repo, "rev-parse", "HEAD").strip()


def test_the_languages_a_container_adds_are_those_whose_tests_the_start_does_not_run(
        tmp_path, monkeypatch):
    """A file of a language the start already held does not keep that language out:
    a helper script is not a test the start's run runs."""
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    repo = _repo(tmp_path / "repo")
    for name in ("MAIN.cbl", "OTHER.cbl", "gen.py"):
        _commit_file(repo, name)
    sandbox.start_tree(str(repo))
    out = tmp_path / "out"
    (out / "repo").mkdir(parents=True)
    defs = {f"repo/{path.replace('.', '_')}/{path}::X": {
        "name": "X", "type": type_, "file": f"repo/{path.replace('.', '_')}/{path}",
        "lines": [1, 2], "signature": "X", "doc": ""}
        for path, type_ in (("MAIN.cbl", "paragraph_header"),
                            ("OTHER.cbl", "paragraph_header"),
                            ("gen.py", "function_definition"))}
    write_index(str(out / "repo" / store.INDEX_FILE), defs, {}, "repo").close()
    for name in ("main.py", "query.sql", "lib.mjs", "view.tsx", "MORE.cpy"):
        head = _commit_file(repo, name)
    # SQL has no test command, .cpy is COBOL's, and TypeScript (.tsx) runs the test
    # command JavaScript (.mjs) runs
    assert sandbox.new_test_ext_list(str(repo), str(out), head) == ["js", "py"]
    start = sandbox.start_tree(str(repo))
    assert sandbox.new_test_ext_list(str(repo), str(out), start) == []
    assert sandbox.new_test_ext_list(str(repo), str(out), start, [head]) == ["js", "py"]
    # The index's primary language has its tests run over the start
    _index_in(tmp_path / "out_py", "function_definition", "repo/a_py/a.py")
    assert sandbox.new_test_ext_list(str(repo), str(tmp_path / "out_py"), head) == ["js"]
    monkeypatch.setenv("TEST_COMMAND", "make test")
    assert sandbox.new_test_ext_list(str(repo), str(out), head) == []
    monkeypatch.delenv("TEST_COMMAND")
    monkeypatch.setattr(config, "RUN_TEST", False)
    assert sandbox.new_test_ext_list(str(repo), str(out), head) == []


def test_a_language_whose_test_command_the_start_already_runs_is_not_added(
        tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    repo = _repo(tmp_path / "repo")
    _commit_file(repo, "a.c")
    sandbox.start_tree(str(repo))
    _index_in(tmp_path / "out", "function_definition", "repo/a_c/a.c")
    # A header is C++'s by its extension, and C++ runs the test command C runs
    head = _commit_file(repo, "a.h")
    assert sandbox.new_test_ext_list(str(repo), str(tmp_path / "out"), head) == []


def test_a_container_given_languages_a_start_has_no_test_of_fills_their_tests(
        tmp_path, monkeypatch):
    """The languages are filled though the start's index has no test to run."""
    repo = _repo(tmp_path / "repo")
    head = _commit_file(repo, "MAIN.cbl")
    call_list = []
    monkeypatch.setattr(sandbox, "_has_test_to_run", lambda *a: False)
    monkeypatch.setattr(sandbox, "warm", lambda *a, **k: call_list.append(k)
                        or {"exit_status": 0, "skip": False})

    def fill(fill_ext_list):
        return _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", str(tmp_path / "cache"),
                          "/usr/bin/true", "img", None, head, fill_ext_list)
    assert fill([]) == 0
    assert call_list == []
    assert fill(["py"]) == 0
    assert [k["fill_ext_list"] for k in call_list] == [["py"]]
    assert call_list[0]["base"] == sandbox.start_tree(str(repo))


def test_a_filling_for_languages_alone_leaves_out_the_tests_already_filled_for(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    head = _commit_file(repo, "MAIN.cbl")
    cache = str(tmp_path / "cache")
    call_list = []
    monkeypatch.setattr(sandbox, "warm", lambda *a, **k: call_list.append(k)
                        or {"exit_status": 0, "skip": False})

    def fill():
        return _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", cache, "/usr/bin/true",
                          "img", None, head, ["py"])
    fill()
    os.makedirs(cache, exist_ok=True)
    pathlib.Path(sandbox.fill_mark_path(cache)).write_text(sandbox._dep_fingerprint(
        str(repo), head, "", [], sandbox._image_id("/usr/bin/true", "img")) + "\n",
        encoding="utf-8")
    fill()
    assert [k["fill_own"] for k in call_list] == [True, False]


def test_a_filling_told_to_leave_out_the_index_s_own_tests_is_given_so(monkeypatch):
    argv = _argv(fill_ext_list=["py"], fill_own=False)
    assert f"{sandbox.CTLOOM_FILL_OWN_ENV}=0" in argv
    assert sandbox.CTLOOM_FILL_OWN_ENV not in " ".join(_argv(fill_ext_list=["py"]))
    monkeypatch.setenv(sandbox.CTLOOM_FILL_OWN_ENV, "0")
    assert sandbox.fill_own_run() is False
    monkeypatch.delenv(sandbox.CTLOOM_FILL_OWN_ENV)
    assert sandbox.fill_own_run() is True


def test_a_container_s_filling_and_refills_are_given_the_languages_its_commit_adds(
        tmp_path, monkeypatch):
    seen = {}
    _start_failing(tmp_path, monkeypatch, broker_error=RuntimeError("stop"))
    monkeypatch.setattr(sandbox, "new_test_ext_list",
                        lambda project_dir, out_dir, base, tip_list: seen.update(
                            base=base, tip=tip_list) or ["py"])
    monkeypatch.setattr(sandbox, "fill_cache_once",
                        lambda *a: seen.update(fill=a[-1]) or 0)
    monkeypatch.setattr(sandbox, "start_broker",
                        lambda *a, **k: seen.update(broker=k["fill_ext_list"])
                        or (_ for _ in ()).throw(RuntimeError("stop")))
    out = str(tmp_path / "out")
    run_file = sandbox.run_file_layout(out, "a", "repo", "20260101-000000-1-3")
    with pytest.raises(RuntimeError, match="stop"):
        sandbox._start_container("", "img", str(tmp_path / "repo"), out, run_file.out_dir,
                                 "repo", "a", "0" * 40, run_file, "20260101-000000-1-3",
                                 ["edit"], [], tip_list=["1" * 40])
    assert seen == {"base": "0" * 40, "tip": ["1" * 40], "fill": ["py"], "broker": ["py"]}


def test_the_languages_of_a_filling_are_part_of_its_fingerprint_only_when_there_are_any(
        tmp_path):
    repo = _repo_with_pyproject(tmp_path / "repo")
    head = _git(repo, "rev-parse", "HEAD").strip()
    plain = sandbox._dep_fingerprint(str(repo), head, "", [], "image")
    blob_list = sorted(sandbox._dep_blob_dict(str(repo), head).items())
    text = json.dumps([sandbox._CACHE_LAYOUT, "image", blob_list])
    assert plain == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert sandbox._dep_fingerprint(str(repo), head, "", [], "image", []) == plain
    with_py = sandbox._dep_fingerprint(str(repo), head, "", [], "image", ["py"])
    assert with_py not in (plain, sandbox._dep_fingerprint(str(repo), head, "", [],
                                                           "image", ["ts"]))


def test_a_filling_is_told_the_languages_it_runs_the_tests_of_and_marks_them(
        tmp_path, monkeypatch):
    repo = _repo_with_pyproject(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    cache = str(tmp_path / "cache")
    _REAL_WARM(str(repo), str(tmp_path / "out"), once=True, cache=cache,
               fill_ext_list=["py", "ts"])
    assert f"{sandbox.CTLOOM_FILL_EXT_ENV}=py,ts" in seen[0]
    mark = sandbox.fill_mark_path(cache)
    assert len(sandbox._fill_mark_set(mark)) == 1
    _REAL_WARM(str(repo), str(tmp_path / "out"), once=True, cache=cache,
               fill_ext_list=["py", "ts"])
    assert len(seen) == 1
    # The same dependency files with no language besides are filled again
    _REAL_WARM(str(repo), str(tmp_path / "out"), once=True, cache=cache)
    assert len(seen) == 2
    assert not any(sandbox.CTLOOM_FILL_EXT_ENV in arg for arg in seen[1])
    assert len(sandbox._fill_mark_set(mark)) == 2


def test_a_container_given_no_languages_to_fill_is_told_of_none():
    assert sandbox.CTLOOM_FILL_EXT_ENV not in " ".join(_argv())


@pytest.mark.parametrize("value, expect", [
    ("py,ts", ["py", "ts"]), ("ts,,py,ts", ["ts", "py"]), ("py,nosuch", ["py"]),
    ("", [])])
def test_the_languages_a_filling_is_told_of_are_read_back(monkeypatch, value, expect):
    monkeypatch.setenv(sandbox.CTLOOM_FILL_EXT_ENV, value)
    assert sandbox.fill_ext_list() == expect


def _repo_with_other_dep(tmp_path):
    """Make a repository whose branch "other" holds another pyproject.toml and code.

    Returns:
        (the repository, the commit of main, the commit of other).
    """
    repo = _repo_with_pyproject(tmp_path / "repo")
    main = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "--quiet", "-b", "other")
    (repo / "pyproject.toml").write_text("[project]\nname = 'other'\n", encoding="utf-8")
    (repo / "kept.txt").write_text("written on other\n", encoding="utf-8")
    _git(repo, "commit", "--quiet", "-am", "other")
    other = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "--quiet", "main")
    return repo, main, other


def _fill_container_list(monkeypatch):
    """Stand in for the engine and return the list each filling's container is put in,
    as {"argv", "dep_file_dict" (the files mounted at /live, by path)}."""
    seen = []

    def ran(argv):
        live_list = [a.split(":")[0] for a in argv if a.endswith(":/live:ro")]
        dep_file_dict = {}
        for root in live_list:
            for directory, _, name_list in os.walk(root):
                for name in name_list:
                    full = os.path.join(directory, name)
                    dep_file_dict[os.path.relpath(full, root)] = \
                        pathlib.Path(full).read_text(encoding="utf-8")
        seen.append({"argv": argv, "dep_file_dict": dep_file_dict})
        return subprocess.CompletedProcess(argv, 0)

    _engine_stand_in(monkeypatch, ran)
    return seen


def test_a_start_commit_with_other_dependency_files_fills_from_the_working_tree_as_a_refill(
        tmp_path, monkeypatch):
    """The clone is checked out at the commit of the working tree, the start commit's
    dependency files alone are copied over it, and the filling counts as one refill."""
    repo, main, other = _repo_with_other_dep(tmp_path)
    seen = _fill_container_list(monkeypatch)
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 3)
    cache = sandbox.make_cache(str(tmp_path / "cache"))
    count = _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", cache, "/usr/bin/true",
                       "img", None, other)
    assert count == 1
    (fill,) = seen
    assert f"CTLOOM_BASE={main}" in fill["argv"] and "--network=none" not in fill["argv"]
    assert "CTLOOM_LIVE=/live/repo" in fill["argv"]
    assert "CTLOOM_DEP_PATH=pyproject.toml" in fill["argv"]
    assert fill["dep_file_dict"] == {"repo/pyproject.toml": "[project]\nname = 'other'\n"}
    # The same start commit is not filled for again, and is not counted again
    assert _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", cache, "/usr/bin/true",
                      "img", None, other) == 0
    assert len(seen) == 1
    assert _git(repo, "branch", "--list", "codetwine-loom/*") == ""


def test_with_no_refill_left_the_start_commit_s_dependency_files_reach_no_filling(
        tmp_path, monkeypatch, caplog):
    repo, main, other = _repo_with_other_dep(tmp_path)
    seen = _fill_container_list(monkeypatch)
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 0)
    cache = sandbox.make_cache(str(tmp_path / "cache"))
    with caplog.at_level(logging.INFO):
        count = _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", cache,
                           "/usr/bin/true", "img", None, other)
    assert count == 0
    # The working tree's own dependency files are filled for, with nothing of other
    (fill,) = seen
    assert f"CTLOOM_BASE={main}" in fill["argv"]
    assert not [a for a in fill["argv"] if "/live" in a or a.startswith("CTLOOM_DEP_PATH")]
    assert other not in " ".join(fill["argv"])
    assert "WARM_MAX_REFILL is 0" in caplog.text
    # Filled for the working tree already: no container at all the next time
    assert _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", cache, "/usr/bin/true",
                      "img", None, other) == 0
    assert len(seen) == 1


def test_a_cache_filled_in_another_image_is_filled_again_once_and_not_counted(
        tmp_path, monkeypatch):
    """The mark is read for the image the run uses: the same image starts no filling,
    a rebuilt one starts one, which is no refill."""
    repo, main, _ = _repo_with_other_dep(tmp_path)
    image_id = ["sha256:first"]
    seen, warm_list, asked_set = [], [], set()

    def answer(engine, name):
        asked_set.add(engine)
        return f"{image_id[0]}-{engine}-{name}"

    def ran(argv):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    def counted_warm(*args, **kwargs):
        warm_list.append(kwargs)
        return _REAL_WARM(*args, **kwargs)

    _engine_stand_in(monkeypatch, ran, answer)
    monkeypatch.setattr(sandbox, "warm", counted_warm)
    cache = sandbox.make_cache(str(tmp_path / "cache"))

    def fill():
        return _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", cache,
                          "/the/engine", "img", None, main)

    assert fill() == 0 and (len(warm_list), len(seen)) == (1, 1)
    assert fill() == 0 and (len(warm_list), len(seen)) == (1, 1)
    image_id[0] = "sha256:second"
    assert fill() == 0 and (len(warm_list), len(seen)) == (2, 2)
    assert asked_set == {"/the/engine"}


def test_a_start_commit_with_the_working_tree_s_dependency_files_is_no_refill(
        tmp_path, monkeypatch):
    """Code alone differs: the filling runs the working tree's commit and is not
    counted, whatever WARM_MAX_REFILL is."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    main = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "--quiet", "-b", "other")
    (repo / "kept.txt").write_text("written on other\n", encoding="utf-8")
    (repo / "requirements.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "other")
    other = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "--quiet", "main")
    (repo / "requirements.txt").write_text("a\n", encoding="utf-8")   # not committed
    seen = _fill_container_list(monkeypatch)
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 0)
    cache = sandbox.make_cache(str(tmp_path / "cache"))
    assert _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", cache, "/usr/bin/true",
                      "img", None, other) == 0
    (fill,) = seen
    base_arg = next(a for a in fill["argv"] if a.startswith("CTLOOM_BASE="))
    assert base_arg not in (f"CTLOOM_BASE={main}", f"CTLOOM_BASE={other}")
    assert gitcmd.is_snapshot(str(repo), base_arg.split("=")[1])
    assert not [a for a in fill["argv"] if "/live" in a]


def _land_other(tmp_path, repo, main, other):
    """Put what other holds over main into the working tree of repo."""
    sandbox.land_in_tree(str(repo), main, other, str(tmp_path / "land.diff"),
                         str(tmp_path / "last.diff"))
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "written on other\n"


@pytest.mark.parametrize("max_refill, dep_file_dict", [
    (0, {}), (3, {"repo/pyproject.toml": "[project]\nname = 'other'\n"})])
def test_a_filling_after_a_landing_runs_the_commit_the_command_started_at(
        tmp_path, monkeypatch, max_refill, dep_file_dict):
    """The working tree holds what the command put into it, dependency files and
    code; the filling's clone is checked out at the commit made before that."""
    repo, main, other = _repo_with_other_dep(tmp_path)
    seen = _fill_container_list(monkeypatch)
    monkeypatch.setattr(config, "WARM_MAX_REFILL", max_refill)
    cache = sandbox.make_cache(str(tmp_path / "cache"))
    with sandbox.tree_branch(str(repo), "20260101-000000-1-1") as (start, _):
        assert start == main
    _land_other(tmp_path, repo, main, other)
    with sandbox.tree_branch(str(repo), "20260101-000000-2-1") as (later, _):
        assert later != main and sandbox.start_tree(str(repo)) == main
        count = _REAL_FILL(str(repo), str(tmp_path / "out"), "a1", cache,
                           "/usr/bin/true", "img", None, later)
    assert count == (1 if dep_file_dict else 0)
    (fill,) = seen
    assert f"CTLOOM_BASE={main}" in fill["argv"] and "--network=none" not in fill["argv"]
    assert later not in " ".join(fill["argv"])
    assert fill["dep_file_dict"] == dep_file_dict
    assert _git(repo, "branch", "--list", "codetwine-loom/*") == ""


def test_the_commit_the_command_started_at_is_made_before_its_first_landing(tmp_path):
    """No commit of the working tree was made before the landing."""
    repo, main, other = _repo_with_other_dep(tmp_path)
    (repo / "note.txt").write_text("not committed\n", encoding="utf-8")
    _land_other(tmp_path, repo, main, other)
    start = sandbox.start_tree(str(repo))
    assert _git(repo, "ls-tree", "-r", "--name-only", start).split() == [
        "kept.txt", "note.txt", "pyproject.toml"]
    assert _git(repo, "show", f"{start}:kept.txt") == "one\n"
    # It stays that commit, and is made again once it is forgotten
    (repo / "note.txt").write_text("changed\n", encoding="utf-8")
    assert sandbox.start_tree(str(repo)) == start
    sandbox.forget_start_tree()
    assert sandbox.start_tree(str(repo)) != start


def test_a_filling_given_a_commit_no_branch_reaches_points_a_branch_at_it(tmp_path,
                                                                         monkeypatch):
    repo = _repo_with_pyproject(tmp_path / "repo")
    (repo / "note.txt").write_text("not committed\n", encoding="utf-8")
    with sandbox.tree_branch(str(repo), "20260101-000000-1-1") as (start, _):
        pass
    assert _git(repo, "branch", "--contains", start) == ""
    seen = []

    def ran(argv):
        source = next(a.split(":")[0] for a in argv if a.endswith(":/src:ro"))
        seen.append(_git(source, "branch", "--contains", start).split())
        return subprocess.CompletedProcess(argv, 0)

    _engine_stand_in(monkeypatch, ran)
    fill_dict = _REAL_WARM(str(repo), str(tmp_path / "out"), base=start)
    assert fill_dict["base"] == start and fill_dict["exit_status"] == 0
    (branch_list,) = seen
    assert len(branch_list) == 1 and sandbox.is_tree_branch(branch_list[0])
    assert _git(repo, "branch", "--list", "codetwine-loom/*") == ""


def test_the_dependency_files_copied_from_a_commit_are_its_regular_files_alone(tmp_path):
    repo = _repo_with_pyproject(tmp_path / "repo")
    main = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "sub").mkdir()
    (repo / "sub" / "requirements.txt").write_text("b\n", encoding="utf-8")
    os.symlink("sub/requirements.txt", repo / "requirements.txt")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "more")
    head = _git(repo, "rev-parse", "HEAD").strip()
    target = tmp_path / "dep" / "repo"
    assert sandbox._write_dep_file(str(repo), main, head, str(target)) \
        == ["sub/requirements.txt"]
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*")
                  if p.is_file()) == ["sub/requirements.txt"]
    # What both commits hold alike is not written
    assert sandbox._write_dep_file(str(repo), head, head, str(tmp_path / "none")) == []
    assert not (tmp_path / "none").exists()


def test_the_broker_of_a_container_filled_as_a_refill_is_left_one_refill_fewer(
        tmp_path, monkeypatch):
    seen = {}
    order_list = []

    def start(sock_dir, *a, **k):
        seen.update(a=a, k=k)
        raise RuntimeError("stop")

    def fill(*a, **k):
        order_list.append("fill")
        return 1

    _start_failing(tmp_path, monkeypatch, fill=fill, broker_error=RuntimeError("x"))
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 3)
    monkeypatch.setattr(sandbox, "start_broker", start)
    monkeypatch.setattr(sandbox, "index_describe",
                        lambda *a: order_list.append("index") or True)
    order_list.clear()
    out = str(tmp_path / "out")
    run_file = sandbox.run_file_layout(out, "a", "repo", "20260101-000000-1-2")
    with pytest.raises(RuntimeError, match="stop"):
        sandbox._start_container("", "img", str(tmp_path / "repo"), out,
                                 run_file.out_dir, "repo", "a", "0" * 40, run_file,
                                 "20260101-000000-1-2", ["edit"], [])
    assert seen["k"]["max_refill"] == 2
    # The container's start commit is not handed to the broker: a refill checks its
    # clone out at the commit the working tree made when the command started
    assert "0" * 40 not in seen["a"]
    assert seen["k"]["base"] == _git(tmp_path / "repo", "rev-parse", "HEAD").strip()
    # The index is read after the filling
    assert order_list == ["fill", "index"]


def test_the_first_filling_is_made_for_the_agent_and_fills_the_shared_cache(
        tmp_path, monkeypatch):
    """The run-tests inside the filling container reads the index at /out, and makes
    the analysis again there where it does not describe the clone. That has to land
    in the agent's own copy, not in what every agent copies; the cache stays shared."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    seen, fillings = [], []
    _stand_in_for_the_container(monkeypatch, seen)
    monkeypatch.setattr(sandbox, "fill_cache_once", _REAL_FILL)
    monkeypatch.setattr(sandbox, "warm", lambda *a, **k: fillings.append((a, k))
                        or {"exit_status": 0})
    monkeypatch.setattr(sandbox, "new_run_id", lambda: "20260101-000000-1-42")
    sandbox.run(str(repo), str(out), "x")
    (args, kwargs), = fillings
    assert args[1] == str(out) and kwargs["agent"] == "20260101-000000-1-42"
    assert kwargs["cache"] == str(out / "agents" / ".repo" / "repo" / "cache")


def test_warm_mounts_the_cache_it_is_named_and_the_copy_of_the_agent_it_is_named(
        tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    sandbox.warm(str(repo), str(out), cache=str(tmp_path / "shared"), agent="codetwine-loom-one")
    assert f"{tmp_path / 'shared'}:/cache" in seen[0]
    assert f"{out / 'agents' / 'codetwine-loom-one'}:/out" in seen[0]
    assert (tmp_path / "shared" / "uv").is_dir()


def test_a_filling_no_agent_asked_for_is_given_a_copy_of_its_own(tmp_path, monkeypatch):
    """The warm subcommand: what its container writes at /out is not what the agents
    copy, and the copy made for it is gone when the filling ends."""
    out, _ = _built(tmp_path)
    repo = _repo(tmp_path / "repo")
    seen = []

    def ran(argv):
        mine = _mounted_at(argv, "/out")
        seen.append((mine, sorted(p.name for p in (mine / "repo").iterdir())))
        return subprocess.CompletedProcess(argv, 0)

    _engine_stand_in(monkeypatch, ran)
    sandbox.warm(str(repo), str(out))
    (mine, name_list), = seen
    assert mine.parent == out / "agents" and mine != out
    assert name_list == ["code_index.sqlite", "codetwine_py", "project_knowledge.sqlite"]
    assert not mine.exists()
    assert not list(out.glob("agents/.host/*"))


def test_a_filling_that_ran_through_leaves_a_mark_that_once_reads(tmp_path,
                                                                  monkeypatch):
    repo = _repo(tmp_path / "repo")
    fill = sandbox.warm                    # taken before the stand-in replaces it
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    first = fill(str(repo), str(tmp_path / "out"), once=True)
    assert first["exit_status"] == 0 and len(seen) == 1
    # The mark is beside the cache, which every container writes, and not in it
    assert sandbox.fill_mark_path(first["cache"]) == first["cache"] + ".filled"
    assert os.path.isfile(sandbox.fill_mark_path(first["cache"]))
    assert sorted(os.listdir(first["cache"])) == ["gradle", "home", "uv", "xdg"]
    # The second filling is left out: no container is started
    second = fill(str(repo), str(tmp_path / "out"), once=True)
    assert second["exit_status"] == 0 and len(seen) == 1
    # Without once, as the warm subcommand and a refill call it, it fills again
    fill(str(repo), str(tmp_path / "out"))
    assert len(seen) == 2


def test_a_mark_left_by_another_image_does_not_leave_a_filling_out(tmp_path,
                                                                    monkeypatch):
    """An image built again, with another toolchain in it, fills the cache once more."""
    repo = _repo(tmp_path / "repo")
    fill = sandbox.warm
    image_id = ["sha256:first"]
    seen = []

    def ran(argv):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    _engine_stand_in(monkeypatch, ran,
                     lambda engine, name: f"{image_id[0]}-{engine}-{name}")
    fill(str(repo), str(tmp_path / "out"), once=True)
    fill(str(repo), str(tmp_path / "out"), once=True)
    assert len(seen) == 1
    image_id[0] = "sha256:second"
    fill(str(repo), str(tmp_path / "out"), once=True)
    fill(str(repo), str(tmp_path / "out"), once=True)
    assert len(seen) == 2


@pytest.mark.parametrize("answer", [
    subprocess.CompletedProcess([], 125, stdout="", stderr="no such image"),
    subprocess.CompletedProcess([], 0, stdout="\n", stderr=""),
    subprocess.TimeoutExpired(["podman"], 30), OSError("not started")])
def test_an_image_the_engine_gives_no_id_for_is_taken_by_its_name(monkeypatch, caplog,
                                                                  answer):
    asked = []

    def run(argv, **kwargs):
        asked.append(argv[-1])
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(sandbox, "_image_id", _REAL_IMAGE_ID)
    monkeypatch.setattr(sandbox, "pick_engine", lambda name="": "/usr/bin/true")
    monkeypatch.setattr(sandbox.subprocess, "run", run)
    with caplog.at_level(logging.INFO):
        assert sandbox._image_id("", "some-image") == "some-image"
    assert "mark names it by its name" in caplog.text
    monkeypatch.setattr(config, "CONTAINER_IMAGE", "the-default")
    assert sandbox._image_id("", "") == "the-default"
    assert asked == ["some-image", "the-default"]


def test_an_image_is_taken_by_its_name_when_there_is_no_engine(monkeypatch, caplog):
    def no_engine(name=""):
        raise RuntimeError("no container engine")

    monkeypatch.setattr(sandbox, "_image_id", _REAL_IMAGE_ID)
    monkeypatch.setattr(sandbox, "pick_engine", no_engine)
    with caplog.at_level(logging.INFO):
        assert sandbox._image_id("", "some-image") == "some-image"
    assert "mark names it by its name" in caplog.text


def test_a_filling_that_failed_leaves_no_mark_and_is_tried_again(tmp_path,
                                                                 monkeypatch):
    repo = _repo(tmp_path / "repo")
    fill = sandbox.warm
    seen = []

    def failed(argv):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 1)

    _engine_stand_in(monkeypatch, failed)
    out = fill(str(repo), str(tmp_path / "out"), once=True)
    assert out["exit_status"] == 1
    assert not os.path.exists(sandbox.fill_mark_path(out["cache"]))
    assert fill(str(repo), str(tmp_path / "out"), once=True)["exit_status"] == 1
    assert len(seen) == 2


def test_what_a_container_left_in_the_cache_under_the_marks_name_is_not_touched(
        tmp_path, monkeypatch):
    """A link there is neither read nor written through."""
    repo = _repo(tmp_path / "repo")
    fill = sandbox.warm
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    cache = sandbox.make_cache(sandbox.cache_dir(str(tmp_path / "out"), "repo"))
    outside = tmp_path / "outside.txt"
    outside.write_text("mine\n", encoding="utf-8")
    os.symlink(outside, os.path.join(cache, ".filled"))
    assert fill(str(repo), str(tmp_path / "out"), once=True)["exit_status"] == 0
    assert outside.read_text(encoding="utf-8") == "mine\n"
    assert fill(str(repo), str(tmp_path / "out"), once=True)["skip"] is True


def test_the_mark_of_a_cache_put_elsewhere_is_beside_that_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_CACHE_DIR", str(tmp_path / "shared" / "cache") + "/")
    cache = sandbox.cache_dir(str(tmp_path / "out"), "repo")
    assert sandbox.fill_mark_path(cache) == str(tmp_path / "shared" / "cache.filled")
    assert sandbox.fill_mark_path(str(tmp_path / "shared" / "cache") + "/") \
        == str(tmp_path / "shared" / "cache.filled")


@pytest.mark.parametrize("leave", ["link to nothing", "link to a directory", "file"])
def test_a_cache_directory_that_is_not_a_directory_of_its_own_stops_in_one_line(
        tmp_path, leave):
    cache = sandbox.make_cache(str(tmp_path / "cache"))
    home = os.path.join(cache, "home")
    os.rmdir(home)
    if leave == "file":
        pathlib.Path(home).write_text("x", encoding="utf-8")
    else:
        target = tmp_path / "elsewhere"
        if leave == "link to a directory":
            target.mkdir()
        os.symlink(target, home)
    with pytest.raises(RuntimeError, match="is not a directory of its own"):
        sandbox.make_cache(cache)
    assert not (tmp_path / "elsewhere" / "anything").exists()


_REAL_WARM = sandbox.warm
_RUN_ID = "20260101-000000-1-7"
_AGENT = "codetwine-loom-one"


def _live_copy(tmp_path, repo, pyproject):
    """Make the running change's copy of repo, holding pyproject, and return it.

    It is a clone, as the change's copy is, made once and written over after that.
    """
    live = pathlib.Path(sandbox.work_tree(str(tmp_path / "out"), _AGENT, repo.name,
                                          _RUN_ID))
    if not live.exists():
        live.parent.mkdir(parents=True)
        _git(live.parent, "clone", "--quiet", str(repo), str(live))
    (live / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return live


def _refill(tmp_path, repo, spent):
    """Ask for a refill as the broker does, recording each time it spends one."""
    return _REAL_WARM(str(repo), str(tmp_path / "out"), live_dep_file=True,
                      run_id=_RUN_ID, once=True, before_fill=lambda: spent.append(1),
                      agent=_AGENT)


def _repo_with_pyproject(path):
    """Make a repository whose one commit holds a pyproject.toml, and return it."""
    repo = _repo(path)
    (repo / "pyproject.toml").write_text("[project]\nname = 'p'\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "pyproject")
    return repo


def test_a_refill_starts_a_filling_only_for_dependency_files_not_yet_filled(
        tmp_path, monkeypatch):
    repo = _repo_with_pyproject(tmp_path / "repo")
    seen, spent = [], []
    _stand_in_for_the_container(monkeypatch, seen)
    _REAL_WARM(str(repo), str(tmp_path / "out"), once=True)
    assert len(seen) == 1
    # The change's copy holds what the commit holds: nothing to fetch, nothing spent
    _live_copy(tmp_path, repo, "[project]\nname = 'p'\n")
    assert _refill(tmp_path, repo, spent)["skip"] is True
    assert len(seen) == 1 and spent == []
    # A dependency added: filled and counted once, then left alone
    _live_copy(tmp_path, repo, "[project]\nname = 'p'\ndependencies = ['x']\n")
    assert _refill(tmp_path, repo, spent)["skip"] is False
    assert _refill(tmp_path, repo, spent)["skip"] is True
    assert len(seen) == 2 and spent == [1]
    # Put back as it was: the cache already holds that
    _live_copy(tmp_path, repo, "[project]\nname = 'p'\n")
    assert _refill(tmp_path, repo, spent)["skip"] is True
    assert len(seen) == 2 and spent == [1]


def test_a_refill_is_given_the_directory_this_machine_made_and_told_where_the_clone_is(
        tmp_path, monkeypatch):
    """The clone is the container's to make, so what is mounted is the directory this
    machine made around it."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    live = _live_copy(tmp_path, repo, "[project]\nname = 'p'\ndependencies = ['x']\n")
    assert _refill(tmp_path, repo, [])["skip"] is False
    assert f"{live.parent}:/live:ro" in seen[0] and "CTLOOM_LIVE=/live/repo" in seen[0]
    assert "CTLOOM_DEP_PATH=pyproject.toml" in seen[0]
    assert live.parent.name == "work" and live.parent.parent.name == f"scratch-{_RUN_ID}"


def test_a_link_in_place_of_the_running_change_s_clone_is_neither_read_nor_mounted(
        tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO", logger="codetwine_loom.sandbox")
    repo = _repo_with_pyproject(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "pyproject.toml").write_text("[project]\ndependencies = ['other']\n",
                                            encoding="utf-8")
    live = pathlib.Path(sandbox.work_tree(str(tmp_path / "out"), _AGENT, repo.name,
                                          _RUN_ID))
    live.parent.mkdir(parents=True)
    live.symlink_to(outside)
    _refill(tmp_path, repo, [])
    assert "no change is running" in caplog.text
    assert not any("/live" in arg or "CTLOOM_DEP_PATH" in arg for arg in seen[0])


def test_a_refill_that_failed_is_counted_and_tried_again(tmp_path, monkeypatch):
    repo = _repo_with_pyproject(tmp_path / "repo")
    spent = []
    real_run = subprocess.run

    def failing(argv, **kwargs):
        if _is_git(argv):
            return real_run(argv, **kwargs)
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(sandbox, "pick_engine", lambda name="": "/usr/bin/true")
    monkeypatch.setattr(sandbox.subprocess, "run", failing)
    _live_copy(tmp_path, repo, "[project]\ndependencies = ['x']\n")
    assert _refill(tmp_path, repo, spent)["exit_status"] == 1
    assert _refill(tmp_path, repo, spent)["exit_status"] == 1
    assert spent == [1, 1]


def test_a_refill_with_no_budget_left_starts_no_container(tmp_path, monkeypatch):
    repo = _repo_with_pyproject(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    _live_copy(tmp_path, repo, "[project]\ndependencies = ['x']\n")

    def spent_out():
        raise RuntimeError("WARM_MAX_REFILL")

    with pytest.raises(RuntimeError, match="WARM_MAX_REFILL"):
        _REAL_WARM(str(repo), str(tmp_path / "out"), live_dep_file=True,
                   run_id=_RUN_ID, once=True, before_fill=spent_out)
    assert seen == []


def test_a_commit_that_adds_a_dependency_is_filled_by_the_next_run(tmp_path,
                                                                   monkeypatch):
    repo = _repo_with_pyproject(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    _REAL_WARM(str(repo), str(tmp_path / "out"), once=True)
    # A commit that leaves the dependency files alone fetches nothing new
    (repo / "kept.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "commit", "--quiet", "-am", "code")
    _REAL_WARM(str(repo), str(tmp_path / "out"), once=True)
    assert len(seen) == 1
    (repo / "pyproject.toml").write_text("[project]\ndependencies = ['x']\n",
                                         encoding="utf-8")
    _git(repo, "commit", "--quiet", "-am", "dependency")
    _REAL_WARM(str(repo), str(tmp_path / "out"), once=True)
    assert len(seen) == 2


@pytest.mark.parametrize("odd_name", ["conftest.py\nx.csproj", "a\rb.csproj",
                                      "sub\ndir/requirements.txt", "tab\there.csproj"])
def test_a_dependency_file_with_a_control_character_in_its_name_reaches_no_filling(
        tmp_path, caplog, odd_name):
    """The list is handed to the filling's container one path to a line: a name that
    holds a line break would be read there as other paths than the one it names."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    base = _git(repo, "rev-parse", "HEAD").strip()
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    for root in (live, repo):
        (root / odd_name).parent.mkdir(exist_ok=True)
        (root / odd_name).write_text("<Project/>\n", encoding="utf-8")
        (root / "conftest.py").write_text("# not a dependency file\n", encoding="utf-8")
        (root / "x.csproj").write_text("<Project/>\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        path_list = sandbox.live_dep_path_list_of(str(repo), str(live))
    assert path_list == ["pyproject.toml", "x.csproj"]
    assert "of the running change holds a control character in its name" in caplog.text
    argv = sandbox.build_run_argv(
        "engine", "img", sandbox.MountLayout(src="/s", out="/o", scratch="/sc",
                                             cache="/c", live=str(tmp_path)),
        base, "live", ["run-tests"], network=True, live_dep_path_list=path_list)
    assert "CTLOOM_DEP_PATH=pyproject.toml\nx.csproj" in argv
    # A commit holding the file: it is none of the commit's dependency files either
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "an odd name")
    odd = _git(repo, "rev-parse", "HEAD").strip()
    assert sorted(sandbox._dep_blob_dict(str(repo), odd)) == ["pyproject.toml", "x.csproj"]
    assert sandbox._write_dep_file(str(repo), base, odd, str(tmp_path / "dep")) == [
        "x.csproj"]
    assert sorted(os.listdir(tmp_path / "dep")) == ["x.csproj"]


def test_the_dependency_files_taken_are_the_ones_git_would_carry(tmp_path):
    """What a tool leaves where git ignores it (uv run's uv.lock, node_modules) is not
    the change's, and must not tell a cache that already holds the change apart."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    (repo / ".gitignore").write_text("uv.lock\nnode_modules/\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "ignore")
    base = _git(repo, "rev-parse", "HEAD").strip()
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))

    def fingerprint():
        return sandbox._dep_fingerprint(
            str(repo), base, str(live),
            sandbox.live_dep_path_list_of(str(repo), str(live)), "image")

    alone = fingerprint()
    assert alone == sandbox._dep_fingerprint(str(repo), base, "", [], "image")
    (live / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (live / "node_modules" / "x").mkdir(parents=True)
    (live / "node_modules" / "x" / "package.json").write_text("{}", encoding="utf-8")
    (live / "elsewhere.txt").write_text("[project]\n", encoding="utf-8")
    (live / "requirements.txt").symlink_to(live / "elsewhere.txt")
    assert fingerprint() == alone
    assert sandbox.live_dep_path_list_of(str(repo), str(live)) == ["pyproject.toml"]
    # A new file git would carry is the change's
    (live / "sub dir").mkdir()
    (live / "sub dir" / "requirements.txt").write_text("x\n", encoding="utf-8")
    assert fingerprint() != alone
    assert sandbox.live_dep_path_list_of(str(repo), str(live)) == [
        "pyproject.toml", "sub dir/requirements.txt"]


def test_a_dependency_file_named_by_a_pattern_is_taken(tmp_path):
    repo = _repo_with_pyproject(tmp_path / "repo")
    base = _git(repo, "rev-parse", "HEAD").strip()
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    alone = sandbox._dep_fingerprint(str(repo), base, "", [], "image")
    (live / "src" / "App").mkdir(parents=True)
    (live / "src" / "App" / "App.csproj").write_text("<Project />\n", encoding="utf-8")
    (live / "src" / "App" / "App.cs").write_text("class App { }\n", encoding="utf-8")
    path_list = sandbox.live_dep_path_list_of(str(repo), str(live))
    assert path_list == ["pyproject.toml", "src/App/App.csproj"]
    assert sandbox._dep_fingerprint(str(repo), base, str(live), path_list,
                                    "image") != alone


def test_a_dependency_file_a_tool_wrote_is_not_taken(tmp_path):
    """uv run writes uv.lock where the repository does not track one."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    (live / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    assert sandbox.live_dep_path_list_of(str(repo), str(live)) == [
        "pyproject.toml", "uv.lock"]
    assert sandbox.live_dep_path_list_of(str(repo), str(live), ("uv.lock",)) == [
        "pyproject.toml"]
    # A tracked one is the change's whatever the patterns say
    assert sandbox.live_dep_path_list_of(str(repo), str(live), ("pyproject.toml",)) == [
        "pyproject.toml", "uv.lock"]


def test_what_another_language_s_tools_wrote_in_the_copy_is_not_taken(tmp_path):
    """Maven copies the test resources under target/, and uv writes uv.lock beside
    pyproject.toml. The copy's own git leaves both out; this listing does too, where
    the projects are, when it is told the primary language."""
    repo = _repo(tmp_path / "repo")
    for name, text in (("svc/pom.xml", "<project/>\n"), ("tools/pyproject.toml", "\n"),
                       ("web/package.json", "{}\n")):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m", "x")
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    for name in ("svc/target/test-classes/fixture/pom.xml", "tools/uv.lock",
                 "web/target/package.json"):
        (live / name).parent.mkdir(parents=True, exist_ok=True)
        (live / name).write_text("x\n", encoding="utf-8")
    kept = ["svc/pom.xml", "tools/pyproject.toml", "web/package.json"]
    assert sandbox.live_dep_path_list_of(str(repo), str(live)) == sorted(
        kept + ["svc/target/test-classes/fixture/pom.xml", "tools/uv.lock",
                "web/target/package.json"])
    # A new dependency file in a directory of that name elsewhere is the change's
    assert sandbox.live_dep_path_list_of(str(repo), str(live), (), "js") == sorted(
        kept + ["web/target/package.json"])
    # The primary language's own patterns are the caller's to pass
    assert "tools/uv.lock" in sandbox.live_dep_path_list_of(str(repo), str(live), (), "py")


def test_a_dependency_file_reached_through_a_linked_directory_is_not_taken(tmp_path):
    """A tracked directory the container put a link in the place of leads out of the
    copy."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    (repo / "sub").mkdir()
    (repo / "sub" / "requirements.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "sub")
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    assert sandbox.live_dep_path_list_of(str(repo), str(live)) == [
        "pyproject.toml", "sub/requirements.txt"]
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "requirements.txt").write_text("not the copy's\n", encoding="utf-8")
    shutil.rmtree(live / "sub")
    (live / "sub").symlink_to(outside)
    assert sandbox.live_dep_path_list_of(str(repo), str(live)) == ["pyproject.toml"]


def test_a_dependency_file_that_became_a_pipe_stops_the_filling_in_one_line(tmp_path):
    repo = _repo_with_pyproject(tmp_path / "repo")
    base = _git(repo, "rev-parse", "HEAD").strip()
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    os.remove(live / "pyproject.toml")
    os.mkfifo(live / "pyproject.toml")
    with pytest.raises(RuntimeError, match="pyproject.toml of the running change .* "
                                           "is not a regular file"):
        sandbox._dep_fingerprint(str(repo), base, str(live), ["pyproject.toml"],
                                 "image")


def test_dependency_files_are_read_up_to_one_read_between_them(tmp_path, monkeypatch):
    """Files nothing but their size is written in are not read on and on: what all of
    them hold together is bounded by what one file may hold."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    base = _git(repo, "rev-parse", "HEAD").strip()
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    path_list = ["pyproject.toml"]
    for n in range(3):
        (live / f"d{n}").mkdir()
        with open(live / f"d{n}" / "requirements.txt", "wb") as handle:
            handle.truncate(40)
        path_list.append(f"d{n}/requirements.txt")
    size = (live / "pyproject.toml").stat().st_size
    monkeypatch.setattr(sandbox.tmpfile, "READ_MAX_BYTE", size + 80)
    assert sandbox._dep_fingerprint(str(repo), base, str(live), path_list[:3], "image")
    with pytest.raises(RuntimeError, match=f"hold more than {size + 80} bytes between "
                                           f"them"):
        sandbox._dep_fingerprint(str(repo), base, str(live), path_list, "image")


def test_the_copy_s_own_git_directory_is_not_what_lists_its_files(tmp_path):
    """The container writes it, and git runs the command a repository's configuration
    names for watching its files."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    mark = tmp_path / "ran"
    hook = tmp_path / "hook.sh"
    hook.write_text(f"#!/bin/sh\ntouch {mark}\n", encoding="utf-8")
    hook.chmod(0o755)
    _git(live, "config", "core.fsmonitor", str(hook))
    (live / "requirements.txt").write_text("x\n", encoding="utf-8")
    assert sandbox.live_dep_path_list_of(str(repo), str(live)) == [
        "pyproject.toml", "requirements.txt"]
    assert not mark.exists()


def test_a_copy_git_cannot_read_stops_the_filling_rather_than_passing_it(tmp_path,
                                                                         monkeypatch):
    """Reading no files there would take the change for one that added nothing and
    leave the cache without what it added."""
    repo = _repo_with_pyproject(tmp_path / "repo")
    seen = []
    _stand_in_for_the_container(monkeypatch, seen)
    live = pathlib.Path(sandbox.work_tree(str(tmp_path / "out"), _AGENT, repo.name,
                                          _RUN_ID))
    live.mkdir(parents=True)
    (live / "pyproject.toml").write_text("[project]\ndependencies = ['x']\n",
                                         encoding="utf-8")
    def run_entry(project_dir, *args, **kwargs):
        assert "ls-files" in args and "--others" in args
        return subprocess.CompletedProcess(args, 128, [], "fatal: not readable")

    monkeypatch.setattr(sandbox.gitcmd, "run_entry", run_entry)
    with pytest.raises(RuntimeError, match="could not list the files"):
        _refill(tmp_path, repo, [])
    assert seen == []
    assert not list((tmp_path / "out").glob("agents/.host/*/*/sandbox/scratch-warm-*"))


def test_a_path_git_would_quote_is_matched_all_the_same(tmp_path):
    repo = _repo(tmp_path / "repo")
    odd = repo / 'a"b'
    odd.mkdir()
    (odd / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "odd")
    base = _git(repo, "rev-parse", "HEAD").strip()
    live = tmp_path / "live"
    _git(tmp_path, "clone", "--quiet", str(repo), str(live))
    assert sandbox._dep_fingerprint(
        str(repo), base, str(live),
        sandbox.live_dep_path_list_of(str(repo), str(live)), "image") == \
        sandbox._dep_fingerprint(str(repo), base, "", [], "image")


# == what a merging run says about how it ended =============
def test_the_outcome_is_named_once_for_both_sides_of_the_boundary():
    """The container writes the file this machine reads, as the split is."""
    files = sandbox.run_file_layout("/out", "a1", "repo", "20260101-000000-1-42")
    assert files.outcome.endswith("outcome-20260101-000000-1-42.json")
    assert files.outcome_inside == f"{sandbox._OUT}/repo/sandbox/" \
                                   f"{os.path.basename(files.outcome)}"
    assert os.path.dirname(files.outcome) == os.path.dirname(files.patch)


def test_the_container_is_told_where_to_say_how_the_merge_ended():
    argv = sandbox.integrate_argv("repo", ["a", "b"],
                                  outcome_out="/out/repo/sandbox/outcome-1.json")
    assert "--outcome-out" in argv and "/out/repo/sandbox/outcome-1.json" in argv
    # Without a path nothing is asked for
    assert "--outcome-out" not in sandbox.integrate_argv("repo", ["a", "b"])


def test_the_container_is_told_the_branch_to_carry_on_from_apart_from_the_others():
    argv = sandbox.integrate_argv("repo", ["a"], resume="codetwine-loom/sandbox/earlier")
    assert argv[argv.index("--resume") + 1] == "codetwine-loom/sandbox/earlier"
    assert argv[argv.index("--branch") + 1] == "a"
    assert "--resume" not in sandbox.integrate_argv("repo", ["a", "b"])


def test_how_the_merge_ended_and_what_it_recorded_reach_the_commit_as_trailers(tmp_path):
    path = tmp_path / "outcome.json"
    path.write_text('{"verdict": "unsettled", "why": "- markers in a.py", "did": "x",'
                    ' "ended": "turns exhausted",'
                    ' "decisions": [{"what": "a.py", "why": "kept a"}]}', encoding="utf-8")
    outcome = sandbox._read_outcome(str(path), str(path.parent), "merge")
    message = sandbox.commit_message("combine a, b",
                                     sandbox.run_trailer_dict({"exit_status": 1}, outcome),
                                     prefix="integrate")
    assert "Codetwine-Loom-Ended: turns exhausted" in message
    assert 'Codetwine-Loom-Decisions: [{"what":"a.py","why":"kept a"}]' in message
    assert "Codetwine-Loom-Separator" not in message


def test_the_separator_lines_an_unsettled_union_may_hold_reach_the_commit_as_a_trailer(
        tmp_path):
    path = tmp_path / "outcome.json"
    separator_dict = {"my doc.rst": {"=======": 1, "======= x  y": 2}, "f.py": {}}
    path.write_text(json.dumps({"verdict": "unsettled", "separator": separator_dict}),
                    encoding="utf-8")
    outcome = sandbox._read_outcome(str(path), str(path.parent), "merge")
    assert outcome["separator"] == separator_dict
    message = sandbox.commit_message("combine a, b", sandbox.run_trailer_dict({}, outcome),
                                     prefix="integrate")
    line = message.splitlines()[-1]
    assert line.startswith("Codetwine-Loom-Separator: ") and line.count(" ") == 1
    assert json.loads(line.partition(": ")[2]) == separator_dict
    path.write_text('{"verdict": "unsettled", "separator": {}}', encoding="utf-8")
    assert "Codetwine-Loom-Separator: {}" in sandbox.commit_message(
        "r", sandbox.run_trailer_dict({}, sandbox._read_outcome(str(path), str(path.parent), "merge")))
    # A record of another shape is read as no outcome at all
    path.write_text('{"verdict": "unsettled", "separator": {"f.py": {"=======": "1"}}}',
                    encoding="utf-8")
    assert sandbox._read_outcome(str(path), str(path.parent), "merge") == {}


def test_the_sides_an_unsettled_union_was_not_merged_for_reach_the_commit_as_a_trailer(
        tmp_path):
    path = tmp_path / "outcome.json"
    no_merge_list = [["a" * 40, "old name.txt", "my doc.rst"], ["b" * 40, None, "f.py"],
                     ["c" * 40, "", "f.py"]]
    path.write_text(json.dumps({"verdict": "unsettled", "no_merge": no_merge_list}),
                    encoding="utf-8")
    outcome = sandbox._read_outcome(str(path), str(path.parent), "merge")
    assert outcome["no_merge"] == no_merge_list
    message = sandbox.commit_message("combine a, b", sandbox.run_trailer_dict({}, outcome),
                                     prefix="integrate")
    line = message.splitlines()[-1]
    assert line.startswith("Codetwine-Loom-No-Merge: ") and line.count(" ") == 1
    assert json.loads(line.partition(": ")[2]) == no_merge_list
    path.write_text('{"verdict": "unsettled", "no_merge": []}', encoding="utf-8")
    assert "Codetwine-Loom-No-Merge: []" in sandbox.commit_message("r", sandbox.run_trailer_dict(
        {}, sandbox._read_outcome(str(path), str(path.parent), "merge")))
    # A record of another shape is read as no outcome at all, and no other kind of
    # run records one
    path.write_text('{"verdict": "unsettled", "no_merge": [["main", "f.py", "f.py"]]}',
                    encoding="utf-8")
    assert sandbox._read_outcome(str(path), str(path.parent), "merge") == {}
    path.write_text(json.dumps({"verdict": "unsettled", "no_merge": no_merge_list}),
                    encoding="utf-8")
    assert "no_merge" not in sandbox._read_outcome(str(path), str(path.parent), "refactor")
    # What a container ending in the model leaves: the sides, and no verdict
    path.write_text(json.dumps({"no_merge": no_merge_list}), encoding="utf-8")
    outcome = sandbox._read_outcome(str(path), str(path.parent), "merge")
    assert outcome == {"no_merge": no_merge_list}
    meta_dict = {"exit_status": 137}
    message = sandbox.commit_message("combine a, b", sandbox.run_trailer_dict(meta_dict, {
        **outcome, **sandbox._merge_record_dict({"stage": "x"}, outcome, 137, meta_dict)}),
        prefix="integrate")
    assert message.splitlines()[-2:] == [
        "Codetwine-Loom-Exit-Status: 137",
        f"Codetwine-Loom-No-Merge: {integrate_mod.no_merge_text(no_merge_list)}"]


def test_the_request_the_whole_run_was_given_reaches_the_commit_as_a_trailer():
    """The message is the task this agent was handed, so the request the tasks were
    split out of is only here."""
    message = sandbox.commit_message("add --version to main.py", sandbox.run_trailer_dict(
        {"exit_status": 1}, {"n": 1, "verdict": "failed", "why": "the turns ran out"},
        "add --version and --verbose to main.py"))
    assert "Codetwine-Loom-Run-Request: add --version and --verbose to main.py" in message


def test_a_run_given_no_request_of_its_own_carries_no_such_trailer():
    message = sandbox.commit_message("combine a, b",
                                     sandbox.run_trailer_dict({"exit_status": 0}, {}),
                                     prefix="integrate")
    assert "Codetwine-Loom-Run-Request" not in message


def test_a_change_run_carries_neither_of_those_trailers():
    message = sandbox.commit_message("add the flag", sandbox.run_trailer_dict(
        {"exit_status": 0}, {"n": 1, "verdict": "passed", "why": "", "did": "added it"}))
    assert "Codetwine-Loom-Ended" not in message and "Codetwine-Loom-Decisions" not in message


def test_what_the_merge_settled_reaches_the_commit_as_trailers(tmp_path):
    """A branch holding an unsettled merge says nothing about why without them."""
    path = tmp_path / "outcome.json"
    path.write_text('{"verdict": "unsettled", "why": "- markers in a.py",'
                    ' "did": "kept both sides"}', encoding="utf-8")
    outcome = sandbox._read_outcome(str(path), str(path.parent), "merge")
    message = sandbox.commit_message("combine a, b",
                                     sandbox.run_trailer_dict({"exit_status": 1}, outcome),
                                     prefix="integrate")
    assert "Codetwine-Loom-Verdict: unsettled" in message
    assert "Codetwine-Loom-Why: - markers in a.py" in message
    assert "Codetwine-Loom-Did: kept both sides" in message


def test_a_merge_that_left_no_account_carries_no_such_trailer(tmp_path):
    outcome = sandbox._read_outcome(str(tmp_path / "not-there.json"), str(tmp_path),
                                    "merge")
    assert outcome == {}
    message = sandbox.commit_message("combine a, b",
                                     sandbox.run_trailer_dict({"exit_status": 0}, outcome))
    assert "Codetwine-Loom-Verdict" not in message and "Codetwine-Loom-Did" not in message


# == the names of branches the union already holds ==========
def test_a_branch_the_union_reaches_is_taken_away(tmp_path):
    """Its commits are kept by the union, so the name is a second name for them."""
    repo = _repo(tmp_path / "repo")
    base = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "branch", "task1", base)
    (repo / "b.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "later")
    union = _git(repo, "rev-parse", "HEAD").strip()
    assert sandbox.drop_branch_under(str(repo), ["task1"], union) == ["task1"]
    assert sandbox.branch_tip(str(repo), "task1") == ""


def test_a_branch_the_union_does_not_reach_is_left_alone(tmp_path):
    """A try that did not finish is not combined, and its branch is the only account
    of what it got to."""
    repo = _repo(tmp_path / "repo")
    union = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "--quiet", "-b", "task1")
    (repo / "half.txt").write_text("part\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "half a change")
    _git(repo, "checkout", "--quiet", "main")
    assert sandbox.drop_branch_under(str(repo), ["task1"], union) == []
    assert sandbox.branch_tip(str(repo), "task1")


def test_a_branch_that_is_not_there_is_not_an_error(tmp_path):
    repo = _repo(tmp_path / "repo")
    union = _git(repo, "rev-parse", "HEAD").strip()
    assert sandbox.drop_branch_under(str(repo), ["never-made"], union) == []


def test_the_task_branches_go_once_the_merge_landed(_dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one", "two"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 0, "verdict": "unverified"})
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: {"exit_status": 0, "commit": "u" * 40,
                                     "branch": "codetwine-loom/combined"})
    asked = []
    monkeypatch.setattr(sandbox, "drop_branch_under",
                        lambda project_dir, branches, commit:
                        asked.append((branches, commit)) or list(branches))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert asked == [(["codetwine-loom/run/20260101-000000-1-42/1",
                       "codetwine-loom/run/20260101-000000-1-42/2"], "u" * 40)]
    assert result["dropped"] == asked[0][0]


def test_nothing_is_taken_away_when_no_merge_landed(_dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    tasks = sandbox.run_file_layout(
        _make_agent(out), "20260101-000000-1-42", "repo", "20260101-000000-1-42").task
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(tasks, {"tasks": ["one"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 0, "verdict": "unverified"})
    monkeypatch.setattr(sandbox, "drop_branch_under",
                        _never("a branch was taken away with no merge behind it"))
    assert sandbox.dispatch(str(repo), str(out), "名前を変えて")["dropped"] == []





def test_the_commit_says_which_tests_the_run_changed():
    """A patch does not say which of the files in it are the tests."""
    message = sandbox.commit_message("戻り値を変えて", sandbox.run_trailer_dict(
        {"exit_status": 0},
        {"n": 1, "verdict": "unverified", "why": "w", "did": "d",
         "tests_changed": ["tests/test_a.py", "tests/test_b.py"]}))
    assert "Codetwine-Loom-Tests-Changed: tests/test_a.py tests/test_b.py" in message


def test_a_run_that_left_the_tests_alone_carries_no_such_trailer():
    message = sandbox.commit_message("r", sandbox.run_trailer_dict(
        {"exit_status": 0}, {"n": 1, "verdict": "unverified", "why": "w", "did": "d"}))
    assert "Codetwine-Loom-Tests-Changed" not in message


# == What the engine and git refuse =========================
def _refuse_git(monkeypatch, *word_tuple):
    """Make gitcmd.run fail when the command holds every word given."""
    real_run = sandbox.gitcmd.run

    def run(project_dir, *args, **kwargs):
        if all(word in args for word in word_tuple):
            return subprocess.CompletedProcess(list(args), 1, "", "refused by git")
        return real_run(project_dir, *args, **kwargs)

    monkeypatch.setattr(sandbox.gitcmd, "run", run)


def test_a_container_that_cannot_be_asked_to_go_is_said_in_the_log(monkeypatch, caplog):
    def blow_up(argv, **kwargs):
        raise OSError("engine gone")

    monkeypatch.setattr(sandbox.subprocess, "run", blow_up)
    sandbox._remove_container("/usr/bin/podman", "ctloom-x")
    assert "could not be taken away" in caplog.text


def test_a_container_that_takes_too_long_to_go_is_said_in_the_log(monkeypatch, caplog):
    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 1)

    monkeypatch.setattr(sandbox.subprocess, "run", slow)
    sandbox._remove_container("/usr/bin/podman", "ctloom-x")
    assert "could not be taken away" in caplog.text


def test_a_container_the_engine_would_not_remove_is_said_in_the_log(monkeypatch, caplog):
    monkeypatch.setattr(sandbox.subprocess, "run",
                        lambda argv, **kwargs: subprocess.CompletedProcess(
                            argv, 1, "", "no such container"))
    sandbox._remove_container("/usr/bin/podman", "ctloom-x")
    assert "ctloom-x is still there: no such container" in caplog.text


def test_a_branch_git_will_not_delete_is_left_where_it_is(tmp_path, monkeypatch, caplog):
    import logging
    repo = _repo(tmp_path / "repo")
    base = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "branch", "task1", base)
    _refuse_git(monkeypatch, "branch", "-D")
    with caplog.at_level(logging.INFO):
        assert sandbox.drop_branch_under(str(repo), ["task1"], base) == []
    assert "task1 was left where it is" in caplog.text
    assert sandbox.branch_tip(str(repo), "task1") == base


def test_a_try_branch_git_will_not_delete_is_left_where_it_is(tmp_path, monkeypatch,
                                                              caplog):
    import logging
    repo = _repo(tmp_path / "repo")
    _git(repo, "branch", "codetwine-loom/run/20260101-000000-1-11/1")
    _refuse_git(monkeypatch, "branch", "-D")
    with caplog.at_level(logging.INFO):
        assert sandbox.drop_try_branch_list(str(repo), "20260101-000000-1-11", []) == []
    assert "was left where it is" in caplog.text


def test_an_old_runs_branch_git_will_not_delete_is_said_and_the_rest_still_goes(
        _runs_left_behind, monkeypatch, caplog):
    import logging
    repo, out, runs = _runs_left_behind
    _refuse_git(monkeypatch, "branch", "-D")
    with caplog.at_level(logging.INFO):
        assert sandbox.remove_old_run(str(repo), str(out), 2) == [runs[0]]
    assert f"codetwine-loom/run/{runs[0]}/1 was left where it is" in caplog.text
    assert not os.path.exists(sandbox.agent_dir(str(out), f"run-{runs[0]}-1"))


def test_a_worktree_that_cannot_be_made_is_an_error(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    base = _git(repo, "rev-parse", "HEAD").strip()
    patch = tmp_path / "p.diff"
    patch.write_text("", encoding="utf-8")
    _refuse_git(monkeypatch, "worktree", "add")
    with pytest.raises(RuntimeError,
                       match="could not make a worktree for codetwine-loom/sandbox/one"):
        sandbox.take_in_patch(str(repo), str(patch), base, "codetwine-loom/sandbox/one",
                              str(tmp_path / "wt"), "sandbox: x")


def test_tasks_that_landed_nothing_leave_nothing_to_combine(_dispatch_repo, monkeypatch,
                                                            caplog):
    import logging
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "", "exit_status": 0, "verdict": "no_change"})
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: pytest.fail("a merge was started"))
    with caplog.at_level(logging.INFO):
        result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert "nothing to combine" in caplog.text
    assert result["integration"] is None and result["resume"] == []


def test_a_task_whose_container_left_no_patch_taken_in_is_named_with_why_and_no_branch(
        _dispatch_repo, monkeypatch, caplog):
    """The entry carries why nothing of it was taken in, and the line saying nothing is
    combined names a branch only for a task whose commit landed on it."""
    repo, out = _dispatch_repo
    why = "the patch does not hold what the run wrote into 1 submodule directory(ies) " \
          "of its clone that git shows as their commit alone: sub. It stays at /o/p.diff"
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 1, "verdict": "failed", "no_patch_why": ""}
        if a[2] == "one" else {"commit": None, "exit_status": 1, "verdict": "unverified",
                               "no_patch_why": why})
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: pytest.fail("a merge was started"))
    with caplog.at_level(logging.ERROR, logger="codetwine_loom.sandbox"):
        result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert [(e["n"], e["error"]) for e in result["branches"]] == [(1, ""), (2, why)]
    message, = [r.getMessage() for r in caplog.records if "did not finish" in r.getMessage()]
    assert message == (
        f"2 of 2 task(s) did not finish, so nothing is combined and the repository is "
        f"left as it is: what task 1's agent got to is on "
        f"codetwine-loom/run/20260101-000000-1-42/1-retry1; nothing of task 2 was taken in ({why})")


def test_a_red_test_of_a_task_with_nothing_taken_in_is_named_with_why_and_no_branch(
        _dispatch_repo, monkeypatch, caplog):
    """The line saying nothing read the red tests names a branch only for a task whose
    commit landed on it."""
    repo, out = _dispatch_repo
    why = "the patch the run wrote holds 1272 bytes, more than the 1000 that are " \
          "taken in. It stays at /o/p.diff"
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": "c" * 40, "exit_status": 0, "verdict": "tests_red", "no_patch_why": ""}
        if a[2] == "one" else {"commit": None, "exit_status": 1, "verdict": "tests_red",
                               "no_patch_why": why})
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: pytest.fail("a merge was started"))
    with caplog.at_level(logging.ERROR, logger="codetwine_loom.sandbox"):
        result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["exit_status"] == 1
    message, = [r.getMessage() for r in caplog.records
                if "nothing read those tests" in r.getMessage()]
    assert message == (
        f"2 task(s) ended with a test failing and nothing combined them afterwards, so "
        f"nothing read those tests: what task 1's agent got to is on "
        f"codetwine-loom/run/20260101-000000-1-42/1; nothing of task 2 was taken in ({why})")


def test_a_red_test_of_tasks_that_changed_nothing_says_so_and_no_exit_status(
        _dispatch_repo, monkeypatch, caplog):
    """A task whose patch held nothing and whose run ended with 0 is named as having
    changed nothing, not by an exit status of 0."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: {
        "commit": None, "exit_status": 0, "verdict": "tests_red", "no_patch_why": ""})
    monkeypatch.setattr(sandbox, "integrate",
                        lambda *a, **k: pytest.fail("a merge was started"))
    with caplog.at_level(logging.ERROR, logger="codetwine_loom.sandbox"):
        result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["exit_status"] == 1
    message, = [r.getMessage() for r in caplog.records
                if "nothing read those tests" in r.getMessage()]
    assert message == (
        "2 task(s) ended with a test failing and nothing combined them afterwards, so "
        "nothing read those tests: nothing of task 1 was taken in (its run changed "
        "nothing); nothing of task 2 was taken in (its run changed nothing)")


@pytest.mark.parametrize("entry_dict, text", [
    ({"n": 1, "branch": "b/1", "commit": "c" * 40, "exit_status": 0, "error": ""},
     "what task 1's agent got to is on b/1"),
    ({"n": 2, "branch": "b/2", "commit": "", "exit_status": 1, "error": "why"},
     "nothing of task 2 was taken in (why)"),
    ({"n": 2, "branch": "b/2", "commit": "", "exit_status": 0, "error": ""},
     "nothing of task 2 was taken in (its run changed nothing)"),
    ({"n": 2, "branch": "b/2", "commit": "", "exit_status": 1, "error": ""},
     "nothing of task 2 was taken in (its run ended with exit status 1 and changed "
     "nothing)"),
])
def test_where_a_task_got_to_names_no_exit_status_of_0_as_why(entry_dict, text):
    assert sandbox._task_place_text(entry_dict) == text


@pytest.mark.parametrize("status", [124, 137, 1, 0])
def test_a_reading_container_that_left_nothing_to_read_stops_the_run_with_its_status(
        _dispatch_repo, monkeypatch, status):
    """The reading run and an answering run alike: the error says how the container
    ended, with what a time limit or a memory limit is called, and carries its status."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SEC", 600)
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: ("/usr/bin/podman", "img", status))
    head = _git(repo, "rev-parse", "HEAD").strip()
    for read, what in ((lambda: sandbox.dispatch(str(repo), str(out), "r"), "split"),
                       (lambda: sandbox.answer(str(repo), str(out), "q", start=head),
                        "answer")):
        with pytest.raises(RuntimeError, match=f"left no {what} in ") as raised:
            read()
        if not status:
            assert not isinstance(raised.value, sandbox.ContainerEndError)
            continue
        assert raised.value.exit_status == status
        assert str(raised.value).startswith(
            f"the container ended ({sandbox.status_text(status)}), and the run that ")
    assert sandbox.status_text(124) == \
        "exit status 124, the status of a container taken away at SANDBOX_TIMEOUT_SEC (600 s)"
    assert sandbox.status_text(137).startswith(
        "exit status 137, the status of a container killed when its processes reach "
        "SANDBOX_MEMORY (")
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SEC", 0)
    assert sandbox.status_text(124) == "exit status 124"


# == A merge the run could not settle ========================
def _merge_run(_dispatch_repo, monkeypatch):
    """Run integrate() with a container that ends unsettled and return the log."""
    repo, out = _dispatch_repo
    _git(repo, "branch", "a")

    def start(engine, image, project_dir, root_out_dir, out_dir, name, agent, base,
              run_file, run_id, argv_tail, exclude_list, **_):
        os.makedirs(os.path.dirname(run_file.outcome), exist_ok=True)
        with open(run_file.outcome, "w", encoding="utf-8") as fh:
            json.dump({"verdict": "unsettled", "why": "markers are still in f.txt",
                       "did": "", "settled": False}, fh)
        return "/usr/bin/podman", "img", 1

    monkeypatch.setattr(sandbox, "_start_container", start)
    monkeypatch.setattr(sandbox, "_land_patch", lambda *a, **k: {
        "patch": "p", "files": 1, "exit_status": 1, "branch": "codetwine-loom/sandbox/u",
        "commit": "c" * 40})
    _git(repo, "branch", "b")
    return sandbox.integrate_into_tree(str(repo), str(out), ["a", "b"])


def test_a_merge_left_unsettled_is_not_put_into_the_tree_and_the_log_says_why(
        _dispatch_repo, monkeypatch, caplog):
    result = _merge_run(_dispatch_repo, monkeypatch)
    assert result["land"] is None
    assert "nothing was put into the working tree" in result["land_why"]
    assert "markers are still in f.txt" in result["land_why"]
    assert result["land_why"] in caplog.text


def test_a_merge_carried_on_from_starts_from_what_the_earlier_tree_became(
        _dispatch_repo, monkeypatch):
    """The earlier merge was written against a commit of the working tree, and the
    working tree has changed since."""
    repo, out = _dispatch_repo
    head = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "notes.txt").write_text("first\n", encoding="utf-8")
    first = gitcmd.snapshot_commit(str(repo))
    identity = ("-c", "user.email=t@t", "-c", "user.name=t")
    union = _git(repo, *identity, "commit-tree", f"{first}^{{tree}}", "-p", first, "-p",
                 head, "-m",
                 "integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: unsettled").strip()
    _git(repo, "branch", "union", union)
    plain = _git(repo, *identity, "commit-tree", f"{head}^{{tree}}", "-p", head, "-p",
                 head, "-m",
                 "integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: unsettled").strip()
    _git(repo, "branch", "plain", plain)
    (repo / "notes.txt").write_text("second\n", encoding="utf-8")
    seen = {}

    def integrate(project, output_dir, branch_list, **kwargs):
        seen["parents"] = _git(repo, "rev-list", "--parents", "-n", "1",
                               kwargs["onto"]).split()[1:]
        return {"commit": "", "land": None}

    monkeypatch.setattr(sandbox, "integrate", integrate)
    sandbox.integrate_into_tree(str(repo), str(out), [], resume="union")
    assert seen["parents"] == [head, first]
    # A merge written against a commit that is no commit of the working tree
    sandbox.integrate_into_tree(str(repo), str(out), [], resume="plain")
    assert seen["parents"] == [head]


# == Starting and stopping a real broker process ============
def _stub_broker(tmp_path, body: str) -> str:
    """Write a shell script that stands in for the interpreter the broker is started
    with, and return its path."""
    path = tmp_path / "stub-python"
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


# Reads the --endpoint argument, makes the socket file and waits to be stopped
_LISTEN_BODY = """while [ "$1" != "--endpoint" ]; do shift; done
: > "${2#unix://}"
exec sleep 30
"""


@pytest.fixture
def real_broker_start(monkeypatch):
    """Put the real start_broker and stop_broker back for one test."""
    monkeypatch.setattr(sandbox, "start_broker", _START_BROKER)
    monkeypatch.setattr(sandbox, "stop_broker", _STOP_BROKER)
    monkeypatch.setattr(config, "BROKER_START_TIMEOUT_SEC", 2)


def test_the_broker_is_waited_for_until_it_listens_and_stopped_afterwards(
        tmp_path, monkeypatch, real_broker_start):
    monkeypatch.setattr(sandbox.sys, "executable", _stub_broker(tmp_path, _LISTEN_BODY))
    sock_dir = tempfile.mkdtemp(prefix="ctloom-test-")
    process = sandbox.start_broker(sock_dir, str(tmp_path), str(tmp_path / "out"),
                                   "20260101-000000-1", str(tmp_path / "cache"))
    assert process.poll() is None
    sandbox.stop_broker(process, sock_dir)
    assert process.poll() is not None and not os.path.exists(sock_dir)


def test_the_broker_is_told_how_many_refills_are_left_and_the_commit_a_refill_starts_at(
        tmp_path, monkeypatch, real_broker_start):
    sock_dir = tempfile.mkdtemp(prefix="ctloom-test-")
    pathlib.Path(sock_dir, "llm.sock").touch()
    seen = {}

    class Going:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(sandbox.subprocess, "Popen",
                        lambda argv, **k: seen.update(argv=argv, env=k["env"]) or Going())
    monkeypatch.delenv("WARM_MAX_REFILL", raising=False)
    sandbox.start_broker(sock_dir, str(tmp_path), str(tmp_path / "out"), "r",
                         max_refill=2, base="c" * 40, fill_ext_list=["py", "ts"])
    assert seen["argv"][seen["argv"].index("--base") + 1] == "c" * 40
    assert seen["argv"][seen["argv"].index("--fill-ext") + 1] == "py,ts"
    assert seen["env"]["WARM_MAX_REFILL"] == "2"
    sandbox.start_broker(sock_dir, str(tmp_path), str(tmp_path / "out"), "r")
    assert "--base" not in seen["argv"] and "--fill-ext" not in seen["argv"]
    sandbox.start_broker(sock_dir)
    assert "WARM_MAX_REFILL" not in seen["env"] and "--base" not in seen["argv"]
    shutil.rmtree(sock_dir)


def test_a_broker_that_ends_before_it_listens_is_an_error_naming_its_status(
        tmp_path, monkeypatch, real_broker_start):
    monkeypatch.setattr(sandbox.sys, "executable", _stub_broker(tmp_path, "exit 3\n"))
    sock_dir = tempfile.mkdtemp(prefix="ctloom-test-")
    with pytest.raises(RuntimeError, match=r"ended before it listened .*exit status 3"):
        sandbox.start_broker(sock_dir)
    shutil.rmtree(sock_dir, ignore_errors=True)


def test_a_broker_that_never_listens_is_stopped_and_an_error(tmp_path, monkeypatch,
                                                             real_broker_start):
    monkeypatch.setattr(config, "BROKER_START_TIMEOUT_SEC", 0.3)
    monkeypatch.setattr(sandbox.sys, "executable", _stub_broker(tmp_path, "exec sleep 30\n"))
    sock_dir = tempfile.mkdtemp(prefix="ctloom-test-")
    with pytest.raises(RuntimeError, match="did not listen"):
        sandbox.start_broker(sock_dir)
    assert not os.path.exists(sock_dir)


def test_a_broker_whose_start_is_stopped_is_ended_before_the_stop_goes_on(
        tmp_path, monkeypatch, real_broker_start):
    monkeypatch.setattr(sandbox.sys, "executable", _stub_broker(tmp_path, "exec sleep 30\n"))
    process_list = []
    real_popen, real_sleep = subprocess.Popen, time.sleep

    def popen(*a, **k):
        process_list.append(real_popen(*a, **k))
        return process_list[-1]

    def stop_once(sec):
        if len(process_list) == 1 and not getattr(stop_once, "sent", False):
            stop_once.sent = True
            raise KeyboardInterrupt("the run was stopped (SIGTERM)")
        real_sleep(sec)

    monkeypatch.setattr(sandbox.subprocess, "Popen", popen)
    monkeypatch.setattr(sandbox.time, "sleep", stop_once)
    sock_dir = tempfile.mkdtemp(prefix="ctloom-test-")
    try:
        with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
            sandbox.start_broker(sock_dir)
        assert process_list[0].poll() is not None and not os.path.exists(sock_dir)
    finally:
        for process in process_list:
            if process.poll() is None:
                process.kill()
                process.wait()
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_a_broker_that_will_not_stop_when_asked_is_killed(tmp_path, monkeypatch,
                                                          real_broker_start, caplog):
    monkeypatch.setattr(config, "BROKER_START_TIMEOUT_SEC", 0.3)
    monkeypatch.setattr(sandbox.sys, "executable",
                        _stub_broker(tmp_path, "trap '' TERM\n" + _LISTEN_BODY.replace(
                            "exec sleep 30", "sleep 3")))
    sock_dir = tempfile.mkdtemp(prefix="ctloom-test-")
    process = sandbox.start_broker(sock_dir)
    sandbox.stop_broker(process, sock_dir)
    assert "did not end when asked" in caplog.text
    assert process.poll() is not None and not os.path.exists(sock_dir)


# == What cannot be taken away is said ======================
def test_a_directory_that_will_not_go_is_said_in_the_log(tmp_path, monkeypatch, caplog):
    def refuse(path):
        raise PermissionError("busy")

    monkeypatch.setattr(sandbox.shutil, "rmtree", refuse)
    assert sandbox._remove_tree(str(tmp_path), "the clone") is False
    assert "the clone at" in caplog.text and "Remove it by hand" in caplog.text


def test_a_file_that_will_not_go_is_said_in_the_log(tmp_path, monkeypatch, caplog):
    def refuse(path):
        raise PermissionError("busy")

    (tmp_path / "x.lock").write_text("", encoding="utf-8")
    monkeypatch.setattr(sandbox.os, "remove", refuse)
    assert sandbox._remove_file(str(tmp_path / "x.lock"), "the lock") is False
    assert "the lock at" in caplog.text


def test_a_directory_or_file_already_gone_is_not_reported(tmp_path, caplog):
    assert sandbox._remove_tree(str(tmp_path / "gone"), "the clone") is True
    assert sandbox._remove_file(str(tmp_path / "gone.lock"), "the lock") is True
    assert caplog.text == ""


def test_what_was_taken_away_is_said_by_the_value_handed_back(tmp_path):
    (tmp_path / "clone" / "inner").mkdir(parents=True)
    (tmp_path / "x.lock").write_text("", encoding="utf-8")
    os.symlink(tmp_path / "nowhere", tmp_path / "link")
    assert sandbox._remove_tree(str(tmp_path / "clone"), "the clone") is True
    assert sandbox._remove_file(str(tmp_path / "x.lock"), "the lock") is True
    assert sandbox._remove_entry(str(tmp_path / "link"), "the link") is True
    assert list(tmp_path.iterdir()) == []


def test_no_patch_means_no_summary_to_read():
    assert sandbox._read_meta("", "") == {}


def test_a_summary_that_cannot_be_written_is_said_in_the_log(tmp_path, caplog):
    sandbox._write_land_meta(str(tmp_path / "nowhere" / "p.diff"), str(tmp_path),
                             {"branch": "b", "commit": "c" * 40})
    assert "could not write" in caplog.text


def _merge_on_new_branch(repo, branch, start, message):
    """Make a branch that points at a commit of two parents over start - start and a
    commit of its own beside it - that changes nothing. Return the commit."""
    identity = ("-c", "user.email=t@t", "-c", "user.name=t")
    tree = _git(repo, "rev-parse", f"{start}^{{tree}}").strip()
    side = _git(repo, *identity, "commit-tree", tree, "-p", start, "-m", "one side").strip()
    commit = _git(repo, *identity, "commit-tree", tree, "-p", start, "-p", side, "-m",
                  message).strip()
    _git(repo, "branch", branch, commit)
    return commit


def _union_left(repo, run_id, verdict):
    """Put a union branch for one run on a merge commit whose trailer records the
    verdict."""
    _merge_on_new_branch(repo, f"codetwine-loom/sandbox/{run_id}", "main",
                         f"integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: {verdict}\n")


def test_a_union_recorded_as_unsettled_is_taken_away_with_its_run(_runs_left_behind):
    """It is what the run left to carry on from, like a task branch, so it takes one of
    the places and goes with the run when the run is beyond them."""
    repo, out, runs = _runs_left_behind
    _union_left(repo, runs[0], "unsettled")
    assert sandbox.remove_old_run(str(repo), str(out), 2) == [runs[0]]
    assert f"codetwine-loom/sandbox/{runs[0]}" not in _branches(repo)
    assert f"codetwine-loom/run/{runs[0]}/1" not in _branches(repo)
    assert not os.path.exists(sandbox.agent_dir(str(out), runs[0]))


def test_a_union_a_later_command_carried_on_from_takes_up_that_command_s_place(
        _runs_left_behind, monkeypatch):
    """The stages carried on from a union write under the agent named after the run
    that made it, each file named after the command that wrote it."""
    repo, out, runs = _runs_left_behind
    first = "20260100-000000-1-5"
    _union_left(repo, first, "unsettled")
    agent = sandbox.agent_name(f"codetwine-loom/sandbox/{first}", first)
    os.makedirs(os.path.join(sandbox.agent_dir(str(out), agent), "repo", "sandbox"))
    # Made by the oldest command, it is the first to go
    assert sandbox.remove_old_run(str(repo), str(out), 3) == [first]
    _union_left(repo, first, "unsettled")
    later = "20260104-000000-3-44"
    for directory in (sandbox.agent_dir(str(out), agent), sandbox.host_dir(str(out), agent)):
        os.makedirs(os.path.join(directory, "repo", "sandbox"), exist_ok=True)
    # The run the later command started there is what this machine recorded; what
    # the agent's copy holds is not read
    sandbox._write_run_mark(sandbox.run_file_layout(str(out), agent, "repo", later), "")
    pathlib.Path(sandbox.agent_dir(str(out), agent), "repo", "sandbox",
                 "outcome-20260109-000000-3-1.json").write_text("{}", encoding="utf-8")
    assert sandbox.remove_old_run(str(repo), str(out), 3) == [runs[0]]
    assert f"codetwine-loom/sandbox/{first}" in _branches(repo)
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [runs[2], runs[1]]
    assert f"codetwine-loom/sandbox/{first}" in _branches(repo)
    # A command carrying it on that is still going keeps it whatever its place
    newer = "20260105-000000-1-55"
    _git(repo, "branch", f"codetwine-loom/run/{newer}/1")
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": run_id != later)
    assert sandbox.remove_old_run(str(repo), str(out), 1) == []
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [first]


def test_a_run_a_command_goes_on_from_is_left_alone_from_the_start_of_that_command(
        _runs_left_behind, monkeypatch):
    """The command records the runs of the branches it was told to go on from before
    it reads them, so another command's sweep leaves them while it goes on, and they
    take up that command's place once it has ended."""
    repo, out, runs = _runs_left_behind
    first = "20260100-000000-1-5"
    _union_left(repo, first, "unsettled")
    _git(repo, "branch", "mine/union")
    later, other = "20260104-000000-1-77", "20260104-000001-1-88"
    sandbox._mark_run_of_branch(str(repo), str(out), "repo",
                                [f"codetwine-loom/sandbox/{first}",
                                 f"codetwine-loom/run/{runs[0]}/1", "mine/union"], later)
    union_agent = sandbox.agent_name(f"codetwine-loom/sandbox/{first}", "")
    assert os.path.isfile(sandbox.run_file_layout(str(out), union_agent, "repo", later).mark)
    assert os.path.isfile(sandbox.run_file_layout(
        str(out), sandbox.agent_name(f"codetwine-loom/run/{runs[0]}/1", ""), "repo", later).mark)
    assert not os.path.exists(sandbox.host_dir(str(out), "mine-union"))
    # A newer command takes the one place, and the command going on keeps both runs
    newer = "20260105-000000-1-99"
    _git(repo, "branch", f"codetwine-loom/run/{newer}/1")
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": run_id != later)
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [runs[2], runs[1]]
    assert f"codetwine-loom/sandbox/{first}" in _branches(repo)
    assert f"codetwine-loom/run/{runs[0]}/1" in _branches(repo)
    # A second command going on from the union that ended later does not hide the
    # first one, which is still going
    sandbox._mark_run_of_branch(str(repo), str(out), "repo", [f"codetwine-loom/sandbox/{first}"],
                                other)
    assert sandbox.remove_old_run(str(repo), str(out), 1) == []
    # Once both have ended, the runs are counted by the command that went on from them
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    assert sandbox.remove_old_run(str(repo), str(out), 3) == []
    assert sandbox.remove_old_run(str(repo), str(out), 2) == [runs[0]]
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [first]
    assert "mine/union" in _branches(repo)


def test_the_sweep_and_the_record_of_a_run_gone_on_from_take_the_branch_lock(
        _runs_left_behind):
    """While another holds the lock, the sweep takes nothing away and a command going
    on from a run waits to record it; once the lock is given up, the record is there
    and the sweep reads it."""
    repo, out, runs = _runs_left_behind
    later = "20260104-000000-1-77"
    held, release = threading.Event(), threading.Event()

    def hold():
        with sandbox._branch_lock(str(repo)):
            held.set()
            release.wait(30)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(30)
    mark = sandbox.run_file_layout(str(out),
                                   sandbox.agent_name(f"codetwine-loom/run/{runs[0]}/1", ""),
                                   "repo", later).mark
    marker = threading.Thread(target=sandbox._mark_run_of_branch,
                              args=(str(repo), str(out), "repo",
                                    [f"codetwine-loom/run/{runs[0]}/1"], later))
    try:
        assert sandbox.remove_old_run(str(repo), str(out), 1) == []
        assert f"codetwine-loom/run/{runs[0]}/1" in _branches(repo)
        marker.start()
        marker.join(0.5)
        assert marker.is_alive() and not os.path.exists(mark)
    finally:
        release.set()
        holder.join(30)
        marker.join(30)
    assert os.path.isfile(mark)
    # The record counts runs[0] as the newest command's, so the others go
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [runs[2], runs[1]]


@pytest.mark.parametrize("kind", ["run", "integrate"])
def test_a_command_going_on_from_branches_records_their_runs_before_it_reads_them(
        _dispatch_repo, monkeypatch, kind):
    repo, out = _dispatch_repo
    union = "codetwine-loom/sandbox/20260100-000000-1-5"
    _union_left(repo, "20260100-000000-1-5", "unsettled")

    def stop(*_args, **_kwargs):
        raise RuntimeError("read")

    monkeypatch.setattr(sandbox, "gap_union_list_of", stop)
    monkeypatch.setattr(sandbox, "stage_resume_dict", stop)
    monkeypatch.setattr(sandbox, "resume_head", stop)
    with pytest.raises(RuntimeError, match="read"):
        if kind == "run":
            sandbox.dispatch(str(repo), str(out), "x", resume_list=[union])
        else:
            sandbox.integrate_into_tree(str(repo), str(out),
                                        ["codetwine-loom/run/20260100-000000-2-5/1"], resume=union)
    mark_list = [union] + (["codetwine-loom/run/20260100-000000-2-5/1"] if kind == "integrate"
                           else [])
    for branch in mark_list:
        assert os.path.isfile(sandbox.run_file_layout(
            str(out), sandbox.agent_name(branch, ""), "repo", "20260101-000000-1-42").mark)


def test_a_union_recorded_as_unsettled_within_the_newest_few_stays(_runs_left_behind):
    repo, out, runs = _runs_left_behind
    _union_left(repo, runs[2], "unsettled")
    assert sandbox.remove_old_run(str(repo), str(out), 2) == [runs[0]]
    assert f"codetwine-loom/sandbox/{runs[2]}" in _branches(repo)


def test_a_union_recorded_as_settled_is_never_taken_away(_runs_left_behind):
    repo, out, runs = _runs_left_behind
    _union_left(repo, runs[0], "settled")
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [runs[1]]
    assert f"codetwine-loom/sandbox/{runs[0]}" in _branches(repo)
    assert os.path.exists(sandbox.agent_dir(str(out), runs[0]))


def test_a_command_that_was_stopped_is_told_the_branches_its_runs_left(_dispatch_repo):
    repo, out = _dispatch_repo
    own, other = f"20260102-000000-2-{os.getpid()}", "20260102-000000-2-1"
    before = f"20260101-000000-1-{os.getpid()}"
    for run_id in (own, other, before):
        _git(repo, "branch", f"codetwine-loom/run/{run_id}/1", "main")
        _git(repo, "branch", f"codetwine-loom/run/{run_id}/tree", "main")
    # A task branch the union does not reach
    _git(repo, "checkout", "--quiet", "-b", f"codetwine-loom/run/{own}/2", "main")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet",
         "--allow-empty", "-m", "half a change")
    _git(repo, "checkout", "--quiet", "main")
    _union_left(repo, own, "unsettled")
    _union_left(repo, f"20260102-000001-3-{os.getpid()}", "settled")
    # Only this process's runs, from the time the command started, and neither the
    # branch of the tree a run started from, a branch holding a result, nor a task
    # branch the union reaches
    assert sandbox.remain_branch_list(str(repo), str(out), f"20260102-000000-1-{os.getpid()}") \
        == [f"codetwine-loom/run/{own}/2", f"codetwine-loom/sandbox/{own}"]
    # A directory that is not a repository holds none
    assert sandbox.remain_branch_list(str(out), str(out), own) == []


def _stage_commit(repo, branch, message):
    """Put one commit that changes nothing on a branch, with a stage's trailers."""
    _git(repo, "checkout", "--quiet", branch)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet",
         "--allow-empty", "-m", message)
    _git(repo, "checkout", "--quiet", "main")


def test_a_settled_union_a_stage_is_ahead_of_is_one_to_carry_on_from(_dispatch_repo,
                                                                    monkeypatch):
    """A run stopped while a stage was going leaves the settled union on its branch,
    and nothing of it in the working tree."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "REFACTOR", True)
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    own = f"20260102-000000-2-{os.getpid()}"
    union, test_branch = f"codetwine-loom/sandbox/{own}", f"codetwine-loom/sandbox/{own}-test/1"
    _git(repo, "branch", f"codetwine-loom/run/{own}/1", "main")
    _union_left(repo, own, "settled")
    _git(repo, "branch", test_branch, union)
    _stage_commit(repo, test_branch, "test-create: x\n\nx\n\nCodetwine-Loom-Stage: test-create\n"
                                     "Codetwine-Loom-Verdict: unsettled\n")
    assert sandbox.stage_resume_dict(str(repo), union)["stage"] == "refactor"
    assert sandbox._settle_branch_list(str(repo), [union, test_branch]) == []
    # The union alone is named: it stands for the task branch it reaches and for the
    # test tasks' branches under it
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [union]
    # It takes one of the places, and goes with its run once it is beyond them
    newer = "20260103-000000-1-77"
    _git(repo, "branch", f"codetwine-loom/run/{newer}/1", "main")
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [own]
    assert union not in _branches(repo) and test_branch not in _branches(repo)


def _commit_on_new_branch(repo, branch, start, message):
    """Make a branch at start and put one commit that changes nothing on it."""
    _git(repo, "checkout", "--quiet", "-b", branch, start)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet",
         "--allow-empty", "-m", message)
    _git(repo, "checkout", "--quiet", "main")


@pytest.mark.parametrize("trailer, is_open", [
    ("Codetwine-Loom-Exit-Status: 137\n", True), ("Codetwine-Loom-Exit-Status: 1\n", True),
    ("Codetwine-Loom-Verdict: settled\nCodetwine-Loom-Exit-Status: 1\n", True),
    ("Codetwine-Loom-Exit-Status: 0\n", False), ("Codetwine-Loom-Run-Request: x\n", False),
    ("Codetwine-Loom-Exit-Status: -1\n", False), ("Codetwine-Loom-Exit-Status: many\n", False)])
def test_a_union_whose_run_ended_with_another_status_than_0_is_one_to_carry_on_from(
        _dispatch_repo, monkeypatch, trailer, is_open):
    """A command that was stopped names the union a command that returned names: the
    one its run did not get through, whatever verdict the commit carries. The runs
    taken away for their age leave it where it is."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    own = f"20260102-000000-2-{os.getpid()}"
    union = f"codetwine-loom/sandbox/{own}"
    _commit_on_new_branch(repo, union, "main", f"integrate: x\n\ncombine a, b\n\n{trailer}")
    assert sandbox.remain_branch_list(str(repo), str(out), own) == ([union] if is_open
                                                                    else [])
    assert sandbox._settle_branch_list(str(repo), [union]) == [union]
    _git(repo, "branch", "codetwine-loom/run/20260103-000000-1-77/1", "main")
    assert sandbox.remove_old_run(str(repo), str(out), 1) == []
    assert union in _branches(repo)


@pytest.mark.parametrize("trailer", [
    "Codetwine-Loom-Verdict: unsettled\nCodetwine-Loom-Exit-Status: 1\n",
    "Codetwine-Loom-Exit-Status: 137\n"])
def test_the_branch_the_test_tasks_were_combined_on_is_named_by_the_union(
        _dispatch_repo, monkeypatch, trailer):
    """The branch the run combining the test tasks left is named under the union, so a
    command that was stopped names the union alone."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    own = f"20260102-000000-2-{os.getpid()}"
    union = f"codetwine-loom/sandbox/{own}"
    combine = sandbox._TEST_UNION_BRANCH.format(branch=union)
    assert combine == f"{union}-test/union" and sandbox.run_in(combine) == own
    _union_left(repo, own, "settled")
    _commit_on_new_branch(repo, combine, union, f"integrate: x\n\ncombine t\n\n{trailer}")
    assert sandbox.stage_resume_dict(str(repo), union)["stage"] == "test-create"
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [union]
    # A union made again from an unsettled one is written against the commit that one
    # was, and both are named
    again = f"codetwine-loom/sandbox/20260102-000002-2-{os.getpid()}"
    _commit_on_new_branch(repo, again, "main",
                          "integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: unsettled\n")
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [union, again]


def test_a_union_a_later_union_of_the_command_was_made_over_is_not_named(
        _dispatch_repo, monkeypatch):
    """The merge made again over an unsettled union stands for it, as it does for a
    command that returned."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "REFACTOR", True)
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    own = f"20260102-000000-2-{os.getpid()}"
    union, again = (f"codetwine-loom/sandbox/{own}",
                    f"codetwine-loom/sandbox/20260102-000001-3-{os.getpid()}")
    _git(repo, "branch", f"codetwine-loom/run/{own}/1", "main")
    _union_left(repo, own, "unsettled")
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [union]
    _merge_on_new_branch(repo, again, union,
                         "integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: settled\n")
    assert sandbox.stage_resume_dict(str(repo), again)["stage"] == "refactor"
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [again]


def test_a_command_stopped_in_a_run_made_again_names_what_that_run_left_or_was_told(
        _dispatch_repo, monkeypatch):
    """A run of the request made again stands for the runs before it once it has
    left a branch, after the unions holding cases a run before it found not done."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    pid = os.getpid()
    first, gap, later = (f"20260102-000000-{n}-{pid}" for n in (2, 3, 6))

    def union_left(run_id):
        _commit_on_new_branch(repo, f"codetwine-loom/sandbox/{run_id}", "main",
                              f"integrate: {run_id}\n\nc\n\nCodetwine-Loom-Verdict: unsettled\n")

    for run_id in (first, gap):
        union_left(run_id)
    monkeypatch.setattr(sandbox, "_RERUN_LIST", [])
    assert sandbox.remain_branch_list(str(repo), str(out), first) == [
        f"codetwine-loom/sandbox/{first}", f"codetwine-loom/sandbox/{gap}"]
    # Started and told the first union, with nothing of its own yet
    sandbox._RERUN_LIST.append((5, [f"codetwine-loom/sandbox/{first}"],
                                [f"codetwine-loom/sandbox/{gap}"], 0))
    assert sandbox.remain_branch_list(str(repo), str(out), first) == [
        f"codetwine-loom/sandbox/{first}"]
    union_left(later)
    assert sandbox.remain_branch_list(str(repo), str(out), first) == [
        f"codetwine-loom/sandbox/{gap}", f"codetwine-loom/sandbox/{later}"]


@pytest.mark.parametrize("form, is_land, verdict, is_alone", [
    ("nothing left", True, "", True), ("a task branch left", True, "", True),
    ("a settled union left", False, "settled", True),
    ("an unsettled union left", False, "unsettled", False)])
def test_a_run_made_again_that_put_its_union_in_or_left_one_to_go_in_names_its_own_alone(
        _dispatch_repo, monkeypatch, form, is_land, verdict, is_alone):
    """Once the last run of the request has put its union into the working tree, or
    has left a settled union the working tree does not hold, it stands for what it was
    told and for the unions holding cases a run before it found not done, as it does
    for a command that returned."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    pid = os.getpid()
    first, later = (f"20260102-000000-{n}-{pid}" for n in (2, 6))
    told = f"codetwine-loom/sandbox/{first}"
    _commit_on_new_branch(repo, told, "main",
                          f"integrate: {first}\n\nc\n\nCodetwine-Loom-Verdict: unsettled\n")
    # The run made again started with one diff put in before it
    monkeypatch.setattr(sandbox, "_RERUN_LIST", [(5, [told], [told], 1)])
    monkeypatch.setattr(sandbox, "_LAND_LIST", [{"files": 1}] * (2 if is_land else 1))
    own_list = []
    if form == "a task branch left":
        own_list = [f"codetwine-loom/run/{later}/1"]
        _commit_on_new_branch(repo, own_list[0], "main", "edit: other")
    elif verdict:
        # A merge that adds a file the working tree does not hold
        own_list = [f"codetwine-loom/sandbox/{later}"]
        identity = ("-c", "user.email=t@t", "-c", "user.name=t")
        _commit_on_new_branch(repo, "side", "main", "edit: side")
        _git(repo, "checkout", "--quiet", "side")
        (repo / "union.txt").write_text("union\n", encoding="utf-8")
        _git(repo, "add", "union.txt")
        _git(repo, *identity, "commit", "--quiet", "--amend", "-m", "edit: side")
        _git(repo, "checkout", "--quiet", "main")
        merge = _git(repo, *identity, "commit-tree", "side^{tree}", "-p", "main", "-p",
                     "side", "-m",
                     f"integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: {verdict}\n").strip()
        _git(repo, "branch", own_list[0], merge)
    assert sandbox.last_run_has_union(str(repo), own_list) is is_alone
    assert sandbox.remain_branch_list(str(repo), str(out), first) == (
        own_list if is_alone else [told, *own_list])


def test_a_union_written_against_a_commit_of_another_union_is_named_for_itself(
        _dispatch_repo, monkeypatch):
    """A request started from the commit a union landed leaves its own union on that
    commit. A command stopped after it names that union, not the one under it."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    earlier, own = "20260102-000000-2-1", f"20260102-000001-2-{os.getpid()}"
    under, union = f"codetwine-loom/sandbox/{earlier}", f"codetwine-loom/sandbox/{own}"
    _union_left(repo, earlier, "settled")
    _commit_on_new_branch(repo, union, under,
                          "integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: unsettled\n")
    assert sandbox.stage_resume_dict(str(repo), under)["stage"] == "test-create"
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [union]


def test_what_carrying_on_with_a_union_s_stages_left_is_none_of_the_command_s_own(
        _dispatch_repo, monkeypatch):
    """A command carrying on with the stages of a union another command left writes
    the test tasks' branches and their combined tests under that union's name, which
    carries the run that made the union."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    earlier, own = "20260102-000000-2-1", f"20260102-000001-2-{os.getpid()}"
    union = f"codetwine-loom/sandbox/{earlier}"
    _union_left(repo, earlier, "settled")
    for branch in (sandbox._TEST_BRANCH.format(branch=union, n=1),
                   sandbox._TEST_UNION_BRANCH.format(branch=union)):
        _commit_on_new_branch(repo, branch, union,
                              "integrate: x\n\ncombine t\n\nCodetwine-Loom-Verdict: unsettled\n")
    assert sandbox.remain_branch_list(str(repo), str(out), own) == []
    # A task branch of the command's own runs is named
    _commit_on_new_branch(repo, f"codetwine-loom/run/{own}/2", "main", "edit: other")
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [f"codetwine-loom/run/{own}/2"]


def test_a_settled_union_whose_tests_found_a_case_not_done_is_one_to_carry_on_from(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    own = f"20260102-000000-2-{os.getpid()}"
    union = f"codetwine-loom/sandbox/{own}"
    _union_left(repo, own, "settled")
    _stage_commit(repo, union, "test-create: x\n\nx\n\nCodetwine-Loom-Stage: test-create\n"
                               "Codetwine-Loom-Verdict: settled\n"
                               'Codetwine-Loom-Gap: [{"what":"case-1","why":"not handled"}]\n')
    assert sandbox._settle_branch_list(str(repo), [union]) == []
    assert sandbox.remain_branch_list(str(repo), str(out), own) == [union]
    assert sandbox.remove_old_run(str(repo), str(out), 1) == []
    _git(repo, "branch", "codetwine-loom/run/20260103-000000-1-77/1", "main")
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [own]


_GAP_TEXT = ("test-create: x\n\nx\n\nCodetwine-Loom-Run-Request: add a\n"
             "Codetwine-Loom-Stage: test-create\nCodetwine-Loom-Verdict: settled\n"
             'Codetwine-Loom-Gap: [{"what":"case-1","why":"not handled"}]\n')


def test_a_union_carried_on_from_for_its_cases_goes_when_the_run_put_its_own_in(
        _dispatch_repo, monkeypatch):
    """Only one this tool named, that holds a result but for those cases, whose change
    the working tree holds, made for the same request."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "live_run_list", lambda *a: [])
    held, plain, open_one = "20260102-000000-1-1", "20260102-000000-2-1", "20260102-000000-3-1"
    for run_id in (held, plain, open_one):
        _union_left(repo, run_id, "settled" if run_id != open_one else "unsettled")
    _stage_commit(repo, f"codetwine-loom/sandbox/{held}", _GAP_TEXT)
    _git(repo, "branch", "mine", f"codetwine-loom/sandbox/{held}")
    name_list = [f"codetwine-loom/sandbox/{held}", f"codetwine-loom/sandbox/{plain}",
                 f"codetwine-loom/sandbox/{open_one}",
                 "mine", "codetwine-loom/sandbox/20260102-000000-9-1"]
    # Another request does not carry on from the union's cases
    assert sandbox.gap_union_list_of(str(repo), name_list, "add b") == []
    union_list = sandbox.gap_union_list_of(str(repo), name_list, "add  a\n")
    assert union_list == [f"codetwine-loom/sandbox/{held}"]
    assert sandbox._drop_resume_branch_list(str(repo), str(out), "repo", "",
                                        union_list) == [f"codetwine-loom/sandbox/{held}"]
    assert set(name_list[1:4]) <= set(_branches(repo))


def _gap_union_in_tree(repo, run_id):
    """Leave a union whose commit adds a line to kept.txt and records a case not done,
    with that line in the working tree, and return the branch."""
    branch = f"codetwine-loom/sandbox/{run_id}"
    _git(repo, "checkout", "--quiet", "-b", branch, "main")
    (repo / "kept.txt").write_text("one\nwhat the union added\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m",
         "integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Run-Request: add a\n"
         "Codetwine-Loom-Verdict: settled\n")
    _git(repo, "checkout", "--quiet", "main")
    _stage_commit(repo, branch, _GAP_TEXT)
    (repo / "kept.txt").write_text("one\nwhat the union added\n", encoding="utf-8")
    return branch


def test_a_union_whose_lines_the_run_changed_again_still_goes(_dispatch_repo, monkeypatch):
    """What says the working tree held the union's change is read before the run put
    its own union in, which changed the same lines."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "live_run_list", lambda *a: [])
    branch = _gap_union_in_tree(repo, "20260102-000000-1-1")
    union_list = sandbox.gap_union_list_of(str(repo), [branch], "add a")
    assert union_list == [branch]
    (repo / "kept.txt").write_text("one\nwhat the next run made of it\n", encoding="utf-8")
    assert sandbox.gap_union_list_of(str(repo), [branch], "add a") == []
    assert sandbox._drop_resume_branch_list(str(repo), str(out), "repo", "",
                                        union_list) == [branch]
    assert branch not in _branches(repo)


def test_a_union_is_left_where_it_is_while_another_process_has_a_run_going(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    branch = _gap_union_in_tree(repo, "20260102-000000-1-1")
    monkeypatch.setattr(sandbox, "live_run_list", lambda *a: [
        f"a1/20260102-000000-5-{os.getpid()}", "a2/20260102-000000-6-1"])
    assert sandbox._drop_resume_branch_list(str(repo), str(out), "repo", "", [branch]) == []
    assert branch in _branches(repo)
    # A run of this process alone does not keep it
    monkeypatch.setattr(sandbox, "live_run_list", lambda *a: [
        f"a1/20260102-000000-5-{os.getpid()}"])
    assert sandbox._drop_resume_branch_list(str(repo), str(out), "repo", "", [branch]) == [branch]


def _settle_union(repo, run_id, path="kept.txt", trailer="Codetwine-Loom-Run-Request: add a\n"):
    """Leave a union on a branch of this tool's naming: a merge commit that adds a
    file and records a settled merge. The working tree is left as it was. Return the
    branch."""
    branch, side = f"codetwine-loom/sandbox/{run_id}", f"side-{run_id}"
    _git(repo, "checkout", "--quiet", "-b", side, "main")
    (repo / path).write_text("what the union added\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m",
         "one side")
    _git(repo, "checkout", "--quiet", "-b", branch, "main")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "merge", "--quiet", "--no-ff",
         "-m", f"integrate: x\n\ncombine a, b\n\n{trailer}Codetwine-Loom-Verdict: settled\n", side)
    _git(repo, "checkout", "--quiet", "main")
    _git(repo, "branch", "--quiet", "-D", side)
    return branch


def test_a_branch_whose_result_the_working_tree_holds_is_none_to_carry_on_from(
        _dispatch_repo):
    """A settled union is one to carry on from until the working tree holds what it
    made. A branch that cannot be read stays one. Only a name this tool made up is
    among the branches a run that got through takes away."""
    repo, _ = _dispatch_repo
    union = _settle_union(repo, "20260102-000000-1-1")
    _git(repo, "branch", "mine/union", union)
    name_list = [union, "mine/union", "codetwine-loom/sandbox/gone"]
    assert sandbox.open_branch_list(str(repo), name_list) == name_list
    assert sandbox._result_in_tree_list(str(repo), name_list) == []
    (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    assert sandbox.open_branch_list(str(repo), name_list) == ["codetwine-loom/sandbox/gone"]
    assert sandbox._result_in_tree_list(str(repo), name_list) == [union]


def test_a_settled_union_the_working_tree_does_not_hold_is_one_to_put_in(_dispatch_repo):
    """The union is a merge commit that records a settled merge, whether or not its
    tests found a case not done. One the working tree holds, a commit of one parent and
    a branch that is not there are none."""
    repo, _ = _dispatch_repo
    union = _settle_union(repo, "20260102-000000-1-1")
    other = _settle_union(repo, "20260102-000000-2-1", path="other.txt")
    _stage_commit(repo, other, _GAP_TEXT)
    _git(repo, "checkout", "--quiet", "-b", "codetwine-loom/sandbox/20260102-000000-3-1", "main")
    (repo / "one.txt").write_text("a run of its own\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m",
         "edit: x\n\nx\n\nCodetwine-Loom-Run-Request: add a\nCodetwine-Loom-Verdict: unverified\n")
    _git(repo, "checkout", "--quiet", "main")
    assert [sandbox.is_union_to_land(str(repo), name) for name in (
        union, other, "codetwine-loom/sandbox/20260102-000000-3-1",
        "codetwine-loom/sandbox/gone")] == [
            True, True, False, False]
    assert not sandbox.is_gap_union(str(repo), other)
    (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    (repo / "other.txt").write_text("what the union added\n", encoding="utf-8")
    assert not sandbox.is_union_to_land(str(repo), union)
    assert not sandbox.is_union_to_land(str(repo), other)
    assert sandbox.is_gap_union(str(repo), other)
    assert not sandbox.is_gap_union(str(repo), union)


_REFACTOR_TEXT = ("refactor: x\n\nx\n\nCodetwine-Loom-Run-Request: add a\n"
                  "Codetwine-Loom-Stage: refactor\nCodetwine-Loom-Verdict: settled\n")


def _stage_change(repo, branch, path, message):
    """Put one commit that adds a file on a branch, with a stage's trailers, and leave
    the working tree as it was."""
    _git(repo, "checkout", "--quiet", branch)
    (repo / path).write_text("what the stage added\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m",
         message)
    _git(repo, "checkout", "--quiet", "main")


def test_a_union_is_held_by_a_working_tree_that_holds_it_up_to_its_tip(_dispatch_repo):
    """A stage settled over a merge the working tree holds, and what the stage wrote
    was not put in: the branch is still one to carry on from, by putting it in."""
    repo, _ = _dispatch_repo
    union = _settle_union(repo, "20260102-000000-1-1")
    _stage_change(repo, union, "stage.txt", _REFACTOR_TEXT)
    gap = _settle_union(repo, "20260102-000000-2-1", path="other.txt")
    _stage_change(repo, gap, "test_other.txt", _GAP_TEXT)

    def read():
        return (sandbox._is_in_tree(str(repo), union),
                sandbox.open_branch_list(str(repo), [union]),
                sandbox._result_in_tree_list(str(repo), [union]),
                sandbox.is_union_to_land(str(repo), union),
                sandbox.is_gap_union(str(repo), gap))

    assert read() == (False, [union], [], True, False)
    # The merge alone: what integrate.branch_record_dict() calls held
    (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    (repo / "other.txt").write_text("what the union added\n", encoding="utf-8")
    assert integrate_mod.branch_record_dict(str(repo), union)["held"] is True
    assert integrate_mod.branch_record_dict(str(repo), gap)["held"] is True
    assert read() == (False, [union], [], True, False)
    # The stage's commit alone
    (repo / "kept.txt").unlink()
    (repo / "stage.txt").write_text("what the stage added\n", encoding="utf-8")
    assert read() == (False, [union], [], True, False)
    # All of it
    (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    (repo / "test_other.txt").write_text("what the stage added\n", encoding="utf-8")
    assert read() == (True, [], [union], False, True)
    # The user took a line of it back out
    (repo / "stage.txt").write_text("mine\n", encoding="utf-8")
    assert read()[:4] == (False, [union], [], True)
    # What is checked out reaches the tip
    _git(repo, "stash", "--quiet", "--include-untracked")
    _git(repo, "merge", "--quiet", "--ff-only", union)
    assert read()[:4] == (True, [], [union], False)
    assert sandbox._is_in_tree(str(repo), "codetwine-loom/sandbox/gone") is False


def test_a_stage_that_takes_back_what_the_merge_added_is_held_only_once_it_is_gone(
        _dispatch_repo):
    """What the merge added and the stage took away is in no diff from the start to
    the tip, and a working tree that still holds it does not hold the stage."""
    repo, _ = _dispatch_repo
    union = _settle_union(repo, "20260102-000000-1-1")
    _git(repo, "checkout", "--quiet", union)
    _git(repo, "rm", "--quiet", "kept.txt")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m",
         _REFACTOR_TEXT)
    _git(repo, "checkout", "--quiet", "main")
    (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    assert (sandbox._is_in_tree(str(repo), union),
            sandbox.open_branch_list(str(repo), [union]),
            sandbox._result_in_tree_list(str(repo), [union]),
            sandbox.is_union_to_land(str(repo), union)) == (False, [union], [], True)
    (repo / "kept.txt").unlink()
    assert (sandbox._is_in_tree(str(repo), union),
            sandbox.open_branch_list(str(repo), [union]),
            sandbox._result_in_tree_list(str(repo), [union]),
            sandbox.is_union_to_land(str(repo), union)) == (True, [], [union], False)


def test_the_tree_commit_a_union_is_carried_on_over_is_what_its_held_commit_became(
        _dispatch_repo, monkeypatch):
    """The commit the working tree makes is written over the last commit of the union
    the working tree holds, so what a stage over it took away is merged as a change."""
    repo, out = _dispatch_repo
    union = _settle_union(repo, "20260102-000000-1-1")
    merge = _git(repo, "rev-parse", union).strip()
    _git(repo, "checkout", "--quiet", union)
    _git(repo, "rm", "--quiet", "kept.txt")
    (repo / "stage.txt").write_text("what the stage added\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet", "-m",
         _REFACTOR_TEXT)
    tip = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "--quiet", "main")
    assert sandbox._held_union_commit(str(repo), union) == ""
    (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    assert sandbox._held_union_commit(str(repo), union) == merge
    earlier_list = []

    @contextlib.contextmanager
    def tree_branch(project_dir, run_id, earlier="", **kwargs):
        earlier_list.append(earlier)
        raise RuntimeError("stop here")
        yield

    monkeypatch.setattr(sandbox, "tree_branch", tree_branch)
    # The union is merged again, as where the working tree holds part of it
    monkeypatch.setattr(sandbox, "_land_union_as_is", lambda *args: None)
    with pytest.raises(RuntimeError, match="stop here"):
        sandbox.integrate_into_tree(str(repo), str(out), [], resume=union)
    (repo / "kept.txt").unlink()
    (repo / "stage.txt").write_text("what the stage added\n", encoding="utf-8")
    assert sandbox._held_union_commit(str(repo), union) == tip
    assert earlier_list == [merge]


def test_a_union_written_against_a_commit_of_the_working_tree_is_carried_on_over_its_held_commit(
        _dispatch_repo, monkeypatch):
    """The earlier merge was written against a commit of a working tree that held a
    file git did not track. The commit the tree makes now is written over the last
    commit of the union the working tree holds all the same, and over that commit of
    the working tree when it holds none of the union."""
    repo, out = _dispatch_repo
    identity = ("-c", "user.email=t@t", "-c", "user.name=t")
    (repo / "notes.txt").write_text("mine\n", encoding="utf-8")
    snapshot = gitcmd.snapshot_commit(str(repo))
    (repo / "notes.txt").unlink()
    union = "codetwine-loom/sandbox/20260102-000000-1-1"
    _git(repo, "checkout", "--quiet", "-b", "side", snapshot)
    (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, *identity, "commit", "--quiet", "-m", "one side")
    _git(repo, "checkout", "--quiet", "-b", union, snapshot)
    _git(repo, *identity, "merge", "--quiet", "--no-ff", "-m",
         "integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Run-Request: add a\n"
         "Codetwine-Loom-Verdict: settled\n",
         "side")
    merge = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "rm", "--quiet", "kept.txt")
    (repo / "stage.txt").write_text("what the stage added\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, *identity, "commit", "--quiet", "-m", _REFACTOR_TEXT)
    _git(repo, "checkout", "--quiet", "main")
    _git(repo, "branch", "--quiet", "-D", "side")
    (repo / "notes.txt").write_text("mine\n", encoding="utf-8")
    assert gitcmd.is_snapshot(str(repo), sandbox.resume_head(str(repo), union)[1])
    earlier_list = []

    @contextlib.contextmanager
    def tree_branch(project_dir, run_id, earlier="", **kwargs):
        earlier_list.append(earlier)
        raise RuntimeError("stop here")
        yield

    monkeypatch.setattr(sandbox, "tree_branch", tree_branch)
    # The union is merged again, as where the working tree holds part of it
    monkeypatch.setattr(sandbox, "_land_union_as_is", lambda *args: None)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="stop here"):
            sandbox.integrate_into_tree(str(repo), str(out), [], resume=union)
        (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    assert earlier_list == [snapshot, merge]


def test_a_branch_whose_change_git_cannot_read_is_not_held(_dispatch_repo, monkeypatch):
    repo, _ = _dispatch_repo
    union = _settle_union(repo, "20260102-000000-1-1")
    (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    assert sandbox._is_in_tree(str(repo), union) is True
    monkeypatch.setattr(gitcmd, "has_change_in_tree", lambda *a: None)
    assert sandbox._is_in_tree(str(repo), union) is False
    assert sandbox.open_branch_list(str(repo), [union]) == [union]
    assert sandbox._result_in_tree_list(str(repo), [union]) == []


@pytest.mark.parametrize("message, is_merge, is_resume", [
    (("integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: settled\n"
      "Codetwine-Loom-Exit-Status: 0\n"), True, True),
    ("integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: settled\n", True, True),
    # The subject alone: a commit of one parent
    (("integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: settled\n"
      "Codetwine-Loom-Exit-Status: 0\n"), False, False),
    # The parents alone: a merge commit somebody else wrote
    (("Merge branch 'side'\n\ncombine a, b\n\nCodetwine-Loom-Verdict: settled\n"
      "Codetwine-Loom-Exit-Status: 0\n"), True, False),
    (("integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: settled\n"
      "Codetwine-Loom-Exit-Status: 1\n"), True, False),
    (("integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: settled\n"
      "Codetwine-Loom-Exit-Status: 137\n"), True, False),
    (("integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: unsettled\n"
      "Codetwine-Loom-Exit-Status: 0\n"), True, False),
    ("integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Exit-Status: 0\n", True, False),
    # The commit of a change run whose container recorded a merge's verdict
    ("sandbox: x\n\nx\n\nCodetwine-Loom-Verdict: settled\nCodetwine-Loom-Exit-Status: 1\n",
     False, False),
    ("sandbox: x\n\nx\n\nCodetwine-Loom-Verdict: settled\nCodetwine-Loom-Exit-Status: 0\n",
     False, False),
    ("sandbox: x\n\nx\n\nCodetwine-Loom-Verdict: settled\nCodetwine-Loom-Exit-Status: 0\n",
     True, False),
    ("sandbox: integrate: x\n\nintegrate: x\n\nCodetwine-Loom-Verdict: settled\n", True, False),
    # A request that holds a record of its own above the commit's
    ("sandbox: x\n\nintegrate: x\n\nCodetwine-Loom-Verdict: settled\n"
     "# ------------------------ >8 ------------------------\n\n"
     "Codetwine-Loom-Verdict: failed\nCodetwine-Loom-Exit-Status: 1\n", True, False)])
def test_only_a_merge_a_merging_run_settled_is_carried_on_from_by_its_stages(
        _dispatch_repo, monkeypatch, message, is_merge, is_resume):
    """The stages run over what the gates of a merge let through, and over nothing a
    container alone said was settled. A commit is such a merge by its subject and
    its second parent together, for the stages and for the line that says how to
    put a union into the working tree alike."""
    repo, _ = _dispatch_repo
    monkeypatch.setattr(config, "REFACTOR", True)
    branch = "codetwine-loom/sandbox/20260102-000000-2-7"
    # The commit adds a file the working tree does not hold, with one parent or two
    _change_on_new_branch(repo, "side", "main", "union.txt", "one side")
    if is_merge:
        _git(repo, "checkout", "--quiet", "-b", branch, "main")
        _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "merge", "--quiet",
             "--no-ff", "-m", message, "side")
        _git(repo, "checkout", "--quiet", "main")
    else:
        _git(repo, "branch", branch, "side")
        _git(repo, "checkout", "--quiet", branch)
        _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--quiet",
             "--amend", "-m", message)
        _git(repo, "checkout", "--quiet", "main")
    _git(repo, "branch", "--quiet", "-D", "side")
    tip = _git(repo, "rev-parse", branch).strip()
    assert integrate_mod.is_settle_merge(str(repo), tip) == is_resume
    assert bool(sandbox.stage_resume_dict(str(repo), branch)) == is_resume
    assert sandbox._is_stage_ahead(str(repo), branch) == is_resume
    # With no stage left to run, the same commit is what says a union is one to put in
    monkeypatch.setattr(config, "REFACTOR", False)
    assert sandbox.stage_resume_dict(str(repo), branch) == {}
    assert sandbox.is_union_to_land(str(repo), branch) == is_resume
    monkeypatch.setattr(config, "REFACTOR", True)
    # The same under the commit of a stage that was left open
    _stage_commit(repo, branch, "refactor: x\n\nx\n\nCodetwine-Loom-Stage: refactor\n"
                                "Codetwine-Loom-Verdict: unsettled\n")
    resume_dict = sandbox.stage_resume_dict(str(repo), branch)
    assert bool(resume_dict) == is_resume
    if is_resume:
        assert (resume_dict["stage"], resume_dict["merge"]) == ("refactor", tip)
    assert integrate_mod.is_settle_merge(str(repo), "no-such-commit") is False


def test_a_union_that_carries_no_request_is_carried_on_from_for_any_request(
        _dispatch_repo):
    repo, _ = _dispatch_repo
    union = _settle_union(repo, "20260102-000000-1-1", trailer="")
    _stage_commit(repo, union, _GAP_TEXT.replace("Codetwine-Loom-Run-Request: add a\n", ""))
    (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    assert sandbox.gap_union_list_of(str(repo), [union], "whatever is asked") == [union]


@pytest.mark.parametrize("merge_status", [0, 1])
def test_a_branch_whose_result_the_tree_holds_goes_once_the_run_carried_on_from_it_got_through(
        _dispatch_repo, monkeypatch, merge_status):
    """The run was passed a settled union the working tree holds, a branch a person
    named at the same commit and a settled union the working tree does not hold.
    Only the first goes, and only when the run got through."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "live_run_list", lambda *a: [])
    held = _settle_union(repo, "20260102-000000-1-1")
    apart = _settle_union(repo, "20260102-000000-2-1", path="other.txt")
    (repo / "kept.txt").write_text("what the union added\n", encoding="utf-8")
    _git(repo, "branch", "mine/union", held)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one"]}))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 0))
    _git(repo, "branch", "codetwine-loom/sandbox/the-new-union")
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        "exit_status": merge_status, "commit": "c" * 40,
        "branch": "codetwine-loom/sandbox/the-new-union",
        "verdict": "unsettled" if merge_status else "settled",
        "land": None if merge_status else {"files": 1, "paths": ["a.py"], "patch": "p",
                                           "last": ""}})
    result = sandbox.dispatch(str(repo), str(out), "add b",
                              resume_list=[held, "mine/union", apart])
    assert result["exit_status"] == merge_status
    assert (held in result["dropped"]) == (not merge_status)
    assert (held in _branches(repo)) == bool(merge_status)
    assert "mine/union" in _branches(repo) and apart in _branches(repo)


def test_a_union_passed_by_hand_keeps_a_run_that_put_nothing_in_from_getting_through(
        _dispatch_repo, monkeypatch):
    """The cases recorded on it count as those of a run before the first: the exit
    status is non-zero, the cases are handed back and the union is named again."""
    repo, out = _dispatch_repo
    branch = _gap_union_in_tree(repo, "20260102-000000-1-1")
    told = []

    def once(*a, **k):
        told.append(list(k["resume_list"]))
        return {"tasks": ["t"], "branches": [], "integration": None, "resume": [],
                "exit_status": 0}

    monkeypatch.setattr(sandbox, "_dispatch_once", once)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 0)
    result = sandbox.dispatch(str(repo), str(out), "add a", resume_list=[branch])
    assert result["exit_status"] == 1 and result["resume"] == [branch]
    assert result["earlier_gap"] == [{"what": "case-1", "why": "not handled"}]
    # Another request passed the same union is not held to its cases
    result = sandbox.dispatch(str(repo), str(out), "add b", resume_list=[branch])
    assert result["exit_status"] == 0 and result["earlier_gap"] == []
    # A question is answered and counts no case
    monkeypatch.setattr(sandbox, "_dispatch_once", lambda *a, **k: {
        "questions": ["q"], "answers": [], "combined": {"answer": "a", "error": ""},
        "answer": "a", "exit_status": 0})
    result = sandbox.dispatch(str(repo), str(out), "add a", resume_list=[branch])
    assert result["exit_status"] == 0 and result["earlier_gap"] == []


@pytest.mark.parametrize("gap_list", [[], [{"what": "case-2", "why": "w"}]])
def test_a_union_carried_on_from_goes_once_the_run_put_its_own_in_cases_or_none(
        _dispatch_repo, monkeypatch, gap_list):
    """The run's own union is in the working tree, with a case its tests found not
    done or with none: either way it stands for the union it carried on from."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "live_run_list", lambda *a: [])
    earlier = _gap_union_in_tree(repo, "20260102-000000-1-1")
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one"]}))
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 0))
    _git(repo, "branch", "codetwine-loom/sandbox/the-new-union")
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        "exit_status": 0, "commit": "c" * 40, "branch": "codetwine-loom/sandbox/the-new-union",
        "land": {"files": 1, "paths": ["a.py"], "patch": "p", "last": ""},
        "gap": gap_list})
    result = sandbox.dispatch(str(repo), str(out), "add a", resume_list=[earlier])
    assert result["exit_status"] == (1 if gap_list else 0)
    assert earlier in result["dropped"] and earlier not in _branches(repo)
    assert result["resume"] == (["codetwine-loom/sandbox/the-new-union"] if gap_list else [])
    # The cases the earlier union recorded are no longer counted
    assert result["earlier_gap"] == []


@pytest.mark.parametrize("split_dict", [{"tasks": ["one"]}, {"questions": ["where?"]}])
def test_a_reading_run_that_ended_with_another_status_after_its_split_read_the_request(
        _dispatch_repo, monkeypatch, caplog, split_dict):
    """Every task got through and the union is in the working tree, or the question
    was answered: the run got through, whatever the reading run's container ended with."""
    repo, out = _dispatch_repo

    def start(*a, **k):
        _split_written(_tasks_file(out), split_dict)(*a, **k)
        return "/usr/bin/podman", "img", sandbox._TIME_OUT_STATUS

    monkeypatch.setattr(sandbox, "_start_container", start)
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 0))
    monkeypatch.setattr(sandbox, "answer", _answered_with("it is a.py"))
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        "exit_status": 0, "commit": "c" * 40, "branch": None,
        "land": {"files": 1, "paths": ["a.py"], "patch": "p", "last": ""}})
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        result = sandbox.dispatch(str(repo), str(out), "add a")
    assert result["exit_status"] == 0 and result["runs"] == 1
    assert not result.get("resume")
    assert "ended with exit status 124 after it wrote the split" in caplog.text


def test_cases_a_run_left_not_done_stay_with_a_run_again_that_put_no_union_in(monkeypatch):
    """The second run changed nothing and ended with 0: the working tree still holds
    the change the first run's tests found a case not done by."""
    land = {"files": 1, "paths": ["a.py"], "patch": "p", "last": "/out/land-last.diff"}
    gap_list = [{"what": "case-1", "why": "not handled"}]
    told = []
    run_list = [
        {"tasks": ["t"], "integration": {"land": land, "gap": gap_list},
         "resume": ["codetwine-loom/u1"], "exit_status": 1},
        {"tasks": ["t"], "branches": [], "integration": None, "resume": [],
         "exit_status": 0}]

    def once(*a, **k):
        told.append(list(k["resume_list"]))
        return run_list.pop(0)

    monkeypatch.setattr(sandbox, "_dispatch_once", once)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
    result = sandbox.dispatch("repo", "out", "add a")
    assert told == [[], ["codetwine-loom/u1"]]
    assert result["runs"] == 2 and result["exit_status"] == 1
    assert result["earlier_gap"] == gap_list and result["resume"] == ["codetwine-loom/u1"]
    assert result["earlier_land"] == [land]


def test_a_run_again_that_left_branches_names_them_after_the_union_with_the_cases(
        monkeypatch):
    gap_list = [{"what": "case-1", "why": "not handled"}]
    run_list = [
        {"tasks": ["t"], "integration": {"land": {"files": 1}, "gap": gap_list},
         "resume": ["codetwine-loom/u1"], "exit_status": 1},
        {"tasks": ["t"], "integration": None, "resume": ["codetwine-loom/run/r2/1"],
         "exit_status": 1},
        {"unclear": ["which?"], "resume": ["codetwine-loom/u1", "codetwine-loom/run/r2/1"],
         "exit_status": 1}]
    told = []

    def once(*a, **k):
        told.append(list(k["resume_list"]))
        return run_list.pop(0)

    monkeypatch.setattr(sandbox, "_dispatch_once", once)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    result = sandbox.dispatch("repo", "out", "add a")
    assert told == [[], ["codetwine-loom/u1"], ["codetwine-loom/u1", "codetwine-loom/run/r2/1"]]
    assert result["resume"] == ["codetwine-loom/u1", "codetwine-loom/run/r2/1"]
    assert result["earlier_gap"] == gap_list and result["exit_status"] == 1


def test_a_run_again_that_put_a_union_in_stands_for_the_cases_of_the_run_before(
        monkeypatch):
    gap_list = [{"what": "case-1", "why": "not handled"}]
    for last_gap_list, status in (([], 0), ([{"what": "case-2", "why": "w"}], 1)):
        run_list = [
            {"tasks": ["t"], "integration": {"land": {"files": 1}, "gap": gap_list},
             "resume": ["codetwine-loom/u1"], "exit_status": 1},
            {"tasks": ["t"], "integration": {"land": {"files": 1}, "gap": last_gap_list},
             "resume": ["codetwine-loom/u2"] if status else [], "exit_status": status}]
        monkeypatch.setattr(sandbox, "_dispatch_once", lambda *a, **k: run_list.pop(0))
        monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 1)
        result = sandbox.dispatch("repo", "out", "add a")
        assert result["earlier_gap"] == [] and result["exit_status"] == status
        assert result["resume"] == (["codetwine-loom/u2"] if status else [])


def test_a_union_kept_out_for_its_paths_is_not_made_again_and_not_carried_on_from(
        _dispatch_repo, monkeypatch, caplog):
    """The tests found a case not done, so the run did not get through; its union is
    one that is kept out of the working tree however often it is made."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "live_run_list", lambda *a: [])
    made = []

    def start(*a, **k):
        made.append("run")
        return _split_written(_tasks_file(out), {"tasks": ["one"]})(*a, **k)

    monkeypatch.setattr(sandbox, "_start_container", start)
    monkeypatch.setattr(sandbox, "run", _landing_on_its_branch(repo, 0))
    _git(repo, "branch", "codetwine-loom/sandbox/the-union")
    monkeypatch.setattr(sandbox, "integrate", lambda *a, **k: {
        "exit_status": 0, "commit": "c" * 40, "branch": "codetwine-loom/sandbox/the-union",
        "land": None, "land_why": "the change writes .env, which ...",
        "land_refuse": True, "gap": [{"what": "case-1", "why": "w"}]})
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        result = sandbox.dispatch(str(repo), str(out), "add a")
    assert made == ["run"] and result["runs"] == 1
    assert result["exit_status"] == 1 and result["resume"] == []
    assert "codetwine-loom/sandbox/the-union" in _branches(repo)
    assert "so the request is not run again" in caplog.text


@pytest.mark.parametrize("more_dict, status, expect_list", [
    ({"land_refuse": True}, 1, []),
    ({"land_refuse": True, "refactor": {"verdict": "unsettled", "skipped": False}}, 0,
     []),
    ({"land_refuse": False, "refactor": {"verdict": "unsettled", "skipped": False}}, 0,
     ["codetwine-loom/sandbox/u"]),
    ({"land_refuse": False}, 1, ["codetwine-loom/sandbox/u"]),
    ({}, 1, ["codetwine-loom/sandbox/u"]),
    ({"land_refuse": False}, 0, ["codetwine-loom/sandbox/u"]),
    ({"land_refuse": False, "land_why": ""}, 0, []),
])
def test_a_change_kept_out_for_its_paths_names_no_branch_to_carry_on_from(
        more_dict, status, expect_list):
    union_dict = {"branch": "codetwine-loom/sandbox/u", "exit_status": 0, "land": None,
                  "land_why": "it was not put in", **more_dict}
    assert sandbox.resume_branch_list([], union_dict, status) == expect_list


def test_what_was_put_into_a_working_tree_is_kept_for_a_command_that_is_stopped(
        tmp_path):
    repo = _repo(tmp_path / "repo")
    base = _git(repo, "rev-parse", "HEAD").strip()
    patch = tmp_path / "patch.diff"
    patch.write_text(_patch_from(repo, lambda clone: (clone / "kept.txt").write_text(
        "one\ntwo\n", encoding="utf-8")), encoding="utf-8")
    taken = sandbox.take_in_patch(str(repo), str(patch), base, "codetwine-loom/sandbox/one",
                                  str(tmp_path / "wt"), "sandbox: x")
    assert sandbox.land_list() == []
    # A diff that touches nothing is not counted, and one that does not apply is not
    assert sandbox.land_in_tree(str(repo), base, base, str(tmp_path / "l0.diff"),
                                str(tmp_path / "last.diff"))["files"] == 0
    land_dict = sandbox.land_in_tree(str(repo), base, taken["commit"],
                                     str(tmp_path / "l1.diff"), str(tmp_path / "last.diff"))
    with pytest.raises(RuntimeError, match="does not apply"):
        sandbox.land_in_tree(str(repo), base, taken["commit"], str(tmp_path / "l2.diff"),
                             str(tmp_path / "last.diff"))
    assert sandbox.land_list() == [land_dict] and land_dict["paths"] == ["kept.txt"]
    sandbox.reset_land_list()
    assert sandbox.land_list() == []


def test_the_branch_of_a_run_on_its_own_that_failed_takes_a_place(_dispatch_repo,
                                                                 monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    old, new = "20260101-000000-1-11", "20260102-000000-1-22"
    for run_id in (old, new):
        _union_left(repo, run_id, "failed")
        os.makedirs(sandbox.agent_dir(str(out), run_id))
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [old]
    assert f"codetwine-loom/sandbox/{new}" in _branches(repo) and f"codetwine-loom/sandbox/{old}" not in _branches(repo)
    assert not os.path.exists(sandbox.agent_dir(str(out), old))


@pytest.mark.parametrize("verdict", ["unverified", "tests_red", "no_change"])
def test_the_branch_of_a_run_on_its_own_that_got_through_is_never_taken_away(
        _dispatch_repo, monkeypatch, verdict):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    run_id = "20260101-000000-1-11"
    _union_left(repo, run_id, verdict)
    os.makedirs(sandbox.agent_dir(str(out), run_id))
    assert sandbox.remove_old_run(str(repo), str(out), 1) == []
    assert f"codetwine-loom/sandbox/{run_id}" in _branches(repo)


def test_the_output_of_runs_on_their_own_that_got_through_is_taken_away(_dispatch_repo,
                                                                       monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    run_list = ["20260101-000000-1-11", "20260102-000000-1-22"]
    for run_id in run_list:
        os.makedirs(sandbox.agent_dir(str(out), run_id))
    monkeypatch.setattr(config, "DISPATCH_KEEP_RUN", 3)
    assert sandbox.remove_old_run_in(str(repo), str(out)) == run_list[::-1]
    assert not any(os.path.exists(sandbox.agent_dir(str(out), run_id))
                   for run_id in run_list)


def test_an_unsettled_union_with_no_task_branch_still_takes_a_place(_dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_run_dead", lambda run_id, engine="": True)
    old, new = "20260101-000000-1-11", "20260102-000000-1-22"
    for run_id in (old, new):
        _union_left(repo, run_id, "unsettled")
        os.makedirs(sandbox.agent_dir(str(out), run_id))
    assert sandbox.remove_old_run(str(repo), str(out), 1) == [old]
    assert f"codetwine-loom/sandbox/{new}" in _branches(repo) and f"codetwine-loom/sandbox/{old}" not in _branches(repo)


# == carrying a merge on from the branch it left =========


def _unsettled_union_off(repo, branch, base):
    """Put an unsettled union on a branch whose first parent is base."""
    _git(repo, "checkout", "-q", "-b", branch, base)
    (repo / "u.py").write_text("<<<<<<< a\nx = 2\n=======\nx = 3\n>>>>>>> b\n",
                               encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm",
         "integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: unsettled\n")
    _git(repo, "checkout", "-q", "main")


def test_a_merge_is_carried_on_from_the_branch_it_left(_dispatch_repo, monkeypatch):
    """One branch is enough with --resume: it is combined onto the commit the earlier
    merge was written against, which is its first parent."""
    repo, out = _dispatch_repo
    base = _git(repo, "rev-parse", "HEAD").strip()
    _unsettled_union_off(repo, "codetwine-loom/sandbox/earlier", base)
    seen = {}
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: seen.update(base=a[7], argv=a[10]) or ("/usr/bin/podman", "img", 0))
    monkeypatch.setattr(sandbox, "_land_patch", lambda *a, **k: {"exit_status": 0})
    out_dict = sandbox.integrate(str(repo), str(out), [], resume="codetwine-loom/sandbox/earlier")
    assert seen["base"] == base and out_dict["branches"] == ["codetwine-loom/sandbox/earlier"]
    assert seen["argv"][seen["argv"].index("--resume") + 1] == "codetwine-loom/sandbox/earlier"
    assert "--branch" not in seen["argv"]


def test_the_run_inside_is_told_which_branch_the_earlier_merge_left(_dispatch_repo,
                                                                     monkeypatch):
    """It combines that branch first and reads its record, so it is named apart from
    the others."""
    repo, out = _dispatch_repo
    base = _git(repo, "rev-parse", "HEAD").strip()
    _unsettled_union_off(repo, "codetwine-loom/sandbox/earlier", base)
    _git(repo, "branch", "c")
    seen = {}
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: seen.update(argv=a[10]) or ("/usr/bin/podman", "img", 0))
    monkeypatch.setattr(sandbox, "_land_patch", lambda *a, **k: {"exit_status": 0})
    sandbox.integrate(str(repo), str(out), ["c"], resume="codetwine-loom/sandbox/earlier")
    argv = seen["argv"]
    assert argv[argv.index("--resume") + 1] == "codetwine-loom/sandbox/earlier"
    assert argv[argv.index("--branch") + 1] == "c" and argv.count("--branch") == 1


def test_the_merge_commit_s_parents_are_the_commits_read_before_the_container(
        _dispatch_repo, monkeypatch):
    repo, out = _dispatch_repo
    base = _git(repo, "rev-parse", "HEAD").strip()
    _unsettled_union_off(repo, "codetwine-loom/sandbox/earlier", base)
    earlier = _git(repo, "rev-parse", "codetwine-loom/sandbox/earlier").strip()
    seen = {}

    def started(*args, **kwargs):
        # The branch moves on while the container runs
        _git(repo, "checkout", "-q", "codetwine-loom/sandbox/earlier")
        (repo / "later.py").write_text("y = 1\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "later")
        _git(repo, "checkout", "-q", "main")
        return "/usr/bin/podman", "img", 0

    monkeypatch.setattr(sandbox, "_start_container", started)
    monkeypatch.setattr(sandbox, "_land_patch",
                        lambda *a, **k: seen.update(parent_list=k["parent_list"])
                        or {"exit_status": 0})
    sandbox.integrate(str(repo), str(out), [], resume="codetwine-loom/sandbox/earlier")
    assert seen["parent_list"] == [earlier]


def test_the_parents_the_run_merged_replace_the_ones_read_before_it():
    read_list = ["1" * 40, "2" * 40]
    assert sandbox._parent_list_of({"parents": ["3" * 40, "4" * 40]}, read_list) == \
        ["3" * 40, "4" * 40]
    for parents in (None, ["3" * 40], ["--all", "4" * 40], [3, "4" * 40], "3" * 40):
        assert sandbox._parent_list_of({"parents": parents}, read_list) == read_list


def test_a_working_tree_behind_where_the_branches_start_is_refused(_dispatch_repo,
                                                                    monkeypatch):
    """The union would be written against the start and put onto the older commit."""
    repo, out = _dispatch_repo
    old = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "b.py").write_text("y = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "two")
    base = _git(repo, "rev-parse", "HEAD").strip()
    _unsettled_union_off(repo, "codetwine-loom/sandbox/earlier", base)
    _git(repo, "branch", "old", old)
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: pytest.fail("a container was started"))
    with pytest.raises(RuntimeError, match="behind"):
        sandbox.integrate(str(repo), str(out), [], resume="codetwine-loom/sandbox/earlier",
                          onto="old")


def test_nothing_to_combine_says_to_pass_resume_as_well(_dispatch_repo):
    repo, out = _dispatch_repo
    with pytest.raises(RuntimeError, match="or --resume"):
        sandbox.integrate(str(repo), str(out), ["a"])


def test_the_names_codetwine_keeps_out_reach_the_container(monkeypatch):
    """The analysis a container makes for itself covers what the analysis here covers."""
    names = sandbox.env_forward_argv()[1::2]
    assert "EXCLUDE_PATTERNS" in names


def test_the_documents_are_asked_about_before_the_lock_is_taken(tmp_path, monkeypatch):
    """A run starting in the same output directory is not held up by a question this
    one is waiting on."""
    order = []
    monkeypatch.setattr(sandbox.gitcmd, "clean_head", lambda *a: "abc123")
    monkeypatch.setattr(sandbox.knowledge, "ask_doc",
                        lambda path: order.append("asked") or False)
    monkeypatch.setattr(sandbox.build, "analyse_and_index",
                        lambda *a, **k: order.append("analysed"))

    @contextlib.contextmanager
    def note_lock(path, what):
        order.append("locked")
        yield

    monkeypatch.setattr(sandbox.filelock, "hold_lock", note_lock)
    assert _analyse_if_none(str(tmp_path / "p"), str(tmp_path / "out"), "p",
                            None, False, ask_doc=True) is True
    assert order == ["asked", "locked", "analysed"]


# == what a question's run counts as ========================

def test_a_run_whose_agents_all_failed_answers_the_request_in_one_run(_dispatch_repo,
                                                                     monkeypatch):
    """With nothing to put together, the combining stage would spend every turn on
    answers that are not there."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"questions": ["where?",
                                                                        "how?"]}))
    handed = []

    def answering(project, output_dir, question, answer_list=None, **kw):
        handed.append(answer_list)
        if answer_list is None and question != "どこで読んでいる?":
            raise RuntimeError("the run that answers handed no answer back")
        return {"answer": "read it myself", "exit_status": 0}

    monkeypatch.setattr(sandbox, "answer", answering)
    result = sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert handed[-1] is None            # the combining run was given no answers
    assert result["answer"] == "read it myself" and result["exit_status"] == 0


def test_a_combining_run_that_hands_nothing_back_gives_the_answers_without_the_frame(
        _dispatch_repo, monkeypatch):
    """What is printed is the answer the user reads, not the block written for a model."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"questions": ["where?"]}))

    def answering(project, output_dir, question, answer_list=None, **kw):
        if answer_list is not None:
            raise RuntimeError("the run that answers handed no answer back")
        return {"answer": "in a.py", "exit_status": 0}

    monkeypatch.setattr(sandbox, "answer", answering)
    result = sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert "## What the agents answered" not in result["answer"]
    assert "The question above" not in result["answer"]
    assert "in a.py" in result["answer"] and result["exit_status"] == 1


def test_the_reason_an_agent_gave_no_answer_names_no_path_of_this_machine(
        _dispatch_repo, monkeypatch):
    """That reason is read by the combining stage, which has no use for a host path."""
    repo, out = _dispatch_repo
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out),
                                       {"questions": ["where?", "how?"]}))
    handed = {}

    def answering(project, output_dir, question, answer_list=None, **kw):
        if answer_list is not None:
            handed["answer_list"] = answer_list
            return {"answer": "open", "exit_status": 0}
        if question == "how?":
            return {"answer": "in b.py", "exit_status": 0}
        raise RuntimeError(f"the run that answers left no answer in {out}/x.json")

    monkeypatch.setattr(sandbox, "answer", answering)
    sandbox.dispatch(str(repo), str(out), "どこで読んでいる?")
    assert str(out) not in handed["answer_list"][0]["error"]
    assert handed["answer_list"][0]["error"] == "the agent handed no answer back"


def test_the_output_of_a_run_that_answered_a_question_is_kept(_runs_left_behind):
    """A question leaves no branch, and the turn log beside its answer is the only
    account of how it was answered."""
    repo, out, runs = _runs_left_behind
    asked = "20260104-000000-1-44"
    run_file = sandbox.run_file_layout(str(out), asked, "repo", asked)
    os.makedirs(run_file.directory)
    with open(run_file.answer, "w", encoding="utf-8") as handle:
        handle.write('{"answer": "in a.py"}')
    sandbox._write_run_mark(run_file, sandbox._ANSWER_KEEP)
    assert asked not in sandbox.remove_old_run(str(repo), str(out), 1)
    assert os.path.exists(sandbox.agent_dir(str(out), asked))


# == The stage that writes tests for a settled merge ==========

_UNION_PATCH = ("diff --git a/a.txt b/a.txt\nnew file mode 100644\n--- /dev/null\n"
                "+++ b/a.txt\n@@ -0,0 +1 @@\n+x\n"
                "diff --git a/b.txt b/b.txt\nnew file mode 100644\n--- /dev/null\n"
                "+++ b/b.txt\n@@ -0,0 +1 @@\n+x\n")


def _test_patch(path_list):
    """Return a patch adding one test file per source file."""
    return "".join(
        f"diff --git a/tests/test_{path} b/tests/test_{path}\nnew file mode 100644\n"
        f"--- /dev/null\n+++ b/tests/test_{path}\n@@ -0,0 +1 @@\n+t\n" for path in path_list)


# What each stage that refactors changes: a file of the union, and a test written for it
_REFACTOR_PATCH_DICT = {
    "refactor": ("diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n"
                 "@@ -1 +1 @@\n-x\n+y\n"),
    "test-refactor": ("diff --git a/tests/test_a.txt b/tests/test_a.txt\n"
                      "--- a/tests/test_a.txt\n+++ b/tests/test_a.txt\n"
                      "@@ -1 +1 @@\n-t\n+u\n")}


def _two_branch_repo(tmp_path):
    """A repository on main with codetwine-loom/a and /b off it, and an analysis in out."""
    repo = _repo(tmp_path / "repo")
    _analysis_in(tmp_path / "out")
    for name in ("codetwine-loom/a", "codetwine-loom/b"):
        _git(repo, "checkout", "--quiet", "-b", name, "main")
        (repo / f"{name[-1]}.txt").write_text("x\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", name)
    _git(repo, "checkout", "--quiet", "main")
    return repo, tmp_path / "out"


def _test_stage_container(monkeypatch, merge_status=0, skipped="", open_path=(),
                          no_patch=False, target=("a.txt", "b.txt"), share_test="",
                          at_once_list=None, refactor_dict=None, gap_path=(),
                          done_gap_path=(), number_list=None, lose_patch=(),
                          no_def_path=(), lang=None, primary_list=None):
    """Replace the container with one that plays every run of a merge and its tests.

    The listing run names no definition for the files in no_def_path, and says it took
    lang as the primary language when lang is given. primary_list is given (what the
    run was, the primary_ext it was started with) for each run writing tests, combining
    them or refactoring.

    A run whose kind, or one of whose files, is in lose_patch writes what it made of
    itself and neither a patch nor the summary beside one.

    A run that refactors does what refactor_dict says for its stage: "settled" (the
    default) and "unsettled" change one file, "nothing" changes none and is settled,
    and "skipped" calls no model.

    The merging run writes the union; the listing run names target as what there is to
    write tests for, every file of it referred to from share_test too when that is
    given; a run writing tests adds one test file per file it is handed, and is left
    unsettled when one of them is in open_path, and records a gap for each of them in
    gap_path, with a test; the run combining the test tasks writes
    what their branches hold, and finds the test of the gap of each file in
    done_gap_path passing. at_once_list is given how many runs writing tests were
    going when each of them started, and number_list the ("first_case", "case_step")
    each of them was handed. The listing run says the case numbers start at 3.

    Returns:
        [(what the run was: "merge", "list", "tests", "combine", "refactor" or
        "test-refactor", its argument list, the commit it started at, the files it was
        handed), ...], filled in as they start.
    """
    run_list = []
    going_list = []
    share_list = [share_test] if share_test else []

    def start(engine, image, project_dir, root_out_dir, out_dir, name, agent, base,
              run_file, run_id, argv_tail, exclude_list, **kwargs):
        os.makedirs(os.path.dirname(run_file.patch), exist_ok=True)
        path_list, patch, status = [], "", 0
        if argv_tail[0] == "integrate" and "-test/" not in " ".join(argv_tail):
            kind, patch, status = "merge", _UNION_PATCH, merge_status
            outcome_dict = {"verdict": "settled" if merge_status == 0 else "unsettled"}
        elif argv_tail[0] == "integrate":
            kind = "combine"
            if primary_list is not None:
                primary_list.append(("combine", kwargs.get("primary_ext", "")))
            branch_list = [argv_tail[i + 1] for i, word in enumerate(argv_tail)
                           if word == "--branch"]
            patch = "".join(_git(project_dir, "diff", "--binary", base, branch)
                            for branch in branch_list)
            outcome_dict = {"verdict": "settled"}
            if "--gap-in" in argv_tail:
                assert argv_tail[argv_tail.index("--gap-in") + 1] == run_file.gap_inside
                with open(run_file.gap, encoding="utf-8") as handle:
                    gap_list = json.load(handle)
                is_done = [any(p in e["why"] for p in done_gap_path) for e in gap_list]
                outcome_dict.update({
                    "gap": [e for e, done in zip(gap_list, is_done) if not done],
                    "gap_done": [e for e, done in zip(gap_list, is_done) if done]})
        elif argv_tail[0] in _REFACTOR_PATCH_DICT:
            kind = argv_tail[0]
            if primary_list is not None:
                primary_list.append((kind, kwargs.get("primary_ext", "")))
            how = (refactor_dict or {}).get(kind, "settled")
            patch = _REFACTOR_PATCH_DICT[kind] if how in ("settled", "unsettled") else ""
            changed_key = "tests_changed" if kind == "test-refactor" else "non_test_changed"
            outcome_dict = {"stage": kind, "tests_changed": [], "non_test_changed": [],
                            "skipped": "nothing to refactor" if how == "skipped" else "",
                            "verdict": "unsettled" if how == "unsettled" else "settled",
                            "why": "- `run them` fails t::new" if how == "unsettled" else "",
                            changed_key: ["tests/test_a.txt" if kind == "test-refactor"
                                          else "a.txt"] if patch else []}
            status = 1 if how == "unsettled" else 0
        elif "--target-only" in argv_tail:
            kind = "list"
            outcome_dict = {"targets": {path: [] if path in no_def_path else ["f"]
                                        for path in target},
                            "tests": {path: [f"tests/test_{path}"] + share_list
                                      for path in target},
                            "baseline": [], "skipped": skipped, "first_case": 3}
            if lang is not None:
                outcome_dict["lang"] = lang
        else:
            kind = "tests"
            if primary_list is not None:
                primary_list.append(("tests", kwargs.get("primary_ext", "")))
            with open(run_file.target, encoding="utf-8") as handle:
                task_dict = json.load(handle)
            path_list = task_dict["paths"]
            assert sorted(task_dict) == ["baseline", "case_step", "first_case", "paths",
                                         "test_paths"]
            if number_list is not None:
                number_list.append((task_dict["first_case"], task_dict["case_step"]))
            assert task_dict["test_paths"] == sorted(
                [f"tests/test_{path}" for path in path_list] + share_list)
            going_list.append(run_id)
            if at_once_list is not None:
                at_once_list.append(len(going_list))
                time.sleep(0.2)
            going_list.remove(run_id)
            is_open = bool(set(path_list) & set(open_path))
            patch = "" if no_patch else _test_patch(path_list)
            outcome_dict = {"stage": "test-create", "skipped": "",
                            "verdict": "unsettled" if is_open else "settled",
                            "why": "- `run them` fails t::new" if is_open else "",
                            "cases": [{"id": "case_1"}], "uncovered": [],
                            "tests_changed": [f"tests/test_{path}" for path in path_list],
                            "non_test_changed": ["a.txt"] if is_open else [],
                            "tests_base": "red",
                            "gap": [{"what": "case_1", "why": f"{path} does not do it",
                                     "test_path": f"tests/test_gap_{path}",
                                     "test_text": "the test\n"}
                                    for path in path_list if path in gap_path]}
            status = 1 if is_open else 0
        run_list.append((kind, argv_tail, base, path_list))
        if kind not in lose_patch and not set(path_list) & set(lose_patch):
            pathlib.Path(run_file.patch).write_text(patch, encoding="utf-8")
            pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(
                json.dumps({"exit_status": status}), encoding="utf-8")
        pathlib.Path(run_file.outcome).write_text(json.dumps(outcome_dict),
                                                  encoding="utf-8")
        return "/usr/bin/true", "img", status

    monkeypatch.setattr(sandbox, "_start_container", start)
    _analysis_call_list(monkeypatch)
    return run_list


def _kind_list(run_list):
    return [run[0] for run in run_list]


def test_the_container_is_asked_to_write_the_tests_against_the_base():
    argv = sandbox.test_create_argv("repo", "a" * 40, "足して", "/out/repo/sandbox/o.json",
                                    target_in="/out/repo/sandbox/t.json")
    assert argv[0] == "test-create" and f"{sandbox._WORK}/repo" in argv
    assert argv[argv.index("--base") + 1] == "a" * 40
    assert "--request=足して" in argv
    assert argv[argv.index("--outcome-out") + 1] == "/out/repo/sandbox/o.json"
    assert argv[-2:] == ["--target-in", "/out/repo/sandbox/t.json"]
    assert sandbox.test_create_argv("repo", "a" * 40, target_only=True)[-1] == "--target-only"
    # The file on how the tests are written is named by a setting the container is
    # forwarded by name, so nothing on the command line names it
    assert "--guide" not in argv


@pytest.mark.parametrize("count, def_max, size_list", [
    (25, 10, [9, 9, 7]), (10, 10, [10]), (11, 10, [6, 5]), (3, 0, [3]), (0, 10, [])])
def test_files_that_share_no_test_are_split_into_test_tasks_of_an_even_share(
        count, def_max, size_list):
    test_by_file_dict = {f"p{n:02}.py": [f"tests/test_p{n:02}.py"]
                         for n in reversed(range(count))}
    task_list = sandbox.test_task_list(
        test_by_file_dict, dict.fromkeys(test_by_file_dict, 1), def_max)
    assert [len(task["paths"]) for task in task_list] == size_list
    assert [path for task in task_list for path in task["paths"]] == sorted(test_by_file_dict)
    for task in task_list:
        assert task["test_paths"] == [f"tests/test_{path}" for path in task["paths"]]
        assert task["chain"] == 0


def test_files_one_test_file_refers_to_are_kept_in_one_test_task():
    test_by_file_dict = {"a.py": ["t/1.py"], "b.py": [], "c.py": ["t/1.py", "t/2.py"],
                         "d.py": ["t/2.py"], "e.py": []}
    task_list = sandbox.test_task_list(
        test_by_file_dict, dict.fromkeys(test_by_file_dict, 1), 3)
    # a, c and d are chained through the two test files, and are one task of three
    assert task_list == [
        {"paths": ["a.py", "c.py", "d.py"], "test_paths": ["t/1.py", "t/2.py"], "chain": 0},
        {"paths": ["b.py", "e.py"], "test_paths": [], "chain": 0}]


def test_a_set_over_the_limit_is_split_into_the_parts_of_one_chain():
    test_by_file_dict = {"a.py": ["t/1.py"], "b.py": [], "c.py": ["t/1.py", "t/2.py"],
                         "d.py": ["t/2.py"], "e.py": []}
    task_list = sandbox.test_task_list(
        test_by_file_dict, dict.fromkeys(test_by_file_dict, 1), 2)
    # Every part keeps the test files that refer to its files, and the first part of
    # the chain and the test task of no chain come before the second part
    assert task_list == [
        {"paths": ["a.py", "c.py"], "test_paths": ["t/1.py", "t/2.py"], "chain": 1},
        {"paths": ["b.py", "e.py"], "test_paths": [], "chain": 0},
        {"paths": ["d.py"], "test_paths": ["t/2.py"], "chain": 1}]


def test_one_test_file_that_refers_to_every_file_goes_to_every_part_of_the_chain():
    test_by_file_dict = {f"p{n:02}.py": ["tests/test_all.py", f"tests/test_p{n:02}.py"]
                         for n in range(25)}
    def_count_dict = dict.fromkeys(test_by_file_dict, 1)
    task_list = sandbox.test_task_list(test_by_file_dict, def_count_dict, 10)
    assert [len(task["paths"]) for task in task_list] == [9, 9, 7]
    assert [path for task in task_list for path in task["paths"]] == sorted(test_by_file_dict)
    for task in task_list:
        assert task["chain"] == 1
        assert task["test_paths"] == sorted(
            ["tests/test_all.py"] + [f"tests/test_{path}" for path in task["paths"]])
    # With no limit the set is one test task of no chain
    assert [(len(task["paths"]), task["chain"]) for task in sandbox.test_task_list(
        test_by_file_dict, def_count_dict, 0)] == [(25, 0)]


def test_two_sets_over_the_limit_are_two_chains_and_one_file_over_it_is_none():
    test_by_file_dict = {"a.py": ["t/1.py"], "b.py": ["t/1.py"], "c.py": ["t/2.py"],
                         "d.py": ["t/2.py"], "e.py": []}
    def_count_dict = {"a.py": 2, "b.py": 2, "c.py": 2, "d.py": 2, "e.py": 5}
    task_list = sandbox.test_task_list(test_by_file_dict, def_count_dict, 3)
    assert [(task["paths"], task["chain"]) for task in task_list] == [
        (["a.py"], 1), (["c.py"], 2), (["e.py"], 0), (["b.py"], 1), (["d.py"], 2)]


def _same_tree(repo, commit, other) -> bool:
    """Say whether two commits hold the same tree, the first being a child of the second."""
    return _git(repo, "rev-parse", f"{commit}^").strip() == other \
        and _git(repo, "rev-parse", f"{commit}^{{tree}}") == \
        _git(repo, "rev-parse", f"{other}^{{tree}}")


def _in_tree(repo, *name_list) -> bool:
    """Say whether every file named is in the working tree."""
    return all((repo / name).exists() for name in name_list)


def _branch_gone(repo, branch) -> bool:
    return _git(repo, "branch", "--list", branch).strip() == ""


def test_the_setting_left_off_starts_one_container_and_lands_the_union(tmp_path,
                                                                       monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", False)
    run_list = _test_stage_container(monkeypatch)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert _kind_list(run_list) == ["merge"]
    assert result["land"]["files"] == 2 and "test_create" not in result
    assert _in_tree(repo, "a.txt", "b.txt") and _branch_gone(repo, result["branch"])
    # The tree holds the union as changes to commit; HEAD is where it was
    assert _git(repo, "rev-parse", "main").strip() == result["base"]
    assert sorted(gitcmd.status_path_list(str(repo))) == ["a.txt", "b.txt"]


def test_settled_tests_land_on_top_of_the_union_and_go_into_the_tree_with_it(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    run_list = _test_stage_container(monkeypatch)
    main_before = _git(repo, "rev-parse", "main").strip()
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True, request="add a and b")
    assert _kind_list(run_list) == ["merge", "list", "tests"]
    # Both later containers start at the union and are told what it was written against
    for _, argv, base, _ in run_list[1:]:
        assert base == result["commit"]
        assert argv[argv.index("--base") + 1] == main_before
    assert run_list[2][3] == ["a.txt", "b.txt"]
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "settled" and create_dict["tests_base"] == "red"
    assert len(create_dict["test_tasks"]) == 1 and create_dict["open_paths"] == []
    # The tests are in the tree with the union, and the branch's name is gone
    assert result["land"]["files"] == 4
    assert _in_tree(repo, "a.txt", "b.txt", "tests/test_a.txt", "tests/test_b.txt")
    assert _branch_gone(repo, result["branch"])
    head = create_dict["commit"]
    assert _git(repo, "rev-parse", f"{head}^").strip() == result["commit"]
    message = _git(repo, "log", "-1", "--format=%B", head)
    assert message.startswith("test-create: add a and b")
    assert "Codetwine-Loom-Stage: test-create" in message \
        and "Codetwine-Loom-Tests-Base: red" in message
    assert "Codetwine-Loom-Tests-Changed: tests/test_a.txt tests/test_b.txt" in message
    assert "Codetwine-Loom-Run-Request: add a and b" in message \
        and "Codetwine-Loom-Test-Open" not in message


def test_a_merge_that_was_not_settled_starts_no_second_container(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    run_list = _test_stage_container(monkeypatch, merge_status=1)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert _kind_list(run_list) == ["merge"]
    assert "test_create" not in result and result["land"] is None
    assert not _in_tree(repo, "a.txt") and not _branch_gone(repo, result["branch"])


def test_a_merge_that_was_not_settled_names_the_branch_it_left(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _test_stage_container(monkeypatch, merge_status=1)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert result["land_why"].startswith("the merge was not settled (exit status 1)")
    assert f"What it left is on {result['branch']}" in result["land_why"]
    assert not _branch_gone(repo, result["branch"])


@pytest.mark.parametrize("patch", [None, ""])
def test_a_merge_whose_container_left_no_patch_names_no_branch(tmp_path, monkeypatch,
                                                              caplog, patch):
    """A container stopped at its time limit, or killed, writes none."""
    repo, out = _two_branch_repo(tmp_path)

    def start(*arg_list, **kwargs):
        if patch is not None:
            os.makedirs(os.path.dirname(arg_list[8].patch), exist_ok=True)
            pathlib.Path(arg_list[8].patch).write_text(patch, encoding="utf-8")
        return "/usr/bin/true", "img", 137

    monkeypatch.setattr(sandbox, "_start_container", start)
    _analysis_call_list(monkeypatch)
    before = _git(repo, "branch", "--list")
    with caplog.at_level(logging.WARNING):
        result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                                   land=True)
    assert result["commit"] is None and result["branch"] is None
    assert result["land_why"] == (
        f"the merging run left no patch that is taken in, so there is no union and "
        f"nothing was put into the working tree, and the branches it was to combine are "
        f"as they were: the container ended (exit status 137, the status of a container "
        f"killed when its processes reach SANDBOX_MEMORY ({config.SANDBOX_MEMORY})) "
        f"before it wrote a patch and the summary beside it")
    assert result["land_why"] in caplog.text and "codetwine-loom/sandbox/" not in result["land_why"]
    assert _git(repo, "branch", "--list") == before


def test_a_merge_whose_container_wrote_no_patch_is_unsettled_whatever_it_said(
        tmp_path, monkeypatch):
    """The outcome says settled and the exit status is 0, and nothing was written."""
    repo, out = _two_branch_repo(tmp_path)
    run_list = _test_stage_container(monkeypatch, lose_patch=("merge",))
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert _kind_list(run_list) == ["merge"]
    assert result["verdict"] == "unsettled" and result["exit_status"] == 1
    assert result["commit"] is None and result["branch"] is None and result["no_patch"]
    assert sandbox.union_status(result) == 1
    assert result["land_why"].startswith(
        "the merging run left no patch that is taken in")
    assert result["land_why"].endswith(
        ": the container ended (exit status 0) before it wrote a patch and the summary "
        "beside it")


@pytest.mark.parametrize("stage", ["refactor", "test-refactor"])
def test_a_stage_whose_container_wrote_no_patch_is_unsettled_whatever_it_said(
        tmp_path, monkeypatch, stage):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "REFACTOR", True)
    monkeypatch.setattr(config, "TEST_REFACTOR", True)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 0)
    _test_stage_container(monkeypatch, lose_patch=(stage,))
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    stage_dict = result[integrate_mod.stage_key(stage)]
    assert stage_dict["verdict"] == "unsettled" and stage_dict["exit_status"] == 1
    assert stage_dict["commit"] is None and stage_dict["patch"] is None
    assert stage_dict["why"] == (
        "the container left no patch that is taken in, so nothing of the stage was taken "
        "in: the container ended (exit status 0) before it wrote a patch and the summary "
        "beside it")
    assert sandbox.open_stage_key(result) == integrate_mod.stage_key(stage)
    # No commit of the stage is on the branch, so a run carrying on starts with it
    assert sandbox.stage_resume_dict(str(repo), result["branch"])["stage"] == stage
    assert sandbox.resume_branch_list([], result, 0) == [result["branch"]]


def test_a_lone_test_task_whose_container_wrote_no_patch_is_unsettled(tmp_path,
                                                                     monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "REFACTOR", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    run_list = _test_stage_container(monkeypatch, lose_patch=("tests",))
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert _kind_list(run_list) == ["merge", "list", "tests"]
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "unsettled" and create_dict["commit"] is None
    assert "left no patch that is taken in" in create_dict["why"]
    assert _git(repo, "rev-parse", result["branch"]).strip() == result["commit"]


def test_a_test_task_whose_container_wrote_no_patch_is_left_open_among_the_others(
        tmp_path, monkeypatch):
    """Its files are named as still open on the commit the other test tasks land in."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    _test_stage_container(monkeypatch, lose_patch=("b.txt",))
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "unsettled" and create_dict["open_paths"] == ["b.txt"]
    assert [(e["paths"], e["verdict"], e["exit_status"], e["commit"] is None)
            for e in create_dict["test_tasks"]] == [
        (["a.txt"], "settled", 0, False), (["b.txt"], "unsettled", 1, True)]
    message = _git(repo, "log", "-1", "--format=%B", create_dict["commit"])
    assert "Codetwine-Loom-Verdict: unsettled" in message \
        and 'Codetwine-Loom-Test-Open: ["b.txt"]' in message
    assert sandbox.open_stage_key(result) == "test_create"


def test_test_tasks_whose_combining_run_wrote_no_patch_name_no_branch(tmp_path,
                                                                     monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    _test_stage_container(monkeypatch, lose_patch=("combine",))
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "unsettled" and create_dict["commit"] is None
    assert create_dict["why"] == (
        "the run combining the test tasks' tests left no patch that is taken in, so they "
        "were not combined: the container ended (exit status 0) before it wrote a patch "
        "and the summary beside it")


@pytest.mark.parametrize("status, what", [
    (0, "changed nothing"), (1, "ended with exit status 1 and changed nothing"),
    (137, "ended with exit status 137, the status of a container killed when its "
          "processes reach SANDBOX_MEMORY (2g), and changed nothing")])
def test_test_tasks_whose_combining_run_changed_nothing_say_so_and_no_exit_status_of_0(
        tmp_path, monkeypatch, status, what):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "SANDBOX_MEMORY", "2g")
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    run_list = _test_stage_container(monkeypatch)
    start = sandbox._start_container

    def empty_combine(*a, **kwargs):
        engine_path, image_name, own_status = start(*a, **kwargs)
        if run_list[-1][0] == "combine":
            pathlib.Path(a[8].patch).write_text("", encoding="utf-8")
            pathlib.Path(sandbox.meta_path(a[8].patch)).write_text(
                json.dumps({"exit_status": status}), encoding="utf-8")
        return engine_path, image_name, own_status

    monkeypatch.setattr(sandbox, "_start_container", empty_combine)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert "combine" in _kind_list(run_list)
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "unsettled" and create_dict["commit"] is None
    assert create_dict["why"] == (
        f"the run combining the test tasks' tests {what}, so they were not combined")


@pytest.mark.parametrize("status, what", [
    (1, "exit status 1"),
    (137, "exit status 137, the status of a container killed when its processes reach "
          "SANDBOX_MEMORY (2g)"),
    (124, "exit status 124, the status of a container taken away at SANDBOX_TIMEOUT_SEC "
          "(3600 s)")])
def test_why_a_run_was_not_taken_in_says_what_its_exit_status_is(monkeypatch, status,
                                                                    what):
    """A summary beside the patch that names 137 or 124 is said as one that the
    container ended with is."""
    monkeypatch.setattr(config, "SANDBOX_MEMORY", "2g")
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SEC", 3600)
    and_what = what if status == 1 else f"{what},"
    assert sandbox._empty_patch_text(status) == f"ended with {and_what} and changed nothing"
    stage = integrate_mod.REFACTOR_STAGE
    assert sandbox._stage_record_dict(stage, {}, status)["why"] == (
        f"the run ended with {and_what} and left no account")
    land_dict = sandbox._land_run("/repo", "b" * 40, {
        "branch": "codetwine-loom/sandbox/r", "commit": "c" * 40, "exit_status": status},
        "/out/land.diff", "/out/last.diff")
    assert land_dict["land"] is None and land_dict["land_why"] == (
        f"the run ended with {what}, so what it changed stays on codetwine-loom/sandbox/r "
        f"({'c' * gitcmd.SHORT_COMMIT_CHAR}) and the working tree is as it was. Read it "
        f"with:")


@pytest.mark.parametrize("max_definition", [10, 1])
def test_the_cases_a_union_carried_on_from_recorded_are_done_once_the_tests_are_written_again(
        tmp_path, monkeypatch, max_definition):
    """The union an earlier merge left is a parent of the new one, so its cases are in
    the new one's range: the commit of the tests written over the new union records
    them as done, and what is counted from then on is what those tests found."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", max_definition)
    _test_stage_container(monkeypatch, gap_path=["b.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              request="add a and b")
    gap_list = [{"what": "case_1", "why": "b.txt does not do it"}]
    assert first["gap"] == gap_list
    _git(repo, "branch", "codetwine-loom/c", "main")
    _test_stage_container(monkeypatch)
    second = sandbox.integrate(str(repo), str(out), ["codetwine-loom/c"], resume=first["branch"],
                               request="add a and b")
    assert second["test_create"]["verdict"] == "settled" and second["gap"] == []
    message = _git(repo, "log", "-1", "--format=%B", second["test_create"]["commit"])
    assert 'Codetwine-Loom-Gap-Done: [{"what":"case_1","why":"b.txt does not do it"}]' in message
    # The earlier union's own branch still says what its tests found
    assert integrate_mod.gap_list_of(integrate_mod.branch_record_dict(
        str(repo), first["branch"])["told"]) == gap_list


def test_the_cases_a_union_carried_on_from_recorded_stay_while_no_test_is_written(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch, gap_path=["b.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    _git(repo, "branch", "codetwine-loom/c", "main")
    # The stage has nothing to write tests for, or is left unsettled
    for stage_kwargs in ({"skipped": "the change touched no definition"},
                         {"open_path": ["a.txt"]}):
        _test_stage_container(monkeypatch, **stage_kwargs)
        monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
        second = sandbox.integrate(str(repo), str(out), ["codetwine-loom/c"],
                                   resume=first["branch"])
        assert second["gap"] == [{"what": "case_1", "why": "b.txt does not do it"}]
        _git(repo, "branch", "-D", second["branch"])


def _merge_over_tree_container(monkeypatch, unsettle_count=0, extra_patch=""):
    """Play a merge written against the working tree's commit over the stand-in there is.

    The patch over that commit adds what each branch passed with --branch adds to main,
    and nothing when none is passed, followed by extra_patch. The first unsettle_count
    merges are left unsettled.

    Returns:
        [the commit of the working tree each such merge was written against, ...].
    """
    inner = sandbox._start_container
    onto_list = []

    def start(*arg_tuple, **kwargs):
        project_dir, run_file, argv_tail = arg_tuple[2], arg_tuple[8], arg_tuple[10]
        if argv_tail[0] != "integrate" or "--onto-commit" not in argv_tail \
                or "-test/" in " ".join(argv_tail):
            return inner(*arg_tuple, **kwargs)
        onto_list.append(argv_tail[argv_tail.index("--onto-commit") + 1])
        patch = "".join(_git(project_dir, "diff", "--binary", "main", argv_tail[i + 1])
                        for i, word in enumerate(argv_tail) if word == "--branch")
        patch += extra_patch
        status = 1 if len(onto_list) <= unsettle_count else 0
        os.makedirs(os.path.dirname(run_file.patch), exist_ok=True)
        pathlib.Path(run_file.patch).write_text(patch, encoding="utf-8")
        pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(
            json.dumps({"exit_status": status}), encoding="utf-8")
        pathlib.Path(run_file.outcome).write_text(json.dumps(
            {"verdict": "unsettled", "why": "- conflict markers are still in: c.txt"}
            if status else {"verdict": "settled"}), encoding="utf-8")
        return "/usr/bin/true", "img", status

    monkeypatch.setattr(sandbox, "_start_container", start)
    return onto_list


def _gap_union_landed(tmp_path, monkeypatch):
    """Put a union whose tests found case_1 into the working tree, and return it."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    _test_stage_container(monkeypatch, gap_path=["a.txt"])
    first = sandbox.integrate_into_tree(str(repo), str(out),
                                        ["codetwine-loom/a", "codetwine-loom/b"])
    assert first["land"]["files"] == 4 and first["gap"] == [_CASE_1]
    return repo, out, first


_CASE_1 = {"what": "case_1", "why": "a.txt does not do it"}


@pytest.mark.parametrize("stage_kwargs", [
    None, {"skipped": "the change touched no definition"}, {"open_path": ["c.txt"]}])
def test_the_cases_of_a_union_the_working_tree_holds_stay_over_one_carried_on_from_it(
        tmp_path, monkeypatch, stage_kwargs):
    """The union is merged again with a branch, written against the working tree's
    commit, which holds it: while no test is written over the new union, the new union
    counts its cases, and so does what its branch says of itself."""
    repo, out, first = _gap_union_landed(tmp_path, monkeypatch)
    _git(repo, "checkout", "--quiet", "-b", "codetwine-loom/c", "main")
    (repo / "c.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "c.txt")
    _git(repo, "commit", "--quiet", "-m", "codetwine-loom/c")
    _git(repo, "checkout", "--quiet", "main")
    monkeypatch.setattr(config, "TEST_CREATE", stage_kwargs is not None)
    _test_stage_container(monkeypatch, target=("c.txt",), **(stage_kwargs or {}))
    onto_list = _merge_over_tree_container(monkeypatch)
    second = sandbox.integrate_into_tree(str(repo), str(out), ["codetwine-loom/c"],
                                         resume=first["branch"])
    assert len(onto_list) == 1 and second["land"]["files"] == 1
    assert second["gap"] == [_CASE_1] and sandbox.union_status(second) == 1
    assert integrate_mod.gap_list_of(integrate_mod.branch_record_dict(
        str(repo), second["branch"])["told"]) == [_CASE_1]
    assert not _branch_gone(repo, second["branch"])
    if stage_kwargs != {"open_path": ["c.txt"]}:
        return
    # The stage carried on from alone counts them until it writes the tests
    for stage_kwargs, gap_list in (({"open_path": ["c.txt"]}, [_CASE_1]), ({}, [])):
        _test_stage_container(monkeypatch, target=("c.txt",), **stage_kwargs)
        third = sandbox.integrate_into_tree(str(repo), str(out), [],
                                            resume=second["branch"])
        assert third["gap"] == gap_list
    assert _branch_gone(repo, second["branch"])


def _c_branch(repo):
    """Make codetwine-loom/c off main, adding c.txt."""
    _git(repo, "checkout", "--quiet", "-b", "codetwine-loom/c", "main")
    (repo / "c.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "c.txt")
    _git(repo, "commit", "--quiet", "-m", "codetwine-loom/c")
    _git(repo, "checkout", "--quiet", "main")


def _write_stand_for_record(repo, out, union):
    """Write the record a command that left union leaves, with the unions it held."""
    sandbox.runrecord.write(sandbox.record_dir(str(out), "repo"), {
        "id": "20260101-000000-1-1", "resume": [union],
        "result": {sandbox.runrecord.STAND_FOR_KEY: sandbox.stand_for_list(str(repo))}})


@pytest.mark.parametrize("branch_list", [[], ["codetwine-loom/e"]])
def test_the_cases_of_a_union_the_working_tree_holds_stay_over_an_unsettled_union(
        tmp_path, monkeypatch, branch_list):
    """The union made over the working tree holding a union with cases is left
    unsettled; carrying on from it - alone, or with a branch that adds nothing - makes a
    union that stands on the same union, counts its cases, and puts that union's branch
    away once it goes in."""
    repo, out, first = _gap_union_landed(tmp_path, monkeypatch)
    _c_branch(repo)
    _git(repo, "branch", "codetwine-loom/e", "main")
    monkeypatch.setattr(config, "TEST_CREATE", False)
    _merge_over_tree_container(monkeypatch, unsettle_count=1)
    second = sandbox.integrate_into_tree(str(repo), str(out), ["codetwine-loom/c"],
                                         resume=first["branch"])
    assert second["exit_status"] and not second["land"]
    held_list = sandbox.stand_for_list(str(repo))
    _write_stand_for_record(repo, out, second["branch"])
    sandbox.reset_land_list()
    third = sandbox.integrate_into_tree(str(repo), str(out), branch_list,
                                        resume=second["branch"])
    assert third["gap"] == [_CASE_1] and sandbox.union_status(third) == 1
    assert third["land"] is not None and sandbox.is_gap_union(str(repo), third["branch"])
    assert held_list == [first["branch"]]
    assert _branch_gone(repo, first["branch"]) and _branch_gone(repo, second["branch"])


def test_the_union_carried_on_from_goes_once_the_union_over_it_goes_in_after_a_stop(
        tmp_path, monkeypatch):
    """The command that made a union over the working tree holding a union with cases
    is stopped before the new union goes in: it holds the union carried on from, and the
    command carrying on from the new union takes that union away once it puts it in."""
    repo, out, first = _gap_union_landed(tmp_path, monkeypatch)
    _c_branch(repo)
    _test_stage_container(monkeypatch, target=("c.txt",))
    _merge_over_tree_container(monkeypatch)
    inner = sandbox._start_container

    def stop(*arg_tuple, **kwargs):
        if "--target-only" in arg_tuple[10]:
            raise KeyboardInterrupt("the run was stopped (SIGINT)")
        return inner(*arg_tuple, **kwargs)

    monkeypatch.setattr(sandbox, "_start_container", stop)
    with pytest.raises(KeyboardInterrupt):
        sandbox.integrate_into_tree(str(repo), str(out), ["codetwine-loom/c"],
                                    resume=first["branch"])
    union = next(branch for branch in _git(
        repo, "for-each-ref", "--format=%(refname:short)",
        "refs/heads/codetwine-loom/sandbox").split()
        if branch != first["branch"] and "-test" not in branch)
    assert sandbox.stand_for_list(str(repo)) == [first["branch"]]
    assert not _branch_gone(repo, first["branch"])
    _write_stand_for_record(repo, out, union)
    sandbox.reset_land_list()
    monkeypatch.setattr(sandbox, "_start_container", inner)
    second = sandbox.integrate_into_tree(str(repo), str(out), [], resume=union)
    assert second["land"]["files"] and second["land"]["stand_drop"] == [first["branch"]]
    assert _branch_gone(repo, first["branch"])


# A line added after the one the union with cases put into a.txt
_A_LINE_PATCH = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1,2 @@\n x\n+y\n"


@pytest.mark.parametrize("change", ["merge", "refactor"])
def test_the_cases_of_a_union_stay_over_one_that_changes_a_file_it_added(
        tmp_path, monkeypatch, change):
    """The union made over the working tree holding a union with cases changes a file
    that union added, in its merge or in the stage that refactors after it: once it goes
    in, the working tree holds both, and the cases counted at the end of the run are the
    ones its branch is carried on from for."""
    repo, out, first = _gap_union_landed(tmp_path, monkeypatch)
    _c_branch(repo)
    monkeypatch.setattr(config, "TEST_CREATE", False)
    monkeypatch.setattr(config, "REFACTOR", change == "refactor")
    monkeypatch.setattr(config, "TEST_REFACTOR", False)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 0)
    _test_stage_container(monkeypatch, target=("c.txt",))
    _merge_over_tree_container(monkeypatch,
                               extra_patch=_A_LINE_PATCH if change == "merge" else "")
    second = sandbox.integrate_into_tree(str(repo), str(out), ["codetwine-loom/c"],
                                         resume=first["branch"])
    assert (repo / "a.txt").read_text(encoding="utf-8") == (
        "x\ny\n" if change == "merge" else "y\n")
    assert second["land"]["files"] == 2
    assert second["gap"] == [_CASE_1] and sandbox.union_status(second) == 1
    record_dict = integrate_mod.branch_record_dict(str(repo), second["branch"])
    assert [part_dict["held"] for part_dict in record_dict["earlier_list"]] == [True]
    assert record_dict["held"] and record_dict["gap"] == [_CASE_1]
    assert sandbox.open_branch_list(str(repo), [second["branch"]]) == [second["branch"]]
    assert _branch_gone(repo, first["branch"])


def test_a_union_carried_on_from_stands_on_none_the_working_tree_no_longer_holds(
        tmp_path, monkeypatch):
    """The union left unsettled over the working tree holding a union with cases is
    carried on from once that union's files are out of the working tree: the new
    union's merge records no union it stands on, and no case counts before it goes in
    or after."""
    repo, out, first = _gap_union_landed(tmp_path, monkeypatch)
    _c_branch(repo)
    monkeypatch.setattr(config, "TEST_CREATE", False)
    _merge_over_tree_container(monkeypatch, unsettle_count=1)
    second = sandbox.integrate_into_tree(str(repo), str(out), ["codetwine-loom/c"],
                                         resume=first["branch"])
    assert second["exit_status"] and not second["land"]
    assert integrate_mod.stand_on_of(str(repo), second["commit"]) == \
        integrate_mod.commit_of(str(repo), first["branch"])
    for name in ("a.txt", "b.txt"):
        (repo / name).unlink()
    shutil.rmtree(repo / "tests")
    sandbox.reset_land_list()
    third = sandbox.integrate_into_tree(str(repo), str(out), ["codetwine-loom/c"],
                                        resume=second["branch"])
    assert third["land"]["files"] == 1 and (repo / "c.txt").is_file()
    assert third["gap"] == [] and sandbox.union_status(third) == 0
    assert integrate_mod.stand_on_of(str(repo), third["commit"]) == ""


def test_a_union_whose_test_file_the_user_changed_is_merged_again_with_its_cases(
        tmp_path, monkeypatch):
    """The working tree holds the merge and a file the stage above it added, changed:
    the union is merged again with the working tree, and a merge that adds nothing to it
    is a union of its own, which reaches the earlier one and counts its cases."""
    repo, out, first = _gap_union_landed(tmp_path, monkeypatch)
    (repo / "tests" / "test_a.txt").write_text("t\nmine\n", encoding="utf-8")
    onto_list = _merge_over_tree_container(monkeypatch)
    second = sandbox.integrate_into_tree(str(repo), str(out), [], resume=first["branch"])
    assert len(onto_list) == 1 and second["commit"] and second["land"]["files"] == 0
    assert second["gap"] == [_CASE_1]
    assert _branch_gone(repo, first["branch"]) and not _branch_gone(repo, second["branch"])
    assert sandbox.is_gap_union(str(repo), second["branch"])
    assert (repo / "tests" / "test_a.txt").read_text(encoding="utf-8") == "t\nmine\n"


def test_the_union_a_merge_by_hand_was_carried_on_from_goes_once_that_merge_is_in(
        tmp_path):
    repo = _repo(tmp_path / "repo")
    earlier = "codetwine-loom/sandbox/20260101-000000-1-42"
    _git(repo, "branch", earlier)
    _git(repo, "branch", "mine")
    tip = _git(repo, "rev-parse", "HEAD").strip()
    result_dict = {"commit": tip, "land": {"files": 1}, "gap": []}
    for resume, result in (
            ("mine", result_dict), (earlier, {**result_dict, "land": None}),
            ("", result_dict)):
        sandbox._drop_resume_union(str(repo), str(tmp_path / "out"), "repo", "", resume,
                                   result)
        assert not _branch_gone(repo, earlier) and not _branch_gone(repo, "mine")
    # The new union is in the working tree and reaches it, whatever is left on the new
    # union's own branch
    for more_dict in ({"gap": [{"what": "c", "why": "w"}]},
                      {"refactor": {"verdict": "unsettled", "skipped": ""}}, {}):
        _git(repo, "branch", "--force", earlier, tip)
        sandbox._drop_resume_union(str(repo), str(tmp_path / "out"), "repo", "", earlier,
                                   {**result_dict, **more_dict})
        assert _branch_gone(repo, earlier) and not _branch_gone(repo, "mine")
    # The new union was kept out of the working tree for the paths it touches: it is
    # kept, and holds what the union carried on from held
    refuse_dict = {**result_dict, "land": None, "land_refuse": True}
    for resume in (earlier, "mine"):
        _git(repo, "branch", "--force", resume, tip)
        sandbox._drop_resume_union(str(repo), str(tmp_path / "out"), "repo", "", resume,
                                   refuse_dict)
    assert _branch_gone(repo, earlier) and not _branch_gone(repo, "mine")


def test_the_union_a_merge_that_changed_nothing_was_carried_on_from_goes_when_the_tree_holds_it(
        tmp_path, monkeypatch):
    """The merge carried on from the union ended with exit status 0 and left no
    commit. The union goes when it holds a result the working tree holds, and stays
    when the working tree does not hold it, when its name is not one this tool made
    up, and when the merge did not get through."""
    monkeypatch.setattr(sandbox, "live_run_list", lambda *a: [])
    repo = _repo(tmp_path / "repo")
    earlier = "codetwine-loom/sandbox/20260101-000000-1-42"
    _git(repo, "checkout", "--quiet", "-b", earlier)
    (repo / "added.txt").write_text("what the union added\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "integrate: x\n\nx\n\nCodetwine-Loom-Verdict: settled\n")
    _git(repo, "checkout", "--quiet", "main")
    _git(repo, "branch", "mine", earlier)
    result_dict = {"commit": None, "land": None, "land_why": "", "exit_status": 0}

    def drop(resume, result):
        sandbox._drop_resume_union(str(repo), str(tmp_path / "out"), "repo", "", resume,
                                   result)

    drop(earlier, result_dict)
    assert not _branch_gone(repo, earlier)
    (repo / "added.txt").write_text("what the union added\n", encoding="utf-8")
    for resume, result in (("mine", result_dict), (earlier, {**result_dict, "exit_status": 1}),
                           (earlier, {**result_dict, "land_why": "it does not apply"}),
                           (earlier, {**result_dict, "commit": "c" * 40})):
        drop(resume, result)
        assert not _branch_gone(repo, earlier) and not _branch_gone(repo, "mine")
    drop(earlier, result_dict)
    assert _branch_gone(repo, earlier) and not _branch_gone(repo, "mine")


def test_a_gap_the_tests_found_is_put_in_kept_on_the_branch_and_makes_the_run_fail(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch, gap_path=["b.txt"])
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True, request="add a and b")
    assert result["gap"] == [{"what": "case_1", "why": "b.txt does not do it"}]
    assert result["test_create"]["verdict"] == "settled"
    # The commit carries the case alone, not the test it was recorded with
    assert "the test" not in _git(repo, "log", "-1", "--format=%B",
                                  result["test_create"]["commit"])
    assert result["land"]["files"] == 4 and _in_tree(repo, "b.txt", "tests/test_b.txt")
    assert result["exit_status"] == 0 and sandbox.union_status(result) == 1
    # The branch stays for the run carrying on from it, which starts a new split
    assert _git(repo, "rev-parse", result["branch"]).strip() == \
        result["test_create"]["commit"]
    assert sandbox.resume_branch_list([], result, 1) == [result["branch"]]
    assert sandbox.stage_resume_dict(str(repo), result["branch"]) == {}
    told = integrate_mod.branch_record_dict(str(repo), result["branch"])["told"]
    assert any(integrate_mod.decision_list_of(t["gap"]) for t in told)
    assert any(t["verdict"] == "settled" and not t["stage"] for t in told)


@pytest.mark.parametrize("lang, expect", [("java", "java"), ("", ""), ("nope", ""),
                                          (None, "")])
def test_the_tests_are_written_and_combined_in_the_language_the_listing_took(
        tmp_path, monkeypatch, lang, expect):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "REFACTOR", True)
    monkeypatch.setattr(config, "TEST_REFACTOR", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    primary_list = []
    _test_stage_container(monkeypatch, lang=lang, primary_list=primary_list)
    sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                      land=True)
    assert sorted(primary_list) == [("combine", expect), ("refactor", expect),
                                    ("test-refactor", expect), ("tests", expect),
                                    ("tests", expect)]


def test_a_container_is_handed_the_language_to_take_as_the_primary_one(monkeypatch):
    assert f"{sandbox.CTLOOM_PRIMARY_EXT_ENV}=java" in _argv(primary_ext="java")
    assert sandbox.CTLOOM_PRIMARY_EXT_ENV not in " ".join(_argv())
    monkeypatch.setenv(sandbox.CTLOOM_PRIMARY_EXT_ENV, "java")
    assert sandbox.primary_ext_of_env() == "java"
    monkeypatch.setenv(sandbox.CTLOOM_PRIMARY_EXT_ENV, "nope")
    assert sandbox.primary_ext_of_env() == ""
    monkeypatch.delenv(sandbox.CTLOOM_PRIMARY_EXT_ENV)
    assert sandbox.primary_ext_of_env() == ""


@pytest.mark.parametrize("done_gap_path, gap_left", [
    (["b.txt"], []), ([], [{"what": "case_1", "why": "b.txt does not do it"}])])
def test_a_gap_another_test_task_did_is_left_out_once_the_tests_are_combined(
        tmp_path, monkeypatch, caplog, done_gap_path, gap_left):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, gap_path=["b.txt"], done_gap_path=done_gap_path)
    with caplog.at_level(logging.INFO, logger="codetwine_loom.sandbox"):
        result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                                   land=True, request="add a and b")
    assert len(result["test_create"]["test_tasks"]) == 2
    # The test task's own commit still names the case; the combining commit says what
    # is left, and that is what the run counts
    assert result["gap"] == gap_left
    assert [{"what": e["what"], "why": e["why"]}
            for e in result["test_create"]["gap"]] == gap_left
    assert sandbox.union_status(result) == (1 if gap_left else 0)
    assert ("case_1 is done over the combined tests" in caplog.text) == (not gap_left)


def test_no_gap_leaves_the_run_passing_and_the_branch_gone(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert result["gap"] == [] and sandbox.union_status(result) == 0
    assert _branch_gone(repo, result["branch"])


def test_a_gap_of_a_test_task_that_wrote_no_file_is_kept_on_the_branch_all_the_same(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch, gap_path=["b.txt"], no_patch=True)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True, request="add a and b")
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "settled" and create_dict["patch"] is None
    # The stage's commit changes nothing and carries what the stage recorded
    assert _same_tree(repo, create_dict["commit"], result["commit"])
    message = _git(repo, "log", "-1", "--format=%B", create_dict["commit"])
    assert "Codetwine-Loom-Stage: test-create" in message \
        and "Codetwine-Loom-Tests-Base: red" in message
    assert 'Codetwine-Loom-Gap: [{"what":"case_1","why":"b.txt does not do it"}]' in message
    assert result["gap"] == [{"what": "case_1", "why": "b.txt does not do it"}]
    assert sandbox.union_status(result) == 1
    # The union is in the tree and its branch stays for the run carrying on from it
    assert result["land"]["files"] == 2
    assert _git(repo, "rev-parse", result["branch"]).strip() == create_dict["commit"]
    assert sandbox.resume_branch_list([], result, 1) == [result["branch"]]
    block = sandbox.dispatch_mod.render_earlier(
        [integrate_mod.branch_record_dict(str(repo), result["branch"])])
    assert "not done: case_1 — b.txt does not do it" in block


def test_the_gaps_of_test_tasks_that_left_no_branch_land_as_one_commit_of_the_stage(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    run_list = _test_stage_container(monkeypatch, gap_path=["b.txt"], no_patch=True)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True, request="add a and b")
    # No branch holds anything, so no combining run is started
    assert sorted(_kind_list(run_list)) == ["list", "merge", "tests", "tests"]
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "settled" and len(create_dict["test_tasks"]) == 2
    assert _same_tree(repo, create_dict["commit"], result["commit"])
    message = _git(repo, "log", "-1", "--format=%B", create_dict["commit"])
    assert message.startswith("test-create: add a and b")
    assert "Codetwine-Loom-Stage: test-create" in message \
        and "Codetwine-Loom-Verdict: settled" in message
    assert "Codetwine-Loom-Gap: " in message and "Codetwine-Loom-Test-Open" not in message
    assert result["gap"] == [{"what": "case_1", "why": "b.txt does not do it"}]
    assert sandbox.union_status(result) == 1 and not _branch_gone(repo, result["branch"])


def test_test_tasks_that_left_no_branch_name_the_open_files_on_the_commit_of_the_stage(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, open_path=["b.txt"], gap_path=["a.txt"],
                          no_patch=True)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "unsettled" and create_dict["open_paths"] == ["b.txt"]
    tip = _git(repo, "rev-parse", result["branch"]).strip()
    assert tip == create_dict["commit"] and _same_tree(repo, tip, result["commit"])
    message = _git(repo, "log", "-1", "--format=%B", tip)
    assert "Codetwine-Loom-Verdict: unsettled" in message \
        and 'Codetwine-Loom-Test-Open: ["b.txt"]' in message
    assert "Codetwine-Loom-Gap: " in message
    assert sandbox.stage_resume_dict(str(repo), result["branch"]) == {
        "stage": "test-create", "union": tip, "settled": result["commit"],
        "merge": result["commit"],
        "base": _git(repo, "rev-parse", f"{result['commit']}^").strip(), "paths": ["b.txt"]}


def test_test_tasks_all_left_open_with_no_branch_land_nothing(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, open_path=["a.txt", "b.txt"], no_patch=True)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    assert result["test_create"]["commit"] is None
    assert _git(repo, "rev-parse", result["branch"]).strip() == result["commit"]


def test_tests_that_were_not_settled_stay_on_the_branch_and_the_union_is_put_in(
        tmp_path, monkeypatch, caplog):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch, open_path=["a.txt"])
    with caplog.at_level(logging.WARNING):
        result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                                   land=True)
    create_dict = result["test_create"]
    assert sandbox.open_stage_key(result)
    # The union alone is in the tree: the tests nobody settled are not
    assert result["land"]["files"] == 2 and _in_tree(repo, "a.txt", "b.txt")
    assert not _in_tree(repo, "tests/test_a.txt")
    # The name this tool made up is kept, its tip being the tests nobody settled
    assert _git(repo, "rev-parse", result["branch"]).strip() == create_dict["commit"]
    assert "Codetwine-Loom-Non-Test-Changed: a.txt" in _git(repo, "log", "-1", "--format=%B",
                                                 create_dict["commit"])
    assert "were not settled" in caplog.text and result["branch"] in caplog.text
    assert result["exit_status"] == 0
    assert sandbox.resume_branch_list([], result, 0) == [result["branch"]]
    assert sandbox.stage_resume_dict(str(repo), result["branch"]) == {
        "stage": "test-create", "union": result["commit"], "settled": result["commit"],
        "merge": result["commit"],
        "base": _git(repo, "rev-parse", f"{result['commit']}^").strip(), "paths": []}


def test_nothing_to_write_tests_for_starts_no_writing_container(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    run_list = _test_stage_container(monkeypatch, skipped="the repository holds no test file")
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert _kind_list(run_list) == ["merge", "list"]
    create_dict = result["test_create"]
    assert create_dict["skipped"] == "the repository holds no test file"
    assert not sandbox.open_stage_key(result)
    # The stage is on the branch as a commit that changes nothing
    assert create_dict["patch"] is None and _same_tree(repo, create_dict["commit"],
                                                       result["commit"])
    message = _git(repo, "log", "-1", "--format=%B", create_dict["commit"])
    assert "Codetwine-Loom-Stage: test-create" in message \
        and "Codetwine-Loom-Verdict: settled" in message
    assert result["settle_commit"] == create_dict["commit"]
    assert result["land"]["files"] == 2 and _branch_gone(repo, result["branch"])
    assert sandbox.resume_branch_list([], result, 0) == []


@pytest.mark.parametrize("no_def_path, kind_list", [
    (("a.txt", "b.txt"), ["merge", "list"]), (("a.txt",), ["merge", "list", "tests"])])
def test_a_file_the_listing_names_no_definition_of_gets_no_test_task(
        tmp_path, monkeypatch, no_def_path, kind_list):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    run_list = _test_stage_container(monkeypatch, no_def_path=no_def_path)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert _kind_list(run_list) == kind_list
    assert not sandbox.open_stage_key(result) and result["land"]["files"] >= 2
    if len(no_def_path) == 2:
        assert result["test_create"]["skipped"] \
            == "none of the files left to write tests for was changed"
    else:
        assert run_list[-1][3] == ["b.txt"]


@pytest.mark.parametrize("test_by_file_dict, def_count_dict, def_max", [
    ({"a.py": []}, {"a.py": 0}, 8),
    ({"a.py": ["t.py"], "b.py": ["t.py"]}, {"a.py": 0, "b.py": 0}, 8),
    ({"a.py": [], "b.py": []}, {"a.py": 0, "b.py": 3}, 8), ({"a.py": []}, {"a.py": 0}, 0)])
def test_files_that_hold_no_definition_between_them_are_one_test_task(
        test_by_file_dict, def_count_dict, def_max):
    task_list = sandbox.test_task_list(test_by_file_dict, def_count_dict, def_max)
    assert [task["paths"] for task in task_list] == [sorted(test_by_file_dict)]


def test_which_branches_the_tests_are_written_over_again_is_read_off_the_tip(tmp_path,
                                                                            monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    # Settled tests leave nothing to write again; a branch no merging run left has none
    assert sandbox.stage_resume_dict(str(repo), result["branch"]) == {}
    assert sandbox.stage_resume_dict(str(repo), "codetwine-loom/a") == {}
    _git(repo, "branch", "codetwine-loom/only-union", result["commit"])
    assert sandbox.stage_resume_dict(str(repo), "codetwine-loom/only-union")["union"] == \
        result["commit"]


def test_carrying_on_writes_the_tests_again_over_the_same_union(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch, open_path=["a.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    run_list = _test_stage_container(monkeypatch)
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True)
    assert _kind_list(run_list) == ["list", "tests"]
    assert [run[2] for run in run_list] == [first["commit"]] * 2
    assert again["commit"] == first["commit"]
    tip = again["test_create"]["commit"]
    assert tip != first["test_create"]["commit"]
    assert _git(repo, "rev-parse", f"{tip}^").strip() == first["commit"]
    # The first run put nothing in, so the merge goes in along with the tests rather
    # than the tests alone
    assert again["land"]["files"] == 4
    assert _in_tree(repo, "a.txt", "b.txt", "tests/test_a.txt", "tests/test_b.txt")
    assert _branch_gone(repo, first["branch"])


def test_carrying_on_over_a_merge_already_in_the_tree_puts_in_the_tests_alone(
        tmp_path, monkeypatch):
    """Changes of the user's own elsewhere in the tree do not stop it being found."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch, open_path=["a.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    assert first["land"]["files"] == 2
    (repo / "kept.txt").write_text("one\nmine\n", encoding="utf-8")
    _test_stage_container(monkeypatch)
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True)
    assert again["land"]["files"] == 2 and again["land_why"] == ""
    assert _in_tree(repo, "tests/test_a.txt", "tests/test_b.txt")
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "one\nmine\n"


def test_carrying_on_over_part_of_a_merge_writes_nothing_and_keeps_the_branch(
        tmp_path, monkeypatch, caplog):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch, open_path=["a.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    (repo / "b.txt").unlink()
    _test_stage_container(monkeypatch)
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True)
    assert again["land"] is None and "holds part of what" in again["land_why"]
    assert "no longer applies to it whole" in again["land_why"]
    assert sandbox.land_status(again) == 1
    assert not _in_tree(repo, "tests/test_a.txt", "b.txt") and _in_tree(repo, "a.txt")
    assert not _branch_gone(repo, first["branch"])
    assert sandbox.land_why_text(again) in caplog.text
    # The command names the branch whole and reads what the stages settled
    assert again["land_command"] == (
        f"git -C {repo} diff --no-textconv --no-ext-diff {again['base']} "
        f"{first['branch']}")
    proc = subprocess.run(["sh", "-c", again["land_command"]], capture_output=True,
                          text=True, check=False)
    assert proc.returncode == 0
    assert "b/tests/test_a.txt" in proc.stdout and "b/b.txt" in proc.stdout


def test_a_union_whose_stages_were_all_settled_runs_none_of_them_again(tmp_path,
                                                                      monkeypatch):
    """Merged again alone, no stage runs over the new union, which records the last
    stage as settled; carried on from by the integrate command, it goes into the working
    tree as it is, with no container started. A union left unsettled runs every stage."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "REFACTOR", True)
    _test_stage_container(monkeypatch, merge_status=1)
    open_union = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    assert sandbox._settle_stage_of(str(repo), open_union["branch"]) == ""
    run_list = _test_stage_container(monkeypatch)
    sandbox.integrate(str(repo), str(out), [], resume=open_union["branch"])
    assert _kind_list(run_list)[0] == "merge" and "refactor" in _kind_list(run_list)
    _test_stage_container(monkeypatch)
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    assert sandbox.stage_resume_dict(str(repo), first["branch"]) == {}
    assert sandbox._settle_stage_of(str(repo), first["branch"]) == "refactor"
    run_list = _test_stage_container(monkeypatch)
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"])
    assert _kind_list(run_list) == ["merge"]
    tip = _git(repo, "rev-parse", again["branch"]).strip()
    assert _git(repo, "rev-parse", f"{tip}^").strip() == again["commit"]
    assert _git(repo, "rev-parse", f"{tip}^{{tree}}").strip() \
        == _git(repo, "rev-parse", f"{again['commit']}^{{tree}}").strip()
    assert sandbox._settle_stage_of(str(repo), again["branch"]) == "refactor"
    assert sandbox.stage_resume_dict(str(repo), again["branch"]) == {}
    run_list = _test_stage_container(monkeypatch)
    result = sandbox.integrate_into_tree(str(repo), str(out), [], resume=first["branch"])
    assert run_list == [] and result["land"]["files"] == 4
    assert _in_tree(repo, "a.txt", "b.txt", "tests/test_a.txt", "tests/test_b.txt")
    assert (repo / "a.txt").read_text(encoding="utf-8") == "y\n"
    assert _branch_gone(repo, first["branch"])


def test_a_union_left_unsettled_by_merging_a_settled_one_alone_runs_no_stage_again(
        tmp_path, monkeypatch):
    """However many merges of it alone were left unsettled, the merge that settles runs
    no stage and records the last one as settled. A merge with a branch added, and a
    union no stage ran over, run every stage."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "REFACTOR", True)
    _test_stage_container(monkeypatch)
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    _test_stage_container(monkeypatch, merge_status=1)
    open_union = sandbox.integrate(str(repo), str(out), [], resume=first["branch"])
    assert open_union["exit_status"] == 1
    assert sandbox._settle_stage_of(str(repo), open_union["branch"]) == ""
    assert sandbox._carry_stage_of(str(repo), open_union["branch"]) == "refactor"
    _test_stage_container(monkeypatch, merge_status=1)
    open_again = sandbox.integrate(str(repo), str(out), [], resume=open_union["branch"])
    assert sandbox._carry_stage_of(str(repo), open_again["branch"]) == "refactor"
    assert sandbox.stage_resume_dict(str(repo), open_again["branch"]) == {}
    run_list = _test_stage_container(monkeypatch)
    again = sandbox.integrate(str(repo), str(out), [], resume=open_again["branch"])
    assert _kind_list(run_list) == ["merge"] and again["exit_status"] == 0
    assert sandbox._settle_stage_of(str(repo), again["branch"]) == "refactor"
    run_list = _test_stage_container(monkeypatch)
    sandbox.integrate(str(repo), str(out), ["codetwine-loom/a"], resume=open_union["branch"])
    assert _kind_list(run_list)[0] == "merge" and "refactor" in _kind_list(run_list)
    # A union no stage ran over, of two branches and of one
    start = _git(repo, "rev-parse", "main").strip()
    for branch_list, base in ((["codetwine-loom/a", "codetwine-loom/b"], ""),
                              (["codetwine-loom/a"], start)):
        _test_stage_container(monkeypatch, merge_status=1)
        fresh = sandbox.integrate(str(repo), str(out), branch_list, base=base)
        assert sandbox._carry_stage_of(str(repo), fresh["branch"]) == ""
        run_list = _test_stage_container(monkeypatch)
        sandbox.integrate(str(repo), str(out), [], resume=fresh["branch"])
        assert _kind_list(run_list)[0] == "merge" and "refactor" in _kind_list(run_list)


def test_a_union_left_unsettled_by_merging_a_settled_one_alone_runs_a_stage_turned_on_after(
        tmp_path, monkeypatch):
    """A stage after the last one settled whose setting is on now runs over the record
    of that one, and no stage before it runs; left unsettled, it is carried on from
    alone. The settled union carried on from at once runs the same stage."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 0)
    _test_stage_container(monkeypatch)
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    assert sandbox._settle_stage_of(str(repo), first["branch"]) == "test-create"
    _test_stage_container(monkeypatch, merge_status=1)
    open_union = sandbox.integrate(str(repo), str(out), [], resume=first["branch"])
    monkeypatch.setattr(config, "REFACTOR", True)
    run_list = _test_stage_container(monkeypatch, refactor_dict={"refactor": "unsettled"})
    again = sandbox.integrate(str(repo), str(out), [], resume=open_union["branch"])
    assert _kind_list(run_list) == ["merge", "refactor"]
    assert sandbox.stage_resume_dict(str(repo), again["branch"])["stage"] == "refactor"
    run_list = _test_stage_container(monkeypatch)
    last = sandbox.integrate(str(repo), str(out), [], resume=again["branch"])
    assert _kind_list(run_list) == ["refactor"] and last["exit_status"] == 0
    assert sandbox._settle_stage_of(str(repo), last["branch"]) == "refactor"
    run_list = _test_stage_container(monkeypatch)
    sandbox.integrate(str(repo), str(out), [], resume=first["branch"])
    assert _kind_list(run_list) == ["refactor"]


def test_a_union_left_unsettled_whose_stage_record_is_not_made_runs_no_stage_over_it(
        tmp_path, monkeypatch):
    """Merged again alone and settled, a union left unsettled by a merge of a settled
    one whose record of the stages is not made runs no stage and goes into no working
    tree: the new union's branch is taken away and the result names the union carried
    on from, which is left as it was. Carried on from once the record is made, it runs
    only the stage after the last one settled."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "REFACTOR", True)
    _test_stage_container(monkeypatch)
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    _test_stage_container(monkeypatch, merge_status=1)
    open_union = sandbox.integrate(str(repo), str(out), [], resume=first["branch"])
    open_tip = _git(repo, "rev-parse", open_union["branch"]).strip()
    branch_list = _branches(repo)
    monkeypatch.setattr(config, "TEST_REFACTOR", True)
    real_record = sandbox._land_stage_record
    monkeypatch.setattr(sandbox, "_land_stage_record", lambda *a, **k: None)
    run_list = _test_stage_container(monkeypatch)
    stop = sandbox.integrate_into_tree(str(repo), str(out), [],
                                       resume=open_union["branch"])
    assert _kind_list(run_list) == ["merge"]
    assert stop["branch"] == open_union["branch"] and stop["commit"] == open_tip
    assert stop["patch"] is None and stop["files"] == 0
    assert stop["land"] is None and sandbox.land_status(stop) == 1
    assert "could not be recorded on the union" in stop["land_why"]
    assert sandbox.resume_branch_list([], stop, sandbox.union_status(stop)) \
        == [open_union["branch"]]
    assert _branches(repo) == branch_list and not _in_tree(repo, "a.txt", "b.txt")
    monkeypatch.setattr(sandbox, "_land_stage_record", real_record)
    run_list = _test_stage_container(monkeypatch)
    sandbox.integrate(str(repo), str(out), [], resume=open_union["branch"])
    assert _kind_list(run_list) == ["merge", "test-refactor"]


def _commit_on(repo, branch, start, path, text):
    """Make a branch at start, or check it out, and commit one file on it; then check
    out main again."""
    _git(repo, "checkout", "--quiet", *(["-b", branch, start] if start else [branch]))
    (repo / path).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", path)
    _git(repo, "checkout", "--quiet", "main")


def _merge_commit(repo, tree, parent_list, message):
    """Return a commit with tree's tree over the parents named."""
    parent_option_list = [word for parent in parent_list for word in ("-p", parent)]
    return _git(repo, "commit-tree", f"{tree}^{{tree}}", *parent_option_list, "-m",
                message).strip()


def test_an_unsettled_union_is_held_by_the_working_tree_from_where_each_branch_parts(
        tmp_path):
    """A branch somebody made that ends with a merge of main or with an empty commit,
    and the last task branch of a chain whose earlier task the merge left out, are
    read from where they part from what the union was written against. A settled
    union merged in is read from the commit it was written against."""
    repo = _repo(tmp_path / "repo")
    for name in ("fa", "fb"):
        _commit_on(repo, name, "main", f"{name}.txt", f"{name}\n")
    _commit_on(repo, "main", "", "main2.txt", "main2\n")
    for name in ("fa", "fb"):
        _git(repo, "checkout", "--quiet", name)
        _git(repo, "merge", "--quiet", "--no-ff", "--no-edit", "main")
    _git(repo, "checkout", "--quiet", "main")
    _commit_on(repo, "fe", "main", "fe.txt", "fe\n")
    _branch_on_top(repo, "fe-empty", "fe", "nothing")
    _commit_on(repo, "c1", "main", "c1.txt", "c1\n")
    _commit_on(repo, "c2", "c1", "c2.txt", "c2\n")
    # A settled union written against a commit holding a file of the tree then
    _commit_on(repo, "snap", "main", "notes.txt", "mine\n")
    _commit_on(repo, "t", "snap", "u.txt", "u\n")
    settle = _merge_commit(repo, "t", ["snap", "t"],
                           "integrate: r\n\nCodetwine-Loom-Verdict: settled\n"
                           "Codetwine-Loom-Exit-Status: 0")
    union_list = []
    for n, side_list in enumerate((["fa", "fb"], ["fe-empty"], ["c2"], [settle, "fa"],
                                   [settle]), 4):
        union_list.append(f"codetwine-loom/sandbox/20260101-000000-{n}-1")
        _git(repo, "branch", union_list[-1], _merge_commit(
            repo, "main", ["main", *side_list],
            "integrate: r\n\nCodetwine-Loom-Verdict: unsettled\nCodetwine-Loom-Exit-Status: 1"))

    def held_list():
        return [sandbox.is_union_in_tree(str(repo), union) for union in union_list]

    assert held_list() == [False] * 5
    for path, text in (("c2.txt", "c2\n"), ("u.txt", "u\n"), ("notes.txt", "mine, too\n")):
        (repo / path).write_text(text, encoding="utf-8")
    assert held_list() == [False, False, False, False, True]
    for path in ("fa.txt", "fb.txt", "fe.txt", "c1.txt"):
        (repo / path).write_text(f"{path[:-4]}\n", encoding="utf-8")
    assert held_list() == [True] * 5


def test_a_settled_union_goes_in_as_it_is_only_over_the_commit_it_was_written_against(
        tmp_path, monkeypatch):
    """A working tree holding none of it that was moved behind that commit stops as the
    merge does, with no container started and the union kept; one moved onto another
    line is merged again with no stage run."""
    repo = _repo(tmp_path / "repo")
    out = tmp_path / "out"
    _analysis_in(out)
    start = gitcmd.require_head(str(repo))
    (repo / "two.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "two")
    for name in ("codetwine-loom/a", "codetwine-loom/b"):
        _git(repo, "checkout", "--quiet", "-b", name, "main")
        (repo / f"{name[-1]}.txt").write_text("x\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "--quiet", "-m", name)
    _git(repo, "checkout", "--quiet", "main")
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "REFACTOR", True)
    _test_stage_container(monkeypatch)
    union = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    assert sandbox._settle_stage_of(str(repo), union["branch"]) == "refactor"
    _git(repo, "checkout", "--quiet", "-b", "old", start)
    run_list = _test_stage_container(monkeypatch)
    with pytest.raises(RuntimeError, match=r"^the working tree is at [0-9a-f]+, behind "
                                           r"[0-9a-f]+, which the branches start from"):
        sandbox.integrate_into_tree(str(repo), str(out), [], resume=union["branch"])
    assert run_list == [] and not _in_tree(repo, "a.txt")
    assert not _branch_gone(repo, union["branch"])
    (repo / "side.txt").write_text("side\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "side")
    run_list = _test_stage_container(monkeypatch)
    result = sandbox.integrate_into_tree(str(repo), str(out), [], resume=union["branch"])
    assert _kind_list(run_list) == ["merge"] and result["land"]["files"] == 2
    assert _in_tree(repo, "a.txt", "b.txt", "side.txt")
    assert _branch_gone(repo, union["branch"])


@pytest.mark.parametrize("branch", ["x$(touch${IFS}MARK)", "a&b", "it's"])
def test_a_stage_left_open_is_logged_with_the_flag_a_shell_reads_whole(caplog, branch):
    sandbox._warn_stage_open(integrate_mod.TEST_CREATE_STAGE, {
        "branch": branch, "why": "a case is not done", "commit": "c" * 40})
    line = caplog.records[-1].getMessage()
    assert line.endswith(f"Carry on from it with integrate --resume {shlex.quote(branch)}")
    assert shlex.split(line.rsplit("integrate ", 1)[1]) == ["--resume", branch]


def test_what_the_tree_holds_of_an_earlier_change_is_told_by_its_context(tmp_path):
    repo = _repo(tmp_path / "repo")
    start = gitcmd.require_head(str(repo))
    (repo / "kept.txt").write_text("one\ntwo\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "the change")
    settled = gitcmd.require_head(str(repo))
    patch = str(tmp_path / "held.diff")
    assert sandbox.held_base(str(repo), start, settled, patch) == settled
    assert sandbox.held_base(str(repo), start, start, patch) == start
    _git(repo, "reset", "--quiet", "--hard", start)
    assert sandbox.held_base(str(repo), start, settled, patch) == start
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    assert sandbox.held_base(str(repo), start, settled, patch) == ""


def test_the_tree_holding_a_merge_and_not_the_stage_above_it_is_held_up_to_the_merge(
        tmp_path):
    """The stage's change was not put in: what the stage is carried on from goes in
    over the merge, and integrate --resume reads the working tree the same way."""
    repo = _repo(tmp_path / "repo")
    identity = ("-c", "user.email=t@t", "-c", "user.name=t")
    start = gitcmd.require_head(str(repo))
    union = "codetwine-loom/sandbox/20260102-000000-1-1"
    _git(repo, "checkout", "--quiet", "-b", "side")
    for name in ("a.txt", "b.txt"):
        (repo / name).write_text(f"{name}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, *identity, "commit", "--quiet", "-m", "one side")
    _git(repo, "checkout", "--quiet", "-b", union, start)
    _git(repo, *identity, "merge", "--quiet", "--no-ff", "-m",
         "integrate: x\n\ncombine\n\nCodetwine-Loom-Run-Request: add a\n"
         "Codetwine-Loom-Verdict: settled\n",
         "side")
    merge = gitcmd.require_head(str(repo))
    _git(repo, "rm", "--quiet", "b.txt")
    _git(repo, *identity, "commit", "--quiet", "-m", _REFACTOR_TEXT)
    settled = gitcmd.require_head(str(repo))
    _git(repo, "checkout", "--quiet", "main")
    patch = str(tmp_path / "held.diff")
    assert sandbox.held_base(str(repo), start, settled, patch) == start
    for name in ("a.txt", "b.txt"):
        (repo / name).write_text(f"{name}\n", encoding="utf-8")
    assert sandbox.held_base(str(repo), start, settled, patch) == merge
    assert sandbox._held_union_commit(str(repo), union) == merge
    (repo / "b.txt").unlink()
    assert sandbox.held_base(str(repo), start, settled, patch) == settled
    assert sandbox._held_union_commit(str(repo), union) == settled
    (repo / "a.txt").write_text("the user's own\n", encoding="utf-8")
    assert sandbox.held_base(str(repo), start, settled, patch) == ""
    assert sandbox._held_union_commit(str(repo), union) == ""


def test_the_tree_holding_a_merge_and_a_file_the_stage_above_added_changed_holds_part(
        tmp_path):
    """What is above the merge it holds goes in over it only where it applies there."""
    repo = _repo(tmp_path / "repo")
    identity = ("-c", "user.email=t@t", "-c", "user.name=t")
    start = gitcmd.require_head(str(repo))
    union = "codetwine-loom/sandbox/20260102-000000-1-1"
    _git(repo, "checkout", "--quiet", "-b", "side")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, *identity, "commit", "--quiet", "-m", "one side")
    _git(repo, "checkout", "--quiet", "-b", union, start)
    _git(repo, *identity, "merge", "--quiet", "--no-ff", "-m",
         "integrate: x\n\ncombine\n\nCodetwine-Loom-Run-Request: add a\n"
         "Codetwine-Loom-Verdict: settled\n",
         "side")
    merge = gitcmd.require_head(str(repo))
    (repo / "test_a.txt").write_text("t\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, *identity, "commit", "--quiet", "-m", _REFACTOR_TEXT)
    settled = gitcmd.require_head(str(repo))
    _git(repo, "checkout", "--quiet", "main")
    patch = str(tmp_path / "held.diff")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    assert sandbox.held_base(str(repo), start, settled, patch) == merge
    (repo / "test_a.txt").write_text("t\n", encoding="utf-8")
    assert sandbox.held_base(str(repo), start, settled, patch) == settled
    (repo / "test_a.txt").write_text("t\nmine\n", encoding="utf-8")
    assert sandbox.held_base(str(repo), start, settled, patch) == ""
    assert sandbox._held_union_commit(str(repo), union) == merge


def test_an_earlier_change_whose_lines_are_found_elsewhere_is_put_in_again(tmp_path):
    """A working tree the user put back to the start does not hold a change that adds
    one more of the lines it is full of, so the change goes in again whole."""
    repo = _repo(tmp_path / "repo")
    (repo / "same.txt").write_text("same\n" * 12, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", "start")
    start = gitcmd.require_head(str(repo))
    (repo / "same.txt").write_text("same\n" * 13, encoding="utf-8")
    _git(repo, "commit", "--quiet", "-am", "the change")
    settled = gitcmd.require_head(str(repo))
    patch = str(tmp_path / "held.diff")
    assert sandbox.held_base(str(repo), start, settled, patch) == settled
    _git(repo, "checkout", "--quiet", start, "--", "same.txt")
    assert sandbox.held_base(str(repo), start, settled, patch) == start


@pytest.mark.parametrize("name, is_own", [
    ("codetwine-loom/sandbox/20260101-000000-1-42", True),
    ("codetwine-loom/run/20260101-000000-1-42/2", True),
    ("codetwine-loom/run/20260101-000000-1-42/2-retry1", True), ("feature-x", False),
    ("codetwine-loom/mine", False), ("mine/20260101-000000-1-42", False), ("", False)])
def test_a_name_this_tool_made_up_is_under_its_root_and_carries_a_run(name, is_own):
    assert sandbox.is_own_branch(name) is is_own


def test_a_branch_somebody_else_named_is_left_once_what_it_holds_is_in_the_tree(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, open_path=["a.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    # The person renames the branch, and has a branch of their own beside it
    _git(repo, "branch", "-m", first["branch"], "feature-x")
    _git(repo, "branch", "feature-x-test/mine", "main")
    run_list = _test_stage_container(monkeypatch)
    again = sandbox.integrate(str(repo), str(out), [], resume="feature-x", land=True)
    assert _kind_list(run_list) == ["list", "tests"]
    assert again["land"]["files"] == 4 and not sandbox.open_stage_key(again)
    # The name stays, on what the stages left, and so does the branch beside it
    assert _git(repo, "rev-parse", "feature-x").strip() == again["test_create"]["commit"]
    assert not _branch_gone(repo, "feature-x-test/mine")


def test_a_run_carrying_on_from_open_tests_hands_out_no_task(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch, open_path=["a.txt"])
    union = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                      request="add a and b")
    run_list = _test_stage_container(monkeypatch)
    result = sandbox.dispatch(str(repo), str(out), "add a and b",
                              resume_list=[union["branch"]])
    assert _kind_list(run_list) == ["list", "tests"]
    assert result["tasks"] == [] and result["resume"] == [] and result["exit_status"] == 0
    assert result["integration"]["test_create"]["verdict"] == "settled"


def test_the_test_tasks_are_filled_by_the_definitions_the_change_touched():
    test_by_file_dict = {f"p{n}.py": [] for n in range(5)}
    def_count_dict = {"p0.py": 6, "p1.py": 1, "p2.py": 12, "p3.py": 2, "p4.py": 3}
    task_list = sandbox.test_task_list(test_by_file_dict, def_count_dict, 10)
    # 24 definitions are three test tasks of 8; a file's definitions stay together
    assert [task["paths"] for task in task_list] == \
        [["p0.py", "p1.py"], ["p2.py"], ["p3.py", "p4.py"]]

@pytest.mark.parametrize("first_case, expect", [
    (5, [(5, 3), (6, 3), (7, 3)]), (None, [None] * 3), (0, [None] * 3), (True, [None] * 3)])
def test_each_test_task_hands_out_case_numbers_no_other_does(first_case, expect):
    task_list = [{"paths": [f"p{n}.py"]} for n in range(3)]
    sandbox.add_case_number(task_list, first_case)
    assert [(task["first_case"], task["case_step"]) if "first_case" in task else None
            for task in task_list] == expect
    if first_case == 5:
        number_set_list = [{task["first_case"] + task["case_step"] * k for k in range(50)}
                           for task in task_list]
        assert sum(len(number_set) for number_set in number_set_list) == \
            len(set().union(*number_set_list))


# == Several test tasks, each in a container of its own ==


def test_several_test_tasks_land_on_branches_of_their_own_and_become_one_commit(
        tmp_path, monkeypatch, caplog):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    number_list = []
    run_list = _test_stage_container(monkeypatch, number_list=number_list)
    with caplog.at_level("INFO", logger="codetwine_loom.progress"):
        result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                                   land=True, request="add a and b")
    # Each test task hands out the case numbers the other does not
    assert sorted(number_list) == [(3, 2), (4, 2)]
    start_list = [r.getMessage() for r in caplog.records
                  if r.getMessage().endswith(": start")]
    assert start_list.count("combine the branches: start") == 1
    assert start_list.count("combine the test branches: start") == 1
    assert _kind_list(run_list)[:2] == ["merge", "list"]
    assert sorted(_kind_list(run_list)[2:4]) == ["tests", "tests"]
    assert _kind_list(run_list)[4:] == ["combine"]
    assert sorted(run[3] for run in run_list[2:4]) == [["a.txt"], ["b.txt"]]
    # The combining run starts at the union and is not followed by tests of its own
    assert run_list[4][2] == result["commit"]
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "settled" and len(create_dict["test_tasks"]) == 2
    assert create_dict["tests_changed"] == ["tests/test_a.txt", "tests/test_b.txt"]
    head = create_dict["commit"]
    assert result["land"]["files"] == 4 and _branch_gone(repo, result["branch"])
    parent_list = _git(repo, "rev-list", "--parents", "-n", "1", head).split()[1:]
    assert parent_list[0] == result["commit"] and len(parent_list) == 3
    message = _git(repo, "log", "-1", "--format=%B", head)
    assert message.startswith("test-create: add a and b")
    assert "Codetwine-Loom-Stage: test-create" in message \
        and "Codetwine-Loom-Verdict: settled" in message
    assert "Codetwine-Loom-Tests-Base: red" in message and "Codetwine-Loom-Test-Open" not in message
    assert (repo / "tests" / "test_a.txt").exists() and (repo / "tests" / "test_b.txt").exists()
    # Every branch the stage made is reached from the commit that went in and is gone
    assert "-test/" not in _git(repo, "branch", "--list")


def test_the_test_tasks_of_one_chain_are_written_one_over_the_other(tmp_path, monkeypatch,
                                                                    caplog):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    at_once_list = []
    run_list = _test_stage_container(monkeypatch, share_test="tests/test_all.txt",
                                     at_once_list=at_once_list)
    with caplog.at_level(logging.INFO):
        result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                                   land=True, request="add a and b")
    assert _kind_list(run_list) == ["merge", "list", "tests", "tests", "combine"]
    assert at_once_list == [1, 1]
    assert "waits for the test tasks before it that write the same test files" in caplog.text
    # The first is written over the union, and the second over what the first left
    union = result["commit"]
    assert run_list[2][2] == union and run_list[3][2] != union
    assert _git(repo, "rev-parse", f"{run_list[3][2]}^").strip() == union
    assert f"tests/test_{run_list[2][3][0]}" in _git(
        repo, "ls-tree", "-r", "--name-only", run_list[3][2])
    # The last branch of the chain alone is combined, and it holds both
    assert [word for word in run_list[4][1] if "-test/" in word] == \
        [f"{result['branch']}-test/{1 + ['a.txt', 'b.txt'].index(run_list[3][3][0])}"]
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "settled"
    assert create_dict["tests_changed"] == ["tests/test_a.txt", "tests/test_b.txt"]
    head = create_dict["commit"]
    assert result["land"]["files"] == 4 and _branch_gone(repo, result["branch"])
    assert len(_git(repo, "rev-list", "--parents", "-n", "1", head).split()[1:]) == 2
    assert (repo / "tests" / "test_a.txt").exists() and (repo / "tests" / "test_b.txt").exists()
    assert "-test/" not in _git(repo, "branch", "--list")


def test_the_next_of_a_chain_is_written_over_the_last_that_settled(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    run_list = _test_stage_container(monkeypatch, share_test="tests/test_all.txt",
                                     open_path=["a.txt"])
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    # Whichever of the two went first, the one that settled was written over the union
    start_dict = {run[3][0]: run[2] for run in run_list if run[0] == "tests"}
    assert start_dict["b.txt"] == result["commit"]
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "unsettled" and create_dict["open_paths"] == ["a.txt"]
    assert result["land"]["files"] == 2 and not _in_tree(repo, "tests/test_b.txt")
    tip = _git(repo, "rev-parse", result["branch"]).strip()
    message = _git(repo, "log", "-1", "--format=%B", tip)
    assert "Codetwine-Loom-Verdict: unsettled" in message \
        and 'Codetwine-Loom-Test-Open: ["a.txt"]' in message
    name_text = _git(repo, "ls-tree", "-r", "--name-only", tip)
    assert "tests/test_b.txt" in name_text and "tests/test_a.txt" not in name_text
    assert f"{result['branch']}-test/1" in _git(repo, "branch", "--list")


def test_one_open_test_task_leaves_the_union_merged_and_names_its_files_on_the_commit(
        tmp_path, monkeypatch, caplog):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, open_path=["b.txt"])
    with caplog.at_level(logging.WARNING):
        result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                                   land=True)
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "unsettled" and create_dict["open_paths"] == ["b.txt"]
    assert "test task 2 (b.txt)" in create_dict["why"]
    # The tree holds the union alone
    assert result["land"]["files"] == 2 and not _in_tree(repo, "tests/test_a.txt")
    # The branch holds the tests of the test task that settled, and says which files are open
    tip = _git(repo, "rev-parse", result["branch"]).strip()
    assert tip == create_dict["commit"]
    message = _git(repo, "log", "-1", "--format=%B", tip)
    assert "Codetwine-Loom-Verdict: unsettled" in message \
        and 'Codetwine-Loom-Test-Open: ["b.txt"]' in message
    assert "tests/test_a.txt" in _git(repo, "show", "--stat", "--format=", tip)
    assert "tests/test_b.txt" not in _git(repo, "ls-tree", "-r", "--name-only", tip)
    # The open test task's own branch is left to be read
    assert f"{result['branch']}-test/2" in _git(repo, "branch", "--list")
    # The tests are written again over the tip, which is not what is merged
    assert sandbox.stage_resume_dict(str(repo), result["branch"]) == {
        "stage": "test-create", "union": tip, "settled": result["commit"],
        "merge": result["commit"],
        "base": _git(repo, "rev-parse", f"{result['commit']}^").strip(), "paths": ["b.txt"]}


def test_carrying_on_that_lands_nothing_leaves_the_union_alone_merged(tmp_path,
                                                                      monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, open_path=["b.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    kept = first["test_create"]["commit"]
    _test_stage_container(monkeypatch, open_path=["b.txt"], no_patch=True)
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True)
    assert again["test_create"]["commit"] is None
    assert again["test_create"]["verdict"] == "unsettled"
    # The branch still holds the tests that settled, and none of them is in the tree
    assert _git(repo, "rev-parse", first["branch"]).strip() == kept
    assert again["land"]["files"] == 0 and _in_tree(repo, "a.txt")
    assert not (repo / "tests" / "test_a.txt").exists()


def test_carrying_on_from_an_open_test_task_writes_that_one_alone_on_top(tmp_path,
                                                                      monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, open_path=["b.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    kept = first["test_create"]["commit"]
    run_list = _test_stage_container(monkeypatch)
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True)
    assert _kind_list(run_list) == ["list", "tests"]
    assert run_list[1][3] == ["b.txt"] and run_list[1][2] == kept
    # What goes in is what the branch now holds over what was already put in: both
    # test tasks' tests, and the branch's name goes
    assert again["commit"] == first["commit"]
    tip = again["test_create"]["commit"]
    assert _git(repo, "rev-parse", f"{tip}^").strip() == kept
    assert again["land"]["files"] == 2
    assert (repo / "tests" / "test_a.txt").exists() and (repo / "tests" / "test_b.txt").exists()
    assert "-test/" not in _git(repo, "branch", "--list")
    assert _branch_gone(repo, first["branch"])


def test_a_gap_of_an_open_test_task_is_not_on_the_commit_the_tasks_are_combined_in(
        tmp_path, monkeypatch):
    """The open test task is written again; the case it recorded goes with the try
    that recorded it, and the settled test task's case stays."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, open_path=["b.txt"], gap_path=["a.txt", "b.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    assert first["test_create"]["verdict"] == "unsettled"
    message = _git(repo, "log", "-1", "--format=%B", first["test_create"]["commit"])
    assert 'Codetwine-Loom-Gap: [{"what":"case_1","why":"a.txt does not do it"}]' in message
    assert "b.txt does not do it" not in message


def test_an_open_test_task_written_again_and_settled_leaves_no_case_of_its_first_try(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, open_path=["b.txt"], gap_path=["b.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    assert "Codetwine-Loom-Gap" not in _git(repo, "log", "-1", "--format=%B",
                                 first["test_create"]["commit"])
    _test_stage_container(monkeypatch)
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True)
    assert again["test_create"]["verdict"] == "settled"
    assert again["gap"] == [] and sandbox.union_status(again) == 0
    assert _branch_gone(repo, first["branch"])


# == The stages that refactor, after the tests ==============


def _all_stage_on(monkeypatch, test_create=True):
    monkeypatch.setattr(config, "TEST_CREATE", test_create)
    monkeypatch.setattr(config, "REFACTOR", True)
    monkeypatch.setattr(config, "TEST_REFACTOR", True)


def test_an_unsettled_refactoring_is_run_again_over_the_commit_it_started_at(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "REFACTOR", True)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 1)
    state = {"refactor": 0}
    _test_stage_container(monkeypatch)
    start = sandbox._start_container

    def start_twice(*arg_list, **kwargs):
        # The first refactoring is left unsettled, the one run again settles
        if arg_list[-2][0] == "refactor":
            state["refactor"] += 1
            if state["refactor"] == 1:
                _test_stage_container(monkeypatch, refactor_dict={"refactor": "unsettled"})
                result = sandbox._start_container(*arg_list, **kwargs)
                monkeypatch.setattr(sandbox, "_start_container", start_twice)
                return result
        return start(*arg_list, **kwargs)

    monkeypatch.setattr(sandbox, "_start_container", start_twice)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               request="add a and b")
    assert state["refactor"] == 2
    assert result["refactor"]["verdict"] == "settled"
    assert not sandbox.open_stage_key(result)
    # The refactoring nobody settled is not under the branch: one commit over the merge
    tip = _git(repo, "rev-parse", result["branch"]).strip()
    assert tip == result["refactor"]["commit"]
    assert _git(repo, "rev-list", "--count", f"{result['commit']}..{tip}").strip() == "1"


def test_a_refactoring_still_unsettled_is_run_again_no_more_than_the_setting(tmp_path,
                                                                            monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "REFACTOR", True)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 2)
    run_list = _test_stage_container(monkeypatch, refactor_dict={"refactor": "unsettled"})
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               request="add a and b")
    assert _kind_list(run_list) == ["merge", "refactor", "refactor", "refactor"]
    assert sandbox.open_stage_key(result) == "refactor"
    # Each run again starts over the settled merge, not over what the one before left
    assert {entry[2] for entry in run_list[1:]} == {result["commit"]}


def test_only_the_open_test_files_are_written_again(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 1)
    run_list = _test_stage_container(monkeypatch, open_path=["b.txt"])
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    test_run_list = [entry for entry in run_list if entry[0] == "tests"]
    # a.txt and b.txt once each, then b.txt alone
    assert sorted(entry[3] for entry in test_run_list[:2]) == [["a.txt"], ["b.txt"]]
    assert [entry[3] for entry in test_run_list[2:]] == [["b.txt"]]
    assert sandbox.open_stage_key(result) == "test_create"
    assert sandbox.stage_resume_dict(str(repo), result["branch"])["paths"] == ["b.txt"]


@pytest.mark.parametrize("path_list", [
    ["a.py"], ["src/my lib/a.py", "b  c.py"],
    ["tab\there.py", "全角\u3000空白.py", 'q"uote.py', "back\\slash.py", "line\nbreak.py"]])
def test_the_paths_a_trailer_holds_are_read_back_as_they_were_written(tmp_path, path_list):
    text = sandbox.path_list_text(path_list)
    assert text == "".join(text.split())
    assert sandbox.path_list_of(text) == path_list
    # Through a commit's message and back, as the files still open are
    repo = _repo(tmp_path / "repo")
    _git(repo, "commit", "--quiet", "--allow-empty", "-m", sandbox.commit_message(
        "r", sandbox.run_trailer_dict({}, {"stage": "test-create", "test_open": path_list})))
    record_dict = integrate_mod.commit_record_list(str(repo), "HEAD^", "HEAD")[-1]
    assert sandbox.path_list_of(record_dict["test_open"]) == path_list


def test_no_path_writes_no_trailer_and_text_that_is_no_list_of_paths_is_not_read():
    assert sandbox.path_list_text([]) == "" and sandbox.path_list_of("") == []
    assert "Codetwine-Loom-Test-Open" not in sandbox.commit_message(
        "r", sandbox.run_trailer_dict({}, {"test_open": []}))
    for text in ("src/my lib/a.py", '{"a": 1}', "[1, 2]", '"a.py"'):
        assert sandbox.path_list_of(text) is None


def _listing_ends_with(monkeypatch, status, lose_outcome=False):
    """Have the run that lists what to write tests for end with status, after it wrote
    its listing or, with lose_outcome, with none left."""
    start = sandbox._start_container

    def wrapped(*arg_list, **kwargs):
        engine, image, own_status = start(*arg_list, **kwargs)
        if "--target-only" not in arg_list[-2]:
            return engine, image, own_status
        if lose_outcome:
            os.remove(arg_list[8].outcome)
        return engine, image, status

    monkeypatch.setattr(sandbox, "_start_container", wrapped)


def test_a_listing_handed_back_is_taken_whatever_its_container_ended_with(
        tmp_path, monkeypatch, caplog):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    run_list = _test_stage_container(monkeypatch)
    _listing_ends_with(monkeypatch, sandbox._TIME_OUT_STATUS)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        result_dict = sandbox.integrate(str(repo), str(out),
                                        ["codetwine-loom/a", "codetwine-loom/b"])
    assert _kind_list(run_list)[:3] == ["merge", "list", "tests"]
    assert result_dict["test_create"]["verdict"] == "settled"
    assert sandbox.union_status(result_dict) == 0
    assert "ended with exit status 124 after it wrote what there is to write tests " \
        "for" in caplog.text


@pytest.mark.parametrize("timeout_sec, what", [
    (0, "exit status 124"),
    (3600, "exit status 124, the status of a container taken away at SANDBOX_TIMEOUT_SEC "
           "(3600 s)")])
def test_a_listing_that_was_not_handed_back_leaves_the_tests_unsettled(tmp_path,
                                                                     monkeypatch,
                                                                     timeout_sec, what):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SEC", timeout_sec)
    run_list = _test_stage_container(monkeypatch)
    _listing_ends_with(monkeypatch, sandbox._TIME_OUT_STATUS, lose_outcome=True)
    result_dict = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    assert _kind_list(run_list) == ["merge", "list"]
    create_dict = result_dict["test_create"]
    assert create_dict["verdict"] == "unsettled"
    assert create_dict["why"] == (f"what there is to write tests for could not be listed "
                                  f"({what})")


_UNSETTLED_OUTCOME = {"verdict": "unsettled", "why": "- conflict markers are left"}


def _combine_left_unsettled(monkeypatch, outcome_dict=_UNSETTLED_OUTCOME, status=1):
    """Have the run combining the test tasks end unsettled: its patch holds a conflict
    marker, its outcome is outcome_dict (None for no outcome at all) and it ends with
    status."""
    start = sandbox._start_container

    def wrapped(*arg_list, **kwargs):
        engine, image, container_status = start(*arg_list, **kwargs)
        argv_tail, run_file = arg_list[-2], arg_list[8]
        if argv_tail[0] == "integrate" and "-test/" in " ".join(argv_tail):
            patch = pathlib.Path(run_file.patch)
            patch.write_text(patch.read_text(encoding="utf-8").replace(
                "+t\n", "+<<<<<<< ours\n", 1), encoding="utf-8")
            if outcome_dict is None:
                pathlib.Path(run_file.outcome).unlink()
            else:
                pathlib.Path(run_file.outcome).write_text(json.dumps(outcome_dict),
                                                          encoding="utf-8")
            pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(
                json.dumps({"exit_status": status}), encoding="utf-8")
            return engine, image, status
        return engine, image, container_status

    monkeypatch.setattr(sandbox, "_start_container", wrapped)


def test_the_test_tasks_combined_and_left_unsettled_are_no_tests_to_carry_on_from(
        tmp_path, monkeypatch):
    """Two test tasks settled and one did not, and the run combining the two ended
    unsettled: the commit it left names no stage and no file as open, so carrying on
    from its branch starts no stage over it."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, target=("a.txt", "b.txt", "c.txt"),
                          open_path=["c.txt"])
    _combine_left_unsettled(monkeypatch)
    result_dict = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                                    land=True, request="add a and b")
    create_dict = result_dict["test_create"]
    assert create_dict["verdict"] == "unsettled" and create_dict["commit"] is None
    combine_branch = create_dict["why"].split("combined on ")[1].split(" and ")[0]
    trailer_text = _git(repo, "log", "-1", "--format=%(trailers)", combine_branch)
    assert "Codetwine-Loom-Verdict: unsettled" in trailer_text
    assert "Codetwine-Loom-Why: - conflict markers are left" in trailer_text
    assert create_dict["why"].endswith(
        f" / the test tasks' tests were combined on {combine_branch} and the merge was not "
        f"settled (exit status 1): - conflict markers are left")
    for name in ("Codetwine-Loom-Stage", "Codetwine-Loom-Test-Open"):
        assert name not in trailer_text
    assert _git(repo, "log", "-1", "--format=%s", combine_branch).startswith("integrate:")
    assert sandbox.stage_resume_dict(str(repo), combine_branch) == {}
    # The branch of the settled merge is still what the tests are written over
    assert sandbox.stage_resume_dict(str(repo), result_dict["branch"])["stage"] == \
        "test-create"


@pytest.mark.parametrize("rerun_count", [0, 1])
def test_the_branch_of_combined_tests_left_unsettled_goes_when_the_stage_is_run_again(
        tmp_path, monkeypatch, rerun_count):
    """The run combining the test tasks is left unsettled once and settles when the
    stage is run again, by the same command or by one carrying on from the union:
    nothing of the request is left on a branch."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", rerun_count)
    _test_stage_container(monkeypatch)
    plain_start = sandbox._start_container
    _combine_left_unsettled(monkeypatch)
    unsettled_start = sandbox._start_container
    combine_count = [0]

    def start(*arg_list, **kwargs):
        argv_tail = arg_list[-2]
        if argv_tail[0] == "integrate" and "-test/" in " ".join(argv_tail):
            combine_count[0] += 1
        return (unsettled_start if combine_count[0] <= 1 else plain_start)(*arg_list, **kwargs)

    monkeypatch.setattr(sandbox, "_start_container", start)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True, request="add a and b")
    if not rerun_count:
        assert sandbox.open_stage_key(result) == "test_create"
        combine_branch = result["test_create"]["why"].split(
            "combined on ")[1].split(" and ")[0]
        assert combine_branch == sandbox._TEST_UNION_BRANCH.format(branch=result["branch"])
        assert combine_branch in _branches(repo)
        result = sandbox.integrate(str(repo), str(out), [], resume=result["branch"],
                                   land=True, request="add a and b")
    assert combine_count == [2] and not sandbox.open_stage_key(result)
    assert [branch for branch in _branches(repo)
            if branch.startswith("codetwine-loom/sandbox/")] == []
    assert _in_tree(repo, "a.txt", "b.txt", "tests/test_a.txt", "tests/test_b.txt")


def _branch_on_top(repo, branch, start, message):
    """Point a new branch at a commit made on top of start that changes nothing,
    without touching the working tree."""
    commit = _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit-tree",
                  f"{start}^{{tree}}", "-p", start, "-m", message).strip()
    _git(repo, "branch", branch, commit)


def test_a_branch_written_against_a_union_s_commit_stays_when_its_stage_is_run_again(
        tmp_path, monkeypatch):
    """Another request started from the commit a union landed leaves a union of its
    own on that commit. The stage of the first union, run again and settled, takes
    away the branches named under that union alone."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    _test_stage_container(monkeypatch)
    plain_start = sandbox._start_container
    _combine_left_unsettled(monkeypatch)
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True, request="add a and b")
    union = first["branch"]
    assert sandbox._TEST_UNION_BRANCH.format(branch=union) in _branches(repo)
    text = "integrate: x\n\ncombine t\n\nCodetwine-Loom-Verdict: unsettled\n"
    other_list = ["codetwine-loom/sandbox/20260102-000001-2-1", "codetwine-loom/sandbox/mine",
                  "feature"]
    for other in other_list:
        _branch_on_top(repo, other, union, text)
    monkeypatch.setattr(sandbox, "_start_container", plain_start)
    again = sandbox.integrate(str(repo), str(out), [], resume=union, land=True,
                              request="add a and b")
    assert again["land"] and not sandbox.open_stage_key(again)
    assert sorted(_branches(repo)) == sorted(["main", "codetwine-loom/a", "codetwine-loom/b",
                                              *other_list])


def test_the_combined_tests_of_a_union_this_tool_did_not_name_land_on_one_branch(
        tmp_path, monkeypatch):
    """The stage of a union under a name somebody else gave it is run twice, its
    combined tests left unsettled both times: the second run makes the branch they
    are combined on again where the first left it."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    _test_stage_container(monkeypatch)
    _combine_left_unsettled(monkeypatch)
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True, request="add a and b")
    _git(repo, "branch", "-m", first["branch"], "mine/union")
    combine_branch = sandbox._TEST_UNION_BRANCH.format(branch="mine/union")
    tip_list = []
    for _ in range(2):
        again = sandbox.integrate(str(repo), str(out), [], resume="mine/union",
                                  land=True, request="add a and b")
        assert sandbox.open_stage_key(again) == "test_create"
        assert f"combined on {combine_branch} and" in again["test_create"]["why"]
        tip_list.append(sandbox.branch_tip(str(repo), combine_branch))
        assert _git(repo, "rev-parse", f"{combine_branch}^").strip() == \
            sandbox.branch_tip(str(repo), "mine/union")
    assert tip_list[0] and tip_list[1]


def test_carrying_on_with_the_stages_takes_away_the_task_branches_the_union_reaches(
        tmp_path, monkeypatch):
    """A run stopped while a stage was going left the task branches its union reaches.
    The run that carries on and gets through leaves none of those the process that
    made the union made, and leaves every other branch where it is: one the union
    does not reach, one of another process, one under a name this tool did not make
    up and the branch of a tree."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch, open_path=["a.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    process = sandbox._run_process(sandbox.run_in(first["branch"]))
    task_list = [f"codetwine-loom/run/20260101-000000-1-{process}/1",
                 f"codetwine-loom/run/20260101-000000-1-{process}/2"]
    for task_branch, side in zip(task_list, ("codetwine-loom/a", "codetwine-loom/b")):
        _git(repo, "branch", task_branch, side)
    other = f"codetwine-loom/run/20260101-000001-1-{process}/1"
    _commit_on_new_branch(repo, other, "main", "edit: something else")
    tree = f"codetwine-loom/run/20260101-000002-1-{process}/tree"
    _git(repo, "branch", tree, "main")
    assert sandbox.is_tree_branch(tree)
    stay_list = [f"codetwine-loom/run/20260101-000000-1-{process}0/1",
                 "codetwine-loom/run/keep-this", "codetwine-loom/run/review/task-1"]
    for stay_branch in stay_list:
        _git(repo, "branch", stay_branch, "codetwine-loom/a")
    _test_stage_container(monkeypatch)
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True)
    assert again["land"] and not sandbox.open_stage_key(again)
    assert sorted(_branches(repo)) == sorted(["main", "codetwine-loom/a", "codetwine-loom/b",
                                              other, tree, *stay_list])


def test_the_task_branches_of_a_union_this_tool_did_not_name_are_none(_dispatch_repo):
    repo, _ = _dispatch_repo
    _git(repo, "branch", "codetwine-loom/run/20260101-000000-1-7/1", "main")
    assert sandbox._union_task_branch_list(str(repo), "feature/union") == []
    assert sandbox._union_task_branch_list(
        str(repo), "codetwine-loom/sandbox/20260101-000000-5-7") == [
            "codetwine-loom/run/20260101-000000-1-7/1"]
    assert sandbox._union_task_branch_list(
        str(repo), "codetwine-loom/sandbox/20260101-000000-5-8") == []


def test_task_branches_stay_while_a_stage_carried_on_with_is_still_open(tmp_path,
                                                                       monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    _test_stage_container(monkeypatch, open_path=["a.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    task_branch = "codetwine-loom/run/20260101-000000-1-7/1"
    _git(repo, "branch", task_branch, "codetwine-loom/a")
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True)
    assert sandbox.open_stage_key(again) == "test_create"
    assert task_branch in _branches(repo) and first["branch"] in _branches(repo)


@pytest.mark.parametrize("meta_status, status", [(1, 0), (0, 1), (0, 137)])
def test_a_merge_whose_container_and_summary_end_differently_is_one_that_did_not_get_through(
        tmp_path, monkeypatch, meta_status, status):
    """The summary beside the patch and the container name different exit statuses
    for a merge that said it settled: no stage follows, nothing goes into the working
    tree, the branch stays and is what the run is carried on from."""
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    run_list = _test_stage_container(monkeypatch)
    start = sandbox._start_container

    def start_merge(*arg_list, **kwargs):
        engine, image, _ = start(*arg_list, **kwargs)
        run_file = arg_list[8]
        pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(
            json.dumps({"exit_status": meta_status}), encoding="utf-8")
        return engine, image, status

    monkeypatch.setattr(sandbox, "_start_container", start_merge)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    want_status = meta_status or status
    assert result["exit_status"] == want_status and result["land"] is None
    assert _kind_list(run_list) == ["merge"]
    assert f"the merge was not settled ({sandbox.status_text(want_status)}), so " \
        in result["land_why"]
    assert ("SANDBOX_MEMORY" in result["land_why"]) == (want_status == 137)
    assert result["branch"] in _branches(repo) and not _in_tree(repo, "a.txt")
    assert sandbox.resume_branch_list([], result, sandbox.union_status(result)) == [
        result["branch"]]
    assert f"Codetwine-Loom-Exit-Status: {want_status}" in _git(
        repo, "log", "-1", "--format=%(trailers)", result["branch"])
    assert sandbox.open_branch_list(str(repo), [result["branch"]]) == [result["branch"]]


@pytest.mark.parametrize("block_list, tail", [
    (["stop"], ". No merge over this HEAD is settled while the tests run over HEAD stop "
               "before every test has run: let the test command run every test when one "
               "fails, or make the tests that fail over HEAD pass, then carry on from "),
    (["stop", "short", "other"],
     ". No merge over this HEAD is settled while the tests run over HEAD stop before every "
     "test has run and the tests run over HEAD do not print how many tests they pass and "
     "fail (or a test ends their process before they print it), or do not name each "
     "test they pass and fail: let the test command run every test when one fails, or "
     "make the tests that fail over HEAD pass; have the test command print how many "
     "tests it passes and fails (for pytest, one -q at most, no plugin or conftest.py "
     "hook that takes that line away, and no test that ends the test process) and the "
     "name of each test it passes and fails, then carry on from "),
    ([], "")])
def test_a_merge_head_s_run_of_the_tests_keeps_unsettled_says_what_to_change(
        tmp_path, monkeypatch, block_list, tail):
    repo, out = _two_branch_repo(tmp_path)
    _test_stage_container(monkeypatch)
    start = sandbox._start_container

    def start_merge(*arg_list, **kwargs):
        engine, image, _ = start(*arg_list, **kwargs)
        run_file = arg_list[8]
        pathlib.Path(run_file.outcome).write_text(json.dumps(
            {"verdict": "unsettled", "why": "- `t` stopped", "head_block": block_list}),
            encoding="utf-8")
        pathlib.Path(sandbox.meta_path(run_file.patch)).write_text(
            json.dumps({"exit_status": 1}), encoding="utf-8")
        return engine, image, 1

    monkeypatch.setattr(sandbox, "_start_container", start_merge)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert result.get("head_block", []) == [b for b in block_list if b != "other"]
    assert result["land_why"] == (
        f"the merge was not settled (exit status 1), so nothing was put into the working "
        f"tree. What it left is on {result['branch']}: - `t` stopped"
        f"{tail}{result['branch'] if tail else ''}")


@pytest.mark.parametrize("outcome_dict, status", [
    (None, 137), ({"verdict": "settled", "why": ""}, 1), ({}, 1)])
def test_the_test_tasks_combined_by_a_run_that_did_not_settle_them_name_no_stage(
        tmp_path, monkeypatch, outcome_dict, status):
    """The run combining the test tasks left a patch and ended with another status
    than 0, with no account of itself or with one that says settled: its commit is no
    commit of a stage, names no file as open and carries no verdict but the run's."""
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, target=("a.txt", "b.txt", "c.txt"),
                          open_path=["c.txt"])
    _combine_left_unsettled(monkeypatch, outcome_dict, status)
    result_dict = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                                    land=True, request="add a and b")
    create_dict = result_dict["test_create"]
    assert create_dict["verdict"] == "unsettled" and create_dict["commit"] is None
    combine_branch = create_dict["why"].split("combined on ")[1].split(" and ")[0]
    assert create_dict["why"].endswith(
        f" and the merge was not settled ({sandbox.status_text(status)}): the run said "
        f"nothing about why")
    trailer_text = _git(repo, "log", "-1", "--format=%(trailers)", combine_branch)
    assert f"Codetwine-Loom-Exit-Status: {status}" in trailer_text
    for name in ("Codetwine-Loom-Stage", "Codetwine-Loom-Test-Open", "Codetwine-Loom-Why",
                 "Codetwine-Loom-Gap-Done", "Codetwine-Loom-Verdict: unsettled"):
        assert name not in trailer_text
    assert ("Codetwine-Loom-Verdict" in trailer_text) is bool(outcome_dict)
    assert _git(repo, "log", "-1", "--format=%s", combine_branch).startswith("integrate:")
    if not outcome_dict:
        assert sandbox.stage_resume_dict(str(repo), combine_branch) == {}
    assert sandbox.stage_resume_dict(str(repo), result_dict["branch"])["stage"] == \
        "test-create"


@pytest.mark.parametrize("outcome_dict, status, meta_status, is_whole", [
    ({"verdict": "settled"}, 0, 0, True), ({"verdict": "settled"}, 0, None, True),
    ({"verdict": "settled"}, 1, 0, False), ({"verdict": "settled"}, 0, 137, False),
    ({"verdict": "settled"}, 0, "x", False), ({"verdict": "unsettled"}, 0, 0, False),
    ({"verdict": ""}, 0, 0, False), ({}, 0, 0, False), ({}, 137, 137, False)])
def test_only_a_merge_the_run_settled_records_what_it_was_to_record(
        outcome_dict, status, meta_status, is_whole):
    record_dict = {"stage": "test-create", "test_open": ["c.txt"], "verdict": "unsettled",
                   "why": "1 test task(s) were not settled", "gap_done": [{"what": "c"}],
                   "gap": [{"what": "d"}], "tests_changed": ["test_a.py"]}
    meta_dict = {} if meta_status is None else {"exit_status": meta_status}
    assert sandbox._merge_record_dict(record_dict, outcome_dict, status, meta_dict) == (
        record_dict if is_whole else {"gap": [{"what": "d"}],
                                      "tests_changed": ["test_a.py"]})


@pytest.mark.parametrize("key, value", [("stage", "test-create"), ("test_open", ["c.txt"])])
def test_what_a_container_recorded_names_no_stage_and_no_open_file(tmp_path, key, value):
    """Which stage a commit is of and which files it leaves open are what this machine
    says; a container's own account does not."""
    for kind in sandbox._RECORD_KIND_DICT:
        record_dict = sandbox._record_dict_of({"why": "left", key: value},
                                              str(tmp_path / "outcome.json"), kind)
        assert record_dict == ({} if kind == "target" else {"why": "left"})
        trailer_dict = sandbox.run_trailer_dict({}, record_dict)
        assert trailer_dict["Codetwine-Loom-Stage"] is None \
            and trailer_dict["Codetwine-Loom-Test-Open"] == ""


# Every key that reaches a commit's trailers, as a run of any kind could write it, and
# keys that reach none
_EVERY_RECORD_DICT = {
    "n": 3, "why": "w", "did": "d", "untooled": ["u.py"], "tests_changed": ["t.py"],
    "ended": "complete", "decisions": [{"what": "a", "why": "b"}],
    "separator": {"a.txt": {"=======": 1}}, "no_merge": [["a" * 40, "a.txt", "b.txt"]],
    "stage": "refactor",
    "non_test_changed": ["n.py"], "tests_base": "red", "test_open": ["o.py"],
    "gap": [{"what": "case", "why": "not done"}],
    "gap_done": [{"what": "other", "why": "done"}], "stand_on": "a" * 40,
    "skipped": "", "parents": ["p"], "request": "r", "at": "2026-10-13T00:00:00"}

# The keys of _EVERY_RECORD_DICT that reach no trailer
_NO_TRAILER_KEY_LIST = ["skipped", "parents", "request", "at"]


@pytest.mark.parametrize("kind, verdict, key_list", [
    ("edit", "failed", ["n", "verdict", "why", "did", "untooled", "tests_changed"]),
    ("merge", "settled", ["verdict", "why", "did", "tests_changed", "ended", "decisions",
                          "separator", "no_merge"]),
    ("test-merge", "unsettled", ["verdict", "why", "did", "tests_changed", "ended",
                                 "decisions", "separator", "no_merge", "gap", "gap_done"]),
    ("test-create", "settled", ["verdict", "why", "did", "untooled", "tests_changed",
                                "non_test_changed", "ended", "decisions", "tests_base",
                                "gap"]),
    ("refactor", "settled", ["verdict", "why", "did", "untooled", "tests_changed",
                             "non_test_changed", "ended", "decisions"]),
    ("test-refactor", "unsettled", ["verdict", "why", "did", "untooled", "tests_changed",
                                    "non_test_changed", "ended", "decisions"]),
    ("target", "settled", [])])
def test_what_a_run_recorded_is_read_by_the_keys_its_kind_of_run_records(
        tmp_path, kind, verdict, key_list):
    """A container writes the record, so what one kind of run does not record is not
    read out of what a run of that kind wrote."""
    path = str(tmp_path / "record.json")
    every_dict = {**_EVERY_RECORD_DICT, "verdict": verdict}
    record_dict = sandbox._record_dict_of(every_dict, path, kind)
    assert record_dict == {key: every_dict[key] for key in every_dict
                           if key in key_list + _NO_TRAILER_KEY_LIST}
    assert set(sandbox._TRAILER_KEY_TUPLE) == set(every_dict) - set(_NO_TRAILER_KEY_LIST)
    # A value of another shape under a key that is not read does not keep the rest
    # from being read; one under a key that is read does
    for key in sandbox._RECORD_SHAPE_DICT:
        if key not in sandbox._TRAILER_KEY_TUPLE:
            continue
        wrong_dict = sandbox._record_dict_of({**every_dict, key: {"x": 5}}, path, kind)
        assert wrong_dict == ({} if key in key_list else record_dict), key


def test_every_key_a_commit_s_trailers_are_written_from_is_one_a_kind_of_run_is_held_to():
    """A key run_trailer_dict() starts to read is read from no run's own record until a
    kind of run is said to record it."""
    seen_set = set()

    class Seen(dict):
        def get(self, key, default=None):
            seen_set.add(key)
            return default

    sandbox.run_trailer_dict({}, Seen(), "")
    assert seen_set == set(sandbox._TRAILER_KEY_TUPLE)
    for key_tuple, _ in sandbox._RECORD_KIND_DICT.values():
        assert set(key_tuple) <= seen_set - {"stage", "test_open"}
    assert set(sandbox._RECORD_KIND_DICT) >= set(integrate_mod.STAGE_TUPLE)


@pytest.mark.parametrize("kind, verdict, is_read", [
    ("edit", "failed", True), ("edit", "unverified", True), ("edit", "tests_red", True),
    ("edit", "no_change", True), ("edit", "", True), ("edit", "settled", False),
    ("edit", "unsettled", False), ("edit", "passed", False), ("edit", "Settled", False),
    ("merge", "settled", True), ("merge", "unsettled", True), ("merge", "", True),
    ("merge", "failed", False), ("merge", "unverified", False),
    ("test-merge", "passed", False), ("test-create", "failed", False),
    ("refactor", "unsettled", True), ("refactor", "no_change", False),
    ("test-refactor", "settled", True), ("test-refactor", "tests_red", False)])
def test_a_verdict_a_run_of_its_kind_does_not_record_is_a_value_of_another_shape(
        tmp_path, caplog, kind, verdict, is_read):
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        record_dict = sandbox._record_dict_of({"verdict": verdict, "why": "w"},
                                              str(tmp_path / "record.json"), kind)
    if is_read:
        assert record_dict == {"verdict": verdict, "why": "w"}
    else:
        assert record_dict == {} and "another shape under verdict" in caplog.text


def test_a_change_run_s_commit_records_no_verdict_and_no_case_of_a_merge_s(
        tmp_path, monkeypatch):
    """The container of a change run wrote its history, a merge's verdict and a case
    the tests found among it."""
    entry_dict = {**_attempt(1, verdict="unverified", why="left"),
                  "gap": [{"what": "case_9", "why": "written by the container"}],
                  "gap_done": [{"what": "case_1", "why": "x"}], "ended": "complete",
                  "separator": {"a.txt": {"=======": 3}}, "tests_base": "red",
                  "stage": "refactor", "test_open": ["a.txt"]}
    _, _, message = _committed(tmp_path / "one", monkeypatch, "do it", [entry_dict])
    line_list = integrate_mod.record_line_list(message)
    assert [line.partition(":")[0] for line in line_list] == [
        "Codetwine-Loom-Run-Request", "Codetwine-Loom-Attempt", "Codetwine-Loom-Verdict",
        "Codetwine-Loom-Why", "Codetwine-Loom-Did", "Codetwine-Loom-Exit-Status"]
    assert "Codetwine-Loom-Verdict: unverified" in line_list
    _, _, message = _committed(tmp_path / "two", monkeypatch, "do it",
                               [_attempt(1, verdict="settled", why="left")])
    assert [line.partition(":")[0] for line in integrate_mod.record_line_list(message)] == [
        "Codetwine-Loom-Run-Request", "Codetwine-Loom-Exit-Status"]


def test_open_files_whose_names_hold_spaces_are_the_ones_written_again(tmp_path,
                                                                      monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 1)
    run_list = _test_stage_container(monkeypatch, target=("b.txt", "my  a.txt"),
                                     open_path=["my  a.txt"], no_patch=True)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    test_run_list = [entry for entry in run_list if entry[0] == "tests"]
    assert [entry[3] for entry in test_run_list[2:]] == [["my  a.txt"]]
    assert sandbox.open_stage_key(result) == "test_create"
    assert sandbox.stage_resume_dict(str(repo), result["branch"])["paths"] == ["my  a.txt"]


@pytest.mark.parametrize("listing_dict, why", [
    ({"target": ("a.txt",)}, "none of them is among the files the change touched"),
    ({"skipped": "the tests could not be run"}, "the tests could not be run")])
def test_open_files_the_listing_has_nothing_for_leave_the_tests_unsettled(
        tmp_path, monkeypatch, listing_dict, why):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    _test_stage_container(monkeypatch, open_path=["b.txt"])
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    keep_commit = first["test_create"]["commit"]
    run_list = _test_stage_container(monkeypatch, **listing_dict)
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True)
    # No test task is started, and the stage is not taken for one with nothing to write
    assert _kind_list(run_list) == ["list"]
    create_dict = again["test_create"]
    assert create_dict["verdict"] == "unsettled" and not create_dict["skipped"]
    assert create_dict["why"] == ("no test was written for the files left to write "
                                  f"tests for (b.txt): {why}")
    assert create_dict["open_paths"] == ["b.txt"]
    assert sandbox.open_stage_key(again) == "test_create"
    assert _git(repo, "rev-parse", first["branch"]).strip() == keep_commit
    assert sandbox.stage_resume_dict(str(repo), first["branch"])["paths"] == ["b.txt"]


def test_open_files_that_cannot_be_read_off_the_commit_have_the_tests_written_again(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    _git(repo, "checkout", "--quiet", result["branch"])
    _git(repo, "commit", "--quiet", "--allow-empty", "-m",
         "test-create: x\n\nx\n\nCodetwine-Loom-Verdict: unsettled\n"
         "Codetwine-Loom-Stage: test-create\nCodetwine-Loom-Test-Open: src/my lib/a.py")
    tip = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "checkout", "--quiet", "main")
    resume_dict = sandbox.stage_resume_dict(str(repo), result["branch"])
    assert resume_dict["stage"] == "test-create" and resume_dict["paths"] == []
    assert resume_dict["union"] == tip and resume_dict["settled"] == result["commit"]


def test_the_container_is_asked_to_refactor_against_the_base():
    argv = sandbox.refactor_argv("repo", "test-refactor", "a" * 40, "足して",
                                 "/out/repo/sandbox/o.json")
    assert argv[0] == "test-refactor" and f"{sandbox._WORK}/repo" in argv
    assert argv[argv.index("--base") + 1] == "a" * 40
    assert "--request=足して" in argv
    assert argv[argv.index("--outcome-out") + 1] == "/out/repo/sandbox/o.json"
    # Each stage's own instruction file is named by a setting the container is
    # forwarded by name, so nothing on the command line names it
    bare = sandbox.refactor_argv("repo", "refactor", "a" * 40)
    assert bare[0] == "refactor" and "--guide" not in bare
    assert not any(arg.startswith("--request") for arg in bare)
    assert "--outcome-out" not in bare


def test_the_stages_follow_one_another_each_over_what_the_one_before_landed(tmp_path,
                                                                           monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    run_list = _test_stage_container(monkeypatch)
    main_before = _git(repo, "rev-parse", "main").strip()
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True, request="add a and b")
    assert _kind_list(run_list) == ["merge", "list", "tests", "refactor", "test-refactor"]
    commit_list = [result["commit"], result["test_create"]["commit"],
                   result["refactor"]["commit"], result["test_refactor"]["commit"]]
    assert [run[2] for run in run_list[2:]] == commit_list[:3]
    for _, argv, _, _ in run_list[3:]:
        assert argv[argv.index("--base") + 1] == main_before
    # One commit per stage, each on top of the one before, and the last one is what
    # goes into the tree
    head = result["test_refactor"]["commit"]
    assert _git(repo, "rev-list", "--first-parent", "-n", "4", head).split() == \
        commit_list[::-1]
    assert result["settle_commit"] == head and not sandbox.open_stage_key(result)
    assert result["land"]["files"] == 4
    assert (repo / "a.txt").read_text(encoding="utf-8") == "y\n"
    assert (repo / "tests" / "test_a.txt").read_text(encoding="utf-8") == "u\n"
    for commit, stage in zip(commit_list[2:], ("refactor", "test-refactor")):
        message = _git(repo, "log", "-1", "--format=%B", commit)
        assert message.startswith(f"{stage}: add a and b")
        assert f"Codetwine-Loom-Stage: {stage}" in message \
            and "Codetwine-Loom-Verdict: settled" in message
    assert "Codetwine-Loom-Non-Test-Changed: a.txt" in _git(repo, "log", "-1", "--format=%B",
                                                 commit_list[2])
    assert "Codetwine-Loom-Tests-Changed: tests/test_a.txt" in _git(
        repo, "log", "-1", "--format=%B", commit_list[3])
    # The name this tool made up is dropped once everything under it is in the tree
    assert _branch_gone(repo, result["branch"])


def test_a_stage_that_changes_nothing_lands_a_commit_holding_its_record_alone(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    run_list = _test_stage_container(monkeypatch, refactor_dict={"refactor": "nothing"})
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert _kind_list(run_list)[-2:] == ["refactor", "test-refactor"]
    refactor_dict = result["refactor"]
    assert refactor_dict["verdict"] == "settled" and refactor_dict["patch"] is None
    assert _same_tree(repo, refactor_dict["commit"], result["test_create"]["commit"])
    message = _git(repo, "log", "-1", "--format=%B", refactor_dict["commit"])
    assert "Codetwine-Loom-Stage: refactor" in message \
        and "Codetwine-Loom-Verdict: settled" in message
    # The next stage starts at that commit
    assert run_list[-1][2] == refactor_dict["commit"]
    assert result["settle_commit"] == result["test_refactor"]["commit"]
    assert (repo / "tests" / "test_a.txt").read_text(encoding="utf-8") == "u\n"


def test_a_stage_with_nothing_to_refactor_is_not_left_open(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    _test_stage_container(monkeypatch, refactor_dict={"refactor": "skipped",
                                                      "test-refactor": "skipped"})
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert result["refactor"]["skipped"] and not sandbox.open_stage_key(result)
    # Each stage that called no model is on the branch as a commit that changes nothing
    assert _same_tree(repo, result["refactor"]["commit"], result["test_create"]["commit"])
    assert _same_tree(repo, result["test_refactor"]["commit"], result["refactor"]["commit"])
    assert result["settle_commit"] == result["test_refactor"]["commit"]
    assert result["land"]["files"] == 4 and _branch_gone(repo, result["branch"])


def test_a_refactoring_nobody_settled_stays_on_the_branch_and_ends_the_stages(
        tmp_path, monkeypatch, caplog):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    run_list = _test_stage_container(monkeypatch, refactor_dict={"refactor": "unsettled"})
    with caplog.at_level(logging.WARNING):
        result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                                   land=True)
    assert _kind_list(run_list) == ["merge", "list", "tests", "refactor"]
    assert sandbox.open_stage_key(result) == "refactor" and "test_refactor" not in result
    # What goes in is the last commit a stage settled, and the rest is left to be read
    test_commit = result["test_create"]["commit"]
    assert result["settle_commit"] == test_commit and result["land"]["files"] == 4
    assert (repo / "a.txt").read_text(encoding="utf-8") == "x\n"
    assert _git(repo, "rev-parse", result["branch"]).strip() == result["refactor"]["commit"]
    assert "the refactoring was not settled" in caplog.text
    assert result["branch"] in caplog.text and result["exit_status"] == 0
    assert sandbox.resume_branch_list([], result, 0) == [result["branch"]]
    assert sandbox.stage_resume_dict(str(repo), result["branch"]) == {
        "stage": "refactor", "union": test_commit, "settled": test_commit,
        "merge": result["commit"],
        "base": _git(repo, "rev-parse", f"{result['commit']}^").strip(), "paths": []}


def test_tests_nobody_settled_start_no_stage_that_refactors(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    run_list = _test_stage_container(monkeypatch, open_path=["a.txt"])
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert _kind_list(run_list) == ["merge", "list", "tests"]
    assert sandbox.open_stage_key(result) == "test_create" and "refactor" not in result
    assert result["land"]["files"] == 2 and not _in_tree(repo, "tests/test_a.txt")


def test_the_change_is_refactored_over_the_union_where_no_tests_are_written(tmp_path,
                                                                           monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch, test_create=False)
    run_list = _test_stage_container(monkeypatch,
                                     refactor_dict={"test-refactor": "skipped"})
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert _kind_list(run_list) == ["merge", "refactor", "test-refactor"]
    assert run_list[1][2] == result["commit"] and "test_create" not in result
    assert result["settle_commit"] == result["test_refactor"]["commit"]
    assert _same_tree(repo, result["settle_commit"], result["refactor"]["commit"])
    assert (repo / "a.txt").read_text(encoding="utf-8") == "y\n"


def test_carrying_on_starts_with_the_stage_that_was_left_open(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    _test_stage_container(monkeypatch, refactor_dict={"refactor": "unsettled"})
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True, request="add a and b")
    run_list = _test_stage_container(monkeypatch)
    again = sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True,
                              request="add a and b")
    assert _kind_list(run_list) == ["refactor", "test-refactor"]
    test_commit = first["test_create"]["commit"]
    assert run_list[0][2] == test_commit
    assert "test_create" not in again and not sandbox.open_stage_key(again)
    tip = again["test_refactor"]["commit"]
    assert _git(repo, "rev-list", "--first-parent", "-n", "3", tip).split() == \
        [tip, again["refactor"]["commit"], test_commit]
    # The refactoring nobody settled is not under what went in
    assert first["refactor"]["commit"] not in _git(repo, "rev-list", tip).split()
    # What went in now is what the two stages changed over the tests put in before
    assert again["land"]["files"] == 2
    assert (repo / "a.txt").read_text(encoding="utf-8") == "y\n"
    assert (repo / "tests" / "test_a.txt").read_text(encoding="utf-8") == "u\n"
    assert _branch_gone(repo, first["branch"])


@pytest.mark.parametrize("before", ["skipped", "nothing"])
def test_a_stage_left_open_over_stages_that_wrote_nothing_is_the_one_carried_on_with(
        tmp_path, monkeypatch, before):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    _test_stage_container(monkeypatch, refactor_dict={"refactor": before,
                                                      "test-refactor": "unsettled"})
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    resume_dict = sandbox.stage_resume_dict(str(repo), result["branch"])
    assert resume_dict["stage"] == "test-refactor"
    assert resume_dict["union"] == result["refactor"]["commit"]


@pytest.mark.parametrize("kind, stage_key, land_count", [
    ("list", "test_create", 2), ("tests", "test_create", 2),
    ("refactor", "refactor", 4), ("test-refactor", "test_refactor", 4)])
def test_a_stage_whose_container_left_its_account_behind_a_link_is_left_unsettled(
        tmp_path, monkeypatch, kind, stage_key, land_count):
    """The container of a stage replaced what it hands back by a link that leads out
    of its copy of the output directory: the stage is the one left open, and the
    settled union and the stages settled before it reach the working tree."""
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 0)
    run_list = _test_stage_container(monkeypatch)
    start = sandbox._start_container
    outside = tmp_path / "outside.json"
    outside.write_text('{"targets": {"a.txt": ["f"]}, "verdict": "settled"}',
                       encoding="utf-8")

    def start_and_link(*arg_list, **kwargs):
        answer = start(*arg_list, **kwargs)
        if run_list[-1][0] == kind:
            os.remove(arg_list[8].outcome)
            os.symlink(outside, arg_list[8].outcome)
        return answer

    monkeypatch.setattr(sandbox, "_start_container", start_and_link)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert sandbox.open_stage_key(result) == stage_key
    assert "is reached through a link that leads out of" in result[stage_key]["why"]
    assert result["exit_status"] == 0 and result["land"]["files"] == land_count
    assert result["branch"] in _branches(repo)


def test_a_stage_left_open_over_stages_that_wrote_nothing_is_run_again(tmp_path,
                                                                      monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 1)
    run_list = _test_stage_container(monkeypatch, refactor_dict={
        "refactor": "skipped", "test-refactor": "unsettled"})
    sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"], land=True)
    assert _kind_list(run_list) == ["merge", "list", "tests", "refactor", "test-refactor",
                                    "test-refactor"]


def _fail_container_of(monkeypatch, stage, fail_count, lose_outcome=False, status=1):
    """Make the first fail_count containers of one stage end badly.

    With lose_outcome the container writes its patch and no account of how it ended,
    and ends with status; without, it is not started at all.

    Returns:
        [the commit each container of that stage was started at, ...].
    """
    start = sandbox._start_container
    start_commit_list = []

    def start_or_fail(*arg_list, **kwargs):
        if arg_list[10][0] != stage or "--target-only" in arg_list[10]:
            return start(*arg_list, **kwargs)
        start_commit_list.append(arg_list[7])
        if len(start_commit_list) > fail_count:
            return start(*arg_list, **kwargs)
        if not lose_outcome:
            raise RuntimeError("the engine could not start the container")
        engine, image, _ = start(*arg_list, **kwargs)
        os.remove(arg_list[8].outcome)
        return engine, image, status

    monkeypatch.setattr(sandbox, "_start_container", start_or_fail)
    return start_commit_list


@pytest.mark.parametrize("kind, stage_key, land_count", [
    ("list", "test_create", 2), ("combine", "test_create", 2), ("refactor", "refactor", 4),
    ("test-refactor", "test_refactor", 4)])
@pytest.mark.parametrize("error, why", [
    (RuntimeError("the engine could not start the container"),
     "the engine could not start the container"),
    (OSError(28, "No space left on device"),
     "OSError: [Errno 28] No space left on device")])
def test_a_stage_that_raises_while_it_is_started_is_left_unsettled_and_the_rest_is_put_in(
        tmp_path, monkeypatch, kind, stage_key, land_count, error, why):
    """Whatever starting a stage's container raises, the settled union and the stages
    settled before it reach the working tree, and the stage is the one left open."""
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 0)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    run_list = _test_stage_container(monkeypatch)
    start = sandbox._start_container

    def start_or_raise(*arg_list, **kwargs):
        argv_tail = arg_list[10]
        own_kind = "list" if "--target-only" in argv_tail else "combine" \
            if argv_tail[0] == "integrate" and "-test/" in " ".join(argv_tail) \
            else argv_tail[0]
        if own_kind == kind:
            raise error
        return start(*arg_list, **kwargs)

    monkeypatch.setattr(sandbox, "_start_container", start_or_raise)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert sandbox.open_stage_key(result) == stage_key
    assert why in result[stage_key]["why"]
    assert result["exit_status"] == 0 and result["land"]["files"] == land_count
    assert _kind_list(run_list)[0] == "merge" and kind not in _kind_list(run_list)


@pytest.mark.parametrize("skipped, refactor_dict, stage", [
    ("the repository holds no test file", {"test-refactor": "skipped"}, "refactor"),
    ("", {"refactor": "nothing"}, "test-refactor"),
    ("", {"refactor": "skipped"}, "test-refactor")])
def test_a_stage_whose_container_did_not_start_is_run_again_over_stages_that_wrote_nothing(
        tmp_path, monkeypatch, skipped, refactor_dict, stage):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 1)
    run_list = _test_stage_container(monkeypatch, skipped=skipped,
                                     refactor_dict=refactor_dict)
    start_commit_list = _fail_container_of(monkeypatch, stage, 1)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    # The stage is started twice, both times over what the stage before it left
    before_key = {"refactor": "test_create", "test-refactor": "refactor"}[stage]
    assert start_commit_list == [result[before_key]["commit"]] * 2
    assert _kind_list(run_list).count(stage) == 1
    assert not sandbox.open_stage_key(result) and _branch_gone(repo, result["branch"])


@pytest.mark.parametrize("skipped, refactor_dict, stage", [
    ("the repository holds no test file", {"test-refactor": "skipped"}, "refactor"),
    ("", {"refactor": "nothing"}, "test-refactor")])
def test_a_stage_that_left_no_commit_is_the_one_carried_on_with_and_none_before_it(
        tmp_path, monkeypatch, skipped, refactor_dict, stage):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 0)
    _test_stage_container(monkeypatch, skipped=skipped, refactor_dict=refactor_dict)
    _fail_container_of(monkeypatch, stage, 1)
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    assert sandbox.open_stage_key(first) == integrate_mod.stage_key(stage)
    assert first[integrate_mod.stage_key(stage)]["commit"] is None
    resume_dict = sandbox.stage_resume_dict(str(repo), first["branch"])
    assert resume_dict["stage"] == stage
    assert resume_dict["union"] == _git(repo, "rev-parse", first["branch"]).strip()
    run_list = _test_stage_container(monkeypatch, refactor_dict=refactor_dict)
    sandbox.integrate(str(repo), str(out), [], resume=first["branch"], land=True)
    assert _kind_list(run_list)[0] == stage and "tests" not in _kind_list(run_list)


def test_a_stage_whose_container_left_no_account_lands_as_an_unsettled_commit_of_its_own(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 0)
    _test_stage_container(monkeypatch)
    _fail_container_of(monkeypatch, "refactor", 1, lose_outcome=True)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    refactor_dict = result["refactor"]
    assert refactor_dict["verdict"] == "unsettled"
    assert refactor_dict["why"] == "the run ended with exit status 1 and left no account"
    message = _git(repo, "log", "-1", "--format=%B", refactor_dict["commit"])
    assert "Codetwine-Loom-Stage: refactor" in message \
        and "Codetwine-Loom-Verdict: unsettled" in message
    assert "Codetwine-Loom-Why: the run ended with exit status 1 and left no account" in message
    # Carrying on starts that stage again over the tests, and the merge is not made again
    test_commit = result["test_create"]["commit"]
    assert sandbox.stage_resume_dict(str(repo), result["branch"])["stage"] == "refactor"
    assert sandbox.stage_resume_dict(str(repo), result["branch"])["union"] == test_commit
    assert sandbox.resume_head(str(repo), result["branch"]) == (
        test_commit, _git(repo, "rev-parse", f"{result['commit']}^").strip())


def test_a_stage_whose_container_left_no_account_is_run_again(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 1)
    _test_stage_container(monkeypatch)
    start_commit_list = _fail_container_of(monkeypatch, "refactor", 1,
                                           lose_outcome=True)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               land=True)
    assert start_commit_list == [result["test_create"]["commit"]] * 2
    assert result["refactor"]["verdict"] == "settled"
    assert not sandbox.open_stage_key(result)


def test_a_stage_whose_container_was_killed_at_the_memory_limit_is_not_run_again(
        tmp_path, monkeypatch, caplog):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    monkeypatch.setattr(config, "REFACTOR_MAX_RERUN", 1)
    _test_stage_container(monkeypatch)
    start_commit_list = _fail_container_of(monkeypatch, "refactor", 1,
                                           lose_outcome=True, status=137)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        result = sandbox.integrate(str(repo), str(out),
                                   ["codetwine-loom/a", "codetwine-loom/b"], land=True)
    assert len(start_commit_list) == 1
    assert result["refactor"]["verdict"] == "unsettled"
    assert result["refactor"]["memory_kill"]
    assert "the refactoring did not settle (exit status 137" in caplog.text


def test_a_test_task_killed_at_the_memory_limit_is_neither_handed_out_nor_run_again(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 1)
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 1)
    monkeypatch.setattr(config, "DISPATCH_PARALLEL", 1)
    _test_stage_container(monkeypatch)
    start_commit_list = _fail_container_of(monkeypatch, "test-create", 1,
                                           lose_outcome=True, status=137)
    result = sandbox.integrate(str(repo), str(out),
                               ["codetwine-loom/a", "codetwine-loom/b"])
    create_dict = result["test_create"]
    assert len(start_commit_list) == 2
    assert create_dict["verdict"] == "unsettled" and create_dict["memory_kill"]
    assert [e["attempt"] for e in create_dict["test_tasks"]] == [1, 1]


def test_tests_whose_combining_run_was_killed_at_the_memory_limit_are_not_written_again(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 1)
    run_list = _test_stage_container(monkeypatch)
    start = sandbox._start_container

    def kill_the_combine(*arg_list, **kwargs):
        engine, image, status = start(*arg_list, **kwargs)
        if arg_list[10][0] == "integrate" and "-test/" in " ".join(arg_list[10]):
            os.remove(arg_list[8].outcome)
            return engine, image, 137
        return engine, image, status

    monkeypatch.setattr(sandbox, "_start_container", kill_the_combine)
    result = sandbox.integrate(str(repo), str(out),
                               ["codetwine-loom/a", "codetwine-loom/b"])
    create_dict = result["test_create"]
    assert [run[0] for run in run_list].count("list") == 1
    assert create_dict["verdict"] == "unsettled" and create_dict["memory_kill"]


def test_tests_left_open_beside_a_test_task_killed_at_the_memory_limit_are_written_again(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 1)
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_PARALLEL", 1)
    run_list = _test_stage_container(monkeypatch, open_path=("b.txt",))
    _fail_container_of(monkeypatch, "test-create", 1, lose_outcome=True, status=137)
    result = sandbox.integrate(str(repo), str(out),
                               ["codetwine-loom/a", "codetwine-loom/b"])
    assert [run[0] for run in run_list].count("tests") > 2
    assert not result["test_create"].get("memory_kill")


def test_a_test_task_whose_container_left_no_account_is_left_open_on_its_own_branch(
        tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 0)
    monkeypatch.setattr(config, "DISPATCH_PARALLEL", 1)
    _test_stage_container(monkeypatch)
    _fail_container_of(monkeypatch, "test-create", 1, lose_outcome=True)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    create_dict = result["test_create"]
    assert create_dict["verdict"] == "unsettled" and len(create_dict["open_paths"]) == 1
    open_entry = next(e for e in create_dict["test_tasks"] if e["verdict"] == "unsettled")
    assert open_entry["why"] == "the run ended with exit status 1 and left no account"
    message = _git(repo, "log", "-1", "--format=%B", open_entry["branch"])
    assert "Codetwine-Loom-Stage: test-create" in message \
        and "Codetwine-Loom-Verdict: unsettled" in message


@pytest.mark.parametrize("gap_path, on_tuple", [
    (["b.txt"], (True, True, True)), (["b.txt"], (True, False, False)),
    ([], (True, True, True))])
def test_a_branch_every_stage_is_through_on_is_carried_on_from_with_a_new_split(
        tmp_path, monkeypatch, gap_path, on_tuple):
    repo, out = _two_branch_repo(tmp_path)
    for name, is_on in zip(("TEST_CREATE", "REFACTOR", "TEST_REFACTOR"), on_tuple):
        monkeypatch.setattr(config, name, is_on)
    _test_stage_container(monkeypatch, gap_path=gap_path, refactor_dict={
        "refactor": "nothing", "test-refactor": "skipped"})
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                               request="add a and b")
    assert sandbox.union_status(result) == (1 if gap_path else 0)
    assert sandbox.stage_resume_dict(str(repo), result["branch"]) == {}


def test_the_head_carried_on_from_passes_over_what_nobody_settled(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    _test_stage_container(monkeypatch, refactor_dict={"refactor": "unsettled"})
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    assert sandbox.resume_head(str(repo), first["branch"]) == (
        first["test_create"]["commit"],
        _git(repo, "rev-parse", f"{first['commit']}^").strip())


@pytest.mark.parametrize("stage_off", [True, False])
def test_carrying_on_another_way_leaves_out_a_stage_nobody_settled(tmp_path, monkeypatch,
                                                                   stage_off):
    """With the stages ahead turned off, or with a branch named besides, the merge is
    carried on from by combining it again, without what the open refactoring wrote."""
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    _test_stage_container(monkeypatch, refactor_dict={"refactor": "unsettled"})
    first = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                              land=True)
    test_commit, open_commit = first["test_create"]["commit"], first["refactor"]["commit"]
    if stage_off:
        _all_stage_on(monkeypatch, test_create=False)
        monkeypatch.setattr(config, "REFACTOR", False)
        monkeypatch.setattr(config, "TEST_REFACTOR", False)
    _test_stage_container(monkeypatch)
    branch_list = [] if stage_off else ["codetwine-loom/a"]
    again = sandbox.integrate(str(repo), str(out), branch_list, resume=first["branch"],
                              land=True)
    assert open_commit not in _git(repo, "rev-list", again["commit"]).split()
    assert test_commit in _git(repo, "rev-list", again["commit"]).split()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "x\n"


def test_a_run_carrying_on_from_an_open_refactoring_hands_out_no_task(tmp_path,
                                                                     monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    _all_stage_on(monkeypatch)
    _test_stage_container(monkeypatch, refactor_dict={"test-refactor": "unsettled"})
    union = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"],
                      request="add a and b")
    run_list = _test_stage_container(monkeypatch)
    result = sandbox.dispatch(str(repo), str(out), "add a and b",
                              resume_list=[union["branch"]])
    assert _kind_list(run_list) == ["test-refactor"]
    assert result["tasks"] == [] and result["resume"] == [] and result["exit_status"] == 0


def test_settled_tests_are_carried_on_from_only_where_a_later_stage_is_on(tmp_path,
                                                                         monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    _test_stage_container(monkeypatch)
    result = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    assert sandbox.stage_resume_dict(str(repo), result["branch"]) == {}
    monkeypatch.setattr(config, "TEST_REFACTOR", True)
    resume_dict = sandbox.stage_resume_dict(str(repo), result["branch"])
    assert resume_dict["stage"] == "test-refactor"
    assert resume_dict["union"] == result["test_create"]["commit"]


def test_open_tests_whose_stage_is_off_are_not_carried_on_from(tmp_path, monkeypatch):
    repo, out = _two_branch_repo(tmp_path)
    monkeypatch.setattr(config, "TEST_CREATE", True)
    monkeypatch.setattr(config, "TEST_CREATE_MAX_DEFINITION", 1)
    _test_stage_container(monkeypatch, open_path=["b.txt"])
    union = sandbox.integrate(str(repo), str(out), ["codetwine-loom/a", "codetwine-loom/b"])
    assert sandbox.stage_resume_dict(str(repo), union["branch"])["paths"] == ["b.txt"]
    monkeypatch.setattr(config, "TEST_CREATE", False)
    monkeypatch.setattr(config, "REFACTOR", True)
    assert sandbox.stage_resume_dict(str(repo), union["branch"]) == {}


@pytest.mark.parametrize("result_dict, target", [
    ({"branch": "codetwine-loom/u", "settle_commit": "c0"}, "codetwine-loom/u"),
    ({"branch": "codetwine-loom/u", "settle_commit": "c0",
      "test_create": {"skipped": "", "verdict": "unsettled"}}, "c0"),
    ({"branch": "codetwine-loom/u", "settle_commit": "c1",
      "test_create": {"skipped": "", "verdict": "settled"},
      "refactor": {"skipped": "", "verdict": "unsettled"}}, "c1"),
    ({"branch": "codetwine-loom/u", "settle_commit": "c2",
      "test_create": {"skipped": "", "verdict": "settled"},
      "refactor": {"skipped": "nothing to refactor", "verdict": ""},
      "test_refactor": {"skipped": "", "verdict": "settled"}}, "codetwine-loom/u")])
def test_what_goes_in_is_the_branch_or_the_last_commit_a_stage_settled(result_dict,
                                                                      target):
    assert sandbox._land_target(result_dict) == target


# == Stopping a run ===========================================

def test_a_stopped_run_starts_no_container(monkeypatch):
    monkeypatch.setattr(sandbox, "_STOP_EVENT", threading.Event())
    sandbox._STOP_EVENT.set()
    monkeypatch.setattr(sandbox.subprocess, "run", _never("a container was started"))
    with pytest.raises(KeyboardInterrupt):
        sandbox.run_the_container(["podman", "run"], "podman", "r1", 0, os.devnull)


def test_a_stop_drops_the_jobs_not_started_and_takes_the_containers_away(monkeypatch):
    monkeypatch.setattr(sandbox, "_STOP_EVENT", threading.Event())
    monkeypatch.setattr(config, "DISPATCH_PARALLEL", 1)
    removed, ran = [], []
    client = _ClientThroughRun(["podman", "run"], None, None)
    client.returncode = 0
    monkeypatch.setattr(sandbox, "_LIVE_CONTAINER_DICT", {"ctloom-r1": ("podman", client)})
    monkeypatch.setattr(sandbox, "_remove_container",
                        lambda engine, name: removed.append((engine, name)) or True)

    def one(n, task, attempt):
        ran.append(n)
        if n == 1:
            raise KeyboardInterrupt
        return {"n": n, "exit_status": 0}

    with pytest.raises(KeyboardInterrupt):
        sandbox._run_agent_batch([(1, "a"), (2, "b"), (3, "c")], 1, one)
    assert ran == [1] and removed == [("podman", "ctloom-r1")]
    assert sandbox._STOP_EVENT.is_set()


def test_a_stop_is_handed_on_once_the_agents_going_have_ended(monkeypatch):
    monkeypatch.setattr(sandbox, "_STOP_EVENT", threading.Event())
    monkeypatch.setattr(config, "DISPATCH_PARALLEL", 2)
    monkeypatch.setattr(sandbox, "_remove_container", lambda engine, name: True)
    ended = []
    going = threading.Event()

    def one(n, task, attempt):
        if n == 1:
            going.wait(10)
            raise KeyboardInterrupt
        # Stands in for an agent whose container is taken away by the stop
        going.set()
        sandbox._STOP_EVENT.wait(10)
        time.sleep(0.3)
        ended.append(n)
        return {"n": n, "exit_status": 0}

    with pytest.raises(KeyboardInterrupt):
        sandbox._run_agent_batch([(1, "a"), (2, "b")], 1, one)
    assert ended == [2]


def _slow_engine(tmp_path, make_sec, run_sec):
    """Write a stand-in for an engine whose "run" takes make_sec to make its container
    and then stays run_sec, whose "rm" ends the run only once the container is made,
    and whose "ps" names the container while it is made and going.

    Returns:
        (the engine's path, the file every command it was given is appended to).
    """
    engine = tmp_path / "podman"
    made, pid, said = (tmp_path / name for name in ("made", "pid", "said"))
    engine.write_text(f"""#!/bin/sh
echo "$1" >> {said}
if [ "$1" = run ]; then
  echo $$ > {pid}; sleep {make_sec}; touch {made}; exec sleep {run_sec}
fi
if [ "$1" = rm ]; then
  if [ -e {made} ] && kill "$(cat {pid})" 2>/dev/null; then exit 0; fi
  echo "no container with that name" >&2; exit 1
fi
if [ "$1" = ps ] && [ -e {made} ] && kill -0 "$(cat {pid})" 2>/dev/null; then echo abc; fi
exit 0
""", encoding="utf-8")
    engine.chmod(0o755)
    return str(engine), said


@pytest.mark.parametrize("make_sec, stop_after_sec", [(1.0, 0.3), (0.2, 1.0)])
def test_a_stop_reaches_the_container_whether_or_not_the_engine_has_made_it(
        tmp_path, monkeypatch, make_sec, stop_after_sec):
    """A container being made when the stop comes has no name to be removed by yet, so
    the engine's client that is making it is ended too."""
    monkeypatch.setattr(sandbox, "_start_client", _START_CLIENT)
    engine, said = _slow_engine(tmp_path, make_sec, 30)
    end_dict = {}

    def one():
        try:
            sandbox.run_the_container([engine, "run"], engine, "20260101-000000-1-1", 0,
                                      str(tmp_path / "out.txt"))
        except KeyboardInterrupt as e:
            end_dict["stop"] = str(e)
        end_dict["at"] = time.monotonic()

    thread = threading.Thread(target=one)
    start = time.monotonic()
    thread.start()
    time.sleep(stop_after_sec)
    sandbox.stop_running_container()
    thread.join(20)
    assert not thread.is_alive() and end_dict["at"] - start < 10
    assert "was stopped while ctloom-20260101-000000-1-1 was going" in end_dict["stop"]
    assert sandbox._LIVE_CONTAINER_DICT == {}
    command_list = said.read_text(encoding="utf-8").split()
    assert command_list[:2] == ["run", "rm"]
    # The client is gone: nothing is left to make or to run the container
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_the_containers_a_stop_takes_away_are_said_to_go_for_what_it_was_told(
        tmp_path, monkeypatch, caplog):
    """A broker ends its filling when the run it served has ended, which stops no run."""
    monkeypatch.setattr(sandbox, "_start_client", _START_CLIENT)
    monkeypatch.setattr(sandbox, "_STOP_WHY_LIST", [])
    engine, said = _slow_engine(tmp_path, 0.1, 30)
    end_dict = {}
    why = "the broker was asked to end (SIGTERM)"

    def one():
        try:
            sandbox.run_the_container([engine, "run"], engine, "20260101-000000-1-1", 0,
                                      str(tmp_path / "out.txt"))
        except KeyboardInterrupt as e:
            end_dict["stop"] = str(e)

    thread = threading.Thread(target=one)
    thread.start()
    time.sleep(0.5)
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        sandbox.stop_running_container(why)
    thread.join(20)
    assert end_dict["stop"] == f"{why} while ctloom-20260101-000000-1-1 was going"
    assert f"{why}, so ctloom-20260101-000000-1-1 is taken away" in caplog.text
    assert "the run was stopped" not in caplog.text
    with pytest.raises(KeyboardInterrupt) as stop:
        sandbox.run_the_container([engine, "run"], engine, "20260101-000000-1-2", 0,
                                  str(tmp_path / "out.txt"))
    assert str(stop.value) == f"{why}, so ctloom-20260101-000000-1-2 is not started"
    # Once the stop is taken back, a later stop says the run was stopped
    sandbox.forget_stop()
    sandbox._STOP_EVENT.set()
    with pytest.raises(KeyboardInterrupt, match="^the run was stopped, so"):
        sandbox.run_the_container([engine, "run"], engine, "20260101-000000-1-3", 0,
                                  str(tmp_path / "out.txt"))


def test_a_container_over_its_time_limit_is_taken_away_with_its_client(tmp_path,
                                                                    monkeypatch):
    monkeypatch.setattr(sandbox, "_start_client", _START_CLIENT)
    engine, said = _slow_engine(tmp_path, 0.1, 30)
    assert sandbox.run_the_container([engine, "run"], engine, "20260101-000000-1-1", 1,
                                     str(tmp_path / "out.txt")) == sandbox._TIME_OUT_STATUS
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid").read_text()), 0)
    assert said.read_text(encoding="utf-8").split() == ["run", "rm"]


def test_a_client_that_does_not_end_when_asked_is_killed(monkeypatch, caplog):
    monkeypatch.setattr(config, "CONTAINER_COMMAND_TIMEOUT_SEC", 1)
    monkeypatch.setattr(sandbox, "_remove_container", lambda engine, name: False)
    monkeypatch.setattr(sandbox, "_container_alive",
                        lambda engine, run_id, wait_sec=None: False)
    client = subprocess.Popen(["sh", "-c", "trap '' TERM; sleep 30"])
    time.sleep(0.3)
    sandbox._stop_container("podman", "ctloom-r1", client)
    assert client.returncode == -9 and "did not end when asked" in caplog.text


def test_a_container_the_engine_made_after_the_first_removal_is_removed(monkeypatch):
    """The client was ended while it was making the container, and the engine still
    names one afterwards."""
    removed = []
    monkeypatch.setattr(sandbox, "_remove_container",
                        lambda engine, name, wait_sec=None: removed.append(name) and False)
    monkeypatch.setattr(sandbox, "_container_alive",
                        lambda engine, run_id, wait_sec=None: run_id == "r1")
    for name in ("ctloom-r1", "ctloom-r2"):
        client = _ClientThroughRun(["podman", "run"], None, None)
        client.returncode = 0
        sandbox._stop_container("podman", name, client)
        assert client.stop_list == ["terminate"]
    assert removed == ["ctloom-r1", "ctloom-r1", "ctloom-r2"]


def _stuck_engine(tmp_path, rm_list, ps):
    """Write a stand-in for an engine whose n-th "rm" does what rm_list[n] says (a
    number of seconds to sleep, or "fail") and whose "ps" does what ps says (seconds
    to sleep, or "signal" for one that ends by SIGINT).

    Returns:
        (the engine's path, the file every command it was given is appended to).
    """
    engine, said = tmp_path / "podman", tmp_path / "said"
    rm_case = "\n".join(
        f"  {n}) " + ("echo no such container >&2; exit 1;;" if what == "fail"
                      else f"exec sleep {what};;") for n, what in enumerate(rm_list, 1))
    ps_line = "kill -INT $$" if ps == "signal" else f"exec sleep {ps}"
    engine.write_text(f"""#!/bin/sh
echo "$1" >> {said}
if [ "$1" = rm ]; then
  count=$(grep -c '^rm$' {said})
  case $count in
{rm_case}
  esac
  exit 0
fi
if [ "$1" = ps ]; then {ps_line}; fi
exit 0
""", encoding="utf-8")
    engine.chmod(0o755)
    return str(engine), said


@pytest.mark.parametrize("rm_list", [[5], ["fail", 5]])
def test_taking_a_container_away_takes_at_most_twice_the_command_time_limit(
        tmp_path, monkeypatch, caplog, rm_list):
    """An engine that does not answer, and a client that does not end when asked: the
    container is asked about and removed again only within what is left, and one that
    may still be there is named with the command that removes it."""
    monkeypatch.setattr(config, "CONTAINER_COMMAND_TIMEOUT_SEC", 0.5)
    engine, said = _stuck_engine(tmp_path, rm_list, 5)
    client = subprocess.Popen(["sh", "-c", "trap '' TERM; sleep 30"])
    time.sleep(0.3)
    start = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        sandbox._stop_container(engine, "ctloom-r1", client)
    assert time.monotonic() - start < 1.4
    assert client.returncode == -9
    assert said.read_text(encoding="utf-8").split() == (
        ["rm"] if rm_list == [5] else ["rm", "ps"])
    assert f"ctloom-r1 may still be there. Take it away with: {engine} rm --force ctloom-r1" \
        in caplog.text


def test_a_container_the_engine_cannot_say_about_once_its_client_ended_is_removed(
        tmp_path, caplog):
    """The question about the container ended by a signal (a Ctrl-C that reaches the
    engine's ps): the container counts as going and is removed again."""
    engine, said = _stuck_engine(tmp_path, ["fail", 0], "signal")
    client = subprocess.Popen(["sh", "-c", "sleep 30"])
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.sandbox"):
        sandbox._stop_container(engine, "ctloom-r1", client)
    assert said.read_text(encoding="utf-8").split() == ["rm", "ps", "rm"]
    assert "counts as still going" in caplog.text
    assert "nothing is taken away" not in caplog.text
    assert "may still be there" not in caplog.text


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_a_second_stop_while_containers_are_taken_away_lets_every_one_be_taken_away(
        monkeypatch, stop_signal):
    """The stop that arrives while the first container is removed is held until every
    container was removed and every client was ended, then stops the command."""
    monkeypatch.setattr(sandbox, "_STOP_EVENT", threading.Event())
    client_dict = {}
    for name in ("ctloom-r1", "ctloom-r2"):
        client_dict[name] = _ClientThroughRun(["podman", "run"], None, None)
        client_dict[name].returncode = 0
    monkeypatch.setattr(sandbox, "_LIVE_CONTAINER_DICT",
                        {name: ("podman", client) for name, client in client_dict.items()})
    removed = []

    def remove(engine, name):
        removed.append(name)
        if len(removed) == 1:
            signal.raise_signal(stop_signal)
        return True

    def stop(number, frame):
        raise KeyboardInterrupt(f"signal {number}")

    monkeypatch.setattr(sandbox, "_remove_container", remove)
    handler_before = signal.signal(stop_signal, stop)
    try:
        with pytest.raises(KeyboardInterrupt, match=f"signal {int(stop_signal)}"):
            sandbox.stop_running_container()
        assert signal.getsignal(stop_signal) is stop
    finally:
        signal.signal(stop_signal, handler_before)
    assert removed == ["ctloom-r1", "ctloom-r2"]
    assert [client.stop_list for client in client_dict.values()] == [["terminate"]] * 2
    assert sandbox._STOP_EVENT.is_set()


def test_a_second_stop_while_containers_are_taken_away_still_waits_for_the_agents(
        monkeypatch):
    monkeypatch.setattr(sandbox, "_STOP_EVENT", threading.Event())
    monkeypatch.setattr(config, "DISPATCH_PARALLEL", 2)
    client = _ClientThroughRun(["podman", "run"], None, None)
    client.returncode = 0
    monkeypatch.setattr(sandbox, "_LIVE_CONTAINER_DICT", {"ctloom-r1": ("podman", client)})

    def remove(engine, name):
        signal.raise_signal(signal.SIGINT)
        return True

    monkeypatch.setattr(sandbox, "_remove_container", remove)
    ended = []
    going = threading.Event()

    def one(n, task, attempt):
        if n == 1:
            going.wait(10)
            raise KeyboardInterrupt
        going.set()
        sandbox._STOP_EVENT.wait(10)
        time.sleep(0.3)
        ended.append(n)
        return {"n": n, "exit_status": 0}

    handler_before = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            sandbox._run_agent_batch([(1, "a"), (2, "b")], 1, one)
    finally:
        signal.signal(signal.SIGINT, handler_before)
    assert ended == [2] and client.stop_list == ["terminate"]


def test_only_the_last_try_of_a_task_is_named_for_a_command_that_was_stopped(
        _dispatch_repo):
    repo, out = _dispatch_repo
    run_id = f"20260102-000000-2-{os.getpid()}"
    for tail in ("1", "2", "2-retry1", "2-retry2", "3-retry1"):
        _git(repo, "branch", f"codetwine-loom/run/{run_id}/{tail}", "main")
    assert sandbox.remain_branch_list(str(repo), str(out), run_id) == [
        f"codetwine-loom/run/{run_id}/{tail}" for tail in ("1", "2-retry2", "3-retry1")]


def test_ctrl_c_while_this_thread_waits_takes_the_container_away(monkeypatch):
    """The sandbox subcommand, the reading run, the merging stage and warm wait for
    their container in the main thread, where Ctrl-C ends the engine's client and not
    the container."""
    monkeypatch.setattr(sandbox, "_STOP_EVENT", threading.Event())
    removed = []
    monkeypatch.setattr(sandbox, "_remove_container",
                        lambda engine, name: removed.append((engine, name)) or True)

    def interrupt(argv, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(sandbox.subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        sandbox.run_the_container(["podman", "run"], "podman", "r1", 0, os.devnull)
    assert removed == [("podman", sandbox.container_name("r1"))]
    assert sandbox._LIVE_CONTAINER_DICT == {}


def _fail_once(fail_task, error):
    """Return a stand-in for run() that raises error once for fail_task."""
    fail_list = []

    def fake_run(project_dir, output_dir, task, **kwargs):
        if task == fail_task and not fail_list:
            fail_list.append(task)
            raise error
        return {"commit": "c" * 40, "exit_status": 0, "verdict": "pass"}

    return fake_run


def test_an_agent_that_raises_an_oserror_stops_no_other_and_is_handed_out_again(
        monkeypatch):
    monkeypatch.setattr(sandbox, "_STOP_EVENT", threading.Event())
    monkeypatch.setattr(config, "DISPATCH_PARALLEL", 2)
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 1)
    stop_list = []
    monkeypatch.setattr(sandbox, "stop_running_container", lambda: stop_list.append(1))
    monkeypatch.setattr(sandbox, "run", _fail_once(
        "two", PermissionError(13, "Permission denied", "/out/scratch")))
    record_list = []
    real_edit = sandbox._edit_task

    def one(n, task, attempt):
        entry_dict = real_edit("/repo", "/out", "20260101-000000-1-1", "", "", None,
                               "req", n, task, attempt, "b" * 40)
        record_list.append(entry_dict)
        return entry_dict

    entry_list = sandbox._run_agent_list(["one", "two"], one, "task")
    assert not stop_list and not sandbox._STOP_EVENT.is_set()
    assert [e["exit_status"] for e in entry_list] == [0, 0]
    assert entry_list[1]["attempt"] == 2
    first = next(e for e in record_list if e["n"] == 2 and e["attempt"] == 1)
    assert first["exit_status"] == 1
    assert first["error"].startswith("PermissionError: ")


def test_a_runtime_error_keeps_its_message_as_it_is(monkeypatch):
    monkeypatch.setattr(sandbox, "run", _fail_once("one", RuntimeError("no apply")))
    entry_dict = sandbox._edit_task("/repo", "/out", "20260101-000000-1-1", "", "",
                                    None, "req", 1, "one", 1, "b" * 40)
    assert entry_dict["exit_status"] == 1 and entry_dict["error"] == "no apply"


def test_an_answering_agent_that_raises_an_oserror_is_an_unanswered_entry(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sandbox, "answer", fail)
    entry_dict = sandbox._answer_agent("/repo", "/out", "", "", None, "b" * 40,
                                       "20260101-000000-1-1", 1, "why?", 1)
    assert entry_dict["exit_status"] == 1
    assert entry_dict["error"] == sandbox._NO_ANSWER_BACK


@pytest.mark.parametrize("status, how", [
    (124, "exit status 124, the status of a container taken away at SANDBOX_TIMEOUT_SEC "
          "(2 s)"),
    (137, "exit status 137, the status of a container killed when its processes reach "
          "SANDBOX_MEMORY (2g)"),
    (1, "exit status 1")])
def test_an_answering_agent_whose_container_ended_says_how_without_a_path(
        monkeypatch, status, how):
    monkeypatch.setattr(config, "SANDBOX_TIMEOUT_SEC", 2)
    monkeypatch.setattr(config, "SANDBOX_MEMORY", "2g")

    def fail(*args, **kwargs):
        raise sandbox.ContainerEndError(
            RuntimeError("the run that answers left no answer in /out/a/answer.json"),
            status)

    monkeypatch.setattr(sandbox, "answer", fail)
    entry_dict = sandbox._answer_agent("/repo", "/out", "", "", None, "b" * 40,
                                       "20260101-000000-1-1", 1, "why?", 1)
    assert entry_dict["exit_status"] == 1
    assert entry_dict["error"] == f"{sandbox._NO_ANSWER_BACK}: the container ended ({how})"


@pytest.mark.parametrize("status, how", [
    (1, "the container ended (exit status 1); "), (0, "")])
def test_an_answering_agent_says_the_reason_its_answer_file_gives(
        tmp_path, monkeypatch, status, how):
    why = "the answer stage handed nothing over: no COMPLETE() within 30 turns"
    path = tmp_path / "answer.json"
    path.write_text(json.dumps({"error": why}), encoding="utf-8")

    def stand_in(*args, **kwargs):
        sandbox._read_after_end(
            lambda: sandbox._read_answer(str(path), str(tmp_path)), status)

    monkeypatch.setattr(sandbox, "answer", stand_in)
    entry_dict = sandbox._answer_agent("/repo", "/out", "", "", None, "b" * 40,
                                       "20260101-000000-1-1", 1, "why?", 1)
    assert entry_dict["exit_status"] == 1
    assert entry_dict["error"] == (
        f"{sandbox._NO_ANSWER_BACK}: {how}the run that answers said: {why}")
    assert str(tmp_path) not in entry_dict["error"]


def test_the_reason_an_answer_file_gives_is_held_to_the_answer_limit(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 20)
    why = "the run stopped early: " + "x" * 100
    path = tmp_path / "answer.json"
    path.write_text(json.dumps({"error": why}), encoding="utf-8")
    with pytest.raises(sandbox._NoAnswerError) as caught:
        sandbox._read_answer(str(path), str(tmp_path))
    assert caught.value.why == sandbox.cut(why, 20)
    assert str(caught.value) == (
        f"the run that answers handed no answer back ({sandbox.cut(why, 20)})")


def test_a_test_task_that_raises_an_oserror_is_an_entry_that_did_not_finish(
        monkeypatch):
    def fail(*args, **kwargs):
        raise OSError(28, "No space left on device")

    drop_list = []
    monkeypatch.setattr(sandbox, "make_agent_dir", fail)
    monkeypatch.setattr(sandbox, "drop_the_copy",
                        lambda *args: drop_list.append(args))
    task_dict_list = [{"paths": [f"{n}.py"], "test_paths": [], "chain": 0}
                      for n in (1, 2)]
    entry_dict = sandbox._write_test_task(
        "/repo", "/out", "repo", "", "", "u" * 40, "b" * 40, "req", None, [],
        task_dict_list,
        {"branch": "codetwine-loom/t", "agent": "codetwine-loom-t", "out_dir": "/out"}, 2,
        "2.py", 1, "u" * 40)
    assert entry_dict["exit_status"] == 1
    assert entry_dict["error"].startswith("OSError: ")
    assert entry_dict["why"] == entry_dict["error"]
    assert len(drop_list) == 1


# == the token counts of one run and MAX_TOKEN_PER_RUN ========================

def _container_ready(tmp_path, monkeypatch, start_broker):
    """Stand in for everything _start_container() starts, and return what it is given.

    Args:
        tmp_path: The test's directory.
        monkeypatch: pytest's fixture.
        start_broker: The stand-in for sandbox.start_broker.

    Returns:
        (the arguments _start_container() takes, the temporary directory it uses).
    """
    repo = _repo(tmp_path / "repo")
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_dir))
    monkeypatch.setattr(sandbox, "pick_engine", lambda engine="": "/usr/bin/true")
    monkeypatch.setattr(sandbox.llm, "missing_credential_text", lambda: "")
    monkeypatch.setattr(sandbox, "index_describe", lambda *a: True)
    monkeypatch.setattr(sandbox, "fill_cache_once", lambda *a, **k: 0)
    monkeypatch.setattr(sandbox, "run_the_container", lambda *a, **k: 0)
    monkeypatch.setattr(sandbox, "start_broker", start_broker)
    out = str(tmp_path / "out")
    run_id = "20260101-000000-1-1"
    run_file = sandbox.run_file_layout(out, "a", "repo", run_id)
    return ("", "img", str(repo), out, run_file.out_dir, "repo", "a", "0" * 40,
            run_file, run_id, ["edit"], []), tmp_dir


def test_what_the_broker_counted_is_added_to_this_process_s_counts(tmp_path, monkeypatch):
    def start(sock_dir, *a, token_out="", **k):
        with open(token_out, "w", encoding="utf-8") as handle:
            json.dump({"call": 4, "input_token": 90, "output_token": 10,
                       "cache_read_token": 50, "cache_write_token": 5}, handle)

    arg_tuple, tmp_dir = _container_ready(tmp_path, monkeypatch, start)
    sandbox._start_container(*arg_tuple)
    total = sandbox.tokencount.total_dict()
    assert (total["call"], total["total"], total["cache_read_token"]) == (4, 100, 50)
    assert [part["part"] for part in total["parts"]] == ["edit 20260101-000000-1-1"]
    # The file the broker wrote stays, in the one directory this process keeps them in,
    # until the counts are started again
    assert [len(os.listdir(tmp_dir / name)) for name in os.listdir(tmp_dir)] == [1]
    sandbox.tokencount.reset()
    assert os.listdir(tmp_dir) == []


def test_every_broker_writes_its_counts_into_one_directory(tmp_path, monkeypatch):
    seen = []
    arg_tuple, tmp_dir = _container_ready(
        tmp_path, monkeypatch, lambda *a, token_out="", **k: seen.append(token_out))
    sandbox._start_container(*arg_tuple)
    sandbox._start_container(*arg_tuple)
    assert len(set(seen)) == 2
    assert len({os.path.dirname(path) for path in seen}) == 1
    assert all(os.path.exists(path) for path in seen)


def test_a_broker_that_made_no_call_adds_no_part(tmp_path, monkeypatch):
    arg_tuple, tmp_dir = _container_ready(tmp_path, monkeypatch, lambda *a, **k: None)
    sandbox._start_container(*arg_tuple)
    assert sandbox.tokencount.total_dict()["parts"] == []
    sandbox.tokencount.reset()
    assert os.listdir(tmp_dir) == []


def test_no_container_is_started_once_the_token_limit_is_reached(tmp_path, monkeypatch):
    started = []
    arg_tuple, tmp_dir = _container_ready(
        tmp_path, monkeypatch, lambda *a, **k: started.append(a))
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 100)
    sandbox.tokencount.add({"input_token": 100})
    with pytest.raises(RuntimeError, match="MAX_TOKEN_PER_RUN=100, so no further "
                                           "container is started"):
        sandbox._start_container(*arg_tuple)
    assert started == [] and os.listdir(tmp_dir) == []
    assert not os.path.exists(arg_tuple[8].scratch)


def test_a_task_is_not_handed_out_again_once_the_token_limit_is_reached(
        _dispatch_repo, monkeypatch, caplog):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one"]}))
    started = []

    def fake_run(project, output_dir, request, **kw):
        started.append(kw["branch"])
        sandbox.tokencount.add({"input_token": 100})
        return {"commit": "c" * 40, "exit_status": 1, "verdict": "failed"}

    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 100)
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 3)
    monkeypatch.setattr(sandbox, "run", fake_run)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert started == ["codetwine-loom/run/20260101-000000-1-42/1"]
    assert result["exit_status"] == 1 and result["token_stop"] == ""
    assert "are not handed out again" in caplog.text


def test_the_merge_is_not_started_once_the_token_limit_is_reached(_dispatch_repo,
                                                                  monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one", "two"]}))

    def fake_run(project, output_dir, request, **kw):
        _git(repo, "branch", kw["branch"])
        sandbox.tokencount.add({"input_token": 60})
        return {"commit": "c" * 40, "exit_status": 0, "verdict": "unverified"}

    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 100)
    monkeypatch.setattr(sandbox, "run", fake_run)
    monkeypatch.setattr(sandbox, "integrate", _never("the limit is reached"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert result["integration"] is None and result["exit_status"] == 1
    assert "MAX_TOKEN_PER_RUN=100" in result["token_stop"]
    # What the tasks left is what a run carrying on is told to read
    assert result["resume"] == ["codetwine-loom/run/20260101-000000-1-42/1",
                                "codetwine-loom/run/20260101-000000-1-42/2"]


def test_a_request_is_not_run_again_once_the_token_limit_is_reached(_dispatch_repo,
                                                                    monkeypatch):
    repo, out = _dispatch_repo
    _run_ids(monkeypatch, 3)
    told = []
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written_where_asked({"tasks": ["one"]}, told))

    def fake_run(project, output_dir, request, **kw):
        _git(repo, "branch", kw["branch"])
        sandbox.tokencount.add({"input_token": 100})
        return {"commit": "c" * 40, "exit_status": 1, "verdict": "failed"}

    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 100)
    monkeypatch.setattr(config, "DISPATCH_MAX_RETRY", 0)
    monkeypatch.setattr(config, "DISPATCH_MAX_RERUN", 2)
    monkeypatch.setattr(sandbox, "run", fake_run)
    monkeypatch.setattr(sandbox, "integrate", _never("a task did not finish"))
    result = sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert len(told) == 1 and result["runs"] == 1 and result["exit_status"] == 1


# == the check at the end of a run on its own ==================================

@pytest.mark.parametrize("check", [True, False])
def test_the_container_is_told_whether_to_check_the_change(monkeypatch, tmp_path, check):
    repo = _repo(tmp_path / "repo")
    seen = []

    def start(*a, **k):
        seen.append(a[10])
        return "/usr/bin/true", "img", 0

    monkeypatch.setattr(sandbox, "_start_container", start)
    sandbox.run(str(repo), str(tmp_path / "out"), "x", check=check)
    assert seen[0][-1] == ("--check" if check else "--no-check")


def test_the_agents_of_a_dispatching_run_are_not_checked_one_by_one(_dispatch_repo,
                                                                     monkeypatch):
    repo, out = _dispatch_repo
    monkeypatch.setattr(sandbox, "_start_container",
                        _split_written(_tasks_file(out), {"tasks": ["one"]}))
    seen = []

    def fake_run(project, output_dir, request, **kw):
        seen.append(kw.get("check", False))
        return {"commit": "", "exit_status": 0, "verdict": "no_change"}

    monkeypatch.setattr(sandbox, "run", fake_run)
    sandbox.dispatch(str(repo), str(out), "設定を整理して")
    assert seen == [False]


@pytest.mark.parametrize("command, flag", [([], "CTLOOM_FORMAT=0"),
                                           (["ruff", "format"], "CTLOOM_FORMAT=1")])
def test_the_container_is_told_whether_there_is_a_formatting_command(monkeypatch,
                                                                     command, flag):
    monkeypatch.setattr(config, "FORMAT_COMMAND", command)
    assert flag in _argv()


# == the network a run's containers join (SANDBOX_NETWORK) ====================

def test_a_run_has_no_network_until_one_is_named(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_NETWORK", "")
    assert "--network=none" in _argv()
    monkeypatch.setattr(config, "SANDBOX_NETWORK", "test-net")
    argv = _argv()
    assert "--network=test-net" in argv and "--network=none" not in argv
    # The build tools still work from the package cache alone
    assert "TOOL_OFFLINE=true" in argv
    # The container that fills the cache keeps the engine's own network
    assert not [arg for arg in _argv(network=True) if arg.startswith("--network")]


@pytest.mark.parametrize("text, is_internal", [
    ('[{"name": "n", "internal": true}]', True), ('[{"Name": "n", "Internal": true}]', True),
    ('[{"name": "n", "internal": false}]', False), ('{"Internal": false}', False),
    ('[{"name": "n"}]', None), ("[]", None), ("not json", None), ('["n"]', None),
    ('[{"internal": "true"}]', None)])
def test_whether_a_network_is_internal_is_read_off_the_engine_s_answer(text, is_internal):
    assert sandbox._is_internal_network(text) is is_internal


def _network_inspect(monkeypatch, returncode=0, stdout="", stderr="", error=None):
    """Answer the engine's network inspect, and return the argv lists it was asked."""
    asked = []

    def run(argv, **kwargs):
        asked.append(argv)
        if error is not None:
            raise error
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    return asked


def test_no_network_named_asks_the_engine_nothing(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_NETWORK", "")
    asked = _network_inspect(monkeypatch)
    sandbox.refuse_open_network("/usr/bin/podman")
    assert asked == []


def test_an_internal_network_is_taken(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_NETWORK", "test-net")
    asked = _network_inspect(monkeypatch, stdout='[{"internal": true}]')
    sandbox.refuse_open_network("/usr/bin/podman")
    assert asked == [["/usr/bin/podman", "network", "inspect", "test-net"]]


@pytest.mark.parametrize("stdout", ['[{"internal": false}]', '[{"name": "test-net"}]', ""])
def test_a_network_not_reported_as_internal_is_refused(monkeypatch, stdout):
    monkeypatch.setattr(config, "SANDBOX_NETWORK", "test-net")
    _network_inspect(monkeypatch, stdout=stdout)
    with pytest.raises(RuntimeError, match="does not report the network test-net "
                                           r"\(SANDBOX_NETWORK\) as internal") as raised:
        sandbox.refuse_open_network("/usr/bin/podman")
    assert "podman network create --internal <name>" in str(raised.value)


def test_a_network_the_engine_does_not_have_is_refused(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_NETWORK", "test-net")
    _network_inspect(monkeypatch, returncode=125, stderr="Error: network test-net: not found")
    with pytest.raises(RuntimeError, match="does not show the network test-net") as raised:
        sandbox.refuse_open_network("/usr/bin/docker")
    assert "not found" in str(raised.value)
    assert "docker network create --internal <name>" in str(raised.value)


@pytest.mark.parametrize("error", [OSError("gone"),
                                   subprocess.TimeoutExpired(["podman"], 30)])
def test_an_engine_that_cannot_be_asked_about_the_network_starts_nothing(monkeypatch,
                                                                        error):
    monkeypatch.setattr(config, "SANDBOX_NETWORK", "test-net")
    _network_inspect(monkeypatch, error=error)
    with pytest.raises(RuntimeError, match="not known to be internal"):
        sandbox.refuse_open_network("/usr/bin/podman")
    # With no network named nothing is asked
    monkeypatch.setattr(config, "SANDBOX_NETWORK", "")
    sandbox.refuse_open_network("/usr/bin/podman")


def test_the_broker_is_told_what_is_left_of_the_token_limit(tmp_path, monkeypatch):
    seen = []
    arg_tuple, _ = _container_ready(
        tmp_path, monkeypatch, lambda *a, token_left=0, **k: seen.append(token_left))
    sandbox._start_container(*arg_tuple)
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 100)
    sandbox.tokencount.add({"input_token": 30, "output_token": 10})
    sandbox._start_container(*arg_tuple)
    assert seen == [0, 60]


def test_what_a_broker_used_is_not_taken_off_what_the_next_is_told(tmp_path, monkeypatch):
    """The next broker counts the first one's file itself (tokencount.total_beside)."""
    seen = []

    def start(sock_dir, *a, token_out="", token_left=0, **k):
        seen.append(token_left)
        with open(token_out, "w", encoding="utf-8") as handle:
            json.dump({"call": 1, "input_token": 30, "output_token": 10}, handle)

    arg_tuple, _ = _container_ready(tmp_path, monkeypatch, start)
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 100)
    sandbox._start_container(*arg_tuple)
    sandbox._start_container(*arg_tuple)
    assert seen == [100, 100] and sandbox.tokencount.total_dict()["total"] == 80


def test_a_broker_given_no_limit_is_started_without_one(tmp_path, monkeypatch):
    """The real start_broker() puts --token-left on the command line only above 0."""
    argv_list = []

    class _Process:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(sandbox, "start_broker", _START_BROKER)
    monkeypatch.setattr(sandbox.subprocess, "Popen",
                        lambda argv, **k: argv_list.append(argv) or _Process())
    for token_left in (0, 70):
        with pytest.raises(RuntimeError, match="ended before it listened"):
            sandbox.start_broker(str(tmp_path), token_left=token_left)
    assert "--token-left" not in argv_list[0]
    assert argv_list[1][argv_list[1].index("--token-left") + 1] == "70"


def test_a_trailer_or_an_engine_answer_nested_too_deep_to_read_is_one_that_cannot_be_read():
    deep = "[" * 100_000 + "]" * 100_000
    assert not sandbox.path_list_of(deep)
    assert sandbox._is_internal_network(deep) is None


@pytest.mark.parametrize("chain_list, parallel, count", [
    ([1, 1, 1, 0, 0], 3, 1), ([0, 1, 1, 0], 3, 2), ([0, 0, 0, 0], 3, 3),
    ([1, 2, 1, 2], 2, 2), ([0, 0], 5, 2), ([1, 1], 0, 1)])
def test_the_agents_that_start_at_once_leave_out_the_tasks_that_wait_in_a_chain(
        chain_list, parallel, count):
    assert sandbox.start_count_of(chain_list, parallel) == count



def test_a_task_of_a_chain_is_told_the_requests_whose_work_it_starts_at(monkeypatch):
    seen_list = []

    def fake_run(project_dir, output_dir, task, **kwargs):
        seen_list.append((kwargs["start"], kwargs.get("request_note", "") + task))
        return {"commit": f"c{len(seen_list)}", "exit_status": 0, "verdict": "ok"}

    monkeypatch.setattr(sandbox, "run", fake_run)
    chain_dict = {1: sandbox.new_chain_dict("base")}
    task_list = ["fix f in a.py", "report on f in a.py"]
    for n, task in enumerate(task_list, 1):
        entry_dict = sandbox._edit_agent("/repo", "/out", "20260101-000000-1-1", "", "",
                                         None, "req", "base", [1, 1], chain_dict,
                                         task_list, n, task, 1)
        # What is kept as the task's request is the request itself
        assert entry_dict["request"] == task
    assert seen_list[0] == ("base", "fix f in a.py")
    start, told = seen_list[1]
    assert start == "c1"
    assert told == sandbox._CHAIN_NOTE.format(list="- fix f in a.py") + "report on f in a.py"
    assert chain_dict[1]["land"] == [1, 2]


def test_a_try_again_does_not_wait_for_a_task_killed_at_the_memory_limit():
    chain_dict = sandbox.new_chain_dict("base")

    def run_one(n, attempt, status):
        return sandbox.run_in_chain(
            chain_dict, [1, 2], n, attempt,
            lambda start: {"commit": "", "branch": "", "exit_status": status},
            lambda e: not e["exit_status"], f"{n} waits")

    run_one(1, 1, sandbox._KILL_STATUS)
    run_one(2, 1, 1)
    # Task 1 is not handed out again; task 2's second try starts without waiting for it
    thread = threading.Thread(target=run_one, args=(2, 2, 0))
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert chain_dict["ran"][2] == {"attempt": 2, "ok": True, "kill": False}


def test_the_container_is_told_where_the_repository_is_on_this_machine(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    argv = sandbox.build_run_argv("/usr/bin/podman", "an-image", _MOUNTS, "abc123", "repo",
                                  ["edit"], host_project=str(tmp_path / "link"))
    value = next(a.split("=", 1)[1] for a in argv
                 if a.startswith(f"{paths.HOST_PROJECT_ENV}="))
    # As written and with its links resolved
    assert value.split("\n") == [str(tmp_path / "link"), str(real)]
    assert not any(a.startswith(f"{paths.HOST_PROJECT_ENV}=") for a in
                   sandbox.build_run_argv("/usr/bin/podman", "an-image", _MOUNTS, "abc123",
                                          "repo", ["edit"]))


def test_a_stage_held_back_by_a_command_not_in_the_image_is_not_run_again(
        monkeypatch, caplog):
    monkeypatch.setattr(config, "TEST_CREATE_MAX_RERUN", 2)
    monkeypatch.setattr(sandbox, "_stage_on_list", lambda: [integrate_mod.TEST_CREATE_STAGE])
    monkeypatch.setattr(integrate_mod, "gap_list_in", lambda *a: [])
    call_list = []

    def open_stage(*args, **kwargs):
        call_list.append(1)
        return {"skipped": "", "verdict": integrate_mod.UNSETTLE_VERDICT, "commit": None,
                "branch": "b", "why": "- `npm test --silent` did not run (sh: 1: bun: not found)"}

    monkeypatch.setattr(sandbox, "_create_test", open_stage)
    with caplog.at_level(logging.WARNING):
        sandbox._run_stage_list("/repo", "/out", "/out", "repo", "a", "", "", "u" * 40,
                                "b" * 40, "b", "req", None)
    assert call_list == [1]
    assert "because bun is not in the image" in caplog.text


@pytest.mark.parametrize("text, want", [
    ("`npm test` did not run (sh: 1: bun: not found)", ["bun"]),
    ("`x` did not run (bash: line 1: deno: command not found)", ["deno"]),
    ("`x` did not run (/bin/sh: 1: bun: not found\nsh: 1: bun: not found)", ["bun"]),
    ("`ctloom-test-c` did not run (make: valgrind: No such file or directory)", ["valgrind"]),
    # A note or a cut mark in front of what the command printed
    ("`npm test` did not run (note: 1 refill(s) left this run\nsh: 1: bun: not found)",
     ["bun"]),
    ("`ctloom-test-c` did not run (... (900 chars omitted)\n"
     "make: valgrind: No such file or directory); `npm test` fails (x)", ["valgrind"]),
    # Only why a test command did not run is read: a test that ran and failed is not
    ("`npm test` fails (/bin/sh: 1: mycli: not found)", []),
    ("`x` did not run (Error: Cannot find module 'y')", [])])
def test_the_commands_a_shell_could_not_find_are_read_off_a_text(text, want):
    assert sandbox.edit_mod.missing_command_list(text) == want


def test_a_stage_of_several_test_tasks_stops_only_when_each_open_one_lacks_a_command():
    bun = "- `npm test --silent` did not run (sh: 1: bun: not found)"
    open_task = {"skipped": "", "verdict": integrate_mod.UNSETTLE_VERDICT}
    stage_dict = {"why": "x", "test_tasks": [{**open_task, "why": bun},
                                              {**open_task, "why": "- a test fails"}]}
    assert sandbox._missing_command_list(stage_dict) == []
    stage_dict["test_tasks"][1]["why"] = bun
    assert sandbox._missing_command_list(stage_dict) == ["bun"]
