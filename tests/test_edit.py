"""The editing tools, replace_line_range and create_file among them.

The checks run against a git repository created under tmp_path. config.TEST_COMMAND is
substituted so that no real build tool has to be called.
"""

import codecs
import collections
import dataclasses
import json
import logging
import os
import pathlib
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time

import pytest
from codetwine.utils import file_utils

from codetwine_loom import agent, config, gitcmd, integrate, paths, rlm, tmpfile
from codetwine_loom.languages import ALIAS_DICT, LANG_DICT, is_dep_file
from codetwine_loom.textcut import CUT_NOTE
from codetwine_loom.tools import edit, source

# The real run_check; conftest.py replaces edit.run_check with one that runs nothing
_RUN_CHECK = edit.run_check

# The head pytest prints its FAILED, ERROR and SUBFAILED lines under
_SUMMARY_HEAD = "=== short test summary info ===\n"


def _git(root, *args) -> str:
    """Run git for the tests."""
    return subprocess.run(["git", *args], cwd=root, capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Create a git repository holding one m.py and point PROJECT_DIR at it."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "m.py").write_text("old\nctx\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    # Read again now that PROJECT_DIR is this repository: what the conftest fixture saw
    # was whatever repository the tests are being run in
    edit.reset_patch_round()
    return root


# == replace_in_file ========
def test_replace_in_file_replaces_one_place(repo):
    r = edit.replace_in_file("m.py", "old", "new")
    assert r["ok"] is True
    assert (repo / "m.py").read_text() == "new\nctx\n"
    assert "line 1" in r["stat"]


def test_replace_in_file_can_replace_several_lines(repo):
    r = edit.replace_in_file("m.py", "old\nctx\n", "a\nb\nc\n")
    assert r["ok"] is True
    assert (repo / "m.py").read_text() == "a\nb\nc\n"


def test_replace_in_file_writes_nothing_when_the_text_is_not_found(repo):
    r = edit.replace_in_file("m.py", "nowhere", "new")
    assert r["ok"] is False
    assert "not found" in r["error"]
    assert (repo / "m.py").read_text() == "old\nctx\n"


def test_replace_in_file_says_so_when_only_whitespace_differs(repo):
    """The guidance for a model that did not paste back the indentation."""
    r = edit.replace_in_file("m.py", "    old", "new")
    assert r["ok"] is False
    assert "whitespace" in r["error"]


def test_replace_in_file_writes_nothing_on_2_or_more_matches(repo):
    (repo / "m.py").write_text("old\nctx\nold\n")
    r = edit.replace_in_file("m.py", "old", "new")
    assert r["ok"] is False
    assert "2 places" in r["error"] and "[1, 3]" in r["error"]
    assert (repo / "m.py").read_text() == "old\nctx\nold\n"


def test_adding_surrounding_lines_makes_replace_in_file_unique(repo):
    (repo / "m.py").write_text("old\nctx\nold\n")
    assert edit.replace_in_file("m.py", "ctx\nold\n", "ctx\nnew\n")["ok"] is True
    assert (repo / "m.py").read_text() == "old\nctx\nnew\n"


def test_replace_in_file_writes_nothing_when_the_texts_are_identical(repo):
    r = edit.replace_in_file("m.py", "old", "old")
    assert r["ok"] is False
    assert "identical" in r["error"]


def test_replace_in_file_rejects_an_empty_old_text(repo):
    out = edit.replace_in_file("m.py", "", "new")
    assert out["ok"] is False and "old_text is empty" in out["error"]


def test_replace_in_file_rejects_a_path_outside_project_dir(repo):
    out = edit.replace_in_file("../outside.py", "old", "new")
    assert out["ok"] is False and "outside the repository" in out["error"]


def test_replace_in_file_rejects_a_path_under_output_dir(tmp_path, monkeypatch):
    """The index's output is not the original source and must never be edited."""
    root = tmp_path / "proj"
    out = root / "out"
    out.mkdir(parents=True)
    (out / "m.py").write_text("old\nctx\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(out))
    out_dict = edit.replace_in_file("out/m.py", "old", "new")
    assert out_dict["ok"] is False and "OUTPUT_DIR" in out_dict["error"]
    assert (out / "m.py").read_text() == "old\nctx\n"


def test_replace_in_file_returns_a_diff_without_writing_under_dry_run(repo, monkeypatch,
                                                                     capsys):
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    r = edit.replace_in_file("m.py", "old", "new")
    assert r["ok"] is True and r["dry_run"] is True
    assert "-old" in r["diff"] and "+new" in r["diff"]
    assert (repo / "m.py").read_text() == "old\nctx\n"
    assert capsys.readouterr().out == ""


def test_replace_in_file_is_counted_in_the_patch_stats(repo):
    edit.replace_in_file("m.py", "old", "new")
    edit.replace_in_file("m.py", "nowhere", "x")
    assert edit.patch_count_dict() == {"ok": 1, "ng": 1, "ng_after_ok": 1}


# == run_test =======================================
# == replace_line_range (replacing by line number) ==
@pytest.fixture
def lines_repo(repo):
    """The repo with a 5-line file, for the line-number checks."""
    (repo / "m.py").write_text("l1\nl2\nl3\nl4\nl5\n")
    return repo


def test_a_write_through_a_symbolic_link_is_kept_under_the_file_it_lands_on(repo):
    (repo / "real.py").write_text("x = 1\n")
    (repo / "link.py").symlink_to("real.py")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "link")
    edit.reset_patch_round()
    assert edit.replace_in_file("link.py", "x = 1", "x = 2")["ok"] is True
    assert edit.write_path_set() == edit.own_change_set() == {"real.py"}
    assert edit.no_tool_path_set() == set()
    assert edit.revert_file("link.py")["ok"] is True
    assert (repo / "real.py").read_text() == "x = 1\n"
    assert edit.write_path_set() == edit.own_change_set() == set()


def test_a_write_through_a_symbolic_link_to_an_ignored_file_is_refused(repo):
    (repo / ".gitignore").write_text("secret.txt\n")
    (repo / "secret.txt").write_text("s = 1\n")
    (repo / "note.txt").symlink_to("secret.txt")
    r = edit.replace_in_file("note.txt", "s = 1", "s = 2")
    assert r["ok"] is False and "git ignores" in r["error"]
    assert (repo / "secret.txt").read_text() == "s = 1\n"


@pytest.fixture
def link_repo(repo):
    """Add a submodule whose files are not checked out (sub) and a directory holding a
    repository of its own that nothing tracks (vendor/lib)."""
    commit = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},sub")
    _git(repo, "commit", "-q", "-m", "sub")
    (repo / "sub").mkdir()
    (repo / "vendor" / "lib").mkdir(parents=True)
    _git(repo / "vendor" / "lib", "init", "-q")
    (repo / "vendor" / "lib" / "x.py").write_text("x = 1\n")
    edit.reset_patch_round()
    return repo


def test_a_write_under_a_repository_of_its_own_is_refused(link_repo):
    r = edit.create_file("sub/pkg/new.py", "y = 1\n")
    assert r["ok"] is False and r["error"] == (
        "sub/pkg/new.py is inside sub, a repository of its own, so nothing written "
        "there is carried out of this run. Write what the change needs into a file git "
        "tracks")
    assert not (link_repo / "sub" / "pkg" / "new.py").exists()
    r = edit.replace_in_file("vendor/lib/x.py", "x = 1", "x = 2")
    assert r["ok"] is False and "is inside vendor/lib, a repository of its own" in r["error"]
    assert edit.replace_line_range("vendor/lib/x.py", 1, 1, "x = 2")["ok"] is False
    r = edit.delete_file("vendor/lib/x.py")
    assert r["ok"] is False and r["error"] == (
        "vendor/lib/x.py is inside vendor/lib, a repository of its own, so deleting it "
        "is not carried out of this run")
    assert (link_repo / "vendor" / "lib" / "x.py").read_text() == "x = 1\n"
    assert edit.write_path_set() == set()
    # A file beside them is written as any other
    assert edit.create_file("vendor/note.py", "n = 1\n")["ok"] is True


def test_revert_file_does_not_say_a_path_git_carries_nothing_of_is_unchanged(link_repo):
    (link_repo / "vendor" / "lib" / "x.py").write_text("x = 2\n")
    r = edit.revert_file("vendor/lib/x.py")
    assert r["ok"] is False and r["error"] == (
        "vendor/lib/x.py is inside vendor/lib, a repository of its own, so what it held "
        "when this request started is not known and it cannot be reverted")
    (link_repo / ".gitignore").write_text("secret.txt\n")
    (link_repo / "secret.txt").write_text("s = 1\n")
    assert edit.revert_file("secret.txt")["error"].startswith("git ignores secret.txt, so ")
    assert "nothing of this request to revert" in edit.revert_file("m.py")["error"]


def test_deleting_a_symbolic_link_deletes_the_link_and_not_its_file(repo):
    (repo / "pkg").mkdir()
    (repo / "pkg" / "real.py").write_text("x = 1\n")
    (repo / "link.py").symlink_to("pkg/real.py")
    (repo / "dir_link").symlink_to("pkg")
    (repo / "gone_link.py").symlink_to("nothing_here.py")
    _git(repo, "add", "-A"), _git(repo, "commit", "-q", "-m", "links")
    edit.reset_patch_round()
    for path in ("link.py", "dir_link", "gone_link.py"):
        r = edit.delete_file(path)
        assert (r["ok"], r["stat"]) == (True, "deleted"), path
        assert not os.path.lexists(repo / path)
    assert (repo / "pkg" / "real.py").read_text() == "x = 1\n"
    assert sorted(gitcmd.name_status(str(repo))["deleted"]) == [
        "dir_link", "gone_link.py", "link.py"]
    assert edit.write_path_set() == {"link.py", "dir_link", "gone_link.py"}
    assert edit.no_tool_path_set() == set()
    # A file reached through a link of a directory is deleted as the file it is
    (repo / "dir_link").symlink_to("pkg")
    assert edit.delete_file("dir_link/real.py")["ok"] is True
    assert not (repo / "pkg" / "real.py").exists()
    # The link comes back from HEAD as a link
    assert edit.revert_file("link.py")["ok"] is True
    assert os.readlink(repo / "link.py") == "pkg/real.py"


def test_a_link_that_leads_out_of_the_repository_is_deleted_as_the_link_it_is(
        repo, tmp_path):
    (tmp_path / "far.txt").write_text("another directory's\n")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "made.txt").write_text("made here\n")
    target_dict = {"far_link": tmp_path / "far.txt", "far_dir_link": tmp_path,
                   "out_link": tmp_path / "out" / "made.txt",
                   "git_link": repo / ".git" / "config", "device_link": "/dev/null"}
    for name, target in target_dict.items():
        (repo / name).symlink_to(target)
    _git(repo, "add", "-A"), _git(repo, "commit", "-q", "-m", "links")
    edit.reset_patch_round()
    for name, target in target_dict.items():
        r = edit.delete_file(name)
        assert (r["ok"], r["stat"]) == (True, "deleted"), (name, r)
        assert not os.path.lexists(repo / name) and os.path.exists(target)
    assert edit.write_path_set() == edit.own_change_set() == set(target_dict)
    # Through the link, what it points at is still out of reach
    (repo / "far_dir_link").symlink_to(tmp_path)
    r = edit.delete_file("far_dir_link/far.txt")
    assert r["ok"] is False and "outside the repository" in r["error"]
    assert (tmp_path / "far.txt").exists()


def test_a_directory_a_tool_made_goes_with_its_last_file_and_one_that_was_there_stays(repo):
    for name in ("mod.py/inner.py", "pkg/deep/a.py", "pkg/deep/b.py", "pkg/keep.py"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x = 1\n")
    (repo / "scratch" / "deep").mkdir(parents=True)
    _git(repo, "add", "-A"), _git(repo, "commit", "-q", "-m", "directories")
    edit.reset_patch_round()
    # A directory that was there when the request started stays, emptied or not
    assert edit.delete_file("pkg/deep/a.py")["ok"] is True
    assert edit.delete_file("pkg/deep/b.py")["ok"] is True
    assert (repo / "pkg" / "deep").is_dir() and (repo / "pkg" / "keep.py").exists()
    assert edit.create_file("scratch/deep/a.py", "y = 1\n")["ok"] is True
    assert edit.revert_file("scratch/deep/a.py")["ok"] is True
    assert (repo / "scratch" / "deep").is_dir()
    # A file put back finds its directory there
    assert edit.revert_file("pkg/deep/a.py")["ok"] is True
    assert (repo / "pkg" / "deep" / "a.py").read_text() == "x = 1\n"
    # The directories a tool made go with the last file in them, and no further up
    assert edit.create_file("fresh/deep/c.py", "z = 3\n")["ok"] is True
    assert edit.create_file("scratch/new/d.py", "z = 4\n")["ok"] is True
    assert edit.revert_file("fresh/deep/c.py")["ok"] is True
    assert edit.delete_file("scratch/new/d.py")["ok"] is True
    assert not (repo / "fresh").exists() and not (repo / "scratch" / "new").exists()
    assert (repo / "scratch" / "deep").is_dir() and repo.is_dir()


def test_a_file_takes_the_place_of_a_directory_with_nothing_in_it(repo, monkeypatch):
    (repo / "mod.py").mkdir()
    (repo / "mod.py" / "inner.py").write_text("x = 1\n")
    (repo / "held").mkdir()
    (repo / "held" / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A"), _git(repo, "commit", "-q", "-m", "directories")
    edit.reset_patch_round()
    assert edit.delete_file("mod.py/inner.py")["ok"] is True
    assert (repo / "mod.py").is_dir()
    r = edit.create_file("mod.py", "y = 2\n")
    assert r["ok"] is True and (repo / "mod.py").read_text() == "y = 2\n"
    r = edit.create_file("held", "y = 2\n")
    assert r["ok"] is False and "already exists" in r["error"] and (repo / "held").is_dir()
    # A dry run reports the file and leaves the directory
    (repo / "empty").mkdir()
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    assert edit.create_file("empty", "y = 2\n")["ok"] is True and (repo / "empty").is_dir()


def test_a_create_file_that_is_refused_leaves_the_directories_as_they_are(repo):
    (repo / ".gitignore").write_text("out\n*.log\n")
    for name in ("out", "trace.log", "data"):
        (repo / name).mkdir()
    for path, text, word in (("out", "y = 2\n", "git ignores out"),
                             ("trace.log", "y = 2\n", "git ignores trace.log"),
                             ("data", "\ud800\n", "cannot write"),
                             ("fresh/deep/x.log", "y = 2\n", "git ignores")):
        r = edit.create_file(path, text)
        assert r["ok"] is False and word in r["error"], path
    assert all((repo / name).is_dir() for name in ("out", "trace.log", "data"))
    assert not (repo / "fresh").exists()
    assert edit.write_path_set() == set()


def test_a_create_file_whose_write_fails_leaves_the_directories_as_they_are(repo,
                                                                            monkeypatch):
    """The directories made for the file go again, and an empty directory the file was
    to take the place of is put back."""
    r = edit.create_file("fresh/deep/" + "n" * 300 + ".py", "x = 1\n")
    assert r["ok"] is False and "cannot write" in r["error"]
    assert not (repo / "fresh").exists()
    (repo / "empty").mkdir()
    real_open = open

    def no_space(file, mode="r", *args, **kwargs):
        if "w" in mode and str(file).endswith(("empty", "made/sub/f.py")):
            raise OSError(28, "No space left on device")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", no_space)
    for path in ("empty", "made/sub/f.py"):
        r = edit.create_file(path, "x = 1\n")
        assert r["ok"] is False and "cannot write" in r["error"], path
    assert (repo / "empty").is_dir() and not (repo / "made").exists()
    assert edit.write_path_set() == set()


def test_a_create_file_whose_directories_cannot_all_be_made_leaves_none_of_them(repo):
    """The directories the call made before the one that could not be made go again;
    one an earlier call made, which holds that call's file, stays."""
    r = edit.create_file("n1/n2/" + "x" * 300 + "/f.py", "x = 1\n")
    assert r["ok"] is False and "cannot create directory" in r["error"]
    assert not (repo / "n1").exists()
    old_mask = os.umask(0o277)
    try:
        r = edit.create_file("u1/u2/u3/f.py", "x = 1\n")
    finally:
        os.umask(old_mask)
    if os.geteuid() != 0:
        assert r["ok"] is False and "cannot create directory" in r["error"]
        assert not (repo / "u1").exists()
    assert edit.create_file("k1/keep.py", "x = 1\n")["ok"] is True
    r = edit.create_file("k1/k2/" + "x" * 300 + "/f.py", "x = 1\n")
    assert r["ok"] is False and sorted(os.listdir(repo / "k1")) == ["keep.py"]
    assert edit.delete_file("k1/keep.py")["ok"] is True and not (repo / "k1").exists()
    assert edit.write_path_set() == set()


def test_a_path_too_deep_or_too_long_to_check_is_answered_and_not_raised(repo):
    """More directories than the interpreter's recursion limit are made one by one, and
    a path the system or git cannot be handed comes back as a refusal."""
    deep_path = "/".join(["d"] * 1100) + "/x.py"
    assert edit.create_file(deep_path, "x = 1\n")["ok"] is True
    assert edit.replace_in_file(deep_path, "x = 1", "x = 2")["ok"] is True
    assert edit.delete_file(deep_path)["ok"] is True and not (repo / "d").exists()
    for path in ("/".join(["e"] * 1500) + "/x.py", "/".join(["e"] * 2100) + "/x.py",
                 "n" * 200_000, "a/" * 50_000 + "x.py"):
        r = edit.create_file(path, "x = 1\n")
        if len(path) > 4000:
            assert r == {**r, "ok": False, "error": "cannot check the path: "
                                                    "File name too long"}
        if r["ok"]:
            assert edit.delete_file(path)["ok"] is True
        for call in (lambda: edit.replace_in_file(path, "a", "b"),
                     lambda: edit.delete_file(path), lambda: edit.revert_file(path)):
            assert call()["ok"] is False
    assert not (repo / "e").exists() and edit.own_change_set() == set()


def test_a_submodule_that_is_not_checked_out_does_not_give_way_to_a_file(link_repo):
    assert list((link_repo / "sub").iterdir()) == []
    r = edit.create_file("sub", "y = 2\n")
    assert r["ok"] is False and r["error"] == (
        "sub is a repository of its own, so nothing written there is carried out of "
        "this run. Write what the change needs into a file git tracks")
    assert (link_repo / "sub").is_dir() and edit.own_change_set() == set()


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser lists any directory")
def test_a_directory_that_cannot_be_listed_is_a_path_that_already_exists(repo):
    (repo / "locked").mkdir()
    (repo / "locked").chmod(0)
    try:
        r = edit.create_file("locked", "y = 2\n")
    finally:
        (repo / "locked").chmod(0o755)
    assert r["ok"] is False and "already exists" in r["error"]
    assert (repo / "locked").is_dir()


def test_the_test_directory_made_for_a_stage_outlives_the_tests_it_takes_back(
        repo, monkeypatch):
    """The stage writes a test under the directory made for it and reverts it: the
    directory is still there for the whole run that follows."""
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    edit.reset_patch_round(0)
    edit.make_test_path()
    assert edit.create_file("tests/unit/test_m.py", "def test_m():\n    assert 0\n")["ok"]
    assert edit.revert_file("tests/unit/test_m.py")["ok"] is True
    assert (repo / "tests").is_dir() and not (repo / "tests" / "unit").exists()
    assert edit.has_test_path() and edit.write_path_set() == set()


@pytest.mark.parametrize("link_git", [True, False])
def test_the_link_git_keeps_the_repository_behind_is_not_deleted(repo, tmp_path, link_git):
    if link_git:
        (repo / ".git").rename(tmp_path / "real_git")
        (repo / ".git").symlink_to(tmp_path / "real_git")
    edit.reset_patch_round()
    for path in (".git", "./.git", ".git/config", ".git/HEAD"):
        r = edit.delete_file(path)
        assert r["ok"] is False, (path, r)
        assert "git directory" in r["error"] or "outside the repository" in r["error"]
    assert os.path.lexists(repo / ".git")
    assert _git(repo, "status", "--porcelain") == ""


def test_a_link_that_was_there_at_the_start_is_compared_as_the_link_it_is(repo):
    (repo / "app.py").write_text("x = 1\n")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "a.py").write_text("y = 1\n")
    _git(repo, "add", "-A"), _git(repo, "commit", "-q", "-m", "files")
    (repo / "alias.py").symlink_to("app.py")
    (repo / "cur").symlink_to("pkg")
    edit.reset_patch_round()
    # Writing what a link points at changes the file and not the link
    assert edit.replace_in_file("app.py", "x = 1", "x = 2")["ok"] is True
    assert edit.own_change_set() == edit.write_path_set() == {"app.py"}
    assert edit.no_tool_path_set() == set()
    # Deleting a link is a change, and reverting it puts the link back
    assert edit.delete_file("cur")["ok"] is True
    assert edit.own_change_set() == edit.write_path_set() == {"app.py", "cur"}
    assert edit.revert_file("cur")["ok"] is True
    assert os.readlink(repo / "cur") == "pkg" and (repo / "pkg" / "a.py").exists()
    assert edit.own_change_set() == edit.write_path_set() == {"app.py"}
    # A link made to point elsewhere without a tool is a change of the link's own
    (repo / "alias.py").unlink()
    (repo / "alias.py").symlink_to("pkg/a.py")
    assert edit.no_tool_path_set() == {"alias.py"}


@pytest.mark.parametrize("name", ["GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS",
                                  "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS"])
def test_a_variable_that_changes_how_git_reads_a_path_does_not_reach_git(
        link_repo, monkeypatch, name):
    (link_repo / "pkg").mkdir()
    (link_repo / "pkg" / "a.py").write_text("x = 1\n")
    (link_repo / ".gitignore").write_text("*.log\n")
    _git(link_repo, "add", "pkg", ".gitignore"), _git(link_repo, "commit", "-q", "-m", "pkg")
    monkeypatch.setenv(name, "1")
    edit.reset_patch_round()
    assert edit.replace_in_file("m.py", "old", "new")["ok"] is True
    assert edit.replace_in_file("m.py", "new", "old")["ok"] is True
    assert edit.write_path_set() == set()
    assert edit.replace_in_file("pkg/a.py", "x = 1", "x = 2")["ok"] is True
    assert edit.revert_file("pkg/a.py")["ok"] is True
    assert (link_repo / "pkg" / "a.py").read_text() == "x = 1\n"
    assert _git(link_repo, "status", "--porcelain").strip() in ("", "?? vendor/")
    assert gitcmd.gitlink_of(str(link_repo), "sub/pkg/a.py") == "sub"
    assert edit.create_file("sub/pkg/a.py", "x = 1\n")["ok"] is False
    assert edit.create_file("Out.LOG", "x\n")["ok"] is True
    assert edit.create_file("out.log", "x\n")["ok"] is False
    assert name not in config.child_env()


@pytest.mark.parametrize("name, is_ignored", [
    ("*.log", True), ("a.log", True), ("tracked.log", False), (":(top)x.log", True),
    (":x.log", True), ("[ab].log", True), ("a.txt", False), ("*.txt", False),
    ("-q.log", True), ("sub/*.log", True),
])
def test_a_name_is_asked_about_as_the_path_it_is_and_not_as_a_pattern(repo, name,
                                                                     is_ignored):
    (repo / ".gitignore").write_text("*.log\n")
    (repo / "tracked.log").write_text("x\n")
    (repo / "a.txt").write_text("x\n")
    _git(repo, "add", "-f", ".gitignore", "tracked.log", "a.txt")
    _git(repo, "commit", "-q", "-m", "ignore")
    assert gitcmd.is_ignored(str(repo), name) is is_ignored
    r = edit.create_file(name, "x\n")
    assert ("git ignores" in r["error"]) is is_ignored, r


def test_a_link_whose_directory_is_outside_the_repository_is_not_deleted(repo, tmp_path):
    (tmp_path / "far").mkdir()
    (tmp_path / "far" / "a_link").symlink_to("x")
    (repo / "out_dir").symlink_to(tmp_path / "far")
    r = edit.delete_file("out_dir/a_link")
    assert r["ok"] is False and "outside the repository" in r["error"]
    assert os.path.lexists(tmp_path / "far" / "a_link")


@pytest.mark.parametrize("call, want", [
    (lambda: edit.replace_line_range("m.py", "1", 1, "new"),
     "start_line must be a whole number, not '1'"),
    (lambda: edit.replace_line_range("m.py", 1, 1.0, "new"),
     "end_line must be a whole number, not 1.0"),
    (lambda: edit.replace_line_range("m.py", True, True, "new"),
     "start_line must be a whole number, not True"),
    (lambda: edit.replace_line_range("m.py", 1, 1, None),
     "new_text must be a string, not NoneType"),
    (lambda: edit.replace_in_file("m.py", b"old", "new"),
     "old_text must be a string, not bytes"),
    (lambda: edit.replace_in_file("m.py", "old", 5), "new_text must be a string, not int"),
    (lambda: edit.create_file("n.py", b"x = 1\n"), "text must be a string, not bytes"),
    (lambda: edit.create_file(None, "x = 1\n"), "a path must be a string, not NoneType"),
    (lambda: edit.delete_file(7), "a path must be a string, not int"),
])
def test_an_argument_of_the_wrong_type_is_refused_and_not_raised(repo, call, want):
    r = call()
    assert r["ok"] is False and r["error"] == want
    assert (repo / "m.py").read_text() == "old\nctx\n" and not (repo / "n.py").exists()
    assert edit.revert_file(3) == {"ok": False, "path": 3,
                                   "error": "a path must be a string, not int"}


def test_a_form_feed_line_is_one_line_to_the_editing_tools(repo):
    """The analysis numbers lines at "\\n" alone; str.splitlines() also breaks at the
    form feed and moved every line after it down by one."""
    (repo / "m.py").write_text("a = 1\n\x0c\nb = 2\nc = 3\n")
    r = edit.replace_line_range("m.py", 3, 3, "b = 22")
    assert r["ok"] is True and "was: b = 2" in r["stat"]
    assert (repo / "m.py").read_text() == "a = 1\n\x0c\nb = 22\nc = 3\n"
    assert edit.replace_in_file("m.py", "c = 3", "c = 33")["stat"] == \
        "m.py line 4 | +1 -1 lines"


def test_a_line_that_ends_with_a_lone_carriage_return_is_not_joined_to_the_next(repo):
    (repo / "m.txt").write_bytes(b"a\rb\rc\r")
    assert edit.replace_line_range("m.txt", 2, 2, "B")["ok"] is True
    assert (repo / "m.txt").read_bytes() == b"a\rB\rc\r"


def test_a_carriage_return_in_front_of_a_line_break_is_a_line_to_the_editing_tools(repo):
    """The line numbers and the text the reading tools give name the same lines here."""
    raw = b"a = 1\r\r\nb = 2\r\r\nc = 3\r\r\n"
    (repo / "m.py").write_bytes(raw)
    shown = file_utils.read_source_text(str(repo / "m.py"))
    assert file_utils.line_list_of(shown) == ["a = 1", "", "b = 2", "", "c = 3", ""]
    r = edit.replace_line_range("m.py", 3, 3, "b = 22")
    assert r["ok"] is True and r["stat"] == "m.py lines 3-3 | +1 -1 lines | was: b = 2"
    assert (repo / "m.py").read_bytes() == b"a = 1\r\r\nb = 22\r\n\r\nc = 3\r\r\n"
    assert edit.replace_line_range("m.py", 7, 7, "d = 4")["ok"] is False
    r = edit.replace_in_file("m.py", "\nc = 3\n", "\nc = 33\n")
    assert r["ok"] is True and r["stat"] == "m.py line 4 | +2 -2 lines"
    assert (repo / "m.py").read_bytes() == b"a = 1\r\r\nb = 22\r\n\r\nc = 33\r\n\r\n"


@pytest.mark.parametrize("ending", ["\r\n", "\r", "\n"])
@pytest.mark.parametrize("text_ending", ["\r\n", "\r", "\n"])
def test_the_line_breaks_of_a_text_a_tool_is_handed_are_written_as_the_file_s(
        repo, ending, text_ending):
    """Whichever of the three line breaks the text holds, no line is added or joined."""
    def stored(text: str) -> bytes:
        return text.replace("\n", ending).encode()

    def handed(text: str) -> str:
        return text.replace("\n", text_ending)

    (repo / "m.txt").write_bytes(stored("a = 1\nb = 2\nc = 3\n"))
    r = edit.replace_line_range("m.txt", 2, 2, handed("b = 20\nb = 21\n"))
    assert r["ok"] is True and r["stat"].startswith("m.txt lines 2-2 | +2 -1 lines")
    assert (repo / "m.txt").read_bytes() == stored("a = 1\nb = 20\nb = 21\nc = 3\n")
    r = edit.replace_in_file("m.txt", handed("b = 20\nb = 21\n"), handed("b = 2\nx\n"))
    assert r["ok"] is True and r["stat"] == "m.txt line 2 | +2 -2 lines"
    assert (repo / "m.txt").read_bytes() == stored("a = 1\nb = 2\nx\nc = 3\n")
    # A new file is written with "\n", whatever the text was handed with
    assert edit.create_file("n.txt", handed("x = 1\ny = 2"))["stat"] == "new file, 2 lines"
    assert (repo / "n.txt").read_bytes() == b"x = 1\ny = 2\n"


def test_the_text_a_reading_tool_gave_matches_in_a_file_of_lone_carriage_returns(repo):
    (repo / "m.txt").write_bytes(b"a = 1\rb = 2\rc = 3\r")
    r = edit.replace_in_file("m.txt", "a = 1\nb = 2\n", "a = 1\nb = 22\n")
    assert r["ok"] is True and r["stat"] == "m.txt line 1 | +2 -2 lines"
    assert (repo / "m.txt").read_bytes() == b"a = 1\rb = 22\rc = 3\r"


def test_replace_lines_replaces_a_range(lines_repo):
    r = edit.replace_line_range("m.py", 2, 3, "new2\nnew3")
    assert r["ok"] is True
    assert (lines_repo / "m.py").read_text() == "l1\nnew2\nnew3\nl4\nl5\n"


def test_replace_lines_can_replace_a_single_line(lines_repo):
    assert edit.replace_line_range("m.py", 5, 5, "last")["ok"] is True
    assert (lines_repo / "m.py").read_text() == "l1\nl2\nl3\nl4\nlast\n"


def test_replace_lines_deletes_the_range_with_an_empty_new_text(lines_repo):
    assert edit.replace_line_range("m.py", 2, 4, "")["ok"] is True
    assert (lines_repo / "m.py").read_text() == "l1\nl5\n"


def test_replace_lines_does_not_join_lines_without_a_trailing_newline(lines_repo):
    """A new_text without a trailing newline is a common way for the model to write."""
    assert edit.replace_line_range("m.py", 2, 2, "x")["ok"] is True
    assert (lines_repo / "m.py").read_text() == "l1\nx\nl3\nl4\nl5\n"


def test_replace_lines_keeps_the_last_lines_newline_as_it_was(repo):
    (repo / "m.py").write_text("a\nb")
    assert edit.replace_line_range("m.py", 2, 2, "c")["ok"] is True
    assert (repo / "m.py").read_text() == "a\nc"


def test_replace_lines_stat_echoes_the_first_3_replaced_lines(lines_repo):
    """The value that lets the model notice on the spot that its lines had drifted."""
    r = edit.replace_line_range("m.py", 1, 5, "x")
    assert "m.py lines 1-5" in r["stat"]
    assert "+1 -5 lines" in r["stat"]
    assert "was: l1 / l2 / l3" in r["stat"]
    assert "l4" not in r["stat"]


def test_replace_lines_rejects_an_end_line_past_the_file_without_writing(lines_repo):
    r = edit.replace_line_range("m.py", 4, 99, "x")
    assert r["ok"] is False
    assert "has only 5 lines" in r["error"]
    assert (lines_repo / "m.py").read_text() == "l1\nl2\nl3\nl4\nl5\n"


def test_replace_lines_rejects_a_start_line_of_0_or_less(lines_repo):
    r = edit.replace_line_range("m.py", 0, 2, "x")
    assert r["ok"] is False and "must be 1 or more" in r["error"]


def test_replace_lines_rejects_an_end_line_below_start_line(lines_repo):
    r = edit.replace_line_range("m.py", 3, 2, "x")
    assert r["ok"] is False and "must be start_line or more" in r["error"]


def test_replace_lines_writes_nothing_when_the_contents_match(lines_repo):
    r = edit.replace_line_range("m.py", 2, 2, "l2")
    assert r["ok"] is False and "already match new_text" in r["error"]


def test_replace_lines_rejects_a_path_outside_project_dir(lines_repo):
    out = edit.replace_line_range("../outside.py", 1, 1, "x")
    assert out["ok"] is False and "outside the repository" in out["error"]


def test_replace_lines_rejects_a_path_under_output_dir(tmp_path, monkeypatch):
    """The index's output is not the original source and must never be edited."""
    root = tmp_path / "proj"
    out = root / "out"
    out.mkdir(parents=True)
    (out / "m.py").write_text("old\nctx\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(out))
    out_dict = edit.replace_line_range("out/m.py", 1, 1, "new")
    assert out_dict["ok"] is False and "OUTPUT_DIR" in out_dict["error"]
    assert (out / "m.py").read_text() == "old\nctx\n"


def test_replace_lines_returns_a_diff_without_writing_under_dry_run(lines_repo,
                                                                   monkeypatch):
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    r = edit.replace_line_range("m.py", 2, 2, "new2")
    assert r["ok"] is True and r["dry_run"] is True
    assert "-l2" in r["diff"] and "+new2" in r["diff"]
    assert (lines_repo / "m.py").read_text() == "l1\nl2\nl3\nl4\nl5\n"


def test_replace_lines_is_counted_in_the_patch_stats(lines_repo):
    edit.replace_line_range("m.py", 1, 1, "a")
    edit.replace_line_range("m.py", 99, 99, "b")
    assert edit.patch_count_dict() == {"ok": 1, "ng": 1, "ng_after_ok": 1}


# == create_file (a feature needing a new module) =====
def test_create_file_writes_a_new_file(repo):
    r = edit.create_file("cache.py", "x = 1\n")
    assert r["ok"] is True
    assert (repo / "cache.py").read_text() == "x = 1\n"
    assert "new file" in r["stat"]


def test_create_file_makes_the_parent_directories(repo):
    assert edit.create_file("pkg/sub/cache.py", "x = 1\n")["ok"] is True
    assert (repo / "pkg" / "sub" / "cache.py").read_text() == "x = 1\n"


def test_create_file_adds_a_missing_trailing_newline(repo):
    edit.create_file("cache.py", "x = 1")
    assert (repo / "cache.py").read_text() == "x = 1\n"


def test_create_file_refuses_a_path_that_exists(repo):
    r = edit.create_file("m.py", "x = 1\n")
    assert r["ok"] is False and "already exists" in r["error"]
    assert (repo / "m.py").read_text() == "old\nctx\n"


def test_an_empty_file_that_is_already_there_can_be_filled(repo):
    """create_file fills an empty file: replace_line_range has no line to name in it
    and replace_in_file no old_text to match.
    """
    (repo / "pkg_init.py").write_text("")
    lines = edit.replace_line_range("pkg_init.py", 1, 1, "x = 1")
    assert lines["ok"] is False
    assert "is empty" in lines["error"] and "create_file" in lines["error"]
    assert edit.replace_in_file("pkg_init.py", "", "x = 1")["ok"] is False
    r = edit.create_file("pkg_init.py", "x = 1")
    assert r["ok"] is True
    assert (repo / "pkg_init.py").read_text() == "x = 1\n"


@pytest.mark.parametrize("bom, codec", [
    (codecs.BOM_UTF8, "utf-8"), (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"), (codecs.BOM_UTF32_LE, "utf-32-le")])
def test_a_file_of_a_bom_alone_can_be_filled_and_keeps_the_bom(repo, bom, codec):
    """replace_line_range sends it to create_file, and create_file takes it."""
    (repo / "bom.py").write_bytes(bom)
    lines = edit.replace_line_range("bom.py", 1, 1, "x = 1")
    assert lines["ok"] is False
    assert "is empty" in lines["error"] and "create_file" in lines["error"]
    r = edit.create_file("bom.py", "x = 1\n# 表\n")
    assert r["ok"] is True and r["stat"] == "new file, 2 lines"
    assert (repo / "bom.py").read_bytes() == bom + "x = 1\n# 表\n".encode(codec)
    # It holds lines now, and is the other two tools' to change
    assert "already exists" in edit.create_file("bom.py", "y = 2\n")["error"]
    assert edit.replace_line_range("bom.py", 1, 1, "x = 2")["ok"] is True
    assert (repo / "bom.py").read_bytes() == bom + "x = 2\n# 表\n".encode(codec)


def test_a_file_with_a_bom_and_text_is_not_create_file_s_to_write(repo):
    (repo / "bom.py").write_bytes(codecs.BOM_UTF8 + b" ")
    assert "already exists" in edit.create_file("bom.py", "x = 1\n")["error"]
    assert (repo / "bom.py").read_bytes() == codecs.BOM_UTF8 + b" "


def test_create_file_refuses_an_empty_text(repo):
    assert edit.create_file("cache.py", "")["ok"] is False
    assert not (repo / "cache.py").exists()


def test_create_file_rejects_a_path_outside_project_dir(repo):
    out = edit.create_file("../escaped.py", "x = 1\n")
    assert out["ok"] is False and "outside the repository" in out["error"]


def test_create_file_rejects_a_path_under_output_dir(tmp_path, monkeypatch, repo):
    out = repo / "out"
    out.mkdir()
    monkeypatch.setattr(config, "OUTPUT_DIR", str(out))
    out_dict = edit.create_file("out/cache.py", "x = 1\n")
    assert out_dict["ok"] is False and "OUTPUT_DIR" in out_dict["error"]


def test_create_file_leaves_no_trace_under_dry_run(repo, monkeypatch):
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    r = edit.create_file("pkg/cache.py", "x = 1\n")
    assert r["ok"] is True and r["dry_run"] is True
    assert "+x = 1" in r["diff"]
    assert not (repo / "pkg").exists()


def test_create_file_is_counted_in_the_patch_stats(repo):
    edit.create_file("cache.py", "x = 1\n")
    edit.create_file("m.py", "x = 1\n")
    assert edit.patch_count_dict() == {"ok": 1, "ng": 1, "ng_after_ok": 1}


def test_replace_in_file_points_at_create_file_when_the_file_is_missing(repo):
    r = edit.replace_in_file("nope.py", "old", "new")
    assert r["ok"] is False and "create_file" in r["error"]


@pytest.fixture
def fake_test_command(monkeypatch):
    """Return a helper installing a fake test command with a given exit code/output."""
    def use(exit_code: int, out: str):
        monkeypatch.setattr(config, "TEST_COMMAND",
                            [sys.executable, "-c",
                             f"import sys; print({out!r}); sys.exit({exit_code})"])
    return use


# A command that writes the time to the file its first argument names, and passes
_WRITE_TIME_ARGV = [sys.executable, "-c",
                    "import sys, time; open(sys.argv[1], 'w').write(str(time.time_ns()))"
                    "; print('1 passed in 0.01s')"]


@pytest.fixture
def report_repo(repo, monkeypatch):
    """The repository with a tracked report.txt and a test command that rewrites it."""
    (repo / "tests").mkdir()
    (repo / "tests" / "t.py").write_text("", encoding="utf-8")
    (repo / "report.txt").write_text("0\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "report")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_COMMAND", [*_WRITE_TIME_ARGV, "report.txt"])
    edit.reset_patch_round()
    return repo


def test_what_the_test_command_wrote_is_told_apart_from_what_the_request_wrote(
        report_repo):
    assert edit.command_write_set() == set()
    assert edit.run_test()["status"] == "passed"
    assert edit.own_change_set() == edit.command_write_set() == {"report.txt"}
    # A run over some paths writes it again
    assert edit.run_test(["tests/t.py"])["status"] == "passed"
    assert edit.command_write_set() == {"report.txt"}
    # What the request writes there itself is the request's, and stays so when a run
    # of the tests writes over it
    (report_repo / "report.txt").write_text("mine\n", encoding="utf-8")
    assert edit.command_write_set() == set()
    edit.run_test()
    assert edit.own_change_set() == {"report.txt"} and edit.command_write_set() == set()
    # Put back to what it held at the start, the next run's write is the command's again
    assert edit.revert_file("report.txt")["ok"] is True
    edit.run_test()
    assert edit.command_write_set() == {"report.txt"}
    edit.reset_patch_round()
    assert edit.command_write_set() == set()


def test_a_file_the_request_wrote_is_not_the_test_commands_after_a_run(report_repo):
    (report_repo / "m.py").write_text("new\nctx\n", encoding="utf-8")
    edit.run_test()
    assert edit.own_change_set() == {"m.py", "report.txt"}
    assert edit.command_write_set() == {"report.txt"}


def test_what_the_static_check_wrote_is_told_apart_too(report_repo, monkeypatch):
    monkeypatch.setattr(config, "CHECK_COMMAND", [*_WRITE_TIME_ARGV, "check.cache"])
    assert _RUN_CHECK() == {}
    assert edit.command_write_set() == {"check.cache"}


def test_what_the_program_of_the_automatic_check_wrote_is_told_apart_too(report_repo,
                                                                        monkeypatch):
    """The program the automatic check starts writes a file git tracks, which is no
    change of the stage since the last whole run."""
    from codetwine_loom.tools import run
    monkeypatch.setattr(config, "RUN_COMMAND", [*_WRITE_TIME_ARGV, "m.py", "--"])
    run.reset_run_state()
    edit.run_test()
    assert edit.command_write_set() == {"report.txt"}
    stage_dict = edit.stage_change_hash_dict()
    assert run.run_once([])["ok"] is True
    assert edit.own_change_set() == {"m.py", "report.txt"}
    assert edit.command_write_set() == {"m.py", "report.txt"}
    assert edit.last_whole_run()["change"] == edit.stage_change_hash_dict() == stage_dict
    run.reset_run_state()


def test_what_a_program_the_model_started_wrote_is_the_request_s_own_change(
        report_repo, monkeypatch):
    """The model chose the program and its arguments, so what it wrote is a change
    made since the last whole run, as its own open() would be."""
    from codetwine_loom import plan
    from codetwine_loom.tools import run
    monkeypatch.setattr(config, "RUN_COMMAND", [*_WRITE_TIME_ARGV, "m.py", "--"])
    run.reset_run_state()
    edit.run_test()
    stage_dict = edit.stage_change_hash_dict()
    assert plan._test_line_list(1) == [
        "The whole test suite, the last time you ran it (on turn 1): passed."]
    assert run.run_project([])["ok"] is True
    assert edit.own_change_set() == edit.no_tool_path_set() == {"m.py", "report.txt"}
    assert edit.command_write_set() == {"report.txt"}
    assert set(edit.stage_change_hash_dict()) == set(stage_dict) | {"m.py"}
    assert plan._test_line_list(1)[1:] == ["You have changed a file since that run."]
    # The test command run after it leaves the file as the request's
    edit.run_test()
    assert edit.command_write_set() == {"report.txt"}
    run.reset_run_state()


def test_what_the_formatting_command_wrote_beside_the_files_is_the_command_s(
        report_repo, monkeypatch):
    (report_repo / "m.py").write_text("mine\nctx\n", encoding="utf-8")
    monkeypatch.setattr(config, "FORMAT_COMMAND", [*_WRITE_TIME_ARGV, "format.cache"])
    assert edit.run_format()["exit_status"] == 0
    assert edit.own_change_set() == {"m.py", "format.cache"}
    assert edit.command_write_set() == {"format.cache"}


def test_run_tests_reads_output_that_is_not_utf8(repo, monkeypatch):
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c",
        "import sys; sys.stdout.buffer.write(b'caf\\xe9 1 passed in 0.10s\\n')"])
    r = edit.run_test()
    assert r["status"] == "passed" and "caf\ufffd 1 passed" in r["output"]


def test_run_tests_returns_ok_and_targets_when_it_passes(repo, fake_test_command,
                                                         monkeypatch):
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    fake_test_command(0, "6 passed in 0.10s")
    r = edit.run_test()
    assert r["status"] == "passed"
    assert r["failed"] == []
    assert r["failed_excerpt"] == ""
    assert r["targets"] == ["tests"]


def test_the_last_run_of_the_whole_suite_is_kept(repo, fake_test_command, monkeypatch):
    """integrate.make_state() reads it: a run over some paths, or one that did not run,
    is not what the whole suite says."""
    (repo / "tests").mkdir()
    (repo / "tests" / "t.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    edit.reset_patch_round()
    assert edit.last_whole_run() is None
    fake_test_command(1, _SUMMARY_HEAD + "FAILED tests/t.py::t_x - AssertionError\n1 passed")
    edit.run_test()
    kept = edit.last_whole_run()
    assert kept["n"] == 1 and kept["result"]["failed"] == ["tests/t.py::t_x"]
    fake_test_command(0, "2 passed")
    edit.run_test(["tests/t.py"])
    assert edit.last_whole_run() == kept
    monkeypatch.setattr(config, "TEST_COMMAND", ["/no/such/command"])
    edit.run_test()
    assert edit.last_whole_run() == kept
    fake_test_command(0, "2 passed")
    edit.run_test()
    assert edit.last_whole_run()["n"] == 2
    assert edit.last_whole_run()["result"]["status"] == "passed"
    edit.reset_patch_round()
    assert edit.last_whole_run() is None


def test_the_last_whole_run_keeps_what_the_stage_had_changed_as_it_ended(
        repo, fake_test_command, monkeypatch):
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    edit.reset_patch_round()
    (repo / "tests" / "t.py").write_text("x = 1\n", encoding="utf-8")
    before_dict = edit.stage_change_hash_dict()
    fake_test_command(0, "2 passed")
    edit.run_test()
    (repo / "tests" / "t.py").write_text("x = 2\n", encoding="utf-8")
    assert before_dict and edit.last_whole_run()["change"] == before_dict
    assert edit.last_whole_run()["change"] != edit.stage_change_hash_dict()


def test_what_the_test_command_wrote_is_no_change_of_the_stage_since_the_whole_run(
        report_repo):
    """The command writes report.txt anew on every run, whole or over some paths."""
    assert edit.replace_in_file("m.py", "old", "new")["ok"] is True
    stage_dict = edit.stage_change_hash_dict()
    assert set(stage_dict) == {"m.py"}
    edit.run_test()
    assert edit.own_change_set() == {"m.py", "report.txt"}
    assert edit.last_whole_run()["change"] == edit.stage_change_hash_dict() == stage_dict
    edit.run_test(["tests/t.py"])
    assert edit.last_whole_run()["change"] == edit.stage_change_hash_dict()
    # What the stage writes there itself is a change of its own
    (report_repo / "report.txt").write_text("mine\n", encoding="utf-8")
    assert set(edit.stage_change_hash_dict()) == {"m.py", "report.txt"}
    assert edit.last_whole_run()["change"] != edit.stage_change_hash_dict()


class _Stop(BaseException):
    """What a test stops a command with, as the turn's time limit and Ctrl-C do."""


def _stop_after_run(monkeypatch):
    """Have every command edit.run_in_project() starts run to its end and then stop."""
    run = edit.run_in_project

    def run_and_stop(argv, timeout_sec, extra_env_dict=None, directory="",
                     keep_line=None):
        run(argv, timeout_sec, extra_env_dict, directory, keep_line=keep_line)
        raise _Stop()

    monkeypatch.setattr(edit, "run_in_project", run_and_stop)


def test_what_a_test_command_wrote_before_it_was_stopped_is_recorded_as_the_commands(
        report_repo, monkeypatch):
    with monkeypatch.context() as stop_patch:
        _stop_after_run(stop_patch)
        with pytest.raises(_Stop):
            edit.run_test()
    assert edit.own_change_set() == edit.command_write_set() == {"report.txt"}
    # The next run that ends writes over what the stopped one left, as the command's
    assert edit.run_test()["status"] == "passed"
    assert edit.command_write_set() == {"report.txt"}


def test_what_a_static_check_wrote_before_it_was_stopped_is_recorded_too(
        report_repo, monkeypatch):
    monkeypatch.setattr(config, "CHECK_COMMAND", [*_WRITE_TIME_ARGV, "check.cache"])
    _stop_after_run(monkeypatch)
    with pytest.raises(_Stop):
        _RUN_CHECK()
    assert edit.command_write_set() == {"check.cache"}


def _second_of(path) -> int:
    """Return the second a file was last modified in."""
    return int(os.stat(path).st_mtime)


def test_a_file_a_tool_writes_is_dated_after_the_one_it_replaces(repo):
    """Each write lands inside the second of the one before it, or before it."""
    path = repo / "m.py"
    start = int(time.time()) + 5
    os.utime(path, (start, start))
    assert edit.replace_in_file("m.py", "old", "new")["ok"] is True
    assert _second_of(path) == start + 1
    assert edit.replace_line_range("m.py", 1, 1, "NEW")["ok"] is True
    assert _second_of(path) == start + 2
    assert edit.revert_file("m.py")["ok"] is True
    assert path.read_text() == "old\nctx\n" and _second_of(path) == start + 3
    # A write over a file last modified in an earlier second keeps the time it is
    # written at, and a new file has no time to be dated after
    os.utime(path, (start - 60, start - 60))
    assert edit.replace_in_file("m.py", "old", "new")["ok"] is True
    assert start - 6 <= _second_of(path) <= int(time.time())
    assert edit.create_file("n.py", "x = 1\n")["ok"] is True
    assert _second_of(repo / "n.py") <= int(time.time())


def test_a_file_put_back_to_what_it_held_before_the_request_is_dated_after_too(repo):
    """The file put back is as long as it was at the start and is written in the
    second it was dated with then; the write between the two was of another size."""
    path = repo / "m.py"
    path.write_text("mine\nctx\n", encoding="utf-8")
    edit.reset_patch_round()
    time.sleep(1.02 - time.time() % 1)
    start = int(time.time())
    os.utime(path, (start, start))
    assert edit.replace_in_file("m.py", "mine", "new")["ok"] is True
    assert _second_of(path) == start
    assert edit.revert_file("m.py")["ok"] is True
    assert path.read_text() == "mine\nctx\n" and _second_of(path) == start + 1


def test_a_file_written_where_a_tool_removed_one_is_dated_after_the_one_removed(repo):
    """create_file after delete_file, and after revert_file removed a new file, in the
    second of the file that was removed and as long as it."""
    time.sleep(1.02 - time.time() % 1)
    start = int(time.time())
    assert edit.create_file("n.py", "x = 1\n")["ok"] is True
    assert edit.delete_file("n.py")["ok"] is True
    assert edit.create_file("n.py", "x = 2\n")["ok"] is True
    assert _second_of(repo / "n.py") == start + 1
    assert edit.revert_file("n.py")["ok"] is True and not (repo / "n.py").exists()
    assert edit.create_file("n.py", "x = 3\n")["ok"] is True
    assert _second_of(repo / "n.py") == start + 2
    # A file of another size keeps the time it is written at
    assert edit.delete_file("n.py")["ok"] is True
    assert edit.create_file("n.py", "x = 1234\n")["ok"] is True
    assert _second_of(repo / "n.py") == start


def test_run_tests_picks_up_the_failing_test_names(repo, fake_test_command,
                                                   monkeypatch):
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    fake_test_command(1, _SUMMARY_HEAD + "FAILED tests/t.py::t_x - AssertionError\n1 passed")
    r = edit.run_test()
    assert r["status"] == "failed"
    assert r["failed"] == ["tests/t.py::t_x"]


def test_a_test_that_errored_in_its_setup_is_named_as_failing(repo, fake_test_command,
                                                             monkeypatch):
    """A fixture that broke is a red test like any other, beside one already red."""
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    fake_test_command(1, _SUMMARY_HEAD + "FAILED tests/test_a.py::test_old_red - assert 0\n"
                         "ERROR tests/test_b.py::test_uses_fixture - RuntimeError: broken\n"
                         "ERROR tests/test_c.py - ImportError: no module\n"
                         "1 failed, 2 errors in 0.10s")
    assert edit.run_test()["failed"] == ["tests/test_a.py::test_old_red",
                                         "tests/test_b.py::test_uses_fixture",
                                         "tests/test_c.py"]


def test_failed_excerpt_runs_from_the_failure_line_to_the_end(repo, fake_test_command,
                                                              monkeypatch):
    """output keeps its tail and the REPL keeps the head, so the excerpt comes first."""
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    fake_test_command(1, "collecting ...\nFAILED tests/t.py::t_x - AssertionError")
    r = edit.run_test()
    assert r["failed_excerpt"] == "FAILED tests/t.py::t_x - AssertionError"
    assert "collecting" not in r["failed_excerpt"]
    # It comes before output in the returned dict (so print(res) shows it first)
    assert list(r).index("failed_excerpt") < list(r).index("output")


def test_failed_excerpt_is_empty_when_the_tests_pass(repo, fake_test_command,
                                                     monkeypatch):
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    fake_test_command(0, "6 passed in 0.10s")
    assert edit.run_test()["failed_excerpt"] == ""


def test_failed_excerpt_falls_back_to_the_tail_when_no_failure_line_parses(
        repo, fake_test_command, monkeypatch):
    """For Java or TypeScript, where a different reporter makes _FAIL_RE_DICT miss."""
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "FAIL_EXCERPT_CHAR", 20)
    fake_test_command(1, "x" * 100 + "the real reason")
    r = edit.run_test()
    assert r["failed"] == []
    assert r["failed_excerpt"].endswith("the real reason")
    assert r["failed_excerpt"].startswith("... (95 chars omitted)\n")
    assert len(r["failed_excerpt"].split("\n", 1)[1]) == 20


def test_failed_excerpt_is_cut_at_failed_excerpt_chars(repo, fake_test_command,
                                                       monkeypatch):
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "FAIL_EXCERPT_CHAR", 30)
    fake_test_command(1, "FAILED tests/t.py::t_x - " + "y" * 200)
    excerpt = edit.run_test()["failed_excerpt"]
    assert excerpt == "... (195 chars omitted)\n" + "y" * 30


def test_a_cut_test_output_starts_with_how_much_was_omitted(repo, fake_test_command,
                                                            monkeypatch):
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", 10)
    fake_test_command(1, "x" * 100 + "1 failed\n")
    output = edit.run_test()["output"]
    assert output.startswith("... (") and "chars omitted)\n" in output
    assert output.split("\n", 1)[1].startswith("1 failed") and len(output.split("\n", 1)[1]) == 10


def test_run_tests_returns_guidance_when_tests_is_missing(repo, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    r = edit.run_test()
    assert r["status"] == "did not run"
    assert "TEST_PATH" in r["output"]


def test_tests_turned_off_says_so_rather_than_naming_test_path(repo, monkeypatch):
    """test_can_run() answers False for two reasons; naming the wrong one sends
    whoever reads it - the model, or the warm run's log - after the wrong setting."""
    monkeypatch.setattr(config, "RUN_TEST", False)
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    r = edit.run_test()
    assert r["status"] == "did not run"
    assert "RUN_TEST" in r["output"] and "TEST_PATH" not in r["output"]


def test_a_language_with_no_test_command_says_which_setting_to_set(repo, monkeypatch):
    """SQL has no default test command: the tool is not handed over, and a call names
    TEST_COMMAND rather than TEST_PATH or RUN_TEST. Explicit paths change nothing,
    since there is no command to put them after."""
    monkeypatch.setattr(config, "TEST_COMMAND", [])
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    assert not edit.test_can_run()
    for r in (edit.run_test(), edit.run_test(["tests/a.sql"])):
        assert r["status"] == "did not run"
        assert "TEST_COMMAND" in r["output"]
        assert "TEST_PATH" not in r["output"] and "RUN_TEST" not in r["output"]


def test_an_empty_list_also_goes_through_the_test_path_gate(repo, monkeypatch):
    """Written as "paths is None", only run_test([]) would slip past."""
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    r = edit.run_test([])
    assert r["status"] == "did not run"
    assert "TEST_PATH" in r["output"]


def test_run_tests_validates_its_paths(repo, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    r = edit.run_test(["../../etc"])
    assert r["status"] == "did not run"
    assert "outside the repository" in r["output"]


@pytest.mark.parametrize("style", ["path", "separator"])
@pytest.mark.parametrize("flag", ["--basetemp=src", "-p"])
def test_run_tests_refuses_a_path_that_is_an_option(repo, fake_test_command, monkeypatch,
                                                    style, flag):
    """The command never starts, so the text never reaches it as an option."""
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", style)
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    fake_test_command(0, "1 passed in 0.10s")
    r = edit.run_test([flag, "tests"])
    assert r["status"] == "did not run" and r["targets"] == []
    assert flag in r["output"] and "cannot start with" in r["output"]


def test_the_gate_applies_only_when_style_is_path(repo, fake_test_command,
                                                  monkeypatch):
    """In a language where TEST_PATH never reaches argv, a missing one is fine."""
    monkeypatch.setattr(config, "PRIMARY_EXT", "java")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_PATH", "src/test/java")  # does not exist
    fake_test_command(0, "Tests run: 3, Failures: 0, Errors: 0, Skipped: 0")
    r = edit.run_test()
    assert r["status"] == "passed"


def test_run_tests_notes_that_none_dropped_the_paths(repo, fake_test_command,
                                                     monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "java")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(0, "Tests run: 1, Failures: 0, Errors: 0")
    r = edit.run_test(["m.py"])
    assert r["targets"] == []
    assert r["output"].startswith("note:")


def test_separator_puts_the_targets_after_the_separator(repo, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "js")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "separator")
    monkeypatch.setattr(config, "TEST_COMMAND", ["nonexistent-build-tool"])
    r = edit.run_test(["m.py"])
    assert r["status"] == "did not run"
    assert "nonexistent-build-tool cannot be started (No such file or directory)" in \
        r["output"]
    assert r["targets"] == ["m.py"]


def test_ok_comes_from_the_exit_code_without_reading_the_output(repo,
                                                                fake_test_command,
                                                                monkeypatch):
    """mvn -q prints no count line on success. ok is True even with empty output."""
    monkeypatch.setattr(config, "PRIMARY_EXT", "java")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(0, "")
    r = edit.run_test()
    assert r["status"] == "passed"
    assert r["failed"] == []


def test_run_tests_returns_guidance_when_the_command_is_missing(repo, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_COMMAND", ["nonexistent-build-tool"])
    r = edit.run_test()
    assert r["status"] == "did not run"
    assert "nonexistent-build-tool cannot be started (No such file or directory)" in \
        r["output"]


def test_run_tests_returns_timeout_when_it_runs_too_long(repo, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_COMMAND",
                        [sys.executable, "-c", "import time; time.sleep(5)"])
    monkeypatch.setattr(config, "TEST_TIMEOUT_SEC", 1)
    r = edit.run_test()
    assert r["status"] == "did not run"
    assert "timeout" in r["output"]


# A test command that starts a child of its own, which writes late.txt into the
# repository once the command's time is up, and then waits far past that time
_LEAVE_A_CHILD = ("import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', "
                  "'import time; time.sleep(1.5); open(\"late.txt\", \"w\").write(\"x\")']); "
                  "time.sleep(30)")


def test_a_timeout_stops_what_the_test_command_started_too(repo, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_COMMAND", [sys.executable, "-c", _LEAVE_A_CHILD])
    monkeypatch.setattr(config, "TEST_TIMEOUT_SEC", 1)
    assert "timeout" in edit.run_test()["output"]
    time.sleep(1.5)
    assert not (repo / "late.txt").exists()


def test_the_turn_time_limit_stops_what_the_test_command_started_too(repo, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_COMMAND", [sys.executable, "-c", _LEAVE_A_CHILD])
    monkeypatch.setattr(config, "TEST_TIMEOUT_SEC", 60)
    monkeypatch.setattr(config, "TURN_TIMEOUT_SEC", 1)
    with pytest.raises(rlm.TurnTimeout), rlm.turn_deadline():
        edit.run_test()
    time.sleep(1.5)
    assert not (repo / "late.txt").exists()


def test_a_command_that_finishes_stops_what_it_left_running(repo):
    """The child keeps no pipe of the command open, so the command ends before it does."""
    leave = ("import subprocess, sys; subprocess.Popen([sys.executable, '-c', "
             "'import time; time.sleep(1.5); open(\"late.txt\", \"w\").write(\"x\")'], "
             "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)")
    proc = edit.run_in_project([sys.executable, "-c", leave], 30)
    assert proc.returncode == 0
    time.sleep(2)
    assert not (repo / "late.txt").exists()


def test_a_command_that_ends_is_not_waited_for_past_its_end(repo):
    """The child holds the command's output open and outlives it."""
    leave = ("import subprocess, sys; subprocess.Popen([sys.executable, '-c', "
             "'import time; time.sleep(1.5); open(\"late.txt\", \"w\").write(\"x\"); "
             "time.sleep(30)']); print('1 passed'); print('note', file=sys.stderr)")
    start = time.monotonic()
    proc = edit.run_in_project([sys.executable, "-c", leave], 20)
    assert time.monotonic() - start < 10
    assert (proc.returncode, proc.stdout, proc.stderr) == (0, "1 passed\n", "note\n")
    time.sleep(2)
    assert not (repo / "late.txt").exists()


def test_the_tests_are_read_when_the_command_left_something_holding_its_output(
        repo, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c",
        "import subprocess, sys; subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)']); print('1 passed in 0.01s')"])
    monkeypatch.setattr(config, "TEST_TIMEOUT_SEC", 20)
    r = edit.run_test()
    assert r["status"] == "passed" and "1 passed in 0.01s" in r["output"]


@pytest.fixture
def _no_start_repo(repo):
    """Add a file without the execute bit, one the system cannot run and a directory."""
    (repo / "plain.sh").write_text("#!/bin/sh\necho hi\n")
    (repo / "no_program").write_text("\x00\x01 not a program\n")
    (repo / "no_program").chmod(0o755)
    (repo / "folder").mkdir()
    return repo


@pytest.mark.parametrize("command, reason", [
    ("./plain.sh", "Permission denied"), ("./no_program", "Exec format error"),
    ("./folder", "Permission denied"), ("./nothing-here", "No such file or directory")])
def test_a_command_that_cannot_be_started_is_one_error_with_the_reason(
        _no_start_repo, command, reason):
    with pytest.raises(edit.StartError, match=f"^{reason}$"):
        edit.run_in_project([command], 20)


def test_a_command_that_is_there_but_cannot_be_started_is_reported_and_not_raised(
        _no_start_repo, monkeypatch, caplog):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_COMMAND", ["./plain.sh"])
    r = edit.run_test()
    assert r["status"] == "did not run" and r["output"] == (
        "./plain.sh cannot be started (Permission denied). Set TEST_COMMAND to "
        "something runnable")
    assert edit.patch_round_count() == 0
    monkeypatch.setattr(config, "CHECK_COMMAND", ["./no_program"])
    assert _RUN_CHECK() is None
    assert ("./no_program cannot be started (Exec format error), so the static check "
            "does not run") in caplog.text
    (_no_start_repo / "m.py").write_text("new\n")
    monkeypatch.setattr(config, "FORMAT_COMMAND", ["./plain.sh"])
    result = edit.run_format()
    assert result["exit_status"] is None and result["output"] == (
        "./plain.sh cannot be started (Permission denied). Set FORMAT_COMMAND to "
        "something runnable")


def test_a_command_that_finishes_comes_back_as_it_printed(repo):
    proc = edit.run_in_project([sys.executable, "-c", "import sys; print('out'); "
                                "print('err', file=sys.stderr); sys.exit(3)"], 30)
    assert (proc.returncode, proc.stdout, proc.stderr) == (3, "out\n", "err\n")


def _thread_and_fd_count() -> tuple[int, int]:
    """Return how many threads run and how many descriptors are open."""
    return threading.active_count(), len(os.listdir("/dev/fd"))


@pytest.mark.parametrize("script, want", [
    ("echo a1; echo a2; echo b > /dev/stdout; echo c", ("a1\na2\nb\nc\n", "")),
    ("echo a1; echo a2; echo b >> /dev/stdout; echo c", ("a1\na2\nb\nc\n", "")),
    ("echo a1; echo a2; echo b > /dev/fd/1; echo c", ("a1\na2\nb\nc\n", "")),
    pytest.param("echo a1; echo a2; echo b > /proc/self/fd/1; echo c",
                 ("a1\na2\nb\nc\n", ""),
                 marks=pytest.mark.skipif(not os.path.isdir("/proc/self/fd"),
                                          reason="the system has no /proc")),
    ("echo e1 >&2; echo e2 >&2; echo x | tee /dev/stderr; echo e3 >&2",
     ("x\n", "e1\ne2\nx\ne3\n")),
    ("echo e1 >&2; echo x | tee -a /dev/stderr; echo e3 >&2", ("x\n", "e1\nx\ne3\n")),
    ("echo a1; echo e1 >&2; echo b > /dev/stderr; echo e2 >&2", ("a1\n", "e1\nb\ne2\n")),
    ("printf 'a\\r\\nb\\rc\\n'", ("a\nb\nc\n", "")),
])
def test_what_a_command_wrote_stays_when_it_opens_its_output_again_by_name(
        repo, script, want):
    count = _thread_and_fd_count()
    proc = edit.run_in_project(["sh", "-c", script], 30)
    assert (proc.stdout, proc.stderr) == want
    assert _thread_and_fd_count() == count


def test_a_program_that_opens_its_output_again_by_name_loses_no_line(repo):
    proc = edit.run_in_project([sys.executable, "-c", (
        "import logging, sys\nprint('a1', flush=True)\n"
        "log = logging.getLogger('x'); log.addHandler(logging.FileHandler('/dev/stdout'))\n"
        "log.warning('LOG1'); print('a2', flush=True); log.warning('LOG2')\n"
        "open('/dev/stdout', 'w').write('b\\n')\nprint('a3')")], 30)
    assert proc.stdout == "a1\nLOG1\na2\nLOG2\nb\na3\n"


def test_a_red_run_is_read_as_red_when_its_command_ends_by_opening_its_output_by_name(
        repo, monkeypatch):
    """What the test tool printed is still there after the line a wrapper adds."""
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_COMMAND", ["sh", "-c", (
        "printf '=== short test summary info ===\\n"
        "FAILED tests/test_a.py::test_x - assert 1 == 2\\n1 failed in 0.01s\\n'; "
        "echo 'report written' > /dev/stdout; exit 1")])
    r = edit.run_test()
    assert (r["status"], r["failed"]) == ("failed", ["tests/test_a.py::test_x"])
    assert r["output"].endswith("1 failed in 0.01s\nreport written\n")
    assert not edit.stop_short_run(r) and edit.last_test_ok() is False


def test_the_output_of_a_command_is_returned_up_to_a_limit_at_each_end(repo, monkeypatch):
    monkeypatch.setattr(edit, "_OUTPUT_HEAD_BYTE", 10)
    monkeypatch.setattr(edit, "_OUTPUT_TAIL_BYTE", 5)
    proc = edit.run_in_project(
        ["sh", "-c", "printf 'aaaaaaaaaa%0100dcdefg' 0; printf 'short' >&2"], 30)
    assert proc.stdout == "a" * 10 + CUT_NOTE.format(omitted=100) + "\ncdefg"
    assert proc.stderr == "short"


def test_a_command_that_writes_far_more_than_is_returned_is_read_to_its_end(
        repo, monkeypatch):
    monkeypatch.setattr(edit, "_OUTPUT_HEAD_BYTE", 1000)
    monkeypatch.setattr(edit, "_OUTPUT_TAIL_BYTE", 1000)
    count = _thread_and_fd_count()
    proc = edit.run_in_project(["sh", "-c", "head -c 50000000 /dev/zero; echo last"], 60)
    assert proc.returncode == 0 and proc.stdout.endswith("\x00last\n")
    assert len(proc.stdout) < 2100 and CUT_NOTE.format(omitted=49998005) in proc.stdout
    assert _thread_and_fd_count() == count


# Prints 8 lines, one line to keep on each stream, and 8 lines more
_KEEP_SCRIPT = ("for n in 1 2 3 4 5 6 7 8; do echo \"plain line $n\"; done; "
                "echo 'keep é this'; echo 'keep that' >&2; "
                "for n in 1 2 3 4 5 6 7 8; do echo \"plain line $n\"; done")


def test_the_lines_a_caller_keeps_are_returned_from_between_the_two_ends(repo,
                                                                        monkeypatch):
    monkeypatch.setattr(edit, "_OUTPUT_HEAD_BYTE", 20)
    monkeypatch.setattr(edit, "_OUTPUT_TAIL_BYTE", 20)
    count = _thread_and_fd_count()
    proc = edit.run_in_project(["sh", "-c", _KEEP_SCRIPT], 30,
                               keep_line=lambda line: int(line.startswith("keep")))
    assert proc.stdout == ("plain line 1\n" + CUT_NOTE.format(omitted=182)
                           + "\nkeep é this\nplain line 8\n")
    assert proc.stderr == "keep that\n"
    assert _thread_and_fd_count() == count
    # No more of them than _OUTPUT_LINE_BYTE
    monkeypatch.setattr(edit, "_OUTPUT_LINE_BYTE", 10)
    proc = edit.run_in_project(["sh", "-c", _KEEP_SCRIPT], 30,
                               keep_line=lambda line: int(line.startswith("keep")))
    assert proc.stdout == ("plain line 1\n" + CUT_NOTE.format(omitted=195)
                           + "\nplain line 8\n")
    # Without keep_line the two ends are returned as they are
    proc = edit.run_in_project(["sh", "-c", _KEEP_SCRIPT], 30)
    assert proc.stdout == ("plain line 1\nplain l" + CUT_NOTE.format(omitted=181)
                           + "\nline 7\nplain line 8\n")


# Prints the text of a captured run between two runs of other lines
_REPLAY_BETWEEN = (
    "import json, pathlib, sys\n"
    "d = json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
    "fill = ''.join(f'    log line {n:04d} of a case, as the code under test wrote it\\n'"
    " for n in range(60))\n"
    "sys.stdout.write(fill + d['output'] + fill)\nsys.exit(d['returncode'])\n")


def _read_tuple(result: dict) -> tuple:
    """Return what one run_test() return value says of the tests."""
    return (result["status"], result["failed"], edit.no_test_run(result),
            edit.early_stop_run(result), edit.fail_stop_run(result), edit.last_test_ok())


def test_what_a_test_command_says_of_its_tests_is_read_between_the_two_ends(
        py_tests, fake_test_command, monkeypatch):
    """The lines after the one saying the run ended early are kept with it."""
    monkeypatch.setattr(edit, "_OUTPUT_HEAD_BYTE", 100)
    monkeypatch.setattr(edit, "_OUTPUT_TAIL_BYTE", 100)
    fill = "\n".join(f"log line {n} of a case" for n in range(40))
    fake_test_command(2, f"{fill}\nERROR tests/test_plot.py - ImportError\n{fill}\n"
                         "!!!!!!!! Interrupted: 1 error during collection !!!!!!!!\n"
                         f"after 1\nafter 2\nafter 3\nafter 4\n{fill}\n"
                         f"1 error in 0.07s\n{fill}")
    result = edit.run_test()
    assert _read_tuple(result) == ("failed", ["tests/test_plot.py"], False, True, False,
                                   False)
    assert result.early_stop_text == (
        "!!!!!!!! Interrupted: 1 error during collection !!!!!!!!\nafter 1\nafter 2\nafter 3")
    assert "log line 20 of a case" not in result["output"]


def test_a_process_outside_the_group_that_holds_the_output_is_read_for_a_moment_only(
        repo):
    """It left the command's group, so nothing kills it; its end is not waited for."""
    hold = ("import subprocess, sys; subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(3)'], process_group=0); print('1 passed')")
    count = _thread_and_fd_count()
    start = time.monotonic()
    proc = edit.run_in_project([sys.executable, "-c", hold], 30)
    assert time.monotonic() - start < 2.5
    assert (proc.returncode, proc.stdout) == (0, "1 passed\n")
    assert _thread_and_fd_count() == count


def test_a_command_that_did_not_end_in_time_or_did_not_start_leaves_nothing_open(repo):
    count = _thread_and_fd_count()
    with pytest.raises(subprocess.TimeoutExpired):
        edit.run_in_project(["sh", "-c", "echo before; sleep 30"], 1)
    with pytest.raises(edit.StartError):
        edit.run_in_project(["./nothing-here"], 30)
    assert _thread_and_fd_count() == count


def test_a_command_of_the_target_is_not_given_this_tool_s_settings(repo, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", "/somewhere/out")
    monkeypatch.setenv("SOME_TOOLCHAIN_CACHE", "/somewhere")
    proc = edit.run_in_project([sys.executable, "-c", "import os; print("
                                "os.environ.get('OUTPUT_DIR'), "
                                "os.environ.get('SOME_TOOLCHAIN_CACHE'))"], 30)
    assert proc.stdout == "None /somewhere\n"


# == last_test_ok =============================
def test_last_tests_ok_is_none_before_the_tests_run(repo):
    """None is "never asked", which history.judge() must not read as a pass."""
    assert edit.last_test_ok() is None


def test_last_tests_ok_remembers_a_pass(repo, fake_test_command, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(0, "all good")
    edit.run_test()
    assert edit.last_test_ok() is True


def test_last_tests_ok_remembers_a_failure(repo, fake_test_command, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(1, "1 failed")
    edit.run_test()
    assert edit.last_test_ok() is False


def test_last_tests_ok_is_reset_between_requests(repo, fake_test_command, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(1, "1 failed")
    edit.run_test()
    edit.reset_patch_round()
    assert edit.last_test_ok() is None


# == A run that never reached a verdict ========
# A non-zero exit code has more causes than a red test. Only a run that reported
# something is read as a failure; the rest are None, which judge() does not call failed
@pytest.fixture
def py_tests(repo, monkeypatch):
    """The repo with a tests directory, as the primary language's runner is given it."""
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    return repo


def test_no_test_was_collected_is_not_a_failure(py_tests, fake_test_command):
    """An empty tests directory: pytest exits 5 and reports nothing that ran."""
    fake_test_command(5, "no tests ran in 0.00s")
    r = edit.run_test()
    # The model still reads "this did not work"; only the judge's signal is held back
    assert r["status"] == "did not run"
    assert edit.last_test_ok() is None


def test_a_run_over_no_test_is_told_from_one_that_could_not_run(py_tests, fake_test_command,
                                                                monkeypatch):
    fake_test_command(5, "no tests ran in 0.00s")
    assert edit.no_test_run(edit.run_test()) is True
    fake_test_command(1, "ModuleNotFoundError: No module named '_pytest'")
    assert edit.no_test_run(edit.run_test()) is False
    fake_test_command(0, "2 passed in 0.10s")
    assert edit.no_test_run(edit.run_test()) is False
    # The lines of the JavaScript runners: jest with no test file, npm with no test
    # script, and the script npm init writes
    monkeypatch.setattr(config, "PRIMARY_EXT", "js")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "separator")
    for line in ("No tests found, exiting with code 1",
                 'npm ERR! Missing script: "test"', 'npm error Missing script: "test"',
                 "Error: no test specified"):
        fake_test_command(1, line)
        assert edit.no_test_run(edit.run_test()) is True, line
    fake_test_command(1, "Test Suites: 1 failed, 1 total")
    assert edit.no_test_run(edit.run_test()) is False
    # A language with no such line never says so
    monkeypatch.setattr(config, "PRIMARY_EXT", "sql")
    fake_test_command(1, "no tests ran in 0.00s")
    assert edit.no_test_run(edit.run_test()) is False


@pytest.mark.parametrize("exit_code, out, is_early_stop, fail_list", [
    # A test file that cannot be collected: pytest runs no test
    (2, "ERROR tests/test_plot.py\n!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during "
        "collection !!!!!!!!!!!!!!!!!!!!\n1 error in 0.07s", True, ["tests/test_plot.py"]),
    (2, "ERROR tests/test_plot.py\n! Interrupted: 2 errors during collection !\n"
        "2 errors in 0.07s", True, ["tests/test_plot.py"]),
    # -x and --maxfail
    (1, "FAILED tests/test_a.py::test_one - assert 1 == 2\n!!!!!!!!!! stopping after 1 "
        "failures !!!!!!!!!!\n1 failed in 0.01s", True, ["tests/test_a.py::test_one"]),
    (1, "!!!! xdist.dsession.Interrupted: stopping after 1 failures !!!!\n1 failed in 1s",
     True, []),
    # A run that went on to its end, whatever it names and prints
    (1, "ERROR tests/test_b.py::test_uses_fixture - RuntimeError: broken\n"
        "ERROR tests/test_c.py - ImportError: no module\n1 passed, 2 errors in 0.1s",
     False, ["tests/test_b.py::test_uses_fixture", "tests/test_c.py"]),
    (1, "FAILED tests/test_a.py::test_one - AssertionError: !!! stopping after 1 failures"
        "\n1 failed in 0.01s", False, ["tests/test_a.py::test_one"]),
    (0, "2 passed in 0.10s", False, []), (5, "no tests ran in 0.00s", False, [])])
def test_a_run_that_ended_before_every_test_was_run_is_told_by_its_own_line(
        py_tests, fake_test_command, monkeypatch, exit_code, out, is_early_stop, fail_list):
    fake_test_command(exit_code, _SUMMARY_HEAD + out)
    result = edit.run_test()
    assert edit.early_stop_run(result) is is_early_stop
    assert result["failed"] == fail_list
    assert [edit.early_stop_run(part) for part in edit.part_dict_of(result).values()] == [
        is_early_stop]
    # A dict that is not a run's own return value is read by the primary language's line
    assert edit.early_stop_run(dict(result)) is is_early_stop
    # A language with no such line never says so
    monkeypatch.setattr(config, "PRIMARY_EXT", "js")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "separator")
    assert edit.early_stop_run(edit.run_test()) is False
    assert edit.early_stop_run(dict(result)) is False


def test_there_is_a_test_command_when_the_tests_are_on_and_one_is_set(monkeypatch):
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", ["run"])
    assert edit.has_test_command() is True
    monkeypatch.setattr(config, "TEST_COMMAND", [])
    assert edit.has_test_command() is False
    monkeypatch.setattr(config, "TEST_COMMAND", ["run"])
    monkeypatch.setattr(config, "RUN_TEST", False)
    assert edit.has_test_command() is False


def test_the_tests_directory_is_made_only_where_the_tests_are_reached_by_path(repo,
                                                                            monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "spec")
    assert edit.has_test_path() is False
    edit.make_test_path()
    assert edit.has_test_path() is True
    edit.make_test_path()
    monkeypatch.setattr(config, "TEST_PATH", "spec2")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    edit.make_test_path()
    assert edit.has_test_path() is False
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "")
    edit.make_test_path()
    assert not (repo / "spec2").exists()


@pytest.mark.parametrize("primary, file_list, expect_list", [
    ("java", ["pom.xml"], ["src/test/java/CtloomFillTest.java"]),
    ("kt", ["settings.gradle.kts", "app/build.gradle.kts"],
     ["app/src/test/kotlin/CtloomFillTest.kt"]),
    ("java", ["pom.xml", "src/test/java/calc/CalcTest.java"], []),
    ("java", ["pom.xml", "src/test/java/data.json"], ["src/test/java/CtloomFillTest.java"]),
    ("java", ["pom.xml", "core/pom.xml"],
     ["core/src/test/java/CtloomFillTest.java", "src/test/java/CtloomFillTest.java"]),
    ("java", ["pom.xml", "core/pom.xml", "core/src/test/java/CoreTest.java"], []),
    ("kt", ["build.gradle"], ["src/test/kotlin/CtloomFillTest.kt"]),
    ("kt", ["build.gradle", "src/test/java/CalcTest.kt"], []),
    ("py", ["pyproject.toml"], []),
    ("java", [], [])])
def test_a_fill_test_is_written_where_a_module_has_no_test(repo, monkeypatch, primary,
                                                           file_list, expect_list):
    monkeypatch.setattr(config, "PRIMARY_EXT", primary)
    monkeypatch.setattr(edit, "other_test_list", lambda file_list=None: [])
    for name in file_list:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x\n", encoding="utf-8")
    assert sorted(edit.write_fill_test()) == expect_list
    assert sorted(str(path.relative_to(repo)) for path in repo.rglob("CtloomFillTest*")) \
        == expect_list
    assert edit.write_fill_test() == []


def test_a_fill_test_is_written_for_a_language_run_beside_the_primary_one(repo,
                                                                         monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "cbl")
    monkeypatch.setattr(edit, "other_test_list", lambda file_list=None: [
        ("java", ["mvn", "test"], "java"), ("py", ["pytest"], "")])
    (repo / "java").mkdir()
    (repo / "java" / "pom.xml").write_text("x\n", encoding="utf-8")
    edit.write_fill_test()
    assert sorted(str(path.relative_to(repo)) for path in repo.rglob("CtloomFillTest*")) \
        == ["java/src/test/java/CtloomFillTest.java"]


def test_the_dependency_file_names_are_the_primary_languages(monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    assert "pyproject.toml" in edit.dep_file_pattern_set()
    monkeypatch.setattr(config, "PRIMARY_EXT", "c")
    assert edit.dep_file_pattern_set() == frozenset({"CMakeLists.txt"})
    monkeypatch.setattr(config, "PRIMARY_EXT", "sql")
    assert edit.dep_file_pattern_set() == frozenset()
    monkeypatch.setattr(config, "PRIMARY_EXT", "zz")
    assert edit.dep_file_pattern_set() == frozenset()


def test_a_runner_that_could_not_start_is_not_a_failure(py_tests, fake_test_command):
    """A broken environment: the runner raises before it collects anything."""
    fake_test_command(1, "Traceback (most recent call last):\n"
                         "ModuleNotFoundError: No module named '_pytest'")
    assert edit.run_test()["status"] == "did not run"
    assert edit.last_test_ok() is None


def test_an_error_during_collection_is_a_failure(py_tests, fake_test_command):
    """An import the change broke is red: the run reported it as an error."""
    fake_test_command(2, "ERROR tests/test_x.py\n1 error in 0.09s")
    edit.run_test()
    assert edit.last_test_ok() is False


def test_a_reported_failure_stays_a_failure(py_tests, fake_test_command):
    fake_test_command(1, "FAILED tests/t.py::t_x - AssertionError\n1 failed in 0.01s")
    edit.run_test()
    assert edit.last_test_ok() is False


def test_an_unreadable_failure_is_not_counted_as_red(repo, fake_test_command,
                                                     monkeypatch):
    """mvn -q prints no count line. A repository with no tests must not read as red."""
    monkeypatch.setattr(config, "PRIMARY_EXT", "java")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(1, "")
    edit.run_test()
    assert edit.last_test_ok() is None


def test_a_named_failing_test_is_red_in_every_language(repo, fake_test_command,
                                                       monkeypatch):
    """_FAIL_RE_DICT covers all 11 extensions, so a named failure needs no report line."""
    monkeypatch.setattr(config, "PRIMARY_EXT", "java")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(1, "[ERROR]   com.example.FooTest.bar:12 expected true")
    r = edit.run_test()
    assert r["failed"] == ["com.example.FooTest.bar"]
    assert edit.last_test_ok() is False


def test_a_run_that_was_refused_drops_the_earlier_verdict(repo, fake_test_command,
                                                          monkeypatch):
    """The tests did not run this time, so the last run's pass no longer stands."""
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(0, "6 passed in 0.10s")
    edit.run_test()
    assert edit.last_test_ok() is True
    monkeypatch.setattr(config, "TEST_COMMAND", ["nonexistent-build-tool"])
    assert edit.run_test()["status"] == "did not run"
    assert edit.last_test_ok() is None


# == Real build tool output ====================
# Every file under tests/data/test_output holds one run of a real test command - its
# exit code and what it printed - captured in a container that had that toolchain. The
# five states each language was captured in are:
#
#   G  every test passes            R  one test fails and is named
#   B  the sources do not build     N  the repository has no tests
#   X  the project was never set up  X2  nothing can be downloaded
#
# A name is <extension>_<state>, with a third part for another form the same command
# prints in (cs_R_mtp: dotnet test under Microsoft.Testing.Platform; r_G_file: ctloom-test-r
# given two test files, one of which checks nothing; java_R_multi: three modules, the
# third depending on the first, which is where the failing test or broken source is;
# java_R_name: failing tests of a nested class, with a parameter and with a name that
# holds spaces; kt_B_kapt: two projects, the annotation processing of one failing;
# js_*_vitest: npm test running vitest 1.3.0; js_*_vitest3: the same with vitest 3.2.4;
# rs_R_abort: one test binary of three killed by a signal before its summary;
# rs_B_warning: a warning of the library, then an error in an integration test;
# rs_R_print: a failing test that printed a line starting with "warning: ";
# py_G_strict: a project whose pytest settings turn every warning into an error;
# js_*_mocha: npm test running mocha 11.7.5, js_L_mocha running eslint 9.39.1 first, which
# warns, and then mocha; js_*_bun: npm test running bun test 1.4.4, js_R_bun_bail with
# --bail and the failing test first, js_R_bun_load with a second test file importing a
# module that is not there, js_R_bun_many with 25 tests that pass, after which bun lists
# the failing test again).
#
# What each of them has to come back as is written out below rather than stored with
# the capture, so a capture cannot agree with itself. The only thing edited in a capture
# is the path the fixture was written to, which is /fixture in all of them
_GOLDEN_DIR = pathlib.Path(__file__).parent / "data" / "test_output"

_GOLDEN_STATUS = {
    "py_G": "passed", "py_R": "failed", "py_B": "failed",
    "py_N": "did not run", "py_X": "did not run", "py_G_strict": "passed",
    "java_G": "passed", "java_R": "failed", "java_B": "failed",
    "java_N": "did not run", "java_X": "did not run", "java_X2": "did not run",
    "kt_G": "passed", "kt_R": "failed", "kt_B": "failed",
    "kt_N": "did not run", "kt_X": "did not run", "kt_X2": "did not run",
    "js_G": "passed", "js_R": "failed", "js_B": "failed",
    "js_N": "did not run", "js_X": "did not run",
    "js_R_many": "failed", "js_R_suite": "failed", "js_R_verbose": "failed",
    "js_G_node": "passed", "js_R_node": "failed", "js_R_node_load": "failed",
    "js_N_node": "did not run",
    "js_G_node_spec": "passed", "js_R_node_spec": "failed", "js_R_node_load_spec": "failed",
    "js_N_node_spec": "did not run",
    "js_G_vitest": "passed", "js_R_vitest": "failed", "js_B_vitest": "failed",
    "js_N_vitest": "did not run",
    "js_G_vitest3": "passed", "js_R_vitest3": "failed", "js_B_vitest3": "failed",
    "js_N_vitest3": "did not run",
    "js_G_mocha": "passed", "js_R_mocha": "failed", "js_B_mocha": "did not run",
    "js_N_mocha": "did not run", "js_L_mocha": "passed",
    "js_G_bun": "passed", "js_R_bun": "failed", "js_B_bun": "failed",
    "js_N_bun": "did not run", "js_R_bun_bail": "failed", "js_R_bun_load": "failed",
    "js_R_bun_many": "failed",
    "kt_R_check": "failed",
    "java_G_multi": "passed", "java_R_multi": "failed", "java_B_multi": "failed",
    "java_N_multi": "did not run",
    "kt_G_multi": "passed", "kt_R_multi": "failed", "kt_B_multi": "failed",
    "kt_N_multi": "did not run",
    "java_R_name": "failed", "kt_R_name": "failed", "kt_B_kapt": "failed",
    "ts_G": "passed", "ts_R": "failed", "ts_B": "failed",
    "ts_N": "did not run", "ts_X": "did not run",
    "c_G": "passed", "c_R": "failed", "c_B": "failed",
    "c_N": "did not run", "c_X": "did not run",
    "cpp_G": "passed", "cpp_R": "failed", "cpp_B": "failed",
    "cpp_N": "did not run", "cpp_X": "did not run",
    "rs_G": "passed", "rs_R": "failed", "rs_B": "failed",
    "rs_N": "did not run", "rs_X": "did not run", "rs_X2": "did not run",
    "rs_R_abort": "failed", "rs_B_warning": "failed", "rs_R_print": "failed",
    "cs_G": "passed", "cs_R": "failed", "cs_B": "failed",
    "cs_N": "did not run", "cs_X": "did not run", "cs_X2": "did not run",
    "cs_G_mtp": "passed", "cs_R_mtp": "failed", "cs_B_mtp": "failed",
    "cs_N_mtp": "did not run",
    "r_G": "passed", "r_G_file": "passed", "r_R": "failed", "r_B": "failed",
    "r_N": "did not run", "r_N_empty": "did not run", "r_X": "did not run",
    "r_X2": "did not run",
    "r_G_script": "passed", "r_R_script": "failed", "r_B_script": "failed",
    "r_N_script": "did not run",
}

# The states in which nothing about the change can be read from the run. None of them
# may ever be red
_NOTHING_RAN = ("N", "X", "X2")

_VERDICT_OF = {word: verdict for verdict, word in edit.TEST_STATUS_DICT.items()}


@pytest.fixture
def golden_test_command(monkeypatch):
    """Return a helper replaying one captured run as the test command."""
    def use(name):
        path = _GOLDEN_DIR / f"{name}.json"
        monkeypatch.setattr(
            config, "TEST_COMMAND",
            [sys.executable, "-c",
             "import json, pathlib, sys;"
             "d = json.loads(pathlib.Path(sys.argv[1]).read_text());"
             "sys.stdout.write(d['output']); sys.exit(d['returncode'])",
             str(path)])
        return json.loads(path.read_text())
    return use


@pytest.fixture
def as_language(repo, monkeypatch):
    """Return a helper putting the repository under one language's test settings."""
    def use(ext):
        cfg = LANG_DICT[ext]
        monkeypatch.setattr(config, "PRIMARY_EXT", ext)
        monkeypatch.setattr(config, "TEST_TARGET_STYLE", cfg.test_target)
        monkeypatch.setattr(config, "TEST_PATH", cfg.test_path)
        (repo / cfg.test_path).mkdir(parents=True, exist_ok=True)
    return use


@pytest.mark.parametrize("name", sorted(_GOLDEN_STATUS))
def test_run_tests_reads_a_real_build_tool_run(name, as_language, golden_test_command):
    """The three words the model reads, against what the tool really printed."""
    as_language(name.split("_")[0])
    golden_test_command(name)
    status = _GOLDEN_STATUS[name]
    assert edit.run_test()["status"] == status
    assert edit.last_test_ok() is _VERDICT_OF[status]


def test_mocha_names_its_failing_tests_once_and_eslint_s_summary_is_no_test(
        as_language, golden_test_command):
    """mocha lists a failing test as it runs and again under its count of failures."""
    as_language("js")
    golden_test_command("js_R_mocha")
    result = edit.run_test()
    assert result["status"] == "failed" and result["failed"] == ["bar"]
    golden_test_command("js_L_mocha")
    result = edit.run_test()
    assert result["status"] == "passed" and result["failed"] == []
    # Its counts are how many tests ran
    for name in ("js_G_mocha", "js_R_mocha", "js_L_mocha"):
        output = json.loads((_GOLDEN_DIR / f"{name}.json").read_text())["output"]
        assert edit._ran_count(output, "js") == 2


@pytest.mark.parametrize("name, failed, ran, is_fail_stop", [
    ("js_G_bun", [], 1, False),
    ("js_R_bun", ["calc > bar"], 2, False),
    ("js_B_bun", ["test/calc.test.js"], 1, False),
    ("js_R_bun_load", ["test/gone.test.js"], 2, False),
    ("js_R_bun_bail", ["calc > bar"], None, True),
    ("js_R_bun_many", ["calc > bar"], 26, False)])
def test_bun_names_its_failing_tests_and_the_files_that_did_not_load(
        as_language, golden_test_command, name, failed, ran, is_fail_stop):
    """bun test's "(fail)" lines name the tests, and the heading above "# Unhandled error
    between tests" names a file; the heading of a file whose tests passed names nothing."""
    as_language("js")
    output = golden_test_command(name)["output"]
    result = edit.run_test()
    assert result["failed"] == failed
    assert edit._ran_count(output, "js") == ran
    assert edit.fail_stop_run(result) is is_fail_stop


@pytest.mark.parametrize("name, part", [
    ("js_R_bun", "Received: 4"), ("js_R_bun_many", "Received: 4"),
    ("js_R_bun_load", "Cannot find module '../src/gone.js'")])
def test_bun_s_failure_text_is_the_report_above_its_fail_line(as_language,
                                                              golden_test_command, name, part):
    """bun prints a failing test's report above its "(fail)" line; the failure text holds
    it, and not the lines of the tests that passed."""
    as_language("js")
    golden_test_command(name)
    excerpt = edit.run_test()["failed_excerpt"]
    assert part in excerpt and "(pass)" not in excerpt


# bun test run with CLAUDECODE=1 in its environment, which prints only the failing tests
# and the headings of their files
_BUN_QUIET_OUTPUT = """bun test v1.4.4 (38afee6c8)

test/c.test.js:

# Unhandled error between tests
-------------------------------
error: Cannot find module './missing.js' from '/fixture/test/c.test.js'
-------------------------------


test/a.test.ts:
error: expect(received).toBe(expected)

Expected: 3
Received: 1

(fail) calc > subs fails [0.13ms]
error: boom
(fail) calc > inner > deep fails [0.06ms]

 3 pass
 1 skip
 1 todo
 3 fail
 1 error
 3 expect() calls
Ran 8 tests across 3 files. [10.00ms]
"""


def test_bun_s_quiet_output_is_read_alike():
    """The quiet form names the same failures and counts the passed and failed tests."""
    fail_list = [name for _, name in edit._fail_pair_list(_BUN_QUIET_OUTPUT, "js")]
    assert fail_list == ["test/c.test.js", "calc > subs fails", "calc > inner > deep fails"]
    assert edit._ran_count(_BUN_QUIET_OUTPUT, "js") == 6
    assert edit._test_verdict(1, _BUN_QUIET_OUTPUT, "js", fail_list) is False


def test_a_bun_run_longer_than_the_output_kept_still_has_its_failure_text(
        as_language, fake_test_command, monkeypatch):
    """bun's "(pass)" lines are left out of the output kept, as pytest's PASSED lines
    are, so the report of a failing test before many passing ones still reaches the
    failure text."""
    as_language("js")
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", 4000)
    record = json.loads((_GOLDEN_DIR / "js_R_bun_many.json").read_text())
    pass_text = "".join(f"(pass) adds {n} to a number of the long table [0.02ms]\n"
                        for n in range(200))
    fake_test_command(1, record["output"].replace("(pass) adds 24 to 0\n",
                                                  "(pass) adds 24 to 0\n" + pass_text, 1))
    result = edit.run_test()
    assert result["failed"] == ["calc > bar"]
    assert "Received: 4" in result["failed_excerpt"]
    assert "(pass)" not in result["output"]


@pytest.mark.parametrize("kind", ["skip", "todo", "stdout"])
def test_a_bun_run_whose_other_lines_outgrow_the_output_kept_has_its_failure_text(
        as_language, fake_test_command, monkeypatch, kind):
    """Many "(skip)" or "(todo)" lines, or what the tests printed to standard output
    ahead of bun's own lines, do not push the report of a failing test out of the
    failure text."""
    as_language("js")
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", 4000)
    output = json.loads((_GOLDEN_DIR / "js_R_bun_many.json").read_text())["output"]
    if kind == "stdout":
        output = "".join(f"log line {n} that a test wrote\n" for n in range(400)) + output
    else:
        output = output.replace("(pass) adds 24 to 0\n", "(pass) adds 24 to 0\n" + "".join(
            f"({kind}) a test of the long table number {n}\n" for n in range(200)), 1)
    fake_test_command(1, output)
    result = edit.run_test()
    assert result["failed"] == ["calc > bar"]
    assert "Received: 4" in result["failed_excerpt"]
    assert len(result["failed_excerpt"]) <= 4000 + 100


def test_what_pytest_printed_outside_its_summary_is_in_the_failure_text(as_language,
                                                                        fake_test_command):
    """A line after pytest's summary (a path it did not find, a check run after it) is
    part of the failure text."""
    as_language("py")
    fake_test_command(4, "============ no tests ran in 0.01s ============\n"
                         "ERROR: file or directory not found: tests/test_gone.py\n")
    assert "file or directory not found" in edit.run_test()["failed_excerpt"]


def test_bun_s_snapshot_line_in_its_summary_leaves_the_counts_read():
    """bun prints "snapshots: 1 failed" between its counts; a run ending with 0 is still
    red and its tests are still counted."""
    output = json.loads((_GOLDEN_DIR / "js_R_bun.json").read_text())["output"].replace(
        " 1 fail\n", " 1 fail\nsnapshots: 1 failed\n", 1)
    fail_list = [name for _, name in edit._fail_pair_list(output, "js")]
    assert edit._test_verdict(0, output, "js", fail_list) is False
    assert edit._ran_count(output, "js") == 2


@pytest.mark.parametrize("message", ["2 errors", "1 error", "1 fail", "0 pass"])
def test_a_jest_console_line_shaped_like_a_bun_count_is_no_count(message):
    """jest indents what a test logs; a line of it is no bun count in an output without
    bun test's first line."""
    output = json.loads((_GOLDEN_DIR / "js_G.json").read_text())["output"].replace(
        "PASS test/foo.test.js\n",
        f"PASS test/foo.test.js\n  console.log\n    {message}\n\n"
        "      at Object.log (test/foo.test.js:3:13)\n\n", 1)
    assert edit._test_verdict(0, output, "js", []) is True
    assert edit._ran_count(output, "js") == 2


@pytest.mark.parametrize("returncode, output", [
    # No test file
    (1, "bun test v1.4.4 (38afee6c8)\nerror: 0 test files matching "
        "**{.test,.spec,_test_,_spec_}.{js,ts,jsx,tsx} in --cwd=\"/fixture\"\n"),
    # Paths it was given that are no test files
    (1, "bun test v1.4.4 (38afee6c8)\nThe following filters did not match any test files "
        "in --cwd=\"/fixture\":\n test/zzz\n7 files were searched [2.00ms]\n"),
    # A -t that matches no test
    (1, "bun test v1.4.4 (38afee6c8)\n\nerror: regex \"zzzz\" matched 0 tests. Searched 1 "
        "file (skipping 12 tests) [4.00ms]\n"),
    # Files whose tests are all skipped
    (0, "bun test v1.4.4 (38afee6c8)\n\n 0 pass\n 1 skip\n 0 fail\n"
        "Ran 1 test across 1 file. [2.00ms]\n")])
def test_a_bun_run_over_no_test_is_no_verdict(returncode, output):
    """Each form bun prints for a run over no test is read as one."""
    assert edit._is_no_test_text(output, "js")
    assert edit._test_verdict(returncode, output, "js", []) is None


@pytest.mark.parametrize("head", ["js_R_mocha", "js_G_mocha"])
def test_a_merge_whose_mocha_files_no_longer_load_is_held_against_head(_js_replay, head):
    """A run that loads no test file ran no test: HEAD's tests ran, so it is a reason."""
    before = _js_replay(head)
    after = _js_replay("js_B_mocha")
    assert after["status"] == "did not run" and after["failed"] == []
    assert "did not run" in integrate._test_reason(
        after, integrate._baseline_set(after, lambda: before))


def test_a_rust_test_binary_killed_before_its_summary_is_named_by_its_target(
        as_language, golden_test_command):
    """The tests of the other targets ran: the run is red and names the target."""
    as_language("rs")
    golden_test_command("rs_R_abort")
    result = edit.run_test()
    assert result["status"] == "failed" and result["failed"] == ["--test abort"]
    # It is no stop: a run before a change that holds the same killed target is red by
    # the same name, so the other targets' tests are what a change is checked by
    assert not edit.early_stop_run(result)
    golden_test_command("rs_R")
    result = edit.run_test()
    assert result["failed"] == ["tests::subtracts", "--lib"]
    assert not edit.early_stop_run(result)


@pytest.fixture
def _js_replay(as_language, monkeypatch, tmp_path):
    """Return a helper running a fixed test command that replays a captured run."""
    as_language("js")
    out_path = tmp_path / "npm.json"
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "import json, pathlib, sys; "
        "d = json.loads(pathlib.Path(sys.argv[1]).read_text()); "
        "sys.stdout.write(d['output']); sys.exit(d['returncode'])", str(out_path)])

    def run(name):
        out_path.write_text((_GOLDEN_DIR / f"{name}.json").read_text())
        return edit.run_test()
    return run


@pytest.fixture
def _rs_replay(as_language, monkeypatch, tmp_path):
    """Return a helper running a fixed test command that prints a given cargo output."""
    as_language("rs")
    out_path = tmp_path / "cargo.out"
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "import pathlib, sys; "
        "sys.stdout.write(pathlib.Path(sys.argv[1]).read_text()); sys.exit(101)",
        str(out_path)])

    def run(text):
        out_path.write_text(text)
        return edit.run_test()
    return run


def test_a_merge_over_a_base_with_a_killed_rust_binary_is_checked_by_the_others(
        _rs_replay):
    """However many of its tests the killed binary printed, the runs compare by count."""
    out = json.loads((_GOLDEN_DIR / "rs_R_abort.json").read_text())["output"]
    quiet_out = out.replace("running 2 tests\ntest first ... ok\n", "running 2 tests\n")
    assert quiet_out != out
    for before_out in (out, quiet_out):
        before = _rs_replay(before_out)
        after = _rs_replay(out)
        assert integrate._test_reason(after, integrate._baseline_set(
            after, lambda: before)) == ""
    broken = _rs_replay(out.replace(
        "test adds_more ... ok\n\ntest result: ok. 1 passed; 0 failed;",
        "test adds_more ... FAILED\n\ntest result: FAILED. 0 passed; 1 failed;")
        + "error: test failed, to rerun pass `--test basic`\n")
    assert "adds_more" in integrate._test_reason(
        broken, integrate._baseline_set(broken, lambda: before))


def test_what_a_killed_rust_test_binary_printed_is_in_the_excerpt(
        as_language, fake_test_command):
    """The binary's own lines on standard error come before cargo's line for it."""
    as_language("rs")
    fake_test_command(101, "running 1 test\ntest deep ... \n"
                           "   Compiling calc v0.1.0 (/x)\n"
                           "     Running tests/so.rs (target/debug/deps/so-1a2b)\n\n"
                           "thread 'deep' has overflowed its stack\n"
                           "fatal runtime error: stack overflow, aborting\n"
                           "error: test failed, to rerun pass `--test so`\n\nCaused by:\n"
                           "  process didn't exit successfully: `/x/target/debug/deps/"
                           "so-1a2b` (signal: 6, SIGABRT: process abort signal)\n")
    result = edit.run_test()
    assert result["failed"] == ["--test so"]
    assert "thread 'deep' has overflowed its stack\nfatal runtime error" \
        in result["failed_excerpt"]
    # What a target printed on standard output with no test named (libtest's terse
    # lines, a harness of its own) comes before cargo's line for it
    fake_test_command(101, "running 1 test\ncase sum.txt: expected 5, got 4\n"
                           "     Running tests/golden.rs (target/debug/deps/golden-1a)\n"
                           "error: test failed, to rerun pass `--test golden`\n\n"
                           "Caused by:\n"
                           "  process didn't exit successfully: `/x/target/debug/deps/"
                           "golden-1a` (exit status: 1)\n")
    result = edit.run_test()
    assert result["failed"] == ["--test golden"] and not edit.early_stop_run(result)
    assert "case sum.txt: expected 5, got 4" in result["failed_excerpt"]
    # libtest's own exit status for failing tests is no stop; a doctest target is named
    fake_test_command(101, "test result: FAILED. 0 passed; 1 failed;\n"
                           "error: test failed, to rerun pass `--lib`\n\nCaused by:\n"
                           "  process didn't exit successfully: `/x/target/debug/deps/"
                           "calc-1a` (exit status: 101)\n"
                           "error: doctest failed, to rerun pass `--doc`\n")
    result = edit.run_test()
    assert result["failed"] == ["--lib", "--doc"] and not edit.early_stop_run(result)


def test_the_rust_excerpt_keeps_the_panic_over_cargo_s_own_lines(
        as_language, golden_test_command, fake_test_command):
    """cargo's lines of what it does, the targets that passed and the warnings are not
    the failure."""
    as_language("rs")
    golden_test_command("rs_R")
    excerpt = edit.run_test()["failed_excerpt"]
    assert "  left: 2" in excerpt and "Compiling" not in excerpt
    assert "add_then_sub" not in excerpt
    warning = ("warning: unused variable: `x`\n --> src/lib.rs:3:9\n  |\n3 |     let x = 1;"
               "\n  |         ^\n\nwarning: unreachable statement\n --> src/lib.rs:9:5\n"
               "  |\n8 |     return 1;\n  |     -------- any code following this\n"
               "9 | /     let y = {\n...  |\n21 | |     };\n   | |______^ unreachable "
               "statement\n   |\n   = note: on by default\n\n")
    fake_test_command(101, "running 1 test\ntest t::subtracts ... FAILED\n\nfailures:\n\n"
                           "---- t::subtracts stdout ----\n"
                           "thread 't::subtracts' panicked at"
                           " src/lib.rs:9:5:\nassertion `left == right` failed\n  left: 2\n"
                           " right: 3\n\ntest result: FAILED. 0 passed; 1 failed; 0 ignored"
                           "\n   Compiling calc v0.1.0 (/x)\n" + warning * 40
                           + "     Running unittests src/lib.rs "
                           "(target/debug/deps/calc-1a)\n"
                           "error: test failed, to rerun pass `--lib`\n")
    excerpt = edit.run_test()["failed_excerpt"]
    assert "  left: 2" in excerpt and "unused variable" not in excerpt
    assert "unreachable" not in excerpt
    # The targets after the failing one: rustdoc's line for each crate's doctests and
    # the progress of libtest's terse form
    fake_test_command(101, "running 1 test\ntest t::subtracts ... FAILED\n\nfailures:\n\n"
                           "---- t::subtracts stdout ----\n"
                           "thread 't::subtracts' panicked at"
                           " src/lib.rs:9:5:\nassertion `left == right` failed\n  left: 2\n"
                           " right: 3\n\ntest result: FAILED. 0 passed; 1 failed; "
                           "0 ignored\n"
                           + "\n..........\n. 1/2\n\nall doctests ran in 0.14s; "
                           "merged doctests compilation took 0.13s\n" * 12
                           + "error: test failed, to rerun pass `--lib`\n")
    excerpt = edit.run_test()["failed_excerpt"]
    assert "  left: 2" in excerpt and "all doctests ran" not in excerpt
    assert "\n..........\n" not in excerpt
    # A warning's summary line followed by an error, and a test's own "warning: " line
    golden_test_command("rs_B_warning")
    excerpt = edit.run_test()["failed_excerpt"]
    assert excerpt.startswith("error[E0425]: cannot find value `two` in this scope")
    assert "could not compile `f` (test \"basic\")" in excerpt and "warning" not in excerpt
    golden_test_command("rs_R_print")
    excerpt = edit.run_test()["failed_excerpt"]
    assert "warning: truncating input of 3 items\nkept [1, 2]" in excerpt


@pytest.mark.parametrize("name", sorted(
    name for name in _GOLDEN_STATUS if name.split("_")[1] == "R"))
def test_a_real_run_that_counts_a_failing_test_is_red_where_its_exit_code_is_0(
        name, as_language, monkeypatch):
    """A tool set to end with 0 whatever its tests did still prints the count."""
    as_language(name.split("_")[0])
    monkeypatch.setattr(
        config, "TEST_COMMAND",
        [sys.executable, "-c", "import json, pathlib, sys;"
         "sys.stdout.write(json.loads(pathlib.Path(sys.argv[1]).read_text())['output'])",
         str(_GOLDEN_DIR / f"{name}.json")])
    result = edit.run_test()
    assert result["status"] == "failed" and result["failed"]


# mocha ends a run whose test files did not load with 1, and prints no line of a failing
# build: it ran no test
_NO_BUILD_LINE_TUPLE = ("js_B_mocha",)


@pytest.mark.parametrize("name", sorted(
    name for name in _GOLDEN_STATUS
    if (name.split("_")[1] == "B" or name.startswith("java_X"))
    and name not in _NO_BUILD_LINE_TUPLE))
def test_a_real_run_whose_build_failed_is_read_alike_where_its_exit_code_is_0(
        name, as_language, golden_test_command, monkeypatch):
    """A tool set to end with 0 whatever its tests did (maven's --fail-never) still
    prints the line of a failing build or test file, and the run is read as with the
    exit code the tool would have ended with."""
    as_language(name.split("_")[0])
    golden_test_command(name)
    plain = _read_tuple(edit.run_test())
    monkeypatch.setattr(
        config, "TEST_COMMAND",
        [sys.executable, "-c", "import json, pathlib, sys;"
         "sys.stdout.write(json.loads(pathlib.Path(sys.argv[1]).read_text())['output'])",
         str(_GOLDEN_DIR / f"{name}.json")])
    assert _read_tuple(edit.run_test()) == plain


def test_a_run_that_ends_with_0_and_says_a_test_file_or_the_build_failed_is_red(
        as_language, fake_test_command):
    """jest under testFailureExitCode 0 with a test file that cannot be loaded, and
    maven under --fail-never with a build that failed, are red."""
    as_language("js")
    fake_test_command(0, "FAIL test/b.test.js\n  ● Test suite failed to run\n\n"
                         "PASS test/a.test.js\n\nTest Suites: 1 failed, 1 passed, 2 total\n"
                         "Tests:       2 passed, 2 total\n")
    result = edit.run_test()
    assert result["status"] == "failed" and "test/b.test.js" in result["failed"]
    fake_test_command(0, "PASS test/a.test.js\n\nTest Suites: 1 passed, 1 total\n"
                         "Tests:       2 passed, 2 total\n")
    assert edit.run_test()["status"] == "passed"
    as_language("java")
    fake_test_command(0, "[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0\n"
                         "[INFO] BUILD FAILURE\n[INFO] Build failures were ignored.\n")
    assert edit.run_test()["status"] == "failed"
    fake_test_command(0, "[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0\n"
                         "[INFO] BUILD SUCCESS\n")
    assert edit.run_test()["status"] == "passed"


@pytest.mark.parametrize("name", sorted(_GOLDEN_STATUS))
def test_a_real_run_is_read_the_same_between_the_first_and_the_last_bytes_returned(
        name, as_language, golden_test_command, monkeypatch):
    """A run whose own lines all lie past the first bytes run_in_project() returns and
    before the last ones is read as the run alone is."""
    as_language(name.split("_")[0])
    record = golden_test_command(name)
    plain = edit.run_test()
    monkeypatch.setattr(edit, "_OUTPUT_HEAD_BYTE", 1000)
    monkeypatch.setattr(edit, "_OUTPUT_TAIL_BYTE", 1000)
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", _REPLAY_BETWEEN, str(_GOLDEN_DIR / f"{name}.json")])
    result = edit.run_test()
    assert "log line 0059" in result["output"] and "log line 0020" not in result["output"]
    assert _read_tuple(result) == _read_tuple(plain)
    assert result["status"] == _GOLDEN_STATUS[name]
    # A run that printed nothing stopped short; with other lines around it, it did not
    if record["output"].strip():
        assert edit.stop_short_run(result) is edit.stop_short_run(plain)


@pytest.mark.parametrize("name", sorted(n for n in _GOLDEN_STATUS
                                        if n.split("_")[1] in _NOTHING_RAN))
def test_a_run_that_never_had_a_test_is_not_red(name, as_language,
                                                golden_test_command):
    """No repository without tests, and no toolchain that could not start, is red."""
    as_language(name.split("_")[0])
    golden_test_command(name)
    assert edit.run_test()["status"] != "failed"
    assert edit.last_test_ok() is not False


@pytest.mark.parametrize("name", sorted(_GOLDEN_STATUS))
def test_a_real_run_says_whether_its_command_stopped_short_of_the_tests(
        name, as_language, golden_test_command):
    """A project never set up and a download that failed stopped short; a run of the
    tests did not, whatever came of it, and neither did one that found no test. The
    line the tool printed says so, whatever program printed it."""
    ext = name.split("_")[0]
    as_language(ext)
    record = golden_test_command(name)
    result = edit.run_test()
    assert result["status"] == _GOLDEN_STATUS[name]
    is_stop_short = name.split("_")[1] in ("X", "X2")
    assert edit.stop_short_run(result) is is_stop_short
    assert edit.stop_short_run(dict(result)) is True
    # The line alone tells the two apart, in every run that was captured
    assert bool(edit._STOP_SHORT_RE_DICT[ext].search(record["output"])) is is_stop_short
    # Python's one captured run that stopped short is a download that failed
    is_fetch_fail = name.split("_")[1] == "X2" or name == "py_X"
    assert edit.fetch_fail_run(result) is is_fetch_fail
    assert edit.fetch_fail_run(dict(result)) is False


# A failing run of another test tool than the language's own, as the program the
# language's default test command starts hands it on:
# (the language, the exit code, what was printed)
_OTHER_TOOL_RED_LIST = [
    # ava under npm test
    ("js", 1, "\n  ✔ adds\n  ✘ [fail]: subtracts\n  ─\n\n  subtracts\n\n  Difference "
              "(- actual, + expected):\n\n  - 2\n  + 3\n\n  ─\n\n  1 test failed\n"),
    # nextest under cargo
    ("rs", 100, "    Starting 2 tests across 1 binary\n        PASS [   0.004s] calc "
                "tests::adds\n        FAIL [   0.004s] calc tests::subtracts\n"
                "     Summary [   0.005s] 2 tests run: 1 passed, 1 failed, 0 skipped\n"
                "error: test run failed\n"),
    # a script of the repository's own
    ("java", 1, "FAIL: check_add expected 3 got 4\n11 ok, 1 not ok\n"),
]


@pytest.mark.parametrize("ext, exit_code, output", _OTHER_TOOL_RED_LIST)
def test_a_failing_run_of_a_tool_the_language_s_lines_do_not_read_did_not_stop_short(
        ext, exit_code, output, as_language, fake_test_command):
    """It reached no verdict, and nothing it printed says it ended before the tests."""
    as_language(ext)
    fake_test_command(exit_code, output)
    result = edit.run_test()
    assert result["status"] == "did not run"
    assert edit.stop_short_run(result) is False


@pytest.mark.parametrize("ext, exit_code, output", [
    # A wrapper of the language's tool prints the tool's own line
    ("java", 1, "[ERROR] Failed to execute goal on project calc: Could not resolve "
                "dependencies for project com.example:calc:jar:1.0: The following "
                "artifacts could not be resolved: org.x:y:jar:1.0 (absent)"),
    ("py", 2, "error: Failed to fetch: `https://pypi.org/simple/pytest/`"),
    ("cs", 1, "/fixture/Calc.csproj : error NU1101: Unable to find package Humanizer."),
    # Nothing at all from the program npm test --silent is
    ("js", 254, ""), ("ts", 254, "")])
def test_a_run_that_printed_its_language_s_line_for_a_stop_before_the_tests_stopped_short(
        ext, exit_code, output, as_language, monkeypatch):
    as_language(ext)
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", f"import sys; sys.stdout.write({output!r}); sys.exit({exit_code})"])
    result = edit.run_test()
    assert result["status"] == "did not run"
    assert edit.stop_short_run(result) is True


@pytest.mark.parametrize("package_json, stop_short", [
    # npm test --silent prints nothing where package.json names no test script
    ('{"name": "x", "scripts": {"build": "next build"}}', False),
    ('{"name": "x"}', False),
    # with a byte order mark, which npm reads past
    ('\ufeff{"name": "x"}', False),
    # and where there is no package.json, or one it cannot read, or the script itself
    # printed nothing
    (None, True), ("{", True), ('{"scripts": {"test": "node t.js"}}', True)])
def test_npm_test_with_no_test_script_finds_no_test_and_does_not_stop_short(
        package_json, stop_short, as_language, monkeypatch, tmp_path):
    as_language("ts")
    bin_dir = tmp_path / "npm-bin"
    bin_dir.mkdir()
    (bin_dir / "npm").write_text("#!/bin/sh\nexit 1\n")
    (bin_dir / "npm").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(config, "TEST_COMMAND", ["npm", "test", "--silent"])
    package_path = pathlib.Path(config.PROJECT_DIR) / "package.json"
    if package_json is not None:
        package_path.write_text(package_json)
    result = edit.run_test()
    assert result["status"] == "did not run"
    assert edit.stop_short_run(result) is stop_short
    assert result.is_no_test is not stop_short
    # The reason reaches the excerpt the stages are shown
    assert ('package.json has no "test" script' in result["failed_excerpt"]) \
        is not stop_short
    # A command set by hand is read by what it prints alone
    if package_json is not None and not stop_short:
        monkeypatch.setattr(config, "TEST_COMMAND", [str(bin_dir / "npm"), "run", "check"])
        assert edit.stop_short_run(edit.run_test()) is True


def _js_repo(repo, as_language, monkeypatch, ext="js", file_dict=None):
    """Put the repository under a language's default test command with some files."""
    as_language(ext)
    monkeypatch.setattr(config, "TEST_COMMAND", list(LANG_DICT[ext].test_command_tuple))
    for path, text in (file_dict or {}).items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)


def test_js_with_no_package_json_runs_node_s_own_test_runner_over_its_test_files(
        repo, as_language, monkeypatch):
    _js_repo(repo, as_language, monkeypatch, file_dict={
        "calc.js": "", "test/calc.test.js": "", "tests/b.js": "", "src/a.spec.js": "",
        "test/data.json": "", "test.js": "", "lib/x_test.mjs": "", "gone.test.js": "",
        "node_modules/x/package.json": "{}", "node_modules/x/test/y.test.js": ""})
    _git(repo, "add", "gone.test.js")
    _git(repo, "commit", "-q", "-m", "gone")
    (repo / "gone.test.js").unlink()
    assert edit.primary_test_command() == ["node", "--test"]
    argv, target_list, _ = edit._build_argv(None, "separator", "test", "js")
    assert argv[:2] == ["node", "--test"] and target_list == []
    # The tool's tests and those node finds by itself, of files that are there
    assert sorted(argv[2:]) == ["lib/x_test.mjs", "src/a.spec.js", "test.js",
                                "test/calc.test.js"]
    # Paths go after it as they are, with no "--"
    argv, target_list, _ = edit._build_argv(["test/calc.test.js"], "separator", "test", "js")
    assert argv == ["node", "--test", "test/calc.test.js"]
    assert edit.test_command_text() == "node --test"


@pytest.mark.parametrize("ext, file_dict", [
    # A package.json anywhere outside node_modules keeps npm's test
    ("js", {"package.json": "{}"}), ("js", {"web/package.json": "{}"}),
    ("js", {"yarn.lock": ""}),
    # TypeScript has no bare test command
    ("ts", {})])
def test_js_keeps_npm_test_where_the_repository_has_a_package_file_and_ts_always(
        ext, file_dict, repo, as_language, monkeypatch):
    _js_repo(repo, as_language, monkeypatch, ext, file_dict)
    assert edit.primary_test_command() == ["npm", "test", "--silent"]
    argv, _, _ = edit._build_argv(["test/a.test.js"], "separator", "test", ext)
    assert argv == ["npm", "test", "--silent", "--", "test/a.test.js"]


def test_a_test_command_set_by_hand_is_run_as_it_is_with_no_package_json(
        repo, as_language, monkeypatch):
    _js_repo(repo, as_language, monkeypatch)
    monkeypatch.setattr(config, "TEST_COMMAND", ["npx", "jest"])
    assert edit.primary_test_command() == ["npx", "jest"]


@pytest.mark.parametrize("root, module, want_old", [
    ("<properties><maven.compiler.release>7</maven.compiler.release></properties>", "",
     True),
    ("", "<configuration><source>1.8</source><target>1.8</target></configuration>", True),
    ("<release>11</release>", "", False),
    # A module that needs a newer JDK than the old one keeps the default
    ("<release>7</release>", "<maven.compiler.release>21</maven.compiler.release>", False),
    ("", "", False),
    # A version set as a property, through another, and Spring Boot's java.version
    ("<properties><java.version>1.7</java.version></properties>",
     "<properties><maven.compiler.source>${java.version}</maven.compiler.source>"
     "</properties>", True),
    ("<properties><base>8</base><jdk>${base}</jdk></properties>",
     "<configuration><source>${jdk}</source></configuration>", True),
    ("<properties><java.version>1.8</java.version></properties>", "", True),
    # A property a module sets again: every value counts
    ("<properties><java.release>21</java.release><maven.compiler.release>${java.release}"
     "</maven.compiler.release></properties>",
     "<properties><java.release>8</java.release></properties>", False),
    # A property no pom.xml of the build sets (a parent's from elsewhere) is not known
    ("<properties><maven.compiler.release>${parent.java}</maven.compiler.release>"
     "</properties>", "<release>8</release>", False),
    # Tests compiled for a newer Java than the old JDK takes keep the default
    ("<maven.compiler.release>8</maven.compiler.release>"
     "<maven.compiler.testRelease>21</maven.compiler.testRelease>", "", False),
    # A comment says nothing
    ("<release>11</release><!-- <release>7</release> -->", "", False),
    # A profile the old JDK does not activate says nothing; one it does, one activated
    # some other way and one whose <jdk> cannot be read count
    ("<release>7</release>",
     "<profiles><profile><id>JDK17</id><activation><jdk>[17,)</jdk></activation>"
     "<properties><maven.compiler.testRelease>17</maven.compiler.testRelease></properties>"
     "</profile></profiles>", True),
    ("<release>7</release>",
     "<profiles><profile><activation><jdk>[11,)</jdk></activation><properties>"
     "<maven.compiler.testRelease>17</maven.compiler.testRelease></properties></profile>"
     "</profiles>", False),
    ("<release>7</release>",
     "<profiles><profile><activation><property><name>x</name></property></activation>"
     "<properties><maven.compiler.testRelease>17</maven.compiler.testRelease></properties>"
     "</profile></profiles>", False),
    ("<release>7</release>",
     "<profiles><profile><activation><jdk>17+</jdk></activation><properties>"
     "<maven.compiler.testRelease>17</maven.compiler.testRelease></properties></profile>"
     "</profiles>", False),
    # Another plugin's <source> and <target> are not versions, and a version may be built
    # from a property and text
    ("<maven.compiler.source>1.7</maven.compiler.source>"
     "<sources><source>src/gen</source></sources><target>2.2</target>", "", True),
    ("<properties><javaVersion>8</javaVersion>"
     "<maven.compiler.source>1.${javaVersion}</maven.compiler.source></properties>",
     "<sources><source>${project.build.directory}/gen</source></sources>", True),
    ("<properties><javaVersion>8</javaVersion>"
     "<maven.compiler.source>1.${javaVersion}</maven.compiler.source></properties>",
     "", True),
    # A directory reached through a property, and a property maven sets used directly
    ("<maven.compiler.source>1.7</maven.compiler.source>"
     "<properties><gen.dir>${project.build.directory}/gen</gen.dir></properties>",
     "<sources><source>${gen.dir}</source><source>${maven.multiModuleProjectDirectory}/x"
     "</source></sources>", True),
    # A version left to the environment is not known
    ("<properties><jdk>${env.JAVA_VERSION}</jdk></properties><release>8</release>",
     "<maven.compiler.release>${jdk}</maven.compiler.release>", False),
    # A place maven sets, named alone
    ("<maven.compiler.source>1.7</maven.compiler.source>",
     "<sources><source>${basedir}</source></sources>", True),
    # A compiler property a parent outside the build sets is not known
    ("<plugin><configuration><release>${maven.compiler.release}</release></configuration>"
     "</plugin>", "<source>8</source>", False)])
def test_maven_builds_a_project_for_old_java_with_the_old_jdk(root, module, want_old,
                                                             monkeypatch, tmp_path):
    repo_dir, jdk_dir, bin_dir = tmp_path / "repo", tmp_path / "jdk11", tmp_path / "bin"
    (repo_dir / "core").mkdir(parents=True)
    (repo_dir / "pom.xml").write_text(
        f"<project><modules><module>core</module></modules>{root}</project>")
    (repo_dir / "core" / "pom.xml").write_text(f"<project>{module}</project>")
    # A pom.xml the build does not name (a test's resource) is not the project's
    (repo_dir / "src" / "test" / "resources").mkdir(parents=True)
    (repo_dir / "src" / "test" / "resources" / "pom.xml").write_text(
        "<project><release>6</release></project>")
    (jdk_dir / "bin").mkdir(parents=True)
    bin_dir.mkdir()
    (bin_dir / "mvn").write_text('#!/bin/sh\nprintf "%s|%s" "${JAVA_HOME:-none}" "$PATH"\n')
    (bin_dir / "mvn").chmod(0o755)
    monkeypatch.setattr(config, "PROJECT_DIR", str(repo_dir))
    monkeypatch.setattr(edit, "_OLD_JDK_DIR", str(jdk_dir))
    monkeypatch.setattr(edit, "_old_jdk_refuse_dir_set", set())
    monkeypatch.setattr(edit, "_default_jdk_version", lambda: 21)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    java_home, path = edit.run_in_project([str(bin_dir / "mvn"), "test"],
                                          30).stdout.split("|")
    assert (java_home == str(jdk_dir)) is want_old
    assert path.startswith(str(jdk_dir / "bin") + ":") is want_old
    # Another command, and an image without the old JDK, are left alone
    assert edit._maven_env(["npm", "test"], "") == {}
    monkeypatch.setattr(edit, "_OLD_JDK_DIR", str(tmp_path / "absent"))
    assert edit._maven_env([str(bin_dir / "mvn"), "test"], "") == {}


def _old_jdk_project(tmp_path, monkeypatch, mvn_body: str):
    """Make a Java 7 project, an old JDK and a stand-in mvn, and return mvn's path."""
    repo_dir, jdk_dir, bin_dir = tmp_path / "repo", tmp_path / "jdk11", tmp_path / "bin"
    repo_dir.mkdir()
    (repo_dir / "pom.xml").write_text("<project><release>7</release></project>")
    (jdk_dir / "bin").mkdir(parents=True)
    bin_dir.mkdir()
    (bin_dir / "mvn").write_text(f"#!/bin/sh\n{mvn_body}\n")
    (bin_dir / "mvn").chmod(0o755)
    monkeypatch.setattr(config, "PROJECT_DIR", str(repo_dir))
    monkeypatch.setattr(edit, "_OLD_JDK_DIR", str(jdk_dir))
    monkeypatch.setattr(edit, "_old_jdk_refuse_dir_set", set())
    monkeypatch.setattr(edit, "_default_jdk_version", lambda: 21)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    return repo_dir, jdk_dir, bin_dir / "mvn"


def test_a_module_only_a_profile_the_old_jdk_skips_names_is_not_read(tmp_path, monkeypatch):
    repo_dir, jdk_dir, mvn = _old_jdk_project(tmp_path, monkeypatch, "true")
    (repo_dir / "pom.xml").write_text(
        "<project><release>7</release><profiles><profile><activation><jdk>[17,)</jdk>"
        "</activation><modules><module>j17</module></modules></profile></profiles>"
        "</project>")
    (repo_dir / "j17").mkdir()
    (repo_dir / "j17" / "pom.xml").write_text("<project><release>17</release></project>")
    assert edit._maven_env([str(mvn), "test"], "")["JAVA_HOME"] == str(jdk_dir)


@pytest.mark.parametrize("refusal", [
    "java.lang.UnsupportedClassVersionError: X has been compiled by a more recent version "
    "of the Java Runtime (class file version 61.0)",
    "Detected JDK version 11.0.32 (JAVA_HOME=/opt/x) is not in the allowed range [17,).",
    "Error: Apache Maven 4.x requires Java 17 or newer to run.",
    "[ERROR] bad class file: /m2/x/X.class\n    class file has wrong version 65.0, "
    "should be 55.0",
    "Detected JDK Version: 11.0.25 is not in the allowed range [17,).",
    "java.lang.IllegalArgumentException: Unsupported class file major version 61"])
def test_a_build_that_refuses_the_old_jdk_runs_again_with_the_default(
        refusal, tmp_path, monkeypatch):
    _, jdk_dir, mvn = _old_jdk_project(
        tmp_path, monkeypatch,
        f'if [ "${{JAVA_HOME:-}}" = "{tmp_path}/jdk11" ]; then '
        f'echo "{refusal}"; exit 1; fi\n'
        'echo "built with ${JAVA_HOME:-the default}"')
    proc = edit.run_in_project([str(mvn), "test"], 30)
    assert (proc.returncode, proc.stdout) == (0, "built with the default\n")
    # The next build there starts on the default JDK
    assert edit._maven_env([str(mvn), "test"], "") == {}


def test_the_default_jdk_s_version_is_read_off_its_release_file(tmp_path, monkeypatch):
    jdk_bin = tmp_path / "jdk" / "bin"
    jdk_bin.mkdir(parents=True)
    (jdk_bin / "javac").write_text("#!/bin/sh\n")
    (jdk_bin / "javac").chmod(0o755)
    (tmp_path / "jdk" / "release").write_text('IMPLEMENTOR="x"\nJAVA_VERSION="21.0.13"\n')
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "javac").symlink_to(jdk_bin / "javac")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    edit._default_jdk_version.cache_clear()
    try:
        assert edit._default_jdk_version() == 21
        edit._default_jdk_version.cache_clear()
        (tmp_path / "jdk" / "release").write_text('JAVA_VERSION="1.8.0_402"\n')
        assert edit._default_jdk_version() == 8
        edit._default_jdk_version.cache_clear()
        (tmp_path / "jdk" / "release").unlink()
        assert edit._default_jdk_version() is None
        edit._default_jdk_version.cache_clear()
        (tmp_path / "bin" / "javac").unlink()
        assert edit._default_jdk_version() is None
    finally:
        edit._default_jdk_version.cache_clear()


def test_the_second_run_has_what_is_left_of_the_time(tmp_path, monkeypatch):
    _, jdk_dir, mvn = _old_jdk_project(
        tmp_path, monkeypatch, 'sleep 1; echo "requires Java 17"; exit 1')
    with pytest.raises(subprocess.TimeoutExpired):
        edit.run_in_project([str(mvn), "test"], 1.6)


def test_a_release_8_build_with_a_profile_for_newer_jdks_keeps_the_default(tmp_path,
                                                                           monkeypatch):
    """The old JDK would skip the profile; Java 7 has no other JDK to build it."""
    repo_dir, jdk_dir, mvn = _old_jdk_project(tmp_path, monkeypatch, "true")
    profile = ("<profiles><profile><activation><jdk>[17,)</jdk></activation><modules>"
               "<module>jdk17</module></modules></profile></profiles>")
    (repo_dir / "jdk17").mkdir()
    (repo_dir / "jdk17" / "pom.xml").write_text("<project><release>17</release></project>")
    (repo_dir / "pom.xml").write_text(f"<project><release>8</release>{profile}</project>")
    assert edit._maven_env([str(mvn), "test"], "") == {}
    (repo_dir / "pom.xml").write_text(f"<project><release>7</release>{profile}</project>")
    assert edit._maven_env([str(mvn), "test"], "")["JAVA_HOME"] == str(jdk_dir)
    # Without a profile of that kind a release 8 build takes the old JDK
    (repo_dir / "pom.xml").write_text("<project><release>8</release></project>")
    assert edit._maven_env([str(mvn), "test"], "")["JAVA_HOME"] == str(jdk_dir)


def test_a_build_that_fails_otherwise_on_the_old_jdk_is_not_run_again(tmp_path,
                                                                      monkeypatch):
    count = tmp_path / "count"
    _, jdk_dir, mvn = _old_jdk_project(
        tmp_path, monkeypatch,
        f'echo x >> {count}; echo "[ERROR] COMPILATION ERROR"; exit 1')
    assert edit.run_in_project([str(mvn), "test"], 30).returncode == 1
    assert count.read_text() == "x\n"


@pytest.mark.parametrize("spec, want", [
    ("[17,)", False), ("1.8", False), ("11", True), ("!1.8", True), ("!11", False),
    ("(,11]", True), ("(,11)", False), ("[1.6,1.8),[11,)", True), ("(11,17)", False),
    ("17+", None), ("[1.8,x)", None), ("1.8.0", None)])
def test_a_profile_s_jdk_activation_is_read_for_the_old_jdk(spec, want):
    assert edit._jdk_activation_match(spec, 11) is want


def _with_install(monkeypatch, tmp_path, exit_code: int, out: str) -> pathlib.Path:
    """Give JavaScript an install command that leaves a mark, prints out and exits."""
    mark = tmp_path / "installed"
    argv = (sys.executable, "-c", f"import sys; open({str(mark)!r}, 'w').close(); "
                                  f"print({out!r}); sys.exit({exit_code})")
    monkeypatch.setitem(LANG_DICT, "js", dataclasses.replace(LANG_DICT["js"],
                                                             install_command_tuple=argv))
    return mark


def test_a_failed_install_in_a_container_stops_short_of_the_tests(
        as_language, fake_test_command, monkeypatch, tmp_path, repo):
    """In a container, an install that fails runs no test and stops short of them, so the
    cache is not marked as filled."""
    as_language("js")
    monkeypatch.setenv("CTLOOM_WORK", str(repo))
    mark = _with_install(monkeypatch, tmp_path, 1, "ctloom-install-js: could not install")
    ran = tmp_path / "ran"
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", f"open({str(ran)!r}, 'w').close(); print('# pass 1')"])
    result = edit.run_test()
    assert mark.exists() and not ran.exists()
    assert result["status"] == "did not run" and edit.stop_short_run(result) is True
    assert edit.fetch_fail_run(result) is True
    assert "ctloom-install-js: could not install" in result["output"]


def test_an_install_in_a_container_that_passes_lets_the_tests_run(
        as_language, fake_test_command, monkeypatch, tmp_path, repo):
    as_language("js")
    monkeypatch.setenv("CTLOOM_WORK", str(repo))
    mark = _with_install(monkeypatch, tmp_path, 0, "")
    fake_test_command(0, "# tests 1\n# pass 1\n# fail 0")
    result = edit.run_test()
    assert mark.exists() and result["status"] == "passed"


def test_the_static_check_runs_after_the_install_in_a_container(
        as_language, monkeypatch, tmp_path, repo):
    """The check before the change and the checks after it see the same packages: the
    install comes before the first check, not with the first test run."""
    as_language("js")
    monkeypatch.setenv("CTLOOM_WORK", str(repo))
    mark = _with_install(monkeypatch, tmp_path, 0, "")
    seen_list = []
    monkeypatch.setattr(config, "CHECK_COMMAND", [sys.executable, "-c", "print()"])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [])
    monkeypatch.setattr(edit, "_check_count",
                        lambda argv: seen_list.append(mark.exists()) or collections.Counter())
    _RUN_CHECK()
    assert seen_list == [True]


def test_no_install_runs_outside_a_container_or_when_it_is_not_in_the_image(
        as_language, fake_test_command, monkeypatch, tmp_path, repo):
    """On the host the user's packages are not touched; an image without the command runs
    the tests as before."""
    as_language("js")
    monkeypatch.delenv("CTLOOM_WORK", raising=False)
    mark = _with_install(monkeypatch, tmp_path, 1, "ctloom-install-js: x")
    fake_test_command(0, "# tests 1\n# pass 1\n# fail 0")
    assert edit.run_test()["status"] == "passed" and not mark.exists()
    monkeypatch.setenv("CTLOOM_WORK", str(repo))
    monkeypatch.setitem(LANG_DICT, "js", dataclasses.replace(
        LANG_DICT["js"], install_command_tuple=(str(tmp_path / "no-such-command"),)))
    assert edit.run_test()["status"] == "passed"


def test_a_line_for_a_stop_before_the_tests_says_nothing_of_a_run_that_reached_a_verdict(
        as_language, fake_test_command):
    as_language("java")
    fake_test_command(1, "[ERROR] Tests run: 2, Failures: 1, Errors: 0, Skipped: 0\n"
                         "[ERROR] the report could not be resolved")
    result = edit.run_test()
    assert result["status"] == "failed" and edit.stop_short_run(result) is False
    fake_test_command(0, "[INFO] No tests to run.\n[ERROR] x could not be resolved")
    result = edit.run_test()
    assert edit.no_test_run(result) and edit.stop_short_run(result) is False


# The primary language each captured failing run is read as in place of its own
_OTHER_LANGUAGE_DICT = {"kt": "java", "java": "kt", "py": "js", "js": "py", "ts": "rs",
                        "rs": "cs", "cs": "r", "r": "c", "c": "cpp", "cpp": "c"}


@pytest.mark.parametrize("name", sorted(n for n in _GOLDEN_STATUS
                                        if n.split("_")[1] in ("R", "B")))
def test_the_tool_of_another_language_that_ran_the_tests_red_did_not_stop_short(
        name, as_language, golden_test_command):
    """A repository whose tests run under a tool that is not its language's own: the
    run ended with the tests red, and nothing says it stopped before them."""
    as_language(_OTHER_LANGUAGE_DICT[name.split("_")[0]])
    golden_test_command(name)
    result = edit.run_test()
    assert result["status"] != "passed"
    assert edit.stop_short_run(result) is False


def test_a_run_past_its_time_limit_did_not_stop_short_and_one_not_started_did(
        py_tests, fake_test_command, monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND",
                        [sys.executable, "-c", "import time; time.sleep(30)"])
    monkeypatch.setattr(config, "TEST_TIMEOUT_SEC", 1)
    result = edit.run_test()
    assert result["status"] == "did not run" and edit.stop_short_run(result) is False
    monkeypatch.setattr(config, "TEST_COMMAND", ["ctloom-no-such-test-program"])
    assert edit.stop_short_run(edit.run_test()) is True
    monkeypatch.setattr(config, "RUN_TEST", False)
    assert edit.stop_short_run(edit.run_test()) is True


def test_one_command_of_several_that_stopped_short_makes_the_run_one_that_did(
        py_tests, repo, fake_test_command, monkeypatch):
    """The other language's tool ends before its tests while the primary one passes."""
    (repo / "web").mkdir()
    (repo / "web" / "package.json").write_text("{}\n")
    (repo / "web" / "test").mkdir()
    fake_test_command(0, "2 passed in 0.10s")
    stop_argv = [sys.executable, "-c", "import sys; sys.exit(254)"]
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [("js", stop_argv)])
    monkeypatch.setattr(edit, "other_test_list", lambda: [("js", stop_argv, "web")])
    monkeypatch.setattr(edit, "_has_other_test_path", lambda ext, directory: True)
    result = edit.run_test()
    assert result["status"] == "did not run" and edit.stop_short_run(result) is True
    none_argv = [sys.executable, "-c", "import sys; print('No tests found'); sys.exit(1)"]
    monkeypatch.setattr(edit, "other_test_list", lambda: [("js", none_argv, "web")])
    result = edit.run_test()
    assert result["status"] == "passed" and edit.stop_short_run(result) is False


@pytest.mark.parametrize("name", sorted(n for n in _GOLDEN_STATUS
                                        if n.split("_")[1] == "R"))
def test_a_real_failing_test_is_named(name, as_language, golden_test_command):
    """_FAIL_RE_DICT has to match the reporter each language really has."""
    as_language(name.split("_")[0])
    golden_test_command(name)
    r = edit.run_test()
    assert r["failed"], f"no failing test name was read out of {name}"
    assert r["failed_excerpt"]


# Every file under tests/data/unittest_output holds one run of
# `python -B -m unittest discover -s tests` over a small repository, taken under the
# Python its name ends with. What each state has to come back as:
# state -> (the status, the failing names)
_UNITTEST_DIR = pathlib.Path(__file__).parent / "data" / "unittest_output"
_UNITTEST_MUL = "test_mul (test_calc.CalcTest.test_mul)"
_UNITTEST_SUB = "test_sub (test_calc.CalcTest.test_sub)"
_UNITTEST_DICT = {
    "G": ("passed", []),
    "R": ("failed", [_UNITTEST_MUL, _UNITTEST_SUB]),
    "R_one": ("failed", [_UNITTEST_MUL]),
    "R_error": ("failed", ["test_div (test_calc.CalcTest.test_div)", _UNITTEST_SUB]),
    "R_sub": ("failed", ["test_sub_table (test_calc.TableTest.test_sub_table)"]),
    # A failing subtest whose arguments' repr changes run to run (a function, an object
    # with no repr of its own, a set of strings), and two failing subtests of one test,
    # each taken twice: the second run (2) gives the names the first one does
    **{f"R_sub_{kind}{run}": ("failed", [f"{method} (test_calc.EachTest.{method})"] * count)
       for kind, method, count in (("fn", "test_each_function", 1), ("obj", "test_each_case", 1),
                                   ("set", "test_each_tag", 1), ("two", "test_table", 2))
       for run in ("", "2")},
    "R_class": ("failed", ["setUpClass (test_calc.SetUpTest)"]),
    "R_verbose": ("failed", [_UNITTEST_MUL, _UNITTEST_SUB]),
    "B": ("failed", ["test_calc (unittest.loader._FailedTest.test_calc)"]),
    # The tests directory holds no test file (N), or a file that holds no test
    "N": ("did not run", []),
    "N_empty": ("did not run", []),
    # A test that skips one of its subtests and passes the others (S_sub), and every
    # test skipped whole (S_all), the same under -v: without -v the two print the same
    # count of skips as tests and are read as runs over no tests
    "S_sub": ("did not run", []),
    "S_sub_verbose": ("passed", []),
    "S_all": ("did not run", []),
    "S_all_verbose": ("did not run", []),
    # Two passing tests beside two classes skipped in setUpClass (S_class), or beside a
    # class and a module skipped in setUpModule (S_module), whose tests "Ran N" does not
    # count; two passing tests beside a test with a docstring that skips a subtest and
    # one with a docstring skipped whole (S_doc)
    **{f"S_{kind}{verbose}": ("passed", []) for kind in ("class", "module", "doc")
       for verbose in ("", "_verbose")},
}


@pytest.fixture
def unittest_run(as_language, monkeypatch):
    """Return a helper replaying one captured unittest run as the test command of a
    Python repository and returning what run_test() read of it."""
    as_language("py")

    def run(name, directory=_UNITTEST_DIR):
        monkeypatch.setattr(
            config, "TEST_COMMAND",
            [sys.executable, "-c",
             "import json, pathlib, sys;"
             "d = json.loads(pathlib.Path(sys.argv[1]).read_text());"
             "sys.stdout.write(d['output']); sys.exit(d['returncode'])",
             str(directory / f"{name}.json")])
        return edit.run_test()
    return run


@pytest.mark.parametrize("python", ["py311", "py312"])
@pytest.mark.parametrize("state", sorted(_UNITTEST_DICT))
def test_a_real_unittest_run_is_read_with_the_names_of_its_failing_tests(
        state, python, unittest_run):
    result = unittest_run(f"{state}_{python}")
    assert (result["status"], result["failed"]) == _UNITTEST_DICT[state]
    assert edit.stop_short_run(result) is False
    assert bool(result["failed_excerpt"]) is (result["status"] != "passed")
    assert edit.no_test_run(result) is (result["status"] == "did not run")
    # How many tests ran: what it ran less the tests it skipped, whole ones alone where
    # -v tells them apart, and at least the tests its line of one character per test
    # says passed
    count_dict = {"S_sub_verbose": 1, "S_class": 2, "S_class_verbose": 2, "S_module": 2,
                  "S_module_verbose": 2, "S_doc": 2, "S_doc_verbose": 3}
    if state in count_dict:
        assert edit.primary_part(result).ran_count == count_dict[state]


@pytest.mark.parametrize("python", ["py311", "py312"])
def test_the_failing_tests_of_two_unittest_runs_are_told_apart_by_name(python,
                                                                        unittest_run):
    """A run with one of two failing tests fixed names no test the run before it did
    not, and one that fails another test as often names that test."""
    before_set = set(unittest_run(f"R_{python}")["failed"])
    assert set(unittest_run(f"R_one_{python}")["failed"]) - before_set == set()
    assert set(unittest_run(f"R_error_{python}")["failed"]) - before_set == {
        "test_div (test_calc.CalcTest.test_div)"}


@pytest.mark.parametrize("ext, line, name", [
    # gradle test --continue: JUnit 5's name of a parameterized test whose argument has no
    # toString() of its own, and of one whose argument is a lambda
    ("kt", "FooTest > doubles(Obj) > [2] com.example.Obj@{h} FAILED",
     "FooTest > doubles(Obj) > [2] com.example.Obj"),
    ("kt", "FooTest > applies(Function) > [1] com.example.FooTest$$Lambda/0x0000{h}0{h}@{h} "
           "FAILED", "FooTest > applies(Function) > [1] com.example.FooTest$$Lambda"),
    ("kt", "FooTest > Inner > pairs(Obj, Obj) > [1] a.Obj@{h}, a.Obj@{h} FAILED",
     "FooTest > Inner > pairs(Obj, Obj) > [1] a.Obj, a.Obj"),
    # mvn test: surefire's line, with the name JUnit 5 gives the test
    ("java", "[ERROR] com.example.FooTest.[2] com.example.Obj@{h} -- Time elapsed: 0.003 s <<< "
             "FAILURE!", "com.example.FooTest.[2] com.example.Obj"),
    # What does not change run to run is kept
    ("kt", "FooTest > doubles(int) > [2] 2 FAILED", "FooTest > doubles(int) > [2] 2"),
    ("kt", "FooTest > mails() > [1] ann@Example.org FAILED", "FooTest > mails() > [1] ann@Example.org"),
    ("java", "[ERROR] com.example.FooTest.doubles(int)[2] -- Time elapsed: 0.003 s <<< FAILURE!",
     "com.example.FooTest.doubles(int)[2]")])
def test_a_jvm_test_s_name_leaves_out_what_differs_from_run_to_run(ext, line, name):
    """Two runs of the same tests give the same name."""
    assert {edit._fail_name(line.replace("{h}", h), ext)
            for h in ("6d06d69c", "1b2c3d")} == {name}


@pytest.mark.parametrize("python", ["py311", "py312"])
@pytest.mark.parametrize("kind", ["fn", "obj", "set"])
def test_two_unittest_runs_of_one_failing_subtest_name_the_same_test(kind, python,
                                                                    unittest_run):
    """The subtest's arguments, printed differently by the two runs, are no part of it."""
    first, second = (unittest_run(f"R_sub_{kind}{run}_{python}") for run in ("", "2"))
    assert first["failed"] == second["failed"]
    if kind != "set":
        assert first["output"] != second["output"]


@pytest.mark.parametrize("line", [
    "FAILED (failures=2)", "FAILED (errors=1)", "FAILED (failures=1, errors=1)",
    "FAILED (skipped=1, unexpected successes=1)", "FAILED (unexpected successes=1)",
    "FAILED (expected failures=1, unexpected successes=2)",
    # pytest's own lines that name no test
    "ERROR: file or directory not found: tests/none", "ERROR: usage: pytest [options]",
    "ERROR:root:the request failed (see the log)", "FAIL: check_add expected 3 got 4",
    # a subtest that passed or was skipped (-rA)
    "SUBPASSED(i=0) tests/t.py::T::test_y", "SUBSKIPPED[a] tests/t.py::test_y - later",
    # a file whose path holds " - " and that a message follows: where its path ends is
    # not known
    "ERROR tests/basics - part1/test_x.py - x",
    # words that end the line in a file's name with no "/" before it
    "ERROR could not open config.yaml"])
def test_a_line_that_counts_or_reports_names_no_failing_python_test(line):
    assert edit._fail_name(line, "py") is None


@pytest.mark.parametrize("line, name", [
    ("FAIL: test_sub (tests.test_calc.CalcTest)", "test_sub (tests.test_calc.CalcTest)"),
    ("ERROR: test_calc (unittest.loader._FailedTest)",
     "test_calc (unittest.loader._FailedTest)"),
    ("FAIL: test_sub (test_calc.CalcTest.test_sub) [the table] (i=1)",
     "test_sub (test_calc.CalcTest.test_sub)"),
    ("FAIL: test_x (t.T.test_x) (fn=<function sub at 0x7b87bd05d1c0>)", "test_x (t.T.test_x)"),
    ("FAILED (a)/test_x.py::test_y - assert 1 == 2", "(a)/test_x.py::test_y"),
    ("ERROR (failures)/test_x.py", "(failures)/test_x.py"),
    # a path that holds " - "
    ("FAILED tests/basics - part1/t.py::test_x - assert 0", "tests/basics - part1/t.py::test_x"),
    ("FAILED tests/part 1 - a/t.py::T::test_x[a - b] - x",
     "tests/part 1 - a/t.py::T::test_x[a - b]"),
    ("ERROR tests/t.py - ValueError: a.b::c", "tests/t.py"),
    # a file that could not be collected, whose path holds " - " or a space, with no
    # message
    ("ERROR tests/basics - part1/test_x.py", "tests/basics - part1/test_x.py"),
    ("ERROR tests/unit tests/test_x.py", "tests/unit tests/test_x.py"),
    ("FAILED tests/v1.0 - x/t.py", "tests/v1.0 - x/t.py"),
    # pytest 9's line for a failing subtest: the test's id, whatever its msg and
    # arguments say
    ("SUBFAILED(i=1) tests/t.py::T::test_y - AssertionError: 1 == 1", "tests/t.py::T::test_y"),
    ("SUBFAILED[a b] tests/t.py::test_y - assert 0", "tests/t.py::test_y"),
    ("SUBFAILED[the table] (i=1, s=a - b) tests/t.py::T::test_y - x", "tests/t.py::T::test_y"),
    ("SUBFAILED[x - y] (<obj at 0x7f>) tests/t.py::test_y[1 2] - x", "tests/t.py::test_y[1 2]"),
    ("SUBFAILED(<subtest>) tests/t.py::test_y", "tests/t.py::test_y")])
def test_a_python_failure_line_of_either_runner_is_named(line, name):
    assert edit._fail_name(line, "py") == name


def test_a_failure_line_longer_than_the_head_it_is_read_by_is_named_by_its_head():
    """A long message after the name (-vv, CI) leaves the name read; a name that runs on
    past the head is not read."""
    message = " - AssertionError: assert {} == {" + "'key': 1, " * 1000 + "}"
    assert edit._fail_name("FAILED tests/t.py::test_big" + message, "py") \
        == edit._fail_name("ERROR tests/t.py::test_big" + message, "py") \
        == "tests/t.py::test_big"
    assert edit._fail_name("FAIL: test_x (t.T.test_x) [" + "m" * 3000 + "] (i=1)", "py") \
        == "test_x (t.T.test_x)"
    head = "a" * edit._MAX_FAIL_LINE_CHAR
    for line, ext in ((f"FAILED tests/t.py::test_big[{head}] - x", "py"),
                      (f"FAILED tests/{head}.py", "py"),
                      (f" FAIL  test/{head}.test.js", "js")):
        assert edit._fail_name(line, ext) is None


def test_a_long_line_whose_name_may_end_past_the_head_names_no_test():
    """Where what ends the name lies past the head - a parameter id that is not closed
    in it, a display name before " -- Time elapsed", a name that runs to the end of the
    line - the line names no test, and a name the head ends is still read."""
    size = edit._MAX_FAIL_LINE_CHAR
    kt_head = "FooTest > "
    for line, ext in (
            ("FAILED tests/t.py::test_x[case - " + "p" * size + "1]", "py"),
            ("FAILED tests/t.py::test_x[case - " + "p" * size + "1] - assert 0", "py"),
            ("SUBFAILED(i=1) tests/t.py::test_x[case - " + "p" * size + "] - x", "py"),
            ("[ERROR] com.e.FooTest.doubles(int)[2] " + "y " * size
             + "-- Time elapsed: 0.1 s <<< FAILURE!", "java"),
            ("  ● FooTest › bar " + "y " * size, "js"),
            ("  ● FooTest › bar" + " y" * size, "js"),
            ("    ✖ bar" + " y" * size + " (1.2ms)", "js"),
            ("  ● FooTest › bar " + "y " * size, "ts"),
            (kt_head + "x" * (size - len(kt_head) - len(" FAILED")) + " FAILED"
             + "y" * 50 + " FAILED", "kt")):
        assert len(line) > size
        assert edit._fail_name(line, ext) is None
    long_message = "<" + "x" * size + ">"
    assert edit._fail_name("[ERROR]   FooTest.bar:12 expected: " + long_message, "java") \
        == "FooTest.bar"
    assert edit._fail_name("[ERROR] com.e.FooTest.bar -- Time elapsed: 0.1 s " + long_message,
                           "java") == "com.e.FooTest.bar"
    assert edit._fail_name("FAILED tests/t.py::test_x[case - p1] - " + long_message, "py") \
        == "tests/t.py::test_x[case - p1]"


@pytest.mark.parametrize("exit_code, output, status, is_no_test", [
    # A test wrote the line of a run over no tests, and the run's own line counts tests
    (0, "Ran 0 tests in 0.000s\n....\n" + "-" * 70 + "\nRan 4 tests in 0.001s\n\nOK",
     "passed", False),
    (1, "Ran 0 tests in 0.000s\nF\n" + "-" * 70 + "\nRan 1 test in 0.001s\n\n"
        "FAILED (failures=1)", "failed", False),
    # The same under pytest, which counts the tests it ran
    (0, "Ran 0 tests in 0.000s\n.\n1 passed in 0.01s", "passed", False),
    (1, "Ran 0 tests in 0.000s\nF\n1 failed in 0.01s", "failed", False),
    # The line alone, whatever the exit code
    (0, "\n" + "-" * 70 + "\nRan 0 tests in 0.000s\n\nOK", "did not run", True),
    (5, "\n" + "-" * 70 + "\nRan 0 tests in 0.000s\n\nNO TESTS RAN", "did not run", True),
    # Not at the start of a line
    (0, "INFO Ran 0 tests in 0.000s", "passed", False)])
def test_the_line_of_a_unittest_run_over_no_tests_is_read_as_pytest_s_is(
        exit_code, output, status, is_no_test, as_language, fake_test_command):
    as_language("py")
    fake_test_command(exit_code, output)
    result = edit.run_test()
    assert result["status"] == status
    assert edit.no_test_run(result) is is_no_test
    assert edit.stop_short_run(result) is False


def test_a_unittest_run_that_names_no_failing_test_is_red_by_its_last_line(
        as_language, fake_test_command):
    as_language("py")
    fake_test_command(1, "u.\n" + "-" * 70 + "\nRan 2 tests in 0.001s\n\n"
                         "FAILED (unexpected successes=1)")
    result = edit.run_test()
    assert result["status"] == "failed" and result["failed"] == []


# Every file under tests/data/pytest_output holds one run of
# `python -B -m pytest [-q] -p no:cacheprovider [-k nothing_named_so] tests` over a small
# repository; a name ending in _q is the run with -q. What each state has to come back as:
# state -> (the status, the failing names, whether it is a run over no tests)
_PYTEST_DIR = pathlib.Path(__file__).parent / "data" / "pytest_output"
_PYTEST_NO_TEST = ("did not run", [], True)
_PYTEST_DICT = {
    "G": ("passed", [], False),
    "R": ("failed", ["tests/test_ok.py::test_b"], False),
    "R_skip": ("failed", ["tests/test_ok.py::test_b"], False),
    # No test file, or files that hold no test pytest collects: no tests ran / 1 warning
    "N": _PYTEST_NO_TEST, "N_warn": _PYTEST_NO_TEST, "N_class": _PYTEST_NO_TEST,
    # Every test file skipped as it is collected: 1 skipped (exit code 5)
    "S_module": _PYTEST_NO_TEST, "S_module_warn": _PYTEST_NO_TEST,
    "S_module_level": _PYTEST_NO_TEST,
    # Every test left out by -k: 2 deselected (exit code 5)
    "D": _PYTEST_NO_TEST,
    # Every test skipped as it is run: 2 skipped (exit code 0)
    "S_all": _PYTEST_NO_TEST,
}


@pytest.mark.parametrize("form", ["", "_q"])
@pytest.mark.parametrize("state", sorted(_PYTEST_DICT))
def test_a_real_pytest_run_that_counts_no_test_it_ran_is_a_run_over_no_tests(
        state, form, unittest_run):
    result = unittest_run(f"{state}{form}", _PYTEST_DIR)
    assert (result["status"], result["failed"], edit.no_test_run(result)) == \
        _PYTEST_DICT[state]
    assert edit.stop_short_run(result) is False


# More runs under tests/data/pytest_output, of pytest 9 and, where a name ends in
# _pytest8, of pytest 8: unittest's subTest (SU: one of three subtests fails, SU_two:
# two, SU_pass: none, SU_msg: one, with a msg), the subtests fixture (SF: a subtest
# fails, SF_skip: and the test skips after it) and a run told to color its output
# (--color=yes). What each has to come back as, as _PYTEST_DICT says it
_PYTEST_SUBTEST_NAME = "tests/test_y.py::T::test_y"
_PYTEST_FIXTURE_NAME = "tests/test_y.py::test_y"
_PYTEST_MORE_DICT = {
    "SU": ("failed", [_PYTEST_SUBTEST_NAME], False),
    "SU_two": ("failed", [_PYTEST_SUBTEST_NAME] * 2, False),
    "SU_pass": ("passed", [], False),
    "SU_msg": ("failed", [_PYTEST_SUBTEST_NAME], False),
    "SF": ("failed", [_PYTEST_FIXTURE_NAME] * 2, False),
    "SF_skip": ("failed", [_PYTEST_FIXTURE_NAME], False),
    "R_color": ("failed", ["tests/test_ok.py::test_b"], False),
    "G_color": ("passed", [], False),
    "S_module_color": _PYTEST_NO_TEST,
    "SU_color": ("failed", [_PYTEST_SUBTEST_NAME], False),
}
# pytest 8 names a test whose subtests fail once, and has no subtests fixture
_PYTEST8_MORE_DICT = {
    state: ("failed", [_PYTEST_SUBTEST_NAME], False) if state.startswith("SU_") and
    state not in ("SU_pass", "SU_color") else value
    for state, value in _PYTEST_MORE_DICT.items() if not state.startswith("SF")}


@pytest.mark.parametrize("form", ["", "_q"])
@pytest.mark.parametrize("name, value", [
    *[(state, value) for state, value in sorted(_PYTEST_MORE_DICT.items())],
    *[(f"{state}|_pytest8", value) for state, value in sorted(_PYTEST8_MORE_DICT.items())]])
def test_a_real_pytest_run_of_subtests_or_in_color_is_read_by_its_failing_tests(
        name, value, form, unittest_run):
    state, _, suffix = name.partition("|")
    result = unittest_run(f"{state}{form}{suffix}", _PYTEST_DIR)
    assert (result["status"], result["failed"], edit.no_test_run(result)) == value
    assert edit.stop_short_run(result) is False and edit.early_stop_run(result) is False
    # What the model is shown holds no control sequence of the color
    assert "\x1b" not in result["output"] + result["failed_excerpt"]
    if value[1]:
        assert result["failed_excerpt"].startswith(("SUBFAILED", "FAILED"))


@pytest.mark.parametrize("line", [
    "1 skipped in 0.00s", "2 deselected in 0.00s", "1 warning in 0.00s",
    "3 warnings in 0.01s", "1 skipped, 2 deselected, 1 warning in 0.01s",
    "===== 1 skipped in 0.00s =====", "===== no tests ran in 0.00s ====="])
def test_a_pytest_summary_that_counts_no_test_it_ran_says_none_ran(line, as_language,
                                                                    fake_test_command):
    as_language("py")
    for exit_code in (0, 1, 5):
        fake_test_command(exit_code, line)
        result = edit.run_test()
        assert result["status"] == "did not run" and edit.no_test_run(result)
    for other in ("1 passed, " + line, "1 failed, " + line, "1 error, " + line,
                  "1 xfailed, " + line, "1 xpassed, " + line):
        fake_test_command(1, other.replace("===== ", ""))
        assert edit.no_test_run(edit.run_test()) is False


def test_the_line_a_command_prints_for_a_test_file_it_runs_is_a_line_that_is_read():
    """Every language with such a line has it kept between the first and the last
    bytes of the output, whichever file it names."""
    assert set(edit._RUN_LINE_RE_DICT) == {
        ext for ext, lang_config in LANG_DICT.items() if lang_config.run_line_pattern}
    for line in ("[INFO] Running com.e.NewTest", "Running NewTest"):
        assert edit._read_line_count(line, "java") == 1
    assert edit._read_line_count("[INFO] Running com.e.NewTest and more", "java") == 0
    assert edit._read_line_count("[INFO] Running com.e.NewTest", "kt") == 0
    long_line = "Running " + "a." * edit._MAX_FAIL_LINE_CHAR + "NewTest"
    assert edit._read_line_count(long_line, "java") == 0


@pytest.mark.parametrize("output, named", [
    ("[INFO] Running com.e.ATest\n[INFO] Running com.e.NewTest\n", True),
    ("[INFO] Running com.e.ATest\n[INFO] Running com.e.ZTest\n", False),
    # A part of the output was left out where it was received
    ("[INFO] Running com.e.ATest\n... (500 chars omitted)\n[INFO] Running com.e.ZTest\n",
     None),
    ("[INFO] Running com.e.ATest\n... (500 chars omitted)\n[INFO] Running com.e.NewTest\n",
     True),
    ("[INFO] BUILD SUCCESS\n", None)])
def test_a_file_is_not_said_to_be_left_out_of_a_run_whose_output_is_not_whole(
        output, named, monkeypatch):
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", 0)
    assert edit._named_in_run("src/test/java/com/e/NewTest.java",
                              LANG_DICT["java"].run_line_pattern,
                              {"output": output}) is named


@pytest.mark.parametrize("ext, line", [
    ("py", "  Caused by: Failed to fetch: `https://pypi.org/simple/pytest/`"),
    ("cs", "  Failed to restore /fixture/Calc.Tests.csproj (in 5.84 sec).")])
def test_a_stop_before_the_tests_is_found_past_many_blank_lines_in_no_time(ext, line):
    text = "collected 0 items\n" + "    \n" * 20000 + "Segmentation fault\n"
    start = time.monotonic()
    assert edit._STOP_SHORT_RE_DICT[ext].search(text) is None
    assert edit._STOP_SHORT_RE_DICT[ext].search(text + line + "\n") is not None
    assert time.monotonic() - start < 0.5


@pytest.mark.parametrize("name, name_list", [
    ("js_R", ["test/foo.test.js", "FooTest › bar"]),
    ("js_R_many", ["test/a.test.js", "FooTest › bar"]),
    ("js_R_suite", ["test/a.test.js", "FooTest › bar", "test/b.test.js",
                    "Test suite failed to run"]),
    ("js_R_verbose", ["test/a.test.js", "FooTest › bar", "test/b.test.js",
                      "Test suite failed to run"])])
def test_jest_names_each_failing_file_and_test_whether_or_not_it_is_verbose(
        name, name_list, as_language, golden_test_command):
    """jest prints the ✕ lines only for a single file or in verbose mode."""
    as_language("js")
    golden_test_command(name)
    assert edit.run_test()["failed"] == name_list


@pytest.mark.parametrize("name, name_set", [
    ("js_R_node", {"bar", "add", "Calc"}),
    ("js_R_node_spec", {"bar", "add", "Calc"}),
    ("js_R_node_load", {"/fixture/test/broken.test.js"}),
    ("js_R_node_load_spec", {"/fixture/test/broken.test.js"})])
def test_node_test_names_each_failing_test_and_file_in_either_reporter(
        name, name_set, as_language, golden_test_command):
    """node --test prints TAP when its output is not a terminal, and spec otherwise."""
    as_language("js")
    golden_test_command(name)
    assert set(edit.run_test()["failed"]) == name_set


@pytest.mark.parametrize("suffix", ["", "_spec"])
def test_a_node_test_file_that_fails_to_load_is_not_hidden_by_a_test_failing_before(
        suffix, as_language, golden_test_command):
    as_language("js")
    golden_test_command(f"js_R_node{suffix}")
    baseline_set = frozenset(edit.run_test()["failed"])
    golden_test_command(f"js_R_node_load{suffix}")
    assert set(edit.run_test()["failed"]) - baseline_set == {"/fixture/test/broken.test.js"}


def test_gradle_names_a_failing_test_of_a_class_whatever_the_class_is_called(
        as_language, golden_test_command):
    """gradle runs every class with a test in it, CalcCheck as well as CalcTest."""
    as_language("kt")
    golden_test_command("kt_R_check")
    assert edit.run_test()["failed"] == ["CalcCheck > checksOnPurpose()"]


@pytest.mark.parametrize("name, name_list, is_early_stop", [
    # maven leaves out the module that depends on the failing one
    ("java_R_multi", ["com.example.a.ATest.one", "ATest.one"], True),
    # The module whose source does not build ran no test
    ("java_B_multi", ["com.example.b.BTest.one", "BTest.one"], True),
    ("java_R", ["com.example.FooTest.bar", "FooTest.bar"], False),
    ("java_B", [], True),
    # gradle runs every project's tests after the failing one
    ("kt_R_multi", ["ATest > one()"], False),
    ("kt_B_multi", ["BTest > one()"], True),
    ("kt_R", ["FooTest > bar()"], False),
    ("kt_B", [], True),
    # A project whose annotation processing failed ran no test
    ("kt_B_kapt", ["BTest > one()"], True),
    # A name is kept whole: a nested class, a parameter, spaces
    ("java_R_name", ["com.example.FooTest.bar", "com.example.FooTest.named",
                     "com.example.FooTest.doubles(int)[2]", "com.example.FooTest$Inner.deep",
                     "FooTest$Inner.deep", "FooTest.bar", "FooTest.doubles",
                     "FooTest.named"], False),
    ("kt_R_name", ["FooTest > adds two numbers()", "FooTest > bar()",
                   "FooTest > doubles(int) > [2] 2", "FooTest > Inner > deep()"], False),
    ("java_G_multi", [], False), ("kt_G_multi", [], False),
    ("java_N_multi", [], False), ("kt_N_multi", [], False)])
def test_a_run_over_several_modules_says_whether_every_module_ran_its_tests(
        name, name_list, is_early_stop, as_language, golden_test_command):
    as_language(name.split("_")[0])
    golden_test_command(name)
    result = edit.run_test()
    assert result["failed"] == name_list
    assert edit.early_stop_run(result) is is_early_stop
    assert edit.no_test_run(result) is (name.split("_")[1] == "N")


@pytest.mark.parametrize("tail", [40000, 1000, 200, 20])
@pytest.mark.parametrize("name, is_early_stop, is_no_test", [
    ("java_R_multi", True, False), ("java_B_multi", True, False),
    ("kt_B_multi", True, False), ("py_B", True, False),
    ("java_N_multi", False, True), ("kt_N_multi", False, True), ("py_N", False, True),
    ("java_R", False, False), ("kt_R_multi", False, False)])
def test_what_a_run_is_told_by_is_read_off_all_it_printed_whatever_the_tail_keeps(
        name, is_early_stop, is_no_test, tail, as_language, golden_test_command,
        monkeypatch):
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", tail)
    as_language(name.split("_")[0])
    golden_test_command(name)
    result = edit.run_test()
    assert edit.early_stop_run(result) is is_early_stop
    assert edit.no_test_run(result) is is_no_test
    assert [(edit.early_stop_run(part), edit.no_test_run(part))
            for part in edit.part_dict_of(result).values()] == [(is_early_stop, is_no_test)]


_SKIP_C_LINE = "[INFO] c .................................................. SKIPPED"


@pytest.mark.parametrize("tail", [40000, 20])
@pytest.mark.parametrize("name, line_list", [
    ("java_R_multi", [_SKIP_C_LINE]),
    ("java_B_multi", ["[ERROR] COMPILATION ERROR :", _SKIP_C_LINE]),
    ("java_B", ["[ERROR] COMPILATION ERROR :"]),
    ("kt_B_multi", ["Execution failed for task ':a:compileKotlin'."]),
    ("kt_B_kapt", ["Execution failed for task ':a:kaptKotlin'."]),
    ("py_B", ["!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection "
              "!!!!!!!!!!!!!!!!!!!!"]),
    ("java_G_multi", []), ("kt_R_multi", []), ("java_R", []), ("py_N", [])])
def test_a_run_that_left_tests_out_says_what_it_left_out(
        name, line_list, tail, as_language, golden_test_command, monkeypatch):
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", tail)
    as_language(name.split("_")[0])
    golden_test_command(name)
    result = edit.run_test()
    assert edit.early_stop_line_list(result) == line_list
    assert [edit.early_stop_line_list(part)
            for part in edit.part_dict_of(result).values()] == [line_list]
    # A dict that is no run's own result is read by the output it holds
    assert edit.early_stop_line_list({"output": "\n".join(line_list)}) == line_list


def test_a_line_that_says_what_a_run_left_out_is_one_short_line_of_text(as_language):
    as_language("py")
    line = "!!!! Interrupted: \x1b[31mred\x07 text\t" + "x" * 100000 + " !!!!"
    out = f"FAILED tests/test_x.py::test_f\n{line}\n1 failed in 0.01s\n"
    kept = edit._early_stop_line_list(out, "py")
    assert len(kept) == 1 and edit._early_stop_line_list(out, "py") == kept
    assert kept[0].startswith("!!!! Interrupted: \\x1b[31mred\\x07 text x")
    assert kept[0].endswith(" chars omitted)") and "\n" not in kept[0]
    assert len(kept[0]) < edit._MAX_FAIL_LINE_CHAR + 40
    assert not any(char < " " for char in kept[0])


def test_a_run_stopped_at_a_count_of_failing_tests_says_nothing_it_left_out(as_language):
    as_language("py")
    out = ("FAILED tests/test_x.py::test_f\n"
           "!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!\n")
    assert edit._early_stop_in({"output": out}, "py") is True
    assert edit._early_stop_line_list(out, "py") == []
    assert edit._early_stop_line_list(out, "no-such-language") == []


def test_a_dict_that_is_no_run_s_own_result_is_read_by_the_output_it_holds(
        py_tests, fake_test_command, monkeypatch):
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", 30)
    fake_test_command(5, "no tests ran in 0.00s\n" + "a line of a warning\n" * 5)
    result = edit.run_test()
    assert edit.no_test_run(result) is True and edit.no_test_run(dict(result)) is False
    fake_test_command(2, "!!!! Interrupted: 1 error during collection !!!!\n"
                         + "a line of what went wrong\n" * 5 + "1 error in 0.07s")
    result = edit.run_test()
    assert edit.early_stop_run(result) is True
    assert edit.early_stop_run(dict(result)) is False


def test_a_run_says_whether_a_test_command_was_started(py_tests, fake_test_command,
                                                       monkeypatch):
    fake_test_command(0, "2 passed in 0.10s")
    assert edit.is_start_run(edit.run_test()) is True
    monkeypatch.setattr(config, "TEST_COMMAND", ["ctloom-no-such-test-program"])
    result = edit.run_test()
    assert result["status"] == "did not run" and edit.is_start_run(result) is False
    assert "cannot be started" in result["output"]
    monkeypatch.setattr(config, "TEST_COMMAND",
                        [sys.executable, "-c", "import time; time.sleep(30)"])
    monkeypatch.setattr(config, "TEST_TIMEOUT_SEC", 1)
    result = edit.run_test()
    assert result["output"].startswith("timeout") and edit.is_start_run(result) is True
    # A call refused before a command is built started none
    monkeypatch.setattr(config, "RUN_TEST", False)
    assert edit.is_start_run(edit.run_test()) is False


def test_a_module_left_out_under_a_long_name_is_a_run_that_ended_early(
        as_language, fake_test_command):
    as_language("java")
    fake_test_command(1, "[ERROR]   ATest.one:7 expected: <2> but was: <1>\n"
                         "[ERROR] Tests run: 1, Failures: 1, Errors: 0, Skipped: 0\n"
                         "[INFO] a .................................................. "
                         "FAILURE [  1.0 s]\n"
                         "[INFO] Example Platform :: Storage :: Object Store Connector "
                         "SKIPPED\n")
    assert edit.early_stop_run(edit.run_test()) is True


@pytest.mark.parametrize("line, name", [
    ("FAILED tests/t.py::t_x[a b] - AssertionError: assert 'a b' == 'a c'",
     "tests/t.py::t_x[a b]"),
    ("FAILED tests/t.py::t_x[x - y] - AssertionError: assert 'x - y' == 'a c'",
     "tests/t.py::t_x[x - y]"),
    ("FAILED tests/t.py::t_x[a] - assert [1] == [2]", "tests/t.py::t_x[a]"),
    ("FAILED tests/t.py::TestBox::t_in - assert 1 == 2", "tests/t.py::TestBox::t_in"),
    ("FAILED tests/t.py::t_x", "tests/t.py::t_x"),
    ("ERROR tests/t.py - ImportError: no module", "tests/t.py"),
    ("ERROR tests/t.py", "tests/t.py")])
def test_a_failing_pytest_name_keeps_its_whole_parameter_id(
        py_tests, fake_test_command, line, name):
    fake_test_command(1, f"{_SUMMARY_HEAD}{line}\n1 failed in 0.01s")
    assert edit.run_test()["failed"] == [name]


def test_a_directory_whose_conftest_cannot_be_imported_is_named_by_its_error_line(
        py_tests, fake_test_command):
    """pytest names a directory below the ones it was given whose conftest.py fails to
    import as "ERROR <dir> - <error>" and runs the other tests: the directory is the
    failing name, and a run that names it is not one whose failing tests go unnamed."""
    fake_test_command(1, _SUMMARY_HEAD + "ERROR tests/sub - ModuleNotFoundError: No module "
                         "named 'x'\nPASSED tests/test_p.py::test_add\n1 passed, 1 error in 0.04s")
    result = edit.run_test()
    part = edit.primary_part(result)
    assert (result["failed"], part.is_name_short, part.ran_name_list) == (
        ["tests/sub"], False, ["tests/test_p.py::test_add"])
    for line, name in (
            ("ERROR integration - docker.errors.DockerException: no server", "integration"),
            ("ERROR tests/sub - RuntimeError", "tests/sub"),
            # A " - " that is followed by the rest of a test file's path ends no name
            ("ERROR tests/a - b/test_t.py - RuntimeError: boom", None),
            ("ERROR tests/a - b/test_t.py", "tests/a - b/test_t.py"),
            # Only an ERROR line names a directory
            ("FAILED tests/sub - AssertionError", None),
            ("ERROR    root:t.py:5 boom - x", None)):
        assert edit._fail_name(line, "py") == name


def test_a_line_in_pytest_s_shape_names_a_failing_test_under_its_short_summary_alone(
        py_tests, fake_test_command):
    """A line a test or the code prints - a log line "ERROR <name> - <message>" - is no
    failing test outside pytest's short summary: in a unittest run, and in a pytest run
    that prints no short summary or before its head."""
    log = ("ERROR calc - adding\nERROR src/calc.py - adding\nFAILED tests/x.py::t - adding\n"
           "ERROR calc\n")
    fake_test_command(1, log + ".F\n" + "=" * 70 + "\nFAIL: test_bad (test_u.T.test_bad)\n"
                      + "-" * 70 + "\nRan 2 tests in 0.001s\n\nFAILED (failures=1)")
    assert edit.run_test()["failed"] == ["test_bad (test_u.T.test_bad)"]
    # pytest with the command line's -r taking the place of -rap: no name is read
    fake_test_command(1, log + "tests/test_p.py .F\n1 failed, 1 passed in 0.02s")
    result = edit.run_test()
    assert (result["failed"], edit.primary_part(result).is_name_short) == ([], True)
    # The lines from the head of the short summary on are read
    fake_test_command(1, log + _SUMMARY_HEAD + "FAILED tests/t.py::test_b - assert 0\n"
                      "ERROR tests/sub - ModuleNotFoundError: x\n1 failed, 1 error in 0.02s")
    assert edit.run_test()["failed"] == ["tests/t.py::test_b", "tests/sub"]
    # An output a part of which was left out is read line by line as it is
    fake_test_command(1, "FAILED tests/t.py::test_b - assert 0" + CUT_NOTE.format(omitted=9)
                      + "\n1 failed in 0.02s")
    assert edit.run_test()["failed"] == ["tests/t.py::test_b"]


def test_a_python_run_names_what_it_could_not_load_by_its_error_lines(
        py_tests, fake_test_command):
    """pytest's ERROR line of a file or a directory and unittest's test of a module it
    could not import name what the run could not load; a failing test does not."""
    fake_test_command(1, _SUMMARY_HEAD + "FAILED tests/t.py - assert 0\n"
                      "ERROR tests/t2.py::test_x - fixture\nERROR tests/c.py - ImportError: x\n"
                      "ERROR tests/sub - ModuleNotFoundError: x\n1 failed, 3 errors in 0.02s")
    result = edit.run_test()
    assert edit.primary_part(result).load_fail_path_list == ["tests/c.py", "tests/sub"]
    assert edit.unrun_path_list(["tests/t.py", "tests/t2.py", "tests/c.py", "tests/sub/t.py",
                                 "tests/subway/t.py"], result) == ["tests/c.py", "tests/sub/t.py"]
    line = "=" * 70 + "\n{0}\n" + "-" * 70 + "\nImportError: x\n\n"
    fake_test_command(1, "EEF\n" + line.format(
        "ERROR: tests.test_sub (unittest.loader._FailedTest.tests.test_sub)")
        + line.format("ERROR: tests.sub (unittest.loader._FailedTest)")
        + line.format("FAIL: test_bad (tests.test_u.T.test_bad)")
        + "Ran 3 tests in 0.001s\n\nFAILED (failures=1, errors=2)")
    result = edit.run_test()
    assert edit.primary_part(result).load_fail_path_list == [
        "tests/test_sub.py", "tests/test_sub", "tests/sub.py", "tests/sub"]
    assert edit.unrun_path_list(["tests/test_sub.py", "tests/sub/test_s.py", "tests/test_u.py",
                                 "tests/test_sub_more.py"], result) == [
        "tests/test_sub.py", "tests/sub/test_s.py"]


def _unittest_import_fail_text(name: str, frame_path: str | None,
                               function: str | None = "<module>",
                               call_path: str | None = None) -> str:
    """Return a unittest run whose module or package name failed to import, with a
    traceback whose frame names frame_path in function (none for None; no function for
    None), then call_path in make (none for None), and a test that ran."""
    frame = "" if frame_path is None else (
        f'  File "{frame_path}", line 1' + (f", in {function}" if function else "") + "\n")
    if call_path is not None:
        frame += f'  File "{call_path}", line 2, in make\n    import nope\n'
    return ("test_ok (test_sub.T.test_ok) ... ok\n"
            f"{name} (unittest.loader._FailedTest.{name}) ... ERROR\n\n" + "=" * 70
            + f"\nERROR: {name} (unittest.loader._FailedTest.{name})\n" + "-" * 70
            + f"\nImportError: Failed to import test module: {name}\n"
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3/unittest/loader.py", line 137, in loadTestsFromName\n'
            "    module = __import__(module_name)\n" + frame
            + "ModuleNotFoundError: No module named 'nope'\n\n\n" + "-" * 70
            + "\nRan 2 tests in 0.001s\n\nFAILED (errors=1)")


@pytest.mark.parametrize("name, frame, load_fail_list, unrun_list", [
    # unittest given pkg.test_sub names the module by its last part alone
    ("test_sub", "{repo}/pkg/test_sub.py", ["pkg/test_sub.py", "pkg/test_sub"],
     ["pkg/test_sub.py"]),
    ("sub", "{repo}/pkg/sub/__init__.py", ["pkg/sub.py", "pkg/sub"], ["pkg/sub/test_s.py"]),
    # a module that is not there at all: no frame, nothing left out
    ("test_sub", None, [], []),
    # a frame of a file of another name, and one outside the repository
    ("test_sub", "{repo}/pkg/helper.py", [], []),
    ("test_sub", "/usr/lib/python3/test_sub.py", [], [])])
def test_a_unittest_run_names_a_module_it_could_not_import_by_the_file_its_traceback_names(
        name, frame, load_fail_list, unrun_list, repo, py_tests, fake_test_command):
    """unittest names a module it was told to load and could not import by the last part
    of its name; the file is the one the traceback under the ERROR line names, never a
    file of that name the run loaded."""
    fake_test_command(1, _unittest_import_fail_text(
        name, None if frame is None else frame.format(repo=repo)))
    result = edit.run_test()
    assert edit.primary_part(result).load_fail_path_list == load_fail_list
    assert edit.unrun_path_list(["test_sub.py", "pkg/test_sub.py", "pkg/sub/test_s.py",
                                 "pkg/test_more.py"], result) == unrun_list


@pytest.mark.parametrize("function, call_path, load_fail_list", [
    # the module's top level calls a function of another file of the same last name
    ("<module>", "{repo}/other/test_sub.py", ["pkg/test_sub.py", "pkg/test_sub"]),
    ("load_tests", None, ["pkg/test_sub.py", "pkg/test_sub"]),
    # the place of a SyntaxError
    (None, None, ["pkg/test_sub.py", "pkg/test_sub"]),
    # a function of the file alone
    ("make", None, [])])
def test_a_unittest_run_names_a_module_it_could_not_import_by_the_frames_loading_it(
        function, call_path, load_fail_list, repo, py_tests, fake_test_command):
    """Of the traceback under a module unittest could not import, only a frame of a
    module's top level, of its load_tests or of the place of a SyntaxError names the
    module; a file of the same name whose function the module called ran its tests."""
    fake_test_command(1, _unittest_import_fail_text(
        "test_sub", f"{repo}/pkg/test_sub.py", function,
        None if call_path is None else call_path.format(repo=repo)))
    result = edit.run_test()
    assert edit.primary_part(result).load_fail_path_list == load_fail_list
    assert edit.unrun_path_list(["pkg/test_sub.py", "other/test_sub.py", "pkg/test_more.py"],
                                result) == load_fail_list[:1]


def test_a_unittest_run_s_output_is_split_into_lines_as_often_however_many_modules_failed(
        repo, py_tests, fake_test_command, monkeypatch):
    """Each module unittest could not import is read off the lines under its own ERROR
    line, the output split into lines a count of times that does not grow with them."""
    split = edit.line_list_of
    split_list = []
    monkeypatch.setattr(edit, "line_list_of",
                        lambda *arg, **kwarg: split_list.append(1) or split(*arg, **kwarg))
    split_count_list = []
    for count in (1, 6):
        fake_test_command(1, "E" * count + "\n" + "".join(
            "=" * 70 + f"\nERROR: test_m{index} (unittest.loader._FailedTest.test_m{index})\n"
            + "-" * 70 + f"\nImportError: Failed to import test module: test_m{index}\n"
            f'Traceback (most recent call last):\n  File "{repo}/pkg/test_m{index}.py", line 1,'
            " in <module>\nModuleNotFoundError: No module named 'nope'\n\n"
            for index in range(count))
            + "-" * 70 + f"\nRan {count} tests in 0.001s\n\nFAILED (errors={count})")
        split_list.clear()
        result = edit.run_test()
        split_count_list.append(len(split_list))
        assert edit.primary_part(result).load_fail_path_list == [
            path for index in range(count)
            for path in (f"pkg/test_m{index}.py", f"pkg/test_m{index}")]
    assert split_count_list[0] == split_count_list[1]


def test_what_a_run_started_in_a_directory_could_not_load_is_named_from_the_root(repo):
    """A command started in a directory under the repository names what it could not load
    from that directory: the path has the directory in front of it, as the names of the
    tests it ran do."""
    (repo / "py" / "tests").mkdir(parents=True)
    pytest_text = (_SUMMARY_HEAD + "ERROR tests/unit - ModuleNotFoundError: x\n"
                   "1 passed, 1 error in 0.02s")
    unittest_text = _unittest_import_fail_text("test_sub", f"{repo}/py/tests/test_sub.py")
    for text, load_fail_list in ((pytest_text, ["py/tests/unit"]),
                                 (unittest_text, ["py/tests/test_sub.py", "py/tests/test_sub"])):
        argv = [sys.executable, "-c", f"import sys; print({text!r}); sys.exit(1)"]
        verdict, part = edit._one_test_run(argv, [], "", "py", "py")
        assert (verdict, part.load_fail_path_list) == (False, load_fail_list)
    assert edit.unrun_path_list(["tests/unit/calc.test.js", "py/tests/test_sub.py",
                                 "tests/test_sub.py"], part) == ["py/tests/test_sub.py"]
    _, part = edit._one_test_run([sys.executable, "-c", f"print({pytest_text!r}); exit(1)"],
                                 [], "", "py", "py")
    assert edit.unrun_path_list(["tests/unit/calc.test.js", "py/tests/unit/test_a.py"],
                                part) == ["py/tests/unit/test_a.py"]


@pytest.mark.parametrize("name, old, new, target", [
    ("c_R", "FooTest.bar", "tests", "tests/test_calc.c"),
    ("js_R_node_spec", "bar", "test", "test/calc.test.js"),
    ("r_R", "add is wrong here", "tests", "tests/testthat/test-calc.R"),
    ("js_R_suite", "", "", "test/b.test.js")])
def test_a_failing_test_of_another_tool_names_nothing_the_run_could_not_load(
        name, old, new, target, as_language, fake_test_command):
    """A failing test whose name is a test directory's, and a jest file that failed to
    run, are no file or directory the run could not load."""
    as_language(name.split("_")[0])
    record = json.loads((_GOLDEN_DIR / f"{name}.json").read_text())
    fake_test_command(record["returncode"], record["output"].replace(old, new))
    result = edit.run_test()
    assert result["status"] == "failed" and (not new or new in result["failed"])
    assert edit.unrun_path_list([target], result) == []


def test_a_jest_file_that_fails_to_run_is_not_hidden_by_a_test_failing_before(
        as_language, golden_test_command):
    as_language("js")
    golden_test_command("js_R_many")
    baseline_set = frozenset(edit.run_test()["failed"])
    for name in ("js_R_suite", "js_R_verbose"):
        golden_test_command(name)
        assert set(edit.run_test()["failed"]) - baseline_set == {
            "test/b.test.js", "Test suite failed to run"}


@pytest.mark.parametrize("name", sorted(n for n in _GOLDEN_STATUS
                                        if n.split("_")[1] == "G"))
def test_a_real_passing_run_names_nothing(name, as_language, golden_test_command):
    """A green run reports no failures, whatever its output holds."""
    as_language(name.split("_")[0])
    golden_test_command(name)
    r = edit.run_test()
    assert r["failed"] == [] and r["failed_excerpt"] == ""


@pytest.mark.parametrize("name", sorted(_GOLDEN_STATUS))
def test_the_captured_run_is_of_the_command_that_ships(name, golden_test_command):
    """A capture says nothing about a command it was not taken from.

    gradle prints the names of its failing tests only without -q, so the words the
    model reads depend on the flags in LANG as much as on the patterns. Changing a
    test_command without capturing that command's output again fails here.
    """
    captured = golden_test_command(name)
    ext = name.split("_")[0]
    assert tuple(captured["command"]) == LANG_DICT[ext].test_command_tuple


# == write_path_set =============================
def test_written_paths_is_empty_before_anything_lands(repo):
    assert edit.write_path_set() == set()


def test_written_paths_holds_what_each_tool_wrote(repo):
    edit.replace_in_file("m.py", "old", "new")
    edit.create_file("sub/fresh.py", "x = 1")
    assert edit.write_path_set() == {"m.py", "sub/fresh.py"}


def test_a_write_that_did_not_land_is_not_counted(repo):
    edit.replace_in_file("m.py", "not there", "x")
    assert edit.write_path_set() == set()


def test_a_revert_takes_a_tracked_path_back_out_of_what_was_written(repo):
    """The write no longer stands, so this request did not leave the file behind."""
    edit.replace_in_file("m.py", "old", "new")
    assert edit.write_path_set() == {"m.py"}
    assert edit.revert_file("m.py")["ok"] is True
    assert edit.write_path_set() == set()


def test_a_revert_takes_a_created_path_back_out_of_what_was_written(repo):
    """revert_file deletes an untracked file, which is what undoes create_file."""
    edit.create_file("sub/fresh.py", "x = 1")
    assert edit.write_path_set() == {"sub/fresh.py"}
    assert edit.revert_file("sub/fresh.py")["ok"] is True
    assert edit.write_path_set() == set()


def test_deleting_a_created_file_takes_the_path_back_out_of_what_was_written(repo):
    """Nothing was there when the request started and nothing is there now."""
    edit.create_file("sub/fresh.py", "x = 1")
    assert edit.delete_file("sub/fresh.py")["ok"] is True
    assert _git(repo, "status", "--porcelain") == ""
    assert edit.write_path_set() == edit.own_change_set() == edit.no_tool_path_set() == set()
    # So is a file a program left there
    (repo / "out.log").write_text("left by a program\n")
    assert edit.delete_file("out.log")["ok"] is True
    assert edit.write_path_set() == set()
    # A tracked file that was deleted is a change
    assert edit.delete_file("m.py")["ok"] is True
    assert edit.write_path_set() == {"m.py"}


def test_deleting_a_file_where_git_does_not_answer_stays_among_what_was_written(
        tmp_path, monkeypatch):
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "m.py").write_text("old\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path / "plain"))
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    edit.reset_patch_round()
    assert edit.delete_file("m.py")["ok"] is True
    assert edit.write_path_set() == {"m.py"}


def test_a_write_back_to_the_head_contents_takes_the_path_back_out(repo):
    """Writing a file back as it was leaves nothing behind, as a revert does."""
    edit.replace_in_file("m.py", "old", "new")
    edit.replace_in_file("m.py", "new", "older")
    assert edit.write_path_set() == {"m.py"}
    edit.replace_in_file("m.py", "older", "old")
    assert edit.write_path_set() == edit.own_change_set() == edit.no_tool_path_set() == set()


@pytest.mark.parametrize("name, value", [
    ("diff.autoRefreshIndex", "false"), ("core.ignoreStat", "true"),
    ("core.trustctime", "false"), ("core.checkStat", "minimal"),
    ("core.untrackedCache", "true"), ("status.showUntrackedFiles", "no"),
    ("diff.renames", "copies"), ("diff.external", "/bin/false"),
    ("diff.ignoreSubmodules", "all"), ("diff.relative", "true"),
])
def test_what_the_request_changed_is_told_the_same_whatever_git_is_set_to(repo, name,
                                                                         value):
    """A file written back, a file put back and a file written after it was put back
    are each read by their contents, under a setting of the repository's own."""
    _git(repo, "config", name, value)
    edit.reset_patch_round()
    edit.replace_in_file("m.py", "old", "new")
    edit.replace_in_file("m.py", "new", "old")
    assert edit.write_path_set() == edit.own_change_set() == set()
    edit.replace_in_file("m.py", "old", "new")
    assert edit.revert_file("m.py")["ok"] is True
    assert edit.own_change_set() == set()
    assert gitcmd.name_status(str(repo)) == {"added": [], "modified": [], "deleted": []}
    edit.replace_in_file("m.py", "old", "newer")
    edit.create_file("sub/fresh.py", "x = 1")
    assert edit.write_path_set() == edit.own_change_set() == {"m.py", "sub/fresh.py"}
    assert gitcmd.tree_diff_path_list(str(repo), "HEAD") == ["m.py"]


def test_a_write_back_to_what_a_changed_file_held_at_the_start_takes_it_out(repo):
    """A file already changed when the request started starts from that, not from HEAD."""
    (repo / "m.py").write_text("mine\nctx\n")
    edit.reset_patch_round()
    edit.replace_in_file("m.py", "mine", "new")
    assert edit.write_path_set() == {"m.py"}
    edit.replace_in_file("m.py", "new", "mine")
    assert edit.write_path_set() == set()
    edit.replace_in_file("m.py", "mine", "old")
    assert edit.write_path_set() == {"m.py"}


def test_a_revert_that_did_not_go_through_leaves_the_path_there(repo, monkeypatch):
    """Only a revert that happened undoes the write."""
    edit.replace_in_file("m.py", "old", "new")
    real = edit.gitcmd.run

    def refuse_the_checkout(project_dir, *args, **kwargs):
        """Let git answer as it does, except that the checkout itself fails."""
        if "checkout" in args:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="fatal: no")
        return real(project_dir, *args, **kwargs)

    monkeypatch.setattr(edit.gitcmd, "run", refuse_the_checkout)
    assert edit.revert_file("m.py")["ok"] is False
    assert edit.write_path_set() == {"m.py"}


def test_a_dry_run_writes_nothing_so_nothing_is_counted(repo, monkeypatch):
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    edit.replace_in_file("m.py", "old", "new")
    assert edit.write_path_set() == set()


def test_written_paths_is_reset_between_requests(repo):
    edit.replace_in_file("m.py", "old", "new")
    edit.reset_patch_round()
    assert edit.write_path_set() == set()


# == change_by_kind_dict =============================
def test_changed_files_splits_the_three_kinds(repo):
    (repo / "m.py").write_text("new\nctx\n")
    (repo / "gone.py").write_text("x\n")
    _git(repo, "add", "gone.py")
    _git(repo, "commit", "-q", "-m", "second")
    (repo / "gone.py").unlink()
    (repo / "fresh.py").write_text("y\n")
    files = edit.change_by_kind_dict()
    assert files == {"added": ["fresh.py"], "modified": ["m.py"],
                     "deleted": ["gone.py"]}


def test_changed_files_is_empty_on_a_clean_repository(repo):
    assert edit.change_by_kind_dict() == {"added": [], "modified": [], "deleted": []}


def test_changed_files_lists_a_created_file_as_added(repo):
    edit.create_file("sub/new.py", "x = 1")
    assert edit.change_by_kind_dict()["added"] == ["sub/new.py"]


# == git_diff_stat and revert_file ============
def test_the_return_keys_in_the_instructions_match_the_real_ones(repo,
                                                                 fake_test_command):
    """The return keys the instructions name are the keys each tool returns."""
    fake_test_command(0, "1 passed")
    changed = edit.replace_in_file("m.py", "old", "new")
    assert set(changed) == set(agent.TOOL_RESULT_KEY_DICT["replace_in_file"][0])
    assert set(edit.replace_line_range("m.py", 1, 1, "new\n")) == set(
        agent.TOOL_RESULT_KEY_DICT["replace_line_range"][0])
    assert set(edit.create_file("cache.py", "x = 1\n")) == set(
        agent.TOOL_RESULT_KEY_DICT["create_file"][0])
    delete_dict = edit.delete_file("cache.py")
    assert delete_dict["stat"] == "deleted"
    assert set(delete_dict) == set(agent.TOOL_RESULT_KEY_DICT["delete_file"][0])
    assert set(edit.run_test()) == set(agent.TOOL_RESULT_KEY_DICT["run_test"][0])
    stat = edit.git_diff_stat()
    assert set(stat) == set(agent.TOOL_RESULT_KEY_DICT["git_diff_stat"][0])
    assert set(stat["files"][0]) == set(
        agent.TOOL_RESULT_KEY_DICT["git_diff_stat"][1]["files"])
    assert set(edit.revert_file("m.py")) == set(
        agent.TOOL_RESULT_KEY_DICT["revert_file"][0])


def test_git_diff_stat_returns_the_changed_files(repo):
    edit.replace_in_file("m.py", "old", "new")
    stat = edit.git_diff_stat()
    assert [f["path"] for f in stat["files"]] == ["m.py"]
    assert stat["total_insertions"] == 1
    assert stat["total_deletions"] == 1


def test_git_diff_stat_names_a_tracked_file_it_cannot_read_and_counts_the_others(repo):
    (repo / "b.py").write_text("b\n")
    _git(repo, "add", "b.py")
    _git(repo, "commit", "-q", "-m", "b")
    edit.replace_in_file("m.py", "old", "new")
    (repo / "b.py").unlink()
    os.mkfifo(repo / "b.py")
    stat = edit.git_diff_stat()
    assert stat["files"] == [{"path": "b.py", "insertions": "?", "deletions": "?"},
                             {"path": "m.py", "insertions": 1, "deletions": 1}]
    assert (stat["total_insertions"], stat["total_deletions"]) == (1, 1)


def test_git_diff_stat_returns_the_untracked_files(repo):
    (repo / "extra.py").write_text("x = 1\n")
    assert edit.git_diff_stat()["untracked"] == ["extra.py"]


def test_revert_file_deletes_an_untracked_file(repo):
    """That is what undoes a create_file: an untracked file has no HEAD to go back to."""
    edit.create_file("cache.py", "x = 1\n")
    assert edit.revert_file("cache.py")["ok"] is True
    assert not (repo / "cache.py").exists()


def test_revert_file_reports_a_path_that_is_not_there(repo):
    r = edit.revert_file("nope.py")
    assert r["ok"] is False and "does not exist" in r["error"]


def test_git_diff_stat_lists_a_created_file_as_untracked(repo):
    edit.create_file("cache.py", "x = 1\n")
    stat = edit.git_diff_stat()
    assert stat["untracked"] == ["cache.py"] and stat["files"] == []


def test_git_diff_stat_names_the_file_inside_a_new_directory(repo):
    """Without -uall it reports "pkg/", which is not a path the model can act on."""
    edit.create_file("pkg/cache.py", "x = 1\n")
    assert edit.git_diff_stat()["untracked"] == ["pkg/cache.py"]


def test_there_are_no_changes_after_revert_file(repo):
    edit.replace_in_file("m.py", "old", "new")
    assert edit.revert_file("m.py")["ok"] is True
    assert edit.git_diff_stat()["files"] == []
    assert (repo / "m.py").read_text() == "old\nctx\n"


def test_revert_file_restores_head_even_with_staged_changes(repo):
    """git checkout -- restores from the index, not HEAD; omitting HEAD breaks this."""
    (repo / "m.py").write_text("staged\nctx\n")
    _git(repo, "add", "m.py")
    (repo / "m.py").write_text("worktree\nctx\n")
    edit.revert_file("m.py")
    assert (repo / "m.py").read_text() == "old\nctx\n"
    assert edit.git_diff_stat()["files"] == []


def test_revert_file_puts_back_a_link_this_request_made_or_changed_as_its_own_path(repo):
    (repo / "v1").mkdir()
    (repo / "v1" / "x.py").write_text("x\n")
    os.symlink("v1", repo / "cur")
    os.symlink("v1/x.py", repo / "alias.py")
    _git(repo, "add", "-A"), _git(repo, "commit", "-q", "-m", "two links")
    edit.reset_patch_round()
    # A tracked file turned into a link, a tracked link pointed elsewhere, and two
    # links that were not there, none of them through a tool
    (repo / "m.py").unlink()
    os.symlink("v1/x.py", repo / "m.py")
    (repo / "cur").unlink()
    os.symlink("m.py", repo / "cur")
    os.symlink("/etc/hostname", repo / "out")
    os.symlink("nowhere", repo / "dang")
    name_list = ["m.py", "cur", "out", "dang"]
    assert edit.no_tool_path_set() == set(name_list)
    for name in name_list:
        assert edit.revert_file(name) == {"ok": True, "path": name, "error": ""}
    assert not (repo / "m.py").is_symlink() and (repo / "m.py").read_text() == "old\nctx\n"
    assert os.readlink(repo / "cur") == "v1"
    assert not os.path.lexists(repo / "out") and not os.path.lexists(repo / "dang")
    assert (repo / "v1" / "x.py").read_text() == "x\n"
    assert _git(repo, "status", "--porcelain") == "" and edit.own_change_set() == set()
    # A link that is as it was is still put back as the file it points at
    assert edit.replace_in_file("alias.py", "x", "y")["ok"] is True
    assert (repo / "v1" / "x.py").read_text() == "y\n"
    assert edit.revert_file("alias.py")["ok"] is True
    assert (repo / "v1" / "x.py").read_text() == "x\n" and (repo / "alias.py").is_symlink()


def test_a_file_that_differed_at_the_start_and_became_a_link_is_put_back_as_a_file(repo):
    (repo / "n.py").write_text("mine\n")
    edit.reset_patch_round()
    assert edit.replace_in_file("n.py", "mine", "theirs")["ok"] is True
    (repo / "n.py").unlink()
    os.symlink("m.py", repo / "n.py")
    assert edit.revert_file("n.py")["ok"] is True
    assert not (repo / "n.py").is_symlink() and (repo / "n.py").read_text() == "mine\n"
    assert (repo / "m.py").read_text() == "old\nctx\n"


def test_revert_file_restores_a_file_taken_out_of_the_index_during_the_request(repo):
    _git(repo, "rm", "-q", "--cached", "m.py")
    (repo / "m.py").write_text("worktree\nctx\n")
    assert edit.revert_file("m.py")["ok"] is True
    assert (repo / "m.py").read_text() == "old\nctx\n"
    assert _git(repo, "status", "--porcelain") == ""


def test_revert_file_does_not_delete_a_file_the_index_holds_and_head_does_not(repo):
    (repo / "n.py").write_text("x = 1\n")
    _git(repo, "add", "n.py")
    assert edit.revert_file("n.py")["ok"] is False
    assert (repo / "n.py").read_text() == "x = 1\n"


def test_git_diff_stat_sees_staged_changes_too(repo):
    """Omitting HEAD from git diff would make staged changes look like "no change"."""
    (repo / "m.py").write_text("staged\nctx\n")
    _git(repo, "add", "m.py")
    assert [f["path"] for f in edit.git_diff_stat()["files"]] == ["m.py"]


def test_revert_file_rejects_a_path_outside_project_dir(repo):
    out = edit.revert_file("../../etc/passwd")
    assert out["ok"] is False and "outside the repository" in out["error"]


@pytest.mark.parametrize("target", [".", "sub", "sub/"])
def test_revert_file_refuses_a_directory(repo, target):
    """A directory would take every tracked file under it back to HEAD."""
    (repo / "sub").mkdir()
    (repo / "sub" / "a.py").write_text("a\n")
    _git(repo, "add", "sub/a.py")
    _git(repo, "commit", "-q", "-m", "sub")
    edit.replace_in_file("m.py", "old", "new")
    edit.replace_in_file("sub/a.py", "a", "b")
    out = edit.revert_file(target)
    assert out["ok"] is False and "is a directory" in out["error"]
    assert (repo / "m.py").read_text() == "new\nctx\n"
    assert (repo / "sub" / "a.py").read_text() == "b\n"
    assert edit.write_path_set() == {"m.py", "sub/a.py"}


def test_revert_file_writes_nothing_under_a_dry_run(repo, monkeypatch):
    (repo / "m.py").write_text("mine\n")
    edit.reset_patch_round()
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    out = edit.revert_file("m.py")
    assert out["ok"] is False and "dry run" in out["error"]
    assert (repo / "m.py").read_text() == "mine\n"


def test_revert_file_leaves_a_change_made_before_the_request_alone(repo):
    """What nobody committed before the request is not the request's to throw away."""
    (repo / "m.py").write_text("mine\n")
    (repo / "notes.txt").write_text("mine too\n")
    edit.reset_patch_round()
    for path in ("m.py", "notes.txt"):
        out = edit.revert_file(path)
        assert out["ok"] is False and "nothing of this request" in out["error"]
    assert (repo / "m.py").read_text() == "mine\n"
    assert (repo / "notes.txt").read_text() == "mine too\n"


def test_revert_file_puts_back_what_a_changed_file_held_at_the_start(repo):
    (repo / "m.py").write_text("mine\nctx\n")
    (repo / "notes.txt").write_text("mine too\n")
    edit.reset_patch_round()
    edit.replace_in_file("m.py", "mine", "new")
    edit.replace_in_file("notes.txt", "mine too", "theirs")
    edit.delete_file("notes.txt")
    assert edit.revert_file("m.py") == {"ok": True, "path": "m.py", "error": ""}
    assert edit.revert_file("notes.txt")["ok"] is True
    assert (repo / "m.py").read_text() == "mine\nctx\n"
    assert (repo / "notes.txt").read_text() == "mine too\n"
    assert edit.write_path_set() == set()


def test_revert_file_refuses_a_changed_file_no_tool_wrote_first(repo):
    """What it held at the start was never kept, and HEAD would throw it away."""
    (repo / "m.py").write_text("mine\n")
    edit.reset_patch_round()
    (repo / "m.py").write_text("written by a program\n")
    out = edit.revert_file("m.py")
    assert out["ok"] is False and "replace_in_file" in out["error"]
    assert (repo / "m.py").read_text() == "written by a program\n"


def test_a_tracked_file_a_named_pipe_stands_in_for_is_not_waited_on(repo):
    """git lists the path; what is compared, read and written there does not open it."""
    (repo / "m.py").unlink()
    os.mkfifo(repo / "m.py")
    assert edit.own_change_set() == edit.no_tool_path_set() == {"m.py"}
    assert edit.run_in_project([sys.executable, "-c", "pass"], 10).returncode == 0
    pipe_error = f"{repo / 'm.py'} is not a regular file"
    assert edit.replace_in_file("m.py", "old", "new")["error"] == f"cannot read: {pipe_error}"
    assert edit.create_file("m.py", "x = 1\n")["error"] == f"cannot write: {pipe_error}"
    assert stat.S_ISFIFO(os.lstat(repo / "m.py").st_mode)
    assert edit.revert_file("m.py")["ok"] is True
    assert (repo / "m.py").read_text() == "old\nctx\n" and edit.own_change_set() == set()


def test_a_test_command_that_leaves_a_named_pipe_does_not_stop_the_next_runs(
        repo, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "import os; os.remove('m.py'); os.mkfifo('m.py');"
                              "print('1 passed in 0.01s')"])
    assert edit.run_test()["status"] == "passed"
    assert edit.run_test()["status"] == "passed"
    assert edit.own_change_set() == {"m.py"}


def test_a_named_pipe_no_file_was_at_is_neither_read_nor_written(repo, monkeypatch):
    os.mkfifo(repo / "p.py")
    pipe_error = f"{repo / 'p.py'} is not a regular file"
    assert edit.create_file("p.py", "x = 1\n")["error"] == f"cannot write: {pipe_error}"
    assert edit.replace_in_file("p.py", "x", "y")["error"] == f"cannot read: {pipe_error}"
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    assert edit.create_file("p.py", "x = 1\n")["error"] == f"cannot write: {pipe_error}"
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    assert edit.delete_file("p.py")["ok"] is True and not os.path.lexists(repo / "p.py")


def _leave_no_file(path, leave: str) -> None:
    """Put a named pipe, a directory holding a file or an empty directory at a path."""
    if os.path.lexists(path):
        os.remove(path)
    if leave == "named pipe":
        os.mkfifo(path)
    else:
        os.mkdir(path)
        if leave == "directory":
            (path / "keep").write_text("k\n")


@pytest.mark.parametrize("leave", ["named pipe", "directory", "empty directory"])
@pytest.mark.parametrize("start, after", [
    ("new", None), ("HEAD", "old\nctx\n"), ("changed", "mine\nctx\n")])
def test_what_a_tool_wrote_and_is_no_file_now_is_its_change_and_is_reverted(
        repo, leave, start, after):
    """The path counts as this request's change whether git lists it or not, and
    revert_file takes away what stands there before it puts the start back."""
    path = "n.py" if start == "new" else "m.py"
    if start == "changed":
        (repo / "m.py").write_text("mine\nctx\n")
        edit.reset_patch_round()
    if start == "new":
        assert edit.create_file(path, "x = 1\n")["ok"] is True
    else:
        assert edit.replace_in_file(path, "ctx", "new")["ok"] is True
    _leave_no_file(repo / path, leave)
    assert path in edit.own_change_set()
    assert edit.revert_file(path) == {"ok": True, "path": path, "error": ""}
    if after is None:
        assert not os.path.lexists(repo / path)
    else:
        assert (repo / path).read_text() == after
    assert edit.own_change_set() == set()


@pytest.mark.parametrize("leave", [None, "directory"])
def test_a_file_made_where_head_holds_a_directory_is_reverted_to_nothing(repo, leave):
    """HEAD's directory is not taken out: a deletion the user had not committed stays,
    and what this request changed and what git lists agree."""
    (repo / "t").mkdir()
    (repo / "t" / "a.txt").write_text("head\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "t")
    shutil.rmtree(repo / "t")
    edit.reset_patch_round()
    assert edit.create_file("t", "now a file\n")["ok"] is True
    if leave == "directory":
        (repo / "t").unlink()
        (repo / "t").mkdir()
        (repo / "t" / "made_by_run.txt").write_text("run\n")
    assert edit.revert_file("t") == {"ok": True, "path": "t", "error": ""}
    assert not os.path.lexists(repo / "t")
    assert edit.own_change_set() == set()
    assert _git(repo, "status", "--porcelain").splitlines() == [" D t/a.txt"]
    assert edit.git_diff_stat()["files"] == [
        {"path": "t/a.txt", "insertions": 0, "deletions": 1}]


def test_a_directory_above_what_a_tool_wrote_made_a_chain_of_links_too_long_to_resolve(
        repo):
    """What this request changed is still told, the path that cannot be resolved as
    one that cannot be looked at, and only that path is refused."""
    assert edit.create_file("d/f.py", "x = 1\n")["ok"] is True
    shutil.rmtree(repo / "d")
    (repo / "c" / "real").mkdir(parents=True)
    link_count = sys.getrecursionlimit() + 100
    for number in range(link_count):
        (repo / "c" / f"l{number}").symlink_to(
            f"l{number + 1}" if number + 1 < link_count else "real")
    (repo / "d").symlink_to("c/l0")
    with pytest.raises(RecursionError):
        os.path.realpath(repo / "d" / "f.py")
    change_set = edit.own_change_set()
    assert "d" in change_set and "d/f.py" not in change_set
    assert "d" in edit.no_tool_path_set()
    assert "d" in edit.stage_change_hash_dict()
    assert edit.revert_file("d/f.py")["error"].startswith("cannot check the path")
    assert edit.revert_file("d") == {"ok": True, "path": "d", "error": ""}
    assert not os.path.lexists(repo / "d")


def test_what_no_tool_wrote_and_is_no_file_is_left_as_it_is(repo):
    os.mkfifo(repo / "p.py")
    os.mkdir(repo / "d.py")
    assert edit.own_change_set() == set()
    assert edit.revert_file("p.py")["error"] == \
        "p.py holds what it held when this request started, so there is nothing of " \
        "this request to revert"
    assert edit.revert_file("d.py")["error"] == "d.py is a directory, not a file"
    assert stat.S_ISFIFO(os.lstat(repo / "p.py").st_mode) and (repo / "d.py").is_dir()


def test_what_a_path_held_at_the_start_is_not_written_over_a_named_pipe(repo):
    """A tool wrote the path and a later write put back what it held at the start."""
    (repo / "m.py").write_text("mine\nctx\n")
    edit.reset_patch_round()
    assert edit.replace_in_file("m.py", "mine", "new")["ok"] is True
    assert edit.replace_in_file("m.py", "new", "mine")["ok"] is True
    (repo / "m.py").unlink()
    os.mkfifo(repo / "m.py")
    out = edit.revert_file("m.py")
    assert out == {"ok": False, "path": "m.py",
                   "error": f"cannot put it back: {repo / 'm.py'} is not a regular file"}
    assert stat.S_ISFIFO(os.lstat(repo / "m.py").st_mode)


def test_a_named_pipe_that_was_there_at_the_start_is_deleted_without_waiting(repo):
    (repo / "m.py").unlink()
    os.mkfifo(repo / "m.py")
    edit.reset_patch_round()
    assert edit.delete_file("m.py")["ok"] is True
    assert not os.path.lexists(repo / "m.py")


def test_a_file_too_large_to_read_is_not_edited(repo, monkeypatch):
    monkeypatch.setattr(tmpfile, "READ_MAX_BYTE", 4)
    out = edit.replace_in_file("m.py", "old", "new")
    assert out["error"] == f"cannot edit m.py: {repo / 'm.py'} is 8 bytes, over the 4 " \
                           f"read of one file"
    assert (repo / "m.py").read_text() == "old\nctx\n"


def test_a_dry_run_keeps_what_the_tools_would_have_written(repo, monkeypatch):
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    edit.replace_in_file("m.py", "old", "new")
    edit.delete_file("m.py")
    change_list = edit.dry_change_list()
    assert "-old\n+new\n" in change_list[0] and change_list[1] == "deleted m.py\n"
    assert (repo / "m.py").read_text() == "old\nctx\n"
    edit.reset_patch_round()
    assert edit.dry_change_list() == []


def test_revert_file_reads_a_path_as_itself_and_not_as_a_pattern(repo):
    """"*.py" names no file; as a pattern it would take back every .py file."""
    edit.replace_in_file("m.py", "old", "new")
    out = edit.revert_file("*.py")
    assert out["ok"] is False and "does not exist" in out["error"]
    assert (repo / "m.py").read_text() == "new\nctx\n"
    assert edit.write_path_set() == {"m.py"}


# == Enforcing the patch rounds ==========
def _patch_then_test(old: str, new: str, repo) -> dict:
    """Run one round (edit then run_test) and return replace_in_file's result."""
    r = edit.replace_in_file("m.py", old, new)
    edit.run_test()
    return r


def test_replace_in_file_stops_writing_once_the_limit_is_reached(repo,
                                                                 fake_test_command,
                                                                 monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 3)
    fake_test_command(0, "1 passed")

    for old, new in (("old", "v1"), ("v1", "v2"), ("v2", "v3")):
        assert _patch_then_test(old, new, repo)["ok"] is True
    assert edit.patch_round_count() == 3

    r = edit.replace_in_file("m.py", "v3", "v4")
    assert r["ok"] is False
    assert r["rounds"] == 3
    assert "limit" in r["error"]
    # Nothing was written; the 4th replacement never landed, so it is still v3
    assert (repo / "m.py").read_text() == "v3\nctx\n"


def test_consecutive_edits_count_as_a_single_round(repo, fake_test_command,
                                                   monkeypatch):
    """Patching 3 files and testing once is one round (the instructions' Editing steps)."""
    (repo / "n.py").write_text("old\nctx\n")
    _git(repo, "add", "n.py")
    _git(repo, "commit", "-q", "-m", "n")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 3)
    fake_test_command(0, "1 passed")

    assert edit.replace_in_file("m.py", "old", "a")["ok"] is True
    assert edit.replace_in_file("n.py", "old", "a")["ok"] is True
    assert edit.patch_round_count() == 0
    edit.run_test()
    assert edit.patch_round_count() == 1


def test_run_tests_without_a_patch_does_not_increase_the_rounds(repo,
                                                                fake_test_command,
                                                                monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(0, "1 passed")
    for _ in range(5):
        edit.run_test()
    assert edit.patch_round_count() == 0


def test_an_edit_that_did_not_land_does_not_increase_the_rounds(repo,
                                                                fake_test_command,
                                                                monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(0, "1 passed")
    assert edit.replace_in_file("m.py", "nosuchline", "a")["ok"] is False
    edit.run_test()
    assert edit.patch_round_count() == 0


def test_a_test_command_that_is_not_there_does_not_spend_a_round(repo, monkeypatch):
    """A round is "an edit, then the tests". A command that could not be started ran no
    test and is not counted, so a repository whose build tool is not installed - mvn,
    gradle, npm and ctest are not on an ordinary machine - keeps its rounds.
    """
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 3)
    monkeypatch.setattr(config, "TEST_COMMAND", ["no-such-build-tool-here"])

    for i in range(5):
        assert edit.replace_in_file("m.py", f"v{i}" if i else "old", f"v{i + 1}")["ok"]
        result = edit.run_test()
        assert result["status"] == "did not run"
    # No round closed, so the budget is untouched and writing still goes through
    assert edit.patch_round_count() == 0
    assert edit.replace_in_file("m.py", "v5", "v6")["ok"] is True


def test_a_test_run_that_timed_out_spends_a_round(repo, monkeypatch):
    """A run that started and spent the whole timeout is counted as a round.
    """
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_TIMEOUT_SEC", 0.5)
    monkeypatch.setattr(config, "TEST_COMMAND",
                        [sys.executable, "-c", "import time; time.sleep(30)"])

    assert edit.replace_in_file("m.py", "old", "new")["ok"] is True
    assert edit.run_test()["status"] == "did not run"
    assert edit.patch_round_count() == 1


def test_dry_run_does_not_increase_the_rounds(repo, fake_test_command, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    fake_test_command(0, "1 passed")
    assert edit.replace_in_file("m.py", "old", "new")["ok"] is True
    edit.run_test()
    assert edit.patch_round_count() == 0


def test_a_run_tests_refused_by_validation_does_not_increase_the_rounds(repo,
                                                                        monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 3)
    assert edit.replace_in_file("m.py", "old", "new")["ok"] is True
    # There is no tests/, so the TEST_PATH gate refuses it and no result is received
    assert edit.run_test()["status"] == "did not run"
    assert edit.patch_round_count() == 0


def test_reset_patch_rounds_restores_the_budget(repo, fake_test_command, monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 1)
    fake_test_command(0, "1 passed")
    _patch_then_test("old", "v1", repo)
    assert edit.replace_in_file("m.py", "v1", "v2")["ok"] is False
    edit.reset_patch_round()
    assert edit.patch_round_count() == 0
    assert edit.replace_in_file("m.py", "v1", "v2")["ok"] is True


def test_replace_lines_also_stops_writing_at_the_limit(lines_repo, fake_test_command,
                                                       monkeypatch):
    """Adding a tool must not change how MAX_PATCH_ROUND behaves."""
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 1)
    fake_test_command(0, "1 passed")
    assert edit.replace_line_range("m.py", 1, 1, "v1")["ok"] is True
    edit.run_test()

    r = edit.replace_line_range("m.py", 1, 1, "v2")
    assert r["ok"] is False and "limit" in r["error"]
    assert (lines_repo / "m.py").read_text() == "v1\nl2\nl3\nl4\nl5\n"


def test_replace_lines_and_replace_in_file_share_the_round_count(lines_repo,
                                                                 fake_test_command,
                                                                 monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    fake_test_command(0, "1 passed")
    assert edit.replace_line_range("m.py", 1, 1, "v1")["ok"] is True
    assert edit.replace_in_file("m.py", "l2", "v2")["ok"] is True
    assert edit.patch_round_count() == 0
    edit.run_test()
    assert edit.patch_round_count() == 1


def test_max_patch_rounds_of_0_enforces_nothing(repo, fake_test_command, monkeypatch):
    """MAX_PATCH_ROUND=0 refuses no edit, however many rounds have run."""
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 0)
    fake_test_command(0, "1 passed")
    for old, new in (("old", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v4")):
        assert _patch_then_test(old, new, repo)["ok"] is True
    assert edit.patch_round_count() == 4


# == Counting patch outcomes ==
def test_patch_stats_counts_the_outcomes(repo):
    edit.replace_in_file("m.py", "old", "a")
    edit.replace_in_file("m.py", "no such line", "x")
    assert edit.patch_count_dict() == {"ok": 1, "ng": 1, "ng_after_ok": 1}


def test_ng_after_ok_does_not_count_failures_before_the_first_success(repo):
    """It measures only "failed part-way through a file-by-file patch"."""
    edit.replace_in_file("m.py", "no such line", "x")
    assert edit.patch_count_dict() == {"ok": 0, "ng": 1, "ng_after_ok": 0}


def test_reset_patch_rounds_also_resets_the_outcome_counters(repo):
    edit.replace_in_file("m.py", "old", "a")
    edit.reset_patch_round()
    assert edit.patch_count_dict() == {"ok": 0, "ng": 0, "ng_after_ok": 0}


# == Filling the package cache again ==========================

def _asked_to_refill(monkeypatch, ok=True, skip=False):
    """Stand in for broker.refill() and return the list it records each ask in."""
    asks = []

    def refill():
        asks.append(1)
        return {"ok": ok, "refills_left": 1, "error": "no broker", "skip": skip}

    monkeypatch.setattr(edit.broker, "refill", refill)
    return asks


def test_a_changed_manifest_is_what_asks_for_the_cache(repo, monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    asks = _asked_to_refill(monkeypatch)
    edit.reset_patch_round()
    edit.create_file("pyproject.toml", "[project]\nname = 'p'\n")
    assert edit.change_dep_path_list() == ["pyproject.toml"]
    edit.refill_cache_on_dep_change()
    assert len(asks) == 1


def test_a_manifest_at_any_depth_is_found(repo, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    edit.create_file("backend/pyproject.toml", "[project]\n")
    assert edit.change_dep_path_list() == ["backend/pyproject.toml"]


def test_a_manifest_named_by_a_pattern_is_found_and_its_sources_are_not(repo,
                                                                       monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "cs")
    edit.create_file("src/App/App.csproj", "<Project />\n")
    edit.create_file("src/App/App.cs", "class App { }\n")
    edit.create_file("Directory.Build.props", "<Project />\n")
    assert edit.change_dep_path_list() == ["Directory.Build.props", "src/App/App.csproj"]


@pytest.mark.parametrize("path, pattern_tuple, want", [
    ("src/App/App.csproj", ("*.csproj",), True),
    ("App.csproj", ("*.csproj",), True),
    ("src/App/App.cs", ("*.csproj",), False),
    ("src/App/App.csproj.bak", ("*.csproj",), False),
    ("backend/pyproject.toml", ("pyproject.toml",), True),
    ("backend/my-pyproject.toml", ("pyproject.toml",), False),
    ("NuGet.config", ("NuGet.Config", "nuget.config"), False),
    ("nuget.config", ("NuGet.Config", "nuget.config"), True),
])
def test_a_dependency_file_is_told_by_its_base_name(path, pattern_tuple, want):
    assert is_dep_file(path, pattern_tuple) is want


def test_a_change_that_touches_no_manifest_asks_for_nothing(repo, monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    asks = _asked_to_refill(monkeypatch)
    edit.reset_patch_round()
    edit.replace_in_file("m.py", "old", "new")
    assert edit.change_dep_path_list() == []
    edit.refill_cache_on_dep_change()
    assert asks == []


def test_a_cache_that_already_holds_it_says_nothing(repo, monkeypatch):
    """Whether the cache holds it is the broker's to tell, and then nothing was done."""
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    _asked_to_refill(monkeypatch, skip=True)
    edit.reset_patch_round()
    edit.create_file("pyproject.toml", "[project]\n")
    assert edit.refill_cache_on_dep_change() == ""


def test_a_run_with_a_network_of_its_own_asks_for_nothing(repo, monkeypatch):
    """LLM_BROKER empty means the toolchain can fetch for itself."""
    monkeypatch.setattr(config, "LLM_BROKER", "")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    asks = _asked_to_refill(monkeypatch)
    edit.reset_patch_round()
    edit.create_file("pyproject.toml", "[project]\n")
    edit.refill_cache_on_dep_change()
    assert asks == []


def test_a_language_with_no_manifest_never_asks(repo, monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(config, "PRIMARY_EXT", "c")
    asks = _asked_to_refill(monkeypatch)
    edit.reset_patch_round()
    edit.create_file("pyproject.toml", "[project]\n")
    assert edit.change_dep_path_list() == []
    edit.refill_cache_on_dep_change()
    assert asks == []


def test_a_refusal_leaves_the_tests_to_run_against_the_cache_as_it_is(repo,
                                                                     monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    _asked_to_refill(monkeypatch, ok=False)
    edit.reset_patch_round()
    edit.create_file("pyproject.toml", "[project]\n")
    note = edit.refill_cache_on_dep_change()  # returns rather than raising
    assert "was not filled again for pyproject.toml (no broker)" in note


def test_a_fill_that_did_not_happen_is_asked_for_again(repo, monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    asks = _asked_to_refill(monkeypatch, ok=False)
    edit.reset_patch_round()
    edit.create_file("pyproject.toml", "[project]\n")
    edit.refill_cache_on_dep_change()
    edit.refill_cache_on_dep_change()
    assert len(asks) == 2


def test_what_came_of_the_fill_is_in_front_of_the_test_output(repo, fake_test_command,
                                                             monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    _asked_to_refill(monkeypatch)
    edit.reset_patch_round()
    edit.create_file("pyproject.toml", "[project]\n")
    fake_test_command(1, "FAILED tests/t.py::t_x - ImportError\n1 passed")
    r = edit.run_test()
    head = "note: the package cache was filled again for pyproject.toml, 1 refill(s) left this run\n"
    assert r["output"].startswith(head)
    assert r["failed_excerpt"].startswith(head + "FAILED tests/t.py::t_x")
    # The broker now finds the cache holding it
    _asked_to_refill(monkeypatch, skip=True)
    fake_test_command(0, "2 passed")
    r = edit.run_test()
    assert r["failed_excerpt"] == "" and "note:" not in r["output"]


# == no_tool_path_set ==========================================
def test_a_change_made_through_the_tools_is_not_untooled(repo):
    edit.replace_in_file("m.py", "old", "new")
    edit.create_file("sub/fresh.py", "x = 1")
    assert edit.no_tool_path_set() == set()


def test_a_write_that_did_not_go_through_a_tool_is_named(repo):
    """The model's own open() lands in the repository without any tool knowing."""
    (repo / "m.py").write_text("written by hand\n")
    assert edit.no_tool_path_set() == {"m.py"}


def test_a_file_created_without_a_tool_is_named_too(repo):
    (repo / "sneaked.py").write_text("x = 1\n")
    assert edit.no_tool_path_set() == {"sneaked.py"}


def test_what_was_already_changed_before_the_request_is_not_this_run_s(repo):
    (repo / "m.py").write_text("changed by whoever ran this\n")
    edit.reset_patch_round()
    assert edit.no_tool_path_set() == set()
    (repo / "other.py").write_text("x = 1\n")
    assert edit.no_tool_path_set() == {"other.py"}


def test_a_second_write_to_a_path_a_tool_wrote_is_named(repo):
    """The path is in both records, so only the contents tell the two writes apart."""
    edit.replace_in_file("m.py", "old", "new")
    assert edit.no_tool_path_set() == set()
    (repo / "m.py").write_text("written by hand after the tool\n")
    assert edit.no_tool_path_set() == {"m.py"}


def test_a_write_over_what_was_already_changed_is_named(repo):
    (repo / "m.py").write_text("changed by whoever ran this\n")
    edit.reset_patch_round()
    (repo / "m.py").write_text("and then written by hand\n")
    assert edit.no_tool_path_set() == {"m.py"}


def test_a_reverted_write_is_not_untooled(repo):
    edit.replace_in_file("m.py", "old", "new")
    assert edit.revert_file("m.py")["ok"] is True
    assert edit.no_tool_path_set() == set()


def test_a_dry_run_that_wrote_anyway_is_all_untooled(repo, monkeypatch):
    """The tools write nothing under a dry run, so what git holds went around them."""
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    edit.replace_in_file("m.py", "old", "new")
    (repo / "m.py").write_text("written by hand\n")
    assert edit.write_path_set() == set()
    assert edit.no_tool_path_set() == {"m.py"}


def test_untooled_paths_answers_empty_when_git_cannot_be_run(repo, monkeypatch):
    def no_git(*args, **kwargs):
        raise OSError("git is not here")

    monkeypatch.setattr(edit.gitcmd, "run", no_git)
    assert edit.no_tool_path_set() == set()


# == the environment a started program is given ==============
def test_the_test_command_is_not_given_the_credentials(repo, monkeypatch):
    """A repository's own test command must not be handed the key."""
    (repo / "tests").mkdir()
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    seen = {}

    start = subprocess.Popen

    def capture(argv, **kwargs):
        seen.update(kwargs.get("env") or {})
        return start([sys.executable, "-c", "print('1 passed')"], **kwargs)

    monkeypatch.setattr(edit.subprocess, "Popen", capture)
    edit.run_test()
    assert "LLM_API_KEY" not in seen
    assert "PATH" in seen


# == the tests a run changed ================================
def _changed(monkeypatch, added=(), modified=(), deleted=()):
    """Report what git says this run changed."""
    monkeypatch.setattr(edit, "change_by_kind_dict",
                        lambda: {"added": list(added), "modified": list(modified),
                                 "deleted": list(deleted)})


def test_a_test_file_the_run_changed_is_counted(monkeypatch):
    """A run can turn a red test green by changing the test, and the patch alone does
    not say which of the files in it are the tests."""
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    _changed(monkeypatch, modified=["tests/test_a.py", "src/a.py"],
             added=["tests/test_b.py"])
    assert edit.test_change_set() == {"tests/test_a.py", "tests/test_b.py"}


def test_a_deleted_test_is_counted_too(monkeypatch):
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    _changed(monkeypatch, deleted=["tests/test_a.py"])
    assert edit.test_change_set() == {"tests/test_a.py"}


def test_a_run_that_left_the_tests_alone_counts_none(monkeypatch):
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    _changed(monkeypatch, modified=["src/a.py", "testsuite/notes.md"])
    assert edit.test_change_set() == set()


def test_a_repository_whose_tests_are_reached_another_way_counts_none(monkeypatch):
    """TEST_PATH is what says where they are, so a language that has no such path
    has nothing to count."""
    monkeypatch.setattr(config, "TEST_PATH", "")
    _changed(monkeypatch, modified=["a.test.js"])
    assert edit.test_change_set() == set()


def test_git_failing_does_not_end_the_run(monkeypatch, caplog):
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(edit, "change_by_kind_dict",
                        lambda: (_ for _ in ()).throw(OSError("no git")))
    assert edit.test_change_set() == set()
    assert "could not be listed" in caplog.text


# == own_change_set: what this request changed ================
def test_what_the_working_tree_already_held_is_not_this_request_s_change(repo):
    """The merging stage starts on the union already placed in the working tree, and a
    test file a branch changed is not recorded as one the merge changed."""
    (repo / "tests").mkdir()
    (repo / "tests" / "test_m.py").write_text("brought in by a branch\n")
    (repo / "m.py").write_text("also brought in\n")
    edit.reset_patch_round()
    assert edit.own_change_set() == set()
    assert edit.test_change_set() == set()
    (repo / "tests" / "test_m.py").write_text("settled by this run\n")
    assert edit.own_change_set() == {"tests/test_m.py"}
    assert edit.test_change_set() == {"tests/test_m.py"}


def test_a_clean_start_counts_every_change_as_before(repo):
    edit.replace_in_file("m.py", "old", "new")
    (repo / "by_hand.py").write_text("x = 1\n")
    assert edit.own_change_set() == {"m.py", "by_hand.py"}
    assert edit.no_tool_path_set() == {"by_hand.py"}


# == the round limit is the request's ==========================
def test_a_request_under_no_round_limit_keeps_writing(repo, fake_test_command,
                                                      monkeypatch):
    """The merging stage's instructions state no limit, so none is enforced there
    whatever MAX_PATCH_ROUND says for the editing agent."""
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 1)
    fake_test_command(0, "1 passed")
    edit.reset_patch_round(round_limit=0)
    _patch_then_test("old", "v1", repo)
    assert edit.replace_in_file("m.py", "v1", "v2")["ok"] is True


def test_a_request_given_no_limit_follows_the_setting(repo, fake_test_command,
                                                      monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 1)
    fake_test_command(0, "1 passed")
    edit.reset_patch_round()
    _patch_then_test("old", "v1", repo)
    assert edit.replace_in_file("m.py", "v1", "v2")["ok"] is False


# == What the operating system refuses =========================
def test_a_file_that_cannot_be_read_is_an_error_not_an_exception(repo):
    (repo / "m.py").chmod(0o000)
    try:
        out = edit.replace_in_file("m.py", "old", "new")
    finally:
        (repo / "m.py").chmod(0o644)
    assert out["ok"] is False and "cannot read" in out["error"]


def test_a_file_that_cannot_be_written_is_an_error_not_an_exception(repo):
    (repo / "m.py").chmod(0o444)
    try:
        out = edit.replace_in_file("m.py", "old", "new")
    finally:
        (repo / "m.py").chmod(0o644)
    assert out["ok"] is False and "cannot write" in out["error"]
    assert (repo / "m.py").read_text() == "old\nctx\n"


def test_a_first_line_that_is_found_is_named_when_the_rest_is_not(repo):
    out = edit.replace_in_file("m.py", "old\nWRONG\n", "new\n")
    assert out["ok"] is False
    assert "line 1 'old' is found; the lines after it disagree" in out["error"]


def test_create_file_is_refused_once_the_rounds_are_used_up(repo, monkeypatch):
    monkeypatch.setattr(config, "MAX_PATCH_ROUND", 1)
    monkeypatch.setattr(edit, "_patch_round_count", 1)
    out = edit.create_file("new.py", "x = 1\n")
    assert out["ok"] is False and "hit their limit" in out["error"]
    assert not (repo / "new.py").exists()


def test_a_parent_directory_that_cannot_be_made_is_an_error(repo):
    (repo / "d").write_text("a file where the directory goes\n")
    out = edit.create_file("d/new.py", "x = 1\n")
    assert out["ok"] is False and "cannot create directory" in out["error"]


def test_an_untracked_file_that_cannot_be_deleted_is_an_error(repo):
    (repo / "sub").mkdir()
    (repo / "sub" / "u.py").write_text("x\n")
    (repo / "sub").chmod(0o555)
    try:
        out = edit.revert_file("sub/u.py")
    finally:
        (repo / "sub").chmod(0o755)
    assert out["ok"] is False and "cannot delete" in out["error"]


# == what git carries out of the run ========================================

def test_a_write_to_a_path_git_ignores_is_refused(repo):
    """The patch is built from what git reports, so a file written there is left
    behind without anything saying so."""
    (repo / ".gitignore").write_text("build/\n")
    _git(repo, "add", "."), _git(repo, "commit", "-q", "-m", "ignore")
    out = edit.create_file("build/cache.py", "x = 1\n")
    assert out["ok"] is False and "git ignores" in out["error"]
    assert not (repo / "build" / "cache.py").exists()


def test_a_write_inside_the_git_directory_is_refused(repo):
    """git carries nothing from inside its own directory, and what is written there
    acts on the run."""
    out = edit.create_file(".git/hooks/pre-commit", "#!/bin/sh\n")
    assert out["ok"] is False and "git directory" in out["error"]
    assert not (repo / ".git" / "hooks" / "pre-commit").exists()


def test_reading_inside_the_git_directory_is_refused(repo):
    from codetwine_loom.tools import source
    assert source.read_source_file(".git/config").startswith("Error: ")


# == the line endings a file already uses ===================================

def test_a_one_line_change_to_a_crlf_file_stays_one_line(repo):
    (repo / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    _git(repo, "add", "."), _git(repo, "commit", "-q", "-m", "crlf")
    out = edit.replace_in_file("crlf.py", "b = 2", "b = 20")
    assert out["ok"] is True
    assert (repo / "crlf.py").read_bytes() == b"a = 1\r\nb = 20\r\nc = 3\r\n"
    stat = _git(repo, "diff", "--numstat").split("\t")
    assert stat[0] == "1" and stat[1] == "1"


def test_a_line_range_change_to_a_crlf_file_keeps_the_endings(repo):
    (repo / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    out = edit.replace_line_range("crlf.py", 2, 2, "b = 20")
    assert out["ok"] is True
    assert (repo / "crlf.py").read_bytes() == b"a = 1\r\nb = 20\r\n"


def test_a_file_no_encoding_decodes_is_refused_with_a_dict(repo):
    """The reading tools answer with a string; the editing tools have to answer with
    the same keys as any other failure rather than raising."""
    raw = b"x = 1\n" + bytes(range(1, 256)) * 50
    (repo / "blob.py").write_bytes(raw)
    out = edit.replace_in_file("blob.py", "x = 1", "x = 2")
    assert out["ok"] is False and "SOURCE_ENCODING" in out["error"]
    assert (repo / "blob.py").read_bytes() == raw


# == Files that are not UTF-8 ==================================
# 表 and 能 end in 0x5c in Shift_JIS; 髙 is 0xfbfc as Windows writes it, which Python's
# cp932 encoder writes as 0xeee0
_SJIS_TEXT = "# 表の能力\ndef add(a, b):\n    return a + b\n# 髙橋\n"
_SJIS_RAW = _SJIS_TEXT.replace("髙", "\x00").encode("cp932").replace(b"\x00", b"\xfb\xfc")


@pytest.fixture
def encoding_repo(repo, monkeypatch):
    """The repo, with SOURCE_ENCODING naming the encodings its files are stored in."""
    monkeypatch.setattr(file_utils, "SOURCE_ENCODING", ["euc_jp", "cp932"])
    return repo


def test_an_edit_keeps_the_encoding_and_every_byte_it_did_not_touch(encoding_repo):
    (encoding_repo / "sj.py").write_bytes(_SJIS_RAW)
    out = edit.replace_in_file("sj.py", "a + b", "a + b + 0  # 足す")
    assert out["ok"] is True, out
    assert (encoding_repo / "sj.py").read_bytes() == _SJIS_RAW.replace(
        b"a + b\n", "a + b + 0  # 足す\n".encode("cp932"))


def test_a_line_range_edit_keeps_the_encoding_too(encoding_repo):
    text = _SJIS_TEXT.replace("髙", "高")
    (encoding_repo / "eu.py").write_bytes(text.encode("euc_jp"))
    assert edit.replace_line_range("eu.py", 1, 1, "# 表示")["ok"] is True
    assert (encoding_repo / "eu.py").read_bytes() == \
        text.replace("# 表の能力", "# 表示").encode("euc_jp")


def test_the_line_endings_of_the_lines_not_touched_stay_as_they_are(encoding_repo):
    raw = "能\r\nb\nc\r\n".encode("cp932")
    (encoding_repo / "mixed.py").write_bytes(raw)
    assert edit.replace_in_file("mixed.py", "c", "c\n表")["ok"] is True
    assert (encoding_repo / "mixed.py").read_bytes() == \
        "能\r\nb\nc\r\n表\r\n".encode("cp932")


def test_a_bom_stays_in_front(repo):
    (repo / "bom.py").write_bytes(codecs.BOM_UTF8 + "x = 1\n# 表\n".encode())
    assert edit.replace_in_file("bom.py", "x = 1", "x = 2")["ok"] is True
    assert (repo / "bom.py").read_bytes() == codecs.BOM_UTF8 + "x = 2\n# 表\n".encode()


@pytest.mark.parametrize("codec", ["utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"])
def test_a_utf_16_or_32_file_keeps_its_bom_and_byte_order(repo, codec):
    bom = {"utf-16-le": codecs.BOM_UTF16_LE, "utf-16-be": codecs.BOM_UTF16_BE,
           "utf-32-le": codecs.BOM_UTF32_LE, "utf-32-be": codecs.BOM_UTF32_BE}[codec]
    (repo / "w.py").write_bytes(bom + "x = 1\r\n# 表\r\n".encode(codec))
    assert source.read_source_file("w.py") == "x = 1\n# 表\n"
    assert edit.replace_in_file("w.py", "x = 1", "x = 2")["ok"] is True
    assert (repo / "w.py").read_bytes() == bom + "x = 2\r\n# 表\r\n".encode(codec)


def test_a_character_the_encoding_cannot_write_is_refused(encoding_repo, monkeypatch):
    (encoding_repo / "sj.py").write_bytes(_SJIS_RAW)
    out = edit.replace_in_file("sj.py", "a + b", "a + b  # 🙂")
    assert out["ok"] is False and "'🙂'" in out["error"] and "line 3" in out["error"]
    assert (encoding_repo / "sj.py").read_bytes() == _SJIS_RAW
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    assert edit.replace_in_file("sj.py", "a + b", "a + b  # 🙂")["ok"] is False


def test_a_new_file_is_stored_as_the_files_with_its_extension_are(encoding_repo):
    (encoding_repo / "sj.py").write_bytes(_SJIS_RAW)
    _git(encoding_repo, "add", "sj.py")
    assert edit.create_file("pkg/new.py", "# 表\n")["ok"] is True
    assert (encoding_repo / "pkg" / "new.py").read_bytes() == "# 表\n".encode("cp932")
    # Another extension says nothing about it
    assert edit.create_file("notes.md", "# 表\n")["ok"] is True
    assert (encoding_repo / "notes.md").read_bytes() == "# 表\n".encode()
    out = edit.create_file("pkg/other.py", "# 🙂\n")
    assert out["ok"] is False and "'🙂'" in out["error"]
    assert not (encoding_repo / "pkg" / "other.py").exists()


def test_a_new_file_is_stored_as_the_files_of_its_language_are(encoding_repo):
    """A new copybook follows the programs of the repository."""
    (encoding_repo / "MAIN.cbl").write_bytes("      * 顧客\n".encode("cp932"))
    _git(encoding_repo, "add", "MAIN.cbl")
    assert edit.create_file("copy/NEWREC.cpy", "      * 表\n")["ok"] is True
    assert (encoding_repo / "copy" / "NEWREC.cpy").read_bytes() == "      * 表\n".encode("cp932")


def test_a_new_file_is_utf_8_when_the_files_disagree(encoding_repo):
    (encoding_repo / "sj.py").write_bytes(_SJIS_RAW)
    (encoding_repo / "u8.py").write_text("# 表\n", encoding="utf-8")
    assert edit.create_file("new.py", "# 表\n")["ok"] is True
    assert (encoding_repo / "new.py").read_bytes() == "# 表\n".encode()


# == The static check ===========================================


def _check_printing(monkeypatch, tmp_path, text):
    """Point CHECK_COMMAND at a program that prints text, and run the real check."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [sys.executable, "-c",
                                                  f"print({text!r})"])
    return _RUN_CHECK()


def test_a_finding_that_moves_is_the_same_finding_and_summary_lines_are_not_findings(
        monkeypatch, tmp_path):
    first = _check_printing(monkeypatch, tmp_path,
                            "a.py:3:5: F821 Undefined name `f`\nFound 1 error.")
    moved = _check_printing(monkeypatch, tmp_path,
                            "a.py:40:1: F821 Undefined name `f`\nAll checks passed!")
    assert first == moved == {"a.py: F821 Undefined name `f`": 1}


def test_a_finding_between_the_first_and_the_last_bytes_returned_is_counted(
        monkeypatch, tmp_path):
    monkeypatch.setattr(edit, "_OUTPUT_HEAD_BYTE", 100)
    monkeypatch.setattr(edit, "_OUTPUT_TAIL_BYTE", 100)
    fill = "\n".join(f"checking file {n}" for n in range(40))
    found = _check_printing(
        monkeypatch, tmp_path,
        f"{fill}\na.py:3:5: F821 Undefined name `f`\n  b.py:9:1: E999 x\n{fill}\n"
        "Found 2 errors.")
    assert found == {"a.py: F821 Undefined name `f`": 1, "b.py: E999 x": 1}


def test_a_finding_the_check_colors_is_read_as_one_it_does_not(monkeypatch, tmp_path):
    """The line ruff prints under FORCE_COLOR."""
    colored = ("\x1b[1ma.py\x1b[0m\x1b[36m:\x1b[0m1\x1b[36m:\x1b[0m5\x1b[36m:\x1b[0m "
               "\x1b[1m\x1b[31mF821\x1b[0m Undefined name `f`\nFound 1 error.")
    assert _check_printing(monkeypatch, tmp_path, colored) == \
        {"a.py: F821 Undefined name `f`": 1}


def test_the_same_finding_twice_counts_twice(monkeypatch, tmp_path):
    found = _check_printing(monkeypatch, tmp_path,
                            "a.py:3:5: F821 Undefined name `f`\na.py:9:1: F821 Undefined name `f`")
    assert found == {"a.py: F821 Undefined name `f`": 2}


def test_no_command_or_a_missing_one_runs_no_check(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    assert _RUN_CHECK() is None
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", ["no-such-checker-here"])
    assert _RUN_CHECK() is None


def test_a_finding_in_parentheses_counts_and_its_line_and_column_are_left_out(
        monkeypatch, tmp_path):
    found = _check_printing(monkeypatch, tmp_path,
                            "src/a.ts(3,5): error TS2304: Cannot find name 'f'.\n"
                            "src/a.ts(9,1): error TS2304: Cannot find name 'f'.")
    assert found == {"src/a.ts: error TS2304: Cannot find name 'f'.": 2}


def test_a_path_under_the_repository_is_made_relative(monkeypatch, tmp_path):
    found = _check_printing(monkeypatch, tmp_path,
                            f"{tmp_path}/src/a.c:4:3: error: 'f' undeclared\n"
                            "/usr/include/stdio.h:1:1: error: x")
    assert found == {"src/a.c: error: 'f' undeclared": 1, "/usr/include/stdio.h: error: x": 1}


def test_the_lines_the_r_check_prints_are_read_as_findings(monkeypatch, tmp_path):
    found = _check_printing(
        monkeypatch, tmp_path,
        "R/broken.R:3:1: unexpected '}'\n"
        "R/cli.R:5: no visible global function definition for 'missing_helper'")
    assert found == {
        "R/broken.R: unexpected '}'": 1,
        "R/cli.R: no visible global function definition for 'missing_helper'": 1}


def test_a_check_that_cannot_check_the_tree_runs_no_check(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [
        sys.executable, "-c", "print('a.js:1:1: x'); raise SystemExit(127)"])
    assert _RUN_CHECK() is None


@pytest.mark.parametrize("body", [
    "import sys; print('error: unexpected argument --no-such-flag', file=sys.stderr); "
    "sys.exit(2)",
    "raise RuntimeError('the checker broke')"])
def test_a_check_that_fails_without_a_finding_runs_no_check(monkeypatch, tmp_path, caplog,
                                                            body):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [sys.executable, "-c", body])
    assert _RUN_CHECK() is None
    assert "without a finding that can be read, so the static check is not counted" in \
        caplog.text


def test_a_check_that_fails_with_findings_or_ends_clean_without_any_is_counted(
        monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [
        sys.executable, "-c", "print('a.py:1:1: F821 Undefined name `f`'); "
                              "raise SystemExit(1)"])
    assert _RUN_CHECK() == {"a.py: F821 Undefined name `f`": 1}
    monkeypatch.setattr(config, "CHECK_COMMAND", [sys.executable, "-c", "print('ok')"])
    count = _RUN_CHECK()
    assert count is not None and count == {}
    assert "is not counted" not in caplog.text


def test_each_language_with_a_compile_free_test_command_has_a_default_check():
    has_check_set = {ALIAS_DICT.get(ext, ext) for ext, lang_config in LANG_DICT.items()
                     if lang_config.check_command_tuple}
    assert has_check_set == {"py", "js", "ts", "c", "cpp", "rs", "cbl", "cs", "r"}


def test_the_python_default_finds_a_name_that_is_not_defined(monkeypatch, tmp_path):
    (tmp_path / "a.py").write_text("def g():\n    return f()\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", list(LANG_DICT["py"].check_command_tuple))
    assert _RUN_CHECK() == {"a.py: F821 Undefined name `f`": 1}


# == The R default check (sandbox/check-r.R) ===============
@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is not installed")
def test_the_r_check_reads_a_call_with_an_empty_argument(tmp_path):
    (tmp_path / "a.R").write_text(
        'x <- data.frame(a = 1)\ny <- x[, "a"]\nif (TRUE) { z <- x[1, ] }\nmissing_fn(y)\n')
    proc = subprocess.run(
        ["Rscript", str(pathlib.Path(__file__).resolve().parent.parent / "sandbox" / "check-r.R")],
        cwd=tmp_path, capture_output=True, text=True, check=False)
    assert "Error" not in proc.stderr
    assert proc.stdout.strip() == "a.R:4: no visible global function definition for 'missing_fn'"


# == The Rust default check (sandbox/check-rs.sh) ===============
_CHECK_RS = pathlib.Path(__file__).resolve().parent.parent / "sandbox" / "check-rs.sh"


@pytest.fixture
def cargo_project(tmp_path, monkeypatch):
    """A Cargo project and a stand-in cargo on PATH that prints $CARGO_OUT and exits
    with $CARGO_STATUS."""
    (tmp_path / "Cargo.toml").write_text("[package]\nname = \"calc\"\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "cargo").write_text('#!/bin/sh\nprintf "%s\\n" "$CARGO_OUT" >&2\n'
                                   'exit "$CARGO_STATUS"\n')
    (bin_dir / "cargo").chmod(0o755)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [str(_CHECK_RS)])
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return tmp_path


def test_the_rust_check_counts_the_errors_and_not_the_warnings(cargo_project, monkeypatch):
    monkeypatch.setenv("CARGO_STATUS", "101")
    monkeypatch.setenv("CARGO_OUT", "\n".join([
        "    Checking calc v0.1.0 (/x)",
        "src/lib.rs:1:4: warning: function `unused` is never used",
        "src/lib.rs:6:9: error[E0425]: cannot find value `c` in this scope",
        "error: could not compile `calc` (lib) due to 1 previous error"]))
    assert _RUN_CHECK() == {"src/lib.rs: error[E0425]: cannot find value `c` in this scope": 1}


def test_the_rust_check_finds_nothing_in_a_tree_that_builds(cargo_project, monkeypatch):
    monkeypatch.setenv("CARGO_STATUS", "0")
    monkeypatch.setenv("CARGO_OUT", "src/lib.rs:1:4: warning: function `unused` is never used")
    assert _RUN_CHECK() == {}


def test_the_rust_check_does_not_run_where_cargo_fails_before_the_sources(cargo_project,
                                                                          monkeypatch):
    """A crate the package cache does not hold says nothing about the change."""
    monkeypatch.setenv("CARGO_STATUS", "101")
    monkeypatch.setenv("CARGO_OUT", "error: no matching package named `itoa` found")
    assert _RUN_CHECK() is None


def test_the_rust_check_does_not_run_without_cargo(cargo_project, monkeypatch):
    (cargo_project / "bin" / "cargo").unlink()
    assert _RUN_CHECK() is None


def test_the_rust_check_finds_nothing_without_a_manifest(cargo_project, monkeypatch):
    (cargo_project / "Cargo.toml").unlink()
    monkeypatch.setenv("CARGO_STATUS", "101")
    monkeypatch.setenv("CARGO_OUT", "src/lib.rs:6:9: error[E0425]: x")
    assert _RUN_CHECK() == {}


# == The C# default check (sandbox/check-cs.sh) ===============
_CHECK_CS = pathlib.Path(__file__).resolve().parent.parent / "sandbox" / "check-cs.sh"


@pytest.fixture
def dotnet_project(tmp_path, monkeypatch):
    """A solution and a stand-in dotnet on PATH that prints $DOTNET_OUT, exits with
    $DOTNET_STATUS and writes what it was asked to build to $tmp_path/argv."""
    (tmp_path / "Calc.slnx").write_text("<Solution />\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "dotnet").write_text(f'#!/bin/sh\necho "$2" > {tmp_path}/argv\n'
                                    'printf "%s\\n" "$DOTNET_OUT"\nexit "$DOTNET_STATUS"\n')
    (bin_dir / "dotnet").chmod(0o755)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [str(_CHECK_CS), "Calc.slnx"])
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return tmp_path


def test_the_csharp_check_counts_each_error_once_and_not_the_warnings(dotnet_project,
                                                                     monkeypatch):
    src = f"{dotnet_project}/src/Calc/Basic.cs"
    monkeypatch.setenv("DOTNET_STATUS", "1")
    monkeypatch.setenv("DOTNET_OUT", "\n".join([
        (f"{src}(3,5): warning CS0168: The variable 'x' is declared but never used "
         f"[{dotnet_project}/src/Calc/Calc.csproj::TargetFramework=net8.0]"),
        (f"{src}(11,32): error CS0103: The name 'c' does not exist in the current context "
         f"[{dotnet_project}/src/Calc/Calc.csproj::TargetFramework=net8.0]"),
        (f"{src}(11,32): error CS0103: The name 'c' does not exist in the current context "
         f"[{dotnet_project}/src/Calc/Calc.csproj::TargetFramework=net10.0]")]))
    assert _RUN_CHECK() == {
        "src/Calc/Basic.cs: error CS0103: The name 'c' does not exist in the current context": 1}
    assert (dotnet_project / "argv").read_text().strip() == "Calc.slnx"


def test_the_csharp_check_finds_nothing_in_a_tree_that_builds(dotnet_project, monkeypatch):
    monkeypatch.setenv("DOTNET_STATUS", "0")
    monkeypatch.setenv("DOTNET_OUT", "")
    assert _RUN_CHECK() == {}


def test_the_csharp_check_does_not_run_where_the_build_fails_before_the_sources(
        dotnet_project, monkeypatch):
    """A package the package cache does not hold says nothing about the change."""
    monkeypatch.setenv("DOTNET_STATUS", "1")
    monkeypatch.setenv("DOTNET_OUT", f"{dotnet_project}/Calc.csproj : error NU1301: "
                                     "Failed to retrieve information about 'Humanizer.Core'")
    assert _RUN_CHECK() is None


def test_the_csharp_check_does_not_run_without_dotnet(dotnet_project):
    (dotnet_project / "bin" / "dotnet").unlink()
    assert _RUN_CHECK() is None


def test_the_csharp_check_does_not_run_without_a_target(dotnet_project, monkeypatch):
    monkeypatch.setattr(config, "CHECK_COMMAND", [str(_CHECK_CS)])
    monkeypatch.setenv("DOTNET_STATUS", "0")
    monkeypatch.setenv("DOTNET_OUT", "")
    assert _RUN_CHECK() is None
    assert not (dotnet_project / "argv").exists()


# == The file a build tool is pointed at in the root ===========
@pytest.mark.parametrize("name_list, want", [
    (["Calc.slnx"], "Calc.slnx"),
    (["Calc.slnx", "Build.csproj"], "Calc.slnx"),
    (["App.sln", "App.Web.slnf"], "App.sln"),
    (["App.csproj"], "App.csproj"),
    (["A.sln", "B.slnx"], ""),
    (["A.csproj", "B.csproj"], ""),
    (["README.md"], ""),
    (["-App.sln"], "./-App.sln"),
    (["--help.csproj"], "./--help.csproj"),
])
def test_the_root_target_is_the_one_solution_else_the_one_project(tmp_path, name_list,
                                                                   want):
    for name in name_list:
        (tmp_path / name).write_text("x\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "Other.sln").write_text("x\n")
    assert edit._root_target(str(tmp_path), "cs") == want


def test_a_language_without_a_root_target_names_none(tmp_path):
    (tmp_path / "Calc.slnx").write_text("x\n")
    assert edit._root_target(str(tmp_path), "py") == ""
    assert edit._root_target(str(tmp_path), "zz") == ""


@pytest.fixture
def csharp_root(tmp_path, monkeypatch):
    """A C# repository root holding one solution and one project."""
    (tmp_path / "Calc.slnx").write_text("x\n")
    (tmp_path / "Build.csproj").write_text("x\n")
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PRIMARY_EXT", "cs")
    return tmp_path


def test_the_default_test_command_is_pointed_at_the_root_target(csharp_root, monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND", ["dotnet", "test"])
    assert edit._build_argv(["tests/x.cs"], "none", "tests", "cs")[0] == [
        "dotnet", "test", "Calc.slnx"]


def test_a_test_command_set_by_hand_is_left_alone(csharp_root, monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND", ["dotnet", "test", "src/App.sln"])
    assert edit._build_argv(None, "none", "tests", "cs")[0] == [
        "dotnet", "test", "src/App.sln"]


def test_the_default_check_is_pointed_at_the_root_target(csharp_root, monkeypatch):
    bin_dir = csharp_root / "bin"
    bin_dir.mkdir()
    (bin_dir / "ctloom-check-cs").write_text(f'#!/bin/sh\necho "$@" > {csharp_root}/argv\n')
    (bin_dir / "ctloom-check-cs").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setattr(config, "CHECK_COMMAND", ["ctloom-check-cs"])
    assert _RUN_CHECK() == {}
    assert (csharp_root / "argv").read_text().strip() == "Calc.slnx"


# == The JavaScript and TypeScript install (sandbox/install-js.sh) =====
_INSTALL_JS = pathlib.Path(__file__).resolve().parent.parent / "sandbox" / "install-js.sh"

# A stand-in package manager: it writes its name, its arguments and the yarn settings to
# $CALL_LOG, fails an install that holds to the lock file when $FAIL_LOCKED is 1, writes
# over the lock file in an install that does not (making one when $MAKE_LOCK is set)
# unless it is told to write none or to save nothing, and then fails it when $FAIL_UNLOCKED is 1, writes over
# package.json when $REWRITE_PACKAGE is set, and makes node_modules
_STAND_IN_MANAGER = """#!/bin/sh
tool=$(basename "$0")
if [ "$tool" = yarn ] && [ "$1" = --version ]; then echo "$YARN_VERSION"; exit 0; fi
echo "$tool $* linker=${YARN_NODE_LINKER:-} global=${YARN_ENABLE_GLOBAL_CACHE:-}\
 network=${YARN_ENABLE_NETWORK:-}" >> "$CALL_LOG"
case "$*" in
"install --frozen-lockfile"*|"install --immutable"*|ci*)
    [ "${FAIL_LOCKED:-0}" = 1 ] && exit 1 ;;
*) case "$*" in
   *--no-package-lock*|*--no-lockfile*|*--no-save*) ;;
   *) for lock in pnpm-lock.yaml yarn.lock package-lock.json bun.lock bun.lockb; do
          [ -f "$lock" ] && echo changed > "$lock"
      done ;;
   esac
   [ -n "${MAKE_LOCK:-}" ] && echo changed > "$MAKE_LOCK"
   [ "${FAIL_UNLOCKED:-0}" = 1 ] && exit 1 ;;
esac
[ -n "${REWRITE_PACKAGE:-}" ] && echo rewritten > package.json
mkdir -p node_modules
exit 0
"""


@pytest.fixture
def js_install_project(tmp_path, monkeypatch):
    """A package.json, stand-ins for node (it prints $NODE_OUT: the dependency count and
    the package manager packageManager names), npm, pnpm, yarn and bun on PATH, and the file
    the stand-ins write their calls to."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text('{"dependencies": {"a": "1"}}\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "node").write_text('#!/bin/sh\necho "${NODE_OUT:-1 }"\n')
    for tool in ("npm", "pnpm", "yarn", "bun"):
        (bin_dir / tool).write_text(_STAND_IN_MANAGER)
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/h")
    monkeypatch.setenv("CALL_LOG", str(tmp_path / "calls"))
    for name in ("npm_config_offline", "FAIL_LOCKED", "FAIL_UNLOCKED", "MAKE_LOCK",
                 "NODE_OUT", "REWRITE_PACKAGE", "CTLOOM_BASE"):
        monkeypatch.delenv(name, raising=False)
    return project


def _install_js(project: pathlib.Path) -> tuple[subprocess.CompletedProcess, list[str]]:
    """Run sandbox/install-js.sh in a project; return the process and the calls made."""
    proc = subprocess.run(["sh", str(_INSTALL_JS)], cwd=project, capture_output=True,
                          text=True, check=False)
    log = project.parent / "calls"
    call_list = log.read_text().splitlines() if log.exists() else []
    if log.exists():
        log.unlink()
    return proc, call_list


_PNPM_OPTION = "--store-dir /h/.local/share/pnpm/store --config.strict-dep-builds=false"


@pytest.mark.parametrize("lock_list, manager_field, yarn_version, offline, expect", [
    (["pnpm-lock.yaml"], "", "", False,
     f"pnpm install --frozen-lockfile {_PNPM_OPTION} linker= global= network="),
    (["pnpm-lock.yaml"], "", "", True,
     f"pnpm install --frozen-lockfile {_PNPM_OPTION} --offline linker= global= network="),
    (["yarn.lock"], "", "1.22.22", True,
     "yarn install --frozen-lockfile --non-interactive --offline linker= global= network="),
    (["yarn.lock"], "", "4.5.0", True,
     "yarn install --immutable linker=node-modules global=true network=0"),
    (["package-lock.json"], "", "", False,
     "npm ci --no-audit --no-fund linker= global= network="),
    (["package-lock.json", "yarn.lock"], "", "1.22.22", False,
     "npm ci --no-audit --no-fund linker= global= network="),
    (["pnpm-lock.yaml", "yarn.lock"], "", "1.22.22", False,
     f"pnpm install --frozen-lockfile {_PNPM_OPTION} linker= global= network="),
    (["package-lock.json", "pnpm-lock.yaml"], "pnpm", "", False,
     f"pnpm install --frozen-lockfile {_PNPM_OPTION} linker= global= network="),
    (["bun.lock"], "", "", False, "bun install --frozen-lockfile linker= global= network="),
    (["bun.lockb"], "", "", True,
     "bun install --frozen-lockfile --offline linker= global= network="),
    (["bun.lock", "bun.lockb"], "", "", False,
     "bun install --frozen-lockfile linker= global= network="),
    (["bunfig.toml"], "", "", False, "bun install --no-save linker= global= network="),
    (["package-lock.json"], "bun", "", True,
     "bun install --no-save --offline linker= global= network="),
    (["yarn.lock", "bun.lock"], "", "1.22.22", False,
     "yarn install --frozen-lockfile --non-interactive linker= global= network=")])
def test_the_packages_are_installed_by_the_tool_the_package_files_name(
        js_install_project, monkeypatch, lock_list, manager_field, yarn_version, offline,
        expect):
    """packageManager first; without it npm's lock file, then pnpm's, then yarn's, then
    bun's or bunfig.toml."""
    for lock in lock_list:
        (js_install_project / lock).write_text("lock\n")
    monkeypatch.setenv("NODE_OUT", f"1 {manager_field}")
    monkeypatch.setenv("YARN_VERSION", yarn_version)
    if offline:
        monkeypatch.setenv("npm_config_offline", "true")
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 0, proc.stderr
    assert call_list == [expect]
    assert (js_install_project / "node_modules" / ".ctloom-installed").exists()


@pytest.mark.parametrize("lock, yarn_version, second", [
    ("pnpm-lock.yaml", "", "pnpm install --no-frozen-lockfile"),
    ("yarn.lock", "1.22.22", "yarn install --no-lockfile --non-interactive"),
    ("yarn.lock", "4.5.0", "yarn install"),
    ("bun.lock", "", "bun install --no-save"),
    ("bun.lockb", "", "bun install --no-save")])
def test_an_install_the_lock_file_refuses_is_made_again_and_the_lock_file_kept(
        js_install_project, monkeypatch, lock, yarn_version, second):
    (js_install_project / lock).write_text("lock\n")
    monkeypatch.setenv("YARN_VERSION", yarn_version)
    monkeypatch.setenv("FAIL_LOCKED", "1")
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 0, proc.stderr
    assert len(call_list) == 2 and call_list[1].startswith(second + " ")
    assert (js_install_project / lock).read_text() == "lock\n"


def test_a_lock_file_the_second_install_makes_is_taken_away(js_install_project,
                                                            monkeypatch):
    """packageManager names pnpm, and the repository holds no pnpm-lock.yaml."""
    monkeypatch.setenv("NODE_OUT", "1 pnpm")
    monkeypatch.setenv("FAIL_LOCKED", "1")
    monkeypatch.setenv("MAKE_LOCK", "pnpm-lock.yaml")
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 0, proc.stderr
    assert len(call_list) == 2
    assert not (js_install_project / "pnpm-lock.yaml").exists()


def test_a_package_file_the_install_rewrites_is_put_back(js_install_project, monkeypatch):
    (js_install_project / "yarn.lock").write_text("lock\n")
    monkeypatch.setenv("YARN_VERSION", "4.5.0")
    monkeypatch.setenv("REWRITE_PACKAGE", "1")
    before = (js_install_project / "package.json").read_text()
    assert _install_js(js_install_project)[0].returncode == 0
    assert (js_install_project / "package.json").read_text() == before
    # The mark matches the files as they are, so nothing is installed again
    assert _install_js(js_install_project)[1] == []


def test_a_changed_pnpm_workspace_file_installs_again(js_install_project):
    (js_install_project / "pnpm-lock.yaml").write_text("lock\n")
    (js_install_project / "pnpm-workspace.yaml").write_text("catalog: {a: 1}\n")
    assert len(_install_js(js_install_project)[1]) == 1
    assert _install_js(js_install_project)[1] == []
    (js_install_project / "pnpm-workspace.yaml").write_text("catalog: {a: 2}\n")
    assert len(_install_js(js_install_project)[1]) == 1


def test_a_changed_bunfig_installs_again(js_install_project):
    (js_install_project / "bun.lock").write_text("lock\n")
    (js_install_project / "bunfig.toml").write_text("[install]\nexact = true\n")
    assert len(_install_js(js_install_project)[1]) == 1
    assert _install_js(js_install_project)[1] == []
    (js_install_project / "bunfig.toml").write_text("[install]\nexact = false\n")
    assert len(_install_js(js_install_project)[1]) == 1


def test_a_pnpm_workspace_root_with_no_dependency_of_its_own_installs(js_install_project,
                                                                      monkeypatch):
    monkeypatch.setenv("NODE_OUT", "0 ")
    (js_install_project / "pnpm-lock.yaml").write_text("lock\n")
    assert _install_js(js_install_project)[1] == []
    (js_install_project / "pnpm-workspace.yaml").write_text("packages: [packages/*]\n")
    assert len(_install_js(js_install_project)[1]) == 1


def test_an_install_stops_when_the_lock_file_s_tool_is_missing(js_install_project):
    (js_install_project / "pnpm-lock.yaml").write_text("lock\n")
    (js_install_project.parent / "bin" / "pnpm").unlink()
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 1 and call_list == []
    assert "ctloom-install-js: pnpm does not exist" in proc.stderr


def _commit_project(project: pathlib.Path) -> str:
    """Make the project a git repository holding its files in one commit; return it."""
    git = ["git", "-C", str(project), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "start"], check=True)
    return subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.mark.parametrize("lock, manager_field, yarn_version, relock", [
    ("package-lock.json", "", "", "npm install --no-audit --no-fund"),
    ("npm-shrinkwrap.json", "", "", "npm install --no-audit --no-fund"),
    ("pnpm-lock.yaml", "", "", f"pnpm install --no-frozen-lockfile {_PNPM_OPTION}"),
    ("yarn.lock", "", "1.22.22", "yarn install --non-interactive"),
    ("yarn.lock", "", "4.5.0", "yarn install"),
    ("bun.lock", "", "", "bun install"),
    ("bun.lockb", "", "", "bun install")])
def test_a_lock_file_a_changed_package_json_no_longer_matches_is_made_again_and_kept(
        js_install_project, monkeypatch, lock, manager_field, yarn_version, relock):
    """The install holding to the lock file fails after the run changed package.json."""
    (js_install_project / lock).write_text("lock\n")
    _commit_project(js_install_project)
    (js_install_project / "package.json").write_text('{"dependencies": {"a": "1", "b": "2"}}\n')
    monkeypatch.setenv("NODE_OUT", f"1 {manager_field}")
    monkeypatch.setenv("YARN_VERSION", yarn_version)
    monkeypatch.setenv("FAIL_LOCKED", "1")
    if lock == "npm-shrinkwrap.json":
        monkeypatch.setenv("MAKE_LOCK", lock)
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 0, proc.stderr
    assert len(call_list) == 2 and call_list[1].startswith(relock + " ")
    assert (js_install_project / lock).read_text() == "changed\n"
    # The mark is that of the lock file the install left, so nothing is installed again
    monkeypatch.delenv("FAIL_LOCKED")
    assert _install_js(js_install_project)[1] == []


def test_a_package_json_changed_since_the_run_s_start_commit_makes_the_lock_file_again(
        js_install_project, monkeypatch):
    """CTLOOM_BASE, not HEAD, is what package.json is compared against."""
    (js_install_project / "package-lock.json").write_text("lock\n")
    base = _commit_project(js_install_project)
    (js_install_project / "package.json").write_text('{"dependencies": {"b": "2"}}\n')
    _commit_project(js_install_project)
    monkeypatch.setenv("CTLOOM_BASE", base)
    monkeypatch.setenv("FAIL_LOCKED", "1")
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 0, proc.stderr
    assert call_list[1].startswith("npm install --no-audit --no-fund ")
    assert (js_install_project / "package-lock.json").read_text() == "changed\n"


def test_a_package_json_git_does_not_track_makes_the_lock_file_again(js_install_project,
                                                                      monkeypatch):
    (js_install_project / "package-lock.json").write_text("lock\n")
    _commit_project(js_install_project)
    (js_install_project / "packages" / "b").mkdir(parents=True)
    (js_install_project / "packages" / "b" / "package.json").write_text('{"name": "b"}\n')
    monkeypatch.setenv("FAIL_LOCKED", "1")
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 0, proc.stderr
    assert call_list[1].startswith("npm install --no-audit --no-fund ")
    assert (js_install_project / "package-lock.json").read_text() == "changed\n"


def test_a_lock_file_is_put_back_when_package_json_is_as_the_commit_holds_it(
        js_install_project, monkeypatch):
    (js_install_project / "package-lock.json").write_text("lock\n")
    _commit_project(js_install_project)
    monkeypatch.setenv("FAIL_LOCKED", "1")
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 0, proc.stderr
    assert call_list[1].startswith("npm install --no-save --no-package-lock ")
    assert (js_install_project / "package-lock.json").read_text() == "lock\n"


def test_a_lock_file_changed_since_the_commit_is_made_again_from_package_json(
        js_install_project, monkeypatch):
    """package.json is as the commit holds it again, and the lock file still is not."""
    (js_install_project / "package-lock.json").write_text("lock\n")
    _commit_project(js_install_project)
    (js_install_project / "package-lock.json").write_text("lock with b\n")
    monkeypatch.setenv("FAIL_LOCKED", "1")
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 0, proc.stderr
    assert call_list[1].startswith("npm install --no-audit --no-fund ")
    assert (js_install_project / "package-lock.json").read_text() == "changed\n"


def test_a_package_json_git_ignores_leaves_the_lock_file_as_it_was(js_install_project,
                                                                    monkeypatch):
    (js_install_project / "package-lock.json").write_text("lock\n")
    (js_install_project / ".gitignore").write_text("dist/\n")
    _commit_project(js_install_project)
    (js_install_project / "dist").mkdir()
    (js_install_project / "dist" / "package.json").write_text('{"name": "dist"}\n')
    monkeypatch.setenv("FAIL_LOCKED", "1")
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 0, proc.stderr
    assert call_list[1].startswith("npm install --no-save --no-package-lock ")
    assert (js_install_project / "package-lock.json").read_text() == "lock\n"


def test_a_lock_file_is_put_back_when_making_it_again_fails(js_install_project,
                                                            monkeypatch):
    (js_install_project / "pnpm-lock.yaml").write_text("lock\n")
    _commit_project(js_install_project)
    (js_install_project / "package.json").write_text('{"dependencies": {"b": "2"}}\n')
    monkeypatch.setenv("FAIL_LOCKED", "1")
    monkeypatch.setenv("FAIL_UNLOCKED", "1")
    proc, call_list = _install_js(js_install_project)
    assert proc.returncode == 1 and len(call_list) == 2
    assert "so the lock file is made again" in proc.stderr
    assert (js_install_project / "pnpm-lock.yaml").read_text() == "lock\n"
    assert not (js_install_project / "node_modules" / ".ctloom-installed").exists()


def test_the_pnpm_and_yarn_files_count_for_javascript():
    """pnpm-workspace.yaml is a dependency file; yarn's install state is no change."""
    lang_config = LANG_DICT["js"]
    assert is_dep_file("web/pnpm-workspace.yaml", frozenset(lang_config.dep_file_tuple))
    assert ".yarn/install-state.gz" in lang_config.build_output_tuple


def test_what_the_javascript_build_tools_write_is_no_change(repo, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "ts")
    for name in ("package.json", "packages/a/package.json", "dist/assets/index-1a2b.js",
                 "coverage/lcov.info", "packages/a/dist/x.js", "src/coverage/plan.ts",
                 "tsconfig.tsbuildinfo", "src/app.ts"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x\n", encoding="utf-8")
    edit.exclude_build_output()
    assert sorted(gitcmd.status_path_list(str(repo), new_only=True)) == [
        "package.json", "packages/a/package.json", "src/app.ts", "src/coverage/plan.ts"]
    assert edit.build_output_path_list(["dist/index.js", "packages/a/dist/x.js", "src/dist/y.ts",
                                 "src/app.ts"]) == ["dist/index.js", "packages/a/dist/x.js"]


def test_a_build_output_place_the_repository_tracks_is_not_hidden(repo, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "ts")
    for name in ("package.json", "dist/index.html"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "track dist")
    (repo / "dist" / "index-1a2b.js").write_text("x\n", encoding="utf-8")
    (repo / "coverage").mkdir()
    (repo / "coverage" / "lcov.info").write_text("x\n", encoding="utf-8")
    edit.exclude_build_output()
    assert gitcmd.status_path_list(str(repo), new_only=True) == ["dist/index-1a2b.js"]
    assert edit.build_output_path_list(["dist/index-1a2b.js"]) == ["dist/index-1a2b.js"]


def test_a_near_entry_of_another_language_stays_hidden_where_the_repository_tracks_it(
        repo, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "cbl")
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [("py", ["pytest"])])
    monkeypatch.setattr(edit, "other_output_place_list", lambda file_list=None: [
        ("svc", ("__pycache__/",))])
    for name in ("svc/pyproject.toml", "svc/pkg/__pycache__/a.pyc", "svc/dist/old.whl"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "track them")
    assert edit.exclude_build_output() == ["/svc/build/", "/svc/**/__pycache__/"]


def test_a_place_entry_of_another_language_is_put_where_no_command_of_it_runs(repo,
                                                                            monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [("js", ["ctloom-check-js"])])
    monkeypatch.setattr(edit.shutil, "which", lambda name: None)
    for name in ("frontend/package.json", "frontend/dist/assets/index-1a2b.js"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x\n", encoding="utf-8")
    assert "/frontend/dist/" in edit._output_pattern_pair()[0]
    assert edit.build_output_path_list(["frontend/dist/assets/index-1a2b.js", "m.py"]) == [
        "frontend/dist/assets/index-1a2b.js"]


def test_a_place_entry_under_a_directory_with_special_characters_is_read_back(repo,
                                                                            monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "js")
    for name in ("a [b]*/package.json", "c [d]/package.json", "c [d]/dist/index.html"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "two packages")
    add_list = edit.exclude_build_output()
    assert "/a \\[b]\\*/dist/" in add_list and "/c \\[d]/dist/" not in add_list
    assert edit.build_output_path_list(["a [b]*/dist/x.js", "c [d]/dist/y.js", "a [b]*/s.js"]) \
        == ["a [b]*/dist/x.js", "c [d]/dist/y.js"]


def test_no_place_entry_is_put_inside_a_directory_named_by_name_alone(tmp_path):
    file_list = ["web/package.json", "web/node_modules/x/package.json"]
    for name in file_list:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    pattern_list = edit.output_pattern_list_of(
        edit.output_place_list_of(["js"], file_list, str(tmp_path)))
    assert "/web/dist/" in pattern_list
    assert not any(pattern.startswith("/web/node_modules/") for pattern in pattern_list)


def test_a_source_in_a_directory_named_as_tool_output_is_still_a_target(repo, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "kt")
    (repo / "build.gradle.kts").write_text("x\n", encoding="utf-8")
    assert edit.build_output_path_list(["src/main/kotlin/com/acme/build/Plan.kt"]) == []


def test_a_place_holding_a_tracked_file_deep_down_is_not_hidden(repo, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "js")
    for name in ("package.json", "build/icons/icon.png"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "track build")
    assert "/build/" not in edit.exclude_build_output()
    assert "/dist/" in edit._output_pattern_pair()[0]


def test_a_place_entry_of_another_language_is_put_in_each_of_its_directories(tmp_path):
    file_list = ["web/package.json", "web/packages/ui/package.json"]
    for name in file_list:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    pattern_list = edit.output_pattern_list_of(
        edit.output_place_list_of(["js"], file_list, str(tmp_path)))
    assert "/web/dist/" in pattern_list and "/web/packages/ui/dist/" in pattern_list
    assert "/web/**/node_modules/" in pattern_list
    assert "/web/packages/ui/**/node_modules/" not in pattern_list


# == The C and C++ default test command (sandbox/test-c.sh) =====
_TEST_C = pathlib.Path(__file__).resolve().parent.parent / "sandbox" / "test-c.sh"


@pytest.fixture
def cmake_project(tmp_path, monkeypatch):
    """A CMake project and stand-in cmake, ninja and ctest on PATH.

    cmake exits with $CONFIGURE_STATUS when configuring, and prints $BUILD_OUT and exits
    with $BUILD_STATUS when building; ctest prints its arguments and exits with
    $CTEST_STATUS."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "CMakeLists.txt").write_text("project(calc C)\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Configuring writes its arguments to $tmp_path/configure and, where FetchContent
    # would fetch, a source under _deps of the build directory
    (bin_dir / "cmake").write_text(
        '#!/bin/sh\ncase " $* " in *" --build "*) printf "%s\\n" "$BUILD_OUT"; '
        'exit "$BUILD_STATUS" ;; esac\n'
        f'echo "$*" > {tmp_path}/configure\n'
        'case " $* " in *FULLY_DISCONNECTED*) ;; *) mkdir -p "$6/_deps/gtest-src" && '
        'echo fetched > "$6/_deps/gtest-src/CMakeLists.txt" ;; esac\n'
        'echo "cmake configure failed"\nexit "$CONFIGURE_STATUS"\n')
    (bin_dir / "ninja").write_text("#!/bin/sh\n")
    (bin_dir / "ctest").write_text('#!/bin/sh\necho "ctest $*"\nexit "$CTEST_STATUS"\n')
    for tool in ("cmake", "ninja", "ctest"):
        (bin_dir / tool).chmod(0o755)
    for name, value in (("CONFIGURE_STATUS", "0"), ("BUILD_STATUS", "0"),
                        ("BUILD_OUT", "[1/1] Linking C executable test_one"),
                        ("CTEST_STATUS", "0"), ("TMPDIR", str(tmp_path / "tmp")),
                        ("XDG_CACHE_HOME", str(tmp_path / "cache"))):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return project


def _test_c(project, *arg_list):
    """Run sandbox/test-c.sh in a project and return the finished process."""
    return subprocess.run(["sh", str(_TEST_C), *arg_list], cwd=project, capture_output=True,
                          text=True, check=False)


def _check_c(project, *arg_list):
    """Run sandbox/check-c.sh in a project and return the finished process."""
    return subprocess.run(["sh", str(_TEST_C.parent / "check-c.sh"), *arg_list],
                          cwd=project, capture_output=True, text=True, check=False)


def test_the_c_test_command_runs_ctest_over_its_own_build_directory(cmake_project, tmp_path):
    proc = _test_c(cmake_project)
    assert proc.returncode == 0
    assert proc.stdout.startswith(f"ctest --test-dir {tmp_path}/tmp/ctloom-test-c-ninja-")
    assert "--output-on-failure" in proc.stdout
    # A build that went through prints nothing of its own
    assert "Linking" not in proc.stdout


def test_the_c_test_command_runs_no_test_after_a_failed_build(cmake_project, monkeypatch):
    """An executable an earlier build left would otherwise run in place of one that no
    longer builds."""
    monkeypatch.setenv("BUILD_STATUS", "1")
    monkeypatch.setenv("BUILD_OUT", "FAILED: CMakeFiles/t.dir/tests/t.c.o\n"
                                    "tests/t.c:3:22: error: too many arguments")
    proc = _test_c(cmake_project)
    assert proc.returncode == 1
    assert "error: too many arguments" in proc.stdout and "ctest" not in proc.stdout


def test_the_c_test_command_does_not_run_what_cannot_be_configured(cmake_project,
                                                                  monkeypatch):
    monkeypatch.setenv("CONFIGURE_STATUS", "1")
    proc = _test_c(cmake_project)
    assert proc.returncode == 1
    assert "cannot be configured" in proc.stderr and "ctest" not in proc.stdout


def test_the_c_test_command_does_not_run_without_a_cmake_project(cmake_project):
    (cmake_project / "CMakeLists.txt").unlink()
    proc = _test_c(cmake_project)
    assert proc.returncode == 1 and "no CMakeLists.txt" in proc.stderr


def test_the_c_test_command_does_not_run_without_its_tools(cmake_project, tmp_path):
    (tmp_path / "bin" / "ninja").unlink()
    proc = _test_c(cmake_project)
    assert proc.returncode == 1 and "ninja does not exist" in proc.stderr


def test_the_c_test_command_keeps_what_fetchcontent_fetched_for_an_offline_run(
        cmake_project, tmp_path):
    """The filling has a network and the runs after it do not."""
    assert _test_c(cmake_project).returncode == 0
    assert "FULLY_DISCONNECTED" not in (tmp_path / "configure").read_text()
    kept = list((tmp_path / "cache" / "ctloom-fetchcontent").glob("*/gtest-src/CMakeLists.txt"))
    assert len(kept) == 1
    # A later run starts from a build directory of its own
    shutil.rmtree(tmp_path / "tmp")
    assert _test_c(cmake_project, "--offline").returncode == 0
    assert "-DFETCHCONTENT_FULLY_DISCONNECTED=ON" in (tmp_path / "configure").read_text()
    copied = list((tmp_path / "tmp").glob("ctloom-test-c-ninja-*/_deps/gtest-src/CMakeLists.txt"))
    assert [p.read_text() for p in copied] == ["fetched\n"]


def test_an_offline_run_takes_what_was_fetched_again_over_what_its_build_directory_holds(
        cmake_project, tmp_path):
    """The cache is filled again while a run still holds its build directory."""
    assert _test_c(cmake_project).returncode == 0
    assert _test_c(cmake_project, "--offline").returncode == 0
    kept, = (tmp_path / "cache" / "ctloom-fetchcontent").glob("*/gtest-src/CMakeLists.txt")
    kept.write_text("fetched again\n")
    assert _test_c(cmake_project, "--offline").returncode == 0
    copied = list((tmp_path / "tmp").glob("ctloom-test-c-ninja-*/_deps/gtest-src/CMakeLists.txt"))
    assert [p.read_text() for p in copied] == ["fetched again\n"]


def test_the_c_check_is_given_the_same_offline_configure(cmake_project, tmp_path, monkeypatch):
    proc = _check_c(cmake_project, "--offline")
    assert proc.returncode == 0
    configure = (tmp_path / "configure").read_text()
    assert "-DFETCHCONTENT_FULLY_DISCONNECTED=ON" in configure and "ctloom-check-c-ninja" in configure


def test_what_fetchcontent_fetched_is_kept_when_the_configure_fails_after_it(
        cmake_project, tmp_path, monkeypatch):
    """A filling has the changed CMakeLists.txt and not the new source it names."""
    monkeypatch.setenv("CONFIGURE_STATUS", "1")
    assert _test_c(cmake_project).returncode == 1
    assert list((tmp_path / "cache" / "ctloom-fetchcontent").glob("*/gtest-src/CMakeLists.txt"))


@pytest.fixture
def make_project(tmp_path, monkeypatch):
    """A project with a Makefile and no CMakeLists.txt, and a stand-in make on PATH.

    make -n [-r] <target> fails with make's "No rule" line unless $MAKE_TARGETS names the
    target, says it has nothing to do for one $MAKE_NOTHING names, and prints a built-in
    rule's command for one $MAKE_BUILTIN names unless -r turns those off; make -k writes
    build/out where it runs, prints where it ran and $MAKE_OUT with "<dir>" written as
    that directory, and exits with $MAKE_STATUS; make --no-print-directory does the same
    and prints each of its arguments first. No cmake, ninja or ctest is on PATH."""
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "Makefile").write_text("test:\n\t./run_tests\n")
    (project / "src" / "a.c").write_text("int a;\n")
    (project / "src" / "b.c").write_text("int b;\n")
    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "make").write_text(
        '#!/bin/sh\ncase "$1" in\n'
        '  -n) shift; no_builtin=; [ "$1" = "-r" ] && { no_builtin=1; shift; }\n'
        '      case " $MAKE_BUILTIN " in *" $1 "*) [ -n "$no_builtin" ] || {\n'
        '        echo "cc $1.c -o $1"; exit 0; } ;; esac\n'
        '      case " $MAKE_NOTHING " in *" $1 "*)\n'
        '        echo "make: Nothing to be done for \'$1\'."; exit 0 ;; esac\n'
        '      case " $MAKE_TARGETS " in *" $1 "*) exit 0 ;; esac\n'
        '      echo "make: *** No rule to make target \'$1\'.  Stop." >&2; exit 2 ;;\n'
        '  -k) [ -e build/out ] && echo "left by an earlier run"\n'
        '      mkdir -p build && echo built > build/out\n'
        '      echo "make -k ${2:-} in $(pwd)"\n'
        '      printf "%s\\n" "$MAKE_OUT" | sed "s|<dir>|$(pwd)|g"\n'
        '      exit "$MAKE_STATUS" ;;\n'
        '  --no-print-directory) shift; mkdir -p build && echo built > build/out\n'
        '      for arg in "$@"; do echo "arg [$arg]"; done; echo "make in $(pwd)"\n'
        '      printf "%s\\n" "$MAKE_OUT" | sed "s|<dir>|$(pwd)|g"\n'
        '      exit "$MAKE_STATUS" ;;\n'
        'esac\n')
    (bin_dir / "make").chmod(0o755)
    for name, value in (("MAKE_TARGETS", "test check"), ("MAKE_NOTHING", ""),
                        ("MAKE_BUILTIN", ""),
                        ("MAKE_OUT", ""), ("MAKE_STATUS", "0"),
                        ("TMPDIR", str(tmp_path / "tmp"))):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return project


def _run_c(project, *arg_list):
    """Run sandbox/run-c.sh in a project and return the finished process."""
    return subprocess.run(["sh", str(_TEST_C.parent / "run-c.sh"), *arg_list], cwd=project,
                          capture_output=True, text=True, check=False)


def test_the_c_launcher_runs_make_with_the_args_in_a_copy(make_project, tmp_path,
                                                          monkeypatch):
    """What make builds stays out of the repository; its paths are the repository's."""
    monkeypatch.setenv("MAKE_OUT", "<dir>/src/a.c:1: error: boom")
    monkeypatch.setenv("MAKE_STATUS", "2")
    proc = _run_c(make_project, "demo", "IN=/w s/x")
    assert proc.returncode == 2
    assert proc.stdout.splitlines() == [
        "arg [demo]", "arg [IN=/w s/x]", f"make in {tmp_path}/tmp/ctloom-run-c-make-"
        + proc.stdout.split("ctloom-run-c-make-", 1)[1].split("\n", 1)[0],
        f"{make_project}/src/a.c:1: error: boom"]
    copy_dir, = (tmp_path / "tmp").glob("ctloom-run-c-make-*")
    assert (copy_dir / "build" / "out").exists() and not (copy_dir / ".git").exists()
    assert not (make_project / "build").exists()


def test_the_c_launcher_says_why_it_runs_nothing_without_a_makefile(make_project):
    (make_project / "Makefile").unlink()
    (make_project / "CMakeLists.txt").write_text("project(calc C)\n")
    proc = _run_c(make_project, "demo")
    assert proc.returncode == 127 and proc.stdout == ""
    assert proc.stderr.startswith(
        f"ctloom-run-c: there is no makefile or configure in {make_project}")


def _autotools_project(make_project, tmp_path, configure_status=0):
    """Turn make_project into an autotools project: configure.ac and no makefile, with a
    stand-in autoreconf that writes a configure script, which writes the Makefile and
    exits with configure_status. Each writes its name to tmp_path/steps."""
    (make_project / "Makefile").unlink()
    (make_project / "configure.ac").write_text("AC_INIT([calc], [1])\n")
    step_path = tmp_path / "steps"
    configure = (f'#!/bin/sh\necho configure >> {step_path}\n'
                 f'echo "checking for gcc... gcc"\n'
                 f'echo "all:" > Makefile\nexit {configure_status}\n')
    autoreconf = tmp_path / "bin" / "autoreconf"
    autoreconf.write_text(f"#!/bin/sh\necho autoreconf $* >> {step_path}\n"
                          f"cat > configure <<'EOF'\n{configure}EOF\n")
    autoreconf.chmod(0o755)
    return step_path


def test_the_c_test_command_configures_an_autotools_project_in_its_copy(make_project,
                                                                        tmp_path):
    step_path = _autotools_project(make_project, tmp_path)
    proc = _test_c(make_project)
    assert proc.returncode == 0, proc.stderr
    assert step_path.read_text().splitlines() == ["autoreconf -fi", "configure"]
    assert proc.stdout.startswith(f"make -k test in {tmp_path}/tmp/ctloom-test-c-make-")
    assert not (make_project / "Makefile").exists()
    assert not (make_project / "configure").exists()


def test_a_configure_script_there_is_run_without_autoreconf(make_project, tmp_path):
    step_path = _autotools_project(make_project, tmp_path)
    (make_project / "configure").write_text(
        f'#!/bin/sh\necho configure >> {step_path}\necho "all:" > Makefile\n')
    proc = _run_c(make_project, "demo")
    assert proc.returncode == 0, proc.stderr
    assert step_path.read_text().splitlines() == ["configure"]


def test_an_autotools_project_that_cannot_be_configured_did_not_run(make_project,
                                                                     tmp_path):
    _autotools_project(make_project, tmp_path, configure_status=1)
    proc = _test_c(make_project)
    assert proc.returncode == 1 and proc.stdout == ""
    assert proc.stderr.startswith("ctloom-test-c: ./configure failed:")
    assert "checking for gcc... gcc" in proc.stderr


# The output of automake's make check with VERBOSE=yes where one test failed: the result
# lines, test-suite.log with the failed test's log under its own "FAIL:" heading, the
# last summary and make's lines
_AUTOMAKE_FAIL_OUT = """PASS: tests/doc_test
XFAIL: tests/known_test
FAIL: tests/calc_test
ERROR: tests/io_test
=================================
   calc 1.0: ./test-suite.log
=================================

# TOTAL: 4
# FAIL:  1

FAIL: tests/calc_test
=====================

FAIL: adds two numbers
*** add(2, 2) gave 5 at line 14
1 of 9 tests passed
FAIL tests/calc_test (exit status: 1)

============================================================================
Testsuite summary for calc 1.0
============================================================================
# TOTAL: 4
# PASS:  1
# FAIL:  1
============================================================================
See ./test-suite.log
============================================================================
make[2]: *** [Makefile:900: test-suite.log] Error 1
make[1]: Leaving directory '/tmp/ctloom-test-c-make-1'
make: *** [Makefile:950: check] Error 2
"""


def test_the_failing_tests_of_automake_s_make_check_are_named():
    """Not from test-suite.log, which names a failed test again and holds what the test
    programs printed."""
    assert [name for _, name in edit._fail_pair_list(_AUTOMAKE_FAIL_OUT, "c")] == [
        "tests/calc_test", "tests/io_test"]


def test_the_excerpt_of_a_failed_make_check_ends_with_the_failed_test_s_log(monkeypatch):
    monkeypatch.setattr(config, "FAIL_EXCERPT_CHAR", 120)
    excerpt = edit._fail_excerpt(_AUTOMAKE_FAIL_OUT, "c")
    assert excerpt.endswith("1 of 9 tests passed\nFAIL tests/calc_test (exit status: 1)")
    assert "make" not in excerpt and "Testsuite summary" not in excerpt


def test_a_test_program_s_own_fail_lines_are_no_names_without_automake_s_summary():
    """ctest prints what a failing test program wrote; its FAIL: lines name no test."""
    out = ("The following tests FAILED:\n\t  2 - t_calc (Failed)\n"
           "ERROR: open /tmp/tmp.Xa81 failed\nFAIL: tests/test_calc.c:42: add(2,2) != 5\n")
    assert [name for _, name in edit._fail_pair_list(out, "c")] == ["t_calc"]


def test_the_c_check_configures_an_autotools_project_in_its_copy(make_project, tmp_path,
                                                                monkeypatch):
    step_path = _autotools_project(make_project, tmp_path)
    monkeypatch.setenv("MAKE_OUT", "<dir>/src/a.c:1:5: error: boom")
    proc = _check_c(make_project)
    assert proc.returncode == 0, proc.stderr
    assert step_path.read_text().splitlines() == ["autoreconf -fi", "configure"]
    assert f"{make_project}/src/a.c:1:5: error: boom" in proc.stdout


def test_the_c_check_does_not_run_where_an_autotools_project_cannot_be_configured(
        make_project, tmp_path):
    _autotools_project(make_project, tmp_path, configure_status=1)
    proc = _check_c(make_project)
    assert proc.returncode == 127
    assert proc.stderr.startswith("ctloom-check-c: ./configure failed:")


def test_an_automake_build_runs_as_many_jobs_as_the_container_is_given(make_project,
                                                                        tmp_path,
                                                                        monkeypatch):
    """A makefile written by hand is made without -j."""
    stand_in = tmp_path / "bin" / "make"
    stand_in.write_text(stand_in.read_text().replace(
        "#!/bin/sh\n", '#!/bin/sh\ncase "$1" in -j*) echo "jobs $1"; shift ;; esac\n', 1))
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    assert "jobs" not in _run_c(make_project, "demo").stdout
    (make_project / "Makefile.am").write_text("TESTS = t\n")
    assert _run_c(make_project, "demo").stdout.startswith("jobs -j2\n")


def test_a_build_error_after_a_bundled_library_s_summary_stays_in_the_excerpt():
    out = ("PASS: lib/sample\n" + "=" * 20 + "\nTestsuite summary for lib 1.0\n"
           + "=" * 20 + "\n# TOTAL: 1\n# PASS:  1\n" + "=" * 20 + "\n"
           "make[2]: Leaving directory '/tmp/x/lib'\n"
           "src/calc.c:12:5: error: 'y' undeclared\n"
           "make[1]: *** [Makefile:9: calc.o] Error 1\n")
    assert edit._fail_excerpt(out, "c").endswith("src/calc.c:12:5: error: 'y' undeclared")


def test_a_missing_source_after_a_bundled_library_s_summary_stays_in_the_excerpt():
    out = ("PASS: lib/sample\n" + "=" * 20 + "\nTestsuite summary for lib 1.0\n"
           + "=" * 20 + "\n# TOTAL: 1\n# PASS:  1\n" + "=" * 20 + "\n"
           "make[2]: *** No rule to make target 'src/calc.c', needed by 'src/calc.lo'."
           "  Stop.\nmake[1]: *** [Makefile:9: all-recursive] Error 1\n")
    assert edit._fail_excerpt(out, "c").endswith(
        "make[2]: *** No rule to make target 'src/calc.c', needed by 'src/calc.lo'.  Stop.")


def test_makefile_am_is_a_file_the_c_tests_are_built_by():
    assert "Makefile.am" in LANG_DICT["c"].test_build_file_tuple


def test_make_check_of_an_automake_build_prints_the_failed_tests_logs(make_project,
                                                                      tmp_path):
    """A makefile written by hand gets no VERBOSE."""
    assert "VERBOSE" not in _test_c(make_project).stdout
    (make_project / "Makefile.am").write_text("TESTS = t\n")
    stand_in = tmp_path / "bin" / "make"
    stand_in.write_text(stand_in.read_text().replace(
        "#!/bin/sh\n", '#!/bin/sh\ncase "$1" in -j*) shift ;; esac\n', 1).replace(
        'echo "make -k ${2:-} in $(pwd)"', 'echo "make $* in $(pwd)"'))
    assert _test_c(make_project).stdout.startswith("make -k test VERBOSE=yes in ")


def test_the_c_test_command_runs_the_makefile_test_target_in_a_copy(make_project, tmp_path):
    """What make writes stays out of the repository, so it never lands in a patch."""
    proc = _test_c(make_project)
    assert proc.returncode == 0
    assert proc.stdout.startswith(f"make -k test in {tmp_path}/tmp/ctloom-test-c-make-")
    copy_dir, = (tmp_path / "tmp").glob("ctloom-test-c-make-*")
    assert (copy_dir / "src" / "a.c").read_text() == "int a;\n"
    assert (copy_dir / "build" / "out").exists() and not (copy_dir / ".git").exists()
    assert not (make_project / "build").exists()


def test_the_c_test_command_builds_no_makefile_where_the_package_cache_is_filled(
        make_project, tmp_path, monkeypatch):
    """A makefile fetches nothing ahead. With or without --offline (a TEST_COMMAND set by
    hand gets none), it builds wherever else it runs."""
    monkeypatch.setenv("CTLOOM_FILL", "1")
    proc = _test_c(make_project)
    assert (proc.returncode, proc.stdout, proc.stderr) == (0, "", "")
    assert not list((tmp_path / "tmp").glob("ctloom-test-c-make-*"))
    monkeypatch.delenv("CTLOOM_FILL")
    for arg_list in ([], ["--offline"]):
        assert _test_c(make_project, *arg_list).stdout.startswith("make -k test in ")


def test_the_c_test_command_builds_the_makefile_from_scratch_each_run(make_project,
                                                                      tmp_path):
    """A makefile that does not name the headers an object depends on would otherwise
    link an object built before the header changed, and a source the change removed
    would still be built."""
    assert _test_c(make_project).returncode == 0
    (make_project / "src" / "b.c").unlink()
    proc = _test_c(make_project)
    assert proc.returncode == 0 and "left by an earlier run" not in proc.stdout
    copy_dir, = (tmp_path / "tmp").glob("ctloom-test-c-make-*")
    assert not (copy_dir / "src" / "b.c").exists()


def test_the_c_test_command_writes_the_copys_paths_as_the_repositorys(make_project,
                                                                     monkeypatch):
    monkeypatch.setenv("MAKE_STATUS", "2")
    monkeypatch.setenv("MAKE_OUT", "<dir>/src/a.c:1:5: error: expected ';'\n"
                                   "make: *** [Makefile:2: test] Error 1")
    proc = _test_c(make_project)
    assert proc.returncode == 2
    assert f"\n{make_project}/src/a.c:1:5: error: expected ';'\n" in proc.stdout
    assert "ctloom-test-c-make-" not in proc.stdout.split("\n", 1)[1]
    assert edit._test_verdict(proc.returncode, proc.stdout, "c", []) is False


def test_the_c_test_command_runs_the_check_target_of_a_makefile_with_no_test_target(
        make_project, monkeypatch):
    monkeypatch.setenv("MAKE_TARGETS", "all check")
    proc = _test_c(make_project)
    assert proc.returncode == 0 and proc.stdout.startswith("make -k check in ")


def test_a_test_directory_with_no_test_rule_is_not_the_makefiles_test_target(make_project,
                                                                            monkeypatch):
    """make has nothing to do for a target named after a directory it finds there."""
    (make_project / "test").mkdir()
    monkeypatch.setenv("MAKE_TARGETS", "all check")
    monkeypatch.setenv("MAKE_NOTHING", "test")
    proc = _test_c(make_project)
    assert proc.returncode == 0 and proc.stdout.startswith("make -k check in ")
    # With no such file or directory, a target make has nothing to do for is one
    (make_project / "test").rmdir()
    proc = _test_c(make_project)
    assert proc.returncode == 0 and proc.stdout.startswith("make -k test in ")


def test_a_test_file_a_built_in_rule_makes_is_not_the_makefiles_test_target(make_project,
                                                                            monkeypatch):
    """make would build test from test.c with its own rules, and run no test."""
    (make_project / "test.c").write_text("int main(void) { return 0; }\n")
    monkeypatch.setenv("MAKE_TARGETS", "check")
    monkeypatch.setenv("MAKE_BUILTIN", "test")
    proc = _test_c(make_project)
    assert proc.returncode == 0 and proc.stdout.startswith("make -k check in ")


def test_the_c_test_command_does_not_run_a_makefile_with_neither_target(make_project,
                                                                       monkeypatch):
    monkeypatch.setenv("MAKE_TARGETS", "all")
    proc = _test_c(make_project)
    assert proc.returncode == 1
    assert "Makefile has no test or check target" in proc.stderr
    assert "make -k" not in proc.stdout
    assert edit._STOP_SHORT_RE_DICT["c"].search(proc.stderr)


def test_the_c_test_command_leaves_a_makefile_alone_when_told_to(make_project):
    """As it is told when C is not the repository's primary language."""
    proc = _test_c(make_project, "--no-make")
    assert proc.returncode == 1
    assert "no CMakeLists.txt, makefile or configure" in proc.stderr
    assert "make -k" not in proc.stdout


@pytest.mark.parametrize("out, missing", [
    # A command a sub-make runs: the outer make says Error 2 of the inner one
    ("/bin/sh: 1: valgrind: not found\nmake[1]: *** [Makefile:3: run] Error 127\n"
     "make: *** [Makefile:2: test] Error 2", "/bin/sh: 1: valgrind: not found"),
    # A command make runs itself, with no shell between
    ("make: valgrind: No such file or directory\nmake: *** [Makefile:2: test] Error 127",
     "make: valgrind: No such file or directory"),
    # A recipe bash runs (SHELL := /bin/bash)
    ("/bin/bash: line 1: valgrind: command not found\n"
     "make: *** [Makefile:3: test] Error 127",
     "/bin/bash: line 1: valgrind: command not found")])
def test_a_makefile_that_runs_a_command_that_is_not_installed_did_not_run(
        make_project, monkeypatch, out, missing):
    """As a CMake project whose configure cannot find it: nothing about the code."""
    monkeypatch.setenv("MAKE_STATUS", "2")
    monkeypatch.setenv("MAKE_OUT", out)
    proc = _test_c(make_project)
    assert proc.returncode == 1
    assert "ctloom-test-c: the makefile runs a command that is not installed" in proc.stderr
    assert missing in proc.stderr and "***" not in proc.stdout
    whole = proc.stdout + proc.stderr
    assert edit._test_verdict(proc.returncode, whole, "c", []) is None
    assert edit._STOP_SHORT_RE_DICT["c"].search(whole)


@pytest.mark.parametrize("out", [
    # A header or a library missing, the machine's or the change's alike
    "src/a.c:1:10: fatal error: zlib.h: No such file or directory\n"
    "make: *** [Makefile:4: a.o] Error 1",
    "/usr/bin/ld: cannot find -lz: No such file or directory\n"
    "collect2: error: ld returned 1 exit status\nmake: *** [Makefile:4: calc] Error 1",
    # A unit test that fails beside a step whose command is not installed
    "make: *** [Makefile:5: unit] Error 1\n/bin/sh: 1: valgrind: not found\n"
    "make: *** [Makefile:7: integration] Error 127",
    # A test program that exits 2 (two failures) beside such a step
    "make: *** [Makefile:5: unit] Error 2\n/bin/sh: 1: valgrind: not found\n"
    "make: *** [Makefile:7: integration] Error 127",
    # A test binary the change renamed, run by its path
    "make: ./calc_test: No such file or directory\nmake: *** [Makefile:2: test] Error 127",
    # A source the makefile names that the change removed
    "make: *** No rule to make target 'src/b.c', needed by 'src/b.o'.",
    # A makefile the change broke
    "Makefile:12: *** missing separator.  Stop."])
def test_a_makefile_run_that_fails_for_more_than_a_missing_command_is_red(
        make_project, monkeypatch, out):
    monkeypatch.setenv("MAKE_STATUS", "2")
    monkeypatch.setenv("MAKE_OUT", out)
    proc = _test_c(make_project)
    assert proc.returncode == 2 and proc.stderr == ""
    assert edit._test_verdict(proc.returncode, proc.stdout, "c", []) is False


def test_the_c_check_counts_a_source_the_makefile_names_that_is_gone(make_project,
                                                                    monkeypatch):
    monkeypatch.setenv("MAKE_OUT", "make: *** No rule to make target 'src/b.c', needed by "
                                   "'src/b.o'.")
    out = _check_c(make_project).stdout
    assert out == ("Makefile:1: error: no rule to make target 'src/b.c', needed by "
                   "'src/b.o'.\n")
    assert edit._CHECK_LINE_RE.match(out.strip())


def test_the_c_check_counts_an_error_of_make_in_a_makefile(make_project, monkeypatch):
    monkeypatch.setenv("MAKE_OUT", "Makefile:12: *** missing separator.  Stop.")
    out = _check_c(make_project).stdout
    assert out == "Makefile:12: error: missing separator.  Stop.\n"
    assert edit._CHECK_LINE_RE.match(out.strip())


def test_the_c_check_leaves_out_a_header_the_repository_does_not_have(make_project,
                                                                     monkeypatch):
    """The machine's, found before the change as after it; the tests say it is red."""
    (make_project / "include").mkdir()
    (make_project / "include" / "calc.h").write_text("int add(int, int);\n")
    monkeypatch.setenv("MAKE_OUT", "src/a.c:1:10: fatal error: curl/curl.h: No such "
                                   "file or directory\n"
                                   "tests/t.c:1:10: fatal error: ../include/calc.h: "
                                   "No such file or directory\n"
                                   "make: *** [Makefile:4: a.o] Error 1")
    assert _check_c(make_project).stdout == (
        "tests/t.c:1:10: fatal error: ../include/calc.h: No such file or directory\n")


def test_the_c_check_counts_a_header_the_change_removed(make_project, monkeypatch):
    """A header of the base commit that is gone is the change's, not the machine's."""
    (make_project / "include").mkdir()
    (make_project / "include" / "calc.h").write_text("int add(int, int);\n")
    shutil.rmtree(make_project / ".git")
    for argv in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "start"]):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@e", *argv],
                       cwd=make_project, check=True)
    (make_project / "include" / "calc.h").rename(make_project / "include" / "calculator.h")
    monkeypatch.setenv("MAKE_OUT", "src/a.c:1:10: fatal error: calc.h: No such file or "
                                   "directory\nmake: *** [Makefile:4: a.o] Error 1")
    assert _check_c(make_project).stdout == (
        "src/a.c:1:10: fatal error: calc.h: No such file or directory\n")


@pytest.mark.parametrize("line, verdict", [
    ("make: *** [Makefile:2: test] Error 1", False),
    ("make[1]: *** [Makefile:2: run] Error 2", False),
    ("make: *** [Makefile:2: test] Aborted (core dumped)", False),
    ("make: *** [Makefile:2: test] Segmentation fault", False),
    ("make: [Makefile:2: test] Error 1 (ignored)", None)])
def test_a_failed_step_of_a_makefile_is_red(line, verdict):
    assert edit._test_verdict(2, line, "c", []) is verdict


def test_the_c_check_builds_a_makefile_in_a_copy_and_prints_its_error_lines(make_project,
                                                                           tmp_path,
                                                                           monkeypatch):
    monkeypatch.setenv("MAKE_OUT", "src/a.c:1:5: error: expected ';'\n"
                                   "<dir>/src/b.c:2:1: error: unknown type\n"
                                   "src/b.c:1:1: warning: unused\n"
                                   "make: *** [Makefile:2: all] Error 1")
    proc = _check_c(make_project)
    assert proc.returncode == 0
    assert proc.stdout == (f"src/a.c:1:5: error: expected ';'\n"
                           f"{make_project}/src/b.c:2:1: error: unknown type\n")
    assert list((tmp_path / "tmp").glob("ctloom-check-c-make-*"))
    assert not (make_project / "build").exists()


def test_the_c_check_leaves_a_makefile_alone_when_told_to(make_project, monkeypatch):
    monkeypatch.setenv("MAKE_OUT", "src/a.c:1:5: error: expected ';'")
    assert (_check_c(make_project, "--no-make").stdout, _check_c(make_project).stdout) == (
        "", "src/a.c:1:5: error: expected ';'\n")


def test_the_c_check_prints_nothing_with_neither_cmake_nor_a_makefile(make_project):
    (make_project / "Makefile").unlink()
    proc = _check_c(make_project)
    assert (proc.returncode, proc.stdout, proc.stderr) == (0, "", "")


def test_a_run_against_the_cache_as_it_is_asks_for_no_refill(repo, monkeypatch,
                                                              fake_test_command):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "separator")
    fake_test_command(0, "1 passed in 0.01s")
    asks = _asked_to_refill(monkeypatch)
    edit.reset_patch_round()
    edit.create_file("pyproject.toml", "[project]\n")
    assert edit.run_test_as_is()["status"] == "passed" and asks == []
    assert edit.run_test()["status"] == "passed" and len(asks) == 1


# == Which files a run takes in ================================


@pytest.mark.parametrize("primary, path_list, want", [
    ("py", ["tests/data/pairs.json", "tests/test_add.py", "tests/conftest.py"],
     ["tests/test_add.py", "tests/conftest.py"]),
    ("js", ["test/a.test.ts", "test/b.test.js", "test/c.json"],
     ["test/a.test.ts", "test/b.test.js"]),
    ("c", ["tests/t.c", "tests/t.h", "tests/t.cpp", "tests/CMakeLists.txt"],
     ["tests/t.c", "tests/t.h", "tests/t.cpp"]),
    ("sql", ["tests/t.sql", "tests/t.py"], ["tests/t.sql"])])
def test_the_files_a_test_command_runs_are_those_of_its_language(primary, path_list,
                                                                 want, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", primary)
    assert edit.runnable_path_list(path_list) == want


# A test command standing in for one that builds what a list file names and then runs
# it: a named file holding "#error" fails the build the way Ninja says it
_BUILD_SCRIPT = """
import pathlib, sys
for name in pathlib.Path("build.txt").read_text().split():
    if "#error" in pathlib.Path(name).read_text():
        print("FAILED: " + name)
        sys.exit(1)
print("100% tests passed, 0 tests failed out of 1")
"""


@pytest.fixture
def build_repo(repo, monkeypatch):
    """repo, with a C test command that builds tests/in.c and not tests/out.c."""
    (repo / "tests").mkdir()
    (repo / "tests" / "in.c").write_bytes(b"int main(void) { return 0; }\r\n")
    (repo / "tests" / "out.c").write_bytes(b"int main(void) { return 1; }")
    (repo / "build.txt").write_text("tests/in.c\n")
    monkeypatch.setattr(config, "PRIMARY_EXT", "c")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "LLM_BROKER", "")
    monkeypatch.setattr(config, "TEST_COMMAND", [sys.executable, "-c", _BUILD_SCRIPT])
    return repo


def test_a_test_file_the_whole_suite_never_builds_is_told_from_one_it_does(build_repo):
    whole_dict = edit.run_test()
    assert whole_dict["status"] == "passed"
    round_count, run_count = edit.patch_round_count(), edit.last_whole_run()["n"]
    assert edit.in_whole_run("tests/in.c", whole_dict) is True
    assert edit.in_whole_run("tests/out.c", whole_dict) is False
    # Each file is put back byte for byte, and the stage's own runs are as they were
    assert (build_repo / "tests" / "in.c").read_bytes() == b"int main(void) { return 0; }\r\n"
    assert (build_repo / "tests" / "out.c").read_bytes() == b"int main(void) { return 1; }"
    assert edit.last_test_ok() is True and edit.patch_round_count() == round_count
    assert edit.last_whole_run()["n"] == run_count


def test_a_red_suite_that_stays_as_red_says_the_file_is_not_built(build_repo, monkeypatch):
    red_dict = {"status": "failed", "failed": ["t_old"], "failed_excerpt": "",
                "targets": [], "output": ""}
    assert edit.in_whole_run("tests/in.c", red_dict) is True
    monkeypatch.setattr(edit, "run_test_as_is", lambda *a, **k: dict(red_dict))
    assert edit.in_whole_run("tests/out.c", red_dict) is False


def test_a_test_file_that_is_or_becomes_a_named_pipe_is_not_waited_on(build_repo,
                                                                       monkeypatch):
    # The run with the line added leaves a named pipe at the file
    monkeypatch.setattr(config, "TEST_COMMAND", [sys.executable, "-c", (
        "import os, sys\n"
        "if '#error' in open('tests/in.c').read():\n"
        "    os.remove('tests/in.c'); os.mkfifo('tests/in.c'); sys.exit(1)\n"
        "print('100% tests passed, 0 tests failed out of 1')\n")])
    whole_dict = edit.run_test()
    with pytest.raises(OSError, match="in.c is not a regular file"):
        edit.in_whole_run("tests/in.c", whole_dict)
    assert stat.S_ISFIFO(os.lstat(build_repo / "tests" / "in.c").st_mode)
    assert edit.in_whole_run("tests/in.c", whole_dict) is None


@pytest.mark.parametrize("signal_number", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("leave", ["named pipe", "directory", "file"])
def test_a_stop_during_the_run_with_a_line_added_stops_whatever_the_file_became(
        build_repo, monkeypatch, caplog, signal_number, leave):
    """The stop is raised as it is; a file that cannot be put back is named in the log."""
    make = {"named pipe": "os.remove(p); os.mkfifo(p)",
            "directory": "os.remove(p); os.mkdir(p)",
            "file": "open(p, 'w').write('int x;\\n')"}[leave]
    monkeypatch.setattr(config, "TEST_COMMAND", [sys.executable, "-c", (
        "import os, signal, sys, time\n"
        "p = 'tests/in.c'\n"
        "if '#error' in open(p).read():\n"
        f"    {make}; os.kill({os.getpid()}, {int(signal_number)}); time.sleep(10)\n"
        "print('100% tests passed, 0 tests failed out of 1')\n")])
    whole_dict = edit.run_test()
    def stop(*_):
        raise KeyboardInterrupt

    before_handler = signal.signal(signal.SIGTERM, stop)
    try:
        with pytest.raises(KeyboardInterrupt):
            edit.in_whole_run("tests/in.c", whole_dict)
    finally:
        signal.signal(signal.SIGTERM, before_handler)
    if leave == "file":
        assert "#error" not in (build_repo / "tests" / "in.c").read_text()
    else:
        assert "tests/in.c could not be put back" in caplog.text


def test_a_python_test_file_the_whole_run_never_collects_is_told_from_one_it_does(
        repo, monkeypatch):
    """pytest runs a file named on its command line whatever the file is called."""
    (repo / "tests").mkdir()
    (repo / "tests" / "test_in.py").write_text("def test_in():\n    assert True\n")
    (repo / "tests" / "check_out.py").write_bytes(b"def test_out():\r\n    assert False\r\n")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "LLM_BROKER", "")
    monkeypatch.setattr(config, "TEST_COMMAND",
                        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    assert edit.run_test(["tests/check_out.py"])["status"] == "failed"
    whole_dict = edit.run_test()
    assert whole_dict["status"] == "passed"
    assert edit.in_whole_run("tests/test_in.py", whole_dict) is True
    assert edit.in_whole_run("tests/check_out.py", whole_dict) is False
    assert (repo / "tests" / "check_out.py").read_bytes() == \
        b"def test_out():\r\n    assert False\r\n"


_SUREFIRE_OUTPUT = """[INFO] -------------------------------------------------------
[INFO]  T E S T S
[INFO] -------------------------------------------------------
[INFO] Running ex.CalcTest
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.04 s -- in ex.CalcTest
[INFO] Running ex.inner.Calc$Test
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0
[INFO] BUILD SUCCESS
"""


@pytest.mark.parametrize("path, want", [
    ("src/test/java/ex/CalcTest.java", True), ("src/test/java/ex/Calc$Test.java", True),
    ("src/test/java/ex/CalcCheck.java", False), ("src/test/java/ex/Test.java", False)])
def test_a_java_test_file_is_run_when_the_whole_run_says_it_runs_its_class(
        repo, monkeypatch, path, want):
    monkeypatch.setattr(config, "PRIMARY_EXT", "java")
    monkeypatch.setattr(edit, "run_test_as_is", lambda *a, **k: pytest.fail("probed"))
    whole_dict = {"status": "passed", "failed": [], "output": _SUREFIRE_OUTPUT}
    assert edit.in_whole_run(path, whole_dict) is want


@pytest.mark.parametrize("output", [
    "[INFO] BUILD SUCCESS\n", "... (9 chars omitted)\n" + _SUREFIRE_OUTPUT])
def test_a_java_run_that_names_no_class_or_was_cut_is_probed(repo, monkeypatch, output):
    (repo / "src" / "test" / "java").mkdir(parents=True)
    (repo / "src" / "test" / "java" / "CalcCheck.java").write_text("class CalcCheck {}\n")
    monkeypatch.setattr(config, "PRIMARY_EXT", "java")
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", len(_SUREFIRE_OUTPUT))
    probe_list = []
    monkeypatch.setattr(edit, "run_test_as_is", lambda *a, **k: probe_list.append(1) or {
        "status": "failed", "failed": [], "output": "COMPILATION ERROR"})
    whole_dict = {"status": "passed", "failed": [], "output": output}
    assert edit.in_whole_run("src/test/java/CalcCheck.java", whole_dict) is True
    assert probe_list == [1]


def test_a_kotlin_test_file_is_told_by_breaking_it_as_gradle_prints_no_line_per_class(
        repo, monkeypatch):
    """gradle names only the tests that failed, so the build is what tells a file in."""
    whole_dict = json.loads((_GOLDEN_DIR / "kt_R_check.json").read_text())
    assert LANG_DICT["kt"].run_line_pattern == ""
    for path in ("src/test/kotlin/CalcCheck.kt", "tests/OutCheck.kt"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("class X\n")
    monkeypatch.setattr(config, "PRIMARY_EXT", "kt")
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", 0)
    whole_dict = {"status": "failed", "failed": ["CalcCheck > checksOnPurpose()"],
                  "output": whole_dict["output"]}

    def probe(*args, **kwargs):
        if "#error" in (repo / "src/test/kotlin/CalcCheck.kt").read_text():
            return {"status": "failed", "failed": [], "output": "e: Syntax error"}
        return dict(whole_dict)

    monkeypatch.setattr(edit, "run_test_as_is", probe)
    assert edit.in_whole_run("src/test/kotlin/CalcCheck.kt", whole_dict) is True
    assert edit.in_whole_run("tests/OutCheck.kt", whole_dict) is False


def test_a_kotlin_test_file_is_told_by_the_run_ending_early_once_it_is_broken(
        repo, as_language, monkeypatch):
    """Another project's test fails before and after the file is broken; the run with
    the line added is the one that ends before every test was run."""
    as_language("kt")
    path = "a/src/test/kotlin/TwoTest.kt"
    (repo / path).parent.mkdir(parents=True)
    (repo / path).write_text("class TwoTest\n")
    red_dict, stop_dict = (json.loads((_GOLDEN_DIR / f"{name}.json").read_text())
                           for name in ("kt_R_multi", "kt_B_multi"))
    red_output = red_dict["output"].replace("ATest", "BTest")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c",
        "import pathlib, sys;"
        "is_stop = '#error' in pathlib.Path(sys.argv[1]).read_text();"
        "sys.stdout.write(sys.argv[3] if is_stop else sys.argv[2]); sys.exit(1)",
        str(repo / path), red_output, stop_dict["output"]])
    whole_dict = edit.run_test()
    assert whole_dict["failed"] == ["BTest > one()"]
    assert edit.early_stop_run(whole_dict) is False
    assert edit.in_whole_run(path, whole_dict) is True
    assert (repo / path).read_text() == "class TwoTest\n"
    # A file the build does not take in leaves the run as it was
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1]); sys.exit(1)",
        red_output])
    assert edit.in_whole_run(path, edit.run_test()) is False


@pytest.mark.parametrize("name, line_list", [
    ("java_B_multi", ["[ERROR] COMPILATION ERROR : ",
                      "[ERROR] /fixture/java/B_multi/a/src/main/java/com/example/a/A.java:"
                      "[4,41] illegal start of expression",
                      "[INFO] c .................................................. SKIPPED"]),
    ("kt_B_multi", ["Execution failed for task ':a:compileKotlin'."]),
    ("kt_B_kapt", ["Execution failed for task ':a:kaptKotlin'."]),
    ("py_B", ["!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection "
              "!!!!!!!!!!!!!!!!!!!!"])])
def test_a_run_that_ended_early_says_why_with_the_lines_that_say_it_ended(
        name, line_list, as_language, golden_test_command, monkeypatch):
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", 300)
    as_language(name.split("_")[0])
    golden_test_command(name)
    result = edit.run_test()
    why = edit.early_stop_why(result)
    assert why.endswith(result["failed_excerpt"])
    head_list = why[:len(why) - len(result["failed_excerpt"])].splitlines()
    assert [line for line in line_list if line in why.splitlines()] == line_list
    assert (line_list[0] in head_list) is (line_list[0] not in result["failed_excerpt"])
    assert len(head_list) <= len(line_list) * (edit._EARLY_STOP_AFTER_LINE + 1)
    assert edit.early_stop_why(edit.part_of(result, next(iter(edit.part_dict_of(result))))) \
        == why


def test_a_run_that_did_not_end_early_says_why_with_its_excerpt_alone(
        as_language, golden_test_command, py_tests, fake_test_command, monkeypatch):
    as_language("kt")
    golden_test_command("kt_R_multi")
    result = edit.run_test()
    assert edit.early_stop_why(result) == result["failed_excerpt"]
    # The lines are not said twice where the excerpt holds them, and are cut to the
    # excerpt's own length where there are many
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    stop_line = "!!!! stopping after 1 failures !!!!"
    fake_test_command(1, f"FAILED tests/test_a.py::test_x - assert 0\n{stop_line}\n1 failed")
    result = edit.run_test()
    assert stop_line in result["failed_excerpt"]
    assert edit.early_stop_why(result) == result["failed_excerpt"]
    assert edit.early_stop_why(dict(result)) == result["failed_excerpt"]
    monkeypatch.setattr(config, "FAIL_EXCERPT_CHAR", 60)
    fake_test_command(1, f"{stop_line}\n" * 50 + "x" * 500 + "\n1 failed")
    why = edit.early_stop_why(edit.run_test())
    assert why.startswith(stop_line) and len(why) < 60 * 2 + 100


@pytest.mark.parametrize("path, status", [
    ("tests/in.ts", "passed"), ("tests/in.c", "did not run"), ("tests/none.c", "passed")])
def test_whether_a_file_is_built_is_not_told_where_nothing_can_tell_it(build_repo, path,
                                                                        status):
    (build_repo / "tests" / "in.ts").write_text("export const x = 1;\n")
    assert edit.in_whole_run(path, {"status": status, "failed": []}) is None


# == the formatting command run over what changed ==============================

@pytest.fixture
def _format_repo(tmp_path, monkeypatch):
    """A repository with one committed file, one changed, one new and one deleted."""
    from conftest import git_repo
    root = pathlib.Path(git_repo(tmp_path / "repo"))
    (root / "gone.py").write_text("y = 1\n", encoding="utf-8")
    (root / "same.py").write_text("z = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "two"], check=True)
    (root / "a.py").write_text("x = 2\n", encoding="utf-8")
    (root / "new dir").mkdir()
    (root / "new dir" / "b.py").write_text("w = 1\n", encoding="utf-8")
    (root / "gone.py").unlink()
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    return root


def _format_script(tmp_path, body: str) -> list[str]:
    """Return a FORMAT_COMMAND that runs a Python body with the files as sys.argv[1:]."""
    script = tmp_path / "format.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def test_the_formatting_command_is_given_the_files_that_changed(_format_repo, tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(config, "FORMAT_COMMAND", _format_script(tmp_path, """
import os, sys
for path in sys.argv[1:]:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("# formatted from " + os.path.basename(os.getcwd()) + "\\n")
"""))
    result = edit.run_format()
    # The deleted file and the one left as it was are not handed over
    assert sorted(result["files"]) == ["a.py", "new dir/b.py"]
    assert result["exit_status"] == 0
    assert (_format_repo / "a.py").read_text() == "x = 2\n# formatted from repo\n"
    assert (_format_repo / "new dir" / "b.py").read_text().endswith("from repo\n")
    assert (_format_repo / "same.py").read_text() == "z = 1\n"


def test_a_changed_file_whose_name_starts_with_a_dash_is_given_as_a_path(
        _format_repo, tmp_path, monkeypatch):
    (_format_repo / "-h.py").write_text("y = 1\n")
    (_format_repo / "--config=other.toml").write_text("a = 1\n")
    seen = tmp_path / "seen.json"
    monkeypatch.setattr(config, "FORMAT_COMMAND", _format_script(
        tmp_path, f"import json, sys\njson.dump(sys.argv[1:], open({str(seen)!r}, 'w'))\n"))
    result = edit.run_format()
    assert sorted(result["files"]) == ["--config=other.toml", "-h.py", "a.py",
                                       "new dir/b.py"]
    assert sorted(json.loads(seen.read_text())) == [
        "./--config=other.toml", "./-h.py", "a.py", "new dir/b.py"]


def test_no_formatting_command_runs_none(_format_repo, monkeypatch):
    monkeypatch.setattr(config, "FORMAT_COMMAND", [])
    assert edit.run_format() is None


def test_the_formatting_command_is_not_run_when_nothing_changed(tmp_path, monkeypatch):
    from conftest import git_repo
    monkeypatch.setattr(config, "PROJECT_DIR", git_repo(tmp_path / "repo"))
    monkeypatch.setattr(config, "FORMAT_COMMAND", ["/does/not/exist"])
    assert edit.run_format() is None


def test_a_formatting_command_that_fails_is_reported_and_not_raised(_format_repo,
                                                                    tmp_path, monkeypatch,
                                                                    caplog):
    monkeypatch.setattr(config, "FORMAT_COMMAND", _format_script(
        tmp_path, "import sys\nprint('bad syntax in a.py')\nsys.exit(3)\n"))
    result = edit.run_format()
    assert result["exit_status"] == 3 and result["output"] == "bad syntax in a.py"
    assert "did not end cleanly" in caplog.text and "exit status 3" in caplog.text


def test_a_formatting_command_that_is_not_there_is_reported(_format_repo, monkeypatch):
    monkeypatch.setattr(config, "FORMAT_COMMAND", ["/does/not/exist", "--fix"])
    result = edit.run_format()
    assert result["exit_status"] is None
    assert "Set FORMAT_COMMAND to something runnable" in result["output"]


def test_a_formatting_command_past_the_time_limit_is_stopped(_format_repo, tmp_path,
                                                             monkeypatch):
    monkeypatch.setattr(config, "TEST_TIMEOUT_SEC", 1)
    monkeypatch.setattr(config, "FORMAT_COMMAND", _format_script(
        tmp_path, "import time\ntime.sleep(30)\n"))
    result = edit.run_format()
    assert result["exit_status"] is None and "ran over 1 seconds" in result["output"]


# == the static checks of the languages besides the primary one ================

def _checker(tmp_path, name: str, line: str) -> str:
    """Write an executable that prints one finding and return its directory."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\necho '{line}'\n", encoding="utf-8")
    path.chmod(0o755)
    return str(bin_dir)


def test_the_checks_of_the_other_languages_run_after_the_primary_one(monkeypatch,
                                                                     tmp_path):
    bin_dir = _checker(tmp_path, "check-two", "web/a.ts(3,5): error TS2304: no f")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [
        sys.executable, "-c", "print('a.py:1:1: F821 Undefined name `f`')"])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [
        ("ts", ["check-two"]), ("rs", ["no-such-checker-here"])])
    assert [argv[0] for argv in edit.check_argv_list()] == [sys.executable, "check-two"]
    assert _RUN_CHECK() == {"a.py: F821 Undefined name `f`": 1,
                            "web/a.ts: error TS2304: no f": 1}
    assert edit.check_command_text().endswith(" / check-two")


def test_another_language_s_check_runs_where_the_primary_language_has_none(monkeypatch,
                                                                           tmp_path):
    bin_dir = _checker(tmp_path, "check-two", "web/a.js:3:5: no-undef f")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [("js", ["check-two"])])
    assert _RUN_CHECK() == {"web/a.js: no-undef f": 1}
    assert edit.check_command_text() == "check-two"


def test_a_check_that_does_not_run_leaves_the_others_counted(monkeypatch, tmp_path,
                                                            caplog):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "check-two").write_text("#!/bin/sh\nexit 127\n", encoding="utf-8")
    (bin_dir / "check-two").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [
        sys.executable, "-c", "print('a.py:1:1: F821 Undefined name `f`')"])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [("ts", ["check-two"])])
    count = _RUN_CHECK()
    assert count == {"a.py: F821 Undefined name `f`": 1}
    assert list(count.by_check_dict) == [
        f"{os.path.basename(sys.executable)} -c "
        "print('a.py:1:1: F821 Undefined name `f`')"]
    assert "cannot check this tree" in caplog.text


def test_a_check_is_the_same_check_whatever_file_of_the_root_it_is_pointed_at(
        monkeypatch, tmp_path):
    """The C# check is given the solution in the root, or the one project there."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ctloom-check-cs").write_text(
        '#!/bin/sh\necho "src/A.cs(1,1): error CS1: $1"\n', encoding="utf-8")
    (bin_dir / "ctloom-check-cs").chmod(0o755)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "App.csproj").write_text("<Project/>\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(config, "PROJECT_DIR", str(repo))
    monkeypatch.setattr(config, "PRIMARY_EXT", "cs")
    monkeypatch.setattr(config, "CHECK_COMMAND", ["ctloom-check-cs"])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [])
    before = _RUN_CHECK()
    (repo / "App.sln").write_text("\n")
    after = _RUN_CHECK()
    assert list(before.by_check_dict) == list(after.by_check_dict) == ["ctloom-check-cs"]
    assert edit.check_command_text() == "ctloom-check-cs App.sln"
    assert edit.new_finding_list(before, after) == ["src/A.cs: error CS1: App.sln"]


def test_what_a_check_found_is_compared_with_its_own_run_before(monkeypatch, tmp_path):
    """A check that ran after a change and not before it brings no finding as new."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "check-two"
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [
        sys.executable, "-c", "print('a.py:1:1: F821 Undefined name `f`')"])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [("ts", ["check-two"])])

    def write(body):
        script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        script.chmod(0o755)
        return _RUN_CHECK()

    unrun = write("exit 127")
    old = write("echo 'web/a.ts(3,5): error TS1: old'")
    new = write("echo 'web/a.ts(3,5): error TS1: old'; echo 'web/b.ts(1,1): error TS2: new'")
    assert edit.new_finding_list(unrun, new) == []
    assert edit.new_finding_list(old, new) == ["web/b.ts: error TS2: new"]
    assert edit.new_finding_list(new, unrun) == []
    # Two plain counts are compared whole
    assert edit.new_finding_list(collections.Counter(), collections.Counter(a=2)) == [
        "a", "a"]


def test_no_check_on_path_is_no_check(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [("ts", ["no-such-checker-here"])])
    assert edit.check_argv_list() == [] and edit.check_command_text() == ""
    assert _RUN_CHECK() is None


# == the tests of the languages besides the primary one ========================

def _test_tool(tmp_path, name: str, body: str) -> None:
    """Put an executable shell script on PATH's own directory of the test."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def _two_language_repo(tmp_path, monkeypatch):
    """A Python repository whose web/ directory is a JavaScript package.

    The primary command prints one passing pytest line; "jstest" stands in for the
    other language's command and is read as JavaScript.
    """
    from conftest import git_repo
    root = pathlib.Path(git_repo(tmp_path / "repo"))
    (root / "tests").mkdir()
    (root / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    (root / "web" / "test").mkdir(parents=True)
    (root / "web" / "package.json").write_text("{}\n")
    (root / "web" / "test" / "view.test.js").write_text("// a test\n")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    _test_tool(tmp_path, "jstest", 'echo "ran in $(basename "$PWD") with: $*"')
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "import sys; print('1 passed, targets:', sys.argv[1:])"])
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [("js", ["jstest"])])
    monkeypatch.setattr(edit, "refill_cache_on_dep_change", lambda: "")
    return root


def test_another_language_s_tests_run_from_where_its_dependency_file_is(
        _two_language_repo):
    assert edit.other_test_list() == [("js", ["jstest"], "web")]
    result = edit.run_test()
    assert result["status"] == "passed"
    head, primary_out, other_head, other_out = [
        line for line in result["output"].split("\n") if line]
    assert head.startswith("== python") and "targets: ['tests']" in primary_out
    assert other_head == "== jstest (in web) =="
    assert other_out == "ran in web with: "


def test_a_javascript_test_command_is_run_as_under_ci(_two_language_repo, tmp_path,
                                                      monkeypatch):
    """CI=true goes to the JavaScript and TypeScript commands, not to Python's."""
    monkeypatch.delenv("CI", raising=False)
    _test_tool(tmp_path, "jstest", 'echo "js CI=${CI:-unset}"')
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c",
        "import os; print('1 passed, py CI=' + os.environ.get('CI', 'unset'))"])
    output = edit.run_test()["output"]
    assert "js CI=true" in output and "py CI=unset" in output


def test_a_dependency_file_among_test_data_runs_no_test(_two_language_repo):
    """A directory with no source of the language under it is no project of it."""
    data = _two_language_repo / "tests" / "samples" / "JavaScript"
    data.mkdir(parents=True)
    (data / "package.json").write_text("{}\n")
    subprocess.run(["git", "-C", str(_two_language_repo), "add", "-A"], check=True)
    assert edit.other_test_list() == [("js", ["jstest"], "web")]
    # TypeScript sources count for the same command
    (_two_language_repo / "web" / "test" / "view.test.js").rename(
        _two_language_repo / "web" / "test" / "view.test.ts")
    subprocess.run(["git", "-C", str(_two_language_repo), "add", "-A"], check=True)
    assert edit.other_test_list() == [("js", ["jstest"], "web")]
    (_two_language_repo / "web" / "test" / "view.test.ts").unlink()
    subprocess.run(["git", "-C", str(_two_language_repo), "add", "-A"], check=True)
    assert edit.other_test_list() == []
    assert edit.other_test_ext_list(["js"], gitcmd.work_tree_file_list(
        str(_two_language_repo)), str(_two_language_repo)) == []
    # Test data nearer the root than the package is passed over
    (_two_language_repo / "samples").mkdir()
    (_two_language_repo / "samples" / "package.json").write_text("{}\n")
    (_two_language_repo / "web" / "src").mkdir()
    (_two_language_repo / "web" / "src" / "app.mjs").write_text("export {}\n")
    subprocess.run(["git", "-C", str(_two_language_repo), "add", "-A"], check=True)
    assert edit.other_test_list() == [("js", ["jstest"], "web")]


def test_a_project_file_among_test_data_is_no_root_target(_three_language_repo):
    """C#'s tests start where a project file is one target and sources are under it."""
    data = _three_language_repo / "samples"
    data.mkdir()
    (data / "Sample.csproj").write_text("<Project/>\n")
    (data / "All.sln").write_text("\n")
    subprocess.run(["git", "-C", str(_three_language_repo), "add", "-A"], check=True)
    assert edit.other_test_list()[1] == ("cs", ["cstool", "Gen.csproj"], "tools/gen")


def test_a_changed_dependency_file_of_another_language_whose_tests_run_is_found(
        _two_language_repo, monkeypatch):
    edit.reset_patch_round()
    assert edit.dep_file_pattern_set() == frozenset(
        LANG_DICT["py"].dep_file_tuple + LANG_DICT["js"].dep_file_tuple)
    assert edit.replace_in_file("web/package.json", "{}", '{"dependencies": {}}')["ok"]
    assert edit.change_dep_path_list() == ["web/package.json"]
    # A language whose tests do not run adds no pattern
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [("js", ["no-such-test-tool-here"])])
    assert edit.dep_file_pattern_set() == frozenset(LANG_DICT["py"].dep_file_tuple)
    assert edit.change_dep_path_list() == []


def test_a_path_written_another_way_goes_to_the_same_command(_two_language_repo):
    want = edit.run_test(["web/test/view.test.js"])
    assert want["output"].strip() == "ran in web with: -- test/view.test.js"
    for path in ("./web/test/view.test.js", "web/../web/test/view.test.js",
                 str(_two_language_repo / "web" / "test" / "view.test.js")):
        result = edit.run_test([path])
        assert (result["output"], result["targets"]) == (want["output"], want["targets"])
    # The primary command is given its paths in the same form
    assert edit.run_test(["./tests/test_a.py"])["targets"] == ["tests/test_a.py"]
    assert "cannot start with" in edit.run_test(["./-k"])["output"]


def test_what_follows_the_file_in_a_test_path_reaches_the_command_as_written(
        _two_language_repo):
    """A parameter of a test can hold "//" and "..", which are not parts of a path."""
    for name in ("test_a[http://x/a]", "test_a[a/../b]", "T::test_a"):
        result = edit.run_test([f"./tests/test_a.py::{name}"])
        assert result["targets"] == [f"tests/test_a.py::{name}"]
        result = edit.run_test([f"./web/test/view.test.js::{name}"])
        assert result["targets"] == [f"test/view.test.js::{name}"]
    # The file is what is checked, and what the option check reads
    assert "outside the repository" in edit.run_test(["../x.py::test_a"])["output"]
    assert "cannot start with" in edit.run_test(["./-k::test_a"])["output"]


def test_a_test_file_that_is_a_link_is_run_under_the_path_it_was_written_with(
        _two_language_repo):
    """The command it goes to is the one of the file the link points at."""
    (_two_language_repo / "tests" / "test_link.py").symlink_to("test_a.py")
    (_two_language_repo / "tests" / "view_link.py").symlink_to("../web/test/view.test.js")
    assert edit.run_test(["tests/test_link.py"])["targets"] == ["tests/test_link.py"]
    result = edit.run_test(["tests/view_link.py"])
    assert result["output"].strip() == "ran in web with: -- ../tests/view_link.py"


def test_the_file_that_is_checked_is_the_file_the_command_is_given(_two_language_repo,
                                                                    tmp_path, monkeypatch):
    """A link to a directory followed by ".." names one file to the system and another
    once the path is written without "..". The path without ".." is the one checked
    and the one run."""
    root = _two_language_repo
    (root / "deep" / "a").mkdir(parents=True)
    (root / "link").symlink_to("deep/a")
    (root / "t.py").write_text("def test_t():\n    assert True\n")
    (root / "deep" / "out").mkdir()
    (root / "out").mkdir()
    (root / "out" / "t.py").write_text("def test_t():\n    assert True\n")
    (tmp_path / "t.py").write_text("def test_t():\n    assert True\n")
    monkeypatch.setattr(config, "OUTPUT_DIR", str(root / "out"))
    assert edit.run_test(["link/../t.py"])["targets"] == ["t.py"]
    # The system reads these two as deep/../t.py and deep/out/t.py
    result = edit.run_test(["link/../../t.py"])
    assert result["status"] == "did not run" and result["targets"] == []
    assert "outside the repository" in result["output"]
    result = edit.run_test(["link/../out/t.py"])
    assert result["status"] == "did not run" and result["targets"] == []
    assert "OUTPUT_DIR holds generated output" in result["output"]
    # A path that leaves the root and comes back is given without the way round
    assert edit.run_test([f"../{root.name}/tests/test_a.py"])["targets"] == [
        "tests/test_a.py"]


def test_a_knowledge_path_goes_to_the_command_of_its_own_language(_two_language_repo):
    from codetwine_loom.paths import repo_to_knowledge
    path = repo_to_knowledge("web/test/view.test.js", _two_language_repo.name)
    assert path != "web/test/view.test.js"
    result = edit.run_test([path])
    assert result["output"].strip() == "ran in web with: -- test/view.test.js"
    assert result["targets"] == ["test/view.test.js"]


def test_a_path_a_command_is_given_is_never_read_as_an_option(_two_language_repo,
                                                              monkeypatch):
    """A file whose name starts with "-" in the directory another language's command
    is started in, the test directory, and a root target."""
    root = _two_language_repo
    (root / "web" / "-x.test.js").write_text("// a test\n")
    result = edit.run_test(["web/-x.test.js::case"])
    assert result["output"].strip() == "ran in web with: -- ./-x.test.js::case"
    assert result["targets"] == ["./-x.test.js::case"]
    assert edit.path_arg("-x.py") == "./-x.py" and edit.path_arg("a/-x.py") == "a/-x.py"
    assert edit._build_argv(None, "path", "-tests", "py", command=["run"]) == (
        ["run", "./-tests"], ["./-tests"], "")
    assert edit._build_argv(["-a", "b"], "separator", "", "js", command=["run"]) == (
        ["run", "--", "./-a", "b"], ["./-a", "b"], "")
    (root / "-All.slnx").write_text("x\n")
    monkeypatch.setattr(config, "TEST_COMMAND", ["dotnet", "test"])
    assert edit._with_root_target(["dotnet", "test"], ("dotnet", "test"), "cs") == [
        "dotnet", "test", "./-All.slnx"]


def test_a_path_like_object_is_a_path_to_the_editing_tools(repo):
    path = pathlib.Path("m.py")
    r = edit.replace_in_file(path, "nowhere", "new")
    assert (r["ok"], r["path"], r["error"]) == (False, "m.py", "old_text not found")
    assert edit.replace_in_file(path, "old", "new")["ok"] is True
    assert edit.create_file(pathlib.Path("n.py"), "x = 1\n")["path"] == "n.py"
    assert edit.delete_file(pathlib.Path("n.py"))["path"] == "n.py"


def test_one_test_path_alone_may_be_a_string(_two_language_repo):
    result = edit.run_test("tests/test_a.py")
    assert result["status"] == "passed" and result["targets"] == ["tests/test_a.py"]
    for path_list, want in ((5, "path_list must be a list of test file paths, not int"),
                            ({"tests/test_a.py": 1},
                             "path_list must be a list of test file paths, not dict"),
                            ([None], "a path must be a string, not NoneType")):
        result = edit.run_test(path_list)
        assert (result["status"], result["output"]) == ("did not run", want)


def test_each_path_goes_to_the_command_of_its_own_language(_two_language_repo):
    result = edit.run_test(["web/test/view.test.js"])
    # The primary command is left out, and the path is given from the package's directory
    assert result["output"].strip() == "ran in web with: -- test/view.test.js"
    assert result["targets"] == ["test/view.test.js"]
    result = edit.run_test(["tests/test_a.py"])
    assert "jstest" not in result["output"] and "['tests/test_a.py']" in result["output"]
    result = edit.run_test(["tests/test_a.py", "web/test/view.test.js"])
    assert result["targets"] == ["tests/test_a.py", "test/view.test.js"]


def test_a_red_test_of_another_language_makes_the_run_red(_two_language_repo, tmp_path):
    _test_tool(tmp_path, "jstest", 'echo "Test Suites: 1 failed, 2 total"; exit 1')
    result = edit.run_test()
    assert result["status"] == "failed"
    assert result["failed_excerpt"].startswith("== jstest (in web) ==")
    assert edit.last_test_ok() is False


def test_another_language_with_no_test_to_run_leaves_the_run_as_it_was(
        _two_language_repo, tmp_path):
    _test_tool(tmp_path, "jstest", 'echo "Error: no test specified"; exit 1')
    result = edit.run_test()
    assert result["status"] == "passed" and edit.last_test_ok() is True


@pytest.mark.parametrize("body", ['echo "Cannot find module x"; exit 1', "sleep 30"])
def test_a_command_that_did_not_run_is_not_hidden_by_one_that_passed(
        _two_language_repo, tmp_path, monkeypatch, body):
    monkeypatch.setattr(config, "TEST_TIMEOUT_SEC", 2)
    _test_tool(tmp_path, "jstest", body)
    result = edit.run_test()
    assert result["status"] == "did not run" and edit.last_test_ok() is None
    assert result["failed_excerpt"].startswith("== jstest (in web) ==")
    assert edit.no_test_run(result) is False


@pytest.mark.parametrize("body, is_no_test", [
    ('echo "Cannot find module x"; exit 1', False), ("sleep 30", False),
    ('echo "Error: no test specified"; exit 1', True)])
def test_a_run_found_no_test_only_when_every_command_found_none(
        _two_language_repo, tmp_path, monkeypatch, body, is_no_test):
    """The primary command prints pytest's line for a run over no tests."""
    monkeypatch.setattr(config, "TEST_TIMEOUT_SEC", 2)
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "print('no tests ran in 0.01s'); raise SystemExit(5)"])
    _test_tool(tmp_path, "jstest", body)
    result = edit.run_test()
    assert result["status"] == "did not run"
    assert edit.no_test_run(result) is is_no_test
    # It prints and compares as the dict it is
    assert repr(result) == repr(dict(result)) and result == dict(result)


def test_one_command_s_run_over_no_tests_is_read_by_its_own_language(_two_language_repo,
                                                                    tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND", [])
    _test_tool(tmp_path, "jstest", 'echo "Error: no test specified"; exit 1')
    result = edit.run_test()
    assert result["status"] == "did not run" and edit.no_test_run(result) is True


def test_a_run_holds_what_each_command_returned_on_its_own(_two_language_repo, tmp_path,
                                                          monkeypatch):
    primary = " ".join(config.TEST_COMMAND)
    _test_tool(tmp_path, "jstest", 'echo "Cannot find module x"; exit 1')
    result = edit.run_test()
    # What the model is shown is the run as a whole
    assert result["status"] == "did not run" and result["failed"] == []
    # A whole run in which a command reached a verdict is kept as the last one
    assert edit.last_whole_run()["result"] is result
    part_dict = edit.part_dict_of(result)
    # Each command is named as the settings hold it, without a target or a path
    assert list(part_dict) == [primary, "jstest"]
    assert part_dict[primary]["status"] == "passed"
    assert part_dict["jstest"]["status"] == "did not run"
    assert "Cannot find module" in part_dict["jstest"]["output"]
    assert not part_dict[primary]["output"].startswith("==")
    assert edit.primary_part(result) is part_dict[primary]
    assert edit.part_of(result, "jstest") is part_dict["jstest"]
    assert edit.part_of(result, "no such command")["status"] == "did not run"
    assert edit.no_test_run(part_dict["jstest"]) is False
    # A command that found no test of its own language says so on its own
    _test_tool(tmp_path, "jstest", 'echo "Error: no test specified"; exit 1')
    assert edit.no_test_run(edit.part_dict_of(edit.run_test())["jstest"]) is True
    # A run of one command holds that command alone
    result = edit.run_test(["tests/test_a.py"])
    assert list(edit.part_dict_of(result)) == [primary]
    # A dict that is not a run's own return value has no parts
    plain = dict(result)
    assert edit.part_dict_of(plain) == {} and edit.primary_part(plain) is plain
    assert edit.part_of(plain, primary) is plain


def test_the_gates_hold_a_real_run_command_by_command(_two_language_repo, tmp_path,
                                                     monkeypatch):
    """The other language's command never runs; the primary one passes, then is red
    without naming a test, then does not run."""
    from codetwine_loom import integrate, testcreate
    monkeypatch.setattr(config, "TEST_COMMAND", ["pytool"])
    _test_tool(tmp_path, "jstest", 'echo "jest: not found"; exit 127')
    _test_tool(tmp_path, "pytool", 'echo "1 passed in 0.10s"')
    before = edit.run_test()
    assert before["status"] == "did not run"
    baseline = integrate._baseline_set(before, edit.run_test)
    assert baseline == testcreate.baseline_of(before) == {
        edit.baseline_mark("green before", "pytool"),
        edit.baseline_mark("not compared", "jstest"),
        # and how many tests the primary command's run ran
        "\x1fran count\x1fpytool\x1f1"}
    assert integrate._test_reason(before, baseline) == ""
    assert testcreate.run_reason(before, baseline) == ""
    # The file the stage wrote is looked for in the primary command's own run: this
    # command reads no file, so the run with the line added says the same
    assert edit.in_whole_run("tests/test_a.py", before) is False
    _test_tool(tmp_path, "jstest", 'echo "Tests: 1 failed, 1 total"; exit 1')
    assert edit.in_whole_run("tests/test_a.py", edit.run_test()) is False
    _test_tool(tmp_path, "jstest", 'echo "jest: not found"; exit 127')
    _test_tool(tmp_path, "pytool",
               'if [ "$(tail -n 1 tests/test_a.py)" = ")" ]; then '
               'echo "1 error in 0.1s"; exit 1; fi; echo "1 passed in 0.10s"')
    assert edit.in_whole_run("tests/test_a.py", edit.run_test()) is True
    _test_tool(tmp_path, "pytool", 'echo "1 passed in 0.10s"')

    _test_tool(tmp_path, "pytool", 'echo "1 error in 0.09s"; exit 1')
    red = edit.run_test()
    assert red["status"] == "failed" and red["failed"] == []
    assert integrate._test_reason(red, baseline).startswith("- `pytool` fails (")
    assert testcreate.run_reason(red, baseline).startswith("`pytool` fails (")

    _test_tool(tmp_path, "pytool", 'echo "cannot import"; exit 2')
    not_run = edit.run_test()
    assert integrate._test_reason(not_run, baseline).startswith("- `pytool` did not run")
    assert testcreate.baseline_of(not_run) is None


def test_a_run_over_some_files_is_held_by_the_commands_it_ran(_two_language_repo,
                                                             tmp_path, monkeypatch):
    """The stages run the test files they wrote alone before the whole suite."""
    from codetwine_loom import integrate, testcreate
    monkeypatch.setattr(config, "TEST_COMMAND", ["pytool"])
    _test_tool(tmp_path, "pytool", 'echo "1 passed in 0.10s"')
    _test_tool(tmp_path, "jstest", 'echo "Tests: 2 passed, 2 total"')
    whole = edit.run_test()
    baseline = testcreate.baseline_of(whole)
    assert baseline == {edit.baseline_mark("green before", "pytool"),
                        edit.baseline_mark("green before", "jstest"),
                        "\x1fran count\x1fpytool\x1f1", "\x1fran count\x1fjstest\x1f2"}
    own = edit.run_test(["tests/test_a.py"])
    assert whole.is_whole is True and own.is_whole is False
    assert list(edit.part_dict_of(own)) == ["pytool"]
    assert testcreate.run_reason(own, baseline) == ""
    assert integrate._test_reason(own, baseline) == ""
    # A whole run the other command is gone from says so
    (_two_language_repo / "web" / "package.json").unlink()
    gone = edit.run_test()
    assert list(edit.part_dict_of(gone)) == ["pytool"]
    assert testcreate.run_reason(gone, baseline).startswith("`jstest` did not run")
    # A file is looked for in the run of the command a call naming it hands it to
    (_two_language_repo / "web" / "package.json").write_text("{}\n")
    # A whole run in which a command runs fewer tests than it did says so
    _test_tool(tmp_path, "jstest", 'echo "Tests: 1 skipped, 1 passed, 2 total"')
    assert testcreate.run_reason(edit.run_test(), baseline) == (
        "`jstest` runs 1 of the 2 tests it ran before you started: the others are not "
        "found, are left out or skip now")
    assert edit._test_command_name("web/test/view.test.js") == "jstest"
    assert edit._test_command_name("tests/test_a.py") == "pytool"
    assert edit.is_other_test_path("web/test/view.test.js") is True
    assert edit.is_other_test_path("tests/test_a.py") is False
    # One outside the directory that command is started from goes to the primary one
    assert edit.is_other_test_path("tools/x.test.js") is False
    # and so does every file where the other command cannot be run
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [("js", ["no-such-test-tool"])])
    assert edit.is_other_test_path("web/test/view.test.js") is False


def test_the_baselines_of_several_commands_are_one_set():
    assert edit.join_baseline({"a": None, "b": None}) is None
    # The primary command alone is its names, as it always was
    assert edit.join_baseline({"a": frozenset({"t1"})}, "a") == {"t1"}
    assert edit.join_baseline({"a": frozenset({"t1"})}) == {
        "t1", edit.baseline_mark("red before", "a")}
    assert edit.join_baseline({"a": None}) is None
    assert edit.not_compared_list(None) == [] == edit.not_compared_list(frozenset({"t"}))
    joined = edit.join_baseline({"a": frozenset({"t1"}), "b": None, "c": frozenset()})
    assert joined == {"t1", edit.baseline_mark("red before", "a"),
                      edit.baseline_mark("not compared", "b"),
                      edit.baseline_mark("green before", "c")}
    # It goes between containers as a list of text and comes back as it was
    assert frozenset(json.loads(json.dumps(sorted(joined)))) == joined
    assert edit.baseline_name_list(joined) == ["t1"]
    assert edit.baseline_name_list(None) == []
    result = edit.RunResult({"status": "failed"})
    result.part_dict = {name: edit.RunResult({"status": name}) for name in "abc"}
    assert [(command, part["status"], has_name) for command, part, has_name
            in edit.compare_part_list(result, joined)] == [("a", "a", True),
                                                           ("c", "c", False)]
    assert edit.not_compared_list(joined) == ["b"]
    # A command of the baseline that a run given paths did not run is passed over;
    # in a run given none it is one that did not run
    del result.part_dict["c"]
    assert [command for command, _, _ in edit.compare_part_list(result, joined)] == ["a"]
    result.is_whole = True
    assert [(command, part["status"]) for command, part, _
            in edit.compare_part_list(result, joined)] == [("a", "a"),
                                                           ("c", "did not run")]
    # A baseline of one command, and a dict with no parts, are compared whole
    assert edit.compare_part_list(dict(result), frozenset({"t1"})) == [
        (edit.test_command_text(), {"status": "failed"}, True)]


def test_a_command_that_found_no_test_before_is_held_only_where_it_finds_one_now(
        monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND", ["a"])
    joined = edit.join_baseline({"a": frozenset(), "b": edit.NO_TEST_BASELINE}, "a")
    assert joined == {edit.baseline_mark("green before", "a"),
                      edit.baseline_mark("no test before", "b")}
    assert edit.baseline_name_list(joined) == [] and edit.not_compared_list(joined) == []
    assert edit.join_baseline({"a": edit.NO_TEST_BASELINE}, "a") == edit.NO_TEST_BASELINE
    zero, red = edit.RunResult({"status": "did not run"}), edit.RunResult({"status": "failed"})
    zero.is_no_test = True

    def run_of(**part_dict):
        result = edit.RunResult({"status": "did not run"})
        result.part_dict = part_dict
        result.primary_name = "a"
        result.is_whole = True
        return result

    held = [(command, part["status"], has_name) for command, part, has_name
            in edit.compare_part_list(run_of(a=zero, b=zero), joined)]
    # Tests that ran before and find none now are held; none before and none now are not
    assert held == [("a", "did not run", False)]
    assert [command for command, _, _
            in edit.compare_part_list(run_of(a=zero, b=red), joined)] == ["a", "b"]
    # The primary command's baseline of one command, and the stage that holds it anyway
    assert edit.compare_part_list(run_of(a=zero), edit.NO_TEST_BASELINE) == []
    assert [command for command, _, _ in edit.compare_part_list(
        run_of(a=zero), edit.NO_TEST_BASELINE, hold_primary_no_test=True)] == ["a"]
    assert edit.compare_part_list(run_of(b=zero), joined, hold_primary_no_test=True) == [
        ("a", edit.RunResult(edit._NO_PART_DICT), False)]


def test_a_pytest_run_names_the_tests_it_ran_where_its_summary_counts_as_many(
        as_language, fake_test_command, monkeypatch):
    """The test command is told to name each test that passed; the names are the
    passed and the failing ones, and the model is shown the output without the former."""
    as_language("py")
    fake_test_command(1, _SUMMARY_HEAD + "PASSED tests/t.py::test_a\n"
                         "PASSED tests/t.py::test_b[1 2]\n"
                         "FAILED tests/u.py::test_c - assert 0\nERROR tests/v.py\n"
                         "SUBFAILED(i=1) tests/u.py::T::test_d - x\n"
                         "2 failed, 2 passed, 1 skipped, 1 xfailed, 1 error in 0.01s")
    result = edit.run_test()
    part = edit.primary_part(result)
    assert part.ran_name_list == ["tests/t.py::test_a", "tests/t.py::test_b[1 2]",
                                  "tests/u.py::T::test_d", "tests/u.py::test_c"]
    # passed, failed, xfailed and error, less the failing subtest the summary counts
    assert part.ran_count == 5
    assert "PASSED" not in result["output"] and "FAILED tests/u.py" in result["output"]
    # A file whose path holds a space, a message that holds "::" and " - "
    fake_test_command(1, _SUMMARY_HEAD + "PASSED tests/sub dir/t.py::test_a\n"
                         "PASSED tests/sub dir/t.py::test_b[a - b]\n"
                         "FAILED tests/sub dir/u.py::T::test_c - assert a::b - c\n"
                         "1 failed, 2 passed in 0.01s")
    result = edit.run_test()
    assert edit.primary_part(result).ran_name_list == [
        "tests/sub dir/t.py::test_a", "tests/sub dir/t.py::test_b[a - b]",
        "tests/sub dir/u.py::T::test_c"]
    assert result["failed"] == ["tests/sub dir/u.py::T::test_c"]
    # A file whose path holds " - ", and a PASSED line whose id is the rest of the line:
    # a path that holds " - " and ":", an item whose name holds spaces, a doctest file
    # whose name holds a space
    fake_test_command(1, _SUMMARY_HEAD + "PASSED tests/basics - part1/t.py::test_a\n"
                         "FAILED tests/basics - part1/u.py::test_b - assert 0\n"
                         "1 failed, 1 passed in 0.01s")
    result = edit.run_test()
    assert edit.primary_part(result).ran_name_list == [
        "tests/basics - part1/t.py::test_a", "tests/basics - part1/u.py::test_b"]
    assert result["failed"] == ["tests/basics - part1/u.py::test_b"]
    fake_test_command(0, "PASSED tests/a:b - c/t.py::test_a\n"
                         "PASSED tests/test_api.yaml::Get user by id\n"
                         "PASSED tests/user guide.txt::user guide.txt\n3 passed in 0.01s")
    result = edit.run_test()
    part = edit.primary_part(result)
    assert (part.ran_name_list, part.ran_count, part.is_name_short) == (
        ["tests/a:b - c/t.py::test_a", "tests/test_api.yaml::Get user by id",
         "tests/user guide.txt::user guide.txt"], 3, False)
    assert "PASSED" not in result["output"]
    # A log line a failing test wrote, whose message ends in a word holding "::"
    fake_test_command(1, _SUMMARY_HEAD + "PASSED tests/t.py::test_a\n"
                         "FAILED tests/t.py::test_b - assert 0\n"
                         "ERROR    app:lib.py:4 lookup failed for db::main\n"
                         "1 failed, 1 passed in 0.01s")
    result = edit.run_test()
    assert edit.primary_part(result).ran_name_list == [
        "tests/t.py::test_a", "tests/t.py::test_b"]
    assert result["failed"] == ["tests/t.py::test_b"]
    # Fewer lines than the summary counts as passed, or more: the count, and a run that
    # names some of the tests it ran, not every one
    fake_test_command(0, "PASSED tests/t.py::test_a\n2 passed in 0.01s")
    part = edit.primary_part(edit.run_test())
    assert (part.ran_name_list, part.ran_count, part.is_name_short) == (None, 2, True)
    fake_test_command(0, "PASSED tests/t.py::test_a\nPASSED tests/t.py::test_b\n"
                         "1 passed in 0.01s")
    part = edit.primary_part(edit.run_test())
    assert (part.ran_name_list, part.ran_count, part.is_name_short) == (None, 1, True)
    # No PASSED line at all, as where the command line's -r takes the place of -rap: the
    # count alone
    fake_test_command(0, "2 passed in 0.01s")
    part = edit.primary_part(edit.run_test())
    assert (part.ran_name_list, part.ran_count, part.is_name_short) == (None, 2, False)
    # A red run that counts failing tests and names none of them, with PASSED lines and
    # without; one that names one of them
    for output in ("PASSED tests/t.py::test_a\n1 failed, 1 passed in 0.01s",
                   "1 failed, 1 passed in 0.01s", "1 passed, 1 error in 0.01s"):
        fake_test_command(1, output)
        assert edit.primary_part(edit.run_test()).is_name_short is True
    fake_test_command(1, _SUMMARY_HEAD + "PASSED tests/t.py::test_a\n"
                         "FAILED tests/t.py::test_b - x\n2 failed, 1 passed in 0.01s")
    assert edit.primary_part(edit.run_test()).is_name_short is False
    # ... one whose failing test's line is long (-vv, CI)
    fake_test_command(1, _SUMMARY_HEAD + "PASSED tests/t.py::test_a\n"
                         "FAILED tests/t.py::test_b - assert "
                         + "x" * 3000 + "\n1 failed, 1 passed in 0.01s")
    result = edit.run_test()
    part = edit.primary_part(result)
    assert (result["failed"], part.is_name_short, part.ran_name_list) == (
        ["tests/t.py::test_b"], False, ["tests/t.py::test_a", "tests/t.py::test_b"])
    # unittest's count: what it ran less what it skipped
    fake_test_command(0, "Ran 3 tests in 0.001s\n\nOK (skipped=1)")
    part = edit.primary_part(edit.run_test())
    assert (part.ran_name_list, part.ran_count, part.is_name_short) == (None, 2, False)
    # A run given paths, and one that ended early, says neither
    fake_test_command(1, _SUMMARY_HEAD + "PASSED tests/t.py::test_a\n"
                         "FAILED tests/u.py::test_c - x\n1 failed, 1 passed in 0.01s")
    part = edit.primary_part(edit.run_test(["m.py"]))
    assert (part.ran_name_list, part.ran_count, part.is_ran_hold) == (None, None, False)
    fake_test_command(0, "PASSED tests/t.py::test_a\n2 passed in 0.01s")
    part = edit.primary_part(edit.run_test(["m.py"]))
    assert (part.ran_count, part.is_name_short, part.is_ran_hold) == (None, False, False)
    # ... but one that is red and names no failing test still says so
    fake_test_command(1, "PASSED tests/t.py::test_a\n1 failed, 1 passed in 0.01s")
    part = edit.primary_part(edit.run_test(["m.py"]))
    assert (part.ran_count, part.is_name_short, part.is_ran_hold) == (None, True, False)
    fake_test_command(1, _SUMMARY_HEAD + "FAILED tests/u.py::test_c - x\n"
                         "!!!!! stopping after 1 failures !!!!!\n1 failed in 0.01s")
    part = edit.primary_part(edit.run_test())
    assert (part.ran_name_list, part.ran_count, part.is_ran_hold) == (None, None, False)
    assert part.is_name_short is False
    # ... but one that ended early and names no failing test it counts says so, given
    # paths or not
    fake_test_command(1, "!!!!! stopping after 1 failures !!!!!\n1 failed, 1 passed in 0.01s")
    for path_list in (None, ["m.py"]):
        part = edit.primary_part(edit.run_test(path_list))
        assert (edit.early_stop_run(part), part.ran_count, part.is_name_short,
                part.is_ran_hold) == (True, None, True, False)
    # A whole run that reached a verdict and says neither: the tests were read off it
    fake_test_command(0, ".  [100%]")
    part = edit.primary_part(edit.run_test())
    assert (part.ran_name_list, part.ran_count, part.is_ran_hold) == (None, None, True)


@pytest.mark.parametrize("stop_line", ["!!!!! Interrupted: done !!!!!",
                                       "!!!!! stopping after 1 failures !!!!!"])
def test_a_run_ending_with_status_0_is_held_to_the_tests_it_ran_whatever_stop_line_it_prints(
        stop_line, repo, as_language, fake_test_command):
    """A whole run with exit code 0 that prints a line saying it ended early is held to the
    tests the run before ran: a test that runs no more is a reason, and so is a run that
    prints no summary line. A red run that ended early with a non-zero exit code is not
    held."""
    as_language("py")
    for path in ("tests/t.py", "tests/u.py"):
        (repo / path).write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "tests")
    fake_test_command(0, _SUMMARY_HEAD + "PASSED tests/t.py::test_a\n"
                         "PASSED tests/u.py::test_c\n2 passed in 0.01s")
    baseline = edit.ran_entry_set(edit.primary_part(edit.run_test()))
    reason_tuple = ("{command} {tests}", "{command} {now}/{before}", "{command} says less",
                    "{command} names some")
    fake_test_command(0, _SUMMARY_HEAD + "PASSED tests/t.py::test_a\n" + stop_line
                      + "\n1 passed in 0.01s")
    part = edit.primary_part(edit.run_test())
    assert (part["status"], edit.early_stop_run(part), part.is_ran_hold, part.ran_count,
            part.ran_name_list) == ("passed", True, True, 1, ["tests/t.py::test_a"])
    assert edit.unrun_reason(part, baseline, "t", *reason_tuple) == "t tests/u.py::test_c"
    fake_test_command(0, ".   [100%]\n" + _SUMMARY_HEAD + stop_line)
    part = edit.primary_part(edit.run_test())
    assert (part["status"], part.is_ran_hold, part.ran_count, part.is_ran_unread) == (
        "passed", True, None, True)
    assert edit.unrun_reason(part, baseline, "t", *reason_tuple) == "t says less"
    fake_test_command(1, _SUMMARY_HEAD + "FAILED tests/t.py::test_a - x\n" + stop_line
                      + "\n1 failed in 0.01s")
    part = edit.primary_part(edit.run_test())
    assert (part["status"], edit.early_stop_run(part), part.is_ran_hold) == (
        "failed", True, False)


def _sep(char: str, title: str) -> str:
    """Return a line pytest writes a title between."""
    return f"{char * 9} {title} {char * 9}"


# A session of pytest that a test ran, as it prints it with and without -q
_INNER_RUN = "\n".join([
    _sep("=", "test session starts"), "collected 1 item", "",
    "test_reports.py F                                     [100%]", "",
    _sep("=", "FAILURES"), _sep("_", "test_inner"), ">       assert 0", "E       assert 0", "",
    _sep("=", "short test summary info"), "FAILED test_reports.py::test_inner - assert 0",
    _sep("=", "1 failed in 0.02s")])
_INNER_QUIET_RUN = "\n".join([
    "F                                                     [100%]",
    _sep("=", "short test summary info"), "PASSED test_gen.py::test_two",
    "FAILED test_gen.py::test_add - assert -1 == 3", "1 failed, 1 passed in 0.01s"])
_OUTER_TAIL = "\n".join([
    _sep("=", "short test summary info"),
    "FAILED tests/test_plugin.py::test_reports - AssertionError: assert 1",
    "PASSED tests/test_plugin.py::test_other"])
_OUTER_NAME_LIST = ["tests/test_plugin.py::test_other", "tests/test_plugin.py::test_reports"]


@pytest.mark.parametrize("exit_code, output, status, failed, count, name_list", [
    # A failing test's output the run took, under -q, holding a session that fails
    (1, "\n".join([".F                                  [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_reports"), "E       AssertionError: assert 1", "",
                   _sep("-", "Captured stdout call"), _INNER_RUN, _OUTER_TAIL,
                   "1 failed, 1 passed in 0.05s"]),
     "failed", ["tests/test_plugin.py::test_reports"], 2, _OUTER_NAME_LIST),
    # The same without -q, the session the test ran printing under -q
    (1, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   "tests/test_plugin.py F.                        [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_reports"),
                   _sep("-", "Captured stdout call"), _INNER_QUIET_RUN, "",
                   _OUTER_TAIL, _sep("=", "1 failed, 1 passed in 0.05s")]),
     "failed", ["tests/test_plugin.py::test_reports"], 2, _OUTER_NAME_LIST),
    # -s and -q: the session a passing test ran is printed among the run's progress
    (0, "\n".join(["." + _INNER_RUN, ".", _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other",
                   "PASSED tests/test_plugin.py::test_reports", "2 passed in 0.05s"]),
     "passed", [], 2, _OUTER_NAME_LIST),
    # A command that runs pytest twice: both runs are read
    (1, "\n".join([".                                   [100%]",
                   _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other", "1 passed in 0.19s",
                   "F                                   [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_m"), _sep("=", "short test summary info"),
                   "FAILED tests/test_more.py::test_m - assert 0", "1 failed in 0.01s"]),
     "failed", ["tests/test_more.py::test_m"], 2,
     ["tests/test_more.py::test_m", "tests/test_plugin.py::test_other"]),
    (0, "\n".join([".                                   [100%]",
                   _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other", "1 passed in 0.19s",
                   "no tests ran in 0.01s"]),
     "passed", [], 1, ["tests/test_plugin.py::test_other"]),
    # The same, neither run printing its progress as a share (-s), the second one a
    # section before its short summary
    (1, "\n".join([".", _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other", "1 passed in 0.19s", "F",
                   _sep("=", "FAILURES"), _sep("_", "test_m"),
                   _sep("=", "short test summary info"),
                   "FAILED tests/test_more.py::test_m - assert 0", "1 failed in 0.01s"]),
     "failed", ["tests/test_more.py::test_m"], 2,
     ["tests/test_more.py::test_m", "tests/test_plugin.py::test_other"]),
    # The first run took a failing test's output, the second one prints no share
    (1, "\n".join(["F                                   [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_m"), _sep("-", "Captured stdout call"), "out",
                   _sep("=", "short test summary info"),
                   "FAILED tests/test_more.py::test_m - assert 0", "1 failed in 0.01s", ".",
                   _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other", "1 passed in 0.19s"]),
     "failed", ["tests/test_more.py::test_m"], 2,
     ["tests/test_more.py::test_m", "tests/test_plugin.py::test_other"]),
    # -vv: a failing test's message in the short summary holds a session a test ran,
    # with its start line, under -q with the head of its short summary, and under -q
    # with neither
    (1, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   _sep("=", "short test summary info"),
                   "FAILED tests/test_plugin.py::test_reports - RuntimeError: "
                   + _sep("=", "test session starts"), "collected 0 items", "",
                   _sep("=", "no tests ran in 0.00s"),
                   "PASSED tests/test_plugin.py::test_other",
                   _sep("=", "1 failed, 1 passed in 0.05s")]),
     "failed", ["tests/test_plugin.py::test_reports"], 2, _OUTER_NAME_LIST),
    (1, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   _sep("=", "short test summary info"),
                   "FAILED tests/test_plugin.py::test_reports - RuntimeError: F   [100%]",
                   _sep("=", "short test summary info"),
                   "FAILED test_reports.py::test_inner - assert 0", "1 failed in 0.02s",
                   "PASSED tests/test_plugin.py::test_other",
                   _sep("=", "1 failed, 1 passed in 0.05s")]),
     "failed", ["tests/test_plugin.py::test_reports"], 2, _OUTER_NAME_LIST),
    (1, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   _sep("=", "short test summary info"),
                   "FAILED tests/test_plugin.py::test_reports - RuntimeError: ",
                   "no tests ran in 0.00s", "PASSED tests/test_plugin.py::test_other",
                   _sep("=", "1 failed, 1 passed in 0.05s")]),
     "failed", ["tests/test_plugin.py::test_reports"], 2, _OUTER_NAME_LIST),
    # A command that runs pytest twice, the first run printing its start line
    (1, "\n".join([_sep("=", "test session starts"), "collected 1 item", "",
                   "tests/test_plugin.py .                         [100%]", "",
                   _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other", _sep("=", "1 passed in 0.19s"),
                   "F", _sep("=", "FAILURES"), _sep("_", "test_m"),
                   _sep("=", "short test summary info"),
                   "FAILED tests/test_more.py::test_m - assert 0", "1 failed in 0.01s"]),
     "failed", ["tests/test_more.py::test_m"], 2,
     ["tests/test_more.py::test_m", "tests/test_plugin.py::test_other"]),
    # What the run took of a failing test's output holds, under -q, a session that
    # prints no line between "=" or "_", and two sessions with their start lines
    (1, "\n".join([".F                                  [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_reports"), _sep("-", "Captured stdout call"),
                   "..                                  [100%]", "2 passed in 0.01s",
                   _OUTER_TAIL, "1 failed, 1 passed in 0.05s"]),
     "failed", ["tests/test_plugin.py::test_reports"], 2, _OUTER_NAME_LIST),
    (1, "\n".join([".F                                  [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_reports"), _sep("-", "Captured stdout call"),
                   _INNER_RUN, _INNER_RUN, _OUTER_TAIL, "1 failed, 1 passed in 0.05s"]),
     "failed", ["tests/test_plugin.py::test_reports"], 2, _OUTER_NAME_LIST),
    # ... a session under -q that stopped at its first failing test, with the head of its
    # short summary: its summary is read as the run's, but not its line of "!"
    (1, "\n".join([".F                                  [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_reports"), _sep("-", "Captured stdout call"),
                   "F                                   [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_inner"), _sep("=", "short test summary info"),
                   "FAILED test_reports.py::test_inner - assert 0",
                   _sep("!", "stopping after 1 failures"), "1 failed in 0.02s",
                   "checked the inner run", _OUTER_TAIL, "1 failed, 1 passed in 0.05s"]),
     "failed", ["test_reports.py::test_inner", "tests/test_plugin.py::test_reports"], 3,
     ["test_reports.py::test_inner", *_OUTER_NAME_LIST]),
    # A command that runs pytest twice, a failing test of the first run printing the start
    # of a session that printed no summary: in what the run took of its output, and
    # among the run's progress (-s)
    (0, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   "tests/test_more.py F.                          [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_m"), "E       assert 0", "",
                   _sep("-", "Captured stdout call"), _sep("=", "test session starts"),
                   "collected 3 items", "", "test_in.py .",
                   _sep("=", "short test summary info"),
                   "FAILED tests/test_more.py::test_m - assert 0",
                   "PASSED tests/test_more.py::test_n", _sep("=", "1 failed, 1 passed in 0.19s"),
                   _sep("=", "test session starts"), "collected 1 item", "",
                   "tests/test_plugin.py .                         [100%]", "",
                   _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other", _sep("=", "1 passed in 0.01s")]),
     "failed", ["tests/test_more.py::test_m"], 3,
     ["tests/test_more.py::test_m", "tests/test_more.py::test_n",
      "tests/test_plugin.py::test_other"]),
    (0, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   "tests/test_more.py " + _sep("=", "test session starts"),
                   "collected 3 items", "", "test_in.py .F.", "",
                   _sep("=", "FAILURES"), _sep("_", "test_m"), "E       assert 0", "",
                   _sep("=", "short test summary info"),
                   "FAILED tests/test_more.py::test_m - assert 0",
                   "PASSED tests/test_more.py::test_n", _sep("=", "1 failed, 1 passed in 0.19s"),
                   _sep("=", "test session starts"), "collected 1 item", "",
                   "tests/test_plugin.py .                         [100%]", "",
                   _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other", _sep("=", "1 passed in 0.01s")]),
     "failed", ["tests/test_more.py::test_m"], 3,
     ["tests/test_more.py::test_m", "tests/test_more.py::test_n",
      "tests/test_plugin.py::test_other"]),
    # ... a failing test of the first run printing the end of a session with no start
    # line before the one that prints a start line with no end line, and the second run
    # printing the end of a session with no start line
    (0, "\n".join([_sep("=", "test session starts"), "collected 3 items", "",
                   "tests/test_more.py FF.                         [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_z"), "E       assert 0", "",
                   _sep("-", "Captured stdout call"), "test_in.py .F", "",
                   _sep("=", "1 failed, 1 passed in 0.12s"),
                   _sep("_", "test_m"), "E       assert 0", "",
                   _sep("-", "Captured stdout call"), _sep("=", "test session starts"),
                   "collected 3 items", "", "test_in.py .",
                   _sep("=", "short test summary info"),
                   "FAILED tests/test_more.py::test_z - assert 0",
                   "FAILED tests/test_more.py::test_m - assert 0",
                   "PASSED tests/test_more.py::test_n", _sep("=", "2 failed, 1 passed in 0.19s"),
                   _sep("=", "test session starts"), "collected 1 item", "",
                   "tests/test_plugin.py .                         [100%]", "",
                   _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other", _sep("=", "1 passed in 0.01s")]),
     "failed", ["tests/test_more.py::test_z", "tests/test_more.py::test_m"], 4,
     ["tests/test_more.py::test_m", "tests/test_more.py::test_n",
      "tests/test_more.py::test_z", "tests/test_plugin.py::test_other"]),
    (0, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   "tests/test_more.py F.                          [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_m"), "E       assert 0", "",
                   _sep("-", "Captured stdout call"), _sep("=", "test session starts"),
                   "collected 3 items", "", "test_in.py .",
                   _sep("=", "short test summary info"),
                   "FAILED tests/test_more.py::test_m - assert 0",
                   "PASSED tests/test_more.py::test_n", _sep("=", "1 failed, 1 passed in 0.19s"),
                   _sep("=", "test session starts"), "collected 2 items", "",
                   "tests/test_plugin.py .F                        [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_reports"), "E       assert 0", "",
                   _sep("-", "Captured stdout call"), "test_in.py .F", "",
                   _sep("=", "1 failed, 1 passed in 0.12s"), _OUTER_TAIL,
                   _sep("=", "1 failed, 1 passed in 0.01s")]),
     "failed", ["tests/test_more.py::test_m", "tests/test_plugin.py::test_reports"], 4,
     ["tests/test_more.py::test_m", "tests/test_more.py::test_n", *_OUTER_NAME_LIST]),
    # ... the first run printing no short summary (-rN), a failing test's output it took
    # ending right before its summary line; the second run printing its start line, not
    # (-q), and not with no failing test
    (0, "\n".join([_sep("=", "test session starts"), "collected 1 item", "",
                   "tests/test_more.py F                           [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_m"), "E       assert 0", "",
                   _sep("-", "Captured stdout call"), "out", _sep("=", "1 failed in 0.19s"),
                   _sep("=", "test session starts"), "collected 1 item", "",
                   "tests/test_plugin.py .                         [100%]", "",
                   _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other", _sep("=", "1 passed in 0.01s")]),
     "failed", [], 2, ["tests/test_plugin.py::test_other"]),
    (1, "\n".join([_sep("=", "test session starts"), "collected 1 item", "",
                   "tests/test_more.py F                           [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_m"), "E       assert 0", "",
                   _sep("-", "Captured stdout call"), "out", _sep("=", "1 failed in 0.19s"),
                   ".F                                  [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_reports"), "E       assert 0", _OUTER_TAIL,
                   "1 failed, 1 passed in 0.01s"]),
     "failed", ["tests/test_plugin.py::test_reports"], 3, _OUTER_NAME_LIST),
    (0, "\n".join([_sep("=", "test session starts"), "collected 1 item", "",
                   "tests/test_more.py F                           [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_m"), "E       assert 0", "",
                   _sep("-", "Captured stdout call"), "out", _sep("=", "1 failed in 0.19s"),
                   ".                                   [100%]",
                   _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other", "1 passed in 0.01s"]),
     "failed", [], 2, ["tests/test_plugin.py::test_other"]),
    # One run: what it took of a failing test's output holds a whole session, and the next
    # failing test's message holds a start line, its output another whole session
    (1, "\n".join([_sep("=", "test session starts"), "collected 3 items", "",
                   "tests/test_plugin.py .FF                       [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_reports"), "E       assert 1 == 0", "",
                   _sep("-", "Captured stdout call"), _INNER_RUN, _sep("_", "test_other"),
                   "E       AssertionError: assert '===' == 'nothing'",
                   "E         + " + _sep("=", "test session starts"), "",
                   _sep("-", "Captured stdout call"), _sep("=", "test session starts"),
                   "collected 1 item", "", "test_one.py .                    [100%]", "",
                   _sep("=", "1 passed in 0.00s"), _sep("=", "short test summary info"),
                   "FAILED tests/test_plugin.py::test_reports - assert 1 == 0",
                   "FAILED tests/test_plugin.py::test_other - AssertionError",
                   "PASSED tests/test_plugin.py::test_add", _sep("=", "2 failed, 1 passed in 0.07s")]),
     "failed", ["tests/test_plugin.py::test_reports", "tests/test_plugin.py::test_other"], 3,
     ["tests/test_plugin.py::test_add", *_OUTER_NAME_LIST]),
    # ... what it took of a failing test's output, and of a passing one's (-rA), ends in
    # the summary line of a session with no start line
    (1, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   "tests/test_plugin.py F.                        [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_reports"), "E       assert 0", "",
                   _sep("-", "Captured stdout call"), "test_in.py .F", "",
                   _sep("=", "1 failed, 1 passed in 0.12s"), _OUTER_TAIL,
                   _sep("=", "1 failed, 1 passed in 0.05s")]),
     "failed", ["tests/test_plugin.py::test_reports"], 2, _OUTER_NAME_LIST),
    (0, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   "tests/test_plugin.py ..                        [100%]", "",
                   _sep("=", "PASSES"), _sep("_", "test_reports"),
                   _sep("-", "Captured stdout call"), "test_in.py .F", "",
                   _sep("=", "1 failed, 1 passed in 0.12s"), _sep("=", "short test summary info"),
                   "PASSED tests/test_plugin.py::test_other",
                   "PASSED tests/test_plugin.py::test_reports", _sep("=", "2 passed in 0.05s")]),
     "passed", [], 2, _OUTER_NAME_LIST),
    # A command that runs pytest twice, the first run printing no short summary (-rN), a
    # failing test's output it took ending right before its summary line; the second
    # run under -q with no short summary: the first run printing its start line, and not
    # with a log line taken
    (0, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   "tests/a/test_a.py F.                           [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_add"), "E       assert -1 == 3", "",
                   _sep("-", "Captured stdout call"), "debug: adding",
                   _sep("=", "1 failed, 1 passed in 0.02s"),
                   "..                                  [100%]", "2 passed in 0.01s"]),
     "failed", [], 4, None),
    (0, "\n".join(["F.                                  [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_add"), "E       assert -1 == 3", "",
                   _sep("-", "Captured log call"), "WARNING  t:test_a.py:4 adding",
                   "1 failed, 1 passed in 0.02s",
                   "..                                  [100%]", "2 passed in 0.01s"]),
     "failed", [], 4, None),
    # ... the second run printing its short summary, right after the first one's summary
    # line and after a line the command printed between the runs, with its start line and
    # without
    (0, "\n".join(["F.                                  [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_add"), "E       assert -1 == 3", "",
                   _sep("-", "Captured stdout call"), "debug: adding",
                   "1 failed, 1 passed in 0.02s",
                   "..                                  [100%]",
                   _sep("=", "short test summary info"), "PASSED tests/b/test_b.py::test_b",
                   "PASSED tests/b/test_b.py::test_new", "2 passed in 0.01s"]),
     "failed", [], 4, None),
    (0, "\n".join([_sep("=", "test session starts"), "collected 2 items", "",
                   "tests/a/test_a.py F.                           [100%]", "",
                   _sep("=", "FAILURES"), _sep("_", "test_add"), "E       assert -1 == 3", "",
                   _sep("-", "Captured stdout call"), "debug: adding",
                   _sep("=", "1 failed, 1 passed in 0.02s"), "done",
                   _sep("=", "test session starts"), "collected 2 items", "",
                   "tests/b/test_b.py ..                           [100%]", "",
                   _sep("=", "short test summary info"), "PASSED tests/b/test_b.py::test_b",
                   "PASSED tests/b/test_b.py::test_new", _sep("=", "2 passed in 0.01s")]),
     "failed", [], 4, None),
    (0, "\n".join(["F.                                  [100%]", _sep("=", "FAILURES"),
                   _sep("_", "test_add"), "E       assert -1 == 3", "",
                   _sep("-", "Captured stdout call"), "debug: adding",
                   "1 failed, 1 passed in 0.02s", "done",
                   "..                                  [100%]",
                   _sep("=", "short test summary info"), "PASSED tests/b/test_b.py::test_b",
                   "PASSED tests/b/test_b.py::test_new", "2 passed in 0.01s"]),
     "failed", [], 4, None),
    # unittest, a test of which printed a session of pytest it ran, and lines that count
    # some of something in a time
    (0, "\n".join([_INNER_RUN, "3 passed in 0.50s", "200 rows in 0.01s", "..",
                   "-" * 70, "Ran 2 tests in 0.010s", "", "OK"]),
     "passed", [], 2, None)],
    ids=["captured", "captured_quiet", "not_captured", "twice", "twice_none",
         "twice_no_share", "twice_captured", "message_session", "message_quiet_head",
         "message_quiet", "twice_first_session", "captured_bare", "captured_sessions",
         "captured_stop", "twice_captured_cut", "twice_printed_cut", "twice_captured_tail",
         "twice_captured_tail_after", "twice_unsummarized", "twice_unsummarized_quiet",
         "twice_unsummarized_green", "captured_and_message_session",
         "captured_tail", "passes_tail", "twice_unsummarized_quiet_tail",
         "twice_quiet_log_tail", "twice_quiet_head_tail", "twice_printed_between_tail",
         "twice_quiet_printed_between_tail", "unittest"])
def test_a_python_run_is_read_by_its_own_report_and_not_by_a_session_a_test_ran(
        exit_code, output, status, failed, count, name_list, as_language,
        fake_test_command):
    """The failing names, the verdict and the tests that ran are those of the run's own
    summaries, not those of a session of pytest a test ran and printed."""
    as_language("py")
    fake_test_command(exit_code, output)
    result = edit.run_test()
    part = edit.primary_part(result)
    assert (result["status"], result["failed"]) == (status, failed)
    assert (part.ran_count, part.ran_name_list) == (count, name_list)
    assert edit.early_stop_run(result) is False and edit.no_test_run(result) is False


def test_a_pytest_run_is_told_in_its_environment_to_name_the_tests_that_pass(
        as_language, monkeypatch):
    as_language("py")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-x")
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "import os; print(os.environ['PYTEST_ADDOPTS'])"])
    assert edit.run_test()["output"].strip() == \
        "-x --verbosity=0 -rap --continue-on-collection-errors"
    # Another language's command is not
    monkeypatch.setattr(config, "PRIMARY_EXT", "rs")
    assert edit.run_test()["output"].strip() == "-x"


@pytest.mark.parametrize("addopts", ["-q", "-ra -q", "-qq", "-v"])
def test_a_pytest_run_s_verbosity_is_its_command_line_s_whatever_its_ini_file_adds(
        addopts, repo, monkeypatch):
    """A -q or -v of the ini file's addopts is undone: the run with the command line's -q
    prints its summary line, so the tests it ran are counted and named, and a run over
    no test says so."""
    (repo / "pytest.ini").write_text(f"[pytest]\naddopts = {addopts}\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_a.py").write_text(
        "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n")
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "LLM_BROKER", "")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setattr(config, "TEST_COMMAND",
                        [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    result = edit.run_test()
    part = edit.primary_part(result)
    assert result["status"] == "passed"
    assert "2 passed in " in result["output"]
    assert (part.ran_count, part.ran_name_list) == (
        2, ["tests/test_a.py::test_a", "tests/test_a.py::test_b"])
    (repo / "tests" / "test_a.py").unlink()
    assert edit.no_test_run(edit.run_test()) is True


@pytest.mark.parametrize("exit_code, output, status", [
    # Its short summary, PASSED lines among it, and no summary line (-qq)
    (1, _SUMMARY_HEAD + "FAILED tests/t.py::test_f - assert False\n"
                        "PASSED tests/t.py::test_a\nPASSED tests/t.py::test_b", "failed"),
    (0, ".s.  [100%]\n" + _SUMMARY_HEAD + "SKIPPED [1] tests/t.py:9: no\n"
                                          "PASSED tests/t.py::test_a", "passed"),
    # The line starting the session, a line of the progress alone
    (0, "===== test session starts =====\nplatform linux\n", "passed"),
    (0, "..   [2/2]", "passed"),
    (0, "........ [ 48%]\n.....", "passed"),
    # ... with the letters of subtests (pytest 9: u, y, -; pytest-subtests: ,) and of a
    # test run again (R)
    (0, "..uuu...   [100%]", "passed"),
    (1, "uuu.F   [100%]", "did not run"),
    (1, "---.F   [100%]", "did not run"),
    (1, "yyy.F   [100%]", "did not run"),
    (0, ",,,.   [100%]", "passed"),
    (0, ".R.   [100%]", "passed"),
    # Ending with status 1, at the last test or before it (the process ended in the
    # middle of the tests)
    (1, "===== test session starts =====\nplatform linux\n\ntests/t.py .F   [100%]\n",
     "did not run"),
    (1, "===== test session starts =====\n\ntests/t.py::test_a PASSED   [ 50%]\n"
        "tests/t.py::test_b FAILED   [100%]\n", "did not run"),
    (1, ".F   [2/2]", "did not run"),
    (1, ".......... [ 90%]\n.F   [100%]", "did not run"),
    (1, "........ [ 48%]\n.....", "did not run"),
    (1, "..   [2/5]\n.", "did not run"),
    (1, "===== test session starts =====\nplatform linux\n\ntests/t.py ", "did not run"),
    (1, "===== test session starts =====\n\ntests/t.py ........   [ 37%]\n......",
     "did not run"),
    (1, "===== test session starts =====\n\ntests/t.py::test_a PASSED   [ 50%]\n",
     "did not run"),
    # Ending with status 1 where pytest stopped after a count of failing tests (-x,
    # --maxfail), with no other line of its report
    (1, "F\n!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!\n",
     "did not run"),
    (1, "..F.F\n!!!!!!!!!!!!!!!!!!!! stopping after 2 failures !!!!!!!!!!!!!!!!!!!!\n",
     "did not run"),
    # No test collected, and nothing printed (-qq over no test)
    (5, "", "did not run")])
def test_a_pytest_run_with_no_summary_line_says_nothing_of_the_tests_it_ran(
        exit_code, output, status, as_language, fake_test_command):
    """A pytest run that prints no line counting its tests reads no count and no name,
    whatever PASSED lines it prints, and its baseline entry is one that does not name
    each test it counts."""
    as_language("py")
    fake_test_command(exit_code, output)
    result = edit.run_test()
    part = edit.primary_part(result)
    assert result["status"] == status
    assert (part.ran_count, part.ran_name_list, part.is_ran_unread) == (None, None, True)
    assert edit.no_test_run(result) is False
    assert edit._ran_dict(edit.ran_entry_set(part)) == {"": (0, None, None, True)}


@pytest.mark.parametrize("exit_code, output", [
    (0, "2 passed in 0.01s"),
    (5, "no tests ran in 0.00s"),
    (5, "===== test session starts =====\n\n===== no tests ran in 0.01s ====="),
    (0, "...\n----------\nRan 3 tests in 0.001s\n\nOK"),
    (0, "PASSED tests/t.py::test_a"),
    (0, ""),
    (1, "...."),
    (0, "-----   [100%]"),
    (1, "F\nstopping after 1 failures"),
    (3, "===== test session starts =====\nINTERNALERROR> Traceback"),
    (5, "ERROR: file or directory not found: tests/x.py")])
def test_a_python_run_that_counts_its_tests_or_is_not_pytest_s_is_read_as_before(
        exit_code, output, as_language, fake_test_command):
    as_language("py")
    fake_test_command(exit_code, output)
    assert edit.primary_part(edit.run_test()).is_ran_unread is False


def test_another_language_s_run_that_prints_nothing_is_read_as_before(
        as_language, fake_test_command):
    as_language("rs")
    fake_test_command(5, "")
    assert edit.primary_part(edit.run_test()).is_ran_unread is False


@pytest.mark.parametrize("argv, level", [
    (["pytest", "-qq"], -2),
    ([sys.executable, "-m", "pytest", "-q", "-q", "tests"], -2),
    (["pytest", "--quiet", "--quiet"], -2),
    (["pytest", "-qqq"], -3),
    (["pytest", "--verbosity=-2"], -2),
    (["pytest", "--verbosity", "-2"], -2),
    (["pytest", "-qx", "-q"], -2),
    (["uv", "run", "--with", "pytest", "pytest", "-qq"], -2),
    (["/venv/bin/py.test", "-qq"], -2),
    ([sys.executable, "-mpytest", "-qq"], -2)])
def test_a_test_command_that_keeps_pytest_from_counting_its_tests_is_refused(
        argv, level, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "TEST_COMMAND", argv)
    assert edit.quiet_test_error() == (
        f"the test command `{' '.join(argv)}` sets pytest's verbosity to {level}, at "
        f"which pytest prints no line counting the tests it ran, so what the tests say "
        f"cannot be checked: give it one -q at most (TEST_COMMAND)")


@pytest.mark.parametrize("ext, argv", [
    ("py", list(LANG_DICT["py"].test_command_tuple)),
    ("py", ["pytest", "-q"]),
    ("py", ["uv", "-q", "run", "pytest", "-q"]),
    ("py", ["pytest", "-rqq"]),
    ("py", ["pytest", "-q", "--", "-q"]),
    ("py", ["pytest", "--verbosity=-2", "-v"]),
    ("py", ["make", "test", "-qq"]),
    ("py", ["tox", "-qq"]),
    ("rs", ["pytest", "-qq"])])
def test_a_test_command_whose_pytest_counts_its_tests_is_not_refused(ext, argv, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", ext)
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "TEST_COMMAND", argv)
    assert edit.quiet_test_error() == ""


@pytest.mark.parametrize("ext, output, count", [
    ("py", "== 1 failed, 2 passed, 1 skipped, 1 xfailed, 2 subtests passed in 0.1s ==", 4),
    ("py", "Ran 3 tests in 0.001s\nFAILED (failures=1, skipped=2)", 1),
    ("py", "no tests ran in 0.00s", None),
    ("java", "[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 1, Time elapsed: 0.1 s "
             "-- in a.T\n[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 1\n"
             "[ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 0", 4),
    ("kt", "3 tests completed, 1 failed", None),
    ("js", "Tests:       1 failed, 1 skipped, 1 todo, 3 passed, 6 total", 4),
    ("js", "# tests 3\n# pass 2\n# fail 1\n# skipped 0", 3),
    ("js", "ℹ tests 3\nℹ pass 1\nℹ fail 0\nℹ skipped 2", 1),
    ("c", "1/4 Test #1: a ....   Passed    0.00 sec\n2/4 Test #2: b ...***Failed  0.00 sec\n"
          "3/4 Test #3: c ...***Skipped   0.00 sec\n"
          "4/4 Test #4: d ...***Not Run (Disabled)   0.00 sec\n"
          "50% tests passed, 1 tests failed out of 4", 2),
    ("c", "1/1 Test #1: a ....   Passed    0.00 sec", None),
    ("rs", "test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n"
           "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out", 4),
    ("cs", "Passed!  - Failed:     0, Passed:     3, Skipped:     1, Total:     4", 3),
    ("cs", "Test run summary: Failed!\n  total: 4\n  failed: 1\n  succeeded: 2\n  skipped: 1", 3),
    ("r", "[ FAIL 0 | WARN 0 | SKIP 0 | PASS 3 ]", None)])
def test_how_many_tests_a_run_ran_is_read_off_its_summary(ext, output, count):
    assert edit._ran_count(output, ext) == count
    # The lines read for it are kept between the first and the last bytes returned
    if count is not None:
        assert any(edit._read_line_count(line, ext) for line in output.split("\n"))


@pytest.mark.parametrize("ext, output", [
    ("py", "test_a (t.T.test_a) ... skipped 'x'\ntest_b (t.T.test_b) ... skipped 'x'\n\n"
           "Ran 2 tests in 0.001s\n\nOK (skipped=2)"),
    ("java", "[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 2"),
    ("js", "Tests:       2 skipped, 2 total"),
    ("js", "# tests 2\n# pass 0\n# fail 0\n# skipped 2"),
    ("c", "1/1 Test #1: a ...***Skipped   0.00 sec\n100% tests passed, 0 tests failed out of 1"),
    ("rs", "running 2 tests\ntest result: ok. 0 passed; 0 failed; 2 ignored; 0 measured"),
    ("cs", "Passed!  - Failed:     0, Passed:     0, Skipped:     2, Total:     2")])
def test_a_run_that_skipped_every_test_it_found_is_a_run_over_no_tests(ext, output):
    assert edit._is_no_test_text(output, ext) is True
    assert edit._test_verdict(0, output, ext, []) is None


@pytest.mark.parametrize("output, count, is_no_test", [
    # As many skips as tests, without -v: read as a run over no tests
    ("s\n----\nRan 1 test in 0.000s\n\nOK (skipped=1)", 0, True),
    ("ss\n----\nRan 1 test in 0.000s\n\nOK (skipped=2)", 0, True),
    ("ss\n----\nRan 3 tests in 0.000s\n\nOK (skipped=2)", 1, False),
    # -v names a skipped subtest on an indented line, and a test skipped whole on its own
    ("test_s (t.T.test_s) ... \n  test_s (t.T.test_s) (x=3) ... skipped 'no'\n\n"
     "Ran 1 test in 0.000s\n\nOK (skipped=1)", 1, False),
    ("test_a (t.T.test_a) ... skipped 'x'\ntest_s (t.T.test_s) ... \n"
     "  test_s (t.T.test_s) (x=2) ... skipped 'no'\n\nRan 2 tests in 0.000s\n\nOK (skipped=2)",
     1, False),
    ("Ran 0 tests in 0.000s\n\nOK", 0, True),
    # A class or a module skipped in its fixture is no test "Ran N" counts: the "." of a
    # test that passed, and the -v line of the fixture, say the others ran
    ("..ss\n" + "-" * 70 + "\nRan 2 tests in 0.000s\n\nOK (skipped=2)", 2, False),
    ("x.s\n" + "-" * 70 + "\nRan 2 tests in 0.000s\n\nOK (skipped=1, expected failures=1)",
     2, False),
    ("ss\n" + "-" * 70 + "\nRan 2 tests in 0.000s\n\nOK (skipped=2)", 0, True),
    ("test_a (t.T.test_a) ... ok\nsetUpClass (t.Db) ... skipped 'no db'\n"
     "setUpModule (n) ... skipped 'no net'\n\n" + "-" * 70
     + "\nRan 1 test in 0.000s\n\nOK (skipped=2)", 1, False),
    # A docstring's line after the test's own line, indented for a subtest
    ("test_s (t.T.test_s)\nEach adds up. ... \n  test_s (t.T.test_s) (x=3)\n"
     "Each adds up. ... skipped 'no'\n\nRan 1 test in 0.000s\n\nOK (skipped=1)", 1, False),
    ("test_a (t.T.test_a) ... ok\ntest_w (t.T.test_w)\nEach multiplies. ... skipped 'x'\n\n"
     "Ran 2 tests in 0.000s\n\nOK (skipped=1)", 1, False),
    ("test_w (t.T.test_w)\nEach multiplies. ... skipped 'x'\n\n"
     "Ran 1 test in 0.000s\n\nOK (skipped=1)", 0, True)])
def test_a_unittest_run_is_over_no_tests_unless_its_v_lines_say_a_test_ran(
        output, count, is_no_test):
    assert edit._ran_count(output, "py") == count
    assert edit._is_no_test_text(output, "py") is is_no_test
    # The -v lines are kept between the first and the last bytes returned
    for line in output.split("\n"):
        if "skipped '" in line or line.endswith(")") or line in ("..ss", "x.s", "-" * 70):
            assert edit._read_line_count(line, "py") == 1


@pytest.mark.parametrize("ext, output, verdict", [
    ("py", "1 failed, 2 passed in 0.01s", False),
    ("py", "==== 2 passed, 1 error in 0.01s ====", False),
    ("py", "Ran 2 tests in 0.001s\n\nFAILED (failures=1)", False),
    ("py", "2 passed, 1 xfailed in 0.01s", True),
    ("java", "[ERROR] Tests run: 2, Failures: 1, Errors: 0, Skipped: 0", False),
    ("java", "[INFO] Tests run: 2, Failures: 0, Errors: 1, Skipped: 0", False),
    ("java", "[INFO] Tests run: 12, Failures: 0, Errors: 0, Skipped: 1", True),
    ("kt", "2 tests completed, 1 failed", False),
    ("js", "Tests:       1 failed, 1 passed, 2 total", False),
    ("js", "Tests:       2 passed, 2 total", True),
    ("js", "# tests 2\n# pass 1\n# fail 1", False),
    ("ts", "ℹ tests 2\nℹ pass 2\nℹ fail 0", True),
    ("c", "50% tests passed, 1 tests failed out of 2", False),
    ("c", "100% tests passed, 0 tests failed out of 2", True),
    ("rs", "test result: FAILED. 1 passed; 1 failed; 0 ignored", False),
    ("cs", "Failed!  - Failed:     1, Passed:     3, Skipped:     0, Total:     4", False),
    ("cs", "Test run summary: Failed!\n  total: 4\n  failed: 1\n  succeeded: 3", False),
    ("r", "[ FAIL 1 | WARN 0 | SKIP 0 | PASS 2 ]", False)])
def test_a_run_whose_summary_counts_a_failing_test_is_red_whatever_its_exit_code(
        ext, output, verdict):
    assert edit._test_verdict(0, output, ext, []) is verdict
    # The line is kept between the first and the last bytes returned
    assert any(edit._read_line_count(line, ext) for line in output.split("\n"))


def test_a_pytest_run_of_expected_failures_alone_is_a_run_over_tests():
    assert edit._is_no_test_text("1 xfailed in 0.01s", "py") is False


def test_a_run_is_held_to_the_tests_the_run_before_it_ran(repo, monkeypatch):
    """By names where both runs name them: a test that ran before and not now is one,
    unless its file is gone, or changed and still has a test that runs."""
    monkeypatch.setattr(config, "TEST_COMMAND", ["t"])
    monkeypatch.setattr(edit, "other_test_list", lambda *a: [])
    for path in ("tests/a.py", "tests/b.py", "tests/c.py", "tests/d.py"):
        (repo / path).parent.mkdir(exist_ok=True)
        (repo / path).write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "tests")

    def run_of(name_list, count=None):
        result = edit.RunResult({"status": "passed"})
        result.ran_name_list = name_list
        result.ran_count = len(name_list) if count is None else count
        return result

    before = run_of(["tests/a.py::t1", "tests/b.py::t1", "tests/b.py::t2", "tests/c.py::t1",
                     "tests/d.py::t1"])
    baseline = edit.ran_entry_set(before)
    reason = "{command} {tests}"
    count_reason = "{command} {now}/{before}"
    unread = "{command} says less"
    short = "{command} names some"

    def unrun(now):
        return edit.unrun_reason(now, baseline, "t", reason, count_reason, unread, short)

    assert unrun(before) == ""
    # a.py is not changed and its test runs no more; b.py is changed and one of its tests
    # still runs; c.py is changed and none of its tests runs; d.py is gone
    (repo / "tests" / "b.py").write_text("y\n")
    (repo / "tests" / "c.py").write_text("y\n")
    (repo / "tests" / "d.py").unlink()
    assert unrun(run_of(["tests/b.py::t3"])) == "t tests/a.py::t1, tests/c.py::t1"
    # By count where the run now names none: fewer is a reason, as many is not
    monkeypatch.setattr(config, "PRIMARY_EXT", "js")
    now = edit.RunResult({"status": "passed"})
    now.ran_count = 4
    assert unrun(now) == "t 4/5"
    now.ran_count = 5
    assert unrun(now) == ""
    count_before = edit.RunResult({"status": "passed"})
    count_before.ran_count = 3
    count_baseline = edit.ran_entry_set(count_before)
    now.ran_count = 2
    assert edit.unrun_reason(now, count_baseline, "t", reason, count_reason, unread, short) == "t 2/3"
    # ... but not where a file the command runs is gone, whose tests are not counted
    # apart: tests/d.py is one where the command is the Python one
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    assert edit.unrun_reason(now, count_baseline, "t", reason, count_reason, unread, short) == ""
    # A run that says neither, and a baseline with no entry, say nothing
    assert edit.unrun_reason(edit.RunResult({"status": "passed"}), baseline, "t", reason,
                             count_reason, unread, short) == ""
    # ... unless they were read off the whole run: it says less than the run before
    for count, name_list in ((None, None), (5, None)):
        unread_run = run_of([], count)
        unread_run.ran_count, unread_run.ran_name_list = count, name_list
        unread_run.is_ran_hold = True
        assert edit.unrun_reason(unread_run, baseline, "t", reason, count_reason,
                                 unread, short) == "t says less"
        assert edit.unrun_reason(unread_run, count_baseline, "t", reason, count_reason,
                                 unread, short) == ("t says less" if count is None else "")
    assert edit.unrun_reason(now, frozenset(), "t", reason, count_reason, unread, short) == ""
    # A whole run that names some of the tests it ran, not every one, is a reason, and
    # so is any whole run held to such a run before it
    short_run = run_of(["tests/a.py::t1"], 5)
    short_run.ran_name_list, short_run.is_name_short = None, True
    short_run.is_ran_hold = True
    for each_baseline in (baseline, count_baseline):
        assert edit.unrun_reason(short_run, each_baseline, "t", reason, count_reason, unread,
                                 short) == "t names some"
    short_baseline = edit.ran_entry_set(short_run)
    assert edit._ran_dict(short_baseline) == {"": (5, None, None, True)}
    held_run = run_of(["tests/a.py::t1", "tests/b.py::t1", "tests/b.py::t2"])
    held_run.is_ran_hold = True
    assert edit.unrun_reason(held_run, short_baseline, "t", reason, count_reason, unread,
                             short) == "t names some"
    # ... and not a run they were not read off: it is held by the count, which a test
    # file that is gone leaves out
    held_run.is_ran_hold = False
    assert edit.unrun_reason(held_run, short_baseline, "t", reason, count_reason, unread,
                             short) == ""


def test_a_test_whose_function_still_runs_or_whose_file_runs_as_many_is_no_reason(
        repo, monkeypatch):
    """A test that runs no more is held only where no test of its function runs now and
    its file runs fewer tests; a doctest is not held where its file was changed, and a
    test function is, wherever its file is."""
    monkeypatch.setattr(config, "TEST_COMMAND", ["t"])
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(edit, "other_test_list", lambda *a: [])
    for path in ("tests/test_a.py", "tests/test_b.py", "tests/helpers.py", "b.py", "c.py"):
        (repo / path).parent.mkdir(exist_ok=True)
        (repo / path).write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "tests")

    def run_of(name_list):
        result = edit.RunResult({"status": "passed"})
        result.ran_name_list, result.ran_count = name_list, len(name_list)
        return result

    before = ["tests/test_a.py::test_q[pg]", "tests/test_a.py::test_q[lite]",
              "tests/test_a.py::T::test_one", "tests/test_b.py::test_b", "b.py::b.fmt",
              "c.py::c.fmt", "tests/helpers.py::helpers.one"]
    baseline = edit.ran_entry_set(run_of(before))

    def unrun(name_list):
        return edit.unrun_reason(run_of(name_list), baseline, "t", "{tests}", "{now}/{before}", "", "")

    same_list = ["tests/test_b.py::test_b", "b.py::b.fmt", "c.py::c.fmt",
                 "tests/helpers.py::helpers.one"]
    # The parameters' ids change, one value goes, or a method of a base class is renamed
    assert unrun(["tests/test_a.py::test_q[postgres]", "tests/test_a.py::test_q[lite]",
                  "tests/test_a.py::T::test_one", *same_list]) == ""
    assert unrun(["tests/test_a.py::test_q[lite]", "tests/test_a.py::T::test_one",
                  *same_list]) == ""
    assert unrun(["tests/test_a.py::test_q[pg]", "tests/test_a.py::test_q[lite]",
                  "tests/test_a.py::T::test_first", *same_list]) == ""
    # A function that runs no more where its file runs fewer tests is a reason
    assert unrun(["tests/test_a.py::test_q[pg]", "tests/test_a.py::test_q[lite]",
                  "tests/test_a.py::T::test_one", "b.py::b.fmt", "c.py::c.fmt",
                  "tests/helpers.py::helpers.one"]) == "tests/test_b.py::test_b"
    assert unrun(["tests/test_a.py::T::test_one", *same_list]) == \
        "tests/test_a.py::test_q[lite], tests/test_a.py::test_q[pg]"
    # A changed file's doctest is not held, an unchanged file's is
    (repo / "b.py").write_text("y\n")
    (repo / "tests" / "helpers.py").write_text("y\n")
    assert unrun(["tests/test_a.py::test_q[pg]", "tests/test_a.py::test_q[lite]",
                  "tests/test_a.py::T::test_one", "tests/test_b.py::test_b"]) == "c.py::c.fmt"
    # A test function of a changed file outside the tests' directory is held
    (repo / "test_main.py").write_text("x\n")
    (repo / "pkg" / "test").mkdir(parents=True)
    (repo / "pkg" / "test" / "test_x.py").write_text("x\n")
    _git(repo, "add", "test_main.py", "pkg")
    _git(repo, "commit", "-q", "-m", "outside")
    out_baseline = edit.ran_entry_set(run_of([*same_list, "test_main.py::test_main",
                                              "test_main.py::test_m2",
                                              "pkg/test/test_x.py::test_x"]))
    (repo / "test_main.py").write_text("y\n")
    (repo / "pkg" / "test" / "test_x.py").write_text("y\n")
    assert edit.unrun_reason(run_of(same_list), out_baseline, "t", "{tests}", "", "", "") == (
        "pkg/test/test_x.py::test_x, test_main.py::test_m2, test_main.py::test_main")
    assert [name for name in ("b.py::b.fmt", "pkg/__init__.py::pkg", "a.py::a",
                              "src/pkg/a.py::pkg.a.A.f", "docs/a.txt::a.txt",
                              "test_main.py::test_main", "t.py::T::test_a", "t.py::t.x[1]",
                              "t.py::other.x", "spec.yaml::case1")
            if edit._is_doctest_name(name)] == [
        "b.py::b.fmt", "pkg/__init__.py::pkg", "a.py::a", "src/pkg/a.py::pkg.a.A.f",
        "docs/a.txt::a.txt"]
    # A file of the same end is another file: tests/test_a.py is not changed
    (repo / "pkg" / "tests").mkdir(parents=True)
    (repo / "pkg" / "tests" / "test_a.py").write_text("x\n")
    assert unrun(["tests/test_a.py::T::test_one", "pkg/tests/test_a.py::test_s",
                  *same_list]) == "tests/test_a.py::test_q[lite], tests/test_a.py::test_q[pg]"
    # cargo's tests are counted by the file of each target
    monkeypatch.setattr(config, "PRIMARY_EXT", "rs")
    rs_baseline = edit.ran_entry_set(run_of(["src/lib.rs::tests::a [x]",
                                             "src/lib.rs::tests::b [y]"]))
    assert edit.unrun_reason(run_of(["src/lib.rs::tests::a [x]", "src/lib.rs::tests::c [x]"]),
                             rs_baseline, "t", "{tests}", "", "", "") == "src/lib.rs::tests::b [y]"
    # A source of the same end that was changed leaves out a target's tests only where
    # the target still runs one
    (repo / "crates" / "x" / "src").mkdir(parents=True)
    (repo / "crates" / "x" / "src" / "lib.rs").write_text("y\n")
    assert edit.unrun_reason(run_of(["src/lib.rs::tests::a [x]", "src/lib.rs::tests::c [x]"]),
                             rs_baseline, "t", "{tests}", "", "", "") == "src/lib.rs::tests::b [y]"
    rs_baseline = edit.ran_entry_set(run_of(["src/lib.rs::tests::a [x]",
                                             "src/lib.rs::tests::b [y]",
                                             "src/lib.rs::tests::d [y]"]))
    assert edit.unrun_reason(run_of(["src/lib.rs::tests::a [x]", "src/lib.rs::tests::b [y]"]),
                             rs_baseline, "t", "{tests}", "", "", "") == ""
    assert edit._name_part_tuple("src/lib.rs::tests::b [y]", "rs") == (
        "src/lib.rs", "src/lib.rs [y]", "src/lib.rs::tests::b [y]")
    assert edit._name_part_tuple("tests/t.py::T::test_q[a::b[c]]", "py") == (
        "tests/t.py", "tests/t.py", "tests/t.py::T::test_q")


def test_the_tests_a_run_ran_go_in_its_baseline_and_between_containers():
    result = edit.RunResult({"status": "passed"})
    assert edit.ran_entry_set(result) == frozenset()
    result.ran_count, result.ran_name_list = 2, ["t::a", "t::b"]
    entry_set = edit.ran_entry_set(result)
    assert edit._ran_dict(entry_set) == {"": (2, {"t::a", "t::b"}, None, False)}
    # It is no failing test and no mark of a command
    assert edit.baseline_name_list(entry_set) == [] and edit.fail_name_list(entry_set) == []
    assert edit._mark_dict(entry_set) == {}
    joined = edit.join_baseline({"a": entry_set | {"t::x"}, "b": edit.ran_entry_set(result)})
    assert edit._ran_dict(joined) == {"a": (2, {"t::a", "t::b"}, None, False),
                                      "b": (2, {"t::a", "t::b"}, None, False)}
    assert edit._mark_dict(joined) == {"a": "red before", "b": "green before"}
    assert frozenset(json.loads(json.dumps(sorted(joined)))) == joined
    # Names that would make the entry too long are left for the count alone
    result.ran_name_list = ["t::" + "n" * edit._MAX_RAN_NAME_CHAR]
    assert edit._ran_dict(edit.ran_entry_set(result)) == {"": (2, None, None, False)}
    # Where the run named the parts it ran tests of and not the tests
    result.ran_name_list = None
    result.ran_unit_dict = {"com.e.ATest": 2, "test/b.test.js": None}
    assert edit._ran_dict(edit.ran_entry_set(result)) == {
        "": (2, None, {"com.e.ATest": 2, "test/b.test.js": None}, False)}


_CARGO_OUTPUT = (
    "\nrunning 2 tests\ntest tests::adds ... ok\ntest tests::subs ... FAILED\n"
    "test tests::later ... ignored\n\ntest result: FAILED. 1 passed; 1 failed; 1 ignored; "
    "0 measured; 0 filtered out; finished in 0.00s\n\n\nrunning 1 test\n"
    "test src/lib.rs - add (line 5) ... ok\n\ntest result: ok. 1 passed; 0 failed; "
    "0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n\n"
    "     Running unittests src/lib.rs (target/debug/deps/calc-f3ab8b639f461af0)\n"
    "   Doc-tests calc\n")
_MAVEN_OUTPUT = (
    "[INFO] Running com.e.ATest\n[ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 1, "
    "Time elapsed: 0.04 s <<< FAILURE! -- in com.e.ATest\n[INFO] Tests run: 1, Failures: 0, "
    "Errors: 0, Skipped: 0, Time elapsed: 0.01 s -- in com.e.BTest$Inner\n[INFO] Results:\n"
    "[ERROR] Tests run: 4, Failures: 1, Errors: 0, Skipped: 1\n")


def test_the_parts_a_run_ran_tests_of_are_read_where_its_output_names_them():
    """cargo names each test under the target its block is of, surefire counts each
    class's tests, jest names each test file; node --test names none."""
    assert edit._ran_name_list(_CARGO_OUTPUT, "rs", [], "web") == [
        "src/lib.rs::src/lib.rs - add (line 5) [calc]", "src/lib.rs::tests::adds [calc]",
        "src/lib.rs::tests::subs [calc]"]
    # A block without its target line, or names fewer than the results count, say none
    assert edit._ran_name_list(_CARGO_OUTPUT.replace("   Doc-tests calc\n", ""), "rs", [],
                               "") is None
    assert edit._ran_unit_dict(_MAVEN_OUTPUT, "java", "core") == {
        "com.e.ATest": 2, "com.e.BTest$Inner": 1}
    assert edit._ran_unit_dict(_MAVEN_OUTPUT.replace("Tests run: 4", "Tests run: 5"),
                               "java", "") is None
    # vitest's failing test is named by its file and describe blocks, and a file that
    # failed to run by the file alone
    for name, failed in (("js_R_vitest", ["test/calc.test.js > calc > bar"]),
                         ("js_R_vitest3", ["test/calc.test.js > calc > bar"]),
                         ("js_B_vitest", ["test/calc.test.js"]),
                         ("js_B_vitest3", ["test/calc.test.js"])):
        out = json.loads((_GOLDEN_DIR / f"{name}.json").read_text())["output"]
        assert [n for _, n in edit._fail_pair_list(edit._plain_text(out), "js")] == failed
    # A name that ends in a bracket of its own keeps it (test.each)
    assert [n for _, n in edit._fail_pair_list(
        "  ● calc › sums [ 1, 2 ]\n  ● calc › sums [ 3, 4 ]\nnot ok 1 - sums [ 1, 2 ]\n"
        " FAIL  test/a.test.js > calc > sums [ 1, 2 ]\n", "js")] == [
        "calc › sums [ 1, 2 ]", "calc › sums [ 3, 4 ]", "sums [ 1, 2 ]",
        "test/a.test.js > calc > sums [ 1, 2 ]"]
    # vitest names each test file too; under a workspace project's label the path is the
    # project's, so the files are not named
    for name in ("js_G_vitest", "js_R_vitest3"):
        out = json.loads((_GOLDEN_DIR / f"{name}.json").read_text())["output"]
        assert edit._ran_unit_dict(out, "js", "") == {"test/calc.test.js": None}
    assert edit._ran_unit_dict(" ✓ |a| test/a.test.ts (2 tests) 3ms\n"
                               " ✓ test/b.test.ts (1 test) 1ms\n"
                               "      Tests  3 passed (3)\n", "js", "") is None
    jest_output = ("PASS test/a.test.js\n FAIL  test/b.test.js (5.2 s)\n  ● b › one\n"
                   " FAIL  test/c.test.js > vitest's line\nTests:       1 failed, 2 passed, 3 total\n")
    assert edit._ran_unit_dict(jest_output, "js", "web") == {
        "web/test/a.test.js": None, "web/test/b.test.js": None}
    assert edit._ran_unit_dict("# tests 2\n# pass 2\n# fail 0\n", "js", "") is None
    # The lines they are read off are kept between the first and the last bytes returned
    for line, ext in (("test tests::adds ... ok", "rs"), ("running 2 tests", "rs"),
                      ("     Running unittests src/lib.rs (target/debug/deps/calc-f3ab)", "rs"),
                      ("   Doc-tests calc", "rs"), (_MAVEN_OUTPUT.split("\n")[2], "java"),
                      ("PASS test/a.test.js", "js")):
        assert edit._read_line_count(line, ext) == 1, line


_MAVEN_FLAKE_OUTPUT = (
    "[INFO] Running com.e.FooTest\n"
    "[ERROR] Tests run: 2, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.04 s "
    "<<< FAILURE! -- in com.e.FooTest\n"
    "[ERROR] com.e.FooTest.adds -- Time elapsed: 0.01 s <<< FAILURE!\n"
    "[INFO] Running com.e.FooTest\n"
    "[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.01 s "
    "-- in com.e.FooTest\n"
    "[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.01 s "
    "-- in com.e.BarTest\n[INFO] Results:\n[INFO] \n[WARNING] Flakes: \n"
    "[WARNING] com.e.FooTest.adds\n"
    "[ERROR]   Run 1: FooTest.adds:12 expected: <3> but was: <4>\n"
    "[INFO]   Run 2: PASS\n[INFO] \n"
    "[WARNING] Tests run: 3, Failures: 0, Errors: 0, Skipped: 0, Flakes: 1\n")


def test_a_maven_test_that_passed_when_run_again_is_read_as_a_passing_test(
        as_language, fake_test_command):
    """surefire's rerunFailingTestsCount: the module's count goes on with ", Flakes: N",
    the class's first try counts the failure, and the test is listed under "Flakes:"."""
    as_language("java")
    fake_test_command(0, _MAVEN_FLAKE_OUTPUT)
    result = edit.primary_part(edit.run_test())
    assert result["status"] == "passed" and result.is_ran_hold
    assert (result.ran_count, result.ran_unit_dict) == (
        3, {"com.e.FooTest": 2, "com.e.BarTest": 1})
    # Beside a test that fails, it is no failing test of the run
    fake_test_command(1, _MAVEN_FLAKE_OUTPUT.replace(
        "[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.01 s "
        "-- in com.e.BarTest\n",
        "[ERROR] Tests run: 1, Failures: 1, Errors: 0, Skipped: 0, Time elapsed: 0.01 s "
        "<<< FAILURE! -- in com.e.BarTest\n"
        "[ERROR] com.e.BarTest.subs -- Time elapsed: 0.01 s <<< FAILURE!\n").replace(
        "[WARNING] Tests run: 3, Failures: 0", "[ERROR] Tests run: 3, Failures: 1"))
    result = edit.run_test()
    assert result["status"] == "failed" and result["failed"] == ["com.e.BarTest.subs"]
    assert edit._ran_count(_MAVEN_FLAKE_OUTPUT.replace(", Flakes: 1", ""), "java") == 3


def test_a_part_of_a_run_is_matched_to_the_file_it_is_of():
    assert edit._part_path_match("tests/test_a.py", {"tests/test_a.py"}, "py")
    assert not edit._part_path_match("tests/test_a.py", {"tests/test_b.py"}, "py")
    # A file named from the repository root is not another file with the same end
    assert not edit._part_path_match("tests/test_a.py", {"pkg/tests/test_a.py"}, "py")
    assert not edit._part_path_match("test/a.test.js", {"web/test/a.test.js"}, "js")
    assert edit._part_path_match("src/lib.rs", {"crates/a/src/lib.rs"}, "rs")
    assert edit._part_path_match("com.e.ATest$Inner", {"core/src/test/java/com/e/ATest.java"},
                                 "java")
    assert not edit._part_path_match("com.e.ATest", {"src/test/java/com/e/ATestMore.java"},
                                     "java")


def test_a_run_is_held_to_the_parts_the_run_before_ran_tests_of(repo, monkeypatch):
    """A class whose tests run fewer, and a file whose tests are not counted that runs no
    more, unless its file is gone, or changed and still runs a test; where a part's tests
    are not counted, the counts are held as well, unless a test file was changed."""
    monkeypatch.setattr(config, "TEST_COMMAND", ["t"])
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(edit, "other_test_list", lambda *a: [])
    for path in ("src/test/java/com/e/ATest.java", "src/test/java/com/e/BTest.java"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "tests")
    reason, count_reason = "{command} {tests}", "{command} {now}/{before}"
    unread, short = "{command} says less", "{command} names some"

    def run_of(unit_dict):
        result = edit.RunResult({"status": "passed"})
        result.ran_unit_dict = unit_dict
        result.ran_count = sum(count or 1 for count in unit_dict.values())
        return result

    monkeypatch.setattr(config, "PRIMARY_EXT", "java")
    baseline = edit.ran_entry_set(run_of({"com.e.ATest": 3, "com.e.BTest": 1}))
    now = run_of({"com.e.ATest": 2, "com.e.BTest": 1})
    assert edit.unrun_reason(now, baseline, "t", reason, count_reason, unread, short) == \
        "t 1 of the 3 tests of com.e.ATest"
    (repo / "src/test/java/com/e/ATest.java").write_text("y\n")
    assert edit.unrun_reason(now, baseline, "t", reason, count_reason, unread, short) == ""
    (repo / "src/test/java/com/e/BTest.java").unlink()
    assert edit.unrun_reason(run_of({"com.e.ATest": 2}), baseline, "t", reason,
                             count_reason, unread, short) == ""
    # A file whose tests are not counted: gone from the run, or fewer tests in all
    monkeypatch.setattr(config, "PRIMARY_EXT", "js")
    monkeypatch.setattr(config, "TEST_PATH", "test")
    (repo / "test").mkdir()
    for name in ("a.test.js", "b.test.js"):
        (repo / "test" / name).write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "js")
    js_baseline = edit.ran_entry_set(run_of({"test/a.test.js": None, "test/b.test.js": None}))
    gone = run_of({"test/a.test.js": None})
    assert edit.unrun_reason(gone, js_baseline, "t", reason, count_reason, unread, short) == \
        "t the tests of test/b.test.js"
    fewer = run_of({"test/a.test.js": None, "test/b.test.js": None})
    fewer.ran_count = 1
    assert edit.unrun_reason(fewer, js_baseline, "t", reason, count_reason, unread, short) == "t 1/2"
    # ... and not where a test file of the language was changed
    (repo / "test" / "a.test.js").write_text("y\n")
    assert edit.unrun_reason(fewer, js_baseline, "t", reason, count_reason, unread, short) == ""
    assert edit._command_test_file_change("t") is True
    assert edit._command_test_file_change("other") is False


def test_jest_s_second_listing_of_the_failing_tests_is_not_counted_again():
    output = ("FAIL test/a.test.js\n  ● FooTest › bar\n\nSummary of all failing tests\n"
              "FAIL test/a.test.js\n  ● FooTest › bar\n\nTest Suites: 1 failed, 21 total\n")
    assert [name for _, name in edit._fail_pair_list(output, "js")] == [
        "test/a.test.js", "FooTest › bar"]
    # Where nothing follows the listing it runs to the end of the output
    assert [name for _, name in edit._fail_pair_list(output.split("Test Suites")[0], "js")] == [
        "test/a.test.js", "FooTest › bar"]


def test_jest_s_colored_badge_names_the_file_as_the_plain_one_does():
    plain = "FAIL  test/b.test.js"
    colored = edit._plain_text("\x1b[0m\x1b[7m\x1b[1m\x1b[31m FAIL \x1b[39m\x1b[22m\x1b[27m\x1b[0m "
                               "\x1b[2mtest/\x1b[22m\x1b[1mb.test.js\x1b[22m")
    assert edit._fail_name(plain, "js") == edit._fail_name(colored, "js") == "test/b.test.js"
    # A line of a test's own output written further in is not read
    assert edit._fail_name("      FAIL something", "js") is None


def test_the_failing_names_keep_how_many_times_each_failed():
    entry_set = edit.fail_entry_set(["t::a", "t::b", "t::a", "t::a"])
    assert entry_set == {"t::a", "t::b", "t::a\x1f2", "t::a\x1f3"}
    assert edit.fail_name_list(entry_set) == ["t::a", "t::a", "t::a", "t::b"]
    # More times than before are left over, and the fewer times are what two runs share
    assert edit.fail_name_list(entry_set - edit.fail_entry_set(["t::a", "t::b"])) == \
        ["t::a", "t::a"]
    assert edit.fail_entry_set(["t::a"]) - entry_set == frozenset()
    assert entry_set & edit.fail_entry_set(["t::a", "t::a"]) == {"t::a", "t::a\x1f2"}
    joined = edit.join_baseline({"a": entry_set, "b": None})
    assert edit.baseline_name_list(joined) == ["t::a", "t::b"]
    assert edit.fail_name_list(joined) == ["t::a", "t::a", "t::a", "t::b"]
    assert frozenset(json.loads(json.dumps(sorted(joined)))) == joined


def test_a_red_test_is_said_whatever_did_not_run_beside_it(_two_language_repo, tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND", [
        sys.executable, "-c", "print('=== short test summary info ===');"
                              "print('FAILED tests/test_a.py::test_a - assert 0'); "
                              "print('1 failed'); raise SystemExit(1)"])
    _test_tool(tmp_path, "jstest", 'echo "Cannot find module x"; exit 1')
    result = edit.run_test()
    assert result["status"] == "failed"
    assert result["failed"] == ["tests/test_a.py::test_a"]


@pytest.mark.parametrize("change", ["no tool", "no dependency file", "set by hand",
                                    "tests off"])
def test_another_language_s_tests_do_not_run_without_what_they_need(
        _two_language_repo, monkeypatch, change):
    if change == "no tool":
        monkeypatch.setattr(config, "OTHER_TEST_LIST", [("js", ["no-such-test-tool"])])
    elif change == "no dependency file":
        subprocess.run(["git", "-C", str(_two_language_repo), "rm", "-q", "-f",
                        "web/package.json"], check=True)
    elif change == "set by hand":
        monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    else:
        monkeypatch.setattr(config, "RUN_TEST", False)
    assert edit.other_test_list() == []
    if change != "tests off":
        # The return value is the primary command's own, with no line naming it
        assert edit.run_test()["output"].startswith("1 passed")


@pytest.mark.parametrize("file_dict, expect", [
    ({"pyproject.toml": "[project]\nname = 'p'\n"}, []),
    ({}, ["--isolated"]),
    ({"setup.py": "", "requirements-dev.txt": "pytest-mock\n"},
     ["--isolated", "--with-editable", ".", "--with-requirements", "requirements-dev.txt"]),
    ({"pyproject.toml": "[tool.black]\nline-length = 88\n", "requirements.txt": "x\n"},
     ["--isolated", "--with-requirements", "requirements.txt"]),
    ({"pyproject.toml": "[build-system]\nrequires = ['poetry-core']\n[tool.poetry]\n"},
     ["--isolated", "--with-editable", "."]),
    ({"pyproject.toml": "[build-system]\nrequires = ['poetry-core']\n"
                        "[tool.poetry]\npackage-mode = false\n"}, ["--isolated"]),
    ({"pyproject.toml": "[project\n"}, []),
    ({"setup.py": "from Cython.Build import cythonize\nimport os\n"}, ["--isolated"]),
    ({"setup.py": "import numpy as np\nfrom setuptools import setup\n",
      "requirements.txt": "numpy\n"},
     ["--isolated", "--with-requirements", "requirements.txt"]),
    ({"setup.py": "try:\n    from Cython.Build import cythonize\nexcept ImportError:\n"
                  "    cythonize = None\nfrom setuptools import setup\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "import versioneer\nfrom distutils.core import setup\n",
      "versioneer.py": ""}, ["--isolated", "--with-editable", "."]),
    ({"setup.py": "from .lib import x\nimport sys, os.path\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "def broken(:\n"}, ["--isolated", "--with-editable", "."]),
    ({"setup.py": "from Cython.Build import cythonize\n",
      "pyproject.toml": "[build-system]\nrequires = ['setuptools', 'Cython>=3']\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "import numpy\n",
      "pyproject.toml": "[build-system]\nrequires = ['setuptools', 'wheel']\n"},
     ["--isolated"]),
    ({"setup.py": "import numpy\n",
      "pyproject.toml": "[build-system]\nrequires = ['scikit-build-core']\n"
                        "build-backend = 'scikit_build_core.build'\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "import numpy\n", "pyproject.toml": "[tool.black]\n"}, ["--isolated"]),
    ({"setup.py": "import pkg_resources\n"}, ["--isolated"]),
    ({"setup.py": "from packaging.version import Version\nimport tomli\n"
                  "from setuptools import setup\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "import os, sys\nsys.path.insert(0, os.path.join("
                  "os.path.dirname(__file__), 'src'))\nfrom mypkg import __version__\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "from Cython.Build import cythonize\n",
      "requirements-dev.txt": "-e .[test]\npytest-mock\n",
      "requirements.txt": "numpy\n"},
     ["--isolated", "--with-requirements", "requirements.txt"]),
    ({"setup.py": "import numpy\n",
      "pyproject.toml": "[build-system]\n"
                        "requires = ['setuptools', 'wheel', 'oldest-supported-numpy']\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "from Cython.Build import cythonize\n",
      "pyproject.toml": "[build-system]\nrequires = ['setuptools>=61', 'wheel']\n"
                        "build-backend = 'setuptools.build_meta'\n"},
     ["--isolated"]),
    ({"setup.py": "import numpy\n", "requirements.txt": "-e .\n",
      "requirements-dev.txt": "-r requirements.txt\npytest\n"}, ["--isolated"]),
    ({"setup.py": "import pkg_resources\n",
      "pyproject.toml": "[build-system]\nrequires = ['setuptools<82', 'wheel']\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "import numpy\n",
      "pyproject.toml": "[build-system]\nrequires = ['setuptools', 'setuptools_scm']\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "from demo import __version__\nfrom setuptools import setup\n",
      "demo/__init__.py": "import numpy\n__version__ = '1'\n"}, ["--isolated"]),
    ({"setup.py": "from demo import __version__\n",
      "demo/__init__.py": "import os\nfrom .core import x\n__version__ = '1'\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "import version\n", "version.py": "import numpy\n"}, ["--isolated"]),
    ({"setup.py": "import sys\nsys.path[0:0] = ['src']\nimport mypkg\n"},
     ["--isolated", "--with-editable", "."]),
    ({"setup.py": "import numpy\n", "requirements.txt": "-r requirements-dev.txt\nx\n",
      "requirements-dev.txt": "-r requirements.txt\npytest\n"},
     ["--isolated", "--with-requirements", "requirements.txt",
      "--with-requirements", "requirements-dev.txt"])])
def test_a_python_directory_with_no_uv_project_runs_isolated_with_its_own_packages(
        tmp_path, file_dict, expect):
    for name, text in file_dict.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    assert edit.uv_project_word_list(str(tmp_path)) == expect


def test_a_setup_py_whose_build_cannot_load_its_imports_is_said_once(tmp_path, caplog):
    (tmp_path / "setup.py").write_text("import numpy\nfrom Cython.Build import cythonize\n")
    edit._unbuild_warn_root_set.discard(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger=edit.__name__):
        edit.uv_project_word_list(str(tmp_path))
        edit.uv_project_word_list(str(tmp_path))
    warn_list = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_list) == 1
    assert "imports Cython, numpy" in warn_list[0]
    assert "leave out" not in warn_list[0]


def test_a_setup_py_with_an_invalid_escape_is_read_without_a_warning(tmp_path, recwarn):
    (tmp_path / "setup.py").write_text('import numpy\nPATTERN = "\\d+"\n')
    edit._unbuild_warn_root_set.add(str(tmp_path))
    assert edit.uv_project_word_list(str(tmp_path)) == ["--isolated"]
    assert not [w for w in recwarn if issubclass(w.category, SyntaxWarning)]


def test_the_warning_names_the_requirement_files_left_out(tmp_path, caplog):
    (tmp_path / "setup.py").write_text("from Cython.Build import cythonize\n")
    (tmp_path / "requirements-dev.txt").write_text("# dev\n--editable .\n")
    edit._unbuild_warn_root_set.discard(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger=edit.__name__):
        assert edit.uv_project_word_list(str(tmp_path)) == ["--isolated"]
    assert "leave out requirements-dev.txt" in caplog.text


def test_only_a_default_uv_run_command_gets_the_project_words(tmp_path):
    (tmp_path / "setup.py").write_text("")
    default = ("uv", "run", "--with", "pytest", "pytest")
    assert edit.with_uv_project_word(list(default), default, str(tmp_path)) == [
        "uv", "run", "--isolated", "--with-editable", ".", "--with", "pytest", "pytest"]
    assert edit.with_uv_project_word(["pytest", "-q"], default, str(tmp_path)) == [
        "pytest", "-q"]
    assert edit.with_uv_project_word(["cargo", "test"], ("cargo", "test"),
                                     str(tmp_path)) == ["cargo", "test"]


def test_another_language_s_python_tests_run_with_that_directory_s_packages(
        _two_language_repo, monkeypatch):
    """Python run from a directory of its own, with the default command."""
    (_two_language_repo / "svc").mkdir()
    (_two_language_repo / "svc" / "setup.py").write_text("")
    (_two_language_repo / "svc" / "app.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(_two_language_repo), "add", "-A"], check=True)
    monkeypatch.setattr(config, "OTHER_TEST_LIST",
                        [("py", list(LANG_DICT["py"].test_command_tuple))])
    monkeypatch.setattr(edit.shutil, "which", lambda name: "/usr/bin/" + name)
    (ext, argv, directory), = edit.other_test_list()
    assert (ext, directory) == ("py", "svc")
    assert argv[:5] == ["uv", "run", "--isolated", "--with-editable", "."]


def test_the_default_python_test_command_runs_with_the_repository_s_packages(
        tmp_path, monkeypatch):
    (tmp_path / "setup.py").write_text("")
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_COMMAND", list(LANG_DICT["py"].test_command_tuple))
    argv, _, _ = edit._build_argv(None, "path", "tests", "py")
    assert argv[:5] == ["uv", "run", "--isolated", "--with-editable", "."]


def test_python_tests_kept_in_another_test_directory_are_run_and_counted(
        tmp_path, monkeypatch):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("x = 1\n")
    (tmp_path / "testing").mkdir()
    (tmp_path / "testing" / "test_core.py").write_text("def test_x():\n    pass\n")
    monkeypatch.delenv("TEST_PATH", raising=False)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    for name in ("PRIMARY_EXT", "TEST_COMMAND", "TEST_PATH", "TEST_TARGET_STYLE",
                 "RUN_COMMAND", "CHECK_COMMAND", "OTHER_CHECK_LIST", "OTHER_TEST_LIST"):
        monkeypatch.setattr(config, name, getattr(config, name))
    config.set_primary_ext("py")
    assert config.TEST_PATH == "testing"
    assert edit._no_run_reason(None) is None
    assert paths.under_test_path(["testing/test_new.py", "pkg/core.py"]) == [
        "testing/test_new.py"]


def test_another_language_s_tests_in_its_other_test_directory_are_found(
        tmp_path, monkeypatch):
    (tmp_path / "svc" / "testing").mkdir(parents=True)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    assert edit._has_other_test_path("py", "svc")
    assert edit._is_language_test_path("testing/test_a.py", "py", "svc")
    assert not edit._is_language_test_path("app.py", "py", "svc")
    assert not edit._has_other_test_path("py", ".")


def test_another_language_s_tests_run_from_its_other_test_directory(
        _two_language_repo, tmp_path, monkeypatch):
    """A language reached by path is given the test directory its start directory holds."""
    (_two_language_repo / "DESCRIPTION").write_text("Package: x\n")
    (_two_language_repo / "R").mkdir()
    (_two_language_repo / "R" / "x.R").write_text("x <- 1\n")
    (_two_language_repo / "rtests").mkdir()
    subprocess.run(["git", "-C", str(_two_language_repo), "add", "-A"], check=True)
    monkeypatch.setitem(LANG_DICT, "r", dataclasses.replace(
        LANG_DICT["r"], test_path_alt_tuple=("rtests",)))
    _test_tool(tmp_path, "rtest", 'echo "ran r with: $*"')
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [("r", ["rtest"])])
    assert "ran r with: rtests" in edit.run_test()["output"]


def test_a_language_reached_by_path_is_left_out_where_its_tests_directory_is_not(
        _two_language_repo, tmp_path, monkeypatch):
    """R's tests are reached by path (tests/testthat)."""
    (_two_language_repo / "DESCRIPTION").write_text("Package: x\n")
    (_two_language_repo / "R").mkdir()
    (_two_language_repo / "R" / "x.R").write_text("x <- 1\n")
    subprocess.run(["git", "-C", str(_two_language_repo), "add", "-A"], check=True)
    _test_tool(tmp_path, "rtest", 'echo "ran r"')
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [("r", ["rtest"])])
    assert "ran r" not in edit.run_test()["output"]
    assert "rtest" not in edit.test_command_text()
    # With the primary language's tests gone as well there is nothing to hand over
    with monkeypatch.context() as patch:
        patch.setattr(config, "TEST_COMMAND", [])
        assert edit.test_can_run() is False
    (_two_language_repo / "tests" / "testthat").mkdir()
    assert "ran r" in edit.run_test()["output"]


# == what the other languages' tools write =====================================

@pytest.fixture
def _three_language_repo(tmp_path, monkeypatch):
    """A Python repository holding a Rust crate in svc/, two C# projects in tools/ and
    a Java file with no dependency file. Every command stands in as "<ext>tool"."""
    from conftest import git_repo
    root = pathlib.Path(git_repo(tmp_path / "repo"))
    for name, text in (("svc/Cargo.toml", "[package]\n"), ("svc/src/lib.rs", "\n"),
                       ("tools/gen/Gen.csproj", "<Project/>\n"),
                       ("tools/gen/Gen.cs", "class Gen {}\n"),
                       ("tools/gen/sub/Sub.csproj", "<Project/>\n"),
                       ("tools/lib/Lib.csproj", "<Project/>\n"),
                       ("tools/lib/Lib.cs", "class Lib {}\n"),
                       ("legacy/Old.java", "class Old {}\n"),
                       ("tools/bin/kept.sh", "#!/bin/sh\n")):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "files"], check=True)
    for name in ("rstool", "cstool", "javatool"):
        _test_tool(tmp_path, name, "true")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [
        ("rs", ["rstool"]), ("cs", ["cstool"]), ("java", ["javatool"])])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [])
    return root


_RS_UNDER = ["/svc/**/target/"]
# Every directory holding a project file, without the one under another of them
_CS_UNDER = ["/tools/gen/**/bin/", "/tools/gen/**/obj/",
             "/tools/lib/**/bin/", "/tools/lib/**/obj/"]


def test_a_language_s_build_output_is_left_out_under_where_its_projects_are(
        _three_language_repo):
    # Java has no dependency file here, so its tests do not run and nothing is named
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == _RS_UNDER + _CS_UNDER


def test_a_language_whose_command_is_not_here_names_no_build_output(
        _three_language_repo, monkeypatch):
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [
        ("rs", ["no-such-test-tool"]), ("cs", ["cstool"])])
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == _CS_UNDER
    monkeypatch.setattr(config, "RUN_TEST", False)
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == []


def test_a_dependency_file_in_the_root_leaves_the_build_output_out_at_any_depth(
        _three_language_repo):
    (_three_language_repo / "Cargo.toml").write_text("[workspace]\n")
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == ["target/"] + _CS_UNDER


def test_a_file_of_the_root_that_is_no_project_leaves_the_patterns_where_they_were(
        _three_language_repo):
    """C# builds beside its project files, wherever its settings files are."""
    (_three_language_repo / "global.json").write_text("{}\n")
    (_three_language_repo / "NuGet.Config").write_text("<configuration/>\n")
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == _RS_UNDER + _CS_UNDER
    # Its tests are run from the nearest directory that holds a project
    assert edit.other_test_list()[1] == ("cs", ["cstool", "Gen.csproj"], "tools/gen")
    # and from the root once the root holds a solution
    (_three_language_repo / "All.sln").write_text("\n")
    assert edit.other_test_list()[1] == ("cs", ["cstool", "All.sln"], "")
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == _RS_UNDER + _CS_UNDER


def test_a_dependency_file_that_is_gone_from_the_tree_counts_no_more(
        _three_language_repo):
    """A file git still tracks and the run has deleted is not a place to start from."""
    root = _three_language_repo
    shutil.rmtree(root / "svc")
    (root / "tools" / "gen" / "Gen.csproj").unlink()
    assert edit.other_test_list() == [("cs", ["cstool", "Lib.csproj"], "tools/lib")]
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == [
        "/tools/gen/sub/**/bin/", "/tools/gen/sub/**/obj/",
        "/tools/lib/**/bin/", "/tools/lib/**/obj/"]


def test_a_solution_below_the_root_is_where_the_tests_start(_three_language_repo):
    """The nearest directory in which the target is one file, a solution first."""
    (_three_language_repo / "tools" / "All.sln").write_text("\n")
    assert edit.other_test_list()[1] == ("cs", ["cstool", "All.sln"], "tools")
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == _RS_UNDER + _CS_UNDER


def test_a_root_target_of_another_language_is_never_read_as_an_option(
        _three_language_repo):
    (_three_language_repo / "tools" / "-All.sln").write_text("\n")
    assert edit.other_test_list()[1] == ("cs", ["cstool", "./-All.sln"], "tools")


def test_a_repository_of_one_language_lists_no_file_for_this(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [])
    monkeypatch.setattr(edit.gitcmd, "work_tree_file_list",
                        lambda project_dir: pytest.fail("the files were listed"))
    assert edit.other_output_place_list() == []


def test_a_language_with_no_project_to_start_from_runs_no_test(_three_language_repo):
    """Two projects in one directory leave the command nothing to be pointed at."""
    for name in ("tools/gen/Gen.csproj", "tools/gen/sub/Sub.csproj"):
        (_three_language_repo / name).unlink()
    (_three_language_repo / "tools" / "lib" / "More.csproj").write_text("<Project/>\n")
    assert [ext for ext, _, _ in edit.other_test_list()] == ["rs"]
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == _RS_UNDER


def test_a_build_output_pattern_names_the_directories_it_matches():
    place_list = [("svc", ("__pycache__/", "*.egg-info/"))]
    assert edit.in_other_build_output("svc/src/calc.egg-info/PKG-INFO", place_list)
    assert not edit.in_other_build_output("svc/src/calc.py", place_list)


def test_grep_does_not_read_what_another_language_s_tools_write(_three_language_repo):
    place_list = edit.other_output_place_list()
    assert place_list == [("svc", ("target/",)), ("tools/gen", ("bin/", "obj/")),
                          ("tools/lib", ("bin/", "obj/"))]
    for path, is_output in (("tools/gen/obj/Debug/Gen.AssemblyInfo.cs", True),
                            ("tools/gen/sub/bin/x.cs", True),
                            ("svc/target/debug/build.rs", True),
                            ("tools/bin/kept.sh", False), ("bin/run.cs", False),
                            ("tools/gen/Gen.cs", False), ("tools/gen/obj", False),
                            ("svc/src/target.rs", False), ("web/target/a.rs", False)):
        assert edit.in_other_build_output(path, place_list) is is_output, path
    root = _three_language_repo
    for name in ("tools/gen/obj/Gen.AssemblyInfo.cs", "tools/bin/new.cs"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("class Marker {}\n")
    hit_list = list(source._iter_match(re.compile("Marker"), ["**/*.cs"], [], []))
    assert [path for path, _, _ in hit_list] == ["tools/bin/new.cs"]


def test_a_check_leaves_the_build_output_out_when_the_root_gives_it_work(
        _three_language_repo, monkeypatch):
    """The C# check is given the solution in the root and builds every project of it."""
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [
        ("rs", ["rstool"]), ("cs", ["cstool"]), ("java", ["javatool"])])
    # With nothing of a language in the root its check has nothing to work on
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == []
    (_three_language_repo / "All.sln").write_text("\n")
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == _CS_UNDER
    (_three_language_repo / "Cargo.toml").write_text("[workspace]\n")
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == ["target/"] + _CS_UNDER
    # A check that is not here writes nothing
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [("cs", ["no-such-checker-here"])])
    assert edit.output_pattern_list_of(edit.other_output_place_list()) == []


def test_what_the_index_s_primary_language_writes_is_left_out_as_well(tmp_path,
                                                                       monkeypatch):
    """The primary language here is not the one the clone was started for: a change
    turned a COBOL repository into a Python one."""
    from conftest import git_repo
    root = pathlib.Path(git_repo(tmp_path / "repo"))
    (root / "MAIN.cbl").write_text("       PROCEDURE DIVISION.\n")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "cobol"], check=True)
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    monkeypatch.setattr(config, "OTHER_CHECK_LIST", [])
    assert edit._output_pattern_pair()[0] == ["__pycache__/", "uv.lock", "*.egg-info/"]
    assert edit.exclude_build_output() == ["__pycache__/", "uv.lock", "*.egg-info/"]
    for name in ("program.py", "__pycache__/program.cpython-312.pyc",
                 "tests/__pycache__/test_program.cpython-312.pyc"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x\n")
    assert gitcmd.status_path_list(str(root)) == ["program.py"]
    monkeypatch.setattr(config, "PRIMARY_EXT", "cbl")
    assert edit._output_pattern_pair()[0] == []


def test_a_directory_name_git_would_read_as_a_pattern_is_named_as_itself(
        _three_language_repo):
    odd = _three_language_repo / "a [b]*" / "x"
    odd.mkdir(parents=True)
    (odd / "Cargo.toml").write_text("[package]\n")
    assert "/a \\[b]\\*/x/**/target/" in edit.output_pattern_list_of(edit.other_output_place_list())
    edit.exclude_build_output()
    (odd / "target").mkdir()
    (odd / "target" / "o.rlib").write_text("x\n")
    assert gitcmd.status_path_list(str(_three_language_repo)) == ["a [b]*/x/Cargo.toml"]


def test_what_is_left_out_is_new_files_where_the_tools_write_and_nowhere_else(
        _three_language_repo, caplog):
    root = _three_language_repo
    caplog.set_level("INFO", logger="codetwine_loom.tools.edit")
    assert edit.exclude_build_output() == ["__pycache__/", "uv.lock", "*.egg-info/"] \
        + _RS_UNDER + _CS_UNDER
    assert "/svc/**/target/" in caplog.text
    for name in ("svc/target/debug/x.o", "svc/sub/target/y.o", "tools/gen/bin/Gen.dll",
                 "tools/gen/sub/obj/Sub.dll", "tools/lib/obj/Lib.dll",
                 "tools/bin/new.sh", "web/target/page.html", "bin/run.sh"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x\n")
    (root / "tools" / "bin" / "kept.sh").write_text("#!/bin/sh\nexit 0\n")
    assert sorted(gitcmd.status_path_list(str(root))) == [
        "bin/run.sh", "tools/bin/kept.sh", "tools/bin/new.sh", "web/target/page.html"]
    # The editing tools write where nothing is left out, and refuse where it is
    assert edit.create_file("tools/bin/more.sh", "#!/bin/sh\n")["ok"] is True
    assert "git ignores" in edit.create_file("svc/target/gen.rs", "\n")["error"]
    # Asked again, nothing is added twice
    assert edit.exclude_build_output() == []


def test_a_project_made_during_the_run_is_left_out_before_the_next_command_runs(
        _three_language_repo, monkeypatch):
    """Inside a container the tests and the checks ask again before they run."""
    root = _three_language_repo
    monkeypatch.setattr(config, "TEST_COMMAND", [])
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    (root / "new" / "app").mkdir(parents=True)
    (root / "new" / "app" / "App.csproj").write_text("<Project/>\n")
    edit.run_test()
    _RUN_CHECK()
    assert not (root / ".git" / "info" / "exclude").read_text().count("/new/app/")
    monkeypatch.setenv("CTLOOM_WORK", str(root))
    edit.run_test()
    assert "/new/app/**/bin/\n" in (root / ".git" / "info" / "exclude").read_text()
    (root / "newer").mkdir()
    (root / "newer" / "Cargo.toml").write_text("[package]\n")
    _RUN_CHECK()
    assert "/newer/**/target/\n" in (root / ".git" / "info" / "exclude").read_text()


def test_a_directory_that_is_not_a_repository_is_left_as_it_is(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    assert gitcmd.add_exclude(str(tmp_path), ["target/"]) == []
    assert os.listdir(tmp_path) == []


def test_a_pattern_is_written_on_a_line_of_its_own(_three_language_repo):
    exclude = _three_language_repo / ".git" / "info" / "exclude"
    exclude.write_text("kept-one\nkept-two")
    assert gitcmd.add_exclude(str(_three_language_repo), ["/a/**/target/", "kept-one"]) == [
        "/a/**/target/"]
    assert exclude.read_text() == "kept-one\nkept-two\n/a/**/target/\n"
    exclude.unlink()
    exclude.parent.rmdir()
    assert gitcmd.add_exclude(str(_three_language_repo), ["bin/"]) == ["bin/"]
    assert exclude.read_text() == "bin/\n"
    # A last line ended by a carriage return alone is ended first too
    exclude.write_bytes(b"kept/\r")
    assert gitcmd.add_exclude(str(_three_language_repo), ["obj/"]) == ["obj/"]
    assert exclude.read_bytes() == b"kept/\r\nobj/\n"
    assert gitcmd.is_ignored(str(_three_language_repo), "x/obj/a.o")


def test_the_tests_can_run_when_only_another_language_has_some(_two_language_repo,
                                                                monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND", [])
    assert edit.test_can_run() is True
    result = edit.run_test()
    assert result["output"].strip() == "ran in web with:"
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    assert edit.test_can_run() is False


def test_the_text_of_the_test_commands_names_every_command_that_runs(_two_language_repo,
                                                                     monkeypatch):
    primary = " ".join(config.TEST_COMMAND)
    assert edit.test_command_text() == f"{primary} / jstest (in web)"
    # The primary command is left out where it cannot be run
    (_two_language_repo / "tests" / "test_a.py").unlink()
    (_two_language_repo / "tests").rmdir()
    assert edit.test_command_text() == "jstest (in web)"
    # With no other language it is TEST_COMMAND as it is, whether or not it can be run
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    assert edit.test_command_text() == primary


@pytest.mark.parametrize("ext, name, output, is_fail_stop", [
    ("py", "", "FAILED tests/test_a.py::test_x - assert 0\n"
               "!!!!!!!! stopping after 1 failures !!!!!!!!\n1 failed in 0.1s", True),
    ("py", "", "FAILED tests/test_a.py::test_x - assert 0\n"
               "!!!!!!!! stopping after 3 failures !!!!!!!!\n3 failed in 0.1s", True),
    ("py", "", "FAILED tests/test_a.py::test_x - assert 0\n!!!! xdist.dsession."
               "Interrupted: stopping after 1 failures !!!!\n1 failed in 0.1s", True),
    ("py", "", "FAILED tests/test_a.py::test_x - assert 0\n"
               "!!!!!!!! Interrupted: stopping after 2 failures !!!!!!!!\n2 failed in 1s",
     True),
    ("py", "", "ERROR tests/test_a.py\n!!!! Interrupted: 1 error during collection !!!!\n"
               "1 error in 0.1s", False),
    ("py", "py_B", "", False),
    ("py", "", "FAILED tests/test_a.py::test_x - assert 0\n1 failed in 0.1s", False),
    ("java", "java_R_multi", "", False), ("java", "java_B_multi", "", False),
    ("kt", "kt_B_multi", "", False), ("kt", "kt_B_kapt", "", False),
    ("js", "", "!!!!!!!! stopping after 1 failures !!!!!!!!\nTests: 1 failed", False)])
def test_a_run_that_stopped_at_a_count_of_failing_tests_is_told_from_one_that_ended_early(
        ext, name, output, is_fail_stop, as_language, golden_test_command,
        fake_test_command, monkeypatch):
    """The output is read whole: the tail the run returns holds none of it."""
    monkeypatch.setattr(config, "TEST_OUTPUT_TAIL", 5)
    as_language(ext)
    if name:
        golden_test_command(name)
    else:
        fake_test_command(1, output)
    result = edit.run_test()
    assert edit.fail_stop_run(result) is is_fail_stop
    assert [edit.fail_stop_run(part) for part in edit.part_dict_of(result).values()] == [
        is_fail_stop]
    assert edit.fail_stop_run(dict(result)) is False
    assert edit.fail_stop_run({**result, "output": output}) is (is_fail_stop and ext == "py")
    assert edit.early_stop_run(result) or not is_fail_stop


def test_a_reason_names_the_primary_command_as_it_runs(repo, as_language, monkeypatch):
    _js_repo(repo, as_language, monkeypatch)
    assert edit.command_text("npm test --silent") == "node --test"
    assert edit.command_text("cargo test") == "cargo test"
    (repo / "package.json").write_text("{}")
    assert edit.command_text("npm test --silent") == "npm test --silent"


def test_the_stage_reasons_name_the_primary_command_as_it_runs(repo, as_language,
                                                                monkeypatch):
    from codetwine_loom import testcreate
    _js_repo(repo, as_language, monkeypatch)
    not_run = {"status": "did not run", "failed_excerpt": "boom", "failed": []}
    assert "`node --test`" in testcreate._part_run_reason(
        not_run, frozenset(), "npm test --silent")
    assert "`node --test`" in integrate._part_test_reason(
        not_run, frozenset(), "npm test --silent", False)


def test_a_path_outside_the_repository_is_refused_with_how_paths_are_written(repo):
    out = edit.create_file("/elsewhere/new.py", "x = 1\n")
    assert out["ok"] is False
    assert "Write the path relative to the repository root, e.g." in out["error"]
