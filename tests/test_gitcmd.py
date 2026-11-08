"""The one place git is started from."""

import os
import random
import shutil
import subprocess
import sys
import tempfile

from codetwine_loom import config, gitcmd


def test_automatic_gc_stays_off(monkeypatch):
    """A commit taken in here shares its object store with the read-only mount other
    runs' containers are cloning from, and a repack under a reader breaks the clone."""
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(gitcmd.subprocess, "run", run)
    gitcmd.run("/somewhere", "commit", "-m", "x")
    assert seen["argv"] == ["git", "-C", "/somewhere", *config.GIT_SETTING_TUPLE,
                            "commit", "-m", "x"]
    assert config.GIT_SETTING_TUPLE[::2] == ("-c",) * (len(config.GIT_SETTING_TUPLE) // 2)
    assert {"gc.auto=0", f"core.hooksPath={os.devnull}", "core.quotePath=false",
            "core.fsmonitor=false", "trailer.separators=:",
            "log.showSignature=false"} <= set(config.GIT_SETTING_TUPLE[1::2])


def test_a_command_held_to_an_address_space_is_started_through_the_limit(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(gitcmd.subprocess, "run", run)
    gitcmd.run("/somewhere", "apply", "x.diff", max_address_byte=1 << 30)
    assert seen["argv"] == [sys.executable, "-I", "-S", "-c", gitcmd._LIMIT_START,
                            str(1 << 30), "git", "-C", "/somewhere",
                            *gitcmd._PACK_MAP_TUPLE, *config.GIT_SETTING_TUPLE,
                            "apply", "x.diff"]


def test_a_command_held_to_an_address_space_ends_once_it_asks_for_more(tmp_path):
    """The same command gets through with room for what it reads, and without a
    limit."""
    _git(tmp_path, "init", "-q", "-b", "main")
    content = b"x" * (96 << 20)
    for limit, is_through in ((None, True), (512 << 20, True), (64 << 20, False)):
        proc = gitcmd.run(str(tmp_path), "hash-object", "--stdin", input_text=content,
                          text=False, max_address_byte=limit)
        assert (proc.returncode == 0) is is_through, proc.stderr
        assert is_through or b"Out of memory" in proc.stderr


def test_a_command_held_to_an_address_space_keeps_a_lower_limit_it_runs_under():
    """The process that starts it runs under a soft limit below the one asked for."""
    soft = 1 << 30
    show = "import resource; print(*resource.getrlimit(resource.RLIMIT_AS))"
    code = ("import resource, subprocess, sys\n"
            "hard = resource.getrlimit(resource.RLIMIT_AS)[1]\n"
            f"resource.setrlimit(resource.RLIMIT_AS, ({soft}, hard))\n"
            "sys.exit(subprocess.run([sys.executable, '-I', '-S', '-c', *sys.argv[1:3], "
            "sys.executable, '-c', sys.argv[3]]).returncode)\n")
    for limit, bound in ((gitcmd.MAX_PATCH_ADDRESS_BYTE, soft), (soft // 2, soft // 2)):
        proc = subprocess.run([sys.executable, "-c", code, gitcmd._LIMIT_START, str(limit),
                               show], capture_output=True, text=True, check=False)
        assert proc.stdout.split() == [str(bound), str(bound)], proc.stderr


# == Reading what git says ======================================
import pytest


def test_a_repository_of_its_own_is_listed_whatever_the_user_s_diff_settings(tmp_path):
    """The commit a repository inside this one is at is a path that changed, between
    two commits and in the working tree's status."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for argv in (["init", "--quiet"], ["config", "user.name", "t"],
                 ["config", "user.email", "t@example.com"]):
        subprocess.run(["git", "-C", str(repo), *argv], check=True)
    commit_list = []
    for number in ("1", "2"):
        subprocess.run(["git", "-C", str(repo), "update-index", "--add", "--cacheinfo",
                        f"160000,{number * 40},inner"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "-m", number],
                       check=True)
        commit_list.append(gitcmd.require_head(str(repo)))
    for value in ("none", "all"):
        subprocess.run(["git", "-C", str(repo), "config", "diff.ignoreSubmodules", value],
                       check=True)
        assert gitcmd.name_status(str(repo), *commit_list)["modified"] == ["inner"]
        assert gitcmd.tree_diff_path_list(str(repo), commit_list[0]) == ["inner"]
        assert [entry["path"] for entry in gitcmd.numstat_list(
            str(repo), commit_list[0])] == ["inner"]


@pytest.mark.parametrize("text", [True, False])
def test_a_command_still_running_at_its_time_limit_is_killed_and_answered(tmp_path, text):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    os.mkfifo(tmp_path / ".gitignore")
    proc = gitcmd.run(str(tmp_path), "ls-files", "--others", "--exclude-standard",
                      text=text, timeout_sec=0.5)
    assert proc.returncode == gitcmd.TIMEOUT_STATUS
    message = "git did not end within 0.5 seconds and was killed"
    assert proc.stderr == (message if text else message.encode())
    assert proc.stdout == ("" if text else b"")


def test_the_entries_a_command_lists_are_kept_as_they_are_read(tmp_path):
    """Only the entries asked for are held; a command that lists more bytes than it
    is given, or that is still running at its time limit, is killed and answered."""
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for number in range(300):
        (tmp_path / f"file-{number:03d}.txt").write_text("", encoding="utf-8")
    (tmp_path / "café.toml").write_text("", encoding="utf-8")
    argv = ("ls-files", "-z", "--others", "--exclude-standard")
    proc = gitcmd.run_entry(str(tmp_path), *argv, max_byte=1 << 20,
                            keep=lambda path: path.endswith(("7.txt", ".toml")))
    assert proc.returncode == 0 and len(proc.stdout) == 31
    assert "café.toml" in proc.stdout and "file-297.txt" in proc.stdout
    proc = gitcmd.run_entry(str(tmp_path), *argv, max_byte=1000, keep=lambda path: True)
    assert (proc.returncode, proc.stdout, proc.stderr) == (
        gitcmd.SIZE_STATUS, [], "git wrote more than 1000 bytes and was killed")
    proc = gitcmd.run_entry(str(tmp_path), "ls-files", "--no-such-option", max_byte=1000,
                            keep=lambda path: True)
    assert proc.returncode == 129 and proc.stdout == [] and "no-such-option" in proc.stderr
    os.mkfifo(tmp_path / ".gitignore")
    proc = gitcmd.run_entry(str(tmp_path), *argv, max_byte=1 << 20, timeout_sec=0.5,
                            keep=lambda path: True)
    assert (proc.returncode, proc.stdout, proc.stderr) == (
        gitcmd.TIMEOUT_STATUS, [], "git did not end within 0.5 seconds and was killed")
    assert subprocess.run(["pgrep", "-f", f"git -C {tmp_path} "], capture_output=True,
                          text=True, check=False).stdout.strip() == ""


def test_a_command_given_no_time_limit_is_started_with_none(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(gitcmd.subprocess, "run", run)
    gitcmd.run("/somewhere", "status")
    assert seen["timeout"] is None


@pytest.mark.parametrize("name", ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                                  "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
                                  "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE"])
def test_a_variable_that_places_git_elsewhere_does_not_reach_the_command(
        monkeypatch, name):
    seen = {}

    def run(argv, **kwargs):
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setenv(name, "/another/place")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "kept")
    monkeypatch.setattr(gitcmd.subprocess, "run", run)
    gitcmd.run("/somewhere", "status")
    assert name not in seen and seen["GIT_AUTHOR_NAME"] == "kept"
    # What the caller hands over for one command is what places it
    gitcmd.run("/somewhere", "status", env_dict={name: "/the/callers"})
    assert seen[name] == "/the/callers"


def test_the_repository_asked_about_is_the_one_read_whatever_git_dir_says(
        repo, tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q", "-b", "main")
    _git(other, "-c", "user.email=t@example.invalid", "-c", "user.name=t",
         "commit", "-q", "--allow-empty", "-m", "other")
    head = gitcmd.head_commit(str(repo))
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert gitcmd.head_commit(str(repo)) == head != gitcmd.head_commit(str(other))
    assert gitcmd.status_path_list(str(repo)) == []
    assert gitcmd.work_tree_id(str(repo), keep_object=False) == \
        gitcmd.tree_of(str(repo), head)


def test_a_diff_holding_bytes_that_are_not_utf8_is_read(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    (directory / "a.txt").write_bytes(b"caf\xe9\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "one")
    proc = gitcmd.run(str(directory), "show", "HEAD")
    assert proc.returncode == 0 and "+caf\ufffd" in proc.stdout


def _git(directory, *args):
    """Run one git command in a test repository and return its stdout."""
    done = subprocess.run(["git", "-C", str(directory), *args],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, f"git {args[0]} failed: {done.stderr}"
    return done.stdout


@pytest.fixture
def repo(tmp_path):
    """A repository with two commits, the second changing and adding a file."""
    directory = tmp_path / "repo"
    directory.mkdir()
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    (directory / "a.txt").write_text("one\n", encoding="utf-8")
    (directory / "gone.txt").write_text("gone\n", encoding="utf-8")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "first")
    (directory / "a.txt").write_text("two\n", encoding="utf-8")
    (directory / "b.txt").write_text("new\n", encoding="utf-8")
    (directory / "gone.txt").unlink()
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "second")
    return directory


_HOOK_NAME_TUPLE = ("pre-commit", "commit-msg", "post-commit", "post-checkout",
                    "reference-transaction", "post-merge")


@pytest.mark.parametrize("hook_dir", [".hooks", ".git/hooks"])
def test_no_hook_of_the_repository_runs(repo, hook_dir):
    """Hooks in the working tree (core.hooksPath) and in .git/hooks stay off for every
    command that commits, merges, moves a branch or adds a worktree."""
    marker = repo.parent / "marker"
    hook_path = repo / hook_dir
    hook_path.mkdir(exist_ok=True)
    for name in _HOOK_NAME_TUPLE:
        script = hook_path / name
        script.write_text(f"#!/bin/sh\necho {name} >> '{marker}'\n", encoding="utf-8")
        script.chmod(0o755)
    if hook_dir == ".hooks":
        _git(repo, "config", "core.hooksPath", ".hooks")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "the hooks run for git itself")
    assert marker.exists()
    marker.unlink()

    project = str(repo)
    for args in (("commit", "-q", "--allow-empty", "-m", "x"),
                 ("branch", "side"),
                 ("branch", "-f", "side", "HEAD~1"),
                 ("merge", "-q", "--no-ff", "-m", "m", "side"),
                 ("worktree", "add", "-q", str(repo.parent / "wt"), "side"),
                 ("worktree", "remove", "--force", str(repo.parent / "wt")),
                 ("branch", "-D", "side")):
        assert gitcmd.run(project, *args).returncode == 0, args
    assert not marker.exists()


def test_short_keeps_the_head_of_a_hash_and_leaves_an_empty_one_empty():
    assert gitcmd.short("0123456789abcdef0123") == "0123456789ab"
    assert gitcmd.short("") == ""


def test_an_empty_repository_path_is_refused_before_git_runs():
    with pytest.raises(FileNotFoundError, match="PROJECT_DIR is not set"):
        gitcmd.run("", "status")


def test_the_status_lists_every_changed_path_and_the_new_ones_alone(repo):
    (repo / "a.txt").write_text("three\n", encoding="utf-8")
    (repo / "d").mkdir()
    (repo / "d" / "c.txt").write_text("c\n", encoding="utf-8")
    assert gitcmd.status_path_list(str(repo)) == ["a.txt", "d/c.txt"]
    assert gitcmd.new_file_list(str(repo)) == ["d/c.txt"]


def test_the_repository_of_its_own_above_a_path_is_named(repo, tmp_path):
    commit = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},lib/sub one")
    (repo / "nest" / "deep").mkdir(parents=True)
    _git(repo / "nest" / "deep", "init", "-q")
    root = str(repo)
    assert gitcmd.gitlink_of(root, "lib/sub one/pkg/a.py") == "lib/sub one"
    assert gitcmd.gitlink_of(root, "lib/sub one/a.py") == "lib/sub one"
    assert gitcmd.gitlink_of(root, "nest/deep/a.py") == "nest/deep"
    # The entry itself is one
    assert gitcmd.gitlink_of(root, "lib/sub one") == "lib/sub one"
    # A file beside it, a file of the repository and the repository root are under none
    assert gitcmd.gitlink_of(root, ".") == "" and gitcmd.gitlink_of(root, "") == ""
    assert gitcmd.gitlink_of(root, "lib/a.py") == ""
    assert gitcmd.gitlink_of(root, "nest/a.py") == ""
    assert gitcmd.gitlink_of(root, "a.txt") == ""
    assert gitcmd.gitlink_of(str(tmp_path), "d/a.txt") == ""


def test_a_path_is_held_to_the_entries_of_another_repository_it_is_given(repo):
    (repo / "nest").mkdir()
    (repo / "nest" / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    root = str(repo)
    assert gitcmd.nest_dir_of(root, b"sub/pkg/a.py", {b"sub"}) == b"sub"
    assert gitcmd.nest_dir_of(root, b"sub", {b"sub"}) == b"sub"
    assert gitcmd.nest_dir_of(root, b"out/sub/in/a.py", {b"out/sub/in", b"out/sub"}) \
        == b"out/sub"
    assert gitcmd.nest_dir_of(root, b"nest/a.py", set()) == b"nest"
    assert gitcmd.nest_dir_of(root, b"nest", set()) == b""
    assert gitcmd.nest_dir_of(root, b"subs/a.py", {b"sub"}) == b""
    assert gitcmd.nest_dir_of(root, b"\xff/a.py", {b"\xff"}) == b"\xff"


def test_the_entries_of_an_index_come_with_their_flags_and_modes(repo, tmp_path):
    commit = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},lib")
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    (repo / "c.txt").write_text("c\n", encoding="utf-8")
    _git(repo, "add", "b.txt", "c.txt")
    _git(repo, "update-index", "--skip-worktree", "b.txt")
    _git(repo, "update-index", "--assume-unchanged", "c.txt")
    entry_dict = {path: (tag, mode)
                  for tag, mode, path in gitcmd.index_entry_list(str(repo))}
    assert entry_dict[b"lib"][1] == gitcmd.GITLINK_MODE
    assert entry_dict[b"a.txt"] == (b"H", b"100644")
    flag_dict = {path: (gitcmd.is_assume_unchanged(tag), gitcmd.is_skip_worktree(tag))
                 for path, (tag, _) in entry_dict.items()}
    assert flag_dict[b"a.txt"] == (False, False)
    assert flag_dict[b"b.txt"] == (False, True) and flag_dict[b"c.txt"] == (True, False)
    assert gitcmd.index_entry_list(str(tmp_path / "nowhere")) is None


def test_a_directory_that_is_not_a_repository_has_no_status(tmp_path):
    assert gitcmd.status_path_list(str(tmp_path)) == []


def test_an_ancestor_is_one_the_other_commit_reaches(repo):
    first = _git(repo, "rev-parse", "HEAD~1").strip()
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert gitcmd.is_ancestor(str(repo), first, head) is True
    assert gitcmd.is_ancestor(str(repo), head, first) is False
    assert gitcmd.is_ancestor(str(repo), "nosuch", head) is False


# The git settings of a user that change what "git diff" prints or what "git apply"
# does with the whitespace of a line
_USER_SETTING_LIST = [("diff.noprefix", "true"), ("diff.mnemonicPrefix", "true"),
                      ("diff.external", "/bin/echo"), ("color.diff", "always"),
                      ("color.ui", "always"), ("apply.whitespace", "fix"),
                      ("apply.whitespace", "error")]


# The diff settings of a user that change the hunks, the file headers or the order of
# the files "git diff" writes
_DIFF_SHAPE_SETTING_LIST = [
    ("diff.context", "0"), ("diff.context", "9"), ("diff.interHunkContext", "9"),
    ("diff.algorithm", "patience"), ("diff.indentHeuristic", "false"),
    ("diff.renames", "false"), ("diff.renames", "copies"), ("diff.relative", "true"),
    ("diff.submodule", "log"), ("diff.ignoreSubmodules", "all"),
    ("diff.orderFile", "order.txt"), ("diff.suppressBlankEmpty", "true"),
    ("diff.wsErrorHighlight", "all"), ("diff.colorMoved", "zebra")]


def _repo_with_a_change_of_every_shape(tmp_path):
    """Make a repository whose second commit changes two places of one file a few
    lines apart with a blank line beside the first, moves a file and changes one in
    a directory. HEAD is left at the first commit.

    Returns:
        (the repository, the first commit, the second commit).
    """
    directory = tmp_path / "shape"
    directory.mkdir()
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    (directory / "sub").mkdir()
    line_list = [f"line {n}" for n in range(1, 21)]
    line_list[4] = ""
    (directory / "a.txt").write_text("\n".join(line_list) + "\n", encoding="utf-8")
    (directory / "old.txt").write_text("moved\nas\nit\nis\n", encoding="utf-8")
    (directory / "sub" / "s.txt").write_text("s\n", encoding="utf-8")
    (directory / "order.txt").write_text("sub/*\nnew.txt\n", encoding="utf-8")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "first")
    first = _git(directory, "rev-parse", "HEAD").strip()
    line_list[2], line_list[11] = "LINE 3", "LINE 12"
    (directory / "a.txt").write_text("\n".join(line_list) + "\n", encoding="utf-8")
    (directory / "sub" / "s.txt").write_text("s2\n", encoding="utf-8")
    _git(directory, "mv", "old.txt", "new.txt")
    _git(directory, "commit", "-qam", "second")
    second = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "--detach", first)
    return directory, first, second


@pytest.mark.parametrize("key, value", _DIFF_SHAPE_SETTING_LIST)
def test_a_patch_to_apply_keeps_its_shape_and_applies_whatever_the_user_s_diff_settings(
        tmp_path, key, value):
    directory, first, second = _repo_with_a_change_of_every_shape(tmp_path)
    plain, with_setting = str(tmp_path / "plain.diff"), str(tmp_path / "set.diff")
    assert gitcmd.write_patch(str(directory), first, second, plain).returncode == 0
    plain_text = gitcmd.diff_text(str(directory), first, second)
    _git(directory, "config", key, value)
    assert gitcmd.write_patch(str(directory), first, second, with_setting).returncode == 0
    with open(plain, "rb") as fp, open(with_setting, "rb") as set_fp:
        assert fp.read() == set_fp.read()
    assert gitcmd.diff_text(str(directory), first, second) == plain_text
    assert "@@ -1,6 +1,6 @@" in plain_text and "@@ -9,7 +9,7 @@" in plain_text
    assert "rename from" not in plain_text and "\n \n" in plain_text
    _git(directory, *gitcmd.APPLY_TUPLE, with_setting)
    assert gitcmd.work_tree_id(str(directory)) == gitcmd.tree_of(str(directory), second)


@pytest.mark.parametrize("key, value", [
    ("diff.default.binary", "true"), ("diff.default.binary", "false"),
    ("diff.renameLimit", "1")])
def test_binary_files_and_moves_are_read_the_same_whatever_the_user_s_diff_settings(
        tmp_path, key, value):
    """A file is binary or text by its contents, and a file that moved with a change
    is found, with a setting of the user's that says otherwise for every file."""
    directory = tmp_path / "repo"
    (directory / "dir").mkdir(parents=True)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    text_dict = {name: "".join(f"{name} {n}\n" for n in range(40)) for name in "bd"}
    for name, text in text_dict.items():
        (directory / "dir" / f"{name}.txt").write_text(text, encoding="utf-8")
    (directory / "i.bin").write_bytes(bytes(range(256)) * 4)
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "first")
    first = _git(directory, "rev-parse", "HEAD").strip()
    (directory / "to").mkdir()
    for name, text in text_dict.items():
        (directory / "dir" / f"{name}.txt").unlink()
        (directory / "to" / f"{name}2.txt").write_text(text + "more\n", encoding="utf-8")
    (directory / "i.bin").write_bytes(bytes(reversed(range(256))) * 4)
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "second")
    second = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "--detach", first)

    def read():
        patch = str(tmp_path / "one.diff")
        assert gitcmd.write_patch(str(directory), first, second, patch).returncode == 0
        with open(patch, "rb") as handle:
            return (handle.read(), gitcmd.move_dict(str(directory), first, second),
                    gitcmd.numstat_list(str(directory), second))

    plain = read()
    assert b"GIT binary patch" in plain[0] and b"+b 39\n+more\n" in plain[0]
    assert plain[1] == {"dir/b.txt": "to/b2.txt", "dir/d.txt": "to/d2.txt"}
    _git(directory, "config", key, value)
    assert read() == plain


@pytest.mark.parametrize("key, value", _USER_SETTING_LIST)
def test_a_patch_to_apply_is_written_the_same_whatever_the_user_s_git_settings(
        repo, tmp_path, key, value):
    first = _git(repo, "rev-parse", "HEAD~1").strip()
    head = _git(repo, "rev-parse", "HEAD").strip()
    plain, with_setting = str(tmp_path / "plain.diff"), str(tmp_path / "set.diff")
    assert gitcmd.write_patch(str(repo), first, head, plain).returncode == 0
    _git(repo, "config", key, value)
    assert gitcmd.write_patch(str(repo), first, head, with_setting).returncode == 0
    with open(plain, "rb") as fp, open(with_setting, "rb") as set_fp:
        assert fp.read() == set_fp.read()
    assert gitcmd.diff_text(str(repo), first, head).startswith("diff --git a/")
    # The tree is told to hold the change, and not to once the change is undone
    assert gitcmd.has_change_in_tree(str(repo), first, head) is True
    _git(repo, "checkout", "-q", first, "--", ".")
    _git(repo, "clean", "-fdq")
    assert gitcmd.has_change_in_tree(str(repo), first, head) is False


def test_the_working_tree_holds_a_change_that_applies_to_it_in_reverse(repo, tmp_path):
    first = _git(repo, "rev-parse", "HEAD~1").strip()
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert gitcmd.has_change_in_tree(str(repo), first, head) is True
    assert gitcmd.has_change_in_tree(str(repo), head, head) is True
    # A change of the user's own elsewhere in the tree does not count against it
    (repo / "other.txt").write_text("mine\n", encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), first, head) is True
    (repo / "b.txt").unlink()
    assert gitcmd.has_change_in_tree(str(repo), first, head) is False
    assert gitcmd.has_change_in_tree(str(repo), "nosuch", head) is None


def test_the_start_of_a_diff_is_read_with_the_count_of_what_follows(tmp_path):
    """The start and the count make up the diff as it is read whole, a line ending
    and a byte that is not UTF-8 read the same way."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_bytes(b"one\r\ntwo\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one")
    first = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "a.txt").write_bytes(b"one\r\ncaf\xe9 \xe3\x81\x82\n" + b"more\n" * 30000)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "two")
    head = _git(repo, "rev-parse", "HEAD").strip()
    whole_text = gitcmd.diff_text(str(repo), first, head)
    assert len(whole_text) > 150000
    for max_char in (1, 60, 100, 70000, len(whole_text) - 1, len(whole_text)):
        start, rest_count = gitcmd.diff_head(str(repo), first, head, max_char)
        assert (start, rest_count) == (whole_text[:max_char], len(whole_text) - max_char)
    for max_char in (0, len(whole_text) + 5):
        assert gitcmd.diff_head(str(repo), first, head, max_char) == (whole_text, 0)
    assert gitcmd.diff_head(str(repo), "nosuch", head, 10) == ("", 0)


@pytest.mark.parametrize("with_text", [False, True])
def test_a_change_of_a_file_s_executable_bit_is_held_only_with_the_bit(repo, tmp_path,
                                                                       with_text):
    """git's own check passes a file whose mode is not the one the change gives it."""
    head = _git(repo, "rev-parse", "HEAD").strip()
    name = sorted(_git(repo, "ls-files").split())[0]
    os.chmod(repo / name, 0o755)
    if with_text:
        with open(repo / name, "a", encoding="utf-8") as handle:
            handle.write("one more line\n")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "mode")
    mode = _git(repo, "rev-parse", "HEAD").strip()
    assert gitcmd.has_change_in_tree(str(repo), head, mode) is True
    os.chmod(repo / name, 0o644)
    assert gitcmd.has_change_in_tree(str(repo), head, mode) is False
    # The bit is not compared where the repository does not keep it
    _git(repo, "config", "core.fileMode", "false")
    assert gitcmd.has_change_in_tree(str(repo), head, mode) is True
    _git(repo, "config", "core.fileMode", "true")
    # The change the other way round clears the bit
    assert gitcmd.has_change_in_tree(str(repo), mode, head) is not with_text
    os.chmod(repo / name, 0o755)
    assert gitcmd.has_change_in_tree(str(repo), head, mode) is True
    (repo / name).unlink()
    assert gitcmd.has_change_in_tree(str(repo), head, mode) is False


def _put_entry(directory, kind):
    """Make the path "p" of a working tree one kind of entry: "file644", "file755",
    "link", "none", or a file or a link of other content ("file644t", "linkt")."""
    path = directory / "p"
    if os.path.lexists(path):
        os.remove(path)
    if kind.startswith("file"):
        path.write_text("target\n" if kind.endswith("t") else "text\n", encoding="utf-8")
        os.chmod(path, 0o755 if "755" in kind else 0o644)
    elif kind.startswith("link"):
        os.symlink("target" if kind.endswith("t") else "text", path)


@pytest.mark.parametrize("before, after, held_list", [
    # What the change leaves at the path, against a working tree that holds there a
    # file without the bit, a file with it, a link, and nothing
    ("none", "file755", [False, True, False, False]),
    ("none", "file644", [True, False, False, False]),
    ("none", "link", [False, False, True, False]),
    ("file644", "file755", [False, True, False, False]),
    ("file755", "file644", [True, False, False, False]),
    ("file644", "link", [False, False, True, False]),
    ("link", "file644", [True, False, False, False]),
    ("file755", "none", [False, False, False, True]),
    ("link", "none", [False, False, False, True]),
    # A change that leaves the mode as it was does not say what the bit is
    ("file644t", "file644", [True, True, False, False]),
    ("linkt", "link", [False, False, True, False])])
def test_the_tree_holds_a_change_only_with_the_kind_and_the_bit_the_change_leaves(
        repo, tmp_path, before, after, held_list):
    _put_entry(repo, before)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "before")
    start = _git(repo, "rev-parse", "HEAD").strip()
    _put_entry(repo, after)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "after")
    end = _git(repo, "rev-parse", "HEAD").strip()
    tree_list = ["file644", "file755", "link", "none"]
    for kind, is_held in zip(tree_list, held_list):
        _put_entry(repo, kind)
        assert gitcmd.has_change_in_tree(str(repo), start, end) is is_held, kind
    # A repository that does not keep the bit is not asked about it
    _git(repo, "config", "core.fileMode", "false")
    for kind, is_held in zip(tree_list, held_list):
        _put_entry(repo, kind)
        is_file = kind.startswith("file") and after.startswith("file")
        assert gitcmd.has_change_in_tree(str(repo), start, end) is (
            is_held or is_file), kind


def _commit_file(repo, file_dict, message="change"):
    """Write files (None takes one away), commit them all and return the commit."""
    for name, content in file_dict.items():
        path = repo / name
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode() if isinstance(content, str) else content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


_LINE_TEXT = "".join(f"line {number}\n" for number in range(1, 41))


def _with(text, old, new):
    """Return text with the first old replaced by new."""
    assert old in text
    return text.replace(old, new, 1)


@pytest.mark.parametrize("start_text, end_text", [
    # The lines after the change are found once more elsewhere in the file
    ("alpha\nbeta\ngamma\n" + "x\n" + "alpha\nbeta\nEXTRA\ngamma\n",
     "alpha\nbeta\ngamma\n" + "x\n" + "alpha\nbeta\ngamma\n"),
    ("same\n" * 12, "same\n" * 13)])
def test_a_change_whose_lines_are_found_elsewhere_in_the_file_is_held_only_when_made(
        repo, tmp_path, start_text, end_text):
    start = _commit_file(repo, {"r.txt": start_text})
    end = _commit_file(repo, {"r.txt": end_text})
    (repo / "r.txt").write_text(start_text, encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False
    (repo / "r.txt").write_text(end_text, encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is True


def test_a_repository_of_its_own_is_held_only_at_the_commit_the_change_names(
        repo, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _git(sub, "init", "-q", "-b", "main")
    _git(sub, "config", "user.email", "t@example.invalid")
    _git(sub, "config", "user.name", "t")
    first_sub = _commit_file(sub, {"s.txt": "one\n"}, "c1")
    second_sub = _commit_file(sub, {"s.txt": "two\n"}, "c2")
    shutil.copytree(sub, repo / "sub", symlinks=True)
    _git(repo / "sub", "checkout", "-q", first_sub)
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first_sub},sub")
    _git(repo, "commit", "-qm", "start")
    start = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "a.txt").write_text("more\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "update-index", "--cacheinfo", f"160000,{second_sub},sub")
    _git(repo, "commit", "-qm", "move sub")
    end = _git(repo, "rev-parse", "HEAD").strip()
    # a.txt is held and sub stays where it was
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False
    _git(repo / "sub", "checkout", "-q", second_sub)
    assert gitcmd.has_change_in_tree(str(repo), start, end) is True
    # A directory that is not a repository of its own holds no commit
    shutil.rmtree(repo / "sub")
    (repo / "sub").mkdir()
    (repo / "sub" / "mine.txt").write_text("mine\n", encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False
    shutil.rmtree(repo / "sub")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False


def test_what_a_later_commit_takes_back_is_held_only_as_the_last_one_leaves_it(
        repo, tmp_path):
    """A file one commit adds and a later one takes away, and lines one commit adds
    and a later one takes out again, do not show in the change from start to end."""
    merge_text = _with(_with(_LINE_TEXT, "line 6\n", "line 6\nimport unused\n"),
                       "line 31\n", "line 31\ndef added():\n")
    end_text = _with(merge_text, "import unused\n", "")
    start = _commit_file(repo, {"f.py": _LINE_TEXT})
    merge = _commit_file(repo, {"f.py": merge_text, "a.txt": "a\n", "b.txt": "b\n"})
    end = _commit_file(repo, {"f.py": end_text, "b.txt": None})
    assert gitcmd.has_change_in_tree(str(repo), start, end) is True
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False
    (repo / "b.txt").unlink()
    (repo / "f.py").write_text(merge_text, encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False
    assert gitcmd.has_change_in_tree(str(repo), start, merge) is False
    # A change of the tree's own elsewhere in a file both commits change does not
    # count against it; the lines the merge added and the stage took out do
    (repo / "f.py").write_text(_with(end_text, "line 39\n", "line 39 mine\n"),
                               encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is True
    (repo / "f.py").write_text(_with(merge_text, "line 39\n", "line 39 mine\n"),
                               encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False
    # A change that adds a file and takes it away again is held where it is not there
    empty = _commit_file(repo, {"c.txt": "c\n"})
    back = _commit_file(repo, {"c.txt": None})
    assert gitcmd.has_change_in_tree(str(repo), end, back) is True
    (repo / "c.txt").write_text("c\n", encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), end, back) is False
    assert gitcmd.has_change_in_tree(str(repo), end, empty) is True


@pytest.mark.parametrize("autocrlf", ["false", "true"])
def test_a_file_one_commit_changed_is_held_with_changes_of_the_tree_s_own_around_it(
        repo, tmp_path, autocrlf):
    _git(repo, "config", "core.autocrlf", autocrlf)
    end_text = _with(_LINE_TEXT, "line 5\n", "line 5 changed\n")
    start = _commit_file(repo, {"f.txt": _LINE_TEXT})
    end = _commit_file(repo, {"f.txt": end_text})
    ending = "\r\n" if autocrlf == "true" else "\n"
    for tree_text, is_held in (
            (_LINE_TEXT, False),
            (end_text, True),
            (_with(end_text, "line 30\n", "line 30 mine\n"), True),
            (_with(_LINE_TEXT, "line 30\n", "line 30 mine\n"), False),
            (_with(_LINE_TEXT, "line 5\n", "line 5 mine\n"), False),
            (_with(end_text, "line 6\n", "line 6 mine\n"), False),
            (_with(_LINE_TEXT, "line 30\n", "line 30\nline 5 changed\n"), False)):
        (repo / "f.txt").write_bytes(tree_text.replace("\n", ending).encode())
        assert gitcmd.has_change_in_tree(str(repo), start, end) is is_held, \
            tree_text


def test_a_file_several_commits_changed_is_held_with_changes_of_the_tree_s_own_around_it(
        repo, tmp_path):
    merge_text = _with(_LINE_TEXT, "line 5\n", "line 5 merge\n")
    end_text = _with(merge_text, "line 20\n", "line 20 stage\n")
    start = _commit_file(repo, {"f.txt": _LINE_TEXT})
    _commit_file(repo, {"f.txt": merge_text})
    end = _commit_file(repo, {"f.txt": end_text})
    for tree_text, is_held in (
            (end_text, True),
            (_with(end_text, "line 35\n", "line 35 mine\n"), True),
            (_with(merge_text, "line 35\n", "line 35 mine\n"), False),
            (_with(_with(_LINE_TEXT, "line 20\n", "line 20 stage\n"), "line 35\n",
                   "line 35 mine\n"), False),
            (_with(end_text, "line 20 stage\n", "line 20 mine\n"), False)):
        (repo / "f.txt").write_text(tree_text, encoding="utf-8")
        assert gitcmd.has_change_in_tree(str(repo), start, end) is is_held, \
            tree_text


def test_a_file_too_large_to_read_whole_holds_a_change_only_as_the_end_has_it(
        repo, tmp_path, monkeypatch):
    end_text = _with(_LINE_TEXT, "line 5\n", "line 5 changed\n")
    start = _commit_file(repo, {"f.txt": _LINE_TEXT, "l": "x"})
    os.remove(repo / "l")
    os.symlink("y" * 300, repo / "l")
    end = _commit_file(repo, {"f.txt": end_text})
    (repo / "f.txt").write_text(_with(end_text, "line 30\n", "line 30 mine\n"),
                                encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is True
    monkeypatch.setattr(gitcmd.tmpfile, "READ_MAX_BYTE", 200)
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False
    (repo / "f.txt").write_text(end_text, encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False
    monkeypatch.setattr(gitcmd.tmpfile, "READ_MAX_BYTE", 400)
    assert gitcmd.has_change_in_tree(str(repo), start, end) is True


def test_git_config_reads_what_every_git_command_reads_whatever_file_it_alone_is_handed(
        repo, tmp_path, monkeypatch):
    _git(repo, "config", "filter.up.clean", "tools/up.sh")
    (tmp_path / "only.gitconfig").write_text("[filter \"x\"]\n\tclean = cat\n",
                                             encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG", str(tmp_path / "only.gitconfig"))
    assert "tools/up.sh" in gitcmd.setting_command_dict(str(repo))
    assert "cat" not in gitcmd.setting_command_dict(str(repo))
    assert "--config-env=filter.up.clean=CTLOOM_NO_DRIVER" in gitcmd.own_driver_tuple(
        str(repo))[0]
    assert "GIT_CONFIG" not in config.child_env()


def test_no_filter_driver_runs_while_the_working_tree_is_read(repo, tmp_path):
    """A path a filter driver is named for is read as it is, without the driver."""
    mark = tmp_path / "ran"
    _git(repo, "config", "filter.up.clean", f"sh -c 'touch {mark}; tr a-z A-Z'")
    _git(repo, "config", "filter.up.smudge", f"sh -c 'touch {mark}; tr A-Z a-z'")
    start = _commit_file(repo, {".gitattributes": "*.u filter=up\n", "k.u": "one\n"})
    end = _commit_file(repo, {"k.u": "two\n", "plain.txt": "two\n"})
    if mark.exists():
        mark.unlink()
    (repo / "k.u").write_text("TWO\n", encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is True
    (repo / "k.u").write_text("two\n", encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False
    (repo / "plain.txt").write_text("two\nmine\n", encoding="utf-8")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False
    assert not mark.exists()


def test_a_path_reached_through_a_link_is_not_held(repo, tmp_path):
    start = _commit_file(repo, {"d/f.txt": "one\n"})
    end = _commit_file(repo, {"d/f.txt": "two\n"})
    assert gitcmd.has_change_in_tree(str(repo), start, end) is True
    elsewhere = tmp_path / "elsewhere"
    shutil.move(repo / "d", elsewhere)
    os.symlink(elsewhere, repo / "d")
    assert gitcmd.has_change_in_tree(str(repo), start, end) is False


def test_a_long_range_of_commits_is_read_with_as_many_git_commands_as_a_short_one(
        repo, monkeypatch):
    """Each commit adds a line to the end of every file; the tree holds the files as
    the end has them with a line of its own at the top of each."""
    name_list = ["f0.txt", "f1.txt", "f2.txt"]
    start = _commit_file(repo, {name: _LINE_TEXT for name in name_list})
    end_list = [_commit_file(repo, {name: _LINE_TEXT + "".join(
        f"more {number}\n" for number in range(count + 1)) for name in name_list})
        for count in range(12)]
    process_list = []
    real_popen = subprocess.Popen

    def counted_popen(*args, **kwargs):
        process_list.append(args[0])
        return real_popen(*args, **kwargs)

    count_list = []
    for count in (2, 12):
        end_text = _LINE_TEXT + "".join(f"more {number}\n" for number in range(count))
        for name in name_list:
            (repo / name).write_text("mine\n" + end_text, encoding="utf-8")
        monkeypatch.setattr(gitcmd.subprocess, "Popen", counted_popen)
        process_list.clear()
        assert gitcmd.has_change_in_tree(str(repo), start, end_list[count - 1]) is True
        count_list.append(len(process_list))
        monkeypatch.setattr(gitcmd.subprocess, "Popen", real_popen)
        # A line one commit added and the tree does not hold is still told
        (repo / name_list[1]).write_text("mine\n" + _with(end_text, "more 0\n", ""),
                                         encoding="utf-8")
        assert gitcmd.has_change_in_tree(str(repo), start, end_list[count - 1]) is False
    assert count_list[0] == count_list[1]


def _merge_file_text(rng, line_tuple, max_line):
    """Return a file of up to max_line lines picked from line_tuple, its last line at
    times with no line feed."""
    text = "".join(rng.choice(line_tuple) for _ in range(rng.randint(0, max_line)))
    return text[:-1] if text and rng.random() < 0.15 else text


def _merge_file_edit(rng, text, line_tuple):
    """Return text with up to four lines put in, taken out or replaced."""
    line_list = gitcmd._record_list(text.encode())
    for _ in range(rng.randint(0, 4)):
        place = rng.randint(0, len(line_list))
        kind = rng.random()
        if kind < 0.4:
            line_list.insert(place, rng.choice(line_tuple).encode())
        elif line_list:
            place = min(place, len(line_list) - 1)
            if kind < 0.7:
                del line_list[place]
            else:
                line_list[place] = rng.choice(line_tuple).encode()
    return b"".join(line_list).decode()


def test_a_merge_into_the_tree_s_file_is_read_as_git_merge_file_reads_it(tmp_path):
    """Versions of few distinct lines, the end made from each and the tree's file made
    from the end or from the version, each read as git merge-file itself merges them."""
    rng = random.Random(18)
    line_tuple = ("a\n", "b\n", "c\n", "\n", "a\r\n")
    case_list = []
    for _ in range(300):
        base_text = _merge_file_text(rng, line_tuple, 12)
        end_text = _merge_file_edit(rng, base_text, line_tuple)
        tree_text = _merge_file_edit(rng, end_text if rng.random() < 0.6 else base_text,
                                     line_tuple)
        case_list.append((tree_text.encode(), base_text.encode(), end_text.encode()))
    for side in ("a", "b"):
        (tmp_path / side).mkdir()
    for index, (tree, base, end) in enumerate(case_list):
        for number, other in ((2 * index, tree), (2 * index + 1, end)):
            (tmp_path / "a" / str(number)).write_bytes(base)
            (tmp_path / "b" / str(number)).write_bytes(other)
    script_dict = gitcmd._pair_script_dict(str(tmp_path))
    env_dict = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_SYSTEM": os.devnull}
    answer_list = []
    for index, (tree, base, end) in enumerate(case_list):
        for name, content in (("tree", tree), ("base", base), ("end", end)):
            (tmp_path / name).write_bytes(content)
        proc = subprocess.run(["git", "merge-file", "-p", "-q", "tree", "base", "end"],
                              cwd=tmp_path, env=env_dict, capture_output=True, check=False)
        is_same = proc.returncode == 0 and proc.stdout == tree
        answer_list.append(is_same)
        assert gitcmd._is_merge_same(
            gitcmd._record_list(tree), gitcmd._record_list(end),
            script_dict.get(2 * index, []), script_dict.get(2 * index + 1, [])) is is_same, \
            (tree, base, end)
    assert True in answer_list and False in answer_list


def test_a_path_git_cat_file_cannot_read_from_its_input_is_read_on_its_own(repo):
    """A path that ends with a carriage return, and one that starts with a space, each
    get the conversion their own name is given in the working tree."""
    name_list = ["c.txt\r", " d.txt", "e.txt"]
    start = _commit_file(repo, {".gitattributes": "*.txt eol=crlf\n/d.txt -text\n",
                                **{name: _LINE_TEXT for name in name_list}})
    end_text = _with(_LINE_TEXT, "line 5\n", "line 5 changed\n")
    end = _commit_file(repo, {name: end_text for name in name_list})
    for tree_text, is_held in ((_with(end_text, "line 30\n", "line 30 mine\n"), True),
                               (_with(_LINE_TEXT, "line 30\n", "line 30 mine\n"), False)):
        for name in name_list:
            ending = "\n" if name.endswith("\r") else "\r\n"
            (repo / name).write_bytes(tree_text.replace("\n", ending).encode())
        assert gitcmd.has_change_in_tree(str(repo), start, end) is is_held


def test_what_one_commit_changed_is_split_by_kind(repo):
    first = _git(repo, "rev-parse", "HEAD~1").strip()
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert gitcmd.name_status(str(repo), first, head) == {
        "added": ["b.txt"], "modified": ["a.txt"], "deleted": ["gone.txt"]}


def test_the_working_tree_is_compared_against_head_when_no_range_is_named(repo):
    (repo / "a.txt").write_text("three\n", encoding="utf-8")
    assert gitcmd.name_status(str(repo)) == {"added": [], "modified": ["a.txt"],
                                             "deleted": []}


def test_the_numstat_counts_the_working_tree_against_the_commit_named(repo):
    (repo / "a.txt").write_text("three\nfour\n", encoding="utf-8")
    first = _git(repo, "rev-parse", "HEAD~1").strip()
    assert gitcmd.numstat_list(str(repo)) == [
        {"path": "a.txt", "insertions": 2, "deletions": 1}]
    assert [e["path"] for e in gitcmd.numstat_list(str(repo), first)] == [
        "a.txt", "b.txt", "gone.txt"]


def test_the_numstat_leaves_out_what_git_does_not_track(repo):
    (repo / "extra.txt").write_text("x\n", encoding="utf-8")
    assert gitcmd.numstat_list(str(repo)) == []


def test_the_numstat_counts_the_other_files_when_a_tracked_path_cannot_be_read(repo):
    """git stops at the named pipe and counts nothing; the pipe is named without counts."""
    (repo / "a.txt").write_text("three\nfour\n", encoding="utf-8")
    (repo / "b.txt").unlink()
    os.mkfifo(repo / "b.txt")
    assert gitcmd.run(str(repo), *gitcmd.PATH_DIFF_TUPLE, "HEAD",
                      "--numstat").returncode != 0
    assert gitcmd.numstat_list(str(repo)) == [
        {"path": "a.txt", "insertions": 2, "deletions": 1},
        {"path": "b.txt", "insertions": gitcmd.UNREAD_COUNT,
         "deletions": gitcmd.UNREAD_COUNT}]


def test_a_diff_of_the_working_tree_is_read_past_a_file_git_cannot_read(repo):
    """The file is staged with a change git counts, and is still named without counts;
    the repository's index keeps no mark."""
    (repo / "b.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "b.txt")
    (repo / "b.txt").unlink()
    os.mkfifo(repo / "b.txt")
    (repo / "a.txt").write_text("three\n", encoding="utf-8")
    proc, unread_list = gitcmd.work_tree_diff(str(repo), "HEAD", *gitcmd.PATH_DIFF_TUPLE,
                                              "HEAD", "--numstat", "-z")
    assert (proc.returncode, unread_list) == (0, [b"b.txt"])
    assert gitcmd.numstat_list(str(repo))[1] == {
        "path": "b.txt", "insertions": gitcmd.UNREAD_COUNT,
        "deletions": gitcmd.UNREAD_COUNT}
    assert "h" not in _git(repo, "ls-files", "-v").split()
    proc, unread_list = gitcmd.work_tree_diff(str(repo), "HEAD", "diff", "HEAD",
                                              "--", "a.txt")
    assert (proc.returncode, unread_list) == (0, [])


def test_the_tracked_paths_of_a_list_are_the_ones_the_index_names(repo):
    (repo / "extra.txt").write_text("x\n", encoding="utf-8")
    assert gitcmd.tracked_path_list(str(repo), ["extra.txt", "a.txt", "*.txt", "no"]) \
        == ["a.txt"]
    assert gitcmd.tracked_path_list(str(repo), []) == []


def test_an_entry_naming_no_path_is_passed_over(monkeypatch):
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, b"M\0a.txt\0D\0\0", b"")

    monkeypatch.setattr(gitcmd.subprocess, "run", run)
    assert gitcmd.name_status("/somewhere") == {"added": [], "modified": ["a.txt"],
                                                "deleted": []}


def _one_commit_repo(path):
    """Make a repository holding one commit of a.py and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.invalid")
    _git(path, "config", "user.name", "t")
    (path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "first")
    return path


# == clean_head: the commit whose tree is on disk =============================
def test_clean_head_is_head_when_nothing_is_changed(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    assert gitcmd.clean_head(str(repo)) == gitcmd.head_commit(str(repo))


@pytest.mark.parametrize("change", ["modify", "add"])
def test_clean_head_is_empty_when_the_tree_differs_from_head(tmp_path, change):
    repo = _one_commit_repo(tmp_path / "repo")
    target = repo / ("a.py" if change == "modify" else "new.py")
    target.write_text("y = 2\n", encoding="utf-8")
    assert gitcmd.head_commit(str(repo)) != ""
    assert gitcmd.clean_head(str(repo)) == ""


def test_clean_head_is_empty_outside_a_repository(tmp_path):
    assert gitcmd.clean_head(str(tmp_path)) == ""
    assert gitcmd.clean_head("") == ""


def test_require_head_refuses_a_directory_with_no_commit(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    assert gitcmd.require_head(str(repo)) == gitcmd.head_commit(str(repo))
    with pytest.raises(RuntimeError, match="not a git repository with a commit"):
        gitcmd.require_head(str(tmp_path / "plain"))


# == is_work_tree_root: what counts as a repository to analyse ================
def test_a_worktree_is_a_repository_although_its_git_is_a_file(tmp_path):
    """git worktree and submodule hold a .git file, and git works in both."""
    repo = _one_commit_repo(tmp_path / "repo")
    tree = tmp_path / "tree"
    _git(repo, "worktree", "add", "--quiet", str(tree))
    assert (tree / ".git").is_file()
    assert gitcmd.is_work_tree_root(str(tree)) is True


def test_is_work_tree_root_refuses_what_is_not_the_top(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / "inside").mkdir()
    assert gitcmd.is_work_tree_root(str(repo)) is True
    assert gitcmd.is_work_tree_root(str(repo / "inside")) is False
    assert gitcmd.is_work_tree_root(str(tmp_path / "plain")) is False
    assert gitcmd.is_work_tree_root("") is False


# == disk_blob_dict / tree_blob_dict: what the files of an analysis hold =======
def test_the_blobs_of_the_files_on_disk_are_the_commit_s_when_nothing_is_changed(
        tmp_path):
    """A clone checked out at that commit gives the same blobs."""
    repo = _one_commit_repo(tmp_path / "repo")
    head = gitcmd.head_commit(str(repo))
    assert gitcmd.disk_blob_dict(str(repo), ["a.py"]) == \
        gitcmd.tree_blob_dict(str(repo), head) == {"a.py": _git(repo, "rev-parse",
                                                                 "HEAD:a.py").strip()}
    assert gitcmd.disk_blob_dict(str(repo), []) == {}


def test_the_files_of_a_tree_leave_out_links_and_repositories_of_their_own(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / "x.sh").write_text("echo\n", encoding="utf-8")
    (repo / "x.sh").chmod(0o755)
    (repo / "in.py").symlink_to("a.py")
    _git(repo, "add", "-A")
    _git(repo, "update-index", "--add", "--cacheinfo",
         f"160000,{gitcmd.head_commit(str(repo))},sub")
    _git(repo, "commit", "-q", "-m", "more")
    assert sorted(gitcmd.tree_blob_dict(str(repo), "HEAD")) == ["a.py", "x.sh"]
    assert gitcmd.tree_blob_dict(str(repo), "no-such-commit") is None


def test_the_blobs_of_the_files_on_disk_leave_the_repository_alone(tmp_path,
                                                                   monkeypatch):
    """The repository's index, its object store and its git directory are as they
    were, and nothing is left in the temporary directory."""
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    repo = _one_commit_repo(tmp_path / "repo")
    before = _object_count(repo)
    (repo / "b.py").write_text("y = 2\n", encoding="utf-8")
    (repo / "a.py").write_text("x = 3\n", encoding="utf-8")
    name_list = sorted(os.listdir(repo / ".git"))
    blob_dict = gitcmd.disk_blob_dict(str(repo), ["a.py", "b.py"])
    assert blob_dict == {
        name: _git(repo, "hash-object", name).strip() for name in ("a.py", "b.py")}
    assert sorted(_git(repo, "status", "--porcelain").splitlines()) == [
        " M a.py", "?? b.py"]
    assert _object_count(repo) == before
    assert sorted(os.listdir(repo / ".git")) == name_list
    assert os.listdir(tmp_path / "tmp") == []


def test_the_index_lent_for_the_working_tree_holds_it_and_is_removed_afterwards(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / "b.py").write_text("y = 2\n", encoding="utf-8")
    with gitcmd.work_tree_index(str(repo)) as (tree, env_dict):
        index_path = env_dict["GIT_INDEX_FILE"]
        assert os.path.isfile(index_path)
        assert gitcmd.run(str(repo), "ls-files", env_dict=env_dict).stdout.split() == [
            "a.py", "b.py"]
        assert gitcmd.run(str(repo), "write-tree", env_dict=env_dict).stdout.strip() == tree
    assert tree == gitcmd.work_tree_id(str(repo))
    assert not os.path.exists(index_path)
    assert _git(repo, "status", "--porcelain").strip() == "?? b.py"


def test_no_index_is_lent_where_there_is_no_repository(tmp_path):
    for directory in (str(tmp_path), ""):
        with gitcmd.work_tree_index(directory) as lent_tuple:
            assert lent_tuple == ("", {})


def test_the_paths_the_working_tree_holds_differently_from_a_commit_are_listed(repo):
    assert gitcmd.tree_diff_path_list(str(repo), "HEAD") == []
    assert sorted(gitcmd.tree_diff_path_list(str(repo), "HEAD~1")) == [
        "a.txt", "b.txt", "gone.txt"]
    (repo / "a.txt").write_text("three\n", encoding="utf-8")
    assert gitcmd.tree_diff_path_list(str(repo), "HEAD") == ["a.txt"]
    assert gitcmd.tree_diff_path_list(str(repo), "no-such-commit") == []


def _object_count(repo) -> int:
    """Return how many loose and packed objects a repository holds."""
    line_dict = dict(line.split(": ", 1) for line in
                     _git(repo, "count-objects", "-v").splitlines())
    return int(line_dict["count"]) + int(line_dict["in-pack"])


def test_the_blobs_are_read_in_a_repository_with_no_index_whose_path_holds_a_separator(
        tmp_path):
    """Without an index of the repository's the tree of HEAD is read, which needs the
    repository's objects."""
    repo = _one_commit_repo(tmp_path / f"re{os.pathsep}po")
    (repo / ".git" / "index").unlink()
    (repo / "b.py").write_text("y = 2\n", encoding="utf-8")
    assert gitcmd.disk_blob_dict(str(repo), ["a.py", "b.py"]) == {
        name: _git(repo, "hash-object", name).strip() for name in ("a.py", "b.py")}
    assert not (repo / ".git" / "index").exists()


def test_the_blobs_are_read_in_a_repository_with_no_commit(tmp_path):
    directory = tmp_path / "fresh"
    directory.mkdir()
    _git(directory, "init", "-q", "-b", "main")
    (directory / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert gitcmd.disk_blob_dict(str(directory), ["a.py"]) == {
        "a.py": _git(directory, "hash-object", "a.py").strip()}


def test_the_blobs_are_the_files_on_disk_whatever_the_index_says_of_them(tmp_path):
    """A file marked assume-unchanged or skip-worktree is read as it is on disk, and
    the repository's own index keeps its marks."""
    repo = _one_commit_repo(tmp_path / "repo")
    for name in ("au.py", "sk.py"):
        (repo / name).write_text("x = 0\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "two more")
    for name, flag in (("au.py", "--assume-unchanged"), ("sk.py", "--skip-worktree")):
        (repo / name).write_text("x = 1\n", encoding="utf-8")
        _git(repo, "update-index", flag, name)
    assert gitcmd.disk_blob_dict(str(repo), ["au.py", "sk.py"]) == {
        name: _git(repo, "hash-object", name).strip() for name in ("au.py", "sk.py")}
    assert _git(repo, "ls-files", "-v", "au.py", "sk.py").split() == ["h", "au.py",
                                                                      "S", "sk.py"]


def test_the_snapshot_takes_the_files_on_disk_whatever_the_index_says_of_them(tmp_path):
    """A file marked assume-unchanged or skip-worktree is taken as it is on disk, and
    the repository's own index keeps its marks."""
    repo = _one_commit_repo(tmp_path / "repo")
    for name in ("au.py", "sk.py"):
        (repo / name).write_text("x = 0\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "two more")
    for name, flag in (("au.py", "--assume-unchanged"), ("sk.py", "--skip-worktree")):
        (repo / name).write_text("x = 1\n", encoding="utf-8")
        _git(repo, "update-index", flag, name)
    commit = gitcmd.snapshot_commit(str(repo))
    assert _git(repo, "show", f"{commit}:au.py") == "x = 1\n"
    assert _git(repo, "show", f"{commit}:sk.py") == "x = 1\n"
    assert _git(repo, "ls-files", "-v", "au.py", "sk.py").split() == ["h", "au.py",
                                                                      "S", "sk.py"]


def test_a_change_git_would_trust_the_stat_data_over_is_still_seen(tmp_path):
    """git compares a file's time to the second, and reads a file again only while the
    index is not older than it. An edit of the same size in that second is seen from
    the repository's own index, so it has to be seen from the copy of it as well."""
    repo = _one_commit_repo(tmp_path / "repo")
    _git(repo, "config", "core.trustctime", "false")
    path, index = repo / "a.py", repo / ".git" / "index"
    stamp = (int(path.stat().st_mtime) - 1000) * 10**9
    os.utime(path, ns=(stamp, stamp))
    _git(repo, "add", "a.py")
    path.write_text("x = 2\n", encoding="utf-8")
    os.utime(path, ns=(stamp, stamp))
    os.utime(index, ns=(stamp, stamp))
    commit = gitcmd.snapshot_commit(str(repo))
    assert _git(repo, "show", f"{commit}:a.py") == "x = 2\n"


def test_the_tree_the_files_make_keeps_the_files_a_sparse_checkout_leaves_out(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / "b").mkdir()
    (repo / "b" / "y.py").write_text("y = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "b")
    _git(repo, "sparse-checkout", "set", "--no-cone", "/a.py")
    assert not (repo / "b" / "y.py").exists()
    assert gitcmd.work_tree_id(str(repo), keep_object=False) == \
        gitcmd.tree_of(str(repo), gitcmd.head_commit(str(repo)))


@pytest.mark.parametrize("define", [["--cone", "src"], ["--no-cone", "/src/", "/a.py"]])
def test_a_file_put_outside_a_sparse_checkout_is_in_the_tree_the_files_make(
        tmp_path, caplog, define):
    """The file is not tracked; a tracked one the checkout leaves out stays as HEAD
    has it."""
    repo = _one_commit_repo(tmp_path / "repo")
    for rel in ("src/s.py", "out/o.py"):
        (repo / rel).parent.mkdir()
        (repo / rel).write_text("v = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "two directories")
    _git(repo, "sparse-checkout", "set", *define)
    assert not (repo / "out").exists()
    (repo / "notes").mkdir()
    (repo / "notes" / "mine.md").write_text("mine\n", encoding="utf-8")
    state = gitcmd.work_tree_id(str(repo), keep_object=False)
    with caplog.at_level("WARNING", logger="codetwine_loom.gitcmd"):
        commit = gitcmd.snapshot_commit(str(repo))
    assert _git(repo, "ls-tree", "-r", "--name-only", commit).split() == [
        "a.py", "notes/mine.md", "out/o.py", "src/s.py"]
    assert caplog.text == ""
    (repo / "notes" / "mine.md").unlink()
    assert gitcmd.work_tree_id(str(repo), keep_object=False) != state


def test_git_that_does_not_know_the_sparse_option_adds_the_files_without_it(
        tmp_path, monkeypatch):
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / "new.py").write_text("n = 1\n", encoding="utf-8")
    real_run = gitcmd.run
    seen_list = []

    def run(project_dir, *args, **kwargs):
        if "add" in args:
            seen_list.append("--sparse" in args)
            if "--sparse" in args:
                return subprocess.CompletedProcess(args, 129, "", "error: unknown option")
        return real_run(project_dir, *args, **kwargs)

    monkeypatch.setattr(gitcmd, "run", run)
    commit = gitcmd.snapshot_commit(str(repo))
    assert seen_list == [True, False]
    assert _git(repo, "show", f"{commit}:new.py") == "n = 1\n"


def test_a_file_that_cannot_be_read_keeps_what_the_index_holds(tmp_path, caplog):
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "a.py").chmod(0)
    (repo / "b.py").write_text("y = 1\n", encoding="utf-8")
    try:
        commit = gitcmd.snapshot_commit(str(repo))
    finally:
        (repo / "a.py").chmod(0o644)
    assert _git(repo, "show", f"{commit}:a.py") == "x = 1\n"
    assert _git(repo, "show", f"{commit}:b.py") == "y = 1\n"
    assert "could not be read" in caplog.text


@pytest.mark.parametrize("leave", ["named pipe", "no read permission"])
def test_the_files_that_cannot_be_read_are_warned_of_once_for_the_same_paths(
        tmp_path, caplog, leave):
    """The warning names them on one line; what git wrote is at DEBUG alone, and the
    same paths again are said at DEBUG."""
    if leave != "named pipe" and os.geteuid() == 0:
        pytest.skip("root reads any file")
    repo = _one_commit_repo(tmp_path / "repo")
    path_list = ["b.py", "c.py"]
    for name in path_list:
        (repo / name).write_text("v = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "two more")

    def leave_unread(name):
        if leave == "named pipe":
            (repo / name).unlink()
            os.mkfifo(repo / name)
        else:
            (repo / name).chmod(0)

    leave_unread("b.py")
    try:
        with caplog.at_level("DEBUG", logger=gitcmd.logger.name):
            for _ in range(3):
                assert gitcmd.work_tree_id(str(repo), keep_object=False)
            leave_unread("c.py")
            gitcmd.work_tree_id(str(repo), keep_object=False)
    finally:
        for name in path_list:
            if (repo / name).is_file():
                (repo / name).chmod(0o644)
    warning_list = [record.getMessage() for record in caplog.records
                    if record.levelname == "WARNING"]
    head = f"some files of {repo} could not be read, so what the index holds for them " \
           f"is used: "
    assert warning_list == [head + "b.py", head + "b.py, c.py"]
    assert caplog.text.count("what git wrote: error: ") == 4


def test_a_git_directory_that_cannot_be_written_is_read_without_writing(tmp_path):
    """Every command reads the tree's state; one that cannot write the tree reads what
    git reports instead, and a run, which has to commit it, says it cannot."""
    repo = _one_commit_repo(tmp_path / "repo")
    head = gitcmd.head_commit(str(repo))
    git_path = repo / ".git"
    mode_dict = {path: path.stat().st_mode for path in [git_path, *git_path.rglob("*")]}
    for path in mode_dict:
        path.chmod(path.stat().st_mode & ~0o222)
    try:
        assert gitcmd.disk_blob_dict(str(repo), ["a.py"]) == \
            gitcmd.tree_blob_dict(str(repo), head)
        (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
        assert gitcmd.disk_blob_dict(str(repo), ["a.py"]) == {
            "a.py": _git(repo, "hash-object", "a.py").strip()}
        with pytest.raises(RuntimeError, match="could not be written"):
            gitcmd.snapshot_commit(str(repo))
    finally:
        for path, mode in mode_dict.items():
            path.chmod(mode)


def test_the_working_trees_of_a_repository_are_listed_with_the_ones_whose_directory_is_gone(
        tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    there, gone = tmp_path / "wt there", tmp_path / "wt-gone"
    for n, path in enumerate((there, gone)):
        _git(repo, "worktree", "add", "--quiet", "-b", f"b{n}", str(path), "HEAD")
    shutil.rmtree(gone)
    assert gitcmd.worktree_path_list(str(repo)) == [str(repo), str(there), str(gone)]
    assert gitcmd.worktree_path_list(str(tmp_path / "nowhere")) == []


# == a path git stops at when it adds the files on disk =======================
def test_a_line_ending_safecrlf_refuses_does_not_keep_the_changes_out_of_the_tree(tmp_path):
    """core.safecrlf=true stops "git add" at the first such file and adds none. The
    tree is written as core.autocrlf converts the file."""
    repo = _one_commit_repo(tmp_path / "repo")
    head = gitcmd.head_commit(str(repo))
    _git(repo, "config", "core.safecrlf", "true")
    _git(repo, "config", "core.autocrlf", "input")
    (repo / "a.py").write_bytes(b"x = 2\r\ny = 3\r\n")
    (repo / "new.py").write_text("z = 1\n", encoding="utf-8")
    commit = gitcmd.snapshot_commit(str(repo))
    assert commit != head
    assert _git(repo, "show", f"{commit}:a.py") == "x = 2\ny = 3\n"
    assert _git(repo, "show", f"{commit}:new.py") == "z = 1\n"
    # The file on disk and the repository's index are as they were
    assert (repo / "a.py").read_bytes() == b"x = 2\r\ny = 3\r\n"
    assert _git(repo, "diff", "--cached", "--name-only") == ""


def _repo_with_a_path_git_stops_at(tmp_path, other_count):
    """Make a repository whose working tree holds a file git cannot add (UTF-16 by its
    attribute, with no byte order mark) beside other_count new files."""
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / ".gitattributes").write_text("*.u16 working-tree-encoding=UTF-16\n",
                                         encoding="utf-8")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-qm", "attributes")
    (repo / "bad.u16").write_text("no byte order mark\n", encoding="utf-8")
    for n in range(other_count):
        (repo / f"new{n}.py").write_text(f"v = {n}\n", encoding="utf-8")
    return repo


def test_a_path_git_stops_at_keeps_only_itself_out_of_the_tree(tmp_path, caplog):
    repo = _repo_with_a_path_git_stops_at(tmp_path, other_count=5)
    (repo / "a.py").unlink()
    commit = gitcmd.snapshot_commit(str(repo))
    name_list = _git(repo, "ls-tree", "-r", "--name-only", commit).split()
    assert name_list == [".gitattributes", *[f"new{n}.py" for n in range(5)]]
    assert "git stopped at 1 path(s)" in caplog.text and "bad.u16" in caplog.text


def test_a_tree_git_adds_none_of_the_changes_to_is_not_written(tmp_path, caplog):
    """The commit of HEAD is not handed back in place of the tree on disk."""
    repo = _repo_with_a_path_git_stops_at(tmp_path, other_count=0)
    assert gitcmd.work_tree_id(str(repo)) == ""
    assert "git added none of the files" in caplog.text and "bad.u16" in caplog.text
    with pytest.raises(RuntimeError, match="could not be written"):
        gitcmd.snapshot_commit(str(repo))
    assert gitcmd.disk_blob_dict(str(repo), ["bad.u16"]) == {
        "bad.u16": _git(repo, "hash-object", "--no-filters", "bad.u16").strip()}


# == snapshot_commit: the commit a run starts from ===========================
def test_the_snapshot_of_a_clean_tree_is_head(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    assert gitcmd.snapshot_commit(str(repo)) == gitcmd.head_commit(str(repo))


def test_the_snapshot_of_a_changed_tree_is_a_commit_of_that_tree(tmp_path):
    """Changed, added and untracked files are in it; HEAD, the index and the working
    tree are left as they were."""
    repo = _one_commit_repo(tmp_path / "repo")
    head = gitcmd.head_commit(str(repo))
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "new.py").write_text("y = 1\n", encoding="utf-8")
    commit = gitcmd.snapshot_commit(str(repo))
    assert commit != head
    assert gitcmd.tree_of(str(repo), commit) == \
        gitcmd.work_tree_id(str(repo), keep_object=False)
    assert _git(repo, "rev-parse", f"{commit}^").strip() == head
    assert _git(repo, "show", f"{commit}:a.py") == "x = 2\n"
    assert _git(repo, "show", f"{commit}:new.py") == "y = 1\n"
    assert gitcmd.head_commit(str(repo)) == head
    assert sorted(_git(repo, "status", "--porcelain").splitlines()) == [" M a.py", "?? new.py"]
    assert _git(repo, "log", "-1", "--format=%s", commit).startswith(gitcmd.SNAPSHOT_SUBJECT)


def test_a_snapshot_after_an_earlier_one_is_what_the_earlier_one_became(tmp_path):
    """The earlier commit is a parent, so a merge of the two takes the later tree whole,
    and the commit is written for a tree HEAD holds as well."""
    repo = _one_commit_repo(tmp_path / "repo")
    head = gitcmd.head_commit(str(repo))
    assert gitcmd.is_snapshot(str(repo), head) is False
    assert gitcmd.is_snapshot(str(repo), "") is False
    (repo / "new.py").write_text("y = 1\n", encoding="utf-8")
    first = gitcmd.snapshot_commit(str(repo))
    assert gitcmd.is_snapshot(str(repo), first) is True
    (repo / "new.py").write_text("y = 2\nz = 3\n", encoding="utf-8")
    later = gitcmd.snapshot_commit(str(repo), first)
    assert _git(repo, "rev-list", "--parents", "-n", "1", later).split() == [
        later, head, first]
    assert gitcmd.tree_of(str(repo), later) == \
        gitcmd.work_tree_id(str(repo), keep_object=False)
    merged = _git(repo, "merge-tree", "--write-tree", first, later).strip()
    assert merged == gitcmd.tree_of(str(repo), later)
    (repo / "new.py").unlink()
    clean = gitcmd.snapshot_commit(str(repo), later)
    assert clean != head and gitcmd.tree_of(str(repo), clean) == gitcmd.tree_of(str(repo), head)
    assert _git(repo, "rev-list", "--parents", "-n", "1", clean).split()[1:] == [head, later]


def test_a_deleted_file_is_not_in_the_snapshot(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / "b.py").write_text("y = 2\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "b")
    (repo / "a.py").unlink()
    commit = gitcmd.snapshot_commit(str(repo))
    assert _git(repo, "ls-tree", "-r", "--name-only", commit).split() == ["b.py"]


def test_the_snapshot_is_written_without_a_configured_identity(tmp_path, monkeypatch):
    repo = _one_commit_repo(tmp_path / "repo")
    _git(repo, "config", "--unset", "user.email")
    _git(repo, "config", "--unset", "user.name")
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                 "GIT_COMMITTER_EMAIL", "EMAIL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    (repo / "new.py").write_text("y = 1\n", encoding="utf-8")
    commit = gitcmd.snapshot_commit(str(repo))
    assert _git(repo, "log", "-1", "--format=%an", commit).strip() == "codetwine-loom"


def test_a_repository_with_no_commit_has_nothing_to_snapshot(tmp_path):
    directory = tmp_path / "fresh"
    directory.mkdir()
    _git(directory, "init", "-q", "-b", "main")
    with pytest.raises(RuntimeError, match="not a git repository with a commit"):
        gitcmd.snapshot_commit(str(directory))


# == the paths git reports keep the characters they were given ================
def test_a_path_that_is_not_ascii_comes_back_as_itself(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / "説明.md").write_text("one\n", encoding="utf-8")
    assert gitcmd.new_file_list(str(repo)) == ["説明.md"]
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    (repo / "説明.md").write_text("two\n", encoding="utf-8")
    assert [e["path"] for e in gitcmd.numstat_list(str(repo))] == ["説明.md"]


# A file name that is not UTF-8, as bytes, and as every reader names it
_LATIN1_NAME = b"caf\xe9.txt"
_LATIN1_SHOWN = "caf\\xe9.txt"


def _write_raw(repo, raw_name, data):
    """Write a file under a name given as bytes."""
    with open(os.path.join(os.fsencode(str(repo)), raw_name), "wb") as handle:
        handle.write(data)


def test_a_name_that_is_not_utf8_is_shown_with_its_bytes_by_every_reader(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    _write_raw(repo, b"mod" + _LATIN1_NAME, b"a\n")
    _write_raw(repo, b"del" + _LATIN1_NAME, b"a\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    _write_raw(repo, b"mod" + _LATIN1_NAME, b"a\nb\n")
    os.remove(os.path.join(os.fsencode(str(repo)), b"del" + _LATIN1_NAME))
    _write_raw(repo, b"new" + _LATIN1_NAME, b"x\n")
    shown_list = ["del" + _LATIN1_SHOWN, "mod" + _LATIN1_SHOWN, "new" + _LATIN1_SHOWN]
    assert sorted(gitcmd.status_path_list(str(repo))) == shown_list
    assert gitcmd.new_file_list(str(repo)) == ["new" + _LATIN1_SHOWN]
    assert sorted(e["path"] for e in gitcmd.numstat_list(str(repo))) == shown_list[:2]
    assert gitcmd.name_status(str(repo)) == {"added": [], "modified": [shown_list[1]],
                                             "deleted": [shown_list[0]]}


def test_a_name_that_is_not_utf8_opens_the_file_when_asked_for_as_on_disk(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    _write_raw(repo, _LATIN1_NAME, b"x\n")
    disk_list = gitcmd.new_file_list(str(repo), disk=True)
    assert disk_list == [os.fsdecode(_LATIN1_NAME)]
    assert os.path.isfile(os.path.join(str(repo), disk_list[0]))
    assert gitcmd.shown_text(disk_list[0]) == _LATIN1_SHOWN


def test_a_mark_on_a_file_whose_name_is_not_utf8_is_taken_as_the_file_on_disk(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    _write_raw(repo, _LATIN1_NAME, b"x = 0\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "one more")
    _write_raw(repo, _LATIN1_NAME, b"x = 1\n")
    subprocess.run(["git", "-C", str(repo), "update-index", "--assume-unchanged",
                    os.fsdecode(_LATIN1_NAME)], check=True)
    commit = gitcmd.snapshot_commit(str(repo))
    shown = subprocess.run(["git", "-C", str(repo), "show",
                            f"{commit}:{os.fsdecode(_LATIN1_NAME)}"],
                           check=True, capture_output=True).stdout
    assert shown == b"x = 1\n"


# A name git quotes in its line output: a space, a double quote, a backslash, a tab
_QUOTE_NAME_TUPLE = ("space name.py", 'a"b.py', "back\\slash.py", "tab\tname.py")


def test_a_path_git_would_quote_comes_back_as_itself_from_the_status(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    for name in _QUOTE_NAME_TUPLE:
        (repo / name).write_text("y = 1\n", encoding="utf-8")
    assert sorted(gitcmd.new_file_list(str(repo))) == sorted(_QUOTE_NAME_TUPLE)
    assert sorted(gitcmd.status_path_list(str(repo))) == sorted(_QUOTE_NAME_TUPLE)
    assert all((repo / path).is_file() for path in gitcmd.new_file_list(str(repo)))


def test_a_path_git_would_quote_comes_back_as_itself_from_the_diff(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    for name in _QUOTE_NAME_TUPLE:
        (repo / name).write_text("y = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    for name in _QUOTE_NAME_TUPLE:
        (repo / name).write_text("y = 2\n", encoding="utf-8")
    assert gitcmd.name_status(str(repo))["modified"] == sorted(_QUOTE_NAME_TUPLE)
    assert sorted(e["path"] for e in gitcmd.numstat_list(str(repo))) == \
        sorted(_QUOTE_NAME_TUPLE)


def test_a_staged_rename_is_its_deleted_and_its_added_path(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    _git(repo, "mv", "a.py", "b.py")
    assert sorted(gitcmd.status_path_list(str(repo))) == ["a.py", "b.py"]
    assert gitcmd.name_status(str(repo)) == {"added": ["b.py"], "modified": [],
                                             "deleted": ["a.py"]}


@pytest.mark.parametrize("name", ["b.py", 'a"b.py', "sp ace.py"])
def test_the_blobs_told_without_git_are_the_ones_git_gives_the_bytes(tmp_path, name):
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / name).write_text("y = 1\n", encoding="utf-8")
    assert gitcmd._raw_blob_dict(str(repo), [name, "a.py"]) == {
        path: _git(repo, "hash-object", "--no-filters", "--", path).strip()
        for path in (name, "a.py")}


def test_the_blobs_told_without_git_do_not_wait_on_a_named_pipe(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / "a.py").unlink()
    os.mkfifo(repo / "a.py")
    assert gitcmd._raw_blob_dict(str(repo), ["a.py"]) is None


@pytest.mark.parametrize("text, path", [
    ("b/src/main.py", "b/src/main.py"),
    ('"b/q\\"t.py"', 'b/q"t.py'),
    ('"b/tab\\tname.py"', "b/tab\tname.py"),
    ('"b/back\\\\slash.py"', "b/back\\slash.py"),
    ("b/space name.py\t", "b/space name.py"),
    ('"b/a \\"b.py"\t', 'b/a "b.py'),
    ('"b/\\346\\227\\245.py"', "b/日.py"),
    ("/dev/null", "/dev/null"),
    (os.fsdecode(b"b/caf\xe9.txt"), "b/caf\\xe9.txt"),
    (os.fsdecode(b'"b/q\\"t\xe9.txt"'), 'b/q"t\\xe9.txt'),
])
def test_a_path_of_a_patch_line_is_read_as_it_is(text, path):
    assert gitcmd.unquote_path(text) == path


# == the files an analysis covers ===========================================
def test_the_file_list_holds_what_git_tracks_or_does_not_ignore(repo):
    (repo / ".gitignore").write_text("local.env\nbuild/\n", encoding="utf-8")
    (repo / "local.env").write_text("KEY=1\n", encoding="utf-8")
    (repo / "build").mkdir()
    (repo / "build" / "out.txt").write_text("made\n", encoding="utf-8")
    (repo / "sub").mkdir()
    (repo / "sub" / "new file.txt").write_text("new\n", encoding="utf-8")
    assert sorted(gitcmd.work_tree_file_list(str(repo))) == \
        [".gitignore", "a.txt", "b.txt", "sub/new file.txt"]


def test_the_file_list_leaves_out_a_symbolic_link_and_what_it_reaches(tmp_path):
    """A link to a file outside, a link to a file inside, and a tracked file under a
    directory replaced by a link to outside."""
    outside = tmp_path / "outside"
    (outside / "d").mkdir(parents=True)
    (outside / "secret.py").write_text("TOKEN = 1\n", encoding="utf-8")
    (outside / "d" / "c.py").write_text("z = 1\n", encoding="utf-8")
    repo = _one_commit_repo(tmp_path / "repo")
    (repo / "d").mkdir()
    (repo / "d" / "c.py").write_text("z = 0\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "d")
    (repo / "out.py").symlink_to(outside / "secret.py")
    (repo / "in.py").symlink_to("a.py")
    _git(repo, "add", "out.py")
    (repo / "d" / "c.py").unlink()
    (repo / "d").rmdir()
    (repo / "d").symlink_to(outside / "d")
    assert gitcmd.work_tree_file_list(str(repo)) == ["a.py"]


def test_the_file_list_leaves_out_a_name_that_is_not_utf8(tmp_path, caplog):
    repo = _one_commit_repo(tmp_path / "repo")
    with open(os.path.join(os.fsencode(repo), b"caf\xe9.py"), "w") as fp:
        fp.write("x = 1\n")
    caplog.set_level("WARNING")
    assert gitcmd.work_tree_file_list(str(repo)) == ["a.py"]
    assert "caf\\xe9.py" in caplog.text
    caplog.clear()
    gitcmd.work_tree_file_list(str(repo))
    assert "not UTF-8" not in caplog.text


def test_the_commits_a_history_starts_from(tmp_path):
    repo = _one_commit_repo(tmp_path / "repo")
    root = gitcmd.head_commit(str(repo))
    _git(repo, "commit", "-q", "--allow-empty", "-m", "two")
    assert gitcmd.root_commit_list(str(repo)) == [root]
    assert gitcmd.has_commit(str(repo), root)
    assert not gitcmd.has_commit(str(repo), "0" * 40)
    assert gitcmd.root_commit_list(str(tmp_path)) == []


def test_a_directory_that_is_not_a_repository_has_no_file_list(tmp_path):
    assert gitcmd.work_tree_file_list(str(tmp_path)) is None


_USER_SETTING_LIST = [
    {},
    {"i18n.commitEncoding": "cp932"},
    {"i18n.logOutputEncoding": "cp932"},
    {"commit.cleanup": "strip"},
    {"commit.cleanup": "strip", "core.commentChar": ";"},
    {"commit.cleanup": "scissors"},
    {"trailer.separators": "#"},
    {"trailer.separators": "="},
    {"log.showSignature": "true"},
]


@pytest.mark.parametrize("how", ["file", "count", "parameter"])
@pytest.mark.parametrize("setting_dict", _USER_SETTING_LIST)
def test_a_commit_message_is_stored_and_read_back_as_it_was_written_whatever_git_is_set_to(
        tmp_path, monkeypatch, how, setting_dict):
    """The user's settings reach git from a file, from GIT_CONFIG_COUNT and from
    GIT_CONFIG_PARAMETERS; the message of a commit this tool writes is the same."""
    message = "# 見出し\n日本語の依頼\n; 2 行目\n\nCodetwine-Loom-Did: 直した\n"
    user_config = tmp_path / "user.gitconfig"
    user_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(user_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    if how == "file":
        for key, value in setting_dict.items():
            subprocess.run(["git", "config", "--file", str(user_config), key, value],
                           check=True)
    elif how == "count":
        monkeypatch.setenv("GIT_CONFIG_COUNT", str(len(setting_dict)))
        for number, (key, value) in enumerate(setting_dict.items()):
            monkeypatch.setenv(f"GIT_CONFIG_KEY_{number}", key)
            monkeypatch.setenv(f"GIT_CONFIG_VALUE_{number}", value)
    else:
        monkeypatch.setenv("GIT_CONFIG_PARAMETERS", " ".join(
            f"'{key.lower()}={value}'" for key, value in setting_dict.items()))
    root = str(tmp_path / "repo")
    os.mkdir(root)
    assert gitcmd.run(root, "init", "-q", "-b", "main").returncode == 0
    proc = gitcmd.run(root, *gitcmd.COMMITTER_TUPLE, "commit", "--quiet", "--allow-empty",
                      "-F", "-", input_text=message)
    assert proc.returncode == 0, proc.stderr
    assert gitcmd.run(root, "log", "-1", "--format=%B").stdout.strip() == message.strip()
    assert gitcmd.run(root, "log", "-1",
                      "--format=%(trailers:key=Codetwine-Loom-Did,valueonly=true)"
                      ).stdout.strip() == "直した"
    head_text = gitcmd.run(root, "cat-file", "commit", "HEAD", text=False).stdout
    assert b"encoding" not in head_text.split(b"\n\n")[0]
    assert message.strip().encode("utf-8") in head_text


@pytest.mark.skipif(shutil.which("ssh-keygen") is None,
                    reason="a commit is signed with ssh-keygen")
@pytest.mark.parametrize("has_signer_file", [True, False])
def test_a_signed_commit_is_read_back_without_a_line_about_its_signature(
        tmp_path, monkeypatch, has_signer_file):
    key = tmp_path / "key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
                   check=True, capture_output=True)
    user_config = tmp_path / "user.gitconfig"
    setting_dict = {"log.showSignature": "true", "gpg.format": "ssh"}
    if has_signer_file:
        (tmp_path / "signers").write_text(f"u@x {(tmp_path / 'key.pub').read_text()}")
        setting_dict["gpg.ssh.allowedSignersFile"] = str(tmp_path / "signers")
    user_config.write_text("")
    for name, value in setting_dict.items():
        subprocess.run(["git", "config", "--file", str(user_config), name, value],
                       check=True)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(user_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    root = str(tmp_path / "repo")
    os.mkdir(root)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "-c", "user.name=u", "-c", "user.email=u@x", "-c", "commit.gpgSign=true",
         "-c", f"user.signingKey={key}.pub", "commit", "-q", "--allow-empty", "-m",
         "add a\n\nthe body of a")
    assert b"gpgsig" in gitcmd.run(root, "cat-file", "commit", "HEAD", text=False).stdout
    assert gitcmd.run(root, "log", "-1", "--format=%s").stdout == "add a\n"
    assert gitcmd.run(root, "log", "-1", "--format=%B").stdout.strip() \
        == "add a\n\nthe body of a"
    assert gitcmd.run(
        root, "log", "-1", "--format=%(trailers:key=Codetwine-Loom-Run-Request,valueonly=true)"
    ).stdout.strip() == ""


# == What git shows a user, and where it reads its settings ====

def test_the_paths_git_status_names_are_read_under_the_settings_of_the_repository(
        tmp_path, monkeypatch):
    """A rule of core.excludesFile and status.showUntrackedFiles are the user's own:
    the first is followed, the second is not what decides which files are listed."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    (root / "inner").mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root / "inner", "init", "-q", "-b", "main")
    for rel in ("a.txt", "sub/b.txt", "sub/c.swp", "moved.txt", "caf\u00e9.txt"):
        (root / rel).write_text("x\n", encoding="utf-8")
    (root / "rules").write_text("*.swp\n", encoding="utf-8")
    _git(root, "config", "core.excludesFile", "rules")
    _git(root, "config", "status.showUntrackedFiles", "no")
    _git(root, "add", "a.txt", "moved.txt")
    _git(root, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-qm",
         "one")
    _git(root, "mv", "moved.txt", "to.txt")
    (root / "a.txt").write_text("y\n", encoding="utf-8")
    assert gitcmd.status_path_set(str(root)) == {
        b"a.txt", b"moved.txt", b"to.txt", b"sub/b.txt", b"rules", b"inner/",
        "caf\u00e9.txt".encode("utf-8")}
    assert gitcmd.status_path_set(str(tmp_path / "none")) is None


def test_the_places_git_reads_its_settings_from_are_listed(tmp_path, monkeypatch):
    """A file a setting includes is listed whether or not it is there and its
    condition holds, at the path its including file gives it; a setting set twice
    counts by both of its values."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("GIT_CONFIG_GLOBAL", raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(tmp_path / "system.cfg"))
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / ".gitconfig").write_text(
        "[core]\n\texcludesFile = ~/rules\n\thooksPath = /first/hooks\n"
        "[include]\n\tpath = more.cfg\n", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "include.path", "../team.cfg")
    _git(root, "config", "includeIf.gitdir:/nowhere/.path", "/abs/cond.cfg")
    _git(root, "config", "core.hooksPath", "tools/hooks")
    (root / "team.cfg").write_text("[include]\n\tpath = deep/inner.cfg\n"
                                   "[core]\n\tattributesFile = conf/attr\n",
                                   encoding="utf-8")
    place_list = gitcmd.setting_place_list(str(root))
    home, git_dir = str(tmp_path / "home"), str(root / ".git")
    assert set(place_list) - set(gitcmd._program_dir_list(str(root))) == {
        f"{home}/.gitconfig", str(tmp_path / "xdg" / "git" / "config"),
        str(tmp_path / "system.cfg"), str(tmp_path / "xdg" / "git" / "attributes"),
        str(tmp_path / "xdg" / "git" / "ignore"), f"{home}/more.cfg", f"{home}/rules",
        f"{git_dir}/config", f"{git_dir}/../team.cfg", f"{git_dir}/../deep/inner.cfg",
        "/abs/cond.cfg", str(root / "conf" / "attr"), str(root / "tools" / "hooks"),
        "/first/hooks", f"{git_dir}/info/attributes", f"{git_dir}/info/exclude",
        f"{git_dir}/hooks"}
    assert len(place_list) == len(set(place_list))
    assert gitcmd.setting_place_list(str(tmp_path / "none")) is None


@pytest.mark.parametrize("how", ["count", "parameters"])
def test_a_place_a_setting_of_the_environment_names_is_listed(tmp_path, monkeypatch, how):
    """A setting handed over in the environment counts beside the one a file sets;
    the hooks directory this tool hands git is no place."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(tmp_path / "system.gitconfig"))
    setting_dict = {"core.attributesFile": "conf/attr", "core.excludesFile": "/abs/rules",
                    "core.hooksPath": "env/hooks"}
    if how == "count":
        monkeypatch.setenv("GIT_CONFIG_COUNT", str(len(setting_dict)))
        for number, (key, value) in enumerate(setting_dict.items()):
            monkeypatch.setenv(f"GIT_CONFIG_KEY_{number}", key)
            monkeypatch.setenv(f"GIT_CONFIG_VALUE_{number}", value)
    else:
        monkeypatch.setenv("GIT_CONFIG_PARAMETERS", " ".join(
            f"'{key.lower()}={value}'" for key, value in setting_dict.items()))
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "core.hooksPath", "file/hooks")
    place_set = set(gitcmd.setting_place_list(str(root)))
    assert {str(root / "conf" / "attr"), "/abs/rules", str(root / "env" / "hooks"),
            str(root / "file" / "hooks")} <= place_set
    assert os.devnull not in place_set


def test_the_links_in_a_hooks_directory_are_among_the_places(tmp_path, monkeypatch):
    """An entry of the repository's hooks and of a directory core.hooksPath names is
    a place when it is a symbolic link, to a file or to a directory, there or not; a
    hook that is a file is none."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    hook_dir, other_dir = root / ".git" / "hooks", tmp_path / "other"
    other_dir.mkdir()
    (hook_dir / "pre-commit").symlink_to("../../scripts/pre-commit")
    (hook_dir / "lib").symlink_to("../../hooklib")
    (hook_dir / "commit-msg").write_text("#!/bin/sh\n", encoding="utf-8")
    (other_dir / "post-merge").symlink_to(root / "scripts" / "merge")
    (other_dir / "pre-push").write_text("#!/bin/sh\n", encoding="utf-8")
    _git(root, "config", "core.hooksPath", str(other_dir))
    place_list = gitcmd.setting_place_list(str(root))
    hook_place_set = {place for place in place_list
                      if os.path.dirname(place) in (str(hook_dir), str(other_dir))}
    assert hook_place_set == {str(hook_dir / "pre-commit"), str(hook_dir / "lib"),
                              str(other_dir / "post-merge")}
    assert {str(hook_dir), str(other_dir)} <= set(place_list)


def test_the_places_of_a_submodule_are_taken_from_its_own_working_tree(tmp_path,
                                                                       monkeypatch):
    """The settings and the hooks of every submodule git walks into, one in a
    submodule as well, count, a relative value taken from the top of the
    submodule's working tree; a place in the submodule's own working tree, the
    working tree a submodule's settings name, a submodule with no .git and a
    directory whose .git is no repository add none."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    allow_tuple = ("-c", "protocol.file.allow=always")
    who_tuple = ("-c", "user.name=a", "-c", "user.email=a@b")
    inner_source, source = tmp_path / "inner_source", tmp_path / "source"
    for directory in (inner_source, source):
        directory.mkdir()
        _git(directory, "init", "-q", "-b", "main")
        (directory / "s.txt").write_text("s\n", encoding="utf-8")
        _git(directory, "add", "-A")
        if directory == source:
            _git(directory, *allow_tuple, "submodule", "add", "-q", str(inner_source),
                 "inner")
        _git(directory, *who_tuple, "commit", "-q", "-m", "s")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "core.hooksPath", "../outer")
    for name in ("sub", "cold"):
        _git(root, *allow_tuple, "submodule", "add", "-q", str(source), name)
    sub, inner = root / "sub", root / "sub" / "inner"
    _git(sub, *allow_tuple, "submodule", "update", "--init", "-q")
    _git(root, "submodule", "deinit", "-q", "-f", "cold")
    _git(root, f"--git-dir={root / '.git' / 'modules' / 'cold'}", "config",
         "core.hooksPath", "../tools/cold")
    link = _git(source, "rev-parse", "HEAD").strip()
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{link},fake")
    (root / "fake" / ".git").mkdir(parents=True)
    _git(sub, "config", "core.hooksPath", "../tools/hooks")
    _git(sub, "config", "core.fsmonitor", "../tools/watch")
    _git(inner, "config", "core.hooksPath", ".husky/_")
    _git(inner, "config", "core.editor", "../../tools/inner")
    sub_hook_dir = os.path.join(_git(sub, "rev-parse", "--path-format=absolute",
                                     "--git-common-dir").strip(), "hooks")
    os.symlink(root / "tools" / "post", os.path.join(sub_hook_dir, "post-index-change"))
    assert gitcmd._repo_dir_list(str(root)) == [str(root), str(sub), str(inner)]
    place_set = set(gitcmd.setting_place_list(str(root)))
    assert {str(root / "../outer"), str(sub / "../tools/hooks"), str(sub / "../tools/watch"),
            str(inner / "../../tools/inner"), sub_hook_dir,
            os.path.join(sub_hook_dir, "post-index-change")} <= place_set
    assert not {str(root / ".husky/_"), str(inner / ".husky/_"),
                str(root / "fake" / "../outer")} & place_set
    worktree_list = [_git(directory, "config", "core.worktree").strip()
                     for directory in (sub, inner)]
    assert not [place for place in place_set
                if place.endswith(tuple(worktree_list)) or "tools/cold" in place]
    assert "../tools/watch" in gitcmd.setting_command_dict(str(root))


def test_a_submodule_move_is_one_a_patch_gives_both_sides_of(tmp_path):
    """A submodule the patch adds, takes away, leaves where it is or turns into a
    file is no move, nor is a file that holds what a submodule's line does; a quoted
    path is read as git apply reads it, and one that leaves the repository is left
    out."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    _git(root, "add", "a.txt")
    _git(root, "-c", "user.name=a", "-c", "user.email=a@b", "commit", "-q", "-m", "a")
    one = _git(root, "rev-parse", "HEAD").strip()
    two = _git(root, "-c", "user.name=a", "-c", "user.email=a@b", "commit-tree", "-m",
               "b", f"{one}^{{tree}}").strip()
    name_tuple = ("moved", 'q"d', "kept", "gone", "new", "typed", "text")

    def commit_with(link_dict, file_dict):
        for name in name_tuple:
            _git(root, "update-index", "--force-remove", name)
        for name, link in link_dict.items():
            _git(root, "update-index", "--add", "--cacheinfo", f"160000,{link},{name}")
        for name, text in file_dict.items():
            blob = subprocess.run(["git", "-C", str(root), "hash-object", "-w", "--stdin"],
                                  input=text, capture_output=True, text=True,
                                  check=True).stdout.strip()
            _git(root, "update-index", "--add", "--cacheinfo", f"100644,{blob},{name}")
        tree = _git(root, "write-tree").strip()
        return _git(root, "-c", "user.name=a", "-c", "user.email=a@b", "commit-tree",
                    "-m", "c", tree).strip()

    base = commit_with({"moved": one, 'q"d': one, "kept": one, "gone": one, "typed": one},
                       {"text": f"Subproject commit {one}\n"})
    commit = commit_with({"moved": two, 'q"d': two, "kept": one, "new": one},
                         {"typed": "t\n", "text": f"Subproject commit {two}\n"})
    patch_path = str(tmp_path / "change.diff")
    assert gitcmd.write_patch(str(root), base, commit, patch_path).returncode == 0
    link_set = {name.encode() for name in name_tuple}
    assert gitcmd.submodule_move_list(patch_path, link_set) == [
        (b"moved", one, two), (b'q"d', one, two)]
    assert gitcmd.submodule_move_list(patch_path, {b'q"d'}) == [(b'q"d', one, two)]
    with open(patch_path, "rb") as patch_file:
        patch = patch_file.read()
    (tmp_path / "out.diff").write_bytes(patch.replace(b"b/moved", b"b/../moved"))
    assert gitcmd.submodule_move_list(str(tmp_path / "out.diff"), link_set) == [
        (b'q"d', one, two)]
    assert gitcmd.submodule_move_list(str(tmp_path / "none.diff"), link_set) == []


def test_a_submodule_move_is_read_only_where_git_apply_reads_one(tmp_path):
    """Only the header that follows a "diff --git" line naming the path on both sides,
    and the one hunk after it, give a move; the lines of a submodule's part put
    elsewhere in the patch give none."""
    a_part = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n"
    one, two = "1" * 40, "2" * 40
    move = f"-Subproject commit {one}\n+Subproject commit {two}\n"
    sub_move = [(b"sub", one, two)]

    def link(head="diff --git a/sub b/sub\n", between="", plus="+++ b/sub\n",
             hunk="@@ -1 +1 @@\n" + move):
        return (head + between + "index 1111..2222 160000\n--- a/sub\n" + plus + hunk)

    case_list = [
        (link(), sub_move),
        (a_part + link(), sub_move),
        (link() + a_part, sub_move),
        (a_part + link(head=""), []),
        (a_part + f"index 1111..2222 160000\n+++ b/sub\n+Subproject commit {two}\n", []),
        (a_part + link(between="+x\n"), []),
        (link(plus="+++ b/other\n"), []),
        (link(head="diff --git a/sub b/other\n", plus="+++ b/other\n"), []),
        (link(plus='+++ "b/s\\165b"\n'), sub_move),
        (link(head='diff --git "a/s\\165b" "b/s\\165b"\n'), sub_move),
        (link(head='diff --git a/sub "b/s\\165b"\n'), sub_move),
        (link(head="diff --git a/s b b/s b\n", plus="+++ b/s b\t\n"),
         [(b"s b", one, two)]),
        (link(hunk=f"@@ -1 +1,2 @@\n Subproject commit {one}\n+Subproject commit {two}\n"),
         []),
        (link(hunk="@@ -1 +0,0 @@\n" + move), []),
        (link(hunk=f"+Subproject commit {two}\n"), []),
        (link(hunk="@@ -1 +1 @@\n" + move + "\\ No newline at end of file\n"), sub_move),
    ]
    for number, (text, expect_list) in enumerate(case_list):
        patch_path = tmp_path / f"{number}.diff"
        patch_path.write_text(text, encoding="utf-8")
        assert gitcmd.submodule_move_list(str(patch_path), {b"sub", b"s b", b"other"}
                                          ) == expect_list, text


def test_a_submodule_move_is_only_a_header_and_a_hunk_git_writes_for_one(tmp_path):
    """A header line that names, copies, adds or takes away the file or changes its
    mode, a hunk that takes away or leaves other than one line, a second hunk, and a
    line between the two commits, each make the file no move."""
    one, two, three = "1" * 40, "2" * 40, "3" * 40
    move = f"-Subproject commit {one}\n+Subproject commit {two}\n"

    def link(before="", after="", hunk="@@ -1 +1 @@\n" + move):
        return ("diff --git a/sub b/sub\n" + before + "index 1111..2222 160000\n"
                "--- a/sub\n+++ b/sub\n" + after + hunk)

    for before, after in [
            ("", "copy from sub\ncopy to x\n"), ("", "rename from sub\nrename to x\n"),
            ("new file mode 160000\n", ""), ("deleted file mode 160000\n", ""),
            ("similarity index 100%\nrename from other\nrename to sub\n", ""),
            ("similarity index 100%\ncopy from other\ncopy to sub\n", ""),
            ("dissimilarity index 10%\nrename old other\nrename new sub\n", ""),
            ("old mode 160000\nnew mode 160000\n", ""), ("similarity index 100%\n", "")]:
        patch_path = tmp_path / "head.diff"
        patch_path.write_text(link(before, after), encoding="utf-8")
        assert gitcmd.submodule_move_list(str(patch_path), {b"sub", b"x"}) == [], \
            before + after
    for hunk in ["@@ -1 +1 @@\n" + move + f"@@ -1 +1 @@\n-Subproject commit {two}\n"
                 f"+Subproject commit {three}\n",
                 f"@@ -1 +1 @@\n-Subproject commit {one}\n\\ No newline at end of file\n"
                 f"+Subproject commit {two}\n",
                 f"@@ -1,2 +1 @@\n-Subproject commit {one}\n-x\n+Subproject commit {two}\n",
                 f"@@ -0,0 +1 @@\n+Subproject commit {two}\n"]:
        patch_path = tmp_path / "hunk.diff"
        patch_path.write_text(link(hunk=hunk), encoding="utf-8")
        assert gitcmd.submodule_move_list(str(patch_path), {b"sub"}) == [], hunk


def test_a_submodule_two_files_of_a_patch_move_is_no_move(tmp_path):
    """A path the patch moves twice gives no move, a path outside the set read gives
    none, and the others are read once each."""
    def link(path, one, two):
        return (f"diff --git a/{path} b/{path}\nindex {one}..{two} 160000\n"
                f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n"
                f"-Subproject commit {one}\n+Subproject commit {two}\n")

    one, two, three, four, five = (digit * 40 for digit in "12345")
    patch_path = tmp_path / "twice.diff"
    patch_path.write_text(link("sub", one, two) + link("lib", one, three)
                          + link("sub", two, four) + link("sub", four, five)
                          + link("ghost", one, two), encoding="utf-8")
    assert gitcmd.submodule_move_list(str(patch_path), {b"sub", b"lib"}) == [
        (b"lib", one, three)]


def test_a_submodule_move_names_both_commits_by_full_ids_of_one_length(tmp_path):
    """Ids of 40 or 64 hex digits, the same length on both lines, give a move; a
    shorter id, a name of hex digits and ids of two lengths give none."""
    sha1_one, sha1_two = "1" * 40, "2" * 40
    sha256_one, sha256_two = "1" * 64, "2" * 64
    for one, two, expect_list in [
            (sha1_one, sha1_two, [(b"sub", sha1_one, sha1_two)]),
            (sha256_one, sha256_two, [(b"sub", sha256_one, sha256_two)]),
            (sha1_one[:7], sha1_two, []), (sha1_one, sha1_two[:12], []),
            (sha1_one, "cafe", []), (sha1_one, sha256_two, []),
            (sha256_one, sha1_two, []), (sha1_one + "0", sha1_two + "0", [])]:
        patch_path = tmp_path / "ids.diff"
        patch_path.write_text(
            "diff --git a/sub b/sub\nindex 1111..2222 160000\n--- a/sub\n+++ b/sub\n"
            f"@@ -1 +1 @@\n-Subproject commit {one}\n+Subproject commit {two}\n",
            encoding="utf-8")
        assert gitcmd.submodule_move_list(str(patch_path), {b"sub"}) == expect_list, \
            (one, two)


def test_a_long_line_of_a_patch_is_read_past_in_parts(tmp_path, monkeypatch):
    """A line longer than the longest one read is read past a part at a time and
    counts as an empty line."""
    monkeypatch.setattr(gitcmd, "_MAX_PATCH_LINE_BYTE", 64)
    one, two = "1" * 40, "2" * 40
    part = ("diff --git a/sub b/sub\nindex 1111..2222 160000\n--- a/sub\n+++ b/sub\n"
            f"@@ -1 +1 @@\n-Subproject commit {one}\n+Subproject commit {two}\n")
    long = "x" * 1000
    patch_path = tmp_path / "long.diff"
    patch_path.write_text(long + "\n" + part + long, encoding="utf-8")
    size_list = []

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def readline(self, size):
            line = self.handle.readline(size)
            size_list.append(len(line))
            return line

    with open(patch_path, "rb") as handle:
        line_list = list(gitcmd._patch_line_iter(Reader(handle)))
    assert line_list[0] == b"" and line_list[-1] == b"" and len(line_list) == 9
    assert max(size_list) <= 65
    assert gitcmd.submodule_move_list(str(patch_path), {b"sub"}) == [(b"sub", one, two)]
    patch_path.write_text(part.replace("+++ b/sub\n", "+++ b/sub\n" + long + "\n"),
                          encoding="utf-8")
    assert gitcmd.submodule_move_list(str(patch_path), {b"sub"}) == []


def test_the_programs_the_settings_name_are_among_the_places(tmp_path, monkeypatch):
    """A program is named by the whole of a value git runs as a command and by its
    first word, there or not; a file by any word of any value a file is at; a
    directory by a word that stands where what it holds is run, and by any word
    but a first one with no directory in it of a command git runs by itself. A
    command found by its name alone, a directory anywhere else in an alias or in a
    setting git runs nothing by, and the top of the working tree name none in the
    working tree."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    root = tmp_path / "repo"
    (root / "cat").mkdir(parents=True)
    for name in ("pkg", "build", "sub", "wt", "parts", "other", "opt", "main", "any",
                 "eq", "js", "inc", "lib"):
        (root / "tools" / name).mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    for name in ("smudge.sh", "arg.py", "conv.sh", "helper.sh", "name.sh", "key.pub",
                 "event.sh", "tidy.sed", "args.txt"):
        (root / "tools" / name).write_text("", encoding="utf-8")
    (root / ".gitmessage").write_text("", encoding="utf-8")
    for key, value in {
            "filter.a.clean": "tools/clean.sh", "filter.b.smudge": "sh ./tools/smudge.sh %f",
            "filter.c.process": "cat", "merge.d.driver": "python3 tools/arg.py %A %O %B",
            "diff.e.textconv": "sh -c 'tools/conv.sh \"$1\" | cat' -",
            "core.fsmonitor": str(root / "tools" / "watch.sh"),
            "alias.go": "!tools/go.sh --now tools/none.sh", "alias.st": "status tools",
            "alias.open": "checkout tools/name.sh",
            "credential.helper": "!f() { tools/helper.sh; }; f",
            "core.pager": "~/bin/pager", "user.signingKey": "tools/key.pub",
            "commit.template": ".gitmessage", "hook.mine.command": "tools/event.sh",
            "filter.f.clean": "python3 -u tools/pkg %f",
            "filter.g.clean": "make -s -Ctools/build",
            "alias.in": "!cd tools/sub && make", "alias.at": "--work-tree=tools/wt status",
            "alias.all": "!sh -c 'run-parts ./tools/parts'",
            "alias.find": "!grep -rn x tools/other", "alias.top": "!cd . && make -C ./",
            "alias.wait": "!python3 -W ignore tools/opt", "init.defaultBranch": "tools",
            "diff.h.textconv": "python3 -W ignore tools/any",
            "core.sshCommand": "ssh -F none --dir=tools/eq",
            "filter.i.clean": "npm --prefix tools/js run clean",
            "user.name": "npm --prefix tools/other",
            "filter.j.clean": "sed -n -ftools/tidy.sed",
            "filter.k.clean": "cc -Itools/inc -E -", "diff.l.textconv": "java @tools/args.txt",
            "alias.cc": "!cc -Itools/lib",
            "remote.near.url": "./tools/main"}.items():
        _git(root, "config", key, value)
    place_set = {os.path.normpath(place) for place in gitcmd.setting_place_list(str(root))}
    in_tree_set = {os.path.relpath(place, root) for place in place_set
                   if place.startswith(f"{root}{os.sep}") and "/.git/" not in place
                   and " " not in place}
    # A word of a command git runs by itself that has a directory in it and nothing
    # at it, and a word nothing is at where what a directory holds is run
    no_entry_set = {"-Ctools/build", "--dir=tools/eq", "-ftools/tidy.sed", "-Itools/inc",
                    "@tools/args.txt", "tools/helper.sh;", "ignore", "run-parts"}
    assert in_tree_set == no_entry_set | {
        "tools/clean.sh", "tools/smudge.sh", "tools/arg.py",
        "tools/conv.sh", "tools/watch.sh", "tools/go.sh",
        "tools/helper.sh", "tools/name.sh", "tools/key.pub",
        ".gitmessage", "tools/event.sh", "tools/pkg", "tools/build",
        "tools/sub", "tools/wt", "tools/parts", "tools/main",
        "tools/any", "tools/eq", "tools/js", "cat", "tools/tidy.sed",
        "tools/inc", "tools/args.txt"}
    assert str(tmp_path / "home" / "bin" / "pager") in place_set
    assert str(root) not in place_set


def test_a_place_a_command_names_counts_before_anything_is_there(tmp_path, monkeypatch):
    """A word of a command names a directory nothing is at yet: any word with a
    directory in it for a command git runs by itself, and a word that stands where
    what a directory holds is run for those and for an alias a shell reads. A path
    written behind the top of the working tree is read from the top. A setting git
    runs nothing by, an alias git reads as its own arguments, a word of an alias
    anywhere else, a word with no directory in it anywhere else and a path written
    behind another variable name none."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_EDITOR", "python3 -W ignore tools/env-new")
    root = tmp_path / "repo"
    (root / "tools" / "there").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    for key, value in {
            "filter.a.clean": "python3 tools/new", "filter.b.clean": "grep -v x tools/gen/far",
            "filter.c.clean": "python3 runner", "filter.d.clean": "git-lfs clean -- %f",
            "filter.e.clean": 'python3 "$(git rev-parse --show-toplevel)/tools/top"',
            "filter.f.clean": 'sh "$PWD/tools/pwd.sh" ${PWD}/tools/brace',
            "filter.g.clean": "python3 $( git  rev-parse --show-toplevel )/tools/there",
            "filter.h.clean": "python3 $HOME/tools/var x$PWD/tools/glue",
            "filter.i.clean": "sh -c 'cd tools/in && python3 .'",
            "core.fsmonitor": "$PWD/tools/watch", "diff.j.textconv": "run-parts tools/parts",
            "alias.new": "!python3 tools/alias-new", "alias.far": "!grep -rn x tools/alias-far",
            "alias.own": "-C tools/alias-own status", "alias.pwd": "!cd $PWD/tools/alias-pwd",
            "user.email": "python3 tools/no-run"}.items():
        _git(root, "config", key, value)
    place_set = {os.path.normpath(place) for place in gitcmd.setting_place_list(str(root))}
    in_tree_set = {os.path.relpath(place, root) for place in place_set
                   if place.startswith(f"{root}{os.sep}") and "/.git/" not in place}
    assert {"tools/new", "tools/gen/far", "runner", "tools/top", "tools/pwd.sh",
            "tools/brace", "tools/there", "tools/in", "tools/watch", "tools/parts",
            "tools/alias-new", "tools/alias-pwd", "tools/env-new"} <= in_tree_set
    assert not {"clean", "tools/var", "tools/glue", "tools/alias-far", "tools/alias-own",
                "tools/no-run", "tools"} & in_tree_set
    assert str(root) not in place_set


def test_what_the_environment_names_a_program_by_is_among_the_places(tmp_path, monkeypatch):
    """A variable git runs as a command is read as a setting of that kind is, and a
    variable that names a directory of programs is a place there or not, each taken
    from the top of the working tree when relative."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "repo"
    (root / "tools" / "ed").mkdir(parents=True)
    (root / "tools" / "any").mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "tools" / "ssh.conf").write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "tools/ext.sh")
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -F tools/ssh.conf")
    monkeypatch.setenv("EDITOR", "python3 tools/ed")
    monkeypatch.setenv("PAGER", "less -FRX")
    monkeypatch.setenv("VISUAL", "tools --wait -W ignore tools/any")
    monkeypatch.setenv("GIT_EXEC_PATH", "libexec")
    monkeypatch.setenv("GIT_TEMPLATE_DIR", str(root / "tpl"))
    place_set = set(gitcmd.setting_place_list(str(root)))
    assert {str(root / "tools" / "ext.sh"), str(root / "tools" / "ssh.conf"),
            str(root / "tools" / "ed"), str(root / "tools" / "any"),
            str(root / "libexec"), str(root / "tpl")} <= place_set
    assert str(root / "tools") not in place_set
    assert list(gitcmd.setting_command_dict(str(root))) == [
        "tools/ext.sh", "ssh -F tools/ssh.conf", "python3 tools/ed",
        "tools --wait -W ignore tools/any", "less -FRX"]


def test_the_tests_look_for_programs_only_in_absolute_directories():
    assert all(os.path.isabs(entry)
               for entry in os.environ.get("PATH", os.defpath).split(os.pathsep))


def test_the_directories_a_program_is_looked_for_in_are_among_the_places(tmp_path,
                                                                         monkeypatch):
    """Every entry of PATH is a place, there or not. A relative entry, an empty one
    among them, is taken from the top of the working tree and from the directory
    this process runs in."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    monkeypatch.chdir(root / "sub")
    usual = os.environ["PATH"]
    monkeypatch.setenv("PATH", os.pathsep.join([str(root / "tools"), "bin", usual]))
    place_set = {os.path.normpath(place) for place in gitcmd.setting_place_list(str(root))}
    assert {str(root / "tools"), str(root / "bin"), str(root / "sub" / "bin")} <= place_set
    assert not {str(root), str(root / "sub")} & place_set
    monkeypatch.setenv("PATH", os.pathsep.join([usual, ""]))
    place_set = {os.path.normpath(place) for place in gitcmd.setting_place_list(str(root))}
    assert {str(root), str(root / "sub")} <= place_set
    monkeypatch.delenv("PATH")
    assert gitcmd._program_dir_list(str(root)) == [
        entry for entry in os.defpath.split(os.pathsep) for _ in range(2)]


def test_a_relative_place_of_the_user_s_settings_is_taken_from_the_top_of_the_tree(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "conf/user.cfg")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "conf/system.cfg")
    monkeypatch.setenv("XDG_CONFIG_HOME", "xdg")
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    assert {str(root / "conf" / "user.cfg"), str(root / "conf" / "system.cfg"),
            str(root / "xdg" / "git" / "attributes"),
            str(root / "xdg" / "git" / "ignore")} <= set(
                gitcmd.setting_place_list(str(root)))


def test_the_values_git_runs_as_a_command_are_listed(tmp_path, monkeypatch):
    """The values are those of the settings git runs as a command, of a file and of
    the environment, then those of the variables it runs as one, each once; an alias
    no shell reads, a setting git runs nothing by and a setting this tool hands git
    are left out."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.sshCommand")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "ssh -F $PWD/conf")
    monkeypatch.setenv("GIT_EDITOR", '"$PWD/tools/ed.sh"')
    monkeypatch.setenv("VISUAL", "true")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    for key, value in {
            "filter.a.clean": '"$(git rev-parse --show-toplevel)/tools/clean.sh"',
            "filter.a.smudge": '"$(git rev-parse --show-toplevel)/tools/clean.sh"',
            "alias.go": "!cd tools && ./go.sh", "alias.st": "status tools",
            "core.fsmonitor": "true", "user.name": "tools/name.sh",
            "hook.mine.command": "tools/on-event.sh"}.items():
        _git(root, "config", key, value)
    assert gitcmd.setting_command_dict(str(root)) == {
        "true": False, '"$(git rev-parse --show-toplevel)/tools/clean.sh"': False,
        "!cd tools && ./go.sh": True, "tools/on-event.sh": False,
        "ssh -F $PWD/conf": False, '"$PWD/tools/ed.sh"': False}
    assert gitcmd.setting_command_dict(str(tmp_path / "none")) is None


def test_every_setting_git_may_read_is_read_whatever_state_it_reads_it_in(
        tmp_path, monkeypatch):
    """Each file an include names is read whatever its condition, with the files it
    includes, and so is the working tree's config.worktree with its extension off:
    the commands, the places and the filters they hold count. A file that is not
    there, one that includes itself and one git cannot read leave the rest as it is,
    and the drivers this tool's own git stops are still the ones git reads now."""
    user = tmp_path / "user.gitconfig"
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(user))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    corp = tmp_path / "corp"
    (corp / "sub").mkdir(parents=True)
    (corp / "remote.cfg").write_text('[filter "remote"]\n\tclean = tools/remote.sh\n',
                                     encoding="utf-8")
    (corp / "dir.cfg").write_text("[include]\n\tpath = sub/deep.cfg\n", encoding="utf-8")
    (corp / "sub" / "deep.cfg").write_text("[core]\n\tpager = tools/deep.sh\n",
                                           encoding="utf-8")
    (corp / "self.cfg").write_text(f"[include]\n\tpath = {corp / 'self.cfg'}\n"
                                   "[core]\n\teditor = tools/self.sh\n", encoding="utf-8")
    (corp / "unread.cfg").write_text("[core]\n\taskPass = tools/unread.sh\n",
                                     encoding="utf-8")
    (corp / "unread.cfg").chmod(0)
    user.write_text('[includeIf "hasconfig:remote.*.url:https://corp.invalid/**"]\n'
                    "\tpath = corp/remote.cfg\n"
                    '[includeIf "gitdir:/nowhere/"]\n\tpath = corp/dir.cfg\n'
                    '[includeIf "gitdir/i:/NOWHERE/"]\n\tpath = corp/self.cfg\n'
                    '[includeIf "onbranch:none"]\n\tpath = corp/missing.cfg\n'
                    '[includeIf "onbranch:none"]\n\tpath = corp/unread.cfg\n'
                    '[includeIf "onbranch:none"]\n\tpath = corp\n', encoding="utf-8")
    (tmp_path / "env.cfg").write_text("[diff]\n\texternal = tools/env.sh\n",
                                      encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "includeIf.onbranch:feat.path")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(tmp_path / "env.cfg"))
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "includeIf.onbranch:feat.path", "feat.cfg")
    (root / ".git" / "feat.cfg").write_text(
        "[core]\n\thooksPath = tools/hooks\n[alias]\n\tgo = !tools/go.sh\n",
        encoding="utf-8")
    (root / ".git" / "config.worktree").write_text('[filter "wt"]\n\tclean = tools/wt.sh\n',
                                                   encoding="utf-8")
    value_dict = gitcmd.setting_command_dict(str(root))
    assert {"tools/remote.sh": False, "tools/deep.sh": False, "tools/self.sh": False,
            "!tools/go.sh": True, "tools/wt.sh": False,
            "tools/env.sh": False}.items() <= value_dict.items()
    place_set = set(gitcmd.setting_place_list(str(root)))
    assert {str(root / "tools" / "hooks"), str(root / "tools" / "remote.sh"),
            str(root / "tools" / "wt.sh"), str(corp / "sub" / "deep.cfg"),
            str(corp / "missing.cfg"), str(corp / "unread.cfg"),
            str(root / ".git" / "config.worktree")} <= place_set
    assert gitcmd.filter_name_set(str(root)) == {"remote", "wt"}
    assert gitcmd.own_driver_tuple(str(root)) == ([], {})
    (corp / "unread.cfg").chmod(0o600)


@pytest.mark.parametrize("over", ["include_now", "same_file", "environment"])
def test_a_value_a_later_one_sets_over_still_counts(tmp_path, monkeypatch, over):
    """core.hooksPath, core.excludesFile and core.attributesFile name their places by
    every value, and a filter driver counts by any command that is not empty, though
    a later value sets each over: in an include whose condition holds now, later in
    the same file, or later in the environment."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "user.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    first_dict = {"core.hooksPath": ".githooks", "core.excludesFile": ".myignore",
                  "core.attributesFile": ".myattr", "filter.mark.clean": "python3 /x/c.py",
                  "filter.sm.smudge": "python3 /x/s.py"}
    later_dict = {"core.hooksPath": str(tmp_path / "nohooks"),
                  "core.excludesFile": str(tmp_path / "noignore"),
                  "core.attributesFile": str(tmp_path / "noattr"),
                  "filter.mark.clean": "", "filter.sm.smudge": ""}
    if over == "environment":
        pair_list = list(first_dict.items()) + list(later_dict.items())
        monkeypatch.setenv("GIT_CONFIG_COUNT", str(len(pair_list)))
        for index, (key, value) in enumerate(pair_list):
            monkeypatch.setenv(f"GIT_CONFIG_KEY_{index}", key)
            monkeypatch.setenv(f"GIT_CONFIG_VALUE_{index}", value)
    else:
        for key, value in first_dict.items():
            _git(root, "config", key, value)
        later_text = (f"[core]\n\thooksPath = {later_dict['core.hooksPath']}\n"
                      f"\texcludesFile = {later_dict['core.excludesFile']}\n"
                      f"\tattributesFile = {later_dict['core.attributesFile']}\n"
                      '[filter "mark"]\n\tclean =\n[filter "sm"]\n\tsmudge =\n')
        if over == "include_now":
            (root / ".git" / "main.cfg").write_text(later_text, encoding="utf-8")
            _git(root, "config", "includeIf.onbranch:main.path", "main.cfg")
        else:
            with open(root / ".git" / "config", "a", encoding="utf-8") as config_file:
                config_file.write(later_text)
    shown = _git(root, "config", "--get", "core.hooksPath")
    assert shown.strip() == str(tmp_path / "nohooks")
    place_set = set(gitcmd.setting_place_list(str(root)))
    assert {str(root / ".githooks"), str(root / ".myignore"),
            str(root / ".myattr")} <= place_set
    assert {"mark", "sm"} <= gitcmd.filter_name_set(str(root))


@pytest.mark.parametrize("line", ["GIT_CONFIG_GLOBAL={empty}", "XDG_CONFIG_HOME={other}",
                                  "GIT_CONFIG_NOSYSTEM=1", "EDITOR=tools/env-ed.sh"])
def test_the_settings_the_user_s_shell_reads_count_whatever_the_dot_env_file_says(
        tmp_path, monkeypatch, line):
    """A git variable only the .env file of this tool sets leaves the settings the
    user's own git reads in the shell among the ones that count."""
    home = tmp_path / "home"
    (home / ".config" / "git").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    for name in ("GIT_CONFIG_GLOBAL", "XDG_CONFIG_HOME", "GIT_CONFIG_NOSYSTEM", "EDITOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(tmp_path / "system.gitconfig"))
    (home / ".gitconfig").write_text('[filter "home"]\n\tclean = tools/home.sh\n',
                                     encoding="utf-8")
    (home / ".config" / "git" / "config").write_text(
        '[filter "xdg"]\n\tclean = tools/xdg.sh\n', encoding="utf-8")
    (tmp_path / "system.gitconfig").write_text('[filter "sys"]\n\tclean = tools/sys.sh\n',
                                              encoding="utf-8")
    (tmp_path / "empty.gitconfig").write_text("", encoding="utf-8")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    shell_dict = gitcmd.setting_command_dict(str(root))
    shell_place_set = set(gitcmd.setting_place_list(str(root)))
    shell_filter_set = gitcmd.filter_name_set(str(root))
    env_file = tmp_path / "dot.env"
    env_file.write_text(line.format(empty=tmp_path / "empty.gitconfig",
                                    other=tmp_path / "other") + "\n", encoding="utf-8")
    monkeypatch.setattr(config, "_ENV_FILE_FILL_DICT", {})
    config._load_env_file(str(env_file))
    name = line.partition("=")[0]
    assert os.environ[name] and name not in config.shell_env()
    assert shell_dict.items() <= gitcmd.setting_command_dict(str(root)).items()
    assert shell_place_set <= set(gitcmd.setting_place_list(str(root)))
    assert gitcmd.filter_name_set(str(root)) == shell_filter_set == {"home", "xdg", "sys"}


def test_what_config_wrote_for_itself_makes_no_second_environment(monkeypatch):
    """A name config wrote for itself or a library is no difference; one the .env
    file filled is."""
    monkeypatch.setenv("CTLOOM_OWN", "written")
    monkeypatch.setenv("CTLOOM_FILE", "file")
    monkeypatch.setattr(config, "_OWN_FILL_DICT", {"CTLOOM_OWN": (None, "written")})
    assert "CTLOOM_OWN" not in config.shell_env()
    assert gitcmd._env_dict_list() == [dict(os.environ)]
    monkeypatch.setattr(config, "_ENV_FILE_FILL_DICT", {"CTLOOM_FILE": (None, "file")})
    process_dict, shell_dict = gitcmd._env_dict_list()
    assert process_dict == dict(os.environ) and "CTLOOM_FILE" not in shell_dict
