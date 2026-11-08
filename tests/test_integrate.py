"""Combining the branches of several runs into one."""

import collections
import functools
import json
import logging
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
import typing

import pytest
from codetwine.utils import file_utils

from codetwine_loom import config, gitcmd, integrate, sandbox, tmpfile
from codetwine_loom import rlm as rlm_mod
from codetwine_loom.languages import LANG_DICT, LANG_NAME_DICT, PATH_EXAMPLE_DICT
from codetwine_loom.rlm import RLM
from codetwine_loom.tools import edit


def _git(directory, *args):
    """Run one git command in a test repository and fail the test when it does not."""
    done = subprocess.run(["git", "-C", str(directory), *args],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, f"git {args[0]} failed: {done.stderr}"
    return done.stdout


def _write(directory, name, text):
    """Write one file in a test repository."""
    with open(os.path.join(str(directory), name), "w", encoding="utf-8") as handle:
        handle.write(text)


def _read(directory, name):
    """Read one file of a test repository."""
    with open(os.path.join(str(directory), name), encoding="utf-8") as handle:
        return handle.read()


def _analysis(no_resolve=None, name=None, range_dict=None, ref=None):
    """One analyse() return value, empty where a test does not say otherwise.

    A definition's own name is the last part of its name, and each name of a file is
    one definition.
    """
    name, range_dict = name or {}, range_dict or {}
    own = {file: {entry: entry.rpartition(".")[2]
                  for entry in [*name.get(file, ()), *[e[2] for e in range_dict.get(file, ())]]}
           for file in {*name, *range_dict}}
    define = {}
    for file, name_list in sorted(name.items()):
        for entry in name_list:
            define.setdefault(entry.rpartition(".")[2], []).append(file)
    return {"no_resolve": no_resolve or {}, "name": name, "range": range_dict,
            "ref": ref or [], "own": own, "define": define}


@pytest.fixture
def repo(tmp_path):
    """A repository with a base commit and two branches off it.

    f.txt is changed by both branches on the same line, so a and b conflict there.
    Each branch also changes a file the other leaves alone.

    Yields:
        (path, base commit).
    """
    directory = str(tmp_path / "repo")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    _write(directory, "f.txt", "one\ntwo\nthree\n")
    _write(directory, "g.txt", "kept\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()

    _git(directory, "checkout", "-q", "-b", "a", base)
    _write(directory, "f.txt", "ONE\ntwo\nthree\n")
    _write(directory, "a_only.txt", "a\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "a")

    _git(directory, "checkout", "-q", "-b", "b", base)
    _write(directory, "f.txt", "1\ntwo\nthree\n")
    _write(directory, "b_only.txt", "b\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "b")

    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


# == Resolving the branches ==============================


def test_a_local_branch_is_found(repo):
    directory, _ = repo
    assert integrate.commit_of(directory, "a") == \
        _git(directory, "rev-parse", "a").strip()


def test_a_branch_that_is_not_there_says_so(repo):
    directory, _ = repo
    with pytest.raises(integrate.CommandError, match="no branch named nope") as raised:
        integrate.commit_of(directory, "nope")
    assert raised.value.why.endswith("they are listed by:")
    assert shlex.split(raised.value.command) == ["git", "-C", directory, "branch",
                                                 "--list"]


def test_the_base_is_what_the_branches_have_in_common(repo):
    directory, base = repo
    assert integrate.merge_base(directory, ["a", "b"]) == base


def test_a_branch_the_clone_reaches_under_origin_is_found(repo, tmp_path):
    """A clone carries the branches of what it was cloned from as remote names."""
    directory, _ = repo
    clone = str(tmp_path / "clone")
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", directory, clone],
                   check=True)
    assert integrate.commit_of(clone, "a") == _git(directory, "rev-parse", "a").strip()
    assert integrate.merge_base(clone, ["a", "b"]) == \
        _git(directory, "rev-parse", "main").strip()


def test_branches_with_nothing_in_common_say_so(repo):
    directory, _ = repo
    _git(directory, "checkout", "-q", "--orphan", "elsewhere")
    _write(directory, "z.txt", "z\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "orphan")
    with pytest.raises(RuntimeError, match="share no starting commit"):
        integrate.merge_base(directory, ["a", "elsewhere"])


# == Merging without a working tree ======================


def test_a_clean_merge_reports_no_conflict(repo):
    directory, base = repo
    tree, conflicted, messages, no_path = integrate._merge_tree(directory, base, "a")
    assert conflicted == [] and messages == [] and no_path == [] and len(tree) >= 40


def test_a_conflicting_merge_names_the_file_and_keeps_gits_own_words(repo):
    directory, _ = repo
    _, conflicted, messages, no_path = integrate._merge_tree(directory, "a", "b")
    assert conflicted == ["f.txt"] and no_path == []
    assert any("f.txt" in m for m in messages)
    # The file it merged on its own is not reported as a conflict
    assert not any("b_only.txt" in m for m in messages)


def test_a_merge_that_cannot_run_at_all_is_an_error(repo):
    directory, _ = repo
    with pytest.raises(RuntimeError, match="could not merge"):
        integrate._merge_tree(directory, "a", "0" * 40)


def test_a_conflicted_merge_still_writes_a_tree(repo):
    directory, _ = repo
    tree, conflicted, _, _ = integrate._merge_tree(directory, "a", "b")
    assert conflicted
    assert _git(directory, "cat-file", "-t", tree).strip() == "tree"


def test_the_markers_carry_what_the_line_was_before_either_side(repo):
    """merge.conflictStyle=zdiff3 puts the base's own line in the middle section."""
    directory, _ = repo
    tree, _, _, _ = integrate._merge_tree(directory, "a", "b")
    text = _git(directory, "show", f"{tree}:f.txt")
    assert "|||||||" in text and "one\n" in text


def test_a_merge_commit_is_written_without_a_configured_identity(repo, monkeypatch,
                                                                 tmp_path):
    """commit-tree asks for a committer; the identity comes from the command line."""
    directory, base = repo
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.delenv("GIT_AUTHOR_NAME", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_NAME", raising=False)
    tree, _, _, _ = integrate._merge_tree(directory, base, "a")
    commit = integrate._commit_tree(directory, tree, base, "a", "a")
    assert _git(directory, "cat-file", "-t", commit).strip() == "commit"


# == Accumulating and putting the union in place =========


def test_every_branch_is_merged_and_the_union_is_in_the_working_tree(repo):
    directory, base = repo
    result = integrate.assemble(directory, base, ["a", "b"])
    assert [entry["branch"] for entry in result["branches"]] == ["a", "b"]
    assert result["conflicted"] == ["f.txt"]
    assert open(os.path.join(directory, "a_only.txt")).read() == "a\n"
    assert open(os.path.join(directory, "b_only.txt")).read() == "b\n"


def test_head_is_left_at_the_base(repo):
    directory, base = repo
    integrate.assemble(directory, base, ["a", "b"])
    assert _git(directory, "rev-parse", "HEAD").strip() == base


def test_what_the_container_writes_out_is_the_patch_of_the_whole_union(repo):
    """The two commands the container's entry point runs, over the assembled tree."""
    directory, base = repo
    integrate.assemble(directory, base, ["a", "b"])
    _git(directory, "add", "-A")
    patch = _git(directory, "diff", "--cached", "--binary")
    assert "a_only.txt" in patch and "b_only.txt" in patch and "f.txt" in patch


def test_the_second_merge_starts_from_the_first(repo):
    """The last merge's second parent is the last branch, and the first is the rest."""
    directory, base = repo
    result = integrate.assemble(directory, base, ["a", "b"])
    parents = _git(directory, "rev-list", "--parents", "-n", "1",
                   result["commit"]).split()
    assert parents[2] == result["branches"][1]["commit"]
    assert _git(directory, "merge-base", "--is-ancestor",
                result["branches"][0]["commit"], parents[1]) == ""


def _second_of(directory, name):
    """Return the second one file of a test repository was last modified in."""
    return int(os.stat(os.path.join(str(directory), name)).st_mtime)


def test_a_file_put_in_place_in_the_second_of_the_one_it_replaces_is_dated_a_second_on(
        repo):
    """f.txt is as long on the branch as at the base, and g.txt is the same on both."""
    directory, base = repo
    time.sleep(1.02 - time.time() % 1)
    now = time.time()
    for name in ("f.txt", "g.txt"):
        os.utime(os.path.join(directory, name), (now, now))
    integrate.place(directory, "a")
    assert _read(directory, "f.txt") == "ONE\ntwo\nthree\n"
    assert _second_of(directory, "f.txt") == int(now) + 1
    assert _second_of(directory, "g.txt") == int(now)
    # A file put in place over an older one keeps the time it was written at, where
    # the path has not had that time and size before
    os.utime(os.path.join(directory, "f.txt"), (now - 60, now - 60))
    tmpfile._stamp_set_dict.clear()
    integrate.place(directory, base)
    assert _read(directory, "f.txt") == "one\ntwo\nthree\n"
    assert int(now) <= _second_of(directory, "f.txt") <= int(time.time())


def test_a_tree_put_in_place_through_a_lent_index_leaves_the_repositorys_own_alone(repo):
    directory, base = repo
    _write(directory, "new.txt", "n\n")
    before = _git(directory, "status", "--porcelain")
    with gitcmd.work_tree_index(directory) as (tree, env_dict):
        integrate.place(directory, "a", env_dict)
        assert _read(directory, "a_only.txt") == "a\n"
        assert not os.path.exists(os.path.join(directory, "new.txt"))
        integrate.place(directory, tree, env_dict)
    assert _read(directory, "new.txt") == "n\n"
    assert not os.path.exists(os.path.join(directory, "a_only.txt"))
    assert _git(directory, "status", "--porcelain") == before == "?? new.txt\n"


def test_branches_that_do_not_overlap_are_merged_with_no_conflict(repo):
    directory, base = repo
    _git(directory, "checkout", "-q", "-b", "c", base)
    _write(directory, "c_only.txt", "c\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "c")
    _git(directory, "checkout", "-q", "--detach", base)
    result = integrate.assemble(directory, base, ["a", "c"])
    assert result["conflicted"] == [] and result["messages"] == []


# == The conflict markers that are left ==================

# A conflict block as a merge made here writes one, between a side named b and what was
# merged before it
_BLOCK_TEXT = ("<<<<<<< refs/codetwine-loom/merged-so-far\nx\n||||||| 1234567\nw\n=======\ny\n"
               ">>>>>>> refs/codetwine-loom/side/b\n")


def test_a_marker_left_in_the_working_tree_is_found(repo):
    directory, base = repo
    integrate.assemble(directory, base, ["a", "b"])
    assert integrate.marker_path_list(directory) == ["f.txt"]


def test_no_marker_is_found_once_the_file_is_settled(repo):
    directory, base = repo
    integrate.assemble(directory, base, ["a", "b"])
    _write(directory, "f.txt", "ONE\ntwo\nthree\n")
    assert integrate.marker_path_list(directory) == []


@pytest.mark.parametrize("name", ["m.java", "m.kt", "m.js", "m.ts", "m.c", "m.cpp",
                                  "m.h", "m.py"])
def test_a_marker_is_found_whatever_the_file_is_called(repo, name):
    """git decides what a marker is, so no language has a rule of its own here."""
    directory, _ = repo
    _write(directory, name, _BLOCK_TEXT)
    _git(directory, "add", "-A")
    assert integrate.marker_path_list(directory) == [name]


@pytest.mark.parametrize("name", ["a:b.txt", "a:1: b.txt", 'q"t.txt', "space name.txt"])
def test_a_marker_is_found_under_the_name_the_file_has(repo, name):
    directory, _ = repo
    _write(directory, name, _BLOCK_TEXT)
    assert integrate.marker_path_list(directory) == [name]
    _git(directory, "add", "-A")
    assert integrate.marker_path_list(directory) == [name]


def _write_raw(directory, raw_name, data):
    """Write one file of a test repository under a name given as bytes."""
    with open(os.path.join(os.fsencode(str(directory)), raw_name), "wb") as handle:
        handle.write(data)


def test_a_marker_in_a_file_whose_name_is_not_utf8_is_found_under_its_shown_name(repo):
    directory, _ = repo
    _write_raw(directory, b"new\xe9.txt", _BLOCK_TEXT.encode())
    assert integrate.marker_path_list(directory) == ["new\\xe9.txt"]
    _git(directory, "add", "-A")
    assert integrate.marker_path_list(directory) == ["new\\xe9.txt"]


def test_a_conflict_in_a_file_whose_name_is_not_utf8_is_named_with_its_bytes(tmp_path):
    directory = str(tmp_path / "repo")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    _write_raw(directory, b"f\xe9.txt", b"one\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    for branch, text in (("a", b"ONE\n"), ("b", b"uno\n")):
        _git(directory, "checkout", "-q", "-b", branch, base)
        _write_raw(directory, b"f\xe9.txt", text)
        _git(directory, "commit", "-qam", branch)
    _git(directory, "checkout", "-q", "--detach", base)
    result = integrate.assemble(directory, base, ["a", "b"])
    assert result["conflicted"] == ["f\\xe9.txt"]
    assert all("\ufffd" not in message for message in result["messages"])
    assert integrate.change_dict_of(directory, base, "a")["line"] == {"f\\xe9.txt": [(1, 1)]}


def test_a_block_in_a_path_git_would_quote_is_told_under_the_name_the_file_has(repo):
    directory, _ = repo
    name = 'q"t.txt'
    _write(directory, name, _BLOCK_TEXT)
    _write(directory, "plain.txt", "no marker\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "marker")
    head = _git(directory, "rev-parse", "HEAD").strip()
    line_list = integrate._marker_line_dict(directory, [head])[head]
    assert integrate._separator_frame_dict(line_list, ["b"]) == {name: {}}


# == Reading the merge output ============================


def test_a_record_the_output_format_does_not_explain_ends_the_reading(caplog):
    caplog.set_level(logging.INFO, logger="codetwine_loom.integrate")
    records = integrate._parse_record_list(["not a count", "1", "f", "CONFLICT", "m"])
    assert records == []
    assert "leaving them out" in caplog.text


def test_a_record_that_runs_off_the_end_ends_the_reading(caplog):
    caplog.set_level(logging.INFO, logger="codetwine_loom.integrate")
    assert integrate._parse_record_list(["9", "f"]) == []
    assert "leaving them out" in caplog.text


def test_the_separator_between_the_sections_is_skipped():
    records = ["", "1", "f.txt", "CONFLICT (contents)", "CONFLICT: in f.txt\n",
               "2", "new/x", "old/x", "Auto-merging", "Auto-merging new/x\n"]
    assert integrate._parse_record_list(records) == [
        (["f.txt"], "CONFLICT (contents)", "CONFLICT: in f.txt"),
        (["new/x", "old/x"], "Auto-merging", "Auto-merging new/x")]


def test_output_with_no_tree_at_all_is_an_error():
    with pytest.raises(RuntimeError, match="wrote no tree"):
        integrate._parse_merge_tree("")


def test_output_that_is_not_a_tree_is_an_error():
    """A refusal to merge comes back with a conflict's status and no tree."""
    with pytest.raises(RuntimeError, match="wrote no tree"):
        integrate._parse_merge_tree("not something we can merge\0")


# == The references the merge broke ======================


def test_only_the_references_that_stopped_resolving_are_returned():
    before = {("a", "b", "gone"): [1]}
    after = {("a", "b", "gone"): [1], ("a", "b", "broken"): [4]}
    assert integrate.broken_ref_dict(before, after) == {("a", "b", "broken"): [4]}


def test_a_reference_that_resolves_again_is_not_returned():
    assert integrate.broken_ref_dict({("a", "b", "x"): [1]}, {}) == {}


# == git is started from the one place ===================


def test_every_git_command_goes_through_the_one_runner(repo, monkeypatch):
    seen = []
    real = gitcmd.run

    def watched(project_dir, *args, **kwargs):
        seen.extend(args)
        return real(project_dir, *args, **kwargs)

    monkeypatch.setattr(integrate.gitcmd, "run", watched)
    directory, base = repo
    integrate.assemble(directory, base, ["a", "b"])
    assert "merge-tree" in seen and "commit-tree" in seen and "read-tree" in seen


# == The work list ========================================


def _merged(conflicted=(), branch_list=("a", "b")):
    return {"base": "0" * 40, "commit": "1" * 40, "messages": [],
            "conflicted": list(conflicted),
            "branches": [{"branch": name, "commit": "2" * 40,
                          "conflicted": list(conflicted), "messages": [],
                          "said": [f"what {name} was asked for"],
                          "did": [f"what {name} said it did"]}
                         for name in branch_list]}


def test_the_work_list_says_when_it_had_to_leave_something_out(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    work = integrate.build_work_dict(_merged(), ["a.py", "b.py", "c.py"], [], frozenset(), "", 0)
    assert work["markers"] == ["a.py", "b.py"]
    assert work["complete"] is False


def test_a_work_list_that_fits_is_complete(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    work = integrate.build_work_dict(_merged(), ["a.py"], [], frozenset(), "", 0)
    assert work["complete"] is True


def test_the_names_offered_for_a_broken_call_are_cut_to_the_same_count(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    broken = integrate._broken_entry_list(
        {("p/a_py/a.py", "p/b_py/b.py", "gone"): [3]},
        {"p/b_py/b.py": ["one", "two", "three"]}, "p")
    assert broken[0]["now_defines"] == ["one", "two"]
    assert broken[0]["file"] == "a.py" and broken[0]["defined_in"] == "b.py"


def test_the_request_names_the_branches_and_what_git_said():
    merged = _merged()
    merged["messages"] = ["CONFLICT (content): Merge conflict in a.py"]
    request = integrate.request_of(merged)
    assert "a, b" in request and "Merge conflict in a.py" in request


def test_the_request_leaves_out_what_changes_from_turn_to_turn():
    # The state block carries these and is rebuilt every turn; the request is not, so
    # stating them here would leave a count the repository has moved past in the prompt
    request = integrate.request_of(_merged())
    assert "a.py" not in request and "gone" not in request


# == What prepare() will not write over ==================


def _prepare(directory, branch_list=("agent1", "agent2"), **kw):
    """Call prepare() with an analysis that is never read, to reach the git side."""
    return integrate.prepare(directory, "", list(branch_list), **kw)


def test_a_repository_with_uncommitted_work_is_refused(repo, monkeypatch):
    """The union is written straight into the working tree, so what is not committed
    anywhere would be gone with nothing to get it back from."""
    directory, base = repo
    _write(directory, "g.txt", "edited by hand\n")
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    with pytest.raises(RuntimeError) as caught:
        _prepare(directory, ["a", "b"])
    assert "uncommitted" in str(caught.value) and "g.txt" in str(caught.value)


def _run_line(line, cwd):
    """Run a command line a refusal names, as a shell would, from another directory."""
    done = subprocess.run(shlex.split(line), cwd=str(cwd), capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0, f"{line} failed: {done.stderr}"


def test_the_refusal_of_uncommitted_work_says_how_to_stash_a_file_git_does_not_track(
        repo, tmp_path, monkeypatch):
    """A file git does not track is counted as uncommitted; git stash alone leaves it,
    and the stash command the refusal names, run from another directory, takes it out
    of the way."""
    directory, _ = repo
    _write(directory, "notes.txt", "mine\n")
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    with pytest.raises(RuntimeError) as caught:
        _prepare(directory, ["a", "b"])
    text = str(caught.value)
    assert "files git neither tracks nor ignores" in text and text.endswith(": notes.txt")
    _git(directory, "stash")
    assert gitcmd.status_path_list(directory) == ["notes.txt"]
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _run_line(re.search(r"\((git -C \S+ stash --include-untracked)\)", text).group(1),
              elsewhere)
    assert gitcmd.status_path_list(directory) == []


@pytest.mark.parametrize("name, text", [("build.log", "x\n"), ("lib.py", "L = 2\n")])
def test_the_refusal_of_uncommitted_work_in_a_submodule_says_how_to_stash_it_there(
        name, text, repo, tmp_path, monkeypatch):
    """A file git does not track and a changed file inside a submodule are counted as
    uncommitted at the submodule's path; git stash in the repository leaves them, and
    the stash command the refusal names for a submodule, run inside it, takes them out
    of the way."""
    directory, _ = repo
    source = str(tmp_path / "lib_source")
    os.makedirs(source)
    _git(source, "init", "-q", "-b", "main")
    _write(source, "lib.py", "L = 1\n")
    _git(source, "add", "-A")
    _git(source, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-qm",
         "lib")
    _git(directory, "-c", "protocol.file.allow=always", "submodule", "add", "-q", source, "lib")
    _git(directory, "commit", "-qm", "lib")
    _write(os.path.join(directory, "lib"), name, text)
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    with pytest.raises(RuntimeError) as caught:
        _prepare(directory, ["a", "b"])
    refusal = str(caught.value)
    assert refusal.endswith(": lib")
    _git(directory, "stash", "--include-untracked")
    assert gitcmd.status_path_list(directory) == ["lib"]
    command = re.search(r"\((git -C \S*<submodule>\S* stash [^)]*)\)", refusal).group(1)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _run_line(command.replace("<submodule>", "lib"), elsewhere)
    assert gitcmd.status_path_list(directory) == []


def test_the_refused_repository_is_left_exactly_as_it_was(repo, monkeypatch):
    directory, base = repo
    _write(directory, "g.txt", "edited by hand\n")
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    with pytest.raises(RuntimeError):
        _prepare(directory, ["a", "b"])
    assert _git(directory, "rev-parse", "HEAD").strip() == base
    assert _read(directory, "g.txt") == "edited by hand\n"
    assert _read(directory, "f.txt") == "one\ntwo\nthree\n"


def test_an_untracked_file_is_refused_too(repo, monkeypatch):
    """git status reports it, and read-tree would write over a path of the same name."""
    directory, _ = repo
    _write(directory, "new.txt", "not committed anywhere\n")
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    with pytest.raises(RuntimeError) as caught:
        _prepare(directory, ["a", "b"])
    assert "new.txt" in str(caught.value)


def _ignored_branch(directory, base, path, text="from the branch\n"):
    """Put one commit on a branch c that tracks a path the repository ignores."""
    with open(os.path.join(directory, ".git", "info", "exclude"), "a",
              encoding="utf-8") as handle:
        handle.write(f"/{path.split('/')[0]}\n")
    _git(directory, "checkout", "-q", "-b", "c", base)
    os.makedirs(os.path.dirname(os.path.join(directory, path)), exist_ok=True)
    _write(directory, path, text)
    _git(directory, "add", "-f", path)
    _git(directory, "commit", "-qm", "c")
    _git(directory, "checkout", "-q", "--detach", base)


@pytest.mark.parametrize("branch_path, user_path, named", [
    (".env", ".env", ".env"),
    ("logs", "logs/today.log", "logs/"),
    ("cfg/local.txt", "cfg", "cfg")])
@pytest.mark.parametrize("how", ["branch", "onto"])
def test_what_git_does_not_track_where_a_branch_holds_a_file_is_refused_and_kept(
        repo, monkeypatch, branch_path, user_path, named, how):
    """A checkout writes over an ignored file at a path the commit holds, and takes
    away one in the way of it."""
    directory, base = repo
    _ignored_branch(directory, base, branch_path)
    os.makedirs(os.path.dirname(os.path.join(directory, user_path)), exist_ok=True)
    _write(directory, user_path, "SECRET=mine\n")
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    with pytest.raises(RuntimeError) as caught:
        if how == "branch":
            _prepare(directory, ["a", "c"])
        else:
            _prepare(directory, ["a", "b"], onto="c")
    assert str(caught.value).endswith(f"only read: {named}")
    assert "Move them out of the repository first" in str(caught.value)
    assert _read(directory, user_path) == "SECRET=mine\n"
    assert _git(directory, "rev-parse", "HEAD").strip() == base


def test_a_branch_that_tracks_an_ignored_path_nothing_is_at_is_merged(repo, monkeypatch):
    directory, base = repo
    _ignored_branch(directory, base, ".env")
    _write(directory, ".other", "not where the branch writes\n")
    with open(os.path.join(directory, ".git", "info", "exclude"), "a",
              encoding="utf-8") as handle:
        handle.write("/.other\n")
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    assert len(_prepare(directory, ["a", "c"])["merged"]["branches"]) == 2
    assert _read(directory, ".env") == "from the branch\n"
    assert _read(directory, ".other") == "not where the branch writes\n"


@pytest.mark.parametrize("mark, change", [
    ("--assume-unchanged", "write"), ("--skip-worktree", "write"),
    ("--assume-unchanged", "remove"), ("--assume-unchanged", "link")])
def test_a_marked_file_that_does_not_hold_what_the_index_holds_is_refused_and_kept(
        repo, monkeypatch, mark, change):
    """git status does not show it, and reset --hard and read-tree write over it."""
    directory, base = repo
    if change == "link":
        os.symlink("f.txt", os.path.join(directory, "to_f"))
        _git(directory, "add", "to_f")
        _git(directory, "commit", "-qm", "link")
        _git(directory, "update-index", mark, "to_f")
        os.remove(os.path.join(directory, "to_f"))
        os.symlink("g.txt", os.path.join(directory, "to_f"))
        name = "to_f"
    else:
        _git(directory, "update-index", mark, "g.txt")
        name = "g.txt"
    if change == "write":
        _write(directory, "g.txt", "kept\nedited by hand\n")
    elif change == "remove":
        os.remove(os.path.join(directory, "g.txt"))
    assert _git(directory, "status", "--porcelain").strip() == ""
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    with pytest.raises(RuntimeError) as caught:
        _prepare(directory, ["a", "b"])
    assert str(caught.value).endswith(f"only read: {name}")
    assert "Take the mark off" in str(caught.value)
    if change == "write":
        assert _read(directory, "g.txt") == "kept\nedited by hand\n"
    command = re.search(r"\((git -C \S+ update-index [^)]*)\)", str(caught.value)).group(1)
    for line in command.replace("<file>", name).split(" && "):
        _run_line(line, directory + "/..")
    assert integrate._mark_change_list(directory) == []


@pytest.mark.parametrize("mark", ["--assume-unchanged", "--skip-worktree"])
def test_a_marked_file_that_holds_what_the_index_holds_or_a_sparse_one_is_merged(
        repo, monkeypatch, mark):
    directory, base = repo
    _git(directory, "update-index", mark, "g.txt")
    if mark == "--skip-worktree":
        os.remove(os.path.join(directory, "g.txt"))
    assert integrate._mark_change_list(directory) == []
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    assert len(_prepare(directory, ["a", "b"])["merged"]["branches"]) == 2


def test_a_clean_repository_is_merged(repo, monkeypatch):
    """The container's clone is checked out by the entry point and is clean."""
    directory, _ = repo
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    assert len(_prepare(directory, ["a", "b"])["merged"]["branches"]) == 2


# == Where the repository was before the union was put in ==


def test_what_the_repository_was_on_is_read_before_head_moves(repo, monkeypatch):
    directory, _ = repo
    _git(directory, "checkout", "-q", "main")
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    assert _prepare(directory, ["a", "b"])["was_on"] == "main"


def test_a_detached_repository_says_the_commit_it_was_on(repo, monkeypatch):
    directory, base = repo
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    assert _prepare(directory, ["a", "b"])["was_on"] == base


# == The commit the union is written out against ==========


def _move_main_on(directory, base):
    """Give main one commit past the base and put HEAD back at the base."""
    _git(directory, "checkout", "-q", "main")
    _write(directory, "moved.txt", "main moved on\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "main moves on")
    moved = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "--detach", base)
    return moved


def test_the_tip_being_merged_into_is_analysed_when_it_is_ahead_of_the_start(repo, monkeypatch):
    """What the merge broke is read against that tip, not against the start."""
    directory, base = repo
    moved = _move_main_on(directory, base)
    start_analysis = _analysis(name={"p/s_py/s.py": ["s"]})
    moved_analysis = _analysis(name={"p/m_py/m.py": ["m"]})
    analysis_dict = {"": start_analysis, "moved.sqlite": moved_analysis}
    seen = {}

    def ensure(project_dir, output_dir, **kwargs):
        seen.update(head=_git(directory, "rev-parse", "HEAD").strip(),
                    output_dir=output_dir, **kwargs)
        return "moved.sqlite", project_dir, "p", False

    monkeypatch.setattr(integrate, "analyse", lambda path: analysis_dict[path])
    monkeypatch.setattr(integrate, "ensure_knowledge", ensure)
    prepared = integrate.prepare(directory, "", ["a", "b"], onto="main", output_dir="out",
                                 exclude_list=["vendor"])
    assert seen == {"head": moved, "output_dir": "out", "rebuild": True,
                    "with_doc": False, "exclude_list": ["vendor"]}
    assert prepared["before"] is start_analysis
    assert prepared["against"] is moved_analysis
    assert prepared["base"] == moved and prepared["merged"]["start"] == base
    assert prepared["side_define"]["main"] is moved_analysis["define"]


def _union_commit(directory, parent_list, trailer="Codetwine-Loom-Verdict: unsettled"):
    """Write a merge commit over the first parent's tree, as the tool writes a union."""
    tree = _git(directory, "rev-parse", f"{parent_list[0]}^{{tree}}").strip()
    argv = ["commit-tree", tree]
    for parent in parent_list:
        argv += ["-p", parent]
    return _git(directory, *argv, "-m", f"integrate: combine\n\n{trailer}").strip()


def test_a_union_is_read_as_the_branches_it_was_merged_from(repo):
    directory, base = repo
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    first = _union_commit(directory, [base, a])
    second = _union_commit(directory, [base, first, b])
    assert integrate.union_side_list(directory, second) == [a, b]
    plain = _union_commit(directory, [base, a], trailer="Signed-off-by: t")
    assert integrate.union_side_list(directory, _union_commit(directory, [base, plain])) \
        == [plain]
    assert integrate.union_side_list(directory, a) == [a]


def test_what_the_union_carried_on_from_broke_is_not_taken_for_a_side_s(repo, monkeypatch):
    """The references are read on the branches the earlier union merged, so what that
    merge broke stays the merge's to settle."""
    directory, base = repo
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    _git(directory, "branch", "union", _union_commit(directory, [base, a, b]))
    union = _git(directory, "rev-parse", "union").strip()
    key = ("p/x_py/x.py", "p/lib_py/lib.py", "foo")
    no_resolve_dict = {union: {key: [5]}, a: {}, b: {}}
    monkeypatch.setattr(integrate, "_analyse_head", lambda project_dir, *rest: {
        "no_resolve": no_resolve_dict[_git(project_dir, "rev-parse", "HEAD").strip()],
        "define": {"u": ["p/u_py/u.py"]}})
    side_dict, define_dict, start_dict = integrate._side_analysis(
        directory, "out", base, ["union"], None, resume="union")
    assert side_dict == {} and start_dict == {}
    assert define_dict == {name: {"u": ["p/u_py/u.py"]}
                           for name in ("union", integrate.side_name(a), integrate.side_name(b))}
    # A union left unsettled is read the same way under any other name it is passed by
    side_dict, define_dict, _ = integrate._side_analysis(directory, "out", base, ["union"],
                                                         None)
    assert side_dict == {} and integrate.side_name(a) in define_dict
    # A merge carrying none of this tool's trailers is one side unless it is resume
    _git(directory, "branch", "plain",
         _union_commit(directory, [base, a, b], trailer="Signed-off-by: t"))
    plain = _git(directory, "rev-parse", "plain").strip()
    no_resolve_dict[plain] = {key: [5]}
    side_dict, define_dict, _ = integrate._side_analysis(directory, "out", base, ["plain"],
                                                         None)
    assert side_dict == {key: [5]} and list(define_dict) == ["plain"]


def test_the_union_carried_on_from_is_analysed_after_what_it_was_merged_from(repo,
                                                                           monkeypatch):
    """The analysis left on disk is the last tip's, which is the union when it is the
    only branch."""
    directory, base = repo
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    _git(directory, "branch", "union", _union_commit(directory, [base, a, b]))
    union = _git(directory, "rev-parse", "union").strip()
    order = []
    monkeypatch.setattr(integrate, "_analyse_head", lambda project_dir, *rest: order.append(
        _git(project_dir, "rev-parse", "HEAD").strip()) or {"no_resolve": {}, "define": {}})
    integrate._side_analysis(directory, "out", base, ["union"], None, resume="union")
    assert order == [a, b, union]


def test_one_branch_s_union_is_the_tree_its_tip_holds(repo):
    """What the analysis records of the tip is what the union in the working tree is."""
    directory, base = repo
    _git(directory, "checkout", "-q", "--detach", "a")
    tip_state = gitcmd.work_tree_id(directory, keep_object=False)
    _git(directory, "checkout", "-q", "--detach", base)
    integrate.assemble(directory, base, ["a"])
    assert gitcmd.work_tree_id(directory, keep_object=False) == tip_state
    _git(directory, "reset", "-q", "--hard")
    integrate.assemble(directory, base, ["a", "b"])
    assert gitcmd.work_tree_id(directory, keep_object=False) != tip_state


def test_the_start_is_what_the_union_is_written_against_when_nothing_is_ahead(repo, monkeypatch):
    directory, base = repo
    start_analysis = _analysis()
    monkeypatch.setattr(integrate, "analyse", lambda _: start_analysis)
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: pytest.fail("no second analysis is made"))
    for kwargs in ({}, {"onto": "main"}):
        prepared = integrate.prepare(directory, "", ["a", "b"], **kwargs)
        assert prepared["against"] is start_analysis
        _git(directory, "checkout", "-q", "--detach", base)
        _git(directory, "reset", "-q", "--hard")


def test_the_run_says_what_it_left_in_place_and_how_to_undo_it(capsys):
    from codetwine_loom import cli
    cli.print_union_in_place("/repo", {"base": "a" * 40, "was_on": "main"})
    printed = capsys.readouterr().out
    assert "the working tree now holds the union, against aaaaaaaaaaaa" in printed
    assert "git -C /repo checkout main && git -C /repo reset --hard" in printed


def test_what_it_left_is_said_without_naming_a_branch_it_cannot_read(capsys):
    from codetwine_loom import cli
    cli.print_union_in_place("/repo", {"base": "b" * 40, "was_on": ""})
    printed = capsys.readouterr().out
    assert "the working tree now holds the union" in printed
    assert "checkout" not in printed


# == The state the stage reads instead of its earlier turns ==


def test_the_state_block_measures_the_markers_again_every_turn(repo, monkeypatch):
    """The prompt carries no earlier turns, so a stale marker list would be all
    the model has."""
    directory, _ = repo
    work = integrate.build_work_dict(_merged(), ["f.txt"], [], frozenset(), "", 0)
    left = ["f.txt"]
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: list(left))
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [])

    assert "f.txt" in turn_note(1, 20)
    left.clear()
    note = turn_note(2, 20)
    assert "f.txt" not in note and "as of this turn: none" in note
    assert work["markers"] == []


def test_the_state_block_measures_what_the_stage_changed_again_every_turn(repo,
                                                                           monkeypatch):
    """What the stage has already written is read from git against the union, not
    from what the model remembers or records."""
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [])

    note = turn_note(1, 20)
    assert "changed since the branches were combined, as of this turn: none" in note
    assert work["changed"] == []

    _write(directory, "f.txt", "one\nTWO\nthree\nfour\n")
    _write(directory, "made.txt", "m\n")
    note = turn_note(2, 20)
    assert "  f.txt  +2 -1" in note and "  made.txt  new file" in note
    assert work["changed"] == [
        {"path": "f.txt", "insertions": 2, "deletions": 1, "new": False, "defs": []},
        {"path": "made.txt", "insertions": 0, "deletions": 0, "new": True, "defs": []}]


def test_what_the_stage_changed_is_read_against_the_union_not_head(repo, monkeypatch):
    """HEAD is the commit every side started from, so a diff against it would list
    every side's work as this stage's."""
    directory, base = repo
    _git(directory, "checkout", "-q", "--detach", "a")
    union = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "--detach", base)
    _git(directory, "read-tree", "-u", "--reset", union)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    turn_note, _watched = integrate.make_state(directory, union, work, lambda: [])
    turn_note(1, 20)
    assert work["changed"] == []
    assert [e["path"] for e in integrate.change_entry_list(directory, "HEAD")] == [
        "a_only.txt", "f.txt"]


def test_what_the_stage_changed_is_cut_to_the_work_list_count(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 1)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [])
    _write(directory, "f.txt", "changed\n")
    _write(directory, "g.txt", "changed\n")
    turn_note(1, 20)
    assert [e["path"] for e in work["changed"]] == ["f.txt"]


def test_the_state_block_is_the_dict_the_repl_was_handed(repo, monkeypatch):
    """Two copies would let what the model reads in code drift from what it is shown."""
    directory, _ = repo
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: ["later.py"])
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [])
    turn_note(1, 20)
    assert work["markers"] == ["later.py"]


def test_the_state_block_says_which_turn_the_whole_check_ran_on(repo, monkeypatch):
    """The references and the tests cost the whole analysis, so they are not measured
    every turn and the state must not imply they were."""
    directory, _ = repo
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    turn_note, watched = integrate.make_state(directory, "HEAD", work,
                                              lambda: ["- something is left"])
    assert "has not checked the whole repository yet" in turn_note(1, 20)
    turn_note(4, 20)
    watched()
    note = turn_note(5, 20)
    assert "checked the whole repository on turn 4" in note
    assert "- something is left" in note
    assert work["checked_at"] == 4 and work["left"] == ["- something is left"]


def test_the_state_block_keeps_the_deadline_countdown(repo, monkeypatch):
    """The state block carries the deadline countdown."""
    directory, _ = repo
    monkeypatch.setattr(config, "INTEGRATE_DEADLINE_TURN", 3)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [])
    assert "turns left" in turn_note(19, 20)


def test_the_state_block_of_a_run_with_no_decision_says_what_recording_keeps(
        repo, monkeypatch):
    """The earlier turns are in the prompt, and the block does not say otherwise."""
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    integrate.reset_stage()
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    note = integrate.make_state(directory, "HEAD", work, lambda: [])[0](2, 20)
    assert note.endswith(
        "You have recorded nothing with record_decision() yet. What you record is kept "
        "in this block and with the merge, where a run continuing it reads it; a reason "
        "you only print is not.")
    assert "not in front of you" not in note
    assert "Your earlier turns stay in the prompt." in integrate.build_integrate_instruction("py")


def test_the_instructions_name_the_start_block_for_what_only_it_holds():
    text = " ".join(integrate.build_integrate_instruction("py").split())
    assert 'holds the version git kept, and "## At the start" says so.' in text
    assert 'Where "## At the start" names a definition more than one side changed' in text
    assert "the state says so" not in text and "the state names a definition" not in text


def test_the_state_block_asked_after_the_last_turn_keeps_that_turn(repo, monkeypatch):
    """The call made after the last turn reads the block without a countdown, and the
    check that follows it is recorded on the last turn."""
    directory, _ = repo
    monkeypatch.setattr(config, "INTEGRATE_DEADLINE_TURN", 3)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    turn_note, check_and_keep = integrate.make_state(directory, "HEAD", work, lambda: [])
    note = turn_note(21, 20)
    assert note.startswith("## State (turn 20/20)") and "turns left" not in note
    check_and_keep()
    assert work["checked_at"] == 20


def test_the_state_block_carries_what_each_branch_was_asked_for_and_did(repo,
                                                                       monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    start = integrate.start_text(work)
    assert start.startswith("## At the start\n\nMerged onto ")
    assert "what a was asked for" in start and "what a said it did" in start
    # What does not change over the run is not sent again with every turn
    note = integrate.make_state(directory, "HEAD", work, lambda: [])[0](1, 20)
    assert note.startswith("## State (turn 1/20)\n\nConflict markers in the working tree")
    assert "what a was asked for" not in note and "Merged onto" not in note


class _ScriptedLLM:
    """A stand-in LLM returning prepared replies in order. It records the prompts."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.replies[len(self.prompts) - 1]


def test_the_instructions_say_the_earlier_turns_stay():
    text = integrate.build_integrate_instruction("py")
    assert "Your earlier turns stay in the prompt" in text
    assert "record_decision(what, why)" in text


def test_the_stage_carries_the_state_along_with_its_earlier_turns(repo, monkeypatch):
    """The whole mechanism, with no model called: what the fifth turn is shown is the
    state the machine rebuilt, on top of the transcript."""
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    integrate.reset_stage()
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [])
    g_path = os.path.join(directory, "g.txt")
    llm = _ScriptedLLM(
        ["```python\nprint('turn one read a.py')\n```",
         "```python\nrecord_decision('f.txt', 'kept both sides')\n```",
         f"```python\nopen({g_path!r}, 'w').write('changed\\n')\n```"]
        + ["```python\nprint('later turn')\n```"] * 2)
    stage = RLM(llm, integrate.integrate_tool_list(), "I", stage="integrate", max_turn=5)
    stage.run("REQ", {"work": work}, integrate.INTEGRATE_VAR_DOC,
              turn_note=turn_note)

    last = llm.prompts[4]
    assert "turn one read a.py" in last and "### Turn 4 output" in last
    assert "f.txt — kept both sides" in last, "the recorded reason was lost"
    assert "changed since the branches were combined, as of this turn:\n  g.txt  +1 -1" \
        in last, "what the stage wrote was not read back from git"
    assert "changed since the branches were combined, as of this turn: none" \
        in llm.prompts[2], "the third turn ran before the write, so it must say none"


# == What the model records about its own decisions =======


def test_a_recorded_decision_reaches_the_next_turns_state(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    integrate.reset_stage()
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [])
    assert "recorded nothing" in turn_note(1, 20)
    integrate.record_decision("f.txt", "kept both sides' arguments")
    note = turn_note(2, 20)
    assert "f.txt — kept both sides' arguments" in note
    assert work["decisions"] == [{"what": "f.txt",
                                 "why": "kept both sides' arguments"}]


def test_no_more_decisions_are_recorded_than_the_work_list_shows(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="codetwine_loom.integrate")
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    integrate.reset_stage()
    assert integrate.record_decision("a", "one")["ok"] is True
    assert integrate.record_decision("b", "two")["ok"] is True
    third = integrate.record_decision("c", "three")
    assert third == {"ok": False, "recorded": 2}
    assert "was not recorded" in caplog.text


def test_a_long_decision_is_cut_to_the_configured_length(monkeypatch):
    monkeypatch.setattr(config, "EDIT_HISTORY_DID_CHAR", 20)
    integrate.reset_stage()
    integrate.record_decision("x" * 200, "y" * 200)
    held = integrate.decision_list()[0]
    assert len(held["what"]) < 60 and len(held["why"]) < 60


def test_what_an_earlier_run_recorded_does_not_reach_the_next_one():
    integrate.reset_stage()
    integrate.record_decision("a", "one")
    integrate.reset_stage()
    assert integrate.decision_list() == []


def test_what_the_earlier_merge_recorded_is_in_the_start_block(repo, monkeypatch):
    """The turns of the run this one carries on from are not in front of it, so what
    that run recorded is shown the way its own decisions are, apart from them."""
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    integrate.reset_stage()
    earlier = {"branch": "sandbox/earlier", "ended": "turns exhausted",
               "why": "- markers in f.txt",
               "decisions": [{"what": "g.txt", "why": "kept b's rename"}]}
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0, earlier_dict=earlier)
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [])
    integrate.record_decision("f.txt", "kept a")
    start, note = integrate.start_text(work), turn_note(2, 20)
    assert 'carries on from, on sandbox/earlier, ended with "turns exhausted", the check ' \
           'finding: - markers in f.txt' in start
    assert "  g.txt — kept b's rename" in start and "g.txt" not in note
    assert "What you have recorded with record_decision():\n  f.txt — kept a" in note
    assert "f.txt — kept a" not in start
    assert work["earlier"] is earlier and work["decisions"] == [{"what": "f.txt", "why": "kept a"}]


def test_an_earlier_merge_that_recorded_nothing_says_so(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    integrate.reset_stage()
    earlier = {"branch": "sandbox/earlier", "ended": "complete", "why": "", "decisions": []}
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0, earlier_dict=earlier)
    start = integrate.start_text(work)
    assert 'ended with "complete". What it recorded' in start and start.endswith("  nothing")
    assert "the check finding" not in start


def test_a_merge_that_carries_on_from_none_shows_no_earlier_merge(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    integrate.reset_stage()
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    note = integrate.make_state(directory, "HEAD", work, lambda: [])[0](1, 20)
    assert work["earlier"] is None and "carries on from" not in note


# == How a merging run ended ==============================


def _ended(left=(), **over) -> dict:
    """Build run_integration()'s return value as far as end_of() reads it, the check
    having found left the last time it was made."""
    result = {"called": True, "outcome": "complete", "refused_completes": 0, "error": "",
              "left": list(left), "work": {"left": list(left)}}
    result.update(over)
    return result


def test_how_the_run_ended_is_told_from_its_result():
    assert integrate.end_of(_ended()) == "complete"
    assert integrate.end_of(_ended(refused_completes=1)) == "complete"
    assert integrate.end_of(_ended(refused_completes=2, left=["- markers in f.txt"])) \
        == "refusals exhausted"
    assert integrate.end_of(_ended(outcome=None)) == "turns exhausted"
    assert integrate.end_of(_ended(outcome=None, error="the call failed")) == "stopped early"
    assert integrate.end_of(_ended(called=False, outcome=None)) == ""


def test_a_run_accepted_with_nothing_left_ended_complete_whatever_the_refusal_count():
    """With no refusal allowed the first COMPLETE() is accepted as it is, and after the
    last refusal the next one is; what the check found afterwards is what says how the
    run ended, not how many times it was sent back."""
    assert integrate.end_of(_ended(refused_completes=0)) == "complete"
    assert integrate.end_of(_ended(refused_completes=3)) == "complete"
    assert integrate.end_of(_ended(refused_completes=0, left=["- markers in f.txt"])) \
        == "refusals exhausted"


@pytest.mark.parametrize("refuse_count", [0, 1])
def test_a_run_the_check_found_nothing_in_did_not_use_up_its_refusals(refuse_count):
    """A stage adds to "left" what its check does not look for: a write it refused."""
    result = _ended(refused_completes=refuse_count,
                    left=["- a change outside this run's files was refused: a.py"],
                    work={"left": []})
    assert integrate.end_of(result) == "complete"


def test_the_stage_is_handed_the_tool_that_records_a_decision():
    """Without it a reason lives only in a turn the prompt no longer carries."""
    assert integrate.record_decision in integrate.integrate_tool_list()


# == The guard on the call that ends the run =============


def test_the_call_that_ends_the_run_is_sent_back_once(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_MAX_REFUSAL", 1)
    reasons = ["- something is left"]
    guard = integrate.make_complete_guard(lambda: reasons)
    first = guard("done")
    assert "COMPLETE() was not accepted" in first and "something is left" in first
    assert guard("done") == ""


def test_a_run_with_nothing_left_is_not_sent_back():
    guard = integrate.make_complete_guard(list)
    assert guard("done") == ""


def test_the_reasons_are_read_again_each_time_until_one_refusal():
    calls = []

    def check():
        calls.append(1)
        return []

    guard = integrate.make_complete_guard(check)
    guard("done")
    guard("done")
    assert len(calls) == 2


# == What the combined change is committed as =============


def _settled_patch(directory, base, tmp_path, branch_list=("a", "b")):
    """Assemble the branches, settle f.txt and write the union out as the container
    would, leaving the repository as it was."""
    integrate.assemble(directory, base, list(branch_list))
    _write(directory, "f.txt", "ONE 1\ntwo\nthree\n")
    _git(directory, "add", "-A")
    patch = tmp_path / "union.diff"
    patch.write_text(_git(directory, "diff", "--cached", "--binary"), encoding="utf-8")
    _git(directory, "reset", "-q", "--hard")
    return str(patch)


def _parents_of(directory, commit):
    return _git(directory, "rev-list", "--parents", "-n", "1", commit).split()[1:]


def test_the_combined_commit_has_every_branch_as_a_parent(repo, tmp_path):
    """So what the branches hold is reached from it: merging one of them again
    changes nothing, and git can tell they are merged."""
    directory, base = repo
    patch = _settled_patch(directory, base, tmp_path)
    taken = sandbox.take_in_patch(directory, patch, base, "codetwine-loom/integration",
                            str(tmp_path / "wt"), "integrate: combine a, b",
                            parent_list=["a", "b"])
    a, b = (_git(directory, "rev-parse", n).strip() for n in ("a", "b"))
    assert _parents_of(directory, taken["commit"]) == [base, a, b]
    assert _git(directory, "show", f"{taken['commit']}:f.txt") == "ONE 1\ntwo\nthree\n"
    assert _git(directory, "show", f"{taken['commit']}:a_only.txt") == "a\n"
    merged = _git(directory, "branch", "--merged", "codetwine-loom/integration").split()
    assert "a" in merged and "b" in merged
    assert _git(directory, "worktree", "list").count("\n") == 1


def test_a_branch_the_base_already_reaches_is_not_a_parent_twice(repo, tmp_path):
    directory, base = repo
    a = _git(directory, "rev-parse", "a").strip()
    _git(directory, "checkout", "-q", "--detach", a)
    patch = _settled_patch(directory, a, tmp_path, branch_list=("b",))
    taken = sandbox.take_in_patch(directory, patch, a, "codetwine-loom/integration",
                            str(tmp_path / "wt"), "integrate: combine a, b",
                            parent_list=["a", "b"])
    assert _parents_of(directory, taken["commit"]) == \
        [a, _git(directory, "rev-parse", "b").strip()]


def test_without_parents_the_commit_is_as_it_always_was(repo, tmp_path):
    directory, base = repo
    patch = _settled_patch(directory, base, tmp_path)
    taken = sandbox.take_in_patch(directory, patch, base, "codetwine-loom/integration",
                            str(tmp_path / "wt"), "integrate: combine a, b")
    assert _parents_of(directory, taken["commit"]) == [base]


def test_an_existing_branch_is_made_again_at_base_when_asked(repo, tmp_path):
    directory, base = repo
    a = _git(directory, "rev-parse", "a").strip()
    _git(directory, "branch", "codetwine-loom/integration", a)
    patch = _settled_patch(directory, base, tmp_path)
    taken = sandbox.take_in_patch(directory, patch, base, "codetwine-loom/integration",
                            str(tmp_path / "wt"), "integrate: combine a, b",
                            parent_list=["a", "b"], reset=True)
    assert _parents_of(directory, taken["commit"])[0] == base
    assert sandbox.branch_tip(directory, "codetwine-loom/integration") == taken["commit"]


# == What the container is asked to do ===================


def test_the_container_is_asked_to_merge_in_the_clone(monkeypatch):
    argv = sandbox.integrate_argv("repo", ["codetwine-loom/a", "codetwine-loom/b"])
    assert argv[:2] == ["integrate", "--here"]
    assert "--project" in argv and "/work/repo" in argv
    assert argv.count("--branch") == 2
    assert argv[argv.index("--branch") + 1] == "codetwine-loom/a"


def test_the_branches_keep_the_order_they_were_passed():
    argv = sandbox.integrate_argv("repo", ["z", "a"])
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--branch"] == ["z", "a"]


def _integrated(repo, tmp_path, monkeypatch, **kwargs):
    """Run sandbox.integrate() with the container replaced, and return the message the
    result would have been committed with."""
    directory, _ = repo
    seen = {}
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: ("/usr/bin/podman", "img", 0))
    monkeypatch.setattr(sandbox, "commit_message",
                        lambda request, trailers, prefix="sandbox":
                        seen.update(request=request) or "m")
    # The message is built by the callable _land_patch() is handed, so calling
    # it is what makes the run say what it would have committed
    monkeypatch.setattr(sandbox, "_land_patch",
                        lambda *a, **k: {"message": a[6]({"exit_status": 0}),
                                         "exit_status": 0})
    sandbox.integrate(directory, str(tmp_path / "out"), ["a", "b"], **kwargs)
    return seen["request"]


def test_the_merge_is_committed_with_what_the_whole_thing_was_asked_for(repo, tmp_path,
                                                                        monkeypatch):
    """A run carrying on from this branch reads the request back off the commit, and
    it is the request somebody made rather than the branches it happened to combine."""
    assert _integrated(repo, tmp_path, monkeypatch,
                       request="--verbose を足して") == "--verbose を足して"


def test_a_merge_asked_for_on_its_own_names_the_branches_instead(repo, tmp_path,
                                                                 monkeypatch):
    """integrate is a subcommand of its own too, and then there is no larger request
    to write."""
    assert _integrated(repo, tmp_path, monkeypatch) == "combine a, b"


def test_one_branch_is_nothing_to_combine_without_a_base(repo, tmp_path):
    directory, _ = repo
    with pytest.raises(RuntimeError, match="nothing to combine"):
        sandbox.integrate(directory, str(tmp_path / "out"), ["a"])


def test_one_branch_is_combined_against_a_base_the_caller_gives(repo, tmp_path,
                                                                 monkeypatch):
    """A dispatching run knows the commit every agent started at, so one branch is
    held to the gates against it rather than refused."""
    directory, base = repo
    seen = {}
    monkeypatch.setattr(sandbox, "_start_container",
                        lambda *a, **k: seen.update(base=a[7]) or ("/usr/bin/podman", "img", 0))
    monkeypatch.setattr(sandbox, "_land_patch", lambda *a, **k: {"exit_status": 0})
    out = sandbox.integrate(directory, str(tmp_path / "out"), ["a"], base=base)
    assert seen["base"] == base and out["base"] == base and out["branches"] == ["a"]


# == The stage's own texts ===============================


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_the_instructions_are_built_for_every_language(ext, monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", ext)
    text = integrate.build_integrate_instruction(ext)
    assert LANG_NAME_DICT[ext] in text and PATH_EXAMPLE_DICT[ext] in text


def test_the_tests_paragraph_is_left_out_where_there_is_no_test_command(monkeypatch):
    monkeypatch.setattr(config, "RUN_TEST", False)
    text = integrate.build_integrate_instruction("py")
    assert "run_test" not in text
    assert edit.run_test not in integrate.integrate_tool_list()


def test_the_tests_paragraph_is_there_where_there_is_one(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "separator")
    text = integrate.build_integrate_instruction("py")
    assert "run_test()" in text
    assert edit.run_test in integrate.integrate_tool_list()


def test_the_summary_is_asked_for_in_the_language_of_the_requests():
    """What COMPLETE() carries is committed and reported as it is."""
    text = integrate.build_integrate_instruction("py")
    assert 'in the language the requests in work["branches"] are written in' in text


def test_the_block_is_replaced_with_every_marker_line_whatever_their_count():
    """A block of the example holds four marker lines, and one inside another eight."""
    text = " ".join(integrate.build_integrate_instruction("py").split())
    assert "Replace the whole block, every marker line included, with the one text" in text
    assert not re.search(r"(four|seven|eight|\d+) marker lines", text)


def test_the_stage_is_handed_no_tool_that_undoes_the_merge():
    """Both are read against HEAD, which here is what every side started from."""
    handed = integrate.integrate_tool_list()
    assert edit.revert_file not in handed and edit.git_diff_stat not in handed


def test_the_stage_is_handed_no_tool_that_leaves_the_repository():
    handed = [tool.__name__ for tool in integrate.integrate_tool_list()]
    assert "web_search" not in handed and "explore" not in handed


def test_a_run_that_calls_no_model_answers_in_the_shape_the_printer_reads():
    """print_integration reads these whether or not a model was called."""
    from codetwine_loom import cli
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    settled = {"work": work, "called": False, "left": [], "settled": True,
               "answer": "combined", "history": [], "iters": 0,
               "refused_completes": 0, "harvested": False, "error": "",
               "outcome": None}
    cli.print_integration(settled)
    cli.print_integration({**settled, "called": True})
    cli.print_integration({**settled, "called": True, "settled": False,
                           "left": ["- markers are still in: a.py"]})


# == What each branch says it was asked to do ============


def _committed(directory, base, branch, message):
    """Put one commit on a branch of its own and return to base."""
    _git(directory, "checkout", "-q", "-b", branch, base)
    _write(directory, f"{branch}.txt", f"{branch}\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", message)
    commit = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "--detach", base)
    return commit


def test_the_request_is_read_out_of_the_commit_a_run_left(repo):
    directory, base = repo
    _committed(directory, base, "asked",
               "sandbox: stop the parser crashing\n\nstop the parser crashing on an "
               "empty file\n\nCodetwine-Loom-Attempt: 1\nCodetwine-Loom-Exit-Status: 0")
    assert [t["asked"] for t in integrate.commit_record_list(directory, base, "asked")] == \
        ["stop the parser crashing on an empty file"]


def test_the_machines_own_trailers_are_not_read_as_a_request():
    assert integrate._request_in_message("sandbox: x\n\nCodetwine-Loom-Verdict: ok\nCodetwine-Loom-Attempt: 2") == ""


def test_a_request_line_that_reads_like_a_trailer_is_still_the_request():
    """Only this tool's own trailers are the machine's record. A line of the request
    that happens to read "Name: value" is a line of what was asked for."""
    assert integrate._request_in_message(
        "sandbox: add the flag\n\nadd --verbose to main.py\nNote: leave the default "
        "off\n\nCodetwine-Loom-Verdict: unverified") == \
        "add --verbose to main.py\nNote: leave the default off"


_CUT_LINE = "# ------------------------ >8 ------------------------"


@pytest.mark.parametrize("message, line_list, asked", [
    # As this tool writes it: subject, request, record
    ("sandbox: do it\n\ndo it\n\nCodetwine-Loom-Run-Request: do it\nCodetwine-Loom-Attempt: 1\n"
     "Codetwine-Loom-Verdict: failed\nCodetwine-Loom-Exit-Status: 1\n",
     ["Codetwine-Loom-Run-Request: do it", "Codetwine-Loom-Attempt: 1",
      "Codetwine-Loom-Verdict: failed", "Codetwine-Loom-Exit-Status: 1"], "do it"),
    # The request holds lines git's own reading of a message stops or starts at
    (f"sandbox: do it\n\ndo it\n\nCodetwine-Loom-Verdict: settled\n{_CUT_LINE}\nmore\n\n"
     "Codetwine-Loom-Verdict: failed\nCodetwine-Loom-Exit-Status: 1\n",
     ["Codetwine-Loom-Verdict: failed", "Codetwine-Loom-Exit-Status: 1"],
     f"do it\n\nCodetwine-Loom-Verdict: settled\n{_CUT_LINE}\nmore"),
    (("sandbox: do it\n\ndo it\n---\nCodetwine-Loom-Verdict: settled\n# note\n\n"
      "Codetwine-Loom-Verdict: failed\n"),
     ["Codetwine-Loom-Verdict: failed"], "do it\n---\nCodetwine-Loom-Verdict: settled\n# note"),
    # Characters other than "\n" end no line and no paragraph
    (("sandbox: x\n\nx\r\rCodetwine-Loom-Verdict: settled\u2028\u2028y\x0c\n\n"
      "Codetwine-Loom-Verdict: failed\n"),
     ["Codetwine-Loom-Verdict: failed"], "x\r\rCodetwine-Loom-Verdict: settled\u2028\u2028y"),
    # A commit that records nothing
    ("integrate: (no request)\n\n(no request)\n\nCodetwine-Loom-Record: none\n",
     ["Codetwine-Loom-Record: none"], "(no request)"),
    ("integrate: (no request)\n\n(no request)\n", [], "(no request)"),
    # A line somebody added to the record's paragraph is passed over
    ("sandbox: x\n\nx\n\nCodetwine-Loom-Verdict: failed\nSigned-off-by: A <a@b.invalid>\n",
     ["Codetwine-Loom-Verdict: failed"], "x"),
    # Commits this tool did not write
    ("Fix the parser\n", [], ""),
    ("Codetwine-Loom-Verdict: settled\n", [], ""),
    ("Fix the parser\n\nCodetwine-Loom-Verdict: settled\n\nand the reader\n", [],
     "Codetwine-Loom-Verdict: settled\n\nand the reader"),
    ("merge feature\n", [], ""),
    ("", [], ""),
])
def test_the_record_of_a_message_is_the_lines_of_its_last_paragraph(message, line_list,
                                                                     asked):
    assert integrate.record_line_list(message) == line_list
    assert integrate._request_in_message(message) == asked


def test_the_value_under_a_name_is_that_of_its_first_line():
    line_list = ["Codetwine-Loom-Verdict: failed", "Codetwine-Loom-Why: no edit: none",
                 "Codetwine-Loom-Verdict: settled"]
    assert integrate.trailer_value(line_list, "Codetwine-Loom-Verdict") == "failed"
    assert integrate.trailer_value(line_list, "Codetwine-Loom-Why") == "no edit: none"
    assert integrate.trailer_value(line_list, "Codetwine-Loom-Verd") == ""
    assert integrate.trailer_value([], "Codetwine-Loom-Verdict") == ""


def test_the_messages_of_a_range_come_back_whole_and_oldest_first(repo):
    directory, base = repo
    first = f"one: a\n\na\n{_CUT_LINE}\nb\n\nCodetwine-Loom-Verdict: failed\n"
    second = "two: 日本語\n\n# c\n\nCodetwine-Loom-Verdict: settled\n"
    _committed(directory, base, "told", first)
    _git(directory, "config", "core.commentChar", "G")
    _git(directory, "checkout", "-q", "told")
    _git(directory, "commit", "-q", "--allow-empty", "--cleanup=verbatim", "-m", second)
    _git(directory, "checkout", "-q", "--detach", base)
    assert integrate.message_list(directory, "--reverse", f"{base}..told") == [first, second]
    assert integrate.message_list(directory, "-1", "told") == [second]
    assert integrate.message_list(directory, f"{base}..{base}") == []
    assert [record_dict["verdict"] for record_dict in
            integrate.commit_record_list(directory, base, "told")] == ["failed", "settled"]
    assert integrate.own_trailer_list(directory, "told") == ["Codetwine-Loom-Verdict: settled"]
    with pytest.raises(RuntimeError, match="no-such-branch"):
        integrate.message_list(directory, "-1", "no-such-branch")


def test_a_commit_that_says_none_of_the_four_is_left_out(repo):
    directory, base = repo
    assert integrate.commit_record_list(directory, base, "a") == []


def test_every_run_on_a_branch_is_read_oldest_first(repo):
    directory, base = repo
    _git(directory, "checkout", "-q", "-b", "twice", base)
    for n, what in ((1, "first thing"), (2, "second thing")):
        _write(directory, f"r{n}.txt", "r\n")
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm",
             f"sandbox: {what}\n\n{what}\n\nCodetwine-Loom-Attempt: {n}")
    assert [t["asked"] for t in integrate.commit_record_list(directory, base, "twice")] == \
        ["first thing", "second thing"]


def test_what_a_run_did_is_read_out_of_its_trailer(repo):
    directory, base = repo
    _committed(directory, base, "told-did",
               "sandbox: rename the helper\n\nrename the helper\n\nCodetwine-Loom-Attempt: 1\n"
               "Codetwine-Loom-Did: renamed helper() to parse_line() and moved it to reader.py")
    assert [t["did"] for t in integrate.commit_record_list(directory, base, "told-did")] == \
        ["renamed helper() to parse_line() and moved it to reader.py"]


def test_how_a_run_ended_is_read_out_of_its_trailers(repo):
    """A run carrying on after this one is told how it ended, which is the machine's
    own record rather than anything the run said."""
    directory, base = repo
    _committed(directory, base, "judged",
               "sandbox: x\n\nadd the flag\n\nCodetwine-Loom-Verdict: failed\n"
               "Codetwine-Loom-Why: the turns ran out before it said the change was done")
    told = integrate.commit_record_list(directory, base, "judged")
    assert [t["verdict"] for t in told] == ["failed"]
    assert [t["why"] for t in told] == \
        ["the turns ran out before it said the change was done"]


def test_every_field_is_read_in_one_pass_without_being_mixed_up(repo):
    directory, base = repo
    _committed(directory, base, "both",
               "sandbox: x\n\nasked for this\n\nCodetwine-Loom-Did: did that\n"
               "Codetwine-Loom-Verdict: unverified\nCodetwine-Loom-Why: nothing checks it\n"
               "Codetwine-Loom-Tests-Changed: tests/test_a.py tests/test_b.py")
    assert integrate.commit_record_list(directory, base, "both") == [
        {"asked": "asked for this", "did": "did that", "verdict": "unverified",
         "why": "nothing checks it",
         "tests_changed": "tests/test_a.py tests/test_b.py", "ended": "",
         "decisions": "", "run_request": "", "stage": "", "non_test_changed": "",
         "tests_base": "", "test_open": "", "gap": "", "gap_done": "",
         "stand_on": ""}]


def test_how_a_merge_ended_and_what_it_recorded_are_read_out_of_its_trailers(repo):
    directory, base = repo
    _committed(directory, base, "merged",
               "integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: unsettled\n"
               "Codetwine-Loom-Ended: turns exhausted\n"
               'Codetwine-Loom-Decisions: [{"what":"f.txt","why":"kept a"}]')
    told = integrate.commit_record_list(directory, base, "merged")
    assert told[0]["ended"] == "turns exhausted"
    assert integrate.decision_list_of(told[0]["decisions"]) == [{"what": "f.txt", "why": "kept a"}]


# == What a merging run recorded, read back for the run carrying on from it ==


def test_decisions_go_through_a_trailer_and_back_unchanged():
    held = [{"what": "f.txt: the block at line 3", "why": "a's request asks for it"},
            {"what": "日本語のパス", "why": "why\nover two lines"}]
    text = integrate.decision_text(held)
    assert "\n" not in text
    assert integrate.decision_list_of(text) == held


def test_no_decision_makes_no_trailer_text():
    assert integrate.decision_text([]) == ""
    assert integrate.decision_list_of("") == []


def test_a_trailer_that_is_not_decisions_is_read_as_none(caplog):
    caplog.set_level(logging.INFO, logger="codetwine_loom.integrate")
    assert integrate.decision_list_of("not json") == []
    assert "cannot be read" in caplog.text
    assert integrate.decision_list_of('{"what": "x"}') == []
    assert integrate.decision_list_of('[{"what": "x"}, {"what": "y", "why": "z"}, 3]') == \
        [{"what": "y", "why": "z"}]


def test_what_the_earlier_merge_recorded_is_read_off_its_branch(repo, monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 1)
    directory, base = repo
    _committed(directory, base, "union",
               "integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: unsettled\n"
               "Codetwine-Loom-Why: - markers in f.txt\nCodetwine-Loom-Ended: refusals exhausted\n"
               'Codetwine-Loom-Decisions: [{"what":"f.txt","why":"kept a"},{"what":"g.txt","why":"kept b"}]')
    assert integrate.earlier_record_dict(directory, "union") == {
        "branch": "union", "ended": "refusals exhausted", "why": "- markers in f.txt",
        "decisions": [{"what": "f.txt", "why": "kept a"}]}


def test_a_branch_with_no_record_gives_no_earlier_merge(repo):
    directory, base = repo
    _committed(directory, base, "bare", "just a commit")
    assert integrate.earlier_record_dict(directory, "bare") is None


def test_preparing_from_an_earlier_merge_reads_its_record(repo, monkeypatch):
    directory, base = repo
    _committed(directory, base, "union",
               "integrate: x\n\ncombine a, b\n\nCodetwine-Loom-Verdict: unsettled\nCodetwine-Loom-Ended: turns exhausted")
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    prepared = _prepare(directory, ["union"], resume="union")
    assert prepared["earlier"]["ended"] == "turns exhausted"
    _git(directory, "reset", "-q", "--hard")
    assert _prepare(directory, ["a", "b"])["earlier"] is None


def test_a_commit_that_changed_no_test_says_so_by_saying_nothing(repo):
    """The trailer is left out by a run that changed no test, so the field is empty
    rather than absent: what reads it need not ask whether it is there."""
    directory, base = repo
    _committed(directory, base, "no-tests",
               "sandbox: x\n\nasked for this\n\nCodetwine-Loom-Did: did that")
    assert integrate.commit_record_list(directory, base, "no-tests")[0]["tests_changed"] == ""


def test_what_a_branch_says_is_not_cut_where_it_is_read(repo, monkeypatch):
    """Two readers have different room for it, so the cutting is each caller's."""
    monkeypatch.setattr(config, "EDIT_HISTORY_REQUEST_CHAR", 20)
    monkeypatch.setattr(config, "EDIT_HISTORY_DID_CHAR", 20)
    directory, base = repo
    _committed(directory, base, "long",
               "sandbox: x\n\n" + "y" * 200 + "\n\nCodetwine-Loom-Did: " + "z" * 200)
    told = integrate.commit_record_list(directory, base, "long")
    assert len(told[0]["asked"]) == 200 and len(told[0]["did"]) == 200


def test_the_work_list_cuts_it_to_what_the_history_cuts_its_own_to(repo, monkeypatch):
    monkeypatch.setattr(config, "EDIT_HISTORY_REQUEST_CHAR", 20)
    monkeypatch.setattr(config, "EDIT_HISTORY_DID_CHAR", 20)
    directory, base = repo
    _committed(directory, base, "long",
               "sandbox: x\n\n" + "y" * 200 + "\n\nCodetwine-Loom-Did: " + "z" * 200)
    merged = integrate.assemble(directory, base, ["long", "a"])
    assert len(merged["branches"][0]["said"][0]) < 60
    assert len(merged["branches"][0]["did"][0]) < 60


def test_a_branch_that_cannot_be_read_gives_nothing_and_says_so(repo, caplog):
    caplog.set_level(logging.INFO, logger="codetwine_loom.integrate")
    directory, base = repo
    assert integrate.commit_record_list(directory, base, "0" * 40) == []
    assert "cannot read what" in caplog.text


# == What a branch an earlier run left holds =============


def test_what_the_last_run_on_a_branch_left_is_read_off_its_tip(repo):
    directory, base = repo
    _committed(directory, base, "left",
               "sandbox: x\n\nadd the flag\n\nCodetwine-Loom-Did: added it\n"
               "Codetwine-Loom-Verdict: failed\nCodetwine-Loom-Why: the turns ran out")
    entry = integrate.branch_record_dict(directory, "left")
    assert entry["branch"] == "left"
    assert entry["commit"] == _git(directory, "rev-parse", "left").strip()
    assert entry["told"] == [{"asked": "add the flag", "did": "added it",
                              "verdict": "failed", "why": "the turns ran out",
                              "tests_changed": "", "ended": "", "decisions": "",
                              "run_request": "", "stage": "",
                              "non_test_changed": "", "tests_base": "",
                              "test_open": "", "gap": "", "gap_done": "",
                              "stand_on": ""}]


def test_the_request_a_run_was_given_is_read_off_the_branch_it_left(repo):
    """The message holds the task, so the request the tasks were split out of is read
    from the trailer the run that committed it wrote."""
    directory, base = repo
    _committed(directory, base, "task-1", sandbox.commit_message(
        "add --version to main.py",
        sandbox.run_trailer_dict({"exit_status": 1}, {"verdict": "failed"},
                                 "main.py に --version と --verbose を足して")))
    assert integrate.request_in_branch_list(directory, ["task-1"]) == \
        "main.py に --version と --verbose を足して"


def test_a_branch_saying_nothing_about_its_run_s_request_gives_none(repo):
    """What the agent on it was asked for is the task, which is not the request."""
    directory, base = repo
    _committed(directory, base, "no-request",
               "sandbox: x\n\nadd the flag\n\nCodetwine-Loom-Verdict: failed")
    assert integrate.request_in_branch_list(directory, ["no-request"]) == ""


def test_a_branch_saying_nothing_does_not_stop_the_one_that_does_being_read(repo):
    """The branches one run left all carry the same request, so one of them carrying
    it is enough."""
    directory, base = repo
    _committed(directory, base, "silent", "sandbox: x\n\nadd the flag\n\nCodetwine-Loom-Did: x")
    _committed(directory, base, "telling",
               "sandbox: x\n\nadd the flag\n\nCodetwine-Loom-Run-Request: 設定を整理して")
    assert integrate.request_in_branch_list(directory, ["silent", "telling"]) == \
        "設定を整理して"


def test_branches_made_for_different_requests_are_read_as_none(repo):
    """Agents run by hand are each given a request of their own, and no one of them is
    what all of their branches were made for."""
    directory, base = repo
    _committed(directory, base, "one",
               "sandbox: x\n\nx\n\nCodetwine-Loom-Run-Request: 設定を整理して")
    _committed(directory, base, "two",
               "sandbox: y\n\ny\n\nCodetwine-Loom-Run-Request: ログを足して")
    assert integrate.request_in_branch_list(directory, ["one", "two"]) == ""


def _union_of(directory, base, branch, parent_list, message):
    """Put one merge commit on a branch of its own, base first among its parents."""
    _git(directory, "checkout", "-q", "-b", branch, base)
    _git(directory, "merge", "-q", "--no-ff", "-m", message, *parent_list)
    _git(directory, "checkout", "-q", "--detach", base)


def test_a_union_says_its_own_request_not_those_of_what_it_combined(repo):
    """The range a union is read over holds the commits it combined, each carrying
    the request its own agent was given."""
    directory, base = repo
    _committed(directory, base, "one",
               "sandbox: x\n\nx\n\nCodetwine-Loom-Run-Request: 設定を整理して")
    _committed(directory, base, "two",
               "sandbox: y\n\ny\n\nCodetwine-Loom-Run-Request: ログを足して")
    _union_of(directory, base, "union", ["one", "two"],
              "integrate: z\n\nz\n\nCodetwine-Loom-Run-Request: 設定とログをまとめて")
    assert sorted(record_dict["run_request"] for record_dict
                  in integrate.branch_record_dict(directory, "union")["told"]) == \
        sorted(["設定を整理して", "ログを足して", "設定とログをまとめて"])
    assert integrate.request_in_branch_list(directory, ["union"]) == \
        "設定とログをまとめて"


def test_a_union_saying_none_is_read_off_what_it_combined(repo):
    directory, base = repo
    _committed(directory, base, "one",
               "sandbox: x\n\nx\n\nCodetwine-Loom-Run-Request: 設定を整理して")
    _committed(directory, base, "two",
               "sandbox: y\n\ny\n\nCodetwine-Loom-Run-Request: 設定を整理して")
    _union_of(directory, base, "union", ["one", "two"], "combine one, two")
    assert integrate.request_in_branch_list(directory, ["union"]) == "設定を整理して"


def test_only_the_last_run_on_that_branch_is_read(repo):
    """Each try gets a branch of its own, so the tip against its first parent is one
    run: what came before it is on the branch that run started from."""
    directory, base = repo
    _git(directory, "checkout", "-q", "-b", "twice", base)
    for what in ("the first try", "the second try"):
        _write(directory, f"{what[4]}.txt", "x\n")
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", f"sandbox: x\n\n{what}")
    _git(directory, "checkout", "-q", "--detach", base)
    assert [t["asked"] for t in integrate.branch_record_dict(directory, "twice")["told"]] == \
        ["the second try"]


def test_a_merge_says_what_the_branches_it_combined_were_asked_for(repo, tmp_path):
    """The tasks' own branch names are taken away once the merge lands, so what they
    hold has to be readable off the merge commit."""
    directory, base = repo
    _committed(directory, base, "task-one", "sandbox: x\n\nthe first task")
    _committed(directory, base, "task-two", "sandbox: x\n\nthe second task")
    merged = integrate.assemble(directory, base, ["task-one", "task-two"])
    sandbox.take_in_patch(directory, _empty_patch(tmp_path), base, "codetwine-loom/combined",
                    str(tmp_path / "worktree"),
                    "integrate: x\n\nthe whole request\n\nCodetwine-Loom-Verdict: unsettled\n"
                    "Codetwine-Loom-Why: conflict markers are still in: f.txt",
                    parent_list=[merged["branches"][0]["commit"],
                             merged["branches"][1]["commit"]])
    told = integrate.branch_record_dict(directory, "codetwine-loom/combined")["told"]
    # The two tasks went at once, so which of them git names first is git's; the merge
    # itself is the tip and comes last
    assert sorted(t["asked"] for t in told[:-1]) == ["the first task",
                                                     "the second task"]
    assert told[-1]["asked"] == "the whole request"
    assert told[-1]["verdict"] == "unsettled"


def _empty_patch(tmp_path):
    """Write a patch that changes one file, for a commit to be made from."""
    path = str(tmp_path / "one.diff")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("diff --git a/g.txt b/g.txt\n--- a/g.txt\n+++ b/g.txt\n"
                     "@@ -1 +1,2 @@\n kept\n+and this\n")
    return path


@pytest.mark.parametrize("in_head, held", [(True, True), (False, False)])
def test_whether_the_repository_already_holds_that_work_is_said(repo, in_head, held):
    """It decides whether the block describes code the stage can read or only an
    account of it."""
    directory, base = repo
    _committed(directory, base, "left", "sandbox: x\n\nadd the flag")
    _git(directory, "checkout", "-q", "--detach", "left" if in_head else base)
    assert integrate.branch_record_dict(directory, "left")["held"] is held


def test_a_branch_that_is_not_there_is_refused(repo):
    directory, _ = repo
    with pytest.raises(RuntimeError, match="no branch named"):
        integrate.branch_record_dict(directory, "never-made")


def test_what_a_branch_did_reaches_the_work_list(repo):
    directory, base = repo
    _git(directory, "checkout", "-q", "-b", "did-told", base)
    _write(directory, "w.txt", "w\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "sandbox: x\n\nx\n\nCodetwine-Loom-Did: what it did")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["did-told", "a"])
    work = integrate.build_work_dict(merged, [], [], frozenset(), "", 0)
    assert work["branches"][0]["did"] == ["what it did"]
    assert work["branches"][1]["did"] == []


def test_what_each_branch_was_asked_for_reaches_the_work_list(repo):
    directory, base = repo
    _git(directory, "checkout", "-q", "-b", "told", base)
    _write(directory, "t.txt", "t\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "sandbox: do the thing\n\ndo the thing")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["a", "told"])
    work = integrate.build_work_dict(merged, [], [], frozenset(), "", 0)
    assert dict(zip([e["branch"] for e in work["branches"]],
                    [e["said"] for e in work["branches"]])) == \
        {"a": [], "told": ["do the thing"]}


# == The branch the result is merged into =================


def test_the_branch_being_merged_into_is_combined_first(repo):
    """It is one more input, so what it changed is settled here like anything else."""
    directory, base = repo
    _git(directory, "checkout", "-q", "main")
    _write(directory, "moved.txt", "main moved on\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "main moves on")
    moved = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "--detach", base)
    result = integrate.assemble(directory, base, ["a", "b"], onto="main")
    assert [entry["branch"] for entry in result["branches"]] == ["main", "a", "b"]
    assert result["base"] == moved
    assert _git(directory, "rev-parse", "HEAD").strip() == moved


def test_what_is_written_out_is_what_the_union_adds_on_top_of_that_branch(repo):
    """The two commands the container's entry point runs, with HEAD left on the
    branch."""
    directory, base = repo
    _git(directory, "checkout", "-q", "main")
    _write(directory, "moved.txt", "main moved on\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "main moves on")
    _git(directory, "checkout", "-q", "--detach", base)
    integrate.assemble(directory, base, ["a"], onto="main")
    _git(directory, "add", "-A")
    patch = _git(directory, "diff", "--cached")
    # what main already had is not in the change; what a adds is
    assert "moved.txt" not in patch and "a_only.txt" in patch


def test_a_branch_the_base_already_reaches_is_not_merged_again(repo, caplog):
    caplog.set_level(logging.INFO, logger="codetwine_loom.integrate")
    directory, base = repo
    result = integrate.assemble(directory, base, ["a", "b"], onto="main")
    assert [entry["branch"] for entry in result["branches"]] == ["a", "b"]
    assert result["base"] == base
    assert "is not merged again" in caplog.text


def test_without_a_branch_to_merge_into_head_stays_at_the_base(repo):
    directory, base = repo
    result = integrate.assemble(directory, base, ["a", "b"])
    assert result["base"] == base
    assert _git(directory, "rev-parse", "HEAD").strip() == base


def test_the_container_is_told_which_branch_the_result_lands_on():
    argv = sandbox.integrate_argv("repo", ["codetwine-loom/a", "codetwine-loom/b"], onto="main")
    assert argv[-2:] == ["--onto", "main"]


def test_no_branch_to_merge_into_is_not_passed_on():
    assert "--onto" not in sandbox.integrate_argv("repo", ["codetwine-loom/a", "codetwine-loom/b"])


# == The tests the merge broke, and the ones not counted ==


def _tested(status, failed=(), excerpt="the excerpt"):
    return {"status": status, "failed": list(failed),
            "failed_excerpt": excerpt if status == "failed" else "",
            "targets": [], "output": ""}


def _never():
    raise AssertionError("HEAD's tree was measured when there was nothing to compare")


def test_a_green_merged_tree_counts_every_failure_and_is_held_to_what_head_ran():
    head = edit.RunResult(_tested("passed"))
    head.ran_count, head.ran_name_list = 1, ["t::one"]
    assert integrate._baseline_set(_tested("passed"), lambda: head) == edit.ran_entry_set(head)
    assert integrate._baseline_set(_tested("passed"), lambda: _tested("passed")) == frozenset()


def test_a_test_the_merge_broke_is_not_in_the_baseline():
    """Green on what the branches were combined onto, red on the merged tree: the
    merge's doing, so it is counted."""
    baseline = integrate._baseline_set(_tested("failed", ["t::one"]),
                                      lambda: _tested("passed"))
    assert baseline == frozenset()


def test_a_test_red_in_both_states_is_what_is_not_counted():
    baseline = integrate._baseline_set(_tested("failed", ["t::one", "t::two"]),
                                      lambda: _tested("failed", ["t::one", "t::gone"]))
    assert baseline == frozenset({"t::one"})


def test_a_test_that_fails_more_times_on_the_merged_tree_is_counted(monkeypatch):
    """A test's subtests are named as the test: one more failing subtest is one more
    time the name fails."""
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    baseline = integrate._baseline_set(_tested("failed", ["t::one", "t::one", "t::two"]),
                                      lambda: _tested("failed", ["t::one", "t::two"]))
    assert edit.baseline_name_list(baseline) == ["t::one", "t::two"]
    assert integrate._test_reason(_tested("failed", ["t::one", "t::two"]), baseline) == ""
    assert integrate._test_reason(_tested("failed", ["t::one", "t::one", "t::two"]),
                                  baseline).startswith("- `run them` fails t::one, which")


def test_a_red_merged_tree_that_named_no_test_is_still_counted_when_head_was_green():
    assert integrate._baseline_set(_tested("failed"),
                                  lambda: _tested("passed")) == frozenset()


def test_two_red_states_where_one_named_no_test_leave_the_tests_out(caplog):
    """There is nothing to tell the two states apart by, so the gate is off rather
    than guessing."""
    with caplog.at_level(logging.INFO, logger="codetwine_loom.integrate"):
        assert integrate._baseline_set(_tested("failed"),
                                      lambda: _tested("failed", ["t::one"])) is None
        assert integrate._baseline_set(_tested("failed", ["t::one"]),
                                      lambda: _tested("failed")) is None
    assert "named no test to compare by" in caplog.text


def test_a_red_run_that_counts_failing_tests_it_does_not_name_keeps_the_tests_in(monkeypatch):
    """Where HEAD's run or the merged tree's counts failing tests and names none of them,
    the gate stays on and says the names cannot be read."""
    monkeypatch.setattr(config, "TEST_COMMAND", ["t"])
    nameless = edit.RunResult(_tested("failed"))
    nameless.is_name_short = nameless.is_ran_hold = True
    nameless.ran_count = 2
    named = edit.RunResult(_tested("failed", ["t::one"]))
    named.ran_count, named.ran_name_list, named.is_ran_hold = 2, ["t::one", "t::two"], True
    green = edit.RunResult(_tested("passed"))
    green.ran_count, green.ran_name_list, green.is_ran_hold = 2, ["t::one", "t::two"], True
    short_reason = ("- `t` does not print how many tests it passed and failed, or its "
                    "`PASSED` and `FAILED` lines do not name each test it counts, now or "
                    "before the branches were combined, so the tests it runs cannot be "
                    "checked against the tests that ran then")
    for union, head in ((nameless, named), (green, nameless)):
        baseline = integrate._baseline_set(union, lambda head=head: head)
        assert baseline is not None
        assert integrate._test_reason(union, baseline) == short_reason
    baseline = integrate._baseline_set(nameless, lambda: nameless)
    assert integrate._test_reason(nameless, baseline) == "- `t` fails (the excerpt)"
    # A test the merged tree's run names that HEAD's nameless run does not is not
    # counted as failing before, and is named as broken
    baseline = integrate._baseline_set(named, lambda: nameless)
    assert edit.baseline_name_list(baseline) == []
    assert integrate._test_reason(named, baseline) == (
        "- `t` fails t::one, which passed before the branches were combined (the excerpt)")
    assert edit.baseline_name_list(integrate._baseline_set(nameless, lambda: named)) \
        == ["t::one"]


def test_a_head_whose_tests_did_not_run_leaves_the_tests_out():
    assert integrate._baseline_set(_tested("failed", ["t::one"]),
                                  lambda: _tested("did not run")) is None


def test_a_run_that_says_nothing_of_the_tests_it_ran_keeps_the_tests_in(monkeypatch):
    """HEAD's run that prints no line counting its tests is not left out, even where it
    did not run: the merged tree's run is held to it and says so, and no merge over that
    HEAD is settled. A merged tree's run of that kind is held to the tests HEAD's ran."""
    monkeypatch.setattr(config, "TEST_COMMAND", ["t"])
    unread_list = []
    for status, failed in (("did not run", ()), ("passed", ()), ("failed", ["t::one"])):
        unread = edit.RunResult(_tested(status, failed))
        unread.is_ran_unread, unread.is_ran_hold = True, status != "did not run"
        unread_list.append(unread)
    green = _ran_tested(["t::one", "t::two"])
    green.is_ran_hold = True
    short_reason = integrate._TEST_SHORT_REASON.format(command="t")
    for head in unread_list:
        baseline = integrate._baseline_set(green, lambda head=head: head)
        assert baseline is not None
        assert integrate._test_reason(green, baseline) == short_reason
        assert integrate._is_head_short(baseline, "t") is True
    # A merged tree that breaks a test over HEAD's run that did not run
    named = _ran_tested(["t::one"], status="failed", failed=["t::one"])
    named.is_ran_hold = True
    baseline = integrate._baseline_set(named, lambda: unread_list[0])
    assert integrate._test_reason(named, baseline) == (
        "- `t` fails t::one, which passed before the branches were combined (the excerpt)")
    # A merged tree's run that no longer counts its tests, over HEAD's that named them
    baseline = integrate._baseline_set(unread_list[1], lambda: green)
    assert integrate._test_reason(unread_list[1], baseline) == (
        "- `t` no longer prints which of its tests ran, as it did before the branches were "
        "combined, so they cannot be checked against the tests that ran then")
    assert integrate._is_head_short(baseline, "t") is False


def test_a_head_whose_tests_ended_before_every_test_was_run_holds_every_merged_run(
        monkeypatch):
    """A test file that cannot be collected ends pytest before any test, as -x ends it
    at its first failure: the tests are kept in the check, whatever the merged tree's
    run says is a reason, and a test it names that HEAD's run did not is named."""
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_COMMAND", ["t"])
    early = dict(_tested("failed", ["tests/test_plot.py"]), output=(
        "ERROR tests/test_plot.py\n!!!!!!!! Interrupted: 1 error during collection "
        "!!!!!!!!\n1 error in 0.07s"))
    stop_reason = ("- `t` stopped before it had run every test before the branches were "
                   "combined, so the tests it runs cannot be checked against how they "
                   "stood then")
    for merged in (dict(early), _tested("passed"), _tested("failed", ["tests/test_plot.py"])):
        baseline = integrate._baseline_set(dict(merged), lambda: dict(early))
        assert edit.baseline_name_list(baseline) == ["tests/test_plot.py"]
        assert edit.is_stop_before(baseline, "t")
        assert integrate._test_reason(dict(merged), baseline) == stop_reason
    baseline = integrate._baseline_set(_tested("did not run"), lambda: dict(early))
    assert integrate._test_reason(_tested("did not run"), baseline).startswith(
        "- `t` did not run")
    assert integrate._test_reason(_tested("failed", ["t::one"]), baseline).startswith(
        "- `t` fails t::one, which passed before the branches were combined")
    # Of several commands, the one whose run at HEAD ended so
    joined = edit.join_baseline({"a": baseline, "b": frozenset()}, "a")
    assert edit.is_stop_before(joined, "a") and not edit.is_stop_before(joined, "b")
    assert not edit.is_stop_before(frozenset({"t::one"}), "t")
    # The merged tree's run ending so, where HEAD's ran to the end, is the merge's doing
    assert integrate._baseline_set(dict(early), lambda: _tested("passed")) == frozenset()
    baseline = integrate._baseline_set(
        dict(early, failed=["tests/test_plot.py", "t::one"]),
        lambda: _tested("failed", ["t::one"]))
    assert baseline == frozenset({"t::one"})
    assert "tests/test_plot.py" in integrate._test_reason(dict(early), baseline)


def _record_output(name):
    """The output of one recorded run of a test command."""
    path = pathlib.Path(__file__).parent / "data" / "test_output" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["output"]


@pytest.mark.parametrize("ext, name_list, output, whole_output, stop_line", [
    ("py", ["tests/test_plot.py"],
     "ERROR tests/test_plot.py\n!!!!!!!! Interrupted: 1 error during collection "
     "!!!!!!!!\n1 error in 0.07s", "ERROR tests/test_plot.py - boom\n1 error in 0.1s",
     "!!!!!!!! Interrupted: 1 error during collection !!!!!!!!"),
    ("java", ["com.example.b.BTest.one", "BTest.one"], _record_output("java_B_multi"),
     _record_output("java_R"), "[ERROR] COMPILATION ERROR : "),
    ("kt", ["BTest > one()"], _record_output("kt_B_multi"), _record_output("kt_R_multi"),
     "Execution failed for task ':a:compileKotlin'.")])
def test_a_union_whose_tests_ended_before_every_test_was_run_is_not_settled_by_them(
        monkeypatch, ext, name_list, output, whole_output, stop_line):
    """Every failing test the union's run names failed at HEAD, whose run went on to
    its end. The reason starts with the lines saying the run ended early."""
    monkeypatch.setattr(config, "PRIMARY_EXT", ext)
    early = dict(_tested("failed", name_list), output=output)
    whole = dict(early, output=whole_output)
    baseline = integrate._baseline_set(dict(early), lambda: dict(whole))
    assert baseline == frozenset(name_list)
    reason = integrate._test_reason(dict(early), baseline)
    assert reason.startswith("- `") and \
        f"stopped before it had run every test ({stop_line}\n" in reason
    assert "\nthe excerpt). The failing tests it named up to there: " in reason
    assert reason.endswith(f"up to there: {', '.join(name_list)}")
    # A run that named no test at all, held to a baseline holding names
    assert integrate._test_reason(dict(early, failed=[]), baseline).endswith(
        "up to there: none")
    # The same names from a run that went on to its end are no reason
    assert integrate._test_reason(dict(whole), baseline) == ""


def test_tests_that_ran_neither_on_the_union_nor_at_head_leave_the_tests_out(caplog):
    with caplog.at_level(logging.INFO, logger="codetwine_loom.integrate"):
        assert integrate._baseline_set(_tested("did not run"),
                                      lambda: _tested("did not run")) is None
    assert "did not run over what the branches were combined onto" in caplog.text


def _run_of(**part_by_command):
    """A run of several test commands as edit.run_test() returns it: {command: its own
    result}. The run as a whole says what edit._merge_test_run() would."""
    status_list = [part["status"] for part in part_by_command.values()]
    status = "failed" if "failed" in status_list else \
        "did not run" if "did not run" in status_list else "passed"
    result = edit.RunResult(_tested(
        status, [name for part in part_by_command.values() for name in part["failed"]]))
    result.part_dict = {command: edit.RunResult(part)
                        for command, part in part_by_command.items()}
    result.primary_name = "py" if "py" in part_by_command else ""
    result.is_whole = True
    return result


_NOT_RUN = _tested("did not run")


def _baseline(name_list=(), **word_by_command):
    """A baseline of several test commands: the names, and what each command is in it
    ("not compared", "green before" or "red before")."""
    return frozenset(name_list) | {edit.baseline_mark(word, command)
                                   for command, word in word_by_command.items()}


def test_a_command_that_runs_neither_before_nor_after_leaves_the_others_compared():
    union = _run_of(py=_tested("passed"), js=_NOT_RUN)
    assert union["status"] == "did not run"
    baseline = integrate._baseline_set(union, lambda: _run_of(py=_tested("passed"),
                                                             js=_NOT_RUN))
    assert baseline == _baseline(py="green before", js="not compared")
    assert integrate._test_reason(union, baseline) == ""
    # A test of the command that does run is still held against the baseline
    reason = integrate._test_reason(
        _run_of(py=_tested("failed", ["t::one"]), js=_NOT_RUN), baseline)
    assert reason.startswith("- `py` fails t::one") and "js" not in reason
    # and so is a red run of it that names no test
    reason = integrate._test_reason(_run_of(py=_tested("failed"), js=_NOT_RUN), baseline)
    assert reason.startswith("- `py` fails (")


def test_a_command_that_stops_running_with_the_merge_is_what_the_merge_broke():
    union = _run_of(py=_tested("passed"), js=_NOT_RUN)
    baseline = integrate._baseline_set(union, lambda: _run_of(py=_tested("passed"),
                                                             js=_tested("passed")))
    assert baseline == _baseline(py="green before", js="green before")
    assert integrate._test_reason(union, baseline).startswith("- `js` did not run")
    # A command that is not part of a later run at all is one that did not run
    assert integrate._test_reason(_run_of(py=_tested("passed")), baseline).startswith(
        "- `js` did not run")
    assert integrate._test_reason(_run_of(js=_tested("passed")), baseline).startswith(
        "- `py` did not run")
    # A run given paths is held by the commands it ran alone
    path_run = _run_of(py=_tested("passed"))
    path_run.is_whole = False
    assert integrate._test_reason(path_run, baseline) == ""
    # and a command the baseline does not know is not held against it
    assert integrate._test_reason(_run_of(py=_tested("passed"), js=_tested("passed"),
                                          rs=_tested("failed", ["r::a"])), baseline) == ""


def test_a_command_that_did_not_run_on_the_union_is_held_to_its_own_run_at_head():
    """Its failures at HEAD are not counted once it runs again, whatever the other
    command did."""
    union = _run_of(py=_NOT_RUN, js=_tested("passed"))
    baseline = integrate._baseline_set(
        union, lambda: _run_of(py=_tested("failed", ["t::old"]), js=_tested("passed")))
    assert baseline == _baseline(["t::old"], py="red before", js="green before")
    assert integrate._test_reason(
        _run_of(py=_tested("failed", ["t::old"]), js=_tested("passed")), baseline) == ""
    assert "t::new" in integrate._test_reason(
        _run_of(py=_tested("failed", ["t::old", "t::new"]), js=_tested("passed")),
        baseline)


def test_a_red_run_naming_no_test_is_not_hidden_by_another_command_s_baseline():
    baseline = integrate._baseline_set(
        _run_of(py=_tested("passed"), js=_tested("failed", ["j::old"])),
        lambda: _run_of(py=_tested("passed"), js=_tested("failed", ["j::old"])))
    assert baseline == _baseline(["j::old"], py="green before", js="red before")
    reason = integrate._test_reason(
        _run_of(py=_tested("failed"), js=_tested("failed", ["j::old"])), baseline)
    assert reason.startswith("- `py` fails (") and "js" not in reason
    # The command with names in the baseline cannot be told apart, as with one command
    assert integrate._test_reason(
        _run_of(py=_tested("passed"), js=_tested("failed")), baseline) == ""


def test_head_s_tree_is_measured_once_for_every_command():
    count = []

    def at_head():
        count.append(1)
        return _run_of(py=_tested("failed", ["t::a"]), js=_tested("failed", ["j::a"]))

    baseline = integrate._baseline_set(
        _run_of(py=_tested("failed", ["t::a"]), js=_tested("failed", ["j::a", "j::b"])),
        at_head)
    assert len(count) == 1
    assert edit.baseline_name_list(baseline) == ["j::a", "t::a"]
    # A command the run at HEAD did not hold is one that did not run there
    baseline = integrate._baseline_set(
        _run_of(py=_tested("passed"), js=_tested("failed", ["j::a"])),
        lambda: _run_of(py=_tested("passed")))
    assert baseline == _baseline(py="green before", js="not compared")


def test_a_baseline_of_one_command_holds_a_later_run_of_several_by_that_command():
    """The command that was not there before is not said to have passed before."""
    later = _run_of(py=_tested("failed", ["t::a"]), js=_tested("failed", ["j::new"]))
    assert integrate._test_reason(later, frozenset({"t::a"})) == ""
    reason = integrate._test_reason(
        _run_of(py=_tested("failed", ["t::b"]), js=_tested("failed")), frozenset({"t::a"}))
    assert reason.startswith("- `py` fails t::b") and "js" not in reason


def test_one_command_that_is_not_the_primary_is_named_in_its_baseline():
    """The primary command joins the run later, and is not what the baseline holds."""
    baseline = integrate._baseline_set(_run_of(js=_tested("passed")),
                                      lambda: _run_of(js=_tested("passed")))
    assert baseline == _baseline(js="green before")
    reason = integrate._test_reason(
        _run_of(py=_tested("passed"), js=_tested("failed", ["j::broke"])), baseline)
    assert reason.startswith("- `js` fails j::broke")
    assert integrate._test_reason(
        _run_of(py=_tested("failed", ["t::a"]), js=_tested("passed")), baseline) == ""


def test_the_primary_command_gone_from_a_whole_run_is_held_to_its_own_baseline(
        monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND", ["py"])
    reason = integrate._test_reason(_run_of(js=_tested("failed", ["j::new"])),
                                    frozenset())
    assert reason.startswith("- `py` did not run") and "j::new" not in reason


def test_the_commands_out_of_the_check_are_said_in_the_state(monkeypatch):
    monkeypatch.setattr(integrate.edit, "test_can_run", lambda: True)
    monkeypatch.setitem(integrate._STAGE_DICT, "head_baseline", False)
    monkeypatch.setitem(integrate._STAGE_DICT, "not_compared", ["npm test --silent"])
    line_list = integrate._test_start_line_list({"already_failing": []})
    assert len(line_list) == 2 and "`npm test --silent`" in line_list[1]
    assert "The other tests are." in line_list[1]
    assert integrate._test_start_line_list(
        {"already_failing": ["t::a"]})[3] == integrate._ALREADY_RED
    # The line for tests that could not be compared at all stands alone
    assert integrate._test_start_line_list({"already_failing": None}) == [
        "", integrate._TEST_NO_COMPARE]
    monkeypatch.setitem(integrate._STAGE_DICT, "not_compared", [])
    assert integrate._test_start_line_list({"already_failing": []}) == []
    integrate._STAGE_DICT["not_compared"] = ["x"]
    integrate.reset_stage()
    assert integrate._STAGE_DICT["not_compared"] == []


def test_the_state_holds_the_test_names_of_a_baseline_of_several_commands():
    """What a baseline holds beside the names is never among the tests the model reads,
    and the variable it is handed has the keys it always had."""
    baseline = _baseline(["t::old"], py="red before", js="not compared")
    work = integrate.build_work_dict(_merged(), [], [], baseline, "", 0)
    assert work["already_failing"] == ["t::old"]
    assert set(work) == set(integrate.build_work_dict(
        _merged(), [], [], frozenset({"t::old"}), "", 0))
    assert "\x1f" not in repr(work)


def test_a_test_named_like_a_command_s_entry_is_a_test():
    names = frozenset({"rows: not compared", "py: red before"})
    assert edit.baseline_name_list(names) == sorted(names)
    assert integrate._test_reason(_tested("failed", sorted(names)), names) == ""


def test_a_run_of_one_command_is_read_as_it_always_was():
    plain = _run_of(py=_NOT_RUN)
    assert integrate._baseline_set(plain, lambda: _run_of(py=_NOT_RUN)) is None
    assert integrate._baseline_set(
        _run_of(py=_tested("failed", ["t::a"])),
        lambda: _run_of(py=_tested("failed", ["t::a"]))) == frozenset({"t::a"})
    reason = integrate._test_reason(_run_of(py=_tested("failed", ["t::b"])),
                                    frozenset({"t::a"}))
    # A dict with no command of its own is named by the commands a whole run runs
    assert reason.replace("`py`", f"`{edit.test_command_text()}`") == \
        integrate._test_reason(_tested("failed", ["t::b"]), frozenset({"t::a"}))


def test_the_names_shown_as_already_failing_are_the_tests_alone():
    baseline = _baseline(["t::old"], py="red before", js="not compared")
    assert edit.baseline_name_list(baseline) == ["t::old"]
    assert edit.not_compared_list(baseline) == ["js"]


def _no_test_run_of(**part_by_command):
    """_run_of(), with the commands named "zero" in part_by_command's values found to
    have no test to run."""
    result = _run_of(**{command: _NOT_RUN if part == "zero" else part
                        for command, part in part_by_command.items()})
    for command, part in part_by_command.items():
        result.part_dict[command].is_no_test = part == "zero"
    return result


def test_another_command_s_run_that_finds_no_test_is_held_by_what_it_ran_before():
    """The same reading for the primary command and another: tests that ran before
    and find nothing to run now are a reason, no test before and none now is not,
    and no test before and a failure now is."""
    green_head = lambda: _run_of(py=_tested("passed"), js=_tested("passed"))  # noqa: E731
    union = _no_test_run_of(py=_tested("passed"), js="zero")
    baseline = integrate._baseline_set(union, green_head)
    assert baseline == _baseline(py="green before", js="green before")
    assert "`js` did not run" in integrate._test_reason(union, baseline)
    zero_head = lambda: _no_test_run_of(py=_tested("passed"), js="zero")  # noqa: E731
    baseline = integrate._baseline_set(union, zero_head)
    assert baseline == _baseline(py="green before", js="no test before")
    assert edit.not_compared_list(baseline) == [] and edit.baseline_name_list(baseline) == []
    assert integrate._test_reason(union, baseline) == ""
    assert "`js` fails" in integrate._test_reason(
        _run_of(py=_tested("passed"), js=_tested("failed", ["j::one"])), baseline)
    py_zero = _no_test_run_of(py="zero", js=_tested("passed"))
    assert "`py` did not run" in integrate._test_reason(
        py_zero, _baseline(py="green before", js="green before"))


def test_a_union_whose_tests_do_not_run_is_held_to_a_green_head():
    """HEAD's suite ran, so a union where it no longer runs is the merge's doing."""
    assert integrate._baseline_set(_tested("did not run"),
                                  lambda: _tested("passed")) == frozenset()


def test_a_union_whose_tests_do_not_run_is_held_to_what_head_named_failing():
    assert integrate._baseline_set(_tested("did not run"),
                                  lambda: _tested("failed", ["t::one"])) \
        == frozenset({"t::one"})


def test_a_union_whose_tests_do_not_run_against_a_head_red_without_names_is_left_out():
    assert integrate._baseline_set(_tested("did not run"), lambda: _tested("failed")) is None


def test_a_union_with_no_test_to_run_is_held_to_what_head_ran(monkeypatch):
    """HEAD is read whatever the union's run is not green with: tests that ran there
    and no longer run on the union are the merge's doing."""
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    no_test = dict(_tested("did not run"), output="1 skipped in 0.01s\n")
    assert integrate._baseline_set(no_test, lambda: _tested("passed")) == frozenset()
    assert integrate._baseline_set(no_test, lambda: _tested("failed", ["t::one"])) \
        == frozenset({"t::one"})
    assert integrate._baseline_set(no_test, lambda: no_test) == edit.NO_TEST_BASELINE


def test_a_red_union_over_a_head_with_no_test_to_run_is_held_as_over_a_green_one(
        monkeypatch):
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    no_test = dict(_tested("did not run"), output="no tests ran in 0.01s\n")
    assert integrate._baseline_set(_tested("failed", ["t::one"]),
                                  lambda: no_test) == frozenset()
    assert integrate._baseline_set(_tested("failed"), lambda: no_test) == frozenset()
    crash = dict(_tested("did not run"), output="ImportError while loading conftest\n")
    assert integrate._baseline_set(crash, lambda: no_test) == edit.NO_TEST_BASELINE


def test_no_test_command_leaves_the_tests_out():
    assert integrate._baseline_set(None, _never) is None


def test_the_tests_over_head_are_measured_on_heads_tree_and_the_union_put_back(
        repo, monkeypatch):
    """The union is uncommitted against HEAD, so HEAD's tree is put in place to run
    the tests and the union put back afterwards, HEAD staying where it was."""
    directory, base = repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    seen = {}

    def run_test():
        seen["f.txt"] = _read(directory, "f.txt")
        return _tested("passed")

    monkeypatch.setattr(edit, "run_test", run_test)
    assert integrate._at_head(directory, merged["commit"], edit.run_test) == _tested("passed")
    assert seen["f.txt"] == "one\ntwo\nthree\n"
    assert "<<<<<<<" in _read(directory, "f.txt")
    assert _read(directory, "a_only.txt") == "a\n"


def test_a_tracked_file_a_run_over_the_union_rewrote_is_as_it_was_after_the_tests_over_head(
        repo):
    """A command run over the union wrote over a file the union tracks."""
    directory, base = repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    _write(directory, "g.txt", "rewritten by a run\n")
    seen = {}

    def run_test():
        seen["g.txt"] = _read(directory, "g.txt")
        return _tested("passed")

    assert integrate._at_head(directory, merged["commit"], run_test) == _tested("passed")
    assert seen == {"g.txt": "kept\n"}
    assert _read(directory, "g.txt") == "rewritten by a run\n"
    assert "<<<<<<<" in _read(directory, "f.txt")


def test_a_file_head_tracks_that_the_union_ignores_is_as_it_was_after_the_tests_over_head(
        tmp_path, monkeypatch):
    """One side stopped tracking a file and ignores it; what a run over the union
    left at its path is there again once HEAD's tree has been measured."""
    directory, base = _two_file_repo(tmp_path, "a.txt", "a\n", ".gitignore", "made.json\n",
                                     {"f.txt": "one\n", "made.json": "{}\n"})
    _git(directory, "checkout", "-q", "b")
    _git(directory, "rm", "-q", "--cached", "made.json")
    _git(directory, "commit", "-qm", "b stops tracking it")
    _git(directory, "checkout", "-q", "--detach", base)
    _git(directory, "clean", "-qfdx")
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert not os.path.exists(os.path.join(directory, "made.json"))
    _write(directory, "made.json", '{"made": "by a run"}\n')
    seen = {}

    def run_test():
        seen["made.json"] = _read(directory, "made.json")
        return _tested("passed")

    assert integrate._at_head(directory, merged["commit"], run_test) == _tested("passed")
    assert seen == {"made.json": "{}\n"}
    assert _read(directory, "made.json") == '{"made": "by a run"}\n'
    assert _git(directory, "status", "--porcelain") == "A  .gitignore\nA  a.txt\nD  made.json\n"
    assert _git(directory, "rev-parse", "HEAD").strip() == base


def test_head_s_tree_is_not_measured_where_what_is_in_its_way_cannot_be_kept(
        tmp_path, monkeypatch):
    """One side stopped tracking a file and ignores it; a pipe is at its path."""
    directory, base = _two_file_repo(tmp_path, "a.txt", "a\n", ".gitignore", "made.json\n",
                                     {"f.txt": "one\n", "made.json": "{}\n"})
    _git(directory, "checkout", "-q", "b")
    _git(directory, "rm", "-q", "--cached", "made.json")
    _git(directory, "commit", "-qm", "b stops tracking it")
    _git(directory, "checkout", "-q", "--detach", base)
    _git(directory, "clean", "-qfdx")
    merged = integrate.assemble(directory, base, ["a", "b"])
    os.mkfifo(os.path.join(directory, "made.json"))
    status = _git(directory, "status", "--porcelain")

    def never():
        raise AssertionError("HEAD's tree was measured")

    monkeypatch.setattr(edit, "run_test", never)
    assert integrate._at_head(directory, merged["commit"], never) is None
    head_dict = integrate._head_test_dict(directory, merged["commit"])
    assert head_dict["status"] == "did not run" and head_dict["failed"] == []
    assert integrate._baseline_set(_tested("failed", ["t::one"]), lambda: head_dict) is None
    assert _git(directory, "status", "--porcelain") == status
    assert not os.path.isfile(os.path.join(directory, "made.json"))


def test_what_is_in_the_way_of_a_path_is_the_first_thing_on_it_that_is_no_directory(
        tmp_path):
    directory = str(tmp_path)
    os.makedirs(os.path.join(directory, "a", "b"))
    _write(directory, "a/b/file.txt", "x\n")
    os.symlink("b", os.path.join(directory, "a", "link"))
    os.mkfifo(os.path.join(directory, "a", "pipe"))
    for rel_path, in_way_path in (
            ("a/b/file.txt", "a/b/file.txt"), ("a/b/file.txt/under/it", "a/b/file.txt"),
            ("a/link/file.txt", "a/link"), ("a/link", "a/link"), ("a/pipe/x", "a/pipe"),
            ("a/b", ""), ("a/b/nothing", ""), ("nothing/at/all", ""), ("a", "")):
        assert integrate._in_way_path(directory, rel_path) == in_way_path, rel_path


def test_a_test_that_was_already_failing_is_not_sent_back(monkeypatch):
    """A test already failing before the merge is not sent back."""
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    assert integrate._test_reason(_tested("failed", ["t::one"]),
                                   frozenset({"t::one"})) == ""


def test_a_test_the_settling_broke_is_sent_back_by_name(monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    reason = integrate._test_reason(_tested("failed", ["t::two"]), frozenset())
    assert "t::two" in reason and "run them" in reason and "the excerpt" in reason


def test_only_the_tests_the_settling_broke_are_sent_back(monkeypatch):
    """A red suite that got redder says which half is this stage's doing."""
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    reason = integrate._test_reason(_tested("failed", ["t::one", "t::two"]),
                                     frozenset({"t::one"}))
    assert "t::two" in reason and "t::one" not in reason.split("(")[0]


def test_a_failure_with_no_name_is_sent_back_where_the_suite_was_green(monkeypatch):
    """Nothing can be named, but the suite passed at the merge, so it is this
    stage's doing."""
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    reason = integrate._test_reason(_tested("failed"), frozenset())
    assert "run them" in reason and "the excerpt" in reason


def test_a_failure_with_no_name_is_not_sent_back_where_the_suite_was_red(monkeypatch):
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    assert integrate._test_reason(_tested("failed"), frozenset({"t::one"})) == ""


def test_a_green_run_is_no_reason_where_head_s_run_said_nothing_of_what_it_ran():
    assert integrate._test_reason(_tested("passed"), frozenset()) == ""


def test_a_run_that_did_not_run_is_a_reason_where_the_tests_are_compared(monkeypatch):
    """The suite ran before; that it no longer runs is what the settling left."""
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    did_not_run = dict(_tested("did not run"), failed_excerpt="ModuleNotFoundError: helper")
    reason = integrate._test_reason(did_not_run, frozenset())
    assert reason == "- `run them` did not run (ModuleNotFoundError: helper)"
    assert integrate._test_reason(did_not_run, frozenset({"t::one"})) == reason
    assert integrate._test_reason(did_not_run, None) == ""


def test_a_run_that_found_no_test_is_a_reason_only_where_the_tests_ran_before(monkeypatch):
    """Tests that ran before and run no more are left out by the change; where they
    found none before either, the two runs cannot be compared."""
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    no_test = dict(_tested("did not run"), output="1 skipped in 0.01s\n",
                   failed_excerpt="1 skipped in 0.01s")
    reason = "- `run them` did not run (1 skipped in 0.01s)"
    assert integrate._test_reason(no_test, frozenset()) == reason
    assert integrate._test_reason(no_test, frozenset({"t::one"})) == reason
    assert integrate._test_reason(no_test, edit.NO_TEST_BASELINE) == ""
    assert "fails" in integrate._test_reason(_tested("failed", ["t::one"]),
                                            edit.NO_TEST_BASELINE)
    assert integrate._test_reason(_tested("passed"), edit.NO_TEST_BASELINE) == ""


def test_the_check_runs_no_test_where_there_is_nothing_to_compare(monkeypatch, repo):
    """None is the one value that keeps the whole gate off."""
    directory, _ = repo
    ran = []
    monkeypatch.setattr(edit, "run_test", lambda *a, **k: ran.append(True))
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    assert integrate._reason_list(directory, "out", _analysis(), None, None) == []
    assert ran == []


def _ran_tested(name_list, status="passed", failed=()):
    """One run_test() return value of one command that named the tests it ran."""
    result = edit.RunResult(_tested(status, failed))
    result.ran_name_list, result.ran_count = sorted(name_list), len(name_list)
    return result


def test_a_test_that_ran_over_head_s_tree_and_runs_no_more_is_a_reason(monkeypatch, repo):
    """Whatever keeps it from running - a branch's setting, conftest.py or skip, or the
    merging stage's - the merged tree is held to the tests HEAD's run ran, a green one
    included. A test of a file a branch changed that still has a test running is a
    branch's own change of its tests."""
    directory, _ = repo
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "TEST_COMMAND", ["t"])
    monkeypatch.setattr(config, "OTHER_TEST_LIST", [])
    os.makedirs(os.path.join(directory, "tests"))
    for name in ("test_a.py", "check_b.py"):
        _write(directory, f"tests/{name}", "x\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-q", "-m", "tests")
    head = _ran_tested(["tests/test_a.py::a", "tests/check_b.py::b"])
    union = _ran_tested(["tests/test_a.py::a"])
    baseline = integrate._baseline_set(union, lambda: head)
    assert integrate._test_reason(union, baseline) == (
        "- `t` no longer runs tests/check_b.py::b, which ran before the branches were "
        "combined: they are not found, are left out or skip now")
    _write(directory, "tests/check_b.py", "y\n")
    assert integrate._test_reason(union, baseline) != ""
    assert integrate._test_reason(_ran_tested(["tests/test_a.py::a", "tests/check_b.py::c"]),
                                  baseline) == ""
    # A whole run that no longer says which tests it ran
    quiet = _ran_tested(["tests/test_a.py::a"])
    quiet.ran_name_list, quiet.is_ran_hold = None, True
    assert integrate._test_reason(quiet, baseline) == (
        "- `t` no longer prints which of its tests ran, as it did before the branches were "
        "combined, so they cannot be checked against the tests that ran then")
    # A whole run that names some of the tests it ran, not every one, now or over HEAD's
    # tree: a test file a branch changed does not leave it held by the count
    short = _ran_tested(["tests/test_a.py::a"])
    short.ran_name_list, short.is_name_short, short.is_ran_hold = None, True, True
    short_reason = ("- `t` does not print how many tests it passed and failed, or its "
                    "`PASSED` and `FAILED` lines do not name each test it counts, now or "
                    "before the branches were combined, so the tests it runs cannot be "
                    "checked against the tests that ran then")
    assert integrate._test_reason(short, baseline) == short_reason
    short_baseline = integrate._baseline_set(short, lambda: short)
    head.is_ran_hold = True
    assert integrate._test_reason(head, short_baseline) == short_reason


def test_the_check_runs_the_tests_where_there_is(monkeypatch, repo):
    """A suite that was red at the merge is still run."""
    directory, _ = repo
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    monkeypatch.setattr(edit, "run_test",
                        lambda *a, **k: _tested("failed", ["t::one", "t::two"]))
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    reasons = integrate._reason_list(directory, "out", _analysis(), frozenset({"t::one"}), None)
    assert len(reasons) == 1 and "t::two" in reasons[0]


def test_the_state_block_names_the_tests_that_were_already_failing(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], frozenset({"t::one"}), "", 0)
    start = integrate.start_text(work)
    assert "failing before the branches were combined and still fail" in start
    assert "t::one" in start
    note = integrate.make_state(directory, "HEAD", work, lambda: [])[0](1, 20)
    assert "t::one" not in note and "The whole test suite, the last time it was run" in note


def test_the_state_block_says_when_the_merge_broke_no_test(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    note = integrate.make_state(directory, "HEAD", work, lambda: [])[0](1, 20)
    assert "said nothing it did not say before the branches were combined" in note
    assert "(before the first turn)" in note


def test_the_state_block_names_the_tests_the_merge_broke(repo, monkeypatch):
    """The line is the one the check would send back, so the two never disagree."""
    directory, _ = repo
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    reason = integrate._test_reason(_tested("failed", ["t::two"]), frozenset())
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), reason, 0)
    note = integrate.make_state(directory, "HEAD", work, lambda: [])[0](1, 20)
    assert "did not say before the branches were combined" in note
    assert reason in note and work["tests"] == reason


def test_the_tests_line_follows_the_last_run_of_the_whole_suite(repo, monkeypatch):
    """Whoever ran it. The line measured before the first turn is replaced by what the
    model's own runs said."""
    directory, _ = repo
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    runs = {"n": 3, "result": _tested("failed", ["t::two"])}
    monkeypatch.setattr(edit, "last_whole_run", lambda: runs)
    first = integrate._test_reason(runs["result"], frozenset())
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), first, 0)
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [],
                                              frozenset())
    # the run make_state() was built after is not a new one
    note = turn_note(1, 20)
    assert "(before the first turn)" in note and "t::two" in note
    # the model ran the whole suite on turn 1 and it passed
    runs = {"n": 4, "result": _tested("passed")}
    monkeypatch.setattr(edit, "last_whole_run", lambda: runs)
    note = turn_note(2, 20)
    assert "(on turn 1)" in note and "t::two" not in note
    assert work["tests"] == "" and work["tests_at"] == 1
    # nothing ran on turn 2, so the line stays
    assert "(on turn 1)" in turn_note(3, 20)


def test_the_tests_line_does_not_move_where_the_tests_are_not_checked(repo,
                                                                    monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    monkeypatch.setattr(edit, "last_whole_run",
                        lambda: {"n": 9, "result": _tested("passed")})
    work = integrate.build_work_dict(_merged(), [], [], None, "", 0)
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [])
    monkeypatch.setattr(edit, "last_whole_run",
                        lambda: {"n": 10, "result": _tested("passed")})
    assert "The whole test suite" not in turn_note(2, 20)
    assert "not part of the check" in integrate.start_text(work)
    assert work["tests_at"] == 0


def test_the_state_block_says_when_the_tests_are_not_part_of_the_check(repo,
                                                                      monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], None, "", 0)
    assert "not part of the check" in integrate.start_text(work)
    note = integrate.make_state(directory, "HEAD", work, lambda: [])[0](1, 20)
    assert "not part of the check" not in note


def test_the_state_block_says_nothing_about_tests_there_are_none_of(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], None, "", 0)
    note = integrate.make_state(directory, "HEAD", work, lambda: [])[0](1, 20)
    assert "test" not in note


def test_the_tests_already_failing_do_not_make_the_work_list_incomplete(monkeypatch):
    """They are not work to settle, so cutting the list says nothing is left out."""
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    work = integrate.build_work_dict(_merged(), [], [],
                             frozenset({"t::one", "t::two", "t::three"}), "", 0)
    assert work["already_failing"] == ["t::one", "t::three"]
    assert work["complete"] is True


# == What the analysis could not resolve ==================


def test_the_work_list_carries_how_much_the_analysis_could_not_resolve():
    """The second gate reports the difference; this is the whole of what it read."""
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 7)
    assert work["unresolved"] == 7


def test_the_state_block_says_how_much_the_analysis_could_not_resolve(repo,
                                                                     monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 7)
    note = integrate.make_state(directory, "HEAD", work, lambda: [])[0](1, 20)
    assert ("References the analysis could not resolve in the merged tree in all, as "
            "measured at the last check: 7.") in note
    assert "could not follow to a file at all" not in note
    assert "could not follow to a file at all is in neither count" in integrate.start_text(work)


def test_what_the_analysis_never_read_is_said_to_be_outside_both_counts(repo,
                                                                       monkeypatch):
    """Nothing here can count those, so the state says they are not counted."""
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    note = integrate.make_state(directory, "HEAD", work, lambda: [])[0](1, 20)
    assert "as measured at the last check: 0." in note
    assert "is in neither count" in integrate.start_text(work)


# == Whether a model is called at all =====================


class _StageThatRuns:
    """A stand-in for RLM that records it was built and answers a settled run."""

    built = []

    def __init__(self, *args, **kwargs):
        _StageThatRuns.built.append(kwargs.get("stage"))

    def run(self, **kwargs):
        return {"answer": "done", "history": [], "iters": 1, "refused_completes": 0,
                "error": "", "outcome": "complete"}


def _integration_of(directory, base, monkeypatch, tests):
    """Run run_integration() over the union of the fixture's branch a, which git settles
    on its own, with the analysis stubbed and run_test() answering from a list, oldest
    first."""
    prepared = {"base": base, "merged": integrate.assemble(directory, base, ["a"]),
                "before": _analysis(), "against": _analysis(), "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "TEST_COMMAND", ["run", "them"])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: True)
    answers = list(tests)
    monkeypatch.setattr(edit, "run_test", lambda *a, **k: answers.pop(0))
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatRuns)
    _StageThatRuns.built.clear()
    return integrate.run_integration(prepared, "", "p", "out")


_BROKEN_KEY = ("p/a_py/a.py", "p/b_py/b.py", "old")


def _integration_against(directory, base, monkeypatch, before, against, after):
    """Run run_integration() over the union of the fixture's branch a, with the three
    analyses given and no test command."""
    prepared = {"base": base, "merged": integrate.assemble(directory, base, ["a"]),
                "before": before, "against": against, "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(integrate, "analyse", lambda _: after)
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatRuns)
    _StageThatRuns.built.clear()
    return integrate.run_integration(prepared, "", "p", "out")


def test_a_call_the_branch_merged_into_broke_on_its_own_is_not_the_merges(repo, monkeypatch):
    """It resolves at the start and not at that branch's tip, and the union is no
    worse than the tip."""
    directory, base = repo
    out = _integration_against(directory, base, monkeypatch,
                               before=_analysis(),
                               against=_analysis(no_resolve={_BROKEN_KEY: [2]}),
                               after=_analysis(no_resolve={_BROKEN_KEY: [2]}))
    assert out["called"] is False and out["work"]["broken"] == []


@pytest.fixture
def unmarked_repo(tmp_path):
    """Two branches conflicting where git writes no marker: a UTF-16 file both changed,
    and a file a deleted while b changed it.

    Yields:
        (path, base commit).
    """
    directory = str(tmp_path / "repo")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")

    def write_utf16(text):
        with open(os.path.join(directory, "res.rc"), "w", encoding="utf-16") as handle:
            handle.write(text)

    write_utf16("STRING one\nSTRING two\n")
    _write(directory, "gone.txt", "line 1\nline 2\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "-b", "a", base)
    write_utf16("STRING ONE (a)\nSTRING two\n")
    _git(directory, "rm", "-q", "gone.txt")
    _git(directory, "commit", "-qam", "a")
    _git(directory, "checkout", "-q", "-b", "b", base)
    write_utf16("STRING one (b)\nSTRING two\n")
    _write(directory, "gone.txt", "line 1 changed by b\nline 2\n")
    _git(directory, "commit", "-qam", "b")
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


def test_a_conflict_git_wrote_no_marker_for_calls_a_model(unmarked_repo, monkeypatch):
    """git keeps one side's version of such a file, so ending with no model would drop
    the other side's change."""
    directory, base = unmarked_repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["unmarked"] == ["gone.txt", "res.rc"]
    assert integrate.marker_path_list(directory) == []
    prepared = {"base": base, "merged": merged, "before": _analysis(),
                "against": _analysis(), "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatRuns)
    _StageThatRuns.built.clear()
    out = integrate.run_integration(prepared, "", "p", "out")
    assert out["called"] is True
    assert out["work"]["unmarked"] == ["gone.txt", "res.rc"]
    line_list = integrate._branch_line_list(out["work"])
    assert any(integrate._UNMARKED_NOTE in line and "res.rc" in line for line in line_list)


def test_each_side_s_version_is_read_by_branch(unmarked_repo, monkeypatch):
    directory, base = unmarked_repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {
        "": base, **{e["branch"]: e["commit"] for e in merged["branches"]}}
    assert integrate.read_branch_file("a", "res.rc") == "STRING ONE (a)\nSTRING two\n"
    assert integrate.read_branch_file("b", "res.rc") == "STRING one (b)\nSTRING two\n"
    assert integrate.read_branch_file("", "gone.txt") == "line 1\nline 2\n"
    assert integrate.read_branch_file("a", "gone.txt").startswith("Error: gone.txt does not")
    assert integrate.read_branch_file("c", "res.rc").startswith("Error: 'c' is not")
    assert integrate.read_branch_file("a", "../x").startswith("Error: ../x does not")
    assert integrate.read_branch_file("a", "res.rc\0").startswith("Error: a path holds")
    assert integrate.read_branch_file("a", 3).startswith("Error: a path must be a string")
    tool = next(t for t in integrate.merge_tool_list() if t.__name__ == "read_branch_file")
    tool("b", "gone.txt")
    assert "gone.txt" in integrate._STAGE_DICT["read_dict"]
    assert integrate._no_read_reason_list(["gone.txt"]) == []


def test_a_call_the_merge_broke_is_work_even_where_the_start_did_not_resolve_it(repo,
                                                                                monkeypatch):
    """The branch merged into settled it after the start; the union lost that."""
    directory, base = repo
    out = _integration_against(directory, base, monkeypatch,
                               before=_analysis(no_resolve={_BROKEN_KEY: [2]}),
                               against=_analysis(),
                               after=_analysis(no_resolve={_BROKEN_KEY: [2]},
                                               name={"p/b_py/b.py": ["new"]}))
    assert out["called"] is True
    assert [entry["name"] for entry in out["work"]["broken"]] == ["old"]


def test_a_test_the_merge_broke_is_work_and_a_model_is_called(repo, monkeypatch):
    """Each side alone left the test green; combined they do not. That is not read as
    already failing, and a model is called."""
    directory, base = repo
    out = _integration_of(directory, base, monkeypatch,
                          [_tested("failed", ["t::one"]), _tested("passed"),
                           _tested("passed")])
    assert out["called"] is True and _StageThatRuns.built == ["integrate"]
    assert "t::one" in out["work"]["tests"]
    assert out["work"]["already_failing"] == []


def test_a_test_red_before_the_branches_were_combined_calls_no_model(repo, monkeypatch):
    """Nothing the merge did: red on what it was combined onto, red on the union."""
    directory, base = repo
    out = _integration_of(directory, base, monkeypatch,
                          [_tested("failed", ["t::one"]),
                           _tested("failed", ["t::one"])])
    assert out["called"] is False and _StageThatRuns.built == []
    assert out["work"]["tests"] == ""
    assert out["work"]["already_failing"] == ["t::one"]


def test_a_union_whose_tests_no_longer_run_calls_a_model(repo, monkeypatch):
    """HEAD was green and the union's suite stops before running a test."""
    directory, base = repo
    out = _integration_of(directory, base, monkeypatch,
                          [_tested("did not run"), _tested("passed"), _tested("passed")])
    assert out["called"] is True and _StageThatRuns.built == ["integrate"]
    assert "did not run" in out["work"]["tests"]
    assert out["work"]["already_failing"] == []


def test_a_green_union_calls_no_model_and_measures_head_once_at_most(repo, monkeypatch):
    directory, base = repo
    # The union and HEAD's tree, which is read for the tests it ran
    out = _integration_of(directory, base, monkeypatch, [_tested("passed"), _tested("passed")])
    assert out["called"] is False and out["settled"] is True


def test_a_union_only_the_static_check_finds_wrong_calls_the_model(repo, monkeypatch):
    """The check is measured on HEAD's tree and then on the union: what is new on the
    union is not settled by the machine alone."""
    directory, base = repo
    old = "a.py: F821 Undefined name `x`"
    new = "b.py: F821 Undefined name `helper`"
    seen_list = []

    def run_check():
        seen_list.append(_read(directory, "f.txt"))
        return collections.Counter([old] if len(seen_list) == 1 else [old, new])

    monkeypatch.setattr(config, "CHECK_COMMAND", ["check", "it"])
    monkeypatch.setattr(edit, "run_check", run_check)
    out = _integration_of(directory, base, monkeypatch, [_tested("passed"), _tested("passed")])
    # HEAD's tree first, then the union put back
    assert seen_list == ["one\ntwo\nthree\n", "ONE\ntwo\nthree\n"]
    assert out["called"] is True and _StageThatRuns.built == ["integrate"]
    assert len(out["work"]["checks"]) == 1 and new in out["work"]["checks"][0]
    assert old not in out["work"]["checks"][0]


def test_the_reasons_carry_what_the_check_finds_that_head_did_not(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(config, "CHECK_COMMAND", ["check", "it"])
    monkeypatch.setattr(edit, "run_check",
                        lambda: collections.Counter(["b.py: F821 Undefined name `g`"]))
    line_list = integrate._reason_list(directory, "out", _analysis(), None, None,
                                       check_before=collections.Counter())
    assert line_list == ["- `check it` finds what it did not find before you started: "
                         "b.py: F821 Undefined name `g`"]
    assert integrate._reason_list(directory, "out", _analysis(), None, None) == []


# == what may be changed in a test ==========================
def test_the_merging_stage_is_told_where_to_stop_in_a_test(monkeypatch):
    monkeypatch.setattr(integrate.edit, "test_can_run", lambda: True)
    text = integrate.build_integrate_instruction("py")
    assert "update that assertion to what that side was asked for" in text
    assert "Never weaken or delete a test to make it pass" in text


def test_a_merge_with_no_tests_to_run_is_told_none_of_that(monkeypatch):
    monkeypatch.setattr(integrate.edit, "test_can_run", lambda: False)
    text = integrate.build_integrate_instruction("py")
    assert "Never weaken" not in text


# == the merging stage is a request of its own to the editing tools ==
def test_the_merging_stage_starts_the_editing_tools_over_the_union_with_no_round_limit(
        repo, monkeypatch):
    """What the tools then count as changed by this run is what is settled from the
    union on, and the editing agent's round limit is not enforced on a stage whose
    instructions never state it."""
    directory, base = repo
    seen = []
    monkeypatch.setattr(edit, "reset_patch_round", lambda **k: seen.append(k))
    out = _integration_of(directory, base, monkeypatch, [_tested("passed"), _tested("passed")])
    assert seen == [{"round_limit": 0}]
    assert out["settled"] is True


# == What git refusing a step comes back as ===============


def _git_refuses(monkeypatch, *word_tuple):
    """Make gitcmd.run fail with status 1 when the command holds every word given."""
    real_run = gitcmd.run

    def run(project_dir, *args, **kwargs):
        if all(word in args for word in word_tuple):
            return subprocess.CompletedProcess(list(args), 1, "", "refused by git")
        return real_run(project_dir, *args, **kwargs)

    monkeypatch.setattr(integrate.gitcmd, "run", run)


def test_a_merge_commit_git_will_not_write_is_an_error(repo, monkeypatch):
    directory, base = repo
    _git_refuses(monkeypatch, "commit-tree")
    with pytest.raises(RuntimeError, match="could not commit the merge of a"):
        integrate.assemble(directory, base, ["a"])


def test_a_head_that_cannot_be_moved_is_an_error(repo, monkeypatch):
    directory, base = repo
    _git_refuses(monkeypatch, "checkout", "--detach")
    with pytest.raises(RuntimeError, match="could not start from"):
        integrate.assemble(directory, base, ["b"], onto="a")


def test_a_tree_that_cannot_be_put_in_place_is_an_error(repo, monkeypatch):
    directory, base = repo
    _git_refuses(monkeypatch, "read-tree")
    with pytest.raises(RuntimeError, match="could not put the merged tree"):
        integrate.assemble(directory, base, ["a"])


def test_a_ref_left_behind_with_its_lock_is_not_merged_under(repo):
    directory, base = repo
    _git(directory, "update-ref", "refs/codetwine-loom/merged-so-far", base)
    _write(directory, ".git/refs/codetwine-loom/merged-so-far.lock", "")
    with pytest.raises(RuntimeError, match="could not point refs/codetwine-loom/merged-so-far at"):
        integrate.assemble(directory, base, ["a", "b"])
    assert _git(directory, "status", "--porcelain") == ""


def test_a_side_s_ref_git_will_not_write_is_an_error(repo, monkeypatch):
    directory, base = repo
    _git_refuses(monkeypatch, "update-ref", "refs/codetwine-loom/side/b")
    with pytest.raises(RuntimeError,
                       match="could not point refs/codetwine-loom/side/b at .*: refused by git"):
        integrate.assemble(directory, base, ["a", "b"])


def test_a_union_whose_parents_or_trailers_cannot_be_read_is_an_error(repo, monkeypatch):
    directory, base = repo
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    union = _union_commit(directory, [base, a, b])
    assert integrate.union_side_list(directory, union) == [a, b]
    with monkeypatch.context() as patch:
        _git_refuses(patch, "rev-list", "--parents")
        with pytest.raises(RuntimeError, match="could not read the parents of"):
            integrate.union_side_list(directory, union)
    _git_refuses(monkeypatch, "--format=%B")
    with pytest.raises(RuntimeError, match="could not read the trailers of"):
        integrate.union_side_list(directory, union)


def test_a_start_git_cannot_name_is_an_error_and_no_shared_commit_is_none(repo, monkeypatch):
    directory, base = repo
    a = _git(directory, "rev-parse", "a").strip()
    real_run = gitcmd.run
    status_list = [128]

    def run(project_dir, *args, **kwargs):
        if "merge-base" in args:
            return subprocess.CompletedProcess(list(args), status_list[0], "", "fatal: no")
        return real_run(project_dir, *args, **kwargs)

    monkeypatch.setattr(integrate.gitcmd, "run", run)
    with pytest.raises(RuntimeError, match="could not read where .* starts from: fatal: no"):
        integrate._side_start_dict(directory, "", base, a, None, {})
    status_list[0] = 1
    assert integrate._side_start_dict(directory, "", base, a, None, {}) is None


def _driver_repo(tmp_path, command, attribute_text="*.txt merge=mine\n"):
    """A repository whose branches a and b change one line of f.txt, with a merge
    driver named mine in its git settings that runs command."""
    directory, base = _two_file_repo(
        tmp_path, "f.txt", "ONE\ntwo\n", "f.txt", "1\ntwo\n",
        {"f.txt": "one\ntwo\n", ".gitattributes": attribute_text})
    _git(directory, "config", "merge.mine.driver", command)
    return directory, base


_STAND_IN_BLOCK = ("<<<<<<< refs/codetwine-loom/merged-so-far\nONE\n||||||| base\none\n=======\n1\n"
                   ">>>>>>> refs/codetwine-loom/side/{name}\ntwo\n")


@pytest.mark.parametrize("setting_list, attribute_text, base_label", [
    ([], "*.txt merge=mine\n", "base"), ([("merge.default", "mine")], "", ""),
    ([("merge.text.driver", "exit 0")], "*.txt merge=text\n", "base"),
    ([("merge.text.driver", "exit 0")], "", "")])
def test_a_merge_driver_of_the_user_s_settings_is_not_run(tmp_path, setting_list,
                                                          attribute_text, base_label):
    """git's own merge stands in for it, under the names the merge is made under. A
    file with no merge attribute is merged as it is with no setting."""
    mark = tmp_path / "ran"
    directory, base = _driver_repo(tmp_path, f"touch {mark}", attribute_text)
    for key, value in setting_list:
        _git(directory, "config", key, value)
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert not mark.exists()
    assert merged["conflicted"] == ["f.txt"] and merged["unmarked"] == []
    base_label = base_label or _git(directory, "rev-parse", "--short", base).strip()
    assert _read(directory, "f.txt") == _STAND_IN_BLOCK.format(name="b").replace(
        "||||||| base", f"||||||| {base_label}")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    _write(directory, "f.txt", "ONE 1\ntwo\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_merge_driver_is_not_run_whatever_file_git_config_alone_is_handed(tmp_path,
                                                                            monkeypatch):
    mark = tmp_path / "ran"
    directory, base = _driver_repo(tmp_path, f"touch {mark}")
    (tmp_path / "only.gitconfig").write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG", str(tmp_path / "only.gitconfig"))
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert not mark.exists()
    assert merged["conflicted"] == ["f.txt"]
    assert (tmp_path / "only.gitconfig").read_text(encoding="utf-8") == ""


def test_the_merge_standing_in_for_a_driver_names_a_side_whatever_its_name_holds(tmp_path):
    name = "x'y%Az $(id)"
    directory, base = _driver_repo(tmp_path, "exit 0")
    _git(directory, "config", "merge.my drv=1 %A.driver", "exit 0")
    option_list, env_dict = integrate._own_driver_tuple(directory, "one", name)
    assert option_list == ["--config-env=merge.mine.driver=CTLOOM_MERGE_DRIVER",
                           "--config-env=merge.mine.recursive=CTLOOM_MERGE_RECURSIVE",
                           "--config-env=merge.my drv=1 %A.driver=CTLOOM_MERGE_DRIVER",
                           "--config-env=merge.my drv=1 %A.recursive=CTLOOM_MERGE_RECURSIVE"]
    assert env_dict == {"CTLOOM_MERGE_RECURSIVE": gitcmd.NO_MERGE_DRIVER, "CTLOOM_MERGE_DRIVER": (
        "git merge-file --marker-size=%L -L one -L base -L 'x'\"'\"'y%%Az $(id)' "
        "%A %O %B || exit 1")}
    _git(directory, "branch", "-m", "b", "x'y%Az")
    integrate.assemble(directory, base, ["a", "x'y%Az"])
    assert _read(directory, "f.txt") == _STAND_IN_BLOCK.format(name="x'y%Az")


_FUNCTION_TEXT = "".join(f"def f{n}():\n    return {n}\n\n\n" for n in range(12))


@pytest.fixture(scope="module")
def setting_repo(tmp_path_factory):
    """A repository holding pairs of branches whose merge a git setting can change:
    {pair name: (commit merged onto, commit merged)}, and two merge bases for one pair.

    Yields:
        (path, the pairs).
    """
    directory = str(tmp_path_factory.mktemp("setting") / "p")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    for name in ("old", "many"):
        os.makedirs(os.path.join(directory, name))
    _write(directory, "old/a.py", _FUNCTION_TEXT)
    _write(directory, "c.py", "A = 1\nB = 2\nC = 3\n")
    for n in range(6):
        _write(directory, f"many/m{n}.py", _FUNCTION_TEXT.replace("f", f"m{n}_"))
    _write_raw(directory, b"crlf.txt", b"one\r\ntwo\r\nthree\r\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")

    def branch(name, change, start="main"):
        _git(directory, "checkout", "-q", "-b", name, start)
        change()
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", name)
        return _git(directory, "rev-parse", "HEAD").strip()

    def move_many():
        for n in range(6):
            _git(directory, "mv", f"many/m{n}.py", f"many/n{n}.py")
            _write(directory, f"many/n{n}.py",
                   _FUNCTION_TEXT.replace("f", f"m{n}_") + f"# moved {n}\n")

    def change_many():
        for n in range(6):
            _write(directory, f"many/m{n}.py",
                   f"# head {n}\n" + _FUNCTION_TEXT.replace("f", f"m{n}_"))

    def to_lf():
        _write(directory, ".gitattributes", "*.txt text\n")
        _write_raw(directory, b"crlf.txt", b"one\ntwo\nthree\n")

    def cross(name, start, other, text):
        """A merge of other into start that settles c.py as text, and a commit on it."""
        _git(directory, "checkout", "-q", "-b", name, start)
        _write(directory, "c.py", text)
        _git(directory, "add", "-A")
        tree = _git(directory, "write-tree").strip()
        commit = _git(directory, "commit-tree", tree, "-p", start, "-p", other, "-m",
                      name).strip()
        _git(directory, "reset", "-q", "--hard", commit)
        _write(directory, f"{name}.txt", name)
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", f"{name} more")
        return _git(directory, "rev-parse", "HEAD").strip()

    pair_dict = {
        "content": (branch("c1", lambda: _write(directory, "c.py", "A = 1\nB = 20\nC = 3\n")),
                    branch("c2", lambda: _write(directory, "c.py", "A = 1\nB = 30\nC = 3\n"))),
        "directory": (branch("d1", lambda: _git(directory, "mv", "old", "new")),
                      branch("d2", lambda: _write(directory, "old/c.py", "NEW = 1\n"))),
        "rename": (branch("m1", move_many), branch("m2", change_many)),
        "line ending": (branch("n1", to_lf), branch("n2", lambda: _write_raw(
            directory, b"crlf.txt", b"one\r\nTWO\r\nthree\r\n")))}
    k1 = branch("k1", lambda: _write(directory, "c.py", "A = 1\nB = 21\nC = 3\n"))
    k2 = branch("k2", lambda: _write(directory, "c.py", "A = 1\nB = 22\nC = 3\n"))
    pair_dict["two merge bases"] = (cross("x1", k1, k2, "A = 1\nB = 21\nC = 3\nD = 1\n"),
                                    cross("x2", k2, k1, "A = 1\nB = 22\nC = 3\nE = 2\n"))
    _git(directory, "checkout", "-q", "main")
    return directory, pair_dict


@pytest.mark.parametrize("setting_list", [
    [("merge.default", "union")], [("merge.default", "binary")],
    [("merge.default", "union"), ("merge.union.driver", "exit 0")],
    [("merge.text.driver", "exit 0")], [("merge.directoryRenames", "true")],
    [("merge.directoryRenames", "false")], [("merge.renames", "false")],
    [("diff.renames", "false")], [("merge.renameLimit", "1")], [("diff.renameLimit", "1")],
    [("merge.renormalize", "true")], [("merge.verbosity", "5")], [("merge.verbosity", "0")],
    [("merge.conflictStyle", "merge")], [("diff.algorithm", "patience")]])
def test_a_merge_made_here_is_the_same_whatever_the_git_settings_say(setting_repo,
                                                                    setting_list):
    directory, pair_dict = setting_repo
    plain_dict = {name: integrate._merge_tree(directory, onto, commit)
                  for name, (onto, commit) in pair_dict.items()}
    assert {name: bool(merge[1]) for name, merge in plain_dict.items()} == {
        "content": True, "directory": True, "rename": False, "line ending": True,
        "two merge bases": True}
    for key, value in setting_list:
        _git(directory, "config", key, value)
    try:
        assert {name: integrate._merge_tree(directory, onto, commit)
                for name, (onto, commit) in pair_dict.items()} == plain_dict
    finally:
        for key, _ in setting_list:
            _git(directory, "config", "--unset", key)


def test_a_merge_driver_merges_common_ancestors_as_git_does_with_no_setting(tmp_path):
    """The driver a user's merge driver names for the merge of two merge bases is not
    the one that merge is made with."""
    directory, base = _driver_repo(tmp_path, "exit 0")
    tree_list = []
    for value in ("", "binary", "union"):
        if value:
            _git(directory, "config", "merge.mine.recursive", value)
        tree_list.append(integrate._merge_tree(directory, "a", "b")[0])
        option_list, env_dict = integrate._own_driver_tuple(directory, "a", "b")
        assert "--config-env=merge.mine.recursive=CTLOOM_MERGE_RECURSIVE" in option_list
        assert env_dict["CTLOOM_MERGE_RECURSIVE"] == gitcmd.NO_MERGE_DRIVER
    assert len(set(tree_list)) == 1


def test_settings_that_define_no_merge_driver_change_nothing_of_the_merge(repo, monkeypatch):
    directory, base = repo
    assert integrate._own_driver_tuple(directory, "a", "b") == ([], {})
    monkeypatch.setattr(integrate.gitcmd, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 128, b"", b"fatal: bad config line 1"))
    with pytest.raises(RuntimeError, match="could not read the drivers .*bad config"):
        integrate._merge_tree(directory, "a", "b")


def test_a_filter_driver_of_the_user_s_settings_is_not_run_by_a_merge(tmp_path):
    """merge.renormalize passes every side through the filters of the settings."""
    mark = tmp_path / "ran"
    directory, base = _driver_repo(tmp_path, "exit 0", "*.txt filter=mine\n")
    _git(directory, "config", "--unset", "merge.mine.driver")
    for key, value in (("merge.renormalize", "true"), ("filter.mine.required", "true"),
                       ("filter.mine.clean", f"sh -c 'touch {mark}; cat'"),
                       ("filter.mine.smudge", f"sh -c 'touch {mark}; cat'"),
                       ("filter.other one.process", f"sh -c 'touch {mark}'")):
        _git(directory, "config", key, value)
    option_list, env_dict = integrate._own_driver_tuple(directory, "a", "b")
    assert option_list == [
        f"--config-env=filter.{name}.{key}={env}"
        for name in ("mine", "other one")
        for key, env in (("clean", "CTLOOM_NO_DRIVER"), ("smudge", "CTLOOM_NO_DRIVER"),
                         ("process", "CTLOOM_NO_DRIVER"), ("required", "CTLOOM_NO_REQUIRE"))]
    assert env_dict == {"CTLOOM_NO_DRIVER": "", "CTLOOM_NO_REQUIRE": "false"}
    mark.unlink(missing_ok=True)
    assert integrate._merge_tree(directory, "a", "b")[1] == ["f.txt"]
    assert not mark.exists()
    proc = integrate.gitcmd.run(directory, "merge-tree", "--write-tree", "a", "b")
    assert proc.returncode == 1 and mark.exists()


def test_a_repository_with_no_commit_cannot_be_prepared(tmp_path):
    directory = str(tmp_path / "empty")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    with pytest.raises(RuntimeError, match="not a git repository with a commit"):
        integrate.prepare(directory, "", ["a"])


# == Reading one state out of its analysis ================


def test_the_analysis_is_read_as_the_names_each_file_defines(tmp_path):
    from conftest import write_knowledge_db
    path = str(tmp_path / "k.sqlite")
    write_knowledge_db({"files": [
        {"file": "p/a_py/a.py",
         "file_dependencies": {
             "definitions": [{"name": "f", "type": "function_definition",
                              "start_line": 1, "end_line": 2, "context": "def f():"},
                             {"name": "g", "type": "function_definition",
                              "start_line": 3, "end_line": 4, "context": "def g():"}],
             "callee_usages": [], "caller_usages": []},
         "doc": {"sections": [], "summary": ""}}]}, path, "p")
    analysis = integrate.analyse(path)
    assert analysis["no_resolve"] == {} and analysis["name"] == {"p/a_py/a.py": ["f", "g"]}
    assert analysis["range"] == {"p/a_py/a.py": [(1, 2, "f"), (3, 4, "g")]}
    assert analysis["ref"] == []


def _two_run_knowledge(tmp_path, usage_list):
    """Write an analysis of calc.py, whose classes A and B each define run, and of a
    test file that uses calc.py as usage_list says."""
    from conftest import write_knowledge_db
    path = str(tmp_path / "k.sqlite")
    definition_list = [
        {"name": "A", "type": "class_definition", "start_line": 1, "end_line": 3,
         "context": "class A:\n    def run(self, a):\n        return a"},
        {"name": "run", "type": "function_definition", "start_line": 2, "end_line": 3,
         "context": "    def run(self, a):\n        return a"},
        {"name": "B", "type": "class_definition", "start_line": 6, "end_line": 8,
         "context": "class B:\n    def run(self, b, c):\n        return b + c"},
        {"name": "run", "type": "function_definition", "start_line": 7, "end_line": 8,
         "context": "    def run(self, b, c):\n        return b + c"}]
    write_knowledge_db({"files": [
        {"file": "p/calc_py/calc.py",
         "file_dependencies": {"definitions": definition_list, "callee_usages": [],
                               "caller_usages": []},
         "doc": {"sections": [], "summary": ""}},
        {"file": "p/tests/test_a_py/test_a.py",
         "file_dependencies": {"definitions": [], "callee_usages": usage_list,
                               "caller_usages": []},
         "doc": {"sections": [], "summary": ""}}]}, path, "p")
    return path


def test_two_methods_of_one_name_in_one_file_are_two_names(tmp_path):
    path = _two_run_knowledge(tmp_path, [
        {"from": "p/calc_py/calc.py", "name": "A", "lines": [5],
         "target_name": "A", "target_start_line": 1},
        {"from": "p/calc_py/calc.py", "name": "B.run", "lines": [6],
         "target_name": "run", "target_start_line": 7},
        {"from": "p/calc_py/calc.py", "name": "a.run", "lines": [7],
         "target_name": "run", "target_start_line": 2}])
    analysis = integrate.analyse(path)
    assert analysis["name"]["p/calc_py/calc.py"] == ["A", "A.run", "B", "B.run"]
    assert analysis["range"]["p/calc_py/calc.py"] == [
        (1, 3, "A"), (2, 3, "A.run"), (6, 8, "B"), (7, 8, "B.run")]
    assert analysis["signature"]["p/calc_py/calc.py"]["A.run"] == "    def run(self, a):"
    assert analysis["signature"]["p/calc_py/calc.py"]["B.run"] == \
        "    def run(self, b, c):"
    # A usage is read as the one of the two the analysis names, whatever name it is written with
    assert sorted((ref[2], ref[3]) for ref in analysis["ref"]) == [
        ("A", [5]), ("A.run", [7]), ("B.run", [6])]


def test_two_branches_that_change_two_methods_of_one_name_do_not_overlap(tmp_path):
    analysis = integrate.analyse(_two_run_knowledge(tmp_path, []))
    side_list = [(branch, integrate.side_dict_of(
        {"add": [], "delete": [], "line": {"calc.py": [(line, line)]}}, analysis["range"],
        "p")) for branch, line in (("a", 3), ("b", 8))]
    assert side_list[0][1] == {"p/calc_py/calc.py": {"A.run"}}
    assert integrate.overlap_entry_list(analysis, analysis, side_list, "p") == []


@pytest.mark.parametrize("entry_list, expect", [
    ([(1, 9, "Outer"), (2, 8, "Mid"), (3, 4, "f")], ["Outer", "Outer.Mid", "Outer.Mid.f"]),
    # Two definitions on the same lines are neither inside the other
    ([(1, 5, "T"), (1, 5, "S"), (2, 3, "f")], ["S", "T", "T.f"]),
    ([(1, 2, "f"), (4, 5, "g")], ["f", "g"])])
def test_a_definition_is_named_with_the_definitions_around_it(entry_list, expect):
    def_dict = {f"p/a_py/a.py::{name}#{n}": {"name": name, "file": "p/a_py/a.py",
                                             "lines": [start, end]}
                for n, (start, end, name) in enumerate(entry_list)}
    assert sorted(integrate.qual_name_dict(def_dict).values()) == sorted(expect)


def test_a_cobol_name_is_read_in_one_case_whatever_case_it_is_written_in(tmp_path):
    """Two states that write one paragraph in two cases define the same name."""
    from conftest import write_knowledge_db
    name_list = []
    for state, written in (("a", "Load-Cust"), ("b", "LOAD-CUST")):
        path = str(tmp_path / f"{state}.sqlite")
        write_knowledge_db({"files": [
            {"file": "p/src/MAIN_cbl/MAIN.cbl",
             "file_dependencies": {
                 "definitions": [{"name": written, "type": "paragraph_header",
                                  "start_line": 3, "end_line": 4,
                                  "context": f"       {written}."}],
                 "callee_usages": [], "caller_usages": []},
             "doc": {"sections": [], "summary": ""}}]}, path, "p")
        analysis = integrate.analyse(path)
        name_list.append(analysis["name"])
        assert analysis["signature"] == {"p/src/MAIN_cbl/MAIN.cbl": {
            "LOAD-CUST": f"       {written}."}}
    assert name_list[0] == name_list[1] == {"p/src/MAIN_cbl/MAIN.cbl": ["LOAD-CUST"]}


def test_a_use_under_another_name_is_a_use_of_the_file_though_it_is_no_reference(tmp_path):
    """`from a import f as other` is recorded under the name it was given, which no
    definition has, so it resolves to nothing and is still a use of a.py."""
    from conftest import write_knowledge_db
    path = str(tmp_path / "k.sqlite")
    definition = {"name": "f", "type": "function_definition", "start_line": 1,
                  "end_line": 2, "context": "def f():"}
    write_knowledge_db({"files": [
        {"file": "p/a_py/a.py",
         "file_dependencies": {"definitions": [definition], "callee_usages": [],
                               "caller_usages": []},
         "doc": {"sections": [], "summary": ""}},
        {"file": "p/tests/test_a_py/test_a.py",
         "file_dependencies": {
             "definitions": [],
             "callee_usages": [{"from": "p/a_py/a.py", "name": "other", "lines": [3],
                                "target_name": None, "target_start_line": None},
                               {"from": "p/a_py/a.py", "name": "f", "lines": [4],
                                "target_name": "f", "target_start_line": 1}],
             "caller_usages": []},
         "doc": {"sections": [], "summary": ""}}]}, path, "p")
    analysis = integrate.analyse(path)
    assert [ref[2] for ref in analysis["ref"]] == ["f"]
    assert analysis["use"] == [("p/tests/test_a_py/test_a.py", "p/a_py/a.py")]


def test_a_call_the_merge_broke_is_a_reason_the_merge_is_not_settled(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("k", directory, "p", False))
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis(name={"p/b_py/b.py": ["new"]}))
    monkeypatch.setattr(integrate, "broken_ref_dict",
                        lambda before, after: {("p/a_py/a.py", "p/b_py/b.py", "old"): [2]})
    reason_list = integrate._reason_list(directory, "out", _analysis(), None, None)
    assert reason_list == ["- these calls still do not resolve: old in a.py"]


def test_the_state_block_lists_what_each_branch_left_unsettled(monkeypatch):
    work = integrate.build_work_dict(_merged(conflicted=["f.txt"]), ["f.txt"],
                                     [{"file": "a.py", "lines": [2], "name": "old",
                                       "defined_in": "b.py", "now_defines": ["new"]}],
                                     frozenset(), "", 0)
    start = integrate.start_text(work)
    assert "git could not settle: f.txt" in start
    assert "old in a.py — b.py now defines new" in start
    text = "\n".join(integrate._state_line_list(work, 1, 5))
    assert "Conflict markers in the working tree, as of this turn:\n  f.txt" in text
    assert "git could not settle" not in text and "now defines" not in text
    assert "  old in a.py — nothing yet" in text


def test_a_broken_call_whose_file_now_defines_nothing_says_so(monkeypatch):
    work = integrate.build_work_dict(_merged(), [],
                                     [{"file": "a.py", "lines": [2], "name": "old",
                                       "defined_in": "b.py", "now_defines": []}],
                                     frozenset(), "", 0)
    assert "b.py now defines nothing" in integrate.start_text(work)


# == A merging stage that ran out of turns ================


class _StageThatRunsOut:
    """A stand-in stage that uses every turn without handing anything over."""

    stage = "integrate"

    def __init__(self, *args, **kwargs):
        pass

    def run(self, **kwargs):
        return {"answer": None, "history": [("code", "output")], "iters": 3,
                "refused_completes": 0, "error": "", "outcome": None}


def test_a_merge_that_ran_out_of_turns_is_asked_once_what_it_settled(repo, monkeypatch):
    directory, base = repo
    prepared = {"base": base, "merged": integrate.assemble(directory, base, ["a", "b"]),
                "before": _analysis(), "against": _analysis(), "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatRunsOut)
    monkeypatch.setattr(rlm_mod, "late_answer",
                        lambda *a, **k: "f.txt was settled by hand")
    out = integrate.run_integration(prepared, "", "p", "out")
    assert out["harvested"] is True and out["answer"] == "f.txt was settled by hand"
    assert out["called"] is True


def test_a_merge_whose_one_extra_call_fails_is_still_checked_and_judged(repo,
                                                                       monkeypatch):
    directory, base = repo
    prepared = {"base": base, "merged": integrate.assemble(directory, base, ["a", "b"]),
                "before": _analysis(), "against": _analysis(), "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatRunsOut)

    def fail(*args, **kwargs):
        raise RuntimeError("503 from the provider")

    monkeypatch.setattr(rlm_mod, "late_answer", fail)
    out = integrate.run_integration(prepared, "", "p", "out")
    assert out["answer"] is None and out["harvested"] is False
    assert out["error"] == "RuntimeError: 503 from the provider"
    assert out["called"] is True and out["settled"] is False and out["left"]
    assert integrate.end_of(out) == integrate.END_ERROR


# == The definitions more than one side's change reaches ==


def _side_repo(tmp_path):
    """A repository whose base holds x.py with f (lines 1-4) and g (lines 5-8).

    Branch a rewrites line 2 (in f); branch b rewrites line 7 (in g) and adds y.py.
    The two never touch the same line, so git merges them without a conflict.

    Returns:
        (path, base commit).
    """
    directory = str(tmp_path / "two_file")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    _write(directory, "x.py", "def f():\n    a = 1\n    b = 2\n    return a\n"
                              "def g():\n    c = 3\n    d = 4\n    return c\n")
    _write(directory, "z.py", "gone = 1\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "-b", "a", base)
    _write(directory, "x.py", "def f():\n    a = 10\n    b = 2\n    return a\n"
                              "def g():\n    c = 3\n    d = 4\n    return c\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "a")
    _git(directory, "checkout", "-q", "-b", "b", base)
    _write(directory, "x.py", "def f():\n    a = 1\n    b = 2\n    return a\n"
                              "def g():\n    c = 3\n    d = 40\n    return c\n")
    _write(directory, "y.py", "from x import f\nf()\n")
    _git(directory, "rm", "-q", "z.py")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "b")
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


_X_RANGE = {"p/x_py/x.py": [(1, 4, "f"), (5, 8, "g")]}


def test_the_lines_a_branch_changed_are_read_in_the_bases_numbering(tmp_path):
    directory, base = _side_repo(tmp_path)
    change = integrate.change_dict_of(directory, base, "a")
    assert change == {"add": [], "delete": [], "line": {"x.py": [(2, 2)]}}
    change = integrate.change_dict_of(directory, base, "b")
    assert change == {"add": ["y.py"], "delete": ["z.py"], "line": {"x.py": [(7, 7)]}}


@pytest.mark.parametrize("key, value", [
    ("diff.noprefix", "true"), ("diff.mnemonicPrefix", "true"),
    ("diff.external", "/bin/echo"), ("color.diff", "always"),
    ("diff.context", "5"), ("diff.interHunkContext", "9"), ("diff.renames", "copies"),
    ("diff.suppressBlankEmpty", "true"), ("diff.ignoreSubmodules", "all")])
def test_the_lines_a_branch_changed_are_read_whatever_the_user_s_diff_settings(
        tmp_path, key, value):
    directory, base = _side_repo(tmp_path)
    _git(directory, "config", key, value)
    assert integrate.change_dict_of(directory, base, "b") == {
        "add": ["y.py"], "delete": ["z.py"], "line": {"x.py": [(7, 7)]}}


def test_a_hunk_that_only_inserts_names_the_line_it_follows_and_the_next(tmp_path):
    directory, base = _side_repo(tmp_path)
    _git(directory, "checkout", "-q", "-b", "c", base)
    _write(directory, "x.py", "def f():\n    a = 1\n    b = 2\n    return a\n"
                              "def g():\n    c = 3\n    e = 5\n    d = 4\n    return c\n")
    _git(directory, "commit", "-qam", "c")
    assert integrate.change_dict_of(directory, base, "c")["line"] == {"x.py": [(6, 7)]}


def test_a_path_git_quotes_in_the_patch_is_read_as_itself(tmp_path):
    directory, base = _side_repo(tmp_path)
    name_tuple = ("space name.py", 'q"t.py', "tab\tname.py", 'a "b.py')
    _git(directory, "checkout", "-q", "-b", "d", base)
    for name in name_tuple:
        _write(directory, name, "one\n")
    _write(directory, "gone me.py", "one\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "names")
    middle = _git(directory, "rev-parse", "HEAD").strip()
    for name in name_tuple:
        _write(directory, name, "two\n")
    _write(directory, "new one.py", "one\n")
    _git(directory, "rm", "-q", "gone me.py")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "change")
    change = integrate.change_dict_of(directory, middle, "d")
    assert change == {"add": ["new one.py"], "delete": ["gone me.py"],
                      "line": {name: [(1, 1)] for name in name_tuple}}


def test_a_changed_line_that_looks_like_a_path_line_is_read_as_a_line(tmp_path):
    directory, base = _side_repo(tmp_path)
    _git(directory, "checkout", "-q", "-b", "e", base)
    _write(directory, "q.sql", "-- one\nselect 1;\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "sql")
    middle = _git(directory, "rev-parse", "HEAD").strip()
    _write(directory, "q.sql", "++ two\nselect 1;\n")
    _git(directory, "commit", "-qam", "sql 2")
    assert integrate.change_dict_of(directory, middle, "e") == {
        "add": [], "delete": [], "line": {"q.sql": [(1, 1)]}}


def test_a_diff_that_cannot_be_read_is_an_error(tmp_path):
    directory, base = _side_repo(tmp_path)
    with pytest.raises(RuntimeError):
        integrate.change_dict_of(directory, base, "no-such-branch")


def test_a_sides_lines_are_mapped_to_the_narrowest_base_definition():
    side = integrate.side_dict_of({"add": [], "delete": [], "line": {"x.py": [(2, 2), (7, 7)]}},
                                  _X_RANGE, "p")
    assert side == {"p/x_py/x.py": {"f", "g"}}


def test_a_line_outside_every_definition_is_the_top_level():
    side = integrate.side_dict_of({"add": [], "delete": [], "line": {"x.py": [(9, 10)]}},
                                  _X_RANGE, "p")
    assert side == {"p/x_py/x.py": {""}}


def test_an_added_file_is_the_top_level_and_a_removed_one_every_name():
    side = integrate.side_dict_of({"add": ["y.py"], "delete": ["x.py"], "line": {}},
                                  _X_RANGE, "p")
    assert side == {"p/y_py/y.py": {""}, "p/x_py/x.py": {"", "f", "g"}}


def _before(**kw):
    return _analysis(range_dict=dict(_X_RANGE), name={"p/x_py/x.py": ["f", "g"]}, **kw)


def test_a_definition_two_sides_changed_is_named_with_both():
    side_list = [("a", {"p/x_py/x.py": {"f"}}), ("b", {"p/x_py/x.py": {"f", "g"}})]
    assert integrate.overlap_entry_list(_before(), _before(), side_list, "p") == [
        {"kind": "both", "file": "x.py", "name": "f", "branches": ["a", "b"]}]


def test_two_sides_changing_different_definitions_of_one_file_are_not_named():
    side_list = [("a", {"p/x_py/x.py": {"f"}}), ("b", {"p/x_py/x.py": {"g"}})]
    assert integrate.overlap_entry_list(_before(), _before(), side_list, "p") == []


def test_a_definition_one_side_changed_and_another_sides_change_uses_is_named():
    """a changed f; b changed h in y.py, and h calls f."""
    after = _analysis(range_dict={**_X_RANGE, "p/y_py/y.py": [(1, 3, "h")]},
                      name={"p/x_py/x.py": ["f", "g"], "p/y_py/y.py": ["h"]},
                      ref=[("p/y_py/y.py", "p/x_py/x.py", "f", [2])])
    before = _analysis(range_dict={**_X_RANGE, "p/y_py/y.py": [(1, 3, "h")]},
                       name={"p/x_py/x.py": ["f", "g"], "p/y_py/y.py": ["h"]})
    side_list = [("a", {"p/x_py/x.py": {"f"}}), ("b", {"p/y_py/y.py": {"h"}})]
    assert integrate.overlap_entry_list(before, after, side_list, "p") == [
        {"kind": "use", "file": "x.py", "name": "f", "changed_by": "a",
         "used_in": "y.py", "lines": [2], "used_by": ["b"]}]


def test_a_use_in_a_definition_the_other_side_left_alone_is_not_named():
    after = _analysis(range_dict={**_X_RANGE, "p/y_py/y.py": [(1, 3, "h"), (4, 6, "k")]},
                      name={"p/x_py/x.py": ["f", "g"], "p/y_py/y.py": ["h", "k"]},
                      ref=[("p/y_py/y.py", "p/x_py/x.py", "f", [2])])
    before = _analysis(range_dict=after["range"], name=after["name"])
    side_list = [("a", {"p/x_py/x.py": {"f"}}), ("b", {"p/y_py/y.py": {"k"}})]
    assert integrate.overlap_entry_list(before, after, side_list, "p") == []


def test_a_use_in_a_file_the_other_side_added_is_named():
    after = _analysis(range_dict={**_X_RANGE, "p/y_py/y.py": [(1, 3, "h")]},
                      name={"p/x_py/x.py": ["f", "g"], "p/y_py/y.py": ["h"]},
                      ref=[("p/y_py/y.py", "p/x_py/x.py", "f", [2])])
    side_list = [("a", {"p/x_py/x.py": {"f"}}), ("b", {"p/y_py/y.py": {""}})]
    assert integrate.overlap_entry_list(_before(), after, side_list, "p") == [
        {"kind": "use", "file": "x.py", "name": "f", "changed_by": "a",
         "used_in": "y.py", "lines": [2], "used_by": ["b"]}]


def test_a_use_in_a_definition_the_other_side_added_inside_an_old_one_is_named():
    """b added method m inside class C; the change lands on C in the base's numbering."""
    after = _analysis(range_dict={**_X_RANGE, "p/y_py/y.py": [(1, 9, "C"), (5, 7, "m")]},
                      name={"p/x_py/x.py": ["f", "g"], "p/y_py/y.py": ["C", "m"]},
                      ref=[("p/y_py/y.py", "p/x_py/x.py", "f", [6])])
    before = _analysis(range_dict={**_X_RANGE, "p/y_py/y.py": [(1, 6, "C")]},
                       name={"p/x_py/x.py": ["f", "g"], "p/y_py/y.py": ["C"]})
    side_list = [("a", {"p/x_py/x.py": {"f"}}), ("b", {"p/y_py/y.py": {"C"}})]
    assert [e["kind"] for e in integrate.overlap_entry_list(before, after, side_list, "p")] \
        == ["use"]


def test_a_side_that_changed_the_definition_and_uses_it_is_not_named_twice():
    after = _analysis(range_dict={**_X_RANGE, "p/y_py/y.py": [(1, 3, "h")]},
                      name={"p/x_py/x.py": ["f", "g"], "p/y_py/y.py": ["h"]},
                      ref=[("p/y_py/y.py", "p/x_py/x.py", "f", [2])])
    before = _analysis(range_dict=after["range"], name=after["name"])
    side_list = [("a", {"p/x_py/x.py": {"f"}, "p/y_py/y.py": {"h"}})]
    assert integrate.overlap_entry_list(before, after, side_list, "p") == []


def test_the_work_list_cuts_the_overlap_to_the_same_count(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 1)
    overlap = [{"kind": "both", "file": "x.py", "name": n, "branches": ["a", "b"]}
               for n in ("f", "g")]
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0, overlap)
    assert len(work["overlap"]) == 1 and work["complete"] is False


def test_the_state_block_names_the_definitions_more_than_one_side_reached():
    overlap = [{"kind": "both", "file": "x.py", "name": "f", "branches": ["a", "b"]},
               {"kind": "use", "file": "x.py", "name": "g", "changed_by": "a",
                "used_in": "y.py", "lines": [2, 5], "used_by": ["b"]}]
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0, overlap)
    start = integrate.start_text(work)
    assert "f in x.py — changed by a, b\n" in start
    assert "g in x.py — changed by a; used at y.py line(s) 2, 5, which b changed" in start
    assert "The machine does not check that the combined code" in start
    text = "\n".join(integrate._state_line_list(work, 1, 5))
    assert "  f in x.py — nothing yet\n  g in x.py — nothing yet" in text
    assert "changed by a" not in text


def test_the_state_says_what_this_stage_has_read_and_changed_at_each_start_entry():
    """The overlap and the broken calls are measured before the first turn, so the
    state has a line for each saying which of its files this stage has already read
    and what it changed in them."""
    overlap = [{"kind": "both", "file": "x.py", "name": "f", "branches": ["a", "b"]},
               {"kind": "use", "file": "x.py", "name": "g", "changed_by": "a",
                "used_in": "y.py", "lines": [2], "used_by": ["b"]},
               {"kind": "both", "file": "z.py", "name": "h", "branches": ["a", "b"]}]
    broken = [{"file": "y.py", "lines": [4], "name": "old", "defined_in": "x.py",
               "now_defines": ["new"]},
              {"file": "w.py", "lines": [1], "name": "old", "defined_in": "z.py",
               "now_defines": []}]
    work = integrate.build_work_dict(_merged(), [], broken, frozenset(), "", 0, overlap)
    work["changed"] = [{"path": "x.py", "insertions": 3, "deletions": 1, "new": False,
                        "defs": ["f", "k"]},
                       {"path": "y.py", "insertions": 0, "deletions": 0, "new": True,
                        "defs": []},
                       {"path": "v.py", "insertions": 1, "deletions": 0, "new": False,
                        "defs": []}]
    work["read"] = [{"path": "v.py", "turn": 1}, {"path": "y.py", "turn": 3}]
    text = "\n".join(integrate._state_line_list(work, 4, 5))
    assert ('What this stage has read and changed in the files of the calls and '
            'definitions named under "At the start", as of this turn:\n'
            "  old in y.py — read y.py on turn 3; changed y.py (new file); "
            "changed f, k in x.py\n"
            "  old in w.py — nothing yet\n"
            "  f in x.py — changed f, k in x.py\n"
            "  g in x.py — changed f, k in x.py; read y.py on turn 3; "
            "changed y.py (new file)\n"
            "  h in z.py — nothing yet\n") in text
    assert "  x.py  +3 -1 (f, k)" in text and "  v.py  +1 -0\n" in text
    assert "read since the branches were combined, as of this turn:\n  v.py  on turn 1\n  y.py  on turn 3" in text
    start = integrate.start_text(work)
    assert start.endswith("  h in z.py — changed by a, b")
    assert "old in w.py — z.py now defines nothing\n" in start
    assert "this stage" not in start.split("The machine does not check")[0]
    assert "The machine does not check that the combined code" in start
    assert "Nobody has checked" not in start + text


def test_the_reading_tools_record_the_file_and_the_turn(monkeypatch, tmp_path):
    """What the stage has read is a machine fact, not something the model has to
    record; the reading tools keep their own name and docstring."""
    def read_source_file(path, start_line=1, end_line=None):
        """Return a file."""
        return "text"

    def read_definition(name, file=None):
        return {"ok": name != "gone", "file": "d.py"}

    def get_use_of_file(target_file):
        return []

    monkeypatch.setattr(integrate.source, "read_source_file", read_source_file)
    monkeypatch.setattr(integrate.source, "read_definition", read_definition)
    monkeypatch.setattr(integrate.source, "get_use_of_file", get_use_of_file)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    (tmp_path / "u.py").write_text("x = 1\n", encoding="utf-8")
    integrate.reset_stage()
    handed = {tool.__name__: tool for tool in integrate.integrate_tool_list()}
    assert handed["read_source_file"].__doc__ == "Return a file."
    integrate._STAGE_DICT["turn"] = 2
    assert handed["read_source_file"]("./a.py") == "text"
    handed["read_definition"]("gone")
    handed["get_use_of_file"](target_file="u.py")
    integrate._STAGE_DICT["turn"] = 3
    handed["read_source_file"](path="a.py", start_line=2)
    handed["read_definition"]("f", "d.py")
    assert integrate.read_list() == [{"path": "a.py", "turn": 3}, {"path": "d.py", "turn": 3},
                                     {"path": "u.py", "turn": 2}]
    integrate.reset_stage()
    assert integrate.read_list() == [] and integrate._STAGE_DICT["turn"] == 0


def test_a_read_that_failed_is_not_recorded(monkeypatch, tmp_path):
    def get_use_of_file(target_file):
        return []

    monkeypatch.setattr(integrate.source, "get_use_of_file", get_use_of_file)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    integrate.reset_stage()
    handed = {tool.__name__: tool for tool in integrate.merge_tool_list()}
    integrate.set_turn(1)
    handed["read_source_file"]("gone.py")
    handed["read_source_file"]("a.py", 99)
    handed["read_branch_file"]("no such branch", "a.py")
    handed["get_use_of_file"]("gone.py")
    handed["get_use_of_file"]("../outside.py")
    assert integrate.read_list() == []
    handed["get_use_of_file"]("a.py")
    assert integrate.read_list() == [{"path": "a.py", "turn": 1}]
    integrate.reset_stage()


def test_a_file_whose_text_starts_as_an_error_is_read(monkeypatch, tmp_path):
    """Whether a call read the file is said by the tool apart from its answer, so a
    file whose text starts the way an error does is read, whole or by its lines."""
    directory = str(tmp_path / "repo")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _write(directory, "msg.txt", "Error: bad input\nline 2\n")
    _git(directory, "add", "-A")
    _git(directory, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    head = _git(directory, "rev-parse", "HEAD").strip()
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {"": head, "b": head}
    handed = {tool.__name__: tool for tool in integrate.merge_tool_list()}
    integrate.set_turn(1)
    assert handed["read_source_file"]("msg.txt") == "Error: bad input\nline 2\n"
    assert handed["read_branch_file"]("b", "msg.txt") == "Error: bad input\nline 2\n"
    assert integrate.read_list() == [{"path": "msg.txt", "turn": 1}]
    assert integrate._no_branch_read_reason_list([("b", "msg.txt", "msg.txt")]) == []
    assert handed["read_branch_file"]("", "msg.txt", 1, 1).startswith("# msg.txt: lines 1-1")
    assert integrate._STAGE_DICT["branch_read_set"] == {("b", "msg.txt"), ("", "msg.txt")}
    handed["read_branch_file"]("b", "msg.txt", 9)
    handed["read_branch_file"]("b", "gone.txt")
    assert integrate._STAGE_DICT["branch_read_set"] == {("b", "msg.txt"), ("", "msg.txt")}
    integrate.reset_stage()


@pytest.mark.parametrize("name, args, kwargs", [
    ("read_source_file", ("a.py",), {"start": 3}),
    ("read_source_file", ("a.py", 1, 2, 3, 4), {}),
    ("read_source_file", (), {}),
    ("read_source_file", ("a.py",), {"path": "b.py"}),
    ("read_branch_file", ("", "a.py"), {"file": "a.py"}),
    ("read_branch_file", ("", "a.py", 1, 2, 3), {}),
    ("read_branch_file", (), {})])
def test_a_wrapped_reading_tool_given_arguments_it_does_not_take_says_so_by_its_name(
        monkeypatch, tmp_path, name, args, kwargs):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    integrate.reset_stage()
    plain = {"read_source_file": integrate.source.read_source_file,
             "read_branch_file": integrate.read_branch_file}[name]
    with pytest.raises(TypeError) as plain_error:
        plain(*args, **kwargs)
    tool_list_list = [integrate.merge_tool_list(),
                      integrate.read_record_tool_list([integrate.source.read_source_file])]
    for tool_list in tool_list_list:
        handed = {tool.__name__: tool for tool in tool_list}
        if name not in handed:
            continue
        with pytest.raises(TypeError) as error:
            handed[name](*args, **kwargs)
        assert str(error.value) == str(plain_error.value)
        assert str(error.value).startswith(f"{name}() ")
    assert integrate.read_list() == []
    integrate.reset_stage()


def test_emptying_the_read_record_keeps_what_the_stage_recorded():
    integrate.reset_stage()
    integrate.record_decision("kept both", "neither side dropped a line")
    integrate.set_turn(4)
    integrate._STAGE_DICT["read_dict"]["a.py"] = 4
    integrate.reset_read()
    assert integrate.read_list() == [] and integrate.stage_turn() == 0
    assert [entry_dict["what"] for entry_dict in integrate.decision_list()] == ["kept both"]
    integrate.reset_stage()


def test_the_state_block_reads_the_turn_a_file_was_read_on(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    integrate.reset_stage()
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    turn_note, _watched = integrate.make_state(directory, "HEAD", work, lambda: [])
    assert "read since the branches were combined, as of this turn: none" in turn_note(1, 20)
    integrate._STAGE_DICT["read_dict"]["f.txt"] = integrate._STAGE_DICT["turn"]
    note = turn_note(2, 20)
    assert "  f.txt  on turn 1" in note and work["read"] == [{"path": "f.txt", "turn": 1}]


def test_what_this_stage_changed_is_named_by_the_unions_definitions(repo, monkeypatch):
    """The changed lines are mapped onto the union's definitions the way each
    branch's are mapped onto the base's."""
    directory, _ = repo
    _write(directory, "m.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "defs")
    union = _git(directory, "rev-parse", "HEAD").strip()
    range_dict = {"m_py/m.py": [(1, 2, "f"), (5, 6, "g")]}
    monkeypatch.setattr(integrate, "repo_to_knowledge", lambda path, name: "m_py/" + path)
    monkeypatch.setattr(integrate, "knowledge_to_repo", lambda path, name: path[5:])
    _write(directory, "m.py", "def f():\n    return 1\n\n\ndef g():\n    return 3\n")
    _write(directory, "f.txt", "changed\n")
    entry_list = integrate.change_entry_list(directory, union, range_dict, "repo")
    assert [(e["path"], e["defs"]) for e in entry_list] == [("f.txt", []), ("m.py", ["g"])]
    assert integrate.change_entry_list(directory, union)[1]["defs"] == []


@pytest.mark.parametrize("leave", ["named pipe", "no read permission"])
def test_a_tracked_file_that_cannot_be_read_leaves_the_others_named_by_definition(
        repo, monkeypatch, leave):
    """The diff is read past it, it is named with no counts and no definition, and the
    other files are named as always."""
    if leave != "named pipe" and os.geteuid() == 0:
        pytest.skip("root reads any file")
    directory, _ = repo
    _write(directory, "m.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    _write(directory, "n.py", "def h():\n    return 1\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "defs")
    union = _git(directory, "rev-parse", "HEAD").strip()
    range_dict = {"m_py/m.py": [(1, 2, "f"), (5, 6, "g")], "n_py/n.py": [(1, 2, "h")]}
    monkeypatch.setattr(integrate, "repo_to_knowledge",
                        lambda path, name: path.replace(".", "_") + "/" + path)
    monkeypatch.setattr(integrate, "knowledge_to_repo",
                        lambda path, name: path.split("/", 1)[1])
    _write(directory, "m.py", "def f():\n    return 1\n\n\ndef g():\n    return 3\n")
    path = os.path.join(directory, "n.py")
    if leave == "named pipe":
        os.remove(path)
        os.mkfifo(path)
    else:
        _write(directory, "n.py", "def h():\n    return 2\n")
        os.chmod(path, 0)
    try:
        assert integrate.change_dict_of(directory, union, "") == {
            "add": [], "delete": [], "line": {"m.py": [(6, 6)]}}
        entry_list = integrate.change_entry_list(directory, union, range_dict, "repo")
    finally:
        if leave != "named pipe":
            os.chmod(path, 0o644)
    assert entry_list == [
        {"path": "m.py", "insertions": 1, "deletions": 1, "new": False, "defs": ["g"]},
        {"path": "n.py", "insertions": "?", "deletions": "?", "new": False, "defs": []}]


def test_a_working_tree_diff_git_cannot_give_leaves_the_files_named_alone(
        repo, monkeypatch, caplog):
    directory, _ = repo
    _write(directory, "f.txt", "changed\n")

    def no_diff(*a):
        raise RuntimeError("could not read what the working tree changed: broken")

    monkeypatch.setattr(integrate, "change_dict_of", no_diff)
    with caplog.at_level("DEBUG", logger=integrate.logger.name):
        for _ in range(3):
            entry_list = integrate.change_entry_list(directory, "HEAD", {}, "repo")
    assert [(e["path"], e["defs"]) for e in entry_list] == [("f.txt", [])]
    # The warning is given once, and what git wrote only at DEBUG
    assert [record.getMessage() for record in caplog.records
            if record.levelname == "WARNING"] == [
        f"the changed lines of {directory} are not named by definition, as git gave no "
        f"diff of them"]
    assert "what git wrote: could not read what the working tree changed: broken" \
           in caplog.text


def test_the_state_block_says_when_no_definition_was_reached_twice():
    work = integrate.build_work_dict(_merged(), [], [], frozenset(), "", 0)
    assert "No definition was changed by more than one side" in integrate.start_text(work)
    text = "\n".join(integrate._state_line_list(work, 1, 5))
    assert "named under" not in text


def _overlap_integration(tmp_path, monkeypatch, range_dict):
    """Run run_integration() over _side_repo()'s union with the analysis stubbed."""
    directory, base = _side_repo(tmp_path)
    return _run_over(directory, base, monkeypatch, range_dict)


def _run_over(directory, base, monkeypatch, range_dict, onto=""):
    """Run run_integration() over a repository's union with the analysis stubbed."""
    merged = integrate.assemble(directory, base, ["a", "b"], onto=onto)
    prepared = {"base": merged["base"], "merged": merged,
                "before": _analysis(range_dict=range_dict),
                "against": _analysis(range_dict=range_dict), "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis(range_dict=range_dict))
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatRuns)
    _StageThatRuns.built.clear()
    return integrate.run_integration(prepared, "", "p", "out")


def test_a_definition_two_sides_changed_is_work_and_a_model_is_called(tmp_path, monkeypatch):
    """git merged both sides' lines of f without a conflict block."""
    out = _overlap_integration(tmp_path, monkeypatch, {"p/x_py/x.py": [(1, 8, "f")]})
    assert out["called"] is True and _StageThatRuns.built == ["integrate"]
    assert out["work"]["overlap"] == [
        {"kind": "both", "file": "x.py", "name": "f", "branches": ["a", "b"]}]


def test_two_sides_changing_different_definitions_call_no_model(tmp_path, monkeypatch):
    out = _overlap_integration(tmp_path, monkeypatch, _X_RANGE)
    assert out["called"] is False and out["work"]["overlap"] == []


def test_the_sides_are_read_from_the_start_when_the_branch_merged_into_moved_on(tmp_path,
                                                                                monkeypatch):
    """main changed a line of f after the branches started; it is one more side, and
    what a and b changed is still read against the start, not against main's tip."""
    directory, base = _side_repo(tmp_path)
    _git(directory, "checkout", "-q", "main")
    _write(directory, "x.py", "def f():\n    a = 1\n    b = 20\n    return a\n"
                              "def g():\n    c = 3\n    d = 4\n    return c\n")
    _git(directory, "commit", "-qam", "main moves on")
    _git(directory, "checkout", "-q", "--detach", base)
    out = _run_over(directory, base, monkeypatch, _X_RANGE, onto="main")
    assert out["work"]["overlap"] == [
        {"kind": "both", "file": "x.py", "name": "f", "branches": ["a", "main"]}]


def test_only_the_referring_lines_in_what_the_other_side_changed_are_named():
    """One reference in an old definition b left alone, one in a definition b added."""
    after = _analysis(range_dict={**_X_RANGE, "p/y_py/y.py": [(1, 3, "h"), (5, 9, "t")]},
                      name={"p/x_py/x.py": ["f", "g"], "p/y_py/y.py": ["h", "t"]},
                      ref=[("p/y_py/y.py", "p/x_py/x.py", "f", [2, 7])])
    before = _analysis(range_dict={**_X_RANGE, "p/y_py/y.py": [(1, 3, "h")]},
                       name={"p/x_py/x.py": ["f", "g"], "p/y_py/y.py": ["h"]})
    side_list = [("a", {"p/x_py/x.py": {"f"}}), ("b", {"p/y_py/y.py": {""}})]
    assert integrate.overlap_entry_list(before, after, side_list, "p") == [
        {"kind": "use", "file": "x.py", "name": "f", "changed_by": "a",
         "used_in": "y.py", "lines": [7], "used_by": ["b"]}]


# == How many times the ending call is sent back ==========


def test_the_call_that_ends_the_run_is_sent_back_as_many_times_as_configured(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_MAX_REFUSAL", 2)
    guard = integrate.make_complete_guard(lambda: ["- something is left"])
    first, second = guard("done"), guard("done")
    assert "It can be sent back 1 more time(s)." in first
    assert "This is the last time it is sent back." in second
    assert guard("done") == ""


def test_with_no_refusal_allowed_the_call_is_accepted_as_it_is(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_MAX_REFUSAL", 0)
    guard = integrate.make_complete_guard(lambda: ["- something is left"])
    assert guard("done") == ""


def test_the_one_refusal_keeps_its_wording(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_MAX_REFUSAL", 1)
    guard = integrate.make_complete_guard(lambda: ["- something is left"])
    assert "This is asked once only." in guard("done")


@pytest.mark.parametrize("count, text", [(0, "are recorded, and the call is accepted as it is."),
                                         (1, "send it back once."),
                                         (3, "send it back, up to 3 times.")])
def test_the_instructions_say_how_many_times_the_call_is_sent_back(monkeypatch, count, text):
    monkeypatch.setattr(config, "INTEGRATE_MAX_REFUSAL", count)
    assert text in integrate.build_integrate_instruction("py")


# == The commit a combined change was written against ==


def test_the_first_parent_of_a_branch_is_read(repo):
    directory, base = repo
    assert integrate.first_parent(directory, "a") == base


def test_a_tip_with_no_parent_is_an_error(tmp_path):
    directory = str(tmp_path / "root")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "root")
    with pytest.raises(RuntimeError, match="has no parent"):
        integrate.first_parent(directory, "main")


# == what the gates read of the sides ======================

def test_a_reference_one_branch_brought_in_is_not_what_the_merge_broke():
    """It does not resolve on that branch on its own, so there is nothing to settle."""
    before = {"no_resolve": {}, "name": {}}
    side = {("p/a_py/a.py", "p/b_py/b.py", "gone"): [3]}
    after = {"no_resolve": {("p/a_py/a.py", "p/b_py/b.py", "gone"): [3]},
             "name": {"p/b_py/b.py": ["kept"]}}
    assert integrate.broken_list_of(before, after, "p", side) == []
    assert len(integrate.broken_list_of(before, after, "p", {})) == 1


def test_a_name_a_side_changed_that_the_union_no_longer_defines_is_a_reason():
    """Markers, references and tests all read the union against itself, so a settling
    that drops one side's work passes every one of them."""
    a_file = "p/a_py/a.py"
    keep_list = [("run/one", {"kept": [a_file], "dropped": [a_file]}), ("run/two", {})]
    reason_list = integrate._drop_reason_list(keep_list, {"kept": [a_file]}, "p")
    assert reason_list == [("- dropped, which run/one added or changed in a.py, is not "
                            "defined in any file now")]


def test_a_rename_a_removal_and_a_move_a_side_was_asked_for_are_not_reasons():
    """What the side's tip no longer defines is what it was asked to take away."""
    lib, helpers = "p/lib_py/lib.py", "p/helpers_py/helpers.py"
    start = _analysis(name={lib: ["foo", "keep", "old_helper"], "p/util_py/util.py": ["moved"]})
    tip = _analysis(name={lib: ["bar", "keep"], helpers: ["moved"]})
    side = {lib: {"foo", "old_helper", "keep"},
            "p/util_py/util.py": {"", "moved"}, helpers: {""}}
    keep_list = integrate.keep_list_of([("a", side)], start, {"a": tip["define"]})
    assert keep_list == [("a", {"bar": [lib], "keep": [lib], "moved": [helpers]})]
    assert integrate._drop_reason_list(keep_list, tip["define"], "p") == []


def test_a_definition_a_side_added_that_the_union_lost_is_a_reason():
    a_file = "p/a_py/a.py"
    start = _analysis(name={a_file: ["old"]})
    keep_list = integrate.keep_list_of(
        [("a", {a_file: {""}})], start, {"a": {"new_fn": [a_file], "old": [a_file]}})
    assert keep_list == [("a", {"new_fn": [a_file]})]
    reason_list = integrate._drop_reason_list(keep_list, start["define"], "p")
    assert len(reason_list) == 1 and "new_fn" in reason_list[0]


def _read_on(turn, *path_tuple):
    """Record each path as read on the turn given, as the reading tools do."""
    for path in path_tuple:
        integrate._STAGE_DICT["read_dict"][path] = turn


def test_the_files_the_sides_met_at_are_what_git_reported_and_where_they_overlap():
    merged = _merged()
    merged["branches"][0]["conflicted"] = ["b.py", "c.py"]
    merged["branches"][1]["conflicted"] = ["c.py"]
    assert integrate.meet_path_list(merged, [{"file": "a.py"}]) == ["a.py", "b.py", "c.py"]


def test_a_file_where_the_sides_met_that_nothing_read_is_a_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    for name in ("a.py", "b.py", "c.py"):
        _write(str(tmp_path), name, "x = 1\n")
    integrate.reset_stage()
    _read_on(2, "b.py")
    reason_list = integrate._no_read_reason_list(["a.py", "b.py", "c.py"])
    assert len(reason_list) == 1
    assert "a.py" in reason_list[0] and "c.py" in reason_list[0]
    assert "b.py" not in reason_list[0]


def test_every_file_where_the_sides_met_having_been_read_is_no_reason():
    integrate.reset_stage()
    _read_on(1, "a.py")
    _read_on(2, "b.py")
    assert integrate._no_read_reason_list(["a.py", "b.py"]) == []


def test_a_file_read_on_the_turn_that_calls_complete_counts_as_read(repo, monkeypatch):
    """The state's "read" is measured at the head of a turn; the gate reads the record
    as it is when COMPLETE() is called."""
    directory, _ = repo
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    integrate.reset_stage()
    work = integrate.build_work_dict(_merged(["f.txt"]), [], [], None, "", 0)
    check = functools.partial(integrate._no_read_reason_list, ["f.txt"])
    turn_note, check_and_keep = integrate.make_state(directory, "HEAD", work, check)
    turn_note(1, 20)
    assert work["read"] == [] and check_and_keep()
    _read_on(1, "f.txt")
    assert check_and_keep() == []


class _StageThatReadsThenCompletes:
    """A stand-in for RLM whose one turn settles f.txt, reads it and calls COMPLETE()."""

    refusal_list: typing.ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        pass

    def run(self, **kwargs):
        kwargs["turn_note"](1, 20)
        _write(config.PROJECT_DIR, "f.txt", "settled\n")
        _read_on(1, "f.txt")
        refusal = kwargs["on_complete"]("done")
        _StageThatReadsThenCompletes.refusal_list.append(refusal)
        return {"answer": "done", "history": [], "iters": 1,
                "refused_completes": 1 if refusal else 0, "error": "",
                "outcome": "complete"}


def test_a_conflicted_file_read_on_the_turn_that_calls_complete_is_not_sent_back(
        repo, monkeypatch):
    directory, base = repo
    prepared = {"base": base, "merged": integrate.assemble(directory, base, ["a", "b"]),
                "before": _analysis(), "against": _analysis(), "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatReadsThenCompletes)
    _StageThatReadsThenCompletes.refusal_list.clear()
    out = integrate.run_integration(prepared, "", "p", "out")
    assert _StageThatReadsThenCompletes.refusal_list == [""]
    assert out["settled"] is True and out["left"] == []


def test_a_file_past_the_work_list_count_is_counted_once_read(repo, monkeypatch):
    """The state shows the first INTEGRATE_WORK_MAX_ITEM files read; the gate reads them
    all."""
    directory, _ = repo
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    integrate.reset_stage()
    _read_on(1, "a.txt", "b.txt", "z.txt")
    work = integrate.build_work_dict(_merged(), [], [], None, "", 0)
    integrate.make_state(directory, "HEAD", work, lambda: [])[0](2, 20)
    assert [entry["path"] for entry in work["read"]] == ["a.txt", "b.txt"]
    assert integrate._no_read_reason_list(["a.txt", "b.txt", "z.txt"]) == []


def test_a_conflicted_file_past_the_work_list_count_has_to_be_read(tmp_path, monkeypatch):
    """The state shows the first INTEGRATE_WORK_MAX_ITEM conflicted files; the gate asks
    for every one of them."""
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 1)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    for name in ("a.txt", "z.txt"):
        _write(str(tmp_path), name, "x\n")
    merged = _merged(["a.txt", "z.txt"])
    integrate.reset_stage()
    _read_on(1, "a.txt")
    assert integrate.build_work_dict(merged, [], [], None, "", 0)["branches"][0][
        "conflicted"] == ["a.txt"]
    reason_list = integrate._no_read_reason_list(integrate.meet_path_list(merged, []))
    assert len(reason_list) == 1 and "z.txt" in reason_list[0]


# == what the state says about itself ======================

def test_whether_the_state_holds_everything_is_worked_out_every_turn(monkeypatch):
    monkeypatch.setattr(integrate.config, "INTEGRATE_WORK_MAX_ITEM", 2)
    work_dict = {"changed": [], "read": [], "overlap": [], "branches": []}
    assert integrate.is_complete(work_dict, ["a", "b", "c"], [], [], []) is False
    assert integrate.is_complete(work_dict, ["a"], [], [], []) is True
    assert integrate.is_complete({**work_dict, "read": [1, 2, 3]}, [], [], [], []) is False


# == the trailer one line of a commit holds ================

def test_entries_over_the_limit_are_left_out_from_the_end_and_the_rest_read_back():
    entry_list = [{"what": f"case {n}", "why": "y" * 200} for n in range(30)]
    text = integrate.decision_text(entry_list, 1000)
    assert len(text) <= 1000
    read_list = integrate.decision_list_of(text)
    assert read_list and read_list == entry_list[:len(read_list)]
    assert integrate.decision_text([{"what": "x" * 200, "why": "y" * 200}], 100) == ""


def test_with_no_limit_every_entry_is_written():
    entry_list = [{"what": f"case {n}", "why": "y" * 200} for n in range(30)]
    assert integrate.decision_list_of(integrate.decision_text(entry_list)) == entry_list
    assert integrate.decision_list_of(integrate.decision_text(entry_list[:1], 6000)) \
        == entry_list[:1]


# == what the stage is asked ===============================

def test_the_request_the_branches_were_made_for_is_at_the_head_of_what_is_asked():
    merge_dict = {"branches": [{"branch": "run/one"}], "messages": []}
    text = integrate.request_of(merge_dict, "設定の読み込みを直して")
    assert text.startswith("The request these branches were made for: 設定の読み込みを直して")
    assert "Combine the changes on run/one." in text


def test_a_merge_with_no_request_is_asked_what_it_always_was():
    merge_dict = {"branches": [{"branch": "run/one"}], "messages": []}
    assert integrate.request_of(merge_dict) == "Combine the changes on run/one."


# == what the markers name and what git is asked for =======

def test_the_markers_name_the_branch_and_what_was_merged_so_far(repo, monkeypatch):
    """Which side of a block is whose is not readable from a commit nobody can look up."""
    directory, _ = repo
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    _prepare(directory, ["a", "b"])
    text = _read(directory, "f.txt")
    assert f"<<<<<<< {integrate.UNION_REF}" in text
    assert f">>>>>>> {integrate.SIDE_REF_ROOT}/b" in text
    # Both names are this tool's own and are taken away again
    assert _git(directory, "for-each-ref", "refs/codetwine-loom/**").strip() == ""


def test_the_repository_is_put_back_when_the_merge_fails(repo, monkeypatch):
    """Whoever ran this in a repository of their own is not left on another commit."""
    directory, base = repo
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "assemble",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no tree")))
    with pytest.raises(RuntimeError, match="no tree"):
        _prepare(directory, ["a", "b"])
    assert _git(directory, "rev-parse", "HEAD").strip() == base


@pytest.mark.parametrize("error", [KeyboardInterrupt("the run was stopped (SIGTERM)"),
                                   RuntimeError("no tree")])
def test_the_repository_is_put_back_when_the_union_in_place_is_stopped_or_fails(
        repo, monkeypatch, error):
    """The union the merges put in the working tree goes as well, and so does the move
    off the branch the repository was on."""
    directory, base = repo
    _git(directory, "checkout", "-q", "main")
    real_assemble = integrate.assemble

    def assemble(*a, **k):
        real_assemble(*a, **k)
        assert _git(directory, "status", "--porcelain").strip()
        raise error

    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "assemble", assemble)
    with pytest.raises(type(error), match=str(error).replace("(", r"\(").replace(")", r"\)")):
        _prepare(directory, ["a", "b"])
    assert _git(directory, "symbolic-ref", "--short", "HEAD").strip() == "main"
    assert _git(directory, "rev-parse", "HEAD").strip() == base
    assert _git(directory, "status", "--porcelain").strip() == ""


def test_a_repository_that_cannot_be_put_back_is_named_with_the_line_that_does_it(
        repo, caplog):
    directory, _ = repo
    _write(directory, "f.txt", "the union\n")
    _write(os.path.join(directory, ".git"), "index.lock", "")
    assert integrate.put_back(directory, "main") is False
    assert "could not be put back on main (fatal: Unable to create" in caplog.text
    assert caplog.text.rstrip().endswith(
        f"to put the repository back: {integrate.put_back_command(directory, 'main')}")
    assert integrate.put_back_command(directory, "main") == (
        f"git -C {directory} checkout main && git -C {directory} reset --hard")
    assert _read(directory, "f.txt") == "the union\n"
    os.remove(os.path.join(directory, ".git", "index.lock"))
    assert integrate.put_back(directory, "main") is True
    assert _git(directory, "symbolic-ref", "--short", "HEAD").strip() == "main"
    assert _read(directory, "f.txt") == "one\ntwo\nthree\n"
    assert integrate.put_back(directory, "") is False


@pytest.mark.parametrize("how", ["stop", "fail"])
@pytest.mark.parametrize("writer", ["place", "move"])
def test_the_files_a_git_that_wrote_a_tree_left_are_taken_away_when_it_does_not_end(
        repo, monkeypatch, how, writer):
    """git writes the files before the index: one stopped between the two leaves files
    the index does not hold, which reset --hard does not take away."""
    directory, base = repo
    _git(directory, "checkout", "-q", "-b", "deep", base)
    os.makedirs(os.path.join(directory, "new", "deep"))
    _write(os.path.join(directory, "new", "deep"), "x.txt", "x\n")
    _write(directory, "a_only.txt", "a\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "deep")
    _git(directory, "checkout", "-q", "--detach", base)
    _write(directory, "g.txt", "kept, but changed\n")
    real_run = gitcmd.run

    def run(project_dir, *args, **kwargs):
        if args[0] in ("read-tree", "checkout"):
            # what a git stopped half way leaves: some files written, the index as it was
            os.makedirs(os.path.join(directory, "new", "deep"), exist_ok=True)
            _write(os.path.join(directory, "new", "deep"), "x.txt", "x\n")
            _write(directory, "a_only.txt", "a\n")
            if how == "stop":
                raise KeyboardInterrupt("the run was stopped (SIGINT)")
            return subprocess.CompletedProcess(list(args), 1, "", "git failed")
        return real_run(project_dir, *args, **kwargs)

    monkeypatch.setattr(integrate.gitcmd, "run", run)
    with pytest.raises(KeyboardInterrupt if how == "stop" else RuntimeError):
        if writer == "place":
            integrate.place(directory, "deep")
        else:
            integrate._move_head(directory, "deep")
    assert not os.path.lexists(os.path.join(directory, "new"))
    assert not os.path.lexists(os.path.join(directory, "a_only.txt"))
    assert _read(directory, "g.txt") == "kept, but changed\n"


def test_a_file_that_was_there_or_that_the_index_holds_is_not_taken_away(repo):
    directory, _ = repo
    _write(directory, "there.txt", "there before\n")
    _write(directory, "held.txt", "held\n")
    _git(directory, "add", "held.txt")
    integrate._remove_new_file(directory, ["held.txt", "gone.txt"])
    assert _read(directory, "held.txt") == "held\n"
    assert _read(directory, "there.txt") == "there before\n"


def test_the_files_the_run_made_are_named_when_the_repository_is_put_back(repo, caplog):
    directory, _ = repo
    _git(directory, "checkout", "-q", "main")
    _write(directory, "made_by_the_run.txt", "x\n")
    _write(os.path.join(directory, ".git"), "index.lock", "")
    assert integrate.put_back(directory, "main") is False
    assert ("could not be put back on main (fatal: Unable to create" in caplog.text)
    assert ("; the 1 file(s) made since the run started that git does not track stay: "
            "made_by_the_run.txt; to put the repository back: "
            f"{integrate.put_back_command(directory, 'main')}") in caplog.text
    caplog.clear()
    os.remove(os.path.join(directory, ".git", "index.lock"))
    assert integrate.put_back(directory, "main") is True
    assert (f"{directory} is back on main; the 1 file(s) made since the run started that "
            f"git does not track stay: made_by_the_run.txt") in caplog.text
    assert _read(directory, "made_by_the_run.txt") == "x\n"
    caplog.clear()
    os.remove(os.path.join(directory, "made_by_the_run.txt"))
    assert integrate.put_back(directory, "main") is True
    assert "is back on" not in caplog.text


def test_a_git_too_old_for_the_merge_is_said_plainly(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(integrate.gitcmd, "version", lambda _: (2, 30, 0))
    with pytest.raises(RuntimeError, match="git 2.38 or later"):
        _prepare(directory, ["a", "b"])


def test_a_marker_in_a_file_no_diff_covers_is_found(repo):
    """A file this stage created is untracked, and a marker in one is a marker."""
    directory, _ = repo
    _write(directory, "new.txt", _BLOCK_TEXT)
    assert "new.txt" in integrate.marker_path_list(directory)


def test_a_marker_in_a_file_that_is_not_utf_8_is_found(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(file_utils, "SOURCE_ENCODING", ["cp932"])
    with open(os.path.join(directory, "sj.py"), "wb") as handle:
        handle.write(("# 表\n" + _BLOCK_TEXT).encode("cp932"))
    assert "sj.py" in integrate.marker_path_list(directory)


def test_a_new_file_without_a_marker_is_not_named(repo):
    directory, _ = repo
    _write(directory, "new.txt", "nothing to settle\n")
    assert integrate.marker_path_list(directory) == []


@pytest.mark.parametrize("text,is_marker", [
    ("<<<<<<< refs/codetwine-loom/merged-so-far\n", True),
    ("<<<<<<< refs/codetwine-loom/merged-so-far\r\n", True),
    ("<<<<<<< refs/codetwine-loom/merged-so-far:old/name.py\n", True),
    (">>>>>>> refs/codetwine-loom/side/task/1\n", True),
    (">>>>>>> refs/codetwine-loom/side/b:f.py\n", True),
    ("<<< refs/codetwine-loom/merged-so-far\n", True),
    ("<" * 32 + " refs/codetwine-loom/merged-so-far\n", True),
    ("<<<<<<<\trefs/codetwine-loom/merged-so-far\n", True),
    ("<<<<<<< refs/codetwine-loom/merged-so-farther\n", False),
    (">>>>>>> refs/codetwine-loom/side/\n", False),
    ("<<<<<<<refs/codetwine-loom/merged-so-far\n", False),
    ("<<<<<<< ours\n", False), (">>>>>>> theirs\n", False), ("<<<<<<<\n", False),
    ("a\n=======\n", False), ("|||||||\tbase\n", False), ("Title\n==========\n", False),
    # Spaces and tabs in front of a name line, and white space after the first name
    ("    <<<<<<< refs/codetwine-loom/merged-so-far\n", True),
    ("\t>>>>>>> refs/codetwine-loom/side/b\n", True),
    (" \t <<< refs/codetwine-loom/merged-so-far:x.py\r\n", True),
    ("<<<<<<< refs/codetwine-loom/merged-so-far \n", True),
    ("<<<<<<< refs/codetwine-loom/merged-so-far\tnote\n", True),
    ("    <<<<<<< ours\n", False), ("    =======\n", False), ("\t||||||| base\n", False),
    # Anything in front of the run
    ("x <<<<<<< refs/codetwine-loom/merged-so-far\n", True),
    ("# <<<<<<< refs/codetwine-loom/merged-so-far\n", True),
    ("#<<<<<<< refs/codetwine-loom/merged-so-far\n", True),
    ("// >>>>>>> refs/codetwine-loom/side/b\n", True),
    ("> >>>>>>> refs/codetwine-loom/side/b:f.py\n", True),
    ("* <<< refs/codetwine-loom/merged-so-far:x\n", True),
    ("======= > refs/codetwine-loom/side/b\n", True),
    ("||||||| < refs/codetwine-loom/merged-so-far\n", True),
    ("# <<<<<<< refs/codetwine-loom/merged-so-farther\n", False),
    ("# >>>>>>> refs/codetwine-loom/side/\n", False),
    ("# <<<<<<<refs/codetwine-loom/merged-so-far\n", False),
    ("# refs/codetwine-loom/merged-so-far\n", False),
    ("# <<<<<<< ours\n", False), ("# =======\n", False), ("// ||||||| base\n", False)])
def test_a_new_file_s_marker_is_a_line_naming_what_this_tool_merges_under(repo, text,
                                                                         is_marker):
    """A "<" or ">" anywhere in the line, white space, and one of the two names."""
    directory, _ = repo
    _write_raw(directory, b"NOTES.md", text.encode())
    assert integrate.marker_path_list(directory) == (["NOTES.md"] if is_marker else [])


def test_git_is_asked_for_the_lines_a_name_line_is_told_by(repo):
    """What git grep finds by _NAME_LINE_PATTERN is what _NAME_LINE finds."""
    directory, _ = repo
    line_list = [
        b"<<<<<<< refs/codetwine-loom/merged-so-far",
        b"<<<<<<< refs/codetwine-loom/merged-so-far\r",
        b"<<<<<<< refs/codetwine-loom/merged-so-far:a b.py",
        b"<<<<<<< refs/codetwine-loom/merged-so-far ",
        b"<<<<<<< refs/codetwine-loom/merged-so-far\tx",
        b"<<<<<<< refs/codetwine-loom/merged-so-far\rx",
        b"<<<<<<< refs/codetwine-loom/merged-so-farther",
        b"<<<<<<< refs/codetwine-loom/merged-so-far/x",
        b"    <<<<<<< refs/codetwine-loom/merged-so-far", b"\t>>> refs/codetwine-loom/side/b",
        b" \t > refs/codetwine-loom/side/task/1:f.py\r", b">>>>>>> refs/codetwine-loom/side/",
        b">>>>>>> refs/codetwine-loom/side/\r", b">>>>>>> refs/codetwine-loom/side/\rx",
        b">>>>>>> refs/codetwine-loom/side/b\r\r", b">>>>>>> refs/codetwine-loom/side/ b",
        b"x <<<<<<< refs/codetwine-loom/merged-so-far", b"<<<<<<<refs/codetwine-loom/merged-so-far",
        b"<<<<<<< ours", b"    >>>>>>> theirs", b"=======", b"    =======",
        b"# <<<<<<< refs/codetwine-loom/merged-so-far",
        b"#<<<<<<< refs/codetwine-loom/merged-so-far:f.py",
        b"// >>>>>>> refs/codetwine-loom/side/b", b"> >>>>>>> refs/codetwine-loom/side/b\r",
        b"\xe9 < refs/codetwine-loom/side/b", b"a `<<<<<<< refs/codetwine-loom/merged-so-far` b",
        b"a <<<<<<< refs/codetwine-loom/merged-so-farther", b"# >>>>>>> refs/codetwine-loom/side/",
        b"# >>>>>>> refs/codetwine-loom/side/\r", b"# refs/codetwine-loom/side/b",
        b"# <<<<<<< ours",
        b"======= > refs/codetwine-loom/side/b", b"<<< >>> refs/codetwine-loom/merged-so-far",
        b"# =======", b"<" * 3000 + b" x"]
    _write_raw(directory, b"lines.txt", b"\n".join(line_list) + b"\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "lines")
    proc = gitcmd.run(directory, "grep", "-a", "-h", "--no-color", "--no-line-number",
                      "-E", "-e", integrate._NAME_LINE_PATTERN, "HEAD", "--", "lines.txt",
                      text=False)
    name_line_list = [line for line in line_list
                      if integrate._NAME_LINE.search(line.rstrip(b"\r"))]
    assert proc.stdout.split(b"\n")[:-1] == name_line_list
    assert len(name_line_list) == 19
    # The name lines, the one separator line, and the one line starting with a run
    # of "<" of another length than the file's markers
    assert [path for path, _ in integrate._marker_line_dict(directory, ["HEAD"])["HEAD"]
            if path == "lines.txt"] == ["lines.txt"] * (len(name_line_list) + 2)


@pytest.mark.parametrize("front", [b"    ", b"\t", b"# ", b"//", b"> ", b"* ", b"=== "])
def test_a_name_line_with_anything_in_front_of_it_is_no_line_of_a_block_git_wrote(front):
    line_list = [("f.txt", b"<<<<<<< refs/codetwine-loom/merged-so-far"), ("f.txt", b"======="),
                 ("f.txt", front + b">>>>>>> refs/codetwine-loom/side/b"), ("f.txt", b"=======")]
    assert integrate._block_separator_dict(line_list, ["b"]) == {}
    assert integrate._block_separator_dict(
        [*line_list, ("f.txt", b">>>>>>> refs/codetwine-loom/side/b")], ["b"]) == {
            "f.txt": {b"=======": 1}}
    # An end with no start is made up for out of the lines no block holds
    assert integrate._block_separator_dict(
        [("f.txt", front + b"<<<<<<< refs/codetwine-loom/merged-so-far"), *line_list[1:],
         ("f.txt", b">>>>>>> refs/codetwine-loom/side/b")], ["b"]) == {
            "f.txt": {b"=======": 1}}


def _two_file_repo(tmp_path, a_file, a_text, b_file, b_text, base_file_dict=None):
    """A repository whose branches a and b each write one file over a base commit."""
    directory = str(tmp_path / "two_file")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    for name, text in (base_file_dict or {"f.txt": "one\n"}).items():
        _write(directory, name, text)
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    for branch, name, text in (("a", a_file, a_text), ("b", b_file, b_text)):
        _git(directory, "checkout", "-q", "-b", branch, base)
        if text is None:
            _git(directory, "rm", "-q", name)
        else:
            os.makedirs(os.path.dirname(os.path.join(directory, name)), exist_ok=True)
            _write(directory, name, text)
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", branch)
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


# A block shown under this tool's own names, as a document of it or a test of it holds
_FIXTURE_TEXT = ("<<<<<<< refs/codetwine-loom/merged-so-far\nx\n=======\ny\n"
                 ">>>>>>> refs/codetwine-loom/side/<branch>\n")
_FIXTURE_START = b"<<<<<<< refs/codetwine-loom/merged-so-far"
_FIXTURE_END = b">>>>>>> refs/codetwine-loom/side/<branch>"


def test_marker_lines_a_branch_added_on_purpose_are_not_the_merges(tmp_path):
    """A merge tool's fixture is that branch's work, not something the merge left."""
    directory, base = _two_file_repo(tmp_path, "tests/fixture.txt", _FIXTURE_TEXT,
                                 "g.txt", "b\n")
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == [] and merged["unmarked"] == []
    assert merged["marker"]["name_count"] == {_FIXTURE_START: 1, _FIXTURE_END: 1}
    assert merged["marker"]["name_at"] == {("tests/fixture.txt", _FIXTURE_START): 1,
                                           ("tests/fixture.txt", _FIXTURE_END): 1}
    assert merged["marker"]["separator"] == {}
    assert integrate.marker_path_list(directory) == ["tests/fixture.txt"]
    assert integrate.marker_path_list(directory, merged["marker"]) == []
    # A name line the settling adds next to them is the stage's own
    with open(os.path.join(directory, "tests/fixture.txt"), "a", encoding="utf-8") as handle:
        handle.write("<<<<<<< refs/codetwine-loom/merged-so-far\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == \
        ["tests/fixture.txt"]


def test_a_conflict_in_a_file_holding_a_side_s_marker_lines_is_still_found(tmp_path):
    base_text = "top\n=======\nbottom\n"
    directory, base = _two_file_repo(tmp_path, "doc.txt", "TOP a\n=======\nbottom\n",
                                 "doc.txt", "TOP b\n=======\nbottom\n",
                                 {"doc.txt": base_text})
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == ["doc.txt"] and merged["unmarked"] == []
    assert merged["marker"]["separator"] == {"doc.txt": {b"=======": 1}}
    assert integrate.marker_path_list(directory, merged["marker"]) == ["doc.txt"]


def test_a_deleted_file_holding_a_side_s_marker_lines_is_still_unmarked(tmp_path):
    """git keeps b's version, whose marker lines are b's own: no block the merge wrote."""
    directory, base = _two_file_repo(tmp_path, "fixture.txt", None,
                                 "fixture.txt", _FIXTURE_TEXT + "changed by b\n",
                                 {"fixture.txt": _FIXTURE_TEXT})
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == ["fixture.txt"]
    assert merged["unmarked"] == ["fixture.txt"]
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_union_whose_only_markers_a_branch_added_calls_no_model(tmp_path, monkeypatch):
    directory, base = _two_file_repo(tmp_path, "tests/fixture.txt", _FIXTURE_TEXT,
                                 "g.txt", "b\n")
    prepared = {"base": base, "merged": integrate.assemble(directory, base, ["a", "b"]),
                "before": _analysis(), "against": _analysis(), "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    out = integrate.run_integration(prepared, "", "p", "out")
    assert out["called"] is False and out["work"]["markers"] == []


def test_the_lines_read_are_the_name_lines_and_each_file_s_separator_lines(tmp_path):
    directory, base = _two_file_repo(
        tmp_path, "a:1.txt",
        "x\n=======\r\n==========\n=======\n>>>>>>> b\n||||||| base\n=======x\n"
        ">>>>>>> refs/codetwine-loom/side/b\n", "g.txt",
        "<<<<<<< refs/codetwine-loom/merged-so-far\r\n")
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    assert integrate._marker_line_dict(directory, [base, a, b]) == {
        base: [],
        a: [("a:1.txt", b"======="), ("a:1.txt", b"======="), ("a:1.txt", b"||||||| base"),
            ("a:1.txt", b">>>>>>> refs/codetwine-loom/side/b")],
        b: [("g.txt", b"<<<<<<< refs/codetwine-loom/merged-so-far")]}
    assert integrate._marker_line_dict(directory, []) == {}


def test_record_gap_keeps_its_own_list_and_is_emptied_with_the_stage():
    integrate.reset_stage()
    assert integrate.record_gap("case_3", "b.py does not do it") == \
        {"ok": True, "recorded": 1, "test": "none"}
    assert integrate.gap_list() == [{"what": "case_3", "why": "b.py does not do it"}]
    assert integrate.decision_list() == []
    integrate.reset_stage()
    assert integrate.gap_list() == []


@pytest.mark.parametrize("result, word, kept", [(True, "failed", True),
                                                (None, "unclear", True),
                                                (False, "passed", False)])
def test_record_gap_runs_the_test_it_is_given_and_keeps_it_unless_it_passes(
        result, word, kept):
    integrate.reset_stage()
    seen = []
    integrate.set_gap_check(lambda path, text: seen.append((path, text)) or result)
    reply = integrate.record_gap("case_3", "b.py does not do it", "tests/test_g.py", "t\n")
    assert seen == [("tests/test_g.py", "t\n")] and reply["test"] == word
    entry = integrate.gap_list()[0]
    assert ("test_path" in entry) == kept
    # The trailer holds the case alone
    assert "test_path" not in integrate.decision_text(integrate.gap_list())
    integrate.reset_stage()
    assert integrate.record_gap("case_4", "c.py")["test"] == "none"


def test_record_gap_refuses_a_case_given_no_test_while_the_stage_can_run_one():
    integrate.reset_stage()
    integrate.set_gap_check(lambda path, text: pytest.fail("a test was run"))
    for test_path, test_text in (("", ""), ("tests/test_g.py", "")):
        reply = integrate.record_gap("case_3", "b.py does not do it", test_path, test_text)
        assert reply["ok"] is False and "record_decision()" in reply["error"]
    assert integrate.gap_list() == []
    integrate.reset_stage()


@pytest.fixture
def _gap_project(tmp_path, monkeypatch):
    """A project directory, and a run_test stand-in that says what it saw."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    seen = {}

    def run_test(path_list=None):
        path = path_list[0] if path_list else ""
        seen["existed"] = os.path.exists(os.path.join(str(tmp_path), path)) if path else None
        seen["text"] = (pathlib.Path(tmp_path, path).read_text(encoding="utf-8")
                        if path else "")
        return seen["result"]

    monkeypatch.setattr(integrate.edit, "run_test", run_test)
    return tmp_path, seen


@pytest.mark.parametrize("status, failed, target, baseline, expect", [
    ("passed", [], "path", set(), False),
    ("failed", ["t::g"], "path", set(), True),
    ("failed", [], "path", set(), True),
    ("failed", [], "none", set(), None),
    ("failed", ["t::old"], "none", {"t::old"}, None),
    ("failed", ["t::old", "t::g"], "none", {"t::old"}, True),
    ("did not run", [], "path", set(), None)])
def test_a_gap_test_is_written_for_one_run_and_taken_back_out(
        _gap_project, monkeypatch, status, failed, target, baseline, expect):
    root, seen = _gap_project
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", target)
    seen["result"] = {"status": status, "failed": failed, "failed_excerpt": "",
                      "targets": [], "output": ""}
    assert integrate.gap_test_result("tests/new/test_g.py", "the test\n",
                                     frozenset(baseline)) is expect
    assert seen["existed"] is True and seen["text"] == "the test\n"
    assert not (root / "tests").exists()


@pytest.mark.parametrize("build, expect", [(True, False), (False, None), (None, None)])
def test_a_gap_test_that_passes_where_no_path_is_taken_counts_only_if_it_is_built(
        _gap_project, monkeypatch, build, expect):
    """Where the tool takes no paths, a pass says the case is done only for a test file
    the whole suite builds."""
    root, seen = _gap_project
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    seen["result"] = {"status": "passed", "failed": [], "failed_excerpt": "",
                      "targets": [], "output": ""}
    probe_list = []

    def in_whole_run(path, whole_dict):
        probe_list.append((path, (root / path).exists(), whole_dict["status"]))
        return build

    monkeypatch.setattr(integrate.edit, "in_whole_run", in_whole_run)
    assert integrate.gap_test_result("tests/new/test_g.c", "the test\n",
                                     frozenset()) is expect
    assert probe_list == [("tests/new/test_g.c", True, "passed")]
    assert not (root / "tests").exists()


def test_a_gap_test_over_a_file_already_there_or_outside_is_not_run(_gap_project):
    root, seen = _gap_project
    (root / "t.py").write_text("mine\n", encoding="utf-8")
    assert integrate.gap_test_result("t.py", "x", frozenset()) is None
    assert integrate.gap_test_result("../t.py", "x", frozenset()) is None
    assert (root / "t.py").read_text(encoding="utf-8") == "mine\n" and "existed" not in seen


def _gap_seen(tmp_path, monkeypatch):
    """Point the gap test at tmp_path and keep the bytes and the second of each run."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    seen_list = []

    def run_test(path_list=None):
        full = os.path.join(str(tmp_path), path_list[0])
        seen_list.append((pathlib.Path(full).read_bytes(), int(os.stat(full).st_mtime)))
        return {"status": "passed", "failed": [], "failed_excerpt": "", "targets": [],
                "output": ""}

    monkeypatch.setattr(integrate.edit, "run_test", run_test)
    return seen_list


def test_a_gap_test_is_stored_as_a_new_file_of_the_repository_is(tmp_path, monkeypatch):
    """The test of a case is written with the encoding create_file gives a new file."""
    seen_list = _gap_seen(tmp_path, monkeypatch)
    monkeypatch.setattr(file_utils, "SOURCE_ENCODING", ["cp932"])
    _git(tmp_path, "init", "-q")
    (tmp_path / "old_test.py").write_bytes("NAME = '日本語'\n".encode("cp932"))
    assert integrate.gap_test_result("tests/test_g.py", "NAME = '日本語'\n",
                                     frozenset()) is False
    assert seen_list[0][0] == "NAME = '日本語'\n".encode("cp932")
    # A character that encoding cannot write leaves the case untold and nothing behind
    assert integrate.gap_test_result("tests/test_g.py", "NAME = 'é'\n",
                                     frozenset()) is None
    assert len(seen_list) == 1 and not (tmp_path / "tests").exists()


def test_a_gap_test_written_again_is_given_a_later_second(tmp_path, monkeypatch):
    """A file written where one was taken away is not left with that file's second."""
    seen_list = _gap_seen(tmp_path, monkeypatch)
    for text in ("CODE = 1\n", "CODE = 0\n", "CODE = 2\n"):
        integrate.gap_test_result("tests/test_g.py", text, frozenset())
        assert not (tmp_path / "tests").exists()
    second_list = [second for _, second in seen_list]
    assert len(second_list) == 3 and second_list == sorted(set(second_list))


def test_a_case_whose_test_another_language_s_command_runs_is_not_told(tmp_path,
                                                                    monkeypatch):
    """The whole-suite check and the way paths are passed are the primary command's."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(edit, "is_other_test_path", lambda path: path.startswith("web/"))
    monkeypatch.setattr(edit, "run_test", lambda path_list=None: pytest.fail("it ran"))
    assert integrate.gap_test_result("web/test/gap.test.js", "x", frozenset()) is None
    assert not (tmp_path / "web").exists()


def test_the_gap_tests_leave_the_stage_s_own_test_state(_gap_project, monkeypatch):
    _, seen = _gap_project
    seen["result"] = {"status": "failed", "failed": ["t::g"], "failed_excerpt": "",
                      "targets": [], "output": ""}
    monkeypatch.setattr(integrate.edit, "_last_test_ok", True)
    integrate.gap_test_result("tests/test_g.py", "x", frozenset())
    assert integrate.edit.last_test_ok() is True


def test_a_gap_test_the_check_cannot_put_back_is_unclear_and_taken_away(tmp_path,
                                                                        monkeypatch):
    """The test command leaves a named pipe at the test the check of the whole run added
    a line to: the case is kept as not done, record_gap says "unclear", and the tree is
    left as it was."""
    root = tmp_path / "p"
    (root / "tests").mkdir(parents=True)
    for args in (("init", "-q"), ("config", "user.email", "t@example.invalid"),
                 ("config", "user.name", "t")):
        _git(root, *args)
    (root / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    monkeypatch.setattr(config, "PROJECT_DIR", str(root))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    monkeypatch.setattr(config, "TEST_PATH", "tests")
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "none")
    monkeypatch.setattr(config, "RUN_TEST", True)
    monkeypatch.setattr(config, "TEST_COMMAND", [sys.executable, "-c", (
        "import os, pathlib, sys\n"
        "for p in pathlib.Path('tests').rglob('*.py'):\n"
        "    if p.is_file() and p.read_text().splitlines()[-1:] == [')']:\n"
        "        os.remove(p); os.mkfifo(p); sys.exit(1)\n"
        "print('1 passed in 0.01s')\n")])
    edit.reset_patch_round()
    entry = {"what": "case_1", "why": "a.py", "test_path": "tests/test_g.py",
             "test_text": "def test_g():\n    pass\n"}
    keep, done = integrate.recheck_gap_list([entry])
    assert (keep, done) == ([entry], [])
    integrate.reset_stage()
    integrate.set_gap_check(functools.partial(integrate.gap_test_result,
                                              baseline_set=frozenset()))
    assert integrate.record_gap("case_1", "a.py", entry["test_path"],
                                entry["test_text"])["test"] == "unclear"
    # A test whose directory cannot be made says nothing either
    assert integrate.gap_test_result("tests/test_a.py/test_g.py", "x = 1\n",
                                     frozenset()) is None
    assert sorted(os.listdir(root / "tests")) == ["test_a.py"]
    assert _git(root, "status", "--porcelain") == ""
    integrate.reset_stage()


def test_the_cases_whose_test_passes_now_are_the_ones_left_out(monkeypatch):
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    monkeypatch.setattr(integrate, "gap_test_result",
                        lambda path, text, baseline: {"a": False, "b": True, "c": None}[path])
    gap_list = [{"what": "case_1", "why": "", "test_path": "a", "test_text": ""},
                {"what": "case_2", "why": "", "test_path": "b", "test_text": ""},
                {"what": "case_3", "why": "", "test_path": "c", "test_text": ""},
                {"what": "case_4", "why": ""}]
    keep, done = integrate.recheck_gap_list(gap_list)
    assert [e["what"] for e in done] == ["case_1"]
    assert [e["what"] for e in keep] == ["case_2", "case_3", "case_4"]


def test_a_case_a_later_commit_found_done_is_left_out_and_the_others_stay():
    task_1 = {"gap": '[{"what":"case_5","why":"a"},{"what":"case_6","why":"b"}]'}
    combine = {"gap": '[{"what":"case_6","why":"b"}]',
               "gap_done": '[{"what":"case_5","why":"a"}]'}
    # A test task written again on top, after the combined commit
    again = {"gap": '[{"what":"case_9","why":"c"}]'}
    assert integrate.gap_list_of([task_1, combine, again]) == [
        {"what": "case_6", "why": "b"}, {"what": "case_9", "why": "c"}]


def test_the_gaps_of_a_range_are_read_out_of_their_trailers(repo):
    directory, base = repo
    _committed(directory, base, "gapped",
               "test-create: x\n\nadd go\n\nCodetwine-Loom-Stage: test-create\n"
               "Codetwine-Loom-Verdict: settled\n"
               'Codetwine-Loom-Gap: [{"what":"case_8","why":"imports are not resolved"}]')
    assert integrate.gap_list_in(directory, base, "gapped") == \
        [{"what": "case_8", "why": "imports are not resolved"}]


def _union_over(directory, base, branch, request, stand_on, gap):
    """Commit a union's merge over base and a stage over it, on a branch of its own.

    The merge records stand_on as the earlier union it stands on ("" records none) and
    the stage the case named gap. Returns the stage's commit, which the branch is at.
    """
    _committed(directory, base, f"{branch}-side", f"{branch} side")
    _git(directory, "checkout", "-q", "-B", branch, base)
    _git(directory, "merge", "-q", "--no-ff", "-m",
         f"integrate: x\n\n{request}\n\nCodetwine-Loom-Run-Request: {request}\n"
         f"Codetwine-Loom-Verdict: settled\n"
         + (f"Codetwine-Loom-Stand-On: {stand_on}\n" if stand_on else ""),
         f"{branch}-side")
    tip = _committed(directory, branch, f"{branch}-test",
                     f"test-create: x\n\n{request}\n\nCodetwine-Loom-Stage: test-create\n"
                     f"Codetwine-Loom-Verdict: settled\nCodetwine-Loom-Gap: "
                     f'[{{"what":"{gap}","why":"{gap} is not done"}}]')
    _git(directory, "branch", "-f", branch, tip)
    return tip


@pytest.mark.parametrize("hold", ["earlier", "none", "plain", "wrong"])
def test_a_union_counts_the_cases_of_the_union_its_merge_was_recorded_to_stand_on(
        repo, hold):
    """The merge of a union written against a commit of the working tree records the
    earlier union that commit holds; the earlier union's runs come first, and its cases
    count while the working tree holds it. A merge that records none - over a commit of
    the working tree with the earlier union's commit as its second parent all the same -
    or records what is no commit counts its own range alone."""
    directory, base = repo
    case_1 = {"what": "case_1", "why": "case_1 is not done"}
    case_2 = {"what": "case_2", "why": "case_2 is not done"}
    tip = _union_over(directory, base, "earlier", "add a", "", "case_1")
    _git(directory, "checkout", "-q", "--detach", base if hold == "none" else tip)
    _write(directory, "tree.txt", "on disk\n")
    snapshot = gitcmd.snapshot_commit(directory, tip)
    os.remove(os.path.join(directory, "tree.txt"))
    stand_on = {"earlier": tip, "none": tip, "plain": "", "wrong": "earlier"}[hold]
    later = _union_over(directory, snapshot, "later", "add c", stand_on, "case_2")
    _git(directory, "checkout", "-q", "--detach", base if hold == "none" else tip)
    record_dict = integrate.branch_record_dict(directory, "later")
    told_list = [entry["asked"] for entry in record_dict["told"]]
    if hold in ("plain", "wrong"):
        assert integrate.gap_list_in(directory, snapshot, later) == [case_2]
        assert told_list == ["add c", "add c"] and record_dict["earlier_list"] == []
        return
    assert told_list == ["add a", "add a", "add c", "add c"]
    assert [part_dict["held"] for part_dict in record_dict["earlier_list"]] == [
        hold == "earlier"]
    assert record_dict["gap"] == ([case_1, case_2] if hold == "earlier" else [case_2])
    # The union read as it goes into the working tree holds the earlier one with it
    assert integrate.gap_list_in(directory, snapshot, later) == [case_1, case_2]


@pytest.mark.parametrize("tree", ["rewrite", "earlier_out", "none"])
def test_an_earlier_union_is_held_with_the_union_over_it_the_working_tree_holds(
        repo, tree):
    """The working tree holds the earlier union a union stands on while it holds that
    union, a stage of which rewrote a file the earlier union added, or while the
    earlier union's own files are gone; holding neither, it holds neither."""
    directory, base = repo
    case_1 = {"what": "case_1", "why": "case_1 is not done"}
    case_2 = {"what": "case_2", "why": "case_2 is not done"}
    tip = _union_over(directory, base, "earlier", "add a", "", "case_1")
    _git(directory, "checkout", "-q", "--detach", tip)
    _write(directory, "tree.txt", "on disk\n")
    snapshot = gitcmd.snapshot_commit(directory, tip)
    os.remove(os.path.join(directory, "tree.txt"))
    later = _union_over(directory, snapshot, "later", "add c", tip, "case_2")
    name_list = ["later-side.txt", "later-test.txt"]
    if tree == "rewrite":
        _git(directory, "checkout", "-q", "later")
        _write(directory, "earlier-side.txt", "earlier-side\nrewritten\n")
        _git(directory, "commit", "-qam",
             f"{integrate.REFACTOR_STAGE}: x\n\nadd c\n\nCodetwine-Loom-Stage: "
             f"{integrate.REFACTOR_STAGE}\nCodetwine-Loom-Verdict: settled")
        later = _git(directory, "rev-parse", "HEAD").strip()
        name_list += ["earlier-side.txt", "earlier-test.txt"]
    # What the later union holds is on disk, uncommitted, as a change put in leaves it
    _git(directory, "checkout", "-q", "--detach", base)
    for name in name_list if tree != "none" else []:
        _write(directory, name, _git(directory, "show", f"{later}:{name}"))
    record_dict = integrate.branch_record_dict(directory, "later")
    held = tree != "none"
    assert record_dict["held"] is held
    assert [part_dict["held"] for part_dict in record_dict["earlier_list"]] == [held]
    assert record_dict["gap"] == ([case_1, case_2] if held else [case_2])
    assert integrate.gap_list_in(directory, snapshot, later) == [case_1, case_2]


def test_the_unions_a_union_stands_on_are_read_down_the_chain_they_were_recorded_in(
        repo):
    """Each union's merge records the one under it; the earliest union's runs come
    first and every case counts while the working tree holds them all."""
    directory, base = repo
    tip = base
    for index in range(3):
        _git(directory, "checkout", "-q", "--detach", tip)
        _write(directory, "tree.txt", f"{index}\n")
        snapshot = gitcmd.snapshot_commit(directory, "" if tip == base else tip)
        os.remove(os.path.join(directory, "tree.txt"))
        tip = _union_over(directory, snapshot, f"u{index}", f"step {index}",
                          "" if not index else tip, f"case_{index}")
    _git(directory, "checkout", "-q", "--detach", tip)
    record_dict = integrate.branch_record_dict(directory, "u2")
    assert [entry["asked"] for entry in record_dict["told"]] == [
        "step 0", "step 0", "step 1", "step 1", "step 2", "step 2"]
    assert [part_dict["held"] for part_dict in record_dict["earlier_list"]] == [True, True]
    assert [entry["what"] for entry in record_dict["gap"]] == ["case_0", "case_1", "case_2"]


def test_a_case_carried_again_by_the_commit_that_combines_is_read_once():
    gap = '[{"what":"case_5","why":"a"},{"what":"case_6","why":"b"}]'
    record_list = [{"gap": gap}, {"gap": ""}, {"gap": gap}]
    assert integrate.gap_list_of(record_list) == [{"what": "case_5", "why": "a"},
                                                  {"what": "case_6", "why": "b"}]


def test_a_branch_record_reads_under_the_stages_down_to_the_merge(repo):
    directory, base = repo
    _git(directory, "checkout", "-q", "-b", "staged", base)
    for message in (("integrate: x\n\nadd go\n\nCodetwine-Loom-Verdict: settled\n"
                     "Codetwine-Loom-Did: merged"),
                    ("test-create: x\n\nadd go\n\nCodetwine-Loom-Stage: test-create\n"
                     "Codetwine-Loom-Verdict: settled\n"
                     'Codetwine-Loom-Gap: [{"what":"case_8","why":"no edge"}]'),
                    ("refactor: x\n\nadd go\n\nCodetwine-Loom-Stage: refactor\n"
                     "Codetwine-Loom-Verdict: settled")):
        _write(directory, f"{len(message)}.txt", "x\n")
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", message)
    told = integrate.branch_record_dict(directory, "staged")["told"]
    assert [t["stage"] for t in told] == ["", "test-create", "refactor"]
    assert told[-1]["stage"] == "refactor"


# == a branch another merged branch already holds ========

def _contained_repo(tmp_path):
    """lib.py changed on a, and a union an earlier merge left with a as its parent."""
    directory = str(tmp_path / "held")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    _write(directory, "lib.py", "def foo():\n    return 1\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "-b", "a", base)
    _write(directory, "lib.py", "def foo():\n    return 2\n")
    _git(directory, "commit", "-qam", "a")
    tip = _git(directory, "rev-parse", "HEAD").strip()
    tree = _git(directory, "rev-parse", "a^{tree}").strip()
    union = _git(directory, "commit-tree", tree, "-p", base, "-p", tip, "-m", "u").strip()
    _git(directory, "branch", "union", union)
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


def test_a_branch_another_branch_holds_is_not_a_side_of_its_own(tmp_path):
    directory, base = _contained_repo(tmp_path)
    merged = integrate.assemble(directory, base, ["union", "a"])
    assert [e["branch"] for e in integrate.own_side_list(directory, merged["branches"])] \
        == ["union"]
    reversed_merged = integrate.assemble(directory, base, ["a", "union"])
    assert [e["branch"] for e in
            integrate.own_side_list(directory, reversed_merged["branches"])] == ["union"]


def test_of_two_branches_at_one_commit_the_first_is_kept(tmp_path):
    directory, base = _contained_repo(tmp_path)
    _git(directory, "branch", "a2", "a")
    merged = integrate.assemble(directory, base, ["a", "a2"])
    assert [e["branch"] for e in integrate.own_side_list(directory, merged["branches"])] \
        == ["a"]


def test_branches_neither_of_which_holds_the_other_are_both_sides(repo):
    directory, base = repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert [e["branch"] for e in integrate.own_side_list(directory, merged["branches"])] \
        == ["a", "b"]


def test_a_branch_the_union_holds_makes_no_overlap_and_calls_no_model(tmp_path, monkeypatch):
    """git adds nothing for it, so nothing more than one side changed is left."""
    directory, base = _contained_repo(tmp_path)
    lib = "p/lib_py/lib.py"
    start = dict(_analysis(range_dict={lib: [(1, 2, "foo")]}, name={lib: ["foo"]}),
                 use=[], signature={})
    merged = integrate.assemble(directory, base, ["union", "a"])
    assert merged["conflicted"] == []
    prepared = {"base": base, "merged": merged, "before": start, "against": start,
                "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(integrate, "analyse", lambda _: start)
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    out = integrate.run_integration(prepared, "", "p", "out")
    assert out["work"]["overlap"] == [] and out["called"] is False


# == The commits merged, the broken lines, the cut lists ===


def test_the_branch_merged_into_is_the_commit_read_not_where_it_points_now(repo):
    directory, base = repo
    _git(directory, "checkout", "-q", "-b", "tree", base)
    _write(directory, "tree1.txt", "read\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "read")
    read_commit = _git(directory, "rev-parse", "HEAD").strip()
    _write(directory, "tree2.txt", "after\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "moved on")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["a", "b"], onto="tree",
                                onto_commit=read_commit)
    assert merged["branches"][0]["commit"] == read_commit
    assert "tree2.txt" not in gitcmd.status_path_list(directory)
    assert integrate.branch_commit_list(merged) == [
        _git(directory, "rev-parse", name).strip() for name in ("a", "b")]


def test_the_commits_merged_are_every_branch_s_without_a_branch_merged_into(repo):
    directory, base = repo
    merged = integrate.assemble(directory, base, ["b", "a"])
    assert integrate.branch_commit_list(merged) == [
        _git(directory, "rev-parse", name).strip() for name in ("b", "a")]


def test_a_call_the_merge_broke_is_not_hidden_by_a_side_s_own_of_the_same_key():
    key = ("p/x_py/x.py", "p/lib_py/lib.py", "foo")
    before = _analysis(name={"p/lib_py/lib.py": ["foo"]})
    after = _analysis(no_resolve={key: [3, 40]}, name={"p/lib_py/lib.py": ["bar"]})
    broken = integrate.broken_list_of(before, after, "p", {key: [3]})
    assert [(entry["name"], entry["lines"]) for entry in broken] == [("foo", [3, 40])]
    assert integrate.broken_list_of(before, after, "p", {key: [3, 9]}) == []


def test_a_key_is_counted_by_the_most_lines_any_one_state_held():
    key = ("a", "b", "x")
    into = {key: [1, 2]}
    integrate.add_no_resolve(into, {key: [5]})
    assert into == {key: [1, 2]}
    integrate.add_no_resolve(into, {key: [5, 6, 7], ("a", "b", "y"): []})
    assert into == {key: [5, 6, 7], ("a", "b", "y"): []}


def test_a_key_held_with_no_line_is_still_one_reference():
    assert integrate.broken_ref_dict({}, {("a", "b", "x"): []}) == {("a", "b", "x"): []}
    assert integrate.broken_ref_dict({("a", "b", "x"): []}, {("a", "b", "x"): []}) == {}


def test_a_state_cut_when_it_was_built_stays_incomplete_on_later_turns(repo, monkeypatch):
    directory, _ = repo
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    monkeypatch.setattr(integrate, "marker_path_list", lambda directory, *_: [])
    merged = _merged(conflicted=["f1.txt", "f2.txt", "f3.txt"])
    broken = [{"file": f"{n}.py", "lines": [1], "name": "gone", "defined_in": "lib.py",
               "now_defines": []} for n in range(3)]
    work = integrate.build_work_dict(merged, [], broken, frozenset(), "", 0)
    assert work["complete"] is False
    turn_note, _ = integrate.make_state(
        directory, "HEAD", work, lambda: [],
        uncut_dict={"broken": broken, "overlap": [], "branches": merged["branches"]})
    turn_note(1, 20)
    assert work["complete"] is False
    fit = _merged(conflicted=["f1.txt"])
    work = integrate.build_work_dict(fit, [], broken, frozenset(), "", 0)
    turn_note, _ = integrate.make_state(
        directory, "HEAD", work, lambda: [],
        uncut_dict={"broken": broken[:1], "overlap": [], "branches": fit["branches"]})
    turn_note(1, 20)
    assert work["complete"] is True


# == Deleting a file in the merging stage ==================


def test_the_merging_stage_can_delete_the_file_one_side_deleted(unmarked_repo, monkeypatch):
    directory, base = unmarked_repo
    integrate.assemble(directory, base, ["a", "b"])
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "EDIT_DRY_RUN", False)
    edit.reset_patch_round(round_limit=0)
    tool_name_list = [tool.__name__ for tool in integrate.merge_tool_list()]
    assert tool_name_list.index("delete_file") == tool_name_list.index("create_file") + 1
    result = edit.delete_file("gone.txt")
    assert result["ok"] is True and result["stat"] == "deleted"
    assert not os.path.exists(os.path.join(directory, "gone.txt"))
    assert "gone.txt" in edit.write_path_set()
    assert "gone.txt" in _git(directory, "status", "--porcelain")
    assert edit.delete_file("gone.txt")["error"] == "gone.txt does not exist"
    os.makedirs(os.path.join(directory, "sub"))
    assert "is a directory" in edit.delete_file("sub")["error"]
    assert edit.delete_file("../x")["ok"] is False


def test_a_dry_run_deletes_nothing(unmarked_repo, monkeypatch):
    directory, base = unmarked_repo
    integrate.assemble(directory, base, ["a", "b"])
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "EDIT_DRY_RUN", True)
    edit.reset_patch_round(round_limit=0)
    assert edit.delete_file("gone.txt")["ok"] is True
    assert os.path.exists(os.path.join(directory, "gone.txt"))


def test_the_instructions_say_a_file_can_be_deleted():
    assert "delete_file" in integrate.build_integrate_instruction("py")


# == Paths read as they are written ========================


def test_a_path_holding_a_glob_character_reads_that_file_alone(tmp_path):
    directory, _ = _two_file_repo(tmp_path, "a*.txt", "x\n", "ab.txt", _BLOCK_TEXT,
                                  {"f.txt": "one\n"})
    _git(directory, "checkout", "-q", "b")
    _write(directory, "a*.txt", "y\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "both")
    commit = _git(directory, "rev-parse", "HEAD").strip()
    line_list = integrate._marker_line_dict(directory, [commit])[commit]
    assert integrate._separator_frame_dict(line_list, ["b"]) == {"ab.txt": {}}


# == Tests failing at HEAD when the union's did not run ====


def test_the_tests_not_counted_are_said_to_be_head_s_when_the_union_s_did_not_run(
        repo, monkeypatch):
    directory, base = repo
    out = _integration_of(directory, base, monkeypatch,
                          [_tested("did not run"), _tested("failed", ["t::one"])])
    assert out["work"]["already_failing"] == ["t::one"]
    line_list = integrate._test_start_line_list(out["work"])
    assert integrate._ALREADY_RED_AT_HEAD in line_list
    assert integrate._ALREADY_RED not in line_list


def test_the_tests_not_counted_are_said_to_still_fail_when_both_runs_named_them(
        repo, monkeypatch):
    directory, base = repo
    out = _integration_of(directory, base, monkeypatch,
                          [_tested("failed", ["t::one"]), _tested("failed", ["t::one"])])
    line_list = integrate._test_start_line_list(out["work"])
    assert integrate._ALREADY_RED in line_list
    assert integrate._ALREADY_RED_AT_HEAD not in line_list


@pytest.mark.parametrize("merged", [
    _tested("passed"), _tested("failed", ["t::bad"]), _tested("did not run")])
def test_the_tests_not_counted_are_said_to_be_unsettled_where_head_s_run_stopped(
        repo, monkeypatch, merged):
    """HEAD's run red and ended before every test was run: whatever the merged tree's
    run says, the start block says what the check says of it, and not that the tests
    did not run on the merged tree or that the others are counted."""
    directory, base = repo
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    head = dict(_tested("failed", ["t::bad"]), output=(
        "FAILED t::bad - assert 0\n!!!!!!!!!! stopping after 1 failures !!!!!!!!!!\n"
        "1 failed, 1 passed in 0.02s"))
    out = _integration_of(directory, base, monkeypatch, [dict(merged), head])
    assert out["work"]["already_failing"] == ["t::bad"]
    line_list = integrate._test_start_line_list(out["work"])
    assert line_list == ["", integrate._ALREADY_RED_AT_STOP.format(commands="`run them`"),
                         "  t::bad"]
    assert merged["status"] == "did not run" \
        or "stopped before it had run every test" in out["work"]["tests"]
    integrate.reset_stage()
    assert integrate._STAGE_DICT["head_stop"] == []


def _run_result(status, failed=(), ran_name_list=None, is_name_short=False):
    """Return a RunResult of a whole run, the tests it ran read off it."""
    result = edit.RunResult(_tested(status, failed))
    result.ran_count, result.ran_name_list = 2, ran_name_list
    result.is_name_short, result.is_ran_hold = is_name_short, True
    return result


@pytest.mark.parametrize("head_form", ["stop", "short", "named"])
def test_what_of_head_s_run_keeps_any_merge_from_being_settled_is_said(
        repo, monkeypatch, head_form):
    """HEAD's run red and ended before every test was run, or not naming each test it
    counts, holds the check of any merge over it unsettled; HEAD's run that names its
    red tests does not."""
    directory, base = repo
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    green = _run_result("passed", ran_name_list=["t::one", "t::two"])
    head, merged, block_list = {
        "stop": (dict(_tested("failed", ["t::bad"]), output=(
            "FAILED t::bad - assert 0\n!!!!!!!!!! stopping after 1 failures !!!!!!!!!!\n"
            "1 failed, 1 passed in 0.02s")), dict(_tested("passed")),
            [integrate.HEAD_STOP_BLOCK]),
        "short": (_run_result("failed", is_name_short=True), green,
                  [integrate.HEAD_SHORT_BLOCK]),
        "named": (_tested("failed", ["t::one"]), _tested("failed", ["t::two"]), []),
    }[head_form]
    out = _integration_of(directory, base, monkeypatch, [merged, head])
    assert out["head_block"] == integrate.head_block_list() == block_list
    integrate.reset_stage()
    assert integrate.head_block_list() == []


@pytest.mark.parametrize("head_form", ["stop", "short", "named"])
def test_the_state_and_the_start_say_no_run_settles_the_tests_where_head_s_run_keeps_them(
        repo, monkeypatch, head_form):
    """Where HEAD's run keeps any merge from being settled, the state block does not say
    that running the whole suite again moves what it said, and the start block says
    the tests HEAD's run named are no baseline the others are counted against."""
    directory, base = repo
    monkeypatch.setattr(config, "PRIMARY_EXT", "py")
    head, merged, start_title, test_title = {
        "stop": (dict(_tested("failed", ["t::bad"]), output=(
            "FAILED t::bad - assert 0\n!!!!!!!!!! stopping after 1 failures !!!!!!!!!!\n"
            "1 failed, 1 passed in 0.02s")), dict(_tested("passed")),
            integrate._ALREADY_RED_AT_STOP, integrate._TEST_HEAD_BLOCK),
        "short": (_run_result("failed", ["t::bad"], is_name_short=True),
                  _run_result("failed", ["t::bad"], ran_name_list=["t::bad", "t::two"]),
                  integrate._ALREADY_RED_AT_SHORT, integrate._TEST_HEAD_BLOCK),
        "named": (_tested("failed", ["t::bad"]), _tested("failed", ["t::bad", "t::two"]),
                  integrate._ALREADY_RED, integrate._TEST_BREAK),
    }[head_form]
    out = _integration_of(directory, base, monkeypatch, [merged, head])
    assert out["work"]["already_failing"] == ["t::bad"] and out["work"]["tests"]
    assert integrate._test_start_line_list(out["work"]) == [
        "", start_title.format(commands="`run them`"), "  t::bad"]
    assert integrate._test_line_list(out["work"])[:2] == [
        "", test_title.format(when=integrate._TEST_BEFORE_FIRST_TURN)]
    integrate.reset_stage()


# == The union an earlier merge left unsettled =============


def _union_branch(directory, base, branch_list, trailer="Codetwine-Loom-Verdict: unsettled"):
    """Commit the union of the branches on a branch named union, as the tool commits
    one, and leave the repository clean at base."""
    merged = integrate.assemble(directory, base, list(branch_list))
    tree = _git(directory, "rev-parse", f"{merged['commit']}^{{tree}}").strip()
    argv = ["commit-tree", tree, "-p", base]
    for parent in integrate.branch_commit_list(merged):
        argv += ["-p", parent]
    commit = _git(directory, *argv, "-m", f"integrate: x\n\ncombine\n\n{trailer}").strip()
    _git(directory, "branch", "union", commit)
    _git(directory, "reset", "-q", "--hard")
    return commit


def test_the_sides_of_an_earlier_union_are_read_only_where_it_was_left_unsettled(repo):
    directory, base = repo
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    assert integrate.earlier_side_list(directory, _union_commit(directory, [base, a, b])) \
        == [a, b]
    settled = _union_commit(directory, [base, a, b], trailer="Codetwine-Loom-Verdict: settled")
    assert integrate.earlier_side_list(directory, settled) == []
    assert integrate.earlier_side_list(directory, a) == []


def test_an_unsettled_union_carried_on_from_is_read_as_what_it_was_merged_from(repo):
    """The tree is the union's; the sides, the conflicts and the marker lines not
    counted are those of the commits it was merged from."""
    directory, base = repo
    union = _union_branch(directory, base, ["a", "b"])
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    # The union names no side as not merged into a file, so every commit it was made
    # from is held to each file whose change of it the union does not hold: the sides
    # to the file they left a block in. What it was written against changed nothing
    assert [(e["branch"], e["commit"], e["in_block"], bool(e.get("carry")))
            for e in merged["branches"]] == [
        (integrate.side_name(a), a, {"f.txt": "f.txt"}, False),
        (integrate.side_name(b), b, {"f.txt": "f.txt"}, False)]
    assert merged["branches"][1]["conflicted"] == ["f.txt"]
    assert merged["conflicted"] == ["f.txt"] and merged["unmarked"] == []
    assert "<<<<<<<" in _read(directory, "f.txt")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    assert integrate.branch_commit_list(merged) == [union]
    assert _git(directory, "for-each-ref", "refs/codetwine-loom") == ""


def test_a_settled_union_carried_on_from_is_one_side(repo):
    directory, base = repo
    _union_branch(directory, base, ["a", "b"], trailer="Codetwine-Loom-Verdict: settled")
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    assert [e["branch"] for e in merged["branches"]] == ["union"]
    assert merged["conflicted"] == []


def test_what_git_kept_one_side_of_is_read_again_for_an_unsettled_union(unmarked_repo):
    directory, base = unmarked_repo
    _union_branch(directory, base, ["a", "b"])
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    assert merged["conflicted"] == ["gone.txt", "res.rc"]
    assert merged["unmarked"] == ["gone.txt", "res.rc"]


def _resume_integration(directory, base, monkeypatch, side_define_dict=None):
    """Run run_integration() over the union carried on from, with the analysis stubbed
    and the stage ending at once, so the check after it says what is left."""
    prepared = {"base": base, "before": _analysis(), "against": _analysis(), "was_on": "",
                "merged": integrate.assemble(directory, base, ["union"], resume="union"),
                "side_define": side_define_dict or {}}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "INTEGRATE_MAX_REFUSAL", 0)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatRuns)
    return integrate.run_integration(prepared, "", "p", "out")


def test_an_unsettled_union_carried_on_from_is_not_settled_by_git_alone(repo, monkeypatch):
    directory, base = repo
    _union_branch(directory, base, ["a", "b"])
    out = _resume_integration(directory, base, monkeypatch)
    assert out["called"] is True and out["settled"] is False
    assert any("conflict markers" in reason and "f.txt" in reason for reason in out["left"])
    assert any("nothing in this run has read" in reason for reason in out["left"])


def test_a_name_a_side_of_an_unsettled_union_added_is_looked_for(repo, monkeypatch):
    directory, base = repo
    _union_branch(directory, base, ["a", "b"])
    name = integrate.side_name(_git(directory, "rev-parse", "b").strip())
    out = _resume_integration(directory, base, monkeypatch,
                              {name: {"from_b": ["p/b_only_txt/b_only.txt"]}})
    assert any(f"from_b, which {name} added or changed in b_only.txt" in reason
               for reason in out["left"])


def test_a_branch_that_does_not_start_from_the_start_is_not_read_as_a_side(repo, caplog):
    directory, base = repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    a = _git(directory, "rev-parse", "a").strip()
    assert [e["branch"] for e in integrate.own_side_list(directory, merged["branches"], a)] \
        == ["a"]
    assert "b does not start from" in caplog.text
    assert len(integrate.own_side_list(directory, merged["branches"], base)) == 2


# == What another side took away ===========================


def test_a_name_another_side_renamed_is_not_looked_for_and_one_it_moved_is():
    lib, core = "p/lib_py/lib.py", "p/core_py/core.py"
    start = _analysis(name={lib: ["foo", "keep"]})
    moved = integrate.keep_list_of(
        [("a", {lib: {"foo"}}), ("b", {lib: {"", "foo", "keep"}, core: {""}})], start,
        {"a": {"foo": [lib], "keep": [lib]}, "b": {"foo": [core], "keep": [core]}})
    assert moved == [("a", {"foo": [lib]}), ("b", {"foo": [core], "keep": [core]})]
    renamed = integrate.keep_list_of(
        [("a", {lib: {"foo"}}), ("b", {lib: {"foo"}})], start,
        {"a": {"foo": [lib], "keep": [lib]}, "b": {"bar": [lib], "keep": [lib]}})
    assert renamed == [("a", {}), ("b", {"bar": [lib]})]
    removed = integrate.keep_list_of(
        [("a", {lib: {"foo", "keep"}}), ("b", {lib: {"", "foo", "keep"}})], start,
        {"a": {"foo": [lib], "keep": [lib]}, "b": {}})
    assert removed == [("a", {}), ("b", {})]


def test_a_name_a_side_added_is_looked_for_whatever_the_other_sides_hold():
    lib = "p/lib_py/lib.py"
    start = _analysis(name={lib: ["foo"]})
    keep_list = integrate.keep_list_of(
        [("a", {lib: {""}}), ("b", {"p/other_py/other.py": {""}})], start,
        {"a": {"foo": [lib], "new_fn": [lib]}, "b": {"foo": [lib]}})
    assert keep_list == [("a", {"new_fn": [lib]}), ("b", {})]
    # The other side removed the file, and with it every name the start defines
    assert integrate.keep_list_of(
        [("a", {lib: {""}}), ("b", {lib: {"", "foo"}})], start,
        {"a": {"foo": [lib], "new_fn": [lib]}, "b": {}}) == [("a", {"new_fn": [lib]}), ("b", {})]


def test_a_side_with_no_analysis_of_its_tip_holds_no_name_and_takes_none_away(caplog):
    caplog.set_level(logging.INFO, logger="codetwine_loom.integrate")
    lib = "p/lib_py/lib.py"
    keep_list = integrate.keep_list_of(
        [("a", {lib: {"foo"}}), ("b", {lib: {"foo"}})], _analysis(name={lib: ["foo"]}),
        {"a": {"foo": [lib]}})
    assert keep_list == [("a", {"foo": [lib]}), ("b", {})]
    assert "b have no analysis of their own tip" in caplog.text


def test_what_the_tip_merged_into_removed_is_not_looked_for_in_the_union(repo, monkeypatch):
    """main deleted g.txt after the branches started; the names the start defines
    there are not a's or b's to keep, and not main's either."""
    directory, base = repo
    _git(directory, "checkout", "-q", "main")
    _git(directory, "rm", "-q", "g.txt")
    _git(directory, "commit", "-qm", "main drops g.txt")
    _git(directory, "checkout", "-q", "--detach", base)
    g_file = "p/g_txt/g.txt"
    start = _analysis(range_dict={g_file: [(1, 1, "old_helper")]}, name={g_file: ["old_helper"]})
    prepared = {"base": base, "before": start, "against": _analysis(), "was_on": "",
                "merged": integrate.assemble(directory, base, ["a", "b"], onto="main")}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    measure = integrate._measure_union(prepared, "", "p", directory)
    assert [branch for branch, _ in measure["side"]] == ["main", "a", "b"]
    assert dict(measure["side"])["main"] == {g_file: {"", "old_helper"}}
    assert measure["keep"] == [("main", {}), ("a", {}), ("b", {})]


# == Markers in a file git does not diff ===================


def test_a_marker_in_a_file_the_attributes_keep_out_of_diffs_is_found(tmp_path):
    directory, base = _two_file_repo(
        tmp_path, "deps.lock", "alpha\nbeta 2\n", "deps.lock", "alpha\nbeta 3\n",
        {"deps.lock": "alpha\nbeta 1\n", ".gitattributes": "*.lock -diff\n"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == ["deps.lock"] and merged["unmarked"] == []
    assert ("deps.lock", b"=======") in integrate._marker_line_dict(
        directory, [merged["commit"]])[merged["commit"]]
    assert integrate.marker_path_list(directory, merged["marker"]) == ["deps.lock"]
    _write(directory, "deps.lock", "alpha\nbeta 3\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_separator_line_in_a_file_git_wrote_no_block_in_is_not_counted(repo):
    directory, base = repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    _write(directory, "f.txt", "ONE\ntwo\nthree\n")
    _write(directory, "doc.txt", "Title\n=======\n||||||| base\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []
    _write(directory, "doc.txt", "Title\n=======\nmore\n>>>>>>> refs/codetwine-loom/side/b\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["doc.txt"]


# == The references the merge broke, past the count shown ==


def test_every_reference_the_merge_broke_is_counted_past_the_work_list_count(monkeypatch):
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 2)
    after = _analysis(no_resolve={(f"p/u{n}_py/u{n}.py", "p/lib_py/lib.py", "gone"): [1]
                                  for n in range(3)})
    broken = integrate.broken_list_of(_analysis(), after, "p")
    assert len(broken) == 3
    work = integrate.build_work_dict(_merged(), [], broken, frozenset(), "", 3)
    assert len(work["broken"]) == 2 and work["complete"] is False


# == Work the files hold without a commit ==================


def test_work_put_into_the_files_without_a_commit_is_held(repo):
    directory, base = repo
    _committed(directory, base, "left", "sandbox: x\n\nadd the flag")
    assert integrate.branch_record_dict(directory, "left")["held"] is False
    _write(directory, "left.txt", "left\n")
    assert integrate.branch_record_dict(directory, "left")["held"] is True


def test_work_held_up_to_a_commit_under_a_stage_s_is_held(repo):
    """The files hold what the merge made; the commit of the stage above it was not
    put into them."""
    directory, base = repo
    _committed(directory, base, "staged",
               "integrate: x\n\nadd it\n\nCodetwine-Loom-Verdict: settled")
    _git(directory, "checkout", "-q", "staged")
    _write(directory, "stage.txt", "stage\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm",
         "refactor: x\n\nadd it\n\nCodetwine-Loom-Stage: refactor\n"
         "Codetwine-Loom-Verdict: unsettled")
    _git(directory, "checkout", "-q", "--detach", base)
    assert integrate.branch_record_dict(directory, "staged")["held"] is False
    _write(directory, "staged.txt", "staged\n")
    assert integrate.branch_record_dict(directory, "staged")["held"] is True



# == One reading of what keeps a merge from being settled ====


def test_every_measure_is_a_reason_and_none_of_them_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    _write(str(tmp_path), "g.txt", "x\n")
    integrate.reset_stage()
    lib = "p/lib_py/lib.py"
    reason_list = integrate._unsettle_reason_list(
        ["f.txt"], ["- the check line"],
        [{"name": "gone", "file": "use.py"}], [("a", {"new_fn": [lib]})], {"foo": [lib]},
        "p", ["g.txt"], "- the test line", ["CONFLICT (x): no file"],
        ["d~refs_codetwine-loom_merged-so-far"])
    assert [reason.split(" ")[1] for reason in reason_list] == [
        "git", "conflict", "these", "the", "these", "new_fn,", "g.txt", "the"]
    assert "CONFLICT (x): no file" in reason_list[0]
    assert "d~refs_codetwine-loom_merged-so-far" in reason_list[2] \
        and "stand-in name" in reason_list[2]
    assert integrate._unsettle_reason_list([], [], [], [("a", {"foo": [lib]})],
                                           {"foo": [lib]}, "p", [], "", [], []) == []


def test_a_file_the_sides_met_at_is_left_until_a_model_has_read_it(repo, monkeypatch):
    """The conflict is settled in the files and nothing has read it in this run."""
    directory, base = repo
    prepared = {"base": base, "merged": integrate.assemble(directory, base, ["a", "b"]),
                "before": _analysis(), "against": _analysis(), "was_on": ""}
    _write(directory, "f.txt", "settled\n")
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "INTEGRATE_MAX_REFUSAL", 0)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatRuns)
    out = integrate.run_integration(prepared, "", "p", "out")
    assert out["called"] is True and out["settled"] is False
    assert out["left"] == [("- f.txt hold what the sides both reached and nothing in this "
                            "run has read them")]


# Stands for the record a union's first run writes of the fixture's one conflicted file
_NO_RECORD = object()


def _union_branch_with(directory, base, file_dict, record_dict=_NO_RECORD,
                       branch="union"):
    """Commit a union of a and b on a branch whose tree holds a's files with file_dict
    written over them, recorded as left unsettled with the separator lines it may hold:
    none in f.txt, the one file a and b conflict in, unless record_dict says otherwise.
    None records none. A union that records its separator lines records no side as
    not merged into a file as well."""
    if record_dict is _NO_RECORD:
        record_dict = {"f.txt": {}}
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    _git(directory, "checkout", "-q", "--detach", a)
    for name, text in file_dict.items():
        _write(directory, name, text)
    _git(directory, "add", "-A")
    tree = _git(directory, "write-tree").strip()
    _git(directory, "reset", "-q", "--hard")
    _git(directory, "checkout", "-q", "--detach", base)
    trailer = "" if record_dict is None else (
        f"\n{integrate.SEPARATOR_TRAILER}: {integrate.separator_text(record_dict)}"
        f"\n{integrate.NO_MERGE_TRAILER}: {integrate.no_merge_text([])}")
    commit = _git(directory, "commit-tree", tree, "-p", base, "-p", a, "-p", b, "-m",
                  f"integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: unsettled{trailer}").strip()
    _git(directory, "branch", branch, commit)
    return commit


def test_a_name_an_earlier_settling_threw_away_is_found_before_a_model_is_called(
        repo, monkeypatch):
    """The union carried on from holds no marker and no definition two sides reached."""
    directory, base = repo
    _union_branch_with(directory, base, {"f.txt": "ONE and 1\ntwo\nthree\n",
                                            "b_only.txt": "emptied\n"})
    name = integrate.side_name(_git(directory, "rev-parse", "b").strip())
    out = _resume_integration(directory, base, monkeypatch,
                              {name: {"from_b": ["p/b_only_txt/b_only.txt"]}})
    assert out["work"]["markers"] == [] and out["work"]["overlap"] == []
    assert out["called"] is True and out["settled"] is False
    assert any(f"from_b, which {name} added or changed in b_only.txt" in reason
               for reason in out["left"])


# == A settled union under the one carried on from ==========


def _stage_commit(directory, commit, name, text):
    """Write a settled refactoring stage's commit over a commit and return it."""
    _git(directory, "checkout", "-q", "--detach", commit)
    _write(directory, name, text)
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm",
         "refactor: x\n\ntidy\n\nCodetwine-Loom-Stage: refactor\nCodetwine-Loom-Verdict: settled")
    return _git(directory, "rev-parse", "HEAD").strip()


def test_a_settled_union_under_an_unsettled_one_is_one_side_with_its_stages(repo):
    directory, base = repo
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    settled = _union_commit(directory, [base, a, b], trailer="Codetwine-Loom-Verdict: settled")
    stage = _stage_commit(directory, settled, "tidy.txt", "tidy\n")
    _git(directory, "checkout", "-q", "-b", "c", base)
    _write(directory, "c_only.txt", "c\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "c")
    c = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "--detach", base)
    second = _union_commit(directory, [base, stage, c])
    assert integrate.union_side_list(directory, second) == [stage, c]
    assert integrate.earlier_side_list(directory, second) == [stage, c]
    # The same union over one left unsettled reaches the branches under both
    open_stage = _stage_commit(directory, _union_commit(directory, [base, a, b]),
                               "tidy.txt", "tidy\n")
    _git(directory, "checkout", "-q", "--detach", base)
    assert integrate.earlier_side_list(
        directory, _union_commit(directory, [base, open_stage, c])) == [a, b, c]
    assert integrate.earlier_side_list(directory, stage) == []


def test_a_settled_union_carried_on_from_is_analysed_as_the_one_side_it_is(repo, monkeypatch):
    directory, base = repo
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    _git(directory, "branch", "union",
         _union_commit(directory, [base, a, b], trailer="Codetwine-Loom-Verdict: settled"))
    union = _git(directory, "rev-parse", "union").strip()
    key = ("p/x_py/x.py", "p/lib_py/lib.py", "foo")
    order = []

    def analyse_head(project_dir, *rest):
        order.append(_git(project_dir, "rev-parse", "HEAD").strip())
        return {"no_resolve": {key: [5]}, "define": {}}

    monkeypatch.setattr(integrate, "_analyse_head", analyse_head)
    side_dict, define_dict, _ = integrate._side_analysis(directory, "out", base, ["union"],
                                                        None, resume="union")
    assert order == [union] and list(define_dict) == ["union"]
    assert side_dict == {key: [5]}


# == A name another side took the place of away ============


def test_a_name_a_side_added_is_looked_for_once_its_place_is_gone():
    lib, core = "p/lib_py/lib.py", "p/core_py/core.py"
    moved = integrate.keep_list_of(
        [("a", {lib: {"foo"}}), ("b", {lib: {"", "foo"}, core: {""}})],
        _analysis(name={lib: ["foo"]}),
        {"a": {"foo": [lib], "new_fn": [lib]}, "b": {"foo": [core]}})
    assert moved == [("a", {"foo": [lib], "new_fn": [lib]}), ("b", {"foo": [core]})]
    # The class around the method was renamed by the other side
    renamed = integrate.keep_list_of(
        [("a", {lib: {"A.run"}}), ("b", {lib: {"A"}})], _analysis(name={lib: ["A", "A.run"]}),
        {"a": {"A": [lib], "extra": [lib], "run": [lib]}, "b": {"B": [lib], "run": [lib]}})
    assert renamed == [("a", {"extra": [lib], "run": [lib]}), ("b", {"B": [lib]})]
    # The start defines nothing in the file the other side removed
    removed = integrate.keep_list_of(
        [("a", {"p/s_py/s.py": {""}}), ("b", {"p/s_py/s.py": {""}})], _analysis(),
        {"a": {"new_fn": ["p/s_py/s.py"]}, "b": {}})
    assert removed == [("a", {"new_fn": ["p/s_py/s.py"]}), ("b", {})]


def test_a_name_is_found_in_any_file_by_its_own_name_or_is_a_reason():
    lib, core = "p/lib_py/lib.py", "p/core_py/core.py"
    keep_list = [("a", {"extra": [lib], "new_fn": [lib, core]})]
    union = _analysis(name={core: ["B", "B.extra", "new_fn"]})
    assert integrate._drop_reason_list(keep_list, union["define"], "p") == []
    union = _analysis(name={core: ["B", "B.extra"]})
    assert integrate._drop_reason_list(keep_list, union["define"], "p") == [
        "- new_fn, which a added or changed in lib.py, core.py, is not defined in any file now"]
    assert len(integrate._drop_reason_list(keep_list, {}, "p")) == 2


# == A side that starts from a commit of its own ============


def test_a_side_is_read_against_the_names_of_the_commit_it_starts_from():
    """The start the other sides share renamed old after a branched; a changed old."""
    lib = "p/lib_py/lib.py"
    start = _analysis(name={lib: ["foo", "renamed"]})
    tip_dict = {"a": {"foo": [lib], "from_a": [lib], "old": [lib]}}
    keep_list = integrate.keep_list_of(
        [("a", {lib: {"old"}})], start, tip_dict, {"a": _analysis(name={lib: ["foo", "old"]})})
    assert keep_list == [("a", {"from_a": [lib]})]
    # Read against the shared start, old would be a name a added
    assert integrate.keep_list_of([("a", {lib: {""}})], start, tip_dict) == [
        ("a", {"from_a": [lib], "old": [lib]})]


def test_the_commit_a_side_starts_from_is_analysed_once_where_it_is_not_the_base(
        repo, monkeypatch):
    directory, base = repo
    moved = _move_main_on(directory, base)
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    order = []

    def analyse_head(project_dir, *rest):
        order.append(_git(project_dir, "rev-parse", "HEAD").strip())
        return _analysis(name={"p/s_py/s.py": ["s"]})

    monkeypatch.setattr(integrate, "_analyse_head", analyse_head)
    _, _, start_dict = integrate._side_analysis(directory, "out", moved, ["a", "b"], None)
    assert order == [base, a, b]
    assert start_dict == {name: {"commit": base, "range": {}, "own": {"p/s_py/s.py": {"s": "s"}},
                                 "define": {"s": ["p/s_py/s.py"]}}
                          for name in ("a", "b")}
    # A side that starts from the base, and one holding the tree the base holds
    again = _git(directory, "commit-tree", f"{moved}^{{tree}}", "-p", base, "-m",
                 "the same tree").strip()
    for commit in (moved, again):
        assert integrate._side_start_dict(directory, "out", moved, commit, None, {}) is None


def test_a_side_with_a_start_of_its_own_is_measured_from_that_start(repo, monkeypatch,
                                                                    caplog):
    directory, base = repo
    moved = _move_main_on(directory, base)
    _git(directory, "checkout", "-q", "--detach", moved)
    a_file = "p/a_only_txt/a_only.txt"
    prepared = {"base": moved, "before": _analysis(), "against": _analysis(), "was_on": "",
                "merged": integrate.assemble(directory, moved, ["a", "b"]),
                "side_define": {"a": {"from_a": [a_file]}},
                "side_start": {"a": {"commit": base, "range": {}, "own": {}, "define": {}}}}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    measure = integrate._measure_union(prepared, "", "p", directory)
    assert measure["side"] == [("a", {"p/f_txt/f.txt": {""}, a_file: {""}})]
    assert measure["keep"] == [("a", {"from_a": [a_file]}),
                               (integrate._START_BRANCH_TEXT, {})]
    assert "b does not start from" in caplog.text and "a does not" not in caplog.text


def test_a_branch_holding_the_tree_of_the_start_is_no_side_and_no_warning(repo, caplog):
    directory, base = repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    a = _git(directory, "rev-parse", "a").strip()
    again = _git(directory, "commit-tree", f"{a}^{{tree}}", "-p", base, "-m", "again").strip()
    entry_list = [merged["branches"][0], {"branch": "tree", "commit": again}]
    caplog.set_level(logging.INFO, logger="codetwine_loom.integrate")
    assert integrate.own_side_list(directory, entry_list, a) == [merged["branches"][0]]
    assert "tree holds what" in caplog.text and "does not start from" not in caplog.text
    assert integrate.own_side_list(directory, merged["branches"], a, {"b"}) == \
        merged["branches"]


# == Marker lines read as git holds them ===================


def _byte_repo(tmp_path, base_dict, a_dict, b_dict, config_list=()):
    """A repository whose branches a and b write files given as bytes over a base."""
    directory = str(tmp_path / "byte")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    base = ""
    for branch, file_dict in (("main", base_dict), ("a", a_dict), ("b", b_dict)):
        if base:
            _git(directory, "checkout", "-q", "-b", branch, base)
        for name, data in file_dict.items():
            _write_raw(directory, os.fsencode(name), data)
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", branch)
        base = base or _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "--detach", base)
    for key, value in config_list:
        _git(directory, "config", key, value)
    return directory, base


def test_a_marker_in_a_file_with_a_working_tree_encoding_is_found(tmp_path, monkeypatch):
    def encode(text):
        return text.encode("utf-16")

    directory, base = _byte_repo(
        tmp_path, {".gitattributes": b"*.ps1 text working-tree-encoding=UTF-16\n",
                   "s.ps1": encode("one\r\ntwo\r\nthree\r\n")},
        {"s.ps1": encode("one\r\nTWO a\r\nthree\r\n")},
        {"s.ps1": encode("one\r\ntwo b\r\nthree\r\n")})
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == ["s.ps1"] and merged["unmarked"] == []
    assert integrate.marker_path_list(directory, merged["marker"]) == ["s.ps1"]
    prepared = {"base": base, "merged": merged, "before": _analysis(),
                "against": _analysis(), "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatRuns)
    assert integrate.run_integration(prepared, "", "p", "out")["called"] is True
    _write_raw(directory, b"s.ps1", encode("one\r\nTWO a b\r\nthree\r\n"))
    assert integrate.marker_path_list(directory, merged["marker"]) == []


@pytest.mark.parametrize("key", ["color.ui", "color.grep"])
def test_the_marker_lines_are_read_whatever_colour_the_user_s_git_writes(tmp_path, key):
    directory, base = _byte_repo(
        tmp_path, {"n.md": b"one\ntwo\nthree\n"},
        {"n.md": b"one\nTWO a\nthree\n", "fixture.txt": _FIXTURE_TEXT.encode()},
        {"n.md": b"one\ntwo b\nthree\n"}, [(key, "always")])
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["marker"]["name_count"] == {_FIXTURE_START: 1, _FIXTURE_END: 1}
    assert merged["conflicted"] == ["n.md"] and merged["unmarked"] == []
    assert integrate.marker_path_list(directory, merged["marker"]) == ["n.md"]


_HEADING_TEXT = b"Title 1\n=======\n\ntext one\n\nTitle 2\n=======\n\ntext two\n"


def test_a_marker_line_left_of_a_block_is_counted_beside_the_same_line_held_before(
        tmp_path):
    directory, base = _byte_repo(tmp_path, {"doc.rst": _HEADING_TEXT},
                                 {"doc.rst": _HEADING_TEXT.replace(b"text one", b"by a")},
                                 {"doc.rst": _HEADING_TEXT.replace(b"text one", b"by b")})
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["marker"]["separator"] == {"doc.rst": {b"=======": 2}}
    assert merged["unmarked"] == []
    assert integrate.marker_path_list(directory, merged["marker"]) == ["doc.rst"]
    # The block is settled but for the line between its two sides
    _write_raw(directory, b"doc.rst",
               _HEADING_TEXT.replace(b"text one", b"by a\n=======\nby b"))
    assert integrate.marker_path_list(directory, merged["marker"]) == ["doc.rst"]
    _write_raw(directory, b"doc.rst", _HEADING_TEXT.replace(b"text one", b"by a and b"))
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_the_same_marker_line_two_sides_each_added_to_a_file_is_neither_s_to_settle(
        tmp_path):
    body = b"one\n\ntwo\n\nthree\n\nfour\n\nfive\n\nsix\n"
    directory, base = _byte_repo(tmp_path, {"doc.rst": body},
                                 {"doc.rst": b"Section\n=======\n\n" + body},
                                 {"doc.rst": body + b"\nLicense\n=======\n"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == []
    assert merged["marker"]["separator"] == {}
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_the_marker_lines_are_read_without_a_path_passed_to_git(repo, monkeypatch):
    """Every file is read in one command however many the union changes."""
    directory, base = repo
    grep_list = []
    real = gitcmd.run

    def watched(project_dir, *args, **kwargs):
        if "grep" in args:
            grep_list.append(args)
        return real(project_dir, *args, **kwargs)

    monkeypatch.setattr(integrate.gitcmd, "run", watched)
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    assert grep_list and all(args[-1] == "--" and "--no-color" in args for args in grep_list)


def test_a_link_and_a_removed_file_hold_no_marker_line(repo):
    directory, base = repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    _write(directory, "held.txt", _FIXTURE_TEXT)
    os.remove(os.path.join(directory, "f.txt"))
    assert integrate.marker_path_list(directory, merged["marker"]) == ["held.txt"]
    os.symlink("held.txt", os.path.join(directory, "f.txt"))
    assert integrate.marker_path_list(directory, merged["marker"]) == ["held.txt"]


def test_a_tree_whose_marker_lines_cannot_be_read_is_an_error(repo):
    directory, _ = repo
    with pytest.raises(RuntimeError, match="could not read the conflict marker lines"):
        integrate._marker_line_dict(directory, ["0" * 40])


# == Markers of a length the attributes give ================


def test_a_marker_as_long_as_the_file_s_attribute_says_is_found(tmp_path):
    """git writes the markers of such a file that long, and they are read that long."""
    directory, base = _two_file_repo(
        tmp_path, "f.txt", "ONE\ntwo\n", "f.txt", "1\ntwo\n",
        {"f.txt": "one\ntwo\n", ".gitattributes": "*.txt conflict-marker-size=32\n"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert _read(directory, "f.txt").startswith("<" * 32 + " ")
    assert merged["marker"]["size"] == {"f.txt": 32}
    assert merged["conflicted"] == ["f.txt"] and merged["unmarked"] == []
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    # One separator of the block left behind is still a marker of the merge's
    _write(directory, "f.txt", "ONE\n" + "=" * 32 + "\ntwo\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    # A line of seven is no marker in that file
    _write(directory, "f.txt", "ONE\n=======\ntwo\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_the_marker_length_is_the_one_in_effect_when_the_merge_is_made(tmp_path):
    """A length only a branch's own attribute file gives is not what git merged with."""
    directory, base = _two_file_repo(
        tmp_path, ".gitattributes", "*.txt conflict-marker-size=12\n", "g.txt", "b\n",
        {"f.txt": "one\ntwo\n"})
    _git(directory, "checkout", "-q", "a")
    _write(directory, "f.txt", "ONE\ntwo\n")
    _git(directory, "commit", "-qam", "a changes f")
    _git(directory, "checkout", "-q", "b")
    _write(directory, "f.txt", "1\ntwo\n")
    _git(directory, "commit", "-qam", "b changes f")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert _read(directory, "f.txt").startswith("<<<<<<< ")
    assert merged["marker"]["size"] == {} and merged["unmarked"] == []
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]


@pytest.mark.parametrize("value, size", [("9", 9), ("12abc", 12), ("0", None), ("-3", None),
                                         ("abc", None), ("7", None),
                                         ("1" * 5000, None)])
def test_the_marker_length_is_the_number_the_attribute_starts_with(tmp_path, value, size):
    directory, base = _two_file_repo(
        tmp_path, "f.txt", "a\n", "g.txt", "b\n",
        {"f.txt": "one\n", ".gitattributes": f"f.txt conflict-marker-size={value}\n"})
    assert integrate.marker_size_dict(directory, [base]) == ({"f.txt": size} if size else {})


def test_a_file_s_marker_lines_are_counted_by_its_own_length(tmp_path):
    directory, _ = _two_file_repo(
        tmp_path, "f.txt", "=======\n=========\n========= x\n==========\n", "g.txt", "b\n")
    commit = _git(directory, "rev-parse", "a").strip()
    assert integrate._marker_line_dict(directory, [commit], size_dict={"f.txt": 9}) == {
        commit: [("f.txt", b"========="), ("f.txt", b"========= x")]}
    assert integrate._marker_line_dict(directory, [commit]) == {
        commit: [("f.txt", b"=======")]}


# == How many marker lines the union holds on its own =======


def test_a_marker_line_a_side_took_away_is_not_one_the_union_may_hold(tmp_path):
    """The separator of a block left in the file is not taken for the line a removed."""
    directory, base = _two_file_repo(
        tmp_path, "doc.txt", "top\nmid\nLOW a\nend\n",
        "doc.txt", "top\n=======\nmid\nLOW b\nend\n",
        {"doc.txt": "top\n=======\nmid\nlow\nend\n"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == ["doc.txt"]
    assert merged["marker"]["separator"] == {"doc.txt": {b"=======": 0}}
    _write(directory, "doc.txt", "top\nmid\nLOW a\n=======\nLOW b\nend\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["doc.txt"]
    _write(directory, "doc.txt", "top\nmid\nLOW a and b\nend\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_marker_line_two_sides_hold_as_one_is_counted_once(tmp_path):
    """b starts from a, so the line a added is the same line on both."""
    directory, base = _two_file_repo(
        tmp_path, "doc.txt", "top\n=======\nmid\nlow\nend\n", "g.txt", "c\n",
        {"doc.txt": "top\nmid\nlow\nend\n"})
    _git(directory, "branch", "-m", "b", "c")
    _git(directory, "checkout", "-q", "c")
    _write(directory, "doc.txt", "top\nmid\nLOW c\nend\n")
    _git(directory, "commit", "-qam", "c changes low")
    _git(directory, "checkout", "-q", "-b", "b", "a")
    _write(directory, "doc.txt", "top\n=======\nmid\nLOW b\nend\n")
    _git(directory, "commit", "-qam", "b changes low")
    _git(directory, "checkout", "-q", "--detach", base)
    for order in (["a", "b", "c"], ["b", "a", "c"], ["c", "a", "b"]):
        _git(directory, "reset", "-q", "--hard", base)
        merged = integrate.assemble(directory, base, order)
        assert merged["marker"]["separator"] == {"doc.txt": {b"=======": 1}}, order
        _write(directory, "doc.txt", "top\n=======\nmid\nLOW b\n=======\nLOW c\nend\n")
        assert integrate.marker_path_list(directory, merged["marker"]) == ["doc.txt"]


def test_a_settled_union_and_a_branch_of_one_of_its_sides_hold_a_marker_line_once(tmp_path):
    directory, base = _two_file_repo(
        tmp_path, "doc.txt", "top\n=======\nmid\nlow\nend\n", "g.txt", "b\n",
        {"doc.txt": "top\nmid\nlow\nend\n"})
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    tree = _git(directory, "merge-tree", "--write-tree", a, b).strip()
    union = _git(directory, "commit-tree", tree, "-p", base, "-p", a, "-p", b, "-m",
                 "integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: settled").strip()
    _git(directory, "branch", "union", union)
    _git(directory, "checkout", "-q", "-b", "a2", "a")
    _write(directory, "doc.txt", "top\n=======\nmid\nLOW a2\nend\n")
    _git(directory, "commit", "-qam", "a2")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["union", "a2"], resume="union")
    assert merged["conflicted"] == []
    assert merged["marker"]["separator"] == {}
    assert integrate.marker_path_list(directory, merged["marker"]) == []


# == A conflict git names no conflicted file for ============


def _renamed_dir_repo(tmp_path, a_move, b_file_dict):
    """A repository whose branch a moves the files of old/ and whose branch b adds files.

    Args:
        a_move: {a file of old/: the directory a moves it to}.
        b_file_dict: {path: text} b adds.
    """
    directory = str(tmp_path / "renamed")
    os.makedirs(os.path.join(directory, "old"))
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    for name in a_move:
        _write(directory, f"old/{name}",
               "".join(f"{name} line {n}\n" for n in range(20)))
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "-b", "a", base)
    for name, target in a_move.items():
        os.makedirs(os.path.join(directory, target), exist_ok=True)
        _git(directory, "mv", f"old/{name}", f"{target}/{name}")
    _git(directory, "commit", "-qm", "a")
    _git(directory, "checkout", "-q", "-b", "b", base)
    for name, text in b_file_dict.items():
        os.makedirs(os.path.dirname(os.path.join(directory, name)), exist_ok=True)
        _write(directory, name, text)
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "b")
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


def test_a_conflict_over_a_directory_is_held_to_the_files_under_it(tmp_path):
    """a moved old/ to two directories; git cannot tell where b's new file goes."""
    directory, _ = _renamed_dir_repo(
        tmp_path, {"f1.txt": "d1", "f2.txt": "d1", "f3.txt": "d2", "f4.txt": "d2"},
        {"old/c.txt": "new in b\n"})
    _, conflicted, messages, no_path = integrate._merge_tree(directory, "a", "b")
    assert conflicted == ["old/c.txt"] and no_path == []
    assert len(messages) == 1 and "Unclear where to rename old" in messages[0]


def test_a_conflict_naming_no_conflicted_file_is_held_to_the_files_it_names(tmp_path):
    """b's own new/x.txt is in the way of the old/x.txt it added."""
    directory, base = _renamed_dir_repo(
        tmp_path, {"f1.txt": "new", "f2.txt": "new"},
        {"old/x.txt": "from b\n", "new/x.txt": "in the way\n"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == ["new/x.txt", "old/x.txt"]
    assert merged["unmarked"] == ["new/x.txt", "old/x.txt"] and merged["no_path"] == []
    assert integrate.meet_path_list(merged, []) == ["new/x.txt", "old/x.txt"]


def test_a_merge_git_ends_with_a_conflict_always_holds_one(repo, monkeypatch):
    """A record of a type that is not known, and no record at all, are both a conflict."""
    directory, base = repo
    real_run = gitcmd.run
    tail = {"records": [b"1", b"nowhere", b"A later type", b"something new\n"]}

    def run(project_dir, *args, **kwargs):
        proc = real_run(project_dir, *args, **kwargs)
        if "merge-tree" in args:
            proc.returncode = 1
            proc.stdout = proc.stdout.rstrip(b"\0") + b"\0\0" + b"".join(
                record + b"\0" for record in tail["records"])
        return proc

    monkeypatch.setattr(gitcmd, "run", run)
    assert integrate._merge_tree(directory, base, "a")[1:] == (
        [], ["something new"], ["something new"])
    tail["records"] = [b"1", b"a_only.txt", b"Auto-merging", b"Auto-merging a_only.txt\n"]
    assert integrate._merge_tree(directory, base, "a")[1:] == (
        [], [integrate._NO_WORD_CONFLICT], [integrate._NO_WORD_CONFLICT])
    tail["records"] = [b"1", b"a_only.txt", b"A later type", b"about a_only.txt\n"]
    assert integrate._merge_tree(directory, base, "a")[1:] == (
        ["a_only.txt"], ["about a_only.txt"], [])


def test_a_known_record_of_a_clean_merge_is_no_conflict(repo, monkeypatch):
    directory, base = repo
    real_run = gitcmd.run

    def run(project_dir, *args, **kwargs):
        proc = real_run(project_dir, *args, **kwargs)
        if "merge-tree" in args:
            proc.stdout = proc.stdout.rstrip(b"\0") + b"\0\0" + b"".join(
                record + b"\0" for record in
                [b"1", b"a_only.txt", b"A later type", b"about a_only.txt\n"])
        return proc

    monkeypatch.setattr(gitcmd, "run", run)
    assert integrate._merge_tree(directory, base, "a")[1:] == ([], [], [])


def test_a_conflict_naming_no_file_keeps_the_merge_from_settling_itself(repo, monkeypatch):
    directory, base = repo
    merged = integrate.assemble(directory, base, ["a"])
    assert merged["no_path"] == []
    merged["no_path"] = ["CONFLICT (new): something git could not place"]
    prepared = {"base": base, "merged": merged, "before": _analysis(),
                "against": _analysis(), "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatReadsThenCompletes)
    _StageThatReadsThenCompletes.refusal_list.clear()
    out = integrate.run_integration(prepared, "", "p", "out")
    assert out["called"] is True and out["settled"] is False
    # The call ending the run and the check after it say the same
    assert "something git could not place" in _StageThatReadsThenCompletes.refusal_list[0]
    assert out["left"] == [integrate._NO_PATH_REASON.format(
        messages="CONFLICT (new): something git could not place")]


# == Where the union holds what a side added ================


def test_a_name_added_where_another_side_put_a_definition_around_is_looked_for_by_its_end():
    """b put the start's names under a section; a's new name lands under it too."""
    lib = "p/CALC_cbl/CALC.cbl"
    keep_list = integrate.keep_list_of(
        [("a", {lib: {"CALC.MAIN"}}), ("b", {lib: {"CALC", "CALC.MAIN"}})],
        _analysis(name={lib: ["CALC", "CALC.MAIN"]}),
        {"a": _analysis(name={lib: ["CALC", "CALC.MAIN", "CALC.SUB"]})["define"],
         "b": _analysis(name={lib: ["CALC", "CALC.SEC", "CALC.SEC.MAIN"]})["define"]})
    assert keep_list == [("a", {"MAIN": [lib], "SUB": [lib]}),
                         ("b", {"CALC": [lib], "MAIN": [lib], "SEC": [lib]})]
    union = _analysis(name={lib: ["CALC", "CALC.SEC", "CALC.SEC.MAIN", "CALC.SEC.SUB"]})
    assert integrate._drop_reason_list(keep_list, union["define"], "p") == []
    union = _analysis(name={lib: ["CALC", "CALC.SEC", "CALC.SEC.MAIN"]})
    reason_list = integrate._drop_reason_list(keep_list, union["define"], "p")
    assert len(reason_list) == 1 and reason_list[0].startswith("- SUB, which a added")


def _dir_rename_repo(tmp_path):
    """Branch a renames pkg/ to core/, branch b adds pkg/extra.py."""
    directory = str(tmp_path / "dir_rename")
    os.makedirs(os.path.join(directory, "pkg"))
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    for name in ("one.py", "two.py"):
        _write(directory, f"pkg/{name}", "".join(f"{name}_{n} = {n}\n" for n in range(20)))
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "-b", "a", base)
    _git(directory, "mv", "pkg", "core")
    _git(directory, "commit", "-qm", "a")
    _git(directory, "checkout", "-q", "-b", "b", base)
    _write(directory, "pkg/extra.py", "def new_fn():\n    return 3\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "b")
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


def test_a_name_added_to_a_directory_another_side_renamed_is_found_wherever_it_is(
        tmp_path, monkeypatch):
    directory, base = _dir_rename_repo(tmp_path)
    old_file, new_file = "p/pkg/extra_py/extra.py", "p/core/extra_py/extra.py"
    prepared = {"base": base, "merged": integrate.assemble(directory, base, ["a", "b"]),
                "before": _analysis(), "against": _analysis(), "was_on": "",
                "side_define": {"a": {}, "b": {"new_fn": [old_file]}}}
    # git puts the file in the renamed directory and reports where it put it
    assert os.path.exists(os.path.join(directory, "core", "extra.py"))
    assert prepared["merged"]["conflicted"] == ["core/extra.py"]
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis(name={new_file: ["new_fn"]}))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    measure = integrate._measure_union(prepared, "", "p", directory)
    assert measure["keep"] == [("a", {}), ("b", {"new_fn": [old_file]})]
    for file in (new_file, old_file, "p/other_py/other.py"):
        assert integrate._drop_reason_list(measure["keep"], {"new_fn": [file]}, "p") == []
    # The definition thrown away is found gone, and named with the file the side has it in
    reason_list = integrate._drop_reason_list(measure["keep"], {"other": [new_file]}, "p")
    assert reason_list == [("- new_fn, which b added or changed in pkg/extra.py, is not "
                            "defined in any file now")]


def _placed_reason_list(directory, base, merged):
    """The gate's reasons over the "in_block" of a union, the run reading nothing."""
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {
        "": base, **{e["branch"]: e["commit"] for e in merged["branches"]}}
    return integrate._no_branch_read_reason_list(integrate.in_block_read_list(merged))


@pytest.mark.parametrize("order", [["a", "b"], ["b", "a"]])
def test_a_file_git_put_at_a_path_no_side_holds_is_read_on_the_side_that_added_it(
        tmp_path, monkeypatch, order):
    """b's new pkg/extra.py lands in the core/ a renamed pkg/ to. The run is held to
    reading b's own version wherever git put it, the model taking the file away
    included, and so is the run carrying on from a union whose commit records no side
    after the file was taken away."""
    directory, base = _dir_rename_repo(tmp_path)
    b = _git(directory, "rev-parse", "b").strip()
    merged = integrate.assemble(directory, base, order)
    assert {e["branch"]: e["in_block"] for e in merged["branches"]} == {
        "a": {}, "b": {"core/extra.py": "pkg/extra.py"}}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    os.remove(os.path.join(directory, "core", "extra.py"))
    assert [reason.split(". ")[-1] for reason in _placed_reason_list(directory, base, merged)
            ] == ["read_branch_file reads it on b at pkg/extra.py"]
    tool = next(t for t in integrate.merge_tool_list() if t.__name__ == "read_branch_file")
    assert tool("b", "pkg/extra.py") == "def new_fn():\n    return 3\n"
    assert integrate._no_branch_read_reason_list(integrate.in_block_read_list(merged)) == []
    # The container ends with the file taken away and no record of the sides
    _git(directory, "add", "-A")
    tree = _git(directory, "write-tree").strip()
    _git(directory, "reset", "-q", "--hard")
    union = _git(directory, "commit-tree", tree, "-p", base, "-p", "a", "-p", "b",
                 "-m", "integrate: x\n\ncombine\n\nCodetwine-Loom-Exit-Status: 137").strip()
    _git(directory, "branch", "union", union)
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    assert {e["commit"]: e["in_block"] for e in merged["branches"]}[b] == {
        "core/extra.py": "pkg/extra.py"}
    assert len(_placed_reason_list(directory, base, merged)) == 1


def _placed_repo(tmp_path, base_file_dict, side_list):
    """A repository whose branches each move and write some files off one commit.

    Args:
        base_file_dict: {path: text} of the commit the branches start from.
        side_list: [(name, [(old path, new path), ...], {path: text}), ...].
    """
    directory = str(tmp_path / "placed")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")

    def commit(file_dict, message):
        for path, text in file_dict.items():
            os.makedirs(os.path.dirname(os.path.join(directory, path)), exist_ok=True)
            _write(directory, path, text)
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", message)

    commit(base_file_dict, "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    for name, move_list, file_dict in side_list:
        _git(directory, "checkout", "-q", "-b", name, base)
        for old, new in move_list:
            os.makedirs(os.path.dirname(os.path.join(directory, new)), exist_ok=True)
            _git(directory, "mv", old, new)
        commit(file_dict, name)
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


@pytest.mark.parametrize("path, fork_list, side_path", [
    ("g.txt", [("w", "g.txt", "g.txt"), ("s", "f.txt", "")], "g.txt"),
    ("g.txt", [("w", "g.txt", ""), ("s", "f.txt", "h.txt")], "h.txt"),
    ("g.txt", [("w", "f.txt", "h.txt"), ("s", "f.txt", "")], "h.txt"),
    ("g.txt", [("w", "g.txt", "g.txt"), ("s", "f.txt", "h.txt")], "g.txt"),
    ("g.txt", [("w", "f.txt", "h.txt"), ("s", "e.txt", "k.txt")], None),
    ("g.txt", [("w", "g.txt", ""), ("s", "f.txt", "")], ""),
    ("g.txt", [("w", "g.txt", "g.txt"), ("v", "g.txt", "h.txt"), ("s", "f.txt", "")], None)])
def test_the_path_a_side_holds_a_file_at_is_read_first_where_it_parted_at_that_path(
        path, fork_list, side_path):
    assert integrate._unheld_side_path(path, fork_list) == side_path


def test_a_side_made_on_another_is_held_to_a_file_the_other_moved_at_the_path_it_has(
        tmp_path):
    """w moves f.txt to g.txt and x, made on w, rewrites most of g.txt. The union the
    container left with no record holds g.txt as w has it: x is held to g.txt, the
    path it holds the file at, whichever of the commits it parted from the others at
    the file is read from."""
    text = "".join(f"line {n} v0\n" for n in range(20))
    directory, base = _placed_repo(tmp_path, {"f.txt": text}, [("w", [("f.txt", "g.txt")], {})])
    _git(directory, "checkout", "-q", "-b", "x", "w")
    _write(directory, "g.txt", "".join(f"line {n} {'X' if 2 <= n < 18 else 'v0'}\n"
                                       for n in range(20)))
    _git(directory, "commit", "-qam", "x")
    _git(directory, "checkout", "-q", "--detach", base)
    x = _git(directory, "rev-parse", "x").strip()
    integrate.assemble(directory, base, ["w", "x"])
    _write(directory, "g.txt", text)
    _git(directory, "add", "-A")
    tree = _git(directory, "write-tree").strip()
    _git(directory, "reset", "-q", "--hard")
    union = _git(directory, "commit-tree", tree, "-p", base, "-p", "w", "-p", "x",
                 "-m", "integrate: x\n\ncombine\n\nCodetwine-Loom-Exit-Status: 137").strip()
    _git(directory, "branch", "union", union)
    assert (x, "g.txt", "g.txt") in integrate._open_no_merge_list(directory, union, True)
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    assert {e["commit"]: e["in_block"] for e in merged["branches"]}[x] == {"g.txt": "g.txt"}


def test_a_file_a_side_holds_that_is_no_text_is_read_when_the_tool_says_so(
        tmp_path, monkeypatch):
    """a and x change img.bin, and the union the container left with no record holds
    a's version: x is held to img.bin, which read_branch_file reads as no text. Once
    a has deleted the file from the tree, the conflicted path is read on a the same
    way."""
    blob = bytes(range(256)) * 8
    directory, base = _placed_repo(tmp_path, {"f.txt": "f\n"}, [])
    side_dict = {}
    for name, data in (("", blob), ("a", b"\0a\0" + blob[3:]), ("x", blob[:100] + b"\0x\0")):
        if name:
            _git(directory, "checkout", "-q", "-b", name, base)
        with open(os.path.join(directory, "img.bin"), "wb") as handle:
            handle.write(data)
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", name or "img")
        side_dict[name] = _git(directory, "rev-parse", "HEAD").strip()
        base = side_dict[""]
    _git(directory, "checkout", "-q", "--detach", base)
    integrate.assemble(directory, base, ["a", "x"])
    _git(directory, "add", "-A")
    tree = _git(directory, "write-tree").strip()
    _git(directory, "reset", "-q", "--hard")
    union = _git(directory, "commit-tree", tree, "-p", base, "-p", "a", "-p", "x",
                 "-m", "integrate: x\n\ncombine\n\nCodetwine-Loom-Exit-Status: 137").strip()
    _git(directory, "branch", "union", union)
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    x_name = integrate.side_name(side_dict["x"])
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    reason_list = _placed_reason_list(directory, base, merged)
    assert [reason.split(". ")[-1] for reason in reason_list] == [
        f"read_branch_file reads it on {x_name} at img.bin"]
    tool = next(t for t in integrate.merge_tool_list() if t.__name__ == "read_branch_file")
    assert tool(x_name, "img.bin") == f"Error: img.bin on {x_name} is not text"
    assert tool(x_name, "none.bin").startswith("Error: none.bin does not exist")
    assert integrate._STAGE_DICT["branch_read_set"] == {(x_name, "img.bin")}
    assert integrate._no_branch_read_reason_list(integrate.in_block_read_list(merged)) == []
    os.remove(os.path.join(directory, "img.bin"))
    a_name = integrate.side_name(side_dict["a"])
    assert integrate._no_read_reason_list(["img.bin"]) == []
    integrate.reset_read()
    assert len(integrate._no_read_reason_list(["img.bin"])) == 1
    tool(a_name, "img.bin")
    assert integrate._no_read_reason_list(["img.bin"]) == []


def test_a_union_recording_its_sides_holds_too_a_side_whose_change_it_left_out(tmp_path):
    """a and b conflict in f.txt and the run before took b's line out and ended
    unsettled with its record: b is held to f.txt in the run carrying on from it, and
    a, whose line the union holds, is not."""
    directory, base = _line_repo(tmp_path, {4: "A"}, {4: "B"})
    b = _git(directory, "rev-parse", "b").strip()
    merged = integrate.assemble(directory, base, ["a", "b"])
    union = _leave_union(directory, merged, "union", {"f.txt": _line_text({4: "A"})})
    assert integrate._no_merge_record_at(directory, union) == []
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    assert [(entry_dict["commit"] == b, entry_dict["in_block"])
            for entry_dict in merged["branches"]] == [(False, {}), (True, {"f.txt": "f.txt"})]


@pytest.mark.parametrize("kind", ["bin", "link"])
def test_a_side_whose_version_git_did_not_keep_is_held_to_the_file(tmp_path, monkeypatch,
                                                                   kind):
    """a and x change img the same way git does not merge as text: git keeps a's
    version and x is held to the file until its version is read."""
    blob = bytes(range(256)) * 8
    directory, base = _placed_repo(tmp_path, {"f.txt": "f\n"}, [])
    side_dict = {}
    for name, value in (("", "t0"), ("a", "ta"), ("x", "tx")):
        if name:
            _git(directory, "checkout", "-q", "-b", name, base)
        full = os.path.join(directory, "img")
        if os.path.lexists(full):
            os.remove(full)
        if kind == "link":
            os.symlink(value, full)
        else:
            with open(full, "wb") as handle:
                handle.write(value.encode() + blob)
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", name or "img")
        side_dict[name] = _git(directory, "rev-parse", "HEAD").strip()
        base = side_dict[""]
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["a", "x"])
    assert merged["unmarked"] == ["img"]
    assert {e["branch"]: e["in_block"] for e in merged["branches"]} == {
        "a": {}, "x": {"img": "img"}}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    assert len(_placed_reason_list(directory, base, merged)) == 1
    tool = next(t for t in integrate.merge_tool_list() if t.__name__ == "read_branch_file")
    tool("x", "img")
    assert integrate._no_branch_read_reason_list(integrate.in_block_read_list(merged)) == []


def test_a_mode_a_side_gave_a_file_is_held_to_as_reading_it_does_not_show_it(
        tmp_path, monkeypatch):
    """x makes run.sh executable and a changes its text: read on both, the merge is
    settled only once the working tree's run.sh is executable as well. a, which left
    the mode as it was, is held to no mode, and a file taken away holds none."""
    directory, base = _placed_repo(tmp_path, {"run.sh": "echo hi\n"},
                                   [("a", [], {"run.sh": "echo a\n"})])
    _git(directory, "checkout", "-q", "-b", "x", base)
    os.chmod(os.path.join(directory, "run.sh"), 0o755)
    _git(directory, "update-index", "--chmod=+x", "run.sh")
    _git(directory, "commit", "-qm", "x")
    _git(directory, "checkout", "-q", "--detach", base)
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {"": base, "a": _git(directory, "rev-parse", "a").strip(),
                                            "x": _git(directory, "rev-parse", "x").strip()}
    in_block_list = [("a", "run.sh", "run.sh"), ("x", "run.sh", "run.sh")]
    integrate._STAGE_DICT["branch_read_set"].update({("a", "run.sh"), ("x", "run.sh")})
    full = os.path.join(directory, "run.sh")
    os.chmod(full, 0o644)
    assert integrate._no_branch_read_reason_list(in_block_list) == [
        "- x made run.sh an executable file, and run.sh in the working tree is a file "
        "that is not executable, which reading the file does not show. Give run.sh the "
        "mode x gave it"]
    os.chmod(full, 0o755)
    assert integrate._no_branch_read_reason_list(in_block_list) == []
    os.remove(full)
    assert integrate._no_branch_read_reason_list(in_block_list) == []
    os.symlink("elsewhere", full)
    assert [reason.split(", and ")[1].split(",")[0]
            for reason in integrate._no_branch_read_reason_list(in_block_list)] == [
        "run.sh in the working tree is a symbolic link"]


def test_a_side_that_moved_a_file_is_held_to_no_mode_it_left_as_it_was(tmp_path,
                                                                       monkeypatch):
    """w moves the executable run.sh to bin/run.sh and changes a line of it, x makes
    run.sh not executable: bin/run.sh not executable in the working tree settles both,
    and an executable one is held to x's mode alone."""
    lines = "".join(f"echo {n}\n" for n in range(20))
    directory, base = _placed_repo(tmp_path, {"run.sh": lines}, [])
    os.chmod(os.path.join(directory, "run.sh"), 0o755)
    _git(directory, "update-index", "--chmod=+x", "run.sh")
    _git(directory, "commit", "-qm", "executable")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "-b", "w", base)
    os.makedirs(os.path.join(directory, "bin"))
    _git(directory, "mv", "run.sh", "bin/run.sh")
    _write(directory, "bin/run.sh", lines.replace("echo 2\n", "echo w\n"))
    _git(directory, "commit", "-qam", "w")
    _git(directory, "checkout", "-q", "-b", "x", base)
    os.chmod(os.path.join(directory, "run.sh"), 0o644)
    _git(directory, "commit", "-qam", "x")
    _git(directory, "checkout", "-q", "--detach", base)
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {
        "": base, **{name: _git(directory, "rev-parse", name).strip() for name in "wx"}}
    in_block_list = [("w", "bin/run.sh", "bin/run.sh"), ("x", "bin/run.sh", "run.sh")]
    integrate._STAGE_DICT["branch_read_set"].update({("w", "bin/run.sh"), ("x", "run.sh")})
    os.makedirs(os.path.join(directory, "bin"), exist_ok=True)
    _write(directory, "bin/run.sh", lines.replace("echo 2\n", "echo w\n"))
    full = os.path.join(directory, "bin", "run.sh")
    os.chmod(full, 0o644)
    assert integrate._no_branch_read_reason_list(in_block_list) == []
    os.chmod(full, 0o755)
    assert integrate._no_branch_read_reason_list(in_block_list) == [
        "- x made run.sh a file that is not executable, and bin/run.sh in the working "
        "tree is an executable file, which reading the file does not show. Give "
        "bin/run.sh the mode x gave it"]


def test_the_files_a_side_is_held_to_are_named_whether_or_not_they_held_a_block():
    """What the work list and the gate say of "in_block" holds for a file with no
    conflict block as well as for one with."""
    for text in (integrate._IN_BLOCK_NOTE, integrate._NO_BRANCH_READ_REASON,
                 integrate._NO_SIDE_PATH_REASON, integrate.INTEGRATE_VAR_DOC,
                 integrate.INTEGRATE_INSTRUCTION_TEMPLATE):
        assert "not known to hold" in " ".join(text.split())
        assert "being merged no further" not in text


def test_a_file_git_moved_out_of_the_way_is_read_on_the_side_that_changed_it(tmp_path):
    """a adds the file d and b the directory d/: git puts a's file under a name of its
    own, which a is held to at d. b moves top.txt into the old/ a renamed: b is held to
    the file git put in new/, and a, which holds top.txt as the start does, is not."""
    lines = "".join(f"line {n}\n" for n in range(20))
    directory, base = _placed_repo(tmp_path / "file", {"f.txt": lines},
                                   [("a", [], {"d": "a\n"}), ("b", [], {"d/x.txt": "b\n"})])
    merged = integrate.assemble(directory, base, ["a", "b"])
    (stand_in,) = merged["stand_in"]
    assert {e["branch"]: e["in_block"] for e in merged["branches"]} == {
        "a": {stand_in: "d"}, "b": {}}
    directory, base = _placed_repo(
        tmp_path / "dir", {"old/f.txt": lines, "old/g.txt": lines + "g\n",
                           "top.txt": "top " + lines},
        [("a", [("old", "new")], {}), ("b", [("top.txt", "old/top.txt")], {})])
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == ["new/top.txt"]
    assert {e["branch"]: e["in_block"] for e in merged["branches"]} == {
        "a": {}, "b": {"new/top.txt": "old/top.txt"}}


# == Reading a file the merged tree does not hold ============


def test_a_file_the_tree_does_not_hold_is_asked_for_on_the_branches_that_hold_it(
        unmarked_repo, monkeypatch):
    directory, base = unmarked_repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {
        "": base, **{e["branch"]: e["commit"] for e in merged["branches"]}}
    assert integrate._no_read_reason_list(["gone.txt", "res.rc"]) == [
        integrate._NO_READ_REASON.format(paths="gone.txt, res.rc")]
    os.remove(os.path.join(directory, "gone.txt"))
    os.mkdir(os.path.join(directory, "sub"))
    reason_list = integrate._no_read_reason_list(["gone.txt", "res.rc", "sub", "nowhere.txt"])
    assert reason_list == [
        integrate._NO_READ_REASON.format(paths="res.rc"),
        integrate._NO_READ_BRANCH_REASON.format(
            path="gone.txt", branches=f"{integrate._START_BRANCH_TEXT}, b"),
        integrate._NO_TOOL_READ_REASON.format(paths="sub")]
    tool = next(t for t in integrate.merge_tool_list() if t.__name__ == "read_branch_file")
    tool("b", "gone.txt")
    assert integrate._no_read_reason_list(["gone.txt"]) == []


# == A merge carried on from over a working tree that has changed ==


def test_the_tree_now_stands_in_place_of_the_tree_an_earlier_merge_was_written_against(
        repo):
    """Both commits of the working tree add notes.txt; the later one is what is merged."""
    directory, _ = repo
    _git(directory, "checkout", "-q", "main")
    _write(directory, "notes.txt", "first\n")
    first = gitcmd.snapshot_commit(directory)
    a = _git(directory, "rev-parse", "a").strip()
    tree = _git(directory, "merge-tree", "--write-tree", first, a).strip()
    union = _git(directory, "commit-tree", tree, "-p", first, "-p", a, "-m",
                 "integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: unsettled").strip()
    _git(directory, "branch", "union", union)
    _write(directory, "notes.txt", "second\nand more\n")
    now = gitcmd.snapshot_commit(directory, first)
    _git(directory, "branch", "tree", now)
    os.remove(os.path.join(directory, "notes.txt"))
    _git(directory, "checkout", "-q", "--detach", first)
    merged = integrate.assemble(directory, first, ["union"], onto="tree", onto_commit=now,
                                resume="union")
    assert merged["conflicted"] == []
    assert _read(directory, "notes.txt") == "second\nand more\n"
    assert _read(directory, "f.txt") == "ONE\ntwo\nthree\n"


# == A definition told by its own name, wherever its file is ==


def test_the_analysis_holds_each_definition_s_own_name_and_where_it_is_defined(tmp_path):
    analysis = integrate.analyse(_two_run_knowledge(tmp_path, []))
    calc = "p/calc_py/calc.py"
    assert analysis["own"][calc] == {"A": "A", "A.run": "run", "B": "B", "B.run": "run"}
    assert analysis["define"] == {"A": [calc], "B": [calc], "run": [calc, calc]}


def test_a_name_holding_a_dot_is_one_own_name(tmp_path):
    from conftest import write_knowledge_db
    path = str(tmp_path / "k.sqlite")
    stats = "p/stats_R/stats.R"
    write_knowledge_db({"files": [
        {"file": stats,
         "file_dependencies": {
             "definitions": [{"name": name, "type": "function_definition",
                              "start_line": line, "end_line": line, "context": name}
                             for line, name in enumerate(("print.shape", "summary.shape"), 1)],
             "callee_usages": [], "caller_usages": []},
         "doc": {"sections": [], "summary": ""}}]}, path, "p")
    assert integrate.analyse(path)["define"] == {"print.shape": [stats],
                                                 "summary.shape": [stats]}


def _definition_knowledge(tmp_path, file_dict):
    """A knowledge database holding the given definitions: {file: [(name, type) or
    (name, type, the definition's text), ...]}."""
    from conftest import write_knowledge_db
    path = str(tmp_path / "k.sqlite")
    write_knowledge_db({"files": [
        {"file": file,
         "file_dependencies": {
             "definitions": [{"name": entry[0], "type": entry[1], "start_line": line,
                              "end_line": line, "context": (entry[2:] or entry[:1])[0]}
                             for line, entry in enumerate(entry_list, 1)],
             "callee_usages": [], "caller_usages": []},
         "doc": {"sections": [], "summary": ""}}
        for file, entry_list in file_dict.items()]}, path, "p")
    return path


def test_a_name_written_with_the_names_it_belongs_to_is_told_by_its_last_part(tmp_path):
    """A member defined outside its class, as the analysis of C++ names one."""
    calc = "p/calc_cpp/calc.cpp"
    analysis = integrate.analyse(_definition_knowledge(tmp_path, {calc: [
        ("Calc::mul", "function_definition"), ("ns::in::deep", "function_definition"),
        ("Calc::operator ns::Flag", "function_definition"),
        ("Calc::operator+", "function_definition"), ("plain", "function_definition")]}))
    assert analysis["own"][calc] == {
        "Calc::mul": "mul", "ns::in::deep": "deep",
        "Calc::operator ns::Flag": "operator ns::Flag", "Calc::operator+": "operator+",
        "plain": "plain"}
    assert sorted(analysis["define"]) == ["deep", "mul", "operator ns::Flag", "operator+",
                                         "plain"]
    assert analysis["name"][calc] == sorted(analysis["own"][calc])


def test_a_definition_named_after_its_type_is_told_by_the_type_s_name(tmp_path):
    """A destructor, in the class, outside it and in C#: one more definition of the
    class's name, as a constructor is."""
    header, source, svc = "p/calc_h/calc.h", "p/calc_cpp/calc.cpp", "p/Svc_cs/Svc.cs"
    analysis = integrate.analyse(_definition_knowledge(tmp_path, {
        header: [("Calc", "class_specifier"), ("Calc", "function_declarator"),
                 ("~Calc", "function_declarator"), ("operator~", "function_declarator")],
        source: [("Calc::Calc", "function_definition"), ("Calc::~Calc", "function_definition"),
                 ("Calc::operator~", "function_definition")],
        svc: [("Svc", "class_declaration"), ("Svc", "constructor_declaration"),
              ("~Svc", "destructor_declaration")]}))
    assert analysis["define"] == {"Calc": [source, source, header, header, header],
                                  "operator~": [source, header], "Svc": [svc, svc, svc]}
    assert analysis["own"][source]["Calc::~Calc"] == "Calc"
    # The name as the analysis gives it is what every other reading holds
    assert "~Calc" in analysis["name"][header] and "~Svc" in analysis["name"][svc]


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_only_the_names_of_a_language_that_writes_them_so_lose_their_front(ext):
    from codetwine_loom.languages import own_name
    path = f"src/x.{ext}"
    is_cpp = LANG_DICT[ext].name_prefix_pattern == LANG_DICT["cpp"].name_prefix_pattern
    assert own_name(path, "Calc::mul") == ("mul" if is_cpp else "Calc::mul")
    assert own_name(path, "~Calc") == ("Calc" if is_cpp or ext == "cs" else "~Calc")
    assert bool(LANG_DICT[ext].name_prefix_pattern) is (is_cpp or ext == "cs")
    for name in ("print.shape", "mul", "operator bool", "operator~", "WS-TOTAL", "%+%",
                 "::", "~"):
        assert own_name(path, name) == name
    assert own_name("no_language.txt", "Calc::~Calc") == "Calc::~Calc"


def test_a_variable_or_a_constant_given_a_value_is_no_definition_a_side_is_held_to(tmp_path):
    file_dict = {
        "p/app_py/app.py": [("TOTAL", "expression_statement"), ("main", "function_definition"),
                            ("App", "class_definition")],
        "p/run_R/run.R": [("total", "binary_operator"), ("main", "function_definition")],
        "p/k_kt/k.kt": [("limit", "property_declaration"), ("go", "function_declaration")],
        "p/c_c/c.c": [("g", "declaration"), ("LIMIT", "preproc_def"),
                      ("SQUARE", "preproc_function_def"), ("api", "function_definition")],
        "p/n_cpp/n.cpp": [("counter", "declaration"), ("outer", "namespace_definition")],
        "p/m_rs/m.rs": [("K", "const_item"), ("S", "static_item"), ("inner", "mod_item")],
        "p/u_js/u.js": [("depth", "lexical_declaration"), ("old", "variable_declaration"),
                        ("alpha", "function_declaration")],
        "p/t_ts/t.ts": [("delta", "lexical_declaration"), ("Shape", "interface_declaration")],
        # A field a type is given is a value as well
        "p/M_java/M.java": [("count", "field_declaration"), ("M", "class_declaration"),
                            ("RED", "enum_constant"), ("MAX", "constant_declaration")],
        "p/S_cs/S.cs": [("Limit", "field_declaration"), ("Value", "property_declaration"),
                        ("Changed", "event_field_declaration")],
        "p/W_js/W.js": [("depth", "field_definition")],
        "p/V_ts/V.ts": [("width", "public_field_definition"), ("size", "property_signature")],
        # A member of a struct and a data item are not
        "p/s_h/s.h": [("x", "field_declaration"), ("FAST", "enumerator")],
        "p/CALC_cbl/CALC.cbl": [("WS-A", "data_description"), ("MAIN-PARA", "paragraph_header")]}
    analysis = integrate.analyse(_definition_knowledge(tmp_path, file_dict))
    assert sorted(analysis["define"]) == sorted([
        "main", "App", "go", "SQUARE", "api", "outer", "inner", "alpha", "Shape",
        "M", "RED", "MAX", "Changed", "size", "x", "FAST", "WS-A", "MAIN-PARA"])
    assert analysis["define"]["main"] == ["p/app_py/app.py", "p/run_R/run.R"]
    assert "TOTAL" not in analysis["own"]["p/app_py/app.py"]
    # The other readings of the analysis keep every definition
    assert analysis["name"]["p/app_py/app.py"] == ["App", "TOTAL", "main"]
    assert [name for _, _, name in analysis["range"]["p/run_R/run.R"]] == ["total", "main"]


def test_a_variable_holding_a_function_is_a_definition_a_side_is_held_to(tmp_path):
    """The type is the one a value has; the definition's own text tells the two apart."""
    file_dict = {
        "p/u_js/u.js": [
            ("gamma", "lexical_declaration", "export const gamma = () => 3;"),
            ("LIMIT", "lexical_declaration", "export const LIMIT = 3;"),
            ("old", "variable_declaration", "var old = function () {\n  return 1;\n};"),
            ("one", "variable_declaration", "var one = 1;")],
        "p/t_ts/t.ts": [
            ("typed", "lexical_declaration", "const typed: (n: number) => number = id;"),
            ("depth", "lexical_declaration", "const depth: number = 2;")],
        "p/a_py/a.py": [("squares", "expression_statement", "squares = lambda n: n * n"),
                        ("TOTAL", "expression_statement", "TOTAL = sum(values)")],
        "p/k_kt/k.kt": [("twice", "property_declaration", "val twice = { x: Int -> x * 2 }"),
                        ("limit", "property_declaration", "val limit = 3")],
        "p/l_cpp/l.cpp": [("lam", "declaration", "auto lam = [](int x) { return x; };"),
                          ("counter", "declaration", "int counter = 0;")],
        "p/s_R/s.R": [("cached", "binary_operator", "cached <- memoise(function(x) x)"),
                      ("K", "binary_operator", "K <- 10")],
        "p/m_rs/m.rs": [("F", "static_item", "static F: fn(i32) -> i32 = helper;"),
                        ("MASK", "const_item", "const MASK: u8 = A | B | C;")]}
    analysis = integrate.analyse(_definition_knowledge(tmp_path, file_dict))
    assert sorted(analysis["define"]) == sorted(
        ["gamma", "old", "typed", "squares", "twice", "lam", "cached", "F"])


@pytest.mark.parametrize("ext, def_type, text, is_value", [
    ("js", "lexical_declaration", "const total = result + 1;", True),
    ("js", "lexical_declaration", "let options = { depth: 2,\n  names: ['a'] };", True),
    ("js", "lexical_declaration", "let { a, b } = obj;", True),
    ("js", "lexical_declaration", "let later;", True),
    ("js", "variable_declaration", "var n = compute(1);", True),
    ("js", "lexical_declaration", "const f = () => 3;", False),
    ("js", "lexical_declaration", "const f = async x => x;", False),
    ("js", "lexical_declaration", "const f = function () {};", False),
    ("js", "lexical_declaration", "const K = class {};", False),
    ("js", "lexical_declaration", "const f = wrap(() => 1);", False),
    ("js", "lexical_declaration", "const o = {\n  run() {\n    return 1;\n  },\n};", False),
    ("js", "variable_declaration", "var one = 1, two = () => 2;", False),
    ("js", "function_declaration", "function f() {}", False),
    ("js", "field_definition", "  count = 1;", True),
    ("js", "field_definition", "  static table = { a: 1 };", True),
    ("js", "field_definition", "  handler = () => 1;", False),
    ("js", "field_definition", "  static Inner = class {};", False),
    ("js", "field_definition", "  run = function () {};", False),
    ("ts", "field_definition", "  count = 1;", False),
    ("tsx", "lexical_declaration", "const LIMIT: number = 3;", True),
    ("ts", "lexical_declaration", "export const f = (n: number): number => n;", False),
    ("ts", "public_field_definition", "  field = 1;", True),
    ("ts", "public_field_definition", "  private readonly total: number;", True),
    ("ts", "public_field_definition", "  handler = (n: number) => n;", False),
    ("ts", "public_field_definition", "  check: (n: number) => boolean;", False),
    ("ts", "property_signature", "  size: number;", False),
    ("py", "expression_statement", "X = {'a': 1}", True),
    ("py", "expression_statement", "handler = staticmethod(lambda x: x)", False),
    ("kt", "property_declaration", "val x: Int get() = 1", False),
    ("kt", "property_declaration", "val total: Int\n    get() = items.sum()", False),
    ("kt", "property_declaration", "val String.lastChar: Char get() = this[length - 1]",
     False),
    ("kt", "property_declaration", "var size = 0\n    private set(value) = store(value)",
     False),
    ("kt", "property_declaration", "val x: Int get /* one */ () = 1", False),
    ("kt", "property_declaration", "val repo: Repo = get()", False),
    ("kt", "property_declaration", "val first = prefs.get(KEY)", True),
    ("kt", "property_declaration", "val last = cache?.get(KEY) ?: 0", True),
    ("kt", "property_declaration", "val names = setOf(1, 2)", True),
    ("kt", "property_declaration", "val v = getValue(x)", True),
    ("kt", "property_declaration", "val f: (Int) -> Int = ::twice", False),
    ("kt", "property_declaration", "val g = fun(x: Int): Int { return x }", False),
    ("c", "declaration", "int g = 1;", True),
    ("cpp", "declaration", "int arr[] = {1, 2};", True),
    ("cpp", "declaration", "auto f = [&] { return 1; };", False),
    ("cpp", "declaration", "auto f = [](int a) { return a; };", False),
    ("cpp", "declaration", "auto f = []<typename T>(T a) { return a; };", False),
    ("cpp", "declaration", "auto f = [] [[nodiscard]] () { return 1; };", False),
    ("cpp", "declaration", "auto f = [n = 0] mutable { return n; };", False),
    ("cpp", "declaration", "auto f = [] -> int { return 1; };", False),
    ("cpp", "declaration", "auto f = [] noexcept { return 1; };", False),
    ("cpp", "declaration", "auto f = [] constexpr { return 1; };", False),
    ("cpp", "declaration", "auto f = [] consteval { return 1; };", False),
    ("cpp", "declaration", "auto f = [] static { return 1; };", False),
    ("cpp", "declaration", "auto f = []\n    requires true { return 1; };", False),
    ("cpp", "preproc_def", "#define ADD [](int a, int b) { return a + b; }", False),
    ("cpp", "declaration", "int grid[2][3];", True),
    ("cpp", "declaration", "const char* names[2] = {\"a\", \"b\"};", True),
    ("cpp", "declaration", "bool small = sizes[0] < 3;", False),
    ("cpp", "field_declaration", "  int depth = 0;", False),
    ("h", "preproc_def", "#define LIMIT 3", True),
    ("rs", "const_item", "const C: Lazy<u8> = Lazy::new(move || 1);", False),
    ("rs", "const_item", "pub const ADD: fn(u32) -> u32 = |x| x + 1;", False),
    ("rs", "static_item", "pub static PTR: fn() = run;", False),
    ("rs", "const_item", "pub const HOOK: &dyn Fn() = &|| ();", False),
    ("rs", "const_item", "const H: Hook = &|| ();", False),
    ("rs", "const_item", "const H: Hook = &mut |x| x;", False),
    ("rs", "static_item", "static T: Table = [(\"a\", &|| ())];", False),
    ("rs", "const_item", "const M: Maybe = Some(&|| ());", False),
    ("rs", "const_item", "const A: Arr = [|x| x];", False),
    ("rs", "const_item", "const B: u32 = { |x: u32| x }(1);", False),
    ("rs", "const_item", "const P: Pair = (1, |x| x);", False),
    ("rs", "const_item", "const L: Later = Lazy::new(async move |x| x);", False),
    ("rs", "const_item", "const R: u32 = { if ON { 1 } else { 2 }; |x| x }(3);", False),
    ("rs", "const_item", "const S: Boxed = Box::new(Wrap(|| 1));", False),
    ("rs", "const_item", "const _: () = { impl Shape for Dot {} };", False),
    ("rs", "const_item", "const F: Box<dyn FnMut(u8)> = make();", False),
    ("rs", "const_item", "const MASK: u8 = A | B | C;", True),
    ("rs", "const_item", "pub const MASK: u32 = FLAGS\n    | 4\n    | 8;", True),
    ("rs", "const_item", "const BITS: u8 = (A) | TABLE[0] | 1;", True),
    ("rs", "const_item", "const SEP: &str = \"|\";", True),
    ("rs", "static_item", "pub static NAME: &str = \"a | b\";", True),
    ("rs", "const_item", "const BAR: char = '|';", True),
    ("rs", "const_item", "const ANY: bool = A || B;", True),
    ("rs", "mod_item", "mod inner {}", False),
    ("r", "binary_operator", "parts <- list(a = 1)", True),
    ("r", "binary_operator", "f <- compose(\\(x) x, g)", False),
    ("java", "field_declaration", "int count = 0;", True),
    ("java", "field_declaration", "    private final List<String> names = new ArrayList<>();",
     True),
    ("java", "field_declaration", "    int[] list = {1, 2};", True),
    ("java", "field_declaration", "    static final Runnable R = () -> {};", False),
    ("java", "field_declaration", "    Function<Integer, Integer> twice = C::dbl;", False),
    ("java", "field_declaration", "    Runnable anon = new Runnable() {\n"
     "        public void run() {}\n    };", False),
    ("java", "field_declaration", "    Comparator<String> c = new Comparator<>()\n    {\n"
     "    };", False),
    ("java", "enum_constant", "    RED,", False),
    ("java", "constant_declaration", "    int MAX = 3;", False),
    ("java", "method_declaration", "    int count() { return 0; }", False),
    ("cs", "property_declaration", "    public int Total { get; set; }", True),
    ("cs", "property_declaration", "    public int Total { get; init; } = 0;", True),
    ("cs", "field_declaration", "    private readonly List<int> items = new();", True),
    ("cs", "property_declaration", "    public int Twice => count * 2;", False),
    ("cs", "property_declaration", "    public int Twice { get => count * 2; }", False),
    ("cs", "field_declaration", "    static readonly Func<int, int> Dbl = x => x * 2;", False),
    ("cs", "field_declaration", "    Action act = delegate { };", False),
    ("cs", "field_declaration", "    Func<int, int> f = delegate (int x) { return x; };",
     False),
    ("cs", "event_field_declaration", "    public event EventHandler Changed;", False),
    ("cs", "method_declaration", "    int Count() { return 0; }", False),
    # An accessor with a body
    ("cs", "property_declaration", "    public int Twice { get { return count * 2; } }",
     False),
    ("cs", "property_declaration",
     "    public int Safe { get { return count; } set { count = value < 0 ? 0 : value; } }",
     False),
    ("cs", "property_declaration", "    public int Id { get; init { id = value; } }", False),
    ("cs", "property_declaration", "    public int Twice\n    {\n        get\n        {\n"
     "            return count * 2;\n        }\n    }", False),
    ("cs", "property_declaration", "    public int Twice { get // doubles\n { return 2; } }",
     False),
    ("cs", "property_declaration", "    public HashSet<int> Seen { get; } = new HashSet<int> "
     "{ 1, 2 };", True),
    ("cs", "field_declaration", "    private HashSet<int> set = new HashSet<int> { 1 };", True),
    ("cs", "property_declaration", "    public int Total { get; private set; } = 3;", True),
    # A method of an object literal with its return type written
    ("ts", "lexical_declaration", "export const userService = {\n  async load(id: string): "
     "Promise<User> { return { id }; },\n  count(): number { return 1; },\n};", False),
    ("ts", "lexical_declaration", "export const reader = { get value(): number { return 1; } };",
     False),
    ("ts", "lexical_declaration", "export const maker = { *each(): Generator<number> "
     "{ yield 1; } };", False),
    ("ts", "lexical_declaration", "const byKey = { [key](): void { run(); } };", False),
    ("ts", "public_field_definition", "  handlers = { onClick(e: Event): void "
     "{ this.count += 1; } };", False),
    ("tsx", "lexical_declaration", "export const store = {\n  inc(n: number): number "
     "{ return n + 1; },\n};", False),
    ("ts", "lexical_declaration", "export const untyped = { load(id: string) { return 1; } };",
     False),
    ("ts", "lexical_declaration", "const size = { w: f(1), h: 2 };", True),
    ("ts", "lexical_declaration", "const conf: { a: number } = make(1);", True),
    ("ts", "lexical_declaration", "const both = [f(1), { a: 1 }];", True),
    ("ts", "public_field_definition", "  count: number = 0;", True),
    ("ts", "lexical_declaration", "const pick = flag ? g(1) : { a: 1 };", False),
    ("js", "lexical_declaration", "const o = { async load() { return 1; } };", False),
    ("js", "lexical_declaration", "const o = { get x() { return 1; } };", False),
    ("js", "lexical_declaration", "const o = { *gen() { yield 1; } };", False),
    ("js", "lexical_declaration", "const o = { [key]() { return 1; } };", False),
    # A comment, an attribute or a macro between the signs a function is told by
    ("rs", "const_item", "pub const NOTED: Adder = // the sum\n    |a, b| a + b;", False),
    ("rs", "const_item", "pub const NOTED: Adder = /* the sum */ |a, b| a + b;", False),
    ("rs", "const_item", "pub const MASK: u8 = A // first\n    | B;", True),
    ("cpp", "declaration", "auto noted = [] /* doubles */ (int a) { return a * 2; };", False),
    ("cpp", "declaration", "auto lined = [] // doubles\n    (int a) { return a * 2; };", False),
    ("cpp", "declaration", "auto attributed = [] __attribute__((noinline)) (int a) "
     "{ return a * 2; };", False),
    ("cpp", "declaration", "auto thrown = [] throw() { return 1; };", False),
    ("cpp", "declaration", "int grid[2] /* rows */ [3];", True),
    ("js", "lexical_declaration", "const noted = { run() // returns one\n  { return 1; } };",
     False),
    ("js", "lexical_declaration", "const blocked = { get value() /* one */ { return 1; } };",
     False),
    ("js", "lexical_declaration", "const url = 'http://x'; // the base", True),
    ("java", "field_declaration", "    Runnable r = new Runnable() // one\n    { };", False),
    ("r", "binary_operator", "f <- function # takes x\n  (x) x", False),
    ("r", "binary_operator", "limit <- 3 # the cap", True),
    ("cbl", "data_description", "01 WS-A PIC 9.", False),
    ("txt", "lexical_declaration", "const a = 1;", False)])
def test_a_value_is_told_from_a_function_by_the_definition_s_own_text(ext, def_type, text,
                                                                     is_value):
    from codetwine_loom.languages import is_value_type
    assert is_value_type(f"src/x.{ext}", def_type, text) is is_value


@pytest.mark.parametrize("ext", sorted(LANG_DICT))
def test_the_value_types_of_a_language_are_among_the_types_its_search_leaves_out(ext):
    lang_config = LANG_DICT[ext]
    assert lang_config.value_type_set <= lang_config.exclude_type_set
    # A language whose values can hold a function says how one is told
    assert bool(lang_config.function_value_pattern) is (
        bool(lang_config.value_type_set) and ext not in ("c",))
    if ext in ("js", "jsx", "mjs", "cjs", "ts", "tsx", "mts", "cts"):
        assert lang_config.value_type_set - {"field_definition", "public_field_definition"} \
            == {"lexical_declaration", "variable_declaration"}


def test_a_name_the_shared_start_still_defines_is_looked_for_on_a_side_starting_before_it():
    """The shared start removed one of two definitions of a name a side changed."""
    email, sms = "p/handlers/email_py/email.py", "p/handlers/sms_py/sms.py"
    own_start = _analysis(name={email: ["send"], sms: ["send"]})
    side_list = [("a", {email: {"send"}})]
    tip_dict = {"a": {"send": [email, sms]}}
    # The start is a side read from the commit a starts from: it removed sms.py
    start_side_list = [({sms: {"", "send"}}, own_start)]
    assert integrate.keep_list_of(
        side_list, _analysis(name={email: ["send"]}), tip_dict, {"a": own_start},
        start_side_list) == [("a", {}), (integrate._START_BRANCH_TEXT, {"send": [email]})]
    # A name the shared start defines nowhere was removed or renamed there
    assert integrate.keep_list_of(
        side_list, _analysis(name={email: ["other"]}), tip_dict, {"a": own_start},
        [({email: {"send"}, sms: {"", "send"}}, own_start)]) == [
            ("a", {}), (integrate._START_BRANCH_TEXT, {"other": [email]})]


def test_a_name_is_looked_for_the_same_whether_the_start_is_a_side_or_the_shared_start():
    """Two definitions of one name: the commit the union is written against renamed
    one, b removed the other's file. The first list is a first pass over the commit
    a and b start from, the second a pass that carries on from its union."""
    email, sms = "p/handlers/email_py/email.py", "p/handlers/sms_py/sms.py"
    old = _analysis(name={email: ["send", "other"], sms: ["send"]})
    tree = _analysis(name={email: ["deliver", "other"], sms: ["send"]})
    tree_side, b_side = {email: {"send"}}, {sms: {"", "send"}}
    b_tip = {"send": [email], "other": [email]}
    first = integrate.keep_list_of(
        [("tree", tree_side), ("a", {}), ("b", b_side)], old,
        {"tree": tree["define"], "a": old["define"], "b": b_tip})
    assert first == [("tree", {"deliver": [email]}), ("a", {}), ("b", {})]
    again = integrate.keep_list_of(
        [("a", {}), ("b", b_side)], tree, {"a": old["define"], "b": b_tip},
        {"a": old, "b": old}, [(tree_side, old)])
    assert again == [("a", {}), ("b", {}),
                     (integrate._START_BRANCH_TEXT, {"deliver": [email]})]
    # A name the start added that a side starting from the start took away again
    assert integrate.keep_list_of(
        [("a", {}), ("c", {email: {"deliver"}})], tree,
        {"a": old["define"], "c": {"other": [email], "send": [sms]}},
        {"a": old}, [(tree_side, old)])[-1] == (integrate._START_BRANCH_TEXT,
                                                {"send": [sms]})


def test_a_name_is_held_against_the_commit_the_two_sides_both_reach_last():
    """The branch the union lands on is at B2; a is cut at B0 and b at B1. B1 added
    helper2 and removed old; b renamed helper2 and added an old of its own. The first
    list is a first pass over B0, the second a pass carrying on from its union."""
    lib = "p/lib_py/lib.py"
    b0 = _analysis(name={lib: ["foo", "old"]})
    b1 = _analysis(name={lib: ["foo", "helper2"]})
    tree_tip, a_tip = b1["define"], b0["define"]
    b_tip = {"foo": [lib], "helper3": [lib], "old": [lib]}
    tree_side, b_side = {lib: {"", "old"}}, {lib: {"", "old"}}
    fork_dict = {("tree", "b"): b1["define"], ("b", "tree"): b1["define"]}
    first = integrate.keep_list_of(
        [("tree", tree_side), ("a", {}), ("b", b_side)], b0,
        {"tree": tree_tip, "a": a_tip, "b": b_tip}, fork_define_by_pair_dict=fork_dict)
    assert first == [("tree", {}), ("a", {}), ("b", {"helper3": [lib], "old": [lib]})]
    again = integrate.keep_list_of(
        [("a", {}), ("b", {lib: {"", "helper2"}})], b1, {"a": a_tip, "b": b_tip},
        {"a": b0}, [(tree_side, b0)], {("a", "b"): b0["define"]})
    assert again == [("a", {}), ("b", {"helper3": [lib], "old": [lib]}),
                     (integrate._START_BRANCH_TEXT, {})]
    # Read against the start alone, the name b renamed is looked for and b's own is not
    assert integrate.keep_list_of(
        [("tree", tree_side), ("a", {}), ("b", b_side)], b0,
        {"tree": tree_tip, "a": a_tip, "b": b_tip}) == [
            ("tree", {"helper2": [lib]}), ("a", {}), ("b", {"helper3": [lib]})]


def test_the_commit_two_sides_both_reach_last_is_analysed_where_neither_starts_there(
        tmp_path, monkeypatch):
    """a is cut at B0, b at B1, and the branch the union lands on is at B2."""
    directory = str(tmp_path / "fork")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    commit_dict = {}
    for name, start in (("b0", ""), ("b1", "b0"), ("b2", "b1"), ("a", "b0"), ("b", "b1")):
        _git(directory, "checkout", "-q", *(["-b", name, start] if start else ["-b", name]))
        _write(directory, f"{name}.txt", name)
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", name)
        commit_dict[name] = _git(directory, "rev-parse", "HEAD").strip()
    base = commit_dict["b0"]
    _git(directory, "checkout", "-q", "--detach", base)
    seen_list = []

    def analyse_head(project_dir, *rest):
        seen_list.append(_git(project_dir, "rev-parse", "HEAD").strip())
        return {"range": {}, "own": {}, "define": {"at": [seen_list[-1]]}}

    monkeypatch.setattr(integrate, "_analyse_head", analyse_head)
    side_dict = {name: commit_dict[name] for name in ("a", "b")}
    side_dict["tree"] = commit_dict["b2"]
    by_commit_dict: dict = {}
    fork_dict = integrate._fork_define_dict(directory, "out", base, side_dict,
                                            {"at": ["start"]}, None, by_commit_dict)
    assert fork_dict == {("b", "tree"): {"at": [commit_dict["b1"]]},
                         ("tree", "b"): {"at": [commit_dict["b1"]]}}
    assert seen_list == [commit_dict["b1"]] and list(by_commit_dict) == [commit_dict["b1"]]
    assert _git(directory, "rev-parse", "HEAD").strip() == base
    # Carried on from over B2: a and b each start from a commit of their own
    start_dict = {commit: {"commit": commit, "define": {"at": [commit]}}
                  for commit in (base, commit_dict["b1"])}
    del side_dict["tree"]
    assert integrate._fork_define_dict(directory, "out", commit_dict["b2"], side_dict,
                                       {"at": ["start"]}, None, start_dict) == {
        ("a", "b"): {"at": [base]}}
    assert len(seen_list) == 1
    assert integrate._fork_define_dict(directory, "", base, side_dict, {}, None, {}) == {}
    assert integrate._side_commit_dict(directory, ["a", "b"]) == side_dict


def test_a_name_another_side_holds_fewer_of_than_its_start_is_not_looked_for():
    """Two definitions of one name: a renamed one of them, b removed the other's file."""
    x_file, y_file = "p/x_py/x.py", "p/y_py/y.py"
    start = _analysis(name={x_file: ["X", "X.render"], y_file: ["Y", "Y.render"]})
    keep_list = integrate.keep_list_of(
        [("a", {x_file: {"X.render"}}), ("b", {y_file: {"", "Y", "Y.render"}})], start,
        {"a": {"X": [x_file], "draw": [x_file], "Y": [y_file], "render": [y_file]},
         "b": {"X": [x_file], "render": [x_file]}})
    assert keep_list == [("a", {"draw": [x_file]}), ("b", {})]
    # Nobody took a definition of the name away, so the one a changed is looked for
    assert integrate.keep_list_of(
        [("a", {x_file: {"X.render"}}), ("b", {})], start,
        {"a": start["define"], "b": start["define"]})[0] == (
            "a", {"render": [x_file, y_file]})


def test_a_side_that_only_moved_a_file_added_none_of_its_names():
    lib, core = "p/lib_py/lib.py", "p/core_py/core.py"
    start = _analysis(name={lib: ["foo", "keep"]})
    keep_list = integrate.keep_list_of(
        [("a", {lib: {"", "foo", "keep"}, core: {""}}), ("b", {lib: {"foo"}})], start,
        {"a": {"foo": [core], "keep": [core]}, "b": {"keep": [lib]}})
    assert keep_list == [("a", {"keep": [core]}), ("b", {})]


# == The lines a union may hold, read off the merge and off its sides ==


def test_the_separator_lines_of_a_block_git_wrote_are_told_by_the_names_merged_under():
    start_line = b"<<<<<<< refs/codetwine-loom/merged-so-far"
    line_list = [("f.txt", line) for line in (
        start_line, start_line + b":old.txt", b"||||||| 1234567:old.txt", b"=======",
        b">>>>>>> refs/codetwine-loom/side/a:f.txt", b"||||||| 1234567", b"=======", b"=======",
        b">>>>>>> refs/codetwine-loom/side/b",
        # A block under a name nothing was merged under, and a start with no end
        start_line, b"=======", b">>>>>>> refs/codetwine-loom/side/<branch>")]
    line_list += [("g.txt", start_line), ("g.txt", b"======="),
                  ("g.txt", b">>>>>>> refs/codetwine-loom/side/a")]
    line_list += [("s.txt", b"<" * 9 + b" refs/codetwine-loom/merged-so-far"), ("s.txt", b"=" * 9),
                  ("s.txt", b">" * 9 + b" refs/codetwine-loom/side/ab")]
    line_list += [("t.txt", b"<<< refs/codetwine-loom/merged-so-far"), ("t.txt", b"||| 1234567"),
                  ("t.txt", b"==="), ("t.txt", b">>> refs/codetwine-loom/side/b")]
    assert integrate._block_separator_dict(line_list, ["a", "b"]) == {
        "f.txt": {b"||||||| 1234567:old.txt": 1, b"=======": 2, b"||||||| 1234567": 1},
        "g.txt": {b"=======": 1}, "t.txt": {b"||| 1234567": 1, b"===": 1}}
    # What the files holding a block may hold is what is left of their separator lines
    line_list += [("f.txt", b"======="), ("f.txt", b"||||||| a table"), ("h.rst", b"=======")]
    assert integrate._separator_frame_dict(line_list, ["a", "b"]) == {
        "f.txt": {b"=======": 3, b"||||||| a table": 1}, "g.txt": {}, "t.txt": {}}


def test_a_marker_like_line_added_to_a_file_another_side_moved_is_not_the_merges(tmp_path):
    text = b"Guide\n=====\n\nintro\n\n" + b"".join(b"line %d\n" % n for n in range(8))
    directory, base = _byte_repo(tmp_path, {"guide.rst": text}, {"a_only.txt": b"a\n"},
                                 {"guide.rst": text + b"\nInstall\n=======\n\nsteps\n"})
    _git(directory, "checkout", "-q", "a")
    os.makedirs(os.path.join(directory, "docs"))
    _git(directory, "mv", "guide.rst", "docs/guide.rst")
    _git(directory, "commit", "-qm", "a moves the guide")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == []
    assert merged["marker"]["separator"] == {}
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_name_line_is_counted_over_the_whole_tree_and_named_where_no_side_holds_it():
    line = _FIXTURE_START
    marker = {"name_count": collections.Counter({line: 1}),
              "name_at": collections.Counter({("old/how.md", line): 1})}

    def path_set(count_dict):
        return integrate._marker_path_set(collections.Counter(count_dict), marker)

    # The file holding the line may be anywhere
    assert path_set({("old/how.md", line): 1}) == set()
    assert path_set({("new/how.md", line): 1}) == set()
    # One line over what the tree may hold is named where no side holds it
    assert path_set({("old/how.md", line): 1, ("f.txt", line): 1}) == {"f.txt"}
    assert path_set({("old/how.md", line): 2}) == {"old/how.md"}
    assert path_set({("new/how.md", line): 1, ("f.txt", line): 1}) == {"new/how.md", "f.txt"}
    # A line no side holds is one too many wherever it is
    assert path_set({("f.txt", b">>>>>>> refs/codetwine-loom/side/b"): 1}) == {"f.txt"}
    # Every file holding the line is named when each holds no more than a side does
    marker["name_at"][("copy/how.md", line)] = 1
    assert path_set({("old/how.md", line): 1, ("copy/how.md", line): 1}) == {
        "old/how.md", "copy/how.md"}
    assert integrate._marker_path_set(collections.Counter({("f.txt", line): 1}), {}) == {
        "f.txt"}


def test_a_separator_line_is_counted_in_the_files_git_wrote_a_block_in_alone():
    marker = {"separator": {"f.txt": collections.Counter({b"=======": 1})}}

    def path_set(count_dict):
        return integrate._marker_path_set(collections.Counter(count_dict), marker)

    assert path_set({("f.txt", b"======="): 1, ("g.rst", b"======="): 3}) == set()
    assert path_set({("f.txt", b"======="): 2}) == {"f.txt"}
    assert path_set({("f.txt", b"||||||| 1234567"): 1}) == {"f.txt"}
    # The same line held by another file, or no longer held there, changes nothing
    assert path_set({("f.txt", b"======="): 2, ("g.rst", b"======="): 0}) == {"f.txt"}


def test_the_name_lines_the_start_and_each_side_hold_are_what_the_union_may_hold(tmp_path):
    """a adds a file showing a block and b moves the one the start holds."""
    directory, base = _two_file_repo(tmp_path, "a/new.md", _FIXTURE_TEXT, "g.txt", "b\n",
                                     {"how.md": _FIXTURE_TEXT, "g.txt": "g\n"})
    _git(directory, "checkout", "-q", "b")
    os.makedirs(os.path.join(directory, "docs"))
    _git(directory, "mv", "how.md", "docs/how.md")
    _git(directory, "commit", "-qm", "b moves the document")
    _git(directory, "checkout", "-q", "--detach", base)
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    frame = integrate._name_frame_dict(directory, base, [a, b, a])
    assert frame["name_count"] == {_FIXTURE_START: 2, _FIXTURE_END: 2}
    assert frame["name_at"] == {
        (path, line): 1 for path in ("how.md", "a/new.md", "docs/how.md")
        for line in (_FIXTURE_START, _FIXTURE_END)}
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["conflicted"] == []
    assert sorted(path for path in integrate._tree_file_list(directory, merged["commit"])
                  if path.endswith(".md")) == ["a/new.md", "docs/how.md"]
    assert integrate.marker_path_list(directory, merged["marker"]) == []
    _write(directory, "g.txt", "<<<<<<< refs/codetwine-loom/merged-so-far\nb\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["g.txt"]


def _three_side_repo(tmp_path):
    """A repository whose branches a and b change one line of f.txt and whose branch c
    moves f.txt to g.txt."""
    directory, base = _two_file_repo(tmp_path, "f.txt", "ONE\ntwo\n", "f.txt", "1\ntwo\n",
                                     {"f.txt": "one\ntwo\n", "doc.rst": "Guide\n=======\n"})
    _git(directory, "checkout", "-q", "-b", "c", base)
    _git(directory, "mv", "f.txt", "g.txt")
    _git(directory, "commit", "-qm", "c")
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


def test_a_block_in_a_file_a_later_side_moved_is_found_where_the_file_is_now(tmp_path):
    directory, base = _three_side_repo(tmp_path)
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    assert merged["conflicted"] == ["f.txt"] and merged["unmarked"] == []
    assert merged["marker"]["separator"] == {"g.txt": {b"=======": 0}}
    assert "<<<<<<< refs/codetwine-loom/merged-so-far" in _read(directory, "g.txt")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["g.txt"]
    # One separator of the block left behind is still a marker of the merge's
    _write(directory, "g.txt", "ONE\n=======\n1\ntwo\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["g.txt"]
    _write(directory, "g.txt", "ONE 1\ntwo\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_separator_left_of_a_block_is_not_made_up_for_by_a_line_another_file_lost(
        tmp_path):
    directory, base = _three_side_repo(tmp_path)
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 0}}
    _write(directory, "f.txt", "ONE\n=======\n1\ntwo\n")
    os.remove(os.path.join(directory, "doc.rst"))
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    _write(directory, "doc.rst", "Guide\n-----\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]


def test_a_block_whose_markers_are_shorter_than_seven_is_found(tmp_path):
    directory, base = _two_file_repo(
        tmp_path, "f.txt", "ONE\ntwo\n", "f.txt", "1\ntwo\n",
        {"f.txt": "one\ntwo\n", ".gitattributes": "*.txt conflict-marker-size=3\n"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    assert _read(directory, "f.txt").startswith("<<< refs/codetwine-loom/merged-so-far\n")
    assert merged["marker"]["size"] == {"f.txt": 3} and merged["unmarked"] == []
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    _write(directory, "f.txt", "ONE\n===\n1\ntwo\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    _write(directory, "f.txt", "ONE 1\n=======\ntwo\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_the_marker_lines_of_a_union_carried_on_from_are_held_to_the_merge_of_its_sides(repo):
    """The union kept one line of a block git wrote; the branch merged after it holds a
    marker-like line of its own."""
    directory, base = repo
    _union_branch_with(directory, base, {"f.txt": "ONE\n=======\ntwo\nthree\n"})
    _git(directory, "checkout", "-q", "-b", "c", base)
    _write(directory, "c.rst", "Heading\n=======\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "c")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["union", "c"], resume="union")
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 0}}
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    _write(directory, "f.txt", "ONE and 1\ntwo\nthree\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def _branch_c(directory, base, file_dict):
    """Commit file_dict on a branch named c off base and come back to base."""
    _git(directory, "checkout", "-q", "-b", "c", base)
    for name, text in file_dict.items():
        _write(directory, name, text)
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "c")
    _git(directory, "checkout", "-q", "--detach", base)


def test_a_marker_like_line_a_branch_merged_after_an_unsettled_union_added_may_stay(repo):
    """The branch merged after the union adds an underline to the file the union's
    sides conflicted in, away from the block."""
    directory, base = repo
    _union_branch_with(directory, base, {"f.txt": "ONE and 1\ntwo\nthree\n"})
    _branch_c(directory, base, {"f.txt": "one\ntwo\nthree\n\nInstall\n=======\n"})
    merged = integrate.assemble(directory, base, ["union", "c"], resume="union")
    assert _read(directory, "f.txt") == "ONE and 1\ntwo\nthree\n\nInstall\n=======\n"
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 1}}
    assert integrate.marker_path_list(directory, merged["marker"]) == []
    _write(directory, "f.txt", "ONE and 1\n=======\ntwo\nthree\n\nInstall\n=======\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]


def test_a_separator_an_unsettled_union_kept_is_found_beside_a_new_block_in_its_file(repo):
    """The union kept one line of a block; the branch merged after it conflicts with
    the union in another line of the same file."""
    directory, base = repo
    _union_branch_with(directory, base, {"f.txt": "ONE\n=======\ntwo\n3\n"})
    _branch_c(directory, base, {"f.txt": "one\ntwo\nTHREE\n"})
    merged = integrate.assemble(directory, base, ["union", "c"], resume="union")
    assert merged["conflicted"] == ["f.txt"] and merged["unmarked"] == []
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 0}}
    _write(directory, "f.txt", "ONE\n=======\ntwo\n3 THREE\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    _write(directory, "f.txt", "ONE and 1\ntwo\n3 THREE\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_block_written_where_the_sides_merge_alone_is_held_to_the_union_s_own_lines(repo):
    """The union changed a file none of its sides did; the branch merged after it
    changes the same line."""
    directory, base = repo
    _union_branch_with(directory, base, {"f.txt": "ONE and 1\ntwo\nthree\n",
                                         "g.txt": "Kept\n=======\n"})
    _branch_c(directory, base, {"g.txt": "kept by c\n"})
    merged = integrate.assemble(directory, base, ["union", "c"], resume="union")
    assert merged["conflicted"] == ["f.txt", "g.txt"] and merged["unmarked"] == []
    assert merged["marker"]["separator"] == {"f.txt": {}, "g.txt": {b"=======": 1}}
    _write(directory, "g.txt", "Kept\n=======\n=======\nkept by c\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["g.txt"]
    _write(directory, "g.txt", "Kept by c\n=======\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_block_an_unsettled_union_holds_is_found_under_the_names_it_was_written_with(
        repo):
    """The pass before wrote the block under the branches' names; this one merges the
    same commits under side_name()'s names."""
    directory, base = repo
    _union_branch_with(directory, base, {"h.txt": _BLOCK_TEXT})
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["h.txt"]
    os.rename(os.path.join(directory, "h.txt"), os.path.join(directory, "moved.txt"))
    assert integrate.marker_path_list(directory, merged["marker"]) == ["moved.txt"]


def _leave_union(directory, merged, branch, file_dict, is_record=True):
    """Write file_dict over the union in the working tree and commit the tree as the
    union a run leaves unsettled: the base and the commits merged as its parents and,
    with is_record, the separator lines it may hold and the sides not merged into a
    file of it in a trailer each. The repository is left clean at merged["start"]."""
    for name, text in file_dict.items():
        _write(directory, name, text)
    _git(directory, "add", "-A")
    tree = _git(directory, "write-tree").strip()
    argv = ["commit-tree", tree]
    for parent in (merged["base"], *merged["parents"]):
        argv += ["-p", parent]
    trailer = "" if not is_record else (
        f"\n{integrate.SEPARATOR_TRAILER}: "
        f"{integrate.separator_text(integrate.separator_record_dict(merged))}"
        f"\n{integrate.NO_MERGE_TRAILER}: "
        f"{integrate.no_merge_text(integrate.no_merge_record_list(merged))}")
    commit = _git(directory, *argv, "-m",
                  f"integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: unsettled{trailer}").strip()
    _git(directory, "branch", branch, commit)
    _git(directory, "reset", "-q", "--hard")
    _git(directory, "checkout", "-q", "--detach", merged["start"])
    return commit


def test_a_separator_a_later_pass_left_is_found_by_every_pass_after_it(repo):
    """The second pass merges a branch that conflicts with what the first pass wrote
    in a file its sides merged alone, and leaves the "=" line of that block."""
    directory, base = repo
    _union_branch_with(directory, base, {"f.txt": "ONE and 1\ntwo\nthree\n",
                                         "g.txt": "Kept\n"})
    _branch_c(directory, base, {"g.txt": "kept by c\n"})
    merged = integrate.assemble(directory, base, ["union", "c"], resume="union")
    assert merged["conflicted"] == ["f.txt", "g.txt"]
    assert merged["marker"]["separator"] == {"f.txt": {}, "g.txt": {b"=======": 0}}
    left_dict = {"g.txt": "Kept\n=======\nkept by c\n"}
    _leave_union(directory, merged, "union2", left_dict)
    for earlier, branch in (("union2", "union3"), ("union3", "union4")):
        merged = integrate.assemble(directory, base, [earlier], resume=earlier)
        assert [entry_dict["branch"] for entry_dict in merged["branches"]] == [
            integrate.side_name(_git(directory, "rev-parse", name).strip())
            for name in ("a", "b", "c")]
        assert merged["marker"]["separator"] == {"f.txt": {}, "g.txt": {b"=======": 0}}
        assert integrate.marker_path_list(directory, merged["marker"]) == ["g.txt"]
        _leave_union(directory, merged, branch, left_dict)
    merged = integrate.assemble(directory, base, ["union4"], resume="union4")
    _write(directory, "g.txt", "Kept by c\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_separator_left_where_the_tree_merged_onto_moved_on_is_found_afterwards(repo):
    """The second pass is written against a tree that changed the line the first pass
    wrote, and leaves the "=" line of that block."""
    directory, base = repo
    _union_branch_with(directory, base, {"f.txt": "ONE and 1\ntwo\nthree\n",
                                         "g.txt": "Kept\n"})
    _git(directory, "checkout", "-q", "-b", "tree", base)
    _write(directory, "g.txt", "kept by the tree\n")
    _git(directory, "commit", "-qam", "tree")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["union"], onto="tree", resume="union")
    assert merged["marker"]["separator"] == {"f.txt": {}, "g.txt": {b"=======": 0}}
    _leave_union(directory, merged, "union2", {"g.txt": "kept by the tree\n=======\nKept\n"})
    tree = _git(directory, "rev-parse", "tree").strip()
    _git(directory, "checkout", "-q", "--detach", tree)
    merged = integrate.assemble(directory, tree, ["union2"], resume="union2")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["g.txt"]
    _write(directory, "g.txt", "Kept by the tree\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_an_unsettled_union_passed_as_any_branch_is_read_as_what_it_was_merged_from(repo):
    """The block the union holds is no line a side holds itself, whichever flag names
    the union."""
    directory, base = repo
    _union_branch_with(directory, base, {"f.txt": _BLOCK_TEXT})
    _branch_c(directory, base, {"g.txt": "kept by c\n"})
    name_list = [integrate.side_name(_git(directory, "rev-parse", name).strip())
                 for name in ("a", "b")]
    for resume in ("union", ""):
        merged = integrate.assemble(directory, base, ["union", "c"], resume=resume)
        assert [entry_dict["branch"] for entry_dict in merged["branches"]] == [*name_list, "c"]
        assert merged["marker"]["name_count"] == {}
        assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
        _git(directory, "reset", "-q", "--hard")
    # A merge carrying none of this tool's trailers is one side, with the lines it holds
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    _git(directory, "branch", "plain", _git(
        directory, "commit-tree", "union^{tree}", "-p", base, "-p", a, "-p", b, "-m",
        "merged by hand").strip())
    merged = integrate.assemble(directory, base, ["plain", "c"])
    assert [entry_dict["branch"] for entry_dict in merged["branches"]] == ["plain", "c"]
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_two_unsettled_unions_merged_together_each_bring_what_they_left(repo):
    """One is carried on from and the other passed beside it; each holds a separator
    line of a block in a file of its own."""
    directory, base = repo
    _union_branch_with(directory, base, {"f.txt": "ONE\n=======\ntwo\nthree\n"})
    _union_branch_with(directory, base, {"f.txt": "ONE and 1\ntwo\nthree\n",
                                         "a_only.txt": "a\n=======\n"},
                       record_dict={"f.txt": {}, "a_only.txt": {}}, branch="other")
    merged = integrate.assemble(directory, base, ["union", "other"], resume="union")
    assert merged["marker"]["separator"]["a_only.txt"] == {b"=======": 0}
    assert merged["marker"]["separator"]["f.txt"] == {b"=======": 0}
    assert "a_only.txt" in integrate.marker_path_list(directory, merged["marker"])
    _write(directory, "f.txt", "ONE and 1\ntwo\nthree\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["a_only.txt"]
    _write(directory, "a_only.txt", "a\n")
    assert integrate.marker_path_list(directory, merged["marker"]) == []


@pytest.mark.parametrize("trailer", [
    "", "Codetwine-Loom-Separator: not json", "Codetwine-Loom-Separator: [1]",
    'Codetwine-Loom-Separator: {"g.txt":3}', 'Codetwine-Loom-Separator: {"g.txt":{"=======":-1}}',
    'Codetwine-Loom-Separator: {"g.txt":{"=======":true}}',
    'Codetwine-Loom-Separator: {"g.txt":{"x=======":1}}', 'Codetwine-Loom-Separator: {"":{}}',
    'Codetwine-Loom-Separator: {"g.txt":{"=======":1}}\nCodetwine-Loom-Separator: {}',
    pytest.param("Codetwine-Loom-Separator: {" + '"p":{},' * 20000 + '"q":{}}', id="too long")])
def test_a_union_that_records_no_separator_line_it_may_hold_may_hold_none(repo, trailer):
    """The commit carries no record, or one that cannot be read."""
    directory, base = repo
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    _union_branch_with(directory, base, {"f.txt": "ONE and 1\ntwo\nthree\n",
                                         "g.txt": "Kept\n=======\n"}, record_dict=None)
    message = "integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: unsettled\n" + trailer
    commit = subprocess.run(
        ["git", "-C", directory, "commit-tree", "union^{tree}", "-p", base, "-p", a,
         "-p", b, "-F", "-"], input=message, capture_output=True, text=True,
        check=True).stdout.strip()
    _git(directory, "branch", "bare", commit)
    assert integrate._separator_record_at(directory, commit) is None
    merged = integrate.assemble(directory, base, ["bare"], resume="bare")
    assert merged["marker"]["separator"] == {"g.txt": {b"=======": 0}}
    assert integrate.marker_path_list(directory, merged["marker"]) == ["g.txt"]


def test_the_separator_lines_a_union_may_hold_are_read_back_as_they_were_recorded(repo):
    directory, base = repo
    separator_dict = {"dir with space/q \"x\".rst": collections.Counter(
        {b"=======": 2, b"======= \t two  spaces": 1, b"||||||| caf\xe9": 1}),
        "f.txt": collections.Counter(), "caf\\xe9.txt": collections.Counter({b"=======": 0})}
    record_dict = integrate.separator_record_dict({"marker": {"separator": separator_dict}})
    assert integrate.is_separator_record(record_dict)
    assert record_dict["f.txt"] == {} and record_dict["caf\\xe9.txt"] == {"=======": 0}
    text = integrate.separator_text(record_dict)
    assert not re.search(r"\s", text) and text.isascii()
    commit = _git(directory, "commit-tree", f"{base}^{{tree}}", "-p", base, "-m",
                  "integrate: x\n\nCodetwine-Loom-Verdict: unsettled\n"
                  f"Codetwine-Loom-Separator: {text}").strip()
    assert integrate._separator_record_at(directory, commit) == {
        path: +count_dict for path, count_dict in separator_dict.items()}
    assert integrate._separator_record_at(directory, base) is None
    assert integrate.separator_text(None) == "" and integrate.separator_text({}) == "{}"
    assert integrate.separator_record_dict({}) == {}


def test_a_record_of_separator_lines_too_long_for_a_commit_is_not_written(caplog):
    record_dict = {f"src/file_{n}.py": {} for n in range(5000)}
    assert integrate.is_separator_record(record_dict)
    assert integrate.separator_text(record_dict) == ""
    assert "none is recorded" in caplog.text
    few_dict = {f"src/file_{n}.py": {"=======": 1} for n in range(100)}
    assert json.loads(integrate.separator_text(few_dict)) == few_dict


@pytest.mark.parametrize("value", [
    None, [], "x", {"f.py": []}, {"f.py": {"=======": "1"}}, {"f.py": {"=======": 1.0}},
    {"f.py": {"=======": True}}, {"f.py": {"=======": -1}}, {"f.py": {"=======": 10 ** 7}},
    {"f.py": {"<<<<<<<": 1}}, {"f.py": {"": 1}}, {"f.py": {"=======\u3042": 1}},
    {"": {}}, {"a\0b": {}}, {"p" * 5000: {}}, {1: {}}, {"f.py": {7: 1}}])
def test_a_record_of_another_shape_is_no_record_of_separator_lines(value):
    assert integrate.is_separator_record(value) is False


def test_the_lines_a_union_left_in_a_file_no_path_is_found_for_are_looked_for_in_new_files(
        repo, monkeypatch):
    """The branch merged after the union moves the file the union left a separator
    line in; where git names no path it moved to, the files the union did not hold
    are held to the line."""
    directory, base = repo
    _union_branch_with(directory, base, {"f.txt": "ONE\n=======\ntwo\nthree\n"})
    _git(directory, "checkout", "-q", "-b", "c", base)
    _git(directory, "mv", "f.txt", "moved.txt")
    _write(directory, "new.rst", "Heading\n=======\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "c")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["union", "c"], resume="union")
    assert merged["marker"]["separator"] == {"moved.txt": {b"=======": 0}}
    assert integrate.marker_path_list(directory, merged["marker"]) == ["moved.txt"]
    _git(directory, "reset", "-q", "--hard")
    monkeypatch.setattr(integrate, "_move_dict", lambda *args: {})
    merged = integrate.assemble(directory, base, ["union", "c"], resume="union")
    assert merged["marker"]["separator"] == {"moved.txt": {b"=======": 0}, "new.rst": {b"=======": 0}}
    assert integrate.marker_path_list(directory, merged["marker"]) == ["moved.txt", "new.rst"]


# == A file left under the name git made up for it ==========


@pytest.mark.parametrize("path, expect", [
    ("docs~refs_codetwine-loom_merged-so-far", True),
    ("docs~refs_codetwine-loom_merged-so-far_0", True),
    ("pkg/x.py~refs_codetwine-loom_side_task_1", True),
    ("docs~refs_codetwine-loom_merged-so-far.bak", False),
    ("docs~other", False), ("d~refs_codetwine-loom_side_a/x.py", False),
    ("refs_codetwine-loom_side_a", False)])
def test_a_stand_in_name_is_the_path_and_the_ref_of_the_side_the_file_is_from(path, expect):
    assert bool(integrate._STAND_IN_NAME.search(path)) is expect


def test_a_file_under_the_name_git_made_up_for_it_is_found_until_it_is_moved(tmp_path):
    directory, base = _two_file_repo(tmp_path, "docs", "notes of a\n", "docs/guide.md",
                                     "# guide of b\n")
    merged = integrate.assemble(directory, base, ["a", "b"])
    stand_in = "docs~refs_codetwine-loom_merged-so-far"
    assert merged["stand_in"] == [stand_in]
    assert integrate._stand_in_now_list(directory, merged["stand_in"]) == [stand_in]
    os.rename(os.path.join(directory, stand_in), os.path.join(directory, "docs_a.txt"))
    assert integrate._stand_in_now_list(directory, merged["stand_in"]) == []


def test_a_file_a_side_holds_under_such_a_name_is_no_stand_in(tmp_path):
    directory, base = _two_file_repo(tmp_path, "notes~refs_codetwine-loom_side_x", "kept by a\n",
                                     "g.txt", "b\n")
    assert integrate.assemble(directory, base, ["a", "b"])["stand_in"] == []


def test_a_stand_in_name_left_in_the_tree_keeps_the_merge_from_settling(tmp_path,
                                                                        monkeypatch):
    directory, base = _two_file_repo(tmp_path, "docs", "notes of a\n", "docs/guide.md",
                                     "# guide of b\n")
    stand_in = "docs~refs_codetwine-loom_merged-so-far"

    class _StageThatReadsTheStandIn:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            kwargs["turn_note"](1, 20)
            _read_on(1, stand_in)
            return {"answer": "done", "history": [], "iters": 1, "refused_completes": 0,
                    "error": "", "outcome": "complete"}

    prepared = {"base": base, "merged": integrate.assemble(directory, base, ["a", "b"]),
                "before": _analysis(), "against": _analysis(), "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "INTEGRATE_MAX_REFUSAL", 0)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(integrate, "RLM", _StageThatReadsTheStandIn)
    out = integrate.run_integration(prepared, "", "p", "out")
    assert out["called"] is True and out["settled"] is False
    # a, whose file git moved there, is held to its own version as well
    assert out["left"] == [integrate._STAND_IN_REASON.format(paths=stand_in),
                           integrate._NO_BRANCH_READ_REASON.format(
                               path=stand_in, branch="a", side_path="docs")]


# == A path of an earlier merge that nothing holds any more ==


def test_a_path_neither_the_tree_nor_a_branch_holds_is_not_asked_for(unmarked_repo,
                                                                    monkeypatch):
    directory, base = unmarked_repo
    merged = integrate.assemble(directory, base, ["a", "b"])
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {
        "": base, **{e["branch"]: e["commit"] for e in merged["branches"]}}
    assert integrate._no_read_reason_list(["res.rc~refs_codetwine-loom_merged-so-far",
                                           "pkg/moved.py"]) == []
    # A directory of the tree, and a link that leads out of the repository, are held
    os.mkdir(os.path.join(directory, "sub"))
    os.symlink(os.path.join(os.path.dirname(directory), "outside"),
               os.path.join(directory, "out_link"))
    assert integrate._no_read_reason_list(["sub", "out_link"]) == [
        integrate._NO_TOOL_READ_REASON.format(paths="sub, out_link")]


# == A link read on a branch ================================


def test_a_link_is_read_on_a_branch_as_the_path_it_holds(tmp_path, monkeypatch):
    directory = str(tmp_path / "link")
    outside = str(tmp_path / "outside.txt")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    _write(directory, "k.py", "X = 1\n")
    os.symlink("k.py", os.path.join(directory, "link"))
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "base")
    base = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "-b", "a")
    os.remove(os.path.join(directory, "link"))
    os.symlink(outside, os.path.join(directory, "link"))
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "a")
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {
        "": base, "a": _git(directory, "rev-parse", "HEAD").strip()}
    assert integrate.read_branch_file("a", "link") == outside
    assert integrate.read_branch_file("", "link") == "k.py"
    assert integrate._no_read_reason_list(["link"]) == [
        integrate._NO_READ_BRANCH_REASON.format(
            path="link", branches=f"{integrate._START_BRANCH_TEXT}, a")]
    tool = next(t for t in integrate.merge_tool_list() if t.__name__ == "read_branch_file")
    tool("a", "link")
    assert integrate._no_read_reason_list(["link"]) == []


# == What a gap test's run wrote in a directory made for it ==


def test_a_directory_made_for_a_gap_test_goes_with_what_the_run_wrote_in_it(
        _gap_project, monkeypatch):
    root, _ = _gap_project
    (root / "tests").mkdir()
    (root / "tests" / "kept.py").write_text("x = 1\n", encoding="utf-8")

    def run_test(path_list=None):
        for directory in ("tests/new/deep/__pycache__", "tests/new/__pycache__",
                          "tests/__pycache__"):
            os.makedirs(root / directory)
            (root / directory / "x.pyc").write_bytes(b"0")
        return {"status": "failed", "failed": ["t::g"], "failed_excerpt": "",
                "targets": [], "output": ""}

    monkeypatch.setattr(integrate.edit, "run_test", run_test)
    monkeypatch.setattr(config, "TEST_TARGET_STYLE", "path")
    assert integrate.gap_test_result("tests/new/deep/test_g.py", "the test\n",
                                     frozenset()) is True
    assert not (root / "tests" / "new").exists()
    assert (root / "tests" / "kept.py").exists()
    assert (root / "tests" / "__pycache__" / "x.pyc").exists()


def test_a_decisions_trailer_nested_too_deep_to_read_holds_no_decision():
    assert integrate.decision_list_of("[" * 100_000 + "]" * 100_000) == []


# == A file holding a conflict block is merged no further ==


def _line_text(change_dict=None):
    """Twelve numbered lines, those of change_dict rewritten with its word."""
    line_list = [f"line {index} v0" for index in range(12)]
    for index, word in (change_dict or {}).items():
        line_list[index] = f"line {index} {word}"
    return "\n".join(line_list) + "\n"


def _line_repo(tmp_path, *change_tuple, name="f.txt", base_file_dict=None):
    """A repository whose branches a, b, c, ... each rewrite some lines of one file of
    twelve lines, all off one commit."""
    directory, base = _two_file_repo(tmp_path, name, _line_text(change_tuple[0]), name,
                                     _line_text(change_tuple[1]),
                                     {name: _line_text(), **(base_file_dict or {})})
    for branch, change_dict in zip("cde", change_tuple[2:]):
        _git(directory, "checkout", "-q", "-b", branch, base)
        _write(directory, name, _line_text(change_dict))
        _git(directory, "commit", "-qam", branch)
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


_FIVE_A = {index: "A" for index in range(3, 8)}
_FIVE_B = {index: "B" for index in range(3, 8)}


@pytest.mark.parametrize("change_tuple, held_list", [
    # c changes the middle one of the five lines a and b both rewrote
    ((_FIVE_A, _FIVE_B, {5: "C"}), [{}, {}, {"f.txt": "f.txt"}]),
    # a leaves that line as it was
    (({3: "A", 4: "A", 6: "A", 7: "A"}, _FIVE_B, {5: "C"}), [{}, {}, {"f.txt": "f.txt"}]),
    # c changes the line after the block, and the one before it
    (({4: "A"}, {4: "B"}, {5: "C"}), [{}, {}, {"f.txt": "f.txt"}]),
    (({4: "A"}, {4: "B"}, {3: "C"}), [{}, {}, {"f.txt": "f.txt"}]),
    # c changes a line away from the block, and the line of the block itself
    (({4: "A"}, {4: "B"}, {9: "C"}), [{}, {}, {"f.txt": "f.txt"}]),
    (({4: "A"}, {4: "B"}, {4: "C"}), [{}, {}, {"f.txt": "f.txt"}]),
    # Four sides, the third and the fourth changing the file
    ((_FIVE_A, _FIVE_B, {5: "C"}, {10: "D"}), [{}, {}, {"f.txt": "f.txt"}, {"f.txt": "f.txt"}]),
    # No side conflicts with another
    (({1: "A"}, {5: "B"}, {9: "C"}), [{}, {}, {}]),
])
def test_a_side_changing_a_file_that_holds_a_conflict_block_is_not_merged_into_it(
        tmp_path, change_tuple, held_list):
    directory, base = _line_repo(tmp_path, *change_tuple)
    two = integrate.assemble(directory, base, ["a", "b"])
    two_text = _read(directory, "f.txt")
    _git(directory, "reset", "-q", "--hard")
    merged = integrate.assemble(directory, base, list("abcd"[:len(change_tuple)]))
    assert [entry_dict["in_block"] for entry_dict in merged["branches"]] == held_list
    assert integrate.in_block_read_list(merged) == [
        (entry_dict["branch"], "f.txt", "f.txt")
        for entry_dict in merged["branches"] if entry_dict["in_block"]]
    if any(held_list):
        # The file is as the merge of a and b left it, in the commit and on disk
        assert _read(directory, "f.txt") == two_text
        assert _git(directory, "rev-parse", f"{merged['commit']}:f.txt") == _git(
            directory, "rev-parse", f"{two['commit']}:f.txt")
        assert [entry_dict["conflicted"] for entry_dict in merged["branches"][2:]] == [
            [] for _ in merged["branches"][2:]]
        assert merged["marker"]["separator"] == {"f.txt": {b"=======": 0}}
    else:
        assert "line 9 C" in _read(directory, "f.txt")


def test_a_file_put_back_as_it_was_is_one_tree_in_the_commit_the_index_and_on_disk(
        tmp_path):
    """c changes the file a and b conflict in and one more file."""
    directory, base = _line_repo(tmp_path, _FIVE_A, _FIVE_B, {5: "C"},
                                 base_file_dict={"g.txt": "one\n"})
    _git(directory, "checkout", "-q", "c")
    _write(directory, "g.txt", "one\ntwo\n")
    _git(directory, "commit", "-qam", "c2")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    assert [entry_dict["conflicted"] for entry_dict in merged["branches"]] == [
        [], ["f.txt"], []]
    assert merged["unmarked"] == []
    text = _read(directory, "f.txt")
    assert "||||||| " in text and "line 5 C" not in text and "line 5 B" in text
    assert _read(directory, "g.txt") == "one\ntwo\n"
    # The working tree and the index hold the tree of the union's commit
    assert _git(directory, "write-tree") == _git(directory, "rev-parse",
                                                 f"{merged['commit']}^{{tree}}")
    assert _git(directory, "diff", "--name-only") == ""
    assert _git(directory, "rev-parse", "HEAD").strip() == base


def test_a_block_a_side_holds_itself_keeps_no_later_side_out_of_its_file(tmp_path):
    """The start holds a block under this tool's names in a document; c rewrites a line
    inside it and a line of the file a and b conflict in."""
    doc = ("<<<<<<< refs/codetwine-loom/merged-so-far\nold\n||||||| 1234567\nwas\n=======\nnew\n"
           ">>>>>>> refs/codetwine-loom/side/b\n")
    directory, base = _line_repo(tmp_path, {4: "A"}, {4: "B"}, {9: "C"},
                                 base_file_dict={"how.md": doc})
    _git(directory, "checkout", "-q", "c")
    _write(directory, "how.md", doc.replace("was", "had been"))
    _git(directory, "commit", "-qam", "c2")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    assert "had been" in _read(directory, "how.md")
    assert [entry_dict["in_block"] for entry_dict in merged["branches"]] == [
        {}, {}, {"f.txt": "f.txt"}]


def test_a_tree_is_written_with_files_as_they_are_given(tmp_path):
    directory, base = _line_repo(tmp_path, {4: "A"}, {4: "B"})
    os.makedirs(os.path.join(directory, "dir", "sub"))
    _write(directory, "dir/sub/g.txt", "one\n")
    _write_raw(directory, b"caf\xe9.txt", b"old\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "more")
    head = _git(directory, "rev-parse", "HEAD").strip()
    entry_dict = integrate._file_entry_dict(directory, head)
    assert sorted(entry_dict) == ["caf\\xe9.txt", "dir/sub/g.txt", "f.txt"]
    assert entry_dict["caf\\xe9.txt"][2] == b"caf\xe9.txt"
    blob = entry_dict["f.txt"][1]
    index_before = _git(directory, "ls-files", "-s")
    tree = integrate._tree_with_entry(directory, head, [
        (b"100644", blob, b"dir/sub/g.txt"), (b"100755", blob, b"caf\xe9.txt")])
    after_dict = integrate._file_entry_dict(directory, tree)
    assert {path: entry[:2] for path, entry in after_dict.items()} == {
        "caf\\xe9.txt": (b"100755", blob), "dir/sub/g.txt": (b"100644", blob),
        "f.txt": (b"100644", blob)}
    assert _git(directory, "ls-files", "-s") == index_before
    assert _git(directory, "status", "--porcelain") == ""
    with pytest.raises(RuntimeError, match="could not write the merged tree"):
        integrate._tree_with_entry(directory, head, [(b"100644", b"0" * 39 + b"1", b"x")])


def test_a_file_a_merge_changed_is_put_back_where_git_moved_it_and_a_link_stays_one(
        tmp_path):
    """One side moves the file, changes it and makes another file a link."""
    directory, base = _line_repo(tmp_path, {4: "A"}, {4: "B"},
                                 base_file_dict={"k.txt": "plain\n"})
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    _git(directory, "checkout", "-q", "b")
    _git(directory, "mv", "f.txt", "g.txt")
    _write(directory, "g.txt", _line_text({9: "B"}))
    os.remove(os.path.join(directory, "k.txt"))
    os.symlink("g.txt", os.path.join(directory, "k.txt"))
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "b moves it")
    b = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "--detach", base)
    tree = _git(directory, "merge-tree", "--write-tree", a, b).split("\n")[0]
    new_tree, row_list = integrate._keep_block_file(
        directory, tree, a, b, {"f.txt", "k.txt", "gone.txt"}, set())
    assert row_list == [(False, "f.txt", "g.txt", "g.txt", True),
                        (False, "k.txt", "k.txt", "k.txt", True)]
    entry_dict = integrate._file_entry_dict(directory, new_tree)
    assert entry_dict["g.txt"][:2] == integrate._file_entry_dict(directory, a)["f.txt"][:2]
    assert entry_dict["k.txt"][:2] == integrate._file_entry_dict(directory, a)["k.txt"][:2]
    # A file the merge left as it was is no file put back, and no set reads no tree
    assert integrate._keep_block_file(directory, tree, a, b, {"other.txt"}, {"k.txt"}) == (
        tree, [(True, "k.txt", "k.txt", "k.txt", False)])
    assert integrate._keep_block_file(directory, tree, a, b, set(), set()) == (tree, [])


def test_a_branch_merged_after_an_unsettled_union_is_not_merged_into_a_file_it_left_a_block_in(
        tmp_path):
    """The union kept the block of its two sides; the branch merged after it changes a
    line the block holds as what the line was."""
    directory, base = _line_repo(tmp_path, _FIVE_A, _FIVE_B, {5: "C"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    union = _leave_union(directory, merged, "union", {})
    union_text = _git(directory, "show", f"{union}:f.txt")
    assert "<<<<<<< refs/codetwine-loom/merged-so-far" in union_text
    merged = integrate.assemble(directory, base, ["union", "c"], resume="union")
    # a and b, whose versions of the file the union holds as a block, are held to it
    # as well
    assert [entry_dict["in_block"] for entry_dict in merged["branches"]] == [
        {"f.txt": "f.txt"}] * 3
    assert _read(directory, "f.txt") == union_text
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 0}}
    # The three sides merged once more from the union's own commits
    _leave_union(directory, merged, "union2", {})
    merged = integrate.assemble(directory, base, ["union2"], resume="union2")
    assert [entry_dict["in_block"] for entry_dict in merged["branches"]] == [
        {"f.txt": "f.txt"}] * 3
    assert _read(directory, "f.txt") == union_text


@pytest.mark.parametrize("is_onto", [True, False])
def test_what_was_merged_before_an_unsettled_union_is_not_merged_into_a_file_it_left_a_block_in(
        tmp_path, is_onto):
    """The union kept the block of a and b. c, which changes a line the block holds as
    what the line was, is what the union is written onto or a branch passed before it."""
    directory, base = _line_repo(tmp_path, _FIVE_A, _FIVE_B, {5: "C"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    union = _leave_union(directory, merged, "union", {})
    union_text = _git(directory, "show", f"{union}:f.txt")
    if is_onto:
        merged = integrate.assemble(directory, base, ["union"], onto="c", resume="union")
    else:
        merged = integrate.assemble(directory, base, ["c", "union"])
    # a and b, whose versions the union holds as a block, are held to the file too
    side_list = [integrate.side_name(merged["branches"][i]["commit"]) for i in (1, 2)]
    assert [(entry_dict["branch"], entry_dict["in_block"], entry_dict["conflicted"])
            for entry_dict in merged["branches"]] == [
        ("c", {"f.txt": "f.txt"}, []), (side_list[0], {"f.txt": "f.txt"}, []),
        (side_list[1], {"f.txt": "f.txt"}, ["f.txt"])]
    assert _read(directory, "f.txt") == union_text
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 0}}
    assert integrate.in_block_read_list(merged) == [
        (side, "f.txt", "f.txt") for side in ("c", *side_list)]


@pytest.mark.parametrize("drop_tuple", [("<<<<<<<", ">>>>>>>"), ("<<<<<<<",), (">>>>>>>",)])
def test_a_branch_is_not_merged_into_a_file_an_unsettled_union_left_part_of_a_block_in(
        tmp_path, drop_tuple):
    """The union's file holds the separator lines of its block and not all its name
    lines."""
    directory, base = _line_repo(tmp_path, _FIVE_A, _FIVE_B, {5: "C"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    part_text = "".join(line for line in _read(directory, "f.txt").splitlines(True)
                        if not line.startswith(drop_tuple))
    assert "||||||| " in part_text and "=======\n" in part_text
    _leave_union(directory, merged, "union", {"f.txt": part_text})
    merged = integrate.assemble(directory, base, ["union", "c"], resume="union")
    assert merged["branches"][2]["in_block"] == {"f.txt": "f.txt"}
    assert _read(directory, "f.txt") == part_text
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]


def test_a_branch_is_merged_into_a_file_an_unsettled_union_left_no_line_of_a_block_in(
        tmp_path):
    """The union settled its block; the branch merged after it changes a line beside
    what the block held."""
    directory, base = _line_repo(tmp_path, {4: "A"}, {4: "B"}, {9: "C"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    _leave_union(directory, merged, "union", {"f.txt": _line_text({4: "A and B"})})
    merged = integrate.assemble(directory, base, ["union", "c"], resume="union")
    # a and b, neither of whose lines the union holds as they wrote them, are held to
    # the file; c is merged into it
    assert [entry_dict["in_block"] for entry_dict in merged["branches"]] == [
        {"f.txt": "f.txt"}, {"f.txt": "f.txt"}, {}]
    assert _read(directory, "f.txt") == _line_text({4: "A and B", 9: "C"})
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_side_is_not_merged_into_one_of_two_blocks_of_the_same_text(tmp_path):
    """a and b rewrite the same lines of two places the same way; c changes one line of
    the second place."""
    region = [f"    step_{index}()" for index in range(5)]
    text = "\n".join(["first:", *region, "end", "", "second:", *region, "end", ""])
    a_text = text.replace("()", "(fast)")
    b_text = text.replace("    step", "    wait step")
    head, _, tail = text.partition("second:")
    directory, base = _two_file_repo(tmp_path, "f.txt", a_text, "f.txt", b_text,
                                     {"f.txt": text})
    _branch_c(directory, base, {"f.txt": head + "second:" + tail.replace(
        "step_2()", "step_2(again)")})
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    now_text = _read(directory, "f.txt")
    assert now_text.count("<<<<<<< refs/codetwine-loom/merged-so-far\n") == 2
    assert "again" not in now_text
    assert merged["branches"][2]["in_block"] == {"f.txt": "f.txt"}


def test_a_file_is_read_on_a_side_at_the_path_that_side_holds_it_at(tmp_path):
    """c moves the file a and b conflict in and changes the middle line of it."""
    directory, base = _line_repo(tmp_path, _FIVE_A, _FIVE_B, {5: "C"})
    _git(directory, "checkout", "-q", "c")
    _git(directory, "mv", "f.txt", "g.txt")
    _git(directory, "commit", "-qm", "c moves it")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    assert merged["branches"][2]["in_block"] == {"g.txt": "g.txt"}
    assert "line 5 C" not in _read(directory, "g.txt") and "||||||| " in _read(
        directory, "g.txt")
    assert not os.path.exists(os.path.join(directory, "f.txt"))
    a, c = (_git(directory, "rev-parse", name).strip() for name in ("a", "c"))
    assert integrate._fork_path_dict(directory, merged["commit"], c)["g.txt"] == [
        (c, "g.txt", "g.txt")]
    assert integrate._fork_path_dict(directory, c, base)["g.txt"] == [
        (base, "f.txt", "f.txt")]
    assert integrate._fork_path_dict(directory, a, c)["f.txt"] == [(base, "f.txt", "g.txt")]
    # A file the side holds at no path is not asked for, one at a path that cannot
    # be told is
    merged["branches"][0]["in_block"] = {"nowhere.txt": "", "lost.txt": None}
    assert integrate.in_block_read_list(merged) == [("a", "lost.txt", None),
                                                    ("c", "g.txt", "g.txt")]


def test_a_file_a_side_was_not_merged_into_is_left_until_it_is_read_on_that_side(
        tmp_path, monkeypatch):
    directory, base = _line_repo(tmp_path, _FIVE_A, _FIVE_B, {5: "C"})
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {"": base, **{
        entry_dict["branch"]: entry_dict["commit"] for entry_dict in merged["branches"]}}
    in_block_list = integrate.in_block_read_list(merged)
    reason = integrate._NO_BRANCH_READ_REASON.format(path="f.txt", branch="c",
                                                     side_path="f.txt")
    assert reason == ("- the merged tree is not known to hold the change of c to f.txt, "
                      "and nothing in this run has read the file as c has it. "
                      "read_branch_file reads it on c at f.txt")
    assert integrate._no_branch_read_reason_list(in_block_list) == [reason]
    _read_on(1, "f.txt")
    tool = next(t for t in integrate.merge_tool_list() if t.__name__ == "read_branch_file")
    # The file read on another side, at the start and at a path that is not there
    tool("b", "f.txt")
    tool("", "f.txt")
    assert tool("c", "g.txt").startswith("Error: ")
    assert integrate._unsettle_reason_list([], [], [], [], {}, "p", ["f.txt"], "", [], [],
                                           in_block_list) == [reason]
    assert integrate._unsettle_reason_list([], [], [], [], {}, "p", ["f.txt"], "", [],
                                           []) == []
    assert "line 5 C" in tool(branch="c", path="./f.txt")
    assert integrate._unsettle_reason_list([], [], [], [], {}, "p", ["f.txt"], "", [], [],
                                           in_block_list) == []
    integrate.reset_read()
    assert integrate._no_branch_read_reason_list(in_block_list) == [reason]


def test_the_work_list_names_the_files_a_side_was_not_merged_into(monkeypatch):
    merged = _merged(["f.txt"], ("a", "b", "c"))
    merged["branches"][2]["conflicted"] = []
    merged["branches"][2]["in_block"] = {"g.txt": "g.txt", "f.txt": "old/f.txt"}
    work = integrate.build_work_dict(merged, ["f.txt"], [], None, "", 0)
    assert [entry_dict["in_block"] for entry_dict in work["branches"]] == [
        [], [], ["f.txt", "g.txt"]]
    line_list = integrate.start_text(work).split("\n")
    assert line_list[line_list.index("  c"):][3] == (
        "    the merged tree is not known to hold its change to these files: f.txt, g.txt")
    assert sum("not known to hold" in line for line in line_list) == 1
    monkeypatch.setattr(config, "INTEGRATE_WORK_MAX_ITEM", 1)
    work = integrate.build_work_dict(merged, [], [], None, "", 0)
    assert work["branches"][2]["in_block"] == ["f.txt"] and work["complete"] is False


def test_a_merge_a_side_was_not_merged_into_a_file_of_is_not_settled_by_reading_the_file(
        tmp_path, monkeypatch):
    """The stand-in stage settles nothing and reads nothing; the file itself was read."""
    directory, base = _line_repo(tmp_path, _FIVE_A, _FIVE_B, {5: "C"})
    prepared = {"base": base, "merged": integrate.assemble(directory, base, ["a", "b", "c"]),
                "before": _analysis(), "against": _analysis(), "was_on": ""}
    _write(directory, "f.txt", _line_text({index: "A and B" for index in range(3, 8)}))
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(integrate, "ensure_knowledge",
                        lambda *a, **k: ("path", directory, "p", True))
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    monkeypatch.setattr(integrate, "make_llm", lambda *a, **k: None)
    monkeypatch.setattr(rlm_mod, "late_answer", lambda *a, **k: "f.txt was settled")
    refusal_list = []

    class _StageThatReads(_StageThatRunsOut):
        def run(self, **kwargs):
            _read_on(1, "f.txt")
            refusal_list.append(kwargs["on_complete"]("done"))
            return super().run(**kwargs)

    monkeypatch.setattr(integrate, "RLM", _StageThatReads)
    out = integrate.run_integration(prepared, "", "p", "out")
    reason = integrate._NO_BRANCH_READ_REASON.format(path="f.txt", branch="c",
                                                     side_path="f.txt")
    assert out["called"] is True and out["settled"] is False
    assert out["left"] == [reason] and reason in refusal_list[0]
    assert out["work"]["branches"][2]["in_block"] == ["f.txt"]


# == The separator lines of blocks that lie across one another ==


def _marker_line_list(text):
    """One file's marker lines as _marker_line_dict() returns them, from a text that
    writes each on a line of its own: "<" and ">" with the name after it, a "|" line
    with its label, and "=" for the seven of it."""
    name_dict = {"<": b"<<<<<<< refs/codetwine-loom/merged-so-far", "=": b"======="}
    line_list = []
    for word in text.split("\n"):
        if word[:1] == ">":
            line = b">>>>>>> refs/codetwine-loom/side/" + word[1:].encode()
        elif word[:1] == "|":
            line = b"||||||| " + word[1:].encode()
        else:
            line = name_dict[word]
        line_list.append(("f.txt", line))
    return line_list


@pytest.mark.parametrize("text, block_dict", [
    # One block, and a block inside the first section of another
    ("<\n|x\n=\n>b", {"||||||| x": 1, "=======": 1}),
    ("<\n<\n|x\n=\n>b\n|y\n=\n>c", {"||||||| x": 1, "||||||| y": 1, "=======": 2}),
    # The second block starts in the middle section of the first, and before the first
    ("<\n|x\n<\n=\n>b\n|x\n=\n>c", {"||||||| x": 2, "=======": 2}),
    ("<\n<\n|x\n|x\n=\n>c\n=\n>b", {"||||||| x": 2, "=======": 2}),
    ("<\n<\n|x\n=\n|y\n>b\n=\n>c", {"||||||| x": 1, "||||||| y": 1, "=======": 2}),
    # The block an earlier run wrote, under a name no side goes by now, inside a new one
    ("<\n<\n|x\n=\n>old\n|y\n=\n>c", {"||||||| y": 1, "=======": 1}),
    # What an earlier run left of a block, inside a new one: its "|" line, its end
    ("<\n|x\n|y\n=\n>c", {"||||||| x": 1, "||||||| y": 1, "=======": 1}),
    ("<\n>old\n|y\n=\n>c", {"||||||| y": 1, "=======": 1}),
    ("<\n|y\n>old\n=\n>c", {"||||||| y": 1, "=======": 1}),
    # A "|" line and an "=" line after the block's own "=" line are the file's
    ("<\n|x\n=\n|z\n=\n>b", {"||||||| x": 1, "=======": 1}),
    # A start nothing ends, before a block
    ("<\n|w\n<\n|x\n=\n>b", {"||||||| x": 1, "=======": 1}),
])
def test_the_separator_lines_of_blocks_lying_across_one_another_are_all_counted(
        text, block_dict):
    expect_dict = {"f.txt": {name.encode(): count for name, count in block_dict.items()}}
    assert integrate._block_separator_dict(_marker_line_list(text), ["b", "c"]) == \
        expect_dict


@pytest.mark.parametrize("text", ["<\n|x\n=\n>old", "<\n|x\n=", "|x\n=\n>old", "=\n|x"])
def test_a_file_holding_the_end_of_no_block_git_wrote_holds_no_block(text):
    assert integrate._block_separator_dict(_marker_line_list(text), ["b", "c"]) == {}


@pytest.mark.parametrize("c_line", [5, 3])
def test_a_side_changing_the_line_beside_a_block_starts_no_block_inside_it(
        tmp_path, c_line):
    """a and b change one line and c the line after it, or the one before."""
    directory, base = _line_repo(tmp_path, {4: "A"}, {4: "B"}, {c_line: "C"})
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    text = _read(directory, "f.txt")
    assert text.count("<<<<<<< refs/codetwine-loom/merged-so-far\n") == 1
    bar = next(line for line in text.split("\n") if line.startswith("|||||||"))
    assert text.count(bar + "\n") == 1 and f"line {c_line} C" not in text
    assert merged["branches"][2]["in_block"] == {"f.txt": "f.txt"}
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 0}}
    for left in (bar, "======="):
        _write(directory, "f.txt", _line_text({4: "A and B", c_line: "C"}).replace(
            "line 4 A and B\n", f"line 4 A and B\n{left}\n"))
        assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    _write(directory, "f.txt", _line_text({4: "A and B", c_line: "C"}))
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_side_merged_again_over_the_block_an_earlier_run_left_writes_no_block_in_its_file(
        tmp_path):
    """The union keeps the block of a and b; a moves on in the line of the block and is
    merged again."""
    directory, base = _line_repo(tmp_path, {4: "A"}, {4: "B"})
    merged = integrate.assemble(directory, base, ["a", "b"])
    union = _leave_union(directory, merged, "union", {})
    _git(directory, "checkout", "-q", "a")
    _write(directory, "f.txt", _line_text({4: "A again"}))
    _git(directory, "commit", "-qam", "a again")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["union", "a"], resume="union")
    text = _read(directory, "f.txt")
    assert text == _git(directory, "show", f"{union}:f.txt")
    bar_list = [line for line in text.split("\n") if line.startswith("|||||||")]
    assert len(bar_list) == 1 and text.count("=======\n") == 1
    assert merged["branches"][2]["branch"] == "a"
    assert merged["branches"][2]["in_block"] == {"f.txt": "f.txt"}
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 0}}
    for left in (*bar_list, "======="):
        _write(directory, "f.txt", _line_text({4: f"A again and B\n{left}"}))
        assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    _write(directory, "f.txt", _line_text({4: "A again and B"}))
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_separator_line_a_side_not_merged_into_a_file_added_to_it_may_stay(tmp_path):
    """a and b conflict in one line; c adds an underline of seven "=" to the file."""
    under = {11: "v0\nMore\n======="}
    directory, base = _line_repo(tmp_path, {4: "A"}, {4: "B"}, under)
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    assert merged["branches"][2]["in_block"] == {"f.txt": "f.txt"}
    assert "More" not in _read(directory, "f.txt")
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 1}}
    _write(directory, "f.txt", _line_text({4: "A and B", **under}))
    assert integrate.marker_path_list(directory, merged["marker"]) == []
    _write(directory, "f.txt", _line_text({4: "A and B\n=======", **under}))
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    # The union left with its block records the line, and the pass after it allows it
    _git(directory, "reset", "-q", "--hard")
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    _leave_union(directory, merged, "union", {})
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    assert merged["branches"][2]["in_block"] == {"f.txt": "f.txt"}
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 1}}
    _write(directory, "f.txt", _line_text({4: "A and B", **under}))
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_the_separator_lines_a_union_left_are_not_taken_off_a_file_put_back_as_another_holds_it(
        tmp_path):
    """Two unsettled unions each left a block in the one file, whose underline both
    may hold; the file is kept as the first holds it."""
    title = {0: "v0\n=======", 4: "A"}
    directory, base = _line_repo(tmp_path, title, {**title, 4: "B"}, {**title, 8: "C"},
                                 {**title, 8: "D"}, base_file_dict={"f.txt": _line_text(
                                     {0: "v0\n======="})})
    one = integrate.assemble(directory, base, ["a", "b"])
    assert one["marker"]["separator"] == {"f.txt": {b"=======": 1}}
    one_union = _leave_union(directory, one, "one", {})
    two = integrate.assemble(directory, base, ["c", "d"])
    _leave_union(directory, two, "two", {})
    merged = integrate.assemble(directory, base, ["one", "two"])
    assert _read(directory, "f.txt") == _git(directory, "show", f"{one_union}:f.txt")
    assert merged["marker"]["separator"] == {"f.txt": {b"=======": 1}}
    # a and b, whose versions the first union holds as a block, are held to it too
    assert [sorted(entry_dict["in_block"]) for entry_dict in merged["branches"]] == [
        ["f.txt"], ["f.txt"], ["f.txt"], ["f.txt"]]


# == Two sides that merged each other ====================


def _cross_repo(tmp_path):
    """A repository whose branches a and b each made one commit off the start and
    then merged the other's, and each made one more commit after that.

    Returns:
        (path, {name: commit} for "b0", "a1", "b1" and the tips "a" and "b").
    """
    directory = str(tmp_path / "cross")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    commit_dict = {}

    def commit(name, start, key):
        _git(directory, "checkout", "-q", "-B", name, *([start] if start else []))
        _write(directory, f"{key}.txt", key)
        _git(directory, "add", "-A")
        _git(directory, "commit", "-qm", key)
        commit_dict[key] = _git(directory, "rev-parse", "HEAD").strip()

    commit("main", "", "b0")
    commit("a", commit_dict["b0"], "a1")
    commit("b", commit_dict["b0"], "b1")
    for name, other in (("a", "b1"), ("b", "a1")):
        _git(directory, "checkout", "-q", name)
        _git(directory, "merge", "-q", "--no-ff", "--no-edit", commit_dict[other])
        commit(name, "", name)
    _git(directory, "checkout", "-q", "--detach", commit_dict["b0"])
    return directory, commit_dict


def test_every_commit_two_sides_both_reach_last_is_analysed_where_there_are_several(
        tmp_path, monkeypatch):
    directory, commit_dict = _cross_repo(tmp_path)
    fork_set = {commit_dict["a1"], commit_dict["b1"]}
    assert set(integrate._merge_base_list(directory, commit_dict["a"],
                                          commit_dict["b"])) == fork_set
    assert integrate._merge_base_list(directory, commit_dict["a1"], commit_dict["a"]) == [
        commit_dict["a1"]]
    define_by_commit_dict = {commit_dict["a1"]: {"one": ["x"], "both": ["x", "y"]},
                             commit_dict["b1"]: {"helper": ["h"], "both": ["x"]}}
    seen_list = []

    def analyse_head(project_dir, *rest):
        seen_list.append(_git(project_dir, "rev-parse", "HEAD").strip())
        return {"range": {}, "own": {seen_list[-1]: {}},
                "define": define_by_commit_dict[seen_list[-1]]}

    monkeypatch.setattr(integrate, "_analyse_head", analyse_head)
    side_dict = {name: commit_dict[name] for name in ("a", "b")}
    most_dict = {"one": ["x"], "both": ["x", "y"], "helper": ["h"]}
    by_commit_dict: dict = {}
    assert integrate._fork_define_dict(directory, "out", commit_dict["b0"], side_dict,
                                       {"at": ["start"]}, None, by_commit_dict) == {
        ("a", "b"): most_dict, ("b", "a"): most_dict}
    assert set(seen_list) == fork_set and len(seen_list) == 2
    assert _git(directory, "rev-parse", "HEAD").strip() == commit_dict["b0"]
    # A side read from a start it has merged and that has merged it
    start_dict = integrate._side_start_dict(directory, "out", commit_dict["a"],
                                            commit_dict["b"], None, by_commit_dict)
    first = integrate._merge_base_list(directory, commit_dict["a"], commit_dict["b"])[0]
    assert start_dict == {"commit": first, "range": {}, "own": {first: {}},
                          "define": most_dict}
    assert len(seen_list) == 2


def test_a_name_one_of_two_sides_that_merged_each_other_took_away_again_is_not_looked_for():
    """b added helper, a and b merged each other, and b took helper away again. One of
    the two commits both reach last holds helper and the other does not."""
    helper, more = "p/helper_py/helper.py", "p/more_py/more.py"
    start = _analysis(name={more: ["three"]})
    a_tip = {"three": [more], "four": [more], "helper": [helper]}
    b_tip = {"three": [more], "four": [more]}
    side_list = [("a", {more: {""}}), ("b", {helper: {""}})]
    most_dict = integrate._most_define_dict(
        [{"three": [more], "four": [more]}, {"three": [more], "helper": [helper]}])
    assert most_dict == {"three": [more], "four": [more], "helper": [helper]}
    fork_dict = {("a", "b"): most_dict, ("b", "a"): most_dict}
    assert integrate.keep_list_of(side_list, start, {"a": a_tip, "b": b_tip},
                                  fork_define_by_pair_dict=fork_dict) == [
        ("a", {"four": [more]}), ("b", {"four": [more]})]
    # Held against the one of them that does not hold it, the name is a's to keep
    one_dict = {"three": [more], "four": [more]}
    assert integrate.keep_list_of(
        side_list, start, {"a": a_tip, "b": b_tip},
        fork_define_by_pair_dict={("a", "b"): one_dict, ("b", "a"): one_dict})[0] == (
            "a", {"four": [more], "helper": [helper]})


# == A merge somebody made by hand over the commit of a stage ==


def _hand_merge(directory, base, verdict):
    """Leave a branch r whose tip is the commit of a refactoring stage with the verdict
    given, and a branch x that merges r with a message of its subject alone.

    Returns:
        (the merge commit, its first parent, r's tip).
    """
    _git(directory, "checkout", "-q", "-b", "r", base)
    _write(directory, "g.txt", "changed\n")
    _git(directory, "commit", "-qam",
         "sandbox: do it\n\ndo it\n\nCodetwine-Loom-Run-Request: do it\n"
         "Codetwine-Loom-Verdict: unverified")
    _write(directory, "g.txt", "tidied\n")
    _git(directory, "commit", "-qam",
         f"refactor: do it\n\ndo it\n\nCodetwine-Loom-Run-Request: do it\n"
         f"Codetwine-Loom-Verdict: {verdict}\nCodetwine-Loom-Stage: refactor")
    stage = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "-b", "x", base)
    _write(directory, "mine.txt", "mine\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "my own work")
    own = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "merge", "-q", "--no-ff", "--no-edit", "r")
    return _git(directory, "rev-parse", "HEAD").strip(), own, stage


@pytest.mark.parametrize("verdict", ["unsettled", "settled"])
def test_a_merge_made_by_hand_over_the_commit_of_a_stage_is_no_commit_of_a_stage(
        repo, verdict):
    directory, base = repo
    merge, own, stage = _hand_merge(directory, base, verdict)
    assert _git(directory, "log", "-1", "--format=%B", merge).strip() == "Merge branch 'r' into x"
    assert integrate.own_record_at(directory, merge) == {}
    assert integrate.own_record_at(directory, stage)["stage"] == "refactor"
    assert integrate.below_stage(directory, merge) == merge
    assert integrate.below_stage(directory, stage) == _git(
        directory, "rev-parse", f"{stage}^").strip()
    assert integrate.union_side_list(directory, merge, is_union=False) == [merge]
    # The branch is carried on from the merge, which stays on it
    assert sandbox.resume_head(directory, "x") == (merge, own)


def test_a_stage_is_read_off_a_commit_whose_subject_is_that_stage_s(repo):
    directory, base = repo
    for subject, trailer, stage in (
            ("refactor: x", "refactor", "refactor"),
            ("test-create: x", "test-create", "test-create"),
            ("test-refactor: x", "test-refactor", "test-refactor"),
            ("refactor x", "refactor", ""), ("Merge branch 'r'", "refactor", ""),
            ("sandbox: x", "refactor", ""), ("refactor: x", "test-create", "")):
        _git(directory, "commit", "-q", "--allow-empty", "-m",
             f"{subject}\n\nask\n\nCodetwine-Loom-Stage: {trailer}\n"
             "Codetwine-Loom-Verdict: settled")
        record_dict = integrate.own_record_at(directory, "HEAD")
        assert (record_dict["stage"], record_dict["verdict"], record_dict["asked"]) == (
            stage, "settled", "ask"), subject
    # A commit holding nothing of this tool's and a name that is no commit
    _git(directory, "commit", "-q", "--allow-empty", "-m", "plain")
    assert integrate.own_record_at(directory, "HEAD") == {}
    assert integrate.own_record_at(directory, "no-such-commit") == {}
    assert integrate.own_record_at(directory, base) == {}


# == The sides a union names as not merged into a file, the path a side holds a file
# == at, and the length of a block's markers


def test_the_commit_a_union_was_written_onto_is_held_to_its_file_on_every_run_after(
        tmp_path, monkeypatch):
    """The union of a and b is carried on from with the tree it is written onto moved
    on, and the union that run leaves is carried on from twice more."""
    directory, base = _line_repo(tmp_path, _FIVE_A, _FIVE_B, {5: "C"})
    a, b, c = (_git(directory, "rev-parse", name).strip() for name in ("a", "b", "c"))
    merged = integrate.assemble(directory, base, ["a", "b"])
    union_text = _read(directory, "f.txt")
    assert integrate.no_merge_record_list(merged) == []
    _leave_union(directory, merged, "union", {})
    merged = integrate.assemble(directory, base, ["union"], onto="c", resume="union")
    # a and b, whose versions the union holds as a block, are held to the file too
    assert sorted(integrate.no_merge_record_list(merged)) == sorted(
        [side, "f.txt", "f.txt"] for side in (a, b, c))
    earlier = "union"
    for branch in ("union2", "union3", "union4"):
        _leave_union(directory, merged, branch, {})
        _git(directory, "checkout", "-q", "--detach", c)
        assert sorted(integrate._no_merge_record_at(directory, _git(
            directory, "rev-parse", branch).strip())) == sorted(
            (side, "f.txt", "f.txt") for side in (a, b, c))
        merged = integrate.assemble(directory, c, [branch], onto="c", resume=branch)
        assert [(entry_dict["branch"], entry_dict["commit"], entry_dict["in_block"],
                 bool(entry_dict.get("carry"))) for entry_dict in merged["branches"]] == [
            (integrate.side_name(a), a, {"f.txt": "f.txt"}, False),
            (integrate.side_name(b), b, {"f.txt": "f.txt"}, False),
            (integrate.side_name(c), c, {"f.txt": "f.txt"}, True)], earlier
        assert _read(directory, "f.txt") == union_text
        assert integrate.in_block_read_list(merged) == [
            (integrate.side_name(side), "f.txt", "f.txt") for side in (a, b, c)]
        earlier = branch
    # The commit carried is read under its name, and is no side whose definitions the
    # union is held to
    prepared = {"base": c, "merged": merged, "before": _analysis(), "against": _analysis(),
                "was_on": ""}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    monkeypatch.setattr(config, "CHECK_COMMAND", [])
    monkeypatch.setattr(integrate, "analyse", lambda _: _analysis())
    monkeypatch.setattr(edit, "test_can_run", lambda: False)
    measure = integrate._measure_union(prepared, "", "p", directory)
    assert integrate.side_name(c) not in [branch for branch, _ in measure["side"]]
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {"": c, **{
        entry_dict["branch"]: entry_dict["commit"] for entry_dict in merged["branches"]}}
    tool = next(t for t in integrate.merge_tool_list() if t.__name__ == "read_branch_file")
    in_block_list = integrate.in_block_read_list(merged)
    assert len(integrate._no_branch_read_reason_list(in_block_list)) == 3
    assert "line 5 C" in tool(integrate.side_name(c), "f.txt")
    assert len(integrate._no_branch_read_reason_list(in_block_list)) == 2
    for side in (a, b):
        tool(integrate.side_name(side), "f.txt")
    assert integrate._no_branch_read_reason_list(in_block_list) == []


def test_a_branch_passed_before_a_union_is_held_to_its_file_on_every_run_after(tmp_path):
    directory, base = _line_repo(tmp_path, _FIVE_A, _FIVE_B, {5: "C"})
    a, b, c = (_git(directory, "rev-parse", name).strip() for name in ("a", "b", "c"))
    merged = integrate.assemble(directory, base, ["a", "b"])
    _leave_union(directory, merged, "union", {})
    merged = integrate.assemble(directory, base, ["c", "union"])
    # a and b, whose versions the union holds as a block, are held to the file too
    assert integrate.no_merge_record_list(merged) == [
        [side, "f.txt", "f.txt"] for side in (c, a, b)]
    for branch in ("union2", "union3"):
        _leave_union(directory, merged, branch, {})
        merged = integrate.assemble(directory, base, [branch], resume=branch)
        assert [(entry_dict["branch"], entry_dict["in_block"], bool(entry_dict.get("carry")))
                for entry_dict in merged["branches"]] == [
            (integrate.side_name(side), {"f.txt": "f.txt"}, False) for side in (c, a, b)]


def test_the_sides_a_union_was_not_merged_for_are_read_back_as_they_were_recorded(repo):
    directory, base = repo
    merged = _merged(["f.txt"], ("a", "b"))
    merged["branches"][0]["in_block"] = {'dir with space/q "x".rst': "old/q.rst",
                                         "g.txt": None}
    merged["branches"][1]["in_block"] = {"caf\\xe9.txt": ""}
    record_list = integrate.no_merge_record_list(merged)
    assert record_list == [["2" * 40, "old/q.rst", 'dir with space/q "x".rst'],
                           ["2" * 40, None, "g.txt"], ["2" * 40, "", "caf\\xe9.txt"]]
    assert integrate.is_no_merge_record(record_list)
    text = integrate.no_merge_text(record_list)
    assert not re.search(r"\s", text) and text.isascii()
    commit = _git(directory, "commit-tree", f"{base}^{{tree}}", "-p", base, "-m",
                  "integrate: x\n\nCodetwine-Loom-Verdict: unsettled\n"
                  f"Codetwine-Loom-No-Merge: {text}").strip()
    assert integrate._no_merge_record_at(directory, commit) == [
        tuple(entry) for entry in record_list]
    assert integrate._no_merge_record_at(directory, base) is None
    assert integrate.no_merge_text(None) == "" and integrate.no_merge_text([]) == "[]"
    assert integrate.no_merge_record_list({}) == []


def test_a_record_of_sides_too_long_for_a_commit_is_not_written(caplog):
    record_list = [["a" * 40, f"src/file_{n}.py", f"src/file_{n}.py"] for n in range(2000)]
    assert integrate.is_no_merge_record(record_list)
    assert integrate.no_merge_text(record_list) == ""
    assert "none is recorded" in caplog.text


@pytest.mark.parametrize("value", [
    None, {}, "x", [["a" * 40, "f", "g"], "x"], [["a" * 40, "f"]], [["main", "f", "g"]],
    [["A" * 40, "f", "g"]], [["a" * 40 + "\n", "f", "g"]], [["a" * 40, 1, "g"]],
    [["a" * 40, "f", ""]], [["a" * 40, "f", None]], [["a" * 40, "a\0b", "g"]],
    [["a" * 40, "f", "p" * 5000]], [[7, "f", "g"]]])
def test_a_record_of_another_shape_is_no_record_of_sides_not_merged(value):
    assert not integrate.is_no_merge_record(value)


@pytest.mark.parametrize("trailer", [
    "", "Codetwine-Loom-No-Merge: not json", "Codetwine-Loom-No-Merge: {}",
    'Codetwine-Loom-No-Merge: [["main","f.txt","f.txt"]]',
    pytest.param("Codetwine-Loom-No-Merge: [" + ('["' + "a" * 40 + '","f","g"],') * 2000
                 + '["' + "a" * 40 + '","f","g"]]', id="too long")])
def test_a_union_naming_no_side_holds_each_commit_it_was_made_from_to_what_it_lacks_of_it(
        repo, trailer):
    """The commit carries no record of the sides, or one that cannot be read. The
    union holds a's files with a block in f.txt: a_only.txt is a's change as the
    union holds it, and b_only.txt, which b added, is not in the union."""
    directory, base = repo
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    _union_branch_with(directory, base, {"f.txt": _BLOCK_TEXT}, record_dict=None)
    message = ("integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: unsettled\n"
               'Codetwine-Loom-Separator: {"f.txt":{}}\n' + trailer)
    commit = subprocess.run(
        ["git", "-C", directory, "commit-tree", "union^{tree}", "-p", base, "-p", a,
         "-p", b, "-F", "-"], input=message, capture_output=True, text=True,
        check=True).stdout.strip()
    _git(directory, "branch", "bare", commit)
    assert integrate._no_merge_record_at(directory, commit) is None
    merged = integrate.assemble(directory, base, ["bare"], resume="bare")
    assert [(entry_dict["commit"], entry_dict["in_block"], bool(entry_dict.get("carry")))
            for entry_dict in merged["branches"]] == [
        (a, {"f.txt": "f.txt"}, False),
        (b, {"b_only.txt": "b_only.txt", "f.txt": "f.txt"}, False)]
    # A union holding the change of every commit it was made from names no side
    _git(directory, "reset", "-q", "--hard")
    _branch_c(directory, base, {"g.txt": "kept by c\n"})
    c = _git(directory, "rev-parse", "c").strip()
    tree = _git(directory, "rev-parse",
                f"{integrate.assemble(directory, base, ['a', 'c'])['commit']}^{{tree}}").strip()
    _git(directory, "reset", "-q", "--hard")
    whole = _git(directory, "commit-tree", tree, "-p", base, "-p", a, "-p", c, "-m",
                 "integrate: x\n\ncombine\n\nCodetwine-Loom-Exit-Status: 137").strip()
    assert integrate._open_no_merge_list(directory, whole, True) == []


def test_a_side_kept_out_of_a_file_the_model_settled_is_held_by_a_union_naming_none(
        tmp_path, monkeypatch):
    """a and b conflict in f.txt and c's change to it is kept out; the run's model
    takes the markers out and the container ends before it says how it ended, so the
    union's commit carries no record of the sides. c is held to f.txt all the same,
    and the run carrying on from the union settles once c's version is read."""
    directory, base = _line_repo(tmp_path, {3: "A"}, {3: "B"}, {9: "C"})
    c = _git(directory, "rev-parse", "c").strip()
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    assert integrate.no_merge_record_list(merged) == [[c, "f.txt", "f.txt"]]
    _write(directory, "f.txt", _line_text({3: "A and B"}))
    _git(directory, "add", "-A")
    tree = _git(directory, "write-tree").strip()
    _git(directory, "reset", "-q", "--hard")
    union = _git(directory, "commit-tree", tree, "-p", base, "-p", "a", "-p", "b", "-p", "c",
                 "-m", "integrate: x\n\ncombine\n\nCodetwine-Loom-Exit-Status: 137").strip()
    _git(directory, "branch", "union", union)
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    in_block_dict = {entry_dict["commit"]: entry_dict["in_block"]
                     for entry_dict in merged["branches"]}
    assert in_block_dict[c] == {"f.txt": "f.txt"}
    assert "line 9 C" not in _read(directory, "f.txt")
    assert (integrate.side_name(c), "f.txt", "f.txt") in integrate.in_block_read_list(merged)
    assert integrate.marker_path_list(directory, merged["marker"]) == []
    integrate.reset_stage()
    integrate._STAGE_DICT["commit_dict"] = {"": base, **{
        entry_dict["branch"]: entry_dict["commit"] for entry_dict in merged["branches"]}}
    monkeypatch.setattr(config, "PROJECT_DIR", directory)
    tool = next(t for t in integrate.merge_tool_list() if t.__name__ == "read_branch_file")
    in_block_list = integrate.in_block_read_list(merged)
    assert any(integrate.side_name(c) in reason
               for reason in integrate._no_branch_read_reason_list(in_block_list))
    for side, _, side_path in in_block_list:
        assert not tool(side, side_path).startswith("Error")
    assert integrate._no_branch_read_reason_list(in_block_list) == []


@pytest.mark.parametrize("message, is_open", [
    ("integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: settled", False),
    ("integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: settled\nCodetwine-Loom-Exit-Status: 0",
     False),
    ("integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: settled\nCodetwine-Loom-Exit-Status: 1",
     True),
    ("integrate: x\n\ncombine\n\nCodetwine-Loom-Verdict: unsettled\nCodetwine-Loom-Exit-Status: 0",
     True),
    ("integrate: x\n\ncombine\n\nCodetwine-Loom-Exit-Status: 1", True),
    ("Merge branch 'a'\n\ncombine\n\nCodetwine-Loom-Verdict: settled", True),
    ("Merge branch 'a'\n\ncombine", False)])
def test_a_merge_is_read_as_its_sides_unless_a_merging_run_wrote_and_settled_it(
        repo, message, is_open):
    directory, base = repo
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    merge = _git(directory, "commit-tree", f"{a}^{{tree}}", "-p", base, "-p", a, "-p", b,
                 "-m", message).strip()
    is_settle = not is_open and message.startswith("integrate: ")
    assert integrate.is_settle_merge(directory, merge) is is_settle
    assert integrate.union_side_list(directory, merge, is_union=False) == (
        [a, b] if is_open else [merge])
    # The branch a run is carried on from is read as its sides whatever it records,
    # and each union opened says what it was written against
    assert integrate._open_union(directory, merge) == (
        ([merge], []) if is_settle else ([a, b], [base]))
    assert integrate.is_settle_merge(directory, a) is False


def _move_file_on(directory, branch, base, old_name, new_name, text=None):
    """Commit a move of one file on a branch off base, its contents rewritten where
    text is given, and come back to base."""
    _git(directory, "checkout", "-q", "-B", branch, base)
    _git(directory, "mv", old_name, new_name)
    if text is not None:
        _write(directory, new_name, text)
    _git(directory, "commit", "-qam", f"{branch} moves {old_name}")
    _git(directory, "checkout", "-q", "--detach", base)


def test_a_file_is_followed_from_a_side_to_another_through_the_commit_they_parted_at(
        tmp_path):
    directory, base = _line_repo(tmp_path, {4: "A"}, {4: "B"},
                                 base_file_dict={"k.txt": "kept\n"})
    a, b = (_git(directory, "rev-parse", name).strip() for name in ("a", "b"))
    _move_file_on(directory, "c", base, "f.txt", "g.txt", _line_text({9: "C"}))
    _git(directory, "checkout", "-q", "-b", "d", base)
    _git(directory, "rm", "-q", "k.txt")
    _write(directory, "new.txt", "d's\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "d")
    _git(directory, "checkout", "-q", "--orphan", "alone")
    _git(directory, "commit", "-qm", "no commit shared")
    _git(directory, "checkout", "-q", "--detach", base)
    c, d, alone = (_git(directory, "rev-parse", name).strip() for name in ("c", "d", "alone"))
    assert integrate._fork_path_dict(directory, a, c) == {
        "f.txt": [(base, "f.txt", "g.txt")], "k.txt": [(base, "k.txt", "k.txt")]}
    assert integrate._fork_path_dict(directory, c, a)["g.txt"] == [(base, "f.txt", "f.txt")]
    assert integrate._fork_path_dict(directory, c, base)["g.txt"] == [(base, "f.txt", "f.txt")]
    # A file one of the two took away, one it added, and two commits that share none
    assert integrate._fork_path_dict(directory, a, d)["k.txt"] == [(base, "k.txt", "")]
    assert integrate._fork_path_dict(directory, d, a) == {
        "f.txt": [(base, "f.txt", "f.txt")], "new.txt": [(base, "", "")]}
    assert integrate._fork_path_dict(directory, d, alone)["new.txt"] == [("", "", "new.txt")]
    assert integrate._fork_path_dict(directory, alone, a)["new.txt"] == [("", "", "")]
    assert integrate._one_path([(base, "f.txt", "g.txt")]) == "g.txt"
    assert integrate._one_path([(base, "f.txt", "")]) == ""
    assert integrate._one_path([(base, "f.txt", "g.txt"), (a, "f.txt", "g.txt")]) == "g.txt"
    assert integrate._one_path([(base, "f.txt", "g.txt"), (a, "f.txt", "")]) is None
    assert integrate._one_path([]) is None


def _value_text(change_dict=None):
    """Forty numbered lines, those of change_dict written as it gives them."""
    change_dict = change_dict or {}
    return "".join(change_dict.get(index, f"value_{index} = {index}") + "\n"
                   for index in range(40))


def test_a_side_that_rewrote_most_of_a_file_another_side_moved_is_held_to_it(tmp_path):
    """a and b conflict in the file, d moves it, and c rewrites the lines after the
    block at the path the file had."""
    directory, base = _two_file_repo(
        tmp_path, "f.txt", _value_text({3: "value_3 = 'a'"}), "f.txt",
        _value_text({3: "value_3 = 'b'"}), {"f.txt": _value_text()})
    _move_file_on(directory, "d", base, "f.txt", "g.txt")
    _branch_c(directory, base, {"f.txt": _value_text(
        {index: f"setting_{index} = compute({index})" for index in range(10, 40)})})
    merged = integrate.assemble(directory, base, ["a", "b", "d", "c"])
    assert [entry_dict["in_block"] for entry_dict in merged["branches"]] == [
        {}, {}, {}, {"g.txt": "f.txt"}]
    assert "setting_" not in _read(directory, "g.txt")
    assert "<<<<<<< refs/codetwine-loom/merged-so-far" in _read(directory, "g.txt")
    assert not os.path.exists(os.path.join(directory, "f.txt"))
    assert integrate.in_block_read_list(merged) == [("c", "g.txt", "f.txt")]
    assert integrate.no_merge_record_list(merged) == [
        [_git(directory, "rev-parse", "c").strip(), "f.txt", "g.txt"]]


@pytest.mark.parametrize("fork_tuple", [("0" * 40, "f.txt", "elsewhere.txt"), None])
def test_a_side_whose_path_of_a_file_cannot_be_told_leaves_the_merge_unsettled(
        tmp_path, monkeypatch, fork_tuple):
    """The commits c and what was merged before it parted at give two paths for the
    file in c, or none at all."""
    directory, base = _line_repo(tmp_path, _FIVE_A, _FIVE_B, {5: "C"})
    fork_path_dict = integrate._fork_path_dict

    def other_fork(project_dir, commit, other_commit):
        return {path: [*fork_list, fork_tuple] if fork_tuple
                else [(fork, fork_path, "") for fork, fork_path, _ in fork_list]
                for path, fork_list
                in fork_path_dict(project_dir, commit, other_commit).items()}

    monkeypatch.setattr(integrate, "_fork_path_dict", other_fork)
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    c = _git(directory, "rev-parse", "c").strip()
    assert merged["branches"][2]["in_block"] == {"f.txt": None}
    assert "line 5 C" not in _read(directory, "f.txt")
    assert integrate.no_merge_record_list(merged) == [[c, None, "f.txt"]]
    work = integrate.build_work_dict(merged, ["f.txt"], [], None, "", 0)
    assert work["branches"][2]["in_block"] == ["f.txt"]
    in_block_list = integrate.in_block_read_list(merged)
    assert in_block_list == [("c", "f.txt", None)]
    reason = ("- the merged tree is not known to hold the change of c to f.txt, and the "
              "path c holds that file at could not be told, so the merge is left "
              "unsettled. Look for the file on c with read_branch_file and say what you "
              "made of it")
    integrate.reset_stage()
    integrate._STAGE_DICT["branch_read_set"].add(("c", "f.txt"))
    assert integrate._no_branch_read_reason_list(in_block_list) == [reason]
    assert integrate._unsettle_reason_list([], [], [], [], {}, "p", [], "", [], [],
                                           in_block_list) == [reason]


def test_the_sides_of_a_commit_a_file_was_not_kept_as_are_told_by_what_each_changed(
        repo, caplog):
    """f.txt is kept as the base holds it against a, which changed it, d, which took
    it away, and the base itself."""
    directory, base = repo
    a = _git(directory, "rev-parse", "a").strip()
    _git(directory, "checkout", "-q", "-b", "d", base)
    _git(directory, "rm", "-q", "f.txt")
    _git(directory, "commit", "-qm", "d drops f.txt")
    d = _git(directory, "rev-parse", "HEAD").strip()
    _git(directory, "checkout", "-q", "--detach", base)
    entry_list = [{"commit": commit, "in_block": {}} for commit in (a, d, base)]
    assert integrate._add_no_merge(directory, entry_list, base, {"f.txt": "f.txt"}) == [
        (a, "f.txt", "f.txt", base), (d, "", "f.txt", base)]
    assert [entry_dict["in_block"] for entry_dict in entry_list] == [
        {"f.txt": "f.txt"}, {"f.txt": ""}, {}]
    # No side holds the file otherwise: every one of them is held to it at a path
    # that cannot be told
    entry_list = [{"commit": base, "in_block": {}}]
    assert integrate._add_no_merge(directory, entry_list, base, {"g.txt": "g.txt"}) == [
        (base, None, "g.txt", base)]
    assert entry_list[0]["in_block"] == {"g.txt": None}
    assert "every one of them is held to it" in caplog.text
    assert integrate._add_no_merge(directory, [], base, {"g.txt": "g.txt"}) == []
    assert integrate._add_no_merge(directory, entry_list, base, {}) == []


def test_a_block_file_a_branch_moved_is_found_where_another_changes_it_at_its_old_path(
        tmp_path):
    """The union left the block of a and b, which takes most of the file. d moves the
    file and c changes a line of it at the path it had."""
    directory, base = _line_repo(tmp_path, {index: "A" for index in range(9)},
                                 {index: "B" for index in range(9)})
    merged = integrate.assemble(directory, base, ["a", "b"])
    union_text = _read(directory, "f.txt")
    _leave_union(directory, merged, "union", {})
    _move_file_on(directory, "d", base, "f.txt", "g.txt")
    _branch_c(directory, base, {"f.txt": _line_text({10: "C"})})
    merged = integrate.assemble(directory, base, ["union", "d", "c"], resume="union")
    assert _read(directory, "g.txt") == union_text
    assert _read(directory, "f.txt") == _line_text({10: "C"})
    assert merged["branches"][3]["conflicted"] == ["f.txt"] and merged["unmarked"] == ["f.txt"]
    assert merged["marker"]["separator"] == {"g.txt": {b"=======": 0}}
    marker_list = [line for line in union_text.splitlines()
                   if line[:8] in ("<<<<<<< ", "||||||| ", ">>>>>>> ") or line == "======="]
    assert len(marker_list) == 4
    for keep in [None, *marker_list]:
        _write(directory, "g.txt", "".join(
            line for line in union_text.splitlines(True)
            if line.rstrip("\n") not in marker_list or line.rstrip("\n") == keep))
        assert integrate.marker_path_list(directory, merged["marker"]) == (
            [] if keep is None else ["g.txt"]), keep


def test_a_marker_like_line_a_side_holds_in_a_file_it_moved_is_counted_from_where_it_was(
        tmp_path):
    """The file a and b conflict in holds an underline. c moves the file and changes
    a line away from the block."""
    head = "Guide\n=======\n\n"
    directory, base = _two_file_repo(
        tmp_path, "f.txt", head + _line_text({4: "A"}), "f.txt", head + _line_text({4: "B"}),
        {"f.txt": head + _line_text()})
    _move_file_on(directory, "c", base, "f.txt", "g.txt", head + _line_text({9: "C"}))
    merged = integrate.assemble(directory, base, ["a", "b", "c"])
    assert merged["branches"][2]["in_block"] == {"g.txt": "g.txt"}
    assert merged["marker"]["separator"] == {"g.txt": {b"=======": 1}}
    text = _read(directory, "g.txt")
    assert text.count("=======\n") == 2
    _write(directory, "g.txt", "".join(
        line for line in text.splitlines(True) if line[:8] not in ("<<<<<<< ", ">>>>>>> ")))
    assert integrate.marker_path_list(directory, merged["marker"]) == ["g.txt"]


def _two_name_repo(tmp_path, head=""):
    """A repository whose branches a and b each move f.txt to a name of their own and
    rewrite one line of it."""
    directory, base = _two_file_repo(tmp_path, "a.txt", "a\n", "b.txt", "b\n",
                                     {"f.txt": head + _line_text()})
    for branch, word in (("a", "A"), ("b", "B")):
        _move_file_on(directory, branch, branch, "f.txt", f"g_{branch}.txt",
                       head + _line_text({4: word}))
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


@pytest.mark.parametrize("head, allow_dict", [
    ("", {b"========": 0}), ("Guide\n=======\n\n", {b"=======": 1, b"========": 0})])
def test_the_separator_lines_of_a_block_are_read_at_the_length_of_its_markers(
        tmp_path, head, allow_dict):
    """git writes the block of a file two sides moved to two names with markers one
    longer than the file's."""
    directory, base = _two_name_repo(tmp_path, head)
    merged = integrate.assemble(directory, base, ["a", "b"])
    text = _read(directory, "g_a.txt")
    assert _read(directory, "g_b.txt") == text
    marker_list = [line for line in text.splitlines()
                   if line[:9] in ("<<<<<<<< ", "|||||||| ", ">>>>>>>> ") or line == "========"]
    assert [line[:8] for line in marker_list] == ["<<<<<<<<", "||||||||", "========",
                                                  ">>>>>>>>"]
    assert merged["marker"]["separator"] == {"g_a.txt": allow_dict, "g_b.txt": allow_dict}
    assert integrate.separator_record_dict(merged)["g_a.txt"] == {
        line.decode(): count for line, count in allow_dict.items()}
    assert integrate.marker_path_list(directory, merged["marker"]) == ["g_a.txt", "g_b.txt"]
    for keep in [None, *marker_list]:
        for name in ("g_a.txt", "g_b.txt"):
            _write(directory, name, "".join(
                line for line in text.splitlines(True)
                if line.rstrip("\n") not in marker_list or line.rstrip("\n") == keep))
        assert integrate.marker_path_list(directory, merged["marker"]) == (
            [] if keep is None else ["g_a.txt", "g_b.txt"]), keep


def test_a_separator_line_longer_than_the_file_s_markers_is_found_by_the_run_after(tmp_path):
    """The union settles one of the two files and leaves the "=" line of the other's
    block, its name lines taken away."""
    directory, base = _two_name_repo(tmp_path)
    merged = integrate.assemble(directory, base, ["a", "b"])
    left_text = _line_text({4: "A"}).replace("line 4 A\n", "line 4 A\n========\nline 4 B\n")
    _leave_union(directory, merged, "union", {"g_a.txt": left_text,
                                              "g_b.txt": _line_text({4: "A and B"})})
    merged = integrate.assemble(directory, base, ["union"], resume="union")
    assert _read(directory, "g_a.txt") == left_text
    assert merged["marker"]["separator"] == {"g_a.txt": {b"========": 0}, "g_b.txt": {}}
    assert integrate.marker_path_list(directory, merged["marker"]) == ["g_a.txt"]
    _write(directory, "g_a.txt", _line_text({4: "A and B"}))
    assert integrate.marker_path_list(directory, merged["marker"]) == []


@pytest.mark.parametrize("before, after", [(0, 9), (9, 0), (8, 9)])
def test_the_separator_line_of_a_block_is_read_at_its_length_after_the_markers_change(
        tmp_path, before, after):
    """The union leaves the "=" line of a block, its other marker lines taken away, and
    the tree the run after writes onto sets the file's conflict-marker-size otherwise
    (0 sets none)."""
    def attribute_dict(size):
        return {".gitattributes": f"f.txt conflict-marker-size={size}\n"} if size else {}

    directory, base = _line_repo(tmp_path, {4: "A"}, {4: "B"},
                                 base_file_dict=attribute_dict(before))
    merged = integrate.assemble(directory, base, ["a", "b"])
    size = before or 7
    assert merged["marker"]["separator"] == {"f.txt": {b"=" * size: 0}}
    assert integrate.separator_record_dict(merged) == {"f.txt": {"=" * size: 0}}
    left_text = _line_text({4: "A"}).replace("line 4 A\n", f"line 4 A\n{'=' * size}\nline 4 B\n")
    _leave_union(directory, merged, "union", {"f.txt": left_text})
    _git(directory, "checkout", "-q", "-b", "tree", base)
    if after:
        _write(directory, ".gitattributes", attribute_dict(after)[".gitattributes"])
    else:
        _git(directory, "rm", "-q", ".gitattributes")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "tree")
    tree = _git(directory, "rev-parse", "tree").strip()
    _git(directory, "checkout", "-q", "--detach", tree)
    merged = integrate.assemble(directory, base, ["union"], onto="tree", onto_commit=tree,
                                resume="union")
    assert merged["marker"]["size"] == ({"f.txt": after} if after else {})
    assert _read(directory, "f.txt") == left_text
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    _write(directory, "f.txt", _line_text({4: "A and B"}))
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_block_inside_another_is_counted_at_its_own_length(tmp_path):
    """a moves the file to a name b adds a file under, and both rewrite one line of
    it: git writes a block of longer markers inside the block of the two files."""
    directory, base = _two_file_repo(tmp_path, "a.txt", "a\n", "f.txt", _line_text({4: "B"}),
                                     {"f.txt": _line_text()})
    _move_file_on(directory, "a", "a", "f.txt", "g.txt", _line_text({4: "A"}))
    _git(directory, "checkout", "-q", "b")
    _write(directory, "g.txt", "b's own file\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "b adds g.txt")
    _git(directory, "checkout", "-q", "--detach", base)
    merged = integrate.assemble(directory, base, ["a", "b"])
    text = _read(directory, "g.txt")
    marker_list = [line for line in text.splitlines()
                   if line[:1] in "<|=>" and len(line) - len(line.lstrip(line[:1])) in (7, 8)]
    assert sorted({len(line) - len(line.lstrip(line[:1])) for line in marker_list}) == [7, 8]
    assert merged["marker"]["separator"] == {"g.txt": {b"=======": 0, b"========": 0}}
    for keep in [None, *marker_list]:
        _write(directory, "g.txt", "".join(
            line for line in text.splitlines(True)
            if line.rstrip("\n") not in marker_list or line.rstrip("\n") == keep))
        assert integrate.marker_path_list(directory, merged["marker"]) == (
            [] if keep is None else ["g.txt"]), keep


def test_a_separator_line_is_a_block_s_own_only_at_the_length_of_its_markers():
    line_list = [("g.txt", line) for line in (
        b"=======", b"<<<<<<< refs/codetwine-loom/merged-so-far",
        b"<<<<<<<< refs/codetwine-loom/merged-so-far:g.txt", b"|||||||| 1234567:f.txt", b"=======",
        b"========", b">>>>>>>> refs/codetwine-loom/side/b:f.txt", b"||||||| 1234567", b"=======",
        b">>>>>>> refs/codetwine-loom/side/b")]
    # The "=" line of another length between the inner block's "|" and "=" lines is
    # that block's own too
    assert integrate._block_separator_dict(line_list, ["b"]) == {"g.txt": {
        b"|||||||| 1234567:f.txt": 1, b"========": 1, b"||||||| 1234567": 1, b"=======": 2}}
    assert integrate._separator_frame_dict(line_list, ["b"]) == {"g.txt": {b"=======": 1}}
    # A block with no separator line as long as its markers takes none of another length
    line_list = [("g.txt", line) for line in (
        b"<<<<<<<< refs/codetwine-loom/merged-so-far:g.txt", b"=======", b"||||||| 1234567",
        b">>>>>>>> refs/codetwine-loom/side/b:f.txt")]
    assert integrate._block_separator_dict(line_list, ["b"]) == {"g.txt": {}}
    assert integrate._line_size_dict(line_list) == {"g.txt": {7, 8}}
    assert integrate._line_size_dict(
        [("g.txt", b"  <<<<<<<<< refs/codetwine-loom/merged-so-far")]) == {}


def test_a_separator_line_is_read_at_the_lengths_its_file_s_markers_have(repo):
    directory, base = repo
    _git(directory, "checkout", "-q", "-b", "m", base)
    _write(directory, "eight.txt", "<<<<<<<< refs/codetwine-loom/merged-so-far:x\n"
                                   "========\n=======\n==========\n|||||||| x\n")
    _write(directory, "front.txt", "  <<<<<<<< refs/codetwine-loom/merged-so-far\n========\n")
    _write(directory, "plain.txt", "========\n=======\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "m")
    commit = _git(directory, "rev-parse", "HEAD").strip()
    name_line = b"<<<<<<<< refs/codetwine-loom/merged-so-far"
    assert integrate._marker_line_dict(directory, [commit])[commit] == [
        ("eight.txt", name_line + b":x"), ("eight.txt", b"========"),
        ("eight.txt", b"======="), ("eight.txt", b"|||||||| x"),
        ("front.txt", b"  " + name_line), ("plain.txt", b"=======")]
    assert integrate._marker_line_dict(
        directory, [commit], more_size_dict={"plain.txt": {8}, "front.txt": {10}}
    )[commit][-3:] == [("front.txt", b"  " + name_line), ("plain.txt", b"========"),
                       ("plain.txt", b"=======")]
    assert integrate._run_size(b"") == 0 and integrate._run_size(b"<<< x") == 3


def _two_fork_repo(tmp_path, head=""):
    """A repository whose branches x and y each merged the other's first commit, settled
    the one line the two rewrote in a way of their own, and rewrote the line after it:
    the two have two merge bases, which conflict with each other."""
    directory, base = _two_file_repo(
        tmp_path, "f.txt", head + _line_text({4: "A"}), "f.txt", head + _line_text({4: "B"}),
        {"f.txt": head + _line_text()})
    for name, own, other in (("x", "a", "b"), ("y", "b", "a")):
        _git(directory, "checkout", "-q", "-b", name, own)
        subprocess.run(["git", "-C", directory, "merge", "-q", "--no-commit", other],
                       capture_output=True, check=False)
        _write(directory, "f.txt", head + _line_text({4: f"{own} then {other}"}))
        _git(directory, "commit", "-qam", f"{name} merges {other}")
        _write(directory, "f.txt", head + _line_text({4: f"{own} then {other}", 5: name}))
        _git(directory, "commit", "-qam", f"{name} rewrites the next line")
    _git(directory, "checkout", "-q", "--detach", base)
    return directory, base


@pytest.mark.parametrize("head, allow_dict", [
    ("", {b"=======": 0, b"=========": 0}),
    ("Guide\n=========\n\n", {b"=======": 0, b"=========": 1})])
def test_the_block_of_two_merge_bases_inside_a_block_is_left_of_it_like_its_own_lines(
        tmp_path, head, allow_dict):
    """git writes the block of the two merge bases in the middle section of the block
    of x and y, under names of its own and in markers two longer."""
    directory, base = _two_fork_repo(tmp_path, head)
    assert len(_git(directory, "merge-base", "--all", "x", "y").split()) == 2
    merged = integrate.assemble(directory, integrate.merge_base(directory, ["x", "y"]),
                                ["x", "y"])
    text = _read(directory, "f.txt")
    marker_list = [line for line in text.splitlines()[3 if head else 0:]
                   if line[:1] in ("<", "|", "=", ">")]
    assert [(line[:1], len(line) - len(line.lstrip(line[:1]))) for line in marker_list] == [
        ("<", 7), ("|", 7), ("<", 9), ("|", 9), ("=", 9), (">", 9), ("=", 7), (">", 7)]
    assert "Temporary merge branch" in marker_list[2]
    assert merged["marker"]["separator"] == {"f.txt": allow_dict}
    kept_text = head + _line_text({4: "a and b", 5: "x and y"})
    for keep_list in ([], *[[line] for line in marker_list], marker_list[2:6], marker_list):
        _write(directory, "f.txt", "".join(
            line for line in text.splitlines(True)
            if line.rstrip("\n") not in marker_list or line.rstrip("\n") in keep_list))
        assert integrate.marker_path_list(directory, merged["marker"]) == (
            ["f.txt"] if keep_list else []), keep_list
    # The union left with one line of the inner block is carried on from
    left_text = kept_text.replace("line 4 a and b\n", "line 4 a and b\n" + marker_list[3] + "\n")
    _leave_union(directory, merged, "union", {"f.txt": left_text})
    merged = integrate.assemble(directory, merged["start"], ["union"], resume="union")
    assert integrate.marker_path_list(directory, merged["marker"]) == ["f.txt"]
    _write(directory, "f.txt", kept_text)
    assert integrate.marker_path_list(directory, merged["marker"]) == []


def test_a_line_between_the_bar_and_the_equal_line_of_a_block_is_the_block_s_own():
    inner_list = [b"<<<<<<<<< Temporary merge branch 1", b"||||||||| 1234567", b"=========",
                  b">>>>>>>>> Temporary merge branch 2"]
    line_list = [("f.txt", line) for line in (
        b"=========", b"<<<<<<< refs/codetwine-loom/merged-so-far", b"=========",
        b"||||||| merged common ancestors", *inner_list, b"=======", b"=========",
        b">>>>>>> refs/codetwine-loom/side/b", b"<<<<<<<<< Temporary merge branch 1")]
    assert integrate._block_separator_dict(line_list, ["b"]) == {"f.txt": {
        b"||||||| merged common ancestors": 1, b"=======": 1,
        **{line: 1 for line in inner_list}}}
    assert integrate._separator_frame_dict(line_list, ["b"]) == {"f.txt": {
        b"=========": 3, b"<<<<<<<<< Temporary merge branch 1": 1}}
    # The same lines in a block no side merged here ends are no block's
    assert integrate._block_separator_dict(line_list, ["c"]) == {}


def test_a_line_that_starts_a_block_under_another_name_is_read_beside_a_block_git_wrote(repo):
    directory, base = repo
    _git(directory, "checkout", "-q", "-b", "m", base)
    edge_text = "<<<<<<<<< Temporary merge branch 1\n=========\n<<<<<<< ours\n>>>>>>> theirs\n"
    _write(directory, "block.txt", "<<<<<<< refs/codetwine-loom/merged-so-far\n" + edge_text)
    _write(directory, "plain.txt", edge_text)
    _git(directory, "add", "-A")
    _git(directory, "commit", "-qm", "m")
    commit = _git(directory, "rev-parse", "HEAD").strip()
    assert integrate._marker_line_dict(directory, [commit])[commit] == [
        ("block.txt", b"<<<<<<< refs/codetwine-loom/merged-so-far"),
        ("block.txt", b"<<<<<<<<< Temporary merge branch 1"), ("block.txt", b"=========")]
    assert integrate._marker_line_dict(directory, [commit],
                                       more_size_dict={"plain.txt": {9}})[commit][3:] == [
        ("plain.txt", b"<<<<<<<<< Temporary merge branch 1"), ("plain.txt", b"=========")]
