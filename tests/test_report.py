"""What a dispatching run tells the person who asked, once it has finished."""

import logging
import os
import subprocess

import pytest

from codetwine_loom import config, gitcmd, report
from codetwine_loom.rlm import BRIEF_RETRY_NOTE


def _git(directory, *args):
    """Run one git command in a test repository and fail the test when it does not."""
    done = subprocess.run(["git", "-C", str(directory), *args],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, f"git {args[0]} failed: {done.stderr}"
    return done.stdout


def _write(directory, name, text):
    """Write one file in a test repository, making the directories it needs."""
    full = os.path.join(str(directory), name)
    os.makedirs(os.path.dirname(full) or str(directory), exist_ok=True)
    with open(full, "w", encoding="utf-8") as handle:
        handle.write(text)


def _commit(directory, message):
    """Commit whatever is in the working tree and return the commit."""
    _git(directory, "add", "-A")
    _git(directory, "commit", "-q", "--allow-empty", "-m", message)
    return _git(directory, "rev-parse", "HEAD").strip()


# The trailers one finished change run leaves on its commit, as sandbox.run_trailer_dict()
# writes them
_TRAILERS = ("Codetwine-Loom-Attempt: 1\nCodetwine-Loom-Verdict: {verdict}\n"
             "Codetwine-Loom-Why: {why}\nCodetwine-Loom-Did: {did}\nCodetwine-Loom-Exit-Status: 0")


def _message(request, did="did it", verdict="unverified", why="nothing checks it",
             tests=""):
    """Build a commit message of the shape sandbox.commit_message() writes."""
    body = f"sandbox: {request}\n\n{request}\n\n" + _TRAILERS.format(
        verdict=verdict, why=why, did=did)
    return f"{body}\nCodetwine-Loom-Tests-Changed: {tests}" if tests else body


@pytest.fixture
def repo(tmp_path):
    """A repository two agents committed to and a third combined.

    Task 1 changes one file and adds a test; task 2 adds a file and deletes another.
    The union is a merge commit whose first parent is the base, as one a merging run
    leaves is.

    Yields:
        (path, {"base", "one", "two", "union"} of commits).
    """
    directory = str(tmp_path / "repo")
    os.makedirs(directory)
    _git(directory, "init", "-q", "-b", "main")
    _git(directory, "config", "user.email", "t@example.invalid")
    _git(directory, "config", "user.name", "t")
    _write(directory, "src/a.py", "one\n")
    _write(directory, "src/gone.py", "old\n")
    base = _commit(directory, "base")

    _git(directory, "checkout", "-q", "-b", "dispatch/1/1", base)
    _write(directory, "src/a.py", "ONE\n")
    _write(directory, "tests/test_a.py", "check\n")
    _git(directory, "add", "-A")
    _git(directory, "commit", "-q", "-m",
         _message("change a", did="changed a and its test",
                  tests="tests/test_a.py"))
    one = _git(directory, "rev-parse", "HEAD").strip()

    _git(directory, "checkout", "-q", "-b", "dispatch/1/2", base)
    _write(directory, "src/b.py", "two\n")
    os.remove(os.path.join(directory, "src/gone.py"))
    _git(directory, "add", "-A")
    _git(directory, "commit", "-q", "-m",
         _message("add b", did="added b", verdict="passed", why=""))
    two = _git(directory, "rev-parse", "HEAD").strip()

    _git(directory, "checkout", "-q", "--detach", base)
    tree = _git(directory, "merge-tree", "--write-tree", one, two).strip()
    union = _git(directory, "commit-tree", tree, "-p", base, "-p", one, "-p", two,
                 "-m", _message("do both", did="merged them", verdict="settled",
                                why="")).strip()
    _git(directory, "branch", "codetwine-loom/union", union)
    return directory, {"base": base, "one": one, "two": two, "union": union}


def _result(commits, integration=True, unfinished=False):
    """Build the return value of sandbox.dispatch() the report is given."""
    return {
        "tasks": ["change a", "add b"],
        "branches": [
            {"n": 1, "request": "change a", "branch": "dispatch/1/1",
             "attempt": 1, "commit": commits["one"], "exit_status": 0,
             "error": "", "verdict": "unverified"},
            {"n": 2, "request": "add b", "branch": "dispatch/1/2", "attempt": 1,
             "commit": "" if unfinished else commits["two"],
             "exit_status": 1 if unfinished else 0,
             "error": "the container was killed" if unfinished else "",
             "verdict": "failed" if unfinished else "passed"},
        ],
        "integration": {"branch": "codetwine-loom/union", "commit": commits["union"],
                        "base": commits["base"], "verdict": "settled", "merged": "main",
                        "exit_status": 0, "land_why": "",
                        "land": {"files": 4, "paths": [], "patch": "p", "last": ""}}
        if integration else None,
        "dropped": [], "swept": [],
        "resume": ["dispatch/1/2"] if unfinished else [],
    }


def _with_tests(directory, commits, trailer="", **override):
    """Commit a test on top of the union and return the result holding the stage."""
    _git(directory, "checkout", "-q", "--detach", commits["union"])
    _write(directory, "tests/test_b.py", "check b\n")
    commit = _commit(directory, _message("do both", did="wrote a test for b",
                                         verdict="settled", why="",
                                         tests="tests/test_b.py")
                     + "\nCodetwine-Loom-Stage: test-create\nCodetwine-Loom-Tests-Base: red"
                     + trailer)
    result_dict = _result(commits)
    result_dict["integration"]["test_create"] = {
        "verdict": "settled", "why": "", "skipped": "", "cases": [],
        "tests_changed": ["tests/test_b.py"],
        "non_test_changed": [], "tests_base": "red", "ended": "complete",
        "exit_status": 0, "patch": "p", "files": 1, "branch": "codetwine-loom/union",
        "commit": commit, **override}
    result_dict["integration"]["settle_commit"] = commit
    return result_dict


# == Reading the name and kind of every change ===========


def test_the_working_tree_is_read_when_no_range_is_named(repo):
    directory, _ = repo
    _write(directory, "src/a.py", "changed\n")
    _write(directory, "src/new.py", "new\n")
    os.remove(os.path.join(directory, "src/gone.py"))
    # An untracked file is not in the diff: change_by_kind_dict() is what adds those
    assert gitcmd.name_status(directory) == {
        "added": [], "modified": ["src/a.py"], "deleted": ["src/gone.py"]}


def test_one_commit_is_read_against_its_parent(repo):
    directory, commits = repo
    assert gitcmd.name_status(directory, f"{commits['one']}^", commits["one"]) == {
        "added": ["tests/test_a.py"], "modified": ["src/a.py"], "deleted": []}
    assert gitcmd.name_status(directory, f"{commits['two']}^", commits["two"]) == {
        "added": ["src/b.py"], "modified": [], "deleted": ["src/gone.py"]}


def test_a_merge_commit_is_read_against_its_first_parent(repo):
    """That parent is what the merge was written against, so the range is the whole
    union rather than what one of the branches added to it."""
    directory, commits = repo
    assert gitcmd.name_status(directory, f"{commits['union']}^", commits["union"]) == {
        "added": ["src/b.py", "tests/test_a.py"], "modified": ["src/a.py"],
        "deleted": ["src/gone.py"]}


def test_a_range_that_cannot_be_read_changes_nothing(repo):
    """A commit that is not there is answered rather than raised, so a report of a run
    whose branch has since gone is still built."""
    directory, _ = repo
    assert gitcmd.name_status(directory, "0" * 40, "1" * 40) == {
        "added": [], "modified": [], "deleted": []}


def test_a_path_is_never_reported_twice(repo):
    """A file both parents of a merge changed is one path in the range, not two."""
    directory, commits = repo
    read = gitcmd.name_status(directory, commits["base"], commits["union"])
    assert read["modified"].count("src/a.py") == 1


# == What the machine reads off the change =================


def test_the_change_is_read_from_the_base_to_the_combined_commit(repo):
    directory, commits = repo
    read = report.fact_dict(directory, "do both", _result(commits))
    assert read["base"] == commits["base"] and read["commit"] == commits["union"]
    assert read["files"] == {"added": ["src/b.py", "tests/test_a.py"],
                             "modified": ["src/a.py"], "deleted": ["src/gone.py"]}
    assert read["tests"] == ["tests/test_a.py"]
    assert read["project_dir"] == directory and "diff" not in read


def test_the_last_settled_commit_ends_the_change(repo):
    directory, commits = repo
    read = report.fact_dict(directory, "do both", _with_tests(directory, commits))
    assert "tests/test_b.py" in read["files"]["added"]
    assert read["tests"] == ["tests/test_a.py", "tests/test_b.py"]


def test_branches_that_were_not_combined_leave_no_change(repo, monkeypatch):
    directory, commits = repo
    monkeypatch.setattr(report.llm, "make_llm",
                        lambda *a, **k: pytest.fail("the model was called"))
    read = report.fact_dict(directory, "do both", _result(commits, integration=False))
    assert read["commit"] == ""
    assert report.render(read) == "" and report.summary(read) == ""
    assert report.report(directory, "do both", _result(commits, integration=False)) == ""


@pytest.mark.parametrize("override", [
    {"verdict": "unsettled", "exit_status": 1, "land": None,
     "land_why": "the merge was not settled"},
    {"land": None, "land_why": "the change does not apply to the working tree"}])
def test_a_union_that_was_not_put_into_the_tree_leaves_no_change(repo, monkeypatch,
                                                                 override):
    directory, commits = repo
    monkeypatch.setattr(config, "REPORT_MODEL", "some/model")
    monkeypatch.setattr(report.llm, "make_llm",
                        lambda *a, **k: pytest.fail("the model was called"))
    result_dict = _result(commits)
    result_dict["integration"].update(override)
    read = report.fact_dict(directory, "do both", result_dict)
    assert read["commit"] == "" and read["gap"] == []
    assert report.report(directory, "do both", result_dict) == ""


def test_a_union_with_nothing_more_to_put_in_is_still_what_the_tree_holds(repo):
    directory, commits = repo
    result_dict = _result(commits)
    result_dict["integration"]["land"] = {"files": 0, "paths": [], "patch": "p",
                                          "last": ""}
    assert report.fact_dict(directory, "do both", result_dict)["commit"] == \
        commits["union"]


def test_the_files_are_printed_once_with_the_letter_git_gives_them(repo):
    directory, commits = repo
    rendered = report.render(report.fact_dict(directory, "do both", _result(commits)))
    assert rendered == ("changed:\n  A src/b.py\n  A tests/test_a.py  (test)\n"
                        "  M src/a.py\n  D src/gone.py")


def _prompt_of(directory, result_dict, monkeypatch):
    """Return the prompt the report's model is handed for one result."""
    prompt_list = []
    monkeypatch.setattr(report.llm, "make_llm",
                        lambda *a, **k: lambda p: prompt_list.append(p) or "ok")
    report.summary(report.fact_dict(directory, "do both", result_dict))
    return prompt_list[0]


def test_the_model_is_handed_the_request_the_files_and_the_diff(repo, monkeypatch):
    directory, commits = repo
    prompt = _prompt_of(directory, _result(commits), monkeypatch)
    assert "do both" in prompt
    assert "A tests/test_a.py  (test)" in prompt and "D src/gone.py" in prompt
    assert "+ONE" in prompt
    # The language comes last, after the diff
    assert prompt.rstrip().endswith(
        "## The language\n\nWrite the report in the language the request above is "
        "written in.")
    assert prompt.index("+ONE") < prompt.index("## The language")


def test_what_the_tests_found_not_done_is_read_and_handed_to_the_model(repo, monkeypatch):
    directory, commits = repo
    result_dict = _with_tests(
        directory, commits,
        trailer='\nCodetwine-Loom-Gap: [{"what":"case_8","why":"b.py never reads the flag"}]')
    assert report.fact_dict(directory, "do both", result_dict)["gap"] == \
        [{"what": "case_8", "why": "b.py never reads the flag"}]
    prompt = _prompt_of(directory, result_dict, monkeypatch)
    assert "What the tests found the change does not do" in prompt
    assert "- case_8: b.py never reads the flag" in prompt
    assert prompt.index("- case_8") < prompt.index("## The language")


def test_no_gap_section_is_handed_over_when_the_tests_found_none(repo, monkeypatch):
    directory, commits = repo
    prompt = _prompt_of(directory, _with_tests(directory, commits), monkeypatch)
    assert "does not do" not in prompt


def test_the_model_is_not_handed_how_the_change_was_made(repo, monkeypatch):
    directory, commits = repo
    prompt = _prompt_of(directory, _result(commits), monkeypatch)
    for word in ("dispatch/1/1", "codetwine-loom/union", "changed a and its test", "merged them",
                 "unverified", "settled"):
        assert word not in prompt


@pytest.mark.parametrize("max_char", [10, 300, 100000, 0])
def test_the_diff_is_cut_to_its_cap(repo, monkeypatch, max_char):
    """The model is handed the start of the diff and the count of what follows, as
    a cut of the whole diff reads; a cap of 0 hands over the whole of it."""
    directory, commits = repo
    monkeypatch.setattr(config, "REPORT_DIFF_MAX_CHAR", max_char)
    prompt = _prompt_of(directory, _result(commits), monkeypatch)
    whole_text = gitcmd.diff_text(directory, commits["base"], commits["union"])
    assert f"## The diff\n\n{report.cut(whole_text, max_char)}\n" in prompt
    assert ("chars omitted" in prompt) is (0 < max_char < len(whole_text))


def test_the_diff_is_not_read_where_no_model_is_asked(repo, monkeypatch):
    directory, commits = repo
    monkeypatch.setattr(config, "REPORT_MODEL", "")
    for name in ("diff_head", "diff_text"):
        monkeypatch.setattr(gitcmd, name, lambda *a: pytest.fail("the diff was read"))
    assert report.report(directory, "do both", _result(commits)).startswith("changed:\n")


def test_a_diff_that_cannot_be_read_leaves_the_machine_s_lines(repo, monkeypatch, caplog):
    directory, commits = repo
    monkeypatch.setattr(config, "REPORT_MODEL", "some/model")
    monkeypatch.setattr(report.llm, "make_llm",
                        lambda *a, **k: pytest.fail("the model was called"))

    def no_git(*args):
        raise OSError("git cannot be started")

    monkeypatch.setattr(gitcmd, "diff_head", no_git)
    with caplog.at_level(logging.WARNING):
        said = report.report(directory, "do both", _result(commits))
    assert said.startswith("changed:\n") and "git cannot be started" in caplog.text


def test_a_limit_of_zero_asks_for_no_length(repo, monkeypatch):
    directory, commits = repo
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 0)
    prompt = _prompt_of(directory, _result(commits), monkeypatch)
    assert "Keep it under" not in prompt
    monkeypatch.setattr(report.llm, "make_llm", lambda *a, **k: lambda p: "w" * 9000)
    assert report.summary(report.fact_dict(directory, "do both", _result(commits))) == "w" * 9000


# == Asking the model ====================================


def test_no_model_is_called_when_there_is_none_to_call(repo, monkeypatch):
    directory, commits = repo
    monkeypatch.setattr(config, "REPORT_MODEL", "")
    monkeypatch.setattr(report.llm, "make_llm",
                        lambda *a, **k: pytest.fail("the model was called"))
    assert report.summary(report.fact_dict(directory, "do both", _result(commits))) == ""


def test_what_the_model_wrote_is_cut_to_the_cap_it_was_given(repo, monkeypatch):
    directory, commits = repo
    monkeypatch.setattr(config, "DISPATCH_ANSWER_MAX_CHAR", 20)
    monkeypatch.setattr(report.llm, "make_llm", lambda *a, **k: lambda p: "w" * 500)
    said = report.summary(report.fact_dict(directory, "do both", _result(commits)))
    assert said.startswith("w" * 20) and "w" * 21 not in said
    assert "chars omitted" in said


class _CutOnceLLM:
    """A stand-in for what make_llm builds, whose first reply is cut off."""

    def __init__(self):
        self.prompt_list = []
        self.last_finish_reason = None

    def __call__(self, prompt):
        self.prompt_list.append(prompt)
        self.last_finish_reason = "length" if len(self.prompt_list) == 1 else "stop"
        return "half of" if len(self.prompt_list) == 1 else "all of it"


def test_a_summary_cut_at_the_output_limit_is_asked_for_again(repo, monkeypatch):
    directory, commits = repo
    llm = _CutOnceLLM()
    monkeypatch.setattr(report.llm, "make_llm", lambda *a, **k: llm)
    assert report.summary(report.fact_dict(directory, "do both", _result(commits))) == "all of it"
    assert len(llm.prompt_list) == 2
    assert llm.prompt_list[1].endswith(BRIEF_RETRY_NOTE)


def test_a_summary_still_cut_after_the_retries_says_so(repo, monkeypatch):
    directory, commits = repo
    monkeypatch.setattr(config, "CUT_REPLY_MAX_RETRY", 0)
    llm = _CutOnceLLM()
    monkeypatch.setattr(report.llm, "make_llm", lambda *a, **k: llm)
    said = report.summary(report.fact_dict(directory, "do both", _result(commits)))
    assert said == "half of" + report.CUT_END_NOTE


# == A report that cannot be written is not a run that failed


def test_a_call_that_does_not_come_back_leaves_the_files_reported(repo, monkeypatch):
    directory, commits = repo

    def raises(*a, **k):
        raise RuntimeError("no route to the provider")

    monkeypatch.setattr(report.llm, "make_llm", raises)
    said = report.report(directory, "do both", _result(commits))
    assert "M src/a.py" in said
    assert "no route" not in said


def test_a_repository_that_cannot_be_read_reports_nothing(repo):
    """It gives back an empty string rather than raising: the run's exit status is
    decided by what the agents did, not by whether it could be described."""
    _, commits = repo
    assert report.report("", "do both", _result(commits)) == ""


def test_the_account_comes_before_the_files_it_is_about(repo, monkeypatch):
    directory, commits = repo
    monkeypatch.setattr(report.llm, "make_llm",
                        lambda *a, **k: lambda p: "both tasks landed")
    said = report.report(directory, "do both", _result(commits))
    assert said.index("both tasks landed") < said.index("changed:")
