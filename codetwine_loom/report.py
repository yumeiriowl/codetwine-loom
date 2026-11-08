"""What a dispatching run tells the person who asked.

The report is about the change that was put into the working tree, read from git: the
commit the combined change was written against, and the last commit that was settled on
top of it. A run whose change was not put in is not reported. Two things are said, by
different writers. The files are read from git and listed as they came. What was changed
in them, and where, is one model call that is handed the request, the list of files and
the diff.

What cannot be read is left out: report() gives back what it has, or an empty string.
"""

import logging

from codetwine_loom import config, gitcmd, integrate, llm
from codetwine_loom.rlm import CUT_END_NOTE, ask_whole, cut, limit_text
from codetwine_loom.textcut import CUT_NOTE

logger = logging.getLogger(__name__)

# The line that names the report's length limit. DISPATCH_ANSWER_MAX_CHAR of 0 is no
# limit, and then the line is left out
_LENGTH_RULE = "Keep it under {limit} characters."

# What the model is asked for: an account of the change, written from the diff
_INSTRUCTION = """You are writing the last thing this tool prints: a report, for the
person who asked, of what was changed in their repository for their request.

Below are the request, the files the change touched, and the diff of the change.

Write a summary, not a walk through the diff:

- what was asked for, in a sentence
- where the change was made and what it does there, in a sentence or two for each file
  or each group of files that serve one purpose, naming the functions, classes,
  settings or sections that were added, changed or removed
- the tests that were added, changed or removed, in a sentence or two: what they cover, not each
  case

Do not quote code, and do not list values, lines or test cases one by one.

Write about the change alone. Do not say how it was made: nothing about tasks, agents,
branches, stages, merges, verdicts, turns, or the order or way the work was done.

Say only what the diff shows. Do not say whether the change is correct, do not suggest
changes of your own, and do not offer to do anything further. Where the diff was cut,
say that the part after the cut was not read.

{length_rule}

## The request

{request}

## The files the change touched

{files}

## The diff

{diff}
{gap_block}
## The language

Write the report in the language the request above is written in.
"""

# Added to the instruction when the tests written for the change recorded cases the
# change does not do
_GAP_BLOCK = """
## What the tests found the change does not do

The tests written for the change found that the code does not do these parts of the
request. End the report with them, under a heading of their own, as what was asked for
and is not done yet: one line each, saying what is missing and which file it is in.

{gap}
"""

_NO_FILE_TEXT = "(no file)"

# What marks a changed file the runs recorded as one of the repository's tests
_TEST_MARK = "  (test)"


def fact_dict(project_dir: str, request: str, result_dict: dict) -> dict:
    """Read the change a dispatching run put into the working tree.

    The change runs from the commit the run started at to the last commit that was
    settled.

    Args:
        project_dir: The target repository, as an absolute path.
        request: The request as it was given.
        result_dict: The return value of sandbox.dispatch() for a request that was split
            into tasks.

    Returns:
        {"project_dir", "request", "base", "commit", "files", "tests", "gap"}. "files"
        is gitcmd.name_status()'s three lists, "tests" the paths the records of the
        commits from "base" to "commit" (integrate.commit_record_list()) name in their
        "tests_changed", and "gap" integrate.gap_list_in() over the same range.
        "commit" is "" and "files", "tests" and "gap" empty when the branches were not
        combined, and when what they were combined into was not put into the working
        tree (the union's "land" is None).
    """
    union_dict = result_dict.get("integration") or {}
    base = union_dict.get("base") or ""
    commit = (union_dict.get("settle_commit") or union_dict.get("commit") or "") \
        if union_dict.get("land") else ""
    if not (base and commit):
        return {"project_dir": project_dir, "request": request, "base": base,
                "commit": "", "tests": [],
                "files": {kind: [] for kind in gitcmd.CHANGE_KIND_TUPLE}, "gap": []}
    record_list = integrate.commit_record_list(project_dir, base, commit)
    test_set = {path for record_dict in record_list
                for path in (record_dict.get("tests_changed") or "").split()}
    return {"project_dir": project_dir, "request": request, "base": base,
            "commit": commit, "files": gitcmd.name_status(project_dir, base, commit),
            "tests": sorted(test_set),
            "gap": integrate.gap_list_in(project_dir, base, commit)}


def _change_line_list(read_dict: dict) -> list[str]:
    """Return one line per file the change touched, with git's letter and the test mark.

    Args:
        read_dict: The return value of fact_dict().

    Returns:
        The lines, in git's kinds and sorted inside each, or one line saying there was
        no file.
    """
    test_set = set(read_dict["tests"])
    line_list = [f"{letter} {path}{_TEST_MARK if path in test_set else ''}"
                 for kind, letter in gitcmd.LETTER_BY_KIND_DICT.items()
                 for path in read_dict["files"][kind]]
    return line_list or [_NO_FILE_TEXT]


def render(read_dict: dict) -> str:
    """Return the files the change touched, as the machine read them from git.

    Args:
        read_dict: The return value of fact_dict().

    Returns:
        The block, starting at "changed:", or "" when there is no change to read.
    """
    if not read_dict["commit"]:
        return ""
    return "\n".join(["changed:", *(f"  {line}" for line in _change_line_list(read_dict))])


def _gap_block(gap_list: list[dict]) -> str:
    """Fill _GAP_BLOCK in with the cases the tests found the change does not do.

    Args:
        gap_list: [{"what", "why"}, ...], as fact_dict() reads them.

    Returns:
        The block, one "- what: why" line per case, or "" when there is no case.
    """
    if not gap_list:
        return ""
    return _GAP_BLOCK.format(gap="\n".join(f"- {entry_dict['what']}: {entry_dict['why']}"
                                           for entry_dict in gap_list))


def _diff_text(read_dict: dict) -> str:
    """Return the diff of the change, held to REPORT_DIFF_MAX_CHAR characters.

    Args:
        read_dict: The return value of fact_dict(), for a change that was read.

    Returns:
        The first REPORT_DIFF_MAX_CHAR characters of the diff from "base" to "commit"
        (gitcmd.diff_head), followed by the "... (N chars omitted)" marker when more
        follow; the whole diff when the setting is 0.
    """
    head, omit_count = gitcmd.diff_head(read_dict["project_dir"], read_dict["base"],
                                        read_dict["commit"], config.REPORT_DIFF_MAX_CHAR)
    return head + (CUT_NOTE.format(omitted=omit_count) if omit_count else "")


def summary(read_dict: dict) -> str:
    """Ask one model call to say what was changed and where, in the language of the request.

    The diff is read here, once there is a model to hand it to.

    Args:
        read_dict: The return value of fact_dict().

    Returns:
        What the model wrote, cut to DISPATCH_ANSWER_MAX_CHAR, or "" when there is no
        model to ask, no change, or the call did not come back. A reply cut off at the
        output limit is asked for again (rlm.ask_whole), and one still cut after that
        ends with rlm.CUT_END_NOTE before the cap is applied. A diff that cannot be
        read and a call that fails are logged and give back "".
    """
    if not config.REPORT_MODEL:
        logger.info("REPORT_MODEL is empty, so the report is the machine's lines alone")
        return ""
    if not read_dict["commit"]:
        return ""
    try:
        diff_text = _diff_text(read_dict)
    except OSError as e:
        logger.warning("the diff of the change could not be read (%s). What the machine "
                       "read is printed without the report", e)
        return ""
    prompt = _INSTRUCTION.format(
        length_rule=limit_text(_LENGTH_RULE, config.DISPATCH_ANSWER_MAX_CHAR),
        request=read_dict["request"], files="\n".join(_change_line_list(read_dict)),
        diff=diff_text, gap_block=_gap_block(read_dict.get("gap") or []))
    try:
        answer_text, is_cut = ask_whole(
            llm.make_llm(config.REPORT_MODEL, config.REPORT_MAX_TOKEN, "REPORT_MODEL"),
            prompt)
    except Exception as e:  # noqa: BLE001 - any provider error leaves the run reported
        logger.warning("the report could not be written (%s). What the machine read is "
                       "printed without it", e)
        return ""
    return cut(answer_text.strip() + (CUT_END_NOTE if is_cut else ""),
               config.DISPATCH_ANSWER_MAX_CHAR)


def report(project_dir: str, request: str, result_dict: dict) -> str:
    """Return the report of a dispatching run.

    Args:
        project_dir: The target repository, as an absolute path.
        request: The request as it was given.
        result_dict: The return value of sandbox.dispatch().

    Returns:
        The model's account and the machine's file list, in that order, or "" when
        there was nothing to report. It never raises: what cannot be read is left out.
    """
    try:
        read_dict = fact_dict(project_dir, request, result_dict)
    except OSError as e:
        logger.warning("the change this run left could not be read (%s), so there is "
                       "no report", e)
        return ""
    return "\n\n".join(part for part in (summary(read_dict), render(read_dict)) if part)
