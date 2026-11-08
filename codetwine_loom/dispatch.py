"""Read the request once and split it into what the agents are given.

The stage reads (and may run the program) with no editing tool in reach, and ends by
handing over one JSON object: a list of questions or a list of change requests, one
entry per agent. It answers nothing itself.

Which of the two the request is is decided here, and each entry is a whole request of
its own.

A third object can be handed over instead: the list of what the request leaves open that
only the person who asked can settle.

The split is by request, not by file: two tasks, or two questions, may touch the same
file. The definitions each task's text names (task_def_list) go out with the split.

A request can also be one an earlier run did not get through. What that run left is on
its branches, and the commits there carry what each agent was asked for, what it said it
did and how the machine judged it. earlier_run_text() reads that back and run_dispatch()
puts it in front of the request. A branch named here is read, never checked out.

Nothing is recorded in the edit history.

This module imports nothing from agent.py.
"""

import logging
import os
import re
from collections import Counter
from collections.abc import Callable

from codetwine_loom import config, history, integrate, plan, testcreate
from codetwine_loom.languages import (LANG_DICT, LANG_NAME_DICT, PATH_EXAMPLE_DICT, name_key,
                                      split_def_key)
from codetwine_loom.paths import knowledge_to_repo
from codetwine_loom.llm import make_llm
from codetwine_loom.rlm import (COMPLETE_KEY, OUTPUT_LIMIT_NOTE, RLM, END_CALL_DICT, cut,
                                limit_text, make_deadline_note, no_answer_reason,
                                output_limit_note, parse_json_answer, read_first_guard,
                                run_stage, text_of, turn_note_chain)
from codetwine_loom.search import scope as scope_mod
from codetwine_loom.search.query import quote_span_list
from codetwine_loom.search.scope import search_scope
from codetwine_loom.tools import run, web

logger = logging.getLogger(__name__)

# The REPL rules and the reading rules, filled in by read_rule()
_READ_RULE = """Each turn you write ONE ```python block. It runs, you see what it printed, then you write
the next block. print() what you want to see{output_limit}. Names you define stay \
available in later turns. Stop at the closing
fence — anything you write after it is thrown away and costs you the turn.

Read {scope_first} first, then
read_definition the ones the request is about. grep_source(pattern, count_only=True)
gives you the distribution before you read anything whole. Never read a whole file unless
a definition alone cannot answer the question. Every path you pass to a tool is relative
to the repository root, e.g. "{path_example}".
{run_note}{web_note}"""

_DISPATCH_INSTRUCTION_TEMPLATE = """You read a {lang_name} repository and decide what is
to be done with one request. You do NOT change anything — you have no editing tools —
and you do NOT answer anything either: what leaves this session is one JSON object
saying what the agents are to be given, and nothing else does.

{read_rule}
## What the request can be

**A question about the repository.** Then split it into the questions the agents are to
answer, one each:

COMPLETE({{"questions": ["<one whole question>", "<another>"]}})

**A change to the repository** — to its code, its tests or its documents: anything that
leaves a file added, changed or deleted. Then split it into the change requests the
agents are given, one each:

COMPLETE({{"tasks": ["<one whole change request>", "<another>"]}})

Both lists follow the same rules:

- One entry is one question or one change somebody could be asked for on its own. Split
  by what is being asked, NOT by which files it touches: two entries about the same
  file is normal and is settled afterwards.
- Each entry is read by an agent that sees nothing else — not the original request, not
  the other entries, not what you read here. So write each one as a whole request, and
  never say "the other one", "as above" or "the same file as question 1".
- Carry into each entry what you established by reading: the paths and definition
  names it is about. An agent that has to find them again spends its turns on that.
- Write each entry in the language of the request. The agent answers in the language
  it is asked in, and so is what is committed and reported afterwards.
- Keep each entry small, and when in doubt, split. An agent given too much runs out
  of turns with the work half done{plan_note}. A request written as one sentence can still \
be several
  entries.
- Change entries run at once and each starts from the repository as it is, except the
  ones that name the same definition together with its file: those run one after
  another in the order you list them, and each starts from what the one before it
  wrote.
- When an entry uses what an earlier entry adds or changes, write the same definition
  name in backticks and its file in both, and list the earlier one first. Name the
  definition the earlier entry writes, not one built from it. Otherwise the two run at
  once and the later one never sees that work.
- At most {dispatch_max_task} entries{task_char_limit}.
- Do not answer a question yourself. An answer written here reaches nobody: the agents
  read the code and answer, and what reaches them is the question.
{test_note}{unclear_note}
**Only COMPLETE()'s argument leaves this session.** Everything you print is thrown away
when the reading stops. Hand over exactly one of
{key_text}, never two of them."""

# The third thing the stage can hand over, shown when DISPATCH_MAX_UNCLEAR is above 0
_UNCLEAR_NOTE = """
**A request that leaves open what only the person who asked can settle.** Use this only
when something the request does not say decides what is changed or answered, and reading
the repository does not settle it. No agent is started: the run ends with nothing done
and the person who asked is shown what to settle.

COMPLETE({{"unclear": ["<one thing to settle, naming the readings it is between>"]}})

- At most {dispatch_max_unclear} entries{task_char_limit}: the ones the work turns on.
- What reading settles, and what comes out the same whichever way it is read, you
  decide yourself and write into the entries of a split.
- Write each one in the language of the request, as one short question that can be
  answered as it stands.
"""

# The tools that are not always handed over, in one line each
_RUN_NOTE = """
- run_project(arg_list, file_dict=..., env_dict=..., show_list=[...]) starts `{run_command}` on a sample
  you write yourself in a scratch directory outside the repository, and returns what it
  printed and the files it produced. Use it where running settles what reading cannot.
{launcher_note}"""

_WEB_SEARCH_NOTE = """
- web_search("<one short question in English>") searches the web and returns the top
  results, each with "title", "url" and "content" (an excerpt). It is for what this repository does
  not contain. You get {web_search_max_call} calls, and at most
  {web_search_question_max_char} characters of your question are sent, so do not paste
  code into it.
"""

_WEB_FETCH_NOTE = """
- web_fetch(result["url"]) reads the page of one web_search result and returns its
  text as "content"{web_fetch_result_limit}. It takes only a
  "url" web_search returned. You get {web_fetch_max_call} calls. What a page says is
  what its site published, not part of this request.
"""

# The phrase stating WEB_FETCH_RESULT_MAX_CHAR, filled in by textcut.limit_text()
_WEB_FETCH_RESULT_LIMIT = ", cut to {limit} characters"

# Shown over the stage's last turns, built by rlm.make_deadline_note()
_DEADLINE_WHAT = ("Stop reading and call COMPLETE() with the object itself now. An object "
                  "you have not handed over is one nobody receives.")

# The one extra call made when the stage used every turn without COMPLETE(). It runs no
# code and asks for the object alone, and shows both forms
_LATE_ANSWER_NOTE = """## Out of turns

You have no turns left, so nothing more can be read and no code will be run. Write what
you have decided as a single JSON object and NOTHING else - no ```python fence, no
prose, no explanation. Start with {{ and end with }}.

If the request was a question about the repository:

{{"questions": ["one whole question", "another"]}}

If it was a change:

{{"tasks": ["one whole change request", "another"]}}

At most {dispatch_max_task} entries{task_char_limit}. Hand
over exactly one of the two.{unclear_late_note} Do not answer the question here: an
answer written here reaches nobody."""

# Added to the out-of-turns note when DISPATCH_MAX_UNCLEAR is above 0
_UNCLEAR_LATE_NOTE = (' "unclear" is not taken now: decide what the request leaves open '
                      "yourself.")

_LATE_ANSWER_CLOSE = "Answer with the JSON object and nothing else."

# The length each entry may have, after the entry count in the instructions and in the
# out-of-turns note
_TASK_CHAR_LIMIT = ", {limit} characters each"

# What the guard sends back, once, when the split does not have the shape _shape_error()
# checks, or breaks a limit the instructions state: an entry longer than they allow, or
# more entries than they allow. What is still over a limit after that is cut
_REFUSAL = ("{call}() was not accepted.\n{reasons}\nFix these and call {call}() again. "
            "This is asked once only; what is still over a limit after that is cut.")
_LENGTH_REASON = ("- these entries are over the {limit} characters each entry may have: "
                  "{over}. Keep the paths and the definition names, drop the rest")
_COUNT_REASON = ('- "{key}" holds {count} entries and at most {limit} may be handed over. '
                 "{advice}")
_COUNT_ADVICE = ("Put together the ones that only make sense as one change or one "
                 "question")
_UNCLEAR_COUNT_ADVICE = ("Keep the ones the work turns on, and decide the rest yourself")
# The shape error of a split whose lists hold only entries with nothing written out
# (is_entry_text), and what the guard adds to it when it sends the split back
_HOLLOW_REASON = ("the split holds no entry written out under {keys}, only marks such as "
                  '"..." or the placeholders the instructions show')
_HOLLOW_ADVICE = ('- write out each entry in full in place of "..." or a placeholder such '
                  'as "<another>", under exactly one of {keys}')

# The keys of the two lists a split is made of, and of the list handed over in place
# of a split
SPLIT_KEY_TUPLE = ("questions", "tasks")
UNCLEAR_KEY = "unclear"

# What one entry of each list is called in a refusal
_ENTRY_NAME_DICT = {"questions": "question", "tasks": "task", UNCLEAR_KEY: "unclear entry"}

# What an earlier run left, put in front of the request
_EARLIER_BLOCK = """## What an earlier run left on {branch}

{runs}"""

# Said once, after every branch, rather than under each of them
_EARLIER_CLOSE = """Read the request below in the light of this: what is already done,
what is still missing, and what went wrong before. An agent you hand a task to sees
neither this block nor those branches, so write into the task itself whatever it needs
from here."""

_EARLIER_RUN = """- asked: {asked}
  did: {did}
  ended: {ended}"""

# One case the tests written for that run found the change does not do
_EARLIER_GAP = "  not done: {what} — {why}"

# Said first in a branch's block when it holds more runs than are shown, and after the
# blocks when more branches were named than are shown
_EARLIER_RUN_CUT = ("- ({count} run(s) before the ones below are on this branch and are "
                    "not shown here)")
_EARLIER_BRANCH_CUT = ("{count} more branch(es) were named and are not shown here: "
                       "{branches}")

# The two things the repository can be, against what that branch holds: holding the
# earlier work in the files the stage reads, or not
_NOT_IN_TREE = ("The repository you are reading does NOT hold this work. It is only on "
                "that branch, so what is described here is not in the files you read.")
_IN_TREE = ("The repository you are reading already holds this work, so it is in the "
            "files you read.")


# What follows "with the work half done" in the instructions when a change run plans
# before it edits
_PLAN_NOTE = """, and an agent making a change plans at most
  {plan_max_file} files"""

# The bullet added when a stage after the merge writes the tests for the change
_TEST_CREATE_NOTE = """- Do not hand over an entry whose work is writing tests. A stage \
after the merge writes the tests for what the change touched; a test written here for \
what another entry adds is red until then. A test the request itself asks for is the \
exception: it goes in the entry whose change it checks, and no entry is told to leave \
it out.
"""

# What an ASCII name looks like in a task's text, hyphen-joined words (a COBOL name)
# included
_NAME_RE = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*")

# The name a word in backticks holds in a task's text, with "()" after it or not
_QUOTE_NAME_RE = re.compile(r"([^\W\d][\w-]*)(?:\(\))?")

# A stretch of code in single backticks within one line of a task's text, as its group
_CODE_SPAN_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")

# A name inside a stretch of code
_CODE_WORD_RE = re.compile(r"[^\W\d][\w-]*")

# A path a task's text holds: words joined by "/", "." or "-", ending in "." and an
# extension
_PATH_RE = re.compile(r"(?<![A-Za-z0-9_./-])[A-Za-z0-9_./-]*[A-Za-z0-9_]\.([A-Za-z0-9]+)"
                      r"(?![A-Za-z0-9_-])")


def dispatch_test_note() -> str:
    """Return the line keeping tests out of the split, or "" when no stage writes them.

    Returns:
        The bullet, ending in a newline, when a stage that writes tests follows the run
        (testcreate.is_stage_on), or an empty string.
    """
    return _TEST_CREATE_NOTE if testcreate.is_stage_on() else ""


def dispatch_plan_note() -> str:
    """Return the words saying how many files a change run plans, or "".

    The empty string is returned when a change run has no planning stage.

    Returns:
        _PLAN_NOTE with PLAN_MAX_FILE filled in when TWO_STAGE is on, or an empty
        string.
    """
    return _PLAN_NOTE.format(plan_max_file=config.PLAN_MAX_FILE) if config.TWO_STAGE \
        else ""


def dispatch_run_note(tool_list: list[Callable]) -> str:
    """Return the run_project line, or "".

    The empty string is returned when the stage is not handed run_project or no launcher
    is configured.

    Args:
        tool_list: The tools the stage is handed.

    Returns:
        The bullet, ending in a newline, or an empty string.
    """
    if not config.RUN_COMMAND or not _has_tool(tool_list, run.run_project):
        return ""
    return _RUN_NOTE.format(run_command=" ".join(config.RUN_COMMAND),
                            launcher_note=run.launcher_note("  "))


def dispatch_web_note(tool_list: list[Callable]) -> str:
    """Return the lines of the tools that reach outside the repository, or "".

    Only the tools the stage is handed get a line; the empty string is returned when it
    is handed neither.

    Args:
        tool_list: The tools the stage is handed.

    Returns:
        The bullets, ending in a newline, or an empty string.
    """
    text = ""
    if _has_tool(tool_list, web.web_search):
        text += _WEB_SEARCH_NOTE.format(
            web_search_max_call=config.WEB_SEARCH_MAX_CALL,
            web_search_question_max_char=config.WEB_SEARCH_QUESTION_MAX_CHAR)
    if _has_tool(tool_list, web.web_fetch):
        text += _WEB_FETCH_NOTE.format(
            web_fetch_max_call=config.WEB_FETCH_MAX_CALL,
            web_fetch_result_limit=limit_text(_WEB_FETCH_RESULT_LIMIT,
                                              config.WEB_FETCH_RESULT_MAX_CHAR))
    return text


def _has_tool(tool_list: list[Callable], tool: Callable) -> bool:
    """Say whether a tool list holds the tool, compared by name.

    Args:
        tool_list: The tools a stage is handed.
        tool: The tool to look for.

    Returns:
        True when one of the tools has the tool's name.
    """
    return any(entry.__name__ == tool.__name__ for entry in tool_list)


def read_rule(primary_ext: str, tool_list: list[Callable]) -> str:
    """Fill _READ_RULE in for the primary language.

    The configured values and the lines of the tools the stage is handed go in as well.

    Args:
        primary_ext: The index's config.primary_ext.
        tool_list: The tools the stage is handed.

    Returns:
        The REPL and reading rules, ending in a newline, with the run_project,
        web_search and web_fetch lines of the tools that are in tool_list.
    """
    return _READ_RULE.format(
        path_example=PATH_EXAMPLE_DICT[primary_ext],
        output_limit=output_limit_note(OUTPUT_LIMIT_NOTE),
        run_note=dispatch_run_note(tool_list),
        web_note=dispatch_web_note(tool_list),
        scope_first=scope_mod.scope_first())


def build_dispatch_instruction(primary_ext: str) -> str:
    """Assemble the splitting stage's instructions for the primary language.

    Args:
        primary_ext: The index's config.primary_ext.

    Returns:
        The assembled instructions.
    """
    task_char_limit = _task_char_limit()
    return _DISPATCH_INSTRUCTION_TEMPLATE.format(
        lang_name=LANG_NAME_DICT[primary_ext],
        read_rule=read_rule(primary_ext, dispatch_tool_list()),
        dispatch_max_task=config.DISPATCH_MAX_TASK,
        task_char_limit=task_char_limit,
        plan_note=dispatch_plan_note(),
        test_note=dispatch_test_note(),
        unclear_note=_dispatch_unclear_note(task_char_limit),
        key_text=_key_text(split_key_tuple()))


def _task_char_limit() -> str:
    """Return the phrase stating how long one entry may be.

    Returns:
        _TASK_CHAR_LIMIT with DISPATCH_TASK_MAX_CHAR filled in (textcut.limit_text), or
        "" when the setting is 0 or less.
    """
    return limit_text(_TASK_CHAR_LIMIT, config.DISPATCH_TASK_MAX_CHAR)


def _dispatch_unclear_note(task_char_limit: str) -> str:
    """Return the block describing the "unclear" object, or "" when it is not taken.

    Args:
        task_char_limit: The phrase stating how long one entry may be, or "".

    Returns:
        _UNCLEAR_NOTE filled in, or an empty string when DISPATCH_MAX_UNCLEAR is 0.
    """
    if config.DISPATCH_MAX_UNCLEAR <= 0:
        return ""
    return _UNCLEAR_NOTE.format(dispatch_max_unclear=config.DISPATCH_MAX_UNCLEAR,
                                task_char_limit=task_char_limit)


def split_key_tuple(is_late_answer: bool = False) -> tuple[str, ...]:
    """Return the keys the stage may hand an object over under.

    Args:
        is_late_answer: Whether the object came from the call made after the last turn
            rather than from COMPLETE().

    Returns:
        SPLIT_KEY_TUPLE, followed by UNCLEAR_KEY when DISPATCH_MAX_UNCLEAR is above 0
        and the object came from COMPLETE().
    """
    if config.DISPATCH_MAX_UNCLEAR > 0 and not is_late_answer:
        return SPLIT_KEY_TUPLE + (UNCLEAR_KEY,)
    return SPLIT_KEY_TUPLE


def max_entry_of(key: str) -> int:
    """Return how many entries the list under a key may hold.

    Args:
        key: One of SPLIT_KEY_TUPLE, or UNCLEAR_KEY.

    Returns:
        DISPATCH_MAX_UNCLEAR for UNCLEAR_KEY, DISPATCH_MAX_TASK for the others.
    """
    return config.DISPATCH_MAX_UNCLEAR if key == UNCLEAR_KEY else config.DISPATCH_MAX_TASK


def _key_text(key_list: list[str] | tuple[str, ...]) -> str:
    """Write keys out as a quoted list.

    Examples:
        ("tasks",)                        -> '"tasks"'
        ("questions", "tasks")            -> '"questions" and "tasks"'
        ("questions", "tasks", "unclear") -> '"questions", "tasks" and "unclear"'

    Args:
        key_list: The keys, one or more.

    Returns:
        The keys in double quotes, joined by commas and a final "and"; one key alone in
        double quotes.
    """
    quote_list = [f'"{key}"' for key in key_list]
    if len(quote_list) == 1:
        return quote_list[0]
    return f"{', '.join(quote_list[:-1])} and {quote_list[-1]}"


def dispatch_tool_list() -> list[Callable]:
    """Return the tools the splitting stage gets.

    They are the planning stage's list as it is (plan.plan_tool_list()).

    Returns:
        The tool callables, in the order they are shown to the model.
    """
    return plan.plan_tool_list()


# A letter or a digit, in any script
_LETTER_OR_DIGIT_RE = re.compile(r"[^\W_]")

# An entry wrapped whole in one pair of angle brackets, with what it wraps
_WRAP_RE = re.compile(r"\s*<([^<>]*)>\s*")


def _example_form(text: str) -> str:
    """Return an entry as it is compared with the entries the instructions show.

    Args:
        text: The entry.

    Returns:
        The entry stripped of white space and of every "<" and ">" at either end, in
        lower case.
    """
    return text.strip().strip("<>").strip().lower()


def _example_entry_set(text: str) -> frozenset[str]:
    """Return the entries the instructions show as the shape of a split.

    Args:
        text: The instructions' texts that show a split, put together.

    Returns:
        _example_form() of each entry of the lists shown as {{"<key>": [...]}}.
    """
    return frozenset(
        _example_form(entry)
        for list_text in re.findall(r'\{\{"\w+": \[(.*?)\]\}\}', text)
        for entry in re.findall(r'"([^"]*)"', list_text))


# The entries the instructions show as the shape of a split
_EXAMPLE_ENTRY_SET = _example_entry_set(
    _DISPATCH_INSTRUCTION_TEMPLATE + _UNCLEAR_NOTE + _LATE_ANSWER_NOTE)


def is_entry_text(entry: object) -> bool:
    """Say whether an entry of one of a split's lists is one to hand out.

    Args:
        entry: One entry of the list.

    Returns:
        True for a string holding a letter or a digit that is not one of the entries
        the instructions show (_EXAMPLE_ENTRY_SET, compared as _example_form() gives
        it).
    """
    return isinstance(entry, str) and bool(_LETTER_OR_DIGIT_RE.search(entry)) \
        and _example_form(entry) not in _EXAMPLE_ENTRY_SET


def _entry_text_list(entry_list: object) -> list[str]:
    """Return the entries of one of the lists that are handed out.

    Args:
        entry_list: What the stage put under one of the keys.

    Returns:
        The entries is_entry_text() takes, stripped, in their order, an entry wrapped
        whole in one pair of angle brackets (_WRAP_RE) given as what it wraps. A value
        that is not a list at all comes back empty.
    """
    if not isinstance(entry_list, list):
        return []
    return [_unwrap(entry) for entry in entry_list if is_entry_text(entry)]


def _unwrap(text: str) -> str:
    """Return an entry stripped, without the one pair of angle brackets wrapping it whole.

    Args:
        text: The entry.

    Returns:
        What _WRAP_RE's brackets wrap, stripped, for an entry they wrap whole; the
        entry stripped otherwise.
    """
    wrap_match = _WRAP_RE.fullmatch(text)
    return wrap_match.group(1).strip() if wrap_match else text.strip()


def _entry_list_within_limit(entry_list: object, max_entry: int) -> list[str]:
    """Return one of the lists as the instructions ask for it: capped and cut.

    Args:
        entry_list: What the stage put under one of the keys.
        max_entry: How many entries the list may hold (max_entry_of()).

    Returns:
        At most max_entry of _entry_text_list()'s strings, each cut to
        DISPATCH_TASK_MAX_CHAR.
    """
    keep_list = []
    for text in _entry_text_list(entry_list):
        keep = cut(text, config.DISPATCH_TASK_MAX_CHAR)
        if keep != text:
            logger.warning("entry %d is %d characters and at most %d are handed out; "
                           "the rest is cut", len(keep_list) + 1, len(text),
                           config.DISPATCH_TASK_MAX_CHAR)
        keep_list.append(keep)
    if len(keep_list) > max_entry:
        logger.warning("the split holds %d entries and at most %d are handed out; the "
                       "rest are dropped", len(keep_list), max_entry)
    return keep_list[:max_entry]


def _key_with_entry_list(split_dict: dict, key_tuple: tuple[str, ...]) -> list[str]:
    """Return which of the keys hold an entry to hand out.

    Args:
        split_dict: The split as read from JSON.
        key_tuple: The keys an object may be handed over under (split_key_tuple()).

    Returns:
        The keys among key_tuple, in that order, whose list keeps an entry after
        _entry_text_list() and the key's cap (max_entry_of()).
    """
    return [key for key in key_tuple
            if _entry_text_list(split_dict.get(key))[:max_entry_of(key)]]


def _shape_error(split_value: object, key_tuple: tuple[str, ...]) -> str:
    """Say what is wrong with the shape of a split read as JSON, or accept it.

    Args:
        split_value: The split as read from JSON.
        key_tuple: The keys an object may be handed over under (split_key_tuple()).

    Returns:
        The error for a split that is not an object, or that holds two or more of the
        lists; for one none of whose lists holds an entry to hand out, _hollow_error()'s
        error when it gives one and the "no entry under" error otherwise; "" when it
        holds exactly one list with an entry to hand out.
    """
    if not isinstance(split_value, dict):
        return "the split is not a JSON object"
    key_list = _key_with_entry_list(split_value, key_tuple)
    if len(key_list) > 1:
        return (f"the split holds {_key_text(key_list)}; it holds exactly one of "
                f"{_key_text(key_tuple)}")
    if not key_list:
        return _hollow_error(split_value, key_tuple) \
            or f"the split holds no entry under any of {_key_text(key_tuple)}"
    return ""


def _hollow_error(split_value: object, key_tuple: tuple[str, ...]) -> str:
    """Say which lists of a split with no entry to hand out hold only strings not taken.

    Args:
        split_value: The split as read from JSON.
        key_tuple: The keys an object may be handed over under (split_key_tuple()).

    Returns:
        _HOLLOW_REASON naming the keys among key_tuple whose list holds a string that is
        not blank and none that is_entry_text() takes, for an object none of whose lists
        holds an entry to hand out; "" when there is no such key or it is no such
        object.
    """
    if not isinstance(split_value, dict) or _key_with_entry_list(split_value, key_tuple):
        return ""
    hollow_key_list = [
        key for key in key_tuple if isinstance(split_value.get(key), list)
        and any(isinstance(entry, str) and entry.strip() for entry in split_value[key])]
    return _HOLLOW_REASON.format(keys=_key_text(hollow_key_list)) if hollow_key_list else ""


def _over_the_limit(answer: object) -> str:
    """Say what is wrong with a split the instructions do not allow, or accept it.

    Three things are checked of a split that can be read as JSON (read_first_guard
    sends back one that cannot): whether it has the shape parse_dispatch() takes
    (_shape_error), how long one entry may be and how many entries there may be. Both
    limits are held to the entries parse_dispatch() takes (_entry_text_list).

    Args:
        answer: The argument of the COMPLETE() that ended the run, one that can be read.

    Returns:
        The refusal text, or "" when the split has the shape and every list is within
        both limits.
    """
    split_value = parse_json_answer(answer, "the split")["value"]
    key_tuple = split_key_tuple()
    shape_error = _shape_error(split_value, key_tuple)
    reason_list = [f"- {shape_error}"] if shape_error else []
    if _hollow_error(split_value, key_tuple):
        reason_list.append(_HOLLOW_ADVICE.format(keys=_key_text(key_tuple)))
    if not isinstance(split_value, dict):
        split_value = {}
    limit = config.DISPATCH_TASK_MAX_CHAR
    for key in key_tuple:
        entry_list = split_value.get(key)
        if not isinstance(entry_list, list):
            continue
        text_count = len(_entry_text_list(entry_list))
        if text_count > max_entry_of(key):
            reason_list.append(_COUNT_REASON.format(
                key=key, count=text_count, limit=max_entry_of(key),
                advice=_UNCLEAR_COUNT_ADVICE if key == UNCLEAR_KEY else _COUNT_ADVICE))
        over_list = [f"{_ENTRY_NAME_DICT[key]} {number} ({len(text)} characters)"
                     for number, text in enumerate(_entry_text_list(entry_list), 1)
                     if cut(text, limit) != text]
        if limit > 0 and over_list:
            reason_list.append(_LENGTH_REASON.format(limit=limit, over=", ".join(over_list)))
    if not reason_list:
        return ""
    logger.info("%s() sent back once: %s", END_CALL_DICT[COMPLETE_KEY],
                "; ".join(reason_list))
    return _REFUSAL.format(call=END_CALL_DICT[COMPLETE_KEY], reasons="\n".join(reason_list))


def _path_in_text(path: str, text: str) -> bool:
    """Say whether a path, or its last part, is in the text as a whole word.

    Examples:
        ("src/cfg.py", "read src/cfg.py first")   -> True
        ("src/cfg.py", "read lib/cfg.py first")   -> True
        ("src/cfg.py", "src/cfg.py’s loader")     -> True
        ("a.py", "read data.py first")            -> False

    Args:
        path: A repository-relative path.
        text: The text to look in.

    Returns:
        True when the path or its base name is there, not preceded by an ASCII letter,
        digit, "_", "." or "-", nor followed by an ASCII letter, digit, "_" or "-".
    """
    return any(re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(part)}(?![A-Za-z0-9_-])", text)
               for part in (path, os.path.basename(path)))


def _task_name_list(task: str) -> list[str]:
    """Return the names a task's text is looked up by in the index.

    Examples:
        "fix MAIN-PARA in a.cbl" -> ["fix", "MAIN-PARA", "MAIN", "PARA", "in", "a", "cbl"]

    Args:
        task: One change request.

    Returns:
        The ASCII words of the text, a hyphen-joined one whole and then in its parts,
        followed by the index's names outside ASCII the text holds
        (scope.text_name_list()), each once, in that order.
    """
    name_list = []
    for word in _NAME_RE.findall(task):
        name_list += [word] + (word.split("-") if "-" in word else [])
    return list(dict.fromkeys(name_list + scope_mod.text_name_list(task)))


def _is_code_name(word: str) -> bool:
    """Say whether an ASCII word of a task's text is written as a name in code.

    Examples:
        "parse_path" -> True
        "parsePath"  -> True
        "MAIN-PARA"  -> True
        "parse"      -> False
        "Parse"      -> False
        "read-only"  -> False

    Args:
        word: One _NAME_RE match.

    Returns:
        True when it holds "_", a lower-case letter followed by an upper-case one, or
        "-" between upper-case words.
    """
    return "_" in word or bool(re.search(r"[a-z][A-Z]", word)) \
        or ("-" in word and word.isupper())


def _task_file_list(task: str, path_list: list[str]) -> tuple[list[str], str]:
    """Return the files a task's text names, and the text without the paths it holds.

    Args:
        task: One change request.
        path_list: The index's files, as knowledge paths.

    Returns:
        (the files, the text with every path it holds taken out). The files are the
        index's files the text names by repository-relative path or base name
        (_path_in_text()), as knowledge paths, followed by the base names of the paths
        the text holds that name no file of the index. A path counts when its
        extension is one of LANG_DICT's or of an index file's.
    """
    file_list = [path for path in path_list if os.path.basename(path) in task
                 and _path_in_text(knowledge_to_repo(path, scope_mod.PROJECT_NAME), task)]
    ext_set = set(LANG_DICT) | {os.path.splitext(path)[1][1:] for path in path_list}
    index_base_set = {os.path.basename(path) for path in file_list}
    text_path_list = [match.group(0) for match in _PATH_RE.finditer(task)
                      if match.group(1) in ext_set]
    file_list += sorted({os.path.basename(path) for path in text_path_list} - index_base_set)
    text = _PATH_RE.sub(lambda match: " " if match.group(1) in ext_set else match.group(0),
                        task)
    return file_list, text


def _quote_name_set(text: str) -> set[str]:
    """Return the names a task's text writes in backticks.

    Args:
        text: A task's text.

    Returns:
        The name of each word in backticks (query.quote_span_list) that is a name
        (_QUOTE_NAME_RE).
    """
    name_set = set()
    for start, end in quote_span_list(text):
        match = _QUOTE_NAME_RE.fullmatch(text, start + 1, end - 1)
        if match:
            name_set.add(match.group(1))
    return name_set


def _code_word_set(text: str) -> set[str]:
    """Return the names a task's text writes inside stretches of code in backticks.

    Examples:
        "add `sub(a: float, b: float) -> float`" -> {"sub", "a", "float", "b"}

    Args:
        text: A task's text.

    Returns:
        The names (_CODE_WORD_RE) of each stretch of code (_CODE_SPAN_RE).
    """
    return {word for span in _CODE_SPAN_RE.finditer(text)
            for word in _CODE_WORD_RE.findall(span.group(1))}


def _new_def_key_tuple(task: str, path_list: list[str],
                       name_set: set[str]) -> tuple[list[str], list[str]]:
    """Return the keys of the definitions a task names that the index does not hold.

    A name is named outright when it is written as a name in code (_is_code_name()) or
    in backticks (_quote_name_set()), and in passing when it stands inside a stretch of
    code in backticks (_code_word_set()). A name that is part of a path, or among the
    names the index holds in a file the text names, counts as neither. Each name is
    paired with every file the text names.

    Args:
        task: One change request.
        path_list: The index's files, as knowledge paths.
        name_set: The names of the index definitions the task names.

    Returns:
        (the keys of the names named outright, the keys of the names named in passing),
        each "<file>::<name>" sorted, the file as _task_file_list() gives it and the name
        in the form the file's language compares names in (name_key()).
    """
    file_list, text = _task_file_list(task, path_list)
    code_name_set = {word for word in _NAME_RE.findall(text) if _is_code_name(word)}
    code_name_set |= _quote_name_set(text)
    return tuple(sorted({f"{path}::{name_key(path, name)}" for name in word_set - name_set
                         for path in file_list})
                 for word_set in (code_name_set, _code_word_set(text)))


def task_def_list(task_list: list[str]) -> list[list[str]]:
    """Return the definitions each task names, as definition keys per task.

    A definition of the index is named when the task's text holds both its name and its
    file, as the file's repository-relative path or as its base name. The names looked
    for are the ASCII words of the text, hyphen-joined ones whole and in their parts,
    and the index's names outside ASCII the text holds (scope.text_name_list()).

    A definition the index does not hold, one a task is to add, is named the same way
    by a name and a file (_new_def_key_tuple()). Its key is kept only when two or more
    tasks name it, outright or in passing, and one of them names it outright.

    Args:
        task_list: The change requests, as the split hands them over.

    Returns:
        One sorted list of definition keys per task, in the tasks' order. Every list is
        empty when no index is loaded.
    """
    if scope_mod.INDEX is None:
        return [[] for _ in task_list]
    path_list = scope_mod.INDEX.file_path_list()
    key_set_list = []
    new_set_list = []
    outright_set = set()
    for task in task_list:
        key_set = set()
        name_set = set()
        for name in _task_name_list(task):
            for def_dict in scope_mod.INDEX.def_list_by_name(name):
                path = split_def_key(scope_mod.to_repo(def_dict["key"]))[0]
                if _path_in_text(path, task):
                    key_set.add(def_dict["key"])
                    name_set.add(name)
        key_set_list.append(key_set)
        outright_list, passing_list = _new_def_key_tuple(task, path_list, name_set)
        new_set_list.append(set(outright_list) | set(passing_list))
        outright_set |= set(outright_list)
    # A key of a definition the index does not hold is kept when two or more tasks name
    # it and one of them names it outright
    count_dict = Counter(key for new_set in new_set_list for key in new_set)
    return [sorted(key_set | {key for key in new_set
                              if count_dict[key] > 1 and key in outright_set})
            for key_set, new_set in zip(key_set_list, new_set_list)]


def parse_dispatch(answer: object, error: str = "", is_late_answer: bool = False,
                   outcome: str | None = None) -> dict:
    """Read the stage's COMPLETE() argument as one of the three lists.

    The lists are of questions, of tasks, or of what the request leaves open. A dict is
    taken as the object and a string is read as JSON, allowing a fence or a sentence
    around it (rlm.parse_json_answer).

    Args:
        answer: The "answer" of RLM.run(): a dict, the JSON as a string, or None when
            the stage handed nothing over.
        error: RLM.run()'s "error". A stage whose model call did not happen is reported
            with it, not as one that used every turn. "" is a stage that did not stop
            early.
        is_late_answer: Whether the object came from the call made after the last turn.
            Such an object is not taken under UNCLEAR_KEY.
        outcome: RLM.run()'s "outcome", which tells a stage that ended itself with
            nothing from one that did not end itself.

    Returns:
        {"ok": True, "questions": [str, ...]} for a question, {"ok": True, "tasks":
        [str, ...]} for a change, {"ok": True, "unclear": [str, ...]} for a request the
        person who asked has to settle first, or {"ok": False, "error": str} (with "raw"
        as well when there was something to show). At most one of the three keys is
        present.
    """
    if answer is None:
        why = no_answer_reason(error, config.DISPATCH_MAX_TURN, "COMPLETE", outcome)
        logger.warning("the splitting stage handed nothing over: %s", why)
        return {"ok": False, "error": f"the splitting stage handed nothing over: {why}"}

    # The limits the instructions state are kept on the lists below
    read_dict = parse_json_answer(answer, "the split")
    if not read_dict["ok"]:
        return read_dict
    split_dict = read_dict["value"]
    key_tuple = split_key_tuple(is_late_answer)
    shape_error = _shape_error(split_dict, key_tuple)
    if shape_error:
        return {"ok": False, "error": shape_error,
                "raw": text_of(answer)[:config.RAW_EXCERPT_CHAR]}
    key = _key_with_entry_list(split_dict, key_tuple)[0]
    return {"ok": True,
            key: _entry_list_within_limit(split_dict.get(key), max_entry_of(key))}


def render_earlier(record_list: list[dict]) -> str:
    """Build the block describing what earlier runs left on the branches named.

    Args:
        record_list: One integrate.branch_record_dict() result per branch, in the order
            they were named.

    Returns:
        One block per branch that says anything, or "" when there are at most
        config.INTEGRATE_WORK_MAX_ITEM branches and none of them says anything. A branch
        whose commits carry neither a request nor a record describes nothing and is left
        out rather than shown empty. Of a branch holding more than
        config.INTEGRATE_WORK_MAX_ITEM runs the newest that many are shown, under a line
        saying how many are not; of more branches than that the first that many are
        shown, and a line after the blocks names the rest. The cases a branch's tests
        found not done are the ones integrate.gap_list_of() reads off all its runs, each
        shown once, under the newest run shown that recorded it. A case no run shown
        recorded goes under the newest run shown for which the repository holds the
        work or not as it does for the newest run that recorded it, or, when there is
        none, under the line saying how many runs are not shown. Before the first run
        shown (or that line), and again before a run where it changes, a line says
        whether the repository holds the work of the runs that follow (_held_list_of).
    """
    limit = config.INTEGRATE_WORK_MAX_ITEM
    block_list = []
    for branch_dict in record_list[:limit]:
        # The runs come oldest first; the newest are the ones shown
        told_list = branch_dict["told"]
        cut_count = max(len(told_list) - limit, 0)
        show_list = told_list[cut_count:]
        all_held_list = _held_list_of(branch_dict)
        held_list = all_held_list[cut_count:]
        gap_list_list: list[list[dict]] = [[] for _ in show_list]
        cut_gap_list = []
        for gap_dict in integrate.gap_list_of(told_list) if show_list else []:
            record_index = next(
                (run_index for run_index in reversed(range(len(told_list)))
                 if gap_dict in integrate.decision_list_of(
                     told_list[run_index].get("gap", ""))),
                len(told_list) - 1)
            index = record_index - cut_count if record_index >= cut_count else next(
                (run_index for run_index in reversed(range(len(show_list)))
                 if held_list[run_index] == all_held_list[record_index]), None)
            if index is None:
                cut_gap_list.append((all_held_list[record_index], gap_dict))
            else:
                gap_list_list[index].append(gap_dict)
        run_line_list = [_earlier_run_text(record_dict, gap_list)
                         for record_dict, gap_list in zip(show_list, gap_list_list)]
        if not run_line_list:
            logger.info("%s says nothing about what was done on it, so it is left out "
                        "of the request", branch_dict["branch"])
            continue
        entry_list = list(zip(held_list, run_line_list))
        if cut_count:
            # The runs not shown carry the cases whose runs are cut and that no run
            # shown holds the same answer for
            entry_list.insert(0, (
                cut_gap_list[0][0] if cut_gap_list else held_list[0],
                "\n".join([_EARLIER_RUN_CUT.format(count=cut_count),
                           *(_EARLIER_GAP.format(**gap_dict)
                             for _, gap_dict in cut_gap_list)])))
        line_list = []
        for index, (held, run_line) in enumerate(entry_list):
            # Whether the repository holds the work is said again where it changes
            if not index or held != entry_list[index - 1][0]:
                line_list.append(_IN_TREE if held else _NOT_IN_TREE)
            line_list.append(run_line)
        block_list.append(_EARLIER_BLOCK.format(branch=branch_dict["branch"],
                                                runs="\n".join(line_list)))
    rest_list = record_list[limit:]
    if rest_list:
        block_list.append(_EARLIER_BRANCH_CUT.format(
            count=len(rest_list),
            branches=", ".join(branch_dict["branch"] for branch_dict in rest_list)))
    return "\n\n".join(block_list + [_EARLIER_CLOSE]) if block_list else ""


def _held_list_of(branch_dict: dict) -> list[bool]:
    """Return whether the working tree holds the work of each run a branch records.

    Args:
        branch_dict: One integrate.branch_record_dict() result.

    Returns:
        One value per entry of its "told", in that order: the "held" of the earlier
        union's part the entry is in, and the branch's own "held" for the rest.
    """
    held_list = [part_dict["held"] for part_dict in branch_dict["earlier_list"]
                 for _ in part_dict["told"]]
    return held_list + [branch_dict["held"]] * (len(branch_dict["told"]) - len(held_list))


def _earlier_run_text(record_dict: dict, gap_list: list[dict]) -> str:
    """Render one run a branch's commits record.

    Args:
        record_dict: One integrate.commit_record_list() entry.
        gap_list: The cases shown under that run, as integrate.gap_list_of() returns
            them.

    Returns:
        _EARLIER_RUN filled in (what it was asked, what it said it did, how it ended),
        followed by one _EARLIER_GAP line per case of gap_list.
    """
    line_list = [_EARLIER_RUN.format(
        asked=cut(record_dict["asked"], config.DISPATCH_TASK_MAX_CHAR),
        did=cut(record_dict["did"], config.EDIT_HISTORY_DID_CHAR) or history.NO_ANSWER_TEXT,
        ended=history.end_text(record_dict))]
    line_list += [_EARLIER_GAP.format(**gap_dict) for gap_dict in gap_list]
    return "\n".join(line_list)


def earlier_run_text(branch_list: list[str]) -> str:
    """Read what the runs on these branches left and render it as one block.

    Args:
        branch_list: The branch names, as they were passed.

    Returns:
        render_earlier()'s block, or "" when nothing was passed.

    Raises:
        RuntimeError: When the repository holds no branch of one of those names.
    """
    if not branch_list:
        return ""
    return render_earlier([integrate.branch_record_dict(config.PROJECT_DIR, name)
                           for name in branch_list])


def run_dispatch(request: str, resume_list: list[str] | None = None) -> dict:
    """Search once, then have the model split the request up.

    A COMPLETE() that cannot be read as JSON is sent back each time (read_first_guard),
    and one the instructions do not allow (_over_the_limit) once for the model to fix;
    only what is still over the limit after that is cut. A stage that
    uses every turn without COMPLETE() is asked once more, in a call that runs nothing,
    for the object itself. That call takes a split alone, not what the request leaves
    open.

    Args:
        request: The request, as the user wrote it.
        resume_list: The branches an earlier run left its work on, whose commits are
            read and put in front of the request. The search is made on the request
            alone either way.

    Returns:
        parse_dispatch()'s result with "outcome", "iters", "refused_completes" (how many
        COMPLETE() calls the guard sent back), "harvested" (whether it came from the one
        extra call rather than from COMPLETE()), "scope" and "history" added, and "defs"
        (task_def_list()) for a split into tasks. "outcome" is COMPLETE_KEY when the
        stage ended itself, None when it did not.

    Raises:
        RuntimeError: When one of the resume branches is not in the repository.
    """
    plan.reset_read_tool()
    # Read before the search: a branch that is not there stops the run before any model
    # call
    prior_text = earlier_run_text(resume_list or [])
    # The search is made on the request alone, without what an earlier run left
    scope_dict = search_scope(request)
    text = "\n\n".join(part for part in (prior_text, request) if part)
    stage = RLM(llm=make_llm(config.DISPATCH_MODEL, config.DISPATCH_MAX_TOKEN,
                             "DISPATCH_MODEL", is_turn_reply=True,
                             think_level=config.DISPATCH_THINK_LEVEL),
                tool_list=integrate.read_record_tool_list(dispatch_tool_list()),
                instruction=build_dispatch_instruction(config.PRIMARY_EXT),
                stage="dispatch",
                max_turn=config.DISPATCH_MAX_TURN)
    state_note = plan.make_state_note()
    out_dict = run_stage(
        stage, text, scope_mod.stage_variable_dict(scope_dict, stage.tool_dict),
        scope_mod.SCOPE_VAR_DOC,
        _LATE_ANSWER_NOTE.format(
            dispatch_max_task=config.DISPATCH_MAX_TASK,
            task_char_limit=_task_char_limit(),
            unclear_late_note=_UNCLEAR_LATE_NOTE if config.DISPATCH_MAX_UNCLEAR > 0 else ""),
        _LATE_ANSWER_CLOSE, late_state=state_note,
        turn_note=turn_note_chain(
            state_note, make_deadline_note(config.DISPATCH_DEADLINE_TURN, _DEADLINE_WHAT)),
        on_complete=read_first_guard("the split", _over_the_limit))
    dispatch_dict = parse_dispatch(out_dict["answer"], out_dict["error"],
                                   is_late_answer=bool(out_dict["harvested"]),
                                   outcome=out_dict["outcome"])
    for key in ("outcome", "iters", "refused_completes", "harvested", "history"):
        dispatch_dict[key] = out_dict[key]
    dispatch_dict["scope"] = scope_dict
    if dispatch_dict["ok"] and "tasks" in dispatch_dict:
        dispatch_dict["defs"] = task_def_list(dispatch_dict["tasks"])
    if dispatch_dict["ok"]:
        key = next(list_key for list_key in SPLIT_KEY_TUPLE + (UNCLEAR_KEY,)
                   if list_key in dispatch_dict)
        logger.info("the request is %d %s(s) after %d turns", len(dispatch_dict[key]),
                    _ENTRY_NAME_DICT[key], out_dict["iters"])
    return dispatch_dict
