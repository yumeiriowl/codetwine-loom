"""The tool that moves exploration into a separate loop.

Finding "where to change" is handed to a read-only RLM that returns nothing but the
places it found (file / lines / signature / why). The explorer's history is dropped
here; only the findings are returned.

This module imports nothing from agent.py. The language name injected into the
instructions comes from LANG_NAME_DICT in languages.py.
"""

import logging

from codetwine_loom import config
from codetwine_loom.languages import (LANG_NAME_DICT, PATH_EXAMPLE_DICT,
                                      SIGNATURE_EXAMPLE_DICT)
from codetwine_loom.llm import make_llm
from codetwine_loom.rlm import (COMPLETE_KEY, END_CALL_DICT, OUTPUT_LIMIT_LINE_NOTE, RLM,
                                json_guard, no_answer_reason, output_limit_note,
                                parse_json_answer, run_stage)
from codetwine_loom.search import scope as scope_mod
from codetwine_loom.textcut import cut_entry_text, json_over_count, limit_text, text_error
from codetwine_loom.tools import source

logger = logging.getLogger(__name__)

# How many explore calls went past the aspect check and the call limit since the last
# reset_explore_count()
_explore_count = 0

# Filled by set_context(): the request and the search result the explorer is given
_request: str = ""
_scope_dict: dict | None = None

# The tools the explorer gets: the reading tools plus search_again. None of the writing
# tools are among them
EXPLORE_TOOL_LIST = [source.read_source_file, source.read_definition,
                     source.grep_source, source.graph_search,
                     source.get_use_of_file, source.search_again]

_EXPLORE_INSTRUCTION_TEMPLATE = """You locate the places to change in a {lang_name}
repository. You do NOT change anything — you have no editing tools.

Each turn you write ONE ```python block, same rules as the main loop:
print() what you want to see{output_limit}.

Read {scope_first} first.
Then use read_definition and grep_source(pattern, count_only=True) to narrow down.
Never read a whole file.
Without count_only, grep_source returns "counts" for each file it walked plus the first
few hits with context lines around them; "truncated" means you did not get all of them,
and "counts_complete" false means the walk stopped before the last file.
{search_again_note}
When you are done, call COMPLETE() with the findings as the list itself and nothing else:

COMPLETE([{{"file": "{path_example}", "lines": [10, 24],
           "signature": "{signature_example}",
           "why": "one short sentence"}}])

Rules for the array:
- At most {explore_max_result} entries. Order them by how sure you are.
- "lines" must come from read_definition or scope["defs"], not from your guess.
{answer_limit}- If you find nothing, call COMPLETE([])."""


# The rule on the length of the findings as JSON
_ANSWER_LIMIT = "- Keep the whole JSON under {limit} characters.\n"

# The search_again lines, left out when SEARCH_AGAIN_MAX_CALL is 0
_SEARCH_AGAIN_NOTE = """search_again("<english terms>") re-runs the index search with \
different words when
the first result set is off target.
"""

# The one extra call made when the explorer used every turn without COMPLETE(). It runs
# no code and asks for the JSON array alone, with the explorer's history in front of it
_LATE_ANSWER_NOTE = """## Out of turns

You have no turns left, so nothing more can be read and no code will be run. Write the
places you have found as a single JSON array and NOTHING else - no ```python fence, no
prose, no explanation. Start with [ and end with ].

[{{"file": "...", "lines": [10, 24], "signature": "...", "why": "one short sentence"}}]

At most {explore_max_result} entries, ordered by how sure you are. "lines" must come
from read_definition or scope["defs"], not from your guess. If you found nothing,
answer []."""

# The last line of that call's prompt, in place of "write the next block"
_LATE_ANSWER_CLOSE = "Answer with the JSON array and nothing else."

# What the model is told when the explorer handed nothing over: because one of its
# model calls did not happen, and for any other reason (rlm.no_answer_reason)
_STOP_EARLY = "the explorer stopped early: {error}"
_NO_ANSWER = "the explorer handed nothing over: {why}"
_OVER_LIMIT = ("the explorer's findings are over the {limit} characters they may take, "
               "even with the first one alone and its texts cut")


def explore_tool_list() -> list:
    """Return the tools the explorer gets.

    They are EXPLORE_TOOL_LIST, without search_again when SEARCH_AGAIN_MAX_CALL is 0.

    Returns:
        The tool callables, in the order they are shown to the model, as a new list.
    """
    if config.SEARCH_AGAIN_MAX_CALL > 0:
        return list(EXPLORE_TOOL_LIST)
    return [tool for tool in EXPLORE_TOOL_LIST
            if tool.__name__ != source.search_again.__name__]


def set_context(request: str, scope_dict: dict) -> None:
    """Store the request and the search result for the explorer.

    Args:
        request: The change request.
        scope_dict: The return value of search_scope().
    """
    global _request, _scope_dict
    _request = request
    _scope_dict = scope_dict


def reset_explore_count() -> None:
    """Reset the explore call counter to 0.

    The counter is a module variable and goes on counting until this is called.
    """
    global _explore_count
    _explore_count = 0


def call_remain_count() -> int:
    """Return how many explore calls are left.

    Returns:
        EXPLORE_MAX_CALL minus what has been used, never below 0.
    """
    return max(0, config.EXPLORE_MAX_CALL - _explore_count)


def _refuse(error: str, **extra: object) -> dict:
    """Return one refusal of this tool, with the budget on it.

    Args:
        error: What went wrong, worded for the model.
        **extra: Further keys this particular failure carries, e.g. "raw".

    Returns:
        {"ok": False, "error": str, "calls_left": int, ...}.
    """
    return {"ok": False, "error": error, "calls_left": call_remain_count(), **extra}


def build_explore_instruction(primary_ext: str) -> str:
    """Assemble the explorer's instructions for the primary language.

    Args:
        primary_ext: The index's config.primary_ext.

    Returns:
        The assembled explorer instructions.
    """
    return _EXPLORE_INSTRUCTION_TEMPLATE.format(
        lang_name=LANG_NAME_DICT[primary_ext],
        path_example=PATH_EXAMPLE_DICT[primary_ext],
        signature_example=SIGNATURE_EXAMPLE_DICT[primary_ext],
        output_limit=output_limit_note(OUTPUT_LIMIT_LINE_NOTE),
        explore_max_result=config.EXPLORE_MAX_RESULT,
        answer_limit=limit_text(_ANSWER_LIMIT, config.EXPLORE_ANSWER_MAX_CHAR),
        search_again_note=_SEARCH_AGAIN_NOTE if config.SEARCH_AGAIN_MAX_CALL > 0 else "",
        scope_first=scope_mod.scope_first())


def parse_explore_answer(answer: object, error: str = "",
                         outcome: str | None = None) -> dict:
    """Read the explorer's COMPLETE() argument, as a list or as JSON text.

    A list or a dict handed to COMPLETE() in the Python REPL is taken as the object, and
    a string is read as JSON (rlm.parse_json_answer).

    Args:
        answer: The "answer" of run_stage(): a list, the JSON as a string, or None when
            the explorer handed nothing over.
        error: Why a model call did not happen. It goes back to the model in the
            refusal. "" is a run whose calls all happened.
        outcome: RLM.run()'s "outcome", which tells an explorer that ended itself with
            nothing from one that did not end itself.

    Returns:
        {"ok": True, "findings": list, "calls_left": int}, or
        {"ok": False, "error": str, "calls_left": int} (with "raw" as well when it is
        not valid JSON). "calls_left" is there either way.
    """
    if answer is None and error:
        logger.warning("the explorer handed nothing over: %s", error)
        return _refuse(_STOP_EARLY.format(error=error))
    if answer is None:
        why = no_answer_reason("", config.EXPLORE_MAX_TURN, END_CALL_DICT[COMPLETE_KEY],
                               outcome)
        logger.warning("the explorer handed nothing over: %s", why)
        return _refuse(_NO_ANSWER.format(why=why))

    read_dict = parse_json_answer(answer, "the explorer's reply")
    if not read_dict["ok"]:
        return _refuse(read_dict["error"], raw=read_dict["raw"])
    finding_list = read_dict["value"]
    # Keep at most EXPLORE_MAX_RESULT entries, then fit them in EXPLORE_ANSWER_MAX_CHAR
    if isinstance(finding_list, list) and len(finding_list) > config.EXPLORE_MAX_RESULT:
        logger.info("the explorer returned %d findings; keeping the first %d",
                    len(finding_list), config.EXPLORE_MAX_RESULT)
        finding_list = finding_list[:config.EXPLORE_MAX_RESULT]
    if isinstance(finding_list, list):
        finding_list = _fit_finding_list(finding_list)
        if _over_budget_count(finding_list):
            return _refuse(_OVER_LIMIT.format(limit=config.EXPLORE_ANSWER_MAX_CHAR))
    return {"ok": True, "findings": finding_list, "calls_left": call_remain_count()}


def _over_budget_count(finding_list: list) -> int:
    """Return how many characters the findings as JSON are over EXPLORE_ANSWER_MAX_CHAR.

    Args:
        finding_list: The findings.

    Returns:
        The number of characters, 0 for a list within the budget and for a budget of 0
        or less.
    """
    return json_over_count(finding_list, config.EXPLORE_ANSWER_MAX_CHAR)


# The key of a finding whose text is never cut
_FILE_KEY = "file"


def _fit_finding_list(finding_list: list) -> list:
    """Drop findings from the end until the list as JSON fits EXPLORE_ANSWER_MAX_CHAR.

    The first finding stays, with its texts but its path (_FILE_KEY) cut by what it
    alone is over (textcut.cut_entry_text).

    Args:
        finding_list: The findings as the explorer handed them over, already cut to
            EXPLORE_MAX_RESULT.

    Returns:
        The same list, trimmed in place. A budget of 0 or less leaves it as it is. It
        is still over the budget (_over_budget_count) when the first finding is with
        every text of it but its path empty.
    """
    find_count = len(finding_list)
    while len(finding_list) > 1 and _over_budget_count(finding_list):
        finding_list.pop()
    if finding_list and isinstance(finding_list[0], dict):
        cut_entry_text(finding_list[0], _over_budget_count(finding_list), (_FILE_KEY,))
    if len(finding_list) < find_count:
        logger.info("the explorer's findings were over the %d chars they were asked "
                    "for; keeping the first %d of %d", config.EXPLORE_ANSWER_MAX_CHAR,
                    len(finding_list), find_count)
    return finding_list


def explore(aspect: str) -> dict:
    """Hand the exploration to a separate loop and get back only the candidates (step 6).

    It builds a fresh RLM holding explore_tool_list() and runs it for at most
    EXPLORE_MAX_TURN turns. An explorer that used every turn without COMPLETE() is asked
    once more, in a call that runs nothing, for the places it has found. The explorer's
    history is not returned.

    Args:
        aspect: What to look into, as a short English phrase, e.g. "error handling".

    Returns:
        {"ok": True, "findings": [{"file", "lines", "signature", "why"}, ...],
         "calls_left": int}, or {"ok": False, "error": str, "calls_left": int} on
        failure. "calls_left" is on the answer either way. An aspect that is not a
        string is such a failure and spends no call.
    """
    global _explore_count
    type_error = text_error(aspect=aspect)
    if type_error:
        return _refuse(type_error)
    if _explore_count >= config.EXPLORE_MAX_CALL:
        logger.info("explore hit its call limit: %d calls", _explore_count)
        return _refuse(f"explore hit its call limit "
                       f"({config.EXPLORE_MAX_CALL} calls)")
    _explore_count += 1

    # A fresh RLM with the explorer's model, tools and instructions
    sub_rlm = RLM(llm=make_llm(config.EXPLORE_MODEL, config.EXPLORE_MAX_TOKEN,
                               "EXPLORE_MODEL", is_turn_reply=True,
                               think_level=config.EXPLORE_THINK_LEVEL),
                  tool_list=explore_tool_list(),
                  instruction=build_explore_instruction(config.PRIMARY_EXT),
                  stage="explore",
                  max_turn=config.EXPLORE_MAX_TURN)
    try:
        out_dict = run_stage(
            sub_rlm, f"{_request}\n\nFocus on this aspect only: {aspect}",
            scope_mod.stage_variable_dict(_scope_dict, sub_rlm.tool_dict),
            scope_mod.SCOPE_VAR_DOC,
            _LATE_ANSWER_NOTE.format(explore_max_result=config.EXPLORE_MAX_RESULT),
            _LATE_ANSWER_CLOSE,
            # Findings that cannot be read as JSON are sent back each time
            on_complete=json_guard("the findings"))
    except Exception as e:                            # noqa: BLE001 - see below
        # An Exception the stage raised is answered like a call of a turn that did not
        # happen
        return parse_explore_answer(None, f"{type(e).__name__}: {e}")
    # The stage's "error" travels with the answer and tells the model why a model call
    # did not happen
    # out_dict["history"] is dropped
    return parse_explore_answer(out_dict["answer"], out_dict["error"], out_dict["outcome"])
