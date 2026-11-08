"""Answer one question about the repository, and put several such answers together.

Two stages live here, both read-only and both ending in COMPLETE("<the answer>"):

- The answering stage is one agent's whole run. It is handed one question, reads the
  code with the reading tools (plan.answer_tool_list()), and hands the answer over. It
  runs on ANSWER_MODEL with ANSWER_MAX_TURN turns.
- The combining stage is handed the request as the user wrote it together with the
  agents' answers, and hands over one answer. It runs on INTEGRATE_MODEL with
  INTEGRATE_MAX_TURN turns, and it has the same tools. Its instructions send it to read
  only what the answers disagree on or leave open.

Nothing is recorded in the edit history.

This module imports nothing from agent.py.
"""

import logging
from collections.abc import Callable

from codetwine_loom import config, dispatch, history, integrate, plan
from codetwine_loom.languages import LANG_NAME_DICT
from codetwine_loom.llm import make_llm
from codetwine_loom.rlm import (COMPLETE_KEY, RLM, END_CALL_DICT, cut, limit_text,
                                make_deadline_note, no_answer_reason, refuse_once,
                                run_stage, text_of, turn_note_chain)
from codetwine_loom.search import scope as scope_mod
from codetwine_loom.search.scope import search_scope

logger = logging.getLogger(__name__)

_ANSWER_INSTRUCTION_TEMPLATE = """You answer one question about a {lang_name} repository.
You do NOT change anything — you have no editing tools. What leaves this session is the
answer you hand to COMPLETE(), and nothing else does.

{read_rule}{combine_note}
## The answer

COMPLETE("<the answer>")

- Answer the question that was asked, in the language it was asked in.
{content_rule}
- Say what you read, not what you would expect. Where you did not check something the
  answer rests on, say which part that is.
{length_rule}

**Only COMPLETE()'s argument leaves this session.** Everything you print is thrown away
when the reading stops, so an answer you printed is an answer nobody receives."""

# The line that says what the answer names: the files and the definitions
CONTENT_RULE = """- Name the files and the definitions it is about — a path and a definition name the
  reader can open, not "the parser" or "that function"."""

# The line that names the answer's length limit. DISPATCH_ANSWER_MAX_CHAR of 0 is no
# limit, and then the line is left out
_LENGTH_RULE = "- Keep it under {limit} characters."

# The combining stage's section. It sits between the reading rules and the answer rules
_COMBINE_NOTE = """
## Putting the agents' answers together

The question was split into the questions listed under "## What the agents answered",
and each of them was answered by an agent that read the code and saw nothing else. Their
answers are what you start from, not the code:

- Keep the paths and definition names they gave. Check that each one is there before
  you repeat it (grep_source or read_definition), and leave out one that is not.
- Read the code only where two answers disagree, where a question came back with no
  answer, or where an answer says it did not check something the question rests on. Do
  not read everything again: your turns are for those gaps.
- Where you could not settle a disagreement or fill a gap, say so in the answer rather
  than choosing.
- Answer the question as the user asked it, which is the one at the top, not the
  agents' questions one after another.
"""

# What the combining stage is handed after the request: the agents' questions and what
# each one answered. An agent that answered nothing is shown as such
_ANSWER_TITLE = """## What the agents answered

The question above was split into these questions, and each was answered by an agent of
its own that read the code and saw nothing else."""

_ANSWER_ENTRY = """{n}. {question}
   answered: {answer}"""

_NO_ANSWER_ENTRY = """{n}. {question}
   no answer came back: {why}"""


# Shown over a stage's last turns, built by rlm.make_deadline_note()
_DEADLINE_WHAT = ("Stop reading and call COMPLETE() with the answer now. An answer you "
                  "have not handed over is one nobody receives.")

# The one extra call made when a stage used every turn without COMPLETE(). It runs no
# code and asks for the answer alone
_LATE_ANSWER_NOTE = """## Out of turns

You have no turns left, so nothing more can be read and no code will be run. Write the
answer from what you have already read, and NOTHING else - no ```python fence. Where you
did not manage to check something the answer rests on, say which part that is instead
of filling it in."""

_LATE_ANSWER_CLOSE = "Answer with the answer and nothing else."

# What the guard sends back, once, when the answer is longer than the instructions
# allow. An answer still over the limit after that is cut
_LENGTH_REFUSAL = ("{call}() was not accepted. The answer is {length} characters and "
                   "the limit is {limit}. Shorten it - keep the paths, the definition "
                   "names and what you did not check, drop the rest - and call {call}() "
                   "again. This is asked once only; an answer still over the limit "
                   "after that is cut at the end.")

# What the guard sends back, once, when the call that ends the run was given no answer
_EMPTY_REFUSAL = ("{call}() was called with an empty answer, and its argument is the "
                  "whole of what leaves this session. Call {call}() again with the "
                  "answer itself.")

# The names of the two stages (RLM's stage)
_ANSWER_STAGE = "answer"
_COMBINE_STAGE = "combine"


def build_answer_instruction(primary_ext: str, content_rule: str = CONTENT_RULE) -> str:
    """Assemble the answering agent's instructions for the primary language.

    Args:
        primary_ext: The index's config.primary_ext.
        content_rule: The line that says what the answer names.

    Returns:
        The assembled instructions.
    """
    return _instruction(primary_ext, combine_note="", content_rule=content_rule)


def build_combine_instruction(primary_ext: str) -> str:
    """Assemble the combining stage's instructions for the primary language.

    They are the answering agent's with the section on the agents' answers added.

    Args:
        primary_ext: The index's config.primary_ext.

    Returns:
        The assembled instructions.
    """
    return _instruction(primary_ext, combine_note=_COMBINE_NOTE, content_rule=CONTENT_RULE)


def _instruction(primary_ext: str, combine_note: str, content_rule: str) -> str:
    """Fill the one template in.

    Args:
        primary_ext: The index's config.primary_ext.
        combine_note: The combining stage's section, or "".
        content_rule: The line that says what the answer names.

    Returns:
        The assembled instructions.
    """
    return _ANSWER_INSTRUCTION_TEMPLATE.format(
        lang_name=LANG_NAME_DICT[primary_ext],
        read_rule=dispatch.read_rule(primary_ext, plan.answer_tool_list()),
        combine_note=combine_note,
        content_rule=content_rule,
        length_rule=limit_text(_LENGTH_RULE, config.DISPATCH_ANSWER_MAX_CHAR))


def render_answer_list(answer_list: list[dict], for_user: bool = False) -> str:
    """Build the block that puts the agents' answers after the request.

    Each question is cut to DISPATCH_TASK_MAX_CHAR and each answer to
    DISPATCH_ANSWER_MAX_CHAR.

    Args:
        answer_list: One entry per question, [{"n", "request", "answer", "error"}, ...],
            "answer" being None or "" for an agent that handed nothing over and "error"
            saying why.
        for_user: True leaves out the heading and the sentence that explain the block to
            the combining stage.

    Returns:
        The block, starting at its heading unless for_user.
    """
    line_list = [] if for_user else [_ANSWER_TITLE, ""]
    for entry_dict in answer_list:
        question = cut(entry_dict["request"], config.DISPATCH_TASK_MAX_CHAR)
        answer = entry_dict.get("answer") or ""
        if answer:
            line_list.append(_ANSWER_ENTRY.format(
                n=entry_dict["n"], question=question,
                answer=cut(answer, config.DISPATCH_ANSWER_MAX_CHAR)))
        else:
            line_list.append(_NO_ANSWER_ENTRY.format(
                n=entry_dict["n"], question=question,
                why=entry_dict.get("error") or history.NO_ANSWER_TEXT))
    return "\n".join(line_list)


def run_answer(question: str, content_rule: str = CONTENT_RULE) -> dict:
    """Have one agent read the code and answer one question.

    Args:
        question: The question the agent answers.
        content_rule: The line that says what the answer names.

    Returns:
        What _run_stage() returns.
    """
    return _run_stage(
        text=question, scope_dict=search_scope(question), stage=_ANSWER_STAGE,
        instruction=build_answer_instruction(config.PRIMARY_EXT, content_rule),
        llm=make_llm(config.ANSWER_MODEL, config.ANSWER_MAX_TOKEN, "ANSWER_MODEL",
                     is_turn_reply=True,
                     think_level=config.ANSWER_THINK_LEVEL),
        max_turn=config.ANSWER_MAX_TURN,
        deadline_turn=config.ANSWER_DEADLINE_TURN,
        late_answer_on=config.ANSWER_LATE_ANSWER)


def run_combine(request: str, answer_list: list[dict]) -> dict:
    """Have one stage put the agents' answers together into the answer to the request.

    The search is made on the request itself: the candidate scope the stage starts from
    is the request's, not any one agent's.

    Args:
        request: The request, as the user wrote it.
        answer_list: The agents' answers, as render_answer_list() takes them.

    Returns:
        What _run_stage() returns.
    """
    return _run_stage(
        text="\n\n".join((request, render_answer_list(answer_list))),
        scope_dict=search_scope(request), stage=_COMBINE_STAGE,
        instruction=build_combine_instruction(config.PRIMARY_EXT),
        llm=make_llm(config.INTEGRATE_MODEL, config.INTEGRATE_MAX_TOKEN,
                     "INTEGRATE_MODEL", is_turn_reply=True,
                     think_level=config.INTEGRATE_THINK_LEVEL),
        max_turn=config.INTEGRATE_MAX_TURN,
        deadline_turn=config.INTEGRATE_DEADLINE_TURN,
        late_answer_on=True)


def _over_the_limit(answer: object) -> str:
    """Say that the answer is empty or longer than DISPATCH_ANSWER_MAX_CHAR.

    Args:
        answer: The argument of the COMPLETE() that ended the run.

    Returns:
        The refusal text, or "" when the answer, stripped of outer white space, is not
        empty and a cut to the limit leaves it as it is: it is within the limit, it is
        what such a cut left, or the limit is off.
    """
    limit = config.DISPATCH_ANSWER_MAX_CHAR
    text = text_of(answer).strip() if answer is not None else ""
    length = len(text)
    if not length:
        logger.info("%s() sent back once: the answer is empty",
                    END_CALL_DICT[COMPLETE_KEY])
        return _EMPTY_REFUSAL.format(call=END_CALL_DICT[COMPLETE_KEY])
    if cut(text, limit) == text:
        return ""
    logger.info("%s() sent back once: the answer is %d characters over %d",
                END_CALL_DICT[COMPLETE_KEY], length, limit)
    return _LENGTH_REFUSAL.format(call=END_CALL_DICT[COMPLETE_KEY], length=length,
                                  limit=limit)


def _run_stage(text: str, scope_dict: dict, stage: str, instruction: str, llm: Callable,
               max_turn: int, deadline_turn: int, late_answer_on: bool) -> dict:
    """Run one read-only stage to its answer.

    A COMPLETE() whose answer is empty or longer than the instructions allow is sent
    back once for the model to fix; only an answer still over the limit after that is
    cut. With late_answer_on True, a stage that uses every turn without COMPLETE() is
    asked once more, in a call that runs nothing, for the answer itself.

    Args:
        text: The request the stage is given.
        scope_dict: search_scope()'s result, put in the REPL as the scope variable.
        stage: _ANSWER_STAGE or _COMBINE_STAGE.
        instruction: The stage's assembled instructions.
        llm: make_llm()'s callable, on the stage's model.
        max_turn: The stage's turn limit.
        deadline_turn: Over how many of the last turns the deadline note is shown.
        late_answer_on: Whether a stage that ran out of turns is asked once more.

    Returns:
        {"ok": whether an answer came back, "answer": the answer cut to
        DISPATCH_ANSWER_MAX_CHAR, or None, "error": why there is none, or "", "outcome"
        (COMPLETE_KEY when the stage ended itself, else None), "iters", "max_turn",
        "refused_completes" (how many COMPLETE() calls the guard sent back), "harvested"
        (whether the answer came from the one extra call rather than from COMPLETE()),
        "scope" and "history"}.
    """
    plan.reset_read_tool()
    rlm = RLM(llm=llm, tool_list=integrate.read_record_tool_list(plan.answer_tool_list()),
              instruction=instruction, stage=stage, max_turn=max_turn)
    state_note = plan.make_state_note()
    out_dict = run_stage(rlm, text,
                         scope_mod.stage_variable_dict(scope_dict, rlm.tool_dict),
                         scope_mod.SCOPE_VAR_DOC,
                         _LATE_ANSWER_NOTE, _LATE_ANSWER_CLOSE, late_on=late_answer_on,
                         late_state=state_note, text_answer=True,
                         turn_note=turn_note_chain(
                             state_note, make_deadline_note(deadline_turn, _DEADLINE_WHAT)),
                         on_complete=refuse_once(_over_the_limit))
    answer = out_dict["answer"]
    answer_text = str(answer).strip() if answer is not None else ""
    if answer_text:
        error = ""
        answer = cut(answer_text, config.DISPATCH_ANSWER_MAX_CHAR)
    else:
        answer = None
        why = no_answer_reason(out_dict["error"], max_turn, END_CALL_DICT[COMPLETE_KEY],
                               out_dict["outcome"])
        error = f"the {stage} stage handed nothing over: {why}"
        logger.warning(error)
    return {"ok": answer is not None, "answer": answer, "error": error,
            "outcome": out_dict["outcome"], "iters": out_dict["iters"],
            "max_turn": max_turn,
            "refused_completes": out_dict["refused_completes"],
            "harvested": out_dict["harvested"],
            "scope": scope_dict, "history": out_dict["history"]}
