"""Start the program automatically after an edit lands.

The check is the machine's, not the model's. After a turn that changes the files, this
module starts config.RUN_COMMAND with config.SMOKE_ARG against a sample of its own and
returns the note on the result for that turn's output (make_after_turn). It also
returns, once, the refusal of the call that would end the run while that check is
failing (make_complete_guard).

What it can say is "the program still runs", not "the change is right".

This module imports nothing from agent.py.
"""

import logging
from collections.abc import Callable

from codetwine_loom import config
from codetwine_loom.paths import list_text
from codetwine_loom.rlm import COMPLETE_KEY, END_CALL_DICT, cut_tail, fence_of
from codetwine_loom.tools import edit, run

logger = logging.getLogger(__name__)

# What one request's checks have done so far.
#   baseline    - whether the program ran before the model changed anything. None until
#                 check_baseline() has run; False disables the check for this request
#   tree_hash_dict - edit.own_change_hash_dict() as of the last check. A turn is checked
#                 only when the files differ from it, whatever changed them
#   last_ok     - the outcome of the most recent check, with last_why and last_output
#                 holding why it failed
#   refused     - whether COMPLETE() has already been sent back once
_state_dict: dict = {}

# The note that goes into the turn's output when the program ran. It ends with what the
# run wrote, compared against the run made before the first edit
_OK_NOTE = ("## Automatic check\n"
            "`{command}` was started for you after this edit and exited 0. That says "
            "the program still runs, not that your change is right.\n"
            "{produced}")

# The two forms of that comparison: the same files as before, or a list of the changes
_SAME_OUTPUT_NOTE = ("It wrote the same {count} files as it did before your first edit, so "
                     "nothing you have written has changed what it produces yet.")
_DIFF_OUTPUT_NOTE = "What it writes has changed since before your first edit: {changes}."

# What goes into the turn's output when it did not run, with the tail of its output. {fence}
# is rlm.fence_of() of that output
_FAIL_NOTE = ("## Automatic check - FAILED\n"
              "`{command}` was started for you after this edit and did not finish "
              "({why}). The program you are editing does not run as it stands. Fix "
              "this before you write anything else, and never call COMPLETE() while it "
              "is failing.\n"
              "{fence}\n{output}\n{fence}")

# What the call that ended the run is sent back with while the check is failing. It is
# refused at most once per request, {call} is the call the model actually made, and
# {fence} is rlm.fence_of() of the output
_COMPLETE_REFUSAL = ("{call}() was not accepted. The program does not run: `{command}` "
                     "was started after your last edit and failed ({why}). Fix it, or "
                     "call {call}() again saying why it cannot be fixed. This is asked "
                     "once only.\n{fence}\n{output}\n{fence}")


def reset_smoke_state() -> None:
    """Forget the previous request's checks.

    The state is module level, one for the process.
    """
    _state_dict.clear()
    _state_dict.update({"baseline": None, "baseline_produced": None, "checks": 0,
                        "failures": 0, "refusals": 0,
                        "tree_hash_dict": {}, "last_ok": None, "last_why": "",
                        "last_output": "", "refused": False})


def is_on() -> bool:
    """Return whether an automatic check is configured at all.

    Returns:
        True when both a launcher (RUN_COMMAND) and arguments (SMOKE_ARG) are set, and
        the run is not a dry one.
    """
    return bool(config.RUN_COMMAND and config.SMOKE_ARG and not config.EDIT_DRY_RUN)


def is_active() -> bool:
    """Return whether checks should run for this request.

    Returns:
        True when a check is configured and the baseline run passed. A project whose
        baseline already failed is not checked again for this request.
    """
    return is_on() and _state_dict.get("baseline") is True


def command_line() -> str:
    """Return the command the check starts.

    Returns:
        The launcher and the check's arguments joined by spaces.
    """
    return " ".join(list(config.RUN_COMMAND) + list(config.SMOKE_ARG))


def count_dict() -> dict:
    """Return the counters for one request.

    Returns:
        {"enabled": whether a check is configured,
         "baseline_ok": whether the program ran before any edit (None when not checked),
         "checks": how many checks ran, the baseline one included,
         "failures": how many of them failed,
         "refusals": how many COMPLETE() calls the guard sent back,
         "last_ok": the outcome of the most recent check,
         "last_why": why that check failed ("" when it did not)}.
    """
    return {"enabled": is_on(), "baseline_ok": _state_dict.get("baseline"),
            "checks": _state_dict.get("checks", 0), "failures": _state_dict.get("failures", 0),
            "refusals": _state_dict.get("refusals", 0), "last_ok": _state_dict.get("last_ok"),
            "last_why": "" if _state_dict.get("last_ok") else _state_dict.get("last_why", "")}


def _why(result_dict: dict) -> str:
    """Say in a few words why a run did not finish successfully.

    Args:
        result_dict: run.run_once()'s return value.

    Returns:
        The tool's "error" when it could not start at all, otherwise the exit code.
    """
    return result_dict["error"] or f"exit code {result_dict['returncode']}"


def _run_check() -> dict:
    """Start the program and record the outcome in _state_dict.

    It starts SMOKE_REPEAT times over one workspace, stopping at the first start that
    fails. The first start empties the workspace and the rest keep what the one before
    produced.

    Returns:
        run.run_once()'s return value for the start that decided the outcome, its
        "produced" holding the files every start up to that one wrote.
    """
    repeat = config.SMOKE_REPEAT
    produce_set: set[str] = set()
    for start in range(1, repeat + 1):
        result_dict = run.run_once(config.SMOKE_ARG, file_dict=config.SMOKE_FILE,
                                   env_dict=config.SMOKE_ENV, reset=start == 1)
        produce_set.update(result_dict["produced"])
        if not result_dict["ok"]:
            break
    result_dict["produced"] = sorted(produce_set)
    _state_dict["checks"] += 1
    _state_dict["last_ok"] = result_dict["ok"]
    _state_dict["last_why"] = (_why(result_dict) + (f", on start {start} of {repeat}"
                                                    if repeat > 1 else ""))
    _state_dict["last_output"] = cut_tail(result_dict["output"], config.SMOKE_OUTPUT_TAIL)
    if not result_dict["ok"]:
        _state_dict["failures"] += 1
    # The next check runs only once the files differ from what this one ran against
    _state_dict["tree_hash_dict"] = edit.own_change_hash_dict()
    return result_dict


def _is_tree_change() -> bool:
    """Say whether the files the request changed differ from what the last check ran on.

    Returns:
        True when edit.own_change_hash_dict() differs from the one the last check kept.
    """
    return edit.own_change_hash_dict() != _state_dict["tree_hash_dict"]


def check_baseline() -> bool:
    """Run the check once before the model starts, and keep it only if it passed.

    A baseline that fails turns the check off for the whole request: no later turn
    carries a failure that was there before the model started. Arguments (SMOKE_ARG)
    with no launcher to start them (RUN_COMMAND) are warned about, unless the run is a
    dry one (EDIT_DRY_RUN).

    Returns:
        True when the program ran and the check stays on for this request.
    """
    if not is_on():
        _state_dict["baseline"] = None
        if config.SMOKE_ARG and not config.RUN_COMMAND and not config.EDIT_DRY_RUN:
            logger.warning("the automatic check is off: SMOKE_ARG is set, but "
                           "RUN_COMMAND is not and this language has no default "
                           "launcher. Set RUN_COMMAND to what starts this project")
        return False
    result_dict = _run_check()
    _state_dict["baseline"] = result_dict["ok"]
    # What the program wrote before any edit; a later check is reported against it
    _state_dict["baseline_produced"] = list(result_dict["produced"]) if result_dict["ok"] else None
    if not result_dict["ok"]:
        logger.warning("the automatic check is off for this request: `%s` already "
                       "failed before any edit (%s). %s", command_line(),
                       _state_dict["last_why"], _state_dict["last_output"])
    else:
        logger.info("the automatic check is on: `%s` ran before any edit",
                    command_line())
    return result_dict["ok"]


def _output_line(output_list: list[str]) -> str:
    """Say how what the run wrote differs from what the baseline run wrote.

    Args:
        output_list: The files this check's run wrote (run.run_once()'s "produced").

    Returns:
        One line, or "" when there is no baseline to compare with.
    """
    base_list = _state_dict.get("baseline_produced")
    if base_list is None:
        return ""
    add_list, drop_list = (sorted(set(output_list) - set(base_list)),
                           sorted(set(base_list) - set(output_list)))
    if not add_list and not drop_list:
        return _SAME_OUTPUT_NOTE.format(count=len(base_list))
    change_list = []
    if add_list:
        change_list.append(f"{len(add_list)} new ({list_text(add_list)})")
    if drop_list:
        change_list.append(f"{len(drop_list)} no longer written ({list_text(drop_list)})")
    return _DIFF_OUTPUT_NOTE.format(changes=", ".join(change_list))


def _render(result_dict: dict) -> str:
    """Render one check's outcome as the note for the turn's output.

    Args:
        result_dict: The return value of the _run_check() that has just run. Its "ok"
            and "produced" are read here; the wording of a failure comes from what that
            call recorded.

    Returns:
        The note for a run that finished, or the failure with the tail of its output.
    """
    if result_dict["ok"]:
        return _OK_NOTE.format(command=command_line(),
                               produced=_output_line(result_dict["produced"])).strip()
    return _FAIL_NOTE.format(command=command_line(), why=_state_dict["last_why"],
                             output=_state_dict["last_output"],
                             fence=fence_of(_state_dict["last_output"]))


def make_after_turn() -> Callable[[int, str], str] | None:
    """Build the hook that checks a turn once it has run.

    It checks only turns that left the files this request changed different from what
    the last check ran against, whatever changed them; a turn that only read is skipped.

    Returns:
        after_turn(turn, output) -> str: the note for that turn's output, "" when
        nothing was checked. None when no check is configured.
    """
    if not is_on():
        return None

    def after_turn(turn: int, output: str) -> str:
        """Check a turn that changed the files, and return the note for its output."""
        if not is_active() or not _is_tree_change():
            return ""
        result_dict = _run_check()
        logger.info("automatic check after turn %d: %s", turn,
                    "ok" if result_dict["ok"] else _why(result_dict))
        return _render(result_dict)

    return after_turn


def make_complete_guard() -> Callable[[str], str] | None:
    """Build the guard that sends COMPLETE() back once while the program does not run.

    The check runs here first whenever the files differ from what the last check ran
    against, a change made on the COMPLETE turn itself among them. Whether the call is
    then sent back is decided afterwards: not with SMOKE_COMPLETE_GUARD off, and not a
    second time.

    Returns:
        on_complete(answer) -> str: "" accepts, text refuses. None when no check is
        configured.
    """
    if not is_on():
        return None

    def on_complete(answer: str) -> str:
        """Return "" to accept the call, or the refusal while the check is failing."""
        if not is_active():
            return ""
        if _is_tree_change():
            result_dict = _run_check()
            logger.info("automatic check on %s(): %s", END_CALL_DICT[COMPLETE_KEY],
                        "ok" if result_dict["ok"] else _why(result_dict))
        if not config.SMOKE_COMPLETE_GUARD or _state_dict["refused"] \
                or _state_dict["last_ok"] is not False:
            return ""
        _state_dict["refused"] = True
        _state_dict["refusals"] += 1
        logger.info("%s() refused once: `%s` is failing",
                    END_CALL_DICT[COMPLETE_KEY], command_line())
        # The failure is repeated in the refusal, even when it is already in the history
        return _COMPLETE_REFUSAL.format(call=END_CALL_DICT[COMPLETE_KEY], command=command_line(),
                                        why=_state_dict["last_why"],
                                        output=_state_dict["last_output"],
                                        fence=fence_of(_state_dict["last_output"]))

    return on_complete


reset_smoke_state()
