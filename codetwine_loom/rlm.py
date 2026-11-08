"""The RLM itself.

All it does is "keep the long input in REPL variables and let the model write code to
inspect it a piece at a time". No external libraries. The recursion depth is 1 (the
model writes code, that code runs — one level).
"""

import ast
import bisect
import contextlib
import functools
import inspect
import io
import json
import keyword
import logging
import os
import re
import selectors
import signal
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from codetwine_loom import config, progress, turnlog
from codetwine_loom.llm import PromptTooLong
# Re-exported: the callers of the loop read the marker and the cut through this module
from codetwine_loom.textcut import (CUT_HEAD_NOTE, CUT_NOTE, CUT_NOTE_RE,  # noqa: F401
                                    CUT_NOTE_SHAPE, cut, cut_tail, limit_text, one_line)

logger = logging.getLogger(__name__)

# A line that opens or closes a fenced block: up to three spaces, three or more backticks
# or tildes, then the rest of the line. After an opening fence the rest is its info
# string, whose first word is the block's label
_FENCE_LINE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$", re.M)

# The labels of a block that is run, in lower case. A block with no label is run only
# when the reply holds no block with one of these, and only when bare_code() takes its text
_PYTHON_LABEL_TUPLE = ("python", "py", "python3", "ipython")

# A run of backticks in a text rendered inside a fence
_BACKTICK_RUN = re.compile(r"`+")

# A call written in markup rather than in a fence: the element that names what is
# called, its closing tag, and one parameter's opening and closing tags
_CALL_OPEN = re.compile(r"<invoke\s+name=\"([^\"]+)\"\s*>")
_CALL_CLOSE = "</invoke>"
_PARAM_OPEN = re.compile(r"\s*<parameter\s+name=\"([^\"]+)\"\s*>")
# A whole tag at the start of a line, a closing tag at the end of one, and what is left of
# a line once its tags are taken away that is a label alone ("python") (_markup_trim)
_EDGE_OPEN_RE = re.compile(r"^\s*<(?:[\w.-]*:)?([\w.-]*)[^<>\n]*>")
_EDGE_CLOSE_RE = re.compile(r"</(?:[\w.-]*:)?([\w.-]*)[^<>\n]*>\s*$")
_EDGE_LABEL_RE = re.compile(r"[\w.-]*")
_PARAM_CLOSE = "</parameter>"
# The closing tag that ends the value of a parameter: _PARAM_CLOSE with nothing but
# whitespace between it and the next parameter, the call's closing tag or the end of the
# reply. _PARAM_SURE_CLOSE_RE leaves out the end of the reply, for a reply cut off at
# the output limit or still arriving
_PARAM_CLOSE_RE = re.compile(
    rf"{re.escape(_PARAM_CLOSE)}(?=\s*(?:<parameter\s|{re.escape(_CALL_CLOSE)}|\Z))")
_PARAM_SURE_CLOSE_RE = re.compile(
    rf"{re.escape(_PARAM_CLOSE)}(?=\s*(?:<parameter\s|{re.escape(_CALL_CLOSE)}))")

# The names of a parameter that holds code: the one a markup call named with one of
# _PYTHON_LABEL_TUPLE holds, and the ones a parameter written with no call around it has
_CODE_PARAM_TUPLE = ("code", *_PYTHON_LABEL_TUPLE)

# The start of a line up to its text: its indent and the marks of Markdown list items
# with the spaces after each
_LIST_MARK_RE = re.compile(r"[ \t]*(?:(?:[-*+]|\d{1,9}[.)])[ \t]+)*")
# Markdown that quotes a markup call written in a reply's text (_in_quote): the start of
# a block quote line, after the marks of list items in front of it if any, a blank line
# that ends a paragraph, the marks that open and close an HTML comment, and the columns
# of indent of a line of an indented code block
_QUOTE_LINE_RE = re.compile(_LIST_MARK_RE.pattern + ">")
_BLANK_LINE_RE = re.compile(r"\n[ \t\r]*(?=\n)")
_COMMENT_OPEN = "<!--"
_COMMENT_CLOSE = "-->"
_CODE_INDENT = 4


# Appended to a turn's output when the history was folded to fit HISTORY_MAX_CHAR. It
# says the output is gone rather than shortened
FOLD_NOTE = ("\n... (this turn's output was folded out of the history: {omitted} chars "
             "dropped. Read the file again if you still need it)")

# Appended to a turn's code when folding the outputs alone did not bring the history
# within the budget. It starts with "# " and stays a comment inside the ```python fence
_CODE_FOLD_NOTE = ("\n# ... (this turn's code was folded out of the history: {omitted} "
                   "chars dropped)")

# Placed in the history of a turn that returned no code block. Keep ``` out of it
NO_CODE_MARK = ("# The previous turn returned no code block. What follows is the "
                "beginning of that reply, which was NOT executed\n# ")

# Added to the history of a turn whose reply was cut off at the output limit. A reply
# cut mid-way has no closing fence and takes the "no code block" path. It names no one
# stage's work
CUT_REPLY_NOTE = ("Your reply hit the output limit and was cut off. Nothing you "
                  "had started writing was executed. Do less in one turn: write a "
                  "shorter block that takes one step, and take the next step in the "
                  "next turn")

# Appended to the end of the prompt only when a cut turn is retried. Never stored in
# the history
CUT_RETRY_NOTE = ("\n\nYour previous reply was cut off at the output limit and "
                  "was not executed. Write the same thing more briefly: one step in "
                  "this block, the rest in later turns")

# Appended to the prompt of a call that runs no code when its reply was cut off at the
# output limit and it is asked again. Never stored anywhere
BRIEF_RETRY_NOTE = ("\n\nYour previous reply was cut off at the output limit. Write the "
                    "same thing more briefly")

# Appended to a reply that was still cut off at the output limit after the retries: it
# says the reply is not the whole
CUT_END_NOTE = "\n... (the reply was cut off at the output limit)"

# Added to the history of a turn whose reply came back empty, which is what a reply that
# starts with TURN_MARK_RE comes back as
EMPTY_REPLY_NOTE = ("Your reply was empty. Starting a history heading (### Turn / "
                    "## Turn) stops generation right there. Start with the ```python "
                    "block instead")

# Appended to the end of the prompt only when an empty turn is retried. Never stored
# in the history
EMPTY_RETRY_NOTE = ("\n\nYour previous reply came back with not a single character. "
                    "Do not write a heading; start with the ```python block")

# Why no code block could be taken, and what to tell the model about it, one text per
# reason. {reason: (text stored in the history, text appended only when retrying)}.
# A reason whose retry text is empty is not retried
NO_CODE_NOTE_DICT: dict[str, tuple[str, str]] = {
    "cut": (CUT_REPLY_NOTE, CUT_RETRY_NOTE),
    "empty": (EMPTY_REPLY_NOTE, EMPTY_RETRY_NOTE),
    "prose": ("No code block. Write Python code and nothing else", ""),
}

# Added to the output of a turn whose fenced block does not compile
FENCE_LINE_NOTE = ("A line of nothing but three or more backticks ends the ```python block "
                   "there; when your code holds such a line (in a Markdown text, say), open "
                   "and close the block with more backticks than that line has, as in "
                   "````python")

# Appended after a turn's output for every tool call that came back with "ok" False
# and whose reason the turn did not print. {result} is the rest of the return value
TOOL_REFUSAL_NOTE = 'note: {tool}() returned "ok": False and it was not printed: {result}'

# A turn's whole output when reading what its block printed, or writing out how the block
# ended, raised. {error} is the name of the exception's type
OUTPUT_READ_ERROR_NOTE = "this turn's output could not be read: {error}"

# A turn's whole output when its block printed nothing, raised nothing and no tool
# refusal is left to note
NO_OUTPUT_NOTE = "this turn printed nothing. Nothing is shown unless you print it"

# Once the history is over HISTORY_MAX_CHAR, it is folded down to this share of it
_HISTORY_FOLD_SHARE = 0.5

# The headings the history is rendered with
TURN_TITLE = "### Turn"       # ### Turn j code / ### Turn j output
NEXT_TURN_TITLE = "## Turn"   # ## Turn i/n (the trailing line with the turns left)

# What a reply that copies the history format and starts the next turn itself writes: a
# line that starts with one of the two headings and a turn number. A reply is ended in
# front of the first
TURN_MARK_RE = re.compile(
    rf"\n(?:{re.escape(TURN_TITLE)}|{re.escape(NEXT_TURN_TITLE)}) \d")

# Reading the unfenced Python of a reply's text (_in_string_or_comment). What the
# reading of a logical line stops at: a quote that opens a string, a comment mark, a
# bracket, a backslash that continues the line, a newline and a backtick. A line that
# holds none of _CODE_LINE_MARK_RE is not read
_CODE_MARK_RE = re.compile(r"'''|\"\"\"|['\"#()\[\]{}`\n]|\\\n")
_CODE_LINE_MARK_RE = re.compile(r"['\"#()\[\]{}\\]")
# The Markdown marks of emphasis and strikethrough that, on both sides of quotes in
# three, make them no Python
_EMPHASIS_MARK_TUPLE = ("*", "~")
# A mark of _CODE_MARK_RE, a word (group 1) followed by spaces and a second word (group
# 2), or a character that is neither ASCII nor a word character (group 3). The last two
# are marks of prose that stop a logical line from being Python (_code_mark): two words
# when neither is one of _PAIR_SKIP_SET, a character when no name may hold it
_CODE_MARK_PAIR_RE = re.compile(
    _CODE_MARK_RE.pattern + r"|(?<!\w)(\w+)(?=[ \t]+(\w+))|([^\w\x00-\x7f])")
# A word after spaces, which right after a closing bracket is a mark of prose too when
# it is none of _PAIR_SKIP_SET
_WORD_AFTER_RE = re.compile(r"[ \t]*(\w+)")
# The keywords and the soft keywords
_PAIR_SKIP_SET = frozenset(keyword.kwlist) | frozenset(keyword.softkwlist) | {"type"}
# The prefix of an f-string or a t-string, right in front of its quote. The marks of
# prose (_code_mark) after such a string whose closing quote, as read, lies inside one of
# its replacement fields do not stop a logical line (_read_code_line)
_TEMPLATE_PREFIX_RE = re.compile(r"(?<!\w)(?:[rR]?[fFtT]|[fFtT][rR])\Z")
# The braces of an f-string or a t-string: doubled ones, which outside a replacement field
# stand for one brace, and single ones; and inside a field the quotes that open a string
# (quotes in three first), the brackets, the colon that starts a format spec and the hash
# that starts a comment
_FIELD_MARK_RE = re.compile(r"\{\{|\}\}|'''|\"\"\"|[{}'\"()\[\]:#]")
# The closing bracket of each opening bracket
_CLOSE_BRACKET_DICT = {"(": ")", "[": "]", "{": "}"}
# What ends a string opened with each quote: the same quote, past a backslash and the
# character after it, and for a quote of one character a newline, which leaves it unclosed
_STRING_END_RE_DICT = {
    quote: re.compile(rf"\\.|{quote}" + (r"|\n" if len(quote) == 1 else ""), re.S)
    for quote in ("'''", '"""', "'", '"')}
# What the text of an f-string or a t-string inside a replacement field is read for, for
# each quote that opens it: a backslash and the character after it other than a brace, the
# braces of _FIELD_MARK_RE, the same quote, and for a quote of one character a newline
_FSTRING_MARK_RE_DICT = {
    quote: re.compile(r"\\[^{}]|\{\{|\}\}|[{}]|" + quote + (r"|\n" if len(quote) == 1 else ""),
                      re.S)
    for quote in ("'''", '"""', "'", '"')}
# Where a statement may start inside a line: a name followed by a bracket, a dot, an
# annotation or an assignment
_CODE_START_RE = re.compile(
    r"(?<![\w.])[^\W\d]\w*(?=[ \t]*(?:[(.\[:]|=(?!=)|(?://|\*\*|>>|<<|[-+*/%&|^@])=))")
# How ast.parse begins its error for Python cut off after the heading of a block, and
# for Python cut off inside a statement: in a heading before its colon, and inside
# brackets after a starred item and in a conditional expression before its "else"
_NO_BLOCK_ERROR = "expected an indented block"
_HEAD_CUT_ERROR = "expected ':'"
_STAR_CUT_ERROR = "cannot use starred expression here"
_ELSE_CUT_ERROR = "expected 'else' after 'if' expression"
# What is put where Python was cut off in front of the "else" of a conditional expression
_ELSE_END = " else _"
# What is also tried in front of the closing brackets of Python cut off at a newline
# inside them: a name after an operator, and "in" after the targets of a "for"
_CUT_END_TUPLE = ("_", "in _")

# What ast.parse raises for a text that is not Python it reads
_PARSE_ERROR_TUPLE = (SyntaxError, ValueError, RecursionError, MemoryError, SystemError,
                      OverflowError)

# The per-run counters RLM.run() reports
RUN_COUNT_DICT = {"truncated_replies": 0, "truncated_lost": 0,
                  "empty_replies": 0, "bare_code_turns": 0, "refused_completes": 0}


# The one way a stage can end itself, and the name of the call that reaches it. The loop
# only reports that the call was made
COMPLETE_KEY = "complete"
END_CALL_DICT = {COMPLETE_KEY: "COMPLETE"}

# A harvested reply that is the call itself rather than its argument, in the form the
# parser cannot read: several paragraphs inside one plain "...", which spans lines and
# is not a Python string. The call name comes from END_CALL_DICT
_WRAP_RE = re.compile(
    r"^(?:" + "|".join(END_CALL_DICT.values()) + r")\(\s*(?P<q>\"\"\"|'''|\"|')"
    r"(?P<body>.*)(?P=q)\s*,?\s*\)$", re.S)


class CallFailure(Exception):
    """Signals that the model call itself did not happen.

    The loop ends on it as it does when the turns run out: whatever the run had already
    written stays where it is, and run() says in "error" why it stopped early.
    """


class CompleteCall(BaseException):
    """Signals that COMPLETE() was called. Only breaks out of the loop.

    It derives from BaseException: neither a try/except Exception in the code the model
    wrote nor the except Exception in Repl.run catches it.
    """

    def __init__(self, answer: str) -> None:
        """Hold what was handed over.

        Args:
            answer: The argument of the call.
        """
        self.answer = answer
        # What the turn printed before the call. The turn that ended the run keeps its
        # output like any other
        self.output = ""


# What a stage's instructions say about the REPL's own file access, as one bullet's text
REPL_REACH_NOTE = ("open() and subprocess do NOT reach the target repository (they run from a\n"
                   "  different working directory). Every read, write and new file goes "
                   "through a tool.")

# The phrase after "print() what you want to see" that states how much of a turn's
# output comes back, in the two wordings the instructions use
OUTPUT_LIMIT_NOTE = "; only the first {limit}\ncharacters come back"
OUTPUT_LIMIT_LINE_NOTE = ", only the first {limit} characters come back"


def output_limit_note(text: str) -> str:
    """Fill in a phrase that states RLM_MAX_OUTPUT_CHAR.

    Args:
        text: OUTPUT_LIMIT_NOTE, OUTPUT_LIMIT_LINE_NOTE or another phrase with "{limit}".

    Returns:
        The phrase with the limit filled in, or "" when RLM_MAX_OUTPUT_CHAR is 0 or
        less and a turn's output is not cut.
    """
    return limit_text(text, config.RLM_MAX_OUTPUT_CHAR)


def _read_python(read: Callable[[str], object], code: str) -> object:
    """Read a text as Python with ast.parse, showing no warning.

    Args:
        read: What reads it: ast.parse.
        code: The text.

    Returns:
        What read returns, or the exception it raised when that is one of
        _PARSE_ERROR_TUPLE.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return read(code)
        except _PARSE_ERROR_TUPLE as e:
            return e


def parse_tree(code: str) -> ast.Module | None:
    """Parse a text as Python.

    Args:
        code: The text.

    Returns:
        The tree ast.parse builds, or None for a text that is not Python: one with a
        syntax error, a NUL or a lone surrogate, and one nested or chained deeper, or
        longer, than the parser reads.
    """
    tree = _read_python(ast.parse, code)
    return None if isinstance(tree, BaseException) else tree


def tool_call_text(code: str, name_list: list[str]) -> str:
    """Name the tools a block of code calls, each with how many times.

    Examples:
        "a = read_file(p)\\nb = read_file(q)\\nrun_test()" -> "read_file x2, run_test"
        "x = 1 + 1"                                        -> ""

    Args:
        code: The block a turn is about to run.
        name_list: The names that count as tools. A call of any other name is left out.

    Returns:
        The names in the order each is first called, joined by ", ", or "" when the
        block calls none of them or cannot be parsed.
    """
    tree = parse_tree(code)
    if tree is None:
        return ""
    call_list = sorted(
        (node.lineno, node.col_offset, node.func.id) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in name_list)
    count_dict: dict[str, int] = {}
    for _, _, name in call_list:
        count_dict[name] = count_dict.get(name, 0) + 1
    return ", ".join(name if count == 1 else f"{name} x{count}"
                     for name, count in count_dict.items())


class TurnTimeout(BaseException):
    """Signals that a turn ran past config.TURN_TIMEOUT_SEC.

    It derives from BaseException. No except Exception on its way up, in the code the
    model wrote or in a tool, holds it back: it reaches the Repl.run whose block started
    the timer.
    """


# How many turn_deadline() blocks are open in this process. Only the outermost one sets
# the timer, and only that one's Repl.run catches TurnTimeout
_deadline_depth = 0

# How many calls of a tool the block being run is inside. A block run inside a tool
# counts from 0 again (Repl.run)
_tool_depth = 0

# How long a block is given once its limit ran out, or once a signal that stops the run
# came, before what holds it is stopped (seconds): the timer rings again every this
# long, and the processes the block started are killed this long after
_TURN_GRACE_SEC = 5.0

# The signals a run is stopped with
_STOP_SIGNAL_NAME_TUPLE = ("SIGINT", "SIGTERM", "SIGHUP")

# Where the system lists its processes, one directory per process id
_PROCESS_DIR = "/proc"


def _child_dict() -> dict[int, list[int]]:
    """Return the processes of the system by the process that started them.

    Returns:
        {process id: [the id of each process it is the parent of, ...]}, read from
        _PROCESS_DIR. Empty on a system without it.
    """
    child_dict: dict[int, list[int]] = {}
    try:
        name_list = os.listdir(_PROCESS_DIR)
    except OSError:
        return child_dict
    for name in name_list:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(_PROCESS_DIR, name, "stat"), "rb") as stat_file:
                # "<id> (<name>) <state> <parent id> ...", and the name may hold ") "
                parent = int(stat_file.read().rsplit(b") ", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        child_dict.setdefault(parent, []).append(int(name))
    return child_dict


def _kill_child(keep_pid_set: set[int]) -> None:
    """Kill the processes this one started, and every process under them.

    Args:
        keep_pid_set: The ids of the processes this one started that are left alone.
    """
    child_dict = _child_dict()
    pid_list = [pid for pid in child_dict.get(os.getpid(), []) if pid not in keep_pid_set]
    # The list grows while it is gone through: each process, then the ones it started
    for pid in pid_list:
        pid_list.extend(child_dict.get(pid, []))
    for pid in pid_list:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def _leave_wake_pipe() -> None:
    """Have the signals of a process forked inside a block written to no pipe.

    Runs in the child of every fork. The pipe the block's watch reads stays the parent's
    alone: a signal the child is sent is not taken for one the parent was.
    """
    if _deadline_depth:
        with contextlib.suppress(OSError, ValueError):
            signal.set_wakeup_fd(-1)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_leave_wake_pipe)


def _is_under(frame: object, top_frame: object) -> bool:
    """Say whether a frame was called, however far down, from another that still runs.

    Args:
        frame: The frame a thread is in, or None.
        top_frame: The frame looked for.

    Returns:
        True when top_frame is frame itself or one of the frames that called it.
    """
    while frame is not None:
        if frame is top_frame:
            return True
        frame = frame.f_back
    return False


class _TurnDeadline:
    """The limit turn_deadline() puts on a block, as a context manager.

    The block that sets the timer has it ring at config.TURN_TIMEOUT_SEC and every
    _TURN_GRACE_SEC after that, until the block is left. The first ring raises
    TurnTimeout wherever the main thread is. A later one finds the block still going:
    inside a tool it raises nothing, and elsewhere it raises TurnTimeout and has it
    raised once more at the next line the block runs (_chase), which an except clause
    that caught the first does not hold.

    A thread watches the block meanwhile (_watch): a block still going _TURN_GRACE_SEC
    after its limit, or after a signal of _STOP_SIGNAL_NAME_TUPLE came, has the
    processes it started killed, which ends a wait for one of them that no signal
    handler can end.

    The signals of _STOP_SIGNAL_NAME_TUPLE that have a handler go through _on_stop()
    while the block has its limit. Once one of them raised, a ring raises what it
    raised again in place of TurnTimeout. One that comes while the limit is being
    lifted is held and sent again once it is lifted.
    """

    def __init__(self) -> None:
        """Start with no timer set."""
        self.second = config.TURN_TIMEOUT_SEC
        self.is_own = False
        self.is_end = False
        self.ring_count = 0
        # Whether _chase() set a trace function, and the one that was set before it
        self.is_chase = False
        self.trace_before: object = None
        # The frame the block is in, and the processes this one had started before it
        self.frame: object = None
        self.child_before_set: set[int] = set()
        # The signals held back before the block
        self.mask_before: set = set()
        # {a signal that stops the run: the handler it had before the block}, the one
        # of them whose handler raised, and the ones held while the limit is lifted
        self.stop_handler_dict: dict[int, Callable] = {}
        self.stop_number: int | None = None
        self.late_stop_list: list[int] = []

    def __enter__(self) -> bool:
        """Set the timer and start the watch, unless this block has no limit of its own.

        Returns:
            Whether this block set the timer.
        """
        global _deadline_depth
        if self.second <= 0 or _deadline_depth or not hasattr(signal, "SIGALRM") \
                or threading.current_thread() is not threading.main_thread():
            return False
        self.is_own = True
        self.frame = sys._getframe(1)
        self.child_before_set = set(_child_dict().get(os.getpid(), []))
        # The timer's signal reaches this process however it was started, and one
        # that arrived before the block and was held back is not taken for the timer's
        self.handler_before = signal.signal(signal.SIGALRM, signal.SIG_IGN)
        self.mask_before = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
        signal.signal(signal.SIGALRM, self._on_alarm)
        for name in _STOP_SIGNAL_NAME_TUPLE:
            number = getattr(signal, name, None)
            if number is not None and callable(handler := signal.getsignal(number)):
                self.stop_handler_dict[number] = handler
                signal.signal(number, self._on_stop)
        # Every signal that arrives is written to the pipe the watch reads, whatever
        # the main thread is in the middle of
        self.wake_read_fd, self.wake_write_fd = os.pipe()
        os.set_blocking(self.wake_read_fd, False)
        os.set_blocking(self.wake_write_fd, False)
        self.wake_fd_before = signal.set_wakeup_fd(self.wake_write_fd,
                                                   warn_on_full_buffer=False)
        self.watch_thread = threading.Thread(target=self._watch, daemon=True)
        self.watch_thread.start()
        signal.setitimer(signal.ITIMER_REAL, self.second, _TURN_GRACE_SEC)
        _deadline_depth += 1
        return True

    def __exit__(self, *error_tuple: object) -> None:
        """Lift the limit again, however the block ended."""
        self.is_end = True
        if self.is_own:
            self._lift()

    def _lift(self) -> None:
        """Stop the timer, put the handlers back and end the watch.

        The signals held meanwhile are sent again (_put_stop_back).
        """
        global _deadline_depth
        self.is_end = True
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.handler_before)
            if signal.SIGALRM in self.mask_before:
                signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            if self.is_chase:
                sys.settrace(self.trace_before)
        finally:
            _deadline_depth = 0
            self.frame = None
            try:
                signal.set_wakeup_fd(self.wake_fd_before)
            finally:
                try:
                    # The byte wakes the watch, which reads is_end and returns
                    with contextlib.suppress(OSError):
                        os.write(self.wake_write_fd, b"\0")
                    self.watch_thread.join()
                    os.close(self.wake_write_fd)
                    os.close(self.wake_read_fd)
                finally:
                    self._put_stop_back()

    def _put_stop_back(self) -> None:
        """Give the signals that stop the run the handlers they had before the block.

        The ones held while the limit was being lifted are sent again. A signal whose
        handler is no longer _on_stop() is left as it is.
        """
        for number, handler in self.stop_handler_dict.items():
            if signal.getsignal(number) == self._on_stop:
                signal.signal(number, handler)
        late_stop_list, self.late_stop_list = self.late_stop_list, []
        for number in late_stop_list:
            signal.raise_signal(number)

    def _on_stop(self, signal_number: int, frame: object) -> None:
        """Hand a signal that stops the run to the handler it had before the block.

        While the limit is being lifted the signal is held instead (_put_stop_back).

        Args:
            signal_number: The signal.
            frame: The frame the main thread is in.

        Raises:
            BaseException: Whatever the handler raises, KeyboardInterrupt as a rule.
        """
        if self.is_end or (frame is not None
                           and frame.f_code is _TurnDeadline.__exit__.__code__
                           and frame.f_locals.get("self") is self):
            self.late_stop_list.append(signal_number)
            return
        try:
            self.stop_handler_dict[signal_number](signal_number, frame)
        except BaseException:
            self.stop_number = signal_number
            raise

    def ring(self, frame: object = None) -> None:
        """Raise what ends the block.

        That is what the signal that stops the run raised, when one came, and the
        block's TurnTimeout otherwise.

        Args:
            frame: The frame the main thread is in, handed to the signal's handler.

        Raises:
            TurnTimeout: Unless a signal that stops the run raised before.
            BaseException: What that signal's handler raises.
        """
        if self.stop_number is not None:
            self.stop_handler_dict[self.stop_number](self.stop_number, frame)
        second = self.second
        raise TurnTimeout(f"the turn did not finish inside TURN_TIMEOUT_SEC ({second}s)")

    def _on_alarm(self, signal_number: int, frame: object) -> None:
        """Raise the block's TurnTimeout when the timer goes off.

        Nothing is raised once the block is being left. A block that was left without
        its limit being lifted has it lifted here.

        Args:
            signal_number: SIGALRM.
            frame: The frame the main thread is in.

        Raises:
            TurnTimeout: At the first ring; at a later one unless the main thread is
                inside a tool.
            BaseException: In its place, what a signal that stops the run raised
                before (ring).
        """
        if self.is_end or frame.f_code is _TurnDeadline.__exit__.__code__:
            return
        if not _is_under(frame, self.frame):
            self._lift()
            return
        self.ring_count += 1
        if self.ring_count > 1:
            if _tool_depth:
                return
            self._chase(frame)
        self.ring(frame)

    def _chase(self, frame: object) -> None:
        """Have TurnTimeout raised at the next line the block runs (_trace).

        Args:
            frame: The frame the main thread is in. It and every frame between it and
                the block's own are traced.
        """
        if not self.is_chase:
            self.is_chase, self.trace_before = True, sys.gettrace()
        while frame is not None and frame is not self.frame:
            frame.f_trace = self._trace
            frame = frame.f_back
        sys.settrace(self._trace)

    def _trace(self, frame: object, event: str, arg: object) -> object:
        """Raise TurnTimeout at a line of the block, as the trace function _chase() sets.

        Args:
            frame: The frame the event is of.
            event: What sys.settrace() names the event.
            arg: The event's own value, which is not read.

        Returns:
            None for the call of a new frame, which is not traced; itself for an event
            of a traced frame nothing is raised at.

        Raises:
            TurnTimeout: At a line run while the block is going, outside every tool.
            BaseException: In its place, what a signal that stops the run raised
                before (ring).
        """
        if event == "call":
            return None
        if event == "line" and not self.is_end and not _tool_depth:
            self.ring(frame)
        return self._trace

    def _watch(self) -> None:
        """Kill what the block started once it is past its limit or a stop, and still going.

        Runs in a thread of its own until the block is left. It waits for
        config.TURN_TIMEOUT_SEC + _TURN_GRACE_SEC, or for _TURN_GRACE_SEC from a signal
        of _STOP_SIGNAL_NAME_TUPLE that arrived before then. A block the main thread is
        still in then, outside every tool, has the processes this one started since
        the block began killed with everything under them (_kill_child), and is looked
        at again every _TURN_GRACE_SEC.
        """
        stop_set = {getattr(signal, name) for name in _STOP_SIGNAL_NAME_TUPLE
                    if hasattr(signal, name)}
        main_ident = threading.main_thread().ident
        kill_time = time.monotonic() + self.second + _TURN_GRACE_SEC
        with selectors.DefaultSelector() as selector:
            selector.register(self.wake_read_fd, selectors.EVENT_READ)
            while not self.is_end:
                if selector.select(max(kill_time - time.monotonic(), 0)):
                    with contextlib.suppress(BlockingIOError):
                        if stop_set.intersection(os.read(self.wake_read_fd, 4096)):
                            kill_time = min(kill_time,
                                            time.monotonic() + _TURN_GRACE_SEC)
                    continue
                if not self.is_end and not _tool_depth and _is_under(
                        sys._current_frames().get(main_ident), self.frame):
                    _kill_child(self.child_before_set)
                kill_time = time.monotonic() + _TURN_GRACE_SEC


def turn_deadline() -> _TurnDeadline:
    """Raise TurnTimeout in the block when config.TURN_TIMEOUT_SEC runs out.

    The limit is a SIGALRM timer that rings again until the block is left, with a
    thread that kills the processes the block started when it is still going after
    that (_TurnDeadline). 0 leaves the block unlimited, and so do a platform without
    the timer and a call from a thread other than the main one. A block opened inside
    another, such as a turn of a loop a tool runs within a turn, leaves the outer
    block's timer as it is.

    Returns:
        A context manager whose "as" value says whether this block set the timer. The
        limit is lifted again however the block ended.
    """
    return _TurnDeadline()


# The descriptors of standard output and standard error
_OUTPUT_FD_TUPLE = (1, 2)

# How many bytes of what a block wrote to the descriptors of _OUTPUT_FD_TUPLE are read
# back into the turn's output
_FD_OUTPUT_MAX_BYTE = 1 << 20

# How many _FdOutput blocks have descriptors pointed at a pipe of theirs, and the
# descriptors the outermost of them pointed
_fd_output_depth = 0
_fd_output_fd_tuple: tuple[int, ...] = ()


def _flush_first_stream() -> None:
    """Flush the streams the process was started with (sys.__stdout__, sys.__stderr__).

    A stream that is not there, is closed or takes no more is left as it is.
    """
    for stream in (sys.__stdout__, sys.__stderr__):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.flush()


class _FdOutput:
    """Takes in what a block writes to the descriptors of _OUTPUT_FD_TUPLE.

    The descriptors are pointed at one pipe while the block runs (progress.PipeOutput)
    and back at what they were when it ends. What went to the pipe - the output of a
    process the block started without capturing it, and what the block wrote to
    sys.__stdout__, sys.__stderr__ or the descriptors themselves - is then in "text":
    its first _FD_OUTPUT_MAX_BYTE bytes read as config.CHILD_TEXT_DICT reads a child's
    output, with CUT_NOTE after them, stating how many bytes followed, when there were
    more. A process the block left running goes on writing to the pipe, and what it
    writes after the block is read and dropped.

    Only a descriptor this process writes its own lines past is pointed at the pipe
    (progress.apart_fd_list), and a block inside another points the ones that block
    did. Nothing is pointed anywhere on a thread other than the main one, nor when the
    pipe cannot be opened: "text" stays "".
    """

    def __init__(self) -> None:
        """Start with nothing taken in."""
        self.text = ""
        self.pipe: progress.PipeOutput | None = None
        # [(a descriptor of _OUTPUT_FD_TUPLE, the copy of what it was pointed at), ...]
        self.fd_copy_list: list[tuple[int, int]] = []

    def __enter__(self) -> "_FdOutput":
        """Point the descriptors at the pipe.

        Returns:
            Itself.
        """
        global _fd_output_depth, _fd_output_fd_tuple
        if threading.current_thread() is not threading.main_thread():
            return self
        apart_fd_list = progress.apart_fd_list()
        fd_tuple = _fd_output_fd_tuple if _fd_output_depth else tuple(
            fd for fd in _OUTPUT_FD_TUPLE if fd in apart_fd_list)
        if not fd_tuple:
            return self
        try:
            self.pipe = progress.PipeOutput(1, _FD_OUTPUT_MAX_BYTE)
        except OSError as e:
            logger.debug("what a block writes to its descriptors is not taken in (%s)", e)
            return self
        _flush_first_stream()
        for fd in fd_tuple:
            # A descriptor that is closed by now stays closed
            with contextlib.suppress(OSError):
                self.fd_copy_list.append((fd, progress.copy_fd(fd)))
                os.dup2(self.pipe.write_fd_list[0], fd)
        self.pipe.close_write()
        if not _fd_output_depth:
            _fd_output_fd_tuple = fd_tuple
        _fd_output_depth += 1
        return self

    def __exit__(self, *error_tuple: object) -> None:
        """Point the descriptors back and read what the pipe took in into "text"."""
        global _fd_output_depth, _fd_output_fd_tuple
        if self.pipe is None:
            return
        try:
            _flush_first_stream()
        finally:
            for fd, fd_copy in self.fd_copy_list:
                os.dup2(fd_copy, fd)
                os.close(fd_copy)
            self.fd_copy_list.clear()
            _fd_output_depth -= 1
            if not _fd_output_depth:
                _fd_output_fd_tuple = ()
            self.pipe.end(progress.PIPE_END_WAIT_SEC, is_drop_after=True)
            self.text = self.pipe.text(0, config.CHILD_TEXT_DICT["encoding"],
                                       config.CHILD_TEXT_DICT["errors"])


def _read_in_deadline(read: Callable[[], object]) -> tuple[object, BaseException | None]:
    """Call a function that reads what a turn's code made, under a turn_deadline().

    Args:
        read: The function, called with no argument.

    Returns:
        (what it returned, None), or (None, what it raised): any exception, SystemExit
        and CompleteCall included, and the TurnTimeout of the timer set here.

    Raises:
        TurnTimeout: When it is raised inside another block's turn_deadline().
        KeyboardInterrupt: As it came.
    """
    is_inner = _deadline_depth > 0
    try:
        with turn_deadline():
            return read(), None
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        if isinstance(e, TurnTimeout) and is_inner:
            raise
        return None, e


class Repl:
    """exec code and return what it printed. The namespace survives between calls.

    This is not a sandbox. The code the model writes runs in the same Python process and
    with the same privileges as this tool.
    """

    def __init__(self, variable_dict: dict[str, object],
                 tool_dict: dict[str, Callable]) -> None:
        """Build the REPL namespace.

        Args:
            variable_dict: The variables to place in the REPL (scope).
            tool_dict: The functions to place in the REPL, as {function name: function}.
        """
        # End the run with answer (CompleteCall); placed in the REPL as COMPLETE
        def complete(answer: str) -> None:
            raise CompleteCall(answer)

        self.refusal_list: list[tuple[str, dict]] = []
        self.namespace_dict = dict(variable_dict)
        self.namespace_dict.update({name: self._keep_refusal(name, tool)
                                    for name, tool in tool_dict.items()})
        self.namespace_dict[END_CALL_DICT[COMPLETE_KEY]] = complete

    def _keep_refusal(self, name: str, tool: Callable) -> Callable:
        """Wrap one tool; a dict it returns with "ok" False is kept with the tool's name.

        What is kept is a copy of the dict as the tool returned it.

        Args:
            name: The name the tool has in the namespace.
            tool: The tool function.

        Returns:
            The wrapped tool, with the original's name, docstring and signature.
        """
        @functools.wraps(tool)
        def keep(*args: object, **kwargs: object) -> object:
            """Call the tool, keeping a copy of a refusal it returns."""
            global _tool_depth
            _tool_depth += 1
            try:
                tool_result = tool(*args, **kwargs)
                if isinstance(tool_result, dict) and tool_result.get("ok") is False:
                    self.refusal_list.append((name, dict(tool_result)))
                return tool_result
            finally:
                _tool_depth -= 1
        return keep

    def _refusal_note_list(self, output: str) -> list[str]:
        """Return one note per kept refusal whose reason is not in output, and forget them.

        The reason is the "error" text, else the "output" text, else the whole return
        value as printed. A return value printed whole counts as printed.

        Args:
            output: What the turn printed, cut as it would be returned alone, and how
                it ended.

        Returns:
            The notes, uncut, in the order the refusals came; empty when every refusal
            was printed or none was kept.
        """
        note_list = []
        for name, refusal_dict in self.refusal_list:
            rest_dict = {key: value for key, value in refusal_dict.items() if key != "ok"}
            # What says why: the error text, else the program's output, else the whole
            reason = next((rest_dict[key] for key in ("error", "output")
                           if isinstance(rest_dict.get(key), str) and rest_dict[key]),
                          str(refusal_dict))
            if reason in output or str(refusal_dict) in output:
                continue
            result_text = json.dumps(rest_dict, ensure_ascii=False, default=str)
            note_list.append(TOOL_REFUSAL_NOTE.format(tool=name, result=result_text))
        self.refusal_list.clear()
        return note_list

    def _turn_output(self, print_text: str, end_text: str, max_char: int) -> str:
        """Join what a turn printed, how it ended and the refusal notes, within max_char.

        How it ended comes first: its lines share max_char (_share_budget). The printed
        text and the notes share what it leaves, a part shorter than its share kept
        whole.

        Args:
            print_text: Everything the block printed.
            end_text: The last traceback lines or the SystemExit note, each line ending
                with a newline; "" when the block ran to its end.
            max_char: The maximum number of characters of the whole, not counting the
                cut markers. 0 or less cuts nothing.

        Returns:
            The printed text, the end text after it, then one refusal note per line.
        """
        note_list = self._refusal_note_list(cut(print_text, max_char) + end_text)
        end_text = "".join(f"{line}\n" for line in
                           _share_budget(end_text.splitlines(), max_char))
        remain_char = max(max_char - len(end_text), 1) if max_char > 0 else 0
        print_text, *note_list = _share_budget([print_text, *note_list], remain_char)
        out_text = print_text + end_text
        if not note_list:
            return out_text
        lead = "" if not out_text or out_text.endswith("\n") else "\n"
        return out_text + lead + "\n".join(note_list)

    def run(self, code: str, max_char: int) -> str:
        """Run one block and return its output. Exceptions come back as text.

        The output is what the block wrote to standard output and standard error, in
        the order it wrote it, then what went to the two descriptors themselves
        (_FdOutput), such as the output of a process the block started. Whatever was
        printed before an exception is kept and comes back with it.

        A block that does not return is stopped after config.TURN_TIMEOUT_SEC and comes
        back as the turn's output like any other exception. SystemExit is answered the
        same way: the only way to end a run is the COMPLETE() call. So is whatever else
        the block raises that is no Exception, but for KeyboardInterrupt.

        A tool call that came back with "ok" False and whose reason the block did not
        print is noted after the output (TOOL_REFUSAL_NOTE).

        A turn whose output cannot be read back, such as one whose block closed
        standard output, returns OUTPUT_READ_ERROR_NOTE in its place (_read_output).

        Args:
            code: The Python code the model wrote.
            max_char: The maximum number of characters to return, not counting the cut
                markers.

        Returns:
            What was printed. On an exception, the output so far plus the last 3 lines
            of the traceback. The printed text, the traceback and the refusal notes
            share max_char (_turn_output).

        Raises:
            CompleteCall: When COMPLETE() was called. Its "output" holds what the turn
                printed before the call.
            TurnTimeout: When the timer that ran out is not the one this block set.
        """
        global _tool_depth
        self.refusal_list.clear()
        print_buffer, fd_output = io.StringIO(), _FdOutput()
        is_own_deadline = False
        depth_before = _tool_depth
        try:
            with fd_output, contextlib.redirect_stdout(print_buffer), \
                    contextlib.redirect_stderr(print_buffer), \
                    turn_deadline() as is_own_deadline:
                _tool_depth = 0
                try:
                    exec(code, self.namespace_dict)
                finally:
                    _tool_depth = depth_before
        except CompleteCall as call:
            call.output = self._read_output(print_buffer, fd_output, None, max_char)
            raise
        except TurnTimeout as e:
            if not is_own_deadline:
                raise
            return self._read_output(print_buffer, fd_output, e, max_char)
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            return self._read_output(print_buffer, fd_output, e, max_char)
        return self._read_output(print_buffer, fd_output, None, max_char) \
            or NO_OUTPUT_NOTE

    def _read_output(self, print_buffer: io.StringIO, fd_output: _FdOutput,
                     end: BaseException | None, max_char: int) -> str:
        """Put a turn's output together from what its block printed and how it ended.

        The printed text is read, with what went to the descriptors after it, the end is
        written out (_end_text) and the refusal notes are made (_turn_output) through
        _read_in_deadline(). Whatever any of the three raises, the timer running out
        included, makes OUTPUT_READ_ERROR_NOTE the turn's output.

        Args:
            print_buffer: Where the block's standard output and standard error went.
            fd_output: What took in the writes to the descriptors themselves.
            end: What the block raised, or None for a block that ran to its end or
                called COMPLETE().
            max_char: The maximum number of characters to return, not counting the cut
                markers.

        Returns:
            The turn's output as _turn_output() joins it, or OUTPUT_READ_ERROR_NOTE
            with the name of the type of what was raised.

        Raises:
            TurnTimeout: When it is raised inside another block's turn_deadline().
            KeyboardInterrupt: As it came.
        """
        out_text, error = _read_in_deadline(lambda: self._turn_output(
            print_buffer.getvalue() + fd_output.text, _end_text(end), max_char))
        if error is None:
            return out_text
        self.refusal_list.clear()
        return OUTPUT_READ_ERROR_NOTE.format(error=type(error).__name__)


def _end_text(end: BaseException | None) -> str:
    """Write out how a turn's block ended, as the last lines of the turn's output.

    Args:
        end: What the block raised. None for a block that ran to its end or called
            COMPLETE().

    Returns:
        "" for None; the note that SystemExit is not allowed for a SystemExit; else the
        last 3 lines of the traceback, a TurnTimeout written as the TimeoutError the
        turn is told of. Every line ends with a newline.
    """
    if end is None:
        return ""
    if isinstance(end, SystemExit):
        return (f"SystemExit({text_of(end.code)}) is not allowed here; call "
                f"{END_CALL_DICT[COMPLETE_KEY]}() to end the run\n")
    if isinstance(end, TurnTimeout):
        end = TimeoutError(*end.args).with_traceback(end.__traceback__)
    return "".join("".join(traceback.format_exception(end)).splitlines(keepends=True)[-3:])


def _share_budget(part_list: list[str], max_char: int) -> list[str]:
    """Cut texts to hold at most max_char characters together.

    The parts are taken from the shortest: each may keep up to an equal share of what
    the parts before it left, and a part shorter than its share stays whole.

    Examples:
        (["ab", "x" * 10, "y" * 10], 12) -> "ab" whole, and the first 5 characters of
        each of the other two followed by the cut marker

    Args:
        part_list: The texts.
        max_char: The characters they may hold together, not counting the cut markers.
            0 or less cuts nothing.

    Returns:
        The texts in the same order, each cut by cut(); an empty text stays empty.
    """
    if max_char <= 0:
        return list(part_list)
    limit_list = [0] * len(part_list)
    remain_char = max_char
    order_list = sorted(range(len(part_list)), key=lambda part_index: len(part_list[part_index]))
    for rank, index in enumerate(order_list):
        limit_list[index] = min(len(part_list[index]), remain_char // (len(order_list) - rank))
        remain_char -= limit_list[index]
    # cut() keeps a whole text for a limit of 0; a part with no room keeps one character
    return [cut(text, max(limit, 1)) if text else text
            for text, limit in zip(part_list, limit_list)]


def bare_code(reply: str) -> str | None:
    """Return the code of an unfenced reply when the whole reply is Python code.

    This covers a reply that is code but carries no ```python fence.

    Only text that parses (parse_tree) *and* contains at least one call is taken. Prose
    does not parse, and a single word such as "OK", which parses as an expression, holds
    no call. Text whose first line is one of _PYTHON_LABEL_TUPLE alone, what is left of
    an opening fence, is not taken.

    Args:
        reply: The model's reply.

    Returns:
        Code that may be executed, or None when nothing should be taken.
    """
    text = reply.strip()
    if not text or text.split("\n", 1)[0].strip() in _PYTHON_LABEL_TUPLE:
        return None
    tree = parse_tree(text)
    if tree is None or not any(isinstance(node, ast.Call) for node in ast.walk(tree)):
        return None
    return text


def _markup_trim(reply: str) -> str:
    """Return a reply without the markup around its code.

    From the first line, while it starts with "<", whole tags are taken away one by one,
    with the closing tags at its end of the same names as tags taken from that line, a
    namespace prefix aside, and the line goes when nothing but a label (_EDGE_LABEL_RE)
    or a tag that is not closed is left. From the last line, the closing tags at its end
    are taken away and a line starting with "<" goes, until a line with something else
    is left, after the fence lines (_FENCE_LINE_RE) at the end of the reply and the
    lines of a label of _PYTHON_LABEL_TUPLE alone at its start are taken away.

    Examples:
        "<invoke_tool>\nprint(run_test())" -> "print(run_test())"
        "<ns:parameter name=\"python\">x = f()\ny()</ns:parameter>\n</invoke>"
            -> "x = f()\ny()"

    Args:
        reply: The model's reply.

    Returns:
        What is left, "" when nothing is.
    """
    line_list = reply.strip().split("\n")
    while line_list and _FENCE_LINE_RE.fullmatch(line_list[-1]):
        line_list.pop()
    while line_list and line_list[0].strip() in _PYTHON_LABEL_TUPLE:
        line_list.pop(0)
    open_name_set: set[str] = set()
    while line_list and line_list[0].lstrip().startswith("<"):
        tag_match = _EDGE_OPEN_RE.match(line_list[0])
        rest = line_list[0][tag_match.end():] if tag_match else ""
        if tag_match:
            open_name_set.add(tag_match.group(1))
        while (close_match := _EDGE_CLOSE_RE.search(rest)) \
                and close_match.group(1) in open_name_set:
            rest = rest[:close_match.start()]
        if _EDGE_LABEL_RE.fullmatch(rest.strip()):
            line_list.pop(0)
            open_name_set.clear()
        else:
            line_list[0] = rest
    while line_list:
        line = line_list[-1]
        while _EDGE_CLOSE_RE.search(line):
            line = _EDGE_CLOSE_RE.sub("", line)
        if line.strip() and not line.lstrip().startswith("<"):
            line_list[-1] = line
            break
        line_list.pop()
    return "\n".join(line_list).strip()


def _trim_code(reply: str) -> str | None:
    """Return the code of an unfenced reply once the markup around it is taken away.

    The reply less its markup (_markup_trim) is taken when it is code (bare_code). Else
    that text is cut at its first line after the first that starts with "<", and what
    comes before the cut, less the markup around it, is taken when it is code. What
    follows the cut (a second call, a result written out) is left out.

    Args:
        reply: The model's reply.

    Returns:
        The code, or None when neither is code.
    """
    text = _markup_trim(reply)
    code = bare_code(text)
    if code:
        return code
    line_list = text.split("\n")
    cut = next((at for at, line in enumerate(line_list)
                if at and line.lstrip().startswith("<")), None)
    return bare_code(_markup_trim("\n".join(line_list[:cut]))) if cut else None


def _no_code_reason(reply: str, is_cut: bool) -> str:
    """Return why no code block could be taken, as a key of NO_CODE_NOTE_DICT.

    Args:
        reply: The model's reply.
        is_cut: Whether that reply was cut off at the output limit.

    Returns:
        "cut" (hit the limit) / "empty" (not a single character came back) /
        "prose" (only prose came back).
    """
    if is_cut:
        return "cut"
    return "empty" if not reply.strip() else "prose"


def _parse_ok(code: str) -> bool:
    """Say whether a text is Python that compiles.

    Args:
        code: The text between an opening fence and one closing fence.

    Returns:
        True when parse_tree() reads it.
    """
    return parse_tree(code) is not None


@dataclass(frozen=True)
class _Block:
    """One fenced block of a reply or one parameter of a markup call in it (_block_iter)."""

    # A fenced block's label: the first word of its info string in lower case, "" for
    # none. A parameter's name as it is written
    label: str
    text: str         # a block's text without its fences and the indent of its opening
                      # fence, or a parameter's value
    start: int        # where the text starts in the reply
    stop: int         # where it stops: its closing fence or tag, or the end of the reply
    is_closed: bool   # whether a closing fence or tag closes it
    # Whether a fenced block's text holds a line that opens a fence of its kind no
    # shorter than its own, or a fence longer than its own closes it
    is_nested: bool
    is_fence: bool    # True for a fenced block, False for a parameter
    # The _CALL_OPEN of the call a parameter belongs to; None for a fenced block and for
    # a parameter that starts the reply
    call: re.Match | None = None
    info: str = ""    # a fenced block's info string without the whitespace around it


def _block_iter(reply: str, is_cut: bool = False) -> Iterator[_Block]:
    """Yield the fenced blocks of a reply and the parameters of its markup calls, in order.

    A block opens at a line of _FENCE_LINE_RE, after backticks one whose info string
    holds no backtick. A line of _FENCE_LINE_RE that has the same character as the
    opening fence, at least as many of it, and nothing but whitespace after them closes
    it. Outside a block, a markup call (_CALL_OPEN, or a parameter that starts the
    reply) that comes before the next fence line gives its parameters
    (_param_block_iter), and no fence line inside them opens a block. A block or a
    parameter nothing closes runs to the end of the reply and is the last one yielded.

    Args:
        reply: The model's reply.
        is_cut: As for _param_block_iter().

    Yields:
        Each block and parameter.
    """
    open_match = None
    is_nested = False
    at = 0
    line_match = _FENCE_LINE_RE.search(reply)
    while True:
        if line_match is not None and line_match.start() < at:
            line_match = _FENCE_LINE_RE.search(reply, at)
        if open_match is None:
            start = _call_start(reply, at, line_match.start() if line_match else len(reply))
            if start is not None:
                at, call_match = start
                for block in _param_block_iter(reply, at, is_cut, call_match):
                    yield block
                    if not block.is_closed:
                        return
                    at = block.stop + len(_PARAM_CLOSE)
                continue
        if line_match is None:
            break
        at = line_match.end()
        fence, info = line_match.group(2), line_match.group(3)
        if open_match is None:
            if fence[0] != "`" or "`" not in info:
                open_match, is_nested = line_match, False
            continue
        open_fence = open_match.group(2)
        if fence[0] != open_fence[0] or len(fence) < len(open_fence):
            continue
        if info.strip():
            is_nested = True
            continue
        yield _fence_block(reply, open_match, line_match,
                           is_nested or len(fence) > len(open_fence))
        open_match = None
    if open_match is not None:
        yield _fence_block(reply, open_match, None, is_nested)


def _call_start(reply: str, at: int, end: int) -> tuple[int, re.Match | None] | None:
    """Return where the parameters of the first markup call between two positions start.

    Args:
        reply: The model's reply.
        at: Where the search starts.
        end: Where it ends.

    Returns:
        (the end of the first _CALL_OPEN, its match), or (0, None) when at is 0 and the
        reply starts with a _PARAM_OPEN; None when there is neither.
    """
    if at == 0 and _PARAM_OPEN.match(reply, 0, end):
        return 0, None
    call_match = _CALL_OPEN.search(reply, at, end)
    return (call_match.end(), call_match) if call_match else None


def _fence_block(reply: str, open_match: re.Match, close_match: re.Match | None,
                 is_nested: bool) -> _Block:
    """Build one fenced block from its opening fence and its closing fence.

    Args:
        reply: The model's reply.
        open_match: The opening fence's line (_FENCE_LINE_RE).
        close_match: The closing fence's line, or None when the block runs to the end
            of the reply.
        is_nested: As _Block.is_nested.

    Returns:
        The block.
    """
    start = min(open_match.end() + 1, len(reply))
    stop = close_match.start() if close_match else len(reply)
    text = reply[start:stop]
    indent = len(open_match.group(1))
    if indent:
        text = re.sub(rf"(?m)^ {{1,{indent}}}", "", text)
    word_list = open_match.group(3).split()
    return _Block(word_list[0].lower() if word_list else "", text, start, stop,
                  close_match is not None, is_nested, True, info=open_match.group(3).strip())


def _param_block_iter(reply: str, at: int, is_cut: bool = False,
                      call_match: re.Match | None = None) -> Iterator[_Block]:
    """Yield the parameters written one after another from a position of a reply.

    The value of a parameter ends at its closing tag (_PARAM_CLOSE_RE).

    Args:
        reply: The model's reply.
        at: Where the first parameter's opening tag is looked for.
        is_cut: Whether the reply was cut off at the output limit or is still arriving,
            where a closing tag at its end may not end a value (_PARAM_SURE_CLOSE_RE).
        call_match: The _CALL_OPEN of the call the parameters belong to (_Block.call).

    Yields:
        Each parameter, up to the first text that is not a parameter. One nothing
        closes is the last.
    """
    close_re = _PARAM_SURE_CLOSE_RE if is_cut else _PARAM_CLOSE_RE
    while (open_match := _PARAM_OPEN.match(reply, at)) is not None:
        close_match = close_re.search(reply, open_match.end())
        stop = close_match.start() if close_match else len(reply)
        yield _Block(open_match.group(1), reply[open_match.end():stop], open_match.end(),
                     stop, close_match is not None, False, False, call_match)
        if close_match is None:
            return
        at = close_match.end()


def _run_part_list(reply: str, part_list: list[_Block], is_cut: bool) -> list[_Block] | None:
    """Return the parts of a reply that the code to run is chosen from.

    The parts are those _block_iter() yields, in order. The list ends after a block
    labelled with one of _PYTHON_LABEL_TUPLE that does not compile, in front of a fence
    line with nothing after it (_is_stray_fence), and, in a reply cut off at the output
    limit, in front of a block that no closing fence closes.

    Examples:
        "```python\\nprint(1)\\n```"                          -> the block
        "```python\\nprint(1\\n```\\n```python\\nprint(2)\\n```"   -> the first block
        "```python\\nprint(1)\\n```\\n```"                    -> the block
        "```python\\nx()\\n```\\n```md\\n```python\\ny\\n```\\n```" -> None

    Args:
        reply: The model's reply.
        part_list: The parts of the reply (_block_iter).
        is_cut: Whether the reply was cut off at the output limit.

    Returns:
        The parts, or None when the reply cannot be split into them: it holds a block
        labelled with none of _PYTHON_LABEL_TUPLE that is nested (_Block.is_nested), a
        parameter that nothing closes, or, when it was not cut off, any other block that
        no closing fence closes.
    """
    run_list = []
    for part in part_list:
        is_python = part.is_fence and part.label in _PYTHON_LABEL_TUPLE
        if not part.is_closed:
            if part.is_fence and (is_cut or _is_stray_fence(reply, part, run_list)):
                break
            return None
        if part.is_nested and not is_python:
            return None
        run_list.append(part)
        if is_python and not _parse_ok(part.text):
            break
    return run_list


def _is_stray_fence(reply: str, part: _Block, run_list: list[_Block]) -> bool:
    """Say whether a fenced block nothing closes is a fence line with nothing after it.

    Args:
        reply: The model's reply.
        part: The block, which runs to the end of the reply.
        run_list: The parts in front of it (_run_part_list).

    Returns:
        True when the block holds nothing but whitespace and a fenced block comes before
        it, and its opening fence either has an info string of more than one word (a line
        of text), or has none and follows the closing fence of the block in front of it
        with nothing but blank lines between, and no block with a label comes after a
        block labelled with none of _PYTHON_LABEL_TUPLE.
    """
    fence_list = [run_part for run_part in run_list if run_part.is_fence]
    if part.text.strip() or not fence_list:
        return False
    if len(part.info.split()) > 1:
        return True
    if part.info or run_list[-1] is not fence_list[-1]:
        return False
    open_end = part.start - 1 if reply.endswith("\n", 0, part.start) else part.start
    open_at = reply.rfind("\n", 0, open_end) + 1
    if reply[fence_list[-1].stop:open_at].partition("\n")[2].strip():
        return False
    other_at = next((index for index, block in enumerate(fence_list)
                     if block.label not in _PYTHON_LABEL_TUPLE), len(fence_list))
    return not any(block.label for block in fence_list[other_at + 1:])


def _param_list(reply: str, at: int, is_cut: bool = False) -> list[tuple[str, str, int]]:
    """Return the closed parameters written one after another from a position of a reply.

    Args:
        reply: The model's reply.
        at: Where the first parameter's opening tag is looked for.
        is_cut: As for _param_block_iter().

    Returns:
        [(name, value, where the parameter ends), ...], up to the first text that is
        not a closed parameter.
    """
    param_list = []
    for block in _param_block_iter(reply, at, is_cut):
        if not block.is_closed:
            break
        param_list.append((block.label, block.text, block.stop + len(_PARAM_CLOSE)))
    return param_list


def _call_param_list(part_list: list[_Block], call_at: int | None) -> list[_Block]:
    """Return the closed parameters of one markup call of a reply.

    Args:
        part_list: The parts of the reply (_block_iter).
        call_at: Where the call's _CALL_OPEN starts, or None for the parameters that
            start the reply.

    Returns:
        The call's parameters in order, up to the first one that nothing closes.
    """
    param_list = []
    for part in part_list:
        if part.is_fence or (part.call.start() if part.call else None) != call_at:
            continue
        if not part.is_closed:
            break
        param_list.append(part)
    return param_list


def _call_code(reply: str, call_match: re.Match | None, param_list: list[_Block],
               is_python_only: bool) -> str | None:
    """Return the code of one call a reply writes in markup.

    A call named with one of _PYTHON_LABEL_TUPLE holds its code in a parameter named
    with one of _CODE_PARAM_TUPLE, and so does a reply whose first parameter has such a
    name and no call around it. A call with any other name is a call of the function of
    that name, its parameters the keyword arguments, and is turned into code that prints
    what the function returns. A parameter's value is read as JSON, and as a string when
    it is not JSON.

    Examples:
        '<invoke name="python">\\n<parameter name="code">print(1)</parameter>\\n</invoke>'
            -> "print(1)"
        '<parameter name="python">print(1)</parameter>'
            -> "print(1)"
        '<invoke name="f">\\n<parameter name="a">[1]</parameter>\\n'
        '<parameter name="b">x</parameter>\\n</invoke>'
            -> "print(f(a=[1], b='x'))"

    Args:
        reply: The model's reply.
        call_match: The call's _CALL_OPEN, or None for the parameters that start the
            reply.
        param_list: The call's closed parameters (_call_param_list).
        is_python_only: Whether only a call named with one of _PYTHON_LABEL_TUPLE is
            taken.

    Returns:
        The code, or None when the call holds none: no parameter holding code, a call to
        a function with no parameter or left open, or a name that is not an identifier.
    """
    if call_match is None:
        return next((part.text for part in param_list[:1]
                     if part.label.lower() in _CODE_PARAM_TUPLE), None)
    name = call_match.group(1)
    if name.lower() in _PYTHON_LABEL_TUPLE:
        return next((part.text for part in param_list
                     if part.label.lower() in _CODE_PARAM_TUPLE), None)
    if is_python_only or not name.isidentifier() or not param_list:
        return None
    if not reply[param_list[-1].stop + len(_PARAM_CLOSE):].lstrip().startswith(_CALL_CLOSE):
        return None
    argument_list = []
    for part in param_list:
        if not part.label.isidentifier():
            return None
        try:
            argument_list.append(f"{part.label}={json.loads(part.text)!r}")
        except (ValueError, RecursionError):
            argument_list.append(f"{part.label}={part.text!r}")
    return f"print({name}({', '.join(argument_list)}))"


def _stray_close(reply: str, at: int, part_list: list[_Block], is_cut: bool) -> bool:
    """Say whether a closing tag of a parameter follows a position outside every part.

    Args:
        reply: The model's reply.
        at: Where to look from.
        part_list: The parts of the reply (_block_iter).
        is_cut: As for _param_block_iter().

    Returns:
        True when a closing tag (_PARAM_CLOSE_RE) after the position is neither inside a
        part nor the closing tag of a parameter.
    """
    close_re = _PARAM_SURE_CLOSE_RE if is_cut else _PARAM_CLOSE_RE
    part_iter = iter(part_list)
    part = next(part_iter, None)
    for close_match in close_re.finditer(reply, at):
        while part is not None and part.stop < close_match.start():
            part = next(part_iter, None)
        if part is None or close_match.start() < part.start:
            return True
    return False


def _prose_call_code(reply: str, part_list: list[_Block], is_cut: bool) -> str | None:
    """Return the code of the first markup call a reply writes outside its fenced blocks.

    The call is the first one _block_iter() gives a parameter for. It is not taken when
    the text between the part in front of it and the call quotes the call (_in_quote),
    and when a closing tag of a parameter follows it outside the parts (_stray_close).

    Args:
        reply: The model's reply.
        part_list: The parts of the reply (_block_iter).
        is_cut: Whether the reply was cut off at the output limit.

    Returns:
        The code (_call_code), or None.
    """
    index = next((part_index for part_index, part in enumerate(part_list)
                  if not part.is_fence), None)
    if index is None:
        return None
    call_match = part_list[index].call
    start = call_match.start() if call_match else 0
    front_at = 0
    if index:
        before = part_list[index - 1]
        front_at = reply.find("\n", before.stop) + 1 or len(reply)
    if _in_quote(reply[front_at:start]) or _stray_close(reply, start, part_list, is_cut):
        return None
    return _call_code(reply, call_match, _call_param_list(
        part_list, call_match.start() if call_match else None), is_python_only=False)


@dataclass(frozen=True)
class _CodeRead:
    """How far one logical line of Python read from a text runs (_read_code_line)."""

    # "line": a newline outside every bracket ends it. "code", "string", "comment": the
    # text ends in its code, inside a string, or inside a comment. "bad": it is no Python
    kind: str
    # Where it stops: after the newline that ends it for "line"; where its code stops for
    # "code", "string" and "comment": the end of the text, the quote that opens the
    # string, the comment mark; the start of the line where it stops being Python for "bad"
    stop: int
    close_text: str = ""   # what closes the string and the brackets it leaves open
    # For a "bad" read that stops where the code read so far does not parse, where that
    # code is cut off: after the newline, or at the quotes in three. None for other reads
    parse_at: int | None = None


def _in_string_or_comment(text: str) -> bool:
    """Say whether a text ends inside a string or comment of the unfenced Python it holds.

    The text is read line by line from its start. A line that no logical line read
    before runs over may start a logical line of Python (_line_code_read), and the lines
    that logical line runs over are read with it. Of a line that starts none, no quote
    and no comment mark is read.

    Args:
        text: The text up to a position of a reply.

    Returns:
        True when the logical line that reaches the end of the text reads as Python and
        ends inside a string or a comment. A NUL and a lone surrogate are read as "?".
    """
    text = text.encode("utf-8", "replace").decode("utf-8").replace("\0", "?")
    text = text.replace("\r\n", "\n")
    line_at = 0
    while line_at < len(text):
        code_read = _line_code_read(text, line_at)
        if code_read.kind not in ("line", "bad"):
            return code_read.kind != "code"
        line_at = code_read.stop
    return False


def _line_code_read(text: str, line_at: int) -> _CodeRead:
    """Read the logical line of Python that one line of a text starts, if it starts one.

    The reads (_read_code_line) from the places in the line where a logical line may
    start (_code_start_iter) are tried in order, and the first that reads as Python
    (_python_error_at) is taken. A "bad" read and a read that ends with the line itself
    are set aside; the latter, from the first place, is only checked when a later place
    gives a read that runs past the line. A later place that a read set aside with no
    parse stop (parse_at) passes in its code outside every bracket is not read, as its
    read is set aside too. Nor is a later place in the comment that a read set aside or
    not taken meets in the line, when it starts the same text as the read does up to
    and including the comment mark (_same_comment_start_iter).

    Args:
        text: The text.
        line_at: Where the line starts.

    Returns:
        The read taken. Else a "bad" read whose stop is where to read on from: the next
        line, or the line where the read from the first place stops being Python when
        that is further on.
    """
    line_end = text.find("\n", line_at)
    line_end = len(text) if line_end < 0 else line_end
    skip_at = line_end + 1
    if not _CODE_LINE_MARK_RE.search(text, line_at, line_end):
        return _CodeRead("bad", skip_at)
    line_read = None   # (place, read) of the first place when its read ends with the line
    start_list = list(_code_start_iter(text, line_at, line_end))
    aside_set: set[int] = set()   # the later places whose reads are known to be set aside
    for index, start in enumerate(start_list):
        if start in aside_set:
            continue
        top_list: list[tuple[int, int]] = []
        comment_list: list[int] = []
        code_read = _read_code_line(text, start, top_list, comment_list)
        if code_read.kind == "bad" or (code_read.kind == "line"
                                       and code_read.stop == line_end + 1):
            if not index:
                line_read = (start, code_read) if code_read.kind == "line" else None
                skip_at = max(skip_at, code_read.stop)
            if code_read.parse_at is None:
                aside_set.update(_start_in_range_iter(start_list, index + 1, top_list))
            aside_set.update(_same_comment_start_iter(
                text, start_list, index, code_read, comment_list, line_end))
            continue
        if line_read is not None and _python_error_at(text, *line_read) is None:
            return line_read[1]
        line_read = None
        error_at = _python_error_at(text, start, code_read)
        if error_at is None:
            return code_read
        if not index:
            skip_at = max(skip_at, error_at)
        aside_set.update(_same_comment_start_iter(
            text, start_list, index, code_read, comment_list, line_end))
    return _CodeRead("bad", skip_at)


def _code_start_iter(text: str, line_at: int, line_end: int) -> Iterator[int]:
    """Yield where a logical line of Python may start in one line of a text, in order.

    Args:
        text: The text.
        line_at: Where the line starts.
        line_end: Where it ends: at its newline, or at the end of the text.

    Yields:
        Its first character after its indent and its list item marks (_LIST_MARK_RE),
        then each place after it where _CODE_START_RE matches outside an inline code
        span (_in_code_span).
    """
    first = _LIST_MARK_RE.match(text, line_at, line_end).end()
    if first == line_end:
        return
    yield first
    run_iter = _BACKTICK_RUN.finditer(text, line_at, line_end)
    run_match = next(run_iter, None)
    open_len = 0
    for start_match in _CODE_START_RE.finditer(text, first + 1, line_end):
        while run_match is not None and run_match.end() <= start_match.start():
            open_len = _span_open_len(open_len, len(run_match.group()))
            run_match = next(run_iter, None)
        if not open_len:
            yield start_match.start()


def _start_in_range_iter(start_list: list[int], index: int,
                         range_list: list[tuple[int, int]]) -> Iterator[int]:
    """Yield the places of a list from an index on that lie in one of a list of ranges.

    Args:
        start_list: The places, in order.
        index: The index of start_list to begin at.
        range_list: (from, to) ranges, in order and apart.

    Yields:
        Each place p with from <= p < to for one of the ranges, in order.
    """
    for range_from, range_to in range_list:
        index = bisect.bisect_left(start_list, range_from, index)
        while index < len(start_list) and start_list[index] < range_to:
            yield start_list[index]
            index += 1


def _same_comment_start_iter(text: str, start_list: list[int], index: int,
                             code_read: _CodeRead, comment_list: list[int],
                             line_end: int) -> Iterator[int]:
    """Yield the later places of a line that start the same text as one up to its comment.

    Args:
        text: The text.
        start_list: The places of the line, in order (_code_start_iter).
        index: The index of the place in start_list.
        code_read: The read from the place (_read_code_line).
        comment_list: The comment marks that read meets, in order (_read_code_line).
        line_end: Where the line ends: at its newline, or at the end of the text.

    Yields:
        Each place of start_list after the first comment mark, where the text from the
        place up to and including the comment mark follows, when that mark lies in the
        line, the comment holds no carriage return, and the read either runs on past the
        line with no parse stop or, with no quotes in three in front of the mark, stops
        where the code up to the line's newline does not parse.
    """
    if not comment_list or comment_list[0] >= line_end:
        return
    lead = text[start_list[index]:comment_list[0] + 1]
    if code_read.parse_at is None:
        is_same = code_read.stop > line_end + 1 or (code_read.stop > line_end
                                                    and code_read.kind != "line")
    else:
        is_same = code_read.parse_at == line_end + 1 \
            and "'''" not in lead and '"""' not in lead
    if not is_same or text.find("\r", comment_list[0], line_end) >= 0:
        return
    at = text.find(lead, comment_list[0] + 1, line_end)
    while at >= 0:
        index = bisect.bisect_left(start_list, at, index)
        if index < len(start_list) and start_list[index] == at:
            yield at
        at = text.find(lead, at + 1, line_end)


def _code_mark(text: str, at: int, is_prose_stop: bool) -> re.Match | None:
    """Return the next mark of _CODE_MARK_RE, or of prose, from a position.

    Args:
        text: The text.
        at: Where to search from.
        is_prose_stop: False to return marks of _CODE_MARK_RE only.

    Returns:
        The first match of _CODE_MARK_PAIR_RE at or after the position that has no group
        (a mark of _CODE_MARK_RE) or, for is_prose_stop True, whose two words are
        neither of _PAIR_SKIP_SET (group 1 is the first word) or whose character no name
        may hold (group 3), or None when there is none.
    """
    while True:
        mark = _CODE_MARK_PAIR_RE.search(text, at)
        if mark is None or mark.lastindex is None:
            return mark
        if mark.group(3) is not None:
            if is_prose_stop and not ("_" + mark.group(3)).isidentifier():
                return mark
            at = mark.end()
        elif is_prose_stop and not {mark.group(1), mark.group(2)} & _PAIR_SKIP_SET:
            return mark
        else:
            at = mark.end(1)


def _read_code_line(text: str, start: int,
                    top_list: list[tuple[int, int]] | None = None,
                    comment_list: list[int] | None = None) -> _CodeRead:
    """Read one logical line of Python from a position of a text.

    Only strings, comments, brackets and the backslash that continues a line are read.
    The line stops being Python at a backtick outside strings and comments, at a closing
    bracket that closes no opening one, at a newline inside a string in one quote, at
    quotes in three with the same one of _EMPHASIS_MARK_TUPLE right in front and right
    after, and at the marks of prose outside strings and comments in front of the first
    string with a prefix of _TEMPLATE_PREFIX_RE whose closing quote, as read, lies
    inside one of its replacement fields (_is_in_field): two words in a row or a
    character that no name may hold (_code_mark), and a word right after a closing
    bracket (_WORD_AFTER_RE). The code read so far is also checked, closed where it
    stops (_python_error_line), at the first quotes that open a string in three and at
    the first newline inside brackets (with each of _CUT_END_TUPLE before the closing
    brackets as well), and again at either once the code has grown to twice the length
    checked last.

    Args:
        text: The text.
        start: Where the logical line starts.
        top_list: A list that gets the (from, to) ranges, in order, where the read runs
            in code outside every bracket, when one is given.
        comment_list: A list that gets where each comment mark the read meets is, in
            order, when one is given.

    Returns:
        How far it runs (_CodeRead).
    """
    close_list: list[str] = []
    check_len = 0
    is_prose_stop = True
    at = start
    while True:
        mark = _code_mark(text, at, is_prose_stop)
        if top_list is not None and not close_list:
            top_list.append((at, len(text) if mark is None else mark.start()))
        if mark is not None and mark.lastindex is not None:
            return _CodeRead("bad", text.rfind("\n", 0, mark.start()) + 1)
        if mark is None:
            return _CodeRead("code", len(text), "".join(reversed(close_list)))
        token, at = mark.group(), mark.end()
        if token in _CLOSE_BRACKET_DICT:
            close_list.append(_CLOSE_BRACKET_DICT[token])
        elif token in _CLOSE_BRACKET_DICT.values():
            if not close_list or close_list.pop() != token:
                return _CodeRead("bad", text.rfind("\n", 0, mark.start()) + 1)
            word_match = _WORD_AFTER_RE.match(text, at)
            if is_prose_stop and word_match \
                    and word_match.group(1) not in _PAIR_SKIP_SET:
                return _CodeRead("bad", text.rfind("\n", 0, mark.start()) + 1)
        elif token == "`":
            return _CodeRead("bad", text.rfind("\n", 0, mark.start()) + 1)
        elif token == "#":
            if comment_list is not None:
                comment_list.append(mark.start())
            line_end = text.find("\n", at)
            if line_end < 0:
                return _CodeRead("comment", mark.start(), "".join(reversed(close_list)))
            at = line_end
        elif token == "\n":
            if not close_list:
                return _CodeRead("line", at)
            if at - start >= check_len:
                close_text = "".join(reversed(close_list))
                error_line = _python_error_line(text[start:at], close_text)
                if error_line and all(_python_error_line(text[start:at] + end, close_text)
                                      for end in _CUT_END_TUPLE):
                    return _CodeRead("bad", _line_start(text, start, error_line),
                                     parse_at=at)
                check_len = 2 * (at - start)
        elif token != "\\\n":
            is_emphasis = text[mark.start() - 1:mark.start()] in _EMPHASIS_MARK_TUPLE \
                and text[mark.start() - 1] == text[at:at + 1]
            if len(token) == 3 and is_emphasis:
                return _CodeRead("bad", text.rfind("\n", 0, mark.start()) + 1)
            close_text = token[0] * 2 + "".join(reversed(close_list))
            if len(token) == 3 and mark.start() - start >= check_len:
                error_line = _python_error_line(text[start:mark.start()], close_text)
                if error_line:
                    return _CodeRead("bad", _line_start(text, start, error_line),
                                     parse_at=mark.start())
                check_len = 2 * (mark.start() - start) + 1
            end_match = next((end_match for end_match
                              in _STRING_END_RE_DICT[token].finditer(text, at)
                              if not end_match.group().startswith("\\")), None)
            if end_match is None:
                return _CodeRead("string", mark.start(), close_text)
            if end_match.group() == "\n":
                return _CodeRead("bad", text.rfind("\n", 0, mark.start()) + 1)
            if _TEMPLATE_PREFIX_RE.search(text, max(mark.start() - 2, 0), mark.start()) \
                    and _is_in_field(text, at, end_match.start()):
                is_prose_stop = False
            at = end_match.end()


def _is_in_field(text: str, start: int, end: int) -> bool:
    """Say whether the text of an f-string or a t-string ends in a replacement field.

    Examples:
        {a}          -> False
        {{a}} b      -> False
        {x:'>10} b   -> False
        {title or    -> True
        {x:{width}   -> True
        {'''a'}''' + -> True
        {x  # }      -> True

    Args:
        text: The text.
        start: Where the string's text starts, after its opening quote.
        end: Where it ends, at the quote that closes it as read.

    Returns:
        True when the string's text ends inside a replacement field. Outside every
        field, a doubled brace is a brace of the string and a brace opens a field;
        inside one, each brace of a doubled one is read on its own. In the expression of
        a field, brackets open and close, a closing brace closes the innermost brace
        with the brackets left open in it, a colon outside its brackets starts the
        format spec, and a hash starts a comment that runs to the end of its line (a
        text that ends in the comment ends in the field). A quote there (quotes in three
        first) with a prefix of _TEMPLATE_PREFIX_RE opens an f-string or a t-string,
        read like the string's text up to the same quote (_FSTRING_MARK_RE_DICT); any
        other one opens a string skipped up to the same quote (_STRING_END_RE_DICT). A
        string that no same quote closes ends the text in the field. In a format spec, a
        brace opens a field and a closing brace closes the spec's field; nothing else is
        read.
    """
    open_list: list[str] = []
    at = start
    while True:
        top = open_list[-1] if open_list else ""
        field_match = _FSTRING_MARK_RE_DICT.get(top, _FIELD_MARK_RE).search(text, at, end)
        if field_match is None:
            return bool(open_list)
        mark, at = field_match.group(), field_match.end()
        if not open_list or top in _FSTRING_MARK_RE_DICT:
            if mark == "{":
                open_list.append("field")
            elif mark == top:
                open_list.pop()
            elif mark == "\n":
                return True
            continue
        if mark in ("{{", "}}"):
            mark, at = mark[0], at - 1
        if mark == "}":
            while open_list and open_list.pop() not in ("field", "spec", "{"):
                pass
        elif mark == "{":
            open_list.append("field" if top == "spec" else "{")
        elif top == "spec":
            continue
        elif mark == "#":
            line_end = text.find("\n", at, end)
            if line_end < 0:
                return True
            at = line_end
        elif mark == ":":
            if top == "field":
                open_list[-1] = "spec"
        elif mark in ("(", "["):
            open_list.append(mark)
        elif mark in (")", "]"):
            if _CLOSE_BRACKET_DICT.get(top) == mark:
                open_list.pop()
        elif _TEMPLATE_PREFIX_RE.search(text, max(field_match.start() - 2, 0),
                                        field_match.start()):
            open_list.append(mark)
        else:
            end_match = next((end_match for end_match
                              in _STRING_END_RE_DICT[mark].finditer(text, at, end)
                              if not end_match.group().startswith("\\")), None)
            if end_match is None or end_match.group() == "\n":
                return True
            at = end_match.end()


def _python_error_at(text: str, start: int, code_read: _CodeRead) -> int | None:
    """Return where a logical line read from a text stops being Python.

    Args:
        text: The text.
        start: Where the logical line starts.
        code_read: Its read (_read_code_line), other than "bad".

    Returns:
        None when it reads as Python (_python_error_line): its text for "line", else its
        code closed with its close_text. Else the start of the line of the text where it
        stops being Python.
    """
    error_line = _python_error_line(
        text[start:code_read.stop],
        None if code_read.kind == "line" else code_read.close_text)
    return _line_start(text, start, error_line) if error_line else None


def _python_error_line(code: str, close_text: str | None) -> int:
    """Return the line where a text stops reading as Python statements.

    Args:
        code: The text.
        close_text: None for a text read whole. Else what closes the string and the
            brackets left open where the text was cut off, which is put after it; the
            text then reads as Python as well when ast.parse stops in it where it was
            cut off (_is_cut_error).

    Returns:
        0 when ast.parse reads it, stops in it at the heading of a block with no block
        after it (_NO_BLOCK_ERROR), stops in it where it was cut off, or raises an error
        that is no SyntaxError. Else the line, counted from 1, where ast.parse stops, and
        1 for a text that is a lone name or a lone string.
    """
    tree = _read_python(ast.parse, code + (close_text or ""))
    if isinstance(tree, ast.Module):
        value = tree.body[0].value if len(tree.body) == 1 \
            and isinstance(tree.body[0], ast.Expr) else None
        return int(isinstance(value, (ast.Name, ast.JoinedStr)) or (
            isinstance(value, ast.Constant) and isinstance(value.value, (str, bytes))))
    if not isinstance(tree, SyntaxError):
        return 0
    if str(tree.msg).startswith(_NO_BLOCK_ERROR) or (
            close_text is not None and _is_cut_error(tree, code, close_text)):
        return 0
    return max(tree.lineno or 1, 1)


def _is_cut_error(error: SyntaxError, code: str, close_text: str) -> bool:
    """Say whether ast.parse stops in Python cut off and closed where it was cut off.

    Args:
        error: The error ast.parse raises for the code with close_text after it.
        code: The Python, cut off.
        close_text: What closes the string and the brackets left open where it was cut
            off.

    Returns:
        True for _HEAD_CUT_ERROR. When close_text closes a bracket, True as well for
        _STAR_CUT_ERROR, and for _ELSE_CUT_ERROR when one try reads: ast.parse reads it,
        or stops in it with _NO_BLOCK_ERROR, _HEAD_CUT_ERROR or _STAR_CUT_ERROR. A try
        puts _ELSE_END after the code and the closed string at one depth of the closing
        brackets, from the innermost out; it is kept for the tries further out when
        ast.parse then stops with _ELSE_CUT_ERROR at another conditional expression.
    """
    if str(error.msg).startswith(_HEAD_CUT_ERROR):
        return True
    quote_text = close_text.rstrip("".join(_CLOSE_BRACKET_DICT.values()))
    bracket_text = close_text[len(quote_text):]
    if not bracket_text:
        return False
    if str(error.msg).startswith(_STAR_CUT_ERROR):
        return True
    if not str(error.msg).startswith(_ELSE_CUT_ERROR):
        return False
    head = code + quote_text
    place = (error.lineno, error.offset)
    for depth in range(len(bracket_text) + 1):
        tree = _read_python(ast.parse, head + _ELSE_END + bracket_text[depth:])
        if not isinstance(tree, SyntaxError) or str(tree.msg).startswith(
                (_NO_BLOCK_ERROR, _HEAD_CUT_ERROR, _STAR_CUT_ERROR)):
            return True
        if str(tree.msg).startswith(_ELSE_CUT_ERROR) \
                and (tree.lineno, tree.offset) != place:
            head += _ELSE_END
            place = (tree.lineno, tree.offset)
        head += bracket_text[depth:depth + 1]
    return False


def _line_start(text: str, start: int, line_number: int) -> int:
    """Return where a line of a text starts, counted from the line of a position.

    Args:
        text: The text.
        start: The position; its line is line 1.
        line_number: The line, counted from 1.

    Returns:
        Where that line starts, or the end of the text when the text has no such line.
    """
    at = text.rfind("\n", 0, start) + 1
    for _ in range(line_number - 1):
        at = text.find("\n", at) + 1
        if not at:
            return len(text)
    return at


def _first_call(reply: str) -> re.Match | None:
    """Return the first markup call of a reply that writes it before any code.

    Args:
        reply: The model's reply, or as much of it as has arrived.

    Returns:
        The match of _CALL_OPEN, or None when the reply holds no call, when it starts
        with a parameter (_PARAM_OPEN), and when its first call comes after a fence line
        (_FENCE_LINE_RE) or is quoted (_in_quote).
    """
    open_match = _CALL_OPEN.search(reply)
    if open_match is None or _PARAM_OPEN.match(reply):
        return None
    front = reply[:open_match.start()]
    if _FENCE_LINE_RE.search(front) or _in_quote(front):
        return None
    return open_match


def _in_quote(front: str) -> bool:
    """Say whether a markup call written after a text is quoted rather than written out.

    Args:
        front: The text in front of the call.

    Returns:
        True when the call stands inside a string or a comment of Python code: the
        call's line holds a quote in front of it, or the text ends inside a string or a
        comment of the unfenced Python it holds (_in_string_or_comment). True as well
        when it stands inside Markdown that quotes it: a block quote line, after the
        marks of list items if any (_QUOTE_LINE_RE), an HTML comment left open, an
        inline code span left open in its paragraph (_in_code_span), or an indented
        code block (_in_indented_block).
    """
    line_front = front.rsplit("\n", 1)[-1]
    if "'" in line_front or '"' in line_front or _QUOTE_LINE_RE.match(line_front):
        return True
    if front.rfind(_COMMENT_OPEN) > front.rfind(_COMMENT_CLOSE):
        return True
    return _in_code_span(front) or _in_indented_block(front) \
        or _in_string_or_comment(front)


def _in_code_span(front: str) -> bool:
    """Say whether the end of a text is inside an inline code span of its last paragraph.

    The runs of backticks after the last blank line are paired from the first: a run
    opens a span, and the next run of the same length closes it.

    Args:
        front: The text.

    Returns:
        True when a run is left open.
    """
    paragraph_at = max((blank_match.end() for blank_match in _BLANK_LINE_RE.finditer(front)),
                       default=0)
    open_len = 0
    for run_match in _BACKTICK_RUN.finditer(front, paragraph_at):
        open_len = _span_open_len(open_len, len(run_match.group()))
    return open_len > 0


def _span_open_len(open_len: int, run_len: int) -> int:
    """Return the length of the backtick run left open after one more run (_in_code_span).

    Args:
        open_len: The length of the run left open before it, 0 for none.
        run_len: The length of the run.

    Returns:
        run_len when no run was open, 0 when the run closes the open one, else open_len.
    """
    if not open_len:
        return run_len
    return 0 if run_len == open_len else open_len


def _in_indented_block(front: str) -> bool:
    """Say whether the end of a text is on a line of an indented code block.

    Args:
        front: The text.

    Returns:
        True when its last line starts with _CODE_INDENT columns of indent or more, and
        the lines in front of it are indented as much, or blank, up to a blank line or
        the start of the text.
    """
    line_list = front.split("\n")
    if _indent_width(line_list[-1]) < _CODE_INDENT:
        return False
    for line in reversed(line_list[:-1]):
        if not line.strip():
            return True
        if _indent_width(line) < _CODE_INDENT:
            return False
    return True


def _indent_width(line: str) -> int:
    """Return the columns of indent a line starts with, a tab counting to the next of 4.

    Args:
        line: The line.

    Returns:
        The columns.
    """
    return len(line[:len(line) - len(line.lstrip(" \t"))].expandtabs(4))


def _first_call_end(reply: str) -> int | None:
    """Return where the code of a reply whose first code is a markup call is complete.

    Args:
        reply: The model's reply, or as much of it as has arrived.

    Returns:
        The position after the call's closing tag, or after the code parameter when the
        next parameter follows it. None when _first_call() finds no call or one not
        named with one of _PYTHON_LABEL_TUPLE, and when that call's code parameter is
        empty or has no closing tag that the next parameter or the call's closing tag
        has come after (_PARAM_SURE_CLOSE_RE).
    """
    open_match = _first_call(reply)
    if open_match is None or open_match.group(1).lower() not in _PYTHON_LABEL_TUPLE:
        return None
    code, at = next(((value, at) for key, value, at
                     in _param_list(reply, open_match.end(), is_cut=True)
                     if key.lower() in _CODE_PARAM_TUPLE), ("", 0))
    if not code.strip():
        return None
    if reply[at:].lstrip().startswith(_CALL_CLOSE):
        return reply.index(_CALL_CLOSE, at) + len(_CALL_CLOSE)
    return at


def _code_span_iter(reply: str) -> Iterator[tuple[int, int | None]]:
    """Yield where the text of each fenced block and markup parameter of a reply runs.

    Args:
        reply: The model's reply, or as much of it as has arrived.

    Yields:
        (where the text of one that _block_iter() yields starts, where it stops: in
        front of its closing fence or tag, None when nothing closes it and it runs to the
        end of the reply), in order.
    """
    for block in _block_iter(reply):
        yield block.start, block.stop if block.is_closed else None


def _turn_mark(reply: str) -> re.Match | None:
    """Return the first TURN_MARK_RE of a reply whose heading is not inside its code.

    Args:
        reply: The model's reply, or as much of it as has arrived.

    Returns:
        The match, or None when the reply holds no heading outside the pieces of code
        _code_span_iter() yields.
    """
    span_iter = _code_span_iter(reply)
    span = next(span_iter, None)
    for mark_match in TURN_MARK_RE.finditer(reply):
        line_start = mark_match.start() + 1
        while span is not None and span[1] is not None and span[1] <= line_start:
            span = next(span_iter, None)
        if span is None or line_start < span[0]:
            return mark_match
        if span[1] is None:
            return None
    return None


def reply_end(reply: str) -> int | None:
    """Return where a reply holds all that extract_code() takes from it.

    The text before that position gives extract_code() the same code as the reply
    written out to its end does.

    Examples:
        '<invoke name="python">\\n<parameter name="code">print(1)</parameter>\\n'
        '</invoke>\\n\\n<invoke name="python">'   -> the position after the first </invoke>
        "```python\\nprint(1)\\n```\\n### Turn 2 output"   -> the position after the fence
        "```python\\nprint(1)\\n```"                       -> None
        a block that holds "\\n## Turn 1"                  -> None

    Args:
        reply: The model's reply, or as much of it as has arrived.

    Returns:
        The smaller of the position in front of the first heading _turn_mark() finds
        and the position _first_call_end() returns; None when the reply holds neither.
    """
    mark_match = _turn_mark(reply)
    end_list = [end for end in (mark_match.start() if mark_match else None,
                                _first_call_end(reply)) if end is not None]
    return min(end_list) if end_list else None


def extract_code(reply: str, is_cut: bool = False) -> tuple[str | None, bool]:
    """Take the code to run out of a reply.

    The code is chosen from the parts _block_iter() splits the reply into. A reply that
    writes a markup call holding code before any code (_first_call), or starts with a
    markup parameter holding code, runs that code. Any other reply runs the last block
    _run_part_list() returns that is labelled with one of _PYTHON_LABEL_TUPLE, else the
    last such block with no label whose text would be taken as unfenced code (bare_code).
    A reply with no fenced block to run is read as unfenced code itself, and a reply that
    is not unfenced code either runs the first markup call it writes outside its fenced
    blocks (_prose_call_code), and else, when its markup gives no parameter, the unfenced
    code left once the markup around it is taken away (_trim_code). A reply
    _run_part_list() cannot split runs nothing unless it is unfenced code as a whole,
    with or without that markup. A reply cut off at the output limit is not read as
    unfenced code.

    Args:
        reply: The model's reply.
        is_cut: Whether the reply was cut off at the output limit.

    Returns:
        (code, whether it was taken from an unfenced reply). (None, False) when no code
        could be taken.
    """
    part_list = list(_block_iter(reply, is_cut))
    first_call = _first_call(reply)
    # The markup is taken away only from a reply whose markup gives no parameter
    is_trim = not is_cut and not any(not part.is_fence for part in part_list)
    if first_call or _PARAM_OPEN.match(reply):
        code = _call_code(reply, first_call, _call_param_list(
            part_list, first_call.start() if first_call else None), is_python_only=True)
        if code and code.strip():
            return code, True
    run_list = _run_part_list(reply, part_list, is_cut)
    if run_list is None:
        code = None if is_cut else bare_code(reply) or (
            _trim_code(reply) if is_trim else None)
        return (code, True) if code else (None, False)
    code_list = ([part.text for part in run_list
                  if part.is_fence and part.label in _PYTHON_LABEL_TUPLE]
                 or [part.text for part in run_list
                     if part.is_fence and not part.label and bare_code(part.text)])
    if code_list:
        return code_list[-1], False
    code = (None if is_cut else bare_code(reply)) \
        or _prose_call_code(reply, part_list, is_cut) \
        or (_trim_code(reply) if is_trim else None)
    return (code, True) if code else (None, False)


def tool_line(fn: Callable) -> str:
    """Render one tool as "- name(signature) — one-line description".

    Examples:
        revert_file -> "- revert_file(path: str) -> dict — Undo this request's changes
        to one file, putting back what it held at the start."

    Args:
        fn: The tool function.

    Returns:
        The single line.
    """
    doc = (fn.__doc__ or "").strip().split("\n")[0]
    return f"- {fn.__name__}{inspect.signature(fn)} — {doc}"


def fence_of(text: str) -> str:
    """Return the fence that encloses a text shown to the model.

    The fence is one backtick longer than the longest run of backticks inside the text,
    and three at least.

    Examples:
        "print(1)"   -> "```"
        "a ``` b"    -> "````"
        "a ````` b"  -> "``````"

    Args:
        text: The text to enclose: a turn's code or output in the history, or what a
            program printed.

    Returns:
        The backticks that open and close it.
    """
    longest = max((len(run_match.group()) for run_match in _BACKTICK_RUN.finditer(text)),
                  default=0)
    return "`" * max(3, longest + 1)


def _history_size(history_list: list[tuple[str, str]]) -> int:
    """Return how many characters a history's codes and outputs hold together.

    Args:
        history_list: [(code, output), ...] as RLM.run() keeps it.

    Returns:
        The sum of the lengths of every code and every output.
    """
    return sum(len(code) + len(output) for code, output in history_list)


def _fold(text: str, head_char: int, note: str) -> str:
    """Cut one turn's code or output down to its head and say how much was dropped.

    Args:
        text: The code or the output as it stands in the history.
        head_char: How many characters of the head to keep.
        note: FOLD_NOTE for an output, _CODE_FOLD_NOTE for a code.

    Returns:
        The text unchanged when it is already that short, otherwise its head plus the
        note with the dropped count filled in.
    """
    if len(text) <= head_char:
        return text
    return text[:head_char] + note.format(omitted=len(text) - head_char)


def fold_by_count(history_list: list[tuple[str, str]], fold_count_tuple: tuple[int, int],
                  head_char: int) -> list[tuple[str, str]]:
    """Fold the output, and the code, of as many of the oldest turns as the counts say.

    Args:
        history_list: [(code, output), ...] as RLM.run() keeps it.
        fold_count_tuple: (how many of the oldest turns have their output folded, how
            many have their code folded).
        head_char: How much of a folded turn's output, or code, stays.

    Returns:
        A new list of the same length. The same list object is returned untouched when
        both counts are 0.
    """
    output_count, code_count = fold_count_tuple
    if not output_count and not code_count:
        return history_list
    return [(_fold(code, head_char, _CODE_FOLD_NOTE) if index < code_count else code,
             _fold(output, head_char, FOLD_NOTE) if index < output_count else output)
            for index, (code, output) in enumerate(history_list)]


def fold_history(history_list: list[tuple[str, str]], max_char: int, keep_turn: int,
                 head_char: int, fold_count_tuple: tuple[int, int] = (0, 0)
                 ) -> tuple[list[tuple[str, str]], tuple[int, int]]:
    """Fold the oldest turns once the history is over max_char.

    The history is folded down to _HISTORY_FOLD_SHARE of max_char. The turns an earlier
    call folded stay folded, and the history is measured with them folded. While it is
    within max_char no further turn is folded, and the turns already in a prompt are
    rendered the same way in the next one.

    The outputs are folded first, oldest first. When every old turn's output is folded
    and the history is still over the target, the code of the same old turns is folded
    the same way. The kept turns are never touched.

    Args:
        history_list: [(code, output), ...] as RLM.run() keeps it.
        max_char: The budget for the whole history. 0 or less folds nothing further.
        keep_turn: How many of the most recent turns are never folded.
        head_char: How much of a folded turn's output, or code, stays.
        fold_count_tuple: What an earlier call over the same history returned as its
            counts. (0, 0) for a history nothing has been folded in.

    Returns:
        (a new list of the same length, or the same list object when nothing is folded,
        (how many of the oldest turns have their output folded, how many have their
        code folded)).
    """
    fold_list = fold_by_count(history_list, fold_count_tuple, head_char)
    total = _history_size(fold_list)
    if max_char <= 0 or total <= max_char:
        return fold_list, fold_count_tuple
    stop_char = max_char * _HISTORY_FOLD_SHARE
    fold_list = list(fold_list)
    count_list = list(fold_count_tuple)
    old_count = max(0, len(history_list) - keep_turn)
    # (which of the two counts, which half of a turn, its note): the outputs, then the code
    for count_at, slot, note in ((0, 1, FOLD_NOTE), (1, 0, _CODE_FOLD_NOTE)):
        for index in range(count_list[count_at], old_count):
            if total <= stop_char:
                return fold_list, (count_list[0], count_list[1])
            pair_list = list(fold_list[index])
            short = _fold(pair_list[slot], head_char, note)
            total -= len(pair_list[slot]) - len(short)
            pair_list[slot] = short
            fold_list[index] = tuple(pair_list)
            count_list[count_at] = index + 1
    return fold_list, (count_list[0], count_list[1])


_NEXT_TURN_CLOSE = "Write the next ```python block."

# What a stage is told over its last few turns (make_deadline_note)
_DEADLINE_NOTE = "## {remain} turns left\n{what}"


def make_deadline_note(over_turn: int, what: str) -> Callable[[int, int], str]:
    """Build a turn_note that counts down over a stage's last turns.

    Args:
        over_turn: How many turns before the end the note starts appearing. 0 turns
            it off.
        what: The sentence telling the model what to do with the turns it has left.

    Returns:
        turn_note(turn, max_turn) -> str for RLM.run(): the note, or "" while there
        is still time and for a turn past max_turn.
    """
    def turn_note(turn: int, max_turn: int) -> str:
        """Return the countdown for one turn, or "" outside the last over_turn turns."""
        remain_turn = max_turn - turn + 1
        return _DEADLINE_NOTE.format(remain=remain_turn, what=what) \
            if 1 <= remain_turn <= over_turn else ""

    return turn_note


def turn_note_chain(*note_tuple: Callable[[int, int], str] | None
                    ) -> Callable[[int, int], str]:
    """Put several per-turn notes one after another, leaving out what is empty.

    Args:
        *note_tuple: turn_note(turn, max_turn) -> str callables, or None for one that
            is off.

    Returns:
        A single turn_note(turn, max_turn) -> str.
    """
    live_list = [note for note in note_tuple if note]

    def turn_note(turn: int, max_turn: int) -> str:
        """Return the notes of one turn that are not empty, a blank line between them."""
        return "\n\n".join(text for text in (note(turn, max_turn) for note in live_list)
                           if text)

    return turn_note


def build_prompt_block_list(instruction: str, tool_dict: dict[str, Callable], var_doc: str,
                            request: str, history_list: list[tuple[str, str]],
                            turn: int, max_turn: int, turn_note: str = "",
                            close_line: str = _NEXT_TURN_CLOSE) -> list[tuple[str, bool]]:
    """Assemble the prompt for one turn as blocks.

    The history goes in as fold_history left it. The blocks joined are the prompt:
    instructions, variables, tools, request, history, turn_note, then the "## Turn i/n"
    line with the closing sentence. The first block holds the four parts that do not
    change over a run, then comes one block per turn of the history, then one block with
    the rest. A cache point follows the first block and the newest turn's block.

    Args:
        instruction: The instructions placed at the top of the prompt.
        tool_dict: {function name: function}.
        var_doc: The description of the variables.
        request: The change request with the search result attached.
        history_list: [(code, output), ...].
        turn: Which turn this is (1-based).
        max_turn: The maximum number of turns.
        turn_note: Text rebuilt every turn and placed after the history. "" adds
            nothing.
        close_line: The last line, after "## Turn i/n". The default asks for the next code
            block.

    Returns:
        [(text, whether a cache point follows it), ...]. Joined, the texts are the
        prompt for one turn.
    """
    head = "\n".join([instruction,
                      "\n## Variables already in the REPL\n" + var_doc,
                      "\n## Tools already in the REPL (call directly, do not import)\n"
                      + "\n".join(tool_line(tool) for tool in tool_dict.values()),
                      "\n## Request\n" + request])
    turn_text_list = [f"\n{TURN_TITLE} {number} code\n{fence_of(code)}python\n{code}\n"
                      f"{fence_of(code)}"
                      f"\n{TURN_TITLE} {number} output\n{fence_of(output)}\n{output}\n"
                      f"{fence_of(output)}"
                      for number, (code, output) in enumerate(history_list, 1)]
    tail_list = (["\n" + turn_note] if turn_note else []) + \
        [f"\n{NEXT_TURN_TITLE} {turn}/{max_turn}\n{close_line}"]
    block_list = [(head + "\n", True)]
    block_list += [(text + "\n", number == len(turn_text_list))
                   for number, text in enumerate(turn_text_list, 1)]
    block_list.append(("\n".join(tail_list), False))
    return block_list


def prompt_text(block_list: list[tuple[str, bool]]) -> str:
    """Join a prompt's blocks into the one string they make.

    Args:
        block_list: build_prompt_block_list()'s return value.

    Returns:
        The prompt for one turn.
    """
    return "".join(text for text, _ in block_list)


def build_prompt(instruction: str, tool_dict: dict[str, Callable], var_doc: str,
                 request: str, history_list: list[tuple[str, str]],
                 turn: int, max_turn: int, turn_note: str = "",
                 close_line: str = _NEXT_TURN_CLOSE) -> str:
    """Assemble the prompt for one turn as one string.

    Takes what build_prompt_block_list() takes.

    Returns:
        The prompt string for one turn.
    """
    return prompt_text(build_prompt_block_list(
        instruction, tool_dict, var_doc, request, history_list, turn, max_turn,
        turn_note=turn_note, close_line=close_line))


def prompt_of(block_list: list[tuple[str, bool]], note: str = "") -> str | list[tuple[str, bool]]:
    """Return what the LLM is sent for one call.

    Args:
        block_list: build_prompt_block_list()'s return value.
        note: Text appended to the end of the prompt for this call alone, "" for none.

    Returns:
        The blocks, with note on the last one, when config.PROMPT_CACHE is on; their
        text as one string otherwise.
    """
    if note:
        last_text, is_cache_point = block_list[-1]
        block_list = block_list[:-1] + [(last_text + note, is_cache_point)]
    return block_list if config.PROMPT_CACHE else prompt_text(block_list)


def _add_usage(total_dict: dict, usage_dict: dict) -> dict:
    """Add one call's token counts into a running total.

    Args:
        total_dict: The total so far, by key; empty to start from.
        usage_dict: One call's counts, as broker.usage_dict_of() shapes them.

    Returns:
        total_dict, with every key of usage_dict added in.
    """
    for key, count in usage_dict.items():
        total_dict[key] = total_dict.get(key, 0) + count
    return total_dict


class RLM:
    """The loop that has the model write code and then runs it, over and over."""

    def __init__(self, llm: Callable[[str], str], tool_list: list[Callable],
                 instruction: str, stage: str = "",
                 max_turn: int | None = None,
                 cut_retry_count: int | None = None) -> None:
        """Take the tools and the instructions and hold the loop settings.

        A setting left None is read from config when the RLM is built.

        Args:
            llm: A callable that takes a prompt and returns a string (make_llm).
            tool_list: The functions to place in the REPL. Their own names are used.
            instruction: The instructions placed at the top of the prompt.
            stage: What this stage is called in the turn log.
            max_turn: The maximum number of turns. None reads config.RLM_MAX_TURN.
            cut_retry_count: How many times a turn that was cut off without yielding
                a single block is asked again. Spent per turn; 0 drops such a turn.
                None reads config.CUT_REPLY_MAX_RETRY.
        """
        self.llm = llm
        self.tool_dict = {tool.__name__: tool for tool in tool_list}
        self.instruction = instruction
        self.stage = stage
        self.max_turn = config.RLM_MAX_TURN if max_turn is None else max_turn
        # How much of one turn's output is kept in the history
        self.max_output_char = config.RLM_MAX_OUTPUT_CHAR
        self.cut_retry_count = (config.CUT_REPLY_MAX_RETRY
                                if cut_retry_count is None else cut_retry_count)
        # The budget the history is folded to. It is lowered for the rest of the run
        # each time the provider refuses a prompt for its length
        self.history_max_char = config.HISTORY_MAX_CHAR
        # (how many of the oldest turns have their output folded, how many have their
        # code folded) in the run in progress, as fold_history() returns them
        self.fold_count_tuple = (0, 0)
        # Which start of this stage the run in progress is, as the turn log counts it
        self.call = 0

    def fold(self, history_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Fold a history down to the stage's budget, keeping what it folded before.

        Args:
            history_list: [(code, output), ...] as run() keeps it.

        Returns:
            The history fold_history() left.
        """
        fold_list, self.fold_count_tuple = fold_history(
            history_list, self.history_max_char, config.HISTORY_KEEP_TURN,
            config.HISTORY_FOLD_HEAD_CHAR, self.fold_count_tuple)
        return fold_list

    def shrink_history_budget(self, history_list: list[tuple[str, str]]) -> None:
        """Lower the history budget after the provider refused a prompt for its length.

        The budget becomes half of what it was, and less than the history the refused
        prompt carried. The next fold() folds further.

        Args:
            history_list: The history as it stands, unfolded.
        """
        size = _history_size(fold_by_count(history_list, self.fold_count_tuple,
                                           config.HISTORY_FOLD_HEAD_CHAR))
        self.history_max_char = max(1, min((self.history_max_char or size) // 2, size - 1))
        logger.warning("the provider refused the prompt of the %s stage for its length; "
                       "the history is folded to %d chars for the rest of the run",
                       self.stage, self.history_max_char)

    def _ask(self, block_list: list[tuple[str, bool]]) -> tuple[str, str | None, bool,
                                                                bool, int, dict]:
        """Call the LLM, asking again only when no block could be taken at all.

        There are two such cases: a reply cut off at the output limit, and a reply with
        not a single character in it, which is what a reply that starts with
        TURN_MARK_RE comes back as.

        A retry does not consume a turn. The note is appended only to this call's prompt
        and never stored in the history. Once one closed block has been taken there is
        no retry, even when the reply was cut off after it.

        Args:
            block_list: One turn's prompt as assembled by build_prompt_block_list().

        Returns:
            (reply, code, is_bare, is_cut, cut_call_count, usage_dict). reply is the
            last call's reply, code is the runnable code taken from it (None when there
            is none), is_bare is whether it came from an unfenced reply, is_cut is
            whether the reply was cut off at the output limit, cut_call_count is how
            many calls were cut off in this turn (truncated_replies counts calls, not
            turns), and usage_dict is the token counts of the turn's calls added up
            (empty for an llm that reports none).

        Raises:
            PromptTooLong: When the provider refused the prompt for its length.
            CallFailure: When the call itself did not happen for any other reason.
        """
        note = ""
        cut_call_count = 0
        usage_dict: dict = {}
        for _ in range(self.cut_retry_count + 1):
            # A prompt refused for its length is raised as it is; any other call that
            # could not be made is raised as CallFailure, without a retry
            try:
                reply = self.llm(prompt_of(block_list, note))
            except PromptTooLong:
                raise
            except Exception as e:
                raise CallFailure(f"{type(e).__name__}: {e}") from e
            # make_llm stores the finish reason and the token counts on the function
            # object; getattr falls back for a callable that has no such attribute
            is_cut = getattr(self.llm, "last_finish_reason", None) == "length"
            cut_call_count += is_cut
            _add_usage(usage_dict, getattr(self.llm, "last_usage_dict", None) or {})
            code, is_bare = extract_code(reply, is_cut)
            # A cut reply whose block does not parse is a block the limit cut short, and
            # is taken as no block at all
            if code and is_cut and not _parse_ok(code):
                code, is_bare = None, False
            if code:
                break
            note = NO_CODE_NOTE_DICT[_no_code_reason(reply, is_cut)][1]
            if not note:
                break   # a prose-only turn is not retried
        return reply, code, is_bare, is_cut, cut_call_count, usage_dict

    def run(self, request: str, variable_dict: dict[str, object], var_doc: str,
            turn_note: Callable[[int, int], str] | None = None,
            on_complete: Callable[[str], str] | None = None,
            after_turn: Callable[[int, str], str] | None = None) -> dict:
        """Run the loop for one request.

        A turn that yielded no code block still counts as a turn, and the loop always
        stops after max_turn even if the model keeps answering with prose only. The
        first COMPLETE() on_complete sends back on the last turn adds one turn, in which
        the refusal is answered.

        Each turn goes through four steps:
        1. Ask for the turn's code (_ask_turn). A call that does not happen ends the
           run.
        2. Keep a turn that yielded no code, its reply commented out
           (_keep_no_code_turn).
        3. Run the code. A COMPLETE() that on_complete accepts ends the run.
        4. Add the notes on how the block was taken (_mark_output) and what after_turn
           returns to the output, and keep the turn.

        Args:
            request: The change request, with the search result already attached.
            variable_dict: The variables to place in the REPL.
            var_doc: The description of those variables, for the prompt.
            turn_note: Called as turn_note(turn, max_turn) before every turn; what it
                returns is placed after the history. None adds nothing.
            on_complete: Called as on_complete(answer) with the argument of the call
                that ended the run. Returning "" accepts it and the loop ends; returning
                text refuses it: that text becomes the turn's output and the model
                carries on. None accepts every COMPLETE().
            after_turn: Called as after_turn(turn, output) once the turn's code has run.
                What it returns is appended to that turn's output after the cut, and is
                never clipped. None appends nothing.

        Returns:
            {"answer": str | None,  # the argument it ended with, else None
             "outcome": str | None,     # COMPLETE_KEY when it ended itself, else None
             "iters": int,          # how many turns actually ran
             "history": list,       # [(code, output), ...]; len(history) == iters
             "error": str,          # why the loop stopped early; "" when it did not
             "truncated_replies": int,  # calls whose reply hit the output limit
             "truncated_lost": int,     # turns lost even after asking again
             "empty_replies": int,      # turns lost to a reply with no characters
             "bare_code_turns": int,    # turns run from an unfenced reply
             "refused_completes": int}    # COMPLETE() calls on_complete sent back
        """
        repl = Repl(variable_dict, self.tool_dict)
        history_list = []
        self.fold_count_tuple = (0, 0)
        count_dict = dict(RUN_COUNT_DICT)
        tool_name_list = [*self.tool_dict, *END_CALL_DICT.values()]
        self.call = turnlog.start_stage(self.stage)
        logger.info("the %s stage starts: at most %d turns", self.stage, self.max_turn)
        # The first refusal on the last turn moves the last turn one further
        last_turn = self.max_turn
        turn = 0
        while turn < last_turn:
            turn += 1
            # == Step 1: ask for the turn's code ==========================
            try:
                reply, code, is_bare, is_cut, cut_call_count, usage_dict = self._ask_turn(
                    request, var_doc, history_list, turn, last_turn, turn_note)
            except CallFailure as e:
                logger.warning("the call for turn %d did not happen (%s), so the run "
                               "stops here with what it had already written", turn, e)
                return self._end({"answer": None, "outcome": None, "iters": turn - 1,
                                  "history": history_list, "error": str(e), **count_dict},
                                 last_turn)
            # code may also come from an unfenced reply; cut_call_count counts the calls
            # cut off at the output limit
            count_dict["truncated_replies"] += cut_call_count
            count_dict["bare_code_turns"] += is_bare

            # == Step 2: keep a turn that yielded no code =================
            if code is None:
                self._keep_no_code_turn(history_list, count_dict, turn, last_turn, reply,
                                        is_cut, usage_dict)
                continue
            logger.debug("%s turn %d of at most %d: %s", self.stage, turn, last_turn,
                         tool_call_text(code, tool_name_list) or "no tool call")

            # == Step 3: run the code =====================================
            try:
                output = repl.run(code, self.max_output_char)
            except CompleteCall as call:
                # What the turn printed before the call is kept either way
                print_text = f"{call.output}\n" if call.output else ""
                # A guard may send the call back. Its refusal becomes this turn's output
                # and the loop goes on
                refusal = on_complete(call.answer) if on_complete else ""
                if refusal:
                    count_dict["refused_completes"] += 1
                    self._keep_turn(history_list, turn, code, print_text + refusal,
                                    usage_dict)
                    if turn == last_turn == self.max_turn:
                        last_turn += 1
                        logger.info("%s() was sent back on the last turn, so the %s stage "
                                    "gets turn %d to answer it", END_CALL_DICT[COMPLETE_KEY],
                                    self.stage, last_turn)
                    continue
                # The turn that ended the run goes into the history before returning
                self._keep_turn(history_list, turn, code,
                                f"{print_text}{END_CALL_DICT[COMPLETE_KEY]}() was called: "
                                f"{text_of(call.answer)}", usage_dict)
                return self._end({"answer": call.answer, "outcome": COMPLETE_KEY,
                                  "iters": turn, "history": history_list, "error": "",
                                  **count_dict}, last_turn)

            # == Step 4: add the notes to the output and keep the turn ====
            output = self._mark_output(output, code, is_bare, is_cut, reply, turn)
            # The machine's own note is added after the cut and is never clipped
            note = after_turn(turn, output) if after_turn else ""
            self._keep_turn(history_list, turn, code, f"{output}\n{note}" if note else output,
                            usage_dict)
        return self._end({"answer": None, "outcome": None, "iters": last_turn,
                          "history": history_list, "error": "", **count_dict}, last_turn)

    def _ask_turn(self, request: str, var_doc: str, history_list: list[tuple[str, str]],
                  turn: int, last_turn: int, turn_note: Callable[[int, int], str] | None
                  ) -> tuple[str, str | None, bool, bool, int, dict]:
        """Build one turn's prompt and ask the LLM for the turn's code (_ask).

        A prompt the provider refuses for its length is folded harder and sent once
        more; refused again, it ends the run as any other failed call does.

        Args:
            request: The request, as run() takes it.
            var_doc: The description of the variables, as run() takes it.
            history_list: The history as run() keeps it, unfolded.
            turn: Which turn this is, counting from 1.
            last_turn: The last turn the stage can run, as run() moved it.
            turn_note: run()'s turn_note, or None.

        Returns:
            _ask()'s return value.

        Raises:
            CallFailure: When the call did not happen, a second refusal for the prompt's
                length included.
        """
        return _send_with_refold(
            self, history_list,
            lambda: build_prompt_block_list(
                self.instruction, self.tool_dict, var_doc, request,
                self.fold(history_list), turn,
                last_turn,
                turn_note=turn_note(turn, last_turn) if turn_note else ""),
            lambda block_list: self._ask(block_list), is_call_failure=True)

    def _keep_turn(self, history_list: list[tuple[str, str]], turn: int, code: str,
                   output: str, usage_dict: dict) -> None:
        """Put one turn in the history and in the turn log.

        Args:
            history_list: The history run() keeps. The turn is appended to it.
            turn: Which turn of this stage it is, counting from 1.
            code: The block that was run, as the history keeps it.
            output: What came back, as the history keeps it.
            usage_dict: The token counts of the turn's calls, as _ask() adds them up.
        """
        history_list.append((code, output))
        turnlog.record(self.stage, turn, code, output, usage_dict=usage_dict,
                       call=self.call)

    def _keep_no_code_turn(self, history_list: list[tuple[str, str]], count_dict: dict,
                           turn: int, last_turn: int, reply: str, is_cut: bool,
                           usage_dict: dict) -> None:
        """Count a turn that yielded no code, and keep it with its reply commented out.

        Args:
            history_list: The history run() keeps. The turn is appended to it.
            count_dict: run()'s counters (RUN_COUNT_DICT), counted up in place.
            turn: Which turn of this stage it is, counting from 1.
            last_turn: The last turn the stage can run, for the log.
            reply: The model's reply.
            is_cut: Whether that reply was cut off at the output limit.
            usage_dict: The token counts of the turn's calls, as _ask() adds them up.
        """
        reason = _no_code_reason(reply, is_cut)
        logger.debug("%s turn %d of at most %d: no code (%s)", self.stage, turn,
                     last_turn, reason)
        count_dict["truncated_lost"] += is_cut
        count_dict["empty_replies"] += reason == "empty"
        # The raw reply never goes into the history's code slot. It is commented out
        # instead, newlines included
        excerpt = reply[:config.RAW_EXCERPT_CHAR].replace("`", "'") \
            .replace("\n", "\n# ")
        self._keep_turn(history_list, turn, NO_CODE_MARK + excerpt,
                        NO_CODE_NOTE_DICT[reason][0], usage_dict)

    def _mark_output(self, output: str, code: str, is_bare: bool, is_cut: bool,
                     reply: str, turn: int) -> str:
        """Add the notes on how a turn's block was taken to the turn's output.

        Args:
            output: What the block printed (Repl.run).
            code: The block.
            is_bare: Whether the block came from an unfenced reply.
            is_cut: Whether the reply was cut off at the output limit.
            reply: The model's reply, whose end goes to the log when it was cut off.
            turn: Which turn of this stage it is, for the log.

        Returns:
            output, with FENCE_LINE_NOTE after it for a fenced block that does not
            compile and CUT_REPLY_NOTE after that for a reply cut off at the output
            limit.
        """
        # A fenced block that does not compile is told where a block ends
        if not is_bare and not _parse_ok(code):
            output = f"{output}\n{FENCE_LINE_NOTE}"
        # Even when a closed block was taken, the reply may have been cut after it. What
        # followed the block is not in the history; its end goes to the log
        if is_cut:
            output = f"{output}\n{CUT_REPLY_NOTE}"
            logger.info("%s turn %d: the reply was cut at the output limit after "
                        "%d characters, and its block was run. It ends with: %s",
                        self.stage, turn, len(reply),
                        reply[-config.RAW_EXCERPT_CHAR:])
        return output

    def _end(self, result_dict: dict, last_turn: int) -> dict:
        """Log how the stage ended, and return what it returns.

        Args:
            result_dict: The return value of run().
            last_turn: The last turn the stage could run, as run() moved it.

        Returns:
            result_dict, unchanged.
        """
        # What the output limit cost the stage goes first; the line that says how the
        # stage ended stays its last
        cut_count_dict = {key: result_dict[key] for key in
                          ("truncated_replies", "truncated_lost", "empty_replies")
                          if result_dict[key]}
        if cut_count_dict:
            is_lost = result_dict["truncated_lost"] or result_dict["empty_replies"]
            logger.log(logging.WARNING if is_lost else logging.INFO,
                       "the %s stage had replies the output limit cut: %s",
                       self.stage, ", ".join(f"{key}={count}"
                                             for key, count in cut_count_dict.items()))
        if result_dict["error"]:
            how = f"a call that did not happen ({result_dict['error']})"
        elif result_dict["outcome"]:
            how = f"{END_CALL_DICT[result_dict['outcome']]}()"
        else:
            how = "no turn left"
        logger.info("the %s stage ends after %d of at most %d turns: %s", self.stage,
                    result_dict["iters"], last_turn, how)
        return result_dict


def _json_body(text: str) -> str:
    """Return the outermost {...} or [...] of a reply, past a fence or a sentence around it.

    Args:
        text: What the stage handed to COMPLETE().

    Returns:
        The text from the first bracket to the last matching one, or the text unchanged
        when there is no such pair.
    """
    open_pos_list = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    if not open_pos_list:
        return text
    start = min(open_pos_list)
    end = text.rfind("}" if text[start] == "{" else "]")
    return text[start:end + 1] if start < end else text


def _plain_value(value: object) -> tuple[object, str]:
    """Return an object as the JSON written of it reads back.

    The object is written with json.dumps and read with json.loads (_read_in_deadline):
    a tuple comes back a list, a key that is a number comes back its text, and a value
    of a subclass of str, int, float, list or dict comes back the plain value.

    Args:
        value: What a stage handed to COMPLETE() as the object itself.

    Returns:
        (the object read back, ""), or (None, "<error type>: <what was raised>") for an
        object holding a value JSON has no form for (a set, bytes, an object of a
        class), one that holds itself, one nested deeper than can be written, and one
        whose writing raises anything else or does not return.
    """
    plain_value, error = _read_in_deadline(
        lambda: json.loads(json.dumps(value, ensure_ascii=False)))
    if error is not None:
        return None, f"{type(error).__name__}: {text_of(error)}"
    return plain_value, ""


def _json_write_error(value: object) -> str:
    """Say why an object cannot be written as JSON, or "" when it can.

    Args:
        value: What a stage handed to COMPLETE() as the object itself.

    Returns:
        The error of _plain_value(), "" for an object that can be written.
    """
    return _plain_value(value)[1]


def text_of(value: object) -> str:
    """Return whatever a stage handed over as text.

    Args:
        value: The argument of the call that ended a run.

    Returns:
        str(value) as a plain str, or the name of its type between "<" and ">" for a
        value str() does not write: an integer of more digits than are written, an
        object nested deeper than is written, and an object whose own str() raises or
        does not return (_read_in_deadline).

    Raises:
        TurnTimeout: When it is raised inside another block's turn_deadline().
        KeyboardInterrupt: As it came.
    """
    if type(value) is str:
        return value
    text, error = _read_in_deadline(lambda: str.__str__(str(value)))
    return f"<{type(value).__name__}>" if error is not None else text


def parse_json_answer(answer: object, what: str) -> dict:
    """Read a stage's COMPLETE() argument as JSON.

    A value handed to COMPLETE() in the Python REPL that is already the object is taken
    as the JSON written of it reads back (_plain_value), and a string is read as JSON. A
    fence or a sentence around the JSON is read past (_json_body), and the value that
    comes back holds lists, dicts with text keys and plain values alone. The string is
    read with no cap on its length.

    Args:
        answer: The "answer" of RLM.run(): the object itself, or the JSON as a string.
        what: What the answer is called in the log and in the error, e.g. "the plan".

    Returns:
        {"ok": True, "value": object}, or {"ok": False, "error": str, "raw": str} when a
        string could not be read as JSON (a number of more digits than are read and a
        nesting deeper than is read included), or an object holds a value JSON cannot
        hold (_json_write_error). The shape of "value" is not checked.
    """
    if not isinstance(answer, str):
        plain_value, error = _plain_value(answer)
        if error:
            logger.warning("%s cannot be written as JSON: %s", what, error)
            return {"ok": False, "error": f"not JSON: {error}",
                    "raw": text_of(answer)[:config.RAW_EXCERPT_CHAR]}
        return {"ok": True, "value": plain_value}
    answer = text_of(answer)
    try:
        return {"ok": True, "value": json.loads(_json_body(answer))}
    except (ValueError, RecursionError) as e:
        logger.warning("%s is not valid JSON: %s: %s", what, type(e).__name__, e)
        return {"ok": False, "error": f"not valid JSON: {e}",
                "raw": answer[:config.RAW_EXCERPT_CHAR]}


# What a stage is told, each time, when what it handed to COMPLETE() cannot be read as
# JSON. It asks for the object itself, whose contents need no escapes
_JSON_READ_REASON = ("{what} could not be read as JSON ({error}). Hand it over as the "
                     "object itself, {call}({{...}}) or {call}([...]), rather than as a "
                     "string, so nothing inside it has to be escaped")

# What a stage is told, each time, when the object it handed to COMPLETE() holds a value
# JSON cannot hold
_JSON_WRITE_REASON = ("{what} holds a value JSON cannot hold ({error}). Hand over an "
                      "object made of dicts, lists, strings, numbers, True, False and "
                      "None alone")

# What a stage is told, once, when it handed None to COMPLETE()
_JSON_NONE_REASON = ("{call}() got None in place of {what}. Hand over the object itself, "
                     "{call}({{...}}) or {call}([...])")

_JSON_REFUSAL = "{call}() was not accepted: {reason}. This is asked once only."

# What a stage is told each time what it handed to COMPLETE() cannot be read as JSON
_JSON_READ_REFUSAL = "{call}() was not accepted: {reason}."


def json_read_reason(answer: object, what: str) -> str:
    """Say why a COMPLETE() argument cannot be read as JSON, or accept it.

    Args:
        answer: The argument of the COMPLETE() that ended the run. An object that is
            not a string is already read, and None is not an answer.
        what: What the answer is called, e.g. "the plan".

    Returns:
        _JSON_READ_REASON filled in, _JSON_NONE_REASON for None, _JSON_WRITE_REASON for
        an object JSON cannot hold, or "" when the answer can be read.
    """
    if answer is None:
        return _JSON_NONE_REASON.format(what=what, call=END_CALL_DICT[COMPLETE_KEY])
    if not isinstance(answer, str):
        error = _json_write_error(answer)
        return _JSON_WRITE_REASON.format(what=what, error=error) if error else ""
    read_dict = parse_json_answer(answer, what)
    if read_dict["ok"]:
        return ""
    return _JSON_READ_REASON.format(what=what, error=read_dict["error"],
                                    call=END_CALL_DICT[COMPLETE_KEY])


def read_first_guard(what: str, check: Callable[[object], str]) -> Callable[[object], str]:
    """Build an on_complete guard that sends back every answer that cannot be read as JSON.

    A string that cannot be read as JSON, and an object JSON cannot hold
    (json_read_reason), are sent back each time they come, while the stage has turns
    left, without counting against the one refusal. None, which hands nothing over, and
    an answer that can be read share the one refusal (refuse_once): None is sent back
    with json_read_reason()'s text, and an answer that can be read goes to check.

    Args:
        what: What the answer is called, e.g. "the plan".
        check: (an answer that can be read) -> the refusal text, or "" to accept.

    Returns:
        on_complete(answer) -> str for RLM.run(): "" accepts, text refuses.
    """
    def check_once(answer: object) -> str:
        """Return the refusal of None, or check's of an answer that can be read."""
        if answer is None:
            return _JSON_REFUSAL.format(call=END_CALL_DICT[COMPLETE_KEY],
                                        reason=json_read_reason(answer, what))
        return check(answer)

    once_guard = refuse_once(check_once)

    def on_complete(answer: object) -> str:
        """Return the refusal of an unreadable answer, or the one refusal while unspent."""
        reason = json_read_reason(answer, what) if answer is not None else ""
        if reason:
            logger.info("%s() sent back: %s", END_CALL_DICT[COMPLETE_KEY], reason)
            return _JSON_READ_REFUSAL.format(call=END_CALL_DICT[COMPLETE_KEY],
                                             reason=reason)
        return once_guard(answer)

    return on_complete


def json_guard(what: str,
               shape_reason: Callable[[object], str] | None = None) -> Callable[[object], str]:
    """Build an on_complete guard for a JSON hand-over.

    An answer that cannot be read is sent back each time, and None or one that can be
    read but has the wrong shape once (read_first_guard).

    Args:
        what: What the answer is called, e.g. "the plan".
        shape_reason: (the object read from the answer) -> what is wrong with its
            shape, without a full stop, or "". None checks no shape.

    Returns:
        on_complete(answer) -> str for RLM.run(): "" accepts, text refuses.
    """
    def check(answer: object) -> str:
        """Return the refusal for an answer that has the wrong shape."""
        reason = shape_reason(parse_json_answer(answer, what)["value"]) \
            if shape_reason else ""
        return _JSON_REFUSAL.format(call=END_CALL_DICT[COMPLETE_KEY],
                                    reason=reason) if reason else ""

    return read_first_guard(what, check)


def unwrap_end_call(reply: str) -> str:
    """Return the argument of a reply that is nothing but a COMPLETE() call.

    A reply that hands the answer over as the call instead of as text is unwrapped here.

    Only a reply that is exactly one call with one string argument is unwrapped. Prose
    comes back untouched, and so does a call with anything else in it.

    Each form is tried twice: parse_tree() first, which resolves the escapes, and
    _WRAP_RE after it, which reads a call whose argument spans lines and is not a
    Python string at all.

    The whole reply is tried before the code extract_code() finds in it: an answer that
    quotes a fenced block of its own is not read as that block.

    Args:
        reply: The stage's reply to the harvest call.

    Returns:
        The string that was handed to the call, or the reply as it came.
    """
    for text in (reply.strip(), (extract_code(reply)[0] or "").strip()):
        match = _unwrap_text(text) if text else None
        if match is not None:
            return match
    return reply


def _unwrap_text(text: str) -> str | None:
    """Return the argument of a text that is exactly one COMPLETE() call.

    Args:
        text: The reply, or the code taken out of it.

    Returns:
        The string handed to the call, or None when this is not such a call.
    """
    tree = parse_tree(text)
    if tree is None:
        match = _WRAP_RE.match(text)
        return match.group("body") if match else None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    call = tree.body[0].value
    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id in END_CALL_DICT.values() and len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)):
        return call.args[0].value
    return None


def no_answer_reason(error: str, max_turn: int, call: str,
                     outcome: str | None = None) -> str:
    """Say why a stage handed nothing over, telling the three cases apart.

    A run that stopped because a call could not be made, a run that ended itself with
    nothing in the call, and a run that used every turn are told apart by "error" and
    "outcome".

    Args:
        error: RLM.run()'s "error", "" when the loop did not stop early.
        max_turn: That stage's turn limit, for the wording of a run that used them up.
        call: The call that would have ended the run, e.g. "COMPLETE".
        outcome: RLM.run()'s "outcome": the key of the call the stage ended itself
            with, or None for a stage that did not.

    Returns:
        The single clause, without a full stop.
    """
    if error:
        return f"the run stopped early: {error}"
    if outcome is not None:
        return f"{call}() was called with an empty answer"
    return f"no {call}() within {max_turn} turns"


def refuse_up_to(check: Callable[[object, int], str], refusal_max: int) -> Callable[[object], str]:
    """Build an on_complete guard that sends the ending call back a bounded number of times.

    Args:
        check: (the argument of the call that ended the run, how many times the call
            has been sent back so far) -> the refusal text, or "" to accept. It is
            asked only while the bound has not been reached.
        refusal_max: How many times the call may be sent back in one run. 0 accepts
            every call.

    Returns:
        on_complete(answer) -> str for RLM.run(): "" accepts, text refuses.
    """
    state_dict = {"refused": 0}

    def on_complete(answer: object) -> str:
        """Return check's refusal while the bound is not reached, else ""."""
        if state_dict["refused"] >= refusal_max:
            return ""
        refusal = check(answer, state_dict["refused"])
        if refusal:
            state_dict["refused"] += 1
        return refusal

    return on_complete


def refuse_once(check: Callable[[object], str]) -> Callable[[object], str]:
    """Build an on_complete guard that sends the ending call back at most once.

    Args:
        check: (the argument of the call that ended the run) -> the refusal text, or
            "" to accept. It is asked only until it refuses once.

    Returns:
        on_complete(answer) -> str for RLM.run(): "" accepts, text refuses.
    """
    return refuse_up_to(lambda answer, _: check(answer), 1)


def ask_whole(llm: Callable, prompt: str | list[tuple[str, bool]],
              retry_count: int | None = None) -> tuple[str, bool]:
    """Make one call that runs no code, asking again more briefly while the reply is cut.

    Args:
        llm: The callable make_llm() built, which carries the finish reason on
            last_finish_reason. A callable without it is taken as never cut. Its whole()
            is called when it has one, and the reply is not ended where a turn's would
            be.
        prompt: What to send: a string, or the blocks build_prompt_block_list() returns.
        retry_count: How many times to ask again. None reads CUT_REPLY_MAX_RETRY.

    Returns:
        (the last reply, whether it was still cut off at the output limit).
    """
    retry_count = config.CUT_REPLY_MAX_RETRY if retry_count is None else retry_count
    send = getattr(llm, "whole", llm)
    note = ""
    for _ in range(retry_count + 1):
        reply = send(prompt + note if isinstance(prompt, str) else prompt_of(prompt, note))
        if getattr(llm, "last_finish_reason", None) != "length":
            return reply, False
        note = BRIEF_RETRY_NOTE
    logger.warning("the reply was still cut off at the output limit after %d more "
                   "call(s)", retry_count)
    return reply, True


def _send_with_refold(stage: RLM, history_list: list[tuple[str, str]],
                      block_list_of: Callable[[], list[tuple[str, bool]]],
                      send: Callable[[list[tuple[str, bool]]], object],
                      is_call_failure: bool = False) -> object:
    """Build a prompt and send it, folding harder and sending once more on a refusal.

    A prompt the provider refuses for its length lowers the stage's history budget
    (RLM.shrink_history_budget), and the prompt is built and sent a second time. A
    second refusal is raised.

    Args:
        stage: The stage's RLM.
        history_list: The stage's history, unfolded.
        block_list_of: Builds the prompt's blocks; called once before each send.
        send: Sends the blocks and returns the reply.
        is_call_failure: True raises the second refusal as CallFailure, False as the
            PromptTooLong itself.

    Returns:
        What send returns.

    Raises:
        PromptTooLong: When the second prompt is refused for its length too and
            is_call_failure is False.
        CallFailure: The same, when is_call_failure is True.
    """
    for is_last in (False, True):
        block_list = block_list_of()
        try:
            return send(block_list)
        except PromptTooLong as e:
            if is_last:
                if is_call_failure:
                    raise CallFailure(str(e)) from e
                raise
            stage.shrink_history_budget(history_list)


def need_late_answer(result_dict: dict, on: bool = True) -> bool:
    """Say whether a stage that handed nothing over is asked once more.

    A run that stopped because a call could not be made is not asked.

    Args:
        result_dict: RLM.run()'s return value.
        on: The stage's own setting for the extra call (a *_LATE_ANSWER setting).

    Returns:
        True when on is True, the stage used every turn without ending itself, and the
        loop did not stop early. A stage that ended itself is not asked, whatever it
        handed over.
    """
    return on and result_dict.get("outcome") is None and not result_dict.get("error")


def late_answer(stage: RLM, request: str, var_doc: str, history_list: list[tuple[str, str]],
                note: str, close_line: str) -> str:
    """Ask a stage that used every turn for its answer, in one call that runs nothing.

    It covers a stage that reads to its last turn without ever calling COMPLETE(). The
    call starts no REPL and runs no code.

    The history is folded the same way as on every other turn: this call sees what the
    last turn saw. A prompt the provider refuses for its length is folded harder and
    sent once more, as a turn's is (_send_with_refold).

    Args:
        stage: The stage's RLM. Its llm, instructions, tools and name are reused: the
            model sees exactly what it saw on its last turn.
        request: The request as the stage saw it.
        var_doc: The variable docs the stage saw.
        history_list: The stage's history; what it built is in front of it. The
            "## Turn i/n" line names the last turn in it, and stage.max_turn when the
            history is shorter than that.
        note: The turn note explaining that the turns are gone and what to answer with.
        close_line: The last line of the prompt, replacing "write the next block".

    Returns:
        The reply. Nothing is parsed here beyond unwrapping a reply that is the
        COMPLETE() call itself rather than its argument. A reply cut off at the output
        limit is asked for again (ask_whole), and one still cut after that ends with
        CUT_END_NOTE.
    """
    last_turn = max(stage.max_turn, len(history_list))
    reply, is_cut = _send_with_refold(
        stage, history_list,
        lambda: build_prompt_block_list(
            stage.instruction, stage.tool_dict, var_doc, request,
            stage.fold(history_list),
            last_turn, last_turn, turn_note=note, close_line=close_line),
        lambda block_list: ask_whole(stage.llm, block_list))
    # A reply still cut after the retries is handed over with the marker, which says it
    # is not the whole
    answer = unwrap_end_call(reply) + (CUT_END_NOTE if is_cut else "")
    # The one call that is not a turn. It is what the answer comes from when the turns
    # ran out, and it goes in the log after the last turn the stage ran
    turnlog.record(stage.stage, len(history_list) + 1, "", f"harvested: {answer}",
                   usage_dict=getattr(stage.llm, "last_usage_dict", None), call=stage.call)
    return answer


def run_stage(stage: RLM, request: str, variable_dict: dict[str, object], var_doc: str,
              late_note: str, late_close: str, late_on: bool = True,
              late_state: Callable[[int, int], str] | None = None,
              text_answer: bool = False, **run_kw: object) -> dict:
    """Run one stage to its answer, asking once more when it used every turn.

    The loop runs, and with late_on True, a stage that did not end itself and did not
    stop early is asked once, in a call that runs nothing, for what it has
    (late_answer). That call's note starts with the stage's state block where the stage
    has one. A call that fails leaves the stage without an answer and says why in
    "error", as a turn's call does.

    Args:
        stage: The stage's RLM.
        request: The request the stage is given.
        variable_dict: The variables to place in the REPL.
        var_doc: The description of those variables.
        late_note: The turn note of the one extra call.
        late_close: The last line of that call's prompt.
        late_on: The stage's own setting for the extra call, as need_late_answer()
            takes it.
        late_state: The stage's state block as a turn_note(turn, max_turn) -> str,
            asked for the turn after the last one the stage ran and placed in front of
            late_note. None places nothing.
        text_answer: True for a stage whose answer is what it says in its own words:
            whatever a stage that ended itself handed over is turned into text, None
            into "". False leaves the object as it was handed over.
        **run_kw: What RLM.run() takes besides the three above (turn_note,
            on_complete, after_turn).

    Returns:
        RLM.run()'s return value with "harvested" added: whether "answer" came from the
        extra call rather than from COMPLETE(). When the extra call fails, "answer"
        stays None, "harvested" False, and "error" holds the reason.
    """
    out_dict = stage.run(request=request, variable_dict=variable_dict, var_doc=var_doc,
                         **run_kw)
    out_dict["harvested"] = False
    if text_answer and out_dict.get("outcome") is not None:
        answer = out_dict["answer"]
        out_dict["answer"] = "" if answer is None else text_of(answer)
    if need_late_answer(out_dict, late_on):
        logger.info("the %s stage ran out of turns; asking once for what it has",
                    stage.stage)
        state = late_state(out_dict["iters"] + 1, out_dict["iters"]) if late_state else ""
        try:
            out_dict["answer"] = late_answer(
                stage, request, var_doc, out_dict["history"],
                "\n\n".join(part for part in (state, late_note) if part), late_close)
        except Exception as e:                        # noqa: BLE001 - any failed call
            out_dict["error"] = f"{type(e).__name__}: {e}"
            logger.warning("the call made after the last turn of the %s stage did not "
                           "happen (%s), so the stage ends without an answer",
                           stage.stage, out_dict["error"])
            return out_dict
        out_dict["harvested"] = True
    return out_dict
