"""Say on stderr what a run is doing while it is going on.

A label names the agent a line belongs to, e.g. "[task 2/5] ". It is held per thread of
work: the log lines written there carry it, and so does each line a child process
started there under relay() writes to stdout or stderr. A step is a part of a run whose
start and end are logged with how long it took. A line written in TERM_LOG_FORMAT holds
the time since the run started, which a process reads from RUN_START_ENV when it is set.

at_terminal() says whether a question can be asked on the terminal, and ask() asks
one.

PipeOutput takes in what a started program writes, through pipes read as it runs.
"""

import collections
import contextlib
import contextvars
import getpass
import io
import logging
import os
import selectors
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import TextIO

from codetwine_loom.textcut import CUT_NOTE, term_text

logger = logging.getLogger(__name__)

# The label of the work going on in this context; "" outside an agent's own work
_LABEL = contextvars.ContextVar("label", default="")

# What guards a line handed on from a child process: two lines never share one write
_WRITE_LOCK = threading.Lock()

# The environment variable holding the time the run started (seconds since the epoch)
RUN_START_ENV = "CTLOOM_RUN_START"

# The format of a line on the terminal. "elapsed" is the time since the run started
TERM_LOG_FORMAT = "%(elapsed)s %(levelname)s %(name)s: %(message)s"

# The time this process started, used when RUN_START_ENV is not set
_PROCESS_START = time.time()

# Whether relay() writes a child's stdout to the file it is given; set by
# keep_child_stdout()
_CHILD_STDOUT_TO_FILE = False


class TermStream:
    """A text stream that hands on what it is given as textcut.term_text() writes it.

    Once the stream under it takes a write no longer, what is written after that is
    dropped. That is logged once, unless the reader of a pipe went away or the stream
    is closed. Everything but write(), writelines() and flush() is the stream's own.
    """

    def __init__(self, stream: TextIO, apart_fd: int | None = None) -> None:
        """Keep the stream written to.

        Args:
            stream: The stream every write goes on to.
            apart_fd: The descriptor of the stream this one takes the place of, when
                stream writes through a copy of that descriptor. None when what is
                written here goes through that descriptor itself, or through none.
        """
        self._stream = stream
        self._is_open = True
        self.apart_fd = apart_fd

    def _stop(self, error: Exception) -> None:
        """Take no write from now on.

        Why is logged when the error says the stream is still there.

        Args:
            error: What the stream raised.
        """
        self._is_open = False
        if isinstance(error, OSError) and not isinstance(error, BrokenPipeError):
            logger.warning("%s takes no more (%s), so what is written to it from here "
                           "on is dropped", getattr(self._stream, "name", "a stream"),
                           error)

    def write(self, text: str) -> int:
        """Write a text with what a terminal acts on written out (textcut.term_text).

        Args:
            text: The text.

        Returns:
            The length of text, whether or not the stream took it.
        """
        if self._is_open:
            try:
                self._stream.write(term_text(text))
            except (OSError, ValueError) as e:
                self._stop(e)
        return len(text)

    def writelines(self, line_list: Iterable[str]) -> None:
        """Write each text of line_list as write() does.

        Args:
            line_list: The texts.
        """
        for line in line_list:
            self.write(line)

    def flush(self) -> None:
        """Flush the stream, unless it has stopped taking what is written."""
        if self._is_open:
            try:
                self._stream.flush()
            except (OSError, ValueError) as e:
                self._stop(e)

    def __getattr__(self, name: str) -> object:
        """Return an attribute of the stream written to.

        Args:
            name: The attribute asked for.

        Returns:
            The stream's attribute of that name.
        """
        return getattr(self._stream, name)


def copy_fd(fd: int) -> int:
    """Return a copy of a descriptor, numbered above those of the three standard streams.

    Args:
        fd: The descriptor.

    Returns:
        The copy, which a process started from this one does not get.

    Raises:
        OSError: When the descriptor cannot be copied.
    """
    low_fd_list = []
    try:
        while (copy := os.dup(fd)) <= 2:
            low_fd_list.append(copy)
        return copy
    finally:
        for low_fd in low_fd_list:
            os.close(low_fd)


def _term_stream(stream: TextIO) -> TermStream:
    """Return the TermStream that takes the place of one of the two streams.

    A stream over a descriptor is flushed and written to from then on through a copy
    of that descriptor (copy_fd), opened with the stream's name, encoding and
    buffering: the lines of this process go where the stream went, whatever its
    descriptor is pointed at later. A stream with no descriptor is written to as it
    is, and so is one whose descriptor cannot be copied.

    Args:
        stream: sys.stdout or sys.stderr as it was found.

    Returns:
        The TermStream, its "apart_fd" the descriptor of stream when it writes through
        a copy of it.
    """
    try:
        fd = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return TermStream(stream)
    try:
        stream.flush()
        raw = io.FileIO(copy_fd(fd), "w")
    except (OSError, ValueError) as e:
        logger.debug("a stream is written to through its own descriptor (%s)", e)
        return TermStream(stream)
    raw.name = getattr(stream, "name", raw.name)
    is_write_through = bool(getattr(stream, "write_through", False))
    return TermStream(io.TextIOWrapper(
        raw if is_write_through else io.BufferedWriter(raw),
        encoding=getattr(stream, "encoding", None), errors="backslashreplace",
        line_buffering=bool(getattr(stream, "line_buffering", False)),
        write_through=is_write_through), apart_fd=fd)


def set_up_output() -> None:
    """Put a TermStream in place of standard output and of standard error.

    What is written to sys.stdout and sys.stderr from then on goes through the two: a
    character their encoding cannot hold is written as its backslash escape, a character
    a terminal acts on as "\\xNN", and what is written once the other end takes no more
    is dropped. Each writes through a descriptor of its own (_term_stream). A stream
    that is closed, or is a TermStream already, is left as it is.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is None or isinstance(stream, TermStream):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="backslashreplace")
            except (OSError, ValueError) as e:
                logger.debug("a stream could not be set to escape what it cannot "
                             "encode (%s)", e)
        setattr(sys, name, _term_stream(stream))


def apart_fd_list() -> list[int]:
    """Return the descriptors that can be pointed elsewhere.

    The lines of this process still go where they went when these are pointed elsewhere.

    Returns:
        The descriptor sys.stdout was found over and the one sys.stderr was, each
        only when the stream is now a TermStream that writes through a copy of it
        (set_up_output). A stream the process was started without gives none.
    """
    return [stream.apart_fd for stream in (sys.stdout, sys.stderr)
            if isinstance(stream, TermStream) and stream.apart_fd is not None]


def output_is_terminal() -> bool:
    """Say whether what this process prints goes to a terminal.

    Returns:
        True when sys.stdout is there and is a terminal.
    """
    return sys.stdout is not None and sys.stdout.isatty()


def at_terminal() -> bool:
    """Say whether a question can be asked on the terminal (ask).

    The input and the output have to be a terminal, and the three standard streams all
    have to be there.

    Returns:
        True when both sys.stdin and sys.stdout are a terminal. False for a process
        started with either of them, or with standard error, closed.
    """
    return sys.stdin is not None and sys.stderr is not None and sys.stdin.isatty() \
        and output_is_terminal()


def ask(question: str, is_secret: bool = False) -> str:
    """Ask one question on the terminal and return the line typed in answer.

    Ask it only when at_terminal() is True. The question is written to standard output,
    where the lines this process prints go, and the answer is read from standard input
    (input(), which reads nothing in a process without all three standard streams). A
    line that holds bytes that are not UTF-8, whether the reader keeps them as
    surrogates or fails to decode them, is asked for again, after a line saying so.

    Args:
        question: The question, written as it is with no newline after it.
        is_secret: Whether the answer is read without being shown as it is typed
            (getpass).

    Returns:
        The line typed, without its newline.

    Raises:
        EOFError: When the input ends before a line.
    """
    while True:
        sys.stdout.write(question)
        sys.stdout.flush()
        try:
            answer = getpass.getpass("") if is_secret else input("")
        except UnicodeDecodeError:
            answer = None
        if answer is not None and not any("\ud800" <= char <= "\udfff" for char in answer):
            return answer
        sys.stdout.write("The answer holds bytes that are not UTF-8. Type it again\n")


def keep_child_stdout(on: bool) -> None:
    """Say whether relay() writes a child's stdout to the file it is given from now on.

    Args:
        on: True writes it to the file, False to this process's stdout.
    """
    global _CHILD_STDOUT_TO_FILE
    _CHILD_STDOUT_TO_FILE = on


def mark_run_start() -> bool:
    """Set RUN_START_ENV to now, unless a process that started this one set it.

    Returns:
        True when this call set it.
    """
    if RUN_START_ENV in os.environ:
        return False
    os.environ[RUN_START_ENV] = str(time.time())
    return True


def forget_run_start() -> None:
    """Take RUN_START_ENV out of this process's environment."""
    os.environ.pop(RUN_START_ENV, None)


def run_start() -> float:
    """Return the time the run started (seconds since the epoch).

    Returns:
        RUN_START_ENV's value, or the time this process started when it is unset or
        not a number.
    """
    try:
        return float(os.environ[RUN_START_ENV])
    except (KeyError, ValueError):
        return _PROCESS_START


def clock_text(second: float) -> str:
    """Return a length of time as hours, minutes and seconds.

    Examples:
        42.4  -> "0:00:42"
        3725  -> "1:02:05"

    Args:
        second: The length of time in seconds. Below 0 counts as 0.

    Returns:
        "H:MM:SS".
    """
    total = max(int(second), 0)
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def label_text() -> str:
    """Return the label of the work going on in this context.

    Returns:
        The label, e.g. "[task 2/5] ", or "" when the context has none.
    """
    return _LABEL.get()


@contextlib.contextmanager
def label(text: str) -> Iterator[None]:
    """Set the label for the work inside the block, and put the one before it back.

    Args:
        text: The label, with its trailing space, e.g. "[task 2/5] ".

    Yields:
        Nothing.
    """
    token = _LABEL.set(text)
    try:
        yield
    finally:
        _LABEL.reset(token)


class LabelFormatter(logging.Formatter):
    """A formatter that puts the current label at the head of every line it writes.

    A record it formats is given "elapsed", the time since the run started as
    clock_text() writes it, for TERM_LOG_FORMAT to use.
    """

    def __init__(self, fmt: str, head: str = "") -> None:
        """Keep the label used where the context has none.

        Args:
            fmt: The line format, as logging.Formatter takes it.
            head: The label of a process that works for one agent throughout, whose
                threads each start without the context's label. "" has none.
        """
        super().__init__(fmt)
        self.head = head

    def format(self, record: logging.LogRecord) -> str:
        """Format the record, then put the label before each of its lines.

        Args:
            record: The record to format.

        Returns:
            The text as logging.Formatter builds it, unchanged when the label is "".
        """
        record.elapsed = clock_text(record.created - run_start())
        text = super().format(record)
        head = label_text() or self.head
        if not head:
            return text
        return "\n".join(head + line for line in text.split("\n"))


class TermFilter(logging.Filter):
    """Let through what the terminal shows: this module's lines and warnings or worse."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Say whether a record is written to the terminal.

        Args:
            record: The record.

        Returns:
            True for a record of this module's logger and for one at WARNING or above.
        """
        return record.name == logger.name or record.levelno >= logging.WARNING


def note(message: str, *arg_tuple: object) -> None:
    """Log one line of how a run is going, which the terminal shows.

    Args:
        message: The line, with %-style placeholders.
        *arg_tuple: The values for the placeholders.
    """
    logger.info(message, *arg_tuple)


def elapsed_text(second: float) -> str:
    """Return a length of time as a short text.

    Examples:
        42.4   -> "42s"
        372    -> "6m12s"
        3725   -> "1h02m"

    Args:
        second: The length of time in seconds.

    Returns:
        Seconds below a minute, minutes and seconds below an hour, hours and minutes
        above it.
    """
    total = int(second)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{total % 3600 // 60:02d}m"


@contextlib.contextmanager
def step(name: str) -> Iterator[None]:
    """Log the start and the end of one part of a run, with how long it took.

    An exception is logged as the part having stopped and is raised again.

    Args:
        name: What the part does, e.g. "split the request".

    Yields:
        Nothing.
    """
    logger.info("%s: start", name)
    start = time.monotonic()
    try:
        yield
    except BaseException:
        logger.info("%s: stopped (%s)", name, elapsed_text(time.monotonic() - start))
        raise
    logger.info("%s: end (%s)", name, elapsed_text(time.monotonic() - start))


@contextlib.contextmanager
def quiet(name: str) -> Iterator[None]:
    """Hold back what one logger writes below a warning, inside the block.

    Nothing is held back when the root logger carries DEBUG lines. The level the logger
    had is put back when the block ends.

    Args:
        name: The logger's name, e.g. the top-level name of a library.

    Yields:
        Nothing.
    """
    target = logging.getLogger(name)
    level = target.level
    if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
        target.setLevel(logging.WARNING)
    try:
        yield
    finally:
        target.setLevel(level)


# How many bytes of a pipe are read at a time
_PIPE_BLOCK_BYTE = 1 << 16

# How many bytes of one line of a pipe are handed on, and what is written after a line
# cut there
_MAX_LINE_BYTE = 1 << 16
_LINE_CUT_MARK = " ... (the rest of this line was left out)"


def _line_piece_list(block: bytes, is_after_cr: bool) -> list[bytes]:
    """Split bytes a pipe carried at the line ends in them.

    Args:
        block: The bytes, in the order they were written.
        is_after_cr: Whether the bytes before them ended in "\\r". A "\\n" the block
            starts with is then the end of that "\\r\\n" and is dropped.

    Returns:
        The pieces between the line ends ("\\n", "\\r" or "\\r\\n"). Every piece but the
        last was followed by a line end.
    """
    if is_after_cr and block.startswith(b"\n"):
        block = block[1:]
    return block.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")


def _pipe_line(read_fd: int) -> Iterator[str]:
    """Yield what a pipe carries, line by line, until its other end is closed.

    The pipe is read _PIPE_BLOCK_BYTE at a time. A line ends at "\\n", "\\r" or
    "\\r\\n". One longer than _MAX_LINE_BYTE is yielded once it is that long, cut there
    with _LINE_CUT_MARK after it, and what follows up to its end is read and dropped.

    Args:
        read_fd: The reading end of the pipe.

    Yields:
        Each line without its end, decoded as UTF-8 with any other byte replaced.
    """
    part, is_over, is_after_cr = b"", False, False
    while True:
        block = os.read(read_fd, _PIPE_BLOCK_BYTE)
        if not block:
            break
        piece_list = _line_piece_list(block, is_after_cr)
        is_after_cr = block.endswith(b"\r")
        for index, piece in enumerate(piece_list):
            if not is_over:
                part += piece
                if len(part) > _MAX_LINE_BYTE:
                    yield part[:_MAX_LINE_BYTE].decode("utf-8", "replace") + _LINE_CUT_MARK
                    part, is_over = b"", True
            # Every piece but the last was followed by the end of its line
            if index < len(piece_list) - 1:
                if not is_over:
                    yield part.decode("utf-8", "replace")
                part, is_over = b"", False
    if part:
        yield part.decode("utf-8", "replace")


def _hand_on(read_fd: int, head: str, target: TextIO) -> None:
    """Write every line read from a pipe to a stream, each with the label before it.

    The pipe is read until its other end is closed (_pipe_line). Once the stream takes
    a line no longer, the lines read after it are dropped.

    Args:
        read_fd: The reading end of the pipe. It is closed when the other end is.
        head: The label, or "".
        target: The stream the lines go to.
    """
    is_open = True
    try:
        for line in _pipe_line(read_fd):
            if not is_open:
                continue
            with _WRITE_LOCK:
                try:
                    target.write(head + line + "\n")
                    target.flush()
                except (OSError, ValueError):
                    is_open = False
    finally:
        os.close(read_fd)


@contextlib.contextmanager
def relay(wait_sec: float | None, out_path: str = "") -> Iterator[tuple[int, int]]:
    """Hand on what a child process writes, line by line, under the current label.

    What is written to the first descriptor goes to this process's stdout, or to
    out_path while keep_child_stdout() has it on, and what is written to the second to
    its stderr. A process started with one of the two closed hands that stream's lines
    on to os.devnull, and a stream that stops taking lines while the child runs has
    the rest of them dropped: the child's pipes are read either way. The label is read
    when the block is entered.

    Args:
        wait_sec: How long each stream is waited for once the block ends (seconds).
            None waits until the last process holding the pipe has closed it.
        out_path: The file the child's stdout is appended to instead of this process's
            stdout while keep_child_stdout() has it on. "" writes it to stdout.

    Yields:
        (the descriptor to pass as the child's stdout, the one to pass as its stderr).
    """
    head = label_text()
    with contextlib.ExitStack() as stack:
        out_target: TextIO | None = sys.stdout
        if out_path and _CHILD_STDOUT_TO_FILE:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            out_target = stack.enter_context(open(out_path, "a", encoding="utf-8"))
        thread_list, write_fd_list = [], []
        for target in (out_target, sys.stderr):
            if target is None:
                target = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
            read_fd, write_fd = os.pipe()
            thread = threading.Thread(target=_hand_on, args=(read_fd, head, target),
                                      daemon=True)
            thread.start()
            thread_list.append(thread)
            write_fd_list.append(write_fd)
        try:
            yield write_fd_list[0], write_fd_list[1]
        finally:
            for write_fd in write_fd_list:
                os.close(write_fd)
            for thread in thread_list:
                thread.join(wait_sec)


# How long the reading of a pipe goes on once this process wrote its last to it, when
# another process still holds it open (seconds)
PIPE_END_WAIT_SEC = 0.5

# How many bytes are still read of a pipe whose reading is told to stop
_PIPE_STOP_BYTE = 1 << 20

# What end() tells the thread that reads: to stop there, or to read on and keep nothing
_STOP_MARK, _DROP_MARK = b"s", b"d"


# The longest line of what a pipe carried that is handed to "keep_line" (bytes)
_KEEP_LINE_MAX_BYTE = 1 << 16


class _PipeKeep:
    """What is kept of the bytes one pipe carried.

    That is the first of them, the last, and the lines between the two that "keep_line"
    keeps.
    """

    def __init__(self, head_byte: int, tail_byte: int,
                 keep_line: Callable[[bytes], int] | None = None,
                 keep_byte: int = 0) -> None:
        """Start with nothing kept.

        Args:
            head_byte: How many of the first bytes are kept.
            tail_byte: How many of the last bytes are kept.
            keep_line: Called with each line that lies between the first bytes and the
                last, without its line end, in the order they were written. It returns
                how many lines are kept from that line on: 0 for none, 1 for the line
                alone. A line longer than _KEEP_LINE_MAX_BYTE is not handed to it and
                is not kept. With it, the first bytes end and the last bytes start at
                a line end whenever bytes between them were dropped, where they hold
                one. None keeps no line.
            keep_byte: How many bytes of such lines are kept, each counted with one
                byte for its line end. Once a line does not fit, no later one is kept.
        """
        self.head_byte, self.tail_byte = head_byte, tail_byte
        self.keep_line, self.keep_byte = keep_line, keep_byte
        self.head = bytearray()
        # The blocks the last tail_byte bytes are in, and the length of all of them
        self.tail_list: collections.deque[bytes] = collections.deque()
        self.tail_size = 0
        self.total = 0
        # The lines kept of what lies between the head and the tail, "\n" after each
        self.line_keep = bytearray()
        # Whether a byte went past the head and out of the tail, and whether the kept
        # lines have reached keep_byte
        self.is_past = False
        self.is_full = False
        # The line being read of what went past: its bytes so far and whether it is
        # longer than _KEEP_LINE_MAX_BYTE
        self.line_start = bytearray()
        self.is_long_line = False
        # Whether the last byte that went past was "\r", and whether it was no line
        # end
        self.is_after_cr = False
        self.is_mid_line = False
        # How many of the next lines are kept whatever keep_line returns for them
        self.keep_count = 0
        # Whether the last bytes have been cut to tail_byte (_cut_tail)
        self.is_tail_cut = False

    def add(self, block: bytes) -> None:
        """Take the bytes that came next.

        Args:
            block: The bytes.
        """
        self.total += len(block)
        room = 0 if self.is_past else self.head_byte - len(self.head)
        if room > 0:
            self.head += block[:room]
            block = block[room:]
        if not block:
            return
        if self.tail_byte <= 0:
            self._pass(block)
            return
        self.tail_list.append(block)
        self.tail_size += len(block)
        while self.tail_size - len(self.tail_list[0]) >= self.tail_byte:
            self.tail_size -= len(self.tail_list[0])
            self._pass(self.tail_list.popleft())

    def _pass(self, block: bytes) -> None:
        """Take the bytes that went past the head and out of the tail next.

        The lines of them keep_line says to keep are kept. The line the head ends in the
        middle of is taken out of the head and read with the bytes that follow it. A
        head that holds no line end stays whole, and what follows it up to the next line
        end is not read.

        Args:
            block: The bytes, in the order they were written.
        """
        if self.keep_line is None:
            return
        if not self.is_past:
            self.is_past = True
            line_at = max(self.head.rfind(b"\n"), self.head.rfind(b"\r")) + 1
            self.line_start = self.head[line_at:] if line_at else bytearray()
            self.is_long_line = not line_at
            self.is_after_cr = self.head.endswith(b"\r")
            del self.head[line_at or len(self.head):]
        if not block:
            return
        is_after_cr, self.is_after_cr = self.is_after_cr, block.endswith(b"\r")
        self.is_mid_line = not block.endswith((b"\n", b"\r"))
        if self.is_full:
            return
        line_list = _line_piece_list(block, is_after_cr)
        if len(line_list) > 1:
            line_list[0] = bytes(self.line_start) + line_list[0]
            self.line_start.clear()
            for line in line_list[:-1]:
                is_long = self.is_long_line or len(line) > _KEEP_LINE_MAX_BYTE
                self.is_long_line = False
                self._take_line(None if is_long else line)
        if not self.is_long_line:
            self.line_start += line_list[-1]
            if len(self.line_start) > _KEEP_LINE_MAX_BYTE:
                self.line_start.clear()
                self.is_long_line = True

    def _take_line(self, line: bytes | None) -> None:
        """Keep one line that went past the head and out of the tail, or drop it.

        Args:
            line: The line without its line end, or None for a line longer than
                _KEEP_LINE_MAX_BYTE.
        """
        if self.is_full:
            return
        if line is not None:
            self.keep_count = max(self.keep_count, self.keep_line(line))
        if self.keep_count <= 0:
            return
        self.keep_count -= 1
        if line is None:
            return
        if len(self.line_keep) + len(line) + 1 > self.keep_byte:
            self.is_full = True
            return
        self.line_keep += line
        self.line_keep += b"\n"

    def _cut_tail(self) -> bytes:
        """Cut the last bytes to tail_byte and return them.

        The bytes in front of them go the way the bytes that left the tail went
        (_pass). When bytes went that way and the last bytes start in the middle of a
        line, so does that line up to its end: the last bytes start at a line. Last
        bytes that hold no line end stay whole.

        Returns:
            The last bytes, as they are kept from then on.
        """
        tail = b"".join(self.tail_list)
        if not self.is_tail_cut:
            self.is_tail_cut = True
            cut_at = max(len(tail) - self.tail_byte, 0)
            if cut_at:
                self._pass(tail[:cut_at])
                tail = tail[cut_at:]
            end_list = [at for at in (tail.find(b"\n"), tail.find(b"\r")) if at >= 0]
            if self.is_past and end_list:
                if self.is_mid_line:
                    line_at = min(end_list) + 1
                    line_at += tail[line_at - 1:line_at + 1] == b"\r\n"
                else:
                    line_at = int(self.is_after_cr and tail.startswith(b"\n"))
                self._pass(tail[:line_at])
                tail = tail[line_at:]
            self.tail_list = collections.deque([tail])
            self.tail_size = len(tail)
        return tail

    def text(self, encoding: str, errors: str) -> str:
        """Return what is kept as text.

        Args:
            encoding: The encoding the bytes are read as.
            errors: What is done with a byte the encoding cannot read, as bytes.decode()
                names it.

        Returns:
            The first bytes, the lines kept of what lies between, each ending in "\\n",
            then the last bytes. When bytes between the first and the last were dropped,
            CUT_NOTE stands after the first bytes with how many, on a line of its own.
        """
        tail = self._cut_tail() if self.tail_byte > 0 else b""
        drop_count = self.total - len(self.head) - len(self.line_keep) - len(tail)
        text = self.head.decode(encoding, errors)
        if drop_count > 0:
            text += CUT_NOTE.format(omitted=drop_count) + (
                "\n" if self.line_keep or tail else "")
        return text + self.line_keep.decode(encoding, errors) + tail.decode(encoding, errors)


class PipeOutput:
    """Pipes that take in what processes write, read on one thread into memory.

    Each descriptor of "write_fd_list" is the writing end of a pipe of its own, to be
    handed to what writes. Of what each pipe carries, the first head_byte bytes and the
    last tail_byte are kept. The bytes between them are read, the lines of them that
    "keep_line" keeps are kept, and the rest is dropped. The reading goes on until
    end(), which the "with" block calls on its way out.
    """

    def __init__(self, pipe_count: int, head_byte: int, tail_byte: int = 0,
                 keep_line: Callable[[bytes], int] | None = None,
                 keep_byte: int = 0) -> None:
        """Open the pipes and start reading them.

        Args:
            pipe_count: How many pipes there are.
            head_byte: How many of the first bytes of each pipe are kept.
            tail_byte: How many of the last bytes of each pipe are kept.
            keep_line: What says which lines between the first bytes and the last are
                kept, as _PipeKeep takes it. It is called on the thread that reads.
                None keeps none.
            keep_byte: How many bytes of such lines are kept of each pipe.

        Raises:
            OSError: When a pipe cannot be opened or the thread cannot be started.
                None is left open.
        """
        self.write_fd_list: list[int] = []
        # {the reading end of a pipe: what is kept of it}, in the order of write_fd_list
        self._keep_dict: dict[int, _PipeKeep] = {}
        self._is_end = False
        try:
            for _ in range(pipe_count):
                read_fd, write_fd = os.pipe()
                self.write_fd_list.append(write_fd)
                self._keep_dict[read_fd] = _PipeKeep(head_byte, tail_byte, keep_line,
                                                     keep_byte)
                os.set_blocking(read_fd, False)
            # The pipe end() writes its mark to, and the thread reads it from
            self._stop_read_fd, self._stop_write_fd = os.pipe()
        except OSError:
            for fd in (*self.write_fd_list, *self._keep_dict):
                os.close(fd)
            raise
        # Set once what is kept changes no more, and whether the thread reads on then
        self._keep_event = threading.Event()
        self._is_read_on = False
        self._thread = threading.Thread(target=self._read, daemon=True)
        try:
            self._thread.start()
        except RuntimeError as e:
            for fd in (*self.write_fd_list, *self._keep_dict, self._stop_read_fd,
                       self._stop_write_fd):
                os.close(fd)
            raise OSError(str(e)) from e

    def __enter__(self) -> "PipeOutput":
        """Enter the block.

        Returns:
            Itself.
        """
        return self

    def __exit__(self, *error_tuple: object) -> None:
        """End the reading (end), however the block ended."""
        self.end(PIPE_END_WAIT_SEC)

    def close_write(self) -> None:
        """Close this process's own writing end of every pipe."""
        while self.write_fd_list:
            os.close(self.write_fd_list.pop())

    def end(self, wait_sec: float, is_drop_after: bool = False) -> None:
        """Close this process's writing ends and end the keeping of what the pipes carry.

        A pipe no other process holds open has been read to its end by then. One that
        another process still holds is read for wait_sec more, and then for what it
        holds at that moment, up to _PIPE_STOP_BYTE. A second call does nothing.

        Args:
            wait_sec: How long the other end of the pipes is waited for (seconds).
            is_drop_after: What becomes of a pipe another process still holds after
                that. False closes it. True has it read on, keeping nothing, until that
                process closes it: a write to it goes on working. The thread that read
                has ended when this returns, unless it reads such a pipe on.
        """
        if self._is_end:
            return
        self._is_end = True
        try:
            self.close_write()
            self._keep_event.wait(wait_sec)
        finally:
            # A thread that ended on its own reads no mark
            with contextlib.suppress(OSError):
                os.write(self._stop_write_fd, _DROP_MARK if is_drop_after else _STOP_MARK)
            os.close(self._stop_write_fd)
            self._keep_event.wait()
            if not self._is_read_on:
                self._thread.join()

    def text(self, index: int, encoding: str, errors: str) -> str:
        """Return what was kept of one pipe, as text (_PipeKeep.text). Call it after end().

        Args:
            index: Which pipe, as its place in write_fd_list was.
            encoding: The encoding the bytes are read as.
            errors: What is done with a byte the encoding cannot read.

        Returns:
            The text _PipeKeep.text() makes of that pipe.
        """
        return list(self._keep_dict.values())[index].text(encoding, errors)

    def _read(self) -> None:
        """Read the pipes until end() writes its mark, then do what the mark says.

        Runs in a thread of its own. A pipe closed at its other end is closed here.
        _STOP_MARK has what the other pipes hold at that moment read, and closes them.
        _DROP_MARK has them read on to their end after that, with none of it kept.
        """
        open_fd_set = set(self._keep_dict)
        mark = b""
        try:
            with selectors.DefaultSelector() as selector:
                for fd in (self._stop_read_fd, *open_fd_set):
                    selector.register(fd, selectors.EVENT_READ)
                while not mark:
                    for key, _ in selector.select():
                        if key.fd == self._stop_read_fd:
                            mark = os.read(key.fd, 1) or _STOP_MARK
                        elif self._read_block(key.fd, True) == 0:
                            self._close_pipe(selector, open_fd_set, key.fd)
                    if not open_fd_set:
                        self._keep_event.set()
                selector.unregister(self._stop_read_fd)
                for read_fd in list(open_fd_set):
                    room, count = _PIPE_STOP_BYTE, None
                    while room > 0 and (count := self._read_block(read_fd, True)):
                        room -= count
                    if count == 0:
                        self._close_pipe(selector, open_fd_set, read_fd)
                self._is_read_on = bool(open_fd_set) and mark == _DROP_MARK
                self._keep_event.set()
                while self._is_read_on and open_fd_set:
                    for key, _ in selector.select():
                        if self._read_block(key.fd, False) == 0:
                            self._close_pipe(selector, open_fd_set, key.fd)
        finally:
            self._keep_event.set()
            for fd in (self._stop_read_fd, *open_fd_set):
                os.close(fd)

    @staticmethod
    def _close_pipe(selector: selectors.BaseSelector, open_fd_set: set[int],
                    read_fd: int) -> None:
        """Close the reading end of a pipe that is at its end.

        Args:
            selector: What waits for the pipes. The pipe is taken out of it.
            open_fd_set: The reading ends still open. The pipe is taken out of it.
            read_fd: The reading end of the pipe.
        """
        selector.unregister(read_fd)
        open_fd_set.discard(read_fd)
        os.close(read_fd)

    def _read_block(self, read_fd: int, is_keep: bool) -> int | None:
        """Read what one pipe holds, up to _PIPE_BLOCK_BYTE.

        Args:
            read_fd: The reading end of the pipe.
            is_keep: Whether the bytes are kept.

        Returns:
            How many bytes were read: 0 at the end of the pipe, None when it holds
            nothing now.
        """
        try:
            block = os.read(read_fd, _PIPE_BLOCK_BYTE)
        except BlockingIOError:
            return None
        if is_keep and block:
            self._keep_dict[read_fd].add(block)
        return len(block)
