"""The label a line is written under, the steps of a run, and the lines a child
process writes being handed on."""

import io
import logging
import os
import shlex
import subprocess
import sys
import threading
import time

import pytest

from codetwine_loom import progress, textcut


def _record(message: str) -> logging.LogRecord:
    """Return an INFO record holding the message."""
    return logging.LogRecord("name", logging.INFO, __file__, 1, message, None, None)


# == The label ==============================================
def test_the_formatter_leaves_a_line_as_it_is_without_a_label():
    assert progress.LabelFormatter("%(levelname)s %(message)s").format(_record("hi")) \
        == "INFO hi"


def test_the_formatter_puts_the_label_before_every_line():
    formatter = progress.LabelFormatter("%(levelname)s %(message)s")
    with progress.label("[task 2/5] "):
        assert formatter.format(_record("one\ntwo")) \
            == "[task 2/5] INFO one\n[task 2/5] two"


def test_a_formatter_given_a_label_uses_it_in_a_thread_that_has_none():
    formatter = progress.LabelFormatter("%(message)s", head="[task 2/5] ")
    line_list = []
    thread = threading.Thread(target=lambda: line_list.append(formatter.format(_record("hi"))))
    thread.start()
    thread.join()
    assert line_list == ["[task 2/5] hi"]


def test_the_label_before_the_block_is_put_back():
    with progress.label("[task 1/2] "):
        with progress.label("[task 1/2 try 2] "):
            assert progress.label_text() == "[task 1/2 try 2] "
        assert progress.label_text() == "[task 1/2] "
    assert progress.label_text() == ""


def test_the_label_is_put_back_when_the_block_raises():
    with pytest.raises(ValueError):
        with progress.label("[task 1/2] "):
            raise ValueError
    assert progress.label_text() == ""


# == How long something took ================================
@pytest.mark.parametrize("second, text", [
    (0, "0s"), (42.9, "42s"), (59.9, "59s"), (60, "1m00s"), (372, "6m12s"),
    (3599, "59m59s"), (3600, "1h00m"), (3725, "1h02m")])
def test_elapsed_text(second, text):
    assert progress.elapsed_text(second) == text


# == A step =================================================
def test_a_step_logs_its_start_and_its_end(caplog):
    with caplog.at_level("INFO", logger="codetwine_loom.progress"):
        with progress.step("split the request"):
            pass
    assert [r.getMessage() for r in caplog.records] \
        == ["split the request: start", "split the request: end (0s)"]


def test_a_step_that_raises_is_logged_as_stopped_and_raises(caplog):
    with caplog.at_level("INFO", logger="codetwine_loom.progress"):
        with pytest.raises(RuntimeError):
            with progress.step("run the tasks"):
                raise RuntimeError("no engine")
    assert [r.getMessage() for r in caplog.records] \
        == ["run the tasks: start", "run the tasks: stopped (0s)"]


# == Holding a library's lines back =========================
def test_quiet_holds_back_what_is_below_a_warning_and_puts_the_level_back(caplog):
    library = logging.getLogger("some_library.pipeline")
    with caplog.at_level("INFO"):
        with progress.quiet("some_library"):
            library.info("OK: a.py")
            library.warning("could not read b.py")
        library.info("after")
    assert [r.getMessage() for r in caplog.records] == ["could not read b.py", "after"]
    assert logging.getLogger("some_library").level == logging.NOTSET


def test_quiet_holds_nothing_back_when_every_line_is_asked_for(caplog):
    with caplog.at_level("DEBUG"):
        with progress.quiet("some_library"):
            logging.getLogger("some_library.pipeline").info("OK: a.py")
    assert [r.getMessage() for r in caplog.records] == ["OK: a.py"]


# == Handing on what a child process writes =================
def _run_child(code: str, wait_sec: float | None = 10) -> None:
    """Run a Python child process whose stdout and stderr are handed on."""
    with progress.relay(wait_sec) as (out_fd, err_fd):
        subprocess.run([sys.executable, "-c", code], check=True, stdout=out_fd,
                       stderr=err_fd, timeout=30)


def test_each_stream_is_handed_on_to_its_own_under_the_label(capsys):
    with progress.label("[task 1/2] "):
        _run_child("import sys; print('to out'); print('to err', file=sys.stderr)")
    out, err = capsys.readouterr()
    assert out == "[task 1/2] to out\n"
    assert err == "[task 1/2] to err\n"


def test_lines_are_handed_on_as_they_are_without_a_label(capsys):
    _run_child("print('branch: sandbox/x'); print('verdict: unverified')")
    assert capsys.readouterr().out == "branch: sandbox/x\nverdict: unverified\n"


def test_a_last_line_without_a_newline_is_given_one(capsys):
    with progress.label("[question 2/3] "):
        _run_child("import sys; sys.stdout.write('one\\ntwo')")
    assert capsys.readouterr().out == "[question 2/3] one\n[question 2/3] two\n"


def test_bytes_that_are_not_text_do_not_stop_the_lines_after_them(capsys):
    _run_child("import sys; sys.stdout.buffer.write(b'\\xff bad\\nfine\\n')")
    assert capsys.readouterr().out.endswith("fine\n")


def test_a_line_longer_than_the_limit_is_cut_and_the_lines_after_it_are_handed_on(capsys):
    _run_child("import sys; sys.stdout.write('a' * 200000 + '\\nnext\\n' + 'b' * 70000)")
    cut = progress._LINE_CUT_MARK
    assert capsys.readouterr().out == (
        "a" * progress._MAX_LINE_BYTE + cut + "\nnext\n"
        + "b" * progress._MAX_LINE_BYTE + cut + "\n")


@pytest.mark.parametrize("block_byte", [1, 2, 3, 4, 1 << 16])
def test_a_line_ends_where_the_pipe_carries_any_of_the_three_line_ends(monkeypatch,
                                                                      block_byte):
    """The end of a line may lie across two reads of the pipe."""
    monkeypatch.setattr(progress, "_PIPE_BLOCK_BYTE", block_byte)
    monkeypatch.setattr(progress, "_MAX_LINE_BYTE", 6)
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"ab\r\ncd\rxy\n\n123456\n1234567890\r\n\xff\nlast")
    os.close(write_fd)
    assert list(progress._pipe_line(read_fd)) == [
        "ab", "cd", "xy", "", "123456", "123456" + progress._LINE_CUT_MARK, "\ufffd",
        "last"]
    os.close(read_fd)


def test_a_process_still_holding_the_pipe_is_waited_for_no_longer_than_the_limit():
    hold = ("import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])")
    thread_set = set(threading.enumerate())
    start = time.monotonic()
    _run_child(hold, wait_sec=0.5)
    assert time.monotonic() - start < 4
    for thread in set(threading.enumerate()) - thread_set:
        thread.join(10)


@pytest.mark.parametrize("name", ["stdout", "stderr"])
def test_a_stream_that_stops_taking_lines_while_the_child_runs_does_not_stop_the_child(
        monkeypatch, tmp_path, name):
    """The reader of this process's output went away after the first line: the child
    still writes every line of both streams and ends with status 0."""
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(write_fd, "w", encoding="utf-8")
    os.close(read_fd)
    done = tmp_path / "done.txt"
    with monkeypatch.context() as patch:
        patch.setattr(progress.sys, name, stream)
        with progress.relay(10) as (out_fd, err_fd):
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import sys, time\nfor n in range(4):\n"
                 "    print('out', n, flush=True)\n"
                 "    print('err', n, file=sys.stderr, flush=True)\n    time.sleep(0.1)\n"
                 f"open({str(done)!r}, 'w').write('all written')"],
                check=False, stdout=out_fd, stderr=err_fd, timeout=30)
    assert proc.returncode == 0 and done.read_text() == "all written"
    with pytest.raises(OSError):
        stream.close()


def test_a_child_stdout_is_given_to_stdout_until_it_is_kept(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "_CHILD_STDOUT_TO_FILE", False)
    out_path = tmp_path / "stdout-r1.txt"
    with progress.relay(10, str(out_path)) as (out_fd, err_fd):
        subprocess.run([sys.executable, "-c", "print('to out')"], check=True,
                       stdout=out_fd, stderr=err_fd, timeout=30)
    assert capsys.readouterr().out == "to out\n" and not out_path.exists()


def test_a_kept_child_stdout_goes_to_the_file(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "_CHILD_STDOUT_TO_FILE", False)
    progress.keep_child_stdout(True)
    out_path = tmp_path / "sandbox" / "stdout-r1.txt"
    with progress.label("[task 1/2] "):
        with progress.relay(10, str(out_path)) as (out_fd, err_fd):
            subprocess.run([sys.executable, "-c",
                            "import sys; print('to out'); print('to err', file=sys.stderr)"],
                           check=True, stdout=out_fd, stderr=err_fd, timeout=30)
    out, err = capsys.readouterr()
    assert out == "" and err == "[task 1/2] to err\n"
    assert out_path.read_text(encoding="utf-8") == "[task 1/2] to out\n"


@pytest.mark.parametrize("name", ["stdout", "stderr"])
def test_a_stream_this_process_was_started_without_takes_the_lines_nowhere(
        capsys, monkeypatch, name):
    """The child writes every line and ends with status 0, whichever stream is closed."""
    with monkeypatch.context() as patch:
        patch.setattr(progress.sys, name, None)
        _run_child("import sys\nfor n in range(3):\n    print('out', n, flush=True)\n"
                   "    print('err', n, file=sys.stderr, flush=True)")
    out, err = capsys.readouterr()
    assert (out, err) == (("", "err 0\nerr 1\nerr 2\n") if name == "stdout"
                          else ("out 0\nout 1\nout 2\n", ""))


# == Whether a question can be asked =========================
class _Stream:
    """A stand-in for sys.stdin or sys.stdout that is, or is not, a terminal."""

    def __init__(self, is_terminal: bool) -> None:
        self.is_terminal = is_terminal

    def isatty(self) -> bool:
        return self.is_terminal


@pytest.mark.parametrize("stdin, stdout, expected", [
    (_Stream(True), _Stream(True), True), (_Stream(False), _Stream(True), False),
    (_Stream(True), _Stream(False), False), (None, _Stream(True), False),
    (_Stream(True), None, False)])
def test_a_question_is_asked_only_with_the_input_and_the_output_at_a_terminal(
        monkeypatch, stdin, stdout, expected):
    monkeypatch.setattr(progress.sys, "stdin", stdin)
    monkeypatch.setattr(progress.sys, "stdout", stdout)
    assert progress.at_terminal() is expected
    monkeypatch.setattr(progress.sys, "stderr", None)
    assert progress.at_terminal() is False


# == The time since the run started ==========================
@pytest.mark.parametrize("second, text", [
    (0, "0:00:00"), (42.9, "0:00:42"), (372, "0:06:12"), (3725, "1:02:05"),
    (-3, "0:00:00")])
def test_clock_text(second, text):
    assert progress.clock_text(second) == text


def test_the_run_start_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(progress.RUN_START_ENV, "1000.5")
    assert progress.run_start() == 1000.5


def test_a_run_start_that_is_not_a_number_is_the_process_start(monkeypatch):
    monkeypatch.setenv(progress.RUN_START_ENV, "soon")
    assert progress.run_start() == progress._PROCESS_START


def test_a_run_start_set_before_is_kept(monkeypatch):
    monkeypatch.setenv(progress.RUN_START_ENV, "1000")
    progress.mark_run_start()
    assert progress.run_start() == 1000


def test_a_line_starts_with_the_time_since_the_run_started(monkeypatch):
    monkeypatch.setenv(progress.RUN_START_ENV, "1000")
    record = _record("hi")
    record.created = 1000 + 3725
    with progress.label("[task 2/5] "):
        assert progress.LabelFormatter(progress.TERM_LOG_FORMAT).format(record) \
            == "[task 2/5] 1:02:05 INFO name: hi"


# == What the terminal shows ==================================
def test_the_terminal_shows_progress_and_warnings_alone():
    term_filter = progress.TermFilter()
    info = logging.LogRecord("codetwine_loom.sandbox", logging.INFO, __file__, 1, "x",
                             None, None)
    warn = logging.LogRecord("codetwine_loom.sandbox", logging.WARNING, __file__, 1, "x",
                             None, None)
    note = logging.LogRecord(progress.logger.name, logging.INFO, __file__, 1, "x",
                             None, None)
    assert [term_filter.filter(r) for r in (info, warn, note)] == [False, True, True]


def test_a_note_is_logged_by_the_progress_logger(caplog):
    with caplog.at_level("INFO", logger="codetwine_loom.progress"):
        progress.note("task %d of %d finished", 1, 3)
    assert [(r.name, r.getMessage()) for r in caplog.records] \
        == [("codetwine_loom.progress", "task 1 of 3 finished")]


# == What the two streams write =============================
class _StopStream:
    """A stream that takes writes until it is told to stop, then raises on each."""

    def __init__(self, error: Exception) -> None:
        self.text_list, self.error, self.is_stop = [], error, False

    def write(self, text: str) -> int:
        if self.is_stop:
            raise self.error
        self.text_list.append(text)
        return len(text)

    def flush(self) -> None:
        if self.is_stop:
            raise self.error


@pytest.mark.parametrize("is_secret", [False, True])
def test_a_question_is_written_where_the_lines_are_printed_and_the_answer_read(
        monkeypatch, is_secret):
    """The question goes through standard output, with what a terminal acts on
    written out, and whichever reader is used is handed no prompt of its own."""
    raw = io.StringIO()
    monkeypatch.setattr(sys, "stdout", progress.TermStream(raw))
    prompt_list = []
    reader_dict = {False: "builtins.input", True: "getpass.getpass"}
    monkeypatch.setattr(reader_dict[is_secret],
                        lambda prompt="": prompt_list.append(prompt) or " typed ")
    monkeypatch.setattr(reader_dict[not is_secret],
                        lambda prompt="": pytest.fail("the other reader was used"))
    assert progress.ask("NAME [a\x1b[2Jb]: ", is_secret) == " typed "
    assert raw.getvalue() == "NAME [a\\x1b[2Jb]: " and prompt_list == [""]


@pytest.mark.parametrize("is_secret", [False, True])
def test_an_answer_with_bytes_that_are_not_utf_8_is_asked_for_again(monkeypatch,
                                                                     is_secret):
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    answer_iter = iter(["/tmp/caf\udce9", "/tmp/cafe"])
    monkeypatch.setattr("getpass.getpass" if is_secret else "builtins.input",
                        lambda prompt="": next(answer_iter))
    assert progress.ask("PROJECT_DIR []: ", is_secret) == "/tmp/cafe"
    assert out.getvalue() == ("PROJECT_DIR []: The answer holds bytes that are not "
                              "UTF-8. Type it again\nPROJECT_DIR []: ")


@pytest.mark.parametrize("is_secret", [False, True])
def test_an_answer_the_reader_fails_to_decode_is_asked_for_again(monkeypatch, is_secret):
    """A reader that decodes strictly raises on the bytes rather than keeping them."""
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    answer_list = [UnicodeDecodeError("utf-8", b"caf\xe9", 3, 4, "invalid"), "/tmp/cafe"]

    def read(prompt=""):
        answer = answer_list.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("getpass.getpass" if is_secret else "builtins.input", read)
    assert progress.ask("PROJECT_DIR []: ", is_secret) == "/tmp/cafe"
    assert out.getvalue() == ("PROJECT_DIR []: The answer holds bytes that are not "
                              "UTF-8. Type it again\nPROJECT_DIR []: ")


def test_a_question_whose_input_ends_raises_as_the_reader_does(monkeypatch):
    def end(prompt=""):
        raise EOFError

    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr("builtins.input", end)
    with pytest.raises(EOFError):
        progress.ask("Carry on? [y/N] ")


_ASK_SCRIPT = """
import sys
from codetwine_loom import progress
progress.set_up_output()
print("asked:", progress.at_terminal())
if progress.at_terminal():
    print("answer:", repr(progress.ask("Carry on from there? [y/N] ")))
"""


@pytest.mark.parametrize("stderr_to, stdout_to, want", [
    ("terminal", "terminal", ["asked: True", "Carry on from there? [y/N] y",
                              "answer: 'y'"]),
    ("file", "terminal", ["asked: True", "Carry on from there? [y/N] y", "answer: 'y'"]),
    ("terminal", "file", []),
    ("closed", "terminal", ["asked: False"])])
def test_a_question_is_shown_on_the_terminal_or_not_asked(tmp_path, stderr_to, stdout_to,
                                                          want):
    """With standard error sent elsewhere the question still reaches the terminal;
    with standard output sent elsewhere, or standard error closed, nothing is asked."""
    import pty

    main, child = pty.openpty()
    side_path = tmp_path / "side.txt"
    with open(side_path, "w", encoding="utf-8") as side:
        proc = subprocess.Popen(
            [sys.executable, "-c", _ASK_SCRIPT], stdin=child,
            stdout=child if stdout_to == "terminal" else side,
            stderr=child if stderr_to == "terminal" else side, close_fds=True,
            preexec_fn=(lambda: os.close(2)) if stderr_to == "closed" else None,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(progress.__file__))))
    os.close(child)
    shown = b""
    limit = time.monotonic() + 30
    is_answer_sent = False
    while time.monotonic() < limit:
        try:
            piece = os.read(main, 65536)
        except OSError:
            break
        if not piece:
            break
        shown += piece
        if b"[y/N] " in shown and not is_answer_sent:
            os.write(main, b"y\n")
            is_answer_sent = True
    assert proc.wait(timeout=30) == 0
    os.close(main)
    assert [line for line in shown.decode().replace("\r\n", "\n").splitlines()
            if line] == want
    side_text = side_path.read_text(encoding="utf-8")
    assert "[y/N]" not in side_text
    assert ("asked: False" in side_text) is (stdout_to == "file")


@pytest.mark.parametrize("io_encoding", ["utf-8:strict", "utf-8:surrogateescape"])
def test_a_line_typed_on_the_terminal_that_is_not_utf_8_is_asked_for_again(io_encoding):
    """Standard input decodes strictly in most locales, with surrogates in a few."""
    import pty

    main, child = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, "-c", _ASK_SCRIPT], stdin=child, stdout=child, stderr=child,
        close_fds=True, env={**os.environ, "PYTHONIOENCODING": io_encoding},
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(progress.__file__))))
    os.close(child)
    shown = b""
    answer_list = [b"caf\xe9\n", b"y\n"]
    limit = time.monotonic() + 30
    while time.monotonic() < limit:
        try:
            piece = os.read(main, 65536)
        except OSError:
            break
        if not piece:
            break
        shown += piece
        if answer_list and shown.count(b"[y/N] ") == 3 - len(answer_list):
            os.write(main, answer_list.pop(0))
    assert proc.wait(timeout=30) == 0
    os.close(main)
    text = shown.decode("utf-8", "replace")
    assert "The answer holds bytes that are not UTF-8. Type it again" in text
    assert "answer: 'y'" in text and "Traceback" not in text


@pytest.mark.parametrize("word", [
    "plain", "", "two  blanks", "a\ttab", "it's", 'say "so"', "x$(touch MARK)",
    "back`touch MARK`tick", "a\nline break", "a;touch MARK", "a && touch MARK", "*", "~",
    "-flag", "日本語 名前", "$HOME", "a\\b", "#not a comment", "> MARK", "! history"])
def test_a_command_is_written_as_the_words_a_shell_reads_back(tmp_path, word):
    """Each word comes back as it was given, and nothing a word holds is run."""
    line = textcut.command_line("printf", "%s\\0", "first", word, "last")
    proc = subprocess.run(["sh", "-c", line], cwd=tmp_path, capture_output=True,
                          text=True, check=True)
    assert proc.stdout.split("\0") == ["first", word, "last", ""]
    assert os.listdir(tmp_path) == []
    assert shlex.split(line) == ["printf", "%s\\0", "first", word, "last"]


def test_an_error_that_ends_with_a_command_holds_the_two_apart():
    command = textcut.command_line("git", "-C", "/a  b", "apply", "x y")
    error = textcut.CommandError("it does not apply.\nApply it by hand with:", command)
    assert isinstance(error, RuntimeError)
    assert str(error) == f"it does not apply.\nApply it by hand with: {command}"
    assert textcut.error_part(error) == ("it does not apply.\nApply it by hand with:",
                                         command)
    assert textcut.error_part(RuntimeError("no engine")) == ("no engine", "")
    assert textcut.with_command(textcut.one_line(error.why), error.command) == (
        f"it does not apply. Apply it by hand with: {command}")
    assert textcut.with_command("why", "") == "why"
    assert textcut.with_command("", "") == ""


@pytest.mark.parametrize("text, want", [
    ("plain text\twith a tab\n", "plain text\twith a tab\n"),
    ("a\x1b[2J\x1b[1;1Hb\x07\n", "a\\x1b[2J\\x1b[1;1Hb\\x07\n"),
    ("zz\r\x1b[2Ksrc\x1b[8m.py", "zz\\x0d\\x1b[2Ksrc\\x1b[8m.py"),
    ("\x00\x08\x0b\x0c\x1f\x7f", "\\x00\\x08\\x0b\\x0c\\x1f\\x7f"),
    ("csi \x9b2J osc \x9d0;t\x9c nel \x85", "csi \\x9b2J osc \\x9d0;t\\x9c nel \\x85"),
    ("日本語 é \u00a0", "日本語 é \u00a0"),
])
def test_what_a_terminal_acts_on_is_written_out(text, want):
    """Tab and newline are kept; every other control character is written as \\xNN."""
    assert textcut.term_text(text) == want
    raw = io.StringIO()
    stream = progress.TermStream(raw)
    assert stream.write(text) == len(text)
    stream.writelines([text, text])
    assert raw.getvalue() == want * 3


def test_the_two_streams_are_put_behind_a_stream_that_writes_controls_out(monkeypatch):
    """print(), a log record and a child's relayed line all go through it, once."""
    raw_dict = {name: io.BytesIO() for name in ("stdout", "stderr")}
    for name, raw in raw_dict.items():
        monkeypatch.setattr(sys, name, io.TextIOWrapper(raw, encoding="utf-8",
                                                        errors="strict"))
    progress.set_up_output()
    stream_dict = {name: getattr(sys, name) for name in raw_dict}
    progress.set_up_output()
    assert all(getattr(sys, name) is stream for name, stream in stream_dict.items())
    assert all(isinstance(stream, progress.TermStream) for stream in stream_dict.values())
    print("said: \ud83d \x1b[2J", flush=True)
    handler = logging.StreamHandler()
    handler.emit(_record("warned: \x1b]0;title\x07"))
    with progress.relay(5) as (out_fd, err_fd):
        os.write(out_fd, b"child \x1b[31m\n")
    for stream in stream_dict.values():
        stream.flush()
    assert raw_dict["stdout"].getvalue() == b"said: \\ud83d \\x1b[2J\nchild \\x1b[31m\n"
    assert raw_dict["stderr"].getvalue() == b"warned: \\x1b]0;title\\x07\n"
    # What the stream itself answers is the answer of the stream under it
    assert sys.stdout.encoding == "utf-8" and sys.stdout.isatty() is False


def test_a_stream_over_a_descriptor_is_written_to_through_a_copy_of_it(tmp_path,
                                                                        monkeypatch):
    """Pointing the stream's own descriptor elsewhere afterwards moves no line."""
    out_path, other_path = tmp_path / "out.txt", tmp_path / "other.txt"
    with open(out_path, "w", encoding="utf-8", buffering=1) as first_stream:
        first_stream.write("written before\n")
        monkeypatch.setattr(sys, "stdout", first_stream)
        monkeypatch.setattr(sys, "stderr", None)
        assert progress.apart_fd_list() == []
        progress.set_up_output()
        stream = sys.stdout
        assert stream.fileno() != first_stream.fileno()
        assert (stream.name, stream.encoding, stream.line_buffering, stream.isatty()) \
            == (first_stream.name, "utf-8", True, False)
        assert stream.apart_fd == first_stream.fileno() > 2
        assert progress.apart_fd_list() == [first_stream.fileno()]
        other_fd = os.open(other_path, os.O_WRONLY | os.O_CREAT)
        os.dup2(other_fd, first_stream.fileno())
        os.close(other_fd)
        print("written after \x1b[2J")
    assert out_path.read_text(encoding="utf-8") == "written before\nwritten after \\x1b[2J\n"
    assert other_path.read_text(encoding="utf-8") == ""


def test_a_stream_whose_descriptor_cannot_be_copied_is_written_to_as_it_is(tmp_path,
                                                                           monkeypatch):
    def no_copy(fd):
        raise OSError(24, "Too many open files")

    with open(tmp_path / "out.txt", "w", encoding="utf-8") as first_stream:
        monkeypatch.setattr(sys, "stdout", first_stream)
        monkeypatch.setattr(sys, "stderr", None)
        monkeypatch.setattr(os, "dup", no_copy)
        progress.set_up_output()
        assert sys.stdout.fileno() == first_stream.fileno()
        assert sys.stdout.apart_fd is None and progress.apart_fd_list() == []
        print("written", flush=True)
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "written\n"


def test_a_stream_that_is_closed_is_left_as_it_is(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    progress.set_up_output()
    assert sys.stdout is None and isinstance(sys.stderr, progress.TermStream)
    assert progress.apart_fd_list() == []
    print("nothing to write this to")


_COPY_SCRIPT = """
import os, sys
from codetwine_loom import progress
progress.set_up_output()
stream_list = [stream for stream in (sys.stdout, sys.stderr) if stream is not None]
open_list = []
for fd in (1, 2):
    try:
        os.fstat(fd)
        open_list.append(fd)
    except OSError:
        pass
with open(sys.argv[1], "w") as result_file:
    print([stream.fileno() > 2 for stream in stream_list], progress.apart_fd_list(),
          open_list, file=result_file)
"""


@pytest.mark.parametrize("close_tuple, want", [
    ((), "[True, True] [1, 2] [1, 2]"), ((1,), "[True] [2] [2]"),
    ((2,), "[True] [1] [1]"), ((1, 2), "[] [] []")])
def test_the_copy_a_stream_is_written_through_takes_no_number_of_a_standard_stream(
        tmp_path, close_tuple, want):
    """A process started with standard output or standard error closed keeps that
    number free, and only the other one is a descriptor that can be pointed elsewhere."""
    def close() -> None:
        for fd in close_tuple:
            os.close(fd)

    result_path = tmp_path / "result.txt"
    subprocess.run(
        [sys.executable, "-c", _COPY_SCRIPT, str(result_path)], check=True, timeout=60,
        preexec_fn=close, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(progress.__file__))))
    assert result_path.read_text().strip() == want


@pytest.mark.parametrize("error, is_said", [
    (BrokenPipeError(32, "Broken pipe"), False),
    (OSError(28, "No space left on device"), True),
    (ValueError("I/O operation on closed file"), False)])
def test_what_is_written_once_a_stream_takes_no_more_is_dropped(monkeypatch, caplog,
                                                                error, is_said):
    """A reader that went away raises nothing in whoever prints after it, and a
    stream that is still there and takes no more is named once."""
    raw = _StopStream(error)
    monkeypatch.setattr(sys, "stdout", raw)
    progress.set_up_output()
    print("first")
    raw.is_stop = True
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.progress"):
        print("second", flush=True)
        raw.is_stop = False
        print("third", flush=True)
        sys.stdout.flush()
    assert raw.text_list == ["first", "\n"]
    assert [record.getMessage() for record in caplog.records] == (
        [f"a stream takes no more ({error}), so what is written to it from here on "
         f"is dropped"] if is_said else [])


# == What pipes took in =====================================
def _thread_and_fd_count() -> tuple[int, int]:
    """Return how many threads run and how many descriptors are open."""
    return threading.active_count(), len(os.listdir("/dev/fd"))


def _wait_for_count(want: tuple[int, int]) -> tuple[int, int]:
    """Return _thread_and_fd_count() once it is want, or as it is 3 seconds on."""
    limit = time.monotonic() + 3
    while _thread_and_fd_count() != want and time.monotonic() < limit:
        time.sleep(0.05)
    return _thread_and_fd_count()


def test_what_a_process_writes_to_the_pipes_is_kept_pipe_by_pipe():
    count = _thread_and_fd_count()
    with progress.PipeOutput(2, 100) as output:
        subprocess.run(["sh", "-c", "echo out; echo err >&2; echo more"], check=True,
                       stdout=output.write_fd_list[0], stderr=output.write_fd_list[1])
    assert [output.text(index, "utf-8", "replace") for index in (0, 1)] == [
        "out\nmore\n", "err\n"]
    assert _thread_and_fd_count() == count


@pytest.mark.parametrize("block_byte", [1, 7, 1 << 16])
@pytest.mark.parametrize("head_byte, tail_byte, want", [
    (100, 0, "0123456789" * 3 + "xyz"),
    (30, 10, "0123456789" * 3 + "xyz"),
    (4, 0, "0123" + textcut.CUT_NOTE.format(omitted=29)),
    (4, 3, "0123" + textcut.CUT_NOTE.format(omitted=26) + "\nxyz"),
    (0, 5, textcut.CUT_NOTE.format(omitted=28) + "\n89xyz"),
])
def test_the_first_and_the_last_bytes_of_a_pipe_are_kept(monkeypatch, block_byte,
                                                         head_byte, tail_byte, want):
    monkeypatch.setattr(progress, "_PIPE_BLOCK_BYTE", block_byte)
    with progress.PipeOutput(1, head_byte, tail_byte) as output:
        os.write(output.write_fd_list[0], b"0123456789" * 3 + b"xyz")
    assert output.text(0, "utf-8", "replace") == want


def test_a_byte_that_is_not_text_is_read_as_the_caller_says():
    with progress.PipeOutput(1, 100) as output:
        os.write(output.write_fd_list[0], b"a\xff\x00b\r\n")
    assert output.text(0, "utf-8", "replace") == "a�\x00b\r\n"


def _keep_word(line: bytes) -> int:
    """Keep a line that holds "keep", and three lines from one that starts with "KEEP3"."""
    if line.startswith(b"KEEP3"):
        return 3
    return int(b"keep" in line)


def _cut(count: int) -> str:
    """Return the line that stands where count bytes were dropped."""
    return textcut.CUT_NOTE.format(omitted=count) + "\n"


@pytest.mark.parametrize("block_byte", [1, 7, 1 << 16])
@pytest.mark.parametrize("data, head_byte, tail_byte, keep_byte, want", [
    # Nothing lies between the two ends
    (b"h1\nkeep\nt1\n", 6, 6, 100, "h1\nkeep\nt1\n"),
    # A line between them that is kept, and the ends cut to whole lines
    (b"h1\nh2\nxx\nkeep 1\nyy\nt1\nt2\n", 7, 7, 100,
     "h1\nh2\n" + _cut(6) + "keep 1\nt1\nt2\n"),
    # No line between them is kept
    (b"h1\nh2\nxx\nzz\nyy\nt1\nt2\n", 7, 7, 100, "h1\nh2\n" + _cut(9) + "t1\nt2\n"),
    # The lines after one that asks for them
    (b"h1\nxx\nKEEP3 a\nn1\nn2\nn3\nt1\n", 3, 3, 100,
     "h1\n" + _cut(6) + "KEEP3 a\nn1\nn2\nt1\n"),
    # The line the first bytes end in and the line the last bytes start in
    (b"h1\nkeep over the end\nxx\nkeep to the tail\nt1\n", 8, 8, 100,
     "h1\n" + _cut(3) + "keep over the end\nkeep to the tail\nt1\n"),
    # Every line end, and a byte that is not text
    (b"h1\r\nxx\rkeep \xff\r\nkeep 2\ryy\r\nt1\r\n", 4, 4, 100,
     "h1\r\n" + _cut(8) + "keep \ufffd\nkeep 2\nt1\r\n"),
    # No more than keep_byte, and none after the first that does not fit
    (b"h1\nkeep 1\nkeep 22\nkeep\nt1\n", 3, 3, 9, "h1\n" + _cut(13) + "keep 1\nt1\n"),
    # First bytes that hold no line end stay whole, and their line is not read
    (b"keep the head\nxx\nkeep 2\nt1\n", 4, 3, 100,
     "keep" + _cut(13) + "keep 2\nt1\n"),
    # Last bytes that hold no line end stay whole
    (b"h1\nkeep 1\nxx\nthe last line", 3, 4, 100, "h1\n" + _cut(12) + "keep 1\nline"),
    # All of what lies between is kept
    (b"h1\nkeep 1\nkeep 2\nt1\n", 3, 3, 100, "h1\nkeep 1\nkeep 2\nt1\n"),
    # No last bytes
    (b"h1\nxx\nkeep 1\nyy\n", 3, 0, 100, "h1\n" + _cut(6) + "keep 1\n"),
])
def test_the_lines_a_caller_keeps_stay_between_the_first_and_the_last_bytes(
        monkeypatch, block_byte, data, head_byte, tail_byte, keep_byte, want):
    monkeypatch.setattr(progress, "_PIPE_BLOCK_BYTE", block_byte)
    with progress.PipeOutput(1, head_byte, tail_byte, _keep_word, keep_byte) as output:
        os.write(output.write_fd_list[0], data)
    assert output.text(0, "utf-8", "replace") == want
    assert output.text(0, "utf-8", "replace") == want


@pytest.mark.parametrize("block_byte", [1, 7, 1 << 16])
def test_a_line_longer_than_the_limit_is_not_asked_about_and_not_kept(monkeypatch,
                                                                      block_byte):
    monkeypatch.setattr(progress, "_PIPE_BLOCK_BYTE", block_byte)
    monkeypatch.setattr(progress, "_KEEP_LINE_MAX_BYTE", 10)
    ask_list = []

    def keep(line):
        ask_list.append(line)
        return _keep_word(line)

    data = (b"h1\n" + b"keep" * 3 + b"\nkeep 1\nKEEP3\n" + b"n" * 11 + b"\nn2\nn3\nt1\n")
    with progress.PipeOutput(1, 3, 3, keep, 100) as output:
        os.write(output.write_fd_list[0], data)
    assert output.text(0, "utf-8", "replace") == "h1\n" + _cut(28) + "keep 1\nKEEP3\nn2\nt1\n"
    assert ask_list == [b"keep 1", b"KEEP3", b"n2", b"n3"]


def test_each_pipe_keeps_its_own_lines():
    with progress.PipeOutput(2, 3, 3, _keep_word, 100) as output:
        os.write(output.write_fd_list[0], b"h1\nKEEP3\nn1\n")
        os.write(output.write_fd_list[1], b"e1\nxx\ne3\ne4\nt2\n")
        os.write(output.write_fd_list[0], b"n2\nn3\nt1\n")
    assert output.text(0, "utf-8", "replace") == "h1\n" + _cut(3) + "KEEP3\nn1\nn2\nt1\n"
    assert output.text(1, "utf-8", "replace") == "e1\n" + _cut(9) + "t2\n"


def test_a_great_many_bytes_with_no_line_end_are_not_held(monkeypatch):
    monkeypatch.setattr(progress, "_KEEP_LINE_MAX_BYTE", 1000)
    keep = progress._PipeKeep(10, 10, _keep_word, 100)
    for _ in range(200):
        keep.add(b"keep" * 1000)
        assert len(keep.line_start) <= 1000 and keep.tail_size <= 4010
    keep.add(b"\nkeep 1\nxx\nlast line\n")
    assert keep.text("utf-8", "replace") == (
        "keepkeepke" + _cut(799994) + "keep 1\nlast line\n")


# A process that writes a line, waits, and writes one more
_WRITE_LATE = ("import sys, time; print('first', flush=True); time.sleep(1); "
               "print('late', flush=True)")


def test_a_pipe_another_process_still_holds_is_closed_once_the_wait_is_over():
    count = _thread_and_fd_count()
    output = progress.PipeOutput(1, 100)
    proc = subprocess.Popen([sys.executable, "-c", _WRITE_LATE],
                            stdout=output.write_fd_list[0], stderr=subprocess.DEVNULL)
    try:
        start = time.monotonic()
        output.end(0.3)
        assert time.monotonic() - start < 0.9
        assert output.text(0, "utf-8", "replace") == "first\n"
        assert _thread_and_fd_count() == count
    finally:
        assert proc.wait(10) != 0


def test_a_pipe_read_on_after_its_end_takes_what_is_written_and_keeps_none_of_it():
    count = _thread_and_fd_count()
    output = progress.PipeOutput(1, 100)
    proc = subprocess.Popen([sys.executable, "-c", _WRITE_LATE],
                            stdout=output.write_fd_list[0], stderr=subprocess.DEVNULL)
    try:
        output.end(0.3, is_drop_after=True)
        assert output.text(0, "utf-8", "replace") == "first\n"
    finally:
        assert proc.wait(10) == 0
    assert output.text(0, "utf-8", "replace") == "first\n"
    assert _wait_for_count(count) == count


def test_a_pipe_no_process_holds_ends_at_once_whichever_way_it_is_ended():
    count = _thread_and_fd_count()
    for is_drop_after in (False, True):
        output = progress.PipeOutput(2, 100)
        os.write(output.write_fd_list[1], b"x")
        start = time.monotonic()
        output.end(5, is_drop_after)
        output.end(5, is_drop_after)
        assert time.monotonic() - start < 2
        assert output.text(1, "utf-8", "replace") == "x"
    assert _wait_for_count(count) == count


def test_a_writer_that_goes_on_after_the_end_is_read_no_further_than_a_limit(monkeypatch):
    monkeypatch.setattr(progress, "_PIPE_STOP_BYTE", 1 << 16)
    count = _thread_and_fd_count()
    output = progress.PipeOutput(1, 1 << 30)
    proc = subprocess.Popen(["yes"], stdout=output.write_fd_list[0])
    try:
        output.end(0.2)
        assert _thread_and_fd_count() == count
    finally:
        proc.kill()
        proc.wait(10)


def test_pipes_that_cannot_all_be_opened_leave_none_open(monkeypatch):
    count = _thread_and_fd_count()
    open_pipe = os.pipe
    call_list = []

    def one_pipe_only() -> tuple[int, int]:
        call_list.append(1)
        if len(call_list) > 1:
            raise OSError(24, "Too many open files")
        return open_pipe()

    with monkeypatch.context() as patch:
        patch.setattr(progress.os, "pipe", one_pipe_only)
        with pytest.raises(OSError, match="Too many open files"):
            progress.PipeOutput(2, 100)
    assert _thread_and_fd_count() == count

    def no_start(self) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(progress.threading.Thread, "start", no_start)
    with pytest.raises(OSError, match="can't start new thread"):
        progress.PipeOutput(2, 100)
    assert _thread_and_fd_count() == count
