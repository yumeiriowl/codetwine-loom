"""What each turn did, kept beside the index.

The loop folds old turns out of the prompt and returns only the last state, so this is
the only place a finished run's turn-by-turn record survives. Nothing here calls an LLM:
the loop is driven by a stand-in that returns fixed blocks.
"""

import json
import logging
import os
from datetime import datetime

import pytest

from codetwine_loom import config, rlm, turnlog
from codetwine_loom.tools import source


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    """Point the turn log at a fresh output directory of its own."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(source, "project_name", "codetwine")
    monkeypatch.setattr(config, "TURN_LOG_KEEP", 20)
    yield tmp_path / "out"
    turnlog.close()


def _line_list(path):
    """Return the log's lines, parsed."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _stage(replies, stage="edit", max_turn=None):
    """Build a loop whose model answers with the given blocks, one per turn."""
    answers = list(replies)

    def llm(_prompt):
        return answers.pop(0)

    return rlm.RLM(llm=llm, tool_list=[], instruction="INSTR", stage=stage,
                   max_turn=max_turn if max_turn is not None else len(replies))


def _run(stage):
    """Run one loop with nothing in the REPL but the terminals."""
    return stage.run(request="do it", variable_dict={}, var_doc="")


def test_every_turn_of_a_run_is_written_as_it_goes(out_dir):
    path = turnlog.open_log("20260829-120000")
    _run(_stage(["```python\nprint('one')\n```",
                 "```python\nCOMPLETE('done')\n```"]))
    lines = _line_list(path)
    assert [(x["stage"], x["turn"]) for x in lines] == [("edit", 1), ("edit", 2)]
    assert lines[0]["code"].strip() == "print('one')"
    assert lines[0]["output"] == "one\n"
    assert "COMPLETE() was called: done" in lines[1]["output"]


def test_the_token_counts_of_a_turn_are_written_with_it(out_dir):
    class _CountingLLM:
        """A stand-in LLM that reports token counts the way make_llm() does."""

        def __init__(self):
            self.last_usage_dict = {}
            self.reply_list = ["```python\nprint('one')\n```",
                               "```python\nCOMPLETE('done')\n```"]

        def __call__(self, prompt):
            self.last_usage_dict = {"input_token": 10, "output_token": 2,
                                    "cache_read_token": 8, "cache_write_token": 1}
            return self.reply_list.pop(0)

    path = turnlog.open_log("20260915-120000")
    _run(rlm.RLM(llm=_CountingLLM(), tool_list=[], instruction="INSTR", stage="edit",
                 max_turn=2))
    lines = _line_list(path)
    assert [x["usage"]["cache_read_token"] for x in lines] == [8, 8]


def test_a_turn_from_an_llm_that_reports_no_counts_is_written_without_usage(out_dir):
    path = turnlog.open_log("20260915-120001")
    _run(_stage(["```python\nCOMPLETE('done')\n```"]))
    assert "usage" not in _line_list(path)[0]


def test_the_log_is_readable_before_the_run_ends(out_dir):
    """A run that is killed still leaves what it did, so each turn is flushed."""
    path = turnlog.open_log("20260829-120000")
    stage = _stage(["```python\nprint('one')\n```"] * 2, max_turn=2)

    def llm(_prompt):
        # The file is read from inside the run, before it has returned
        assert len(_line_list(path)) == 0 or _line_list(path)[-1]["turn"] >= 1
        return "```python\nprint('one')\n```"

    stage.llm = llm
    _run(stage)
    assert len(_line_list(path)) == 2


def test_every_stage_of_one_run_lands_in_the_same_file(out_dir):
    path = turnlog.open_log("20260829-120000")
    _run(_stage(["```python\nCOMPLETE('a')\n```"], stage="plan"))
    _run(_stage(["```python\nCOMPLETE('b')\n```"], stage="edit"))
    assert [x["stage"] for x in _line_list(path)] == ["plan", "edit"]


def test_a_turn_that_yielded_no_code_is_written_too(out_dir):
    path = turnlog.open_log("20260829-120000")
    _run(_stage(["nothing but prose"]))
    assert len(_line_list(path)) == 1


def test_the_one_call_after_the_turns_ran_out_is_written(out_dir):
    path = turnlog.open_log("20260829-120000")
    stage = _stage(["```python\nprint('one')\n```"], max_turn=1)
    _run(stage)
    stage.llm = lambda _prompt: "I changed one file"
    rlm.late_answer(stage, "do it", "", [("code", "out")], "note", "closing")
    lines = _line_list(path)
    assert lines[-1]["turn"] == stage.max_turn + 1
    assert lines[-1]["output"] == "harvested: I changed one file"


def test_the_one_call_after_an_added_last_turn_comes_after_that_turn(out_dir):
    """A COMPLETE() sent back on the last turn adds one, and the call is not given its
    number."""
    path = turnlog.open_log("20260829-120000")
    stage = _stage(["```python\nCOMPLETE('a')\n```", "```python\nprint('one')\n```"],
                   max_turn=1)
    out_dict = stage.run(request="do it", variable_dict={}, var_doc="",
                         on_complete=lambda _answer: "not yet")
    stage.llm = lambda _prompt: "I changed one file"
    rlm.late_answer(stage, "do it", "", out_dict["history"], "note", "closing")
    assert [x["turn"] for x in _line_list(path)] == [1, 2, 3]


def test_each_start_of_a_stage_is_told_apart_in_the_log(out_dir):
    """A stage built again in one run, as each explore() call does, counts its turns
    from 1 again."""
    path = turnlog.open_log("20260829-120000")
    _run(_stage(["```python\nCOMPLETE('a')\n```"], stage="edit"))
    _run(_stage(["```python\nprint(1)\n```", "```python\nCOMPLETE('b')\n```"],
                stage="explore"))
    stage = _stage(["```python\nprint(2)\n```"], stage="explore", max_turn=1)
    _run(stage)
    stage.llm = lambda _prompt: "found nothing"
    rlm.late_answer(stage, "do it", "", [("code", "out")], "note", "closing")
    assert [(x["stage"], x["call"], x["turn"]) for x in _line_list(path)] == [
        ("edit", 1, 1), ("explore", 1, 1), ("explore", 1, 2), ("explore", 2, 1),
        ("explore", 2, 2)]
    turnlog.open_log("20260829-120001")
    assert turnlog.start_stage("explore") == 1


def test_the_oldest_runs_are_taken_away_when_a_new_one_starts(out_dir, monkeypatch):
    monkeypatch.setattr(config, "TURN_LOG_KEEP", 3)
    for name in ("20260829-120001", "20260829-120002", "20260829-120003",
                 "20260829-120004"):
        turnlog.open_log(name)
        turnlog.record("edit", 1, "code", "output")
        turnlog.close()
    left = sorted(p.name for p in (out_dir / "codetwine" / "turns").iterdir())
    assert left == ["20260829-120002.jsonl", "20260829-120002.log",
                    "20260829-120003.jsonl", "20260829-120003.log",
                    "20260829-120004.jsonl", "20260829-120004.log"]


def test_keeping_none_writes_nothing_at_all(out_dir, monkeypatch):
    monkeypatch.setattr(config, "TURN_LOG_KEEP", 0)
    assert turnlog.open_log("20260829-120000") == ""
    _run(_stage(["```python\nCOMPLETE('done')\n```"]))
    assert not (out_dir / "codetwine" / "turns").exists()


def test_a_run_with_no_log_open_is_not_an_error(out_dir):
    """Nothing opens one for a library caller, and the loop must not care."""
    result = _run(_stage(["```python\nCOMPLETE('done')\n```"]))
    assert result["answer"] == "done"


def test_a_log_that_cannot_be_written_does_not_stop_the_run(out_dir, caplog,
                                                            monkeypatch):
    turnlog.open_log("20260829-120000")

    class _Refuses:
        def write(self, _line):
            raise OSError("no space left on device")

        def flush(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(turnlog, "_turn_file", _Refuses())
    result = _run(_stage(["```python\nprint('one')\n```",
                          "```python\nCOMPLETE('done')\n```"]))
    assert result["answer"] == "done"
    assert "the rest of this run is not logged" in caplog.text.lower()
    # The second turn is not tried again once it has given up
    assert caplog.text.lower().count("the rest of this run is not logged") == 1


def test_a_character_utf8_cannot_encode_is_written_as_its_escape(out_dir, caplog):
    path = turnlog.open_log("20260829-120000")
    turnlog.record("edit", 1, "print('\ud83d')", "name: \udce3\udc81")
    turnlog.record("edit", 2, "print(2)", "2")
    with caplog.at_level(logging.INFO):
        logging.getLogger("codetwine_loom.rlm").info("it ends with: \udce3")
    turnlog.close()
    lines = _line_list(path)
    assert [x["turn"] for x in lines] == [1, 2]
    assert lines[0]["code"] == "print('\ud83d')"
    assert lines[0]["output"] == "name: \udce3\udc81"
    with open(path, encoding="utf-8") as fh:
        assert "name: \\udce3\\udc81" in fh.read()
    with open(path.replace(".jsonl", ".log"), encoding="utf-8") as fh:
        assert "it ends with: \\udce3" in fh.read()


def test_the_log_goes_where_the_editing_tools_refuse_to_write(out_dir):
    """It sits under OUTPUT_DIR, so nothing of it reaches the target repository."""
    assert turnlog.directory().startswith(config.OUTPUT_DIR)


def test_two_runs_starting_in_one_second_write_two_files(out_dir):
    """The name is not the time alone, so runs sharing an output directory - agents on
    one machine, each in a container with a process numbering of its own - write their
    turns into files of their own."""
    first = turnlog.open_log()
    turnlog.record("edit", 1, "code", "the first run")
    second = turnlog.open_log()
    turnlog.record("edit", 1, "code", "the second run")
    turnlog.close()
    assert first != second
    assert len(list((out_dir / "codetwine" / "turns").glob("*.jsonl"))) == 2


def _log_text(path):
    """Return the .log file beside the turn file."""
    with open(path[:-len(".jsonl")] + ".log", encoding="utf-8") as fh:
        return fh.read()


def test_a_filter_given_with_the_log_is_passed_by_every_line_written_to_it(out_dir,
                                                                          caplog):
    class _Mark(logging.Filter):
        def filter(self, record):
            record.msg, record.args = record.getMessage().replace("secret", "<x>"), None
            return True

    caplog.set_level(logging.INFO)
    path = turnlog.open_log("20260829-120000", log_filter=_Mark())
    logging.getLogger("some.library").warning("sent to %s", "the secret place")
    turnlog.close()
    assert _log_text(path).splitlines()[1].endswith(
        "WARNING some.library: sent to the <x> place")


def test_what_the_tool_logged_during_the_run_is_kept_beside_the_turns(out_dir, caplog):
    caplog.set_level(logging.INFO)
    path = turnlog.open_log("20260829-120000")
    logging.getLogger("codetwine_loom.something").warning("the tests did not run")
    turnlog.close()
    line_list = _log_text(path).splitlines()
    assert line_list[0].endswith(
        f"INFO codetwine_loom.turnlog: writing what each turn did to {path}")
    assert line_list[1].endswith(
        "WARNING codetwine_loom.something: the tests did not run")
    # Each line starts with the time, so the lines can be set against the turns' "at"
    assert line_list[0].split(" ")[0] == datetime.now().strftime("%Y-%m-%d")


def test_how_a_run_ended_with_an_error_is_in_the_log(out_dir):
    with pytest.raises(SystemExit) as caught:
        with turnlog.open_run():
            raise ValueError("boom")
    assert caught.value.code == 1
    path_list = list((out_dir / "codetwine" / "turns").glob("*.log"))
    assert len(path_list) == 1
    text = path_list[0].read_text(encoding="utf-8")
    assert "ERROR codetwine_loom.turnlog: the run ended with ValueError: boom" in text
    assert "Traceback (most recent call last):" in text
    assert 'raise ValueError("boom")' in text


def test_a_run_that_ended_well_leaves_no_error_in_the_log(out_dir):
    with turnlog.open_run():
        pass
    path_list = list((out_dir / "codetwine" / "turns").glob("*.log"))
    assert "ERROR" not in path_list[0].read_text(encoding="utf-8")


def test_a_run_works_from_a_directory_of_its_own_and_comes_back(out_dir, tmp_path,
                                                               monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen_list = []
    with turnlog.open_run():
        seen_list.append(os.getcwd())
        with open("left-behind.txt", "w", encoding="utf-8") as handle:
            handle.write("x")
    assert os.path.basename(seen_list[0]).startswith("ctloom-run-")
    assert os.getcwd() == str(tmp_path)
    assert not os.path.exists(seen_list[0]) and not (tmp_path / "left-behind.txt").exists()
    # A run that ended with an error comes back as well
    with pytest.raises(SystemExit):
        with turnlog.open_run():
            raise ValueError("boom")
    assert os.getcwd() == str(tmp_path)


def test_closing_takes_the_handler_off_the_root_logger(out_dir):
    before = list(logging.getLogger().handlers)
    turnlog.open_log("20260829-120000")
    assert len(logging.getLogger().handlers) == len(before) + 1
    turnlog.close()
    assert logging.getLogger().handlers == before


# == Taking the oldest runs away ================================
def _run_file_list(directory, *stem_list):
    """Write the .jsonl and .log of each stem into directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for stem in stem_list:
        for suffix in (".jsonl", ".log"):
            (directory / (stem + suffix)).write_text("", encoding="utf-8")


def test_the_trim_leaves_room_for_the_run_about_to_start(out_dir):
    turns = out_dir / "codetwine" / "turns"
    _run_file_list(turns, "a", "b", "c")
    turnlog._trim(2)
    assert sorted(p.name for p in turns.iterdir()) == ["c.jsonl", "c.log"]


def test_the_oldest_run_is_the_one_whose_files_were_written_longest_ago(out_dir):
    """A run named after another clock's time is as old as its files are."""
    import os
    turns = out_dir / "codetwine" / "turns"
    _run_file_list(turns, "20260923-171043-aaaaaa", "20260924-021040-bbbbbb",
                   "20260923-171444-cccccc", "20260924-021431-dddddd")
    for second, stem in enumerate(("20260924-021040-bbbbbb", "20260923-171043-aaaaaa",
                                   "20260924-021431-dddddd", "20260923-171444-cccccc")):
        for suffix in (".jsonl", ".log"):
            os.utime(turns / (stem + suffix), (1_000_000 + second, 1_000_000 + second))
    # The later of a run's two files is how old the run is
    os.utime(turns / "20260924-021040-bbbbbb.log", (1_000_009, 1_000_009))
    turnlog._trim(3)
    assert sorted(p.name for p in turns.iterdir()) == [
        "20260923-171444-cccccc.jsonl", "20260923-171444-cccccc.log",
        "20260924-021040-bbbbbb.jsonl", "20260924-021040-bbbbbb.log"]


def test_runs_written_at_the_same_time_are_taken_away_in_the_order_of_their_names(
        out_dir):
    import os
    turns = out_dir / "codetwine" / "turns"
    _run_file_list(turns, "b", "c", "a")
    for path in turns.iterdir():
        os.utime(path, (1_000_000, 1_000_000))
    turnlog._trim(2)
    assert sorted(p.name for p in turns.iterdir()) == ["c.jsonl", "c.log"]


def test_a_run_missing_one_of_its_two_files_is_still_taken_away(out_dir):
    turns = out_dir / "codetwine" / "turns"
    _run_file_list(turns, "a", "b")
    (turns / "a.log").unlink()
    turnlog._trim(1)
    assert sorted(p.name for p in turns.iterdir()) == []


def test_a_directory_that_is_not_there_yet_is_nothing_to_trim(out_dir):
    turnlog._trim(1)
    assert not (out_dir / "codetwine" / "turns").exists()


def test_a_file_that_will_not_go_is_said_in_the_log(out_dir, monkeypatch, caplog):
    import os
    turns = out_dir / "codetwine" / "turns"
    _run_file_list(turns, "a", "b")

    def refuse(path):
        raise PermissionError("read-only")

    monkeypatch.setattr(os, "remove", refuse)
    with caplog.at_level(logging.INFO):
        turnlog._trim(1)
    assert "could not be taken away" in caplog.text
    assert sorted(p.name for p in turns.iterdir()) == ["a.jsonl", "a.log", "b.jsonl", "b.log"]


# == A directory that cannot be made ============================
def test_a_log_directory_that_cannot_be_made_leaves_the_run_without_one(out_dir, caplog):
    out_dir.mkdir()
    (out_dir / "codetwine").write_text("a file where the directory goes", encoding="utf-8")
    assert turnlog.open_log("20260829-120000") == ""
    assert "could not be opened" in caplog.text
    turnlog.record("edit", 1, "code", "output")


def test_closing_a_file_that_refuses_to_close_is_not_an_error(out_dir, monkeypatch):
    class _Refuse:
        def close(self):
            raise OSError("already gone")

    monkeypatch.setattr(turnlog, "_turn_file", _Refuse())
    turnlog.close()
    assert turnlog._turn_file is None
