"""The token counts one process holds, the file they cross a process boundary in, and
the limit held against them.
"""

import json
import os

import pytest

from codetwine_loom import config, tokencount

_ZERO = {"call": 0, "input_token": 0, "output_token": 0, "cache_read_token": 0,
         "cache_write_token": 0}


def _usage(**count_dict):
    """Return an object shaped like litellm's usage."""
    detail = type("Detail", (), {"cached_tokens": count_dict.pop("cached", None),
                                 "cache_creation_tokens": count_dict.pop("made", None)})()
    return type("Usage", (), {**count_dict, "prompt_tokens_details": detail})()


def test_the_counts_are_read_off_the_usage_object():
    usage = _usage(prompt_tokens=10, completion_tokens=3, cached=6, made=2)
    assert tokencount.count_dict_of(usage) == {
        "input_token": 10, "output_token": 3, "cache_read_token": 6,
        "cache_write_token": 2}


def test_a_reply_without_usage_counts_nothing():
    assert tokencount.count_dict_of(None) == {key: 0 for key in tokencount.KEY_TUPLE}


def test_every_call_is_added_to_the_total():
    tokencount.add({"input_token": 10, "output_token": 3, "cache_read_token": 6})
    tokencount.add({"input_token": 5, "output_token": 1})
    assert tokencount.total_dict() == {
        "call": 2, "input_token": 15, "output_token": 4, "cache_read_token": 6,
        "cache_write_token": 0, "total": 19, "parts": []}


@pytest.mark.parametrize("value", [-1, 1.5, "7", None, True])
def test_what_is_not_a_whole_number_adds_nothing(value):
    tokencount.add({"input_token": value, "output_token": 2})
    assert tokencount.total_dict()["input_token"] == 0
    assert tokencount.total_dict()["total"] == 2


def test_the_counts_added_under_a_part_are_listed_under_it():
    tokencount.add({"input_token": 4, "output_token": 1}, call_count=2, part="edit r1")
    assert tokencount.total_dict()["parts"] == [
        {"part": "edit r1", "call": 2, "input_token": 4, "output_token": 1,
         "cache_read_token": 0, "cache_write_token": 0, "total": 5}]


def test_the_total_another_process_wrote_is_added(tmp_path):
    tokencount.add({"input_token": 10, "output_token": 3})
    tokencount.write(str(tmp_path / "token.json"))
    assert json.loads((tmp_path / "token.json").read_text()) == {
        **_ZERO, "call": 1, "input_token": 10, "output_token": 3, "total": 13}
    tokencount.reset()
    tokencount.add({"input_token": 1})
    added = tokencount.add_file(str(tmp_path / "token.json"), part="edit r1")
    assert added == {**_ZERO, "call": 1, "input_token": 10, "output_token": 3}
    total = tokencount.total_dict()
    assert (total["call"], total["total"]) == (2, 14)
    assert [part["part"] for part in total["parts"]] == ["edit r1"]


@pytest.mark.parametrize("text", [None, "", "not json", "[1, 2]", '{"input_token": "x"}',
                                  "[" * 200000])
def test_a_file_that_holds_no_counts_adds_nothing(tmp_path, text):
    path = tmp_path / "token.json"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    assert tokencount.add_file(str(path), part="p") == _ZERO
    assert tokencount.total_dict() == {**_ZERO, "total": 0, "parts": []}


def test_a_file_that_cannot_be_written_is_logged_and_not_raised(tmp_path, caplog):
    tokencount.write(str(tmp_path / "missing" / "token.json"))
    assert "could not be written" in caplog.text


def test_no_limit_is_reached_while_the_setting_is_0(monkeypatch):
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 0)
    tokencount.add({"input_token": 10 ** 9})
    assert tokencount.limit_text() == ""


def test_the_limit_is_reached_at_the_setting_and_not_below_it(monkeypatch):
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 100)
    tokencount.add({"input_token": 90, "output_token": 9, "cache_read_token": 80})
    assert tokencount.limit_text() == ""
    tokencount.add({"output_token": 1})
    assert "100 token(s)" in tokencount.limit_text()
    assert "MAX_TOKEN_PER_RUN=100" in tokencount.limit_text()


def test_the_file_holds_every_call_when_several_threads_write_it(tmp_path):
    import threading
    path = str(tmp_path / "token.json")

    def call():
        for _ in range(40):
            tokencount.add({"input_token": 1, "output_token": 1})
            tokencount.write(path)

    thread_list = [threading.Thread(target=call) for _ in range(8)]
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
    assert json.loads((tmp_path / "token.json").read_text())["call"] == 320
    assert os.listdir(tmp_path) == ["token.json"]


def test_a_file_is_added_once(tmp_path):
    path = tmp_path / "token.json"
    path.write_text(json.dumps({"call": 1, "input_token": 40}))
    assert tokencount.add_file(str(path), part="edit r1")["input_token"] == 40
    assert tokencount.add_file(str(path), part="edit r1")["input_token"] == 0
    assert tokencount.total_dict()["total"] == 40
    assert len(tokencount.total_dict()["parts"]) == 1


def test_the_files_nothing_added_yet_are_added_for_a_command_that_was_stopped(
        tmp_path, monkeypatch):
    monkeypatch.setattr(tokencount.tempfile, "tempdir", str(tmp_path))
    tokencount.add_remain_file("not waited for")
    over, going, empty = (tokencount.child_file() for _ in range(3))
    for path, count in ((over, 40), (going, 25)):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"call": 1, "input_token": count}, handle)
    tokencount.add_file(over, part="edit r1")
    tokencount.add_remain_file("not waited for")
    total = tokencount.total_dict()
    assert total["total"] == 65
    assert [(part["part"], part["total"]) for part in total["parts"]] == [
        ("edit r1", 40), ("not waited for", 25)]
    # The process that wrote it adds nothing more when it ends
    tokencount.add_file(going, part="edit r2")
    assert tokencount.total_dict()["total"] == 65 and os.path.getsize(empty) == 0


def test_what_a_file_added_is_never_counted_twice_by_a_reader_beside_it(tmp_path,
                                                                        monkeypatch):
    """remain_token() and limit_text() read the total and what the files added as one."""
    import threading
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 1000)
    seen = []
    stop = threading.Event()

    def read():
        while not stop.is_set():
            seen.append(tokencount.remain_token())

    reader = threading.Thread(target=read)
    reader.start()
    for n in range(200):
        path = tmp_path / f"{n}.json"
        path.write_text(json.dumps({"call": 1, "input_token": 3}))
        tokencount.add_file(str(path))
    stop.set()
    reader.join()
    assert set(seen) == {1000}


def test_the_line_names_every_count():
    assert tokencount.line({"call": 3, "input_token": 1200, "output_token": 300,
                            "cache_read_token": 800, "cache_write_token": 100,
                            "total": 1500}) == (
        "1,500 in 3 call(s): 1,200 input (800 read from the cache, 100 written to it), "
        "300 output")


def test_what_is_left_of_the_limit_is_at_least_1_and_0_without_a_limit(monkeypatch):
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 0)
    assert tokencount.remain_token() == 0
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 100)
    assert tokencount.remain_token() == 100
    tokencount.add({"input_token": 60, "output_token": 10})
    assert tokencount.remain_token() == 30
    tokencount.add({"input_token": 500})
    assert tokencount.remain_token() == 1


def test_what_a_file_added_is_not_taken_off_what_is_left(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 100)
    (tmp_path / "token.json").write_text(json.dumps({"call": 1, "input_token": 40}))
    tokencount.add_file(str(tmp_path / "token.json"), part="edit r1")
    tokencount.add({"input_token": 10})
    assert tokencount.total_dict()["total"] == 50 and tokencount.remain_token() == 90
    tokencount.reset()
    assert tokencount.remain_token() == 100


def test_the_files_of_the_processes_started_are_made_in_one_directory(tmp_path,
                                                                      monkeypatch):
    monkeypatch.setattr(tokencount.tempfile, "tempdir", str(tmp_path))
    first, second = tokencount.child_file(), tokencount.child_file()
    assert first != second and os.path.dirname(first) == os.path.dirname(second)
    assert os.path.dirname(os.path.dirname(first)) == str(tmp_path)
    assert os.path.getsize(first) == 0
    tokencount.reset()
    assert os.listdir(tmp_path) == []
    # A file asked for afterwards is made in a directory of its own again
    assert os.path.exists(tokencount.child_file())


def test_the_limit_is_held_against_what_the_processes_still_going_have_written(
        tmp_path, monkeypatch):
    monkeypatch.setattr(tokencount.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(config, "MAX_TOKEN_PER_RUN", 100)
    over, going = tokencount.child_file(), tokencount.child_file()
    with open(over, "w", encoding="utf-8") as handle:
        json.dump({"call": 1, "input_token": 40}, handle)
    tokencount.add_file(over, part="edit r1")
    tokencount.add({"input_token": 10})
    assert tokencount.limit_text() == ""
    with open(going, "w", encoding="utf-8") as handle:
        json.dump({"call": 1, "input_token": 50}, handle)
    assert "have used 100 token(s)" in tokencount.limit_text()
    # A file added from somewhere else is counted once
    tokencount.reset()
    (tmp_path / "token.json").write_text(json.dumps({"call": 1, "input_token": 100}))
    tokencount.add_file(str(tmp_path / "token.json"))
    assert "have used 100 token(s)" in tokencount.limit_text()


def test_a_file_that_cannot_be_made_is_logged_and_named_empty(monkeypatch, caplog):
    def refuse(**kwargs):
        raise OSError("no space")

    monkeypatch.setattr(tokencount.tempfile, "mkdtemp", refuse)
    assert tokencount.child_file() == ""
    assert "are not counted" in caplog.text


def test_the_total_beside_a_file_is_this_process_s_and_the_other_files(tmp_path,
                                                                       monkeypatch):
    monkeypatch.setattr(tokencount.tempfile, "tempdir", str(tmp_path))
    own, other, empty = (tokencount.child_file() for _ in range(3))
    tokencount.add({"input_token": 10, "output_token": 2})
    assert tokencount.total_beside("") == 12 and tokencount.total_beside(own) == 12
    tokencount.write(own)
    with open(other, "w", encoding="utf-8") as handle:
        json.dump({"call": 2, "input_token": 30, "output_token": 5,
                   "cache_read_token": 20}, handle)
    # A file being written and a file that is not a count are passed over
    for name in (f"{os.path.basename(other)}.123.tmp", "note.txt"):
        with open(os.path.join(os.path.dirname(own), name), "w", encoding="utf-8") as handle:
            json.dump({"input_token": 1000}, handle)
    assert os.path.getsize(empty) == 0
    assert tokencount.total_beside(own) == 12 + 35
    assert tokencount.total_beside(str(tmp_path / "missing" / "own.json")) == 12
