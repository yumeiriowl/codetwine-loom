"""The record a command that starts containers leaves, and the removal of old ones."""

import json
import os

import pytest

from codetwine_loom import runrecord


def _write(directory, record_id, **field_dict):
    """Write one record and return its path."""
    return runrecord.write(str(directory), {"id": record_id, **field_dict})


def test_a_record_is_written_under_its_id_and_read_back(tmp_path):
    directory = tmp_path / "sandbox" / "record"
    path = _write(directory, "20260101-000000-1-1", command="run", request="依頼")
    assert path == str(directory / "20260101-000000-1-1.json")
    assert runrecord.read_list(str(directory)) == [
        {"id": "20260101-000000-1-1", "command": "run", "request": "依頼", "path": path}]
    assert "依頼" in (directory / "20260101-000000-1-1.json").read_text(encoding="utf-8")


def test_a_value_json_cannot_write_is_written_as_text(tmp_path):
    _write(tmp_path, "a", result={"where": tmp_path})
    assert runrecord.read_list(str(tmp_path))[0]["result"] == {"where": str(tmp_path)}


def test_a_character_utf8_cannot_encode_is_written_as_its_escape(tmp_path):
    path = _write(tmp_path, "a", request="name: \udce3\udc81")
    assert runrecord.read_list(str(tmp_path))[0]["request"] == "name: \udce3\udc81"
    with open(path, encoding="utf-8") as handle:
        assert "name: \\udce3\\udc81" in handle.read()


def test_a_record_that_cannot_be_written_is_logged_and_not_raised(tmp_path, caplog):
    (tmp_path / "record").write_text("a file, not a directory", encoding="utf-8")
    assert _write(tmp_path / "record", "a") == ""
    assert "could not be written" in caplog.text


def test_the_records_come_back_oldest_first_and_a_broken_one_is_left_out(tmp_path, caplog):
    _write(tmp_path, "20260102-000000-1-1")
    _write(tmp_path, "20260101-000000-1-1")
    (tmp_path / "20260103-000000-1-1.json").write_text("{", encoding="utf-8")
    (tmp_path / "20260104-000000-1-1.json").write_text("[]", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert [record["id"] for record in runrecord.read_list(str(tmp_path))] == [
        "20260101-000000-1-1", "20260102-000000-1-1"]
    assert "could not be read" in caplog.text


@pytest.mark.parametrize("field_dict", [
    {"token": [1]}, {"token": "many"}, {"token": {"total": None}},
    {"token": {"total": "12"}}, {"token": {"total": float("nan")}},
    {"elapsed_sec": "1.0"}, {"elapsed_sec": [1]}, {"elapsed_sec": float("inf")},
    {"elapsed_sec": True}, {"command": ["run"]}, {"command": {"a": 1}},
    {"resume": 5}, {"resume": "codetwine-loom/run/r/1"}, {"resume": ["codetwine-loom/run/r/1", 5]},
    {"resume": {"a": 1}}, {"request": 5}, {"error": {"a": 1}}, {"exit_status": None},
    {"error_command": ["git"]}, {"error_command": 5},
    {"exit_status": "1"}, {"id": ["a"]}, {"result": "none"}, {"result": [1]},
    {"resume": ["a\x00b"]}, {"resume": ["codetwine-loom/run/r/1", "\ud800"]},
    {"result": {"branch": "x\x00y"}}, {"result": {"branch": "\udce3"}},
    {"result": {"integration": {"land_refuse": True, "branch": "x\x00y"}}},
    {"result": {"integration": {"branch": "\ud800"}, "branch": None}},
    {"result": {"pass_branch": ["x\x00y"]}}, {"result": {"pass_branch": ["x\ud800y"]}},
    {"result": {"pass_branch": "main"}}, {"result": {"pass_branch": ["fc", None]}},
    {"result": {"pass_branch": [["a"]]}}, {"result": {"pass_branch": {"b": 1}}},
    {"result": {"stand_for": ["x\x00y"]}}, {"result": {"stand_for": "codetwine-loom/sandbox/u"}},
    {"result": {"stand_for": [5]}},
])
def test_a_record_of_another_shape_is_left_out(tmp_path, caplog, field_dict):
    """What the readers of a record take off it is of the shape they read it in, or
    the record is not read."""
    _write(tmp_path, "20260101-000000-1-1", command="run", resume=["b"])
    path = tmp_path / "20260102-000000-1-1.json"
    path.write_text(json.dumps({"id": "20260102-000000-1-1", **field_dict}),
                    encoding="utf-8")
    assert [record["id"] for record in runrecord.read_list(str(tmp_path))] == [
        "20260101-000000-1-1"]
    assert f"the record {path} is not of the shape" in caplog.text
    # It is neither asked about nor removed
    asked = []
    assert runrecord.remove_old(str(tmp_path), 1, lambda record: asked.append(
        record["id"]) or False) == []
    assert asked == [] and path.is_file()


@pytest.mark.parametrize("field_dict", [
    {}, {"id": "a", "command": "run"}, {"token": {}}, {"token": {"total": 1.5}},
    {"error": "why", "error_command": "git -C '/a  b' apply"},
    {"elapsed_sec": 2}, {"resume": []}, {"exit_status": 130, "error": "", "request": ""},
    {"result": {"integration": None}}, {"files": 5, "report": None},
    {"resume": ["codetwine-loom/run/r/1", "a b", "--help", "日本語"]},
    {"result": {"branch": None, "integration": {"branch": ["b"], "land_why": 5}}},
    {"result": {"branch": "codetwine-loom/sandbox/r", "integration": "s", "answer": "a\x00\ud83d"},
     "request": "r\x00", "error": "e\udce3"},
    {"result": {"pass_branch": []}}, {"result": {"pass_branch": ["fc", "a b", "-h", ""]}},
    {"result": {"stand_for": ["codetwine-loom/sandbox/20260101-000000-5-1"]}},
])
def test_a_record_holding_fewer_keys_or_others_is_read(tmp_path, field_dict):
    (tmp_path / "a.json").write_text(json.dumps(field_dict), encoding="utf-8")
    assert runrecord.read_list(str(tmp_path)) == [
        {**field_dict, "path": str(tmp_path / "a.json")}]


def test_a_record_nested_deeper_than_is_read_is_left_out(tmp_path, caplog):
    (tmp_path / "a.json").write_text("[" * 200000, encoding="utf-8")
    (tmp_path / "b.json").write_bytes(b"\xff\xfe")
    (tmp_path / "c.json").mkdir()
    assert runrecord.read_list(str(tmp_path)) == []
    assert caplog.text.count("could not be read") == 3


def test_a_directory_that_is_not_there_holds_no_record(tmp_path):
    assert runrecord.read_list(str(tmp_path / "none")) == []


def test_the_records_beyond_the_newest_few_are_removed(tmp_path):
    for day in range(1, 6):
        _write(tmp_path, f"2026010{day}-000000-1-1")
    removed = runrecord.remove_old(str(tmp_path), 2, lambda record: False)
    assert [os.path.basename(path) for path in removed] == [
        "20260101-000000-1-1.json", "20260102-000000-1-1.json",
        "20260103-000000-1-1.json"]
    assert [record["id"] for record in runrecord.read_list(str(tmp_path))] == [
        "20260104-000000-1-1", "20260105-000000-1-1"]


def test_an_old_record_that_can_be_carried_on_from_stays(tmp_path):
    for day in range(1, 5):
        _write(tmp_path, f"2026010{day}-000000-1-1", resume=["b"] if day == 1 else [])
    runrecord.remove_old(str(tmp_path), 1, lambda record: bool(record["resume"]))
    assert [record["id"] for record in runrecord.read_list(str(tmp_path))] == [
        "20260101-000000-1-1", "20260104-000000-1-1"]


def test_a_keep_count_of_0_removes_nothing(tmp_path):
    _write(tmp_path, "a")
    _write(tmp_path, "b")
    assert runrecord.remove_old(str(tmp_path), 0, lambda record: False) == []
    assert len(runrecord.read_list(str(tmp_path))) == 2


def test_the_file_is_one_json_object(tmp_path):
    path = _write(tmp_path, "a", exit_status=0)
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle) == {"id": "a", "exit_status": 0}
