"""A turn's reply received as a stream and ended where rlm.reply_end() says.

A local HTTP server stands in for the provider and streams a scripted reply in the
Anthropic, OpenAI and Ollama formats, so litellm's own stream handling is exercised.
"""

import asyncio
import contextlib
import io
import json
import signal
import socket
import struct
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import httpx2
import litellm
import openai
import pytest
from litellm.llms.custom_httpx import http_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

from codetwine_loom import broker, config, providercall, rlm, tokencount

_CALL = '<invoke name="python">\n<parameter name="code">print(1)</parameter>\n</invoke>'
_MORE = '\n\n<invoke name="python">\n<parameter name="code">pass</parameter>\n</invoke>'

# (model, the path its stream is asked for at)
_PROVIDER_TUPLE = (("anthropic/claude-haiku-4-5", "/v1/messages"),
                   ("openai/some-model", "/chat/completions"),
                   ("ollama_chat/some-model", "/api/chat"),
                   ("ollama/some-model", "/api/generate"))


class _Server(ThreadingHTTPServer):
    """A local server that ends the connections it took when it is closed."""

    daemon_threads = True

    def __init__(self, *arg_tuple):
        super().__init__(*arg_tuple)
        self.connection_list = []

    def get_request(self):
        connection, address = super().get_request()
        self.connection_list.append(connection)
        return connection, address

    def server_close(self):
        super().server_close()
        for connection in self.connection_list:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)


@pytest.fixture(scope="module", autouse=True)
def _no_request_thread_left():
    """Wait, once the tests of this file are over, for the threads the servers answered
    on."""
    thread_set = set(threading.enumerate())
    yield
    for thread in set(threading.enumerate()) - thread_set:
        if "process_request_thread" in thread.name:
            thread.join(70)


class _Handler(BaseHTTPRequestHandler):
    """Stream the server's pieces in the format of the path that was asked for."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *arg_tuple):
        pass

    def _write(self, text: str) -> None:
        data = text.encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _event(self, name: str, field_dict: dict) -> None:
        self._write(f"event: {name}\ndata: {json.dumps({'type': name, **field_dict})}\n\n")

    def _json(self, status: int, body_dict: dict) -> None:
        data = json.dumps(body_dict).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _anthropic(self, piece_list: list[str], record: dict) -> None:
        self._event("message_start", {"message": {
            "id": "m", "type": "message", "role": "assistant", "model": "x",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 20, "cache_read_input_tokens": 80,
                      "cache_creation_input_tokens": 5, "output_tokens": 1}}})
        self._event("content_block_start",
                    {"index": 0, "content_block": {"type": "text", "text": ""}})
        for piece in piece_list:
            self._event("content_block_delta",
                        {"index": 0, "delta": {"type": "text_delta", "text": piece}})
            self._sent(record)
        self._event("content_block_stop", {"index": 0})
        self._event("message_delta", {"delta": {"stop_reason": "max_tokens",
                                                "stop_sequence": None},
                                      "usage": {"output_tokens": 50}})
        self._event("message_stop", {})

    def _openai(self, piece_list: list[str], record: dict) -> None:
        def chunk(delta_dict, finish_reason):
            return "data: " + json.dumps({
                "id": "c", "object": "chat.completion.chunk", "model": "x",
                "choices": [{"index": 0, "delta": delta_dict,
                             "finish_reason": finish_reason}]}) + "\n\n"
        for index, piece in enumerate(piece_list):
            self._write(chunk(self._tool_delta(index, piece) if self.server.is_tool_call
                              else {"content": piece}, None))
            self._sent(record)
        self._write(chunk({}, "length"))
        # The token counts come only when the request asks for them
        if (record["body"].get("stream_options") or {}).get("include_usage"):
            self._write("data: " + json.dumps({
                "id": "c", "object": "chat.completion.chunk", "model": "x", "choices": [],
                "usage": {"prompt_tokens": 105, "completion_tokens": 50,
                          "total_tokens": 155,
                          "prompt_tokens_details": {"cached_tokens": 80}}}) + "\n\n")
        self._write("data: [DONE]\n\n")

    @staticmethod
    def _tool_delta(index: int, piece: str) -> dict:
        """One piece of a reply, sent as the arguments of a tool call."""
        call_dict = {"index": 0, "function": {"arguments": piece}}
        if index == 0:
            call_dict.update({"id": "call_1", "type": "function"})
            call_dict["function"]["name"] = "python"
        return {"tool_calls": [call_dict]}

    def _ollama(self, piece_list: list[str], record: dict) -> None:
        def row(piece, field_dict):
            text_dict = ({"message": {"role": "assistant", "content": piece}}
                         if self.path.endswith("/api/chat") else {"response": piece})
            return json.dumps({"model": "x", **text_dict, **field_dict}) + "\n"
        for piece in piece_list:
            self._write(row(piece, {"done": False}))
            self._sent(record)
        self._write(row("", {"done": True, "done_reason": "length",
                             "prompt_eval_count": 20, "eval_count": 50}))

    def _sent(self, record: dict) -> None:
        record["sent"] += 1
        if record["sent"] == self.server.silent_after:
            time.sleep(60)
        if record["sent"] == self.server.break_after and len(self.server.record_list) == 1:
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            raise BrokenPipeError
        time.sleep(0.002)

    def do_POST(self):
        body_dict = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if self.path.endswith("/api/show"):
            return self._json(200, {"template": "", "model_info": {}})
        record = {"path": self.path, "body": body_dict, "sent": 0, "is_client_gone": False}
        self.server.record_list.append(record)
        if self.server.error_status == 400:
            return self._json(400, {"type": "error", "error": {
                "type": "invalid_request_error", "message": "thinking is not supported"}})
        if self.server.error_status or self.server.fail_first:
            self.server.fail_first = max(0, self.server.fail_first - 1)
            return self._json(self.server.error_status or 502, {"type": "error", "error": {
                "type": "api_error", "message": "try again later"}})
        if not body_dict.get("stream"):
            return self._json(200, {
                "id": "m", "type": "message", "role": "assistant", "model": "x",
                "content": [{"type": "text", "text": "".join(self.server.piece_list)}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 7}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if "api" not in self.path
                         else "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        write = (self._anthropic if self.path.endswith("/messages")
                 else self._openai if self.path.endswith("/chat/completions")
                 else self._ollama)
        try:
            write(self.server.piece_list, record)
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            record["is_client_gone"] = True


@pytest.fixture
def provider(monkeypatch):
    """Serve a scripted stream on a local port. A test sets piece_list before the call."""
    server = _Server(("127.0.0.1", 0), _Handler)
    server.piece_list = []
    server.record_list = []
    server.silent_after = server.break_after = None
    server.error_status = None
    server.fail_first = 0
    server.is_tool_call = False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "LLM_RETRY_WAIT_SEC", 0)
    yield server
    server.shutdown()
    server.server_close()


def _pieces(text: str, size: int = 7) -> list[str]:
    """Cut a reply into the pieces the server streams."""
    return [text[i:i + size] for i in range(0, len(text), size)]


def _complete(server, model: str = _PROVIDER_TUPLE[0][0], **param_dict) -> object:
    """Make one call marked as a turn's reply against the server."""
    return providercall.completion({
        "model": model, "max_tokens": 100, "api_key": "k", "num_retries": 0,
        "api_base": f"http://127.0.0.1:{server.server_address[1]}", "timeout": 30,
        "messages": [{"role": "user", "content": "hi"}],
        providercall.TURN_REPLY_KEY: True, **param_dict})


def _wait_for_the_server(record: dict) -> None:
    """Wait until the server has seen the client close the stream."""
    limit = time.monotonic() + 5
    while not record["is_client_gone"] and time.monotonic() < limit:
        time.sleep(0.01)


@pytest.mark.parametrize("model, path", _PROVIDER_TUPLE)
def test_a_reply_that_writes_on_is_ended_after_its_first_call(provider, model, path):
    provider.piece_list = _pieces(_CALL + _MORE * 200)
    response = _complete(provider, model)
    choice = response.choices[0]
    assert (choice.message.content, choice.finish_reason) == (_CALL, "stop")
    record = provider.record_list[-1]
    assert record["path"].endswith(path) and record["body"]["stream"] is True
    assert not [name for name in record["body"] if name.startswith("stop")]
    _wait_for_the_server(record)
    assert record["is_client_gone"] and record["sent"] < len(provider.piece_list) / 2


@pytest.mark.parametrize("model, path", _PROVIDER_TUPLE)
def test_a_reply_that_starts_the_next_turn_is_ended_in_front_of_it(provider, model, path):
    block = "```python\nprint(1)\n```"
    provider.piece_list = _pieces(block + "\n### Turn 2 output\n```\n1\n```\n" * 100)
    choice = _complete(provider, model).choices[0]
    assert (choice.message.content, choice.finish_reason) == (block, "stop")


@pytest.mark.parametrize("model, path", _PROVIDER_TUPLE)
def test_a_reply_with_nowhere_to_end_is_received_whole(provider, model, path):
    reply = "```python\nprint(1)\n```\n```python\nprint(2)\n```"
    provider.piece_list = _pieces(reply)
    response = _complete(provider, model)
    assert response.choices[0].message.content == reply
    assert not provider.record_list[-1]["is_client_gone"]


def test_a_call_whose_code_does_not_compile_is_ended_after_it(provider):
    bad = '<invoke name="python">\n<parameter name="code">print(1\n</parameter>\n</invoke>'
    provider.piece_list = _pieces(bad + ("\n\n" + bad) * 200)
    choice = _complete(provider).choices[0]
    assert (choice.message.content, choice.finish_reason) == (bad, "stop")


def test_a_heading_inside_a_string_of_the_code_does_not_end_the_reply(provider):
    reply = ('```python\nwrite_file("rules.md", """# Rules\n\n## Turn 1\nDraw a card.\n'
             '""")\nprint("ok")\n```')
    provider.piece_list = _pieces(reply)
    choice = _complete(provider).choices[0]
    assert choice.message.content == reply


def test_a_heading_in_the_last_piece_is_cut_off_when_the_stream_ends(provider):
    block = "```python\n" + "x = 1\n" * 200 + "```"
    provider.piece_list = [block, "\n### Turn 2 output\n```\n1\n```\n"]
    choice = _complete(provider).choices[0]
    assert (choice.message.content, choice.finish_reason) == (block, "stop")


def test_a_long_reply_is_not_looked_at_on_every_piece(provider, monkeypatch):
    call_list = []
    reply_end = rlm.reply_end
    monkeypatch.setattr(rlm, "reply_end",
                        lambda text: call_list.append(len(text)) or reply_end(text))
    provider.piece_list = _pieces("```python\n" + "x = 1\n" * 2000)
    _complete(provider, max_tokens=4000)
    assert len(call_list) < len(provider.piece_list) // 10
    assert call_list[-1] == len("".join(provider.piece_list))


@pytest.mark.parametrize("max_token, max_char", [
    (100, 3200), (1, 32), (0, 0), (-5, 0), (None, 0), ("100", 0), (True, 0), (2.5, 0)])
def test_a_reply_may_hold_so_many_characters_for_each_token_asked_for(max_token,
                                                                       max_char):
    assert providercall.reply_max_char(max_token) == max_char


@pytest.mark.parametrize("model, path", _PROVIDER_TUPLE)
def test_a_stream_that_grows_past_what_its_tokens_cover_is_closed_there(provider, model,
                                                                        path, caplog):
    """A provider that writes on past max_tokens is not read to its end: the stream
    is closed and the reply ends there as one cut for its length."""
    reply = "word " * 20000
    provider.piece_list = _pieces(reply, 50)
    response = _complete(provider, model, max_tokens=100)
    choice = response.choices[0]
    assert choice.message.content == reply[:3200] and choice.finish_reason == "length"
    assert broker.usage_dict_of(response.usage)["estimate_count"] == 1
    _wait_for_the_server(provider.record_list[-1])
    assert provider.record_list[-1]["is_client_gone"]
    assert provider.record_list[-1]["sent"] < len(provider.piece_list) // 2
    assert "grew past the 3200 characters its max_tokens covers" in caplog.text


def test_a_reply_that_ends_before_its_tokens_run_out_is_ended_where_it_ends(provider):
    """What rlm.reply_end() finds in front of the limit is where the reply ends."""
    provider.piece_list = _pieces(_CALL + _MORE * 2000, 50)
    choice = _complete(provider, max_tokens=100).choices[0]
    assert (choice.message.content, choice.finish_reason) == (_CALL, "stop")


def test_a_whole_reply_past_what_its_tokens_cover_is_cut_there(provider, caplog):
    reply = "word " * 20000
    provider.piece_list = [reply]
    response = _complete(provider, max_tokens=100, **{providercall.TURN_REPLY_KEY: False})
    choice = response.choices[0]
    assert choice.message.content == reply[:3200] and choice.finish_reason == "length"
    assert "grew past the 3200 characters" in caplog.text
    provider.piece_list = [reply[:3200]]
    response = _complete(provider, max_tokens=100, **{providercall.TURN_REPLY_KEY: False})
    assert response.choices[0].message.content == reply[:3200]
    assert response.choices[0].finish_reason != "length"


def test_thinking_that_grows_past_what_the_tokens_cover_closes_the_stream():
    """The thinking counts with the content: a stream of thinking alone is closed too."""
    class Stream:
        def __init__(self):
            self.count, self.is_closed = 0, False

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.count += 1
            delta = types.SimpleNamespace(reasoning_content="think " * 10, content=None)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])

        async def aclose(self):
            self.is_closed = True

    stream, piece_list = Stream(), []
    chunk_list, text, end = asyncio.run(providercall._read_to_end(
        stream, lambda text: None, "m", piece_list, 320))
    assert (len(chunk_list), text, end, stream.is_closed) == (6, "", None, True)
    assert sum(len(piece) for piece in piece_list) == 360


@pytest.mark.parametrize("delta_of", [
    lambda piece: types.SimpleNamespace(content=None, tool_calls=[types.SimpleNamespace(
        function=types.SimpleNamespace(arguments=piece))]),
    lambda piece: types.SimpleNamespace(content=None, tool_calls=[
        {"function": {"arguments": piece[:30]}}, {"function": {"arguments": piece[30:]}}]),
    lambda piece: types.SimpleNamespace(content=None, tool_calls=None,
                                        function_call={"arguments": piece}),
    lambda piece: types.SimpleNamespace(content=None, function_call=types.SimpleNamespace(
        arguments=piece))])
def test_tool_call_arguments_that_grow_past_what_the_tokens_cover_close_the_stream(
        delta_of):
    """The arguments of a tool call count with the thinking and the content."""
    class Stream:
        def __init__(self):
            self.count, self.is_closed = 0, False

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.count += 1
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=delta_of("argue " * 10))])

        async def aclose(self):
            self.is_closed = True

    stream, piece_list = Stream(), []
    chunk_list, text, end = asyncio.run(providercall._read_to_end(
        stream, lambda text: None, "m", piece_list, 320))
    assert (len(chunk_list), text, end, stream.is_closed) == (6, "", None, True)
    assert sum(len(piece) for piece in piece_list) == 360


@pytest.mark.parametrize("delta", [
    None, types.SimpleNamespace(content="x"),
    types.SimpleNamespace(tool_calls=[None, {"function": None}, {"function": {}},
                                      types.SimpleNamespace(function=types.SimpleNamespace(
                                          name="f", arguments=None))],
                          function_call={"arguments": 3}),
    types.SimpleNamespace(tool_calls="not a list", function_call="nor an object")])
def test_a_chunk_that_carries_no_tool_call_arguments_adds_none(delta):
    assert providercall._call_argument(delta) == ""


def test_a_stream_of_tool_call_arguments_ends_as_a_reply_cut_at_its_length(provider):
    """A reply that arrives as the arguments of a tool call is closed at the limit of
    its max_tokens and comes back with no content and the finish reason "length"."""
    provider.piece_list = ["x" * 100] * 1000
    provider.is_tool_call = True
    response = _complete(provider, _PROVIDER_TUPLE[1][0], max_tokens=100)
    assert response.choices[0].finish_reason == "length"
    assert not response.choices[0].message.content
    record = provider.record_list[-1]
    _wait_for_the_server(record)
    assert record["is_client_gone"] and record["sent"] < 500


def test_an_ended_reply_keeps_the_counts_the_provider_had_sent(provider):
    provider.piece_list = _pieces(_CALL + _MORE * 200)
    usage_dict = broker.usage_dict_of(_complete(provider).usage)
    assert usage_dict["input_token"] == 105
    assert (usage_dict["cache_read_token"], usage_dict["cache_write_token"]) == (80, 5)
    # The provider had reported 1; what arrived is counted here
    assert 1 < usage_dict["output_token"] <= litellm.token_counter(
        model=_PROVIDER_TUPLE[0][0], text=_CALL + _MORE)


def test_a_whole_reply_reports_what_the_provider_ends_with(provider):
    provider.piece_list = _pieces("```python\nprint(1)\n```")
    response = _complete(provider)
    assert response.choices[0].finish_reason == "length"
    usage_dict = broker.usage_dict_of(response.usage)
    assert (usage_dict["input_token"], usage_dict["output_token"]) == (105, 50)


@pytest.mark.parametrize("model", ["anthropic/claude-haiku-4-5", "openai/some-model"])
def test_a_whole_reply_keeps_the_input_and_cache_counts_the_provider_sent(provider, model):
    provider.piece_list = _pieces("```python\nprint(1)\n```")
    usage_dict = broker.usage_dict_of(_complete(provider, model).usage)
    assert (usage_dict["input_token"], usage_dict["cache_read_token"]) == (105, 80)


@pytest.mark.parametrize("model, path", _PROVIDER_TUPLE)
def test_an_ended_reply_marks_its_counts_as_estimates(provider, model, path):
    provider.piece_list = _pieces(_CALL + _MORE * 200)
    usage_dict = broker.usage_dict_of(_complete(provider, model).usage)
    assert (usage_dict["estimate_count"], usage_dict["retry_count"]) == (1, 0)


@pytest.mark.parametrize("model, path", _PROVIDER_TUPLE)
def test_a_whole_reply_with_the_providers_counts_is_not_an_estimate(provider, model, path):
    provider.piece_list = _pieces("```python\nprint(1)\n```")
    assert broker.usage_dict_of(_complete(provider, model).usage)["estimate_count"] == 0


def test_a_stream_whose_last_chunk_has_no_counts_is_an_estimate(provider, monkeypatch):
    provider.piece_list = _pieces("```python\nprint(1)\n```")
    monkeypatch.setattr(providercall, "STREAM_OPTION_DICT", {"include_usage": False})
    usage_dict = broker.usage_dict_of(_complete(provider, "openai/some-model").usage)
    assert usage_dict["estimate_count"] == 1


def test_the_attempts_made_before_the_answer_are_counted(provider):
    provider.piece_list = _pieces(_CALL + _MORE)
    provider.break_after = 3
    usage_dict = broker.usage_dict_of(_complete(provider, num_retries=1).usage)
    assert usage_dict["retry_count"] == 1


def test_what_an_attempt_made_again_had_received_is_counted_as_an_estimate(provider):
    provider.piece_list = _pieces("```python\nprint('" + "x" * 300 + "')\n```")
    provider.break_after = 30
    usage_dict = broker.usage_dict_of(_complete(provider, num_retries=1).usage)
    assert len(provider.record_list) == 2
    assert usage_dict["output_token"] > 50 + 20
    assert (usage_dict["estimate_count"], usage_dict["retry_count"]) == (1, 1)


def test_an_attempt_that_failed_before_anything_arrived_adds_no_token(provider, caplog):
    provider.piece_list = _pieces("```python\nprint(1)\n```")
    provider.fail_first = 1
    caplog.set_level("INFO", logger="codetwine_loom.providercall")
    usage_dict = broker.usage_dict_of(_complete(provider, num_retries=1).usage)
    assert (usage_dict["output_token"], usage_dict["estimate_count"],
            usage_dict["retry_count"]) == (50, 0, 1)
    assert "before any of the reply arrived" in caplog.text


def test_every_call_is_added_to_the_counts_of_this_process(provider):
    provider.piece_list = [_CALL]
    _complete(provider, **{providercall.TURN_REPLY_KEY: False})
    first = tokencount.total_dict()
    assert first["call"] == 1 and first["input_token"] > 0 and first["output_token"] > 0
    _complete(provider, **{providercall.TURN_REPLY_KEY: False})
    assert tokencount.total_dict()["call"] == 2
    assert tokencount.total_dict()["total"] == 2 * first["total"]


def test_a_call_that_failed_is_not_counted(provider):
    provider.error_status = 400
    with pytest.raises(Exception):
        _complete(provider)
    assert tokencount.total_dict()["call"] == 0


def test_a_call_answered_at_once_counts_no_attempt_before_it(provider):
    provider.piece_list = [_CALL]
    response = _complete(provider, **{providercall.TURN_REPLY_KEY: False})
    assert broker.usage_dict_of(response.usage)["retry_count"] == 0


def test_a_call_not_marked_as_a_turn_is_not_received_as_a_stream(provider):
    provider.piece_list = [_CALL + _MORE]
    response = _complete(provider, **{providercall.TURN_REPLY_KEY: False})
    assert response.choices[0].message.content == _CALL + _MORE
    assert not provider.record_list[-1]["body"].get("stream")


def test_a_stream_that_goes_silent_runs_out_of_the_attempt_and_is_asked_again(provider):
    provider.piece_list = _pieces(_CALL[:40])
    provider.silent_after = 2
    with pytest.raises(litellm.Timeout):
        _complete(provider, timeout=0.5, num_retries=1)
    assert len(provider.record_list) == 2


def test_a_stream_that_goes_silent_past_the_idle_limit_is_asked_again_early(provider,
                                                                           monkeypatch):
    monkeypatch.setattr(config, "LLM_IDLE_TIMEOUT_SEC", 0.3)
    provider.piece_list = _pieces(_CALL[:40])
    provider.silent_after = 2
    start = time.monotonic()
    with pytest.raises(litellm.Timeout, match="no part of the reply arrived for 0.3 s"):
        _complete(provider, timeout=30, num_retries=1)
    assert len(provider.record_list) == 2
    assert time.monotonic() - start < 10


def test_a_stream_that_keeps_arriving_is_not_cut_by_the_idle_limit(provider, monkeypatch):
    monkeypatch.setattr(config, "LLM_IDLE_TIMEOUT_SEC", 0.5)
    reply = "```python\n" + "x = 1\n" * 200 + "```"
    provider.piece_list = _pieces(reply)
    assert _complete(provider).choices[0].message.content == reply
    assert len(provider.record_list) == 1


@pytest.mark.parametrize("error_type", [TimeoutError, rlm.TurnTimeout])
def test_the_turn_deadline_while_waiting_closes_the_stream_and_is_not_asked_again(
        provider, error_type):
    provider.piece_list = _pieces(_CALL[:40])
    provider.silent_after = 2

    def ring(signal_number, frame):
        raise error_type("the turn did not finish")

    before = signal.signal(signal.SIGALRM, ring)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        with pytest.raises(error_type) as raised:
            _complete(provider, timeout=30, num_retries=2)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, before)
    assert type(raised.value) is error_type
    assert len(provider.record_list) == 1
    assert _running_stream_count() == 0


async def _other_task_count() -> int:
    """Count the tasks on the running loop other than this one."""
    return len(asyncio.all_tasks()) - 1


def _running_stream_count() -> int:
    """Return how many streams are still being read, once they have had time to close."""
    count = 0
    for _ in range(50):
        count = asyncio.run_coroutine_threadsafe(_other_task_count(),
                                                 providercall._stream_loop()).result()
        if not count:
            return 0
        time.sleep(0.1)
    return count


def test_a_stream_that_breaks_on_the_way_is_asked_again(provider):
    provider.piece_list = _pieces(_CALL + _MORE)
    provider.break_after = 3
    assert _complete(provider, num_retries=1).choices[0].message.content == _CALL
    assert len(provider.record_list) == 2


def test_a_refusal_of_a_stream_is_raised_and_not_asked_again(provider):
    provider.error_status = 400
    with pytest.raises(litellm.BadRequestError):
        _complete(provider, num_retries=3)
    assert len(provider.record_list) == 1


@pytest.mark.parametrize("model, path", _PROVIDER_TUPLE)
@pytest.mark.parametrize("status, request_count", [(502, 3), (529, 3), (409, 1)])
def test_a_stream_answered_with_5xx_is_asked_again_and_other_statuses_are_not(
        provider, model, path, status, request_count):
    provider.error_status = status
    with pytest.raises(openai.APIError):
        _complete(provider, model, num_retries=2)
    assert len(provider.record_list) == request_count


def test_the_broker_ends_a_turn_reply_and_takes_the_mark(provider, monkeypatch):
    provider.piece_list = _pieces(_CALL + _MORE * 200)
    monkeypatch.setattr(config, "LLM_API_KEY", "scripted-key")
    monkeypatch.setattr(config, "LLM_API_BASE",
                        f"http://127.0.0.1:{provider.server_address[1]}")
    assert providercall.TURN_REPLY_KEY in broker.COMPLETION_PARAM_TUPLE
    result_dict = broker._call_the_provider({
        "model": _PROVIDER_TUPLE[0][0], "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        providercall.TURN_REPLY_KEY: True})
    assert (result_dict["content"], result_dict["finish_reason"]) == (_CALL, "stop")
    assert result_dict["usage"]["cache_read_token"] == 80


def test_half_of_a_surrogate_pair_in_a_prompt_is_sent_as_the_no_character_mark(
        monkeypatch):
    """The call is made once and is not left to fail where the request is encoded."""
    send_list = []

    def completion(**kwarg_dict):
        json.dumps(kwarg_dict["messages"], ensure_ascii=False).encode("utf-8")
        send_list.append(kwarg_dict["messages"])
        return types.SimpleNamespace(usage=None)

    monkeypatch.setattr(providercall.litellm, "completion", completion)
    block_list = [{"type": "text", "text": "half \ud83d and whole \U0001f600"}]
    for content, want in (("the search said: \udc00 tail", "the search said: \ufffd tail"),
                          (block_list, [{"type": "text",
                                         "text": "half \ufffd and whole \U0001f600"}])):
        providercall.completion({"model": "m", "max_tokens": 10, "num_retries": 3,
                                 "messages": [{"role": "user", "content": content}]})
        assert send_list[-1] == [{"role": "user", "content": want}]
    assert len(send_list) == 2
    assert block_list[0]["text"] == "half \ud83d and whole \U0001f600"


# == a redirect the endpoint answers with ======================

_FAKE_KEY = "sk-test-not-a-key"


class _RedirectHandler(BaseHTTPRequestHandler):
    """Answer every request with the server's status and Location, or with 404 when
    the server names none, and record which headers held the key. While the server's
    reset_count is above 0, a request other than Ollama's /api/show is read and its
    connection reset in place of an answer."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *arg_tuple):
        pass

    def _answer(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.server.seen_list.append((
            self.command, self.path,
            sorted(name.lower() for name, value in self.headers.items()
                   if _FAKE_KEY in value), b"PROMPT-MARK" in body))
        if self.server.reset_count > 0 and not self.path.endswith("/api/show"):
            self.server.reset_count -= 1
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                       struct.pack("ii", 1, 0))
            self.rfile.close()
            self.wfile.close()
            self.connection.close()
            self.wfile = io.BytesIO()
            self.close_connection = True
            return
        self.send_response(self.server.status)
        if self.server.location:
            self.send_header("Location", self.server.location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_GET = do_POST = _answer


@pytest.fixture
def redirect_pair(monkeypatch):
    """Two local servers: the first answers with a redirect to the second, under
    another host name and port. Returns (the first, the second)."""
    server_list = []
    for _ in range(2):
        server = _Server(("127.0.0.1", 0), _RedirectHandler)
        server.seen_list = []
        server.status, server.location, server.reset_count = 404, "", 0
        threading.Thread(target=server.serve_forever, daemon=True).start()
        server_list.append(server)
    first, other = server_list
    first.location = f"http://localhost:{other.server_address[1]}/moved"
    monkeypatch.setattr(config, "LLM_RETRY_WAIT_SEC", 0)
    yield first, other
    for server in server_list:
        server.shutdown()
        server.server_close()


# The models whose calls litellm sends through each of the HTTP stacks it holds: its own
# handlers, the OpenAI SDK (chat, text completion and the Responses API) and aiohttp
_REDIRECT_MODEL_TUPLE = (
    "anthropic/claude-haiku-4-5", "openai/some-model", "azure/some-model",
    "hosted_vllm/some-model", "ollama_chat/some-model", "ollama/some-model",
    "gemini/gemini-2.5-flash", "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "openai/responses/some-model", "azure/responses/some-model",
    "text-completion-openai/some-model", "aiohttp_openai/some-model")


def _redirect_call(first, model: str, is_turn: bool) -> Exception:
    """Make one call to the first server of redirect_pair and return what it raised."""
    with pytest.raises(Exception) as caught:
        providercall.completion({
            "model": model, "max_tokens": 100, "api_key": _FAKE_KEY, "num_retries": 0,
            "api_base": f"http://127.0.0.1:{first.server_address[1]}", "timeout": 20,
            "messages": [{"role": "user", "content": "PROMPT-MARK"}],
            providercall.TURN_REPLY_KEY: is_turn})
    return caught.value


@pytest.mark.parametrize("status", [302, 307])
@pytest.mark.parametrize("is_turn", [False, True])
@pytest.mark.parametrize("model", _REDIRECT_MODEL_TUPLE)
def test_a_call_whose_endpoint_answers_with_a_redirect_is_sent_nowhere_else(
        redirect_pair, monkeypatch, model, is_turn, status):
    """The key and the prompt go to the endpoint the call names, the address the
    endpoint redirects to is not asked, and the failure does not name it."""
    first, other = redirect_pair
    first.status = status
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    error = _redirect_call(first, model, is_turn)
    chat_list = [seen for seen in first.seen_list if not seen[1].endswith("/api/show")]
    assert len(chat_list) == 1 and chat_list[0][3]
    assert other.seen_list == []
    assert f":{other.server_address[1]}" not in str(error)


@pytest.mark.parametrize("status", [302, 307])
@pytest.mark.parametrize("model", [
    "anthropic/claude-haiku-4-5", "hosted_vllm/some-model", "ollama_chat/some-model",
    "ollama/some-model", "gemini/gemini-2.5-flash",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0"])
def test_a_request_sent_again_after_a_reset_follows_no_redirect_either(
        redirect_pair, monkeypatch, model, status):
    """The first connection of the stream is reset, and the request litellm sends again
    in its place is answered with the redirect."""
    first, other = redirect_pair
    first.status, first.reset_count = status, 1
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    _redirect_call(first, model, True)
    chat_list = [seen for seen in first.seen_list if not seen[1].endswith("/api/show")]
    assert first.reset_count == 0 and chat_list and all(seen[3] for seen in chat_list)
    assert other.seen_list == []


@pytest.mark.parametrize("handler_class", [HTTPHandler, AsyncHTTPHandler])
def test_every_client_a_litellm_handler_hands_out_or_makes_follows_no_redirect(
        handler_class):
    handler = handler_class()
    try:
        assert handler.client.follow_redirects is False
        # A handler that holds a client made another way hands it out the same
        handler._client.follow_redirects = True
        assert handler.client.follow_redirects is False
        made = (handler.create_client() if handler_class is HTTPHandler
                else handler.create_client(timeout=None, event_hooks=None))
        assert made.follow_redirects is False
    finally:
        if handler_class is HTTPHandler:
            handler.close()
            made.close()


def test_a_handler_makes_a_client_that_follows_no_redirect_in_place_of_a_closed_one():
    handler = HTTPHandler()
    handler.client.close()
    assert handler.client.follow_redirects is False and not handler.client.is_closed
    handler.close()


@pytest.mark.parametrize("client_class, http_class", [
    (openai.OpenAI, httpx.Client), (openai.AsyncOpenAI, httpx.AsyncClient),
    (openai.OpenAI, httpx2.Client), (openai.AsyncOpenAI, httpx2.AsyncClient)])
def test_every_openai_sdk_client_sends_with_a_client_that_follows_no_redirect(
        client_class, http_class):
    """The client it is given, and the one it makes when it is given none."""
    own_client = http_class(follow_redirects=True)
    assert client_class(api_key="k", http_client=own_client)._client is own_client
    assert own_client.follow_redirects is False
    assert client_class(api_key="k")._client.follow_redirects is False


def test_litellm_is_given_shared_clients_of_its_openai_calls_where_it_holds_none(
        monkeypatch):
    """A client the process had set is kept; one that cannot be made is left unset and
    made at the next check, and the check itself does not fail for it."""
    from litellm.llms.openai.common_utils import BaseOpenAILLM
    own_client = httpx.AsyncClient(follow_redirects=True)
    monkeypatch.setattr(litellm, "client_session", None)
    monkeypatch.setattr(litellm, "aclient_session", own_client)
    make = BaseOpenAILLM._get_sync_http_client

    def fail():
        raise ValueError("Unknown scheme for proxy URL")
    monkeypatch.setattr(BaseOpenAILLM, "_get_sync_http_client", staticmethod(fail))
    providercall.check_before_call()
    assert litellm.client_session is None and litellm.aclient_session is own_client
    monkeypatch.setattr(BaseOpenAILLM, "_get_sync_http_client", staticmethod(make))
    providercall.check_before_call()
    assert litellm.client_session.follow_redirects is False
    assert litellm.aclient_session is own_client
    litellm.client_session.close()


@pytest.mark.parametrize("model", ["openai/some-model", "azure/some-model"])
def test_a_design_document_call_follows_no_redirect(redirect_pair, monkeypatch, model):
    """codetwine's client, made as knowledge.run_codetwine() makes it, calls litellm
    itself, outside completion()."""
    from codetwine.llm.client import LLMClient

    first, other = redirect_pair
    first.status = 307
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
    client = LLMClient(model=model, api_key=_FAKE_KEY,
                       api_base=f"http://127.0.0.1:{first.server_address[1]}",
                       effort=config.DOC_CREATE_EFFORT)
    asyncio.run(client.generate("PROMPT-MARK", 100))
    assert [seen[3] for seen in first.seen_list] == [True]
    assert other.seen_list == []


@pytest.mark.parametrize("model", ["openai/some-model", "azure/some-model"])
def test_a_stream_after_a_proxy_variable_no_client_reads_follows_no_redirect(
        redirect_pair, monkeypatch, model):
    """The call received whole fails where its client is made, and the stream after it
    is made with another client."""
    first, other = redirect_pair
    first.status = 307
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:9/")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    param_dict = {"model": model, "max_tokens": 100, "api_key": _FAKE_KEY,
                  "api_base": f"http://127.0.0.1:{first.server_address[1]}", "timeout": 20,
                  "num_retries": 0, "messages": [{"role": "user", "content": "PROMPT-MARK"}]}
    with pytest.raises(Exception):
        litellm.completion(**param_dict)

    async def stream():
        async for _ in await litellm.acompletion(**param_dict, stream=True):
            pass
    with pytest.raises(Exception):
        asyncio.run(stream())
    assert other.seen_list == []


def test_a_litellm_whose_clients_cannot_be_kept_from_a_redirect_is_not_called(
        monkeypatch):
    """A handler class in another form leaves a reason, and every call fails with it
    before litellm is asked."""
    call_list = []
    monkeypatch.setattr(providercall.litellm, "completion",
                        lambda **kwarg_dict: call_list.append(kwarg_dict))
    monkeypatch.setattr(providercall, "_redirect_block_text", "")
    monkeypatch.setattr(http_handler, "HTTPHandler", type("HTTPHandler", (), {}))
    providercall._keep_from_redirect()
    with pytest.raises(RuntimeError, match="cannot be kept from following a redirect"):
        providercall.completion({"model": "openai/some-model", "max_tokens": 10,
                                 "num_retries": 3,
                                 "messages": [{"role": "user", "content": "x"}]})
    assert call_list == []
    assert "\n" not in providercall._redirect_block_text


def test_a_replacement_stopped_by_any_error_leaves_the_reason_no_call_is_made(
        monkeypatch):
    """Whatever the error, every check after it refuses the call the same."""
    def stop(*arg_tuple):
        raise LookupError("a form not known")

    monkeypatch.setattr(providercall, "_redirect_block_text", "")
    monkeypatch.setattr(providercall, "_no_redirect_maker", stop)
    providercall._keep_from_redirect()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="LookupError: a form not known"):
            providercall.check_before_call()


def test_a_client_of_another_kind_leaves_the_reason_no_call_is_made(monkeypatch):
    monkeypatch.setattr(providercall, "_redirect_block_text", "")
    with pytest.raises(TypeError):
        providercall._no_redirect(object())
    with pytest.raises(RuntimeError, match=r"the HTTP clients of litellm \S+ and openai "
                                           r"\S+ .*object is not an httpx or httpx2 "
                                           r"client"):
        providercall.check_before_call()


@pytest.mark.parametrize("module", [httpx, httpx2])
def test_a_client_of_either_httpx_module_is_set_to_follow_no_redirect(module):
    """A client of httpx or httpx2, sync or async, leaves no reason to refuse a call."""
    with module.Client(follow_redirects=True) as client:
        assert providercall._no_redirect(client) is client
        assert client.follow_redirects is False
    async_client = module.AsyncClient(follow_redirects=True)
    assert providercall._no_redirect(async_client) is async_client
    assert async_client.follow_redirects is False
    asyncio.run(async_client.aclose())
    assert providercall._redirect_block_text == ""


@pytest.mark.parametrize("name, value", [
    ("ALL_PROXY", "socks5://svcname:secretpw99@127.0.0.1:1080"),
    ("https_proxy", "http://svcname:pa#secretpw99@127.0.0.1:3128")])
def test_a_proxy_variable_no_client_reads_stops_every_call_before_it_is_made(
        monkeypatch, name, value):
    call_list = []
    monkeypatch.setattr(providercall.litellm, "completion",
                        lambda **kwarg_dict: call_list.append(kwarg_dict))
    monkeypatch.setenv(name, value)
    for _ in range(2):
        with pytest.raises(RuntimeError, match=f"^no call is made to a model: {name} ") \
                as caught:
            providercall.completion({"model": "openai/some-model", "max_tokens": 10,
                                     "messages": [{"role": "user", "content": "x"}]})
        assert "secretpw99" not in str(caught.value) and "svcname" not in str(caught.value)
    assert call_list == []


def test_the_model_information_is_asked_with_the_calls_credentials(monkeypatch):
    lookup_list = []

    def model_info(model, **kwarg_dict):
        lookup_list.append((model, kwarg_dict))
        return {"max_output_tokens": 8192, "key": "m"}

    monkeypatch.setattr(litellm, "get_model_info", model_info)
    monkeypatch.setattr(config, "LLM_API_KEY", "k")
    monkeypatch.setattr(config, "LLM_API_BASE", "http://host:11434")
    assert providercall.table_output_limit("ollama/m") == 8192
    assert broker._table_key("ollama/m") == "m"
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    assert providercall.table_output_limit("ollama/m") == 8192
    assert lookup_list == [
        ("ollama/m", {"api_key": "k", "api_base": "http://host:11434"}),
        ("ollama/m", {"api_key": "k", "api_base": "http://host:11434"}),
        ("ollama/m", {})]


def test_a_process_with_a_broker_asks_for_the_model_information_without_credentials(
        monkeypatch):
    lookup_list = []

    def model_info(model, **kwarg_dict):
        lookup_list.append(kwarg_dict)
        return {}

    monkeypatch.setattr(litellm, "get_model_info", model_info)
    monkeypatch.setattr(config, "LLM_API_KEY", "k")
    monkeypatch.setattr(config, "LLM_API_BASE", "http://host:11434")
    monkeypatch.setattr(config, "LLM_BROKER", "tcp://broker:9000")
    providercall.model_info_of("ollama/m")
    assert lookup_list == [{}]


def test_a_process_that_refuses_model_calls_does_not_look_the_model_up(monkeypatch):
    monkeypatch.setattr(litellm, "get_model_info",
                        lambda model, **kwarg_dict: pytest.fail("the model was looked up"))
    monkeypatch.setattr(config, "LLM_BROKER", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "k")
    monkeypatch.setattr(providercall, "_redirect_block_text", "no call is made to a model")
    assert providercall.model_info_of("ollama/m") == {}
    assert providercall.table_output_limit("ollama/m") == 0


def test_the_credentials_do_not_change_what_the_table_holds_for_a_model_it_holds(
        monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    without = providercall.table_output_limit("anthropic/claude-haiku-4-5")
    monkeypatch.setattr(config, "LLM_API_KEY", "k")
    monkeypatch.setattr(config, "LLM_API_BASE", "http://127.0.0.1:9")
    assert providercall.table_output_limit("anthropic/claude-haiku-4-5") == without > 0


def test_a_model_litellm_signs_in_for_is_not_asked_for_its_information(monkeypatch):
    monkeypatch.setattr(litellm, "get_model_info",
                        lambda model, **kwarg_dict: pytest.fail("the model was looked up"))
    assert providercall.model_info_of("github_copilot/gpt-4o") == {}
    assert providercall.table_output_limit("github_copilot/gpt-4o") == 0


def test_a_huggingface_model_s_config_is_fetched_only_where_the_model_table_is(
        monkeypatch):
    """Where litellm reads the model table it ships with, the model information of a
    huggingface/ model is made as when the fetch of its config.json fails."""
    import litellm.utils

    url_list = []

    def get(url, *arg_tuple, **kwarg_dict):
        url_list.append(url)
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(litellm.module_level_client, "get", get)
    monkeypatch.setenv(providercall.LOCAL_MODEL_TABLE_ENV, "True")
    assert litellm.utils._get_max_position_embeddings("meta-llama/Llama-3.1-8B") is None
    info = litellm.utils._get_model_info_helper("huggingface/meta-llama/Llama-3.1-8B")
    assert info["max_tokens"] is None and info["mode"] == "chat" and url_list == []
    monkeypatch.delenv(providercall.LOCAL_MODEL_TABLE_ENV)
    assert litellm.utils._get_max_position_embeddings("meta-llama/Llama-3.1-8B") is None
    assert url_list == ["https://huggingface.co/meta-llama/Llama-3.1-8B/raw/main/config.json"]


def test_the_max_tokens_of_a_huggingface_model_are_fetched_only_where_the_model_table_is(
        monkeypatch):
    """litellm's own change of max_tokens asks get_max_tokens(), whose inner function
    fetches the config.json; where litellm reads the model table it ships with, the
    answer is the one of a fetch that failed."""
    import litellm.utils
    from litellm.litellm_core_utils import token_counter

    url_list = []

    def get(url, *arg_tuple, **kwarg_dict):
        url_list.append(url)
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(litellm.module_level_client, "get", get)
    monkeypatch.setenv(providercall.LOCAL_MODEL_TABLE_ENV, "True")
    model = "huggingface/meta-llama/Llama-3.1-8B"
    assert litellm.get_max_tokens is litellm.utils.get_max_tokens
    assert litellm.get_max_tokens(model) is None
    assert token_counter.get_modified_max_tokens(
        model=model, base_model=model, messages=[{"role": "user", "content": "hi"}],
        user_max_tokens=20, buffer_perc=None, buffer_num=None) == 20
    assert litellm.get_max_tokens("gpt-4o") == litellm.model_cost["gpt-4o"][
        "max_output_tokens"]
    assert url_list == []
    monkeypatch.delenv(providercall.LOCAL_MODEL_TABLE_ENV)
    assert litellm.get_max_tokens(model) is None
    assert url_list == ["https://huggingface.co/meta-llama/Llama-3.1-8B/raw/main/config.json"]


def test_a_token_count_asks_another_machine_for_no_tokenizer(monkeypatch):
    """A model whose name litellm has a tokenizer to fetch for is counted with the ones
    it ships with."""
    import tokenizers

    def fetch(*arg_tuple, **kwarg_dict):
        raise AssertionError(f"a tokenizer was asked for: {arg_tuple}")

    monkeypatch.setattr(tokenizers.Tokenizer, "from_pretrained", staticmethod(fetch))
    assert litellm.disable_hf_tokenizer_download is True
    for model in ("ollama_chat/llama-3.1", "hosted_vllm/Llama-2-7b", "replicate/meta/llama-3"):
        assert providercall._piece_token(model, ["a reply of a few words"]) > 0
