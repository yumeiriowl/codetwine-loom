"""The broker: the address it listens on, the credentials it adds, and one round trip.

litellm.completion is substituted throughout and no API is called. The round trip runs
a real server over a real Unix socket, so the framing is exercised rather than assumed.
"""

import json
import logging
import os
import shlex
import shutil
import socket
import ssl
import tempfile
import threading
import time
from types import SimpleNamespace

import httpx
import litellm
import pytest

from codetwine_loom import broker, config, llm, providercall, rlm, tokencount, websearch


class _FakeCompletion:
    """A stand-in for litellm.completion recording its kwargs and returning a reply."""

    def __init__(self, content="ok", finish_reason="stop", raises=None):
        self.kwargs = None
        self._content = content
        self._finish_reason = finish_reason
        self._raises = raises

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self._raises is not None:
            raise self._raises
        message = type("Message", (), {"content": self._content})()
        choice = type("Choice", (),
                      {"message": message, "finish_reason": self._finish_reason})()
        return type("Response", (), {"choices": [choice]})()


# What usage_dict_of() returns for a reply that carried no usage
_NO_USAGE = {"input_token": 0, "output_token": 0, "cache_read_token": 0,
             "cache_write_token": 0, "estimate_count": 0, "retry_count": 0}


# The one message make_llm() sends, and the params of a completion carrying it
_MESSAGE_LIST = [{"role": "user", "content": "q"}]
_ASK_DICT = {"model": "m", "max_tokens": 10, "messages": _MESSAGE_LIST}


# The models the requests below ask for, set as the broker's own settings
_MODEL_SETTING_DICT = {"EDIT_MODEL": "m", "PLAN_MODEL": "anthropic/m",
                       "ANSWER_MODEL": "openai/m", "DISPATCH_MODEL": "some/model"}


@pytest.fixture(autouse=True)
def model_setting(monkeypatch):
    """Set the models the requests ask for as this broker's settings."""
    for name, model in _MODEL_SETTING_DICT.items():
        monkeypatch.setattr(config, name, model)


@pytest.fixture
def fake(monkeypatch):
    """Return the stand-in that replaced litellm.completion."""
    completion = _FakeCompletion()
    monkeypatch.setattr(broker.litellm, "completion", completion)
    return completion


# == The token counts of one call ==============================

def test_the_token_counts_are_read_off_litellm_usage():
    from litellm.types.utils import Usage
    usage = Usage(prompt_tokens=10, completion_tokens=3, cache_read_input_tokens=6,
                  cache_creation_input_tokens=2)
    assert broker.usage_dict_of(usage) == {**_NO_USAGE, "input_token": 10,
                                           "output_token": 3, "cache_read_token": 6,
                                           "cache_write_token": 2}


def test_the_marks_completion_leaves_on_the_usage_are_read_too():
    from litellm.types.utils import Usage
    usage = Usage(prompt_tokens=10, completion_tokens=3)
    usage.estimate_count, usage.retry_count = 1, 2
    usage_dict = broker.usage_dict_of(usage)
    assert (usage_dict["estimate_count"], usage_dict["retry_count"]) == (1, 2)


def test_a_reply_without_usage_counts_nothing():
    assert broker.usage_dict_of(None) == _NO_USAGE
    assert broker.usage_dict_of(type("Usage", (), {"prompt_tokens": None})()) == _NO_USAGE


def test_the_result_carries_the_token_counts(monkeypatch):
    class _Completion(_FakeCompletion):
        def __call__(self, **kwargs):
            response = super().__call__(**kwargs)
            response.usage = type("Usage", (), {"prompt_tokens": 7, "completion_tokens": 1})()
            return response

    monkeypatch.setattr(broker.litellm, "completion", _Completion())
    reply = broker.answer({"id": 1, "method": "completion", "params": dict(_ASK_DICT)},
                          broker._Budget())
    assert reply["result"]["usage"] == {**_NO_USAGE, "input_token": 7, "output_token": 1}
    assert broker._choice_of(reply["result"]).usage_dict["input_token"] == 7


@pytest.fixture
def refills(monkeypatch):
    """Return the list one refill appends to, with a budget of 2.

    Yields:
        The list broker.serve()'s refill callable appends to.
    """
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 2)
    filled = []
    yield filled


@pytest.fixture
def serving(monkeypatch, refills):
    """Serve on a Unix socket of its own and point config.LLM_BROKER at it.

    The socket goes in a directory of its own rather than in tmp_path, whose name
    carries the test's and can run past what an address may be.

    The server runs in a daemon thread; there is no way to ask serve() to return, and
    the thread ends with the process.

    Yields:
        The address, as config.LLM_BROKER holds it.
    """
    directory = tempfile.mkdtemp(prefix="ctloom-test-")
    url = f"{broker.UNIX_SCHEME}{directory}/llm.sock"

    def refill_cache(spend):
        spend()
        refills.append(1)
        return {"exit_status": 0, "skip": False}

    threading.Thread(target=broker.serve, args=(url,),
                     kwargs={"refill_cache": refill_cache}, daemon=True).start()
    _wait_for(url)
    monkeypatch.setattr(config, "LLM_BROKER", url)
    monkeypatch.setattr(config, "LLM_BROKER_TIMEOUT_SEC", 10)
    yield url
    shutil.rmtree(directory, ignore_errors=True)


def _wait_for(url, timeout=5.0):
    """Block until the broker at url accepts a connection, or fail the test."""
    family, address = broker.endpoint(url)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.connect(address)
            return
        except OSError:
            time.sleep(0.02)
    pytest.fail(f"the broker never listened on {url}")


# == The address ==============================================

def test_a_unix_address_is_a_path():
    assert broker.endpoint("unix:///run/ctloom/llm.sock") == (socket.AF_UNIX,
                                                           "/run/ctloom/llm.sock")


def test_a_tcp_address_is_a_host_and_a_port():
    assert broker.endpoint("tcp://broker:9000") == (socket.AF_INET, ("broker", 9000))


@pytest.mark.parametrize("url", ["http://broker:9000", "unix://", "tcp://broker",
                                 "tcp://broker:nine", "/run/ctloom/llm.sock", ""])
def test_an_address_of_neither_form_says_how_to_write_one(url):
    with pytest.raises(ValueError, match="LLM_BROKER"):
        broker.endpoint(url)


# == What the broker adds and what it refuses =================

def test_the_broker_adds_the_credentials_the_caller_does_not_have(fake, monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "a-key")
    monkeypatch.setattr(config, "LLM_API_BASE", "http://somewhere.invalid")
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": dict(_ASK_DICT)},
                          broker._Budget())
    assert reply["result"] == {"content": "ok", "finish_reason": "stop", "usage": _NO_USAGE}
    assert fake.kwargs["api_key"] == "a-key"
    assert fake.kwargs["api_base"] == "http://somewhere.invalid"


def test_an_empty_credential_is_not_passed_at_all(fake, monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    broker.answer({"id": 1, "method": "completion", "params": dict(_ASK_DICT)},
                  broker._Budget())
    assert "api_key" not in fake.kwargs and "api_base" not in fake.kwargs


@pytest.mark.parametrize("name", ["api_key", "api_base", "base_url",
                                  "web_search_options", "tools", "extra_headers"])
def test_a_call_that_brought_another_argument_is_refused(fake, name):
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": {"model": "m", "max_tokens": 10, name: "smuggled"}},
                          broker._Budget())
    assert name in reply["error"]["message"]
    assert fake.kwargs is None


def test_the_names_a_call_is_refused_for_are_quoted_and_the_log_line_is_held(fake, caplog):
    """A name holding line breaks and controls is logged as its repr, on a line of
    LOG_EXCERPT_CHAR characters at most."""
    name = "\x1b[2J0:00:01 WARNING x: every check passed\n0:00:02 INFO x: next\x07"
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.broker"):
        reply = broker.answer({"id": 1, "method": "completion", "params": {
            name: 1, "y" * 100000: 2}}, broker._Budget())
    assert reply["error"]["message"].startswith(f"the call carried {name!r}, 'yyy")
    line = caplog.records[-1].getMessage()
    assert line.startswith("the completion call was refused: the call carried '\\x1b[2J")
    assert "\n" not in line and "\x1b" not in line and "\x07" not in line
    assert len(line) <= len("the completion call was refused: ") + config.LOG_EXCERPT_CHAR
    assert fake.kwargs is None


def test_a_call_holding_every_argument_a_completion_takes_is_made(fake):
    param_dict = {"model": "m", "max_tokens": 4096, "num_retries": 1,
                  "messages": [{"role": "user", "content": "q"}],
                  providercall.TURN_REPLY_KEY: False,
                  "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "low"}}
    assert set(param_dict) == set(broker.COMPLETION_PARAM_TUPLE)
    reply = broker.answer({"id": 1, "method": "completion", "params": param_dict},
                          broker._Budget())
    assert "result" in reply
    # The broker makes the call again itself, so litellm is given no retries, and the
    # mark of a turn's reply is taken off before litellm is called
    own_tuple = ("num_retries", providercall.TURN_REPLY_KEY)
    assert {name: fake.kwargs[name] for name in param_dict if name not in own_tuple} == \
        {name: value for name, value in param_dict.items() if name not in own_tuple}
    assert providercall.TURN_REPLY_KEY not in fake.kwargs
    assert fake.kwargs["num_retries"] == 0


def test_a_method_the_broker_does_not_serve_says_which_one_it_does(fake):
    reply = broker.answer({"id": 1, "method": "shell", "params": {}},
                          broker._Budget())
    assert "completion" in reply["error"]["message"]
    assert fake.kwargs is None


def test_the_kind_of_failure_travels_with_it(monkeypatch):
    """make_llm() reads it to decide whether to repeat the call with the next thinking form."""
    refusal = litellm.BadRequestError(message="thinking is not supported", model="m",
                                      llm_provider="anthropic")
    monkeypatch.setattr(broker.litellm, "completion",
                        _FakeCompletion(raises=refusal))
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": dict(_ASK_DICT)}, broker._Budget())
    assert reply["error"]["data"]["error_type"] == broker.BAD_REQUEST
    assert "thinking" in reply["error"]["message"]


def test_a_context_window_refusal_travels_under_its_own_name(monkeypatch):
    refusal = litellm.ContextWindowExceededError(message="prompt is too long", model="m",
                                                 llm_provider="anthropic")
    monkeypatch.setattr(broker.litellm, "completion",
                        _FakeCompletion(raises=refusal))
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": dict(_ASK_DICT)}, broker._Budget())
    assert reply["error"]["data"]["error_type"] == broker.CONTEXT_WINDOW


def test_a_refusal_raised_as_a_subclass_travels_under_the_one_name(monkeypatch):
    refusal = litellm.UnsupportedParamsError(
        message="openai does not support parameters: ['thinking']", model="m",
        llm_provider="openai")
    monkeypatch.setattr(broker.litellm, "completion",
                        _FakeCompletion(raises=refusal))
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": dict(_ASK_DICT)}, broker._Budget())
    assert reply["error"]["data"]["error_type"] == broker.BAD_REQUEST


def test_a_failure_of_any_other_kind_is_still_one_reply(monkeypatch):
    """A provider that raises must not take the broker down with it."""
    monkeypatch.setattr(broker.litellm, "completion",
                        _FakeCompletion(raises=RuntimeError("the socket went away")))
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": dict(_ASK_DICT)}, broker._Budget())
    assert reply["error"]["data"]["error_type"] == "RuntimeError"
    assert "result" not in reply


def test_a_reply_nobody_is_waiting_for_does_not_take_the_thread_down(fake):
    """A caller gives up at LLM_BROKER_TIMEOUT_SEC and closes the connection, so the
    reply to a slow call has nowhere to go. It ends the connection, not the thread."""
    ours, theirs = socket.socketpair()
    with theirs:
        theirs.sendall(json.dumps({"id": 1, "method": "completion",
                                   "params": dict(_ASK_DICT)}).encode() + b"\n")
    ours.shutdown(socket.SHUT_RDWR)     # the caller is gone before the reply is written
    broker._serve_one(ours, broker._Budget())


def test_a_reply_that_can_be_written_still_is(fake):
    ours, theirs = socket.socketpair()
    with theirs:
        theirs.sendall(json.dumps({"id": 1, "method": "completion",
                                   "params": dict(_ASK_DICT)}).encode() + b"\n")
        theirs.shutdown(socket.SHUT_WR)
        broker._serve_one(ours, broker._Budget())
        reply = json.loads(theirs.makefile("rb").readline())
    assert reply["result"] == {"content": "ok", "finish_reason": "stop", "usage": _NO_USAGE}


def _served(request_dict: dict, token_out: str) -> dict:
    """Serve one request on a connection of its own and return the reply."""
    ours, theirs = socket.socketpair()
    with theirs:
        theirs.sendall(json.dumps(request_dict).encode() + b"\n")
        theirs.shutdown(socket.SHUT_WR)
        broker._serve_one(ours, broker._Budget(), None, token_out)
        return json.loads(theirs.makefile("rb").readline())


def test_the_counts_of_every_call_are_written_where_the_broker_was_told(monkeypatch,
                                                                        tmp_path):
    class _Completion(_FakeCompletion):
        def __call__(self, **kwargs):
            response = super().__call__(**kwargs)
            response.usage = type("Usage", (), {"prompt_tokens": 7, "completion_tokens": 1})()
            return response

    monkeypatch.setattr(broker.litellm, "completion", _Completion())
    token_path = tmp_path / "token.json"
    request_dict = {"id": 1, "method": "completion",
                    "params": dict(_ASK_DICT)}
    _served(request_dict, str(token_path))
    assert json.loads(token_path.read_text())["total"] == 8
    _served(request_dict, str(token_path))
    assert json.loads(token_path.read_text()) == {
        "call": 2, "input_token": 14, "output_token": 2, "cache_read_token": 0,
        "cache_write_token": 0, "total": 16}


def test_a_completion_is_refused_once_the_tokens_it_was_given_are_used(monkeypatch):
    class _Completion(_FakeCompletion):
        def __call__(self, **kwargs):
            response = super().__call__(**kwargs)
            response.usage = type("Usage", (), {"prompt_tokens": 7, "completion_tokens": 1})()
            return response

    call_list = []

    class _Counting(_Completion):
        def __call__(self, **kwargs):
            call_list.append(kwargs)
            return super().__call__(**kwargs)

    monkeypatch.setattr(broker.litellm, "completion", _Counting())
    budget = broker._Budget(None, token_left=10)
    request_dict = {"id": 1, "method": "completion",
                    "params": dict(_ASK_DICT)}
    assert "result" in broker.answer(request_dict, budget)
    assert "result" in broker.answer(request_dict, budget)
    reply = broker.answer(request_dict, budget)
    assert ("used 16 token(s), which reaches the 10 its containers may use between "
            "them (MAX_TOKEN_PER_RUN)") in reply["error"]["message"]
    assert reply["error"]["data"]["error_type"] == "RuntimeError"
    assert len(call_list) == 2
    # A search is refused as a completion is
    reply = broker.answer({"id": 2, "method": "search", "params": {"question": "q"}},
                          budget)
    assert "MAX_TOKEN_PER_RUN" in reply["error"]["message"]
    # A broker given no limit goes on
    assert "result" in broker.answer(request_dict, broker._Budget())


def test_what_the_brokers_beside_it_used_is_counted_against_the_limit(fake, tmp_path,
                                                                     monkeypatch):
    class _Counting(_FakeCompletion):
        def __call__(self, **kwargs):
            response = super().__call__(**kwargs)
            response.usage = type("Usage", (), {"prompt_tokens": 7, "completion_tokens": 1})()
            return response

    monkeypatch.setattr(broker.litellm, "completion", _Counting())
    own_path = str(tmp_path / "own.json")
    budget = broker._Budget(None, token_left=20, token_out=own_path)
    request_dict = {"id": 1, "method": "completion",
                    "params": dict(_ASK_DICT)}
    assert "result" in broker.answer(request_dict, budget)
    # This broker's own file is not counted on top of what it holds in memory
    tokencount.write(own_path)
    assert "result" in broker.answer(request_dict, budget)
    (tmp_path / "other.json").write_text(json.dumps({"input_token": 3, "output_token": 1}))
    reply = broker.answer(request_dict, budget)
    assert "used 20 token(s), which reaches the 20" in reply["error"]["message"]


def test_a_request_that_makes_no_model_call_writes_no_counts(fake, tmp_path):
    _served({"id": 1, "method": "refill"}, str(tmp_path / "token.json"))
    _served({"id": 1, "method": "fetch", "params": {"url": 7}}, str(tmp_path / "token.json"))
    assert not (tmp_path / "token.json").exists()


def test_a_broker_told_no_file_writes_none(fake, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _served({"id": 1, "method": "completion",
             "params": dict(_ASK_DICT)}, "")
    assert list(tmp_path.iterdir()) == []


# == What the container may send ==============================

def test_a_request_over_the_limit_is_answered_rather_than_read(monkeypatch, fake):
    """What arrives is the container's, which is the side of the boundary this process
    does not trust. Reading a line with no end to it is what a limit is for."""
    monkeypatch.setattr(config, "LLM_BROKER_MAX_REQUEST_BYTE", 200)
    ours, theirs = socket.socketpair()
    with theirs:
        theirs.sendall(json.dumps({"id": 1, "method": "completion",
                                   "params": {"model": "m", "max_tokens": 10, "pad": "x" * 500}}
                                  ).encode() + b"\n")
        broker._serve_one(ours, broker._Budget())
        reply = json.loads(theirs.makefile("rb").readline())
    assert reply["error"]["code"] == broker._CALL_FAILURE
    assert "LLM_BROKER_MAX_REQUEST_BYTE" in reply["error"]["message"]
    assert fake.kwargs is None      # the provider was never reached


def test_a_request_within_the_limit_is_answered_as_it_always_was(monkeypatch, fake):
    monkeypatch.setattr(config, "LLM_BROKER_MAX_REQUEST_BYTE", 4096)
    ours, theirs = socket.socketpair()
    with theirs:
        theirs.sendall(json.dumps({"id": 1, "method": "completion",
                                   "params": dict(_ASK_DICT)}).encode() + b"\n")
        theirs.shutdown(socket.SHUT_WR)
        broker._serve_one(ours, broker._Budget())
        reply = json.loads(theirs.makefile("rb").readline())
    assert reply["result"] == {"content": "ok", "finish_reason": "stop", "usage": _NO_USAGE}


def test_a_request_that_is_not_an_object_is_answered_and_the_connection_kept(fake):
    ours, theirs = socket.socketpair()
    with theirs:
        theirs.sendall(b'["not", "an", "object"]\n' + json.dumps(
            {"id": 2, "method": "completion",
             "params": dict(_ASK_DICT)}).encode() + b"\n")
        theirs.shutdown(socket.SHUT_WR)
        broker._serve_one(ours, broker._Budget())
        reader = theirs.makefile("rb")
        first, second = json.loads(reader.readline()), json.loads(reader.readline())
    assert first["error"]["code"] == broker._PARSE_ERROR
    assert second["result"]["content"] == "ok"


@pytest.mark.parametrize("request_dict", [
    {"id": 1, "method": "completion", "params": ["model"]},
    {"id": 1, "method": "search", "params": {"question": 5}},
    {"id": 1, "method": "search", "params": ["question"]},
    {"id": 1, "method": "fetch", "params": {}},
    {"id": 1, "method": "fetch", "params": {"url": ["https://a.invalid/"]}}])
def test_malformed_params_are_refused_before_anything_is_spent(monkeypatch, fake,
                                                               request_dict):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    monkeypatch.setattr(websearch, "search", lambda question: list(_RESULTS))
    monkeypatch.setattr(websearch, "fetch", lambda url: "page")
    budget = broker._Budget()
    reply = broker.answer(request_dict, budget)
    assert reply["error"]["code"] == broker._CALL_FAILURE
    assert fake.kwargs is None
    for method in (broker.COMPLETION, broker.SEARCH, broker.FETCH):
        assert budget.remain_count(method) == broker._Budget().remain_count(method)


def test_a_caller_over_the_connection_limit_is_refused_rather_than_queued(monkeypatch):
    """A thread that may never come is not something to wait behind. The refusal is one
    reply, so the caller reads an answer instead of a closed connection."""
    monkeypatch.setattr(config, "LLM_BROKER_MAX_CONNECTION", 1)
    ours, theirs = socket.socketpair()
    with theirs:
        broker._refuse(ours, broker._TOO_MANY.format(limit=1))
        reply = json.loads(theirs.makefile("rb").readline())
    assert reply["error"]["code"] == broker._CALL_FAILURE
    assert "LLM_BROKER_MAX_CONNECTION" in reply["error"]["message"]


def test_the_slot_is_given_back_when_the_connection_ends():
    """Without it the broker answers LLM_BROKER_MAX_CONNECTION callers and no more."""
    answering = threading.Semaphore(1)
    assert answering.acquire(blocking=False)
    ours, theirs = socket.socketpair()
    with theirs:
        theirs.shutdown(socket.SHUT_WR)
        broker._serve_one(ours, broker._Budget(), answering)
    assert answering.acquire(blocking=False)


# == One round trip over a real socket ========================

def test_a_call_and_its_reply_cross_the_socket(serving, fake):
    choice = broker.complete({"model": "m", "max_tokens": 10,
                              "messages": [{"role": "user", "content": "PROMPT"}]})
    assert choice.message.content == "ok"
    assert choice.finish_reason == "stop"
    assert fake.kwargs["messages"] == [{"role": "user", "content": "PROMPT"}]


def test_half_of_a_surrogate_pair_crosses_to_the_broker_and_back(serving, monkeypatch):
    """The request reaches the broker with it, the provider is sent U+FFFD in its
    place, and the reply comes back as the provider gave it."""
    fake = _FakeCompletion(content="reply \ud83d cut")
    monkeypatch.setattr(broker.litellm, "completion", fake)
    prompt = "prompt \ude00 cut and a whole \U0001f600"
    choice = broker.complete({**_ASK_DICT, "messages": [{"role": "user",
                                                         "content": prompt}]})
    assert fake.kwargs["messages"][0]["content"] == \
        "prompt \ufffd cut and a whole \U0001f600"
    assert choice.message.content == "reply \ud83d cut"


def test_a_request_whose_answer_raises_is_answered_with_an_error_and_the_next_one_too(
        serving, fake, monkeypatch, caplog):
    answer = broker.answer

    def raising(request_dict, budget):
        if request_dict["id"] == 1:
            raise KeyError("no such key")
        return answer(request_dict, budget)

    monkeypatch.setattr(broker, "answer", raising)
    payload = b"".join(json.dumps({"jsonrpc": "2.0", "id": ident, "method": "completion",
                                   "params": dict(_ASK_DICT)}).encode() + b"\n"
                       for ident in (1, 2))
    family, address = broker.endpoint(serving)
    with socket.socket(family, socket.SOCK_STREAM) as conn:
        conn.settimeout(5)
        conn.connect(address)
        conn.sendall(payload)
        with conn.makefile("rb") as reader:
            first, second = json.loads(reader.readline()), json.loads(reader.readline())
    assert first["id"] == 1 and first["error"]["data"]["error_type"] == "KeyError"
    assert "the broker could not answer the request: KeyError" in first["error"]["message"]
    assert second["id"] == 2 and second["result"]["content"] == "ok"
    assert "a request could not be answered: KeyError" in caplog.text


def test_a_reply_that_cannot_be_written_as_json_is_answered_with_an_error(serving,
                                                                          monkeypatch):
    monkeypatch.setattr(broker, "answer", lambda request_dict, budget: {
        "jsonrpc": "2.0", "id": request_dict["id"], "result": {"content": {1, 2}}})
    reply = _raw_line(serving, json.dumps({"jsonrpc": "2.0", "id": 5, "method": "completion",
                                           "params": dict(_ASK_DICT)}).encode() + b"\n")
    assert reply["id"] == 5 and reply["error"]["data"]["error_type"] == "TypeError"


def test_a_prompt_of_many_lines_crosses_in_one_piece(serving, fake):
    """One message is one line, so the newlines inside it must not end it early."""
    prompt = "\n".join(f"line {n} with a \\ and a \" in it" for n in range(20000))
    choice = broker.complete({"model": "m", "max_tokens": 10, "messages":
                              [{"role": "user", "content": prompt}]})
    assert choice.message.content == "ok"
    assert fake.kwargs["messages"][0]["content"] == prompt


def test_a_request_is_sent_as_utf8_rather_than_escaped():
    """The byte limit then counts a character outside ASCII as its UTF-8 bytes."""
    sent = []
    conn = type("Conn", (), {"sendall": lambda self, data: sent.append(data)})()
    broker._send(conn, {"prompt": "日本語\n"})
    assert sent == ['{"prompt": "日本語\\n"}\n'.encode("utf-8")]
    assert b"\\u" not in sent[0]


def test_the_provider_side_search_tool_does_not_cross_with_a_completion(serving, fake):
    """A search is the search method's to make, which WEB_SEARCH_MAX_CALL counts."""
    tool = {"type": "web_search_20250305", "name": "web_search"}
    with pytest.raises(RuntimeError, match="tools"):
        broker.complete({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "q"}],
                         "tools": [tool]})
    assert fake.kwargs is None


def test_a_reply_with_no_content_block_comes_back_as_none(serving, monkeypatch):
    monkeypatch.setattr(broker.litellm, "completion",
                        _FakeCompletion(content=None, finish_reason="stop"))
    assert broker.complete(dict(_ASK_DICT)).message.content is None


def test_a_refusal_of_the_parameters_is_raised_as_the_provider_raised_it(serving,
                                                                        monkeypatch):
    refusal = litellm.BadRequestError(message="effort is not supported", model="m",
                                      llm_provider="anthropic")
    monkeypatch.setattr(broker.litellm, "completion",
                        _FakeCompletion(raises=refusal))
    with pytest.raises(litellm.BadRequestError, match="effort"):
        broker.complete({**_ASK_DICT, "model": "anthropic/m"})


def test_a_refusal_raised_as_a_subclass_is_raised_as_one_the_call_site_retries(
        serving, monkeypatch):
    refusal = litellm.UnsupportedParamsError(
        message="openai does not support parameters: ['thinking']", model="m",
        llm_provider="openai")
    monkeypatch.setattr(broker.litellm, "completion",
                        _FakeCompletion(raises=refusal))
    with pytest.raises(litellm.BadRequestError, match="thinking"):
        broker.complete({**_ASK_DICT, "model": "openai/m"})


def test_a_context_window_refusal_is_raised_as_the_subclass(serving, monkeypatch):
    refusal = litellm.ContextWindowExceededError(message="prompt is too long", model="m",
                                                 llm_provider="anthropic")
    monkeypatch.setattr(broker.litellm, "completion",
                        _FakeCompletion(raises=refusal))
    with pytest.raises(litellm.ContextWindowExceededError, match="too long"):
        broker.complete({**_ASK_DICT, "model": "anthropic/m"})


def test_every_other_failure_is_raised_as_one_the_call_site_does_not_retry(serving,
                                                                          monkeypatch):
    monkeypatch.setattr(broker.litellm, "completion",
                        _FakeCompletion(raises=RuntimeError("no route")))
    with pytest.raises(RuntimeError, match="no route"):
        broker.complete(dict(_ASK_DICT))


def test_the_broker_gives_each_attempt_the_call_timeout(fake, monkeypatch):
    monkeypatch.setattr(config, "LLM_CALL_TIMEOUT_SEC", 7)
    broker._call_the_provider({"model": "m", "messages": []})
    assert fake.kwargs["timeout"] == 7


def test_the_broker_retries_as_often_as_it_says_whatever_the_caller_asks(monkeypatch):
    seen = {}
    monkeypatch.setattr(broker.providercall, "completion",
                        lambda param_dict: seen.update(param_dict) or _FakeCompletion()())
    monkeypatch.setattr(config, "LLM_MAX_RETRY", 2)
    broker._call_the_provider({"model": "m", "messages": [], "num_retries": 1000})
    assert seen["num_retries"] == 2


def test_the_broker_is_waited_for_over_every_attempt(monkeypatch):
    import importlib
    monkeypatch.setenv("LLM_CALL_TIMEOUT_SEC", "30")
    monkeypatch.setenv("LLM_MAX_RETRY", "2")
    importlib.reload(config)
    try:
        # 3 attempts of 30 s and 2 waits of at most 30 s
        assert config.LLM_BROKER_TIMEOUT_SEC == 150
    finally:
        monkeypatch.delenv("LLM_CALL_TIMEOUT_SEC")
        monkeypatch.delenv("LLM_MAX_RETRY")
        importlib.reload(config)


def test_a_call_that_takes_too_long_is_not_told_to_start_a_broker(monkeypatch, tmp_path):
    """The broker is there; the call ran past every attempt's time."""
    path = f"{tmp_path}/b.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
        server.listen(1)
        monkeypatch.setattr(config, "LLM_BROKER", f"{broker.UNIX_SCHEME}{path}")
        monkeypatch.setattr(config, "LLM_BROKER_TIMEOUT_SEC", 1)
        with pytest.raises(RuntimeError, match="no reply within 1 s") as caught:
            broker.complete(dict(_ASK_DICT))
    assert "broker --endpoint" not in str(caught.value)


@pytest.mark.parametrize("method, name", [(broker.search, "WEB_SEARCH_TIMEOUT_SEC"),
                                          (broker.fetch, "WEB_FETCH_TIMEOUT_SEC")])
def test_a_search_or_a_read_is_waited_for_its_turn_and_its_request_too(monkeypatch,
                                                                     tmp_path, method,
                                                                     name):
    path = f"{tmp_path}/b.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
        server.listen(1)
        monkeypatch.setattr(config, "LLM_BROKER", f"{broker.UNIX_SCHEME}{path}")
        monkeypatch.setattr(config, "LLM_BROKER_TIMEOUT_SEC", 0.25)
        monkeypatch.setattr(config, name, 0.25)
        start = time.monotonic()
        with pytest.raises(RuntimeError, match=f"within 0.75 s.*2 x {name}"):
            method("q")
    assert time.monotonic() - start >= 0.7


def test_a_broker_that_is_not_there_says_how_to_start_one(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLM_BROKER",
                        f"{broker.UNIX_SCHEME}{tmp_path}/nothing.sock")
    monkeypatch.setattr(config, "LLM_BROKER_TIMEOUT_SEC", 1)
    with pytest.raises(RuntimeError, match="broker --endpoint"):
        broker.complete(dict(_ASK_DICT))


def test_a_request_that_is_not_json_is_answered_rather_than_dropped(serving):
    family, address = broker.endpoint(serving)
    with socket.socket(family, socket.SOCK_STREAM) as conn:
        conn.settimeout(5)
        conn.connect(address)
        conn.sendall(b"{not json\n")
        with conn.makefile("rb") as reader:
            reply = json.loads(reader.readline())
    assert "not JSON" in reply["error"]["message"]


# == Filling the package cache again ==========================

def _budget(exit_status=0, calls=None, skip=False):
    """Return a _Budget that records its fillings and reports exit_status.

    With skip it stands for a cache that already holds what is asked for, which starts
    no filling and so spends nothing.
    """
    def refill_cache(spend):
        if skip:
            return {"exit_status": 0, "skip": True}
        spend()
        if calls is not None:
            calls.append(1)
        return {"exit_status": exit_status, "skip": False}
    return broker._Budget(refill_cache)


def test_a_refill_runs_what_the_broker_was_given(monkeypatch):
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 2)
    calls = []
    reply = broker.answer({"id": 1, "method": "refill", "params": {}},
                          _budget(calls=calls))
    assert reply["result"] == {"refills_left": 1, "skip": False}
    assert len(calls) == 1


def test_a_refill_budget_of_0_starts_no_filling(monkeypatch):
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 0)
    calls = []
    reply = broker.answer({"id": 1, "method": "refill"}, _budget(calls=calls))
    assert "WARM_MAX_REFILL=0" in reply["error"]["message"]
    assert calls == []


def test_a_refill_limit_of_0_runs_nothing_of_what_fills_the_cache(monkeypatch):
    """The request is refused before anything of the filling starts."""
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 0)
    run_list = []
    budget = broker._Budget(lambda spend: run_list.append(1) or {"exit_status": 0,
                                                                 "skip": True})
    reply = broker.answer({"id": 1, "method": "refill"}, budget)
    assert "WARM_MAX_REFILL=0" in reply["error"]["message"] and run_list == []


@pytest.mark.parametrize("is_held", [True, False])
def test_with_every_refill_spent_a_cache_that_holds_it_is_still_answered(monkeypatch,
                                                                         is_held):
    """What fills the cache is asked whether the cache holds it: one that does is
    answered as held and not counted, one that does not is refused."""
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 2)
    run_list = []

    def refill_cache(spend):
        run_list.append(1)
        if is_held and len(run_list) > 2:
            return {"exit_status": 0, "skip": True}
        spend()
        return {"exit_status": 0, "skip": False}

    budget = broker._Budget(refill_cache)
    for ident in range(2):
        assert "result" in broker.answer({"id": ident, "method": "refill"}, budget)
    reply = broker.answer({"id": 9, "method": "refill"}, budget)
    if is_held:
        assert reply["result"] == {"refills_left": 0, "skip": True}
    else:
        assert "WARM_MAX_REFILL=2" in reply["error"]["message"]
    assert len(run_list) == 3


def test_a_cache_that_already_holds_it_is_not_counted(monkeypatch):
    """Only a request that starts a filling spends one, however often it is sent."""
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 1)
    held = _budget(skip=True)
    for ident in range(3):
        reply = broker.answer({"id": ident, "method": "refill"}, held)
        assert reply["result"] == {"refills_left": 1, "skip": True}


def test_a_filling_that_failed_is_counted(monkeypatch):
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 1)
    budget = _budget(exit_status=2)
    broker.answer({"id": 1, "method": "refill"}, budget)
    second = broker.answer({"id": 2, "method": "refill"}, budget)
    assert "WARM_MAX_REFILL" in second["error"]["message"]


def test_the_budget_is_counted_where_the_container_cannot_reach_it(monkeypatch):
    """One run gets WARM_MAX_REFILL of them however many times it asks."""
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 1)
    calls = []
    budget = _budget(calls=calls)
    assert "result" in broker.answer({"id": 1, "method": "refill"}, budget)
    second = broker.answer({"id": 2, "method": "refill"}, budget)
    assert "WARM_MAX_REFILL" in second["error"]["message"]
    assert len(calls) == 1


def test_a_broker_with_no_repository_fills_no_cache():
    """A broker started without one holds a budget with no cache to fill."""
    reply = broker.answer({"id": 1, "method": "refill"}, broker._Budget())
    assert "fills no cache" in reply["error"]["message"]


def test_a_refill_that_did_not_finish_cleanly_is_an_error(monkeypatch):
    monkeypatch.setattr(config, "WARM_MAX_REFILL", 1)
    reply = broker.answer({"id": 1, "method": "refill"}, _budget(exit_status=2))
    assert "exit status 2" in reply["error"]["message"]


def test_a_socket_path_written_from_the_start_directory_is_reached_from_another(
        serving, refills, monkeypatch, tmp_path):
    """The address is read where the command starts and used after it has moved."""
    monkeypatch.chdir(os.path.dirname(serving[len(broker.UNIX_SCHEME):]))
    monkeypatch.setattr(config, "LLM_BROKER", config.absolute_broker_url("unix://llm.sock"))
    monkeypatch.chdir(tmp_path)
    assert broker.refill()["ok"] is True


@pytest.mark.parametrize("url", ["tcp://127.0.0.1:99999", "tcp://127.0.0.1:0",
                                 "tcp://[::1]:8080"])
def test_an_address_no_socket_takes_is_refused_before_one_is_made(monkeypatch, url):
    monkeypatch.setattr(broker.socket, "socket",
                        lambda *args: pytest.fail("no socket is made"))
    with pytest.raises(ValueError, match="^LLM_BROKER="):
        broker.serve(url)
    monkeypatch.setattr(config, "LLM_BROKER", url)
    with pytest.raises(ValueError, match="^LLM_BROKER="):
        broker.search("what changed")


def test_the_refill_request_carries_nothing_the_caller_chose(serving, refills):
    """What is fetched follows from the repository, not from anything sent here."""
    assert broker.refill() == {"ok": True, "refills_left": 1, "skip": False}
    assert len(refills) == 1


def test_a_refusal_to_refill_never_raises(monkeypatch, tmp_path):
    """run_test carries on with the cache as it is, so this returns rather than raises."""
    monkeypatch.setattr(config, "LLM_BROKER",
                        f"{broker.UNIX_SCHEME}{tmp_path}/nothing.sock")
    monkeypatch.setattr(config, "LLM_BROKER_TIMEOUT_SEC", 1)
    result = broker.refill()
    assert result["ok"] is False and "broker" in result["error"]


# == Searching the web ========================================

_RESULTS = [{"title": "T", "url": "https://example.invalid/", "content": "C"}]


def test_a_search_goes_out_from_the_broker(serving, monkeypatch):
    """The container has no network, so the question is searched for here."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    asked = []
    monkeypatch.setattr(websearch, "search",
                        lambda q: asked.append(q) or list(_RESULTS))
    assert broker.search("what changed") == _RESULTS
    assert asked == ["what changed"]


def test_the_broker_checks_the_question_length_itself(monkeypatch):
    """The tool checks it too, but a caller that skipped the tool must not get past."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    monkeypatch.setattr(config, "WEB_SEARCH_QUESTION_MAX_CHAR", 10)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/")
    monkeypatch.setattr(websearch, "_open",
                        lambda *a, **k: pytest.fail("it left the machine"))
    reply = broker.answer({"id": 1, "method": "search",
                           "params": {"question": "x" * 11}}, broker._Budget())
    assert "at most 10" in reply["error"]["message"]


def test_a_broker_with_no_endpoint_says_which_setting_to_set(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "")
    reply = broker.answer({"id": 1, "method": "search",
                           "params": {"question": "q"}}, broker._Budget())
    assert "WEB_SEARCH_PROVIDER" in reply["error"]["message"]


def test_a_search_that_failed_is_still_one_reply(monkeypatch):
    """A failing endpoint must not take the broker down with it."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)

    def raising(question):
        raise RuntimeError("the search request failed")
    monkeypatch.setattr(websearch, "search", raising)
    reply = broker.answer({"id": 1, "method": "search",
                           "params": {"question": "q"}}, broker._Budget())
    assert reply["error"]["data"]["error_type"] == "RuntimeError"
    assert "result" not in reply


def test_an_error_keeps_the_plain_values_of_a_preset_and_hides_its_key(monkeypatch):
    preset = config.WEB_SEARCH_PRESET_DICT["brave"]
    monkeypatch.setattr(config, "WEB_SEARCH_API_KEY", "BSA-secret")
    monkeypatch.setattr(config, "WEB_SEARCH_PARAM", preset["params"])
    monkeypatch.setattr(config, "WEB_SEARCH_HEADER",
                        {name: value.format(key="BSA-secret")
                         for name, value in preset["headers"].items()})
    message = broker._error(1, broker._CALL_FAILURE,
                            '"stream": false, Accept: application/json, '
                            'token BSA-secret')["error"]["message"]
    assert message == ('"stream": false, Accept: application/json, '
                       "token <the broker's own>")


def test_an_error_hides_what_the_providers_own_variables_hold(monkeypatch):
    monkeypatch.setattr(llm, "provider_variable_tuple",
                        lambda: ("ONE_API_KEY", "ONE_API_BASE"))
    monkeypatch.setenv("ONE_API_KEY", "k-1-of-one")
    monkeypatch.setenv("ONE_API_BASE", "http://one")
    assert broker.hide_own_value("401 for k-1-of-one at http://one/v1") == \
        "401 for <the broker's own> at <the broker's own>/v1"


def test_the_text_of_a_reply_comes_back_as_the_model_wrote_it(monkeypatch):
    """A key that is also a word of the reply leaves the reply as it is."""
    monkeypatch.setattr(config, "LLM_API_KEY", "password")
    monkeypatch.setattr(config, "LLM_API_BASE", "http://one-of-them")
    monkeypatch.setattr(llm, "provider_variable_tuple", lambda: ("ONE_API_KEY",))
    monkeypatch.setenv("ONE_API_KEY", "required")
    text = ("replace_in_file('auth.py', 'password = None', 'password = read_password()')"
            "  # required by http://one-of-them")
    monkeypatch.setattr(providercall, "completion", lambda param_dict: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                 finish_reason="stop")], usage=None))
    reply_dict = broker.answer({"jsonrpc": "2.0", "id": 1, "method": "completion",
                                "params": {"model": config.EDIT_MODEL, "max_tokens": 10,
                                           "messages": [{"role": "user",
                                                         "content": "hi"}]}},
                               broker._Budget())
    assert reply_dict["result"]["content"] == text


def test_a_failure_that_quotes_a_key_that_is_a_word_comes_back_without_it(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "password")
    monkeypatch.setattr(llm, "provider_variable_tuple", lambda: ())

    def refuse(param_dict):
        raise RuntimeError(f"401: the key {param_dict['api_key']} was refused")

    monkeypatch.setattr(providercall, "completion", refuse)
    reply_dict = broker.answer({"jsonrpc": "2.0", "id": 1, "method": "completion",
                                "params": {"model": config.EDIT_MODEL, "max_tokens": 10,
                                           "messages": [{"role": "user",
                                                         "content": "hi"}]}},
                               broker._Budget())
    assert reply_dict["error"]["message"] == "401: the key <the broker's own> was refused"


@pytest.mark.parametrize("text, want", [
    ("Cannot connect to host llm.corp.example:8443 ssl:default",
     "Cannot connect to host <the broker's own> ssl:default"),
    ("certificate is not valid for 'llm.corp.example'",
     "certificate is not valid for '<the broker's own>'"),
    ("for url https://me:pw-of-ten!@llm.corp.example:8443/v1/messages",
     "for url <the broker's own>/messages"),
    ("for url https://me:pw-of-ten!@LLM.Corp.Example:8443/x",
     "for url https://<the broker's own>@LLM.Corp.Example:8443/x"),
    ("the password pw-of-ten! was sent to ollama-host:11434 and to OLLAMA-HOST",
     "the password <the broker's own> was sent to <the broker's own> and to OLLAMA-HOST"),
    ("the profile dev-profile-1 of sk-0123456789 at host a.b", 
     "the profile <the broker's own> of <the broker's own> at host a.b"),
])
def test_the_parts_a_url_names_its_endpoint_by_are_taken_out_too(monkeypatch, text, want):
    """The host, the host with its port and the user with the password of a value
    that is a URL are taken out wherever they stand, as the whole value is."""
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-0123456789")
    monkeypatch.setattr(config, "LLM_API_BASE", "https://me:pw-of-ten!@llm.corp.example:8443/v1")
    monkeypatch.setattr(llm, "provider_variable_tuple",
                        lambda: ("OLLAMA_API_BASE", "ONE_PROFILE", "SHORT_API_BASE"))
    monkeypatch.setenv("OLLAMA_API_BASE", "http://ollama-host:11434")
    monkeypatch.setenv("ONE_PROFILE", "dev-profile-1")
    monkeypatch.setenv("SHORT_API_BASE", "http://a.b/v1/long-path")
    assert broker.hide_own_value(text) == want


@pytest.mark.parametrize("value, want", [
    ("https://me:pw@llm.corp.example:8443/v1",
     ["me:pw@llm.corp.example:8443", "llm.corp.example:8443", "llm.corp.example", "me:pw",
      "pw", "me"]),
    ("http://Host.Example", ["Host.Example", "host.example"]),
    ("http://[::1]:8080/v1", ["[::1]:8080", "[::1]", "::1"]),
    ("http://svc:Secr3t#Pass99@proxy.corp.example:3128",
     ["svc:Secr3t", "svc", "svc:Secr3t#Pass99", "Secr3t", "Pass99",
      "proxy.corp.example:3128", "proxy.corp.example"]),
    ("socks://svc-proxy-user:secretpw99@[::1/",
     ["svc-proxy-user:secretpw99", "svc-proxy-user", "secretpw99", "[::1", "::1"]),
    # A URL urlsplit() does not read: its host and port are cut from it as they stand
    ("http://svc:secr[et]pw99@llm-endpoint.corp.example:8080/v1",
     ["svc:secr[et]pw99", "svc", "secr[et]pw99", "llm-endpoint.corp.example:8080",
      "llm-endpoint.corp.example"]),
    ("http://u[ser:pw@llm-endpoint.corp.example/v1",
     ["u[ser:pw", "u[ser", "pw", "llm-endpoint.corp.example"]),
    ("http://[::1", ["[::1", "::1"]),
    ("sk-0123456789", []), ("host:8080", []), ("http://", []), ("/a/path", []), ("", [])])
def test_only_a_url_with_a_host_names_an_endpoint(value, want):
    assert broker._endpoint_part_list(value) == want


def test_the_endpoint_of_a_url_urlsplit_cannot_read_is_taken_out(monkeypatch):
    monkeypatch.setattr(llm, "provider_variable_tuple", lambda: ())
    monkeypatch.setattr(config, "LLM_API_BASE",
                        "http://svc:secr[et]pw99@llm-endpoint.corp.example:8080/v1")
    text = ("litellm.APIConnectionError: 'llm-endpoint.corp.example' does not appear to be "
            "an IPv4 or IPv6 address (llm-endpoint.corp.example:8080)")
    assert broker.hide_own_value(text) == (
        "litellm.APIConnectionError: '<the broker's own>' does not appear to be an IPv4 or "
        "IPv6 address (<the broker's own>)")


def _raise_from(error, cause):
    """Raise an error whose cause is another, as a library wraps what it caught."""
    try:
        raise cause
    except type(cause) as e:
        raise error from e


# What a call no response came back to says of the route with none of its variables set
_NO_ROUTE = ("none of HTTPS_PROXY, HTTP_PROXY, ALL_PROXY, NO_PROXY, SSL_CERT_FILE set "
             "where the call was made")


@pytest.mark.parametrize("cause, name", [
    (ConnectionRefusedError(111, "Connect call failed ('10.1.2.3', 8443)"),
     "ConnectionRefusedError"),
    (ConnectionResetError(104, "Connection reset by peer"), "ConnectionResetError"),
    (OSError(113, "Connect call failed ('10.1.2.3', 8443)"), "OSError EHOSTUNREACH"),
    (TimeoutError(110, "Connect call failed ('10.1.2.3', 8443)"), "TimeoutError"),
    (socket.gaierror(-3, "Temporary failure in name resolution"), "gaierror"),
    (ssl.SSLCertVerificationError(1, "certificate is not valid for '10.1.2.3'"),
     "SSLCertVerificationError"),
    (FileNotFoundError(2, "No such file", "/home/me/ca.pem"), "FileNotFoundError"),
    (httpx.ProxyError("403 Forbidden"), "ProxyError"),
    (httpx.RequestError("403, message='Forbidden', url='http://me:pw@10.1.2.3:3128'"),
     "RequestError"),
    (httpx.ReadTimeout("timed out"), "ReadTimeout"),
    (httpx.RemoteProtocolError("Server disconnected"), "RemoteProtocolError")])
def test_a_call_no_response_came_back_to_is_told_without_its_text(monkeypatch, caplog,
                                                                  cause, name):
    """What the caller is told and what the broker logs name the exception and its
    last cause and nothing of their text, which holds the address that was tried."""
    monkeypatch.setattr(llm, "provider_variable_tuple", lambda: ())

    def fail(param_dict):
        try:
            _raise_from(RuntimeError("Cannot connect to host 10.1.2.3:8443"), cause)
        except RuntimeError as e:
            raise litellm.InternalServerError(
                message=f"XException - {e}", llm_provider="x", model="m") from e

    monkeypatch.setattr(providercall, "completion", fail)
    with caplog.at_level(logging.DEBUG):
        reply_dict = broker.answer({"jsonrpc": "2.0", "id": 1, "method": "completion",
                                    "params": dict(_ASK_DICT)}, broker._Budget())
    assert reply_dict["error"] == {
        "code": broker._CALL_FAILURE, "data": {"error_type": "InternalServerError"},
        "message": f"InternalServerError: the endpoint gave no answer ({name}; {_NO_ROUTE})"}
    assert [record.getMessage() for record in caplog.records] == [
        f"the call failed: InternalServerError: InternalServerError: the endpoint gave "
        f"no answer ({name}; {_NO_ROUTE}). model={_ASK_DICT['model']}"]


@pytest.mark.parametrize("method, param_dict, name, line", [
    (broker.SEARCH, {"question": "q"}, "search", "the search was not made"),
    (broker.FETCH, {"url": "https://example.com/"}, "fetch", "the page was not read")])
def test_a_search_or_a_page_read_that_got_no_response_is_told_the_same_way(
        monkeypatch, caplog, method, param_dict, name, line):
    monkeypatch.setattr(llm, "provider_variable_tuple", lambda: ())

    def fail(value):
        _raise_from(RuntimeError("the request failed: <urlopen error [Errno 113] "
                                 "No route to host 10.1.2.3>"),
                    OSError(113, "No route to host 10.1.2.3"))

    monkeypatch.setattr(websearch, name, fail)
    reply_dict = broker.answer({"jsonrpc": "2.0", "id": 1, "method": method,
                                "params": param_dict}, broker._Budget())
    want = f"RuntimeError: the endpoint gave no answer (OSError EHOSTUNREACH; {_NO_ROUTE})"
    assert reply_dict["error"]["message"] == want
    assert [record.getMessage() for record in caplog.records
            if record.levelname == "WARNING"] == [f"{line}: RuntimeError: {want}"]


@pytest.mark.parametrize("error", [
    RuntimeError("401: the key was refused"),
    litellm.Timeout(message="no part of the reply arrived for 120 s", model="m",
                    llm_provider=""),
    TimeoutError("timed out"),
    httpx.HTTPStatusError("Client error '401 Unauthorized'", request=None, response=None)])
def test_a_failure_the_endpoint_answered_with_is_told_as_it_is(error):
    assert broker.failure_text(error) == str(error)
    try:
        _raise_from(RuntimeError("the search failed: 429"), error)
    except RuntimeError as e:
        assert broker.failure_text(e) == "the search failed: 429"


def test_a_wait_of_this_process_that_ran_out_is_told_in_its_own_words():
    """The time limits of providercall end a wait with a TimeoutError that carries no
    errno, which is not a call no response came back to."""
    try:
        _raise_from(litellm.Timeout(message="the reply was not complete after 30 s",
                                    model="m", llm_provider=""), TimeoutError())
    except litellm.Timeout as e:
        assert broker.failure_text(e) == str(e)
        assert "the reply was not complete after 30 s" in str(e)


@pytest.mark.parametrize("name, value, text, want", [
    ("HTTPS_PROXY", "http://puser:proxy-pw-123@ctloom-proxy.corp.example:3128",
     "403, message='Forbidden', url='http://puser:proxy-pw-123@ctloom-proxy.corp.example:3128'",
     "403, message='Forbidden', url='<the broker's own>'"),
    ("https_proxy", "http://puser:proxy-pw-123@ctloom-proxy.corp.example:3128",
     "url='http://REDACTED@ctloom-proxy.corp.example:3128' for puser:proxy-pw-123",
     "url='http://REDACTED@<the broker's own>' for <the broker's own>"),
    ("HTTP_PROXY", "ctloom-proxy.corp.example:3128",
     "Cannot connect to host ctloom-proxy.corp.example:3128 or to ctloom-proxy.corp.example",
     "Cannot connect to host <the broker's own> or to <the broker's own>"),
    ("ALL_PROXY", "socks5://ctloom-socks.corp.example:1080",
     "socks at ctloom-socks.corp.example:1080", "socks at <the broker's own>"),
    ("no_proxy", "localhost, .corp.example,10.20.30.40:8443,short",
     "localhost and llm.corp.example and 10.20.30.40:8443 and 10.20.30.40 and short",
     "localhost and llm<the broker's own> and <the broker's own> and "
     "<the broker's own> and short"),
    # A word that is no host name with a domain and no address is left in the text
    ("NO_PROXY", "internal,127.0.0.1,buildhost,fd00:1234::1",
     "internal_server_error at 127.0.0.1 on buildhost or fd00:1234::1",
     "internal_server_error at 127.0.0.1 on buildhost or <the broker's own>"),
    # The names of this machine itself are left; with a port they are not
    ("HTTPS_PROXY", "http://localhost:3128",
     "localhost refused; the proxy http://localhost:3128, localhost:3128",
     "localhost refused; the proxy <the broker's own>, <the broker's own>"),
    # The user and the password as a client that cut the URL elsewhere writes them
    ("ALL_PROXY", "socks://svc-proxy-user:secretpw99@ctloom-proxy.corp.example:1080/",
     "Unknown scheme for proxy URL URL('socks://svc-proxy-user:[secure]@"
     "ctloom-proxy.corp.example:1080/')",
     "Unknown scheme for proxy URL URL('socks://<the broker's own>:[secure]@"
     "<the broker's own>/')"),
    ("HTTPS_PROXY", "http://svc:Secr3tPw#Pass99x@ctloom-proxy.corp.example:3128",
     "Invalid port: 'Secr3tPw'", "Invalid port: '<the broker's own>'"),
    # A URL urlsplit() does not read still has its host and port taken out
    ("HTTPS_PROXY", "http://svc-proxy-user:secr[et]pw99@ctloom-proxy.corp.example:3128",
     "AnthropicException - http://REDACTED@ctloom-proxy.corp.example:3128.",
     "AnthropicException - http://REDACTED@<the broker's own>.")])
def test_what_the_proxy_variables_name_their_hosts_by_is_taken_out(monkeypatch, name,
                                                                   value, text, want):
    monkeypatch.setattr(llm, "provider_variable_tuple", lambda: ())
    for other in config.PROXY_NAME_TUPLE:
        monkeypatch.delenv(other, raising=False)
        monkeypatch.delenv(other.lower(), raising=False)
    assert broker.hide_own_value(text) == text
    monkeypatch.setenv(name, value)
    assert broker.hide_own_value(text) == want


def test_a_call_no_response_came_back_to_names_the_route_variables_that_are_set(
        monkeypatch, tmp_path):
    """The names alone: no proxy's address and no path is in what the caller is told."""
    monkeypatch.setattr(llm, "provider_variable_tuple", lambda: ())
    monkeypatch.setenv("HTTPS_PROXY", "http://puser:proxy-pw-123@ctloom-proxy.corp.example:3128")
    monkeypatch.setenv("no_proxy", "localhost,.corp.example")
    (tmp_path / "ctloom-ca.pem").write_text("", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "ctloom-ca.pem"))
    try:
        _raise_from(RuntimeError("Cannot connect to host ctloom-proxy.corp.example:3128"),
                    ConnectionRefusedError(111, "refused"))
    except RuntimeError as e:
        text = broker.failure_text(e)
    assert text == ("RuntimeError: the endpoint gave no answer (ConnectionRefusedError; "
                    "HTTPS_PROXY, no_proxy, SSL_CERT_FILE set where the call was made)")
    assert broker.hide_own_value(text) == text


def test_a_connection_urllib_could_not_make_in_time_is_a_call_no_response_came_back_to():
    import urllib.error
    try:
        _raise_from(RuntimeError("the search request failed: URLError: <urlopen error "
                                 "timed out>"), urllib.error.URLError(TimeoutError("timed out")))
    except RuntimeError as e:
        assert broker.failure_text(e) == (
            f"RuntimeError: the endpoint gave no answer (URLError; {_NO_ROUTE})")


@pytest.mark.parametrize("cause", [
    "tunnel", TimeoutError("timed out")])
def test_what_a_proxy_answered_and_a_reply_that_stopped_are_told_as_they_are(cause):
    """A proxy that turns the tunnel down has answered, and a reply that stops coming
    came over a connection that was made."""
    import urllib.error
    if cause == "tunnel":
        cause = urllib.error.URLError(
            OSError("Tunnel connection failed: 407 Proxy Authentication Required"))
    try:
        _raise_from(RuntimeError(f"the search request failed: {cause}"), cause)
    except RuntimeError as e:
        assert broker.failure_text(e) == str(e)


def _short_value(monkeypatch) -> None:
    """Set a key, a provider's key and a provider's plain value of under 8 characters."""
    monkeypatch.setattr(config, "LLM_API_KEY", "test")
    monkeypatch.setattr(config, "WEB_SEARCH_API_KEY", "a-key-of-12c")
    monkeypatch.setattr(llm, "provider_variable_tuple",
                        lambda: ("ONE_API_KEY", "ONE_PROFILE", "TWO_API_KEY"))
    monkeypatch.setenv("ONE_API_KEY", "key")
    monkeypatch.setenv("ONE_PROFILE", "dev")
    monkeypatch.setenv("TWO_API_KEY", "k-2-of-two")


def test_a_value_under_eight_characters_is_left_where_it_stands(monkeypatch):
    _short_value(monkeypatch)
    text = "run_test(['tests/test_key.py']) for dev with k-2-of-two and a-key-of-12c"
    assert broker.hide_own_value(text) == (
        "run_test(['tests/test_key.py']) for dev with <the broker's own> and "
        "<the broker's own>")
    monkeypatch.setattr(config, "LLM_API_KEY", "12345678")
    assert broker.hide_own_value("a 12345678 b") == "a <the broker's own> b"


def test_a_short_key_does_not_mark_the_headers_that_hold_its_letters(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_API_KEY", "a")
    monkeypatch.setattr(config, "WEB_SEARCH_HEADER", {"Accept": "application/json"})
    assert broker.hide_own_value("sent as application/json") == "sent as application/json"


def test_the_keys_under_eight_characters_are_named_when_the_broker_starts(monkeypatch,
                                                                         caplog):
    _short_value(monkeypatch)
    assert broker.short_key_name_list() == ["LLM_API_KEY", "ONE_API_KEY"]

    def refuse(family, kind):
        raise OSError("no socket here")

    monkeypatch.setattr(broker.socket, "socket", refuse)
    with pytest.raises(OSError, match="no socket here"):
        broker.serve("tcp://127.0.0.1:9")
    line_list = [record.getMessage() for record in caplog.records
                 if record.levelname == "WARNING"]
    assert line_list == ["LLM_API_KEY, ONE_API_KEY holds fewer than 8 characters, so its "
                         "value is not taken out of the failure messages the broker "
                         "sends back and logs"]
    assert "test" not in line_list[0].split() and "key" not in line_list[0].split()


def test_keys_of_eight_characters_or_more_are_named_by_no_warning(monkeypatch, caplog):
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-the-real-one")
    monkeypatch.setattr(llm, "provider_variable_tuple", lambda: ())
    assert broker.short_key_name_list() == []


def test_a_header_that_carries_the_key_is_hidden_whole(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_API_KEY", "jina-secret")
    monkeypatch.setattr(config, "WEB_FETCH_HEADER", {"Authorization": "Bearer jina-secret",
                                                     "Accept": "application/json"})
    assert broker.hide_own_value("sent Bearer jina-secret as application/json") == \
        "sent <the broker's own> as application/json"


def test_the_filter_takes_the_brokers_values_out_of_a_record(monkeypatch):
    import logging
    import sys
    monkeypatch.setattr(config, "LLM_API_BASE", "http://base.invalid")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-the-real-one")
    try:
        raise RuntimeError("401 at http://base.invalid/v1 for sk-the-real-one")
    except RuntimeError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("some.library", logging.DEBUG, "f.py", 1,
                               "POST %s with %s",
                               ("http://base.invalid/v1", "sk-the-real-one"), exc_info)
    assert broker.OwnValueFilter().filter(record) is True
    text = logging.Formatter("%(message)s").format(record)
    assert "http://base.invalid" not in text and "sk-the-real-one" not in text
    assert text.startswith("POST <the broker's own>/v1 with <the broker's own>\n")
    assert text.endswith("RuntimeError: 401 at <the broker's own>/v1 for <the broker's own>")


def test_a_record_logged_while_the_filter_reads_the_values_is_passed_as_it_is(
        monkeypatch):
    import logging
    record_filter = broker.OwnValueFilter()
    inner = logging.LogRecord("x", logging.WARNING, "f.py", 1, "inner %s", ("one",), None)
    seen = []

    def hide(text):
        seen.append(record_filter.filter(inner))
        return text.upper()

    monkeypatch.setattr(broker, "hide_own_value", hide)
    outer = logging.LogRecord("x", logging.WARNING, "f.py", 1, "outer", (), None)
    assert record_filter.filter(outer) is True
    assert (outer.getMessage(), inner.getMessage(), seen) == ("OUTER", "inner one", [True])


def test_the_three_methods_are_named_when_one_is_not_known(fake):
    reply = broker.answer({"id": 1, "method": "shell", "params": {}},
                          broker._Budget())
    for method in (broker.COMPLETION, broker.SEARCH, broker.REFILL):
        assert method in reply["error"]["message"]


# == The limits, counted where the container cannot reach them ==

def _completion(ident=1):
    """Return one completion request, as a caller that skipped make_llm() would send."""
    return {"id": ident, "method": "completion",
            "params": {"model": "some/model", "max_tokens": 10, "messages": _MESSAGE_LIST}}


def test_a_search_past_the_limit_is_refused_however_it_was_sent(monkeypatch):
    """The tool's own counter is in the container; this one is not."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    monkeypatch.setattr(websearch, "search", lambda q: list(_RESULTS))
    budget = broker._Budget()
    assert "result" in broker.answer({"id": 1, "method": "search",
                                      "params": {"question": "one"}}, budget)
    second = broker.answer({"id": 2, "method": "search",
                            "params": {"question": "two"}}, budget)
    assert "WEB_SEARCH_MAX_CALL" in second["error"]["message"]


def test_a_search_is_refused_when_the_tool_is_off(monkeypatch):
    """0 hands the tool to neither stage, so no search belongs to the run either."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    monkeypatch.setattr(websearch, "search",
                        lambda q: pytest.fail("it left the machine"))
    reply = broker.answer({"id": 1, "method": "search",
                           "params": {"question": "one"}}, broker._Budget())
    assert "WEB_SEARCH_MAX_CALL" in reply["error"]["message"]


def test_the_model_calls_are_counted_when_a_limit_is_set(monkeypatch, fake):
    monkeypatch.setattr(config, "LLM_MAX_CALL", 2)
    budget = broker._Budget()
    assert "result" in broker.answer(_completion(1), budget)
    assert "result" in broker.answer(_completion(2), budget)
    third = broker.answer(_completion(3), budget)
    assert "LLM_MAX_CALL" in third["error"]["message"]


def test_no_limit_on_the_model_calls_is_the_default(monkeypatch, fake):
    """A run makes as many as its own loop allows; what one costs is capped elsewhere."""
    monkeypatch.setattr(config, "LLM_MAX_CALL", 0)
    budget = broker._Budget()
    for ident in range(5):
        assert "result" in broker.answer(_completion(ident), budget)


def test_a_refused_call_is_not_made_at_all(monkeypatch, fake):
    monkeypatch.setattr(config, "LLM_MAX_CALL", 1)
    budget = broker._Budget()
    broker.answer(_completion(1), budget)
    fake.kwargs = None
    broker.answer(_completion(2), budget)
    assert fake.kwargs is None


def test_a_limit_rewritten_at_run_time_is_followed(monkeypatch, fake):
    """The limits are read as each call is served, not held from when it started."""
    monkeypatch.setattr(config, "LLM_MAX_CALL", 1)
    budget = broker._Budget()
    assert "result" in broker.answer(_completion(1), budget)
    monkeypatch.setattr(config, "LLM_MAX_CALL", 2)
    assert "result" in broker.answer(_completion(2), budget)


def test_a_budget_of_its_own_is_what_one_run_gets(monkeypatch, fake):
    """sandbox.run() starts a broker for one run, so the next run starts full."""
    monkeypatch.setattr(config, "LLM_MAX_CALL", 1)
    assert "result" in broker.answer(_completion(1), broker._Budget())
    assert "result" in broker.answer(_completion(1), broker._Budget())


# == What a completion may ask for ============================

@pytest.mark.parametrize("change_dict, word", [
    ({"model": "openai/gpt-4o"}, "*_MODEL"),
    ({"model": ["m"]}, "*_MODEL"),
    ({"max_tokens": 128000}, "*_MAX_TOKEN"),
    ({"max_tokens": 0}, "*_MAX_TOKEN"),
    ({"max_tokens": True}, "*_MAX_TOKEN"),
    ({"max_tokens": "10"}, "*_MAX_TOKEN"),
    ({"thinking": {"type": "enabled", "budget_tokens": 120000}}, "thinking"),
    ({"thinking": {"type": "enabled", "budget_tokens": 3000}}, "thinking"),
    ({"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}}, "thinking"),
    ({"output_config": {"effort": "low"}}, "thinking")])
def test_a_call_outside_the_brokers_settings_is_refused_uncounted(monkeypatch, fake,
                                                                   change_dict, word):
    monkeypatch.setattr(config, "LLM_MAX_CALL", 1)
    budget = broker._Budget()
    param_dict = {"model": "m", "max_tokens": 16384, "messages": _MESSAGE_LIST, **change_dict}
    reply = broker.answer({"id": 1, "method": "completion", "params": param_dict},
                          budget)
    assert word in reply["error"]["message"] and "data" not in reply["error"]
    assert fake.kwargs is None and budget.remain_count(broker.COMPLETION) == 1


def test_a_call_with_no_max_tokens_is_refused(fake):
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": {"model": "m", "messages": _MESSAGE_LIST}}, broker._Budget())
    assert "max_tokens=None" in reply["error"]["message"] and fake.kwargs is None


def test_params_that_are_not_an_object_are_refused(fake):
    reply = broker.answer({"id": 1, "method": "completion", "params": ["model"]},
                          broker._Budget())
    assert "not a JSON object" in reply["error"]["message"] and fake.kwargs is None


@pytest.mark.parametrize("level", ["", "off", "low", "medium", "high"])
def test_every_form_of_thinking_make_llm_sends_is_made(fake, level):
    for think_dict in llm.think_param_chain(level, 16384):
        reply = broker.answer({"id": 1, "method": "completion", "params": {
            "model": "m", "max_tokens": 16384, "messages": _MESSAGE_LIST, **think_dict}},
            broker._Budget())
        assert "result" in reply, reply


def test_the_largest_max_token_setting_is_the_limit(monkeypatch, fake):
    monkeypatch.setattr(config, "PLAN_MAX_TOKEN", 100000)
    param_dict = {"model": "m", "max_tokens": 100000, "messages": _MESSAGE_LIST}
    assert "result" in broker.answer({"id": 1, "method": "completion",
                                      "params": param_dict}, broker._Budget())
    param_dict["max_tokens"] = 100001
    assert "error" in broker.answer({"id": 2, "method": "completion",
                                     "params": param_dict}, broker._Budget())


@pytest.mark.parametrize("message_list", [
    None, "q", [],
    [{"role": "user", "content": "q"}, {"role": "user", "content": "q"}],
    [{"role": "system", "content": "q"}],
    [{"role": "assistant", "content": "q"}],
    [{"role": "user", "content": "q", "name": "x"}],
    [{"role": "user"}],
    [{"role": "user", "content": None}],
    [{"role": "user", "content": ["q"]}],
    [{"role": "user", "content": [{"type": "text", "text": 5}]}],
    [{"role": "user", "content": [
        {"type": "text", "text": "q"},
        {"type": "image_url", "image_url": {"url": "http://collector.invalid/x.png?d=R"}}]}],
    [{"role": "user", "content": [
        {"type": "document", "source": {"type": "url", "url": "http://collector.invalid/"}}]}],
    [{"role": "user", "content": [
        {"type": "text", "text": "q", "source": {"url": "http://collector.invalid/"}}]}],
    [{"role": "user", "content": [
        {"type": "text", "text": "q", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}],
])
def test_messages_of_another_form_than_make_llm_sends_are_refused_uncounted(
        monkeypatch, fake, message_list):
    monkeypatch.setattr(config, "LLM_MAX_CALL", 1)
    budget = broker._Budget()
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": {**_ASK_DICT, "messages": message_list}}, budget)
    assert "the call's messages are not one message" in reply["error"]["message"]
    assert fake.kwargs is None and budget.remain_count(broker.COMPLETION) == 1


def test_a_call_without_messages_is_refused(fake):
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": {"model": "m", "max_tokens": 10}}, broker._Budget())
    assert "the call's messages" in reply["error"]["message"] and fake.kwargs is None


@pytest.mark.parametrize("prompt", [
    "PROMPT", "", [("head", True), ("body", False)], [("only", True)], []])
def test_every_form_of_message_make_llm_sends_is_made(serving, fake, prompt):
    assert llm.make_llm("m", 10)(prompt) == "ok"
    content = prompt if isinstance(prompt, str) else llm.content_list_of(prompt)
    assert fake.kwargs["messages"] == [{"role": "user", "content": content}]


def test_the_settings_the_container_is_not_given_allow_nothing(monkeypatch, fake):
    monkeypatch.setattr(config, "WEB_SEARCH_MODEL", "other/model")
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_TOKEN", 100000)
    for change_dict in ({"model": "other/model"}, {"max_tokens": 100000}):
        reply = broker.answer({"id": 1, "method": "completion", "params": {
            "model": "m", "max_tokens": 10, "messages": _MESSAGE_LIST, **change_dict}},
            broker._Budget())
        assert "error" in reply
    assert fake.kwargs is None


def test_a_stage_whose_model_the_broker_refuses_ends_with_the_error(serving, fake):
    """The refusal reaches the container as a failed call, which ends the loop."""
    call = llm.make_llm("openai/gpt-4o", 10, is_turn_reply=True)
    with pytest.raises(RuntimeError, match="no \\*_MODEL setting"):
        call("q")
    result_dict = rlm.RLM(call, [], "I", max_turn=3).run("REQ", {}, "V")
    assert result_dict["answer"] is None and "*_MODEL" in result_dict["error"]
    assert result_dict["iters"] == 0 and fake.kwargs is None


# == What the reply must not carry back =======================

def test_a_key_quoted_by_a_failure_does_not_reach_the_caller(monkeypatch):
    """A provider's own error quotes what it was handed: a key it could not send comes
    back as the header it was put in. The caller is the container the credentials are
    kept from, and it records what it is told."""
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-the-real-one")
    monkeypatch.setattr(broker.litellm, "completion", _FakeCompletion(
        raises=RuntimeError("Illegal header value b'sk-the-real-one\r'")))
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": dict(_ASK_DICT)}, broker._Budget())
    assert "sk-the-real-one" not in json.dumps(reply)
    assert "Illegal header value" in reply["error"]["message"]


def test_what_the_search_is_authorised_with_does_not_reach_the_caller(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    monkeypatch.setattr(config, "WEB_SEARCH_API_KEY", "the-search-key")
    monkeypatch.setattr(config, "WEB_SEARCH_HEADER", {"X-Key": "the-search-key"})
    monkeypatch.setattr(broker.websearch, "search",
                        _raises(RuntimeError("401 with X-Key: the-search-key")))
    reply = broker.answer({"id": 1, "method": "search",
                           "params": {"question": "q"}}, broker._Budget())
    assert "the-search-key" not in json.dumps(reply)


def test_the_presets_key_does_not_reach_the_caller_either(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    monkeypatch.setattr(config, "WEB_SEARCH_API_KEY", "tvly-the-key")
    monkeypatch.setattr(config, "WEB_SEARCH_HEADER", {})
    monkeypatch.setattr(broker.websearch, "search",
                        _raises(RuntimeError("401 for Bearer tvly-the-key")))
    reply = broker.answer({"id": 1, "method": "search",
                           "params": {"question": "q"}}, broker._Budget())
    assert "tvly-the-key" not in json.dumps(reply)


def test_a_failure_naming_nothing_of_the_brokers_is_passed_on_as_it_is(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-the-real-one")
    monkeypatch.setattr(broker.litellm, "completion",
                        _FakeCompletion(raises=RuntimeError("the socket went away")))
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": dict(_ASK_DICT)}, broker._Budget())
    assert reply["error"]["message"] == "the socket went away"


def _raises(error):
    """Return a stand-in that raises error whatever it is called with."""
    def raise_it(*args, **kwargs):
        raise error
    return raise_it


def test_what_a_reply_says_about_anything_else_is_left_alone(monkeypatch):
    """A model may legitimately write about an address or a search setting."""
    said = "it calls https://api.example/v1 and that is fine"
    monkeypatch.setattr(config, "LLM_API_BASE", "https://api.example/v1")
    monkeypatch.setattr(broker.litellm, "completion", _FakeCompletion(content=said))
    reply = broker.answer({"id": 1, "method": "completion",
                           "params": dict(_ASK_DICT)}, broker._Budget())
    assert reply["result"]["content"] == said


# == What the client says when the broker does not answer ===
def test_a_broker_that_is_not_there_is_said_with_how_to_start_it(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLM_BROKER", f"{broker.UNIX_SCHEME}{tmp_path}/none.sock")
    with pytest.raises(RuntimeError, match="did not answer"):
        broker.complete({"model": "m", "max_tokens": 10, "messages": _MESSAGE_LIST})


def _server_that_writes(reply_byte_list: list[bytes]) -> str:
    """Start a Unix socket server answering one connection with the bytes given.

    Returns:
        The address, in the form config.LLM_BROKER takes.
    """
    directory = tempfile.mkdtemp(prefix="ctloom-test-")
    path = f"{directory}/llm.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)

    def serve_one():
        conn, _ = server.accept()
        with conn, conn.makefile("rb") as reader:
            reader.readline()
            for reply in reply_byte_list:
                conn.sendall(reply)
        server.close()
        shutil.rmtree(directory, ignore_errors=True)

    threading.Thread(target=serve_one, daemon=True).start()
    return f"{broker.UNIX_SCHEME}{path}"


def test_a_broker_that_closes_without_answering_is_said(monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", _server_that_writes([]))
    with pytest.raises(RuntimeError, match="closed the connection without answering"):
        broker.complete({"model": "m", "max_tokens": 10, "messages": _MESSAGE_LIST})


def test_a_broker_that_answers_with_something_that_is_not_json_is_said(monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", _server_that_writes([b"not json\n"]))
    with pytest.raises(RuntimeError, match="not JSON"):
        broker.complete({"model": "m", "max_tokens": 10, "messages": _MESSAGE_LIST})


def _server_that_reads_nothing(reply: bytes) -> str:
    """Start a Unix socket server that answers one connection with the bytes given
    and closes it, reading nothing of what the caller sends.

    Returns:
        The address, in the form config.LLM_BROKER takes.
    """
    directory = tempfile.mkdtemp(prefix="ctloom-test-")
    path = f"{directory}/llm.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)

    def serve_one():
        conn, _ = server.accept()
        with conn:
            conn.sendall(reply)
        server.close()
        shutil.rmtree(directory, ignore_errors=True)

    threading.Thread(target=serve_one, daemon=True).start()
    return f"{broker.UNIX_SCHEME}{path}"


_LARGE_ASK_DICT = {"model": "m", "max_tokens": 10,
                   "messages": [{"role": "user", "content": "x" * 5_000_000}]}


def test_the_refusal_of_a_broker_that_read_nothing_is_what_the_caller_is_told(
        monkeypatch):
    """The broker answered and closed before the request was sent whole: the caller
    reads the answer, and is not told to start a broker that is there."""
    reply = json.dumps(broker._error(None, broker._CALL_FAILURE,
                                     broker._TOO_MANY.format(limit=1))).encode() + b"\n"
    monkeypatch.setattr(config, "LLM_BROKER", _server_that_reads_nothing(reply))
    with pytest.raises(RuntimeError, match="this broker is already answering 1 caller"):
        broker.complete(_LARGE_ASK_DICT)


def test_a_broker_that_closes_before_the_request_is_sent_whole_did_not_answer(
        monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", _server_that_reads_nothing(b""))
    with pytest.raises(RuntimeError, match="did not answer .*Start it with") as raised:
        broker.complete(_LARGE_ASK_DICT)
    assert raised.value.why.endswith("Start it with:")
    assert raised.value.command == (
        f"codetwine-loom broker --endpoint {config.LLM_BROKER}")


def test_the_command_that_starts_a_broker_names_its_address_as_a_shell_reads_it(
        monkeypatch, tmp_path):
    address = f"unix://{tmp_path}/a  dir; $(x)/llm.sock"
    monkeypatch.setattr(config, "LLM_BROKER", address)
    with pytest.raises(broker.CommandError) as raised:
        broker.complete(dict(_ASK_DICT))
    assert shlex.split(raised.value.command) == [
        "codetwine-loom", "broker", "--endpoint", address]


@pytest.mark.parametrize("over_count", [1, 4_000_000, 30_000_000])
def test_a_request_over_the_limit_is_told_so_however_much_of_it_was_sent(
        serving, monkeypatch, fake, caplog, over_count):
    """The broker stops reading at the limit and answers; the caller that could not
    send the rest reads that answer, and the broker goes on serving."""
    monkeypatch.setattr(config, "LLM_BROKER_MAX_REQUEST_BYTE", 4096)
    ask_dict = {"model": "m", "max_tokens": 10,
                "messages": [{"role": "user", "content": "x" * (4096 + over_count)}]}
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.broker"):
        with pytest.raises(RuntimeError, match="the broker did not make the call: the "
                                               "request is over 4096 bytes"):
            broker.complete(ask_dict)
    assert "a caller was refused: the request is over 4096 bytes" in caplog.text
    assert fake.kwargs is None
    assert broker.complete(dict(_ASK_DICT)).message.content == "ok"


# == What the server does with a caller it will not serve ===
def _raw_line(url: str, payload: bytes, hold_list: list | None = None) -> dict:
    """Send raw bytes to the broker at url and read back one JSON line.

    Args:
        url: The address, as config.LLM_BROKER holds it.
        payload: What to send.
        hold_list: A list the open connection is appended to, to keep it open.
    """
    family, address = broker.endpoint(url)
    conn = socket.socket(family, socket.SOCK_STREAM)
    conn.settimeout(5)
    conn.connect(address)
    conn.sendall(payload)
    line = conn.makefile("rb").readline()
    if hold_list is not None:
        hold_list.append(conn)
    else:
        conn.close()
    return json.loads(line) if line else {}


def test_a_request_that_is_not_json_is_answered_with_an_error(serving):
    reply = _raw_line(serving, b"{not json\n")
    assert reply["error"]["code"] == broker._PARSE_ERROR
    assert "not JSON" in reply["error"]["message"]


@pytest.mark.parametrize("line", [
    b'{"id": 1, "method": "completion", "params": "\xff\xfe"}',
    b"\xff\xfe",
    b"[" * 200000,
    b'{"id": 1, "method": "search", "params": {"question": ' + b"[" * 100000
    + b"]" * 100000 + b"}}",
    b'{"id": ' + b"9" * 5000 + b', "method": "completion"}',
    '{"id": 1, "method": "completion"}'.encode("utf-16"),
])
def test_a_line_the_parser_cannot_read_is_answered_with_an_error(line):
    """Bytes that are not UTF-8, nesting too deep and a number too long are not JSON."""
    ours, theirs = socket.socketpair()
    with theirs:
        theirs.sendall(line + b"\n")
        theirs.shutdown(socket.SHUT_WR)
        broker._serve_one(ours, broker._Budget())
        reply = json.loads(theirs.makefile("rb").readline())
    assert reply["error"]["code"] == broker._PARSE_ERROR and reply["id"] is None
    assert "not JSON" in reply["error"]["message"]


def test_an_empty_line_is_passed_over_and_the_request_after_it_is_answered(serving, fake):
    request = json.dumps({"jsonrpc": "2.0", "id": 7, "method": broker.COMPLETION,
                          "params": {"model": "m", "max_tokens": 10, "messages": _MESSAGE_LIST}})
    reply = _raw_line(serving, b"\n" + request.encode() + b"\n")
    assert reply["id"] == 7 and "result" in reply


@pytest.fixture
def socket_file():
    """Bind a Unix socket in a directory of its own, close it and hand out its path.

    The directory is taken away at the end of the test.

    Yields:
        The path of the socket file left behind.
    """
    directory = tempfile.mkdtemp(prefix="ctloom-test-")
    path = f"{directory}/llm.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
    yield path
    shutil.rmtree(directory, ignore_errors=True)


def test_a_socket_file_that_will_not_go_is_said_in_the_log(socket_file, monkeypatch,
                                                           caplog):
    def refuse(path):
        raise PermissionError("read-only")

    with monkeypatch.context() as patch:
        patch.setattr(broker.os, "unlink", refuse)
        broker._clear_socket_file(socket_file)
    assert "could not be taken away" in caplog.text


def test_a_socket_left_at_the_path_is_taken_away_and_nothing_there_is_fine(tmp_path,
                                                                           socket_file):
    path = socket_file
    broker._clear_socket_file(path)
    assert not os.path.lexists(path)
    broker._clear_socket_file(str(tmp_path / "none.sock"))


def test_a_file_at_the_path_of_the_socket_is_left_and_the_broker_stops(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hi\n", encoding="utf-8")
    link = tmp_path / "link.sock"
    link.symlink_to(path)
    for at in (path, link, tmp_path):
        with pytest.raises(OSError, match="is not a socket"):
            broker.serve(f"{broker.UNIX_SCHEME}{at}")
    assert path.read_text(encoding="utf-8") == "hi\n" and link.is_symlink()


def test_a_count_of_callers_the_broker_cannot_hold_stops_it_before_it_binds(
        monkeypatch, tmp_path, socket_file):
    path = socket_file
    monkeypatch.setattr(config, "LLM_BROKER_MAX_CONNECTION", -1)
    with pytest.raises(ValueError):
        broker.serve(f"{broker.UNIX_SCHEME}{path}")
    assert os.path.lexists(path)
    with pytest.raises(ValueError):
        broker.serve(f"{broker.UNIX_SCHEME}{tmp_path}/new.sock")
    assert not os.path.lexists(tmp_path / "new.sock")


def test_a_connection_that_ends_before_it_is_read_is_said_in_the_log(caplog):
    import logging

    class _GoneConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def makefile(self, mode):
            raise OSError("connection reset")

    with caplog.at_level(logging.INFO):
        broker._serve_one(_GoneConn(), broker._Budget(None))
    assert "the connection ended before it was answered" in caplog.text


def test_serving_stops_when_the_process_is_interrupted(monkeypatch, caplog):
    import logging

    class _Server:
        def __init__(self, family, kind):
            self.call_list = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def setsockopt(self, *a):
            self.call_list.append("setsockopt")

        def bind(self, address):
            self.call_list.append(("bind", address))

        def listen(self, backlog):
            self.call_list.append(("listen", backlog))

        def accept(self):
            raise KeyboardInterrupt

    made = []

    def make(family, kind):
        server = _Server(family, kind)
        made.append(server)
        return server

    monkeypatch.setattr(broker.socket, "socket", make)
    with caplog.at_level(logging.INFO):
        broker.serve("tcp://127.0.0.1:9")
    assert "the broker was stopped" in caplog.text
    assert made[0].call_list == ["setsockopt", ("bind", ("127.0.0.1", 9)),
                                 ("listen", config.LLM_BROKER_MAX_CONNECTION)]


def test_the_model_table_entries_of_the_settings_models_are_handed_over(monkeypatch):
    """An entry is found under the model's name as the setting writes it, and under the
    name without its provider; the models the table does not hold are left out."""
    monkeypatch.setattr(litellm, "model_cost", {
        "m": {"litellm_provider": "openai", "max_tokens": 1},
        "anthropic/m": {"litellm_provider": "anthropic", "max_tokens": 2},
        "another": {"litellm_provider": "openai", "max_tokens": 3}})
    assert json.loads(broker.model_table_text()) == {
        "m": {"litellm_provider": "openai", "max_tokens": 1},
        "anthropic/m": {"litellm_provider": "anthropic", "max_tokens": 2}}
    monkeypatch.setattr(litellm, "model_cost", {})
    assert broker.model_table_text() == "{}"


def test_the_entry_of_a_model_litellm_signs_in_for_is_handed_over_without_a_sign_in(
        monkeypatch, tmp_path):
    """Found under the model's name as the setting writes it; nothing is made in HOME."""
    entry = {"litellm_provider": "github_copilot", "max_tokens": 1}
    for name, value in config.setting_value_dict("_MODEL").items():
        if value:
            monkeypatch.setattr(config, name, "github_copilot/m")
    monkeypatch.setattr(litellm, "model_cost", {"github_copilot/m": entry})
    monkeypatch.setenv("HOME", str(tmp_path))
    assert json.loads(broker.model_table_text()) == {"github_copilot/m": entry}
    assert list(tmp_path.iterdir()) == []


def test_the_entry_of_a_model_only_the_broker_reads_is_handed_over(monkeypatch):
    entry = {"litellm_provider": "openai", "max_tokens": 1}
    monkeypatch.setattr(config, "WEB_SEARCH_MODEL", "search-only")
    monkeypatch.setattr(litellm, "model_cost", {"search-only": entry})
    assert json.loads(broker.model_table_text()) == {"search-only": entry}


def _new_entry(provider, **more_dict):
    """Return a model table entry of a chat model of one provider."""
    return {"litellm_provider": provider, "mode": "chat", "max_tokens": 1000,
            "input_cost_per_token": 1e-06, "output_cost_per_token": 1e-06, **more_dict}


# Models the table litellm ships with does not hold, as a table fetched when litellm is
# imported would: {the setting's value: (the entry's key, the entry)}. Some are named
# without a provider, some with one, and some with a region or a route between the
# provider and the model's id
_NEW_MODEL_DICT = {
    "ctloom-t-oai": ("ctloom-t-oai", _new_entry("openai")),
    "ctloom-t-claude": ("ctloom-t-claude", _new_entry("anthropic", thinking_always_on=True)),
    "gemini-ctloom-t": ("gemini-ctloom-t", _new_entry("vertex_ai-language-models")),
    "us.ctloom-t.conv-v1:0": ("us.ctloom-t.conv-v1:0",
                           _new_entry("bedrock_converse", thinking_always_on=True)),
    "command-ctloom-t": ("command-ctloom-t", _new_entry("cohere_chat")),
    "ctloom-t.bedrock-old-v1": ("ctloom-t.bedrock-old-v1", _new_entry("bedrock")),
    "anthropic/ctloom-t-claude-2": ("ctloom-t-claude-2",
                                 _new_entry("anthropic", thinking_always_on=True)),
    "vertex_ai/gemini-ctloom-t-2": ("gemini-ctloom-t-2",
                                 _new_entry("vertex_ai-language-models")),
    "gemini/gemini-ctloom-t-3": ("gemini/gemini-ctloom-t-3", _new_entry("gemini")),
    "bedrock/us.ctloom-t.conv-2-v1:0": (
        "us.ctloom-t.conv-2-v1:0", _new_entry("bedrock_converse", thinking_always_on=True)),
    "bedrock/us-west-2/us.ctloom-t.conv-3-v1:0": (
        "us.ctloom-t.conv-3-v1:0", _new_entry("bedrock_converse", thinking_always_on=True)),
    "bedrock/invoke/us.ctloom-t.conv-4-v1:0": (
        "us.ctloom-t.conv-4-v1:0", _new_entry("bedrock_converse", thinking_always_on=True)),
    "bedrock/converse/us.ctloom-t.conv-5-v1:0": (
        "us.ctloom-t.conv-5-v1:0", _new_entry("bedrock_converse", thinking_always_on=True)),
    "ctloom-t-oai-search": ("ctloom-t-oai-search", _new_entry("openai")),
}

# What a process started with the entries in MODEL_TABLE_ENV answers for the models
# named as its arguments, once it has put the entries in as the broker does
_BROKER_ANSWER_CODE = """
import json, os, sys
from codetwine_loom import broker
sys.path.insert(0, os.environ["CTLOOM_TEST_DIR"])
import test_broker
broker.load_model_table(os.environ[broker.MODEL_TABLE_ENV])
print(json.dumps({model: test_broker._model_answer(model) for model in sys.argv[1:]}))
"""


def _model_answer(model):
    """Return what this process's litellm says of a model: its provider, whether it
    always thinks, and the entry litellm finds for it (None when it finds none)."""
    try:
        info_dict = json.loads(json.dumps(litellm.get_model_info(model), default=str))
    except Exception:                     # noqa: BLE001 - an unknown model raises
        info_dict = None
    return [llm.provider_of(model), providercall.always_thinks(model), info_dict]


def _answer_pair(monkeypatch, setting_dict, table_dict):
    """Return what this process answers for the models of some settings while it
    holds entries as a table fetched at import would, and what a process answers that
    was started with the table litellm ships with and handed the entries.

    Returns:
        ({model: _model_answer()} of this process, the same of the other, the keys of
        the entries that were handed over, what llm.no_provider_text() said in this
        process).
    """
    import subprocess
    import sys
    for name, model in setting_dict.items():
        monkeypatch.setattr(config, name, model)
    model_list = list(setting_dict.values())
    for key, entry in table_dict.items():
        litellm.model_cost[key] = dict(entry)
    litellm.register_model(table_dict)
    litellm.add_known_models(table_dict)
    try:
        own_dict = {model: _model_answer(model) for model in model_list}
        no_provider = llm.no_provider_text()
        text = broker.model_table_text()
    finally:
        for key in table_dict:
            litellm.model_cost.pop(key, None)
        litellm.register_model({})
    proc = subprocess.run(
        [sys.executable, "-c", _BROKER_ANSWER_CODE, *model_list], capture_output=True,
        text=True, check=False, timeout=120,
        env=config.own_child_env({broker.LOCAL_MODEL_TABLE_ENV: "True",
                                  broker.MODEL_TABLE_ENV: text,
                                  "CTLOOM_TEST_DIR": os.path.dirname(__file__)}))
    assert proc.returncode == 0, proc.stderr
    return own_dict, json.loads(proc.stdout.strip().splitlines()[-1]), sorted(
        json.loads(text)), no_provider


def test_a_broker_answers_for_the_settings_models_as_the_process_that_started_it(
        monkeypatch):
    """The provider of every model the settings name, whether it always thinks and
    the entry litellm finds for it are the same in a process that holds the entries
    as a fetched table would and in one started with the table litellm ships with
    and the entries handed over."""
    table_dict = dict(_NEW_MODEL_DICT.values())
    name_list = sorted(name for name in config.setting_value_dict("_MODEL")
                       if name != "WEB_SEARCH_MODEL") + ["WEB_SEARCH_MODEL"]
    assert len(name_list) == len(_NEW_MODEL_DICT)
    own_dict, broker_dict, key_list, no_provider = _answer_pair(
        monkeypatch, dict(zip(name_list, _NEW_MODEL_DICT)), table_dict)
    assert key_list == sorted(table_dict) and no_provider == ""
    assert broker_dict == own_dict
    assert own_dict["gemini-ctloom-t"][:2] == ["vertex_ai", False]
    assert own_dict["bedrock/us-west-2/us.ctloom-t.conv-3-v1:0"][:2] == ["bedrock", True]


def _ship_base_key():
    """Return a key of the table litellm ships with that holds a Bedrock model and
    has no entry of its own under the "us." region, or "" when there is none."""
    return next((key for key, entry in sorted(litellm.model_cost.items())
                 if key.startswith("anthropic.")
                 and entry.get("litellm_provider") == "bedrock_converse"
                 and f"us.{key}" not in litellm.model_cost), "")


def test_an_entry_under_a_region_is_held_under_its_own_key_beside_the_one_it_names(
        monkeypatch):
    """An entry whose key is a region's name of a model keeps its key: where the
    settings name the model both ways, and where the table litellm ships with holds
    the model without the region."""
    new_entry = _new_entry("bedrock_converse", thinking_always_on=True)
    base, region = "anthropic.claude-ctloom-t-9-v1:0", "us.anthropic.claude-ctloom-t-9-v1:0"
    setting_dict = {"HIGH_MODEL": f"bedrock/{base}",
                    "MEDIUM_MODEL": f"bedrock/us-west-2/{region}"}
    table_dict = {base: new_entry, region: new_entry}
    ship_key = _ship_base_key()
    if ship_key:
        setting_dict["EDIT_MODEL"] = f"bedrock/us-west-2/us.{ship_key}"
        table_dict[f"us.{ship_key}"] = {
            **litellm.model_cost[ship_key],
            "thinking_always_on": not litellm.model_cost[ship_key].get(
                "thinking_always_on", False)}
    # The other model settings name no model
    for name in config.setting_value_dict("_MODEL"):
        if name not in setting_dict:
            monkeypatch.setattr(config, name, "")
    own_dict, broker_dict, key_list, _ = _answer_pair(monkeypatch, setting_dict,
                                                      table_dict)
    assert key_list == sorted(table_dict)
    assert broker_dict == own_dict
    for model, key in zip(setting_dict.values(), table_dict):
        assert own_dict[model][2]["key"] == key
        assert own_dict[model][1] is bool(table_dict[key].get("thinking_always_on"))


def test_the_entries_handed_over_are_put_into_the_model_table():
    name = "made-up-model-for-the-table-test"
    entry = {"litellm_provider": "anthropic", "mode": "chat", "max_tokens": 10,
             "input_cost_per_token": 1e-06, "output_cost_per_token": 1e-06,
             "thinking_always_on": True}
    assert not providercall.always_thinks(f"anthropic/{name}")
    try:
        broker.load_model_table(json.dumps({name: entry}))
        assert providercall.always_thinks(f"anthropic/{name}")
    finally:
        litellm.model_cost.pop(name, None)


@pytest.mark.parametrize("text", [
    "{not json", "[1, 2]", '{"m": 3}',
    '{"made-up-model-for-the-read-test": {"litellm_provider": "openai"}, "m": 3}'])
def test_entries_that_cannot_be_read_leave_the_table_as_it_is(text, caplog):
    count = len(litellm.model_cost)
    broker.load_model_table(text)
    assert "CTLOOM_MODEL_TABLE could not be read" in caplog.text
    assert len(litellm.model_cost) == count


def test_no_entries_handed_over_is_not_an_error(caplog):
    broker.load_model_table("")
    broker.load_model_table("{}")
    assert caplog.text == ""


def test_the_broker_reads_the_entries_before_it_checks_what_its_models_need(monkeypatch):
    """A model named without a provider is one the table alone names the provider of:
    the entries are in before the check of the credentials, and before the address."""
    from codetwine_loom import cli
    seen = []
    monkeypatch.setenv("CTLOOM_MODEL_TABLE", '{"x": {}}')
    monkeypatch.setattr(broker, "load_model_table", seen.append)
    monkeypatch.setattr(cli.llm, "missing_credential_text",
                        lambda: seen.append("credentials") or "")
    monkeypatch.setattr(cli, "_serve_broker", lambda args, handler: seen.append("serve"))
    cli.main(["broker", "--endpoint", "tcp://127.0.0.1:9"])
    assert seen == ['{"x": {}}', "credentials", "serve"]
    # No other subcommand reads them
    seen.clear()
    with pytest.raises(SystemExit):
        cli.main(["history", "--project", "/nonexistent/repo"])
    assert '{"x": {}}' not in seen


def test_serving_without_an_address_is_refused(monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "")
    with pytest.raises(ValueError, match="no address to listen on"):
        broker.serve("")


# == Reading one page of the web ===============================

_PAGE_TEXT = "the page, in full"
_PAGE_URL = "https://example.invalid/guide"


def test_a_page_read_goes_out_from_the_broker(serving, monkeypatch):
    """The container has no network, so the page is read from here."""
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    asked = []
    monkeypatch.setattr(websearch, "fetch", lambda url: asked.append(url) or _PAGE_TEXT)
    assert broker.fetch(_PAGE_URL) == _PAGE_TEXT
    assert asked == [_PAGE_URL]


def test_a_broker_with_no_reading_endpoint_says_which_setting_to_set(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "")
    websearch._remember_url([{"url": _PAGE_URL}])
    reply = broker.answer({"id": 1, "method": "fetch",
                           "params": {"url": _PAGE_URL}}, broker._Budget())
    assert "WEB_FETCH_PROVIDER" in reply["error"]["message"]


def test_a_page_read_that_failed_is_still_one_reply(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    monkeypatch.setattr(websearch, "fetch",
                        _raises(RuntimeError("the page request failed")))
    reply = broker.answer({"id": 1, "method": "fetch",
                           "params": {"url": _PAGE_URL}}, broker._Budget())
    assert reply["error"]["data"]["error_type"] == "RuntimeError"
    assert "result" not in reply


def test_a_page_read_past_the_limit_is_refused_however_it_was_sent(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    monkeypatch.setattr(websearch, "fetch", lambda url: _PAGE_TEXT)
    budget = broker._Budget()
    assert "result" in broker.answer({"id": 1, "method": "fetch",
                                      "params": {"url": _PAGE_URL}}, budget)
    second = broker.answer({"id": 2, "method": "fetch",
                            "params": {"url": _PAGE_URL}}, budget)
    assert "WEB_FETCH_MAX_CALL" in second["error"]["message"]


def test_a_page_read_is_refused_when_the_tool_is_off(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 0)
    monkeypatch.setattr(websearch, "fetch",
                        lambda url: pytest.fail("it left the machine"))
    reply = broker.answer({"id": 1, "method": "fetch",
                           "params": {"url": _PAGE_URL}}, broker._Budget())
    assert "WEB_FETCH_MAX_CALL" in reply["error"]["message"]


def test_the_reads_and_the_searches_are_counted_apart(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 1)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    monkeypatch.setattr(websearch, "search", lambda q: list(_RESULTS))
    monkeypatch.setattr(websearch, "fetch", lambda url: _PAGE_TEXT)
    budget = broker._Budget()
    assert "result" in broker.answer({"id": 1, "method": "search",
                                      "params": {"question": "one"}}, budget)
    assert "result" in broker.answer({"id": 2, "method": "fetch",
                                      "params": {"url": _PAGE_URL}}, budget)


def test_what_the_page_read_is_authorised_with_does_not_reach_the_caller(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    monkeypatch.setattr(config, "WEB_FETCH_API_KEY", "jina_the_key")
    monkeypatch.setattr(config, "WEB_FETCH_HEADER", {"X-Key": "Token jina_the_key"})
    monkeypatch.setattr(websearch, "fetch",
                        _raises(RuntimeError("401 with X-Key: Token jina_the_key")))
    reply = broker.answer({"id": 1, "method": "fetch",
                           "params": {"url": _PAGE_URL}}, broker._Budget())
    assert "jina_the_key" not in json.dumps(reply)
    assert "Token" not in json.dumps(reply)


def test_the_fetch_method_is_named_with_the_others_when_one_is_not_known(fake):
    reply = broker.answer({"id": 1, "method": "shell", "params": {}},
                          broker._Budget())
    assert broker.FETCH in reply["error"]["message"]



def test_the_broker_reads_only_what_its_own_searches_returned(monkeypatch):
    """A container that went round the tool gets no URL of its own out through the
    page-reading API."""
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/")
    monkeypatch.setattr(websearch, "_open",
                        lambda *a, **k: pytest.fail("it left the machine"))
    reply = broker.answer({"id": 1, "method": "fetch",
                           "params": {"url": "https://attacker.invalid/?d=x"}},
                          broker._Budget())
    assert "web_search" in reply["error"]["message"]


# == Making a call again =======================================

def _busy(status_class, retry_after=None):
    """Return a litellm error of the class, carrying a retry-after when one is given."""
    error = status_class(message="busy", llm_provider="anthropic", model="m")
    error.litellm_response_headers = {"retry-after": retry_after} if retry_after else {}
    return error


@pytest.mark.parametrize("error_class", [litellm.RateLimitError,
                                         litellm.InternalServerError,
                                         litellm.ServiceUnavailableError])
def test_a_busy_provider_is_waited_for_and_asked_again(monkeypatch, error_class):
    answer_list = [_busy(error_class, "7"), _busy(error_class), "ok"]
    call_list, sleep_list = [], []

    def completion(**kwargs):
        call_list.append(kwargs)
        answer = answer_list.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(providercall.litellm, "completion", completion)
    monkeypatch.setattr(providercall.time, "sleep", sleep_list.append)
    monkeypatch.setattr(config, "LLM_RETRY_WAIT_SEC", 5)
    monkeypatch.setattr(config, "LLM_CALL_TIMEOUT_SEC", 600)
    assert providercall.completion({"model": "m", "num_retries": 3}) == "ok"
    # The provider's retry-after when longer, then twice the first wait
    assert sleep_list == [7.0, 10]
    assert all(kwargs["num_retries"] == 0 for kwargs in call_list) and len(call_list) == 3


def test_the_last_retry_raises_and_a_wait_is_never_over_one_attempt(monkeypatch):
    sleep_list = []
    monkeypatch.setattr(providercall.litellm, "completion",
                        lambda **kwargs: (_ for _ in ()).throw(
                            _busy(litellm.RateLimitError, "9999")))
    monkeypatch.setattr(providercall.time, "sleep", sleep_list.append)
    monkeypatch.setattr(config, "LLM_CALL_TIMEOUT_SEC", 60)
    with pytest.raises(litellm.RateLimitError):
        providercall.completion({"model": "m", "num_retries": 2})
    assert sleep_list == [60, 60]


@pytest.mark.parametrize("status", [502, 529, 500])
def test_a_5xx_of_any_class_is_asked_again(monkeypatch, status):
    error_list = [
        litellm.BadGatewayError(message="bad gateway", model="m", llm_provider="openai"),
        litellm.APIError(status_code=status, message="overloaded", model="m",
                         llm_provider="openai")]
    call_list = []

    def completion(**kwargs):
        call_list.append(kwargs)
        if len(call_list) <= len(error_list):
            raise error_list[len(call_list) - 1]
        return "ok"

    monkeypatch.setattr(providercall.litellm, "completion", completion)
    monkeypatch.setattr(providercall.time, "sleep", lambda wait: None)
    assert providercall.completion({"model": "m", "num_retries": 3}) == "ok"
    assert len(call_list) == 3


def test_a_status_under_500_other_than_429_is_not_asked_again(monkeypatch):
    call_list = []

    def completion(**kwargs):
        call_list.append(kwargs)
        raise litellm.APIError(status_code=409, message="conflict", model="m",
                               llm_provider="openai")

    monkeypatch.setattr(providercall.litellm, "completion", completion)
    with pytest.raises(litellm.APIError):
        providercall.completion({"model": "m", "num_retries": 3})
    assert len(call_list) == 1


def test_a_refusal_is_not_asked_again(monkeypatch):
    call_list = []

    def completion(**kwargs):
        call_list.append(kwargs)
        raise litellm.BadRequestError(message="effort is not supported", model="m",
                                      llm_provider="anthropic")

    monkeypatch.setattr(providercall.litellm, "completion", completion)
    with pytest.raises(litellm.BadRequestError):
        providercall.completion({"model": "m", "num_retries": 3})
    assert len(call_list) == 1


def test_the_broker_asks_a_model_that_always_thinks_for_the_least_thinking(fake, monkeypatch):
    monkeypatch.setattr(providercall, "always_thinks", lambda model: True)
    broker._call_the_provider({"model": "m", "messages": [],
                               "thinking": {"type": "disabled"}})
    assert fake.kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fake.kwargs["output_config"] == {"effort": "low"}
