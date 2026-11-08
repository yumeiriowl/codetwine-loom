"""The web question (tools/web.py) and the search behind it (websearch.py).

The search is one HTTP request; websearch._open is substituted for these checks and nothing here
reaches a network. No model takes part in a search, so none is substituted either. The
clock and the sleep the pacing uses are substituted too, so no check here waits.
"""

import email.message
import http.client
import http.server
import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from codetwine_loom import config, websearch
from codetwine_loom.rlm import CUT_END_NOTE
from codetwine_loom.tools import web

RESULTS = [{"title": "Changelog", "url": "https://example.invalid/changelog",
            "content": "Version 2 renamed it."},
           {"title": "Guide", "url": "https://example.invalid/guide",
            "content": "How to use it now."}]


@pytest.fixture
def on(monkeypatch):
    """Switch the tool on with a budget of 2 calls and a stand-in for the search."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(web, "_search", lambda question: list(RESULTS))


def _clock(monkeypatch, start=1000.0):
    """Stand in for the clock and the sleep the pacing reads and calls.

    Args:
        monkeypatch: pytest's fixture.
        start: What the clock reads first.

    Returns:
        The list each sleep's length is recorded in. A sleep moves the clock on by
        that much.
    """
    now = [start]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(websearch.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(websearch.time, "sleep", sleep)
    return slept


class _Reply:
    """What litellm.completion returns, cut to what the search reads."""

    def __init__(self, answer, extra=None):
        self.answer = answer
        self.extra = extra or {}

    @property
    def choices(self):
        message = type("Message", (), {"content": self.answer})()
        return [type("Choice", (), {"message": message})()]

    def model_dump(self):
        return {"choices": [{"message": {"content": self.answer, **self.extra}}]}


def _model(monkeypatch, reply):
    """Stand in for the model's provider search, so nothing here reaches a network.

    Args:
        monkeypatch: pytest's fixture.
        reply: The _Reply to return, or an exception to raise instead.

    Returns:
        The list each call's keyword arguments are recorded in.
    """
    monkeypatch.setattr(config, "WEB_SEARCH_PROVIDER", config.WEB_SEARCH_BY_THE_MODEL)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "")
    called = []

    def completion(**kwargs):
        called.append(kwargs)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(websearch.providercall.litellm, "completion", completion)
    return called


# What one provider puts in a reply: the pages the model cited, with the text it
# quoted, and the whole list it searched through
_CITED = {"provider_specific_fields": {
    "citations": [[{"type": "web_search_result_location", "url": "https://a.invalid/",
                    "title": "A", "cited_text": "what A says"}]],
    "web_search_results": [{"content": [
        {"type": "web_search_result", "title": "B", "url": "https://b.invalid/",
         "encrypted_content": "..."},
        {"type": "web_search_result", "title": "A again", "url": "https://a.invalid/",
         "encrypted_content": "..."}]}]}}


def _slow_down(code=429, retry_after=None):
    """Build the HTTP answer an endpoint says "not now" with.

    Args:
        code: The status, 429 or 503.
        retry_after: The Retry-After header's value, or None for none.

    Returns:
        The HTTPError urlopen raises for it.
    """
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("https://search.invalid/search", code, "not now",
                                  headers, None)


def _endpoint(monkeypatch, reply, status=200):
    """Point the search at an endpoint and stand in for the request.

    Args:
        monkeypatch: pytest's fixture.
        reply: What the endpoint returns, as an object to serialise, or bytes as they
            are.
        status: Unused by the stand-in; an error is raised by passing an exception as
            reply instead.

    Returns:
        The list each request's URL is recorded in.
    """
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    asked = []

    def urlopen(request, timeout=None):
        asked.append((request.full_url, dict(request.headers)))
        if isinstance(reply, Exception):
            raise reply
        body = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
        return io.BytesIO(body)

    monkeypatch.setattr(websearch, "_open", urlopen)
    return asked


# == what comes back ==========================================

def test_the_results_come_back_with_their_4_keys(on):
    out = web.web_search("what does that interface look like now")
    assert out["ok"] is True
    assert out["results"] == RESULTS
    assert out["question"] == "what does that interface look like now"
    assert out["calls_left"] == 1


def test_the_results_are_cut_to_the_limit(monkeypatch, on):
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_MAX_CHAR", 120)
    out = web.web_search("what changed")
    assert out["ok"] is True
    assert len(out["results"]) == 1 and out["truncated"] is True


def test_the_answer_says_when_no_result_was_dropped(monkeypatch, on):
    out = web.web_search("what changed")
    assert out["ok"] is True and out["truncated"] is False


def test_the_first_result_comes_back_even_when_it_alone_is_over_the_limit(monkeypatch,
                                                                         on):
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_MAX_CHAR", 5)
    assert len(web.web_search("what changed")["results"]) == 1


def test_a_limit_of_0_keeps_every_result(monkeypatch, on):
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_MAX_CHAR", 0)
    out = web.web_search("what changed")
    assert out["results"] == RESULTS and out["truncated"] is False


def test_a_search_that_found_nothing_is_a_refusal_not_an_empty_answer(monkeypatch, on):
    monkeypatch.setattr(web, "_search", lambda question: [])
    out = web.web_search("what changed")
    assert out["ok"] is False and "found nothing" in out["error"]


# == the budget ===============================================

def test_the_budget_is_spent_and_then_refused(on):
    assert web.web_search("one")["calls_left"] == 1
    assert web.web_search("two")["calls_left"] == 0
    out = web.web_search("three")
    assert out["ok"] is False and "call limit" in out["error"]


def test_the_counter_is_reset_between_requests(on):
    web.web_search("one")
    assert web.web_search_count() == 1
    web.reset_web_search_count()
    assert web.web_search_count() == 0


def test_the_tool_refuses_when_it_is_switched_off(monkeypatch, on):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    out = web.web_search("what changed")
    assert out["ok"] is False and "turned off" in out["error"]


def test_no_endpoint_is_the_tool_being_off(monkeypatch):
    """A budget with no endpoint behind it would hand over a tool whose every call can
    only be refused."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 3)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "")
    assert websearch.can_search() is False
    assert web.available() is False
    out = web.web_search("what changed")
    assert out["ok"] is False and "turned off" in out["error"]


def test_a_question_over_the_limit_is_refused_without_spending_a_call(monkeypatch, on):
    monkeypatch.setattr(config, "WEB_SEARCH_QUESTION_MAX_CHAR", 10)
    out = web.web_search("x" * 11)
    assert out["ok"] is False and out["calls_left"] == 2


def test_an_empty_question_raises(on):
    with pytest.raises(ValueError, match="empty question"):
        web.web_search("   ")


def test_a_search_that_did_not_run_is_a_refusal_not_an_answer(monkeypatch, caplog, on):

    def raising(question):
        raise RuntimeError("the search request failed: no route")

    monkeypatch.setattr(web, "_search", raising)
    out = web.web_search("what changed")
    assert out["ok"] is False and "RuntimeError" in out["error"]
    assert "the search did not run" in caplog.text


def test_a_stage_without_run_project_is_not_sent_to_it_by_a_search_that_did_not_run(
        monkeypatch, on):
    import inspect

    def raising(question):
        raise RuntimeError("the search request failed: no route")

    monkeypatch.setattr(web, "_search", raising)
    assert web.search_tool(True) is web.web_search
    assert web.web_search("what changed")["error"] == (
        "the search did not run (RuntimeError). Settle this from the repository or "
        "from run_project instead")
    no_run = web.search_tool(False)
    assert (no_run.__name__, no_run.__doc__) == ("web_search", web.web_search.__doc__)
    assert inspect.signature(no_run) == inspect.signature(web.web_search)
    assert no_run("what changed") == {
        "ok": False, "calls_left": 0,
        "error": "the search did not run (RuntimeError). Settle this from the "
                 "repository instead"}
    assert web.web_search_count() == 2


# == where the search is made =================================

def test_the_search_is_made_here_when_no_broker_is_set(monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "")
    made = []
    monkeypatch.setattr(websearch, "search", lambda q: made.append(q) or list(RESULTS))
    assert web._search("what changed") == RESULTS
    assert made == ["what changed"]


def test_the_search_goes_to_the_broker_when_one_is_set(monkeypatch):
    """The container has no network, so the process that holds it makes the request."""
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(websearch, "search",
                        lambda q: pytest.fail("the container must not search itself"))
    asked = []
    monkeypatch.setattr(web.broker, "search",
                        lambda q: asked.append(q) or list(RESULTS))
    assert web._search("what changed") == RESULTS
    assert asked == ["what changed"]


# == the request itself =======================================

def test_the_question_and_the_settings_build_the_url(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_PARAM", {"format": "json"})
    monkeypatch.setattr(config, "WEB_SEARCH_QUERY_PARAM", "query")
    asked = _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("what changed")
    url = asked[0][0]
    assert url.startswith("https://search.invalid/search?")
    assert "format=json" in url and "query=what+changed" in url


def test_a_post_sends_the_question_and_the_parameters_as_json(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_METHOD", "POST")
    monkeypatch.setattr(config, "WEB_SEARCH_QUERY_PARAM", "query")
    monkeypatch.setattr(config, "WEB_SEARCH_PARAM", {"max_results": 5})
    sent = []
    monkeypatch.setattr(websearch, "_open",
                        lambda request, timeout=None: sent.append(request) or
                        io.BytesIO(json.dumps({"results": RESULTS}).encode()))
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    websearch.search("what changed")
    request = sent[0]
    assert request.get_method() == "POST"
    assert request.full_url == "https://search.invalid/search"
    assert json.loads(request.data) == {"max_results": 5, "query": "what changed"}
    assert request.headers.get("Content-type") == "application/json"
    assert request.headers.get("User-agent", "").startswith("codetwine-loom/")


def test_the_headers_are_sent_as_they_are_set(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_HEADER", {"X-Token": "a-key"})
    asked = _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("what changed")
    assert asked[0][1].get("X-token") == "a-key"


def test_the_request_names_this_package(monkeypatch):
    asked = _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("what changed")
    assert asked[0][1].get("User-agent", "").startswith("codetwine-loom/")


def test_a_user_agent_that_is_set_is_sent_instead(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_HEADER",
                        {"User-Agent": "my-run/1 (ops@example.invalid)"})
    asked = _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("what changed")
    assert asked[0][1].get("User-agent") == "my-run/1 (ops@example.invalid)"


def test_the_results_are_found_where_the_setting_points(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_PATH", "web.results")
    _endpoint(monkeypatch, {"web": {"results": RESULTS}})
    assert websearch.search("what changed") == RESULTS


def test_a_reply_that_is_itself_the_list_needs_no_path(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_PATH", "")
    _endpoint(monkeypatch, RESULTS)
    assert websearch.search("what changed") == RESULTS


def test_only_the_keys_that_are_kept_come_back(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_FIELD", "url,title")
    _endpoint(monkeypatch, {"results": [{"title": "T", "url": "U", "junk": "J"}]})
    assert websearch.search("what changed") == [{"url": "U", "title": "T"}]


def test_half_of_a_surrogate_pair_in_a_result_comes_back_as_the_no_character_mark(
        monkeypatch):
    _endpoint(monkeypatch, b'{"results": [{"title": "cut \\ud83d", "url": "U", '
                           b'"content": "whole \\ud83d\\ude00 and half \\ude00"}]}')
    result_list = websearch.search("what changed")
    assert result_list == [{"title": "cut \ufffd", "url": "U",
                            "content": "whole \U0001f600 and half \ufffd"}]
    json.dumps(result_list, ensure_ascii=False).encode("utf-8")


def test_half_of_a_surrogate_pair_in_a_page_comes_back_as_the_no_character_mark(
        monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "data")
    monkeypatch.setattr(config, "WEB_FETCH_CONTENT_FIELD", "content")
    _reader(monkeypatch, b'{"data": {"content": "cut \\ud83d here"}}')
    assert websearch.fetch(PAGE_URL) == "cut \ufffd here"


def test_half_of_a_surrogate_pair_in_the_models_search_comes_back_as_the_mark_too(
        monkeypatch):
    _clock(monkeypatch)
    cited = {"citations": [{"url": "https://a.invalid/", "title": "T \udc00",
                            "cited_text": "half \ud83d here"}]}
    _model(monkeypatch, _Reply("answer \ud83d tail", cited))
    result_list = websearch.search("what changed")
    assert result_list == [
        {"title": "", "url": "", "content": "answer \ufffd tail"},
        {"title": "T \ufffd", "url": "https://a.invalid/", "content": "half \ufffd here"}]
    json.dumps(result_list, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("url", ["http://[::1", "http://ex\uff0fample.invalid/",
                                 "http://[not-an-address]/"])
def test_a_result_whose_url_cannot_be_read_is_left_out_and_the_rest_come_back(
        monkeypatch, url):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    _clock(monkeypatch)
    _endpoint(monkeypatch, {"results": [RESULTS[0], {"title": "t", "url": url,
                                                     "content": "c"}, RESULTS[1]]})
    out = web.web_search("what changed")
    assert out["ok"] is True and out["results"] == RESULTS
    assert websearch.url_key(url) == "" and url not in str(websearch._result_url_dict)
    # The model's own search leaves the page out the same way
    _model(monkeypatch, _Reply("an answer", {"citations": [
        {"url": url, "title": "t"}, {"url": "https://a.invalid/", "title": "A"}]}))
    assert [r["url"] for r in websearch.search("what changed")] == [
        "", "https://a.invalid/"]


def test_a_result_missing_a_kept_key_keeps_the_rest(monkeypatch):
    _endpoint(monkeypatch, {"results": [{"title": "T", "url": "U"}]})
    assert websearch.search("what changed") == [{"title": "T", "url": "U"}]


def test_no_more_results_than_asked_for(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_RESULT", 1)
    _endpoint(monkeypatch, {"results": RESULTS})
    assert len(websearch.search("what changed")) == 1


def test_no_endpoint_says_which_setting_to_set(monkeypatch):
    """A caller that went round the tool is stopped here, before anything is sent."""
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "")
    with pytest.raises(ValueError, match="WEB_SEARCH_PROVIDER"):
        websearch.search("what changed")


def test_a_question_over_the_limit_does_not_leave_this_machine(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_QUESTION_MAX_CHAR", 10)
    asked = _endpoint(monkeypatch, {"results": RESULTS})
    with pytest.raises(ValueError, match="at most 10"):
        websearch.search("x" * 11)
    assert asked == []


def test_a_request_that_failed_says_so(monkeypatch):
    _endpoint(monkeypatch, urllib.error.URLError("no route"))
    with pytest.raises(RuntimeError, match="the search request failed"):
        websearch.search("what changed")


@pytest.mark.parametrize("error", [http.client.IncompleteRead(b"12345"),
                                   http.client.BadStatusLine("NOT-HTTP AT ALL"),
                                   http.client.LineTooLong("header line")])
def test_an_answer_that_cannot_be_read_as_http_is_a_request_that_failed(monkeypatch,
                                                                        error):
    _endpoint(monkeypatch, error)
    with pytest.raises(RuntimeError, match="the search request failed: "
                                           + type(error).__name__):
        websearch.search("what changed")
    _reader(monkeypatch, error)
    with pytest.raises(RuntimeError, match="the page request failed"):
        websearch.fetch(PAGE_URL)


def test_the_tools_answer_ok_false_for_an_answer_that_cannot_be_read_as_http(
        monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "")
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 2)
    _endpoint(monkeypatch, http.client.IncompleteRead(b"12345"))
    out = web.web_search("what changed")
    assert out["ok"] is False and out["calls_left"] == 1
    _reader(monkeypatch, http.client.IncompleteRead(b"12345"))
    out = web.web_fetch(PAGE_URL)
    assert out["ok"] is False and out["calls_left"] == 1


def test_a_reply_that_is_not_json_says_so(monkeypatch):
    _endpoint(monkeypatch, b"<html>not json</html>")
    with pytest.raises(RuntimeError, match="not JSON"):
        websearch.search("what changed")


def test_a_reply_with_no_list_where_the_path_points_says_which_setting(monkeypatch):
    _endpoint(monkeypatch, {"items": RESULTS})
    with pytest.raises(ValueError, match="holds no list at 'results'"):
        websearch.search("what changed")


# == the pace =================================================

def test_an_interval_over_the_timeout_is_waited_rather_than_refused(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 40.0)
    monkeypatch.setattr(config, "WEB_SEARCH_TIMEOUT_SEC", 30)
    slept = _clock(monkeypatch)
    asked = _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("one")
    assert websearch.search("two") == RESULTS
    assert slept == [40.0] and len(asked) == 2


def test_a_wait_the_endpoint_asked_for_is_refused_under_a_long_interval_too(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 40.0)
    monkeypatch.setattr(config, "WEB_SEARCH_TIMEOUT_SEC", 30)
    slept = _clock(monkeypatch)
    _endpoint(monkeypatch, _slow_down(429, retry_after=45))
    with pytest.raises(RuntimeError, match="slow down"):
        websearch.search("one")
    asked = _endpoint(monkeypatch, {"results": RESULTS})
    with pytest.raises(RuntimeError, match="asked for a wait of 45 s"):
        websearch.search("two")
    assert asked == [] and slept == []
    websearch.time.sleep(20.0)
    slept.clear()
    websearch.search("three")
    assert slept == [25.0] and len(asked) == 1


def test_two_searches_are_the_interval_apart(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 2.0)
    slept = _clock(monkeypatch)
    asked = _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("one")
    websearch.search("two")
    assert slept == [2.0] and len(asked) == 2


def test_searches_made_at_once_go_out_the_interval_apart(monkeypatch):
    """The clock is the real one here: three threads ask at the same moment."""
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 0.2)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    sent_list = []

    def urlopen(request, timeout=None):
        assert websearch._turn_lock_dict["WEB_SEARCH"].locked()
        sent_list.append(time.monotonic())
        return io.BytesIO(json.dumps({"results": RESULTS}).encode())

    monkeypatch.setattr(websearch, "_open", urlopen)
    start = threading.Barrier(3)

    def ask(number):
        start.wait()
        websearch.search(f"question {number}")

    thread_list = [threading.Thread(target=ask, args=(number,)) for number in range(3)]
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
    assert len(sent_list) == 3
    assert all(later - earlier >= 0.19 for earlier, later in zip(sent_list, sent_list[1:]))


def test_a_search_that_waits_behind_one_told_to_slow_down_waits_that_long_too(
        monkeypatch):
    """The clock is the real one here: the second thread asks while the first
    request is out."""
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(websearch, "_retry_after", lambda error: 0.3)
    is_out, may_answer = threading.Event(), threading.Event()
    moment_dict = {}

    def urlopen(request, timeout=None):
        if not is_out.is_set():
            is_out.set()
            may_answer.wait(5)
            moment_dict["told"] = time.monotonic()
            raise _slow_down(429, retry_after=1)
        moment_dict["sent"] = time.monotonic()
        return io.BytesIO(json.dumps({"results": RESULTS}).encode())

    monkeypatch.setattr(websearch, "_open", urlopen)
    outcome_list = []

    def ask(question):
        try:
            outcome_list.append(websearch.search(question))
        except RuntimeError as e:
            outcome_list.append(str(e))

    first = threading.Thread(target=ask, args=("one",))
    first.start()
    assert is_out.wait(5)
    second = threading.Thread(target=ask, args=("two",))
    second.start()
    time.sleep(0.05)
    may_answer.set()
    first.join()
    second.join()
    assert outcome_list == ["the endpoint asked to slow down (HTTP 429)", RESULTS]
    assert moment_dict["sent"] - moment_dict["told"] >= 0.29


def test_a_search_after_the_interval_does_not_wait(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 2.0)
    slept = _clock(monkeypatch)
    _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("one")
    websearch.time.sleep(3.0)
    slept.clear()
    websearch.search("two")
    assert slept == []


def test_a_slow_down_answer_is_one_refusal_and_its_wait_is_kept(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 1.0)
    slept = _clock(monkeypatch)
    _endpoint(monkeypatch, _slow_down(429, retry_after=5))
    with pytest.raises(RuntimeError, match="slow down"):
        websearch.search("one")
    _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("two")
    assert slept == [5.0]


def test_a_503_is_a_slow_down_answer_too(monkeypatch):
    _clock(monkeypatch)
    _endpoint(monkeypatch, _slow_down(503, retry_after=2))
    with pytest.raises(RuntimeError, match="slow down"):
        websearch.search("one")


def test_a_slow_down_answer_without_a_retry_after_keeps_the_interval(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 1.0)
    slept = _clock(monkeypatch)
    _endpoint(monkeypatch, _slow_down(429))
    with pytest.raises(RuntimeError, match="slow down"):
        websearch.search("one")
    _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("two")
    assert slept == [1.0]


@pytest.mark.parametrize("value", ["\u00b2", "\u2460", "soon", "-1", "1.5"])
def test_a_retry_after_that_is_no_whole_number_of_seconds_reads_as_none(value):
    assert websearch._retry_after(_slow_down(429, retry_after=value)) == 0.0


def test_a_wait_over_the_timeout_is_refused_without_sending(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_TIMEOUT_SEC", 30)
    slept = _clock(monkeypatch)
    _endpoint(monkeypatch, _slow_down(429, retry_after=3600))
    with pytest.raises(RuntimeError, match="slow down"):
        websearch.search("one")
    asked = _endpoint(monkeypatch, {"results": RESULTS})
    with pytest.raises(RuntimeError, match="WEB_SEARCH_TIMEOUT_SEC"):
        websearch.search("two")
    assert asked == [] and slept == []


def test_a_failed_request_that_is_not_a_slow_down_keeps_the_interval(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 1.0)
    slept = _clock(monkeypatch)
    _endpoint(monkeypatch, _slow_down(500))
    with pytest.raises(RuntimeError, match="the search request failed"):
        websearch.search("one")
    _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("two")
    assert slept == [1.0]


# == the model's own provider search ==========================

def test_the_model_provider_search_is_a_way_to_search(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 3)
    _model(monkeypatch, _Reply("an answer"))
    assert websearch.can_search() is True
    assert web.available() is True


def test_the_answer_comes_first_and_the_cited_pages_follow(monkeypatch):
    _clock(monkeypatch)
    _model(monkeypatch, _Reply("  the answer  ", _CITED))
    assert websearch.search("what changed") == [
        {"title": "", "url": "", "content": "the answer"},
        {"title": "A", "url": "https://a.invalid/", "content": "what A says"},
        {"title": "B", "url": "https://b.invalid/", "content": ""}]


def test_the_models_answer_says_when_it_was_cut_at_the_output_limit(monkeypatch):
    class _CutReply(_Reply):
        @property
        def choices(self):
            message = type("Message", (), {"content": self.answer})()
            return [type("Choice", (), {"message": message, "finish_reason": "length"})()]

    _clock(monkeypatch)
    _model(monkeypatch, _CutReply("the first half of the answer", _CITED))
    assert websearch.search("what changed")[0]["content"] == \
        "the first half of the answer" + CUT_END_NOTE


def test_the_call_carries_the_model_the_credentials_and_the_search(monkeypatch):
    _clock(monkeypatch)
    monkeypatch.setattr(config, "WEB_SEARCH_MODEL", "provider/searcher")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-1")
    called = _model(monkeypatch, _Reply("an answer", _CITED))
    websearch.search("what changed")
    assert called[0]["model"] == "provider/searcher"
    assert called[0]["api_key"] == "sk-1"
    assert called[0]["web_search_options"]
    assert called[0]["messages"] == [{"role": "user", "content": "what changed"}]


def test_the_search_asks_for_no_more_output_than_the_model_takes(monkeypatch):
    _clock(monkeypatch)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_TOKEN", 32768)
    monkeypatch.setattr(websearch.providercall.litellm, "get_model_info",
                        lambda model, **kwarg_dict: {"max_output_tokens": 16384})
    called = _model(monkeypatch, _Reply("an answer", _CITED))
    websearch.search("what changed")
    assert called[0]["max_tokens"] == 16384


@pytest.mark.parametrize("answer", [None, "From memory: it is called foo()."])
def test_a_reply_that_names_no_page_is_a_failed_search(monkeypatch, on, answer):
    _clock(monkeypatch)
    _model(monkeypatch, _Reply(answer))
    with pytest.raises(RuntimeError, match="names no page"):
        websearch.search("what changed")
    monkeypatch.setattr(web, "_search", websearch.search)
    out = web.web_search("what changed")
    assert out["ok"] is False and "the search did not run (RuntimeError)" in out["error"]
    assert "results" not in out


def test_the_models_results_are_cut_to_the_limit_too(monkeypatch):
    _clock(monkeypatch)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_RESULT", 2)
    _model(monkeypatch, _Reply("the answer", _CITED))
    assert len(websearch.search("what changed")) == 2


def test_the_models_answer_and_cited_text_are_cut_to_their_share_of_the_limit(monkeypatch):
    _clock(monkeypatch)
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_MAX_CHAR", 100)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_RESULT", 4)
    cited = {"citations": [{"url": "https://a.invalid/", "title": "A",
                            "cited_text": "y" * 500}]}
    _model(monkeypatch, _Reply("x" * 500, cited))
    result_list = websearch.search("what changed")
    assert result_list[0]["content"].startswith("x" * 25 + "\n... (475 chars omitted)")
    assert result_list[1]["content"].startswith("y" * 25 + "\n... (475 chars omitted)")


def test_the_models_whole_result_list_stays_near_the_limit(monkeypatch, on):
    _clock(monkeypatch)
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_MAX_CHAR", 1000)
    monkeypatch.setattr(web, "_search", websearch.search)
    _model(monkeypatch, _Reply("x" * 50000, _CITED))
    out = web.web_search("what changed")
    assert len(json.dumps(out["results"], ensure_ascii=False)) < 1000


def test_a_limit_of_0_cuts_no_result(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_MAX_CHAR", 0)
    _endpoint(monkeypatch, {"results": [{"title": "T", "url": "U", "content": "x" * 500}
                                        for _ in range(3)]})
    assert [r["content"] for r in websearch.search("what changed")] == ["x" * 500] * 3


def test_a_limit_under_the_result_count_still_cuts_each_value(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_MAX_CHAR", 5)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_RESULT", 10)
    _endpoint(monkeypatch, {"results": [{"title": "T", "url": "U", "content": "x" * 500}]})
    assert websearch.search("what changed")[0]["content"].startswith("x\n... (499 chars")


def test_a_provider_without_a_search_is_one_failed_search(monkeypatch):
    _clock(monkeypatch)
    _model(monkeypatch, ValueError("web_search_options is not supported"))
    with pytest.raises(RuntimeError, match="the model's search failed"):
        websearch.search("what changed")


def test_the_models_searches_keep_the_interval_too(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 2.0)
    slept = _clock(monkeypatch)
    _model(monkeypatch, _Reply("an answer", _CITED))
    websearch.search("one")
    websearch.search("two")
    assert slept == [2.0]


def test_a_page_named_with_uri_and_a_page_without_a_title_are_told_apart():
    pages = websearch._page_list_in({"grounding": [{"web": {"uri": "https://g.invalid/",
                                                        "title": "G"}},
                                               {"url": "https://no-title.invalid/"}]})
    assert pages == [{"title": "G", "url": "https://g.invalid/", "content": ""}]


# == what the search refuses before it sends anything =========
def test_an_empty_question_is_refused_with_what_to_pass():
    assert "the question is empty" in websearch.check_question("  ")


def test_a_question_within_the_limit_is_not_refused():
    assert websearch.check_question("what changed") == ""


def test_a_path_that_runs_through_something_that_is_not_an_object_finds_nothing():
    assert websearch._at_path({"web": [1, 2]}, "web.results") is None
    assert websearch._at_path({"web": {"results": [1]}}, "web.results") == [1]


def test_a_reply_over_the_size_limit_is_refused_unread(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_BYTE", 8)
    _endpoint(monkeypatch, b'{"results": []}')
    with pytest.raises(RuntimeError, match="more than 8 bytes"):
        websearch.search("what changed")


# == reading one page (web_fetch) ==============================
# The read is one HTTP request too, and the page comes back whole from websearch and cut
# from the tool

PAGE_TEXT = "# Guide\n\nHow to use it now, in full."
PAGE_URL = "https://example.invalid/guide"


def _found(url=PAGE_URL):
    """Record url as one a web_search of this request returned, at the tool and at the
    process that reads."""
    web._result_url_set.add(websearch.url_key(url))
    websearch._remember_url([{"url": url}])


@pytest.fixture
def fetch_on(monkeypatch):
    """Switch web_fetch on with a budget of 2 calls and a stand-in for the read, with
    web_search on beside it and PAGE_URL among what it returned."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    monkeypatch.setattr(web, "_fetch", lambda url: PAGE_TEXT)
    _found()


def _reader(monkeypatch, reply):
    """Point the read at an endpoint and stand in for the request.

    Args:
        monkeypatch: pytest's fixture.
        reply: What the endpoint returns, as an object to serialise, or bytes as they
            are, or an exception to raise.

    Returns:
        The list each request is recorded in, as (URL, headers, body).
    """
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    _found()
    asked = []

    def urlopen(request, timeout=None):
        asked.append((request.full_url, dict(request.headers), request.data))
        if isinstance(reply, Exception):
            raise reply
        body = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
        return io.BytesIO(body)

    monkeypatch.setattr(websearch, "_open", urlopen)
    return asked


def test_the_page_comes_back_with_its_5_keys(fetch_on):
    out = web.web_fetch(PAGE_URL)
    assert out == {"ok": True, "url": PAGE_URL, "content": PAGE_TEXT,
                   "truncated": False, "calls_left": 1}


def test_the_page_is_cut_to_the_limit_with_the_marker(monkeypatch, fetch_on):
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_MAX_CHAR", 10)
    out = web.web_fetch(PAGE_URL)
    assert out["ok"] is True and out["truncated"] is True
    assert out["content"].startswith(PAGE_TEXT[:10]) and "chars omitted" in out["content"]


def test_a_read_limit_of_0_keeps_the_whole_page_and_says_nothing_was_cut(monkeypatch,
                                                                          fetch_on):
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_MAX_CHAR", 0)
    out = web.web_fetch(PAGE_URL)
    assert (out["content"], out["truncated"]) == (PAGE_TEXT, False)


def test_the_read_budget_is_spent_and_then_refused(fetch_on):
    assert web.web_fetch(PAGE_URL)["calls_left"] == 1
    assert web.web_fetch(PAGE_URL)["calls_left"] == 0
    out = web.web_fetch(PAGE_URL)
    assert out["ok"] is False and "call limit" in out["error"]


def test_the_read_counter_is_reset_between_requests(fetch_on):
    web.web_fetch(PAGE_URL)
    assert web.fetch_call_remain_count() == 1
    web.reset_web_fetch_count()
    assert web.fetch_call_remain_count() == 2


def test_the_read_and_the_search_have_budgets_of_their_own(on, fetch_on):
    web.web_search("one")
    web.web_search("two")
    assert web.web_search("three")["ok"] is False
    assert web.web_fetch(PAGE_URL)["ok"] is True


def test_the_read_tool_refuses_when_it_is_switched_off(monkeypatch, fetch_on):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 0)
    out = web.web_fetch(PAGE_URL)
    assert out["ok"] is False and "turned off" in out["error"]


def test_the_read_tool_is_off_when_the_search_is_switched_off(monkeypatch, fetch_on):
    assert web.available() and web.fetch_available()
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    assert websearch.can_fetch() and config.WEB_FETCH_MAX_CALL > 0
    assert web.fetch_available() is False
    out = web.web_fetch(PAGE_URL)
    assert out["ok"] is False and "turned off" in out["error"]


def test_no_reading_endpoint_is_the_tool_being_off(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 3)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "")
    assert websearch.can_fetch() is False
    assert web.fetch_available() is False
    out = web.web_fetch(PAGE_URL)
    assert out["ok"] is False and "turned off" in out["error"]


@pytest.mark.parametrize("url", ["what does that interface look like now",
                                 "ftp://example.invalid/guide",
                                 "file:///etc/passwd",
                                 "https://",
                                 "https://example.invalid/a b"])
def test_what_is_not_a_page_s_url_is_refused_without_spending_a_call(fetch_on, url):
    out = web.web_fetch(url)
    assert out["ok"] is False and "web_search" in out["error"]
    assert out["calls_left"] == 2


@pytest.mark.parametrize("url", ["http://[::1", "http://ex\uff0fample.invalid/"])
def test_a_text_that_cannot_be_read_as_a_url_is_refused_like_any_other(fetch_on, url):
    out = web.web_fetch(url)
    assert out == {"ok": False, "error": websearch.UNKNOWN_URL_ERROR, "calls_left": 2}
    with pytest.raises(ValueError, match="not one a web_search"):
        websearch.fetch(url)


@pytest.mark.parametrize("value", [5, ["https://example.invalid/guide"], None,
                                   b"https://example.invalid/guide", {"url": "x"}])
def test_a_url_or_a_question_that_is_no_string_is_refused_without_spending_a_call(
        on, fetch_on, value):
    out = web.web_fetch(value)
    assert out == {"ok": False, "calls_left": 2,
                   "error": f"url must be a string, not {type(value).__name__}"}
    out = web.web_search(value)
    assert out == {"ok": False, "calls_left": 2,
                   "error": f"question must be a string, not {type(value).__name__}"}
    assert web.search_tool(False)(value) == out
    assert (web.web_search_count(), web.web_fetch_count()) == (0, 0)


def test_an_empty_url_raises(fetch_on):
    with pytest.raises(ValueError, match="empty URL"):
        web.web_fetch("   ")


def test_a_page_that_was_not_read_is_a_refusal_not_an_answer(monkeypatch, caplog, fetch_on):

    def raising(url):
        raise RuntimeError("the page request failed: no route")

    monkeypatch.setattr(web, "_fetch", raising)
    out = web.web_fetch(PAGE_URL)
    assert out["ok"] is False and "RuntimeError" in out["error"]
    assert out["calls_left"] == 1
    assert "the page was not read" in caplog.text


def test_the_page_is_read_here_when_no_broker_is_set(monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "")
    asked = []
    monkeypatch.setattr(websearch, "fetch", lambda url: asked.append(url) or PAGE_TEXT)
    assert web._fetch(PAGE_URL) == PAGE_TEXT
    assert asked == [PAGE_URL]


def test_the_read_goes_to_the_broker_when_one_is_set(monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(websearch, "fetch",
                        lambda url: pytest.fail("the container must not read itself"))
    asked = []
    monkeypatch.setattr(web.broker, "fetch", lambda url: asked.append(url) or PAGE_TEXT)
    assert web._fetch(PAGE_URL) == PAGE_TEXT
    assert asked == [PAGE_URL]


# == the read request itself ===================================

def test_a_post_sends_the_url_under_its_parameter_with_the_headers(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_METHOD", "POST")
    monkeypatch.setattr(config, "WEB_FETCH_URL_PARAM", "urls")
    monkeypatch.setattr(config, "WEB_FETCH_PARAM", {"format": "text"})
    monkeypatch.setattr(config, "WEB_FETCH_HEADER", {"Authorization": "Bearer r-key"})
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "results")
    monkeypatch.setattr(config, "WEB_FETCH_CONTENT_FIELD", "raw_content")
    asked = _reader(monkeypatch, {"results": [{"url": PAGE_URL, "raw_content": PAGE_TEXT}],
                                  "failed_results": []})
    assert websearch.fetch(PAGE_URL) == PAGE_TEXT
    url, header_dict, body = asked[0]
    assert url == "https://reader.invalid/extract"
    assert json.loads(body) == {"format": "text", "urls": PAGE_URL}
    assert header_dict.get("Authorization") == "Bearer r-key"
    assert header_dict.get("Content-type") == "application/json"
    assert header_dict.get("User-agent", "").startswith("codetwine-loom/")


def test_a_get_puts_the_url_in_the_query(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_METHOD", "GET")
    monkeypatch.setattr(config, "WEB_FETCH_URL_PARAM", "url")
    monkeypatch.setattr(config, "WEB_FETCH_PARAM", {})
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "data")
    monkeypatch.setattr(config, "WEB_FETCH_CONTENT_FIELD", "content")
    asked = _reader(monkeypatch, {"data": {"title": "Guide", "content": PAGE_TEXT}})
    assert websearch.fetch(PAGE_URL) == PAGE_TEXT
    url, _, body = asked[0]
    assert url.startswith("https://reader.invalid/extract?")
    assert "url=https%3A%2F%2Fexample.invalid%2Fguide" in url and body is None


def test_the_page_is_read_out_of_an_object_or_the_first_of_a_list(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "results")
    monkeypatch.setattr(config, "WEB_FETCH_CONTENT_FIELD", "content")
    _reader(monkeypatch, {"results": [{"content": "first"}, {"content": "second"}]})
    assert websearch.fetch(PAGE_URL) == "first"
    _reader(monkeypatch, {"results": {"content": "the one"}})
    assert websearch.fetch(PAGE_URL) == "the one"


def test_an_empty_list_is_no_page_read(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "results")
    _reader(monkeypatch, {"results": [], "failed_results": [{"url": PAGE_URL, "error": "x"}]})
    with pytest.raises(RuntimeError, match="read no page"):
        websearch.fetch(PAGE_URL)


def test_a_reply_with_no_page_where_the_path_points_says_which_setting(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "data")
    _reader(monkeypatch, {"results": [{"content": PAGE_TEXT}]})
    with pytest.raises(ValueError, match="holds no page at 'data'"):
        websearch.fetch(PAGE_URL)


def test_a_page_without_text_under_the_field_says_which_setting(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "data")
    monkeypatch.setattr(config, "WEB_FETCH_CONTENT_FIELD", "content")
    _reader(monkeypatch, {"data": {"raw_content": PAGE_TEXT}})
    with pytest.raises(ValueError, match="no text under 'content'"):
        websearch.fetch(PAGE_URL)
    _reader(monkeypatch, {"data": {"content": "   "}})
    with pytest.raises(ValueError, match="no text under 'content'"):
        websearch.fetch(PAGE_URL)


def test_a_url_that_may_not_be_sent_does_not_leave_this_machine(monkeypatch):
    """A caller that went round the tool is stopped here, before anything is sent."""
    asked = _reader(monkeypatch, {"data": {"content": PAGE_TEXT}})
    for url in ("https://example.invalid/" + "x" * 30, "what changed", ""):
        with pytest.raises(ValueError, match="web_search"):
            websearch.fetch(url)
    assert asked == []


def test_no_reading_endpoint_says_which_setting_to_set(monkeypatch):
    _found()
    monkeypatch.setattr(config, "WEB_FETCH_URL", "")
    with pytest.raises(ValueError, match="WEB_FETCH_PROVIDER"):
        websearch.fetch(PAGE_URL)


def test_a_read_request_that_failed_says_so(monkeypatch):
    _reader(monkeypatch, urllib.error.URLError("no route"))
    with pytest.raises(RuntimeError, match="the page request failed"):
        websearch.fetch(PAGE_URL)


def test_a_reply_over_the_read_size_limit_is_refused_unread(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_BYTE", 8)
    _reader(monkeypatch, b'{"data": {"content": "x"}}')
    with pytest.raises(RuntimeError, match="more than 8 bytes.*WEB_FETCH_MAX_BYTE"):
        websearch.fetch(PAGE_URL)


def test_the_reads_keep_an_interval_of_their_own(monkeypatch):
    """A search and a read go to different endpoints, so one does not wait for the
    other; two reads do."""
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 2.0)
    monkeypatch.setattr(config, "WEB_FETCH_MIN_INTERVAL_SEC", 3.0)
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "data")
    monkeypatch.setattr(config, "WEB_FETCH_CONTENT_FIELD", "content")
    slept = _clock(monkeypatch)
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")

    def urlopen(request, timeout=None):
        if request.full_url.startswith("https://search.invalid/"):
            return io.BytesIO(json.dumps({"results": RESULTS}).encode())
        return io.BytesIO(json.dumps({"data": {"content": PAGE_TEXT}}).encode())

    monkeypatch.setattr(websearch, "_open", urlopen)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    websearch.search("one")
    websearch.fetch(PAGE_URL)
    assert slept == []
    websearch.fetch(PAGE_URL)
    assert slept == [3.0]


def test_a_slow_down_answer_to_a_read_holds_the_next_read_off(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_MIN_INTERVAL_SEC", 1.0)
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "data")
    slept = _clock(monkeypatch)
    _reader(monkeypatch, _slow_down(429, retry_after=5))
    with pytest.raises(RuntimeError, match="slow down"):
        websearch.fetch(PAGE_URL)
    _reader(monkeypatch, {"data": {"content": PAGE_TEXT}})
    websearch.fetch(PAGE_URL)
    assert slept == [5.0]


# == what the read refuses before it sends anything ============


# == a run with a broker ======================================

def test_a_broker_is_a_way_to_search_and_to_read_inside_a_container(monkeypatch):
    """The endpoint's settings stay with the broker, so a run in a container is handed
    the tools on its budget alone."""
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 2)
    assert not websearch.can_search() and not websearch.can_fetch()
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setenv("CTLOOM_WORK", "/work/repo")
    assert websearch.can_search() and websearch.can_fetch()
    assert web.available() and web.fetch_available()
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_CALL", 0)
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 0)
    assert not web.available() and not web.fetch_available()


# == what a preset can ask for besides the value =============

def test_the_page_s_url_goes_with_the_further_parameters(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_METHOD", "POST")
    monkeypatch.setattr(config, "WEB_FETCH_URL_PARAM", "urls")
    monkeypatch.setattr(config, "WEB_FETCH_PARAM", {"extract_depth": "basic"})
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "results")
    monkeypatch.setattr(config, "WEB_FETCH_CONTENT_FIELD", "raw_content")
    asked = _reader(monkeypatch, {"results": [{"url": PAGE_URL, "raw_content": PAGE_TEXT}]})
    assert websearch.fetch(PAGE_URL) == PAGE_TEXT
    assert json.loads(asked[0][2]) == {"extract_depth": "basic", "urls": PAGE_URL}


def test_a_reply_that_is_the_page_itself_is_read_with_an_empty_path(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "")
    monkeypatch.setattr(config, "WEB_FETCH_CONTENT_FIELD", "content")
    _reader(monkeypatch, {"title": "Guide", "content": PAGE_TEXT, "links": []})
    assert websearch.fetch(PAGE_URL) == PAGE_TEXT



def test_a_broker_set_by_hand_on_this_machine_is_no_way_by_itself(monkeypatch):
    """Nothing tells a process outside a container whether its broker has an endpoint,
    so it goes by its own settings."""
    monkeypatch.delenv("CTLOOM_WORK", raising=False)
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    assert not websearch.can_search() and not websearch.can_fetch()
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    assert websearch.can_search()


# == web_fetch reads only what web_search returned =============

def test_a_url_no_search_returned_is_refused_without_spending_a_call(fetch_on):
    out = web.web_fetch("https://attacker.invalid/?data=the-repository")
    assert out["ok"] is False and "web_search" in out["error"]
    assert out["calls_left"] == 2


def test_a_url_the_search_returned_is_read(monkeypatch, on):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 2)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    monkeypatch.setattr(web, "_fetch", lambda url: PAGE_TEXT)
    assert web.web_fetch(PAGE_URL)["ok"] is False
    web.web_search("what changed")
    assert web.web_fetch(PAGE_URL)["ok"] is True
    # The same page written with a trailing slash or a fragment is the same URL
    assert web.web_fetch(PAGE_URL + "/#usage")["ok"] is True


def test_the_urls_are_forgotten_between_requests(fetch_on):
    web.reset_web_search_count()
    assert web.web_fetch(PAGE_URL)["ok"] is False
    assert websearch._result_url_dict == {}


def test_the_reading_process_refuses_a_url_no_search_of_its_own_returned(monkeypatch):
    """A caller that went round the tool is stopped where the request is made."""
    asked = _reader(monkeypatch, {"data": {"content": PAGE_TEXT}})
    websearch.forget_result_url()
    with pytest.raises(ValueError, match="web_search"):
        websearch.fetch(PAGE_URL)
    assert asked == []


@pytest.mark.parametrize("written", [
    "https://example.invalid/guide#" + "repository text " * 50,
    "HTTPS://EXAMPLE.invalid/guide///#top",
    "https://example.invalid/gu\tide\n",
    " https://example.invalid/guide ",
])
def test_the_url_the_reader_is_sent_is_the_one_the_search_returned(monkeypatch, written):
    """Whatever the caller adds to a URL a search returned stays on this machine."""
    asked = _reader(monkeypatch, {"data": {"content": PAGE_TEXT}})
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "data")
    websearch.forget_result_url()
    websearch._remember_url([{"url": "https://Example.invalid/guide/#section-2"}])
    assert websearch.fetch(written) == PAGE_TEXT
    url, _, body = asked[0]
    sent = urllib.parse.unquote(url) + (body or b"").decode()
    assert "https://Example.invalid/guide/" in sent
    for part in ("repository text", "EXAMPLE", "///", "#", "section-2", "\t", "\n"):
        assert part not in sent


def test_the_searches_of_the_reading_process_are_what_it_reads(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_PATH", "data")
    _endpoint(monkeypatch, {"results": RESULTS})
    websearch.search("what changed")
    assert websearch._result_url_dict[websearch.url_key(PAGE_URL)] == PAGE_URL


def test_the_page_read_is_off_without_the_search(monkeypatch):
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 3)
    monkeypatch.setattr(config, "WEB_FETCH_URL", "https://reader.invalid/extract")
    assert web.fetch_available() is False
    monkeypatch.setattr(config, "WEB_SEARCH_URL", "https://search.invalid/search")
    assert web.fetch_available() is True


def test_url_key_ignores_case_of_the_host_a_fragment_and_a_trailing_slash():
    assert websearch.url_key("HTTPS://Example.invalid/guide/#top") == \
        websearch.url_key("https://example.invalid/guide")
    assert websearch.url_key("https://example.invalid/guide?a=1") != \
        websearch.url_key("https://example.invalid/guide?a=2")


# == what a result is kept as ==================================

def test_a_key_is_kept_under_the_name_the_setting_gives(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_FIELD", "title,link:url,snippet:content")
    _endpoint(monkeypatch, {"results": [{"title": "T", "link": "L", "snippet": "S",
                                         "position": 1}]})
    assert websearch.search("what changed") == [{"title": "T", "url": "L", "content": "S"}]


def test_every_search_preset_gives_title_url_and_content():
    for name, preset in config.WEB_SEARCH_PRESET_DICT.items():
        names = [part.partition(":")[2] or part for part in preset["result_keys"].split(",")]
        assert names == ["title", "url", "content"], name


def test_each_result_is_cut_to_its_share_of_the_limit(monkeypatch):
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_MAX_CHAR", 100)
    monkeypatch.setattr(config, "WEB_SEARCH_MAX_RESULT", 4)
    _endpoint(monkeypatch, {"results": [{"title": "T", "url": "U", "content": "x" * 500}
                                        for _ in range(4)]})
    result_list = websearch.search("what changed")
    assert len(result_list) == 4
    assert all(r["content"].startswith("x" * 25) and "chars omitted" in r["content"]
               for r in result_list)


# == the limits as the instructions state them ================

def _web_note_list(monkeypatch, limit):
    """Render the web notes of the editing, planning and dispatch instructions."""
    from codetwine_loom import agent, dispatch, plan

    monkeypatch.setattr(web, "available", lambda: True)
    monkeypatch.setattr(web, "fetch_available", lambda: True)
    monkeypatch.setattr(config, "WEB_SEARCH_RESULT_MAX_CHAR", limit)
    monkeypatch.setattr(config, "WEB_FETCH_RESULT_MAX_CHAR", limit)
    return [agent.web_note(), plan.plan_web_note(),
            dispatch.dispatch_web_note([web.web_search, web.web_fetch])]


def test_the_instructions_state_the_result_limits(monkeypatch):
    agent_note, plan_note, dispatch_note = _web_note_list(monkeypatch, 40000)
    assert "cut to\n  40000 characters over the whole list." in agent_note
    assert '"content", cut to 40000 characters. Use it' in agent_note
    assert "(an excerpt), cut to 40000\ncharacters over the whole list." in plan_note
    assert "the page's text, cut to\n40000 characters. Use it" in plan_note
    assert '"content", cut to 40000 characters. It takes' in dispatch_note


def test_the_instructions_state_no_result_limit_of_0(monkeypatch):
    for note in _web_note_list(monkeypatch, 0):
        assert "web_fetch" in note and "cut to" not in note


# == a redirect the endpoint answers with ======================

_FAKE_KEY = "sk-test-not-a-key"


def _server(answer):
    """Start an HTTP server on a port of this machine that answers every request one way.

    Args:
        answer: (the handler) -> None, which writes the answer.

    Returns:
        (the server, the list each request is recorded in as (method, path, the
        header names in lower case holding _FAKE_KEY, the body)).
    """
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _take(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen.append((self.command, self.path,
                         sorted(name.lower() for name, value in self.headers.items()
                                if _FAKE_KEY in value), body))
            answer(self)

        do_GET = do_POST = _take

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


@pytest.fixture
def _redirect_pair():
    """Two servers: (the port of the one that redirects, what each one saw, a
    function that sets the status the first answers with)."""
    status = [302]

    def found(handler):
        body = json.dumps({"results": RESULTS, "data": {"content": PAGE_TEXT}}).encode()
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    other, other_seen = _server(found)

    def redirect(handler):
        handler.send_response(status[0])
        handler.send_header(
            "Location", f"http://localhost:{other.server_address[1]}/moved")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    first, first_seen = _server(redirect)
    yield first.server_address[1], first_seen, other_seen, status
    for server in (first, other):
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("status", [300, 301, 302, 303, 307, 308])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_a_search_endpoint_that_answers_with_a_redirect_is_asked_nowhere_else(
        monkeypatch, _redirect_pair, status, method):
    """The request, and the key in its headers, goes to the endpoint alone, and what
    another address would answer is no search result."""
    port, first_seen, other_seen, status_now = _redirect_pair
    status_now[0] = status
    monkeypatch.setattr(config, "WEB_SEARCH_URL", f"http://127.0.0.1:{port}/search")
    monkeypatch.setattr(config, "WEB_SEARCH_METHOD", method)
    monkeypatch.setattr(config, "WEB_SEARCH_HEADER", {
        "Authorization": f"Bearer {_FAKE_KEY}", "X-Subscription-Token": _FAKE_KEY})
    monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 0.0)
    with pytest.raises(RuntimeError) as raised:
        websearch.search("what changed")
    assert str(raised.value) == (f"the search endpoint answered with a redirect "
                                 f"(HTTP {status}), which is not followed")
    assert [(seen[0], seen[2]) for seen in first_seen] == [
        (method, ["authorization", "x-subscription-token"])]
    assert other_seen == []


def test_a_page_reading_endpoint_that_answers_with_a_redirect_is_asked_nowhere_else(
        monkeypatch, _redirect_pair):
    port, first_seen, other_seen, _ = _redirect_pair
    _found()
    monkeypatch.setattr(config, "WEB_FETCH_URL", f"http://127.0.0.1:{port}/extract")
    monkeypatch.setattr(config, "WEB_FETCH_METHOD", "POST")
    monkeypatch.setattr(config, "WEB_FETCH_HEADER", {"Authorization": f"Bearer {_FAKE_KEY}"})
    monkeypatch.setattr(config, "WEB_FETCH_MIN_INTERVAL_SEC", 0.0)
    with pytest.raises(RuntimeError) as raised:
        websearch.fetch(PAGE_URL)
    assert str(raised.value) == ("the page endpoint answered with a redirect "
                                 "(HTTP 302), which is not followed")
    assert len(first_seen) == 1 and other_seen == []
    # What the tool hands the model names the kind of failure alone
    monkeypatch.setattr(config, "WEB_FETCH_MAX_CALL", 1)
    out = web.web_fetch(PAGE_URL)
    assert out["ok"] is False and "localhost" not in out["error"] \
        and "moved" not in out["error"]
    assert other_seen == []


def test_a_redirect_to_an_address_urllib_would_refuse_is_told_the_same_way(monkeypatch):
    """The address a redirect names is not read, so it is in no message."""
    def answer(handler):
        handler.send_response(302)
        handler.send_header("Location", "file:///etc/hostname?ctloom-location-mark")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    server, seen = _server(answer)
    try:
        monkeypatch.setattr(config, "WEB_SEARCH_URL",
                            f"http://127.0.0.1:{server.server_address[1]}/search")
        monkeypatch.setattr(config, "WEB_SEARCH_MIN_INTERVAL_SEC", 0.0)
        with pytest.raises(RuntimeError) as raised:
            websearch.search("what changed")
    finally:
        server.shutdown()
        server.server_close()
    assert str(raised.value) == ("the search endpoint answered with a redirect "
                                 "(HTTP 302), which is not followed")
    assert len(seen) == 1


# == the proxy a request is sent through ======================
def _without_proxy_variable(monkeypatch) -> None:
    for name in config.PROXY_NAME_TUPLE:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


def test_all_proxy_stands_in_for_the_scheme_that_names_no_proxy(monkeypatch):
    _without_proxy_variable(monkeypatch)
    assert websearch._proxy_dict() == {}
    monkeypatch.setenv("ALL_PROXY", "http://all.example:3128")
    assert websearch._proxy_dict() == {
        "all": "http://all.example:3128", "http": "http://all.example:3128",
        "https": "http://all.example:3128"}
    monkeypatch.setenv("https_proxy", "http://secure.example:3128")
    assert websearch._proxy_dict()["https"] == "http://secure.example:3128"
    assert websearch._proxy_dict()["http"] == "http://all.example:3128"


@pytest.mark.parametrize("value, want", [
    ("http://ctloom-proxy.corp.example", "http://ctloom-proxy.corp.example:80"),
    ("http://ctloom-proxy.corp.example/", "http://ctloom-proxy.corp.example:80/"),
    ("me:p%23w@ctloom-proxy.corp.example", "http://me:p%23w@ctloom-proxy.corp.example:80"),
    ("ctloom-proxy.corp.example:3128", "http://ctloom-proxy.corp.example:3128"),
    ("HTTP://[fd00::1]", "HTTP://[fd00::1]:80"),
    ("http://[fd00::1]:3128", "http://[fd00::1]:3128"),
    ("http://ctloom-proxy.corp.example:3128", "http://ctloom-proxy.corp.example:3128"),
    ("https://ctloom-proxy.corp.example", "https://ctloom-proxy.corp.example"),
    ("socks5://ctloom-proxy.corp.example", "socks5://ctloom-proxy.corp.example")])
def test_urllib_is_handed_the_port_an_http_client_reads_for_a_proxy(monkeypatch, value,
                                                                    want):
    """httpx goes to port 80 for an http:// proxy that names no port, whatever the
    scheme of the request."""
    _without_proxy_variable(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", value)
    monkeypatch.setenv("no_proxy", "localhost")
    proxy_dict = websearch._proxy_dict()
    assert proxy_dict["https"] == want and proxy_dict["no"] == "localhost"
    assert websearch._proxy_name("https", want) == "HTTPS_PROXY"


def test_a_request_is_sent_to_the_proxy_all_proxy_names(monkeypatch):
    """The proxy is a socket of this process, which answers the CONNECT with a 403."""
    import socket
    import threading
    _without_proxy_variable(monkeypatch)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    line_list = []

    def answer():
        conn, _ = server.accept()
        with conn:
            line_list.append(conn.recv(4096).split(b"\r\n")[0].decode())
            conn.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
    thread = threading.Thread(target=answer, daemon=True)
    thread.start()
    monkeypatch.setenv("ALL_PROXY", f"http://127.0.0.1:{server.getsockname()[1]}")
    request = urllib.request.Request("https://search.example/v1?q=x")
    with pytest.raises(OSError, match="Tunnel connection failed: 403"):
        websearch._open(request, timeout=10)
    thread.join(10)
    server.close()
    assert [line.rpartition(" ")[0] for line in line_list] == ["CONNECT search.example:443"]
    # A host the variables name as one reached without the proxy is not sent to it
    monkeypatch.setenv("NO_PROXY", "search.example")
    assert urllib.request.proxy_bypass("search.example")


@pytest.mark.parametrize("name, value, why", [
    ("HTTPS_PROXY", "https://svcname:secretpw@127.0.0.1:{port}",
     "web_search and web_fetch go through http:// proxies only, and HTTPS_PROXY names"),
    ("all_proxy", "socks5://svcname:secretpw@127.0.0.1:{port}",
     "all_proxy points at a SOCKS proxy"),
    ("https_proxy", "http://svcname:pa#secretpw@127.0.0.1:{port}",
     "https_proxy holds a '#', '/' or '?' in the user or the password")])
def test_a_proxy_urllib_cannot_go_through_as_meant_has_no_request_sent(monkeypatch, name,
                                                                       value, why):
    """Nothing reaches the proxy's port, and the reason names the variable alone."""
    import socket
    _without_proxy_variable(monkeypatch)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(0.5)
    monkeypatch.setenv(name, value.format(port=server.getsockname()[1]))
    request = urllib.request.Request("https://search.example/v1?q=x")
    with pytest.raises(RuntimeError, match="^no request is sent: ") as caught:
        websearch._open(request, timeout=10)
    assert why in str(caught.value)
    assert "secretpw" not in str(caught.value) and "svcname" not in str(caught.value)
    with pytest.raises(socket.timeout):
        server.accept()
    server.close()
    # A host reached without the proxy is sent to directly, whatever the proxy is
    monkeypatch.setenv("NO_PROXY", "search.example")
    with pytest.raises(OSError) as caught:
        websearch._open(request, timeout=10)
    assert not isinstance(caught.value, RuntimeError)

