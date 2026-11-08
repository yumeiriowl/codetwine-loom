"""The search a question of the web is, and the read of one page of it.

A search is made one of two ways. Over HTTP, the question goes to the search API that
WEB_SEARCH_PROVIDER names and no model takes part. Nothing about any one API is written
here: its URL, the query parameter, the further parameters, the headers and where the
results sit in the reply are read from the WEB_SEARCH_* settings (_search_endpoint), and
the keys of a result that are kept from WEB_SEARCH_RESULT_FIELD.

With WEB_SEARCH_PROVIDER set to WEB_SEARCH_BY_THE_MODEL, the search is one call of
WEB_SEARCH_MODEL with its provider's own web search on, made with the LLM credentials.
The model reads what it finds and answers; the answer comes back as the first result and
the pages it cited follow, in the same shape as an HTTP search's results.

A page is read the HTTP way alone: its URL goes to the page-reading API that
WEB_FETCH_PROVIDER names, which fetches the page itself, and the page's text is read out
of the reply where WEB_FETCH_RESULT_PATH and WEB_FETCH_CONTENT_FIELD say. Nothing here
fetches a page directly. A URL no search made by this process returned is refused, and
one that is read is sent as the search returned it, without its fragment.

With no endpoint there is no search, or no read.

A request names this package in its User-Agent, two requests to one endpoint from one
process are the endpoint's minimum interval apart at the least, and an endpoint that
answers 429 or 503 with a Retry-After is not asked again before that time has passed.
The requests several threads of one process make to one endpoint go out one after
another. A redirect an endpoint answers with is not followed: the request, with the
headers it carries, goes to the endpoint's own address alone, and the search or the read
fails (_open).
"""

import collections
import contextlib
import dataclasses
import http.client
import importlib.metadata
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

from codetwine_loom import config, providercall, textcut

logger = logging.getLogger(__name__)

# What the request says it is. A User-Agent in the endpoint's headers is sent instead
try:
    _VERSION = importlib.metadata.version("codetwine-loom")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "dev"
_USER_AGENT = f"codetwine-loom/{_VERSION}"

# The statuses an endpoint says "not now" with. Their Retry-After says for how long
_SLOW_DOWN_STATUS_TUPLE = (429, 503)

# The moment, on time.monotonic()'s clock, before which this process makes no request to
# an endpoint, by the endpoint's setting prefix. Every request moves it the endpoint's
# minimum interval on
_not_before_dict: dict[str, float] = {}

# The moment before which an endpoint asked to be left alone, by the endpoint's setting
# prefix. A slow-down answer moves it as far on as its Retry-After says
_hold_until_dict: dict[str, float] = {}

# What one thread holds from the wait for its request to an endpoint until the answer
# is read, by the endpoint's setting prefix
_turn_lock_dict: dict[str, threading.Lock] = {}

# The URLs the searches this process made returned since forget_result_url() last
# cleared them: url_key() of each, to the URL without its fragment
# (_url_without_fragment)
_result_url_dict: dict[str, str] = {}


@dataclasses.dataclass(frozen=True)
class _Endpoint:
    """One endpoint as its settings describe it.

    Attributes:
        prefix: The prefix its settings share: "WEB_SEARCH" or "WEB_FETCH".
        kind: What a request to it is, in a failure's words: "search" or "page".
        url: The endpoint's URL.
        method: "GET" or "POST".
        value_param: The parameter the question or the page's URL is sent as.
        param_dict: The further parameters.
        header_dict: The request's headers, the key among them.
        result_path: The dotted path to the results, or the page, in the reply.
        timeout_sec: The time limit of one request.
        min_interval_sec: The shortest time between two requests from this process.
        max_byte: The largest reply read.
    """
    prefix: str
    kind: str
    url: str
    method: str
    value_param: str
    param_dict: dict
    header_dict: dict
    result_path: str
    timeout_sec: float
    min_interval_sec: float
    max_byte: int


def _search_endpoint() -> _Endpoint:
    """Return the search endpoint as the WEB_SEARCH_* settings describe it now."""
    return _Endpoint("WEB_SEARCH", "search", config.WEB_SEARCH_URL, config.WEB_SEARCH_METHOD,
                     config.WEB_SEARCH_QUERY_PARAM, config.WEB_SEARCH_PARAM,
                     config.WEB_SEARCH_HEADER, config.WEB_SEARCH_RESULT_PATH,
                     config.WEB_SEARCH_TIMEOUT_SEC, config.WEB_SEARCH_MIN_INTERVAL_SEC,
                     config.WEB_SEARCH_MAX_BYTE)


def _fetch_endpoint() -> _Endpoint:
    """Return the page-reading endpoint as the WEB_FETCH_* settings describe it now."""
    return _Endpoint("WEB_FETCH", "page", config.WEB_FETCH_URL, config.WEB_FETCH_METHOD,
                     config.WEB_FETCH_URL_PARAM, config.WEB_FETCH_PARAM,
                     config.WEB_FETCH_HEADER, config.WEB_FETCH_RESULT_PATH,
                     config.WEB_FETCH_TIMEOUT_SEC, config.WEB_FETCH_MIN_INTERVAL_SEC,
                     config.WEB_FETCH_MAX_BYTE)


def _by_the_model() -> bool:
    """Whether the search is made by the model's own provider search."""
    return config.WEB_SEARCH_PROVIDER == config.WEB_SEARCH_BY_THE_MODEL


def _through_the_broker() -> bool:
    """Whether LLM_BROKER names a broker and this process runs in a container.

    Returns:
        True when LLM_BROKER names a broker and this process runs in a container.
    """
    # sandbox imports this module: in_container is imported at call time
    from codetwine_loom.sandbox import in_container
    return bool(config.LLM_BROKER) and in_container()


def can_search() -> bool:
    """Whether there is a way to search.

    Returns:
        True when WEB_SEARCH_PROVIDER names a search API or the model's provider, or
        the run is in a container with a broker.
    """
    return bool(config.WEB_SEARCH_URL) or _by_the_model() or _through_the_broker()


def can_fetch() -> bool:
    """Whether there is a way to read a page.

    Returns:
        True when WEB_FETCH_PROVIDER names a page-reading API, or the run is in a
        container with a broker.
    """
    return bool(config.WEB_FETCH_URL) or _through_the_broker()


def url_key(url: str) -> str:
    """Return the form two URLs of one page are compared in.

    The scheme and the host are lowered, the fragment is dropped and a trailing slash
    of the path does not count.

    Examples:
        "HTTPS://Example.invalid/guide/#top" -> "https://example.invalid/guide"
        "https://example.invalid"            -> "https://example.invalid"

    Args:
        url: The URL as it was written.

    Returns:
        The URL in the compared form, or "" for a text urllib cannot read as a URL.
    """
    try:
        url_split = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return ""
    return urllib.parse.urlunsplit((url_split.scheme.lower(), url_split.netloc.lower(),
                                    url_split.path.rstrip("/"), url_split.query, ""))


def _url_without_fragment(url: str) -> str:
    """Return the form a URL a search returned is sent to the page-reading API in.

    Examples:
        "https://Example.invalid/guide/#top" -> "https://Example.invalid/guide/"
        " https://example.invalid/a?b=1 "    -> "https://example.invalid/a?b=1"

    Args:
        url: The URL as the search returned it.

    Returns:
        The URL without its fragment and without the blanks around it.
    """
    url_split = urllib.parse.urlsplit(url.strip())
    return urllib.parse.urlunsplit(url_split._replace(fragment=""))


def forget_result_url() -> None:
    """Forget the URLs the searches returned."""
    _result_url_dict.clear()


def _remember_url(result_list: list[dict]) -> list[dict]:
    """Keep the URLs of the results a search returned.

    Of the URLs url_key() gives one form for, the first one returned is kept.

    Args:
        result_list: The results, each maybe holding "url".

    Returns:
        The results without the ones whose URL url_key() cannot read.
    """
    keep_list = []
    for result_dict in result_list:
        url = result_dict.get("url")
        if isinstance(url, str) and url:
            key = url_key(url)
            if not key:
                logger.info("a result whose URL cannot be read is left out")
                continue
            _result_url_dict.setdefault(key, _url_without_fragment(url))
        keep_list.append(result_dict)
    return keep_list


def check_question(question: str) -> str:
    """Say why a question may not be sent, or "" when it may.

    Args:
        question: The question as it was written.

    Returns:
        "" when the question may be sent, otherwise a sentence saying what to do.
    """
    if not question or not question.strip():
        return "the question is empty. Pass one short question in English"
    if len(question) > config.WEB_SEARCH_QUESTION_MAX_CHAR:
        return (f"the question is {len(question)} characters; at most "
                f"{config.WEB_SEARCH_QUESTION_MAX_CHAR} are sent. Ask what you need "
                f"to know, without pasting code into it")
    return ""


# What a URL no search returned is refused with
UNKNOWN_URL_ERROR = ("that URL is not one a web_search of this request returned. Pass the "
                     "\"url\" of one of its results")


def _url_for(endpoint: _Endpoint, value: str) -> str:
    """Build a GET request's URL.

    The value goes in the endpoint's value parameter and its further parameters supply
    the rest. A parameter already in the endpoint's URL is kept and the settings are
    added to it.

    Args:
        endpoint: The endpoint.
        value: The question, or the page's URL.

    Returns:
        The whole URL.
    """
    url_split = urllib.parse.urlsplit(endpoint.url)
    query_list = urllib.parse.parse_qsl(url_split.query)
    query_list += [(str(name), str(param_value))
                   for name, param_value in endpoint.param_dict.items()]
    query_list.append((endpoint.value_param, value))
    return urllib.parse.urlunsplit(
        (url_split.scheme, url_split.netloc, url_split.path, urllib.parse.urlencode(query_list),
         url_split.fragment))


def _request_for(endpoint: _Endpoint, value: str) -> urllib.request.Request:
    """Build the request, the way the endpoint's method says.

    GET puts the further parameters and the value in the URL. POST sends them as a
    JSON object to the endpoint's URL as it is, under the value parameter and their own
    names.

    Args:
        endpoint: The endpoint.
        value: The question, or the page's URL, already checked.

    Returns:
        The request, with the User-Agent and the endpoint's headers on it.
    """
    header_dict = {"User-Agent": _USER_AGENT, **endpoint.header_dict}
    if endpoint.method == "POST":
        body = json.dumps({**endpoint.param_dict,
                           endpoint.value_param: value}).encode("utf-8")
        return urllib.request.Request(
            endpoint.url, data=body, method="POST",
            headers={**header_dict, "Content-Type": "application/json"})
    return urllib.request.Request(_url_for(endpoint, value), headers=header_dict)


def _at_path(reply: object, path: str) -> object:
    """Follow a dotted path into a parsed reply.

    Examples:
        ({"web": {"results": [1]}}, "web.results") -> [1]
        ([1, 2], "")                               -> [1, 2]

    Args:
        reply: The parsed JSON reply.
        path: The dotted path, or "" for the reply itself.

    Returns:
        Whatever is at that path, or None when the path is not in the reply.
    """
    node = reply
    for step in (part for part in path.split(".") if part):
        if not isinstance(node, dict):
            return None
        node = node.get(step)
    return node


def _keep_key_dict(result_dict: dict) -> dict:
    """Return one result cut to the keys that are kept.

    Args:
        result_dict: One element of the endpoint's result list.

    Returns:
        {name: value} for the keys of WEB_SEARCH_RESULT_FIELD the result holds, in that
        order and under the names it gives, with the values as strings, each cut to
        _field_max_char().
    """
    if not isinstance(result_dict, dict):
        return {}
    return {name: textcut.cut(str(result_dict[key]), _field_max_char())
            for key, name in _field_pair_list() if key in result_dict}


def _field_max_char() -> int:
    """Return how many characters one value of one result keeps.

    Returns:
        An equal share of WEB_SEARCH_RESULT_MAX_CHAR over WEB_SEARCH_MAX_RESULT
        results, at least 1, or 0 (no cap) when WEB_SEARCH_RESULT_MAX_CHAR is 0.
    """
    if config.WEB_SEARCH_RESULT_MAX_CHAR <= 0:
        return 0
    return max(1, config.WEB_SEARCH_RESULT_MAX_CHAR // max(1, config.WEB_SEARCH_MAX_RESULT))


def _field_pair_list() -> list[tuple[str, str]]:
    """Read WEB_SEARCH_RESULT_FIELD into the keys kept and the names they are kept under.

    Examples:
        "title,url,description:content" -> [("title", "title"), ("url", "url"),
                                             ("description", "content")]

    Returns:
        [(the key in a result, the name it is kept under)], in the setting's order.
    """
    pair_list = []
    for part in (piece.strip() for piece in config.WEB_SEARCH_RESULT_FIELD.split(",")):
        if part:
            key, _, name = part.partition(":")
            pair_list.append((key.strip(), name.strip() or key.strip()))
    return pair_list


@contextlib.contextmanager
def _turn(endpoint: _Endpoint) -> Iterator[None]:
    """Hold an endpoint for one request of this process.

    The calls several threads make at once take the endpoint one after another: each
    waits its turn (_wait_turn), makes its request inside the block and records what
    the answer asked for (_hold_off) before the next one reads the wait.

    Args:
        endpoint: The endpoint.

    Yields:
        Nothing. The block runs with the endpoint held.

    Raises:
        RuntimeError: What _wait_turn() raises. The block is not entered.
    """
    with _turn_lock_dict.setdefault(endpoint.prefix, threading.Lock()):
        _wait_turn(endpoint)
        yield


def _wait_turn(endpoint: _Endpoint) -> None:
    """Wait until this process may make a request to the endpoint, and hold the next off.

    Called with the endpoint held (_turn). The wait is the longer of what is left of the
    interval since the last request, which is waited whatever its length, and what a
    slow-down answer asked for. The next request is held off by the endpoint's minimum
    interval from now.

    Args:
        endpoint: The endpoint.

    Raises:
        RuntimeError: When the wait a slow-down answer asked for would be longer than
            the endpoint's time limit. Nothing is sent, and the wait stands for the
            next call.
    """
    now = time.monotonic()
    ask_wait = _hold_until_dict.get(endpoint.prefix, 0.0) - now
    if ask_wait > endpoint.timeout_sec:
        raise RuntimeError(f"the endpoint asked for a wait of {ask_wait:.0f} s, which is "
                           f"over {endpoint.prefix}_TIMEOUT_SEC ({endpoint.timeout_sec})")
    wait = max(ask_wait, _not_before_dict.get(endpoint.prefix, 0.0) - now)
    if wait > 0:
        logger.info("waiting %.1f s before the next %s request", wait, endpoint.kind)
        time.sleep(wait)
    _not_before_dict[endpoint.prefix] = time.monotonic() + endpoint.min_interval_sec


def _hold_off(endpoint: _Endpoint, wait_sec: float) -> None:
    """Make no request to the endpoint for the next seconds, unless a longer wait stands.

    Args:
        endpoint: The endpoint.
        wait_sec: How long, from now.
    """
    _hold_until_dict[endpoint.prefix] = max(_hold_until_dict.get(endpoint.prefix, 0.0),
                                            time.monotonic() + wait_sec)


def _retry_after(error: urllib.error.HTTPError) -> float:
    """Read how long a slow-down answer says to wait.

    Args:
        error: The 429 or 503 answer.

    Returns:
        Its Retry-After header as seconds. 0 when there is none, or when it is not a
        whole number of seconds.
    """
    value = (error.headers.get("Retry-After") or "").strip()
    return float(value) if value.isascii() and value.isdigit() else 0.0


def search(question: str) -> list[dict]:
    """Search the web for one question.

    The question and the endpoint are checked here.

    Args:
        question: One short question. check_question() is applied first: a question
            over the limit does not leave this machine.

    Returns:
        Up to WEB_SEARCH_MAX_RESULT results, each cut to WEB_SEARCH_RESULT_FIELD. An
        endpoint that found nothing gives an empty list.

    Raises:
        ValueError: When the question may not be sent, when WEB_SEARCH_PROVIDER names
            no search, or when the reply holds no result list where the preset says.
        RuntimeError: When the endpoint asked for a wait over WEB_SEARCH_TIMEOUT_SEC or
            said to slow down, when the request failed, or when the reply is over
            WEB_SEARCH_MAX_BYTE or is not JSON.
    """
    refusal = check_question(question)
    if refusal:
        raise ValueError(refusal)
    if _by_the_model():
        return _search_by_model(question.strip())
    if not can_search():
        raise ValueError("WEB_SEARCH_PROVIDER names no search. Set it to one of "
                         f"{', '.join(config.WEB_SEARCH_PRESET_DICT)}, "
                         f"{config.WEB_SEARCH_BY_THE_MODEL}")
    return _search_over_http(question.strip())


def _search_by_model(question: str) -> list[dict]:
    """Have WEB_SEARCH_MODEL search, through its provider's own web search.

    Args:
        question: The question, already checked.

    Returns:
        Up to WEB_SEARCH_MAX_RESULT results: the model's answer first, with no title and
        no URL, then the pages the reply names, each with the text the model cited from
        it as its content. Each value is cut to _field_max_char() and holds no half of a
        surrogate pair (textcut.no_lone_surrogate). A page whose URL cannot be read is
        left out (_remember_url).

    Raises:
        RuntimeError: When the call failed, the provider has no web search, or the
            reply names no page, which is a reply made without a search.
    """
    with _turn(_search_endpoint()):
        try:
            reply = providercall.completion({
                "model": config.WEB_SEARCH_MODEL,
                "max_tokens": providercall.output_limit_of(config.WEB_SEARCH_MODEL,
                                                           config.WEB_SEARCH_MAX_TOKEN),
                "timeout": config.LLM_CALL_TIMEOUT_SEC,
                "messages": [{"role": "user", "content": question}],
                "web_search_options": {
                    "search_context_size": config.WEB_SEARCH_CONTEXT_SIZE},
                **config.credential_dict()})
        except Exception as e:                # noqa: BLE001 - see the comment below
            # Whatever the provider raises is one failed search, raised as RuntimeError
            raise RuntimeError(
                f"the model's search failed: {type(e).__name__}: {e}") from e
    answer = (reply.choices[0].message.content or "").strip()
    # An answer cut off at WEB_SEARCH_MAX_TOKEN is kept, with the marker at its end. rlm
    # reaches this module through llm and broker: the marker is imported at call time
    if answer and getattr(reply.choices[0], "finish_reason", None) == "length":
        from codetwine_loom.rlm import CUT_END_NOTE
        answer += CUT_END_NOTE
    page_list = _page_list_in(reply.model_dump())
    if not page_list:
        raise RuntimeError(
            f"the reply of {config.WEB_SEARCH_MODEL} names no page, so it was made "
            f"without a search. Set WEB_SEARCH_MODEL to a model whose provider "
            f"searches the web, or WEB_SEARCH_PROVIDER to a search API")
    result_list = ([{"title": "", "url": "", "content": answer}] if answer else []) \
        + page_list
    logger.info("the model's search returned %d page(s)", len(page_list))
    result_list = [{key: textcut.cut(textcut.no_lone_surrogate(value), _field_max_char())
                    for key, value in result_dict.items()}
                   for result_dict in result_list[:config.WEB_SEARCH_MAX_RESULT]]
    return _remember_url(result_list)


def _page_list_in(reply_dict: dict) -> list[dict]:
    """Collect the pages a reply names, wherever its provider put them.

    Any object holding "title" and "url" (or "uri") is a page, and one holding
    "cited_text" as well carries the text the model quoted from it. Each URL is kept
    once, where it first appears.

    Args:
        reply_dict: The reply as a dict, as litellm's model_dump() gives it.

    Returns:
        [{"title": str, "url": str, "content": str}], in the order found.
    """
    page_dict: dict[str, dict] = {}
    queue = collections.deque([reply_dict])
    while queue:
        node = queue.popleft()
        if isinstance(node, dict):
            url = node.get("url") or node.get("uri")
            if isinstance(url, str) and url and "title" in node:
                page_dict.setdefault(url, {"title": str(node.get("title") or ""),
                                           "url": url,
                                           "content": str(node.get("cited_text") or "")})
            queue.extend(node.values())
        elif isinstance(node, list):
            queue.extend(node)
    return list(page_dict.values())


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follows no redirect: a 3xx answer is raised as the HTTPError it is.

    The address the answer names is not read.
    """

    def http_error_302(self, req: urllib.request.Request, fp: object, code: int,
                       msg: str, headers: object) -> None:
        """Return no reply in place of the redirect, whatever its status."""
        return None

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _proxy_dict() -> dict[str, str]:
    """Return the proxy each scheme is sent through, as the proxy variables name them.

    Examples:
        ALL_PROXY=http://proxy.corp.example:3128, HTTP_PROXY unset, HTTPS_PROXY unset
            -> {"all": "http://proxy.corp.example:3128",
                "http": "http://proxy.corp.example:3128",
                "https": "http://proxy.corp.example:3128"}
        HTTPS_PROXY=http://proxy.corp.example -> {"https": "http://proxy.corp.example:80"}

    Returns:
        What urllib.request.getproxies() gives, with the proxy of "all" standing in
        for "http" and for "https" where it names none, and each proxy's URL through
        config.proxy_with_port().
    """
    proxy_dict = urllib.request.getproxies()
    for scheme in ("http", "https"):
        if "all" in proxy_dict:
            proxy_dict.setdefault(scheme, proxy_dict["all"])
    return {scheme: proxy if scheme == "no" else config.proxy_with_port(proxy)
            for scheme, proxy in proxy_dict.items()}


def _proxy_name(scheme: str, proxy: str) -> str:
    """Name the proxy variable urllib took a scheme's proxy from.

    Args:
        scheme: The scheme of the request: "http" or "https".
        proxy: The proxy _proxy_dict() names for it.

    Returns:
        The first of <scheme>_proxy, <SCHEME>_PROXY, all_proxy and ALL_PROXY whose
        value config.proxy_with_port() gives as proxy, as the environment holds it.
        "" when none does.
    """
    return next((name for name in (f"{scheme}_proxy", f"{scheme.upper()}_PROXY",
                                   "all_proxy", "ALL_PROXY")
                 if os.environ.get(name)
                 and config.proxy_with_port(os.environ[name]) == proxy), "")


def _refuse_proxy(request: urllib.request.Request, proxy_dict: dict[str, str]) -> None:
    """Refuse a request whose proxy urllib cannot be sent through as it is meant to be.

    The proxy is the one proxy_dict names for the request's scheme, when the proxy
    variables do not name its host as one reached without a proxy. It is refused when
    config.proxy_url_error() refuses its URL, and when its scheme is not http (a value
    without "://" is read as http://).

    Args:
        request: The request, not sent yet.
        proxy_dict: What _proxy_dict() returned.

    Raises:
        RuntimeError: Naming the proxy variable and why, without its value.
    """
    proxy = proxy_dict.get(request.type, "")
    if not proxy or urllib.request.proxy_bypass(request.host):
        return
    name = _proxy_name(request.type, proxy) or "the proxy variable"
    error = config.proxy_url_error(proxy)
    if error:
        raise RuntimeError(f"no request is sent: {name} {error}")
    if "://" in proxy and proxy.partition("://")[0].lower() != "http":
        raise RuntimeError(f"no request is sent: web_search and web_fetch go through "
                           f"http:// proxies only, and {name} names another")


def _open(request: urllib.request.Request, timeout: float) -> object:
    """Send one request and return the open reply, following no redirect.

    The request goes to the address it names and nowhere else: the headers it
    carries are sent to that address alone. It is sent through the proxy _proxy_dict()
    names for its scheme, unless the proxy variables name its host as one reached
    without a proxy. A proxy _refuse_proxy() refuses has no request sent.

    Args:
        request: The request.
        timeout: The seconds to wait for the endpoint.

    Returns:
        The reply, as urllib.request.urlopen() returns one.

    Raises:
        urllib.error.HTTPError: When the endpoint answered with a status that is not
            2xx, a redirect included.
        urllib.error.URLError, http.client.HTTPException, OSError: As
            urllib.request.urlopen() raises them.
        RuntimeError: When _refuse_proxy() refuses the proxy.
    """
    proxy_dict = _proxy_dict()
    _refuse_proxy(request, proxy_dict)
    return urllib.request.build_opener(
        _NoRedirectHandler, urllib.request.ProxyHandler(proxy_dict)
    ).open(request, timeout=timeout)


def _reply_of(endpoint: _Endpoint, value: str) -> object:
    """Send one value to the endpoint and return its parsed reply.

    One request is made (_open). An endpoint that answers with a redirect is not
    asked anywhere else.

    Args:
        endpoint: The endpoint.
        value: The question, or the page's URL, already checked.

    Returns:
        The reply, parsed from JSON, its strings holding no half of a surrogate pair
        (textcut.no_lone_surrogate).

    Raises:
        RuntimeError: When the endpoint asked for a wait over its time limit or said to
            slow down, when it answered with a redirect, when the request failed or
            its answer could not be read as HTTP, or when the reply is over the
            endpoint's largest reply or is not JSON.
    """
    request = _request_for(endpoint, value)
    with _turn(endpoint):
        try:
            with _open(request, timeout=endpoint.timeout_sec) as reply:
                # One byte over the limit is read and nothing beyond it: an endpoint
                # that answers without end is not followed
                body = reply.read(endpoint.max_byte + 1)
        except urllib.error.HTTPError as e:
            if e.code in _SLOW_DOWN_STATUS_TUPLE:
                _hold_off(endpoint, _retry_after(e))
                raise RuntimeError(
                    f"the endpoint asked to slow down (HTTP {e.code})") from e
            if 300 <= e.code < 400:
                raise RuntimeError(
                    f"the {endpoint.kind} endpoint answered with a redirect "
                    f"(HTTP {e.code}), which is not followed") from e
            raise RuntimeError(f"the {endpoint.kind} request failed: {e}") from e
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            raise RuntimeError(f"the {endpoint.kind} request failed: "
                               f"{type(e).__name__}: {e}") from e
    if len(body) > endpoint.max_byte:
        raise RuntimeError(f"the {endpoint.kind} endpoint answered with more than "
                           f"{endpoint.max_byte} bytes, which is as much as is read "
                           f"({endpoint.prefix}_MAX_BYTE)")
    try:
        return textcut.no_lone_surrogate(json.loads(body))
    except (ValueError, RecursionError) as e:
        raise RuntimeError(f"the {endpoint.kind} endpoint answered with something that "
                           f"is not JSON: {e}") from e


def _search_over_http(question: str) -> list[dict]:
    """Send the question to the search API WEB_SEARCH_PROVIDER names and read the results.

    Args:
        question: The question, already checked.

    Returns:
        Up to WEB_SEARCH_MAX_RESULT results, each cut to WEB_SEARCH_RESULT_FIELD. A
        result whose URL cannot be read is left out (_remember_url).

    Raises:
        ValueError: When the reply holds no result list where the preset says.
        RuntimeError: When the endpoint asked for a wait over WEB_SEARCH_TIMEOUT_SEC or
            said to slow down, when the request failed, or when the reply is over
            WEB_SEARCH_MAX_BYTE or is not JSON.
    """
    endpoint = _search_endpoint()
    result_node = _at_path(_reply_of(endpoint, question), endpoint.result_path)
    if not isinstance(result_node, list):
        raise ValueError(
            f"the reply of {config.WEB_SEARCH_PROVIDER} holds no list at "
            f"{endpoint.result_path!r}, where its preset reads the results")
    result_list = [keep_dict for result_dict in result_node[:config.WEB_SEARCH_MAX_RESULT]
                   if (keep_dict := _keep_key_dict(result_dict))]
    logger.info("the endpoint returned %d result(s) of %d", len(result_list), len(result_node))
    return _remember_url(result_list)


def fetch(url: str) -> str:
    """Read one page of the web, through the page-reading API WEB_FETCH_PROVIDER names.

    The URL and the endpoint are checked here.

    Args:
        url: The page's URL. A URL no search of this process returned is refused and
            does not leave this machine. What is sent is the URL as the search
            returned it, without its fragment (_url_without_fragment), whatever
            fragment, case of the host or trailing slash the caller wrote.

    Returns:
        The page's text, whole.

    Raises:
        ValueError: When no search returned the URL, when WEB_FETCH_PROVIDER names no
            page-reading API, or when the reply holds no page, or no text in it, where
            the preset says.
        RuntimeError: When the endpoint asked for a wait over WEB_FETCH_TIMEOUT_SEC or
            said to slow down, when the request failed, when the reply is over
            WEB_FETCH_MAX_BYTE or is not JSON, or when the endpoint read no page for
            that URL.
    """
    search_url = _result_url_dict.get(url_key(url)) if url else None
    if not search_url:
        raise ValueError(UNKNOWN_URL_ERROR)
    if not can_fetch():
        raise ValueError("WEB_FETCH_PROVIDER names no page-reading API. Set it to one of "
                         f"{', '.join(config.WEB_FETCH_PRESET_DICT)}")
    endpoint = _fetch_endpoint()
    page_node = _at_path(_reply_of(endpoint, search_url), endpoint.result_path)
    # An endpoint that answers with a list of pages read gives the one asked for as its
    # only element, and an empty list when it could not read the page
    if isinstance(page_node, list):
        if not page_node:
            raise RuntimeError("the endpoint read no page at that URL")
        page_node = page_node[0]
    if not isinstance(page_node, dict):
        raise ValueError(
            f"the reply of {config.WEB_FETCH_PROVIDER} holds no page at "
            f"{endpoint.result_path!r}, where its preset reads the page")
    content = page_node.get(config.WEB_FETCH_CONTENT_FIELD)
    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"the page {config.WEB_FETCH_PROVIDER} returned holds no text under "
            f"{config.WEB_FETCH_CONTENT_FIELD!r}")
    logger.info("the endpoint read %d character(s) of the page", len(content))
    return content
