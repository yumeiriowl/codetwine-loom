"""The LLM call, made from outside the container.

A call comes here as one JSON-RPC request over a Unix socket or a TCP socket; this
process adds LLM_API_KEY and LLM_API_BASE, calls litellm, and sends back the reply's
text, its finish reason and its token counts. A call whose model, max_tokens or thinking
this process's own settings do not hold is refused before it is made
(_completion_refusal()).

It answers three more calls:

- refill() asks for the package cache to be filled again. The request carries nothing.
- search() asks one question of the web. The request carries the question alone, and its
  length is checked here.
- fetch() reads one page of the web. The request carries the page's URL alone, and it is
  checked here.

How many calls of each of the four one run may make is counted here (_Budget).

serve() is the server side, complete(), refill(), search() and fetch() the client side,
which send to the address config.LLM_BROKER holds.
"""

import errno
import json
import logging
import os
import re
import socket
import stat
import threading
import urllib.error
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field

import litellm

from codetwine_loom import config, providercall, tokencount, websearch
from codetwine_loom.textcut import CommandError, command_line

logger = logging.getLogger(__name__)

# The JSON-RPC methods served. COMPLETION takes the arguments litellm is called with,
# SEARCH takes one question, FETCH takes one page's URL, and REFILL takes nothing
COMPLETION = "completion"
REFILL = "refill"
SEARCH = "search"
FETCH = "fetch"

# The setting holding how many calls of each method one run may make
_LIMIT_DICT = {COMPLETION: "LLM_MAX_CALL",
               SEARCH: "WEB_SEARCH_MAX_CALL",
               FETCH: "WEB_FETCH_MAX_CALL",
               REFILL: "WARM_MAX_REFILL"}

# The methods that may make a model call: a completion, and a search the model makes
_MODEL_METHOD_TUPLE = (COMPLETION, SEARCH)

# The methods whose limit of 0 sets no limit rather than allowing none
_NO_LIMIT_TUPLE = (COMPLETION,)

# The variable that has litellm read the model table it ships with, and the variable
# that holds the model table entries of a broker's models, as JSON
LOCAL_MODEL_TABLE_ENV = providercall.LOCAL_MODEL_TABLE_ENV
MODEL_TABLE_ENV = "CTLOOM_MODEL_TABLE"

# The form of config.LLM_BROKER that names a Unix socket
UNIX_SCHEME = config.BROKER_UNIX_SCHEME

# The arguments a completion request may carry. A request carrying any other name is
# refused before the provider is called; api_key and api_base are the broker's own to
# add
COMPLETION_PARAM_TUPLE = ("model", "max_tokens", "num_retries", "messages",
                          providercall.TURN_REPLY_KEY, "thinking", "output_config")

# The one argument a search and a page read take, a string
_VALUE_KEY_DICT = {SEARCH: "question", FETCH: "url"}

# The setting holding the time limit of a search and of a page read, which bounds both
# the wait for the endpoint's turn and the request itself
_WEB_TIMEOUT_NAME_DICT = {SEARCH: "WEB_SEARCH_TIMEOUT_SEC", FETCH: "WEB_FETCH_TIMEOUT_SEC"}

# The arguments of a completion that carry its form of thinking, as
# llm.think_param_chain() builds them
_THINK_KEY_TUPLE = ("thinking", "output_config")

# What stands in a message where one of those values was
_HIDE_TEXT = "<the broker's own>"

# The fewest characters a value holds to be taken out of a message or a log record
_MIN_HIDE_CHAR = 8

# The names of this machine itself, which are taken out of no message
_LOOPBACK_HOST_TUPLE = ("localhost", "127.0.0.1")

# The proxy variable that names hosts rather than a proxy
_NO_PROXY_NAME = "NO_PROXY"

# The settings whose values this process puts into a call it makes for a caller
_OWN_SETTING_TUPLE = ("LLM_API_KEY", "LLM_API_BASE", "WEB_SEARCH_API_KEY",
                      "WEB_FETCH_API_KEY")

# The error a provider's refusal of the parameters travels as. A subclass litellm raises
# it as travels under this name too, but for CONTEXT_WINDOW
BAD_REQUEST = "BadRequestError"

# The one subclass that travels under its own name: a prompt refused for its length
CONTEXT_WINDOW = "ContextWindowExceededError"

# What a caller is told when it sent more than one request may be, or when this process
# is already answering as many callers as it answers at once
_TOO_LARGE = ("the request is over {limit} bytes, which is as large as one request may "
              "be (LLM_BROKER_MAX_REQUEST_BYTE)")
_TOO_MANY = ("this broker is already answering {limit} caller(s), which is as many as "
             "it answers at once (LLM_BROKER_MAX_CONNECTION)")

# JSON-RPC error codes. -32000 is the application's own range
_PARSE_ERROR = -32700
_METHOD_NOT_FOUND = -32601
_CALL_FAILURE = -32000


@dataclass(frozen=True)
class Message:
    """The reply's text.

    Attributes:
        content: What the model wrote, or None when the reply held no content block.
    """

    content: str | None


@dataclass(frozen=True)
class Choice:
    """One reply: its message content, its finish reason and the call's token counts.

    Attributes:
        message: The reply's text.
        finish_reason: "stop", "length" or whatever else the provider reported, and
            None when it reported nothing.
        usage_dict: The call's token counts, as usage_dict_of() returns them; empty
            when the reply carried none.
    """

    message: Message
    finish_reason: str | None
    usage_dict: dict = field(default_factory=dict)


def usage_dict_of(usage: object) -> dict:
    """Read one call's token counts off litellm's usage object.

    Args:
        usage: The response's usage, or None when the reply carried none.

    Returns:
        {"input_token", "output_token", "cache_read_token", "cache_write_token",
        providercall.ESTIMATE_KEY, providercall.RETRY_KEY}, each an int; 0 where the
        provider or providercall.completion() reported nothing.
    """
    usage_dict = tokencount.count_dict_of(usage)
    for key in (providercall.ESTIMATE_KEY, providercall.RETRY_KEY):
        value = getattr(usage, key, None)
        usage_dict[key] = value if isinstance(value, int) else 0
    return usage_dict


def endpoint(url: str) -> tuple[int, str | tuple[str, int]]:
    """Turn a broker address into the two arguments a socket is built and bound with.

    Examples:
        "unix:///run/ctloom/llm.sock" -> (AF_UNIX, "/run/ctloom/llm.sock")
        "tcp://broker:9000"        -> (AF_INET, ("broker", 9000))

    Args:
        url: The address, as config.LLM_BROKER holds it.

    Returns:
        (family, address), which socket.socket() and bind() / connect() take.

    Raises:
        ValueError: When url starts with neither scheme, when it names no path, or when
            a tcp:// address has no numeric port (config.broker_address_of()).
    """
    scheme, address = config.broker_address_of(url)
    return (socket.AF_UNIX if scheme == UNIX_SCHEME else socket.AF_INET), address


def _send(conn: socket.socket, message_dict: dict) -> None:
    """Write one JSON-RPC message, followed by the newline that ends it.

    json.dumps escapes the newlines inside the message: one line is one message however
    long the prompt is. Characters outside ASCII are sent as UTF-8, not escaped. A
    character UTF-8 cannot encode (half of a surrogate pair) is sent as its JSON escape.

    Args:
        conn: The connected socket.
        message_dict: The request or the reply.
    """
    conn.sendall(json.dumps(message_dict, ensure_ascii=False)
                 .encode("utf-8", "backslashreplace") + b"\n")


def _own_pair_list() -> list[tuple[str, str]]:
    """Return what this process puts into a call it makes for a caller, by name.

    Returns:
        (name, value) for each setting of _OWN_SETTING_TUPLE and each of the providers'
        own variables the models read (llm.provider_variable_tuple), the value
        without the blanks around it. A name with no value is left out.
    """
    from codetwine_loom import llm
    pair_list = [(name, getattr(config, name)) for name in _OWN_SETTING_TUPLE]
    pair_list += [(name, os.environ.get(name, "")) for name in llm.provider_variable_tuple()]
    return [(name, value.strip()) for name, value in pair_list
            if isinstance(value, str) and value.strip()]


def _hide_value_list(value_list: list[str]) -> list[str]:
    """Return the values that are taken out of a text, of the ones given.

    Args:
        value_list: The values this process puts into a call.

    Returns:
        Those of _MIN_HIDE_CHAR characters or more that are not a name of this machine
        itself (_LOOPBACK_HOST_TUPLE), each once, the longest first.
    """
    return sorted({value for value in value_list
                   if len(value) >= _MIN_HIDE_CHAR
                   and value.lower() not in _LOOPBACK_HOST_TUPLE},
                  key=len, reverse=True)


def short_key_name_list() -> list[str]:
    """Return the names of the credentials too short to be taken out of a message.

    Returns:
        The names of _own_pair_list() that mark a credential (llm.is_credential_name)
        and whose value holds fewer than _MIN_HIDE_CHAR characters.
    """
    from codetwine_loom import llm
    return [name for name, value in _own_pair_list()
            if llm.is_credential_name(name) and len(value) < _MIN_HIDE_CHAR]


def _endpoint_part_list(value: str) -> list[str]:
    """Return the parts of a URL that name its endpoint without the URL written whole.

    Examples:
        "https://me:pw@llm.corp.example:8443/v1"
            -> ["me:pw@llm.corp.example:8443", "llm.corp.example:8443",
                "llm.corp.example", "me:pw", "pw", "me"]
        "http://svc:Secr3t#Pass99@proxy.corp.example:3128"
            -> ["svc:Secr3t", "svc", "svc:Secr3t#Pass99", "Secr3t", "Pass99",
                "proxy.corp.example:3128", "proxy.corp.example"]
        "http://svc:secr[et]pw@llm.corp.example:8443/v1"
            -> ["svc:secr[et]pw", "svc", "secr[et]pw", "llm.corp.example:8443",
                "llm.corp.example"]
        "sk-0123456789" -> []

    Args:
        value: One value of _own_pair_list().

    Returns:
        For a value with "://", each once: the user, password, host and port as
        urlsplit() reads them together, the host with its port, the host alone (as
        written and as urlsplit() reads it), the user with the password, and the
        password, when urlsplit() reads a host; then what stands before the last "@"
        after the scheme, whole and cut at each ":", "/", "@", "#" and "?"; then the
        host with its port and the host alone (without "[" and "]"), as they stand after
        the last "@" of the part before the first "/", "?" or "#", and after the last
        "@" of the value cut at the first "/", "?" or "#" after it (a URL urlsplit()
        does not read included). [] for a value without "://".
    """
    scheme, _, rest = value.partition("://")
    if not scheme or not rest:
        return []
    user_text = rest.rpartition("@")[0]
    part_list = [user_text, *re.split(r"[:/@#?]", user_text)] if user_text else []
    for host_port in (re.split(r"[/?#]", rest)[0].rpartition("@")[2],
                      re.split(r"[/?#]", rest.rpartition("@")[2])[0]):
        host = (host_port[1:].partition("]")[0] if host_port.startswith("[")
                else host_port.rpartition(":")[0] or host_port)
        part_list += [host_port, host]
    try:
        part = urllib.parse.urlsplit(value)
        host = part.hostname or ""
    except ValueError:
        host = ""
    if host:
        netloc_user, _, host_text = part.netloc.rpartition("@")
        part_list = [part.netloc, host_text, host_text.rpartition(":")[0], host,
                     netloc_user, part.password or "", *part_list]
    return list(dict.fromkeys(text for text in part_list if text))


def _proxy_value_list() -> list[str]:
    """Return what the proxy variables of this process name their hosts by.

    Examples:
        HTTPS_PROXY=http://me:pw@proxy.corp.example:3128
            -> ["http://me:pw@proxy.corp.example:3128", "me:pw@proxy.corp.example:3128",
                "proxy.corp.example:3128", "proxy.corp.example", "me:pw", "pw", "me"]
        no_proxy=internal,.corp.example -> [".corp.example", ".corp.example"]

    Returns:
        Each entry of the variables config.proxy_name_list() names, a value split at
        its commas, followed by the parts _endpoint_part_list() gives for it; an entry
        that names no scheme is read as a host with or without a port. An entry of
        NO_PROXY that holds neither "." nor ":" is left out.
    """
    value_list = []
    for name in config.proxy_name_list():
        is_host_list = name.upper() == _NO_PROXY_NAME
        for entry in (text.strip() for text in os.environ[name].split(",")):
            if not entry or (is_host_list and "." not in entry and ":" not in entry):
                continue
            url = entry if "://" in entry else f"http://{entry}"
            value_list += [entry, *_endpoint_part_list(url)]
    return value_list


def _own_value_list() -> list[str]:
    """Return the values this process puts into a call it makes for a caller.

    They are the values of _own_pair_list(), the parts a value that is a URL names its
    endpoint by (_endpoint_part_list), what the proxy variables name their hosts by
    (_proxy_value_list), and the values of the headers and parameters of the search and
    the page read that carry one of them.

    Returns:
        The values _hide_value_list() keeps, the longest first.
    """
    own_list = [value for _, value in _own_pair_list()]
    secret_list = _hide_value_list(own_list + _proxy_value_list() + [
        text for value in own_list for text in _endpoint_part_list(value)])
    value_list = list(secret_list)
    for holder_dict in (config.WEB_SEARCH_HEADER, config.WEB_SEARCH_PARAM,
                        config.WEB_FETCH_HEADER, config.WEB_FETCH_PARAM):
        if isinstance(holder_dict, dict):
            value_list += [str(value).strip() for value in holder_dict.values()
                           if any(secret in str(value) for secret in secret_list)]
    return _hide_value_list(value_list)


def hide_own_value(text: str) -> str:
    """Return one message with what this process adds to a call taken out of it.

    The values are those _own_value_list() returns. A value of _MIN_HIDE_CHAR characters
    or more that is also ordinary text is taken out as well.

    Args:
        text: The message.

    Returns:
        The message, with each of those values replaced.
    """
    for value in _own_value_list():
        text = text.replace(value, _HIDE_TEXT)
    return text


class OwnValueFilter(logging.Filter):
    """A filter that takes what this process adds to a call out of each record.

    The record's message, the text of its exception and its stack are passed through
    hide_own_value(). The record is changed in place.
    """

    def __init__(self) -> None:
        """Start with no record in hand on any thread."""
        super().__init__()
        self._local = threading.local()

    def filter(self, record: logging.LogRecord) -> bool:
        """Take the values out of one record.

        A record logged while another one is in hand on the same thread is passed as
        it is.

        Args:
            record: The record, changed in place.

        Returns:
            True: every record is written.
        """
        if getattr(self._local, "is_busy", False):
            return True
        self._local.is_busy = True
        try:
            record.msg, record.args = hide_own_value(record.getMessage()), None
            if record.exc_info and not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = hide_own_value(record.exc_text)
            if record.stack_info:
                record.stack_info = hide_own_value(record.stack_info)
        finally:
            self._local.is_busy = False
        return True


def _error(ident: int | str | None, code: int, message: str,
           error_type: str = "") -> dict:
    """Build one JSON-RPC error reply.

    Args:
        ident: The request's id, or None when the request could not be read.
        code: The JSON-RPC error code.
        message: What went wrong. What this process adds to a call is taken out of it.
        error_type: The name of the exception the provider raised. Empty leaves it out.

    Returns:
        The reply, ready for _send().
    """
    error_dict = {"code": code, "message": hide_own_value(message)}
    if error_type:
        error_dict["data"] = {"error_type": error_type}
    return {"jsonrpc": "2.0", "id": ident, "error": error_dict}


def _result_reply_dict(ident: int | str | None, result: dict) -> dict:
    """Build one JSON-RPC reply that carries a result.

    Args:
        ident: The request's id.
        result: What the call gave.

    Returns:
        The reply, ready for _send().
    """
    return {"jsonrpc": "2.0", "id": ident, "result": result}


# How many causes of an exception are followed
_MAX_CAUSE_COUNT = 16


def _cause_list(error: BaseException) -> list[BaseException]:
    """Return an exception and what caused it, the exception first.

    Args:
        error: What was raised.

    Returns:
        error, then the __cause__ of each (its __context__ where it has no __cause__),
        at most _MAX_CAUSE_COUNT of them.
    """
    cause_list = []
    cause = error
    while cause is not None and len(cause_list) < _MAX_CAUSE_COUNT:
        cause_list.append(cause)
        cause = cause.__cause__ or cause.__context__
    return cause_list


# What the modules of config.HTTPX_MODULE_TUPLE raise for a request no response came
# back to
_REQUEST_ERROR_TUPLE = tuple(module.RequestError for module in config.HTTPX_MODULE_TUPLE)


def _is_no_answer(cause: BaseException) -> bool:
    """Say whether one cause shows that a request ended with no HTTP response.

    Args:
        cause: One exception of _cause_list().

    Returns:
        True for the failure of a system call (an OSError that carries an errno), for
        what an HTTP client raises for a request no response came back to (the
        RequestError of a module of config.HTTPX_MODULE_TUPLE), and for a connection
        urllib could not make in time (a urllib.error.URLError whose reason is a
        TimeoutError).
    """
    return isinstance(cause, _REQUEST_ERROR_TUPLE) \
        or (isinstance(cause, OSError) and cause.errno is not None) \
        or (isinstance(cause, urllib.error.URLError)
            and isinstance(cause.reason, TimeoutError))


def _cause_name(cause: BaseException) -> str:
    """Return the name a cause is told by.

    Examples:
        ConnectionRefusedError(111, "...") -> "ConnectionRefusedError"
        OSError(113, "...")                -> "OSError EHOSTUNREACH"

    Args:
        cause: One exception of _cause_list().

    Returns:
        The name of its class, followed by the name of its errno when the class is
        OSError itself.
    """
    name = type(cause).__name__
    if type(cause) is OSError and cause.errno in errno.errorcode:
        name += f" {errno.errorcode[cause.errno]}"
    return name


def _route_text() -> str:
    """Say which of the variables an HTTP client reads its route from this process holds.

    Examples:
        HTTPS_PROXY and SSL_CERT_FILE set
            -> "HTTPS_PROXY, SSL_CERT_FILE set where the call was made"
        none set
            -> "none of HTTPS_PROXY, HTTP_PROXY, ALL_PROXY, NO_PROXY, SSL_CERT_FILE set
                where the call was made"

    Returns:
        The names config.proxy_name_list() gives, followed by config.CA_FILE_ENV when
        it is set and not empty, or every name read when none is. No value is in it.
    """
    name_list = config.proxy_name_list()
    if os.environ.get(config.CA_FILE_ENV):
        name_list.append(config.CA_FILE_ENV)
    if not name_list:
        name_list = ["none of " + ", ".join((*config.PROXY_NAME_TUPLE,
                                             config.CA_FILE_ENV))]
    return f"{', '.join(name_list)} set where the call was made"


def failure_text(error: BaseException) -> str:
    """Return what a caller is told of an exception a call to an endpoint ended with.

    An exception one of whose causes _is_no_answer() came from a request the endpoint
    sent no HTTP response to: the connection was not made, the name was not found, the
    TLS handshake failed, a proxy turned a model call down, or the response did not
    arrive. Of such an exception the caller is told the exception's name, that the
    endpoint gave no answer, the name of the last cause and what _route_text() says, and
    nothing of its text. A proxy that turns the tunnel of a search or a page read down
    is told in the text urllib gives it.

    Examples:
        litellm.InternalServerError("... Cannot connect to host 10.1.2.3:8443 ...")
            caused by ConnectionRefusedError, with HTTPS_PROXY set
            -> "InternalServerError: the endpoint gave no answer
                (ConnectionRefusedError; HTTPS_PROXY set where the call was made)"
        RuntimeError("401: the key was refused") -> "401: the key was refused"

    Args:
        error: What the call raised.

    Returns:
        The text.
    """
    cause_list = _cause_list(error)
    if any(_is_no_answer(cause) for cause in cause_list):
        return (f"{type(error).__name__}: the endpoint gave no answer "
                f"({_cause_name(cause_list[-1])}; {_route_text()})")
    return str(error)


def _error_type(error: BaseException) -> str:
    """Return the name an exception travels under in an error reply.

    Args:
        error: What the provider raised.

    Returns:
        CONTEXT_WINDOW for a prompt refused for its length, BAD_REQUEST for any other
        refusal of the parameters, whichever subclass litellm raised it as; the
        exception's own class name otherwise.
    """
    if isinstance(error, litellm.ContextWindowExceededError):
        return CONTEXT_WINDOW
    return BAD_REQUEST if isinstance(error, litellm.BadRequestError) else type(error).__name__


# == The client side ==========================================

def complete(param_dict: dict) -> Choice:
    """Send one completion to the broker and return the reply.

    The credentials are not in param_dict and are not sent.

    Args:
        param_dict: The keyword arguments litellm.completion takes, as make_llm() built
            them: the names in COMPLETION_PARAM_TUPLE and no other.

    Returns:
        The reply's first choice.

    Raises:
        litellm.ContextWindowExceededError: When the provider refused the prompt for its
            length.
        litellm.BadRequestError: When the provider refused the parameters.
        RuntimeError: When the broker cannot be reached, answers nothing within
            LLM_BROKER_TIMEOUT_SEC, or reports any other failure.
        ValueError: When config.LLM_BROKER is not an address endpoint() understands.
    """
    return _choice_of(_call(COMPLETION, param_dict))


def _exchange(conn: socket.socket, request_dict: dict) -> bytes:
    """Send one request over a connection and return the line that came back.

    When sending fails part-way, the answer is read all the same. The failure to send is
    raised only when there is none.

    Args:
        conn: The connected socket.
        request_dict: The request.

    Returns:
        The reply's line, or b"" when the broker closed the connection without one.

    Raises:
        TimeoutError: When the request cannot be sent, or no reply arrives, within
            the socket's time limit.
        OSError: When the request could not be sent and no reply could be read, or
            reading the reply of a request that was sent fails.
    """
    send_error = None
    try:
        _send(conn, request_dict)
    except TimeoutError:
        raise
    except OSError as e:
        send_error = e
    try:
        with conn.makefile("rb") as reader:
            line = reader.readline()
    except OSError:
        if send_error is None:
            raise
        line = b""
    if send_error is not None and not line:
        raise send_error
    return line


def _call(method: str, param_dict: dict) -> dict:
    """Send one request to the broker and return what came back.

    The reply is waited for LLM_BROKER_TIMEOUT_SEC, and a search or a page read twice
    its own time limit (_WEB_TIMEOUT_NAME_DICT) longer. A reply the broker wrote before
    it had read the whole request is read too (_exchange).

    Args:
        method: COMPLETION, REFILL, SEARCH or FETCH.
        param_dict: The request's params.

    Returns:
        The reply's "result".

    Raises:
        litellm.ContextWindowExceededError: When the provider refused the prompt for its
            length.
        litellm.BadRequestError: When the provider refused the parameters.
        RuntimeError: When the broker cannot be reached, answers nothing within the wait
            above, or reports any other failure.
        ValueError: When config.LLM_BROKER is not an address endpoint() understands.
    """
    family, address = endpoint(config.LLM_BROKER)
    request_dict = {"jsonrpc": "2.0", "id": 1, "method": method, "params": param_dict}
    web_timeout_name = _WEB_TIMEOUT_NAME_DICT.get(method, "")
    wait = config.LLM_BROKER_TIMEOUT_SEC + (2 * getattr(config, web_timeout_name)
                                            if web_timeout_name else 0)
    with socket.socket(family, socket.SOCK_STREAM) as conn:
        conn.settimeout(wait)
        try:
            conn.connect(address)
            line = _exchange(conn, request_dict)
        except TimeoutError as e:
            web_term = f" + 2 x {web_timeout_name}" if web_timeout_name else ""
            raise RuntimeError(
                f"the call through the broker at {config.LLM_BROKER} got no reply within "
                f"{wait} s, the time every attempt of it and the waits between them may "
                f"take (LLM_CALL_TIMEOUT_SEC x (2 x LLM_MAX_RETRY + 1){web_term})") from e
        except OSError as e:
            raise CommandError(
                f"the broker at {config.LLM_BROKER} did not answer ({e}). Start it "
                f"with:", command_line("codetwine-loom", "broker", "--endpoint",
                                       config.LLM_BROKER)) from e
    if not line:
        raise RuntimeError(f"the broker at {config.LLM_BROKER} closed the connection "
                           f"without answering")
    try:
        reply_dict = json.loads(line)
    except (ValueError, RecursionError) as e:
        raise RuntimeError(f"the broker at {config.LLM_BROKER} answered with something "
                           f"that is not JSON: {e}") from e
    if "error" in reply_dict:
        _raise(reply_dict["error"], param_dict)
    return reply_dict.get("result") or {}


def refill() -> dict:
    """Ask the broker to fill the package cache again.

    The request carries nothing.

    Returns:
        {"ok": True, "refills_left": int, "skip": bool}, the last two as the broker
        returned them, or {"ok": False, "error": str, "refills_left": 0} when the call
        failed. It never raises.
    """
    try:
        result_dict = _call(REFILL, {})
    except (OSError, RuntimeError, ValueError) as e:
        logger.info("the cache was not filled again: %s", e)
        return {"ok": False, "error": str(e), "refills_left": 0}
    return {"ok": True, "refills_left": result_dict.get("refills_left", 0),
            "skip": bool(result_dict.get("skip"))}


def search(question: str) -> list[dict]:
    """Ask the broker one question of the web.

    Args:
        question: One short question.

    Returns:
        The results, each holding title, url and content as far as the search found
        them.

    Raises:
        RuntimeError: When the broker cannot be reached, refuses the question, has no
            endpoint to search, or the search failed.
        ValueError: When config.LLM_BROKER is not an address endpoint() understands.
    """
    return _call(SEARCH, {"question": question}).get("results", [])


def fetch(url: str) -> str:
    """Ask the broker to read one page of the web.

    Args:
        url: The page's URL.

    Returns:
        The page's text, whole.

    Raises:
        RuntimeError: When the broker cannot be reached, refuses the URL, has no
            endpoint to read through, or the read failed.
        ValueError: When config.LLM_BROKER is not an address endpoint() understands.
    """
    return _call(FETCH, {"url": url}).get("content", "")


def _choice_of(result_dict: dict) -> Choice:
    """Turn the broker's result into a Choice.

    Args:
        result_dict: {"content", "finish_reason", "usage"}, as _call_the_provider()
            returns it.

    Returns:
        The reply's first choice.
    """
    return Choice(message=Message(content=result_dict.get("content")),
                  finish_reason=result_dict.get("finish_reason"),
                  usage_dict=result_dict.get("usage") or {})


def _raise(error_dict: dict, param_dict: dict) -> None:
    """Raise the exception the broker's error stands for.

    Args:
        error_dict: The JSON-RPC error object.
        param_dict: The request's params, which carry the model name the exception holds.

    Raises:
        litellm.ContextWindowExceededError: When the provider refused the prompt for its
            length.
        litellm.BadRequestError: When the provider refused the parameters.
        RuntimeError: For every other failure.
    """
    message = error_dict.get("message", "")
    error_type = (error_dict.get("data") or {}).get("error_type")
    model = param_dict.get("model", "")
    if error_type == CONTEXT_WINDOW:
        raise litellm.ContextWindowExceededError(message=message, model=model,
                                                 llm_provider=model.partition("/")[0])
    if error_type == BAD_REQUEST:
        raise litellm.BadRequestError(message=message, model=model,
                                      llm_provider=model.partition("/")[0])
    raise RuntimeError(f"the broker did not make the call: {message}")


# == The server side ==========================================

def _limit(method: str) -> int:
    """Return how many calls of method one run may make.

    Args:
        method: The JSON-RPC method.

    Returns:
        The setting's value, or 0 for a method no setting names.
    """
    return getattr(config, _LIMIT_DICT.get(method, ""), 0)


class _Budget:
    """How many calls of each method this process still allows, and how to refill.

    Nothing resets the counts spend() keeps against the settings of _LIMIT_DICT.
    """

    def __init__(self,
                 refill_cache: Callable[[Callable[[], object]], dict] | None = None,
                 token_left: int = 0, token_out: str = "") -> None:
        """Start every counter at 0.

        The limits themselves are read from the settings as each call is served: a
        setting rewritten at run time is followed.

        Args:
            refill_cache: What one refill request runs, or None when the broker serves
                none. It is given what takes one refill out of the budget.
            token_left: How many tokens the model calls of this process and of the
                processes writing their counts beside token_out may use between them
                (tokencount.total_beside). A completion or a search asked for once they
                have used that many is refused. 0 or less sets no limit.
            token_out: The file this process's token counts are written to, or "" for
                none, which counts this process's model calls alone.
        """
        self._refill_cache = refill_cache
        self._token_left = token_left
        self._token_out = token_out
        self._use_count_dict: dict[str, int] = {}
        self._lock = threading.Lock()
        # Its own lock: one refill runs while it is held, and spend() takes _lock inside
        self._refill_lock = threading.Lock()

    def remain_count(self, method: str) -> int:
        """Return how many calls of method are still allowed.

        Args:
            method: The JSON-RPC method.

        Returns:
            The limit minus what has been spent, never below 0.
        """
        return max(0, _limit(method) - self._use_count_dict.get(method, 0))

    def spend(self, method: str) -> int:
        """Take one call of method out of the budget.

        Args:
            method: The JSON-RPC method.

        Returns:
            How many calls of method are left after this one, or 0, with nothing taken,
            for a method of _NO_LIMIT_TUPLE whose limit is 0 or less.

        Raises:
            RuntimeError: When the budget for method is spent, or for a completion or a
                search when the model calls counted against token_left have used that
                many tokens.
        """
        if method in _MODEL_METHOD_TUPLE and self._token_left > 0:
            use_count = tokencount.total_beside(self._token_out)
            if use_count >= self._token_left:
                raise RuntimeError(
                    f"the model calls of this run have used {use_count} token(s), "
                    f"which reaches the {self._token_left} its containers may use "
                    f"between them (MAX_TOKEN_PER_RUN)")
        if method in _NO_LIMIT_TUPLE and _limit(method) <= 0:
            return 0
        with self._lock:
            self._require_remain(method)
            self._use_count_dict[method] = self._use_count_dict.get(method, 0) + 1
            return self.remain_count(method)

    def _require_remain(self, method: str) -> None:
        """Raise when no call of method is left in the budget.

        Args:
            method: The JSON-RPC method.

        Raises:
            RuntimeError: When the budget for method is spent.
        """
        if self.remain_count(method) <= 0:
            raise RuntimeError(f"this broker has served as many {method} calls as "
                               f"one run may ({_LIMIT_DICT.get(method, method)}="
                               f"{_limit(method)})")

    def refill(self) -> dict:
        """Fill the cache once, if it lacks what it is asked for and a refill is left.

        The lock is held for the whole run: two connections asking at once fill the
        cache one after the other, not at the same time. The refill is counted when
        refill_cache calls the function it is given, whether the filling then ends
        cleanly or not. Where the limit of refills is 0, what the request runs is not
        started.

        Returns:
            {"refills_left": int, "skip": bool}, "skip" as refill_cache returned it.

        Raises:
            RuntimeError: When the broker serves no refill, when the budget is spent, or
                when the run that fills the cache did not finish cleanly.
        """
        with self._refill_lock:
            if self._refill_cache is None:
                raise RuntimeError("this broker fills no cache. Start it with the "
                                   "repository to fill it from")
            if _limit(REFILL) <= 0:
                self._require_remain(REFILL)
            result_dict = self._refill_cache(lambda: self.spend(REFILL))
        status = result_dict.get("exit_status")
        if status:
            raise RuntimeError(f"filling the cache ended with exit status {status}")
        return {"refills_left": self.remain_count(REFILL),
                "skip": bool(result_dict.get("skip"))}


def _table_key(model: str) -> str:
    """Return the key litellm's model table holds a model's entry under.

    Args:
        model: A model name in litellm format.

    Returns:
        The key litellm.get_model_info() found the entry under, or "" when it found
        none, whatever it raises. "" for a model litellm is not asked about
        (providercall.model_info_of).
    """
    try:
        return str(providercall.model_info_of(model).get("key") or "")
    except Exception:                     # noqa: BLE001 - an unknown model raises
        return ""


def model_table_text() -> str:
    """Return the entries litellm's model table holds for this process's models, as JSON.

    The models are the values of the settings named *_MODEL
    (config.setting_value_dict()). An entry is looked up under the model as the setting
    writes it, under that value without its provider (the part up to the first "/"), and
    under the key litellm itself finds the model's entry by (_table_key).

    Returns:
        {the entry's key: the entry} as JSON; "{}" when the table holds none of them.
    """
    entry_dict = {}
    for model in sorted({value for value in config.setting_value_dict("_MODEL").values()
                         if value and isinstance(value, str)}):
        for key in (model, model.partition("/")[2], _table_key(model)):
            if key in litellm.model_cost:
                entry_dict[key] = litellm.model_cost[key]
    return json.dumps(entry_dict)


def load_model_table(text: str) -> None:
    """Put the entries model_table_text() returned into litellm's model table.

    Each entry is put into the table under its key in text, then litellm is told of them
    (litellm.register_model) and they are added to the lists litellm names the provider
    of a model by (litellm.add_known_models). A text that is not a JSON object of
    entries, or that litellm does not take, is logged and leaves the table as it is.

    Args:
        text: The JSON text, or "" for none.
    """
    if not text:
        return
    try:
        entry_dict = json.loads(text)
        if not isinstance(entry_dict, dict) or not all(
                isinstance(entry, dict) for entry in entry_dict.values()):
            raise ValueError("not a JSON object of entries")
        if entry_dict:
            # Each entry is put in under its own key first. litellm.register_model()
            # then finds it under that key and puts it into no other entry
            for key, entry in entry_dict.items():
                litellm.model_cost[key] = dict(entry)
            litellm.register_model(entry_dict)
            litellm.add_known_models(entry_dict)
    except Exception as e:                # noqa: BLE001 - whatever litellm raises
        logger.warning("%s could not be read, so the model table is the one litellm "
                       "ships with: %s", MODEL_TABLE_ENV, e)


def serve(url: str = "",
          refill_cache: Callable[[Callable[[], object]], dict] | None = None,
          token_out: str = "", token_left: int = 0) -> None:
    """Answer calls until the process is stopped.

    A Unix socket left by an earlier broker is taken away before the address is bound,
    and the one this call makes is taken away when it returns. The credentials too short
    to be taken out of a message are named in one warning first (short_key_name_list).
    With token_out, the token counts of the model calls this process has made are
    written there after every completion and every search it answers.

    Args:
        url: Where to listen, in the form endpoint() takes. Empty uses
            config.LLM_BROKER.
        refill_cache: What one refill request runs, given what takes one refill out of
            the budget and returning a dict with "exit_status" and "skip". None refuses
            the request. At most config.WARM_MAX_REFILL refills are counted for the life
            of this process.
        token_out: The file this process's token counts are written to
            (tokencount.write), as an absolute path. Empty writes none.
        token_left: How many tokens this process's model calls and those of the
            processes writing their counts beside token_out may use between them, after
            which a completion or a search is refused (_Budget). 0 or less sets no
            limit.

    Raises:
        ValueError: When no address is given, or it is one endpoint() does not
            understand.
        OSError: When the address cannot be bound, or the path of a Unix socket holds
            something that is not a socket.
    """
    address_url = url or config.LLM_BROKER
    if not address_url:
        raise ValueError("no address to listen on. Pass --endpoint, or set LLM_BROKER")
    family, address = endpoint(address_url)
    short_name_list = short_key_name_list()
    if short_name_list:
        logger.warning("%s holds fewer than %d characters, so its value is not taken "
                       "out of the failure messages the broker sends back and logs",
                       ", ".join(short_name_list), _MIN_HIDE_CHAR)
    # Taken before a thread is started and given back when it ends. A caller over the
    # cap is answered with an error and closed rather than queued. A connection that
    # sends nothing holds a slot for at most LLM_BROKER_TIMEOUT_SEC
    caller_slot = threading.Semaphore(config.LLM_BROKER_MAX_CONNECTION)
    if family == socket.AF_UNIX:
        _clear_socket_file(address)
    # One budget for the whole process
    budget = _Budget(refill_cache, token_left, token_out)
    with socket.socket(family, socket.SOCK_STREAM) as server:
        if family == socket.AF_INET:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(address)
        server.listen(config.LLM_BROKER_MAX_CONNECTION)
        logger.info("the broker is listening on %s", address_url)
        try:
            while True:
                conn, _ = server.accept()
                if not caller_slot.acquire(blocking=False):
                    _refuse(conn, _TOO_MANY.format(
                        limit=config.LLM_BROKER_MAX_CONNECTION))
                    continue
                threading.Thread(target=_serve_one,
                                 args=(conn, budget, caller_slot, token_out),
                                 daemon=True).start()
        except KeyboardInterrupt:
            logger.info("the broker was stopped")
        finally:
            if family == socket.AF_UNIX:
                _clear_socket_file(address)


def _clear_socket_file(path: str) -> None:
    """Take away the socket file at path, saying so when it cannot be taken away.

    Only a socket is taken away. A path that holds nothing is left as it is.

    Args:
        path: The socket's path.

    Raises:
        OSError: When what is at path is not a socket. It is left where it is.
    """
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    except OSError as e:
        logger.warning("the socket at %s could not be taken away (%s)", path, e)
        return
    if not stat.S_ISSOCK(mode):
        raise OSError(f"{path} is there and is not a socket, so it is not taken away. "
                      f"Pass an --endpoint whose path is free, or remove it")
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("the socket at %s could not be taken away (%s)", path, e)


def _refuse(conn: socket.socket, message: str) -> None:
    """Answer one connection with an error and close it.

    It is the answer to a connection that is never served: nothing is read from it, and
    the reply carries no request id.

    Args:
        conn: The connected socket, closed when this returns.
        message: What went wrong.
    """
    logger.warning("a caller was refused: %s", message)
    with conn:
        _reply(conn, _error(None, _CALL_FAILURE, message))


def _serve_one(conn: socket.socket, budget: _Budget,
               caller_slot: threading.Semaphore | None = None,
               token_out: str = "") -> None:
    """Answer every request on one connection, then close it.

    The connection is given LLM_BROKER_TIMEOUT_SEC, and is closed when nothing arrives
    within it. A request over LLM_BROKER_MAX_REQUEST_BYTE is answered with an error
    rather than read and the connection closed, a line that cannot be parsed is answered
    with an error and the connection closed, and a request whose answer could not be
    built or written is answered with an error too (_fail_reply).

    Args:
        conn: The connected socket, closed when this returns.
        budget: What this process has left to serve.
        caller_slot: The count of callers being answered, given back when this returns.
            None counts nothing.
        token_out: The file this process's token counts are written to after each
            request that may have made a model call, or "" for none.
    """
    limit = config.LLM_BROKER_MAX_REQUEST_BYTE
    try:
        with conn, conn.makefile("rb") as reader:
            conn.settimeout(config.LLM_BROKER_TIMEOUT_SEC)
            while True:
                # readline() stops at the limit: a line that came back without a newline
                # is either the last of the connection or longer than a request may be
                line = reader.readline(limit + 1)
                if not line:
                    return
                if len(line) > limit:
                    message = _TOO_LARGE.format(limit=limit)
                    logger.warning("a caller was refused: %s", message)
                    _reply(conn, _error(None, _CALL_FAILURE, message))
                    return
                if not line.strip():
                    continue
                # A line that is not UTF-8, nests deeper than the parser follows or
                # holds a number it does not read is not JSON either
                try:
                    request_dict = json.loads(line)
                except (ValueError, RecursionError) as e:
                    _reply(conn, _error(None, _PARSE_ERROR,
                                        f"the request is not JSON: {e}"))
                    return
                try:
                    reply_dict = answer(request_dict, budget)
                    # The counts are written before the reply is sent
                    if token_out and isinstance(request_dict, dict) \
                            and request_dict.get("method") in _MODEL_METHOD_TUPLE:
                        tokencount.write(token_out)
                    is_open = _reply(conn, reply_dict)
                except OSError:
                    raise
                except Exception as e:    # noqa: BLE001 - whatever the request raised
                    is_open = _reply(conn, _fail_reply(request_dict, e))
                if not is_open:
                    return
    except OSError as e:
        logger.info("the connection ended before it was answered (%s)", e)
    finally:
        if caller_slot is not None:
            caller_slot.release()


def _fail_reply(request_dict: object, error: Exception) -> dict:
    """Build the reply to a request whose answer could not be built or written.

    Args:
        request_dict: The parsed request.
        error: What was raised.

    Returns:
        An error reply carrying the request's id and the exception's name and message.
    """
    logger.warning("a request could not be answered: %s: %s", type(error).__name__,
                   hide_own_value(str(error))[-config.LOG_EXCERPT_CHAR:])
    ident = request_dict.get("id") if isinstance(request_dict, dict) else None
    return _error(ident, _CALL_FAILURE,
                  f"the broker could not answer the request: {type(error).__name__}: "
                  f"{error}", type(error).__name__)


def _reply(conn: socket.socket, message_dict: dict) -> bool:
    """Write one reply, saying so in the log when there is nobody left to write it to.

    A reply that cannot be written is logged, and False comes back in place of an
    exception.

    Args:
        conn: The connected socket.
        message_dict: The reply.

    Returns:
        True when it was written, False when the connection is gone.
    """
    try:
        _send(conn, message_dict)
    except OSError as e:
        logger.info("the caller was gone before the reply could be sent (%s)", e)
        return False
    return True


def answer(request_dict: dict, budget: _Budget) -> dict:
    """Make the call one request asks for and build the reply.

    Args:
        request_dict: The parsed JSON-RPC request.
        budget: What this process has left to serve.

    Returns:
        The reply, holding "result" or "error".
    """
    if not isinstance(request_dict, dict):
        return _error(None, _PARSE_ERROR, "the request is not a JSON object")
    ident = request_dict.get("id")
    method = request_dict.get("method")
    if method == REFILL:
        return _answer_refill(ident, budget)
    if method not in (COMPLETION, SEARCH, FETCH):
        return _error(ident, _METHOD_NOT_FOUND,
                      f"{method!r} is not a method of this broker. They are "
                      f"{COMPLETION!r}, {SEARCH!r}, {FETCH!r} and {REFILL!r}")
    param_dict = request_dict.get("params") or {}
    # A request this broker would not serve is refused before it is counted
    refusal = (_completion_refusal(param_dict) if method == COMPLETION
               else _value_refusal(method, param_dict))
    if refusal:
        logger.warning("the %s call was refused: %s", method,
                       refusal[:config.LOG_EXCERPT_CHAR])
        return _error(ident, _CALL_FAILURE, refusal)
    # Counted before the work is done
    try:
        budget.spend(method)
    except RuntimeError as e:
        logger.warning("the %s call was refused: %s", method, e)
        return _error(ident, _CALL_FAILURE, str(e), type(e).__name__)
    if method == SEARCH:
        return _answer_search(ident, param_dict)
    if method == FETCH:
        return _answer_fetch(ident, param_dict)
    return _answer_completion(ident, param_dict)


def _value_refusal(method: str, param_dict: object) -> str:
    """Say why this broker would not make a search or a page read, from its params.

    Args:
        method: SEARCH or FETCH.
        param_dict: The request's params, which hold the one string the method takes
            (_VALUE_KEY_DICT).

    Returns:
        What the caller is told, or "" when the request may be served.
    """
    if not isinstance(param_dict, dict):
        return f"the params of a {method} are not a JSON object"
    key = _VALUE_KEY_DICT[method]
    if not isinstance(param_dict.get(key), str):
        return f"a {method} takes {key!r} as a string"
    return ""


def _completion_refusal(param_dict: object) -> str:
    """Say why this broker would not make a completion, from the request's params.

    What it makes is decided by its own settings, config.SECRET_NAME_TUPLE left out
    (_setting_value_list): the model is the value of a setting named *_MODEL, max_tokens
    is a whole number from 1 to the largest setting named *_MAX_TOKEN, and thinking and
    output_config together are one of the forms llm.think_param_chain() builds for that
    max_tokens, or neither. The messages are checked by _message_refusal().

    Args:
        param_dict: The request's params.

    Returns:
        What the caller is told, or "" when the call may be made.
    """
    # llm imports this module
    from codetwine_loom import llm

    if not isinstance(param_dict, dict):
        return "the params of a completion are not a JSON object"
    refuse_name_list = sorted(repr(name) for name in param_dict
                              if name not in COMPLETION_PARAM_TUPLE)
    if refuse_name_list:
        return (f"the call carried {', '.join(refuse_name_list)}. A completion takes "
                f"{', '.join(COMPLETION_PARAM_TUPLE)} and nothing else; the broker "
                f"holds the credentials and adds them itself")
    model_set = set(_setting_value_list("_MODEL"))
    model = param_dict.get("model")
    if not isinstance(model, str) or model not in model_set:
        return (f"the call asked for the model {model!r}, which no *_MODEL setting of "
                f"this broker holds. It calls {', '.join(sorted(model_set))}")
    max_token_limit = max((value for value in _setting_value_list("_MAX_TOKEN")
                           if isinstance(value, int)), default=0)
    max_token = param_dict.get("max_tokens")
    if type(max_token) is not int or not 0 < max_token <= max_token_limit:
        return (f"the call asked for max_tokens={max_token!r}. This broker takes a "
                f"whole number from 1 to {max_token_limit}, the largest *_MAX_TOKEN "
                f"setting it holds")
    think_dict = {key: param_dict[key] for key in _THINK_KEY_TUPLE if key in param_dict}
    form_list = [form for level in config.THINK_LEVEL_TUPLE
                 for form in llm.think_param_chain(level, max_token)]
    if think_dict not in form_list:
        return (f"the call asked for thinking {think_dict}, which is not a form "
                f"THINK_LEVEL builds for max_tokens={max_token}")
    return _message_refusal(param_dict.get("messages"))


# The keys a message and a text block may hold
_MESSAGE_KEY_SET = frozenset({"role", "content"})
_TEXT_BLOCK_KEY_SET = frozenset({"type", "text", "cache_control"})


def _message_refusal(message_list: object) -> str:
    """Say why this broker would not send the messages of a completion.

    What it sends is one message: {"role": "user", "content": a string, or a list of
    text blocks}. A text block is {"type": "text", "text": a string}, with or without a
    "cache_control" equal to llm.CACHE_CONTROL_DICT.

    Args:
        message_list: The request's "messages".

    Returns:
        What the caller is told, or "" when the messages may be sent.
    """
    from codetwine_loom import llm

    refusal = ("the call's messages are not one message of the role \"user\" whose "
               "content is a string or a list of {\"type\": \"text\", \"text\": str} "
               "blocks, each with or without cache_control "
               f"{json.dumps(llm.CACHE_CONTROL_DICT)}")
    if not isinstance(message_list, list) or len(message_list) != 1:
        return refusal
    message = message_list[0]
    if not isinstance(message, dict) or set(message) != _MESSAGE_KEY_SET \
            or message["role"] != "user":
        return refusal
    content = message["content"]
    if isinstance(content, str):
        return ""
    if not isinstance(content, list):
        return refusal
    for block in content:
        if not isinstance(block, dict) or not set(block) <= _TEXT_BLOCK_KEY_SET \
                or block.get("type") != "text" or not isinstance(block.get("text"), str) \
                or block.get("cache_control", llm.CACHE_CONTROL_DICT) \
                != llm.CACHE_CONTROL_DICT:
            return refusal
    return ""


def _setting_value_list(suffix: str) -> list:
    """Return the values of the settings named *suffix but config.SECRET_NAME_TUPLE.

    Args:
        suffix: The end of the setting names, e.g. "_MODEL".

    Returns:
        The non-empty values, from config.setting_value_dict(), of every name not in
        config.SECRET_NAME_TUPLE.
    """
    return [value for name, value in config.setting_value_dict(suffix).items()
            if value and name not in config.SECRET_NAME_TUPLE]


def _answer_completion(ident: int | str | None, param_dict: dict) -> dict:
    """Make one model call and build the reply.

    Args:
        ident: The request's id.
        param_dict: The keyword arguments litellm.completion takes, which
            _completion_refusal() has let through.

    Returns:
        The reply, holding "result" or "error".
    """
    # Every provider failure, whichever of litellm's exceptions it is, comes back as one
    # error reply, and the broker answers the next call
    try:
        result_dict = _call_the_provider(param_dict)
    except Exception as e:
        failure = failure_text(e)
        logger.info("the call failed: %s: %s. model=%s",
                    type(e).__name__,
                    hide_own_value(failure)[-config.LOG_EXCERPT_CHAR:],
                    param_dict.get("model", ""))
        return _error(ident, _CALL_FAILURE, failure, _error_type(e))
    return _result_reply_dict(ident, result_dict)


def _answer_search(ident: int | str | None, param_dict: dict) -> dict:
    """Search the web for one question and build the reply.

    Args:
        ident: The request's id.
        param_dict: {"question": str}.

    Returns:
        The reply, holding "result" or "error".
    """
    # A search that could not be made comes back as one error reply, and this process
    # carries on
    try:
        result_list = websearch.search(param_dict["question"])
    except (ValueError, RuntimeError) as e:
        failure = failure_text(e)
        logger.warning("the search was not made: %s: %s",
                       type(e).__name__, hide_own_value(failure))
        return _error(ident, _CALL_FAILURE, failure, type(e).__name__)
    return _result_reply_dict(ident, {"results": result_list})


def _answer_fetch(ident: int | str | None, param_dict: dict) -> dict:
    """Read one page of the web and build the reply.

    Args:
        ident: The request's id.
        param_dict: {"url": str}.

    Returns:
        The reply, holding "result" or "error".
    """
    # A page that could not be read comes back as one error reply, and this process
    # carries on
    try:
        content = websearch.fetch(param_dict["url"])
    except (ValueError, RuntimeError) as e:
        failure = failure_text(e)
        logger.warning("the page was not read: %s: %s",
                       type(e).__name__, hide_own_value(failure))
        return _error(ident, _CALL_FAILURE, failure, type(e).__name__)
    return _result_reply_dict(ident, {"content": content})


def _answer_refill(ident: int | str | None, budget: _Budget) -> dict:
    """Fill the package cache again and build the reply.

    The request's params are not read.

    Args:
        ident: The request's id.
        budget: What this process has left to serve. One holding no refill_cache
            refuses.

    Returns:
        The reply, holding "result" or "error".
    """
    logger.info("asked by the run to fill the package cache again")
    # A refill that could not happen comes back as one error reply
    try:
        result_dict = budget.refill()
    except Exception as e:
        logger.warning("the cache was not filled again: %s: %s",
                       type(e).__name__, hide_own_value(str(e)))
        return _error(ident, _CALL_FAILURE, str(e), type(e).__name__)
    if result_dict["skip"]:
        logger.info("the cache already holds what the run's dependency files fetch. "
                    "%d refill(s) left", result_dict["refills_left"])
    else:
        logger.info("the cache was filled again. %d refill(s) left",
                    result_dict["refills_left"])
    return _result_reply_dict(ident, result_dict)


def _call_the_provider(param_dict: dict) -> dict:
    """Call litellm with the credentials this process holds.

    Each attempt is given LLM_CALL_TIMEOUT_SEC and made again at most this process's
    LLM_MAX_RETRY times.

    Args:
        param_dict: The keyword arguments litellm.completion takes, holding the names
            in COMPLETION_PARAM_TUPLE and no other.

    Returns:
        {"content": the reply's text as the model wrote it or None,
        "finish_reason": str or None, "usage": usage_dict_of()'s dict}.
    """
    response = providercall.completion({**param_dict, **config.credential_dict(),
                                        "timeout": config.LLM_CALL_TIMEOUT_SEC,
                                        "num_retries": config.LLM_MAX_RETRY})
    choice = response.choices[0]
    return {"content": choice.message.content,
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": usage_dict_of(getattr(response, "usage", None))}
