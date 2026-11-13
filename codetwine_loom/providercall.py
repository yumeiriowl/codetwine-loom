"""One completion through litellm, made again after the errors a later attempt can pass.

litellm is given no retries of its own. A call the provider answers with status 429 or
5xx, whatever error class litellm gives it, one whose connection fails, and one whose
attempt runs out of time are made again after a wait, up to LLM_MAX_RETRY times: the
wait is LLM_RETRY_WAIT_SEC, doubled for each retry and at most LLM_CALL_TIMEOUT_SEC, or
what the provider's retry-after says when that is longer. Any other error is raised at
once.

A call asking a model that cannot stop thinking (always_thinks, as litellm decides it)
to think not at all is sent with _LEAST_THINK_DICT instead.

A call marked TURN_REPLY_KEY is received as a stream that ends with the provider's token
counts (STREAM_OPTION_DICT). What has arrived is asked of rlm.reply_end() each time it
has grown by a share of itself (_CHECK_GROWTH_DIVISOR) and once more when the stream
ends, and at the first position it returns the stream is closed and the reply ends
there. The attempt as a whole is given the call's "timeout", and the wait for each part
of the stream LLM_IDLE_TIMEOUT_SEC. The output tokens of an attempt made again are
counted from the thinking, content and tool call arguments it had received, and added to
the counts of the attempt that answered.

A reply may hold _MAX_CHAR_PER_TOKEN characters for each token of the call's max_tokens
(reply_max_char). A stream whose thinking, content and tool call arguments have grown
past that is closed there, and ends with the finish reason "length" where
rlm.reply_end() finds no end in the content kept. A reply received whole whose content
has grown past that has its content cut there and ends with the finish reason "length".

From the import of this module on, the HTTP clients litellm's own handlers hand out or
make, the client an OpenAI SDK client sends with, and the requests of an aiohttp session
follow no redirect (_keep_from_redirect): a call's key and prompt are sent to the
endpoint the call names, and an endpoint that answers with a redirect fails the attempt.
A litellm or an OpenAI SDK whose clients cannot be set so, and a proxy variable an HTTP
client of this tool cannot read (config.proxy_error_text), have completion() fail before
an attempt is made (check_before_call).
"""

import asyncio
import functools
import importlib.metadata
import inspect
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from typing import NoReturn, TypeVar

import httpx
import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import (
    declared_authenticating_provider)
from litellm.llms.anthropic.common_utils import AnthropicModelInfo

from codetwine_loom import config, textcut, tokencount

logger = logging.getLogger(__name__)

# What look_up_model() returns: the answer of the lookup it is given
LookupT = TypeVar("LookupT")

# litellm prints nothing of its own to standard output about a call that failed or a
# model it finds no provider for
litellm.suppress_debug_info = True

# litellm counts tokens with the tokenizers it ships with and fetches none
litellm.disable_hf_tokenizer_download = True

# The variable that has litellm read the model table it ships with instead of fetching
# one. Where it is "true", litellm fetches no Hugging Face model's config.json either
# (_keep_from_model_config_fetch)
LOCAL_MODEL_TABLE_ENV = "LITELLM_LOCAL_MODEL_COST_MAP"

# The errors a call is made again after whatever status they carry, as litellm raises
# them: a connection that failed and an attempt that ran out of time
_RETRY_ERROR_TUPLE = (litellm.Timeout, litellm.APIConnectionError)

# The status a provider answers with when it is too busy. It and every status from
# _MIN_RETRY_SERVER_STATUS up have the call made again
_BUSY_STATUS = 429
_MIN_RETRY_SERVER_STATUS = 500


# The thinking parameter of adaptive thinking: the thinking comes back summarized, and a
# reply received as a stream carries the thinking as it goes
ADAPTIVE_THINK_DICT = {"type": "adaptive", "display": "summarized"}

# What a call asking for no thinking is sent with to a model that cannot stop thinking
_LEAST_THINK_DICT = {"thinking": ADAPTIVE_THINK_DICT, "output_config": {"effort": "low"}}


# The key of a call's arguments that marks its reply as one turn of an RLM loop. It is
# taken off before litellm is called
TURN_REPLY_KEY = "turn_reply"

# The stream_options a turn's reply is asked for with: the provider's token counts at the
# end of the stream. litellm leaves them out for a provider that takes none
STREAM_OPTION_DICT = {"include_usage": True}

# How much a reply still arriving grows before rlm.reply_end() is asked again: the
# divisor of what has arrived
_CHECK_GROWTH_DIVISOR = 8

# How many characters a reply may hold for each token of its max_tokens, counted over
# the thinking, content and tool call arguments of a stream and over the content of a
# reply received whole, and the finish reason of a reply ended at that many
_MAX_CHAR_PER_TOKEN = 32
_LENGTH_FINISH = "length"

# The attributes completion() leaves on the response's usage: on a turn's reply, 1 when
# its token counts are not all the provider's own (a reply ended before the provider's
# last chunk, a stream whose last chunk carried no counts, or output tokens counted from
# what an attempt made again had received) and 0 otherwise; on either kind of reply, how
# many attempts were made before the one that answered
ESTIMATE_KEY = "estimate_count"
RETRY_KEY = "retry_count"

# The event loop the streams of this process are read on and the thread that runs it
_loop_lock = threading.Lock()
_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None

# Why an HTTP client of this process could not be set to follow no redirect; ""
# otherwise
_redirect_block_text = ""

# The client classes of the modules of config.HTTPX_MODULE_TUPLE
_HTTPX_CLIENT_CLASS_TUPLE = tuple(client_class for module in config.HTTPX_MODULE_TUPLE
                                  for client_class in (module.Client, module.AsyncClient))

# Held while litellm is given the HTTP clients its calls through the OpenAI SDK share
# (_share_client)
_share_lock = threading.Lock()


def look_up_model(lookup: Callable[[str], LookupT], model: str,
                  unknown: LookupT) -> LookupT:
    """Return what one of litellm's lookups answers for a model, without a sign-in.

    lookup is a litellm function that resolves the model's provider on the way
    (get_llm_provider, validate_environment, get_model_info). Resolving a model of a
    provider of PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO, as litellm's own
    declared_authenticating_provider() names it, signs in to the provider, or makes the
    directory litellm keeps the sign-in in under HOME. For such a model lookup is not
    run: no request is sent, nothing waits for a person and nothing is written under
    HOME.

    Args:
        lookup: The litellm function, called with the model alone.
        model: The model, in litellm form.
        unknown: What is returned in place of lookup's answer for a model litellm signs
            in for.

    Returns:
        lookup(model), or unknown for a model litellm signs in for.
    """
    if declared_authenticating_provider(model) is not None:
        return unknown
    return lookup(model)


def model_info_of(model: str) -> dict:
    """Return litellm's model information for a model, asked as this process's calls are.

    A provider litellm asks for the model's details is asked with this process's
    credentials (config.credential_dict()), at LLM_API_BASE when it is set, and with
    none when config.LLM_BROKER is set. Nothing is looked up while check_before_call()
    refuses a model call, nor for a model litellm signs in for (look_up_model).

    Args:
        model: A model name in litellm format.

    Returns:
        What litellm.get_model_info() returns, or {} when nothing is looked up.

    Raises:
        Exception: What litellm.get_model_info() raises, for a model it does not know
            among others.
    """
    credential_dict = _lookup_credential_dict()
    if credential_dict is None:
        return {}
    return look_up_model(lambda name: litellm.get_model_info(name, **credential_dict),
                         model, {})


def _lookup_credential_dict() -> dict[str, str] | None:
    """Return the credentials model_info_of() asks with.

    Returns:
        {} when config.LLM_BROKER is set, None when check_before_call() refuses a model
        call, and config.credential_dict() otherwise.
    """
    if config.LLM_BROKER:
        return {}
    try:
        check_before_call()
    except RuntimeError:
        return None
    return config.credential_dict()


def output_limit_of(model: str, max_token: int) -> int:
    """Return an output limit no higher than the model's own.

    Args:
        model: A model name in litellm format.
        max_token: The output limit the settings give.

    Returns:
        limit_within() of max_token and the model's table_output_limit().
    """
    return limit_within(model, max_token, table_output_limit(model))


def limit_within(model: str, max_token: int, model_limit: int) -> int:
    """Return an output limit no higher than a model's own.

    Args:
        model: A model name in litellm format, named in the log.
        max_token: The output limit the settings give.
        model_limit: The model's own output limit, table_output_limit()'s value; 0 for
            none.

    Returns:
        The lower of max_token and model_limit, or max_token when model_limit is 0.
    """
    if 0 < model_limit < max_token:
        logger.debug("%s takes up to %d output tokens, so a call asks for that many in "
                     "place of %d", model, model_limit, max_token)
        return model_limit
    return max_token


def table_output_limit(model: str) -> int:
    """Return the max_output_tokens litellm's model table holds for a model.

    Args:
        model: A model name in litellm format.

    Returns:
        The max_output_tokens of model_info_of(), or 0 when it holds no whole number
        above 0 or litellm does not know the model.
    """
    try:
        model_limit = model_info_of(model).get("max_output_tokens")
    except Exception:                     # noqa: BLE001 - an unknown model raises
        return 0
    return model_limit if type(model_limit) is int and model_limit > 0 else 0


def sign_in_model_list(model_list: list[str]) -> list[str]:
    """Return the models litellm signs in to the provider of as it resolves them.

    Args:
        model_list: The models, in litellm form.

    Returns:
        Those of model_list look_up_model() does not look up, in its order.
    """
    return [model for model in model_list
            if declared_authenticating_provider(model) is not None]


def always_thinks(model: str) -> bool:
    """Say whether a model thinks whatever it is asked.

    Args:
        model: A model name in litellm format.

    Returns:
        What AnthropicModelInfo._is_always_on_thinking_model says of the model under its
        litellm provider. For a model of a provider litellm signs in to as it resolves
        it (look_up_model), what _table_always_thinks() says. A model of a provider
        litellm does not know is taken as one that does not.
    """
    try:
        return look_up_model(
            lambda name: AnthropicModelInfo._is_always_on_thinking_model(
                *litellm.get_llm_provider(name)[:2]),
            model, _table_always_thinks(model))
    except litellm.BadRequestError:
        return False


def _table_always_thinks(model: str) -> bool:
    """Say whether litellm's model table marks a model as one that thinks whatever asked.

    Args:
        model: A model name in litellm format.

    Returns:
        thinking_always_on of the table's entry under the model as it is written; False
        for a model the table has no entry for.
    """
    return litellm.model_cost.get(model, {}).get("thinking_always_on") is True


def _is_retry_error(error: Exception) -> bool:
    """Say whether a call that raised an error is made again.

    Args:
        error: The error an attempt ended with.

    Returns:
        True for one of _RETRY_ERROR_TUPLE and for an error whose status_code is
        _BUSY_STATUS or _MIN_RETRY_SERVER_STATUS and up, whatever class litellm gave it.
    """
    if isinstance(error, _RETRY_ERROR_TUPLE):
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and (status == _BUSY_STATUS
                                        or status >= _MIN_RETRY_SERVER_STATUS)


def _retry_wait_sec(error: Exception, retry_index: int) -> float:
    """Return how long to wait before the next attempt.

    Args:
        error: The error the attempt ended with.
        retry_index: How many retries came before this one (0 for the first).

    Returns:
        LLM_RETRY_WAIT_SEC x 2 ** retry_index, or the provider's retry-after when that is
        longer, at most LLM_CALL_TIMEOUT_SEC.
    """
    wait = config.LLM_RETRY_WAIT_SEC * 2 ** retry_index
    header_dict = getattr(error, "litellm_response_headers", None) or {}
    try:
        wait = max(wait, float(header_dict.get("retry-after", 0)))
    except (AttributeError, TypeError, ValueError):
        pass
    return min(wait, config.LLM_CALL_TIMEOUT_SEC)


def reply_max_char(max_token: object) -> int:
    """Return how many characters the reply of a call may hold.

    Args:
        max_token: The call's max_tokens.

    Returns:
        max_token x _MAX_CHAR_PER_TOKEN, or 0, which sets no limit, when max_token is
        not a whole number above 0.
    """
    if type(max_token) is not int or max_token <= 0:
        return 0
    return max_token * _MAX_CHAR_PER_TOKEN


def _cut_reply(response: object, text: str, model: str, max_char: int) -> None:
    """Shape the response of a reply that grew past what its max_tokens covers.

    Args:
        response: litellm's response. It is changed in place: the content becomes
            the first max_char characters of text and the finish reason _LENGTH_FINISH.
        text: The content that arrived.
        model: The model name, which the log names.
        max_char: What reply_max_char() gave for the call.
    """
    logger.warning("the reply of %s grew past the %d characters its max_tokens covers "
                   "(%d for each token), so it ends there", model or "?", max_char,
                   _MAX_CHAR_PER_TOKEN)
    choice = response.choices[0]
    choice.message.content = text[:max_char]
    choice.finish_reason = _LENGTH_FINISH


def _stream_loop() -> asyncio.AbstractEventLoop:
    """Return the event loop streams are read on, starting its thread when none runs.

    Returns:
        The loop, running in a daemon thread of this process.
    """
    global _loop, _loop_thread
    with _loop_lock:
        if _loop_thread is None or not _loop_thread.is_alive():
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(target=_loop.run_forever,
                                            name="ctloom-stream", daemon=True)
            _loop_thread.start()
        return _loop


def _block_redirect_text(error: Exception) -> str:
    """Return the reason no call is made, for an HTTP client that may follow a redirect.

    Args:
        error: What kept a client of litellm or the OpenAI SDK from being set to follow
            none.

    Returns:
        One line naming the versions of litellm and the OpenAI SDK, and the error.
    """
    version_list = []
    for package in ("litellm", "openai"):
        try:
            version_list.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            version_list.append(f"{package} ?")
    return (f"no call is made to a model: the HTTP clients of {' and '.join(version_list)} "
            f"cannot be kept from following a redirect "
            f"({type(error).__name__}: {textcut.one_line(str(error))}); "
            f"install versions of them this tool supports")


def _no_redirect(client: object) -> object:
    """Set one HTTP client litellm or the OpenAI SDK sends with to follow no redirect.

    Args:
        client: What one of litellm's functions that make an HTTP client returned, or
            the client an OpenAI SDK client sends with.

    Returns:
        client, whose follow_redirects is False.

    Raises:
        TypeError: When client is not a client of a module of
            config.HTTPX_MODULE_TUPLE. _redirect_block_text holds the reason from then
            on.
    """
    global _redirect_block_text
    if not isinstance(client, _HTTPX_CLIENT_CLASS_TUPLE):
        module_text = " or ".join(module.__name__ for module in config.HTTPX_MODULE_TUPLE)
        error = TypeError(f"{type(client).__name__} is not an {module_text} client")
        _redirect_block_text = _block_redirect_text(error)
        raise error
    client.follow_redirects = False
    return client


def _no_redirect_maker(client_of: Callable) -> Callable:
    """Wrap a function that returns an HTTP client: the client is set to follow no redirect.

    Args:
        client_of: create_client of one of litellm's HTTP handlers, or the getter of
            its client property.

    Returns:
        A function that takes what client_of takes and returns what it returns,
        after _no_redirect().
    """
    @functools.wraps(client_of)
    def no_redirect_client_of(*arg_tuple: object, **kwarg_dict: object) -> object:
        """Return what client_of returns, set to follow no redirect."""
        return _no_redirect(client_of(*arg_tuple, **kwarg_dict))
    return no_redirect_client_of


def _no_redirect_request(send_request: Callable) -> Callable:
    """Wrap aiohttp's request function: a request sent through it follows no redirect.

    Args:
        send_request: aiohttp.ClientSession._request.

    Returns:
        A coroutine function that takes what send_request takes and calls it with
        allow_redirects False.
    """
    @functools.wraps(send_request)
    async def send_no_redirect_request(*arg_tuple: object, **kwarg_dict: object) -> object:
        """Send the request with allow_redirects False."""
        return await send_request(*arg_tuple, **{**kwarg_dict, "allow_redirects": False})
    return send_no_redirect_request


def _no_redirect_start(start: Callable) -> Callable:
    """Wrap an OpenAI SDK client's __init__: its HTTP client is set to follow no redirect.

    The HTTP client is the one the OpenAI SDK client sends with.

    Args:
        start: SyncAPIClient.__init__ or AsyncAPIClient.__init__ of the OpenAI SDK.

    Returns:
        A function that takes what start takes, runs it, and puts the client's
        _client, the one it was given or the one it made, through _no_redirect().
    """
    @functools.wraps(start)
    def start_no_redirect(self: object, *arg_tuple: object, **kwarg_dict: object) -> None:
        """Run start, then set the client's _client to follow no redirect."""
        start(self, *arg_tuple, **kwarg_dict)
        _no_redirect(getattr(self, "_client", None))
    return start_no_redirect


def _keep_from_redirect() -> None:
    """Set litellm's, the OpenAI SDK's and aiohttp's HTTP clients to follow no redirect.

    It is run as this module is imported. On litellm's HTTPHandler and AsyncHTTPHandler,
    create_client and the getter of client are replaced by _no_redirect_maker()'s, on
    the class: a handler made before this module was imported hands out a client set the
    same way. The __init__ of the OpenAI SDK's SyncAPIClient and AsyncAPIClient is
    replaced by _no_redirect_start()'s, and aiohttp.ClientSession._request by
    _no_redirect_request()'s. A litellm, an OpenAI SDK or an aiohttp that holds them in
    another form, and whatever else stops a replacement, leaves the reason in
    _redirect_block_text.
    """
    global _redirect_block_text
    try:
        import aiohttp
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
        from openai._base_client import AsyncAPIClient, SyncAPIClient

        for handler_class in (HTTPHandler, AsyncHTTPHandler):
            make_client = vars(handler_class).get("create_client")
            client_property = vars(handler_class).get("client")
            if not inspect.isfunction(make_client) \
                    or not isinstance(client_property, property) \
                    or not inspect.isfunction(client_property.fget):
                raise TypeError(f"{handler_class.__name__} holds no create_client "
                                f"function or no client property")
            handler_class.create_client = _no_redirect_maker(make_client)
            handler_class.client = client_property.getter(
                _no_redirect_maker(client_property.fget))
        for api_class in (SyncAPIClient, AsyncAPIClient):
            start = vars(api_class).get("__init__")
            if not inspect.isfunction(start) \
                    or "http_client" not in inspect.signature(start).parameters:
                raise TypeError(f"openai's {api_class.__name__} takes no http_client")
            api_class.__init__ = _no_redirect_start(start)
        send_request = vars(aiohttp.ClientSession).get("_request")
        if not inspect.iscoroutinefunction(send_request) \
                or "allow_redirects" not in inspect.signature(send_request).parameters:
            raise TypeError("aiohttp.ClientSession._request takes no allow_redirects")
        aiohttp.ClientSession._request = _no_redirect_request(send_request)
    except Exception as e:
        _redirect_block_text = _block_redirect_text(e)


_keep_from_redirect()


def _is_local_table() -> bool:
    """Say whether litellm reads the model table it ships with, as the environment is now.

    Returns:
        True when LOCAL_MODEL_TABLE_ENV is "true", in any case.
    """
    return os.environ.get(LOCAL_MODEL_TABLE_ENV, "").lower() == "true"


def _no_model_config_fetch(fetch_config: Callable) -> Callable:
    """Wrap litellm's fetch of a Hugging Face model's config.json to skip it when local.

    The fetch is not made while LOCAL_MODEL_TABLE_ENV is "true".

    Args:
        fetch_config: litellm.utils._get_max_position_embeddings.

    Returns:
        A function that takes what fetch_config takes and returns None, the answer
        fetch_config gives when its request fails, while LOCAL_MODEL_TABLE_ENV is
        "true" (in any case), and what fetch_config returns otherwise.
    """
    @functools.wraps(fetch_config)
    def fetch_config_unless_local(*arg_tuple: object, **kwarg_dict: object) -> object:
        """Return None while the model table is local, fetch_config's answer otherwise."""
        if _is_local_table():
            return None
        return fetch_config(*arg_tuple, **kwarg_dict)
    return fetch_config_unless_local


class _NoRequestClient:
    """An HTTP client whose requests fail without being sent."""

    def get(self, *arg_tuple: object, **kwarg_dict: object) -> NoReturn:
        """Fail as a request that could not be sent."""
        raise httpx.ConnectError("no request is sent where the model table is local")


class _LitellmWithNoRequest:
    """litellm, with a module_level_client whose requests fail without being sent."""

    module_level_client = _NoRequestClient()

    def __getattr__(self, name: str) -> object:
        """Return litellm's own attribute name."""
        return getattr(litellm, name)


def _no_max_token_fetch(get_max_tokens: Callable) -> Callable:
    """Wrap litellm's get_max_tokens() to skip its config.json fetch when local.

    The fetch of a Hugging Face model's config.json its inner function makes is not made
    while LOCAL_MODEL_TABLE_ENV is "true".

    Args:
        get_max_tokens: litellm.utils.get_max_tokens.

    Returns:
        A function that takes what get_max_tokens takes. While LOCAL_MODEL_TABLE_ENV is
        "true" (in any case) it runs get_max_tokens's own code with litellm standing for
        _LitellmWithNoRequest, and the fetch fails as one that could not be made;
        otherwise it calls get_max_tokens.
    """
    get_max_tokens_here = config.with_global(get_max_tokens, "litellm",
                                             _LitellmWithNoRequest())

    @functools.wraps(get_max_tokens)
    def get_max_tokens_unless_local(*arg_tuple: object, **kwarg_dict: object) -> object:
        """Run get_max_tokens_here while the model table is local, else get_max_tokens."""
        if _is_local_table():
            return get_max_tokens_here(*arg_tuple, **kwarg_dict)
        return get_max_tokens(*arg_tuple, **kwarg_dict)
    return get_max_tokens_unless_local


def _keep_from_model_config_fetch() -> None:
    """Replace litellm's two functions that fetch a Hugging Face model's config.json.

    litellm.utils._get_max_position_embeddings is replaced by
    _no_model_config_fetch()'s, and litellm.utils.get_max_tokens and
    litellm.get_max_tokens by _no_max_token_fetch()'s.

    It is run as this module is imported. A litellm that holds no such function is left
    as it is.
    """
    import litellm.utils

    fetch_config = vars(litellm.utils).get("_get_max_position_embeddings")
    if inspect.isfunction(fetch_config):
        litellm.utils._get_max_position_embeddings = _no_model_config_fetch(fetch_config)
    get_max_tokens = vars(litellm.utils).get("get_max_tokens")
    if inspect.isfunction(get_max_tokens):
        litellm.utils.get_max_tokens = litellm.get_max_tokens = _no_max_token_fetch(
            get_max_tokens)


_keep_from_model_config_fetch()


def _share_client() -> None:
    """Give litellm the HTTP clients its calls through the OpenAI SDK share, if it has none.

    litellm.client_session and litellm.aclient_session are each set, when None, to the
    client BaseOpenAILLM._get_sync_http_client() or _get_async_http_client() makes under
    the environment of that moment, set to follow no redirect (_no_redirect). A client
    that cannot be made is left unset, and is made at the next call. Nothing raises
    here.
    """
    with _share_lock:
        try:
            from litellm.llms.openai.common_utils import BaseOpenAILLM

            if litellm.client_session is None:
                litellm.client_session = _no_redirect(
                    BaseOpenAILLM._get_sync_http_client())
            if litellm.aclient_session is None:
                litellm.aclient_session = _no_redirect(
                    BaseOpenAILLM._get_async_http_client())
        except Exception as e:
            logger.debug("litellm is given no shared HTTP client for now (%s)",
                         type(e).__name__)


def check_before_call() -> None:
    """Refuse a model call that could go astray, and otherwise share litellm's clients.

    A call is refused while it could follow a redirect or go through a proxy variable an
    HTTP client of this tool cannot read. Otherwise litellm is given the clients its
    calls through the OpenAI SDK share (_share_client).

    The answer rests on what is set alone: the same state gives the same answer on
    every call.

    Raises:
        RuntimeError: With _redirect_block_text, when _keep_from_redirect() or
            _no_redirect() left one, and otherwise with config.proxy_error_text(),
            when it names a proxy variable.
    """
    if _redirect_block_text:
        raise RuntimeError(_redirect_block_text)
    proxy_text = config.proxy_error_text()
    if proxy_text:
        raise RuntimeError(f"no call is made to a model: {proxy_text}")
    _share_client()


async def _receive(kwarg_dict: dict, piece_list: list[str]) -> object:
    """Receive one reply as a stream, closing it where rlm.reply_end() says.

    Args:
        kwarg_dict: The keyword arguments litellm.acompletion takes.
        piece_list: Where each piece of thinking, content and tool call arguments is
            appended as it arrives.

    Returns:
        The response litellm builds from the chunks that arrived, as _end_reply()
        shapes it when the reply was ended, and as _cut_reply() shapes it when the
        stream was closed at reply_max_char() with nowhere to end the reply before
        that.

    Raises:
        litellm.APIConnectionError: When the stream ended with no chunk.
        litellm.Timeout: When no part of the reply arrived for LLM_IDLE_TIMEOUT_SEC.
    """
    # rlm imports llm, which imports this module
    from codetwine_loom import rlm

    stream = await _within_idle_limit(
        litellm.acompletion(**kwarg_dict, stream=True, stream_options=STREAM_OPTION_DICT,
                            num_retries=0),
        kwarg_dict.get("model", ""))
    max_char = reply_max_char(kwarg_dict.get("max_tokens"))
    chunk_list, text, end = await _read_to_end(
        stream, rlm.reply_end, kwarg_dict.get("model", ""), piece_list, max_char)
    is_over = 0 < max_char < sum(len(piece) for piece in piece_list)
    if is_over:
        text = text[:max_char]
    # What arrived after the last look is looked at once more
    if end is None and text:
        end = rlm.reply_end(text)
    # The stream keeps the chunks with the token counts the provider sent on them
    chunk_list = getattr(stream, "chunks", None) or chunk_list
    response = litellm.stream_chunk_builder(chunk_list, messages=kwarg_dict.get("messages"))
    if response is None:
        raise litellm.APIConnectionError(
            message="the stream ended with no chunk",
            llm_provider="", model=kwarg_dict.get("model", ""))
    is_estimate = not getattr(chunk_list[-1], "usage", None)
    if end is not None:
        logger.debug("the reply of %s was ended at character %d of the %d that had "
                     "arrived", kwarg_dict.get("model", "?"), end, len(text))
        _end_reply(response, text, end, kwarg_dict.get("model", ""))
        is_estimate = True
    elif is_over:
        _cut_reply(response, text, kwarg_dict.get("model", ""), max_char)
        is_estimate = True
    _mark_usage(response, ESTIMATE_KEY, int(is_estimate))
    return response


def _mark_usage(response: object, key: str, value: int) -> None:
    """Leave one count on a response's usage, when it has one.

    Args:
        response: litellm's response.
        key: ESTIMATE_KEY or RETRY_KEY.
        value: The count.
    """
    usage = getattr(response, "usage", None)
    if usage is not None:
        setattr(usage, key, value)


def _field(holder: object, name: str) -> object:
    """Read one field of a part of a chunk, which is an object or a dict.

    Args:
        holder: The part, or None.
        name: The field.

    Returns:
        The value, or None when holder does not carry it.
    """
    if isinstance(holder, dict):
        return holder.get(name)
    return getattr(holder, name, None)


def _call_argument(delta: object) -> str:
    """Return the tool call arguments one chunk of a stream carries.

    Args:
        delta: The chunk's delta, or None.

    Returns:
        The "arguments" of the function of each of its tool_calls and of its
        function_call, joined; "" when it carries none.
    """
    tool_call_list = _field(delta, "tool_calls")
    function_list = [_field(tool_call, "function") for tool_call
                     in (tool_call_list if isinstance(tool_call_list, list) else [])]
    function_list.append(_field(delta, "function_call"))
    argument_list = [_field(function, "arguments") for function in function_list]
    return "".join(argument for argument in argument_list if isinstance(argument, str))


async def _read_to_end(stream: object, reply_end: Callable[[str], int | None],
                       model: str, piece_list: list[str],
                       max_char: int = 0) -> tuple[list, str, int | None]:
    """Read a stream until it ends, reply_end() gives a position or it is too long.

    The stream is too long when the thinking, content and tool call arguments that
    arrived hold more than max_char characters. The stream is closed at the end.
    reply_end() is asked each time the content has grown by a share of itself
    (_CHECK_GROWTH_DIVISOR) since it was last asked.

    Args:
        stream: The stream litellm.acompletion returned.
        reply_end: rlm.reply_end.
        model: The model name, which the timeout error carries.
        piece_list: Where each piece of thinking, content and tool call arguments
            (_call_argument) is appended as it arrives.
        max_char: How many characters the pieces may hold between them
            (reply_max_char). 0 sets no limit.

    Returns:
        (the chunks read, the content they carried, the position reply_end() gave or
        None when it gave none).

    Raises:
        litellm.Timeout: When no chunk arrived for LLM_IDLE_TIMEOUT_SEC.
    """
    chunk_list = []
    text = ""
    end = None
    check_at = 0
    char_count = 0
    chunk_iterator = aiter(stream)
    try:
        while True:
            try:
                chunk = await _within_idle_limit(anext(chunk_iterator), model)
            except StopAsyncIteration:
                break
            chunk_list.append(chunk)
            delta = chunk.choices[0].delta if chunk.choices else None
            think_piece = getattr(delta, "reasoning_content", None)
            if think_piece:
                piece_list.append(think_piece)
                char_count += len(think_piece)
            argument_piece = _call_argument(delta)
            if argument_piece:
                piece_list.append(argument_piece)
                char_count += len(argument_piece)
            piece = getattr(delta, "content", None)
            if piece:
                piece_list.append(piece)
                text += piece
                char_count += len(piece)
            if 0 < max_char < char_count:
                break
            if not piece or len(text) < check_at:
                continue
            check_at = len(text) + len(text) // _CHECK_GROWTH_DIVISOR
            end = reply_end(text)
            if end is not None:
                break
    finally:
        await stream.aclose()
    return chunk_list, text, end


async def _within_idle_limit(awaitable: Awaitable, model: str) -> object:
    """Wait for the next part of a reply for at most LLM_IDLE_TIMEOUT_SEC.

    Args:
        awaitable: What gives the stream or its next chunk.
        model: The model name, which the timeout error carries.

    Returns:
        What awaitable gives.

    Raises:
        litellm.Timeout: When it gave nothing within LLM_IDLE_TIMEOUT_SEC. 0 waits as
            long as it takes.
    """
    try:
        return await asyncio.wait_for(awaitable, config.LLM_IDLE_TIMEOUT_SEC or None)
    except TimeoutError as e:
        raise litellm.Timeout(message=f"no part of the reply arrived for "
                                      f"{config.LLM_IDLE_TIMEOUT_SEC} s",
                              model=model, llm_provider="") from e


def _end_reply(response: object, text: str, end: int, model: str) -> None:
    """Shape the response of a reply that ends where rlm.reply_end() says.

    Args:
        response: What litellm built from the chunks that arrived. It is changed in
            place: the content becomes text[:end], the finish reason "stop", and the
            output token count litellm's own count of what arrived when the provider
            had reported fewer.
        text: The content that arrived.
        end: Where rlm.reply_end() ended it.
        model: The model name, which picks the tokenizer.
    """
    choice = response.choices[0]
    think_text = getattr(choice.message, "reasoning_content", None) or ""
    choice.message.content = text[:end]
    choice.finish_reason = "stop"
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    output_token = litellm.token_counter(model=model, text=think_text + text)
    if output_token > (usage.completion_tokens or 0):
        usage.total_tokens = (usage.prompt_tokens or 0) + output_token
        usage.completion_tokens = output_token


async def _receive_within(kwarg_dict: dict, limit: float | None,
                          piece_list: list[str]) -> object:
    """Receive one reply as _receive() does, within a limit on the whole of it.

    Args:
        kwarg_dict: The keyword arguments litellm.acompletion takes.
        limit: How long receiving may take from the request to the end of the reply
            (seconds), or None for no limit.
        piece_list: Where each piece of the reply is appended as it arrives
            (_read_to_end).

    Returns:
        What _receive() returns.

    Raises:
        litellm.Timeout: When the reply was not complete within limit.
    """
    try:
        return await asyncio.wait_for(_receive(kwarg_dict, piece_list), limit)
    except TimeoutError as e:
        raise litellm.Timeout(message=f"the reply was not complete after {limit} s",
                              model=kwarg_dict.get("model", ""), llm_provider="") from e


def _turn_reply(kwarg_dict: dict, piece_list: list[str]) -> object:
    """Make one attempt at a call whose reply is one turn of an RLM loop.

    The stream is read on the event loop of _stream_loop() while this thread waits. An
    exception raised in this thread while it waits closes the stream and is raised as it
    is.

    Args:
        kwarg_dict: The keyword arguments litellm.acompletion takes. A number under
            "timeout" is how long the attempt may take from the request to the end of
            the reply.
        piece_list: Where each piece of the reply is appended as it arrives
            (_read_to_end), which holds what had arrived when the attempt fails.

    Returns:
        What _receive() returns.

    Raises:
        litellm.Timeout: When the attempt ran past its "timeout".
    """
    timeout = kwarg_dict.get("timeout")
    limit = timeout if isinstance(timeout, (int, float)) else None
    future = asyncio.run_coroutine_threadsafe(_receive_within(kwarg_dict, limit, piece_list),
                                              _stream_loop())
    try:
        return future.result()
    finally:
        future.cancel()


def _whole_reply(kwarg_dict: dict) -> object:
    """Make one attempt at a call whose reply is received whole.

    Args:
        kwarg_dict: The keyword arguments litellm.completion takes.

    Returns:
        litellm's response, as _cut_reply() shapes it when its content holds more
        characters than reply_max_char() gives for the call.
    """
    response = litellm.completion(**kwarg_dict, num_retries=0)
    max_char = reply_max_char(kwarg_dict.get("max_tokens"))
    choice_list = getattr(response, "choices", None)
    content = getattr(getattr(choice_list[0], "message", None), "content", None) \
        if choice_list else None
    if max_char and isinstance(content, str) and len(content) > max_char:
        _cut_reply(response, content, kwarg_dict.get("model", ""), max_char)
    return response


def completion(param_dict: dict) -> object:
    """Call litellm.completion, making the call again after the errors that may pass.

    Args:
        param_dict: The keyword arguments litellm.completion takes. "num_retries" is how
            many times the call is made again here, LLM_MAX_RETRY when it is absent, and
            litellm itself is given none. {"type": "disabled"} thinking is sent as
            _LEAST_THINK_DICT to a model that always_thinks(). A true TURN_REPLY_KEY has
            the reply received as a stream and ended where rlm.reply_end() says. A reply
            that holds more than reply_max_char() gives for "max_tokens" is cut there
            and ends with the finish reason _LENGTH_FINISH; a turn's reply ends instead
            where rlm.reply_end() finds an end in what is kept, when it finds one. Every
            half of a surrogate pair in "messages" is sent as U+FFFD
            (textcut.no_lone_surrogate).

    Returns:
        litellm's response, with ESTIMATE_KEY (on a turn's reply) and RETRY_KEY left on
        its usage. The output tokens counted from what the turn's attempts made again
        had received are added to its counts (_add_fail_output), and its counts are
        added to this process's total (tokencount.add).

    Raises:
        Exception: What the last attempt raised, or what an attempt raised that
            _is_retry_error() does not make again.
        RuntimeError: Before an attempt is made, when an HTTP client of this process
            may follow a redirect or a proxy variable cannot be read
            (check_before_call).
    """
    kwarg_dict = dict(param_dict)
    if "messages" in kwarg_dict:
        kwarg_dict["messages"] = textcut.no_lone_surrogate(kwarg_dict["messages"])
    retry_count = kwarg_dict.pop("num_retries", config.LLM_MAX_RETRY)
    is_turn_reply = kwarg_dict.pop(TURN_REPLY_KEY, False)
    if kwarg_dict.get("thinking") == {"type": "disabled"} \
            and always_thinks(kwarg_dict.get("model", "")):
        kwarg_dict.update(_LEAST_THINK_DICT)
    retry_index = 0
    fail_token = 0
    while True:
        check_before_call()
        piece_list: list[str] = []
        try:
            response = (_turn_reply(kwarg_dict, piece_list) if is_turn_reply
                        else _whole_reply(kwarg_dict))
            _mark_usage(response, RETRY_KEY, retry_index)
            _add_fail_output(response, fail_token)
            tokencount.add(tokencount.count_dict_of(getattr(response, "usage", None)))
            return response
        except Exception as e:
            if not _is_retry_error(e) or retry_index >= retry_count:
                raise
            attempt_token = _piece_token(kwarg_dict.get("model", ""), piece_list)
            fail_token += attempt_token
            wait = _retry_wait_sec(e, retry_index)
            logger.info("the call to %s failed (%s) %s, so it is made again in %.0f s "
                        "(retry %d of %d)", kwarg_dict.get("model", "?"),
                        type(e).__name__,
                        (f"after about {attempt_token} output tokens had arrived"
                         if piece_list else "before any of the reply arrived"),
                        wait, retry_index + 1, retry_count)
            time.sleep(wait)
            retry_index += 1


def _piece_token(model: str, piece_list: list[str]) -> int:
    """Count the output tokens of the pieces of a reply one attempt had received.

    Args:
        model: The model name, which picks the tokenizer.
        piece_list: The pieces the attempt had received.

    Returns:
        litellm's count of them, 0 when nothing had arrived.
    """
    if not piece_list:
        return 0
    return litellm.token_counter(model=model, text="".join(piece_list))


def _add_fail_output(response: object, fail_token: int) -> None:
    """Add the output tokens of the attempts made again to the response's counts.

    Args:
        response: litellm's response. Its usage is changed in place: completion_tokens
            and total_tokens grow by fail_token, and ESTIMATE_KEY becomes 1.
        fail_token: The output tokens counted from what those attempts had received.
            0 leaves the response as it is.
    """
    usage = getattr(response, "usage", None)
    if not fail_token or usage is None:
        return
    usage.completion_tokens = (usage.completion_tokens or 0) + fail_token
    usage.total_tokens = (usage.total_tokens or 0) + fail_token
    setattr(usage, ESTIMATE_KEY, 1)
