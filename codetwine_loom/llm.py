"""Build a callable that calls the LLM (make_llm).

The call goes through litellm, which picks the provider from the head of the model name.

config.LLM_BROKER decides where litellm is called: empty calls it in this process, with
the credentials read here, and an address sends the call to a broker that holds them
(broker.py).
"""

import functools
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import litellm

from codetwine_loom import broker, config, providercall

logger = logging.getLogger(__name__)


class PromptTooLong(Exception):
    """Signals that the provider refused the prompt for its length.

    It stands for litellm's ContextWindowExceededError, from a call made here or through
    the broker.
    """


# For a model that takes a thinking budget rather than adaptive thinking: the share of
# the output limit each level may think with, as the divisor of that limit
_THINK_BUDGET_DIVISOR_DICT = {"low": 8, "medium": 4, "high": 2}

# The smallest thinking budget the provider takes (tokens)
_MIN_THINK_BUDGET = 1024

# The thinking types each model has refused in this process, as model ->
# {thinking.type}. A form in it is not sent to that model again
_refuse_think_type_dict: dict[str, set[str]] = {}

# The least output limit a call refused for the context length is sent again with
# (tokens): a call whose halved limit would fall below it is not sent again
_MIN_LOWER_OUTPUT_TOKEN = 4096

# The (model, output limit setting) pairs a kept output limit has been logged for as a
# warning in this process; the later ones are logged at INFO
_fit_warn_pair_set: set[tuple[str, str]] = set()


# The settings that stand for a provider's variable, by the part of the variable's name
# each stands for
_SHARE_SETTING_DICT = {"api_key": "LLM_API_KEY", "api_base": "LLM_API_BASE"}

# The parts of a provider's variable name that mark its value as a credential
_CREDENTIAL_NAME_PART_TUPLE = ("KEY", "TOKEN", "SECRET")

# The variables a call to a provider goes through with in place of the ones litellm's
# check names as missing, by provider. A model of the provider lacks nothing when
# LLM_API_KEY or one of them is set
_OTHER_WAY_DICT = {"bedrock": ("AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK")}

# The cache point placed after a prompt block, as litellm passes it to the provider
CACHE_CONTROL_DICT = {"type": "ephemeral"}


def content_list_of(block_list: list[tuple[str, bool]]) -> list[dict]:
    """Turn a prompt's blocks into the content blocks of one message.

    Args:
        block_list: [(text, whether a cache point follows it), ...], as
            rlm.build_prompt_block_list() returns them.

    Returns:
        One text block per entry, carrying cache_control where a cache point follows.
    """
    return [{"type": "text", "text": text,
             **({"cache_control": dict(CACHE_CONTROL_DICT)} if is_cache_point else {})}
            for text, is_cache_point in block_list]


def _choice(param_dict: dict) -> tuple[object, dict]:
    """Make one call, here or through the broker, and return the reply's first choice.

    Args:
        param_dict: The keyword arguments litellm.completion takes.

    Returns:
        (the first choice, holding .message.content and .finish_reason, the call's
        token counts as broker.usage_dict_of() returns them).
    """
    if config.LLM_BROKER:
        choice = broker.complete(param_dict)
        return choice, choice.usage_dict
    response = providercall.completion(param_dict)
    return response.choices[0], broker.usage_dict_of(getattr(response, "usage", None))


# The words in a provider's refusal that say it was the thinking parameters it would not
# take
_REFUSAL_WORD_TUPLE = ("thinking", "effort", "output_config")


def _think_param_dict(level: str) -> dict:
    """Turn a thinking level into the kwargs litellm takes.

    The output limit covers thinking and reply together: what is spent on thinking is
    taken off the reply.

    Args:
        level: A value of config.THINK_LEVEL_TUPLE. "" passes nothing and leaves it to
            the provider; "off" turns thinking off; "low" / "medium" / "high" ask for
            adaptive thinking at that effort, with the thinking summarized
            (providercall.ADAPTIVE_THINK_DICT).

    Returns:
        The kwargs to add to the litellm call, empty when nothing should be passed.

    Raises:
        ValueError: When level is not one of config.THINK_LEVEL_TUPLE.
    """
    if level not in config.THINK_LEVEL_TUPLE:
        raise ValueError(f"the thinking level {level!r} is not one of "
                         f"{'/'.join(value or '(empty)' for value in config.THINK_LEVEL_TUPLE)}")
    if not level:
        return {}
    if level == "off":
        return {"thinking": {"type": "disabled"}}
    return {"thinking": providercall.ADAPTIVE_THINK_DICT, "output_config": {"effort": level}}


def think_budget(level: str, max_token: int) -> int:
    """Return the thinking budget a level asks for under an output limit.

    Examples:
        ("low", 16384) -> 2048, ("high", 16384) -> 8192, ("low", 4096) -> 1024,
        ("low", 1024) -> 0, ("off", 16384) -> 0

    Args:
        level: A value of config.THINK_LEVEL_TUPLE.
        max_token: The output limit of one call, which the budget is part of.

    Returns:
        The limit divided by the level's divisor, raised to _MIN_THINK_BUDGET. 0 for a
        level that asks for no thinking, and for a limit that leaves no room past that
        budget for the reply.
    """
    if level not in _THINK_BUDGET_DIVISOR_DICT:
        return 0
    budget = max(_MIN_THINK_BUDGET, max_token // _THINK_BUDGET_DIVISOR_DICT[level])
    return budget if budget < max_token else 0


def think_param_chain(level: str, max_token: int) -> list[dict]:
    """Return the thinking kwargs to try, in order, until the model takes one.

    Args:
        level: A value of config.THINK_LEVEL_TUPLE.
        max_token: The output limit of one call.

    Returns:
        _think_param_dict(level) first, then a fixed thinking budget when the level asks
        for thinking and the limit has room for one, then {} when anything came before.

    Raises:
        ValueError: When level is not one of config.THINK_LEVEL_TUPLE.
    """
    chain = [_think_param_dict(level)]
    budget = think_budget(level, max_token)
    if budget:
        chain.append({"thinking": {"type": "enabled", "budget_tokens": budget}})
    if chain[0]:
        chain.append({})
    return chain


def _think_type(param_dict: dict) -> str:
    """Return the thinking.type a form of thinking kwargs sends.

    Args:
        param_dict: One entry of think_param_chain().

    Returns:
        "adaptive", "enabled" or "disabled", or "" for the form that sends nothing.
    """
    return param_dict.get("thinking", {}).get("type", "")


def _is_think_refusal(error: Exception, think_param_list: list[dict]) -> bool:
    """Say whether a refusal is one of the thinking parameters with a form left to try.

    Args:
        error: The BadRequestError the call ended with.
        think_param_list: The forms still to try, the one in use first.

    Returns:
        True when another form follows the one in use and the refusal names one of
        _REFUSAL_WORD_TUPLE.
    """
    return len(think_param_list) >= 2 and any(word in str(error).lower()
                                              for word in _REFUSAL_WORD_TUPLE)


def _move_to_next_think_form(model: str, think_param_list: list[dict], param_dict: dict,
                             think_level: str, error: Exception) -> None:
    """Record the form in use as refused by model and put the next one on the call.

    Args:
        model: The model that refused it.
        think_param_list: The forms still to try, the one in use first. Its head is
            removed.
        param_dict: The call's keyword arguments. The refused form's keys are taken
            off it and the next form's are added, fitted to its max_tokens
            (_fit_think_budget).
        think_level: The level the call's thinking came from.
        error: The refusal, quoted in the log.
    """
    _refuse_think_type_dict.setdefault(model, set()).add(_think_type(think_param_list[0]))
    for key in think_param_list.pop(0):
        param_dict.pop(key, None)
    param_dict.update(think_param_list[0])
    _fit_think_budget(param_dict, think_level)
    logger.info("%s does not take the thinking parameters it was called "
                "with. Calling it with %s from now on (%s)", model,
                think_param_list[0] or "none of them",
                str(error)[-config.LOG_EXCERPT_CHAR:])


def _half_output_token(max_token: int) -> int:
    """Return the output limit a call refused for the context length is sent again with.

    Args:
        max_token: The output limit the refused call asked for.

    Returns:
        Half of max_token, or 0 when that is below _MIN_LOWER_OUTPUT_TOKEN.
    """
    half = max_token // 2
    return half if half >= _MIN_LOWER_OUTPUT_TOKEN else 0


def _fit_think_budget(param_dict: dict, think_level: str) -> None:
    """Fit the thinking budget a call carries to its max_tokens.

    The budget becomes what think_budget() gives for the call's max_tokens, and the
    thinking parameters are taken off when that is none. A call without a budget is
    left as it is.

    Args:
        param_dict: The call's keyword arguments, changed in place.
        think_level: The level the call's thinking came from.
    """
    think_dict = param_dict.get("thinking")
    if not isinstance(think_dict, dict) or "budget_tokens" not in think_dict:
        return
    budget = think_budget(think_level, param_dict["max_tokens"])
    if budget:
        param_dict["thinking"] = {**think_dict, "budget_tokens": budget}
    else:
        param_dict.pop("thinking")


def _max_token_setting(setting_name: str) -> str:
    """Return the name of the output limit setting that goes with a model setting.

    Args:
        setting_name: A setting named *_MODEL, e.g. "EXPLORE_MODEL".

    Returns:
        "AGENT_MAX_TOKEN" for "EDIT_MODEL", and the name with _MODEL replaced by
        _MAX_TOKEN for the others, e.g. "EXPLORE_MAX_TOKEN".
    """
    if setting_name == "EDIT_MODEL":
        return "AGENT_MAX_TOKEN"
    return setting_name.removesuffix("_MODEL") + "_MAX_TOKEN"


def _log_fit_output(model: str, fit_token: int, first_token: int,
                    setting_name: str) -> None:
    """Log the output limit a callable keeps after refusals for the context length.

    The first one of each model and output limit setting in this process is a warning,
    the later ones INFO.

    Args:
        model: The model.
        fit_token: The output limit the call went through with.
        first_token: The output limit the call first asked for.
        setting_name: The setting the model came from; the message names the output
            limit setting that goes with it (_max_token_setting).
    """
    max_token_setting = _max_token_setting(setting_name)
    pair = (model, max_token_setting)
    log = logger.info if pair in _fit_warn_pair_set else logger.warning
    _fit_warn_pair_set.add(pair)
    follow_text = "" if max_token_setting == "AGENT_MAX_TOKEN" \
        else ", or AGENT_MAX_TOKEN, which it follows when unset,"
    log("%s refused, for the context length, calls asking for %d output tokens, so the "
        "stage's later calls ask for no more than %d. When the model's context is short, "
        "set %s%s to what the context leaves for the reply", model, first_token,
        fit_token, max_token_setting, follow_text)


def _log_usage(model: str, usage_dict: dict) -> None:
    """Log one call's token counts at DEBUG.

    Args:
        model: The model that was called.
        usage_dict: broker.usage_dict_of()'s dict. An empty one logs nothing.
    """
    if usage_dict:
        logger.debug("%s: %d input token(s), of which %d read from the cache and %d "
                     "written to it; %d output token(s)", model,
                     usage_dict["input_token"], usage_dict["cache_read_token"],
                     usage_dict["cache_write_token"], usage_dict["output_token"])


def make_llm(model: str, max_token: int, setting_name: str = "EDIT_MODEL",
             is_turn_reply: bool = False,
             think_level: str = "") -> Callable[[str | list[tuple[str, bool]]], str]:
    """Build a callable that takes a prompt and returns a string.

    A call refused for the context length to a model whose output limit litellm's
    table does not hold is sent again with half the output limit, down to
    _MIN_LOWER_OUTPUT_TOKEN, unless an earlier call of the same callable went through
    asking for as many output tokens. The limit a halved call takes is kept by the
    callable, and its later calls ask for no more than it. A refusal that is not sent
    again, or that the halving does not get past, is raised as PromptTooLong.

    Args:
        model: A model name in litellm format, e.g. "anthropic/claude-haiku-4-5".
        max_token: The output limit of one call, lowered to the model's own
            (providercall.limit_within()).
        setting_name: The name of the setting model came from, used in the message when
            it is empty, and whose output limit setting (_max_token_setting) the warning
            on a kept output limit names.
        is_turn_reply: Whether the reply is one turn of an RLM loop. Such a reply is
            received as a stream and ends where rlm.reply_end() says
            (providercall.completion()).
        think_level: How much thinking to ask for, as a value of
            config.THINK_LEVEL_TUPLE. The default "" passes nothing and leaves it to the
            provider. A model that will not take these parameters is called with the
            next form think_param_chain() gives from the first refusal on, down to none
            at all, and a form it refused is not sent to it again
            (_refuse_think_type_dict).

    Returns:
        call(prompt) -> str, an empty string when no content block came back. prompt
        is the one user message: a string, or the blocks rlm.build_prompt_block_list()
        returns, sent as content blocks with a cache point where one is marked.
        call.whole(prompt) makes the same call with the reply received whole even when
        is_turn_reply is set. Both leave the last call's finish reason on
        call.last_finish_reason ("length" when max_tokens cut it, "stop" when
        rlm.reply_end() ended it) and the last call's token counts on
        call.last_usage_dict (broker.usage_dict_of()'s dict).

    Raises:
        ValueError: When model is empty, or thinking is not a known level.
    """
    if not model:
        raise ValueError(f"the model name is empty. Set {setting_name}")
    table_limit = providercall.table_output_limit(model)
    max_token = providercall.limit_within(model, max_token, table_limit)
    # Only an output limit litellm's table does not vouch for is lowered on a refusal
    # for the context length
    is_lowerable = not table_limit
    # The forms still to try, the one in use first: the refusal path below drops the
    # head for the rest of the run
    think_param_list = think_param_chain(think_level, max_token)
    # The output limit a halved call took, which later calls ask for no more than, and
    # the largest output limit a call went through with
    fit_token = max_token
    pass_token = 0

    def send(prompt: str | list[tuple[str, bool]], is_turn: bool) -> str:
        """Make one call; is_turn marks the reply as one turn of an RLM loop."""
        nonlocal fit_token, pass_token
        # A form this model has refused before is not sent again
        refuse_set = _refuse_think_type_dict.get(model, set())
        while len(think_param_list) > 1 and _think_type(think_param_list[0]) in refuse_set:
            think_param_list.pop(0)
        content = prompt if isinstance(prompt, str) else content_list_of(prompt)
        param_dict = {"model": model,
                      "max_tokens": fit_token,
                      "num_retries": config.LLM_MAX_RETRY,
                      "messages": [{"role": "user", "content": content}]}
        # Passed only when they hold something; empty values are left out entirely. A
        # call that goes to the broker carries neither
        if not config.LLM_BROKER:
            param_dict.update(config.credential_dict())
            param_dict["timeout"] = config.LLM_CALL_TIMEOUT_SEC
        if is_turn:
            param_dict[providercall.TURN_REPLY_KEY] = True
        param_dict.update(think_param_list[0])
        _fit_think_budget(param_dict, think_level)
        first_token = param_dict["max_tokens"]
        while True:
            try:
                choice, usage_dict = _choice(param_dict)
                break
            except litellm.ContextWindowExceededError as e:
                # A refusal for the context length is sent again with half the output
                # limit while no earlier call went through asking for as many;
                # otherwise it is raised as PromptTooLong. It is a BadRequestError too
                # and is caught before that one
                half = _half_output_token(param_dict["max_tokens"]) \
                    if is_lowerable and pass_token < param_dict["max_tokens"] else 0
                if not half:
                    raise PromptTooLong(f"{type(e).__name__}: {e}") from e
                logger.info("%s refused a call for the context length, so it is sent "
                            "again asking for %d output tokens in place of %d (%s)",
                            model, half, param_dict["max_tokens"],
                            str(e)[-config.LOG_EXCERPT_CHAR:])
                param_dict["max_tokens"] = half
                _fit_think_budget(param_dict, think_level)
            except litellm.BadRequestError as e:
                # A refusal naming the thinking parameters moves to the next form and
                # repeats the call; this model is called with that form from here on
                if not _is_think_refusal(e, think_param_list):
                    raise
                _move_to_next_think_form(model, think_param_list, param_dict,
                                         think_level, e)
        pass_token = max(pass_token, param_dict["max_tokens"])
        if param_dict["max_tokens"] < first_token:
            fit_token = param_dict["max_tokens"]
            _log_fit_output(model, fit_token, first_token, setting_name)
        # The finish reason and the token counts travel on the function's attributes,
        # not in the return value
        call.last_finish_reason = getattr(choice, "finish_reason", None)
        call.last_usage_dict = usage_dict
        _log_usage(model, usage_dict)
        # A reply with no content block at all is reported as None. It is normalised
        # to ""
        if choice.message.content is None:
            logger.info("the reply has no content. finish_reason=%s model=%s",
                        call.last_finish_reason, model)
            return ""
        return choice.message.content

    def call(prompt: str | list[tuple[str, bool]]) -> str:
        """Make one call, its reply marked as a turn's when is_turn_reply is set."""
        return send(prompt, is_turn_reply)

    call.whole = lambda prompt: send(prompt, False)
    call.last_finish_reason = None
    call.last_usage_dict = {}
    return call


@dataclass(frozen=True)
class MissingCredential:
    """One variable a model's provider needs and does not have.

    Attributes:
        model: The model, in litellm form.
        name_list: The settings holding that model.
        variable: The provider's own variable.
        setting: The name to set it under: LLM_API_KEY or LLM_API_BASE when every model
            lacks the same variables of that kind, and the variable itself otherwise.
        other_way_tuple: The variables the provider's calls go through with in place
            of this one (_OTHER_WAY_DICT); empty for a provider with none.
    """
    model: str
    name_list: list[str]
    variable: str
    setting: str
    other_way_tuple: tuple[str, ...] = ()


def _setting_of(variable: str, variable_list_by_model_dict: dict[str, list[str]]) -> str:
    """Return the name a missing variable is set under.

    LLM_API_KEY or LLM_API_BASE is returned for a variable of its kind when every model
    lacks the same variables of that kind: one variable, or the several names one
    provider reads the same value under.

    Args:
        variable: The provider's own variable.
        variable_list_by_model_dict: {model: the variables it lacks}, for every model
            the settings name.

    Returns:
        LLM_API_KEY, LLM_API_BASE, or variable.
    """
    for part, setting in _SHARE_SETTING_DICT.items():
        if part not in variable.lower():
            continue
        kind_set_list = [{other for other in variable_list if part in other.lower()}
                         for variable_list in variable_list_by_model_dict.values()]
        if all(kind_set == kind_set_list[0] for kind_set in kind_set_list):
            return setting
    return variable


def _name_list_by_model_dict() -> dict[str, list[str]]:
    """Return the models the settings name, each with the settings holding it.

    Returns:
        {model: the names of the settings ending in _MODEL that hold it}, in the
        settings' order. A setting holding no model is left out.
    """
    name_list_dict: dict[str, list[str]] = {}
    for name, model in config.setting_value_dict("_MODEL").items():
        if model:
            name_list_dict.setdefault(model, []).append(name)
    return name_list_dict


def provider_of(model: str) -> str:
    """Return the provider litellm calls a model through.

    A model written "<provider>/<model>" with a provider litellm lists is that
    provider's, as it is written. Any other model is looked up by litellm.

    Args:
        model: The model, in litellm form.

    Returns:
        The provider's name, or "" when litellm finds none for the model, whatever it
        raises. No request is made.
    """
    provider, separator, _ = model.partition("/")
    if separator and provider in litellm.provider_list:
        return provider
    try:
        return providercall.look_up_model(
            lambda name: litellm.get_llm_provider(
                model=name, api_base=config.LLM_API_BASE or None)[1] or "",
            model, provider)
    except Exception:                     # noqa: BLE001 - whatever litellm raises
        return ""


def _other_way_tuple(model: str) -> tuple[str, ...]:
    """Return the other variables a model's calls go through with (_OTHER_WAY_DICT).

    They are the ones besides those litellm's check names.

    Args:
        model: The model, in litellm form.

    Returns:
        The variables' names; empty for a provider with none.
    """
    return _OTHER_WAY_DICT.get(provider_of(model), ())


def _has_other_way(model: str) -> bool:
    """Say whether a model's calls go through without the variables litellm names missing.

    Args:
        model: The model, in litellm form.

    Returns:
        True when its provider has variables in _OTHER_WAY_DICT and LLM_API_KEY or one
        of them is set.
    """
    other_way_tuple = _other_way_tuple(model)
    return bool(other_way_tuple) and bool(
        config.LLM_API_KEY or any(os.environ.get(name) for name in other_way_tuple))


def missing_credential_list() -> list[MissingCredential]:
    """Return what a model call made from this process would lack.

    The models asked about are the values of the settings whose name ends in _MODEL
    (config.setting_value_dict()). litellm says which of the provider's own variables
    the model needs and whether it is there: the key passed as LLM_API_KEY, the address
    passed as LLM_API_BASE, or the provider's variable already in the environment. A
    variable litellm names as missing while the environment holds a value for it is left
    out, and so is every variable of a model whose calls go through another way
    (_has_other_way). A model litellm finds no provider for is not asked about here, nor
    is a model of a provider litellm signs in to as it resolves it
    (providercall.look_up_model): no variable of it is missing. No model is called and
    no sign-in is started (sign_in_text() starts them).

    Returns:
        One MissingCredential per model and missing variable, in the settings' order;
        empty when nothing is missing.
    """
    name_list_dict = _name_list_by_model_dict()
    variable_list_dict = {
        model: [] if not provider_of(model) or _has_other_way(model) else [
            variable for variable in providercall.look_up_model(
                lambda name: litellm.validate_environment(
                    name, api_key=config.LLM_API_KEY or None,
                    api_base=config.LLM_API_BASE or None),
                model, {}).get("missing_keys", [])
            if not os.environ.get(variable)]
        for model in name_list_dict}
    return [MissingCredential(model, name_list_dict[model], variable,
                              _setting_of(variable, variable_list_dict),
                              _other_way_tuple(model))
            for model, variable_list in variable_list_dict.items()
            for variable in variable_list]


def no_provider_text() -> str:
    """Say which models of the settings litellm finds no provider for.

    Returns:
        One line per such model, naming the model and the settings holding it; "" when
        every model has a provider.
    """
    return "\n".join(
        f"{model} ({', '.join(name_list)}) names no provider litellm calls. Set "
        f"{', '.join(name_list)} to a model in litellm's form, <provider>/<model>"
        for model, name_list in _name_list_by_model_dict().items()
        if not provider_of(model))


def credential_text(missing_list: list[MissingCredential]) -> str:
    """Say what missing_credential_list() returned, one line per missing variable.

    Args:
        missing_list: The missing variables.

    Returns:
        The lines, each naming the model, the settings holding it, the provider's
        variable and the name to set it under, followed by the variables that can be
        set in its place where the provider has any; "" for an empty list.
    """
    return "\n".join(
        f"{missing.model} ({', '.join(missing.name_list)}) needs {missing.variable}, "
        f"which is not set. Set {missing.setting}"
        + (f", or set {' or '.join(missing.other_way_tuple)} in its place"
           if missing.other_way_tuple else "")
        for missing in missing_list)


def missing_credential_text() -> str:
    """Say what a model call made from this process would lack.

    Returns:
        no_provider_text() followed by credential_text() of missing_credential_list();
        "" when every model has a provider and nothing is missing.
    """
    return "\n".join(text for text in (no_provider_text(),
                                       credential_text(missing_credential_list()))
                     if text)


def sign_in_text() -> str:
    """Have litellm sign in to the providers of the settings' models it signs in to.

    Those are the models providercall.look_up_model() does not look up. Each is resolved
    by litellm.get_llm_provider(), which signs in as a call of the model does: litellm
    reads the sign-in it keeps, or prints what to do to sign in and waits for it.

    Returns:
        One line per model whose sign-in failed, naming the model, the settings holding
        it and what litellm raised, its blanks run together; "" when each of them is
        signed in to.
    """
    name_list_dict = _name_list_by_model_dict()
    line_list = []
    for model in providercall.sign_in_model_list(list(name_list_dict)):
        try:
            litellm.get_llm_provider(model)
        except Exception as e:            # noqa: BLE001 - whatever litellm raises
            line_list.append(f"{model} ({', '.join(name_list_dict[model])}): litellm "
                             f"did not sign in to its provider ({type(e).__name__}: "
                             f"{' '.join(str(e).split())})")
    return "\n".join(line_list)


class _AskNameEnviron(dict):
    """An environment that holds nothing and keeps each name it is asked for.

    Attributes:
        name_list: The names asked for, in order.
    """

    def __init__(self) -> None:
        """Start empty, with no name asked for."""
        super().__init__()
        self.name_list: list[str] = []

    def __contains__(self, name: object) -> bool:
        """Keep name and say it is not held."""
        self.name_list.append(str(name))
        return False

    def __missing__(self, name: str) -> str:
        """Keep name and raise KeyError, as a dict does for a key it does not hold."""
        self.name_list.append(name)
        raise KeyError(name)

    def get(self, name: str, default: str | None = None) -> str | None:
        """Keep name and return default, as a dict does for a key it does not hold."""
        self.name_list.append(name)
        return default


class _EmptyEnvironOs:
    """Stand in for the os module, with an _AskNameEnviron as its environment."""

    def __init__(self) -> None:
        """Start with an empty _AskNameEnviron."""
        self.environ = _AskNameEnviron()

    def __getattr__(self, name: str) -> object:
        """Return the os module's own attribute name."""
        return getattr(os, name)

    def getenv(self, key: str, default: str | None = None) -> str | None:
        """Keep key and return default, as os.getenv() does for a name that is not set."""
        return self.environ.get(key, default)


@functools.lru_cache(maxsize=None)
def _variable_tuple_of(model_tuple: tuple[str, ...]) -> tuple[str, ...]:
    """Return the variables litellm's check reads for the models' providers, set or not.

    litellm.validate_environment() is run with an empty environment in place of this
    process's: it names a variable that is set as it names one that is missing. The
    names are those it asks that environment for and those it returns as missing. The
    process's own environment is not touched. A model of a provider litellm signs in to
    as it resolves it (providercall.look_up_model) is not checked and adds no name.

    Args:
        model_tuple: The models, in litellm form.

    Returns:
        The variables' names, each once, in the models' order. Empty, with a warning,
        when litellm's check cannot be run that way, whatever it raises.
    """
    check = litellm.validate_environment
    empty_os = _EmptyEnvironOs()
    name_list = []
    # Whatever litellm's own code raises when run over the stand-in module is taken
    # as the check not running that way
    try:
        check_empty = config.with_global(check, "os", empty_os)
        for model in model_tuple:
            missing_list = providercall.look_up_model(check_empty, model, {}).get(
                "missing_keys", [])
            name_list += [*empty_os.environ.name_list, *missing_list]
            empty_os.environ.name_list.clear()
    except Exception as e:
        logger.warning("litellm's check could not be run over an empty environment "
                       "(%s: %s), so the providers' own variables are not known: they "
                       "stay in the environment of the programs started here",
                       type(e).__name__, e)
        return ()
    return tuple(dict.fromkeys(name_list))


def provider_variable_tuple() -> tuple[str, ...]:
    """Return the providers' own variables the models of the settings read.

    The models are the values of the settings whose name ends in _MODEL. The names are
    those litellm's check reads for each model's provider, whether set in this process's
    environment or not.

    Returns:
        The variables' names, each once.
    """
    return _variable_tuple_of(tuple(dict.fromkeys(
        model for model in config.setting_value_dict("_MODEL").values() if model)))


def is_credential_name(name: str) -> bool:
    """Say whether a variable's name marks its value as a credential.

    Examples:
        "OPENAI_API_KEY"     -> True
        "AWS_ACCESS_KEY_ID"  -> True
        "AZURE_API_VERSION"  -> False

    Args:
        name: The variable's name.

    Returns:
        True when the name holds one of _CREDENTIAL_NAME_PART_TUPLE.
    """
    return any(part in name for part in _CREDENTIAL_NAME_PART_TUPLE)
