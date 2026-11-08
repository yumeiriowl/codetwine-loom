"""The LLM call site.

litellm.completion is substituted and only the kwargs passed and the handling of the
return value are checked. No API is called.
"""

import functools
import logging
import os
import sys

import litellm
import pytest

from codetwine_loom import broker as broker_mod
from codetwine_loom import config, llm as llm_mod
from codetwine_loom.llm import make_llm


class _FakeCompletion:
    """A stand-in for litellm.completion recording its kwargs and returning a reply."""

    def __init__(self, content: str = "ok", finish_reason: str = "stop"):
        self.kwargs = None
        self._content = content
        self._finish_reason = finish_reason

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        message = type("Message", (), {"content": self._content})()
        choice = type("Choice", (),
                      {"message": message, "finish_reason": self._finish_reason})()
        return type("Response", (), {"choices": [choice]})()


@pytest.fixture(autouse=True)
def no_refusal_known(monkeypatch):
    """Start every test with no model known to refuse a thinking form and no kept output
    limit warned of."""
    monkeypatch.setattr(llm_mod, "_refuse_think_type_dict", {})
    monkeypatch.setattr(llm_mod, "_fit_warn_pair_set", set())


@pytest.fixture
def fake(monkeypatch):
    """Return the stand-in that replaced litellm.completion."""
    completion = _FakeCompletion()
    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    return completion


@pytest.mark.parametrize("info, expect", [
    ({"max_output_tokens": 16384}, 16384), ({"max_output_tokens": 128000}, 32768),
    ({}, 32768), (ValueError("no such model"), 32768)])
def test_a_call_asks_for_no_more_output_than_the_model_takes(fake, monkeypatch, info,
                                                             expect):
    def model_info(model, **kwarg_dict):
        if isinstance(info, Exception):
            raise info
        return info

    monkeypatch.setattr(llm_mod.litellm, "get_model_info", model_info)
    make_llm("m", 32768)("PROMPT")
    assert fake.kwargs["max_tokens"] == expect


def test_a_callable_looks_the_model_up_once(fake, monkeypatch):
    lookup_list = []

    def model_info(model, **kwarg_dict):
        lookup_list.append(model)
        return {"max_output_tokens": 16384}

    monkeypatch.setattr(llm_mod.litellm, "get_model_info", model_info)
    make_llm("m", 32768)("PROMPT")
    assert lookup_list == ["m"] and fake.kwargs["max_tokens"] == 16384


def test_each_attempt_is_given_the_call_timeout(fake, monkeypatch):
    monkeypatch.setattr(config, "LLM_CALL_TIMEOUT_SEC", 7)
    make_llm("m", 100)("hi")
    assert fake.kwargs["timeout"] == 7


def test_a_call_through_the_broker_carries_no_timeout(monkeypatch):
    seen = {}

    def complete(param_dict):
        seen.update(param_dict)
        choice = type("Choice", (), {"message": type("M", (), {"content": "ok"})(),
                                     "finish_reason": "stop", "usage_dict": {}})()
        return choice

    monkeypatch.setattr(config, "LLM_BROKER", "unix:///x.sock")
    monkeypatch.setattr(llm_mod.broker, "complete", complete)
    make_llm("m", 100)("hi")
    assert "timeout" not in seen


def test_off_is_the_least_thinking_for_a_model_that_always_thinks(fake, monkeypatch):
    monkeypatch.setattr(llm_mod.providercall, "always_thinks", lambda model: model == "always")
    make_llm("always", 1000, think_level="off")("hi")
    assert fake.kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fake.kwargs["output_config"] == {"effort": "low"}
    make_llm("other", 1000, think_level="off")("hi")
    assert fake.kwargs["thinking"] == {"type": "disabled"}


def test_a_call_through_the_broker_is_sent_as_it_was_asked(monkeypatch):
    """The broker makes the call, so its litellm decides what the model takes."""
    seen = {}

    def complete(param_dict):
        seen.update(param_dict)
        return type("Choice", (), {"message": type("M", (), {"content": "ok"})(),
                                   "finish_reason": "stop", "usage_dict": {}})()

    monkeypatch.setattr(config, "LLM_BROKER", "unix:///x.sock")
    monkeypatch.setattr(llm_mod.broker, "complete", complete)
    monkeypatch.setattr(llm_mod.providercall, "always_thinks", lambda model: True)
    make_llm("always", 1000, think_level="off")("hi")
    assert seen["thinking"] == {"type": "disabled"}


def test_a_model_the_table_does_not_hold_is_taken_as_one_that_can_stop_thinking(monkeypatch):
    monkeypatch.delenv("USE_LITELLM_PROXY", raising=False)
    assert not llm_mod.providercall.always_thinks("provider/unknown")
    assert not llm_mod.providercall.always_thinks("anthropic/no-such-model")


def test_a_model_that_always_thinks_is_the_one_litellm_drops_disabled_for(monkeypatch):
    monkeypatch.delenv("USE_LITELLM_PROXY", raising=False)
    ask_list = []

    def is_always_on(model, provider):
        ask_list.append((model, provider))
        return model == "claude-x-20261001"

    monkeypatch.setattr(llm_mod.providercall.AnthropicModelInfo,
                        "_is_always_on_thinking_model", staticmethod(is_always_on))
    assert llm_mod.providercall.always_thinks("anthropic/claude-x-20261001")
    assert not llm_mod.providercall.always_thinks("anthropic/claude-y")
    assert ask_list == [("claude-x-20261001", "anthropic"), ("claude-y", "anthropic")]
    param_dict = {"thinking": {"type": "disabled"}}
    llm_mod.providercall.AnthropicModelInfo.maybe_drop_disabled_thinking(
        "claude-x-20261001", param_dict, "anthropic")
    assert param_dict == {}


def test_another_error_of_the_model_lookup_is_raised(monkeypatch):
    def broken(model):
        raise RuntimeError("broken")

    monkeypatch.setattr(llm_mod.providercall.litellm, "get_llm_provider", broken)
    with pytest.raises(RuntimeError):
        llm_mod.providercall.always_thinks("anthropic/m")


def test_an_empty_model_name_fails_naming_the_setting():
    with pytest.raises(ValueError, match="QUERY_EXPAND_MODEL"):
        make_llm("", 100, "QUERY_EXPAND_MODEL")


def test_the_prompt_is_passed_as_one_user_message(fake):
    assert make_llm("m", 100)("PROMPT") == "ok"
    assert fake.kwargs["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert fake.kwargs["max_tokens"] == 100


def test_the_blocks_are_sent_as_content_blocks_with_a_cache_point_where_marked(fake):
    make_llm("m", 100)([("HEAD\n", True), ("TURN\n", True), ("TAIL", False)])
    assert fake.kwargs["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "HEAD\n", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "TURN\n", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "TAIL"}]}]


def test_the_token_counts_are_stored_on_the_attribute(monkeypatch):
    class _Completion(_FakeCompletion):
        def __call__(self, **kwargs):
            response = super().__call__(**kwargs)
            response.usage = type("Usage", (), {"prompt_tokens": 9, "completion_tokens": 2})()
            return response

    monkeypatch.setattr(llm_mod.litellm, "completion", _Completion())
    call = make_llm("m", 100)
    assert call.last_usage_dict == {}   # nothing before the first call
    call("PROMPT")
    assert call.last_usage_dict == {"input_token": 9, "output_token": 2,
                                    "cache_read_token": 0, "cache_write_token": 0,
                                    "estimate_count": 0, "retry_count": 0}


def test_a_turn_reply_is_marked_for_the_call(monkeypatch):
    seen = {}

    def completion(param_dict):
        seen.update(param_dict)
        return _FakeCompletion()()

    monkeypatch.setattr(llm_mod.providercall, "completion", completion)
    make_llm("m", 100, is_turn_reply=True)("PROMPT")
    assert seen[llm_mod.providercall.TURN_REPLY_KEY] is True and "stop" not in seen


def test_a_call_that_takes_the_reply_whole_carries_no_mark(monkeypatch):
    seen_list = []

    def completion(param_dict):
        seen_list.append(dict(param_dict))
        return _FakeCompletion()()

    monkeypatch.setattr(llm_mod.providercall, "completion", completion)
    call = make_llm("m", 100, is_turn_reply=True)
    call.whole("PROMPT")
    call("PROMPT")
    assert [llm_mod.providercall.TURN_REPLY_KEY in seen for seen in seen_list] == [False, True]
    assert call.last_finish_reason is not None


def test_any_other_call_carries_no_mark(fake):
    make_llm("m", 100)("PROMPT")
    assert llm_mod.providercall.TURN_REPLY_KEY not in fake.kwargs


def test_nothing_about_thinking_is_passed_by_default(fake):
    """With THINK_LEVEL "", nothing about thinking is passed and the provider decides."""
    make_llm("m", 100)("PROMPT")
    assert "thinking" not in fake.kwargs and "output_config" not in fake.kwargs


def test_off_turns_thinking_off(fake):
    """The one control both model generations here accept."""
    make_llm("m", 8192, think_level="off")("PROMPT")
    assert fake.kwargs["thinking"] == {"type": "disabled"}
    assert "output_config" not in fake.kwargs


def test_a_level_asks_for_adaptive_thinking_at_that_effort(fake):
    make_llm("m", 8192, think_level="low")("PROMPT")
    assert fake.kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert fake.kwargs["output_config"] == {"effort": "low"}


def test_a_level_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="the thinking level 'medium-ish' is not one of") \
            as raised:
        make_llm("m", 8192, think_level="medium-ish")
    assert "THINK_LEVEL=" not in str(raised.value)


def _refusing(monkeypatch, refused_type_set):
    """Stand in for litellm.completion, refusing the thinking types named.

    Returns:
        The list each call's kwargs are appended to.
    """
    seen = []

    def completion(**kwargs):
        seen.append(kwargs)
        thinking_type = kwargs.get("thinking", {}).get("type")
        if thinking_type in refused_type_set:
            raise llm_mod.litellm.BadRequestError(
                f"{thinking_type} thinking is not supported on this model", "m",
                "anthropic")
        message = type("Message", (), {"content": "ok"})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice]})()

    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    return seen


def test_a_model_refusing_adaptive_thinking_is_given_a_thinking_budget(monkeypatch):
    """A model that refuses adaptive thinking and effort is given budget_tokens."""
    seen = _refusing(monkeypatch, {"adaptive"})
    call = make_llm("m", 16384, think_level="low")
    assert call("PROMPT") == "ok"
    assert len(seen) == 2 and "output_config" not in seen[1]
    assert seen[1]["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    # and the second call does not pay for the refusal again
    assert call("PROMPT") == "ok"
    assert len(seen) == 3 and seen[2]["thinking"]["type"] == "enabled"


def test_a_refusal_is_remembered_by_every_stage_calling_that_model(monkeypatch):
    """The plan stage and the edit stage build calls of their own; the model is refused
    once between them."""
    seen = _refusing(monkeypatch, {"adaptive"})
    built_before = make_llm("m", 16384, think_level="low")
    make_llm("m", 16384, think_level="low")("PROMPT")
    assert len(seen) == 2
    make_llm("m", 8192, think_level="high")("PROMPT")
    assert seen[2]["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    built_before("PROMPT")
    assert seen[3]["thinking"]["type"] == "enabled" and len(seen) == 4


def test_a_refusal_by_one_model_is_not_held_against_another(monkeypatch):
    seen = _refusing(monkeypatch, set())
    llm_mod._refuse_think_type_dict["other"] = {"adaptive"}
    make_llm("m", 16384, think_level="low")("PROMPT")
    assert seen[0]["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_a_model_refusing_every_thinking_form_is_called_without_them(monkeypatch):
    seen = _refusing(monkeypatch, {"adaptive", "enabled"})
    call = make_llm("m", 8192, think_level="low")
    assert call("PROMPT") == "ok"
    assert len(seen) == 3 and "thinking" not in seen[2]
    assert call("PROMPT") == "ok"
    assert len(seen) == 4 and "thinking" not in seen[3]


@pytest.mark.parametrize("level, max_token, budget", [
    ("low", 16384, 2048), ("medium", 16384, 4096), ("high", 16384, 8192),
    ("low", 8192, 1024), ("low", 4096, 1024), ("high", 1024, 0), ("low", 1025, 1024),
    ("off", 16384, 0), ("", 16384, 0)])
def test_the_thinking_budget_is_a_share_of_the_output_limit(level, max_token, budget):
    assert llm_mod.think_budget(level, max_token) == budget


def test_a_limit_with_no_room_for_a_budget_goes_straight_to_no_thinking(monkeypatch):
    seen = _refusing(monkeypatch, {"adaptive"})
    assert make_llm("m", 1024, think_level="low")("PROMPT") == "ok"
    assert len(seen) == 2 and "thinking" not in seen[1]


def test_a_context_window_refusal_comes_back_as_prompt_too_long(monkeypatch):
    def completion(**kwargs):
        raise llm_mod.litellm.ContextWindowExceededError("prompt is too long", "m",
                                                         "anthropic")

    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    with pytest.raises(llm_mod.PromptTooLong, match="ContextWindowExceededError"):
        make_llm("m", 100, think_level="low")("PROMPT")


def test_a_context_window_refusal_naming_thinking_is_not_taken_as_a_thinking_refusal(
        monkeypatch):
    """The subclass is told apart before the word match, so the call is not repeated."""
    seen = []

    def completion(**kwargs):
        seen.append(kwargs)
        raise llm_mod.litellm.ContextWindowExceededError(
            "prompt is too long, even with thinking", "m", "anthropic")

    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    monkeypatch.setattr(llm_mod.litellm, "get_model_info",
                        lambda model, **kwarg_dict: {"max_output_tokens": 8192})
    with pytest.raises(llm_mod.PromptTooLong):
        make_llm("m", 8192, think_level="low")("PROMPT")
    assert len(seen) == 1


def _context_refusing(monkeypatch, fit_token):
    """Stand in for litellm.completion, refusing calls asking for over fit_token output.

    Returns:
        The list each call's max_tokens is appended to.
    """
    seen = []

    def completion(**kwargs):
        seen.append(kwargs["max_tokens"])
        if kwargs["max_tokens"] > fit_token:
            raise llm_mod.litellm.ContextWindowExceededError(
                "maximum context length is 12000 tokens", "m", "hosted_vllm")
        message = type("Message", (), {"content": "ok"})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice]})()

    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    return seen


def test_a_context_refusal_lowers_the_output_limit_of_a_model_the_table_does_not_hold(
        monkeypatch):
    seen = _context_refusing(monkeypatch, 8192)
    call = make_llm("m", 32768)
    assert call("PROMPT") == "ok"
    assert seen == [32768, 16384, 8192]
    # The callable keeps the limit for its next calls
    seen.clear()
    assert call("PROMPT") == "ok"
    assert seen == [8192]


def test_a_new_callable_starts_from_its_own_limit(monkeypatch):
    seen = _context_refusing(monkeypatch, 8192)
    make_llm("m", 32768)("PROMPT")
    seen.clear()
    make_llm("m", 2000)("PROMPT")
    assert make_llm("m", 32768)("PROMPT") == "ok"
    assert seen == [2000, 32768, 16384, 8192]


def test_a_refusal_at_the_kept_limit_is_raised_at_once(monkeypatch):
    _context_refusing(monkeypatch, 8192)
    call = make_llm("m", 32768)
    call("PROMPT")
    seen = _context_refusing(monkeypatch, 4096)
    with pytest.raises(llm_mod.PromptTooLong):
        call("PROMPT")
    assert seen == [8192]


def test_a_refusal_of_a_limit_a_call_went_through_with_is_raised_and_keeps_nothing(
        monkeypatch):
    """The prompt grew past what the context holds: the history folds instead."""
    _context_refusing(monkeypatch, 32768)
    call = make_llm("m", 32768)
    call("PROMPT")
    seen = _context_refusing(monkeypatch, 16384)
    with pytest.raises(llm_mod.PromptTooLong):
        call("PROMPT")
    assert seen == [32768]
    # The next call asks for the whole limit again
    seen = _context_refusing(monkeypatch, 32768)
    call("PROMPT")
    assert seen == [32768]


def test_the_next_call_fits_its_budget_to_the_kept_limit(monkeypatch):
    seen = []

    def completion(**kwargs):
        seen.append(kwargs)
        if kwargs["max_tokens"] > 8192:
            raise llm_mod.litellm.ContextWindowExceededError(
                "maximum context length is 12000 tokens", "m", "hosted_vllm")
        if kwargs.get("thinking", {}).get("type") == "adaptive":
            raise llm_mod.litellm.BadRequestError(
                "adaptive thinking is not supported on this model", "m", "hosted_vllm")
        message = type("Message", (), {"content": "ok"})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice]})()

    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    monkeypatch.setattr(config, "EDIT_MODEL", "m")
    monkeypatch.setattr(config, "AGENT_MAX_TOKEN", 32768)
    call = make_llm("m", 32768, think_level="high")
    call("PROMPT")
    seen.clear()
    assert call("PROMPT") == "ok"
    first = seen[0]
    assert first["max_tokens"] == 8192
    assert first["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    param_dict = {key: first[key] for key in ("model", "max_tokens", "messages", "thinking")}
    assert broker_mod._completion_refusal(param_dict) == ""


def test_a_refusal_through_the_broker_is_halved_with_params_the_broker_takes(
        monkeypatch):
    seen = []

    def call(method, param_dict):
        seen.append(param_dict["max_tokens"])
        assert broker_mod._completion_refusal(param_dict) == ""
        if param_dict["max_tokens"] > 8192:
            broker_mod._raise({"message": "maximum context length is 12000 tokens",
                               "data": {"error_type": broker_mod.CONTEXT_WINDOW}},
                              param_dict)
        return {"content": "ok", "finish_reason": "stop", "usage": {}}

    monkeypatch.setattr(config, "LLM_BROKER", "unix:///x.sock")
    monkeypatch.setattr(config, "EDIT_MODEL", "m")
    monkeypatch.setattr(config, "AGENT_MAX_TOKEN", 32768)
    monkeypatch.setattr(broker_mod, "_call", call)
    assert make_llm("m", 32768, think_level="low")("PROMPT") == "ok"
    assert seen == [32768, 16384, 8192]


@pytest.mark.parametrize("setting_name, expect", [
    ("EDIT_MODEL", "AGENT_MAX_TOKEN"), ("EXPLORE_MODEL", "EXPLORE_MAX_TOKEN"),
    ("QUERY_EXPAND_MODEL", "QUERY_EXPAND_MAX_TOKEN")])
def test_the_kept_limit_warning_names_the_stages_output_limit_setting(
        monkeypatch, caplog, setting_name, expect):
    _context_refusing(monkeypatch, 8192)
    caplog.set_level(logging.WARNING, logger=llm_mod.logger.name)
    make_llm("m", 32768, setting_name)("PROMPT")
    assert f"set {expect}" in caplog.text
    assert ("which it follows when unset" in caplog.text) == (expect != "AGENT_MAX_TOKEN")
    assert hasattr(config, expect)


@pytest.mark.parametrize("setting_name", [
    "ANSWER_MODEL", "DISPATCH_MODEL", "EDIT_MODEL", "EXPLORE_MODEL", "INTEGRATE_MODEL",
    "PLAN_MODEL", "QUERY_EXPAND_MODEL", "REFACTOR_MODEL", "REPORT_MODEL",
    "TEST_CREATE_MODEL"])
def test_every_stage_model_setting_has_its_output_limit_setting(setting_name):
    assert hasattr(config, llm_mod._max_token_setting(setting_name))


def test_each_halving_fits_a_fixed_thinking_budget_to_the_halved_limit(monkeypatch):
    sent = []

    def completion(**kwargs):
        sent.append((kwargs["max_tokens"], kwargs.get("thinking")))
        if kwargs["max_tokens"] > 4096:
            raise llm_mod.litellm.ContextWindowExceededError(
                "maximum context length is 12000 tokens", "m", "hosted_vllm")
        message = type("Message", (), {"content": "ok"})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice]})()

    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    llm_mod._refuse_think_type_dict["m"] = {"adaptive"}
    assert make_llm("m", 32768, think_level="high")("PROMPT") == "ok"
    assert sent == [(32768, {"type": "enabled", "budget_tokens": 16384}),
                    (16384, {"type": "enabled", "budget_tokens": 8192}),
                    (8192, {"type": "enabled", "budget_tokens": 4096}),
                    (4096, {"type": "enabled", "budget_tokens": 2048})]


def test_a_context_refusal_is_not_lowered_for_a_model_the_table_holds(monkeypatch):
    seen = _context_refusing(monkeypatch, 8192)
    monkeypatch.setattr(llm_mod.litellm, "get_model_info",
                        lambda model, **kwarg_dict: {"max_output_tokens": 64000})
    with pytest.raises(llm_mod.PromptTooLong):
        make_llm("m", 32768)("PROMPT")
    assert seen == [32768]


def test_the_halving_stops_at_the_least_output_limit_and_keeps_nothing(monkeypatch):
    seen = _context_refusing(monkeypatch, 0)
    call = make_llm("m", 32768)
    with pytest.raises(llm_mod.PromptTooLong, match="ContextWindowExceededError"):
        call("PROMPT")
    assert seen == [32768, 16384, 8192, 4096]
    seen = _context_refusing(monkeypatch, 32768)
    call("PROMPT")
    assert seen == [32768]


def test_the_first_kept_limit_of_a_model_and_setting_is_a_warning_and_the_rest_info(
        monkeypatch, caplog):
    _context_refusing(monkeypatch, 8192)
    caplog.set_level(logging.INFO, logger=llm_mod.logger.name)
    make_llm("m", 32768)("PROMPT")
    make_llm("m", 32768)("PROMPT")
    make_llm("m", 32768, "PLAN_MODEL")("PROMPT")
    level_list = [record.levelno for record in caplog.records
                  if "for the context length" in record.getMessage()]
    assert level_list == [logging.INFO, logging.INFO, logging.WARNING,
                          logging.INFO, logging.INFO, logging.INFO,
                          logging.INFO, logging.INFO, logging.WARNING]


@pytest.mark.parametrize("level, max_token, expect", [
    ("high", 16384, {"max_tokens": 16384,
                     "thinking": {"type": "enabled", "budget_tokens": 8192}}),
    ("low", 1024, {"max_tokens": 1024}),
    ("low", 8192, {"max_tokens": 8192,
                   "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}})])
def test_a_thinking_budget_is_fitted_to_the_calls_limit(level, max_token, expect):
    think_dict = {"type": "adaptive"} if "output_config" in expect \
        else {"type": "enabled", "budget_tokens": 16384}
    param_dict = {"max_tokens": max_token, "thinking": think_dict,
                  **({"output_config": {"effort": "low"}} if "output_config" in expect
                     else {})}
    llm_mod._fit_think_budget(param_dict, level)
    assert param_dict == expect


def test_a_budget_taken_after_a_halving_fits_the_halved_limit_and_the_broker(monkeypatch):
    """A context refusal comes first, then the model refuses adaptive thinking."""
    seen = []

    def completion(**kwargs):
        seen.append(kwargs)
        if kwargs["max_tokens"] > 8192:
            raise llm_mod.litellm.ContextWindowExceededError(
                "maximum context length is 12000 tokens", "m", "hosted_vllm")
        if kwargs.get("thinking", {}).get("type") == "adaptive":
            raise llm_mod.litellm.BadRequestError(
                "adaptive thinking is not supported on this model", "m", "hosted_vllm")
        message = type("Message", (), {"content": "ok"})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice]})()

    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    monkeypatch.setattr(config, "EDIT_MODEL", "m")
    monkeypatch.setattr(config, "AGENT_MAX_TOKEN", 32768)
    assert make_llm("m", 32768, think_level="high")("PROMPT") == "ok"
    last = seen[-1]
    assert last["max_tokens"] == 8192
    assert last["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    param_dict = {key: last[key] for key in ("model", "max_tokens", "messages", "thinking")}
    assert broker_mod._completion_refusal(param_dict) == ""


def test_a_bad_request_that_is_not_about_thinking_is_raised(monkeypatch):
    def completion(**kwargs):
        raise llm_mod.litellm.BadRequestError("credit balance is too low", "m",
                                              "anthropic")

    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    with pytest.raises(llm_mod.litellm.BadRequestError):
        make_llm("m", 8192, think_level="low")("PROMPT")


def test_an_empty_api_key_is_not_passed(fake, monkeypatch):
    """An empty string would be used by litellm and break environment auth."""
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    make_llm("m", 100)("PROMPT")
    assert "api_key" not in fake.kwargs and "api_base" not in fake.kwargs


def test_a_none_content_is_normalised_to_an_empty_string(monkeypatch):
    """A stop hit on the first generated character returns no content block."""
    monkeypatch.setattr(llm_mod.litellm, "completion", _FakeCompletion(content=None))
    assert make_llm("m", 100)("PROMPT") == ""


def test_the_finish_reason_is_stored_on_the_attribute(monkeypatch):
    """The only path passing "was it cut off" to the RLM without changing the type."""
    monkeypatch.setattr(llm_mod.litellm, "completion",
                        _FakeCompletion(finish_reason="length"))
    call = make_llm("m", 100)
    assert call.last_finish_reason is None   # None before the first call
    call("PROMPT")
    assert call.last_finish_reason == "length"


# == Where the call is made ===================================

def test_the_call_is_made_here_when_no_broker_is_set(fake, monkeypatch):
    """The default is what it has always been: litellm, called in this process."""
    monkeypatch.setattr(config, "LLM_BROKER", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "a-key")
    assert make_llm("m", 100)("PROMPT") == "ok"
    assert fake.kwargs["api_key"] == "a-key"


def test_a_broker_takes_the_call_and_litellm_is_not_touched(fake, monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    sent = []

    def complete(params):
        sent.append(params)
        return broker_mod.Choice(message=broker_mod.Message(content="from the broker"),
                                 finish_reason="stop")

    monkeypatch.setattr(llm_mod.broker, "complete", complete)
    call = make_llm("m", 100)
    assert call("PROMPT") == "from the broker"
    assert call.last_finish_reason == "stop"
    assert sent[0]["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert fake.kwargs is None


def test_the_brokers_reply_carries_its_token_counts(monkeypatch):
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(llm_mod.broker, "complete", lambda params: broker_mod.Choice(
        message=broker_mod.Message(content="x"), finish_reason="stop",
        usage_dict={"input_token": 5, "output_token": 1, "cache_read_token": 4,
                    "cache_write_token": 0}))
    call = make_llm("m", 100)
    call("PROMPT")
    assert call.last_usage_dict["cache_read_token"] == 4


def test_no_credential_is_sent_to_the_broker(monkeypatch):
    """The broker holds them and refuses a call that brought them."""
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(config, "LLM_API_KEY", "a-key")
    monkeypatch.setattr(config, "LLM_API_BASE", "http://somewhere.invalid")
    sent = []
    monkeypatch.setattr(llm_mod.broker, "complete",
                        lambda params: sent.append(params) or broker_mod.Choice(
                            message=broker_mod.Message(content="ok"),
                            finish_reason="stop"))
    make_llm("m", 100)("PROMPT")
    assert "api_key" not in sent[0] and "api_base" not in sent[0]


@pytest.mark.parametrize("think_level", config.THINK_LEVEL_TUPLE)
def test_every_argument_sent_to_the_broker_is_one_it_takes(monkeypatch, think_level):
    """Each form of the thinking parameters is sent once, the last one after two refusals."""
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    monkeypatch.setattr(llm_mod, "_refuse_think_type_dict", {})
    sent = []

    def complete(params):
        sent.append(dict(params))
        if "thinking" in params:
            raise litellm.BadRequestError(message="thinking is not supported", model="m",
                                          llm_provider="anthropic")
        return broker_mod.Choice(message=broker_mod.Message(content="ok"),
                                 finish_reason="stop")

    monkeypatch.setattr(llm_mod.broker, "complete", complete)
    make_llm("m", 16384, is_turn_reply=True, think_level=think_level)("PROMPT")
    assert len(sent) == len(llm_mod.think_param_chain(think_level, 16384))
    for param_dict in sent:
        assert not [name for name in param_dict
                    if name not in broker_mod.COMPLETION_PARAM_TUPLE]


def test_a_refusal_from_the_broker_drops_the_thinking_and_repeats_the_call(monkeypatch):
    """The same handling as a refusal raised here, since it is the same exception."""
    monkeypatch.setattr(config, "LLM_BROKER", "unix:///run/ctloom/llm.sock")
    sent = []

    def complete(params):
        sent.append(dict(params))
        if len(sent) == 1:
            raise litellm.BadRequestError(message="thinking is not supported",
                                          model="m", llm_provider="anthropic")
        return broker_mod.Choice(message=broker_mod.Message(content="ok"),
                                 finish_reason="stop")

    monkeypatch.setattr(llm_mod.broker, "complete", complete)
    call = make_llm("m", 100, think_level="high")
    assert call("PROMPT") == "ok"
    assert "thinking" in sent[0] and "output_config" in sent[0]
    assert "thinking" not in sent[1] and "output_config" not in sent[1]
    # The model is called without them from here on, so a later call sends neither
    call("AGAIN")
    assert "thinking" not in sent[2]


# == What a call from here would lack =========================

def _asked(monkeypatch, missing_dict: dict) -> list:
    """Replace litellm's check with one answering from missing_dict, and record the
    calls."""
    asked = []

    def validate_environment(model, api_key=None, api_base=None, **kwargs):
        asked.append((model, api_key, api_base))
        return {"keys_in_environment": not missing_dict.get(model),
                "missing_keys": list(missing_dict.get(model, []))}

    monkeypatch.setattr(llm_mod.litellm, "validate_environment", validate_environment)
    monkeypatch.setattr(llm_mod.litellm, "get_llm_provider",
                        lambda model, api_base=None: (model, "provider", None, None))
    return asked


def test_nothing_is_missing_when_the_provider_has_what_it_needs(monkeypatch):
    _asked(monkeypatch, {})
    assert llm_mod.missing_credential_text() == ""


def _every_model_is(monkeypatch, model: str) -> None:
    """Set every *_MODEL setting that holds a model to the one model."""
    for name, value in config.setting_value_dict("_MODEL").items():
        if value:
            monkeypatch.setattr(config, name, model)


def test_a_missing_key_names_the_model_its_settings_and_the_setting_to_fill(monkeypatch):
    _every_model_is(monkeypatch, "provider/one")
    _asked(monkeypatch, {"provider/one": ["PROVIDER_API_KEY"]})
    text = llm_mod.missing_credential_text()
    assert "provider/one" in text and "PROVIDER_API_KEY" in text
    assert "EDIT_MODEL" in text and "MEDIUM_MODEL" in text
    assert text.endswith("Set LLM_API_KEY")


def test_a_missing_address_points_at_the_address_setting(monkeypatch):
    _every_model_is(monkeypatch, "provider/local")
    _asked(monkeypatch, {"provider/local": ["PROVIDER_API_BASE"]})
    assert llm_mod.missing_credential_text().endswith("Set LLM_API_BASE")


def test_a_key_only_some_models_lack_is_set_under_the_providers_own_name(monkeypatch):
    """LLM_API_KEY is put on every model's call, the ones that lack nothing among them."""
    _every_model_is(monkeypatch, "provider/one")
    monkeypatch.setattr(config, "HIGH_MODEL", "provider/two")
    _asked(monkeypatch, {"provider/two": ["TWO_API_KEY"]})
    missing_list = llm_mod.missing_credential_list()
    assert [(missing.model, missing.name_list, missing.variable, missing.setting)
            for missing in missing_list] == [
                ("provider/two", ["HIGH_MODEL"], "TWO_API_KEY", "TWO_API_KEY")]
    assert llm_mod.missing_credential_text().endswith("Set TWO_API_KEY")
    assert "provider/one" not in llm_mod.missing_credential_text()


def test_models_that_lack_different_keys_are_each_set_under_their_own_name(monkeypatch):
    _every_model_is(monkeypatch, "provider/one")
    monkeypatch.setattr(config, "HIGH_MODEL", "provider/two")
    _asked(monkeypatch, {"provider/one": ["ONE_API_KEY"], "provider/two": ["TWO_API_KEY"]})
    assert {missing.variable: missing.setting
            for missing in llm_mod.missing_credential_list()} == {
                "ONE_API_KEY": "ONE_API_KEY", "TWO_API_KEY": "TWO_API_KEY"}


def test_a_variable_that_is_neither_a_key_nor_an_address_keeps_its_name(monkeypatch):
    _every_model_is(monkeypatch, "provider/one")
    _asked(monkeypatch, {"provider/one": ["ONE_API_BASE", "ONE_API_VERSION",
                                          "ONE_API_KEY"]})
    assert {missing.variable: missing.setting
            for missing in llm_mod.missing_credential_list()} == {
                "ONE_API_BASE": "LLM_API_BASE", "ONE_API_VERSION": "ONE_API_VERSION",
                "ONE_API_KEY": "LLM_API_KEY"}


def test_a_variable_named_as_missing_that_the_environment_holds_is_left_out(monkeypatch):
    _every_model_is(monkeypatch, "provider/one")
    _asked(monkeypatch, {"provider/one": ["ONE_API_VERSION", "ONE_REGION"]})
    monkeypatch.setenv("ONE_API_VERSION", "v-1")
    assert [missing.variable for missing in llm_mod.missing_credential_list()] == [
        "ONE_REGION"]
    monkeypatch.setenv("ONE_API_VERSION", "")
    assert [missing.variable for missing in llm_mod.missing_credential_list()] == [
        "ONE_API_VERSION", "ONE_REGION"]


def test_a_provider_given_its_key_and_address_lacks_only_what_is_not_set(monkeypatch):
    """The real check, for a provider litellm asks 3 variables of at once."""
    _every_model_is(monkeypatch, "azure/some-model")
    monkeypatch.setattr(config, "LLM_API_KEY", "k-1")
    monkeypatch.setattr(config, "LLM_API_BASE", "https://one.example")
    for name in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"):
        monkeypatch.delenv(name, raising=False)
    assert {missing.variable for missing in llm_mod.missing_credential_list()} == {
        "AZURE_API_VERSION"}
    monkeypatch.setenv("AZURE_API_VERSION", "v-1")
    assert llm_mod.missing_credential_list() == []


def test_a_key_every_model_reads_under_two_names_is_set_under_the_one_setting(monkeypatch):
    _every_model_is(monkeypatch, "provider/one")
    _asked(monkeypatch, {"provider/one": ["ONE_API_KEY", "OTHER_API_KEY"]})
    assert {missing.variable: missing.setting
            for missing in llm_mod.missing_credential_list()} == {
                "ONE_API_KEY": "LLM_API_KEY", "OTHER_API_KEY": "LLM_API_KEY"}


def test_two_names_only_some_models_read_a_key_under_keep_their_names(monkeypatch):
    _every_model_is(monkeypatch, "provider/one")
    monkeypatch.setattr(config, "HIGH_MODEL", "provider/two")
    _asked(monkeypatch, {"provider/one": ["ONE_API_KEY", "OTHER_API_KEY"],
                         "provider/two": ["TWO_API_KEY"]})
    assert {missing.variable: missing.setting
            for missing in llm_mod.missing_credential_list()} == {
                "ONE_API_KEY": "ONE_API_KEY", "OTHER_API_KEY": "OTHER_API_KEY",
                "TWO_API_KEY": "TWO_API_KEY"}


def test_a_provider_with_two_names_for_its_key_is_told_the_one_setting(monkeypatch):
    """The real check, for a provider litellm reads the key of under either of 2 names."""
    _every_model_is(monkeypatch, "gemini/some-model")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert {missing.setting for missing in llm_mod.missing_credential_list()} == {
        "LLM_API_KEY"}


def test_each_model_is_asked_about_once_with_this_process_credentials(monkeypatch):
    """Every *_MODEL setting is read; the same name is asked about once; an empty
    one (REPORT_MODEL="" writes no report) is not asked about."""
    for name in config.env_name_tuple():
        if name.endswith("_MODEL"):
            monkeypatch.setattr(config, name, "provider/same")
    monkeypatch.setattr(config, "REPORT_MODEL", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-1")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    asked = _asked(monkeypatch, {})
    llm_mod.missing_credential_text()
    assert asked == [("provider/same", "sk-1", None)]


def test_the_real_check_names_the_providers_own_variable(monkeypatch):
    """No request is made: litellm only says which variable the provider reads."""
    for name in config.env_name_tuple():
        if name.endswith("_MODEL"):
            monkeypatch.setattr(config, name, "anthropic/some-model")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" in llm_mod.missing_credential_text()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "in-the-environment")
    assert llm_mod.missing_credential_text() == ""


# == a provider whose calls go through another way ============
_BEDROCK_MODEL = "bedrock/some-model"


def _bedrock_alone(monkeypatch) -> None:
    """Set every model to one of Bedrock, with nothing of AWS in the environment."""
    _every_model_is(monkeypatch, _BEDROCK_MODEL)
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)


def test_a_bedrock_model_with_nothing_set_is_told_every_way_to_it(monkeypatch):
    _bedrock_alone(monkeypatch)
    line_list = llm_mod.missing_credential_text().splitlines()
    assert [line.partition(" needs ")[2] for line in line_list] == [
        "AWS_ACCESS_KEY_ID, which is not set. Set AWS_ACCESS_KEY_ID, or set "
        "AWS_PROFILE or AWS_BEARER_TOKEN_BEDROCK in its place",
        "AWS_SECRET_ACCESS_KEY, which is not set. Set AWS_SECRET_ACCESS_KEY, or set "
        "AWS_PROFILE or AWS_BEARER_TOKEN_BEDROCK in its place"]


@pytest.mark.parametrize("name", ["AWS_BEARER_TOKEN_BEDROCK", "AWS_PROFILE", "LLM_API_KEY"])
def test_a_bedrock_model_lacks_nothing_with_a_token_a_profile_or_the_key(monkeypatch,
                                                                         name):
    _bedrock_alone(monkeypatch)
    if name == "LLM_API_KEY":
        monkeypatch.setattr(config, "LLM_API_KEY", "a-bedrock-key")
    else:
        monkeypatch.setenv(name, "value")
    assert llm_mod.missing_credential_text() == ""


def test_another_provider_is_told_no_other_way(monkeypatch):
    _every_model_is(monkeypatch, "anthropic/some-model")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "value")
    assert llm_mod.missing_credential_text().endswith(
        "needs ANTHROPIC_API_KEY, which is not set. Set LLM_API_KEY")


# == a model litellm finds no provider for =====================
@pytest.mark.parametrize("model", ["sonnet", "anthropics/some-model"])
def test_a_model_with_no_provider_is_named_with_its_settings(monkeypatch, capsys, model):
    _every_model_is(monkeypatch, "anthropic/some-model")
    monkeypatch.setattr(config, "HIGH_MODEL", model)
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-of-the-provider")
    assert llm_mod.provider_of(model) == ""
    assert llm_mod.missing_credential_list() == []
    assert llm_mod.missing_credential_text() == (
        f"{model} (HIGH_MODEL) names no provider litellm calls. Set HIGH_MODEL to a "
        f"model in litellm's form, <provider>/<model>")
    assert capsys.readouterr().out == ""


def test_a_model_written_with_a_provider_litellm_lists_is_not_looked_up(monkeypatch):
    """A look-up of some providers starts a sign-in and fails until it is finished."""
    def look_up(*args, **kwargs):
        raise RuntimeError("the sign-in was not finished")

    monkeypatch.setattr(litellm, "get_llm_provider", look_up)
    provider = getattr(litellm.provider_list[0], "value", litellm.provider_list[0])
    assert llm_mod.provider_of(f"{provider}/some-model") == provider
    assert llm_mod.provider_of("some-model") == ""
    _every_model_is(monkeypatch, f"{provider}/some-model")
    assert llm_mod.no_provider_text() == ""


def test_asking_for_a_provider_leaves_standard_output_where_it_is(monkeypatch, capsys):
    stream_list = []

    def look_up(*args, **kwargs):
        stream_list.append(sys.stdout)
        print("what litellm asks of the user")
        raise RuntimeError("no provider")

    monkeypatch.setattr(litellm, "get_llm_provider", look_up)
    stream = sys.stdout
    assert llm_mod.provider_of("some-model") == ""
    assert stream_list == [stream] and sys.stdout is stream
    assert capsys.readouterr().out == "what litellm asks of the user\n"


def test_litellm_is_told_to_print_no_notice_of_its_own():
    assert litellm.suppress_debug_info is True


def test_a_model_with_no_provider_comes_before_what_the_others_lack(monkeypatch):
    _every_model_is(monkeypatch, "anthropic/some-model")
    monkeypatch.setattr(config, "HIGH_MODEL", "sonnet")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    line_list = llm_mod.missing_credential_text().splitlines()
    assert len(line_list) == 2 and "names no provider" in line_list[0]
    assert "needs ANTHROPIC_API_KEY" in line_list[1]


# == the providers' own variables the models read =============
@pytest.fixture
def no_variable_cache():
    """Ask litellm again in this test, and in the tests after it."""
    llm_mod._variable_tuple_of.cache_clear()
    yield
    llm_mod._variable_tuple_of.cache_clear()


def test_a_variable_that_is_set_is_named_as_one_that_is_missing(monkeypatch,
                                                                no_variable_cache):
    """The real check, over an empty environment: this process's own is not touched."""
    _every_model_is(monkeypatch, "anthropic/some-model")
    monkeypatch.setattr(config, "HIGH_MODEL", "openai/some-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "in-the-environment")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    environ_dict = dict(os.environ)
    assert {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} <= set(llm_mod.provider_variable_tuple())
    assert dict(os.environ) == environ_dict


def test_each_variable_is_named_once_and_an_empty_model_is_not_asked_about(
        monkeypatch, no_variable_cache):
    _every_model_is(monkeypatch, "openai/some-model")
    monkeypatch.setattr(config, "REPORT_MODEL", "")
    assert llm_mod.provider_variable_tuple() == ("OPENAI_API_KEY",)


def test_a_variable_the_check_reads_in_place_of_another_is_named_too(monkeypatch,
                                                                     no_variable_cache):
    """The real check: a name it passes on is named beside the one it reports missing."""
    _every_model_is(monkeypatch, "anthropic/some-model")
    monkeypatch.setattr(config, "HIGH_MODEL", "bedrock/some-model")
    name_tuple = llm_mod.provider_variable_tuple()
    assert name_tuple[:2] == ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    assert {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE"} <= set(name_tuple)
    assert len(set(name_tuple)) == len(name_tuple)


def test_the_stand_in_environment_holds_nothing_and_keeps_what_it_is_asked_for():
    empty_os = llm_mod._EmptyEnvironOs()
    assert "ONE" not in empty_os.environ and empty_os.environ.get("TWO", "d") == "d"
    assert empty_os.getenv("THREE") is None and empty_os.path is os.path
    with pytest.raises(KeyError):
        empty_os.environ["FOUR"]
    assert empty_os.environ.name_list == ["ONE", "TWO", "THREE", "FOUR"]


@pytest.mark.parametrize("name, is_credential", [
    ("ONE_API_KEY", True), ("ONE_ACCESS_KEY_ID", True), ("ONE_AUTH_TOKEN", True),
    ("ONE_CLIENT_SECRET", True), ("ONE_API_BASE", False), ("ONE_API_VERSION", False)])
def test_a_name_marks_a_credential_by_the_part_it_holds(name, is_credential):
    assert llm_mod.is_credential_name(name) is is_credential


def test_a_check_that_cannot_be_run_that_way_names_nothing_and_says_so(
        monkeypatch, no_variable_cache, caplog):
    monkeypatch.setattr(llm_mod.litellm, "validate_environment",
                        functools.partial(litellm.validate_environment))
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.llm"):
        assert llm_mod.provider_variable_tuple() == ()
    assert "the providers' own variables are not known" in caplog.text


def test_a_check_that_raises_over_the_stand_in_names_nothing_and_says_so(
        monkeypatch, no_variable_cache, caplog):
    def validate_environment(model=None, api_key=None, api_base=None):
        return {"missing_keys": [os.environ["ONE_API_KEY"]]}
    monkeypatch.setattr(llm_mod.litellm, "validate_environment", validate_environment)
    monkeypatch.setenv("ONE_API_KEY", "k-1")
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.llm"):
        assert llm_mod.provider_variable_tuple() == ()
    assert "KeyError: 'ONE_API_KEY'" in caplog.text
    assert "the providers' own variables are not known" in caplog.text


@pytest.fixture
def sign_in_list(monkeypatch):
    """Record each sign-in litellm starts as it resolves a model, and fail it."""
    start_list = []

    def start(self, model, *args, **kwargs):
        start_list.append(model)
        raise RuntimeError("no sign-in here")

    for config_class in (litellm.GithubCopilotConfig, litellm.ChatGPTConfig):
        monkeypatch.setattr(config_class, "_get_openai_compatible_provider_info", start)
    return start_list


@pytest.mark.parametrize("model", ["github_copilot/gpt-x", "chatgpt/gpt-x"])
def test_no_lookup_of_a_model_signs_in_to_its_provider(monkeypatch, no_variable_cache,
                                                       sign_in_list, model):
    """Its variables and what it lacks are not known; its provider is as written."""
    _every_model_is(monkeypatch, model)
    assert llm_mod.provider_variable_tuple() == ()
    assert llm_mod.missing_credential_list() == []
    assert llm_mod.provider_of(model) == model.partition("/")[0]
    assert not llm_mod.providercall.always_thinks(model)
    assert sign_in_list == []


def test_the_providers_not_looked_up_are_those_litellm_lists_as_signing_in(monkeypatch):
    from litellm.litellm_core_utils import get_llm_provider_logic
    ask_list = []
    monkeypatch.setattr(get_llm_provider_logic, "PROVIDERS_THAT_AUTHENTICATE_ON_PROVIDER_INFO",
                        frozenset({"openai"}))
    look_up = llm_mod.providercall.look_up_model
    assert look_up(lambda model: ask_list.append(model) or "asked", "openai/x", "u") == "u"
    assert look_up(lambda model: ask_list.append(model) or "asked", "github_copilot/x",
                   "u") == "asked"
    assert ask_list == ["github_copilot/x"]
    assert llm_mod.providercall.sign_in_model_list(
        ["openai/x", "anthropic/x", "openai/y"]) == ["openai/x", "openai/y"]


def test_a_model_litellm_signs_in_for_thinks_as_the_model_table_says(monkeypatch,
                                                                     sign_in_list):
    monkeypatch.setitem(litellm.model_cost, "github_copilot/always-x",
                        {"thinking_always_on": True})
    assert llm_mod.providercall.always_thinks("github_copilot/always-x")
    assert not llm_mod.providercall.always_thinks("github_copilot/other-x")
    assert sign_in_list == []


def test_the_sign_in_is_started_once_for_each_model_that_needs_one(monkeypatch,
                                                                   sign_in_list):
    """A failed one is named with the settings holding the model, on one line."""
    _every_model_is(monkeypatch, "anthropic/some-model")
    monkeypatch.setattr(config, "EDIT_MODEL", "github_copilot/gpt-x")
    monkeypatch.setattr(config, "PLAN_MODEL", "github_copilot/gpt-x")
    text = llm_mod.sign_in_text()
    assert sign_in_list == ["gpt-x"]
    assert text.startswith("github_copilot/gpt-x (EDIT_MODEL, PLAN_MODEL): litellm did "
                           "not sign in to its provider (")
    assert "no sign-in here" in text and "\n" not in text
    sign_in_list.clear()
    monkeypatch.setattr(litellm.GithubCopilotConfig, "_get_openai_compatible_provider_info",
                        lambda self, model, *a, **k: (sign_in_list.append(model)
                                                      or ("https://x.invalid", "k",
                                                          "github_copilot")))
    assert llm_mod.sign_in_text() == "" and sign_in_list == ["gpt-x"]
    _every_model_is(monkeypatch, "anthropic/some-model")
    sign_in_list.clear()
    assert llm_mod.sign_in_text() == "" and sign_in_list == []


def test_a_refusal_naming_only_output_config_moves_to_the_thinking_budget(monkeypatch):
    seen = []

    def completion(**kwargs):
        seen.append(kwargs)
        if "output_config" in kwargs:
            raise llm_mod.litellm.BadRequestError(
                'Invalid JSON payload received. Unknown name "output_config"', "m", "gemini")
        message = type("Message", (), {"content": "ok"})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice]})()

    monkeypatch.setattr(llm_mod.litellm, "completion", completion)
    assert make_llm("m", 16384, think_level="low")("PROMPT") == "ok"
    assert "output_config" in seen[0] and "output_config" not in seen[1]
    assert seen[1]["thinking"]["type"] == "enabled"
