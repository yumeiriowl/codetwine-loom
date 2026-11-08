"""config.env() and set_primary_ext()."""

import builtins
import importlib
import importlib.util
import json
import logging
import os
import pathlib
import re
import stat
import subprocess
import sys

import httpx
import pytest
import yarl

from conftest import CERTIFICATE_PEM, git_repo
from codetwine_loom import SettingError, config, llm
from codetwine_loom.languages import LANG_DICT
from codetwine_loom.paths import under_test_path


def test_env_returns_the_default_when_unset(monkeypatch):
    monkeypatch.delenv("DEF_K", raising=False)
    assert config.env("DEF_K", 12) == 12


@pytest.mark.parametrize("name, default", [("DEF_K", 12), ("EXPAND_DECAY", 0.5),
                                           ("QUERY_EXPAND", True),
                                           ("MEDIUM_MODEL", "anthropic/claude-opus-5")])
def test_an_empty_value_is_the_default_of_a_setting_whose_default_is_not_empty(
        monkeypatch, name, default):
    monkeypatch.setenv(name, "")
    assert config.env(name, default) == default


def test_an_empty_value_is_the_value_of_a_setting_whose_default_is_empty(monkeypatch):
    monkeypatch.setenv("LLM_API_BASE", "")
    assert config.env("LLM_API_BASE", "") == ""


@pytest.mark.parametrize("name", ["EDIT_GUIDE", "TEST_GUIDE", "REFACTOR_GUIDE",
                                  "TEST_REFACTOR_GUIDE", "REPORT_MODEL", "THINK_LEVEL",
                                  "PLAN_THINK_LEVEL", "TEST_CREATE_THINK_LEVEL"])
def test_an_empty_value_turns_off_the_settings_that_give_it_a_meaning(monkeypatch, name):
    read = _reloaded(monkeypatch, **{name: ""})
    assert read[name] == ""


def test_an_empty_thinking_level_is_what_the_stages_follow(monkeypatch):
    read = _reloaded(monkeypatch, THINK_LEVEL="", PLAN_THINK_LEVEL=None,
                     EXPLORE_THINK_LEVEL=None)
    assert read == {"THINK_LEVEL": "", "PLAN_THINK_LEVEL": "", "EXPLORE_THINK_LEVEL": ""}


def test_env_checks_bool_before_int(monkeypatch):
    """bool derives from int; checking int first makes int("False") raise."""
    monkeypatch.setenv("INCLUDE_CONSTANT", "False")
    value = config.env("INCLUDE_CONSTANT", False)
    assert value is False
    assert isinstance(value, bool)


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_the_truthy_spellings_of_a_bool(monkeypatch, raw):
    monkeypatch.setenv("QUERY_EXPAND", raw)
    assert config.env("QUERY_EXPAND", False) is True


@pytest.mark.parametrize("raw", ["0", "false", "No", "off"])
def test_the_falsy_spellings_of_a_bool(monkeypatch, raw):
    monkeypatch.setenv("QUERY_EXPAND", raw)
    assert config.env("QUERY_EXPAND", True) is False


def test_a_bool_with_something_else_raises_value_error(monkeypatch):
    monkeypatch.setenv("QUERY_EXPAND", "maybe")
    with pytest.raises(ValueError):
        config.env("QUERY_EXPAND", True)


def test_env_converts_to_int(monkeypatch):
    monkeypatch.setenv("DEF_K", "20")
    value = config.env("DEF_K", 12)
    assert value == 20
    assert isinstance(value, int)


def test_an_int_with_a_non_numeric_string_raises_value_error(monkeypatch):
    monkeypatch.setenv("DEF_K", "abc")
    with pytest.raises(ValueError, match="^DEF_K must be a whole number: abc$"):
        config.env("DEF_K", 12)


def test_an_int_with_a_fraction_names_the_setting(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_WAIT_SEC", "1.5")
    with pytest.raises(ValueError, match="^LLM_RETRY_WAIT_SEC must be a whole number: 1.5$"):
        config.env("LLM_RETRY_WAIT_SEC", 5)


def test_a_float_with_a_non_numeric_string_names_the_setting(monkeypatch):
    monkeypatch.setenv("EXPAND_DECAY", "half")
    with pytest.raises(ValueError, match="^EXPAND_DECAY must be a number: half$"):
        config.env("EXPAND_DECAY", 0.5)


def test_env_converts_to_float(monkeypatch):
    monkeypatch.setenv("EXPAND_DECAY", "0.25")
    assert config.env("EXPAND_DECAY", 0.5) == 0.25


def test_env_returns_a_string_unchanged(monkeypatch):
    monkeypatch.setenv("MEDIUM_MODEL", "openai/gpt-5")
    assert config.env("MEDIUM_MODEL", "anthropic/claude-opus-5") == "openai/gpt-5"


def test_set_primary_ext_rebuilds_the_language_defaults(monkeypatch):
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.delenv("TEST_PATH", raising=False)
    monkeypatch.delenv("TEST_TARGET_STYLE", raising=False)
    config.set_primary_ext("java")
    assert config.PRIMARY_EXT == "java"
    assert config.TEST_COMMAND == ["mvn", "-B", "-fae", "test"]
    assert config.TEST_PATH == "src/test/java"
    assert config.TEST_TARGET_STYLE == "none"
    config.set_primary_ext("py")  # restore it for the tests that follow


# The settings set_primary_ext() rebuilds
_PRIMARY_SETTING_TUPLE = ("PRIMARY_EXT", "TEST_COMMAND", "TEST_PATH", "TEST_TARGET_STYLE",
                          "RUN_COMMAND", "CHECK_COMMAND", "OTHER_CHECK_LIST",
                          "OTHER_TEST_LIST")


@pytest.mark.parametrize("ext, dir_list, expect", [
    ("py", ["testing"], "testing"), ("py", ["test", "testing"], "test"),
    ("py", ["tests", "testing"], "tests"), ("py", [], "tests"),
    ("js", ["tests"], "tests"), ("ts", ["tests"], "tests"), ("cpp", ["test"], "test"),
    ("cs", ["test"], "test"),
    ("java", ["test"], "src/test/java"),
    ("py", ["tests", "testing/a", "testing/b"], "testing")])
def test_the_test_path_is_the_language_s_test_directory_the_repository_holds(
        monkeypatch, tmp_path, ext, dir_list, expect):
    monkeypatch.delenv("TEST_PATH", raising=False)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    for name in _PRIMARY_SETTING_TUPLE:
        monkeypatch.setattr(config, name, getattr(config, name))
    for name in dir_list:
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "test_x.py").write_text("")
    config.set_primary_ext(ext)
    assert config.TEST_PATH == expect
    # The setting wins over what the repository holds
    monkeypatch.setenv("TEST_PATH", "spec")
    config.set_primary_ext(ext)
    assert config.TEST_PATH == "spec"


def test_only_source_files_count_toward_the_test_directory(monkeypatch, tmp_path):
    """Compiled files left in a test directory do not move the choice."""
    monkeypatch.delenv("TEST_PATH", raising=False)
    (tmp_path / "tests" / "__pycache__").mkdir(parents=True)
    (tmp_path / "tests" / "test_a.py").write_text("")
    for n in range(5):
        (tmp_path / "tests" / "__pycache__" / f"test_{n}.cpython-312.pyc").write_text("")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "test_a.py").write_text("")
    (tmp_path / "test" / "test_b.py").write_text("")
    assert config.test_dir_of("py", str(tmp_path)) == "test"


@pytest.mark.parametrize("ext, dir_list, expected", [
    ("ts", [], ("tests",)), ("ts", ["test"], ("tests",)), ("ts", ["tests"], ("test",)),
    ("py", [], ("test", "testing")), ("py", ["testing"], ("tests", "test")),
    ("java", [], ())])
def test_the_other_test_directories_are_the_language_s_besides_the_one_taken(
        tmp_path, ext, dir_list, expected):
    """Every directory the language keeps tests in counts besides the one TEST_PATH
    takes, whether it is there or not."""
    for name in dir_list:
        (tmp_path / name).mkdir()
    test_dir = config.test_dir_of(ext, str(tmp_path))
    assert config.test_path_alt_tuple_of(ext, test_dir) == expected


def test_a_test_path_set_by_hand_counts_alone(monkeypatch, tmp_path):
    """TEST_PATH set in the environment is the tests' only directory."""
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.delenv("TEST_PATH", raising=False)
    config.set_primary_ext("ts")
    assert (config.TEST_PATH, config.TEST_PATH_ALT_TUPLE) == ("test", ("tests",))
    (tmp_path / "tests").mkdir()
    config.set_primary_ext("ts")
    assert (config.TEST_PATH, config.TEST_PATH_ALT_TUPLE) == ("tests", ("test",))
    monkeypatch.setenv("TEST_PATH", "test")
    config.set_primary_ext("ts")
    assert (config.TEST_PATH, config.TEST_PATH_ALT_TUPLE) == ("test", ())


@pytest.mark.parametrize("ext", ["c", "cpp", "h"])
def test_c_and_cpp_run_make_where_the_top_holds_a_makefile(monkeypatch, tmp_path, ext):
    """The launcher takes no offline argument; a CMake project alone has no launcher."""
    monkeypatch.delenv("RUN_COMMAND", raising=False)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "TOOL_OFFLINE", True)
    try:
        (tmp_path / "CMakeLists.txt").write_text("project(x)\n")
        config.set_primary_ext(ext)
        assert config.RUN_COMMAND == []
        (tmp_path / "Makefile").write_text("all:\n")
        config.set_primary_ext(ext)
        assert config.RUN_COMMAND == ["ctloom-run-c"]
        assert config.CHECK_COMMAND == ["ctloom-check-c", "--offline"]
    finally:
        config.set_primary_ext("py")


@pytest.mark.parametrize("name", ["configure", "configure.ac", "configure.in"])
def test_c_runs_make_where_the_top_holds_an_autotools_configure(monkeypatch, tmp_path,
                                                                 name):
    monkeypatch.delenv("RUN_COMMAND", raising=False)
    monkeypatch.setattr(config, "PROJECT_DIR", str(tmp_path))
    for setting in _PRIMARY_SETTING_TUPLE:
        monkeypatch.setattr(config, setting, getattr(config, setting))
    (tmp_path / name).write_text("AC_INIT([x], [1])\n")
    config.set_primary_ext("c")
    assert config.RUN_COMMAND == ["ctloom-run-c"]


@pytest.mark.parametrize("value", ["tests/", "./tests", "tests//", "./tests/"])
def test_test_path_is_read_in_one_form(monkeypatch, value):
    """Every reader of TEST_PATH, under_test_path() among them, sees the one form."""
    monkeypatch.setenv("TEST_PATH", value)
    try:
        config.set_primary_ext("py")
        assert config.TEST_PATH == "tests"
        assert under_test_path(["tests/test_a.py", "a.py"]) == ["tests/test_a.py"]
    finally:
        monkeypatch.delenv("TEST_PATH")
        config.set_primary_ext("py")


def test_a_nested_test_path_keeps_its_parts(monkeypatch):
    monkeypatch.setenv("TEST_PATH", "./src//test/java/")
    try:
        config.set_primary_ext("java")
        assert config.TEST_PATH == "src/test/java"
    finally:
        monkeypatch.delenv("TEST_PATH")
        config.set_primary_ext("py")


@pytest.mark.parametrize("value", ["/repo/tests", "../tests", "tests/../../x", ".."])
def test_a_test_path_outside_the_repository_root_is_refused_naming_it(monkeypatch, value):
    """A path under_test_path() could never match is refused instead of matching
    nothing."""
    monkeypatch.setenv("TEST_PATH", value)
    try:
        with pytest.raises(ValueError, match="TEST_PATH"):
            config.set_primary_ext("py")
    finally:
        monkeypatch.delenv("TEST_PATH")
        config.set_primary_ext("py")


def test_the_repository_root_as_the_test_path_is_kept(monkeypatch):
    """run_test runs the test command over the whole repository then."""
    monkeypatch.setenv("TEST_PATH", "./")
    try:
        config.set_primary_ext("py")
        assert config.TEST_PATH == "."
    finally:
        monkeypatch.delenv("TEST_PATH")
        config.set_primary_ext("py")


def test_the_tool_with_no_setting_of_its_own_is_told_on_the_command_line(monkeypatch):
    """A container with no network sets TOOL_OFFLINE; gradle takes it only here."""
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.setattr(config, "TOOL_OFFLINE", True)
    try:
        config.set_primary_ext("kt")
        assert config.TEST_COMMAND == ["gradle", "test", "--continue", "--console=plain", "--offline"]
        # The tools with a setting of their own are told through the environment
        config.set_primary_ext("py")
        assert config.TEST_COMMAND == list(LANG_DICT["py"].test_command_tuple)
    finally:
        config.set_primary_ext("py")


def test_the_check_of_a_tool_with_no_setting_of_its_own_is_told_too(monkeypatch):
    """CMake configures for the static check as well as for the tests."""
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.delenv("CHECK_COMMAND", raising=False)
    monkeypatch.setattr(config, "TOOL_OFFLINE", True)
    try:
        config.set_primary_ext("c")
        assert config.TEST_COMMAND == ["ctloom-test-c", "--offline"]
        assert config.CHECK_COMMAND == ["ctloom-check-c", "--offline"]
        monkeypatch.setattr(config, "TOOL_OFFLINE", False)
        config.set_primary_ext("c")
        assert config.CHECK_COMMAND == ["ctloom-check-c"]
        monkeypatch.setattr(config, "TOOL_OFFLINE", True)
        monkeypatch.setenv("CHECK_COMMAND", "ctloom-check-c")
        config.set_primary_ext("c")
        assert config.CHECK_COMMAND == ["ctloom-check-c"]
    finally:
        monkeypatch.delenv("CHECK_COMMAND", raising=False)
        config.set_primary_ext("py")


def test_a_language_with_no_test_command_leaves_it_empty(monkeypatch):
    """SQL has no test runner of its own: nothing runs until TEST_COMMAND is set, and
    the offline flag has nothing to be added to.
    """
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.setattr(config, "TOOL_OFFLINE", True)
    try:
        config.set_primary_ext("sql")
        assert config.TEST_COMMAND == []
        assert config.TEST_TARGET_STYLE == "none"
        monkeypatch.setenv("TEST_COMMAND", "pg_prove tests")
        config.set_primary_ext("sql")
        assert config.TEST_COMMAND == ["pg_prove", "tests"]
    finally:
        monkeypatch.delenv("TEST_COMMAND", raising=False)
        config.set_primary_ext("py")


def test_a_test_command_written_by_hand_is_left_alone(monkeypatch):
    """The writer's command is theirs, and carries whatever flag it needs."""
    monkeypatch.setattr(config, "TOOL_OFFLINE", True)
    monkeypatch.setenv("TEST_COMMAND", "my-runner --all")
    try:
        config.set_primary_ext("kt")
        assert config.TEST_COMMAND == ["my-runner", "--all"]
    finally:
        monkeypatch.delenv("TEST_COMMAND", raising=False)
        config.set_primary_ext("py")


def test_set_primary_ext_falls_back_to_py_for_an_unknown_extension(monkeypatch):
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.delenv("TEST_TARGET_STYLE", raising=False)
    config.set_primary_ext("xyz")
    assert config.PRIMARY_EXT == "py"
    config.set_primary_ext("py")


def test_falling_back_says_so_in_the_log(monkeypatch, caplog):
    """A silent Python test command run against another language's repository is the
    one outcome nothing else in the run would point at."""
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.delenv("TEST_TARGET_STYLE", raising=False)
    with caplog.at_level("WARNING"):
        config.set_primary_ext("xyz")
    assert "'xyz'" in caplog.text and "falling back to py" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        config.set_primary_ext("")
    assert "no definition" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        config.set_primary_ext("py")
    assert caplog.text == ""


def test_fallback_ext_has_to_be_a_language_with_a_test_command(monkeypatch):
    """LANG[PRIMARY_EXT] would otherwise raise KeyError with nothing naming the cause."""
    monkeypatch.setenv("FALLBACK_EXT", "go")
    with pytest.raises(ValueError, match="FALLBACK_EXT"):
        importlib.reload(config)
    monkeypatch.delenv("FALLBACK_EXT")
    importlib.reload(config)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_dispatch_max_task_below_1_raises_at_import(monkeypatch, value):
    """-1 would quietly drop the last task of every split, and 0 would fail them all."""
    monkeypatch.setenv("DISPATCH_MAX_TASK", value)
    with pytest.raises(ValueError, match="DISPATCH_MAX_TASK"):
        importlib.reload(config)
    monkeypatch.delenv("DISPATCH_MAX_TASK")
    importlib.reload(config)


def test_the_token_limit_of_a_run_is_off_until_it_is_set(monkeypatch):
    assert _reloaded(monkeypatch, MAX_TOKEN_PER_RUN=None)["MAX_TOKEN_PER_RUN"] == 0
    assert _reloaded(monkeypatch, MAX_TOKEN_PER_RUN="5000")["MAX_TOKEN_PER_RUN"] == 5000


def test_the_token_limit_of_a_run_is_not_an_output_limit_of_a_call():
    """The broker reads the largest setting named *_MAX_TOKEN as what one call may ask."""
    assert "MAX_TOKEN_PER_RUN" not in config.setting_value_dict("_MAX_TOKEN")


def test_the_open_point_count_is_3_and_0_is_taken(monkeypatch):
    assert _reloaded(monkeypatch, DISPATCH_MAX_UNCLEAR=None)["DISPATCH_MAX_UNCLEAR"] == 3
    assert _reloaded(monkeypatch, DISPATCH_MAX_UNCLEAR="0")["DISPATCH_MAX_UNCLEAR"] == 0




def test_a_whitespace_only_test_command_falls_back_to_the_language_default(monkeypatch):
    """Letting an empty argv through makes subprocess.run die with an IndexError."""
    monkeypatch.setenv("TEST_COMMAND", "   ")
    monkeypatch.delenv("TEST_TARGET_STYLE", raising=False)
    config.set_primary_ext("py")
    assert config.TEST_COMMAND[0] == "uv"


def test_a_test_targets_outside_the_3_values_raises_value_error(monkeypatch):
    monkeypatch.setenv("TEST_TARGET_STYLE", "paths")
    with pytest.raises(ValueError):
        config.set_primary_ext("py")
    monkeypatch.delenv("TEST_TARGET_STYLE")
    config.set_primary_ext("py")


def test_gettexts_own_language_variable_is_not_read_as_a_setting(monkeypatch):
    """LANGUAGE=en_US:en is set on most Linux desktops; importing must survive it."""
    monkeypatch.setenv("LANGUAGE", "en_US:en")
    importlib.reload(config)
    assert "LANGUAGE" not in config.env_name_tuple()
    monkeypatch.delenv("LANGUAGE")
    importlib.reload(config)


# == a stage that follows the main one until it is set apart ==

# The stages that call a model of their own. Each has a MODEL, a MAX_TOKENS and, for
# those that run a loop, a THINK_LEVEL; the two the search's step 1 owns are its own
_PER_STAGE_TOKENS = ["PLAN_MAX_TOKEN", "DISPATCH_MAX_TOKEN", "INTEGRATE_MAX_TOKEN",
                     "REPORT_MAX_TOKEN", "EXPLORE_MAX_TOKEN"]
_PER_STAGE_THINKING = ["PLAN_THINK_LEVEL", "DISPATCH_THINK_LEVEL", "INTEGRATE_THINK_LEVEL",
                       "EXPLORE_THINK_LEVEL"]


def _reloaded(monkeypatch, **environment):
    """Reload config with these settings in the environment, then put it back.

    The per-stage settings follow the main one at import, so they are read by importing
    the module again rather than by writing to it.

    Args:
        monkeypatch: pytest's fixture.
        **environment: The names to set, or None for a name to be unset.

    Returns:
        The values of every name asked about, as the reloaded module has them, with
        config itself put back the way this process's environment builds it.
    """
    for name, value in environment.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    try:
        importlib.reload(config)
        return {name: getattr(config, name) for name in environment}
    finally:
        for name in environment:
            monkeypatch.delenv(name, raising=False)
        importlib.reload(config)


@pytest.mark.parametrize("name", _PER_STAGE_TOKENS)
def test_a_stage_left_unset_takes_the_main_output_limit(monkeypatch, name):
    """A stage on a model of its own may want a budget of its own, and one nobody set
    apart must not quietly differ from the loop the run is mostly made of."""
    read = _reloaded(monkeypatch, AGENT_MAX_TOKEN="1234", **{name: None})
    assert read[name] == 1234


@pytest.mark.parametrize("name", _PER_STAGE_TOKENS)
def test_a_stage_that_was_set_apart_keeps_its_own_output_limit(monkeypatch, name):
    read = _reloaded(monkeypatch, AGENT_MAX_TOKEN="1234", **{name: "4321"})
    assert read[name] == 4321 and read["AGENT_MAX_TOKEN"] == 1234


@pytest.mark.parametrize("name", _PER_STAGE_THINKING)
def test_a_stage_left_unset_asks_for_the_main_thinking(monkeypatch, name):
    read = _reloaded(monkeypatch, THINK_LEVEL="high", **{name: None})
    assert read[name] == "high"


@pytest.mark.parametrize("name", _PER_STAGE_THINKING)
def test_a_stage_that_was_set_apart_keeps_its_own_thinking(monkeypatch, name):
    read = _reloaded(monkeypatch, THINK_LEVEL="high", **{name: "off"})
    assert read[name] == "off" and read["THINK_LEVEL"] == "high"


@pytest.mark.parametrize("name", sorted(config.setting_value_dict("THINK_LEVEL")))
@pytest.mark.parametrize("value", ["hgih", "lowest"])
def test_a_thinking_level_out_of_the_choices_stops_the_import_naming_it(monkeypatch, name,
                                                                         value):
    with pytest.raises(ValueError, match=f"^{name} must be one of .*: {value}$"):
        _reloaded(monkeypatch, **{name: value})


@pytest.mark.parametrize("name", sorted(config.setting_value_dict("THINK_LEVEL")))
@pytest.mark.parametrize("value", ["Low", " low", "LOW "])
def test_a_thinking_level_is_read_in_any_case_and_without_blanks(monkeypatch, name, value):
    assert _reloaded(monkeypatch, **{name: value})[name] == "low"


def test_every_thinking_setting_is_checked(monkeypatch):
    name_set = set(config.setting_value_dict("THINK_LEVEL"))
    assert name_set >= {"THINK_LEVEL", *_PER_STAGE_THINKING, "ANSWER_THINK_LEVEL",
                        "TEST_CREATE_THINK_LEVEL", "REFACTOR_THINK_LEVEL"}
    assert all(config.choice_tuple_of(name) == config.THINK_LEVEL_TUPLE
               for name in name_set)


# == the environment a started program is given ==============
def test_the_credentials_are_kept_from_a_program_this_one_starts(monkeypatch):
    for name in config.SECRET_NAME_TUPLE:
        monkeypatch.setenv(name, "something")
    given = config.child_env()
    assert not [name for name in config.SECRET_NAME_TUPLE if name in given]


def test_the_providers_own_variables_are_kept_from_it_too(monkeypatch):
    monkeypatch.setattr(llm, "provider_variable_tuple", lambda: ("ONE_API_KEY",))
    monkeypatch.setenv("ONE_API_KEY", "k-1")
    monkeypatch.setenv("OTHER_API_KEY", "k-2")
    given = config.child_env()
    assert "ONE_API_KEY" not in given and given["OTHER_API_KEY"] == "k-2"
    assert "ONE_API_KEY" not in config.project_env()


def test_the_key_of_the_models_provider_does_not_reach_it(monkeypatch):
    """The real check names the variable, with the settings' own models."""
    for name, value in config.setting_value_dict("_MODEL").items():
        if value:
            monkeypatch.setattr(config, name, "anthropic/some-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "in-the-environment")
    assert "in-the-environment" not in config.child_env().values()
    assert "in-the-environment" not in config.project_env().values()


def test_where_the_broker_listens_is_kept_from_it_too(monkeypatch):
    monkeypatch.setenv("LLM_BROKER", "unix:///run/ctloom/llm.sock")
    assert "LLM_BROKER" not in config.child_env()


def test_everything_else_reaches_the_program_unchanged(monkeypatch):
    """A toolchain reads PATH, HOME and its own cache directories."""
    monkeypatch.setenv("SOME_TOOLCHAIN_CACHE", "/somewhere")
    given = config.child_env()
    assert given["SOME_TOOLCHAIN_CACHE"] == "/somewhere"
    assert "PATH" in given


def test_what_the_caller_adds_wins(monkeypatch):
    monkeypatch.setenv("SOME_FEATURE", "on")
    assert config.child_env({"SOME_FEATURE": "off"})["SOME_FEATURE"] == "off"


def test_this_tool_s_settings_do_not_reach_a_command_of_the_target(monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", "/somewhere/out")
    monkeypatch.setenv("PRESET", "high")
    monkeypatch.setenv("SOME_TOOLCHAIN_CACHE", "/somewhere")
    given = config.project_env({"SMOKE_FEATURE": "on"})
    assert "OUTPUT_DIR" not in given and "PRESET" not in given
    assert given["SOME_TOOLCHAIN_CACHE"] == "/somewhere" and given["SMOKE_FEATURE"] == "on"
    assert not [name for name in config.SECRET_NAME_TUPLE if name in given]
    assert "OUTPUT_DIR" in config.child_env()


def test_the_settings_a_language_check_reads_reach_it(monkeypatch):
    monkeypatch.setenv("COBOL_DIALECT", "ibm")
    assert config.project_env()["COBOL_DIALECT"] == "ibm"


def test_every_setting_a_script_under_sandbox_reads_reaches_the_command():
    """The checks under sandbox/ run as the target's commands, through project_env()."""
    sandbox_dir = pathlib.Path(config.__file__).parent.parent / "sandbox"
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                     for path in sandbox_dir.iterdir() if path.is_file())
    read_set = set(re.findall(r"(?:environ(?:\.get\(|\[)[\"']|process\.env\.|\$\{?)"
                              r"([A-Z][A-Z0-9_]*)", text))
    assert "COBOL_DIALECT" in read_set
    read_set &= set(config.env_name_tuple())
    assert read_set <= set(config.PROJECT_SETTING_TUPLE)


def test_the_credentials_cannot_be_added_back_by_accident(monkeypatch):
    """A caller passing its own environment on would undo the whole thing."""
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    assert "secret-key" not in config.child_env().values()


# == the search presets =======================================
_PRESET_NAMES = ("WEB_SEARCH_URL", "WEB_SEARCH_METHOD", "WEB_SEARCH_QUERY_PARAM",
                 "WEB_SEARCH_PARAM",
                 "WEB_SEARCH_HEADER", "WEB_SEARCH_RESULT_PATH",
                 "WEB_SEARCH_RESULT_FIELD")


def _with_preset(monkeypatch, provider: str, **by_hand) -> dict:
    """Reload config with a preset and a key, and return the endpoint settings.

    Args:
        monkeypatch: pytest's fixture.
        provider: The WEB_SEARCH_PROVIDER to set.
        **by_hand: Endpoint settings to set as well, which win over the preset.

    Returns:
        {setting name: value} for every endpoint setting.
    """
    unset = {name: None for name in _PRESET_NAMES if name not in by_hand}
    return _reloaded(monkeypatch, WEB_SEARCH_PROVIDER=provider,
                     WEB_SEARCH_API_KEY="a-key", **by_hand, **unset)


@pytest.mark.parametrize("provider", sorted(config.WEB_SEARCH_PRESET_DICT))
def test_a_preset_describes_the_endpoint_and_puts_the_key_in(monkeypatch, provider):
    read = _with_preset(monkeypatch, provider)
    preset = config.WEB_SEARCH_PRESET_DICT[provider]
    assert read["WEB_SEARCH_URL"] == preset["url"]
    assert read["WEB_SEARCH_METHOD"] == preset["method"]
    assert read["WEB_SEARCH_QUERY_PARAM"] == preset["query_param"]
    assert read["WEB_SEARCH_RESULT_PATH"] == preset["results_path"]
    assert read["WEB_SEARCH_RESULT_FIELD"] == preset["result_keys"]
    assert read["WEB_SEARCH_PARAM"] == preset.get("params", {})
    assert any("a-key" in value for value in read["WEB_SEARCH_HEADER"].values())
    assert not any("{key}" in value for value in read["WEB_SEARCH_HEADER"].values())


def test_no_preset_leaves_the_endpoint_unset(monkeypatch):
    read = _reloaded(monkeypatch, WEB_SEARCH_PROVIDER=None, WEB_SEARCH_API_KEY=None,
                     **{name: None for name in _PRESET_NAMES})
    assert read["WEB_SEARCH_URL"] == "" and read["WEB_SEARCH_HEADER"] == {}
    assert read["WEB_SEARCH_METHOD"] == "GET"


def test_a_preset_that_is_not_there_says_which_are(monkeypatch):
    with pytest.raises(ValueError, match="^WEB_SEARCH_PROVIDER must be one of .*tavily"):
        _with_preset(monkeypatch, "nowhere")


def test_a_preset_without_a_key_says_which_setting_to_set(monkeypatch):
    provider = sorted(config.WEB_SEARCH_PRESET_DICT)[0]
    with pytest.raises(ValueError, match="WEB_SEARCH_API_KEY"):
        _reloaded(monkeypatch, WEB_SEARCH_PROVIDER=provider, WEB_SEARCH_API_KEY=None)


def test_the_key_is_kept_from_a_program_this_one_starts(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_API_KEY", "a-key")
    assert "WEB_SEARCH_API_KEY" not in config.child_env()


def test_the_model_provider_search_needs_no_key_of_its_own(monkeypatch):
    read = _reloaded(monkeypatch, WEB_SEARCH_PROVIDER=config.WEB_SEARCH_BY_THE_MODEL,
                     WEB_SEARCH_API_KEY=None, WEB_SEARCH_URL=None,
                     WEB_SEARCH_MODEL=None, MEDIUM_MODEL="provider/some-model")
    assert read["WEB_SEARCH_URL"] == ""
    assert read["WEB_SEARCH_MODEL"] == "provider/some-model"


def test_the_searching_model_can_be_set_apart(monkeypatch):
    read = _reloaded(monkeypatch, WEB_SEARCH_MODEL="provider/searcher",
                     MEDIUM_MODEL="provider/editor")
    assert read["WEB_SEARCH_MODEL"] == "provider/searcher"


# == the credentials on a model call ==========================
def test_the_credentials_go_on_a_call_only_when_they_are_set(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_API_BASE", "")
    assert config.credential_dict() == {}
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-1")
    assert config.credential_dict() == {"api_key": "sk-1"}
    monkeypatch.setattr(config, "LLM_API_BASE", "https://llm.invalid/")
    assert config.credential_dict() == {"api_key": "sk-1", "api_base": "https://llm.invalid/"}


# == the .env file ============================================
def test_the_dot_env_file_beside_the_command_fills_in_what_the_shell_left_unset(
        monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("DEF_K=99\n")
    monkeypatch.chdir(tmp_path)
    assert _reloaded(monkeypatch, DEF_K=None)["DEF_K"] == 99


def test_a_dot_env_file_above_the_command_is_not_read(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("DEF_K=99\n")
    (tmp_path / "below").mkdir()
    monkeypatch.chdir(tmp_path / "below")
    assert _reloaded(monkeypatch, DEF_K=None)["DEF_K"] == 12
    assert config.ENV_FILE_PATH == ""


@pytest.mark.parametrize("value, read", [("1", False), ("TRUE", False), ("0", True),
                                         ("", True)])
def test_a_process_told_to_load_no_dot_env_file_reads_none(monkeypatch, tmp_path, value,
                                                          read):
    (tmp_path / ".env").write_text("DEF_K=99\nLLM_API_KEY=from-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config.NO_ENV_FILE_ENV, value)
    got = _reloaded(monkeypatch, DEF_K=None, LLM_API_KEY=None)
    assert (got["DEF_K"], got["LLM_API_KEY"]) == ((99, "from-file") if read else (12, ""))


def test_a_process_of_this_tool_started_without_the_credentials_reads_no_dot_env_file(
        monkeypatch, tmp_path):
    """What the starting process read of .env is in the environment it hands on, and
    what child_env() leaves out of that is not read back in from the file."""
    (tmp_path / ".env").write_text(
        "LLM_API_KEY=from-file\nWEB_SEARCH_PROVIDER=tavily\nRLM_MAX_TURN=17\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "from-file")
    monkeypatch.setenv("RLM_MAX_TURN", "17")
    env_dict = config.own_child_env({"BUILD_EXCLUDE": "x"})
    assert env_dict[config.NO_ENV_FILE_ENV] == "1" and env_dict["BUILD_EXCLUDE"] == "x"
    assert "LLM_API_KEY" not in env_dict and env_dict["RLM_MAX_TURN"] == "17"
    code = ("import os; from codetwine_loom import config, llm; "
            "print(config.ENV_FILE_PATH == '', config.LLM_API_KEY == '', "
            "config.WEB_SEARCH_PROVIDER == '', config.RLM_MAX_TURN, "
            "'LLM_API_KEY' in os.environ)")
    proc = subprocess.run([sys.executable, "-c", code], env=env_dict, check=True,
                          capture_output=True, text=True)
    assert proc.stdout.split() == ["True", "True", "True", "17", "False"]


def test_a_process_of_this_tool_is_started_off_its_working_directory(tmp_path):
    """The working directory holds modules of the names the interpreter imports as it
    starts on a module, and a package of this one's name."""
    target = pathlib.Path(git_repo(tmp_path / "target"))
    (target / "codetwine_loom").mkdir()
    for name in ("warnings.py", "threading.py", "_weakrefset.py",
                 "codetwine_loom/__init__.py", "codetwine_loom/__main__.py"):
        (target / name).write_text(f"open({str(tmp_path / 'ran')!r}, 'w').close()\n",
                                   encoding="utf-8")
    argv = config.own_argv(["history", "--project", ".", "--output-dir",
                            str(tmp_path / "out")])
    assert argv[:2] == [sys.executable, "-P"]
    proc = subprocess.run(
        argv, cwd=target, capture_output=True, text=True, timeout=120,
        env=config.own_child_env({"LITELLM_LOCAL_MODEL_COST_MAP": "True"}))
    assert proc.returncode == 0, proc.stderr
    assert "no attempt recorded yet" in proc.stdout
    assert not (tmp_path / "ran").exists()


def test_a_process_of_this_tool_runs_the_package_the_starting_one_loaded(tmp_path):
    """The package is found nowhere the interpreter looks, and the working directory
    holds another of its name."""
    for place, text in (("own", "the one named"), ("cwd", "the one beside")):
        (tmp_path / place / "stand_in").mkdir(parents=True)
        (tmp_path / place / "stand_in" / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / place / "stand_in" / "__main__.py").write_text(
            "import sys\nfrom stand_in import part\n\n\ndef main():\n"
            "    print(part.TEXT, sys.argv[1:], sys.flags.safe_path)\n", encoding="utf-8")
        (tmp_path / place / "stand_in" / "part.py").write_text(f"TEXT = {text!r}\n",
                                                               encoding="utf-8")
    argv = config.own_argv(["a", "--b"])
    at = argv.index("codetwine_loom")
    argv[at:at + 2] = ["stand_in", str(tmp_path / "own" / "stand_in" / "__init__.py")]
    proc = subprocess.run(argv, cwd=tmp_path / "cwd", capture_output=True, text=True,
                          check=True, timeout=60)
    assert proc.stdout.strip() == "the one named ['a', '--b'] True"


@pytest.mark.parametrize("before", [None, "0", "1"])
def test_the_libraries_are_imported_with_their_own_dot_env_search_off(monkeypatch,
                                                                      before):
    """The switch is on while they are imported and back to what it was afterwards."""
    if before is None:
        monkeypatch.delenv(config.NO_ENV_FILE_ENV, raising=False)
    else:
        monkeypatch.setenv(config.NO_ENV_FILE_ENV, before)
    seen_list = []
    import_module = builtins.__import__

    def record(name, *args, **kwargs):
        seen_list.append((name, os.environ.get(config.NO_ENV_FILE_ENV)))
        return import_module(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", record)
    config._import_without_env_search()
    import_list = list(seen_list)
    monkeypatch.setattr(builtins, "__import__", import_module)
    assert import_list == [("codetwine.config.settings", "1"), ("litellm", "1")]
    assert os.environ.get(config.NO_ENV_FILE_ENV) == before


@pytest.mark.parametrize("first_import", ["from codetwine_loom import cli",
                                          "from codetwine_loom import search_scope"])
def test_a_dot_env_file_above_the_libraries_is_read_by_none_of_them(tmp_path,
                                                                    first_import):
    """The libraries stand under a directory of their own here, a .env file beside
    them, and the command is run from another directory with its own .env file."""
    site_dir, run_dir = tmp_path / "site", tmp_path / "run"
    site_dir.mkdir()
    run_dir.mkdir()
    for name in ("codetwine", "litellm"):
        os.symlink(os.path.dirname(importlib.util.find_spec(name).origin),
                   site_dir / name)
    (site_dir / ".env").write_text("DEF_K=77\nLLM_API_KEY=from-above\nONLY_ABOVE=1\n")
    (run_dir / ".env").write_text("DEF_K=33\nOUTPUT_LANGUAGE=Japanese\n")
    (tmp_path / "probe.py").write_text(
        f"{first_import}  # noqa: F401\n"
        "import os\n"
        "from codetwine.config import settings\n"
        "from codetwine_loom import config\n"
        "print(config.DEF_K, config.LLM_API_KEY == '', 'ONLY_ABOVE' in os.environ,\n"
        "      settings.OUTPUT_LANGUAGE, config.NO_ENV_FILE_ENV in os.environ)\n")
    package_dir = os.path.dirname(os.path.dirname(config.__file__))
    env_dict = {name: value for name, value in os.environ.items()
                if name not in ("DEF_K", "LLM_API_KEY", "OUTPUT_LANGUAGE", "ONLY_ABOVE",
                                config.NO_ENV_FILE_ENV)}
    env_dict.update(PYTHONPATH=os.pathsep.join([str(site_dir), package_dir]),
                    LITELLM_LOCAL_MODEL_COST_MAP="True")
    proc = subprocess.run([sys.executable, str(tmp_path / "probe.py")], env=env_dict,
                          cwd=run_dir, check=True, capture_output=True, text=True)
    assert proc.stdout.split() == ["33", "True", "False", "Japanese", "False"]


def test_the_shell_wins_over_the_dot_env_file(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("DEF_K=99\n")
    monkeypatch.chdir(tmp_path)
    assert _reloaded(monkeypatch, DEF_K="5")["DEF_K"] == 5


def test_the_dot_env_file_fills_in_what_the_shell_left_empty(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("DEF_K=99\nLLM_API_KEY=from-file\nEDIT_GUIDE=\n")
    monkeypatch.chdir(tmp_path)
    read = _reloaded(monkeypatch, DEF_K="", LLM_API_KEY="", EDIT_GUIDE=None)
    assert read == {"DEF_K": 99, "LLM_API_KEY": "from-file", "EDIT_GUIDE": ""}


def test_a_name_the_shell_left_empty_and_the_dot_env_file_does_not_name_stays_empty(
        monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("DEF_K=99\n")
    monkeypatch.chdir(tmp_path)
    read = _reloaded(monkeypatch, LLM_API_BASE="", TEST_GUIDE="")
    assert read == {"LLM_API_BASE": "", "TEST_GUIDE": ""}


@pytest.mark.parametrize("shell_name, file_name", [
    ("https_proxy", "HTTPS_PROXY"), ("HTTPS_PROXY", "https_proxy"),
    ("no_proxy", "NO_PROXY"), ("ALL_PROXY", "all_proxy")])
def test_a_proxy_variable_the_shell_holds_in_one_case_is_not_filled_in_the_other(
        monkeypatch, tmp_path, shell_name, file_name):
    """An HTTP client reads both names, and which of the two it reads first differs."""
    for name in config.PROXY_NAME_TUPLE:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    (tmp_path / ".env").write_text(f"{file_name}=http://from-file.example:1\nDEF_K=99\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(shell_name, "http://from-shell.example:2")
    monkeypatch.delenv("DEF_K", raising=False)
    importlib.reload(config)
    assert config.ENV_FILE_PATH == str(tmp_path / ".env") and config.DEF_K == 99
    assert os.environ[shell_name] == "http://from-shell.example:2"
    assert file_name not in os.environ
    assert config.proxy_name_list() == [shell_name]
    # The value a run reads with the file written is worked out the same way
    config.unload_env_file()
    file_dict = {file_name: "http://from-file.example:1"}
    assert config.run_value(file_name, file_dict) == ""
    assert config.run_value(shell_name, file_dict) == "http://from-shell.example:2"
    assert config.proxy_run_value(file_name.upper(), file_dict) == (
        "http://from-shell.example:2")


def test_the_value_the_http_clients_take_of_a_proxy_variable_is_the_lower_case_one(
        monkeypatch):
    for name in config.PROXY_NAME_TUPLE:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    answer_dict = {"HTTPS_PROXY": "http://upper.example:1", "https_proxy": "http://lower:2",
                   "NO_PROXY": "a.example"}
    assert config.proxy_run_value("HTTPS_PROXY", answer_dict) == "http://lower:2"
    assert config.proxy_run_value("NO_PROXY", answer_dict) == "a.example"
    assert config.proxy_run_value("ALL_PROXY", answer_dict) == ""


def test_the_value_a_run_takes_is_read_as_its_setting_reads_it(monkeypatch):
    """A value the setting does not take, and one of a setting that takes any, are kept
    as they are written; the shell's value wins."""
    for name in ("WEB_SEARCH_PROVIDER", "PRESET", "LLM_API_BASE", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    answer_dict = {"WEB_SEARCH_PROVIDER": " Tavily ", "PRESET": "hgih",
                   "LLM_API_BASE": "HTTP://Llm/v1", "NO_PROXY": "a.example",
                   "no_proxy": "b.example"}
    assert config.run_setting_value("WEB_SEARCH_PROVIDER", answer_dict) == "tavily"
    assert config.run_setting_value("PRESET", answer_dict) == "hgih"
    assert config.run_setting_value("LLM_API_BASE", answer_dict) == "HTTP://Llm/v1"
    assert config.run_setting_value("NO_PROXY", answer_dict) == "b.example"
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "BRAVE")
    assert config.run_setting_value("WEB_SEARCH_PROVIDER", answer_dict) == "brave"
    assert config.setting_form("PRESET", " High") == "high"


def test_a_value_its_setting_reads_alike_is_not_written_over(tmp_path):
    path = tmp_path / ".env"
    before = "PRESET=High\nWEB_SEARCH_PROVIDER=' Tavily'\nWEB_SEARCH_API_KEY=tv-1\n"
    path.write_text(before, encoding="utf-8")
    current_dict = config.env_file_value_dict(str(path))
    assert not config.rewrite_env_file(str(path), current_dict, {
        **current_dict, "PRESET": "high", "WEB_SEARCH_PROVIDER": "TAVILY"})
    assert path.read_text(encoding="utf-8") == before
    assert config.rewrite_env_file(str(path), current_dict,
                                   {**current_dict, "PRESET": "low"})
    assert path.read_text(encoding="utf-8").startswith("PRESET=low\n")


def test_the_dot_env_file_does_not_turn_the_loading_of_dot_env_files_off(monkeypatch,
                                                                          tmp_path):
    """The file read stays read when config is read again, and the switch is not set."""
    monkeypatch.delenv(config.NO_ENV_FILE_ENV, raising=False)
    monkeypatch.delenv("DEF_K", raising=False)
    (tmp_path / ".env").write_text(f"{config.NO_ENV_FILE_ENV}=1\nDEF_K=99\n")
    monkeypatch.chdir(tmp_path)
    importlib.reload(config)
    importlib.reload(config)
    assert config.ENV_FILE_PATH == str(tmp_path / ".env") and config.DEF_K == 99
    assert not config.is_env_file_off() and config.NO_ENV_FILE_ENV not in os.environ
    assert config.run_value(config.NO_ENV_FILE_ENV, {config.NO_ENV_FILE_ENV: "1"}) == ""


def test_a_proxy_variable_the_shell_holds_in_neither_case_is_filled_as_the_file_names_it(
        monkeypatch, tmp_path):
    for name in config.PROXY_NAME_TUPLE:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    (tmp_path / ".env").write_text(
        "HTTPS_PROXY=http://from-file.example:1\nno_proxy=a.example\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HTTP_PROXY", "")
    importlib.reload(config)
    assert config.proxy_name_list() == ["HTTPS_PROXY", "no_proxy"]
    assert os.environ["HTTP_PROXY"] == ""


@pytest.mark.parametrize("shell_dict, want", [
    ({"SSL_CERT_FILE": "ca.pem"}, "ca.pem"),
    ({"SSL_CERT_FILE": "ca.pem", "REQUESTS_CA_BUNDLE": ""}, "ca.pem"),
    ({"SSL_CERT_FILE": "ca.pem", "REQUESTS_CA_BUNDLE": "/etc/other.pem"},
     "/etc/other.pem"),
    ({"SSL_CERT_FILE": ""}, None),
    ({}, None)])
def test_the_libraries_that_read_requests_ca_bundle_are_given_the_ca_file(
        monkeypatch, tmp_path, shell_dict, want):
    (tmp_path / "ca.pem").write_text(CERTIFICATE_PEM, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in shell_dict.items():
        monkeypatch.setenv(name, value)
    importlib.reload(config)
    # A file named relative to this directory is handed on by its absolute path
    want = str(tmp_path / want) if want == "ca.pem" else want
    assert (os.environ.get(config.REQUESTS_CA_FILE_ENV) or None) == want
    if shell_dict.get("SSL_CERT_FILE") == "ca.pem":
        assert os.environ["SSL_CERT_FILE"] == str(tmp_path / "ca.pem")
    else:
        assert os.environ.get("SSL_CERT_FILE") == shell_dict.get("SSL_CERT_FILE")


@pytest.mark.parametrize("content", [
    b"", b"not a certificate\n", b"0\x82\x01\x82\x30\x82",
    b"-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n"])
def test_a_ca_file_that_holds_no_pem_certificate_stops_in_one_line(monkeypatch, tmp_path,
                                                                 content):
    """litellm reads the file when it is imported, and stops there with a traceback."""
    ca_file = tmp_path / "ca.cer"
    ca_file.write_bytes(content)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", "ca.cer")
    with pytest.raises(config.SettingError) as error:
        importlib.reload(config)
    assert str(error.value).startswith(
        f"SSL_CERT_FILE names {ca_file}, which cannot be read as certificate authorities (")
    assert "Point it at a PEM file" in str(error.value)
    assert "REQUESTS_CA_BUNDLE" not in os.environ
    monkeypatch.delenv("SSL_CERT_FILE")
    importlib.reload(config)
    assert config.ca_file_error(str(ca_file)) and not config.ca_file_error(
        _pem_file(tmp_path))


@pytest.mark.parametrize("value", ["no-such.pem", "~/ca.pem", "."])
def test_a_ca_file_that_is_not_there_stops_in_one_line(monkeypatch, tmp_path, value):
    """The HTTP client litellm fetches its tables with stops at one with a traceback."""
    (tmp_path / "ca.pem").write_text(CERTIFICATE_PEM, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", value)
    with pytest.raises(config.SettingError) as error:
        importlib.reload(config)
    assert str(error.value) == (
        f"SSL_CERT_FILE names {value}, which is not a file. Write the path of a PEM "
        f"file out in full (a leading ~ is not expanded), or unset it")
    assert "REQUESTS_CA_BUNDLE" not in os.environ
    monkeypatch.delenv("SSL_CERT_FILE")
    importlib.reload(config)


def _pem_file(directory) -> str:
    """Write a certificate to a file in directory and return the file's path."""
    path = directory / "good.pem"
    path.write_text(CERTIFICATE_PEM, encoding="utf-8")
    return str(path)


def test_the_ca_file_the_dot_env_file_names_is_followed_too(monkeypatch, tmp_path):
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        monkeypatch.delenv(name, raising=False)
    ca_file = tmp_path / "from-file.pem"
    ca_file.write_text(CERTIFICATE_PEM, encoding="utf-8")
    (tmp_path / ".env").write_text(f"SSL_CERT_FILE={ca_file}\n")
    monkeypatch.chdir(tmp_path)
    importlib.reload(config)
    assert os.environ["REQUESTS_CA_BUNDLE"] == str(ca_file)


@pytest.mark.parametrize("shell_name, file_name", [
    ("https_proxy", "HTTPS_PROXY"), ("NO_PROXY", "no_proxy")])
def test_a_proxy_variable_the_shell_holds_empty_in_one_case_is_unset_when_the_file_fills_the_other(
        monkeypatch, tmp_path, shell_name, file_name):
    """An HTTP client reads the empty name in lower case as no proxy at all."""
    for name in config.PROXY_NAME_TUPLE:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    (tmp_path / ".env").write_text(f"{file_name}=from-file.example:1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(shell_name, "")
    monkeypatch.setenv("ALL_PROXY", "")
    importlib.reload(config)
    assert os.environ[file_name] == "from-file.example:1" and shell_name not in os.environ
    assert os.environ["ALL_PROXY"] == ""
    assert config.proxy_name_list() == [file_name]


def test_a_proxy_variable_empty_in_one_case_is_unset_with_no_dot_env_file_too(
        monkeypatch, tmp_path):
    for name in config.PROXY_NAME_TUPLE:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HTTPS_PROXY", "http://from-shell.example:2")
    monkeypatch.setenv("https_proxy", "")
    monkeypatch.setenv("no_proxy", "")
    importlib.reload(config)
    assert config.ENV_FILE_PATH == "" and "https_proxy" not in os.environ
    assert os.environ["no_proxy"] == "" and config.proxy_name_list() == ["HTTPS_PROXY"]
    import urllib.request
    assert urllib.request.getproxies()["https"] == "http://from-shell.example:2"


def test_a_program_this_process_starts_is_given_the_proxy_variables_and_the_ca_file(
        monkeypatch):
    """A container engine fills the names it is handed from the environment it is
    started with."""
    for name in ("HTTPS_PROXY", "no_proxy", config.CA_FILE_ENV):
        monkeypatch.setenv(name, f"value-of-{name}")
    for child_dict in (config.child_env(), config.own_child_env()):
        assert {name: child_dict.get(name) for name in (
            "HTTPS_PROXY", "no_proxy", config.CA_FILE_ENV)} == {
                "HTTPS_PROXY": "value-of-HTTPS_PROXY", "no_proxy": "value-of-no_proxy",
                config.CA_FILE_ENV: f"value-of-{config.CA_FILE_ENV}"}


def _repo_with_dot_env(path, track: bool) -> None:
    """Make a git repository at path with a .env file in it, in git's index or not."""
    git_repo(path)
    (path / ".env").write_text("DEF_K=99\nCONTAINER_ENGINE=/bin/other\n")
    if track:
        subprocess.run(["git", "-C", str(path), "add", "-f", ".env"], check=True,
                       capture_output=True)


def test_a_dot_env_file_git_tracks_is_not_read_and_its_settings_are_named(monkeypatch,
                                                                         tmp_path, caplog):
    _repo_with_dot_env(tmp_path / "repo", track=True)
    monkeypatch.chdir(tmp_path / "repo")
    with caplog.at_level("WARNING", logger="codetwine_loom.config"):
        for name in ("DEF_K", "CONTAINER_ENGINE"):
            monkeypatch.delenv(name, raising=False)
        importlib.reload(config)
    assert (config.DEF_K, config.CONTAINER_ENGINE, config.ENV_FILE_PATH,
            config.ENV_FILE_SKIP_PATH, config.ENV_FILE_SKIP_STATE) \
        == (12, "", "", str(tmp_path / "repo" / ".env"), config.INDEX_HOLD)
    assert config.env_file_skip_text() == "is tracked by git and was not read"
    assert config.skip_env_setting_list() == ["CONTAINER_ENGINE", "DEF_K"]
    assert caplog.records == []
    assert "DEF_K" not in os.environ and "CONTAINER_ENGINE" not in os.environ


@pytest.mark.parametrize("text, name_list", [
    ("DATABASE_URL=postgres://x\nSECRET_KEY_BASE=abc\nexport PORT=3000\n", []),
    ("", []),
    ("# DEF_K=99\nDEF_KK=1\ndef_k=2\n", []),
    ("PORT=3000\nexport WARM_MAX_REFILL=0\nMAX_TOKEN_PER_RUN = 5000\n",
     ["MAX_TOKEN_PER_RUN", "WARM_MAX_REFILL"]),
    ("ANTHROPIC_API_KEY=sk-x\nOPENAI_API_KEY=sk-y\n", ["ANTHROPIC_API_KEY"]),
    ("DOC_TEMPLATE_PATH=${HOME}/t.json\nX=$(touch never)\n", ["DOC_TEMPLATE_PATH"]),
    ("\xff\xfeDEF_K=1\nRLM_MAX_TURN=3\n", ["RLM_MAX_TURN"]),
    # The names every HTTP client reads are the repository's own as much as this tool's
    ("HTTPS_PROXY=http://proxy.corp.example:3128\nNO_PROXY=localhost\nhttp_proxy=x\n"
     "SSL_CERT_FILE=/etc/ca.pem\nALL_PROXY=x\n", []),
    ("HTTPS_PROXY=http://proxy.corp.example:3128\nDEF_K=3\n", ["DEF_K"]),
])
def test_the_settings_of_this_tool_an_unread_dot_env_file_sets_are_read_by_name_alone(
        monkeypatch, tmp_path, text, name_list):
    """The values are neither expanded nor put into the environment."""
    git_repo(tmp_path)
    (tmp_path / ".env").write_bytes(text.encode("latin-1"))
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", ".env"], check=True,
                   capture_output=True)
    monkeypatch.chdir(tmp_path)
    for name in ("MEDIUM_MODEL", "DEF_K", "RLM_MAX_TURN", "MAX_TOKEN_PER_RUN",
                 "WARM_MAX_REFILL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PRESET", "medium")
    importlib.reload(config)
    environ_before = dict(os.environ)
    assert config.ENV_FILE_SKIP_STATE == config.INDEX_HOLD
    assert config.skip_env_setting_list() == name_list
    assert dict(os.environ) == environ_before and not (tmp_path / "never").exists()


def test_a_setting_the_shell_holds_is_not_one_an_unread_dot_env_file_sets(monkeypatch,
                                                                         tmp_path):
    """The file fills only the names the shell left unset or empty."""
    _repo_with_dot_env(tmp_path, track=True)
    monkeypatch.chdir(tmp_path)
    for shell_dict, name_list in (
            ({"DEF_K": None, "CONTAINER_ENGINE": None}, ["CONTAINER_ENGINE", "DEF_K"]),
            ({"DEF_K": "7", "CONTAINER_ENGINE": None}, ["CONTAINER_ENGINE"]),
            ({"DEF_K": "", "CONTAINER_ENGINE": "/bin/e"}, ["DEF_K"]),
            ({"DEF_K": "7", "CONTAINER_ENGINE": "/bin/e"}, [])):
        for name, value in shell_dict.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        importlib.reload(config)
        assert config.ENV_FILE_SKIP_STATE == config.INDEX_HOLD
        assert config.skip_env_setting_list() == name_list, shell_dict


def test_the_lines_of_an_unread_dot_env_file_that_cannot_be_read_are_passed_over_in_silence(
        monkeypatch, tmp_path, caplog):
    git_repo(tmp_path)
    (tmp_path / ".env").write_text("FOO BAR=baz\nPORT=1\n=x\nDEF_K=3\n'open\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-f", ".env"], check=True,
                   capture_output=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEF_K", raising=False)
    level = logging.getLogger("dotenv.main").level
    with caplog.at_level("WARNING"):
        importlib.reload(config)
        assert config.skip_env_setting_list() == ["DEF_K"]
    assert caplog.records == []
    assert logging.getLogger("dotenv.main").level == level


def test_a_dot_env_file_left_unread_by_the_switch_is_named_and_stops_nothing(monkeypatch,
                                                                            tmp_path):
    (tmp_path / ".env").write_text("DEF_K=99\nLLM_API_KEY=from-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config.NO_ENV_FILE_ENV, "1")
    monkeypatch.delenv("DEF_K", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("git was asked"))
    importlib.reload(config)
    assert (config.DEF_K, config.ENV_FILE_PATH, config.ENV_FILE_SKIP_PATH,
            config.ENV_FILE_SKIP_STATE) \
        == (12, "", str(tmp_path / ".env"), config.ENV_FILE_OFF)
    assert config.is_env_file_off() is True and config.skip_env_setting_list() == []
    assert config.env_file_skip_text() == (
        "was not read: PYTHON_DOTENV_DISABLED turns the loading of .env files off")


def test_no_dot_env_file_left_unread_sets_nothing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    importlib.reload(config)
    assert config.ENV_FILE_SKIP_PATH == "" and config.skip_env_setting_list() == []


def test_git_is_asked_about_a_dot_env_file_once_in_a_process(monkeypatch, tmp_path):
    """What git answered the first time stands for every later read of the settings."""
    _repo_with_dot_env(tmp_path / "repo", track=False)
    monkeypatch.chdir(tmp_path / "repo")
    monkeypatch.delenv("DEF_K", raising=False)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    importlib.reload(config)
    assert (config.DEF_K, config.ENV_FILE_SKIP_STATE) == (12, config.INDEX_UNKNOWN)
    monkeypatch.delenv("GIT_TEST_ASSUME_DIFFERENT_OWNER")
    assert config.git_index_state(str(tmp_path / "repo" / ".env")) == config.INDEX_FREE
    importlib.reload(config)
    assert (config.DEF_K, config.ENV_FILE_PATH, config.ENV_FILE_SKIP_STATE) \
        == (12, "", config.INDEX_UNKNOWN)
    assert "DEF_K" not in os.environ


def test_a_dot_env_file_this_process_wrote_is_read_without_asking_git(monkeypatch,
                                                                      tmp_path):
    git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEF_K", raising=False)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    importlib.reload(config)
    (tmp_path / ".env").write_text("DEF_K=99\n")
    config.accept_env_file(str(tmp_path / ".env"))
    importlib.reload(config)
    assert (config.DEF_K, config.ENV_FILE_PATH) == (99, str(tmp_path / ".env"))


def test_a_dot_env_file_git_does_not_track_is_read_inside_a_repository(monkeypatch,
                                                                       tmp_path):
    _repo_with_dot_env(tmp_path / "repo", track=False)
    (tmp_path / "repo" / ".gitignore").write_text(".env\n")
    monkeypatch.chdir(tmp_path / "repo")
    assert _reloaded(monkeypatch, DEF_K=None)["DEF_K"] == 99


def test_a_tracked_dot_env_file_is_read_as_before_where_git_is_not_installed(
        monkeypatch, tmp_path):
    _repo_with_dot_env(tmp_path / "repo", track=True)
    monkeypatch.chdir(tmp_path / "repo")
    (tmp_path / "bin").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    assert config.git_index_state(str(tmp_path / "repo" / ".env")) == config.INDEX_FREE
    assert _reloaded(monkeypatch, DEF_K=None)["DEF_K"] == 99


def test_git_is_asked_about_the_directory_of_the_file_whatever_the_shell_names(
        monkeypatch, tmp_path):
    _repo_with_dot_env(tmp_path / "repo", track=True)
    git_repo(tmp_path / "other")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "other" / ".git"))
    assert config.git_index_state(str(tmp_path / "repo" / ".env")) == config.INDEX_HOLD
    assert config.git_index_state(str(tmp_path / "repo" / "none.env")) \
        == config.INDEX_FREE
    assert config.git_index_state(str(tmp_path / ".env")) == config.INDEX_FREE


@pytest.mark.parametrize("name", [".ENV", ".Env", "sub/.eNV"])
def test_a_tracked_file_whose_name_differs_only_in_case_is_one_git_tracks(
        monkeypatch, tmp_path, name):
    """What a file system that folds case opens for ".env" is the file the index names
    in another case; git is asked for the name whatever its case."""
    repo = tmp_path / "repo"
    git_repo(repo)
    (repo / name).parent.mkdir(exist_ok=True)
    (repo / name).write_text("DEF_K=99\n")
    subprocess.run(["git", "-C", str(repo), "add", "-f", name], check=True,
                   capture_output=True)
    directory = (repo / name).parent
    assert config.git_index_state(str(directory / ".env")) == config.INDEX_HOLD
    assert config.git_index_state(str(directory / "x.env")) == config.INDEX_FREE
    # The file the user put beside it is not read either
    (directory / ".env").write_text("DEF_K=98\n")
    monkeypatch.chdir(directory)
    monkeypatch.delenv("DEF_K", raising=False)
    importlib.reload(config)
    assert (config.DEF_K, config.ENV_FILE_PATH, config.ENV_FILE_SKIP_PATH) \
        == (12, "", str(directory / ".env"))


def test_a_name_with_glob_characters_is_asked_for_as_it_is_written(tmp_path):
    repo = tmp_path / "repo"
    git_repo(repo)
    (repo / "ab.env").write_text("x\n")
    subprocess.run(["git", "-C", str(repo), "add", "-f", "ab.env"], check=True,
                   capture_output=True)
    assert config.git_index_state(str(repo / "a?.env")) == config.INDEX_FREE
    assert config.git_index_state(str(repo / "*.env")) == config.INDEX_FREE
    assert config.git_index_state(str(repo / "AB.ENV")) == config.INDEX_HOLD


def test_the_question_starts_no_program_the_repository_names(tmp_path):
    repo = tmp_path / "repo"
    _repo_with_dot_env(repo, track=True)
    mark = tmp_path / "mark"
    program = tmp_path / "monitor.sh"
    program.write_text(f"#!/bin/sh\necho ran >> {mark}\n")
    program.chmod(0o755)
    (repo / "hook").mkdir()
    for name in ("post-index-change", "reference-transaction"):
        (repo / "hook" / name).write_text(f"#!/bin/sh\necho ran >> {mark}\n")
        (repo / "hook" / name).chmod(0o755)
    for name, value in (("core.fsmonitor", str(program)),
                        ("core.hooksPath", str(repo / "hook"))):
        subprocess.run(["git", "-C", str(repo), "config", name, value], check=True)
    assert config.git_index_state(str(repo / ".env")) == config.INDEX_HOLD
    assert not mark.exists()


def test_git_not_answering_in_time_leaves_a_file_in_a_repository_unread(
        monkeypatch, tmp_path):
    def late(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    _repo_with_dot_env(tmp_path / "repo", track=True)
    (tmp_path / "plain").mkdir()
    monkeypatch.setattr(config.subprocess, "run", late)
    assert config.git_index_state(str(tmp_path / "repo" / ".env")) \
        == config.INDEX_UNKNOWN
    assert config.git_index_state(str(tmp_path / "plain" / ".env")) == config.INDEX_FREE


def test_a_directory_that_is_not_there_is_answered_by_what_lies_above_it(tmp_path):
    _repo_with_dot_env(tmp_path / "repo", track=True)
    assert config.git_index_state(str(tmp_path / "repo" / "new" / "deep" / ".env")) \
        == config.INDEX_UNKNOWN
    assert config.git_index_state(str(tmp_path / "none" / ".env")) == config.INDEX_FREE


@pytest.mark.parametrize("track", [True, False])
def test_a_dot_env_file_in_a_repository_git_refuses_is_not_read_and_its_settings_are_named(
        monkeypatch, tmp_path, track):
    """git answers nothing in a repository another user owns; whether the file is one
    the repository holds is not known there, so it is no setting."""
    _repo_with_dot_env(tmp_path / "repo", track=track)
    (tmp_path / "repo" / "sub").mkdir()
    (tmp_path / "repo" / "sub" / ".env").write_text("DEF_K=98\n")
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    for directory, name_list in ((tmp_path / "repo", ["CONTAINER_ENGINE", "DEF_K"]),
                                 (tmp_path / "repo" / "sub", ["DEF_K"])):
        monkeypatch.chdir(directory)
        for name in ("DEF_K", "CONTAINER_ENGINE"):
            monkeypatch.delenv(name, raising=False)
        importlib.reload(config)
        assert (config.DEF_K, config.CONTAINER_ENGINE, config.ENV_FILE_PATH,
                config.ENV_FILE_SKIP_PATH, config.ENV_FILE_SKIP_STATE) \
            == (12, "", "", str(directory / ".env"), config.INDEX_UNKNOWN)
        assert config.skip_env_setting_list() == name_list
        assert config.env_file_skip_text() \
            == "is in a git repository git did not answer for and was not read"


def test_a_dot_env_file_outside_any_repository_is_read_where_git_fails(
        monkeypatch, tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / ".env").write_text("DEF_K=99\n")
    monkeypatch.chdir(tmp_path / "plain")
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    assert config.git_index_state(str(tmp_path / "plain" / ".env")) == config.INDEX_FREE
    assert _reloaded(monkeypatch, DEF_K=None)["DEF_K"] == 99


@pytest.mark.parametrize("content, mode", [(b"# \x90\xdd\x92\xe8\nDEF_K=5\n", 0o600),
                                           (b"DEF_K=5\n", 0o000),
                                           (b"CTLOOM_TEST_NUL_VALUE=5\x000\n", 0o600),
                                           (b"CTLOOM_TEST\x00NAME=1\n", 0o600)])
def test_a_dot_env_file_that_cannot_be_read_stops_with_one_line_naming_it(
        monkeypatch, tmp_path, content, mode):
    path = tmp_path / ".env"
    path.write_bytes(content)
    path.chmod(mode)
    if mode == 0 and os.access(path, os.R_OK):
        pytest.skip("this user reads a file of mode 000")
    monkeypatch.setenv("TEST_GUIDE", "")
    with pytest.raises(SettingError, match="could not be read as the .env file") as e:
        config._load_env_file(str(path))
    assert str(path) in str(e.value)
    assert os.environ["TEST_GUIDE"] == ""


def test_the_template_names_the_settings_in_its_order_with_their_help():
    entry_list = config.env_template_entry_list()
    name_list = [entry.name for entry in entry_list]
    assert name_list[:2] == ["LLM_API_KEY", "MEDIUM_MODEL"]
    assert {"HIGH_MODEL", "LLM_API_BASE", "PRESET", "PROJECT_DIR"} <= set(name_list)
    by_name = {entry.name: (entry.value, entry.help_list) for entry in entry_list}
    assert by_name["LLM_API_KEY"][0] == "" and by_name["LLM_API_KEY"][1]
    assert by_name["PRESET"] == ("medium", by_name["PRESET"][1]) and by_name["PRESET"][1]
    # The heading at the top of the file belongs to no setting
    assert "codetwine-loom" not in " ".join(by_name["LLM_API_KEY"][1])


def test_the_template_names_the_proxy_s_settings_with_their_help():
    by_name = {entry.name: entry for entry in config.env_template_entry_list()}
    for name in ("HTTPS_PROXY", "NO_PROXY", config.CA_FILE_ENV):
        assert by_name[name].value == "" and by_name[name].help_list
        assert not by_name[name].question


def test_the_proxy_variables_that_hold_a_value_are_named_as_they_are_set(monkeypatch):
    for name in config.PROXY_NAME_TUPLE:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    assert config.proxy_name_list() == []
    monkeypatch.setenv("no_proxy", "localhost")
    monkeypatch.setenv("HTTPS_PROXY", "http://ctloom-proxy.corp.example:3128")
    monkeypatch.setenv("ALL_PROXY", "")
    assert config.proxy_name_list() == ["HTTPS_PROXY", "no_proxy"]


@pytest.mark.parametrize("value", [
    "http://ctloom-proxy.corp.example:3128", "http://svcname:p%23ss@ctloom-proxy.corp.example:3128",
    "https://ctloom-proxy.corp.example", "ctloom-proxy.corp.example:3128", "http://[::1]:3128",
    "[::1]:3128", "HTTP://Ctloom-Proxy.Corp.Example:3128/", "http://svc@name:pa@ss@h:3128",
    "http://svcname:pa%27%22%24%25%5C@ctloom-proxy.corp.example:3128", "http://ctloom-proxy:",
    "http://proxy.bücher.example:3128", "http://proxy.会社.jp:8080",
    "http://xn--bcher-kva.example:3128", "http://プロキシ.会社.jp:8080",
    "http://xn--zz.example:3128", "http://[::ffff:127.0.0.1]:3128",
    "http://svcname:secr%5Bet%5Dpw@ctloom-proxy.corp.example:3128",
    "https://svcname@ctloom-proxy.corp.example:3128"])
def test_a_proxy_s_url_every_client_reads_alike_is_taken(value):
    """httpx and yarl connect to the same host and port with the same user."""
    assert config.proxy_url_error(value) == ""
    url = value if "://" in value else f"http://{value}"
    client_url = httpx.Proxy(url).url
    yarl_url = yarl.URL(url)
    assert client_url.raw_host.decode() in (yarl_url.raw_host, "::ffff:127.0.0.1")
    assert (client_url.port or yarl_url.port) == yarl_url.port
    assert httpx.Proxy(url).auth in (None, (yarl_url.user, yarl_url.password or ""))


@pytest.mark.parametrize("value, why", [
    ("socks5://svcname:secretpw@h:1080", "points at a SOCKS proxy"),
    ("socks://svcname:secretpw@h:1080/", "points at a SOCKS proxy"),
    ("SOCKS4://h:1080", "points at a SOCKS proxy"),
    ("ftp://h:21", "is not an http:// or https:// URL"),
    ("http://svcname:Secr3t#secretpw@h:3128", "write them as %23, %2F and %3F"),
    ("http://svcname:Secr3t/secretpw@h:3128", "write them as %23, %2F and %3F"),
    ("http://svcname:Secr3t?secretpw@h:3128", "write them as %23, %2F and %3F"),
    ("http://svcname:8080#secretpw@h:3128", "write them as %23, %2F and %3F"),
    ("http://h:3128/secretpw", "holds a path, a query or a fragment"),
    ("http://h:3128#secretpw", "holds a path, a query or a fragment"),
    ("http://h:abc", "a port that is not a number from 1 to 65535"),
    ("http://h:99999", "a port that is not a number from 1 to 65535"),
    ("http://h:-1", "a port that is not a number from 1 to 65535"),
    ("http://h:31a8", "a port that is not a number from 1 to 65535"),
    ("http://h:0", "a port that is not a number from 1 to 65535"),
    ("http://[::1", "names a host not every HTTP client of this tool can connect to"),
    ("http://☃.example:3128", "names a host not every HTTP client of this tool can connect to"),
    ("http://ＡＢＣ.example:3128", "names a host not every HTTP client of this tool can connect to"),
    ("http://127.1:3128", "names a host not every HTTP client of this tool can connect to"),
    ("http://svcname:secretpw@2130706433:3128",
     "names a host not every HTTP client of this tool can connect to"),
    ("http://☃.example:99999", "a port that is not a number from 1 to 65535"),
    ("http://", "names no host"), ("http://:3128", "names no host"),
    ("http://h :3128", "holds a blank or a control character"),
    ("http://h\x00:3128", "holds a blank or a control character"),
    ("http://svcname:secr[et]pw@h:3128", "holds a user or a password not every HTTP client of this tool can read"),
    ("svcname:[secretpw@h:3128", "holds a user or a password not every HTTP client of this tool can read"),
    ("http://CORP\\svcname:secretpw@h:3128", "holds a user or a password not every HTTP client of this tool can read"),
    ("http://svcname:secretpw\\@h:3128", "holds a user or a password not every HTTP client of this tool can read"),
    ("http://svcname:日本語secretpw@h:3128", "holds a user or a password with a character outside ASCII"),
    ("http://svcname:%E6%97%A5secretpw@h:3128", "holds a user or a password with a character outside ASCII"),
    ("http://svcname:pässecretpw@h:3128", "holds a user or a password with a character outside ASCII"),
    ("http://svcname:p%C3%A4secretpw@h:3128", "holds a user or a password with a character outside ASCII"),
    ("http://svcname:p%E4secretpw@h:3128", "holds a user or a password with a character outside ASCII"),
    ("http://svcname@h:3128", "names a user without a password"),
    ("http://svcname:@h:3128", "names a user without a password"),
    ("http://:secretpw@h:3128", "or a password without a user"),
    ("http://svcname:secretpw@straße.example:3128", "names a host the HTTP clients"),
    ("http://svcname:secretpw@proxy.straße.example:3128", "names a host the HTTP clients"),
    ("http://svcname:secretpw@ς.example:3128", "names a host the HTTP clients"),
    ("http://pro%78y.example:3128", "names a host written with '%'"),
    ("http://[fe80::1%25eth0]:3128", "names a host written with '%'")])
def test_a_proxy_s_url_a_client_of_this_tool_cannot_read_is_said_without_its_value(value,
                                                                                    why):
    error = config.proxy_url_error(value)
    assert why in error
    assert "secretpw" not in error and "svcname" not in error and "Secr3t" not in error


@pytest.mark.parametrize("error", [
    ValueError("svcname secretpw"), UnicodeError("svcname secretpw"),
    KeyError("svcname secretpw"), RuntimeError("svcname secretpw")])
@pytest.mark.parametrize("reading_name", [
    "_httpx_proxy_reading", "_aiohttp_proxy_reading", "_urllib_proxy_reading_list"])
def test_an_error_of_any_client_s_reading_is_said_in_a_set_sentence(monkeypatch, error,
                                                                     reading_name):
    """Whatever a reading raises, the sentence holds nothing of the error's text."""
    def fail(url):
        raise error
    monkeypatch.setattr(config, reading_name, fail)
    assert config.proxy_url_error("http://svcname:secretpw@ctloom-proxy.corp.example:3128") == (
        "cannot be read by every HTTP client of this tool: write it as http://host:port")


@pytest.mark.parametrize("value, written", [
    ("ctloom-proxy.corp.example:3128", True), ("ctloom-proxy.corp.example:80", True),
    ("[::1]:3128", True), ("ctloom-proxy.corp.example", False), ("[::1]", False),
    ("svcname:p%23ss@ctloom-proxy.corp.example", False)])
def test_a_proxy_s_url_that_names_no_port_is_refused_where_a_port_is_needed(value,
                                                                            written):
    assert config.proxy_url_error(value) == ""
    assert config.proxy_url_error(value, port_needed=True) == (
        "" if written else "names no port: write it as http://host:port or host:port")


@pytest.mark.parametrize("proxy, want", [
    ("http://proxy.corp.example", "http://proxy.corp.example:80"),
    ("http://proxy.corp.example:", "http://proxy.corp.example:80"),
    ("http://[fd00::1]:", "http://[fd00::1]:80"),
    ("me:pw@proxy.corp.example/", "http://me:pw@proxy.corp.example:80/"),
    ("http://proxy.corp.example:3128", "http://proxy.corp.example:3128"),
    ("https://proxy.corp.example", "https://proxy.corp.example")])
def test_an_http_proxy_s_url_is_written_with_its_port(proxy, want):
    assert config.proxy_with_port(proxy) == want


def _clear_proxy_variable(monkeypatch):
    """Take every proxy variable, in any spelling, out of the environment."""
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)


def test_the_first_proxy_variable_no_client_reads_is_named(monkeypatch):
    _clear_proxy_variable(monkeypatch)
    monkeypatch.setenv("NO_PROXY", "socks5://not-read-as-a-proxy")
    monkeypatch.setenv("HTTPS_PROXY", "http://ctloom-proxy.corp.example:3128")
    assert config.proxy_error_text() == ""
    monkeypatch.setenv("all_proxy", "socks5://ctloom-proxy.corp.example:1080")
    assert config.proxy_error_text().startswith("all_proxy points at a SOCKS proxy")


def test_a_proxy_variable_the_clients_read_in_place_of_another_is_the_one_looked_at(
        monkeypatch):
    """A name in lower case that holds a value is read, the same name in upper case
    is not; a name in mixed case is read when no other spelling holds a value."""
    _clear_proxy_variable(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "socks5://ctloom-proxy.corp.example:1080")
    monkeypatch.setenv("https_proxy", "http://ctloom-proxy.corp.example:3128")
    assert config.proxy_error_text() == ""
    monkeypatch.delenv("https_proxy")
    assert config.proxy_error_text().startswith("HTTPS_PROXY points at a SOCKS proxy")
    monkeypatch.delenv("HTTPS_PROXY")
    monkeypatch.setenv("Http_Proxy", "socks://svcname:secretpw@ctloom-proxy.corp.example:1080")
    text = config.proxy_error_text()
    assert text.startswith("Http_Proxy points at a SOCKS proxy") and "secretpw" not in text


def _client_goes_through_proxy(endpoint):
    """Ask a real httpx.Client and litellm's real aiohttp transport, under the
    environment as it is, whether a request to endpoint goes through a proxy."""
    from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport

    url = httpx.URL(endpoint)
    with httpx.Client() as client:
        httpx_answer = client._transport_for_url(url) is not client._transport
    transport = LiteLLMAiohttpTransport(client=lambda: None)
    return httpx_answer, transport._proxy_from_env(url) is not None


@pytest.mark.parametrize("no_proxy, api_base", [
    ("localhost,llm.corp.example", "http://llm.corp.example:8080/v1"),
    ("localhost,.corp.example", "http://llm.corp.example:8080/v1"),
    ("corp.example", "http://llm.corp.example:8080/v1"),
    ("LLM.Corp.Example", "https://llm.corp.example/v1"),
    ("*", "https://llm.corp.example/v1"), ("10.1.2.3", "http://10.1.2.3:8080/v1"),
    ("::1", "http://[::1]:8080/v1"), ("llm.corp.example:443", "https://llm.corp.example/v1"),
    ("https://llm.corp.example", "http://llm.corp.example:8080/v1"),
    ("localhost", ""), ("*", ""), ("10.0.0.0/8,.corp.example", "")])
def test_a_no_proxy_both_clients_of_the_model_calls_read_alike_is_taken(
        monkeypatch, no_proxy, api_base):
    _clear_proxy_variable(monkeypatch)
    _set_model(monkeypatch, "ctloom-unknown-model")
    monkeypatch.setenv("HTTPS_PROXY", "http://ctloom-proxy.corp.example:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://ctloom-proxy.corp.example:3128")
    assert config.no_proxy_error(no_proxy, api_base) == ""
    monkeypatch.setenv("NO_PROXY", no_proxy)
    httpx_answer, aiohttp_answer = _client_goes_through_proxy(
        api_base or "https://ctloom-endpoint.invalid/")
    assert httpx_answer == aiohttp_answer


@pytest.mark.parametrize("no_proxy, api_base, kind", [
    ("127.0.0.2:8080", "http://127.0.0.2:8080/v1", "an entry with a port (host:port)"),
    ("llm.corp.example:8080", "http://llm.corp.example:8080/v1",
     "an entry with a port (host:port)"),
    (".corp.example", "http://corp.example:8080/v1",
     "an entry with a leading '.' that names the host itself"),
    ("localhost,.corp.example", "http://corp.example/v1",
     "an entry with a leading '.' that names the host itself"),
    ("localhost, *", "http://llm.corp.example:8080/v1", "'*' among other entries"),
    (" *", "https://llm.corp.example/v1", "'*' among other entries"),
    ("localhost,*", "", "'*' among other entries"),
    ("http://llm.corp.example", "http://llm.corp.example:8080/v1",
     "an entry with a scheme (scheme://host)"),
    ("localhost,http://llm.corp.example", "http://llm.corp.example/v1",
     "an entry with a scheme (scheme://host)")])
def test_a_no_proxy_the_clients_of_the_model_calls_read_apart_is_said_by_its_entry(
        monkeypatch, no_proxy, api_base, kind):
    """With the kind of entry and the endpoint's name, nothing of the value; the real
    clients do send the calls apart."""
    _clear_proxy_variable(monkeypatch)
    _set_model(monkeypatch, "ctloom-unknown-model")
    monkeypatch.setenv("HTTPS_PROXY", "http://ctloom-proxy.corp.example:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://ctloom-proxy.corp.example:3128")
    error = config.no_proxy_error(no_proxy, api_base)
    name = "LLM_API_BASE" if api_base else "the providers' own endpoints"
    assert error.startswith(f"is not read alike by every HTTP client of this tool for "
                            f"{name}, ")
    assert kind in error and "corp" not in error and "127.0" not in error
    monkeypatch.setenv("NO_PROXY", no_proxy)
    httpx_answer, aiohttp_answer = _client_goes_through_proxy(
        api_base or "https://ctloom-endpoint.invalid/")
    assert httpx_answer != aiohttp_answer


@pytest.mark.parametrize("no_proxy", ["[::1]", "localhost,bücher.example"])
@pytest.mark.parametrize("proxy", ["http://ctloom-proxy.corp.example:3128", ""])
def test_a_no_proxy_httpx_cannot_read_is_refused(monkeypatch, no_proxy, proxy):
    """httpx.Client cannot be made under it, with a proxy or without, so every call made
    whole would fail; the check before a call says so."""
    _clear_proxy_variable(monkeypatch)
    if proxy:
        monkeypatch.setenv("HTTPS_PROXY", proxy)
    assert config.no_proxy_error(no_proxy, "http://llm.corp.example/v1").startswith(
        "holds an entry not every HTTP client of this tool can read")
    monkeypatch.setenv("NO_PROXY", no_proxy)
    assert config.proxy_error_text().startswith(
        "NO_PROXY holds an entry not every HTTP client of this tool can read")
    with pytest.raises(httpx.InvalidURL):
        httpx.Client()


def test_a_no_proxy_a_client_s_reading_fails_over_is_refused_in_a_set_sentence(
        monkeypatch):
    _clear_proxy_variable(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://ctloom-proxy.corp.example:3128")

    def fail(proxy_dict, url):
        raise AttributeError("llm.corp.example")
    monkeypatch.setattr(config, "_aiohttp_goes_through_proxy", fail)
    assert config.no_proxy_error("llm.corp.example", "https://llm.corp.example/v1") == (
        "cannot be read by every HTTP client of this tool: write each host alone, in "
        "ASCII, separated by commas")


def _set_model(monkeypatch, model):
    """Make model every model of the settings, with no endpoint variable set."""
    for name in config.setting_value_dict("_MODEL"):
        monkeypatch.setattr(config, name, model)
    for name in list(os.environ):
        if config._ENDPOINT_NAME_RE.fullmatch(name):
            monkeypatch.delenv(name)


@pytest.mark.parametrize("model, variable, endpoint, name", [
    ("hosted_vllm/ctloom-m", "HOSTED_VLLM_API_BASE", "http://vllm.corp.example:8000",
     "HOSTED_VLLM_API_BASE"),
    ("openai/ctloom-m", "OPENAI_BASE_URL", "http://vllm.corp.example:8000/v1", "OPENAI_BASE_URL"),
    ("openai/ctloom-m", "OPENAI_API_BASE", "http://vllm.corp.example:8000/v1", "OPENAI_API_BASE"),
    ("anthropic/ctloom-m", "ANTHROPIC_API_BASE", "http://vllm.corp.example:8000",
     "ANTHROPIC_API_BASE"),
    ("anthropic/ctloom-m", "AZURE_API_BASE", "http://vllm.corp.example:8000", "AZURE_API_BASE"),
    ("llamafile/ctloom-m", "", "", ""),
    ("llamafile/ctloom-m", "LLAMAFILE_API_BASE", "http://vllm.corp.example:8000/v1",
     "LLAMAFILE_API_BASE"),
    ("huggingface/ctloom-org/ctloom-m", "HF_API_BASE", "http://vllm.corp.example:8000/v1",
     "HF_API_BASE"),
    ("watsonx/ctloom-m", "WATSONX_URL", "http://vllm.corp.example:8000", "WATSONX_URL")])
def test_a_no_proxy_is_checked_against_the_endpoint_litellm_calls(
        monkeypatch, model, variable, endpoint, name):
    """With LLM_API_BASE empty, the endpoint is the one litellm itself reads for the
    model, from the provider's variable or as the provider's default, and every
    endpoint variable litellm reads counts whatever the model."""
    _clear_proxy_variable(monkeypatch)
    _set_model(monkeypatch, model)
    if variable:
        monkeypatch.setenv(variable, endpoint)
    else:
        endpoint = config.model_endpoint_dict("")[f"the default endpoint of {model}"]
        name = f"the default endpoint of {model}"
    assert config.model_endpoint_dict("")[name] == endpoint
    assert config.model_endpoint_dict("http://llm/v1") == {"LLM_API_BASE": "http://llm/v1"}
    monkeypatch.setenv("HTTP_PROXY", "http://ctloom-proxy:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://ctloom-proxy:3128")
    host_port = httpx.URL(endpoint).netloc.decode()
    error = config.no_proxy_error(f"localhost,{host_port}", "")
    assert error.startswith(
        f"is not read alike by every HTTP client of this tool for {name}, ")
    assert "corp" not in error and "127.0" not in error
    assert config.no_proxy_error(f"localhost,{httpx.URL(endpoint).host}", "") == ""
    monkeypatch.setenv("NO_PROXY", f"localhost,{host_port}")
    httpx_answer, aiohttp_answer = _client_goes_through_proxy(endpoint)
    assert httpx_answer != aiohttp_answer


def test_an_endpoint_litellm_does_not_tell_is_stood_for(monkeypatch):
    """A model litellm finds no endpoint for is checked against a host no entry names."""
    _clear_proxy_variable(monkeypatch)
    _set_model(monkeypatch, "ctloom-unknown-model")
    assert config.model_endpoint_dict("") == {
        "the providers' own endpoints": "https://ctloom-endpoint.invalid/"}
    monkeypatch.setenv("HTTPS_PROXY", "http://ctloom-proxy:3128")
    assert config.no_proxy_error("localhost,*", "").startswith(
        "is not read alike by every HTTP client of this tool for the providers' own "
        "endpoints, ")
    assert config.no_proxy_error("localhost", "") == ""


@pytest.mark.parametrize("model", ["github_copilot/gpt-x", "chatgpt/gpt-x"])
def test_the_endpoint_of_a_model_litellm_signs_in_for_is_not_asked_and_it_is_said(
        monkeypatch, caplog, tmp_path, model):
    """No sign-in starts, nothing is made under HOME, and a warning names the model
    once, however many times NO_PROXY is checked."""
    import litellm
    start_list = []
    for config_class in (litellm.GithubCopilotConfig, litellm.ChatGPTConfig):
        monkeypatch.setattr(config_class, "_get_openai_compatible_provider_info",
                            lambda self, name, *a, **k: start_list.append(name))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "_UNASKED_ENDPOINT_MODEL_SET", set(), raising=False)
    _clear_proxy_variable(monkeypatch)
    _set_model(monkeypatch, model)
    monkeypatch.setenv("HTTPS_PROXY", "http://ctloom-proxy:3128")
    with caplog.at_level(logging.WARNING, logger="codetwine_loom.config"):
        assert config.model_endpoint_dict("") == {
            "the providers' own endpoints": "https://ctloom-endpoint.invalid/"}
        assert config.no_proxy_error("localhost", "") == ""
        assert config.proxy_error_text() == ""
    assert start_list == [] and list(tmp_path.iterdir()) == []
    assert caplog.text.count(
        f"the endpoint of {model} is not known without signing in to its provider, so "
        f"NO_PROXY is not checked against it") == 1


def test_the_endpoint_variables_are_counted_by_the_shape_of_their_name(monkeypatch):
    """A variable whose name ends in _API_BASE, _URL or _ENDPOINT and whose value is an
    http or https URL is an endpoint, whatever model reads it."""
    _set_model(monkeypatch, "ctloom-unknown-model")
    for name in ("HF_API_BASE", "OPENAI_BASE_URL", "WATSONX_URL", "AI_API_URL",
                 "AWS_BEDROCK_RUNTIME_ENDPOINT", "CTLOOM_NEW_PROVIDER_API_BASE"):
        monkeypatch.setenv(name, f"HTTPS://{name.lower()}.corp.example/v1")
    monkeypatch.setenv("DATABASE_URL", "postgres://db.corp.example:5432/x")
    monkeypatch.setenv("OPENAI_API_KEY", "http://key.corp.example")
    monkeypatch.setenv("openai_base_url", "http://lower.corp.example")
    monkeypatch.setenv("CTLOOM_EMPTY_API_BASE", "")
    monkeypatch.setenv("CTLOOM_PLAIN_URL", "llm.corp.example:8080")
    assert config.model_endpoint_dict("") == {
        "AI_API_URL": "HTTPS://ai_api_url.corp.example/v1",
        "AWS_BEDROCK_RUNTIME_ENDPOINT": "HTTPS://aws_bedrock_runtime_endpoint.corp.example/v1",
        "CTLOOM_NEW_PROVIDER_API_BASE": "HTTPS://ctloom_new_provider_api_base.corp.example/v1",
        "HF_API_BASE": "HTTPS://hf_api_base.corp.example/v1",
        "OPENAI_BASE_URL": "HTTPS://openai_base_url.corp.example/v1",
        "WATSONX_URL": "HTTPS://watsonx_url.corp.example/v1",
        "the providers' own endpoints": "https://ctloom-endpoint.invalid/"}


def test_a_no_proxy_is_checked_against_the_proxy_it_is_given(monkeypatch):
    """No proxy, only what httpx cannot read is refused; a proxy answered counts."""
    _clear_proxy_variable(monkeypatch)
    _set_model(monkeypatch, "anthropic/ctloom-m")
    assert config.no_proxy_error("localhost, *", "") == ""
    assert config.no_proxy_error("localhost, *", "", "http://ctloom-proxy:3128") != ""


def test_the_no_proxy_variable_the_clients_read_apart_is_named(monkeypatch):
    _clear_proxy_variable(monkeypatch)
    monkeypatch.setattr(config, "LLM_API_BASE", "http://llm.corp.example:8080/v1")
    monkeypatch.setenv("HTTP_PROXY", "http://ctloom-proxy.corp.example:3128")
    monkeypatch.setenv("no_proxy", "localhost,llm.corp.example")
    assert config.proxy_error_text() == ""
    monkeypatch.setenv("no_proxy", "localhost,llm.corp.example:8080")
    text = config.proxy_error_text()
    assert text.startswith("no_proxy is not read alike by every HTTP client of this "
                           "tool for LLM_API_BASE") and "corp" not in text


def test_the_dot_env_of_the_directory_the_tests_start_in_sets_no_route_variable(
        monkeypatch, tmp_path):
    """The other settings it holds are read; a .env file elsewhere sets them all."""
    import conftest
    for name in ("HTTPS_PROXY", "https_proxy", "NO_PROXY", "SSL_CERT_FILE", "PRESET"):
        monkeypatch.delenv(name, raising=False)
    text = ("HTTPS_PROXY=http://ctloom-proxy.corp.example:3128\nNO_PROXY=localhost\n"
            "SSL_CERT_FILE=/nonexistent/ca.pem\nPRESET=high\n")
    (tmp_path / ".env").write_text(text, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(conftest, "_START_ENV_FILE", str(tmp_path / ".env"))
    importlib.reload(config)
    assert config.ENV_FILE_PATH == str(tmp_path / ".env") and config.PRESET == "high"
    assert not {"HTTPS_PROXY", "https_proxy", "NO_PROXY", "SSL_CERT_FILE"} & set(os.environ)
    other = tmp_path / "other"
    other.mkdir()
    (other / ".env").write_text(text.replace("/nonexistent/ca.pem", ""), encoding="utf-8")
    monkeypatch.chdir(other)
    importlib.reload(config)
    assert os.environ["HTTPS_PROXY"] == "http://ctloom-proxy.corp.example:3128"


def test_the_dot_env_of_the_directory_the_tests_start_in_sets_no_git_variable(
        monkeypatch, tmp_path):
    """A git variable it holds stays unset whenever config reads it again; the other
    settings it holds are read."""
    import conftest
    from codetwine_loom import gitcmd
    name_list = ["GIT_DIR", "GIT_CONFIG", "GIT_INDEX_FILE", "GIT_LITERAL_PATHSPECS",
                 "GIT_EDITOR", "GIT_EXEC_PATH", "HTTPS_PROXY"]
    assert {*gitcmd.COMMAND_ENV_TUPLE, *gitcmd.DIR_ENV_TUPLE, *config.GIT_PLACE_NAME_TUPLE,
            *config.GIT_ANSWER_NAME_TUPLE, *name_list} <= set(conftest._KEEP_OUT_NAME_TUPLE)
    for name in (*name_list, "PRESET"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "".join(f"{name}={tmp_path / 'victim' / name}\n" for name in name_list)
        + "PRESET=high\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(conftest, "_START_ENV_FILE", str(tmp_path / ".env"))
    importlib.reload(config)
    importlib.reload(config)
    assert config.ENV_FILE_PATH == str(tmp_path / ".env") and config.PRESET == "high"
    assert not set(name_list) & set(os.environ)


def test_the_shell_s_environment_leaves_out_what_the_dot_env_file_filled(monkeypatch,
                                                                          tmp_path):
    """A name the file filled is put back as the shell held it, unset or empty, while
    it holds the file's value, also after config is read again; a name the shell held
    a value for, and one set again since, is as the environment holds it."""
    monkeypatch.setenv("CTLOOM_SHELL_KEPT", "shell")
    monkeypatch.setenv("CTLOOM_SHELL_EMPTY", "")
    for name in ("CTLOOM_SHELL_UNSET", "CTLOOM_SHELL_LATER"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / "dot.env"
    env_file.write_text("CTLOOM_SHELL_KEPT=file\nCTLOOM_SHELL_EMPTY=file\nCTLOOM_SHELL_UNSET=file\n"
                        "CTLOOM_SHELL_LATER=file\n", encoding="utf-8")
    monkeypatch.setattr(config, "_ENV_FILE_FILL_DICT", {})
    config._load_env_file(str(env_file))
    os.environ["CTLOOM_SHELL_LATER"] = "set again"
    monkeypatch.chdir(tmp_path)
    importlib.reload(config)
    shell_dict = config.shell_env()
    assert [os.environ[name] for name in ("CTLOOM_SHELL_KEPT", "CTLOOM_SHELL_EMPTY",
                                          "CTLOOM_SHELL_UNSET")] == ["shell", "file", "file"]
    assert "CTLOOM_SHELL_UNSET" not in shell_dict and shell_dict["CTLOOM_SHELL_EMPTY"] == ""
    assert shell_dict["CTLOOM_SHELL_KEPT"] == "shell"
    assert shell_dict["CTLOOM_SHELL_LATER"] == "set again"
    child_dict = config.child_env({"CTLOOM_EXTRA": "x"}, shell_dict)
    assert "CTLOOM_SHELL_UNSET" not in child_dict and child_dict["CTLOOM_EXTRA"] == "x"
    assert config.child_env()["CTLOOM_SHELL_UNSET"] == "file"


def test_the_shell_s_environment_leaves_out_what_the_command_line_gave(monkeypatch,
                                                                       tmp_path):
    """A name the command line wrote is put back as it was before, also over a value
    the .env file filled and after config is read again; one set again since is as the
    environment holds it."""
    monkeypatch.setenv("CTLOOM_OPT_SHELL", "shell")
    for name in ("CTLOOM_OPT_UNSET", "CTLOOM_OPT_FILE", "CTLOOM_OPT_LATER"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / "dot.env"
    env_file.write_text("CTLOOM_OPT_FILE=file\n", encoding="utf-8")
    config._load_env_file(str(env_file))
    for name, value in (("CTLOOM_OPT_SHELL", "option"), ("CTLOOM_OPT_UNSET", "option"),
                        ("CTLOOM_OPT_FILE", "option"), ("CTLOOM_OPT_LATER", "option")):
        config.keep_option_value(name, os.environ.get(name), value)
        monkeypatch.setenv(name, value)
    os.environ["CTLOOM_OPT_LATER"] = "set again"
    monkeypatch.chdir(tmp_path)
    importlib.reload(config)
    shell_dict = config.shell_env()
    assert shell_dict["CTLOOM_OPT_SHELL"] == "shell"
    assert "CTLOOM_OPT_UNSET" not in shell_dict and "CTLOOM_OPT_FILE" not in shell_dict
    assert shell_dict["CTLOOM_OPT_LATER"] == "set again"
    assert os.environ["CTLOOM_OPT_FILE"] == "option"


def test_the_shell_s_environment_leaves_out_what_config_wrote_for_itself(monkeypatch,
                                                                          tmp_path):
    """The certificate file written out in full, the variable made from it, an empty
    proxy variable taken out and a name a library sets when it is imported are put
    back as the shell held them, over a value the .env file filled as well, also
    after config is read again; the environment keeps what config wrote."""
    (tmp_path / "ca.pem").write_text(CERTIFICATE_PEM, encoding="utf-8")
    (tmp_path / "litellm.py").write_text(
        "import os\nos.environ.setdefault('CTLOOM_LIBRARY_DIR', 'library')\n",
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CTLOOM_LIBRARY_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("https_proxy", "")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    (tmp_path / "dot.env").write_text("SSL_CERT_FILE=ca.pem\n", encoding="utf-8")
    config._load_env_file(str(tmp_path / "dot.env"))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "litellm")
    importlib.reload(config)
    importlib.reload(config)
    ca_file = str(tmp_path / "ca.pem")
    assert os.environ["SSL_CERT_FILE"] == os.environ["REQUESTS_CA_BUNDLE"] == ca_file
    assert "https_proxy" not in os.environ and os.environ["CTLOOM_LIBRARY_DIR"] == "library"
    shell_dict = config.shell_env()
    assert not {"SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CTLOOM_LIBRARY_DIR"} & set(shell_dict)
    assert shell_dict["https_proxy"] == "" and shell_dict["HTTPS_PROXY"]
    assert config.own_name_set() == {"SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "https_proxy",
                                     "CTLOOM_LIBRARY_DIR"}


def test_a_question_line_goes_to_the_setting_below_it(monkeypatch, tmp_path):
    template = tmp_path / ".env.example"
    template.write_text("# about A\n#? y=false Limit it?\n# A=true\n\n# about B\n# B=1\n",
                        encoding="utf-8")
    monkeypatch.setattr(config, "ENV_TEMPLATE_PATH", str(template))
    first, second = config.env_template_entry_list()
    assert (first.name, first.value, first.help_list) == ("A", "true", ["about A"])
    assert (first.question, first.yes_value) == ("Limit it?", "false")
    assert (second.name, second.question, second.yes_value) == ("B", "", "")


def test_a_skip_line_keeps_the_setting_below_it_from_being_asked(monkeypatch, tmp_path):
    template = tmp_path / ".env.example"
    template.write_text("# about A\n#? skip\n# A=low\n\n# about B\n# B=1\n",
                        encoding="utf-8")
    monkeypatch.setattr(config, "ENV_TEMPLATE_PATH", str(template))
    first, second = config.env_template_entry_list()
    assert (first.name, first.help_list, first.skip) == ("A", ["about A"], True)
    assert (second.name, second.skip) == ("B", False)


def test_the_design_document_settings_are_in_the_template_and_not_asked():
    by_name = {entry.name: entry for entry in config.env_template_entry_list()}
    assert by_name["DOC_CREATE_MODEL"].skip and by_name["DOC_CREATE_EFFORT"].skip
    assert by_name["DOC_CREATE_EFFORT"].value == "low"


def test_the_template_asks_for_the_refill_with_a_question(monkeypatch):
    by_name = {entry.name: entry for entry in config.env_template_entry_list()}
    entry = by_name["WARM_MAX_REFILL"]
    assert entry.yes_value == "0" and entry.question
    assert int(entry.value) == _reloaded(monkeypatch, WARM_MAX_REFILL=None)["WARM_MAX_REFILL"]


@pytest.mark.parametrize("name", ["OVERVIEW", "TEST_CREATE", "REFACTOR", "TEST_REFACTOR"])
def test_the_overview_and_the_stages_after_the_merge_are_on_and_not_asked_for(monkeypatch,
                                                                              name):
    assert name not in {entry.name for entry in config.env_template_entry_list()}
    assert _reloaded(monkeypatch, **{name: None})[name] is True


def test_the_stage_that_writes_tests_follows_the_shared_settings_when_unset(monkeypatch):
    value_dict = _reloaded(monkeypatch, TEST_CREATE_MODEL=None, TEST_CREATE_MAX_TURN=None,
                           TEST_CREATE_MAX_REFUSAL=None, MEDIUM_MODEL="m/x",
                           RLM_MAX_TURN="7", INTEGRATE_MAX_REFUSAL="4")
    assert value_dict["TEST_CREATE_MODEL"] == "m/x"
    # The turns are set apart from RLM_MAX_TURN
    assert value_dict["TEST_CREATE_MAX_TURN"] == 50
    assert value_dict["TEST_CREATE_MAX_REFUSAL"] == 4
    assert _reloaded(monkeypatch, TEST_CREATE_MAX_DEFINITION=None)["TEST_CREATE_MAX_DEFINITION"] == 10


def test_the_stages_that_refactor_follow_the_shared_settings_when_unset(monkeypatch):
    value_dict = _reloaded(monkeypatch, REFACTOR_MODEL=None, REFACTOR_MAX_TOKEN=None,
                           REFACTOR_THINK_LEVEL=None, REFACTOR_MAX_TURN=None,
                           REFACTOR_MAX_REFUSAL=None, MEDIUM_MODEL="m/x",
                           AGENT_MAX_TOKEN="900", THINK_LEVEL="low", RLM_MAX_TURN="7",
                           INTEGRATE_MAX_REFUSAL="4")
    assert value_dict["REFACTOR_MODEL"] == "m/x"
    assert value_dict["REFACTOR_MAX_TOKEN"] == 900
    assert value_dict["REFACTOR_THINK_LEVEL"] == "low"
    # The turns are set apart from RLM_MAX_TURN
    assert value_dict["REFACTOR_MAX_TURN"] == 50
    assert value_dict["REFACTOR_MAX_REFUSAL"] == 4


@pytest.mark.parametrize("preset", ["low", "medium", "high"])
def test_the_turns_are_the_same_in_every_preset(monkeypatch, preset):
    value_dict = _reloaded(monkeypatch, PRESET=preset, RLM_MAX_TURN=None,
                           EXPLORE_MAX_TURN=None, INTEGRATE_MAX_TURN=None,
                           TEST_CREATE_MAX_TURN=None, REFACTOR_MAX_TURN=None)
    assert value_dict["RLM_MAX_TURN"] == 40 and value_dict["EXPLORE_MAX_TURN"] == 20
    assert value_dict["INTEGRATE_MAX_TURN"] == value_dict["TEST_CREATE_MAX_TURN"] \
        == value_dict["REFACTOR_MAX_TURN"] == 50



def test_the_template_names_every_search_and_page_reading_api():
    by_name = {entry.name: entry for entry in config.env_template_entry_list()}
    assert by_name["WEB_SEARCH_PROVIDER"].value == "" == by_name["WEB_SEARCH_API_KEY"].value
    assert by_name["WEB_FETCH_PROVIDER"].value == "" == by_name["WEB_FETCH_API_KEY"].value
    help_text = " ".join(by_name["WEB_SEARCH_PROVIDER"].help_list)
    for choice in (config.WEB_SEARCH_BY_THE_MODEL, *config.WEB_SEARCH_PRESET_DICT):
        assert choice in help_text
    help_text = " ".join(by_name["WEB_FETCH_PROVIDER"].help_list)
    for choice in config.WEB_FETCH_PRESET_DICT:
        assert choice in help_text


def test_the_endpoint_is_not_read_from_the_environment(monkeypatch):
    """It comes from the named preset alone."""
    read = _reloaded(monkeypatch, WEB_SEARCH_PROVIDER=None, WEB_SEARCH_API_KEY=None,
                     WEB_SEARCH_URL="https://elsewhere.invalid/",
                     WEB_FETCH_URL="https://elsewhere.invalid/")
    assert read["WEB_SEARCH_URL"] == "" and read["WEB_FETCH_URL"] == ""
    for name in ("WEB_SEARCH_URL", "WEB_SEARCH_HEADER", "WEB_FETCH_URL", "WEB_FETCH_HEADER"):
        assert name not in config.env_name_tuple()

def test_the_template_is_written_with_the_answers_set(tmp_path):
    path = str(tmp_path / ".env")
    config.write_env_file(path, {"LLM_API_KEY": "sk-1", "PRESET": "high"})
    text = pathlib.Path(path).read_text(encoding="utf-8")
    template = pathlib.Path(config.ENV_TEMPLATE_PATH).read_text(encoding="utf-8")
    assert "\nLLM_API_KEY=sk-1\n" in text and "\nPRESET=high\n" in text
    assert "# PRESET=medium" not in text
    # Everything else is the template's own, commented lines included
    assert "# MEDIUM_MODEL=anthropic/claude-sonnet-5-5" in text
    assert "# HIGH_MODEL=anthropic/claude-opus-5-5" in text
    assert len(text.splitlines()) == len(template.splitlines())
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    with pytest.raises(FileExistsError):
        config.write_env_file(path, {})
    assert os.listdir(tmp_path) == [".env"]


def test_the_file_is_readable_by_its_owner_alone_before_the_key_is_written(
        tmp_path, monkeypatch):
    path = str(tmp_path / ".env")
    mode_list = []

    def opened(file, mode="r", *arg_tuple, **kwarg_dict):
        if "w" in mode:
            mode_list.append(stat.S_IMODE(os.stat(file).st_mode))
        return open(file, mode, *arg_tuple, **kwarg_dict)
    monkeypatch.setattr(config, "open", opened, raising=False)
    # The mode is the one the file is made with, not one set after it was made
    monkeypatch.setattr(os, "chmod", lambda *arg_tuple, **kwarg_dict: None)
    config.write_env_file(path, {"LLM_API_KEY": "sk-1"})
    assert mode_list == [0o600]


def test_a_write_that_fails_leaves_nothing_behind(tmp_path, monkeypatch):
    """Neither the file nor its temporary name, whether the text cannot be encoded or
    the write itself fails."""
    path = str(tmp_path / ".env")
    with pytest.raises(UnicodeEncodeError):
        config.write_env_file(path, {"PROJECT_DIR": "/tmp/caf\udce9"})
    assert os.listdir(tmp_path) == []

    def full(file, mode="r", *arg_tuple, **kwarg_dict):
        if "w" in mode:
            raise OSError(27, "File too large")
        return open(file, mode, *arg_tuple, **kwarg_dict)
    monkeypatch.setattr(config, "open", full, raising=False)
    with pytest.raises(OSError, match="File too large"):
        config.write_env_file(path, {"LLM_API_KEY": "sk-1"})
    monkeypatch.undo()
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("value", [
    "sk-1", "/home/u/my #1 repo", "http://h/${X}/v1", "${X}", "$X and ${:-$}{X}",
    "'quoted' and more", '"quoted"', "it's", "a\\b", "back\\", "a  b ", " # x",
    "名前 を変えて", "a=b", ""])
def test_an_answer_is_read_back_from_the_file_as_it_was_given(tmp_path, monkeypatch,
                                                            value):
    from dotenv import dotenv_values
    path = str(tmp_path / ".env")
    config.write_env_file(path, {"PROJECT_DIR": value, "NAME_NOT_IN_THE_TEMPLATE": value})
    for env_value in (None, "set"):
        if env_value is None:
            monkeypatch.delenv("X", raising=False)
        else:
            monkeypatch.setenv("X", env_value)
        read_dict = dotenv_values(path)
        assert (read_dict["PROJECT_DIR"], read_dict["NAME_NOT_IN_THE_TEMPLATE"]) \
            == (value, value)


def test_a_dot_env_is_written_over_only_where_a_value_changes(tmp_path):
    path = str(tmp_path / ".env")
    config.write_env_file(path, {"LLM_API_KEY": "sk-1", "PRESET": "high",
                                 "HTTPS_PROXY": "http://proxy:3128"})
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("# my note\nexport OWN_NAME='a b'\nAGENT_LOG_LEVEL=INFO\n"
                     "LLM_API_KEY=sk-1")
    before = pathlib.Path(path).read_text(encoding="utf-8")
    current_dict = config.env_file_value_dict(path)
    assert current_dict["OWN_NAME"] == "a b" and "MEDIUM_MODEL" not in current_dict
    assert not config.rewrite_env_file(path, current_dict, dict(current_dict))
    assert pathlib.Path(path).read_text(encoding="utf-8") == before
    answer_dict = {**current_dict, "LLM_API_KEY": "sk-2", "NO_PROXY": "localhost",
                   "NEW_NAME": "x"}
    for name in ("PRESET", "HTTPS_PROXY", "AGENT_LOG_LEVEL"):
        del answer_dict[name]
    assert config.rewrite_env_file(path, current_dict, answer_dict)
    text = pathlib.Path(path).read_text(encoding="utf-8")
    # The key's first line takes the new value and its later line goes
    assert text.count("LLM_API_KEY=") == 1 and "\nLLM_API_KEY=sk-2\n" in text
    # A setting the template comments out takes that line, and one it does not name
    # goes after the last line
    assert "\nNO_PROXY=localhost\n" in text and "# NO_PROXY=" not in text
    assert text.endswith("\nNEW_NAME=x\n")
    # A setting taken out gets the template's line back, or loses its line
    assert "\n# PRESET=medium\n" in text and "\n# HTTPS_PROXY=\n" in text
    assert "AGENT_LOG_LEVEL" not in text
    # Every other line is as it was: two lines are taken out and one is added
    assert "\n# my note\nexport OWN_NAME='a b'\n" in text
    assert len(text.splitlines()) == len(before.splitlines()) - 1
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert config.env_file_value_dict(path) == answer_dict
    assert os.listdir(tmp_path) == [".env"]


def test_the_dot_env_read_is_put_back_out_of_the_environment(tmp_path, monkeypatch):
    """The shell's values stay, and the next import reads no .env."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LLM_API_KEY=sk-file\nPRESET=high\n", encoding="utf-8")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("PRESET", "")
    monkeypatch.delenv(config.NO_ENV_FILE_ENV, raising=False)
    importlib.reload(config)
    assert config.LLM_API_KEY == "sk-file" and config.PRESET == "high"
    config.unload_env_file()
    importlib.reload(config)
    assert "LLM_API_KEY" not in os.environ and os.environ["PRESET"] == ""
    assert config.ENV_FILE_PATH == "" and config.LLM_API_KEY == ""
    assert config.ENV_FILE_SKIP_STATE == config.ENV_FILE_HOLD
    assert config.NO_ENV_FILE_ENV not in os.environ and config.skip_env_setting_list() == []


def test_a_setting_taken_out_leaves_one_line_naming_it_commented_out(tmp_path):
    path = str(tmp_path / ".env")
    config.write_env_file(path, {})
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("PRESET=high\n")
    assert config.rewrite_env_file(path, {"PRESET": "high"}, {})
    text = pathlib.Path(path).read_text(encoding="utf-8")
    assert text.count("PRESET=") == 1 and "\n# PRESET=medium\n" in text
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("# PRESET=low\n")
    assert config.rewrite_env_file(path, {}, {"PRESET": "high"})
    text = pathlib.Path(path).read_text(encoding="utf-8")
    assert "\nPRESET=high\n" in text and text.endswith("\n# PRESET=low\n")


@pytest.mark.parametrize("name, comment, value", [
    ("MEDIUM_MODEL", "# MEDIUM_MODEL=openai/gpt-spare", "anthropic/claude-haiku-5-5"),
    ("OPENAI_API_KEY", "#OPENAI_API_KEY=sk-old", "sk-new"),
    ("PRESET", "# PRESET=low", "high")])
def test_a_value_kept_commented_out_by_hand_stays_above_the_one_written(tmp_path, name,
                                                                        comment, value):
    """A line commented out that is the template's, or names no value, is the place."""
    path = tmp_path / ".env"
    path.write_text(f"LLM_API_KEY=sk-1\n# spare:\n{comment}\n# NO_PROXY=\n"
                    f"# PRESET=medium\n", encoding="utf-8")
    assert config.rewrite_env_file(str(path), {"LLM_API_KEY": "sk-1"},
                                   {"LLM_API_KEY": "sk-1", name: value, "NO_PROXY": "lo"})
    assert path.read_text(encoding="utf-8") == (
        f"LLM_API_KEY=sk-1\n# spare:\n{comment}\n{name}={value}\nNO_PROXY=lo\n"
        f"# PRESET=medium\n")
    path.write_text("# MEDIUM_MODEL=anthropic/claude-sonnet-5-5\n# PRESET=low",
                    encoding="utf-8")
    assert config.rewrite_env_file(str(path), {}, {"MEDIUM_MODEL": "a/b", "PRESET": "high"})
    assert path.read_text(encoding="utf-8") == (
        "MEDIUM_MODEL=a/b\n# PRESET=low\nPRESET=high\n")


@pytest.mark.parametrize("before, set_dict, drop_list, after", [
    (b"A=1\r\n# mine\r\n\r\nPRESET=high\r\nMY='a\r\nb'\r\n", {"PRESET": "low"}, [],
     b"A=1\r\n# mine\r\n\r\nPRESET=low\r\nMY='a\r\nb'\r\n"),
    (b"A=1\r# mine\rPRESET=high\rMY='a\rb'\r", {"PRESET": "low"}, [],
     b"A=1\r# mine\rPRESET=low\rMY='a\rb'\r"),
    (b"A=1\r\nB=2\nPRESET=high\rC=3\r\n", {"PRESET": "low", "NEW_ONE": "x"}, [],
     b"A=1\r\nB=2\nPRESET=low\rC=3\r\nNEW_ONE=x\r\n"),
    (b"\xef\xbb\xbf# mine\r\nLLM_API_KEY=sk-1\r\n", {"LLM_API_KEY": "sk-2"}, [],
     b"\xef\xbb\xbf# mine\r\nLLM_API_KEY=sk-2\r\n"),
    (b"A=1\nPRESET=high", {"PRESET": "low"}, [], b"A=1\nPRESET=low"),
    (b"A=1\nMY=x", {"NEW_ONE": "y"}, [], b"A=1\nMY=x\nNEW_ONE=y\n"),
    (b"# note\r\nPRESET=high\r\nMY=1\r\nPRESET=x\r\n", {"PRESET": "low"}, [],
     b"# note\r\nPRESET=low\r\nMY=1\r\n"),
    (b"A=1\nMY_OWN=2", {}, ["MY_OWN"], b"A=1\n"),
    (b"A=1\r\n# HTTPS_PROXY=\r\nPRESET=high\r\n", {"HTTPS_PROXY": "http://p:3128"},
     ["PRESET"], b"A=1\r\nHTTPS_PROXY=http://p:3128\r\n# PRESET=medium\r\n"),
    (b"A=1\r\n# PRESET=low", {"PRESET": "high"}, [], b"A=1\r\n# PRESET=low\r\nPRESET=high\r\n")])
def test_the_lines_not_answered_are_written_back_byte_for_byte(tmp_path, before, set_dict,
                                                               drop_list, after):
    """Their line endings and the byte order mark stay. A line written in place of another
    takes its line ending, and a line added the one the file uses most."""
    path = tmp_path / ".env"
    path.write_bytes(before)
    current_dict = config.env_file_value_dict(str(path))
    answer_dict = {name: value for name, value in {**current_dict, **set_dict}.items()
                   if name not in drop_list}
    assert config.rewrite_env_file(str(path), current_dict, answer_dict)
    assert path.read_bytes() == after
    assert config.env_file_value_dict(str(path)) == answer_dict


@pytest.mark.parametrize("blank", [" \t", "\x0c", "\x0b", "\xa0", "　", " ",
                                   "\x85"])
def test_a_line_of_blanks_above_an_answered_setting_is_written_back_byte_for_byte(
        tmp_path, blank):
    """Of any blank python-dotenv reads as one: above a line answered, a line taken out
    and the template's line an answer takes the place of, which is read as that line
    with blanks before it on its own line too."""
    path = tmp_path / ".env"
    for before, set_dict, drop_list, after in (
            (f"A=1\n{blank}\nPRESET=high\n", {"PRESET": "low"}, [],
             f"A=1\n{blank}\nPRESET=low\n"),
            (f"A=1\r\n{blank}\r\nPRESET=high\r\n", {}, ["PRESET"],
             f"A=1\r\n{blank}\r\n# PRESET=medium\r\n"),
            (f"A=1\n{blank}\n# PRESET=medium\n", {"PRESET": "high"}, [],
             f"A=1\n{blank}\nPRESET=high\n"),
            (f"A=1\n{blank}# PRESET=medium\n", {"PRESET": "high"}, [],
             "A=1\nPRESET=high\n")):
        path.write_bytes(before.encode("utf-8"))
        current_dict = config.env_file_value_dict(str(path))
        answer_dict = {name: value for name, value in {**current_dict, **set_dict}.items()
                       if name not in drop_list}
        assert config.rewrite_env_file(str(path), current_dict, answer_dict)
        assert path.read_bytes() == after.encode("utf-8")
        assert config.env_file_value_dict(str(path)) == answer_dict


# == the settings that take one of a few values ================
def test_the_log_level_is_read_in_any_case_and_checked(monkeypatch):
    assert _reloaded(monkeypatch, AGENT_LOG_LEVEL=" debug")["AGENT_LOG_LEVEL"] == "DEBUG"
    for value in ("loud", "WARN", "NOTSET"):
        with pytest.raises(ValueError, match=f"^AGENT_LOG_LEVEL must be one of DEBUG / "
                                             f"INFO / WARNING / ERROR / CRITICAL: {value}$"):
            _reloaded(monkeypatch, AGENT_LOG_LEVEL=value)


@pytest.mark.parametrize("value", ["trace", "verbose", "debug"])
def test_a_log_level_set_for_another_tool_is_not_read(monkeypatch, value):
    monkeypatch.setenv("LOG_LEVEL", value)
    assert _reloaded(monkeypatch, AGENT_LOG_LEVEL=None)["AGENT_LOG_LEVEL"] == "INFO"


@pytest.mark.parametrize("name, value, read", [
    ("PRESET", " High", "high"),
    ("WEB_SEARCH_CONTEXT_SIZE", "MEDIUM ", "medium"),
    ("WEB_SEARCH_PROVIDER", "Model", "model")])
def test_a_setting_of_a_few_values_is_read_in_any_case_and_without_blanks(
        monkeypatch, name, value, read):
    assert _reloaded(monkeypatch, **{name: value})[name] == read


def test_the_test_target_is_read_in_any_case_and_checked(monkeypatch):
    monkeypatch.setenv("TEST_TARGET_STYLE", " None")
    config.set_primary_ext("py")
    assert config.TEST_TARGET_STYLE == "none"
    monkeypatch.setenv("TEST_TARGET_STYLE", "unit")
    with pytest.raises(ValueError, match="^TEST_TARGET_STYLE must be one of path / separator / "
                                         "none: unit$"):
        config.set_primary_ext("py")
    monkeypatch.delenv("TEST_TARGET_STYLE")
    config.set_primary_ext("py")


def test_the_old_name_of_the_test_target_style_is_not_read(monkeypatch):
    monkeypatch.delenv("TEST_TARGET_STYLE", raising=False)
    monkeypatch.setenv("TEST_TARGET", "unit")
    config.set_primary_ext("py")
    assert config.TEST_TARGET_STYLE == LANG_DICT["py"].test_target


@pytest.mark.parametrize("name, value", [
    ("PRESET", "bogus"), ("RLM_MAX_TURN", "1.5"), ("TEST_PATH", "/abs"),
    ("AGENT_LOG_LEVEL", "trace"), ("QUERY_EXPAND", "maybe"),
    ("DISPATCH_MAX_TASK", "0"), ("DISPATCH_MAX_UNCLEAR", "-1"),
    ("MAX_TOKEN_PER_RUN", "-1"), ("DOC_CREATE_EFFORT", "max")])
def test_every_value_config_refuses_is_a_setting_error_naming_it(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(SettingError, match=name):
        importlib.reload(config)
    monkeypatch.delenv(name)
    importlib.reload(config)


def test_the_design_document_effort_is_read_from_the_environment(monkeypatch):
    assert _reloaded(monkeypatch, DOC_CREATE_EFFORT=" Medium ")["DOC_CREATE_EFFORT"] \
        == "medium"
    assert _reloaded(monkeypatch, DOC_CREATE_EFFORT=None)["DOC_CREATE_EFFORT"] == "low"


def test_the_choices_are_kept_for_the_first_run_questions():
    assert config.choice_tuple_of("PRESET") == config.PRESET_LEVEL_TUPLE
    assert config.choice_tuple_of("MEDIUM_MODEL") == ()


def test_the_search_context_size_is_checked(monkeypatch):
    assert _reloaded(monkeypatch, WEB_SEARCH_CONTEXT_SIZE="high")["WEB_SEARCH_CONTEXT_SIZE"] == "high"
    monkeypatch.setenv("WEB_SEARCH_CONTEXT_SIZE", "huge")
    with pytest.raises(ValueError, match="WEB_SEARCH_CONTEXT_SIZE"):
        importlib.reload(config)
    monkeypatch.delenv("WEB_SEARCH_CONTEXT_SIZE")
    importlib.reload(config)


# == PRESET: one column for the settings that scale accuracy against cost ==
def test_preset_defaults_to_medium(monkeypatch):
    monkeypatch.delenv("PRESET", raising=False)
    importlib.reload(config)
    assert config.PRESET == "medium"


def test_a_preset_outside_the_3_values_raises_at_import(monkeypatch):
    monkeypatch.setenv("PRESET", "max")
    with pytest.raises(ValueError, match="PRESET"):
        importlib.reload(config)
    monkeypatch.delenv("PRESET")
    importlib.reload(config)


def test_every_column_names_the_same_settings():
    key_set_list = [set(column) for column in config._PRESET_TABLE_DICT.values()]
    assert all(key_set == key_set_list[0] for key_set in key_set_list)
    assert tuple(config._PRESET_TABLE_DICT) == config.PRESET_LEVEL_TUPLE


def test_every_column_asks_for_the_thinking_one_level_under_its_name():
    assert [column["THINK_LEVEL"] for column in config._PRESET_TABLE_DICT.values()] == \
        ["off", "low", "medium"]


def test_every_setting_in_the_table_is_read_from_the_environment():
    """A name only in the table would be a column value nothing reads."""
    read_set = set(config.env_name_tuple())
    for column in config._PRESET_TABLE_DICT.values():
        assert set(column) <= read_set


def test_the_column_reaches_the_settings_that_follow_them(monkeypatch):
    monkeypatch.setenv("PRESET", "high")
    for name in ("RLM_MAX_TURN", "PLAN_MAX_TURN", "TWO_STAGE", "PLAN_MAX_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(config)
    assert config.TWO_STAGE is True
    assert config.RLM_MAX_TURN == config.PLAN_MAX_TURN == 40
    assert config.PLAN_MAX_TOKEN == config.AGENT_MAX_TOKEN == 32768
    monkeypatch.delenv("PRESET")
    importlib.reload(config)


def test_a_setting_set_by_name_wins_over_the_column(monkeypatch):
    monkeypatch.setenv("PRESET", "high")
    monkeypatch.setenv("RLM_MAX_TURN", "7")
    importlib.reload(config)
    assert config.RLM_MAX_TURN == 7
    assert config.PLAN_MAX_TURN == 7
    monkeypatch.delenv("PRESET")
    monkeypatch.delenv("RLM_MAX_TURN")
    importlib.reload(config)


def test_preset_env_refuses_a_name_outside_the_table():
    with pytest.raises(KeyError):
        config.preset_env("DEF_K")


# == The settings that follow another when unset ============
def test_the_medium_model_reaches_the_stages_that_follow_it(monkeypatch):
    monkeypatch.setenv("MEDIUM_MODEL", "provider/worker")
    for name in ("EDIT_MODEL", "ANSWER_MODEL", "EXPLORE_MODEL", "WEB_SEARCH_MODEL",
                 "QUERY_EXPAND_MODEL", "REPORT_MODEL", "DOC_CREATE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(config)
    assert (config.EDIT_MODEL == config.ANSWER_MODEL == config.EXPLORE_MODEL
            == config.WEB_SEARCH_MODEL == config.QUERY_EXPAND_MODEL
            == config.REPORT_MODEL == config.DOC_CREATE_MODEL == "provider/worker")
    monkeypatch.delenv("MEDIUM_MODEL")
    importlib.reload(config)


def test_the_high_model_reaches_the_stages_that_follow_it(monkeypatch):
    monkeypatch.setenv("HIGH_MODEL", "provider/strong")
    for name in ("PLAN_MODEL", "DISPATCH_MODEL", "INTEGRATE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(config)
    assert (config.PLAN_MODEL == config.DISPATCH_MODEL == config.INTEGRATE_MODEL
            == "provider/strong")
    monkeypatch.delenv("HIGH_MODEL")
    importlib.reload(config)


def test_the_two_parents_have_a_model_each(monkeypatch):
    for name in ("MEDIUM_MODEL", "HIGH_MODEL"):
        monkeypatch.delenv(name, raising=False)
    importlib.reload(config)
    assert config.MEDIUM_MODEL != config.HIGH_MODEL
    assert config.EDIT_MODEL == config.MEDIUM_MODEL
    assert config.PLAN_MODEL == config.HIGH_MODEL


# == The settings read as JSON objects =======================
def test_a_json_setting_that_is_unset_or_empty_is_an_empty_object(monkeypatch):
    monkeypatch.delenv("SMOKE_ENV", raising=False)
    assert config.json_env("SMOKE_ENV") == {}
    monkeypatch.setenv("SMOKE_ENV", "")
    assert config.json_env("SMOKE_ENV") == {}


def test_a_json_setting_is_read_as_the_object_it_holds(monkeypatch):
    monkeypatch.setenv("SMOKE_ENV", '{"SOME_FEATURE": "off"}')
    assert config.json_env("SMOKE_ENV") == {"SOME_FEATURE": "off"}


def test_a_json_setting_that_does_not_parse_names_the_setting(monkeypatch):
    monkeypatch.setenv("SMOKE_ENV", "{not json")
    with pytest.raises(ValueError, match="SMOKE_ENV must be a JSON object"):
        config.json_env("SMOKE_ENV")


def test_a_json_setting_that_is_not_an_object_names_what_it_is(monkeypatch):
    monkeypatch.setenv("SMOKE_ENV", "[1, 2]")
    with pytest.raises(ValueError, match="not list"):
        config.json_env("SMOKE_ENV")


def test_the_refill_and_the_web_search_are_turned_off_each_on_its_own(monkeypatch):
    from codetwine_loom.tools import web
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "model")
    monkeypatch.setenv("WARM_MAX_REFILL", "0")
    monkeypatch.delenv("WEB_SEARCH_MAX_CALL", raising=False)
    monkeypatch.delenv("WEB_FETCH_MAX_CALL", raising=False)
    importlib.reload(config)
    try:
        assert config.WARM_MAX_REFILL == 0 and config.WEB_FETCH_MAX_CALL > 0
        assert config.WEB_SEARCH_MAX_CALL > 0 and web.available()
        monkeypatch.delenv("WARM_MAX_REFILL")
        monkeypatch.setenv("WEB_SEARCH_MAX_CALL", "0")
        importlib.reload(config)
        assert config.WARM_MAX_REFILL > 0 and config.WEB_FETCH_MAX_CALL > 0
        assert config.WEB_SEARCH_MAX_CALL == 0 and not web.available()
    finally:
        monkeypatch.delenv("WEB_SEARCH_PROVIDER")
        monkeypatch.delenv("WEB_SEARCH_MAX_CALL", raising=False)
        monkeypatch.delenv("WARM_MAX_REFILL", raising=False)
        importlib.reload(config)


def test_the_static_check_is_the_language_default_unless_it_is_set(monkeypatch):
    monkeypatch.delenv("CHECK_COMMAND", raising=False)
    try:
        config.set_primary_ext("py")
        assert config.CHECK_COMMAND[1:5] == ["-m", "ruff", "check", "--select"]
        config.set_primary_ext("java")
        assert config.CHECK_COMMAND == []
        monkeypatch.setenv("CHECK_COMMAND", "eslint --format unix .")
        config.set_primary_ext("java")
        assert config.CHECK_COMMAND == ["eslint", "--format", "unix", "."]
    finally:
        monkeypatch.delenv("CHECK_COMMAND", raising=False)
        config.set_primary_ext("py")


# == the page-reading presets =================================
_FETCH_PRESET_NAMES = ("WEB_FETCH_URL", "WEB_FETCH_METHOD", "WEB_FETCH_URL_PARAM",
                       "WEB_FETCH_PARAM",
                       "WEB_FETCH_HEADER", "WEB_FETCH_RESULT_PATH",
                       "WEB_FETCH_CONTENT_FIELD")


def _with_fetch_preset(monkeypatch, provider: str, **by_hand) -> dict:
    """Reload config with a page-reading preset and a key, and return its settings.

    Args:
        monkeypatch: pytest's fixture.
        provider: The WEB_FETCH_PROVIDER to set.
        **by_hand: Endpoint settings to set as well, which win over the preset.

    Returns:
        {setting name: value} for every page-reading endpoint setting.
    """
    unset = {name: None for name in _FETCH_PRESET_NAMES if name not in by_hand}
    return _reloaded(monkeypatch, WEB_FETCH_PROVIDER=provider, WEB_FETCH_API_KEY="r-key",
                     WEB_SEARCH_PROVIDER=None, WEB_SEARCH_API_KEY=None, **by_hand, **unset)


@pytest.mark.parametrize("provider", sorted(config.WEB_FETCH_PRESET_DICT))
def test_a_page_reading_preset_describes_the_endpoint_and_puts_the_key_in(monkeypatch,
                                                                           provider):
    read = _with_fetch_preset(monkeypatch, provider)
    preset = config.WEB_FETCH_PRESET_DICT[provider]
    assert read["WEB_FETCH_URL"] == preset["url"]
    assert read["WEB_FETCH_METHOD"] == preset["method"]
    assert read["WEB_FETCH_URL_PARAM"] == preset["url_param"]
    assert read["WEB_FETCH_RESULT_PATH"] == preset["result_path"]
    assert read["WEB_FETCH_CONTENT_FIELD"] == preset["content_field"]
    assert read["WEB_FETCH_PARAM"] == preset.get("params", {})
    assert any("r-key" in value for value in read["WEB_FETCH_HEADER"].values())
    assert not any("{key}" in value for value in read["WEB_FETCH_HEADER"].values())


def test_no_page_reading_preset_leaves_the_endpoint_unset(monkeypatch):
    read = _reloaded(monkeypatch, WEB_FETCH_PROVIDER=None, WEB_FETCH_API_KEY=None,
                     **{name: None for name in _FETCH_PRESET_NAMES})
    assert read["WEB_FETCH_URL"] == "" and read["WEB_FETCH_HEADER"] == {}
    assert read["WEB_FETCH_METHOD"] == "POST"


def test_a_page_reading_preset_that_is_not_there_says_which_are(monkeypatch):
    with pytest.raises(ValueError, match="^WEB_FETCH_PROVIDER must be one of .*tavily"):
        _with_fetch_preset(monkeypatch, "nowhere")


def test_the_model_s_own_search_is_no_way_to_read_a_page(monkeypatch):
    with pytest.raises(ValueError, match="^WEB_FETCH_PROVIDER must be one of"):
        _with_fetch_preset(monkeypatch, config.WEB_SEARCH_BY_THE_MODEL)


def test_a_page_reading_preset_without_a_key_says_which_setting_to_set(monkeypatch):
    provider = sorted(config.WEB_FETCH_PRESET_DICT)[0]
    with pytest.raises(ValueError, match="WEB_FETCH_API_KEY"):
        _reloaded(monkeypatch, WEB_FETCH_PROVIDER=provider, WEB_FETCH_API_KEY=None,
                  WEB_SEARCH_PROVIDER=None, WEB_SEARCH_API_KEY=None)


def test_the_key_follows_the_search_s_when_both_name_the_same_api(monkeypatch):
    both = sorted(set(config.WEB_SEARCH_PRESET_DICT) & set(config.WEB_FETCH_PRESET_DICT))[0]
    read = _reloaded(monkeypatch, WEB_SEARCH_PROVIDER=both, WEB_SEARCH_API_KEY="k-1",
                     WEB_FETCH_PROVIDER=both, WEB_FETCH_API_KEY=None, WEB_FETCH_HEADER=None,
                     WEB_SEARCH_HEADER=None)
    assert read["WEB_FETCH_API_KEY"] == "k-1"
    assert any("k-1" in value for value in read["WEB_FETCH_HEADER"].values())


def test_the_key_does_not_follow_the_search_s_for_another_api(monkeypatch):
    search_name, fetch_name = sorted(config.WEB_SEARCH_PRESET_DICT)[:2]
    assert fetch_name in config.WEB_FETCH_PRESET_DICT
    with pytest.raises(ValueError, match="WEB_FETCH_API_KEY"):
        _reloaded(monkeypatch, WEB_SEARCH_PROVIDER=search_name, WEB_SEARCH_API_KEY="k-1",
                  WEB_FETCH_PROVIDER=fetch_name, WEB_FETCH_API_KEY=None)


def test_every_preset_of_both_kinds_names_its_whole_endpoint():
    for name, preset in config.WEB_SEARCH_PRESET_DICT.items():
        assert {"url", "method", "query_param", "headers", "results_path",
                "result_keys"} <= set(preset), name
        assert preset["url"].startswith("https://") and preset["method"] in ("GET", "POST"), name
        assert any("{key}" in value for value in preset["headers"].values()), name
    for name, preset in config.WEB_FETCH_PRESET_DICT.items():
        assert {"url", "method", "url_param", "headers", "result_path",
                "content_field"} <= set(preset), name
        assert preset["url"].startswith("https://") and preset["method"] in ("GET", "POST"), name
        assert any("{key}" in value for value in preset["headers"].values()), name


def test_the_page_reading_settings_stay_with_the_process_that_reads():
    for name in ("WEB_FETCH_PROVIDER", "WEB_FETCH_API_KEY", "WEB_FETCH_TIMEOUT_SEC",
                 "WEB_FETCH_MIN_INTERVAL_SEC"):
        assert name in config.SECRET_NAME_TUPLE, name


def test_every_preset_column_names_the_page_read_budget():
    for level in config.PRESET_LEVEL_TUPLE:
        assert config._PRESET_TABLE_DICT[level]["WEB_FETCH_MAX_CALL"] > 0




# == the static checks of the languages besides the primary one ================

def test_the_other_languages_of_the_index_bring_their_default_checks(monkeypatch):
    monkeypatch.delenv("CHECK_COMMAND", raising=False)
    monkeypatch.setattr(config, "TOOL_OFFLINE", False)
    try:
        config.set_primary_ext("py")
        config.set_other_ext(["py", "ts", "tsx", "js", "java", "sql", "md", "", "c"])
        # One check for each language that has one, the primary language's left out
        # The C check is told to leave a makefile alone, which it builds for C alone
        assert config.OTHER_CHECK_LIST == [
            ("c", ["ctloom-check-c", "--no-make"]), ("js", ["ctloom-check-js"]),
            ("ts", ["ctloom-check-ts"])]
        monkeypatch.setattr(config, "TOOL_OFFLINE", True)
        config.set_other_ext(["c"])
        assert config.OTHER_CHECK_LIST == [
            ("c", ["ctloom-check-c", "--offline", "--no-make"])]
        # A new primary language starts without them
        config.set_primary_ext("js")
        assert config.OTHER_CHECK_LIST == []
        config.set_other_ext(["py", "js", "mjs"])
        assert [ext for ext, _ in config.OTHER_CHECK_LIST] == ["py"]
    finally:
        config.set_primary_ext("py")


def test_a_check_set_by_hand_is_the_one_check_run(monkeypatch):
    monkeypatch.setenv("CHECK_COMMAND", "make lint")
    try:
        config.set_primary_ext("py")
        config.set_other_ext(["ts", "c"])
        assert config.CHECK_COMMAND == ["make", "lint"]
        assert config.OTHER_CHECK_LIST == []
    finally:
        monkeypatch.delenv("CHECK_COMMAND")
        config.set_primary_ext("py")



@pytest.mark.parametrize("name", ["EDIT_DRY_RUN", "EDIT_RESUME"])
def test_what_a_flag_of_edit_says_is_not_read_from_the_environment(monkeypatch, name):
    monkeypatch.setenv(name, "true")
    importlib.reload(config)
    assert name not in config.env_name_tuple()
    assert config.EDIT_DRY_RUN is False and not hasattr(config, "EDIT_RESUME")
    monkeypatch.delenv(name)
    importlib.reload(config)


def test_the_other_languages_of_the_index_bring_their_default_test_commands(monkeypatch):
    monkeypatch.delenv("TEST_COMMAND", raising=False)
    monkeypatch.setattr(config, "TOOL_OFFLINE", False)
    try:
        config.set_primary_ext("py")
        config.set_other_ext(["py", "ts", "js", "java", "sql", "cbl", "c", "cpp"])
        # One command for each language that has one, the primary language's left out
        assert config.OTHER_TEST_LIST == [
            ("c", ["ctloom-test-c", "--no-make"]), ("java", ["mvn", "-B", "-fae", "test"]),
            ("js", ["npm", "test", "--silent"])]
        monkeypatch.setenv("TEST_COMMAND", "make test")
        config.set_primary_ext("py")
        config.set_other_ext(["js"])
        assert config.OTHER_TEST_LIST == []
        assert [ext for ext, _ in config.OTHER_CHECK_LIST] == ["js"]
    finally:
        monkeypatch.delenv("TEST_COMMAND", raising=False)
        config.set_primary_ext("py")


# == the lowest value of every numeric setting =================
def _number_name_list() -> list[str]:
    """Return the settings env() reads as a whole number or a fraction, sorted."""
    return [name for name in config.env_name_tuple()
            if isinstance(config.default_of(name), (int, float))
            and not isinstance(config.default_of(name), bool)]


def test_every_numeric_setting_names_its_lowest_value():
    name_list = _number_name_list()
    assert len(name_list) > 100
    assert [name for name in name_list if config.low_of(name) is None] == []


@pytest.mark.parametrize("name", _number_name_list())
def test_a_number_under_its_lowest_value_stops_the_import_and_the_lowest_is_read(
        monkeypatch, name):
    low = config.low_of(name)
    with pytest.raises(SettingError, match=f"^{name} must be {low} or more: {low - 1}$"):
        _reloaded(monkeypatch, **{name: str(low - 1)})
    assert _reloaded(monkeypatch, **{name: str(low)})[name] == low


@pytest.mark.parametrize("name", [name for name in _number_name_list()
                                  if isinstance(config.default_of(name), float)])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_a_fraction_that_is_not_finite_stops_the_import_naming_it(monkeypatch, name,
                                                                   value):
    with pytest.raises(SettingError, match=f"^{name} must be a finite number: {value}$"):
        _reloaded(monkeypatch, **{name: value})


def test_a_setting_without_a_lowest_value_has_none():
    assert config.low_of("PRESET") is None and config.low_of("QUERY_EXPAND") is None


# == the settings read as shell-quoted lists ==================
@pytest.mark.parametrize("name", ["BUILD_EXCLUDE", "TEST_COMMAND", "RUN_COMMAND",
                                  "CHECK_COMMAND", "FORMAT_COMMAND", "SMOKE_ARG"])
def test_a_quotation_left_open_stops_the_import_naming_the_setting(monkeypatch, name):
    with pytest.raises(SettingError,
                       match=f"^{name} must be a shell-quoted list .*: a \"b$"):
        _reloaded(monkeypatch, **{name: 'a "b'})


def test_a_shell_quoted_list_is_read_item_by_item(monkeypatch):
    monkeypatch.setenv("BUILD_EXCLUDE", "vendor 'old code'")
    assert config.shell_list_env("BUILD_EXCLUDE") == ["vendor", "old code"]
    monkeypatch.delenv("BUILD_EXCLUDE")
    assert config.shell_list_env("BUILD_EXCLUDE") == []


@pytest.mark.parametrize("name", ["TEST_COMMAND", "CHECK_COMMAND"])
def test_the_commands_of_the_other_languages_refuse_a_quotation_left_open(monkeypatch,
                                                                          name):
    monkeypatch.setenv(name, "make 'lint")
    with pytest.raises(SettingError, match=name):
        config.set_other_ext(["js", "c"])
    monkeypatch.delenv(name)
    config.set_primary_ext("py")


@pytest.mark.parametrize("value", [
    "no-such-codec", "hex", "base64", "bz2", "zlib", "zip", "quopri", "uu", "rot13",
    "undefined", "cp932,hex_codec", "rot_13,utf-8"])
def test_an_unknown_source_encoding_is_a_setting_error_naming_it(monkeypatch, value):
    """A codec that neither decodes bytes nor encodes a string is no encoding of text."""
    with pytest.raises(SettingError) as error:
        _reloaded(monkeypatch, SOURCE_ENCODING=value)
    wrong_name = next(name for name in value.split(",") if name not in ("cp932", "utf-8"))
    assert str(error.value) == (
        f"SOURCE_ENCODING names an encoding Python does not know: {wrong_name}. Use "
        f"Python codec names such as cp1252 or latin-1, with commas between them")


def test_the_encodings_of_text_are_taken_as_a_source_encoding(monkeypatch):
    value = "cp932, euc_jp,cp1252,latin-1,UTF8,shift_jis,utf_16,utf-32,idna,punycode"
    assert _reloaded(monkeypatch, SOURCE_ENCODING=value)["SOURCE_ENCODING"] == [
        name.strip() for name in value.split(",")]


@pytest.mark.parametrize("name", ["SMOKE_FILE", "SMOKE_ENV"])
@pytest.mark.parametrize("value, held", [
    ('{"a.txt": 1}', "int"), ('{"a.txt": null}', "NoneType"),
    ('{"a.txt": {"b": "c"}}', "dict"), ('{"a.txt": ["b"]}', "list")])
def test_a_json_setting_with_a_value_that_is_not_a_string_names_it(monkeypatch, name,
                                                                   value, held):
    with pytest.raises(SettingError, match=f"^{name} must hold a string .*'a.txt' "
                                           f"holds {held}$"):
        _reloaded(monkeypatch, **{name: value})


# == the form of LLM_BROKER and SANDBOX_MEMORY ================
@pytest.mark.parametrize("value", ["foo", "unix://", "tcp://broker", "tcp://:9000",
                                   "http://broker:9000", "tcp://broker:99999",
                                   "tcp://broker:65536", "tcp://broker:0",
                                   "tcp://broker:\u00b2", "tcp://broker:\uff18\uff10",
                                   "tcp://broker:-1", "tcp://[::1]:8080",
                                   "tcp://::1:8080"])
def test_a_broker_address_of_neither_form_stops_the_import(monkeypatch, value):
    with pytest.raises(SettingError, match=f"^LLM_BROKER={re.escape(value)} "):
        _reloaded(monkeypatch, LLM_BROKER=value)


@pytest.mark.parametrize("port, is_read", [("9" * 4300, False), ("9" * 5000, False),
                                           ("9" * 6, False), ("0" * 5000, False),
                                           ("0" * 5000 + "9000", True)],
                         ids=["4300 nines", "5000 nines", "6 nines", "5000 zeros",
                              "5000 zeros and 9000"])
def test_a_port_of_any_number_of_digits_is_read_or_refused_by_the_setting_s_name(
        monkeypatch, port, is_read):
    value = f"tcp://broker:{port}"
    if is_read:
        assert config.broker_address_of(value) == (config.BROKER_TCP_SCHEME,
                                                   ("broker", 9000))
        return
    with pytest.raises(SettingError) as stop:
        _reloaded(monkeypatch, LLM_BROKER=value)
    assert str(stop.value).startswith(f"LLM_BROKER={value[:60]}")
    assert str(stop.value).endswith("names a port outside 1 to 65535")
    assert len(str(stop.value)) < 200


def test_a_value_is_written_out_on_one_line_a_terminal_shows_as_it_is():
    assert config._value_text("12abc 日本語") == "12abc 日本語"
    assert config._value_text("1\x1b]0;owned\x07\n2\r\t\x7f\x9b") \
        == "1\\x1b]0;owned\\x07\\x0a2\\x0d\\x09\\x7f\\x9b"
    assert config._value_text("a\u3000b\u202ec\U000e0001") \
        == "a\\u3000b\\u202ec\\U000e0001"
    assert config._value_text(os.fsdecode(b"\xff\xfe.x")) == "\\xff\\xfe.x"
    assert config._value_text("\n" * 61) == "\\x0a" * 60 + "... (61 characters)"


# One value no setting takes: an escape sequence, a line break and a byte that is not
# UTF-8, and one too long for a line
_BAD_VALUE_TUPLE = ("1\x1b[2J\r\n2" + os.fsdecode(b"\xff"), "'" + "x" * 5000)


@pytest.mark.parametrize("name, value", [
    (name, value)
    for name in ("DEF_K", "WEB_SEARCH_MIN_INTERVAL_SEC", "RUN_TEST", "PRESET",
                 "THINK_LEVEL", "TEST_TARGET_STYLE", "LLM_BROKER", "SANDBOX_MEMORY",
                 "SOURCE_ENCODING", "SMOKE_ENV")
    for value in _BAD_VALUE_TUPLE] + [
    # A list in the shell's quoting takes any character; what it refuses is the quote
    (name, _BAD_VALUE_TUPLE[1] + _BAD_VALUE_TUPLE[0])
    for name in ("TEST_COMMAND", "BUILD_EXCLUDE", "SMOKE_ARG")],
    ids=lambda value: value if value.isidentifier() else f"{len(value)} characters")
def test_a_setting_that_is_refused_is_named_on_one_line_with_no_control_character(
        monkeypatch, name, value):
    with pytest.raises(SettingError) as stop:
        _reloaded(monkeypatch, **{name: value})
        config.set_primary_ext("py")
    text = str(stop.value)
    assert text.startswith(name) and len(text) < 500
    assert text.isprintable(), text


def test_a_test_path_and_a_name_of_a_mapping_are_written_out_the_same_way(monkeypatch):
    monkeypatch.setenv("TEST_PATH", "../\x1b[2J" + "x" * 100)
    with pytest.raises(SettingError) as stop:
        config.set_primary_ext("py")
    assert str(stop.value).startswith("TEST_PATH must be ") \
        and str(stop.value).isprintable() and len(str(stop.value)) < 300
    monkeypatch.delenv("TEST_PATH")
    with pytest.raises(SettingError) as stop:
        _reloaded(monkeypatch, SMOKE_ENV=json.dumps({"A\x1b" + "x" * 100: 1}))
    assert str(stop.value).startswith("SMOKE_ENV must hold a string under every name: "
                                      "'A\\x1bxxx")
    assert str(stop.value).isprintable() and len(str(stop.value)) < 300


@pytest.mark.parametrize("value, address", [
    ("unix:///run/ctloom/llm.sock", (config.BROKER_UNIX_SCHEME, "/run/ctloom/llm.sock")),
    ("tcp://broker:9000", (config.BROKER_TCP_SCHEME, ("broker", 9000))),
    ("tcp://127.0.0.1:1", (config.BROKER_TCP_SCHEME, ("127.0.0.1", 1))),
    ("tcp://127.0.0.1:65535", (config.BROKER_TCP_SCHEME, ("127.0.0.1", 65535)))])
def test_a_broker_address_of_either_form_is_read(monkeypatch, value, address):
    assert _reloaded(monkeypatch, LLM_BROKER=value)["LLM_BROKER"] == value
    assert config.broker_address_of(value) == address


@pytest.mark.parametrize("value, path", [("unix://llm.sock", "llm.sock"),
                                         ("unix://./run/../llm.sock", "llm.sock"),
                                         ("unix://run/llm.sock", "run/llm.sock")])
def test_a_socket_path_that_is_not_absolute_is_read_from_the_start_directory(
        monkeypatch, tmp_path, value, path):
    monkeypatch.chdir(tmp_path)
    held = _reloaded(monkeypatch, LLM_BROKER=value)["LLM_BROKER"]
    assert held == f"unix://{tmp_path / path}"
    assert config.absolute_broker_url("tcp://broker:9000") == "tcp://broker:9000"


@pytest.mark.parametrize("value", ["lots", "0", "0g", "00m", "0.0", "0.0g", "-1g", "g",
                                   "1.2.3g", "2  g", "2x", "2bg", "2gg", ".5g", "1.g",
                                   "2g ", " 2g", "1e3",
                                   # Digits and letters that are not ASCII
                                   "\uff12g", "\u0662g", "2\u212a", "2g\u0130b",
                                   # Less than a byte, and more than the engine holds
                                   "0.0000001", "0.0001k", "9" * 400 + "g", "9" * 5000,
                                   "8192p", "9223372036854775808"],
                         ids=lambda value: value if len(value) < 30 else
                         f"{len(value)} characters")
def test_a_sandbox_memory_of_another_form_or_of_nothing_stops_the_import(monkeypatch,
                                                                         value):
    with pytest.raises(SettingError, match="^SANDBOX_MEMORY must be ") as stop:
        _reloaded(monkeypatch, SANDBOX_MEMORY=value)
    assert len(str(stop.value)) < 400


@pytest.mark.parametrize("value", ["2g", "512M", "4K", "1b", "2147483648", "010m", "1.5g",
                                   "2gb", "2GB", "2GiB", "2 g", "4t", "1p", "0.5G",
                                   "512mib", "100B", "8191p", "0.001k",
                                   "8589934592"])
def test_a_sandbox_memory_in_the_engine_s_form_is_read_as_written(monkeypatch, value):
    assert _reloaded(monkeypatch, SANDBOX_MEMORY=value)["SANDBOX_MEMORY"] == value


def test_a_whole_number_too_long_for_a_float_is_refused_as_one_over_the_most(monkeypatch):
    value = "1" + "0" * 320
    with pytest.raises(SettingError) as stop:
        _reloaded(monkeypatch, DEF_K=value)
    assert str(stop.value) == (f"DEF_K must be {config.NUMBER_MAX} or less: {value[:60]}... "
                               f"(321 characters)")
    with pytest.raises(SettingError, match="^WEB_SEARCH_MIN_INTERVAL_SEC must be a "
                                           "finite number"):
        _reloaded(monkeypatch, WEB_SEARCH_MIN_INTERVAL_SEC="1e999")
    with pytest.raises(SettingError, match="^WEB_SEARCH_MIN_INTERVAL_SEC must be "
                                           f"{config.SECOND_MAX} or less: 1e300$"):
        _reloaded(monkeypatch, WEB_SEARCH_MIN_INTERVAL_SEC="1e300")


# == a setting that is set while the one it works with is empty ==
_NO_WEB_DICT = {"WEB_SEARCH_PROVIDER": None, "WEB_SEARCH_API_KEY": None,
                "WEB_FETCH_PROVIDER": None, "WEB_FETCH_API_KEY": None,
                "WEB_SEARCH_MAX_CALL": None, "WEB_FETCH_MAX_CALL": None,
                "SMOKE_ARG": None, "SMOKE_FILE": None, "SMOKE_ENV": None}


def _no_partner_list(monkeypatch, **environment) -> list[str]:
    """Return config.no_partner_setting_list() as it is with these settings and no
    other of the settings it reads, and put config back."""
    setting_dict = {**_NO_WEB_DICT, **environment}
    for name, value in setting_dict.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    try:
        importlib.reload(config)
        return config.no_partner_setting_list()
    finally:
        for name in setting_dict:
            monkeypatch.delenv(name, raising=False)
        importlib.reload(config)


@pytest.mark.parametrize("environment, no_partner_list", [
    ({}, []),
    ({"WEB_FETCH_PROVIDER": "jina", "WEB_FETCH_API_KEY": "k"},
     ["WEB_FETCH_PROVIDER (WEB_SEARCH_PROVIDER is empty)"]),
    ({"WEB_SEARCH_API_KEY": "k"}, ["WEB_SEARCH_API_KEY (WEB_SEARCH_PROVIDER is empty)"]),
    ({"WEB_FETCH_API_KEY": "k"}, ["WEB_FETCH_API_KEY (WEB_FETCH_PROVIDER is empty)"]),
    ({"SMOKE_FILE": '{"a.txt": "x"}', "SMOKE_ENV": '{"A": "b"}'},
     ["SMOKE_FILE (SMOKE_ARG is empty)", "SMOKE_ENV (SMOKE_ARG is empty)"]),
    ({"SMOKE_FILE": '{"a.txt": "x"}', "SMOKE_ARG": "--in a.txt"}, []),
    ({"WEB_SEARCH_PROVIDER": "tavily", "WEB_SEARCH_API_KEY": "k",
      "WEB_FETCH_PROVIDER": "tavily"}, []),
    ({"WEB_SEARCH_PROVIDER": "model", "WEB_FETCH_PROVIDER": "jina",
      "WEB_FETCH_API_KEY": "k"}, []),
    ({"WEB_FETCH_PROVIDER": "jina", "WEB_FETCH_API_KEY": "k", "WEB_SEARCH_API_KEY": "k",
      "WEB_SEARCH_MAX_CALL": "0"}, []),
    ({"WEB_FETCH_PROVIDER": "jina", "WEB_FETCH_API_KEY": "k", "WEB_FETCH_MAX_CALL": "0"},
     []),
])
def test_a_setting_without_the_one_it_works_with_is_named(monkeypatch, environment,
                                                          no_partner_list):
    assert _no_partner_list(monkeypatch, **environment) == no_partner_list


def test_reading_the_settings_logs_nothing_about_one_without_a_partner(monkeypatch,
                                                                        caplog):
    with caplog.at_level("WARNING"):
        _no_partner_list(monkeypatch, WEB_FETCH_PROVIDER="jina", WEB_FETCH_API_KEY="k",
                   SMOKE_FILE='{"a.txt": "x"}')
    assert caplog.records == []


def test_child_env_leaves_out_what_places_git_elsewhere(monkeypatch):
    for name in config.GIT_PLACE_NAME_TUPLE:
        monkeypatch.setenv(name, "/another/place")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "kept")
    env_dict = config.child_env({"GIT_INDEX_FILE": "/the/callers"})
    assert [n for n in config.GIT_PLACE_NAME_TUPLE if n in env_dict] == ["GIT_INDEX_FILE"]
    assert env_dict["GIT_INDEX_FILE"] == "/the/callers"
    assert env_dict["GIT_AUTHOR_NAME"] == "kept"
    assert "GIT_DIR" not in config.project_env()


def test_child_env_leaves_out_what_changes_git_s_answer(monkeypatch):
    for name in config.GIT_ANSWER_NAME_TUPLE:
        monkeypatch.setenv(name, "1")
    env_dict = config.child_env({"GIT_LITERAL_PATHSPECS": "1"})
    assert [n for n in config.GIT_ANSWER_NAME_TUPLE if n in env_dict] \
        == ["GIT_LITERAL_PATHSPECS"]
    assert not set(config.GIT_ANSWER_NAME_TUPLE) & set(config.project_env())
    assert {"GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS",
            "GIT_ICASE_PATHSPECS", "GIT_DIFF_OPTS", "GIT_ATTR_SOURCE"} \
        <= set(config.GIT_ANSWER_NAME_TUPLE)


@pytest.mark.parametrize("name", ["GIT_LITERAL_PATHSPECS", "GIT_ICASE_PATHSPECS"])
def test_the_dot_env_file_is_asked_about_by_name_whatever_the_pathspec_variables_say(
        monkeypatch, tmp_path, name):
    _repo_with_dot_env(tmp_path / "repo", track=True)
    monkeypatch.setenv(name, "1")
    assert config.git_index_state(str(tmp_path / "repo" / ".env")) == config.INDEX_HOLD
    assert config.git_index_state(str(tmp_path / "repo" / "x.env")) == config.INDEX_FREE


def test_every_numeric_setting_refuses_a_value_above_what_it_takes(monkeypatch):
    """A value no timer, socket or count takes stops the command where it is read."""
    name_list = _number_name_list()
    assert len(name_list) > 100
    for name in name_list:
        high = config.high_of(name)
        assert high == (config.SECOND_MAX if name.endswith("_SEC") else config.NUMBER_MAX)
        for raw in (str(high + 1), "99999999999", "9" * 5000):
            monkeypatch.setenv(name, raw)
            with pytest.raises(SettingError) as stop:
                config.env(name, config.default_of(name), low=config.low_of(name))
            assert str(stop.value).startswith(f"{name} must be ")
            assert len(str(stop.value)) < 200
        if isinstance(config.default_of(name), int):
            assert f"must be {high} or less: {high + 1}" in str(
                pytest.raises(SettingError, _env_of, monkeypatch, name, str(high + 1)).value)
        monkeypatch.setenv(name, str(high))
        assert config.env(name, config.default_of(name), low=config.low_of(name)) == high
        monkeypatch.delenv(name)


def _env_of(monkeypatch, name: str, raw: str):
    """Read one setting with raw in the environment."""
    monkeypatch.setenv(name, raw)
    return config.env(name, config.default_of(name), low=config.low_of(name))


def test_every_default_and_every_preset_value_is_within_what_a_setting_takes():
    for name in _number_name_list():
        assert config.low_of(name) <= config.default_of(name) <= config.high_of(name), name
        assert getattr(config, name) <= config.high_of(name), name
    for column_dict in config._PRESET_TABLE_DICT.values():
        for name, value in column_dict.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert value <= config.high_of(name), name
    assert config.SECOND_MAX < 2_147_483 and config.NUMBER_MAX < 2 ** 31


@pytest.mark.parametrize("environment, text", [
    ({"LLM_MAX_RETRY": "1999999999"},
     "LLM_CALL_TIMEOUT_SEC x (2 x LLM_MAX_RETRY + 1) is 2399999999400 seconds, and must "
     "be 2000000 or less"),
    ({"LLM_CALL_TIMEOUT_SEC": "2000000", "LLM_MAX_RETRY": "1"},
     "LLM_CALL_TIMEOUT_SEC x (2 x LLM_MAX_RETRY + 1) is 6000000 seconds"),
    ({"TEST_TIMEOUT_SEC": "2000000", "RUN_TIMEOUT_SEC": "1", "TURN_TIMEOUT_SEC": None},
     "TURN_TIMEOUT_SEC (TEST_TIMEOUT_SEC + RUN_TIMEOUT_SEC when it is not set) is "
     "2000001 seconds, and must be 2000000 or less"),
    ({"TURN_TIMEOUT_SEC": "99999999999"},
     "TURN_TIMEOUT_SEC must be 2000000 or less: 99999999999"),
])
def test_a_wait_worked_out_from_settings_is_refused_when_it_is_over_the_most_seconds(
        monkeypatch, environment, text):
    with pytest.raises(SettingError) as stop:
        _reloaded(monkeypatch, **environment)
    assert text in str(stop.value)


def test_a_turn_limit_set_by_itself_is_taken_whatever_the_two_it_follows_add_up_to(
        monkeypatch):
    read = _reloaded(monkeypatch, TEST_TIMEOUT_SEC="2000000", RUN_TIMEOUT_SEC="2000000",
                     TURN_TIMEOUT_SEC="900")
    assert read["TURN_TIMEOUT_SEC"] == 900
